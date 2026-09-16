"""Control-tier contracts: the state map, the plan model, the rule, the replay.

Nothing here needs Cascade. The state map is exercised on canonical rigid-body
states, the plan model on a fabricated learner, the decision on fabricated
trial results, and the replay on a run directory that was never tracked. The
tests that drive the simulator itself carry the ``cascade`` marker at the
bottom of this module.
"""

import copy
import hashlib
import importlib
import json
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from glassbox.control.plan import (
    PlanModel,
    PlanValues,
    ReferenceTrajectory,
    SafetyEnvelope,
    TrackingTolerances,
)
from glassbox.core.data import (
    Trajectory,
    save_trajectory_npz,
    trajectory_content_digest,
)
from glassbox.core.geometry import (
    nearest_rotation,
    quaternion_from_euler,
    quaternion_to_rotation_matrices,
    rotation_to_quaternion,
)
from glassbox.core.metrics import state_rmse_metrics
from glassbox.experimental import harness
from glassbox.experimental.default_model import fit, steps_for
from glassbox.experimental.learned_plan import (
    OBSERVED_CHANNELS,
    LearnedPlanController,
    LearnedPlanModel,
    ObservedHistory,
    fitted_solver_policy,
    learned_plan_model,
    observed_from_state,
    states_from_observed,
)
from glassbox.experimental.sequence_collection import (
    SequenceCollection,
    SequenceSegment,
)

MANIFEST = Path(__file__).resolve().parents[1] / "docs/harness/control-v3.json"
DT_S = 0.05
MINIMUM = np.array([0.0, -0.35, -0.35])
MAXIMUM = np.array([1.0, 0.35, 0.35])
COMMAND_CHANNELS = (
    "throttle [normalized,throttle]",
    "roll [rad,roll]",
    "pitch [rad,pitch]",
)
LEVEL = np.array([1.0, -2.0, 100.0, 18.0, 0.3, -0.2, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def _recording(name, seed, rows=140):
    """One synthetic fifteen-channel recording with real rotation entries."""

    rng = np.random.default_rng(seed)
    commands = rng.uniform(MINIMUM, MAXIMUM, size=(rows - 1, 3))
    states = np.zeros((rows, 13))
    states[0] = LEVEL
    angles = np.zeros(3)
    for index, command in enumerate(commands):
        angles = 0.98 * angles + 0.05 * np.r_[command[1], command[2], 0.01 * index]
        states[index + 1, 0:3] = states[index, 0:3] + DT_S * states[index, 3:6]
        states[index + 1, 3:6] = (
            0.97 * states[index, 3:6]
            + np.r_[0.4 * command[0], 0.1 * angles[0], 0.1 * angles[1]]
        )
        states[index + 1, 6:10] = quaternion_from_euler(*angles)
        states[index + 1, 10:13] = 0.9 * states[index, 10:13] + 0.2 * angles
    return states, commands


def _segment(name, seed):
    states, commands = _recording(name, seed)
    return SequenceSegment(
        name, "whole", harness.observed_from_states(states), commands, DT_S
    )


@pytest.fixture(scope="module")
def learned():
    """One really fitted learner on the observed contract the tier requires."""

    return fit(
        SequenceCollection(
            tuple(_segment(name, seed) for name, seed in (("a", 1), ("b", 2))),
            configuration_id="control-tier-tests",
            state_channels=OBSERVED_CHANNELS,
            input_channels=COMMAND_CHANNELS,
        )
    )


@pytest.fixture(scope="module")
def fabricated(learned):
    """A stand-in carrying only what the plan model is allowed to read."""

    return SimpleNamespace(
        _model=learned._model,
        contract=learned.contract,
        recipe=learned.recipe,
        history_steps=learned.history_steps,
        horizon_steps=learned.horizon_steps,
    )


def _plan(model, **keywords):
    return learned_plan_model(
        model,
        TrackingTolerances.for_platform("fixedwing"),
        SafetyEnvelope(),
        command_minimum=MINIMUM,
        command_maximum=MAXIMUM,
        **keywords,
    )


def _controller(model):
    return LearnedPlanController(
        model,
        command_minimum=MINIMUM,
        command_maximum=MAXIMUM,
        policy=replace(
            fitted_solver_policy(model),
            maximum_iterations=4,
            allow_unresolved_parameters=True,
        ),
    )


# --- the state map ----------------------------------------------------------


def test_predicted_rotation_entries_project_onto_a_proper_rotation():
    rng = np.random.default_rng(0)
    for _ in range(50):
        quaternion = quaternion_from_euler(*rng.uniform(-1.2, 1.2, 3))
        entries = quaternion_to_rotation_matrices(quaternion) + rng.normal(
            scale=0.05, size=(3, 3)
        )
        projected = np.asarray(nearest_rotation(jnp.asarray(entries)))
        np.testing.assert_allclose(projected @ projected.T, np.eye(3), atol=1e-6)
        assert float(np.linalg.det(projected)) == pytest.approx(1.0, abs=1e-6)
        recovered = np.asarray(rotation_to_quaternion(jnp.asarray(projected)))
        assert float(np.linalg.norm(recovered)) == pytest.approx(1.0, abs=1e-6)
        np.testing.assert_allclose(
            quaternion_to_rotation_matrices(recovered), projected, atol=1e-6
        )


def test_an_exact_rotation_is_its_own_projection():
    rotation = quaternion_to_rotation_matrices(quaternion_from_euler(0.3, -0.2, 2.9))
    np.testing.assert_allclose(
        np.asarray(nearest_rotation(jnp.asarray(rotation))), rotation, atol=1e-6
    )


def test_position_integrates_trapezoidally_from_a_constant_velocity():
    velocity = np.array([18.0, -1.0, 0.5])
    state = LEVEL.copy()
    state[3:6] = velocity
    observed = np.tile(observed_from_state(state), (6, 1))
    states = np.asarray(
        states_from_observed(jnp.asarray(state), jnp.asarray(observed), DT_S)
    )
    assert states.shape == (7, 13)
    np.testing.assert_allclose(states[0], state, atol=1e-6)
    expected = state[0:3] + np.arange(1, 7)[:, None] * DT_S * velocity
    np.testing.assert_allclose(states[1:, 0:3], expected, atol=1e-6)
    np.testing.assert_allclose(states[1:, 3:6], np.tile(velocity, (6, 1)), atol=1e-6)


def test_the_channel_map_round_trips_one_canonical_state():
    observed = np.asarray(observed_from_state(LEVEL))
    assert observed.shape == (15,)
    np.testing.assert_allclose(observed[0:3], LEVEL[3:6], atol=1e-6)
    np.testing.assert_allclose(observed[3:6], LEVEL[10:13], atol=1e-6)
    states = np.asarray(
        states_from_observed(jnp.asarray(LEVEL), jnp.asarray(observed[None]), DT_S)
    )
    np.testing.assert_allclose(states[1, 3:6], LEVEL[3:6], atol=1e-6)
    np.testing.assert_allclose(states[1, 10:13], LEVEL[10:13], atol=1e-6)
    np.testing.assert_allclose(
        quaternion_to_rotation_matrices(states[1, 6:10]),
        quaternion_to_rotation_matrices(LEVEL[6:10]),
        atol=1e-6,
    )


def test_the_channel_map_agrees_with_the_platform_adapter():
    states, _ = _recording("a", 3)
    rows = harness.observed_from_states(states)
    for index in (0, 7, len(states) - 1):
        np.testing.assert_allclose(
            np.asarray(observed_from_state(states[index])), rows[index], atol=1e-6
        )


# --- the carried history ----------------------------------------------------


def test_history_is_carried_interval_by_interval_and_reset(learned):
    controller = _controller(learned)
    states, commands = _recording("carry", 5)
    assert controller.required_observations == controller.plan.delay_steps + 1
    assert not controller.ready
    with pytest.raises(ValueError, match="nothing is padded"):
        controller.history()
    for index in range(controller.required_observations - 1):
        controller.observe(states[index])
        assert not controller.ready
        controller.command_applied(commands[index])
    controller.observe(states[controller.required_observations - 1])
    assert controller.ready

    delay = controller.plan.delay_steps
    history = controller.history()
    assert history.states.shape == (delay, 15)
    assert history.commands.shape == (delay, 3)
    assert history.memory.shape == (controller.plan.memory_size,)

    for index in range(controller.required_observations - 1, 40):
        controller.command_applied(commands[index])
        controller.observe(states[index + 1])
    history = controller.history()
    np.testing.assert_allclose(
        history.states,
        harness.observed_from_states(states[40 - delay : 40]),
        atol=1e-6,
    )
    np.testing.assert_allclose(history.commands, commands[40 - delay : 40], atol=1e-12)
    assert float(np.max(np.abs(history.memory))) > 0.0

    # Asked for after the command it computed, the history would shift the
    # commands one interval past the observations they belong to.
    controller.command_applied(commands[40])
    with pytest.raises(ValueError, match="newest observed state"):
        controller.history()

    controller.reset()
    assert not controller.ready
    assert controller.observed_states == 0
    with pytest.raises(ValueError, match="nothing is padded"):
        controller.history()


def test_observations_and_commands_must_alternate(learned):
    controller = _controller(learned)
    states, commands = _recording("alternate", 6)
    with pytest.raises(ValueError, match="follows the state"):
        controller.command_applied(commands[0])
    controller.observe(states[0])
    with pytest.raises(ValueError, match="alternate"):
        controller.observe(states[1])
    with pytest.raises(ValueError, match="finite command vector"):
        controller.command_applied(np.array([0.0, 0.0]))


def test_the_carried_history_reproduces_the_recipes_own_forecast(learned):
    """A full context carried forward is the context ``predict`` would consume."""

    controller = _controller(learned)
    states, commands = _recording("equivalent", 7)
    context = controller.context_steps
    origin = context + 4
    for index in range(origin):
        controller.observe(states[index])
        controller.command_applied(commands[index])
    controller.observe(states[origin])

    horizon = controller.prediction_steps
    future = commands[origin : origin + horizon]
    plan = controller.plan.with_history(controller.history())
    prediction = plan.rollout_commands(
        jnp.asarray(future),
        jnp.asarray(states[origin]),
        jnp.zeros(0),
        jnp.zeros((horizon, 0)),
        plan.values,
    )
    expected = np.asarray(
        learned.predict(
            harness.observed_from_states(states[origin - context : origin + 1]),
            commands[origin - context : origin],
            future,
        )
    )
    predicted = np.asarray(prediction.mean_states)
    np.testing.assert_allclose(
        predicted[1:, 3:6], expected[:, 0:3], rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        predicted[1:, 10:13], expected[:, 3:6], rtol=1e-5, atol=1e-5
    )


# --- the plan model ---------------------------------------------------------


def test_the_plan_model_satisfies_the_seam_with_a_fabricated_learner(fabricated):
    plan = _plan(fabricated)
    assert isinstance(plan, LearnedPlanModel)
    assert isinstance(plan, PlanModel)
    horizon = plan.horizon_steps
    assert horizon == steps_for(DT_S)["horizon"]
    assert plan.block_count <= horizon
    assert plan.sample_period_s == DT_S
    assert plan.command_size == 3
    assert plan.exogenous_size == 0
    assert plan.latent_size == 0
    assert plan.uncertainty_available is False
    assert plan.uncertainty_complete is False
    np.testing.assert_allclose(np.asarray(plan.command_minimum), MINIMUM)
    np.testing.assert_allclose(np.asarray(plan.command_maximum), MAXIMUM)

    delay = plan.delay_steps
    history = ObservedHistory(
        states=jnp.asarray(np.tile(observed_from_state(LEVEL), (delay, 1))),
        commands=jnp.zeros((delay, 3)),
        memory=jnp.zeros(plan.memory_size),
    )
    ready = plan.with_history(history)
    assert ready.compile_signature == plan.compile_signature

    latent = ready.initial_latent(jnp.zeros(3), ready.values)
    assert latent.shape == (0,)
    blocks = jnp.zeros((plan.block_count, 3))
    exogenous = jnp.zeros((horizon, 0))
    prediction = ready.rollout(
        blocks, jnp.asarray(LEVEL), latent, exogenous, ready.values
    )
    assert prediction.mean_states.shape == (horizon + 1, 13)
    assert prediction.tangent_covariance.shape == (horizon, 12, 12)
    assert prediction.commands.shape == (horizon, 3)
    assert prediction.latent_states.shape == (horizon + 1, 0)
    assert float(jnp.max(jnp.abs(prediction.tangent_covariance))) == 0.0
    assert bool(np.isfinite(np.asarray(prediction.mean_states)).all())
    np.testing.assert_allclose(np.asarray(prediction.mean_states[0]), LEVEL, atol=1e-6)

    reference = ReferenceTrajectory(np.tile(LEVEL, (horizon + 1, 1)))
    cost = ready.stage_cost(prediction, reference.states, jnp.zeros(3), ready.policy)
    assert cost.shape == ()
    assert math.isfinite(float(cost))

    measurements = ready.measure(prediction)
    assert float(measurements.maximum_validity_utilization) == 0.0
    assert float(measurements.maximum_normalized_uncertainty) == 0.0
    assert math.isfinite(float(measurements.maximum_normalized_safety_violation))


def test_the_plan_model_is_differentiable_in_the_command_plan(fabricated):
    plan = _plan(fabricated)
    delay = plan.delay_steps
    history = ObservedHistory(
        states=jnp.asarray(np.tile(observed_from_state(LEVEL), (delay, 1))),
        commands=jnp.zeros((delay, 3)),
        memory=jnp.zeros(plan.memory_size),
    )
    ready = plan.with_history(history)
    reference = jnp.asarray(np.tile(LEVEL, (plan.horizon_steps + 1, 1)))

    def objective(blocks):
        prediction = ready.rollout(
            blocks,
            jnp.asarray(LEVEL),
            jnp.zeros(0),
            jnp.zeros((plan.horizon_steps, 0)),
            ready.values,
        )
        return ready.stage_cost(prediction, reference, jnp.zeros(3), ready.policy)

    gradient = jax.grad(objective)(jnp.zeros((plan.block_count, 3)))
    assert gradient.shape == (plan.block_count, 3)
    assert bool(np.isfinite(np.asarray(gradient)).all())


def test_command_bounds_are_the_callers_facts_not_the_learners(fabricated):
    plan = _plan(
        fabricated,
    )
    other = learned_plan_model(
        fabricated,
        TrackingTolerances.for_platform("fixedwing"),
        SafetyEnvelope(),
        command_minimum=[-1.0, -1.0, -1.0],
        command_maximum=[2.0, 1.0, 1.0],
    )
    np.testing.assert_allclose(np.asarray(other.command_minimum), [-1.0, -1.0, -1.0])
    assert other.compile_signature != plan.compile_signature
    with pytest.raises(ValueError, match="strictly increasing"):
        learned_plan_model(
            fabricated,
            TrackingTolerances.for_platform("fixedwing"),
            SafetyEnvelope(),
            command_minimum=[1.0, 0.0, 0.0],
            command_maximum=[1.0, 1.0, 1.0],
        )
    with pytest.raises(ValueError, match="input channels differ"):
        learned_plan_model(
            fabricated,
            TrackingTolerances.for_platform("fixedwing"),
            SafetyEnvelope(),
            command_minimum=[0.0, 0.0],
            command_maximum=[1.0, 1.0],
        )


def test_a_horizon_beyond_the_fitted_one_is_refused(fabricated):
    fitted = fitted_solver_policy(fabricated)
    assert fitted.horizon_steps == steps_for(DT_S)["horizon"]
    with pytest.raises(ValueError, match="rejects horizons beyond its fitted range"):
        _plan(fabricated, policy=replace(fitted, horizon_steps=12, block_count=6))


def test_another_channel_contract_is_refused(fabricated):
    other = SimpleNamespace(
        _model=fabricated._model,
        contract={**fabricated.contract, "state_channels": ["x [unitless]"] * 15},
        recipe=fabricated.recipe,
        history_steps=fabricated.history_steps,
        horizon_steps=fabricated.horizon_steps,
    )
    with pytest.raises(ValueError, match="fifteen rigid-body observation"):
        _plan(other)


def test_a_horizon_without_history_is_refused_rather_than_padded(fabricated):
    plan = _plan(fabricated)
    assert plan.values.observed_history is None
    with pytest.raises(ValueError, match="observed history"):
        plan.rollout(
            jnp.zeros((plan.block_count, 3)),
            jnp.asarray(LEVEL),
            jnp.zeros(0),
            jnp.zeros((plan.horizon_steps, 0)),
            plan.values,
        )


def test_the_seam_refuses_to_plan_without_the_explicit_no_evidence_override(learned):
    """The learner claims no covariance, so a solve needs the caller to say so."""

    from glassbox.control.solver import BoundedShootingSolver

    controller = _controller(learned)
    states, commands = _recording("override", 8)
    for index in range(controller.required_observations - 1):
        controller.observe(states[index])
        controller.command_applied(commands[index])
    controller.observe(states[controller.required_observations - 1])
    origin = controller.required_observations - 1

    strict = replace(controller.policy, allow_unresolved_parameters=False)
    plan = controller.plan.with_history(controller.history())
    solver = BoundedShootingSolver(plan, strict)
    reference = ReferenceTrajectory(
        np.tile(states[origin], (plan.horizon_steps + 1, 1))
    )
    refused = solver.solve(states[origin], reference, commands[origin])
    assert str(refused.status) == "unresolved_model"
    assert refused.used_fallback is True

    allowed = controller.solve(states[origin], reference, commands[origin])
    assert allowed.used_fallback is False
    assert allowed.diagnostics.parameter_uncertainty_complete is False
    assert allowed.diagnostics.unresolved_parameters_allowed is True
    assert np.asarray(allowed.predicted_states).shape == (plan.horizon_steps + 1, 13)


def test_plan_values_carry_history_without_changing_how_a_belief_is_presented():
    """The seam extension is optional and defaulted; a belief supplies three values."""

    values = PlanValues(
        parameters=object(),
        covariance_factor=None,
        forecast_error_covariance=jnp.zeros((2, 12, 12)),
    )
    assert values.observed_history is None
    assert len(values) == 4


# --- the frozen manifest ----------------------------------------------------


def test_the_control_manifest_digest_is_the_gate(tmp_path):
    manifest = harness.frozen_control_manifest(MANIFEST)
    assert manifest["id"] == "control-v3"
    assert manifest["decision"]["enforced"] is True
    assert "this manifest" in manifest["decision"]["gates_from"]
    assert tuple(manifest["telemetry"]["observed_channels"]) == OBSERVED_CHANNELS
    budget = manifest["information_budget"]
    assert (
        budget["context_steps"],
        budget["delay_steps"],
        budget["horizon_steps"],
    ) == (
        steps_for(DT_S)["history"],
        steps_for(DT_S)["delay"],
        steps_for(DT_S)["horizon"],
    )
    loosened = copy.deepcopy(manifest)
    loosened["decision"]["enforced"] = False
    path = tmp_path / "control-v3.json"
    path.write_text(json.dumps(loosened, indent=2) + "\n")
    with pytest.raises(ValueError, match="differs from the frozen harness contract"):
        harness.frozen_control_manifest(path)


CONTROL_V2_SHA256 = "d5f446238a042438eb8c75ba3d432150c27e7ca6b4ecd72eed11fcfaa26e5787"
"""The digest of the deleted control-v2.json, whose protocol v3 carries."""

OLD_DECISION = """  \"decision\": {
    \"enforced\": true,
    \"rule\": \"On every trial, the generic arm's position RMSE and attitude RMSE are at or below the structured arm's on that same trial, and no trial is terminated.\",
    \"gates_from\": \"this manifest. The protocol is control-v1's, constant for constant; only the rule's standing changed.\",
    \"meaning\": \"A Cascade X8 tracking measurement of this trial set only. Not hardware readiness, not a real-time claim, and not calibrated uncertainty. The rule is a gate: one trial whose generic position or attitude RMSE is above the structured arm's, or one terminated trial, rejects the run.\"
  }
}
"""


def test_the_committed_manifest_carries_control_v2s_protocol_byte_for_byte():
    """Undoing the id and the decision reproduces the deleted control-v2 exactly.

    The plant hash, the calibration, the reference, the duration, the
    repetitions, the arms and the controller policy cannot have moved in this
    change, because reversing the id and the decision block recovers v2's own
    digest from v3's bytes.
    """

    text = MANIFEST.read_text()
    assert text.count('  "id": "control-v3",\n') == 1
    text = text.replace('  "id": "control-v3",\n', '  "id": "control-v2",\n', 1)
    head, marker, _ = text.partition('  "reference": {\n')
    assert marker
    assert (
        hashlib.sha256((head + OLD_DECISION).encode()).hexdigest() == CONTROL_V2_SHA256
    )


def test_the_committed_manifest_matches_the_module_constant():
    assert harness.COMMITTED_CONTROL_MANIFEST.name == "control-v3.json"
    assert harness.sha256(MANIFEST) == harness.CONTROL_MANIFEST_SHA256


# --- the decision -----------------------------------------------------------


def _row(repetition, arm, position, attitude, **overrides):
    row = dict(
        repetition=repetition,
        arm=arm,
        terminated=False,
        completed_intervals=240,
        requested_intervals=240,
        failure=None,
        tracking_rmse=dict(
            position_rmse_m=position,
            velocity_rmse_m_s=0.1,
            attitude_rmse_deg=attitude,
            angular_velocity_rmse_rad_s=0.05,
        ),
        deadline_misses=3,
        solve_deadline_misses=0,
        directory=f"trial-{repetition}/{arm}",
    )
    row.update(overrides)
    return row


def _rows(generic_position=0.4, generic_attitude=0.5):
    return [
        _row(repetition, arm, position, attitude)
        for repetition in (0, 1)
        for arm, position, attitude in (
            ("generic", generic_position, generic_attitude),
            ("structured", 0.5, 0.6),
        )
    ]


@pytest.fixture(scope="module")
def manifest():
    return harness.frozen_control_manifest(MANIFEST)


def _reporting(manifest):
    """The same manifest with the rule reporting rather than gating."""

    loosened = copy.deepcopy(manifest)
    loosened["decision"]["enforced"] = False
    return loosened


def _reference(rows):
    """The reference a run of these rows would have been frozen from."""

    tracking = {}
    for row in rows:
        tracking.setdefault(str(row["repetition"]), {})[row["arm"]] = {
            metric: row["tracking_rmse"][metric] for metric in harness.CONTROL_METRICS
        }
    return dict(tracking_rmse=tracking)


def _failing_reference():
    """A reference whose generic arm failed the rule on every trial and metric."""

    return _reference(_rows(generic_position=9.0, generic_attitude=9.0))


def test_a_generic_arm_at_or_below_the_structured_arm_meets_the_rule(manifest):
    decision = harness.control_decide(manifest, _rows())
    assert decision["rule_met"] is True
    assert decision["rule_enforced"] is True
    assert decision["accepted"] is True
    assert decision["rule_breaches"] == []
    assert decision["gating_rule_breaches"] == 0
    assert decision["tracking_rmse"]["0"]["position_rmse_m"] == dict(
        generic=0.4, structured=0.5, reference=None, reference_meets_rule=None
    )
    assert decision["reference_compared"] is False
    assert harness.control_decide(_reporting(manifest), _rows())["accepted"] is True


def test_the_rule_gates_only_a_trial_the_reference_already_met(manifest):
    """The whole semantics, isolated: no regression, one case gated, one not."""

    reference = _reference(_rows())
    reference["tracking_rmse"]["0"]["generic"]["position_rmse_m"] = 0.5
    reference["tracking_rmse"]["1"]["generic"]["attitude_rmse_deg"] = 0.7
    rows = _rows()
    # Repetition 0 position: the reference met the rule at 0.5, equal to the
    # structured arm, and this run does not, inside its own regression limit
    # 0.5 * 1.05 + 0.005 = 0.53.
    rows[0]["tracking_rmse"]["position_rmse_m"] = 0.52
    # Repetition 1 attitude: the reference already failed at 0.7, above the
    # structured arm's 0.6, and this run still fails inside its own limit.
    rows[2]["tracking_rmse"]["attitude_rmse_deg"] = 0.71
    decision = harness.control_decide(manifest, rows, reference, "b" * 64)
    assert decision["reference_regressions"] == []
    assert decision["reference_compared"] is True
    assert decision["reference_sha256"] == "b" * 64
    assert decision["rule_met"] is False
    assert decision["accepted"] is False and decision["decision"] == "reject"
    assert decision["gating_rule_breaches"] == 1
    gated = [b for b in decision["rule_breaches"] if b["gating"]]
    reported = [b for b in decision["rule_breaches"] if not b["gating"]]
    assert [b["trial"] for b in gated] == ["0-generic"]
    assert [b["trial"] for b in reported] == ["1-generic"]
    summary = decision["tracking_rmse"]["1"]["attitude_rmse_deg"]
    assert summary["reference"] == 0.7 and summary["reference_meets_rule"] is False
    json.dumps(decision, allow_nan=False)


def test_a_trial_the_reference_already_failed_is_reported_and_accepted(manifest):
    """The defect this manifest fixes: an unmet trial must not block a change."""

    reference = _reference(_rows(generic_position=0.6))
    rows = _rows(generic_position=0.61)
    decision = harness.control_decide(manifest, rows, reference)
    assert decision["accepted"] is True and decision["decision"] == "accept"
    # Accepted is not met: the status table's row reads rule_met, not accepted.
    assert decision["rule_met"] is False
    assert decision["gating_rule_breaches"] == 0
    assert [b["gating"] for b in decision["rule_breaches"]] == [False, False]


def test_a_regression_rejects_a_run_that_meets_the_rule(manifest):
    """Condition (a) stands on its own: the rule held and the run still fails."""

    reference = _reference(_rows())
    rows = _rows(generic_position=0.4 * 1.05 + 0.005 + 1e-9)
    decision = harness.control_decide(manifest, rows, reference)
    assert decision["rule_met"] is True and decision["accepted"] is False
    assert decision["rule_breaches"] == []
    assert [r["trial"] for r in decision["reference_regressions"]] == [
        "0-generic",
        "1-generic",
    ]
    assert decision["reference_regressions"][0]["gate"] == "reference_tracking_rmse"
    json.dumps(decision, allow_nan=False)


def test_a_reference_without_the_trial_fails_closed(manifest):
    reference = _reference(_rows())
    reference["tracking_rmse"].pop("1")
    decision = harness.control_decide(manifest, _rows(), reference)
    assert decision["accepted"] is False
    assert {r["gate"] for r in decision["reference_regressions"]} == {
        "reference_present"
    }


@pytest.mark.parametrize("value", [float("nan"), float("inf"), None, "0.1", True])
def test_a_reference_value_that_is_not_a_number_fails_closed(manifest, value):
    reference = _reference(_rows())
    reference["tracking_rmse"]["0"]["generic"]["position_rmse_m"] = value
    decision = harness.control_decide(manifest, _rows(), reference)
    assert decision["accepted"] is False
    assert decision["reference_regressions"][0] == dict(
        trial="0-generic", metric="position_rmse_m", gate="reference_present"
    )
    json.dumps(decision, allow_nan=False)


def test_with_no_reference_the_rule_is_reported_and_gates_nothing(manifest):
    decision = harness.control_decide(manifest, _rows(generic_position=9.0))
    assert decision["rule_met"] is False and decision["accepted"] is True
    assert decision["gating_rule_breaches"] == 0
    assert all(b["gating"] is False for b in decision["rule_breaches"])


def test_a_trial_above_the_structured_arm_now_rejects_the_run(manifest):
    reference = _reference(_rows())
    rows = _rows(generic_position=0.7)
    decision = harness.control_decide(manifest, rows, reference)
    assert decision["rule_met"] is False
    assert decision["accepted"] is False
    assert decision["decision"] == "reject"
    assert [breach["metric"] for breach in decision["rule_breaches"]] == [
        "position_rmse_m",
        "position_rmse_m",
    ]
    assert decision["rule_breaches"][0]["value"] == 0.7
    assert decision["rule_breaches"][0]["limit"] == 0.5
    assert decision["rule_breaches"][0]["gating"] is True
    reporting = harness.control_decide(_reporting(manifest), rows, reference)
    assert reporting["rule_met"] is False
    assert reporting["accepted"] is False
    assert [r["trial"] for r in reporting["reference_regressions"]] == [
        "0-generic",
        "1-generic",
    ]


def test_one_trial_above_the_structured_arm_is_enough_to_reject(manifest):
    reference = _reference(_rows(generic_attitude=0.58))
    rows = _rows()
    rows[0] = _row(0, "generic", 0.4, 0.61)
    decision = harness.control_decide(manifest, rows, reference)
    assert decision["rule_met"] is False
    assert decision["accepted"] is False
    assert decision["reference_regressions"] == []
    assert decision["rule_breaches"][0]["metric"] == "attitude_rmse_deg"
    assert decision["gating_rule_breaches"] == 1
    assert harness.control_decide(_reporting(manifest), rows, reference)["accepted"]


def test_an_equal_metric_meets_the_rule_and_a_hair_above_does_not(manifest):
    """The rule is at-or-below: equality passes, one ulp above rejects."""

    reference = _reference(_rows(0.5, 0.6))
    assert harness.control_decide(manifest, _rows(0.5, 0.6), reference)["accepted"]
    above = harness.control_decide(
        manifest, _rows(math.nextafter(0.5, 1.0), 0.6), reference
    )
    assert above["accepted"] is False
    assert above["reference_regressions"] == []
    assert above["rule_breaches"][0]["gate"] == "structured_arm_rmse"
    assert above["rule_breaches"][0]["gating"] is True


def test_a_terminated_trial_fails_closed(manifest):
    rows = _rows()
    rows[0] = _row(
        0,
        "generic",
        0.1,
        0.1,
        terminated=True,
        completed_intervals=117,
        failure="nonfinite plant state",
    )
    # Structure never waits for the reference: a reference that failed the rule
    # on every trial cannot excuse a trial that did not finish.
    for candidate in (manifest, _reporting(manifest)):
        for reference in (None, _failing_reference()):
            decision = harness.control_decide(candidate, rows, reference)
            assert decision["accepted"] is False
            assert decision["rule_met"] is False
            assert decision["gate_breaches"][0]["gate"] == "trial_complete"


def test_a_short_trial_fails_closed_even_when_it_does_not_say_so(manifest):
    rows = _rows()
    rows[0] = _row(0, "generic", 0.1, 0.1, completed_intervals=239)
    decision = harness.control_decide(manifest, rows, _failing_reference())
    assert decision["accepted"] is False
    assert decision["gate_breaches"][0]["gate"] == "declared_intervals"


def test_missing_and_duplicate_and_undeclared_trials_fail_closed(manifest):
    missing = harness.control_decide(manifest, _rows()[:3])
    assert missing["accepted"] is False
    assert missing["gate_breaches"][0]["gate"] == "trial_present"

    duplicated = harness.control_decide(
        manifest, _rows() + [_row(0, "generic", 0.1, 0.1)]
    )
    assert duplicated["accepted"] is False
    assert any(b["gate"] == "trial_unique" for b in duplicated["gate_breaches"])

    undeclared = harness.control_decide(
        manifest, _rows() + [_row(0, "bootstrap", 0.1, 0.1)]
    )
    assert undeclared["accepted"] is False
    assert any(b["gate"] == "trial_declared" for b in undeclared["gate_breaches"])

    absent = harness.control_decide(manifest, [])
    assert absent["accepted"] is False
    assert len(absent["gate_breaches"]) == 4


def test_a_nonfinite_metric_fails_closed_and_stays_json_serializable(manifest):
    for value in (float("nan"), float("inf"), None, "0.1", True):
        rows = _rows()
        rows[0]["tracking_rmse"]["position_rmse_m"] = value
        decision = harness.control_decide(manifest, rows)
        assert decision["accepted"] is False
        assert decision["gate_breaches"][0]["gate"] == "finite_rmse"
        assert decision["gate_breaches"][0]["value"] is None
        json.dumps(decision, allow_nan=False)

    rows = _rows()
    rows[0]["tracking_rmse"] = None
    decision = harness.control_decide(manifest, rows)
    assert decision["accepted"] is False
    json.dumps(decision, allow_nan=False)


def test_a_negative_metric_fails_closed(manifest):
    rows = _rows()
    rows[0]["tracking_rmse"]["attitude_rmse_deg"] = -1.0
    decision = harness.control_decide(manifest, rows)
    assert decision["accepted"] is False
    assert decision["gate_breaches"][0]["gate"] == "finite_rmse"


# --- the replay -------------------------------------------------------------


def _trajectory(seed):
    states, commands = _recording(f"r{seed}", 20 + seed)
    return Trajectory(
        time_s=np.arange(len(states)) * DT_S,
        states=states,
        controls=commands,
        control_prefix=commands[:1],
        spec=harness.control_telemetry_spec(harness.frozen_control_manifest(MANIFEST)),
        labels={"source_group": f"cascade-calibration-{seed}"},
    )


def _anchor(directory):
    """Where a fabricated run's committed reference lives, beside the run."""

    return Path(directory).parent / "control-reference.json"


def _fabricate_run(directory, manifest, learned, offsets, with_reference=True):
    """A control run directory that was never tracked, for the replay to check."""

    directory.mkdir(parents=True, exist_ok=True)
    import shutil

    shutil.copyfile(MANIFEST, directory / "manifest.json")
    names = []
    digests = {}
    for seed in manifest["calibration"]["seeds"]:
        flight = _trajectory(seed)
        save_trajectory_npz(flight, directory / f"recording-{seed}.npz")
        names.append(f"recording-{seed}.npz")
        digests[f"recording-{seed}"] = trajectory_content_digest(flight)
    learned.save(directory / "generic.npz")
    (directory / "structured.json").write_text('{"belief": "fabricated"}\n')
    (directory / "structured_report.json").write_text('{"report": "fabricated"}\n')

    initial_state = LEVEL.copy()
    declared = manifest["tracking_reference"]
    intervals = manifest["trial"]["intervals"]
    times = np.arange(intervals + 1) * DT_S
    reference = harness.control_reference(initial_state, times, declared)
    rows = []
    for repetition in range(manifest["trial"]["repetitions"]):
        for arm in ("generic", "structured"):
            case = directory / f"trial-{repetition}" / arm
            case.mkdir(parents=True, exist_ok=True)
            states = reference.copy()
            states[:, 0] += offsets[arm]
            np.savez_compressed(
                case / "tracking.npz",
                time_s=times,
                states=states,
                reference_states=reference,
                commands=np.zeros((intervals, 3)),
                tick_times_s=np.full(intervals, 0.01),
                source_clock_lags_s=np.zeros(intervals),
                solve_times_s=np.full(intervals, 0.002),
                solver_used=np.ones(intervals, dtype=bool),
                used_fallback=np.zeros(intervals, dtype=bool),
                initial_state=initial_state,
            )
            metrics = state_rmse_metrics(states[1:], reference[1:])
            row = dict(
                arm=arm,
                repetition=repetition,
                completed_intervals=intervals,
                requested_intervals=intervals,
                terminated=False,
                failure=None,
                tracking_rmse=metrics,
                deadline_misses=0,
                solve_deadline_misses=0,
                directory=f"trial-{repetition}/{arm}",
                files=harness._files(case, ["tracking.npz"]),
            )
            harness.write(case / "trial.json", row)
            rows.append(row)
    harness.write(directory / "results.json", rows)
    harness.write(
        directory / "calibration.json",
        dict(
            recordings=digests,
            training=[
                f"recording-{s}" for s in manifest["calibration"]["training_seeds"]
            ],
            reserved=[
                f"recording-{s}" for s in manifest["calibration"]["reserved_seeds"]
            ],
            generic_fingerprint=learned.fingerprint(),
            files=harness._files(
                directory,
                names + ["structured.json", "structured_report.json", "generic.npz"],
            ),
        ),
    )
    anchor, digest = None, None
    if with_reference:
        committed = _anchor(directory)
        harness.write(committed, _reference(rows))
        shutil.copyfile(committed, directory / "reference.json")
        anchor, digest = harness.read(committed), harness.sha256(committed)
    harness.write(
        directory / "decision.json",
        harness.control_decide(manifest, rows, anchor, digest),
    )
    return rows


def test_the_replay_recomputes_every_metric_and_the_decision(
    tmp_path, manifest, learned
):
    directory = tmp_path / "control-run"
    _fabricate_run(directory, manifest, learned, dict(generic=0.4, structured=0.5))
    result = harness.verify_control(directory, manifest, _anchor(directory))
    assert result["tier"] == "control"
    assert result["verified_trials"] == 4
    assert result["decision"]["accepted"] is True
    assert result["decision"]["rule_met"] is True
    assert "not rerun" in result["meaning"]
    # The tier is chosen by the digest of the manifest the run copied.
    assert harness.verify(directory, _anchor(directory))["tier"] == "control"


def test_the_replay_rejects_an_altered_tracking_array(tmp_path, manifest, learned):
    directory = tmp_path / "control-run"
    _fabricate_run(directory, manifest, learned, dict(generic=0.4, structured=0.5))
    case = directory / "trial-0" / "generic"
    with np.load(case / "tracking.npz", allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    arrays["states"] = arrays["reference_states"].copy()
    np.savez_compressed(case / "tracking.npz", **arrays)
    with pytest.raises(ValueError, match="altered artifact"):
        harness.verify_control(directory, manifest, _anchor(directory))


def test_the_replay_rejects_an_altered_calibration_recording(
    tmp_path, manifest, learned
):
    directory = tmp_path / "control-run"
    _fabricate_run(directory, manifest, learned, dict(generic=0.4, structured=0.5))
    calibration = harness.read(directory / "calibration.json")
    calibration["files"].pop("recording-0.npz")
    harness.write(directory / "calibration.json", calibration)
    save_trajectory_npz(_trajectory(9), directory / "recording-0.npz")
    with pytest.raises(ValueError, match="altered calibration recording"):
        harness.verify_control(directory, manifest, _anchor(directory))


def test_the_replay_rejects_an_altered_fitted_model(tmp_path, manifest, learned):
    directory = tmp_path / "control-run"
    _fabricate_run(directory, manifest, learned, dict(generic=0.4, structured=0.5))
    calibration = harness.read(directory / "calibration.json")
    calibration["generic_fingerprint"] = "0" * 64
    calibration["files"].pop("generic.npz")
    harness.write(directory / "calibration.json", calibration)
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        harness.verify_control(directory, manifest, _anchor(directory))


def test_the_replay_rejects_a_reference_the_run_invented(tmp_path, manifest, learned):
    directory = tmp_path / "control-run"
    _fabricate_run(directory, manifest, learned, dict(generic=0.4, structured=0.5))
    case = directory / "trial-0" / "generic"
    with np.load(case / "tracking.npz", allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    arrays["reference_states"] = arrays["states"].copy()
    np.savez_compressed(case / "tracking.npz", **arrays)
    row = harness.read(case / "trial.json")
    row["tracking_rmse"] = state_rmse_metrics(
        arrays["states"][1:], arrays["states"][1:]
    )
    row["files"] = harness._files(case, ["tracking.npz"])
    harness.write(case / "trial.json", row)
    rows = harness.read(directory / "results.json")
    rows[0] = row
    harness.write(directory / "results.json", rows)
    harness.write(directory / "decision.json", harness.control_decide(manifest, rows))
    with pytest.raises(AssertionError):
        harness.verify_control(directory, manifest, _anchor(directory))


def test_the_replay_rejects_a_decision_that_does_not_follow(
    tmp_path, manifest, learned
):
    directory = tmp_path / "control-run"
    _fabricate_run(directory, manifest, learned, dict(generic=0.9, structured=0.5))
    decision = harness.read(directory / "decision.json")
    assert decision["rule_met"] is False
    decision["rule_met"] = True
    decision["rule_breaches"] = []
    harness.write(directory / "decision.json", decision)
    with pytest.raises(ValueError, match="replayed decision differs"):
        harness.verify_control(directory, manifest, _anchor(directory))


def _loosen(value, factor=10):
    if isinstance(value, float):
        return value * factor
    if isinstance(value, dict):
        return {key: _loosen(item, factor) for key, item in value.items()}
    return value


def test_a_control_run_cannot_relax_its_own_regression_gate(
    tmp_path, manifest, learned
):
    """The replay must not read the threshold the run saved beside itself."""

    directory = tmp_path / "control-run"
    _fabricate_run(directory, manifest, learned, dict(generic=0.4, structured=0.5))
    copied = directory / "reference.json"
    harness.write(copied, _loosen(harness.read(copied)))
    with pytest.raises(ValueError, match="cannot relax its own regression gate"):
        harness.verify_control(directory, manifest, _anchor(directory))


def test_a_control_run_and_its_reference_must_agree_about_existing(
    tmp_path, manifest, learned
):
    compared = tmp_path / "compared"
    _fabricate_run(compared, manifest, learned, dict(generic=0.4, structured=0.5))
    with pytest.raises(ValueError, match="cannot be anchored"):
        harness.verify_control(compared, manifest, tmp_path / "absent.json")

    uncompared = tmp_path / "second" / "control-run"
    _fabricate_run(
        uncompared,
        manifest,
        learned,
        dict(generic=0.4, structured=0.5),
        with_reference=False,
    )
    assert not harness.read(uncompared / "decision.json")["reference_compared"]
    harness.write(_anchor(uncompared), {"tracking_rmse": {}})
    with pytest.raises(ValueError, match="cannot be anchored"):
        harness.verify_control(uncompared, manifest, _anchor(uncompared))


def test_a_forged_control_reference_digest_in_the_decision_is_rejected(
    tmp_path, manifest, learned
):
    directory = tmp_path / "control-run"
    _fabricate_run(directory, manifest, learned, dict(generic=0.4, structured=0.5))
    decision = harness.read(directory / "decision.json")
    decision["reference_sha256"] = "0" * 64
    harness.write(directory / "decision.json", decision)
    with pytest.raises(ValueError, match="reference_sha256"):
        harness.verify_control(directory, manifest, _anchor(directory))


def test_a_control_replay_recomputes_the_reference_regressions(
    tmp_path, manifest, learned
):
    directory = tmp_path / "control-run"
    _fabricate_run(directory, manifest, learned, dict(generic=0.4, structured=0.5))
    decision = harness.read(directory / "decision.json")
    decision["reference_regressions"] = [dict(trial="0-generic", gate="invented")]
    harness.write(directory / "decision.json", decision)
    with pytest.raises(ValueError, match="reference_regressions"):
        harness.verify_control(directory, manifest, _anchor(directory))


# --- the tests that drive Cascade itself ------------------------------------


@pytest.mark.cascade
def test_the_harness_collects_the_examples_own_calibration_recording(monkeypatch):
    """The tier's recording protocol is the example's, constant for constant."""

    pytest.importorskip("cascade")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    example = importlib.import_module("cascade_refinement")

    manifest = copy.deepcopy(harness.frozen_control_manifest(MANIFEST))
    manifest["calibration"]["duration_s"] = 0.5
    spec, model, trim, state, command = harness.control_fixture(manifest)
    tier_plant = harness._control_plant(manifest, spec, model)
    mine = harness.control_recording(manifest, tier_plant, trim, state, command, 0)

    plant, example_trim, example_state, example_command = example.fixture()
    theirs = example.collect_recording(
        plant, example_trim, example_state, example_command, 0, 0.5
    )
    np.testing.assert_allclose(mine.states, theirs.states, rtol=0, atol=0)
    np.testing.assert_allclose(mine.controls, theirs.controls, rtol=0, atol=0)
    assert mine.spec.prediction_spec() == theirs.spec.prediction_spec()
    assert mine.spec.to_dict() == example.telemetry_spec().to_dict()


@pytest.mark.cascade
def test_the_harness_reference_is_the_examples_cruise_reference(monkeypatch):
    pytest.importorskip("cascade")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    example = importlib.import_module("cascade_refinement")

    manifest = harness.frozen_control_manifest(MANIFEST)
    _, _, _, state, command = harness.control_fixture(manifest)
    plant = example.tracking_plant(state, command)
    times = np.arange(0, 12.0, 0.05)
    np.testing.assert_allclose(
        harness.control_reference(state, times, manifest["tracking_reference"]),
        plant.reference(times),
        rtol=0,
        atol=0,
    )


@pytest.mark.cascade
def test_the_generic_arm_tracks_the_cascade_plant_for_a_short_trial(tmp_path):
    """One short paced trial end to end: the seam, the plant, and the metrics."""

    pytest.importorskip("cascade")
    manifest = copy.deepcopy(harness.frozen_control_manifest(MANIFEST))
    manifest["calibration"]["duration_s"] = 3.0
    manifest["trial"]["intervals"] = 20
    manifest["trial"]["duration_s"] = 1.0

    spec, model, trim, state, command = harness.control_fixture(manifest)
    plant = harness._control_plant(manifest, spec, model)
    recordings = [
        (
            f"recording-{seed}",
            harness.control_recording(manifest, plant, trim, state, command, seed),
        )
        for seed in (0, 1)
    ]
    with jax.enable_x64(True):
        model = fit(harness.control_collection(manifest, recordings))
    arm = harness._GenericArm(manifest, model)
    assert arm.summary()["uncertainty_available"] is False
    assert arm.summary()["horizon_s"] == pytest.approx(0.25)

    def reference_fn(times):
        return harness.control_reference(state, times, manifest["tracking_reference"])

    harness._control_prewarm(arm, manifest, recordings[0][1], reference_fn)
    tracking = harness._control_tracking_plant(manifest, state, command)
    row = harness._control_trial(manifest, arm, tracking, reference_fn, tmp_path / "t")
    assert row["terminated"] is False
    assert row["completed_intervals"] == 20
    assert row["model_not_ready_intervals"] == arm.controller.required_observations - 1
    assert row["maximum_command_bound_violation"] == 0.0
    assert math.isfinite(row["tracking_rmse"]["position_rmse_m"])
