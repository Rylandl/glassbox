"""Control-tier contracts: the state map, the plan model, the rule, the replay.

Nothing here needs Cascade. The state map is exercised on canonical rigid-body
states, the plan model on a fabricated learner, the decision on fabricated
trial results, and the replay on a run directory that was never tracked. The
tests that drive the simulator itself carry the ``cascade`` marker at the
bottom of this module.
"""

import copy
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
from conftest import seal_evidence

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
    state_plus_tangent,
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

MANIFEST = Path(__file__).resolve().parents[1] / "docs/harness/control-v4.json"
DT_S = 0.05
MINIMUM = np.array([0.0, -0.35, -0.35])
MAXIMUM = np.array([1.0, 0.35, 0.35])
COMMAND_CHANNELS = (
    "throttle [normalized,throttle]",
    "roll [rad,roll]",
    "pitch [rad,pitch]",
)
LEVEL = np.array([1.0, -2.0, 100.0, 18.0, 0.3, -0.2, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
TRIM = np.array([0.0, 0.0, 100.0, 18.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
"""Level cruise at the plant's own trim, the state the reference is anchored to."""


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
        envelope=learned.envelope,
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
    # The learner measures a forecast-error envelope and resolves no parameter
    # direction, so it declares the first and not the second.
    assert plan.uncertainty_available is True
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
    # The measured envelope reaches the plan, and it is the one the model's own
    # values carry rather than anything the rollout invented.
    assert float(jnp.max(jnp.abs(prediction.tangent_covariance))) > 0.0
    np.testing.assert_array_equal(
        np.asarray(prediction.tangent_covariance),
        np.asarray(ready.values.forecast_error_covariance),
    )
    assert bool(np.isfinite(np.asarray(prediction.mean_states)).all())
    np.testing.assert_allclose(np.asarray(prediction.mean_states[0]), LEVEL, atol=1e-6)

    reference = ReferenceTrajectory(np.tile(LEVEL, (horizon + 1, 1)))
    cost = ready.stage_cost(prediction, reference.states, jnp.zeros(3), ready.policy)
    assert cost.shape == ()
    assert math.isfinite(float(cost))

    measurements = ready.measure(prediction)
    # Validity utilization stays zero because no support envelope is declared;
    # normalized uncertainty is now the envelope's own widest marginal.
    assert float(measurements.maximum_validity_utilization) == 0.0
    assert float(measurements.maximum_normalized_uncertainty) > 0.0
    assert math.isfinite(float(measurements.maximum_normalized_uncertainty))
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
    assert manifest["id"] == "control-v4"
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
    path = tmp_path / "control-v4.json"
    path.write_text(json.dumps(loosened, indent=2) + "\n")
    with pytest.raises(ValueError, match="differs from the frozen harness contract"):
        harness.frozen_control_manifest(path)


def test_the_committed_manifest_carries_control_v2s_plant_and_controller(manifest):
    """What control-v3 changed, and what it did not.

    The plant, the arms, the controller policy, the pilot, the durations and
    the calibration seeds are control-v2's. The reference, the trial length,
    the per-trial initial state and the airspeed setpoint amplitude are not.
    """

    plant = manifest["plant"]
    assert plant["source_revision"] == "d6613886f8bbdc514a9ae88ff344c23af994df6a"
    assert (
        plant["spec_hash"]
        == "fc169f5d036c16cbeff8764d4784f4157327a7f48767e7118a35fbd07efb341e"
    )
    assert plant["trim"] == {"airspeed_m_s": 18.0, "altitude_m": 100.0}
    calibration = manifest["calibration"]
    assert calibration["seeds"] == [0, 1, 2, 3]
    assert calibration["training_seeds"] == [0, 1, 2]
    assert calibration["reserved_seeds"] == [3]
    assert calibration["duration_s"] == 8.0
    assert calibration["pilot_periods"] == {"rate": 1, "attitude": 1, "guidance": 2}
    assert calibration["excitation_amplitudes"] == [0.035, 0.025, 0.02]
    assert calibration["excitation_rates_rad_s"] == [1.3, 2.1, 1.7]
    assert manifest["controller"]["maximum_iterations"] == 4
    assert manifest["arms"]["structured"]["optimization_steps"] == 200
    assert manifest["arms"]["structured"]["evaluation_horizons_s"] == [0.1, 0.4, 0.8]
    assert sorted(manifest["arms"]) == ["generic", "structured"]
    # What changed, and only this.
    assert calibration["setpoint"]["airspeed_amplitude_m_s"] == 2.5
    assert manifest["tracking_reference"]["kind"] == "lateral_and_altitude_tracking"
    assert manifest["trial"]["duration_s"] == 16.0
    assert manifest["trial"]["intervals"] == 320
    assert manifest["trial"]["initial_state_seeds"] == [101, 102]


def test_the_manifest_declares_the_pages_task_and_its_pass_criterion(manifest):
    reference = manifest["tracking_reference"]
    assert reference["source"] == "docs/cascade-accuracy.md"
    assert reference["lateral_amplitude_m"] == 1.0
    assert reference["lateral_rate_rad_s"] == 0.35
    assert reference["altitude_amplitude_m"] == 0.75
    assert reference["altitude_rate_rad_s"] == 0.3
    criterion = manifest["metrics"]["pass_criterion"]
    assert criterion["tolerance_m"] == 0.5
    assert criterion["settled_after_s"] == 2.0
    assert criterion["minimum_fraction"] == 0.95
    assert criterion["scored_samples"] == 281
    assert "pass_criterion" in manifest["metrics"]["informational"]
    assert "pass_criterion" not in manifest["metrics"]["gating"]
    assert (
        manifest["calibration"]["excitation"]["minimum_standard_deviation_fraction"]
        == 0.1
    )


def test_the_reference_is_the_pages_own_lateral_and_altitude_task(manifest):
    """sin(0.35 t) laterally, 100 + 0.75 sin(0.3 t) in altitude, 18 m/s forward."""

    times = np.arange(0.0, 16.0 + DT_S, DT_S)
    rows = harness.control_reference(TRIM, times, manifest["tracking_reference"])
    np.testing.assert_allclose(rows[:, 0], 18.0 * times, rtol=0, atol=1e-12)
    np.testing.assert_allclose(rows[:, 1], np.sin(0.35 * times), rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        rows[:, 2], 100.0 + 0.75 * np.sin(0.3 * times), rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        rows[:, 3], np.full_like(times, 18.0), rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        rows[:, 4], 0.35 * np.cos(0.35 * times), rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        rows[:, 5], 0.225 * np.cos(0.3 * times), rtol=0, atol=1e-12
    )
    # The task does not move with a trial's own perturbed start.
    moved = harness.control_initial_state(manifest, TRIM, 101)
    assert not np.allclose(moved[:3], TRIM[:3])
    np.testing.assert_allclose(
        harness.control_reference(TRIM, times, manifest["tracking_reference"]),
        rows,
        rtol=0,
        atol=0,
    )


def test_the_initial_state_is_the_pages_own_perturbation(manifest):
    """The same twelve-vector draw, in the same order, through the retraction."""

    for seed in manifest["trial"]["initial_state_seeds"]:
        generator = np.random.default_rng(seed)
        offset = np.zeros(12)
        offset[1:3] = generator.uniform(-0.15, 0.15, 2)
        offset[4:6] = generator.uniform(-0.05, 0.05, 2)
        offset[6:9] = generator.uniform(-0.01, 0.01, 3)
        offset[9:12] = generator.uniform(-0.02, 0.02, 3)
        with jax.enable_x64(True):
            expected = np.asarray(
                state_plus_tangent(jnp.asarray(LEVEL), jnp.asarray(offset))
            )
            measured = harness.control_initial_state(manifest, LEVEL, seed)
        np.testing.assert_allclose(measured, expected, rtol=0, atol=0)
        assert abs(measured[0] - LEVEL[0]) < 1e-12
        assert 0 < abs(measured[1] - LEVEL[1]) <= 0.15
        assert 0 < abs(measured[2] - LEVEL[2]) <= 0.15
        np.testing.assert_allclose(
            np.linalg.norm(measured[6:10]), 1.0, rtol=0, atol=1e-12
        )
    first, second = (
        harness.control_initial_state(manifest, LEVEL, seed)
        for seed in manifest["trial"]["initial_state_seeds"]
    )
    assert not np.allclose(first, second)


def _tracked(manifest, lateral, altitude, intervals=None):
    """One trial's saved states: the reference with a declared offset added."""

    requested = manifest["trial"]["intervals"]
    times = np.arange(requested + 1) * DT_S
    states = harness.control_reference(LEVEL, times, manifest["tracking_reference"])
    states = states.copy()
    states[:, 1] += lateral
    states[:, 2] += altitude
    return states if intervals is None else states[: intervals + 1]


def test_the_pass_criterion_is_the_pages_own(manifest):
    exact = harness.control_pass_criterion(
        _tracked(manifest, 0.0, 0.0), LEVEL, manifest
    )
    assert exact["scored_samples"] == 281
    assert exact["within_samples"] == 281
    assert exact["within_tolerance_fraction"] == 1.0
    assert exact["met"] is True and exact["terminated"] is False
    assert exact["lateral_rmse_m"] == pytest.approx(0.0)

    inside = harness.control_pass_criterion(
        _tracked(manifest, 0.49, -0.49), LEVEL, manifest
    )
    assert inside["within_tolerance_fraction"] == 1.0 and inside["met"] is True
    assert inside["altitude_rmse_m"] == pytest.approx(0.49)

    outside = harness.control_pass_criterion(
        _tracked(manifest, 0.0, 0.51), LEVEL, manifest
    )
    assert outside["within_tolerance_fraction"] == 0.0 and outside["met"] is False
    json.dumps(outside, allow_nan=False)


def test_an_unexecuted_interval_counts_as_outside_the_pass_tolerance(manifest):
    """The page's accounting: a failed trial is not discarded, it is outside."""

    stopped = harness.control_pass_criterion(
        _tracked(manifest, 0.0, 0.0, intervals=200), LEVEL, manifest
    )
    assert stopped["terminated"] is True and stopped["met"] is False
    assert stopped["scored_samples"] == 281
    assert stopped["within_samples"] == 200 - 39
    assert stopped["within_tolerance_fraction"] == pytest.approx(161 / 281)
    # An unbounded error is serialized as null, never as zero.
    assert stopped["lateral_rmse_m"] is None
    json.dumps(stopped, allow_nan=False)


def test_command_excitation_is_measured_against_the_declared_range(manifest):
    steps = np.zeros((200, 3))
    steps[::2] = [0.2, 0.14, 0.14]
    measured = harness.control_excitation(manifest, [("recording-0", _sampled(steps))])
    assert measured["declared_range"] == [1.0, 0.7, 0.7]
    np.testing.assert_allclose(
        measured["standard_deviation"]["recording-0"], [0.1, 0.07, 0.07]
    )
    np.testing.assert_allclose(measured["fraction"]["recording-0"], [0.1, 0.1, 0.1])
    assert (
        harness.control_excitation_shortfall(manifest, measured, ["recording-0"]) == []
    )


def test_a_command_channel_below_the_declared_fraction_falls_short(manifest):
    steps = np.zeros((200, 3))
    steps[::2] = [0.2, 0.14, 0.139]
    measured = harness.control_excitation(manifest, [("recording-0", _sampled(steps))])
    short = harness.control_excitation_shortfall(manifest, measured, ["recording-0"])
    assert [entry["channel"] for entry in short] == [2]
    assert short[0]["minimum"] == 0.1 and short[0]["fraction"] < 0.1
    # A recording that is not fitted does not have to meet it.
    assert harness.control_excitation_shortfall(manifest, measured, []) == []


def _sampled(controls):
    return SimpleNamespace(controls=np.asarray(controls, dtype=float))


def test_the_committed_control_reference_covers_every_trial(manifest):
    """The incumbent measurement, one entry per repetition, arm and metric."""

    assert harness.COMMITTED_CONTROL_REFERENCE.name == "control-reference.json"
    reference = harness.read(harness.COMMITTED_CONTROL_REFERENCE)
    assert reference["manifest"] == manifest["id"]
    assert reference["manifest_sha256"] == harness.sha256(MANIFEST)
    repetitions = [str(index) for index in range(manifest["trial"]["repetitions"])]
    assert sorted(reference["tracking_rmse"]) == repetitions
    assert sorted(reference["pass_criterion"]) == repetitions
    for index in repetitions:
        assert sorted(reference["tracking_rmse"][index]) == sorted(harness.CONTROL_ARMS)
        for metrics in reference["tracking_rmse"][index].values():
            assert sorted(metrics) == sorted(harness.CONTROL_METRICS)
            assert all(harness._number(value) is not None for value in metrics.values())
        for criterion in reference["pass_criterion"][index].values():
            assert criterion["scored_samples"] == 281
    for fractions in reference["command_excitation"].values():
        assert min(fractions) >= 0.1
    # The incumbent meets the rule on no trial and no metric, so the rule gates
    # none of them and a candidate is held to the regression reference instead.
    decision = harness.control_decide(manifest, _rows(), reference)
    assert decision["gating_rule_breaches"] == 0
    for index in repetitions:
        for metric in harness.CONTROL_METRICS:
            recorded = reference["tracking_rmse"][index]
            assert recorded["generic"][metric] > recorded["structured"][metric]


def test_the_committed_manifest_matches_the_module_constant():
    assert harness.COMMITTED_CONTROL_MANIFEST.name == "control-v4.json"
    assert harness.sha256(MANIFEST) == harness.CONTROL_MANIFEST_SHA256


# --- the decision -----------------------------------------------------------


def _row(repetition, arm, position, attitude, **overrides):
    row = dict(
        repetition=repetition,
        arm=arm,
        terminated=False,
        completed_intervals=320,
        requested_intervals=320,
        failure=None,
        tracking_rmse=dict(
            position_rmse_m=position,
            velocity_rmse_m_s=0.1,
            attitude_rmse_deg=attitude,
            angular_velocity_rmse_rad_s=0.05,
        ),
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
    rows[0] = _row(0, "generic", 0.1, 0.1, completed_intervals=319)
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
    """One fabricated calibration recording that meets the declared excitation."""

    states, commands = _recording(f"r{seed}", 20 + seed)
    commands = np.asarray(commands, dtype=float).copy()
    commands[::2] += np.array([0.25, 0.2, 0.2])
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

    reserved = [f"recording-{s}" for s in manifest["calibration"]["reserved_seeds"]]
    evidence_arrays = harness.control_evidence_arrays(
        manifest, [(name, _trajectory(int(name.split("-")[1]))) for name in reserved]
    )
    with jax.enable_x64(True):
        prediction, half_width, coverage = harness.control_evidence(
            learned, evidence_arrays
        )
    np.savez_compressed(
        directory / "evidence.npz",
        **evidence_arrays,
        prediction=prediction,
        envelope_half_width=half_width,
    )
    names.append("evidence.npz")

    anchor_state = LEVEL.copy()
    declared = manifest["tracking_reference"]
    intervals = manifest["trial"]["intervals"]
    times = np.arange(intervals + 1) * DT_S
    reference = harness.control_reference(anchor_state, times, declared)
    rows = []
    for repetition in range(manifest["trial"]["repetitions"]):
        seed = manifest["trial"]["initial_state_seeds"][repetition]
        initial_state = harness.control_initial_state(manifest, anchor_state, seed)
        for arm in ("generic", "structured"):
            case = directory / f"trial-{repetition}" / arm
            case.mkdir(parents=True, exist_ok=True)
            states = reference.copy()
            states[:, 0] += offsets[arm]
            states[0] = initial_state
            np.savez_compressed(
                case / "tracking.npz",
                time_s=times,
                states=states,
                reference_states=reference,
                commands=np.zeros((intervals, 3)),
                solver_used=np.ones(intervals, dtype=bool),
                used_fallback=np.zeros(intervals, dtype=bool),
                initial_state=initial_state,
                reference_anchor_state=anchor_state,
            )
            np.savez_compressed(
                case / "timing.npz",
                tick_times_s=np.full(intervals, 0.01),
                solve_times_s=np.full(intervals, 0.002),
                deadline_assessed=np.zeros(intervals, dtype=bool),
            )
            metrics = state_rmse_metrics(states[1:], reference[1:])
            row = dict(
                arm=arm,
                repetition=repetition,
                initial_state_seed=seed,
                completed_intervals=intervals,
                requested_intervals=intervals,
                terminated=False,
                failure=None,
                tracking_rmse=metrics,
                pass_criterion=harness.control_pass_criterion(
                    states, anchor_state, manifest
                ),
                solver_statuses={"converged": intervals},
                wall=harness.simulated_time_wall(
                    meaning=harness.SIMULATED_TIME_MEANING,
                    dt_s=DT_S,
                    deadline_s=manifest["trial"]["solve_deadline_s"],
                    tick_times=[0.01] * intervals,
                    solve_times=[0.002] * intervals,
                    elapsed_s=1.0,
                    solve_deadline_applied=False,
                    deadline_assessed_intervals=0,
                ),
                directory=f"trial-{repetition}/{arm}",
                files=harness._files(case, ["tracking.npz", "timing.npz"]),
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
            reserved=reserved,
            evidence=coverage,
            generic_fingerprint=learned.fingerprint(),
            command_excitation=harness.control_excitation(
                manifest,
                [
                    (f"recording-{seed}", _trajectory(seed))
                    for seed in manifest["calibration"]["seeds"]
                ],
            ),
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
        seal_evidence(
            directory,
            harness.control_decide(manifest, rows, anchor, digest),
            "control",
            manifest,
            rows=harness.control_evidence_table(coverage),
            reserved=reserved,
        ),
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
    assert result["decision"]["rule_met"] is True
    # The control side of the decision holds: nothing structural failed and
    # nothing regressed against this run's own fabricated reference. What
    # rejects it is the evidence band, which `evidence-v2` enforces: a
    # fabricated reserved recording forecast by a toy learner does not cover
    # the way the committed evidence reference records, and an enforced band
    # says so rather than reporting it.
    assert not result["decision"]["gate_breaches"]
    assert not result["decision"]["reference_regressions"]
    assert result["decision"]["evidence"]["band_enforced"] is True
    assert result["decision"]["evidence"]["accepted"] is False
    assert result["decision"]["accepted"] is False
    assert result["decision"]["pass_criterion"]["0-generic"]["scored_samples"] == 281
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


# --- simulated time ---------------------------------------------------------


class _RecordingArm:
    """One arm that answers with a declared command and remembers what it solved.

    It stands in for a fitted model so the loop itself can be measured: every
    solve is recorded with the keywords the loop passed, so a test can say both
    what the solver was asked for and what the plant was stepped with.
    """

    name = "recording"
    prediction_steps = 5

    def __init__(self, commands):
        self.commands = np.asarray(commands, dtype=float)
        self.solved = []
        self.keywords = []
        self.applied = []
        self.observed = 0

    @property
    def ready(self):
        # Two intervals of warm-up, so the not-ready branch is exercised too.
        return self.observed > 2

    def reset(self, initial_state, initial_command):
        self.observed = 0

    def observe(self, state):
        self.observed += 1

    def command_applied(self, command):
        self.applied.append(np.asarray(command, dtype=float).copy())

    def solve(self, state, reference, previous_command, **keywords):
        self.keywords.append(set(keywords))
        # Recorded as the array the loop is handed, so a test comparing it
        # with the applied command compares the solve with what was flown and
        # not one array with a differently rounded copy of itself.
        solved = jnp.asarray(self.commands[len(self.solved) % len(self.commands)])
        self.solved.append(np.asarray(solved, dtype=float))
        return SimpleNamespace(
            command=solved,
            warm_start=None,
            status="converged",
            used_fallback=False,
            deadline_met=None,
            diagnostics=SimpleNamespace(solve_time_s=0.003),
        )

    def summary(self):
        return dict(arm=self.name)


def _stationary_plant(state, command):
    """A plant that holds its state, so the loop is the only thing measured."""

    return SimpleNamespace(
        initial_state=np.asarray(state, dtype=float).copy(),
        initial_command=np.asarray(command, dtype=float).copy(),
        advance=lambda applied: np.asarray(state, dtype=float).copy(),
        source="stationary",
    )


def test_the_trial_applies_the_solved_command_and_records_its_solve_times(
    tmp_path, manifest
):
    """The solver's command is the plant's command, and the clock decides nothing.

    control-v4 computes the trajectory in simulated time: the solver is given
    no deadline, so nothing it returns is a fallback for want of time, and the
    command it solved is the command the plant is stepped with on every
    interval the model was ready. What a clock measured is still recorded, in
    timing.npz and in the row's wall block, and the trajectory carries none of
    it.
    """

    manifest = copy.deepcopy(manifest)
    manifest["trial"]["intervals"] = 8
    manifest["trial"]["duration_s"] = 8 * DT_S
    manifest["metrics"]["pass_criterion"]["settled_after_s"] = 0.1
    commands = np.array([[0.44, 0.01, -0.02], [0.46, -0.01, 0.03]])
    arm = _RecordingArm(commands)
    plant = _stationary_plant(TRIM, np.array([0.45, 0.0, 0.0]))

    def reference_fn(times):
        return harness.control_reference(TRIM, times, manifest["tracking_reference"])

    row = harness._control_trial(
        manifest, arm, plant, reference_fn, TRIM, tmp_path / "t"
    )

    # No solve was given a deadline, and the row says so from the solver's own
    # report rather than from the manifest.
    assert arm.keywords and all(keys == {"warm_start"} for keys in arm.keywords)
    wall = row["wall"]
    assert wall["solve_deadline_applied"] is False
    assert wall["deadline_assessed_intervals"] == 0
    assert "deadline_exceeded" not in row["solver_statuses"]
    assert row["fallback_count"] == 0

    # The solve-time statistics exist, are the ones the manifest calls
    # informational, and are measurements rather than zeroes.
    assert wall["maximum_solve_seconds"] == pytest.approx(0.003)
    assert wall["solve_deadline_s"] == manifest["trial"]["solve_deadline_s"]
    assert wall["solves_over_deadline"] == 0
    assert wall["intervals_over_sample_interval"] >= 0
    assert math.isfinite(wall["maximum_interval_seconds"])
    assert math.isfinite(wall["elapsed_seconds"])

    with np.load(tmp_path / "t" / "timing.npz", allow_pickle=False) as timing:
        assert len(timing["solve_times_s"]) == 8
        assert len(timing["tick_times_s"]) == 8
        assert not timing["deadline_assessed"].any()

    # The applied commands are the solved ones, interval for interval, and the
    # two warm-up intervals hold the initial command rather than inventing one.
    with np.load(tmp_path / "t" / "tracking.npz", allow_pickle=False) as tracking:
        applied = tracking["commands"]
        used = tracking["solver_used"]
        # Nothing a clock measured reaches the trajectory's own artifact.
        assert "solve_times_s" not in tracking
        assert "tick_times_s" not in tracking
        assert "source_clock_lags_s" not in tracking
    assert len(applied) == 8
    np.testing.assert_array_equal(used, [False, False] + [True] * 6)
    np.testing.assert_array_equal(applied[:2], np.tile(plant.initial_command, (2, 1)))
    np.testing.assert_array_equal(applied[2:], np.asarray(arm.solved))
    np.testing.assert_array_equal(applied, np.asarray(arm.applied))
    assert row["model_not_ready_intervals"] == 2


def test_the_replay_rejects_a_run_whose_solve_times_assessed_a_deadline(
    tmp_path, manifest, learned
):
    """A recorded deadline is a rejected run, not a reported measurement."""

    directory = tmp_path / "control-run"
    _fabricate_run(directory, manifest, learned, dict(generic=0.4, structured=0.5))
    harness.verify_control(directory, manifest, _anchor(directory))

    case = directory / "trial-0" / "generic"
    row = harness.read(case / "trial.json")
    row["wall"]["deadline_assessed_intervals"] = 1
    harness.write(case / "trial.json", row)
    rows = harness.read(directory / "results.json")
    rows[0] = row
    harness.write(directory / "results.json", rows)
    with pytest.raises(ValueError, match="a deadline was assessed"):
        harness.verify_control(directory, manifest, _anchor(directory))


def test_the_replay_rejects_a_run_that_reports_a_deadline_expiring(
    tmp_path, manifest, learned
):
    """A solver status is the other thing that can say a deadline decided."""

    directory = tmp_path / "control-run"
    _fabricate_run(directory, manifest, learned, dict(generic=0.4, structured=0.5))
    case = directory / "trial-0" / "generic"
    row = harness.read(case / "trial.json")
    row["solver_statuses"] = {"converged": 319, "deadline_exceeded": 1}
    harness.write(case / "trial.json", row)
    rows = harness.read(directory / "results.json")
    rows[0] = row
    harness.write(directory / "results.json", rows)
    with pytest.raises(ValueError, match="cut short by a deadline"):
        harness.verify_control(directory, manifest, _anchor(directory))


def test_the_replay_recomputes_the_recorded_solve_time_statistics(
    tmp_path, manifest, learned
):
    """The counts decide nothing, which is why the replay recomputes them."""

    directory = tmp_path / "control-run"
    _fabricate_run(directory, manifest, learned, dict(generic=0.4, structured=0.5))
    case = directory / "trial-0" / "generic"
    row = harness.read(case / "trial.json")
    row["wall"]["solves_over_deadline"] = 7
    harness.write(case / "trial.json", row)
    rows = harness.read(directory / "results.json")
    rows[0] = row
    harness.write(directory / "results.json", rows)
    with pytest.raises(ValueError, match="recomputed host measurements differ"):
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
    # The one constant control-v3 moved. Put it back and the tier's recording is
    # the example's, byte for byte: the pilot, its periods, the trim
    # feedforward, the additive excitation and the phases are unchanged.
    assert manifest["calibration"]["setpoint"]["airspeed_amplitude_m_s"] == 2.5
    manifest["calibration"]["setpoint"]["airspeed_amplitude_m_s"] = 0.4
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
def test_the_harness_reference_is_the_accuracy_pages_own_reference(monkeypatch):
    """The tier's rows are the page's reference, position and velocity."""

    pytest.importorskip("cascade")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    example = importlib.import_module("cascade_accuracy")

    manifest = harness.frozen_control_manifest(MANIFEST)
    with jax.enable_x64(True):
        _, _, _, state, _ = harness.control_fixture(manifest)
        times = np.arange(0, manifest["trial"]["duration_s"] + 0.05, 0.05)
        rows = harness.control_reference(state, times, manifest["tracking_reference"])
        position, velocity = example.reference(jnp.asarray(times))
    np.testing.assert_allclose(rows[:, 0:3], np.asarray(position), rtol=0, atol=1e-12)
    np.testing.assert_allclose(rows[:, 3:6], np.asarray(velocity), rtol=0, atol=1e-12)


@pytest.mark.cascade
def test_the_harness_initial_states_are_the_accuracy_pages_own(monkeypatch):
    pytest.importorskip("cascade")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    example = importlib.import_module("cascade_accuracy")

    manifest = harness.frozen_control_manifest(MANIFEST)
    with jax.enable_x64(True):
        _, _, _, state, _ = harness.control_fixture(manifest)
        for seed in manifest["trial"]["initial_state_seeds"]:
            np.testing.assert_allclose(
                harness.control_initial_state(manifest, state, seed),
                example.initial_condition(state, seed),
                rtol=0,
                atol=0,
            )


@pytest.mark.cascade
def test_the_committed_calibration_meets_the_declared_excitation():
    """Measured, not asserted: the three fitted recordings move every channel."""

    pytest.importorskip("cascade")
    manifest = harness.frozen_control_manifest(MANIFEST)
    with jax.enable_x64(True):
        spec, model, trim, state, command = harness.control_fixture(manifest)
        plant = harness._control_plant(manifest, spec, model)
        recordings = [
            (
                f"recording-{seed}",
                harness.control_recording(manifest, plant, trim, state, command, seed),
            )
            for seed in manifest["calibration"]["training_seeds"]
        ]
    measured = harness.control_excitation(manifest, recordings)
    assert (
        harness.control_excitation_shortfall(
            manifest, measured, [name for name, _ in recordings]
        )
        == []
    )
    for fractions in measured["fraction"].values():
        assert min(fractions) >= 0.1


@pytest.mark.cascade
def test_the_generic_arm_tracks_the_cascade_plant_for_a_short_trial(tmp_path):
    """One short paced trial end to end: the seam, the plant, and the metrics."""

    pytest.importorskip("cascade")
    manifest = copy.deepcopy(harness.frozen_control_manifest(MANIFEST))
    manifest["calibration"]["duration_s"] = 3.0
    manifest["trial"]["intervals"] = 20
    manifest["trial"]["duration_s"] = 1.0
    manifest["metrics"]["pass_criterion"]["settled_after_s"] = 0.5

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
    assert arm.summary()["uncertainty_available"] is True
    assert arm.summary()["uncertainty_complete"] is False
    assert arm.summary()["envelope_nominal_coverage"] == 0.9
    assert arm.summary()["maximum_tangent_standard_deviation"] > 0.0
    assert arm.summary()["horizon_s"] == pytest.approx(0.25)

    def reference_fn(times):
        return harness.control_reference(state, times, manifest["tracking_reference"])

    harness._control_prewarm(arm, manifest, recordings[0][1], reference_fn)
    start = harness.control_initial_state(manifest, state, 101)
    tracking = harness._control_tracking_plant(manifest, start, command)
    row = harness._control_trial(
        manifest, arm, tracking, reference_fn, state, tmp_path / "t"
    )
    assert row["terminated"] is False
    assert row["pass_criterion"]["scored_samples"] == 11
    assert row["completed_intervals"] == 20
    assert row["model_not_ready_intervals"] == arm.controller.required_observations - 1
    assert row["maximum_command_bound_violation"] == 0.0
    assert math.isfinite(row["tracking_rmse"]["position_rmse_m"])
