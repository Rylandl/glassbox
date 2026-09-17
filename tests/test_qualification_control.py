"""Qualification seams tested without fitting or simulator installation."""

import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from glassbox.control.plan import ReferenceTrajectory
from glassbox.experimental import qualification_control as qualification
from glassbox.experimental.default_model import RECIPE
from glassbox.experimental.harness import control_reference
from glassbox.experimental.learned_plan import observed_from_state, states_from_observed

ROOT = Path(__file__).resolve().parents[1]
CONTROL = json.loads((ROOT / "docs/harness/control-v5.json").read_text())
FROZEN = json.loads(
    (ROOT / "docs/harness/evaluation-qualification-v1.json").read_text()
)
INITIAL = np.array([0, 0, 100, 18, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=float)
COMMAND = np.array([0.5, 0.0, 0.0])


class Equations:
    """A differentiable public-state stand-in; no model is fitted."""

    def __init__(self):
        self.compile_signature = "qualification-test-equations-v1"

        def advance(state, command):
            velocity = state[3:6] + 0.05 * command
            position = state[:3] + 0.025 * (state[3:6] + velocity)
            return state.at[:3].set(position).at[3:6].set(velocity)

        def predict(state, commands):
            def step(current, command):
                future = advance(current, command)
                return future, observed_from_state(future)

            return jax.lax.scan(step, state, commands)[1]

        self.advance = jax.jit(advance)
        self.predict = jax.jit(predict)
        self.reset = lambda state, command: jnp.asarray(state)
        self.canonical = lambda state: state


@pytest.fixture
def learned():
    """Only the public plan construction metadata; the mean is never invoked."""
    model = SimpleNamespace(
        params={"memory": np.zeros((4, 8))},
        norms={},
        delay_steps=2,
        metadata=lambda: {"fixture": "qualification"},
        arrays=lambda: {"memory": np.zeros((4, 8))},
        memory_state=lambda states, commands: jnp.zeros(8),
    )
    return SimpleNamespace(
        _model=model,
        contract={
            "dt_s": 0.05,
            "state_channels": CONTROL["telemetry"]["observed_channels"],
            "input_channels": ["throttle", "roll", "pitch"],
        },
        recipe=dict(RECIPE),
        history_steps=10,
        horizon_steps=5,
        envelope=lambda steps: np.full((steps, 15), 0.2),
    )


def arm_for(learned):
    return qualification.OracleArm(CONTROL, learned, None, _equations=Equations())


def test_constants_match_frozen_qualification():
    frozen = FROZEN["control_qualification"]
    assert qualification.HORIZON_STEPS == frozen["horizon_steps"]
    assert qualification.BLOCK_COUNT == frozen["block_count"]
    assert qualification.MAXIMUM_ITERATIONS == frozen["maximum_iterations"]
    assert qualification.WARMUP_INTERVALS == frozen["warmup_intervals"]


def test_oracle_preserves_generic_map_cost_and_covariance(learned):
    arm = arm_for(learned)
    plan = arm.plan.with_causal_state(jnp.asarray(INITIAL))
    candidate = jnp.tile(jnp.asarray(COMMAND), (5, 1))
    predicted = plan.rollout_commands(
        candidate, jnp.asarray(INITIAL), None, None, plan.values
    )
    expected = states_from_observed(
        jnp.asarray(INITIAL),
        arm.equations.predict(jnp.asarray(INITIAL), candidate),
        0.05,
    )
    np.testing.assert_array_equal(predicted.mean_states, expected)
    assert predicted.tangent_covariance is plan.base.values.forecast_error_covariance
    assert plan.stage_cost == plan.base.stage_cost
    assert plan.measure == plan.base.measure
    assert plan.policy == plan.base.policy
    assert plan.compile_signature != plan.base.compile_signature


def test_oracle_mean_and_existing_objective_have_finite_command_gradients(learned):
    arm = arm_for(learned)
    plan = arm.plan.with_causal_state(jnp.asarray(INITIAL))
    reference = jnp.asarray(
        control_reference(INITIAL, np.arange(6) * 0.05, CONTROL["tracking_reference"])
    )

    def objective(blocks):
        prediction = plan.rollout(blocks, jnp.asarray(INITIAL), None, None, plan.values)
        return plan.stage_cost(prediction, reference, jnp.asarray(COMMAND), plan.policy)

    gradient = np.asarray(jax.grad(objective)(jnp.zeros((5, 3))))
    assert np.isfinite(gradient).all()
    assert np.max(np.abs(gradient)) > 0
    epsilon = 1e-3
    direction = np.zeros((5, 3))
    direction[0, 1] = epsilon
    difference = float(
        (objective(jnp.asarray(direction)) - objective(jnp.asarray(-direction)))
        / (2 * epsilon)
    )
    np.testing.assert_allclose(gradient[0, 1], difference, rtol=0.02, atol=2e-5)


def test_oracle_requires_real_history_and_causal_observations(learned):
    arm = arm_for(learned)
    arm.reset(INITIAL, COMMAND)
    state = jnp.asarray(INITIAL)
    for index in range(3):
        arm.observe(state)
        assert arm.ready is (index == 2)
        with pytest.raises(ValueError, match="alternate"):
            arm.observe(state)
        arm.command_applied(COMMAND)
        state = arm.equations.advance(state, jnp.asarray(COMMAND))
    with pytest.raises(AssertionError):
        arm.observe(np.asarray(state) + 1)


def short_trial(learned):
    arm = arm_for(learned)
    arm.reset(INITIAL, COMMAND)
    states, commands = [INITIAL], []
    previous, state, warm_start = COMMAND, INITIAL, None
    for index in range(4):
        arm.observe(state)
        if arm.ready:
            reference = ReferenceTrajectory(
                control_reference(
                    INITIAL,
                    (index + np.arange(6)) * 0.05,
                    CONTROL["tracking_reference"],
                )
            )
            result = arm.solve(state, reference, previous, warm_start=warm_start)
            command = np.asarray(result.command)
            warm_start = result.warm_start
        else:
            command = previous
        state = np.asarray(
            arm.equations.advance(jnp.asarray(state), jnp.asarray(command))
        )
        arm.command_applied(command)
        commands.append(command)
        states.append(state)
        previous = command
    return arm, dict(
        states=np.asarray(states),
        commands=np.asarray(commands),
        initial_command=COMMAND,
        reference_anchor_state=INITIAL,
    )


def test_saved_oracle_forecasts_and_causal_replay_verify(learned):
    arm, tracking = short_trial(learned)
    report = qualification.verify_oracle_diagnostics(
        CONTROL,
        learned,
        None,
        tracking,
        arm.diagnostic_arrays(),
        FROZEN["control_qualification"]["replay"],
        _equations=arm.equations,
    )
    assert report["checked_forecasts"] == 2
    assert report["checked_causal_states"] == 5
    assert (
        report["maximum_forecast_difference"]
        < FROZEN["control_qualification"]["replay"]["state_atol"]
    )


@pytest.mark.parametrize(
    "array",
    ["candidate_commands", "forecast_states", "causal_states", "final_objectives"],
)
def test_oracle_replay_rejects_altered_diagnostics(learned, array):
    arm, tracking = short_trial(learned)
    diagnostics = arm.diagnostic_arrays()
    diagnostics[array] = np.array(diagnostics[array], copy=True)
    diagnostics[array].flat[0] += 0.1
    with pytest.raises((AssertionError, ValueError)):
        qualification.verify_oracle_diagnostics(
            CONTROL,
            learned,
            None,
            tracking,
            diagnostics,
            FROZEN["control_qualification"]["replay"],
            _equations=arm.equations,
        )


def test_matched_structured_retains_adapter_and_matches_only_frozen_changes(
    monkeypatch,
):
    import glassbox.control.fitted as fitted
    from glassbox.control.plan import SolverPolicy

    original = SolverPolicy(horizon_steps=16, block_count=8, maximum_iterations=8)
    model = object()
    belief = SimpleNamespace(model=model)
    called = {}

    def controller(saved_model, *, policy):
        called.update(model=saved_model, policy=policy)
        return SimpleNamespace(plan=SimpleNamespace(policy=policy))

    monkeypatch.setattr(fitted, "default_solver_policy", lambda saved: original)
    monkeypatch.setattr(fitted, "NMPCController", controller)
    arm = qualification.MatchedStructuredArm(CONTROL, belief)
    assert called["model"] is model
    assert arm.policy.horizon_steps == arm.policy.block_count == 5
    assert arm.policy.maximum_iterations == 4
    assert arm.policy.initial_step_size == original.initial_step_size
    arm.reset(INITIAL, COMMAND)
    for index in range(3):
        arm.observe(INITIAL)
        assert arm.ready is (index == 2)
        arm.command_applied(COMMAND)
    np.testing.assert_array_equal(np.asarray(arm._history)[-1], COMMAND)


@pytest.mark.cascade
def test_public_cascade_replay_forecast_and_command_gradient(learned):
    pytest.importorskip("cascade")
    from glassbox.experimental.harness import control_fixture
    from glassbox.integrations.cascade import CascadePlant, CascadePlantConfig

    spec, model, _trim, initial_state, initial_command = control_fixture(CONTROL)
    plant = CascadePlant(
        CascadePlantConfig(control_frequency_hz=20), spec=spec, model=model
    )
    sample = plant.reset(initial_state, applied_control=initial_command)
    arm = qualification.OracleArm(CONTROL, learned, model)
    arm.reset(sample.state, initial_command)
    issued = initial_command + np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.02, 0.01, -0.01], [-0.01, -0.02, 0.01]]
    )
    for command in issued:
        arm.observe(sample.state)
        arm.command_applied(command)
        sample = plant.step(command)
    np.testing.assert_allclose(
        arm.equations.canonical(arm._state), sample.state, rtol=1e-5, atol=1e-5
    )
    origin_state = sample.state
    candidate = jnp.tile(jnp.asarray(initial_command), (5, 1))
    plan = arm.plan.with_causal_state(arm._state)

    def predict(commands):
        return plan.rollout_commands(
            commands,
            jnp.asarray(origin_state),
            None,
            jnp.zeros((5, 0)),
            plan.values,
        ).mean_states

    gradient = np.asarray(jax.jacfwd(predict)(candidate))
    assert np.isfinite(gradient).all()
    assert np.max(np.abs(gradient)) > 0
    epsilon = 1e-3
    direction = jnp.zeros_like(candidate).at[0, 1].set(epsilon)
    difference = np.asarray(
        (predict(candidate + direction) - predict(candidate - direction))
        / (2 * epsilon)
    )
    np.testing.assert_allclose(gradient[:, :, 0, 1], difference, rtol=0.02, atol=5e-3)
    future_observations = []
    for command in np.asarray(candidate):
        future_observations.append(
            np.asarray(observed_from_state(plant.step(command).state))
        )
    expected = states_from_observed(
        jnp.asarray(origin_state),
        jnp.asarray(future_observations),
        0.05,
    )
    np.testing.assert_allclose(predict(candidate), expected, rtol=1e-5, atol=1e-5)
