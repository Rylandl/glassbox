from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from glassbox.data import (
    RIGID_BODY_STATE_SCHEMA,
    ControlChannel,
    TrajectorySpec,
    VehicleConfigurationSpec,
)
from glassbox.dynamics import (
    fixed_wing_trim_control,
    hover_control,
    initial_residual_parameters,
)
from glassbox.fixedwing_synthetic import (
    TRIM_AIRSPEED_M_S,
    fixed_wing_trim_state,
    generate_fixed_wing_trajectory,
    true_fixed_wing_parameters,
)
from glassbox.nmpc import (
    NMPCController,
    NMPCWarmStart,
    SafetyEnvelope,
    SolveStatus,
    quaternion_log_error,
)
from glassbox.nmpc.solver import _SolverPolicy
from glassbox.runtime import (
    DirectActuationMap,
    ModelValidityEnvelope,
    RuntimeDynamicsModel,
    RuntimeModelSpec,
)
from glassbox.synthetic import (
    generate_trajectory,
    resting_state,
    true_parameters,
)


def _runtime_spec(dt_s: float) -> RuntimeModelSpec:
    return RuntimeModelSpec(
        sample_period_s=dt_s,
        validity_envelope=ModelValidityEnvelope(
            body_velocity_center_m_s=(0.0, 0.0, 0.0),
            body_velocity_half_width_m_s=(100.0, 100.0, 100.0),
            angular_velocity_center_rad_s=(0.0, 0.0, 0.0),
            angular_velocity_half_width_rad_s=(100.0, 100.0, 100.0),
        ),
    )


def _multirotor_runtime(*, residual: bool = False) -> RuntimeDynamicsModel:
    trajectory = generate_trajectory(seed=0, duration_s=0.1)
    params = true_parameters()
    if residual:
        params = initial_residual_parameters(params, hidden_units=3)
    return RuntimeDynamicsModel(
        params,
        trajectory.spec,
        _runtime_spec(trajectory.nominal_dt_s),
        DirectActuationMap(trajectory.spec.controls),
    )


def _fixed_wing_runtime() -> RuntimeDynamicsModel:
    trajectory = generate_fixed_wing_trajectory(seed=0, duration_s=0.1)
    return RuntimeDynamicsModel(
        true_fixed_wing_parameters(),
        trajectory.spec,
        _runtime_spec(trajectory.nominal_dt_s),
        DirectActuationMap(trajectory.spec.controls),
    )


def _flying_wing_runtime() -> RuntimeDynamicsModel:
    controls = (
        ControlChannel(
            "propulsion_command",
            "throttle",
            "normalized_command",
            "1",
            0.0,
            1.0,
        ),
        ControlChannel(
            "elevon_roll_command",
            "roll",
            "normalized_generalized_command",
            "1",
            -1.0,
            1.0,
            "FLU",
        ),
        ControlChannel(
            "elevon_pitch_command",
            "pitch",
            "normalized_generalized_command",
            "1",
            -1.0,
            1.0,
            "FLU",
        ),
    )
    spec = TrajectorySpec(
        state_schema=RIGID_BODY_STATE_SCHEMA,
        observation_source="simulator_truth",
        controls=controls,
        vehicle=VehicleConfigurationSpec(
            family="fixedwing",
            configuration_id="synthetic_flying_wing",
            controlled_axes=("roll", "pitch"),
            propulsion="single_propeller",
        ),
    )
    return RuntimeDynamicsModel(
        true_fixed_wing_parameters(),
        spec,
        _runtime_spec(0.02),
        DirectActuationMap(spec.controls),
    )


def _test_policy(*, horizon_steps: int = 6) -> _SolverPolicy:
    return _SolverPolicy(
        horizon_steps=horizon_steps,
        block_count=3,
        maximum_iterations=4,
        line_search_steps=5,
    )


def test_quaternion_error_is_sign_invariant_and_has_finite_identity_gradient() -> None:
    reference = jnp.asarray([1.0, 0.0, 0.0, 0.0])
    actual = jnp.asarray([math.cos(0.2), math.sin(0.2), 0.0, 0.0])

    error = quaternion_log_error(reference, actual)
    negated_error = quaternion_log_error(-reference, actual)
    identity_gradient = jax.jacrev(quaternion_log_error, argnums=1)(
        reference, reference
    )

    np.testing.assert_allclose(error, [0.4, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(error, negated_error, atol=1e-6)
    assert np.all(np.isfinite(identity_gradient))


def test_control_blocks_expand_and_commands_remain_bounded() -> None:
    model = _multirotor_runtime()
    controller = NMPCController(model, _policy=_test_policy(horizon_steps=5))
    blocks = jnp.asarray(
        [
            [-1.0] * 4,
            [0.0] * 4,
            [1.0] * 4,
        ]
    )

    expanded = controller._backend._expand_normalized_blocks(blocks)
    commands = controller._backend._commands_from_normalized(expanded)

    np.testing.assert_allclose(expanded[:, 0], [-1.0, -1.0, 0.0, 0.0, 1.0])
    assert np.min(commands) >= 0.0
    assert np.max(commands) <= 1.0


def test_solver_propagates_latent_state_and_returns_bounded_plan() -> None:
    model = _multirotor_runtime()
    controller = NMPCController(model, _policy=_test_policy())
    target = resting_state()
    state = target.copy()
    state[2] = -0.2
    previous = hover_control(true_parameters())

    result = controller.solve(
        jnp.asarray(state),
        controller.hold_reference(jnp.asarray(target)),
        previous,
    )

    assert result.status in {SolveStatus.CONVERGED, SolveStatus.ITERATION_LIMIT}
    assert not result.used_fallback
    assert result.predicted_states.shape == (controller.prediction_steps + 1, 13)
    assert result.predicted_latent_states.shape == (
        controller.prediction_steps + 1,
        model.latent_size,
    )
    assert result.diagnostics.maximum_command_bound_violation <= 1e-6
    assert np.all(np.isfinite(result.predicted_states))


def test_applied_command_initializes_latent_actuator_state() -> None:
    model = _multirotor_runtime()
    controller = NMPCController(model, _policy=_test_policy())
    target = resting_state()
    previous = hover_control(true_parameters())
    applied = jnp.clip(previous - 0.05, 0.0, 1.0)

    result = controller.solve(
        jnp.asarray(target),
        controller.hold_reference(jnp.asarray(target)),
        previous,
        applied_command=applied,
    )

    np.testing.assert_allclose(
        result.predicted_latent_states[0],
        model.initial_latent_state(applied),
        atol=1e-6,
    )


def test_warm_start_is_selected_only_when_no_worse_than_cold_start() -> None:
    model = _multirotor_runtime()
    controller = NMPCController(model, _policy=_test_policy())
    target = resting_state()
    state = target.copy()
    state[0] = 0.3
    previous = hover_control(true_parameters())
    reference = controller.hold_reference(jnp.asarray(target))
    first = controller.solve(jnp.asarray(state), reference, previous)

    cold = controller.solve(jnp.asarray(state), reference, previous)
    warm = controller.solve(
        jnp.asarray(state),
        reference,
        previous,
        warm_start=first.warm_start,
    )

    assert warm.diagnostics.initial_objective <= (
        cold.diagnostics.initial_objective + 1e-6
    )


def test_invalid_estimate_and_deadline_return_bounded_fallback() -> None:
    model = _multirotor_runtime()
    controller = NMPCController(model, _policy=_test_policy())
    target = resting_state()
    previous = hover_control(true_parameters())
    invalid = target.copy()
    invalid[3] = np.nan

    invalid_result = controller.solve(
        jnp.asarray(invalid),
        controller.hold_reference(jnp.asarray(target)),
        previous,
    )
    deadline_result = controller.solve(
        jnp.asarray(target),
        controller.hold_reference(jnp.asarray(target)),
        previous,
        deadline_s=1e-12,
    )

    assert invalid_result.status is SolveStatus.INVALID_INPUT
    assert deadline_result.status is SolveStatus.DEADLINE_EXCEEDED
    for result in (invalid_result, deadline_result):
        assert result.used_fallback
        assert np.all(np.isfinite(result.command))
        assert np.min(result.command) >= np.min(model.command_minimum)
        assert np.max(result.command) <= np.max(model.command_maximum)


def test_safety_envelope_reports_normalized_prediction_violation() -> None:
    model = _multirotor_runtime()
    controller = NMPCController(
        model,
        safety_envelope=SafetyEnvelope(maximum_position_m=(0.1, 0.1, 0.1)),
        _policy=_test_policy(),
    )
    state = resting_state()
    state[0] = 0.3
    previous = hover_control(true_parameters())

    result = controller.solve(
        jnp.asarray(state),
        controller.hold_reference(jnp.asarray(resting_state())),
        previous,
    )

    assert result.diagnostics.maximum_normalized_safety_violation > 0.0


@pytest.mark.parametrize("model_kind", ["multirotor", "fixedwing", "residual"])
def test_objective_gradient_agrees_with_central_difference(model_kind: str) -> None:
    if model_kind == "fixedwing":
        model = _fixed_wing_runtime()
        state = fixed_wing_trim_state()
        state[2] = -2.0
        state[4] = 1.0
        target = fixed_wing_trim_state()
        previous = fixed_wing_trim_control(
            true_fixed_wing_parameters(), TRIM_AIRSPEED_M_S
        )
    else:
        model = _multirotor_runtime(residual=model_kind == "residual")
        state = resting_state()
        state[0] = 0.1
        target = resting_state()
        previous = hover_control(true_parameters())
    controller = NMPCController(model, _policy=_test_policy(horizon_steps=4))
    reference = controller.hold_reference(jnp.asarray(target))
    latent = model.initial_latent_state(previous)
    exogenous = jnp.zeros((controller.prediction_steps, model.exogenous_size))
    blocks = controller._backend._cold_blocks(previous) + jnp.asarray(
        [
            [0.10, 0.08, -0.06, 0.05],
            [0.02, -0.10, 0.07, -0.04],
            [-0.05, 0.03, 0.04, 0.02],
        ]
    )
    objective = jax.jit(controller._backend._objective)
    _, analytic = controller._backend._objective_and_gradient(
        blocks,
        jnp.asarray(state),
        latent,
        reference.states,
        previous,
        exogenous,
    )
    epsilon = 2e-3
    finite_difference = np.empty(blocks.shape)
    for index in np.ndindex(*blocks.shape):
        direction = jnp.zeros_like(blocks).at[index].set(epsilon)
        plus = objective(
            blocks + direction,
            jnp.asarray(state),
            latent,
            reference.states,
            previous,
            exogenous,
        )
        minus = objective(
            blocks - direction,
            jnp.asarray(state),
            latent,
            reference.states,
            previous,
            exogenous,
        )
        finite_difference[index] = float((plus - minus) / (2.0 * epsilon))

    analytic_array = np.asarray(analytic)
    relative_error = np.linalg.norm(analytic_array - finite_difference) / max(
        np.linalg.norm(analytic_array),
        np.linalg.norm(finite_difference),
        1e-6,
    )
    assert np.all(np.isfinite(analytic_array))
    assert relative_error <= 2e-3


def test_incompatible_warm_start_is_safely_ignored() -> None:
    model = _fixed_wing_runtime()
    controller = NMPCController(model, _policy=_test_policy())
    state = fixed_wing_trim_state()
    previous = fixed_wing_trim_control(true_fixed_wing_parameters(), TRIM_AIRSPEED_M_S)
    warm_start = NMPCWarmStart(jnp.zeros((2, 2)))

    result = controller.solve(
        jnp.asarray(state),
        controller.hold_reference(jnp.asarray(state)),
        previous,
        warm_start=warm_start,
    )

    assert not result.diagnostics.warm_start_used


def test_fixedwing_generalized_roles_support_flying_wing_command_names() -> None:
    model = _flying_wing_runtime()
    controller = NMPCController(model, _policy=_test_policy())
    state = fixed_wing_trim_state()
    previous = fixed_wing_trim_control(
        true_fixed_wing_parameters(),
        TRIM_AIRSPEED_M_S,
        model.input_spec.control_roles,
    )

    result = controller.solve(
        jnp.asarray(state),
        controller.hold_reference(jnp.asarray(state)),
        previous,
    )

    assert result.command.shape == (3,)
    assert not result.used_fallback
