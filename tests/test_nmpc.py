from __future__ import annotations

import itertools
import math
from collections.abc import Callable
from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import glassbox.control.nmpc.solver as nmpc_solver
from glassbox.belief.belief import (
    DynamicsBelief,
    EmpiricalErrorSample,
    EmpiricalHorizonPredictiveError,
    LocalGaussianParameterBelief,
    structured_parameter_names,
    structured_parameter_vector,
)
from glassbox.control.nmpc import (
    NMPCController,
    NMPCWarmStart,
    SafetyEnvelope,
    SolveStatus,
    SupportFilterMode,
    quaternion_log_error,
)
from glassbox.control.nmpc.solver import (
    _block_steps_for,
    _blocks_cover_horizon,
    _maintained_block_count,
    _projected_gradient_norm,
    _SolverPolicy,
)
from glassbox.core.data import (
    RIGID_BODY_STATE_SCHEMA,
    ControlChannel,
    ExogenousChannel,
    TrajectorySpec,
    VehicleConfigurationSpec,
)
from glassbox.core.dynamics import (
    MOTOR_MIXER,
    fixed_wing_trim_control,
    hover_control,
    initial_residual_parameters,
)
from glassbox.core.fixedwing_synthetic import (
    TRIM_AIRSPEED_M_S,
    fixed_wing_trim_state,
    generate_fixed_wing_trajectory,
    true_fixed_wing_parameters,
)
from glassbox.core.runtime import (
    DirectActuationMap,
    ModelValidityEnvelope,
    RuntimeDynamicsModel,
    RuntimeModelSpec,
)
from glassbox.core.synthetic import (
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


def _multirotor_runtime(
    *, residual: bool = False, sample_period_s: float | None = None
) -> RuntimeDynamicsModel:
    trajectory = generate_trajectory(seed=0, duration_s=0.1)
    params = true_parameters()
    if residual:
        params = initial_residual_parameters(params, hidden_units=3)
    return RuntimeDynamicsModel(
        params,
        trajectory.spec,
        _runtime_spec(
            trajectory.nominal_dt_s if sample_period_s is None else sample_period_s
        ),
        DirectActuationMap(trajectory.spec.controls),
    )


def _narrow_rate_envelope_multirotor() -> RuntimeDynamicsModel:
    model = _multirotor_runtime()
    runtime_spec = replace(
        model.runtime_spec,
        validity_envelope=ModelValidityEnvelope(
            body_velocity_center_m_s=(0.0, 0.0, 0.0),
            body_velocity_half_width_m_s=(10.0, 10.0, 10.0),
            angular_velocity_center_rad_s=(0.0, 0.0, 0.0),
            angular_velocity_half_width_rad_s=(0.2, 0.2, 0.2),
        ),
    )
    return RuntimeDynamicsModel(
        model.params,
        model.input_spec,
        runtime_spec,
        model.actuation,
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
    block_count = max(
        count for count in range(1, 4) if _blocks_cover_horizon(horizon_steps, count)
    )
    return _SolverPolicy(
        horizon_steps=horizon_steps,
        block_count=block_count,
        maximum_iterations=4,
        line_search_steps=5,
    )


def _scripted_perf_counter(readings: tuple[float, ...]) -> Callable[[], float]:
    """Return a ``perf_counter`` stand-in that never runs out of readings.

    The scripted readings are handed out in order and every later call keeps
    advancing by a millisecond.  A solve that reads the clock one more time
    than the script anticipates therefore still runs past the deadline and
    fails the assertion under test, instead of raising ``StopIteration`` out of
    an exhausted iterator and hiding what changed.
    """

    scripted = iter(readings)
    overflow = itertools.count(1)

    def perf_counter() -> float:
        try:
            return next(scripted)
        except StopIteration:
            return readings[-1] + 0.001 * next(overflow)

    return perf_counter


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


def test_maintained_block_layout_covers_every_horizon_without_dead_blocks() -> None:
    for horizon_steps in range(1, 61):
        block_count = _maintained_block_count(horizon_steps)
        block_steps = _block_steps_for(horizon_steps, block_count)
        expanded = np.repeat(np.arange(block_count), block_steps)[:horizon_steps]

        assert 1 <= block_count <= 10
        assert len(expanded) == horizon_steps
        assert sorted(set(expanded.tolist())) == list(range(block_count))
        # Every horizon with a usable divisor is covered exactly, so no block
        # is ever held for fewer steps than its neighbours.
        if horizon_steps % block_count == 0:
            assert block_steps * block_count == horizon_steps
        else:
            # Only a prime horizon longer than the block cap has to truncate
            # its final block, and it still uses more than a single block.
            assert horizon_steps > 10 and block_count > 1
            assert all(horizon_steps % divisor for divisor in range(2, 11))


def test_default_multirotor_block_layout_has_no_dead_blocks_at_fifty_hertz() -> None:
    controller = NMPCController(_multirotor_runtime(sample_period_s=0.05))
    backend = controller._backend
    blocks = jnp.repeat(
        jnp.linspace(-1.0, 1.0, backend.command_block_count)[:, None],
        controller.model.command_size,
        axis=1,
    )

    expanded = np.asarray(backend._expand_normalized_blocks(blocks))

    assert controller.prediction_steps == 12
    assert backend.command_block_count == 6
    assert backend._block_steps * backend.command_block_count == 12
    assert expanded.shape == (12, controller.model.command_size)
    assert len(np.unique(expanded[:, 0])) == backend.command_block_count


def test_solver_policy_rejects_a_layout_with_dead_command_blocks() -> None:
    with pytest.raises(ValueError, match="drive no prediction step"):
        _SolverPolicy(horizon_steps=4, block_count=3)


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
    assert controller._backend._support_batch_warmed


def test_solver_consumes_predictive_and_parameter_uncertainty() -> None:
    model = _multirotor_runtime()
    parameter_count = len(structured_parameter_vector(model.params))
    parameter_covariance = np.zeros((parameter_count, parameter_count))
    parameter_covariance[0, 0] = 0.04
    endpoint_errors = 0.02 * np.concatenate((np.eye(12), -np.eye(12)))
    error_samples = (
        EmpiricalErrorSample(endpoint_errors, "group-a", "flight-a"),
        EmpiricalErrorSample(endpoint_errors, "group-b", "flight-b"),
    )
    belief = DynamicsBelief(
        params=model.params,
        input_spec=model.input_spec,
        runtime_spec=model.runtime_spec,
        predictive_error=EmpiricalHorizonPredictiveError.from_samples(
            {0.1: error_samples, 0.2: error_samples}
        ),
        parameter_belief=LocalGaussianParameterBelief(
            parameter_names=structured_parameter_names(model.params),
            covariance=parameter_covariance,
            source="fleet_prior",
            evidence_count=4,
            effective_sample_count=4.0,
        ),
    )
    controller = NMPCController(belief, _policy=_test_policy(horizon_steps=4))
    target = resting_state()
    previous = hover_control(true_parameters())

    result = controller.solve(
        jnp.asarray(target),
        controller.hold_reference(jnp.asarray(target)),
        previous,
    )

    assert result.diagnostics.model_uncertainty_available
    assert result.diagnostics.prediction_error_model_available
    assert result.diagnostics.prediction_error_model_current
    assert result.diagnostics.parameter_uncertainty_available
    assert (
        result.diagnostics.maximum_normalized_model_uncertainty_standard_deviation > 0.0
    )
    assert result.diagnostics.uncertainty_aware_command_selection


def test_large_model_uncertainty_bounds_command_authority() -> None:
    model = _multirotor_runtime()
    endpoint_errors = 2.0 * np.concatenate((np.eye(12), -np.eye(12)))
    error_samples = (
        EmpiricalErrorSample(endpoint_errors, "group-a", "flight-a"),
        EmpiricalErrorSample(endpoint_errors, "group-b", "flight-b"),
    )
    belief = DynamicsBelief(
        params=model.params,
        input_spec=model.input_spec,
        runtime_spec=model.runtime_spec,
        predictive_error=EmpiricalHorizonPredictiveError.from_samples(
            {0.1: error_samples, 0.2: error_samples}
        ),
    )
    policy = _test_policy(horizon_steps=4)
    nominal_controller = NMPCController(model, _policy=policy)
    uncertain_controller = NMPCController(belief, _policy=policy)
    target = resting_state()
    state = target.copy()
    state[2] = -0.2
    previous = hover_control(true_parameters())

    nominal = nominal_controller.solve(
        jnp.asarray(state),
        nominal_controller.hold_reference(jnp.asarray(target)),
        previous,
    )
    uncertain = uncertain_controller.solve(
        jnp.asarray(state),
        uncertain_controller.hold_reference(jnp.asarray(target)),
        previous,
    )

    authority = uncertain.diagnostics.command_authority_fraction
    assert 0.0 < authority < 1.0
    assert uncertain.diagnostics.uncertainty_aware_command_selection
    np.testing.assert_allclose(
        uncertain.command - previous,
        authority * (nominal.command - previous),
        atol=1e-5,
    )


def test_default_horizon_does_not_exceed_predictive_error_evidence() -> None:
    model = _multirotor_runtime()
    endpoint_errors = 0.02 * np.concatenate((np.eye(12), -np.eye(12)))
    samples = (
        EmpiricalErrorSample(endpoint_errors, "group-a", "flight-a"),
        EmpiricalErrorSample(endpoint_errors, "group-b", "flight-b"),
    )
    belief = DynamicsBelief(
        params=model.params,
        input_spec=model.input_spec,
        runtime_spec=model.runtime_spec,
        predictive_error=EmpiricalHorizonPredictiveError.from_samples({0.1: samples}),
    )

    controller = NMPCController(belief)

    assert controller.prediction_horizon_s == pytest.approx(0.1)


def test_default_multirotor_horizon_snaps_near_integer_sample_ratio() -> None:
    controller = NMPCController(_multirotor_runtime())

    assert controller.prediction_steps == 30
    assert controller.prediction_horizon_s == pytest.approx(0.6)


def test_stale_predictive_error_does_not_cap_default_horizon() -> None:
    model = _multirotor_runtime()
    endpoint_errors = 0.02 * np.concatenate((np.eye(12), -np.eye(12)))
    samples = (
        EmpiricalErrorSample(endpoint_errors, "group-a", "flight-a"),
        EmpiricalErrorSample(endpoint_errors, "group-b", "flight-b"),
    )
    parameter_count = len(structured_parameter_names(model.params))
    stale_belief = DynamicsBelief(
        params=model.params,
        input_spec=model.input_spec,
        runtime_spec=model.runtime_spec,
        predictive_error=EmpiricalHorizonPredictiveError.from_samples({0.1: samples}),
        parameter_belief=LocalGaussianParameterBelief(
            parameter_names=structured_parameter_names(model.params),
            covariance=np.eye(parameter_count),
            source="updated parameter belief",
            evidence_count=1,
            effective_sample_count=1.0,
            update_count=1,
        ),
        predictive_error_parameter_update_count=0,
    )

    controller = NMPCController(stale_belief)

    assert controller.belief.maximum_error_horizon_s is None
    assert controller.prediction_steps == 30
    assert controller.prediction_horizon_s == pytest.approx(0.6)


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


def test_warm_start_seed_advances_the_previous_plan_by_one_block() -> None:
    model = _multirotor_runtime()
    controller = NMPCController(model, _policy=_test_policy(horizon_steps=6))
    backend = controller._backend
    block_values = (0.2, 0.5, 0.8)
    previous_blocks = np.asarray(
        [[value] * model.command_size for value in block_values]
    )
    previous_plan = np.repeat(previous_blocks, backend._block_steps, axis=0)

    seed = backend._warm_blocks(NMPCWarmStart(jnp.asarray(previous_plan)))
    seed_commands = np.asarray(backend._commands_from_normalized(seed))

    assert backend._block_steps == 2
    np.testing.assert_allclose(seed_commands[0], previous_blocks[1], atol=1e-6)
    np.testing.assert_allclose(seed_commands[-1], previous_blocks[-1], atol=1e-6)
    np.testing.assert_allclose(
        seed_commands,
        previous_blocks[[1, 2, 2]],
        atol=1e-6,
    )
    # The seed must not simply reproduce the previous unshifted plan.
    assert np.max(np.abs(seed_commands - previous_blocks)) > 0.1


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


def test_previous_command_within_rounding_of_a_bound_is_accepted() -> None:
    model = _multirotor_runtime()
    controller = NMPCController(model, _policy=_test_policy())
    target = resting_state()
    reference = controller.hold_reference(jnp.asarray(target))
    maximum = np.asarray(model.command_maximum, dtype=np.float64)
    at_bound = maximum + 1e-9
    beyond_bound = maximum + 1e-3

    accepted = controller.solve(jnp.asarray(target), reference, at_bound)
    applied = controller.solve(
        jnp.asarray(target),
        reference,
        maximum,
        applied_command=at_bound,
    )
    rejected = controller.solve(jnp.asarray(target), reference, beyond_bound)

    for result in (accepted, applied):
        assert result.status is not SolveStatus.INVALID_INPUT
        assert not result.used_fallback
        assert np.all(np.asarray(result.command) <= maximum + 1e-6)
    assert rejected.status is SolveStatus.INVALID_INPUT
    assert rejected.message == "previous command lies outside the command bounds"
    assert np.all(np.asarray(rejected.command) <= maximum + 1e-6)


def test_deadline_includes_prediction_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _multirotor_runtime()
    controller = NMPCController(model, _policy=_test_policy())
    target = resting_state()
    previous = hover_control(true_parameters())
    reference = controller.hold_reference(jnp.asarray(target))
    controller.solve(jnp.asarray(target), reference, previous)

    monkeypatch.setattr(
        nmpc_solver,
        "time",
        SimpleNamespace(
            perf_counter=_scripted_perf_counter((0.0, 0.001, 0.002, 0.030, 0.031))
        ),
    )
    result = controller.solve(
        jnp.asarray(target),
        reference,
        previous,
        deadline_s=0.020,
    )

    assert result.status is SolveStatus.DEADLINE_EXCEEDED
    assert result.used_fallback
    np.testing.assert_allclose(result.command, previous)
    assert result.diagnostics.solve_time_s == pytest.approx(0.031)


def test_forced_line_search_failure_returns_bounded_fallback() -> None:
    model = _multirotor_runtime()
    policy = _SolverPolicy(
        horizon_steps=6,
        block_count=3,
        maximum_iterations=2,
        line_search_steps=1,
        armijo_fraction=1e6,
    )
    controller = NMPCController(model, _policy=policy)
    target = resting_state()
    state = target.copy()
    state[2] = -0.3
    previous = hover_control(true_parameters())

    result = controller.solve(
        jnp.asarray(state),
        controller.hold_reference(jnp.asarray(target)),
        previous,
    )

    assert result.status is SolveStatus.LINE_SEARCH_FAILED
    assert result.used_fallback
    np.testing.assert_allclose(result.command, previous, atol=1e-6)


def test_projected_gradient_measures_stationarity_inside_the_command_box() -> None:
    at_upper_bound = jnp.asarray([[1.0, 1.0]])
    interior = jnp.asarray([[0.0, 0.0]])

    outward = _projected_gradient_norm(at_upper_bound, jnp.asarray([[-5.0, -5.0]]))
    inward = _projected_gradient_norm(at_upper_bound, jnp.asarray([[5.0, 5.0]]))
    unconstrained = _projected_gradient_norm(interior, jnp.asarray([[0.3, -0.4]]))

    # An outward gradient at an active bound offers no feasible descent, so the
    # raw infinity norm of 5.0 is not evidence of an unconverged solve.
    assert float(outward) == pytest.approx(0.0)
    assert float(inward) == pytest.approx(2.0)
    assert float(unconstrained) == pytest.approx(0.4)


def test_converged_status_requires_the_first_order_criterion() -> None:
    model = _multirotor_runtime()
    policy = _test_policy()
    controller = NMPCController(model, _policy=policy)
    target = resting_state()
    previous = hover_control(true_parameters())

    result = controller.solve(
        jnp.asarray(target),
        controller.hold_reference(jnp.asarray(target)),
        previous,
    )

    assert result.status is SolveStatus.CONVERGED
    assert result.command_usable
    assert result.diagnostics.final_projected_gradient_inf_norm <= (
        policy.gradient_tolerance
    )
    assert result.message == "first-order convergence criterion satisfied"


def test_improvement_stall_is_reported_as_stalled_rather_than_converged() -> None:
    model = _multirotor_runtime()
    policy = _SolverPolicy(
        horizon_steps=6,
        block_count=3,
        maximum_iterations=4,
        line_search_steps=5,
        gradient_tolerance=1e-9,
        relative_improvement_tolerance=1.0,
    )
    controller = NMPCController(model, _policy=policy)
    target = resting_state()
    state = target.copy()
    state[2] = -0.2
    previous = hover_control(true_parameters())

    result = controller.solve(
        jnp.asarray(state),
        controller.hold_reference(jnp.asarray(target)),
        previous,
    )

    assert result.status is SolveStatus.STALLED
    assert not result.used_fallback
    # A stall is exactly as usable as an iteration-limit plan.
    assert result.command_usable
    assert result.diagnostics.final_projected_gradient_inf_norm > (
        policy.gradient_tolerance
    )
    assert "converg" not in result.message
    assert "stalled" in result.message


def test_line_search_stall_after_progress_keeps_the_improved_plan() -> None:
    model = _multirotor_runtime()
    policy = _SolverPolicy(
        horizon_steps=6,
        block_count=3,
        maximum_iterations=12,
        line_search_steps=1,
        initial_step_size=0.5,
        armijo_fraction=0.5,
        gradient_tolerance=1e-9,
        relative_improvement_tolerance=1e-12,
    )
    controller = NMPCController(model, _policy=policy)
    target = resting_state()
    state = target.copy()
    state[0] = 0.3
    state[2] = -0.3
    previous = hover_control(true_parameters())

    result = controller.solve(
        jnp.asarray(state),
        controller.hold_reference(jnp.asarray(target)),
        previous,
    )

    assert result.status is SolveStatus.STALLED
    assert not result.used_fallback
    assert result.diagnostics.iterations >= 2
    assert result.diagnostics.final_objective < result.diagnostics.initial_objective
    assert "line search" in result.message
    assert not np.allclose(np.asarray(result.command), np.asarray(previous))


def test_support_candidates_are_vehicle_agnostic_nmpc_projections() -> None:
    model = _fixed_wing_runtime()
    controller = NMPCController(model, _policy=_test_policy(horizon_steps=4))
    previous = np.asarray(fixed_wing_trim_control(model.params, TRIM_AIRSPEED_M_S))
    nominal = np.clip(
        previous + np.asarray((0.1, 0.2, -0.2, 0.1)),
        np.asarray(model.command_minimum),
        np.asarray(model.command_maximum),
    )

    candidates, fractions = controller._backend._support_candidates(
        nominal,
        previous,
    )

    assert not hasattr(controller, "supervisor")
    assert fractions == [1.0, 0.75, 0.5, 0.25, 0.0]
    for candidate, fraction in zip(candidates, fractions):
        np.testing.assert_allclose(
            candidate,
            previous + fraction * (nominal - previous),
            atol=1e-7,
        )


def test_support_filter_keeps_next_step_inside_from_envelope_boundary() -> None:
    model = _narrow_rate_envelope_multirotor()
    controller = NMPCController(model, _policy=_test_policy(horizon_steps=4))
    state = resting_state()
    state[10] = 0.19
    previous = np.full(4, 0.5)
    nominal = np.clip(previous + 0.15 * np.asarray(MOTOR_MIXER[0]), 0.0, 1.0)
    decision = controller._backend._select_support_command(
        jnp.asarray(state),
        model.initial_latent_state(previous),
        jnp.asarray(nominal),
        jnp.asarray(previous),
        jnp.zeros(0),
    )

    assert decision.mode is SupportFilterMode.BOUNDARY_FILTERED
    assert decision.applied
    assert decision.nominal_fraction < 1.0
    assert decision.current_validity <= 1.0
    assert decision.support_horizon_maximum_robust_validity <= 1.0 + 1e-6


def test_support_filter_tightens_boundary_for_predictive_error() -> None:
    model = _narrow_rate_envelope_multirotor()
    endpoint_errors = np.zeros((4, 12))
    endpoint_errors[:, 9] = (-0.04, 0.04, -0.04, 0.04)
    samples = (
        EmpiricalErrorSample(endpoint_errors, "group-a", "flight-a"),
        EmpiricalErrorSample(-endpoint_errors, "group-b", "flight-b"),
    )
    belief = DynamicsBelief(
        params=model.params,
        input_spec=model.input_spec,
        runtime_spec=model.runtime_spec,
        predictive_error=EmpiricalHorizonPredictiveError.from_samples(
            {model.runtime_spec.sample_period_s: samples}
        ),
    )
    policy = _test_policy(horizon_steps=4)
    point_controller = NMPCController(model, _policy=policy)
    belief_controller = NMPCController(belief, _policy=policy)
    state = resting_state()
    state[10] = 0.10
    previous = np.full(4, 0.5)
    nominal = np.clip(previous + 0.04 * np.asarray(MOTOR_MIXER[0]), 0.0, 1.0)

    def decide(controller: NMPCController):
        return controller._backend._select_support_command(
            jnp.asarray(state),
            controller.model.initial_latent_state(previous),
            jnp.asarray(nominal),
            jnp.asarray(previous),
            jnp.zeros(0),
        )

    point = decide(point_controller)
    uncertain = decide(belief_controller)

    assert point.mode is SupportFilterMode.NOMINAL_SAFE
    assert not point.applied
    assert uncertain.mode is SupportFilterMode.BOUNDARY_FILTERED
    assert uncertain.applied
    assert uncertain.nominal_fraction < 1.0
    assert uncertain.next_mean_validity < point.next_mean_validity
    assert uncertain.support_horizon_maximum_robust_validity <= 1.0 + 1e-6


def test_support_filter_requires_validity_and_rate_progress_outside_envelope() -> None:
    model = _narrow_rate_envelope_multirotor()
    controller = NMPCController(model, _policy=_test_policy(horizon_steps=4))
    state = resting_state()
    state[10] = 0.25
    previous = np.full(4, 0.5)
    nominal = np.clip(previous + 0.15 * np.asarray(MOTOR_MIXER[0]), 0.0, 1.0)
    decision = controller._backend._select_support_command(
        jnp.asarray(state),
        model.initial_latent_state(previous),
        jnp.asarray(nominal),
        jnp.asarray(previous),
        jnp.zeros(0),
    )

    assert decision.mode is SupportFilterMode.RECOVERY_FILTERED
    assert decision.applied
    assert decision.nominal_fraction < 1.0
    assert decision.current_validity > 1.0
    assert decision.support_horizon_terminal_robust_validity < decision.current_validity
    assert decision.support_horizon_terminal_rate_energy < (
        decision.current_rate_energy
    )


def test_solver_failure_does_not_inject_an_independent_controller() -> None:
    model = _multirotor_runtime()
    policy = _SolverPolicy(
        horizon_steps=6,
        block_count=3,
        maximum_iterations=2,
        line_search_steps=1,
        armijo_fraction=1e6,
    )
    controller = NMPCController(model, _policy=policy)
    target = resting_state()
    state = target.copy()
    state[2] = -0.3
    state[10] = 2.0
    previous = hover_control(true_parameters())

    result = controller.solve(
        jnp.asarray(state),
        controller.hold_reference(jnp.asarray(target)),
        previous,
    )

    assert result.status is SolveStatus.LINE_SEARCH_FAILED
    assert result.used_fallback
    assert result.diagnostics.support_filter_mode is SupportFilterMode.SOLVER_FALLBACK
    np.testing.assert_allclose(result.command, previous)


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
    controller = NMPCController(model, _policy=_test_policy())
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
        model.params,
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
            model.params,
        )
        minus = objective(
            blocks - direction,
            jnp.asarray(state),
            latent,
            reference.states,
            previous,
            exogenous,
            model.params,
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


def test_compatible_belief_rebind_reuses_compiled_parameterized_kernels() -> None:
    model = _multirotor_runtime()
    controller = NMPCController(model, _policy=_test_policy(horizon_steps=4))
    state = resting_state()
    state[7] = 0.08
    state[10] = 0.3
    previous = hover_control(true_parameters())
    reference = controller.hold_reference(jnp.asarray(resting_state()))

    original = controller.solve(jnp.asarray(state), reference, previous)
    jax.block_until_ready(original.command)
    compiled_functions = (
        controller._backend._initial_latent_compiled,
        controller._backend._objective_and_gradient,
        controller._backend._optimize_compiled,
        controller._backend._rollout_compiled,
        controller._backend._uncertainty_support_compiled,
        controller._backend._support_metric_compiled,
        controller._backend._support_metrics_compiled,
    )
    cache_sizes_before = tuple(
        function._cache_size() for function in compiled_functions
    )

    changed_params = model.params._replace(
        log_angular_accel=model.params.log_angular_accel + math.log(0.6)
    )
    rebound = controller.rebind_belief(replace(model, params=changed_params))
    changed = rebound.solve(jnp.asarray(state), reference, previous)
    jax.block_until_ready(changed.command)
    cache_sizes_after = tuple(function._cache_size() for function in compiled_functions)

    assert rebound._backend._objective_and_gradient is compiled_functions[1]
    assert cache_sizes_after == cache_sizes_before
    assert not original.used_fallback
    assert not changed.used_fallback
    assert not np.allclose(
        np.asarray(changed.predicted_states),
        np.asarray(original.predicted_states),
    )


def test_belief_rebind_rejects_a_changed_runtime_contract() -> None:
    model = _multirotor_runtime()
    controller = NMPCController(model, _policy=_test_policy(horizon_steps=4))
    changed_runtime = replace(
        model,
        runtime_spec=replace(model.runtime_spec, sample_period_s=0.01),
    )

    with pytest.raises(ValueError, match="runtime specification changed"):
        controller.rebind_belief(changed_runtime)


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


def test_exogenous_wind_forecast_flows_through_prediction() -> None:
    trajectory = generate_trajectory(seed=0, duration_s=0.1)
    exogenous = tuple(
        ExogenousChannel(
            name=f"wind_{axis}_m_s",
            role=f"wind_{axis}",
            semantic="forecast_world_wind",
            unit="m/s",
            frame="NWU",
        )
        for axis in ("north", "west", "up")
    )
    spec = replace(trajectory.spec, exogenous=exogenous)
    model = RuntimeDynamicsModel(
        true_parameters(),
        spec,
        _runtime_spec(trajectory.nominal_dt_s),
        DirectActuationMap(spec.controls),
    )
    controller = NMPCController(model, _policy=_test_policy())
    target = resting_state()
    previous = hover_control(true_parameters())

    result = controller.solve(
        jnp.asarray(target),
        controller.hold_reference(
            jnp.asarray(target), exogenous=jnp.asarray([0.2, -0.1, 0.0])
        ),
        previous,
    )

    assert result.predicted_states.shape == (controller.prediction_steps + 1, 13)
    assert np.all(np.isfinite(result.predicted_states))


def test_controller_honors_certified_prediction_horizon() -> None:
    model = _multirotor_runtime()
    certified_runtime = RuntimeModelSpec(
        sample_period_s=model.runtime_spec.sample_period_s,
        validity_envelope=model.runtime_spec.validity_envelope,
        certified_prediction_horizon_s=0.1,
        certification_source="synthetic horizon audit",
    )
    certified_model = replace(model, runtime_spec=certified_runtime)

    controller = NMPCController(certified_model)

    assert controller.prediction_horizon_s <= 0.1
    assert controller.prediction_steps == 5
