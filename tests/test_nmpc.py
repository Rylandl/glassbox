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

import glassbox.control.solver as nmpc_solver
from glassbox.belief.belief import DynamicsBelief
from glassbox.belief.forecast_error import (
    EmpiricalErrorSample,
    ForecastErrorEnvelope,
)
from glassbox.belief.information import ParameterInformation
from glassbox.control.fitted import BeliefPlanModel, NMPCController
from glassbox.control.plan import (
    NMPCWarmStart,
    SafetyEnvelope,
    SolverPolicy,
    SolveStatus,
    block_steps_for,
    blocks_cover_horizon,
    maintained_block_count,
)
from glassbox.control.solver import _objective, _projected_gradient_norm
from glassbox.core.data import (
    RIGID_BODY_STATE_SCHEMA,
    Channel,
    Trajectory,
    TrajectorySpec,
    VehicleConfigurationSpec,
)
from glassbox.core.dynamics import (
    fixed_wing_trim_control,
    hover_control,
    initial_residual_parameters,
    structured_parameter_vector,
)
from glassbox.core.fixedwing_synthetic import (
    TRIM_AIRSPEED_M_S,
    fixed_wing_trim_state,
    true_fixed_wing_parameters,
)
from glassbox.core.geometry import quaternion_log_error, rigid_body_local_error
from glassbox.core.model import (
    DirectActuationMap,
    ExecutableModel,
    ModelValidityEnvelope,
    RuntimeModelSpec,
)
from glassbox.core.synthetic import (
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
    trajectory: Trajectory,
    *,
    residual: bool = False,
    sample_period_s: float | None = None,
) -> ExecutableModel:
    params = true_parameters()
    if residual:
        params = initial_residual_parameters(params, hidden_units=3)
    return ExecutableModel(
        params,
        trajectory.spec,
        _runtime_spec(
            trajectory.nominal_dt_s if sample_period_s is None else sample_period_s
        ),
        DirectActuationMap(trajectory.spec.controls),
    )


def _narrow_rate_envelope_multirotor(
    trajectory: Trajectory,
) -> ExecutableModel:
    model = _multirotor_runtime(trajectory)
    runtime_spec = replace(
        model.runtime_spec,
        validity_envelope=ModelValidityEnvelope(
            body_velocity_center_m_s=(0.0, 0.0, 0.0),
            body_velocity_half_width_m_s=(10.0, 10.0, 10.0),
            angular_velocity_center_rad_s=(0.0, 0.0, 0.0),
            angular_velocity_half_width_rad_s=(0.2, 0.2, 0.2),
        ),
    )
    return ExecutableModel(
        model.params,
        model.input_spec,
        runtime_spec,
        model.actuation,
    )


def _fixed_wing_runtime(trajectory: Trajectory) -> ExecutableModel:
    return ExecutableModel(
        true_fixed_wing_parameters(),
        trajectory.spec,
        _runtime_spec(trajectory.nominal_dt_s),
        DirectActuationMap(trajectory.spec.controls),
    )


def _flying_wing_runtime() -> ExecutableModel:
    controls = (
        Channel(
            name="propulsion_command",
            role="throttle",
            semantic="normalized_command",
            unit="1",
            kind="control",
            minimum=0.0,
            maximum=1.0,
        ),
        Channel(
            name="elevon_roll_command",
            role="roll",
            semantic="normalized_generalized_command",
            unit="1",
            kind="control",
            frame="FLU",
            minimum=-1.0,
            maximum=1.0,
        ),
        Channel(
            name="elevon_pitch_command",
            role="pitch",
            semantic="normalized_generalized_command",
            unit="1",
            kind="control",
            frame="FLU",
            minimum=-1.0,
            maximum=1.0,
        ),
    )
    spec = TrajectorySpec(
        state_schema=RIGID_BODY_STATE_SCHEMA,
        observation_source="simulator_truth",
        channels=controls,
        vehicle=VehicleConfigurationSpec(
            family="fixedwing",
            configuration_id="synthetic_flying_wing",
            controlled_axes=("roll", "pitch"),
        ),
    )
    return ExecutableModel(
        true_fixed_wing_parameters(),
        spec,
        _runtime_spec(0.02),
        DirectActuationMap(spec.controls),
    )


def _test_policy(*, horizon_steps: int = 6) -> SolverPolicy:
    block_count = max(
        count for count in range(1, 4) if blocks_cover_horizon(horizon_steps, count)
    )
    return SolverPolicy(
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


# Compiling one ``NMPCController`` costs seconds; a warm solve costs
# milliseconds, and the compiled kernels live on the instance rather than on
# the class.  One controller per distinct (model or belief, policy, tolerance,
# envelope) combination is therefore shared across every test that only reads
# solve results.  A solve mutates nothing on the controller.


@pytest.fixture(scope="module")
def multirotor_model(
    quadrotor_trajectory_seed0_dur0_1s: Trajectory,
) -> ExecutableModel:
    return _multirotor_runtime(quadrotor_trajectory_seed0_dur0_1s)


@pytest.fixture(scope="module")
def narrow_envelope_model(
    quadrotor_trajectory_seed0_dur0_1s: Trajectory,
) -> ExecutableModel:
    return _narrow_rate_envelope_multirotor(quadrotor_trajectory_seed0_dur0_1s)


@pytest.fixture(scope="module")
def fixedwing_model(
    fixedwing_trajectory_seed0_dur0_1s: Trajectory,
) -> ExecutableModel:
    return _fixed_wing_runtime(fixedwing_trajectory_seed0_dur0_1s)


@pytest.fixture(scope="module")
def multirotor_controller(multirotor_model: ExecutableModel) -> NMPCController:
    """Default multirotor controller on the six-step test policy."""

    return NMPCController(multirotor_model, policy=_test_policy())


@pytest.fixture(scope="module")
def multirotor_controller_four_step(
    multirotor_model: ExecutableModel,
) -> NMPCController:
    return NMPCController(multirotor_model, policy=_test_policy(horizon_steps=4))


@pytest.fixture(scope="module")
def narrow_envelope_controller(
    narrow_envelope_model: ExecutableModel,
) -> NMPCController:
    return NMPCController(narrow_envelope_model, policy=_test_policy(horizon_steps=4))


@pytest.fixture(scope="module")
def fixedwing_controller(fixedwing_model: ExecutableModel) -> NMPCController:
    return NMPCController(fixedwing_model, policy=_test_policy())


@pytest.fixture(scope="module")
def line_search_failure_policy() -> SolverPolicy:
    """A policy whose Armijo condition no step can satisfy."""

    return SolverPolicy(
        horizon_steps=6,
        block_count=3,
        maximum_iterations=2,
        line_search_steps=1,
        armijo_fraction=1e6,
    )


@pytest.fixture(scope="module")
def line_search_failure_controller(
    multirotor_model: ExecutableModel, line_search_failure_policy: SolverPolicy
) -> NMPCController:
    return NMPCController(multirotor_model, policy=line_search_failure_policy)


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


def test_control_blocks_expand_and_commands_remain_bounded(
    multirotor_model: ExecutableModel,
) -> None:
    controller = NMPCController(multirotor_model, policy=_test_policy(horizon_steps=5))
    blocks = jnp.asarray(
        [
            [-1.0] * 4,
            [0.0] * 4,
            [1.0] * 4,
        ]
    )

    expanded = controller.plan._expand_normalized_blocks(blocks)
    commands = controller.plan._commands_from_normalized(expanded)

    np.testing.assert_allclose(expanded[:, 0], [-1.0, -1.0, 0.0, 0.0, 1.0])
    assert np.min(commands) >= 0.0
    assert np.max(commands) <= 1.0


def test_maintained_block_layout_covers_every_horizon_without_dead_blocks() -> None:
    for horizon_steps in range(1, 61):
        block_count = maintained_block_count(horizon_steps)
        block_steps = block_steps_for(horizon_steps, block_count)
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


def test_default_multirotor_block_layout_has_no_dead_blocks_at_fifty_hertz(
    quadrotor_trajectory_seed0_dur0_1s: Trajectory,
) -> None:
    controller = NMPCController(
        _multirotor_runtime(quadrotor_trajectory_seed0_dur0_1s, sample_period_s=0.05)
    )
    plan = controller.plan
    blocks = jnp.repeat(
        jnp.linspace(-1.0, 1.0, plan.block_count)[:, None],
        controller.model.command_size,
        axis=1,
    )

    expanded = np.asarray(plan._expand_normalized_blocks(blocks))

    assert controller.prediction_steps == 12
    assert plan.block_count == 6
    assert plan.policy.block_steps * plan.block_count == 12
    assert expanded.shape == (12, controller.model.command_size)
    assert len(np.unique(expanded[:, 0])) == plan.block_count


def test_solver_policy_rejects_a_layout_with_dead_command_blocks() -> None:
    with pytest.raises(ValueError, match="drive no prediction step"):
        SolverPolicy(horizon_steps=4, block_count=3)


def test_solver_propagates_latent_state_and_returns_bounded_plan(
    multirotor_model: ExecutableModel,
    multirotor_controller: NMPCController,
) -> None:
    model = multirotor_model
    controller = multirotor_controller
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


def _information_with_variance(
    model: ExecutableModel, index: int, variance: float
) -> ParameterInformation:
    """One coordinate resolved to a stated variance, everything else unknown."""

    information = ParameterInformation.unknown(model.params)
    precision = np.zeros_like(information.precision)
    precision[index, index] = 1.0 / variance
    return information.with_precision(precision, effective_count=4.0)


def test_solver_consumes_predictive_and_parameter_uncertainty(
    multirotor_model: ExecutableModel,
) -> None:
    model = multirotor_model
    endpoint_errors = 0.02 * np.concatenate((np.eye(12), -np.eye(12)))
    error_samples = (
        EmpiricalErrorSample(endpoint_errors, "group-a", "flight-a"),
        EmpiricalErrorSample(endpoint_errors, "group-b", "flight-b"),
    )
    belief = DynamicsBelief(
        model=ExecutableModel(model.params, model.input_spec, model.runtime_spec),
        information=_information_with_variance(model, 0, 0.04),
        forecast_error=ForecastErrorEnvelope.from_samples(
            {0.1: error_samples, 0.2: error_samples}
        ),
    )
    controller = NMPCController(belief, policy=_test_policy(horizon_steps=4))
    target = resting_state()
    previous = hover_control(true_parameters())

    result = controller.solve(
        jnp.asarray(target),
        controller.hold_reference(jnp.asarray(target)),
        previous,
    )

    assert (
        result.diagnostics.maximum_normalized_model_uncertainty_standard_deviation > 0.0
    )


def _point_objective(
    plan: BeliefPlanModel,
    blocks: jax.Array,
    state: jax.Array,
    latent: jax.Array,
    reference_states: jax.Array,
    previous_command: jax.Array,
    exogenous: jax.Array,
) -> jax.Array:
    """The point objective the solver scored before it charged spread.

    Written out here rather than imported, so that the identity the solver is
    asked to preserve is stated independently of the solver.
    """

    policy = plan.policy
    states, _, commands = plan._mean_rollout(
        blocks, state, latent, exogenous, plan.values.parameters
    )
    local_error = jax.vmap(rigid_body_local_error)(reference_states[1:], states[1:])
    normalized_error = local_error / plan.tolerances.local_state_scale
    tracking = jnp.mean(jnp.sum(jnp.square(normalized_error), axis=1))
    terminal = policy.terminal_weight * jnp.sum(jnp.square(normalized_error[-1]))
    command_range = plan.command_maximum - plan.command_minimum
    command_delta = jnp.diff(
        jnp.concatenate((previous_command[None, :], commands), axis=0), axis=0
    )
    normalized_delta = command_delta / (policy.command_change_fraction * command_range)
    smoothness = policy.command_change_weight * jnp.mean(jnp.square(normalized_delta))
    utilization = jax.vmap(plan._validity_utilization)(states[1:], exogenous)
    validity = policy.validity_weight * jnp.mean(
        jnp.square(jax.nn.relu(utilization - 1.0))
    )
    safety = policy.safety_weight * jnp.mean(
        jnp.square(jax.vmap(plan._safety_violation)(states[1:]))
    )
    return tracking + terminal + smoothness + validity + safety


def _charged_objective(controller: NMPCController, *arguments: jax.Array) -> float:
    return float(
        _objective(
            controller.plan,
            controller.plan.policy,
            *arguments,
            controller.plan.values,
        )
    )


def _objective_arguments(
    controller: NMPCController,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    solver = controller.solver
    target = resting_state()
    state = target.copy()
    state[2] = -0.2
    previous = jnp.asarray(hover_control(true_parameters()))
    latent = solver._initial_latent(previous, None, None)
    reference = controller.hold_reference(jnp.asarray(target))
    exogenous = solver._exogenous_forecast(reference)
    blocks = 0.25 + 0.5 * solver._cold_blocks(previous)
    return blocks, jnp.asarray(state), latent, reference.states, previous, exogenous


def test_point_model_objective_is_the_point_objective_bit_for_bit(
    multirotor_controller_four_step: NMPCController,
) -> None:
    controller = multirotor_controller_four_step
    arguments = _objective_arguments(controller)

    assert controller.plan.values.covariance_factor is None
    charged = _charged_objective(controller, *arguments)
    point = float(_point_objective(controller.plan, *arguments))
    assert charged == point


def test_a_belief_with_covariance_is_charged_more_than_a_point_belief(
    multirotor_model: ExecutableModel,
    multirotor_controller_four_step: NMPCController,
) -> None:
    model = multirotor_model
    parameter_count = len(structured_parameter_vector(model.params))
    belief = DynamicsBelief(
        model=ExecutableModel(model.params, model.input_spec, model.runtime_spec),
        information=_information_with_variance(model, 0, 0.04),
    )
    uncertain = NMPCController(belief, policy=_test_policy(horizon_steps=4))
    arguments = _objective_arguments(multirotor_controller_four_step)

    point_value = _charged_objective(multirotor_controller_four_step, *arguments)
    uncertain_value = _charged_objective(uncertain, *arguments)

    assert uncertain.plan.values.covariance_factor is not None
    assert uncertain.plan.values.covariance_factor.shape == (parameter_count, 1)
    assert uncertain_value > point_value


def test_default_horizon_does_not_exceed_predictive_error_evidence(
    multirotor_model: ExecutableModel,
) -> None:
    model = multirotor_model
    endpoint_errors = 0.02 * np.concatenate((np.eye(12), -np.eye(12)))
    samples = (
        EmpiricalErrorSample(endpoint_errors, "group-a", "flight-a"),
        EmpiricalErrorSample(endpoint_errors, "group-b", "flight-b"),
    )
    belief = DynamicsBelief(
        model=ExecutableModel(model.params, model.input_spec, model.runtime_spec),
        forecast_error=ForecastErrorEnvelope.from_samples({0.1: samples}),
    )

    controller = NMPCController(belief)

    assert controller.prediction_horizon_s == pytest.approx(0.1)


def test_default_multirotor_horizon_snaps_near_integer_sample_ratio(
    multirotor_model: ExecutableModel,
) -> None:
    controller = NMPCController(multirotor_model)

    assert controller.prediction_steps == 30
    assert controller.prediction_horizon_s == pytest.approx(0.6)


def test_applied_command_initializes_latent_actuator_state(
    multirotor_model: ExecutableModel, multirotor_controller: NMPCController
) -> None:
    model = multirotor_model
    controller = multirotor_controller
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


def test_warm_start_is_selected_only_when_no_worse_than_cold_start(
    multirotor_controller: NMPCController,
) -> None:
    controller = multirotor_controller
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


def test_warm_start_seed_advances_the_previous_plan_by_one_block(
    multirotor_model: ExecutableModel, multirotor_controller: NMPCController
) -> None:
    model = multirotor_model
    solver = multirotor_controller.solver
    plan = multirotor_controller.plan
    block_values = (0.2, 0.5, 0.8)
    previous_blocks = np.asarray(
        [[value] * model.command_size for value in block_values]
    )
    previous_plan = np.repeat(previous_blocks, plan.policy.block_steps, axis=0)

    seed = solver._warm_blocks(NMPCWarmStart(jnp.asarray(previous_plan)))
    seed_commands = np.asarray(plan._commands_from_normalized(seed))

    assert plan.policy.block_steps == 2
    np.testing.assert_allclose(seed_commands[0], previous_blocks[1], atol=1e-6)
    np.testing.assert_allclose(seed_commands[-1], previous_blocks[-1], atol=1e-6)
    np.testing.assert_allclose(
        seed_commands,
        previous_blocks[[1, 2, 2]],
        atol=1e-6,
    )
    # The seed must not simply reproduce the previous unshifted plan.
    assert np.max(np.abs(seed_commands - previous_blocks)) > 0.1


def test_invalid_estimate_and_deadline_return_bounded_fallback(
    multirotor_model: ExecutableModel, multirotor_controller: NMPCController
) -> None:
    model = multirotor_model
    controller = multirotor_controller
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


def test_previous_command_within_rounding_of_a_bound_is_accepted(
    multirotor_model: ExecutableModel, multirotor_controller: NMPCController
) -> None:
    model = multirotor_model
    controller = multirotor_controller
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
    monkeypatch: pytest.MonkeyPatch, multirotor_controller: NMPCController
) -> None:
    controller = multirotor_controller
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


def test_forced_line_search_failure_returns_bounded_fallback(
    line_search_failure_controller: NMPCController,
) -> None:
    controller = line_search_failure_controller
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


def test_converged_status_requires_the_first_order_criterion(
    multirotor_controller: NMPCController,
) -> None:
    policy = _test_policy()
    controller = multirotor_controller
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


def test_improvement_stall_is_reported_as_stalled_rather_than_converged(
    multirotor_model: ExecutableModel,
) -> None:
    model = multirotor_model
    policy = SolverPolicy(
        horizon_steps=6,
        block_count=3,
        maximum_iterations=4,
        line_search_steps=5,
        gradient_tolerance=1e-9,
        relative_improvement_tolerance=1.0,
    )
    controller = NMPCController(model, policy=policy)
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


def test_line_search_stall_after_progress_keeps_the_improved_plan(
    multirotor_model: ExecutableModel,
) -> None:
    model = multirotor_model
    policy = SolverPolicy(
        horizon_steps=6,
        block_count=3,
        maximum_iterations=12,
        line_search_steps=1,
        initial_step_size=0.5,
        armijo_fraction=0.5,
        gradient_tolerance=1e-9,
        relative_improvement_tolerance=1e-12,
    )
    controller = NMPCController(model, policy=policy)
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


def test_solver_failure_does_not_inject_an_independent_controller(
    line_search_failure_controller: NMPCController,
) -> None:
    controller = line_search_failure_controller
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
    np.testing.assert_allclose(result.command, previous)


def test_safety_envelope_reports_normalized_prediction_violation(
    multirotor_model: ExecutableModel,
) -> None:
    controller = NMPCController(
        multirotor_model,
        safety_envelope=SafetyEnvelope(maximum_position_m=(0.1, 0.1, 0.1)),
        policy=_test_policy(),
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
def test_objective_gradient_agrees_with_central_difference(
    model_kind: str,
    quadrotor_trajectory_seed0_dur0_1s: Trajectory,
    fixedwing_model: ExecutableModel,
    multirotor_controller: NMPCController,
    fixedwing_controller: NMPCController,
) -> None:
    if model_kind == "fixedwing":
        model = fixedwing_model
        state = fixed_wing_trim_state()
        state[2] = -2.0
        state[4] = 1.0
        target = fixed_wing_trim_state()
        previous = fixed_wing_trim_control(
            true_fixed_wing_parameters(), TRIM_AIRSPEED_M_S
        )
    else:
        model = _multirotor_runtime(
            quadrotor_trajectory_seed0_dur0_1s, residual=model_kind == "residual"
        )
        state = resting_state()
        state[0] = 0.1
        target = resting_state()
        previous = hover_control(true_parameters())
    if model_kind == "fixedwing":
        controller = fixedwing_controller
    elif model_kind == "multirotor":
        controller = multirotor_controller
    else:
        controller = NMPCController(model, policy=_test_policy())
    reference = controller.hold_reference(jnp.asarray(target))
    latent = model.initial_latent_state(previous)
    exogenous = jnp.zeros((controller.prediction_steps, model.exogenous_size))
    blocks = controller.solver._cold_blocks(previous) + jnp.asarray(
        [
            [0.10, 0.08, -0.06, 0.05],
            [0.02, -0.10, 0.07, -0.04],
            [-0.05, 0.03, 0.04, 0.02],
        ]
    )
    plan = controller.plan
    objective = jax.jit(lambda *arguments: _objective(plan, plan.policy, *arguments))
    _, analytic = controller.solver._kernels.objective_and_gradient(
        blocks,
        jnp.asarray(state),
        latent,
        reference.states,
        previous,
        exogenous,
        plan.values,
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
            plan.values,
        )
        minus = objective(
            blocks - direction,
            jnp.asarray(state),
            latent,
            reference.states,
            previous,
            exogenous,
            plan.values,
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


def test_incompatible_warm_start_is_safely_ignored(
    fixedwing_controller: NMPCController,
) -> None:
    controller = fixedwing_controller
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
    controller = NMPCController(model, policy=_test_policy())
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


def test_exogenous_wind_forecast_flows_through_prediction(
    quadrotor_trajectory_seed0_dur0_1s: Trajectory,
) -> None:
    trajectory = quadrotor_trajectory_seed0_dur0_1s
    exogenous = tuple(
        Channel(
            name=f"wind_{axis}_m_s",
            role=f"wind_{axis}",
            semantic="forecast_world_wind",
            unit="m/s",
            kind="exogenous",
            frame="NWU",
        )
        for axis in ("north", "west", "up")
    )
    spec = replace(trajectory.spec, channels=(*trajectory.spec.channels, *exogenous))
    model = ExecutableModel(
        true_parameters(),
        spec,
        _runtime_spec(trajectory.nominal_dt_s),
        DirectActuationMap(spec.controls),
    )
    controller = NMPCController(model, policy=_test_policy())
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


def test_controller_honors_certified_prediction_horizon(
    multirotor_model: ExecutableModel,
) -> None:
    model = multirotor_model
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
