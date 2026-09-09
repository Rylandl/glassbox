"""Regressions at the belief, actuator, and telemetry boundaries."""

from dataclasses import dataclass, replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from glassbox import DynamicsBelief, NMPCController, SolverPolicy, SolveStatus
from glassbox.belief.information import ParameterInformation
from glassbox.belief.update import one_step_linearization
from glassbox.control.fitted import parameter_covariance_factor
from glassbox.control.solver import _optimize_step
from glassbox.control.supervisor import (
    MultirotorFlightSupervisor,
    MultirotorSupervisorConfig,
    SupervisorReason,
)
from glassbox.core.data import (
    Channel,
    Trajectory,
    make_trajectory_spec,
    trajectory_segment,
)
from glassbox.core.dynamics import (
    GRAVITY_M_S2,
    QUADROTOR_CONTROL_NAMES,
    hover_control,
    rollout,
    step_with_latent,
    structured_parameter_vector,
    with_structured_parameter_vector,
)
from glassbox.core.model import (
    DirectActuationMap,
    ExecutableModel,
    ModelValidityEnvelope,
    NonActionableModelError,
    RuntimeModelSpec,
)
from glassbox.core.synthetic import resting_state, true_parameters
from glassbox.integrations.px4 import PX4MavlinkLink, PX4StateSample


@pytest.fixture
def model():
    return ExecutableModel(
        true_parameters(),
        make_trajectory_spec(
            QUADROTOR_CONTROL_NAMES,
            family="multirotor",
            observation_source="simulator_truth",
            configuration_id="runtime_contract",
        ),
        RuntimeModelSpec(
            0.02,
            ModelValidityEnvelope(
                (0.0, 0.0, 0.0),
                (100.0, 100.0, 100.0),
                (0.0, 0.0, 0.0),
                (100.0, 100.0, 100.0),
            ),
        ),
    )


@dataclass(frozen=True)
class GainMap:
    command_channels: tuple[Channel, ...]
    gain: float = 1.0
    model_control_size: int = 4

    def model_control(self, command):
        return self.gain * command


def test_default_line_search_can_descend_a_steep_bounded_objective():
    policy = SolverPolicy(horizon_steps=1, block_count=1, maximum_iterations=1)
    objective = jax.value_and_grad(
        lambda blocks, *_: 1500.0 * jnp.sum(jnp.square(blocks - 0.1))
    )
    blocks = jnp.zeros((1, 1))
    value, gradient = objective(blocks)
    outcome = _optimize_step(
        policy, objective, blocks, value, gradient, None, None, None, None, None, None
    )
    assert float(outcome[1]) < float(value)
    assert not bool(outcome[6])  # No line-search failure.
    assert bool(outcome[7])  # An improving step was accepted.
    assert np.all(np.abs(outcome[0]) <= 1.0)


@pytest.mark.parametrize("maximum", (0.3, 1.0))
def test_custom_mapping_and_bounds_do_not_alias_cached_controllers(model, maximum):
    policy = SolverPolicy(
        allow_unresolved_parameters=True, horizon_steps=2, block_count=2
    )
    first = NMPCController(
        replace(model, actuation=GainMap(model.input_spec.controls)), policy=policy
    )
    channels = tuple(replace(c, maximum=maximum) for c in model.input_spec.controls)
    second = NMPCController(
        replace(model, actuation=GainMap(channels, 0.5)), policy=policy
    )
    assert first.plan.compile_signature != second.plan.compile_signature
    state = jnp.asarray(resting_state())
    blocks = jnp.zeros((2, 4))
    latent = jnp.full(4, 0.15)
    exogenous = jnp.zeros((2, 0))
    expected = second.plan.rollout(blocks, state, latent, exogenous, second.plan.values)
    actual = second.solver._kernels.rollout(
        blocks, state, latent, exogenous, second.plan.values
    )
    np.testing.assert_allclose(actual.mean_states, expected.mean_states, atol=1e-7)
    result = second.solve(state, second.hold_reference(state), latent)
    assert result.command_usable
    assert np.all(result.command <= second.model.command_maximum)


def test_direct_mapping_bounds_are_part_of_cache_identity(model):
    policy = SolverPolicy(
        allow_unresolved_parameters=True, horizon_steps=2, block_count=2
    )
    first = NMPCController(model, policy=policy)
    channels = tuple(replace(c, maximum=0.3) for c in model.input_spec.controls)
    second = NMPCController(
        replace(model, actuation=DirectActuationMap(channels)), policy=policy
    )
    assert first.plan.compile_signature != second.plan.compile_signature
    rebound = NMPCController(model.rebind_parameters(model.params), policy=policy)
    assert first.solver._kernels is rebound.solver._kernels


@pytest.mark.parametrize("minimum,maximum", ((0.0, 1.0), (-3.0, 0.2)))
def test_normalized_command_bounds_preserve_feasible_inward_derivatives(
    model, minimum, maximum
):
    channels = tuple(
        replace(channel, minimum=minimum, maximum=maximum)
        for channel in model.input_spec.controls
    )
    controller = NMPCController(
        replace(model, actuation=DirectActuationMap(channels)),
        policy=SolverPolicy(horizon_steps=2, block_count=2),
    )
    normalize = controller.plan._commands_from_normalized
    blocks = jnp.asarray([-1.0, 1.0, -1.0, 1.0])
    direction = -blocks
    commands, derivative = jax.jvp(normalize, (blocks,), (direction,))
    np.testing.assert_array_equal(
        commands, jnp.asarray([minimum, maximum, minimum, maximum])
    )
    np.testing.assert_allclose(derivative, 0.5 * (maximum - minimum) * direction)
    step = 1e-3
    inward_difference = (normalize(blocks + step * direction) - commands) / step
    np.testing.assert_allclose(derivative, inward_difference, rtol=2e-4)


def test_rollout_linearization_matches_a_feasible_step_off_active_command_bounds(model):
    controller = NMPCController(
        model, policy=SolverPolicy(horizon_steps=2, block_count=2)
    )
    plan = controller.plan
    blocks = jnp.asarray([[-1.0, 1.0, -1.0, 1.0]] * 2)
    direction = -blocks

    def final_state(candidate):
        return plan.rollout(
            candidate,
            jnp.asarray(resting_state()),
            jnp.full(4, 0.5),
            jnp.zeros((2, 0)),
            plan.values,
        ).mean_states[-1]

    value, derivative = jax.jvp(final_state, (blocks,), (direction,))
    step = 1e-3
    inward_difference = (final_state(blocks + step * direction) - value) / step
    np.testing.assert_allclose(derivative, inward_difference, rtol=5e-3, atol=1e-4)


def test_support_measurement_includes_an_initial_state_that_reenters_the_envelope(
    model,
):
    controller = NMPCController(
        model, policy=SolverPolicy(horizon_steps=2, block_count=2)
    )
    rest = jnp.asarray(resting_state())
    prediction = controller.plan.rollout(
        jnp.zeros((2, 4)),
        rest,
        jnp.full(4, 0.5),
        jnp.zeros((2, 0)),
        controller.plan.values,
    )
    outside = rest.at[3].set(200.0)
    prediction = prediction._replace(mean_states=jnp.stack((outside, rest, rest)))
    measurements = controller.plan.measure(prediction)
    assert float(measurements.maximum_validity_utilization) == pytest.approx(2.0)


def test_solver_refuses_out_of_bounds_output_from_a_plan_model(model, monkeypatch):
    controller = NMPCController(
        model,
        policy=SolverPolicy(
            allow_unresolved_parameters=True, horizon_steps=2, block_count=2
        ),
    )
    kernels = controller.solver._kernels

    def bad_rollout(*args):
        prediction = kernels.rollout(*args)
        return prediction._replace(commands=jnp.full_like(prediction.commands, 1.1))

    monkeypatch.setattr(
        controller.solver, "_kernels", replace(kernels, rollout=bad_rollout)
    )
    state = jnp.asarray(resting_state())
    result = controller.solve(state, controller.hold_reference(state), jnp.full(4, 0.5))
    assert result.status == SolveStatus.COMMAND_BOUND_VIOLATION
    assert not result.command_usable
    np.testing.assert_allclose(result.command, 0.5)


def test_belief_requires_explicit_custom_mapping_on_reload(model, tmp_path):
    actuation = GainMap(model.input_spec.controls, 0.5)
    belief = DynamicsBelief(replace(model, actuation=actuation))
    path = tmp_path / "belief.json"
    belief.save(path)
    for loader in (DynamicsBelief.load, ExecutableModel.load):
        with pytest.raises(NonActionableModelError, match="external actuation map"):
            loader(path)
    restored = DynamicsBelief.load(path, actuation=actuation)
    command = jnp.full((2, 4), 0.5)
    np.testing.assert_array_equal(
        restored.rollout(resting_state(), command).states,
        belief.rollout(resting_state(), command).states,
    )
    assert ExecutableModel.load(path, actuation=actuation).actuation is actuation
    wrong_channels = tuple(replace(c, maximum=0.3) for c in actuation.command_channels)
    with pytest.raises(ValueError, match="saved interface"):
        DynamicsBelief.load(path, actuation=GainMap(wrong_channels))


def test_direct_command_bounds_round_trip(model, tmp_path):
    channels = tuple(replace(c, maximum=0.3) for c in model.input_spec.controls)
    belief = DynamicsBelief(replace(model, actuation=DirectActuationMap(channels)))
    path = tmp_path / "limited.json"
    belief.save(path)
    restored = DynamicsBelief.load(path)
    np.testing.assert_array_equal(
        restored.model.command_maximum, belief.model.command_maximum
    )


def test_segment_linearization_uses_its_control_prefix(model):
    controls = jnp.concatenate((jnp.full((50, 4), 0.2), jnp.full((10, 4), 0.8)))
    flight = Trajectory(
        np.arange(61) * 0.02,
        np.asarray(rollout(model.params, jnp.asarray(resting_state()), controls, 0.02)),
        np.asarray(controls),
        model.input_spec,
    )
    segment = trajectory_segment(flight, 50, 60)
    errors, jacobian = one_step_linearization(model, flight, np.array([50]))
    segment_errors, segment_jacobian = one_step_linearization(
        model, segment, np.array([0])
    )
    np.testing.assert_allclose(segment_errors, errors, atol=1e-7)
    np.testing.assert_allclose(segment_jacobian, jacobian, atol=1e-7)


def test_px4_preserves_reception_age_through_command_alignment(monkeypatch):
    now = [10.0]
    monkeypatch.setattr("glassbox.integrations.px4.time.monotonic", lambda: now[0])
    sample = PX4StateSample(resting_state(), 1000, 1000, 0.0, 0.2, received_at_s=9.8)

    def applied(*args, **kwargs):
        now[0] += 0.1
        return SimpleNamespace(
            command=np.full(4, 0.5), source_time_us=1_000_000, armed=True
        )

    link = PX4MavlinkLink(
        SimpleNamespace(next_sample=lambda **kwargs: sample),
        command_size=4,
        command_bounds=(np.zeros(4), np.ones(4)),
        applied_command_source=SimpleNamespace(sample_nearest=applied),
    )
    observation = link.read(timeout_s=0.2)
    assert observation.received_at_s == 9.8
    assert observation.receive_age_s == pytest.approx(0.3)
    supervisor = MultirotorFlightSupervisor(MultirotorSupervisorConfig())
    decision = supervisor.supervise(
        state=observation.state,
        state_received_at_s=observation.received_at_s,
        candidate_command=np.full(4, 0.5),
        command_generated_at_s=now[0],
        now_s=now[0],
        controller_command_usable=True,
        previous_applied_command=observation.applied_command,
    )
    assert SupervisorReason.STATE_STALE in decision.reasons
    assert not decision.nominal_command_accepted


@pytest.mark.parametrize("factor", (5.0, 10.0, 100.0))
def test_absorb_keeps_large_nonlinear_updates_finite_and_decreasing(model, factor):
    controls = jnp.broadcast_to(hover_control(model.params), (2, 4))
    flight = Trajectory(
        np.arange(3) * 0.02,
        np.asarray(rollout(model.params, jnp.asarray(resting_state()), controls, 0.02)),
        np.asarray(controls),
        model.input_spec,
    )
    vector = np.asarray(structured_parameter_vector(model.params)).copy()
    vector[0] -= np.log(factor)
    params = with_structured_parameter_vector(model.params, jnp.asarray(vector))
    mask = np.zeros(len(vector), dtype=bool)
    mask[0] = True
    belief = DynamicsBelief(
        replace(model, params=params),
        ParameterInformation.unknown(params, estimable=mask),
    )
    updated, result = belief.absorb(flight)
    assert result.absorbed
    assert np.isfinite(result.innovation_rms_after)
    assert result.innovation_rms_after < result.innovation_rms_before
    assert np.isfinite(updated.params.physical()["thrust_accel"])
    step = np.asarray(structured_parameter_vector(updated.params)) - vector
    assert np.linalg.norm(step / belief.information.scale) <= 1.0 + 1e-6
    # The actual residual, not the linearized zero, is the noise observation.
    assert updated.information.innovation_noise[5] > belief.information.noise_floor[5]


def test_added_information_does_not_erase_an_unrelated_covariance(tmp_path):
    info = ParameterInformation(
        names=("x", "y"),
        precision=np.eye(2),
        scale=np.ones(2),
        estimable=np.ones(2, dtype=bool),
        innovation_noise=np.ones(12),
        noise_floor=np.ones(12) * 1e-8,
        effective_count=1,
        rank_relative_tolerance=0.01,
    )
    updated = info.with_precision(info.precision + np.diag([1000.0, 0.0]))
    assert updated.resolved_rank() == info.resolved_rank() == 2
    assert updated.covariance()[1, 1] == pytest.approx(info.covariance()[1, 1])
    from glassbox.belief.information import parameter_information_from_dict

    restored = parameter_information_from_dict(updated.to_dict())
    assert restored.rank_threshold == info.rank_threshold
    np.testing.assert_array_equal(restored.covariance(), updated.covariance())


def test_planner_preserves_small_variance_with_large_prediction_sensitivity(model):
    base = ParameterInformation.unknown(model.params)
    mask = np.zeros(len(base.names), dtype=bool)
    mask[:2] = True
    precision = np.zeros_like(base.precision)
    precision[0, 0], precision[1, 1] = 1e12, 1.0
    information = replace(
        base, precision=precision, estimable=mask, rank_threshold=1e-6
    )
    factor = parameter_covariance_factor(DynamicsBelief(model, information))
    sensitivity = np.zeros(len(base.names))
    sensitivity[0] = 1e6
    assert information.complete
    assert factor.shape[1] == information.resolved_rank() == 2
    expected = sensitivity @ information.covariance() @ sensitivity
    assert float(np.sum(np.square(sensitivity @ factor))) == pytest.approx(expected)
    assert expected == pytest.approx(1.0)


def test_unresolved_belief_requires_explicit_planning_override(model):
    belief = DynamicsBelief(model)
    policy = SolverPolicy(horizon_steps=2, block_count=2)
    controller = NMPCController(belief, policy=policy)
    state = jnp.asarray(resting_state())
    previous = jnp.full(4, 0.4)
    result = controller.solve(state, controller.hold_reference(state), previous)
    assert result.status is SolveStatus.UNRESOLVED_MODEL
    assert not result.command_usable
    np.testing.assert_array_equal(result.command, previous)
    assert np.isinf(
        result.diagnostics.maximum_normalized_model_uncertainty_standard_deviation
    )
    allowed = NMPCController(
        belief, policy=replace(policy, allow_unresolved_parameters=True)
    )
    result = allowed.solve(state, allowed.hold_reference(state), previous)
    assert result.command_usable
    assert not result.diagnostics.parameter_uncertainty_complete
    assert result.diagnostics.unresolved_parameters_allowed
    assert np.isinf(
        result.diagnostics.maximum_normalized_model_uncertainty_standard_deviation
    )
    prediction = belief.rollout(state, jnp.tile(previous, (2, 1)))
    assert (
        prediction.unresolved_parameter_basis.shape[1]
        == belief.information.estimable_count
    )
    assert not prediction.parameter_information_complete
    assert not prediction.uncertainty_available


@pytest.mark.parametrize("tau", (1e-6, 1e-4, 0.005, 0.08, 10.0))
def test_actuator_forcing_integrates_exact_velocity_and_position_moments(model, tau):
    dt = 0.02
    params = model.params._replace(
        log_motor_time_constant=jnp.log(jnp.asarray(tau)),
        log_linear_drag=jnp.log(jnp.asarray(1e-10)),
    )
    state, latent = step_with_latent(
        params, jnp.asarray(resting_state()), jnp.zeros(4), jnp.ones(4), dt
    )
    tau = float(params.physical()["motor_time_constant"])
    thrust = 4 * float(params.physical()["thrust_accel"])
    impulse = dt + tau * np.expm1(-dt / tau)
    position_moment = dt**2 / 2 - tau * dt - tau**2 * np.expm1(-dt / tau)
    expected_vz = thrust * impulse - GRAVITY_M_S2 * dt
    expected_z = thrust * position_moment - GRAVITY_M_S2 * dt**2 / 2
    assert float(state[5]) == pytest.approx(expected_vz, abs=1e-7)
    assert float(state[2]) == pytest.approx(expected_z, abs=2e-9)
    np.testing.assert_allclose(latent, -np.expm1(-dt / tau), atol=1e-7)
