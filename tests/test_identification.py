from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from glassbox.core.data import make_trajectory_spec, trajectory_windows
from glassbox.core.dynamics import (
    initial_residual_parameters,
    with_thrust_command_offset,
)
from glassbox.core.identification import (
    MAXIMUM_WINDOWS_PER_HORIZON,
    _fit_objective,
    _optimization_batch_schedules,
    deterministic_weighted_batch_schedule,
    dynamic_envelope_penalty,
    fit_dynamics,
    residual_initialization_statistics,
    rollout_loss_configuration,
    supports_multirotor_thrust_command_offset,
)
from glassbox.core.model import ModelValidityEnvelope, model_validity_utilization
from glassbox.core.synthetic import (
    generate_trajectory,
    initial_parameter_guess,
    true_parameters,
)
from glassbox.io.nanodrone_reference import nanodrone_trajectory_spec


def test_multistep_fit_reduces_training_loss(quadrotor_flight) -> None:
    trajectories = [
        quadrotor_flight(0, 2.0),
        quadrotor_flight(1, 2.0),
    ]
    windows = trajectory_windows(trajectories, horizon=10, stride=10)

    result = fit_dynamics(
        [windows],
        initial_parameter_guess(),
        steps=80,
        learning_rate=0.03,
    )

    assert result.final_loss < 0.25 * result.initial_loss


@pytest.mark.parametrize("minibatch", [False, True])
def test_final_iterate_selection_uses_comparable_losses(quadrotor_flight, minibatch):
    initial = initial_parameter_guess()
    target = initial.log_thrust_accel + 1.0

    def objective(params):
        return (params.log_thrust_accel - target) ** 2

    configuration = rollout_loss_configuration(
        [trajectory_windows([quadrotor_flight(0, 0.2)], horizon=5)]
    )
    result = _fit_objective(
        objective,
        lambda params: jnp.atleast_1d(objective(params)),
        initial,
        steps=1 if minibatch else 2,
        learning_rate=0.5 if minibatch else 1.0,
        gradient_clip_norm=10.0,
        fixed_motor_time_constant=False,
        fixed_thrust_command_offset=False,
        loss_configuration=configuration,
        batch_objective=(
            (lambda params, indices: 0.001 * objective(params)) if minibatch else None
        ),
        batch_schedules=(np.zeros((1, 1), dtype=np.int64),) if minibatch else None,
        batch_window_counts=(2,) if minibatch else None,
    )

    assert result.final_loss == pytest.approx(float(objective(result.params)))
    assert result.final_loss == pytest.approx(0.25 if minibatch else 0.0, abs=1e-5)
    assert result.completed_steps == (1 if minibatch else 2)


def test_deterministic_weighted_batches_span_large_window_sets() -> None:
    first = deterministic_weighted_batch_schedule(
        None,
        window_count=20,
        steps=20,
        maximum_batch_size=4,
    )
    second = deterministic_weighted_batch_schedule(
        None,
        window_count=20,
        steps=20,
        maximum_batch_size=4,
    )

    np.testing.assert_array_equal(first, second)
    assert first.shape == (20, 4)
    assert len(np.unique(first)) == 20

    weighted = deterministic_weighted_batch_schedule(
        np.concatenate((np.ones(10), 2.0 * np.ones(10))),
        window_count=20,
        steps=100,
        maximum_batch_size=5,
    )
    counts = np.bincount(weighted.ravel(), minlength=20)
    assert np.sum(counts[10:]) == pytest.approx(2.0 * np.sum(counts[:10]), rel=0.02)


def test_affordable_fit_uses_every_window(quadrotor_trajectory_seed11_dur0_4s) -> None:
    # Any window count under MAXIMUM_WINDOWS_PER_HORIZON (8192)
    # selects the full-batch policy; a 0.4s rollout at horizon 1 already
    # produces one, so the rollout no longer has to be eleven seconds long.
    windows = trajectory_windows(
        [quadrotor_trajectory_seed11_dur0_4s],
        horizon=1,
        stride=1,
    )

    result = fit_dynamics(
        [windows],
        initial_parameter_guess(),
        steps=1,
        learning_rate=0.01,
    )

    assert result.optimization_policy == "full_batch_v1"
    assert result.batch_sizes == ()
    assert result.window_coverage == ()
    assert np.isfinite(result.initial_loss)
    assert np.isfinite(result.final_loss)


def test_automatic_minibatch_caps_large_short_horizon_window_set(
    quadrotor_flight,
) -> None:
    # The oversized window set below is synthesized with ``np.resize``, which
    # tiles whatever it is given, so the source rollout only has to be long
    # enough to yield one window.
    windows = trajectory_windows(
        [quadrotor_flight(12, 0.4)],
        horizon=1,
        stride=1,
    )
    repeated_count = MAXIMUM_WINDOWS_PER_HORIZON + 10
    repeated = replace(
        windows,
        initial_states=np.resize(
            windows.initial_states,
            (repeated_count, 13),
        ),
        control_histories=np.resize(
            windows.control_histories,
            (repeated_count, *windows.control_histories.shape[1:]),
        ),
        controls=np.resize(
            windows.controls,
            (repeated_count, *windows.controls.shape[1:]),
        ),
        target_states=np.resize(
            windows.target_states,
            (repeated_count, *windows.target_states.shape[1:]),
        ),
        initial_exogenous=np.resize(
            windows.initial_exogenous,
            (repeated_count, windows.initial_exogenous.shape[1]),
        ),
        window_weights=None,
        trajectory_indices=None,
        start_indices=None,
        candidate_window_counts=None,
    )

    schedules = _optimization_batch_schedules((repeated,), steps=2)

    assert schedules is not None
    assert schedules[0].shape == (2, MAXIMUM_WINDOWS_PER_HORIZON)


def test_motor_time_constant_can_be_held_fixed(quadrotor_flight) -> None:
    windows = trajectory_windows(
        [quadrotor_flight(3, 0.5)],
        horizon=5,
        stride=5,
    )

    result = fit_dynamics(
        [windows],
        initial_parameter_guess(),
        steps=5,
        fixed_motor_time_constant_s=0.001,
    )

    assert float(result.params.physical()["motor_time_constant"]) == pytest.approx(
        0.001
    )


def test_normalized_motor_commands_support_shared_thrust_offset(
    quadrotor_flight,
) -> None:
    windows = trajectory_windows(
        [quadrotor_flight(15, 0.5)],
        horizon=5,
        stride=5,
    )

    assert (
        supports_multirotor_thrust_command_offset(initial_parameter_guess(), windows)
        is True
    )

    result = fit_dynamics(
        [windows],
        initial_parameter_guess(),
        steps=3,
    )

    assert float(result.params.physical()["thrust_command_offset"]) == 0.0


def test_normalized_motor_command_offset_is_recoverable() -> None:
    hidden = with_thrust_command_offset(true_parameters(), -0.1)
    windows = trajectory_windows(
        [generate_trajectory(seed=30, duration_s=2.0, params=hidden)],
        horizon=10,
        stride=10,
    )

    result = fit_dynamics(
        [windows],
        initial_parameter_guess(),
        steps=20,
        learning_rate=0.03,
        learn_thrust_command_offset=True,
    )

    assert float(result.params.physical()["thrust_command_offset"]) < -0.05
    assert result.final_loss < 0.05 * result.initial_loss


def test_squared_rotor_speed_proxy_fixes_thrust_offset_to_zero(
    quadrotor_flight,
) -> None:
    trajectory = replace(
        quadrotor_flight(16, 0.5),
        spec=nanodrone_trajectory_spec(),
        observations=np.zeros((26, 3)),
    )
    windows = trajectory_windows([trajectory], horizon=5, stride=5)

    assert (
        supports_multirotor_thrust_command_offset(initial_parameter_guess(), windows)
        is False
    )

    result = fit_dynamics(
        [windows],
        initial_parameter_guess(),
        steps=3,
    )

    assert float(result.params.physical()["thrust_command_offset"]) == 0.0


def test_diagonal_angular_control_holds_the_mixer_on_its_canonical_axes(
    quadrotor_flight,
) -> None:
    windows = trajectory_windows(
        [quadrotor_flight(13, 0.5)],
        horizon=5,
        stride=5,
    )

    held = fit_dynamics(
        [windows],
        initial_parameter_guess(),
        steps=3,
        diagonal_angular_control=True,
    )
    learned = fit_dynamics([windows], initial_parameter_guess(), steps=3)

    np.testing.assert_allclose(
        held.params.physical()["angular_control_cross_coupling"], 0.0, atol=1e-8
    )
    assert not np.allclose(
        learned.params.physical()["angular_control_cross_coupling"], 0.0, atol=1e-8
    )


def test_rollout_loss_configuration_ignores_world_position_origin(
    quadrotor_flight,
) -> None:
    windows = trajectory_windows([quadrotor_flight(8, 0.5)], horizon=5, stride=5)
    initial_states = np.asarray(windows.initial_states).copy()
    target_states = np.asarray(windows.target_states).copy()
    initial_states[:, 0:3] += np.asarray([100.0, -20.0, 7.0])
    target_states[..., 0:3] += np.asarray([100.0, -20.0, 7.0])
    translated = replace(
        windows,
        initial_states=initial_states,
        target_states=target_states,
    )

    original = rollout_loss_configuration([windows])
    shifted = rollout_loss_configuration([translated])

    np.testing.assert_allclose(original.position_scale_m, shifted.position_scale_m)
    np.testing.assert_allclose(
        original.validity_envelope.body_velocity_half_width_m_s,
        shifted.validity_envelope.body_velocity_half_width_m_s,
    )


def test_fit_records_configured_long_rollout_policy(quadrotor_flight) -> None:
    windows = trajectory_windows([quadrotor_flight(2, 0.4)], horizon=5, stride=5)

    result = fit_dynamics(
        [windows],
        initial_parameter_guess(),
        steps=1,
        endpoint_weight=2.5,
        stability_regularization=0.02,
    )

    assert result.loss_configuration is not None
    assert result.loss_configuration.endpoint_weight == pytest.approx(2.5)
    assert result.loss_configuration.stability_regularization == pytest.approx(0.02)
    assert np.all(result.loss_configuration.position_scale_m > 0.0)


def test_dynamic_envelope_penalizes_velocity_escape(quadrotor_flight) -> None:
    windows = trajectory_windows([quadrotor_flight(6, 0.4)], horizon=5, stride=5)
    configuration = rollout_loss_configuration([windows])
    states = jnp.asarray(windows.target_states[:, 1:])
    escaped = states.at[..., 3].add(
        10.0 * configuration.validity_envelope.body_velocity_half_width_m_s[0]
    )

    nominal_penalty = dynamic_envelope_penalty(states, configuration)
    escaped_penalty = dynamic_envelope_penalty(escaped, configuration)

    assert float(jnp.mean(escaped_penalty)) > float(jnp.mean(nominal_penalty)) + 1.0


@pytest.mark.parametrize("coordinate", range(6))
def test_training_and_runtime_use_the_same_support_in_body_coordinates(
    quadrotor_flight, coordinate
) -> None:
    flight = quadrotor_flight(6, 0.4)
    windows = trajectory_windows([flight], horizon=5, stride=5)
    envelope = ModelValidityEnvelope(
        (1.0, -2.0, 0.5), (2.0, 3.0, 4.0), (0.1, -0.2, 0.3), (0.5, 0.6, 0.7)
    )
    configuration = replace(
        rollout_loss_configuration([windows]), validity_envelope=envelope
    )
    # A 180-degree yaw makes world and body velocities differ without
    # introducing roundoff from an approximately represented quaternion.
    rotation = np.diag([-1.0, -1.0, 1.0])
    center = np.r_[
        envelope.body_velocity_center_m_s, envelope.angular_velocity_center_rad_s
    ]
    widths = np.r_[
        envelope.body_velocity_half_width_m_s,
        envelope.angular_velocity_half_width_rad_s,
    ]
    for fraction in (0.5, 2.0):
        local = center.copy()
        local[coordinate] -= fraction * widths[coordinate]
        state = jnp.asarray(
            np.r_[np.zeros(3), rotation @ local[:3], 0, 0, 0, 1, local[3:]]
        )
        utilization = model_validity_utilization(
            state, jnp.empty(0), flight.spec, envelope
        )
        penalty = dynamic_envelope_penalty(state[None, None, :], configuration)

        assert float(jnp.max(utilization)) == pytest.approx(fraction)
        assert float(penalty[0, 0]) == pytest.approx(
            max(fraction - 1.0, 0.0) ** 2 / 6.0, abs=1e-7
        )


def test_residual_parameters_can_be_fit_through_rollouts(quadrotor_flight) -> None:
    windows = trajectory_windows([quadrotor_flight(5, 0.4)], horizon=5, stride=5)
    statistics = residual_initialization_statistics([windows])
    initial = initial_residual_parameters(
        initial_parameter_guess(), hidden_units=4, **statistics
    )

    result = fit_dynamics([windows], initial, steps=3, learning_rate=0.01)

    assert jnp.linalg.norm(result.params.output_weights) > 0.0
    np.testing.assert_allclose(result.params.feature_mean, initial.feature_mean)
    np.testing.assert_allclose(result.params.feature_scale, initial.feature_scale)
    np.testing.assert_allclose(result.params.correction_scale, initial.correction_scale)


def test_quadrotor_fit_rejects_non_quadrotor_control_schema(quadrotor_flight) -> None:
    trajectory = quadrotor_flight(7, 0.4)
    six_channel = replace(
        trajectory,
        controls=jnp.zeros((len(trajectory.controls), 6)),
        spec=make_trajectory_spec(
            ("throttle", "aileron", "elevator", "rudder", "flap", "spoiler"),
            family="fixedwing",
            observation_source="simulator_truth",
        ),
    )
    windows = trajectory_windows([six_channel], horizon=5)

    with pytest.raises(ValueError, match="requires ordered control roles"):
        fit_dynamics([windows], initial_parameter_guess(), steps=1)


@pytest.mark.parametrize("operation", ("fit", "residual_statistics"))
def test_training_horizons_must_share_input_units(quadrotor_flight, operation) -> None:
    trajectory = quadrotor_flight(7, 0.4)
    channels = list(trajectory.spec.channels)
    channels[0] = replace(channels[0], unit="rad/s")
    different_units = replace(
        trajectory, spec=replace(trajectory.spec, channels=tuple(channels))
    )
    window_sets = [
        trajectory_windows([trajectory], horizon=5),
        trajectory_windows([different_units], horizon=10),
    ]

    with pytest.raises(ValueError, match=r"windows:.*controls units"):
        if operation == "fit":
            fit_dynamics(window_sets, initial_parameter_guess(), steps=1)
        else:
            residual_initialization_statistics(window_sets)


def test_minibatch_realizes_window_weights_exactly_once(quadrotor_flight) -> None:
    from dataclasses import replace

    from glassbox.core.identification import (
        _window_loss,
        deterministic_weighted_batch_schedule,
    )

    trajectories = [quadrotor_flight(seed, 0.4) for seed in range(2)]
    windows = trajectory_windows(trajectories, horizon=5, stride=5)
    count = len(windows.initial_states)
    weights = np.where(np.arange(count) < count // 2, 1.0, 3.0)
    windows = replace(windows, window_weights=weights)
    configuration = rollout_loss_configuration([windows])
    params = initial_parameter_guess()

    per_window = np.asarray(
        [
            float(
                _window_loss(
                    params, windows, configuration, indices=jnp.asarray([index])
                )
            )
            for index in range(count)
        ]
    )
    weighted_full = float(_window_loss(params, windows, configuration))
    assert weighted_full == pytest.approx(
        float(np.sum(weights * per_window) / np.sum(weights)), rel=1e-5
    )

    schedule = deterministic_weighted_batch_schedule(
        weights, window_count=count, steps=300, maximum_batch_size=max(1, count // 3)
    )
    for row in schedule[:3]:
        # The schedule already draws windows in proportion to weight, so each
        # sampled batch is averaged uniformly rather than weighted again.
        assert float(
            _window_loss(params, windows, configuration, indices=jnp.asarray(row))
        ) == pytest.approx(float(np.mean(per_window[row])), rel=1e-5)
    realized = float(np.mean(per_window[schedule]))
    squared_weight_mean = float(np.sum(weights**2 * per_window) / np.sum(weights**2))
    assert realized == pytest.approx(weighted_full, rel=0.02)
    assert abs(realized - weighted_full) < abs(realized - squared_weight_mean)


def test_fit_reports_divergence_and_returns_finite_parameters(quadrotor_flight) -> None:
    windows = trajectory_windows([quadrotor_flight(5, 0.4)], horizon=5, stride=5)

    result = fit_dynamics(
        [windows], initial_parameter_guess(), steps=20, learning_rate=50.0
    )

    assert result.diverged
    assert result.completed_steps is not None and result.completed_steps < 20
    assert all(
        bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree.leaves(result.params)
    )
    assert np.isfinite(result.final_loss)
    assert np.all(np.isfinite(result.loss_history))
    assert len(result.loss_history) == result.completed_steps + 2
