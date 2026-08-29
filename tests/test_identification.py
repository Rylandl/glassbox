import pytest
import jax.numpy as jnp
import numpy as np
from dataclasses import replace

from glassbox.data import make_trajectory_spec, trajectory_windows
from glassbox.dynamics import initial_residual_parameters
from glassbox.identification import (
    MAX_OPTIMIZATION_WINDOWS_PER_HORIZON,
    OPTIMIZATION_POLICY_VERSION,
    _optimization_batch_schedules,
    deterministic_weighted_batch_schedule,
    dynamic_envelope_penalty,
    fit_dynamics,
    residual_initialization_statistics,
    rollout_loss_configuration,
)
from glassbox.synthetic import (
    generate_trajectory,
    initial_parameter_guess,
)


def test_multistep_fit_reduces_training_loss() -> None:
    trajectories = [
        generate_trajectory(seed=0, duration_s=2.0),
        generate_trajectory(seed=1, duration_s=2.0),
    ]
    windows = trajectory_windows(trajectories, horizon=10, stride=10)

    result = fit_dynamics(
        windows,
        initial_parameter_guess(),
        steps=80,
        learning_rate=0.03,
    )

    assert result.final_loss < 0.25 * result.initial_loss


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


def test_affordable_fit_uses_every_window() -> None:
    windows = trajectory_windows(
        [generate_trajectory(seed=11, duration_s=11.0)],
        horizon=1,
        stride=1,
    )

    result = fit_dynamics(
        windows,
        initial_parameter_guess(),
        steps=1,
        learning_rate=0.01,
    )

    assert result.optimization_policy == "full_batch_v1"
    assert result.batch_sizes == ()
    assert result.window_coverage == ()
    assert np.isfinite(result.initial_loss)
    assert np.isfinite(result.final_loss)


def test_automatic_minibatch_caps_large_short_horizon_window_set() -> None:
    windows = trajectory_windows(
        [generate_trajectory(seed=12, duration_s=11.0)],
        horizon=1,
        stride=1,
    )
    repeated_count = MAX_OPTIMIZATION_WINDOWS_PER_HORIZON + 10
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
    assert schedules[0].shape == (2, MAX_OPTIMIZATION_WINDOWS_PER_HORIZON)


def test_motor_time_constant_can_be_held_fixed() -> None:
    windows = trajectory_windows(
        [generate_trajectory(seed=3, duration_s=0.5)],
        horizon=5,
        stride=5,
    )

    result = fit_dynamics(
        windows,
        initial_parameter_guess(),
        steps=5,
        fixed_motor_time_constant_s=0.001,
    )

    assert float(result.params.physical()["motor_time_constant"]) == pytest.approx(
        0.001
    )


def test_rotational_response_ablation_is_held_instantaneous_and_diagonal() -> None:
    windows = trajectory_windows(
        [generate_trajectory(seed=13, duration_s=0.5)],
        horizon=5,
        stride=5,
    )

    result = fit_dynamics(
        windows,
        initial_parameter_guess(),
        steps=3,
        instantaneous_rotational_response=True,
    )
    physical = result.params.physical()

    np.testing.assert_allclose(
        physical["angular_response_time_constant"], 1e-4, rtol=1e-6
    )
    np.testing.assert_allclose(
        physical["angular_control_cross_coupling"], 0.0, atol=1e-8
    )


def test_diagonal_control_fit_still_learns_rotational_memory() -> None:
    windows = trajectory_windows(
        [generate_trajectory(seed=14, duration_s=0.5)],
        horizon=5,
        stride=5,
    )

    result = fit_dynamics(
        windows,
        initial_parameter_guess(),
        steps=3,
        diagonal_angular_control=True,
    )
    physical = result.params.physical()

    np.testing.assert_allclose(
        physical["angular_control_cross_coupling"], 0.0, atol=1e-8
    )
    assert not np.allclose(
        physical["angular_response_time_constant"],
        initial_parameter_guess().physical()["angular_response_time_constant"],
    )


def test_rollout_loss_configuration_ignores_world_position_origin() -> None:
    windows = trajectory_windows(
        [generate_trajectory(seed=8, duration_s=0.5)], horizon=5, stride=5
    )
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

    np.testing.assert_allclose(
        original.position_scale_m, shifted.position_scale_m
    )
    np.testing.assert_allclose(
        original.body_velocity_bound_m_s,
        shifted.body_velocity_bound_m_s,
    )


def test_fit_records_configured_long_rollout_policy() -> None:
    windows = trajectory_windows(
        [generate_trajectory(seed=2, duration_s=0.4)], horizon=5, stride=5
    )

    result = fit_dynamics(
        windows,
        initial_parameter_guess(),
        steps=1,
        endpoint_weight=2.5,
        stability_regularization=0.02,
    )

    assert result.loss_configuration is not None
    assert result.loss_configuration.endpoint_weight == pytest.approx(2.5)
    assert result.loss_configuration.stability_regularization == pytest.approx(
        0.02
    )
    assert np.all(result.loss_configuration.position_scale_m > 0.0)


def test_dynamic_envelope_penalizes_velocity_escape() -> None:
    windows = trajectory_windows(
        [generate_trajectory(seed=6, duration_s=0.4)], horizon=5, stride=5
    )
    configuration = rollout_loss_configuration([windows])
    states = jnp.asarray(windows.target_states[:, 1:])
    escaped = states.at[..., 3].add(
        10.0 * configuration.body_velocity_bound_m_s[0]
    )

    nominal_penalty = dynamic_envelope_penalty(states, configuration)
    escaped_penalty = dynamic_envelope_penalty(escaped, configuration)

    assert float(jnp.mean(escaped_penalty)) > float(
        jnp.mean(nominal_penalty)
    ) + 1.0


def test_residual_parameters_can_be_fit_through_rollouts() -> None:
    windows = trajectory_windows(
        [generate_trajectory(seed=5, duration_s=0.4)], horizon=5, stride=5
    )
    statistics = residual_initialization_statistics([windows])
    initial = initial_residual_parameters(
        initial_parameter_guess(), hidden_units=4, **statistics
    )

    result = fit_dynamics(windows, initial, steps=3, learning_rate=0.01)

    assert jnp.linalg.norm(result.params.output_weights) > 0.0
    np.testing.assert_allclose(result.params.feature_mean, initial.feature_mean)
    np.testing.assert_allclose(result.params.feature_scale, initial.feature_scale)
    np.testing.assert_allclose(
        result.params.correction_scale, initial.correction_scale
    )


def test_quadrotor_fit_rejects_non_quadrotor_control_schema() -> None:
    trajectory = generate_trajectory(seed=7, duration_s=0.4)
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
        fit_dynamics(windows, initial_parameter_guess(), steps=1)
