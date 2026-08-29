"""Held-out rollout metrics and parameter reporting."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from glassbox.data import Trajectory, trajectory_windows
from glassbox.dynamics import (
    DynamicsParams,
    FixedWingDynamicsParams,
    ModelParams,
    ResidualDynamicsParams,
    control_state_after_history,
    physics_parameters,
    rollout_with_latent,
    structured_parameters,
    validate_control_schema,
)


DIVERGENCE_ERROR_THRESHOLDS = {
    "position_error_m": 10.0,
    "velocity_error_m_s": 5.0,
    "attitude_error_deg": 45.0,
    "angular_velocity_error_rad_s": 1.0,
}


def parameter_dict(params: ModelParams) -> dict[str, Any]:
    """Convert physical parameter arrays to JSON-compatible values."""

    base = structured_parameters(params)
    if isinstance(base, FixedWingDynamicsParams):
        physical = base.physical()
        result: dict[str, Any] = {
            "thrust_accel": float(physical["thrust_accel"]),
            "lift_accel_per_speed_sq": float(
                physical["lift_accel_per_speed_sq"]
            ),
            "lift_alpha_accel_per_speed_sq": float(
                physical["lift_alpha_accel_per_speed_sq"]
            ),
            "drag_accel_per_speed_sq": float(
                physical["drag_accel_per_speed_sq"]
            ),
            "side_force_accel_per_speed": float(
                physical["side_force_accel_per_speed"]
            ),
            "surface_angular_accel_per_speed_sq": np.asarray(
                physical["surface_angular_accel_per_speed_sq"]
            ).tolist(),
            "lateral_surface_cross_angular_accel_per_speed_sq": np.asarray(
                physical[
                    "lateral_surface_cross_angular_accel_per_speed_sq"
                ]
            ).tolist(),
            "pitch_stability_accel_per_speed_sq": float(
                physical["pitch_stability_accel_per_speed_sq"]
            ),
            "lateral_stability_angular_accel_per_speed_sq": np.asarray(
                physical["lateral_stability_angular_accel_per_speed_sq"]
            ).tolist(),
            "angular_drag_per_speed": np.asarray(
                physical["angular_drag_per_speed"]
            ).tolist(),
            "surface_trim": np.asarray(physical["surface_trim"]).tolist(),
            "flap_lift_accel_per_speed_sq": float(
                physical["flap_lift_accel_per_speed_sq"]
            ),
            "flap_drag_accel_per_speed_sq": float(
                physical["flap_drag_accel_per_speed_sq"]
            ),
            "flap_pitch_angular_accel_per_speed_sq": float(
                physical["flap_pitch_angular_accel_per_speed_sq"]
            ),
            "flap_trim": float(physical["flap_trim"]),
            "actuator_time_constant": float(
                physical["actuator_time_constant"]
            ),
        }
    else:
        physical = physics_parameters(params).physical()
        result = {
            "thrust_accel": float(physical["thrust_accel"]),
            "angular_accel": np.asarray(physical["angular_accel"]).tolist(),
            "linear_drag": float(physical["linear_drag"]),
            "angular_drag": np.asarray(physical["angular_drag"]).tolist(),
            "motor_time_constant": float(physical["motor_time_constant"]),
            "angular_response_time_constant": np.asarray(
                physical["angular_response_time_constant"]
            ).tolist(),
            "angular_control_cross_coupling": np.asarray(
                physical["angular_control_cross_coupling"]
            ).tolist(),
        }
    if isinstance(params, ResidualDynamicsParams):
        result["residual"] = {
            "input_features": int(params.feature_mean.shape[0]),
            "hidden_units": int(params.hidden_weights.shape[0]),
            "output_accelerations": 6,
            "hidden_weight_norm": float(np.linalg.norm(params.hidden_weights)),
            "output_weight_norm": float(np.linalg.norm(params.output_weights)),
            "feature_mean": np.asarray(params.feature_mean).tolist(),
            "feature_scale": np.asarray(params.feature_scale).tolist(),
            "correction_scale": np.asarray(params.correction_scale).tolist(),
            "frame": "body",
            "bounded_output": True,
        }
    return result


def rollout_predictions(
    params: ModelParams,
    trajectory: Trajectory,
    *,
    control_history: np.ndarray | None = None,
) -> np.ndarray:
    """Return one complete logged-input prediction from the measured start."""

    validate_control_schema(
        params,
        trajectory.control_names,
        trajectory.spec.control_roles,
    )

    initial_motor_state = None
    if control_history is not None:
        if len(control_history) < 1:
            raise ValueError("control_history must be nonempty when provided")
        initial_motor_state = control_state_after_history(
            params,
            jnp.asarray(control_history),
            trajectory.nominal_dt_s,
            trajectory.spec.control_roles,
        )
    return np.asarray(
        rollout_with_latent(
            params,
            jnp.asarray(trajectory.states[0]),
            jnp.asarray(trajectory.controls),
            trajectory.nominal_dt_s,
            initial_motor_state,
            trajectory.spec.control_roles,
            jnp.asarray(trajectory.exogenous[0]),
            trajectory.spec.exogenous_roles,
        )[0],
        dtype=np.float64,
    )


def rollout_metrics(
    params: ModelParams,
    trajectory: Trajectory,
    *,
    control_history: np.ndarray | None = None,
) -> dict[str, Any]:
    """Evaluate one full logged-input rollout against a trajectory."""

    predicted = rollout_predictions(
        params,
        trajectory,
        control_history=control_history,
    )
    return _state_error_metrics(
        predicted,
        trajectory.states,
        duration_s=float(trajectory.time_s[-1] - trajectory.time_s[0]),
    )


def rollout_divergence_metrics(
    params: ModelParams,
    trajectory: Trajectory,
    *,
    control_history: np.ndarray | None = None,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Report when a complete rollout first exceeds a useful error envelope."""

    selected_thresholds = dict(DIVERGENCE_ERROR_THRESHOLDS)
    if thresholds is not None:
        unknown = set(thresholds) - set(selected_thresholds)
        if unknown:
            raise ValueError(
                "unknown divergence threshold(s): " + ", ".join(sorted(unknown))
            )
        selected_thresholds.update(thresholds)
    if any(
        not np.isfinite(value) or value <= 0.0
        for value in selected_thresholds.values()
    ):
        raise ValueError("divergence thresholds must be finite and positive")

    predicted = rollout_predictions(
        params,
        trajectory,
        control_history=control_history,
    )
    target = trajectory.states
    predicted_quaternion_norm = np.linalg.norm(predicted[:, 6:10], axis=1)
    finite = np.all(np.isfinite(predicted), axis=1) & (
        predicted_quaternion_norm > 0.0
    )
    position_error = np.linalg.norm(predicted[:, 0:3] - target[:, 0:3], axis=1)
    velocity_error = np.linalg.norm(predicted[:, 3:6] - target[:, 3:6], axis=1)
    angular_velocity_error = np.linalg.norm(
        predicted[:, 10:13] - target[:, 10:13], axis=1
    )
    attitude_error = np.full(len(predicted), np.inf, dtype=np.float64)
    if np.any(finite):
        predicted_quaternion = predicted[finite, 6:10]
        target_quaternion = target[finite, 6:10]
        predicted_quaternion /= np.linalg.norm(
            predicted_quaternion, axis=1, keepdims=True
        )
        target_quaternion /= np.linalg.norm(
            target_quaternion, axis=1, keepdims=True
        )
        quaternion_dot = np.clip(
            np.abs(np.sum(predicted_quaternion * target_quaternion, axis=1)),
            0.0,
            1.0,
        )
        attitude_error[finite] = np.rad2deg(2.0 * np.arccos(quaternion_dot))

    traces = {
        "position_error_m": position_error,
        "velocity_error_m_s": velocity_error,
        "attitude_error_deg": attitude_error,
        "angular_velocity_error_rad_s": angular_velocity_error,
    }
    crossed = ~finite
    for name, values in traces.items():
        crossed |= values > selected_thresholds[name]
    crossing_indices = np.flatnonzero(crossed)
    divergence_index = (
        None if len(crossing_indices) == 0 else int(crossing_indices[0])
    )
    if divergence_index is None:
        causes: list[str] = []
        stable_through_s = float(trajectory.time_s[-1] - trajectory.time_s[0])
    else:
        causes = []
        if not finite[divergence_index]:
            causes.append("non_finite_state")
        for name, values in traces.items():
            if values[divergence_index] > selected_thresholds[name]:
                causes.append(name)
        stable_index = max(0, divergence_index - 1)
        stable_through_s = float(
            trajectory.time_s[stable_index] - trajectory.time_s[0]
        )
    duration_s = float(trajectory.time_s[-1] - trajectory.time_s[0])
    return {
        "thresholds": selected_thresholds,
        "full_rollout_finite": bool(np.all(finite)),
        "first_nonfinite_time_s": (
            None
            if np.all(finite)
            else float(
                trajectory.time_s[int(np.flatnonzero(~finite)[0])]
                - trajectory.time_s[0]
            )
        ),
        "diverged": divergence_index is not None,
        "divergence_time_s": (
            None
            if divergence_index is None
            else float(trajectory.time_s[divergence_index] - trajectory.time_s[0])
        ),
        "divergence_causes": causes,
        "stable_through_s": stable_through_s,
        "stable_fraction": stable_through_s / duration_s,
        "duration_s": duration_s,
        "final_errors": {
            name: (float(values[-1]) if np.isfinite(values[-1]) else None)
            for name, values in traces.items()
        },
    }


def _state_error_metrics(
    predicted: np.ndarray,
    target: np.ndarray,
    *,
    duration_s: float,
) -> dict[str, Any]:
    """Calculate metrics for one rollout or a batch of rollout windows."""

    position_error = predicted[..., 0:3] - target[..., 0:3]
    velocity_error = predicted[..., 3:6] - target[..., 3:6]
    angular_velocity_error = predicted[..., 10:13] - target[..., 10:13]

    predicted_quaternion = predicted[..., 6:10] / np.linalg.norm(
        predicted[..., 6:10], axis=-1, keepdims=True
    )
    target_quaternion = target[..., 6:10] / np.linalg.norm(
        target[..., 6:10], axis=-1, keepdims=True
    )
    quaternion_dot = np.abs(
        np.sum(predicted_quaternion * target_quaternion, axis=-1)
    )
    quaternion_dot = np.clip(quaternion_dot, -1.0, 1.0)
    attitude_error_deg = np.rad2deg(2.0 * np.arccos(quaternion_dot))

    # Express target^-1 * prediction as a shortest-path rotation vector. Its
    # components expose which body rotation axes dominate attitude drift.
    target_w = target_quaternion[..., 0]
    target_xyz = -target_quaternion[..., 1:4]
    predicted_w = predicted_quaternion[..., 0]
    predicted_xyz = predicted_quaternion[..., 1:4]
    relative_w = target_w * predicted_w - np.sum(
        target_xyz * predicted_xyz, axis=-1
    )
    relative_xyz = (
        target_w[..., np.newaxis] * predicted_xyz
        + predicted_w[..., np.newaxis] * target_xyz
        + np.cross(target_xyz, predicted_xyz)
    )
    relative_sign = np.where(relative_w < 0.0, -1.0, 1.0)
    relative_w *= relative_sign
    relative_xyz *= relative_sign[..., np.newaxis]
    relative_vector_norm = np.linalg.norm(relative_xyz, axis=-1)
    relative_angle = 2.0 * np.arctan2(
        relative_vector_norm, np.clip(relative_w, 0.0, 1.0)
    )
    rotation_scale = np.divide(
        relative_angle,
        relative_vector_norm,
        out=np.full_like(relative_angle, 2.0),
        where=relative_vector_norm > 1e-12,
    )
    attitude_rotation_vector_deg = np.rad2deg(
        relative_xyz * rotation_scale[..., np.newaxis]
    )

    reduction_axes = tuple(range(position_error.ndim - 1))

    rollout_count = 1 if predicted.ndim == 2 else int(predicted.shape[0])
    sample_count = int(np.prod(predicted.shape[:-1]))
    endpoint_position_error = np.linalg.norm(
        position_error[..., -1, :] if predicted.ndim > 2 else position_error[-1],
        axis=-1,
    )
    return {
        "position_rmse_m": float(np.sqrt(np.mean(np.square(position_error)))),
        "position_rmse_xyz_m": np.sqrt(
            np.mean(np.square(position_error), axis=reduction_axes)
        ).tolist(),
        "velocity_rmse_m_s": float(np.sqrt(np.mean(np.square(velocity_error)))),
        "velocity_rmse_xyz_m_s": np.sqrt(
            np.mean(np.square(velocity_error), axis=reduction_axes)
        ).tolist(),
        "attitude_rmse_deg": float(
            np.sqrt(np.mean(np.square(attitude_error_deg)))
        ),
        "angular_velocity_rmse_rad_s": float(
            np.sqrt(np.mean(np.square(angular_velocity_error)))
        ),
        "angular_velocity_rmse_xyz_rad_s": np.sqrt(
            np.mean(np.square(angular_velocity_error), axis=reduction_axes)
        ).tolist(),
        "attitude_rotation_vector_rmse_xyz_deg": np.sqrt(
            np.mean(np.square(attitude_rotation_vector_deg), axis=reduction_axes)
        ).tolist(),
        "final_position_error_m": float(
            np.sqrt(np.mean(np.square(endpoint_position_error)))
        ),
        "duration_s": duration_s,
        "sample_count": sample_count,
        "rollout_count": rollout_count,
    }


def windowed_rollout_metrics(
    params: ModelParams,
    trajectory: Trajectory,
    *,
    horizon_steps: int,
    stride_steps: int | None = None,
) -> dict[str, Any]:
    """Evaluate fixed-horizon rollouts initialized throughout one flight."""

    predicted, target, dt_s = windowed_rollout_predictions(
        params,
        trajectory,
        horizon_steps=horizon_steps,
        stride_steps=stride_steps,
    )
    return _state_error_metrics(
        predicted,
        target,
        duration_s=horizon_steps * dt_s,
    )


def kinematic_persistence_windowed_metrics(
    trajectory: Trajectory,
    *,
    horizon_steps: int,
    stride_steps: int | None = None,
) -> dict[str, Any]:
    """Score a constant-world-velocity and constant-body-rate baseline."""

    windows = trajectory_windows(
        [trajectory],
        horizon=horizon_steps,
        stride=horizon_steps if stride_steps is None else stride_steps,
    )
    predicted = np.empty_like(windows.target_states)
    predicted[:, 0] = windows.initial_states
    state = windows.initial_states.copy()
    for index in range(horizon_steps):
        quaternion = state[:, 6:10]
        angular_velocity = state[:, 10:13]
        w, x, y, z = np.moveaxis(quaternion, -1, 0)
        wx, wy, wz = np.moveaxis(angular_velocity, -1, 0)
        quaternion_rate = 0.5 * np.column_stack(
            (
                -x * wx - y * wy - z * wz,
                w * wx + y * wz - z * wy,
                w * wy - x * wz + z * wx,
                w * wz + x * wy - y * wx,
            )
        )
        state[:, 0:3] += windows.dt_s * state[:, 3:6]
        state[:, 6:10] += windows.dt_s * quaternion_rate
        state[:, 6:10] /= np.linalg.norm(
            state[:, 6:10], axis=1, keepdims=True
        )
        predicted[:, index + 1] = state
    return _state_error_metrics(
        predicted,
        windows.target_states,
        duration_s=horizon_steps * windows.dt_s,
    )


def windowed_rollout_predictions(
    params: ModelParams,
    trajectory: Trajectory,
    *,
    horizon_steps: int,
    stride_steps: int | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return predicted and measured fixed-horizon state windows.

    Both arrays include the measured initial state at index zero and have shape
    ``(window, horizon + 1, 13)``. Use ``stride_steps=1`` for rolling benchmark
    protocols that initialize a prediction at every admissible sample.
    """

    validate_control_schema(
        params,
        trajectory.control_names,
        trajectory.spec.control_roles,
    )

    windows = trajectory_windows(
        [trajectory],
        horizon=horizon_steps,
        stride=horizon_steps if stride_steps is None else stride_steps,
    )
    initial_motor_states = jax.vmap(
        lambda history: control_state_after_history(
            params, history, windows.dt_s, windows.control_roles
        )
    )(jnp.asarray(windows.control_histories))
    predicted, _ = jax.vmap(
        lambda initial, control_sequence, initial_control, context: rollout_with_latent(
            params,
            initial,
            control_sequence,
            windows.dt_s,
            initial_control,
            windows.control_roles,
            context,
            windows.exogenous_roles,
        )
    )(
        jnp.asarray(windows.initial_states),
        jnp.asarray(windows.controls),
        initial_motor_states,
        jnp.asarray(windows.initial_exogenous),
    )
    return (
        np.asarray(predicted, dtype=np.float64),
        windows.target_states,
        windows.dt_s,
    )


def aggregate_rollout_metrics(
    metrics: list[dict[str, Any]],
    *,
    weighting: str = "sample",
) -> dict[str, Any]:
    """Aggregate compatible metrics using sample or equal-item weighting."""

    if not metrics:
        raise ValueError("at least one metric set is required")
    if weighting not in {"sample", "equal"}:
        raise ValueError("weighting must be 'sample' or 'equal'")

    sample_count = sum(int(item["sample_count"]) for item in metrics)
    rollout_count = sum(int(item["rollout_count"]) for item in metrics)
    sample_weights = (
        [int(item["sample_count"]) for item in metrics]
        if weighting == "sample"
        else [1] * len(metrics)
    )
    endpoint_weights = (
        [int(item["rollout_count"]) for item in metrics]
        if weighting == "sample"
        else [1] * len(metrics)
    )
    sample_metrics = (
        "position_rmse_m",
        "velocity_rmse_m_s",
        "attitude_rmse_deg",
        "angular_velocity_rmse_rad_s",
    )
    result = {
        name: float(
            np.sqrt(
                sum(
                    float(item[name]) ** 2 * weight
                    for item, weight in zip(metrics, sample_weights)
                )
                / sum(sample_weights)
            )
        )
        for name in sample_metrics
    }
    vector_metrics = (
        "position_rmse_xyz_m",
        "velocity_rmse_xyz_m_s",
        "attitude_rotation_vector_rmse_xyz_deg",
        "angular_velocity_rmse_xyz_rad_s",
    )
    for name in vector_metrics:
        squared = sum(
            np.square(np.asarray(item[name], dtype=np.float64)) * weight
            for item, weight in zip(metrics, sample_weights)
        ) / sum(sample_weights)
        result[name] = np.sqrt(squared).tolist()
    result.update(
        {
            "final_position_error_m": float(
                np.sqrt(
                    sum(
                        float(item["final_position_error_m"]) ** 2
                        * weight
                        for item, weight in zip(metrics, endpoint_weights)
                    )
                    / sum(endpoint_weights)
                )
            ),
            "duration_s": float(sum(float(item["duration_s"]) for item in metrics)),
            "sample_count": sample_count,
            "rollout_count": rollout_count,
            "weighting": weighting,
        }
    )
    return result
