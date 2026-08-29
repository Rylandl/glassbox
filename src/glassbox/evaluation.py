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
    step_with_latent,
    structured_parameters,
    validate_control_schema,
)


DIVERGENCE_ERROR_THRESHOLDS = {
    "position_error_m": 10.0,
    "velocity_error_m_s": 5.0,
    "attitude_error_deg": 45.0,
    "angular_velocity_error_rad_s": 1.0,
}
ROLLOUT_METRICS = (
    "position_rmse_m",
    "velocity_rmse_m_s",
    "attitude_rmse_deg",
    "angular_velocity_rmse_rad_s",
)

# Ratios below these small, physically meaningful floors are treated as equal.
# This prevents numerical noise around zero from becoming a large regression.
METRIC_FLOORS = {
    "position_rmse_m": 1e-3,
    "velocity_rmse_m_s": 1e-3,
    "attitude_rmse_deg": 1e-2,
    "angular_velocity_rmse_rad_s": 1e-3,
}

INNOVATION_MAXIMUM_LAG_S = 0.5
INNOVATION_MAXIMUM_LAG_STEPS = 50
INNOVATION_SIMULTANEOUS_ALPHA = 0.05
INNOVATION_MINIMUM_SAMPLES = 16
KINEMATIC_POSITION_RATE_FLOOR_M_S = 1e-3
KINEMATIC_ATTITUDE_RATE_FLOOR_RAD_S = 1e-3
INNOVATION_GROUPS = {
    "position": (0, 1, 2),
    "velocity": (3, 4, 5),
    "attitude": (6, 7, 8),
    "angular_velocity": (9, 10, 11),
}
INNOVATION_CHANNELS = (
    ("position_x_m", "m"),
    ("position_y_m", "m"),
    ("position_z_m", "m"),
    ("velocity_x_m_s", "m/s"),
    ("velocity_y_m_s", "m/s"),
    ("velocity_z_m_s", "m/s"),
    ("attitude_x_rad", "rad"),
    ("attitude_y_rad", "rad"),
    ("attitude_z_rad", "rad"),
    ("angular_velocity_x_rad_s", "rad/s"),
    ("angular_velocity_y_rad_s", "rad/s"),
    ("angular_velocity_z_rad_s", "rad/s"),
)


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
            "thrust_command_offset": float(
                physical["thrust_command_offset"]
            ),
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


def _attitude_innovation(
    predicted_wxyz: np.ndarray,
    observed_wxyz: np.ndarray,
) -> np.ndarray:
    """Return the shortest predicted-to-observed rotation vector."""

    predicted = predicted_wxyz / np.linalg.norm(
        predicted_wxyz, axis=1, keepdims=True
    )
    observed = observed_wxyz / np.linalg.norm(
        observed_wxyz, axis=1, keepdims=True
    )
    predicted_w = predicted[:, 0]
    predicted_xyz = predicted[:, 1:4]
    observed_w = observed[:, 0]
    observed_xyz = observed[:, 1:4]
    relative_w = predicted_w * observed_w + np.sum(
        predicted_xyz * observed_xyz, axis=1
    )
    relative_xyz = (
        predicted_w[:, None] * observed_xyz
        - observed_w[:, None] * predicted_xyz
        - np.cross(predicted_xyz, observed_xyz)
    )
    sign = np.where(relative_w < 0.0, -1.0, 1.0)
    relative_w *= sign
    relative_xyz *= sign[:, None]
    vector_norm = np.linalg.norm(relative_xyz, axis=1)
    angle = 2.0 * np.arctan2(
        vector_norm, np.clip(relative_w, 0.0, 1.0)
    )
    scale = np.divide(
        angle,
        vector_norm,
        out=np.full_like(angle, 2.0),
        where=vector_norm > 1e-12,
    )
    return relative_xyz * scale[:, None]


def _quaternion_rotation_matrices(quaternion_wxyz: np.ndarray) -> np.ndarray:
    quaternion = quaternion_wxyz / np.linalg.norm(
        quaternion_wxyz, axis=1, keepdims=True
    )
    w, x, y, z = quaternion.T
    return np.stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape((-1, 3, 3))


def state_kinematic_compatibility_diagnostics(
    trajectory: Trajectory,
) -> dict[str, Any]:
    """Measure whether independently observed rigid-body states agree.

    Position increments should match trapezoidal world velocity. Attitude
    increments should match trapezoidal body angular velocity after expressing
    the terminal rate in the initial body frame. These checks do not use a
    learned dynamics model and therefore expose a telemetry consistency floor.
    """

    dt_s = trajectory.nominal_dt_s
    states = trajectory.states
    position_rate_residual = (
        np.diff(states[:, 0:3], axis=0) / dt_s
        - 0.5 * (states[:-1, 3:6] + states[1:, 3:6])
    )
    attitude_increment = _attitude_innovation(
        states[:-1, 6:10], states[1:, 6:10]
    )
    rotations = _quaternion_rotation_matrices(states[:, 6:10])
    next_rate_in_initial_body = np.einsum(
        "nji,njk,nk->ni",
        rotations[:-1],
        rotations[1:],
        states[1:, 10:13],
    )
    attitude_rate_residual = (
        attitude_increment / dt_s
        - 0.5 * (states[:-1, 10:13] + next_rate_in_initial_body)
    )
    sample_count = len(position_rate_residual)
    common = {
        "policy": "trapezoidal_state_kinematic_compatibility_v1",
        "sample_count": sample_count,
        "future_state_used_for_diagnostic_only": True,
        "dynamics_model_used": False,
    }
    if sample_count < INNOVATION_MINIMUM_SAMPLES:
        return {
            **common,
            "status": "insufficient_samples",
            "minimum_sample_count": INNOVATION_MINIMUM_SAMPLES,
        }
    maximum_lag_steps = min(
        INNOVATION_MAXIMUM_LAG_STEPS,
        max(1, int(np.floor(INNOVATION_MAXIMUM_LAG_S / dt_s))),
        max(1, sample_count // 4),
    )
    bound = _simultaneous_correlation_bound(
        sample_count, 3 * maximum_lag_steps
    )

    def group_report(
        values: np.ndarray,
        unit: str,
        materiality_floor: float,
    ) -> dict[str, Any]:
        axes = {}
        for index, axis in enumerate("xyz"):
            correlation, lag = _strongest_autocorrelation(
                values[:, index], maximum_lag_steps
            )
            rmse = float(np.sqrt(np.mean(np.square(values[:, index]))))
            axes[axis] = {
                "unit": unit,
                "mean": float(np.mean(values[:, index])),
                "rmse": rmse,
                "materiality_floor": materiality_floor,
                "strongest_autocorrelation": correlation,
                "autocorrelation_lag_steps": lag,
                "autocorrelation_lag_s": lag * dt_s,
                "autocorrelation_bound": bound,
                "temporally_colored": bool(
                    rmse > materiality_floor and abs(correlation) > bound
                ),
            }
        return {
            "axes": axes,
            "vector_rmse": float(
                np.sqrt(np.mean(np.sum(np.square(values), axis=1)))
            ),
            "temporally_colored": any(
                bool(item["temporally_colored"]) for item in axes.values()
            ),
        }

    position = group_report(
        position_rate_residual,
        "m/s",
        KINEMATIC_POSITION_RATE_FLOOR_M_S,
    )
    attitude = group_report(
        attitude_rate_residual,
        "rad/s",
        KINEMATIC_ATTITUDE_RATE_FLOOR_RAD_S,
    )
    return {
        **common,
        "status": "ok",
        "maximum_lag_steps": maximum_lag_steps,
        "maximum_lag_s": maximum_lag_steps * dt_s,
        "position_velocity_compatibility": position,
        "attitude_rate_compatibility": attitude,
        "state_observations_temporally_inconsistent": bool(
            position["temporally_colored"] or attitude["temporally_colored"]
        ),
        "interpretation": (
            "This is a data-compatibility diagnostic, not a prediction score. "
            "Large or colored residuals mean pose and velocity/rate channels do "
            "not describe one exactly sampled rigid-body trajectory; model-error "
            "attribution must account for estimator and sensor behavior."
        ),
    }


def _measured_state_reset_predictions(
    params: ModelParams,
    trajectory: Trajectory,
    *,
    control_history: np.ndarray | None,
) -> np.ndarray:
    """Predict every next sample while carrying only the latent actuator state."""

    validate_control_schema(
        params,
        trajectory.control_names,
        trajectory.spec.control_roles,
    )
    history = (
        trajectory.controls[:1]
        if control_history is None
        else np.asarray(control_history, dtype=np.float64)
    )
    if history.ndim != 2 or history.shape[1] != trajectory.control_size:
        raise ValueError(
            "control_history must have the same channel count as the trajectory"
        )
    if len(history) < 1:
        raise ValueError("control_history must be nonempty when provided")
    initial_latent = control_state_after_history(
        params,
        jnp.asarray(history),
        trajectory.nominal_dt_s,
        trajectory.spec.control_roles,
    )

    def scan_step(
        latent_state: jax.Array,
        samples: tuple[jax.Array, jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array]:
        measured_state, control, exogenous = samples
        predicted_state, next_latent = step_with_latent(
            params,
            measured_state,
            latent_state,
            control,
            trajectory.nominal_dt_s,
            trajectory.spec.control_roles,
            exogenous,
            trajectory.spec.exogenous_roles,
        )
        return next_latent, predicted_state

    _, predictions = jax.lax.scan(
        scan_step,
        initial_latent,
        (
            jnp.asarray(trajectory.states[:-1]),
            jnp.asarray(trajectory.controls),
            jnp.asarray(trajectory.exogenous[:-1]),
        ),
    )
    return np.asarray(predictions, dtype=np.float64)


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if len(left) < 3:
        return 0.0
    left = left - np.mean(left)
    right = right - np.mean(right)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return 0.0 if denominator <= 1e-12 else float(left @ right / denominator)


def _simultaneous_correlation_bound(
    sample_count: int,
    comparison_count: int,
) -> float:
    """Return a conservative finite-comparison noise correlation envelope."""

    if sample_count < 1 or comparison_count < 1:
        return 1.0
    return min(
        1.0,
        float(
            np.sqrt(
                2.0
                * np.log(
                    2.0
                    * comparison_count
                    / INNOVATION_SIMULTANEOUS_ALPHA
                )
                / sample_count
            )
        ),
    )


def _strongest_autocorrelation(
    values: np.ndarray,
    maximum_lag_steps: int,
) -> tuple[float, int]:
    candidates = [
        (_correlation(values[:-lag], values[lag:]), lag)
        for lag in range(1, maximum_lag_steps + 1)
    ]
    return max(candidates, key=lambda item: abs(item[0]))


def _strongest_input_correlation(
    values: np.ndarray,
    controls: np.ndarray,
    maximum_lag_steps: int,
) -> tuple[float, int, int]:
    candidates = []
    for control_index in range(controls.shape[1]):
        control = controls[:, control_index]
        for lag in range(maximum_lag_steps + 1):
            correlation = _correlation(
                control[:-lag] if lag else control,
                values[lag:],
            )
            candidates.append((correlation, lag, control_index))
    return max(candidates, key=lambda item: abs(item[0]))


def one_step_innovation_diagnostics(
    params: ModelParams,
    trajectory: Trajectory,
    *,
    control_history: np.ndarray | None = None,
) -> dict[str, Any]:
    """Diagnose held-out local residual structure without rollout drift.

    Each interval starts from the measured rigid-body state while the model's
    latent actuator state is carried causally through the command sequence.
    Residual autocorrelation exposes omitted temporal state; correlation with
    current or past controls exposes unexplained input-response structure.
    """

    predicted = _measured_state_reset_predictions(
        params,
        trajectory,
        control_history=control_history,
    )
    observed = trajectory.states[1:]
    innovations = np.column_stack(
        (
            observed[:, 0:3] - predicted[:, 0:3],
            observed[:, 3:6] - predicted[:, 3:6],
            _attitude_innovation(predicted[:, 6:10], observed[:, 6:10]),
            observed[:, 10:13] - predicted[:, 10:13],
        )
    )
    controls = np.asarray(trajectory.controls, dtype=np.float64)
    finite = np.all(np.isfinite(innovations), axis=1) & np.all(
        np.isfinite(controls), axis=1
    )
    innovations = innovations[finite]
    controls = controls[finite]
    initialization_discard_steps = (
        0
        if control_history is not None
        else min(
            int(np.floor(0.5 / trajectory.nominal_dt_s)),
            len(innovations) // 10,
        )
    )
    innovations = innovations[initialization_discard_steps:]
    controls = controls[initialization_discard_steps:]
    sample_count = len(innovations)
    common = {
        "policy": "measured_state_reset_innovation_v1",
        "interval_count": len(trajectory.controls),
        "sample_count": sample_count,
        "nonfinite_interval_count": int(np.sum(~finite)),
        "initialization_discard_steps": initialization_discard_steps,
        "initialization_discard_s": (
            initialization_discard_steps * trajectory.nominal_dt_s
        ),
        "latent_actuator_state_carried": True,
        "rigid_body_state_reset_each_interval": True,
        "future_measurements_used": False,
        "state_kinematic_compatibility": (
            state_kinematic_compatibility_diagnostics(trajectory)
        ),
    }
    if sample_count < INNOVATION_MINIMUM_SAMPLES:
        return {
            **common,
            "status": "insufficient_samples",
            "minimum_sample_count": INNOVATION_MINIMUM_SAMPLES,
            "groups": {},
            "channels": {},
        }

    maximum_lag_steps = min(
        INNOVATION_MAXIMUM_LAG_STEPS,
        max(1, int(np.floor(INNOVATION_MAXIMUM_LAG_S / trajectory.nominal_dt_s))),
        max(1, sample_count // 4),
    )
    autocorrelation_bound = _simultaneous_correlation_bound(
        sample_count, maximum_lag_steps
    )
    input_correlation_bound = _simultaneous_correlation_bound(
        sample_count,
        (maximum_lag_steps + 1) * trajectory.control_size,
    )
    channels: dict[str, Any] = {}
    for index, (name, unit) in enumerate(INNOVATION_CHANNELS):
        values = innovations[:, index]
        autocorrelation, autocorrelation_lag = _strongest_autocorrelation(
            values, maximum_lag_steps
        )
        input_correlation, input_lag, input_index = (
            _strongest_input_correlation(
                values, controls, maximum_lag_steps
            )
        )
        channels[name] = {
            "unit": unit,
            "mean": float(np.mean(values)),
            "standard_deviation": float(np.std(values)),
            "rmse": float(np.sqrt(np.mean(np.square(values)))),
            "strongest_autocorrelation": autocorrelation,
            "autocorrelation_lag_steps": autocorrelation_lag,
            "autocorrelation_lag_s": (
                autocorrelation_lag * trajectory.nominal_dt_s
            ),
            "autocorrelation_bound": autocorrelation_bound,
            "temporally_colored": abs(autocorrelation) > autocorrelation_bound,
            "strongest_past_or_current_input_correlation": input_correlation,
            "input_correlation_control": trajectory.control_names[input_index],
            "input_correlation_control_role": trajectory.spec.control_roles[
                input_index
            ],
            "input_correlation_lag_steps": input_lag,
            "input_correlation_lag_s": input_lag * trajectory.nominal_dt_s,
            "input_correlation_bound": input_correlation_bound,
            "input_correlated": abs(input_correlation) > input_correlation_bound,
        }

    groups = {}
    channel_names = tuple(name for name, _ in INNOVATION_CHANNELS)
    for group, indices in INNOVATION_GROUPS.items():
        items = [channels[channel_names[index]] for index in indices]
        groups[group] = {
            "temporally_colored": any(
                bool(item["temporally_colored"]) for item in items
            ),
            "input_correlated": any(
                bool(item["input_correlated"]) for item in items
            ),
            "maximum_abs_autocorrelation": max(
                abs(float(item["strongest_autocorrelation"])) for item in items
            ),
            "maximum_abs_past_or_current_input_correlation": max(
                abs(
                    float(
                        item["strongest_past_or_current_input_correlation"]
                    )
                )
                for item in items
            ),
        }
    return {
        **common,
        "status": "ok",
        "maximum_lag_steps": maximum_lag_steps,
        "maximum_lag_s": maximum_lag_steps * trajectory.nominal_dt_s,
        "simultaneous_alpha": INNOVATION_SIMULTANEOUS_ALPHA,
        "groups": groups,
        "channels": channels,
        "summary": {
            "temporally_colored_group_count": sum(
                bool(item["temporally_colored"]) for item in groups.values()
            ),
            "input_correlated_group_count": sum(
                bool(item["input_correlated"]) for item in groups.values()
            ),
            "structured_innovation_detected": any(
                bool(item["temporally_colored"] or item["input_correlated"])
                for item in groups.values()
            ),
        },
        "interpretation": (
            "Correlation flags use a conservative simultaneous noise envelope. "
            "They distinguish white local error from structured held-out mismatch. "
            "Closed-loop feedback, estimator filtering, and state inconsistency can "
            "all create input correlation, so a flag is not a causal model term or "
            "permission to increase model complexity."
        ),
    }


def aggregate_innovation_diagnostics(
    diagnostics: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate innovation flags with equal weight per held-out flight."""

    valid = [item for item in diagnostics if item["status"] == "ok"]
    if not valid:
        return {
            "policy": "equal_flight_innovation_summary_v1",
            "status": "insufficient_samples",
            "flight_count": len(diagnostics),
            "valid_flight_count": 0,
        }
    groups = {}
    for group in INNOVATION_GROUPS:
        items = [item["groups"][group] for item in valid]
        groups[group] = {
            "temporally_colored_flight_fraction": float(
                np.mean([bool(item["temporally_colored"]) for item in items])
            ),
            "input_correlated_flight_fraction": float(
                np.mean([bool(item["input_correlated"]) for item in items])
            ),
            "mean_maximum_abs_autocorrelation": float(
                np.mean([item["maximum_abs_autocorrelation"] for item in items])
            ),
            "mean_maximum_abs_past_or_current_input_correlation": float(
                np.mean(
                    [
                        item[
                            "maximum_abs_past_or_current_input_correlation"
                        ]
                        for item in items
                    ]
                )
            ),
        }
    compatibility = [
        item["state_kinematic_compatibility"]
        for item in valid
        if item["state_kinematic_compatibility"]["status"] == "ok"
    ]
    return {
        "policy": "equal_flight_innovation_summary_v1",
        "status": "ok",
        "flight_count": len(diagnostics),
        "valid_flight_count": len(valid),
        "groups": groups,
        "state_kinematic_compatibility": {
            "valid_flight_count": len(compatibility),
            "inconsistent_flight_fraction": (
                None
                if not compatibility
                else float(
                    np.mean(
                        [
                            item["state_observations_temporally_inconsistent"]
                            for item in compatibility
                        ]
                    )
                )
            ),
            "position_velocity_colored_flight_fraction": (
                None
                if not compatibility
                else float(
                    np.mean(
                        [
                            item["position_velocity_compatibility"][
                                "temporally_colored"
                            ]
                            for item in compatibility
                        ]
                    )
                )
            ),
            "attitude_rate_colored_flight_fraction": (
                None
                if not compatibility
                else float(
                    np.mean(
                        [
                            item["attitude_rate_compatibility"][
                                "temporally_colored"
                            ]
                            for item in compatibility
                        ]
                    )
                )
            ),
            "mean_position_velocity_vector_rmse_m_s": (
                None
                if not compatibility
                else float(
                    np.mean(
                        [
                            item["position_velocity_compatibility"]["vector_rmse"]
                            for item in compatibility
                        ]
                    )
                )
            ),
            "mean_attitude_rate_vector_rmse_rad_s": (
                None
                if not compatibility
                else float(
                    np.mean(
                        [
                            item["attitude_rate_compatibility"]["vector_rmse"]
                            for item in compatibility
                        ]
                    )
                )
            ),
        },
        "flight_fraction_with_any_structured_innovation": float(
            np.mean(
                [
                    item["summary"]["structured_innovation_detected"]
                    for item in valid
                ]
            )
        ),
    }


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
