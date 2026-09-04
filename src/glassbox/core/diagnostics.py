"""Opt-in innovation and kinematic-compatibility diagnostics.

These score a fit rather than a forecast: one-step innovations expose model
error that a rollout metric hides inside accumulated drift, and the kinematic
compatibility check uses no model at all, so it separates telemetry
inconsistency from prediction error. They are the answer to "why is the
rollout wrong", not to "how wrong is it", and run only when a caller asks
(:attr:`glassbox.fitting.FitSpec.diagnostics`, ``glassbox fit --diagnostics``).
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from glassbox.core.data import Trajectory
from glassbox.core.dynamics import (
    ModelParams,
    control_state_after_history,
    step_with_latent,
    validate_control_schema,
)
from glassbox.core.geometry import quaternion_to_rotation_matrices

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


def attitude_innovation(
    predicted_wxyz: np.ndarray,
    observed_wxyz: np.ndarray,
) -> np.ndarray:
    """Return the shortest predicted-to-observed rotation vector."""

    predicted = predicted_wxyz / np.linalg.norm(predicted_wxyz, axis=1, keepdims=True)
    observed = observed_wxyz / np.linalg.norm(observed_wxyz, axis=1, keepdims=True)
    predicted_w = predicted[:, 0]
    predicted_xyz = predicted[:, 1:4]
    observed_w = observed[:, 0]
    observed_xyz = observed[:, 1:4]
    relative_w = predicted_w * observed_w + np.sum(predicted_xyz * observed_xyz, axis=1)
    relative_xyz = (
        predicted_w[:, None] * observed_xyz
        - observed_w[:, None] * predicted_xyz
        - np.cross(predicted_xyz, observed_xyz)
    )
    sign = np.where(relative_w < 0.0, -1.0, 1.0)
    relative_w *= sign
    relative_xyz *= sign[:, None]
    vector_norm = np.linalg.norm(relative_xyz, axis=1)
    angle = 2.0 * np.arctan2(vector_norm, np.clip(relative_w, 0.0, 1.0))
    scale = np.divide(
        angle,
        vector_norm,
        out=np.full_like(angle, 2.0),
        where=vector_norm > 1e-12,
    )
    return relative_xyz * scale[:, None]


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
    position_rate_residual = np.diff(states[:, 0:3], axis=0) / dt_s - 0.5 * (
        states[:-1, 3:6] + states[1:, 3:6]
    )
    attitude_increment = attitude_innovation(states[:-1, 6:10], states[1:, 6:10])
    rotations = quaternion_to_rotation_matrices(states[:, 6:10])
    next_rate_in_initial_body = np.einsum(
        "nji,njk,nk->ni",
        rotations[:-1],
        rotations[1:],
        states[1:, 10:13],
    )
    attitude_rate_residual = attitude_increment / dt_s - 0.5 * (
        states[:-1, 10:13] + next_rate_in_initial_body
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
    bound = _simultaneous_correlation_bound(sample_count, 3 * maximum_lag_steps)

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
            "vector_rmse": float(np.sqrt(np.mean(np.sum(np.square(values), axis=1)))),
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


def pearson_correlation(left: np.ndarray, right: np.ndarray) -> float:
    """Return the centered correlation of two equal-length one-dimensional signals.

    Series shorter than three samples, and pairs where either side is
    effectively constant, score zero rather than an unstable ratio.
    """

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
                * np.log(2.0 * comparison_count / INNOVATION_SIMULTANEOUS_ALPHA)
                / sample_count
            )
        ),
    )


def _strongest_autocorrelation(
    values: np.ndarray,
    maximum_lag_steps: int,
) -> tuple[float, int]:
    candidates = [
        (pearson_correlation(values[:-lag], values[lag:]), lag)
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
            correlation = pearson_correlation(
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
            attitude_innovation(predicted[:, 6:10], observed[:, 6:10]),
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
        input_correlation, input_lag, input_index = _strongest_input_correlation(
            values, controls, maximum_lag_steps
        )
        channels[name] = {
            "unit": unit,
            "mean": float(np.mean(values)),
            "standard_deviation": float(np.std(values)),
            "rmse": float(np.sqrt(np.mean(np.square(values)))),
            "strongest_autocorrelation": autocorrelation,
            "autocorrelation_lag_steps": autocorrelation_lag,
            "autocorrelation_lag_s": (autocorrelation_lag * trajectory.nominal_dt_s),
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
            "input_correlated": any(bool(item["input_correlated"]) for item in items),
            "maximum_abs_autocorrelation": max(
                abs(float(item["strongest_autocorrelation"])) for item in items
            ),
            "maximum_abs_past_or_current_input_correlation": max(
                abs(float(item["strongest_past_or_current_input_correlation"]))
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
                        item["maximum_abs_past_or_current_input_correlation"]
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
                            item["attitude_rate_compatibility"]["temporally_colored"]
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
                [item["summary"]["structured_innovation_detected"] for item in valid]
            )
        ),
    }
