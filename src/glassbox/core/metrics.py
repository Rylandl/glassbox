"""Rollout predictions and the metrics that score them.

Two prediction entry points cover every protocol the library runs.
:func:`predict` rolls one complete flight from its measured start;
:func:`predict_windows` rolls fixed-horizon windows initialized throughout a
flight. Both return a :class:`RolloutPrediction`, and the metric functions here
score that object, so a scoring convention is chosen once by the caller rather
than baked into a per-protocol entry point.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from glassbox.core.data import Trajectory, trajectory_windows
from glassbox.core.dynamics import (
    ModelParams,
    control_state_after_history,
    rollout_with_latent,
    validate_control_schema,
)

# Rollout error statistics exclude the measured initial sample shared by
# prediction and target. Reports carry this identifier so recorded results
# produced under the earlier convention, which averaged over it, are
# distinguishable.
ROLLOUT_METRIC_POLICY = "rollout_error_excludes_measured_initial_sample_v2"

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

# Floors used when a scorer wants nothing but protection from log(0), rather
# than the physically meaningful equality floors above.
NEGLIGIBLE_METRIC_FLOORS = dict.fromkeys(ROLLOUT_METRICS, 1e-12)

SEQUENTIAL_LOG_MEAN = "sequential_log_mean"
VECTORIZED_LOG_MEAN = "vectorized_log_mean"


@dataclass(frozen=True)
class RolloutPrediction:
    """Aligned predicted and measured states from one rollout protocol.

    ``predicted`` and ``target`` have the same shape and both carry the
    measured initial state at time index zero, which every metric here
    excludes because prediction and target share it. A complete rollout is
    ``(sample, 13)``; windows are ``(window, horizon + 1, 13)``.
    """

    predicted: np.ndarray
    target: np.ndarray
    duration_s: float
    dt_s: float

    def endpoint_tangent_errors(self) -> np.ndarray:
        """Return the final-step error in 12 rigid-body local coordinates."""

        return rigid_body_tangent_errors(
            self.predicted[..., -1, :], self.target[..., -1, :]
        )


def predict(
    params: ModelParams,
    trajectory: Trajectory,
    *,
    control_history: np.ndarray | None = None,
) -> RolloutPrediction:
    """Return one complete logged-input prediction from the measured start.

    Logged exogenous context such as wind is applied per step, so a complete
    rollout sees the same time-varying context as the streaming evaluator.
    Fixed-horizon windows instead hold the context recorded at each window's
    start, matching how the fitter builds its training windows.
    """

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
    predicted = np.asarray(
        rollout_with_latent(
            params,
            jnp.asarray(trajectory.states[0]),
            jnp.asarray(trajectory.controls),
            trajectory.nominal_dt_s,
            initial_motor_state,
            trajectory.spec.control_roles,
            jnp.asarray(trajectory.exogenous[:-1]),
            trajectory.spec.exogenous_roles,
        )[0],
        dtype=np.float64,
    )
    return RolloutPrediction(
        predicted=predicted,
        target=trajectory.states,
        duration_s=float(trajectory.time_s[-1] - trajectory.time_s[0]),
        dt_s=trajectory.nominal_dt_s,
    )


def predict_windows(
    params: ModelParams,
    trajectory: Trajectory,
    *,
    horizon_steps: int,
    stride: int | None = None,
) -> RolloutPrediction:
    """Return fixed-horizon predictions initialized throughout one flight.

    Windows start every ``stride`` samples, defaulting to back-to-back windows
    one horizon apart. Use ``stride=1`` for rolling benchmark protocols that
    initialize a prediction at every admissible sample.
    """

    validate_control_schema(
        params,
        trajectory.control_names,
        trajectory.spec.control_roles,
    )

    windows = trajectory_windows(
        [trajectory],
        horizon=horizon_steps,
        stride=horizon_steps if stride is None else stride,
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
    return RolloutPrediction(
        predicted=np.asarray(predicted, dtype=np.float64),
        target=windows.target_states,
        duration_s=horizon_steps * windows.dt_s,
        dt_s=windows.dt_s,
    )


def rollout_metrics(prediction: RolloutPrediction) -> dict[str, Any]:
    """Score one prediction with the rollout RMSE convention."""

    return state_error_metrics(
        prediction.predicted,
        prediction.target,
        duration_s=prediction.duration_s,
    )


@dataclass(frozen=True)
class StateErrorTerms:
    """Per-sample rigid-body error components for aligned state arrays."""

    position_error: np.ndarray
    velocity_error: np.ndarray
    angular_velocity_error: np.ndarray
    attitude_error_deg: np.ndarray
    predicted_quaternion: np.ndarray
    target_quaternion: np.ndarray


def state_error_terms(
    predicted: np.ndarray,
    target: np.ndarray,
    *,
    normalize_quaternions: bool = True,
) -> StateErrorTerms:
    """Return the shared error components for two aligned ``(..., 13)`` arrays.

    Every sample pair supplied is scored; callers that share a measured initial
    state with the target drop it before calling.

    ``normalize_quaternions`` renormalizes both attitudes before the angle,
    which rollout scoring needs because integrated predictions drift off the
    unit sphere. The streaming evaluator scores validated unit quaternions and
    turns it off so that its published numbers stay bit-identical to the
    measurements they came from.
    """

    quaternion_predicted = predicted[..., 6:10]
    quaternion_target = target[..., 6:10]
    if normalize_quaternions:
        quaternion_predicted = quaternion_predicted / np.linalg.norm(
            quaternion_predicted, axis=-1, keepdims=True
        )
        quaternion_target = quaternion_target / np.linalg.norm(
            quaternion_target, axis=-1, keepdims=True
        )
    quaternion_dot = np.abs(np.sum(quaternion_predicted * quaternion_target, axis=-1))
    quaternion_dot = np.clip(quaternion_dot, -1.0, 1.0)
    return StateErrorTerms(
        position_error=predicted[..., 0:3] - target[..., 0:3],
        velocity_error=predicted[..., 3:6] - target[..., 3:6],
        angular_velocity_error=predicted[..., 10:13] - target[..., 10:13],
        attitude_error_deg=np.rad2deg(2.0 * np.arccos(quaternion_dot)),
        predicted_quaternion=quaternion_predicted,
        target_quaternion=quaternion_target,
    )


def _rmse_metrics_from_terms(terms: StateErrorTerms) -> dict[str, float]:
    return {
        "position_rmse_m": float(np.sqrt(np.mean(np.square(terms.position_error)))),
        "velocity_rmse_m_s": float(np.sqrt(np.mean(np.square(terms.velocity_error)))),
        "attitude_rmse_deg": float(
            np.sqrt(np.mean(np.square(terms.attitude_error_deg)))
        ),
        "angular_velocity_rmse_rad_s": float(
            np.sqrt(np.mean(np.square(terms.angular_velocity_error)))
        ),
    }


def state_rmse_metrics(
    predicted: np.ndarray,
    target: np.ndarray,
    *,
    normalize_quaternions: bool = True,
) -> dict[str, float]:
    """Return the four headline :data:`ROLLOUT_METRICS` over every sample pair."""

    return _rmse_metrics_from_terms(
        state_error_terms(
            predicted, target, normalize_quaternions=normalize_quaternions
        )
    )


def state_error_magnitudes(
    predicted: np.ndarray,
    target: np.ndarray,
    *,
    normalize_quaternions: bool = True,
) -> dict[str, Any]:
    """Return per-sample error magnitudes keyed like the divergence thresholds."""

    terms = state_error_terms(
        predicted, target, normalize_quaternions=normalize_quaternions
    )
    return {
        "position_error_m": np.linalg.norm(terms.position_error, axis=-1),
        "velocity_error_m_s": np.linalg.norm(terms.velocity_error, axis=-1),
        "attitude_error_deg": terms.attitude_error_deg,
        "angular_velocity_error_rad_s": np.linalg.norm(
            terms.angular_velocity_error, axis=-1
        ),
    }


def state_error_metrics(
    predicted: np.ndarray,
    target: np.ndarray,
    *,
    duration_s: float,
) -> dict[str, Any]:
    """Calculate metrics for one rollout or a batch of rollout windows.

    Both arrays carry the measured initial state at time index zero. That
    sample is shared by prediction and target, so it is excluded from every
    error statistic; ``sample_count`` counts predicted steps only.
    """

    if predicted.shape != target.shape:
        raise ValueError("predicted and target state arrays must match")
    if predicted.shape[-2] < 2:
        raise ValueError(
            "rollout metrics need the measured initial state followed by at "
            "least one predicted step"
        )
    predicted = predicted[..., 1:, :]
    target = target[..., 1:, :]

    terms = state_error_terms(predicted, target)
    position_error = terms.position_error
    velocity_error = terms.velocity_error
    angular_velocity_error = terms.angular_velocity_error
    predicted_quaternion = terms.predicted_quaternion
    target_quaternion = terms.target_quaternion

    # Express target^-1 * prediction as a shortest-path rotation vector. Its
    # components expose which body rotation axes dominate attitude drift.
    target_w = target_quaternion[..., 0]
    target_xyz = -target_quaternion[..., 1:4]
    predicted_w = predicted_quaternion[..., 0]
    predicted_xyz = predicted_quaternion[..., 1:4]
    relative_w = target_w * predicted_w - np.sum(target_xyz * predicted_xyz, axis=-1)
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
    headline = _rmse_metrics_from_terms(terms)
    return {
        "position_rmse_m": headline["position_rmse_m"],
        "position_rmse_xyz_m": np.sqrt(
            np.mean(np.square(position_error), axis=reduction_axes)
        ).tolist(),
        "velocity_rmse_m_s": headline["velocity_rmse_m_s"],
        "velocity_rmse_xyz_m_s": np.sqrt(
            np.mean(np.square(velocity_error), axis=reduction_axes)
        ).tolist(),
        "attitude_rmse_deg": headline["attitude_rmse_deg"],
        "angular_velocity_rmse_rad_s": headline["angular_velocity_rmse_rad_s"],
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
        "metric_policy": ROLLOUT_METRIC_POLICY,
    }


def rigid_body_tangent_errors(
    predicted: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    """Return target-minus-prediction errors in 12 rigid-body local coordinates."""

    predicted = np.asarray(predicted, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if predicted.shape != target.shape or predicted.shape[-1] != 13:
        raise ValueError("rigid-body error inputs must have matching (..., 13) shape")
    predicted_quaternion = predicted[..., 6:10] / np.linalg.norm(
        predicted[..., 6:10], axis=-1, keepdims=True
    )
    target_quaternion = target[..., 6:10] / np.linalg.norm(
        target[..., 6:10], axis=-1, keepdims=True
    )
    predicted_w = predicted_quaternion[..., 0]
    predicted_xyz = predicted_quaternion[..., 1:4]
    target_w = target_quaternion[..., 0]
    target_xyz = target_quaternion[..., 1:4]
    relative_w = predicted_w * target_w + np.sum(predicted_xyz * target_xyz, axis=-1)
    relative_xyz = (
        predicted_w[..., np.newaxis] * target_xyz
        - target_w[..., np.newaxis] * predicted_xyz
        - np.cross(predicted_xyz, target_xyz)
    )
    relative_sign = np.where(relative_w < 0.0, -1.0, 1.0)
    relative_w *= relative_sign
    relative_xyz *= relative_sign[..., np.newaxis]
    vector_norm = np.linalg.norm(relative_xyz, axis=-1)
    angle = 2.0 * np.arctan2(vector_norm, np.clip(relative_w, 0.0, 1.0))
    scale = np.divide(
        angle,
        vector_norm,
        out=np.full_like(angle, 2.0),
        where=vector_norm > 1e-12,
    )
    return np.concatenate(
        (
            target[..., 0:3] - predicted[..., 0:3],
            target[..., 3:6] - predicted[..., 3:6],
            relative_xyz * scale[..., np.newaxis],
            target[..., 10:13] - predicted[..., 10:13],
        ),
        axis=-1,
    )


def kinematic_persistence_windowed_metrics(
    trajectory: Trajectory,
    *,
    horizon_steps: int,
    stride: int | None = None,
) -> dict[str, Any]:
    """Score a constant-world-velocity and constant-body-rate baseline."""

    windows = trajectory_windows(
        [trajectory],
        horizon=horizon_steps,
        stride=horizon_steps if stride is None else stride,
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
        state[:, 6:10] /= np.linalg.norm(state[:, 6:10], axis=1, keepdims=True)
        predicted[:, index + 1] = state
    return state_error_metrics(
        predicted,
        windows.target_states,
        duration_s=horizon_steps * windows.dt_s,
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
        not np.isfinite(value) or value <= 0.0 for value in selected_thresholds.values()
    ):
        raise ValueError("divergence thresholds must be finite and positive")

    prediction = predict(params, trajectory, control_history=control_history)
    predicted = prediction.predicted
    target = prediction.target
    predicted_quaternion_norm = np.linalg.norm(predicted[:, 6:10], axis=1)
    finite = np.all(np.isfinite(predicted), axis=1) & (predicted_quaternion_norm > 0.0)
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
        target_quaternion /= np.linalg.norm(target_quaternion, axis=1, keepdims=True)
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
    divergence_index = None if len(crossing_indices) == 0 else int(crossing_indices[0])
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
        stable_through_s = float(trajectory.time_s[stable_index] - trajectory.time_s[0])
    duration_s = float(trajectory.time_s[-1] - trajectory.time_s[0])
    return {
        "thresholds": selected_thresholds,
        "full_rollout_finite": bool(np.all(finite)),
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
    }


def persistence_score(
    candidate: Mapping[str, Mapping[str, Any]],
    persistence: Mapping[str, Mapping[str, Any]],
    *,
    horizons: Sequence[float] | None,
    floors: Mapping[str, float],
    metrics: Sequence[str] = ROLLOUT_METRICS,
    aggregation: str = SEQUENTIAL_LOG_MEAN,
) -> float:
    """Score one candidate against a baseline over horizons and state metrics.

    Both mappings are keyed by a horizon label and then by metric name. The
    result is the geometric mean of the floored candidate-over-baseline ratios,
    so values below one favor the candidate and every metric and horizon
    contributes equally.

    ``horizons`` are seconds, formatted as ``f"{seconds:g}s"`` to index both
    mappings; pass ``None`` to score every label the candidate carries, in its
    own order. ``floors`` supplies the per-metric value below which a term is
    treated as equal, which stops numerical noise near zero from dominating.

    ``aggregation`` selects the floating-point reduction. The two options
    compute the same geometric mean but round differently, and recorded reports
    pin the one that produced them: ``SEQUENTIAL_LOG_MEAN`` sums
    :func:`math.log` terms in order, while ``VECTORIZED_LOG_MEAN`` takes
    NumPy's pairwise-summed mean of :func:`numpy.log`, which is what the
    recorded Skywalker X8 scores were produced with.
    """

    labels = (
        list(candidate)
        if horizons is None
        else [f"{seconds:g}s" for seconds in horizons]
    )
    ratios = [
        max(float(candidate[label][metric]), floors[metric])
        / max(float(persistence[label][metric]), floors[metric])
        for label in labels
        for metric in metrics
    ]
    if not ratios:
        raise ValueError("cannot average an empty score collection")
    if aggregation == SEQUENTIAL_LOG_MEAN:
        return float(math.exp(sum(math.log(value) for value in ratios) / len(ratios)))
    if aggregation == VECTORIZED_LOG_MEAN:
        return float(np.exp(np.mean(np.log(ratios))))
    raise ValueError(f"unknown persistence score aggregation: {aggregation!r}")


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
                        float(item["final_position_error_m"]) ** 2 * weight
                        for item, weight in zip(metrics, endpoint_weights)
                    )
                    / sum(endpoint_weights)
                )
            ),
            "duration_s": float(sum(float(item["duration_s"]) for item in metrics)),
            "sample_count": sample_count,
            "rollout_count": rollout_count,
            "weighting": weighting,
            "metric_policy": ROLLOUT_METRIC_POLICY,
        }
    )
    return result
