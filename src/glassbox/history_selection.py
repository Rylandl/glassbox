"""Opinionated selection of causal residual-innovation memory."""

from __future__ import annotations

import math
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from glassbox.data import (
    Trajectory,
    TrajectoryWindows,
    duration_to_steps,
    trajectory_windows,
)
from glassbox.dynamics import (
    INSTANTANEOUS_RESIDUAL_RESPONSE_TIME_CONSTANT_S,
    HistoryResidualDynamicsParams,
    ModelParams,
    ResidualDynamicsParams,
    history_residual_from_residual,
    latent_state_after_history,
    rollout_with_latent,
    with_instantaneous_residual_response,
)
from glassbox.evaluation import METRIC_FLOORS, ROLLOUT_METRICS

HISTORY_RESPONSE_CANDIDATES_S = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5)
HISTORY_SELECTION_HORIZON_LIMIT_S = 1.0
HISTORY_INITIALIZATION_DURATION_S = 0.5
MAXIMUM_HISTORY_SELECTION_WINDOWS = 1_024
HISTORY_SELECTION_MINIMUM_IMPROVEMENT = 0.02
HISTORY_SELECTION_MAXIMUM_REGRESSION = 0.05

_TRANSLATION_METRICS = ("position_rmse_m", "velocity_rmse_m_s")
_ROTATION_METRICS = ("attitude_rmse_deg", "angular_velocity_rmse_rad_s")


def _geometric_mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot score an empty metric collection")
    return float(math.exp(sum(math.log(value) for value in values) / len(values)))


def _window_predictions(params: ModelParams, windows: TrajectoryWindows) -> np.ndarray:
    initial_latent = jax.vmap(
        lambda state_history, control_history, exogenous_history, valid: (
            latent_state_after_history(
                params,
                state_history,
                control_history,
                windows.dt_s,
                windows.control_roles,
                exogenous_history,
                windows.exogenous_roles,
                valid,
            )
        )
    )(
        jnp.asarray(windows.state_histories),
        jnp.asarray(windows.control_histories),
        jnp.asarray(windows.exogenous_histories),
        jnp.asarray(windows.history_valid),
    )
    predicted, _ = jax.vmap(
        lambda state, controls, latent, exogenous: rollout_with_latent(
            params,
            state,
            controls,
            windows.dt_s,
            latent,
            windows.control_roles,
            exogenous,
            windows.exogenous_roles,
        )
    )(
        jnp.asarray(windows.initial_states),
        jnp.asarray(windows.controls),
        initial_latent,
        jnp.asarray(windows.initial_exogenous),
    )
    return np.asarray(predicted, dtype=np.float64)


def _per_trajectory_metrics(
    predicted: np.ndarray,
    windows: TrajectoryWindows,
) -> list[dict[str, float]]:
    target = windows.target_states
    predicted = predicted[:, 1:]
    target = target[:, 1:]
    position_mse = np.mean(
        np.square(predicted[..., 0:3] - target[..., 0:3]), axis=(1, 2)
    )
    velocity_mse = np.mean(
        np.square(predicted[..., 3:6] - target[..., 3:6]), axis=(1, 2)
    )
    angular_velocity_mse = np.mean(
        np.square(predicted[..., 10:13] - target[..., 10:13]), axis=(1, 2)
    )
    predicted_quaternion = predicted[..., 6:10] / np.linalg.norm(
        predicted[..., 6:10], axis=-1, keepdims=True
    )
    target_quaternion = target[..., 6:10] / np.linalg.norm(
        target[..., 6:10], axis=-1, keepdims=True
    )
    quaternion_dot = np.clip(
        np.abs(np.sum(predicted_quaternion * target_quaternion, axis=-1)),
        0.0,
        1.0,
    )
    attitude_mse = np.mean(
        np.square(np.rad2deg(2.0 * np.arccos(quaternion_dot))), axis=1
    )
    per_window = {
        "position_rmse_m": position_mse,
        "velocity_rmse_m_s": velocity_mse,
        "attitude_rmse_deg": attitude_mse,
        "angular_velocity_rmse_rad_s": angular_velocity_mse,
    }
    result = []
    for trajectory_index in range(len(windows.candidate_window_counts)):
        selected = windows.trajectory_indices == trajectory_index
        if not np.any(selected):
            continue
        result.append(
            {
                name: float(np.sqrt(np.mean(values[selected])))
                for name, values in per_window.items()
            }
        )
    return result


def _evaluate(
    params: ModelParams,
    window_sets: Sequence[TrajectoryWindows],
) -> list[list[dict[str, float]]]:
    return [
        _per_trajectory_metrics(_window_predictions(params, windows), windows)
        for windows in window_sets
    ]


def _ratio_score(
    candidate: Sequence[Sequence[dict[str, float]]],
    reference: Sequence[Sequence[dict[str, float]]],
    metrics: Sequence[str],
) -> dict[str, Any]:
    ratios = []
    for candidate_horizon, reference_horizon in zip(candidate, reference, strict=True):
        if len(candidate_horizon) != len(reference_horizon):
            raise ValueError("candidate and reference trajectory counts differ")
        for candidate_item, reference_item in zip(
            candidate_horizon, reference_horizon, strict=True
        ):
            for metric in metrics:
                floor = METRIC_FLOORS[metric]
                ratios.append(
                    max(candidate_item[metric], floor)
                    / max(reference_item[metric], floor)
                )
    return {
        "geometric_ratio": _geometric_mean(ratios),
        "maximum_ratio": max(ratios),
        "ratios": ratios,
    }


def _eligible(score: dict[str, Any]) -> bool:
    return bool(
        score["geometric_ratio"] <= 1.0 - HISTORY_SELECTION_MINIMUM_IMPROVEMENT
        and score["maximum_ratio"] <= 1.0 + HISTORY_SELECTION_MAXIMUM_REGRESSION
    )


def select_history_residual(
    params: ResidualDynamicsParams,
    validation_trajectories: Sequence[Trajectory],
    *,
    horizons_s: Sequence[float] = (0.1, 0.5, 1.0),
) -> tuple[ModelParams, dict[str, Any]]:
    """Select bounded force/moment memory or retain the exact no-memory model.

    The policy is intentionally fixed rather than exposed as user tuning. It
    uses only validation trajectories, scores complete state groups with equal
    trajectory influence, and requires both aggregate improvement and a
    per-trajectory metric guardrail.
    """

    trajectories = tuple(validation_trajectories)
    if not trajectories:
        raise ValueError("history selection requires validation trajectories")
    dt_s = trajectories[0].nominal_dt_s
    selected_horizons = tuple(
        dict.fromkeys(
            float(value)
            for value in horizons_s
            if 0.0 < value <= HISTORY_SELECTION_HORIZON_LIMIT_S
            and duration_to_steps(value, dt_s)
            <= min(len(trajectory.controls) for trajectory in trajectories)
        )
    )
    if not selected_horizons:
        raise ValueError("no history-selection horizon fits the validation data")
    window_sets = tuple(
        trajectory_windows(
            trajectories,
            horizon=duration_to_steps(seconds, dt_s),
            stride=duration_to_steps(seconds, dt_s),
            motor_history_s=HISTORY_INITIALIZATION_DURATION_S,
            balance_trajectories=True,
            maximum_windows=MAXIMUM_HISTORY_SELECTION_WINDOWS,
        )
        for seconds in selected_horizons
    )
    reference = with_instantaneous_residual_response(
        history_residual_from_residual(params)
    )
    reference_metrics = _evaluate(reference, window_sets)
    instantaneous = INSTANTANEOUS_RESIDUAL_RESPONSE_TIME_CONSTANT_S

    def select_axis(
        *,
        force: bool,
        metrics: Sequence[str],
    ) -> tuple[float, dict[str, Any]]:
        scores = {}
        for value in HISTORY_RESPONSE_CANDIDATES_S:
            time_constants = (value, instantaneous) if force else (instantaneous, value)
            candidate = history_residual_from_residual(
                params, response_time_constant_s=time_constants
            )
            scores[f"{value:g}s"] = _ratio_score(
                _evaluate(candidate, window_sets), reference_metrics, metrics
            )
        eligible = [
            value
            for value in HISTORY_RESPONSE_CANDIDATES_S
            if _eligible(scores[f"{value:g}s"])
        ]
        selected = (
            min(
                eligible,
                key=lambda value: scores[f"{value:g}s"]["geometric_ratio"],
            )
            if eligible
            else instantaneous
        )
        return selected, {
            "selected_time_constant_s": (
                None if selected == instantaneous else selected
            ),
            "candidate_scores": scores,
        }

    force_time, force_report = select_axis(force=True, metrics=_TRANSLATION_METRICS)
    moment_time, moment_report = select_axis(force=False, metrics=_ROTATION_METRICS)
    combined = history_residual_from_residual(
        params,
        response_time_constant_s=(force_time, moment_time),
    )
    combined_score = _ratio_score(
        _evaluate(combined, window_sets),
        reference_metrics,
        ROLLOUT_METRICS,
    )
    selected: ModelParams = combined if _eligible(combined_score) else params
    selected_history = isinstance(selected, HistoryResidualDynamicsParams)
    return selected, {
        "policy": "bounded_causal_residual_innovation_selection_v1",
        "status": (
            "selected_history_observer"
            if selected_history
            else "retained_instantaneous_reference"
        ),
        "selected": selected_history,
        "history_initialization_duration_s": HISTORY_INITIALIZATION_DURATION_S,
        "horizons_s": list(selected_horizons),
        "candidate_time_constants_s": list(HISTORY_RESPONSE_CANDIDATES_S),
        "minimum_aggregate_improvement": HISTORY_SELECTION_MINIMUM_IMPROVEMENT,
        "maximum_per_metric_regression": HISTORY_SELECTION_MAXIMUM_REGRESSION,
        "force": force_report,
        "moment": moment_report,
        "combined_score": combined_score,
        "data_role": "model_selection",
        "unbiased_performance_estimate": False,
        "production_lockbox_required": True,
    }
