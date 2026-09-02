"""Research-only rollout scoring through a body-rate observation model."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from glassbox.data import Trajectory, duration_to_steps
from glassbox.dynamics import ModelParams
from glassbox.evaluation import (
    _state_error_metrics,
    aggregate_rollout_metrics,
    windowed_rollout_predictions,
)
from glassbox.observation_compatibility import FirstOrderObservationFilter

BODY_RATE_ROLLOUT_MATERIAL_RATIO = 0.9
BODY_RATE_ROLLOUT_MAXIMUM_REGRESSION_RATIO = 1.05


def _error_ratio(candidate: float, reference: float) -> float:
    if reference > 0.0:
        return candidate / reference
    return 1.0 if candidate == 0.0 else float("inf")


def first_order_body_rate_observations(
    physical_rate_rad_s: np.ndarray,
    *,
    initial_reported_rate_rad_s: np.ndarray,
    model: FirstOrderObservationFilter,
    dt_s: float,
) -> np.ndarray:
    """Map physical body rate to reported rate without changing dynamics.

    ``physical_rate_rad_s`` may have arbitrary leading dimensions followed by
    ``(time, xyz)``. The filter state is initialized by inverting the affine
    observation at the measured rollout start.
    """

    physical = np.asarray(physical_rate_rad_s, dtype=np.float64)
    initial = np.asarray(initial_reported_rate_rad_s, dtype=np.float64)
    if physical.ndim < 2 or physical.shape[-1] != 3:
        raise ValueError("physical body rate must end in (time, xyz)")
    if initial.shape != physical.shape[:-2] + (3,):
        raise ValueError("initial reported rate shape does not match rollouts")
    if not np.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("dt_s must be finite and positive")

    observed = np.empty_like(physical)
    observed[..., 0, :] = initial
    latent = (initial - model.angular_rate_bias_rad_s) / model.angular_rate_scale
    time_constant_s = model.angular_rate_time_constant_s
    decay = np.zeros(3, dtype=np.float64)
    temporal = time_constant_s > 0.0
    decay[temporal] = np.exp(-dt_s / time_constant_s[temporal])
    for index in range(1, physical.shape[-2]):
        latent = decay * latent + (1.0 - decay) * physical[..., index, :]
        observed[..., index, :] = (
            model.angular_rate_scale * latent + model.angular_rate_bias_rad_s
        )
    return observed


def body_rate_observation_metrics_from_predictions(
    predicted: np.ndarray,
    target: np.ndarray,
    *,
    model: FirstOrderObservationFilter,
    dt_s: float,
) -> dict[str, Any]:
    """Score one prediction batch after replacing only reported body rate."""

    if predicted.shape != target.shape:
        raise ValueError("predicted and target state arrays must match")
    if predicted.ndim not in (2, 3) or predicted.shape[-1] != 13:
        raise ValueError(
            "state predictions must have shape (time, 13) or (batch, time, 13)"
        )
    observed_prediction = np.asarray(predicted, dtype=np.float64).copy()
    observed_prediction[..., 10:13] = first_order_body_rate_observations(
        predicted[..., 10:13],
        initial_reported_rate_rad_s=target[..., 0, 10:13]
        if target.ndim == 3
        else target[0, 10:13],
        model=model,
        dt_s=dt_s,
    )
    return _state_error_metrics(
        observed_prediction,
        target,
        duration_s=(predicted.shape[-2] - 1) * dt_s,
    )


def body_rate_observation_windowed_metrics(
    params: ModelParams,
    trajectory: Trajectory,
    *,
    model: FirstOrderObservationFilter,
    horizon_steps: int,
    stride_steps: int = 1,
) -> dict[str, Any]:
    """Evaluate rolling dynamics predictions through one rate observation."""

    predicted, target, dt_s = windowed_rollout_predictions(
        params,
        trajectory,
        horizon_steps=horizon_steps,
        stride_steps=stride_steps,
    )
    return body_rate_observation_metrics_from_predictions(
        predicted,
        target,
        model=model,
        dt_s=dt_s,
    )


def evaluate_body_rate_observation_rollouts(
    params: ModelParams,
    candidate: FirstOrderObservationFilter,
    instantaneous_reference: FirstOrderObservationFilter,
    trajectories: Sequence[Trajectory],
    *,
    horizons_s: tuple[float, ...] = (0.1, 0.5, 1.0),
) -> dict[str, Any]:
    """Run a fixed output-error A/B over complete research-validation flights."""

    if not trajectories:
        raise ValueError("at least one trajectory is required")
    if not horizons_s or any(horizon <= 0.0 for horizon in horizons_s):
        raise ValueError("horizons_s must contain positive values")
    per_trajectory = []
    for trajectory in trajectories:
        horizons: dict[str, Any] = {}
        for horizon_s in horizons_s:
            steps = duration_to_steps(horizon_s, trajectory.nominal_dt_s)
            label = f"{steps * trajectory.nominal_dt_s:g}s"
            predicted, target, dt_s = windowed_rollout_predictions(
                params,
                trajectory,
                horizon_steps=steps,
                stride_steps=steps,
            )
            candidate_metrics = body_rate_observation_metrics_from_predictions(
                predicted,
                target,
                model=candidate,
                dt_s=dt_s,
            )
            reference_metrics = body_rate_observation_metrics_from_predictions(
                predicted,
                target,
                model=instantaneous_reference,
                dt_s=dt_s,
            )
            horizons[label] = {
                "candidate": candidate_metrics,
                "instantaneous_reference": reference_metrics,
                "angular_velocity_ratio": _error_ratio(
                    candidate_metrics["angular_velocity_rmse_rad_s"],
                    reference_metrics["angular_velocity_rmse_rad_s"],
                ),
            }
        per_trajectory.append(
            {
                "source_group": trajectory.labels.get("source_group"),
                "profile": trajectory.labels.get("profile"),
                "replicate": trajectory.labels.get("replicate"),
                "horizons": horizons,
            }
        )

    labels = tuple(per_trajectory[0]["horizons"])
    aggregate = {}
    all_ratios = []
    for label in labels:
        candidate_metrics = aggregate_rollout_metrics(
            [item["horizons"][label]["candidate"] for item in per_trajectory],
            weighting="equal",
        )
        reference_metrics = aggregate_rollout_metrics(
            [
                item["horizons"][label]["instantaneous_reference"]
                for item in per_trajectory
            ],
            weighting="equal",
        )
        ratio = _error_ratio(
            candidate_metrics["angular_velocity_rmse_rad_s"],
            reference_metrics["angular_velocity_rmse_rad_s"],
        )
        all_ratios.extend(
            item["horizons"][label]["angular_velocity_ratio"] for item in per_trajectory
        )
        aggregate[label] = {
            "candidate": candidate_metrics,
            "instantaneous_reference": reference_metrics,
            "angular_velocity_ratio": ratio,
        }
    aggregate_ratios = [aggregate[label]["angular_velocity_ratio"] for label in labels]
    geometric_ratio = (
        0.0
        if any(ratio == 0.0 for ratio in aggregate_ratios)
        else float(np.exp(np.mean(np.log(aggregate_ratios))))
    )
    material = bool(geometric_ratio <= BODY_RATE_ROLLOUT_MATERIAL_RATIO)
    no_regression = bool(
        max(aggregate_ratios + all_ratios) <= BODY_RATE_ROLLOUT_MAXIMUM_REGRESSION_RATIO
    )
    return {
        "policy": "conditional_body_rate_observation_rollout_v1",
        "trajectory_count": len(trajectories),
        "horizons_s": list(horizons_s),
        "window_stride": "one nonoverlapping horizon",
        "candidate": candidate.to_dict(),
        "instantaneous_reference": instantaneous_reference.to_dict(),
        "aggregate": aggregate,
        "per_trajectory": per_trajectory,
        "gate": {
            "geometric_angular_velocity_ratio": geometric_ratio,
            "materiality_scope": (
                "geometric mean of equal-trajectory aggregate ratios across "
                "all requested horizons"
            ),
            "material_improvement_ratio": BODY_RATE_ROLLOUT_MATERIAL_RATIO,
            "maximum_regression_ratio": (BODY_RATE_ROLLOUT_MAXIMUM_REGRESSION_RATIO),
            "improves_materially": material,
            "no_horizon_or_complete_trajectory_regresses": no_regression,
            "research_rollout_passes": bool(material and no_regression),
            "production_promotion_requires_fresh_lockbox": True,
        },
        "invariants": {
            "dynamics_parameters_changed": False,
            "physical_rollout_changed": False,
            "position_velocity_attitude_metrics_changed": False,
            "only_reported_body_rate_changed": True,
        },
    }
