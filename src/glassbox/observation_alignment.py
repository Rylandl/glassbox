"""Research-only state-channel timing alignment diagnostics.

This module tests clock/transport alignment separately from physical actuator
lag and estimator filtering. It is intentionally absent from the package root
and the normal fitting path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import numpy.typing as npt

from glassbox.data import Trajectory
from glassbox.evaluation import (
    KINEMATIC_ATTITUDE_RATE_FLOOR_RAD_S,
    KINEMATIC_POSITION_RATE_FLOOR_M_S,
)
from glassbox.observation_compatibility import (
    MAXIMUM_ANGULAR_RATE_BIAS_RAD_S,
    MAXIMUM_SCALE,
    MAXIMUM_VELOCITY_BIAS_M_S,
    MINIMUM_SCALE,
    _bounded_affine_from_statistics,
    _observation_rate_pairs,
    _validate_observation_fit_data,
    observation_channel_transfer_gate,
)


ALIGNMENT_POLICY = "bounded_state_observation_alignment_v1"
MAXIMUM_ABSOLUTE_ALIGNMENT_S = 0.1
ALIGNMENT_CANDIDATES_S = np.linspace(
    -MAXIMUM_ABSOLUTE_ALIGNMENT_S,
    MAXIMUM_ABSOLUTE_ALIGNMENT_S,
    num=41,
)


@dataclass(frozen=True)
class StateObservationAlignment:
    """Shared-axis timing alignment plus bounded per-axis affine terms."""

    velocity_delay_s: float
    velocity_scale: npt.NDArray[np.float64]
    velocity_bias_m_s: npt.NDArray[np.float64]
    angular_rate_delay_s: float
    angular_rate_scale: npt.NDArray[np.float64]
    angular_rate_bias_rad_s: npt.NDArray[np.float64]

    def __post_init__(self) -> None:
        for name in ("velocity_delay_s", "angular_rate_delay_s"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or (
                abs(value) > MAXIMUM_ABSOLUTE_ALIGNMENT_S
            ):
                raise ValueError(f"{name} exceeds the maintained bounds")
            object.__setattr__(self, name, value)
        for name in (
            "velocity_scale",
            "velocity_bias_m_s",
            "angular_rate_scale",
            "angular_rate_bias_rad_s",
        ):
            values = np.asarray(getattr(self, name), dtype=np.float64).copy()
            if values.shape != (3,) or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must contain three finite values")
            values.setflags(write=False)
            object.__setattr__(self, name, values)
        for name in ("velocity_scale", "angular_rate_scale"):
            values = getattr(self, name)
            if np.any(values < MINIMUM_SCALE) or np.any(values > MAXIMUM_SCALE):
                raise ValueError(f"{name} exceeds the maintained bounds")
        if np.any(np.abs(self.velocity_bias_m_s) > MAXIMUM_VELOCITY_BIAS_M_S):
            raise ValueError("velocity_bias_m_s exceeds the maintained bounds")
        if np.any(
            np.abs(self.angular_rate_bias_rad_s)
            > MAXIMUM_ANGULAR_RATE_BIAS_RAD_S
        ):
            raise ValueError(
                "angular_rate_bias_rad_s exceeds the maintained bounds"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": ALIGNMENT_POLICY,
            "delay_convention": (
                "positive means the reported channel lags pose-implied motion"
            ),
            "velocity_delay_s": self.velocity_delay_s,
            "velocity_scale": self.velocity_scale.tolist(),
            "velocity_bias_m_s": self.velocity_bias_m_s.tolist(),
            "angular_rate_delay_s": self.angular_rate_delay_s,
            "angular_rate_scale": self.angular_rate_scale.tolist(),
            "angular_rate_bias_rad_s": (
                self.angular_rate_bias_rad_s.tolist()
            ),
            "bounds": {
                "alignment_s": [
                    -MAXIMUM_ABSOLUTE_ALIGNMENT_S,
                    MAXIMUM_ABSOLUTE_ALIGNMENT_S,
                ],
                "scale": [MINIMUM_SCALE, MAXIMUM_SCALE],
                "velocity_bias_m_s": [
                    -MAXIMUM_VELOCITY_BIAS_M_S,
                    MAXIMUM_VELOCITY_BIAS_M_S,
                ],
                "angular_rate_bias_rad_s": [
                    -MAXIMUM_ANGULAR_RATE_BIAS_RAD_S,
                    MAXIMUM_ANGULAR_RATE_BIAS_RAD_S,
                ],
            },
        }


@dataclass(frozen=True)
class StateObservationAlignmentFit:
    candidate: StateObservationAlignment
    instantaneous_reference: StateObservationAlignment
    report: dict[str, Any]


def _source_group_key(trajectory: Trajectory, index: int) -> str:
    source_group = trajectory.labels.get("source_group")
    if source_group is None:
        return f"trajectory:{index}"
    return f"source_group:{source_group}"


def _interval_center_times(trajectory: Trajectory) -> np.ndarray:
    return 0.5 * (trajectory.time_s[:-1] + trajectory.time_s[1:])


def _alignment_mask(center_times_s: np.ndarray) -> np.ndarray:
    lower = center_times_s[0] + MAXIMUM_ABSOLUTE_ALIGNMENT_S
    upper = center_times_s[-1] - MAXIMUM_ABSOLUTE_ALIGNMENT_S
    mask = (center_times_s >= lower) & (center_times_s <= upper)
    if np.count_nonzero(mask) < 20:
        raise ValueError(
            "state observation alignment requires at least 20 interior intervals"
        )
    return mask


def _shifted_response(
    values: np.ndarray,
    center_times_s: np.ndarray,
    delay_s: float,
) -> np.ndarray:
    query_times_s = center_times_s - delay_s
    return np.column_stack(
        [
            np.interp(query_times_s, center_times_s, values[:, axis])
            for axis in range(3)
        ]
    )


def _alignment_statistics(
    values: np.ndarray,
    target: np.ndarray,
    center_times_s: np.ndarray,
    candidates_s: np.ndarray,
) -> dict[str, np.ndarray | float]:
    mask = _alignment_mask(center_times_s)
    observed = target[mask]
    count = float(len(observed))
    sum_x = np.zeros((len(candidates_s), 3), dtype=np.float64)
    sum_x2 = np.zeros_like(sum_x)
    sum_xy = np.zeros_like(sum_x)
    for candidate_index, delay_s in enumerate(candidates_s):
        shifted = _shifted_response(values, center_times_s, float(delay_s))[mask]
        sum_x[candidate_index] = np.sum(shifted, axis=0)
        sum_x2[candidate_index] = np.sum(shifted * shifted, axis=0)
        sum_xy[candidate_index] = np.sum(shifted * observed, axis=0)
    return {
        "sum_x": sum_x,
        "sum_x2": sum_x2,
        "sum_xy": sum_xy,
        "sum_y": np.sum(observed, axis=0),
        "sum_y2": np.sum(observed * observed, axis=0),
        "count": count,
    }


def _fit_alignment_group(
    trajectories: Sequence[Trajectory],
    *,
    input_name: str,
    target_name: str,
    maximum_bias: float,
    candidates_s: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, dict[str, Any]]:
    statistics = []
    for index, trajectory in enumerate(trajectories):
        pair = _observation_rate_pairs(trajectory)
        statistics.append(
            (
                _source_group_key(trajectory, index),
                _alignment_statistics(
                    pair[input_name],
                    pair[target_name],
                    _interval_center_times(trajectory),
                    candidates_s,
                ),
            )
        )
    group_counts: dict[str, float] = {}
    for group_key, statistic in statistics:
        group_counts[group_key] = (
            group_counts.get(group_key, 0.0) + float(statistic["count"])
        )
    combined: dict[str, np.ndarray | float] = {}
    for name in statistics[0][1]:
        combined[name] = sum(
            (
                statistic[name] / group_counts[group_key]
                for group_key, statistic in statistics
            ),
            start=np.zeros_like(statistics[0][1][name]),
        )

    candidate_reports = []
    best: tuple[float, float, float, np.ndarray, np.ndarray, list[dict[str, float]]]
    best = (
        float("inf"),
        float("inf"),
        float("inf"),
        np.ones(3),
        np.zeros(3),
        [],
    )
    for candidate_index, delay_s in enumerate(candidates_s):
        scales = np.empty(3, dtype=np.float64)
        biases = np.empty(3, dtype=np.float64)
        axis_reports = []
        total_mean_square_error = 0.0
        for axis in range(3):
            scale, bias, unconstrained, mean_square_error = (
                _bounded_affine_from_statistics(
                    sum_x=float(combined["sum_x"][candidate_index, axis]),
                    sum_x2=float(combined["sum_x2"][candidate_index, axis]),
                    sum_xy=float(combined["sum_xy"][candidate_index, axis]),
                    sum_y=float(combined["sum_y"][axis]),
                    sum_y2=float(combined["sum_y2"][axis]),
                    count=float(combined["count"]),
                    maximum_bias=maximum_bias,
                )
            )
            scales[axis] = scale
            biases[axis] = bias
            total_mean_square_error += mean_square_error
            axis_reports.append(
                {
                    "axis": axis,
                    "rmse": float(np.sqrt(mean_square_error)),
                    "unconstrained_scale": float(unconstrained[0]),
                    "unconstrained_bias": float(unconstrained[1]),
                }
            )
        candidate_report = {
            "delay_s": float(delay_s),
            "vector_rmse": float(np.sqrt(total_mean_square_error)),
            "axes": axis_reports,
        }
        candidate_reports.append(candidate_report)
        key = (
            total_mean_square_error,
            abs(float(delay_s)),
            float(delay_s),
        )
        if key < best[:3]:
            best = (*key, scales, biases, axis_reports)
    _, _, selected_delay_s, scales, biases, axis_reports = best
    return selected_delay_s, scales, biases, {
        "selected_delay_s": selected_delay_s,
        "selected_boundary": bool(
            np.isclose(abs(selected_delay_s), MAXIMUM_ABSOLUTE_ALIGNMENT_S)
        ),
        "selected_axes": axis_reports,
        "candidate_scores": candidate_reports,
    }


def _make_alignment(
    trajectories: Sequence[Trajectory],
    candidates_s: np.ndarray,
) -> tuple[StateObservationAlignment, dict[str, Any]]:
    velocity_delay, velocity_scale, velocity_bias, velocity_report = (
        _fit_alignment_group(
            trajectories,
            input_name="pose_velocity",
            target_name="reported_velocity",
            maximum_bias=MAXIMUM_VELOCITY_BIAS_M_S,
            candidates_s=candidates_s,
        )
    )
    angular_delay, angular_scale, angular_bias, angular_report = (
        _fit_alignment_group(
            trajectories,
            input_name="pose_angular_rate",
            target_name="reported_angular_rate",
            maximum_bias=MAXIMUM_ANGULAR_RATE_BIAS_RAD_S,
            candidates_s=candidates_s,
        )
    )
    alignment = StateObservationAlignment(
        velocity_delay_s=velocity_delay,
        velocity_scale=velocity_scale,
        velocity_bias_m_s=velocity_bias,
        angular_rate_delay_s=angular_delay,
        angular_rate_scale=angular_scale,
        angular_rate_bias_rad_s=angular_bias,
    )
    return alignment, {
        "velocity": velocity_report,
        "angular_rate": angular_report,
    }


def _alignment_errors(
    model: StateObservationAlignment,
    trajectory: Trajectory,
) -> dict[str, float]:
    pair = _observation_rate_pairs(trajectory)
    center_times_s = _interval_center_times(trajectory)
    mask = _alignment_mask(center_times_s)
    predicted_velocity = (
        _shifted_response(
            pair["pose_velocity"], center_times_s, model.velocity_delay_s
        )
        * model.velocity_scale
        + model.velocity_bias_m_s
    )
    predicted_angular_rate = (
        _shifted_response(
            pair["pose_angular_rate"],
            center_times_s,
            model.angular_rate_delay_s,
        )
        * model.angular_rate_scale
        + model.angular_rate_bias_rad_s
    )
    velocity_error = predicted_velocity[mask] - pair["reported_velocity"][mask]
    angular_error = (
        predicted_angular_rate[mask] - pair["reported_angular_rate"][mask]
    )
    return {
        "position_velocity_vector_rmse_m_s": float(
            np.sqrt(np.mean(np.sum(velocity_error * velocity_error, axis=1)))
        ),
        "attitude_rate_vector_rmse_rad_s": float(
            np.sqrt(np.mean(np.sum(angular_error * angular_error, axis=1)))
        ),
    }


def fit_state_observation_alignment(
    trajectories: Sequence[Trajectory],
) -> StateObservationAlignmentFit:
    """Fit bounded state-channel delays on development trajectories only."""

    configuration_id, observation_source, interval_count = (
        _validate_observation_fit_data(trajectories)
    )
    candidate, candidate_details = _make_alignment(
        trajectories, ALIGNMENT_CANDIDATES_S
    )
    reference, reference_details = _make_alignment(
        trajectories, np.asarray([0.0], dtype=np.float64)
    )
    source_groups = {
        _source_group_key(trajectory, index)
        for index, trajectory in enumerate(trajectories)
    }
    training = evaluate_state_observation_alignment(
        candidate, reference, trajectories
    )
    return StateObservationAlignmentFit(
        candidate=candidate,
        instantaneous_reference=reference,
        report={
            "policy": ALIGNMENT_POLICY,
            "status": "research_only",
            "trajectory_count": len(trajectories),
            "interval_count": interval_count,
            "source_group_count": len(source_groups),
            "fit_weighting": "equal_source_group",
            "vehicle_configuration_id": configuration_id,
            "observation_source": observation_source,
            "protected_split_used": False,
            "candidate": candidate.to_dict(),
            "instantaneous_reference": reference.to_dict(),
            "fit_details": {
                "candidate": candidate_details,
                "instantaneous_reference": reference_details,
            },
            "training_comparison": training,
            "interpretation": (
                "This diagnostic estimates source-channel timing semantics, "
                "not physical actuator lag. A fitting-data shift must transfer "
                "to complete held-out flights before any rollout experiment."
            ),
        },
    )


def evaluate_state_observation_alignment(
    candidate: StateObservationAlignment,
    instantaneous_reference: StateObservationAlignment,
    trajectories: Sequence[Trajectory],
) -> dict[str, Any]:
    """Compare fixed time-aligned and instantaneous observation models."""

    if not trajectories:
        raise ValueError("at least one trajectory is required")
    per_trajectory = []
    candidate_metrics = []
    reference_metrics = []
    for trajectory in trajectories:
        candidate_error = _alignment_errors(candidate, trajectory)
        reference_error = _alignment_errors(
            instantaneous_reference, trajectory
        )
        candidate_metrics.append(candidate_error)
        reference_metrics.append(reference_error)
        per_trajectory.append(
            {
                "source_group": trajectory.labels.get("source_group"),
                "profile": trajectory.labels.get("profile"),
                "replicate": trajectory.labels.get("replicate"),
                "candidate": candidate_error,
                "instantaneous_reference": reference_error,
                "candidate_over_reference": {
                    name: (
                        candidate_error[name] / reference_error[name]
                        if reference_error[name] > 0.0
                        else None
                    )
                    for name in reference_error
                },
            }
        )
    names = tuple(reference_metrics[0])
    candidate_mean = {
        name: float(np.mean([metric[name] for metric in candidate_metrics]))
        for name in names
    }
    reference_mean = {
        name: float(np.mean([metric[name] for metric in reference_metrics]))
        for name in names
    }
    ratios = {
        name: (
            candidate_mean[name] / reference_mean[name]
            if reference_mean[name] > 0.0
            else None
        )
        for name in names
    }
    materiality_floors = {
        "position_velocity_vector_rmse_m_s": float(
            np.sqrt(3.0) * KINEMATIC_POSITION_RATE_FLOOR_M_S
        ),
        "attitude_rate_vector_rmse_rad_s": float(
            np.sqrt(3.0) * KINEMATIC_ATTITUDE_RATE_FLOOR_RAD_S
        ),
    }
    gate = observation_channel_transfer_gate(
        reference_error=reference_mean,
        candidate_over_reference=ratios,
        per_trajectory=per_trajectory,
        materiality_floors=materiality_floors,
        group_interior={
            "position_velocity_vector_rmse_m_s": bool(
                abs(candidate.velocity_delay_s)
                < MAXIMUM_ABSOLUTE_ALIGNMENT_S
            ),
            "attitude_rate_vector_rmse_rad_s": bool(
                abs(candidate.angular_rate_delay_s)
                < MAXIMUM_ABSOLUTE_ALIGNMENT_S
            ),
        },
    )
    return {
        "policy": "research_validation_state_observation_alignment_transfer_v2",
        "trajectory_count": len(trajectories),
        "candidate": candidate.to_dict(),
        "instantaneous_reference": instantaneous_reference.to_dict(),
        "candidate_error": candidate_mean,
        "instantaneous_reference_error": reference_mean,
        "candidate_over_reference": ratios,
        "gate": gate,
        "per_trajectory": per_trajectory,
    }
