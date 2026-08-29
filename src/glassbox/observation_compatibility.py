"""Bounded state-observation models for flight-data compatibility.

The research utilities here compare pose increments with reported world
velocity and body angular rate. They cover a static diagonal scale/bias
correction and a bounded first-order temporal response. Neither model is part of
rollout dynamics or the default fitter.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Sequence

import numpy as np
import numpy.typing as npt

from glassbox.data import Trajectory
from glassbox.evaluation import (
    KINEMATIC_ATTITUDE_RATE_FLOOR_RAD_S,
    KINEMATIC_POSITION_RATE_FLOOR_M_S,
    state_kinematic_compatibility_diagnostics,
)


MINIMUM_SCALE = 0.8
MAXIMUM_SCALE = 1.2
MAXIMUM_VELOCITY_BIAS_M_S = 1.0
MAXIMUM_ANGULAR_RATE_BIAS_RAD_S = 0.5
MINIMUM_FIT_INTERVALS = 20
MATERIAL_IMPROVEMENT_RATIO = 0.9
MAXIMUM_COMPATIBILITY_REGRESSION_RATIO = 1.05
PROTECTED_SPLITS = frozenset({"test", "validation", "holdout", "protected"})
CORRECTION_POLICY = "bounded_affine_state_compatibility_v1"
TEMPORAL_FILTER_POLICY = "bounded_first_order_observation_filter_v1"
MINIMUM_TIME_CONSTANT_S = 0.003
MAXIMUM_TIME_CONSTANT_S = 0.5
TEMPORAL_FILTER_CANDIDATES_S = np.concatenate(
    (
        np.asarray([0.0]),
        np.geomspace(
            MINIMUM_TIME_CONSTANT_S,
            MAXIMUM_TIME_CONSTANT_S,
            num=32,
        ),
    )
)
TEMPORAL_WARMUP_S = 0.5


@dataclass(frozen=True)
class StateObservationCorrection:
    """Diagonal scale and bias corrections for two measured state groups."""

    velocity_scale: npt.NDArray[np.float64]
    velocity_bias_m_s: npt.NDArray[np.float64]
    angular_rate_scale: npt.NDArray[np.float64]
    angular_rate_bias_rad_s: npt.NDArray[np.float64]

    def __post_init__(self) -> None:
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
        if np.any(self.velocity_scale < MINIMUM_SCALE) or np.any(
            self.velocity_scale > MAXIMUM_SCALE
        ):
            raise ValueError("velocity_scale exceeds the maintained bounds")
        if np.any(self.angular_rate_scale < MINIMUM_SCALE) or np.any(
            self.angular_rate_scale > MAXIMUM_SCALE
        ):
            raise ValueError("angular_rate_scale exceeds the maintained bounds")
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
            "policy": CORRECTION_POLICY,
            "velocity_scale": self.velocity_scale.tolist(),
            "velocity_bias_m_s": self.velocity_bias_m_s.tolist(),
            "angular_rate_scale": self.angular_rate_scale.tolist(),
            "angular_rate_bias_rad_s": self.angular_rate_bias_rad_s.tolist(),
            "bounds": {
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
class StateObservationCorrectionFit:
    correction: StateObservationCorrection
    report: dict[str, Any]


@dataclass(frozen=True)
class FirstOrderObservationFilter:
    """A bounded forward model from pose increments to reported state rates."""

    velocity_time_constant_s: npt.NDArray[np.float64]
    velocity_scale: npt.NDArray[np.float64]
    velocity_bias_m_s: npt.NDArray[np.float64]
    angular_rate_time_constant_s: npt.NDArray[np.float64]
    angular_rate_scale: npt.NDArray[np.float64]
    angular_rate_bias_rad_s: npt.NDArray[np.float64]

    def __post_init__(self) -> None:
        for name in (
            "velocity_time_constant_s",
            "velocity_scale",
            "velocity_bias_m_s",
            "angular_rate_time_constant_s",
            "angular_rate_scale",
            "angular_rate_bias_rad_s",
        ):
            values = np.asarray(getattr(self, name), dtype=np.float64).copy()
            if values.shape != (3,) or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must contain three finite values")
            values.setflags(write=False)
            object.__setattr__(self, name, values)
        for name in (
            "velocity_time_constant_s",
            "angular_rate_time_constant_s",
        ):
            values = getattr(self, name)
            if np.any(values < 0.0) or np.any(
                values > MAXIMUM_TIME_CONSTANT_S
            ):
                raise ValueError(f"{name} exceeds the maintained bounds")
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
            "policy": TEMPORAL_FILTER_POLICY,
            "velocity_time_constant_s": self.velocity_time_constant_s.tolist(),
            "velocity_scale": self.velocity_scale.tolist(),
            "velocity_bias_m_s": self.velocity_bias_m_s.tolist(),
            "angular_rate_time_constant_s": (
                self.angular_rate_time_constant_s.tolist()
            ),
            "angular_rate_scale": self.angular_rate_scale.tolist(),
            "angular_rate_bias_rad_s": (
                self.angular_rate_bias_rad_s.tolist()
            ),
            "bounds": {
                "time_constant_s": [0.0, MAXIMUM_TIME_CONSTANT_S],
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
class FirstOrderObservationFilterFit:
    candidate: FirstOrderObservationFilter
    instantaneous_reference: FirstOrderObservationFilter
    report: dict[str, Any]


def _attitude_increment(quaternion_wxyz: np.ndarray) -> np.ndarray:
    predicted = quaternion_wxyz[:-1] / np.linalg.norm(
        quaternion_wxyz[:-1], axis=1, keepdims=True
    )
    observed = quaternion_wxyz[1:] / np.linalg.norm(
        quaternion_wxyz[1:], axis=1, keepdims=True
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
    norm = np.linalg.norm(relative_xyz, axis=1)
    angle = 2.0 * np.arctan2(norm, np.clip(relative_w, 0.0, 1.0))
    scale = np.divide(
        angle,
        norm,
        out=np.full_like(angle, 2.0),
        where=norm > 1e-12,
    )
    return relative_xyz * scale[:, None]


def _rotation_matrices(quaternion_wxyz: np.ndarray) -> np.ndarray:
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


def _compatibility_system(
    trajectory: Trajectory,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    states = trajectory.states
    count = len(trajectory.controls)
    identity = np.eye(3, dtype=np.float64)

    velocity_design = np.zeros((count, 3, 6), dtype=np.float64)
    average_velocity = 0.5 * (states[:-1, 3:6] + states[1:, 3:6])
    velocity_design[:, :, 0:3] = identity[None, :, :] * average_velocity[:, None, :]
    velocity_design[:, :, 3:6] = identity
    velocity_target = np.diff(states[:, 0:3], axis=0) / trajectory.nominal_dt_s

    rotations = _rotation_matrices(states[:, 6:10])
    relative_rotation = np.einsum(
        "nji,njk->nik", rotations[:-1], rotations[1:]
    )
    initial_rate_design = identity[None, :, :] * states[:-1, None, 10:13]
    terminal_rate_design = np.einsum(
        "nij,njk->nik",
        relative_rotation,
        identity[None, :, :] * states[1:, None, 10:13],
    )
    angular_design = np.empty((count, 3, 6), dtype=np.float64)
    angular_design[:, :, 0:3] = 0.5 * (
        initial_rate_design + terminal_rate_design
    )
    angular_design[:, :, 3:6] = 0.5 * (
        identity[None, :, :] + relative_rotation
    )
    angular_target = _attitude_increment(states[:, 6:10]) / trajectory.nominal_dt_s
    return (
        velocity_design.reshape((-1, 6)),
        velocity_target.reshape(-1),
        angular_design.reshape((-1, 6)),
        angular_target.reshape(-1),
    )


def _bounded_solution(
    design: np.ndarray,
    target: np.ndarray,
    *,
    maximum_bias: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unconstrained, *_ = np.linalg.lstsq(design, target, rcond=None)
    scale = np.clip(unconstrained[:3], MINIMUM_SCALE, MAXIMUM_SCALE)
    residual_target = target - design[:, :3] @ scale
    bias, *_ = np.linalg.lstsq(design[:, 3:6], residual_target, rcond=None)
    bias = np.clip(bias, -maximum_bias, maximum_bias)
    bounded = np.concatenate((scale, bias))
    return scale, bias, unconstrained


def _mean_compatibility(
    trajectories: Sequence[Trajectory],
) -> dict[str, float]:
    reports = [
        state_kinematic_compatibility_diagnostics(trajectory)
        for trajectory in trajectories
    ]
    valid = [report for report in reports if report["status"] == "ok"]
    if not valid:
        return {
            "position_velocity_vector_rmse_m_s": float("nan"),
            "attitude_rate_vector_rmse_rad_s": float("nan"),
        }
    return {
        "position_velocity_vector_rmse_m_s": float(
            np.mean(
                [
                    report["position_velocity_compatibility"]["vector_rmse"]
                    for report in valid
                ]
            )
        ),
        "attitude_rate_vector_rmse_rad_s": float(
            np.mean(
                [
                    report["attitude_rate_compatibility"]["vector_rmse"]
                    for report in valid
                ]
            )
        ),
    }


def apply_state_observation_correction(
    trajectory: Trajectory,
    correction: StateObservationCorrection,
) -> Trajectory:
    """Apply one audited correction without modifying pose or other channels."""

    if "state_observation_correction" in trajectory.provenance:
        raise ValueError("trajectory already has a state observation correction")
    states = trajectory.states.copy()
    states[:, 3:6] = (
        states[:, 3:6] * correction.velocity_scale
        + correction.velocity_bias_m_s
    )
    states[:, 10:13] = (
        states[:, 10:13] * correction.angular_rate_scale
        + correction.angular_rate_bias_rad_s
    )
    provenance = dict(trajectory.provenance)
    provenance["state_observation_correction"] = correction.to_dict()
    return Trajectory(
        time_s=trajectory.time_s,
        states=states,
        controls=trajectory.controls,
        exogenous=trajectory.exogenous,
        observations=trajectory.observations,
        spec=replace(
            trajectory.spec,
            observation_source=(
                f"{trajectory.spec.observation_source}+{CORRECTION_POLICY}"
            ),
        ),
        labels=trajectory.labels,
        provenance=provenance,
    )


def fit_state_observation_correction(
    trajectories: Sequence[Trajectory],
) -> StateObservationCorrectionFit:
    """Fit one bounded correction from development trajectories only."""

    if not trajectories:
        raise ValueError("at least one trajectory is required")
    protected = [
        str(trajectory.labels.get("benchmark_split"))
        for trajectory in trajectories
        if str(trajectory.labels.get("benchmark_split", "")).lower()
        in PROTECTED_SPLITS
    ]
    if protected:
        raise ValueError(
            "state observation correction fitting rejects protected benchmark "
            f"splits: {', '.join(sorted(set(protected)))}"
        )
    reference_vehicle = trajectories[0].spec.vehicle
    reference_source = trajectories[0].spec.observation_source
    if any(
        trajectory.spec.vehicle != reference_vehicle
        or trajectory.spec.observation_source != reference_source
        for trajectory in trajectories[1:]
    ):
        raise ValueError(
            "state observation correction requires one vehicle configuration "
            "and observation source"
        )
    interval_count = sum(len(trajectory.controls) for trajectory in trajectories)
    if interval_count < MINIMUM_FIT_INTERVALS:
        raise ValueError(
            "state observation correction requires at least "
            f"{MINIMUM_FIT_INTERVALS} intervals"
        )

    systems = [_compatibility_system(trajectory) for trajectory in trajectories]
    velocity_design = np.concatenate([system[0] for system in systems])
    velocity_target = np.concatenate([system[1] for system in systems])
    angular_design = np.concatenate([system[2] for system in systems])
    angular_target = np.concatenate([system[3] for system in systems])
    velocity_scale, velocity_bias, velocity_unconstrained = _bounded_solution(
        velocity_design,
        velocity_target,
        maximum_bias=MAXIMUM_VELOCITY_BIAS_M_S,
    )
    angular_scale, angular_bias, angular_unconstrained = _bounded_solution(
        angular_design,
        angular_target,
        maximum_bias=MAXIMUM_ANGULAR_RATE_BIAS_RAD_S,
    )
    correction = StateObservationCorrection(
        velocity_scale=velocity_scale,
        velocity_bias_m_s=velocity_bias,
        angular_rate_scale=angular_scale,
        angular_rate_bias_rad_s=angular_bias,
    )
    corrected = [
        apply_state_observation_correction(trajectory, correction)
        for trajectory in trajectories
    ]
    before = _mean_compatibility(trajectories)
    after = _mean_compatibility(corrected)
    report = {
        "policy": CORRECTION_POLICY,
        "status": "research_only",
        "trajectory_count": len(trajectories),
        "interval_count": interval_count,
        "vehicle_configuration_id": reference_vehicle.configuration_id,
        "observation_source": reference_source,
        "protected_split_used": False,
        "correction": correction.to_dict(),
        "unconstrained_solution": {
            "velocity_scale": velocity_unconstrained[:3].tolist(),
            "velocity_bias_m_s": velocity_unconstrained[3:6].tolist(),
            "angular_rate_scale": angular_unconstrained[:3].tolist(),
            "angular_rate_bias_rad_s": angular_unconstrained[3:6].tolist(),
        },
        "training_compatibility": {
            "before": before,
            "after": after,
            "after_over_before": {
                name: (
                    after[name] / before[name]
                    if np.isfinite(before[name]) and before[name] > 0.0
                    else None
                )
                for name in before
            },
        },
        "interpretation": (
            "The correction estimates only bounded diagonal scale and constant "
            "bias terms. Compatibility improvement on the fitting trajectories "
            "is not promotion evidence; coefficients must transfer to complete "
            "held-out flights before any rollout experiment."
        ),
    }
    return StateObservationCorrectionFit(correction=correction, report=report)


def evaluate_state_observation_correction(
    correction: StateObservationCorrection,
    trajectories: Sequence[Trajectory],
) -> dict[str, Any]:
    """Evaluate a fixed correction on untouched complete trajectories."""

    if not trajectories:
        raise ValueError("at least one trajectory is required")
    corrected = [
        apply_state_observation_correction(trajectory, correction)
        for trajectory in trajectories
    ]
    per_trajectory = []
    for trajectory, corrected_trajectory in zip(trajectories, corrected):
        before = _mean_compatibility([trajectory])
        after = _mean_compatibility([corrected_trajectory])
        per_trajectory.append(
            {
                "source_group": trajectory.labels.get("source_group"),
                "profile": trajectory.labels.get("profile"),
                "replicate": trajectory.labels.get("replicate"),
                "before": before,
                "after": after,
                "after_over_before": {
                    name: (
                        after[name] / before[name]
                        if np.isfinite(before[name]) and before[name] > 0.0
                        else None
                    )
                    for name in before
                },
            }
        )
    before = _mean_compatibility(trajectories)
    after = _mean_compatibility(corrected)
    ratios = {
        name: (
            after[name] / before[name]
            if np.isfinite(before[name]) and before[name] > 0.0
            else None
        )
        for name in before
    }
    materiality_floors = {
        "position_velocity_vector_rmse_m_s": float(
            np.sqrt(3.0) * KINEMATIC_POSITION_RATE_FLOOR_M_S
        ),
        "attitude_rate_vector_rmse_rad_s": float(
            np.sqrt(3.0) * KINEMATIC_ATTITUDE_RATE_FLOOR_RAD_S
        ),
    }
    eligible_groups = [
        name
        for name, value in before.items()
        if np.isfinite(value) and value > materiality_floors[name]
    ]
    eligible_ratios = [
        ratios[name] for name in eligible_groups if ratios[name] is not None
    ]
    return {
        "policy": "held_out_state_compatibility_transfer_v1",
        "trajectory_count": len(trajectories),
        "correction": correction.to_dict(),
        "before": before,
        "after": after,
        "after_over_before": ratios,
        "gate": {
            "material_improvement_ratio": MATERIAL_IMPROVEMENT_RATIO,
            "maximum_regression_ratio": (
                MAXIMUM_COMPATIBILITY_REGRESSION_RATIO
            ),
            "materiality_floors": materiality_floors,
            "eligible_groups": eligible_groups,
            "already_consistent_groups": [
                name for name in before if name not in eligible_groups
            ],
            "all_groups_improve_materially": bool(
                eligible_ratios
                and all(
                    ratio <= MATERIAL_IMPROVEMENT_RATIO
                    for ratio in eligible_ratios
                )
            ),
            "no_group_regresses": bool(
                all(
                    ratio <= MAXIMUM_COMPATIBILITY_REGRESSION_RATIO
                    for ratio in eligible_ratios
                )
            ),
        },
        "per_trajectory": per_trajectory,
    }


def _observation_rate_pairs(
    trajectory: Trajectory,
) -> dict[str, np.ndarray]:
    """Return pose-implied inputs and aligned reported rate observations."""

    states = trajectory.states
    interval_s = np.diff(trajectory.time_s)
    pose_velocity = np.diff(states[:, 0:3], axis=0) / interval_s[:, None]
    reported_velocity = 0.5 * (states[:-1, 3:6] + states[1:, 3:6])

    rotations = _rotation_matrices(states[:, 6:10])
    relative_rotation = np.einsum(
        "nji,njk->nik", rotations[:-1], rotations[1:]
    )
    terminal_rate = np.einsum(
        "nij,nj->ni", relative_rotation, states[1:, 10:13]
    )
    pose_angular_rate = (
        _attitude_increment(states[:, 6:10]) / interval_s[:, None]
    )
    reported_angular_rate = 0.5 * (states[:-1, 10:13] + terminal_rate)
    center_step_s = np.empty_like(interval_s)
    center_step_s[0] = interval_s[0]
    if len(interval_s) > 1:
        center_step_s[1:] = 0.5 * (interval_s[:-1] + interval_s[1:])
    return {
        "center_step_s": center_step_s,
        "pose_velocity": pose_velocity,
        "reported_velocity": reported_velocity,
        "pose_angular_rate": pose_angular_rate,
        "reported_angular_rate": reported_angular_rate,
    }


def _first_order_response(
    values: np.ndarray,
    center_step_s: np.ndarray,
    time_constant_s: np.ndarray,
) -> np.ndarray:
    response = np.empty_like(values)
    response[0] = values[0]
    for index in range(1, len(values)):
        decay = np.zeros_like(time_constant_s)
        temporal = time_constant_s > 0.0
        decay[temporal] = np.exp(
            -center_step_s[index] / time_constant_s[temporal]
        )
        response[index] = (
            decay * response[index - 1] + (1.0 - decay) * values[index]
        )
    return response


def _warmup_start(trajectory: Trajectory) -> int:
    interval_count = len(trajectory.controls)
    elapsed = trajectory.time_s[1:] - trajectory.time_s[0]
    half_second = int(np.searchsorted(elapsed, TEMPORAL_WARMUP_S, side="left"))
    return min(half_second, interval_count // 10)


def _temporal_sufficient_statistics(
    values: np.ndarray,
    target: np.ndarray,
    center_step_s: np.ndarray,
    start: int,
    candidates_s: np.ndarray,
) -> dict[str, np.ndarray | float]:
    """Accumulate affine-fit statistics for every candidate in one pass."""

    response = np.repeat(values[0][None, :], len(candidates_s), axis=0)
    sum_x = np.zeros_like(response)
    sum_x2 = np.zeros_like(response)
    sum_xy = np.zeros_like(response)
    sum_y = np.zeros(3, dtype=np.float64)
    sum_y2 = np.zeros(3, dtype=np.float64)
    count = 0
    for index in range(len(values)):
        if index > 0:
            decay = np.zeros_like(candidates_s)
            temporal = candidates_s > 0.0
            decay[temporal] = np.exp(
                -center_step_s[index] / candidates_s[temporal]
            )
            response = (
                decay[:, None] * response
                + (1.0 - decay[:, None]) * values[index][None, :]
            )
        if index < start:
            continue
        observed = target[index]
        sum_x += response
        sum_x2 += response * response
        sum_xy += response * observed[None, :]
        sum_y += observed
        sum_y2 += observed * observed
        count += 1
    return {
        "sum_x": sum_x,
        "sum_x2": sum_x2,
        "sum_xy": sum_xy,
        "sum_y": sum_y,
        "sum_y2": sum_y2,
        "count": float(count),
    }


def _bounded_affine_from_statistics(
    *,
    sum_x: float,
    sum_x2: float,
    sum_xy: float,
    sum_y: float,
    sum_y2: float,
    count: float,
    maximum_bias: float,
) -> tuple[float, float, np.ndarray, float]:
    normal = np.asarray([[sum_x2, sum_x], [sum_x, count]])
    rhs = np.asarray([sum_xy, sum_y])
    unconstrained, *_ = np.linalg.lstsq(normal, rhs, rcond=None)
    scale = float(np.clip(unconstrained[0], MINIMUM_SCALE, MAXIMUM_SCALE))
    bias = float(
        np.clip(
            (sum_y - scale * sum_x) / count,
            -maximum_bias,
            maximum_bias,
        )
    )
    squared_error = (
        scale * scale * sum_x2
        + 2.0 * scale * bias * sum_x
        + count * bias * bias
        - 2.0 * scale * sum_xy
        - 2.0 * bias * sum_y
        + sum_y2
    )
    return scale, bias, unconstrained, float(max(squared_error / count, 0.0))


def _fit_temporal_group(
    trajectories: Sequence[Trajectory],
    *,
    input_name: str,
    target_name: str,
    maximum_bias: float,
    candidates_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    statistics = []
    for trajectory_index, trajectory in enumerate(trajectories):
        pair = _observation_rate_pairs(trajectory)
        source_group = trajectory.labels.get("source_group")
        group_key = (
            f"source_group:{source_group}"
            if source_group is not None
            else f"trajectory:{trajectory_index}"
        )
        statistics.append(
            (
                group_key,
                _temporal_sufficient_statistics(
                    pair[input_name],
                    pair[target_name],
                    pair["center_step_s"],
                    _warmup_start(trajectory),
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
    selected_time_constants = np.empty(3, dtype=np.float64)
    scales = np.empty(3, dtype=np.float64)
    biases = np.empty(3, dtype=np.float64)
    axis_reports: list[dict[str, Any]] = []
    for axis in range(3):
        best: tuple[float, float, float, np.ndarray, float] | None = None
        candidate_scores: list[dict[str, float]] = []
        for candidate_index, time_constant_s in enumerate(candidates_s):
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
            candidate_scores.append(
                {
                    "time_constant_s": float(time_constant_s),
                    "rmse": float(np.sqrt(mean_square_error)),
                }
            )
            result = (
                mean_square_error,
                float(time_constant_s),
                scale,
                unconstrained,
                bias,
            )
            if best is None or result[:2] < best[:2]:
                best = result
        assert best is not None
        mean_square_error, time_constant_s, scale, unconstrained, bias = best
        selected_time_constants[axis] = time_constant_s
        scales[axis] = scale
        biases[axis] = bias
        axis_reports.append(
            {
                "axis": axis,
                "selected_time_constant_s": time_constant_s,
                "selected_rmse": float(np.sqrt(mean_square_error)),
                "unconstrained_scale": float(unconstrained[0]),
                "unconstrained_bias": float(unconstrained[1]),
                "selected_maximum_time_constant": bool(
                    np.isclose(time_constant_s, MAXIMUM_TIME_CONSTANT_S)
                ),
                "candidate_scores": candidate_scores,
            }
        )
    return selected_time_constants, scales, biases, axis_reports


def _validate_observation_fit_data(
    trajectories: Sequence[Trajectory],
) -> tuple[str, str, int]:
    if not trajectories:
        raise ValueError("at least one trajectory is required")
    protected = [
        str(trajectory.labels.get("benchmark_split"))
        for trajectory in trajectories
        if str(trajectory.labels.get("benchmark_split", "")).lower()
        in PROTECTED_SPLITS
    ]
    if protected:
        raise ValueError(
            "observation model fitting rejects protected benchmark splits: "
            f"{', '.join(sorted(set(protected)))}"
        )
    reference_vehicle = trajectories[0].spec.vehicle
    reference_source = trajectories[0].spec.observation_source
    if any(
        trajectory.spec.vehicle != reference_vehicle
        or trajectory.spec.observation_source != reference_source
        for trajectory in trajectories[1:]
    ):
        raise ValueError(
            "observation model fitting requires one vehicle configuration "
            "and observation source"
        )
    interval_count = sum(len(trajectory.controls) for trajectory in trajectories)
    if interval_count < MINIMUM_FIT_INTERVALS:
        raise ValueError(
            "observation model fitting requires at least "
            f"{MINIMUM_FIT_INTERVALS} intervals"
        )
    return reference_vehicle.configuration_id, reference_source, interval_count


def _make_first_order_filter(
    trajectories: Sequence[Trajectory],
    candidates_s: np.ndarray,
) -> tuple[FirstOrderObservationFilter, dict[str, Any]]:
    velocity_tau, velocity_scale, velocity_bias, velocity_report = (
        _fit_temporal_group(
            trajectories,
            input_name="pose_velocity",
            target_name="reported_velocity",
            maximum_bias=MAXIMUM_VELOCITY_BIAS_M_S,
            candidates_s=candidates_s,
        )
    )
    angular_tau, angular_scale, angular_bias, angular_report = (
        _fit_temporal_group(
            trajectories,
            input_name="pose_angular_rate",
            target_name="reported_angular_rate",
            maximum_bias=MAXIMUM_ANGULAR_RATE_BIAS_RAD_S,
            candidates_s=candidates_s,
        )
    )
    model = FirstOrderObservationFilter(
        velocity_time_constant_s=velocity_tau,
        velocity_scale=velocity_scale,
        velocity_bias_m_s=velocity_bias,
        angular_rate_time_constant_s=angular_tau,
        angular_rate_scale=angular_scale,
        angular_rate_bias_rad_s=angular_bias,
    )
    return model, {
        "velocity_axes": velocity_report,
        "angular_rate_axes": angular_report,
    }


def _observation_filter_errors(
    model: FirstOrderObservationFilter,
    trajectory: Trajectory,
) -> dict[str, float]:
    pair = _observation_rate_pairs(trajectory)
    start = _warmup_start(trajectory)
    predicted_velocity = (
        _first_order_response(
            pair["pose_velocity"],
            pair["center_step_s"],
            model.velocity_time_constant_s,
        )
        * model.velocity_scale
        + model.velocity_bias_m_s
    )
    predicted_angular_rate = (
        _first_order_response(
            pair["pose_angular_rate"],
            pair["center_step_s"],
            model.angular_rate_time_constant_s,
        )
        * model.angular_rate_scale
        + model.angular_rate_bias_rad_s
    )
    velocity_error = predicted_velocity[start:] - pair["reported_velocity"][start:]
    angular_error = (
        predicted_angular_rate[start:] - pair["reported_angular_rate"][start:]
    )
    return {
        "position_velocity_vector_rmse_m_s": float(
            np.sqrt(np.mean(np.sum(velocity_error * velocity_error, axis=1)))
        ),
        "attitude_rate_vector_rmse_rad_s": float(
            np.sqrt(np.mean(np.sum(angular_error * angular_error, axis=1)))
        ),
    }


def fit_first_order_observation_filter(
    trajectories: Sequence[Trajectory],
) -> FirstOrderObservationFilterFit:
    """Fit one maintained temporal model and its instantaneous ablation."""

    configuration_id, observation_source, interval_count = (
        _validate_observation_fit_data(trajectories)
    )
    candidate, candidate_details = _make_first_order_filter(
        trajectories, TEMPORAL_FILTER_CANDIDATES_S
    )
    reference, reference_details = _make_first_order_filter(
        trajectories, np.asarray([0.0], dtype=np.float64)
    )
    training = evaluate_first_order_observation_filter(
        candidate, reference, trajectories
    )
    report = {
        "policy": TEMPORAL_FILTER_POLICY,
        "status": "research_only",
        "trajectory_count": len(trajectories),
        "interval_count": interval_count,
        "source_group_count": len(
            {
                (
                    f"source_group:{trajectory.labels['source_group']}"
                    if trajectory.labels.get("source_group") is not None
                    else f"trajectory:{index}"
                )
                for index, trajectory in enumerate(trajectories)
            }
        ),
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
            "The candidate differs from the instantaneous reference only by "
            "one bounded causal time constant per axis. Selection on fitting "
            "data is not promotion evidence; the frozen model must improve "
            "complete held-out flights across platforms."
        ),
    }
    return FirstOrderObservationFilterFit(
        candidate=candidate,
        instantaneous_reference=reference,
        report=report,
    )


def evaluate_first_order_observation_filter(
    candidate: FirstOrderObservationFilter,
    instantaneous_reference: FirstOrderObservationFilter,
    trajectories: Sequence[Trajectory],
) -> dict[str, Any]:
    """Compare fixed temporal and memoryless observation models."""

    if not trajectories:
        raise ValueError("at least one trajectory is required")
    per_trajectory = []
    candidate_metrics = []
    reference_metrics = []
    for trajectory in trajectories:
        candidate_error = _observation_filter_errors(candidate, trajectory)
        reference_error = _observation_filter_errors(
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
    eligible_groups = [
        name
        for name in names
        if reference_mean[name] > materiality_floors[name]
    ]
    eligible_ratios = [ratios[name] for name in eligible_groups]
    interior = bool(
        np.all(
            candidate.velocity_time_constant_s < MAXIMUM_TIME_CONSTANT_S
        )
        and np.all(
            candidate.angular_rate_time_constant_s < MAXIMUM_TIME_CONSTANT_S
        )
    )
    material = bool(
        eligible_ratios
        and all(
            ratio is not None and ratio <= MATERIAL_IMPROVEMENT_RATIO
            for ratio in eligible_ratios
        )
    )
    no_regression = bool(
        all(
            ratio is not None
            and ratio <= MAXIMUM_COMPATIBILITY_REGRESSION_RATIO
            for ratio in eligible_ratios
        )
        and all(
            trajectory_report["candidate_over_reference"][name] is not None
            and trajectory_report["candidate_over_reference"][name]
            <= MAXIMUM_COMPATIBILITY_REGRESSION_RATIO
            for trajectory_report in per_trajectory
            for name in eligible_groups
        )
    )
    return {
        "policy": "held_out_first_order_observation_transfer_v1",
        "trajectory_count": len(trajectories),
        "candidate": candidate.to_dict(),
        "instantaneous_reference": instantaneous_reference.to_dict(),
        "candidate_error": candidate_mean,
        "instantaneous_reference_error": reference_mean,
        "candidate_over_reference": ratios,
        "gate": {
            "material_improvement_ratio": MATERIAL_IMPROVEMENT_RATIO,
            "maximum_regression_ratio": MAXIMUM_COMPATIBILITY_REGRESSION_RATIO,
            "materiality_floors": materiality_floors,
            "eligible_groups": eligible_groups,
            "already_consistent_groups": [
                name for name in names if name not in eligible_groups
            ],
            "all_groups_improve_materially": material,
            "no_group_regresses": no_regression,
            "regression_guard_scope": (
                "aggregate and every complete held-out trajectory"
            ),
            "time_constants_interior": interior,
            "passes": bool(material and no_regression and interior),
        },
        "per_trajectory": per_trajectory,
    }
