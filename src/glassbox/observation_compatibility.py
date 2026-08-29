"""Bounded state-observation corrections for flight-data compatibility.

The correction model is intentionally small and interpretable. It estimates a
diagonal scale and constant bias for world velocity and body angular rate so
those measured channels agree more closely with position and attitude
increments. It is a research-stage observation model, not part of rollout
dynamics or the default fitter.
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
        "position_velocity_vector_rmse_m_s": (
            np.sqrt(3.0) * KINEMATIC_POSITION_RATE_FLOOR_M_S
        ),
        "attitude_rate_vector_rmse_rad_s": (
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
