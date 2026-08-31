"""Offline grouped predictive ensembles and uncertainty diagnostics.

This module deliberately describes an empirical ensemble, not a Bayesian
posterior. Members differ because complete, independent source groups are
resampled. A separate group partition scales their disagreement before an outer
partition is evaluated. The result measures corpus sensitivity and empirical
scale transfer; it does not include process noise, observation noise, or model
structures that were not fitted.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import numpy as np

from glassbox.data import Trajectory, duration_to_steps, load_trajectory_npz
from glassbox.dynamics import ModelParams, model_family
from glassbox.evaluation import attitude_innovation, windowed_rollout_predictions
from glassbox.fit_cli import fit_trajectory_artifacts
from glassbox.model_io import load_dynamics_model, save_dynamics_model
from glassbox.runtime import runtime_spec_from_fit_report

DEFAULT_COVERAGE_LEVELS = (0.5, 0.8, 0.9)
DEFAULT_ENSEMBLE_MEMBER_COUNT = 8
PREDICTIVE_ENSEMBLE_FORMAT_VERSION = 4
PREDICTIVE_ENSEMBLE_METHOD = "balanced_group_calibrated_bootstrap_v4"
DISAGREEMENT_CALIBRATION_METHOD = "source_group_split_scale_v1"
PREDICTIVE_GROUPS = (
    ("position", "m", slice(0, 3)),
    ("velocity", "m/s", slice(3, 6)),
    ("attitude", "rad", slice(6, 10)),
    ("angular_velocity", "rad/s", slice(10, 13)),
)


@dataclass(frozen=True)
class PredictiveEnsemble:
    """A homogeneous set of fitted models supported by grouped resamples."""

    members: tuple[ModelParams, ...]
    member_ids: tuple[str, ...]
    method: str = PREDICTIVE_ENSEMBLE_METHOD

    def __post_init__(self) -> None:
        if len(self.members) < 2:
            raise ValueError("a predictive ensemble requires at least two members")
        if len(self.member_ids) != len(self.members):
            raise ValueError("member_ids must match ensemble members")
        if len(set(self.member_ids)) != len(self.member_ids):
            raise ValueError("member_ids must be unique")
        families = {model_family(member).key for member in self.members}
        if len(families) != 1:
            raise ValueError("predictive ensemble members must use one model family")
        if self.method != PREDICTIVE_ENSEMBLE_METHOD:
            raise ValueError(f"unsupported predictive ensemble method: {self.method}")

    @property
    def member_count(self) -> int:
        return len(self.members)

    @property
    def unique_member_count(self) -> int:
        return len({_parameter_digest(member) for member in self.members})


def _parameter_digest(params: ModelParams) -> str:
    digest = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(params):
        values = np.asarray(leaf)
        digest.update(str(values.dtype).encode())
        digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


def _prediction_member_digest(
    predictions: np.ndarray, finite: np.ndarray
) -> str:
    values = np.asarray(predictions, dtype=np.float64).copy()
    for sample_index in np.flatnonzero(finite):
        quaternion = values[sample_index, 6:10]
        quaternion /= np.linalg.norm(quaternion)
        sign_index = int(np.argmax(np.abs(quaternion)))
        if quaternion[sign_index] < 0.0:
            quaternion *= -1.0
        values[sample_index, 6:10] = quaternion
    values[~finite] = 0.0
    digest = hashlib.sha256()
    digest.update(np.asarray(finite, dtype=np.uint8).tobytes())
    digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


def grouped_bootstrap_multiplicities(
    groups: Sequence[str | int],
    *,
    strata: Mapping[str | int, str | int] | None = None,
    member_count: int = DEFAULT_ENSEMBLE_MEMBER_COUNT,
    seed: int = 0,
) -> tuple[dict[str | int, int], ...]:
    """Draw complete groups with replacement, independently within strata.

    Each stratum retains its original number of draws. This prevents a profile
    with many source groups from disappearing merely because another profile was
    sampled repeatedly.
    """

    ordered_groups = tuple(dict.fromkeys(groups))
    if len(ordered_groups) < 2:
        raise ValueError("group bootstrap requires at least two source groups")
    if len(ordered_groups) != len(groups):
        raise ValueError("groups must contain each source group exactly once")
    if member_count < 2:
        raise ValueError("member_count must be at least two")
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")

    if strata is None:
        group_strata = {group: "all" for group in ordered_groups}
    else:
        if set(strata) != set(ordered_groups):
            raise ValueError("strata must contain exactly the source groups")
        group_strata = dict(strata)
    stratum_order = tuple(dict.fromkeys(group_strata[group] for group in ordered_groups))
    groups_by_stratum = {
        stratum: tuple(
            group for group in ordered_groups if group_strata[group] == stratum
        )
        for stratum in stratum_order
    }

    rng = np.random.default_rng(seed)
    members = []
    for _ in range(member_count):
        counts = {group: 0 for group in ordered_groups}
        for stratum in stratum_order:
            candidates = groups_by_stratum[stratum]
            sampled_indices = rng.integers(
                0, len(candidates), size=len(candidates)
            )
            for candidate_index in sampled_indices.tolist():
                group = candidates[candidate_index]
                counts[group] += 1
        members.append({group: count for group, count in counts.items() if count})
    return tuple(members)


def _normalized_quaternions(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return np.divide(
        values,
        norms,
        out=np.full_like(values, np.nan, dtype=np.float64),
        where=norms > 0.0,
    )


def _ensemble_centers(
    predictions: np.ndarray, finite: np.ndarray
) -> np.ndarray:
    """Return samplewise Euclidean means and sign-invariant quaternion means."""

    _, sample_count, _ = predictions.shape
    centers = np.full((sample_count, 13), np.nan, dtype=np.float64)
    for sample_index in range(sample_count):
        valid = finite[:, sample_index]
        if not np.any(valid):
            continue
        sample = predictions[valid, sample_index]
        centers[sample_index, 0:6] = np.mean(sample[:, 0:6], axis=0)
        centers[sample_index, 10:13] = np.mean(sample[:, 10:13], axis=0)
        quaternions = _normalized_quaternions(sample[:, 6:10])
        reference = quaternions[0]
        signs = np.where(quaternions @ reference < 0.0, -1.0, 1.0)
        mean = np.mean(quaternions * signs[:, None], axis=0)
        norm = np.linalg.norm(mean)
        if norm > 0.0:
            centers[sample_index, 6:10] = mean / norm
    return centers


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def _spearman_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    finite = np.isfinite(left) & np.isfinite(right)
    left = left[finite]
    right = right[finite]
    if len(left) < 3:
        return None
    left_rank = _average_ranks(left)
    right_rank = _average_ranks(right)
    if np.std(left_rank) <= 0.0 or np.std(right_rank) <= 0.0:
        return None
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _group_coordinates(
    values: np.ndarray,
    centers: np.ndarray,
    group_name: str,
    state_slice: slice,
) -> np.ndarray:
    if group_name != "attitude":
        return values[..., state_slice] - centers[None, :, state_slice]
    member_count, sample_count, _ = values.shape
    result = np.full((member_count, sample_count, 3), np.nan, dtype=np.float64)
    for member_index in range(member_count):
        valid = np.all(np.isfinite(values[member_index, :, 6:10]), axis=1) & np.all(
            np.isfinite(centers[:, 6:10]), axis=1
        )
        if np.any(valid):
            result[member_index, valid] = attitude_innovation(
                centers[valid, 6:10], values[member_index, valid, 6:10]
            )
    return result


def _target_coordinates(
    target: np.ndarray,
    centers: np.ndarray,
    group_name: str,
    state_slice: slice,
) -> np.ndarray:
    if group_name != "attitude":
        return target[:, state_slice] - centers[:, state_slice]
    result = np.full((len(target), 3), np.nan, dtype=np.float64)
    valid = np.all(np.isfinite(target[:, 6:10]), axis=1) & np.all(
        np.isfinite(centers[:, 6:10]), axis=1
    )
    if np.any(valid):
        result[valid] = attitude_innovation(
            centers[valid, 6:10], target[valid, 6:10]
        )
    return result


def _finite_mean(values: Sequence[float | None] | np.ndarray) -> float | None:
    finite = np.asarray(
        [value for value in values if value is not None], dtype=np.float64
    )
    finite = finite[np.isfinite(finite)]
    return float(np.mean(finite)) if len(finite) else None


def _finite_rms(values: Sequence[float | None] | np.ndarray) -> float | None:
    finite = np.asarray(
        [value for value in values if value is not None], dtype=np.float64
    )
    finite = finite[np.isfinite(finite)]
    return float(np.sqrt(np.mean(np.square(finite)))) if len(finite) else None


def _finite_quantile(
    values: Sequence[float | None] | np.ndarray, quantile: float
) -> float | None:
    finite = np.asarray(
        [value for value in values if value is not None], dtype=np.float64
    )
    finite = finite[np.isfinite(finite)]
    return float(np.quantile(finite, quantile)) if len(finite) else None


def _column_rms(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    counts = np.sum(finite, axis=0)
    result = np.full(values.shape[1], np.nan, dtype=np.float64)
    valid = counts > 0
    result[valid] = np.sqrt(
        np.sum(np.where(finite, np.square(values), 0.0), axis=0)[valid]
        / counts[valid]
    )
    return result


def _column_quantile(values: np.ndarray, quantile: float) -> np.ndarray:
    result = np.full(values.shape[1], np.nan, dtype=np.float64)
    for column_index in range(values.shape[1]):
        finite = values[:, column_index]
        finite = finite[np.isfinite(finite)]
        if len(finite):
            result[column_index] = np.quantile(
                finite, quantile, method="higher"
            )
    return result


def _energy_score(
    member_coordinates: np.ndarray, target: np.ndarray
) -> float | None:
    scores = []
    for sample_index in range(member_coordinates.shape[1]):
        members = member_coordinates[:, sample_index]
        valid = np.all(np.isfinite(members), axis=1)
        if not np.all(np.isfinite(target[sample_index])) or not np.any(valid):
            continue
        members = members[valid]
        first = np.mean(np.linalg.norm(members - target[sample_index], axis=1))
        pairwise = members[:, None, :] - members[None, :, :]
        second = 0.5 * np.mean(np.linalg.norm(pairwise, axis=2))
        scores.append(first - second)
    return float(np.mean(scores)) if scores else None


def _predictive_ensemble_result(
    ensemble: PredictiveEnsemble,
    trajectory: Trajectory,
    *,
    horizon_steps: int,
    stride_steps: int | None = None,
    coverage_levels: Sequence[float] = DEFAULT_COVERAGE_LEVELS,
    disagreement_calibration: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate one trajectory and retain private scale-fitting samples."""

    levels = tuple(dict.fromkeys(float(level) for level in coverage_levels))
    if not levels or any(level <= 0.0 or level >= 1.0 for level in levels):
        raise ValueError("coverage levels must be strictly between zero and one")

    member_predictions = []
    target = None
    dt_s = None
    for member in ensemble.members:
        predicted, member_target, member_dt_s = windowed_rollout_predictions(
            member,
            trajectory,
            horizon_steps=horizon_steps,
            stride_steps=stride_steps,
        )
        if target is None:
            target = member_target
            dt_s = member_dt_s
        elif predicted.shape != member_predictions[0].shape:
            raise ValueError("ensemble members produced incompatible rollout shapes")
        member_predictions.append(predicted)
    assert target is not None and dt_s is not None

    # The measured initial state is common to every member and contains no
    # predictive information, so exclude it from spread and calibration.
    prediction_paths = np.stack(member_predictions)[:, :, 1:, :]
    _, rollout_count, horizon_count, _ = prediction_paths.shape
    predictions = prediction_paths[:, :, -1, :]
    target_flat = target[:, -1, :]
    path_quaternion_norm = np.linalg.norm(prediction_paths[..., 6:10], axis=3)
    path_finite = np.all(np.isfinite(prediction_paths), axis=3) & (
        path_quaternion_norm > 0.0
    )
    quaternion_norm = np.linalg.norm(predictions[:, :, 6:10], axis=2)
    finite = np.all(np.isfinite(predictions), axis=2) & (quaternion_norm > 0.0)
    centers = _ensemble_centers(predictions, finite)
    finite_counts = np.sum(finite, axis=0)
    unique_prediction_member_count = len(
        {
            _prediction_member_digest(member, member_finite)
            for member, member_finite in zip(predictions, finite)
        }
    )

    groups: dict[str, Any] = {}
    calibration_samples: dict[str, Any] = {}
    for group_name, unit, state_slice in PREDICTIVE_GROUPS:
        member_coordinates = _group_coordinates(
            predictions, centers, group_name, state_slice
        )
        # A member is either a valid state prediction or absent. Reusing a
        # finite position from a state rejected for a NaN in another component
        # would make the center and disagreement population inconsistent.
        member_coordinates[~finite] = np.nan
        target_coordinates = _target_coordinates(
            target_flat, centers, group_name, state_slice
        )
        member_radius = np.linalg.norm(member_coordinates, axis=2)
        target_error = np.linalg.norm(target_coordinates, axis=1)
        spread = _column_rms(member_radius)
        valid_target = np.isfinite(target_error) & np.any(
            np.isfinite(member_radius), axis=0
        )
        calibration = {}
        scaled_calibration = {}
        group_samples = {}
        for level in levels:
            radius = _column_quantile(member_radius, level)
            valid = valid_target & np.isfinite(radius)
            attained_mass = np.full(len(radius), np.nan, dtype=np.float64)
            positive = valid & (finite_counts > 0)
            quantile_indices = np.ceil(
                level * (finite_counts[positive] - 1)
            )
            attained_mass[positive] = (
                quantile_indices + 1
            ) / finite_counts[positive]
            empirical_coverage = (
                float(np.mean(target_error[valid] <= radius[valid]))
                if np.any(valid)
                else None
            )
            calibration[f"{level:g}"] = {
                "nominal_coverage": level,
                "empirical_coverage": empirical_coverage,
                "coverage_error": (
                    None
                    if empirical_coverage is None
                    else empirical_coverage - level
                ),
                "mean_radius": _finite_mean(radius[valid]),
                "p90_radius": _finite_quantile(radius[valid], 0.9),
                "mean_attained_finite_member_mass": _finite_mean(
                    attained_mass[valid]
                ),
                "minimum_attained_finite_member_mass": _finite_quantile(
                    attained_mass[valid], 0.0
                ),
                "maximum_attained_finite_member_mass": _finite_quantile(
                    attained_mass[valid], 1.0
                ),
            }
            group_samples[f"{level:g}"] = {
                "target_error": target_error,
                "member_radius": radius,
                "valid": valid,
            }
            if disagreement_calibration is not None:
                setting = disagreement_calibration["groups"][group_name][
                    f"{level:g}"
                ]
                scale = setting["scale"]
                if scale is None:
                    scaled_calibration[f"{level:g}"] = {
                        "nominal_coverage": level,
                        "status": "unavailable",
                        "reason": setting["reason"],
                        "scale": None,
                        "empirical_coverage": None,
                        "coverage_error": None,
                        "mean_radius": None,
                        "p90_radius": None,
                    }
                else:
                    scaled_radius = radius * float(scale)
                    scaled_valid = valid & np.isfinite(scaled_radius)
                    empirical_coverage = (
                        float(
                            np.mean(
                                target_error[scaled_valid]
                                <= scaled_radius[scaled_valid]
                            )
                        )
                        if np.any(scaled_valid)
                        else None
                    )
                    scaled_calibration[f"{level:g}"] = {
                        "nominal_coverage": level,
                        "status": "available",
                        "reason": None,
                        "scale": float(scale),
                        "empirical_coverage": empirical_coverage,
                        "coverage_error": (
                            None
                            if empirical_coverage is None
                            else empirical_coverage - level
                        ),
                        "mean_radius": _finite_mean(
                            scaled_radius[scaled_valid]
                        ),
                        "p90_radius": _finite_quantile(
                            scaled_radius[scaled_valid], 0.9
                        ),
                    }
        groups[group_name] = {
            "unit": unit,
            "center_vector_rmse": _finite_rms(target_error),
            "mean_disagreement_radius": _finite_mean(spread),
            "p90_disagreement_radius": _finite_quantile(spread, 0.9),
            "error_disagreement_spearman": _spearman_correlation(
                target_error, spread
            ),
            "energy_score": _energy_score(member_coordinates, target_coordinates),
            "calibration": calibration,
        }
        if disagreement_calibration is not None:
            groups[group_name]["scaled_calibration"] = scaled_calibration
        calibration_samples[group_name] = group_samples

    report = {
        "policy": "group_bootstrap_predictive_disagreement_v4",
        "uncertainty_semantics": {
            "kind": "empirical_epistemic_sensitivity",
            "posterior": False,
            "calibrated_distribution": False,
            "interval_claim": False,
            "coverage_role": (
                "independently_scaled_group_diagnostic"
                if disagreement_calibration is not None
                else "coarse_disagreement_diagnostic"
            ),
            "includes_parameter_resampling": True,
            "includes_process_noise": False,
            "includes_observation_noise": False,
            "includes_unfitted_model_form": False,
        },
        "member_count": ensemble.member_count,
        "unique_parameter_member_count": ensemble.unique_member_count,
        "unique_prediction_member_count": unique_prediction_member_count,
        "rollout_count": rollout_count,
        "prediction_count": rollout_count,
        "path_prediction_count": rollout_count * horizon_count,
        "horizon_steps": horizon_steps,
        "horizon_s": horizon_steps * dt_s,
        "prediction_target": "rollout_endpoint",
        "initial_measured_state_excluded": True,
        "finite_member_prediction_fraction": float(np.mean(finite)),
        "finite_member_path_fraction": float(np.mean(path_finite)),
        "fully_finite_member_fraction": float(
            np.mean(np.all(path_finite, axis=(1, 2)))
        ),
        "minimum_finite_members_per_prediction": int(np.min(finite_counts)),
        "median_finite_members_per_prediction": float(
            np.median(finite_counts)
        ),
        "maximum_finite_members_per_prediction": int(np.max(finite_counts)),
        "groups": groups,
    }
    if disagreement_calibration is not None:
        report["disagreement_calibration"] = {
            "method": disagreement_calibration["method"],
            "calibration_source_group_count": disagreement_calibration[
                "calibration_source_group_count"
            ],
            "independent_from_evaluation": True,
        }
    return report, calibration_samples


def predictive_ensemble_metrics(
    ensemble: PredictiveEnsemble,
    trajectory: Trajectory,
    *,
    horizon_steps: int,
    stride_steps: int | None = None,
    coverage_levels: Sequence[float] = DEFAULT_COVERAGE_LEVELS,
    disagreement_calibration: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate ensemble disagreement on fixed-horizon rollouts."""

    report, _ = _predictive_ensemble_result(
        ensemble,
        trajectory,
        horizon_steps=horizon_steps,
        stride_steps=stride_steps,
        coverage_levels=coverage_levels,
        disagreement_calibration=disagreement_calibration,
    )
    return report


def fit_grouped_disagreement_calibration(
    ensemble: PredictiveEnsemble,
    trajectories: Sequence[Trajectory],
    *,
    horizon_steps: int,
    coverage_levels: Sequence[float] = DEFAULT_COVERAGE_LEVELS,
) -> dict[str, Any]:
    """Fit source-group split scales on data excluded from every member fit.

    Each source group first contributes the multiplier required to attain the
    requested endpoint coverage within that complete group. The selected scale
    is then the finite-sample corrected quantile across independent groups. A
    level is unavailable when the calibration partition has too few groups.
    """

    if not trajectories:
        raise ValueError("calibration requires at least one trajectory")
    levels = tuple(dict.fromkeys(float(level) for level in coverage_levels))
    if not levels or any(level <= 0.0 or level >= 1.0 for level in levels):
        raise ValueError("coverage levels must be strictly between zero and one")
    source_groups = [trajectory.labels.get("source_group") for trajectory in trajectories]
    if any(
        not isinstance(group, (str, int))
        or (isinstance(group, str) and not group.strip())
        for group in source_groups
    ):
        raise ValueError(
            "calibration requires a non-empty source_group on every trajectory"
        )
    ordered_groups = tuple(dict.fromkeys(source_groups))
    if len({str(group) for group in ordered_groups}) != len(ordered_groups):
        raise ValueError(
            "calibration source_group labels need unique string representations"
        )

    samples_by_group: dict[str | int, dict[str, dict[str, list[np.ndarray]]]] = {
        group: {
            group_name: {f"{level:g}": [] for level in levels}
            for group_name, _, _ in PREDICTIVE_GROUPS
        }
        for group in ordered_groups
    }
    endpoint_counts = {group: 0 for group in ordered_groups}
    for trajectory, source_group in zip(trajectories, source_groups):
        assert isinstance(source_group, (str, int))
        _, samples = _predictive_ensemble_result(
            ensemble,
            trajectory,
            horizon_steps=horizon_steps,
            coverage_levels=levels,
        )
        for group_name, _, _ in PREDICTIVE_GROUPS:
            for level in levels:
                item = samples[group_name][f"{level:g}"]
                valid = item["valid"]
                target_error = item["target_error"][valid]
                member_radius = item["member_radius"][valid]
                ratio = np.full(len(target_error), np.inf, dtype=np.float64)
                positive = member_radius > 0.0
                ratio[positive] = target_error[positive] / member_radius[positive]
                ratio[(member_radius == 0.0) & (target_error == 0.0)] = 0.0
                samples_by_group[source_group][group_name][f"{level:g}"].append(
                    ratio
                )
                if group_name == PREDICTIVE_GROUPS[0][0] and level == levels[0]:
                    endpoint_counts[source_group] += len(ratio)

    group_count = len(ordered_groups)
    calibrated_groups: dict[str, Any] = {}
    for group_name, unit, _ in PREDICTIVE_GROUPS:
        calibrated_levels = {}
        for level in levels:
            label = f"{level:g}"
            group_scores: dict[str, float | None] = {}
            sortable_scores = []
            for source_group in ordered_groups:
                parts = samples_by_group[source_group][group_name][label]
                ratios = np.concatenate(parts) if parts else np.empty(0)
                score = (
                    float(np.quantile(ratios, level, method="higher"))
                    if len(ratios)
                    else math.inf
                )
                group_scores[str(source_group)] = (
                    score if np.isfinite(score) else None
                )
                sortable_scores.append(score)
            conformal_rank = math.ceil((group_count + 1) * level)
            minimum_group_count = math.ceil(
                level / (1.0 - level) - 1e-12
            )
            if conformal_rank > group_count:
                scale = None
                reason = "insufficient_independent_calibration_groups"
            else:
                selected = sorted(sortable_scores)[conformal_rank - 1]
                if np.isfinite(selected):
                    scale = float(selected)
                    reason = None
                else:
                    scale = None
                    reason = "nonfinite_required_scale"
            calibrated_levels[label] = {
                "nominal_coverage": level,
                "scale": scale,
                "status": "available" if scale is not None else "unavailable",
                "reason": reason,
                "conformal_group_rank": conformal_rank,
                "minimum_calibration_group_count": minimum_group_count,
                "calibration_source_group_count": group_count,
                "source_group_required_scales": group_scores,
            }
        calibrated_groups[group_name] = {
            "unit": unit,
            **calibrated_levels,
        }

    return {
        "format_version": PREDICTIVE_ENSEMBLE_FORMAT_VERSION,
        "method": DISAGREEMENT_CALIBRATION_METHOD,
        "independence_unit": "source_group",
        "endpoint_sampling": "nonoverlapping_fixed_horizon_rollouts",
        "group_score": "higher_quantile_required_member_radius_multiplier",
        "finite_sample_correction": "ceil((group_count + 1) * coverage)",
        "calibration_source_groups": [str(group) for group in ordered_groups],
        "calibration_source_group_count": group_count,
        "endpoint_count_by_source_group": {
            str(group): endpoint_counts[group] for group in ordered_groups
        },
        "groups": calibrated_groups,
        "uncertainty_semantics": {
            "posterior": False,
            "calibrated_distribution": False,
            "interval_claim": False,
            "coverage_role": "independently_scaled_group_diagnostic",
        },
    }


def aggregate_predictive_ensemble_metrics(
    reports: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return an equal-item macro summary of compatible ensemble reports."""

    if not reports:
        raise ValueError("at least one predictive ensemble report is required")
    reference_groups = tuple(reports[0]["groups"])
    if any(tuple(report["groups"]) != reference_groups for report in reports):
        raise ValueError("predictive ensemble reports have incompatible groups")
    levels = tuple(reports[0]["groups"][reference_groups[0]]["calibration"])
    unique_parameter_counts = [
        float(
            report["unique_parameter_member_count"]
            if "unique_parameter_member_count" in report
            else report["mean_unique_parameter_member_count"]
        )
        for report in reports
    ]
    minimum_unique_parameter_counts = [
        int(
            report["unique_parameter_member_count"]
            if "unique_parameter_member_count" in report
            else report["minimum_unique_parameter_member_count"]
        )
        for report in reports
    ]
    unique_prediction_counts = [
        float(
            report["unique_prediction_member_count"]
            if "unique_prediction_member_count" in report
            else report["mean_unique_prediction_member_count"]
        )
        for report in reports
    ]
    minimum_unique_prediction_counts = [
        int(
            report["unique_prediction_member_count"]
            if "unique_prediction_member_count" in report
            else report["minimum_unique_prediction_member_count"]
        )
        for report in reports
    ]
    minimum_finite_counts = [
        int(report["minimum_finite_members_per_prediction"])
        if "minimum_finite_members_per_prediction" in report
        else int(report["minimum_finite_members_per_prediction_aggregate"])
        for report in reports
    ]
    median_finite_counts = [
        float(report["median_finite_members_per_prediction"])
        if "median_finite_members_per_prediction" in report
        else float(report["mean_median_finite_members_per_prediction"])
        for report in reports
    ]
    maximum_finite_counts = [
        int(report["maximum_finite_members_per_prediction"])
        if "maximum_finite_members_per_prediction" in report
        else int(report["maximum_finite_members_per_prediction_aggregate"])
        for report in reports
    ]
    groups = {}
    for group_name in reference_groups:
        items = [report["groups"][group_name] for report in reports]
        correlations = [
            float(item["error_disagreement_spearman"])
            for item in items
            if item["error_disagreement_spearman"] is not None
        ]
        group_report = {
            "unit": items[0]["unit"],
            "center_vector_rmse": _finite_rms(
                [item["center_vector_rmse"] for item in items]
            ),
            "mean_disagreement_radius": _finite_mean(
                [item["mean_disagreement_radius"] for item in items]
            ),
            "p90_disagreement_radius": _finite_mean(
                [item["p90_disagreement_radius"] for item in items]
            ),
            "error_disagreement_spearman": (
                float(np.mean(correlations)) if correlations else None
            ),
            "energy_score": _finite_mean(
                [item["energy_score"] for item in items]
            ),
            "calibration": {
                level: {
                    "nominal_coverage": float(
                        items[0]["calibration"][level]["nominal_coverage"]
                    ),
                    "empirical_coverage": _finite_mean(
                        [
                            item["calibration"][level]["empirical_coverage"]
                            for item in items
                        ]
                    ),
                    "coverage_error": _finite_mean(
                        [
                            item["calibration"][level]["coverage_error"]
                            for item in items
                        ]
                    ),
                    "mean_radius": _finite_mean(
                        [
                            item["calibration"][level]["mean_radius"]
                            for item in items
                        ]
                    ),
                    "p90_radius": _finite_mean(
                        [
                            item["calibration"][level]["p90_radius"]
                            for item in items
                        ]
                    ),
                    "mean_attained_finite_member_mass": _finite_mean(
                        [
                            item["calibration"][level][
                                "mean_attained_finite_member_mass"
                            ]
                            for item in items
                        ]
                    ),
                    "minimum_attained_finite_member_mass": _finite_quantile(
                        [
                            item["calibration"][level][
                                "minimum_attained_finite_member_mass"
                            ]
                            for item in items
                        ],
                        0.0,
                    ),
                    "maximum_attained_finite_member_mass": _finite_quantile(
                        [
                            item["calibration"][level][
                                "maximum_attained_finite_member_mass"
                            ]
                            for item in items
                        ],
                        1.0,
                    ),
                }
                for level in levels
            },
        }
        if all("scaled_calibration" in item for item in items):
            scaled = {}
            for level in levels:
                level_items = [item["scaled_calibration"][level] for item in items]
                available = [
                    item for item in level_items if item["status"] == "available"
                ]
                available_fraction = len(available) / len(level_items)
                scaled[level] = {
                    "nominal_coverage": float(
                        level_items[0]["nominal_coverage"]
                    ),
                    "status": (
                        "available"
                        if available_fraction == 1.0
                        else (
                            "unavailable"
                            if available_fraction == 0.0
                            else "partially_available"
                        )
                    ),
                    "available_item_fraction": available_fraction,
                    "reasons": sorted(
                        {
                            reason
                            for item in level_items
                            for reason in (
                                [str(item["reason"])]
                                if item.get("reason") is not None
                                else [
                                    str(value)
                                    for value in item.get("reasons", [])
                                ]
                            )
                        }
                    ),
                    "mean_scale": _finite_mean(
                        [
                            item.get("scale", item.get("mean_scale"))
                            for item in available
                        ]
                    ),
                    "empirical_coverage": _finite_mean(
                        [item["empirical_coverage"] for item in available]
                    ),
                    "coverage_error": _finite_mean(
                        [item["coverage_error"] for item in available]
                    ),
                    "mean_radius": _finite_mean(
                        [item["mean_radius"] for item in available]
                    ),
                    "p90_radius": _finite_mean(
                        [item["p90_radius"] for item in available]
                    ),
                }
            group_report["scaled_calibration"] = scaled
        groups[group_name] = group_report
    return {
        "weighting": "equal_item",
        "item_count": len(reports),
        "mean_unique_parameter_member_count": float(
            np.mean(unique_parameter_counts)
        ),
        "minimum_unique_parameter_member_count": int(
            min(minimum_unique_parameter_counts)
        ),
        "mean_unique_prediction_member_count": float(
            np.mean(unique_prediction_counts)
        ),
        "minimum_unique_prediction_member_count": int(
            min(minimum_unique_prediction_counts)
        ),
        "minimum_finite_members_per_prediction_aggregate": int(
            min(minimum_finite_counts)
        ),
        "mean_median_finite_members_per_prediction": float(
            np.mean(median_finite_counts)
        ),
        "maximum_finite_members_per_prediction_aggregate": int(
            max(maximum_finite_counts)
        ),
        "finite_member_prediction_fraction": float(
            np.mean(
                [float(report["finite_member_prediction_fraction"]) for report in reports]
            )
        ),
        "finite_member_path_fraction": float(
            np.mean(
                [float(report["finite_member_path_fraction"]) for report in reports]
            )
        ),
        "fully_finite_member_fraction": float(
            np.mean([float(report["fully_finite_member_fraction"]) for report in reports])
        ),
        "groups": groups,
    }


def _safe_name(value: str | int) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value)).strip("_")
    return slug or "fold"


def _file_record(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _read_json_mapping(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _artifact_record_matches(record: Mapping[str, Any]) -> bool:
    try:
        path = Path(str(record["path"]))
        if not path.is_file():
            return False
        current = _file_record(path)
        return (
            current["size_bytes"] == record["size_bytes"]
            and current["sha256"] == record["sha256"]
        )
    except (KeyError, OSError, TypeError, ValueError):
        return False


def _summary_artifacts_valid(summary: Mapping[str, Any]) -> bool:
    try:
        manifest_paths = [
            Path(str(fold["ensemble"]))
            for fold in summary["per_fold"].values()
        ]
        for manifest_path in manifest_paths:
            if not manifest_path.is_file():
                return False
            manifest = json.loads(manifest_path.read_text())
            if (
                manifest.get("format_version")
                != PREDICTIVE_ENSEMBLE_FORMAT_VERSION
                or manifest.get("method") != PREDICTIVE_ENSEMBLE_METHOD
            ):
                return False
            if not _artifact_record_matches(
                manifest["disagreement_calibration_artifact"]
            ):
                return False
            for member in manifest["members"]:
                if not _artifact_record_matches(member["model_artifact"]):
                    return False
                if not _artifact_record_matches(
                    member["fit_report_artifact"]
                ):
                    return False
    except (AttributeError, KeyError, OSError, TypeError, ValueError):
        return False
    return True


def _request_digest(request: Mapping[str, Any]) -> str:
    serialized = json.dumps(
        request, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(serialized.encode()).hexdigest()


def _automatic_outer_axis(trajectories: Sequence[Trajectory]) -> str:
    profiles = [trajectory.labels.get("profile") for trajectory in trajectories]
    if all(profile is not None for profile in profiles) and len(set(profiles)) >= 2:
        return "profile"
    return "source_group"


def _automatic_member_count(training_group_count: int) -> int:
    return min(DEFAULT_ENSEMBLE_MEMBER_COUNT, max(4, training_group_count))


def _balanced_calibration_groups(
    candidate_groups: Sequence[str | int],
    *,
    profiles: Mapping[str | int, str | int] | None,
    conditions: Mapping[str | int, str | int] | None,
    fold_index: int,
) -> tuple[str | int, ...]:
    """Reserve deterministic groups while preserving every usable stratum.

    When typed profile/condition strata contain replicate groups, half of each
    stratum calibrates and half fits. Otherwise the same split is performed per
    profile, or globally when profiles are absent. No stratum contributes all
    of its groups to calibration.
    """

    ordered = tuple(sorted(candidate_groups, key=str))
    if len(ordered) < 3:
        raise ValueError(
            "independent calibration requires at least three candidate groups"
        )
    if profiles is not None and set(profiles) != set(ordered):
        raise ValueError("profiles must contain exactly the candidate groups")
    if conditions is not None and set(conditions) != set(ordered):
        raise ValueError("conditions must contain exactly the candidate groups")

    if profiles is None:
        strata = (ordered,)
    else:
        profile_values = tuple(dict.fromkeys(profiles[group] for group in ordered))
        profile_strata = tuple(
            tuple(group for group in ordered if profiles[group] == profile)
            for profile in profile_values
        )
        condition_strata = []
        condition_balanced = conditions is not None
        if condition_balanced:
            for profile_groups in profile_strata:
                condition_values = tuple(
                    dict.fromkeys(conditions[group] for group in profile_groups)
                )
                subdivisions = tuple(
                    tuple(
                        group
                        for group in profile_groups
                        if conditions[group] == condition
                    )
                    for condition in condition_values
                )
                if any(len(subdivision) < 2 for subdivision in subdivisions):
                    condition_balanced = False
                    break
                condition_strata.extend(subdivisions)
        strata = tuple(condition_strata) if condition_balanced else profile_strata

    selected = []
    for stratum_index, stratum in enumerate(strata):
        if len(stratum) < 2:
            continue
        calibration_count = max(1, len(stratum) // 2)
        offset = (fold_index + stratum_index) % len(stratum)
        rotated = stratum[offset:] + stratum[:offset]
        selected.extend(rotated[:calibration_count])
    if not selected:
        selected.append(ordered[fold_index % len(ordered)])
    selected_set = set(selected)
    if len(ordered) - len(selected_set) < 2:
        raise ValueError(
            "calibration partition leaves fewer than two member-fitting groups"
        )
    return tuple(group for group in ordered if group in selected_set)


def _group_objective_weights(
    groups: Sequence[str | int],
    *,
    multiplicities: Mapping[str | int, int] | None,
    strata: Mapping[str | int, str | int] | None,
) -> dict[str | int, float]:
    """Give profiles equal objective mass and groups bootstrap multiplicity."""

    if strata is None:
        return {
            group: float(
                1 if multiplicities is None else multiplicities.get(group, 0)
            )
            for group in groups
        }
    stratum_counts = {
        stratum: sum(strata[group] == stratum for group in groups)
        for stratum in dict.fromkeys(strata[group] for group in groups)
    }
    return {
        group: float(
            1 if multiplicities is None else multiplicities.get(group, 0)
        )
        / stratum_counts[strata[group]]
        for group in groups
    }


def _source_tree_digest() -> str:
    root = Path(__file__).resolve().parents[2]
    source_root = root / "src" / "glassbox"
    if source_root.is_dir():
        candidates = [
            *sorted(source_root.rglob("*.py")),
            root / "pyproject.toml",
            root / "uv.lock",
        ]
        relative_root = root
    else:
        source_root = Path(__file__).resolve().parent
        candidates = sorted(source_root.rglob("*.py"))
        relative_root = source_root
    digest = hashlib.sha256()
    for path in candidates:
        if not path.is_file():
            continue
        digest.update(str(path.relative_to(relative_root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _git_revision() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "tracked_worktree_dirty": None}
    return {"commit": commit, "tracked_worktree_dirty": dirty}


def _implementation_fingerprint() -> dict[str, Any]:
    try:
        glassbox_version = importlib.metadata.version("glassbox")
    except importlib.metadata.PackageNotFoundError:
        glassbox_version = None
    return {
        "ensemble_format_version": PREDICTIVE_ENSEMBLE_FORMAT_VERSION,
        "ensemble_method": PREDICTIVE_ENSEMBLE_METHOD,
        "glassbox_version": glassbox_version,
        "source_tree_sha256": _source_tree_digest(),
        "git": _git_revision(),
        "python": sys.version,
        "jax": jax.__version__,
        "numpy": np.__version__,
        "jax_backend": jax.default_backend(),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def benchmark_predictive_ensemble(
    trajectory_paths: Sequence[str | Path],
    output_dir: str | Path,
    *,
    training_horizons_s: tuple[float, ...] = (0.1, 0.5, 2.0),
    evaluation_horizons_s: tuple[float, ...] = (0.1, 0.5, 1.0, 2.0),
    steps: int = 400,
    learning_rate: float = 0.02,
    model_class: str = "structured_residual",
    member_count: int | None = None,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    """Run nested fit, calibration, and outer-evaluation partitions."""

    paths = [Path(path).resolve() for path in trajectory_paths]
    if len(paths) < 4:
        raise ValueError("predictive ensemble benchmark requires four trajectories")
    trajectories = [load_trajectory_npz(path) for path in paths]
    reference_spec = trajectories[0].spec
    if any(trajectory.spec != reference_spec for trajectory in trajectories[1:]):
        raise ValueError("predictive ensemble benchmark requires one trajectory spec")
    group_by_trajectory = [
        trajectory.labels.get("source_group") for trajectory in trajectories
    ]
    if any(group is None for group in group_by_trajectory):
        raise ValueError(
            "predictive ensemble benchmark requires source_group on every trajectory"
        )
    if any(
        not isinstance(group, (str, int))
        or (isinstance(group, str) and not group.strip())
        for group in group_by_trajectory
    ):
        raise ValueError(
            "source_group labels must be non-empty strings or integers"
        )
    groups = tuple(dict.fromkeys(group_by_trajectory))
    if len(groups) < 4:
        raise ValueError(
            "predictive ensemble benchmark requires four independent source groups"
        )
    if len({str(group) for group in groups}) != len(groups):
        raise ValueError(
            "source_group labels must have unique string representations"
        )
    profile_by_trajectory = [
        trajectory.labels.get("profile") for trajectory in trajectories
    ]
    if any(
        profile is not None
        and (
            not isinstance(profile, (str, int))
            or (isinstance(profile, str) and not profile.strip())
        )
        for profile in profile_by_trajectory
    ):
        raise ValueError("profile labels must be non-empty strings or integers")
    group_profiles: dict[str | int, str | int] = {}
    condition_by_trajectory = [
        trajectory.labels.get("condition") for trajectory in trajectories
    ]
    group_conditions: dict[str | int, str | int] = {}
    for group, profile, condition in zip(
        group_by_trajectory,
        profile_by_trajectory,
        condition_by_trajectory,
    ):
        assert isinstance(group, (str, int))
        if profile is None:
            pass
        elif not isinstance(profile, (str, int)):
            raise ValueError("profile labels must be strings or integers")
        else:
            if group in group_profiles and group_profiles[group] != profile:
                raise ValueError("one source_group cannot span multiple profiles")
            group_profiles[group] = profile
        if condition is None:
            continue
        if not isinstance(condition, (str, int)) or (
            isinstance(condition, str) and not condition.strip()
        ):
            raise ValueError("condition labels must be non-empty strings or integers")
        if group in group_conditions and group_conditions[group] != condition:
            raise ValueError("one source_group cannot span multiple conditions")
        group_conditions[group] = condition

    outer_axis = _automatic_outer_axis(trajectories)
    outer_values = (
        tuple(dict.fromkeys(profile_by_trajectory))
        if outer_axis == "profile"
        else groups
    )
    if len({str(value) for value in outer_values}) != len(outer_values):
        raise ValueError("outer fold labels must have unique string representations")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    request = {
        "format_version": PREDICTIVE_ENSEMBLE_FORMAT_VERSION,
        "evaluation": "nested_group_calibrated_predictive_ensemble",
        "implementation": _implementation_fingerprint(),
        "files": [_file_record(path) for path in paths],
        "outer_axis": outer_axis,
        "configuration": {
            "training_horizons_s": list(training_horizons_s),
            "evaluation_horizons_s": list(evaluation_horizons_s),
            "optimization_steps_per_member": steps,
            "learning_rate": learning_rate,
            "model_class": model_class,
            "member_count_policy": (
                "automatic_min_4_max_8_by_training_group_count"
                if member_count is None
                else "explicit_test_or_research_override"
            ),
            "member_count_override": member_count,
            "bootstrap_seed": bootstrap_seed,
            "bootstrap_stratification": (
                "profile" if group_profiles else "none"
            ),
            "bootstrap_estimator": (
                "shared_fit_partition_statistics_with_profile_balanced_"
                "group_multiplicity_loss_v2"
            ),
            "calibration_partition": (
                "balanced_complete_source_groups_within_profile_and_condition_"
                "strata_when_replicates_exist"
            ),
            "calibration_method": DISAGREEMENT_CALIBRATION_METHOD,
            "coverage_levels": list(DEFAULT_COVERAGE_LEVELS),
        },
    }
    request_path = destination / "request.json"
    summary_path = destination / "summary.json"
    if request_path.exists() and summary_path.exists():
        recorded_request = _read_json_mapping(request_path)
        recorded_summary = _read_json_mapping(summary_path)
        if (
            recorded_request == request
            and recorded_summary is not None
            and _summary_artifacts_valid(recorded_summary)
        ):
            return recorded_summary
    request_path.write_text(
        json.dumps(request, indent=2, allow_nan=False) + "\n"
    )
    dataset_digest = _request_digest(request)

    folds: dict[str, Any] = {}
    fold_aggregate_reports = []
    for fold_index, outer_value in enumerate(outer_values, start=1):
        validation_indices = [
            index
            for index, (group, profile) in enumerate(
                zip(group_by_trajectory, profile_by_trajectory)
            )
            if (profile if outer_axis == "profile" else group) == outer_value
        ]
        validation_groups = tuple(
            dict.fromkeys(group_by_trajectory[index] for index in validation_indices)
        )
        calibration_axis = "source_group"
        calibration_value = "balanced_within_outer_training_strata"
        calibration_candidates = tuple(
            group for group in groups if group not in validation_groups
        )
        candidate_profiles = (
            {group: group_profiles[group] for group in calibration_candidates}
            if all(group in group_profiles for group in calibration_candidates)
            else None
        )
        candidate_conditions = (
            {group: group_conditions[group] for group in calibration_candidates}
            if all(group in group_conditions for group in calibration_candidates)
            else None
        )
        calibration_groups = _balanced_calibration_groups(
            calibration_candidates,
            profiles=candidate_profiles,
            conditions=candidate_conditions,
            fold_index=fold_index,
        )
        calibration_indices = [
            index
            for index, group in enumerate(group_by_trajectory)
            if group in calibration_groups
        ]
        training_groups = tuple(
            group
            for group in groups
            if group not in validation_groups and group not in calibration_groups
        )
        if len(training_groups) < 2:
            raise ValueError(
                f"outer {outer_axis} fold {outer_value!r} leaves fewer than two "
                "member-fitting source groups after calibration is reserved"
            )
        selected_member_count = (
            _automatic_member_count(len(training_groups))
            if member_count is None
            else member_count
        )
        strata = (
            {group: group_profiles[group] for group in training_groups}
            if group_profiles and all(group in group_profiles for group in training_groups)
            else None
        )
        multiplicities = grouped_bootstrap_multiplicities(
            training_groups,
            strata=strata,
            member_count=selected_member_count,
            seed=bootstrap_seed + fold_index - 1,
        )
        normalization_group_weights = _group_objective_weights(
            training_groups,
            multiplicities=None,
            strata=strata,
        )
        training_indices = [
            index
            for index, group in enumerate(group_by_trajectory)
            if group in training_groups
        ]
        fold_prefix = f"fold_{fold_index:02d}_{_safe_name(outer_value)}"
        fold_dir = destination / fold_prefix
        fold_dir.mkdir(parents=True, exist_ok=True)
        members = []
        member_records = []
        for member_index, counts in enumerate(multiplicities, start=1):
            loss_group_weights = _group_objective_weights(
                training_groups,
                multiplicities=counts,
                strata=strata,
            )
            fit_paths = [
                *(paths[index] for index in training_indices),
                *(paths[index] for index in calibration_indices),
            ]
            member_id = f"member_{member_index:02d}"
            model_path = fold_dir / f"{member_id}_model.json"
            report_path = fold_dir / f"{member_id}_report.json"
            member_request_path = fold_dir / f"{member_id}_request.json"
            member_request = {
                "format_version": PREDICTIVE_ENSEMBLE_FORMAT_VERSION,
                "dataset_request_sha256": dataset_digest,
                "outer_axis": outer_axis,
                "outer_value": outer_value,
                "calibration_axis": calibration_axis,
                "calibration_value": calibration_value,
                "member_index": member_index,
                "training_source_group_multiplicities": {
                    str(group): count for group, count in counts.items()
                },
                "training_source_group_loss_weights": {
                    str(group): loss_group_weights[group]
                    for group in training_groups
                },
                "normalization_source_group_weights": {
                    str(group): normalization_group_weights[group]
                    for group in training_groups
                },
            }
            member_state = _read_json_mapping(member_request_path)
            reusable = bool(
                member_state is not None
                and member_state.get("request") == member_request
                and _artifact_record_matches(
                    member_state.get("model_artifact", {})
                )
                and _artifact_record_matches(
                    member_state.get("fit_report_artifact", {})
                )
            )
            if reusable:
                params, _ = load_dynamics_model(model_path)
                report = json.loads(report_path.read_text())
            else:
                params, _, report = fit_trajectory_artifacts(
                    fit_paths,
                    holdout_count=len(calibration_groups),
                    holdout_profiles=None,
                    training_horizons_s=training_horizons_s,
                    evaluation_horizons_s=evaluation_horizons_s,
                    steps=steps,
                    learning_rate=learning_rate,
                    run_no_lag_ablation=False,
                    training_source_group_weights=loss_group_weights,
                    normalization_source_group_weights=(
                        normalization_group_weights
                    ),
                    model_class=model_class,
                )
                report_path.write_text(json.dumps(report, indent=2) + "\n")
                save_dynamics_model(
                    params,
                    model_path,
                    input_spec=reference_spec,
                    runtime_spec=runtime_spec_from_fit_report(report),
                    provenance={
                        "ensemble_method": PREDICTIVE_ENSEMBLE_METHOD,
                        "outer_axis": outer_axis,
                        "outer_value": outer_value,
                        "calibration_axis": calibration_axis,
                        "calibration_value": calibration_value,
                        "member_index": member_index,
                        "fit_report": str(report_path),
                        "training_source_group_multiplicities": {
                            str(group): count for group, count in counts.items()
                        },
                        "training_source_group_loss_weights": {
                            str(group): loss_group_weights[group]
                            for group in training_groups
                        },
                    },
                )
            model_artifact = _file_record(model_path)
            fit_report_artifact = _file_record(report_path)
            member_request_path.write_text(
                json.dumps(
                    {
                        "request": member_request,
                        "model_artifact": model_artifact,
                        "fit_report_artifact": fit_report_artifact,
                    },
                    indent=2,
                    allow_nan=False,
                )
                + "\n"
            )
            members.append(params)
            member_records.append(
                {
                    "member_id": member_id,
                    "model": str(model_path),
                    "report": str(report_path),
                    "training_source_group_multiplicities": {
                        str(group): count for group, count in counts.items()
                    },
                    "training_source_group_loss_weights": {
                        str(group): loss_group_weights[group]
                        for group in training_groups
                    },
                    "model_artifact": model_artifact,
                    "fit_report_artifact": fit_report_artifact,
                }
            )

        ensemble = PredictiveEnsemble(
            members=tuple(members),
            member_ids=tuple(record["member_id"] for record in member_records),
        )
        calibration_trajectories = [
            trajectories[index] for index in calibration_indices
        ]
        disagreement_calibrations = {
            f"{seconds:g}s": fit_grouped_disagreement_calibration(
                ensemble,
                calibration_trajectories,
                horizon_steps=duration_to_steps(
                    seconds, calibration_trajectories[0].nominal_dt_s
                ),
            )
            for seconds in evaluation_horizons_s
        }
        calibration_path = fold_dir / "disagreement_calibration.json"
        calibration_path.write_text(
            json.dumps(
                {
                    "format_version": PREDICTIVE_ENSEMBLE_FORMAT_VERSION,
                    "outer_axis": outer_axis,
                    "outer_value": outer_value,
                    "calibration_axis": calibration_axis,
                    "calibration_value": calibration_value,
                    "horizon_rollouts": disagreement_calibrations,
                },
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )
        calibration_artifact = _file_record(calibration_path)
        per_trajectory = {}
        trajectory_aggregate_inputs: dict[str, list[dict[str, Any]]] = {
            f"{seconds:g}s": [] for seconds in evaluation_horizons_s
        }
        for validation_index in validation_indices:
            trajectory = trajectories[validation_index]
            trajectory_report = {}
            for seconds in evaluation_horizons_s:
                label = f"{seconds:g}s"
                metrics = predictive_ensemble_metrics(
                    ensemble,
                    trajectory,
                    horizon_steps=duration_to_steps(
                        seconds, trajectory.nominal_dt_s
                    ),
                    disagreement_calibration=disagreement_calibrations[label],
                )
                trajectory_report[label] = metrics
                trajectory_aggregate_inputs[label].append(metrics)
            per_trajectory[str(paths[validation_index])] = trajectory_report
        aggregate = {
            label: aggregate_predictive_ensemble_metrics(reports)
            for label, reports in trajectory_aggregate_inputs.items()
        }
        fold_aggregate_reports.append(aggregate)
        unique_resample_count = len(
            {
                tuple(counts.get(group, 0) for group in training_groups)
                for counts in multiplicities
            }
        )
        manifest = {
            "format_version": PREDICTIVE_ENSEMBLE_FORMAT_VERSION,
            "artifact_type": "empirical_calibrated_predictive_ensemble",
            "method": PREDICTIVE_ENSEMBLE_METHOD,
            "posterior": False,
            "implementation": request["implementation"],
            "outer_axis": outer_axis,
            "outer_value": outer_value,
            "calibration_axis": calibration_axis,
            "calibration_value": calibration_value,
            "shared_fit_statistics": {
                "policy": "complete_member_fit_partition_v1",
                "normalization_source_group_weights": {
                    str(group): normalization_group_weights[group]
                    for group in training_groups
                },
            },
            "member_count": selected_member_count,
            "unique_resample_count": unique_resample_count,
            "unique_parameter_member_count": ensemble.unique_member_count,
            "disagreement_calibration": str(calibration_path),
            "disagreement_calibration_artifact": calibration_artifact,
            "members": member_records,
        }
        manifest_path = fold_dir / "ensemble.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n"
        )
        folds[str(outer_value)] = {
            "outer_value": outer_value,
            "calibration_axis": calibration_axis,
            "calibration_value": calibration_value,
            "validation_source_groups": list(validation_groups),
            "calibration_source_groups": list(calibration_groups),
            "training_source_groups": list(training_groups),
            "member_count": selected_member_count,
            "unique_resample_count": unique_resample_count,
            "unique_parameter_member_count": ensemble.unique_member_count,
            "ensemble": str(manifest_path),
            "disagreement_calibration": str(calibration_path),
            "aggregate": aggregate,
            "per_trajectory": per_trajectory,
        }

    aggregate = {
        label: aggregate_predictive_ensemble_metrics(
            [fold[label] for fold in fold_aggregate_reports]
        )
        for label in (f"{seconds:g}s" for seconds in evaluation_horizons_s)
    }
    summary = {
        "format_version": PREDICTIVE_ENSEMBLE_FORMAT_VERSION,
        "evaluation": "nested_group_calibrated_predictive_ensemble",
        "implementation": request["implementation"],
        "outer_axis": outer_axis,
        "trajectory_count": len(paths),
        "source_group_count": len(groups),
        "outer_fold_count": len(outer_values),
        "configuration": request["configuration"],
        "uncertainty_semantics": {
            "kind": "empirical_epistemic_sensitivity",
            "posterior": False,
            "calibrated_distribution": False,
            "interval_claim": False,
            "coverage_role": "independently_scaled_group_diagnostic",
            "process_noise_included": False,
            "observation_noise_included": False,
            "unfitted_model_form_included": False,
        },
        "aggregate": {
            "weighting": f"equal_{outer_axis}",
            "horizon_rollouts": aggregate,
        },
        "per_fold": folds,
        "promotion": {
            "status": "diagnostic_only",
            "passed": None,
            "reason": (
                "outer-fold results become development evidence once inspected; "
                "versioned promotion thresholds require a subsequently untouched "
                "corpus, airframe, or configuration"
            ),
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model-class",
        choices=("structured", "structured_residual"),
        default="structured_residual",
    )
    args = parser.parse_args()
    benchmark_predictive_ensemble(
        args.trajectory,
        args.output_dir,
        model_class=args.model_class,
    )


if __name__ == "__main__":
    main()
