"""Offline grouped predictive ensembles and uncertainty diagnostics.

This module deliberately describes an empirical ensemble, not a Bayesian
posterior. Members differ because complete, independent source groups are
resampled. The resulting spread measures sensitivity to the available corpus;
it does not include process noise, observation noise, or model structures that
were not fitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from glassbox.data import Trajectory, duration_to_steps, load_trajectory_npz
from glassbox.dynamics import ModelParams, model_family
from glassbox.evaluation import attitude_innovation, windowed_rollout_predictions
from glassbox.fit_cli import fit_trajectory_artifacts
from glassbox.model_io import load_dynamics_model, save_dynamics_model
from glassbox.runtime import runtime_spec_from_fit_report

DEFAULT_COVERAGE_LEVELS = (0.5, 0.8, 0.9)
DEFAULT_ENSEMBLE_MEMBER_COUNT = 8
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
    method: str = "group_bootstrap_v1"

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
        if self.method != "group_bootstrap_v1":
            raise ValueError(f"unsupported predictive ensemble method: {self.method}")

    @property
    def member_count(self) -> int:
        return len(self.members)


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


def predictive_ensemble_metrics(
    ensemble: PredictiveEnsemble,
    trajectory: Trajectory,
    *,
    horizon_steps: int,
    stride_steps: int | None = None,
    coverage_levels: Sequence[float] = DEFAULT_COVERAGE_LEVELS,
) -> dict[str, Any]:
    """Evaluate uncalibrated ensemble disagreement on fixed-horizon rollouts."""

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

    groups: dict[str, Any] = {}
    for group_name, unit, state_slice in PREDICTIVE_GROUPS:
        member_coordinates = _group_coordinates(
            predictions, centers, group_name, state_slice
        )
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
        for level in levels:
            radius = _column_quantile(member_radius, level)
            valid = valid_target & np.isfinite(radius)
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

    finite_counts = np.sum(finite, axis=0)
    return {
        "policy": "group_bootstrap_predictive_disagreement_v1",
        "uncertainty_semantics": {
            "kind": "empirical_epistemic_sensitivity",
            "posterior": False,
            "calibrated_distribution": False,
            "includes_parameter_resampling": True,
            "includes_process_noise": False,
            "includes_observation_noise": False,
            "includes_unfitted_model_form": False,
        },
        "member_count": ensemble.member_count,
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
        "groups": groups,
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
    groups = {}
    for group_name in reference_groups:
        items = [report["groups"][group_name] for report in reports]
        correlations = [
            float(item["error_disagreement_spearman"])
            for item in items
            if item["error_disagreement_spearman"] is not None
        ]
        groups[group_name] = {
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
                }
                for level in levels
            },
        }
    return {
        "weighting": "equal_item",
        "item_count": len(reports),
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


def benchmark_predictive_ensemble(
    trajectory_paths: Sequence[str | Path],
    output_dir: str | Path,
    *,
    training_horizons_s: tuple[float, ...] = (0.1, 0.5, 2.0),
    evaluation_horizons_s: tuple[float, ...] = (0.1, 0.5, 1.0, 2.0),
    steps: int = 400,
    learning_rate: float = 0.02,
    model_class: str = "structured",
    member_count: int | None = None,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    """Run nested outer holdouts with training-only grouped bootstraps."""

    paths = [Path(path).resolve() for path in trajectory_paths]
    if len(paths) < 3:
        raise ValueError("predictive ensemble benchmark requires three trajectories")
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
    if len(groups) < 3:
        raise ValueError(
            "predictive ensemble benchmark requires three independent source groups"
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
    for group, profile in zip(group_by_trajectory, profile_by_trajectory):
        assert isinstance(group, (str, int))
        if profile is None:
            continue
        if group in group_profiles and group_profiles[group] != profile:
            raise ValueError("one source_group cannot span multiple profiles")
        group_profiles[group] = profile

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
        "format_version": 1,
        "evaluation": "nested_group_bootstrap_predictive_ensemble",
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
            "coverage_levels": list(DEFAULT_COVERAGE_LEVELS),
        },
    }
    request_path = destination / "request.json"
    summary_path = destination / "summary.json"
    if request_path.exists() and summary_path.exists():
        if json.loads(request_path.read_text()) == request:
            return json.loads(summary_path.read_text())
    request_path.write_text(json.dumps(request, indent=2) + "\n")
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
        training_groups = tuple(
            group
            for group in groups
            if all(group_by_trajectory[index] != group for index in validation_indices)
        )
        if len(training_groups) < 2:
            raise ValueError(
                f"outer {outer_axis} fold {outer_value!r} leaves fewer than two "
                "training source groups"
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
        fold_prefix = f"fold_{fold_index:02d}_{_safe_name(outer_value)}"
        fold_dir = destination / fold_prefix
        fold_dir.mkdir(parents=True, exist_ok=True)
        members = []
        member_records = []
        for member_index, counts in enumerate(multiplicities, start=1):
            selected_groups = tuple(group for group in training_groups if group in counts)
            training_indices = [
                index
                for index, group in enumerate(group_by_trajectory)
                if group in selected_groups
            ]
            fit_paths = [
                *(paths[index] for index in training_indices),
                *(paths[index] for index in validation_indices),
            ]
            member_id = f"member_{member_index:02d}"
            model_path = fold_dir / f"{member_id}_model.json"
            report_path = fold_dir / f"{member_id}_report.json"
            member_request_path = fold_dir / f"{member_id}_request.json"
            member_request = {
                "format_version": 1,
                "dataset_request_sha256": dataset_digest,
                "outer_axis": outer_axis,
                "outer_value": outer_value,
                "member_index": member_index,
                "training_source_group_multiplicities": {
                    str(group): count for group, count in counts.items()
                },
            }
            reusable = (
                member_request_path.exists()
                and report_path.exists()
                and model_path.exists()
                and json.loads(member_request_path.read_text()) == member_request
            )
            if reusable:
                params, _ = load_dynamics_model(model_path)
                report = json.loads(report_path.read_text())
            else:
                params, _, report = fit_trajectory_artifacts(
                    fit_paths,
                    holdout_count=1,
                    holdout_profiles=(outer_value,) if outer_axis == "profile" else None,
                    training_horizons_s=training_horizons_s,
                    evaluation_horizons_s=evaluation_horizons_s,
                    steps=steps,
                    learning_rate=learning_rate,
                    run_no_lag_ablation=False,
                    training_source_group_weights=counts,
                    model_class=model_class,
                )
                report_path.write_text(json.dumps(report, indent=2) + "\n")
                save_dynamics_model(
                    params,
                    model_path,
                    input_spec=reference_spec,
                    runtime_spec=runtime_spec_from_fit_report(report),
                    provenance={
                        "ensemble_method": "group_bootstrap_v1",
                        "outer_axis": outer_axis,
                        "outer_value": outer_value,
                        "member_index": member_index,
                        "fit_report": str(report_path),
                        "training_source_group_multiplicities": {
                            str(group): count for group, count in counts.items()
                        },
                    },
                )
                member_request_path.write_text(
                    json.dumps(member_request, indent=2) + "\n"
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
                }
            )

        ensemble = PredictiveEnsemble(
            members=tuple(members),
            member_ids=tuple(record["member_id"] for record in member_records),
        )
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
                )
                trajectory_report[label] = metrics
                trajectory_aggregate_inputs[label].append(metrics)
            per_trajectory[str(paths[validation_index])] = trajectory_report
        aggregate = {
            label: aggregate_predictive_ensemble_metrics(reports)
            for label, reports in trajectory_aggregate_inputs.items()
        }
        fold_aggregate_reports.append(aggregate)
        manifest = {
            "format_version": 1,
            "artifact_type": "empirical_predictive_ensemble",
            "method": "group_bootstrap_v1",
            "posterior": False,
            "outer_axis": outer_axis,
            "outer_value": outer_value,
            "members": member_records,
        }
        manifest_path = fold_dir / "ensemble.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        folds[str(outer_value)] = {
            "outer_value": outer_value,
            "validation_source_groups": list(
                dict.fromkeys(group_by_trajectory[index] for index in validation_indices)
            ),
            "training_source_groups": list(training_groups),
            "member_count": selected_member_count,
            "ensemble": str(manifest_path),
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
        "format_version": 1,
        "evaluation": "nested_group_bootstrap_predictive_ensemble",
        "outer_axis": outer_axis,
        "trajectory_count": len(paths),
        "source_group_count": len(groups),
        "outer_fold_count": len(outer_values),
        "configuration": request["configuration"],
        "uncertainty_semantics": {
            "kind": "empirical_epistemic_sensitivity",
            "posterior": False,
            "calibrated_distribution": False,
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
                "disagreement skill and useful calibration thresholds require "
                "protected empirical results before they can be versioned"
            ),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model-class",
        choices=("structured", "structured_residual"),
        default="structured",
    )
    args = parser.parse_args()
    benchmark_predictive_ensemble(
        args.trajectory,
        args.output_dir,
        model_class=args.model_class,
    )


if __name__ == "__main__":
    main()
