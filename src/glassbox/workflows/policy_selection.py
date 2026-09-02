"""Select a shared fitting policy across platforms and maneuver profiles."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from glassbox.core.data import load_trajectory_npz
from glassbox.core.evaluation import METRIC_FLOORS, ROLLOUT_METRICS
from glassbox.workflows.profile_benchmark import benchmark_profiles
from glassbox.workflows.selection import (
    MAXIMUM_METRIC_REGRESSION,
    MAXIMUM_PLATFORM_REGRESSION,
    MINIMUM_OVERALL_IMPROVEMENT,
)
from glassbox.workflows.source_group_benchmark import benchmark_source_groups


@dataclass(frozen=True)
class PolicyCandidate:
    """One platform-neutral fitting-policy candidate."""

    model_class: str
    training_horizons_s: tuple[float, ...]
    endpoint_weight: float
    stability_regularization: float

    def __post_init__(self) -> None:
        if self.model_class not in {"structured", "structured_residual"}:
            raise ValueError(f"unsupported model class {self.model_class!r}")
        if not self.training_horizons_s or any(
            horizon <= 0.0 for horizon in self.training_horizons_s
        ):
            raise ValueError("training horizons must be positive")
        if self.endpoint_weight <= 0.0:
            raise ValueError("endpoint weight must be positive")
        if self.stability_regularization < 0.0:
            raise ValueError("stability regularization cannot be negative")

    @property
    def candidate_id(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(payload.encode()).hexdigest()[:10]
        horizon_label = "-".join(
            _number_label(value) for value in self.training_horizons_s
        )
        return (
            f"{self.model_class}__h-{horizon_label}"
            f"__ew-{_number_label(self.endpoint_weight)}"
            f"__sr-{_number_label(self.stability_regularization)}__{digest}"
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["training_horizons_s"] = list(self.training_horizons_s)
        return result


@dataclass(frozen=True)
class PolicySelectionPlan:
    """Maintainer-owned search and decision policy.

    This is deliberately a Python-level object rather than a collection of CLI
    flags. It keeps experiments reproducible while the normal user interface
    stays opinionated.
    """

    name: str
    candidates: tuple[PolicyCandidate, ...]
    evaluation_horizons_s: tuple[float, ...]
    steps: int
    learning_rate: float = 0.01
    maximum_metric_regression: float = MAXIMUM_METRIC_REGRESSION
    maximum_platform_regression: float = MAXIMUM_PLATFORM_REGRESSION
    minimum_overall_improvement: float = MINIMUM_OVERALL_IMPROVEMENT

    def __post_init__(self) -> None:
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.name) is None:
            raise ValueError("plan name must be safe for use in an artifact record")
        candidates = tuple(self.candidates)
        if not candidates:
            raise ValueError("policy-selection plan needs at least one candidate")
        if len({candidate.candidate_id for candidate in candidates}) != len(candidates):
            raise ValueError("policy-selection plan candidates must be unique")
        reference = candidates[0]
        if not (
            reference.model_class == "structured"
            and reference.endpoint_weight == 1.0
            and reference.stability_regularization == 0.0
        ):
            raise ValueError(
                "the first candidate must be the unregularized structured reference"
            )
        horizons = tuple(float(value) for value in self.evaluation_horizons_s)
        if not horizons or any(horizon <= 0.0 for horizon in horizons):
            raise ValueError("evaluation horizons must be positive")
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning rate must be positive")
        if self.maximum_metric_regression < 1.0:
            raise ValueError("maximum metric regression must be at least one")
        if self.maximum_platform_regression < 1.0:
            raise ValueError("maximum platform regression must be at least one")
        if not 0.0 <= self.minimum_overall_improvement < 1.0:
            raise ValueError("minimum overall improvement must be in [0, 1)")
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "evaluation_horizons_s", horizons)

    @property
    def reference(self) -> PolicyCandidate:
        return self.candidates[0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "evaluation_horizons_s": list(self.evaluation_horizons_s),
            "steps": self.steps,
            "learning_rate": self.learning_rate,
            "maximum_metric_regression": self.maximum_metric_regression,
            "maximum_platform_regression": self.maximum_platform_regression,
            "minimum_overall_improvement": self.minimum_overall_improvement,
        }


def maintained_policy_selection_plan(*, smoke: bool = False) -> PolicySelectionPlan:
    """Return the versioned, maintained policy search used by the CLI."""

    short_horizons = (0.1, 0.5)
    long_horizons = (0.1, 0.5, 1.0)
    reference = PolicyCandidate("structured", short_horizons, 1.0, 0.0)
    shared_objective = PolicyCandidate("structured", short_horizons, 3.0, 0.01)
    candidates = (
        reference,
        shared_objective,
    )
    if not smoke:
        candidates += (
            PolicyCandidate("structured", long_horizons, 3.0, 0.01),
            PolicyCandidate("structured_residual", short_horizons, 3.0, 0.01),
            PolicyCandidate("structured_residual", long_horizons, 3.0, 0.01),
        )
    return PolicySelectionPlan(
        name="smoke_v1" if smoke else "maintained_v1",
        candidates=candidates,
        evaluation_horizons_s=(0.1, 0.5) if smoke else (0.1, 0.5, 1.0),
        steps=1 if smoke else 400,
    )


def _number_label(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def _geometric_mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty collection")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _file_record(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _validate_training_dataset(name: str, paths: Sequence[Path]) -> dict[str, str]:
    if not paths:
        raise ValueError(f"dataset {name!r} has no trajectories")
    trajectories = [load_trajectory_npz(path) for path in paths]
    split_values = {
        str(trajectory.labels["benchmark_split"]).strip().lower()
        for trajectory in trajectories
        if "benchmark_split" in trajectory.labels
    }
    forbidden = sorted(split for split in split_values if split != "train")
    if forbidden:
        raise ValueError(
            f"dataset {name!r} contains non-training benchmark split(s): "
            f"{', '.join(forbidden)}"
        )
    profiles = [trajectory.labels.get("profile") for trajectory in trajectories]
    source_groups = [
        trajectory.labels.get("source_group") for trajectory in trajectories
    ]
    if (
        all(profile is not None for profile in profiles)
        and len({str(profile) for profile in profiles}) >= 2
    ):
        fold_axis = "profile"
    elif (
        all(group is not None for group in source_groups)
        and len({str(group) for group in source_groups}) >= 2
    ):
        fold_axis = "source_group"
    else:
        raise ValueError(
            f"dataset {name!r} needs at least two complete profiles or source groups"
        )
    platforms = {
        trajectory.spec.vehicle.family
        for trajectory in trajectories
        if trajectory.spec is not None
    }
    if len(platforms) != 1:
        raise ValueError(f"dataset {name!r} must contain exactly one platform family")
    return {
        "platform": str(next(iter(platforms))),
        "fold_axis": fold_axis,
    }


def _summary_folds(
    summary: Mapping[str, Any],
) -> tuple[str, Mapping[str, Mapping[str, Any]]]:
    if "per_profile" in summary:
        return "profile", summary["per_profile"]
    if "per_source_group" in summary:
        return "source_group", summary["per_source_group"]
    raise ValueError("benchmark summary has no supported fold collection")


def _all_rollouts_finite(summary: Mapping[str, Any]) -> bool:
    def finite_numbers(value: Any) -> bool:
        if isinstance(value, Mapping):
            return all(finite_numbers(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return all(finite_numbers(item) for item in value)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return math.isfinite(float(value))
        return True

    _, folds = _summary_folds(summary)
    for fold in folds.values():
        rollouts = [fold["full_rollout"], *fold["horizon_rollouts"].values()]
        if not all(finite_numbers(rollout) for rollout in rollouts):
            return False
    return True


def score_policy_candidates(
    candidate_summaries: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    reference_id: str,
    evaluation_horizons_s: Sequence[float],
    maximum_metric_regression: float = MAXIMUM_METRIC_REGRESSION,
    maximum_platform_regression: float = MAXIMUM_PLATFORM_REGRESSION,
    minimum_overall_improvement: float = MINIMUM_OVERALL_IMPROVEMENT,
) -> tuple[dict[str, dict[str, Any]], list[str], str]:
    """Score candidates with equal platform and profile influence.

    Each leaf score is a ratio to the explicit reference policy. Metrics and
    evaluation horizons are geometric-mean aggregated within a profile,
    profiles are equally aggregated within a platform, and platforms are
    equally aggregated globally. Lower is better.
    """

    if reference_id not in candidate_summaries:
        raise ValueError(f"reference candidate {reference_id!r} is missing")
    if maximum_metric_regression < 1.0:
        raise ValueError("maximum metric regression must be at least one")
    if maximum_platform_regression < 1.0:
        raise ValueError("maximum platform regression must be at least one")
    if not 0.0 <= minimum_overall_improvement < 1.0:
        raise ValueError("minimum overall improvement must be in [0, 1)")
    horizon_labels = tuple(f"{value:g}s" for value in evaluation_horizons_s)
    reference = candidate_summaries[reference_id]
    scored: dict[str, dict[str, Any]] = {}

    for candidate_id, datasets in candidate_summaries.items():
        rejection_reasons: list[str] = []
        fold_scores_by_platform: dict[str, list[float]] = {}
        fold_scores: dict[str, float] = {}
        largest_metric_ratio = 0.0

        if set(datasets) != set(reference):
            rejection_reasons.append("candidate and reference dataset sets differ")
        for dataset_name in sorted(set(datasets) & set(reference)):
            summary = datasets[dataset_name]
            reference_summary = reference[dataset_name]
            platform = str(summary["platform"])
            if platform != str(reference_summary["platform"]):
                rejection_reasons.append(
                    f"{dataset_name}: platform differs from reference"
                )
                continue
            if not _all_rollouts_finite(summary):
                rejection_reasons.append(f"{dataset_name}: non-finite rollout metric")
                continue
            candidate_axis, candidate_folds = _summary_folds(summary)
            reference_axis, reference_folds = _summary_folds(reference_summary)
            if candidate_axis != reference_axis:
                rejection_reasons.append(
                    f"{dataset_name}: fold axis differs from reference"
                )
                continue
            if set(candidate_folds) != set(reference_folds):
                rejection_reasons.append(
                    f"{dataset_name}: held-out fold set differs from reference"
                )
                continue

            for fold_name in sorted(candidate_folds):
                ratios: list[float] = []
                candidate_horizons = candidate_folds[fold_name]["horizon_rollouts"]
                reference_horizons = reference_folds[fold_name]["horizon_rollouts"]
                missing = [
                    label
                    for label in horizon_labels
                    if label not in candidate_horizons
                    or label not in reference_horizons
                ]
                if missing:
                    rejection_reasons.append(
                        f"{dataset_name}/{fold_name}: missing horizons "
                        + ", ".join(missing)
                    )
                    continue
                for label in horizon_labels:
                    for metric in ROLLOUT_METRICS:
                        candidate_value = float(candidate_horizons[label][metric])
                        reference_value = float(reference_horizons[label][metric])
                        if not (
                            math.isfinite(candidate_value)
                            and math.isfinite(reference_value)
                            and candidate_value >= 0.0
                            and reference_value >= 0.0
                        ):
                            rejection_reasons.append(
                                f"{dataset_name}/{fold_name}/{label}/{metric}: "
                                "invalid comparison value"
                            )
                            continue
                        floor = METRIC_FLOORS[metric]
                        ratio = max(candidate_value, floor) / max(
                            reference_value, floor
                        )
                        ratios.append(ratio)
                        largest_metric_ratio = max(largest_metric_ratio, ratio)
                if ratios:
                    fold_key = f"{dataset_name}/{fold_name}"
                    fold_score = _geometric_mean(ratios)
                    fold_scores[fold_key] = fold_score
                    fold_scores_by_platform.setdefault(platform, []).append(fold_score)

        platform_scores = {
            platform: _geometric_mean(values)
            for platform, values in sorted(fold_scores_by_platform.items())
        }
        overall_score: float | None = (
            _geometric_mean(list(platform_scores.values())) if platform_scores else None
        )
        if (
            candidate_id != reference_id
            and largest_metric_ratio > maximum_metric_regression
        ):
            rejection_reasons.append(
                "largest metric regression "
                f"{largest_metric_ratio:.4g} exceeds {maximum_metric_regression:g}"
            )
        if (
            candidate_id != reference_id
            and overall_score is not None
            and overall_score < 1.0
        ):
            regressed_platforms = {
                platform: score
                for platform, score in platform_scores.items()
                if score > maximum_platform_regression
            }
            if regressed_platforms:
                details = ", ".join(
                    f"{platform}={score:.4g}"
                    for platform, score in regressed_platforms.items()
                )
                rejection_reasons.append(
                    "aggregate improvement is concentrated; platform regression: "
                    + details
                )
        scored[candidate_id] = {
            "eligible": not rejection_reasons,
            "overall_score": overall_score,
            "clears_minimum_improvement": (
                candidate_id == reference_id
                or (
                    overall_score is not None
                    and overall_score <= 1.0 - minimum_overall_improvement
                )
            ),
            "platform_scores": platform_scores,
            "fold_scores": fold_scores,
            "largest_metric_ratio": largest_metric_ratio,
            "rejection_reasons": rejection_reasons,
        }

    rankable = [
        candidate_id
        for candidate_id, result in scored.items()
        if result["eligible"] and result["overall_score"] is not None
    ]

    def ranking_key(candidate_id: str) -> tuple[float, int, str]:
        result = scored[candidate_id]
        score = float(result["overall_score"])
        if candidate_id == reference_id:
            return (1.0, 0, candidate_id)
        if not result["clears_minimum_improvement"] and score < 1.0:
            return (1.0, 1, candidate_id)
        return (score, 1, candidate_id)

    ranking = sorted(rankable, key=ranking_key)
    if not ranking:
        raise RuntimeError("no eligible policy candidate remains")
    return scored, ranking, ranking[0]


def select_fitting_policy(
    datasets: Mapping[str, Sequence[str | Path]],
    output_dir: str | Path,
    *,
    plan: PolicySelectionPlan | None = None,
) -> dict[str, Any]:
    """Run or resume the matrix, score it, and write an auditable decision."""

    plan = maintained_policy_selection_plan() if plan is None else plan
    if not datasets:
        raise ValueError("at least one dataset is required")
    invalid_names = [
        str(name)
        for name in datasets
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", str(name)) is None
    ]
    if invalid_names:
        raise ValueError(
            "dataset names must use letters, digits, dot, underscore, or hyphen: "
            + ", ".join(invalid_names)
        )
    resolved_datasets = {
        str(name): tuple(Path(path).resolve() for path in paths)
        for name, paths in datasets.items()
    }
    dataset_metadata = {
        name: _validate_training_dataset(name, paths)
        for name, paths in resolved_datasets.items()
    }
    dataset_platforms = {
        name: metadata["platform"] for name, metadata in dataset_metadata.items()
    }
    dataset_fold_axes = {
        name: metadata["fold_axis"] for name, metadata in dataset_metadata.items()
    }
    dataset_files = {
        name: [_file_record(path) for path in paths]
        for name, paths in resolved_datasets.items()
    }
    candidates = {candidate.candidate_id: candidate for candidate in plan.candidates}
    reference = plan.reference

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, dict[str, dict[str, Any]]] = {}
    result_records: dict[str, Any] = {}
    for candidate_id, candidate in sorted(candidates.items()):
        print(f"candidate: {candidate_id}")
        summaries[candidate_id] = {}
        result_records[candidate_id] = {
            "configuration": candidate.to_dict(),
            "datasets": {},
        }
        for dataset_name, paths in resolved_datasets.items():
            fold_directory = destination / "candidates" / candidate_id / dataset_name
            summary_path = fold_directory / "summary.json"
            request_path = fold_directory / "request.json"
            request = {
                "format_version": 1,
                "candidate": candidate.to_dict(),
                "dataset": dataset_name,
                "platform": dataset_platforms[dataset_name],
                "fold_axis": dataset_fold_axes[dataset_name],
                "files": dataset_files[dataset_name],
                "evaluation_horizons_s": list(plan.evaluation_horizons_s),
                "steps": plan.steps,
                "learning_rate": plan.learning_rate,
            }
            reusable = (
                summary_path.exists()
                and request_path.exists()
                and _load_summary(request_path) == request
            )
            if reusable:
                print(f"  resume {dataset_name}: {summary_path}")
                summary = _load_summary(summary_path)
            else:
                fold_axis = dataset_fold_axes[dataset_name]
                print(
                    f"  run {dataset_name} ({dataset_platforms[dataset_name]}, "
                    f"{fold_axis})"
                )
                benchmark = (
                    benchmark_profiles
                    if fold_axis == "profile"
                    else benchmark_source_groups
                )
                summary = benchmark(
                    paths,
                    fold_directory,
                    training_horizons_s=candidate.training_horizons_s,
                    evaluation_horizons_s=plan.evaluation_horizons_s,
                    steps=plan.steps,
                    learning_rate=plan.learning_rate,
                    run_no_lag_ablation=False,
                    model_class=candidate.model_class,
                    endpoint_weight=candidate.endpoint_weight,
                    stability_regularization=candidate.stability_regularization,
                )
                request_path.write_text(json.dumps(request, indent=2) + "\n")
            summaries[candidate_id][dataset_name] = summary
            result_records[candidate_id]["datasets"][dataset_name] = {
                "platform": str(summary["platform"]),
                "fold_axis": dataset_fold_axes[dataset_name],
                "summary": str(summary_path.resolve()),
            }

    scored, ranking, selected_id = score_policy_candidates(
        summaries,
        reference_id=reference.candidate_id,
        evaluation_horizons_s=plan.evaluation_horizons_s,
        maximum_metric_regression=plan.maximum_metric_regression,
        maximum_platform_regression=plan.maximum_platform_regression,
        minimum_overall_improvement=plan.minimum_overall_improvement,
    )
    for candidate_id, result in scored.items():
        result_records[candidate_id]["score"] = result

    decision = {
        "format_version": 1,
        "evaluation": "cross_platform_grouped_policy_selection",
        "decision_scope": {
            "status": "provisional",
            "uses_external_test_data": False,
            "promotion_requires_external_validation": True,
        },
        "policy_plan": plan.to_dict(),
        "data_policy": {
            "benchmark_test_trajectories_allowed": False,
            "dataset_paths": {
                name: [str(path) for path in paths]
                for name, paths in resolved_datasets.items()
            },
            "dataset_files": dataset_files,
            "dataset_platforms": dataset_platforms,
            "dataset_fold_axes": dataset_fold_axes,
        },
        "scoring": {
            "reference_candidate": reference.candidate_id,
            "metrics": list(ROLLOUT_METRICS),
            "metric_floors": METRIC_FLOORS,
            "evaluation_horizons_s": list(plan.evaluation_horizons_s),
            "aggregation": "geometric_mean_equal_metric_horizon_fold_platform",
            "maximum_metric_regression": plan.maximum_metric_regression,
            "maximum_platform_regression": plan.maximum_platform_regression,
            "minimum_overall_improvement": plan.minimum_overall_improvement,
        },
        "optimization": {
            "steps_per_fold": plan.steps,
            "learning_rate": plan.learning_rate,
        },
        "candidate_count": len(candidates),
        "candidates": result_records,
        "ranking": ranking,
        "selected_candidate": selected_id,
        "selected_configuration": candidates[selected_id].to_dict(),
    }
    decision_path = destination / "selection.json"
    decision_path.write_text(json.dumps(decision, indent=2, allow_nan=False) + "\n")
    print(f"cross-validation selection: {selected_id}")
    print(f"wrote {decision_path}")
    return decision


def _dataset(value: str) -> tuple[str, tuple[Path, ...]]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("dataset must have the form NAME=GLOB")
    name, pattern = value.split("=", 1)
    name = name.strip()
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) is None:
        raise argparse.ArgumentTypeError(
            "dataset name must start with a letter or digit and use only "
            "letters, digits, dot, underscore, or hyphen"
        )
    paths = tuple(Path(path) for path in sorted(glob.glob(pattern, recursive=True)))
    if not paths:
        raise argparse.ArgumentTypeError(f"dataset glob matched no files: {pattern}")
    return name, paths


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="append",
        type=_dataset,
        required=True,
        metavar="NAME=GLOB",
        help="repeat for each corpus; quote globs so Glassbox expands them",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/policy_selection")
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="validate ingestion and the full workflow with one optimizer step",
    )
    args = parser.parse_args(argv)

    datasets: dict[str, tuple[Path, ...]] = {}
    for name, paths in args.dataset:
        if name in datasets:
            parser.error(f"duplicate dataset name: {name}")
        datasets[name] = paths
    select_fitting_policy(
        datasets,
        args.output_dir,
        plan=maintained_policy_selection_plan(smoke=args.smoke),
    )


if __name__ == "__main__":
    main()
