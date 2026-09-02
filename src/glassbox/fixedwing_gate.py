"""Cross-airframe fixed-wing development and promotion gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from glassbox.acceptance import evaluate_fixedwing_accuracy
from glassbox.data import Trajectory, duration_to_steps, load_trajectory_npz
from glassbox.evaluation import (
    METRIC_FLOORS,
    ROLLOUT_METRICS,
    aggregate_rollout_metrics,
    kinematic_persistence_windowed_metrics,
    rollout_divergence_metrics,
)
from glassbox.model_io import load_dynamics_model

FIXED_WING_GATE_HORIZONS_S = (0.1, 0.5, 1.0, 2.0)
FIXED_WING_SCORE_HORIZONS_S = (0.5, 1.0, 2.0)


def _sha256_record(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _resolve_recorded_path(value: str | Path, *, anchor: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    return (anchor / path).resolve()


def _geometric_mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty score collection")
    return float(math.exp(sum(math.log(value) for value in values) / len(values)))


def _score_against_persistence(
    candidate: Mapping[str, Mapping[str, Any]],
    persistence: Mapping[str, Mapping[str, Any]],
) -> float:
    ratios = []
    for seconds in FIXED_WING_SCORE_HORIZONS_S:
        label = f"{seconds:g}s"
        if label not in candidate or label not in persistence:
            raise ValueError(f"missing fixed-wing gate horizon {label}")
        for metric in ROLLOUT_METRICS:
            floor = METRIC_FLOORS[metric]
            ratios.append(
                max(float(candidate[label][metric]), floor)
                / max(float(persistence[label][metric]), floor)
            )
    return _geometric_mean(ratios)


def _p90_horizons(
    per_item: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, float]]:
    if not per_item:
        raise ValueError("p90 horizon summary requires at least one item")
    result: dict[str, dict[str, float]] = {}
    for seconds in FIXED_WING_GATE_HORIZONS_S:
        label = f"{seconds:g}s"
        result[label] = {
            metric: float(
                np.quantile(
                    [
                        float(item["horizon_rollouts"][label][metric])
                        for item in per_item
                    ],
                    0.9,
                )
            )
            for metric in ROLLOUT_METRICS
        }
    return result


def _persistence_horizons(
    trajectories: Sequence[Trajectory],
) -> dict[str, dict[str, Any]]:
    per_trajectory = []
    for trajectory in trajectories:
        per_trajectory.append(
            {
                f"{seconds:g}s": kinematic_persistence_windowed_metrics(
                    trajectory,
                    horizon_steps=duration_to_steps(seconds, trajectory.nominal_dt_s),
                )
                for seconds in FIXED_WING_GATE_HORIZONS_S
            }
        )
    return {
        f"{seconds:g}s": aggregate_rollout_metrics(
            [item[f"{seconds:g}s"] for item in per_trajectory],
            weighting="equal",
        )
        for seconds in FIXED_WING_GATE_HORIZONS_S
    }


def _summarize_divergence(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("divergence summary requires at least one rollout")
    stable_times = np.asarray(
        [float(item["stable_through_s"]) for item in items], dtype=np.float64
    )
    stable_fractions = np.asarray(
        [float(item["stable_fraction"]) for item in items], dtype=np.float64
    )
    causes = Counter(cause for item in items for cause in item["divergence_causes"])
    finite_count = sum(bool(item["full_rollout_finite"]) for item in items)
    return {
        "trajectory_count": len(items),
        "full_rollout_finite_fraction": finite_count / len(items),
        "diverged_fraction": sum(bool(item["diverged"]) for item in items) / len(items),
        "stable_through_s": {
            "minimum": float(np.min(stable_times)),
            "p10": float(np.quantile(stable_times, 0.1)),
            "median": float(np.median(stable_times)),
            "maximum": float(np.max(stable_times)),
        },
        "stable_fraction": {
            "minimum": float(np.min(stable_fractions)),
            "p10": float(np.quantile(stable_fractions, 0.1)),
            "median": float(np.median(stable_fractions)),
            "maximum": float(np.max(stable_fractions)),
        },
        "cause_counts": dict(sorted(causes.items())),
        "per_trajectory": list(items),
    }


def _source_group_airframe(summary_path: Path) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text())
    if summary.get("platform") != "fixedwing" or "per_source_group" not in summary:
        raise ValueError("IDF input must be a fixed-wing source-group summary")
    anchor = summary_path.parent
    persistence_by_group = []
    divergence_items = []
    candidate_per_group = []
    configuration_ids: set[str] = set()
    for group_name, fold in summary["per_source_group"].items():
        report_path = _resolve_recorded_path(fold["report"], anchor=anchor)
        model_path = _resolve_recorded_path(fold["model"], anchor=anchor)
        report = json.loads(report_path.read_text())
        validation_paths = [
            _resolve_recorded_path(item["path"], anchor=report_path.parent)
            for item in report["split"]["validation_flights"]
        ]
        trajectories = [load_trajectory_npz(path) for path in validation_paths]
        configuration_ids.update(
            trajectory.spec.vehicle.configuration_id or "unknown"
            for trajectory in trajectories
        )
        persistence_by_group.append(_persistence_horizons(trajectories))
        candidate_per_group.append({"horizon_rollouts": fold["horizon_rollouts"]})
        params, _ = load_dynamics_model(model_path)
        for path, trajectory in zip(validation_paths, trajectories):
            diagnostic = rollout_divergence_metrics(params, trajectory)
            divergence_items.append(
                {
                    "path": str(path),
                    "source_group": group_name,
                    **diagnostic,
                }
            )

    persistence = {
        f"{seconds:g}s": aggregate_rollout_metrics(
            [item[f"{seconds:g}s"] for item in persistence_by_group],
            weighting="equal",
        )
        for seconds in FIXED_WING_GATE_HORIZONS_S
    }
    aggregate = summary["aggregate"]["horizon_rollouts"]
    p90 = {
        f"{seconds:g}s": {
            metric: float(
                summary["distribution"]["horizon_rollouts"][f"{seconds:g}s"][metric][
                    "p90"
                ]
            )
            for metric in ROLLOUT_METRICS
        }
        for seconds in FIXED_WING_GATE_HORIZONS_S
    }
    divergence = _summarize_divergence(divergence_items)
    return {
        "configuration_ids": sorted(configuration_ids),
        "protocol": "leave_one_source_group_out",
        "aggregate_horizon_rollouts": aggregate,
        "p90_horizon_rollouts": p90,
        "kinematic_persistence_horizon_rollouts": persistence,
        "score_vs_kinematic_persistence": _score_against_persistence(
            aggregate, persistence
        ),
        "full_rollout_finite_fraction": divergence["full_rollout_finite_fraction"],
        "divergence": divergence,
        "source": _sha256_record(summary_path),
    }


def _x8_airframe(report_path: Path, model_name: str) -> dict[str, Any]:
    report = json.loads(report_path.read_text())
    if model_name not in report.get("models", {}):
        raise ValueError(f"X8 benchmark has no model named {model_name!r}")
    model = report["models"][model_name]
    aggregate = model["aggregate"]["horizon_rollouts"]
    persistence = report["kinematic_persistence"]["aggregate"]["horizon_rollouts"]
    p90 = _p90_horizons(model["per_trajectory"])
    model_path = _resolve_recorded_path(model["path"], anchor=report_path.parent)
    params, _ = load_dynamics_model(model_path)
    divergence_items = []
    configuration_ids: set[str] = set()
    for item in model["per_trajectory"]:
        path = _resolve_recorded_path(item["path"], anchor=report_path.parent)
        trajectory = load_trajectory_npz(path)
        configuration_ids.add(trajectory.spec.vehicle.configuration_id or "unknown")
        divergence_items.append(
            {
                "path": str(path),
                **rollout_divergence_metrics(params, trajectory),
            }
        )
    divergence = _summarize_divergence(divergence_items)
    return {
        "configuration_ids": sorted(configuration_ids),
        "protocol": "frozen_upstream_validation",
        "aggregate_horizon_rollouts": aggregate,
        "p90_horizon_rollouts": p90,
        "kinematic_persistence_horizon_rollouts": persistence,
        "score_vs_kinematic_persistence": _score_against_persistence(
            aggregate, persistence
        ),
        "full_rollout_finite_fraction": divergence["full_rollout_finite_fraction"],
        "divergence": divergence,
        "source": _sha256_record(report_path),
        "model": _sha256_record(model_path),
    }


def evaluate_fixedwing_gate(
    idf_summary_path: str | Path,
    x8_report_path: str | Path,
    *,
    x8_model_name: str,
    candidate_name: str,
) -> dict[str, Any]:
    """Evaluate one candidate on conventional and flying-wing airframes."""

    if not candidate_name.strip():
        raise ValueError("candidate_name cannot be empty")
    idf_path = Path(idf_summary_path).resolve()
    x8_path = Path(x8_report_path).resolve()
    airframes = {
        "idf_conventional": _source_group_airframe(idf_path),
        "x8_flying_wing": _x8_airframe(x8_path, x8_model_name),
    }
    return {
        "format_version": 1,
        "evaluation": "fixedwing_cross_airframe_development_gate",
        "candidate": candidate_name,
        "data_roles": {
            "idf_conventional": "development_cross_validation",
            "x8_flying_wing": "frozen_external_promotion",
        },
        "scoring": {
            "horizons_s": list(FIXED_WING_GATE_HORIZONS_S),
            "persistence_score_horizons_s": list(FIXED_WING_SCORE_HORIZONS_S),
            "metrics": list(ROLLOUT_METRICS),
            "persistence_baseline": ("constant_world_velocity_and_constant_body_rate"),
            "airframe_weighting": "equal",
            "divergence_thresholds": next(iter(airframes.values()))["divergence"][
                "per_trajectory"
            ][0]["thresholds"],
        },
        "airframes": airframes,
        "equal_airframe_score_vs_kinematic_persistence": _geometric_mean(
            [
                float(airframe["score_vs_kinematic_persistence"])
                for airframe in airframes.values()
            ]
        ),
        "acceptance": evaluate_fixedwing_accuracy(airframes),
    }


def save_fixedwing_gate(report: Mapping[str, Any], path: str | Path) -> None:
    """Write a deterministic, finite cross-airframe gate report."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def compare_fixedwing_gates(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    """Select a candidate only when gains are shared by every airframe."""

    if set(reference["airframes"]) != set(candidate["airframes"]):
        raise ValueError("fixed-wing gate airframe sets differ")
    airframe_scores: dict[str, float] = {}
    stable_horizon_ratios: dict[str, float] = {}
    largest_metric_ratio = 0.0
    for airframe_name in reference["airframes"]:
        reference_airframe = reference["airframes"][airframe_name]
        candidate_airframe = candidate["airframes"][airframe_name]
        ratios = []
        for seconds in FIXED_WING_SCORE_HORIZONS_S:
            label = f"{seconds:g}s"
            for metric in ROLLOUT_METRICS:
                floor = METRIC_FLOORS[metric]
                ratio = max(
                    float(
                        candidate_airframe["aggregate_horizon_rollouts"][label][metric]
                    ),
                    floor,
                ) / max(
                    float(
                        reference_airframe["aggregate_horizon_rollouts"][label][metric]
                    ),
                    floor,
                )
                ratios.append(ratio)
                largest_metric_ratio = max(largest_metric_ratio, ratio)
        airframe_scores[airframe_name] = _geometric_mean(ratios)
        reference_stable = float(
            reference_airframe["divergence"]["stable_through_s"]["median"]
        )
        candidate_stable = float(
            candidate_airframe["divergence"]["stable_through_s"]["median"]
        )
        stable_horizon_ratios[airframe_name] = (
            candidate_stable / reference_stable if reference_stable > 0.0 else math.inf
        )

    overall_score = _geometric_mean(list(airframe_scores.values()))
    rejection_reasons = []
    regressed_airframes = {
        name: score for name, score in airframe_scores.items() if score > 1.05
    }
    if regressed_airframes:
        rejection_reasons.append(
            "airframe regression exceeds 5%: "
            + ", ".join(
                f"{name}={score:.4g}"
                for name, score in sorted(regressed_airframes.items())
            )
        )
    if largest_metric_ratio > 1.5:
        rejection_reasons.append(
            f"largest metric regression {largest_metric_ratio:.4g} exceeds 1.5"
        )
    nonfinite_airframes = [
        name
        for name, airframe in candidate["airframes"].items()
        if float(airframe["full_rollout_finite_fraction"]) < 1.0
    ]
    if nonfinite_airframes:
        rejection_reasons.append(
            "non-finite complete rollout: " + ", ".join(nonfinite_airframes)
        )
    clears_minimum_improvement = overall_score <= 0.99
    if not clears_minimum_improvement:
        rejection_reasons.append(
            f"overall score {overall_score:.4g} does not improve at least 1%"
        )
    eligible = not rejection_reasons
    selected_candidate = (
        str(candidate["candidate"]) if eligible else str(reference["candidate"])
    )
    return {
        "format_version": 1,
        "evaluation": "fixedwing_cross_airframe_candidate_selection",
        "reference_candidate": str(reference["candidate"]),
        "candidate": str(candidate["candidate"]),
        "ratio_definition": (
            "candidate/reference geometric mean over four state metrics at "
            "0.5, 1, and 2 seconds; values below one favor the candidate"
        ),
        "airframe_scores": airframe_scores,
        "overall_score": overall_score,
        "largest_metric_ratio": largest_metric_ratio,
        "median_stable_horizon_ratio": stable_horizon_ratios,
        "maximum_airframe_regression": 1.05,
        "maximum_metric_regression": 1.5,
        "minimum_overall_improvement": 0.01,
        "clears_minimum_improvement": clears_minimum_improvement,
        "eligible": eligible,
        "rejection_reasons": rejection_reasons,
        "selected_candidate": selected_candidate,
        "selected_candidate_contract_status": (
            candidate["acceptance"]["status"]
            if eligible
            else reference["acceptance"]["status"]
        ),
        "interpretation": (
            "selected_for_continued_development_but_contract_not_met"
            if eligible and candidate["acceptance"]["status"] != "pass"
            else "selected_candidate_meets_contract"
            if eligible
            else "reference_retained"
        ),
    }


def screen_fixedwing_airframe_candidate(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    model_name: str,
    airframe_name: str,
    candidate_name: str,
) -> dict[str, Any]:
    """Fail fast on one airframe before running a cross-airframe candidate."""

    if (
        not model_name.strip()
        or not airframe_name.strip()
        or not candidate_name.strip()
    ):
        raise ValueError("model, airframe, and candidate names must be non-empty")
    try:
        reference_horizons = reference["models"][model_name]["aggregate"][
            "horizon_rollouts"
        ]
        candidate_horizons = candidate["models"][model_name]["aggregate"][
            "horizon_rollouts"
        ]
    except KeyError as error:
        raise ValueError(
            f"benchmark report is missing model data for {model_name!r}"
        ) from error

    ratios: dict[str, dict[str, float]] = {}
    flattened: list[tuple[str, str, float]] = []
    for seconds in FIXED_WING_SCORE_HORIZONS_S:
        label = f"{seconds:g}s"
        if label not in reference_horizons or label not in candidate_horizons:
            raise ValueError(f"benchmark report is missing horizon {label}")
        ratios[label] = {}
        for metric in ROLLOUT_METRICS:
            floor = METRIC_FLOORS[metric]
            ratio = max(float(candidate_horizons[label][metric]), floor) / max(
                float(reference_horizons[label][metric]), floor
            )
            ratios[label][metric] = ratio
            flattened.append((label, metric, ratio))

    overall_score = _geometric_mean([item[2] for item in flattened])
    largest_label, largest_metric, largest_ratio = max(
        flattened, key=lambda item: item[2]
    )
    rejection_reasons = []
    if overall_score > 0.99:
        rejection_reasons.append(
            f"overall score {overall_score:.4g} does not improve at least 1%"
        )
    if largest_ratio > 1.5:
        rejection_reasons.append(
            f"largest metric regression {largest_ratio:.4g} exceeds 1.5"
        )
    eligible = not rejection_reasons
    return {
        "format_version": 1,
        "evaluation": "fixedwing_single_airframe_candidate_screen",
        "airframe": airframe_name,
        "model_name": model_name,
        "candidate": candidate_name,
        "scope": "fail_fast_development_screen_only",
        "can_promote_model": False,
        "ratio_definition": (
            "candidate/reference geometric mean over four state metrics at "
            "0.5, 1, and 2 seconds; values below one favor the candidate"
        ),
        "horizons_s": list(FIXED_WING_SCORE_HORIZONS_S),
        "metrics": list(ROLLOUT_METRICS),
        "metric_ratios": ratios,
        "overall_score": overall_score,
        "largest_metric_ratio": {
            "horizon": largest_label,
            "metric": largest_metric,
            "value": largest_ratio,
        },
        "minimum_overall_improvement": 0.01,
        "maximum_metric_regression": 1.5,
        "eligible_for_cross_airframe_evaluation": eligible,
        "rejection_reasons": rejection_reasons,
        "interpretation": (
            "advance_to_cross_airframe_gate"
            if eligible
            else "reject_before_cross_airframe_fit"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluate_parser = subparsers.add_parser(
        "evaluate", help="evaluate one candidate on both airframes"
    )
    evaluate_parser.add_argument("--idf-summary", type=Path, required=True)
    evaluate_parser.add_argument("--x8-report", type=Path, required=True)
    evaluate_parser.add_argument("--x8-model-name", required=True)
    evaluate_parser.add_argument("--candidate-name", required=True)
    evaluate_parser.add_argument("--output", type=Path, required=True)
    compare_parser = subparsers.add_parser(
        "compare", help="compare two completed cross-airframe gate reports"
    )
    compare_parser.add_argument("--reference", type=Path, required=True)
    compare_parser.add_argument("--candidate", type=Path, required=True)
    compare_parser.add_argument("--output", type=Path, required=True)
    screen_parser = subparsers.add_parser(
        "screen", help="screen one airframe before a cross-airframe fit"
    )
    screen_parser.add_argument("--reference-report", type=Path, required=True)
    screen_parser.add_argument("--candidate-report", type=Path, required=True)
    screen_parser.add_argument("--model-name", required=True)
    screen_parser.add_argument("--airframe-name", required=True)
    screen_parser.add_argument("--candidate-name", required=True)
    screen_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "screen":
        reference = json.loads(args.reference_report.read_text())
        candidate = json.loads(args.candidate_report.read_text())
        report = screen_fixedwing_airframe_candidate(
            reference,
            candidate,
            model_name=args.model_name,
            airframe_name=args.airframe_name,
            candidate_name=args.candidate_name,
        )
        report["sources"] = {
            "reference": _sha256_record(args.reference_report),
            "candidate": _sha256_record(args.candidate_report),
        }
        save_fixedwing_gate(report, args.output)
        print(
            f"{args.candidate_name}: "
            f"eligible={report['eligible_for_cross_airframe_evaluation']} "
            f"score={report['overall_score']:.3f}"
        )
        print(f"wrote {args.output}")
        return
    if args.command == "compare":
        reference = json.loads(args.reference.read_text())
        candidate = json.loads(args.candidate.read_text())
        report = compare_fixedwing_gates(reference, candidate)
        save_fixedwing_gate(report, args.output)
        print(
            f"selected={report['selected_candidate']} "
            f"score={report['overall_score']:.3f} "
            f"contract={report['selected_candidate_contract_status']}"
        )
        print(f"wrote {args.output}")
        return
    report = evaluate_fixedwing_gate(
        args.idf_summary,
        args.x8_report,
        x8_model_name=args.x8_model_name,
        candidate_name=args.candidate_name,
    )
    save_fixedwing_gate(report, args.output)
    print(
        f"{args.candidate_name}: status={report['acceptance']['status']} "
        "score/persistence="
        f"{report['equal_airframe_score_vs_kinematic_persistence']:.3f}"
    )
    for name, airframe in report["airframes"].items():
        divergence = airframe["divergence"]
        print(
            f"  {name}: score={airframe['score_vs_kinematic_persistence']:.3f} "
            f"median_stable={divergence['stable_through_s']['median']:.2f}s "
            f"finite={divergence['full_rollout_finite_fraction']:.1%}"
        )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
