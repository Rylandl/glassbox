"""Honest same-flight characterization for the EPFL TOPOPlane2 corpus."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from glassbox.data import duration_to_steps, load_trajectory_npz
from glassbox.evaluation import (
    METRIC_FLOORS,
    ROLLOUT_METRICS,
    aggregate_rollout_metrics,
    kinematic_persistence_windowed_metrics,
)

EPFL_CHARACTERIZATION_HORIZONS_S = (0.2, 0.5, 1.0, 2.0)
EPFL_SCORE_HORIZONS_S = (0.5, 1.0, 2.0)


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


def _resolve_trajectory_path(value: str, *, report_path: Path) -> Path:
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    anchored = report_path.parent / path
    if anchored.exists():
        return anchored
    raise FileNotFoundError(path)


def _score_against_persistence(
    model: Mapping[str, Mapping[str, Any]],
    persistence: Mapping[str, Mapping[str, Any]],
) -> float:
    ratios = []
    for seconds in EPFL_SCORE_HORIZONS_S:
        label = f"{seconds:g}s"
        for metric in ROLLOUT_METRICS:
            floor = METRIC_FLOORS[metric]
            ratios.append(
                max(float(model[label][metric]), floor)
                / max(float(persistence[label][metric]), floor)
            )
    return float(math.exp(sum(math.log(value) for value in ratios) / len(ratios)))


def evaluate_epfl_characterization(
    structured_report_path: str | Path,
    residual_report_path: str | Path,
) -> dict[str, Any]:
    """Compare maintained models with persistence on the chronological holdout."""

    report_paths = {
        "structured": Path(structured_report_path),
        "structured_residual": Path(residual_report_path),
    }
    reports = {
        name: json.loads(path.read_text()) for name, path in report_paths.items()
    }
    expected_classes = {
        "structured": "structured",
        "structured_residual": "structured_residual",
    }
    for name, report in reports.items():
        if report["configuration"]["model_class"] != expected_classes[name]:
            raise ValueError(f"{name} fit report has the wrong model class")
        if report["split"]["mode"] != (
            "chronological_segments_within_source_group_characterization"
        ):
            raise ValueError(f"{name} report is not an EPFL characterization split")
        if report["split"]["independent_source_group_holdout"] is not False:
            raise ValueError(
                "EPFL characterization must not claim an independent holdout"
            )

    structured_split = reports["structured"]["split"]
    residual_split = reports["structured_residual"]["split"]
    structured_validation = [
        item["path"] for item in structured_split["validation_flights"]
    ]
    residual_validation = [
        item["path"] for item in residual_split["validation_flights"]
    ]
    if structured_validation != residual_validation:
        raise ValueError("EPFL reports must evaluate the same validation segments")
    if [item["path"] for item in structured_split["training_flights"]] != [
        item["path"] for item in residual_split["training_flights"]
    ]:
        raise ValueError("EPFL reports must fit the same training segments")

    validation_paths = [
        _resolve_trajectory_path(value, report_path=report_paths["structured"])
        for value in structured_validation
    ]
    validation = [load_trajectory_npz(path) for path in validation_paths]
    sample_rate_hz = 1.0 / validation[0].nominal_dt_s
    persistence = {}
    effective_horizons = {}
    for seconds in EPFL_CHARACTERIZATION_HORIZONS_S:
        label = f"{seconds:g}s"
        per_trajectory = []
        step_counts = set()
        for trajectory in validation:
            steps = duration_to_steps(seconds, trajectory.nominal_dt_s)
            step_counts.add(steps)
            per_trajectory.append(
                kinematic_persistence_windowed_metrics(
                    trajectory,
                    horizon_steps=steps,
                    stride_steps=steps,
                )
            )
        if len(step_counts) != 1:
            raise ValueError("EPFL validation trajectories use inconsistent rates")
        steps = step_counts.pop()
        effective_horizons[label] = {
            "requested_s": seconds,
            "steps": steps,
            "effective_s": steps / sample_rate_hz,
        }
        persistence[label] = aggregate_rollout_metrics(
            per_trajectory, weighting="equal"
        )

    models = {}
    for name, report in reports.items():
        learned = report["models"]["learned_lag"]
        horizons = learned["validation"]["aggregate"]["horizon_rollouts"]
        if any(
            f"{seconds:g}s" not in horizons
            for seconds in EPFL_CHARACTERIZATION_HORIZONS_S
        ):
            raise ValueError(f"{name} report is missing a characterization horizon")
        models[name] = {
            "fit_report": _sha256_record(report_paths[name]),
            "fit": {
                "initial_loss": learned["fit"]["initial_loss"],
                "final_loss": learned["fit"]["final_loss"],
                "loss_reduction": learned["fit"]["loss_reduction"],
                "wall_time_s": learned["fit"]["wall_time_s"],
            },
            "aggregate_horizon_rollouts": horizons,
            "aggregate_full_rollout": learned["validation"]["aggregate"][
                "full_rollout"
            ],
            "score_vs_kinematic_persistence": _score_against_persistence(
                horizons, persistence
            ),
        }

    selected_model = min(
        models,
        key=lambda name: models[name]["score_vs_kinematic_persistence"],
    )
    return {
        "format_version": 1,
        "evaluation": "epfl_topoplane2_same_flight_characterization",
        "protocol": {
            "split": "chronological_segments_within_one_source_flight",
            "independent_source_group_holdout": False,
            "training_segment_count": len(structured_split["training_flights"]),
            "validation_segment_count": len(validation),
            "sample_rate_hz": sample_rate_hz,
            "requested_and_effective_horizons": effective_horizons,
            "score_horizons_s": list(EPFL_SCORE_HORIZONS_S),
            "score_metrics": list(ROLLOUT_METRICS),
            "score_definition": (
                "model/persistence geometric mean; values below one favor the model"
            ),
        },
        "dataset": reports["structured"]["dataset"],
        "kinematic_persistence": {
            "aggregate_horizon_rollouts": persistence,
        },
        "models": models,
        "selected_model": selected_model,
        "can_promote_model": False,
        "interpretation": (
            "useful same-flight airframe characterization; independent flights "
            "are required before this result can enter the promotion gate"
        ),
        "limitations": [
            "all retained segments come from one published flight",
            "angular velocity is derived from attitude at 5 Hz",
            "the requested 0.5-second horizon resolves to 0.4 seconds at 5 Hz",
            "complete-segment open-loop errors are diagnostic, not an operational claim",
        ],
    }


def save_epfl_characterization(report: Mapping[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
