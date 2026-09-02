"""Untouched-validation evaluation for the Skywalker X8 reference campaign."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from glassbox.core.data import Trajectory, duration_to_steps, load_trajectory_npz
from glassbox.core.evaluation import (
    aggregate_rollout_metrics,
    kinematic_persistence_windowed_metrics,
    windowed_rollout_metrics,
)
from glassbox.core.model_io import load_dynamics_model
from glassbox.io.x8_reference import (
    X8_REFERENCE_DOI,
    X8_REFERENCE_NAME,
    X8_REFERENCE_VERSION,
    x8_trajectory_spec,
)

X8_EVALUATION_HORIZONS_S = (0.1, 0.5, 1.0, 2.0)
_SCORE_METRICS = (
    "position_rmse_m",
    "velocity_rmse_m_s",
    "attitude_rmse_deg",
    "angular_velocity_rmse_rad_s",
)


def _validation_trajectories(
    paths: list[str | Path] | tuple[str | Path, ...],
) -> tuple[list[Path], list[Trajectory]]:
    resolved = [Path(path).resolve() for path in paths]
    if not resolved:
        raise ValueError("at least one Skywalker X8 validation trajectory is required")
    trajectories = [load_trajectory_npz(path) for path in resolved]
    expected_spec = x8_trajectory_spec()
    for path, trajectory in zip(resolved, trajectories):
        if trajectory.spec != expected_spec:
            raise ValueError(f"trajectory does not match the Skywalker X8 spec: {path}")
        if trajectory.labels.get("benchmark_split") != "validation":
            raise ValueError(
                f"trajectory is not in the upstream validation split: {path}"
            )
    return resolved, trajectories


def _horizon_steps(trajectory: Trajectory, horizon_s: float) -> int:
    steps = duration_to_steps(horizon_s, trajectory.nominal_dt_s)
    if not np.isclose(steps * trajectory.nominal_dt_s, horizon_s, atol=1e-9, rtol=0.0):
        raise ValueError(
            f"horizon {horizon_s:g}s is not representable at the sample rate"
        )
    return steps


def _aggregate_horizons(
    per_trajectory: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    labels = tuple(per_trajectory[0]["horizon_rollouts"])
    return {
        label: aggregate_rollout_metrics(
            [item["horizon_rollouts"][label] for item in per_trajectory],
            weighting="equal",
        )
        for label in labels
    }


def _geometric_ratio(
    candidate: Mapping[str, Mapping[str, Any]],
    reference: Mapping[str, Mapping[str, Any]],
) -> float:
    ratios = [
        max(float(candidate[label][metric]), 1e-12)
        / max(float(reference[label][metric]), 1e-12)
        for label in candidate
        for metric in _SCORE_METRICS
    ]
    return float(np.exp(np.mean(np.log(ratios))))


def evaluate_x8_reference_models(
    model_paths: Mapping[str, str | Path],
    validation_paths: list[str | Path] | tuple[str | Path, ...],
    *,
    horizons_s: tuple[float, ...] = X8_EVALUATION_HORIZONS_S,
) -> dict[str, Any]:
    """Compare saved models and kinematic persistence on the upstream split."""

    if not model_paths:
        raise ValueError("at least one model artifact is required")
    if any(horizon <= 0.0 for horizon in horizons_s):
        raise ValueError("evaluation horizons must be positive")
    paths, trajectories = _validation_trajectories(validation_paths)
    horizon_steps = {
        f"{horizon:g}s": _horizon_steps(trajectories[0], horizon)
        for horizon in horizons_s
    }

    persistence_per_trajectory = []
    for path, trajectory in zip(paths, trajectories):
        persistence_per_trajectory.append(
            {
                "path": str(path),
                "horizon_rollouts": {
                    label: kinematic_persistence_windowed_metrics(
                        trajectory,
                        horizon_steps=steps,
                        stride_steps=1,
                    )
                    for label, steps in horizon_steps.items()
                },
            }
        )
    persistence_aggregate = _aggregate_horizons(persistence_per_trajectory)

    models: dict[str, Any] = {}
    for name, model_path_value in model_paths.items():
        if not name.strip():
            raise ValueError("model names must be non-empty")
        model_path = Path(model_path_value).resolve()
        params, payload = load_dynamics_model(model_path)
        if payload["input_spec"] != x8_trajectory_spec().to_dict():
            raise ValueError(
                f"model input spec does not match Skywalker X8: {model_path}"
            )
        per_trajectory = []
        for path, trajectory in zip(paths, trajectories):
            per_trajectory.append(
                {
                    "path": str(path),
                    "horizon_rollouts": {
                        label: windowed_rollout_metrics(
                            params,
                            trajectory,
                            horizon_steps=steps,
                            stride_steps=1,
                        )
                        for label, steps in horizon_steps.items()
                    },
                }
            )
        aggregate = _aggregate_horizons(per_trajectory)
        models[name] = {
            "path": str(model_path),
            "model_type": payload["model_type"],
            "aggregate": {"horizon_rollouts": aggregate},
            "per_trajectory": per_trajectory,
            "score_vs_kinematic_persistence": _geometric_ratio(
                aggregate, persistence_aggregate
            ),
        }

    comparisons = {
        f"{candidate}_vs_{reference}": {
            "ratio_definition": (
                "candidate/reference geometric mean over four state metrics "
                "and every horizon; values below one favor the candidate"
            ),
            "score": _geometric_ratio(
                models[candidate]["aggregate"]["horizon_rollouts"],
                models[reference]["aggregate"]["horizon_rollouts"],
            ),
        }
        for candidate in models
        for reference in models
        if candidate != reference
    }
    return {
        "format_version": 1,
        "benchmark": {
            "name": X8_REFERENCE_NAME,
            "doi": X8_REFERENCE_DOI,
            "version": X8_REFERENCE_VERSION,
        },
        "protocol": {
            "split": "upstream_validation",
            "initialization": "every_admissible_sample",
            "flight_boundaries_crossed": False,
            "horizons_s": list(horizons_s),
            "aggregation": "equal_validation_maneuver",
            "baseline": "constant_world_velocity_and_constant_body_rate",
        },
        "dataset": {
            "validation_trajectory_count": len(trajectories),
            "validation_duration_s": float(
                sum(trajectory.time_s[-1] for trajectory in trajectories)
            ),
            "trajectory_spec": x8_trajectory_spec().to_dict(),
        },
        "kinematic_persistence": {
            "aggregate": {"horizon_rollouts": persistence_aggregate},
            "per_trajectory": persistence_per_trajectory,
        },
        "models": models,
        "comparisons": comparisons,
        "acceptance": {
            "status": "not_scored",
            "passed": None,
            "reason": (
                "one public flying-wing campaign is evidence of configuration "
                "coverage, but is insufficient to set a cross-airframe threshold"
            ),
        },
    }


def save_x8_reference_report(report: Mapping[str, Any], path: str | Path) -> None:
    """Write a deterministic X8 comparison report."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
