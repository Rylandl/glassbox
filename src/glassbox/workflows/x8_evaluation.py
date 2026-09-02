"""Untouched-validation evaluation for the Skywalker X8 reference campaign."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from glassbox.core.data import Trajectory, duration_to_steps, load_trajectory_npz
from glassbox.core.evaluation import (
    NEGLIGIBLE_METRIC_FLOORS,
    ROLLOUT_METRICS,
    VECTORIZED_LOG_MEAN,
    aggregate_rollout_metrics,
    kinematic_persistence_windowed_metrics,
    persistence_score,
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


def load_validation_trajectories(
    paths: list[str | Path] | tuple[str | Path, ...],
) -> tuple[list[Path], list[Trajectory]]:
    """Load the upstream validation maneuvers and check their pinned identity."""

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


def horizon_steps_for_duration(trajectory: Trajectory, horizon_s: float) -> int:
    """Return the integer step count for one horizon at a trajectory's rate."""

    steps = duration_to_steps(horizon_s, trajectory.nominal_dt_s)
    if not np.isclose(steps * trajectory.nominal_dt_s, horizon_s, atol=1e-9, rtol=0.0):
        raise ValueError(
            f"horizon {horizon_s:g}s is not representable at the sample rate"
        )
    return steps


def aggregate_horizon_rollouts(
    per_trajectory: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Average each horizon's rollout metrics across validation maneuvers."""

    labels = tuple(per_trajectory[0]["horizon_rollouts"])
    return {
        label: aggregate_rollout_metrics(
            [item["horizon_rollouts"][label] for item in per_trajectory],
            weighting="equal",
        )
        for label in labels
    }


def geometric_ratio(
    candidate: Mapping[str, Mapping[str, Any]],
    reference: Mapping[str, Mapping[str, Any]],
) -> float:
    """Score a candidate against a reference over every horizon it carries.

    The campaign reports every horizon it evaluates, so this scores all of
    them, and uses only a negligible floor because both sides are real flight
    errors rather than a possibly-zero baseline.
    """

    return persistence_score(
        candidate,
        reference,
        horizons=None,
        floors=NEGLIGIBLE_METRIC_FLOORS,
        metrics=ROLLOUT_METRICS,
        aggregation=VECTORIZED_LOG_MEAN,
    )


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
    paths, trajectories = load_validation_trajectories(validation_paths)
    horizon_steps = {
        f"{horizon:g}s": horizon_steps_for_duration(trajectories[0], horizon)
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
    persistence_aggregate = aggregate_horizon_rollouts(persistence_per_trajectory)

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
        aggregate = aggregate_horizon_rollouts(per_trajectory)
        models[name] = {
            "path": str(model_path),
            "model_type": payload["model_type"],
            "aggregate": {"horizon_rollouts": aggregate},
            "per_trajectory": per_trajectory,
            "score_vs_kinematic_persistence": geometric_ratio(
                aggregate, persistence_aggregate
            ),
            "score_horizons_s": list(horizons_s),
        }

    comparisons = {
        f"{candidate}_vs_{reference}": {
            "ratio_definition": (
                "candidate/reference geometric mean over four state metrics "
                "and every horizon; values below one favor the candidate"
            ),
            "score": geometric_ratio(
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
            "score_horizons_s": list(horizons_s),
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
