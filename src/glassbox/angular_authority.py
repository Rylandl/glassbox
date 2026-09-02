"""Train-only selection of conservative multirotor angular dynamics."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from glassbox.data import Trajectory, duration_to_steps, load_trajectory_npz
from glassbox.dynamics import (
    ModelParams,
    model_family,
    with_angular_dynamics_authority,
)
from glassbox.evaluation import (
    METRIC_FLOORS,
    ROLLOUT_METRICS,
    aggregate_rollout_metrics,
    kinematic_persistence_windowed_metrics,
    rollout_divergence_metrics,
    rollout_metrics,
    windowed_rollout_metrics,
)
from glassbox.model_io import load_dynamics_model
from glassbox.policy_selection import score_policy_candidates

ANGULAR_AUTHORITY_CANDIDATES = (0.0, 0.25, 0.5, 0.75, 1.0)
ANGULAR_AUTHORITY_REFERENCE = "authority_1"
ANGULAR_AUTHORITY_HORIZONS_S = (0.1, 0.5, 1.0)
AUTHORITY_PROMOTION_METRICS = (
    "position_rmse_m",
    "velocity_rmse_m_s",
    "attitude_rmse_deg",
    "angular_velocity_rmse_rad_s",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _authority_id(value: float) -> str:
    return f"authority_{value:g}".replace(".", "p")


def select_angular_dynamics_authority(
    fold_models: Mapping[str, str | Path],
    trajectory_paths: Sequence[str | Path],
    output_path: str | Path,
    *,
    fold_axis: str,
    dataset_name: str,
    required_benchmark_split: str | None = None,
) -> dict[str, Any]:
    """Select one bounded authority from complete held-out training folds.

    ``fold_models`` maps each profile or source-group label to the model trained
    while holding out that label. Candidate values and safety thresholds are
    intentionally maintained here instead of exposed as end-user fit knobs.
    """

    if fold_axis not in {"profile", "source_group"}:
        raise ValueError("fold_axis must be profile or source_group")
    if not dataset_name.strip():
        raise ValueError("dataset_name cannot be empty")

    model_paths = {
        str(fold): Path(path).resolve() for fold, path in fold_models.items()
    }
    paths = [Path(path).resolve() for path in trajectory_paths]
    if not paths:
        raise ValueError("angular authority selection requires trajectories")
    trajectories = [load_trajectory_npz(path) for path in paths]
    if required_benchmark_split is not None:
        invalid_splits = sorted(
            {
                str(trajectory.labels.get("benchmark_split"))
                for trajectory in trajectories
                if trajectory.labels.get("benchmark_split") != required_benchmark_split
            }
        )
        if invalid_splits:
            raise ValueError(
                "angular authority selection contains forbidden benchmark "
                "splits: " + ", ".join(invalid_splits)
            )

    fold_values = []
    for path, trajectory in zip(paths, trajectories):
        value = trajectory.labels.get(fold_axis)
        if value is None or not str(value).strip():
            raise ValueError(f"trajectory {path} lacks a {fold_axis} label")
        fold_values.append(str(value))
    folds = tuple(sorted(set(fold_values)))
    if set(model_paths) != set(folds):
        raise ValueError(
            f"fold_models must contain exactly one held-out model for every {fold_axis}"
        )

    reference_spec = trajectories[0].spec.to_dict()
    if any(trajectory.spec.to_dict() != reference_spec for trajectory in trajectories):
        raise ValueError("authority selection requires one trajectory spec")
    if trajectories[0].spec.vehicle.family != "multirotor":
        raise ValueError("angular authority selection requires multirotor data")

    loaded_models = {
        fold: load_dynamics_model(path) for fold, path in model_paths.items()
    }
    for fold, (_, payload) in loaded_models.items():
        if payload.get("platform") != "multirotor":
            raise ValueError(f"held-out model for {fold!r} is not multirotor")
        if payload.get("input_spec") != reference_spec:
            raise ValueError(f"held-out model input spec differs for {fold!r}")

    fold_collection_key = (
        "per_profile" if fold_axis == "profile" else "per_source_group"
    )
    persistence_by_fold: dict[str, Any] = {}
    for fold in folds:
        held_out = [
            trajectory
            for trajectory, value in zip(trajectories, fold_values)
            if value == fold
        ]
        horizon_rollouts = {}
        for seconds in ANGULAR_AUTHORITY_HORIZONS_S:
            metrics = []
            for trajectory in held_out:
                horizon_steps = duration_to_steps(seconds, trajectory.nominal_dt_s)
                metrics.append(
                    kinematic_persistence_windowed_metrics(
                        trajectory,
                        horizon_steps=horizon_steps,
                        stride_steps=horizon_steps,
                    )
                )
            horizon_rollouts[f"{seconds:g}s"] = aggregate_rollout_metrics(
                metrics, weighting="equal"
            )
        persistence_by_fold[fold] = {
            "validation_trajectory_count": len(held_out),
            "horizon_rollouts": horizon_rollouts,
        }
    candidate_summaries: dict[str, dict[str, dict[str, Any]]] = {}
    candidate_evaluation: dict[str, dict[str, Any]] = {}
    for authority in ANGULAR_AUTHORITY_CANDIDATES:
        candidate_id = _authority_id(authority)
        per_fold: dict[str, Any] = {}
        for fold in folds:
            params, _ = loaded_models[fold]
            candidate_params = with_angular_dynamics_authority(params, authority)
            held_out = [
                trajectory
                for trajectory, value in zip(trajectories, fold_values)
                if value == fold
            ]
            full_rollouts = [
                rollout_metrics(candidate_params, trajectory) for trajectory in held_out
            ]
            horizon_rollouts: dict[str, Any] = {}
            for seconds in ANGULAR_AUTHORITY_HORIZONS_S:
                metrics = []
                for trajectory in held_out:
                    horizon_steps = duration_to_steps(seconds, trajectory.nominal_dt_s)
                    metrics.append(
                        windowed_rollout_metrics(
                            candidate_params,
                            trajectory,
                            horizon_steps=horizon_steps,
                            stride_steps=horizon_steps,
                        )
                    )
                horizon_rollouts[f"{seconds:g}s"] = aggregate_rollout_metrics(
                    metrics, weighting="equal"
                )
            per_fold[fold] = {
                "validation_trajectory_count": len(held_out),
                "full_rollout": aggregate_rollout_metrics(
                    full_rollouts, weighting="equal"
                ),
                "horizon_rollouts": horizon_rollouts,
            }
        summary = {
            "platform": "multirotor",
            "fold_axis": fold_axis,
            "folds": list(folds),
            "trajectory_count": len(trajectories),
            "angular_dynamics_authority": authority,
            fold_collection_key: per_fold,
        }
        candidate_evaluation[candidate_id] = summary
        candidate_summaries[candidate_id] = {dataset_name: summary}

    scored, ranking, selected = score_policy_candidates(
        candidate_summaries,
        reference_id=ANGULAR_AUTHORITY_REFERENCE,
        evaluation_horizons_s=ANGULAR_AUTHORITY_HORIZONS_S,
        maximum_metric_regression=1.25,
        maximum_platform_regression=1.0,
        minimum_overall_improvement=0.01,
    )
    authorities_by_id = {
        _authority_id(value): value for value in ANGULAR_AUTHORITY_CANDIDATES
    }
    selected_folds = candidate_evaluation[selected][fold_collection_key]
    aggregate_vs_persistence = {}
    selected_persistence_ratios = []
    for seconds in ANGULAR_AUTHORITY_HORIZONS_S:
        label = f"{seconds:g}s"
        candidate = aggregate_rollout_metrics(
            [selected_folds[fold]["horizon_rollouts"][label] for fold in folds],
            weighting="equal",
        )
        persistence = aggregate_rollout_metrics(
            [persistence_by_fold[fold]["horizon_rollouts"][label] for fold in folds],
            weighting="equal",
        )
        ratios = {
            metric: max(float(candidate[metric]), METRIC_FLOORS[metric])
            / max(float(persistence[metric]), METRIC_FLOORS[metric])
            for metric in ROLLOUT_METRICS
        }
        selected_persistence_ratios.extend(ratios.values())
        aggregate_vs_persistence[label] = {
            "candidate": candidate,
            "kinematic_persistence": persistence,
            "candidate_over_kinematic_persistence": ratios,
        }
    selected_persistence_geometric_ratio = math.exp(
        float(np.mean(np.log(np.asarray(selected_persistence_ratios))))
    )
    decision = {
        "format_version": 1,
        "evaluation": "train_only_angular_dynamics_authority",
        "dataset": dataset_name,
        "decision_scope": {
            "status": "provisional",
            "fold_axis": fold_axis,
            "required_benchmark_split": required_benchmark_split,
            "uses_protected_evaluation_data": False,
            "promotion_requires_untouched_evaluation": True,
            "complete_flight_accuracy_claim": False,
        },
        "folds": list(folds),
        "evaluation_horizons_s": list(ANGULAR_AUTHORITY_HORIZONS_S),
        "candidate_authorities": list(ANGULAR_AUTHORITY_CANDIDATES),
        "reference_candidate": ANGULAR_AUTHORITY_REFERENCE,
        "fold_models": {
            fold: {"path": str(path), "sha256": _sha256(path)}
            for fold, path in sorted(model_paths.items())
        },
        "selection_trajectories": [
            {"path": str(path), "sha256": _sha256(path)} for path in paths
        ],
        "scores": scored,
        "ranking": ranking,
        "selected_candidate": selected,
        "selected_authority": authorities_by_id[selected],
        "candidate_evaluation": candidate_evaluation,
        "kinematic_persistence_evaluation": {
            "platform": "multirotor",
            "fold_axis": fold_axis,
            "folds": list(folds),
            fold_collection_key: persistence_by_fold,
        },
        "selected_candidate_vs_kinematic_persistence": {
            "geometric_ratio": selected_persistence_geometric_ratio,
            "maximum_individual_ratio": max(selected_persistence_ratios),
            "aggregate_horizon_rollouts": aggregate_vs_persistence,
            "selection_criterion": False,
        },
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(decision, indent=2) + "\n")
    return decision


def evaluate_angular_dynamics_candidate(
    reference_params: ModelParams,
    candidate_params: ModelParams,
    trajectories: Sequence[Trajectory],
) -> dict[str, Any]:
    """Gate an already-selected candidate on untouched complete trajectories."""

    if not trajectories:
        raise ValueError("candidate evaluation requires untouched trajectories")
    if any(
        trajectory.spec.vehicle.family != "multirotor" for trajectory in trajectories
    ):
        raise ValueError("candidate evaluation requires multirotor trajectories")
    if model_family(reference_params).platform != "multirotor":
        raise ValueError("candidate evaluation requires a multirotor reference")
    if model_family(candidate_params).platform != "multirotor":
        raise ValueError("candidate evaluation requires a multirotor candidate")
    per_trajectory: list[dict[str, Any]] = []
    reference_by_horizon: dict[str, list[dict[str, Any]]] = {
        f"{seconds:g}s": [] for seconds in ANGULAR_AUTHORITY_HORIZONS_S
    }
    candidate_by_horizon = {label: [] for label in reference_by_horizon}
    persistence_by_horizon = {label: [] for label in reference_by_horizon}
    candidate_divergence = []
    for trajectory in trajectories:
        trajectory_horizons = {}
        for seconds in ANGULAR_AUTHORITY_HORIZONS_S:
            label = f"{seconds:g}s"
            horizon_steps = duration_to_steps(seconds, trajectory.nominal_dt_s)
            reference = windowed_rollout_metrics(
                reference_params,
                trajectory,
                horizon_steps=horizon_steps,
                stride_steps=horizon_steps,
            )
            candidate = windowed_rollout_metrics(
                candidate_params,
                trajectory,
                horizon_steps=horizon_steps,
                stride_steps=horizon_steps,
            )
            persistence = kinematic_persistence_windowed_metrics(
                trajectory,
                horizon_steps=horizon_steps,
                stride_steps=horizon_steps,
            )
            reference_by_horizon[label].append(reference)
            candidate_by_horizon[label].append(candidate)
            persistence_by_horizon[label].append(persistence)
            trajectory_horizons[label] = {
                "reference": reference,
                "candidate": candidate,
                "kinematic_persistence": persistence,
            }
        divergence = rollout_divergence_metrics(candidate_params, trajectory)
        candidate_divergence.append(divergence)
        per_trajectory.append(
            {
                "source_group": trajectory.labels.get("source_group"),
                "duration_s": float(trajectory.time_s[-1]),
                "horizon_rollouts": trajectory_horizons,
                "reference_full_rollout": rollout_metrics(reference_params, trajectory),
                "candidate_full_rollout": rollout_metrics(candidate_params, trajectory),
                "reference_divergence": rollout_divergence_metrics(
                    reference_params, trajectory
                ),
                "candidate_divergence": divergence,
            }
        )

    aggregate_horizons: dict[str, Any] = {}
    candidate_reference_ratios = []
    candidate_persistence_ratios = []
    rotational_reference_ratios = []
    rotational_persistence_ratios = []
    for label in reference_by_horizon:
        reference = aggregate_rollout_metrics(
            reference_by_horizon[label], weighting="equal"
        )
        candidate = aggregate_rollout_metrics(
            candidate_by_horizon[label], weighting="equal"
        )
        persistence = aggregate_rollout_metrics(
            persistence_by_horizon[label], weighting="equal"
        )
        over_reference = {
            name: max(float(candidate[name]), METRIC_FLOORS[name])
            / max(float(reference[name]), METRIC_FLOORS[name])
            for name in AUTHORITY_PROMOTION_METRICS
        }
        over_persistence = {
            name: max(float(candidate[name]), METRIC_FLOORS[name])
            / max(float(persistence[name]), METRIC_FLOORS[name])
            for name in AUTHORITY_PROMOTION_METRICS
        }
        candidate_reference_ratios.extend(over_reference.values())
        candidate_persistence_ratios.extend(over_persistence.values())
        rotational_reference_ratios.extend(
            over_reference[name]
            for name in (
                "attitude_rmse_deg",
                "angular_velocity_rmse_rad_s",
            )
        )
        rotational_persistence_ratios.extend(
            over_persistence[name]
            for name in (
                "attitude_rmse_deg",
                "angular_velocity_rmse_rad_s",
            )
        )
        aggregate_horizons[label] = {
            "reference": reference,
            "candidate": candidate,
            "kinematic_persistence": persistence,
            "candidate_over_reference": over_reference,
            "candidate_over_persistence": over_persistence,
        }

    def geometric(values: list[float]) -> float:
        return math.exp(float(np.mean(np.log(np.asarray(values)))))

    reference_ratio = geometric(candidate_reference_ratios)
    persistence_ratio = geometric(candidate_persistence_ratios)
    improves_reference = (
        reference_ratio <= 0.99 and max(candidate_reference_ratios) <= 1.10
    )
    beats_persistence = persistence_ratio <= 0.99
    complete_flight_stable = all(not item["diverged"] for item in candidate_divergence)
    if improves_reference and beats_persistence:
        status = (
            "promote_complete_flight"
            if complete_flight_stable
            else "promote_short_horizon_only"
        )
    elif improves_reference:
        status = "improves_reference_only"
    else:
        status = "reject"
    return {
        "format_version": 1,
        "evaluation": "untouched_multirotor_angular_candidate",
        "scope": "logged_input_prediction_not_flight_safety",
        "horizons_s": list(ANGULAR_AUTHORITY_HORIZONS_S),
        "trajectory_count": len(trajectories),
        "gates": {
            "improves_fitted_reference": {
                "passed": improves_reference,
                "geometric_ratio": reference_ratio,
                "maximum_individual_ratio": max(candidate_reference_ratios),
                "required_geometric_ratio": 0.99,
                "maximum_allowed_individual_ratio": 1.10,
            },
            "beats_kinematic_persistence": {
                "passed": beats_persistence,
                "geometric_ratio": persistence_ratio,
                "required_geometric_ratio": 0.99,
            },
            "rotational_ratio_vs_reference": geometric(rotational_reference_ratios),
            "rotational_ratio_vs_kinematic_persistence": geometric(
                rotational_persistence_ratios
            ),
            "complete_flight_stable": {
                "passed": complete_flight_stable,
                "policy": "no configured divergence threshold crossed",
            },
        },
        "status": status,
        "aggregate_horizon_rollouts": aggregate_horizons,
        "per_trajectory": per_trajectory,
    }
