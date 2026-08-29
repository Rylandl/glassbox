"""Opinionated train-only selection for multirotor rotational dynamics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from glassbox.data import duration_to_steps, load_trajectory_npz
from glassbox.dynamics import with_angular_dynamics_authority
from glassbox.evaluation import (
    aggregate_rollout_metrics,
    rollout_metrics,
    windowed_rollout_metrics,
)
from glassbox.model_io import load_dynamics_model
from glassbox.policy_selection import score_policy_candidates


ROTATION_SELECTION_HORIZONS_S = (0.1, 0.5, 1.0)
ROTATION_SELECTION_PROFILES = ("chirp", "random", "square")
ROTATION_REFERENCE = "instantaneous_diagonal"
ANGULAR_AUTHORITY_CANDIDATES = (0.0, 0.25, 0.5, 0.75, 1.0)
ANGULAR_AUTHORITY_REFERENCE = "authority_1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_nanodrone_rotation_candidate(
    summaries: Mapping[str, str | Path],
    output_path: str | Path,
) -> dict[str, Any]:
    """Select rotational structure using only NanoDrone training profiles.

    The public Melon test split is intentionally absent from this interface.
    Candidates must improve the equal-profile, equal-metric geometric score by
    at least one percent under the shared policy-selection safety checks.
    """

    if ROTATION_REFERENCE not in summaries:
        raise ValueError(
            f"rotation selection requires {ROTATION_REFERENCE!r} reference"
        )
    paths = {name: Path(path).resolve() for name, path in summaries.items()}
    loaded: dict[str, dict[str, dict[str, Any]]] = {}
    for name, path in paths.items():
        summary = json.loads(path.read_text())
        profiles = tuple(sorted(str(value) for value in summary["profiles"]))
        if profiles != ROTATION_SELECTION_PROFILES:
            raise ValueError(
                f"candidate {name!r} must use only train profiles "
                f"{ROTATION_SELECTION_PROFILES}, got {profiles}"
            )
        if summary.get("platform") != "multirotor":
            raise ValueError(f"candidate {name!r} is not a multirotor summary")
        loaded[name] = {"nanodrone_train": summary}

    scored, ranking, selected = score_policy_candidates(
        loaded,
        reference_id=ROTATION_REFERENCE,
        evaluation_horizons_s=ROTATION_SELECTION_HORIZONS_S,
        maximum_metric_regression=1.5,
        maximum_platform_regression=1.05,
        minimum_overall_improvement=0.01,
    )
    decision = {
        "format_version": 1,
        "evaluation": "nanodrone_train_only_rotational_structure_selection",
        "decision_scope": {
            "status": "provisional",
            "uses_public_melon_test_data": False,
            "promotion_requires_one_shot_melon_evaluation": True,
            "complete_flight_accuracy_claim": False,
        },
        "profiles": list(ROTATION_SELECTION_PROFILES),
        "evaluation_horizons_s": list(ROTATION_SELECTION_HORIZONS_S),
        "reference_candidate": ROTATION_REFERENCE,
        "minimum_improvement": 0.01,
        "candidate_summaries": {
            name: {
                "path": str(path),
                "sha256": _sha256(path),
            }
            for name, path in sorted(paths.items())
        },
        "scores": scored,
        "ranking": ranking,
        "selected_candidate": selected,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(decision, indent=2) + "\n")
    return decision


def _authority_id(value: float) -> str:
    return f"authority_{value:g}".replace(".", "p")


def select_nanodrone_angular_authority(
    fold_models: Mapping[str, str | Path],
    trajectory_paths: Sequence[str | Path],
    output_path: str | Path,
) -> dict[str, Any]:
    """Select conservative angular-dynamics authority without Melon data.

    Each profile is scored only by the model for which that profile was held
    out. The fixed candidate grid is maintainer-owned, so the normal fitting
    interface gains no additional user-facing hyperparameter.
    """

    if tuple(sorted(fold_models)) != ROTATION_SELECTION_PROFILES:
        raise ValueError(
            "angular authority selection requires one held-out model for each "
            f"training profile {ROTATION_SELECTION_PROFILES}"
        )
    model_paths = {
        profile: Path(path).resolve() for profile, path in fold_models.items()
    }
    loaded_models = {
        profile: load_dynamics_model(path)
        for profile, path in model_paths.items()
    }
    paths = [Path(path).resolve() for path in trajectory_paths]
    trajectories = [load_trajectory_npz(path) for path in paths]
    if not trajectories:
        raise ValueError("angular authority selection requires training trajectories")
    invalid_splits = sorted(
        {
            str(trajectory.labels.get("benchmark_split"))
            for trajectory in trajectories
            if trajectory.labels.get("benchmark_split") != "train"
        }
    )
    if invalid_splits:
        raise ValueError(
            "angular authority selection cannot use non-training splits: "
            + ", ".join(invalid_splits)
        )
    profiles = tuple(
        sorted(
            {str(trajectory.labels.get("profile")) for trajectory in trajectories}
        )
    )
    if profiles != ROTATION_SELECTION_PROFILES:
        raise ValueError(
            f"training trajectories must contain profiles {ROTATION_SELECTION_PROFILES}"
        )
    for profile, (_, payload) in loaded_models.items():
        if payload.get("platform") != "multirotor":
            raise ValueError(f"held-out model for {profile!r} is not multirotor")
        matching = [
            trajectory
            for trajectory in trajectories
            if trajectory.labels.get("profile") == profile
        ]
        if any(
            trajectory.spec.to_dict() != payload["input_spec"]
            for trajectory in matching
        ):
            raise ValueError(f"held-out model input spec differs for {profile!r}")

    candidate_summaries: dict[str, dict[str, dict[str, Any]]] = {}
    evaluation: dict[str, dict[str, Any]] = {}
    for authority in ANGULAR_AUTHORITY_CANDIDATES:
        candidate_id = _authority_id(authority)
        per_profile: dict[str, Any] = {}
        for profile in ROTATION_SELECTION_PROFILES:
            params, _ = loaded_models[profile]
            candidate_params = with_angular_dynamics_authority(params, authority)
            held_out = [
                trajectory
                for trajectory in trajectories
                if trajectory.labels.get("profile") == profile
            ]
            full_rollouts = [
                rollout_metrics(candidate_params, trajectory)
                for trajectory in held_out
            ]
            horizon_rollouts: dict[str, Any] = {}
            for seconds in ROTATION_SELECTION_HORIZONS_S:
                metrics = []
                for trajectory in held_out:
                    horizon_steps = duration_to_steps(
                        seconds, trajectory.nominal_dt_s
                    )
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
            per_profile[profile] = {
                "validation_flight_count": len(held_out),
                "full_rollout": aggregate_rollout_metrics(
                    full_rollouts, weighting="equal"
                ),
                "horizon_rollouts": horizon_rollouts,
            }
        summary = {
            "platform": "multirotor",
            "profiles": list(ROTATION_SELECTION_PROFILES),
            "trajectory_count": len(trajectories),
            "angular_dynamics_authority": authority,
            "per_profile": per_profile,
        }
        evaluation[candidate_id] = summary
        candidate_summaries[candidate_id] = {"nanodrone_train": summary}

    scored, ranking, selected = score_policy_candidates(
        candidate_summaries,
        reference_id=ANGULAR_AUTHORITY_REFERENCE,
        evaluation_horizons_s=ROTATION_SELECTION_HORIZONS_S,
        maximum_metric_regression=1.25,
        maximum_platform_regression=1.0,
        minimum_overall_improvement=0.01,
    )
    authorities_by_id = {
        _authority_id(value): value for value in ANGULAR_AUTHORITY_CANDIDATES
    }
    decision = {
        "format_version": 1,
        "evaluation": "nanodrone_train_only_angular_dynamics_authority",
        "decision_scope": {
            "status": "provisional",
            "uses_public_melon_test_data": False,
            "promotion_requires_one_shot_melon_evaluation": True,
            "complete_flight_accuracy_claim": False,
        },
        "profiles": list(ROTATION_SELECTION_PROFILES),
        "evaluation_horizons_s": list(ROTATION_SELECTION_HORIZONS_S),
        "candidate_authorities": list(ANGULAR_AUTHORITY_CANDIDATES),
        "reference_candidate": ANGULAR_AUTHORITY_REFERENCE,
        "fold_models": {
            profile: {"path": str(path), "sha256": _sha256(path)}
            for profile, path in sorted(model_paths.items())
        },
        "training_trajectories": [
            {"path": str(path), "sha256": _sha256(path)} for path in paths
        ],
        "scores": scored,
        "ranking": ranking,
        "selected_candidate": selected,
        "selected_authority": authorities_by_id[selected],
        "candidate_evaluation": evaluation,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(decision, indent=2) + "\n")
    return decision
