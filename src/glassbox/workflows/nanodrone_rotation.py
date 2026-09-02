"""Opinionated train-only selection for multirotor rotational dynamics."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from glassbox.workflows.angular_authority import (
    select_angular_dynamics_authority,
)
from glassbox.workflows.policy_selection import score_policy_candidates
from glassbox.workflows.selection import (
    MAXIMUM_METRIC_REGRESSION,
    MAXIMUM_PLATFORM_REGRESSION,
    MINIMUM_OVERALL_IMPROVEMENT,
)

ROTATION_SELECTION_HORIZONS_S = (0.1, 0.5, 1.0)
ROTATION_SELECTION_PROFILES = ("chirp", "random", "square")
ROTATION_REFERENCE = "instantaneous_diagonal"


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
        maximum_metric_regression=MAXIMUM_METRIC_REGRESSION,
        maximum_platform_regression=MAXIMUM_PLATFORM_REGRESSION,
        minimum_overall_improvement=MINIMUM_OVERALL_IMPROVEMENT,
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
        "minimum_improvement": MINIMUM_OVERALL_IMPROVEMENT,
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
    return select_angular_dynamics_authority(
        fold_models,
        trajectory_paths,
        output_path,
        fold_axis="profile",
        dataset_name="nanodrone_train",
        required_benchmark_split="train",
    )
