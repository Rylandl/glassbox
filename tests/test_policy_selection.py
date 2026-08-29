import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from glassbox.data import Trajectory, make_trajectory_spec, save_trajectory_npz
from glassbox.policy_selection import (
    ROLLOUT_METRICS,
    PolicyCandidate,
    PolicySelectionPlan,
    maintained_policy_selection_plan,
    score_policy_candidates,
    select_fitting_policy,
)


def _metrics(value: float) -> dict[str, float]:
    return {metric: value for metric in ROLLOUT_METRICS}


def _summary(
    platform: str,
    profile_values: dict[str, float],
    *,
    full_value: float | None = None,
) -> dict:
    return {
        "platform": platform,
        "per_profile": {
            profile: {
                "full_rollout": _metrics(value if full_value is None else full_value),
                "horizon_rollouts": {
                    "0.1s": _metrics(value),
                    "0.5s": _metrics(value),
                },
            }
            for profile, value in profile_values.items()
        },
    }


def _source_summary(platform: str, source_values: dict[str, float]) -> dict:
    summary = _summary(platform, source_values)
    summary["per_source_group"] = summary.pop("per_profile")
    return summary


def test_scoring_weights_platforms_equally_and_selects_shared_improvement() -> None:
    candidates = {
        "reference": {
            "multirotor_a": _summary("multirotor", {"chirp": 1.0, "random": 1.0}),
            "multirotor_b": _summary("multirotor", {"square": 1.0}),
            "fixedwing": _summary("fixedwing", {"multisine": 1.0}),
        },
        "shared_improvement": {
            "multirotor_a": _summary("multirotor", {"chirp": 0.25, "random": 0.25}),
            "multirotor_b": _summary("multirotor", {"square": 0.25}),
            "fixedwing": _summary("fixedwing", {"multisine": 1.0}),
        },
    }

    scored, ranking, selected = score_policy_candidates(
        candidates,
        reference_id="reference",
        evaluation_horizons_s=(0.1, 0.5),
        maximum_metric_regression=2.0,
    )

    assert scored["shared_improvement"]["platform_scores"] == pytest.approx(
        {"multirotor": 0.25, "fixedwing": 1.0}
    )
    assert scored["shared_improvement"]["overall_score"] == pytest.approx(0.5)
    assert ranking == ["shared_improvement", "reference"]
    assert selected == "shared_improvement"


def test_scoring_combines_profile_and_source_group_folds() -> None:
    candidates = {
        "reference": {
            "profiles": _summary("multirotor", {"a": 1.0, "b": 1.0}),
            "sessions": _source_summary(
                "fixedwing", {"session-1": 1.0, "session-2": 1.0}
            ),
        },
        "improved": {
            "profiles": _summary("multirotor", {"a": 0.8, "b": 0.8}),
            "sessions": _source_summary(
                "fixedwing", {"session-1": 0.9, "session-2": 0.9}
            ),
        },
    }

    scored, _, selected = score_policy_candidates(
        candidates,
        reference_id="reference",
        evaluation_horizons_s=(0.1, 0.5),
    )

    assert selected == "improved"
    assert set(scored["improved"]["fold_scores"]) == {
        "profiles/a",
        "profiles/b",
        "sessions/session-1",
        "sessions/session-2",
    }


def test_scoring_rejects_concentrated_and_metric_level_regressions() -> None:
    reference = {
        "nano": _summary("multirotor", {"chirp": 1.0}),
        "plane": _summary("fixedwing", {"multisine": 1.0}),
    }
    candidates = {
        "reference": reference,
        "concentrated": {
            "nano": _summary("multirotor", {"chirp": 0.5}),
            "plane": _summary("fixedwing", {"multisine": 1.2}),
        },
        "spike": {
            "nano": _summary("multirotor", {"chirp": 0.5}),
            "plane": _summary("fixedwing", {"multisine": 2.0}),
        },
    }

    scored, ranking, selected = score_policy_candidates(
        candidates,
        reference_id="reference",
        evaluation_horizons_s=(0.1, 0.5),
        maximum_metric_regression=1.5,
        maximum_platform_regression=1.05,
    )

    assert not scored["concentrated"]["eligible"]
    assert any(
        "aggregate improvement is concentrated" in reason
        for reason in scored["concentrated"]["rejection_reasons"]
    )
    assert not scored["spike"]["eligible"]
    assert any(
        "largest metric regression" in reason
        for reason in scored["spike"]["rejection_reasons"]
    )
    assert ranking == ["reference"]
    assert selected == "reference"


def test_scoring_rejects_nonfinite_full_rollout() -> None:
    candidates = {
        "reference": {"nano": _summary("multirotor", {"chirp": 1.0})},
        "unstable": {
            "nano": _summary(
                "multirotor", {"chirp": 0.5}, full_value=float("inf")
            )
        },
    }

    scored, ranking, selected = score_policy_candidates(
        candidates,
        reference_id="reference",
        evaluation_horizons_s=(0.1, 0.5),
    )

    assert not scored["unstable"]["eligible"]
    assert scored["unstable"]["overall_score"] is None
    assert ranking == ["reference"]
    assert selected == "reference"


def test_scoring_keeps_reference_for_numerical_near_tie() -> None:
    candidates = {
        "reference": {"nano": _summary("multirotor", {"chirp": 1.0})},
        "near_tie": {"nano": _summary("multirotor", {"chirp": 0.999})},
    }

    scored, ranking, selected = score_policy_candidates(
        candidates,
        reference_id="reference",
        evaluation_horizons_s=(0.1, 0.5),
        minimum_overall_improvement=0.01,
    )

    assert not scored["near_tie"]["clears_minimum_improvement"]
    assert ranking == ["reference", "near_tie"]
    assert selected == "reference"


def test_maintained_plans_keep_the_user_facing_search_bounded() -> None:
    standard = maintained_policy_selection_plan()
    smoke = maintained_policy_selection_plan(smoke=True)

    assert standard.name == "maintained_v1"
    assert standard.steps == 400
    assert len(standard.candidates) == 5
    assert smoke.name == "smoke_v1"
    assert smoke.steps == 1
    assert len(smoke.candidates) == 2
    assert standard.reference == smoke.reference


def _write_dataset(
    root: Path,
    name: str,
    family: str,
    *,
    benchmark_split: str = "train",
) -> tuple[Path, ...]:
    paths = []
    for index, profile in enumerate(("profile_a", "profile_b")):
        states = np.zeros((6, 13), dtype=float)
        states[:, 6] = 1.0
        trajectory = Trajectory(
            time_s=np.arange(6, dtype=float) * 0.02,
            states=states,
            controls=np.zeros((5, 2), dtype=float),
            spec=make_trajectory_spec(
                ("control_a", "control_b"),
                family=family,
                observation_source="simulator_truth",
            ),
            labels={"profile": profile, "benchmark_split": benchmark_split},
        )
        path = root / f"{name}_{index}.npz"
        save_trajectory_npz(trajectory, path)
        paths.append(path)
    return tuple(paths)


def _write_source_group_dataset(
    root: Path, name: str, family: str
) -> tuple[Path, ...]:
    paths = []
    for index, group in enumerate(("session_a", "session_b")):
        states = np.zeros((6, 13), dtype=float)
        states[:, 6] = 1.0
        trajectory = Trajectory(
            time_s=np.arange(6, dtype=float) * 0.02,
            states=states,
            controls=np.zeros((5, 2), dtype=float),
            spec=make_trajectory_spec(
                ("control_a", "control_b"),
                family=family,
                observation_source="simulator_truth",
            ),
            labels={
                "profile": "shared_profile",
                "source_group": group,
                "benchmark_split": "train",
            },
        )
        path = root / f"{name}_{index}.npz"
        save_trajectory_npz(trajectory, path)
        paths.append(path)
    return tuple(paths)


def test_selection_is_resumable_and_writes_auditable_decision(
    tmp_path, monkeypatch
) -> None:
    datasets = {
        "nano": _write_dataset(tmp_path, "nano", "multirotor"),
        "plane": _write_dataset(tmp_path, "plane", "fixedwing"),
    }
    calls = []

    def fake_benchmark(paths, output_dir, **configuration):
        calls.append((tuple(paths), Path(output_dir), configuration))
        platform = "multirotor" if "nano" in str(paths[0]) else "fixedwing"
        value = 0.8 if configuration["endpoint_weight"] == 3.0 else 1.0
        summary = _summary(platform, {"profile_a": value, "profile_b": value})
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "summary.json").write_text(json.dumps(summary) + "\n")
        return summary

    monkeypatch.setattr(
        "glassbox.policy_selection.benchmark_profiles", fake_benchmark
    )
    output_dir = tmp_path / "selection"
    plan = PolicySelectionPlan(
        name="test_v1",
        candidates=(
            PolicyCandidate("structured", (0.1, 0.5), 1.0, 0.0),
            PolicyCandidate("structured", (0.1, 0.5), 3.0, 0.0),
        ),
        evaluation_horizons_s=(0.1, 0.5),
        steps=2,
    )

    decision = select_fitting_policy(datasets, output_dir, plan=plan)

    assert len(calls) == 4
    assert decision["candidate_count"] == 2
    assert decision["selected_configuration"]["endpoint_weight"] == 3.0
    assert decision["decision_scope"] == {
        "status": "provisional",
        "uses_external_test_data": False,
        "promotion_requires_external_validation": True,
    }
    assert decision["data_policy"]["dataset_platforms"] == {
        "nano": "multirotor",
        "plane": "fixedwing",
    }
    written = json.loads((output_dir / "selection.json").read_text())
    assert written["selected_candidate"] == decision["selected_candidate"]

    calls.clear()
    resumed = select_fitting_policy(datasets, output_dir, plan=plan)

    assert calls == []
    assert resumed["selected_candidate"] == decision["selected_candidate"]

    calls.clear()
    select_fitting_policy(datasets, output_dir, plan=replace(plan, steps=3))

    assert len(calls) == 4


def test_selection_rejects_benchmark_test_trajectories(tmp_path, monkeypatch) -> None:
    datasets = {
        "nano_test": _write_dataset(
            tmp_path, "nano_test", "multirotor", benchmark_split="test"
        )
    }
    monkeypatch.setattr(
        "glassbox.policy_selection.benchmark_profiles",
        lambda *args, **kwargs: pytest.fail("fit must not run"),
    )

    with pytest.raises(ValueError, match="non-training benchmark split.*test"):
        select_fitting_policy(
            datasets,
            tmp_path / "selection",
            plan=PolicySelectionPlan(
                name="test_guard",
                candidates=(PolicyCandidate("structured", (0.1,), 1.0, 0.0),),
                evaluation_horizons_s=(0.1,),
                steps=1,
            ),
        )


def test_selection_dispatches_single_profile_corpus_to_source_groups(
    tmp_path, monkeypatch
) -> None:
    datasets = {
        "idf": _write_source_group_dataset(tmp_path, "idf", "fixedwing")
    }
    calls = []

    def fake_source_benchmark(paths, output_dir, **configuration):
        calls.append((tuple(paths), Path(output_dir), configuration))
        summary = _source_summary(
            "fixedwing", {"session_a": 1.0, "session_b": 1.0}
        )
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "summary.json").write_text(json.dumps(summary) + "\n")
        return summary

    monkeypatch.setattr(
        "glassbox.policy_selection.benchmark_source_groups",
        fake_source_benchmark,
    )
    monkeypatch.setattr(
        "glassbox.policy_selection.benchmark_profiles",
        lambda *args, **kwargs: pytest.fail("profile benchmark must not run"),
    )
    plan = PolicySelectionPlan(
        name="source_groups_v1",
        candidates=(PolicyCandidate("structured", (0.1,), 1.0, 0.0),),
        evaluation_horizons_s=(0.1, 0.5),
        steps=1,
    )

    decision = select_fitting_policy(
        datasets, tmp_path / "source_selection", plan=plan
    )

    assert len(calls) == 1
    assert decision["data_policy"]["dataset_fold_axes"] == {
        "idf": "source_group"
    }
