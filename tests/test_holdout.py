import json
from dataclasses import replace
from shutil import copytree

import pytest

from glassbox.core.data import save_trajectory_npz
from glassbox.fitting import FitSpec, Holdout, WeightingPolicy
from glassbox.workflows.holdout import evaluate_holdout


def test_source_group_holdout_moves_every_segment_into_the_same_fold(
    tmp_path, monkeypatch, fixedwing_flight
) -> None:
    paths = []
    groups = ("session-a", "session-a", "session-b", "session-c")
    for seed, group in enumerate(groups):
        trajectory = fixedwing_flight(seed, 0.3)
        trajectory = replace(
            trajectory,
            labels={**trajectory.labels, "source_group": group},
        )
        path = tmp_path / f"segment_{seed}.npz"
        save_trajectory_npz(trajectory, path)
        paths.append(path)

    summary = evaluate_holdout(
        paths,
        hold_out="source_group",
        spec=FitSpec(horizons_s=(0.1,), evaluation_horizons_s=(0.1,), steps=1),
        output_dir=tmp_path / "benchmark",
    )

    assert summary["evaluation"] == "leave_one_source_group_out"
    assert summary["holdout_label"] == "source_group"
    assert summary["protocol"] == "windowed"
    assert summary["fold_count"] == 3
    assert set(summary["per_fold"]) == {
        "session-a",
        "session-b",
        "session-c",
    }
    assert summary["per_fold"]["session-a"]["validation_trajectory_count"] == 2
    assert summary["aggregate"]["weighting"] == "equal_source_group"
    assert summary["configuration"]["exogenous_size"] == 0
    assert summary["configuration"]["exogenous_names"] == []
    assert summary["configuration"]["exogenous_roles"] == []
    assert summary["configuration"]["learn_thrust_command_offset"] is False
    assert summary["configuration"]["diagonal_angular_control"] is True
    assert summary["configuration"]["holdout_label"] == "source_group"
    assert "0.1s" in summary["aggregate"]["baseline_horizon_rollouts"]
    assert set(summary["aggregate"]["model_over_baseline"]["0.1s"]) == {
        "position_rmse_m",
        "velocity_rmse_m_s",
        "attitude_rmse_deg",
        "angular_velocity_rmse_rad_s",
    }
    position_distribution = summary["distribution"]["horizon_rollouts"]["0.1s"][
        "position_rmse_m"
    ]
    assert position_distribution["minimum"] <= position_distribution["median"]
    assert position_distribution["median"] <= position_distribution["p90"]
    assert position_distribution["p90"] <= position_distribution["maximum"]
    for fold in summary["per_fold"].values():
        assert fold["training_window_selection"]["budget_policy"] == (
            "one_window_budget_v1"
        )
    summary_path = tmp_path / "benchmark" / "summary.json"
    assert summary_path.exists()

    summary_path.unlink()
    monkeypatch.setattr(
        "glassbox.workflows.holdout.fit",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("completed folds must be resumed")
        ),
    )
    resumed = evaluate_holdout(
        paths,
        hold_out="source_group",
        spec=FitSpec(horizons_s=(0.1,), evaluation_horizons_s=(0.1,), steps=1),
        output_dir=tmp_path / "benchmark",
    )

    assert resumed == summary
    assert summary_path.exists()


def test_a_fold_limit_shortens_the_run_and_records_that_it_did(
    tmp_path, fixedwing_flight
) -> None:
    paths = []
    for seed, group in enumerate(("session-a", "session-b", "session-c")):
        trajectory = fixedwing_flight(seed, 0.3)
        trajectory = replace(
            trajectory, labels={**trajectory.labels, "source_group": group}
        )
        path = tmp_path / f"segment_{seed}.npz"
        save_trajectory_npz(trajectory, path)
        paths.append(path)
    spec = FitSpec(horizons_s=(0.1,), evaluation_horizons_s=(0.1,), steps=1)

    summary = evaluate_holdout(
        paths,
        hold_out="source_group",
        spec=spec,
        output_dir=tmp_path / "shortened",
        fold_limit=2,
    )

    assert summary["fold_count"] == 2
    assert summary["folds"] == ["session-a", "session-b"]
    assert set(summary["per_fold"]) == {"session-a", "session-b"}
    # The shortened selection is recorded, and it is part of the request a
    # resume must match, so a shortened run cannot stand in for a whole one.
    assert (
        summary["configuration"]["fold_selection"] == "first_2_of_3_source_group_values"
    )
    request = json.loads((tmp_path / "shortened" / "request.json").read_text())
    assert request["folds"] == ["session-a", "session-b"]


def test_a_fold_limit_below_two_is_refused(tmp_path, fixedwing_flight) -> None:
    paths = []
    for seed, group in enumerate(("session-a", "session-b")):
        trajectory = fixedwing_flight(seed, 0.3)
        trajectory = replace(
            trajectory, labels={**trajectory.labels, "source_group": group}
        )
        path = tmp_path / f"segment_{seed}.npz"
        save_trajectory_npz(trajectory, path)
        paths.append(path)

    with pytest.raises(ValueError, match="at least two folds"):
        evaluate_holdout(
            paths,
            hold_out="source_group",
            spec=FitSpec(horizons_s=(0.1,), evaluation_horizons_s=(0.1,), steps=1),
            output_dir=tmp_path / "too-short",
            fold_limit=1,
        )


@pytest.fixture(scope="module")
def completed_holdout(tmp_path_factory, quadrotor_flight):
    root = tmp_path_factory.mktemp("completed-holdout")
    paths = []
    for seed, group in enumerate(("session-a", "session-b")):
        flight = replace(quadrotor_flight(seed, 0.2), labels={"source_group": group})
        path = root / f"{group}.npz"
        save_trajectory_npz(flight, path)
        paths.append(path)
    spec = FitSpec(
        horizons_s=(0.04,),
        evaluation_horizons_s=(0.04,),
        steps=1,
        parameter_evidence=False,
    )
    output = root / "results"
    summary = evaluate_holdout(paths, spec=spec, output_dir=output)
    return paths, spec, output, summary


@pytest.mark.parametrize("complete", [True, False], ids=["summary", "folds"])
@pytest.mark.parametrize(
    "changes",
    [
        {"stride": 1},
        {"weighting": WeightingPolicy(balanced=False)},
        {"weighting": WeightingPolicy(group_weights={"session-a": 2.0})},
        {"parameter_evidence": True},
        {"fixed_response_time_constant_s": 0.2},
        {"diagnostics": True},
    ],
    ids=["stride", "balancing", "group-weights", "evidence", "lag", "diagnostics"],
)
def test_changed_fit_settings_invalidate_resumed_results(
    completed_holdout, tmp_path, monkeypatch, complete, changes
) -> None:
    paths, spec, original, _ = completed_holdout
    output = copytree(original, tmp_path / "results")
    if not complete:
        (output / "summary.json").unlink()

    def require_refit(_paths, requested):
        for name, value in changes.items():
            assert getattr(requested, name) == value
        raise RuntimeError("refit requested")

    monkeypatch.setattr("glassbox.workflows.holdout.fit", require_refit)
    with pytest.raises(RuntimeError, match="refit requested"):
        evaluate_holdout(paths, spec=replace(spec, **changes), output_dir=output)


def test_resuming_ignores_the_holdout_rule_that_each_fold_overrides(
    completed_holdout, tmp_path, monkeypatch
) -> None:
    paths, spec, original, summary = completed_holdout
    output = copytree(original, tmp_path / "results")

    def unexpected_refit(*_args, **_kwargs):
        pytest.fail("the fold supplies its own holdout rule")

    monkeypatch.setattr("glassbox.workflows.holdout.fit", unexpected_refit)
    assert (
        evaluate_holdout(
            paths, spec=replace(spec, holdout=Holdout.temporal()), output_dir=output
        )
        == summary
    )


@pytest.mark.parametrize("force", [False, True], ids=["changed-request", "forced"])
def test_interrupted_reruns_do_not_reuse_old_completion_records(
    completed_holdout, tmp_path, monkeypatch, force
) -> None:
    paths, spec, original, _ = completed_holdout
    output = copytree(original, tmp_path / "results")
    requested = spec if force else replace(spec, steps=2)

    def interrupt(*_args, **_kwargs):
        raise RuntimeError("interrupted fit")

    monkeypatch.setattr("glassbox.workflows.holdout.fit", interrupt)
    with pytest.raises(RuntimeError, match="interrupted fit"):
        evaluate_holdout(paths, spec=requested, output_dir=output, resume=not force)
    assert not (output / "summary.json").exists()
    assert not (output / "fold_01_session-a_request.json").exists()

    # A retry must enter the unfinished fold, even when it has the same request
    # and stale model/report files as the previously completed run.
    with pytest.raises(RuntimeError, match="interrupted fit"):
        evaluate_holdout(paths, spec=requested, output_dir=output)
