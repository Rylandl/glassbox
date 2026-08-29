from dataclasses import replace

from glassbox.data import save_trajectory_npz
from glassbox.fixedwing_synthetic import generate_fixed_wing_trajectory
from glassbox.source_group_benchmark import benchmark_source_groups


def test_source_group_benchmark_moves_every_segment_into_the_same_fold(
    tmp_path, monkeypatch
) -> None:
    paths = []
    groups = ("session-a", "session-a", "session-b", "session-c")
    for seed, group in enumerate(groups):
        trajectory = generate_fixed_wing_trajectory(seed=seed, duration_s=0.3)
        trajectory = replace(
            trajectory,
            labels={**trajectory.labels, "source_group": group},
        )
        path = tmp_path / f"segment_{seed}.npz"
        save_trajectory_npz(trajectory, path)
        paths.append(path)

    summary = benchmark_source_groups(
        paths,
        tmp_path / "benchmark",
        training_horizons_s=(0.1,),
        evaluation_horizons_s=(0.1,),
        steps=1,
    )

    assert summary["evaluation"] == "leave_one_source_group_out"
    assert summary["source_group_count"] == 3
    assert set(summary["per_source_group"]) == {
        "session-a",
        "session-b",
        "session-c",
    }
    assert summary["per_source_group"]["session-a"][
        "validation_trajectory_count"
    ] == 2
    assert summary["aggregate"]["weighting"] == "equal_source_group"
    assert summary["configuration"]["exogenous_size"] == 0
    assert summary["configuration"]["exogenous_names"] == []
    assert summary["configuration"]["exogenous_roles"] == []
    assert summary["configuration"]["multirotor_thrust_command_offset"] == (
        "not_applicable_fixedwing"
    )
    assert "0.1s" in summary["aggregate"][
        "kinematic_persistence_horizon_rollouts"
    ]
    assert set(
        summary["aggregate"]["model_over_kinematic_persistence"]["0.1s"]
    ) == {
        "position_rmse_m",
        "velocity_rmse_m_s",
        "attitude_rmse_deg",
        "angular_velocity_rmse_rad_s",
    }
    position_distribution = summary["distribution"]["horizon_rollouts"][
        "0.1s"
    ]["position_rmse_m"]
    assert position_distribution["minimum"] <= position_distribution["median"]
    assert position_distribution["median"] <= position_distribution["p90"]
    assert position_distribution["p90"] <= position_distribution["maximum"]
    for fold in summary["per_source_group"].values():
        assert fold["training_window_selection"]["budget_policy"] == (
            "automatic_corpus_and_horizon"
        )
    summary_path = tmp_path / "benchmark" / "summary.json"
    assert summary_path.exists()

    summary_path.unlink()
    monkeypatch.setattr(
        "glassbox.source_group_benchmark.fit_trajectory_artifacts",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("completed folds must be resumed")
        ),
    )
    resumed = benchmark_source_groups(
        paths,
        tmp_path / "benchmark",
        training_horizons_s=(0.1,),
        evaluation_horizons_s=(0.1,),
        steps=1,
    )

    assert resumed == summary
    assert summary_path.exists()
