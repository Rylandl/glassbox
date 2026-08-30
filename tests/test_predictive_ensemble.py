from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from glassbox.data import save_trajectory_npz, trajectory_windows
from glassbox.fixedwing_synthetic import generate_fixed_wing_trajectory
from glassbox.predictive_ensemble import (
    PredictiveEnsemble,
    aggregate_predictive_ensemble_metrics,
    benchmark_predictive_ensemble,
    grouped_bootstrap_multiplicities,
    predictive_ensemble_metrics,
)
from glassbox.synthetic import (
    generate_trajectory,
    initial_parameter_guess,
    true_parameters,
)


def test_grouped_bootstrap_is_deterministic_and_preserves_stratum_draw_counts() -> None:
    groups = ("a-1", "a-2", "b-1", "b-2", "b-3")
    strata = {group: group[0] for group in groups}

    first = grouped_bootstrap_multiplicities(
        groups, strata=strata, member_count=6, seed=12
    )
    second = grouped_bootstrap_multiplicities(
        groups, strata=strata, member_count=6, seed=12
    )

    assert first == second
    for member in first:
        assert sum(count for group, count in member.items() if group[0] == "a") == 2
        assert sum(count for group, count in member.items() if group[0] == "b") == 3
        assert set(member) <= set(groups)


def test_group_weighted_windows_preserve_complete_group_multiplicity() -> None:
    trajectories = [
        generate_trajectory(seed=0, duration_s=0.4),
        generate_trajectory(seed=1, duration_s=0.4),
        generate_trajectory(seed=2, duration_s=0.4),
    ]
    windows = trajectory_windows(
        trajectories,
        horizon=5,
        stride=5,
        trajectory_groups=("a", "a", "b"),
        trajectory_group_weights={"a": 2.0, "b": 1.0},
        maximum_windows=9,
    )
    assert windows.window_weights is not None
    group_a = np.isin(windows.trajectory_indices, (0, 1))
    group_b = windows.trajectory_indices == 2
    weight_a = float(np.sum(windows.window_weights[group_a]))
    weight_b = float(np.sum(windows.window_weights[group_b]))

    assert weight_a / (weight_a + weight_b) == pytest.approx(2.0 / 3.0)
    assert weight_b / (weight_a + weight_b) == pytest.approx(1.0 / 3.0)


def test_predictive_metrics_use_quaternion_geometry_and_exclude_initial_state(
    monkeypatch,
) -> None:
    trajectory = generate_trajectory(seed=0, duration_s=0.1)
    target = np.zeros((3, 4, 13), dtype=np.float64)
    target[..., 6] = 1.0
    target[:, -1, 0] = (1.0, 2.0, 3.0)
    first = target.copy()
    second = target.copy()
    first[:, -1, 0] = (0.5, 1.0, 2.0)
    second[:, -1, 0] = (1.5, 5.0, 8.0)
    second[..., 6] = -1.0  # Same rotations under the quaternion antipode.
    predictions = iter((first, second))

    def fake_predictions(*_args, **_kwargs):
        return next(predictions), target, 0.02

    monkeypatch.setattr(
        "glassbox.predictive_ensemble.windowed_rollout_predictions",
        fake_predictions,
    )
    ensemble = PredictiveEnsemble(
        members=(true_parameters(), initial_parameter_guess()),
        member_ids=("first", "second"),
    )

    report = predictive_ensemble_metrics(
        ensemble,
        trajectory,
        horizon_steps=3,
    )

    assert report["prediction_count"] == 3
    assert report["path_prediction_count"] == 9
    assert report["prediction_target"] == "rollout_endpoint"
    assert report["initial_measured_state_excluded"] is True
    assert report["groups"]["position"]["error_disagreement_spearman"] == (
        pytest.approx(1.0)
    )
    assert report["groups"]["attitude"]["center_vector_rmse"] == pytest.approx(0.0)
    assert report["groups"]["attitude"]["mean_disagreement_radius"] == (
        pytest.approx(0.0)
    )
    assert report["uncertainty_semantics"]["posterior"] is False


def test_aggregate_predictive_metrics_uses_equal_item_weighting(monkeypatch) -> None:
    trajectory = generate_trajectory(seed=0, duration_s=0.1)
    ensemble = PredictiveEnsemble(
        members=(true_parameters(), initial_parameter_guess()),
        member_ids=("first", "second"),
    )
    report = predictive_ensemble_metrics(ensemble, trajectory, horizon_steps=2)

    aggregate = aggregate_predictive_ensemble_metrics((report, report))

    assert aggregate["weighting"] == "equal_item"
    assert aggregate["item_count"] == 2
    assert aggregate["groups"]["position"]["energy_score"] == pytest.approx(
        report["groups"]["position"]["energy_score"]
    )


def test_nested_ensemble_benchmark_keeps_outer_profiles_out_of_every_member(
    tmp_path,
) -> None:
    paths = []
    for seed, profile in enumerate(("vertical", "lateral", "yaw")):
        trajectory = generate_fixed_wing_trajectory(seed=seed, duration_s=0.3)
        trajectory = replace(
            trajectory,
            labels={
                **trajectory.labels,
                "profile": profile,
                "source_group": f"session-{seed}",
            },
        )
        path = tmp_path / f"flight_{seed}.npz"
        save_trajectory_npz(trajectory, path)
        paths.append(path)

    summary = benchmark_predictive_ensemble(
        paths,
        tmp_path / "ensemble",
        training_horizons_s=(0.1,),
        evaluation_horizons_s=(0.1,),
        steps=1,
        member_count=2,
    )

    assert summary["outer_axis"] == "profile"
    assert summary["outer_fold_count"] == 3
    assert summary["promotion"]["status"] == "diagnostic_only"
    assert summary["uncertainty_semantics"]["posterior"] is False
    for fold in summary["per_fold"].values():
        assert fold["member_count"] == 2
        assert len(fold["validation_source_groups"]) == 1
        assert set(fold["validation_source_groups"]).isdisjoint(
            fold["training_source_groups"]
        )
        manifest = Path(fold["ensemble"])
        assert manifest.exists()
    assert (tmp_path / "ensemble" / "summary.json").exists()
