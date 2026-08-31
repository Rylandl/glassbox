import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from glassbox.data import save_trajectory_npz, trajectory_windows
from glassbox.fit_cli import _fit_on_windows
from glassbox.fixedwing_synthetic import generate_fixed_wing_trajectory
from glassbox.identification import FitResult
from glassbox.predictive_ensemble import (
    PredictiveEnsemble,
    _balanced_calibration_groups,
    aggregate_predictive_ensemble_metrics,
    benchmark_predictive_ensemble,
    fit_grouped_disagreement_calibration,
    grouped_bootstrap_multiplicities,
    predictive_ensemble_metrics,
    predictive_uncertainty_candidate_gate,
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


def test_balanced_calibration_partition_preserves_profile_condition_replicates() -> (
    None
):
    groups = tuple(
        f"{profile}-{condition}-{replicate}"
        for profile in ("a", "b", "c")
        for condition in ("low", "medium", "high")
        for replicate in range(2)
    )
    profiles = {group: group.split("-")[0] for group in groups}
    conditions = {group: group.split("-")[1] for group in groups}

    selected = _balanced_calibration_groups(
        groups,
        profiles=profiles,
        conditions=conditions,
        fold_index=1,
    )

    assert len(selected) == 9
    for profile in ("a", "b", "c"):
        for condition in ("low", "medium", "high"):
            stratum = {
                group
                for group in groups
                if profiles[group] == profile and conditions[group] == condition
            }
            assert len(stratum & set(selected)) == 1
            assert len(stratum - set(selected)) == 1
    assert selected == _balanced_calibration_groups(
        groups,
        profiles=profiles,
        conditions=conditions,
        fold_index=1,
    )


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

    omitted = trajectory_windows(
        trajectories,
        horizon=5,
        stride=5,
        trajectory_groups=("a", "a", "b"),
        trajectory_group_weights={"a": 1.0, "b": 0.0},
    )
    assert omitted.window_weights is not None
    assert np.all(omitted.trajectory_indices != 2)
    assert np.all(omitted.window_weights > 0.0)
    assert omitted.candidate_window_counts[2] == 0
    assert omitted.selection_policy == "all_positive_weight_candidates"


def test_shared_outer_statistics_fix_residual_coordinates_across_members(
    monkeypatch,
) -> None:
    trajectories = [
        generate_trajectory(seed=0, duration_s=0.4),
        generate_trajectory(seed=1, duration_s=0.4),
    ]

    def window_sets(weights):
        return tuple(
            trajectory_windows(
                trajectories,
                horizon=horizon,
                stride=horizon,
                trajectory_groups=("a", "b"),
                trajectory_group_weights=weights,
            )
            for horizon in (5, 10)
        )

    normalization = window_sets({"a": 1.0, "b": 1.0})
    captured = []

    def fake_fit(windows, initial_params, **kwargs):
        captured.append((initial_params, kwargs))
        return FitResult(
            params=initial_params,
            loss_history=np.asarray((1.0, 1.0)),
            component_initial_losses=np.asarray((1.0, 1.0)),
            component_final_losses=np.asarray((1.0, 1.0)),
            component_loss_normalizers=np.asarray((2.0, 3.0)),
            loss_configuration=kwargs["loss_configuration"],
        )

    monkeypatch.setattr("glassbox.fit_cli.fit_dynamics_multi_horizon", fake_fit)
    reports = []
    for weights in ({"a": 1.0, "b": 0.0}, {"a": 0.0, "b": 1.0}):
        _, report = _fit_on_windows(
            window_sets(weights),
            steps=1,
            learning_rate=0.01,
            model_class="structured_residual",
            normalization_windows=normalization,
        )
        reports.append(report)

    first_params, first_kwargs = captured[0]
    second_params, second_kwargs = captured[1]
    np.testing.assert_array_equal(first_params.feature_mean, second_params.feature_mean)
    np.testing.assert_array_equal(
        first_params.feature_scale, second_params.feature_scale
    )
    np.testing.assert_array_equal(
        first_params.correction_scale, second_params.correction_scale
    )
    assert (
        first_kwargs["loss_configuration"].to_dict()
        == second_kwargs["loss_configuration"].to_dict()
    )
    assert first_kwargs["loss_normalization_window_sets"] is normalization
    assert second_kwargs["loss_normalization_window_sets"] is normalization
    assert reports[0]["fit"]["statistics_source"] == ("shared_outer_training_windows")
    assert reports[0]["fit"]["rollout_loss"] == reports[1]["fit"]["rollout_loss"]


def test_ensemble_rejects_source_groups_with_colliding_json_keys(
    tmp_path,
) -> None:
    paths = []
    for seed, source_group in enumerate((1, "1", 2, 3)):
        trajectory = generate_trajectory(seed=seed, duration_s=0.2)
        trajectory = replace(
            trajectory,
            labels={
                **trajectory.labels,
                "source_group": source_group,
            },
        )
        path = tmp_path / f"flight_{seed}.npz"
        save_trajectory_npz(trajectory, path)
        paths.append(path)

    with pytest.raises(ValueError, match="unique string representations"):
        benchmark_predictive_ensemble(paths, tmp_path / "ensemble")


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
    assert report["groups"]["position"]["calibration"]["0.9"][
        "mean_attained_finite_member_mass"
    ] == pytest.approx(1.0)


def test_predictive_metrics_reject_every_component_of_a_nonfinite_member(
    monkeypatch,
) -> None:
    trajectory = generate_trajectory(seed=0, duration_s=0.1)
    target = np.zeros((3, 2, 13), dtype=np.float64)
    target[..., 6] = 1.0
    valid = target.copy()
    partial = target.copy()
    partial[:, -1, 0] = 10.0
    partial[:, -1, 10] = np.nan
    predictions = iter((valid, partial))

    def fake_predictions(*_args, **_kwargs):
        return next(predictions), target, 0.02

    monkeypatch.setattr(
        "glassbox.predictive_ensemble.windowed_rollout_predictions",
        fake_predictions,
    )
    ensemble = PredictiveEnsemble(
        members=(true_parameters(), initial_parameter_guess()),
        member_ids=("valid", "partial"),
    )

    report = predictive_ensemble_metrics(
        ensemble,
        trajectory,
        horizon_steps=1,
    )

    assert report["finite_member_prediction_fraction"] == pytest.approx(0.5)
    assert report["groups"]["position"]["center_vector_rmse"] == pytest.approx(0.0)
    assert report["groups"]["position"]["mean_disagreement_radius"] == (
        pytest.approx(0.0)
    )


def test_grouped_scale_calibration_respects_independent_group_resolution() -> None:
    trajectories = []
    for seed in range(2):
        trajectory = generate_trajectory(seed=seed, duration_s=0.2)
        trajectories.append(
            replace(
                trajectory,
                labels={
                    **trajectory.labels,
                    "source_group": f"calibration-{seed}",
                },
            )
        )
    ensemble = PredictiveEnsemble(
        members=(true_parameters(), initial_parameter_guess()),
        member_ids=("first", "second"),
    )

    calibration = fit_grouped_disagreement_calibration(
        ensemble,
        trajectories,
        horizon_steps=2,
    )

    assert calibration["calibration_source_group_count"] == 2
    assert calibration["groups"]["position"]["0.5"]["status"] == "available"
    assert calibration["groups"]["position"]["0.5"]["scale"] is not None
    assert (
        calibration["groups"]["position"]["0.5"]["constant_error_radius_status"]
        == "available"
    )
    assert calibration["groups"]["position"]["0.5"]["constant_error_radius"] is not None
    for level in ("0.8", "0.9"):
        setting = calibration["groups"]["position"][level]
        assert setting["status"] == "unavailable"
        assert setting["reason"] == "insufficient_independent_calibration_groups"
        assert setting["constant_error_radius_status"] == "unavailable"

    report = predictive_ensemble_metrics(
        ensemble,
        trajectories[0],
        horizon_steps=2,
        disagreement_calibration=calibration,
    )
    assert (
        report["groups"]["position"]["scaled_calibration"]["0.5"]["status"]
        == "available"
    )
    assert (
        report["groups"]["position"]["scaled_calibration"]["0.9"]["status"]
        == "unavailable"
    )
    assert (
        report["groups"]["position"]["constant_radius_baseline"]["0.5"]["status"]
        == "available"
    )
    assert (
        report["groups"]["position"]["scaled_calibration"]["0.5"]["mean_set_score"]
        is not None
    )
    aggregate = aggregate_predictive_ensemble_metrics((report, report))
    assert (
        aggregate["groups"]["position"]["constant_radius_baseline"]["0.5"][
            "mean_set_score"
        ]
        is not None
    )
    assert (
        aggregate["groups"]["position"]["scaled_calibration"]["0.5"][
            "set_score_skill_vs_constant"
        ]
        is not None
    )
    assert report["uncertainty_semantics"]["posterior"] is False


def test_uncertainty_candidate_gate_requires_calibration_skill_and_finiteness() -> None:
    groups = {}
    for name in ("position", "velocity", "attitude", "angular_velocity"):
        groups[name] = {
            "error_disagreement_spearman": 0.5,
            "scaled_calibration": {
                level: {
                    "status": "available",
                    "empirical_coverage": float(level),
                    "set_score_skill_vs_constant": 0.2,
                }
                for level in ("0.5", "0.8")
            },
            "constant_radius_baseline": {
                level: {"status": "available"} for level in ("0.5", "0.8")
            },
        }
    horizon = {
        "finite_member_prediction_fraction": 1.0,
        "finite_member_path_fraction": 1.0,
        "fully_finite_member_fraction": 1.0,
        "minimum_unique_parameter_member_count": 4,
        "minimum_unique_prediction_member_count": 4,
        "groups": groups,
    }

    passing = predictive_uncertainty_candidate_gate({"1s": horizon})
    assert passing["passed"] is True
    assert passing["runtime_promotion"] is False

    failing_horizon = {**horizon, "finite_member_path_fraction": 0.99}
    failing = predictive_uncertainty_candidate_gate({"1s": failing_horizon})
    assert failing["passed"] is False
    assert failing["checks"]["member_finiteness"]["passed"] is False


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
    monkeypatch,
) -> None:
    paths = []
    source_group_by_path = {}
    seed = 0
    for profile in ("vertical", "lateral", "yaw"):
        for replicate in range(4):
            trajectory = generate_fixed_wing_trajectory(seed=seed, duration_s=0.2)
            trajectory = replace(
                trajectory,
                labels={
                    **trajectory.labels,
                    "profile": profile,
                    "source_group": f"{profile}-{replicate}",
                },
            )
            path = tmp_path / f"flight_{seed}.npz"
            save_trajectory_npz(trajectory, path)
            paths.append(path)
            source_group_by_path[str(path.resolve())] = f"{profile}-{replicate}"
            seed += 1

    summary = benchmark_predictive_ensemble(
        paths,
        tmp_path / "ensemble",
        training_horizons_s=(0.1, 0.2),
        evaluation_horizons_s=(0.1,),
        steps=1,
        member_count=2,
        model_class="structured",
    )

    assert summary["outer_axis"] == "profile"
    assert summary["outer_fold_count"] == 3
    assert summary["evaluation"] == "nested_group_calibrated_predictive_ensemble"
    assert summary["promotion"]["status"] == "diagnostic_only"
    assert summary["promotion"]["candidate_gate"]["gate_version"] == (
        "predictive_uncertainty_gate_v1"
    )
    assert summary["promotion"]["candidate_gate"]["runtime_promotion"] is False
    assert summary["uncertainty_semantics"]["posterior"] is False
    assert len(summary["implementation"]["source_tree_sha256"]) == 64
    observed_distinct_resample = False
    observed_omitted_group = False
    for fold in summary["per_fold"].values():
        assert fold["member_count"] == 2
        assert len(fold["validation_source_groups"]) == 4
        assert len(fold["calibration_source_groups"]) == 4
        assert len(fold["training_source_groups"]) == 4
        assert set(fold["validation_source_groups"]).isdisjoint(
            fold["training_source_groups"]
        )
        assert set(fold["calibration_source_groups"]).isdisjoint(
            fold["training_source_groups"]
        )
        assert set(fold["calibration_source_groups"]).isdisjoint(
            fold["validation_source_groups"]
        )
        assert {group.split("-")[0] for group in fold["calibration_source_groups"]} == {
            group.split("-")[0] for group in fold["training_source_groups"]
        }
        manifest = Path(fold["ensemble"])
        assert manifest.exists()
        payload = json.loads(manifest.read_text())
        assert payload["shared_fit_statistics"]["policy"] == (
            "complete_member_fit_partition_v1"
        )
        assert payload["member_fit_execution"] == {
            "policy": "automatic_threaded_max_2",
            "worker_count": 2,
        }
        assert len(payload["disagreement_calibration_artifact"]["sha256"]) == 64
        calibration = json.loads(Path(fold["disagreement_calibration"]).read_text())
        assert set(
            calibration["horizon_rollouts"]["0.1s"]["calibration_source_groups"]
        ) == set(fold["calibration_source_groups"])
        observed_distinct_resample |= payload["unique_resample_count"] > 1
        reports = []
        for member in payload["members"]:
            weights = member["training_source_group_loss_weights"]
            training_profiles = {
                group.split("-")[0] for group in fold["training_source_groups"]
            }
            for profile in training_profiles:
                assert sum(
                    weight
                    for group, weight in weights.items()
                    if group.startswith(f"{profile}-")
                ) == pytest.approx(1.0)
            assert len(member["model_artifact"]["sha256"]) == 64
            assert len(member["fit_report_artifact"]["sha256"]) == 64
            report = json.loads(Path(member["report"]).read_text())
            assert set(report["split"]["training_source_groups"]) == set(
                fold["training_source_groups"]
            )
            assert set(report["split"]["validation_source_groups"]) == set(
                fold["calibration_source_groups"]
            )
            for horizon_counts in report["configuration"][
                "candidate_training_windows_per_flight_by_horizon"
            ].values():
                for path, count in horizon_counts.items():
                    source_group = source_group_by_path[path]
                    if weights[source_group] == 0.0:
                        observed_omitted_group = True
                        assert count == 0
                    else:
                        assert count > 0
            for horizon_counts in report["configuration"][
                "training_windows_per_flight_by_horizon"
            ].values():
                for path, count in horizon_counts.items():
                    source_group = source_group_by_path[path]
                    assert (count > 0) is (weights[source_group] > 0.0)
            reports.append(report)
        assert all(
            report["configuration"]["fit_statistics"]["policy"]
            == "shared_outer_training_windows_v1"
            for report in reports
        )
        assert (
            reports[0]["models"]["learned_lag"]["fit"]["rollout_loss"]
            == reports[1]["models"]["learned_lag"]["fit"]["rollout_loss"]
        )
        assert (
            reports[0]["models"]["learned_lag"]["fit"]["multi_horizon_loss_normalizers"]
            == reports[1]["models"]["learned_lag"]["fit"][
                "multi_horizon_loss_normalizers"
            ]
        )
    assert observed_distinct_resample is True
    assert observed_omitted_group is True
    assert (tmp_path / "ensemble" / "summary.json").exists()

    monkeypatch.setattr(
        "glassbox.predictive_ensemble.fit_trajectory_artifacts",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("an identical fingerprint must resume")
        ),
    )
    resumed = benchmark_predictive_ensemble(
        paths,
        tmp_path / "ensemble",
        training_horizons_s=(0.1, 0.2),
        evaluation_horizons_s=(0.1,),
        steps=1,
        member_count=2,
        model_class="structured",
    )
    assert resumed == summary

    first_manifest = json.loads(
        Path(next(iter(summary["per_fold"].values()))["ensemble"]).read_text()
    )
    first_model = Path(first_manifest["members"][0]["model"])
    first_model.write_text(first_model.read_text() + " ")
    with pytest.raises(AssertionError, match="identical fingerprint"):
        benchmark_predictive_ensemble(
            paths,
            tmp_path / "ensemble",
            training_horizons_s=(0.1, 0.2),
            evaluation_horizons_s=(0.1,),
            steps=1,
            member_count=2,
            model_class="structured",
        )
