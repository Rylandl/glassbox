import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from glassbox import fit_cli
from glassbox.data import save_trajectory_npz
from glassbox.evaluation import (
    kinematic_persistence_windowed_metrics,
    rollout_divergence_metrics,
    windowed_rollout_metrics,
)
from glassbox.fit_cli import (
    BenchmarkSplitHoldoutConflict,
    _automatic_training_window_budget,
    _dataset_contract,
    fit_trajectory_artifacts,
)
from glassbox.fixedwing_synthetic import (
    generate_fixed_wing_trajectory,
    true_fixed_wing_parameters,
)
from glassbox.profile_benchmark import benchmark_profiles
from glassbox.synthetic import generate_trajectory, true_parameters


def _px4_provenance(*, motor_index: int, surface_indices: list[int]) -> dict:
    return {
        "source": f"fixture_{motor_index}.ulg",
        "adapter": {"name": "px4_ulog", "schema_version": 1},
        "px4": {
            "topics": {
                "motor_actuator": f"motors_{motor_index}",
                "servo_actuator": f"servos_{motor_index}",
            },
            "actuator_mapping": {
                "motor_index": motor_index,
                "surface_indices": surface_indices,
                "canonical_surface_mixing_matrix": [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                "actuator_mapping_verified": True,
            },
        },
    }


def test_dataset_pooling_uses_canonical_semantics_not_px4_source_layout() -> None:
    base = generate_fixed_wing_trajectory(seed=0, duration_s=0.2)
    first = replace(
        base,
        provenance=_px4_provenance(motor_index=0, surface_indices=[0, 1, 2]),
    )
    second = replace(
        base,
        provenance=_px4_provenance(motor_index=4, surface_indices=[5, 7, 9]),
    )

    contract = _dataset_contract(
        [Path("first.npz"), Path("second.npz")],
        [first, second],
    )

    assert contract["pooling_basis"] == "canonical_trajectory_spec"
    assert contract["source_type_counts"] == {"px4_ulog": 2}
    assert contract["exogenous_size"] == 0
    assert contract["exogenous_names"] == []
    assert contract["exogenous_roles"] == []
    assert "surface_indices" not in contract
    assert "actuator_topics" not in contract


def test_dataset_pooling_rejects_different_vehicle_configuration_ids() -> None:
    first = generate_fixed_wing_trajectory(seed=0, duration_s=0.2)
    assert first.spec is not None
    first = replace(
        first,
        spec=replace(
            first.spec,
            vehicle=replace(first.spec.vehicle, configuration_id="plane-a"),
        ),
    )
    second = replace(
        first,
        spec=replace(
            first.spec,
            vehicle=replace(first.spec.vehicle, configuration_id="plane-b"),
        ),
    )

    with pytest.raises(ValueError, match="inconsistent dataset trajectory_spec"):
        _dataset_contract(
            [Path("plane-a.npz"), Path("plane-b.npz")],
            [first, second],
        )


def test_windowed_metrics_cover_multiple_initial_conditions() -> None:
    trajectory = generate_trajectory(seed=9, duration_s=0.4)

    metrics = windowed_rollout_metrics(
        true_parameters(),
        trajectory,
        horizon_steps=5,
    )

    assert metrics["rollout_count"] == 4
    # Four windows of five predicted steps; the shared initial sample is excluded.
    assert metrics["sample_count"] == 20
    assert metrics["position_rmse_m"] < 1e-5
    assert len(metrics["position_rmse_xyz_m"]) == 3
    assert len(metrics["attitude_rotation_vector_rmse_xyz_deg"]) == 3


def test_kinematic_persistence_is_exact_for_constant_velocity() -> None:
    trajectory = generate_trajectory(seed=4, duration_s=0.4)
    states = trajectory.states.copy()
    states[:, 0:3] = trajectory.time_s[:, None] * np.asarray((1.0, -2.0, 0.5))
    states[:, 3:6] = (1.0, -2.0, 0.5)
    states[:, 6:10] = (1.0, 0.0, 0.0, 0.0)
    states[:, 10:13] = 0.0
    trajectory = replace(trajectory, states=states)

    metrics = kinematic_persistence_windowed_metrics(
        trajectory,
        horizon_steps=5,
    )

    assert metrics["position_rmse_m"] < 1e-12
    assert metrics["velocity_rmse_m_s"] == 0.0
    assert metrics["attitude_rmse_deg"] == 0.0
    assert metrics["angular_velocity_rmse_rad_s"] == 0.0


def test_divergence_diagnostic_reports_stable_matching_rollout() -> None:
    trajectory = generate_fixed_wing_trajectory(seed=3, duration_s=0.3)

    diagnostic = rollout_divergence_metrics(
        true_fixed_wing_parameters(),
        trajectory,
    )

    assert diagnostic["diverged"] is False
    assert diagnostic["divergence_time_s"] is None
    assert diagnostic["stable_fraction"] == 1.0


def test_divergence_diagnostic_validates_threshold_names() -> None:
    trajectory = generate_fixed_wing_trajectory(seed=3, duration_s=0.2)

    with pytest.raises(ValueError, match="unknown divergence threshold"):
        rollout_divergence_metrics(
            true_fixed_wing_parameters(),
            trajectory,
            thresholds={"unknown": 1.0},
        )


def test_training_window_budget_scales_with_diversity_and_horizon() -> None:
    assert _automatic_training_window_budget(
        horizon_steps=100, source_group_count=1
    ) == 5_242
    assert _automatic_training_window_budget(
        horizon_steps=100, source_group_count=12
    ) == 5_242
    assert _automatic_training_window_budget(
        horizon_steps=1_000, source_group_count=12
    ) == 524
    assert _automatic_training_window_budget(
        horizon_steps=5, source_group_count=40
    ) == 8_192


def test_multi_flight_fit_reserves_complete_final_flight(tmp_path) -> None:
    paths = []
    for seed in range(3):
        path = tmp_path / f"flight_{seed}.npz"
        save_trajectory_npz(
            generate_trajectory(seed=seed, duration_s=0.4),
            path,
        )
        paths.append(path)

    _, baseline_params, report = fit_trajectory_artifacts(
        paths,
        horizon=5,
        training_horizons_s=(0.1, 0.2),
        steps=5,
        evaluation_horizons_s=(0.1,),
    )

    assert report["split"]["mode"] == "leave_complete_flights_out"
    assert len(report["split"]["training_flights"]) == 2
    assert report["split"]["validation_flights"][0]["path"] == str(paths[2])
    assert baseline_params is not None
    assert report["configuration"]["training_horizon_steps"] == [5, 10]
    assert report["configuration"]["training_flight_weighting"] == "equal_flight"
    shares = report["configuration"][
        "training_weight_share_per_flight_by_horizon"
    ]["0.1s"]
    assert list(shares.values()) == pytest.approx([0.5, 0.5])
    assert report["models"]["learned_lag"]["validation"]["aggregate"][
        "weighting"
    ] == "equal_flight"
    assert set(report["models"]) == {"learned_lag", "no_lag"}
    assert set(report["models"]["learned_lag"]["fit"]["component_losses"]) == {
        "0.1s",
        "0.2s",
    }
    rollout_loss = report["models"]["learned_lag"]["fit"]["rollout_loss"]
    assert rollout_loss["endpoint_weight"] == pytest.approx(3.0)
    assert rollout_loss["stability_regularization"] == pytest.approx(0.01)
    assert "0.1s" in report["models"]["learned_lag"]["validation"][
        "aggregate"
    ]["horizon_rollouts"]
    predictive_error = report["models"]["learned_lag"]["validation"][
        "predictive_error"
    ]
    assert predictive_error["kind"] == "empirical_horizon_tangent_moments"
    assert predictive_error["horizons_s"] == [0.1]
    assert predictive_error["independent_group_count"] == [1]
    innovation = report["models"]["learned_lag"]["validation"]
    assert innovation["aggregate"]["one_step_innovation"]["status"] == "ok"
    assert innovation["per_flight"][0]["one_step_innovation"]["policy"] == (
        "measured_state_reset_innovation_v1"
    )
    assert np.isfinite(
        report["comparison"]["aggregate_full_rollout"]["position_rmse_m"]
    )


def _write_benchmark_split_flights(tmp_path, splits) -> list[Path]:
    paths = []
    for seed, split in enumerate(splits):
        path = tmp_path / f"flight_{seed}.npz"
        trajectory = generate_trajectory(seed=seed, duration_s=0.4)
        trajectory = replace(
            trajectory,
            labels={**trajectory.labels, "benchmark_split": split},
        )
        save_trajectory_npz(trajectory, path)
        paths.append(path)
    return paths


def test_benchmark_split_labels_determine_holdout_regardless_of_argument_order(
    tmp_path,
) -> None:
    paths = _write_benchmark_split_flights(
        tmp_path, ("training", "training", "training", "validation")
    )

    _, _, forward_report = fit_trajectory_artifacts(
        paths,
        horizon=5,
        steps=1,
        run_no_lag_ablation=False,
        evaluation_horizons_s=(0.1,),
    )
    _, _, reversed_report = fit_trajectory_artifacts(
        list(reversed(paths)),
        horizon=5,
        steps=1,
        run_no_lag_ablation=False,
        evaluation_horizons_s=(0.1,),
    )

    for report in (forward_report, reversed_report):
        assert report["split"]["mode"] == "benchmark_split_holdout"
        assert report["split"]["benchmark_split_holdout"] is True
        assert [
            flight["path"] for flight in report["split"]["validation_flights"]
        ] == [str(paths[3])]
        assert {
            flight["path"] for flight in report["split"]["training_flights"]
        } == {str(paths[0]), str(paths[1]), str(paths[2])}
        assert report["split"]["benchmark_split_training"] == [
            "training",
            "training",
            "training",
        ]
        assert report["split"]["benchmark_split_validation"] == ["validation"]
        assert report["configuration"]["holdout_count"] == 1


def test_positional_holdout_still_follows_argument_order_without_labels(
    tmp_path,
) -> None:
    paths = []
    for seed in range(3):
        path = tmp_path / f"flight_{seed}.npz"
        save_trajectory_npz(generate_trajectory(seed=seed, duration_s=0.4), path)
        paths.append(path)

    _, _, forward_report = fit_trajectory_artifacts(
        paths,
        horizon=5,
        steps=1,
        run_no_lag_ablation=False,
        evaluation_horizons_s=(0.1,),
    )
    _, _, reversed_report = fit_trajectory_artifacts(
        list(reversed(paths)),
        horizon=5,
        steps=1,
        run_no_lag_ablation=False,
        evaluation_horizons_s=(0.1,),
    )

    assert forward_report["split"]["mode"] == "leave_complete_flights_out"
    assert reversed_report["split"]["mode"] == "leave_complete_flights_out"
    assert [
        flight["path"] for flight in forward_report["split"]["validation_flights"]
    ] == [str(paths[2])]
    assert [
        flight["path"] for flight in reversed_report["split"]["validation_flights"]
    ] == [str(paths[0])]


def test_benchmark_split_holdout_rejects_an_explicit_holdout_count(tmp_path) -> None:
    paths = _write_benchmark_split_flights(tmp_path, ("training", "validation"))

    with pytest.raises(BenchmarkSplitHoldoutConflict, match="holdout_count"):
        fit_trajectory_artifacts(paths, holdout_count=2, steps=1)


def test_benchmark_split_holdout_rejects_explicit_holdout_profiles(tmp_path) -> None:
    paths = []
    for seed, split in enumerate(("training", "validation")):
        path = tmp_path / f"flight_{seed}.npz"
        trajectory = generate_trajectory(seed=seed, duration_s=0.4)
        trajectory = replace(
            trajectory,
            labels={
                **trajectory.labels,
                "benchmark_split": split,
                "profile": "hover",
            },
        )
        save_trajectory_npz(trajectory, path)
        paths.append(path)

    with pytest.raises(BenchmarkSplitHoldoutConflict, match="holdout_profiles"):
        fit_trajectory_artifacts(paths, holdout_profiles=("hover",), steps=1)


def test_fit_cli_rejects_holdout_count_when_benchmark_split_labels_are_present(
    tmp_path, monkeypatch, capsys
) -> None:
    paths = _write_benchmark_split_flights(tmp_path, ("training", "validation"))

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "glassbox-fit",
            *(str(path) for path in paths),
            "--holdout-count",
            "2",
            "--steps",
            "1",
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        fit_cli.main()

    assert excinfo.value.code == 2
    assert "benchmark_split" in capsys.readouterr().err


def test_requested_fit_builds_rank_aware_parameter_evidence(tmp_path) -> None:
    paths = []
    for seed in range(3):
        path = tmp_path / f"evidence_flight_{seed}.npz"
        save_trajectory_npz(
            generate_trajectory(seed=seed, duration_s=0.4),
            path,
        )
        paths.append(path)

    _, _, report = fit_trajectory_artifacts(
        paths,
        horizon=5,
        steps=1,
        evaluation_horizons_s=(0.1,),
        run_no_lag_ablation=False,
        build_parameter_evidence=True,
    )

    evidence = report["models"]["learned_lag"]["parameter_evidence"]
    assert evidence["kind"] == "local_structured_parameter_information"
    assert evidence["posterior"] is False
    assert evidence["complete_parameter_uncertainty"] is False
    assert evidence["independent_group_count"] == 2
    assert evidence["fitted_parameter_count"] == 9
    assert evidence["numerical_rank"] <= evidence["fitted_parameter_count"]
    assert report["configuration"]["parameter_evidence"]["requested"] is True
    # The report is written with plain ``json.dumps``; every leaf must be JSON-native.
    assert type(evidence["rank_relative_tolerance"]) is float
    assert json.loads(json.dumps(report))["models"]["learned_lag"]["parameter_evidence"] == evidence


def test_source_group_holdout_keeps_dropout_segments_together(tmp_path) -> None:
    paths = []
    groups_and_durations = (
        ("session-1", 0.4),
        ("session-1", 0.8),
        ("session-2", 0.4),
        ("session-3", 0.4),
        ("session-3", 0.8),
    )
    for seed, (source_group, duration_s) in enumerate(groups_and_durations):
        path = tmp_path / f"segment_{seed}.npz"
        trajectory = generate_trajectory(seed=seed, duration_s=duration_s)
        trajectory = replace(
            trajectory,
            labels={**trajectory.labels, "source_group": source_group},
        )
        save_trajectory_npz(trajectory, path)
        paths.append(path)

    _, _, report = fit_trajectory_artifacts(
        paths,
        horizon=5,
        steps=1,
        run_no_lag_ablation=False,
        evaluation_horizons_s=(0.1,),
    )

    assert report["split"]["mode"] == "leave_source_groups_out"
    assert report["split"]["training_source_groups"] == [
        "session-1",
        "session-2",
    ]
    assert report["split"]["validation_source_groups"] == ["session-3"]
    assert len(report["split"]["training_flights"]) == 3
    assert len(report["split"]["validation_flights"]) == 2
    assert report["dataset"]["source_group_count"] == 3
    assert report["configuration"]["holdout_source_group_count"] == 1
    assert report["configuration"]["training_flight_weighting"] == (
        "equal_source_group_then_equal_window"
    )
    group_shares = report["configuration"][
        "training_weight_share_per_source_group_by_horizon"
    ]["0.1s"]
    assert group_shares == pytest.approx({"session-1": 0.5, "session-2": 0.5})
    selection = report["configuration"]["training_window_selection"]
    assert selection["budget_policy"] == "automatic_corpus_and_horizon"
    assert selection["selection_policy_by_horizon"] == {
        "0.1s": "all_candidates"
    }
    assert selection["candidate_windows_by_horizon"] == {
        "0.1s": selection["selected_windows_by_horizon"]["0.1s"]
    }


def test_characterization_only_segments_use_honest_chronological_holdout(
    tmp_path,
) -> None:
    paths = []
    for seed in range(3):
        path = tmp_path / f"segment_{seed}.npz"
        trajectory = generate_trajectory(seed=seed, duration_s=0.4)
        trajectory = replace(
            trajectory,
            labels={
                **trajectory.labels,
                "source_group": "one-recording",
                "benchmark_split": "characterization_only",
            },
        )
        save_trajectory_npz(trajectory, path)
        paths.append(path)

    _, _, report = fit_trajectory_artifacts(
        paths,
        holdout_count=1,
        horizon=5,
        steps=1,
        run_no_lag_ablation=False,
        evaluation_horizons_s=(0.1,),
    )

    assert report["split"]["mode"] == (
        "chronological_segments_within_source_group_characterization"
    )
    assert report["split"]["independent_source_group_holdout"] is False
    assert len(report["split"]["training_flights"]) == 2
    assert report["split"]["validation_flights"][0]["path"] == str(paths[-1])


def test_multi_flight_fit_cannot_hold_out_every_flight(tmp_path) -> None:
    paths = []
    for seed in range(2):
        path = tmp_path / f"flight_{seed}.npz"
        save_trajectory_npz(generate_trajectory(seed=seed, duration_s=0.2), path)
        paths.append(path)

    with pytest.raises(ValueError, match="not all flights"):
        fit_trajectory_artifacts(paths, holdout_count=2, steps=1)


def test_multi_flight_fit_rejects_mixed_sample_rates(tmp_path) -> None:
    paths = []
    for seed, dt_s in ((0, 0.02), (1, 0.01)):
        path = tmp_path / f"flight_{seed}.npz"
        trajectory = generate_trajectory(seed=seed, duration_s=0.4, dt_s=dt_s)
        save_trajectory_npz(trajectory, path)
        paths.append(path)

    with pytest.raises(ValueError, match="inconsistent dataset sample_rate_hz"):
        fit_trajectory_artifacts(paths, steps=1)


def test_profile_holdout_reserves_every_flight_in_profile(tmp_path) -> None:
    paths = []
    profiles = ("hover", "lateral", "lateral")
    for seed, profile in enumerate(profiles):
        path = tmp_path / f"flight_{seed}.npz"
        trajectory = generate_trajectory(seed=seed, duration_s=0.4)
        trajectory = replace(
            trajectory,
            labels={**trajectory.labels, "profile": profile},
        )
        save_trajectory_npz(trajectory, path)
        paths.append(path)

    _, _, report = fit_trajectory_artifacts(
        paths,
        holdout_profiles=("lateral",),
        horizon=5,
        steps=1,
        run_no_lag_ablation=False,
    )

    assert report["split"]["mode"] == "leave_profiles_out"
    assert report["split"]["held_out_profiles"] == ["lateral"]
    assert len(report["split"]["training_flights"]) == 1
    assert len(report["split"]["validation_flights"]) == 2


def test_profile_labeled_training_balances_profiles_before_flights(tmp_path) -> None:
    paths = []
    profiles = ("vertical", "vertical", "lateral", "yaw")
    for seed, profile in enumerate(profiles):
        path = tmp_path / f"flight_{seed}.npz"
        trajectory = generate_trajectory(seed=seed, duration_s=0.4)
        trajectory = replace(
            trajectory,
            labels={**trajectory.labels, "profile": profile},
        )
        save_trajectory_npz(trajectory, path)
        paths.append(path)

    _, _, report = fit_trajectory_artifacts(
        paths,
        holdout_profiles=("yaw",),
        horizon=5,
        steps=1,
        run_no_lag_ablation=False,
    )

    assert report["configuration"]["training_flight_weighting"] == (
        "equal_profile_then_equal_flight"
    )
    shares = list(
        report["configuration"]["training_weight_share_per_flight_by_horizon"][
            "0.1s"
        ].values()
    )
    assert shares == pytest.approx([0.25, 0.25, 0.5])


def test_profile_benchmark_runs_one_fold_per_profile(tmp_path) -> None:
    paths = []
    for seed, profile in enumerate(("vertical", "lateral", "yaw")):
        path = tmp_path / f"flight_{seed}.npz"
        trajectory = generate_trajectory(seed=seed, duration_s=0.3)
        trajectory = replace(
            trajectory,
            labels={**trajectory.labels, "profile": profile},
        )
        save_trajectory_npz(trajectory, path)
        paths.append(path)

    summary = benchmark_profiles(
        paths,
        tmp_path / "benchmark",
        training_horizons_s=(0.1,),
        evaluation_horizons_s=(0.1,),
        steps=1,
    )

    assert summary["profile_count"] == 3
    assert set(summary["per_profile"]) == {"vertical", "lateral", "yaw"}
    assert summary["aggregate"]["weighting"] == "equal_profile"
    assert summary["acceptance"]["status"] == "not_scored"
    assert summary["configuration"]["control_names"] == [
        "motor_front_left",
        "motor_front_right",
        "motor_rear_right",
        "motor_rear_left",
    ]
    assert (tmp_path / "benchmark" / "summary.json").exists()


def test_rollout_error_excludes_the_measured_initial_sample() -> None:
    from glassbox.evaluation import ROLLOUT_METRIC_POLICY, _state_error_metrics
    from glassbox.synthetic import resting_state

    horizon = 5
    target = np.tile(resting_state(), (3, horizon + 1, 1))
    predicted = target.copy()
    predicted[:, 1:, 0:3] += 0.3

    metrics = _state_error_metrics(predicted, target, duration_s=0.1)

    assert metrics["position_rmse_m"] == pytest.approx(0.3)
    assert metrics["sample_count"] == 3 * horizon
    assert metrics["rollout_count"] == 3
    assert metrics["metric_policy"] == ROLLOUT_METRIC_POLICY
    with pytest.raises(ValueError, match="at least one predicted step"):
        _state_error_metrics(predicted[:, :1], target[:, :1], duration_s=0.0)
