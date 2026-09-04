import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from glassbox.core.data import load_trajectory_npz, save_trajectory_npz
from glassbox.core.fixedwing_synthetic import (
    true_fixed_wing_parameters,
)
from glassbox.core.metrics import (
    kinematic_persistence_windowed_metrics,
    predict_windows,
    rollout_divergence_metrics,
    rollout_metrics,
)
from glassbox.core.synthetic import true_parameters
from glassbox.fitting import (
    FitSpec,
    Holdout,
    _automatic_training_window_budget,
    _dataset_contract,
    build_training_windows,
    fit,
)
from glassbox.workflows.holdout import evaluate_holdout


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


def test_dataset_pooling_uses_canonical_semantics_not_px4_source_layout(
    fixedwing_flight,
) -> None:
    base = fixedwing_flight(0, 0.2)
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


def test_dataset_pooling_rejects_different_vehicle_configuration_ids(
    fixedwing_flight,
) -> None:
    first = fixedwing_flight(0, 0.2)
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


def test_windowed_metrics_cover_multiple_initial_conditions(quadrotor_flight) -> None:
    trajectory = quadrotor_flight(9)

    metrics = rollout_metrics(
        predict_windows(
            true_parameters(),
            trajectory,
            horizon_steps=5,
        )
    )

    assert metrics["rollout_count"] == 4
    # Four windows of five predicted steps; the shared initial sample is excluded.
    assert metrics["sample_count"] == 20
    assert metrics["position_rmse_m"] < 1e-5
    assert len(metrics["position_rmse_xyz_m"]) == 3
    assert len(metrics["attitude_rotation_vector_rmse_xyz_deg"]) == 3


def test_kinematic_persistence_is_exact_for_constant_velocity(
    quadrotor_flight,
) -> None:
    trajectory = quadrotor_flight(4)
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


def test_divergence_diagnostic_reports_stable_matching_rollout(
    fixedwing_flight,
) -> None:
    trajectory = fixedwing_flight(3, 0.3)

    diagnostic = rollout_divergence_metrics(
        true_fixed_wing_parameters(),
        trajectory,
    )

    assert diagnostic["diverged"] is False
    assert diagnostic["divergence_time_s"] is None
    assert diagnostic["stable_fraction"] == 1.0


def test_divergence_diagnostic_validates_threshold_names(fixedwing_flight) -> None:
    trajectory = fixedwing_flight(3, 0.2)

    with pytest.raises(ValueError, match="unknown divergence threshold"):
        rollout_divergence_metrics(
            true_fixed_wing_parameters(),
            trajectory,
            thresholds={"unknown": 1.0},
        )


def test_training_window_budget_scales_with_diversity_and_horizon() -> None:
    assert (
        _automatic_training_window_budget(horizon_steps=100, source_group_count=1)
        == 5_242
    )
    assert (
        _automatic_training_window_budget(horizon_steps=100, source_group_count=12)
        == 5_242
    )
    assert (
        _automatic_training_window_budget(horizon_steps=1_000, source_group_count=12)
        == 524
    )
    assert (
        _automatic_training_window_budget(horizon_steps=5, source_group_count=40)
        == 8_192
    )


@pytest.mark.slow
def test_multi_flight_fit_reserves_complete_final_flight(
    tmp_path, quadrotor_flight
) -> None:
    paths = []
    for seed in range(3):
        path = tmp_path / f"flight_{seed}.npz"
        save_trajectory_npz(quadrotor_flight(seed), path)
        paths.append(path)

    outcome = fit(
        paths,
        FitSpec(
            horizon_steps=5,
            horizons_s=(0.1, 0.2),
            steps=5,
            evaluation_horizons_s=(0.1,),
            ablations=("no_lag",),
        ),
    )
    report = outcome.report

    assert report["split"]["mode"] == "leave_complete_flights_out"
    assert len(report["split"]["training_flights"]) == 2
    assert report["split"]["validation_flights"][0]["path"] == str(paths[2])
    assert set(outcome.ablations) == {"no_lag"}
    assert report["configuration"]["training_horizon_steps"] == [5, 10]
    assert report["configuration"]["training_flight_weighting"] == "equal_flight"
    assert (
        report["models"]["learned_lag"]["validation"]["aggregate"]["weighting"]
        == "equal_flight"
    )
    assert set(report["models"]) == {"learned_lag", "no_lag"}
    assert set(report["models"]["learned_lag"]["fit"]["component_losses"]) == {
        "0.1s",
        "0.2s",
    }
    rollout_loss = report["models"]["learned_lag"]["fit"]["rollout_loss"]
    assert rollout_loss["endpoint_weight"] == pytest.approx(3.0)
    assert rollout_loss["stability_regularization"] == pytest.approx(0.01)
    assert (
        "0.1s"
        in report["models"]["learned_lag"]["validation"]["aggregate"][
            "horizon_rollouts"
        ]
    )
    forecast_error = report["models"]["learned_lag"]["validation"]["forecast_error"]
    assert forecast_error["kind"] == "held_out_horizon_tangent_second_moments"
    assert forecast_error["centered"] is False
    assert forecast_error["horizons_s"] == [0.1]
    assert forecast_error["independent_group_count"] == [1]
    # The noise model is measured on every fit, whether or not the caller asked
    # for the precision the belief accumulates around it.
    validation_block = report["models"]["learned_lag"]["validation"]
    assert len(validation_block["innovation_noise"]) == 12
    assert "0.1s" in validation_block["aggregate"]["held_out_mean_tangent_error"]
    # Innovation diagnostics are opt-in, so nothing runs them by default.
    validation = report["models"]["learned_lag"]["validation"]
    assert report["configuration"]["diagnostics"] is False
    assert "one_step_innovation" not in validation["aggregate"]
    assert "one_step_innovation" not in validation["per_flight"][0]
    assert np.isfinite(
        report["comparison"]["aggregate_full_rollout"]["position_rmse_m"]
    )


def _write_benchmark_split_flights(tmp_path, quadrotor_flight, splits) -> list[Path]:
    paths = []
    for seed, split in enumerate(splits):
        path = tmp_path / f"flight_{seed}.npz"
        trajectory = quadrotor_flight(seed)
        trajectory = replace(
            trajectory,
            labels={**trajectory.labels, "benchmark_split": split},
        )
        save_trajectory_npz(trajectory, path)
        paths.append(path)
    return paths


def test_requested_fit_builds_the_information_its_training_supports(
    tmp_path, quadrotor_flight
) -> None:
    # A label holdout reserves exactly the flight positional holdout would
    # have reserved, so this single fit also carries the label rule in its
    # split section. Which flights the labels select is pinned on the rules in
    # ``test_holdout_plan.py``.
    paths = _write_benchmark_split_flights(
        tmp_path, quadrotor_flight, ("training", "training", "validation")
    )

    report = fit(
        paths,
        FitSpec(
            holdout=Holdout.by_label("benchmark_split", ("validation",)),
            horizon_steps=5,
            steps=1,
            evaluation_horizons_s=(0.1,),
            parameter_evidence=True,
        ),
    ).report

    assert report["split"]["mode"] == "leave_labeled_out"
    assert report["split"]["holdout"] == {
        "rule": "label",
        "key": "benchmark_split",
        "values": ["validation"],
    }
    assert report["configuration"]["holdout_count"] == 1
    evidence = report["models"]["learned_lag"]["parameter_evidence"]
    assert evidence["kind"] == "structured_parameter_information"
    assert evidence["estimable_count"] == 9
    assert 0 < evidence["resolved_rank"] <= evidence["estimable_count"]
    assert evidence["effective_count"] > 0.0
    assert len(evidence["innovation_noise"]) == 12
    assert len(evidence["noise_floor"]) == 12
    assert report["configuration"]["parameter_evidence"]["requested"] is True
    # The report is written with plain ``json.dumps``; every leaf must be JSON-native.
    assert type(evidence["rank_relative_tolerance"]) is float
    assert (
        json.loads(json.dumps(report))["models"]["learned_lag"]["parameter_evidence"]
        == evidence
    )


def test_source_group_training_weights_equalize_groups_not_segments(
    tmp_path, quadrotor_flight
) -> None:
    """The training weight a source group receives does not follow its segments.

    Which flights a source-group holdout reserves is pinned on the planner in
    ``test_holdout_plan.py::test_source_group_holdout_keeps_every_segment_of_a_group_together``;
    what only a real fit can show is the weighting the surviving groups get,
    so that is all this asserts.
    """

    paths = []
    groups_and_durations = (
        ("session-1", 0.2),
        ("session-1", 0.4),
        ("session-2", 0.2),
        ("session-3", 0.2),
        ("session-3", 0.4),
    )
    for seed, (source_group, duration_s) in enumerate(groups_and_durations):
        path = tmp_path / f"segment_{seed}.npz"
        trajectory = quadrotor_flight(seed, duration_s)
        trajectory = replace(
            trajectory,
            labels={**trajectory.labels, "source_group": source_group},
        )
        save_trajectory_npz(trajectory, path)
        paths.append(path)

    report = fit(
        paths,
        FitSpec(horizon_steps=5, steps=1, evaluation_horizons_s=(0.1,)),
    ).report

    assert report["dataset"]["source_group_count"] == 3
    assert report["configuration"]["holdout_source_group_count"] == 1
    assert report["configuration"]["training_flight_weighting"] == (
        "equal_source_group_then_equal_window"
    )
    selection = report["configuration"]["training_window_selection"]
    assert selection["budget_policy"] == "automatic_corpus_and_horizon"
    assert selection["source_group_count"] == 2
    assert selection["stratification"] == "source_group"

    # Each training source group carries half the total weight, whatever the
    # window counts of its member flights.
    windows = build_training_windows(
        Holdout.by_group().plan([load_trajectory_npz(path) for path in paths], paths),
        FitSpec(horizon_steps=5),
    ).window_sets[0]
    groups = np.asarray([0, 0, 1])[windows.trajectory_indices]
    total = float(np.sum(windows.window_weights))
    assert np.sum(windows.window_weights[groups == 0]) / total == pytest.approx(0.5)
    assert np.sum(windows.window_weights[groups == 1]) / total == pytest.approx(0.5)


def test_multi_flight_fit_rejects_mixed_sample_rates(
    tmp_path, quadrotor_flight
) -> None:
    paths = []
    for seed, dt_s in ((0, 0.02), (1, 0.01)):
        path = tmp_path / f"flight_{seed}.npz"
        trajectory = quadrotor_flight(seed, 0.4, dt_s)
        save_trajectory_npz(trajectory, path)
        paths.append(path)

    with pytest.raises(ValueError, match="inconsistent dataset sample_rate_hz"):
        fit(paths, FitSpec(steps=1))


def test_profile_labeled_training_balances_profiles_before_flights(
    tmp_path, quadrotor_flight
) -> None:
    paths = []
    profiles = ("vertical", "vertical", "lateral", "yaw")
    for seed, profile in enumerate(profiles):
        path = tmp_path / f"flight_{seed}.npz"
        trajectory = quadrotor_flight(seed)
        trajectory = replace(
            trajectory,
            labels={**trajectory.labels, "profile": profile},
        )
        save_trajectory_npz(trajectory, path)
        paths.append(path)

    spec = FitSpec(
        holdout=Holdout.by_label("profile", ("yaw",)),
        horizon_steps=5,
        steps=1,
    )
    report = fit(paths, spec).report

    assert report["configuration"]["training_flight_weighting"] == (
        "equal_profile_then_equal_flight"
    )
    # Each maneuver family carries equal total weight and its replicates split
    # it, so the two vertical flights get a quarter each and lateral a half.
    windows = build_training_windows(
        spec.holdout.plan([load_trajectory_npz(path) for path in paths], paths),
        spec,
    ).window_sets[0]
    total = float(np.sum(windows.window_weights))
    shares = [
        float(np.sum(windows.window_weights[windows.trajectory_indices == index]))
        / total
        for index in range(3)
    ]
    assert shares == pytest.approx([0.25, 0.25, 0.5])


def test_profile_holdout_runs_one_fold_per_profile(tmp_path, quadrotor_flight) -> None:
    paths = []
    for seed, profile in enumerate(("vertical", "lateral", "yaw")):
        path = tmp_path / f"flight_{seed}.npz"
        trajectory = quadrotor_flight(seed, 0.3)
        trajectory = replace(
            trajectory,
            labels={**trajectory.labels, "profile": profile},
        )
        save_trajectory_npz(trajectory, path)
        paths.append(path)

    summary = evaluate_holdout(
        paths,
        hold_out="profile",
        spec=FitSpec(horizons_s=(0.1,), evaluation_horizons_s=(0.1,), steps=1),
        output_dir=tmp_path / "benchmark",
    )

    assert summary["evaluation"] == "leave_one_profile_out"
    assert summary["fold_count"] == 3
    assert set(summary["per_fold"]) == {"vertical", "lateral", "yaw"}
    assert summary["aggregate"]["weighting"] == "equal_profile"
    assert summary["configuration"]["control_names"] == [
        "motor_front_left",
        "motor_front_right",
        "motor_rear_right",
        "motor_rear_left",
    ]
    assert (tmp_path / "benchmark" / "summary.json").exists()


def test_rollout_error_excludes_the_measured_initial_sample() -> None:
    from glassbox.core.metrics import ROLLOUT_METRIC_POLICY, state_error_metrics
    from glassbox.core.synthetic import resting_state

    horizon = 5
    target = np.tile(resting_state(), (3, horizon + 1, 1))
    predicted = target.copy()
    predicted[:, 1:, 0:3] += 0.3

    metrics = state_error_metrics(predicted, target, duration_s=0.1)

    assert metrics["position_rmse_m"] == pytest.approx(0.3)
    assert metrics["sample_count"] == 3 * horizon
    assert metrics["rollout_count"] == 3
    assert metrics["metric_policy"] == ROLLOUT_METRIC_POLICY
    with pytest.raises(ValueError, match="at least one predicted step"):
        state_error_metrics(predicted[:, :1], target[:, :1], duration_s=0.0)
