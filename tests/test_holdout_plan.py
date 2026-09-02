"""Unit tests for the pure holdout planner behind ``fit_trajectory_artifacts``.

``plan_holdout`` decides which flights train and which are reserved without
loading a file or fitting anything, so every split rule is exercised here on
in-memory trajectories in milliseconds.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from glassbox.core.synthetic import generate_trajectory
from glassbox.workflows.fitting import (
    BenchmarkSplitHoldoutConflict,
    FitRequest,
    plan_holdout,
)


def _flights(count: int, **labels_by_index) -> list:
    trajectories = []
    for seed in range(count):
        trajectory = generate_trajectory(seed=seed, duration_s=0.4)
        labels = {
            key: values[seed]
            for key, values in labels_by_index.items()
            if values[seed] is not None
        }
        if labels:
            trajectory = replace(trajectory, labels={**trajectory.labels, **labels})
        trajectories.append(trajectory)
    return trajectories


def _paths(count: int) -> list[str]:
    return [f"flight_{index}.npz" for index in range(count)]


def test_single_trajectory_is_split_temporally() -> None:
    trajectories = _flights(1)

    plan = plan_holdout(trajectories, FitRequest(train_fraction=0.5), ["only.npz"])

    assert plan.mode == "temporal_within_flight"
    assert plan.training_labels == ("only.npz#training",)
    assert [flight.path for flight in plan.validation] == ["only.npz#validation"]
    # The held-out segment is scored with the training controls as its history.
    assert plan.validation[0].control_history is plan.training[0].controls
    assert plan.training_source_groups is None


def test_positional_holdout_reserves_the_final_flights_in_argument_order() -> None:
    trajectories = _flights(3)
    paths = _paths(3)

    forward = plan_holdout(trajectories, FitRequest(), paths)
    reversed_plan = plan_holdout(
        list(reversed(trajectories)), FitRequest(), list(reversed(paths))
    )

    assert forward.mode == "leave_complete_flights_out"
    assert forward.training_labels == ("flight_0.npz", "flight_1.npz")
    assert [flight.path for flight in forward.validation] == ["flight_2.npz"]
    # Positional splitting is, by design, argument-order dependent.
    assert [flight.path for flight in reversed_plan.validation] == ["flight_0.npz"]


def test_positional_holdout_honours_an_explicit_count() -> None:
    plan = plan_holdout(_flights(4), FitRequest(holdout_count=2), _paths(4))

    assert len(plan.training) == 2
    assert [flight.path for flight in plan.validation] == [
        "flight_2.npz",
        "flight_3.npz",
    ]


def test_positional_holdout_cannot_reserve_every_flight() -> None:
    with pytest.raises(ValueError, match="not all flights"):
        plan_holdout(_flights(2), FitRequest(holdout_count=2), _paths(2))


def test_benchmark_split_labels_determine_the_holdout_in_any_order() -> None:
    splits = ("training", "training", "training", "validation")
    trajectories = _flights(4, benchmark_split=splits)
    paths = _paths(4)

    forward = plan_holdout(trajectories, FitRequest(), paths)
    backward = plan_holdout(
        list(reversed(trajectories)), FitRequest(), list(reversed(paths))
    )

    for plan in (forward, backward):
        assert plan.mode == "benchmark_split_holdout"
        assert [flight.path for flight in plan.validation] == ["flight_3.npz"]
        assert set(plan.training_labels) == {
            "flight_0.npz",
            "flight_1.npz",
            "flight_2.npz",
        }


def test_benchmark_split_labels_reject_an_explicit_holdout_count() -> None:
    trajectories = _flights(2, benchmark_split=("training", "validation"))

    with pytest.raises(BenchmarkSplitHoldoutConflict, match="holdout_count"):
        plan_holdout(trajectories, FitRequest(holdout_count=2), _paths(2))


def test_benchmark_split_labels_reject_explicit_holdout_profiles() -> None:
    trajectories = _flights(
        2,
        benchmark_split=("training", "validation"),
        profile=("hover", "hover"),
    )

    with pytest.raises(BenchmarkSplitHoldoutConflict, match="holdout_profiles"):
        plan_holdout(trajectories, FitRequest(holdout_profiles=("hover",)), _paths(2))


def test_benchmark_split_labels_can_be_overridden() -> None:
    trajectories = _flights(3, benchmark_split=("validation", "training", "training"))

    plan = plan_holdout(
        trajectories, FitRequest(respect_benchmark_split=False), _paths(3)
    )

    assert plan.mode == "leave_complete_flights_out"
    assert [flight.path for flight in plan.validation] == ["flight_2.npz"]


def test_partial_benchmark_split_labels_fall_back_to_argument_order() -> None:
    trajectories = _flights(3, benchmark_split=("training", "validation", None))

    plan = plan_holdout(trajectories, FitRequest(), _paths(3))

    assert plan.mode == "leave_complete_flights_out"


def test_profile_holdout_reserves_every_flight_in_the_named_profiles() -> None:
    trajectories = _flights(3, profile=("hover", "lateral", "lateral"))

    plan = plan_holdout(
        trajectories, FitRequest(holdout_profiles=("lateral",)), _paths(3)
    )

    assert plan.mode == "leave_profiles_out"
    assert plan.training_labels == ("flight_0.npz",)
    assert [flight.path for flight in plan.validation] == [
        "flight_1.npz",
        "flight_2.npz",
    ]


def test_profile_holdout_requires_every_flight_to_carry_a_profile() -> None:
    trajectories = _flights(3, profile=("hover", "lateral", None))

    with pytest.raises(ValueError, match=r"unlabeled: flight_2\.npz"):
        plan_holdout(trajectories, FitRequest(holdout_profiles=("hover",)), _paths(3))


def test_profile_holdout_rejects_an_absent_profile() -> None:
    trajectories = _flights(2, profile=("hover", "lateral"))

    with pytest.raises(ValueError, match="holdout profiles are absent: yaw"):
        plan_holdout(trajectories, FitRequest(holdout_profiles=("yaw",)), _paths(2))


def test_profile_holdout_cannot_reserve_every_trajectory() -> None:
    trajectories = _flights(2, profile=("hover", "hover"))

    with pytest.raises(ValueError, match="cannot reserve every trajectory"):
        plan_holdout(trajectories, FitRequest(holdout_profiles=("hover",)), _paths(2))


def test_profile_holdout_requires_multiple_trajectories() -> None:
    trajectories = _flights(1, profile=("hover",))

    with pytest.raises(ValueError, match="requires multiple trajectories"):
        plan_holdout(trajectories, FitRequest(holdout_profiles=("hover",)), _paths(1))


def test_source_group_holdout_keeps_every_segment_of_a_group_together() -> None:
    groups = ("session-1", "session-1", "session-2", "session-3", "session-3")
    trajectories = _flights(5, source_group=groups)

    plan = plan_holdout(trajectories, FitRequest(), _paths(5))

    assert plan.mode == "leave_source_groups_out"
    assert plan.training_group_order == ["session-1", "session-2"]
    assert plan.validation_group_order == ["session-3"]
    assert len(plan.training) == 3
    assert [flight.path for flight in plan.validation] == [
        "flight_3.npz",
        "flight_4.npz",
    ]


def test_source_group_holdout_cannot_reserve_every_group() -> None:
    trajectories = _flights(3, source_group=("a", "b", "c"))

    with pytest.raises(ValueError, match="not all source groups"):
        plan_holdout(trajectories, FitRequest(holdout_count=3), _paths(3))


def test_source_group_labels_must_cover_every_trajectory() -> None:
    trajectories = _flights(2, source_group=("a", None))

    with pytest.raises(ValueError, match=r"unlabeled: flight_1\.npz"):
        plan_holdout(trajectories, FitRequest(), _paths(2))


def test_single_group_characterization_falls_back_to_chronological_segments() -> None:
    trajectories = _flights(
        3,
        source_group=("one-recording",) * 3,
        benchmark_split=("characterization_only",) * 3,
    )

    plan = plan_holdout(trajectories, FitRequest(), _paths(3))

    assert plan.mode == "chronological_segments_within_source_group_characterization"
    assert len(plan.training) == 2
    assert [flight.path for flight in plan.validation] == ["flight_2.npz"]
    assert not set(plan.training_source_groups).isdisjoint(plan.validation_group_order)


def test_paths_default_to_positional_placeholders() -> None:
    plan = plan_holdout(_flights(2), FitRequest())

    assert plan.training_labels == ("trajectory_0",)
    assert [flight.path for flight in plan.validation] == ["trajectory_1"]


def test_paths_must_label_every_trajectory() -> None:
    with pytest.raises(ValueError, match="paths must label every trajectory"):
        plan_holdout(_flights(2), FitRequest(), ["only-one.npz"])


def test_planning_requires_at_least_one_trajectory() -> None:
    with pytest.raises(ValueError, match="at least one trajectory is required"):
        plan_holdout([], FitRequest())


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"evaluation_horizons_s": (0.0,)}, "evaluation horizons must be positive"),
        ({"training_horizons_s": (-1.0,)}, "training horizons must be positive"),
        ({"model_class": "nope"}, "model_class must be structured"),
        ({"endpoint_weight": 0.5}, "endpoint_weight must be at least one"),
        (
            {"stability_regularization": -1.0},
            "stability_regularization must be nonnegative",
        ),
        (
            {"training_source_group_weights": {"a": -1.0}},
            "training_source_group_weights values must be finite",
        ),
        (
            {"normalization_source_group_weights": {"a": 1.0}},
            "normalization_source_group_weights requires explicit training",
        ),
    ),
)
def test_fit_request_validates_its_knobs(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        FitRequest(**kwargs)


def test_fit_request_normalizes_sequence_knobs_to_tuples() -> None:
    request = FitRequest(
        evaluation_horizons_s=[0.1, 0.5],
        training_horizons_s=[0.2],
        holdout_profiles=["hover", "hover"],
    )

    assert request.evaluation_horizons_s == (0.1, 0.5)
    assert request.training_horizons_s == (0.2,)
    assert request.holdout_profiles == ("hover", "hover")


def test_fit_request_stride_defaults_to_the_horizon() -> None:
    assert FitRequest().stride_for(25) == 25
    assert FitRequest(stride=3).stride_for(25) == 3
