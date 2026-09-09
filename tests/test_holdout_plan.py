"""Unit tests for the three holdout rules and the spec behind ``fit``.

:meth:`Holdout.plan` decides which flights train and which are reserved without
loading a file or fitting anything, so every rule is exercised here on
in-memory trajectories in milliseconds.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from glassbox.fitting import FitSpec, Holdout, LossPolicy, WeightingPolicy


@pytest.fixture
def flights(quadrotor_flight):
    """Return a builder for ``count`` labelled rollouts.

    Every test here wants the same first few 0.4s rollouts under different
    labels, and the planner never looks at their samples, so the rollouts come
    from the session cache and only the labels are rebuilt per call.
    """

    def build(count: int, **labels_by_index) -> list:
        trajectories = []
        for seed in range(count):
            trajectory = quadrotor_flight(seed)
            labels = {
                key: values[seed]
                for key, values in labels_by_index.items()
                if values[seed] is not None
            }
            if labels:
                trajectory = replace(trajectory, labels={**trajectory.labels, **labels})
            trajectories.append(trajectory)
        return trajectories

    return build


def _paths(count: int) -> list[str]:
    return [f"flight_{index}.npz" for index in range(count)]


def test_temporal_splits_one_trajectory_chronologically(flights) -> None:
    trajectories = flights(1)

    plan = Holdout.temporal(0.5).plan(trajectories, ["only.npz"])

    assert plan.mode == "temporal_within_flight"
    assert plan.training_labels == ("only.npz#training",)
    assert [flight.path for flight in plan.validation] == ["only.npz#validation"]
    np.testing.assert_array_equal(
        plan.validation[0].trajectory.control_prefix, plan.training[0].controls
    )
    assert plan.training_source_groups is None


def test_temporal_rejects_more_than_one_trajectory(flights) -> None:
    with pytest.raises(ValueError, match="exactly one trajectory"):
        Holdout.temporal(0.5).plan(flights(2), _paths(2))


def test_group_holdout_falls_back_to_argument_order_without_the_label(
    flights,
) -> None:
    trajectories = flights(3)
    paths = _paths(3)

    forward = Holdout.by_group().plan(trajectories, paths)
    reversed_plan = Holdout.by_group().plan(
        list(reversed(trajectories)), list(reversed(paths))
    )

    assert forward.mode == "leave_complete_flights_out"
    assert forward.training_labels == ("flight_0.npz", "flight_1.npz")
    assert [flight.path for flight in forward.validation] == ["flight_2.npz"]
    # Positional splitting is, by design, argument-order dependent.
    assert [flight.path for flight in reversed_plan.validation] == ["flight_0.npz"]


def test_group_holdout_honours_an_explicit_count(flights) -> None:
    plan = Holdout.by_group(2).plan(flights(4), _paths(4))

    assert len(plan.training) == 2
    assert [flight.path for flight in plan.validation] == [
        "flight_2.npz",
        "flight_3.npz",
    ]


def test_group_holdout_cannot_reserve_every_flight(flights) -> None:
    with pytest.raises(ValueError, match="not all flights"):
        Holdout.by_group(2).plan(flights(2), _paths(2))


def test_group_holdout_needs_more_than_one_trajectory(flights) -> None:
    with pytest.raises(ValueError, match="requires multiple trajectories"):
        Holdout.by_group().plan(flights(1), _paths(1))


def test_label_holdout_is_independent_of_argument_order(flights) -> None:
    splits = ("training", "training", "training", "validation")
    trajectories = flights(4, benchmark_split=splits)
    paths = _paths(4)
    holdout = Holdout.by_label("benchmark_split", ("validation",))

    forward = holdout.plan(trajectories, paths)
    backward = holdout.plan(list(reversed(trajectories)), list(reversed(paths)))

    for plan in (forward, backward):
        assert plan.mode == "leave_labeled_out"
        assert [flight.path for flight in plan.validation] == ["flight_3.npz"]
        assert set(plan.training_labels) == {
            "flight_0.npz",
            "flight_1.npz",
            "flight_2.npz",
        }


def test_label_holdout_reserves_every_flight_in_the_named_values(flights) -> None:
    trajectories = flights(3, profile=("hover", "lateral", "lateral"))

    plan = Holdout.by_label("profile", ("lateral",)).plan(trajectories, _paths(3))

    assert plan.mode == "leave_labeled_out"
    assert plan.training_labels == ("flight_0.npz",)
    assert [flight.path for flight in plan.validation] == [
        "flight_1.npz",
        "flight_2.npz",
    ]


def test_label_holdout_requires_every_flight_to_carry_the_label(flights) -> None:
    trajectories = flights(3, profile=("hover", "lateral", None))

    with pytest.raises(ValueError, match=r"unlabeled: flight_2\.npz"):
        Holdout.by_label("profile", ("hover",)).plan(trajectories, _paths(3))


def test_label_holdout_rejects_an_absent_value(flights) -> None:
    trajectories = flights(2, profile=("hover", "lateral"))

    with pytest.raises(ValueError, match="profile values are absent: yaw"):
        Holdout.by_label("profile", ("yaw",)).plan(trajectories, _paths(2))


def test_label_holdout_cannot_reserve_every_trajectory(flights) -> None:
    trajectories = flights(2, profile=("hover", "hover"))

    with pytest.raises(ValueError, match="cannot reserve every trajectory"):
        Holdout.by_label("profile", ("hover",)).plan(trajectories, _paths(2))


def test_label_holdout_needs_more_than_one_trajectory(flights) -> None:
    trajectories = flights(1, profile=("hover",))

    with pytest.raises(ValueError, match="requires multiple trajectories"):
        Holdout.by_label("profile", ("hover",)).plan(trajectories, _paths(1))


def test_label_holdout_needs_at_least_one_held_out_value() -> None:
    with pytest.raises(ValueError, match="at least one held-out value"):
        Holdout.by_label("profile", ())


def test_group_holdout_keeps_every_segment_of_a_group_together(flights) -> None:
    groups = ("session-1", "session-1", "session-2", "session-3", "session-3")
    trajectories = flights(5, source_group=groups)

    plan = Holdout.by_group().plan(trajectories, _paths(5))

    assert plan.mode == "leave_source_groups_out"
    assert plan.training_group_order == ["session-1", "session-2"]
    assert plan.validation_group_order == ["session-3"]
    assert len(plan.training) == 3
    assert [flight.path for flight in plan.validation] == [
        "flight_3.npz",
        "flight_4.npz",
    ]


def test_group_holdout_cannot_reserve_every_group(flights) -> None:
    trajectories = flights(3, source_group=("a", "b", "c"))

    with pytest.raises(ValueError, match="not all source groups"):
        Holdout.by_group(3).plan(trajectories, _paths(3))


def test_source_group_labels_must_cover_every_trajectory(flights) -> None:
    trajectories = flights(2, source_group=("a", None))

    with pytest.raises(ValueError, match=r"unlabeled: flight_1\.npz"):
        Holdout.by_group().plan(trajectories, _paths(2))


def test_one_group_falls_back_to_chronological_segments(flights) -> None:
    """A label that separates nothing splits the flights in argument order."""

    trajectories = flights(3, source_group=("one-recording",) * 3)

    plan = Holdout.by_group().plan(trajectories, _paths(3))

    assert plan.mode == "leave_complete_flights_out"
    assert len(plan.training) == 2
    assert [flight.path for flight in plan.validation] == ["flight_2.npz"]
    assert not set(plan.training_source_groups).isdisjoint(plan.validation_group_order)


def test_group_holdout_can_split_on_another_label(flights) -> None:
    trajectories = flights(4, vehicle_id=("a", "a", "b", "b"), source_group=("s",) * 4)

    plan = Holdout.by_group(key="vehicle_id").plan(trajectories, _paths(4))

    assert plan.mode == "leave_source_groups_out"
    assert [flight.path for flight in plan.validation] == [
        "flight_2.npz",
        "flight_3.npz",
    ]


def test_paths_default_to_positional_placeholders(flights) -> None:
    plan = Holdout.by_group().plan(flights(2))

    assert plan.training_labels == ("trajectory_0",)
    assert [flight.path for flight in plan.validation] == ["trajectory_1"]


def test_paths_must_label_every_trajectory(flights) -> None:
    with pytest.raises(ValueError, match="paths must label every trajectory"):
        Holdout.by_group().plan(flights(2), ["only-one.npz"])


def test_planning_requires_at_least_one_trajectory() -> None:
    with pytest.raises(ValueError, match="at least one trajectory is required"):
        Holdout.by_group().plan([])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"rule": "nope"}, "holdout rule must be one of"),
        ({"rule": "group", "count": 0}, "holdout count must be at least one"),
        ({"rule": "temporal", "fraction": 1.0}, "fraction must be between"),
        ({"rule": "group", "key": " "}, "holdout key cannot be empty"),
        ({"rule": "label", "values": (" ",)}, "holdout values cannot be empty"),
    ),
)
def test_holdout_validates_its_arguments(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Holdout(**kwargs)


def test_holdout_serializes_only_the_arguments_its_rule_uses() -> None:
    assert Holdout.by_group(2).to_dict() == {
        "rule": "group",
        "key": "source_group",
        "count": 2,
    }
    assert Holdout.by_label("profile", ("hover", "hover")).to_dict() == {
        "rule": "label",
        "key": "profile",
        "values": ["hover"],
    }
    assert Holdout.temporal(0.6).to_dict() == {"rule": "temporal", "fraction": 0.6}


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"evaluation_horizons_s": (0.0,)}, "evaluation horizons must be positive"),
        ({"horizons_s": (-1.0,)}, "training horizons must be positive"),
        ({"model_class": "nope"}, "model_class must be structured"),
        ({"ablations": ("nope",)}, "unknown ablation"),
        (
            {
                "fixed_response_time_constant_s": 0.05,
                "ablations": ("no_lag",),
            },
            "the no-lag ablation does not apply",
        ),
    ),
)
def test_fit_spec_validates_its_knobs(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        FitSpec(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"endpoint_weight": 0.5}, "endpoint_weight must be at least one"),
        (
            {"stability_regularization": -1.0},
            "stability_regularization must be nonnegative",
        ),
    ),
)
def test_loss_policy_validates_its_knobs(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        LossPolicy(**kwargs)


def test_weighting_policy_rejects_impossible_group_weights() -> None:
    with pytest.raises(ValueError, match="group_weights values must be finite"):
        WeightingPolicy(group_weights={"a": -1.0})


def test_fit_spec_normalizes_sequence_knobs_to_tuples() -> None:
    request = FitSpec(
        evaluation_horizons_s=[0.1, 0.5],
        horizons_s=[0.2],
        ablations=["no_lag", "no_lag"],
    )

    assert request.evaluation_horizons_s == (0.1, 0.5)
    assert request.horizons_s == (0.2,)
    assert request.ablations == ("no_lag",)


def test_fit_spec_stride_defaults_to_the_horizon() -> None:
    assert FitSpec().stride_for(25) == 25
    assert FitSpec(stride=3).stride_for(25) == 3
