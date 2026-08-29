import numpy as np
import pytest

from glassbox.data import (
    ControlChannel,
    ExogenousChannel,
    RIGID_BODY_STATE_SCHEMA,
    Trajectory,
    TrajectorySpec,
    VehicleConfigurationSpec,
    duration_to_steps,
    load_trajectory_npz,
    make_trajectory_spec,
    save_trajectory_npz,
    split_trajectory,
    trajectory_windows,
)


def test_duration_to_steps_is_stable_at_half_sample_boundary() -> None:
    assert duration_to_steps(0.5, 0.2) == 2
    assert duration_to_steps(0.5, 0.1999999999999993) == 2
    assert duration_to_steps(0.5, 0.200000000000001) == 2
    assert duration_to_steps(0.1, 0.020000000000001) == 5


def make_trajectory(
    intervals: int = 10,
    dt_s: float = 0.02,
    control_size: int = 4,
    control_names: tuple[str, ...] | None = None,
) -> Trajectory:
    states = np.zeros((intervals + 1, 13))
    states[:, 6] = 1.0
    names = (
        tuple(f"control_{index}" for index in range(control_size))
        if control_names is None
        else control_names
    )
    return Trajectory(
        time_s=np.arange(intervals + 1) * dt_s,
        states=states,
        controls=np.zeros((intervals, control_size)),
        spec=make_trajectory_spec(
            names,
            family="test_vehicle",
            observation_source="simulator_truth",
        ),
    )


def test_trajectory_rejects_mismatched_controls() -> None:
    with pytest.raises(ValueError, match="controls must have shape"):
        Trajectory(
            time_s=np.arange(5, dtype=float),
            states=np.zeros((5, 13)),
            controls=np.zeros((5, 4)),
            spec=make_trajectory_spec(
                ("a", "b", "c", "d"),
                family="test_vehicle",
                observation_source="simulator_truth",
            ),
        )


def test_windows_do_not_cross_trajectory_boundaries() -> None:
    windows = trajectory_windows(
        [make_trajectory(), make_trajectory()], horizon=5, stride=5
    )

    assert windows.initial_states.shape == (4, 13)
    assert windows.control_histories.shape == (4, 50, 4)
    assert windows.controls.shape == (4, 5, 4)
    assert windows.target_states.shape == (4, 6, 13)
    assert windows.dt_s == pytest.approx(0.02)


def test_windows_support_named_variable_control_channels() -> None:
    names = ("throttle", "aileron", "elevator", "rudder", "flap", "spoiler")
    trajectories = [
        make_trajectory(control_size=6, control_names=names),
        make_trajectory(control_size=6, control_names=names),
    ]

    windows = trajectory_windows(trajectories, horizon=5, stride=5)

    assert windows.controls.shape == (4, 5, 6)
    assert windows.control_histories.shape == (4, 50, 6)
    assert windows.control_size == 6
    assert windows.control_names == names


def test_windows_reject_mixed_control_schemas() -> None:
    with pytest.raises(ValueError, match="ordered control_names"):
        trajectory_windows(
            [
                make_trajectory(control_names=("a", "b", "c", "d")),
                make_trajectory(control_names=("b", "a", "c", "d")),
            ],
            horizon=5,
        )


def test_balanced_windows_give_each_trajectory_equal_total_weight() -> None:
    windows = trajectory_windows(
        [make_trajectory(intervals=10), make_trajectory(intervals=20)],
        horizon=5,
        stride=5,
        balance_trajectories=True,
    )

    assert windows.trajectory_indices is not None
    assert windows.window_weights is not None
    assert np.sum(windows.trajectory_indices == 0) == 2
    assert np.sum(windows.trajectory_indices == 1) == 4
    assert np.sum(windows.window_weights[windows.trajectory_indices == 0]) == pytest.approx(1.0)
    assert np.sum(windows.window_weights[windows.trajectory_indices == 1]) == pytest.approx(1.0)


def test_explicit_trajectory_weights_are_distributed_over_each_flights_windows() -> None:
    windows = trajectory_windows(
        [
            make_trajectory(intervals=10),
            make_trajectory(intervals=20),
            make_trajectory(intervals=15),
        ],
        horizon=5,
        stride=5,
        trajectory_weights=(0.5, 0.5, 1.0),
    )

    assert windows.trajectory_indices is not None
    assert windows.window_weights is not None
    totals = [
        np.sum(windows.window_weights[windows.trajectory_indices == index])
        for index in range(3)
    ]
    assert totals == pytest.approx([0.5, 0.5, 1.0])


def test_group_balancing_is_uniform_per_window_within_equal_groups() -> None:
    windows = trajectory_windows(
        [
            make_trajectory(intervals=10),
            make_trajectory(intervals=20),
            make_trajectory(intervals=15),
        ],
        horizon=5,
        stride=5,
        trajectory_groups=("session-a", "session-a", "session-b"),
    )

    assert windows.trajectory_indices is not None
    assert windows.window_weights is not None
    first_group = windows.trajectory_indices < 2
    second_group = windows.trajectory_indices == 2
    assert np.sum(windows.window_weights[first_group]) == pytest.approx(1.0)
    assert np.sum(windows.window_weights[second_group]) == pytest.approx(1.0)
    assert np.unique(windows.window_weights[first_group]).size == 1


def test_window_cap_is_deterministic_and_spans_each_source_group() -> None:
    trajectories = [
        make_trajectory(intervals=100),
        make_trajectory(intervals=200),
        make_trajectory(intervals=80),
    ]
    arguments = {
        "horizon": 5,
        "stride": 5,
        "trajectory_groups": ("session-a", "session-a", "session-b"),
        "maximum_windows": 12,
    }

    first = trajectory_windows(trajectories, **arguments)
    second = trajectory_windows(trajectories, **arguments)

    assert first.selection_policy == "deterministic_stratified_midpoint"
    assert first.candidate_window_count == 76
    np.testing.assert_array_equal(first.candidate_window_counts, [20, 40, 16])
    np.testing.assert_array_equal(first.trajectory_indices, second.trajectory_indices)
    np.testing.assert_array_equal(first.start_indices, second.start_indices)
    assert len(first.initial_states) == 12
    first_group = first.trajectory_indices < 2
    second_group = first.trajectory_indices == 2
    assert np.sum(first_group) == 6
    assert np.sum(second_group) == 6
    assert np.sum(first.window_weights[first_group]) == pytest.approx(1.0)
    assert np.sum(first.window_weights[second_group]) == pytest.approx(1.0)
    assert np.min(first.start_indices[first_group]) == 25
    assert np.max(first.start_indices[first_group]) == 175
    assert np.min(first.start_indices[second_group]) == 5
    assert np.max(first.start_indices[second_group]) == 70


def test_window_cap_must_be_positive() -> None:
    with pytest.raises(ValueError, match="maximum_windows must be positive"):
        trajectory_windows([make_trajectory()], horizon=5, maximum_windows=0)


def test_npz_round_trip(tmp_path) -> None:
    trajectory = make_trajectory()
    trajectory = Trajectory(
        time_s=trajectory.time_s,
        states=trajectory.states,
        controls=trajectory.controls,
        spec=make_trajectory_spec(
            ("throttle", "aileron", "elevator", "rudder"),
            family="fixedwing",
            observation_source="simulator_truth",
            configuration_id="test-plane",
            exogenous=(
                ExogenousChannel(
                    name="wind_north_m_s",
                    role="wind_north",
                    semantic="estimated_environment_at_prediction_start",
                    unit="m/s",
                    frame="NWU",
                ),
            ),
        ),
        exogenous=np.linspace(1.0, 2.0, len(trajectory.time_s))[:, None],
        labels={"profile": "test"},
        provenance={"adapter": {"name": "test", "schema_version": 1}},
    )
    path = tmp_path / "trajectory.npz"

    save_trajectory_npz(trajectory, path)
    restored = load_trajectory_npz(path)

    np.testing.assert_array_equal(restored.time_s, trajectory.time_s)
    np.testing.assert_array_equal(restored.states, trajectory.states)
    np.testing.assert_array_equal(restored.controls, trajectory.controls)
    np.testing.assert_array_equal(restored.exogenous, trajectory.exogenous)
    assert restored.control_names == trajectory.control_names
    assert restored.spec == trajectory.spec
    assert restored.labels == trajectory.labels
    assert restored.provenance == trajectory.provenance

    with np.load(path, allow_pickle=False) as archive:
        assert int(archive["format_version"]) == 2
        np.testing.assert_array_equal(archive["exogenous"], trajectory.exogenous)
        assert set(archive.files) == {
            "format_version",
            "time_s",
            "states",
            "controls",
            "exogenous",
            "spec_json",
            "labels_json",
            "provenance_json",
        }


def test_typed_spec_labels_and_provenance_round_trip(tmp_path) -> None:
    base = make_trajectory(control_size=2)
    spec = TrajectorySpec(
        state_schema=RIGID_BODY_STATE_SCHEMA,
        observation_source="mocap",
        controls=(
            ControlChannel(
                name="throttle",
                role="throttle",
                semantic="normalized_command",
                unit="1",
                minimum=0.0,
                maximum=1.0,
            ),
            ControlChannel(
                name="roll_command",
                role="roll",
                semantic="normalized_generalized_command",
                unit="1",
                minimum=-1.0,
                maximum=1.0,
                frame="FLU",
            ),
        ),
        vehicle=VehicleConfigurationSpec(
            family="fixedwing",
            configuration_id="flying-wing-01",
            controlled_axes=("roll",),
            propulsion="single_propeller",
            fixed_states={"flap": 0.0},
            auxiliary_controls=("flap",),
        ),
    )
    trajectory = Trajectory(
        time_s=base.time_s,
        states=base.states,
        controls=base.controls,
        spec=spec,
        labels={"profile": "roll_steps", "replicate": 2},
        provenance={
            "source": "fixture.csv",
            "adapter": {"name": "csv", "schema_version": 1},
        },
    )
    path = tmp_path / "typed.npz"

    save_trajectory_npz(trajectory, path)
    restored = load_trajectory_npz(path)

    assert restored.spec == spec
    assert restored.control_names == ("throttle", "roll_command")
    assert restored.labels == {"profile": "roll_steps", "replicate": 2}
    assert restored.provenance == trajectory.provenance


def test_rejects_noncurrent_trajectory_format(tmp_path) -> None:
    path = tmp_path / "old.npz"
    np.savez_compressed(
        path,
        format_version=np.asarray(1, dtype=np.int64),
    )

    with pytest.raises(ValueError, match="unsupported trajectory format version: 1"):
        load_trajectory_npz(path)


def test_split_trajectory_preserves_the_boundary_state() -> None:
    trajectory = make_trajectory(intervals=10)

    training, validation = split_trajectory(trajectory, train_fraction=0.6)

    assert len(training.controls) == 6
    assert len(validation.controls) == 4
    np.testing.assert_array_equal(training.states[-1], validation.states[0])
    assert validation.time_s[0] == 0.0
