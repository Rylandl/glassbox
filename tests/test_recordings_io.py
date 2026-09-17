"""Generic archive identity, channel contracts and canonical adapter causality."""

import json
from dataclasses import replace

import numpy as np
import pytest

from glassbox import SequenceCollection, SequenceSegment
from glassbox.io.recordings import (
    concatenate_recordings,
    from_trajectories,
    load_recordings,
    save_recordings,
)


def collection(*segments):
    return SequenceCollection(
        segments, "machine-v1", ("position [m]",), ("force [N,command]",)
    )


def segment(name="a", start=0, declared=True):
    return SequenceSegment(
        name,
        f"rows-{start}",
        np.arange(7.0)[:, None],
        np.arange(6.0)[:, None],
        0.1,
        start,
        np.ones((6, 1)) if declared else None,
    )


def test_roundtrip_preserves_segments_contracts_and_excitation(tmp_path):
    original = collection(segment(start=0), segment(start=10), segment("b"))
    path = tmp_path / "arbitrary-extension"
    save_recordings(original, path)
    restored = load_recordings(path)
    assert restored.configuration_id == original.configuration_id
    assert restored.state_channels == original.state_channels
    assert restored.input_channels == original.input_channels
    for saved, loaded in zip(original.segments, restored.segments, strict=True):
        for name in ("recording_id", "segment_id", "start_row", "dt_s"):
            assert getattr(saved, name) == getattr(loaded, name)
        for name in ("states", "inputs", "excitation"):
            np.testing.assert_array_equal(getattr(saved, name), getattr(loaded, name))
            assert not getattr(loaded, name).flags.writeable


@pytest.mark.parametrize("field", ["array", "configuration", "channel", "boundary"])
def test_archive_rejects_altered_arrays_and_facts(tmp_path, field):
    path = tmp_path / "recordings.npz"
    save_recordings(collection(segment()), path)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    if field == "array":
        arrays["segment_0_inputs"][0, 0] += 1
    else:
        meta = json.loads(str(arrays["metadata"]))
        if field == "configuration":
            meta["contract"]["configuration_id"] = "another-machine"
        elif field == "channel":
            meta["contract"]["state_channels"] = ["position [cm]"]
        else:
            meta["segments"][0]["start_row"] = 10
        arrays["metadata"] = json.dumps(meta)
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="fingerprint"):
        load_recordings(path)


def test_concatenation_checks_contracts_and_preserves_recording_boundaries():
    first, second = collection(segment()), collection(segment("b"))
    joined = concatenate_recordings([first, second])
    assert [s.recording_id for s in joined.segments] == ["a", "b"]
    with pytest.raises(ValueError, match="configuration, channels"):
        concatenate_recordings(
            [first, replace(second, state_channels=("position [cm]",))]
        )
    with pytest.raises(ValueError, match="duplicate"):
        concatenate_recordings([first, first])
    with pytest.raises(ValueError, match="at least one"):
        concatenate_recordings([])


@pytest.mark.parametrize("gapped", [False, True])
def test_adapter_matches_existing_harness_coordinates_and_command_alignment(
    quadrotor_flight, gapped
):
    from glassbox.experimental.harness import (
        observed_rows,
        trajectory_segments,
    )

    trajectory = quadrotor_flight(41)
    if gapped:
        times = trajectory.time_s.copy()
        times[10:] += 0.5
        trajectory = replace(trajectory, time_s=times)
    adapted = from_trajectories(
        {"flight": trajectory}, configuration_id="declared-machine"
    )
    expected = trajectory_segments(
        "flight", trajectory, dt_s=trajectory.nominal_dt_s, tolerance_fraction=1e-6
    )
    assert adapted.configuration_id == "declared-machine"
    assert adapted.input_channels == tuple(
        f"{c.name} [{c.unit},{c.frame or 'unframed'},{c.semantic},{c.role}]"
        for c in trajectory.spec.controls
    )
    assert len(adapted.segments) == (2 if gapped else 1)
    for actual, wanted in zip(adapted.segments, expected, strict=True):
        assert (actual.recording_id, actual.segment_id, actual.start_row) == (
            wanted.recording_id,
            wanted.segment_id,
            wanted.start_row,
        )
        np.testing.assert_array_equal(actual.states, wanted.states)
        np.testing.assert_array_equal(actual.inputs, wanted.inputs)
        a = actual.start_row
        np.testing.assert_array_equal(
            actual.states, observed_rows(trajectory)[a : a + len(actual.states)]
        )
        np.testing.assert_array_equal(
            actual.inputs, trajectory.controls[a : a + len(actual.inputs)]
        )


def test_adapter_refuses_to_silently_drop_a_recording(quadrotor_flight):
    first = quadrotor_flight(41)
    second = replace(quadrotor_flight(42), time_s=first.time_s * 2)
    with pytest.raises(ValueError, match=r"'wrong-grid'.*no valid segment"):
        from_trajectories(
            {"good": first, "wrong-grid": second}, configuration_id="declared"
        )


@pytest.mark.parametrize("dt_s,horizon", [(0.05, 5), (0.1, 2)])
def test_adapter_grid_is_stable_across_separately_converted_recording_lengths(
    quadrotor_flight, dt_s, horizon
):
    from glassbox.learner import _contract, steps_for

    short = from_trajectories(
        {"short": quadrotor_flight(1, 0.8, dt_s)}, configuration_id="same-grid"
    )
    long = from_trajectories(
        {"long": quadrotor_flight(2, 3.0, dt_s)}, configuration_id="same-grid"
    )
    assert _contract(short) == _contract(long)
    assert short.segments[0].dt_s == dt_s
    assert steps_for(short.segments[0].dt_s)["horizon"] == horizon
    assert len(concatenate_recordings([short, long]).segments) == 2


def test_generic_archive_does_not_round_explicit_sample_interval(tmp_path):
    dt_s = np.nextafter(0.1, 0.0)
    original = collection(replace(segment(), dt_s=dt_s))
    path = tmp_path / "precise.npz"
    save_recordings(original, path)
    assert load_recordings(path).segments[0].dt_s == dt_s


def test_adapter_splits_nonfinite_observed_rows_without_command_shift(quadrotor_flight):
    # Defect injection checks the adapter's mask even though normal canonical
    # construction already rejects nonfinite arrays.
    trajectory = replace(quadrotor_flight(43))
    states = trajectory.states.copy()
    states[8, 3] = np.nan
    object.__setattr__(trajectory, "states", states)
    adapted = from_trajectories({"injected": trajectory}, configuration_id="declared")
    assert [segment.start_row for segment in adapted.segments] == [0, 9]
    for segment in adapted.segments:
        start = segment.start_row
        np.testing.assert_array_equal(
            segment.inputs, trajectory.controls[start : start + len(segment.inputs)]
        )
        assert np.isfinite(segment.states).all()


@pytest.mark.parametrize(
    "change", [{"semantic": "measured_actuation"}, {"frame": "FLU"}]
)
def test_separately_converted_command_semantics_and_frames_remain_distinct(
    quadrotor_flight, change
):
    from glassbox.learner import _contract

    trajectory = quadrotor_flight(44)
    channels = list(trajectory.spec.channels)
    channels[0] = replace(channels[0], **change)
    modified = replace(
        trajectory, spec=replace(trajectory.spec, channels=tuple(channels))
    )
    original = from_trajectories(
        {"original": trajectory}, configuration_id="same-machine"
    )
    changed = from_trajectories({"changed": modified}, configuration_id="same-machine")
    assert _contract(original) != _contract(changed)
    with pytest.raises(ValueError, match="configuration, channels"):
        concatenate_recordings([original, changed])
