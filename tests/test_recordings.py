"""Signal timing, segmentation and integrity at the public recording boundary."""

from dataclasses import replace

import numpy as np
import pytest

from glassbox import (
    STATE_CHANNELS,
    SequenceCollection,
    SequenceSegment,
    segments_from_mask,
)
from glassbox._learner_arrays import load_arrays, save_arrays
from glassbox.io.recordings import (
    concatenate_recordings,
    load_recordings,
    save_recordings,
)


def collection(name="one", offset=0):
    x = np.zeros((14, 15))
    x[:, 0] = np.arange(14) + offset
    x[:, 6:] = np.eye(3).reshape(9)
    u = np.arange(26).reshape(13, 2) * 0.01
    return SequenceCollection(
        (SequenceSegment(name, "whole", x, u, 0.05),),
        "rig",
        STATE_CHANNELS,
        ("u0", "u1"),
    )


def test_window_timing_and_provenance():
    r = collection()
    keys = r.window_keys(history_steps=3, horizon_steps=2)
    windows = r.extract(keys, history_steps=3, horizon_steps=2)
    np.testing.assert_array_equal(
        windows.batch.past_states[0], r.segments[0].states[:4]
    )
    np.testing.assert_array_equal(
        windows.batch.future_states[0], r.segments[0].states[4:6]
    )
    np.testing.assert_array_equal(
        windows.batch.future_inputs[0], r.segments[0].inputs[3:5]
    )
    assert windows.source_origins[0] == 3
    assert windows.coverage()["one"]["unique_input_rows"] == 13


def test_missing_rows_split_without_padding():
    s = collection().segments[0]
    valid = np.ones(len(s.states), bool)
    valid[6] = False
    segments = segments_from_mask("one", s.states, s.inputs, valid, dt_s=0.05)
    assert [(s.start_row, len(s.states)) for s in segments] == [(0, 6), (7, 7)]
    r = SequenceCollection(segments)
    windows = r.extract(
        r.window_keys(history_steps=2, horizon_steps=1),
        history_steps=2,
        horizon_steps=1,
    )
    assert np.all(np.diff(windows.batch.past_states[..., 0], axis=1) == 1)


def test_segments_copy_arrays_and_reject_overlap():
    r = collection()
    assert not r.segments[0].states.flags.writeable
    with pytest.raises(ValueError, match="overlap"):
        SequenceCollection(
            (r.segments[0], replace(r.segments[0], segment_id="other", start_row=10))
        )


@pytest.mark.parametrize(
    "field,value",
    [("dt_s", 0), ("states", np.full((14, 15), np.nan)), ("inputs", np.zeros((14, 2)))],
)
def test_invalid_recording_rejected(field, value):
    with pytest.raises(ValueError):
        replace(collection().segments[0], **{field: value})


def test_recording_roundtrip_and_concatenation(tmp_path):
    path = tmp_path / "recording.npz"
    r = collection()
    save_recordings(r, path)
    loaded = load_recordings(path)
    np.testing.assert_array_equal(loaded.segments[0].states, r.segments[0].states)
    combined = concatenate_recordings((loaded, collection("two", 1)))
    assert len(combined.segments) == 2
    with pytest.raises(ValueError, match="configuration"):
        concatenate_recordings(
            (loaded, replace(collection("two"), configuration_id="other"))
        )


def test_corruption_and_repaired_unknown_arrays_rejected(tmp_path):
    path = tmp_path / "recording.npz"
    save_recordings(collection(), path)
    metadata, arrays = load_arrays(path)
    arrays["unknown"] = np.zeros(1)
    save_arrays(path, metadata, arrays)
    with pytest.raises(ValueError, match="arrays"):
        load_recordings(path)
