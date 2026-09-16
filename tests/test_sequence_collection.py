"""Boundary and coverage contracts for sampled observation collections."""

import numpy as np
import pytest

from glassbox.experimental.sequence_collection import (
    SequenceCollection,
    SequenceSegment,
    SequenceWindows,
    WindowKey,
    segments_from_mask,
)


def collection():
    states = np.arange(16.0)[:, None]
    inputs = (np.arange(15.0) + 100)[:, None]
    valid = np.ones(16, dtype=bool)
    valid[6:8] = False
    states[6:8] = np.nan
    return SequenceCollection(segments_from_mask("r", states, inputs, valid, dt_s=0.1))


def test_mask_extracts_only_complete_windows_with_provenance():
    c = collection()
    keys = c.window_keys(history_steps=1, horizon_steps=2)
    assert len(keys) == 8
    chosen = [keys[0], keys[1], keys[3]]
    windows = c.extract(chosen, history_steps=1, horizon_steps=2)
    assert windows.source_origins == (1, 2, 9)
    np.testing.assert_array_equal(
        windows.batch.past_states[..., 0], [[0, 1], [1, 2], [8, 9]]
    )
    np.testing.assert_array_equal(
        windows.batch.future_states[..., 0], [[2, 3], [3, 4], [10, 11]]
    )
    np.testing.assert_array_equal(
        windows.batch.future_inputs[..., 0], [[101, 102], [102, 103], [109, 110]]
    )
    assert windows.coverage() == {
        "r": dict(
            windows=3,
            segments=2,
            unique_state_rows=9,
            unique_input_rows=7,
            observed_transition_time_s=pytest.approx(0.7),
        )
    }
    assert not windows.batch.past_states.flags.writeable


def test_recording_identity_distinguishes_same_source_rows():
    a = SequenceSegment("a", "s", np.arange(6)[:, None], np.zeros((5, 1)), 0.1)
    b = SequenceSegment("b", "s", a.states, a.inputs, 0.1)
    c = SequenceCollection((a, b))
    windows = c.extract(
        c.window_keys(history_steps=1, horizon_steps=1),
        history_steps=1,
        horizon_steps=1,
    )
    assert {n: r["unique_state_rows"] for n, r in windows.coverage().items()} == {
        "a": 6,
        "b": 6,
    }


@pytest.mark.parametrize(
    "kind", ["gap", "duplicate", "unknown", "fractional", "boolean"]
)
def test_invalid_window_keys_rejected(kind):
    c = collection()
    keys = list(c.window_keys(history_steps=1, horizon_steps=2)[:3])
    k = keys[0]
    keys[0] = {
        "gap": WindowKey(k.recording_id, k.segment_id, 4),
        "duplicate": keys[1],
        "unknown": WindowKey("missing", k.segment_id, 1),
        "fractional": WindowKey(k.recording_id, k.segment_id, 1.5),
        "boolean": WindowKey(k.recording_id, k.segment_id, True),
    }[kind]
    with pytest.raises(ValueError):
        c.extract(keys, history_steps=1, horizon_steps=2)


@pytest.mark.parametrize("kind", ["duplicate", "overlap", "dt", "channels"])
def test_inconsistent_segment_collections_rejected(kind):
    a = collection().segments[0]
    b = SequenceSegment(
        "r",
        a.segment_id if kind == "duplicate" else "other",
        np.zeros((6, 2 if kind == "channels" else 1)),
        np.zeros((5, 1)),
        0.2 if kind == "dt" else 0.1,
        3 if kind == "overlap" else 8,
    )
    with pytest.raises(ValueError):
        SequenceCollection((a, b))


def test_mask_requires_boolean_and_finite_retained_rows():
    with pytest.raises(ValueError):
        segments_from_mask(
            "r", np.zeros((4, 1)), np.zeros((3, 1)), np.ones(4), dt_s=0.1
        )
    with pytest.raises(ValueError):
        segments_from_mask(
            "r",
            np.full((4, 1), np.nan),
            np.zeros((3, 1)),
            np.ones(4, dtype=bool),
            dt_s=0.1,
        )


def excited_collection():
    """The same masked recording, with a declared excitation beside its inputs."""
    states = np.arange(16.0)[:, None]
    inputs = (np.arange(15.0) + 100)[:, None]
    excitation = np.sin(np.arange(15.0))[:, None]
    valid = np.ones(16, dtype=bool)
    valid[6:8] = False
    states[6:8] = np.nan
    return (
        SequenceCollection(
            segments_from_mask(
                "r", states, inputs, valid, dt_s=0.1, excitation=excitation
            )
        ),
        excitation,
    )


@pytest.mark.parametrize("kind", ["short", "wide", "nonfinite", "ragged"])
def test_declared_excitation_is_validated_like_the_other_arrays(kind):
    states, inputs = np.arange(6.0)[:, None], np.zeros((5, 2))
    excitation = {
        "short": np.zeros((4, 2)),
        "wide": np.zeros((5, 3)),
        "nonfinite": np.full((5, 2), np.nan),
        "ragged": np.zeros(5),
    }[kind]
    with pytest.raises(ValueError):
        SequenceSegment("r", "s", states, inputs, 0.1, 0, excitation)
    with pytest.raises(ValueError):
        segments_from_mask(
            "r",
            states,
            inputs,
            np.ones(6, dtype=bool),
            dt_s=0.1,
            excitation=excitation,
        )


def test_excitation_survives_masking_and_window_extraction():
    c, excitation = excited_collection()
    assert c.excitation_declared
    # Masking cuts the excitation with the inputs it belongs to, row for row.
    assert [s.segment_id for s in c.segments] == ["rows-0-6", "rows-8-16"]
    np.testing.assert_array_equal(c.segments[0].excitation, excitation[:5])
    np.testing.assert_array_equal(c.segments[1].excitation, excitation[8:15])
    keys = c.window_keys(history_steps=1, horizon_steps=2)
    windows = c.extract([keys[0], keys[1], keys[3]], history_steps=1, horizon_steps=2)
    assert windows.excitation_declared
    assert windows.past_excitation.shape == windows.batch.past_inputs.shape
    assert windows.future_excitation.shape == windows.batch.future_inputs.shape
    np.testing.assert_array_equal(
        windows.future_excitation[..., 0],
        [excitation[1:3, 0], excitation[2:4, 0], excitation[9:11, 0]],
    )
    assert not windows.past_excitation.flags.writeable


def test_undeclared_excitation_stays_absent_everywhere():
    c = collection()
    assert not c.excitation_declared
    keys = c.window_keys(history_steps=1, horizon_steps=2)
    windows = c.extract(keys[:3], history_steps=1, horizon_steps=2)
    assert windows.past_excitation is None and windows.future_excitation is None
    assert not windows.excitation_declared


def test_a_collection_declares_excitation_for_every_recording_or_none():
    states, inputs = np.arange(6.0)[:, None], np.zeros((5, 1))
    declared = SequenceSegment("a", "s", states, inputs, 0.1, 0, np.ones((5, 1)))
    silent = SequenceSegment("b", "s", states, inputs, 0.1)
    with pytest.raises(ValueError):
        SequenceCollection((declared, silent))


def test_window_excitation_covers_both_input_arrays_or_neither():
    c, _ = excited_collection()
    keys = c.window_keys(history_steps=1, horizon_steps=2)
    windows = c.extract(keys[:3], history_steps=1, horizon_steps=2)
    with pytest.raises(ValueError):
        SequenceWindows(
            windows.batch, windows.keys, windows.source_origins, windows.past_excitation
        )
    with pytest.raises(ValueError):
        SequenceWindows(
            windows.batch,
            windows.keys,
            windows.source_origins,
            windows.future_excitation,
            windows.past_excitation,
        )
