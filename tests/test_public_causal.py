"""Public lifecycle of the single episode-fitted causal dynamics model."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from glassbox import (
    STATE_CHANNELS,
    LearnedDynamics,
    OnlineFit,
    SequenceCollection,
    SequenceSegment,
    fit,
)
from glassbox._learner_arrays import load_arrays, save_arrays


def recordings(prefix="source", count=4):
    segments = []
    dt_s, steps = 0.02, 40
    for i in range(count):
        time = np.arange(steps) * dt_s + (0.13 if prefix == "fresh" else 0.0)
        commands = np.stack(
            (
                0.3 + 0.1 * np.sin(2 * time + 0.2 * i),
                0.2 + 0.1 * np.cos(3 * time + 0.3 * i),
            ),
            axis=-1,
        )
        states = np.zeros((steps + 1, 15))
        states[:, 6:] = np.eye(3).ravel()
        for row, command in enumerate(commands):
            states[row + 1, :3] = states[row, :3] + dt_s * np.array(
                [2 * command[0], command[1], -9.80665]
            )
        segments.append(
            SequenceSegment(f"{prefix}-{i}", "whole", states, commands, dt_s)
        )
    return SequenceCollection(
        tuple(segments), "opaque-system", STATE_CHANNELS, ("A [1]", "B [1]")
    )


def test_fit_predict_update_and_self_contained_revision(tmp_path):
    source = recordings()
    model = fit(source)
    segment = source.segments[0]
    x, past, future = segment.states[:31], segment.inputs[:30], segment.inputs[30:35]
    with jax.enable_x64(True):
        prediction = np.asarray(model.predict(x, past, future))
        jacobian = jax.jacfwd(
            lambda controls: model.predict(x, past, controls)[-1, :3]
        )(jnp.asarray(future))
    assert prediction.shape == (5, 15)
    assert np.isfinite(jacobian).all()
    assert np.linalg.norm(jacobian) > 0
    assert model.report["envelope"]["available"] is False
    with pytest.raises(ValueError, match="no calibrated"):
        model.envelope()

    path = tmp_path / "model.npz"
    model.save(path)
    restored = LearnedDynamics.load(path)
    assert restored.fingerprint() == model.fingerprint()
    with jax.enable_x64(True):
        np.testing.assert_array_equal(restored.predict(x, past, future), prediction)

    old_fingerprint = model.fingerprint()
    revised = model.update(recordings("fresh", 2))
    assert revised.report["previous_revision"] == old_fingerprint
    assert model.fingerprint() == old_fingerprint
    assert revised.fingerprint() != old_fingerprint
    with pytest.raises(ValueError, match="reuses"):
        model.update(source)
    with pytest.raises(ValueError, match="differ"):
        model.update(replace(recordings("other", 2), configuration_id="wrong"))


def test_single_recording_fit_uses_all_available_observations():
    model = fit(recordings(count=1))
    assert model.report["training_recordings"] == ["source-0"]
    assert model.report["optimization"]["completed_transitions"] == 40


@pytest.mark.parametrize("x64", [False, True])
def test_recording_forecasts_share_one_trace_without_changing_predictions(x64):
    source = recordings(count=1)
    model = fit(source)
    segment = source.segments[0]
    origins = (10, 20, 30)
    horizon = 3
    with jax.enable_x64(x64):
        fast = np.asarray(model._predict_origins(segment, origins, horizon))
        direct = np.stack(
            [
                np.asarray(
                    model.predict(
                        segment.states[: origin + 1],
                        segment.inputs[:origin],
                        segment.inputs[origin : origin + horizon],
                    )
                )
                for origin in origins
            ]
        )
    np.testing.assert_allclose(fast, direct, rtol=0, atol=2e-12 if x64 else 2e-5)


def test_rechecksummed_parameter_tampering_is_rejected(tmp_path):
    model = fit(recordings())
    path = tmp_path / "good.npz"
    model.save(path)
    metadata, arrays = load_arrays(path)
    arrays["model_force"][0, 0] += 0.1
    changed = tmp_path / "changed.npz"
    save_arrays(changed, metadata, arrays)
    with pytest.raises(ValueError, match="fingerprint"):
        LearnedDynamics.load(changed)


def test_online_background_publication_and_resume(tmp_path):
    segment = recordings().segments[0]
    prefix = SequenceCollection(
        (replace(segment, states=segment.states[:31], inputs=segment.inputs[:30]),),
        "opaque-system",
        STATE_CHANNELS,
        ("A [1]", "B [1]"),
    )
    session = OnlineFit(prefix)
    try:
        initial = session.model
        for row in (30, 31):
            session.observe(row, segment.inputs[row], segment.states[row + 1])
        for _ in range(2):
            session.wait_for_publication(timeout=3)
        assert session.published_cursor == session.cursor == 32
        assert initial.fingerprint != session.model.fingerprint
        path = tmp_path / "episode.npz"
        session.save(path)
        resumed = OnlineFit.load(path)
        try:
            assert resumed.fingerprint() == session.fingerprint()
            prediction = session.predict(
                segment.states[:33], segment.inputs[:32], segment.inputs[32:34]
            )
            np.testing.assert_array_equal(
                resumed.predict(
                    segment.states[:33], segment.inputs[:32], segment.inputs[32:34]
                ),
                prediction,
            )
            resumed.observe(32, segment.inputs[32], segment.states[33])
            resumed.wait_for_publication(timeout=3)
            assert resumed.published_cursor == 33
        finally:
            resumed.close()
    finally:
        session.close()
