"""Lifecycle evidence with analytic recording fixtures, not physical science."""

import json
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from glassbox import learner as vehicle
from glassbox._learner_arrays import load_arrays, save_arrays
from glassbox.recordings import SequenceCollection, SequenceSegment


def recordings(prefix="train", m=3, count=4):
    segments = []
    t = np.arange(41) * 0.05 + (0.13 if prefix == "fresh" else 0.0)
    for i in range(count):
        u = np.stack([np.sin(t[:-1] * (j + 1) + i * 0.3) for j in range(m)], axis=-1)
        x = np.zeros((len(t), 15))
        x[:, :3] = np.stack(
            [0.2 * t + 0.01 * i, 0.03 * t**2, 0.1 * np.sin(t + i)], axis=-1
        )
        x[:, 6:] = np.eye(3).reshape(9)
        segments.append(SequenceSegment(f"{prefix}-{i}", "whole", x, u, 0.05))
    return SequenceCollection(
        tuple(segments),
        configuration_id="opaque-system",
        state_channels=vehicle.STATE_CHANNELS,
        input_channels=tuple(f"command-{j} [unitless]" for j in range(m)),
    )


@pytest.fixture
def tiny_fit(monkeypatch):
    original = vehicle.fit_sequence

    def quick(*args, **kwargs):
        return original(*args, **kwargs, _steps=0)

    monkeypatch.setattr(vehicle, "fit_sequence", quick)
    return vehicle.fit(recordings())


def test_lifecycle_roundtrip_history_and_error_scope(tiny_fit, tmp_path):
    model = tiny_fit
    window = model._development.batch
    x, up = window.past_states[0], window.past_inputs[0]
    uf = window.future_inputs[0]
    before = np.asarray(model.predict(x, up, uf))
    path = tmp_path / "model.npz"
    model.save(path)
    loaded = vehicle.LearnedDynamics.load(path)
    assert loaded.fingerprint() == model.fingerprint()
    np.testing.assert_array_equal(before, loaded.predict(x, up, uf))
    assert loaded.predict(x, up, np.tile(uf, (5, 1))[:24]).shape == (24, 15)
    assert loaded.envelope().shape == (5, 15)
    with pytest.raises(ValueError, match="calibrated"):
        loaded.envelope(24)
    with pytest.raises(ValueError, match="history"):
        loaded.predict(x[1:], up[1:], uf)
    with pytest.raises(ValueError, match="limit"):
        loaded.predict(x, up, np.tile(uf, (5, 1)))
    metadata, arrays = load_arrays(path)
    assert metadata["report"]["precision"]["fitting"] == "float64"
    assert all(v.dtype == np.float64 for v in arrays.values())
    changed = loaded.report
    changed["recipe"]["width"] = 1
    assert loaded.recipe["width"] == 32


def test_update_pins_development_and_preserves_revision(tiny_fit, tmp_path):
    old = tiny_fit
    fingerprint = old.fingerprint()
    before = {k: np.array(v, copy=True) for k, v in old._arrays().items()}
    updated = old.update(recordings("fresh", count=2))
    assert old.fingerprint() == fingerprint
    assert updated.report["previous_revision"] == fingerprint
    assert set(updated._seen) > set(old._seen)
    np.testing.assert_array_equal(
        updated._development.batch.past_states, old._development.batch.past_states
    )
    for k, v in before.items():
        np.testing.assert_array_equal(old._arrays()[k], v)
    with pytest.raises(ValueError, match="reuses"):
        old.update(recordings())
    # Renaming the original recordings does not make their content independent.
    with pytest.raises(ValueError, match="reuses"):
        old.update(recordings("renamed"))
    with pytest.raises(ValueError, match="differ"):
        old.update(replace(recordings("fresh"), configuration_id="other"))


def test_semantics_rejected_before_fit(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("invalid semantic input reached the fitter")

    monkeypatch.setattr(vehicle, "fit_sequence", fail)
    r = recordings()
    with pytest.raises(ValueError, match="canonical"):
        vehicle.fit(replace(r, state_channels=tuple(reversed(r.state_channels))))
    with pytest.raises(ValueError, match="canonical"):
        vehicle.fit(replace(r, state_channels=tuple(f"x{i}" for i in range(15))))
    bad = np.array(r.segments[0].states)
    bad[:, 6] = -1  # Orthogonal reflection, not a valid body orientation.
    segment = replace(r.segments[0], states=bad)
    with pytest.raises(ValueError, match="proper"):
        vehicle.fit(replace(r, segments=(segment, *r.segments[1:])))


def test_archive_semantic_tampering_rejected_even_with_repaired_checksum(
    tiny_fit, tmp_path
):
    path = tmp_path / "good.npz"
    tiny_fit.save(path)
    mutations = [
        lambda m, a: a["train_past_states"].__setitem__(
            (0, 0, 0), a["train_past_states"][0, 0, 0] + 1.0
        ),
        lambda m, a: m.update(unknown="discarded"),
        lambda m, a: m["windows"].update(extra={}),
        lambda m, a: m["windows"]["train"].update(extra=True),
        lambda m, a: m["contract"]["state_channels"].reverse(),
        lambda m, a: m["windows"]["development"]["keys"][0].update(
            recording_id=m["windows"]["train"]["keys"][0]["recording_id"]
        ),
        lambda m, a: m["report"].update(prediction_limit_steps=1200),
        lambda m, a: a.update(unknown=np.zeros(1)),
        lambda m, a: a.update(param_bias=a["param_bias"] + 0.1),
        lambda m, a: m["report"]["envelope"].update(nominal_coverage=1.0),
    ]
    for i, mutation in enumerate(mutations):
        meta, arrays = load_arrays(path)
        mutation(meta, arrays)
        changed = tmp_path / f"changed-{i}.npz"
        save_arrays(changed, meta, arrays)
        with pytest.raises(ValueError):
            vehicle.LearnedDynamics.load(changed)
    with np.load(path) as data:
        fields = {k: data[k] for k in data.files}
    meta = json.loads(str(fields["metadata"]))
    meta["report"]["previous_revision"] = "changed"
    fields["metadata"] = json.dumps(meta)
    np.savez_compressed(tmp_path / "raw.npz", **fields)
    with pytest.raises(ValueError, match="fingerprint"):
        vehicle.LearnedDynamics.load(tmp_path / "raw.npz")


def test_planning_reverse_gradients_and_precision(tiny_fit):
    model = tiny_fit
    b = model._development.batch
    x, up = b.past_states[0], b.past_inputs[0]
    uf = np.tile(b.future_inputs[0], (4, 1))
    ambient = jax.config.x64_enabled
    for enabled, dtype in [(False, np.float32), (True, np.float64)]:
        with jax.enable_x64(enabled):
            command = jnp.asarray(uf)
            f = jax.jit(lambda u: model.predict(x, up, u))
            prediction = f(command)
            gradient = jax.jit(
                jax.grad(lambda u, forward=f: jnp.sum(forward(u)[-1, :6] ** 2))
            )(command)
            assert prediction.dtype == dtype
            assert np.isfinite(gradient).all()
            assert np.any(np.abs(gradient) > 1e-9)
    assert jax.config.x64_enabled == ambient


def test_configuration_and_command_labels_do_not_choose_equations(tiny_fit):
    source = recordings()
    relabeled = replace(
        source,
        configuration_id="unrecognized-vehicle-with-no-layout",
        input_channels=(
            "arbitrary-A [unitless]",
            "arbitrary-B [unitless]",
            "arbitrary-C [unitless]",
        ),
    )
    second = vehicle.fit(relabeled)
    assert second.contract != tiny_fit.contract
    for name, value in tiny_fit._model.arrays().items():
        np.testing.assert_array_equal(second._model.arrays()[name], value)
