"""Causal memory contract: rest start, in-recording carry, boundaries, old kinds."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from glassbox.experimental.sequence_collection import (
    SequenceCollection,
    segments_from_mask,
)
from glassbox.experimental.sequence_model import (
    SequenceBatch,
    SequenceModel,
    fit_sequence_model,
    initialize_sequence_model,
    sequence_windows,
)

CONTEXT, DELAY, HORIZON = 10, 2, 5


def delayed_recording(seed, n=160, delay=3):
    """x[k+1] = 0.8 x[k] + 0.2 u[k-3]: the G05 witness with independent inputs."""
    rng = np.random.default_rng(seed)
    u = rng.uniform(-1, 1, (n, 1))
    x = np.zeros((n + 1, 1))
    for k in range(n):
        x[k + 1] = 0.8 * x[k] + (0.2 * u[k - delay] if k >= delay else 0.0)
    return x, u


def windows(seeds, context, stride=3):
    parts = [
        sequence_windows(
            *delayed_recording(s),
            np.arange(context, 160 - HORIZON, stride),
            history_steps=context,
            horizon_steps=HORIZON,
            dt_s=0.05,
        )
        for s in seeds
    ]
    return SequenceBatch(
        **{
            k: np.concatenate([getattr(b, k) for b in parts])
            for k in ("past_states", "past_inputs", "future_inputs", "future_states")
        },
        dt_s=0.05,
    )


def shorten(batch, context):
    return SequenceBatch(
        batch.past_states[:, -context - 1 :],
        batch.past_inputs[:, -context:],
        batch.future_inputs,
        batch.future_states,
        batch.dt_s,
    )


def paired_probe(context):
    """Opposite inputs three intervals before the origin; everything else equal."""
    full = np.zeros((2, context + HORIZON, 1))
    full[:, context - 3, 0] = [-1.0, 1.0]
    states = []
    for u in full:
        x = np.zeros((context + HORIZON + 1, 1))
        for k in range(context + HORIZON):
            x[k + 1] = 0.8 * x[k] + (0.2 * u[k - 3] if k >= 3 else 0.0)
        states.append(x)
    states = np.stack(states)
    return (
        states[:, : context + 1],
        full[:, :context],
        full[:, context:],
        states[:, context + 1 :],
    )


@pytest.fixture(autouse=True)
def float64():
    """Exactness claims are checked in float64; the contract itself is precision-free."""
    with jax.enable_x64(True):
        yield


@pytest.fixture(scope="module")
def batch():
    return windows((1, 2), CONTEXT)


@pytest.fixture(scope="module")
def model(batch):
    m = initialize_sequence_model(
        batch, kind="filter_mlp", width=4, memory=3, delay_steps=DELAY
    )
    # Give the memory a nonzero readout so that carrying it matters.
    m.params["linear"][-3:] = 0.1
    m.params["w2"][:] = 0.04
    return m


def numpy_rollout(model, x, up, uf):
    """Independent scalar-loop replay of the filter, then the recursive forecast."""
    n, p, delay = model.norms, model.params, model.delay_steps
    xs = (x - n["state_mean"]) / n["state_scale"]
    us = (up - n["input_mean"]) / n["input_scale"]
    future = (uf - n["input_mean"]) / n["input_scale"]

    def features(current, command, history, commands, hidden):
        return (
            np.concatenate(
                (
                    current,
                    command,
                    (history - current).ravel(),
                    (commands - command).ravel(),
                    hidden,
                )
            )
            / n["feature_scale"]
        )

    hidden = np.zeros(p["memory"].shape[1])
    for j in range(delay, len(us)):
        z = features(xs[j], us[j], xs[j - delay : j], us[j - delay : j], hidden)
        hidden = np.tanh(z @ p["memory"] + p["memory_bias"])
    history, commands = xs[len(us) - delay :], us[len(us) - delay :]
    output = []
    for command in future:
        current = history[-1]
        z = features(current, command, history[:-1], commands, hidden)
        delta = z @ p["linear"] + p["bias"] + np.tanh(z @ p["w1"] + p["b1"]) @ p["w2"]
        predicted = current + n["delta_scale"] * delta
        hidden = np.tanh(z @ p["memory"] + p["memory_bias"])
        output.append(predicted * n["state_scale"] + n["state_mean"])
        history = np.concatenate((history[1:], predicted[None]))
        commands = np.concatenate((commands[1:], command[None]))
    return np.asarray(output)


def test_memory_starts_at_rest_and_is_a_nontrivial_function_of_the_context(
    model, batch
):
    x, up = batch.past_states[:4], batch.past_inputs[:4]
    rest = np.zeros((4, 3))
    np.testing.assert_array_equal(
        model.memory_state(x, up), model.memory_state(x, up, memory=rest)
    )
    # Observations before the consumed context are never implied: the same
    # context after different earlier data differs only if that state is carried.
    carried = model.memory_state(batch.past_states[4:8], batch.past_inputs[4:8])
    assert not np.allclose(model.memory_state(x, up, memory=carried), rest)
    assert not np.allclose(
        model.memory_state(x, up), model.memory_state(x, up, memory=carried)
    )
    assert model.memory_state(x[0], up[0]).shape == (3,)


@pytest.mark.parametrize("split", [DELAY, 6, CONTEXT - 1])
def test_carrying_memory_within_a_recording_matches_the_whole_context(
    model, batch, split
):
    x, up, uf = batch.past_states[:3], batch.past_inputs[:3], batch.future_inputs[:3]
    whole = model.rollout(x, up, uf)
    carried = model.memory_state(x[:, : split + 1], up[:, :split])
    continued = model.rollout(
        x[:, split - DELAY :], up[:, split - DELAY :], uf, memory=carried
    )
    np.testing.assert_allclose(continued, whole, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(
        model.memory_state(
            x[:, split - DELAY :], up[:, split - DELAY :], memory=carried
        ),
        model.memory_state(x, up),
        rtol=1e-10,
        atol=1e-12,
    )
    # A carried state is the only way to use a longer or shorter context.
    with pytest.raises(ValueError, match="shapes"):
        model.rollout(x[:, 1:], up[:, 1:], uf)
    with pytest.raises(ValueError, match="shapes"):
        model.rollout(x[:, : DELAY + 1], up[:, :DELAY], uf)


def test_context_windows_never_bridge_a_segment_boundary(model):
    x, u = delayed_recording(5, n=60)
    valid = np.ones(61, dtype=bool)
    valid[30:33] = False
    segments = segments_from_mask("r", x, u, valid, dt_s=0.05)
    assert len(segments) == 2
    collection = SequenceCollection(segments)
    keys = collection.window_keys(history_steps=CONTEXT, horizon_steps=HORIZON)
    assert len(keys) == sum(max(0, len(s.states) - CONTEXT - HORIZON) for s in segments)
    assert all(k.origin >= CONTEXT for k in keys)
    extracted = collection.extract(keys, history_steps=CONTEXT, horizon_steps=HORIZON)
    for key, origin, past in zip(
        extracted.keys, extracted.source_origins, extracted.batch.past_states
    ):
        segment = next(s for s in segments if s.segment_id == key.segment_id)
        assert segment.start_row <= origin - CONTEXT
        np.testing.assert_array_equal(past, x[origin - CONTEXT : origin + 1])
        assert not np.isnan(past).any()
    # The memory of a second-segment window is a function of that segment only.
    later = [
        i for i, k in enumerate(extracted.keys) if k.segment_id != keys[0].segment_id
    ]
    b = extracted.batch
    direct = model.memory_state(b.past_states[later], b.past_inputs[later])
    np.testing.assert_array_equal(
        direct,
        model.memory_state(
            b.past_states[later], b.past_inputs[later], memory=np.zeros((len(later), 3))
        ),
    )


def test_forecast_is_causal_prefix_consistent_and_matches_independent_replay(
    model, batch
):
    x, up, uf = batch.past_states[0], batch.past_inputs[0], batch.future_inputs[0]
    whole = np.asarray(jax.jit(model.rollout)(x, up, uf))
    np.testing.assert_allclose(model.rollout(x, up, uf[:2]), whole[:2], atol=1e-12)
    np.testing.assert_allclose(numpy_rollout(model, x, up, uf), whole, atol=1e-10)
    jac = jax.jacfwd(lambda u: model.rollout(x, up, u))(jnp.asarray(uf))
    assert np.isfinite(jac).all()
    for h in range(HORIZON):
        np.testing.assert_array_equal(jac[h, :, h + 1 :], 0)
    # The first forecast depends on inputs older than the explicit delay history.
    older = jax.jacfwd(lambda u: model.rollout(x, u, uf)[0])(jnp.asarray(up))
    assert np.abs(older[:, : CONTEXT - DELAY]).max() > 0


def test_rest_start_reproduces_the_delay_model_affine_initialization(batch):
    short = shorten(batch, DELAY)
    filtered = initialize_sequence_model(
        batch, kind="filter_mlp", width=4, memory=3, delay_steps=DELAY
    )
    delayed = initialize_sequence_model(short, kind="delay_mlp", width=4)
    for key in delayed.norms:
        np.testing.assert_allclose(
            filtered.norms[key][: len(delayed.norms[key])], delayed.norms[key]
        )
    np.testing.assert_allclose(
        filtered.params["linear"][:-3], delayed.params["linear"], atol=1e-12
    )
    np.testing.assert_array_equal(filtered.params["linear"][-3:], 0)
    np.testing.assert_allclose(
        filtered.params["bias"], delayed.params["bias"], atol=1e-12
    )
    np.testing.assert_array_equal(filtered.params["w2"], 0)
    np.testing.assert_allclose(
        filtered.rollout(batch.past_states, batch.past_inputs, batch.future_inputs),
        delayed.rollout(short.past_states, short.past_inputs, short.future_inputs),
        rtol=1e-10,
        atol=1e-12,
    )
    assert filtered.history_steps == CONTEXT and filtered.delay_steps == DELAY
    assert delayed.history_steps == DELAY and delayed.delay_steps is None


def test_memory_recovers_a_delayed_input_response_that_explicit_history_cannot():
    train, development = windows((1, 2, 3, 4, 5, 6), CONTEXT), windows((7, 8), CONTEXT)
    settings = dict(
        width=8,
        ridge=0.01 * len(train.past_states) * HORIZON,
        steps=400,
        batch_size=64,
        learning_rate=0.002,
        check_every=100,
    )
    px, pu, uf, target = paired_probe(CONTEXT)
    np.testing.assert_allclose(
        target[:, :, 0], np.stack((-1, 1))[:, None] * 0.2 * 0.8 ** np.arange(HORIZON)
    )
    np.testing.assert_array_equal(px[0], px[1])
    assert np.count_nonzero(pu[0] != pu[1]) == 1
    with jax.enable_x64(True):
        incumbent, _ = fit_sequence_model(
            shorten(train, DELAY),
            shorten(development, DELAY),
            kind="delay_mlp",
            **settings,
        )
        candidate, report = fit_sequence_model(
            train,
            development,
            kind="filter_mlp",
            memory=4,
            delay_steps=DELAY,
            **settings,
        )
        blind = np.asarray(incumbent.rollout(px[:, -DELAY - 1 :], pu[:, -DELAY:], uf))
        informed = np.asarray(candidate.rollout(px, pu, uf))
    paired = lambda y: np.sqrt(np.mean((y - target) ** 2, axis=(0, 2)))  # noqa: E731
    np.testing.assert_array_equal(blind[0], blind[1])
    assert paired(blind)[0] >= 0.2 - 1e-9
    assert paired(informed)[0] < 0.05
    assert report["context_steps"] == CONTEXT and report["delay_steps"] == DELAY


def test_round_trip_and_unchanged_contracts_for_other_kinds(model, batch, tmp_path):
    path = tmp_path / "memory.npz"
    model.save(path)
    loaded = SequenceModel.load(path)
    assert loaded.fingerprint() == model.fingerprint()
    assert loaded.delay_steps == DELAY and loaded.metadata()["delay_steps"] == DELAY
    x, up, uf = batch.past_states[:2], batch.past_inputs[:2], batch.future_inputs[:2]
    np.testing.assert_array_equal(loaded.rollout(x, up, uf), model.rollout(x, up, uf))
    short = shorten(batch, DELAY)
    plain = initialize_sequence_model(short, kind="delay_mlp", width=4)
    assert "delay_steps" not in plain.metadata()
    assert (
        SequenceModel(
            *(
                getattr(plain, f)
                for f in ("kind", "dt_s", "history_steps", "params", "norms")
            )
        ).fingerprint()
        == plain.fingerprint()
    )
    with pytest.raises(ValueError, match="memory"):
        plain.rollout(
            short.past_states[:2], short.past_inputs[:2], uf, memory=np.zeros((2, 3))
        )
    with pytest.raises(ValueError, match="memory state"):
        plain.memory_state(short.past_states[:2], short.past_inputs[:2])
    for bad in (
        dict(kind="delay_mlp", delay_steps=2),
        dict(kind="filter_mlp"),
        dict(kind="filter_mlp", delay_steps=CONTEXT),
    ):
        with pytest.raises(ValueError):
            initialize_sequence_model(batch, width=4, **bad)
    with pytest.raises(ValueError, match="delay_steps"):
        SequenceModel("delay_mlp", 0.05, 2, plain.params, plain.norms, delay_steps=1)
    with pytest.raises(ValueError, match="delay_steps"):
        SequenceModel("filter_mlp", 0.05, 2, model.params, model.norms, delay_steps=2)
