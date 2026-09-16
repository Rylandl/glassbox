"""Direct prediction must preserve time direction and the meaning of each head."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from glassbox.experimental.direct_forecast import DirectForecast, fit_direct_forecast
from glassbox.experimental.sequence_model import SequenceBatch


def batch(seed=0, n=200):
    rng = np.random.default_rng(seed)
    past, up, uf = (
        rng.normal(size=(n, 4, 2)),
        rng.normal(size=(n, 3, 1)),
        rng.normal(size=(n, 5, 1)),
    )
    current = past[:, -1].copy()
    targets = []
    for t in range(5):
        current = current @ np.array([[0.8, 0.1], [0, 0.9]]) + uf[:, t] * [0.2, -0.1]
        targets.append(current.copy())
    return SequenceBatch(past, up, uf, np.stack(targets, 1), 0.02)


@pytest.mark.parametrize("history", [True, False])
def test_direct_heads_recover_affine_dynamics_and_roundtrip(tmp_path, history):
    train, test = batch(), batch(1, 30)
    model = fit_direct_forecast(
        train, (1, 3, 5), use_history=history, ridge_fraction=1e-7
    )
    predicted = model.predict(test.past_states, test.past_inputs, test.future_inputs)
    np.testing.assert_allclose(predicted, test.future_states[:, [0, 2, 4]], atol=2e-6)
    path = tmp_path / "direct.npz"
    model.save(path)
    restored = DirectForecast.load(path)
    assert restored.fingerprint() == model.fingerprint()
    np.testing.assert_array_equal(
        restored.predict(test.past_states, test.past_inputs, test.future_inputs),
        predicted,
    )
    np.testing.assert_allclose(
        model.predict(test.past_states[0], test.past_inputs[0], test.future_inputs[0]),
        predicted[0],
        atol=1e-6,
    )


def test_later_commands_cannot_affect_an_earlier_head():
    train = batch()
    model = fit_direct_forecast(train, (1, 3, 5))
    x, up, uf = (
        train.past_states[0],
        train.past_inputs[0],
        jnp.asarray(train.future_inputs[0]),
    )
    jac = jax.jacfwd(lambda u: model.predict(x, up, u))(uf)
    np.testing.assert_array_equal(jac[0, :, 1:], 0)
    np.testing.assert_array_equal(jac[1, :, 3:], 0)
    assert np.linalg.norm(jac[2, :, -1]) > 0
    epsilon = 1e-3
    finite = (
        model.predict(x, up, uf.at[2, 0].add(epsilon))
        - model.predict(x, up, uf.at[2, 0].add(-epsilon))
    ) / (2 * epsilon)
    np.testing.assert_allclose(finite, jac[:, :, 2, 0], atol=1e-4)


def test_penalty_is_invariant_to_duplicating_training_windows():
    train = batch()
    repeated = SequenceBatch(
        **{
            k: np.tile(getattr(train, k), (2, 1, 1))
            for k in ("past_states", "past_inputs", "future_inputs", "future_states")
        },
        dt_s=train.dt_s,
    )
    a, b = (
        fit_direct_forecast(v, (1, 5), ridge_fraction=0.1) for v in (train, repeated)
    )
    np.testing.assert_allclose(
        a.predict(train.past_states, train.past_inputs, train.future_inputs),
        b.predict(train.past_states, train.past_inputs, train.future_inputs),
        atol=1e-6,
    )


@pytest.mark.parametrize(
    "horizons,ridge", [((0,), 0.1), ((6,), 0.1), ((1, 1), 0.1), ((), 0.1), ((1,), 0)]
)
def test_invalid_fit_contract_rejected(horizons, ridge):
    with pytest.raises(ValueError):
        fit_direct_forecast(batch(), horizons, ridge_fraction=ridge)


def test_insufficient_future_command_prefix_rejected():
    train = batch()
    model = fit_direct_forecast(train, (1, 5))
    with pytest.raises(ValueError):
        model.predict(train.past_states, train.past_inputs, train.future_inputs[:, :3])
