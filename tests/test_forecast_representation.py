"""Representation options encode explicit generic hypotheses, not force laws."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from glassbox.experimental.direct_forecast import (
    DirectForecast,
    fit_direct_forecast,
    prefix_features,
)
from glassbox.experimental.sequence_model import SequenceBatch


def sample(seed=13):
    rng = np.random.default_rng(seed)
    return SequenceBatch(
        rng.normal(size=(48, 6, 3)),
        rng.normal(size=(48, 5, 2)),
        rng.normal(size=(48, 12, 2)),
        rng.normal(size=(48, 12, 3)),
        0.02,
    )


@pytest.mark.parametrize(
    "representation",
    ["relative", "linear_history", "quadratic_history", "pca95", "pca99"],
)
def test_representations_roundtrip_and_preserve_command_prefixes(
    tmp_path, representation
):
    b = sample()
    m = fit_direct_forecast(b, (1, 5, 12), representation=representation)
    p = m.predict(b.past_states, b.past_inputs, b.future_inputs)
    m.save(tmp_path / "model.npz")
    loaded = DirectForecast.load(tmp_path / "model.npz")
    assert loaded.fingerprint() == m.fingerprint()
    np.testing.assert_array_equal(
        p, loaded.predict(b.past_states, b.past_inputs, b.future_inputs)
    )
    jac = jax.jacfwd(lambda u: m.predict(b.past_states[0], b.past_inputs[0], u))(
        jnp.asarray(b.future_inputs[0])
    )
    np.testing.assert_array_equal(jac[0, :, 1:], 0)
    np.testing.assert_array_equal(jac[1, :, 5:], 0)


def test_relative_representation_is_equivariant_to_observation_offsets():
    b = sample()
    m = fit_direct_forecast(b, (1, 12), representation="relative")
    shift = np.array([2, -3, 1])
    original = m.predict(b.past_states, b.past_inputs, b.future_inputs)
    changed = m.predict(b.past_states + shift, b.past_inputs, b.future_inputs)
    np.testing.assert_allclose(changed, original + shift, atol=2e-6)


@pytest.mark.parametrize("order", [1, 2])
def test_history_compression_recovers_known_temporal_polynomial(order):
    time = np.arange(-5, 1) / 5
    coeff = np.array([[2.0, 3.0], [-0.2, 0.4]])[:order]
    history = sum(time[:, None] ** (k + 1) * coeff[k] for k in range(order))
    x = np.broadcast_to(history, (3, 6, 2))
    up = np.zeros((3, 5, 1))
    uf = np.zeros((3, 12, 1))
    features = prefix_features(
        x,
        up,
        uf,
        1,
        use_history=True,
        representation="linear_history" if order == 1 else "quadratic_history",
    )
    np.testing.assert_allclose(
        features[:, 3 : 3 + order * 2],
        np.broadcast_to(coeff.ravel(), (3, order * 2)),
        atol=1e-12,
    )


def test_pca_basis_is_training_derived_and_has_declared_energy():
    b = sample()
    m = fit_direct_forecast(b, (5,), representation="pca95")
    p = m.arrays["projection_5"]
    s = m.arrays["singular_values_5"]
    rank = p.shape[1]
    np.testing.assert_allclose(p.T @ p, np.eye(rank), atol=1e-12)
    assert np.sum(s[:rank] ** 2) >= 0.95 * np.sum(s**2)
    assert np.sum(s[: rank - 1] ** 2) < 0.95 * np.sum(s**2)


def test_full_default_preserves_v1_metadata_and_predictions():
    b = sample()
    a = fit_direct_forecast(b, (1, 5))
    c = fit_direct_forecast(b, (1, 5), representation="full")
    assert a.metadata()["format"] == "glassbox-direct-forecast-v1"
    assert "representation" not in a.metadata()
    assert a.fingerprint() == c.fingerprint()


@pytest.mark.parametrize(
    "representation,history",
    [("unknown", True), ("relative", False), ("quadratic_history", False)],
)
def test_invalid_representation_contract_rejected(representation, history):
    with pytest.raises(ValueError):
        fit_direct_forecast(
            sample(), (1, 5), representation=representation, use_history=history
        )
