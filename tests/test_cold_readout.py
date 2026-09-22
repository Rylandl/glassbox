import jax
import jax.numpy as jnp
import numpy as np
import pytest
from screen_cold_readout import (
    Readout,
    coefficients,
    readout_features,
    recursive_update,
)
from test_dynamics import constant_model
from test_online import prefix, stream

from glassbox import _dynamics as core


@pytest.mark.parametrize("commands,delay", [(3, 2), (4, 10)])
def test_readout_reconstructs_all_acceleration_paths(commands, delay):
    model = constant_model(commands, history=delay + 1, delay=delay)
    rng = np.random.default_rng(851)
    params = {k: rng.normal(size=v.shape) * 0.1 for k, v in model.params.items()}
    state = np.r_[rng.normal(size=6) * 0.2, np.eye(3).ravel()]
    command = rng.normal(size=commands) * 0.1
    history = rng.normal(size=(delay, 9 + 2 * commands)) * 0.1
    hidden = rng.normal(size=2) * 0.1
    with jax.enable_x64(True):
        physical, current, _ = core._head(
            params,
            model.norms,
            jnp.asarray(state),
            jnp.asarray(command),
            jnp.asarray(command),
            jnp.asarray(history),
            jnp.asarray(hidden),
        )
        actual = readout_features(
            params, model.norms, current, jnp.asarray(history), jnp.asarray(hidden)
        ) @ coefficients(params)
        np.testing.assert_allclose(
            actual * model.norms["output_scale"], physical, rtol=1e-12, atol=1e-12
        )


def test_recursive_estimator_matches_independent_regularized_batch_solution():
    rng = np.random.default_rng(12)
    x = rng.normal(size=(30, 12))
    y = rng.normal(size=(30, 6))
    prior = rng.normal(size=(12, 6))
    ridge = 0.25
    with jax.enable_x64(True):
        inverse, mean = jnp.eye(12) / ridge, jnp.asarray(prior)
        for row in range(len(x)):
            inverse, mean = recursive_update(inverse, mean, x[row], y[row])
        precision = ridge * np.eye(12) + x.T @ x
        expected = np.linalg.solve(precision, ridge * prior + x.T @ y)
        np.testing.assert_allclose(mean, expected, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(
            inverse, np.linalg.inv(precision), rtol=1e-10, atol=1e-10
        )


@pytest.mark.parametrize("commands,dt", [(3, 0.05), (4, 0.01)])
def test_fresh_update_preserves_features_and_rejects_duplicate_row(commands, dt):
    states, inputs = stream(commands, dt=dt)
    candidate = Readout(prefix(states, inputs, offset=0, dt=dt))
    before = candidate.session.model.arrays()
    row = candidate.session.cursor
    candidate.observe(row, inputs[row], states[row + 1])
    after = candidate.session.model.arrays()
    for key, value in before.items():
        if key not in ("param_linear", "param_quadratic", "param_bias", "param_w2"):
            np.testing.assert_array_equal(after[key], value)
    with pytest.raises(ValueError, match="noncausal"):
        candidate.observe(row, inputs[row], states[row + 1])
