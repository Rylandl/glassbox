"""Focused checks for the temporary trainable-subset experiment."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree
from screen_cold_readout_rollout import Readout, verify_feature_functions
from test_online import prefix, stream

from glassbox import online


def test_subset_solves_restricted_damped_problem(monkeypatch):
    """A frozen coordinate affects loss but must not receive its nonzero gradient."""
    params = dict(
        linear=jnp.array([[0.2]]),
        quadratic=jnp.zeros((1, 1)),
        bias=jnp.zeros(1),
        w2=jnp.zeros((1, 1)),
        w1=jnp.array([[0.7]]),
    )

    def residual(p, *args):
        return (
            jnp.zeros((1, 1, 15))
            .at[0, 0, 0]
            .set(p["linear"][0, 0] + p["w1"][0, 0] - 0.6)
        )

    linearize = jax.linearize

    def check_masked_jvp(function, flat):
        assert flat.shape == (4,)  # The fifth, frozen coordinate is absent.
        value, push = linearize(function, flat)
        direction = jnp.array([0.0, 1.0, 0.0, 0.0], dtype=flat.dtype)
        full_direction = jax.tree.map(jnp.zeros_like, params)
        full_direction["linear"] = jnp.ones_like(params["linear"])
        expected = jax.jvp(residual, (params,), (full_direction,))[1]
        np.testing.assert_allclose(push(direction), expected, atol=1e-12)
        return value, push

    monkeypatch.setattr(jax, "linearize", check_masked_jvp)
    monkeypatch.setattr(online, "_residual", residual)
    monkeypatch.setattr(
        online,
        "_curvature_diagonal",
        lambda p, *a, **k: jnp.zeros_like(ravel_pytree(p)[0]),
    )
    with jax.enable_x64(True):
        params = jax.tree.map(lambda a: jnp.asarray(a, dtype=jnp.float64), params)
        actual, current, trial, predicted, finite, evidence = (
            online._proposal.__wrapped__(
                params,
                {},
                (),
                jnp.ones((1, 15)),
                jnp.ones(1),
                jnp.array(1.0),
                delay=2,
                dt_s=0.05,
                readout_only=True,
            )
        )
    expected = (
        float(params["linear"][0, 0])
        - (float(params["linear"][0, 0]) + float(params["w1"][0, 0]) - 0.6) / 4
    )
    np.testing.assert_allclose(actual["linear"][0, 0], expected, atol=1e-8)
    np.testing.assert_array_equal(actual["w1"], params["w1"])
    assert bool(finite) and trial < current and predicted > 0
    assert evidence["selected_alpha"] == 1


@pytest.mark.parametrize("commands,dt", [(3, 0.05), (4, 0.01)])
def test_full_observe_preserves_feature_functions_and_checks_exact_loss(commands, dt):
    states, inputs = stream(commands, steps=100, dt=dt)
    p = prefix(states, inputs, offset=0, dt=dt)
    session = Readout(p)
    initial = session.model.arrays()
    for row in range(session.cursor, session.cursor + 3):
        session.observe(row, inputs[row], states[row + 1])
        verify_feature_functions(initial, session.model.arrays())
        online._validate_proposal_report(session.report["last_proposal"])
    assert session.report["accepted_proposals"] > 0
    assert session.report["cg_iterations"] == 48
    assert not np.array_equal(
        initial["param_linear"], session.model.arrays()["param_linear"]
    )
    corrupted = {k: a.copy() for k, a in session.model.arrays().items()}
    corrupted["param_w1"][0, 0] += 0.1
    with pytest.raises(AssertionError):
        verify_feature_functions(initial, corrupted)
