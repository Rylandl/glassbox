"""Temporal causality, recording boundaries and identity-mean behavior."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from glassbox.experimental.temporal import forecast_windows
from glassbox.experimental.transition_gp import (
    GaussianTransition,
    TransitionSamples,
    fit_transition_gp,
)


def windows(states=None, controls=None, **kwargs):
    x = np.arange(100.0)[:, None] if states is None else states
    u = np.arange(99.0)[:, None] if controls is None else controls
    return forecast_windows(
        x,
        u,
        recording_id="one-flight",
        dt_s=0.01,
        horizon_steps=10,
        anchors=[20, 30, 40],
        history_lags=(5, 10),
        control_bins=2,
        **kwargs,
    )


def test_windows_use_future_actuation_but_only_origin_and_past_states():
    result = windows()
    np.testing.assert_array_equal(result.samples.states[:, 0], [20, 30, 40])
    np.testing.assert_array_equal(result.samples.next_states[:, 0], [30, 40, 50])
    # Two future input means, then state/input changes at two causal lags.
    np.testing.assert_array_equal(result.samples.context[0], [22, 27, -5, -5, -10, -10])
    assert result.samples.dt_s == 0.1
    changed = np.arange(100.0)[:, None]
    changed[41:] = 10000
    edited = windows(states=changed)
    np.testing.assert_array_equal(result.samples.features, edited.samples.features)
    assert edited.samples.next_states[-1, 0] == 10000
    assert result.sample_ids[0] == "one-flight/anchor20/h10"


def test_auxiliary_measurements_are_sampled_only_at_origin():
    auxiliary = np.arange(100.0)[:, None] * 3
    a = windows(context=auxiliary)
    auxiliary[41:] = -500
    b = windows(context=auxiliary)
    np.testing.assert_array_equal(a.samples.features, b.samples.features)
    np.testing.assert_array_equal(a.samples.context[:, 0], [60, 90, 120])


@pytest.mark.parametrize("horizon, bins", [(1, 1), (25, 10)])
def test_default_control_bins_work_for_short_and_long_horizons(horizon, bins):
    result = forecast_windows(
        np.arange(100.0)[:, None],
        np.arange(99.0)[:, None],
        recording_id="flight",
        dt_s=0.01,
        horizon_steps=horizon,
        anchors=[10, 20, 30],
    )
    assert result.samples.context.shape == (3, bins)
    assert result.control_bin_edges[0] == 0
    assert result.control_bin_edges[-1] == horizon


@pytest.mark.parametrize(
    "anchors", [[0, 20, 30], [20, 30, 90], [20, 20, 30], [20.0, 30.0, 40.0]]
)
def test_incomplete_or_duplicate_windows_are_rejected(anchors):
    with pytest.raises(ValueError):
        forecast_windows(
            np.zeros((100, 2)),
            np.zeros((99, 3)),
            recording_id="flight",
            dt_s=0.01,
            horizon_steps=10,
            anchors=anchors,
            history_lags=(10,),
        )


def test_increment_mean_adds_identity_to_predictions_and_derivatives(tmp_path):
    # An independently constructed constant GP residual has known derivatives.
    from test_transition_gp import repeated_observation_model

    base = repeated_observation_model("rbf")
    residual = replace(base, mean_mode="increment")
    query = jnp.array([3.0])
    control = jnp.array([0.2])
    np.testing.assert_allclose(jax.jit(residual.predict)(query, control).mean, [3.0])
    response = residual.local_response(query, control)
    np.testing.assert_allclose(response.jacobian, [[1, 0]], atol=1e-7)
    np.testing.assert_allclose(response.hessian, 0, atol=1e-7)
    np.testing.assert_allclose(
        jax.jit(residual.mean_rollout)(query, jnp.zeros((4, 1))), np.full((5, 1), 3.0)
    )
    assert residual.fingerprint() != base.fingerprint()
    path = tmp_path / "increment.npz"
    residual.save(path)
    loaded = GaussianTransition.load(path)
    assert loaded.mean_mode == "increment"
    assert loaded.fingerprint() == residual.fingerprint()
    np.testing.assert_array_equal(
        loaded.predict(query, control).mean, residual.predict(query, control).mean
    )


def test_increment_fitter_preserves_the_external_next_state_contract():
    x = np.linspace(-2, 2, 24)[:, None]
    u = np.cos(3 * x)
    y = x + 0.05 * u
    model = fit_transition_gp(
        TransitionSamples(x, u, y, 0.1), mean_mode="increment", steps=60, restarts=1
    )
    np.testing.assert_allclose(model.target_mean, (y - x).mean(axis=0), atol=1e-7)
    np.testing.assert_allclose(model.predict(x, u).mean, y, atol=0.01)
    assert model.fit_report["mean_mode"] == "increment"
