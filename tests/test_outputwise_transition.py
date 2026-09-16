"""Coupled-state prediction and artifact integrity for separate output kernels."""

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from glassbox.experimental.outputwise_transition import (
    OutputwiseTransition,
    fit_outputwise_transition_gp,
    output_feature_order,
)
from glassbox.experimental.transition_gp import TransitionSamples


@pytest.fixture(scope="module")
def coupled():
    rng = np.random.default_rng(12)
    x = rng.uniform(-1, 1, (48, 2))
    u = rng.uniform(-1, 1, (48, 1))
    y = x + np.column_stack((0.15 * x[:, 1] + 0.03 * u[:, 0], -0.1 * x[:, 0]))
    samples = TransitionSamples(x, u, y, 0.1)
    model = fit_outputwise_transition_gp(
        samples, kernel="rq", steps=80, restarts=1, mean_mode="increment"
    )
    return model, samples


def test_outputs_retain_coupling_and_have_jittable_vector_predictions(coupled):
    model, _ = coupled
    x = jnp.array([[0.2, 0.4], [0.2, -0.4]])
    u = jnp.zeros((2, 1))
    result = jax.jit(model.predict)(x, u)
    np.testing.assert_allclose(result.mean, [[0.26, 0.38], [0.14, -0.42]], atol=0.008)
    assert result.function_variance.shape == (2, 2)
    assert np.all(result.observation_variance >= result.function_variance)
    jacobian = jax.jacfwd(lambda state: model.predict(state, jnp.zeros(1)).mean)(x[0])
    np.testing.assert_allclose(jacobian, [[1, 0.15], [-0.1, 1]], atol=0.04)


def test_rollout_updates_outputs_simultaneously(coupled):
    model, _ = coupled
    state = np.array([0.2, 0.4])
    expected = [state]
    for _ in range(3):
        state = np.array([[1, 0.15], [-0.1, 1]]) @ state
        expected.append(state)
    result = jax.jit(model.mean_rollout)(jnp.array(expected[0]), jnp.zeros((3, 1)))
    np.testing.assert_allclose(result, expected, atol=0.015)


def test_scalar_training_preserves_all_features_exactly(coupled):
    model, samples = coupled
    for i, member in enumerate(model.members):
        order = output_feature_order(2, 1, 0, i)
        assert sorted(order) == list(range(3))
        raw = np.asarray(member.features * member.feature_scale + member.feature_mean)
        np.testing.assert_allclose(raw, samples.features[:, order], atol=2e-7)
        np.testing.assert_allclose(
            member.target_mean,
            (samples.next_states - samples.states).mean(axis=0)[i : i + 1],
            atol=1e-7,
        )


def test_round_trip_and_tampered_member_rejected(coupled, tmp_path):
    model, samples = coupled
    model.save(tmp_path / "model")
    loaded = OutputwiseTransition.load(tmp_path / "model")
    assert loaded.fingerprint() == model.fingerprint()
    np.testing.assert_array_equal(
        loaded.predict(samples.states, samples.commands).mean,
        model.predict(samples.states, samples.commands).mean,
    )
    paths = [tmp_path / "model" / f"output-{i}.npz" for i in range(2)]
    left, right = [p.read_bytes() for p in paths]
    paths[0].write_bytes(right)
    paths[1].write_bytes(left)
    with pytest.raises(ValueError, match="fingerprint"):
        OutputwiseTransition.load(tmp_path / "model")


def test_context_and_arbitrary_command_width():
    rng = np.random.default_rng(5)
    x, u, c = rng.normal(size=(3, 12, 3))
    samples = TransitionSamples(x, u, x + 0.1 * c, 0.2, c)
    model = fit_outputwise_transition_gp(
        samples, steps=2, restarts=1, mean_mode="increment"
    )
    assert (model.state_size, model.command_size, model.context_size, model.dt_s) == (
        3,
        3,
        3,
        0.2,
    )
    for i, member in enumerate(model.members):
        raw = np.asarray(member.features * member.feature_scale + member.feature_mean)
        np.testing.assert_allclose(
            raw, samples.features[:, output_feature_order(3, 3, 3, i)], atol=5e-7
        )
    with pytest.raises(ValueError, match="requires context"):
        model.predict(x, u)
    assert model.predict(x[0], u[0], context=c[0]).mean.shape == (3,)


def test_invalid_manifest_and_empty_model(tmp_path):
    with pytest.raises(ValueError, match="at least one"):
        OutputwiseTransition(())
    (tmp_path / "manifest.json").write_text(json.dumps({"format": "wrong"}))
    with pytest.raises(ValueError, match="unsupported"):
        OutputwiseTransition.load(tmp_path)
