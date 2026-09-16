"""Independent algebra, causal alignment, gradients, and artifact contracts."""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from glassbox.experimental.sequence_model import (
    SequenceModel,
    fit_sequence_model,
    initialize_sequence_model,
    sequence_windows,
)
from glassbox.experimental.structured_regression import (
    StructuredRegressor,
    covariance,
    fit_structured_regressor,
)


def test_pairwise_kernel_matches_explicit_products_and_is_psd():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(20, 4))
    individual = [np.exp(-0.5 * (x[:, None, i] - x[None, :, i]) ** 2) for i in range(4)]
    explicit = (
        np.mean(individual, axis=0)
        + np.mean(
            [
                individual[i] * individual[j]
                for i, j in itertools.combinations(range(4), 2)
            ],
            axis=0,
        )
    ) / 2
    actual = covariance(x, x, "pairwise", 1)
    np.testing.assert_allclose(actual, explicit, atol=1e-12)
    np.testing.assert_allclose(np.diag(actual), 1)
    assert np.linalg.eigvalsh(actual).min() > -1e-10


@pytest.mark.parametrize("kind", ["linear", "rbf", "additive", "pairwise"])
def test_regression_predictions_gradients_and_serialization(kind, tmp_path):
    rng = np.random.default_rng(1)
    x = rng.normal(size=(40, 3))
    y = np.column_stack((x[:, 0] ** 2 + x[:, 1], np.sin(x[:, 2])))
    model = fit_structured_regressor(x, y, kind=kind)
    a = model.arrays
    query = x[:4] + 0.1
    z = (query - a["feature_mean"]) / a["feature_scale"]
    expected = z @ a["linear"]
    if kind != "linear":
        expected += covariance(z, a["features"], kind, 1) @ a["alpha"]
    expected = expected * a["target_scale"] + a["target_mean"]
    np.testing.assert_allclose(jax.jit(model.predict)(query), expected, atol=2e-5)
    derivative = np.asarray(jax.jacrev(model.predict)(jnp.asarray(query[0])))
    epsilon = 1e-3
    numerical = np.column_stack(
        [
            (
                np.asarray(model.predict(query[0] + epsilon * np.eye(3)[i]))
                - np.asarray(model.predict(query[0] - epsilon * np.eye(3)[i]))
            )
            / (2 * epsilon)
            for i in range(3)
        ]
    )
    np.testing.assert_allclose(derivative, numerical, atol=3e-3)
    path = tmp_path / "regressor.npz"
    model.save(path)
    loaded = StructuredRegressor.load(path)
    assert loaded.fingerprint() == model.fingerprint()
    np.testing.assert_array_equal(loaded.predict(query), model.predict(query))
    with np.load(path) as archive:
        changed = {k: archive[k] for k in archive.files}
    changed["linear"] = changed["linear"] + 1
    np.savez(path, **changed)
    with pytest.raises(ValueError, match="fingerprint"):
        StructuredRegressor.load(path)


def example_batch():
    rng = np.random.default_rng(9)
    u = rng.normal(size=(119, 2))
    x = np.zeros((120, 2))
    for t in range(119):
        x[t + 1] = 0.8 * x[t] + 0.1 * u[t] + 0.02 * np.sin(x[t, ::-1])
    return sequence_windows(
        x, u, np.arange(10, 90, 2), history_steps=3, horizon_steps=5, dt_s=0.01
    )


def test_sequence_alignment_and_no_future_state_in_inputs():
    x, u = np.arange(100.0)[:, None], np.arange(99.0)[:, None] * 2
    args = dict(anchors=[20, 30, 40], history_steps=3, horizon_steps=5, dt_s=0.01)
    a = sequence_windows(x, u, **args)
    np.testing.assert_array_equal(a.past_states[0, :, 0], [17, 18, 19, 20])
    np.testing.assert_array_equal(a.past_inputs[0, :, 0], [34, 36, 38])
    np.testing.assert_array_equal(a.future_inputs[0, :, 0], [40, 42, 44, 46, 48])
    np.testing.assert_array_equal(a.future_states[0, :, 0], [21, 22, 23, 24, 25])
    x[41:] = 10000
    b = sequence_windows(x, u, **args)
    np.testing.assert_array_equal(a.past_states, b.past_states)
    assert b.future_states[-1, 0, 0] == 10000
    for anchors in ([1, 20, 30], [20, 30, 98], [20, 20, 30], [20.0, 30.0, 40.0]):
        with pytest.raises(ValueError):
            sequence_windows(x, u, **{**args, "anchors": anchors})


def numpy_rollout(model, past, up, uf):
    """Independent scalar-loop replay, including delayed coordinates and memory."""
    n, p = model.norms, model.params
    history = (past - n["state_mean"]) / n["state_scale"]
    commands = (up - n["input_mean"]) / n["input_scale"]
    future = (uf - n["input_mean"]) / n["input_scale"]
    hidden = np.empty(0)
    if model.kind == "latent":
        hidden = np.tanh(
            np.r_[history.ravel(), commands.ravel()] @ p["encoder"] + p["encoder_bias"]
        )
    output = []
    for command in future:
        current = history[-1]
        features = [current, command]
        if model.kind in ("delay", "delay_mlp"):
            features += [(history[:-1] - current).ravel(), (commands - command).ravel()]
        if model.kind == "latent":
            features += [hidden]
        features = np.concatenate(features) / n["feature_scale"]
        delta = features @ p["linear"] + p["bias"]
        if model.kind in ("mlp", "latent", "delay_mlp"):
            delta += np.tanh(features @ p["w1"] + p["b1"]) @ p["w2"]
        predicted = current + n["delta_scale"] * delta
        if model.kind == "latent":
            hidden = np.tanh(features @ p["memory"] + p["memory_bias"])
        output.append(predicted * n["state_scale"] + n["state_mean"])
        history = np.concatenate((history[1:], predicted[None]))
        commands = np.concatenate((commands[1:], command[None]))
    return np.asarray(output)


@pytest.mark.parametrize("kind", ["linear", "delay", "mlp", "latent", "delay_mlp"])
def test_recursive_prediction_matches_independent_loop(kind, tmp_path):
    batch = example_batch()
    model = initialize_sequence_model(batch, kind=kind, width=6, memory=3)
    if kind in ("mlp", "latent", "delay_mlp"):
        model.params["w2"][:] = 0.04
    if kind == "latent":
        model.params["linear"][-3:] = 0.1
    x, up, uf = batch.past_states[0], batch.past_inputs[0], batch.future_inputs[0]
    expected = numpy_rollout(model, x, up, uf)
    actual = np.asarray(jax.jit(model.rollout)(x, up, uf))
    np.testing.assert_allclose(actual, expected, atol=2e-6)
    np.testing.assert_allclose(model.rollout(x, up, uf[:2]), actual[:2], atol=2e-6)
    assert np.isfinite(
        jax.jacrev(lambda u: model.rollout(x, up, u)[-1])(jnp.asarray(uf))
    ).all()
    path = tmp_path / "sequence.npz"
    model.save(path)
    loaded = SequenceModel.load(path)
    assert loaded.fingerprint() == model.fingerprint()
    np.testing.assert_array_equal(loaded.rollout(x, up, uf), model.rollout(x, up, uf))


@pytest.mark.parametrize("objective", ["rollout", "teacher"])
def test_sequence_training_runs_with_both_objectives(objective):
    batch = example_batch()
    model, report = fit_sequence_model(
        batch, batch, steps=3, check_every=3, width=5, memory=2, objective=objective
    )
    assert np.isfinite(
        model.rollout(batch.past_states, batch.past_inputs, batch.future_inputs)
    ).all()
    assert np.isfinite(report["validation_rollout_mse"])


def test_invalid_regression_inputs_are_rejected():
    x = np.ones((5, 2))
    for options in (
        dict(kind="bad"),
        dict(regularization=0),
        dict(length_scale=np.nan),
    ):
        with pytest.raises(ValueError):
            fit_structured_regressor(x, x, **options)
    with pytest.raises(ValueError):
        fit_structured_regressor(x, x[:-1])
