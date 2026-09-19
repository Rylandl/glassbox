"""Bounded synthetic checks of the frozen channel objective and one-fit boundary."""

import copy
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_excited_public_matching import prepared_pair
from test_state_quadratic_model import small_batch

from glassbox._learner_arrays import array_fingerprint
from glassbox.experimental import affine_anchored_model as anchored
from glassbox.experimental import initial_channel_balance_model as balanced
from glassbox.experimental import state_quadratic_model as quadratic


@pytest.fixture(autouse=True)
def float64():
    with jax.enable_x64(True):
        yield


def hold_scale(batch):
    context = batch.past_inputs.shape[1]
    states = np.concatenate((batch.past_states, batch.future_states), axis=1)
    scale = states[:, context:-1].std((0, 1))
    scale = np.where(scale > 1e-8, scale, 1)
    hold = np.repeat(batch.past_states[:, -1:], batch.future_states.shape[1], 1)
    return np.maximum(
        np.sqrt(np.mean((hold - batch.future_states) ** 2, 0)), 0.01 * scale
    )


def settings(batch, **extra):
    return dict(
        seed=7,
        width=4,
        memory=2,
        ridge=0.7,
        delay_steps=1,
        error_scale=hold_scale(batch),
        **extra,
    )


def test_manual_full_training_weights_floor_and_unchanged_initial_model():
    train, dev = small_batch(4), small_batch(5)
    args = settings(train)
    initial = anchored.initialize_candidate(
        train, **{k: v for k, v in args.items() if k != "error_scale"}
    )
    prediction = balanced.initial_training_forecast(
        initial.params, initial.norms, train, 1
    )
    e0 = np.mean(
        ((prediction - train.future_states) / args["error_scale"]) ** 2, axis=(0, 1)
    )
    raw = 1 / np.maximum(e0, 0.0001)
    expected = raw / raw.mean()
    model, report = balanced.fit_candidate_sequence(train, dev, steps=0, **args)
    objective = report["objective"]
    for key, value in (
        ("initial_channel_mse", e0),
        ("raw_channel_weights", raw),
        ("channel_weights", expected),
    ):
        np.testing.assert_array_equal(objective[key], value)
    assert objective["weight_floor"] == 0.0001
    assert objective["weight_normalizer"] == raw.mean()
    assert 2 in objective["floor_active_channels"]  # A perfectly constant output.
    assert e0[2] == 0 and raw[2] == 10000
    np.testing.assert_allclose(expected.mean(), 1, atol=1e-15)
    for key, values in initial.arrays().items():
        np.testing.assert_array_equal(model.arrays()[key], values)
    np.testing.assert_array_equal(report["error_scale"], args["error_scale"])
    predicted_dev = np.asarray(
        model.rollout(dev.past_states, dev.past_inputs, dev.future_inputs)
    )
    squared = ((predicted_dev - dev.future_states) / args["error_scale"]) ** 2
    np.testing.assert_allclose(
        report["validation_rollout_mse"], np.mean(expected * squared), rtol=1e-12
    )
    np.testing.assert_allclose(
        report["unweighted_development_trace"][0]["validation_rollout_mse"],
        squared.mean(),
        rtol=1e-12,
    )
    assert objective["initial_parameters_and_norms_fingerprint"] == array_fingerprint(
        {}, initial.arrays()
    )
    assert objective["initial_training_prediction_fingerprint"] == array_fingerprint(
        {}, {"prediction": prediction}
    )
    assert balanced.initialize_candidate is anchored.initialize_candidate
    assert balanced._rollout is quadratic._rollout


def test_weight_forecast_is_recursive_and_canonical_replay_is_exact():
    train = small_batch(3)
    initial = anchored.initialize_candidate(train, width=4, memory=2, delay_steps=1)
    first = balanced.initial_training_forecast(initial.params, initial.norms, train, 1)
    second = balanced.initial_training_forecast(
        jax.tree.map(jnp.asarray, initial.params),
        jax.tree.map(jnp.asarray, initial.norms),
        train,
        1,
    )
    np.testing.assert_array_equal(first, second)
    changed_targets = replace(train, future_states=train.future_states + 100)
    np.testing.assert_array_equal(
        first,
        balanced.initial_training_forecast(
            initial.params, initial.norms, changed_targets, 1
        ),
    )
    assert first.dtype == np.float64 and first.shape == train.future_states.shape


def test_development_does_not_determine_weights_or_initial_parameters():
    train, dev = small_batch(), small_batch(1)
    first, a = balanced.fit_candidate_sequence(train, dev, steps=0, **settings(train))
    second, b = balanced.fit_candidate_sequence(
        train,
        replace(dev, future_states=dev.future_states + 0.5),
        steps=0,
        **settings(train),
    )
    assert a["objective"] == b["objective"]
    assert a["validation_rollout_mse"] != b["validation_rollout_mse"]
    for key, values in first.arrays().items():
        np.testing.assert_array_equal(second.arrays()[key], values)


def test_one_initializer_two_solves_one_weight_forecast_and_original_rng(monkeypatch):
    train, dev = small_batch(), small_batch(1)
    original = (
        balanced.initialize_candidate,
        np.linalg.solve,
        balanced.initial_training_forecast,
        np.random.default_rng,
    )
    counts, seeds = [0, 0, 0], []

    def count(index, *args, **kwargs):
        counts[index] += 1
        return original[index](*args, **kwargs)

    def rng(seed):
        seeds.append(seed)
        return original[3](seed)

    monkeypatch.setattr(
        balanced, "initialize_candidate", lambda *a, **k: count(0, *a, **k)
    )
    monkeypatch.setattr(np.linalg, "solve", lambda *a, **k: count(1, *a, **k))
    monkeypatch.setattr(
        balanced, "initial_training_forecast", lambda *a, **k: count(2, *a, **k)
    )
    monkeypatch.setattr(np.random, "default_rng", rng)
    _, report = balanced.fit_candidate_sequence(
        train, dev, steps=3, check_every=1, **settings(train)
    )
    assert counts == [1, 2, 1]
    assert seeds == [7, 10007]
    assert [row["step"] for row in report["trace"]] == [0, 1, 2, 3]
    assert (
        report["selected_step"]
        == min(report["trace"], key=lambda row: row["validation_rollout_mse"])["step"]
    )
    assert all(
        set(row) == {"step", "validation_rollout_mse"}
        for row in report["unweighted_development_trace"]
    )


def test_first_adam_update_matches_independent_weighted_gradient_and_clipping():
    train = small_batch(4)
    args = settings(train)
    initial = anchored.initialize_candidate(
        train, **{k: v for k, v in args.items() if k != "error_scale"}
    )
    prediction = balanced.initial_training_forecast(
        initial.params, initial.norms, train, 1
    )
    e0 = np.mean(
        ((prediction - train.future_states) / args["error_scale"]) ** 2, (0, 1)
    )
    weights = 1 / np.maximum(e0, 0.0001)
    weights /= weights.mean()
    params, norms = jax.tree.map(jnp.asarray, (initial.params, initial.norms))
    indices = np.random.default_rng(10007).integers(len(train.past_states), size=7)
    x, up, uf, target = [
        jnp.asarray(getattr(train, name)[indices]) for name in balanced._ARRAYS
    ]

    def loss(values):
        result = quadratic._rollout(values, norms, x, up, uf, 1)
        return jnp.mean(
            jnp.asarray(weights) * ((result - target) / args["error_scale"]) ** 2
        )

    grad = jax.grad(loss)(params)
    magnitude = jnp.sqrt(sum(jnp.sum(g * g) for g in jax.tree.leaves(grad)))
    grad = jax.tree.map(lambda g: g * jnp.minimum(1.0, 5.0 / (magnitude + 1e-12)), grad)
    first = jax.tree.map(lambda g: 0.1 * g, grad)
    second = jax.tree.map(lambda g: 0.001 * g * g, grad)
    rate = 1e-4
    expected = jax.tree.map(
        lambda w, m, v: w - rate * (m / (1 - 0.9)) / (jnp.sqrt(v / (1 - 0.999)) + 1e-8),
        params,
        first,
        second,
    )
    result, report = balanced.fit_candidate_sequence(
        train, train, steps=1, check_every=1, batch_size=7, learning_rate=rate, **args
    )
    assert report["selected_step"] == 1
    for key, values in expected.items():
        np.testing.assert_allclose(result.params[key], values, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize(
    "value,message",
    [
        (np.nan, "nonfinite initial training forecast"),
        (1e308, "nonfinite initial channel weighting"),
    ],
)
def test_nonfinite_weight_construction_is_an_explicit_numerical_failure(
    monkeypatch, value, message
):
    train = small_batch()
    monkeypatch.setattr(
        balanced,
        "initial_training_forecast",
        lambda *a: np.full(train.future_states.shape, value),
    )
    with pytest.raises(balanced.CandidateFitError, match=message):
        balanced.fit_candidate_sequence(
            train, small_batch(1), steps=1, **settings(train)
        )


def test_all_perfect_channels_are_finite_uniform_weights(monkeypatch):
    train = small_batch()
    monkeypatch.setattr(
        balanced, "initial_training_forecast", lambda *a: train.future_states.copy()
    )
    _, report = balanced.fit_candidate_sequence(
        train, train, steps=0, **settings(train)
    )
    np.testing.assert_array_equal(report["objective"]["initial_channel_mse"], 0)
    np.testing.assert_array_equal(report["objective"]["channel_weights"], 1)
    assert report["objective"]["floor_active_channels"] == [0, 1, 2]


def test_wrapper_preserves_public_revision_and_private_formats(monkeypatch, tmp_path):
    public, _, _ = prepared_pair()
    original = public.fingerprint()
    monkeypatch.setitem(balanced.FITTING_RECIPE, "steps", 2)
    monkeypatch.setitem(balanced.FITTING_RECIPE, "check_every", 1)
    fitted = balanced.fit_candidate(public)
    assert public.fingerprint() == original
    assert fitted._train is public._train and fitted._development is public._development
    assert type(fitted) is balanced.CandidateDynamics
    assert type(fitted._model) is balanced.BalancedQuadraticSequenceModel
    assert fitted.recipe == balanced.RECIPE
    assert fitted.report["optimization"]["selection_objective"] == balanced.OBJECTIVE
    path = tmp_path / "model.npz"
    fitted.save(path)
    loaded = balanced.CandidateDynamics.load(path)
    assert loaded.fingerprint() == fitted.fingerprint()
    with pytest.raises(ValueError, match="format"):
        anchored.CandidateDynamics.load(path)
    batch = public._train.batch
    args = (batch.past_states[:2], batch.past_inputs[:2], batch.future_inputs[:2])
    np.testing.assert_array_equal(loaded.predict(*args), fitted.predict(*args))
    np.testing.assert_allclose(
        jax.jit(loaded.predict)(*args), fitted.predict(*args), atol=1e-12
    )
    assert np.isfinite(
        jax.jacfwd(lambda u: loaded.predict(args[0][0], args[1][0], u))(
            jnp.asarray(args[2][0])
        )
    ).all()
    assert not hasattr(loaded, "update")
    assert anchored.RECIPE["id"] == "affine-anchored-quadratic-v1"
    assert "objective" not in anchored.RECIPE
    assert anchored.FITTING_RECIPE["steps"] == 1000
    assert balanced._wrapper._shared is not anchored._shared


@pytest.mark.parametrize("change", ["scale", "norm", "update"])
def test_invalid_public_preparation_aborts_before_fitter(monkeypatch, change):
    public, _, _ = prepared_pair()
    if change == "scale":
        public._report["optimization"]["error_scale"][0][0] += 1
    elif change == "norm":
        values = copy.deepcopy(public._model.norms)
        values["state_scale"][0] += 1
        public._model = replace(public._model, norms=values)
    else:
        public._report["previous_revision"] = "changed"
    monkeypatch.setattr(
        balanced._wrapper,
        "fit_candidate_sequence",
        lambda *a, **k: pytest.fail("invalid preparation fitted"),
    )
    with pytest.raises(ValueError):
        balanced.fit_candidate(public)
