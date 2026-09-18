"""Independent mechanism and compatibility witnesses for the quadratic path."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from glassbox._sequence_model import SequenceBatch, initialize_sequence_model
from glassbox.experimental import state_input_model as historical
from glassbox.experimental import state_quadratic_model as quadratic
from glassbox.experimental.state_quadratic_model import (
    RECIPE,
    CandidateDynamics,
    QuadraticSequenceModel,
    autonomous_features,
    fit_candidate_sequence,
    initialize_candidate,
)
from glassbox.recordings import SequenceWindows, WindowKey


@pytest.fixture(autouse=True)
def float64():
    with jax.enable_x64(True):
        yield


def small_batch(seed=0):
    rng = np.random.default_rng(seed)
    x, u = rng.normal(size=(24, 7, 3)), rng.normal(size=(24, 6, 2))
    x[..., 2], u[..., 1] = 2.0, -3.0
    return SequenceBatch(x[:, :5], u[:, :4], u[:, 4:], x[:, 5:], 0.05)


def bilinear_with_memory(batch):
    base = historical.initialize_candidate(batch, width=4, memory=2, delay_steps=1)
    params = {key: value.copy() for key, value in base.params.items()}
    params["linear"][-2:] = 0.02
    params["w2"][:] = 0.01
    return replace(base, params=params)


def add_autonomous(base, coefficients=None):
    d = len(base.norms["state_mean"])
    count = d * (d + 1) // 2
    return QuadraticSequenceModel(
        base.kind,
        base.dt_s,
        base.history_steps,
        {
            **base.params,
            "autonomous": np.zeros((count, d))
            if coefficients is None
            else coefficients,
        },
        {**base.norms, "autonomous_scale": np.ones(count)},
        base.delay_steps,
    )


def test_upper_triangle_keeps_squares_and_cross_products_once_in_lexical_order():
    x = np.arange(12.0).reshape(2, 2, 3) - 2.5
    expected = np.stack(
        [x[..., i] * x[..., j] for i in range(3) for j in range(i, 3)], axis=-1
    )
    np.testing.assert_array_equal(autonomous_features(x, xp=np), expected)
    np.testing.assert_array_equal(jax.jit(autonomous_features)(x), expected)
    np.testing.assert_array_equal(autonomous_features(x[0, 0]), expected[0, 0])


def test_zero_quadratic_path_preserves_bilinear_batch_jit_and_derivatives():
    batch = small_batch()
    base = bilinear_with_memory(batch)
    candidate = add_autonomous(base)
    x, up, uf = batch.past_states[:3], batch.past_inputs[:3], batch.future_inputs[:3]
    expected = base.rollout(x, up, uf)
    np.testing.assert_allclose(candidate.rollout(x, up, uf), expected, atol=1e-12)
    np.testing.assert_allclose(
        jax.jit(candidate.rollout)(x, up, uf), expected, atol=1e-12
    )
    np.testing.assert_allclose(
        candidate.rollout(x[0], up[0], uf[0]), expected[0], atol=1e-12
    )
    np.testing.assert_allclose(
        jax.jacfwd(lambda u: candidate.rollout(x[0], up[0], u))(jnp.asarray(uf[0])),
        jax.jacfwd(lambda u: base.rollout(x[0], up[0], u))(jnp.asarray(uf[0])),
        atol=1e-12,
    )


def test_multistep_recurrence_and_command_gradient_include_predicted_quadratic_state():
    model = add_autonomous(bilinear_with_memory(small_batch()))
    params = {key: np.zeros_like(value) for key, value in model.params.items()}
    alpha, beta, gamma = -0.07, 0.13, 0.2
    params["autonomous"][0, 0] = alpha
    params["interaction"][0, 0] = beta
    params["linear"][3, 0] = gamma
    norms = {key: np.ones_like(value) for key, value in model.norms.items()}
    norms["state_mean"][:] = 0
    norms["input_mean"][:] = 0
    model = replace(model, params=params, norms=norms)
    x = np.tile([1.7, 0.0, 0.0], (5, 1))
    up = np.zeros((4, 2))
    uf = np.array([[0.5, 0.0], [-0.2, 0.0], [0.8, 0.0]])
    expected, gradients, state = [], [], x[-1, 0]
    derivative = np.zeros(len(uf))
    for t, command in enumerate(uf[:, 0]):
        derivative = (1 + 2 * alpha * state + beta * command) * derivative
        derivative[t] += beta * state + gamma
        state = state + alpha * state**2 + beta * state * command + gamma * command
        expected.append(state)
        gradients.append(derivative.copy())
    predicted = np.asarray(jax.jit(model.rollout)(x, up, uf))
    jacobian = np.asarray(
        jax.jacfwd(lambda u: model.rollout(x, up, u))(jnp.asarray(uf))
    )
    np.testing.assert_allclose(predicted[:, 0], expected, atol=1e-12)
    np.testing.assert_array_equal(predicted[:, 1:], 0)
    np.testing.assert_allclose(jacobian[:, 0, :, 0], gradients, atol=1e-12)
    np.testing.assert_array_equal(jacobian[:, 1:], 0)
    np.testing.assert_array_equal(jacobian[..., 1], 0)
    np.testing.assert_allclose(model.rollout(x, up, uf[:2]), predicted[:2], atol=1e-12)
    for t in range(len(uf)):
        np.testing.assert_array_equal(jacobian[t, :, t + 1 :], 0)


def test_nonunit_normalization_and_cross_terms_match_independent_physical_output():
    model = add_autonomous(bilinear_with_memory(small_batch()))
    params = {key: np.zeros_like(value) for key, value in model.params.items()}
    params["autonomous"][:] = np.arange(18).reshape(6, 3) / 23 - 0.2
    norms = {
        **model.norms,
        "state_mean": np.array([0.4, -0.7, 1.2]),
        "state_scale": np.array([2.0, 3.0, 0.5]),
        "delta_scale": np.array([0.2, 0.3, 0.4]),
        "autonomous_scale": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    }
    model = replace(model, params=params, norms=norms)
    state = np.array([1.0, 2.0, 3.0])
    normalized = (state - norms["state_mean"]) / norms["state_scale"]
    products = np.array(
        [normalized[i] * normalized[j] for i in range(3) for j in range(i, 3)]
    )
    expected = state + norms["state_scale"] * norms["delta_scale"] * (
        (products / norms["autonomous_scale"]) @ params["autonomous"]
    )
    prediction = model.rollout(
        np.tile(state, (5, 1)), np.zeros((4, 2)), np.zeros((1, 2))
    )
    np.testing.assert_allclose(prediction[0], expected, atol=1e-12)


def test_observed_memory_and_carried_context_do_not_receive_quadratic_features():
    batch = small_batch()
    base = bilinear_with_memory(batch)
    model = add_autonomous(base, np.full((6, 3), 0.02))
    x, up, uf = batch.past_states[:3], batch.past_inputs[:3], batch.future_inputs[:3]
    np.testing.assert_array_equal(model.memory_state(x, up), base.memory_state(x, up))
    assert np.abs(model.memory_state(x, up)).max() > 0
    split = 2
    memory = model.memory_state(x[:, : split + 1], up[:, :split])
    continued = model.rollout(
        x[:, split - model.delay_steps :],
        up[:, split - model.delay_steps :],
        uf,
        memory=memory,
    )
    np.testing.assert_allclose(continued, model.rollout(x, up, uf), atol=1e-12)
    with pytest.raises(ValueError, match="shapes"):
        model.rollout(x[:, 1:], up[:, 1:], uf)


def test_initialization_preserves_shared_draws_scales_and_zero_variance_handling():
    batch = small_batch()
    settings = dict(seed=17, width=4, memory=2, ridge=0.7, delay_steps=1)
    base = initialize_sequence_model(batch, **settings)
    bilinear = historical.initialize_candidate(batch, **settings)
    model = initialize_candidate(batch, **settings)
    for name, value in bilinear.norms.items():
        np.testing.assert_array_equal(model.norms[name], value)
    for name in ("w1", "w2", "b1", "memory", "memory_bias"):
        np.testing.assert_array_equal(model.params[name], base.params[name])
    current = np.concatenate(
        (batch.past_states[:, -1:], batch.future_states[:, :-1]), 1
    )
    normalized = (current - base.norms["state_mean"]) / base.norms["state_scale"]
    products = np.stack(
        [
            normalized[..., i] * normalized[..., j]
            for i in range(3)
            for j in range(i, 3)
        ],
        axis=-1,
    )
    scale = products.reshape(-1, 6).std(0)
    np.testing.assert_allclose(
        model.norms["autonomous_scale"],
        np.where(scale > 1e-8, scale, 1),
        rtol=1e-14,
        atol=1e-14,
    )
    np.testing.assert_array_equal(model.norms["autonomous_scale"][[2, 4, 5]], 1)
    np.testing.assert_array_equal(model.params["autonomous"][[2, 4, 5]], 0)
    np.testing.assert_array_equal(model.params["linear"][-2:], 0)
    assert sum(v.size for v in model.params.values()) == (
        sum(v.size for v in bilinear.params.values()) + 18
    )


def test_joint_ridge_matches_independent_explicit_normal_equations():
    batch = small_batch(4)
    model = initialize_candidate(batch, width=4, memory=2, ridge=0.7, delay_steps=1)
    n = model.norms
    physical = np.concatenate((batch.past_states, batch.future_states), 1)
    x = (physical - n["state_mean"]) / n["state_scale"]
    u = (
        np.concatenate((batch.past_inputs, batch.future_inputs), 1) - n["input_mean"]
    ) / n["input_scale"]
    rows, targets = [], []
    for i in range(len(x)):
        for j in range(4, 6):
            z = np.r_[
                x[i, j], u[i, j], x[i, j - 1] - x[i, j], u[i, j - 1] - u[i, j], 0, 0
            ]
            xu = np.array([x[i, j, a] * u[i, j, b] for a in range(3) for b in range(2)])
            xx = np.array(
                [x[i, j, a] * x[i, j, b] for a in range(3) for b in range(a, 3)]
            )
            rows.append(
                np.r_[
                    z / n["feature_scale"],
                    xu / n["interaction_scale"],
                    xx / n["autonomous_scale"],
                    1,
                ]
            )
            targets.append(
                (physical[i, j + 1] - physical[i, j])
                / n["state_scale"]
                / n["delta_scale"]
            )
    design, target = np.asarray(rows), np.asarray(targets)
    penalty = np.diag(np.r_[np.full(design.shape[1] - 1, 0.7), 0.0])
    expected = np.linalg.solve(design.T @ design + penalty, design.T @ target)
    actual = np.concatenate(
        [model.params[name] for name in ("linear", "interaction", "autonomous")]
        + [model.params["bias"][None]]
    )
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)


def test_quadratic_basis_recovers_autonomous_law_that_bilinear_cannot():
    state, command = np.meshgrid([-2.0, -1.0, 1.0, 2.0], [-0.9, -0.3, 0.3, 0.9])
    state, command = state.ravel(), command.ravel()
    target = state + 0.1 * state**2 + 0.2 * state * command
    batch = SequenceBatch(
        np.broadcast_to(state[:, None, None], (16, 4, 1)),
        np.zeros((16, 3, 1)),
        command[:, None, None],
        target[:, None, None],
        0.05,
    )
    settings = dict(width=4, memory=2, ridge=1e-8, delay_steps=1)
    candidate = initialize_candidate(batch, **settings)
    bilinear = historical.initialize_candidate(batch, **settings)
    inputs = (batch.past_states, batch.past_inputs, batch.future_inputs)
    candidate_error = np.sqrt(
        np.mean((candidate.rollout(*inputs) - batch.future_states) ** 2)
    )
    bilinear_error = np.sqrt(
        np.mean((bilinear.rollout(*inputs) - batch.future_states) ** 2)
    )
    assert candidate_error < 1e-8
    assert bilinear_error > 0.1


def test_development_data_does_not_change_initial_weights_or_normalization():
    train, development = small_batch(), small_batch(1)
    shifted = SequenceBatch(
        development.past_states + 0.7,
        development.past_inputs - 0.4,
        development.future_inputs - 0.4,
        development.future_states + 0.7,
        development.dt_s,
    )
    settings = dict(steps=0, width=4, memory=2, delay_steps=1)
    model, report = fit_candidate_sequence(train, development, **settings)
    other, other_report = fit_candidate_sequence(train, shifted, **settings)
    for name, value in model.arrays().items():
        np.testing.assert_array_equal(other.arrays()[name], value)
    assert report["validation_rollout_mse"] != other_report["validation_rollout_mse"]
    assert report["mechanism"] == RECIPE["id"]


def test_private_optimizer_runs_quadratic_parameters_without_altering_historical_module():
    model, report = fit_candidate_sequence(
        small_batch(),
        small_batch(1),
        steps=2,
        check_every=1,
        width=4,
        memory=2,
        delay_steps=1,
    )
    assert isinstance(model, QuadraticSequenceModel)
    assert len(report["trace"]) == 3
    assert np.isfinite(report["validation_rollout_mse"])
    assert historical.RECIPE["id"] == "state-input-interaction-v1"
    assert historical._MODEL_FORMAT == "glassbox-state-input-sequence-v1"
    assert historical._FORMAT == "glassbox-state-input-candidate-v1"
    assert historical.fit_candidate_sequence.__globals__ is historical.__dict__
    assert quadratic.fit_candidate_sequence.__globals__ is quadratic._shared.__dict__
    assert (
        historical.fit_candidate_sequence.__code__.co_code
        == quadratic.fit_candidate_sequence.__code__.co_code
    )
    assert (
        historical.fit_candidate.__code__.co_code
        == quadratic.fit_candidate.__code__.co_code
    )
    historical_model = historical.initialize_candidate(
        small_batch(), width=4, memory=2, delay_steps=1
    )
    assert type(historical_model) is historical.BilinearSequenceModel
    assert "autonomous" not in historical_model.params
    assert historical.CandidateDynamics is not CandidateDynamics
    assert historical._rollout is not quadratic._rollout


@pytest.mark.parametrize("value", [np.zeros((5, 3)), np.full((6, 3), np.nan)])
def test_bad_quadratic_coefficients_are_rejected(value):
    model = initialize_candidate(small_batch(), width=4, memory=2, delay_steps=1)
    with pytest.raises(ValueError, match="autonomous coefficients"):
        replace(model, params={**model.params, "autonomous": value})


@pytest.mark.parametrize(
    "value", [np.ones(5), np.zeros(6), -np.ones(6), np.full(6, np.nan)]
)
def test_bad_quadratic_scales_are_rejected(value):
    model = initialize_candidate(small_batch(), width=4, memory=2, delay_steps=1)
    with pytest.raises(ValueError, match="autonomous scales"):
        replace(model, norms={**model.norms, "autonomous_scale": value})


def test_sequence_save_replays_and_rejects_modified_quadratic_coefficients(tmp_path):
    batch = small_batch()
    model = initialize_candidate(batch, width=4, memory=2, delay_steps=1)
    path = tmp_path / "sequence.npz"
    model.save(path)
    loaded = QuadraticSequenceModel.load(path)
    assert loaded.fingerprint() == model.fingerprint()
    assert loaded.metadata()["format"] == "glassbox-state-quadratic-sequence-v1"
    inputs = (batch.past_states, batch.past_inputs, batch.future_inputs)
    np.testing.assert_array_equal(loaded.rollout(*inputs), model.rollout(*inputs))
    with pytest.raises(ValueError, match="format"):
        historical.BilinearSequenceModel.load(path)
    with np.load(path, allow_pickle=False) as archive:
        values = {name: archive[name].copy() for name in archive.files}
    values["param_autonomous"][0, 0] += 0.01
    np.savez_compressed(path, **values)
    with pytest.raises(ValueError, match="fingerprint"):
        QuadraticSequenceModel.load(path)


def test_saved_wrapper_preserves_caches_contract_predictions_and_envelope(tmp_path):
    batch = small_batch()
    sequence = initialize_candidate(batch, width=4, memory=2, delay_steps=1)
    windows = SequenceWindows(
        batch,
        tuple(
            WindowKey(f"recording-{i}", "segment", 4)
            for i in range(len(batch.past_states))
        ),
        (4,) * len(batch.past_states),
    )
    contract = {
        "state_channels": ["a [m]", "b [rad]", "c [m/s]"],
        "input_channels": ["u [N]", "v [N]"],
        "dt_s": batch.dt_s,
    }
    model = CandidateDynamics(
        sequence, windows, windows, contract, {"recipe": RECIPE}, np.full((2, 3), 0.3)
    )
    path = tmp_path / "wrapped.npz"
    model.save(path)
    loaded = CandidateDynamics.load(path)
    assert loaded.fingerprint() == model.fingerprint()
    assert loaded._train.keys == windows.keys
    assert loaded._development.source_origins == windows.source_origins
    assert loaded.contract == contract
    assert loaded.history_steps == 4 and loaded.horizon_steps == 2
    x, up, uf = batch.past_states[:2], batch.past_inputs[:2], batch.future_inputs[:2]
    expected = sequence.rollout(x, up, uf)
    np.testing.assert_array_equal(loaded.predict(x, up, uf), expected)
    np.testing.assert_allclose(jax.jit(loaded.predict)(x, up, uf), expected, atol=1e-12)
    np.testing.assert_allclose(
        loaded.predict(x[0], up[0], uf[0]), expected[0], atol=1e-12
    )
    extended_x = np.concatenate((np.full((2, 2, 3), -100.0), x), axis=1)
    extended_u = np.concatenate((np.full((2, 2, 2), 100.0), up), axis=1)
    np.testing.assert_array_equal(loaded.predict(extended_x, extended_u, uf), expected)
    np.testing.assert_array_equal(loaded.envelope(1), np.full((1, 3), 0.3))
    with pytest.raises(ValueError, match="format"):
        historical.CandidateDynamics.load(path)
    with pytest.raises(ValueError, match="history"):
        loaded.predict(x[:, 1:], up[:, 1:], uf)
    with pytest.raises(ValueError, match="horizon"):
        loaded.predict(x, up, np.tile(uf, (1, 2, 1)))
    assert not hasattr(loaded, "update")
