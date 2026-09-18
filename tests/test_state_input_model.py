"""Mechanism witnesses for the isolated state/input interaction experiment."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from glassbox._sequence_model import SequenceBatch, initialize_sequence_model
from glassbox.experimental.state_input_model import (
    RECIPE,
    BilinearSequenceModel,
    CandidateDynamics,
    fit_candidate_sequence,
    initialize_candidate,
    interaction_features,
)
from glassbox.recordings import SequenceWindows, WindowKey


@pytest.fixture(autouse=True)
def float64():
    with jax.enable_x64(True):
        yield


def small_batch(seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(24, 7, 3))
    u = rng.normal(size=(24, 6, 2))
    # Constant channels exercise zero-variance scaling and zero product columns.
    x[..., 2], u[..., 1] = 2.0, -3.0
    return SequenceBatch(x[:, :5], u[:, :4], u[:, 4:], x[:, 5:], 0.05)


def base_with_active_memory(batch):
    base = initialize_sequence_model(batch, width=4, memory=2, delay_steps=1)
    params = {key: value.copy() for key, value in base.params.items()}
    params["linear"][-2:] = 0.02
    params["w2"][:] = 0.01
    return replace(base, params=params)


def as_candidate(base, interaction=None):
    dimension = len(base.norms["state_mean"]) * len(base.norms["input_mean"])
    return BilinearSequenceModel(
        base.kind,
        base.dt_s,
        base.history_steps,
        {
            **{key: value.copy() for key, value in base.params.items()},
            "interaction": np.zeros((dimension, len(base.norms["state_mean"])))
            if interaction is None
            else interaction,
        },
        {
            **{key: value.copy() for key, value in base.norms.items()},
            "interaction_scale": np.ones(dimension),
        },
        base.delay_steps,
    )


def test_products_have_state_major_input_minor_order_for_all_leading_axes():
    state = np.arange(12.0).reshape(2, 2, 3) - 2
    command = np.arange(8.0).reshape(2, 2, 2) + 0.5
    expected = np.stack(
        [state[..., i] * command[..., j] for i in range(3) for j in range(2)],
        axis=-1,
    )
    np.testing.assert_array_equal(interaction_features(state, command, xp=np), expected)
    np.testing.assert_array_equal(
        jax.jit(interaction_features)(state, command), expected
    )
    np.testing.assert_array_equal(
        interaction_features(state[0, 0], command[0, 0]), expected[0, 0]
    )


def test_command_jacobian_obeys_analytic_state_dependent_physical_scaling():
    batch = small_batch()
    model = as_candidate(base_with_active_memory(batch))
    params = {key: np.zeros_like(value) for key, value in model.params.items()}
    params["interaction"] = np.arange(18.0).reshape(6, 3) / 17 - 0.4
    norms = {
        **model.norms,
        "state_mean": np.array([0.4, -0.7, 1.2]),
        "state_scale": np.array([2.0, 3.0, 0.5]),
        "input_mean": np.array([-0.2, 0.6]),
        "input_scale": np.array([4.0, 0.25]),
        "delta_scale": np.array([0.2, 0.3, 0.4]),
        "interaction_scale": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    }
    model = replace(model, params=params, norms=norms)
    command = np.array([[0.3, -0.2]])
    past_u = np.zeros((4, 2))
    jacobians = []
    for current in (np.array([1.0, 2.0, 3.0]), np.array([-2.0, 0.5, 2.0])):
        history = np.broadcast_to(current, (5, 3))
        normalized_x = (current - norms["state_mean"]) / norms["state_scale"]
        normalized_u = (command[0] - norms["input_mean"]) / norms["input_scale"]
        tensor = (params["interaction"] / norms["interaction_scale"][:, None]).reshape(
            3, 2, 3
        )
        output_scale = norms["state_scale"] * norms["delta_scale"]
        expected_output = current + output_scale * np.einsum(
            "i,j,ijk->k", normalized_x, normalized_u, tensor
        )
        expected_jacobian = (
            np.einsum("i,ijk->kj", normalized_x, tensor)
            * output_scale[:, None]
            / norms["input_scale"][None, :]
        )
        actual = model.rollout(history, past_u, command)
        jacobian = jax.jacfwd(
            lambda u, history=history: model.rollout(history, past_u, u)[0]
        )(jnp.asarray(command))[:, 0, :]
        np.testing.assert_allclose(actual[0], expected_output, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(jacobian, expected_jacobian, atol=1e-12, rtol=1e-12)
        jacobians.append(np.asarray(jacobian))
    assert not np.allclose(*jacobians)


def test_zero_interaction_preserves_baseline_batch_jit_and_input_derivatives():
    batch = small_batch()
    base = base_with_active_memory(batch)
    model = as_candidate(base)
    x, up, uf = batch.past_states[:3], batch.past_inputs[:3], batch.future_inputs[:3]
    expected = base.rollout(x, up, uf)
    np.testing.assert_allclose(
        model.rollout(x, up, uf), expected, atol=1e-12, rtol=1e-12
    )
    np.testing.assert_allclose(jax.jit(model.rollout)(x, up, uf), expected, atol=1e-12)
    np.testing.assert_allclose(
        model.rollout(x[0], up[0], uf[0]), expected[0], atol=1e-12
    )
    np.testing.assert_allclose(
        jax.jacfwd(lambda u: model.rollout(x[0], up[0], u))(jnp.asarray(uf[0])),
        jax.jacfwd(lambda u: base.rollout(x[0], up[0], u))(jnp.asarray(uf[0])),
        atol=1e-12,
        rtol=1e-12,
    )


def test_interaction_leaves_observed_memory_and_context_carry_contract_unchanged():
    batch = small_batch()
    base = base_with_active_memory(batch)
    model = as_candidate(base, np.full((6, 3), 0.02))
    x, up, uf = batch.past_states[:3], batch.past_inputs[:3], batch.future_inputs[:3]
    np.testing.assert_array_equal(model.memory_state(x, up), base.memory_state(x, up))
    assert np.abs(model.memory_state(x, up)).max() > 0
    full = model.rollout(x, up, uf)
    split = 2
    memory = model.memory_state(x[:, : split + 1], up[:, :split])
    continued = model.rollout(
        x[:, split - model.delay_steps :],
        up[:, split - model.delay_steps :],
        uf,
        memory=memory,
    )
    np.testing.assert_allclose(continued, full, atol=1e-12, rtol=1e-12)
    with pytest.raises(ValueError, match="shapes"):
        model.rollout(x[:, 1:], up[:, 1:], uf)
    with pytest.raises(ValueError, match="shapes"):
        model.rollout(x[:, -2:], up[:, -1:], uf)


def test_recursive_products_use_predicted_state_and_preserve_causal_prefixes():
    base = initialize_sequence_model(small_batch(), width=4, memory=2, delay_steps=1)
    model = as_candidate(base)
    params = {key: np.zeros_like(value) for key, value in model.params.items()}
    # Only x_0 * u_0 affects x_0, giving an independently solvable recurrence.
    params["interaction"][0, 0] = 0.2
    norms = {key: np.ones_like(value) for key, value in model.norms.items()}
    norms["state_mean"][:] = 0
    norms["input_mean"][:] = 0
    model = replace(model, params=params, norms=norms)
    x = np.tile([2.0, 0.0, 0.0], (5, 1))
    up = np.zeros((4, 2))
    uf = np.array([[0.5, 0.0], [-0.2, 0.0], [0.8, 0.0]])
    expected = 2.0 * np.cumprod(1 + 0.2 * uf[:, 0])
    predicted = np.asarray(jax.jit(model.rollout)(x, up, uf))
    np.testing.assert_allclose(predicted[:, 0], expected, atol=1e-12)
    np.testing.assert_array_equal(predicted[:, 1:], 0)
    np.testing.assert_allclose(model.rollout(x, up, uf[:2]), predicted[:2], atol=1e-12)
    jacobian = jax.jacfwd(lambda u: model.rollout(x, up, u))(jnp.asarray(uf))
    for horizon in range(len(uf)):
        np.testing.assert_array_equal(jacobian[horizon, :, horizon + 1 :], 0)


def test_initialization_preserves_normalization_and_random_draws():
    batch = small_batch()
    settings = dict(seed=17, width=4, memory=2, ridge=0.7, delay_steps=1)
    base = initialize_sequence_model(batch, **settings)
    candidate = initialize_candidate(batch, **settings)
    for key, value in base.norms.items():
        np.testing.assert_array_equal(candidate.norms[key], value)
    for key in ("w1", "w2", "b1", "memory", "memory_bias"):
        np.testing.assert_array_equal(candidate.params[key], base.params[key])
    np.testing.assert_array_equal(candidate.params["linear"][-2:], 0)
    current = np.concatenate(
        (batch.past_states[:, -1:], batch.future_states[:, :-1]), 1
    )
    normalized_x = (current - base.norms["state_mean"]) / base.norms["state_scale"]
    normalized_u = (batch.future_inputs - base.norms["input_mean"]) / base.norms[
        "input_scale"
    ]
    products = (normalized_x[..., :, None] * normalized_u[..., None, :]).reshape(
        len(current), current.shape[1], -1
    )
    scale = products.std((0, 1))
    np.testing.assert_allclose(
        candidate.norms["interaction_scale"],
        np.where(scale > 1e-8, scale, 1),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_array_equal(candidate.norms["interaction_scale"][[1, 3, 4, 5]], 1)
    np.testing.assert_array_equal(candidate.params["interaction"][[1, 3, 4, 5]], 0)


def test_development_data_cannot_change_initial_weights_or_scales():
    train, development = small_batch(), small_batch(1)
    shifted = SequenceBatch(
        development.past_states + 10,
        development.past_inputs - 7,
        development.future_inputs - 7,
        development.future_states + 10,
        development.dt_s,
    )
    settings = dict(steps=0, width=4, memory=2, delay_steps=1)
    model, report = fit_candidate_sequence(train, development, **settings)
    shifted_model, shifted_report = fit_candidate_sequence(train, shifted, **settings)
    assert report["selected_step"] == shifted_report["selected_step"] == 0
    for key, value in model.arrays().items():
        np.testing.assert_array_equal(shifted_model.arrays()[key], value)
    assert report["validation_rollout_mse"] != shifted_report["validation_rollout_mse"]


def test_joint_ridge_matches_independent_normal_equations():
    batch = small_batch(4)
    model = initialize_candidate(batch, width=4, memory=2, ridge=0.7, delay_steps=1)
    n = model.norms
    x = (
        np.concatenate((batch.past_states, batch.future_states), 1) - n["state_mean"]
    ) / n["state_scale"]
    u = (
        np.concatenate((batch.past_inputs, batch.future_inputs), 1) - n["input_mean"]
    ) / n["input_scale"]
    rows, targets = [], []
    for i in range(len(x)):
        for j in range(4, 6):
            # Independent, explicit feature enumeration avoids calling learner helpers.
            z = np.r_[
                x[i, j], u[i, j], x[i, j - 1] - x[i, j], u[i, j - 1] - u[i, j], 0, 0
            ]
            q = np.outer(x[i, j], u[i, j]).reshape(-1)
            rows.append(np.r_[z / n["feature_scale"], q / n["interaction_scale"], 1])
            targets.append((x[i, j + 1] - x[i, j]) / n["delta_scale"])
    design, target = np.asarray(rows), np.asarray(targets)
    penalty = np.diag(np.r_[np.full(design.shape[1] - 1, 0.7), 0.0])
    expected = np.linalg.solve(design.T @ design + penalty, design.T @ target)
    actual = np.concatenate(
        (
            model.params["linear"],
            model.params["interaction"],
            model.params["bias"][None],
        )
    )
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)


def test_joint_ridge_recovers_a_bilinear_law_that_affine_initialization_cannot():
    state, command = np.meshgrid([-2.0, -1.0, 1.0, 2.0], [-0.9, -0.3, 0.3, 0.9])
    state, command = state.ravel(), command.ravel()
    batch = SequenceBatch(
        np.broadcast_to(state[:, None, None], (16, 4, 1)),
        np.zeros((16, 3, 1)),
        command[:, None, None],
        (state * (1 + 0.2 * command))[:, None, None],
        0.05,
    )
    settings = dict(width=4, memory=2, ridge=1e-8, delay_steps=1)
    candidate, baseline = (
        initialize_candidate(batch, **settings),
        initialize_sequence_model(batch, **settings),
    )
    inputs = (batch.past_states, batch.past_inputs, batch.future_inputs)
    candidate_error = np.sqrt(
        np.mean((candidate.rollout(*inputs) - batch.future_states) ** 2)
    )
    baseline_error = np.sqrt(
        np.mean((baseline.rollout(*inputs) - batch.future_states) ** 2)
    )
    assert candidate_error < 1e-8
    assert baseline_error > 0.1


def test_saved_candidate_replays_and_rejects_changed_interaction(tmp_path):
    batch = small_batch()
    model = initialize_candidate(batch, width=4, memory=2, delay_steps=1)
    path = tmp_path / "candidate.npz"
    model.save(path)
    loaded = BilinearSequenceModel.load(path)
    assert loaded.fingerprint() == model.fingerprint()
    inputs = (batch.past_states, batch.past_inputs, batch.future_inputs)
    np.testing.assert_array_equal(loaded.rollout(*inputs), model.rollout(*inputs))
    with np.load(path, allow_pickle=False) as archive:
        values = {key: archive[key].copy() for key in archive.files}
    values["param_interaction"][0, 0] += 0.01
    np.savez_compressed(path, **values)
    with pytest.raises(ValueError, match="fingerprint"):
        BilinearSequenceModel.load(path)


def test_saved_wrapper_keeps_window_provenance_prediction_and_envelope_contract(
    tmp_path,
):
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
    envelope_copy, contract_copy = loaded.envelope(), loaded.contract
    envelope_copy[:] = 8
    contract_copy["state_channels"][0] = "changed"
    assert loaded.contract == contract
    np.testing.assert_array_equal(loaded.envelope(), np.full((2, 3), 0.3))
    with pytest.raises(ValueError, match="horizon"):
        loaded.predict(x, up, np.tile(uf, (1, 2, 1)))
    with pytest.raises(ValueError, match="history"):
        loaded.predict(x[:, 1:], up[:, 1:], uf)
    assert not hasattr(
        loaded, "update"
    )  # This research result claims no public adoption.
