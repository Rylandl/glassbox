"""Analytic vehicles and bounded fitting only; no simulator fixtures."""

from dataclasses import replace
from itertools import pairwise

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from glassbox._sequence_model import SequenceBatch, SequenceFitError
from glassbox.experimental import shared_vehicle_core as core


def constant_model(commands=3, *, dt=0.05, history=3, delay=2):
    b = 9 + 2 * commands
    f = (delay + 1) * b + 2
    q = b * (b + 1) // 2
    params = dict(
        linear=np.zeros((f, 6)),
        quadratic=np.zeros((q, 6)),
        bias=np.zeros(6),
        w1=np.zeros((f, 3)),
        b1=np.zeros(3),
        w2=np.zeros((3, 6)),
        memory=np.zeros((f, 2)),
        memory_bias=np.zeros(2),
        raw_tau=np.full(commands, np.log(np.expm1(0.049))),
    )
    norms = dict(
        body_mean=np.zeros(9),
        body_scale=np.ones(9),
        input_mean=np.zeros(commands),
        input_scale=np.ones(commands),
        feature_scale=np.ones(f),
        quadratic_scale=np.ones(q),
        output_scale=np.ones(6),
        state_mean=np.zeros(15),
        state_scale=np.ones(15),
    )
    return core.VehicleSequenceModel(dt, history, delay, params, norms)


def inputs(model, horizon=4, *, omega=None):
    state = np.r_[np.zeros(6), np.eye(3).reshape(9)]
    if omega is not None:
        state[3:6] = omega
    return (
        np.repeat(state[None], model.history_steps + 1, 0),
        np.zeros((model.history_steps, len(model.norms["input_mean"]))),
        np.zeros((horizon, len(model.norms["input_mean"]))),
    )


def analytic_batch(commands=3):
    # Native dt .05 gives two physical substeps and a two-step delay.
    n, context, horizon, dt = 4, 3, 3, 0.05
    rng = np.random.default_rng(42)
    commands_array = rng.normal(size=(n, context + horizon, commands)) * 0.3
    states = np.zeros((n, context + horizon + 1, 15))
    states[..., 6:] = np.eye(3).reshape(9)
    for t in range(context + horizon):
        acceleration = np.zeros((n, 3))
        acceleration[:, 0] = commands_array[:, t].sum(-1)
        acceleration[:, 2] = core.GRAVITY[2]
        states[:, t + 1, :3] = states[:, t, :3] + dt * acceleration
    return SequenceBatch(
        states[:, : context + 1],
        commands_array[:, :context],
        commands_array[:, context:],
        states[:, context + 1 :],
        dt,
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_rotation_exp_zero_and_derivatives(dtype):
    with jax.enable_x64(dtype is np.float64):
        zero = jnp.zeros(3, dtype=dtype)
        np.testing.assert_array_equal(core.rotation_exp(zero), np.eye(3))
        jacobian = jax.jacfwd(core.rotation_exp)(zero)
        assert np.isfinite(jacobian).all()
        assert np.isfinite(jax.jacrev(core.rotation_exp)(zero)).all()
        assert np.isfinite(jax.jacfwd(jax.jacrev(core.rotation_exp))(zero)).all()
        expected = np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]])
        np.testing.assert_array_equal(jacobian[..., 0], expected)
        np.testing.assert_array_equal(jax.jit(core.rotation_exp)(zero), np.eye(3))


@pytest.mark.parametrize("commands", [1, 3, 4, 6])
def test_freefall_and_constant_body_acceleration(commands):
    with jax.enable_x64(True):
        model = constant_model(commands)
        args = inputs(model)
        predicted = np.asarray(model.rollout(*args))
        times = model.dt_s * np.arange(1, 5)
        np.testing.assert_allclose(
            predicted[:, :3], times[:, None] * np.asarray(core.GRAVITY), atol=1e-14
        )
        np.testing.assert_allclose(
            predicted[:, 6:], np.broadcast_to(np.eye(3).reshape(9), (4, 9)), atol=1e-15
        )
        params = {k: v.copy() for k, v in model.params.items()}
        params["bias"][:3] = [2.0, 3.0, 9.80665]
        accelerated = replace(model, params=params)
        np.testing.assert_allclose(
            accelerated.rollout(*args)[:, :3], times[:, None] * [2, 3, 0], atol=1e-14
        )


def test_long_rotation_matches_analytic_and_stays_on_so3():
    with jax.enable_x64(True):
        model = constant_model(dt=0.01, history=11, delay=10)
        omega = np.array([0.7, -0.2, 1.1])
        predicted = np.asarray(model.rollout(*inputs(model, 120, omega=omega)))
        rotations = predicted[:, 6:].reshape(-1, 3, 3)
        expected = np.asarray(
            core.rotation_exp(np.arange(1, 121)[:, None] * 0.01 * omega)
        )
        np.testing.assert_allclose(rotations, expected, atol=2e-14)
        np.testing.assert_allclose(
            rotations.swapaxes(-1, -2) @ rotations,
            np.broadcast_to(np.eye(3), rotations.shape),
            atol=2e-14,
        )
        np.testing.assert_allclose(np.linalg.det(rotations), 1, atol=2e-14)


def test_midpoint_angular_acceleration_and_two_substeps():
    with jax.enable_x64(True):
        model = constant_model()
        params = {k: v.copy() for k, v in model.params.items()}
        params["bias"][5] = 2.0
        model = replace(model, params=params)
        y = np.asarray(model.rollout(*inputs(model, 1)))
        np.testing.assert_allclose(y[0, 3:6], [0, 0, 0.1], atol=1e-15)
        np.testing.assert_allclose(
            y[0, 6:].reshape(3, 3),
            core.rotation_exp(np.array([0, 0, 0.0025])),
            atol=1e-15,
        )


def test_command_filter_uses_entire_prefix_and_raw_commands_are_available():
    with jax.enable_x64(True):
        model = constant_model(commands=1)
        past, up, _ = inputs(model, 2)
        up[:, 0] = [1, 0, 0]
        params, norms = jax.tree.map(jnp.asarray, (model.params, model.norms))
        a, _, _ = core._history(
            params, norms, jnp.asarray(past[None]), jnp.asarray(up[None]), 2, 0.05
        )
        np.testing.assert_allclose(a, [[np.exp(-2.0)]], atol=1e-15)
        b = core.current_features(
            past[-1], np.array([0.7]), np.array([0.2]), model.norms
        )
        np.testing.assert_array_equal(b[-2:], [0.7, 0.2])
        assert np.all(np.asarray(core.time_constants(params)) > 0.001)


@pytest.mark.parametrize("x64", [False, True])
def test_native_precision_jit_jvp_reverse_and_no_global_flag_change(x64):
    with jax.enable_x64(x64):
        model = constant_model(commands=1)
        params = {k: v.copy() for k, v in model.params.items()}
        params["linear"][9, 0] = 2.0
        model = replace(model, params=params)
        x, up, uf = inputs(model)
        uf[:] = 0.4
        eager = model.rollout(x, up, uf)
        compiled = jax.jit(model.rollout)(x, up, uf)
        assert eager.dtype == (jnp.float64 if x64 else jnp.float32)
        np.testing.assert_allclose(eager, compiled, atol=2e-7)

        def fn(command):
            return model.rollout(x, up, command)[-1, 0]

        _, derivative = jax.jvp(
            fn, (jnp.asarray(uf),), (jnp.ones_like(jnp.asarray(uf)),)
        )
        np.testing.assert_allclose(derivative, 0.4, atol=1e-7)
        gradient = jax.jit(jax.grad(fn))(uf)
        np.testing.assert_allclose(gradient, 0.1, atol=1e-7)
        assert bool(jax.config.x64_enabled) is x64


def test_future_causality_and_one_memory_update_per_sample():
    with jax.enable_x64(True):
        model = constant_model(commands=1)
        params = {k: v.copy() for k, v in model.params.items()}
        params["linear"][9, 0] = 1
        params["linear"][-2, 0] = 1
        params["memory"][-2:, :] = np.eye(2)
        params["memory_bias"][:] = 0.5
        model = replace(model, params=params)
        x, up, uf = inputs(model)
        changed = uf.copy()
        changed[2:] = 10
        np.testing.assert_array_equal(
            model.rollout(x, up, uf)[:2], model.rollout(x, up, changed)[:2]
        )
        p, n = jax.tree.map(jnp.asarray, (model.params, model.norms))
        a, history, hidden = core._history(
            p, n, jnp.asarray(x[None]), jnp.asarray(up[None]), 2, 0.05
        )
        result = core.physical_step(
            p, n, jnp.asarray(x[-1:]), jnp.zeros((1, 1)), a, history, hidden, 0.05
        )
        np.testing.assert_allclose(
            result[3], np.tanh(np.asarray(hidden) + 0.5), atol=1e-15
        )
        np.testing.assert_allclose(result[0][0, 0], 0.05 * hidden[0, 0], atol=1e-15)


def test_one_compiled_model_across_precision_contexts():
    model = constant_model(commands=1)
    arguments = inputs(model)
    compiled = jax.jit(model.rollout)
    for enabled in (False, True, False, True):
        with jax.enable_x64(enabled):
            actual = compiled(*arguments)
            assert actual.dtype == (jnp.float64 if enabled else jnp.float32)
            np.testing.assert_allclose(actual, model.rollout(*arguments), atol=2e-7)


def test_common_world_heading_rotation_equivariance():
    with jax.enable_x64(True):
        model = constant_model()
        params = {k: v.copy() for k, v in model.params.items()}
        params["bias"][:3] = [1, 2, 9.80665]
        model = replace(model, params=params)
        past, up, uf = inputs(model)
        heading = np.asarray(core.rotation_exp(np.array([0.0, 0.0, 1.2])))
        rotated = past.copy()
        rotated[..., :3] = past[..., :3] @ heading.T
        rotated[..., 6:] = np.broadcast_to(heading.reshape(9), rotated[..., 6:].shape)
        original = np.asarray(model.rollout(past, up, uf))
        actual = np.asarray(model.rollout(rotated, up, uf))
        np.testing.assert_allclose(
            actual[..., :3], original[..., :3] @ heading.T, atol=1e-14
        )
        np.testing.assert_allclose(actual[..., 3:6], original[..., 3:6], atol=1e-14)
        np.testing.assert_allclose(
            actual[..., 6:].reshape(-1, 3, 3),
            heading @ original[..., 6:].reshape(-1, 3, 3),
            atol=1e-14,
        )


def test_initialization_is_two_solves_and_training_only(monkeypatch):
    batch = analytic_batch(3)
    solve = np.linalg.solve
    calls = []
    monkeypatch.setattr(
        np.linalg, "solve", lambda a, b: (calls.append(a.shape), solve(a, b))[1]
    )
    events = []
    model = core.initialize(batch, observer=events.append)
    assert len(calls) == 2
    assert [e["phase"] for e in events] == [
        "initializer_started",
        "ridge_solve",
        "ridge_solve",
    ]
    assert [e["ordinal"] for e in events[1:]] == [1, 2]
    np.testing.assert_array_equal(model.params["w2"], 0)
    assert all(a.dtype == np.float64 for a in model.arrays().values())
    assert model.fingerprint == core.initialize(batch).fingerprint
    altered = batch.future_states.copy()
    altered[..., 6] = 2
    with pytest.raises(ValueError, match="proper"):
        core.initialize(
            SequenceBatch(
                batch.past_states,
                batch.past_inputs,
                batch.future_inputs,
                altered,
                batch.dt_s,
            )
        )


def test_fit_imported_objective_counts_selection_and_immutable_initial():
    batch = analytic_batch(1)
    events = []
    scale = np.ones((3, 15))
    weights = np.arange(1, 16, dtype=float)
    before = bool(jax.config.x64_enabled)
    model, report = core.fit_sequence(
        batch,
        batch,
        error_scale=scale,
        channel_weights=weights,
        observer=events.append,
        _steps=2,
        _check_every=1,
    )
    assert bool(jax.config.x64_enabled) is before
    phases = [e["phase"] for e in events]
    assert phases.count("initializer_started") == 1
    assert phases.count("ridge_solve") == 2
    assert phases.count("initialized") == 1
    assert phases.count("completed") == 2
    assert report["gradient"]["known_gradient_window_visits"] == 8
    assert report["initial_weight_forecasts"] == 0
    np.testing.assert_array_equal(report["error_scale"], scale)
    np.testing.assert_array_equal(report["channel_weights"], weights)
    losses = report["safeguard"]["checkpoint_full_training_losses"]
    assert all(
        b["full_training_loss"] <= a["full_training_loss"] for a, b in pairwise(losses)
    )
    checkpoints = [e for e in events if e["phase"] == "checkpoint"]
    selected = min(checkpoints, key=lambda e: e["validation_rollout_mse"])
    assert report["selected_step"] == selected["step"]
    for key, value in model.params.items():
        np.testing.assert_array_equal(value, selected["params"][key])
    for row in [e for e in events if e["phase"] == "started"]:
        np.testing.assert_array_equal(row["indices"], np.arange(4, dtype=np.int64))
    initial = next(e["model"] for e in events if e["phase"] == "initialized")
    assert initial.fingerprint == report["initial_fingerprint"]


def test_ordinary_weights_are_initial_training_only_and_float_scope_restored():
    batch = analytic_batch(1)
    events = []
    with jax.enable_x64(False):
        _, report = core.fit_sequence(batch, batch, observer=events.append, _steps=0)
        assert not jax.config.x64_enabled
    weighting = next(e for e in events if e["phase"] == "weights")
    mse = np.mean(
        (
            (weighting["initial_training_prediction"] - batch.future_states)
            / np.asarray(weighting["normalization"])
        )
        ** 2,
        (0, 1),
    )
    raw = 1 / np.maximum(mse, 0.0001)
    np.testing.assert_array_equal(weighting["channel_weights"], raw / raw.mean())
    assert report["initial_weight_forecasts"] == 1
    assert report["selected_step"] == 0


def test_equal_loss_rejects_all_scales_but_advances_moments(monkeypatch):
    # Present identical current parameters for every trial: equality rejects.
    monkeypatch.setattr(
        core, "trial_parameters", lambda current, proposal, alpha: current
    )
    batch = analytic_batch(1)
    events = []
    model, report = core.fit_sequence(
        batch,
        batch,
        error_scale=np.ones((3, 15)),
        channel_weights=np.ones(15),
        observer=events.append,
        _steps=2,
        _check_every=1,
    )
    assert report["selected_step"] == 0
    assert report["safeguard"]["rejected_attempts"] == 2
    assert report["full_training_objective_calls"] == 17
    initial = next(e["model"] for e in events if e["phase"] == "initialized")
    assert model.fingerprint == initial.fingerprint
    completed = [e for e in events if e["phase"] == "completed"]
    assert any(np.any(np.asarray(v) != 0) for v in completed[0]["first"].values())
    assert any(
        not np.array_equal(completed[0]["first"][k], completed[1]["first"][k])
        for k in completed[0]["first"]
    )


def test_archive_roundtrip_and_shape_dtype_corruption():
    model = constant_model()
    assert (
        core.VehicleSequenceModel.from_arrays(
            model.metadata(), model.arrays()
        ).fingerprint
        == model.fingerprint
    )
    assert set(model.metadata()) == {"format", "dt_s", "history_steps", "delay_steps"}
    arrays = model.arrays()
    arrays["param_bias"] = arrays["param_bias"].astype(np.float32)
    with pytest.raises(ValueError, match="float64"):
        core.VehicleSequenceModel.from_arrays(model.metadata(), arrays)
    with pytest.raises(ValueError, match="archive"):
        core.VehicleSequenceModel.from_arrays(
            {**model.metadata(), "format": "old"}, model.arrays()
        )
    with pytest.raises(ValueError, match="shapes"):
        model.rollout(*inputs(model)[:2], np.zeros((2, 4)))
    with pytest.raises(ValueError, match="positive"):
        replace(model, norms={**model.norms, "output_scale": np.zeros(6)})


def test_initialization_failure_and_timeout_preserve_prefix_and_scope(monkeypatch):
    batch = analytic_batch(1)
    events = []

    def fail(*args):
        raise np.linalg.LinAlgError("test singular")

    with monkeypatch.context() as patch:
        patch.setattr(np.linalg, "solve", fail)
        with jax.enable_x64(False), pytest.raises(np.linalg.LinAlgError):
            core.fit_sequence(batch, batch, observer=events.append, _steps=0)
    assert [e["phase"] for e in events] == ["initializer_started"]
    with monkeypatch.context() as patch:
        patch.setattr(core, "FIT_WALL_TIME_LIMIT_S", -1)
        with (
            jax.enable_x64(False),
            pytest.raises(SequenceFitError, match="fit_time_limit"),
        ):
            core.fit_sequence(batch, batch, channel_weights=np.ones(15), _steps=0)
