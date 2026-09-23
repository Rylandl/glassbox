"""Analytic vehicles and bounded fitting only; no simulator fixtures."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from glassbox import _dynamics as core


def constant_model(commands=3, *, dt=0.05, history=3, delay=2):
    b = 9 + 2 * commands
    f = (delay + 1) * b + 2
    nonlinear = (min(4, delay) + 1) * b + 2
    q = b * (b + 1) // 2
    params = dict(
        linear=np.zeros((f, 3)),
        quadratic=np.zeros((q, 3)),
        bias=np.zeros(3),
        w1=np.zeros((nonlinear, 3)),
        b1=np.zeros(3),
        w2=np.zeros((3, 3)),
        memory=np.zeros((b, 2)),
        memory_bias=np.zeros(2),
        raw_tau=np.full(commands, np.log(np.expm1(0.049))),
        raw_memory_tau=np.full(2, np.log(np.expm1(0.049))),
        rate=np.zeros((3, 2 * commands + 3)),
    )
    norms = dict(
        body_mean=np.zeros(9),
        body_scale=np.ones(9),
        motion_bound_scale=np.full(6, 4.0),
        input_mean=np.zeros(commands),
        input_scale=np.ones(commands),
        feature_scale=np.ones(f),
        nonlinear_scale=np.ones(nonlinear),
        quadratic_scale=np.ones(q),
        output_scale=np.ones(3),
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


@pytest.mark.parametrize("x64", [False, True])
@pytest.mark.parametrize("commands,delay", [(1, 2), (4, 10)])
def test_centered_projection_preserves_head_and_all_input_derivatives(
    x64, commands, delay
):
    with jax.enable_x64(x64):
        rng = np.random.default_rng(73)
        model = constant_model(commands, history=delay + 1, delay=delay)
        params = {k: rng.normal(size=v.shape) * 0.05 for k, v in model.params.items()}
        norms = {k: v.copy() for k, v in model.norms.items()}
        width = 9 + 2 * commands
        norms["feature_scale"][width : (delay + 1) * width] = 0.001
        norms["nonlinear_scale"][width : (min(4, delay) + 1) * width] = 0.001
        norms["quadratic_scale"][:] = 1000
        anchor = rng.uniform(-32, 32, size=(2, width))
        current = anchor + rng.normal(size=anchor.shape) * 0.003
        history = anchor[:, None] + rng.normal(size=(2, delay, width)) * 0.001
        hidden = rng.normal(size=(2, 2))
        value = jax.tree.map(jnp.asarray, (params, current, anchor, history, hidden))
        tangent = jax.tree.map(
            lambda x: jnp.asarray(rng.normal(size=x.shape) * 0.01, dtype=x.dtype), value
        )

        def direct(value):
            p, b, _, past, h = value
            state = core.state_without_current_command(b)
            z = core.sampled_features(state, past, h) / norms["feature_scale"]
            weights = p["w1"] / norms["nonlinear_scale"][:, None]
            end = (min(4, delay) + 1) * width
            expanded = jnp.concatenate(
                (
                    weights[:width],
                    jnp.einsum(
                        "dr,rch->dch",
                        jnp.asarray(core.temporal_basis(delay)),
                        weights[width:end].reshape(min(4, delay), width, -1),
                    ).reshape(delay * width, -1),
                    weights[end:],
                )
            )
            linear = z @ p["linear"]
            nonlinear = core.sampled_features(state, past, h) @ expanded
            control = (
                b[..., 9:] / norms["feature_scale"][9:width]
            ) @ p["linear"][9:width]
            acceleration = (
                linear
                + control
                + (core.additive_quadratic_features(b) / norms["quadratic_scale"])
                @ p["quadratic"]
                + p["bias"]
                + jnp.tanh(nonlinear + p["b1"]) @ p["w2"]
            ) * norms["output_scale"]
            return jnp.concatenate((acceleration, linear, nonlinear), axis=-1)

        def reused(value):
            p, b, start, past, h = value
            anchor = core.state_without_current_command(start)
            base, effective = core._prepare_head(p, norms, anchor, past, h)
            projection = base + (core.state_without_current_command(b) - anchor) @ effective
            acceleration = core._acceleration(p, norms, b, projection)
            return jnp.concatenate((acceleration, projection), axis=-1)

        direct, reused = jax.jit(direct), jax.jit(reused)
        expected, expected_jvp = jax.jvp(direct, (value,), (tangent,))
        actual, actual_jvp = jax.jvp(reused, (value,), (tangent,))
        cotangent = jnp.asarray(rng.normal(size=actual.shape), dtype=actual.dtype)
        cotangent /= jnp.linalg.norm(cotangent)
        expected_vjp = jax.vjp(direct, value)[1](cotangent)
        actual_vjp = jax.vjp(reused, value)[1](cotangent)
        tolerance = 1e-10 if x64 else 2e-3
        for a, b in zip(
            jax.tree.leaves((actual, actual_jvp, actual_vjp)),
            jax.tree.leaves((expected, expected_jvp, expected_vjp)),
            strict=True,
        ):
            np.testing.assert_allclose(a, b, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("delay", [1, 2, 4, 5, 10, 20])
def test_temporal_basis_is_orthonormal_and_preserves_declared_polynomial_span(delay):
    basis = core.temporal_basis(delay)
    np.testing.assert_allclose(basis.T @ basis, np.eye(min(delay, 4)), atol=1e-14)
    if delay <= 4:
        np.testing.assert_array_equal(basis, np.eye(delay))
    else:
        t = np.linspace(-1, 1, delay)
        for degree in range(4):
            np.testing.assert_allclose(
                basis @ (basis.T @ t**degree), t**degree, atol=1e-14
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
        params["rate"][2, 0] = 2.0
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
        rate_applied, rate_memory = core.rate_history(
            jnp.asarray(x[None]), jnp.asarray(up[None]), 0.05
        )
        result = core.physical_step(
            p, n, jnp.asarray(x[-1:]), jnp.zeros((1, 1)), a, history, hidden,
            rate_applied, rate_memory, 0.05,
        )
        np.testing.assert_allclose(
            result[3],
            np.exp(-1) * np.asarray(hidden) + (1 - np.exp(-1)) * np.tanh(0.5),
            atol=1e-15,
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
        replace(model, norms={**model.norms, "output_scale": np.zeros(3)})


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_supported_coordinates_are_identity_inside_and_smooth_bounded_outside(dtype):
    with jax.enable_x64(dtype is np.float64):
        bound = jnp.asarray(8.0, dtype=dtype)
        values = jnp.asarray([-2.0, -0.1, 0.0, 0.1, 2.0], dtype=dtype)
        np.testing.assert_array_equal(core.supported_motion(values, bound), values)
        far = jnp.asarray([-1e6, -10, 10, 1e6], dtype=dtype)
        result = np.asarray(core.supported_motion(far, bound))
        assert np.all(np.abs(result) <= bound)
        assert np.all(np.diff(result) > 0)

        def function(z):
            return core.supported_motion(z, bound)

        for point in [-2.0, 0.0, 2.0]:
            point = jnp.asarray(point, dtype=dtype)
            np.testing.assert_allclose(jax.grad(function)(point), 1, atol=1e-7)
            np.testing.assert_allclose(
                jax.grad(jax.grad(function))(point), 0, atol=1e-7
            )
        assert np.isfinite(jax.jit(jax.jacfwd(function))(far)).all()


def test_batched_rollout_matches_individual_and_rejects_missing_context():
    model = constant_model(commands=2)
    x, up, uf = inputs(model)
    xs = np.stack((x, x))
    ups = np.stack((up, up))
    ufs = np.stack((uf, uf))
    predicted = np.asarray(model.rollout(xs, ups, ufs))
    np.testing.assert_array_equal(predicted[0], model.rollout(x, up, uf))
    np.testing.assert_array_equal(predicted[1], predicted[0])
    with pytest.raises(ValueError, match="shapes/history"):
        model.rollout(x[1:], up[1:], uf)


def test_core_arrays_are_defensive_readonly_copies():
    model = constant_model()
    fingerprint = model.fingerprint
    exported = model.arrays()
    exported["param_bias"][0] += 1
    assert model.fingerprint == fingerprint
    with pytest.raises(ValueError, match="read-only"):
        model.params["bias"][0] = 1
    arrays = model.arrays()
    del arrays["norm_motion_bound_scale"]
    with pytest.raises(ValueError, match="archive"):
        core.VehicleSequenceModel.from_arrays(model.metadata(), arrays)
