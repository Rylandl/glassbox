import jax
import jax.numpy as jnp
import numpy as np
from diagnose_readout_stability import (
    ablate,
    carry_from,
    difference,
    dual_step,
    features,
    jacobian,
    leverage,
    perturb,
    precision_root,
    step,
)
from test_dynamics import constant_model, inputs

from glassbox import _dynamics as core


def setup_model():
    model = constant_model()
    rng = np.random.default_rng(822)
    model.params.update(
        {k: rng.normal(size=v.shape) * 0.025 for k, v in model.params.items()}
    )
    past, issued, future = inputs(model, omega=np.array([0.1, -0.05, 0.02]))
    return model, past, issued, future


def test_identical_branches_reduce_to_original_integrator():
    model, past, issued, future = setup_model()
    with jax.enable_x64(True):
        p, n = jax.tree.map(jnp.asarray, (model.params, model.norms))
        left = right = carry_from(
            p, n, jnp.asarray(past), jnp.asarray(issued), delay=2, dt_s=0.05
        )
        expected = left
        for command in future:
            left, right = dual_step(
                p, n, p, n, left, right, jnp.asarray(command), dt_s=0.05
            )
            expected = step(p, n, expected, jnp.asarray(command), dt_s=0.05)
            for a, b, c in zip(left, right, expected):
                np.testing.assert_allclose(a, c, atol=1e-12, rtol=1e-12)
                np.testing.assert_allclose(b, c, atol=1e-12, rtol=1e-12)


def test_swap_selects_physical_force_and_rate_components():
    model = constant_model()
    past, issued, future = inputs(model)
    left = dict(model.params, bias=np.array([1.0, 2.0, 3.0, 10.0, 20.0, 30.0]))
    right = dict(model.params, bias=np.array([10.0, 20.0, 30.0, 0.1, 0.2, 0.3]))
    merged = dict(model.params, bias=np.r_[left["bias"][:3], right["bias"][3:]])
    with jax.enable_x64(True):
        c = carry_from(
            merged,
            model.norms,
            jnp.asarray(past),
            jnp.asarray(issued),
            delay=2,
            dt_s=0.05,
        )
        actual, _ = dual_step(
            left,
            model.norms,
            right,
            model.norms,
            c,
            c,
            jnp.asarray(future[0]),
            dt_s=0.05,
        )
        expected = step(merged, model.norms, c, jnp.asarray(future[0]), dt_s=0.05)
        np.testing.assert_allclose(actual[0], expected[0], atol=1e-12, rtol=1e-12)


def test_rate_jacobian_matches_known_midpoint_damping_and_full_fd():
    model = constant_model()
    rates = np.array([-2.0, -3.0, -4.0])
    model.params.update({k: v.copy() for k, v in model.params.items()})
    model.params["linear"][3:6, 3:6] = np.diag(rates)
    model.params["bias"][2] = 9.81
    past, issued, future = inputs(model)
    with jax.enable_x64(True):
        p, n = jax.tree.map(jnp.asarray, (model.params, model.norms))
        c = carry_from(p, n, jnp.asarray(past), jnp.asarray(issued), delay=2, dt_s=0.05)
        J = np.asarray(jacobian(p, n, c, jnp.asarray(future[0]), dt_s=0.05))
        expected = (1 + 0.025 * rates + 0.5 * (0.025 * rates) ** 2) ** 2
        np.testing.assert_allclose(
            J[3:6, 3:6], np.diag(expected), atol=1e-12, rtol=1e-12
        )
        direction = np.random.default_rng(8).normal(size=J.shape[0])
        nominal = step(p, n, c, jnp.asarray(future[0]), dt_s=0.05)
        plus = step(
            p,
            n,
            perturb(c, jnp.asarray(direction * 1e-6)),
            jnp.asarray(future[0]),
            dt_s=0.05,
        )
        minus = step(
            p,
            n,
            perturb(c, jnp.asarray(-direction * 1e-6)),
            jnp.asarray(future[0]),
            dt_s=0.05,
        )
        actual = np.asarray(
            (difference(plus, nominal) - difference(minus, nominal)) / 2e-6
        )
        np.testing.assert_allclose(J @ direction, actual, atol=1e-9, rtol=1e-9)


def test_leverage_qr_matches_independent_precision_solve():
    rng = np.random.default_rng(17)
    phi = rng.normal(size=(20, 14)) * np.geomspace(0.1, 10, 14)
    queries = rng.normal(size=(5, 14))
    for penalty in (None, np.geomspace(0.01, 10, 14)):
        root = precision_root(phi, 0.25, penalty)
        gram = 0.25 * np.eye(14) + phi.T @ phi
        if penalty is not None:
            gram += np.diag(penalty)
        expected = np.sum(queries * np.linalg.solve(gram, queries.T).T, axis=1)
        np.testing.assert_allclose(
            leverage(root, queries), expected, atol=1e-12, rtol=1e-12
        )


def test_ablation_only_removes_selected_rate_readout():
    model, _, _, _ = setup_model()
    for variant, key in [
        ("no_quadratic_rate", "quadratic"),
        ("no_nonlinear_rate", "w2"),
        ("no_lag_rate", "linear"),
    ]:
        result = ablate(model, variant)
        for name, value in model.params.items():
            if name != key:
                np.testing.assert_array_equal(result.params[name], value)
        np.testing.assert_array_equal(
            result.params[key][:, :3], model.params[key][:, :3]
        )
        if key == "linear":
            np.testing.assert_array_equal(
                result.params[key][:15], model.params[key][:15]
            )
            np.testing.assert_array_equal(
                result.params[key][-2:], model.params[key][-2:]
            )
            assert not np.any(result.params[key][15:45, 3:])
        else:
            assert not np.any(result.params[key][:, 3:])


def test_diagnostic_feature_vector_reconstructs_readout():
    model, past, issued, future = setup_model()
    with jax.enable_x64(True):
        p, n = jax.tree.map(jnp.asarray, (model.params, model.norms))
        c = carry_from(p, n, jnp.asarray(past), jnp.asarray(issued), delay=2, dt_s=0.05)
        phi = features(p, n, c, jnp.asarray(future[0]))
        mean = jnp.concatenate((p["linear"], p["quadratic"], p["bias"][None], p["w2"]))
        physical = core._head(p, n, c[0], jnp.asarray(future[:1]), *c[1:])[0][0]
        np.testing.assert_allclose(
            phi @ mean * n["output_scale"], physical, atol=1e-12, rtol=1e-12
        )
