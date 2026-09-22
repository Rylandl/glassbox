"""Analytic validation of diagnostic AST prefixes and dynamic linearization tapes."""

import ast
import copy
import inspect
import textwrap

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _online_cost_kernels import _proposal_tree, generated_sources, make_kernels

from glassbox import online


def assert_tree_close(actual, expected):
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for left, right in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        np.testing.assert_array_equal(np.isfinite(left), np.isfinite(right))
        np.testing.assert_allclose(left, right, rtol=2e-8, atol=2e-10, equal_nan=True)


def without_checkpoints(statements):
    return [
        statement
        for statement in statements
        if not (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and statement.targets[0].id.startswith("_profile_")
        )
    ]


def test_generated_prefixes_preserve_native_ast_arithmetic_signature_and_solve():
    native, endpoints = _proposal_tree(inspect.getsource(online._proposal.__wrapped__))
    sources = generated_sources()
    assert set(sources) == {
        "setup",
        "solve",
        "trust",
        "instrumented",
        "linearization",
        "cached_jvp",
        "cached_vjp",
        "cached_curvature",
    }
    for stage in ("setup", "solve", "trust", "instrumented"):
        function = ast.parse(sources[stage]).body[0]
        assert ast.dump(function.args) == ast.dump(native.args)
        assert [ast.dump(item) for item in function.decorator_list] == [
            ast.dump(item) for item in native.decorator_list
        ]
        body = without_checkpoints(function.body)
        if stage == "instrumented":
            restored = copy.deepcopy(function)
            restored.name = native.name
            restored.body = body
            restored.body[-1].value = restored.body[-1].value.elts[0]
            assert ast.dump(restored) == ast.dump(native)
        else:
            assert isinstance(body[-1], ast.Return)
            assert body[-1].value.id == f"_profile_{stage}"
            assert [ast.dump(item) for item in body[:-1]] == [
                ast.dump(item) for item in native.body[: endpoints[stage] + 1]
            ]


@pytest.mark.parametrize("mutation", ["iterations", "gradient", "direction"])
def test_source_stage_extraction_fails_closed_when_contract_changes(mutation):
    source = textwrap.dedent(inspect.getsource(online._proposal.__wrapped__))
    tree = ast.parse(source)
    if mutation == "iterations":
        for item in ast.walk(tree):
            if (
                isinstance(item, ast.Call)
                and ast.unparse(item.func) == "jax.lax.fori_loop"
            ):
                item.args[1] = ast.Constant(8)
    else:
        name = "gradient" if mutation == "gradient" else "direction_finite"
        tree.body[0].body = [
            item
            for item in tree.body[0].body
            if not (
                isinstance(item, ast.Assign)
                and isinstance(item.targets[0], ast.Name)
                and item.targets[0].id == name
            )
        ]
    with pytest.raises(ValueError):
        _proposal_tree(ast.unparse(tree))


@pytest.mark.parametrize(
    "start,cutoff,conditioning,trial_count,selected",
    [
        (0.1, None, True, 3, 0.25),
        (0.1, 2.0, True, 3, 0.25),
        (0.1, None, False, 5, 0.0),
        (1.0, None, True, 5, 0.0),
    ],
)
def test_instrumented_native_first_acceptance_and_prefixes_agree(
    monkeypatch, start, cutoff, conditioning, trial_count, selected
):
    def residual(params, norms, data, scale, delay, dt_s):
        value = params["x"][0]
        error = value**2 - data[0]
        if cutoff is not None:
            error = jnp.where(value > cutoff, jnp.nan, error)
        return jnp.zeros((1, 1, 15)).at[0, 0, 0].set(error) / scale

    monkeypatch.setattr(online, "_residual", residual)
    monkeypatch.setattr(online, "_curvature_diagonal", lambda *a, **kw: jnp.zeros(1))
    kernels = make_kernels()

    def fresh_native(params, norms, data, scale, weights, damping, **kwargs):
        return online._proposal.__wrapped__(
            params, norms, data, scale, weights, damping, **kwargs
        )

    native = jax.jit(fresh_native, static_argnames=("delay", "dt_s"))
    with jax.enable_x64(True):
        args = (
            {"x": jnp.asarray([start])},
            {},
            (jnp.asarray(1.0),),
            jnp.ones((1, 15)),
            jnp.ones(1),
            jnp.asarray(1e-10),
        )
        kwargs = dict(delay=1, dt_s=0.05, conditioning_finite=jnp.asarray(conditioning))
        expected = native(*args, **kwargs)
        actual, checkpoints = kernels["instrumented"](*args, **kwargs)
        assert_tree_close(actual, expected)
        assert int(actual[-1]["trial_evaluations"]) == trial_count
        assert float(actual[-1]["selected_alpha"]) == selected
        for stage in ("setup", "solve", "trust"):
            assert_tree_close(kernels[stage](*args, **kwargs), checkpoints[stage])
        raw, push = kernels["linearization"](*args[:4], delay=1, dt_s=0.05)
        assert_tree_close(raw, checkpoints["setup"]["raw"])
        assert isinstance(push, jax.tree_util.Partial)


def test_retained_tape_arrays_remain_dynamic_and_actions_match_analytic_jacobian(
    monkeypatch,
):
    def residual(params, norms, data, scale, delay, dt_s):
        x, y = params["x"]
        a, b = norms["coefficient"]
        values = jnp.stack((a * x**2 + b * y - data[0], jnp.sin(y) + data[1] * x))
        return jnp.zeros((1, 1, 15)).at[0, 0, :2].set(values) / scale

    monkeypatch.setattr(online, "_residual", residual)
    kernels = make_kernels()
    with jax.enable_x64(True):
        p = jnp.asarray([0.3, -0.7])
        c = jnp.arange(15, dtype=jnp.float64).reshape(1, 1, 15) + 0.4
        weight = jnp.asarray([[[0.2]]])
        irls = jnp.linspace(0.4, 1.0, 15).reshape(1, 1, 15)
        preconditioner = jnp.asarray([0.2, 1.3])
        cases = [
            ([0.4, -0.3], [1.2, 0.7], [0.5, 0.2], [1.0, 2.0]),
            ([-0.2, 0.6], [0.8, -0.5], [-0.1, 0.9], [0.7, 1.3]),
        ]
        tapes, jacobians = [], []
        for theta, coefficient, data, scale in cases:
            padded_scale = jnp.ones((1, 15)).at[0, :2].set(jnp.asarray(scale))
            raw, push = kernels["linearization"](
                {"x": jnp.asarray(theta)},
                {"coefficient": jnp.asarray(coefficient)},
                jnp.asarray(data),
                padded_scale,
                delay=1,
                dt_s=0.05,
            )
            x, y = theta
            a, b = coefficient
            jacobian = np.zeros((15, 2))
            jacobian[:2] = (
                np.asarray([[2 * a * x, b], [data[1], np.cos(y)]])
                / np.asarray(scale)[:, None]
            )
            expected = np.zeros((1, 1, 15))
            expected[0, 0, :2] = (
                np.asarray([a * x**2 + b * y - data[0], np.sin(y) + data[1] * x])
                / scale
            )
            np.testing.assert_allclose(raw, expected, rtol=1e-14, atol=1e-14)
            assert isinstance(push, jax.tree_util.Partial)
            assert sum(np.asarray(leaf).nbytes for leaf in jax.tree.leaves(push)) > 0
            tapes.append(push)
            jacobians.append(jacobian)
        assert jax.tree.structure(tapes[0]) == jax.tree.structure(tapes[1])
        # Compile with the first tape, then execute exactly that executable with
        # the changed retained residual arrays. This excludes captured constants.
        jvp = kernels["cached_jvp"].lower(tapes[0], p).compile()
        vjp = kernels["cached_vjp"].lower(tapes[0], p, c).compile()
        curvature = (
            kernels["cached_curvature"]
            .lower(tapes[0], p, weight, irls, preconditioner)
            .compile()
        )
        for push, jacobian in zip(tapes, jacobians, strict=True):
            projected = jvp(push, p)
            transposed = vjp(push, p, c)
            normal = curvature(push, p, weight, irls, preconditioner)
            expected_projected = jacobian @ np.asarray(p)
            expected_transposed = jacobian.T @ np.asarray(c).ravel()
            expected_normal = jacobian.T @ (
                np.asarray(weight * irls).ravel() * expected_projected
            ) + np.asarray(preconditioner * p)
            np.testing.assert_allclose(
                projected.ravel(), expected_projected, atol=1e-14
            )
            np.testing.assert_allclose(transposed, expected_transposed, atol=1e-14)
            np.testing.assert_allclose(normal, expected_normal, atol=1e-14)
            np.testing.assert_allclose(
                np.vdot(projected, c), np.vdot(p, transposed), atol=1e-14
            )
        assert not np.allclose(jvp(tapes[0], p), jvp(tapes[1], p))
