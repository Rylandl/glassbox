"""Diagnostic prefixes of the unchanged online proposal, generated from its AST.

These kernels expose otherwise internal arrays for a read-only cost profile.
Additional outputs can change compiler fusion and materialization; their timings
are diagnostic and are not additive production timing shares.
"""

from __future__ import annotations

import ast
import copy
import inspect
import textwrap
from functools import partial

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from glassbox import online

_CHECKPOINTS = {
    "setup": (
        "flat",
        "prior",
        "preconditioner",
        "raw",
        "weight",
        "root_weight",
        "irls",
        "gradient",
    ),
    "solve": ("delta", "finite"),
    "trust": (
        "delta",
        "linearized",
        "radius",
        "shrink",
        "linear_decrease",
        "quadratic_cost",
        "current_loss",
        "direction_finite",
    ),
}


def _assignment_name(statement):
    if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
        target = statement.targets[0]
        if isinstance(target, ast.Name):
            return target.id
    return None


def _proposal_tree(source):
    """Locate strict stage boundaries, failing closed when the source changes."""
    module = ast.parse(textwrap.dedent(source))
    if len(module.body) != 1 or not isinstance(module.body[0], ast.FunctionDef):
        raise ValueError("expected one online proposal function")
    function = module.body[0]
    if function.name != "_proposal":
        raise ValueError("expected the online proposal source")
    endpoints = {}
    for index, statement in enumerate(function.body):
        name = _assignment_name(statement)
        if name in {"gradient", "direction_finite"}:
            stage = "setup" if name == "gradient" else "trust"
            if stage in endpoints:
                raise ValueError("ambiguous proposal stage endpoint")
            endpoints[stage] = index
        if isinstance(statement, ast.Assign) and isinstance(statement.value, ast.Call):
            call = statement.value
            if ast.unparse(call.func) == "jax.lax.fori_loop":
                target = statement.targets
                expected = ast.parse("delta, _, _, _, finite = None").body[0].targets
                if (
                    len(call.args) != 4
                    or ast.dump(call.args[0]) != ast.dump(ast.Constant(0))
                    or ast.dump(call.args[1]) != ast.dump(ast.Constant(16))
                    or [ast.dump(item) for item in target]
                    != [ast.dump(item) for item in expected]
                    or "solve" in endpoints
                ):
                    raise ValueError(
                        "proposal no longer has the declared 16-step solve"
                    )
                endpoints["solve"] = index
    if (
        set(endpoints) != set(_CHECKPOINTS)
        or not endpoints["setup"] < endpoints["solve"] < endpoints["trust"]
        or not isinstance(function.body[-1], ast.Return)
        or not isinstance(function.body[-1].value, ast.Tuple)
        or len(function.body[-1].value.elts) != 6
    ):
        raise ValueError("proposal stage/return structure differs")
    return function, endpoints


def _snapshot(stage):
    return ast.Assign(
        targets=[ast.Name(id=f"_profile_{stage}", ctx=ast.Store())],
        value=ast.Dict(
            keys=[ast.Constant(name) for name in _CHECKPOINTS[stage]],
            values=[ast.Name(id=name, ctx=ast.Load()) for name in _CHECKPOINTS[stage]],
        ),
    )


def _prefix_sources():
    function, endpoints = _proposal_tree(
        inspect.getsource(online._proposal.__wrapped__)
    )
    boundary = {index: stage for stage, index in endpoints.items()}
    sources = {}
    for scope in (*_CHECKPOINTS, "instrumented"):
        generated = copy.deepcopy(function)
        generated.name = f"_profile_{scope}_kernel"
        generated.body = []
        for index, original in enumerate(function.body):
            statement = copy.deepcopy(original)
            if isinstance(statement, ast.Return):
                statement.value = ast.Tuple(
                    elts=[
                        statement.value,
                        ast.Dict(
                            keys=[ast.Constant(stage) for stage in _CHECKPOINTS],
                            values=[
                                ast.Name(id=f"_profile_{stage}", ctx=ast.Load())
                                for stage in _CHECKPOINTS
                            ],
                        ),
                    ],
                    ctx=ast.Load(),
                )
            generated.body.append(statement)
            if index in boundary:
                stage = boundary[index]
                generated.body.append(_snapshot(stage))
                if scope == stage:
                    generated.body.append(
                        ast.Return(
                            value=ast.Name(id=f"_profile_{stage}", ctx=ast.Load())
                        )
                    )
                    break
        module = ast.fix_missing_locations(
            ast.Module(body=[generated], type_ignores=[])
        )
        sources[scope] = ast.unparse(module) + "\n"
    return sources


@partial(jax.jit, static_argnames=("delay", "dt_s"))
def _linearization(params, norms, data, scale, *, delay, dt_s):
    flat, unpack = ravel_pytree(params)
    return jax.linearize(
        lambda value: online._residual(unpack(value), norms, data, scale, delay, dt_s),
        flat,
    )


@jax.jit
def _cached_jvp(push, p):
    return push(p)


@jax.jit
def _cached_vjp(push, theta_shape_template, c):
    return jax.linear_transpose(push, jnp.zeros_like(theta_shape_template))(c)[0]


@jax.jit
def _cached_curvature(push, p, weight, irls, preconditioner):
    projected = push(p)
    transposed = jax.linear_transpose(push, jnp.zeros_like(p))(
        weight * irls * projected
    )[0]
    return transposed + preconditioner * p


_HELPERS = {
    "linearization": _linearization,
    "cached_jvp": _cached_jvp,
    "cached_vjp": _cached_vjp,
    "cached_curvature": _cached_curvature,
}


def generated_sources():
    """Return the complete generated/diagnostic kernel source for the archive."""
    return {
        **_prefix_sources(),
        **{
            name: textwrap.dedent(inspect.getsource(function.__wrapped__))
            for name, function in _HELPERS.items()
        },
    }


def make_kernels():
    """Compile fresh diagnostic functions, with every model/tape array dynamic.

    Only source is inspected here. No model execution or saved-array loading takes
    place until callers invoke one of the returned functions. A fresh namespace
    keeps the arithmetic bound to the same production helpers as the native call.
    """
    namespace = dict(vars(online), online=online)
    kernels = {}
    for scope, source in generated_sources().items():
        exec(compile(source, f"<online-cost-profile:{scope}>", "exec"), namespace)
        name = f"_profile_{scope}_kernel" if scope not in _HELPERS else f"_{scope}"
        kernels[scope] = namespace[name]
    return kernels
