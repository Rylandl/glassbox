"""Check the independent optimizer used to diagnose recovery convergence."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from glassbox.control.plan import SolverPolicy

_SCRIPT = Path(__file__).parents[1] / "scripts" / "investigate_recovery.py"
_SPEC = importlib.util.spec_from_file_location("recovery_investigation", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
investigation = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(investigation)


def reference_problem():
    target = jnp.asarray([[2.0, -0.4], [0.25, -0.2]])
    curvature = jnp.asarray([[0.1, 1.0], [10.0, 1000.0]])
    objective = jax.jit(
        jax.value_and_grad(
            lambda blocks, *_: 0.5 * jnp.sum(curvature * (blocks - target) ** 2)
        )
    )
    solver = object.__new__(investigation.OfflineReferenceSolver)
    solver.policy = SolverPolicy(horizon_steps=2, block_count=2)
    solver.model = SimpleNamespace(values=None)
    solver._kernels = SimpleNamespace(objective_and_gradient=objective)
    return solver, objective, target


def test_offline_reference_solves_a_bounded_ill_conditioned_problem():
    solver, objective, target = reference_problem()
    blocks = jnp.zeros((2, 2))
    value, gradient = objective(blocks)
    result = solver._optimize_plan(
        blocks, value, gradient, None, None, SimpleNamespace(states=None), None, None
    )
    assert result.converged
    assert result.finite
    assert result.progressed
    np.testing.assert_allclose(result.blocks, np.clip(target, -1, 1), atol=2e-3)
    # The unconstrained gradient cannot vanish at the active upper bound.
    assert abs(float(result.gradient[0, 0])) > solver.policy.gradient_tolerance


def test_offline_reference_does_not_replace_a_better_seed(monkeypatch):
    solver, objective, _ = reference_problem()
    monkeypatch.setattr(
        investigation,
        "minimize",
        lambda *args, **kwargs: SimpleNamespace(
            x=np.ones(4), nit=1, message="bad step"
        ),
    )
    blocks = jnp.zeros((2, 2))
    value, gradient = objective(blocks)
    result = solver._optimize_plan(
        blocks, value, gradient, None, None, SimpleNamespace(states=None), None, None
    )
    np.testing.assert_array_equal(result.blocks, blocks)
    assert float(result.value) == float(value)
    assert not result.progressed
    assert not result.converged
