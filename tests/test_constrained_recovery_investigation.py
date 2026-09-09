"""The constrained reference must preserve feasibility and report the right residual."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from glassbox.control.plan import SolverPolicy
from glassbox.control.solver import _SolveAbort


@pytest.fixture
def investigation(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    return importlib.import_module("investigate_constrained_recovery")


def problem(investigation):
    objective = jax.jit(jax.value_and_grad(lambda blocks, *_: jnp.sum(blocks**2)))
    constraints = jax.jit(lambda blocks, *_: (blocks - 0.5).ravel())
    solver = object.__new__(investigation.SupportConstrainedReference)
    solver.policy = SolverPolicy(horizon_steps=1, block_count=1)
    solver.model = SimpleNamespace(values=None)
    solver._kernels = SimpleNamespace(objective_and_gradient=objective)
    solver.constraints = constraints
    solver.constraint_jacobian = jax.jit(jax.jacfwd(constraints))
    solver.maximum_iterations = 100
    solver.reports = []
    return solver, objective


def optimize(solver, objective, seed):
    blocks = jnp.asarray([[seed]])
    value, gradient = objective(blocks)
    return solver._optimize_plan(
        blocks, value, gradient, None, None, SimpleNamespace(states=None), None, None
    )


def test_feasible_constrained_solution_can_cost_more_than_the_infeasible_seed(
    investigation,
):
    solver, objective = problem(investigation)
    outcome = optimize(solver, objective, 0.0)
    np.testing.assert_allclose(outcome.blocks, [[0.5]], atol=1e-6)
    assert float(outcome.value) > 0.0
    assert not outcome.progressed
    assert not outcome.converged
    report = solver.reports[-1]
    assert report["feasible"]
    assert report["selected"] == "optimizer"
    assert report["maximum_constraint_violation"] <= 1e-6
    assert report["projected_lagrangian_gradient_inf_norm"] < 1e-6
    assert report["maximum_complementarity_residual"] < 1e-6
    assert report["maximum_dual_violation"] == 0
    # The constrained optimum has nonzero objective gradient. A box-only
    # residual would incorrectly call it a failure to optimize.
    assert float(outcome.gradient[0, 0]) > 0.9


@pytest.mark.parametrize("candidate", (0.0, 0.8, 1.2, float("nan")))
def test_bad_optimizer_output_does_not_displace_a_better_feasible_seed(
    investigation, monkeypatch, candidate
):
    solver, objective = problem(investigation)
    monkeypatch.setattr(
        investigation,
        "minimize",
        lambda *args, **kwargs: SimpleNamespace(
            x=np.asarray([candidate]),
            nit=1,
            success=False,
            status=9,
            message="test termination",
            multipliers=np.asarray([0.0]),
        ),
    )
    outcome = optimize(solver, objective, 0.6)
    np.testing.assert_allclose(outcome.blocks, [[0.6]])
    assert solver.reports[-1]["selected"] == "seed"
    assert solver.reports[-1]["projected_lagrangian_gradient_inf_norm"] is None


def test_no_feasible_iterate_is_an_explicit_failure(investigation, monkeypatch):
    solver, objective = problem(investigation)
    monkeypatch.setattr(
        investigation,
        "minimize",
        lambda *args, **kwargs: SimpleNamespace(
            x=np.asarray([0.1]),
            nit=1,
            success=False,
            status=9,
            message="iteration limit",
            multipliers=None,
        ),
    )
    with pytest.raises(_SolveAbort, match="no feasible command plan"):
        optimize(solver, objective, 0.0)
    assert not solver.reports[-1]["feasible"]
    assert solver.reports[-1]["selected"] is None
