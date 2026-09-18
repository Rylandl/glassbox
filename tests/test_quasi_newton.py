"""Optimizer mechanism, work limits and independent first-order qualification."""

from dataclasses import replace
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
import test_task_qualification as task_fixture

from glassbox.control.plan import ReferenceTrajectory, SolverPolicy, SolveStatus
from glassbox.control.solver import BoundedShootingSolver, _projected_gradient_norm
from glassbox.experimental import quasi_newton as q

POLICY = SolverPolicy(horizon_steps=1, block_count=1, maximum_iterations=64)
learned = task_fixture.learned


def quadratic(target=(0.7, -0.4), weights=(1.0, 0.05)):
    target, weights = np.array([target], np.float32), np.array([weights], np.float32)
    calls = []

    def objective(blocks):
        x = np.asarray(blocks)
        assert x.dtype == np.float32
        assert np.all(np.abs(x) <= 1)
        calls.append(x.copy())
        error = x - target
        return np.sum(weights * error**2), 2 * weights * error

    return objective, calls


def solve(objective, policy=POLICY):
    seed = jnp.zeros((1, 2), dtype=jnp.float32)
    value, gradient = objective(seed)
    return q.bounded_minimize(objective, seed, value, gradient, policy)


@pytest.mark.parametrize("target", [(0.7, -0.4), (2.0, -2.0)])
def test_quadratic_reaches_interior_or_active_bound_optimum(target):
    objective, calls = quadratic(target)
    outcome, work = solve(objective)
    assert outcome.converged
    assert float(_projected_gradient_norm(outcome.blocks, outcome.gradient)) <= 0.002
    np.testing.assert_allclose(outcome.blocks, np.clip([target], -1, 1), atol=0.02)
    assert outcome.finite
    assert work["accepted_iterations"] <= 64
    assert work["new_objective_evaluations"] <= 1024
    assert work["audit_objective_evaluations"] == 1
    assert len(calls) == 1 + work["new_objective_evaluations"] + 1


def fake_result(x, *, status=0, success=True, message="relative improvement", nit=1):
    return SimpleNamespace(
        x=np.asarray(x).ravel(),
        fun=-999.0,
        jac=np.zeros(np.asarray(x).size),
        status=status,
        success=success,
        message=message,
        nit=nit,
    )


def test_backend_success_and_forged_scores_do_not_determine_convergence(monkeypatch):
    def backend(fun, x0, **kwargs):
        fun(x0)
        x = np.array([0.1, -0.1])
        fun(x)
        kwargs["callback"](x)
        return fake_result(x)

    monkeypatch.setattr(q, "minimize", backend)
    objective, _ = quadratic()
    outcome, work = solve(objective)
    value, gradient = objective(outcome.blocks)
    assert float(outcome.value) == float(value)
    np.testing.assert_array_equal(outcome.gradient, gradient)
    assert not outcome.converged
    assert work["backend_success"] is True
    assert work["accepted_iterations"] == 1
    assert work["cached_seed_requests"] == 1
    assert outcome.stalled


def test_hard_budget_retains_best_canonical_evaluated_point(monkeypatch):
    monkeypatch.setattr(q, "MAXIMUM_EVALUATIONS", 2)

    def backend(fun, x0, **kwargs):
        fun(x0)
        fun(np.array([0.1, -0.1]))
        fun(np.array([0.2, -0.2]))
        fun(np.array([0.3, -0.3]))
        pytest.fail("evaluation budget must stop the backend")

    monkeypatch.setattr(q, "minimize", backend)
    objective, calls = quadratic()
    outcome, work = solve(objective)
    assert work["new_objective_evaluations"] == 2
    assert work["accepted_iterations"] == 0
    assert work["audit_objective_evaluations"] == 1
    np.testing.assert_array_equal(outcome.blocks, np.array([[0.2, -0.2]], np.float32))
    assert len(calls) == 4  # Seed, two new points, independent final audit.
    assert not outcome.converged


def test_nonfinite_evaluation_remains_explicit_while_finite_seed_is_retained(
    monkeypatch,
):
    def backend(fun, x0, **kwargs):
        fun(np.array([0.5, 0.5]))
        pytest.fail("nonfinite callback must stop the backend")

    def objective(x):
        if np.any(np.asarray(x) != 0):
            return np.float32(np.inf), np.full((1, 2), np.nan, np.float32)
        return np.float32(1), np.ones((1, 2), np.float32)

    monkeypatch.setattr(q, "minimize", backend)
    outcome, work = solve(objective)
    assert work["nonfinite_evaluation"]
    assert work["new_objective_evaluations"] == 1
    np.testing.assert_array_equal(outcome.blocks, np.zeros((1, 2)))
    assert not outcome.converged


def test_backend_abnormal_stop_is_not_silently_successful(monkeypatch):
    monkeypatch.setattr(
        q,
        "minimize",
        lambda fun, x0, **kw: fake_result(
            x0,
            status=2,
            success=False,
            message="ABNORMAL",
            nit=0,
        ),
    )
    outcome, work = solve(quadratic()[0])
    assert work["backend_failure"]
    assert not work["backend_success"]
    assert not outcome.converged


def test_nonfinite_backend_proposal_is_an_explicit_failed_origin(monkeypatch):
    def backend(fun, x0, **kwargs):
        fun(np.array([np.nan, 0.0]))
        pytest.fail("nonfinite proposal must stop the backend")

    monkeypatch.setattr(q, "minimize", backend)
    outcome, work = solve(quadratic()[0])
    assert work["nonfinite_evaluation"]
    assert work["backend_failure"]
    assert work["new_objective_evaluations"] == 0
    np.testing.assert_array_equal(outcome.blocks, np.zeros((1, 2)))
    assert not outcome.converged


def test_unexpected_backend_error_is_not_hidden_as_a_result(monkeypatch):
    def broken(*_, **__):
        raise RuntimeError("programming error")

    monkeypatch.setattr(q, "minimize", broken)
    with pytest.raises(RuntimeError, match="programming error"):
        solve(quadratic()[0])


def test_independent_audit_rejects_objective_drift(monkeypatch):
    monkeypatch.setattr(q, "minimize", lambda fun, x0, **kw: fake_result(x0, nit=0))
    calls = 0

    def changed_objective(x):
        nonlocal calls
        calls += 1
        return np.float32(calls), np.ones((1, 2), np.float32)

    with pytest.raises(AssertionError):
        solve(changed_objective)


def test_frozen_algorithm_options(monkeypatch):
    def backend(fun, x0, **kwargs):
        assert kwargs["method"] == "L-BFGS-B"
        assert kwargs["jac"] is True
        assert kwargs["options"] == dict(
            maxiter=64, maxls=16, maxcor=10, ftol=1e-5, gtol=0.002, maxfun=1024
        )
        np.testing.assert_array_equal(kwargs["bounds"], [(-1, 1), (-1, 1)])
        return fake_result(x0, nit=0)

    monkeypatch.setattr(q, "minimize", backend)
    solve(quadratic()[0])


def test_full_solver_keeps_original_seed_and_shared_kernels_untouched(learned):
    arm = task_fixture.make_arm(learned, task_fixture.q.CANDIDATE)
    arm.reset(task_fixture.INITIAL, task_fixture.COMMAND)
    model = arm.plan.with_causal_state(arm._state)
    policy = replace(arm.policy, maximum_iterations=64)
    reference = ReferenceTrajectory.hold(task_fixture.INITIAL, 5)
    baseline = BoundedShootingSolver(model, policy)
    kernels = baseline._kernels
    candidate = q.QuasiNewtonSolver(model, policy)
    original = baseline.solve(task_fixture.INITIAL, reference, task_fixture.COMMAND)
    for warm in (None, original.warm_start):
        a = baseline.solve(
            task_fixture.INITIAL, reference, task_fixture.COMMAND, warm_start=warm
        )
        b = candidate.solve(
            task_fixture.INITIAL, reference, task_fixture.COMMAND, warm_start=warm
        )
        assert a.diagnostics.initial_objective == b.diagnostics.initial_objective
        assert a.diagnostics.warm_start_used == b.diagnostics.warm_start_used
        assert candidate.last_work["seed_objective_evaluations"] == (
            1 if warm is None else 2
        )
        assert candidate.last_work["audit_objective_evaluations"] == 1
        assert b.diagnostics.maximum_command_bound_violation == 0
        assert not b.used_fallback
        assert (b.status == SolveStatus.CONVERGED) == (
            b.diagnostics.final_projected_gradient_inf_norm <= policy.gradient_tolerance
        )
        assert baseline._kernels is kernels
        assert candidate._kernels is kernels
