"""One frozen L-BFGS-B probe using the maintained solver's exact objective.

This is an offline diagnostic backend, not a consumer option or a real-time
controller. Curvature and every evaluated point belong to one solve only.
"""

from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from glassbox.control.solver import (
    BoundedShootingSolver,
    _OptimizerOutcome,
    _projected_gradient_norm,
)

MAXIMUM_EVALUATIONS = 1024
MAXIMUM_ITERATIONS = 64
LINE_SEARCH_STEPS = 16
CORRECTION_PAIRS = 10
SCORE_TOLERANCE = dict(rtol=1e-7, atol=1e-9)


class _EvaluationLimit(Exception):
    pass


class _NonfiniteEvaluation(Exception):
    pass


def _empty_work():
    return dict(
        backend_status=None,
        backend_success=False,
        backend_message="inherited solver did not reach optimization",
        stop_cause="inherited_failure",
        accepted_iterations=0,
        new_objective_evaluations=0,
        seed_objective_evaluations=0,
        cached_seed_requests=0,
        audit_objective_evaluations=0,
        nonfinite_evaluation=False,
        backend_failure=True,
        independent_projected_gradient_inf_norm=None,
        independent_objective=None,
        returned_canonical_blocks=None,
        independent_gradient=None,
        returned_seed=False,
    )


def bounded_minimize(
    objective_gradient, seed_blocks, seed_value, seed_gradient, policy
):
    """Return an audited best finite point and exact optimizer work counters.

    The objective receives the seed's original shape and dtype. Only the seed's
    already computed value/gradient may be reused; every other request executes
    the objective and consumes the hard budget, even if it repeats a point.
    """
    seed = np.asarray(seed_blocks)
    initial_value = float(np.asarray(seed_value))
    initial_gradient = np.asarray(seed_gradient)
    if (
        seed.dtype.kind != "f"
        or not seed.size
        or initial_gradient.shape != seed.shape
        or not np.isfinite(seed).all()
        or not np.isfinite(initial_value)
        or not np.isfinite(initial_gradient).all()
        or np.any(np.abs(seed) > 1)
    ):
        raise ValueError("L-BFGS-B requires a finite bounded seed and gradient")
    if (
        policy.maximum_iterations != MAXIMUM_ITERATIONS
        or policy.line_search_steps != LINE_SEARCH_STEPS
        or policy.relative_improvement_tolerance != 1e-5
        or policy.gradient_tolerance != 0.002
    ):
        raise ValueError("L-BFGS-B policy differs from the frozen candidate")
    best = [seed.copy(), initial_value, initial_gradient.copy()]
    work = _empty_work()
    work.update(backend_message="", stop_cause="", backend_failure=False)

    def evaluate(proposal):
        canonical = np.asarray(proposal, dtype=seed.dtype).reshape(seed.shape)
        if not np.isfinite(canonical).all():
            work["nonfinite_evaluation"] = True
            raise _NonfiniteEvaluation
        if np.any(np.abs(canonical) > 1):
            raise ValueError("L-BFGS-B proposed out-of-bounds blocks")
        if np.array_equal(canonical, seed):
            work["cached_seed_requests"] += 1
            return initial_value, initial_gradient.astype(np.float64).ravel()
        if work["new_objective_evaluations"] >= MAXIMUM_EVALUATIONS:
            raise _EvaluationLimit
        work["new_objective_evaluations"] += 1
        value, gradient = objective_gradient(canonical)
        value, gradient = float(np.asarray(value)), np.asarray(gradient)
        if gradient.shape != seed.shape:
            raise ValueError("L-BFGS-B objective gradient shape differs")
        if not np.isfinite(value) or not np.isfinite(gradient).all():
            work["nonfinite_evaluation"] = True
            raise _NonfiniteEvaluation
        if value < best[1]:
            best[:] = canonical.copy(), value, gradient.copy()
        return value, gradient.astype(np.float64).ravel()

    def accepted(_blocks):
        work["accepted_iterations"] += 1
        if work["accepted_iterations"] > MAXIMUM_ITERATIONS:
            raise RuntimeError("L-BFGS-B exceeded its accepted iteration bound")

    try:
        result = minimize(
            evaluate,
            seed.astype(np.float64).ravel(),
            method="L-BFGS-B",
            jac=True,
            bounds=[(-1.0, 1.0)] * seed.size,
            callback=accepted,
            options=dict(
                maxiter=MAXIMUM_ITERATIONS,
                maxls=LINE_SEARCH_STEPS,
                maxcor=CORRECTION_PAIRS,
                ftol=policy.relative_improvement_tolerance,
                gtol=policy.gradient_tolerance,
                maxfun=MAXIMUM_EVALUATIONS,
            ),
        )
        work.update(
            backend_status=int(result.status),
            backend_success=bool(result.success),
            backend_message=str(result.message),
        )
        if int(result.nit) != work["accepted_iterations"]:
            raise ValueError("L-BFGS-B accepted iteration accounting differs")
        if result.status == 0:
            work["stop_cause"] = "relative_improvement"
        elif result.status == 1:
            work["stop_cause"] = (
                "iteration_limit"
                if work["accepted_iterations"] >= MAXIMUM_ITERATIONS
                else "evaluation_limit"
            )
        elif result.status == 2:
            work.update(stop_cause="backend_failure", backend_failure=True)
        else:
            raise ValueError("L-BFGS-B returned an unknown backend status")
    except _EvaluationLimit:
        work.update(
            stop_cause="evaluation_limit",
            backend_message="hard objective/gradient evaluation budget exhausted",
        )
    except _NonfiniteEvaluation:
        work.update(
            stop_cause="nonfinite_evaluation",
            backend_failure=True,
            backend_message="nonfinite objective or gradient during search",
        )

    # The returned point may be a line-search trial that the backend did not
    # accept. Its exact canonical blocks, rather than result.x/fun/jac, own the
    # saved score. A fresh objective call certifies that cached evaluation.
    blocks, cached_value, cached_gradient = best
    work["audit_objective_evaluations"] += 1
    value, gradient = objective_gradient(blocks.copy())
    value, gradient = float(np.asarray(value)), np.asarray(gradient)
    if gradient.shape != seed.shape:
        raise ValueError("independent L-BFGS-B gradient shape differs")
    np.testing.assert_allclose(value, cached_value, **SCORE_TOLERANCE)
    np.testing.assert_allclose(gradient, cached_gradient, **SCORE_TOLERANCE)
    if not np.isfinite(value) or not np.isfinite(gradient).all():
        raise ValueError("independent L-BFGS-B final audit is nonfinite")
    residual = float(
        np.asarray(_projected_gradient_norm(jnp.asarray(blocks), jnp.asarray(gradient)))
    )
    converged = residual <= policy.gradient_tolerance
    if converged and not work["backend_failure"]:
        work["stop_cause"] = "projected_gradient"
    work.update(
        independent_projected_gradient_inf_norm=residual,
        independent_objective=value,
        returned_canonical_blocks=blocks.tolist(),
        independent_gradient=gradient.tolist(),
        returned_seed=bool(np.array_equal(blocks, seed)),
    )
    return _OptimizerOutcome(
        blocks=jnp.asarray(blocks),
        value=jnp.asarray(value, dtype=np.asarray(seed_value).dtype),
        gradient=jnp.asarray(gradient),
        iterations=work["accepted_iterations"],
        converged=converged,
        stalled=not converged and work["stop_cause"] != "iteration_limit",
        line_search_failed=work["backend_status"] == 2,
        progressed=value < initial_value,
        finite=True,
        stall_message=f"L-BFGS-B stopped: {work['stop_cause']}",
    ), work


class QuasiNewtonSolver(BoundedShootingSolver):
    """Replace only optimization; inherit seeds, dynamics and result checks."""

    def __init__(self, model, policy):
        super().__init__(model, policy)
        self.last_work = _empty_work()
        self._seed_evaluations = 0

    def solve(self, *args, **kwargs):
        self.last_work = _empty_work()
        self._seed_evaluations = 0
        return super().solve(*args, **kwargs)

    def _seed_plan(self, *args, **kwargs):
        kernels = self._kernels

        def counted(*arguments):
            self._seed_evaluations += 1
            return kernels.objective_and_gradient(*arguments)

        self._kernels = replace(kernels, objective_and_gradient=counted)
        try:
            return super()._seed_plan(*args, **kwargs)
        finally:
            self._kernels = kernels
            self.last_work["seed_objective_evaluations"] = self._seed_evaluations

    def _optimize_plan(
        self,
        blocks,
        value,
        gradient,
        state,
        latent,
        reference,
        previous_command,
        exogenous,
        *,
        budget=None,
    ):
        def objective_gradient(candidate):
            return self._kernels.objective_and_gradient(
                jnp.asarray(candidate),
                state,
                latent,
                reference.states,
                previous_command,
                exogenous,
                self.model.values,
            )

        outcome, self.last_work = bounded_minimize(
            objective_gradient, blocks, value, gradient, self.policy
        )
        self.last_work["seed_objective_evaluations"] = self._seed_evaluations
        return outcome
