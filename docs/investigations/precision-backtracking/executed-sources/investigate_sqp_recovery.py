"""Profile Gauss-Newton SQP for uncertainty-aware, NMPC-only recovery.

The quadratic subproblem is whitened before SLSQP solves its linear constraints.
The nonlinear objective, covariance and model envelope retain their definitions.
This offline experiment disables deadlines, prewarms kernels, and stops rather
than applying a fallback when a solve cannot produce a feasible plan.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from dataclasses import dataclass, fields
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import scipy
from investigate_constrained_recovery import simulate
from investigate_recovery import ADDITIONAL_DURATION_S, ADDITIONAL_SEEDS, initial_state
from scipy.linalg import solve_triangular
from scipy.optimize import minimize

from glassbox.control.fitted import (
    BeliefPlanModel,
    NMPCController,
)
from glassbox.control.plan import (
    ConstrainedLeastSquaresPlanModel,
    NonlinearFeasibility,
    SolveStatus,
)
from glassbox.control.solver import (
    BoundedShootingSolver,
    _OptimizerOutcome,
    _PlanEvaluation,
    _SolveAbort,
)
from glassbox.workflows.benchmarks import recovery

FEASIBILITY_TOLERANCE = 1e-6
NUMERICAL_INTERIOR_MARGIN = 1e-5
CURVATURE_REGULARIZATION = 1e-4


@dataclass(frozen=True)
class SQPWorkEstimates:
    """Host admission estimates, not worst-case execution-time guarantees.

    Output reserve covers final scoring, feasibility/finite/bound checks, and
    result assembly. Estimates are explicit experiment inputs; an unexpectedly
    slow operation still faces the common solve boundary's deadline rejection.
    """

    linearization_s: float = 0.0055
    quadratic_step_s: float = 0.0015
    evaluation_s: float = 0.00075
    output_reserve_s: float = 0.003

    def __post_init__(self):
        values = tuple(getattr(self, field.name) for field in fields(self))
        if not np.all(np.isfinite(values)) or np.any(np.asarray(values) <= 0):
            raise ValueError("SQP work estimates must be finite and positive")


DEFAULT_WORK_ESTIMATES = SQPWorkEstimates()


def residuals_and_margins(model, policy, prediction, reference, previous):
    """Read the formulation from the model's optional constrained contract."""
    return model.optimization_terms(prediction, reference, previous, policy)


def quadratic_step(hessian, gradient, margins, jacobian, blocks):
    """Solve the SQP subproblem in coordinates with identity curvature."""
    size = blocks.size
    factor = np.linalg.cholesky(hessian)
    transform = solve_triangular(factor.T, np.eye(size), lower=False)
    scaled_gradient = transform.T @ gradient
    matrix = np.vstack((jacobian @ transform, transform, -transform))
    # A constant satisfied constraint (such as initial support at its boundary)
    # cannot be moved into the interior by changing future commands.
    interior = np.where(np.any(jacobian != 0, axis=1), NUMERICAL_INTERIOR_MARGIN, 0.0)
    offsets = np.concatenate((margins - interior, 1 + blocks, 1 - blocks))
    result = minimize(
        lambda vector: (
            0.5 * vector @ vector + scaled_gradient @ vector,
            vector + scaled_gradient,
        ),
        np.zeros(size),
        jac=True,
        method="SLSQP",
        constraints={
            "type": "ineq",
            "fun": lambda vector: offsets + matrix @ vector,
            "jac": lambda vector: matrix,
        },
        options={"ftol": 1e-9, "maxiter": 20},
    )
    step = transform @ result.x
    multipliers = getattr(result, "multipliers", None)
    usable = bool(
        np.all(np.isfinite(step))
        and multipliers is not None
        and np.all(np.isfinite(multipliers))
        and np.min(margins + jacobian @ step, initial=0.0) >= -FEASIBILITY_TOLERANCE
        and np.all(np.abs(blocks + step) <= 1.0 + FEASIBILITY_TOLERANCE)
    )
    return step, multipliers, usable, bool(result.success)


class GaussNewtonReference(BoundedShootingSolver):
    """A bounded SQP experiment; feasible output is distinct from convergence."""

    def __init__(
        self,
        model,
        policy,
        *,
        warm_iterations=2,
        legacy_shift=False,
        legacy_seeding=False,
        work_estimates: SQPWorkEstimates | None = DEFAULT_WORK_ESTIMATES,
        fused_output=True,
        prepared_checkpoints=True,
        reuse_single_seed=False,
        use_observed_linearization_cost=False,
        precision_stopping=False,
    ):
        super().__init__(model, policy)
        if not isinstance(model, ConstrainedLeastSquaresPlanModel):
            raise TypeError("SQP requires a ConstrainedLeastSquaresPlanModel")
        if warm_iterations < 1:
            raise ValueError("warm_iterations must be positive")
        self.work_estimates = work_estimates
        self.fused_output = fused_output
        self.prepared_checkpoints = prepared_checkpoints
        self.reuse_single_seed = reuse_single_seed
        self.use_observed_linearization_cost = use_observed_linearization_cost
        self.precision_stopping = precision_stopping
        self.warm_iterations = warm_iterations
        self.legacy_shift = legacy_shift
        self.reports = []
        self.legacy_seeding = legacy_seeding
        self._seed_derivative = None
        self._seed_prediction = None
        self._iteration_budget = 8
        self._seed_linearization_time_s = 0.0
        # Optional research-only observation. Disabled requests retain the
        # original NumPy materialization and no additional readiness fences.
        self._seed_observer = None
        linearization_rollout = self._linearization_rollout(model)

        def packed(blocks, state, latent, reference, previous, exogenous, values):
            prediction = model.rollout(blocks, state, latent, exogenous, values)
            residuals, margins = residuals_and_margins(
                model, policy, prediction, reference, previous
            )
            return residuals, margins

        def with_aux(blocks, state, latent, reference, previous, exogenous, values):
            prediction = linearization_rollout(blocks, state, latent, exogenous, values)
            terms = residuals_and_margins(
                model, policy, prediction, reference, previous
            )
            pair = tuple(terms)
            if not self.prepared_checkpoints:
                return pair, (pair, None)
            return pair, (
                pair,
                (
                    prediction,
                    model.measure(prediction),
                    model.stage_cost(prediction, reference, previous, policy),
                ),
            )

        self.linearize = jax.jit(jax.jacfwd(with_aux, has_aux=True))
        self.evaluate = jax.jit(packed)

        def final_objective(
            blocks, state, latent, reference, previous, exogenous, values
        ):
            prediction = model.rollout(blocks, state, latent, exogenous, values)
            value = model.stage_cost(prediction, reference, previous, policy)
            measurements = model.measure(prediction)
            margins = model.optimization_terms(
                prediction, reference, previous, policy
            ).inequality_margins
            return value, (prediction, measurements, margins)

        self.finalize = jax.jit(jax.value_and_grad(final_objective, has_aux=True))

    def _linearization_rollout(self, model):
        """Bind an equivalent research rollout only for derivative construction."""
        return model.rollout

    def _permits(self, budget, work_s):
        return (
            self.work_estimates is None
            or budget is None
            or budget.permits(work_s, reserve_s=self.work_estimates.output_reserve_s)
        )

    def _warm_blocks(self, warm_start):
        if not self.legacy_shift:
            return super()._warm_blocks(warm_start)
        commands = np.asarray(warm_start.commands)
        blocks = commands[np.arange(self.model.block_count) * self.policy.block_steps]
        shifted = np.concatenate((blocks[1:], blocks[-1:]))
        return jnp.clip(self._normalized_from_commands(jnp.asarray(shifted)), -1.0, 1.0)

    def _seed_plan(
        self,
        cold_blocks,
        warm_start,
        state,
        latent,
        reference,
        previous_command,
        exogenous,
        *,
        budget=None,
    ):
        # This cache is local to the immediately following optimizer call. Every
        # new request clears it, including requests after an earlier deadline.
        observer = getattr(self, "_seed_observer", None)
        host = np.asarray if observer is None else observer.to_host

        def mark(label):
            if observer is not None:
                observer.mark(label)

        self._seed_derivative = None
        self._seed_prediction = None
        self._seed_linearization_time_s = 0.0
        warm_blocks = None if warm_start is None else self._warm_blocks(warm_start)
        self._iteration_budget = 8 if warm_blocks is None else self.warm_iterations
        if self.legacy_seeding:
            return super()._seed_plan(
                cold_blocks,
                warm_start,
                state,
                latent,
                reference,
                previous_command,
                exogenous,
                budget=budget,
            )
        context = (
            state,
            latent,
            reference.states,
            previous_command,
            exogenous,
            self.model.values,
        )
        estimates = self.work_estimates or SQPWorkEstimates()
        single_seed = getattr(self, "reuse_single_seed", False) and (
            (cold_blocks is None) != (warm_blocks is None)
        )
        if single_seed:
            # There is no ranking decision to make. The mandatory first
            # linearization supplies this same seed's values and prediction.
            warm = cold_blocks is None
            blocks = warm_blocks if warm else cold_blocks
        else:
            best = None
            for blocks, warm in ((cold_blocks, False), (warm_blocks, True)):
                if blocks is None:
                    continue
                if not self._permits(
                    budget, estimates.evaluation_s + estimates.linearization_s
                ):
                    if best is not None:
                        break
                    raise _SolveAbort(
                        SolveStatus.DEADLINE_EXCEEDED, "insufficient SQP seed budget"
                    )
                residuals, margins = (
                    host(part, dtype=float) for part in self.evaluate(blocks, *context)
                )
                mark("seed.rank.start")
                cost = float(residuals @ residuals)
                if not np.isfinite(cost) or not np.all(np.isfinite(margins)):
                    mark("seed.rank.end")
                    continue
                violation = np.maximum(-margins, 0.0)
                feasible = np.max(violation, initial=0.0) <= FEASIBILITY_TOLERANCE
                # Feasible seeds outrank cheaper infeasible ones. If both need
                # restoration, start from the smaller L1 violation before cost.
                key = (not feasible, 0.0 if feasible else float(violation.sum()), cost)
                if best is None or key <= best[0]:
                    best = key, blocks, warm
                mark("seed.rank.end")
            if best is None:
                raise _SolveAbort(SolveStatus.NONFINITE_OBJECTIVE, "no finite SQP seed")
            _, blocks, warm = best
        if not self._permits(budget, estimates.linearization_s):
            raise _SolveAbort(
                SolveStatus.DEADLINE_EXCEEDED, "insufficient SQP linearization budget"
            )
        started = time.perf_counter()
        derivative, (values, prediction) = self.linearize(blocks, *context)
        derivative, values = (
            tuple(host(part, dtype=float) for part in item)
            for item in (derivative, values)
        )
        self._seed_linearization_time_s = time.perf_counter() - started
        mark("seed.finite.start")
        finite = all(np.all(np.isfinite(part)) for part in (*derivative, *values))
        mark("seed.finite.end")
        if not finite:
            raise _SolveAbort(
                SolveStatus.NONFINITE_OBJECTIVE, "SQP seed linearization is non-finite"
            )
        residuals, _ = values
        mark("seed.cost.start")
        cost = float(residuals @ residuals)
        mark("seed.cost.end")
        if not np.isfinite(cost):
            raise _SolveAbort(
                SolveStatus.NONFINITE_OBJECTIVE, "SQP seed objective is non-finite"
            )
        self._seed_derivative = derivative, values
        self._seed_prediction = prediction
        mark("seed.gradient.start")
        gradient = 2 * derivative[0].reshape(-1, blocks.size).T @ residuals
        mark("seed.gradient.end")
        mark("seed.return_arrays.start")
        result = (
            blocks,
            jnp.asarray(cost),
            jnp.asarray(gradient.reshape(blocks.shape)),
            cost,
            warm,
        )
        mark("seed.return_arrays.end")
        return result

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
        shape, size = blocks.shape, blocks.size
        vector = np.asarray(blocks, dtype=float).ravel()
        context = (
            state,
            latent,
            reference.states,
            previous_command,
            exogenous,
            self.model.values,
        )
        best = None
        checkpoint = None
        iteration_budget = self._iteration_budget
        estimates = self.work_estimates or SQPWorkEstimates()
        seed_derivative, self._seed_derivative = self._seed_derivative, None
        seed_prediction, self._seed_prediction = self._seed_prediction, None
        precision_stopping = getattr(self, "precision_stopping", False)
        budgeted = (
            self.work_estimates is not None
            and budget is not None
            and budget.deadline_at is not None
        )
        linearization_estimate = estimates.linearization_s
        if (
            budgeted
            and getattr(self, "use_observed_linearization_cost", False)
            and seed_derivative is not None
            and np.isfinite(self._seed_linearization_time_s)
            and self._seed_linearization_time_s > 0
        ):
            # A request-local scheduling hint, including materialization. Do not
            # carry contention into future requests or mutate configured costs.
            linearization_estimate = max(
                linearization_estimate, self._seed_linearization_time_s
            )
        report = {
            "iteration_budget": iteration_budget,
            "iterations": 0,
            "seed_linearization_time_s": self._seed_linearization_time_s,
            "linearization_admission_estimate_s": linearization_estimate
            if budgeted
            else None,
            "observed_linearization_floor_applied": budgeted
            and linearization_estimate > estimates.linearization_s,
            "linearization_time_s": 0.0,
            "quadratic_step_time_s": 0.0,
            "line_search_time_s": 0.0,
            "qp_unsuccessful_count": 0,
            "precision_stopping_enabled": precision_stopping,
            "precision_stop": None,
            "stop_reason": "iteration_limit",
        }
        started = time.perf_counter()

        def retain(candidate, residuals, margins):
            nonlocal best
            cost = float(residuals @ residuals)
            if (
                np.isfinite(cost)
                and np.all(np.isfinite(margins))
                and np.min(margins, initial=0.0) >= -FEASIBILITY_TOLERANCE
                and np.all(np.abs(candidate) <= 1.0)
                and (best is None or cost < best[0])
            ):
                best = cost, candidate.copy(), float(-np.min(margins, initial=0.0))
            return cost

        def prepare(candidate, derivative, values, prediction):
            nonlocal checkpoint
            if prediction is None:
                return
            residuals, margins = values
            cost = float(prediction[2])
            if (
                np.min(margins, initial=0.0) >= -FEASIBILITY_TOLERANCE
                and np.all(np.isfinite(margins))
                and np.all(np.abs(candidate) <= 1.0)
                and (checkpoint is None or cost < checkpoint.value_float)
            ):
                gradient_np = (
                    2 * np.asarray(derivative[0]).reshape(-1, size).T @ residuals
                )
                plan = _PlanEvaluation.from_prediction(
                    jnp.asarray(candidate.reshape(shape)),
                    prediction[2],
                    jnp.asarray(gradient_np.reshape(shape)),
                    prediction[0],
                    prediction[1],
                    NonlinearFeasibility.from_margins(
                        margins, tolerance=FEASIBILITY_TOLERANCE
                    ),
                )
                if plan.prediction_finite and np.all(np.isfinite(gradient_np)):
                    checkpoint = plan

        # The cached seed has already been evaluated at this request's state,
        # latent state, forecast and model values. It can survive an early stop.
        if seed_derivative is not None:
            retain(vector, *seed_derivative[1])
            prepare(vector, *seed_derivative, seed_prediction)

        for iteration in range(iteration_budget):
            needed = estimates.quadratic_step_s
            if iteration != 0 or seed_derivative is None:
                needed = linearization_estimate
            if not self._permits(budget, needed):
                report["stop_reason"] = "time_budget"
                break
            begin = time.perf_counter()
            if iteration == 0 and seed_derivative is not None:
                derivative, values = seed_derivative
                prediction = seed_prediction
            else:
                derivative, (values, prediction) = self.linearize(
                    jnp.asarray(vector.reshape(shape)), *context
                )
            residuals, margins = (np.asarray(part, dtype=float) for part in values)
            residual_jacobian, constraint_jacobian = (
                np.asarray(part, dtype=float).reshape(part.shape[0], size)
                for part in derivative
            )
            count = margins.size
            report["linearization_time_s"] += time.perf_counter() - begin
            report["iterations"] = iteration + 1
            if not all(
                np.all(np.isfinite(part))
                for part in (
                    residuals,
                    margins,
                    residual_jacobian,
                    constraint_jacobian,
                )
            ):
                report["stop_reason"] = "nonfinite_linearization"
                break
            cost = retain(vector, residuals, margins)
            if iteration != 0 or seed_derivative is None:
                prepare(vector, derivative, (residuals, margins), prediction)
            gradient_np = 2 * residual_jacobian.T @ residuals
            hessian = (
                2 * residual_jacobian.T @ residual_jacobian
                + CURVATURE_REGULARIZATION * np.eye(size)
            )
            if not self._permits(budget, estimates.quadratic_step_s):
                report["stop_reason"] = "time_budget"
                break
            begin = time.perf_counter()
            try:
                step, multipliers, usable, success = quadratic_step(
                    hessian, gradient_np, margins, constraint_jacobian, vector
                )
            except np.linalg.LinAlgError:
                usable, success = False, False
            report["quadratic_step_time_s"] += time.perf_counter() - begin
            report["qp_unsuccessful_count"] += int(not success)
            if not usable:
                report["stop_reason"] = "quadratic_step_failed"
                break
            penalty = max(1.0, 1.1 * np.max(multipliers[:count], initial=0.0))
            violation = np.maximum(-margins, 0).sum()
            merit = cost + penalty * violation
            slope = gradient_np @ step - penalty * violation
            resolution_stop = None
            if precision_stopping:
                current_blocks = np.asarray(jnp.asarray(vector.reshape(shape)))
                if (
                    success
                    and violation == 0.0
                    and checkpoint is not None
                    and np.array_equal(current_blocks, np.asarray(checkpoint.blocks))
                    and np.all(np.abs(vector + step) <= 1.0)
                ):
                    # The regularized GN curvature is positive definite. Along
                    # this un-clipped ray, -g.T p bounds the quadratic model's
                    # predicted decrease for 0 <= alpha <= 1. This is only a
                    # stopping heuristic, not a bound on nonlinear improvement.
                    resolution = float(np.spacing(np.asarray(checkpoint.value)))
                    decrease_bound = float(-gradient_np @ step)
                    if (
                        np.isfinite(resolution)
                        and resolution > 0.0
                        and 0.0 <= decrease_bound < resolution
                    ):
                        resolution_stop = {
                            "iteration": iteration + 1,
                            "objective_resolution": resolution,
                            "linear_decrease_bound": decrease_bound,
                            "objective_dtype": str(np.asarray(checkpoint.value).dtype),
                        }
            begin = time.perf_counter()
            accepted = False
            for trial in range(12):
                if not self._permits(budget, estimates.evaluation_s):
                    report["stop_reason"] = "time_budget"
                    break
                alpha = 0.5**trial
                candidate = np.clip(vector + alpha * step, -1.0, 1.0)
                candidate_blocks = jnp.asarray(candidate.reshape(shape))
                if (
                    precision_stopping
                    and np.asarray(candidate_blocks).tobytes()
                    == current_blocks.tobytes()
                ):
                    # Further halving stays in the current point's rounding
                    # cell. A repeated *trial* point would not imply this.
                    report["stop_reason"] = "representable_step"
                    report["precision_stop"] = {
                        "iteration": iteration + 1,
                        "trial": trial,
                        "blocks_dtype": str(current_blocks.dtype),
                    }
                    break
                next_residuals, next_margins = (
                    np.asarray(part, dtype=float)
                    for part in self.evaluate(candidate_blocks, *context)
                )
                next_cost = retain(candidate, next_residuals, next_margins)
                finite_trial = all(
                    np.all(np.isfinite(part))
                    for part in (
                        next_residuals,
                        next_margins,
                    )
                ) and np.isfinite(next_cost)
                if finite_trial and next_cost + penalty * np.maximum(
                    -next_margins, 0
                ).sum() <= merit + 1e-4 * alpha * min(slope, 0.0):
                    vector, accepted = candidate, True
                    break
                if (
                    resolution_stop is not None
                    and finite_trial
                    and np.min(next_margins, initial=0.0) >= 0.0
                ):
                    # Give useful full steps the unchanged acceptance test.
                    # Only stop backtracking after an actually checked,
                    # feasible trial fails it; do not truncate feasibility repair.
                    report["stop_reason"] = "model_resolution"
                    report["precision_stop"] = {
                        **resolution_stop,
                        "rejected_trial": trial,
                        "rejected_alpha": alpha,
                    }
                    break
            report["line_search_time_s"] += time.perf_counter() - begin
            if not accepted:
                if report["stop_reason"] not in (
                    "time_budget",
                    "representable_step",
                    "model_resolution",
                ):
                    report["stop_reason"] = "line_search_stalled"
                break
        report["optimizer_elapsed_s"] = time.perf_counter() - started
        report["feasible"] = best is not None
        report["maximum_constraint_violation"] = None if best is None else best[2]
        self.reports.append(report)
        if best is None:
            if report["stop_reason"] == "time_budget":
                raise _SolveAbort(
                    SolveStatus.DEADLINE_EXCEEDED,
                    "SQP budget exhausted before finding a feasible plan",
                )
            raise _SolveAbort(
                SolveStatus.LINE_SEARCH_FAILED, "SQP returned no feasible command plan"
            )
        # A prepared checkpoint needs no further model evaluation. Reuse it
        # when it is already the selected plan, or when a better trial cannot
        # be finalized before the output reserve begins.
        can_finalize = self._permits(budget, 0.0)
        checkpoint_selected = checkpoint is not None and (
            not can_finalize
            or np.array_equal(
                np.asarray(best[1].reshape(shape), dtype=checkpoint.blocks.dtype),
                np.asarray(checkpoint.blocks),
            )
        )
        if checkpoint_selected:
            if not can_finalize:
                report["stop_reason"] = "time_budget"
            report["output_source"] = "linearization_checkpoint"
            report["finalization_feasible"] = True
            report["finalization_time_s"] = 0.0
            return _OptimizerOutcome(
                blocks=checkpoint.blocks,
                value=checkpoint.value,
                gradient=checkpoint.gradient,
                evaluation=checkpoint,
                iterations=report["iterations"],
                converged=False,
                stalled=True,
                line_search_failed=False,
                progressed=checkpoint.value_float < float(value),
                finite=True,
                stall_message=(
                    "offline SQP feasible iterate returned at the time budget"
                    if report["stop_reason"] == "time_budget"
                    else "offline SQP feasible iterate; convergence not asserted"
                ),
            )
        if not can_finalize:
            raise _SolveAbort(
                SolveStatus.DEADLINE_EXCEEDED, "insufficient SQP output budget"
            )
        _, vector, _ = best
        output_blocks = jnp.asarray(vector.reshape(shape))
        started_output = time.perf_counter()
        evaluation = None
        report["finalization_feasible"] = None
        report["output_source"] = (
            "finalizer" if self.fused_output else "separate_evaluation"
        )
        if self.fused_output:
            (final_value, (prediction, measurements, margins)), final_gradient = (
                self.finalize(output_blocks, *context)
            )
            margins = np.asarray(margins)
            report["finalization_feasible"] = bool(
                np.all(np.isfinite(margins))
                and np.min(margins, initial=0.0) >= -FEASIBILITY_TOLERANCE
            )
            if not report["finalization_feasible"]:
                raise _SolveAbort(
                    SolveStatus.LINE_SEARCH_FAILED,
                    "final SQP prediction is not feasible",
                )
            evaluation = _PlanEvaluation.from_prediction(
                output_blocks,
                final_value,
                final_gradient,
                prediction,
                measurements,
                NonlinearFeasibility.from_margins(
                    margins, tolerance=FEASIBILITY_TOLERANCE
                ),
            )
        else:
            final_value, final_gradient = self._kernels.objective_and_gradient(
                output_blocks, *context
            )
        finite = bool(np.isfinite(final_value) and np.all(np.isfinite(final_gradient)))
        report["finalization_time_s"] = time.perf_counter() - started_output
        return _OptimizerOutcome(
            blocks=output_blocks,
            value=final_value,
            gradient=final_gradient,
            iterations=report["iterations"],
            converged=False,
            stalled=True,
            line_search_failed=False,
            progressed=float(final_value) < float(value),
            finite=finite,
            stall_message=(
                "offline SQP feasible iterate returned at the time budget"
                if report["stop_reason"] == "time_budget"
                else "offline SQP feasible iterate; convergence not asserted"
            ),
            evaluation=evaluation,
        )


class LegacyClippedPlan(BeliefPlanModel):
    """The removed double clipping, retained only as a controlled ablation."""

    def _commands_from_normalized(self, normalized):
        return jnp.clip(
            self.command_minimum
            + 0.5
            * (jnp.clip(normalized, -1.0, 1.0) + 1.0)
            * (self.command_maximum - self.command_minimum),
            self.command_minimum,
            self.command_maximum,
        )


def run(output):
    stale, belief, target, _ = recovery._build_beliefs()
    for seed in ADDITIONAL_SEEDS:
        trajectory = recovery._configuration_trajectory(
            target,
            recovery.TARGET_LOG_ARM_LENGTH_RATIO,
            seed=seed,
            duration_s=ADDITIONAL_DURATION_S,
            source_group=f"additional-adaptation-{seed}",
        )
        belief, _ = belief.absorb(trajectory)
    report = {
        "format_version": 1,
        "diagnostic_only": True,
        "complete": False,
        "semantics": {
            "synthetic": True,
            "supervisor_or_secondary_controller": False,
            "deadlines_disabled": True,
            "kernels_prewarmed": True,
            "objective_and_uncertainty_unchanged": True,
            "support_unchanged": True,
            "failed_solve_stops_without_applying_fallback": True,
            "feasibility_is_not_a_convergence_claim": True,
        },
        "environment": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "configuration": {
            "sample_period_s": recovery.SAMPLE_DT_S,
            "duration_s": 2.4,
            "cold_sqp_iterations": 8,
            "constraint_tolerance": FEASIBILITY_TOLERANCE,
            "numerical_interior_margin": NUMERICAL_INTERIOR_MARGIN,
            "curvature_regularization": CURVATURE_REGULARIZATION,
            "support": belief.support.to_dict(),
            "additional_identification_seeds": list(ADDITIONAL_SEEDS),
        },
        "scenarios": [],
    }
    cases = [
        ("slsqp", "original", 100),
        ("sqp", "original", 8),
        ("sqp", "original", 2),
        ("whole_block_shift", "original", 2),
        ("clipped_derivatives", "original", 2),
        ("sqp", "small", 2),
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    for method, disturbance, budget in cases:
        name = f"{method}_{disturbance}_{budget}"
        print(name, flush=True)
        controller = NMPCController(belief)
        optimizer = "slsqp_robust" if method == "slsqp" else "gauss_newton_sqp"
        if method != "slsqp":
            if method == "clipped_derivatives":
                values = {
                    field.name: getattr(controller.plan, field.name)
                    for field in fields(controller.plan)
                }
                values["compile_signature"] += ":legacy_clipping"
                controller.plan = LegacyClippedPlan(**values)
            controller.solver = GaussNewtonReference(
                controller.plan,
                controller.plan.policy,
                warm_iterations=budget,
                legacy_shift=method == "whole_block_shift",
            )
        row = simulate(
            belief,
            target,
            initial_state(stale, disturbance),
            optimizer=optimizer,
            maximum_iterations=100,
            controller=controller,
            prewarm=True,
        )
        times = np.asarray(row["solve_times_s"])
        row["steady_solve_time_median_s"] = (
            float(np.median(times[1:])) if len(times) > 1 else None
        )
        row["steady_solve_time_maximum_s"] = (
            float(np.max(times[1:])) if len(times) > 1 else None
        )
        row["steady_deadline_miss_count"] = int(
            np.sum(times[1:] > recovery.SAMPLE_DT_S)
        )
        row["cold_solve_time_s"] = float(times[0])
        row["name"] = name
        row["disturbance"] = disturbance
        report["scenarios"].append(row)
        print(
            {
                key: row[key]
                for key in (
                    "complete",
                    "executed_intervals",
                    "tail_normalized_tracking_rms",
                    "maximum_actual_validity_utilization",
                    "steady_solve_time_median_s",
                    "steady_deadline_miss_count",
                )
            },
            flush=True,
        )
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    sources = [
        Path(__file__),
        Path(__file__).with_name("investigate_recovery.py"),
        Path(__file__).with_name("investigate_constrained_recovery.py"),
        *[
            Path(recovery.glassbox.__file__).parent / name
            for name in recovery.BENCHMARK_SOURCE_FILES
        ],
    ]
    report["source_sha256"] = {
        str(path.relative_to(Path.cwd())): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sources
    }
    report["complete"] = True
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
