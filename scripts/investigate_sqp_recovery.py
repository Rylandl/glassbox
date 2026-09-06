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
from dataclasses import fields
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
from glassbox.control.plan import ConstrainedLeastSquaresPlanModel, SolveStatus
from glassbox.control.solver import (
    BoundedShootingSolver,
    _OptimizerOutcome,
    _SolveAbort,
)
from glassbox.workflows.benchmarks import recovery

FEASIBILITY_TOLERANCE = 1e-6
NUMERICAL_INTERIOR_MARGIN = 1e-5
CURVATURE_REGULARIZATION = 1e-4


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
    ):
        super().__init__(model, policy)
        if not isinstance(model, ConstrainedLeastSquaresPlanModel):
            raise TypeError("SQP requires a ConstrainedLeastSquaresPlanModel")
        if warm_iterations < 1:
            raise ValueError("warm_iterations must be positive")
        self.warm_iterations = warm_iterations
        self.legacy_shift = legacy_shift
        self.reports = []
        self.legacy_seeding = legacy_seeding
        self._seed_derivative = None
        self._iteration_budget = 8
        self._seed_linearization_time_s = 0.0

        def packed(blocks, state, latent, reference, previous, exogenous, values):
            prediction = model.rollout(blocks, state, latent, exogenous, values)
            residuals, margins = residuals_and_margins(
                model, policy, prediction, reference, previous
            )
            return residuals, margins

        def with_aux(*arguments):
            value = packed(*arguments)
            return value, value

        self.linearize = jax.jit(jax.jacfwd(with_aux, has_aux=True))
        self.evaluate = jax.jit(packed)

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
    ):
        # This cache is local to the immediately following optimizer call. Every
        # new request clears it, including requests after an earlier deadline.
        self._seed_derivative = None
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
            )
        context = (
            state,
            latent,
            reference.states,
            previous_command,
            exogenous,
            self.model.values,
        )
        best = None
        for blocks, warm in ((cold_blocks, False), (warm_blocks, True)):
            if blocks is None:
                continue
            residuals, margins = (
                np.asarray(part, dtype=float)
                for part in self.evaluate(blocks, *context)
            )
            cost = float(residuals @ residuals)
            if not np.isfinite(cost) or not np.all(np.isfinite(margins)):
                continue
            violation = np.maximum(-margins, 0.0)
            feasible = np.max(violation, initial=0.0) <= FEASIBILITY_TOLERANCE
            # Feasible seeds outrank cheaper infeasible ones. If both need
            # restoration, start from the smaller L1 violation before cost.
            key = (not feasible, 0.0 if feasible else float(violation.sum()), cost)
            if best is None or key <= best[0]:
                best = key, blocks, warm
        if best is None:
            raise _SolveAbort(SolveStatus.NONFINITE_OBJECTIVE, "no finite SQP seed")
        _, blocks, warm = best
        started = time.perf_counter()
        derivative, values = self.linearize(blocks, *context)
        derivative, values = (
            tuple(np.asarray(part, dtype=float) for part in item)
            for item in (derivative, values)
        )
        self._seed_linearization_time_s = time.perf_counter() - started
        if not all(np.all(np.isfinite(part)) for part in (*derivative, *values)):
            raise _SolveAbort(
                SolveStatus.NONFINITE_OBJECTIVE, "SQP seed linearization is non-finite"
            )
        self._seed_derivative = derivative, values
        residuals, _ = values
        cost = float(residuals @ residuals)
        gradient = 2 * derivative[0].reshape(-1, blocks.size).T @ residuals
        return (
            blocks,
            jnp.asarray(cost),
            jnp.asarray(gradient.reshape(blocks.shape)),
            cost,
            warm,
        )

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
        budget = self._iteration_budget
        seed_derivative, self._seed_derivative = self._seed_derivative, None
        report = {
            "iteration_budget": budget,
            "iterations": 0,
            "seed_linearization_time_s": self._seed_linearization_time_s,
            "linearization_time_s": 0.0,
            "quadratic_step_time_s": 0.0,
            "line_search_time_s": 0.0,
            "qp_unsuccessful_count": 0,
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

        for iteration in range(budget):
            begin = time.perf_counter()
            if iteration == 0 and seed_derivative is not None:
                derivative, values = seed_derivative
            else:
                derivative, values = self.linearize(
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
            gradient_np = 2 * residual_jacobian.T @ residuals
            hessian = (
                2 * residual_jacobian.T @ residual_jacobian
                + CURVATURE_REGULARIZATION * np.eye(size)
            )
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
            begin = time.perf_counter()
            accepted = False
            for trial in range(12):
                alpha = 0.5**trial
                candidate = np.clip(vector + alpha * step, -1.0, 1.0)
                next_residuals, next_margins = (
                    np.asarray(part, dtype=float)
                    for part in self.evaluate(
                        jnp.asarray(candidate.reshape(shape)), *context
                    )
                )
                next_cost = retain(candidate, next_residuals, next_margins)
                if all(
                    np.all(np.isfinite(part))
                    for part in (
                        next_residuals,
                        next_margins,
                    )
                ) and next_cost + penalty * np.maximum(
                    -next_margins, 0
                ).sum() <= merit + 1e-4 * alpha * min(slope, 0.0):
                    vector, accepted = candidate, True
                    break
            report["line_search_time_s"] += time.perf_counter() - begin
            if not accepted:
                report["stop_reason"] = "line_search_stalled"
                break
        report["optimizer_elapsed_s"] = time.perf_counter() - started
        report["feasible"] = best is not None
        report["maximum_constraint_violation"] = None if best is None else best[2]
        self.reports.append(report)
        if best is None:
            raise _SolveAbort(
                SolveStatus.LINE_SEARCH_FAILED, "SQP returned no feasible command plan"
            )
        _, vector, _ = best
        final_value, final_gradient = self._kernels.objective_and_gradient(
            jnp.asarray(vector.reshape(shape)),
            state,
            latent,
            reference.states,
            previous_command,
            exogenous,
            self.model.values,
        )
        return _OptimizerOutcome(
            blocks=jnp.asarray(vector.reshape(shape)),
            value=final_value,
            gradient=final_gradient,
            iterations=report["iterations"],
            converged=False,
            stalled=True,
            line_search_failed=False,
            progressed=float(final_value) < float(value),
            finite=bool(
                np.isfinite(final_value) and np.all(np.isfinite(final_gradient))
            ),
            stall_message="offline SQP feasible iterate; convergence not asserted",
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
