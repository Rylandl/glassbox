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
    _marginal_standard_deviation,
)
from glassbox.control.plan import SolveStatus
from glassbox.control.solver import (
    BoundedShootingSolver,
    _OptimizerOutcome,
    _SolveAbort,
)
from glassbox.core.geometry import rigid_body_local_error
from glassbox.workflows.benchmarks import recovery

FEASIBILITY_TOLERANCE = 1e-6
NUMERICAL_INTERIOR_MARGIN = 1e-5
CURVATURE_REGULARIZATION = 1e-4


def residuals_and_margins(model, policy, prediction, reference, previous):
    """Express the maintained objective as squared residuals, without reweighting."""
    error = (
        jax.vmap(rigid_body_local_error)(reference[1:], prediction.mean_states[1:])
        / model.tolerances.local_state_scale
    )
    spread = (
        _marginal_standard_deviation(
            jnp.diagonal(prediction.tangent_covariance, axis1=-2, axis2=-1)
        )
        / model.tolerances.local_state_scale
    )
    delta = jnp.diff(
        jnp.concatenate((previous[None, :], prediction.commands), axis=0), axis=0
    ) / (
        policy.command_change_fraction * (model.command_maximum - model.command_minimum)
    )
    utilization = jax.vmap(model._robust_validity_utilization)(
        prediction.mean_states[1:], prediction.tangent_covariance, prediction.exogenous
    )
    safety = jax.vmap(model._safety_violation)(prediction.mean_states[1:])
    residuals = jnp.concatenate(
        (
            (error / jnp.sqrt(len(error))).ravel(),
            (spread / jnp.sqrt(len(error))).ravel(),
            jnp.sqrt(policy.terminal_weight) * error[-1],
            jnp.sqrt(policy.terminal_weight) * spread[-1],
            (delta * jnp.sqrt(policy.command_change_weight / delta.size)).ravel(),
            (
                jax.nn.relu(utilization - 1)
                * jnp.sqrt(policy.validity_weight / utilization.size)
            ).ravel(),
            (safety * jnp.sqrt(policy.safety_weight / safety.size)).ravel(),
        )
    )
    return residuals, (1.0 - utilization).ravel()


def quadratic_step(hessian, gradient, margins, jacobian, blocks):
    """Solve the SQP subproblem in coordinates with identity curvature."""
    size = blocks.size
    factor = np.linalg.cholesky(hessian)
    transform = solve_triangular(factor.T, np.eye(size), lower=False)
    scaled_gradient = transform.T @ gradient
    matrix = np.vstack((jacobian @ transform, transform, -transform))
    offsets = np.concatenate(
        (margins - NUMERICAL_INTERIOR_MARGIN, 1 + blocks, 1 - blocks)
    )
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
        and np.min(margins + jacobian @ step) >= -FEASIBILITY_TOLERANCE
        and np.all(np.abs(blocks + step) <= 1.0 + FEASIBILITY_TOLERANCE)
    )
    return step, multipliers, usable, bool(result.success)


class GaussNewtonReference(BoundedShootingSolver):
    """A bounded SQP experiment; feasible output is distinct from convergence."""

    def __init__(self, model, policy, *, warm_iterations=2, legacy_shift=False):
        super().__init__(model, policy)
        self.warm_iterations = warm_iterations
        self.legacy_shift = legacy_shift
        self.reports = []
        self.constraint_count = model.horizon_steps * 6

        def packed(blocks, state, latent, reference, previous, exogenous, values):
            prediction = model.rollout(blocks, state, latent, exogenous, values)
            residuals, margins = residuals_and_margins(
                model, policy, prediction, reference, previous
            )
            return jnp.concatenate((residuals, margins))

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
        count = self.constraint_count
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
        budget = 8 if not self.reports else self.warm_iterations
        report = {
            "iteration_budget": budget,
            "iterations": 0,
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
                and np.min(margins) >= -FEASIBILITY_TOLERANCE
                and np.all(np.abs(candidate) <= 1.0)
                and (best is None or cost < best[0])
            ):
                best = cost, candidate.copy(), float(max(-np.min(margins), 0.0))
            return cost

        for iteration in range(budget):
            begin = time.perf_counter()
            derivative, values = self.linearize(
                jnp.asarray(vector.reshape(shape)), *context
            )
            derivative = np.asarray(derivative, dtype=float).reshape(-1, size)
            values = np.asarray(values, dtype=float)
            report["linearization_time_s"] += time.perf_counter() - begin
            report["iterations"] = iteration + 1
            if not np.all(np.isfinite(values)) or not np.all(np.isfinite(derivative)):
                report["stop_reason"] = "nonfinite_linearization"
                break
            residuals, margins = values[:-count], values[-count:]
            residual_jacobian, constraint_jacobian = (
                derivative[:-count],
                derivative[-count:],
            )
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
            penalty = max(1.0, 1.1 * np.max(multipliers[:count]))
            violation = np.maximum(-margins, 0).sum()
            merit = cost + penalty * violation
            slope = gradient_np @ step - penalty * violation
            begin = time.perf_counter()
            accepted = False
            for trial in range(12):
                alpha = 0.5**trial
                candidate = np.clip(vector + alpha * step, -1.0, 1.0)
                trial_values = np.asarray(
                    self.evaluate(jnp.asarray(candidate.reshape(shape)), *context),
                    dtype=float,
                )
                next_residuals, next_margins = (
                    trial_values[:-count],
                    trial_values[-count:],
                )
                next_cost = retain(candidate, next_residuals, next_margins)
                if np.all(
                    np.isfinite(trial_values)
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
