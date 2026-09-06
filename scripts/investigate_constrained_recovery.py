"""Compare soft and explicit support constraints in NMPC-only recovery.

This is an offline reference, not a production solver or real-time benchmark.
It keeps the belief, uncertainty, horizon, objective and command blocks fixed.
Only the optimizer and its explicit support constraints differ between arms.
No supervisor or alternate controller supplies commands in these simulations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from collections import Counter
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import scipy
from investigate_recovery import (
    ADDITIONAL_DURATION_S,
    ADDITIONAL_SEEDS,
    OfflineReferenceSolver,
    initial_state,
)
from scipy.optimize import minimize

from glassbox.control.fitted import NMPCController
from glassbox.control.plan import SolveStatus
from glassbox.control.solver import (
    BoundedShootingSolver,
    _OptimizerOutcome,
    _SolveAbort,
)
from glassbox.core.dynamics import hover_control, step_with_latent
from glassbox.core.geometry import rigid_body_local_error
from glassbox.core.synthetic import resting_state
from glassbox.workflows.benchmarks import recovery

DT = recovery.SAMPLE_DT_S
INTERVALS = 120
FEASIBILITY_TOLERANCE = 1e-6


def support_margins(model, blocks, state, latent, exogenous, values, *, robust):
    prediction = model.rollout(blocks, state, latent, exogenous, values)
    if robust:
        utilization = jax.vmap(model._robust_validity_utilization)(
            prediction.mean_states[1:], prediction.tangent_covariance, exogenous
        )
    else:
        utilization = jax.vmap(model._validity_utilization)(
            prediction.mean_states[1:], exogenous
        )
    # The initial state is fixed, so it cannot be repaired by an optimization
    # variable. Its support is checked separately in the trajectory report.
    return (1.0 - utilization).ravel()


class SupportConstrainedReference(BoundedShootingSolver):
    """SLSQP reference with the maintained objective and explicit support bounds.

    SolveResult's convergence flag describes the production box-only gradient
    criterion. This reference never labels a constrained stationary point with
    that flag: the corresponding Lagrangian residuals are recorded separately.
    A feasible iterate may have higher cost than an infeasible seed. Conversely,
    no infeasible optimizer result displaces an available feasible seed.
    """

    def __init__(self, model, policy, *, robust, maximum_iterations=100):
        super().__init__(model, policy)
        self.maximum_iterations = maximum_iterations
        self.reports = []

        def constraints(blocks, state, latent, exogenous, values):
            return support_margins(
                model, blocks, state, latent, exogenous, values, robust=robust
            )

        self.constraints = jax.jit(constraints)
        self.constraint_jacobian = jax.jit(jax.jacfwd(constraints))

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
        shape = blocks.shape
        context = (state, latent, exogenous, self.model.values)

        def objective(vector):
            objective_value, derivative = self._kernels.objective_and_gradient(
                jnp.asarray(vector.reshape(shape)),
                state,
                latent,
                reference.states,
                previous_command,
                exogenous,
                self.model.values,
            )
            return float(objective_value), np.asarray(derivative, dtype=float).ravel()

        def constraints(vector):
            return np.asarray(
                self.constraints(jnp.asarray(vector.reshape(shape)), *context),
                dtype=float,
            )

        def jacobian(vector):
            return np.asarray(
                self.constraint_jacobian(jnp.asarray(vector.reshape(shape)), *context),
                dtype=float,
            ).reshape(-1, blocks.size)

        seed = np.asarray(blocks, dtype=float).ravel()
        started = time.perf_counter()
        result = minimize(
            objective,
            seed,
            method="SLSQP",
            jac=True,
            bounds=[(-1.0, 1.0)] * blocks.size,
            constraints={"type": "ineq", "fun": constraints, "jac": jacobian},
            options={"maxiter": self.maximum_iterations, "ftol": 1e-6},
        )
        candidates = []
        for name, vector in (("seed", seed), ("optimizer", result.x)):
            candidate_value, derivative = objective(vector)
            margins = constraints(vector)
            if (
                np.isfinite(candidate_value)
                and np.all(np.isfinite(derivative))
                and np.all(np.isfinite(margins))
                and np.all(np.abs(vector) <= 1.0 + FEASIBILITY_TOLERANCE)
                and np.min(margins) >= -FEASIBILITY_TOLERANCE
            ):
                candidates.append((candidate_value, name, vector, derivative, margins))
        selected = min(candidates, key=lambda row: row[0]) if candidates else None
        record = {
            "optimizer_success": bool(result.success),
            "optimizer_status": int(result.status),
            "optimizer_message": str(result.message),
            "iterations": int(result.nit),
            "selected": None if selected is None else selected[1],
            "feasible": selected is not None,
            "maximum_constraint_violation": None,
            "projected_lagrangian_gradient_inf_norm": None,
            "maximum_complementarity_residual": None,
            "maximum_dual_violation": None,
        }
        if selected is not None:
            final_value, name, vector, derivative, margins = selected
            record["maximum_constraint_violation"] = float(max(-np.min(margins), 0.0))
            multipliers = getattr(result, "multipliers", None)
            if (
                name == "optimizer"
                and multipliers is not None
                and np.all(np.isfinite(multipliers))
            ):
                lagrangian = derivative - jacobian(vector).T @ multipliers
                record["projected_lagrangian_gradient_inf_norm"] = float(
                    np.max(np.abs(vector - np.clip(vector - lagrangian, -1.0, 1.0)))
                )
                record["maximum_complementarity_residual"] = float(
                    np.max(np.abs(multipliers * margins))
                )
                record["maximum_dual_violation"] = float(max(-np.min(multipliers), 0.0))
        record["optimizer_elapsed_s"] = time.perf_counter() - started
        self.reports.append(record)
        if selected is None:
            raise _SolveAbort(
                SolveStatus.LINE_SEARCH_FAILED,
                "offline constrained optimization returned no feasible command plan",
            )
        return _OptimizerOutcome(
            blocks=jnp.asarray(vector.reshape(shape)),
            value=jnp.asarray(final_value),
            gradient=jnp.asarray(derivative.reshape(shape)),
            iterations=int(result.nit),
            converged=False,
            stalled=True,
            line_search_failed=False,
            progressed=final_value < float(value),
            finite=True,
            stall_message="offline constrained result; see separate feasibility and KKT diagnostics",
        )


def simulate(
    belief,
    target,
    initial,
    *,
    optimizer,
    maximum_iterations=100,
    controller=None,
    prewarm=False,
):
    controller = NMPCController(belief) if controller is None else controller
    if optimizer == "lbfgsb_soft":
        controller.solver = OfflineReferenceSolver(
            controller.plan, controller.plan.policy
        )
    elif optimizer in ("slsqp_mean", "slsqp_robust"):
        controller.solver = SupportConstrainedReference(
            controller.plan,
            controller.plan.policy,
            robust=optimizer == "slsqp_robust",
            maximum_iterations=maximum_iterations,
        )
    state = jnp.asarray(initial)
    latent = hover_control(target)
    previous = latent
    warm = None
    states = [state]
    commands = []
    results = []
    robust_utilizations = []
    reference = controller.hold_reference(jnp.asarray(resting_state()))
    prewarm_started = time.perf_counter()
    if prewarm:
        first = controller.solve(
            state, reference, previous, applied_command=latent, deadline_s=None
        )
        if first.warm_start is not None:
            controller.solve(
                state,
                reference,
                previous,
                applied_command=latent,
                warm_start=first.warm_start,
                deadline_s=None,
            )
        if hasattr(controller.solver, "reports"):
            controller.solver.reports.clear()
    prewarm_time = time.perf_counter() - prewarm_started if prewarm else 0.0

    @jax.jit
    def robust_utilization(blocks, state, applied):
        return 1.0 - jnp.min(
            support_margins(
                controller.plan,
                blocks,
                state,
                applied,
                jnp.zeros(
                    (controller.prediction_steps, controller.model.exogenous_size)
                ),
                controller.plan.values,
                robust=True,
            )
        )

    for _ in range(INTERVALS):
        result = controller.solve(
            state,
            reference,
            previous,
            applied_command=latent,
            warm_start=warm,
            deadline_s=None,
        )
        results.append(result)
        if not result.command_usable:
            # A failed reference solve ends this arm. A hold or an arrest must
            # not quietly turn an NMPC-only comparison into a different scheme.
            break
        indices = (
            np.arange(controller.plan.block_count) * controller.plan.policy.block_steps
        )
        blocks = controller.solver._normalized_from_commands(
            result.predicted_commands[indices]
        )
        robust_utilizations.append(float(robust_utilization(blocks, state, latent)))
        state, latent = step_with_latent(
            target, state, latent, result.command, DT, belief.input_spec.control_roles
        )
        previous, warm = result.command, result.warm_start
        commands.append(np.asarray(result.command))
        states.append(state)
    states = jnp.asarray(states)
    errors = np.asarray(
        jax.vmap(
            lambda state: rigid_body_local_error(jnp.asarray(resting_state()), state)
        )(states)
    ) / np.asarray(controller.tolerances.local_state_scale)
    utilization = np.asarray(jax.vmap(belief.model.validity_utilization)(states))
    outside = np.flatnonzero(np.max(utilization, axis=1) > 1.0 + FEASIBILITY_TOLERANCE)
    complete = len(commands) == INTERVALS
    cpu_times = np.asarray([r.diagnostics.solve_time_s for r in results])
    reports = getattr(controller.solver, "reports", [])
    return {
        "optimizer": optimizer,
        "maximum_slsqp_iterations": maximum_iterations
        if isinstance(controller.solver, SupportConstrainedReference)
        else None,
        "prewarm_time_s": prewarm_time,
        "solve_times_s": cpu_times.tolist(),
        "complete": complete,
        "executed_intervals": len(commands),
        "stop_status": None if complete else results[-1].status.value,
        "stop_message": None if complete else results[-1].message,
        "tail_normalized_tracking_rms": float(np.sqrt(np.mean(errors[-20:] ** 2)))
        if complete
        else None,
        "terminal_attitude_rate_within_tolerances": bool(
            np.all(np.abs(errors[-1, 6:]) <= 1.0)
        ),
        "maximum_actual_validity_utilization": float(np.max(utilization)),
        "first_actual_support_exit_s": float(outside[0] * DT) if len(outside) else None,
        "all_actual_states_within_support": not len(outside),
        "maximum_planned_mean_validity_utilization": max(
            r.diagnostics.maximum_validity_utilization
            for r in results
            if r.command_usable
        )
        if commands
        else None,
        "maximum_planned_robust_validity_utilization": max(robust_utilizations)
        if robust_utilizations
        else None,
        "finite_states": bool(np.all(np.isfinite(states))),
        "maximum_command_bound_violation": float(
            max(np.max(-np.asarray(commands)), np.max(np.asarray(commands) - 1.0), 0.0)
        )
        if commands
        else None,
        "status_counts": dict(Counter(r.status.value for r in results)),
        "solve_time_median_s": float(np.median(cpu_times)),
        "solve_time_maximum_s": float(np.max(cpu_times)),
        "solve_time_exceeding_model_interval_count": int(np.sum(cpu_times > DT)),
        "constrained_optimizer_reports": reports,
    }


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
            "nominal_deadline_disabled": True,
            "model_support_unchanged": True,
            "mean_covariance_and_objective_unchanged": True,
            "failed_solve_ends_scenario_without_applying_fallback": True,
            "previous_command_is_last_emitted_command": True,
            "robust_constraints_use_existing_marginal_standard_deviation": True,
            "robust_constraints_are_not_a_joint_probability_guarantee": True,
        },
        "environment": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "configuration": {
            "sample_period_s": DT,
            "recovery_duration_s": INTERVALS * DT,
            "tail_duration_s": 20 * DT,
            "constraint_feasibility_tolerance": FEASIBILITY_TOLERANCE,
            "support": belief.support.to_dict(),
            "additional_identification_seeds": list(ADDITIONAL_SEEDS),
        },
        "scenarios": [],
    }
    cases = [
        (name, "original", 100)
        for name in ("maintained_soft", "lbfgsb_soft", "slsqp_mean", "slsqp_robust")
    ]
    cases += [("slsqp_robust", "small", 100), ("slsqp_robust", "original", 8)]
    output.parent.mkdir(parents=True, exist_ok=True)
    for optimizer, disturbance, budget in cases:
        name = f"{optimizer}_{disturbance}"
        if optimizer.startswith("slsqp"):
            name += f"_{budget}"
        print(name, flush=True)
        row = simulate(
            belief,
            target,
            initial_state(stale, disturbance),
            optimizer=optimizer,
            maximum_iterations=budget,
        )
        report["scenarios"].append({"name": name, "disturbance": disturbance, **row})
        print(
            {
                k: row[k]
                for k in (
                    "complete",
                    "executed_intervals",
                    "tail_normalized_tracking_rms",
                    "maximum_actual_validity_utilization",
                    "maximum_planned_robust_validity_utilization",
                )
            },
            flush=True,
        )
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    sources = [
        Path(__file__),
        Path(__file__).with_name("investigate_recovery.py"),
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
