"""Audit receding-horizon feasibility and test moving command-block boundaries.

Moving boundaries preserve the shifted command sequence with the same number
of variables. They do not certify a repeated terminal command or the updated
state forecast. All resulting plans still face nonlinear support checks.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
import time
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import scipy
from investigate_constrained_recovery import simulate
from investigate_recovery import ADDITIONAL_DURATION_S, ADDITIONAL_SEEDS, initial_state
from investigate_sqp_recovery import (
    DEFAULT_WORK_ESTIMATES,
    FEASIBILITY_TOLERANCE,
    GaussNewtonReference,
)
from scipy.optimize import minimize

from glassbox.control.fitted import BeliefPlanModel, NMPCController
from glassbox.control.plan import NMPCWarmStart, SolveStatus, blocks_cover_horizon
from glassbox.control.solver import BoundedShootingSolver
from glassbox.core.dynamics import hover_control, step_with_latent
from glassbox.core.synthetic import resting_state
from glassbox.workflows.benchmarks import recovery


def block_indices(horizon_steps, block_count, phase):
    if (
        not isinstance(horizon_steps, int)
        or not isinstance(block_count, int)
        or not 1 <= block_count <= horizon_steps
    ):
        raise ValueError("horizon and block count must be positive integers")
    if not blocks_cover_horizon(horizon_steps, block_count):
        raise ValueError("block layout must cover the horizon without empty blocks")
    width = (horizon_steps + block_count - 1) // block_count
    period = width if block_count > 1 else 1
    if not isinstance(phase, int) or not 0 <= phase < period:
        raise ValueError("block phase is outside the layout period")
    return np.minimum((np.arange(horizon_steps) + phase) // width, block_count - 1)


@dataclass(frozen=True)
class MovingWarmStart(NMPCWarmStart):
    block_count: int
    phase: int

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.block_count, int) or not 1 <= self.block_count <= len(
            self.commands
        ):
            raise ValueError("warm-start block count is invalid")
        if not isinstance(self.phase, int):
            raise ValueError("warm-start phase must be an integer")
        block_indices(len(self.commands), self.block_count, self.phase)


@dataclass(frozen=True, eq=False)
class MovingBlockPlan(BeliefPlanModel):
    phase: int = 0

    def __post_init__(self):
        block_indices(self.horizon_steps, self.block_count, self.phase)

    def _expand_normalized_blocks(self, blocks):
        indices = block_indices(self.horizon_steps, self.block_count, self.phase)
        return blocks[jnp.asarray(indices)]


class PhaseSolver(GaussNewtonReference):
    def _warm_blocks(self, warm_start):
        if not isinstance(warm_start, MovingWarmStart):
            return None
        commands = np.asarray(warm_start.commands)
        if (
            commands.shape != (self.prediction_steps, self.model.command_size)
            or warm_start.block_count != self.model.block_count
        ):
            return None
        period = self.policy.block_steps if self.model.block_count > 1 else 1
        if (warm_start.phase + 1) % period != self.model.phase:
            return None
        shifted = np.concatenate((commands[1:], commands[-1:]))
        indices = block_indices(
            self.prediction_steps, self.model.block_count, self.model.phase
        )
        starts = np.flatnonzero(np.r_[True, np.diff(indices) != 0])
        blocks = shifted[starts]
        # Metadata alone is insufficient: a foreign or edited sequence may not
        # belong to the declared layout. Never silently average it here.
        if not np.array_equal(blocks[indices], shifted):
            return None
        return jnp.clip(self._normalized_from_commands(jnp.asarray(blocks)), -1.0, 1.0)


class MovingBlockSolver(BoundedShootingSolver):
    """An offline phase dispatcher whose warm start carries the layout state."""

    def __init__(self, model, policy, **options):
        self.model, self.policy = model, policy
        self.period = policy.block_steps if model.block_count > 1 else 1
        self.reports = []
        self.solvers = []
        for phase in range(self.period):
            values = {
                field.name: getattr(model, field.name)
                for field in fields(BeliefPlanModel)
            }
            values["compile_signature"] += f":moving-block-phase-{phase}"
            plan = MovingBlockPlan(**values, phase=phase)
            solver = PhaseSolver(plan, policy, **options)
            solver.reports = self.reports
            self.solvers.append(solver)

    def prewarm(self, state, reference, previous_command, *, applied_command):
        # Compile every static phase explicitly. Merely making two warm-up
        # solves would leave the third phase's compilation inside a deadline.
        for solver in self.solvers:
            blocks = solver._cold_blocks(previous_command)
            latent = solver._initial_latent(previous_command, applied_command, None)
            context = (
                state,
                latent,
                reference.states,
                previous_command,
                solver._exogenous_forecast(reference),
                solver.model.values,
            )
            for kernel in (solver.evaluate, solver.linearize, solver.finalize):
                jax.block_until_ready(kernel(blocks, *context))

    def solve(
        self,
        state,
        reference,
        previous_command,
        *,
        applied_command=None,
        latent_state=None,
        warm_start=None,
        deadline_s=None,
    ):
        started = time.perf_counter()
        phase = 0
        compatible = (
            isinstance(warm_start, MovingWarmStart)
            and warm_start.commands.shape
            == (self.model.horizon_steps, self.model.command_size)
            and warm_start.block_count == self.model.block_count
        )
        if compatible:
            phase = (warm_start.phase + 1) % self.period
        solver = self.solvers[phase]
        deadline = deadline_s
        if deadline is not None and np.isfinite(deadline) and deadline > 0:
            deadline -= time.perf_counter() - started
            if deadline <= 0:
                return replace(
                    solver._failure_result(
                        SolveStatus.DEADLINE_EXCEEDED,
                        "deadline expired during block-phase selection",
                        previous_command,
                        started,
                    ),
                    deadline_met=False,
                )
        before = len(self.reports)
        result = solver.solve(
            state,
            reference,
            previous_command,
            applied_command=applied_command,
            latent_state=latent_state,
            warm_start=warm_start if compatible else None,
            deadline_s=deadline,
        )
        if len(self.reports) > before:
            self.reports[-1]["block_phase"] = phase
        if result.command_usable:
            result = replace(
                result,
                warm_start=MovingWarmStart(
                    result.predicted_commands, self.model.block_count, phase
                ),
            )
        elapsed = time.perf_counter() - started
        if result.command_usable and deadline_s is not None and elapsed >= deadline_s:
            return replace(
                solver._failure_result(
                    SolveStatus.DEADLINE_EXCEEDED,
                    "deadline expired during block-phase result assembly",
                    previous_command,
                    started,
                ),
                deadline_met=False,
            )
        return replace(
            result,
            diagnostics=replace(result.diagnostics, solve_time_s=elapsed),
            deadline_met=(
                None
                if deadline_s is None or not np.isfinite(deadline_s) or deadline_s <= 0
                else elapsed < deadline_s
            ),
        )


def terminal_command_probe(plan, commands, state, latent, reference):
    """Search only the last command, leaving the retained prefix unchanged.

    The search is a diagnostic, not a proof of infeasibility or a controller.
    It evaluates the existing last forecast stage, never an unsupported extra
    stage beyond the belief's forecast-error evidence.
    """
    exogenous = jnp.zeros((plan.horizon_steps, plan.exogenous_size))

    @jax.jit
    def margins(tail):
        prediction = plan.rollout_commands(
            commands.at[-1].set(tail), state, latent, exogenous, plan.values
        )
        return plan.optimization_terms(
            prediction, reference.states, latent, plan.policy
        ).inequality_margins.reshape(plan.horizon_steps + 1, -1)[-1]

    derivative = jax.jit(jax.jacfwd(margins))

    def objective(tail):
        violation = np.minimum(np.asarray(margins(tail), dtype=float), 0.0)
        jacobian = np.asarray(derivative(tail), dtype=float)
        return float(violation @ violation), 2 * violation @ jacobian

    lower, upper = np.asarray(plan.command_minimum), np.asarray(plan.command_maximum)
    # Four motor commands: all vertices, the box center, and the held tail.
    starts = [np.asarray(commands[-1]), (lower + upper) / 2]
    starts.extend(
        np.where(bits, upper, lower)
        for bits in itertools.product((False, True), repeat=plan.command_size)
    )
    candidates = []
    for start in starts:
        result = minimize(
            objective,
            np.asarray(start, dtype=float),
            jac=True,
            bounds=list(zip(lower, upper, strict=True)),
            method="L-BFGS-B",
            options={"maxiter": 100, "ftol": 1e-12, "gtol": 1e-8},
        )
        for command in (start, result.x):
            values = np.asarray(margins(command))
            if np.all(np.isfinite(values)):
                candidates.append((float(max(-values.min(), 0.0)), command, values))
    violation, best, values = min(candidates, key=lambda item: item[0])
    return {
        "only_final_command_varies": True,
        "infeasibility_proof": False,
        "starts": len(starts),
        "held_command": np.asarray(commands[-1]).tolist(),
        "best_command": np.asarray(best).tolist(),
        "best_tail_margins": values.tolist(),
        "best_maximum_violation": violation,
        "feasible_found": violation <= FEASIBILITY_TOLERANCE,
    }


def audit_shift(belief, target, initial):
    controller = NMPCController(belief)
    controller.solver = GaussNewtonReference(
        controller.plan, controller.plan.policy, warm_iterations=1, work_estimates=None
    )
    plan = controller.plan
    exogenous = jnp.zeros((plan.horizon_steps, plan.exogenous_size))
    reference = controller.hold_reference(jnp.asarray(resting_state()))

    @jax.jit
    def utilization(commands, state, latent):
        prediction = plan.rollout_commands(
            commands, state, latent, exogenous, plan.values
        )
        margins = plan.optimization_terms(
            prediction, reference.states, latent, plan.policy
        ).inequality_margins
        return (1 - margins).reshape(plan.horizon_steps + 1, -1)

    def summary(commands, state, latent):
        values = np.asarray(utilization(commands, state, latent))
        return {
            "initial": float(values[0].max()),
            "retained_prefix": float(values[1:-1].max()),
            "new_tail": float(values[-1].max()),
            "maximum": float(values.max()),
        }

    state, latent = jnp.asarray(initial), hover_control(target)
    previous, warm, last = latent, None, None
    rows = []
    tail_probe = None
    for tick in range(8):
        row = {"tick": tick, "time_s": tick * plan.sample_period_s}
        if warm is not None:
            shifted = jnp.concatenate((warm.commands[1:], warm.commands[-1:]))
            blocks = controller.solver._warm_blocks(warm)
            averaged = plan._commands_from_normalized(
                plan._expand_normalized_blocks(blocks)
            )
            row.update(
                exact_predicted=summary(
                    shifted, last.predicted_states[1], last.predicted_latent_states[1]
                ),
                exact_actual=summary(shifted, state, latent),
                averaged_actual=summary(averaged, state, latent),
            )
        result = controller.solve(
            state,
            reference,
            previous,
            applied_command=latent,
            warm_start=warm,
            deadline_s=None,
        )
        row.update(status=result.status.value, message=result.message)
        rows.append(row)
        if not result.command_usable:
            if warm is not None:
                tail_probe = terminal_command_probe(
                    plan, shifted, state, latent, reference
                )
            break
        state, latent = step_with_latent(
            target,
            state,
            latent,
            result.command,
            plan.sample_period_s,
            belief.input_spec.control_roles,
        )
        previous, warm, last = result.command, result.warm_start, result
    return {
        "warm_sqp_iterations": 1,
        "ticks": rows,
        "terminal_command_probe": tail_probe,
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
    plan = NMPCController(belief).plan
    report = {
        "format_version": 1,
        "diagnostic_only": True,
        "complete": False,
        "semantics": {
            "synthetic": True,
            "supervisor_or_secondary_controller": False,
            "objective_and_uncertainty_unchanged": True,
            "model_support_unchanged": True,
            "decision_variable_count_unchanged": True,
            "kernels_prewarmed_without_advancing_plant": True,
            "failed_solve_stops_without_applying_fallback": True,
            "moving_blocks_do_not_guarantee_recursive_feasibility": True,
            "timing_is_not_a_hard_realtime_guarantee": True,
        },
        "environment": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "configuration": {
            "sample_period_s": plan.sample_period_s,
            "horizon_steps": plan.horizon_steps,
            "block_count": plan.block_count,
            "command_size": plan.command_size,
            "maximum_error_horizon_s": belief.maximum_error_horizon_s,
            "cold_sqp_iterations": 8,
            "constraint_tolerance": FEASIBILITY_TOLERANCE,
            "support": belief.support.to_dict(),
            "additional_identification_seeds": list(ADDITIONAL_SEEDS),
            "moving_block_indices": [
                block_indices(plan.horizon_steps, plan.block_count, phase).tolist()
                for phase in range(plan.policy.block_steps)
            ],
        },
        "shift_audit": audit_shift(belief, target, initial_state(stale, "original")),
        "scenarios": [],
    }
    cases = [
        ("fixed_one_update", False, 1, "original", None),
        ("moving_one_update", True, 1, "original", None),
        ("fixed_two_updates", False, 2, "original", None),
        ("moving_two_updates", True, 2, "original", None),
        ("moving_budgeted_original", True, 2, "original", 0.02),
        ("moving_budgeted_small", True, 2, "small", 0.02),
        ("moving_outside_support", True, 2, "outside", None),
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    for name, moving, iterations, disturbance, deadline in cases:
        print(name, flush=True)
        controller = NMPCController(belief)
        estimates = None if deadline is None else DEFAULT_WORK_ESTIMATES
        solver_type = MovingBlockSolver if moving else GaussNewtonReference
        controller.solver = solver_type(
            controller.plan,
            controller.plan.policy,
            warm_iterations=iterations,
            work_estimates=estimates,
        )
        initial = np.asarray(
            initial_state(
                stale, "original" if disturbance == "outside" else disturbance
            )
        ).copy()
        if disturbance == "outside":
            envelope = belief.model.runtime_spec.validity_envelope
            initial[10] = (
                envelope.angular_velocity_center_rad_s[0]
                + 1.1 * envelope.angular_velocity_half_width_rad_s[0]
            )
        row = simulate(
            belief,
            target,
            initial,
            optimizer="gauss_newton_sqp",
            controller=controller,
            prewarm=True,
            deadline_s=deadline,
            startup_deadline_s=0.1 if deadline is not None else None,
        )
        times = np.asarray(row["solve_times_s"])
        row.update(
            name=name,
            moving_blocks=moving,
            warm_sqp_iterations=iterations,
            work_estimates=None if estimates is None else asdict(estimates),
            cold_solve_time_s=float(times[0]),
            steady_solve_time_median_s=float(np.median(times[1:]))
            if len(times) > 1
            else None,
            steady_solve_time_maximum_s=float(np.max(times[1:]))
            if len(times) > 1
            else None,
            steady_deadline_miss_count=int(np.sum(times[1:] > plan.sample_period_s)),
        )
        report["scenarios"].append(row)
        print(
            {
                key: row[key]
                for key in (
                    "complete",
                    "executed_intervals",
                    "stop_status",
                    "tail_normalized_tracking_rms",
                    "steady_solve_time_median_s",
                )
            },
            flush=True,
        )
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    sources = [
        Path(__file__),
        *(
            Path(__file__).with_name(name)
            for name in (
                "investigate_sqp_recovery.py",
                "investigate_constrained_recovery.py",
                "investigate_recovery.py",
            )
        ),
        *(
            Path(recovery.glassbox.__file__).parent / name
            for name in recovery.BENCHMARK_SOURCE_FILES
        ),
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
