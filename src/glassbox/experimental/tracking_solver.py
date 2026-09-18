"""Three frozen oracle solver arms with separate causal histories and work."""

from __future__ import annotations

import time
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from glassbox.control.solver import BoundedShootingSolver, _OptimizerOutcome

from . import task_qualification as task
from .first_order_qualification import FirstOrderSolver
from .precision_solver import PrecisionFactory, SeedCaptureSolver
from .quasi_newton import _empty_work
from .solver_budget import candidate_policy

PG4 = "oracle_pg4"
LBFGS32 = "oracle_lbfgs32"
LBFGS64 = "oracle_lbfgs64"
ARMS = (PG4, LBFGS32, LBFGS64)
RECORD_FIELDS = {
    "arm",
    "origin",
    "work",
    "capture",
    "precision",
    "audits",
    "seed_selection",
}


def _default_precision():
    if jax.config.x64_enabled:
        raise ValueError("tracking and causal replay require default float32")


class SeedOnlySolver(SeedCaptureSolver):
    """Run original float32 cold/warm selection without any optimization."""

    def _optimize_plan(self, blocks, value, gradient, *args, **kwargs):
        del args, kwargs
        return _OptimizerOutcome(
            blocks=blocks,
            value=value,
            gradient=gradient,
            iterations=0,
            converged=False,
            stalled=False,
            line_search_failed=False,
            progressed=False,
            finite=True,
            stall_message="selected seed retained without optimization",
        )


class TrackingOracleArm(task.TaskOracleArm):
    """Use existing solvers behind the unchanged task-scaled oracle seam."""

    def __init__(self, plan, manifest, learned, model, name, *, _equations=None):
        _default_precision()
        if name not in ARMS:
            raise ValueError("undeclared tracking solver arm")
        # The historical task constructor owns the precise scale replacement.
        task_plan, _ = task.frozen_plan()
        super().__init__(
            task_plan,
            manifest,
            learned,
            model,
            task.CANDIDATE,
            _equations=_equations,
        )
        q = plan["control_qualification"]
        for actual, key in (
            (list(self.plan.tolerances.position_m), "position_m"),
            (self.prediction_steps, "horizon_steps"),
            (self.policy.block_count, "block_count"),
            (task.WARMUP_INTERVALS, "warmup_intervals"),
            (self.plan.sample_period_s, "sample_interval_s"),
        ):
            task.same(actual, q[key], label=f"tracking arm {key}")
        self.name = name
        if name != PG4:
            self.policy = candidate_policy(self.policy)
        # Static construction is prewarmed and then all per-trial state resets.
        self._precision_factory = PrecisionFactory(model) if name == LBFGS64 else None
        self.records = []

    def reset(self, initial_state, initial_command):
        _default_precision()
        super().reset(initial_state, initial_command)
        self.records = []

    def observe(self, state):
        _default_precision()
        return super().observe(state)

    def command_applied(self, command):
        _default_precision()
        return super().command_applied(command)

    def solve(self, state, reference, previous_command, **keywords):
        _default_precision()
        if not self.ready or self._applied != self._observed - 1:
            raise ValueError("an oracle solve requires a ready, observed origin")
        if keywords.get("deadline_s") is not None:
            raise ValueError("frozen tracking trials do not apply a solver deadline")
        started_at = time.perf_counter()
        model = self.plan.with_causal_state(self._state)
        constant_objective = self.constant_objective
        record = dict(
            arm=self.name,
            origin=self._applied,
            work=None,
            capture=None,
            precision=None,
            audits=dict(lifted_seed64=None, returned_plan64=None),
            seed_selection=None,
        )
        if self.name == PG4:
            result = BoundedShootingSolver(model, self.policy).solve(
                state, reference, previous_command, **keywords
            )
        elif self.name == LBFGS32:
            solver = FirstOrderSolver(model, self.policy)
            result = solver.solve(state, reference, previous_command, **keywords)
            record["work"] = dict(solver.last_work)
        else:
            selector = SeedOnlySolver(model, self.policy)
            selected = selector.solve(state, reference, previous_command, **keywords)
            capture = selector.last_capture
            selection_succeeded = capture is not None and not selected.used_fallback
            record["seed_selection"] = dict(
                objective_evaluations=selector.last_work["seed_objective_evaluations"],
                successful=selection_succeeded,
                used_warm_start=selected.diagnostics.warm_start_used,
                status=str(selected.status),
                used_fallback=selected.used_fallback,
            )
            if selection_succeeded:
                record["capture"] = capture.record()
                result, work, audits, precision = self._precision_factory.solve(
                    model, self.policy, capture, capture.blocks
                )
                record["work"] = work
                record["precision"] = precision
                record["audits"] = dict(
                    lifted_seed64=audits["baseline"],
                    returned_plan64=audits["candidate"],
                )
                # The native64 variable-objective diagnostic removes the offset
                # with the same lifted coefficients and64 arithmetic as the solve.
                with jax.enable_x64(True):
                    variance = jnp.diagonal(
                        jnp.asarray(
                            capture.values.forecast_error_covariance, dtype=jnp.float64
                        ),
                        axis1=-2,
                        axis2=-1,
                    )
                    scale = jnp.asarray(capture.local_state_scale, dtype=jnp.float64)
                    spread = jnp.sum(variance / jnp.square(scale), axis=1)
                    constant_objective = float(
                        jnp.mean(spread) + self.policy.terminal_weight * spread[-1]
                    )
            else:
                # The inherited bounded hold remains a real, recorded decision.
                # No float64 objective or optimizer was called on this origin.
                result = selected
                record["work"] = _empty_work()
        _default_precision()
        # Synchronize every returned array, including warm starts, before stopping
        # the outer clock. This includes seed setup and both extra64 audits.
        materialized = (
            result.command,
            result.predicted_commands,
            result.predicted_states,
            result.predicted_latent_states,
            None if result.warm_start is None else result.warm_start.commands,
        )
        jax.block_until_ready(materialized)
        self._indices.append(self._applied)
        self._commands.append(np.asarray(result.predicted_commands))
        self._forecasts.append(
            np.zeros((task.HORIZON_STEPS + 1, 13))
            if result.used_fallback
            else np.asarray(result.predicted_states)
        )
        diagnostics = result.diagnostics
        self._objectives.append(diagnostics.final_objective)
        measured = dict(
            initial_objectives=diagnostics.initial_objective,
            projected_gradient_inf_norm=diagnostics.final_projected_gradient_inf_norm,
            iterations=diagnostics.iterations,
            statuses=str(result.status),
            constant_covariance_objectives=constant_objective,
            variable_final_objectives=diagnostics.final_objective - constant_objective,
            used_fallback=result.used_fallback,
        )
        for key, value in measured.items():
            self._extras[key].append(value)
        self.records.append(record)
        elapsed = time.perf_counter() - started_at
        return replace(result, diagnostics=replace(diagnostics, solve_time_s=elapsed))

    def summary(self):
        return dict(
            super().summary(),
            solver={
                PG4: "projected_gradient4",
                LBFGS32: "lbfgs_float32",
                LBFGS64: "lbfgs_float64",
            }[self.name],
            objective_evaluation_counts_available=self.name != PG4,
            constant_covariance_objective_meaning=(
                "float32 seed-selection reference; native per-solve offsets are saved in diagnostics"
                if self.name == LBFGS64
                else "native float32 offset"
            ),
            timing_meaning="synchronized instrumented outer decision including all diagnostic audits",
        )
