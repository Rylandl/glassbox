"""Serialize one prewarmed suffix continuation with a complete request deadline."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from dataclasses import asdict, replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import scipy
from investigate_fast_suffix import FastSuffixSolver
from investigate_terminal_suffix import fingerprints

import glassbox
from glassbox.belief.belief import DynamicsBelief
from glassbox.control.fitted import NMPCController
from glassbox.control.plan import SolveStatus
from glassbox.core.dynamics import step_with_latent
from glassbox.core.synthetic import resting_state, true_parameters
from glassbox.workflows.benchmarks import recovery


def solve_waveform(
    solver,
    commands,
    state,
    reference,
    previous_command,
    *,
    latent_state,
    shift_seed=False,
    deadline_s=None,
    phase_times=None,
):
    """Include waveform shifting, validation, transfer and result assembly.

    Construction and compilation are explicit setup, outside this warm request.
    Deadlines are host observations, not preemption of an in-flight JAX kernel.
    """
    started = time.perf_counter()
    valid_deadline = (
        deadline_s is not None and np.isfinite(deadline_s) and deadline_s > 0
    )

    def failure(status, message):
        result = solver._failure_result(status, message, previous_command, started)
        elapsed = time.perf_counter() - started
        return replace(
            result,
            diagnostics=replace(result.diagnostics, solve_time_s=elapsed),
            deadline_met=elapsed < deadline_s if valid_deadline else None,
        )

    if deadline_s is not None and not valid_deadline:
        return failure(SolveStatus.INVALID_INPUT, "invalid waveform request deadline")
    try:
        if shift_seed:
            commands = solver.validate_commands(solver.model, commands)
            commands = jnp.concatenate((commands[1:], commands[-1:]))
        solver.set_seed(commands)
        jax.block_until_ready(solver.seed_commands)
    except (TypeError, ValueError) as exc:
        return failure(SolveStatus.INVALID_INPUT, str(exc))
    remaining = deadline_s
    if phase_times is not None:
        phase_times["preparation_s"] = time.perf_counter() - started
    if valid_deadline:
        remaining -= time.perf_counter() - started
        if remaining <= 0:
            return failure(
                SolveStatus.DEADLINE_EXCEEDED,
                "deadline expired during waveform preparation",
            )
    result = solver.solve(
        state,
        reference,
        previous_command,
        latent_state=latent_state,
        deadline_s=remaining,
    )
    jax.block_until_ready(
        (
            result.command,
            result.predicted_commands,
            result.predicted_states,
            result.predicted_latent_states,
        )
    )
    result = replace(
        result,
        diagnostics=replace(
            result.diagnostics, solve_time_s=time.perf_counter() - started
        ),
    )
    elapsed = time.perf_counter() - started
    if result.command_usable and valid_deadline and elapsed >= deadline_s:
        return failure(
            SolveStatus.DEADLINE_EXCEEDED,
            "deadline expired during waveform result assembly",
        )
    return replace(
        result,
        diagnostics=replace(result.diagnostics, solve_time_s=elapsed),
        deadline_met=elapsed < deadline_s if valid_deadline else None,
    )


def run(fixtures, output):
    belief = DynamicsBelief.load(fixtures / "rich-belief.json")
    checkpoint = np.load(fixtures / "horizon-shift-states.npz")
    controller = NMPCController(belief)
    plan = controller.plan
    reference = controller.hold_reference(jnp.asarray(resting_state()))
    np.testing.assert_array_equal(reference.states, checkpoint["reference_states"])
    state, latent, previous, commands = [
        jnp.asarray(checkpoint["tick4_" + name])
        for name in ("state", "latent", "previous_command", "shifted_commands")
    ]
    target = recovery._arm_configuration_parameters(
        true_parameters(), recovery.TARGET_LOG_ARM_LENGTH_RATIO
    )
    solver = FastSuffixSolver(plan, commands, updates=2, work_estimates=None)
    previous_waveform = jnp.asarray(checkpoint["tick4_warm_commands"])
    np.testing.assert_array_equal(
        jnp.concatenate((previous_waveform[1:], previous_waveform[-1:])), commands
    )
    # Exercise the shifted-input and failure branches too. Both warm-up solves
    # still optimize exactly the measured tick4 seed; the plant never advances.
    for _ in range(2):
        warmed = solve_waveform(
            solver,
            previous_waveform,
            state,
            reference,
            previous,
            latent_state=latent,
            shift_seed=True,
        )
        if not warmed.command_usable:
            raise RuntimeError("the fixed prewarm request did not repair the fixture")
    hold = solver._failure_result(
        SolveStatus.DEADLINE_EXCEEDED,
        "prewarm hold only",
        previous,
        time.perf_counter(),
    )
    jax.block_until_ready(
        (
            hold.command,
            hold.predicted_commands,
            hold.predicted_states,
            hold.predicted_latent_states,
        )
    )
    solver.reports.clear()
    solver.counts.clear()
    events = []
    # Synchronization is already required by each caller. Keeping it here lets
    # early seed failures be attributed even when no optimizer report exists.
    for name in ("evaluate", "linearize", "finalize", "_failure_result"):
        original = getattr(solver, name)

        def measured(*args, _name=name, _original=original, **kwargs):
            started = time.perf_counter()
            value = _original(*args, **kwargs)
            if _name == "_failure_result":
                jax.block_until_ready(
                    (
                        value.command,
                        value.predicted_commands,
                        value.predicted_states,
                        value.predicted_latent_states,
                    )
                )
            else:
                jax.block_until_ready(value)
            events.append({"phase": _name, "elapsed_s": time.perf_counter() - started})
            return value

        setattr(solver, name, measured)
    report = {
        "diagnostic_only": True,
        "glassbox_import": glassbox.__file__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "jax_version": jax.__version__,
        "scipy_version": scipy.__version__,
        "x64": bool(jax.config.jax_enable_x64),
        "deadline_s": 0.02,
        "updates_budget": 2,
        "interval_limit": 36,
        "prewarm_requests": 2,
        "prewarm_shift_branch": True,
        "prewarm_failure_buffers": True,
        "compilation_logging_during_measurement": True,
        "work_estimates": None,
        "timing_scope": "warm waveform shift/validation/transfer, full solve, synchronized outputs and result assembly; plant step and report logging excluded",
        "limitations": [
            "Single serialized run on this host; no hard real-time qualification.",
            "Supplied previous waveform; construction, compilation and cold recovery startup are not measured.",
            "No admission estimates; in-flight kernels cannot be preempted. Late outputs are rejected.",
            "Known physical and actuator states; unchanged fitted uncertainty, support, objective and 0.6 s horizon.",
        ],
        "requests": [],
    }
    for tick in range(36):
        report_count = len(solver.reports)
        events.clear()
        phase_times = {}
        with jax.log_compiles(True):
            caller_started = time.perf_counter()
            result = solve_waveform(
                solver,
                commands,
                state,
                reference,
                previous,
                latent_state=latent,
                shift_seed=tick > 0,
                deadline_s=0.02,
                phase_times=phase_times,
            )
            caller_elapsed = time.perf_counter() - caller_started
        row = {
            "tick": tick,
            "caller_elapsed_s": caller_elapsed,
            "request_elapsed_s": result.diagnostics.solve_time_s,
            "status": str(result.status),
            "message": result.message,
            "command_usable": result.command_usable,
            "deadline_met": result.deadline_met,
            "nonlinear_feasibility": asdict(result.nonlinear_feasibility),
            "counts": dict(solver.counts),
            "phase_times": phase_times,
            "events": list(events),
            "optimizer": solver.reports[-1]
            if len(solver.reports) > report_count
            else None,
            "applied": False,
        }
        report["requests"].append(row)
        # The caller independently gates the complete wrapper return too.
        if not result.command_usable or caller_elapsed >= 0.02:
            report["stop_reason"] = (
                "unusable_solve" if not result.command_usable else "caller_deadline"
            )
            break
        assert result.deadline_met is True
        assert result.nonlinear_feasibility.status == "feasible"
        np.testing.assert_array_equal(
            result.predicted_commands[:24], solver.seed_commands[:24]
        )
        if tick == 0:
            first_suffix = np.asarray(result.predicted_commands[24]).copy()
        if tick == 24:
            np.testing.assert_array_equal(result.command, first_suffix)
        previous = result.command
        commands = result.predicted_commands
        state, latent = step_with_latent(
            target,
            state,
            latent,
            previous,
            plan.sample_period_s,
            belief.input_spec.control_roles,
        )
        actual = float(
            np.max(
                plan._validity_utilization(
                    state, solver._exogenous_forecast(reference)[0]
                )
            )
        )
        row.update(applied=True, actual_support_utilization=actual)
        if (
            not np.all(np.isfinite(state))
            or not np.all(np.isfinite(latent))
            or not (np.isfinite(actual) and actual <= 1 + 1e-6)
        ):
            report["stop_reason"] = "actual_state_failed"
            break
    report.setdefault("stop_reason", "bounded_36_intervals_complete")
    report["applied_intervals"] = sum(row["applied"] for row in report["requests"])
    report.update(fingerprints(fixtures))
    for name in ("investigate_fast_suffix", "investigate_fast_suffix_runtime"):
        path = Path(__file__).with_name(name + ".py")
        report["source_sha256"][f"scripts/{path.name}"] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "stop_reason": report["stop_reason"],
                "applied_intervals": report["applied_intervals"],
                "caller_elapsed_s": [
                    row["caller_elapsed_s"] for row in report["requests"]
                ],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.fixtures, args.output)
