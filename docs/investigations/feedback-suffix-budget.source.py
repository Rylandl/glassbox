"""One serialized warm deadline trial of the fixed head/suffix formulation."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import scipy
from investigate_fast_suffix_runtime import solve_waveform
from investigate_feedback_suffix import DESIGN, FeedbackSuffixSolver, checker, describe
from investigate_sqp_recovery import DEFAULT_WORK_ESTIMATES
from investigate_terminal_suffix import fingerprints

import glassbox
from glassbox.belief.belief import DynamicsBelief
from glassbox.control.fitted import NMPCController
from glassbox.control.plan import SolveStatus
from glassbox.core.dynamics import step_with_latent
from glassbox.core.synthetic import resting_state, true_parameters
from glassbox.workflows.benchmarks import recovery


def run(fixtures, output, *, admission=False):
    belief = DynamicsBelief.load(fixtures / "rich-belief.json")
    checkpoint = np.load(fixtures / "horizon-shift-states.npz")
    controller = NMPCController(belief)
    plan = controller.plan
    reference = controller.hold_reference(jnp.asarray(resting_state()))
    np.testing.assert_array_equal(reference.states, checkpoint["reference_states"])
    state, latent, previous, seed = [
        jnp.asarray(checkpoint["tick4_" + name])
        for name in ("state", "latent", "previous_command", "shifted_commands")
    ]
    previous_waveform = jnp.asarray(checkpoint["tick4_warm_commands"])
    np.testing.assert_array_equal(
        jnp.concatenate((previous_waveform[1:], previous_waveform[-1:])), seed
    )
    estimates = DEFAULT_WORK_ESTIMATES if admission else None
    solver = FeedbackSuffixSolver(plan, seed, work_estimates=estimates)
    check = checker(plan, reference)
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
            raise RuntimeError("the unchanged prewarm request failed")
    jax.block_until_ready(check(warmed.predicted_commands, state, latent, previous))
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
    target = recovery._arm_configuration_parameters(
        true_parameters(), recovery.TARGET_LOG_ARM_LENGTH_RATIO
    )
    report = {
        "diagnostic_only": True,
        "glassbox_import": glassbox.__file__,
        "platform": platform.platform(),
        "jax_version": jax.__version__,
        "scipy_version": scipy.__version__,
        "x64": bool(jax.config.jax_enable_x64),
        "formulation_design": {**DESIGN, "timing_studied": True},
        "deadline_s": 0.02,
        "prewarm_shifted_requests": 2,
        "prewarm_failure_buffers": True,
        "work_estimates": asdict(estimates) if admission else None,
        "compilation_logging_during_measurement": True,
        "timing_scope": "waveform shifting, validation, transfer, full bounded solve and synchronized result assembly; caller applies a second elapsed deadline gate",
        "excluded_from_timing": "construction, compilation, independent waveform verification, reporting and simulated plant steps",
        "limitations": [
            "One run on this host; no hardware real-time or cold-start qualification.",
            "Known physical and actuator states, fixed belief and original 0.6 s uncertainty evidence.",
            "Same maximum of two updates; optional unchanged historical admission estimates are not execution-time bounds. No in-flight kernel preemption.",
        ],
        "requests": [],
    }
    for tick in range(DESIGN["continuation_intervals"]):
        before = len(solver.reports)
        phase_times = {}
        with jax.log_compiles(True):
            started = time.perf_counter()
            result = solve_waveform(
                solver,
                seed,
                state,
                reference,
                previous,
                latent_state=latent,
                shift_seed=tick > 0,
                deadline_s=report["deadline_s"],
                phase_times=phase_times,
            )
            caller_elapsed = time.perf_counter() - started
        # Independent validation is logged outside the measured control path.
        row = describe(solver, result, state, latent, previous, check, before)
        row.update(
            tick=tick,
            caller_elapsed_s=caller_elapsed,
            request_elapsed_s=result.diagnostics.solve_time_s,
            phase_times=phase_times,
            optimizer_phases=solver.reports[-1]
            if len(solver.reports) > before
            else None,
            applied=False,
        )
        report["requests"].append(row)
        if not result.command_usable or caller_elapsed >= report["deadline_s"]:
            report["stop_reason"] = (
                "unusable_solve" if not result.command_usable else "caller_deadline"
            )
            break
        assert result.deadline_met is True
        previous = result.command
        seed = result.predicted_commands
        state, latent = step_with_latent(
            target,
            state,
            latent,
            previous,
            plan.sample_period_s,
            belief.input_spec.control_roles,
        )
        actual = float(np.max(belief.model.validity_utilization(state)))
        row.update(
            applied=True,
            actual_support_utilization=actual,
            actual_state=np.asarray(state).tolist(),
        )
        if (
            not np.all(np.isfinite(state))
            or not np.all(np.isfinite(latent))
            or not (np.isfinite(actual) and actual <= 1 + 1e-6)
        ):
            report["stop_reason"] = "actual_state_failed"
            break
    report.setdefault("stop_reason", "bounded_36_intervals_complete")
    report["applied_intervals"] = sum(r["applied"] for r in report["requests"])
    report.update(fingerprints(fixtures))
    root = Path(__file__).resolve().parents[1]
    report["baseline_revision"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    for name in (
        "investigate_feedback_suffix_runtime.py",
        "investigate_feedback_suffix.py",
        "investigate_fast_suffix.py",
        "investigate_fast_suffix_runtime.py",
    ):
        path = Path(__file__).with_name(name)
        report["source_sha256"][f"scripts/{name}"] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".source.py").write_bytes(Path(__file__).read_bytes())
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "stop_reason": report["stop_reason"],
                "applied_intervals": report["applied_intervals"],
                "caller_elapsed_s": [r["caller_elapsed_s"] for r in report["requests"]],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--admission",
        action="store_true",
        help="use the existing unchanged SQP work estimates",
    )
    args = parser.parse_args()
    run(args.fixtures, args.output, admission=args.admission)
