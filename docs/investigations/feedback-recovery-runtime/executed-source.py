"""Fixed cold-start/full-recovery validation of the experimental NMPC layout."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from investigate_fast_suffix_runtime import solve_waveform
from investigate_feedback_suffix import FeedbackSuffixSolver, checker, describe, digest
from investigate_recovery import initial_state
from investigate_sqp_recovery import DEFAULT_WORK_ESTIMATES
from investigate_terminal_suffix import fingerprints

import glassbox
from glassbox.belief.belief import DynamicsBelief
from glassbox.control.fitted import NMPCController
from glassbox.control.plan import SolveStatus
from glassbox.core.dynamics import step_with_latent
from glassbox.core.geometry import rigid_body_local_error
from glassbox.core.synthetic import resting_state, true_parameters
from glassbox.workflows.benchmarks import recovery

DESIGN = {
    "end_absolute_tick": 120,
    "sample_period_s": 0.02,
    "cold_updates": 8,
    "warm_updates": 2,
    "kick_absolute_tick": 60,
    "kick_roll_half_width_fraction": 0.02,
    "startup_deadline_s": 0.1,
    "warm_deadline_s": 0.02,
    "cold_seed": "30 repeated previous commands; no solved plan",
    "formulation": "free commands0:4 and24:30; exact supplied middle4:24",
    "selection": "cold original and small; if original cannot finish, separately labeled tick4 continuation; matched kick on completed original or fixture arm",
    "timing_selection": "only cases that completed the deadline-free formulation study",
}


@dataclass(frozen=True)
class SeedRequest:
    previous_command: object
    warm_commands: object = None
    already_shifted: bool = False


class RecoverySolver(FeedbackSuffixSolver):
    """Make cold versus warm mode explicit inside waveform preparation."""

    def set_seed(self, request):
        if not isinstance(request, SeedRequest):
            raise TypeError("recovery requires an explicit seed request")
        cold = request.warm_commands is None
        if cold:
            commands = jnp.repeat(
                jnp.asarray(request.previous_command)[None, :], 30, axis=0
            )
        else:
            commands = self.validate_commands(self.model, request.warm_commands)
            if not request.already_shifted:
                commands = jnp.concatenate((commands[1:], commands[-1:]))
        super().set_seed(commands)
        self.startup_request = cold

    def _seed_plan(self, *args, **kwargs):
        result = super()._seed_plan(*args, **kwargs)
        self._iteration_budget = (
            DESIGN["cold_updates"] if self.startup_request else DESIGN["warm_updates"]
        )
        return result


def kick_state(state, absolute_tick, enabled, magnitude):
    if enabled and absolute_tick == DESIGN["kick_absolute_tick"]:
        return state.at[10].add(magnitude), True
    return state, False


def score(states, scale, *, complete, actual_inside):
    errors = np.asarray(
        jax.vmap(lambda x: rigid_body_local_error(jnp.asarray(resting_state()), x))(
            jnp.asarray(states)
        )
    ) / np.asarray(scale)
    qualified = complete and actual_inside
    return {
        "tail_normalized_tracking_rms": float(np.sqrt(np.mean(errors[-20:] ** 2)))
        if qualified
        else None,
        "terminal_attitude_rate_within_tolerances": bool(
            np.all(np.abs(errors[-1, 6:]) <= 1)
        )
        if qualified
        else None,
        "terminal_full_state_within_tolerances": bool(np.all(np.abs(errors[-1]) <= 1))
        if qualified
        else None,
        "terminal_normalized_error": errors[-1].tolist(),
    }


def json_finite(value):
    """Keep failure reports writable; raw nonfinite arrays remain in the NPZ."""
    if isinstance(value, dict):
        return {k: json_finite(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_finite(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def prewarm(solver, state, latent, previous, seed_request, reference, check):
    held = jnp.repeat(previous[None, :], 30, axis=0)
    # Cover hold construction and both warm seed forms without retaining any
    # solved waveform. These commands are never applied to the plant.
    for request in (SeedRequest(previous), SeedRequest(previous, held), seed_request):
        solver.set_seed(request)
        jax.block_until_ready(solver.seed_commands)
    blocks = solver._cold_blocks(previous)
    context = (
        state,
        latent,
        reference.states,
        previous,
        solver._exogenous_forecast(reference),
        solver.model.values,
    )
    for kernel in (solver.evaluate, solver.linearize, solver.finalize):
        jax.block_until_ready(kernel(blocks, *context))
    for _ in range(2):
        solve_waveform(
            solver, seed_request, state, reference, previous, latent_state=latent
        )
    hold = solver._failure_result(
        SolveStatus.DEADLINE_EXCEEDED, "prewarm only", previous, time.perf_counter()
    )
    jax.block_until_ready(
        (
            hold.command,
            hold.predicted_commands,
            hold.predicted_states,
            hold.predicted_latent_states,
        )
    )
    jax.block_until_ready(check(held, state, latent, previous))
    solver.reports.clear()
    solver.counts.clear()


def simulate(
    solver, plan, target, reference, check, start, *, kick=False, timing=False
):
    state, latent, previous = [
        jnp.asarray(start[k]) for k in ("state", "latent", "previous")
    ]
    first_tick = start["absolute_tick"]
    warm = None if start["commands"] is None else jnp.asarray(start["commands"])
    first_request = SeedRequest(previous, warm, already_shifted=warm is not None)
    if timing:
        prewarm(solver, state, latent, previous, first_request, reference, check)
    rows, states, latents, seeds, forecasts = (
        [],
        [np.asarray(state)],
        [np.asarray(latent)],
        [],
        [],
    )
    actual_utilizations = [float(np.max(plan.model.validity_utilization(state)))]
    actual_inside = actual_utilizations[0] <= 1 + 1e-6
    stop = "interval_limit"
    kick_record = None
    half_width = (
        plan.model.runtime_spec.validity_envelope.angular_velocity_half_width_rad_s[0]
    )
    magnitude = DESIGN["kick_roll_half_width_fraction"] * half_width
    for tick in range(first_tick, DESIGN["end_absolute_tick"]):
        state_before_kick = state
        state, applied_kick = kick_state(state, tick, kick, magnitude)
        if applied_kick:
            # The sample at the impulse time is its right-hand state. Preserve
            # the left-hand state separately for the matched-history audit.
            states[-1] = np.asarray(state)
            kick_record = {
                "absolute_tick": tick,
                "delta_roll_rad_s": magnitude,
                "before_state": np.asarray(state_before_kick).tolist(),
                "after_state": np.asarray(state).tolist(),
                "latent_sha256": digest(latent),
                "previous_command_sha256": digest(previous),
                "incoming_waveform_sha256": digest(warm),
                "support_after_kick": float(
                    np.max(plan.model.validity_utilization(state))
                ),
            }
            actual_utilizations.append(kick_record["support_after_kick"])
        if (
            not np.all(np.isfinite(state))
            or not np.isfinite(actual_utilizations[-1])
            or actual_utilizations[-1] > 1 + 1e-6
        ):
            actual_inside, stop = False, "actual_support_exit"
            break
        request = first_request if tick == first_tick else SeedRequest(previous, warm)
        cold = request.warm_commands is None
        deadline = (
            (DESIGN["startup_deadline_s"] if cold else DESIGN["warm_deadline_s"])
            if timing
            else None
        )
        before = len(solver.reports)
        with jax.log_compiles(timing):
            started = time.perf_counter()
            result = solve_waveform(
                solver,
                request,
                state,
                reference,
                previous,
                latent_state=latent,
                deadline_s=deadline,
            )
            elapsed = time.perf_counter() - started
        row = describe(solver, result, state, latent, previous, check, before)
        commands = row.pop("commands", None)
        row.update(
            absolute_tick=tick,
            relative_tick=tick - first_tick,
            startup=cold,
            deadline_s=deadline,
            applied=False,
        )
        if timing:
            row.update(
                caller_elapsed_s=elapsed,
                request_elapsed_s=result.diagnostics.solve_time_s,
                optimizer_phases=solver.reports[-1]
                if len(solver.reports) > before
                else None,
            )
        if row["optimizer"] is not None:
            assert row["optimizer"]["iteration_budget"] == (8 if cold else 2)
        if cold:
            np.testing.assert_array_equal(
                solver.seed_commands, jnp.repeat(previous[None, :], 30, axis=0)
            )
        seeds.append(np.asarray(solver.seed_commands))
        forecasts.append(
            np.asarray(commands) if commands is not None else np.empty((0, 4))
        )
        rows.append(row)
        if not result.command_usable or (timing and elapsed >= deadline):
            stop = "unusable_solve" if not result.command_usable else "caller_deadline"
            break
        if timing:
            assert result.deadline_met is True
        previous, warm = result.command, result.predicted_commands
        state, latent = step_with_latent(
            target,
            state,
            latent,
            previous,
            plan.sample_period_s,
            plan.belief.input_spec.control_roles,
        )
        states.append(np.asarray(state))
        latents.append(np.asarray(latent))
        actual = float(np.max(plan.model.validity_utilization(state)))
        actual_utilizations.append(actual)
        row.update(applied=True, actual_support_utilization=actual)
        if (
            not np.all(np.isfinite(state))
            or not np.all(np.isfinite(latent))
            or not np.isfinite(actual)
            or actual > 1 + 1e-6
        ):
            actual_inside, stop = False, "actual_support_exit"
            break
    count = len(states) - 1
    complete = count == DESIGN["end_absolute_tick"] - first_tick and actual_inside
    case = {
        "initial_absolute_tick": first_tick,
        "initial_absolute_time_s": first_tick * plan.sample_period_s,
        "final_absolute_time_s": (first_tick + count) * plan.sample_period_s,
        "supplied_solved_waveform": start["commands"] is not None,
        "timing": timing,
        "kick_requested": kick,
        "kick": kick_record,
        "applied_intervals": count,
        "interval_limit_completed": complete,
        "stop_reason": stop,
        "all_actual_states_within_support": actual_inside,
        "maximum_actual_support_utilization": max(actual_utilizations)
        if np.all(np.isfinite(actual_utilizations))
        else None,
        "finite_states": bool(np.all(np.isfinite(states))),
        "finite_latent_states": bool(np.all(np.isfinite(latents))),
        "requests": rows,
        **score(
            states,
            plan.tolerances.local_state_scale,
            complete=complete,
            actual_inside=actual_inside,
        ),
    }
    arrays = {
        "states": np.asarray(states),
        "latent_states": np.asarray(latents),
        "seed_waveforms": np.asarray(seeds),
    }
    for i, commands in enumerate(forecasts):
        arrays[f"forecast_{i}"] = commands
    return case, arrays


def run(fixtures, output, *, formulation_report=None):
    output.mkdir(parents=True, exist_ok=True)
    (output / "design.json").write_text(json.dumps(DESIGN, indent=2) + "\n")
    belief = DynamicsBelief.load(fixtures / "rich-belief.json")
    checkpoint = np.load(fixtures / "horizon-shift-states.npz")
    controller = NMPCController(belief)
    plan = controller.plan
    np.testing.assert_allclose(
        plan.sample_period_s, DESIGN["sample_period_s"], atol=1e-8, rtol=0
    )
    reference = controller.hold_reference(jnp.asarray(resting_state()))
    np.testing.assert_array_equal(reference.states, checkpoint["reference_states"])
    original = {
        k: checkpoint["tick0_" + n]
        for k, n in (
            ("state", "state"),
            ("latent", "latent"),
            ("previous", "previous_command"),
        )
    }
    original.update(absolute_tick=0, commands=None)
    small = {**original, "state": initial_state(belief, "small")}
    fixture = {
        k: checkpoint["tick4_" + n]
        for k, n in (
            ("state", "state"),
            ("latent", "latent"),
            ("previous", "previous_command"),
            ("commands", "shifted_commands"),
        )
    }
    fixture["absolute_tick"] = 4
    starts = {
        "cold_original": original,
        "cold_small": small,
        "fixture_original": fixture,
    }
    timing = formulation_report is not None
    estimates = DEFAULT_WORK_ESTIMATES if timing else None
    solver = RecoverySolver(
        plan,
        jnp.repeat(jnp.asarray(original["previous"])[None, :], 30, axis=0),
        work_estimates=estimates,
    )
    check = checker(plan, reference)
    target = recovery._arm_configuration_parameters(
        true_parameters(), recovery.TARGET_LOG_ARM_LENGTH_RATIO
    )
    report = {
        "diagnostic_only": True,
        "glassbox_import": glassbox.__file__,
        "design": DESIGN,
        "timing": timing,
        "platform": platform.platform(),
        "jax_version": jax.__version__,
        "work_estimates": asdict(estimates) if timing else None,
        "cases": {},
        "limitations": [
            "Known physical and actuator states; fixed learned belief and original uncertainty/support.",
            "Completion is a finite interval count, separate from terminal tolerance checks; no stability or recursive-feasibility claim.",
            "Timing is a single host observation with compilation excluded and prewarm outputs discarded; failures never apply their holds.",
        ],
    }
    if timing:
        prior = json.loads(formulation_report.read_text())
        cases = [
            name
            for name, case in prior["cases"].items()
            if case["interval_limit_completed"]
        ]
        report["formulation_report_sha256"] = hashlib.sha256(
            formulation_report.read_bytes()
        ).hexdigest()
    else:
        cases = ["cold_original", "cold_small"]
    cursor = 0
    while cursor < len(cases):
        name = cases[cursor]
        base = name.removesuffix("_kick")
        case, arrays = simulate(
            solver,
            plan,
            target,
            reference,
            check,
            starts[base],
            kick=name.endswith("_kick"),
            timing=timing,
        )
        trace_path = output / (name + ".npz")
        np.savez_compressed(trace_path, **arrays)
        case["trace_file"] = trace_path.name
        case["trace_sha256"] = hashlib.sha256(trace_path.read_bytes()).hexdigest()
        report["cases"][name] = case
        print(
            name,
            case["stop_reason"],
            case["applied_intervals"],
            case["tail_normalized_tracking_rms"],
            flush=True,
        )
        if not timing:
            if name == "cold_original":
                cases.append(
                    "cold_original_kick"
                    if case["interval_limit_completed"]
                    else "fixture_original"
                )
            elif name == "fixture_original" and case["interval_limit_completed"]:
                cases.append("fixture_original_kick")
        cursor += 1
    if not timing:
        for name, case in report["cases"].items():
            if name.endswith("_kick") and case["kick"] is not None:
                baseline = np.load(output / (name.removesuffix("_kick") + ".npz"))
                varied = np.load(output / (name + ".npz"))
                i = DESIGN["kick_absolute_tick"] - case["initial_absolute_tick"]
                np.testing.assert_array_equal(
                    varied["states"][:i], baseline["states"][:i]
                )
                np.testing.assert_array_equal(
                    case["kick"]["before_state"], baseline["states"][i]
                )
                np.testing.assert_array_equal(
                    varied["latent_states"][: i + 1], baseline["latent_states"][: i + 1]
                )
                np.testing.assert_array_equal(
                    varied["seed_waveforms"][:i], baseline["seed_waveforms"][:i]
                )
                if len(varied["seed_waveforms"]) > i:
                    np.testing.assert_array_equal(
                        varied["seed_waveforms"][i], baseline["seed_waveforms"][i]
                    )
                case["matched_prefix_verified"] = True
    report.update(fingerprints(fixtures))
    root = Path(__file__).resolve().parents[1]
    report["baseline_revision"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    for name in (
        "investigate_feedback_recovery.py",
        "investigate_feedback_suffix.py",
        "investigate_fast_suffix.py",
        "investigate_fast_suffix_runtime.py",
        "investigate_recovery.py",
    ):
        path = Path(__file__).with_name(name)
        report["source_sha256"][f"scripts/{name}"] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    (output / "executed-source.py").write_bytes(Path(__file__).read_bytes())
    (output / "report.json").write_text(
        json.dumps(json_finite(report), indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--formulation-report",
        type=Path,
        help="time only completed cases from this prior report",
    )
    args = parser.parse_args()
    run(args.fixtures, args.output, formulation_report=args.formulation_report)
