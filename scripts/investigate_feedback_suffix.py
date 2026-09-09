"""Fixed four-command head and six-command suffix NMPC feedback experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict, replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from investigate_fast_suffix import FastSuffixSolver, SuffixValues
from investigate_fast_suffix_runtime import solve_waveform
from investigate_sqp_recovery import GaussNewtonReference
from investigate_terminal_suffix import fingerprints

import glassbox
from glassbox.belief.belief import DynamicsBelief
from glassbox.control.fitted import BeliefPlanModel, NMPCController
from glassbox.core.dynamics import step_with_latent
from glassbox.core.synthetic import resting_state, true_parameters
from glassbox.workflows.benchmarks import recovery

HEAD_STEPS = 4
TAIL_STEPS = 6
DESIGN = {
    "horizon_steps": 30,
    "free_command_indices": [0, 1, 2, 3, 24, 25, 26, 27, 28, 29],
    "frozen_command_indices": list(range(4, 24)),
    "updates": 2,
    "roll_perturbation_half_width_fraction": 0.02,
    "continuation_intervals": 36,
    "feedback_difference_tolerance": 1e-5,
    "cold_start_studied": False,
    "timing_studied": False,
}


class FeedbackSuffixPlan(BeliefPlanModel):
    """Physical head/tail commands around an exact dynamic middle waveform."""

    def rollout(self, blocks, initial_state, initial_latent, exogenous, values):
        free = self._commands_from_normalized(blocks)
        commands = jnp.concatenate(
            (free[:HEAD_STEPS], values.frozen_commands, free[HEAD_STEPS:])
        )
        return self.rollout_commands(
            commands, initial_state, initial_latent, exogenous, values
        )


class FeedbackSuffixSolver(FastSuffixSolver):
    """Use the existing one-waveform seed contract and two-update GN backend.

    The inherited generic block expansion is deliberately bypassed. Ten rows
    here denote ten independent physical commands, not ten three-step holds.
    """

    plan_type = FeedbackSuffixPlan
    plan_signature = ":four-head-six-tail-v1"

    def __init__(self, plan, commands, *, work_estimates=None):
        if plan.horizon_steps != DESIGN["horizon_steps"]:
            raise ValueError("the fixed feedback experiment requires 30 stages")
        self.seed_commands = self.validate_commands(plan, commands)
        policy = replace(plan.policy, block_count=HEAD_STEPS + TAIL_STEPS)
        model = self.plan_type(
            belief=plan.belief,
            tolerances=plan.tolerances,
            safety_envelope=plan.safety_envelope,
            policy=policy,
            values=SuffixValues(
                *plan.values, self.seed_commands[HEAD_STEPS:-TAIL_STEPS]
            ),
            compile_signature=plan.compile_signature + self.plan_signature,
        )
        GaussNewtonReference.__init__(
            self, model, policy, warm_iterations=2, work_estimates=work_estimates
        )
        self.counts = {}
        for name in ("evaluate", "linearize", "finalize"):
            original = getattr(self, name)

            def counted(*args, _name=name, _original=original):
                self.counts[_name] = self.counts.get(_name, 0) + 1
                return _original(*args)

            setattr(self, name, counted)

    def set_seed(self, commands):
        self.seed_commands = self.validate_commands(self.model, commands)
        self.model = replace(
            self.model,
            values=self.model.values._replace(
                frozen_commands=self.seed_commands[HEAD_STEPS:-TAIL_STEPS]
            ),
        )
        self.counts.clear()

    def _cold_blocks(self, previous_command):
        return self._normalized_from_commands(
            jnp.concatenate(
                (self.seed_commands[:HEAD_STEPS], self.seed_commands[-TAIL_STEPS:])
            )
        )


def digest(array):
    return hashlib.sha256(np.asarray(array).tobytes()).hexdigest()


def checker(plan, reference):
    """Independent physical-waveform scoring, bypassing the adapter entirely."""

    @jax.jit
    def check(commands, state, latent, previous):
        prediction = plan.rollout_commands(
            commands,
            state,
            latent,
            jnp.zeros((plan.horizon_steps, plan.exogenous_size)),
            plan.values,
        )
        terms = plan.optimization_terms(
            prediction, reference.states, previous, plan.policy
        )
        return prediction, jnp.sum(terms.residuals**2), terms.inequality_margins

    return check


def describe(solver, result, state, latent, previous, check, report_count):
    row = {
        "state": np.asarray(state).tolist(),
        "state_sha256": digest(state),
        "latent_sha256": digest(latent),
        "previous_command_sha256": digest(previous),
        "seed_sha256": digest(solver.seed_commands),
        "status": str(result.status),
        "message": result.message,
        "command_usable": result.command_usable,
        "deadline_met": result.deadline_met,
        "nonlinear_feasibility": asdict(result.nonlinear_feasibility),
        "counts": dict(solver.counts),
        "optimizer": {
            k: v for k, v in solver.reports[-1].items() if not k.endswith("_s")
        }
        if len(solver.reports) > report_count
        else None,
    }
    if not result.command_usable:
        assert result.nonlinear_feasibility.status == "not_assessed"
        return row
    assert result.nonlinear_feasibility.status == "feasible"
    np.testing.assert_array_equal(result.command, result.predicted_commands[0])
    np.testing.assert_array_equal(
        result.predicted_commands[HEAD_STEPS:-TAIL_STEPS],
        solver.seed_commands[HEAD_STEPS:-TAIL_STEPS],
    )
    prediction, cost, margins = check(
        result.predicted_commands, state, latent, previous
    )
    for value in prediction:
        assert np.all(np.isfinite(value))
    margins = np.asarray(margins)
    assert np.min(margins) >= -1e-6
    assert np.isfinite(float(cost))
    np.testing.assert_array_equal(prediction.commands, result.predicted_commands)
    np.testing.assert_allclose(
        prediction.mean_states, result.predicted_states, rtol=2e-6, atol=2e-6
    )
    np.testing.assert_allclose(
        prediction.latent_states,
        result.predicted_latent_states,
        rtol=2e-6,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        cost, result.diagnostics.final_objective, rtol=2e-6, atol=2e-6
    )
    row.update(
        independent_cost=float(cost),
        independent_maximum_utilization=float(1 - margins.min()),
        command=np.asarray(result.command).tolist(),
        commands=np.asarray(result.predicted_commands).tolist(),
        exact_frozen_middle=True,
    )
    return row


def save_report(output, report, fixtures):
    root = Path(__file__).resolve().parents[1]
    report.update(fingerprints(fixtures))
    report["baseline_revision"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    for name in (
        "investigate_feedback_suffix.py",
        "investigate_fast_suffix.py",
        "investigate_fast_suffix_runtime.py",
    ):
        path = Path(__file__).with_name(name)
        report["source_sha256"][f"scripts/{name}"] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Preserve the exact executed research source before any later reporting edit.
    output.with_suffix(".source.py").write_bytes(Path(__file__).read_bytes())
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def run(fixtures, output):
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
    solver = FeedbackSuffixSolver(plan, seed)
    check = checker(plan, reference)
    half_width = (
        belief.model.runtime_spec.validity_envelope.angular_velocity_half_width_rad_s[0]
    )
    perturbation = DESIGN["roll_perturbation_half_width_fraction"] * half_width
    report = {
        "diagnostic_only": True,
        "glassbox_import": glassbox.__file__,
        "design": DESIGN,
        "reference_sha256": digest(reference.states),
        "roll_perturbation_rad_s": perturbation,
        "matched_requests": [],
        "continuation": [],
        "limitations": [
            "Local matched-state response and bounded continuation; no stability or recursive-feasibility claim.",
            "Twenty middle commands are fixed per request; this differs from the fixed-grid space despite equal dimension.",
            "Known physical and actuator states, one supplied shifted waveform, no cold startup.",
        ],
    }
    nominal_result = None
    for label, offset in (
        ("nominal", 0),
        ("minus", -perturbation),
        ("plus", perturbation),
    ):
        varied = state.at[10].add(offset)
        before = len(solver.reports)
        result = solve_waveform(
            solver, seed, varied, reference, previous, latent_state=latent
        )
        row = describe(solver, result, varied, latent, previous, check, before)
        row["request"] = label
        row["initial_support_utilization"] = float(
            np.max(belief.model.validity_utilization(varied))
        )
        report["matched_requests"].append(row)
        if label == "nominal":
            nominal_result = result
        print(label, row["status"], row.get("independent_cost"), flush=True)
    rows = report["matched_requests"]
    for name in ("latent_sha256", "previous_command_sha256", "seed_sha256"):
        assert len({r[name] for r in rows}) == 1
    if all(r["command_usable"] for r in rows):
        nominal, minus, plus = [np.asarray(r["command"]) for r in rows]
        width = np.asarray(plan.command_maximum - plan.command_minimum)
        difference = (plus - minus) / width
        report["feedback"] = {
            "plus_minus_command_difference": (plus - minus).tolist(),
            "normalized_difference_inf": float(np.max(np.abs(difference))),
            "minus_nominal_difference": (minus - nominal).tolist(),
            "plus_nominal_difference": (plus - nominal).tolist(),
            "immediate_response_observed": bool(
                np.max(np.abs(difference)) > DESIGN["feedback_difference_tolerance"]
            ),
        }
    else:
        report["feedback"] = {
            "immediate_response_observed": None,
            "reason": "a matched request was rejected",
        }
    target = recovery._arm_configuration_parameters(
        true_parameters(), recovery.TARGET_LOG_ARM_LENGTH_RATIO
    )
    result = nominal_result
    for tick in range(DESIGN["continuation_intervals"]):
        if not result.command_usable:
            report["continuation_stop"] = "unusable_solve"
            break
        # Every current command can change, including commands formerly in the
        # tail. There is intentionally no old tick24 command-identity assertion.
        previous = result.command
        state, latent = step_with_latent(
            target,
            state,
            latent,
            previous,
            plan.sample_period_s,
            belief.input_spec.control_roles,
        )
        actual = float(np.max(belief.model.validity_utilization(state)))
        row = {"tick": tick, "actual_support_utilization": actual}
        report["continuation"].append(row)
        if (
            not np.all(np.isfinite(state))
            or not np.all(np.isfinite(latent))
            or not (np.isfinite(actual) and actual <= 1 + 1e-6)
        ):
            report["continuation_stop"] = "actual_state_failed"
            break
        if tick < DESIGN["continuation_intervals"] - 1:
            before = len(solver.reports)
            result = solve_waveform(
                solver,
                result.predicted_commands,
                state,
                reference,
                previous,
                latent_state=latent,
                shift_seed=True,
            )
            row["next_solve"] = describe(
                solver, result, state, latent, previous, check, before
            )
    report.setdefault("continuation_stop", "bounded_36_intervals_complete")
    report["applied_intervals"] = len(report["continuation"])
    save_report(output, report, fixtures)
    print(
        json.dumps(
            {
                "feedback": report["feedback"],
                "continuation_stop": report["continuation_stop"],
                "applied_intervals": report["applied_intervals"],
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
