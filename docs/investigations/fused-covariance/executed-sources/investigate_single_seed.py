"""Compare redundant seed evaluation with first-linearization value reuse."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from investigate_fast_suffix_runtime import solve_waveform
from investigate_feedback_recovery import (
    RecoverySolver,
    SeedRequest,
    json_finite,
    prewarm,
    simulate,
)
from investigate_feedback_suffix import checker, describe, digest
from investigate_sqp_recovery import DEFAULT_WORK_ESTIMATES
from investigate_terminal_suffix import fingerprints

from glassbox.belief.belief import DynamicsBelief
from glassbox.control.fitted import NMPCController
from glassbox.core.synthetic import resting_state, true_parameters
from glassbox.workflows.benchmarks import recovery

DESIGN = {
    "parity_ticks": [0, 4, 10, 39, 60, 68, 100, 119],
    "parity_cases": ["cold_original", "cold_small", "cold_original_kick"],
    "additional_parity_inputs": ["recorded_tick10", "recorded_small68"],
    "paired_inputs": ["recorded_tick10", "recorded_small68"],
    "paired_blocks_per_input": 32,
    "paired_order": ["baseline", "reused", "reused", "baseline"],
    "paired_requests": 256,
    "warm_deadline_s": 0.02,
    "cold_deadline_s": 0.1,
    "cold_updates": 8,
    "warm_updates": 2,
    "closed_loop_order": [
        ["cold_original", "baseline"],
        ["cold_original", "reused"],
        ["cold_small", "reused"],
        ["cold_small", "baseline"],
        ["cold_original_kick", "baseline"],
        ["cold_original_kick", "reused"],
    ],
    "closed_loop_limit": 120,
    "selection": "run all fixed pairs and one pass per closed-loop arm only after all 26 deadline-free seed/result parity checks pass",
    "unchanged": "model, covariance, support, objective, horizon, head/middle/tail layout, iteration limits, admission estimates and output reserve",
}


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_request(directory, case_name, tick, initial_previous):
    report = json.loads((directory / "report.json").read_text())
    case = report["cases"][case_name]
    path = directory / case["trace_file"]
    assert file_hash(path) == case["trace_sha256"]
    row = case["requests"][tick]
    assert row["absolute_tick"] == tick
    with np.load(path) as trace:
        state, latent, seed = [
            trace[name][tick].copy()
            for name in ("states", "latent_states", "seed_waveforms")
        ]
        incoming = (
            None if tick == 0 else trace[f"forecast_{tick - 1}"].astype(seed.dtype)
        )
        if incoming is not None:
            np.testing.assert_array_equal(incoming, trace[f"forecast_{tick - 1}"])
        previous = initial_previous if incoming is None else incoming[0]
    for key, value in (
        ("state", state),
        ("latent", latent),
        ("previous_command", previous),
        ("seed", seed),
    ):
        assert digest(value) == row[key + "_sha256"]
    expected = (
        np.repeat(previous[None, :], 30, axis=0)
        if incoming is None
        else np.concatenate((incoming[1:], incoming[-1:]))
    )
    np.testing.assert_array_equal(expected, seed)
    state, latent, previous, seed = map(jnp.asarray, (state, latent, previous, seed))
    return {
        "state": state,
        "latent": latent,
        "previous": previous,
        "seed": seed,
        "request": SeedRequest(
            previous, None if incoming is None else jnp.asarray(incoming)
        ),
        "source": {
            "case": case_name,
            "tick": tick,
            "report_sha256": file_hash(directory / "report.json"),
            "trace_sha256": case["trace_sha256"],
            **{
                key: row[key]
                for key in (
                    "state_sha256",
                    "latent_sha256",
                    "previous_command_sha256",
                    "seed_sha256",
                )
            },
        },
    }


def seed_snapshot(solver, item, reference):
    solver.set_seed(item["request"])
    arguments = solver._seed_plan(
        solver._cold_blocks(item["previous"]),
        None,
        item["state"],
        item["latent"],
        reference,
        item["previous"],
        solver._exogenous_forecast(reference),
    )
    snapshot = jax.tree.map(
        lambda x: np.asarray(x).copy(),
        (arguments, solver._seed_derivative, solver._seed_prediction),
    )
    return snapshot, dict(solver.counts), solver._iteration_budget


def same_tree(left, right):
    assert jax.tree.structure(left) == jax.tree.structure(right)
    for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right)):
        a, b = np.asarray(a), np.asarray(b)
        if a.shape != b.shape or a.dtype != b.dtype or a.tobytes() != b.tobytes():
            return False
    return True


def request(solver, item, reference, check, *, deadline=None):
    solver.reports.clear()
    with jax.log_compiles(deadline is not None):
        started = time.perf_counter()
        result = solve_waveform(
            solver,
            item["request"],
            item["state"],
            reference,
            item["previous"],
            latent_state=item["latent"],
            deadline_s=deadline,
        )
        elapsed = time.perf_counter() - started
    row = describe(
        solver, result, item["state"], item["latent"], item["previous"], check, 0
    )
    np.testing.assert_array_equal(solver.seed_commands, item["seed"])
    if row["optimizer"] is not None:
        assert row["optimizer"]["iteration_budget"] == (
            8 if solver.startup_request else 2
        )
    for key in ("state", "commands"):
        row.pop(key, None)
    row.update(
        caller_elapsed_s=elapsed,
        request_elapsed_s=result.diagnostics.solve_time_s,
        accepted=result.command_usable and (deadline is None or elapsed < deadline),
        applied=False,
    )
    if result.command_usable:
        row["forecast_sha256"] = digest(result.predicted_commands)
    if deadline is not None:
        row["caller_budget_fraction"] = elapsed / deadline
    return result, row


def timing_summary(rows):
    summary = {}
    for name in DESIGN["paired_inputs"]:
        group = {}
        for arm in ("baseline", "reused"):
            selected = [
                row for row in rows if row["input"] == name and row["arm"] == arm
            ]
            fractions = [row["caller_budget_fraction"] for row in selected]
            group[arm] = {
                "requests": len(selected),
                "accepted": sum(row["accepted"] for row in selected),
                "caller_budget_fraction": {
                    "median": float(np.median(fractions)),
                    "p95": float(np.quantile(fractions, 0.95)),
                    "max": max(fractions),
                },
                "status_counts": dict(Counter(row["status"] for row in selected)),
                "message_counts": dict(Counter(row["message"] for row in selected)),
                "evaluate_counts": dict(
                    Counter(row["counts"].get("evaluate", 0) for row in selected)
                ),
                "accepted_forecast_hash_counts": dict(
                    Counter(
                        row["forecast_sha256"] for row in selected if row["accepted"]
                    )
                ),
            }
        group["reused_to_baseline_median_ratio"] = (
            group["reused"]["caller_budget_fraction"]["median"]
            / group["baseline"]["caller_budget_fraction"]["median"]
        )
        matched_ratios = []
        for block in range(DESIGN["paired_blocks_per_input"]):
            selected = [
                row for row in rows if row["input"] == name and row["block"] == block
            ]
            if (
                all(row["accepted"] for row in selected)
                and len({row["forecast_sha256"] for row in selected}) == 1
                and len({row["optimizer"]["iterations"] for row in selected}) == 1
            ):
                baseline = [
                    row["caller_elapsed_s"]
                    for row in selected
                    if row["arm"] == "baseline"
                ]
                reused = [
                    row["caller_elapsed_s"]
                    for row in selected
                    if row["arm"] == "reused"
                ]
                matched_ratios.append(float(np.mean(reused) / np.mean(baseline)))
        group["matched_output_blocks"] = len(matched_ratios)
        group["matched_output_block_median_ratio"] = (
            float(np.median(matched_ratios)) if matched_ratios else None
        )
        summary[name] = group
    return summary


def save(output, report):
    (output / "report.json").write_text(
        json.dumps(json_finite(report), indent=2, allow_nan=False) + "\n"
    )


def run(fixtures, records, output):
    output.mkdir(parents=True, exist_ok=False)
    (output / "design.json").write_text(json.dumps(DESIGN, indent=2) + "\n")
    controller = NMPCController(DynamicsBelief.load(fixtures / "rich-belief.json"))
    plan = controller.plan
    reference = controller.hold_reference(jnp.asarray(resting_state()))
    with np.load(fixtures / "horizon-shift-states.npz") as fixture:
        previous = fixture["tick0_previous_command"].copy()
        np.testing.assert_array_equal(reference.states, fixture["reference_states"])
    inputs = {
        f"{case}_{tick}": load_request(
            records / "feedback-recovery", case, tick, previous
        )
        for case in DESIGN["parity_cases"]
        for tick in DESIGN["parity_ticks"]
    }
    inputs["recorded_tick10"] = load_request(
        records / "feedback-recovery-runtime", "cold_original_kick", 10, previous
    )
    inputs["recorded_small68"] = load_request(
        records / "seed-timing-history", "cold_small", 68, previous
    )
    solver = RecoverySolver(
        plan, inputs["cold_original_0"]["seed"], work_estimates=None
    )
    check = checker(plan, reference)
    provenance = fingerprints(fixtures)
    for dirname in (
        "feedback-recovery",
        "feedback-recovery-runtime",
        "seed-timing-history",
    ):
        original = json.loads((records / dirname / "report.json").read_text())
        assert provenance["fixture_sha256"] == original["fixture_sha256"]
    report = {
        "diagnostic_only": True,
        "design": DESIGN,
        "platform": platform.platform(),
        "jax_version": jax.__version__,
        "x64": bool(jax.config.jax_enable_x64),
        "work_estimates": asdict(DEFAULT_WORK_ESTIMATES),
        "inputs": {name: item["source"] for name, item in inputs.items()},
        "parity": [],
        "paired_requests": [],
        "closed_loop": {},
        "limitations": [
            "Experimental opt-in; maintained production solver and research defaults are unchanged.",
            "Removing an evaluation changes admission work accounting but retains all numerical feasibility and outer deadline gates.",
            "Paired requests reset from saved inputs after every call and apply no command. Closed-loop arms stop on rejection.",
            "Prewarming discards solved outputs and covers both modes. One host observation cannot establish an execution-time bound.",
            "No detailed timing/GC observer is installed; independent checks and report serialization occur outside request timers.",
            "Closed-loop order is predeclared and partly counterbalanced, not randomized; earlier timing choices may change later physical states.",
        ],
        **provenance,
    }
    root = Path(__file__).resolve().parents[1]
    report["baseline_revision"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    archive = output / "executed-sources"
    archive.mkdir()
    for name in (
        Path(__file__).name,
        "investigate_sqp_recovery.py",
        "investigate_feedback_recovery.py",
        "investigate_feedback_suffix.py",
        "investigate_fast_suffix.py",
        "investigate_fast_suffix_runtime.py",
    ):
        source = Path(__file__).with_name(name).read_bytes()
        (archive / name).write_bytes(source)
        report["source_sha256"][f"scripts/{name}"] = hashlib.sha256(source).hexdigest()
    save(output, report)
    for name, item in inputs.items():
        report["pending_parity_input"] = name
        save(output, report)
        solver.reuse_single_seed = False
        baseline, before_counts, before_budget = seed_snapshot(solver, item, reference)
        before, before_row = request(solver, item, reference, check)
        solver.reuse_single_seed = True
        reused, after_counts, after_budget = seed_snapshot(solver, item, reference)
        after, after_row = request(solver, item, reference, check)
        names = (
            "command",
            "predicted_commands",
            "predicted_states",
            "predicted_latent_states",
        )
        parity = {
            "input": name,
            "seed_bitwise": same_tree(baseline, reused),
            "result_bitwise": same_tree(
                tuple(getattr(before, k) for k in names),
                tuple(getattr(after, k) for k in names),
            ),
            "objective_equal": before.diagnostics.final_objective
            == after.diagnostics.final_objective,
            "feasibility_equal": before.nonlinear_feasibility
            == after.nonlinear_feasibility,
            "status_equal": before.status == after.status,
            "both_usable": before.command_usable and after.command_usable,
            "budget_equal": before_budget == after_budget,
            "baseline_seed_counts": before_counts,
            "reused_seed_counts": after_counts,
            "baseline_solve": before_row,
            "reused_solve": after_row,
        }
        parity["passed"] = (
            all(
                parity[key]
                for key in (
                    "seed_bitwise",
                    "result_bitwise",
                    "objective_equal",
                    "feasibility_equal",
                    "status_equal",
                    "both_usable",
                    "budget_equal",
                )
            )
            and before_counts == {"evaluate": 1, "linearize": 1}
            and after_counts == {"linearize": 1}
        )
        report["parity"].append(parity)
    report.pop("pending_parity_input", None)
    save(output, report)
    if not all(row["passed"] for row in report["parity"]):
        print("Parity gate failed; no timing arms executed", flush=True)
        return
    print(f"All {len(inputs)} seed/result parity checks passed", flush=True)
    solver.work_estimates = DEFAULT_WORK_ESTIMATES
    forecasts = {}
    for name in DESIGN["paired_inputs"]:
        item = inputs[name]
        for reuse in (False, True):
            solver.reuse_single_seed = reuse
            prewarm(
                solver,
                item["state"],
                item["latent"],
                item["previous"],
                item["request"],
                reference,
                check,
            )
        for block in range(DESIGN["paired_blocks_per_input"]):
            for arm in DESIGN["paired_order"]:
                solver.reuse_single_seed = arm == "reused"
                result, row = request(
                    solver, item, reference, check, deadline=DESIGN["warm_deadline_s"]
                )
                row.update(input=name, block=block, arm=arm)
                index = len(report["paired_requests"])
                forecasts[f"forecast_{index}"] = np.asarray(result.predicted_commands)
                report["paired_requests"].append(row)
        print(f"Fixed pairs complete for {name}", flush=True)
    assert len(report["paired_requests"]) == DESIGN["paired_requests"]
    report["paired_summary"] = timing_summary(report["paired_requests"])
    np.savez_compressed(output / "paired-forecasts.npz", **forecasts)
    report["paired_forecasts_sha256"] = file_hash(output / "paired-forecasts.npz")
    save(output, report)
    target = recovery._arm_configuration_parameters(
        true_parameters(), recovery.TARGET_LOG_ARM_LENGTH_RATIO
    )
    for name, arm in DESIGN["closed_loop_order"]:
        solver.reuse_single_seed = arm == "reused"
        item = inputs[name + "_0"]
        start = {key: item[key] for key in ("state", "latent", "previous")}
        start.update(absolute_tick=0, commands=None)
        case, arrays = simulate(
            solver,
            plan,
            target,
            reference,
            check,
            start,
            kick=name.endswith("_kick"),
            timing=True,
        )
        key = name + "_" + arm
        path = output / (key + ".npz")
        np.savez_compressed(path, **arrays)
        case.update(
            trace_file=path.name,
            trace_sha256=file_hash(path),
            reuse_single_seed=solver.reuse_single_seed,
        )
        report["closed_loop"][key] = case
        save(output, report)
        print(key, case["applied_intervals"], case["stop_reason"], flush=True)
    print(json.dumps(report["paired_summary"], indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--records", type=Path, default=Path("docs/investigations"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.fixtures, args.records, args.output)
