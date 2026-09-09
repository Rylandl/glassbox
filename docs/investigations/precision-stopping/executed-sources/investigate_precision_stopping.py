"""Check finite-precision SQP stopping without hardware timing criteria."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from investigate_feedback_recovery import initial_state, simulate
from investigate_feedback_suffix import checker
from investigate_fused_covariance import FusedLinearizationSolver, compare
from investigate_single_seed import load_request, request
from investigate_terminal_suffix import fingerprints

from glassbox.belief.belief import DynamicsBelief
from glassbox.control.fitted import NMPCController
from glassbox.core.synthetic import resting_state, true_parameters
from glassbox.workflows.benchmarks import recovery

DESIGN = {
    "saved_requests": 26,
    "saved_request_tolerances": {"rtol": 2e-5, "atol": 2e-5},
    "unchanged_request_criterion": "bitwise outputs/objective and identical work counts when no precision stop occurs",
    "stopped_request_criterion": "existing output tolerance, usable independently checked feasible result, identical status and no increase in any kernel call count",
    "closed_loop_order": [
        [case, arm]
        for case in ("cold_original", "cold_small", "cold_original_kick")
        for arm in ("baseline", "precision")
    ],
    "closed_loop_criterion": "All six deadline-free arms complete 120 intervals inside support and satisfy existing terminal full-state tolerances. Tracking and work differences are reported, not retuned.",
    "precision_rule": "Stop with a checked current checkpoint if the successful bounded QP ray has no initial constraint violation and its full linear predicted decrease is nonnegative but below one representable increment of the reported objective. Separately stop if a candidate rounds exactly to the current evaluator input.",
    "limitations": "The objective-resolution rule is a heuristic about the local model, not a nonlinear improvement bound or KKT/stability guarantee. Armijo and feasibility tolerances are unchanged.",
}


def run(fixtures, records, output):
    output.mkdir(parents=True, exist_ok=False)
    prior_path = records / "single-seed-reuse/report.json"
    prior = json.loads(prior_path.read_text())
    provenance = fingerprints(fixtures)
    assert provenance["fixture_sha256"] == prior["fixture_sha256"]
    belief = DynamicsBelief.load(fixtures / "rich-belief.json")
    controller = NMPCController(belief)
    plan = controller.plan
    reference = controller.hold_reference(jnp.asarray(resting_state()))
    with np.load(fixtures / "horizon-shift-states.npz") as fixture:
        original = {
            key: fixture["tick0_" + name].copy()
            for key, name in (
                ("state", "state"),
                ("latent", "latent"),
                ("previous", "previous_command"),
            )
        }
        np.testing.assert_array_equal(reference.states, fixture["reference_states"])
    original.update(absolute_tick=0, commands=None)
    starts = {
        "cold_original": original,
        "cold_small": {**original, "state": initial_state(belief, "small")},
    }
    solvers = {}
    for arm in ("baseline", "precision"):
        solver = FusedLinearizationSolver(
            plan,
            jnp.repeat(jnp.asarray(original["previous"])[None, :], 30, axis=0),
            work_estimates=None,
        )
        solver.reuse_single_seed = True
        solver.precision_stopping = arm == "precision"
        solvers[arm] = solver
    check = checker(plan, reference)
    report = {
        "diagnostic_only": True,
        "deadline_s": None,
        "host_timing_acceptance_gate": False,
        "runtime_performance_claim": False,
        "design": DESIGN,
        "input_report_sha256": hashlib.sha256(prior_path.read_bytes()).hexdigest(),
        "checks": [],
        "cases": {},
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
        "investigate_fused_covariance.py",
        "investigate_single_seed.py",
        "investigate_feedback_recovery.py",
        "investigate_feedback_suffix.py",
        "investigate_fast_suffix.py",
        "investigate_fast_suffix_runtime.py",
        "investigate_recovery.py",
    ):
        content = Path(__file__).with_name(name).read_bytes()
        (archive / name).write_bytes(content)
        report["source_sha256"]["scripts/" + name] = hashlib.sha256(content).hexdigest()

    def save():
        (output / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )

    save()
    for name, source in prior["inputs"].items():
        directory = "feedback-recovery"
        if name == "recorded_tick10":
            directory = "feedback-recovery-runtime"
        elif name == "recorded_small68":
            directory = "seed-timing-history"
        item = load_request(
            records / directory, source["case"], source["tick"], original["previous"]
        )
        assert item["source"] == source
        report["pending_input"] = name
        save()
        outcomes, descriptions = [], []
        for solver in solvers.values():
            result, description = request(solver, item, reference, check)
            outcomes.append(result)
            descriptions.append(description)
        before, after = outcomes
        arrays = (
            "command",
            "predicted_commands",
            "predicted_states",
            "predicted_latent_states",
        )
        row = {
            "input": name,
            "source": source,
            "returned_arrays": compare(
                tuple(getattr(before, key) for key in arrays),
                tuple(getattr(after, key) for key in arrays),
            ),
            "objective": compare(
                before.diagnostics.final_objective, after.diagnostics.final_objective
            ),
            "baseline_objective": before.diagnostics.final_objective,
            "precision_objective": after.diagnostics.final_objective,
            "both_usable": before.command_usable and after.command_usable,
            "status_equal": before.status == after.status,
            "feasibility_equal": before.nonlinear_feasibility
            == after.nonlinear_feasibility,
            "baseline_counts": descriptions[0]["counts"],
            "precision_counts": descriptions[1]["counts"],
            "precision_stop": descriptions[1]["optimizer"]["precision_stop"],
            "precision_stop_reason": descriptions[1]["optimizer"]["stop_reason"],
            "output_source": descriptions[1]["optimizer"]["output_source"],
        }
        row["no_more_kernel_calls"] = all(
            row["precision_counts"].get(key, 0) <= row["baseline_counts"].get(key, 0)
            for key in ("linearize", "evaluate", "finalize")
        )
        row["passed"] = all(
            row[key]
            for key in (
                "both_usable",
                "status_equal",
                "feasibility_equal",
                "no_more_kernel_calls",
            )
        ) and all(
            row[key]["within_tolerance"] for key in ("returned_arrays", "objective")
        )
        if row["precision_stop"] is None:
            row["passed"] = (
                row["passed"]
                and all(
                    row[key]["bitwise_equal"]
                    for key in ("returned_arrays", "objective")
                )
                and row["baseline_counts"] == row["precision_counts"]
            )
        report["checks"].append(row)
        print(
            name,
            "pass" if row["passed"] else "FAIL",
            row["precision_stop_reason"],
            flush=True,
        )
        save()
    report.pop("pending_input", None)
    report["saved_requests_passed"] = len(report["checks"]) == 26 and all(
        row["passed"] for row in report["checks"]
    )
    save()
    if not report["saved_requests_passed"]:
        raise SystemExit(
            "Saved-request criteria failed; closed-loop extension not started."
        )

    target = recovery._arm_configuration_parameters(
        true_parameters(), recovery.TARGET_LOG_ARM_LENGTH_RATIO
    )
    for case_name, arm in DESIGN["closed_loop_order"]:
        name = case_name + "_" + arm
        report["pending_case"] = name
        save()
        case, arrays = simulate(
            solvers[arm],
            plan,
            target,
            reference,
            check,
            starts[case_name.removesuffix("_kick")],
            kick=case_name.endswith("_kick"),
            timing=False,
        )
        path = output / (name + ".npz")
        np.savez_compressed(path, **arrays)
        case.update(
            trace_file=path.name,
            trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            precision_stop_counts=dict(
                Counter(
                    row["optimizer"]["stop_reason"]
                    for row in case["requests"]
                    if row["optimizer"]
                    and row["optimizer"]["precision_stop"] is not None
                )
            ),
            total_kernel_calls=dict(
                sum((Counter(row["counts"]) for row in case["requests"]), Counter())
            ),
        )
        if case_name.endswith("_kick") and case["kick"] is not None:
            with np.load(
                output / (case_name.removesuffix("_kick") + "_" + arm + ".npz")
            ) as baseline:
                case["matched_kick_prefix_verified"] = bool(
                    len(baseline["states"]) > 60
                    and np.array_equal(arrays["states"][:60], baseline["states"][:60])
                    and np.array_equal(
                        case["kick"]["before_state"], baseline["states"][60]
                    )
                    and np.array_equal(
                        arrays["latent_states"][:61], baseline["latent_states"][:61]
                    )
                    and np.array_equal(
                        arrays["seed_waveforms"][:61], baseline["seed_waveforms"][:61]
                    )
                )
        report["cases"][name] = case
        print(name, case["applied_intervals"], case["stop_reason"], flush=True)
        save()
    report.pop("pending_case", None)
    report["all_passed"] = len(report["cases"]) == 6 and all(
        case["interval_limit_completed"]
        and case["all_actual_states_within_support"]
        and case["terminal_full_state_within_tolerances"]
        and case.get("matched_kick_prefix_verified", True)
        for case in report["cases"].values()
    )
    save()
    if not report["all_passed"]:
        raise SystemExit("Closed-loop criteria failed; retain the recorded outcomes.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--records", type=Path, default=Path("docs/investigations"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.fixtures, args.records, args.output)
