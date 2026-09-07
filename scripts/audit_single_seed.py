"""Audit saved single-seed comparisons without rerunning optimization."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from investigate_single_seed import timing_summary


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(directory, output):
    report = json.loads((directory / "report.json").read_text())
    for path in (directory / "executed-sources").glob("*.py"):
        assert digest(path) == report["source_sha256"]["scripts/" + path.name]
    assert len(report["parity"]) == 26 and all(
        row["passed"] for row in report["parity"]
    )
    for row in report["parity"]:
        before, after = row["baseline_solve"], row["reused_solve"]
        assert before["forecast_sha256"] == after["forecast_sha256"]
        assert before["nonlinear_feasibility"] == after["nonlinear_feasibility"]
        assert (
            before["counts"].get("evaluate", 0)
            == after["counts"].get("evaluate", 0) + 1
        )
        for kernel in ("linearize", "finalize"):
            assert before["counts"].get(kernel, 0) == after["counts"].get(kernel, 0)
    assert (
        digest(directory / "paired-forecasts.npz") == report["paired_forecasts_sha256"]
    )
    rows = report["paired_requests"]
    assert len(rows) == report["design"]["paired_requests"]
    expected = [
        (name, arm)
        for name in report["design"]["paired_inputs"]
        for _ in range(report["design"]["paired_blocks_per_input"])
        for arm in report["design"]["paired_order"]
    ]
    assert [(row["input"], row["arm"]) for row in rows] == expected
    with np.load(directory / "paired-forecasts.npz") as arrays:
        for i, row in enumerate(rows):
            assert not row["applied"]
            assert row["accepted"] == (
                row["command_usable"]
                and row["caller_elapsed_s"] < report["design"]["warm_deadline_s"]
            )
            for key in (
                "state_sha256",
                "latent_sha256",
                "previous_command_sha256",
                "seed_sha256",
            ):
                assert row[key] == report["inputs"][row["input"]][key]
            if row["command_usable"]:
                commands = arrays[f"forecast_{i}"]
                assert (
                    hashlib.sha256(commands.tobytes()).hexdigest()
                    == row["forecast_sha256"]
                )
                np.testing.assert_array_equal(commands[0], row["command"])
                feasibility = row["nonlinear_feasibility"]
                assert feasibility["maximum_violation"] <= feasibility["tolerance"]
            else:
                assert row["nonlinear_feasibility"]["constraint_count"] is None
    summary = json.loads(json.dumps(timing_summary(rows)))
    assert summary == report["paired_summary"]
    cases = {}
    for name, case in report["closed_loop"].items():
        assert digest(directory / case["trace_file"]) == case["trace_sha256"]
        with np.load(directory / case["trace_file"]) as trace:
            assert len(trace["states"]) == case["applied_intervals"] + 1
        assert not case["interval_limit_completed"]
        assert case["tail_normalized_tracking_rms"] is None
        last = case["requests"][-1]
        assert not last["applied"]
        assert (
            not last["command_usable"] or last["caller_elapsed_s"] >= last["deadline_s"]
        )
        for row in case["requests"]:
            if row["optimizer"] is not None:
                assert row["optimizer"]["iteration_budget"] == (
                    8 if row["startup"] else 2
                )
        phases = last["optimizer_phases"]
        cases[name] = {
            "applied_intervals": case["applied_intervals"],
            "failed_tick": last["absolute_tick"],
            "failed_startup": last["startup"],
            "message": last["message"],
            "caller_budget_fraction": last["caller_elapsed_s"] / last["deadline_s"],
            "seed_linearization_to_admission_estimate_ratio": phases[
                "seed_linearization_time_s"
            ]
            / report["work_estimates"]["linearization_s"]
            if phases
            else None,
            "optimizer_had_feasible_candidate": last["optimizer"]["feasible"]
            if last["optimizer"]
            else False,
        }
    accepted = {}
    for arm in ("baseline", "reused"):
        group = [row for row in rows if row["arm"] == arm]
        accepted[arm] = {
            "requests": len(group),
            "accepted": sum(row["accepted"] for row in group),
            "caller_rejected_usable_outputs": sum(
                row["command_usable"] and not row["accepted"] for row in group
            ),
            "accepted_optimizer_iteration_counts": dict(
                Counter(
                    row["optimizer"]["iterations"] for row in group if row["accepted"]
                )
            ),
        }
    audit = {
        "postprocessing_only": True,
        "report_sha256": digest(directory / "report.json"),
        "parity_cases": len(report["parity"]),
        "one_fewer_evaluation_with_same_linearization_and_finalizer_counts": True,
        "paired_summary_verified": True,
        "paired_acceptance": accepted,
        "closed_loop": cases,
        "scope": "Audit of these saved failed closed-loop observations; a new timing run can have different outcomes.",
        "source_sha256": {Path(__file__).name: digest(Path(__file__))},
    }
    output.write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.directory, args.output)
