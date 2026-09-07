"""Check deadline-free parity of the opt-in request-local admission heuristic."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from investigate_feedback_recovery import RecoverySolver
from investigate_feedback_suffix import checker
from investigate_single_seed import load_request, request, same_tree
from investigate_sqp_recovery import DEFAULT_WORK_ESTIMATES
from investigate_terminal_suffix import fingerprints

from glassbox.belief.belief import DynamicsBelief
from glassbox.control.fitted import NMPCController
from glassbox.core.synthetic import resting_state


def run(fixtures, records, output):
    output.mkdir(parents=True, exist_ok=False)
    prior_path = records / "single-seed-reuse/report.json"
    prior = json.loads(prior_path.read_text())
    provenance = fingerprints(fixtures)
    assert provenance["fixture_sha256"] == prior["fixture_sha256"]
    controller = NMPCController(DynamicsBelief.load(fixtures / "rich-belief.json"))
    reference = controller.hold_reference(jnp.asarray(resting_state()))
    with np.load(fixtures / "horizon-shift-states.npz") as fixture:
        previous = fixture["tick0_previous_command"].copy()
        np.testing.assert_array_equal(reference.states, fixture["reference_states"])
    solver = RecoverySolver(
        controller.plan,
        jnp.repeat(jnp.asarray(previous)[None, :], 30, axis=0),
        work_estimates=DEFAULT_WORK_ESTIMATES,
    )
    solver.reuse_single_seed = True
    check = checker(controller.plan, reference)
    report = {
        "verification_only": True,
        "deadline_s": None,
        "host_timing_acceptance_gate": False,
        "runtime_performance_claim": False,
        "design": "Replay the same 26 saved requests as the single-seed comparison, once per flag setting, without deadlines. Compare complete returned arrays, cost, feasibility, status and work counts; independently check both outputs.",
        "input_report_sha256": hashlib.sha256(prior_path.read_bytes()).hexdigest(),
        "checks": [],
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
        "investigate_single_seed.py",
        "investigate_feedback_recovery.py",
        "investigate_feedback_suffix.py",
        "investigate_fast_suffix.py",
        "investigate_fast_suffix_runtime.py",
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
            records / directory, source["case"], source["tick"], previous
        )
        assert item["source"] == source
        report["pending_input"] = name
        save()
        solver.use_observed_linearization_cost = False
        before, baseline = request(solver, item, reference, check)
        solver.use_observed_linearization_cost = True
        after, enabled = request(solver, item, reference, check)
        arrays = (
            "command",
            "predicted_commands",
            "predicted_states",
            "predicted_latent_states",
        )
        row = {
            "input": name,
            "arrays_bitwise_equal": same_tree(
                tuple(getattr(before, k) for k in arrays),
                tuple(getattr(after, k) for k in arrays),
            ),
            "objective_equal": before.diagnostics.final_objective
            == after.diagnostics.final_objective,
            "feasibility_equal": before.nonlinear_feasibility
            == after.nonlinear_feasibility,
            "status_equal": before.status == after.status,
            "work_counts_equal": baseline["counts"] == enabled["counts"],
            "floor_inactive": not enabled["optimizer"][
                "observed_linearization_floor_applied"
            ],
            "both_usable": before.command_usable and after.command_usable,
            "no_deadline_assessment": before.deadline_met is None
            and after.deadline_met is None,
        }
        row["passed"] = all(value for key, value in row.items() if key != "input")
        report["checks"].append(row)
    report.pop("pending_input", None)
    report["all_passed"] = len(report["checks"]) == 26 and all(
        row["passed"] for row in report["checks"]
    )
    save()
    print(
        f"{sum(row['passed'] for row in report['checks'])}/{len(report['checks'])} deadline-free parity checks passed; no timing acceptance gate"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--records", type=Path, default=Path("docs/investigations"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.fixtures, args.records, args.output)
