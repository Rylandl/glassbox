"""Localize the recorded line-search difference without deadlines or timing."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import investigate_sqp_recovery as sqp
import jax.numpy as jnp
import numpy as np
from investigate_feedback_recovery import RecoverySolver
from investigate_feedback_suffix import checker
from investigate_fused_covariance import FusedRecoverySolver
from investigate_single_seed import load_request, request
from investigate_terminal_suffix import fingerprints

from glassbox.belief.belief import DynamicsBelief
from glassbox.control.fitted import NMPCController
from glassbox.core.synthetic import resting_state


def trace(solver, item, reference, check):
    events, arrays = [], {}
    active = {}
    linearize, evaluate, quadratic = (
        solver.linearize,
        solver.evaluate,
        sqp.quadratic_step,
    )

    def capture(kind, **values):
        prefix = f"event_{len(events)}"
        event = {"kind": kind, "arrays": {}}
        for key, value in values.items():
            name = prefix + "_" + key
            arrays[name] = np.asarray(value).copy()
            event["arrays"][key] = name
        events.append(event)
        return event

    def traced_linearize(*args):
        result = linearize(*args)
        residuals, margins = result[1][0]
        active["residuals"] = np.asarray(residuals, dtype=float)
        capture("linearize", blocks=args[0], residuals=residuals, margins=margins)
        return result

    def traced_quadratic(hessian, gradient, margins, jacobian, blocks):
        result = quadratic(hessian, gradient, margins, jacobian, blocks)
        step, multipliers, usable, success = result
        penalty = max(1.0, 1.1 * np.max(multipliers[: margins.size], initial=0.0))
        violation = np.maximum(-margins, 0).sum()
        residuals = active["residuals"]
        active.update(
            penalty=penalty,
            merit=float(residuals @ residuals) + penalty * violation,
            slope=gradient @ step - penalty * violation,
            trial=0,
        )
        capture(
            "quadratic",
            blocks=blocks,
            gradient=gradient,
            step=step,
            multipliers=multipliers,
        ).update(usable=bool(usable), success=bool(success))
        return result

    def traced_evaluate(*args):
        result = evaluate(*args)
        residuals, margins = (np.asarray(part, dtype=float) for part in result)
        cost = float(residuals @ residuals)
        merit = cost + active["penalty"] * np.maximum(-margins, 0).sum()
        alpha = 0.5 ** active["trial"]
        threshold = active["merit"] + 1e-4 * alpha * min(active["slope"], 0.0)
        capture(
            "evaluate", blocks=args[0], residuals=result[0], margins=result[1]
        ).update(
            trial=active["trial"],
            alpha=alpha,
            cost=cost,
            merit=float(merit),
            armijo_threshold=float(threshold),
            merit_minus_threshold=float(merit - threshold),
            accepted=bool(merit <= threshold),
        )
        active["trial"] += 1
        return result

    solver.linearize, solver.evaluate = traced_linearize, traced_evaluate
    sqp.quadratic_step = traced_quadratic
    try:
        outcome, description = request(solver, item, reference, check)
    finally:
        solver.linearize, solver.evaluate = linearize, evaluate
        sqp.quadratic_step = quadratic
    for key in ("predicted_commands", "predicted_states", "predicted_latent_states"):
        arrays[key] = np.asarray(getattr(outcome, key))
    return {
        "events": events,
        "counts": description["counts"],
        "status": str(outcome.status),
        "command_usable": outcome.command_usable,
        "objective": outcome.diagnostics.final_objective,
    }, arrays


def verify_sources(expected_sources):
    root = Path(__file__).resolve().parents[1]
    for name, expected in expected_sources.items():
        path = (
            Path(importlib.util.find_spec(Path(name).stem).origin)
            if name.startswith("scripts/")
            else root / name
        )
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name


def run(fixtures, records, comparison, output):
    output.mkdir(parents=True, exist_ok=False)
    prior = json.loads(comparison.read_text())
    failed = [row for row in prior["checks"] if not row["passed"]]
    assert len(failed) == 1 and failed[0]["input"] == "cold_small_39"
    provenance = fingerprints(fixtures)
    assert provenance["fixture_sha256"] == prior["fixture_sha256"]
    verify_sources(prior["source_sha256"])
    controller = NMPCController(DynamicsBelief.load(fixtures / "rich-belief.json"))
    reference = controller.hold_reference(jnp.asarray(resting_state()))
    with np.load(fixtures / "horizon-shift-states.npz") as fixture:
        previous = fixture["tick0_previous_command"].copy()
    item = load_request(records / "feedback-recovery", "cold_small", 39, previous)
    assert item["source"] == failed[0]["source"]
    check = checker(controller.plan, reference)
    content = Path(__file__).read_bytes()
    (output / Path(__file__).name).write_bytes(content)
    report = {
        "diagnostic_only": True,
        "deadline_s": None,
        "timing_recorded": False,
        "comparison_sha256": hashlib.sha256(comparison.read_bytes()).hexdigest(),
        "trace_source_sha256": hashlib.sha256(content).hexdigest(),
        "compared_sources_sha256": prior["source_sha256"],
        "fixture_sha256": provenance["fixture_sha256"],
        "input": item["source"],
        "arms": {},
    }
    all_arrays = {}
    for arm, cls in (("baseline", RecoverySolver), ("fused", FusedRecoverySolver)):
        solver = cls(controller.plan, item["seed"], work_estimates=None)
        solver.reuse_single_seed = True
        summary, arrays = trace(solver, item, reference, check)
        all_arrays[arm] = arrays
        path = output / f"{arm}.npz"
        np.savez_compressed(path, **arrays)
        summary.update(
            trace_file=path.name,
            trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            reproduces_work_counts=summary["counts"] == failed[0][arm + "_counts"],
        )
        report["arms"][arm] = summary
    paired = []
    for left, right in zip(
        report["arms"]["baseline"]["events"], report["arms"]["fused"]["events"]
    ):
        if left["kind"] != right["kind"]:
            break
        row = {"kind": left["kind"]}
        for key in left["arrays"]:
            a = all_arrays["baseline"][left["arrays"][key]]
            b = all_arrays["fused"][right["arrays"][key]]
            row[key + "_bitwise_equal"] = (
                a.dtype == b.dtype and a.tobytes() == b.tobytes()
            )
            row[key + "_maximum_absolute_difference"] = float(
                np.max(np.abs(a - b), initial=0)
            )
        if left["kind"] == "evaluate":
            row["acceptance_equal"] = left["accepted"] == right["accepted"]
        paired.append(row)
        if left.get("accepted") != right.get("accepted"):
            break
    report["paired_events_through_first_branch_change"] = paired
    (output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                arm: {key: value for key, value in summary.items() if key != "events"}
                for arm, summary in report["arms"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--records", type=Path, default=Path("docs/investigations"))
    parser.add_argument(
        "--comparison",
        type=Path,
        default=Path("docs/investigations/fused-covariance/report.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.fixtures, args.records, args.comparison, args.output)
