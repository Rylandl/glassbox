"""Postprocess seed timing and GC records without fitting or running a solver."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def spans(events, name):
    pending, result = [], []
    for label, *stamp in events:
        if label == name + ".start":
            pending.append(np.asarray(stamp))
        elif label == name + ".end":
            result.append(np.asarray(stamp) - pending.pop())
    return result


def distribution(values):
    if not values:
        return None
    return {"median": float(np.median(values)), "max": float(np.max(values))}


def seed_materialization(trace):
    parts = spans(trace["events"], "seed.materialize")
    if len(parts) != 6:
        return None
    return {
        "evaluation_s": sum(part[0] for part in parts[:2]),
        "linearization_s": sum(part[0] for part in parts[2:]),
    }


def gc_summary(traces):
    generations, durations, clipped_durations = [], [], []
    for trace in traces:
        pending = {}
        left, right = trace["caller_interval_s"]
        for phase, wall, _process, _thread, info in trace["gc_events"]:
            generation = info["generation"]
            if phase == "start":
                pending[generation] = wall
            elif generation in pending:
                start = pending.pop(generation)
                overlap = max(0.0, min(wall, right) - max(start, left))
                if overlap:
                    generations.append(generation)
                    durations.append(wall - start)
                    clipped_durations.append(overlap)
    return {
        "collections_overlapping_caller_by_generation": dict(Counter(generations)),
        "collection_duration_s": distribution(durations),
        "overlap_duration_s": distribution(clipped_durations),
    }


def archived_sources(directory, report):
    paths = sorted((directory / "executed-sources").glob("*.py"))
    if (directory / "executed-source.py").exists():
        paths.append(directory / "executed-source.py")
    verified = []
    for path in paths:
        name = (
            "investigate_feedback_recovery.py"
            if path.name == "executed-source.py"
            else path.name
        )
        assert digest(path) == report["source_sha256"]["scripts/" + name]
        verified.append(name)
    return verified


def run(replay, history, prior, output):
    fixed = json.loads((replay / "report.json").read_text())
    varied = json.loads((history / "report.json").read_text())
    old = json.loads((prior / "report.json").read_text())
    assert fixed["input_report_sha256"] == digest(prior / "report.json")
    assert fixed["forecasts_sha256"] == digest(replay / "forecasts.npz")
    assert fixed["fixture_sha256"] == varied["fixture_sha256"] == old["fixture_sha256"]
    rows = fixed["requests"]
    expected = fixed["design"]["order_per_block"] * fixed["design"]["blocks"]
    assert [row["arm"] for row in rows] == expected
    forecast_hashes = []
    with np.load(replay / "forecasts.npz") as forecasts:
        for row in rows:
            assert not row["applied"]
            for key, value in fixed["input_hashes"].items():
                assert row[key] == value
            if row["accepted"]:
                assert row["command_usable"] and row["deadline_met"]
                assert row["caller_budget_fraction"] < 1
                feasibility = row["nonlinear_feasibility"]
                assert feasibility["maximum_violation"] <= feasibility["tolerance"]
                np.testing.assert_array_equal(
                    forecasts[f"forecast_{row['index']}"][0], row["command"]
                )
                forecast_hashes.append(
                    hashlib.sha256(
                        forecasts[f"forecast_{row['index']}"].tobytes()
                    ).hexdigest()
                )
    evaluation, linearization = [], []
    for row in rows:
        if row["arm"] != "traced":
            continue
        parts = spans(row["events"], "seed.materialize")
        # A completed one-waveform seed has two evaluation leaves followed by
        # four derivative/value leaves. Failed partial seeds are not imputed.
        if len(parts) == 6:
            evaluation.append(sum(part[0] for part in parts[:2]))
            linearization.append(sum(part[0] for part in parts[2:]))
    keys = ("state_sha256", "latent_sha256", "previous_command_sha256", "seed_sha256")
    cases = {}
    for name, case in varied["cases"].items():
        trace_path = history / case["trace_file"]
        assert digest(trace_path) == case["trace_sha256"]
        with np.load(trace_path) as trace:
            assert len(trace["states"]) == case["applied_intervals"] + 1
        current, original = case["requests"], old["cases"][name]["requests"]
        matching = []
        for a, b in zip(current, original):
            if any(a[key] != b[key] for key in keys):
                break
            matching.append(a["absolute_tick"])
        failed = [row for row in current if not row["applied"]]
        for row in failed:
            assert (
                not row["command_usable"]
                or row["caller_elapsed_s"] >= row["deadline_s"]
            )
        if failed:
            assert case["tail_normalized_tracking_rms"] is None
        cases[name] = {
            "applied_intervals": case["applied_intervals"],
            "complete": case["interval_limit_completed"],
            "tail_normalized_tracking_rms": case["tail_normalized_tracking_rms"],
            "terminal_full_state_within_tolerances": case[
                "terminal_full_state_within_tolerances"
            ],
            "maximum_caller_budget_fraction": max(
                row["caller_elapsed_s"] / row["deadline_s"] for row in current
            ),
            "status_counts": dict(Counter(row["status"] for row in current)),
            "optimizer_candidate_sources": dict(
                Counter(
                    row["optimizer"].get("output_source", "no_returned_plan")
                    for row in current
                    if row["optimizer"] is not None
                )
            ),
            "prior_matching_request_prefix_ticks": matching,
            "first_input_difference_tick": current[len(matching)]["absolute_tick"]
            if len(matching) < min(len(current), len(original))
            else None,
            "gc": gc_summary([row["seed_trace"] for row in current]),
            "accepted_warm_seed_materialization_s": {
                key: distribution(
                    [
                        parts[key]
                        for row in current
                        if row["applied"]
                        and not row["startup"]
                        and (parts := seed_materialization(row["seed_trace"]))
                        is not None
                    ]
                )
                for key in ("evaluation_s", "linearization_s")
            },
            "failed_requests": [
                {
                    "tick": row["absolute_tick"],
                    "message": row["message"],
                    "budget_fraction": row["caller_elapsed_s"] / row["deadline_s"],
                    "nonlinear_feasibility": row["nonlinear_feasibility"],
                    "phase_totals_s": row["seed_trace"]["phase_totals_s"],
                    "seed_materialization": seed_materialization(row["seed_trace"]),
                    "gc_events_in_caller": row["seed_trace"]["gc_events_in_caller"],
                }
                for row in failed
            ],
        }
    result = {
        "postprocessing_only": True,
        "source_reports_sha256": {
            "replay": digest(replay / "report.json"),
            "history": digest(history / "report.json"),
            "prior": digest(prior / "report.json"),
        },
        "archived_sources_verified": {
            "replay": archived_sources(replay, fixed),
            "history": archived_sources(history, varied),
        },
        "replay": {
            "accepted_forecast_hash_counts": dict(Counter(forecast_hashes)),
            "gc": gc_summary(rows),
            "completed_seed_materialization_samples": len(evaluation),
            "evaluation_materialization_s": distribution(evaluation),
            "linearization_materialization_s": distribution(linearization),
        },
        "history": cases,
        "source_sha256": {Path(__file__).name: digest(Path(__file__))},
    }
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.replay, args.history, args.prior, args.output)
