"""Audit saved streaming trials without fitting, learning, or running control."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from glassbox.core.metrics import state_rmse_metrics


def audit_trial(directory, report, gate):
    events = [
        json.loads(line)
        for line in (directory / "events.jsonl").read_text().splitlines()
    ]
    worker = report["worker"]
    assert report["failure"] is None and worker["error"] is None
    assert report["completed_intervals"] == report["requested_intervals"]
    assert worker["peak_queue_blocks"] <= 2
    assert worker["retained_blocks"] <= 2 and worker["retained_revisions"] <= 9
    assert not worker["pending_offer"]
    assert (
        worker["submitted_blocks"]
        == worker["processed_blocks"] + worker["dropped_blocks"]
    )
    blocks, offers, acknowledgements = {}, {}, []
    cursor = skipped = processed = 0
    prior_candidate = None
    for event in events:
        kind = event["kind"]
        if kind in {"gap", "block"}:
            assert event["start_interval"] == cursor
            cursor = event["stop_interval"]
        if kind == "gap":
            skipped += event["stop_interval"] - event["start_interval"]
        elif kind == "block":
            assert event["stop_interval"] - event["start_interval"] == 4
            assert event["command_history_steps"] == 20
            assert event["update"]["window_count"] <= 4
            candidate = event["candidate_score"]["revision"]
            if prior_candidate is not None:
                assert candidate == prior_candidate
            prior_candidate = event["candidate_after"]
            artifact = json.loads((directory / event["candidate_artifact"]).read_text())
            assert (
                artifact["provenance"]["update_count"]
                == prior_candidate["update_count"]
            )
            assert prior_candidate["update_count"] == candidate["update_count"] + int(
                event["update"]["absorbed"]
            )
            for score in (event["active_score"], event["candidate_score"]):
                assert score["finite_state_forecast"]
                assert not score["parameter_covariance_computed"]
            blocks[event["stop_interval"]] = event
            processed += 1
        elif kind == "offer":
            block = blocks[event["scored_stop_interval"]]
            assert event["revision"] == block["candidate_score"]["revision"]
            assert (
                event["revision"]["revision_id"]
                != block["candidate_after"]["revision_id"]
            )
            assert block["candidate_score"]["maximum_validity_utilization"] <= 1
            old = np.array(
                [
                    block["active_score"]["rmse"][key] / scale
                    for key, scale in gate["normalized_metric_scales"].items()
                ]
            )
            new = np.array(
                [
                    block["candidate_score"]["rmse"][key] / scale
                    for key, scale in gate["normalized_metric_scales"].items()
                ]
            )
            assert np.linalg.norm(new) <= (
                1 - gate["minimum_relative_prediction_gain"]
            ) * np.linalg.norm(old)
            assert np.all(new <= (1 + gate["maximum_headline_regression"]) * old + 1e-6)
            offers[event["revision"]["revision_id"]] = event
        elif kind == "adoption":
            assert event["adopted_revision_id"] in offers
            acknowledgements.append(event)
        else:
            assert kind == "offer_rejected", kind
    assert processed == worker["processed_blocks"]
    assert skipped == worker["skipped_intervals"]
    assert cursor == skipped + processed * 4
    assert (
        cursor + report["partial_intervals_at_shutdown"]
        == report["completed_intervals"]
    )
    assert worker["dropped_intervals"] == worker["dropped_blocks"] * 4
    assert skipped == worker["dropped_intervals"] + report["buffer_discarded_intervals"]
    assert prior_candidate == worker["final_candidate"]
    applied = report["applied_adoptions"]
    assert [item["revision_id"] for item in applied] == [
        item["adopted_revision_id"] for item in acknowledgements
    ]
    with np.load(directory / "tracking.npz", allow_pickle=False) as trace:
        n = report["completed_intervals"]
        assert trace["states"].shape == trace["reference_states"].shape == (n + 1, 13)
        for key in trace.files:
            if key != "revision_ids":
                assert np.isfinite(trace[key]).all(), key
        np.testing.assert_allclose(np.diff(trace["time_s"]), 0.05, rtol=0, atol=1e-12)
        metrics = state_rmse_metrics(trace["states"][1:], trace["reference_states"][1:])
        for key, value in metrics.items():
            np.testing.assert_allclose(
                value, report["tracking_rmse"][key], rtol=1e-12, atol=1e-12
            )
        assert np.sum(trace["tick_times_s"] > 0.05) == report["deadline_misses"]
        assert np.sum(trace["solve_times_s"] > 0.05) == report["solve_deadline_misses"]
        initial = next(iter(blocks.values()))["active_score"]["revision"]["revision_id"]
        expected = np.full(n, initial, dtype=trace["revision_ids"].dtype)
        active = initial
        for adoption, ack in zip(applied, acknowledgements):
            assert ack["previous_revision_id"] == active
            active = adoption["revision_id"]
            assert (
                offers[active]["scored_stop_interval"]
                == adoption["scored_stop_interval"]
            )
            assert (
                0
                <= adoption["interval"] - adoption["scored_stop_interval"]
                <= gate["maximum_score_age_intervals"]
            )
            expected[adoption["interval"] :] = active
        np.testing.assert_array_equal(trace["revision_ids"], expected)
        assert active == worker["acknowledged_active"]["revision_id"]
        artifact = json.loads((directory / "belief-0000.json").read_text())
        channels = artifact["nominal_model"]["input_spec"]["channels"]
        bounds = [
            (c["minimum"], c["maximum"]) for c in channels if c["kind"] == "control"
        ]
        commands = trace["commands"]
        assert commands.shape == (n, len(bounds))
        for column, (lower, upper) in enumerate(bounds):
            assert np.all(commands[:, column] >= lower)
            assert np.all(commands[:, column] <= upper)
        assert report["maximum_command_bound_violation"] == 0
    return {
        "trial": directory.name,
        "intervals": n,
        "blocks": processed,
        "skipped_intervals": skipped,
        "applied_adoptions": len(applied),
        "passed": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory
    comparison = json.loads((root / "comparison.json").read_text())
    trials = []
    for platform in comparison["comparisons"]:
        name = f"{platform['family']}-{platform['variant']}"
        for mode, report in platform["trials"].items():
            directory = root / name / mode
            assert json.loads((directory / "summary.json").read_text()) == report
            result = audit_trial(directory, report, comparison["gate"])
            trials.append({"platform": name, **result})
    manifest_path = root / "source/manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for path, expected_hash in manifest.items():
        assert (
            hashlib.sha256((root / "source" / path).read_bytes()).hexdigest()
            == expected_hash
        )
    result = {
        "scope": "saved-array metrics, bounds, interval accounting, gated scored-revision handoffs, artifact counts and source hashes; no numerical rerun",
        "trials": trials,
        "verified_source_files": len(manifest),
        "comparison_sha256": hashlib.sha256(
            (root / "comparison.json").read_bytes()
        ).hexdigest(),
        "audit_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "passed": True,
    }
    (root / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
