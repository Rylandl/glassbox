"""Audit saved precision-stopping artifacts; never solve or refit a model."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from investigate_feedback_recovery import score

from glassbox.belief.belief import DynamicsBelief
from glassbox.control.fitted import NMPCController


def digest(value):
    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_verified(directory, fixtures):
    report = json.loads((directory / "report.json").read_text())
    for name, expected in report["fixture_sha256"].items():
        assert file_hash(fixtures / name) == expected, name
    archived = []
    for path in sorted((directory / "executed-sources").glob("*.py")):
        assert file_hash(path) == report["source_sha256"]["scripts/" + path.name]
        archived.append(path.name)
    assert archived
    assert report["deadline_s"] is None and not report["runtime_performance_claim"]
    return report, archived


def calls(rows, key="counts"):
    return dict(sum((Counter(row[key]) for row in rows), Counter()))


def request_audit(report, prior, records):
    assert len(report["checks"]) == 26
    assert report["input_report_sha256"] == file_hash(
        records / "single-seed-reuse/report.json"
    )
    assert [row["input"] for row in report["checks"]] == list(prior["inputs"])
    failures, unchanged, stops = [], [], Counter()
    for row in report["checks"]:
        assert row["source"] == prior["inputs"][row["input"]]
        source = row["source"]
        directory = "feedback-recovery"
        if row["input"] == "recorded_tick10":
            directory = "feedback-recovery-runtime"
        elif row["input"] == "recorded_small68":
            directory = "seed-timing-history"
        original_path = records / directory / "report.json"
        assert file_hash(original_path) == source["report_sha256"]
        original = json.loads(original_path.read_text())
        case = original["cases"][source["case"]]
        assert (
            file_hash(records / directory / case["trace_file"])
            == source["trace_sha256"]
        )
        assert all(
            source[key] == case["requests"][source["tick"]][key]
            for key in (
                "state_sha256",
                "latent_sha256",
                "previous_command_sha256",
                "seed_sha256",
            )
        )
        tolerance = report["design"]["saved_request_tolerances"]
        scalar_equal = row["baseline_objective"] == row["precision_objective"]
        scalar_close = np.isclose(
            row["baseline_objective"], row["precision_objective"], **tolerance
        )
        assert scalar_equal == row["objective"]["bitwise_equal"]
        assert bool(scalar_close) == row["objective"]["within_tolerance"]
        assert (
            abs(row["baseline_objective"] - row["precision_objective"])
            == row["objective"]["maximum_absolute_difference"]
        )
        no_more = all(
            row["precision_counts"].get(k, 0) <= row["baseline_counts"].get(k, 0)
            for k in ("evaluate", "linearize", "finalize")
        )
        assert no_more == row["no_more_kernel_calls"]
        passed = (
            all(row[k] for k in ("both_usable", "status_equal", "feasibility_equal"))
            and no_more
        )
        passed &= all(
            row[k]["within_tolerance"] for k in ("returned_arrays", "objective")
        )
        if row["precision_stop"] is None:
            passed &= all(
                row[k]["bitwise_equal"] for k in ("returned_arrays", "objective")
            )
            passed &= row["baseline_counts"] == row["precision_counts"]
            unchanged.append(row["input"])
        else:
            stops[row["precision_stop_reason"]] += 1
        assert bool(passed) == row["passed"]
        if not passed:
            failures.append(
                {
                    "input": row["input"],
                    "array_max_abs_difference": row["returned_arrays"][
                        "maximum_absolute_difference"
                    ],
                    "objective_max_abs_difference": row["objective"][
                        "maximum_absolute_difference"
                    ],
                }
            )
    assert report["saved_requests_passed"] == (not failures)
    return {
        "passed": 26 - len(failures),
        "failures": failures,
        "strict_unchanged_requests": unchanged,
        "stops": dict(stops),
        "baseline_kernel_calls": calls(report["checks"], "baseline_counts"),
        "precision_kernel_calls": calls(report["checks"], "precision_counts"),
        "maximum_returned_array_difference": max(
            x["returned_arrays"]["maximum_absolute_difference"]
            for x in report["checks"]
        ),
    }


def run(records, fixtures, output):
    revised_dir, rejected_dir = (
        records / "precision-backtracking",
        records / "precision-stopping",
    )
    revised, revised_archive = read_verified(revised_dir, fixtures)
    rejected, rejected_archive = read_verified(rejected_dir, fixtures)
    prior = json.loads((records / "single-seed-reuse/report.json").read_text())
    result = {
        "postprocessing_only": True,
        "solves_or_refits": 0,
        "host_speedup_claim": False,
        "reports_sha256": {
            "revised": file_hash(revised_dir / "report.json"),
            "rejected": file_hash(rejected_dir / "report.json"),
        },
        "archives_verified": {"revised": revised_archive, "rejected": rejected_archive},
        "rejected_design": request_audit(rejected, prior, records),
        "revised_design": request_audit(revised, prior, records),
        "cases": {},
        "comparisons": {},
        "limitations": [
            "Saved-request output parity is audited from recorded comparisons: their full returned arrays were not archived.",
            "Future nonlinear forecasts and plant dynamics are not rerun. Forecast waveform identity, recorded nonlinear assessments, actual-state support and terminal scores are checked.",
            "No timing inference; operation counts describe this bounded experiment only.",
        ],
    }
    assert not rejected["cases"] and len(result["rejected_design"]["failures"]) == 2
    assert result["revised_design"]["passed"] == 26
    controller = NMPCController(DynamicsBelief.load(fixtures / "rich-belief.json"))
    plan = controller.plan
    support = jax.jit(jax.vmap(plan.model.validity_utilization))
    with np.load(fixtures / "horizon-shift-states.npz") as initial:
        initial_previous = initial["tick0_previous_command"].copy()
    traces = {}
    assert len(revised["cases"]) == 6
    for name, case in revised["cases"].items():
        path = revised_dir / case["trace_file"]
        assert file_hash(path) == case["trace_sha256"]
        with np.load(path) as archive:
            trace = {k: archive[k].copy() for k in archive.files}
        traces[name] = trace
        states, latents, seeds = [
            trace[k] for k in ("states", "latent_states", "seed_waveforms")
        ]
        rows = case["requests"]
        assert case["initial_absolute_tick"] == 0 and len(rows) == 120
        assert len(states) == len(latents) == 121 and seeds.shape == (120, 30, 4)
        assert all(np.all(np.isfinite(x)) for x in (states, latents, seeds))
        previous = initial_previous
        incoming = None
        decoded = []
        for i, row in enumerate(rows):
            assert row["absolute_tick"] == row["relative_tick"] == i
            assert (
                row["applied"] and row["command_usable"] and row["deadline_met"] is None
            )
            assert row["status"] != "converged"
            assessment = row["nonlinear_feasibility"]
            assert (
                assessment["constraint_count"] == 186
                and assessment["tolerance"] == 1e-6
            )
            assert assessment["maximum_violation"] <= assessment["tolerance"]
            expected_seed = (
                np.repeat(previous[None, :], 30, axis=0)
                if i == 0
                else np.concatenate((incoming[1:], incoming[-1:]))
            )
            np.testing.assert_array_equal(seeds[i], expected_seed)
            for key, value in (
                ("state", states[i]),
                ("latent", latents[i]),
                ("previous_command", previous),
                ("seed", seeds[i]),
            ):
                assert row[key + "_sha256"] == digest(value), (name, i, key)
            raw = trace[f"forecast_{i}"]
            waveform = raw.astype(seeds.dtype)
            np.testing.assert_array_equal(
                waveform, raw
            )  # Lossless restoration of recorded JAX dtype.
            assert waveform.shape == (30, 4)
            assert np.all(np.isfinite(waveform))
            assert np.all(waveform >= np.asarray(plan.command_minimum)) and np.all(
                waveform <= np.asarray(plan.command_maximum)
            )
            np.testing.assert_array_equal(waveform[4:24], seeds[i, 4:24])
            np.testing.assert_array_equal(waveform[0], row["command"])
            previous, incoming = waveform[0], waveform
            decoded.append(waveform)
        trace["decoded_forecasts"] = np.asarray(decoded)
        actual_support = np.asarray(support(jnp.asarray(states)))
        if case["kick"] is not None:
            actual_support = np.concatenate(
                (
                    actual_support,
                    np.asarray(support(jnp.asarray([case["kick"]["before_state"]]))),
                )
            )
            np.testing.assert_array_equal(states[60], case["kick"]["after_state"])
            assert case["kick"]["absolute_tick"] == 60
            expected_kicked = np.asarray(
                case["kick"]["before_state"], dtype=states.dtype
            ).copy()
            expected_kicked[10] += np.asarray(
                case["kick"]["delta_roll_rad_s"], dtype=states.dtype
            )
            np.testing.assert_array_equal(states[60], expected_kicked)

        maximum = float(actual_support.max())
        assert maximum <= 1 + 1e-6 and case["all_actual_states_within_support"]
        np.testing.assert_allclose(
            maximum, case["maximum_actual_support_utilization"], rtol=2e-6, atol=2e-7
        )
        recomputed = score(
            states, plan.tolerances.local_state_scale, complete=True, actual_inside=True
        )
        for k, v in recomputed.items():
            np.testing.assert_allclose(v, case[k], rtol=2e-6, atol=2e-7)
        assert (
            case["interval_limit_completed"]
            and case["applied_intervals"] == sum(row["applied"] for row in rows) == 120
        )
        assert (
            case["terminal_full_state_within_tolerances"]
            and case["terminal_attitude_rate_within_tolerances"]
        )
        counts = calls(rows)
        stops = dict(
            Counter(
                row["optimizer"]["stop_reason"]
                for row in rows
                if row["optimizer"]["precision_stop"] is not None
            )
        )
        assert (
            counts == case["total_kernel_calls"]
            and stops == case["precision_stop_counts"]
        )
        result["cases"][name] = {
            "applied_intervals": 120,
            "kernel_calls": counts,
            "precision_stops": stops,
            "support_recomputed_maximum": maximum,
            "tail_rms": recomputed["tail_normalized_tracking_rms"],
            "terminal_full_state_within_tolerances": True,
        }
    for arm in ("baseline", "precision"):
        normal, kicked = (
            traces["cold_original_" + arm],
            traces["cold_original_kick_" + arm],
        )
        case = revised["cases"]["cold_original_kick_" + arm]
        np.testing.assert_array_equal(normal["states"][:60], kicked["states"][:60])
        np.testing.assert_array_equal(
            normal["states"][60], case["kick"]["before_state"]
        )
        np.testing.assert_array_equal(
            normal["latent_states"][:61], kicked["latent_states"][:61]
        )
        np.testing.assert_array_equal(
            normal["seed_waveforms"][:61], kicked["seed_waveforms"][:61]
        )
        np.testing.assert_array_equal(
            normal["decoded_forecasts"][:60], kicked["decoded_forecasts"][:60]
        )
        assert case["matched_kick_prefix_verified"]
    for stem in ("cold_original", "cold_small", "cold_original_kick"):
        baseline, precision = traces[stem + "_baseline"], traces[stem + "_precision"]
        a, b = result["cases"][stem + "_baseline"], result["cases"][stem + "_precision"]
        result["comparisons"][stem] = {
            "precision_to_baseline_tail_rms_ratio": b["tail_rms"] / a["tail_rms"],
            "maximum_absolute_differences": {
                key: float(np.max(np.abs(baseline[key] - precision[key])))
                for key in (
                    "states",
                    "latent_states",
                    "decoded_forecasts",
                    "seed_waveforms",
                )
            },
            "kernel_call_reduction": {
                key: a["kernel_calls"].get(key, 0) - b["kernel_calls"].get(key, 0)
                for key in ("linearize", "evaluate", "finalize")
            },
        }
    result["matched_kick_prefixes_verified"] = True
    result["audit_source_sha256"] = file_hash(Path(__file__))
    result["all_checks_passed"] = True
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "rejected": result["rejected_design"],
                "revised": result["revised_design"],
                "comparisons": result["comparisons"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=Path("docs/investigations"))
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/investigations/precision-backtracking-audit.json"),
    )
    args = parser.parse_args()
    run(args.records, args.fixtures, args.output)
