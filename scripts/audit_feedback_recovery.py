"""Audit saved recovery scores, matched disturbance inputs, and deadline failure."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from investigate_feedback_recovery import score

from glassbox.belief.belief import DynamicsBelief
from glassbox.control.fitted import NMPCController


def run(formulation, runtime, belief_path, output):
    reports = {
        name: json.loads((directory / "report.json").read_text())
        for name, directory in (("formulation", formulation), ("runtime", runtime))
    }
    scale = np.asarray(
        NMPCController(DynamicsBelief.load(belief_path)).tolerances.local_state_scale
    )
    arrays = {}
    scored = {}
    for phase, directory in (("formulation", formulation), ("runtime", runtime)):
        scored[phase] = {}
        for name, case in reports[phase]["cases"].items():
            trace_path = directory / case["trace_file"]
            assert (
                hashlib.sha256(trace_path.read_bytes()).hexdigest()
                == case["trace_sha256"]
            )
            trace = dict(np.load(trace_path))
            arrays[phase, name] = trace
            assert len(trace["states"]) - 1 == case["applied_intervals"]
            rescored = score(
                trace["states"],
                scale,
                complete=case["interval_limit_completed"],
                actual_inside=case["all_actual_states_within_support"],
            )
            for key in (
                "tail_normalized_tracking_rms",
                "terminal_attitude_rate_within_tolerances",
                "terminal_full_state_within_tolerances",
            ):
                assert rescored[key] == case[key]
            scored[phase][name] = rescored
    nominal = reports["formulation"]["cases"]["cold_original"]
    kicked = reports["formulation"]["cases"]["cold_original_kick"]
    tick = kicked["kick"]["absolute_tick"]
    left, right = nominal["requests"][tick], kicked["requests"][tick]
    for key in ("latent_sha256", "previous_command_sha256", "seed_sha256"):
        assert left[key] == right[key]
    difference = np.asarray(right["state"]) - np.asarray(left["state"])
    np.testing.assert_array_equal(difference[np.arange(13) != 10], np.zeros(12))
    np.testing.assert_allclose(
        difference[10], kicked["kick"]["delta_roll_rad_s"], atol=1e-7, rtol=0
    )
    command_difference = np.asarray(right["command"]) - np.asarray(left["command"])
    failed_case = reports["runtime"]["cases"]["cold_original_kick"]
    failed = failed_case["requests"][-1]
    succeeded = reports["runtime"]["cases"]["cold_original"]["requests"][
        failed["absolute_tick"]
    ]
    matched = {
        key: failed[key] == succeeded[key]
        for key in (
            "state_sha256",
            "latent_sha256",
            "previous_command_sha256",
            "seed_sha256",
        )
    }
    assert all(matched.values())
    assert succeeded["applied"] and not failed["applied"]
    assert failed_case["kick"] is None
    a, b = arrays["runtime", "cold_original"], arrays["runtime", "cold_original_kick"]
    for key in ("states", "latent_states", "seed_waveforms"):
        np.testing.assert_array_equal(a[key][: len(b[key])], b[key])
    result = {
        "postprocessing_only": True,
        "local_state_tolerance_scale": scale.tolist(),
        "rescored": scored,
        "kick_comparison": {
            "absolute_tick": tick,
            "changed_initial_state_index": 10,
            "seed_latent_previous_identical": True,
            "immediate_command_difference": command_difference.tolist(),
            "immediate_command_difference_inf": float(
                np.max(np.abs(command_difference))
            ),
        },
        "runtime_failure_comparison": {
            "absolute_tick": failed["absolute_tick"],
            "before_scheduled_kick": True,
            "same_request_inputs": matched,
            "identical_prior_state_actuator_and_seed_histories": True,
            "passed_caller_budget_fraction": succeeded["caller_elapsed_s"]
            / succeeded["deadline_s"],
            "failed_caller_budget_fraction": failed["caller_elapsed_s"]
            / failed["deadline_s"],
            "failed_message": failed["message"],
            "interpretation": "The same numerical request passed earlier. The failure preceded the disturbance; its host/kernel scheduling cause is not identified by these recordings.",
        },
        "source_reports_sha256": {
            phase: hashlib.sha256((directory / "report.json").read_bytes()).hexdigest()
            for phase, directory in (("formulation", formulation), ("runtime", runtime))
        },
        "belief_sha256": hashlib.sha256(belief_path.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["runtime_failure_comparison"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formulation", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--belief", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.formulation, args.runtime, args.belief, args.output)
