from __future__ import annotations

import json
import re

import numpy as np
import pytest
from _recorded import assert_recorded_close, recorded_result

from glassbox.workflows.benchmarks.recovery import run_adaptive_recovery_benchmark
from glassbox.workflows.record_results import ADAPTIVE_RECOVERY_VOLATILE

# Recorded-tier policy for docs/results/adaptive-recovery-results.json.
_ADAPTIVE_RECOVERY_TOLERANCES = {
    # One offline JAX rollout with no simulator in the loop: byte-stable
    # across the refactors so far, so 1e-5 relative (with a 1e-7 absolute
    # floor for near-zero entries) leaves headroom without accepting drift.
    "*": (1e-5, 1e-7),
}
_ADAPTIVE_RECOVERY_EXACT = (
    # Seeds, durations, the initial state, and offline functions of them.
    "configuration.*",
)
# The manifest declares which paths vary with the host and the source tree,
# so ``record-results --check`` and this test exclude exactly the same ones.
_ADAPTIVE_RECOVERY_IGNORE = ADAPTIVE_RECOVERY_VOLATILE


@pytest.mark.slow
def test_adaptive_recovery_benchmark_is_finite_and_auditable() -> None:
    report = run_adaptive_recovery_benchmark()

    # Contract tier: structure, semantics, and internal consistency.
    assert report["artifact_type"] == (
        "glassbox_synthetic_adaptive_recovery_diagnostic"
    )
    assert report["format_version"] == 6
    assert report["semantics"]["diagnostic_only"]
    assert not report["semantics"]["acceptance_gate"]
    assert not report["semantics"]["flight_safety_claim"]
    assert not report["semantics"]["throw_to_recover_claim"]
    assert report["semantics"]["prewarmed_controller"]
    assert not report["semantics"]["independent_fallback_controller_included"]
    assert report["semantics"]["solver_failure_returns_explicit_bounded_hold"]
    assert report["semantics"]["predicted_spread_charged_in_the_tracking_cost"]
    assert report["semantics"]["robust_validity_charged_in_the_objective"]
    assert not report["semantics"]["independent_flight_watchdog_included"]
    assert not report["semantics"][
        "hard_prediction_horizon_validity_constraint_included"
    ]
    # The update is one recursive absorb, and its covariance really moves:
    # that is the whole point of retargeting this diagnostic.
    assert report["semantics"]["update_is_a_recursive_information_absorb"]
    assert report["semantics"]["nonlinear_update_backtracking"]
    assert report["semantics"]["unresolved_parameter_planning_explicitly_allowed"]
    assert not report["semantics"]["update_proposal_and_validation_split"]
    assert not report["semantics"]["update_improvement_margin"]
    assert report["semantics"]["parameter_covariance_updated_by_the_update"]
    assert not report["semantics"]["information_discounted_or_forgotten"]
    assert not report["semantics"]["held_out_forecast_bias_applied_at_runtime"]
    # The source digest is provenance: the artifact records which sources
    # produced its numbers, and a source edit that leaves every number
    # unchanged does not make the artifact stale. It is checked for shape
    # here and excluded from the recorded comparison below.
    assert re.fullmatch(r"[0-9a-f]{64}", report["implementation"]["source_sha256"])
    assert report["implementation"]["source_files"]
    assert (
        report["observations"]["update_absorbed"]
        == report["evidence"]["adaptation"]["absorbed"]
    )
    assert report["evidence"]["adaptation"]["absorbed"]
    assert report["evidence"]["adaptation"]["information_gain_nats"] > 0.0
    fleet = report["evidence"]["fleet"]
    assert fleet["posterior_resolved_rank"] > fleet["seed_resolved_rank"]
    assert report["observations"]["posterior_resolves_more_than_the_seed"]
    assert report["observations"]["independent_prediction_improved"] == (
        report["evidence"]["independent_prediction"]["normalized_rms_after"]
        < report["evidence"]["independent_prediction"]["normalized_rms_before"]
    )
    assert {item["condition"] for item in report["recovery"]} == {
        "stale_belief",
        "adapted_belief",
        "adapted_mean_point",
        "oracle_mean_point",
    }
    for item in report["recovery"]:
        assert item["finite"]
        assert sum(item["solve_status_counts"].values()) == 60
        assert item["unresolved_parameters_allowed"]
        assert item["fallback_count"] == 0
        assert item["maximum_command_bound_violation"] <= 1e-6
        assert item["prediction_horizon_s"] == pytest.approx(0.6)
        assert (
            item["inside_support_step_count"] + item["outside_support_step_count"] == 60
        )
    assert report["observations"]["all_recovery_traces_finite"]
    assert report["observations"]["all_commands_within_bounds"]
    assert report["observations"]["all_recovery_traces_without_fallback"]
    assert report["observations"]["any_solve_started_outside_validity_support"] is any(
        item["outside_support_step_count"] > 0 for item in report["recovery"]
    )
    assert report["observations"]["all_actual_recovery_within_validity_support"] == all(
        item["maximum_actual_validity_utilization"] <= 1.0
        for item in report["recovery"]
    )
    assert report["observations"][
        "all_full_nmpc_predictions_within_validity_support"
    ] == all(
        item["maximum_predicted_validity_utilization"] is not None
        and item["maximum_predicted_validity_utilization"] <= 1.0
        for item in report["recovery"]
    )
    assert all(np.isfinite(value) for value in report["comparisons"].values())
    json.dumps(report, allow_nan=False)

    # Recorded tier.
    assert_recorded_close(
        report,
        recorded_result("adaptive-recovery-results.json"),
        tolerances=_ADAPTIVE_RECOVERY_TOLERANCES,
        exact=_ADAPTIVE_RECOVERY_EXACT,
        ignore=_ADAPTIVE_RECOVERY_IGNORE,
    )
