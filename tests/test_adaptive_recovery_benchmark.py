from __future__ import annotations

import json
import re

import numpy as np
import pytest
from _recorded import assert_recorded_close, recorded_result

from glassbox.workflows.adaptive_recovery_benchmark import (
    normalized_adaptive_recovery_report,
    run_adaptive_recovery_benchmark,
)

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
_ADAPTIVE_RECOVERY_IGNORE = (
    # Nothing further: ``normalized_adaptive_recovery_report`` already drops
    # the environment block, per-trace wall clock, and source provenance.
)


@pytest.mark.slow
def test_adaptive_recovery_benchmark_is_finite_and_auditable() -> None:
    report = run_adaptive_recovery_benchmark()

    # Contract tier: structure, semantics, and internal consistency.
    assert report["artifact_type"] == (
        "glassbox_synthetic_adaptive_recovery_diagnostic"
    )
    assert report["format_version"] == 4
    assert report["semantics"]["diagnostic_only"]
    assert not report["semantics"]["acceptance_gate"]
    assert not report["semantics"]["flight_safety_claim"]
    assert not report["semantics"]["throw_to_recover_claim"]
    assert report["semantics"]["prewarmed_controller"]
    assert not report["semantics"]["independent_fallback_controller_included"]
    assert report["semantics"][
        "support_candidates_derived_only_from_nmpc_and_previous_command"
    ]
    assert report["semantics"]["solver_failure_returns_explicit_bounded_hold"]
    assert report["semantics"][
        "actuator_reaction_horizon_belief_support_filter_included"
    ]
    assert not report["semantics"]["independent_flight_watchdog_included"]
    assert not report["semantics"][
        "hard_prediction_horizon_validity_constraint_included"
    ]
    assert report["semantics"]["validation_actuator_context_excluded_from_evidence"]
    assert report["semantics"]["stale_predictive_error_is_not_applied_at_runtime"]
    # The source digest is provenance: the artifact records which sources
    # produced its numbers, and a source edit that leaves every number
    # unchanged does not make the artifact stale. It is checked for shape
    # here and excluded from the recorded comparison below.
    assert re.fullmatch(r"[0-9a-f]{64}", report["implementation"]["source_sha256"])
    assert report["implementation"]["source_files"]
    assert (
        report["observations"]["update_applied"]
        == report["evidence"]["adaptation"]["applied"]
    )
    assert report["evidence"]["adaptation"]["predictive_error_marked_stale"]
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
        assert item["fallback_count"] == 0
        assert item["maximum_command_bound_violation"] <= 1e-6
        assert item["prediction_horizon_s"] == pytest.approx(0.6)
        assert 0.1 <= item["support_horizon_s"] <= 0.3
        assert sum(item["support_filter_mode_counts"].values()) == 60
        assert item["support_filter_applied_count"] >= 0
        assert (
            item["inside_support_step_count"] + item["outside_support_step_count"] == 60
        )
    assert report["observations"]["all_recovery_traces_finite"]
    assert report["observations"]["all_commands_within_bounds"]
    assert report["observations"]["all_recovery_traces_without_fallback"]
    assert report["observations"]["support_filter_intervened"] == any(
        item["support_filter_applied_count"] > 0 for item in report["recovery"]
    )
    assert report["observations"][
        "all_inside_support_steps_found_supported_commands"
    ] == all(
        item["inside_support_best_effort_count"] == 0 for item in report["recovery"]
    )
    outside_exercised = any(
        item["outside_support_step_count"] > 0 for item in report["recovery"]
    )
    assert (
        report["observations"]["outside_support_progress_condition_exercised"]
        is outside_exercised
    )
    assert report["observations"][
        "all_outside_support_steps_found_progress_commands"
    ] == (
        all(
            item["outside_support_best_effort_count"] == 0
            for item in report["recovery"]
        )
        if outside_exercised
        else None
    )
    assert report["observations"]["all_actual_recovery_within_validity_support"] == all(
        item["maximum_actual_validity_utilization"] <= 1.0
        for item in report["recovery"]
    )
    assert report["observations"][
        "all_next_step_robust_predictions_within_validity_support"
    ] == all(
        item["maximum_next_step_robust_validity_utilization"] <= 1.0
        for item in report["recovery"]
    )
    assert report["observations"][
        "all_support_horizon_projections_within_validity_support"
    ] == all(
        item["maximum_support_horizon_robust_validity_utilization"] <= 1.0
        for item in report["recovery"]
    )
    assert report["observations"][
        "all_full_nmpc_predictions_within_validity_support"
    ] == all(
        item["maximum_predicted_validity_utilization"] <= 1.0
        for item in report["recovery"]
    )
    assert all(np.isfinite(value) for value in report["comparisons"].values())
    json.dumps(report, allow_nan=False)

    # Recorded tier.
    assert_recorded_close(
        normalized_adaptive_recovery_report(report),
        normalized_adaptive_recovery_report(
            recorded_result("adaptive-recovery-results.json")
        ),
        tolerances=_ADAPTIVE_RECOVERY_TOLERANCES,
        exact=_ADAPTIVE_RECOVERY_EXACT,
        ignore=_ADAPTIVE_RECOVERY_IGNORE,
    )
