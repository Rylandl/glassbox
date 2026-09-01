from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from glassbox.adaptive_recovery_benchmark import (
    adaptive_recovery_source_fingerprint,
    normalized_adaptive_recovery_report,
    run_adaptive_recovery_benchmark,
)


def _assert_nested_close(actual, expected, path: str = "report") -> None:
    if isinstance(expected, dict):
        assert isinstance(actual, dict), path
        assert actual.keys() == expected.keys(), path
        for key in expected:
            _assert_nested_close(actual[key], expected[key], f"{path}.{key}")
        return
    if isinstance(expected, list):
        assert isinstance(actual, list), path
        assert len(actual) == len(expected), path
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            _assert_nested_close(
                actual_item,
                expected_item,
                f"{path}[{index}]",
            )
        return
    if isinstance(expected, float):
        assert actual == pytest.approx(expected, rel=1e-5, abs=1e-7), path
        return
    assert actual == expected, path


def test_adaptive_recovery_benchmark_is_finite_and_auditable() -> None:
    report = run_adaptive_recovery_benchmark()

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
    assert (
        report["implementation"]["source_sha256"]
        == adaptive_recovery_source_fingerprint()
    )
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

    recorded = json.loads(
        (Path(__file__).parents[1] / "docs/adaptive-recovery-results.json").read_text()
    )
    _assert_nested_close(
        normalized_adaptive_recovery_report(report),
        normalized_adaptive_recovery_report(recorded),
    )
