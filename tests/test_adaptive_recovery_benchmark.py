from __future__ import annotations

import json

import numpy as np
import pytest

from glassbox.adaptive_recovery_benchmark import run_adaptive_recovery_benchmark


def test_adaptive_recovery_benchmark_is_finite_and_auditable() -> None:
    report = run_adaptive_recovery_benchmark()

    assert report["artifact_type"] == (
        "glassbox_synthetic_adaptive_recovery_diagnostic"
    )
    assert report["semantics"]["diagnostic_only"]
    assert not report["semantics"]["acceptance_gate"]
    assert not report["semantics"]["flight_safety_claim"]
    assert not report["semantics"]["throw_to_recover_claim"]
    assert report["semantics"]["prewarmed_controller"]
    assert report["semantics"]["stale_predictive_error_is_not_applied_at_runtime"]
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
    assert report["observations"]["all_recovery_traces_finite"]
    assert report["observations"]["all_commands_within_bounds"]
    assert report["observations"]["all_recovery_traces_without_fallback"]
    assert report["observations"]["all_recovery_within_validity_support"] == all(
        max(
            item["maximum_actual_validity_utilization"],
            item["maximum_predicted_validity_utilization"],
        )
        <= 1.0
        for item in report["recovery"]
    )
    assert all(np.isfinite(value) for value in report["comparisons"].values())
    json.dumps(report, allow_nan=False)
