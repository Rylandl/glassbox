from __future__ import annotations

import json

from glassbox.adaptation_benchmark import run_adaptation_benchmark


def test_compact_fleet_prior_adaptation_is_predictively_useful() -> None:
    report = run_adaptation_benchmark()

    assert report["semantics"] == {
        "diagnostic_only": True,
        "acceptance_gate": False,
        "synthetic": True,
        "posterior_calibration_claim": False,
        "parameter_covariance_contraction_claim": False,
        "forecast_error_used_as_generalized_loss_geometry": True,
        "physical_parameter_recovery_required": False,
        "evaluation_telemetry_is_disjoint_from_adaptation_telemetry": True,
        "runtime_envelope_uses_target_telemetry": False,
    }
    assert {item["family"] for item in report["scenarios"]} == {
        "multirotor",
        "fixedwing",
    }
    assert report["observations"] == {
        "all_updates_applied": True,
        "all_independent_predictions_improved": True,
        "all_parameter_covariances_preserved": True,
    }
    for scenario in report["scenarios"]:
        fleet = scenario["fleet_evidence"]
        adaptation = scenario["adaptation"]
        evaluation = scenario["independent_evaluation"]
        parameters = scenario["parameter_diagnostics"]
        assert 0 < fleet["empirical_rank"] < fleet["parameter_count"]
        assert fleet["predictive_error_group_count"] == fleet["member_count"]
        assert 0.0 < fleet["completion_fraction_in_natural_coordinates"] < 1.0
        assert adaptation["applied"]
        assert (
            adaptation["normalized_innovation_rms_after"]
            < (adaptation["normalized_innovation_rms_before"])
        )
        assert (
            evaluation["normalized_prediction_rms_after"]
            < (evaluation["normalized_prediction_rms_before"])
        )
        assert (
            parameters["normalized_covariance_trace_after"]
            == (parameters["normalized_covariance_trace_before"])
        )
    json.dumps(report, allow_nan=False)
