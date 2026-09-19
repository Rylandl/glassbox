"""Prospective capability policy guards, without fitting any candidate."""

import copy
from pathlib import Path

import numpy as np
import pytest

from glassbox.experimental.public_mean_scoring import (
    capability_decision,
    forecast_scores,
    probe_scores,
    specification,
    verify_forecast_scores,
)

ROOT = Path(__file__).resolve().parents[1]


def test_candidate_normalization_cannot_change_required_score():
    target = np.zeros((3, 5, 2))
    prediction = np.broadcast_to([2.0, 8.0], target.shape)
    reference = np.array([2.0, 4.0])
    first = forecast_scores(prediction, target, reference, reference, ["a", "a", "b"])
    inflated = forecast_scores(
        prediction, target, reference, reference * 1000, ["a", "a", "b"]
    )
    assert first["fixed_reference"] == inflated["fixed_reference"]
    np.testing.assert_allclose(first["fixed_reference"]["horizon_rmse"], np.sqrt(2.5))
    assert inflated["legacy_candidate_scaled"]["overall_rmse"] < 0.002
    np.testing.assert_array_equal(
        first["fixed_reference"]["channel_rmse"], np.tile([2, 8], (5, 1))
    )
    forged = copy.deepcopy(inflated)
    forged["fixed_reference"] = forged["legacy_candidate_scaled"]
    with pytest.raises(ValueError, match="fixed-reference reduction"):
        verify_forecast_scores(
            forged, prediction, target, reference, reference * 1000, ["a", "a", "b"]
        )


def test_probe_keeps_physical_cap_when_display_units_change():
    truth = np.zeros((2, 5, 1))
    prediction = np.full_like(truth, 0.05)
    scores = probe_scores(prediction, truth, [0.2])
    assert scores["physical_first_step_pass"]
    assert scores["physical_first_step_limit"] == 0.05
    assert scores["fixed_reference_first_step_limit"] == 0.25
    assert scores["fixed_reference_rmse"][0] == 0.25
    assert not probe_scores(prediction + 1e-8, truth, [0.2])["physical_first_step_pass"]


def passing_roster():
    cases, ledger = specification(ROOT)
    rows = [
        {
            "case": case["name"],
            "regime": regime,
            "scores": {
                "reference_state_scale": ledger["cases"][case["name"]][
                    "reference_state_scale"
                ],
                "fixed_reference": {
                    "horizon_rmse": [case["caps"][regime]] * 5,
                    "overall_rmse": case["caps"][regime],
                },
            },
        }
        for case in cases
        for regime in case["regimes"]
    ]
    probes = [
        {"seed": seed, "scores": {"physical_rmse": [0.05] * 5}}
        for seed in (101, 202, 303)
    ]
    return [case["name"] for case in cases], rows, probes


def test_decision_rejects_candidate_units_mislabeled_as_fixed_reference():
    fits, rows, probes = passing_roster()
    rows[0]["scores"]["reference_state_scale"] = [
        1000 * v for v in rows[0]["scores"]["reference_state_scale"]
    ]
    with pytest.raises(ValueError, match="denominator"):
        capability_decision(ROOT, fits, rows, probes)


def test_inclusive_caps_and_local_failure_are_applied_to_all_255_cells():
    fits, rows, probes = passing_roster()
    result = capability_decision(ROOT, fits, rows, probes)
    assert result["fixed_reference_absolute_capability_pass"]
    assert len(result["horizon_checks"]) == 255
    rows[-1]["scores"]["fixed_reference"]["horizon_rmse"][-1] += 1e-8
    assert not capability_decision(ROOT, fits, rows, probes)[
        "fixed_reference_absolute_capability_pass"
    ]


@pytest.mark.parametrize("count", [0, 1, 4, 6])
def test_truncated_or_extended_horizon_roster_cannot_pass(count):
    fits, rows, probes = passing_roster()
    rows[0]["scores"]["fixed_reference"]["horizon_rmse"] = [0.0] * count
    with pytest.raises(ValueError, match="exactly five"):
        capability_decision(ROOT, fits, rows, probes)


@pytest.mark.parametrize(
    "mutation", ["case", "regime", "duplicate_regime", "probe", "nonfinite", "negative"]
)
def test_missing_duplicate_or_invalid_evidence_cannot_pass(mutation):
    fits, rows, probes = passing_roster()
    if mutation == "case":
        fits.pop()
    elif mutation == "regime":
        rows.pop()
    elif mutation == "duplicate_regime":
        rows[-1] = copy.deepcopy(rows[0])
    elif mutation == "probe":
        probes[-1] = copy.deepcopy(probes[0])
    elif mutation == "nonfinite":
        rows[0]["scores"]["fixed_reference"]["horizon_rmse"][0] = float("nan")
    else:
        rows[0]["scores"]["fixed_reference"]["overall_rmse"] = -1.0
    with pytest.raises(ValueError):
        capability_decision(ROOT, fits, rows, probes)
