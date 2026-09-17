"""The retrospective readout must preserve the question its statistic answers."""

import io
import json
from pathlib import Path

import numpy as np
import pytest

from glassbox.experimental.qualification_references import (
    prefix_vector_p95,
    reference_report,
)


def test_prefix_vector_p95_preserves_short_recording_and_intermediate_error():
    target = np.zeros((101, 2, 6))
    prediction = target.copy()
    prediction[-1, 0, :2] = [3, 4]
    ids = np.array(["long"] * 100 + ["short"])
    measured = prefix_vector_p95(prediction, target, ids)
    velocity = measured["velocity_m_s"]
    assert velocity["worst_recording_prefix_vector_p95"] == 5
    assert velocity["worst_recordings"] == ["short"]
    assert velocity["recordings"]["long"]["prefix_vector_p95"] == 0
    assert velocity["recordings"]["short"]["windows"] == 1
    assert measured["body_rate_rad_s"]["worst_recording_prefix_vector_p95"] == 0
    # Both alternate questions hide the error this statistic is meant to expose.
    assert np.sqrt(np.mean(prediction[:, -1, :3] ** 2)) == 0
    pooled = np.linalg.norm(prediction[:, :, :3], axis=-1).max(axis=1)
    assert np.sort(pooled)[int(np.ceil(0.95 * len(pooled))) - 1] == 0


@pytest.mark.parametrize(
    ("values", "expected_rank", "expected"),
    [([0] * 19 + [100], 19, 0), ([0] * 19 + [2, 9], 20, 2)],
)
def test_percentile_uses_nearest_rank_without_interpolation(
    values, expected_rank, expected
):
    target = np.zeros((len(values), 1, 6))
    prediction = target.copy()
    prediction[:, 0, 3] = values
    result = prefix_vector_p95(prediction, target, np.array(["one"] * len(values)))
    rate = result["body_rate_rad_s"]["recordings"]["one"]
    assert rate == {
        "windows": len(values),
        "nearest_rank": expected_rank,
        "prefix_vector_p95": expected,
    }


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_forecasts_rejected(invalid):
    data = np.zeros((1, 1, 6))
    data[0, 0, 5] = invalid
    with pytest.raises(ValueError, match="finite"):
        prefix_vector_p95(data, np.zeros_like(data), ["a"])


@pytest.mark.parametrize(
    ("shape", "ids"),
    [((0, 1, 6), []), ((1, 0, 6), ["a"]), ((1, 1, 5), ["a"]), ((2, 1, 6), ["a"])],
)
def test_incomplete_forecasts_rejected(shape, ids):
    data = np.zeros(shape)
    with pytest.raises(ValueError, match="matching nonempty"):
        prefix_vector_p95(data, data, ids)


@pytest.mark.parametrize("ids", [[""], [b""], [1], [["a"]]])
def test_invalid_recording_identities_rejected(ids):
    data = np.zeros((1, 1, 6))
    with pytest.raises(ValueError):
        prefix_vector_p95(data, data, ids)


def _bytes_json(value):
    return json.dumps(value, allow_nan=False).encode()


@pytest.fixture
def report_inputs():
    """Tiny snapshots with a passing legacy gate and a failing vector percentile."""
    decision = {
        "manifest": "test",
        "accepted": True,
        "rule_met": False,
        "reference_compared": True,
        "reference_sha256": "test-reference",
        "gating_rule_breaches": 0,
        "tracking_rmse": {
            "0": {
                "position_rmse_m": {"reference_meets_rule": False},
                "attitude_rmse_deg": {"reference_meets_rule": False},
            }
        },
    }
    platform_manifest = {
        "corpora": [
            {
                "name": "test",
                "structured": {"arms": ["structured"]},
                "allowance": {"velocity_rmse_m_s": 0.2, "body_rate_rmse_rad_s": 0.2},
            }
        ]
    }
    inputs = {
        "platform/manifest.json": _bytes_json(platform_manifest),
        "platform/decision.json": _bytes_json(decision),
        "platform/results.json": _bytes_json(
            [
                {
                    "corpus": "test",
                    "evaluation_rows": 1,
                    "horizon_steps": 2,
                    "horizon_s": 0.1,
                    "generic_fingerprint": "test-model",
                    "recordings": {"held_out": ["record"]},
                    "generic": {
                        "final_step": {
                            "velocity_rmse_m_s": 0.0,
                            "body_rate_rmse_rad_s": 0.0,
                        }
                    },
                }
            ]
        ),
        "control/manifest.json": _bytes_json(
            {
                "trial": {"repetitions": 1},
                "metrics": {"pass_criterion": {"tolerance_m": 0.5}},
            }
        ),
        "control/decision.json": _bytes_json(decision),
        "control/calibration.json": _bytes_json({"generic_fingerprint": "test-model"}),
        "control/reference.json": _bytes_json(
            {
                "manifest": "test",
                "manifest_sha256": "test-hash",
                "source": "historical generic",
                "environment": {},
                "generic_fingerprint": "test-model",
            }
        ),
    }
    arrays = {
        "targets": np.zeros((1, 2, 6)),
        "generic_prediction": np.zeros((1, 2, 6)),
        "hold_prediction": np.zeros((1, 2, 6)),
        "structured_structured_prediction": np.zeros((1, 2, 6)),
        "recording_ids": np.array(["record"]),
    }
    arrays["generic_prediction"][0, 0, 0] = 1.0
    arrays["structured_structured_prediction"][0, 0, 0] = 0.5
    stream = io.BytesIO()
    np.savez(stream, **arrays)
    inputs["platform/test/evaluation.npz"] = stream.getvalue()
    for arm, horizon, blocks, available in (
        ("generic", 0.25, 5, True),
        ("structured", 0.8, 8, False),
    ):
        inputs[f"control/trial-0/{arm}/trial.json"] = _bytes_json(
            {
                "arm": arm,
                "repetition": 0,
                "tracking_rmse": {"position_rmse_m": 1.0},
                "pass_criterion": {"met": False, "within_tolerance_fraction": 0.0},
                "controller": {
                    "horizon_s": horizon,
                    "block_count": blocks,
                    "uncertainty_available": available,
                },
                "model_not_ready_intervals": 2 if arm == "generic" else 0,
            }
        )
    plan = {
        "classification": {"reference_audit": "Retrospective"},
        "accepted_executable_source": "test-source",
        "baseline_runs": {"platform": "test-platform", "control": "test-control"},
        "platform_readout": {"percentile": 0.95},
        "input_sha256": {key: "validated-by-caller" for key in inputs},
        "source_sha256": {
            f"docs/harness/{name}": "test-hash"
            for name in (
                "reference.json",
                "platform-reference.json",
                "control-reference.json",
                "live-reference.json",
                "evidence-reference.json",
            )
        },
    }
    return plan, inputs


def test_report_uses_snapshots_and_keeps_legacy_acceptance_separate(
    report_inputs, monkeypatch
):
    plan, inputs = report_inputs

    def forbidden(*args, **kwargs):
        raise AssertionError("the report must not reopen a source path")

    monkeypatch.setattr(Path, "open", forbidden)
    result = reference_report(plan, inputs)
    json.dumps(result, allow_nan=False)
    assert result["acceptance_recomputed"] is False
    platform = result["platform"]
    assert platform["legacy_decision_unchanged"]["accepted"] is True
    assert platform["task_sufficiency_established"] is False
    corpus = platform["corpora"]["test"]
    assert set(corpus["predictors"]) == {
        "generic",
        "hold_current",
        "structured:structured",
    }
    comparison = corpus["informational_comparisons"]["velocity_m_s"]
    assert comparison["legacy_pooled_final_step_component_rmse"] == 0
    assert comparison["generic_within_descriptive_ceiling"] is False
    assert comparison["generic_at_or_below_structured"] is False
    control = result["control"]
    assert control["legacy_decision_unchanged"]["accepted"] is True
    assert control["trials"]["0-structured"]["application_criterion"]["met"] is False
    assert control["trials"]["0-generic"]["controller"]["horizon_s"] == 0.25
    assert control["trials"]["0-structured"]["controller"]["horizon_s"] == 0.8


def test_report_rejects_wrong_snapshot_inventory(report_inputs):
    plan, inputs = report_inputs
    del inputs["control/calibration.json"]
    with pytest.raises(ValueError, match="input snapshots"):
        reference_report(plan, inputs)


def test_report_rejects_nonfinite_saved_summary(report_inputs):
    plan, inputs = report_inputs
    row = json.loads(inputs["control/trial-0/structured/trial.json"])
    row["tracking_rmse"]["position_rmse_m"] = float("nan")
    inputs["control/trial-0/structured/trial.json"] = json.dumps(row).encode()
    with pytest.raises(ValueError, match="Out of range float"):
        reference_report(plan, inputs)
