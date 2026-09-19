"""Synthetic qualification integrity/control flow, without learned-model fits."""

import copy
import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from glassbox import learner
from glassbox.experimental import public_mean_scoring as scoring
from glassbox.experimental import public_mean_synthetic as synthetic
from glassbox.io.recordings import save_recordings
from glassbox.recordings import SequenceCollection, SequenceSegment

ROOT = Path(__file__).resolve().parents[1]


def _passing_results():
    cases, ledger = scoring.specification(ROOT)
    results = []
    for case in cases:
        scale = np.asarray(ledger["cases"][case["name"]]["reference_state_scale"])
        array = np.zeros((4, 5, len(scale)))
        scores = scoring.forecast_scores(
            array, array, scale, scale * 10, ["a", "b", "c", "d"]
        )
        rows = [
            {"case": case["name"], "regime": r, "scores": copy.deepcopy(scores)}
            for r in case["regimes"]
        ]
        probes = []
        if case["family"] == "hidden_input_delay":
            probe = np.zeros((2, 5, 1))
            probes = [
                {
                    "seed": case["data_seed"],
                    "scores": scoring.probe_scores(probe, probe, scale),
                }
            ]
        measured = {
            "status": "complete",
            "rows": rows,
            "probes": probes,
            "legacy": {"name": case["name"], "status": "complete"},
        }
        results.append(
            {
                "case": case["name"],
                "status": "complete",
                "fit_complete": True,
                "precisions": {
                    "float32": copy.deepcopy(measured),
                    "float64": copy.deepcopy(measured),
                },
            }
        )
    return cases, results


def test_exact_full_roster_reduces_255_required_caps_and_three_probes():
    cases, results = _passing_results()
    decision, legacy = synthetic.aggregate(ROOT, cases, results)
    assert decision["fixed_reference_absolute_capability_pass"]
    assert len(decision["decisions"]["float32"]["horizon_checks"]) == 255
    assert decision["decisions"]["float32"]["probe_pass"] == [True] * 3
    assert len(legacy["float32"]) == 27
    assert decision["legacy_flags_diagnostic_only"]


def test_default32_failure_cannot_select_passing_float64_fallback():
    cases, results = _passing_results()
    results[0]["precisions"]["float32"] = {"status": "failed", "error": "nonfinite"}
    decision, legacy = synthetic.aggregate(ROOT, cases, results)
    assert not decision["fixed_reference_absolute_capability_pass"]
    assert decision["decisions"]["float64"]["fixed_reference_absolute_capability_pass"]
    assert decision["decisions"]["float32"]["unavailable_cases"] == [cases[0]["name"]]
    assert legacy["float32"][0]["status"] == "failed"
    assert len(legacy["float32"]) == 27


def test_failed_fit_retained_even_if_some_scores_are_present():
    cases, results = _passing_results()
    results[0]["fit_complete"] = False
    results[0]["status"] = "failed"
    decision, _ = synthetic.aggregate(ROOT, cases, results)
    assert not decision["fixed_reference_absolute_capability_pass"]
    assert len(decision["completed_fits"]) == 26
    assert decision["failed_cases"] == [cases[0]["name"]]


@pytest.mark.parametrize(
    "mutation", ["missing", "duplicate", "reorder", "string_status"]
)
def test_case_roster_cannot_be_reduced_reordered_or_retyped(mutation):
    cases, results = _passing_results()
    if mutation == "missing":
        results.pop()
    elif mutation == "duplicate":
        results[-1] = copy.deepcopy(results[0])
    elif mutation == "reorder":
        results.reverse()
    else:
        results[0]["fit_complete"] = "true"
    with pytest.raises(ValueError):
        synthetic.aggregate(ROOT, cases, results)


def test_fixed_denominator_substitution_is_a_required_failure():
    cases, results = _passing_results()
    row = results[0]["precisions"]["float32"]["rows"][0]
    row["scores"]["reference_state_scale"] = row["scores"]["candidate_state_scale"]
    decision, _ = synthetic.aggregate(ROOT, cases, results)
    assert not decision["fixed_reference_absolute_capability_pass"]
    assert (
        "denominator"
        in decision["decisions"]["float32"]["incomplete_or_invalid_roster"]
    )


def test_required_caps_and_physical_probe_remain_inclusive():
    cases, results = _passing_results()
    case = cases[0]
    for row in results[0]["precisions"]["float32"]["rows"]:
        row["scores"]["fixed_reference"]["horizon_rmse"] = [
            case["caps"][row["regime"]]
        ] * 5
    results[-1]["precisions"]["float32"]["probes"][0]["scores"]["physical_rmse"][0] = (
        0.05
    )
    decision, _ = synthetic.aggregate(ROOT, cases, results)
    assert decision["fixed_reference_absolute_capability_pass"]
    results[-1]["precisions"]["float32"]["probes"][0]["scores"]["physical_rmse"][0] = (
        np.nextafter(0.05, np.inf)
    )
    decision, _ = synthetic.aggregate(ROOT, cases, results)
    assert not decision["fixed_reference_absolute_capability_pass"]


def _cache_fixture(tmp_path):
    segments = []
    for i in range(8):
        t = np.arange(161, dtype=np.float64)
        x = np.column_stack((t / 100 + i, t**2 / 10000 + i / 8))
        u = (np.arange(160, dtype=np.float64) / 200 + i / 7)[:, None]
        segments.append(SequenceSegment(f"calibration-{i}", "whole", x, u, 0.05))
    supplied = SequenceCollection(
        tuple(segments),
        configuration_id="cache-test",
        state_channels=("x [unitless]", "y [unitless]"),
        input_channels=("u [unitless]",),
    )
    names = sorted(
        (s.recording_id for s in segments),
        key=lambda n: hashlib.sha256(
            json.dumps(n, sort_keys=True).encode()
        ).hexdigest(),
    )
    train = learner._extract(supplied, names[2:], 384)
    development = learner._extract(supplied, names[:2], 256)
    meta = {
        "seen": learner._recording_content(supplied),
        "contract": learner._contract(supplied),
        "model": {"dt_s": 0.05},
        "windows": {
            role: {
                "keys": [asdict(k) for k in windows.keys],
                "source_origins": list(windows.source_origins),
            }
            for role, windows in (("train", train), ("development", development))
        },
    }
    arrays = {
        f"{role}_{key}": getattr(w.batch, key)
        for role, w in (("train", train), ("development", development))
        for key in synthetic.ARRAYS
    }
    arrays["norm_state_scale"] = np.ones(2)
    calibration = tmp_path / "calibration.npz"
    old = tmp_path / "old.npz"
    save_recordings(supplied, calibration)
    np.savez_compressed(old, metadata=json.dumps(meta), **arrays)
    return calibration, old, meta, arrays


def test_preparation_checks_old384_prefix_and_preserves_all876_new_windows(
    tmp_path, monkeypatch
):
    calibration, old, _, _ = _cache_fixture(tmp_path)
    monkeypatch.setattr(
        synthetic.glassbox, "fit", lambda *a: pytest.fail("preparation fitted a model")
    )
    _, train, development, report = synthetic.preparation(calibration, old)
    assert len(train.keys) == 876 and len(development.keys) == 256
    assert report["old_training_prefix"] == 384
    assert (
        len(report["roles"]["train"]) == 6 and len(report["roles"]["development"]) == 2
    )
    assert len(set(train.keys)) == 876


@pytest.mark.parametrize(
    "target",
    ["training_array", "development_array", "origin", "key", "contract", "dtype"],
)
def test_preparation_refuses_old_cache_or_identity_mismatches(tmp_path, target):
    calibration, _old, meta, arrays = _cache_fixture(tmp_path)
    arrays = {k: np.array(v, copy=True) for k, v in arrays.items()}
    if target == "training_array":
        arrays["train_future_inputs"][0, 0, 0] += 0.01
    elif target == "development_array":
        arrays["development_future_states"][0, 0, 0] += 0.01
    elif target == "origin":
        meta["windows"]["train"]["source_origins"][0] += 1
    elif target == "key":
        meta["windows"]["train"]["keys"][0]["recording_id"] = "substitution"
    elif target == "contract":
        meta["contract"]["configuration_id"] = "other"
    else:
        arrays["train_past_states"] = arrays["train_past_states"].astype(np.float32)
    changed = tmp_path / "changed.npz"
    np.savez_compressed(changed, metadata=json.dumps(meta), **arrays)
    with pytest.raises(ValueError):
        synthetic.preparation(calibration, changed)


def test_generation_packet_requires_external_anchor_and_complete_payloads(tmp_path):
    directory = tmp_path / "data"
    directory.mkdir()
    (directory / "sample.bin").write_bytes(b"data")
    cases = [{"name": "one"}]
    report = {
        "status": "complete",
        "binding_sha256": "binding",
        "cases": ["one"],
        "numerical_fits": 0,
        "files": synthetic._payloads(directory),
    }
    synthetic._json(directory / "result.json", report)
    digest = synthetic._sha(directory / "result.json")
    synthetic._verify_data(directory, digest, "binding", cases)
    with pytest.raises(ValueError, match="external generated-data"):
        synthetic._verify_data(directory, "0" * 64, "binding", cases)
    (directory / "sample.bin").write_bytes(b"altered")
    with pytest.raises(ValueError, match="payload integrity"):
        synthetic._verify_data(directory, digest, "binding", cases)


def test_legacy_conversion_does_not_change_score_units_or_recording_breakdown():
    target = np.zeros((2, 5, 1))
    scores = scoring.forecast_scores(
        np.ones_like(target), target, [2.0], [4.0], ["first", "second"]
    )
    row = synthetic._legacy_metrics(scores, 2)
    assert row["horizon_scaled_rmse"] == [0.25] * 5
    assert row["channel_rmse"] == [[1.0]] * 5
    assert row["recordings"]["first"]["overall_scaled_rmse"] == 0.25
    assert scores["fixed_reference"]["overall_rmse"] == 0.5


def test_coverage_is_inclusive_per_channel_and_has_no_mean_gate():
    predicted = np.array([[[0.0, 2.0], [3.0, 0.0]]])
    result = synthetic._coverage(
        predicted, np.zeros_like(predicted), np.ones_like(predicted) * 2
    )
    assert result["coverage"] == {"channel_0": [1.0, 0.0], "channel_1": [1.0, 1.0]}
    assert "accepted" not in result


def test_exact_prediction_tapes_require_identity_dtype_and_every_array(tmp_path):
    arrays = {
        "past_states": np.zeros((1, 11, 1)),
        "past_inputs": np.zeros((1, 10, 1)),
        "future_inputs": np.zeros((1, 5, 1)),
        "targets": np.zeros((1, 5, 1)),
        "recording_ids": np.array(["matched-0"]),
        "source_origins": np.array([12]),
    }
    path = tmp_path / "historical.npz"
    np.savez_compressed(path, **arrays, prediction=np.zeros((1, 5, 1)))
    synthetic._validate_queries(arrays, path)
    changed = {**arrays, "source_origins": np.array([17])}
    with pytest.raises(ValueError, match="evaluation tape"):
        synthetic._validate_queries(changed, path)
    with pytest.raises(ValueError, match="query roster"):
        synthetic._validate_queries(
            {k: v for k, v in arrays.items() if k != "targets"}, path
        )


def test_launch_retains_nonzero_exit_without_retry(tmp_path, monkeypatch):
    calls = []

    def failed(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(synthetic.subprocess, "run", failed)
    result = synthetic._launch(["worker"], tmp_path, {}, tmp_path / "stage")
    assert result["status"] == "worker_failed" and result["returncode"] == 7
    assert len(calls) == 1
    assert synthetic._read(tmp_path / "stage.exit.json") == result


def test_hard_timeout_is_incomplete_not_internal_fit_failure(tmp_path, monkeypatch):
    def timeout(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, 14400)

    monkeypatch.setattr(synthetic.subprocess, "run", timeout)
    result = synthetic._launch(["worker"], tmp_path, {}, tmp_path / "stage")
    assert (
        result["status"] == "hard_timeout_incomplete" and result["returncode"] is None
    )


def test_run_and_replay_never_overwrite_existing_attempt(tmp_path):
    with pytest.raises(FileExistsError):
        synthetic.run(
            tmp_path,
            binding_path="unused",
            binding_sha256="unused",
            artifact_root="unused",
        )
    with pytest.raises(ValueError, match="outside"):
        synthetic.replay(
            tmp_path,
            "unused",
            tmp_path / "nested",
            binding_path="unused",
            binding_sha256="unused",
            artifact_root="unused",
        )


def test_wrong_replay_anchor_retains_failure_without_loading_model(
    tmp_path, monkeypatch
):
    directory = tmp_path / "sealed"
    directory.mkdir()
    synthetic._json(directory / "run.json", {"status": "complete"})
    monkeypatch.setattr(
        synthetic.glassbox.LearnedDynamics,
        "load",
        lambda *args: pytest.fail("unauthenticated model load"),
    )
    output = tmp_path / "replay"
    with pytest.raises(ValueError, match="external synthetic"):
        synthetic.replay(
            directory,
            "0" * 64,
            output,
            binding_path="unused",
            binding_sha256="unused",
            artifact_root="unused",
        )
    assert synthetic._read(output / "replay.json")["status"] == "failed"


def test_outer_failure_lists_every_remaining_case_as_not_attempted(
    tmp_path, monkeypatch
):
    cases, _ = scoring.specification(ROOT)
    monkeypatch.setattr(synthetic, "_bound", lambda *args: {"public_root": str(ROOT)})
    monkeypatch.setattr(
        scoring,
        "authenticate_reference_scales",
        lambda *args: {case["name"]: np.ones(1) for case in cases},
    )

    def failed(*args, **kwargs):
        raise ValueError("generation failed")

    monkeypatch.setattr(synthetic, "_historical", failed)
    output = tmp_path / "attempt"
    with pytest.raises(ValueError, match="generation failed"):
        synthetic.run(
            output, binding_path="binding", binding_sha256="sha", artifact_root="unused"
        )
    report = synthetic._read(output / "run.json")
    assert len(report["case_results"]) == len(report["not_attempted_cases"]) == 27
    assert all(
        r["status"] == "not_attempted" and not r["fit_complete"]
        for r in report["case_results"]
    )
    assert report["known_public_fit_calls"] == 0


def test_root_case_mirror_is_checked_before_any_replay_forecast(tmp_path):
    original = tmp_path / "case"
    original.mkdir()
    synthetic._json(original / "result.json", {"case": "one", "status": "failed"})
    with pytest.raises(ValueError, match="root/per-case"):
        synthetic._verify_case(
            original,
            tmp_path / "out",
            {"name": "one"},
            tmp_path,
            tmp_path,
            [1.0],
            {"case": "other"},
        )
