"""Acceptance must use frozen gates and equal family weights, not win counts."""

import copy
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def decision(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from check_generic_fit import decide

    return decide


def fixture():
    manifest = dict(
        id="unit",
        required_families=["a", "b"],
        stress_families=["noise"],
        data_seeds=[1],
        extra_regression_cases=[],
        compute=dict(
            max_training_window_gradient_evaluations=100, max_development_passes=12
        ),
        error_caps={f: dict(matched=1.0, shifted=1.0) for f in ["a", "b", "noise"]},
        primary=dict(floor=0.005, maximum_ratio=0.95),
    )

    def metric(value):
        return dict(horizon_scaled_rmse=[value] * 5, overall_scaled_rmse=value)

    rows = [
        dict(
            name=f,
            family=f,
            data_seed=1,
            extra=False,
            status="complete",
            correctness_passed=True,
            optimization=dict(
                resource_usage=dict(
                    training_window_gradient_evaluations=100, development_passes=12
                )
            ),
            regimes={
                r: dict(
                    incumbent=metric(0.2), candidate=metric(0.1 if f == "a" else 0.21)
                )
                for r in ["matched", "shifted"]
            },
        )
        for f in ["a", "b", "noise"]
    ]
    return manifest, rows


def test_small_tradeoff_is_accepted_when_primary_and_caps_pass(decision):
    manifest, rows = fixture()
    answer = decision(manifest, rows)
    assert answer["accepted"]
    assert answer["primary_ratio"] == pytest.approx(np.sqrt(0.5 * 1.05))
    assert set(answer["per_family_ratios"]) == {"a", "b"}


@pytest.mark.parametrize("breach", ["accuracy", "compute", "correctness"])
def test_aggregate_gain_cannot_hide_hard_gate_failure(decision, breach):
    manifest, rows = fixture()
    if breach == "accuracy":
        rows[0]["regimes"]["matched"]["candidate"]["horizon_scaled_rmse"][2] = 1.01
    if breach == "compute":
        rows[0]["optimization"]["resource_usage"][
            "training_window_gradient_evaluations"
        ] = 101
    if breach == "correctness":
        rows[0]["correctness_passed"] = False
    answer = decision(manifest, rows)
    assert not answer["accepted"] and answer["hard_gate_breaches"]


def test_missing_or_duplicate_cases_cannot_change_weights(decision):
    manifest, rows = fixture()
    for invalid in [rows[:-1], rows + [copy.deepcopy(rows[0])]]:
        with pytest.raises(ValueError, match="frozen cases"):
            decision(manifest, invalid)


def test_tiny_error_floor_and_tie_retain_incumbent(decision):
    manifest, rows = fixture()
    for row in rows:
        for metrics in row["regimes"].values():
            metrics["incumbent"]["overall_scaled_rmse"] = 0.001
            metrics["candidate"]["overall_scaled_rmse"] = 0.00001
    answer = decision(manifest, rows)
    assert answer["primary_ratio"] == 1 and not answer["accepted"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0])
def test_invalid_scores_fail_closed_and_remain_json_serializable(decision, value):
    import json

    manifest, rows = fixture()
    rows[0]["regimes"]["matched"]["candidate"]["overall_scaled_rmse"] = value
    rows[0]["regimes"]["matched"]["candidate"]["horizon_scaled_rmse"][0] = value
    answer = decision(manifest, rows)
    assert not answer["accepted"]
    json.dumps(answer, allow_nan=False)


def test_missing_stress_regime_and_negative_compute_fail_closed(decision):
    manifest, rows = fixture()
    del rows[-1]["regimes"]["shifted"]
    with pytest.raises(ValueError, match="evaluation regimes"):
        decision(manifest, rows)
    manifest, rows = fixture()
    rows[0]["optimization"]["resource_usage"]["development_passes"] = -1
    assert not decision(manifest, rows)["accepted"]


@pytest.mark.parametrize("mode", ["fit", "verify"])
def test_manifest_changes_cannot_silently_create_a_new_acceptance_contract(
    monkeypatch, tmp_path, mode
):
    from types import SimpleNamespace

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from check_generic_fit import main, verify_run

    path = tmp_path / "manifest.json"
    path.write_text('{"id": "generic-fit-m1-v1"}\n')
    with pytest.raises(ValueError, match="manifest digest"):
        if mode == "fit":
            main(SimpleNamespace(manifest=path, output=tmp_path / "output"))
        else:
            verify_run(tmp_path, path)
    assert not (tmp_path / "output").exists()


def test_numpy_replay_matches_causal_model_without_a_fit(monkeypatch):
    import jax

    from glassbox.experimental.sequence_model import (
        initialize_sequence_model,
        sequence_windows,
    )

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from check_generic_fit import recurrence

    rng = np.random.default_rng(100)
    b = sequence_windows(
        rng.normal(size=(31, 2)),
        rng.normal(size=(30, 3)),
        np.arange(2, 25),
        history_steps=2,
        horizon_steps=5,
        dt_s=0.05,
    )
    model = initialize_sequence_model(b, kind="delay_mlp", width=4)
    with jax.enable_x64(True):
        expected = np.asarray(
            model.rollout(b.past_states, b.past_inputs, b.future_inputs)
        )
    actual = recurrence(model, b.past_states, b.past_inputs, b.future_inputs)
    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-9)
