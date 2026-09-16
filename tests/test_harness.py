"""Harness contracts: the frozen manifest, the decision gates, and the replay.

These tests never fit a case. They exercise the gates on synthetic rows, the
manifest digest, the NumPy replay against the JAX rollout, and the witness
probe construction.
"""

import copy
import json
import shutil
from pathlib import Path

import jax
import numpy as np
import pytest

from glassbox.experimental import harness
from glassbox.experimental.sequence_model import (
    initialize_sequence_model,
    sequence_windows,
)

MANIFEST = Path(__file__).resolve().parents[1] / "docs/harness/v1.json"
CONTEXT, DELAY, HORIZON = 10, 2, 5


@pytest.fixture(scope="module")
def manifest():
    return harness.frozen_manifest(MANIFEST)


def passing_rows(manifest):
    """One complete, comfortably passing row per declared case."""
    rows = []
    for family, seed in sorted(harness.expected_cases(manifest)):
        regimes = harness.case_regimes(manifest, family)
        rows.append(
            dict(
                name=f"{family}-{seed}",
                family=family,
                data_seed=seed,
                status="complete",
                regimes={
                    regime: dict(
                        windows=4,
                        channel_rmse=[[0.0]],
                        horizon_scaled_rmse=[manifest["error_caps"][family][regime] / 4]
                        * HORIZON,
                        overall_scaled_rmse=manifest["error_caps"][family][regime] / 4,
                        recordings={},
                    )
                    for regime in regimes
                },
            )
        )
        if family == harness.WITNESS:
            rows[-1]["probe"] = dict(paired_rmse=[0.001] * HORIZON)
    return rows


def reference_from(rows):
    return dict(
        overall_scaled_rmse={
            row["name"]: {
                regime: metrics["overall_scaled_rmse"]
                for regime, metrics in row["regimes"].items()
            }
            for row in rows
        }
    )


def gates(decision):
    return {b["gate"] for b in decision["gate_breaches"]} | {
        r["gate"] for r in decision["reference_regressions"]
    }


def test_a_complete_run_is_accepted_and_json_serializable(manifest):
    decision = harness.decide(manifest, passing_rows(manifest))
    assert decision["accepted"] and decision["decision"] == "accept"
    assert decision["cases"] == 27 and not decision["reference_compared"]
    json.dumps(decision, allow_nan=False)


def test_a_cap_breach_fails_closed(manifest):
    rows = passing_rows(manifest)
    row = next(r for r in rows if r["family"] == "stable_affine")
    cap = manifest["error_caps"]["stable_affine"]["matched"]
    row["regimes"]["matched"]["horizon_scaled_rmse"][2] = cap * 1.0001
    decision = harness.decide(manifest, rows)
    assert not decision["accepted"] and gates(decision) == {"horizon_scaled_rmse"}
    breach = decision["gate_breaches"][0]
    assert breach["case"] == row["name"] and breach["limit"] == cap


def test_a_probe_breach_fails_closed(manifest):
    rows = passing_rows(manifest)
    row = next(r for r in rows if r["family"] == harness.WITNESS)
    limit = manifest["witness"]["paired_probe_first_step_rmse_maximum"]
    row["probe"]["paired_rmse"][0] = limit * 1.01
    decision = harness.decide(manifest, rows)
    assert not decision["accepted"]
    assert gates(decision) == {"paired_probe_first_step_rmse"}
    assert decision["paired_probe_first_step_rmse"][row["name"]] == limit * 1.01


def test_a_missing_probe_fails_closed(manifest):
    rows = passing_rows(manifest)
    next(r for r in rows if r["family"] == harness.WITNESS).pop("probe")
    assert gates(harness.decide(manifest, rows)) == {"paired_probe_first_step_rmse"}


def test_a_reference_regression_fails_closed(manifest):
    rows = passing_rows(manifest)
    reference = reference_from(copy.deepcopy(rows))
    allowance = manifest["reference"]
    row = next(r for r in rows if r["family"] == "noisy_observation")
    base = reference["overall_scaled_rmse"][row["name"]]["shifted"]
    inside = base * (1 + allowance["relative_tolerance"])
    row["regimes"]["shifted"]["overall_scaled_rmse"] = inside
    decision = harness.decide(manifest, rows, reference)
    assert decision["accepted"] and decision["reference_compared"]
    row["regimes"]["shifted"]["overall_scaled_rmse"] = (
        inside + allowance["absolute_tolerance"] * 1.01
    )
    decision = harness.decide(manifest, rows, reference)
    assert not decision["accepted"]
    assert gates(decision) == {"reference_overall_scaled_rmse"}


def test_a_reference_without_the_case_fails_closed(manifest):
    rows = passing_rows(manifest)
    reference = reference_from(copy.deepcopy(rows))
    reference["overall_scaled_rmse"].pop("off_periodic-6202")
    assert gates(harness.decide(manifest, rows, reference)) == {"reference_present"}


@pytest.mark.parametrize(
    "problem",
    ["missing", "duplicate", "undeclared", "unknown_family", "incomplete"],
)
def test_missing_duplicate_and_undeclared_cases_fail_closed(manifest, problem):
    rows = passing_rows(manifest)
    expected = {
        "missing": "case_present",
        "duplicate": "case_unique",
        "undeclared": "case_declared",
        "unknown_family": "family_declared",
        "incomplete": "fit_complete",
    }[problem]
    if problem == "missing":
        rows.pop(0)
    if problem == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    if problem == "undeclared":
        rows[0] = {**copy.deepcopy(rows[0]), "data_seed": 9999, "name": "extra"}
    if problem == "unknown_family":
        rows[0] = {**copy.deepcopy(rows[0]), "family": "invented", "name": "invented"}
    if problem == "incomplete":
        rows[0] = dict(
            name=rows[0]["name"],
            family=rows[0]["family"],
            data_seed=rows[0]["data_seed"],
            status="failed",
        )
    decision = harness.decide(manifest, rows)
    assert not decision["accepted"] and expected in gates(decision)
    json.dumps(decision, allow_nan=False)


def test_missing_or_extra_regimes_fail_closed(manifest):
    rows = passing_rows(manifest)
    rows[0]["regimes"].pop("shifted")
    assert gates(harness.decide(manifest, rows)) == {"regimes_declared"}
    rows = passing_rows(manifest)
    witness = next(r for r in rows if r["family"] == harness.WITNESS)
    witness["regimes"]["shifted"] = copy.deepcopy(witness["regimes"]["matched"])
    assert gates(harness.decide(manifest, rows)) == {"regimes_declared"}


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_scores_fail_closed_and_stay_serializable(manifest, value):
    rows = passing_rows(manifest)
    rows[0]["regimes"]["matched"]["horizon_scaled_rmse"][0] = value
    rows[0]["regimes"]["matched"]["overall_scaled_rmse"] = value
    reference = reference_from(copy.deepcopy(passing_rows(manifest)))
    decision = harness.decide(manifest, rows, reference)
    assert not decision["accepted"]
    assert gates(decision) >= {"horizon_scaled_rmse", "finite_overall_scaled_rmse"}
    name = rows[0]["name"]
    assert decision["overall_scaled_rmse"][name]["matched"] is None
    json.dumps(decision, allow_nan=False)


def test_the_manifest_digest_gates_run_and_verify(tmp_path, manifest):
    assert harness.sha256(MANIFEST) == harness.MANIFEST_SHA256
    altered = tmp_path / "v1.json"
    tampered = copy.deepcopy(manifest)
    tampered["error_caps"]["stable_affine"]["matched"] = 1.0
    harness.write(altered, tampered)
    with pytest.raises(ValueError, match="manifest digest"):
        harness.frozen_manifest(altered)
    with pytest.raises(ValueError, match="manifest digest"):
        harness.run(altered, tmp_path / "out")
    assert not (tmp_path / "out").exists()
    directory = tmp_path / "saved"
    directory.mkdir()
    harness.write(directory / "manifest.json", tampered)
    with pytest.raises(ValueError, match="manifest digest"):
        harness.verify(directory)


def test_the_manifest_carries_the_recipe_and_the_frozen_plan(manifest):
    from glassbox.experimental.default_model import RECIPE

    # The manifest records the recipe it was frozen against in full, and pins
    # the evaluation plan that recipe is cut from. It does not pin the
    # candidate: a frozen gate that only its own recipe could pass would never
    # measure a change.
    assert set(manifest["recipe"]) == set(RECIPE)
    assert harness.declared_plan(manifest) == manifest["recipe"]
    plan = manifest["dataset"]
    assert (plan["dt_s"], plan["intervals"], plan["rng_salt"]) == (0.05, 160, 1207)
    assert plan["calibration_recordings"] == 8
    assert plan["evaluation_recordings_per_regime"] == 4
    assert (plan["evaluation_origin_start"], plan["evaluation_stride"]) == (2, 5)
    assert sorted(plan["regimes"]) == ["calibration", "matched", "shifted"]
    assert manifest["data_seeds"] == [6101, 6202, 6303]
    assert manifest["witness"]["data_seeds"] == [101, 202, 303]
    assert manifest["witness"]["paired_probe_first_step_rmse_maximum"] == 0.05
    assert tuple(manifest["families"]) == harness.FAMILIES
    assert manifest["error_caps"]["hidden_input_delay"] == {"matched": 0.9}


def test_a_manifest_declaring_another_evaluation_plan_is_refused(manifest):
    """The plan is pinned even though the rest of the recipe is not."""
    for name in harness.PLAN_CONSTANTS:
        other = copy.deepcopy(manifest)
        other["recipe"][name] = manifest["recipe"][name] * 2
        with pytest.raises(ValueError, match="evaluation plan"):
            harness.declared_plan(other)
    missing = copy.deepcopy(manifest)
    del missing["recipe"]["context_s"]
    with pytest.raises(ValueError, match="evaluation plan"):
        harness.declared_plan(missing)


def witness_batch(seeds):
    plan = harness.read(MANIFEST)["dataset"]
    parts = []
    for seed in seeds:
        supplied = harness.witness_recordings(plan, seed)
        for segment in supplied.segments[:2]:
            parts.append(
                sequence_windows(
                    segment.states,
                    segment.inputs,
                    np.arange(CONTEXT, 40, 7),
                    history_steps=CONTEXT,
                    horizon_steps=HORIZON,
                    dt_s=plan["dt_s"],
                )
            )
    return type(parts[0])(
        **{
            k: np.concatenate([getattr(b, k) for b in parts])
            for k in ("past_states", "past_inputs", "future_inputs", "future_states")
        },
        dt_s=plan["dt_s"],
    )


def test_numpy_replay_matches_the_jax_rollout_without_fitting():
    batch = witness_batch((11, 12))
    model = initialize_sequence_model(batch, width=4, memory=3, delay_steps=DELAY)
    # An affine start reads the memory out as zero; give it weight so the
    # replayed filter has to agree, not merely cancel.
    model.params["linear"][-3:] = 0.15
    model.params["w2"][:] = 0.05
    x, up, uf = batch.past_states, batch.past_inputs, batch.future_inputs
    with jax.enable_x64(True):
        rollout = np.asarray(model.rollout(x, up, uf))
    replayed = harness.replay(model, x, up, uf)
    assert replayed.shape == rollout.shape
    np.testing.assert_allclose(replayed, rollout, **harness.REPLAY_TOLERANCE)
    # The replay runs the filter, not just the explicit delay history: moving
    # the last input outside that history still moves every forecast.
    moved = np.array(up)
    moved[:, CONTEXT - DELAY - 1] += 5.0
    difference = np.abs(harness.replay(model, x, moved, uf) - replayed)
    assert difference.max() > 1e-3


def test_the_witness_probe_withholds_exactly_one_input():
    px, pu, uf, target = harness.witness_probe(CONTEXT, HORIZON)
    assert px.shape == (2, CONTEXT + 1, 1) and pu.shape == (2, CONTEXT, 1)
    assert uf.shape == (2, HORIZON, 1)
    np.testing.assert_array_equal(px[0], px[1])
    np.testing.assert_array_equal(uf[0], uf[1])
    np.testing.assert_array_equal(uf, 0)
    assert np.count_nonzero(pu[0] != pu[1]) == 1
    np.testing.assert_array_equal(pu[:, CONTEXT - 3, 0], [-1.0, 1.0])
    np.testing.assert_allclose(
        target[:, :, 0],
        np.array([[-1.0], [1.0]]) * 0.2 * 0.8 ** np.arange(HORIZON),
        atol=1e-15,
    )
    # A forecast blind to the withheld command answers identically, so its
    # first-step paired RMSE cannot fall below 0.2.
    blind = np.repeat(px[:, -1:], HORIZON, axis=1)
    assert harness.paired_rmse(blind, target)[0] >= 0.2 - 1e-12


def test_the_witness_recordings_are_the_declared_delayed_system():
    plan = harness.read(MANIFEST)["dataset"]
    supplied = harness.witness_recordings(plan, 101)
    assert len(supplied.segments) == plan["calibration_recordings"]
    assert len(harness.witness_recordings(plan, 101, evaluation=True).segments) == 4
    fitted = {s.recording_id for s in supplied.segments}
    evaluated = {
        s.recording_id
        for s in harness.witness_recordings(plan, 101, evaluation=True).segments
    }
    assert not fitted & evaluated
    segment = supplied.segments[0]
    x, u = segment.states, segment.inputs
    assert len(u) == plan["intervals"]
    for k in range(len(u)):
        delayed = 0.2 * u[k - 3] if k >= 3 else 0.0
        np.testing.assert_allclose(x[k + 1], 0.8 * x[k] + delayed, atol=1e-12)


@pytest.mark.parametrize("problem", ["artifact", "result"])
def test_verify_rejects_an_altered_run(tmp_path, manifest, problem):
    directory = tmp_path / "run"
    (directory / "stable_affine-6101").mkdir(parents=True)
    shutil.copyfile(MANIFEST, directory / "manifest.json")
    case = directory / "stable_affine-6101"
    (case / "model.npz").write_bytes(b"not a model")
    row = dict(
        name="stable_affine-6101",
        family="stable_affine",
        data_seed=6101,
        status="complete",
        files={"model.npz": harness.sha256(case / "model.npz")},
        regimes={},
    )
    harness.write(directory / "results.json", [row])
    harness.write(case / "result.json", row)
    if problem == "artifact":
        (case / "model.npz").write_bytes(b"tampered model")
        expected = "altered artifact"
    else:
        harness.write(case / "result.json", {**row, "status": "failed"})
        expected = "case result mismatch"
    with pytest.raises(ValueError, match=expected):
        harness.verify(directory)


REFERENCE = MANIFEST.parent / "reference.json"


def scored_run(tmp_path, manifest, *, reference=REFERENCE):
    """A run directory with no cases, so a replay is only the anchoring check.

    Every gate that needs a fit is already covered above on synthetic rows.
    What this exercises is the part a saved run can lie about: the reference
    its own regression gate was measured against.
    """
    directory = tmp_path / "run"
    directory.mkdir(parents=True)
    shutil.copyfile(MANIFEST, directory / "manifest.json")
    harness.write(directory / "results.json", [])
    anchor, digest = None, None
    if reference is not None:
        shutil.copyfile(reference, directory / "reference.json")
        anchor, digest = harness.read(reference), harness.sha256(reference)
    harness.write(
        directory / "decision.json", harness.decide(manifest, [], anchor, digest)
    )
    return directory


def loosen(value, factor=10):
    if isinstance(value, float):
        return value * factor
    if isinstance(value, dict):
        return {k: loosen(v, factor) for k, v in value.items()}
    if isinstance(value, list):
        return [loosen(v, factor) for v in value]
    return value


def test_the_committed_reference_anchors_an_honest_replay(tmp_path, manifest):
    directory = scored_run(tmp_path, manifest)
    result = harness.verify(directory)
    assert result["decision"]["reference_compared"]
    assert result["decision"]["reference_sha256"] == harness.sha256(REFERENCE)
    assert harness.COMMITTED_REFERENCE == REFERENCE


def test_a_run_cannot_relax_its_own_regression_gate(tmp_path, manifest):
    """The defect: a replay must not read the threshold the run saved itself."""
    directory = scored_run(tmp_path, manifest)
    copied = directory / "reference.json"
    harness.write(copied, loosen(harness.read(copied)))
    assert harness.sha256(copied) != harness.sha256(REFERENCE)
    with pytest.raises(ValueError, match="cannot relax its own regression gate"):
        harness.verify(directory)


def test_a_mismatched_committed_reference_is_rejected(tmp_path, manifest):
    directory = scored_run(tmp_path, manifest)
    other = tmp_path / "other-reference.json"
    harness.write(other, loosen(harness.read(REFERENCE)))
    with pytest.raises(ValueError, match="cannot relax its own regression gate"):
        harness.verify(directory, other)


def test_a_run_and_a_committed_reference_must_agree_about_existing(tmp_path, manifest):
    compared = scored_run(tmp_path, manifest)
    missing = tmp_path / "absent-reference.json"
    with pytest.raises(ValueError, match="cannot be anchored"):
        harness.verify(compared, missing)
    uncompared = scored_run(tmp_path / "second", manifest, reference=None)
    assert not harness.read(uncompared / "decision.json")["reference_compared"]
    with pytest.raises(ValueError, match="cannot be anchored"):
        harness.verify(uncompared)
    assert harness.verify(uncompared, missing)["decision"]["reference_sha256"] is None


def test_a_forged_reference_digest_in_the_decision_is_rejected(tmp_path, manifest):
    directory = scored_run(tmp_path, manifest)
    decision = harness.read(directory / "decision.json")
    decision["reference_sha256"] = "0" * 64
    harness.write(directory / "decision.json", decision)
    with pytest.raises(ValueError, match="reference_sha256"):
        harness.verify(directory)
