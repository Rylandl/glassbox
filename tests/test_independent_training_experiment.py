"""Bounded selector, public-update, failure and scientific-reduction contracts.

These fixtures are artificial arrays. No simulator or numerical fit is executed.
"""

import json
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from glassbox import LearnedDynamics, learner
from glassbox._learner_arrays import load_arrays, save_arrays
from glassbox.experimental import independent_training_experiment as subject
from glassbox.recordings import SequenceCollection, SequenceSegment


def collection(prefix, count, *, short=None):
    segments = []
    for i in range(count):
        n = 12 if i == short else 40 + i % 5
        t = np.arange(n + 1, dtype=float)
        states = np.stack(
            (t / 11 + i + (0 if prefix == "old" else 1000), np.sin(t / 9 + i)), axis=1
        )
        inputs = np.cos(np.arange(n, dtype=float) / 7 + i)[:, None]
        segments.append(
            SequenceSegment(
                f"{prefix}/{i:03}", "segment", states, inputs, 0.05, start_row=3
            )
        )
    return SequenceCollection(
        tuple(segments),
        configuration_id="toy",
        state_channels=("x", "y"),
        input_channels=("u",),
    )


class Control:
    update = LearnedDynamics.update

    def __init__(self, old, roles):
        self._contract = learner._contract(old)
        self._seen = learner._recording_content(old)
        self._train = learner._extract(old, roles["training"], 1536)
        self._development = learner._extract(old, roles["development"], 256)

    def fingerprint(self):
        return "unchanged-original-control"


@pytest.fixture
def preparation(monkeypatch):
    old = collection("old", 96)
    fresh = collection("new", 72)
    roles = {
        "training": [f"old/{i:03}" for i in range(72)],
        "development": [f"old/{i:03}" for i in range(72, 96)],
    }
    plan = {
        "original_roles": {"crazyflow": roles},
        "added_training": [
            {
                "id": f"new/{i:03}",
                "simulator": "crazyflow",
                "cell": "toy",
                "source_condition_parent": f"old/{i:03}",
            }
            for i in range(72)
        ],
    }
    monkeypatch.setattr(subject, "roster", lambda p: deepcopy(plan))
    return old, fresh, roles, Control(old, roles)


def test_actual_public_update_merges_exact_full_pool_and_preserves_revision(
    preparation, monkeypatch
):
    old, fresh, _, control = preparation
    prepared = subject.prepare_update("crazyflow", control, old, fresh, {})
    assert len(prepared.expected_train.keys) == 1536
    assert len({k.recording_id for k in prepared.expected_train.keys}) == 144
    calls = []
    sentinel = object()

    def no_fit(t, d, c, s, **kw):
        calls.append((t, d, c, s, kw))
        return sentinel

    monkeypatch.setattr(learner, "_train", no_fit)
    assert control.update(fresh) is sentinel
    assert len(calls) == 1
    t, d, c, s, kw = calls[0]
    subject.saved.equal_windows(t, prepared.expected_train, "public merge")
    assert d is control._development
    assert c == prepared.contract and s == prepared.seen and len(s) == 168
    assert kw["previous"] == control.fingerprint()
    assert (
        prepared.provenance["selected_old_windows"]
        + prepared.provenance["selected_new_windows"]
        == 1536
    )
    assert (
        prepared.provenance["training_overlap"]["target_transition_visits"] == 1536 * 5
    )


def test_missing_new_parent_is_explicit_unavailable(preparation):
    old, fresh, _, control = preparation
    fewer = SequenceCollection(
        fresh.segments[:-1],
        configuration_id=fresh.configuration_id,
        state_channels=fresh.state_channels,
        input_channels=fresh.input_channels,
    )
    with pytest.raises(subject.PreparationUnavailable) as error:
        subject.prepare_update("crazyflow", control, old, fewer, {})
    assert error.value.details["missing_support"] == ["new/071"]


def test_short_new_parent_is_not_replaced(preparation):
    old, _, _, control = preparation
    with pytest.raises(subject.PreparationUnavailable) as error:
        subject.prepare_update(
            "crazyflow", control, old, collection("new", 72, short=3), {}
        )
    assert error.value.details["legal_windows_per_new_parent"]["new/003"] == 0


def test_changed_control_training_command_fails_before_update(preparation):
    old, fresh, _, control = preparation
    b = control._train.batch
    altered = b.future_inputs.copy()
    altered[0, 0, 0] += 0.25
    from glassbox._sequence_model import SequenceBatch
    from glassbox.recordings import SequenceWindows

    control._train = SequenceWindows(
        SequenceBatch(b.past_states, b.past_inputs, altered, b.future_states, b.dt_s),
        control._train.keys,
        control._train.source_origins,
    )
    with pytest.raises(ValueError, match="cache"):
        subject.prepare_update("crazyflow", control, old, fresh, {})


def test_duplicate_content_is_not_new_trajectory_support(preparation):
    old, fresh, _, control = preparation
    copied = []
    for i, s in enumerate(old.segments[:72]):
        copied.append(
            SequenceSegment(
                f"new/{i:03}",
                s.segment_id,
                s.states,
                s.inputs,
                s.dt_s,
                start_row=s.start_row,
            )
        )
    duplicate = SequenceCollection(
        tuple(copied),
        configuration_id=fresh.configuration_id,
        state_channels=fresh.state_channels,
        input_channels=fresh.input_channels,
    )
    with pytest.raises(subject.IntegrityError, match="duplicates original content"):
        subject.prepare_update("crazyflow", control, old, duplicate, {})


def test_actual_preparation_persisted_before_any_numeric_entry(
    preparation, monkeypatch, tmp_path
):
    old, fresh, _, control = preparation
    prepared = subject.prepare_update("crazyflow", control, old, fresh, {})
    entered = []

    def stop(t, d, c, s, **kw):
        assert (tmp_path / "expected-preparation.npz").exists()
        assert (tmp_path / "actual-preparation.npz").exists()
        subject._check_preparation_file(tmp_path, prepared, "actual-preparation.npz")
        entered.append(kw["previous"])
        raise subject.core.SequenceFitError("synthetic interruption before numeric fit")

    monkeypatch.setattr(learner, "_train", stop)
    monkeypatch.setattr(
        subject.capture, "_configuration", lambda: {"jax_enable_x64": False}
    )
    with pytest.raises(subject.core.SequenceFitError):
        subject.perform_update("crazyflow", tmp_path, control, prepared, {})
    assert entered == [control.fingerprint()]
    timing = subject.read(tmp_path / "timing.json")
    assert timing["public_update_calls"] == timing["training_entry_calls"] == 1
    assert timing["initializer_calls"] == timing["fitter_calls"] == 0
    assert timing["original_unchanged"]


@pytest.mark.parametrize("field", ["train_future_inputs", "development_future_states"])
def test_coherently_resealed_preparation_mutant_rejected(preparation, tmp_path, field):
    old, fresh, _, control = preparation
    prepared = subject.prepare_update("crazyflow", control, old, fresh, {})
    subject.save_preparation(tmp_path, prepared)
    meta, arrays = load_arrays(tmp_path / "expected-preparation.npz")
    arrays[field][0, 0, 0] += 0.1
    save_arrays(tmp_path / "mutated.npz", meta, arrays)
    with pytest.raises(ValueError, match="byte-identical"):
        subject._check_preparation_file(tmp_path, prepared, "mutated.npz")


def decision_fixture(monkeypatch, factual=0.99, response=0.8):
    protocol = json.loads((subject.ROOT / subject.PROTOCOL_PATH).read_text())
    p = deepcopy(protocol)
    policy = p["decision"]["aggregation"]
    policy["horizons_s"] = [0.25]
    policy["scope_weights"] = {"primary": 1.0}
    p["cells"] = {s: [{"id": "p", "group": "primary"}] for s in subject.SIMULATORS}
    p["recordings"] = [
        dict(id=f"{s}/p/{i}", simulator=s, role="test", cell="p")
        for s in p["cells"]
        for i in range(2)
    ]
    rows = {}
    for sim in p["cells"]:
        values = []
        for i in range(2):
            for kind in ("factual", "response"):
                for group in policy["groups"]:
                    for arm in subject.ARMS:
                        ratio = (
                            (response if kind == "response" else factual)
                            if arm == "candidate"
                            else 1
                        )
                        mse = ((i + 1) * ratio) ** 2
                        values.append(
                            dict(
                                simulator=sim,
                                scope="primary",
                                cell="p",
                                parent=f"{sim}/p/{i}",
                                query="q-" + kind,
                                origin=1,
                                kind=kind,
                                arm=arm,
                                horizon_s=0.25,
                                horizon_steps=5,
                                group=group,
                                statistic="endpoint",
                                truth_eligible=True,
                                prediction_finite=True,
                                mse=mse,
                                finite_subset_mse=mse,
                                components=3,
                                pair_nonweak=False,
                            )
                        )
        rows[sim] = values
    monkeypatch.setattr(subject, "resolved", lambda _: deepcopy(p))
    return p, rows


def test_broad_progress_uses_same_bootstrap_and_preserves_local_losses(monkeypatch):
    p, rows = decision_fixture(monkeypatch)
    for r in rows["cascade"]:
        if (
            r["arm"] == "candidate"
            and r["group"] == "velocity_m_s"
            and r["kind"] == "response"
        ):
            r["mse"] = r["finite_subset_mse"] = (1.08 * (int(r["parent"][-1]) + 1)) ** 2
    out = subject.decide(rows, p)
    assert out["residual_criteria_pass"]
    assert out["angular_retention"]["denominator_arm"] == "baseline"
    assert (
        out["angular_retention"]["bootstrap"]["parent_draws_sha256"]
        == out["comparison"]["bootstrap"]["parent_draws_sha256"]
    )
    assert out["public_promotion"] is False


def test_angular_retention_can_fail_despite_broad_mean_progress(monkeypatch):
    p, rows = decision_fixture(monkeypatch, response=0.5)
    for r in rows["crazyflow"]:
        if (
            r["arm"] == "candidate"
            and r["group"] == "body_rate_rad_s"
            and r["kind"] == "factual"
        ):
            r["mse"] = r["finite_subset_mse"] = (
                1.051 * (int(r["parent"][-1]) + 1)
            ) ** 2
    out = subject.decide(rows, p)
    assert out["comparison"]["residual_criteria_pass"]
    assert not out["angular_retention"]["checks"]["factual_aggregate_rmse"]
    assert not out["residual_criteria_pass"]


@pytest.mark.parametrize("flag", ["all_available", "all_input_finite"])
def test_availability_and_input_prefix_finiteness_are_separate_mandatory_checks(
    monkeypatch, flag
):
    p, rows = decision_fixture(monkeypatch)
    out = subject.decide(rows, p, **{flag: False})
    assert out["comparison"]["residual_criteria_pass"]
    assert not out["residual_criteria_pass"]


@pytest.mark.parametrize(
    "mutation", ["omitted_parent", "different_mask", "missing_hold"]
)
def test_cohort_omissions_and_truth_mask_changes_rejected(monkeypatch, mutation):
    p, rows = decision_fixture(monkeypatch)
    if mutation == "omitted_parent":
        rows["crazyflow"] = [
            r for r in rows["crazyflow"] if r["parent"] != "crazyflow/p/1"
        ]
    if mutation == "different_mask":
        rows["crazyflow"][1]["truth_eligible"] = False
    if mutation == "missing_hold":
        rows["cascade"] = [r for r in rows["cascade"] if r["arm"] != "hold"]
    with pytest.raises(ValueError):
        subject.decide(rows, p)


def test_nonfinite_predictions_do_not_become_smaller_eligible_subset(monkeypatch):
    p, rows = decision_fixture(monkeypatch)
    rows["cascade"][1].update(prediction_finite=False, mse=None, finite_subset_mse=None)
    out = subject.decide(rows, p)
    assert not out["comparison"]["checks"]["finite_eligible_predictions"]
    assert not out["residual_criteria_pass"]


def test_factual_regression_bound_not_replaced_by_response_win(monkeypatch):
    p, rows = decision_fixture(monkeypatch, factual=1.051, response=0.5)
    out = subject.decide(rows, p)
    assert out["comparison"]["checks"]["response_gain"]
    assert not out["comparison"]["checks"]["factual_regression"]
    assert not out["residual_criteria_pass"]


def test_external_seal_and_full_inventory_both_required(tmp_path):
    subject.write(tmp_path / "payload.json", {"value": 1})
    subject.write(tmp_path / "run.json", {"files": subject.inventory(tmp_path)})
    sha = subject.digest(tmp_path / "run.json")
    subject.sealed(tmp_path / "run.json", sha)
    (tmp_path / "payload.json").write_text("changed")
    with pytest.raises(subject.IntegrityError, match="inventory"):
        subject.sealed(tmp_path / "run.json", sha)
    with pytest.raises(subject.IntegrityError, match="external"):
        subject.sealed(tmp_path / "run.json", "0" * 64)


def test_supervisor_preserves_timeout_and_partial_payload(monkeypatch, tmp_path):
    import subprocess

    monkeypatch.setattr(
        subject, "authenticate", lambda *a: ({}, dict(interpreter="python"))
    )

    def fail(*a, **kw):
        raise subprocess.TimeoutExpired(a[0], 14400)

    monkeypatch.setattr(subject.subprocess, "run", fail)
    out = tmp_path / "attempt"
    with pytest.raises(subprocess.TimeoutExpired):
        subject.update_candidate(
            "crazyflow",
            out,
            training_root=tmp_path,
            training_sha256="a" * 64,
            protocol_path="protocol",
            protocol_sha256=subject.PROTOCOL_SHA256,
            binding_path="binding",
            binding_sha256="b" * 64,
        )
    result = subject.read(out / "run.json")
    assert result["status"] == "hard_timeout_incomplete"
    assert (out / "request.json").exists() and (out / "worker.log").exists()
    assert result["files"] == subject.inventory(out)
    with pytest.raises(FileExistsError):
        subject.update_candidate(
            "crazyflow",
            out,
            training_root=tmp_path,
            training_sha256="a" * 64,
            protocol_path="protocol",
            protocol_sha256=subject.PROTOCOL_SHA256,
            binding_path="binding",
            binding_sha256="b" * 64,
        )


def test_finalization_cannot_exclude_external_stage_payload(monkeypatch, tmp_path):
    monkeypatch.setattr(
        subject, "authenticate", lambda *a: ({}, dict(implementation_commit="test"))
    )
    with pytest.raises(subject.IntegrityError, match="include every stage"):
        subject.finalize(
            tmp_path / "bundle",
            stages={
                "crazyflow": {
                    "candidate": {"root": str(tmp_path / "outside"), "sha256": "a" * 64}
                }
            },
            protocol_path="p",
            binding_path="b",
            binding_sha256="c" * 64,
        )


def test_unavailable_model_has_explicit_nan_slots():
    values = {"target": np.zeros((5, 15))}
    actual = subject._prediction_arrays(None, {}, values)
    assert (
        actual.dtype == np.float32
        and actual.shape == (5, 15)
        and np.isnan(actual).all()
    )


def raw_score_fixture(tmp_path, monkeypatch, *, response=False):
    data = tmp_path / "data"
    evaluation = tmp_path / "evaluation"
    data.mkdir()
    evaluation.mkdir()
    p = {
        "generation": {"crazyflow": {"dt_s": 0.05}},
        "evaluation": {
            "horizons_s": [0.05, 0.15, 0.25],
            "responses": {
                "weak_endpoint_thresholds": {
                    "velocity_m_s": 0.001,
                    "body_rate_rad_s": 0.001,
                    "rotation_entries": 0.0001,
                }
            },
        },
        "decision": {
            "aggregation": {
                "groups": ["velocity_m_s", "body_rate_rad_s", "rotation_entries"]
            }
        },
        "cells": {"crazyflow": [{"id": "c", "group": "primary"}]},
        "recordings": [
            {"id": "parent", "simulator": "crazyflow", "role": "test", "cell": "c"}
        ],
    }
    queries = []
    for sign in (-1, 1) if response else (None,):
        query = dict(
            parent="parent",
            id=str(sign),
            simulator="crazyflow",
            scope="primary",
            cell="c",
            origin=1,
            kind="response" if response else "factual",
            history_eligible=True,
            path=f"{sign}.npz",
        )
        if response:
            query.update(sign=sign, channel=0)
        queries.append(query)
        values = dict(
            past_states=np.zeros((11, 15)),
            past_inputs=np.zeros((10, 1)),
            future_inputs=np.zeros((5, 1)),
            target=np.zeros((5, 15)),
            valid=np.ones(5, dtype=bool),
        )
        predictions = {
            "baseline": np.ones((5, 15), dtype=np.float32),
            "candidate": np.full((5, 15), 0.5, dtype=np.float32),
        }
        if response:
            values.update(
                factual_inputs=np.zeros((5, 1)), factual_target=np.zeros((5, 15))
            )
            predictions.update(
                baseline_factual=np.zeros((5, 15), dtype=np.float32),
                candidate_factual=np.zeros((5, 15), dtype=np.float32),
            )
        np.savez_compressed(data / query["path"], **values)
        path = evaluation / subject.physical.query_path(query)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **predictions)
    np.savez_compressed(
        evaluation / "envelopes.npz",
        baseline=np.ones((5, 15)),
        candidate=np.ones((5, 15)),
    )
    monkeypatch.setattr(subject.physical, "_planned", lambda *a: ([], queries, {}))
    return data, evaluation, p, queries


def test_three_arm_raw_scoring_preserves_every_horizon_and_group(tmp_path, monkeypatch):
    data, evaluation, p, queries = raw_score_fixture(tmp_path, monkeypatch)
    result = subject.score(data, evaluation, "crazyflow", p)
    assert len(result["rows"]) == 3 * 3 * 2 * 3
    assert {row["arm"] for row in result["rows"]} == set(subject.ARMS)
    assert all(
        row["mse"] == pytest.approx(0.25)
        for row in result["rows"]
        if row["arm"] == "candidate"
    )
    assert all(row["mse"] == 0 for row in result["rows"] if row["arm"] == "hold")
    subject.validate_rows(result["rows"], queries, p, "crazyflow")
    with pytest.raises(subject.IntegrityError, match="metric slots"):
        subject.validate_rows(result["rows"][:-1], queries, p, "crazyflow")


def test_response_subtraction_is_float64_after_native32_storage(tmp_path, monkeypatch):
    data, evaluation, p, queries = raw_score_fixture(
        tmp_path, monkeypatch, response=True
    )
    for query in queries:
        path = evaluation / subject.physical.query_path(query)
        arrays = subject.physical.arrays(path)
        arrays["candidate"][:] = np.float32(2e38)
        arrays["candidate_factual"][:] = np.float32(-2e38)
        np.savez_compressed(path, **arrays)
    result = subject.score(data, evaluation, "crazyflow", p)
    selected = [row for row in result["rows"] if row["arm"] == "candidate"]
    assert selected and all(
        row["prediction_finite"] and np.isfinite(row["mse"]) for row in selected
    )
    expected = (float(np.float32(2e38)) - float(np.float32(-2e38))) ** 2
    assert selected[0]["mse"] == pytest.approx(expected)
    assert all(row["pair_nonweak"] is False for row in selected)


def test_prediction_array_omission_is_integrity_failure(tmp_path, monkeypatch):
    data, evaluation, p, queries = raw_score_fixture(
        tmp_path, monkeypatch, response=True
    )
    path = evaluation / subject.physical.query_path(queries[0])
    arrays = subject.physical.arrays(path)
    arrays.pop("candidate_factual")
    np.savez_compressed(path, **arrays)
    with pytest.raises(subject.IntegrityError, match="arm roster"):
        subject.score(data, evaluation, "crazyflow", p)


def test_missing_later_truth_keeps_all_planned_slots(tmp_path, monkeypatch):
    data, evaluation, p, queries = raw_score_fixture(tmp_path, monkeypatch)
    path = data / queries[0]["path"]
    arrays = subject.physical.arrays(path)
    arrays["valid"][3:] = False
    arrays["target"][3:] = np.nan
    np.savez_compressed(path, **arrays)
    result = subject.score(data, evaluation, "crazyflow", p)
    assert len(result["rows"]) == 54
    assert all(
        not row["truth_eligible"] for row in result["rows"] if row["horizon_s"] == 0.25
    )
    assert all(
        row["truth_eligible"]
        for row in result["rows"]
        if row["horizon_s"] in (0.05, 0.15)
    )


@pytest.fixture
def work_links(preparation, monkeypatch, tmp_path):
    old, fresh, _, control = preparation
    prepared = subject.prepare_update("crazyflow", control, old, fresh, {})
    subject.save_preparation(tmp_path, prepared)
    meta, arrays = load_arrays(tmp_path / "expected-preparation.npz")
    save_arrays(tmp_path / "actual-preparation.npz", meta, arrays)
    norms = subject.saved.training_norms(prepared.expected_train.batch)
    initial = {
        **{
            "param_" + name: np.array([float(i)])
            for i, name in enumerate(subject.INITIAL_SUBSET)
        },
        **{"norm_" + k: v for k, v in norms.items()},
    }
    initial_model = SimpleNamespace(arrays=lambda: initial)
    monkeypatch.setattr(subject, "_initial_model", lambda *a: initial_model)
    np.savez_compressed(tmp_path / "initializer-return.npz", **initial)
    np.savez_compressed(tmp_path / "external-checkpoint.npz", **initial)
    b = prepared.expected_train.batch
    hold = np.repeat(b.past_states[:, -1:], 5, 1)
    scale = np.maximum(
        np.sqrt(np.mean((hold - b.future_states) ** 2, 0)), 0.01 * norms["state_scale"]
    )
    pred = b.future_states + scale * np.array([1.0, 2.0])
    mse = np.mean(((pred - b.future_states) / scale) ** 2, axis=(0, 1))
    raw = 1 / np.maximum(mse, 0.0001)
    normalizer = np.mean(raw)
    w = raw / normalizer
    weights = dict(
        normalization=scale,
        initial_training_prediction=pred,
        initial_channel_mse=mse,
        raw_channel_weights=raw,
        channel_weights=w,
        fixed_weights=w,
        weight_floor=np.asarray(0.0001),
        weight_normalizer=np.asarray(normalizer),
    )
    np.savez_compressed(tmp_path / "weights.npz", **weights)
    protocol = {
        "fitting": {"recipe": deepcopy(learner.RECIPE)},
        "imported_evidence": {
            "controls": {
                "crazyflow": {
                    "checkpoint0": {
                        "path": str(tmp_path / "external-checkpoint.npz"),
                        "sha256": subject.digest(tmp_path / "external-checkpoint.npz"),
                    }
                }
            }
        },
    }
    model = SimpleNamespace(
        _train=prepared.expected_train,
        _development=prepared.development,
        _seen=prepared.seen,
        _contract=prepared.contract,
        _model=SimpleNamespace(norms=norms, arrays=lambda: initial),
        report={
            "previous_revision": control.fingerprint(),
            "recipe": deepcopy(learner.RECIPE),
            "optimization": {
                "steps": 1000,
                "batch_size": 1536,
                "ridge": 0.01 * 1536 * 5,
                "delay_steps": 2,
                "error_scale": scale.tolist(),
                "selection_objective": subject.core.OBJECTIVE,
                "objective": subject.core.weighting_metadata(
                    initial_model, pred, mse, raw, w, 0.0001, normalizer
                ),
            },
        },
    )
    calls = []

    def retained_work(directory, actual, witness):
        assert actual is model
        assert witness.ridge == 0.01 * 1536 * 5 and witness.delay == 2
        subject.capture._same_tree(witness.initial_arrays, initial, "actual initial")
        subject.capture._same_tree(witness.objective_arrays, weights, "actual weights")
        calls.append(True)
        return {"unchanged_work_helper": True}

    monkeypatch.setattr(subject.capture, "_work", retained_work)
    return tmp_path, prepared, control, protocol, model, calls


def test_within_fit_witness_uses_actual_horizon_ridge_and_external_initial_subset(
    work_links,
):
    directory, prepared, control, p, model, calls = work_links
    assert subject.check_work(directory, model, prepared, control, p, "crazyflow") == {
        "unchanged_work_helper": True
    }
    assert calls == [True]


def test_wrong_initialization_prior_subset_fails_before_scalar_work(work_links):
    directory, prepared, control, p, model, calls = work_links
    arrays = subject.capture._npz(directory / "initializer-return.npz")
    arrays["param_w1"] += 0.1
    np.savez_compressed(directory / "initializer-return.npz", **arrays)
    with pytest.raises(ValueError, match="data-independent initializer"):
        subject.check_work(directory, model, prepared, control, p, "crazyflow")
    assert not calls


def test_changed_fixed_weights_fail_independent_arithmetic_reduction(work_links):
    directory, prepared, control, p, model, calls = work_links
    weights = subject.capture._npz(directory / "weights.npz")
    weights["fixed_weights"][0] += 0.1
    np.savez_compressed(directory / "weights.npz", **weights)
    with pytest.raises(ValueError, match="arithmetic objective"):
        subject.check_work(directory, model, prepared, control, p, "crazyflow")
    assert not calls


def test_wrong_parent_revision_is_not_a_qualified_update(work_links):
    directory, prepared, control, p, model, calls = work_links
    model.report["previous_revision"] = "different"
    with pytest.raises(subject.IntegrityError, match="previous revision"):
        subject.check_work(directory, model, prepared, control, p, "crazyflow")
    assert not calls


@pytest.mark.parametrize(
    "field,value",
    [
        ("steps", 999),
        ("batch_size", 384),
        ("ridge", 1.0),
        ("delay_steps", 1),
        ("selection_objective", "unweighted"),
    ],
)
def test_optimizer_metadata_cannot_drift_from_actual_frozen_work(
    work_links, field, value
):
    directory, prepared, control, p, model, calls = work_links
    model.report["optimization"][field] = value
    with pytest.raises(ValueError):
        subject.check_work(directory, model, prepared, control, p, "crazyflow")
    assert not calls


def test_reported_hold_scale_is_independently_derived(work_links):
    directory, prepared, control, p, model, calls = work_links
    model.report["optimization"]["error_scale"][0][0] += 0.1
    with pytest.raises(ValueError, match="reported hold scales"):
        subject.check_work(directory, model, prepared, control, p, "crazyflow")
    assert not calls


def test_coherent_query_identity_mutation_is_rejected(tmp_path, monkeypatch):
    data, evaluation, p, queries = raw_score_fixture(tmp_path, monkeypatch)
    rows = subject.score(data, evaluation, "crazyflow", p)["rows"]
    for row in rows:
        row["origin"] = 2
    with pytest.raises(subject.IntegrityError, match="metric identity"):
        subject.validate_rows(rows, queries, p, "crazyflow")


def test_evaluation_telemetry_owns64_and_restores_native32(monkeypatch, tmp_path):
    from glassbox.experimental import independent_training_data as data

    assert not subject.jax.config.x64_enabled
    observations = []
    model = SimpleNamespace(
        predict=lambda *args: None,
        envelope=lambda: np.ones(15),
        fingerprint=lambda: "toy",
    )
    data_root = tmp_path / "data"
    output = tmp_path / "evaluation"
    data_root.mkdir()
    output.mkdir()
    subject.write(data_root / "queries.json", [])
    monkeypatch.setattr(data, "verify_data", lambda *a, **kw: {})
    monkeypatch.setattr(subject, "verify_candidate", lambda *a, **kw: (model, {}))
    monkeypatch.setattr(subject, "import_control", lambda *a: (model, {}))
    monkeypatch.setattr(subject, "resolved", lambda p: {})
    monkeypatch.setattr(
        subject.physical,
        "reconstruct_truth",
        lambda *a: observations.append(("telemetry", subject.jax.config.x64_enabled)),
    )

    def jit(function):
        observations.append(("jit", subject.jax.config.x64_enabled))
        return function

    monkeypatch.setattr(subject.jax, "jit", jit)
    monkeypatch.setattr(subject, "score", lambda *a: {})
    outcome = subject._evaluation_worker(
        dict(
            output=str(output),
            simulator="crazyflow",
            data_root=str(data_root),
            data_sha256="data",
            candidate_root="candidate",
            candidate_sha256="candidate",
        ),
        {},
        {},
    )
    assert observations == [("telemetry", True), ("jit", False), ("jit", False)]
    assert outcome["native_forecast_calls"] == 0
    assert not subject.jax.config.x64_enabled


@pytest.mark.parametrize(
    "key,value",
    [
        ("protocol_sha256", "wrong"),
        ("implementation_sha256", "wrong"),
        ("binding_sha256", "wrong"),
    ],
)
def test_stage_source_association_requires_current_binding(key, value):
    row = dict(
        protocol_sha256=subject.PROTOCOL_SHA256,
        implementation_sha256="current",
        binding_sha256="current",
    )
    subject._stage_source(row, "current", "toy")
    row[key] = value
    with pytest.raises(subject.IntegrityError, match="source association"):
        subject._stage_source(row, "current", "toy")


@pytest.mark.parametrize(
    "field,value",
    [
        ("path", "/different/candidate/run.json"),
        ("sha256", "different"),
        ("status", "fit_failed"),
    ],
)
def test_confirmation_binds_exact_both_candidate_outcomes(tmp_path, field, value):
    stages, expected = {}, {}
    for simulator in subject.SIMULATORS:
        directory = tmp_path / simulator
        directory.mkdir()
        subject.write(directory / "run.json", dict(status="complete", files={}))
        sha = subject.digest(directory / "run.json")
        stages[simulator] = {"candidate": dict(root=str(directory), sha256=sha)}
        expected[simulator] = dict(
            path=str(directory / "run.json"), sha256=sha, status="complete"
        )
    subject._confirmation_associations(expected, stages)
    expected["cascade"][field] = value
    with pytest.raises(subject.IntegrityError, match="exact candidate outcome"):
        subject._confirmation_associations(expected, stages)


@pytest.mark.parametrize(
    "field,value",
    [
        ("public_update_calls", 2),
        ("training_entry_calls", 0),
        ("initializer_calls", True),
        ("fitter_calls", 2),
        ("calibration_calls", 0),
        ("original_unchanged", False),
        ("configuration_after", {"jax_enable_x64": True, "environment_sha256": "same"}),
    ],
)
def test_single_update_saved_timing_witness(field, value):
    configuration = dict(jax_enable_x64=False, environment_sha256="same")
    timing = dict(
        public_update_calls=1,
        training_entry_calls=1,
        initializer_calls=1,
        fitter_calls=1,
        calibration_calls=1,
        original_unchanged=True,
        configuration_before=configuration,
        configuration_after=configuration,
    )
    subject._check_timing(timing)
    timing[field] = value
    with pytest.raises(subject.IntegrityError):
        subject._check_timing(timing)


def test_complete_stage_links_bind_collected_data_and_both_candidate_outcomes(
    monkeypatch, tmp_path
):
    from glassbox.experimental import independent_training_data as data

    source = dict(
        protocol_sha256=subject.PROTOCOL_SHA256,
        implementation_sha256="bound",
        binding_sha256="bound",
    )
    stages = {}
    for simulator in subject.SIMULATORS:
        parent = tmp_path / simulator
        candidate = parent / "candidate"
        candidate.mkdir(parents=True)
        subject.write(
            candidate / "request.json",
            dict(
                **source,
                simulator=simulator,
                training_root=str(parent / "training/data"),
                training_sha256="pending",
            ),
        )
        stages[simulator] = {"candidate": {"root": str(candidate)}}
        for kind in ("training", "confirmation"):
            root = parent / kind / "data"
            root.mkdir(parents=True)
            subject.write(root / "queries.json", [])
            stages[simulator][kind] = {"root": str(root)}

    def seal_data(simulator, kind, prerequisites):
        entry = stages[simulator][kind]
        root = subject.Path(entry["root"])
        subject.write(
            root / "seal.json", dict(**source, candidate_outcomes=prerequisites)
        )
        entry["sha256"] = subject.digest(root / "seal.json")
        subject.write(root.parent / "request.json", dict(**source, directory=str(root)))
        subject.write(
            root.parent / "run.json",
            dict(
                **source,
                format=data.STAGE_FORMAT,
                status="complete",
                kind=kind,
                simulator=simulator,
                data_sha256=entry["sha256"],
                files=subject.inventory(root.parent),
            ),
        )

    prerequisites = {}
    for simulator in subject.SIMULATORS:
        seal_data(simulator, "training", {})
        root = subject.Path(stages[simulator]["candidate"]["root"])
        request = subject.read(root / "request.json")
        request["training_sha256"] = stages[simulator]["training"]["sha256"]
        (root / "request.json").write_text(json.dumps(request))
        subject.write(
            root / "run.json",
            dict(**source, status="complete", files=subject.inventory(root)),
        )
        sha = subject.digest(root / "run.json")
        stages[simulator]["candidate"]["sha256"] = sha
        prerequisites[simulator] = dict(
            path=str(root / "run.json"), sha256=sha, status="complete"
        )
    for simulator in subject.SIMULATORS:
        seal_data(simulator, "confirmation", prerequisites)
        root = tmp_path / simulator / "evaluation"
        root.mkdir()
        subject.write(
            root / "request.json",
            dict(
                **source,
                candidate_root=stages[simulator]["candidate"]["root"],
                candidate_sha256=stages[simulator]["candidate"]["sha256"],
                data_root=stages[simulator]["confirmation"]["root"],
                data_sha256=stages[simulator]["confirmation"]["sha256"],
            ),
        )
        subject.write(root / "metrics.json", {"rows": []})
        subject.write(
            root / "outcome.json",
            dict(
                candidate_available=True,
                native_forecast_calls=0,
                all_input_eligible_predictions_finite=True,
            ),
        )
        subject.write(
            root / "run.json",
            dict(
                **source,
                status="complete",
                simulator=simulator,
                files=subject.inventory(root),
            ),
        )
        stages[simulator]["evaluation"] = dict(
            root=str(root), sha256=subject.digest(root / "run.json")
        )
    monkeypatch.setattr(
        data,
        "verify_data",
        lambda root, *a, **kw: subject.read(subject.Path(root) / "seal.json"),
    )
    monkeypatch.setattr(subject, "verify_candidate", lambda *a, **kw: (object(), {}))
    monkeypatch.setattr(subject, "resolved", lambda p: {})
    monkeypatch.setattr(subject, "score", lambda *a: {"rows": []})
    assert subject._checked_stages(stages, {}, "bound") == (
        {s: [] for s in subject.SIMULATORS},
        True,
        True,
    )
    with pytest.raises(subject.IntegrityError, match="source association"):
        subject._checked_stages(stages, {}, "another-binding")


def test_audit_scalar_late_truth_and_nonfinite():
    target = np.zeros((3, 15))
    prediction = np.ones_like(target)
    prediction[0, 0] = np.nan
    valid = np.array([True, True, False])
    assert subject._audit_scalar(
        prediction, target, valid, 2, slice(0, 3), "endpoint"
    ) == (True, True, 1.0)
    assert subject._audit_scalar(
        prediction, target, valid, 2, slice(0, 3), "cumulative"
    ) == (True, False, None)
    assert subject._audit_scalar(
        prediction, target, valid, 3, slice(0, 3), "endpoint"
    ) == (False, False, None)


def test_audit_hierarchy_weights_parents_not_queries():
    rows = [
        dict(cell="c", parent="a", origin=0, truth_eligible=True, mse=1.0),
        dict(cell="c", parent="a", origin=0, truth_eligible=True, mse=3.0),
        dict(cell="c", parent="b", origin=0, truth_eligible=True, mse=8.0),
    ]
    assert subject._audit_hierarchy(rows) == pytest.approx(np.sqrt(5))
    rows[0]["mse"] = None
    assert subject._audit_hierarchy(rows) is None
    rows[0]["truth_eligible"] = False
    assert subject._audit_hierarchy(rows) == pytest.approx(np.sqrt(5.5))


def test_audit_rejection_requires_named_semantic_failure():
    def wrong():
        raise ValueError("environment failure")

    with pytest.raises(subject.IntegrityError, match="wrong semantic"):
        subject._audit_rejection(wrong, "cache mismatch")
    with pytest.raises(subject.IntegrityError, match="alteration was accepted"):
        subject._audit_rejection(lambda: None, "cache mismatch")


def test_independent_saved_reduction_repeated_query_ids(tmp_path, monkeypatch):
    original, data_root, evaluation = (
        tmp_path / n for n in ("bundle", "data", "evaluation")
    )
    for p in (original, data_root, evaluation):
        p.mkdir()
    queries, rows = [], []
    for parent, error in (("one", 1.0), ("two", 3.0)):
        query = dict(
            parent=parent,
            id="factual-0001",
            path=parent + ".npz",
            kind="factual",
            history_eligible=True,
            scope="primary",
            cell="c",
            origin=1,
        )
        queries.append(query)
        values = dict(
            target=np.zeros((1, 15)),
            valid=np.ones(1, bool),
            past_states=np.zeros((2, 15)),
            past_inputs=np.zeros((2, 1)),
            future_inputs=np.zeros((1, 1)),
        )
        np.savez(data_root / query["path"], **values)
        predicted = dict(
            baseline=np.full((1, 15), error, np.float32),
            candidate=np.full((1, 15), error / 2, np.float32),
        )
        np.savez(evaluation / (parent + ".npz"), **predicted)
        for arm in subject.ARMS:
            forecast = predicted.get(arm, np.zeros((1, 15)))
            for r in subject.score_forecast(
                forecast,
                values["target"],
                values["valid"],
                dt_s=0.05,
                horizons_s=[0.05],
            ):
                rows.append(
                    dict(
                        r,
                        simulator="toy",
                        scope="primary",
                        cell="c",
                        parent=parent,
                        query=query["id"],
                        origin=1,
                        kind="factual",
                        arm=arm,
                    )
                )
    summary = subject.aggregate(rows)
    subject.write(data_root / "queries.json", queries)
    subject.write(evaluation / "metrics.json", dict(rows=rows, summary=summary))
    policy = dict(
        floors={"velocity_m_s": 0.01},
        simulator_weights={"toy": 1.0},
        scope_weights={"primary": 1.0},
    )
    comparison_row = dict(
        simulator="toy",
        scope="primary",
        kind="factual",
        horizon_s=0.05,
        group="velocity_m_s",
        baseline_rmse=np.sqrt(5),
        candidate_rmse=np.sqrt(5) / 2,
        ratio=0.5,
    )
    tail = dict(
        simulator="toy",
        scope="primary",
        kind="factual",
        horizon_s=0.05,
        group="velocity_m_s",
        statistic="endpoint",
        arm="baseline",
        p50=2.0,
        p95=2.9,
        max=3.0,
        planned_parents=2,
        available_parents=2,
    )
    decision = dict(
        comparison=dict(
            physical_comparisons=[comparison_row],
            weighted_geometric_mean_ratios={"factual": 0.5},
            parent_error_tails=[tail],
        )
    )
    subject.write(original / "decision.json", decision)
    subject.write(original / "run.json", {"files": subject.inventory(original)})
    source = dict(
        stages={
            "toy": {
                "confirmation": {"root": str(data_root)},
                "evaluation": {"root": str(evaluation)},
            }
        }
    )
    monkeypatch.setattr(subject, "verify_bundle", lambda *a, **kw: source)
    monkeypatch.setattr(subject, "authenticate", lambda **kw: ({}, {}))
    monkeypatch.setattr(
        subject, "resolved", lambda p: {"decision": {"aggregation": policy}}
    )
    monkeypatch.setattr(subject.physical, "query_path", lambda q: q["parent"] + ".npz")
    result = subject.independent_reduce(
        original, subject.digest(original / "run.json"), audit_output=tmp_path / "audit"
    )
    assert result["passed"] and result["counts"]["raw_rows"] == 36
    assert result["counts"]["parents"] == 36
    # A coherently resaved metric from the wrong parent cannot pass the raw reducer.
    wrong = deepcopy(rows)
    wrong[0]["mse"] = 9.0
    (evaluation / "metrics.json").unlink()
    subject.write(evaluation / "metrics.json", dict(rows=wrong, summary=summary))
    with pytest.raises(
        subject.IntegrityError, match="independent reduction differs: raw"
    ):
        subject.independent_reduce(
            original,
            subject.digest(original / "run.json"),
            audit_output=tmp_path / "bad",
        )


def test_audit_local_reseal_preserves_external_anchor(tmp_path):
    root = tmp_path / "bundle"
    stage = root / "stage"
    stage.mkdir(parents=True)
    subject.write(stage / "payload.json", {"value": 1})
    subject.write(stage / "run.json", {"files": subject.inventory(stage)})
    subject.write(root / "run.json", {"files": subject.inventory(root)})
    original = subject.digest(root / "run.json")
    subject._audit_replace(stage / "payload.json", {"value": 2})
    changed = subject._audit_reseal(root)
    assert changed != original
    subject.sealed(root / "run.json", changed)
    subject.sealed(stage / "run.json", subject.digest(stage / "run.json"))
    with pytest.raises(subject.IntegrityError, match="external stage SHA"):
        subject.sealed(root / "run.json", original)


def test_audit_actual_cache_message_is_existing_semantic_boundary(
    tmp_path, monkeypatch
):
    expected = {"train_future_inputs": np.zeros((2, 3, 1))}
    altered = {"train_future_inputs": np.ones((2, 3, 1))}
    subject.save_arrays(tmp_path / "actual-preparation.npz", {}, altered)
    monkeypatch.setattr(subject.capture, "_cache_payload", lambda *a: ({}, expected))
    prepared = SimpleNamespace(
        expected_train=None, development=None, contract=None, seen=None
    )
    message = subject._audit_rejection(
        lambda: subject._check_preparation_file(
            tmp_path, prepared, "actual-preparation.npz"
        ),
        "saved actual preparation:train_future_inputs is not byte-identical",
    )
    assert "train_future_inputs" in message


def test_independent_reduction_rejects_destination_inside_science(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(subject, "verify_bundle", lambda *a, **kw: {})
    monkeypatch.setattr(subject, "authenticate", lambda **kw: ({}, {}))
    monkeypatch.setattr(subject, "resolved", lambda p: {})
    with pytest.raises(subject.IntegrityError, match="outside original"):
        subject.independent_reduce(
            tmp_path, "external", audit_output=tmp_path / "unsafe"
        )
    assert not (tmp_path / "unsafe").exists()
