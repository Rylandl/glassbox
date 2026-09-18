"""Small deterministic boundary checks; no simulator collection or learner fits."""

from copy import deepcopy

import numpy as np
import pytest

from glassbox.experimental import state_input_experiment as experiment
from glassbox.experimental import two_simulator_flight as parent


def tiny_queries(tmp_path):
    entry = next(
        r
        for r in experiment.protocol()["recordings"]
        if r["simulator"] == "cascade" and r["role"] == "test"
    )
    queries = []
    for name, kind, sign in (
        ("factual", "factual", None),
        ("lower", "response", -1),
        ("upper", "response", 1),
    ):
        q = dict(
            id=name,
            kind=kind,
            parent=entry["id"],
            simulator="cascade",
            scope="primary",
            cell=entry["cell"],
            origin=20,
            history_eligible=True,
            path=f"queries/{name}.npz",
        )
        if sign is not None:
            q.update(channel=0, sign=sign)
        arrays = dict(
            past_states=np.zeros((11, 15)),
            past_inputs=np.zeros((10, 3)),
            future_inputs=np.full((5, 3), 0.0 if sign is None else sign * 0.1),
            factual_inputs=np.zeros((5, 3)),
            target=np.zeros((5, 15)),
            factual_target=np.zeros((5, 15)),
            valid=np.ones(5, dtype=bool),
        )
        if sign is not None:
            arrays["target"][:, :3] = sign * 0.1 * np.arange(1, 6)[:, None]
        experiment.save_arrays(tmp_path / "cascade/data" / q["path"], arrays)
        queries.append(q)
    experiment.write_json(tmp_path / "cascade/data/queries.json", queries)
    return queries


class ToyPredictors:
    def __init__(self, *_):
        pass

    def predict(self, query, arrays):
        motion = np.zeros((5, 15))
        motion[:, :3] = np.cumsum(arrays["future_inputs"], axis=0)
        return {"baseline": motion, "candidate": 2 * motion, "hold": 0 * motion}

    def envelope(self, arm):
        return None if arm == "hold" else np.ones((5, 15))


def toy_evaluation(monkeypatch, tmp_path):
    queries = tiny_queries(tmp_path)
    monkeypatch.setattr(experiment, "check_links", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(experiment, "Predictors", ToyPredictors)
    for path in ("data", "baseline", "candidate"):
        experiment.write_json(tmp_path / "cascade" / path / "seal.json", {})
    experiment.evaluate("cascade", tmp_path)
    return queries


def test_private_parent_binding_preserves_historical_module():
    original = parent.PROTOCOL
    private = experiment.flight()
    assert private is not parent
    assert private.PROTOCOL == experiment.PROTOCOL
    assert parent.PROTOCOL == original
    assert private.generate.__globals__ is private.__dict__
    assert parent.generate.__globals__ is parent.__dict__
    assert experiment.protocol()["id"] == "state-input-interaction-v1"


def test_frozen_calibration_and_fresh_seed_rosters():
    new = experiment.protocol()
    old = parent.read_json(parent.PROTOCOL)
    assert new["generation"] == old["generation"]
    assert new["cells"] == old["cells"]
    for key in (
        "factual_origins_s",
        "first_step",
        "horizons_s",
        "metrics",
        "population",
    ):
        assert new["evaluation"][key] == old["evaluation"][key]
    for key in (
        "commands",
        "future_duration_s",
        "origins_s",
        "truth",
        "weak_endpoint_thresholds",
    ):
        assert (
            new["evaluation"]["responses"][key] == old["evaluation"]["responses"][key]
        )
    for actual, previous in zip(new["recordings"], old["recordings"], strict=True):
        if previous["role"] == "calibration_pool":
            assert actual == previous
        else:
            expected = deepcopy(previous)
            expected["seed"] += 1000000
            expected["id"] = (
                previous["id"].rsplit("-", 1)[0] + "-" + str(expected["seed"])
            )
            assert actual == expected
    for name, digest in new["parent_source_sha256"].items():
        assert experiment.digest(experiment.ROOT / name) == digest


def test_evaluation_subtracts_each_arm_factual_and_replays(monkeypatch, tmp_path):
    queries = toy_evaluation(monkeypatch, tmp_path)
    path = tmp_path / "cascade/evaluation" / queries[-1]["parent"] / "upper.npz"
    arrays = experiment.load_arrays(path)
    assert set(arrays) == {
        name
        for arm in experiment.PREDICTORS
        for name in (arm, arm + "_factual", arm + "_response")
    }
    np.testing.assert_array_equal(
        arrays["candidate_response"], 2 * arrays["baseline_response"]
    )
    assert not arrays["hold_response"].any()
    report = experiment.evaluate("cascade", tmp_path, replay=True)
    assert report == {"prediction_queries": 3, "metric_rows": 162, "exact": True}
    rows = experiment.read_json(tmp_path / "cascade/evaluation/rows.json")
    assert all("pair_nonweak" in r for r in rows if r["kind"] == "response")
    assert all(r["mse"] == 0 for r in rows if r["arm"] == "baseline")


def test_replay_rejects_coherently_rewritten_prediction(monkeypatch, tmp_path):
    queries = toy_evaluation(monkeypatch, tmp_path)
    path = tmp_path / "cascade/evaluation" / queries[-1]["parent"] / "upper.npz"
    arrays = experiment.load_arrays(path)
    arrays["candidate"][0, 0] += 0.5
    experiment.save_arrays(path, arrays)
    with pytest.raises(ValueError, match="fresh array bytes differ"):
        experiment.evaluate("cascade", tmp_path, replay=True)


def test_rows_require_every_planned_slot_even_when_both_arms_missing(
    monkeypatch, tmp_path
):
    toy_evaluation(monkeypatch, tmp_path)
    rows = experiment.read_json(tmp_path / "cascade/evaluation/rows.json")
    removed = [r for r in rows if r["query"] != "lower"]
    with pytest.raises(ValueError, match="complete planned roster"):
        experiment.validate_rows("cascade", tmp_path, removed)
    duplicate = rows[:-1] + [deepcopy(rows[0])]
    with pytest.raises(ValueError, match="duplicate or unplanned"):
        experiment.validate_rows("cascade", tmp_path, duplicate)
    shifted = deepcopy(rows)
    shifted[0]["origin"] += 1
    with pytest.raises(ValueError, match="query identity"):
        experiment.validate_rows("cascade", tmp_path, shifted)


def test_missing_history_and_partial_command_tail_preserve_slots():
    predictors = experiment.Predictors.__new__(experiment.Predictors)
    predictors.functions = {
        "baseline": lambda x, up, uf: np.zeros((len(uf), 15)),
        "candidate": None,
    }
    arrays = dict(
        past_states=np.ones((11, 15)),
        past_inputs=np.zeros((10, 3)),
        future_inputs=np.zeros((5, 3)),
    )
    missing = predictors.predict({"history_eligible": False}, arrays)
    assert all(np.isnan(value).all() for value in missing.values())
    arrays["future_inputs"][3:] = np.nan
    partial = predictors.predict({"history_eligible": True}, arrays)
    assert np.isfinite(partial["baseline"][:3]).all()
    assert np.isnan(partial["baseline"][3:]).all()
    assert np.isnan(partial["candidate"]).all()
    assert (partial["hold"][:3] == 1).all()


def test_only_known_numerical_failure_messages_are_fit_outcomes():
    assert experiment._nonfinite_fit_error(
        ValueError("initial recursive validation loss is nonfinite")
    )
    assert experiment._nonfinite_fit_error(
        ValueError("nonfinite sequence training at step 300")
    )
    assert not experiment._nonfinite_fit_error(ValueError("invalid error_scale"))
    assert not experiment._nonfinite_fit_error(
        RuntimeError("nonfinite sequence training at step 300")
    )


def test_external_anchor_cannot_be_replaced_by_local_rehash(tmp_path):
    experiment.write_json(tmp_path / "run.json", {"files": {}})
    old_digest = experiment.digest(tmp_path / "run.json")
    experiment.write_json(tmp_path / "run.json", {"files": {}, "changed": True})
    with pytest.raises(ValueError, match="externally trusted"):
        experiment.verify_bundle(tmp_path, old_digest)
