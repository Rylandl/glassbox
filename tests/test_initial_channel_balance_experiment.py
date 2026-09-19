"""Small deterministic boundary checks; no simulator collection or learner fits."""

from copy import deepcopy

import numpy as np
import pytest

from glassbox.experimental import initial_channel_balance_experiment as experiment
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
        return {
            "baseline": motion,
            "quadratic": 1.5 * motion,
            "candidate": 2 * motion,
            "anchored": 1.75 * motion,
            "hold": 0 * motion,
        }

    def envelope(self, arm):
        return None if arm == "hold" else np.ones((5, 15))


def toy_evaluation(monkeypatch, tmp_path):
    queries = tiny_queries(tmp_path)
    monkeypatch.setattr(experiment, "check_links", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(experiment, "Predictors", ToyPredictors)
    for path in ("data", "baseline", "quadratic", "anchored", "candidate"):
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
    assert experiment.protocol()["id"] == "initial-channel-balance-v1"


def test_frozen_calibration_and_fresh_seed_rosters():
    new = experiment.protocol()
    old = parent.read_json(parent.PROTOCOL)
    assert {k: v for k, v in new["generation"].items() if k != "admission"} == {
        k: v for k, v in old["generation"].items() if k != "admission"
    }
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
            expected["seed"] += 7000000
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
    assert report == {"prediction_queries": 3, "metric_rows": 270, "exact": True}
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
        "quadratic": lambda x, up, uf: np.ones((len(uf), 15)),
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
    assert experiment.inherited._nonfinite_fit_error(
        ValueError("initial recursive validation loss is nonfinite")
    )
    assert experiment.inherited._nonfinite_fit_error(
        ValueError("nonfinite sequence training at step 300")
    )
    assert not experiment.inherited._nonfinite_fit_error(
        ValueError("invalid error_scale")
    )
    assert not experiment.inherited._nonfinite_fit_error(
        RuntimeError("nonfinite sequence training at step 300")
    )


def test_external_anchor_cannot_be_replaced_by_local_rehash(tmp_path):
    experiment.write_json(tmp_path / "run.json", {"files": {}})
    old_digest = experiment.digest(tmp_path / "run.json")
    experiment.write_json(tmp_path / "run.json", {"files": {}, "changed": True})
    with pytest.raises(ValueError, match="externally trusted"):
        experiment.verify_bundle(tmp_path, old_digest)


def test_historical_interaction_globals_are_unchanged():
    old = experiment.inherited
    before = (old.PROTOCOL, old.PROTOCOL_SHA256, old.ARMS, old.PREDICTORS)
    private = experiment._implementation()
    assert private is not old
    assert private.evaluate.__globals__ is private.__dict__
    assert old.evaluate.__globals__ is old.__dict__
    assert (old.PROTOCOL, old.PROTOCOL_SHA256, old.ARMS, old.PREDICTORS) == before
    assert private.ARMS == ("baseline", "quadratic", "anchored", "candidate")
    assert private.PREDICTORS == (
        "baseline",
        "quadratic",
        "anchored",
        "candidate",
        "hold",
    )
    assert old.protocol()["id"] == "state-input-interaction-v1"
    assert old.flight().PROTOCOL == old.PROTOCOL
    assert experiment.flight() is not old.flight()


def test_missing_original_rows_and_nonfinite_predictions_are_not_dropped(
    monkeypatch, tmp_path
):
    actual_predictors = experiment.Predictors
    toy_evaluation(monkeypatch, tmp_path)
    rows = experiment.read_json(tmp_path / "cascade/evaluation/rows.json")
    assert {r["arm"] for r in rows} == set(experiment.PREDICTORS)
    with pytest.raises(ValueError, match="complete planned roster"):
        experiment.validate_rows(
            "cascade", tmp_path, [r for r in rows if r["arm"] != "quadratic"]
        )
    predictors = actual_predictors.__new__(actual_predictors)
    predictors.functions = {arm: None for arm in experiment.ARMS}
    prediction = predictors.predict(
        {"history_eligible": True},
        {
            "future_inputs": np.zeros((5, 3)),
            "past_states": np.zeros((11, 15)),
        },
    )
    assert set(prediction) == set(experiment.PREDICTORS)
    assert all(np.isnan(prediction[arm]).all() for arm in experiment.ARMS)
    assert np.isfinite(prediction["hold"]).all()


def reference_fixture(monkeypatch, tmp_path):
    """Small byte-level historical imports; no physical model or fitting run."""
    source, output = tmp_path / "source", tmp_path / "output"
    references = {}
    for arm, source_arm in experiment.REFERENCE_ARMS.items():
        directory = source / "cascade" / source_arm
        experiment.write_json(directory / "model.npz", {"saved_arm": source_arm})
        for name, value in (
            ("report.json", {"source_arm": source_arm}),
            ("contract.json", {"dt": 0.05}),
            ("start.json", {"arm": source_arm, "runtime": "historical"}),
            (
                "outcome.json",
                {"status": "complete", "model_available": True, "reason": None},
            ),
            ("checkpoints/manifest.json", {"arm": source_arm}),
        ):
            experiment.write_json(directory / name, value)
        experiment.freeze_files(directory, "seal.json", {"historical": True})
        files = {
            str(p.relative_to(directory)): experiment.digest(p)
            for p in directory.rglob("*")
            if p.is_file()
        }
        references[arm] = dict(
            source_arm=source_arm,
            directory=directory,
            files=files,
            anchors={"revision_fingerprint": arm + "-fingerprint"},
        )

    class FakeModel:
        def __init__(self, arm):
            self.arm = arm
            self.report = {"source_arm": experiment.REFERENCE_ARMS[arm]}
            self.contract = {"dt": 0.05}

        def fingerprint(self):
            return self.arm + "-fingerprint"

    def loader(arm):
        class Loader:
            @staticmethod
            def load(_):
                return FakeModel(arm)

        return Loader

    classes = {arm: loader(arm) for arm in experiment.REFERENCE_ARMS}
    monkeypatch.setattr(experiment, "_model_classes", lambda: classes)
    manifest = {"runtime": {"source": "historical150"}}
    monkeypatch.setattr(
        experiment, "_reference_sources", lambda _: (source, manifest, references)
    )
    for name in ("data/seal.json", "excitation/seal.json", "reuse.json"):
        experiment.write_json(output / "cascade" / name, {"current": name})
    runtime = {"source": "current161"}
    return source, output, runtime, references


def test_reference_import_preserves_old_arm_and_runtime(monkeypatch, tmp_path):
    source, output, runtime, references = reference_fixture(monkeypatch, tmp_path)
    result = experiment.import_references("cascade", output, runtime)
    assert result["fresh_reference_fits"] == 0
    assert result["historical_runtime"] != result["runtime"]
    assert result["arms"]["anchored"]["source_arm"] == "candidate"
    assert (
        experiment.read_json(output / "cascade/anchored/start.json")["arm"]
        == "candidate"
    )
    for arm, entry in references.items():
        for name in entry["files"]:
            assert (output / "cascade" / arm / name).read_bytes() == (
                source / "cascade" / entry["source_arm"] / name
            ).read_bytes()
    assert experiment.check_reference_import("cascade", output, runtime) == result


def test_duplicate_reference_import_never_overwrites(monkeypatch, tmp_path):
    _, output, runtime, _ = reference_fixture(monkeypatch, tmp_path)
    experiment.import_references("cascade", output, runtime)
    before = (output / "cascade/anchored/start.json").read_bytes()
    with pytest.raises(FileExistsError, match="already exists"):
        experiment.import_references("cascade", output, runtime)
    assert (output / "cascade/anchored/start.json").read_bytes() == before


def test_coherent_reference_reseal_cannot_replace_external_bytes(monkeypatch, tmp_path):
    _, output, runtime, _ = reference_fixture(monkeypatch, tmp_path)
    experiment.import_references("cascade", output, runtime)
    base = output / "cascade/anchored"
    experiment.write_json(base / "model.npz", {"fabricated": True})
    seal = experiment.read_json(base / "seal.json")
    seal["files"]["model.npz"] = experiment.digest(base / "model.npz")
    experiment.write_json(base / "seal.json", seal)
    with pytest.raises(ValueError, match="imported reference bytes differ"):
        experiment.check_reference_import("cascade", output, runtime)


@pytest.mark.parametrize("kind", ["extra", "symlink", "association", "current_runtime"])
def test_reference_import_rejects_unbound_changes(monkeypatch, tmp_path, kind):
    _, output, runtime, _ = reference_fixture(monkeypatch, tmp_path)
    experiment.import_references("cascade", output, runtime)
    base = output / "cascade"
    if kind == "extra":
        (base / "baseline/extra.txt").write_text("unsealed")
    elif kind == "symlink":
        (base / "baseline/link").symlink_to(base / "baseline/model.npz")
    elif kind == "association":
        manifest = experiment.read_json(base / "reference-reuse.json")
        manifest["arms"]["anchored"]["source_arm"] = "quadratic"
        experiment.write_json(base / "reference-reuse.json", manifest)
    else:
        runtime = {"source": "different"}
    with pytest.raises(ValueError):
        experiment.check_reference_import("cascade", output, runtime)


def test_only_new_candidate_can_be_fitted(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("data read or optimizer should not run")

    monkeypatch.setattr(experiment, "data_ready", forbidden)
    for arm in (*experiment.REFERENCE_ARMS, "unknown"):
        with pytest.raises(ValueError, match="only candidate"):
            experiment.fit_arm("cascade", arm, tmp_path)


def test_outcome_status_is_strict():
    for value in (0, 1, "true"):
        with pytest.raises(ValueError, match="availability/status/reason"):
            experiment._outcome_expectation(
                dict(model_available=value, status="complete", reason=None),
                None,
                "candidate",
                {},
            )
    failed = experiment._outcome_expectation(
        dict(model_available=False, status="fit_failure", reason="nonfinite"),
        None,
        "candidate",
        {},
    )
    assert failed["completed"] is False
    assert failed["trace"] is None


def test_reference_checks_keep_historical_binding(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from glassbox.experimental import affine_anchored_checkpoints as old_checkpoints

    path = tmp_path / "cascade/anchored"
    outcome = dict(
        status="complete",
        model_available=True,
        reason=None,
        checkpoint_manifest_sha256="outer",
    )
    experiment.write_json(path / "outcome.json", outcome)
    experiment.write_json(
        tmp_path / "historical/run.json", {"runtime": {"historical": True}}
    )
    model = SimpleNamespace(
        report={"optimization": {"selected_step": 0, "trace": [{"step": 0}]}}
    )
    monkeypatch.setattr(
        experiment,
        "_model_classes",
        lambda: {"anchored": SimpleNamespace(load=lambda _: model)},
    )
    monkeypatch.setattr(experiment, "trusted_anchored", lambda: tmp_path / "historical")

    def arguments(sim, arm, source, runtime, loaded):
        assert sim == "cascade" and arm == "candidate"
        assert source == tmp_path / "historical" and runtime == {"historical": True}
        assert loaded is model
        return dict(provenance="historical", source_sha256={"old": "sha"})

    monkeypatch.setattr(experiment.previous, "_checkpoint_arguments", arguments)
    monkeypatch.setattr(
        experiment.previous, "_expected_recipe", lambda arm: {"original": arm}
    )

    def checker(directory, **kwargs):
        assert directory == path / "checkpoints"
        assert kwargs["provenance"] == "historical"
        assert kwargs["source_sha256"] == {"old": "sha"}
        assert kwargs["expectation"]["arm"] == "candidate"
        assert kwargs["expectation"]["recipe"] == {"original": "candidate"}
        assert "unweighted_trace" not in kwargs["expectation"]
        return {"historical_binding": True}

    monkeypatch.setattr(old_checkpoints, "check_checkpoint_links", checker)
    assert experiment.check_checkpoints(tmp_path, "cascade", "anchored") == {
        "historical_binding": True
    }


def test_candidate_checkpoint_report_fields_bind_to_observer(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from glassbox.experimental import initial_channel_balance_checkpoints as observer
    from glassbox.experimental.initial_channel_balance_model import RECIPE

    optimization = dict(
        selected_step=100,
        trace=[{"step": 0}, {"step": 100}],
        unweighted_development_trace=[{"old": 1}],
        objective={"actual_weights": [0.5, 1.5]},
    )
    model = SimpleNamespace(report={"optimization": optimization}, _model="selected")
    public = SimpleNamespace(_train="training", _development="development")
    experiment.write_json(
        tmp_path / "cascade/candidate/outcome.json",
        dict(
            status="complete",
            model_available=True,
            reason=None,
            checkpoint_manifest_sha256="outer",
        ),
    )
    monkeypatch.setattr(
        experiment,
        "_model_classes",
        lambda: {
            "candidate": SimpleNamespace(load=lambda _: model),
            "baseline": SimpleNamespace(load=lambda _: public),
        },
    )
    monkeypatch.setattr(
        experiment,
        "check_sources",
        lambda: {"checkpoint_source_sha256": {"new": "161"}},
    )
    monkeypatch.setattr(experiment, "fit_provenance", lambda *args: {"current": True})
    monkeypatch.setattr(
        experiment, "anchor_reference", lambda *args: {"step0": "anchored"}
    )

    def checker(path, **arguments):
        assert path == tmp_path / "cascade/candidate/checkpoints"
        assert arguments["train"] == "training"
        assert arguments["development"] == "development"
        assert arguments["source_sha256"] == {"new": "161"}
        assert arguments["selected_model"] == "selected"
        assert arguments["anchor_reference"] == {"step0": "anchored"}
        expectation = arguments["expectation"]
        assert expectation["recipe"] == RECIPE
        assert expectation["trace"] is optimization["trace"]
        assert (
            expectation["unweighted_trace"]
            is optimization["unweighted_development_trace"]
        )
        assert expectation["objective"] is optimization["objective"]
        return {"bindings": True}

    monkeypatch.setattr(observer, "check_checkpoint_links", checker)
    assert experiment.check_checkpoints(tmp_path, "cascade", "candidate") == {
        "bindings": True
    }
