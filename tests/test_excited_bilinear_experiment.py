"""Small deterministic boundary checks; no simulator collection or learner fits."""

from copy import deepcopy

import numpy as np
import pytest

from glassbox.experimental import excited_bilinear_experiment as experiment
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
            "hold": 0 * motion,
        }

    def envelope(self, arm):
        return None if arm == "hold" else np.ones((5, 15))


def toy_evaluation(monkeypatch, tmp_path):
    queries = tiny_queries(tmp_path)
    monkeypatch.setattr(experiment, "check_links", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(experiment, "Predictors", ToyPredictors)
    for path in ("data", "baseline", "quadratic", "candidate"):
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
    assert experiment.protocol()["id"] == "excited-bilinear-ablation-v1"


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
            expected["seed"] += 5000000
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
    assert report == {"prediction_queries": 3, "metric_rows": 216, "exact": True}
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
    assert private.ARMS == ("baseline", "quadratic", "candidate")
    assert private.PREDICTORS == ("baseline", "quadratic", "candidate", "hold")
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


class Revision:
    def __init__(self):
        self.metadata = {
            "contract": {"dt": 0.05},
            "report": {"step": 0},
            "windows": ["x"],
        }
        self.arrays = {"param_x": np.ones(3), "train_future_states": np.ones((2, 3))}
        self.identity = "unchanged"

    def _metadata(self):
        return self.metadata

    def _arrays(self):
        return self.arrays

    def fingerprint(self):
        return self.identity


@pytest.mark.parametrize("arm", ["baseline", "quadratic"])
@pytest.mark.parametrize("change", ["parameter", "cache", "report", "fingerprint"])
def test_comparator_mismatch_is_rejected(arm, change):
    old, current = Revision(), Revision()
    result = experiment._same_revision(current, old, arm)
    assert result["all_saved_arrays_and_metadata_exact"]
    if change == "parameter":
        current.arrays["param_x"][0] += 1
    elif change == "cache":
        current.arrays["train_future_states"][0, 0] += 1
    elif change == "report":
        current.metadata["report"]["step"] = 100
    else:
        current.identity = "changed"
    with pytest.raises(ValueError):
        experiment._same_revision(current, old, arm)


def test_fit_provenance_binds_imported_data_without_quadratic_dependency(tmp_path):
    base = tmp_path / "cascade"
    for name in (
        "data/seal.json",
        "calibration.json",
        "baseline/report.json",
        "excitation/seal.json",
        "reuse.json",
    ):
        experiment.write_json(base / name, {})
    experiment.freeze_files(base / "baseline", "seal.json", {})
    runtime = {"revision": "current"}
    provenance = {
        arm: experiment.fit_provenance("cascade", arm, tmp_path, runtime)
        for arm in experiment.ARMS
    }
    for value in provenance.values():
        assert value["runtime"] == runtime
        assert value["imported_excitation_seal"] == experiment.digest(
            base / "excitation/seal.json"
        )
        assert value["reuse_sha256"] == experiment.digest(base / "reuse.json")
        assert (
            value["trusted_excitation_bundle_sha256"]
            == experiment.protocol()["prior_excitation_bundle"]["manifest_sha256"]
        )
        assert value["fitting_collection"] == "excitation"
        assert value["checkpoint_steps"] == list(range(0, 1001, 100))
        assert (
            value["trusted_architecture_bundle_sha256"]
            == experiment.protocol()["prior_architecture_bundle"]["manifest_sha256"]
        )
    assert "baseline_seal" not in provenance["baseline"]
    assert (
        provenance["candidate"]["baseline_seal"]
        == provenance["quadratic"]["baseline_seal"]
    )
    assert "quadratic_seal" not in provenance["candidate"]
    assert provenance["quadratic"]["trusted_preparation_reference_arm"] == "original"
    experiment.write_json(base / "baseline/report.json", {"changed": True})
    for arm in ("quadratic", "candidate"):
        with pytest.raises(ValueError, match="altered artifact"):
            experiment.fit_provenance("cascade", arm, tmp_path, runtime)


def test_prior_source_and_protocol_pins_reject_changes(monkeypatch):
    p = experiment.protocol()
    experiment._check_inherited_sources(p)
    original = experiment.digest
    paths = [
        p[key]["path"]
        for key in (
            "prior_interaction_protocol",
            "prior_quadratic_protocol",
            "prior_excitation_protocol",
            "prior_architecture_protocol",
        )
    ]
    paths.extend(p["inherited_source_sha256"])
    for path in paths:
        target = experiment.ROOT / path
        monkeypatch.setattr(
            experiment,
            "digest",
            lambda value, target=target: (
                "changed" if value == target else original(value)
            ),
        )
        with pytest.raises(ValueError, match="changed"):
            experiment._check_inherited_sources(p)
    monkeypatch.setattr(experiment, "digest", original)


def test_both_imports_checked_before_any_fit(monkeypatch, tmp_path):
    runtime = {"source": "pinned"}
    monkeypatch.setattr(
        experiment._implementation(), "data_ready", lambda output: runtime
    )
    calls = []
    monkeypatch.setattr(
        experiment,
        "check_import",
        lambda sim, output, r: calls.append((sim, output, r)),
    )
    assert experiment.data_ready(tmp_path) == runtime
    assert calls == [(sim, tmp_path, runtime) for sim in experiment.SIMULATORS]


def test_saved_reuse_evidence_must_match_verified_import(monkeypatch, tmp_path):
    from glassbox.experimental import excited_public_data as data

    runtime = {"source": "current"}
    calls = []

    def verify(*args):
        calls.append(args)
        return {"current_runtime": runtime, "imported": True}

    monkeypatch.setattr(data, "check_import", verify)
    experiment.write_json(tmp_path / "cascade/reuse.json", verify())
    experiment.check_import("cascade", tmp_path, runtime)
    assert calls[-1] == (
        "cascade",
        tmp_path,
        experiment.flight(),
        experiment.protocol(),
        runtime,
    )
    experiment.write_json(tmp_path / "cascade/reuse.json", {"imported": True})
    with pytest.raises(ValueError, match="reuse evidence differs"):
        experiment.check_import("cascade", tmp_path, runtime)


def test_candidate_matching_uses_trusted_same_data_reference(monkeypatch):
    from glassbox.experimental import excited_bilinear_matching as matching

    public, candidate, trusted = object(), object(), object()
    monkeypatch.setattr(experiment, "trusted_baseline_model", lambda sim: public)
    monkeypatch.setattr(experiment, "trusted_quadratic_model", lambda sim: trusted)
    calls = []

    def check(*args):
        calls.append(args)
        return {"same_data": True}

    monkeypatch.setattr(matching, "check_matched_models", check)
    assert experiment.compare_candidate(candidate, "cascade") == {"same_data": True}
    assert calls == [(public, candidate, trusted)]


def test_links_keep_candidate_validation_when_fresh_quadratic_unavailable(
    monkeypatch, tmp_path
):
    from types import SimpleNamespace

    monkeypatch.setattr(
        experiment._implementation(), "_original_check_links", lambda *a, **k: None
    )
    monkeypatch.setattr(experiment, "check_sources", lambda: {})
    monkeypatch.setattr(experiment, "check_import", lambda *a: None)
    models = {arm: SimpleNamespace(report={}, contract={}) for arm in experiment.ARMS}
    monkeypatch.setattr(
        experiment,
        "_model_classes",
        lambda: {
            arm: SimpleNamespace(load=lambda path, arm=arm: models[arm])
            for arm in experiment.ARMS
        },
    )
    checked = []
    for arm in experiment.ARMS:
        monkeypatch.setattr(
            experiment,
            "compare_" + arm,
            lambda model, sim, arm=arm: checked.append(arm) or {},
        )
        for filename, value in (
            ("outcome.json", {"model_available": arm != "quadratic"}),
            ("report.json", {}),
            ("contract.json", {}),
            ("matching.json", {}),
        ):
            experiment.write_json(tmp_path / "cascade" / arm / filename, value)
    checkpoint_calls = []
    monkeypatch.setattr(
        experiment, "check_checkpoints", lambda *a: checkpoint_calls.append(a[2])
    )
    experiment.check_links(tmp_path, "cascade")
    assert checkpoint_calls == list(experiment.ARMS)
    assert checked == ["baseline", "candidate"]
    experiment.write_json(
        tmp_path / "cascade/candidate/matching.json", {"changed": True}
    )
    with pytest.raises(ValueError, match="matching evidence differs"):
        experiment.check_links(tmp_path, "cascade")


def fake_fit_environment(monkeypatch, tmp_path):
    from contextlib import contextmanager
    from types import SimpleNamespace

    from glassbox.experimental import fit_checkpoints as checkpoints
    from glassbox.learner import LearnedDynamics

    collection, train, development, inner = object(), object(), object(), object()
    calls = {"collection": [], "capture": [], "save": []}
    model = SimpleNamespace(
        report={"optimization": {"selected_step": 0}},
        contract={},
        _train=train,
        _development=development,
        _model=inner,
        save=lambda path: experiment.save_arrays(path, {"x": np.ones(1)}),
    )

    @contextmanager
    def capture(arm, provenance, **kwargs):
        calls["capture"].append((arm, provenance, kwargs))
        yield "captured"

    def save(path, captured, **kwargs):
        calls["save"].append((path, captured, kwargs))
        experiment.write_json(path / "diagnostics.json", {"failure": kwargs["failure"]})
        sha = experiment.freeze_files(path, "manifest.json", {})
        return {"manifest_sha256": sha}

    def collect(path):
        calls["collection"].append(path)
        return collection, []

    monkeypatch.setattr(checkpoints, "capture_fit", capture)
    monkeypatch.setattr(checkpoints, "save_checkpoints", save)
    monkeypatch.setattr(
        experiment,
        "data_ready",
        lambda output: {
            "bilinear_ablation_implementation": {"observer": "sha"},
            "checkpoint_source_sha256": {"observer": "sha"},
        },
    )
    monkeypatch.setattr(experiment, "fit_provenance", lambda *a: {"stage": a[1]})
    monkeypatch.setattr(experiment, "trusted_baseline_model", lambda sim: model)
    monkeypatch.setattr(experiment.flight(), "collection_and_trajectories", collect)
    monkeypatch.setattr(LearnedDynamics, "load", lambda path: model)
    for arm in experiment.ARMS:
        monkeypatch.setattr(experiment, "compare_" + arm, lambda *a: {"exact": True})
    return collection, model, calls


def test_public_excited_fit_calls_plain_public_entrypoint_once(monkeypatch, tmp_path):
    import glassbox

    collection, model, calls = fake_fit_environment(monkeypatch, tmp_path)
    fitted = []
    monkeypatch.setattr(glassbox, "fit", lambda *a, **k: fitted.append((a, k)) or model)
    experiment.fit_arm("cascade", "baseline", tmp_path)
    assert fitted == [((collection,), {})]
    assert calls["collection"] == [tmp_path / "cascade/excitation"]
    assert calls["capture"] == [
        (
            "baseline",
            {"stage": "baseline"},
            {
                "source_sha256": {"observer": "sha"},
                "expected_steps": tuple(range(0, 1001, 100)),
            },
        )
    ]
    assert calls["save"][0][1] == "captured"
    assert calls["save"][0][2]["train"] is model._train
    assert calls["save"][0][2]["development"] is model._development
    assert calls["save"][0][2]["selected_model"] is model._model
    assert calls["save"][0][2]["selected_step"] == 0
    directory = tmp_path / "cascade/baseline"
    outcome = experiment.read_json(directory / "outcome.json")
    assert outcome["model_available"] and outcome["status"] == "complete"
    assert outcome["checkpoint_manifest_sha256"] == experiment.digest(
        directory / "checkpoints/manifest.json"
    )
    assert (
        "checkpoints/diagnostics.json"
        in experiment.verify_files(directory, "seal.json")["files"]
    )


def test_bilinear_fit_consumes_current_baseline_once_without_preparation_refit(
    monkeypatch, tmp_path
):
    import glassbox
    from glassbox.experimental import state_input_model as bilinear

    _, model, calls = fake_fit_environment(monkeypatch, tmp_path)
    experiment.write_json(
        tmp_path / "cascade/baseline/outcome.json", {"model_available": True}
    )
    fitted = []
    monkeypatch.setattr(
        bilinear, "fit_candidate", lambda *a, **k: fitted.append((a, k)) or model
    )
    monkeypatch.setattr(
        glassbox, "fit", lambda *a: pytest.fail("unexpected public preparation refit")
    )
    experiment.fit_arm("cascade", "candidate", tmp_path)
    assert fitted == [((model,), {})]
    assert calls["collection"] == []
    assert calls["capture"][0][0] == "candidate"
    assert experiment.read_json(tmp_path / "cascade/candidate/outcome.json")[
        "model_available"
    ]


def test_unavailable_baseline_records_unattempted_bilinear_preparation(
    monkeypatch, tmp_path
):
    from glassbox.experimental import state_input_model as bilinear

    _, _, calls = fake_fit_environment(monkeypatch, tmp_path)
    experiment.write_json(
        tmp_path / "cascade/baseline/outcome.json", {"model_available": False}
    )
    monkeypatch.setattr(
        bilinear,
        "fit_candidate",
        lambda *a: pytest.fail("unavailable preparation cannot be fitted"),
    )
    experiment.fit_arm("cascade", "candidate", tmp_path)
    outcome = experiment.read_json(tmp_path / "cascade/candidate/outcome.json")
    assert outcome["status"] == "preparation_unavailable"
    assert not outcome["model_available"]
    assert "not attempted" in outcome["reason"]
    assert calls["save"][0][2]["failure"] == outcome["reason"]
    assert calls["save"][0][2]["selected_model"] is None


@pytest.mark.parametrize("arm", ["baseline", "candidate", "quadratic"])
def test_only_expected_numeric_fit_errors_are_recorded(monkeypatch, tmp_path, arm):
    import glassbox
    from glassbox.experimental import command_excitation_fit as excited
    from glassbox.experimental import state_input_model as bilinear
    from glassbox.experimental import state_quadratic_model as quadratic

    _, _, calls = fake_fit_environment(monkeypatch, tmp_path)
    experiment.write_json(
        tmp_path / "cascade/baseline/outcome.json", {"model_available": True}
    )
    if arm == "baseline":
        # fit_arm creates its own baseline directory, unlike dependent arms.
        (tmp_path / "cascade/baseline/outcome.json").unlink()
        (tmp_path / "cascade/baseline").rmdir()
    error = {
        "baseline": ValueError,
        "candidate": bilinear.CandidateFitError,
        "quadratic": quadratic.CandidateFitError,
    }[arm]

    def fail(*a, **k):
        raise error("nonfinite sequence training at step 300")

    monkeypatch.setattr(glassbox, "fit", fail)
    monkeypatch.setattr(bilinear, "fit_candidate", fail)
    monkeypatch.setattr(excited, "fit_excited", fail)
    monkeypatch.setattr(quadratic.CandidateDynamics, "load", lambda path: object())
    monkeypatch.setattr(experiment, "trusted_excitation", lambda: tmp_path)
    experiment.fit_arm("cascade", arm, tmp_path)
    outcome = experiment.read_json(tmp_path / "cascade" / arm / "outcome.json")
    assert outcome["status"] == "fit_failure" and not outcome["model_available"]
    assert calls["save"][0][2]["failure"] == outcome["reason"]
    assert calls["save"][0][2]["selected_step"] is None


def test_programming_failure_aborts_without_publishing_fit_outcome(
    monkeypatch, tmp_path
):
    import glassbox

    _, _, calls = fake_fit_environment(monkeypatch, tmp_path)

    def fail(*a):
        raise ValueError("invalid error_scale")

    monkeypatch.setattr(glassbox, "fit", fail)
    with pytest.raises(ValueError, match="invalid error_scale"):
        experiment.fit_arm("cascade", "baseline", tmp_path)
    assert not calls["save"]
    assert not (tmp_path / "cascade/baseline/outcome.json").exists()


def test_checkpoint_validation_uses_sealed_manifest_and_trusted_caches(
    monkeypatch, tmp_path
):
    from glassbox.experimental import fit_checkpoints as checkpoints

    _, model, _ = fake_fit_environment(monkeypatch, tmp_path)
    monkeypatch.setattr(
        experiment,
        "check_sources",
        lambda: {
            "bilinear_ablation_implementation": {"observer": "sha"},
            "checkpoint_source_sha256": {"observer": "sha"},
        },
    )
    directory = tmp_path / "cascade/candidate"
    experiment.write_json(
        directory / "outcome.json",
        {
            "model_available": False,
            "checkpoint_manifest_sha256": "anchor",
            "status": "fit_failure",
            "reason": "nonfinite sequence training at step 300",
        },
    )
    calls = []
    monkeypatch.setattr(
        checkpoints,
        "check_checkpoint_links",
        lambda *a, **k: calls.append(("links", a, k)) or {"exact": True},
    )
    monkeypatch.setattr(
        checkpoints,
        "replay_checkpoints",
        lambda *a, **k: calls.append(("replay", a, k)) or {"exact": True},
    )
    assert experiment.check_checkpoints(tmp_path, "cascade", "candidate") == {
        "exact": True
    }
    assert experiment.check_checkpoints(
        tmp_path, "cascade", "candidate", replay=True
    ) == {"exact": True}
    assert [x[0] for x in calls] == ["links", "replay"]
    for _, args, kwargs in calls:
        assert args == (directory / "checkpoints",)
        assert kwargs["expected_sha256"] == "anchor"
        assert (
            kwargs["train"] is model._train
            and kwargs["development"] is model._development
        )
        assert kwargs["provenance"] == {"stage": "candidate"}
        assert kwargs["source_sha256"] == {"observer": "sha"}
        assert kwargs["selected_model"] is None
        assert kwargs["expectation"] == dict(
            arm="candidate",
            recipe=experiment._expected_recipe("candidate"),
            expected_steps=list(range(0, 1001, 100)),
            completed=False,
            failure="nonfinite sequence training at step 300",
            selected_step=None,
            trace=None,
            status="fit_failure",
        )


def test_model_arm_classes_are_distinct_existing_recipes():
    from glassbox.experimental.state_input_model import CandidateDynamics as Bilinear
    from glassbox.experimental.state_quadratic_model import (
        CandidateDynamics as Quadratic,
    )
    from glassbox.learner import LearnedDynamics

    assert experiment._model_classes() == {
        "baseline": LearnedDynamics,
        "candidate": Bilinear,
        "quadratic": Quadratic,
    }


def test_trusted_architecture_requires_external_anchor(monkeypatch, tmp_path):
    p = deepcopy(experiment.protocol())
    p["prior_architecture_bundle"]["path"] = str(tmp_path)
    experiment.write_json(
        tmp_path / "run.json",
        {"files": {}, "protocol_sha256": p["prior_architecture_protocol"]["sha256"]},
    )
    p["prior_architecture_bundle"]["manifest_sha256"] = experiment.digest(
        tmp_path / "run.json"
    )
    monkeypatch.setattr(experiment, "protocol", lambda: p)
    assert experiment.trusted_architecture() == tmp_path
    value = experiment.read_json(tmp_path / "run.json")
    value["modified"] = True
    experiment.write_json(tmp_path / "run.json", value)
    with pytest.raises(ValueError, match="trusted anchor"):
        experiment.trusted_architecture()


def test_checkpoint_source_map_covers_new_and_inherited_numerical_modules():
    p = experiment.protocol()
    added = experiment._ablation_sources()
    sources = experiment._checkpoint_sources(p, added)
    assert added.items() <= sources.items()
    for name in (
        "src/glassbox/_sequence_model.py",
        "src/glassbox/experimental/state_input_model.py",
        "src/glassbox/experimental/state_quadratic_model.py",
        "src/glassbox/experimental/fit_checkpoints.py",
    ):
        assert sources[name] == experiment.digest(experiment.ROOT / name)
    with pytest.raises(ValueError, match="conflicting checkpoint"):
        experiment._checkpoint_sources(
            p, {"src/glassbox/_sequence_model.py": "changed"}
        )
