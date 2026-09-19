"""Protocol, matched truth, import provenance and stage boundary regression checks."""

import copy
import json
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from test_excited_bilinear_decision import fail_arm, set_scale
from test_initial_channel_balance_decision import fixture as parent_fixture

from glassbox._learner_arrays import array_fingerprint
from glassbox.experimental import initial_channel_balance_experiment as previous
from glassbox.experimental import safeguarded_adam_experiment as adapters
from glassbox.experimental import safeguarded_adam_experiment as experiment
from glassbox.experimental import two_simulator_flight as original
from glassbox.experimental.two_simulator_metrics import aggregate


def test_frozen_delta_resolves_only_fresh_test_seeds_and_current_mechanism():
    current, base, first = (
        experiment.protocol(),
        previous.protocol(),
        original.read_json(original.PROTOCOL),
    )
    assert current["id"] == "safeguarded-adam-v1"
    assert experiment.digest(experiment.PROTOCOL) == experiment.PROTOCOL_SHA256
    assert current["fitting"]["generic_recipe"] == base["fitting"]["generic_recipe"]
    for field in experiment.delta_protocol()["resolution"]["inherit_unchanged"]:
        if field != "inherited_source_sha256":
            assert current[field] == base[field]
    for row, old, initial in zip(
        current["recordings"], base["recordings"], first["recordings"], strict=True
    ):
        if row["role"] == "calibration_pool":
            assert row == old == initial
        else:
            expected = deepcopy(initial)
            expected["seed"] += 8000000
            expected["id"] = (
                initial["id"].rsplit("-", 1)[0] + "-" + str(expected["seed"])
            )
            assert row == expected
            assert row["seed"] == old["seed"] + 1000000
    assert set(current["decision"]["comparisons"]) == {
        "safeguarded_public_progress",
        "balanced_progress",
        "quadratic_gain_retention",
    }
    assert current["decision"]["angular_retention"]["denominator"] == "balanced"
    assert (
        current["decision"]["comparisons"]["balanced_progress"][
            "response_weighted_geometric_mean_ratio_max"
        ]
        == 0.95
    )


def test_resolved_protocol_is_reconstructed_and_alteration_rejected(tmp_path):
    sha = experiment.resolved_protocol(tmp_path, create=True)
    assert sha == experiment.resolved_digest()
    value = experiment.read_json(tmp_path / "resolved-protocol.json")
    value["fitting"]["optimizer"]["scales"][1] = 0.75
    experiment.write_json(tmp_path / "resolved-protocol.json", value)
    with pytest.raises(ValueError, match="resolved protocol"):
        experiment.resolved_protocol(tmp_path)


def test_private_bindings_do_not_change_historical_entrypoints():
    old = (previous.PROTOCOL, previous.ARMS, previous.PREDICTORS, original.PROTOCOL)
    module = experiment._implementation()
    assert module is not experiment.inherited
    assert module.evaluate.__globals__ is module.__dict__
    assert module.ARMS == experiment.ARMS
    assert module.PREDICTORS == experiment.PREDICTORS
    assert experiment.flight() is not original
    assert experiment.flight().PROTOCOL == experiment.PROTOCOL
    assert (
        previous.PROTOCOL,
        previous.ARMS,
        previous.PREDICTORS,
        original.PROTOCOL,
    ) == old


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
        query = dict(
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
            query.update(channel=0, sign=sign)
        arrays = dict(
            past_states=np.zeros((11, 15)),
            past_inputs=np.zeros((10, 3)),
            future_inputs=np.full((5, 3), 0.0 if sign is None else 0.1 * sign),
            factual_inputs=np.zeros((5, 3)),
            target=np.zeros((5, 15)),
            factual_target=np.zeros((5, 15)),
            valid=np.ones(5, dtype=bool),
        )
        if sign is not None:
            arrays["target"][:, :3] = sign * 0.1 * np.arange(1, 6)[:, None]
        experiment.save_arrays(tmp_path / "cascade/data" / query["path"], arrays)
        queries.append(query)
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
            "balanced": 1.75 * motion,
            "candidate": 2 * motion,
            "hold": 0 * motion,
        }

    def envelope(self, arm):
        return None if arm == "hold" else np.ones((5, 15))


def toy_evaluation(monkeypatch, tmp_path):
    queries = tiny_queries(tmp_path)
    monkeypatch.setattr(experiment, "check_links", lambda *_a, **_k: None)
    monkeypatch.setattr(experiment, "Predictors", ToyPredictors)
    for name in ("data", *experiment.ARMS):
        experiment.write_json(tmp_path / "cascade" / name / "seal.json", {})
    experiment.evaluate("cascade", tmp_path)
    return queries


def test_each_arm_response_uses_its_own_factual_prediction_and_replays(
    monkeypatch, tmp_path
):
    queries = toy_evaluation(monkeypatch, tmp_path)
    values = experiment.load_arrays(
        tmp_path / "cascade/evaluation" / queries[-1]["parent"] / "upper.npz"
    )
    assert set(values) == {
        key
        for arm in experiment.PREDICTORS
        for key in (arm, arm + "_factual", arm + "_response")
    }
    np.testing.assert_array_equal(
        values["candidate_response"], 2 * values["baseline_response"]
    )
    assert experiment.evaluate("cascade", tmp_path, replay=True) == dict(
        prediction_queries=3, metric_rows=270, exact=True
    )
    rows = experiment.read_json(tmp_path / "cascade/evaluation/rows.json")
    assert {row["arm"] for row in rows} == set(experiment.PREDICTORS)
    with pytest.raises(ValueError, match="complete planned roster"):
        experiment.validate_rows(
            "cascade", tmp_path, [r for r in rows if r["arm"] != "balanced"]
        )


def test_replay_rejects_changed_saved_forecast(monkeypatch, tmp_path):
    queries = toy_evaluation(monkeypatch, tmp_path)
    path = tmp_path / "cascade/evaluation" / queries[-1]["parent"] / "upper.npz"
    arrays = experiment.load_arrays(path)
    arrays["candidate"][0, 0] += 0.5
    experiment.save_arrays(path, arrays)
    with pytest.raises(ValueError, match="fresh array bytes differ"):
        experiment.evaluate("cascade", tmp_path, replay=True)


def reference_fixture(monkeypatch, tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    references = {}
    classes = {}
    for arm, old_arm in experiment.REFERENCE_ARMS.items():
        directory = source / "cascade" / old_arm
        report = {"old_arm": old_arm}
        contract = {"dt_s": 0.05}
        for name, value in [
            ("model.npz", {"saved": old_arm}),
            ("report.json", report),
            ("contract.json", contract),
            ("start.json", {"arm": old_arm, "runtime": "historical"}),
            (
                "outcome.json",
                dict(status="complete", model_available=True, reason=None),
            ),
            ("checkpoints/manifest.json", {"old_arm": old_arm}),
        ]:
            experiment.write_json(directory / name, value)
        experiment.freeze_files(directory, "seal.json", {"historical": True})
        files = {
            str(f.relative_to(directory)): experiment.digest(f)
            for f in directory.rglob("*")
            if f.is_file()
        }
        references[arm] = dict(
            source_arm=old_arm,
            directory=directory,
            files=files,
            anchors={"revision_fingerprint": arm},
        )

        def loader(arm, report, contract):
            class Model:
                @staticmethod
                def load(_):
                    return Model()

                def fingerprint(self):
                    return arm

            Model.report = report
            Model.contract = contract
            return Model

        classes[arm] = loader(arm, report, contract)
    monkeypatch.setattr(experiment, "_model_classes", lambda: classes)
    monkeypatch.setattr(
        experiment,
        "_reference_sources",
        lambda _: (source, {"runtime": "historical"}, references),
    )
    for path in ("data/seal.json", "excitation/seal.json", "reuse.json"):
        experiment.write_json(output / "cascade" / path, {"current": path})
    experiment.resolved_protocol(output, create=True)
    return source, output, {"current": "runtime"}, references


def test_reference_import_keeps_historical_candidate_identity(monkeypatch, tmp_path):
    _source, output, runtime, refs = reference_fixture(monkeypatch, tmp_path)
    result = experiment.import_references("cascade", output, runtime)
    assert result["fresh_reference_fits"] == 0
    assert result["historical_runtime"] != result["runtime"]
    assert result["arms"]["balanced"]["source_arm"] == "candidate"
    assert (
        experiment.read_json(output / "cascade/balanced/start.json")["arm"]
        == "candidate"
    )
    for arm, item in refs.items():
        for name in item["files"]:
            assert (output / "cascade" / arm / name).read_bytes() == (
                item["directory"] / name
            ).read_bytes()
    with pytest.raises(FileExistsError):
        experiment.import_references("cascade", output, runtime)


@pytest.mark.parametrize("mode", ["changed", "added", "relabelled", "association"])
def test_reference_import_rejects_altered_or_relabelled_evidence(
    monkeypatch, tmp_path, mode
):
    _, output, runtime, _ = reference_fixture(monkeypatch, tmp_path)
    experiment.import_references("cascade", output, runtime)
    if mode == "association":
        path = output / "cascade/reference-reuse.json"
        d = experiment.read_json(path)
        d["arms"]["balanced"]["source_arm"] = "balanced"
        experiment.write_json(path, d)
    else:
        path = (
            output
            / "cascade/balanced"
            / (
                {
                    "changed": "model.npz",
                    "added": "extra.json",
                    "relabelled": "start.json",
                }[mode]
            )
        )
        experiment.write_json(path, {"altered": mode})
    with pytest.raises(ValueError):
        experiment.check_reference_import("cascade", output, runtime)


def test_external_root_anchor_cannot_be_replaced_by_local_hash(monkeypatch, tmp_path):
    experiment.resolved_protocol(tmp_path, create=True)
    experiment.write_json(tmp_path / "run.json", {"files": {}})
    trusted = experiment.digest(tmp_path / "run.json")
    experiment.write_json(tmp_path / "run.json", {"files": {}, "changed": True})
    with pytest.raises(ValueError, match="externally trusted"):
        experiment.verify_bundle(tmp_path, trusted)


def test_wrong_arm_cannot_start_fresh_reference_fit(monkeypatch, tmp_path):
    # Importing classes is permitted; neither simulator generation nor a fitter runs.
    with pytest.raises(ValueError, match="only frozen candidate"):
        experiment.fit_arm("crazyflow", "balanced", tmp_path)


DELTA = adapters.delta_protocol()


def resolved_fixture():
    p, rows = parent_fixture()
    p["id"] = DELTA["id"]
    for values in rows.values():
        for row in values:
            if row["arm"] == "anchored":
                row["arm"] = "balanced"
    decision = DELTA["decision"]
    p["decision"]["comparisons"] = {
        name: {**decision["guards_for_every_required_comparison"], **value}
        for name, value in decision["comparisons"].items()
    }
    for key in ("angular_retention", "diagnostic_optimization_progress"):
        p["decision"][key] = copy.deepcopy(decision[key])
    p["fitting"]["optimizer"] = copy.deepcopy(DELTA["fitting"]["optimizer"])
    p["imported_references"] = copy.deepcopy(DELTA["imported_references"])
    return p, rows


def optimization():
    steps = list(range(0, 1001, 100))
    losses = [dict(step=i, full_training_loss=1 - i / 2000) for i in steps]
    return dict(
        kind="filter_mlp",
        steps=1000,
        seed=0,
        context_steps=10,
        delay_steps=2,
        parameter_count=2,
        loss_scale_mode="explicit_horizon_channel",
        minibatch_seed=10000,
        mechanism="safeguarded-adam-v1",
        selected_step=1000,
        validation_rollout_mse=0.5,
        trace=[
            dict(
                step=i,
                validation_rollout_mse=1 - i / 2000,
                **({} if i == 0 else {"training_batch_mse": 1.0}),
            )
            for i in steps
        ],
        error_scale=[[1.0, 1.0]] * 5,
        objective={"exact": "weights"},
        safeguard=dict(
            id="safeguarded-adam-v1",
            acceptance="first_finite_strict_decrease",
            moment_policy="advance_on_every_finite_proposal",
            proposal_attempts=1000,
            completed_attempts=1000,
            accepted_attempts=1000,
            rejected_attempts=0,
            scales=DELTA["fitting"]["optimizer"]["scales"],
            accepted_scale_counts=[
                dict(scale=s, attempts=1000 if i == 0 else 0)
                for i, s in enumerate(DELTA["fitting"]["optimizer"]["scales"])
            ],
            maximum_rejection_streak=0,
            gradient_proposal_calls=1000,
            full_training_objective_calls=1001,
            full_training_objective_calls_max=8001,
            initial_full_training_loss=1.0,
            final_full_training_loss=0.5,
            selected_full_training_loss=0.5,
            checkpoint_full_training_losses=losses,
            fit_wall_time_limit_s=7200,
        ),
    )


def run(p, rows, opt):
    return adapters.reduce_safeguarded(
        {sim: aggregate(values) for sim, values in rows.items()},
        p,
        rows=rows,
        optimization=opt,
    )


def test_direction_angular_denominator_bootstrap_and_no_mutation():
    p, rows = resolved_fixture()
    before = copy.deepcopy((p, rows))
    result = run(p, rows, optimization())
    for name, expected in [
        ("safeguarded_public_progress", 0.6),
        ("balanced_progress", 0.6 / 0.85),
        ("quadratic_gain_retention", 0.6 / 0.8),
    ]:
        assert result[name]
        for kind in ("factual", "response"):
            assert result["comparisons"][name]["weighted_geometric_mean_ratios"][
                kind
            ] == pytest.approx(expected)
    assert result["angular_retention_comparison"]["denominator_arm"] == "balanced"
    assert result["shared_bootstrap_parent_draws_verified"]
    assert (
        result["residual_criteria_pass"] and result["diagnostic_optimization_progress"]
    )
    assert (
        result["numerical_mechanism_criteria_pass"] and not result["public_promotion"]
    )
    assert (p, rows) == before
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("kind,limit", [("factual", 1.05), ("response", 0.95)])
@pytest.mark.parametrize("offset,expected", [(-1e-6, True), (1e-6, False)])
def test_direct_balanced_comparison_limits(kind, limit, offset, expected):
    p, rows = resolved_fixture()
    set_scale(rows, 0.85 * (limit + offset), kind=kind)
    assert run(p, rows, optimization())["balanced_progress"] is expected


@pytest.mark.parametrize(
    "arm", ["baseline", "quadratic", "balanced", "candidate", "hold"]
)
def test_failure_preserves_unaffected_pair_reports(arm):
    p, rows = resolved_fixture()
    fail_arm(rows, arm)
    result = run(p, rows, None if arm == "candidate" else optimization())
    for item in result["comparisons"].values():
        affected = arm in (item["numerator_arm"], item["denominator_arm"])
        assert item["pairwise_predictions_available"] is not affected
    assert result["angular_retention"] is (arm not in ("balanced", "candidate"))
    assert result["residual_criteria_pass"] is (arm == "hold")


@pytest.mark.parametrize(
    "arm", ["baseline", "quadratic", "balanced", "candidate", "hold"]
)
def test_omitted_slots_reject(arm):
    p, rows = resolved_fixture()
    rows["crazyflow"].remove(
        next(row for row in rows["crazyflow"] if row["arm"] == arm)
    )
    with pytest.raises(ValueError, match="unmatched"):
        run(p, rows, optimization())


def test_historical_anchored_label_cannot_impersonate_balanced():
    p, rows = resolved_fixture()
    for values in rows.values():
        for row in values:
            if row["arm"] == "balanced":
                row["arm"] = "anchored"
    with pytest.raises(ValueError, match="five-arm roster"):
        run(p, rows, optimization())


def test_physical_success_does_not_override_failed_optimization_diagnostic():
    p, rows = resolved_fixture()
    opt = optimization()
    opt["selected_step"] = 0
    opt["safeguard"]["selected_full_training_loss"] = 1.0
    for row in opt["trace"]:
        row["validation_rollout_mse"] = 1.0 + row["step"] / 2000
    opt["validation_rollout_mse"] = 1.0
    result = run(p, rows, opt)
    assert result["residual_criteria_pass"]
    assert not result["diagnostic_optimization_progress"]
    assert not result["numerical_mechanism_criteria_pass"]


def test_diagnostic_uses_selected_canonical_loss_not_final_or_posthoc_diagnostics():
    p, _ = resolved_fixture()
    opt = optimization()
    opt["selected_step"] = 100
    opt["safeguard"]["selected_full_training_loss"] = 0.95
    for row in opt["trace"]:
        row["validation_rollout_mse"] = 0.9 if row["step"] == 100 else 1.0
    result = adapters.optimization_progress(opt, p)
    assert result["passed"]
    assert result["canonical_training_ratio"] == 0.95
    assert result["canonical_development_ratio"] == 0.9
    opt["safeguard"]["selected_full_training_loss"] = 0.5
    with pytest.raises(ValueError, match="mirrors"):
        adapters.optimization_progress(opt, p)


@pytest.mark.parametrize(
    "change", ["calls", "scales", "moments", "loss_increase", "step_roster"]
)
def test_inconsistent_safeguard_report_rejected(change):
    p, _ = resolved_fixture()
    opt = optimization()
    guard = opt["safeguard"]
    if change == "calls":
        guard["full_training_objective_calls"] += 1
    elif change == "scales":
        guard["scales"] = list(reversed(guard["scales"]))
    elif change == "moments":
        guard["moment_policy"] = "rollback"
    elif change == "loss_increase":
        guard["checkpoint_full_training_losses"][2]["full_training_loss"] = 2.0
    else:
        guard["checkpoint_full_training_losses"].pop()
    with pytest.raises(ValueError):
        adapters.optimization_progress(opt, p)


class Cache:
    def __init__(self):
        self.values = np.zeros((2, 3))

    def coverage(self):
        return {"windows": 2}


class Mean:
    def __init__(self):
        self.params = {"bias": np.zeros(2)}
        self.norms = {"scale": np.ones(2)}
        self.kind = "filter_mlp"
        self.dt_s = 0.05
        self.delay_steps = 2


class Model:
    def __init__(self, recipe):
        self._model = Mean()
        self._train = Cache()
        self._development = Cache()
        self.recipe = copy.deepcopy(recipe)
        self.history_steps = 10
        self.horizon_steps = 5
        self.contract = {"channels": ["x", "y"]}
        self.report = dict(
            recipe=self.recipe,
            previous_revision=None,
            history_steps=10,
            horizon_steps=5,
            delay_steps=2,
            training=self._train.coverage(),
            development=self._development.coverage(),
            matching={"same": "data"},
            optimization=optimization(),
        )

    def fingerprint(self):
        return "frozen"


@pytest.fixture
def models(monkeypatch):
    p, _ = resolved_fixture()
    recipe = p["fitting"]["generic_recipe"]
    module = SimpleNamespace(
        CandidateDynamics=Model,
        BalancedQuadraticSequenceModel=Mean,
        RECIPE={**recipe, "id": p["id"]},
    )
    public, quad, balanced = (Model(recipe) for _ in range(3))
    candidate = Model(module.RECIPE)
    scale = array_fingerprint(
        {}, {"error_scale": np.asarray(candidate.report["optimization"]["error_scale"])}
    )
    reference = dict(
        population="crazyflow",
        fitting_recipe=recipe,
        roles=p["planned_automatic_roles"]["crazyflow"],
        loss_scale_fingerprint=scale,
    )
    monkeypatch.setattr(adapters, "protocol", lambda: p, raising=False)
    monkeypatch.setattr(
        adapters._shared_matching, "check_matched_models", lambda *args: reference
    )
    monkeypatch.setattr(
        adapters._balanced_matching,
        "_objective",
        lambda model, recipe: model.report["optimization"]["objective"],
    )
    monkeypatch.setattr(
        adapters,
        "_window_fingerprint",
        lambda cache: array_fingerprint({}, {"values": cache.values}),
    )
    for value in p["imported_references"]["trusted_comparator_revisions"][
        "crazyflow"
    ].values():
        value["revision_fingerprint"] = "frozen"
    return public, candidate, quad, balanced, module, p


def match(models):
    public, candidate, quad, balanced, module, _ = models
    return adapters.check_matched_models(
        public, candidate, quad, balanced, "crazyflow", candidate_module=module
    )


def test_matching_accepts_exact_cached_objective_and_declared_budget(models):
    result = match(models)
    assert result["exact_same_data"] and result["exact_same_numeric_objective"]
    assert result["safeguard"]["full_training_objective_calls"] == 1001
    assert not result["equal_flops_claimed"]


@pytest.mark.parametrize(
    "change",
    [
        "cache",
        "coverage",
        "norms",
        "shapes",
        "scale",
        "objective",
        "recipe",
        "draw_seed",
        "budget",
        "fingerprint",
    ],
)
def test_matching_detects_independent_contract_corruptions(models, change):
    _, candidate, _, _, _, p = models
    if change == "cache":
        candidate._train.values[0, 0] = 1
    elif change == "coverage":
        candidate.report["training"]["windows"] = 3
    elif change == "norms":
        candidate._model.norms["scale"][0] = 2
    elif change == "shapes":
        candidate._model.params["bias"] = np.zeros(3)
    elif change == "scale":
        candidate.report["optimization"]["error_scale"][0] = [2.0, 2.0]
    elif change == "objective":
        candidate.report["optimization"]["objective"] = {"exact": "different"}
    elif change == "recipe":
        candidate.recipe["width"] = 99
    elif change == "draw_seed":
        candidate.report["optimization"]["minibatch_seed"] = 123
    elif change == "budget":
        candidate.report["optimization"]["steps"] = 999
    else:
        p["imported_references"]["trusted_comparator_revisions"]["crazyflow"][
            "balanced"
        ]["revision_fingerprint"] = "altered"
    with pytest.raises(ValueError):
        match(models)


@pytest.mark.parametrize("available", [True, False])
@pytest.mark.parametrize("replay", [True, False])
def test_candidate_checkpoint_expectations_bind_report_and_failure_prefix(
    monkeypatch, tmp_path, available, replay
):
    import sys
    from types import ModuleType

    import glassbox.experimental as package
    from glassbox.experimental.safeguarded_adam_model import RECIPE

    opt = optimization()
    opt["unweighted_development_trace"] = [
        dict(step=row["step"], validation_rollout_mse=2.0) for row in opt["trace"]
    ]
    selected = object()
    candidate = SimpleNamespace(_model=selected, report={"optimization": opt})
    baseline = SimpleNamespace(_train="trusted train", _development="trusted dev")
    calls = []

    def candidate_load(path):
        assert available, "failed candidate must not load an unavailable model"
        return candidate

    monkeypatch.setattr(
        experiment,
        "_model_classes",
        lambda: {
            "candidate": SimpleNamespace(load=candidate_load),
            "baseline": SimpleNamespace(load=lambda _: baseline),
        },
    )
    current = {"checkpoint_source_sha256": {"new-source": "abc"}}
    monkeypatch.setattr(experiment, "check_sources", lambda: current)
    monkeypatch.setattr(experiment, "fit_provenance", lambda *args: {"fit": "current"})
    monkeypatch.setattr(
        experiment, "anchor_reference", lambda *args: {"anchor": "balanced"}
    )
    fake = ModuleType("glassbox.experimental.safeguarded_adam_checkpoints")

    def check(path, **kwargs):
        calls.append(("check", path, kwargs))
        return {"kind": "check"}

    def replay_check(path, **kwargs):
        calls.append(("replay", path, kwargs))
        return {"kind": "replay"}

    fake.check_checkpoint_links = check
    fake.replay_checkpoints = replay_check
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    monkeypatch.setattr(package, "safeguarded_adam_checkpoints", fake, raising=False)
    reason = None if available else "fit_time_limit"
    experiment.write_json(
        tmp_path / "crazyflow/candidate/outcome.json",
        dict(
            status="complete" if available else "fit_failure",
            model_available=available,
            reason=reason,
            checkpoint_manifest_sha256="sealed manifest",
        ),
    )
    result = experiment.check_checkpoints(
        tmp_path, "crazyflow", "candidate", replay=replay
    )
    assert result["kind"] == ("replay" if replay else "check")
    assert len(calls) == 1
    _, path, kwargs = calls[0]
    assert path == tmp_path / "crazyflow/candidate/checkpoints"
    assert kwargs["train"] == baseline._train
    assert kwargs["development"] == baseline._development
    assert kwargs["provenance"] == {"fit": "current"}
    assert kwargs["source_sha256"] == current["checkpoint_source_sha256"]
    assert kwargs["expected_sha256"] == "sealed manifest"
    assert kwargs["anchor_reference"] == {"anchor": "balanced"}
    assert kwargs["selected_model"] is (selected if available else None)
    expected = kwargs["expectation"]
    assert expected["recipe"] == RECIPE
    assert expected["completed"] is available
    assert expected["failure"] == reason
    for field, report_key in (
        ("trace", "trace"),
        ("unweighted_trace", "unweighted_development_trace"),
        ("objective", "objective"),
        ("safeguard", "safeguard"),
    ):
        assert expected[field] == (opt[report_key] if available else None)


@pytest.mark.parametrize("replay", [False, True])
def test_imported_balanced_checkpoint_dispatches_to_original_candidate_checker(
    monkeypatch, tmp_path, replay
):
    source, output, runtime, _ = reference_fixture(monkeypatch, tmp_path)
    experiment.import_references("cascade", output, runtime)
    calls = []

    def historical(directory, simulator, arm, *, replay):
        calls.append((directory, simulator, arm, replay))
        return {"original": True}

    monkeypatch.setattr(previous, "check_checkpoints", historical)
    assert experiment.check_checkpoints(
        output, "cascade", "balanced", replay=replay
    ) == {"original": True}
    assert calls == [(source, "cascade", "candidate", replay)]


def test_resealed_imported_checkpoint_rejects_before_historical_dispatch(
    monkeypatch, tmp_path
):
    _, output, runtime, _ = reference_fixture(monkeypatch, tmp_path)
    experiment.import_references("cascade", output, runtime)
    base = output / "cascade/balanced"
    path = base / "checkpoints/manifest.json"
    experiment.write_json(path, {"altered": "checkpoint"})
    seal = experiment.read_json(base / "seal.json")
    seal["files"]["checkpoints/manifest.json"] = experiment.digest(path)
    experiment.write_json(base / "seal.json", seal)
    monkeypatch.setattr(
        previous,
        "check_checkpoints",
        lambda *args, **kwargs: pytest.fail("altered import reached historical replay"),
    )
    with pytest.raises(ValueError, match="reference checkpoint import differs"):
        experiment.check_checkpoints(output, "cascade", "balanced", replay=True)


@pytest.mark.parametrize("crazyflow_available", [False, True])
def test_decision_uses_only_crazyflow_candidate_canonical_optimizer_report(
    monkeypatch, tmp_path, crazyflow_available
):
    monkeypatch.setattr(experiment, "check_links", lambda *args, **kwargs: None)
    monkeypatch.setattr(experiment, "validate_rows", lambda *args: None)
    expected = None
    for sim in experiment.SIMULATORS:
        available = crazyflow_available if sim == "crazyflow" else True
        base = tmp_path / sim
        experiment.write_json(base / "evaluation/summary.json", {"simulator": sim})
        experiment.write_json(base / "evaluation/rows.json", [{"simulator": sim}])
        experiment.write_json(
            base / "candidate/outcome.json", {"model_available": available}
        )
        opt = {"canonical-source": sim}
        if available:
            experiment.write_json(base / "candidate/report.json", {"optimization": opt})
        if sim == "crazyflow" and available:
            expected = opt
    received = []

    def reducer(summaries, p, *, rows, optimization):
        received.append(optimization)
        assert set(rows) == set(summaries) == set(experiment.SIMULATORS)
        return {"diagnostic": optimization}

    monkeypatch.setattr(experiment, "reduce_safeguarded", reducer)
    assert experiment.decision(tmp_path) == {"diagnostic": expected}
    assert received == [expected]


@pytest.mark.parametrize("entrypoint", ["generate", "replay_data"])
def test_real_private_flight_entrypoint_receives_resolved_protocol_only(
    monkeypatch, tmp_path, entrypoint
):
    flight = experiment.flight()
    expected = experiment.protocol()
    runtime = {"toy": "source check only"}
    observed = []

    class StopBeforeSimulatorConstruction(RuntimeError):
        pass

    def sources(p):
        assert p == expected
        return runtime

    def fixture(simulator, p):
        observed.append((simulator, p))
        raise StopBeforeSimulatorConstruction

    monkeypatch.setattr(flight, "check_sources", sources)
    monkeypatch.setattr(flight, "fixture_for", fixture)
    monkeypatch.setattr(flight, "verify_files", lambda *args: {"runtime": runtime})
    with pytest.raises(StopBeforeSimulatorConstruction):
        getattr(flight, entrypoint)("crazyflow", tmp_path)
    assert observed == [("crazyflow", expected)]
    assert "recordings" in observed[0][1] and "generation" in observed[0][1]
    assert original.read_json(experiment.PROTOCOL) == experiment.delta_protocol()
    assert "recordings" not in original.read_json(experiment.PROTOCOL)
    ordinary = tmp_path / "ordinary.json"
    experiment.write_json(ordinary, {"ordinary": "read unchanged"})
    assert flight.read_json(ordinary) == original.read_json(ordinary)


def test_fresh_ids_and_seeds_are_independently_disjoint_all_eight_prior_cohorts():
    current = experiment.protocol()
    rows = [row for row in current["recordings"] if row["role"] == "test"]
    ids, seeds = {row["id"] for row in rows}, {row["seed"] for row in rows}
    identities = {(row["simulator"], row["seed"]) for row in rows}
    assert len(ids) == len(identities) == len(rows) == 168
    assert len(seeds) == 84  # The two distinct simulators share numeric seed values.
    for name in (
        "two-simulator-flight-v1",
        "state-input-interaction-v1",
        "autonomous-state-quadratic-v1",
        "independent-command-excitation-v1",
        "excited-public-architecture-v1",
        "excited-bilinear-ablation-v1",
        "affine-anchored-quadratic-v1",
        "initial-channel-balance-v1",
    ):
        prior = experiment.read_json(
            experiment.ROOT / "docs/harness" / (name + ".json")
        )
        old = [row for row in prior["recordings"] if row["role"] == "test"]
        assert old, name
        assert ids.isdisjoint(row["id"] for row in old), name
        assert seeds.isdisjoint(row["seed"] for row in old), name
