"""Full-cache mechanism comparisons and exact matched-data/report boundaries."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from test_excited_bilinear_decision import fail_arm, set_scale
from test_initial_channel_balance_decision import fixture as metric_fixture
from test_safeguarded_adam_experiment import optimization as previous_optimization

from glassbox._learner_arrays import array_fingerprint
from glassbox.experimental import full_cache_gradient_experiment as experiment
from glassbox.experimental import safeguarded_adam_experiment as previous
from glassbox.experimental.two_simulator_metrics import aggregate

DELTA = json.loads(
    (
        Path(__file__).resolve().parents[1] / "docs/harness/full-cache-gradient-v1.json"
    ).read_text()
)


def resolved_fixture():
    p, rows = metric_fixture()
    p["id"] = DELTA["id"]
    for values in rows.values():
        for row in values:
            row["arm"] = {"anchored": "balanced", "quadratic": "safeguarded"}.get(
                row["arm"], row["arm"]
            )
    decision = DELTA["decision"]
    p["decision"]["comparisons"] = {
        name: {**decision["guards_for_every_required_comparison"], **policy}
        for name, policy in decision["comparisons"].items()
    }
    for key in ("angular_retention", "diagnostic_optimization_progress"):
        p["decision"][key] = copy.deepcopy(decision[key])
    p["fitting"]["optimizer"] = copy.deepcopy(DELTA["fitting"]["optimizer"])
    p["imported_references"] = copy.deepcopy(DELTA["imported_references"])
    return p, rows


def optimization():
    opt = previous_optimization()
    opt.update(
        mechanism=DELTA["id"],
        minibatch_seed=None,
        gradient_sampling="ordered_full_cache",
    )
    opt["safeguard"]["id"] = DELTA["id"]
    opt["gradient"] = dict(
        policy="ordered_full_cache",
        loss_scope="full_training_cache_pre_proposal",
        training_windows=384,
        index_dtype="int64",
        sampling_seed=None,
        attempts_started=1000,
        gradient_proposal_calls_returned=1000,
        completed_acceptance_attempts=1000,
        known_gradient_window_visits=384000,
        incomplete_gradient_work_unknown=False,
    )
    return opt


def reduce(p, rows, opt):
    return experiment.reduce_full_cache(
        {sim: aggregate(values) for sim, values in rows.items()},
        p,
        rows=rows,
        optimization=opt,
    )


def test_three_real_directions_angular_reference_and_no_historical_mutation():
    p, rows = resolved_fixture()
    before = copy.deepcopy((p, rows, previous._SAFEGUARD_PAIRS))
    value = reduce(p, rows, optimization())
    for name, denominator, ratio in [
        ("full_cache_public_progress", "baseline", 0.6),
        ("balanced_progress", "balanced", 0.6 / 0.85),
        ("safeguarded_progress", "safeguarded", 0.6 / 0.8),
    ]:
        assert value[name]
        pair = value["comparisons"][name]
        assert (
            pair["numerator_arm"] == "candidate"
            and pair["denominator_arm"] == denominator
        )
        assert pair["weighted_geometric_mean_ratios"] == pytest.approx(
            dict(factual=ratio, response=ratio)
        )
    assert value["angular_retention_comparison"]["denominator_arm"] == "balanced"
    assert value["shared_bootstrap_parent_draws_verified"]
    assert all(
        set(c["arms"]) == {"baseline", "balanced", "safeguarded", "candidate", "hold"}
        for c in value["cohorts"].values()
    )
    assert "quadratic" not in json.dumps(value)
    assert value["residual_criteria_pass"] and not value["public_promotion"]
    assert (p, rows, previous._SAFEGUARD_PAIRS) == before
    json.dumps(value, allow_nan=False)


@pytest.mark.parametrize("kind,limit", [("factual", 1.05), ("response", 0.95)])
@pytest.mark.parametrize("offset,expected", [(-1e-6, True), (1e-6, False)])
@pytest.mark.parametrize(
    "pair,reference,scale",
    [
        ("balanced_progress", "balanced", 0.85),
        ("safeguarded_progress", "safeguarded", 0.8),
    ],
)
def test_each_direct_reference_uses_its_frozen_margin(
    kind, limit, offset, expected, pair, reference, scale
):
    p, rows = resolved_fixture()
    set_scale(rows, scale * (limit + offset), kind=kind)
    value = reduce(p, rows, optimization())
    assert value[pair] is expected
    assert value["comparisons"][pair]["denominator_arm"] == reference


@pytest.mark.parametrize(
    "arm", ["baseline", "balanced", "safeguarded", "candidate", "hold"]
)
def test_unavailable_arm_cannot_erase_unaffected_comparisons(arm):
    p, rows = resolved_fixture()
    fail_arm(rows, arm)
    value = reduce(p, rows, None if arm == "candidate" else optimization())
    for pair in value["comparisons"].values():
        assert pair["pairwise_predictions_available"] is (
            arm not in (pair["numerator_arm"], pair["denominator_arm"])
        )
    assert value["angular_retention"] is (arm not in ("balanced", "candidate"))
    assert value["residual_criteria_pass"] is (arm == "hold")


@pytest.mark.parametrize(
    "arm", ["baseline", "balanced", "safeguarded", "candidate", "hold"]
)
def test_missing_query_slots_are_integrity_failures(arm):
    p, rows = resolved_fixture()
    rows["crazyflow"].remove(
        next(row for row in rows["crazyflow"] if row["arm"] == arm)
    )
    with pytest.raises(ValueError, match="unmatched"):
        reduce(p, rows, optimization())


def test_physical_success_is_not_vetoed_by_optimization_diagnostic():
    p, rows = resolved_fixture()
    opt = optimization()
    opt["selected_step"] = 0
    opt["safeguard"]["selected_full_training_loss"] = 1.0
    for row in opt["trace"]:
        row["validation_rollout_mse"] = 1.0 + row["step"] / 2000
    opt["validation_rollout_mse"] = 1.0
    value = reduce(p, rows, opt)
    assert value["residual_criteria_pass"]
    assert not value["diagnostic_optimization_progress"]
    assert (
        value["verification_status"]
        == "requires_integrity_replay_tamper_and_focused_tests"
    )
    assert not value["public_promotion"]
    assert "numerical_mechanism_criteria_pass" not in value


def test_diagnostic_uses_originally_selected_canonical_scalar_not_final():
    p, _ = resolved_fixture()
    opt = optimization()
    opt["selected_step"] = 100
    opt["safeguard"]["selected_full_training_loss"] = 0.95
    for row in opt["trace"]:
        row["validation_rollout_mse"] = 0.9 if row["step"] == 100 else 1.0
    opt["validation_rollout_mse"] = 0.9
    value = experiment.optimization_progress(opt, p)
    assert (
        value["canonical_training_ratio"] == 0.95
        and value["canonical_development_ratio"] == 0.9
    )
    opt["safeguard"]["selected_full_training_loss"] = 0.5
    with pytest.raises(ValueError, match="mirrors"):
        experiment.optimization_progress(opt, p)


def test_old_quadratic_arm_cannot_impersonate_new_mechanism_control():
    p, rows = resolved_fixture()
    for values in rows.values():
        for row in values:
            if row["arm"] == "safeguarded":
                row["arm"] = "quadratic"
    with pytest.raises(ValueError, match="five-arm roster"):
        reduce(p, rows, optimization())


class Cache:
    def __init__(self, n, parents):
        self.values = np.zeros((n, 2))
        self.keys = [
            SimpleNamespace(recording_id=parents[i % len(parents)]) for i in range(n)
        ]
        self.batch = SimpleNamespace(
            past_states=np.zeros((n, 10, 2)), future_states=np.ones((n, 5, 2))
        )

    def coverage(self):
        return {"windows": len(self.keys)}


class Mean:
    def __init__(self):
        self.params = {"bias": np.zeros(2)}
        self.norms = {key: np.ones(2) for key in experiment._shared_matching.BASE_NORMS}
        self.kind = "filter_mlp"
        self.dt_s = 0.05
        self.delay_steps = 2


class Model:
    def __init__(self, recipe, roles):
        self.recipe = copy.deepcopy(recipe)
        self._model = Mean()
        self._train = Cache(384, roles["training"])
        self._development = Cache(256, roles["development"])
        self._seen = {
            name: "0" * 64 for name in roles["training"] + roles["development"]
        }
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
            matching=dict(data="same", batch_size=64, minibatch_seed=10000),
            optimization=optimization(),
        )

    def fingerprint(self):
        return "frozen"


@pytest.fixture
def models(monkeypatch):
    p, _ = resolved_fixture()
    recipe = p["fitting"]["generic_recipe"]
    roles = p["planned_automatic_roles"]["crazyflow"]
    public = Model(recipe, roles)
    balanced = Model(dict(recipe, id="balanced"), roles)
    safeguarded = Model(dict(recipe, id="safeguarded"), roles)
    candidate_recipe = dict(
        safeguarded.recipe,
        id=p["id"],
        batch_size=384,
        gradient_sampling="ordered_full_cache",
    )
    candidate = Model(candidate_recipe, roles)
    candidate.report["matching"].update(
        batch_size=384, minibatch_seed=None, gradient_sampling="ordered_full_cache"
    )
    module = SimpleNamespace(
        CandidateDynamics=Model,
        BalancedQuadraticSequenceModel=Mean,
        RECIPE=candidate_recipe,
    )
    monkeypatch.setattr(experiment, "protocol", lambda: p, raising=False)
    monkeypatch.setattr(
        experiment._balanced_matching,
        "_objective",
        lambda model, recipe: model.report["optimization"]["objective"],
    )
    monkeypatch.setattr(
        experiment,
        "_window_fingerprint",
        lambda cache: array_fingerprint({}, dict(values=cache.values)),
    )
    for value in p["imported_references"]["trusted_comparator_revisions"][
        "crazyflow"
    ].values():
        value["revision_fingerprint"] = "frozen"
    return public, candidate, balanced, safeguarded, module, p


def match(models):
    public, candidate, balanced, safeguarded, module, _ = models
    return experiment.check_matched_models(
        public, candidate, balanced, safeguarded, "crazyflow", candidate_module=module
    )


def test_matching_changes_only_gradient_metadata_and_keeps_references_64(models):
    before = copy.deepcopy([m.report for m in (models[0], models[2], models[3])])
    value = match(models)
    assert value["exact_same_data"] and value["exact_same_numeric_objective"]
    assert value["gradient"]["known_gradient_window_visits"] == 384000
    assert value["fitting_recipe"]["batch_size"] == 384
    assert set(value["reference_fingerprints"]) == {
        "baseline",
        "balanced",
        "safeguarded",
    }
    assert not value["equal_flops_claimed"]
    assert before == [m.report for m in (models[0], models[2], models[3])]


@pytest.mark.parametrize(
    "change",
    [
        "cache",
        "coverage",
        "norms",
        "shape",
        "scale",
        "objective",
        "recipe",
        "preparation",
        "roles",
        "reference",
        "seed",
        "sampling",
        "scope",
        "count",
        "returned",
        "unknown",
    ],
)
def test_matching_rejects_data_objective_and_gradient_contract_mutations(
    models, change
):
    public, candidate, _balanced, _, _, p = models
    opt = candidate.report["optimization"]
    if change == "cache":
        candidate._train.values[0, 0] = 1
    elif change == "coverage":
        candidate.report["training"]["windows"] = 385
    elif change == "norms":
        candidate._model.norms["state_scale"][0] = 2
    elif change == "shape":
        candidate._model.params["bias"] = np.zeros(3)
    elif change == "scale":
        opt["error_scale"][0] = [2.0, 2.0]
    elif change == "objective":
        opt["objective"] = {"wrong": "weights"}
    elif change == "recipe":
        candidate.recipe["width"] = 33
    elif change == "preparation":
        candidate.report["matching"]["batch_size"] = 64
    elif change == "roles":
        public._seen.pop(next(iter(public._seen)))
    elif change == "reference":
        p["imported_references"]["trusted_comparator_revisions"]["crazyflow"][
            "balanced"
        ]["revision_fingerprint"] = "altered"
    elif change == "seed":
        opt["minibatch_seed"] = 10000
    elif change == "sampling":
        opt["gradient_sampling"] = "with_replacement"
    elif change == "scope":
        opt["gradient"]["loss_scope"] = "minibatch"
    elif change == "count":
        opt["gradient"]["known_gradient_window_visits"] = 64000
    elif change == "returned":
        opt["gradient"]["gradient_proposal_calls_returned"] = 999
    else:
        opt["gradient"]["incomplete_gradient_work_unknown"] = True
    with pytest.raises(ValueError):
        match(models)


@pytest.fixture(autouse=True)
def isolated_private_bindings():
    for name in ("_parent", "_runner", "_implementation"):
        function = getattr(experiment, name, None)
        if function is not None:
            function.cache_clear()
    yield
    for name in ("_parent", "_runner", "_implementation"):
        function = getattr(experiment, name, None)
        if function is not None:
            function.cache_clear()


def test_resolved_protocol_preserves_calibration_and_excludes_nine_previous_test_cohorts():
    current = experiment.protocol()
    base = previous.protocol()
    original = experiment.read_json(
        experiment.ROOT / DELTA["cohort"]["original_protocol"]["path"]
    )
    for row, old in zip(current["recordings"], base["recordings"], strict=True):
        if row["role"] == "calibration_pool":
            assert row == old
        else:
            assert row["seed"] == old["seed"] + 1000000
    fresh = [r for r in current["recordings"] if r["role"] == "test"]
    ids = {r["id"] for r in fresh}
    seeds = {r["seed"] for r in fresh}
    assert len(ids) == len({(r["simulator"], r["seed"]) for r in fresh}) == 168
    assert len(seeds) == 84
    for shift in range(0, 9000000, 1000000):
        old = [r for r in original["recordings"] if r["role"] == "test"]
        assert not seeds & {r["seed"] + shift for r in old}
        assert not ids & {
            r["id"].rsplit("-", 1)[0] + "-" + str(r["seed"] + shift) for r in old
        }
    assert set(current["fitting"]["arms"]) == {
        "baseline",
        "balanced",
        "safeguarded",
        "candidate",
    }
    assert set(current["decision"]["comparisons"]) == set(
        DELTA["decision"]["comparisons"]
    )
    assert current["fitting"]["generic_recipe"] == base["fitting"]["generic_recipe"]


def test_private_parent_callbacks_route_current_generation_without_global_mutation(
    monkeypatch,
):
    old = (
        previous.protocol,
        previous.REFERENCE_ARMS,
        previous.check_checkpoints,
        previous.Predictors,
    )
    marker = {"current": "callback"}
    monkeypatch.setattr(experiment, "check_sources", lambda: marker)
    parent = experiment._parent()
    runner = experiment._runner()
    interaction = experiment._implementation()
    assert parent.previous is previous
    assert parent.check_sources() is marker
    assert parent.protocol is experiment.protocol
    assert parent._native["check_checkpoints"] is not parent.check_checkpoints
    assert (
        set(parent.REFERENCE_ARMS)
        == set(runner.REFERENCE_ARMS)
        == {"baseline", "balanced", "safeguarded"}
    )
    assert set(interaction.PREDICTORS) == {
        "baseline",
        "balanced",
        "safeguarded",
        "candidate",
        "hold",
    }
    assert (
        previous.protocol,
        previous.REFERENCE_ARMS,
        previous.check_checkpoints,
        previous.Predictors,
    ) == old


@pytest.mark.parametrize("entrypoint", ["generate", "replay_data"])
def test_private_physics_entrypoint_reads_resolved_current_protocol(
    monkeypatch, tmp_path, entrypoint
):
    from glassbox.experimental import two_simulator_flight as original

    flight = experiment.flight()
    expected = experiment.protocol()
    runtime = {"toy": "no physics"}
    seen = []

    class StopBeforePhysics(RuntimeError):
        pass

    def fixture(simulator, p):
        seen.append((simulator, p))
        raise StopBeforePhysics

    monkeypatch.setattr(flight, "fixture_for", fixture)
    monkeypatch.setattr(
        flight,
        "check_sources",
        lambda p: (
            runtime if p == expected else pytest.fail("raw protocol reached physics")
        ),
    )
    monkeypatch.setattr(flight, "verify_files", lambda *args: {"runtime": runtime})
    with pytest.raises(StopBeforePhysics):
        getattr(flight, entrypoint)("crazyflow", tmp_path)
    assert seen == [("crazyflow", expected)]
    assert "recordings" not in original.read_json(experiment.PROTOCOL)
    assert original.read_json(experiment.PROTOCOL) == experiment.delta_protocol()


def reference_fixture(monkeypatch, tmp_path):
    # Reuse the generic byte-copy toy fixture with isolated helper globals.
    from types import FunctionType

    from test_safeguarded_adam_experiment import reference_fixture as old

    scoped = FunctionType(old.__code__, {**old.__globals__, "experiment": experiment})
    return scoped(monkeypatch, tmp_path)


@pytest.mark.parametrize(
    "arm,old_arm",
    [("baseline", "baseline"), ("balanced", "balanced"), ("safeguarded", "candidate")],
)
@pytest.mark.parametrize("replay", [False, True])
def test_imported_checkpoint_uses_original_arm_and_original_checker(
    monkeypatch, tmp_path, arm, old_arm, replay
):
    source, output, runtime, refs = reference_fixture(monkeypatch, tmp_path)
    result = experiment.import_references("cascade", output, runtime)
    assert result["format"] == "glassbox-full-cache-reference-import-v1"
    assert set(result["arms"]) == {"baseline", "balanced", "safeguarded"}
    assert result["fresh_reference_fits"] == 0
    for name in refs[arm]["files"]:
        assert (output / "cascade" / arm / name).read_bytes() == (
            refs[arm]["directory"] / name
        ).read_bytes()
    assert (
        experiment.read_json(output / "cascade" / arm / "start.json")["arm"] == old_arm
    )
    calls = []

    def historical(directory, simulator, source_arm, *, replay):
        calls.append((directory, simulator, source_arm, replay))
        return {"original": True}

    monkeypatch.setattr(previous, "check_checkpoints", historical)
    assert experiment.check_checkpoints(output, "cascade", arm, replay=replay) == {
        "original": True
    }
    assert calls == [(source, "cascade", old_arm, replay)]


def test_resealed_safeguarded_import_rejects_before_original_checker(
    monkeypatch, tmp_path
):
    _, output, runtime, _ = reference_fixture(monkeypatch, tmp_path)
    experiment.import_references("cascade", output, runtime)
    base = output / "cascade/safeguarded"
    path = base / "checkpoints/manifest.json"
    experiment.write_json(path, {"relabelled": "full cache"})
    seal = experiment.read_json(base / "seal.json")
    seal["files"]["checkpoints/manifest.json"] = experiment.digest(path)
    experiment.write_json(base / "seal.json", seal)
    monkeypatch.setattr(
        previous,
        "check_checkpoints",
        lambda *a, **kw: pytest.fail("altered import reached historical checker"),
    )
    with pytest.raises(ValueError, match="reference checkpoint import differs"):
        experiment.check_checkpoints(output, "cascade", "safeguarded", replay=True)


@pytest.mark.parametrize("available", [False, True])
def test_current_diagnostic_reads_only_crazyflow_selected_candidate_report(
    monkeypatch, tmp_path, available
):
    monkeypatch.setattr(experiment, "check_links", lambda *a, **kw: None)
    monkeypatch.setattr(experiment, "validate_rows", lambda *a: None)
    for sim in experiment.SIMULATORS:
        base = tmp_path / sim
        experiment.write_json(base / "evaluation/summary.json", {"simulator": sim})
        experiment.write_json(base / "evaluation/rows.json", [{"simulator": sim}])
        experiment.write_json(
            base / "candidate/outcome.json",
            {"model_available": available if sim == "crazyflow" else True},
        )
        experiment.write_json(
            base / "candidate/report.json", {"optimization": {"canonical": sim}}
        )
    received = []

    def reducer(summaries, p, *, rows, optimization):
        received.append(optimization)
        return {"diagnostic": optimization}

    monkeypatch.setattr(experiment, "reduce_full_cache", reducer)
    expected = {"canonical": "crazyflow"} if available else None
    assert experiment.decision(tmp_path) == {"diagnostic": expected}
    assert received == [expected]
