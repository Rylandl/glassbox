"""Expanded-cache comparisons, preparation boundaries and authentic imports."""

import copy
import json
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from test_excited_bilinear_decision import fail_arm, set_scale
from test_initial_channel_balance_decision import fixture as metric_fixture
from test_safeguarded_adam_experiment import optimization as previous_optimization

from glassbox import learner
from glassbox._learner_arrays import array_fingerprint
from glassbox._sequence_model import SequenceBatch
from glassbox.experimental import expanded_training_cache_checkpoints as observer
from glassbox.experimental import expanded_training_cache_experiment as experiment
from glassbox.experimental import expanded_training_cache_model as candidate_module
from glassbox.experimental import full_cache_gradient_experiment as previous
from glassbox.experimental.two_simulator_metrics import aggregate
from glassbox.learner import _subset
from glassbox.recordings import SequenceWindows, WindowKey

DELTA = json.loads(
    (
        Path(__file__).resolve().parents[1]
        / "docs/harness/expanded-training-cache-v1.json"
    ).read_text()
)


def resolved_fixture():
    p, rows = metric_fixture()
    p["id"] = DELTA["id"]
    for values in rows.values():
        for row in values:
            row["arm"] = {"anchored": "balanced", "quadratic": "fullcache384"}.get(
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
    p["data_support"] = copy.deepcopy(DELTA["data_support"])
    p["fitting"]["candidate_recipe_overrides"] = copy.deepcopy(
        DELTA["fitting"]["candidate_recipe_overrides"]
    )
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
        training_windows=1536,
        index_dtype="int64",
        sampling_seed=None,
        attempts_started=1000,
        gradient_proposal_calls_returned=1000,
        completed_acceptance_attempts=1000,
        known_gradient_window_visits=1536000,
        incomplete_gradient_work_unknown=False,
    )
    return opt


def reduce(p, rows, opt):
    return experiment.reduce_expanded_cache(
        {sim: aggregate(values) for sim, values in rows.items()},
        p,
        rows=rows,
        optimization=opt,
    )


def test_three_real_directions_angular_reference_and_no_historical_mutation():
    p, rows = resolved_fixture()
    before = copy.deepcopy((p, rows, previous._FULL_CACHE_PAIRS))
    value = reduce(p, rows, optimization())
    for name, denominator, ratio in [
        ("expanded_public_progress", "baseline", 0.6),
        ("balanced_progress", "balanced", 0.6 / 0.85),
        ("data_support_progress", "fullcache384", 0.6 / 0.8),
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
        set(c["arms"]) == {"baseline", "balanced", "fullcache384", "candidate", "hold"}
        for c in value["cohorts"].values()
    )
    assert "quadratic" not in json.dumps(value)
    assert value["residual_criteria_pass"] and not value["public_promotion"]
    assert (p, rows, previous._FULL_CACHE_PAIRS) == before
    json.dumps(value, allow_nan=False)


def test_actual_resolved_decision_dispatch_preserves_frozen_guards_and_pair_limits():
    protocol = experiment.protocol()
    decision = protocol["decision"]
    guards = DELTA["decision"]["guards_for_every_required_comparison"]
    assert decision["accept_research_candidate"] == guards
    assert decision["outcomes"] == DELTA["decision"]["outcomes"]
    fixture, rows = resolved_fixture()
    # Bound only the synthetic population and aggregation grid. The entire
    # actual resolved decision policy, including all numerical limits, remains.
    protocol["cells"] = fixture["cells"]
    protocol["recordings"] = fixture["recordings"]
    decision["aggregation"] = fixture["decision"]["aggregation"]
    before = copy.deepcopy(protocol)
    value = reduce(protocol, rows, optimization())
    for name, policy in DELTA["decision"]["comparisons"].items():
        comparison = value["comparisons"][name]
        assert comparison["limits"] == {
            **{key: guards[key] for key in guards if key.endswith("ratio_max")},
            **{key: policy[key] for key in policy if key.endswith("ratio_max")},
        }
        assert comparison["denominator_arm"] == policy["denominator"]
        assert len(comparison["tail_checks"]) == 12
        assert comparison["residual_criteria_pass"]
    assert value["residual_criteria_pass"]
    assert protocol == before


@pytest.mark.parametrize("kind,limit", [("factual", 1.05), ("response", 0.95)])
@pytest.mark.parametrize("offset,expected", [(-1e-6, True), (1e-6, False)])
@pytest.mark.parametrize(
    "pair,reference,scale",
    [
        ("balanced_progress", "balanced", 0.85),
        ("data_support_progress", "fullcache384", 0.8),
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
    "arm", ["baseline", "balanced", "fullcache384", "candidate", "hold"]
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
    "arm", ["baseline", "balanced", "fullcache384", "candidate", "hold"]
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
            if row["arm"] == "fullcache384":
                row["arm"] = "quadratic"
    with pytest.raises(ValueError, match="five-arm roster"):
        reduce(p, rows, optimization())


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


def test_resolved_protocol_preserves_calibration_and_excludes_ten_previous_test_cohorts():
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
    for shift in range(0, 10000000, 1000000):
        old = [r for r in original["recordings"] if r["role"] == "test"]
        assert not seeds & {r["seed"] + shift for r in old}
        assert not ids & {
            r["id"].rsplit("-", 1)[0] + "-" + str(r["seed"] + shift) for r in old
        }
    assert set(current["fitting"]["arms"]) == {
        "baseline",
        "balanced",
        "fullcache384",
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
        == {"baseline", "balanced", "fullcache384"}
    )
    assert set(interaction.PREDICTORS) == {
        "baseline",
        "balanced",
        "fullcache384",
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
    [("baseline", "baseline"), ("balanced", "balanced"), ("fullcache384", "candidate")],
)
@pytest.mark.parametrize("replay", [False, True])
def test_imported_checkpoint_uses_original_arm_and_original_checker(
    monkeypatch, tmp_path, arm, old_arm, replay
):
    source, output, runtime, refs = reference_fixture(monkeypatch, tmp_path)
    result = experiment.import_references("cascade", output, runtime)
    assert result["format"] == "glassbox-expanded-cache-reference-import-v1"
    assert set(result["arms"]) == {"baseline", "balanced", "fullcache384"}
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


def test_resealed_fullcache_import_rejects_before_original_checker(
    monkeypatch, tmp_path
):
    _, output, runtime, _ = reference_fixture(monkeypatch, tmp_path)
    experiment.import_references("cascade", output, runtime)
    base = output / "cascade/fullcache384"
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
        experiment.check_checkpoints(output, "cascade", "fullcache384", replay=True)


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

    monkeypatch.setattr(experiment, "reduce_expanded_cache", reducer)
    expected = {"canonical": "crazyflow"} if available else None
    assert experiment.decision(tmp_path) == {"diagnostic": expected}
    assert received == [expected]


def toy_windows(n, parents, *, expanded=False):
    rng = np.random.default_rng(882 if expanded else 91)
    arrays = dict(
        past_states=rng.normal(size=(n, 11, 2)),
        past_inputs=rng.normal(size=(n, 10, 1)),
        future_inputs=rng.normal(size=(n, 5, 1)),
        future_states=rng.normal(size=(n, 5, 2)),
    )
    if expanded:
        arrays["past_states"][384:] += np.array([4.0, -2.0])
        arrays["future_states"][384:] += np.array([4.0, -2.0])
    keys = tuple(
        WindowKey(parents[i % len(parents)], "segment", 10 + i // len(parents))
        for i in range(n)
    )
    return SequenceWindows(
        SequenceBatch(**arrays, dt_s=0.05), keys, tuple(k.origin for k in keys)
    )


class ToyMean:
    def __init__(self, norms):
        self.params = {"bias": np.zeros(2)}
        self.norms = copy.deepcopy(norms)
        self.kind, self.dt_s, self.delay_steps = "filter_mlp", 0.05, 2


class ToyModel:
    def __init__(self, arm, recipe, train, development, norms, seen):
        self.arm, self.recipe = arm, copy.deepcopy(recipe)
        self._train, self._development = copy.deepcopy((train, development))
        self._model, self._seen = ToyMean(norms), copy.deepcopy(seen)
        self.history_steps, self.horizon_steps = 10, 5
        self.contract = {"channels": ["x", "y"], "dt_s": 0.05}
        opt = optimization()
        scale = np.sqrt(
            np.mean(
                (train.batch.past_states[:, -1:] - train.batch.future_states) ** 2,
                axis=0,
            )
        )
        opt["error_scale"] = scale.tolist()
        mse = np.array([2.0, 3.0]) if arm == "candidate" else np.array([1.0, 4.0])
        raw = 1 / mse
        opt["objective"] = dict(
            id="fixed_initial_training_channel_balance",
            reduction="numpy_float64_mean_over_training_windows_and_horizon",
            weighting_data="initial_recursive_training_predictions_only",
            initial_channel_mse=mse.tolist(),
            raw_channel_weights=raw.tolist(),
            channel_weights=(raw / raw.mean()).tolist(),
            weight_floor=0.0001,
            weight_normalizer=float(raw.mean()),
            floor_active_channels=[],
            initial_parameters_and_norms_fingerprint="1" * 64,
            initial_training_prediction_fingerprint="2" * 64,
            fixed_during_training_and_selection=True,
            original_hold_scale_preserved=True,
            additional_training_weight_forecasts=1,
        )
        opt["unweighted_development_trace"] = [
            dict(step=r["step"], validation_rollout_mse=r["validation_rollout_mse"])
            for r in opt["trace"]
        ]
        opt["selection_objective"] = "fixed_initial_training_channel_balance"
        self.report = dict(
            recipe=self.recipe,
            history_steps=10,
            horizon_steps=5,
            delay_steps=2,
            training=self._train.coverage(),
            development=self._development.coverage(),
            optimization=opt,
        )

    def fingerprint(self):
        return self.arm


@pytest.fixture
def matched(monkeypatch):
    from glassbox.experimental import expanded_training_cache_model as numeric
    from glassbox.experimental.full_cache_gradient_model import (
        RECIPE as reference_recipe,
    )

    p, _ = resolved_fixture()
    roles = p["planned_automatic_roles"]["crazyflow"]
    train = toy_windows(1536, roles["training"], expanded=True)
    dev = toy_windows(256, roles["development"])
    old = _subset(train, np.arange(384))
    seen = {
        name: f"{i:064x}"
        for i, name in enumerate(roles["training"] + roles["development"])
    }
    oldnorms, norms = (
        numeric.training_norms(old.batch),
        numeric.training_norms(train.batch),
    )
    recipes = dict(
        baseline=p["fitting"]["generic_recipe"],
        balanced=dict(reference_recipe, id="initial-channel-balance-v1", batch_size=64),
        fullcache384=reference_recipe,
        candidate=dict(reference_recipe, **p["fitting"]["candidate_recipe_overrides"]),
    )
    models = {
        arm: ToyModel(
            arm,
            recipe,
            train if arm == "candidate" else old,
            dev,
            norms if arm == "candidate" else oldnorms,
            seen,
        )
        for arm, recipe in recipes.items()
    }
    fitting = dict(
        p["fitting"]["generic_recipe"], training_windows=1536, batch_size=1536
    )
    candidate, public = models["candidate"], models["baseline"]
    scale = np.asarray(candidate.report["optimization"]["error_scale"])
    ridge = fitting["ridge_fraction"] * 1536 * 5
    provenance = dict(
        intervention=p["id"],
        roles=copy.deepcopy(roles),
        public_reference_fingerprint=public.fingerprint(),
        recording_content_fingerprints=seen,
        training_windows_fingerprint=experiment._window_fingerprint(train),
        development_windows_fingerprint=experiment._window_fingerprint(dev),
        reference_training_windows_fingerprint=experiment._window_fingerprint(old),
        training_norms_fingerprint=array_fingerprint({}, norms),
        error_scale_fingerprint=array_fingerprint({}, {"error_scale": scale}),
        fitting_recipe=fitting,
        ridge=ridge,
        delay=2,
        parameter_count=2,
        training_windows=1536,
        development_windows=256,
        reference_training_windows=384,
        added_training_windows=1152,
        same_recordings=True,
        same_development_cache=True,
        training_prefix_exact=True,
        same_training_cache=False,
        same_numeric_objective=False,
        same_all_initial_parameters_required=False,
        equal_compute=False,
    )
    prepared = SimpleNamespace(
        train=train,
        development=dev,
        contract=public.contract,
        norms=norms,
        error_scale=scale,
        ridge=ridge,
        delay=2,
        provenance=provenance,
    )
    candidate.report["preparation"] = copy.deepcopy(provenance)
    module = SimpleNamespace(
        CandidateDynamics=ToyModel,
        BalancedQuadraticSequenceModel=ToyMean,
        RECIPE=recipes["candidate"],
        FITTING_RECIPE=fitting,
        training_norms=numeric.training_norms,
    )
    for arm, entry in p["imported_references"]["trusted_comparator_revisions"][
        "crazyflow"
    ].items():
        entry["revision_fingerprint"] = arm
    monkeypatch.setattr(experiment, "protocol", lambda: p)
    return models, prepared, module, p


def match(matched):
    models, prepared, module, _ = matched
    return experiment.check_matched_models(
        models["baseline"],
        models["candidate"],
        models["balanced"],
        models["fullcache384"],
        "crazyflow",
        prepared=prepared,
        candidate_module=module,
    )


def test_matching_accepts_recomputed_norms_and_weights_without_false_identity(matched):
    models, _, _, _ = matched
    assert not np.array_equal(
        models["candidate"]._model.norms["state_mean"],
        models["balanced"]._model.norms["state_mean"],
    )
    assert (
        models["candidate"].report["optimization"]["objective"]
        != models["balanced"].report["optimization"]["objective"]
    )
    result = match(matched)
    assert result["training_prefix_exact"] and result["same_development_cache"]
    assert not result["same_training_cache"] and not result["same_numeric_objective"]
    assert result["gradient"]["known_gradient_window_visits"] == 1536000
    assert result["same_objective_formula_arms"] == [
        "candidate",
        "balanced",
        "fullcache384",
    ]
    assert set(result["reference_fingerprints"]) == {
        "baseline",
        "balanced",
        "fullcache384",
    }


@pytest.mark.parametrize(
    "change",
    [
        "tail",
        "prefix",
        "development",
        "origins",
        "roles",
        "norms",
        "coherent_norms",
        "scale",
        "ridge",
        "preparation",
        "recipe",
        "gradient",
        "weights",
        "reference",
        "shape",
        "declaration",
    ],
)
def test_matching_rejects_cache_and_derived_contract_corruption(matched, change):
    models, prepared, _, p = matched
    candidate = models["candidate"]
    if change in ("tail", "prefix", "development"):
        attribute = "_development" if change == "development" else "_train"
        cache = getattr(candidate, attribute)
        values = cache.batch.future_inputs.copy()
        values[1000 if change == "tail" else 0, 0, 0] += 1
        setattr(
            candidate,
            attribute,
            replace(cache, batch=replace(cache.batch, future_inputs=values)),
        )
    elif change == "origins":
        candidate._train = replace(
            candidate._train,
            source_origins=tuple(i + 1 for i in candidate._train.source_origins),
        )
    elif change == "roles":
        prepared.provenance["roles"]["training"].pop()
    elif change in ("norms", "coherent_norms"):
        candidate._model.norms["state_scale"][0] *= 2
        if change == "coherent_norms":
            prepared.norms["state_scale"][0] *= 2
    elif change == "scale":
        candidate.report["optimization"]["error_scale"][0][0] *= 2
    elif change == "ridge":
        prepared.ridge /= 4
    elif change == "preparation":
        prepared.provenance["training_windows"] = 384
        candidate.report["preparation"]["training_windows"] = 384
    elif change == "recipe":
        candidate.recipe["width"] = 99
    elif change == "gradient":
        candidate.report["optimization"]["gradient"]["known_gradient_window_visits"] = (
            384000
        )
    elif change == "weights":
        candidate.report["optimization"]["objective"]["channel_weights"][0] *= 2
    elif change == "reference":
        p["imported_references"]["trusted_comparator_revisions"]["crazyflow"][
            "fullcache384"
        ]["revision_fingerprint"] = "wrong"
    elif change == "shape":
        candidate._model.params["bias"] = np.zeros(3)
    else:
        candidate._train = replace(
            candidate._train,
            past_excitation=np.zeros_like(candidate._train.batch.past_inputs),
            future_excitation=np.zeros_like(candidate._train.batch.future_inputs),
        )
    with pytest.raises(ValueError):
        match(matched)


def test_resolver_removes_current_full_initial_identity_and_preserves_public_recipe():
    p = experiment.protocol()
    assert "trusted_initial_snapshots" not in p["fitting"]
    assert "current-data" in p["fitting"]["normalization"]
    assert "1536" in p["fitting"]["mechanism"]["initial_prediction"]
    assert "five-array" in p["checkpoint_diagnostics"]["initialization"]
    assert p["fitting"]["generic_recipe"]["training_windows"] == 384
    assert p["fitting"]["candidate_recipe_overrides"]["training_windows"] == 1536
    assert p["data_support"]["development_windows"] == 256


def runner_preparation_case(monkeypatch, tmp_path, *, failure=None):
    events = []
    train, development, captured = object(), object(), object()
    prepared = SimpleNamespace(
        train=train, development=development, provenance={"current": True}
    )
    runtime = {"checkpoint_source_sha256": {"source": "a" * 64}}
    directory = tmp_path / "crazyflow/candidate"
    report = dict(
        optimization=dict(
            selected_step=100,
            trace=[{"step": 0}, {"step": 100}],
            unweighted_development_trace=[],
            objective={},
            safeguard={},
            gradient={},
        )
    )
    report["timing"] = dict(
        fit_and_capture_wall_time_s=0.1, calibration_wall_time_s=0.02
    )
    model = SimpleNamespace(
        report=report,
        contract={"test": True},
        _model=object(),
        save=lambda path: path.write_bytes(b"toy model"),
    )
    monkeypatch.setattr(
        experiment,
        "protocol",
        lambda: {
            "fitting": {"generic_recipe": learner.RECIPE},
            "checkpoint_diagnostics": {"steps": [0, 100]},
        },
    )
    monkeypatch.setattr(experiment, "data_ready", lambda output: runtime)
    monkeypatch.setattr(experiment, "check_sources", lambda: runtime)
    original_check = experiment.check_checkpoints
    monkeypatch.setattr(
        experiment, "check_checkpoints", lambda *a, **kw: events.append("reference")
    )
    monkeypatch.setattr(experiment, "fit_provenance", lambda *a: {"frozen": True})

    def prepare(*args):
        events.append("prepare")
        if failure == "preparation":
            raise candidate_module.PreparationUnavailable(
                {"available_training_windows": 1000}
            )
        if failure == "integrity":
            raise ValueError("synthetic source ledger mismatch")
        return prepared

    monkeypatch.setattr(experiment, "_prepare", prepare)

    def save_preparation(path, actual):
        assert actual is prepared and path == directory / "preparation.npz"
        events.append("save_preparation")
        path.write_bytes(b"prepared cache")

    monkeypatch.setattr(candidate_module, "save_preparation", save_preparation)

    def verify_preparation(path, actual):
        assert actual is prepared and path == directory / "preparation.npz"
        assert path.read_bytes() == b"prepared cache"
        events.append("verify_preparation")

    monkeypatch.setattr(candidate_module, "verify_preparation", verify_preparation)
    context = {"typed_current_data": True}

    def initialization_context(simulator, output, actual):
        assert actual is prepared
        events.append("context")
        return context

    monkeypatch.setattr(experiment, "initialization_context", initialization_context)

    @contextmanager
    def capture_fit(*args, **kwargs):
        assert "save_preparation" in events
        events.append("capture_enter")
        try:
            yield captured
        finally:
            events.append("capture_exit")

    monkeypatch.setattr(observer, "capture_fit", capture_fit)

    def fit(actual):
        assert actual is prepared and events[-1] == "capture_enter"
        assert (directory / "preparation.npz").read_bytes() == b"prepared cache"
        events.append("fit")
        if failure == "numeric":
            raise candidate_module.CandidateFitError("synthetic numeric failure")
        return model

    monkeypatch.setattr(candidate_module, "fit_candidate", fit)
    monkeypatch.setattr(experiment, "compare_candidate", lambda *a: {"matched": True})

    def save_checkpoints(path, actual, **kwargs):
        assert actual is captured
        assert kwargs["train"] is train and kwargs["development"] is development
        assert kwargs["initialization_context"] is context
        assert kwargs["failure"] == (
            "synthetic numeric failure" if failure == "numeric" else None
        )
        assert kwargs["selected_model"] is (
            None if failure == "numeric" else model._model
        )
        events.append("save_checkpoints")
        path.mkdir()
        (path / "manifest.json").write_text("{}")
        return {"manifest_sha256": "b" * 64}

    monkeypatch.setattr(observer, "save_checkpoints", save_checkpoints)
    monkeypatch.setattr(
        experiment,
        "_model_classes",
        lambda: {
            "candidate": SimpleNamespace(load=lambda path: model),
        },
    )

    def expectation(outcome, selected, arm, recipe):
        assert selected is (None if failure == "numeric" else model)
        return {"status": outcome["status"]}

    monkeypatch.setattr(
        experiment, "_runner", lambda: SimpleNamespace(_outcome_expectation=expectation)
    )

    def check(path, **kwargs):
        assert kwargs["train"] is train and kwargs["development"] is development
        assert kwargs["initialization_context"] is context
        assert "verify_preparation" in events
        assert kwargs["expectation"]["gradient"] == (
            None if failure == "numeric" else {}
        )
        events.append("checker")
        return {"verified": True}

    monkeypatch.setattr(observer, "check_checkpoint_links", check)
    monkeypatch.setattr(observer, "replay_checkpoints", check)
    return (
        directory,
        events,
        lambda replay=False: original_check(
            tmp_path, "crazyflow", "candidate", replay=replay
        ),
    )


@pytest.mark.parametrize("failure", [None, "numeric"])
@pytest.mark.parametrize("replay", [False, True])
def test_runner_persists_actual_preparation_before_fit_and_reuses_it_on_failure(
    monkeypatch, tmp_path, failure, replay
):
    directory, events, check = runner_preparation_case(
        monkeypatch, tmp_path, failure=failure
    )
    experiment.fit_arm("crazyflow", "candidate", tmp_path)
    assert (
        events.index("save_preparation")
        < events.index("capture_enter")
        < events.index("fit")
        < events.index("save_checkpoints")
    )
    assert check(replay) == {"verified": True}
    outcome = experiment.read_json(directory / "outcome.json")
    assert outcome["status"] == ("fit_failure" if failure else "complete")
    seal = experiment.read_json(directory / "seal.json")
    assert "preparation.npz" in seal["files"]
    assert seal["preparation_wall_time_s"] >= 0
    assert seal["fit_and_capture_and_calibration_wall_time_s"] >= 0
    assert seal["fit_and_capture_wall_time_s"] == (None if failure else 0.1)
    assert seal["calibration_wall_time_s"] == (None if failure else 0.02)


def test_runner_unavailable_preparation_never_enters_capture_fit_or_cache_validator(
    monkeypatch, tmp_path
):
    directory, events, check = runner_preparation_case(
        monkeypatch, tmp_path, failure="preparation"
    )
    experiment.fit_arm("crazyflow", "candidate", tmp_path)
    result = check(True)
    assert (
        result["status"] == "preparation_unavailable" and result["fit_entered"] is False
    )
    assert not {
        "save_preparation",
        "context",
        "capture_enter",
        "fit",
        "save_checkpoints",
        "verify_preparation",
        "checker",
    } & set(events)
    assert not (directory / "preparation.npz").exists()
    assert not (directory / "checkpoints").exists()
    outcome = experiment.read_json(directory / "outcome.json")
    assert (
        outcome["failure_stage"] == "preparation"
        and outcome["checkpoint_manifest_sha256"] is None
    )


def test_runner_integrity_failure_is_not_serialized_as_preparation_unavailable(
    monkeypatch, tmp_path
):
    directory, events, _ = runner_preparation_case(
        monkeypatch, tmp_path, failure="integrity"
    )
    with pytest.raises(ValueError, match="source ledger"):
        experiment.fit_arm("crazyflow", "candidate", tmp_path)
    assert "capture_enter" not in events and "fit" not in events
    assert not (directory / "outcome.json").exists()
    assert not (directory / "preparation-unavailable.json").exists()


@pytest.mark.parametrize(
    "artifact", ["preparation.npz", "report.json", "matching.json", "contract.json"]
)
def test_runner_unavailable_preparation_rejects_success_only_artifacts(
    monkeypatch, tmp_path, artifact
):
    directory, _, check = runner_preparation_case(
        monkeypatch, tmp_path, failure="preparation"
    )
    experiment.fit_arm("crazyflow", "candidate", tmp_path)
    (directory / artifact).write_bytes(b"stale success evidence")
    with pytest.raises(ValueError, match="preparation-unavailable"):
        check()
