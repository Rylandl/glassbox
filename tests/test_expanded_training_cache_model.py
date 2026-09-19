"""Expanded preparation, numerical reuse and archive integrity on toy recordings."""

import copy
from dataclasses import replace

import jax
import numpy as np
import pytest

from glassbox import learner
from glassbox._learner_arrays import load_arrays, save_arrays
from glassbox._sequence_model import initialize_sequence_model
from glassbox.experimental import expanded_training_cache_model as expanded
from glassbox.experimental import full_cache_gradient_model as historical
from glassbox.recordings import SequenceCollection, SequenceSegment


@pytest.fixture(autouse=True)
def float64():
    with jax.enable_x64(True):
        yield


@pytest.fixture
def case(monkeypatch):
    recipe = {
        **expanded.FITTING_RECIPE,
        "training_windows": 48,
        "development_windows": 8,
        "batch_size": 48,
        "width": 4,
        "memory": 2,
        "steps": 2,
        "check_every": 1,
    }
    monkeypatch.setattr(expanded, "FITTING_RECIPE", recipe)
    monkeypatch.setattr(expanded, "ROLE_COUNTS", {"training": 3, "development": 1})
    monkeypatch.setattr(expanded, "REFERENCE_TRAINING_WINDOWS", 12)
    rng = np.random.default_rng(612)
    segments = []
    for i in range(4):
        commands = rng.normal(size=(38 + 7 * i, 2))
        states = [rng.normal(size=3)]
        for a, b in commands:
            x, y, z = states[-1]
            states.append(
                [
                    0.94 * x + 0.04 * a,
                    0.89 * y + 0.03 * b + 0.02 * x,
                    0.9 * z + 0.01 * a * b,
                ]
            )
        segments.append(
            SequenceSegment(
                f"parent-{i}", "valid-prefix", np.asarray(states), commands, 0.05
            )
        )
    collection = SequenceCollection(
        tuple(segments),
        configuration_id="synthetic-expanded",
        state_channels=("x:m", "y:m", "z:m"),
        input_channels=("a:1", "b:1"),
    )
    seen = learner._recording_content(collection)
    names = sorted(seen, key=learner._priority)
    roles = dict(development=names[:1], training=names[1:])
    train = learner._extract(collection, roles["training"], 12)
    dev = learner._extract(collection, roles["development"], 8)
    # Truthful synthetic step-zero archive, not a claimed public optimizer run.
    initial = initialize_sequence_model(
        train.batch, width=4, memory=2, ridge=0.6, delay_steps=2
    )
    public = learner.LearnedDynamics(
        initial,
        train,
        dev,
        learner._contract(collection),
        seen,
        {"recipe": copy.deepcopy(learner.RECIPE), "previous_revision": None},
        np.ones((5, 3)),
    )
    seals = {key: "a" * 64 for key in expanded.DATA_SEAL_KEYS}
    return collection, public, roles, seals


def prepare(case):
    collection, public, roles, seals = case
    return expanded.prepare(collection, public, roles=roles, data_seal=seals)


def forbidden(*args, **kwargs):
    raise AssertionError(
        "preparation/replay must not initialize, draw, forecast or solve"
    )


def test_private_numerical_code_and_historical_globals_unchanged():
    assert (
        expanded.fit_candidate_sequence.__code__
        == historical.fit_candidate_sequence.__code__
    )
    assert expanded.fit_candidate_sequence.__globals__ is vars(expanded._engine)
    for name in (
        "initialize_candidate",
        "initial_training_forecast",
        "weighting_metadata",
        "make_training_objective",
        "trial_parameters",
        "_rollout",
    ):
        assert getattr(expanded, name) is getattr(historical, name)
    assert (
        historical.RECIPE["training_windows"] == historical.RECIPE["batch_size"] == 384
    )
    assert expanded.RECIPE == {
        **historical.RECIPE,
        "id": expanded.EXPERIMENT,
        "training_windows": 1536,
        "batch_size": 1536,
    }
    assert expanded.FIT_WALL_TIME_LIMIT_S == 7200
    assert learner.RECIPE["training_windows"] == 384
    assert learner.RECIPE["batch_size"] == 64


def test_prepare_exact_prefix_development_and_no_numerical_work(case, monkeypatch):
    monkeypatch.setattr(np.random, "default_rng", forbidden)
    monkeypatch.setattr(np.linalg, "solve", forbidden)
    monkeypatch.setattr(expanded._engine, "initialize_candidate", forbidden)
    monkeypatch.setattr(expanded._engine, "initial_training_forecast", forbidden)
    p = prepare(case)
    assert (
        len(p.train.keys) == len(set(zip(p.train.keys, p.train.source_origins))) == 48
    )
    expanded._equal_windows(
        learner._subset(p.train, list(range(12))), case[1]._train, "prefix"
    )
    expanded._equal_windows(p.development, case[1]._development, "development")
    assert p.ridge == 2.4 and p.delay == 2
    assert p.provenance["added_training_windows"] == 36
    assert set(p.provenance["support"]["per_parent_selected_origins"].values()) == {16}
    assert p.provenance["same_numeric_objective"] is False
    assert p.provenance["same_all_initial_parameters_required"] is False
    assert p.provenance["recording_content_fingerprints"] == case[1]._seen
    assert not np.array_equal(p.norms["state_mean"], case[1]._model.norms["state_mean"])


def test_all_eight_moments_match_actual_initializer_without_redundant_rng(case):
    p = prepare(case)
    model = expanded.initialize_candidate(
        p.train.batch, width=4, memory=2, ridge=p.ridge, delay_steps=p.delay
    )
    assert set(p.norms) == set(model.norms)
    for key in p.norms:
        np.testing.assert_array_equal(p.norms[key], model.norms[key])
    assert p.provenance["parameter_count"] == sum(v.size for v in model.params.values())


def test_one_initializer_two_solves_one_rng_and_actual_full_cache(case, monkeypatch):
    p = prepare(case)
    calls = {"initializer": 0, "solve": 0, "rng": 0}
    init, solve, rng = (
        expanded._engine.initialize_candidate,
        np.linalg.solve,
        np.random.default_rng,
    )

    def initialize(*args, **kwargs):
        calls["initializer"] += 1
        assert args[0] is p.train.batch and kwargs["ridge"] == 2.4
        return init(*args, **kwargs)

    def solve_once(*args, **kwargs):
        calls["solve"] += 1
        return solve(*args, **kwargs)

    def rng_once(*args, **kwargs):
        calls["rng"] += 1
        return rng(*args, **kwargs)

    events = []
    monkeypatch.setattr(expanded._engine, "initialize_candidate", initialize)
    monkeypatch.setattr(np.linalg, "solve", solve_once)
    monkeypatch.setattr(np.random, "default_rng", rng_once)
    monkeypatch.setattr(expanded._engine, "_observe_attempt", events.append)
    model = expanded.fit_candidate(p)
    assert calls == {"initializer": 1, "solve": 2, "rng": 1}
    starts = [event for event in events if event["phase"] == "started"]
    assert len(starts) == 2
    for event in starts:
        assert event["indices"].dtype == np.dtype("int64")
        np.testing.assert_array_equal(event["indices"], np.arange(48))
    report = model.report
    assert report["preparation"] == p.provenance and "matching" not in report
    assert report["optimization"]["gradient"]["known_gradient_window_visits"] == 96
    assert report["optimization"]["safeguard"]["id"] == expanded.EXPERIMENT
    assert all(value >= 0 for value in report["timing"].values())


def test_private_fitter_exact_toy_parity_with_original_full_cache(case):
    p = prepare(case)
    kwargs = dict(
        seed=0,
        steps=2,
        batch_size=48,
        learning_rate=0.002,
        width=4,
        memory=2,
        ridge=p.ridge,
        check_every=1,
        error_scale=p.error_scale,
        delay_steps=p.delay,
    )
    left, a = expanded.fit_candidate_sequence(
        p.train.batch, p.development.batch, **kwargs
    )
    right, b = historical.fit_candidate_sequence(
        p.train.batch, p.development.batch, **kwargs
    )
    for key, value in left.arrays().items():
        np.testing.assert_array_equal(value, right.arrays()[key])
    b["safeguard"]["id"] = expanded.EXPERIMENT
    b["mechanism"] = expanded.EXPERIMENT
    assert a == b


def test_preparation_and_candidate_archives_replay_without_numerical_work(
    case, monkeypatch, tmp_path
):
    p = prepare(case)
    prep_path, model_path = tmp_path / "preparation.npz", tmp_path / "model.npz"
    expanded.save_preparation(prep_path, p)
    model = expanded.fit_candidate(p)
    model.save(model_path)
    restored = expanded.CandidateDynamics.load(model_path)
    assert restored.fingerprint() == model.fingerprint()
    with pytest.raises(ValueError, match="unsupported"):
        historical.CandidateDynamics.load(model_path)
    monkeypatch.setattr(np.random, "default_rng", forbidden)
    monkeypatch.setattr(np.linalg, "solve", forbidden)
    monkeypatch.setattr(expanded, "fit_candidate_sequence", forbidden)
    monkeypatch.setattr(expanded.BalancedQuadraticSequenceModel, "rollout", forbidden)
    assert expanded.verify_preparation(prep_path, prepare(case))
    assert (
        expanded.check_preparation(
            restored, case[0], case[1], roles=case[2], data_seal=case[3]
        )
        == p.provenance
    )


@pytest.mark.parametrize("key", expanded._ARRAYS)
def test_same_prefix_corrupted_added_tail_rejected(case, tmp_path, key):
    p = prepare(case)
    path = tmp_path / "preparation.npz"
    expanded.save_preparation(path, p)
    metadata, arrays = load_arrays(path)
    arrays["train_" + key][-1, -1, 0] += 0.1
    save_arrays(path, metadata, arrays)
    with pytest.raises(ValueError, match="reconstructed"):
        expanded.verify_preparation(path, p)
    values = getattr(p.train.batch, key).copy()
    values[-1, -1, 0] += 0.1
    changed = replace(
        p, train=replace(p.train, batch=replace(p.train.batch, **{key: values}))
    )
    with pytest.raises(ValueError, match="fingerprint"):
        expanded.fit_candidate(changed)


@pytest.mark.parametrize("field", ["inputs", "states"])
@pytest.mark.parametrize("role", ["training", "development"])
def test_changed_source_content_rejected(case, field, role):
    collection, public, roles, seals = case
    name = roles[role][0]
    segments = []
    for segment in collection.segments:
        if segment.recording_id == name:
            values = getattr(segment, field).copy()
            values[-1, 0] += 0.001
            segment = replace(segment, **{field: values})
        segments.append(segment)
    with pytest.raises(ValueError, match="ledger"):
        expanded.prepare(
            replace(collection, segments=tuple(segments)),
            public,
            roles=roles,
            data_seal=seals,
        )


def test_roles_contract_and_reference_cache_reject(case):
    collection, public, roles, seals = case
    changed = copy.deepcopy(roles)
    changed["training"][0], changed["development"][0] = (
        changed["development"][0],
        changed["training"][0],
    )
    with pytest.raises(ValueError, match="roles"):
        expanded.prepare(collection, public, roles=changed, data_seal=seals)
    with pytest.raises(ValueError, match="contract"):
        expanded.prepare(
            replace(collection, configuration_id="different"),
            public,
            roles=roles,
            data_seal=seals,
        )
    public._train = learner._subset(public._train, list(reversed(range(12))))
    with pytest.raises(ValueError, match="original training"):
        prepare(case)


def test_insufficient_support_has_actual_counts_and_no_initializer(case, monkeypatch):
    monkeypatch.setattr(
        expanded,
        "FITTING_RECIPE",
        {**expanded.FITTING_RECIPE, "training_windows": 10000, "batch_size": 10000},
    )
    monkeypatch.setattr(expanded._engine, "initialize_candidate", forbidden)
    with pytest.raises(expanded.PreparationUnavailable) as caught:
        prepare(case)
    d = caught.value.diagnostics
    assert d["required_training_windows"] == 10000
    assert d["available_training_windows"] == sum(
        d["per_parent_legal_origins"].values()
    )
    assert 48 < d["available_training_windows"] < 10000
    assert d["source_ledger_exact"] and d["development_cache_exact"]


@pytest.mark.parametrize("field", ["ridge", "delay", "error_scale", "norms"])
def test_prepared_derived_mutations_fail_before_fit(case, monkeypatch, field):
    p = prepare(case)
    values = {
        "ridge": p.ridge / 4,
        "delay": p.delay + 1,
        "error_scale": p.error_scale * 2,
        "norms": {**p.norms, "state_mean": p.norms["state_mean"] + 0.1},
    }
    monkeypatch.setattr(expanded, "fit_candidate_sequence", forbidden)
    with pytest.raises(ValueError, match="prepared"):
        expanded.fit_candidate(replace(p, **{field: values[field]}))


def test_numerical_failure_leaves_prepared_actual_caches_available(case, monkeypatch):
    p = prepare(case)

    def fail(train, development, **kwargs):
        assert train is p.train.batch and development is p.development.batch
        assert len(train.past_states) == kwargs["batch_size"] == 48
        raise expanded.CandidateFitError("synthetic numerical failure")

    monkeypatch.setattr(expanded, "fit_candidate_sequence", fail)
    with pytest.raises(expanded.CandidateFitError):
        expanded.fit_candidate(p)
    assert len(p.train.keys) == 48 and len(p.development.keys) == 8


def test_declared_excitation_is_preserved_not_consumed(case, tmp_path):
    collection, public, roles, seals = case
    collection = replace(
        collection,
        segments=tuple(
            replace(segment, excitation=np.full_like(segment.inputs, 0.05))
            for segment in collection.segments
        ),
    )
    public._train = learner._extract(collection, roles["training"], 12)
    public._development = learner._extract(collection, roles["development"], 8)
    p = expanded.prepare(collection, public, roles=roles, data_seal=seals)
    assert p.train.excitation_declared
    model = expanded.fit_candidate(p)
    path = tmp_path / "excitation-model.npz"
    model.save(path)
    restored = expanded.CandidateDynamics.load(path)
    assert restored.fingerprint() == model.fingerprint()
    expanded._equal_windows(restored._train, p.train, "excitation")
    prep = tmp_path / "preparation.npz"
    expanded.save_preparation(prep, p)
    expanded.verify_preparation(prep, p)


@pytest.mark.parametrize("change", ["missing", "extra", "invalid", "string"])
def test_unbound_or_malformed_seals_are_rejected(case, change):
    collection, public, roles, seals = case
    seals = copy.deepcopy(seals)
    if change == "missing":
        seals.pop(next(iter(seals)))
    elif change == "extra":
        seals["unfrozen_anchor"] = "a" * 64
    elif change == "invalid":
        seals[next(iter(seals))] = "z" * 64
    else:
        seals = "a" * 64
    with pytest.raises(ValueError, match="authenticated SHA256"):
        expanded.prepare(collection, public, roles=roles, data_seal=seals)


@pytest.mark.parametrize(
    "field", ["norm_state_scale", "error_scale", "train_future_excitation"]
)
def test_preparation_norm_scale_and_unexpected_sidecar_tamper_reject(
    case, tmp_path, field
):
    prepared = prepare(case)
    path = tmp_path / "preparation.npz"
    expanded.save_preparation(path, prepared)
    metadata, arrays = load_arrays(path)
    if field in arrays:
        arrays[field] *= 1.01
    else:
        arrays[field] = np.zeros_like(arrays["train_future_inputs"])
    save_arrays(path, metadata, arrays)
    with pytest.raises(ValueError, match="reconstructed"):
        expanded.verify_preparation(path, prepared)


def test_empty_or_untrusted_source_is_integrity_failure(case):
    collection, public, roles, seals = case
    reduced = replace(collection, segments=collection.segments[:-1])
    with pytest.raises(ValueError, match="ledger") as caught:
        expanded.prepare(reduced, public, roles=roles, data_seal=seals)
    assert not isinstance(caught.value, expanded.PreparationUnavailable)
