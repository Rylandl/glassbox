"""Direct-fit input matching and saved preparation integrity, using tiny data."""

import copy
from dataclasses import replace

import jax
import numpy as np
import pytest

from glassbox import learner
from glassbox.experimental import command_excitation_fit as fit_module
from glassbox.experimental import state_quadratic_model as quadratic
from glassbox.recordings import SequenceCollection, SequenceSegment


@pytest.fixture(autouse=True)
def float64():
    with jax.enable_x64(True):
        yield


def _trajectory(inputs, start):
    states = [start]
    for command in inputs[:, 0]:
        x, y = states[-1]
        states.append(np.array([0.96 * x + 0.09 * command, 0.91 * y + 0.03 * x]))
    return np.asarray(states)


@pytest.fixture
def case(monkeypatch):
    recipe = {
        **fit_module.FITTING_RECIPE,
        "training_windows": 12,
        "development_windows": 4,
        "width": 4,
        "memory": 2,
        "steps": 2,
        "batch_size": 4,
        "check_every": 1,
    }
    monkeypatch.setattr(fit_module, "FITTING_RECIPE", recipe)
    monkeypatch.setattr(fit_module, "ROLE_COUNTS", {"training": 3, "development": 1})
    names = sorted([f"parent-{i}" for i in range(4)], key=learner._priority)
    roles = {"development": names[:1], "training": names[1:]}
    segments = []
    rng = np.random.default_rng(49)
    for name in names:
        commands = rng.normal(size=(45, 1)) * 0.3
        states = _trajectory(commands, rng.normal(size=2))
        segments.append(SequenceSegment(name, "valid-prefix", states, commands, 0.05))
    original_collection = SequenceCollection(
        tuple(segments),
        configuration_id="test",
        state_channels=("state:x:m", "state:y:m"),
        input_channels=("input:u:1",),
    )
    train = learner._extract(original_collection, roles["training"], 12)
    development = learner._extract(original_collection, roles["development"], 4)
    initial = quadratic.initialize_candidate(
        train.batch,
        width=4,
        memory=2,
        ridge=0.6,
        delay_steps=2,
    )
    hold = np.repeat(train.batch.past_states[:, -1:], 5, axis=1)
    error_scale = np.maximum(
        np.sqrt(np.mean((hold - train.batch.future_states) ** 2, axis=0)),
        recipe["hold_scale_floor"] * initial.norms["state_scale"],
    )
    original = quadratic.CandidateDynamics(
        initial,
        train,
        development,
        learner._contract(original_collection),
        {
            "recipe": copy.deepcopy(quadratic.RECIPE),
            "optimization": {"error_scale": error_scale.tolist()},
        },
        np.ones((5, 2)),
    )
    excited_segments = []
    for segment in segments:
        if segment.recording_id in roles["development"]:
            excited_segments.append(segment)
            continue
        command = segment.inputs.copy()
        command[15:, 0] += 0.18 * (2 * rng.integers(0, 2, size=30) - 1)
        excited_segments.append(
            replace(
                segment,
                inputs=command,
                states=_trajectory(command, segment.states[0]),
            )
        )
    excited = replace(original_collection, segments=tuple(excited_segments))
    return excited, original, roles, "a" * 64


def _fit(case):
    collection, original, roles, seal = case
    return fit_module.fit_excited(collection, original, roles=roles, data_seal=seal)


def test_direct_fit_uses_one_existing_optimizer_and_exact_changed_windows(
    case, monkeypatch
):
    collection, original, roles, seal = case
    calls = []
    actual = quadratic.fit_candidate_sequence

    def spy(train, development, **kwargs):
        calls.append((train, development, kwargs))
        return actual(train, development, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("no public-model fit or fit_candidate cache proxy")

    monkeypatch.setattr(quadratic, "fit_candidate_sequence", spy)
    monkeypatch.setattr(quadratic, "fit_candidate", forbidden)
    monkeypatch.setattr(learner, "fit", forbidden)
    monkeypatch.setattr(learner, "_train", forbidden)
    model = _fit(case)
    assert len(calls) == 1
    train, development, settings = calls[0]
    expected = learner._extract(collection, roles["training"], 12)
    for name in fit_module._ARRAYS:
        np.testing.assert_array_equal(
            getattr(train, name), getattr(expected.batch, name)
        )
        np.testing.assert_array_equal(
            getattr(development, name),
            getattr(original._development.batch, name),
        )
    assert settings["steps"] == 2 and settings["batch_size"] == 4
    assert settings["seed"] == 0 and settings["delay_steps"] == 2
    assert settings["ridge"] == 0.6
    assert model.recipe == quadratic.RECIPE
    assert "matching" not in model.report
    p = model.report["provenance"]
    assert p["collection_intervention"] == fit_module.EXPERIMENT
    assert p["unchanged_development_reference"] == original.fingerprint()
    assert p["same_training_data"] is False
    assert p["same_development"] is True
    assert p["optimizer_fit_calls"] == 1
    assert p["data_seal"] == seal
    assert p["original_error_scale"] == original.report["optimization"]["error_scale"]
    assert not np.array_equal(settings["error_scale"], p["original_error_scale"])
    assert p["training_norms_fingerprint"] != p["original_training_norms_fingerprint"]
    assert (
        fit_module.check_preparation(
            model,
            collection,
            original,
            roles=roles,
            data_seal=seal,
        )
        == p
    )


def test_moment_and_rng_reconstruction_exactly_match_frozen_initializer(case):
    collection, _, roles, _ = case
    batch = learner._extract(collection, roles["training"], 12).batch
    norms, draws, delay, count = fit_module._training_moments(batch)
    initial = quadratic.initialize_candidate(
        batch,
        width=4,
        memory=2,
        ridge=0.6,
        delay_steps=delay,
    )
    assert set(norms) == set(initial.norms)
    for name, values in norms.items():
        np.testing.assert_array_equal(values, initial.norms[name])
    for name, values in draws.items():
        np.testing.assert_array_equal(values, initial.params[name])
    assert count == sum(v.size for v in initial.params.values())


def test_saved_preparation_replays_without_optimizer_or_ridge_solve(
    case, monkeypatch, tmp_path
):
    model = _fit(case)
    path = tmp_path / "model.npz"
    model.save(path)
    restored = quadratic.CandidateDynamics.load(path)
    assert restored.fingerprint() == model.fingerprint()

    def forbidden(*args, **kwargs):
        raise AssertionError("preparation replay must not solve or optimize")

    monkeypatch.setattr(np.linalg, "solve", forbidden)
    monkeypatch.setattr(quadratic, "initialize_candidate", forbidden)
    monkeypatch.setattr(quadratic, "fit_candidate_sequence", forbidden)
    collection, original, roles, seal = case
    fit_module.check_preparation(
        restored,
        collection,
        original,
        roles=roles,
        data_seal=seal,
    )


@pytest.mark.parametrize("field", ["states", "inputs"])
def test_changed_development_is_rejected_before_fitting(case, monkeypatch, field):
    collection, original, roles, seal = case
    segments = []
    for segment in collection.segments:
        if segment.recording_id in roles["development"]:
            values = getattr(segment, field).copy() + 0.001
            segment = replace(segment, **{field: values})
        segments.append(segment)
    changed = replace(collection, segments=tuple(segments))
    monkeypatch.setattr(
        quadratic, "fit_candidate_sequence", lambda *a, **k: pytest.fail("fit called")
    )
    with pytest.raises(ValueError, match="development window") as error:
        fit_module.fit_excited(changed, original, roles=roles, data_seal=seal)
    assert not isinstance(error.value, fit_module.ExcitationDataError)


def test_role_reshuffling_is_integrity_error(case):
    collection, original, roles, seal = case
    changed = copy.deepcopy(roles)
    changed["training"][0], changed["development"][0] = (
        changed["development"][0],
        changed["training"][0],
    )
    with pytest.raises(ValueError, match="automatic roles"):
        fit_module.fit_excited(collection, original, roles=changed, data_seal=seal)


def test_insufficient_valid_training_is_recordable_without_parent_replacement(case):
    collection, original, roles, seal = case
    segments = tuple(
        replace(s, states=s.states[:16], inputs=s.inputs[:15])
        if s.recording_id in roles["training"]
        else s
        for s in collection.segments
    )
    with pytest.raises(fit_module.ExcitationDataError, match="window/parent budget"):
        fit_module.fit_excited(
            replace(collection, segments=segments),
            original,
            roles=roles,
            data_seal=seal,
        )


def test_missing_training_parent_is_recordable_but_missing_development_is_integrity(
    case,
):
    collection, original, roles, seal = case
    for role in ("training", "development"):
        changed = replace(
            collection,
            segments=tuple(
                s for s in collection.segments if s.recording_id != roles[role][0]
            ),
        )
        with pytest.raises(ValueError, match=f"missing frozen {role}") as error:
            fit_module.fit_excited(changed, original, roles=roles, data_seal=seal)
        assert isinstance(error.value, fit_module.ExcitationDataError) == (
            role == "training"
        )


@pytest.mark.parametrize("alteration", ["norm", "loss", "provenance", "cache", "seal"])
def test_saved_preparation_rejects_coherent_local_mismatches(case, alteration):
    model = _fit(case)
    collection, original, roles, seal = case
    if alteration == "norm":
        norms = {key: value.copy() for key, value in model._model.norms.items()}
        norms["input_mean"] += 0.001
        model._model = replace(model._model, norms=norms)
    elif alteration == "loss":
        model._report["optimization"]["error_scale"][0][0] += 0.001
    elif alteration == "provenance":
        model._report["provenance"]["same_training_data"] = True
    elif alteration == "cache":
        batch = replace(
            model._train.batch, future_inputs=model._train.batch.future_inputs + 0.001
        )
        model._train = replace(model._train, batch=batch)
    else:
        seal = "b" * 64
    with pytest.raises(ValueError):
        fit_module.check_preparation(
            model, collection, original, roles=roles, data_seal=seal
        )


def test_contract_change_is_rejected(case):
    collection, original, roles, seal = case
    with pytest.raises(ValueError, match="signal/timing contract"):
        fit_module.fit_excited(
            replace(collection, input_channels=("different:u:1",)),
            original,
            roles=roles,
            data_seal=seal,
        )


def test_moment_reconstruction_keeps_constant_channel_floors(case):
    collection, _, roles, _ = case
    batch = learner._extract(collection, roles["training"], 12).batch
    batch = replace(
        batch,
        past_inputs=np.full_like(batch.past_inputs, 0.7),
        future_inputs=np.full_like(batch.future_inputs, 0.7),
        past_states=np.full_like(batch.past_states, -2.0),
        future_states=np.full_like(batch.future_states, -2.0),
    )
    norms, _, delay, _ = fit_module._training_moments(batch)
    initial = quadratic.initialize_candidate(
        batch, width=4, memory=2, ridge=0.6, delay_steps=delay
    )
    for name, values in norms.items():
        np.testing.assert_array_equal(values, initial.norms[name])
    np.testing.assert_array_equal(norms["state_scale"], 1)
    np.testing.assert_array_equal(norms["input_scale"], 1)
    np.testing.assert_array_equal(norms["delta_scale"], 1e-4)


def test_one_unrepresented_training_parent_cannot_be_dropped(case):
    collection, original, roles, seal = case
    changed = replace(
        collection,
        segments=tuple(
            replace(s, states=s.states[:10], inputs=s.inputs[:9])
            if s.recording_id == roles["training"][0]
            else s
            for s in collection.segments
        ),
    )
    with pytest.raises(fit_module.ExcitationDataError, match="no complete"):
        fit_module.fit_excited(changed, original, roles=roles, data_seal=seal)
