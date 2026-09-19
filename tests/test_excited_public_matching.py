"""Same-data witnesses from synthetic prepared artifacts, without optimizer fits."""

import copy
from dataclasses import replace

import numpy as np
import pytest

from glassbox import learner
from glassbox._sequence_model import initialize_sequence_model
from glassbox.experimental import command_excitation_fit as excitation
from glassbox.experimental import excited_public_matching as matching
from glassbox.experimental import state_quadratic_model as quadratic
from glassbox.recordings import SequenceCollection, SequenceSegment


def prepared_pair():
    """Construct truthful step-zero means with synthetic metadata for verifier tests.

    The reported optimizer trace is a test fixture, not a claim of performed
    training. These tests exercise saved evidence validation, never performance.
    """
    rng = np.random.default_rng(99)
    segments = []
    for index in range(96):
        inputs = rng.normal(size=(40, 1))
        states = [rng.normal(size=2)]
        for u in inputs[:, 0]:
            states.append(0.95 * states[-1] + np.array([0.04 * u, -0.02 * u]))
        segments.append(
            SequenceSegment(
                f"parent-{index}", "valid-prefix", np.asarray(states), inputs, 0.05
            )
        )
    collection = SequenceCollection(
        tuple(segments),
        configuration_id="synthetic",
        state_channels=("state:x:m", "state:y:m"),
        input_channels=("input:u:1",),
    )
    seen = learner._recording_content(collection)
    names = sorted(seen, key=learner._priority)
    roles = dict(development=names[:24], training=names[24:])
    train = learner._extract(collection, roles["training"], 384)
    development = learner._extract(collection, roles["development"], 256)
    ridge = 0.01 * 384 * 5
    public_mean = initialize_sequence_model(train.batch, ridge=ridge, delay_steps=2)
    quadratic_mean = quadratic.initialize_candidate(
        train.batch, ridge=ridge, delay_steps=2
    )
    hold = np.repeat(train.batch.past_states[:, -1:], 5, axis=1)
    scale = np.maximum(
        np.sqrt(np.mean((hold - train.batch.future_states) ** 2, axis=0)),
        0.01 * public_mean.norms["state_scale"],
    )
    optimization = dict(
        kind="filter_mlp",
        steps=1000,
        selected_step=0,
        validation_rollout_mse=1.0,
        seed=0,
        trace=[dict(step=0, validation_rollout_mse=1.0)]
        + [
            dict(step=i, validation_rollout_mse=2.0, training_batch_mse=1.0)
            for i in range(100, 1001, 100)
        ],
        loss_scale_mode="explicit_horizon_channel",
        error_scale=scale.tolist(),
        context_steps=10,
        delay_steps=2,
    )
    contract = learner._contract(collection)
    common = dict(
        history_steps=10,
        horizon_steps=5,
        delay_steps=2,
        training=train.coverage(),
        development=development.coverage(),
    )
    public = learner.LearnedDynamics(
        public_mean,
        train,
        development,
        contract,
        seen,
        dict(
            common,
            recipe=copy.deepcopy(learner.RECIPE),
            previous_revision=None,
            optimization=dict(
                optimization,
                parameter_count=sum(v.size for v in public_mean.params.values()),
            ),
        ),
        np.ones((5, 2)),
    )
    old_quadratic = quadratic.CandidateDynamics(
        quadratic_mean,
        train,
        development,
        contract,
        dict(common, recipe=copy.deepcopy(quadratic.RECIPE), optimization=optimization),
        np.ones((5, 2)),
    )
    preparation = excitation._prepare(
        collection, old_quadratic, roles=roles, data_seal="a" * 64
    )
    model = quadratic.CandidateDynamics(
        quadratic_mean,
        train,
        development,
        contract,
        dict(
            common,
            recipe=copy.deepcopy(quadratic.RECIPE),
            provenance=preparation.provenance,
            optimization=dict(
                optimization,
                parameter_count=sum(v.size for v in quadratic_mean.params.values()),
                mechanism="autonomous-state-quadratic-v1",
                minibatch_seed=10000,
            ),
        ),
        np.ones((5, 2)),
    )
    return public, model, roles


@pytest.fixture
def pair(monkeypatch):
    public, model, roles = prepared_pair()
    protocol = matching._protocol()
    protocol["planned_automatic_roles"] = {"synthetic": roles}
    protocol["imported_calibration"]["excitation_seal_sha256"] = {"synthetic": "a" * 64}
    monkeypatch.setattr(matching, "_protocol", lambda: copy.deepcopy(protocol))
    return public, model


def test_matching_is_deterministic_and_has_no_fit_solve_or_rollout(
    pair, monkeypatch, tmp_path
):
    public, quadratic_model = pair

    def forbidden(*args, **kwargs):
        raise AssertionError("saved matching cannot fit, solve or execute a mean")

    monkeypatch.setattr(np.linalg, "solve", forbidden)
    monkeypatch.setattr(learner, "fit", forbidden)
    monkeypatch.setattr(learner, "_train", forbidden)
    monkeypatch.setattr(quadratic, "fit_candidate_sequence", forbidden)
    monkeypatch.setattr(type(public._model), "rollout", forbidden)
    monkeypatch.setattr(type(quadratic_model._model), "rollout", forbidden)
    expected = matching.check_matched_models(public, quadratic_model)
    assert expected["window_counts"] == {"training": 384, "development": 256}
    assert expected["exact_same_data"] and expected["exact_same_numeric_objective"]
    assert expected["equal_flops_claimed"] is False
    assert len(expected["source_mechanics"]["source_sha256"]) == 8
    assert (
        expected["optimizations"]["quadratic"]["parameter_count"]
        > expected["optimizations"]["public"]["parameter_count"]
    )
    public.save(tmp_path / "public.npz")
    quadratic_model.save(tmp_path / "quadratic.npz")
    restored = matching.check_matched_models(
        learner.LearnedDynamics.load(tmp_path / "public.npz"),
        quadratic.CandidateDynamics.load(tmp_path / "quadratic.npz"),
    )
    assert restored == expected


@pytest.mark.parametrize(
    "role,field",
    [
        ("_train", "past_inputs"),
        ("_train", "future_inputs"),
        ("_development", "past_states"),
        ("_development", "future_states"),
    ],
)
def test_changed_cached_commands_or_observations_rejected(pair, role, field):
    public, model = pair
    windows = getattr(public, role)
    values = getattr(windows.batch, field).copy()
    values[0, 0, 0] += 0.001
    setattr(
        public, role, replace(windows, batch=replace(windows.batch, **{field: values}))
    )
    with pytest.raises(ValueError, match="cache keys, origins, dt or arrays"):
        matching.check_matched_models(public, model)


@pytest.mark.parametrize("change", ["key_order", "origin", "dt", "count"])
def test_cache_identity_and_count_cannot_be_replaced_by_equal_values(pair, change):
    public, model = pair
    windows = public._train
    if change == "key_order":
        public._train = replace(windows, keys=tuple(reversed(windows.keys)))
    elif change == "origin":
        origins = list(windows.source_origins)
        origins[0] += 1
        public._train = replace(windows, source_origins=tuple(origins))
    elif change == "dt":
        public._train = replace(windows, batch=replace(windows.batch, dt_s=0.1))
    else:
        public._train = learner._subset(windows, np.arange(383))
    with pytest.raises(ValueError, match="cache"):
        matching.check_matched_models(public, model)


@pytest.mark.parametrize(
    "arm,norm", [(0, "input_mean"), (1, "state_scale"), (1, "interaction_scale")]
)
def test_shared_or_product_normalization_must_derive_from_training(pair, arm, norm):
    target = pair[arm]
    norms = {key: value.copy() for key, value in target._model.norms.items()}
    norms[norm] += 0.001
    target._model = replace(target._model, norms=norms)
    with pytest.raises(ValueError, match="normalization"):
        matching.check_matched_models(*pair)


def test_identical_but_wrong_shared_norms_are_rejected(pair):
    for model in pair:
        norms = {key: value.copy() for key, value in model._model.norms.items()}
        norms["input_scale"] += 0.1
        model._model = replace(model._model, norms=norms)
    with pytest.raises(ValueError, match="normalization"):
        matching.check_matched_models(*pair)


@pytest.mark.parametrize(
    "change",
    [
        "content",
        "extra_parent",
        "roles",
        "seal",
        "contract",
        "loss_scale",
        "recipe",
        "budget",
        "checkpoint",
        "shape",
        "update",
    ],
)
def test_metadata_mismatches_are_rejected(pair, change):
    public, model = pair
    if change == "content":
        public._seen[next(iter(public._seen))] = "b" * 64
    elif change == "extra_parent":
        public._seen["foreign-parent"] = "b" * 64
        model._report["provenance"]["recording_content_fingerprints"] = (
            public._seen.copy()
        )
    elif change == "roles":
        roles = model._report["provenance"]["roles"]
        roles["training"][0], roles["development"][0] = (
            roles["development"][0],
            roles["training"][0],
        )
    elif change == "seal":
        model._report["provenance"]["data_seal"] = "b" * 64
    elif change == "contract":
        public._contract["input_channels"] = ["other"]
    elif change == "loss_scale":
        public._report["optimization"]["error_scale"][0][0] += 0.1
    elif change == "recipe":
        public._report["recipe"]["batch_size"] = 32
    elif change == "budget":
        public._report["optimization"]["steps"] = 999
    elif change == "checkpoint":
        public._report["optimization"]["selected_step"] = 100
    elif change == "shape":
        public._model.params["foreign_parameter"] = np.zeros(1)
    elif change == "update":
        public._report["previous_revision"] = "b" * 64
    with pytest.raises(ValueError):
        matching.check_matched_models(*pair)


def test_wrong_model_classes_are_not_silently_interchanged(pair):
    with pytest.raises(TypeError, match="public"):
        matching.check_matched_models(pair[1], pair[0])
    with pytest.raises(TypeError, match="quadratic"):
        matching.check_matched_models(pair[0], pair[0])


def test_changed_pinned_mechanics_rejected_without_modifying_source(pair, monkeypatch):
    digest = matching._digest
    monkeypatch.setattr(
        matching,
        "_digest",
        lambda p: "0" * 64 if str(p).endswith("/learner.py") else digest(p),
    )
    with pytest.raises(ValueError, match="frozen fitting source changed"):
        matching.check_matched_models(*pair)
