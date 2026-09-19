"""Meaningful saved-data/objective mutations, with no benchmark optimizer fits."""

import copy
import json
from dataclasses import replace

import numpy as np
import pytest
from test_affine_anchored_matching import prepared as prepared
from test_affine_anchored_matching import trio as trio

from glassbox import learner
from glassbox.experimental import affine_anchored_model as anchored_model
from glassbox.experimental import initial_channel_balance_matching as matching
from glassbox.experimental import initial_channel_balance_model as balanced
from glassbox.experimental import state_quadratic_model as quadratic_model


@pytest.fixture
def quartet(trio, monkeypatch):
    public, anchored, quadratic = trio
    train = public._train.batch
    old = anchored._model
    sequence = balanced.BalancedQuadraticSequenceModel(
        old.kind, old.dt_s, old.history_steps, old.params, old.norms, old.delay_steps
    )
    report = copy.deepcopy(anchored.report)
    report["recipe"] = balanced.RECIPE
    report["optimization"]["mechanism"] = balanced.EXPERIMENT
    prediction = balanced.initial_training_forecast(
        old.params, old.norms, train, old.delay_steps
    )
    scale = np.asarray(report["optimization"]["error_scale"])
    mse = np.mean(((prediction - train.future_states) / scale) ** 2, (0, 1))
    raw = 1 / np.maximum(mse, 0.0001)
    report["optimization"].update(
        objective=balanced.weighting_metadata(
            old, prediction, mse, raw, raw / raw.mean(), 0.0001, raw.mean()
        ),
        selection_objective=balanced.OBJECTIVE,
        unweighted_development_trace=[
            {key: row[key] for key in ("step", "validation_rollout_mse")}
            for row in report["optimization"]["trace"]
        ],
    )
    candidate = balanced.CandidateDynamics(
        sequence,
        public._train,
        public._development,
        public.contract,
        report,
        anchored.envelope(),
    )
    protocol = matching._protocol()
    protocol["planned_automatic_roles"] = {
        "synthetic": matching.inherited._protocol()["planned_automatic_roles"][
            "synthetic"
        ]
    }
    protocol["imported_calibration"]["excitation_seal_sha256"] = {"synthetic": "a" * 64}
    refs = dict(baseline=public, quadratic=quadratic, anchored=anchored)
    protocol["trusted_comparator_revisions"] = {
        "synthetic": {
            arm: {"revision_fingerprint": model.fingerprint()}
            for arm, model in refs.items()
        }
    }
    protocol["fitting"]["mechanism"]["counts"] = {
        "synthetic": {
            arm: sum(v.size for v in model._model.params.values())
            for arm, model in {**refs, "candidate": candidate}.items()
        }
    }
    monkeypatch.setattr(matching, "_protocol", lambda: copy.deepcopy(protocol))
    return public, candidate, quadratic, anchored


def test_matching_and_archive_replay_without_fit_initializer_solve_or_forecast(
    quartet, monkeypatch, tmp_path
):
    def forbidden(*a, **k):
        pytest.fail("matching attempted fitting, initialization, solve or prediction")

    for module, names in (
        (
            balanced,
            (
                "fit_candidate",
                "fit_candidate_sequence",
                "initialize_candidate",
                "initial_training_forecast",
            ),
        ),
        (
            anchored_model,
            ("fit_candidate", "fit_candidate_sequence", "initialize_candidate"),
        ),
        (quadratic_model, ("fit_candidate_sequence", "initialize_candidate")),
        (learner, ("fit", "initialize_sequence_model")),
        (np.linalg, ("solve", "lstsq")),
    ):
        for name in names:
            monkeypatch.setattr(module, name, forbidden)
    for model in quartet:
        monkeypatch.setattr(type(model._model), "rollout", forbidden)
    before = [m.fingerprint() for m in quartet]
    result = matching.check_matched_models(*quartet)
    assert result["exact_same_data"] and result["exact_same_original_hold_scale"]
    assert result["exact_same_numeric_objective"] is False
    assert result["equal_end_to_end_compute_claimed"] is False
    assert result["objective"]["selected_initial_parameters_checked"]
    assert result["objective"][
        "actual_initial_arrays_and_prediction_verified_separately_by_observer"
    ]
    assert [m.fingerprint() for m in quartet] == before
    models = []
    for index, model in enumerate(quartet):
        path = tmp_path / f"model-{index}.npz"
        model.save(path)
        models.append(type(model).load(path))
    assert matching.check_matched_models(*models) == result
    assert json.loads(json.dumps(result, allow_nan=False)) == result


@pytest.mark.parametrize("role", ["_train", "_development"])
@pytest.mark.parametrize(
    "name", ["past_states", "past_inputs", "future_inputs", "future_states"]
)
def test_each_changed_candidate_cache_array_rejected(quartet, role, name):
    candidate = quartet[1]
    windows = getattr(candidate, role)
    values = getattr(windows.batch, name).copy()
    values[0, 0, 0] += 0.01
    setattr(
        candidate,
        role,
        replace(windows, batch=replace(windows.batch, **{name: values})),
    )
    with pytest.raises(ValueError, match="cache differs"):
        matching.check_matched_models(*quartet)


@pytest.mark.parametrize(
    "name",
    [
        "state_mean",
        "state_scale",
        "input_mean",
        "input_scale",
        "feature_scale",
        "delta_scale",
        "interaction_scale",
        "autonomous_scale",
    ],
)
def test_each_changed_prediction_normalization_rejected(quartet, name):
    candidate = quartet[1]
    values = copy.deepcopy(candidate._model.norms)
    values[name][0] += 0.01
    candidate._model = replace(candidate._model, norms=values)
    with pytest.raises(ValueError, match="normalizations differ"):
        matching.check_matched_models(*quartet)


@pytest.mark.parametrize(
    "change",
    [
        "e0",
        "raw",
        "weight",
        "normalizer",
        "floor",
        "floor_roster",
        "shape",
        "nonfinite",
        "negative",
        "source",
        "fixed",
        "scale_redefined",
        "forecast_count",
        "initial_identity",
        "prediction_identity",
        "extra",
        "selection",
        "old_trace_steps",
        "old_trace_nonfinite",
    ],
)
def test_objective_mutations_are_rejected(quartet, change):
    optimization = quartet[1]._report["optimization"]
    objective = optimization["objective"]
    if change in ("e0", "raw", "weight"):
        name = {
            "e0": "initial_channel_mse",
            "raw": "raw_channel_weights",
            "weight": "channel_weights",
        }[change]
        objective[name][0] += 0.25
    elif change == "normalizer":
        objective["weight_normalizer"] += 1
    elif change == "floor":
        objective["weight_floor"] = 0.001
    elif change == "floor_roster":
        objective["floor_active_channels"] = [-1]
    elif change == "shape":
        objective["initial_channel_mse"] = [[0.5, 0.5]]
    elif change == "nonfinite":
        objective["initial_channel_mse"][0] = float("nan")
    elif change == "negative":
        objective["initial_channel_mse"][0] = -1
    elif change == "source":
        objective["weighting_data"] = "development"
    elif change == "fixed":
        objective["fixed_during_training_and_selection"] = False
    elif change == "scale_redefined":
        objective["original_hold_scale_preserved"] = False
    elif change == "forecast_count":
        objective["additional_training_weight_forecasts"] = 2
    elif change == "initial_identity":
        objective["initial_parameters_and_norms_fingerprint"] = "a" * 64
    elif change == "prediction_identity":
        objective["initial_training_prediction_fingerprint"] = "bad"
    elif change == "extra":
        objective["adaptive_weighting"] = True
    elif change == "selection":
        optimization["selection_objective"] = "unweighted"
    elif change == "old_trace_steps":
        optimization["unweighted_development_trace"].pop()
    else:
        optimization["unweighted_development_trace"][0]["validation_rollout_mse"] = (
            float("inf")
        )
    with pytest.raises(ValueError):
        matching.check_matched_models(*quartet)


@pytest.mark.parametrize(
    "change",
    [
        "contract",
        "hold_scale",
        "recipe",
        "steps",
        "selected_step",
        "matching",
        "parameter_shape",
        "timing",
        "coverage",
        "update",
    ],
)
def test_candidate_contract_budget_or_selection_mutation_rejected(quartet, change):
    candidate = quartet[1]
    if change == "contract":
        candidate._contract["state_channels"][0] = "changed"
    elif change == "hold_scale":
        candidate._report["optimization"]["error_scale"][0][0] += 0.01
    elif change == "recipe":
        candidate._report["recipe"]["objective"] = "other"
    elif change == "steps":
        candidate._report["optimization"]["steps"] = 999
    elif change == "selected_step":
        candidate._report["optimization"]["selected_step"] = 100
    elif change == "matching":
        candidate._report["matching"]["baseline_fingerprint"] = "a" * 64
    elif change == "parameter_shape":
        values = copy.deepcopy(candidate._model.params)
        values["autonomous"] = values["autonomous"][:-1]
        object.__setattr__(candidate._model, "params", values)
    elif change == "timing":
        candidate._report["delay_steps"] += 1
    elif change == "coverage":
        next(iter(candidate._report["training"].values()))["windows"] -= 1
    else:
        candidate._report["previous_revision"] = "updated"
    with pytest.raises(ValueError):
        matching.check_matched_models(*quartet)


@pytest.mark.parametrize("field", ["roles", "seal", "trusted", "counts"])
def test_frozen_reference_contracts_reject_mutations(quartet, monkeypatch, field):
    p = matching._protocol()
    if field == "roles":
        p["planned_automatic_roles"]["synthetic"]["training"][0] = "other"
    elif field == "seal":
        p["imported_calibration"]["excitation_seal_sha256"]["synthetic"] = "b" * 64
    elif field == "trusted":
        p["trusted_comparator_revisions"]["synthetic"]["anchored"][
            "revision_fingerprint"
        ] = "b" * 64
    else:
        p["fitting"]["mechanism"]["counts"]["synthetic"]["candidate"] += 1
    monkeypatch.setattr(matching, "_protocol", lambda: p)
    with pytest.raises(ValueError):
        matching.check_matched_models(*quartet)


def test_new_protocol_and_inherited_source_pins_are_checked(monkeypatch):
    original = matching._digest
    monkeypatch.setattr(
        matching,
        "_digest",
        lambda path: "changed" if path == matching.PROTOCOL else original(path),
    )
    with pytest.raises(ValueError, match="protocol changed"):
        matching._protocol()


def test_changed_initializer_source_is_rejected(quartet, monkeypatch):
    original = matching._digest
    target = matching.ROOT / "src/glassbox/experimental/affine_anchored_model.py"
    monkeypatch.setattr(
        matching,
        "_digest",
        lambda path: "changed" if path == target else original(path),
    )
    with pytest.raises(ValueError, match="source changed"):
        matching.check_matched_models(*quartet)
