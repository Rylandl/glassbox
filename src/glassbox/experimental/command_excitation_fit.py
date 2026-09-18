"""Fit the frozen quadratic recipe directly on independently excited recordings.

Only collection contents change. Preparation uses the public recording split and
window selection, and the existing quadratic optimizer and saved-model wrapper.
No public model is fitted to obtain caches, and no excitation labels are consumed.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from .._learner_arrays import array_fingerprint
from .._sequence_model import _features
from ..learner import (
    ENVELOPE_COVERAGE,
    _calibrate,
    _contract,
    _extract,
    _measure,
    _priority,
    _recording_content,
    steps_for,
)
from ..learner import RECIPE as BASE_RECIPE
from . import state_quadratic_model as quadratic
from .state_input_model import _window_fingerprint

EXPERIMENT = "independent-command-excitation-v1"
FITTING_RECIPE = copy.deepcopy(BASE_RECIPE)
ROLE_COUNTS = {"training": 72, "development": 24}
_ARRAYS = ("past_states", "past_inputs", "future_inputs", "future_states")
_SHARED_PARAMS = ("w1", "b1", "w2", "memory", "memory_bias")


class ExcitationDataError(ValueError):
    """The frozen collection cannot supply all required parents or windows."""


@dataclass(frozen=True)
class _Preparation:
    train: object
    development: object
    contract: dict
    norms: dict
    delay: int
    error_scale: np.ndarray
    provenance: dict


def _equal_windows(actual, expected, role):
    if _window_fingerprint(actual) != _window_fingerprint(expected):
        raise ValueError(f"{role} window keys, origins, timing or arrays changed")


def _training_moments(batch):
    """The frozen initializer's moments and random draws, without any ridge solve."""
    recipe = FITTING_RECIPE
    context = batch.past_inputs.shape[1]
    delay = steps_for(batch.dt_s)["delay"]
    physical = np.concatenate((batch.past_states, batch.future_states), axis=1)
    commands = np.concatenate((batch.past_inputs, batch.future_inputs), axis=1)
    current = physical[:, context:-1]
    xm, xs = current.mean((0, 1)), current.std((0, 1))
    um, us = batch.future_inputs.mean((0, 1)), batch.future_inputs.std((0, 1))
    xs, us = np.where(xs > 1e-8, xs, 1), np.where(us > 1e-8, us, 1)
    x, u = (physical - xm) / xs, (commands - um) / us
    delta = (batch.future_states - current) / xs
    ds = np.maximum(delta.std((0, 1)), 1e-4)
    hidden = np.zeros((len(current), recipe["memory"]))
    features = np.stack(
        [
            _features(
                x[:, context + t],
                u[:, context + t],
                x[:, context + t - delay : context + t],
                u[:, context + t - delay : context + t],
                hidden,
                xp=np,
            )
            for t in range(current.shape[1])
        ],
        axis=1,
    ).reshape(
        -1, (delay + 1) * (current.shape[-1] + commands.shape[-1]) + recipe["memory"]
    )
    fs = features.std(axis=0)
    norms = dict(
        state_mean=xm,
        state_scale=xs,
        input_mean=um,
        input_scale=us,
        feature_scale=np.where(fs > 1e-8, fs, 1.0),
        delta_scale=ds,
    )
    products = {
        "interaction": quadratic.interaction_features(
            x[:, context:-1], u[:, context:], xp=np
        ),
        "autonomous": quadratic.autonomous_features(x[:, context:-1], xp=np),
    }
    for name, values in products.items():
        scale = values.reshape(len(features), -1).std(axis=0)
        norms[name + "_scale"] = np.where(scale > 1e-8, scale, 1.0)
    f, d = features.shape[-1], current.shape[-1]
    rng = np.random.default_rng(recipe["seed"])
    shared = dict(
        w1=rng.normal(size=(f, recipe["width"])) / np.sqrt(f),
        b1=np.zeros(recipe["width"]),
        w2=np.zeros((recipe["width"], d)),
        memory=rng.normal(size=(f, recipe["memory"])) / np.sqrt(f),
        memory_bias=np.zeros(recipe["memory"]),
    )
    parameter_count = (
        f * d
        + d
        + sum(value.size for value in shared.values())
        + d * commands.shape[-1] * d
        + (d * (d + 1) // 2) * d
    )
    return norms, shared, delay, parameter_count


def _prepare(collection, original_model, *, roles, data_seal):
    if not isinstance(original_model, quadratic.CandidateDynamics):
        raise TypeError("original_model must be the frozen quadratic candidate")
    if original_model.recipe != quadratic.RECIPE:
        raise ValueError("original quadratic recipe differs")
    if (
        not isinstance(data_seal, str)
        or len(data_seal) != 64
        or any(c not in "0123456789abcdef" for c in data_seal)
    ):
        raise ValueError("data_seal must be the SHA256 of the sealed excited data")
    contract = _contract(collection)
    if contract != original_model.contract:
        raise ValueError("excited collection signal/timing contract changed")
    if set(roles) != set(ROLE_COUNTS) or any(
        len(roles[role]) != count or len(set(roles[role])) != count
        for role, count in ROLE_COUNTS.items()
    ):
        raise ValueError("frozen training/development role counts differ")
    if set(roles["training"]) & set(roles["development"]):
        raise ValueError("training and development roles overlap")
    seen = _recording_content(collection)
    required = set(roles["training"]) | set(roles["development"])
    if set(seen) - required:
        raise ValueError("collection contains recordings outside frozen roles")
    if set(roles["development"]) - set(seen):
        raise ValueError("collection is missing frozen development parents")
    if required - set(seen):
        raise ExcitationDataError("collection is missing frozen training parents")
    names = sorted(seen, key=_priority)
    count = min(len(names) - 1, max(1, int(np.ceil(len(names) / 4))))
    automatic = {"development": names[:count], "training": names[count:]}
    if automatic != roles:
        raise ValueError("public automatic roles differ from frozen original roles")
    for role, attribute in (("training", "_train"), ("development", "_development")):
        if {k.recording_id for k in getattr(original_model, attribute).keys} != set(
            roles[role]
        ):
            raise ValueError(f"original quadratic {role} parents differ")
    windows = {}
    for role, budget_key in (
        ("training", "training_windows"),
        ("development", "development_windows"),
    ):
        try:
            selected = _extract(collection, roles[role], FITTING_RECIPE[budget_key])
        except ValueError as error:
            failure = ExcitationDataError if role == "training" else ValueError
            raise failure(f"{role} window extraction failed: {error}") from error
        if len(selected.keys) != FITTING_RECIPE[budget_key] or {
            k.recording_id for k in selected.keys
        } != set(roles[role]):
            failure = ExcitationDataError if role == "training" else ValueError
            raise failure(f"{role} lacks the frozen window/parent budget")
        windows[role] = selected
    train, development = windows["training"], windows["development"]
    _equal_windows(development, original_model._development, "development")
    b, recipe = train.batch, FITTING_RECIPE
    delay = steps_for(b.dt_s)["delay"]
    ridge = recipe["ridge_fraction"] * len(b.past_states) * b.future_states.shape[1]
    norms, shared, delay, parameter_count = _training_moments(b)
    hold = np.repeat(b.past_states[:, -1:], b.future_states.shape[1], 1)
    error_scale = np.maximum(
        np.sqrt(np.mean((hold - b.future_states) ** 2, 0)),
        recipe["hold_scale_floor"] * norms["state_scale"],
    )
    original_scale = np.asarray(original_model.report["optimization"]["error_scale"])
    original_windows = _window_fingerprint(original_model._train)
    current_windows = _window_fingerprint(train)
    old_origins = set(
        zip(original_model._train.keys, original_model._train.source_origins)
    )
    new_origins = set(zip(train.keys, train.source_origins))
    if parameter_count != sum(v.size for v in original_model._model.params.values()):
        raise ValueError("quadratic parameter count differs from original model")
    provenance = dict(
        collection_intervention=EXPERIMENT,
        data_seal=data_seal,
        unchanged_development_reference=original_model.fingerprint(),
        recording_content_fingerprints=seen,
        roles=copy.deepcopy(roles),
        training_windows_fingerprint=current_windows,
        development_windows_fingerprint=_window_fingerprint(development),
        original_training_windows_fingerprint=original_windows,
        training_window_origin_changes=dict(
            retained=len(old_origins & new_origins),
            removed=len(old_origins - new_origins),
            added=len(new_origins - old_origins),
        ),
        training_norms_fingerprint=array_fingerprint({}, norms),
        original_training_norms_fingerprint=array_fingerprint(
            {}, original_model._model.norms
        ),
        error_scale_fingerprint=array_fingerprint({}, {"error_scale": error_scale}),
        original_error_scale=original_scale.tolist(),
        original_error_scale_fingerprint=array_fingerprint(
            {}, {"error_scale": original_scale}
        ),
        shared_initialization_fingerprint=array_fingerprint(
            {}, {name: shared[name] for name in _SHARED_PARAMS}
        ),
        fitting_recipe=copy.deepcopy(recipe),
        ridge=ridge,
        minibatch_seed=recipe["seed"] + 10000,
        optimizer_steps=recipe["steps"],
        optimizer_fit_calls=1,
        batch_size=recipe["batch_size"],
        parameter_count=parameter_count,
        same_architecture=True,
        same_development=True,
        same_training_data=current_windows == original_windows,
        same_optimizer_budget=True,
        equal_end_to_end_compute_claimed=False,
    )
    return _Preparation(
        train, development, contract, norms, delay, error_scale, provenance
    )


def fit_excited(collection, original_model, *, roles, data_seal):
    """Fit one unchanged quadratic optimizer trajectory on the excited collection."""
    prepared = _prepare(collection, original_model, roles=roles, data_seal=data_seal)
    train, development = prepared.train, prepared.development
    b, recipe = train.batch, FITTING_RECIPE
    model, optimization = quadratic.fit_candidate_sequence(
        b,
        development.batch,
        width=recipe["width"],
        memory=recipe["memory"],
        ridge=prepared.provenance["ridge"],
        seed=recipe["seed"],
        steps=recipe["steps"],
        batch_size=recipe["batch_size"],
        learning_rate=recipe["learning_rate"],
        check_every=recipe["check_every"],
        error_scale=prepared.error_scale,
        delay_steps=prepared.delay,
    )
    prediction = np.asarray(
        model.rollout(
            development.batch.past_states,
            development.batch.past_inputs,
            development.batch.future_inputs,
        )
    )
    if not np.isfinite(prediction).all():
        raise quadratic.CandidateFitError(
            "nonfinite development prediction cannot calibrate envelope"
        )
    envelope, count, rank = _calibrate(model, development)
    report = dict(
        recipe=copy.deepcopy(quadratic.RECIPE),
        history_steps=model.history_steps,
        horizon_steps=b.future_states.shape[1],
        delay_steps=model.delay_steps,
        training=train.coverage(),
        development=development.coverage(),
        optimization=optimization,
        development_errors=_measure(model, development),
        provenance=prepared.provenance,
        envelope=dict(
            nominal_coverage=ENVELOPE_COVERAGE,
            method="split_conformal_absolute_error",
            calibrated_on="development",
            calibration_windows=count,
            quantile_rank=rank,
            units="physical, per horizon step and per channel, aligned with predict",
            half_width=envelope.tolist(),
        ),
        evidence_limits=[
            "isolated data-intervention research fit; no public recipe, update, controller or envelope qualification",
            "development windows select the checkpoint and calibrate the envelope",
            "only observed states and issued commands enter fitting; excitation labels and paired branches do not",
            "changed training support and normalization also change development checkpoint loss weights",
            "matched architecture and optimizer budgets do not imply equal end-to-end compute",
            "Euclidean forecasts do not enforce manifold constraints",
        ],
    )
    return quadratic.CandidateDynamics(
        model, train, development, prepared.contract, report, envelope
    )


def check_preparation(model, collection, original_model, *, roles, data_seal):
    """Recompute fit-input provenance and normalization without optimizer updates."""
    prepared = _prepare(collection, original_model, roles=roles, data_seal=data_seal)
    if not isinstance(model, quadratic.CandidateDynamics):
        raise TypeError("model must be the frozen quadratic candidate")
    if model.contract != prepared.contract:
        raise ValueError("saved excited model contract differs")
    _equal_windows(model._train, prepared.train, "saved training")
    _equal_windows(model._development, prepared.development, "saved development")
    if array_fingerprint({}, model._model.norms) != array_fingerprint(
        {}, prepared.norms
    ):
        raise ValueError("saved excited model training normalization differs")
    report = model.report
    if report.get("provenance") != prepared.provenance or "matching" in report:
        raise ValueError("saved excited model preparation provenance differs")
    if not np.array_equal(
        np.asarray(report["optimization"]["error_scale"]), prepared.error_scale
    ):
        raise ValueError("saved excited model loss normalization differs")
    for role, windows in (
        ("training", prepared.train),
        ("development", prepared.development),
    ):
        if report[role] != windows.coverage():
            raise ValueError(f"saved excited {role} coverage differs")
    if (
        report["history_steps"] != prepared.train.batch.past_inputs.shape[1]
        or report["delay_steps"] != prepared.delay
        or report["horizon_steps"] != prepared.train.batch.future_states.shape[1]
        or sum(v.size for v in model._model.params.values())
        != prepared.provenance["parameter_count"]
    ):
        raise ValueError("saved excited model architecture or timing differs")
    return copy.deepcopy(prepared.provenance)
