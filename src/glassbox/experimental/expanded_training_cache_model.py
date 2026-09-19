"""One expanded cache, with the pinned full-cache numerical fitter unchanged."""

from __future__ import annotations

import copy
import importlib.util
import time
from collections import Counter
from dataclasses import dataclass, replace

import numpy as np

from .._learner_arrays import array_fingerprint, load_arrays, save_arrays
from .._sequence_model import _features
from ..learner import (
    ENVELOPE_COVERAGE,
    LearnedDynamics,
    _calibrate,
    _contract,
    _extract,
    _measure,
    _priority,
    _recording_content,
    _subset,
    steps_for,
)
from ..learner import RECIPE as PUBLIC_RECIPE
from . import full_cache_gradient_model as historical
from .affine_anchored_model import autonomous_features, interaction_features
from .state_input_model import _window_fingerprint, _window_metadata

_spec = importlib.util.spec_from_file_location(
    f"{__package__}._expanded_training_cache_engine", historical.__file__
)
_engine = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_engine)

EXPERIMENT = "expanded-training-cache-v1"
RECIPE = {
    **historical.RECIPE,
    "id": EXPERIMENT,
    "training_windows": 1536,
    "batch_size": 1536,
}
FITTING_RECIPE = {
    **historical.FITTING_RECIPE,
    "training_windows": 1536,
    "batch_size": 1536,
}
ROLE_COUNTS = {"training": 72, "development": 24}
REFERENCE_TRAINING_WINDOWS = 384
DATA_SEAL_KEYS = {
    "common_data_seal_sha256",
    "excitation_seal_sha256",
    "excitation_reuse_sha256",
    "reference_reuse_sha256",
}
_FORMAT = "glassbox-expanded-training-cache-candidate-v1"
_MODEL_FORMAT = "glassbox-expanded-training-cache-sequence-v1"
_PREPARATION_FORMAT = "glassbox-expanded-training-cache-preparation-v1"
_ARRAYS = historical._ARRAYS
_EXCITATION = ("past_excitation", "future_excitation")

# These globals belong only to the freshly loaded engine and its private archive.
_engine.EXPERIMENT, _engine.RECIPE = EXPERIMENT, RECIPE
_engine.FITTING_RECIPE = FITTING_RECIPE
_engine._MODEL_FORMAT, _engine._FORMAT = _MODEL_FORMAT, _FORMAT
_engine._wrapper.RECIPE = _engine._wrapper._shared.RECIPE = RECIPE
_engine._wrapper.FITTING_RECIPE = FITTING_RECIPE
_engine._wrapper._shared._FORMAT = _FORMAT
_engine._wrapper._shared._MODEL_FORMAT = _MODEL_FORMAT

BalancedQuadraticSequenceModel = _engine.BalancedQuadraticSequenceModel
CandidateFitError = _engine.CandidateFitError
fit_candidate_sequence = _engine.fit_candidate_sequence
initialize_candidate = _engine.initialize_candidate
initial_training_forecast = _engine.initial_training_forecast
weighting_metadata = _engine.weighting_metadata
make_training_objective = _engine.make_training_objective
trial_parameters = _engine.trial_parameters
safeguard_metadata = _engine.safeguard_metadata
gradient_metadata = _engine.gradient_metadata
_observe_attempt = _engine._observe_attempt
_rollout = _engine._rollout
OBJECTIVE = _engine.OBJECTIVE
SCALES = _engine.SCALES
FIT_WALL_TIME_LIMIT_S = _engine.FIT_WALL_TIME_LIMIT_S


class PreparationUnavailable(ValueError):
    """Authenticated recordings cannot supply the frozen expanded window count."""

    def __init__(self, diagnostics):
        self.diagnostics = copy.deepcopy(diagnostics)
        super().__init__("insufficient legal training windows for expanded cache")


@dataclass(frozen=True)
class PreparedFit:
    train: object
    development: object
    contract: dict
    norms: dict
    error_scale: np.ndarray
    delay: int
    ridge: float
    provenance: dict


def _excitation_fingerprint(windows):
    return array_fingerprint(
        {"declared": windows.excitation_declared},
        {key: getattr(windows, key) for key in _EXCITATION}
        if windows.excitation_declared
        else {},
    )


def _equal_windows(actual, expected, label):
    if _window_fingerprint(actual) != _window_fingerprint(
        expected
    ) or _excitation_fingerprint(actual) != _excitation_fingerprint(expected):
        raise ValueError(f"{label} keys, origins, arrays or excitation differ")


def training_norms(batch):
    """All eight exact initializer moments, with no draws, forecasts or solves."""
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
    memory = FITTING_RECIPE["memory"]
    hidden = np.zeros((len(current), memory))
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
    ).reshape(-1, (delay + 1) * (current.shape[-1] + commands.shape[-1]) + memory)
    fs = features.std(axis=0)
    norms = dict(
        state_mean=xm,
        state_scale=xs,
        input_mean=um,
        input_scale=us,
        feature_scale=np.where(fs > 1e-8, fs, 1.0),
        delta_scale=ds,
    )
    for name, values in (
        ("interaction", interaction_features(x[:, context:-1], u[:, context:], xp=np)),
        ("autonomous", autonomous_features(x[:, context:-1], xp=np)),
    ):
        scale = values.reshape(len(features), -1).std(axis=0)
        norms[name + "_scale"] = np.where(scale > 1e-8, scale, 1.0)
    return norms


def _hold_scale(batch, norms):
    hold = np.repeat(batch.past_states[:, -1:], batch.future_states.shape[1], axis=1)
    return np.maximum(
        np.sqrt(np.mean((hold - batch.future_states) ** 2, axis=0)),
        FITTING_RECIPE["hold_scale_floor"] * norms["state_scale"],
    )


def _support(collection, roles, steps, train=None):
    keys = [
        key
        for key in collection.window_keys(
            history_steps=steps["history"], horizon_steps=steps["horizon"]
        )
        if key.recording_id in roles["training"]
    ]
    eligible = Counter(key.recording_id for key in keys)
    result = dict(
        available_training_windows=len(keys),
        required_training_windows=FITTING_RECIPE["training_windows"],
        per_parent_legal_origins={name: eligible[name] for name in roles["training"]},
        represented_training_parents=len(eligible),
    )
    if train is not None:
        targets = set()
        for key, origin in zip(train.keys, train.source_origins, strict=True):
            targets.update(
                (key.recording_id, key.segment_id, origin + offset)
                for offset in range(1, steps["horizon"] + 1)
            )
        counts = Counter(key.recording_id for key in train.keys)
        appearances = len(train.keys) * steps["horizon"]
        result.update(
            selected_training_windows=len(train.keys),
            per_parent_selected_origins=dict(counts),
            unique_forecast_target_transitions=len(targets),
            target_transition_appearances=appearances,
            repeated_target_transition_appearances=appearances - len(targets),
        )
    return result


def prepare(collection, public_reference, *, roles, data_seal):
    """Reconstruct one cache from the authenticated collection; never initialize."""
    if (
        type(public_reference) is not LearnedDynamics
        or public_reference.recipe != PUBLIC_RECIPE
    ):
        raise TypeError("preparation requires the adopted public reference")
    if public_reference.report.get("previous_revision") is not None:
        raise ValueError("preparation requires an initial public revision")
    if (
        not isinstance(data_seal, dict)
        or set(data_seal) != DATA_SEAL_KEYS
        or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)
            for value in data_seal.values()
        )
    ):
        raise ValueError("preparation requires the four authenticated SHA256 anchors")
    if (
        set(roles) != set(ROLE_COUNTS)
        or any(
            len(roles[role]) != count or len(set(roles[role])) != count
            for role, count in ROLE_COUNTS.items()
        )
        or set(roles["training"]) & set(roles["development"])
    ):
        raise ValueError("frozen recording roles differ")
    contract, seen = _contract(collection), _recording_content(collection)
    if contract != public_reference.contract:
        raise ValueError("source signal/timing contract differs")
    if seen != public_reference._seen or set(seen) != set(roles["training"]) | set(
        roles["development"]
    ):
        raise ValueError("source recording content ledger differs")
    names = sorted(seen, key=_priority)
    count = min(len(names) - 1, max(1, int(np.ceil(len(names) / 4))))
    if roles != {"development": names[:count], "training": names[count:]}:
        raise ValueError("automatic recording roles differ")
    recipe = FITTING_RECIPE
    development = _extract(
        collection, roles["development"], recipe["development_windows"]
    )
    _equal_windows(development, public_reference._development, "development")
    if len(development.keys) != recipe["development_windows"]:
        raise ValueError("development budget differs")
    original = _extract(collection, roles["training"], REFERENCE_TRAINING_WINDOWS)
    _equal_windows(original, public_reference._train, "original training")
    if len(original.keys) != REFERENCE_TRAINING_WINDOWS:
        raise ValueError("original training budget differs")
    steps = steps_for(development.batch.dt_s)
    support = _support(collection, roles, steps)
    if support["available_training_windows"] < recipe["training_windows"]:
        raise PreparationUnavailable(
            dict(
                support,
                data_seal=copy.deepcopy(data_seal),
                roles=copy.deepcopy(roles),
                source_ledger_exact=True,
                development_cache_exact=True,
                reference_training_cache_exact=True,
            )
        )
    train = _extract(collection, roles["training"], recipe["training_windows"])
    if (
        len(train.keys) != recipe["training_windows"]
        or {key.recording_id for key in train.keys} != set(roles["training"])
        or len(set(zip(train.keys, train.source_origins))) != len(train.keys)
    ):
        raise ValueError("expanded selection count, unique origins or parents differ")
    _equal_windows(
        _subset(train, list(range(REFERENCE_TRAINING_WINDOWS))),
        original,
        "training prefix",
    )
    b = train.batch
    norms = training_norms(b)
    error_scale = _hold_scale(b, norms)
    ridge = recipe["ridge_fraction"] * len(train.keys) * b.future_states.shape[1]
    d, u, f = (
        b.future_states.shape[-1],
        b.future_inputs.shape[-1],
        len(norms["feature_scale"]),
    )
    count = (
        f * d
        + d
        + f * recipe["width"]
        + recipe["width"]
        + recipe["width"] * d
        + f * recipe["memory"]
        + recipe["memory"]
        + d * u * d
        + d * (d + 1) // 2 * d
    )
    provenance = dict(
        intervention=EXPERIMENT,
        data_seal=copy.deepcopy(data_seal),
        public_reference_fingerprint=public_reference.fingerprint(),
        recording_content_fingerprints=seen,
        roles=copy.deepcopy(roles),
        training_windows_fingerprint=_window_fingerprint(train),
        development_windows_fingerprint=_window_fingerprint(development),
        reference_training_windows_fingerprint=_window_fingerprint(original),
        training_excitation_fingerprint=_excitation_fingerprint(train),
        development_excitation_fingerprint=_excitation_fingerprint(development),
        training_norms_fingerprint=array_fingerprint({}, norms),
        error_scale_fingerprint=array_fingerprint({}, {"error_scale": error_scale}),
        fitting_recipe=copy.deepcopy(recipe),
        ridge=ridge,
        delay=steps["delay"],
        parameter_count=count,
        training_windows=len(train.keys),
        development_windows=len(development.keys),
        reference_training_windows=len(original.keys),
        added_training_windows=len(train.keys) - len(original.keys),
        support=_support(collection, roles, steps, train),
        same_recordings=True,
        same_development_cache=True,
        training_prefix_exact=True,
        same_training_cache=False,
        same_numeric_objective=False,
        same_all_initial_parameters_required=False,
        equal_compute=False,
    )
    return PreparedFit(
        train,
        development,
        contract,
        norms,
        error_scale,
        steps["delay"],
        ridge,
        provenance,
    )


def _preparation_parts(prepared):
    windows = {"train": prepared.train, "development": prepared.development}
    metadata = dict(
        format=_PREPARATION_FORMAT,
        contract=prepared.contract,
        delay=prepared.delay,
        ridge=prepared.ridge,
        provenance=prepared.provenance,
        windows={
            role: {
                **_window_metadata(value),
                "dt_s": value.batch.dt_s,
                "excitation_declared": value.excitation_declared,
            }
            for role, value in windows.items()
        },
    )
    arrays = {
        "error_scale": prepared.error_scale,
        **{f"norm_{k}": v for k, v in prepared.norms.items()},
    }
    for role, value in windows.items():
        arrays.update({f"{role}_{key}": getattr(value.batch, key) for key in _ARRAYS})
        if value.excitation_declared:
            arrays.update({f"{role}_{key}": getattr(value, key) for key in _EXCITATION})
    return metadata, arrays


def save_preparation(path, prepared):
    """Persist actual caches and preparation facts before invoking the fitter."""
    save_arrays(path, *_preparation_parts(prepared))


def verify_preparation(path, prepared):
    """Verify the saved preparation against fresh authenticated reconstruction."""
    actual_meta, actual_arrays = load_arrays(path)
    metadata, arrays = _preparation_parts(prepared)
    expected = array_fingerprint(metadata, arrays)
    if array_fingerprint(actual_meta, actual_arrays) != expected:
        raise ValueError("saved preparation differs from reconstructed data")
    return expected


class CandidateDynamics(_engine.CandidateDynamics):
    """Same saved mean interface, retaining any declared cache excitation sidecars."""

    def _metadata(self):
        metadata = super()._metadata()
        for role, windows in (
            ("train", self._train),
            ("development", self._development),
        ):
            metadata["windows"][role]["excitation_declared"] = (
                windows.excitation_declared
            )
        return metadata

    def _arrays(self):
        arrays = super()._arrays()
        for role, windows in (
            ("train", self._train),
            ("development", self._development),
        ):
            if windows.excitation_declared:
                arrays.update(
                    {f"{role}_{key}": getattr(windows, key) for key in _EXCITATION}
                )
        return arrays

    @classmethod
    def load(cls, path):
        model = super().load(path)
        metadata, arrays = load_arrays(path)
        for role, attribute in (("train", "_train"), ("development", "_development")):
            declared = metadata["windows"][role].get("excitation_declared")
            if type(declared) is not bool or any(
                (f"{role}_{key}" in arrays) != declared for key in _EXCITATION
            ):
                raise ValueError("saved excitation declaration differs")
            if declared:
                setattr(
                    model,
                    attribute,
                    replace(
                        getattr(model, attribute),
                        **{key: arrays[f"{role}_{key}"] for key in _EXCITATION},
                    ),
                )
        return model


def fit_candidate(prepared):
    """Fit once on previously prepared data; callers retain it after any failure."""
    if not isinstance(prepared, PreparedFit):
        raise TypeError("fit_candidate requires PreparedFit")
    train, development, recipe = prepared.train, prepared.development, FITTING_RECIPE
    if (
        prepared.provenance["fitting_recipe"] != recipe
        or len(train.keys) != recipe["training_windows"]
    ):
        raise ValueError("prepared fitting recipe/budget differs")
    if (
        prepared.ridge
        != recipe["ridge_fraction"]
        * len(train.keys)
        * train.batch.future_states.shape[1]
        or prepared.ridge != prepared.provenance["ridge"]
        or prepared.delay != steps_for(train.batch.dt_s)["delay"]
        or prepared.delay != prepared.provenance["delay"]
    ):
        raise ValueError("prepared ridge or delay differs")
    for key, actual in (
        ("training_windows_fingerprint", _window_fingerprint(train)),
        ("development_windows_fingerprint", _window_fingerprint(development)),
        ("training_norms_fingerprint", array_fingerprint({}, prepared.norms)),
        (
            "error_scale_fingerprint",
            array_fingerprint({}, {"error_scale": prepared.error_scale}),
        ),
    ):
        if prepared.provenance[key] != actual:
            raise ValueError("prepared " + key + " differs")
    fit_started = time.perf_counter()
    model, optimization = fit_candidate_sequence(
        train.batch,
        development.batch,
        width=recipe["width"],
        memory=recipe["memory"],
        ridge=prepared.ridge,
        seed=recipe["seed"],
        steps=recipe["steps"],
        batch_size=recipe["batch_size"],
        learning_rate=recipe["learning_rate"],
        check_every=recipe["check_every"],
        error_scale=prepared.error_scale,
        delay_steps=prepared.delay,
    )
    fit_wall_time = time.perf_counter() - fit_started
    calibration_started = time.perf_counter()
    if array_fingerprint({}, model.norms) != array_fingerprint({}, prepared.norms):
        raise ValueError("actual fitted normalization differs from preparation")
    prediction = np.asarray(
        model.rollout(
            development.batch.past_states,
            development.batch.past_inputs,
            development.batch.future_inputs,
        )
    )
    if not np.isfinite(prediction).all():
        raise CandidateFitError(
            "nonfinite development prediction cannot calibrate envelope"
        )
    envelope, count, rank = _calibrate(model, development)
    report = dict(
        recipe=copy.deepcopy(RECIPE),
        history_steps=model.history_steps,
        horizon_steps=train.batch.future_states.shape[1],
        delay_steps=model.delay_steps,
        training=train.coverage(),
        development=development.coverage(),
        optimization=optimization,
        development_errors=_measure(model, development),
        preparation=prepared.provenance,
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
            "Expanded overlapping windows do not add independent recordings or shifted conditions.",
            "Training-derived normalization, initialization and numerical objective change naturally; formulas remain fixed.",
            "Development windows select the checkpoint and calibrate the envelope; no new uncertainty qualification.",
            "Additional full-cache work is explicit; no equal-compute, public-recipe, update or controller claim.",
        ],
    )
    report["timing"] = dict(
        fit_and_capture_wall_time_s=fit_wall_time,
        calibration_wall_time_s=time.perf_counter() - calibration_started,
    )
    return CandidateDynamics(
        model, train, development, prepared.contract, report, envelope
    )


def check_preparation(model, collection, public_reference, *, roles, data_seal):
    """Reconstruct the complete cache and training reductions without refitting."""
    prepared = prepare(collection, public_reference, roles=roles, data_seal=data_seal)
    if type(model) is not CandidateDynamics or model.contract != prepared.contract:
        raise ValueError("saved candidate class or contract differs")
    _equal_windows(model._train, prepared.train, "saved training")
    _equal_windows(model._development, prepared.development, "saved development")
    report = model.report
    if model.recipe != RECIPE or report.get("recipe") != RECIPE:
        raise ValueError("saved candidate recipe differs")
    if report.get("preparation") != prepared.provenance:
        raise ValueError("saved preparation provenance differs")
    if array_fingerprint({}, model._model.norms) != array_fingerprint(
        {}, prepared.norms
    ):
        raise ValueError("saved training norms differ")
    if not np.array_equal(
        np.asarray(report["optimization"]["error_scale"]), prepared.error_scale
    ):
        raise ValueError("saved error scale differs")
    if any(
        report[role] != windows.coverage()
        for role, windows in (
            ("training", prepared.train),
            ("development", prepared.development),
        )
    ):
        raise ValueError("saved window coverage differs")
    if (
        report["history_steps"] != prepared.train.batch.past_inputs.shape[1]
        or report["horizon_steps"] != prepared.train.batch.future_states.shape[1]
        or report["delay_steps"] != prepared.delay
        or sum(v.size for v in model._model.params.values())
        != prepared.provenance["parameter_count"]
    ):
        raise ValueError("saved architecture or timing differs")
    return copy.deepcopy(prepared.provenance)
