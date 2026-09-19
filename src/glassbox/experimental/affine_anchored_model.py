"""Full quadratic dynamics with an affine-centered initialization ridge prior.

Only the joint solve RHS changes. The existing affine initializer supplies the
prior; no affine fit or diagnostic solve is added. Historical modules keep their
own globals, recipes and formats. Witnesses are captured externally by observers.
"""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import numpy as np

from .._learner_arrays import array_fingerprint, load_arrays
from .._sequence_model import KIND, _features, initialize_sequence_model
from ..learner import (
    ENVELOPE_COVERAGE,
    LearnedDynamics,
    _calibrate,
    _measure,
    steps_for,
)
from ..learner import RECIPE as BASE_RECIPE
from . import state_quadratic_model as quadratic
from .command_excitation_fit import _training_moments
from .state_input_model import _window_fingerprint

_spec = importlib.util.spec_from_file_location(
    f"{__package__}._affine_anchored_shared",
    Path(__file__).with_name("state_input_model.py"),
)
_shared = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_shared)

EXPERIMENT = "affine-anchored-quadratic-v1"
RECIPE = {
    **quadratic.RECIPE,
    "id": EXPERIMENT,
    "ridge_prior": "existing_affine_initializer",
}
FITTING_RECIPE = copy.deepcopy(BASE_RECIPE)
_MODEL_FORMAT = "glassbox-affine-anchored-quadratic-sequence-v1"
_FORMAT = "glassbox-affine-anchored-quadratic-candidate-v1"
CandidateFitError = _shared.CandidateFitError
interaction_features = quadratic.interaction_features
autonomous_features = quadratic.autonomous_features
_rollout = quadratic._rollout


class AnchoredQuadraticSequenceModel(quadratic.QuadraticSequenceModel):
    """The unchanged quadratic predictor with an explicit initializer identity."""

    def metadata(self):
        return {**super().metadata(), "format": _MODEL_FORMAT}

    @classmethod
    def load(cls, path):
        metadata, arrays = load_arrays(path)
        if metadata.pop("format") != _MODEL_FORMAT:
            raise ValueError("unsupported affine-anchored quadratic sequence format")
        return cls(
            **metadata,
            params={k[6:]: v for k, v in arrays.items() if k.startswith("param_")},
            norms={k[5:]: v for k, v in arrays.items() if k.startswith("norm_")},
        )


def initialize_candidate(
    batch, *, seed=0, width=32, memory=8, ridge=1.0, delay_steps=None
):
    """One existing affine precursor, then the same full joint solve centered on it."""
    baseline = initialize_sequence_model(
        batch,
        seed=seed,
        width=width,
        memory=memory,
        ridge=ridge,
        delay_steps=delay_steps,
    )
    params = {k: np.array(v, copy=True) for k, v in baseline.params.items()}
    norms = {k: np.array(v, copy=True) for k, v in baseline.norms.items()}
    context, p = baseline.history_steps, baseline.delay_steps
    physical = np.concatenate((batch.past_states, batch.future_states), axis=1)
    x = (physical - norms["state_mean"]) / norms["state_scale"]
    u = np.concatenate((batch.past_inputs, batch.future_inputs), axis=1)
    u = (u - norms["input_mean"]) / norms["input_scale"]
    current, commands = x[:, context:-1], u[:, context:]
    hidden = np.zeros((len(current), memory))
    features = np.stack(
        [
            _features(
                x[:, context + t],
                u[:, context + t],
                x[:, context + t - p : context + t],
                u[:, context + t - p : context + t],
                hidden,
                xp=np,
            )
            for t in range(current.shape[1])
        ],
        axis=1,
    ).reshape(-1, len(norms["feature_scale"]))
    products = interaction_features(current, commands, xp=np).reshape(len(features), -1)
    autonomous = autonomous_features(current, xp=np).reshape(len(features), -1)
    for name, values in (("interaction", products), ("autonomous", autonomous)):
        scale = values.std(axis=0)
        norms[f"{name}_scale"] = np.where(scale > 1e-8, scale, 1.0)
    design = np.column_stack(
        (
            features / norms["feature_scale"],
            products / norms["interaction_scale"],
            autonomous / norms["autonomous_scale"],
            np.ones(len(features)),
        )
    )
    penalty = np.diag(np.r_[np.full(design.shape[1] - 1, ridge), 0.0])
    # Preserve the frozen physical subtraction and both normalization operations.
    delta = (batch.future_states - physical[:, context:-1]) / norms["state_scale"]
    target = (delta / norms["delta_scale"]).reshape(len(features), -1)
    base_end = features.shape[-1]
    interaction_end = base_end + products.shape[-1]
    system = design.T @ design + penalty
    zero_rhs = design.T @ target
    anchor = np.vstack(
        (
            baseline.params["linear"],
            np.zeros((products.shape[-1] + autonomous.shape[-1], target.shape[-1])),
            baseline.params["bias"][None, :],
        )
    )
    anchored_rhs = zero_rhs + penalty @ anchor
    # These named actual solve inputs are the prospective read-only witness boundary.
    coefficients = np.linalg.solve(system, anchored_rhs)
    params.update(
        linear=coefficients[:base_end],
        interaction=coefficients[base_end:interaction_end],
        autonomous=coefficients[interaction_end:-1],
        bias=coefficients[-1],
    )
    return AnchoredQuadraticSequenceModel(KIND, batch.dt_s, context, params, norms, p)


class CandidateDynamics(_shared.CandidateDynamics):
    """Saved research revision; no public selection or update API."""


# Reuse the exact optimizer, prediction wrapper and archive code with private globals.
_shared.EXPERIMENT = EXPERIMENT
_shared.RECIPE = RECIPE
_shared._MODEL_FORMAT = _MODEL_FORMAT
_shared._FORMAT = _FORMAT
_shared._rollout = _rollout
_shared.BilinearSequenceModel = AnchoredQuadraticSequenceModel
_shared.initialize_candidate = initialize_candidate
_shared.CandidateDynamics = CandidateDynamics
fit_candidate_sequence = _shared.fit_candidate_sequence


def fit_candidate(baseline):
    """Fit on public cached windows without solving a preparation-only initializer."""
    if type(baseline) is not LearnedDynamics or baseline.recipe != BASE_RECIPE:
        raise TypeError("anchored fit requires the adopted LearnedDynamics baseline")
    if baseline.report.get("previous_revision") is not None:
        raise ValueError("anchored fit requires an initial public revision")
    train, development = baseline._train, baseline._development
    b, recipe = train.batch, FITTING_RECIPE
    delay = steps_for(b.dt_s)["delay"]
    ridge = recipe["ridge_fraction"] * len(b.past_states) * b.future_states.shape[1]
    norms, draws, _, _ = _training_moments(b)
    base_norms = {
        key: norms[key]
        for key in (
            "state_mean",
            "state_scale",
            "input_mean",
            "input_scale",
            "feature_scale",
            "delta_scale",
        )
    }
    if array_fingerprint({}, baseline._model.norms) != array_fingerprint(
        {}, base_norms
    ):
        raise ValueError(
            "baseline normalization does not match its cached training data"
        )
    hold = np.repeat(b.past_states[:, -1:], b.future_states.shape[1], 1)
    error_scale = np.maximum(
        np.sqrt(np.mean((hold - b.future_states) ** 2, 0)),
        recipe["hold_scale_floor"] * norms["state_scale"],
    )
    if not np.array_equal(
        error_scale, np.asarray(baseline.report["optimization"]["error_scale"])
    ):
        raise ValueError("baseline loss scale does not match its cached training data")
    model, optimization = fit_candidate_sequence(
        b,
        development.batch,
        width=recipe["width"],
        memory=recipe["memory"],
        ridge=ridge,
        seed=recipe["seed"],
        steps=recipe["steps"],
        batch_size=recipe["batch_size"],
        learning_rate=recipe["learning_rate"],
        check_every=recipe["check_every"],
        error_scale=error_scale,
        delay_steps=delay,
    )
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
        horizon_steps=b.future_states.shape[1],
        delay_steps=delay,
        training=train.coverage(),
        development=development.coverage(),
        optimization=optimization,
        development_errors=_measure(model, development),
        matching=dict(
            baseline_fingerprint=baseline.fingerprint(),
            training_windows_fingerprint=_window_fingerprint(train),
            development_windows_fingerprint=_window_fingerprint(development),
            baseline_norms_fingerprint=array_fingerprint({}, base_norms),
            error_scale_fingerprint=array_fingerprint({}, {"error_scale": error_scale}),
            shared_initialization_fingerprint=array_fingerprint({}, draws),
            minibatch_seed=recipe["seed"] + 10000,
            optimizer_steps=recipe["steps"],
            batch_size=recipe["batch_size"],
            baseline_parameter_count=sum(
                v.size for v in baseline._model.params.values()
            ),
            candidate_parameter_count=sum(v.size for v in model.params.values()),
            equal_compute=False,
        ),
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
            "isolated research candidate; no public recipe, update, controller or envelope qualification",
            "development windows select the checkpoint and calibrate the envelope",
            "affine-centered ridge applies only to initialization, not gradient training or prediction",
            "matched data and optimizer updates do not imply equal FLOPs",
            "Euclidean forecasts do not enforce manifold constraints",
        ],
    )
    return CandidateDynamics(
        model, train, development, baseline.contract, report, envelope
    )
