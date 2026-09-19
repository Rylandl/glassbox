"""Bounded full-training loss acceptance around unchanged balanced Adam proposals.

Only proposal acceptance changes. Historical fitters and public options remain
untouched; prediction, initialization and fixed objective arithmetic are reused.
"""

from __future__ import annotations

import copy
import importlib.util
import time

import jax
import jax.numpy as jnp
import numpy as np

from .._learner_arrays import load_arrays
from .._sequence_model import KIND
from ..learner import RECIPE as BASE_RECIPE
from . import affine_anchored_model as anchored
from . import initial_channel_balance_model as balanced

_spec = importlib.util.spec_from_file_location(
    f"{__package__}._safeguarded_adam_wrapper", anchored.__file__
)
_wrapper = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_wrapper)

EXPERIMENT = "safeguarded-adam-v1"
OBJECTIVE = "fixed_initial_training_channel_balance"
RECIPE = {
    **balanced.RECIPE,
    "id": EXPERIMENT,
    "optimizer": "bounded_full_training_adam_acceptance",
}
FITTING_RECIPE = copy.deepcopy(BASE_RECIPE)
_MODEL_FORMAT = "glassbox-safeguarded-adam-sequence-v1"
_FORMAT = "glassbox-safeguarded-adam-candidate-v1"
_ARRAYS = ("past_states", "past_inputs", "future_inputs", "future_states")
CandidateFitError = _wrapper.CandidateFitError
initialize_candidate = anchored.initialize_candidate
_rollout = anchored._rollout


class BalancedQuadraticSequenceModel(anchored.AnchoredQuadraticSequenceModel):
    """The identical full quadratic predictor under a distinct research identity."""

    def metadata(self):
        return {**super().metadata(), "format": _MODEL_FORMAT}

    @classmethod
    def load(cls, path):
        metadata, arrays = load_arrays(path)
        if metadata.pop("format") != _MODEL_FORMAT:
            raise ValueError("unsupported safeguarded Adam sequence format")
        return cls(
            **metadata,
            params={k[6:]: v for k, v in arrays.items() if k.startswith("param_")},
            norms={k[5:]: v for k, v in arrays.items() if k.startswith("norm_")},
        )


initial_training_forecast = balanced.initial_training_forecast
weighting_metadata = balanced.weighting_metadata
SCALES = tuple(2.0**-index for index in range(8))
FIT_WALL_TIME_LIMIT_S = 7200


def _observe_attempt(state):
    """Read-only external observers may copy actual execution state at this call."""


def trial_parameters(current, proposal, alpha):
    """Canonical fit/replay interpolation; alpha one preserves proposal bytes."""
    if alpha == 1.0:
        return proposal
    return jax.tree.map(lambda old, new: old + alpha * (new - old), current, proposal)


def make_training_objective(norms, train, error_scale, weights, delay):
    """The same closed, jitted full-cache scalar evaluator in fitting and replay."""
    data = tuple(jnp.asarray(getattr(train, name)) for name in _ARRAYS)
    norms = jax.tree.map(jnp.asarray, norms)
    normalization = jnp.asarray(error_scale)
    fixed_weights = jax.lax.stop_gradient(jnp.asarray(weights))

    @jax.jit
    def evaluate(params):
        x, up, uf, target = data
        predicted = _rollout(params, norms, x, up, uf, delay)
        return jnp.mean(fixed_weights * ((predicted - target) / normalization) ** 2)

    return evaluate


def safeguard_metadata(
    *,
    steps,
    accepted_scales,
    objective_calls,
    initial_loss,
    final_loss,
    checkpoint_losses,
    selected_step,
):
    """Deterministic report independently rebuildable from actual observations."""
    rejected = sum(scale == 0 for scale in accepted_scales)
    longest = current = 0
    for scale in accepted_scales:
        current = current + 1 if scale == 0 else 0
        longest = max(longest, current)
    return dict(
        id=EXPERIMENT,
        acceptance="first_finite_strict_decrease",
        moment_policy="advance_on_every_finite_proposal",
        proposal_attempts=steps,
        completed_attempts=len(accepted_scales),
        accepted_attempts=len(accepted_scales) - rejected,
        rejected_attempts=rejected,
        scales=list(SCALES),
        accepted_scale_counts=[
            dict(scale=scale, attempts=accepted_scales.count(scale)) for scale in SCALES
        ],
        maximum_rejection_streak=longest,
        gradient_proposal_calls=steps,
        full_training_objective_calls=objective_calls,
        full_training_objective_calls_max=1 + len(SCALES) * steps,
        initial_full_training_loss=initial_loss,
        final_full_training_loss=final_loss,
        selected_full_training_loss=next(
            row["full_training_loss"]
            for row in checkpoint_losses
            if row["step"] == selected_step
        ),
        checkpoint_full_training_losses=checkpoint_losses,
        fit_wall_time_limit_s=FIT_WALL_TIME_LIMIT_S,
    )


def fit_candidate_sequence(
    train,
    validation,
    *,
    seed=0,
    steps=1000,
    batch_size=64,
    learning_rate=0.002,
    width=32,
    memory=8,
    ridge=1.0,
    check_every=100,
    error_scale=None,
    delay_steps=None,
):
    """One initializer, fixed objective, and bounded acceptance of Adam proposals."""
    fit_started_at = time.perf_counter()
    if steps < 0 or check_every < 1:
        raise ValueError("invalid training steps")
    if batch_size < 1 or not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("batch_size and learning_rate must be positive")
    if train.dt_s != validation.dt_s or any(
        getattr(train, k).shape[1:] != getattr(validation, k).shape[1:] for k in _ARRAYS
    ):
        raise ValueError("train and validation contracts differ")
    model = initialize_candidate(
        train,
        seed=seed,
        width=width,
        memory=memory,
        ridge=ridge,
        delay_steps=delay_steps,
    )
    delay = model.delay_steps
    params, norms = jax.tree.map(jnp.asarray, (model.params, model.norms))
    error_scale = np.asarray(error_scale, dtype=float)
    if (
        error_scale.shape != train.future_states.shape[1:]
        or not np.isfinite(error_scale).all()
        or np.any(error_scale <= 0)
    ):
        raise ValueError("error_scale must be positive [horizon,channel] values")
    normalization = jnp.asarray(error_scale)
    training = tuple(jnp.asarray(getattr(train, key)) for key in _ARRAYS)
    development = tuple(jnp.asarray(getattr(validation, key)) for key in _ARRAYS)

    initial_training_prediction = initial_training_forecast(params, norms, train, delay)
    if initial_training_prediction.dtype != np.dtype("float64"):
        raise ValueError(
            "initial channel weighting requires the frozen float64 runtime"
        )
    if not np.isfinite(initial_training_prediction).all():
        raise CandidateFitError("nonfinite initial training forecast")
    weight_floor = BASE_RECIPE["hold_scale_floor"] ** 2
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        initial_channel_mse = np.mean(
            (
                (initial_training_prediction - train.future_states)
                / np.asarray(normalization)
            )
            ** 2,
            axis=(0, 1),
        )
        raw_channel_weights = 1 / np.maximum(initial_channel_mse, weight_floor)
        weight_normalizer = np.mean(raw_channel_weights)
        channel_weights = raw_channel_weights / weight_normalizer
    dimension = train.future_states.shape[-1]
    if (
        initial_channel_mse.shape != (dimension,)
        or not np.isfinite(initial_channel_mse).all()
        or np.any(initial_channel_mse < 0)
        or not np.isfinite(weight_normalizer)
        or weight_normalizer <= 0
        or any(
            values.shape != (dimension,)
            or not np.isfinite(values).all()
            or np.any(values <= 0)
            for values in (raw_channel_weights, channel_weights)
        )
    ):
        raise CandidateFitError("nonfinite initial channel weighting")
    for values in (
        initial_training_prediction,
        initial_channel_mse,
        raw_channel_weights,
        channel_weights,
    ):
        values.setflags(write=False)
    fixed_weights = jax.lax.stop_gradient(jnp.asarray(channel_weights))

    def loss_components(par, data):
        x, up, uf, target = data
        prediction = _rollout(par, norms, x, up, uf, delay)
        squared = ((prediction - target) / normalization) ** 2
        return jnp.mean(fixed_weights * squared), jnp.mean(squared)

    def loss(par, data):
        return loss_components(par, data)[0]

    @jax.jit
    def evaluate(par):
        return loss_components(par, development)

    @jax.jit
    def update(par, first, second, index, indices):
        data = tuple(value[indices] for value in training)
        value, grad = jax.value_and_grad(loss)(par, data)
        gradient_finite = jnp.all(
            jnp.stack([jnp.all(jnp.isfinite(g)) for g in jax.tree.leaves(grad)])
        )
        norm = jnp.sqrt(sum(jnp.sum(g * g) for g in jax.tree.leaves(grad)))
        grad = jax.tree.map(lambda g: g * jnp.minimum(1.0, 5.0 / (norm + 1e-12)), grad)
        first = jax.tree.map(lambda m, g: 0.9 * m + 0.1 * g, first, grad)
        second = jax.tree.map(lambda v, g: 0.999 * v + 0.001 * g * g, second, grad)
        par = jax.tree.map(
            lambda w, m, v: (
                w
                - learning_rate
                * (m / (1 - 0.9**index))
                / (jnp.sqrt(v / (1 - 0.999**index)) + 1e-8)
            ),
            par,
            first,
            second,
        )
        return par, first, second, value, norm, grad, gradient_finite

    best = params
    best_loss, unweighted_best_initial_loss = map(float, evaluate(params))
    if not np.isfinite([best_loss, unweighted_best_initial_loss]).all():
        raise CandidateFitError("initial recursive validation loss is nonfinite")
    trace = [dict(step=0, validation_rollout_mse=best_loss)]
    unweighted_development_trace = [
        dict(step=0, validation_rollout_mse=unweighted_best_initial_loss)
    ]
    first, second = (jax.tree.map(jnp.zeros_like, params) for _ in range(2))
    full_training_objective = make_training_objective(
        norms, train, normalization, fixed_weights, delay
    )
    current_full_training_loss = float(full_training_objective(params))
    objective_calls = 1
    _observe_attempt(
        dict(phase="initial_objective", loss=current_full_training_loss, params=params)
    )
    if not np.isfinite(current_full_training_loss):
        raise CandidateFitError("nonfinite initial full-training acceptance loss")
    initial_full_training_loss = current_full_training_loss
    checkpoint_losses = [dict(step=0, full_training_loss=current_full_training_loss)]
    accepted_scales = []
    rng = np.random.default_rng(seed + 10000)
    best_step = 0
    for i in range(1, steps + 1):
        if time.perf_counter() - fit_started_at > FIT_WALL_TIME_LIMIT_S:
            raise CandidateFitError("fit_time_limit")
        indices = rng.integers(
            len(train.past_states), size=min(batch_size, len(train.past_states))
        )
        _observe_attempt(
            dict(
                phase="started",
                attempt=i,
                indices=indices,
                params=params,
                first=first,
                second=second,
                current_full_training_loss=current_full_training_loss,
            )
        )
        (
            proposal,
            new_first,
            new_second,
            value,
            gradient_norm,
            clipped_gradient,
            gradient_finite,
        ) = update(params, first, second, i, indices)
        minibatch_loss, gradient_norm = float(value), float(gradient_norm)
        gradient_finite = bool(gradient_finite)
        _observe_attempt(
            dict(
                phase="proposed",
                attempt=i,
                proposal=proposal,
                new_first=new_first,
                new_second=new_second,
                minibatch_loss=minibatch_loss,
                gradient_norm=gradient_norm,
                clipped_gradient=clipped_gradient,
                gradient_finite=gradient_finite,
            )
        )
        if (
            not np.isfinite([minibatch_loss, gradient_norm]).all()
            or not gradient_finite
            or any(
                not np.isfinite(np.asarray(v)).all()
                for v in jax.tree.leaves(
                    (clipped_gradient, new_first, new_second, proposal)
                )
            )
        ):
            raise CandidateFitError(f"nonfinite Adam proposal at attempt {i}")
        first, second = new_first, new_second
        accepted_scale, accepted_trial_index = 0.0, None
        for trial_index, scale in enumerate(SCALES):
            trial = trial_parameters(params, proposal, scale)
            trial_loss = float(full_training_objective(trial))
            objective_calls += 1
            _observe_attempt(
                dict(
                    phase="trial",
                    attempt=i,
                    trial_index=trial_index,
                    scale=scale,
                    loss=trial_loss,
                    parameters=trial,
                )
            )
            if np.isfinite(trial_loss) and trial_loss < current_full_training_loss:
                params, current_full_training_loss = trial, trial_loss
                accepted_scale, accepted_trial_index = scale, trial_index
                break
        accepted_scales.append(accepted_scale)
        _observe_attempt(
            dict(
                phase="completed",
                attempt=i,
                params=params,
                first=first,
                second=second,
                loss=current_full_training_loss,
                accepted_scale=accepted_scale,
                accepted_trial_index=accepted_trial_index,
            )
        )
        if time.perf_counter() - fit_started_at > FIT_WALL_TIME_LIMIT_S:
            raise CandidateFitError("fit_time_limit")
        if i % check_every == 0 or i == steps:
            train_loss = minibatch_loss
            val_loss, unweighted_val_loss = map(float, evaluate(params))
            if not np.isfinite([train_loss, val_loss, unweighted_val_loss]).all():
                raise CandidateFitError(f"nonfinite sequence training at step {i}")
            trace.append(
                dict(
                    step=i,
                    training_batch_mse=train_loss,
                    validation_rollout_mse=val_loss,
                )
            )
            unweighted_development_trace.append(
                dict(step=i, validation_rollout_mse=unweighted_val_loss)
            )
            checkpoint_losses.append(
                dict(step=i, full_training_loss=current_full_training_loss)
            )
            if val_loss < best_loss:
                best, best_loss, best_step = params, val_loss, i
    if time.perf_counter() - fit_started_at > FIT_WALL_TIME_LIMIT_S:
        raise CandidateFitError("fit_time_limit")
    result = BalancedQuadraticSequenceModel(
        KIND,
        train.dt_s,
        model.history_steps,
        jax.tree.map(np.asarray, best),
        jax.tree.map(np.asarray, norms),
        delay,
    )
    return result, dict(
        kind=KIND,
        mechanism=EXPERIMENT,
        steps=steps,
        selected_step=best_step,
        validation_rollout_mse=best_loss,
        seed=seed,
        minibatch_seed=seed + 10000,
        trace=trace,
        parameter_count=sum(v.size for v in result.params.values()),
        loss_scale_mode="explicit_horizon_channel",
        objective=weighting_metadata(
            model,
            initial_training_prediction,
            initial_channel_mse,
            raw_channel_weights,
            channel_weights,
            weight_floor,
            weight_normalizer,
        ),
        unweighted_development_trace=unweighted_development_trace,
        selection_objective="fixed_initial_training_channel_balance",
        safeguard=safeguard_metadata(
            steps=steps,
            accepted_scales=accepted_scales,
            objective_calls=objective_calls,
            initial_loss=initial_full_training_loss,
            final_loss=current_full_training_loss,
            checkpoint_losses=checkpoint_losses,
            selected_step=best_step,
        ),
        error_scale=np.asarray(normalization).tolist(),
        context_steps=model.history_steps,
        delay_steps=delay,
    )


class CandidateDynamics(_wrapper._shared.CandidateDynamics):
    """Self-contained saved research mean; weights do not enter prediction."""


# Archive, cache preparation and envelope calibration keep isolated historical
# globals. Only the private fitter, recipe identity and model classes are rebound.
_wrapper.RECIPE = RECIPE
_wrapper.FITTING_RECIPE = FITTING_RECIPE
_wrapper.fit_candidate_sequence = fit_candidate_sequence
_wrapper.CandidateDynamics = CandidateDynamics
_wrapper._shared.RECIPE = RECIPE
_wrapper._shared._FORMAT = _FORMAT
_wrapper._shared._MODEL_FORMAT = _MODEL_FORMAT
_wrapper._shared.BilinearSequenceModel = BalancedQuadraticSequenceModel
_wrapper._shared.CandidateDynamics = CandidateDynamics


def fit_candidate(public_reference):
    """Fit once from the imported public model's exact training/development caches."""
    model = _wrapper.fit_candidate(public_reference)
    model._report["evidence_limits"].append(
        "bounded full-training acceptance safeguards existing Adam proposals; equal proposal counts do not mean equal accepted updates, compute, or generalization"
    )
    return model
