"""Private fixed initial-training channel balance on the anchored quadratic model.

Only the training/development objective changes. Prediction, initialization and
calibration reuse their pinned implementations; no public fitting option is added.
"""

from __future__ import annotations

import copy
import importlib.util

import jax
import jax.numpy as jnp
import numpy as np

from .._learner_arrays import array_fingerprint, load_arrays
from .._sequence_model import KIND
from ..learner import RECIPE as BASE_RECIPE
from . import affine_anchored_model as anchored

_spec = importlib.util.spec_from_file_location(
    f"{__package__}._initial_channel_balance_wrapper", anchored.__file__
)
_wrapper = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_wrapper)

EXPERIMENT = "initial-channel-balance-v1"
OBJECTIVE = "fixed_initial_training_channel_balance"
RECIPE = {**anchored.RECIPE, "id": EXPERIMENT, "objective": OBJECTIVE}
FITTING_RECIPE = copy.deepcopy(BASE_RECIPE)
_MODEL_FORMAT = "glassbox-initial-channel-balance-sequence-v1"
_FORMAT = "glassbox-initial-channel-balance-candidate-v1"
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
            raise ValueError("unsupported initial-channel-balance sequence format")
        return cls(
            **metadata,
            params={k[6:]: v for k, v in arrays.items() if k.startswith("param_")},
            norms={k[5:]: v for k, v in arrays.items() if k.startswith("norm_")},
        )


def initial_training_forecast(params, norms, train, delay):
    """The exact shared eager/scan forecast path for fitting and witness replay."""
    params, norms = jax.tree.map(jnp.asarray, (params, norms))
    inputs = tuple(jnp.asarray(getattr(train, key)) for key in _ARRAYS[:3])
    return np.array(_rollout(params, norms, *inputs, delay), copy=True)


def weighting_metadata(model, prediction, mse, raw, weights, floor, normalizer):
    """Small saved model report; the actual prospective arrays are external evidence."""
    return dict(
        id=OBJECTIVE,
        reduction="numpy_float64_mean_over_training_windows_and_horizon",
        weighting_data="initial_recursive_training_predictions_only",
        initial_channel_mse=np.asarray(mse).tolist(),
        raw_channel_weights=np.asarray(raw).tolist(),
        channel_weights=np.asarray(weights).tolist(),
        weight_floor=float(floor),
        weight_normalizer=float(normalizer),
        floor_active_channels=np.flatnonzero(mse < floor).tolist(),
        initial_parameters_and_norms_fingerprint=array_fingerprint({}, model.arrays()),
        initial_training_prediction_fingerprint=array_fingerprint(
            {}, {"prediction": prediction}
        ),
        fixed_during_training_and_selection=True,
        original_hold_scale_preserved=True,
        additional_training_weight_forecasts=1,
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
    """One anchored initialization and a fixed training-derived channel objective."""
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
        return par, first, second, value

    best = params
    best_loss, unweighted_best_initial_loss = map(float, evaluate(params))
    if not np.isfinite(best_loss):
        raise CandidateFitError("initial recursive validation loss is nonfinite")
    trace = [dict(step=0, validation_rollout_mse=best_loss)]
    unweighted_development_trace = [
        dict(step=0, validation_rollout_mse=unweighted_best_initial_loss)
    ]
    first, second = (jax.tree.map(jnp.zeros_like, params) for _ in range(2))
    rng = np.random.default_rng(seed + 10000)
    best_step = 0
    for i in range(1, steps + 1):
        indices = rng.integers(
            len(train.past_states), size=min(batch_size, len(train.past_states))
        )
        params, first, second, value = update(params, first, second, i, indices)
        if i % check_every == 0 or i == steps:
            train_loss = float(value)
            val_loss, unweighted_val_loss = map(float, evaluate(params))
            if not np.isfinite([train_loss, val_loss]).all():
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
            if val_loss < best_loss:
                best, best_loss, best_step = params, val_loss, i
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
        "fixed training-derived channel weighting changes optimization and checkpoint selection; it is not uncertainty or an all-channel guarantee"
    )
    return model
