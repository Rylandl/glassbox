"""Frozen state-input-interaction-v1 research candidate, not a public recipe.

The sole added mechanism is a ridge-initialized observed-state/current-command
bilinear output. Memory and neural features, initialization draws, training
windows, optimizer and checkpoint rule remain those of the adopted learner.
"""

from __future__ import annotations

import copy
from dataclasses import asdict

import jax
import jax.numpy as jnp
import numpy as np

from .._learner_arrays import array_fingerprint, load_arrays, save_arrays
from .._sequence_model import (
    KIND,
    SequenceBatch,
    SequenceModel,
    _features,
    _filter,
    initialize_sequence_model,
)
from ..learner import (
    ENVELOPE_COVERAGE,
    LearnedDynamics,
    _calibrate,
    _measure,
    steps_for,
)
from ..learner import RECIPE as BASE_RECIPE
from ..recordings import SequenceWindows, WindowKey

EXPERIMENT = "state-input-interaction-v1"
RECIPE = {**BASE_RECIPE, "id": EXPERIMENT, "interaction": "current_state_input_outer"}
_MODEL_FORMAT = "glassbox-state-input-sequence-v1"
_FORMAT = "glassbox-state-input-candidate-v1"
_ARRAYS = ("past_states", "past_inputs", "future_inputs", "future_states")


class CandidateFitError(ValueError):
    """A nonfinite candidate optimization/calibration outcome, not a harness bug."""


def interaction_features(x, u, xp=jnp):
    """State-major/input-minor products, preserving every leading dimension."""
    return (x[..., :, None] * u[..., None, :]).reshape(
        (*x.shape[:-1], x.shape[-1] * u.shape[-1])
    )


def _rollout(params, norms, past, past_u, future_u, delay, memory=None):
    x = (past - norms["state_mean"]) / norms["state_scale"]
    up = (past_u - norms["input_mean"]) / norms["input_scale"]
    uf = (future_u - norms["input_mean"]) / norms["input_scale"]
    h = _filter(params, norms, x, up, delay, memory)
    x, up = x[:, -delay - 1 :], up[:, -delay:]

    def step(carry, command):
        current, history, inputs, hidden = carry
        z = _features(current, command, history, inputs, hidden)
        z = z / norms["feature_scale"]
        q = interaction_features(current, command) / norms["interaction_scale"]
        delta = z @ params["linear"] + q @ params["interaction"] + params["bias"]
        delta = delta + jnp.tanh(z @ params["w1"] + params["b1"]) @ params["w2"]
        predicted = current + delta * norms["delta_scale"]
        hidden = jnp.tanh(z @ params["memory"] + params["memory_bias"])
        return (
            predicted,
            jnp.concatenate((history[:, 1:], current[:, None]), axis=1),
            jnp.concatenate((inputs[:, 1:], command[:, None]), axis=1),
            hidden,
        ), predicted

    _, predicted = jax.lax.scan(step, (x[:, -1], x[:, :-1], up, h), uf.swapaxes(0, 1))
    return predicted.swapaxes(0, 1) * norms["state_scale"] + norms["state_mean"]


class BilinearSequenceModel(SequenceModel):
    """The baseline memory contract plus one separately serialized output path."""

    def __post_init__(self):
        super().__post_init__()
        d, m = len(self.norms["state_mean"]), len(self.norms["input_mean"])
        if self.params.get("interaction", np.empty(0)).shape != (d * m, d):
            raise ValueError(
                "interaction coefficients must have shape [state*input,state]"
            )
        scale = np.asarray(self.norms.get("interaction_scale", []))
        if (
            scale.shape != (d * m,)
            or not np.isfinite(scale).all()
            or np.any(scale <= 0)
        ):
            raise ValueError("interaction scales must be finite positive [state*input]")

    def rollout(self, past_states, past_inputs, future_inputs, *, memory=None):
        x, up, uf = map(jnp.asarray, (past_states, past_inputs, future_inputs))
        single = x.ndim == 2
        if memory is not None:
            memory = jnp.asarray(memory)
            if single:
                memory = memory[None]
        if single:
            x, up, uf = x[None], up[None], uf[None]
        self._check(x, up, uf, memory)
        y = _rollout(self.params, self.norms, x, up, uf, self.delay_steps, memory)
        return y[0] if single else y

    def metadata(self):
        return {**super().metadata(), "format": _MODEL_FORMAT}

    @classmethod
    def load(cls, path):
        metadata, arrays = load_arrays(path)
        if metadata.pop("format") != _MODEL_FORMAT:
            raise ValueError("unsupported state-input sequence format")
        return cls(
            **metadata,
            params={k[6:]: v for k, v in arrays.items() if k.startswith("param_")},
            norms={k[5:]: v for k, v in arrays.items() if k.startswith("norm_")},
        )


def initialize_candidate(
    batch, *, seed=0, width=32, memory=8, ridge=1.0, delay_steps=None
):
    """Joint ridge output initialization with unchanged baseline random draws."""
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
    x = np.concatenate((batch.past_states, batch.future_states), axis=1)
    u = np.concatenate((batch.past_inputs, batch.future_inputs), axis=1)
    x = (x - norms["state_mean"]) / norms["state_scale"]
    u = (u - norms["input_mean"]) / norms["input_scale"]
    current = x[:, context:-1]
    commands = u[:, context:]
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
    products = interaction_features(current, commands, xp=np).reshape(
        -1, current.shape[-1] * commands.shape[-1]
    )
    scales = products.std(axis=0)
    scales = np.where(scales > 1e-8, scales, 1.0)
    design = np.column_stack(
        (features / norms["feature_scale"], products / scales, np.ones(len(features)))
    )
    penalty = np.diag(np.r_[np.full(design.shape[1] - 1, ridge), 0.0])
    # Preserve baseline subtraction order: form physical increments before scaling.
    physical_current = np.concatenate((batch.past_states, batch.future_states), 1)[
        :, context:-1
    ]
    delta = (batch.future_states - physical_current) / norms["state_scale"]
    target = (delta / norms["delta_scale"]).reshape(len(features), -1)
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ target)
    count = features.shape[-1]
    params["linear"], params["interaction"], params["bias"] = (
        coefficients[:count],
        coefficients[count:-1],
        coefficients[-1],
    )
    norms["interaction_scale"] = scales
    return BilinearSequenceModel(KIND, batch.dt_s, context, params, norms, p)


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
    """The unchanged baseline Adam loop operating on the bilinear candidate."""
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
    if error_scale is None:
        normalization = norms["state_scale"]
    else:
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

    def loss(par, data):
        x, up, uf, target = data
        prediction = _rollout(par, norms, x, up, uf, delay)
        return jnp.mean(((prediction - target) / normalization) ** 2)

    @jax.jit
    def evaluate(par):
        return loss(par, development)

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
    best_loss = float(evaluate(params))
    if not np.isfinite(best_loss):
        raise CandidateFitError("initial recursive validation loss is nonfinite")
    trace = [dict(step=0, validation_rollout_mse=best_loss)]
    first, second = (jax.tree.map(jnp.zeros_like, params) for _ in range(2))
    rng = np.random.default_rng(seed + 10000)
    best_step = 0
    for i in range(1, steps + 1):
        indices = rng.integers(
            len(train.past_states), size=min(batch_size, len(train.past_states))
        )
        params, first, second, value = update(params, first, second, i, indices)
        if i % check_every == 0 or i == steps:
            train_loss, val_loss = float(value), float(evaluate(params))
            if not np.isfinite([train_loss, val_loss]).all():
                raise CandidateFitError(f"nonfinite sequence training at step {i}")
            trace.append(
                dict(
                    step=i,
                    training_batch_mse=train_loss,
                    validation_rollout_mse=val_loss,
                )
            )
            if val_loss < best_loss:
                best, best_loss, best_step = params, val_loss, i
    result = BilinearSequenceModel(
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
        loss_scale_mode="state_standard_deviation"
        if error_scale is None
        else "explicit_horizon_channel",
        error_scale=np.asarray(normalization).tolist(),
        context_steps=model.history_steps,
        delay_steps=delay,
    )


def _window_metadata(windows):
    return dict(
        keys=[asdict(k) for k in windows.keys],
        source_origins=list(windows.source_origins),
    )


def _window_fingerprint(windows):
    return array_fingerprint(
        {**_window_metadata(windows), "dt_s": windows.batch.dt_s},
        {k: getattr(windows.batch, k) for k in _ARRAYS},
    )


def fit_candidate(baseline):
    """Fit only the frozen candidate on an adopted model's exact cached windows."""
    if not isinstance(baseline, LearnedDynamics) or baseline.recipe != BASE_RECIPE:
        raise TypeError("candidate fit requires the adopted LearnedDynamics baseline")
    train, development = baseline._train, baseline._development
    b = train.batch
    recipe = BASE_RECIPE
    delay = steps_for(b.dt_s)["delay"]
    ridge = recipe["ridge_fraction"] * len(b.past_states) * b.future_states.shape[1]
    initial = initialize_sequence_model(
        b,
        width=recipe["width"],
        memory=recipe["memory"],
        seed=recipe["seed"],
        ridge=ridge,
        delay_steps=delay,
    )
    if any(
        not np.array_equal(value, baseline._model.norms[name])
        for name, value in initial.norms.items()
    ):
        raise ValueError(
            "baseline normalization does not match its cached training data"
        )
    hold = np.repeat(b.past_states[:, -1:], b.future_states.shape[1], 1)
    error_scale = np.maximum(
        np.sqrt(np.mean((hold - b.future_states) ** 2, 0)),
        recipe["hold_scale_floor"] * initial.norms["state_scale"],
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
            baseline_norms_fingerprint=array_fingerprint({}, initial.norms),
            error_scale_fingerprint=array_fingerprint({}, {"error_scale": error_scale}),
            shared_initialization_fingerprint=array_fingerprint(
                {},
                {
                    k: initial.params[k]
                    for k in ("w1", "b1", "w2", "memory", "memory_bias")
                },
            ),
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
            "state-input interactions are learned from ordinary recordings, not paired interventions",
            "matched data and optimizer updates do not imply equal FLOPs",
            "Euclidean forecasts do not enforce manifold constraints",
        ],
    )
    return CandidateDynamics(
        model, train, development, baseline.contract, report, envelope
    )


class CandidateDynamics:
    """Saved research mean, evidence and exact fit caches; no public adoption API."""

    def __init__(self, model, train, development, contract, report, envelope):
        if not isinstance(model, BilinearSequenceModel) or report["recipe"] != RECIPE:
            raise ValueError("unsupported state-input candidate recipe")
        self._model, self._train, self._development = model, train, development
        self._contract, self._report = copy.deepcopy(contract), copy.deepcopy(report)
        self._envelope = np.array(envelope, dtype=float, copy=True)
        if (
            self._envelope.shape
            != (self.horizon_steps, len(contract["state_channels"]))
            or not np.isfinite(self._envelope).all()
            or np.any(self._envelope < 0)
        ):
            raise ValueError(
                "candidate envelope must be finite nonnegative [horizon,state]"
            )
        self._envelope.setflags(write=False)

    @property
    def history_steps(self):
        return self._model.history_steps

    @property
    def horizon_steps(self):
        return self._train.batch.future_states.shape[1]

    @property
    def contract(self):
        return copy.deepcopy(self._contract)

    @property
    def report(self):
        return copy.deepcopy(self._report)

    @property
    def recipe(self):
        return copy.deepcopy(RECIPE)

    def predict(self, past_states, past_inputs, future_inputs):
        x, up, uf = map(jnp.asarray, (past_states, past_inputs, future_inputs))
        if x.ndim not in (2, 3) or up.ndim != x.ndim or uf.ndim != x.ndim:
            raise ValueError(
                "predict expects aligned [time,channel] or [batch,time,channel]"
            )
        if x.shape[-2] != up.shape[-2] + 1 or x.shape[-2] < self.history_steps + 1:
            raise ValueError("predict requires complete aligned observed history")
        if not 1 <= uf.shape[-2] <= self.horizon_steps:
            raise ValueError("unsupported forecast horizon")
        return self._model.rollout(
            x[..., -self.history_steps - 1 :, :], up[..., -self.history_steps :, :], uf
        )

    def envelope(self, horizon_steps=None):
        steps = self.horizon_steps if horizon_steps is None else int(horizon_steps)
        if not 1 <= steps <= self.horizon_steps:
            raise ValueError("unsupported forecast horizon")
        return np.array(self._envelope[:steps], dtype=float)

    def _metadata(self):
        return dict(
            format=_FORMAT,
            recipe=copy.deepcopy(RECIPE),
            model=self._model.metadata(),
            contract=self.contract,
            report=self.report,
            windows={
                role: _window_metadata(windows)
                for role, windows in (
                    ("train", self._train),
                    ("development", self._development),
                )
            },
        )

    def _arrays(self):
        return {
            **self._model.arrays(),
            "envelope_half_width": self._envelope,
            **{
                f"{role}_{k}": getattr(windows.batch, k)
                for role, windows in (
                    ("train", self._train),
                    ("development", self._development),
                )
                for k in _ARRAYS
            },
        }

    def fingerprint(self):
        return array_fingerprint(self._metadata(), self._arrays())

    def save(self, path):
        save_arrays(path, self._metadata(), self._arrays())

    @classmethod
    def load(cls, path):
        metadata, arrays = load_arrays(path)
        if metadata.get("recipe") != RECIPE or metadata.get("format") != _FORMAT:
            raise ValueError("unsupported state-input candidate format")
        model_metadata = dict(metadata["model"])
        if model_metadata.pop("format") != _MODEL_FORMAT:
            raise ValueError("unsupported state-input sequence format")
        model = BilinearSequenceModel(
            **model_metadata,
            params={k[6:]: v for k, v in arrays.items() if k.startswith("param_")},
            norms={k[5:]: v for k, v in arrays.items() if k.startswith("norm_")},
        )
        windows = {}
        for role in ("train", "development"):
            saved = metadata["windows"][role]
            windows[role] = SequenceWindows(
                SequenceBatch(
                    **{k: arrays[f"{role}_{k}"] for k in _ARRAYS}, dt_s=model.dt_s
                ),
                tuple(WindowKey(**k) for k in saved["keys"]),
                tuple(saved["source_origins"]),
            )
        return cls(
            model,
            windows["train"],
            windows["development"],
            metadata["contract"],
            metadata["report"],
            arrays["envelope_half_width"],
        )
