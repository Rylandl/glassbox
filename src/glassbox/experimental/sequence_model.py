"""Small generic recurrent predictors trained on observed trajectory segments.

All predicted observation channels advance recursively; future observations are
never required by rollout(). Optional latent memory is initialized only from
past observations/inputs. Coordinates are treated as Euclidean: this module does
not enforce rotation geometry, infer physical bounds, or calibrate uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from .sequence_objective import SequenceGuard
from .structured_regression import array_fingerprint, load_arrays, save_arrays


@dataclass(frozen=True)
class SequenceBatch:
    """Past x[-P:0], u[-P:-1]; future u[0:H-1], target x[1:H]."""

    past_states: np.ndarray
    past_inputs: np.ndarray
    future_inputs: np.ndarray
    future_states: np.ndarray
    dt_s: float

    def __post_init__(self):
        for key in ("past_states", "past_inputs", "future_inputs", "future_states"):
            value = np.array(getattr(self, key), dtype=float, copy=True)
            if value.ndim != 3 or not np.isfinite(value).all():
                raise ValueError(f"{key} must be a finite [batch,time,channel] array")
            value.setflags(write=False)
            object.__setattr__(self, key, value)
        n, p1, d = self.past_states.shape
        nf, h, u = self.future_inputs.shape
        if (
            n < 3
            or min(p1 - 1, d, h, u) < 1
            or nf != n
            or self.past_inputs.shape != (n, p1 - 1, u)
            or self.future_states.shape != (n, h, d)
            or not np.isfinite(self.dt_s)
            or self.dt_s <= 0
        ):
            raise ValueError("sequence shapes or dt_s are inconsistent")


def sequence_windows(states, inputs, anchors, *, history_steps, horizon_steps, dt_s):
    """Extract complete, unpadded windows from exactly one recording."""
    x, u, a = np.asarray(states), np.asarray(inputs), np.asarray(anchors)
    if (
        x.ndim != 2
        or u.ndim != 2
        or len(u) != len(x) - 1
        or a.ndim != 1
        or not np.issubdtype(a.dtype, np.integer)
        or len(np.unique(a)) != len(a)
        or not isinstance(history_steps, int)
        or not isinstance(horizon_steps, int)
        or min(history_steps, horizon_steps) < 1
        or np.any(a < history_steps)
        or np.any(a + horizon_steps >= len(x))
    ):
        raise ValueError("invalid or incomplete sequence windows")
    return SequenceBatch(
        x[a[:, None] + np.arange(-history_steps, 1)],
        u[a[:, None] + np.arange(-history_steps, 0)],
        u[a[:, None] + np.arange(horizon_steps)],
        x[a[:, None] + np.arange(1, horizon_steps + 1)],
        dt_s,
    )


def _features(x, u, xpast, upast, hidden, kind, xp=jnp):
    parts = [x, u]
    if kind in ("delay", "delay_mlp"):
        parts.extend(
            (
                (xpast - x[:, None]).reshape(len(x), -1),
                (upast - u[:, None]).reshape(len(x), -1),
            )
        )
    if kind == "latent":
        parts.append(hidden)
    return xp.concatenate(parts, axis=-1)


def _rollout(params, norms, kind, past, past_u, future_u, truth=None):
    x = (past - norms["state_mean"]) / norms["state_scale"]
    up = (past_u - norms["input_mean"]) / norms["input_scale"]
    uf = (future_u - norms["input_mean"]) / norms["input_scale"]
    h = jnp.zeros((len(x), 0), dtype=x.dtype)
    if kind == "latent":
        encoder_input = jnp.concatenate(
            (x.reshape(len(x), -1), up.reshape(len(x), -1)), -1
        )
        h = jnp.tanh(encoder_input @ params["encoder"] + params["encoder_bias"])

    def step(carry, data):
        current, history, inputs, hidden = carry
        command, target = data
        z = _features(current, command, history, inputs, hidden, kind)
        z = z / norms["feature_scale"]
        delta = z @ params["linear"] + params["bias"]
        if kind in ("mlp", "latent", "delay_mlp"):
            delta = delta + jnp.tanh(z @ params["w1"] + params["b1"]) @ params["w2"]
        predicted = current + delta * norms["delta_scale"]
        if kind == "latent":
            hidden = jnp.tanh(z @ params["memory"] + params["memory_bias"])
        next_current = predicted if truth is None else target
        return (
            next_current,
            jnp.concatenate((history[:, 1:], current[:, None]), axis=1),
            jnp.concatenate((inputs[:, 1:], command[:, None]), axis=1),
            hidden,
        ), predicted

    targets = (
        jnp.zeros((*uf.shape[:2], x.shape[-1]), dtype=x.dtype)
        if truth is None
        else (truth - norms["state_mean"]) / norms["state_scale"]
    )
    _, predicted = jax.lax.scan(
        step,
        (x[:, -1], x[:, :-1], up, h),
        (uf.swapaxes(0, 1), targets.swapaxes(0, 1)),
    )
    return predicted.swapaxes(0, 1) * norms["state_scale"] + norms["state_mean"]


@dataclass(frozen=True)
class SequenceModel:
    kind: str
    dt_s: float
    history_steps: int
    params: dict
    norms: dict

    def rollout(self, past_states, past_inputs, future_inputs):
        """Return future means, excluding the initial observation; JAX compatible."""
        x, up, uf = map(jnp.asarray, (past_states, past_inputs, future_inputs))
        single = x.ndim == 2
        if single:
            x, up, uf = x[None], up[None], uf[None]
        d, u = len(self.norms["state_mean"]), len(self.norms["input_mean"])
        if (
            x.ndim != 3
            or up.ndim != 3
            or uf.ndim != 3
            or x.shape[1:] != (self.history_steps + 1, d)
            or up.shape != (len(x), self.history_steps, u)
            or uf.shape[0] != len(x)
            or uf.shape[-1] != u
            or uf.shape[1] < 1
        ):
            raise ValueError("rollout shapes do not match model history/channels")
        y = _rollout(self.params, self.norms, self.kind, x, up, uf)
        return y[0] if single else y

    def metadata(self):
        return dict(
            format="glassbox-sequence-v1",
            kind=self.kind,
            dt_s=self.dt_s,
            history_steps=self.history_steps,
        )

    def arrays(self):
        return {
            **{f"param_{k}": v for k, v in self.params.items()},
            **{f"norm_{k}": v for k, v in self.norms.items()},
        }

    def fingerprint(self):
        return array_fingerprint(self.metadata(), self.arrays())

    def save(self, path):
        save_arrays(path, self.metadata(), self.arrays())

    @classmethod
    def load(cls, path):
        meta, arrays = load_arrays(path)
        if meta.pop("format") != "glassbox-sequence-v1":
            raise ValueError("unsupported sequence format")
        return cls(
            **meta,
            params={k[6:]: v for k, v in arrays.items() if k.startswith("param_")},
            norms={k[5:]: v for k, v in arrays.items() if k.startswith("norm_")},
        )


def initialize_sequence_model(
    batch, *, kind="latent", seed=0, width=32, memory=8, ridge=1.0
):
    """Initialize from a learned one-step affine model, with zero neural residual."""
    if kind not in ("linear", "delay", "mlp", "latent", "delay_mlp"):
        raise ValueError("unknown sequence model kind")
    if min(width, memory) < 1 or not np.isfinite(ridge) or ridge <= 0:
        raise ValueError("width, memory and ridge must be positive")
    p = batch.past_inputs.shape[1]
    complete_x = np.concatenate((batch.past_states, batch.future_states), 1)
    complete_u = np.concatenate((batch.past_inputs, batch.future_inputs), 1)
    current = complete_x[:, p:-1]
    xm, xs = current.mean((0, 1)), current.std((0, 1))
    um, us = batch.future_inputs.mean((0, 1)), batch.future_inputs.std((0, 1))
    xs, us = np.where(xs > 1e-8, xs, 1), np.where(us > 1e-8, us, 1)
    xall, uall = (complete_x - xm) / xs, (complete_u - um) / us
    delta = (batch.future_states - current) / xs
    ds = np.maximum(delta.std((0, 1)), 1e-4)
    hidden = np.zeros((len(current), memory if kind == "latent" else 0))
    features = np.stack(
        [
            _features(
                xall[:, p + t],
                uall[:, p + t],
                xall[:, t : p + t],
                uall[:, t : p + t],
                hidden,
                kind,
                xp=np,
            )
            for t in range(current.shape[1])
        ],
        1,
    ).reshape(
        -1,
        current.shape[-1]
        + batch.future_inputs.shape[-1]
        + (
            p * (current.shape[-1] + batch.future_inputs.shape[-1])
            if kind in ("delay", "delay_mlp")
            else 0
        )
        + (memory if kind == "latent" else 0),
    )
    fs = features.std(0)
    fs = np.where(fs > 1e-8, fs, 1.0)
    design = np.column_stack((features / fs, np.ones(len(features))))
    penalty = np.diag(np.r_[np.full(features.shape[-1], ridge), 0.0])
    coefficients = np.linalg.solve(
        design.T @ design + penalty, design.T @ (delta / ds).reshape(len(features), -1)
    )
    params = dict(linear=coefficients[:-1], bias=coefficients[-1])
    rng = np.random.default_rng(seed)
    f, d = coefficients.shape[0] - 1, coefficients.shape[1]
    if kind in ("mlp", "latent", "delay_mlp"):
        params.update(
            w1=rng.normal(size=(f, width)) / np.sqrt(f),
            b1=np.zeros(width),
            w2=np.zeros((width, d)),
        )
    if kind == "latent":
        encoder_width = (p + 1) * d + p * batch.future_inputs.shape[-1]
        params.update(
            encoder=rng.normal(size=(encoder_width, memory)) / np.sqrt(encoder_width),
            encoder_bias=np.zeros(memory),
            memory=rng.normal(size=(f, memory)) / np.sqrt(f),
            memory_bias=np.zeros(memory),
        )
    norms = dict(
        state_mean=xm,
        state_scale=xs,
        input_mean=um,
        input_scale=us,
        feature_scale=fs,
        delta_scale=ds,
    )
    return SequenceModel(kind, batch.dt_s, p, params, norms)


def fit_sequence_model(
    train,
    validation,
    *,
    kind="latent",
    objective="rollout",
    seed=0,
    steps=1000,
    batch_size=64,
    learning_rate=0.002,
    width=32,
    memory=8,
    ridge=1.0,
    check_every=100,
    error_scale=None,
    selection_guard: SequenceGuard | None = None,
):
    """Adam with gradient clipping and development-rollout checkpoint selection.

    Teacher forcing is a training-only ablation. Every checkpoint is evaluated
    recursively. By default all future channels have equal weight after train-only
    state scaling. error_scale may instead supply positive [horizon,channel] loss
    scales. A selection_guard rejects development regressions vs initialization.
    """
    if objective not in ("rollout", "teacher") or steps < 0 or check_every < 1:
        raise ValueError("invalid objective or training steps")
    if batch_size < 1 or not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("batch_size and learning_rate must be positive")
    if train.dt_s != validation.dt_s or any(
        getattr(train, k).shape[1:] != getattr(validation, k).shape[1:]
        for k in ("past_states", "past_inputs", "future_states", "future_inputs")
    ):
        raise ValueError("train and validation contracts differ")
    model = initialize_sequence_model(
        train, kind=kind, seed=seed, width=width, memory=memory, ridge=ridge
    )
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
    if selection_guard is not None:
        selection_guard.validate_shape(*train.future_states.shape[1:])
    names = ("past_states", "past_inputs", "future_inputs", "future_states")
    training = tuple(jnp.asarray(getattr(train, key)) for key in names)
    development = tuple(jnp.asarray(getattr(validation, key)) for key in names)

    def loss(par, data, teacher):
        x, up, uf, target = data
        prediction = _rollout(par, norms, kind, x, up, uf, target if teacher else None)
        return jnp.mean(((prediction - target) / normalization) ** 2)

    @jax.jit
    def evaluate(par):
        return loss(par, development, False)

    @jax.jit
    def guard_errors(par):
        x, up, uf, target = development
        predicted = _rollout(par, norms, kind, x, up, uf)
        return selection_guard.errors(predicted, target)

    @jax.jit
    def update(par, first, second, index, indices):
        data = tuple(value[indices] for value in training)
        value, grad = jax.value_and_grad(loss)(par, data, objective == "teacher")
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
        raise ValueError("initial recursive validation loss is nonfinite")
    trace = [dict(step=0, validation_rollout_mse=best_loss)]
    reference_errors = None
    if selection_guard is not None:
        reference_errors = np.asarray(guard_errors(params))
        trace[0].update(guard_errors=reference_errors.tolist(), guard_accepted=True)
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
                raise ValueError(f"nonfinite sequence training at step {i}")
            accepted = True
            guard_report = {}
            if selection_guard is not None:
                candidate_errors = np.asarray(guard_errors(params))
                accepted = selection_guard.accepts(candidate_errors, reference_errors)
                guard_report = dict(
                    guard_errors=candidate_errors.tolist(), guard_accepted=accepted
                )
            trace.append(
                dict(
                    step=i,
                    training_batch_mse=train_loss,
                    validation_rollout_mse=val_loss,
                    **guard_report,
                )
            )
            if accepted and val_loss < best_loss:
                best, best_loss, best_step = params, val_loss, i
    result = SequenceModel(
        kind,
        train.dt_s,
        model.history_steps,
        jax.tree.map(np.asarray, best),
        jax.tree.map(np.asarray, norms),
    )
    return result, dict(
        kind=kind,
        objective=objective,
        steps=steps,
        selected_step=best_step,
        validation_rollout_mse=best_loss,
        seed=seed,
        trace=trace,
        parameter_count=sum(v.size for v in result.params.values()),
        loss_scale_mode="state_standard_deviation"
        if error_scale is None
        else "explicit_horizon_channel",
        error_scale=np.asarray(normalization).tolist(),
        selection_guard=None if selection_guard is None else selection_guard.metadata(),
    )
