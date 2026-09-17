"""One generic recurrent predictor trained on observed trajectory segments.

Every predicted observation channel advances recursively; future observations
are never required by rollout(). The latent memory is initialized only from
past observations/inputs. Coordinates are treated as Euclidean: this module
does not enforce rotation geometry, infer physical bounds, or calibrate
uncertainty.

Causal memory contract: the model consumes a fixed number of consecutive
observed transitions before the forecast origin. Its memory starts at rest at
the first consumed observation, advances once per observed transition inside
one recording segment, and continues from predicted observations during the
forecast. The consumed context is the information budget; nothing before it is
implied. A caller may carry the memory within a recording through
``memory_state`` and ``rollout(..., memory=...)``, which is exactly equivalent
to consuming the whole context at once.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from .arrays import array_fingerprint, load_arrays, save_arrays

KIND = "filter_mlp"


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


def command_feature_rows(state_width, command_width, delay_steps, channel):
    """The affine block's own columns for one command channel.

    The feature vector ``_features`` builds is
    ``[x, u, xpast - x, upast - u, hidden]``, so one command channel owns its
    level column and one difference column per explicit lag, and nothing else
    in the design carries that command.
    """
    d, u, p = int(state_width), int(command_width), int(delay_steps)
    return [d + channel] + [d + u + p * d + lag * u + channel for lag in range(p)]


def held_command_coefficients(norms, response, held, state_width, command_width, delay):
    """Affine coefficients that express a declared one-step command response.

    ``response`` is physical state change per unit command, one row per command
    channel. A held channel's level column carries the whole of it, in the
    design's own normalized units, and its difference columns carry zero, so
    the affine block's response to that command is exactly ``response`` and
    nothing else in the block can restate it.
    """
    rows, values = [], []
    for channel in np.flatnonzero(np.asarray(held, bool)):
        columns = command_feature_rows(state_width, command_width, delay, int(channel))
        level = columns[0]
        rows.append(level)
        values.append(
            np.asarray(response[channel], dtype=float)
            * norms["input_scale"][channel]
            * norms["feature_scale"][level]
            / (norms["state_scale"] * norms["delta_scale"])
        )
        for column in columns[1:]:
            rows.append(column)
            values.append(np.zeros(len(norms["state_scale"])))
    return np.asarray(rows, dtype=int), (
        np.asarray(values, dtype=float)
        if values
        else np.zeros((0, len(norms["state_scale"])))
    )


def _features(x, u, xpast, upast, hidden, xp=jnp):
    return xp.concatenate(
        [
            x,
            u,
            (xpast - x[:, None]).reshape(len(x), -1),
            (upast - u[:, None]).reshape(len(x), -1),
            hidden,
        ],
        axis=-1,
    )


def _filter(params, norms, x, up, delay, memory=None):
    """Advance the memory over every observed transition in normalized arrays.

    Transition j (from x[:, j] to x[:, j+1]) is consumed for j from ``delay``
    to the last supplied input, because each update needs ``delay`` earlier
    observations and inputs. The memory starts at rest unless supplied.
    """
    context = up.shape[1]
    if memory is None:
        memory = jnp.zeros((len(x), params["memory"].shape[1]), dtype=x.dtype)
    if context <= delay:
        return memory
    observed = (
        jnp.stack([x[:, j - delay : j + 1] for j in range(delay, context)]),
        jnp.stack([up[:, j - delay : j + 1] for j in range(delay, context)]),
    )

    def observe(hidden, data):
        states, inputs = data
        z = _features(
            states[:, -1],
            inputs[:, -1],
            states[:, :-1],
            inputs[:, :-1],
            hidden,
        )
        z = z / norms["feature_scale"]
        return jnp.tanh(z @ params["memory"] + params["memory_bias"]), None

    memory, _ = jax.lax.scan(observe, memory, observed)
    return memory


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
        delta = z @ params["linear"] + params["bias"]
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


@dataclass(frozen=True)
class SequenceModel:
    """A recursive predictor; ``history_steps`` is the context every forecast needs.

    ``history_steps`` is the consumed context (the information budget) and
    ``delay_steps`` the shorter explicit-difference history inside it.
    """

    kind: str
    dt_s: float
    history_steps: int
    params: dict
    norms: dict
    delay_steps: int

    def __post_init__(self):
        if (
            self.kind != KIND
            or not isinstance(self.delay_steps, (int, np.integer))
            or isinstance(self.delay_steps, bool)
            or not 1 <= self.delay_steps < self.history_steps
        ):
            raise ValueError(
                f"{KIND} is the only kind and needs 1 <= delay_steps < history_steps"
            )

    def _check(self, x, up, uf, memory):
        d, u = len(self.norms["state_mean"]), len(self.norms["input_mean"])
        required = self.history_steps if memory is None else None
        if (
            x.ndim != 3
            or up.ndim != 3
            or uf.ndim != 3
            or x.shape[-1] != d
            or up.shape[:2] != (len(x), x.shape[1] - 1)
            or up.shape[-1] != u
            or (required is not None and x.shape[1] != required + 1)
            or (required is None and x.shape[1] < self.delay_steps + 1)
            or uf.shape[0] != len(x)
            or uf.shape[-1] != u
            or uf.shape[1] < 1
            or (
                memory is not None
                and memory.shape != (len(x), self.params["memory"].shape[1])
            )
        ):
            raise ValueError("rollout shapes do not match model history/channels")

    def rollout(self, past_states, past_inputs, future_inputs, *, memory=None):
        """Return future means, excluding the initial observation; JAX compatible.

        Without ``memory`` the past must hold exactly ``history_steps + 1``
        observations and the memory starts at rest at the first one. With
        ``memory`` it is the state after the transition into
        ``past_states[..., delay_steps, :]``; the past may then be any length of
        at least ``delay_steps + 1`` observations within the same recording.
        """
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

    def memory_state(self, past_states, past_inputs, *, memory=None):
        """Return the memory after consuming every supplied transition.

        Starting from rest, or from ``memory`` as defined for ``rollout``. The
        supplied observations must lie inside one recording segment; a recording
        boundary means starting again from rest.
        """
        x, up = map(jnp.asarray, (past_states, past_inputs))
        single = x.ndim == 2
        if memory is not None:
            memory = jnp.asarray(memory)
            if single:
                memory = memory[None]
        if single:
            x, up = x[None], up[None]
        width = self.params["memory"].shape[1]
        placeholder = jnp.zeros((len(x), 1, len(self.norms["input_mean"])), x.dtype)
        # Any in-segment context of at least delay_steps + 1 observations is
        # valid here; the fitted budget applies to forecasts from rest.
        self._check(
            x, up, placeholder, jnp.zeros((len(x), width)) if memory is None else memory
        )
        h = _filter(
            self.params,
            self.norms,
            (x - self.norms["state_mean"]) / self.norms["state_scale"],
            (up - self.norms["input_mean"]) / self.norms["input_scale"],
            self.delay_steps,
            memory,
        )
        return h[0] if single else h

    def metadata(self):
        return dict(
            format="glassbox-sequence-v1",
            kind=self.kind,
            dt_s=self.dt_s,
            history_steps=self.history_steps,
            delay_steps=self.delay_steps,
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
    batch,
    *,
    seed=0,
    width=32,
    memory=8,
    ridge=1.0,
    delay_steps=None,
    command_response=None,
):
    """Initialize from a learned one-step affine model, with zero neural residual.

    The affine fit uses the forecast-phase transitions of every window. The
    window's past holds the whole consumed context and ``delay_steps`` sets the
    explicit-difference history; the memory starts at rest and reads out as
    zero, so checkpoint zero is the affine start.

    ``command_response`` is ``(response, held)``: a measured one-step command
    response, physical state change per unit command and one row per command
    channel, and which of those channels it is known well enough to state. A
    held channel's affine columns are set to that response rather than solved,
    and the rest of the block is solved by the same ridge around them. With no
    channel held the solve is the unconstrained one, coefficient for
    coefficient.
    """
    if min(width, memory) < 1 or not np.isfinite(ridge) or ridge <= 0:
        raise ValueError("width, memory and ridge must be positive")
    context = batch.past_inputs.shape[1]
    if (
        not isinstance(delay_steps, (int, np.integer))
        or isinstance(delay_steps, bool)
        or not 1 <= delay_steps < context
    ):
        raise ValueError("this model needs 1 <= delay_steps < the window context")
    p = int(delay_steps)
    complete_x = np.concatenate((batch.past_states, batch.future_states), 1)
    complete_u = np.concatenate((batch.past_inputs, batch.future_inputs), 1)
    current = complete_x[:, context:-1]
    xm, xs = current.mean((0, 1)), current.std((0, 1))
    um, us = batch.future_inputs.mean((0, 1)), batch.future_inputs.std((0, 1))
    xs, us = np.where(xs > 1e-8, xs, 1), np.where(us > 1e-8, us, 1)
    xall, uall = (complete_x - xm) / xs, (complete_u - um) / us
    delta = (batch.future_states - current) / xs
    ds = np.maximum(delta.std((0, 1)), 1e-4)
    hidden = np.zeros((len(current), memory))
    features = np.stack(
        [
            _features(
                xall[:, context + t],
                uall[:, context + t],
                xall[:, context + t - p : context + t],
                uall[:, context + t - p : context + t],
                hidden,
                xp=np,
            )
            for t in range(current.shape[1])
        ],
        1,
    ).reshape(
        -1,
        (p + 1) * (current.shape[-1] + batch.future_inputs.shape[-1]) + memory,
    )
    fs = features.std(0)
    fs = np.where(fs > 1e-8, fs, 1.0)
    design = np.column_stack((features / fs, np.ones(len(features))))
    penalty = np.diag(np.r_[np.full(features.shape[-1], ridge), 0.0])
    target = (delta / ds).reshape(len(features), -1)
    norms = dict(
        state_mean=xm,
        state_scale=xs,
        input_mean=um,
        input_scale=us,
        feature_scale=fs,
        delta_scale=ds,
    )
    held_rows, held_values = (
        held_command_coefficients(
            norms,
            *command_response,
            current.shape[-1],
            batch.future_inputs.shape[-1],
            p,
        )
        if command_response is not None
        else (np.zeros(0, dtype=int), np.zeros((0, current.shape[-1])))
    )
    if held_rows.size:
        # Solve the rest of the block around the held columns rather than
        # beside them: the remaining coefficients are the ridge's best fit
        # given the command response the recordings identified.
        free = np.setdiff1d(np.arange(design.shape[1]), held_rows)
        residual = target - design[:, held_rows] @ held_values
        solved = np.linalg.solve(
            design[:, free].T @ design[:, free] + penalty[np.ix_(free, free)],
            design[:, free].T @ residual,
        )
        coefficients = np.zeros((design.shape[1], target.shape[1]))
        coefficients[free] = solved
        coefficients[held_rows] = held_values
    else:
        coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ target)
    params = dict(linear=coefficients[:-1], bias=coefficients[-1])
    rng = np.random.default_rng(seed)
    f, d = coefficients.shape[0] - 1, coefficients.shape[1]
    params.update(
        w1=rng.normal(size=(f, width)) / np.sqrt(f),
        b1=np.zeros(width),
        w2=np.zeros((width, d)),
        memory=rng.normal(size=(f, memory)) / np.sqrt(f),
        memory_bias=np.zeros(memory),
    )
    return SequenceModel(KIND, batch.dt_s, context, params, norms, p)


def fit_sequence_model(
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
    command_response=None,
):
    """Adam with gradient clipping and development-rollout checkpoint selection.

    Every checkpoint is evaluated recursively. By default all future channels
    have equal weight after train-only state scaling. error_scale may instead
    supply positive [horizon,channel] loss scales.

    ``command_response`` is the ``(response, held)`` pair
    :func:`initialize_sequence_model` takes. A held channel's affine columns
    are held there for every step of this optimization: they carry no gradient,
    they are restored after every update, and the nonlinear correction and the
    memory keep their own rows of that command and learn around them.
    """
    if steps < 0 or check_every < 1:
        raise ValueError("invalid training steps")
    if batch_size < 1 or not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("batch_size and learning_rate must be positive")
    if train.dt_s != validation.dt_s or any(
        getattr(train, k).shape[1:] != getattr(validation, k).shape[1:]
        for k in ("past_states", "past_inputs", "future_states", "future_inputs")
    ):
        raise ValueError("train and validation contracts differ")
    model = initialize_sequence_model(
        train,
        seed=seed,
        width=width,
        memory=memory,
        ridge=ridge,
        delay_steps=delay_steps,
        command_response=command_response,
    )
    delay = model.delay_steps
    params, norms = jax.tree.map(jnp.asarray, (model.params, model.norms))
    held_rows = (
        held_command_coefficients(
            model.norms,
            *command_response,
            train.future_states.shape[-1],
            train.future_inputs.shape[-1],
            delay,
        )[0]
        if command_response is not None
        else np.zeros(0, dtype=int)
    )
    if held_rows.size:
        free = np.ones_like(np.asarray(model.params["linear"]))
        free[held_rows] = 0.0
        held_mask = jnp.asarray(free)
        held_linear = jnp.asarray(np.asarray(model.params["linear"]) * (1.0 - free))
    else:
        held_mask = held_linear = None
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
    names = ("past_states", "past_inputs", "future_inputs", "future_states")
    training = tuple(jnp.asarray(getattr(train, key)) for key in names)
    development = tuple(jnp.asarray(getattr(validation, key)) for key in names)

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
        if held_mask is not None:
            # A held coefficient carries no gradient at all, so it does not
            # move and does not enter the clipping norm the others share.
            grad = dict(grad, linear=grad["linear"] * held_mask)
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
        if held_mask is not None:
            par = dict(par, linear=par["linear"] * held_mask + held_linear)
        return par, first, second, value

    best = params
    best_loss = float(evaluate(params))
    if not np.isfinite(best_loss):
        raise ValueError("initial recursive validation loss is nonfinite")
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
                raise ValueError(f"nonfinite sequence training at step {i}")
            trace.append(
                dict(
                    step=i,
                    training_batch_mse=train_loss,
                    validation_rollout_mse=val_loss,
                )
            )
            if val_loss < best_loss:
                best, best_loss, best_step = params, val_loss, i
    result = SequenceModel(
        KIND,
        train.dt_s,
        model.history_steps,
        jax.tree.map(np.asarray, best),
        jax.tree.map(np.asarray, norms),
        delay,
    )
    return result, dict(
        kind=KIND,
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
        context_steps=model.history_steps,
        delay_steps=delay,
    )
