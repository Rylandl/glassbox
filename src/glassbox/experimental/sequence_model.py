"""One generic recurrent predictor trained on observed trajectory segments.

Every predicted observation channel advances recursively; future observations
are never required by rollout(). The latent memory is initialized only from
past observations/inputs. Coordinates are treated as Euclidean: this module
does not enforce rotation geometry, infer physical bounds, or calibrate
uncertainty.

Recursion contract: a forecast is a recursion, so the fit bounds its gain. Past
the first horizon step, where the recursion has not run yet, an error the size
of the process's own one-step motion may not come out of the recursion larger
than the process's own motion has grown by that step; a step that leaves it
larger has the paths from the observed state back into the next prediction
scaled down until it does not. The ceiling is hold-current's own error growth on
the training windows, floored at no amplification, and nothing in it reads a
target, a development row or a held-out row.

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

_BISECTIONS = 10
"""Bisections of the recursion factor.

A numerical tolerance on a bracket that is already ``[0, 1]``, not a modelling
choice: the bound itself is the process's own motion growth and has no free
value. The search keeps the lower end of the bracket feasible at every step, so
whatever it returns satisfies the bound whether or not the gain is monotone in
the factor.
"""


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


def recursion_rows(channels, commands, delay_steps, memory):
    """The feature rows the recursion reads back: state, its differences, memory.

    The command rows of :func:`_features` are exogenous and never appear in the
    one-step map's Jacobian with respect to the state, so the gain of the
    recursion is carried by these rows and only by these rows. Scaling them
    scales every path from an observed channel back into the next prediction,
    and at a factor of zero the increment depends on the commands alone, which
    makes the augmented one-step map the plant's own hold-and-shift.
    """
    mask = np.zeros((delay_steps + 1) * (channels + commands) + memory, dtype=bool)
    mask[:channels] = True
    start = channels + commands
    mask[start : start + delay_steps * channels] = True
    mask[-memory:] = True
    return mask


def recursion_ceiling(batch):
    """The error unit and the gain ceiling the process itself names, per step.

    Hold-current -- carrying the last observed state forward -- is the process's
    own free response, so its error at the first horizon step is the size of an
    error the process makes in one step, and the growth of that error over the
    horizon is how fast the process's own motion runs away from an origin. Both
    are read off the training windows and nothing else: the unit is hold's
    per-channel first-step error, and the ceiling at step ``h`` is how much
    larger hold's error is by then, in those units.

    The ceiling never falls below one. A recursion that returns an error
    unchanged is admissible whatever the process does, and the hold-and-shift
    map the factor of zero produces returns exactly that, so flooring the
    ceiling is what makes the bound reachable rather than a value chosen for it.
    """
    horizon = batch.future_states.shape[1]
    hold = np.repeat(batch.past_states[:, -1:], horizon, 1)
    motion = np.sqrt(np.mean((hold - batch.future_states) ** 2, axis=0))
    unit = np.where(motion[0] > 0.0, motion[0], 1.0)
    return unit, np.maximum(np.sqrt(np.mean((motion / unit) ** 2, axis=1)), 1.0)


def _damped(params, rows, factor):
    """The same parameters with every path from the state scaled by ``factor``."""
    mask = jnp.asarray(rows)[:, None]
    return {
        key: (
            jnp.where(mask, value * factor, value)
            if key in ("linear", "w1", "memory")
            else value
        )
        for key, value in params.items()
    }


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
    batch, *, seed=0, width=32, memory=8, ridge=1.0, delay_steps=None
):
    """Initialize from a learned one-step affine model, with zero neural residual.

    The affine fit uses the forecast-phase transitions of every window. The
    window's past holds the whole consumed context and ``delay_steps`` sets the
    explicit-difference history; the memory starts at rest and reads out as
    zero, so checkpoint zero is the affine start.
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
    coefficients = np.linalg.solve(
        design.T @ design + penalty, design.T @ (delta / ds).reshape(len(features), -1)
    )
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
    norms = dict(
        state_mean=xm,
        state_scale=xs,
        input_mean=um,
        input_scale=us,
        feature_scale=fs,
        delta_scale=ds,
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
):
    """Adam with gradient clipping and development-rollout checkpoint selection.

    Every checkpoint is evaluated recursively. By default all future channels
    have equal weight after train-only state scaling. error_scale may instead
    supply positive [horizon,channel] loss scales.

    The fit carries a bound on the recursion's gain, :func:`recursion_ceiling`.
    Past the first horizon step, where the recursion has not run yet, an error
    the size of the process's own one-step motion may not come out of the
    recursion larger than the process's own motion has grown by that step.
    Whenever a step leaves it larger, the paths from the observed state back
    into the next prediction are scaled down, by bisection on every training
    origin, until it does not; the start is bounded the same way before any step
    is taken. The bound reads no target, no development row and no held-out row,
    carries no free value, and leaves the parameters untouched wherever the
    recursion already meets it.
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
    names = ("past_states", "past_inputs", "future_inputs", "future_states")
    training = tuple(jnp.asarray(getattr(train, key)) for key in names)
    development = tuple(jnp.asarray(getattr(validation, key)) for key in names)

    rows = recursion_rows(
        train.past_states.shape[-1],
        train.past_inputs.shape[-1],
        delay,
        memory,
    )
    unit, ceiling = recursion_ceiling(train)
    unit, ceiling = jnp.asarray(unit), jnp.asarray(ceiling)
    recursive = train.future_states.shape[1] > 1
    probe = jnp.asarray(
        np.random.default_rng(seed + 20000).standard_normal(
            (len(train.past_states), train.past_states.shape[-1])
        )
    )

    def loss(par, data):
        x, up, uf, target = data
        prediction = _rollout(par, norms, x, up, uf, delay)
        return jnp.mean(((prediction - target) / normalization) ** 2)

    def excess(par, data, direction):
        """How far the recursion's gain sits above the process's own growth.

        The forecast origin is moved by an error the size of the process's own
        one-step motion; what the recursion returns at each later step, in those
        same units, is compared with how far the process itself has moved by
        then. Nothing here reads a target.
        """
        x, up, uf = data[0], data[1], data[2]
        nominal = _rollout(par, norms, x, up, uf, delay)
        moved = x.at[:, -1].add(direction * unit)
        perturbed = _rollout(par, norms, moved, up, uf, delay)
        deviation = (perturbed - nominal) / unit
        gain = jnp.sqrt(jnp.mean(deviation**2, axis=(0, 2))) / jnp.sqrt(
            jnp.mean(direction**2)
        )
        return jnp.max(gain[1:] - ceiling[1:])

    def bound(par, data, direction):
        """Scale the recursion back until its gain meets the process's own."""
        if not recursive:
            return par, 1.0

        def search(_index, bracket):
            low, high = bracket
            middle = 0.5 * (low + high)
            met = excess(_damped(par, rows, middle), data, direction) <= 0.0
            return jnp.where(met, middle, low), jnp.where(met, high, middle)

        def damp(par):
            low, _ = jax.lax.fori_loop(0, _BISECTIONS, search, (0.0, 1.0))
            return _damped(par, rows, low), low

        return jax.lax.cond(
            excess(par, data, direction) > 0.0, damp, lambda par: (par, 1.0), par
        )

    @jax.jit
    def evaluate(par):
        return loss(par, development)

    @jax.jit
    def start(par):
        return bound(par, training, probe)

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
        par, factor = bound(par, training, probe)
        return par, first, second, value, factor

    params, factor = start(params)
    factors = [float(factor)]
    best = params
    best_loss = float(evaluate(params))
    if not np.isfinite(best_loss):
        raise ValueError("initial recursive validation loss is nonfinite")
    trace = [
        dict(step=0, validation_rollout_mse=best_loss, recursion_factor=factors[0])
    ]
    first, second = (jax.tree.map(jnp.zeros_like, params) for _ in range(2))
    rng = np.random.default_rng(seed + 10000)
    best_step = 0
    for i in range(1, steps + 1):
        indices = rng.integers(
            len(train.past_states), size=min(batch_size, len(train.past_states))
        )
        params, first, second, value, factor = update(params, first, second, i, indices)
        factors.append(float(factor))
        if i % check_every == 0 or i == steps:
            train_loss, val_loss = float(value), float(evaluate(params))
            if not np.isfinite([train_loss, val_loss]).all():
                raise ValueError(f"nonfinite sequence training at step {i}")
            trace.append(
                dict(
                    step=i,
                    training_batch_mse=train_loss,
                    validation_rollout_mse=val_loss,
                    recursion_factor=factors[-1],
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
    bounded = np.asarray(factors) < 1.0
    remaining = float(np.asarray(excess(best, training, probe))) if recursive else 0.0
    return result, dict(
        kind=KIND,
        steps=steps,
        selected_step=best_step,
        validation_rollout_mse=best_loss,
        seed=seed,
        trace=trace,
        recursion_bound=dict(
            rule="past the first horizon step, an error the size of the process's own one-step motion may not come out of the recursion larger than the process's own motion has grown by that step",
            applies=bool(recursive),
            bisections=_BISECTIONS,
            bounded_steps=int(bounded.sum()),
            smallest_factor=float(np.min(factors)),
            final_factor=factors[-1],
            ceiling=np.asarray(ceiling).tolist(),
            selected_gain_excess=remaining,
        ),
        parameter_count=sum(v.size for v in result.params.values()),
        loss_scale_mode="state_standard_deviation"
        if error_scale is None
        else "explicit_horizon_channel",
        error_scale=np.asarray(normalization).tolist(),
        context_steps=model.history_steps,
        delay_steps=delay,
    )
