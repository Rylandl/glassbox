"""One maintained quadratic recurrent dynamics mean.

The causal memory consumes observed transitions within a recording and advances
with predicted observations during recursive forecasting. Fitting uses an affine
centered quadratic initialization, fixed initial training channel weights, and
ordered full-cache Adam with bounded strict-decrease acceptance. The public
learner owns the float64 fitting scope; inference follows ordinary ambient JAX.
No controller, simulator, research module or runtime model choice is involved.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from ._learner_arrays import array_fingerprint, load_arrays, save_arrays

KIND = "filter_mlp"
_MODEL_FORMAT = "glassbox-sequence-v2"
RECIPE_ID = "generic-memory-v4-prototype"
OBJECTIVE = "fixed_initial_training_channel_balance"
HOLD_SCALE_FLOOR = 0.01
SCALES = tuple(2.0**-index for index in range(8))
FIT_WALL_TIME_LIMIT_S = 7200
_ARRAYS = ("past_states", "past_inputs", "future_inputs", "future_states")
_PARAMETERS = frozenset(
    (
        "linear",
        "interaction",
        "autonomous",
        "bias",
        "w1",
        "b1",
        "w2",
        "memory",
        "memory_bias",
    )
)
_NORMALIZATIONS = frozenset(
    (
        "state_mean",
        "state_scale",
        "input_mean",
        "input_scale",
        "feature_scale",
        "delta_scale",
        "interaction_scale",
        "autonomous_scale",
    )
)


class SequenceFitError(ValueError):
    """A nonfinite or time-limited fit, rather than an admitted revision."""


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


def interaction_features(x, u, xp=jnp):
    """State-major/input-minor products, preserving every leading dimension."""
    return (x[..., :, None] * u[..., None, :]).reshape(
        (*x.shape[:-1], x.shape[-1] * u.shape[-1])
    )


def autonomous_features(x, xp=jnp):
    """Uncentered x_i*x_j, i<=j, in lexicographic order, including squares once."""
    i, j = xp.triu_indices(x.shape[-1])
    return x[..., i] * x[..., j]


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
        a = autonomous_features(current) / norms["autonomous_scale"]
        delta = z @ params["linear"] + q @ params["interaction"]
        delta = delta + a @ params["autonomous"] + params["bias"]
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
            or not isinstance(self.dt_s, (int, float, np.integer, np.floating))
            or isinstance(self.dt_s, (bool, np.bool_))
            or not np.isfinite(self.dt_s)
            or self.dt_s <= 0
            or not isinstance(self.history_steps, (int, np.integer))
            or isinstance(self.history_steps, (bool, np.bool_))
            or not isinstance(self.delay_steps, (int, np.integer))
            or isinstance(self.delay_steps, (bool, np.bool_))
            or not 1 <= self.delay_steps < self.history_steps
        ):
            raise ValueError(
                "sequence timing requires positive dt and integer 1 <= delay < context"
            )
        if (
            not isinstance(self.params, dict)
            or set(self.params) != _PARAMETERS
            or not isinstance(self.norms, dict)
            or set(self.norms) != _NORMALIZATIONS
        ):
            raise ValueError(
                "sequence parameters and normalizations have an unsupported roster"
            )
        params = {key: np.asarray(value) for key, value in self.params.items()}
        norms = {key: np.asarray(value) for key, value in self.norms.items()}
        if any(
            value.dtype != np.dtype("float64") or not np.isfinite(value).all()
            for value in (*params.values(), *norms.values())
        ):
            raise ValueError(
                "sequence parameters and normalizations must be finite float64 arrays"
            )
        if any(norms[key].ndim != 1 or not len(norms[key]) for key in _NORMALIZATIONS):
            raise ValueError("sequence normalizations must be nonempty vectors")
        if any(
            params[key].ndim != 1 or not len(params[key])
            for key in ("b1", "memory_bias")
        ):
            raise ValueError("sequence hidden biases must be nonempty vectors")
        d, m = len(norms["state_mean"]), len(norms["input_mean"])
        width, memory = len(params["b1"]), len(params["memory_bias"])
        features = (int(self.delay_steps) + 1) * (d + m) + memory
        interaction, autonomous = d * m, d * (d + 1) // 2
        shapes = dict(
            linear=(features, d),
            interaction=(interaction, d),
            autonomous=(autonomous, d),
            bias=(d,),
            w1=(features, width),
            b1=(width,),
            w2=(width, d),
            memory=(features, memory),
            memory_bias=(memory,),
        )
        norm_shapes = dict(
            state_mean=(d,),
            state_scale=(d,),
            input_mean=(m,),
            input_scale=(m,),
            feature_scale=(features,),
            delta_scale=(d,),
            interaction_scale=(interaction,),
            autonomous_scale=(autonomous,),
        )
        if any(params[key].shape != shape for key, shape in shapes.items()) or any(
            norms[key].shape != shape for key, shape in norm_shapes.items()
        ):
            raise ValueError(
                "sequence arrays do not match their state/input/hidden dimensions"
            )
        if any(
            np.any(value <= 0) for key, value in norms.items() if key.endswith("_scale")
        ):
            raise ValueError("sequence normalization scales must be positive")
        # Own host float64 arrays: inference may safely convert them under either
        # ambient JAX precision, independently of the fitting scope.
        object.__setattr__(
            self, "params", {k: np.array(v, copy=True) for k, v in params.items()}
        )
        object.__setattr__(
            self, "norms", {k: np.array(v, copy=True) for k, v in norms.items()}
        )
        object.__setattr__(self, "dt_s", float(self.dt_s))
        object.__setattr__(self, "history_steps", int(self.history_steps))
        object.__setattr__(self, "delay_steps", int(self.delay_steps))

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
            format=_MODEL_FORMAT,
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
        if (
            not isinstance(meta, dict)
            or set(meta) != {"format", "kind", "dt_s", "history_steps", "delay_steps"}
            or meta.get("format") != _MODEL_FORMAT
        ):
            raise ValueError("unsupported sequence format or metadata")
        expected = {"param_" + key for key in _PARAMETERS} | {
            "norm_" + key for key in _NORMALIZATIONS
        }
        if set(arrays) != expected:
            raise ValueError("unsupported sequence array roster")
        meta = {key: value for key, value in meta.items() if key != "format"}
        return cls(
            **meta,
            params={
                key[6:]: value
                for key, value in arrays.items()
                if key.startswith("param_")
            },
            norms={
                key[5:]: value
                for key, value in arrays.items()
                if key.startswith("norm_")
            },
        )


def _initialize_affine(
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
    return params, norms, context, p


def initialize_sequence_model(
    batch, *, seed=0, width=32, memory=8, ridge=1.0, delay_steps=None
):
    """One existing affine precursor, then the same full joint solve centered on it."""
    affine_params, affine_norms, context, p = _initialize_affine(
        batch,
        seed=seed,
        width=width,
        memory=memory,
        ridge=ridge,
        delay_steps=delay_steps,
    )
    params = {k: np.array(v, copy=True) for k, v in affine_params.items()}
    norms = {k: np.array(v, copy=True) for k, v in affine_norms.items()}
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
            affine_params["linear"],
            np.zeros((products.shape[-1] + autonomous.shape[-1], target.shape[-1])),
            affine_params["bias"][None, :],
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
    return SequenceModel(KIND, batch.dt_s, context, params, norms, p)


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


def gradient_metadata(
    training_windows,
    *,
    attempts_started,
    gradient_proposal_calls_returned,
    completed_acceptance_attempts,
):
    """Describe only returned gradient work; unfinished calls remain unknown."""
    counts = (
        completed_acceptance_attempts,
        gradient_proposal_calls_returned,
        attempts_started,
    )
    if (
        type(training_windows) is not int
        or training_windows < 1
        or any(type(value) is not int or value < 0 for value in counts)
        or not counts[0] <= counts[1] <= counts[2]
        or counts[2] - counts[0] > 1
    ):
        raise ValueError("invalid actual full-cache gradient accounting")
    return dict(
        policy="ordered_full_cache",
        loss_scope="full_training_cache_pre_proposal",
        training_windows=training_windows,
        index_dtype="int64",
        sampling_seed=None,
        attempts_started=attempts_started,
        gradient_proposal_calls_returned=gradient_proposal_calls_returned,
        completed_acceptance_attempts=completed_acceptance_attempts,
        known_gradient_window_visits=training_windows
        * gradient_proposal_calls_returned,
        incomplete_gradient_work_unknown=attempts_started
        > gradient_proposal_calls_returned,
    )


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
        id=RECIPE_ID,
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


def fit_sequence_model(
    train,
    validation,
    *,
    seed=0,
    steps=1000,
    batch_size=None,
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
    if batch_size is None:
        batch_size = len(train.past_states)
    if (
        type(batch_size) is not int
        or batch_size != len(train.past_states)
        or batch_size < 1
    ):
        raise ValueError("batch_size must equal the complete positive training cache")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if train.dt_s != validation.dt_s or any(
        getattr(train, k).shape[1:] != getattr(validation, k).shape[1:] for k in _ARRAYS
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
    _observe_attempt(dict(phase="initialized", model=model))
    delay = model.delay_steps
    params, norms = jax.tree.map(jnp.asarray, (model.params, model.norms))
    if error_scale is None:
        hold = np.repeat(train.past_states[:, -1:], train.future_states.shape[1], 1)
        error_scale = np.maximum(
            np.sqrt(np.mean((hold - train.future_states) ** 2, 0)),
            HOLD_SCALE_FLOOR * model.norms["state_scale"],
        )
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
        raise SequenceFitError("nonfinite initial training forecast")
    weight_floor = HOLD_SCALE_FLOOR**2
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
        raise SequenceFitError("nonfinite initial channel weighting")
    for values in (
        initial_training_prediction,
        initial_channel_mse,
        raw_channel_weights,
        channel_weights,
    ):
        values.setflags(write=False)
    fixed_weights = jax.lax.stop_gradient(jnp.asarray(channel_weights))
    _observe_attempt(
        dict(
            phase="weights",
            model=model,
            params=params,
            norms=norms,
            normalization=normalization,
            initial_training_prediction=initial_training_prediction,
            initial_channel_mse=initial_channel_mse,
            raw_channel_weights=raw_channel_weights,
            channel_weights=channel_weights,
            fixed_weights=fixed_weights,
            weight_floor=weight_floor,
            weight_normalizer=weight_normalizer,
        )
    )

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
        raise SequenceFitError("initial recursive validation loss is nonfinite")
    trace = [dict(step=0, validation_rollout_mse=best_loss)]
    unweighted_development_trace = [
        dict(step=0, validation_rollout_mse=unweighted_best_initial_loss)
    ]
    _observe_attempt(
        dict(
            phase="checkpoint",
            step=0,
            params=params,
            norms=norms,
            validation_rollout_mse=best_loss,
            unweighted_validation_rollout_mse=unweighted_best_initial_loss,
        )
    )
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
        raise SequenceFitError("nonfinite initial full-training acceptance loss")
    initial_full_training_loss = current_full_training_loss
    checkpoint_losses = [dict(step=0, full_training_loss=current_full_training_loss)]
    accepted_scales = []
    attempts_started = 0
    gradient_proposal_calls_returned = 0
    completed_acceptance_attempts = 0
    best_step = 0
    for i in range(1, steps + 1):
        if time.perf_counter() - fit_started_at > FIT_WALL_TIME_LIMIT_S:
            raise SequenceFitError("fit_time_limit")
        indices = np.arange(len(train.past_states), dtype=np.int64)
        attempts_started += 1
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
        gradient_proposal_calls_returned += 1
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
            raise SequenceFitError(f"nonfinite Adam proposal at attempt {i}")
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
        completed_acceptance_attempts += 1
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
            raise SequenceFitError("fit_time_limit")
        if i % check_every == 0 or i == steps:
            train_loss = minibatch_loss
            val_loss, unweighted_val_loss = map(float, evaluate(params))
            if not np.isfinite([train_loss, val_loss, unweighted_val_loss]).all():
                raise SequenceFitError(f"nonfinite sequence training at step {i}")
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
            _observe_attempt(
                dict(
                    phase="checkpoint",
                    step=i,
                    params=params,
                    norms=norms,
                    validation_rollout_mse=val_loss,
                    unweighted_validation_rollout_mse=unweighted_val_loss,
                    full_training_loss=current_full_training_loss,
                    selected_step=best_step,
                )
            )
    if time.perf_counter() - fit_started_at > FIT_WALL_TIME_LIMIT_S:
        raise SequenceFitError("fit_time_limit")
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
        mechanism=RECIPE_ID,
        ridge=float(ridge),
        batch_size=batch_size,
        steps=steps,
        selected_step=best_step,
        validation_rollout_mse=best_loss,
        seed=seed,
        minibatch_seed=None,
        gradient_sampling="ordered_full_cache",
        gradient=gradient_metadata(
            len(train.past_states),
            attempts_started=attempts_started,
            gradient_proposal_calls_returned=gradient_proposal_calls_returned,
            completed_acceptance_attempts=completed_acceptance_attempts,
        ),
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
