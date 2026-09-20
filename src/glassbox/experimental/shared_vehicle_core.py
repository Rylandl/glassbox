"""One research rigid-body learner, with configuration learned from recordings.

Canonical observations are world NWU velocity, body FLU angular velocity and
row-major body-to-world rotation (15 values). Commands have no actuator roles.
Only gravity and rigid-body kinematics are prescribed. No vehicle-family code
or simulator parameters enter this module.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from glassbox._learner_arrays import array_fingerprint
from glassbox._sequence_model import (
    SequenceBatch,
    SequenceFitError,
    gradient_metadata,
    safeguard_metadata,
    trial_parameters,
)

FORMAT = "glassbox-shared-vehicle-sequence-v1"
RECIPE_ID = "shared-vehicle-physics-v1"
GRAVITY = (0.0, 0.0, -9.80665)
MAX_SUBSTEP_S = 0.025
STEPS = 1000
CHECK_EVERY = 100
FIT_WALL_TIME_LIMIT_S = 7200
SCALES = tuple(2.0**-i for i in range(8))
_ARRAYS = ("past_states", "past_inputs", "future_inputs", "future_states")
_PARAMETERS = {
    "linear",
    "quadratic",
    "bias",
    "w1",
    "b1",
    "w2",
    "memory",
    "memory_bias",
    "raw_tau",
}
_NORMS = {
    "body_mean",
    "body_scale",
    "input_mean",
    "input_scale",
    "feature_scale",
    "quadratic_scale",
    "output_scale",
    "state_mean",
    "state_scale",
}


def rotation_exp(vector):
    """Batched SO(3) exponential with finite first/second derivatives at zero."""
    vector = jnp.asarray(vector)
    squared = jnp.sum(vector * vector, axis=-1)
    small = squared < 1e-6
    # Never differentiate sqrt at zero, even in an unselected branch.
    angle = jnp.sqrt(jnp.where(small, jnp.ones_like(squared), squared))
    a = jnp.where(
        small,
        1 - squared / 6 + squared**2 / 120 - squared**3 / 5040,
        jnp.sin(angle) / angle,
    )
    b = jnp.where(
        small,
        0.5 - squared / 24 + squared**2 / 720 - squared**3 / 40320,
        (1 - jnp.cos(angle)) / (angle * angle),
    )
    x, y, z = (vector[..., i] for i in range(3))
    zero = jnp.zeros_like(x)
    cross = jnp.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1)
    cross = cross.reshape((*vector.shape[:-1], 3, 3))
    return (
        jnp.eye(3, dtype=vector.dtype)
        + a[..., None, None] * cross
        + b[..., None, None] * (cross @ cross)
    )


def time_constants(params):
    return 0.001 + jax.nn.softplus(params["raw_tau"])


def _body_features(states, xp=jnp):
    rotation = states[..., 6:].reshape((*states.shape[:-1], 3, 3))
    velocity = xp.einsum("...ji,...j->...i", rotation, states[..., :3])
    gravity = xp.einsum(
        "...ji,j->...i", rotation, xp.asarray(GRAVITY, dtype=states.dtype)
    ) / abs(GRAVITY[2])
    return xp.concatenate((velocity, states[..., 3:6], gravity), axis=-1)


def current_features(states, commands, filtered, norms, xp=jnp):
    return xp.concatenate(
        (
            (_body_features(states, xp) - norms["body_mean"]) / norms["body_scale"],
            (commands - norms["input_mean"]) / norms["input_scale"],
            (filtered - norms["input_mean"]) / norms["input_scale"],
        ),
        axis=-1,
    )


def sampled_features(current, past, hidden, xp=jnp):
    return xp.concatenate(
        (
            current,
            (past - current[..., None, :]).reshape((*current.shape[:-1], -1)),
            hidden,
        ),
        axis=-1,
    )


def quadratic_features(current, xp=jnp):
    i, j = xp.triu_indices(current.shape[-1])
    return current[..., i] * current[..., j]


def _head(params, norms, states, commands, filtered, history, hidden):
    b = current_features(states, commands, filtered, norms)
    z = sampled_features(b, history, hidden) / norms["feature_scale"]
    q = quadratic_features(b) / norms["quadratic_scale"]
    acceleration = z @ params["linear"] + q @ params["quadratic"] + params["bias"]
    acceleration += jnp.tanh(z @ params["w1"] + params["b1"]) @ params["w2"]
    return acceleration * norms["output_scale"], b, z


def physical_step(params, norms, states, commands, filtered, history, hidden, dt_s):
    """One observation interval; history and hidden memory do not substep."""
    count = max(1, math.ceil(dt_s / MAX_SUBSTEP_S))
    duration = dt_s / count
    tau = time_constants(params)
    gravity = jnp.asarray(GRAVITY, dtype=states.dtype)
    _, start_features, start_z = _head(
        params, norms, states, commands, filtered, history, hidden
    )

    def substep(_, carry):
        state, applied = carry
        v, omega = state[..., :3], state[..., 3:6]
        rotation = state[..., 6:].reshape((*state.shape[:-1], 3, 3))
        first, _, _ = _head(params, norms, state, commands, applied, history, hidden)
        world_first = gravity + jnp.einsum("...ij,...j->...i", rotation, first[..., :3])
        rotation_half = rotation @ rotation_exp(0.5 * duration * omega)
        velocity_half = v + 0.5 * duration * world_first
        omega_half = omega + 0.5 * duration * first[..., 3:]
        applied_half = commands + (applied - commands) * jnp.exp(-0.5 * duration / tau)
        state_half = jnp.concatenate(
            (velocity_half, omega_half, rotation_half.reshape((*state.shape[:-1], 9))),
            -1,
        )
        middle, _, _ = _head(
            params, norms, state_half, commands, applied_half, history, hidden
        )
        velocity_next = v + duration * (
            gravity + jnp.einsum("...ij,...j->...i", rotation_half, middle[..., :3])
        )
        omega_next = omega + duration * middle[..., 3:]
        rotation_next = rotation @ rotation_exp(duration * omega_half)
        state_next = jnp.concatenate(
            (velocity_next, omega_next, rotation_next.reshape((*state.shape[:-1], 9))),
            -1,
        )
        applied_next = commands + (applied - commands) * jnp.exp(-duration / tau)
        return state_next, applied_next

    state, applied = jax.lax.fori_loop(0, count, substep, (states, filtered))
    hidden_next = jnp.tanh(start_z @ params["memory"] + params["memory_bias"])
    history_next = jnp.concatenate((history[:, 1:], start_features[:, None]), axis=1)
    return state, applied, history_next, hidden_next


def _history(params, norms, past, past_inputs, delay, dt_s):
    tau = time_constants(params)

    def filter_step(applied, command):
        return command + (applied - command) * jnp.exp(-dt_s / tau), applied

    applied, preceding = jax.lax.scan(
        filter_step, past_inputs[:, 0], past_inputs.swapaxes(0, 1)
    )
    b = current_features(past[:, :-1], past_inputs, preceding.swapaxes(0, 1), norms)
    hidden = jnp.zeros((len(past), params["memory"].shape[1]), dtype=past.dtype)
    context = past_inputs.shape[1]
    histories = jnp.stack([b[:, j - delay : j] for j in range(delay, context)])

    def remember(memory, values):
        current, previous = values
        z = sampled_features(current, previous, memory) / norms["feature_scale"]
        return jnp.tanh(z @ params["memory"] + params["memory_bias"]), None

    hidden, _ = jax.lax.scan(remember, hidden, (b[:, delay:].swapaxes(0, 1), histories))
    return applied, b[:, -delay:], hidden


def _rollout(params, norms, past, past_inputs, future_inputs, delay, dt_s):
    applied, history, hidden = _history(params, norms, past, past_inputs, delay, dt_s)

    def advance(carry, command):
        state, applied, history, hidden = carry
        result = physical_step(
            params, norms, state, command, applied, history, hidden, dt_s
        )
        return result, result[0]

    _, prediction = jax.lax.scan(
        advance, (past[:, -1], applied, history, hidden), future_inputs.swapaxes(0, 1)
    )
    return prediction.swapaxes(0, 1)


@dataclass(frozen=True)
class VehicleSequenceModel:
    dt_s: float
    history_steps: int
    delay_steps: int
    params: dict
    norms: dict

    def __post_init__(self):
        if (
            isinstance(self.dt_s, bool)
            or not isinstance(self.dt_s, (int, float, np.number))
            or not np.isfinite(self.dt_s)
            or self.dt_s <= 0
            or type(self.history_steps) is not int
            or type(self.delay_steps) is not int
            or not 1 <= self.delay_steps < self.history_steps
        ):
            raise ValueError("invalid vehicle timing/history")
        if set(self.params) != _PARAMETERS or set(self.norms) != _NORMS:
            raise ValueError("invalid vehicle parameter/normalization keys")
        p, n = {}, {}
        for source, destination in ((self.params, p), (self.norms, n)):
            for key, value in source.items():
                array = np.asarray(value)
                if array.dtype != np.dtype("float64") or not np.isfinite(array).all():
                    raise ValueError("vehicle arrays must be finite float64")
                destination[key] = np.array(array, copy=True)
                destination[key].setflags(write=False)
        if (
            n["input_mean"].ndim != 1
            or not n["input_mean"].size
            or p["b1"].ndim != 1
            or p["memory_bias"].ndim != 1
        ):
            raise ValueError("invalid vehicle dimensions")
        m, width, memory = len(n["input_mean"]), len(p["b1"]), len(p["memory_bias"])
        if min(width, memory) < 1:
            raise ValueError("invalid vehicle hidden dimensions")
        current = 9 + 2 * m
        feature = (self.delay_steps + 1) * current + memory
        quadratic = current * (current + 1) // 2
        ps = dict(
            linear=(feature, 6),
            quadratic=(quadratic, 6),
            bias=(6,),
            w1=(feature, width),
            b1=(width,),
            w2=(width, 6),
            memory=(feature, memory),
            memory_bias=(memory,),
            raw_tau=(m,),
        )
        ns = dict(
            body_mean=(9,),
            body_scale=(9,),
            input_mean=(m,),
            input_scale=(m,),
            feature_scale=(feature,),
            quadratic_scale=(quadratic,),
            output_scale=(6,),
            state_mean=(15,),
            state_scale=(15,),
        )
        if any(p[k].shape != shape for k, shape in ps.items()) or any(
            n[k].shape != shape for k, shape in ns.items()
        ):
            raise ValueError("vehicle arrays do not match dimensions")
        if any(np.any(v <= 0) for k, v in n.items() if k.endswith("_scale")):
            raise ValueError("vehicle scales must be positive")
        object.__setattr__(self, "params", p)
        object.__setattr__(self, "norms", n)
        object.__setattr__(self, "dt_s", float(self.dt_s))

    def rollout(self, past_states, past_inputs, future_inputs):
        x, up, uf = map(jnp.asarray, (past_states, past_inputs, future_inputs))
        single = x.ndim == 2
        if single:
            x, up, uf = x[None], up[None], uf[None]
        m = len(self.norms["input_mean"])
        if (
            x.ndim != 3
            or up.ndim != 3
            or uf.ndim != 3
            or x.shape[1:] != (self.history_steps + 1, 15)
            or up.shape != (len(x), self.history_steps, m)
            or uf.shape[0] != len(x)
            or uf.shape[-1] != m
            or uf.shape[1] < 1
        ):
            raise ValueError("vehicle rollout shapes/history do not match")
        dtype = jnp.result_type(self.norms["state_scale"])
        params = {k: jnp.asarray(v, dtype=dtype) for k, v in self.params.items()}
        norms = {k: jnp.asarray(v, dtype=dtype) for k, v in self.norms.items()}
        x, up, uf = (jnp.asarray(v, dtype=dtype) for v in (x, up, uf))
        prediction = _rollout(params, norms, x, up, uf, self.delay_steps, self.dt_s)
        return prediction[0] if single else prediction

    def arrays(self):
        return {
            **{f"param_{k}": v.copy() for k, v in self.params.items()},
            **{f"norm_{k}": v.copy() for k, v in self.norms.items()},
        }

    def metadata(self):
        return dict(
            format=FORMAT,
            dt_s=self.dt_s,
            history_steps=self.history_steps,
            delay_steps=self.delay_steps,
        )

    @property
    def fingerprint(self):
        return array_fingerprint(self.metadata(), self.arrays())

    @classmethod
    def from_arrays(cls, metadata, arrays):
        if (
            set(metadata) != {"format", "dt_s", "history_steps", "delay_steps"}
            or metadata["format"] != FORMAT
            or set(arrays)
            != {*("param_" + k for k in _PARAMETERS), *("norm_" + k for k in _NORMS)}
        ):
            raise ValueError("invalid vehicle core archive")
        return cls(
            metadata["dt_s"],
            metadata["history_steps"],
            metadata["delay_steps"],
            {k: arrays["param_" + k] for k in _PARAMETERS},
            {k: arrays["norm_" + k] for k in _NORMS},
        )


def _validate_batch(batch):
    if not isinstance(batch, SequenceBatch) or batch.past_states.shape[-1] != 15:
        raise ValueError("vehicle fitting requires canonical15 SequenceBatch")
    for states in (batch.past_states, batch.future_states):
        rotation = states[..., 6:].reshape(-1, 3, 3)
        if not np.allclose(
            rotation.swapaxes(-1, -2) @ rotation, np.eye(3), rtol=0, atol=1e-5
        ) or not np.allclose(np.linalg.det(rotation), 1, rtol=0, atol=1e-5):
            raise ValueError("observations must contain proper body-to-world rotations")


def initialize(
    train,
    *,
    width=32,
    memory=8,
    ridge_fraction=0.01,
    seed=0,
    delay_steps=None,
    observer=None,
):
    """Two training-only ridge solves; secants are not simulator derivatives."""
    _validate_batch(train)
    if observer is not None:
        observer(
            dict(phase="initializer_started", training_windows=len(train.past_states))
        )
    if (
        type(width) is not int
        or type(memory) is not int
        or min(width, memory) < 1
        or not np.isfinite(ridge_fraction)
        or ridge_fraction <= 0
    ):
        raise ValueError("invalid vehicle initialization recipe")
    context = train.past_inputs.shape[1]
    delay = max(1, round(0.1 / train.dt_s)) if delay_steps is None else delay_steps
    if type(delay) is not int or not 1 <= delay < context:
        raise ValueError("vehicle initialization needs delay shorter than history")
    n, horizon, m = train.future_inputs.shape
    xall = np.concatenate((train.past_states, train.future_states[:, :-1]), axis=1)
    uall = np.concatenate((train.past_inputs, train.future_inputs), axis=1)
    current = xall[:, context:]
    body = _body_features(current, np)
    norms = dict(
        body_mean=body.mean((0, 1)),
        body_scale=np.maximum(body.std((0, 1)), 1e-4),
        input_mean=train.future_inputs.mean((0, 1)),
        input_scale=np.maximum(train.future_inputs.std((0, 1)), 1e-4),
        state_mean=current.mean((0, 1)),
        state_scale=np.maximum(current.std((0, 1)), 1e-4),
    )
    applied = uall[:, 0].copy()
    filtered = []
    for command in uall.swapaxes(0, 1):
        filtered.append(applied.copy())
        applied = command + (applied - command) * np.exp(-train.dt_s / 0.05)
    b = current_features(xall, uall, np.stack(filtered, 1), norms, np)
    z = np.stack(
        [
            sampled_features(
                b[:, context + t],
                b[:, context + t - delay : context + t],
                np.zeros((n, memory)),
                np,
            )
            for t in range(horizon)
        ],
        1,
    )
    q = quadratic_features(b[:, context:], np)
    rotation = current[..., 6:].reshape(n, horizon, 3, 3)
    force = np.einsum(
        "...ji,...j->...i",
        rotation,
        (train.future_states[..., :3] - current[..., :3]) / train.dt_s
        - np.asarray(GRAVITY),
    )
    angular = (train.future_states[..., 3:6] - current[..., 3:6]) / train.dt_s
    target = np.concatenate((force, angular), -1)
    norms["output_scale"] = np.maximum(target.std((0, 1)), 1e-4)
    norms["feature_scale"] = np.where(z.std((0, 1)) > 1e-8, z.std((0, 1)), 1.0)
    norms["quadratic_scale"] = np.where(q.std((0, 1)) > 1e-8, q.std((0, 1)), 1.0)
    target = (target / norms["output_scale"]).reshape(n * horizon, 6)
    linear = (z / norms["feature_scale"]).reshape(n * horizon, -1)
    quadratic = (q / norms["quadratic_scale"]).reshape(n * horizon, -1)
    design_affine = np.column_stack((linear, np.ones(n * horizon)))
    ridge = ridge_fraction * n * horizon
    penalty_affine = np.diag(np.r_[np.full(linear.shape[1], ridge), 0.0])
    affine = np.linalg.solve(
        design_affine.T @ design_affine + penalty_affine, design_affine.T @ target
    )
    if observer is not None:
        observer(
            dict(
                phase="ridge_solve", ordinal=1, system_shape=list(penalty_affine.shape)
            )
        )
    design = np.column_stack((linear, quadratic, np.ones(n * horizon)))
    anchor = np.concatenate(
        (affine[:-1], np.zeros((quadratic.shape[1], 6)), affine[-1:])
    )
    penalty = np.diag(np.r_[np.full(design.shape[1] - 1, ridge), 0.0])
    coefficients = np.linalg.solve(
        design.T @ design + penalty, design.T @ target + penalty @ anchor
    )
    if observer is not None:
        observer(dict(phase="ridge_solve", ordinal=2, system_shape=list(penalty.shape)))
    f = linear.shape[1]
    rng = np.random.default_rng(seed)
    params = dict(
        linear=coefficients[:f],
        quadratic=coefficients[f:-1],
        bias=coefficients[-1],
        w1=rng.normal(size=(f, width)) / np.sqrt(f),
        b1=np.zeros(width),
        w2=np.zeros((width, 6)),
        memory=rng.normal(size=(f, memory)) / np.sqrt(f),
        memory_bias=np.zeros(memory),
        raw_tau=np.full(m, np.log(np.expm1(0.049))),
    )
    return VehicleSequenceModel(train.dt_s, context, delay, params, norms)


def fit_sequence(
    train,
    development,
    *,
    error_scale=None,
    channel_weights=None,
    observer=None,
    _steps=STEPS,
    _check_every=CHECK_EVERY,
):
    """Fixed scientific fit. Private step overrides exist only for analytic tests."""
    with jax.enable_x64(True):
        return _fit_sequence(
            train,
            development,
            error_scale=error_scale,
            channel_weights=channel_weights,
            observer=observer,
            steps=_steps,
            check_every=_check_every,
        )


def _fit_sequence(
    train, development, *, error_scale, channel_weights, observer, steps, check_every
):
    started = time.perf_counter()
    if (
        type(steps) is not int
        or steps < 0
        or type(check_every) is not int
        or check_every < 1
    ):
        raise ValueError("invalid private fit budget")
    _validate_batch(train)
    _validate_batch(development)
    if train.dt_s != development.dt_s or any(
        getattr(train, k).shape[1:] != getattr(development, k).shape[1:]
        for k in _ARRAYS
    ):
        raise ValueError("vehicle training/development contracts differ")

    def emit(phase, **values):
        if observer is not None:
            observer(dict(phase=phase, **values))

    def elapsed_check():
        if time.perf_counter() - started > FIT_WALL_TIME_LIMIT_S:
            raise SequenceFitError("fit_time_limit")

    initial = initialize(train, observer=observer)
    emit("initialized", model=initial)
    params, norms = jax.tree.map(jnp.asarray, (initial.params, initial.norms))
    training = tuple(jnp.asarray(getattr(train, k)) for k in _ARRAYS)
    validation = tuple(jnp.asarray(getattr(development, k)) for k in _ARRAYS)
    if error_scale is None:
        error_scale = np.maximum(
            np.sqrt(np.mean((train.past_states[:, -1:] - train.future_states) ** 2, 0)),
            0.01 * initial.norms["state_scale"],
        )
    scale = np.asarray(error_scale, dtype=np.float64)
    if (
        scale.shape != train.future_states.shape[1:]
        or not np.isfinite(scale).all()
        or np.any(scale <= 0)
    ):
        raise ValueError("vehicle error_scale must be positive horizon by15")
    imported_weights = channel_weights is not None
    initial_prediction = None
    if channel_weights is None:
        initial_prediction = np.asarray(initial.rollout(*training[:3]))
        if not np.isfinite(initial_prediction).all():
            raise SequenceFitError("nonfinite initial forecast")
        mse = np.mean(((initial_prediction - train.future_states) / scale) ** 2, (0, 1))
        raw = 1 / np.maximum(mse, 0.0001)
        channel_weights = raw / raw.mean()
    weights = np.asarray(channel_weights, dtype=np.float64)
    if weights.shape != (15,) or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("vehicle channel_weights must be positive15")
    normalization = jnp.asarray(scale)
    fixed_weights = jax.lax.stop_gradient(jnp.asarray(weights))
    emit(
        "weights",
        params=params,
        norms=norms,
        normalization=normalization,
        channel_weights=weights,
        fixed_weights=fixed_weights,
        initial_training_prediction=initial_prediction,
        imported_weights=imported_weights,
    )

    def loss_components(par, data):
        prediction = _rollout(par, norms, *data[:3], initial.delay_steps, train.dt_s)
        squared = ((prediction - data[3]) / normalization) ** 2
        return jnp.mean(fixed_weights * squared), jnp.mean(squared)

    full_objective = jax.jit(lambda par: loss_components(par, training)[0])
    evaluate = jax.jit(lambda par: loss_components(par, validation))

    @jax.jit
    def update(par, first, second, index):
        value, grad = jax.value_and_grad(lambda p: loss_components(p, training)[0])(par)
        finite = jnp.all(
            jnp.stack([jnp.all(jnp.isfinite(g)) for g in jax.tree.leaves(grad)])
        )
        norm = jnp.sqrt(sum(jnp.sum(g * g) for g in jax.tree.leaves(grad)))
        grad = jax.tree.map(lambda g: g * jnp.minimum(1.0, 5.0 / (norm + 1e-12)), grad)
        first = jax.tree.map(lambda a, g: 0.9 * a + 0.1 * g, first, grad)
        second = jax.tree.map(lambda a, g: 0.999 * a + 0.001 * g * g, second, grad)
        proposal = jax.tree.map(
            lambda w, a, b: (
                w
                - 0.002
                * (a / (1 - 0.9**index))
                / (jnp.sqrt(b / (1 - 0.999**index)) + 1e-8)
            ),
            par,
            first,
            second,
        )
        return proposal, first, second, value, norm, grad, finite

    best = params
    best_loss, initial_unweighted = map(float, evaluate(params))
    if not np.isfinite([best_loss, initial_unweighted]).all():
        raise SequenceFitError("nonfinite initial development loss")
    initial_loss = current_loss = float(full_objective(params))
    objective_calls = 1
    emit("initial_objective", loss=current_loss, params=params)
    if not np.isfinite(current_loss):
        raise SequenceFitError("nonfinite initial training loss")
    trace = [dict(step=0, validation_rollout_mse=best_loss)]
    unweighted = [dict(step=0, validation_rollout_mse=initial_unweighted)]
    checkpoint_losses = [dict(step=0, full_training_loss=current_loss)]
    emit(
        "checkpoint",
        step=0,
        params=params,
        norms=norms,
        validation_rollout_mse=best_loss,
        unweighted_validation_rollout_mse=initial_unweighted,
        full_training_loss=current_loss,
        selected_step=0,
    )
    first, second = (jax.tree.map(jnp.zeros_like, params) for _ in range(2))
    accepted = []
    best_step = 0
    for i in range(1, steps + 1):
        elapsed_check()
        indices = np.arange(len(train.past_states), dtype=np.int64)
        emit(
            "started",
            attempt=i,
            indices=indices,
            params=params,
            first=first,
            second=second,
            current_full_training_loss=current_loss,
        )
        proposal, new_first, new_second, value, norm, grad, finite = update(
            params, first, second, i
        )
        value, norm, finite = float(value), float(norm), bool(finite)
        emit(
            "proposed",
            attempt=i,
            proposal=proposal,
            new_first=new_first,
            new_second=new_second,
            minibatch_loss=value,
            gradient_norm=norm,
            clipped_gradient=grad,
            gradient_finite=finite,
        )
        if (
            not finite
            or not np.isfinite([value, norm]).all()
            or any(
                not np.isfinite(np.asarray(v)).all()
                for v in jax.tree.leaves((proposal, new_first, new_second, grad))
            )
        ):
            raise SequenceFitError(f"nonfinite Adam proposal at attempt {i}")
        first, second = new_first, new_second
        accepted_scale, accepted_index = 0.0, None
        for trial_index, alpha in enumerate(SCALES):
            trial = trial_parameters(params, proposal, alpha)
            trial_loss = float(full_objective(trial))
            objective_calls += 1
            emit(
                "trial",
                attempt=i,
                trial_index=trial_index,
                scale=alpha,
                loss=trial_loss,
                parameters=trial,
            )
            if np.isfinite(trial_loss) and trial_loss < current_loss:
                params, current_loss = trial, trial_loss
                accepted_scale, accepted_index = alpha, trial_index
                break
        accepted.append(accepted_scale)
        emit(
            "completed",
            attempt=i,
            params=params,
            first=first,
            second=second,
            loss=current_loss,
            accepted_scale=accepted_scale,
            accepted_trial_index=accepted_index,
        )
        elapsed_check()
        if i % check_every == 0 or i == steps:
            val_loss, plain_loss = map(float, evaluate(params))
            if not np.isfinite([val_loss, plain_loss]).all():
                raise SequenceFitError(f"nonfinite development loss at {i}")
            trace.append(
                dict(step=i, training_batch_mse=value, validation_rollout_mse=val_loss)
            )
            unweighted.append(dict(step=i, validation_rollout_mse=plain_loss))
            checkpoint_losses.append(dict(step=i, full_training_loss=current_loss))
            if val_loss < best_loss:
                best, best_loss, best_step = params, val_loss, i
            emit(
                "checkpoint",
                step=i,
                params=params,
                norms=norms,
                validation_rollout_mse=val_loss,
                unweighted_validation_rollout_mse=plain_loss,
                full_training_loss=current_loss,
                selected_step=best_step,
            )
    elapsed_check()
    model = VehicleSequenceModel(
        train.dt_s,
        initial.history_steps,
        initial.delay_steps,
        jax.tree.map(np.asarray, best),
        initial.norms,
    )
    safeguard = safeguard_metadata(
        steps=steps,
        accepted_scales=accepted,
        objective_calls=objective_calls,
        initial_loss=initial_loss,
        final_loss=current_loss,
        checkpoint_losses=checkpoint_losses,
        selected_step=best_step,
    )
    safeguard["id"] = RECIPE_ID
    report = dict(
        recipe=RECIPE_ID,
        steps=steps,
        selected_step=best_step,
        validation_rollout_mse=best_loss,
        trace=trace,
        unweighted_development_trace=unweighted,
        safeguard=safeguard,
        gradient=gradient_metadata(
            len(train.past_states),
            attempts_started=steps,
            gradient_proposal_calls_returned=steps,
            completed_acceptance_attempts=steps,
        ),
        batch_size=len(train.past_states),
        parameter_count=sum(v.size for v in model.params.values()),
        ridge=0.01 * len(train.past_states) * train.future_states.shape[1],
        error_scale=scale.tolist(),
        channel_weights=weights.tolist(),
        objective="weighted_normalized_recursive_mse",
        imported_weights=imported_weights,
        initial_fingerprint=initial.fingerprint,
        selected_fingerprint=model.fingerprint,
        actual_initializers=1,
        actual_ridge_solves=2,
        initial_weight_forecasts=int(not imported_weights),
        development_forecasts=len(trace),
        full_training_objective_calls=objective_calls,
        memory_timing="incoming_hidden_fixed_across_substeps_then_advance_once_from_sample_start",
        wall_time_s=time.perf_counter() - started,
    )
    emit("finished", model=model, report=report)
    return model, report
