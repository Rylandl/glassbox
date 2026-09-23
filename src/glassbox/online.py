"""Bounded episode-only readout fitting for the shared rigid-body model."""

from __future__ import annotations

import copy
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import cho_solve

from ._dynamics import (
    GRAVITY,
    VehicleSequenceModel,
    _history,
    _rollout,
    additive_readout_features,
    current_features,
    initialize,
    nonlinear_features,
    rotation_exp,
    time_constants,
)
from ._learner_arrays import array_fingerprint, load_arrays, save_arrays
from ._rate import fit_rate, step_memories, window_initial_memories
from ._training import SequenceBatch
from .learner import _contract, _validate_rotations, steps_for

_RECIPE = dict(
    bootstrap_s=0.25,
    horizon_s=0.05,
    rate_window_transitions=25,
    command_time_constant_s=0.08,
    angular_memory_time_constant_s=0.1,
    force_readout="regularized_closed_form_with_motion_and_attitude_sensitivity",
    initial_correction_s=0.25,
    prior_strength=0.01,
)
_FORMAT = "glassbox-online-rate-memory-v1"
_FIELDS = ("past_states", "past_inputs", "future_inputs", "future_states")


def _windows(states, inputs, origins, history, horizon):
    return dict(zip(_FIELDS, (
        np.stack([states[index - history : index + 1] for index in origins]),
        np.stack([inputs[index - history : index] for index in origins]),
        np.stack([inputs[index : index + horizon] for index in origins]),
        np.stack([states[index + 1 : index + horizon + 1] for index in origins]),
    )))


def _full_cache(bootstrap, recent):
    """Fixed slots duplicate actual rows; zero weights exclude unfilled slots."""
    blocks, role_weights = [], []
    for windows in (bootstrap, recent):
        count = len(windows["past_states"])
        if not count:
            raise ValueError("a readout update needs a window in each role")
        indices = np.arange(32) % count
        blocks.append({key: windows[key][indices] for key in _FIELDS})
        role_weights.append(np.where(np.arange(32) < count, 0.5 / count, 0.0))
    return (
        tuple(np.concatenate((blocks[0][key], blocks[1][key])) for key in _FIELDS),
        np.concatenate(role_weights),
    )


def _scale(windows):
    """Fixed physical-group residual scales from the causal bootstrap."""
    origins = windows["past_states"][:, -1:]
    targets = windows["future_states"]
    scales = []
    for beginning, end, factor in ((0, 3, 1.0), (3, 6, 1.0), (6, 15, 0.5)):
        origin = origins[..., beginning:end]
        change = np.sqrt(
            factor * np.mean(np.sum((origin - targets[..., beginning:end]) ** 2, axis=-1), axis=0)
        )
        spread = np.sqrt(
            factor * np.mean(np.sum((origin - origin.mean(axis=0)) ** 2, axis=-1))
        )
        group = np.maximum(change, 0.01 * max(float(spread), 1e-4))
        scales.append(np.repeat((group / np.sqrt(factor))[:, None], end - beginning, axis=1))
    return np.concatenate(scales, axis=1)


def _curvature_diagonal(params, norms, data, scale, weights, *, delay, dt_s):
    """Physical quadratic-head penalty on measured support."""
    past, past_inputs, future_inputs, targets = data
    states = jnp.concatenate((past, targets[:, :-1]), axis=1)
    commands = jnp.concatenate((past_inputs, future_inputs), axis=1)
    current = current_features(states, commands, commands, norms)[:, delay:]
    rms = jnp.sqrt(
        jnp.sum(weights[:, None, None] * current**2, axis=(0, 1)) / current.shape[1]
    )
    count = commands.shape[-1]
    issued = jnp.maximum(1.0, rms[9 : 9 + count])
    domain = jnp.concatenate((
        jnp.maximum(1.0, rms[:6]),
        1.0 / norms["body_scale"][6:9],
        issued, issued,
    ))
    first, second = jnp.triu_indices(domain.shape[0])
    factor = jnp.where(first == second, 2.0, jnp.sqrt(2.0)) * domain[first] * domain[second]
    coefficient = (
        factor[:, None] * dt_s * norms["output_scale"][None, :]
        / (norms["quadratic_scale"][:, None] * scale[0, :3])
    )
    return (0.005 * coefficient**2).reshape(-1)


def readout_matrix(params):
    return np.concatenate(
        (params["linear"], params["quadratic"], params["bias"][None], params["w2"])
    )


def with_readout(model, matrix, rate):
    feature, quadratic = len(model.params["linear"]), len(model.params["quadratic"])
    params = dict(
        model.params,
        linear=matrix[:feature],
        quadratic=matrix[feature : feature + quadratic],
        bias=matrix[feature + quadratic],
        w2=matrix[feature + quadratic + 1 :],
        rate=rate,
    )
    return VehicleSequenceModel(
        model.dt_s, model.history_steps, model.delay_steps, params, model.norms
    )


def readout_features(params, norms, current, history, hidden):
    sampled, quadratic, compact = additive_readout_features(
        current, history, hidden
    )
    return jnp.concatenate(
        (
            sampled / norms["feature_scale"],
            quadratic / norms["quadratic_scale"],
            jnp.ones((*current.shape[:-1], 1), dtype=current.dtype),
            jnp.tanh(
                (compact / norms["nonlinear_scale"]) @ params["w1"] + params["b1"]
            ),
        ),
        axis=-1,
    )


def measurement(params, norms, past, inputs, command, following, *, delay, dt_s):
    applied, history, hidden = _history(
        params, norms, past[None], inputs[None], delay, dt_s
    )
    start = past[-1]
    rotation = start[6:].reshape(3, 3) @ rotation_exp(
        dt_s * 0.25 * (start[3:6] + following[3:6])
    )
    midpoint = jnp.concatenate(((start[:6] + following[:6]) * 0.5, rotation.reshape(9)))
    filtered = command + (applied[0] - command) * jnp.exp(
        -0.5 * dt_s / time_constants(params)
    )
    current = current_features(midpoint, command, filtered, norms)
    phi = readout_features(params, norms, current, history[0], hidden[0])
    target = rotation.T @ (
        (following[:3] - start[:3]) / dt_s
        - jnp.asarray(GRAVITY, dtype=start.dtype)
    ) / norms["output_scale"]
    return phi, target


def sensitivity_bases(params, norms, delay):
    current = 9 + 2 * len(norms["input_mean"])
    size = len(params["linear"])
    basis = np.zeros((size, 6 * (delay + 1)))
    basis[:6, :6] = np.eye(6)
    for lag in range(delay):
        start = (lag + 1) * current
        basis[start : start + 6, :6] = -np.eye(6)
        basis[start : start + 6, (lag + 1) * 6 : (lag + 2) * 6] = np.eye(6)
    compact = nonlinear_features(basis.T, current, delay, np).T
    return (
        basis / norms["feature_scale"][:, None],
        params["w1"].T @ (compact / norms["nonlinear_scale"][:, None]),
    )


def feature_jacobian(phi, norms, linear, nonlinear):
    current = 9 + 2 * norms["input_mean"].shape[0]
    x = phi[:current] * norms["feature_scale"][:current]
    i, j = jnp.triu_indices(current)
    axes = jnp.arange(6)
    retained = ((i < 9) & (j < 9)) | ((i < 3) & (j >= 9))
    quadratic = (
        (i[:, None] == axes) * x[j, None] + (j[:, None] == axes) * x[i, None]
    ) * retained[:, None] / norms["quadratic_scale"][:, None]
    quadratic = jnp.pad(quadratic, ((0, 0), (0, linear.shape[1] - 6)))
    hidden = phi[-nonlinear.shape[0] :]
    return jnp.concatenate(
        (
            linear,
            quadratic,
            jnp.zeros((1, linear.shape[1])),
            (1 - hidden[:, None] ** 2) * nonlinear,
        )
    )


def attitude_derivative(params, norms, past, inputs, command, following, delay, dt_s):
    applied, history, hidden = _history(
        params, norms, past[None], inputs[None], delay, dt_s
    )
    start = past[-1]
    rotation = start[6:].reshape(3, 3) @ rotation_exp(
        dt_s * 0.25 * (start[3:6] + following[3:6])
    )
    midpoint = jnp.concatenate(((start[:6] + following[:6]) * 0.5, rotation.reshape(9)))
    filtered = command + (applied[0] - command) * jnp.exp(
        -0.5 * dt_s / time_constants(params)
    )

    def features(theta):
        perturbed = midpoint.at[6:].set((rotation @ rotation_exp(theta)).reshape(9))
        current = current_features(perturbed, command, filtered, norms)
        return readout_features(params, norms, current, history[0], hidden[0])

    return jax.jacfwd(features)(jnp.zeros(3, dtype=past.dtype))


@partial(jax.jit, static_argnames=("delay", "dt_s"))
def update_readout(
    params, norms, gram, rhs, motion, attitude, bases, count,
    data, scale, weights, past, inputs, command, following, *, delay, dt_s,
):
    phi, target = measurement(
        params, norms, past, inputs, command, following, delay=delay, dt_s=dt_s
    )
    diagonal = _curvature_diagonal(
        {"quadratic": params["quadratic"]},
        norms, data, scale, weights, delay=delay, dt_s=dt_s,
    ).reshape((-1, 3))
    beta = dt_s * norms["output_scale"][0] / scale[0, 0]
    feature, quadratic = len(params["linear"]), len(params["quadratic"])
    penalty = jnp.zeros(len(phi), dtype=phi.dtype).at[feature : feature + quadratic].set(
        3 * count * diagonal[:, 0] / beta**2
    )
    d_motion = feature_jacobian(phi, norms, *bases)
    d_attitude = attitude_derivative(
        params, norms, past, inputs, command, following, delay, dt_s
    )
    gram = gram + jnp.outer(phi, phi)
    rhs = rhs + jnp.outer(phi, target)
    motion = motion + d_motion @ d_motion.T
    attitude = attitude + d_attitude @ d_attitude.T
    system = gram + jnp.diag(penalty) + 0.01 * motion + 0.0001 * attitude
    root = jnp.sqrt(jnp.diag(system))
    lower = jnp.linalg.cholesky(system / root[:, None] / root[None, :])
    mean = cho_solve((lower, True), rhs / root[:, None]) / root[:, None]
    return gram, rhs, motion, attitude, mean


@partial(jax.jit, static_argnames=("delay", "dt_s"))
def trajectory_correction(
    params, norms, mean, information, count,
    past, past_inputs, future_inputs, targets, scale, *, delay, dt_s,
):
    feature, quadratic = len(params["linear"]), len(params["quadratic"])

    def residual(matrix):
        trial = dict(
            params,
            linear=matrix[:feature],
            quadratic=matrix[feature : feature + quadratic],
            bias=matrix[feature + quadratic],
            w2=matrix[feature + quadratic + 1 :],
        )
        forecast = _rollout(trial, norms, past, past_inputs, future_inputs, delay, dt_s)
        return (forecast - targets) / scale

    error, pullback = jax.vjp(residual, mean)
    length = error.size
    gradient = pullback(error)[0] / length
    root = jnp.sqrt(jnp.diag(information))
    lower = jnp.linalg.cholesky(information / root[:, None] / root[None, :])
    direction = cho_solve((lower, True), gradient / root[:, None]) / root[:, None]
    effect = jax.jvp(residual, (mean,), (direction,))[1]
    curvature = jnp.sum(direction * (information @ direction))
    step = count * jnp.sum(error * effect) / (
        length * curvature + count * jnp.sum(effect * effect)
    )
    return mean - step * direction


@partial(jax.jit, static_argnames=("delay", "dt_s"))
def predict(params, norms, past, inputs, future, *, delay, dt_s):
    return _rollout(params, norms, past, inputs, future, delay, dt_s)


class OnlineFit:
    """Causal mutable fitter whose snapshots use the public dynamics rollout."""

    def __init__(self, prefix):
        contract = _contract(prefix)
        if len(prefix.segments) != 1:
            raise ValueError("online initialization requires one contiguous segment")
        segment = prefix.segments[0]
        history = steps_for(segment.dt_s)["history"]
        count = max(1, int(np.rint(0.25 / segment.dt_s)))
        horizon = max(1, int(np.rint(0.05 / segment.dt_s)))
        if count < 3 or len(segment.inputs) < history + count:
            raise ValueError("online prefix needs 0.5 s history and 0.25 s transitions")
        states = segment.states[-history - count - 1 :].copy()
        inputs = segment.inputs[-history - count :].copy()
        one_step = _windows(states, inputs, range(history, history + count), history, 1)
        with jax.enable_x64(True):
            model = initialize(SequenceBatch(**one_step, dt_s=segment.dt_s))
        rate_count = min(25, len(segment.inputs))
        self._rate_states = segment.states[-rate_count - 1 :].copy()
        self._rate_inputs = segment.inputs[-rate_count:].copy()
        self._rate_applied_start, self._rate_memory_start = window_initial_memories(
            segment.states, segment.inputs, segment.dt_s, rate_count
        )
        model = with_readout(
            model, readout_matrix(model.params),
            fit_rate(
                self._rate_states, self._rate_inputs, model.dt_s,
                initial_applied=self._rate_applied_start,
                initial_memory=self._rate_memory_start,
            ),
        )
        self._bootstrap = _windows(
            states, inputs,
            list(range(history, history + count - horizon + 1))[-32:],
            history, horizon,
        )
        self._recent = {key: value[:0].copy() for key, value in self._bootstrap.items()}
        self._scale = _scale(self._bootstrap)
        self._states = states[-history - horizon - 1 :].copy()
        self._inputs = inputs[-history - horizon :].copy()
        self._contract = contract
        self._initial_cursor = segment.start_row + len(segment.inputs)
        self._cursor = self._initial_cursor
        self._initial_count, self._horizon = count, horizon
        self._count = 0
        size = len(readout_matrix(model.params))
        with jax.enable_x64(True):
            self._mean = readout_matrix(model.params)
            ridge = 0.01 * count
            self._gram = ridge * jnp.eye(size, dtype=jnp.float64)
            self._rhs = ridge * self._mean
            self._motion = jnp.zeros_like(self._gram)
            self._attitude = jnp.zeros_like(self._gram)
            self._bases = tuple(
                jnp.asarray(value)
                for value in sensitivity_bases(model.params, model.norms, model.delay_steps)
            )
            trajectory_length = round(0.25 / model.dt_s)
            size = history + trajectory_length
            corrected = trajectory_correction(
                model.params, model.norms, self._mean, self._gram, count,
                jnp.asarray(states[-size - 1 : -trajectory_length])[None],
                jnp.asarray(inputs[-size:-trajectory_length])[None],
                jnp.asarray(inputs[-trajectory_length:])[None],
                jnp.asarray(states[-trajectory_length:])[None],
                jnp.asarray(self._scale[-1]),
                delay=model.delay_steps, dt_s=model.dt_s,
            )
            corrected = np.asarray(corrected)
            if not np.isfinite(corrected).all():
                raise ValueError("nonfinite initial trajectory readout")
            self._mean = jnp.asarray(corrected)
            self._rhs = self._gram @ self._mean
        self._model = with_readout(model, corrected, model.params["rate"])

    @property
    def cursor(self):
        return self._cursor

    @property
    def model(self):
        model = self._model
        return VehicleSequenceModel(
            model.dt_s, model.history_steps, model.delay_steps, model.params, model.norms
        )

    @property
    def report(self):
        return dict(
            recipe=copy.deepcopy(_RECIPE),
            cursor=self.cursor,
            observations=self._count,
            initialized_transition_count=self._initial_count,
            history_steps=self._model.history_steps,
            training_horizon_steps=self._horizon,
            bootstrap_windows=len(self._bootstrap["past_states"]),
            recent_windows=len(self._recent["past_states"]),
            envelope=dict(available=False),
        )

    def predict(self, past_states, past_inputs, future_inputs):
        x, inputs, future = map(jnp.asarray, (past_states, past_inputs, future_inputs))
        model = self._model
        channels, history = len(model.norms["input_mean"]), model.history_steps
        if (
            x.ndim not in (2, 3)
            or inputs.ndim != x.ndim
            or future.ndim != x.ndim
            or x.shape[:-2] != inputs.shape[:-2]
            or x.shape[:-2] != future.shape[:-2]
            or x.shape[-1] != 15
            or inputs.shape[-1] != channels
            or future.shape[-1] != channels
            or x.shape[-2] != inputs.shape[-2] + 1
            or inputs.shape[-2] < history
            or not 1 <= future.shape[-2] <= max(1, int(np.rint(1.2 / model.dt_s)))
        ):
            raise ValueError("online prediction needs aligned observed history")
        if not any(isinstance(value, jax.core.Tracer) for value in (x, inputs, future)):
            _validate_rotations(np.asarray(x[..., -history - 1 :, :]))
            if not np.isfinite(np.asarray(inputs)).all() or not np.isfinite(np.asarray(future)).all():
                raise ValueError("commands must be finite")
        single = x.ndim == 2
        if single:
            x, inputs, future = x[None], inputs[None], future[None]
        dtype = jnp.result_type(model.norms["state_scale"])
        params, norms = jax.tree.map(
            lambda value: jnp.asarray(value, dtype=dtype), (model.params, model.norms)
        )
        result = predict(
            params, norms,
            jnp.asarray(x[:, -history - 1 :], dtype=dtype),
            jnp.asarray(inputs[:, -history:], dtype=dtype),
            jnp.asarray(future, dtype=dtype),
            delay=model.delay_steps, dt_s=model.dt_s,
        )
        return result[0] if single else result

    def observe(self, index, command, next_observation):
        if isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)) or index != self.cursor:
            raise ValueError("observation index must equal the next command row")
        command = np.asarray(command, dtype=float)
        state = np.asarray(next_observation, dtype=float)
        if command.shape != (len(self._contract["input_channels"]),) or not np.isfinite(command).all() or state.shape != (15,):
            raise ValueError("invalid observed command/state")
        _validate_rotations(state)
        model = self._model
        states = np.concatenate((self._states[1:], state[None]))
        inputs = np.concatenate((self._inputs[1:], command[None]))
        new = _windows(states, inputs, [model.history_steps], model.history_steps, self._horizon)
        recent = {
            key: np.concatenate((self._recent[key], new[key]))[-32:]
            for key in _FIELDS
        }
        data, weights = _full_cache(self._bootstrap, recent)
        applied_start, memory_start = self._rate_applied_start, self._rate_memory_start
        if len(self._rate_inputs) == 25:
            applied_start, memory_start = step_memories(
                applied_start, memory_start, self._rate_inputs[0],
                self._rate_states[0, 3:6], model.dt_s,
            )
        rate_states = np.concatenate((self._rate_states, state[None]))[-26:]
        rate_inputs = np.concatenate((self._rate_inputs, command[None]))[-25:]
        fitted_rate = fit_rate(
            rate_states, rate_inputs, model.dt_s,
            initial_applied=applied_start, initial_memory=memory_start,
        )
        with jax.enable_x64(True):
            values = update_readout(
                model.params, model.norms,
                self._gram, self._rhs, self._motion, self._attitude,
                self._bases, self._count + 1,
                tuple(jnp.asarray(value) for value in data),
                jnp.asarray(self._scale), jnp.asarray(weights),
                jnp.asarray(self._states[-model.history_steps - 1 :]),
                jnp.asarray(self._inputs[-model.history_steps:]),
                jnp.asarray(command), jnp.asarray(state),
                delay=model.delay_steps, dt_s=model.dt_s,
            )
            gram, rhs, motion, attitude, mean = values
            host = np.asarray(mean)
            if not all(np.isfinite(np.asarray(value)).all() for value in values):
                raise ValueError("nonfinite online readout")
        following_model = with_readout(model, host, fitted_rate)
        self._model = following_model
        self._gram, self._rhs = gram, rhs
        self._motion, self._attitude, self._mean = motion, attitude, mean
        self._states, self._inputs, self._recent = states, inputs, recent
        self._rate_states, self._rate_inputs = rate_states, rate_inputs
        self._rate_applied_start, self._rate_memory_start = applied_start, memory_start
        self._count += 1
        self._cursor += 1

    def _metadata(self):
        return dict(
            format=_FORMAT,
            recipe=copy.deepcopy(_RECIPE),
            model=self._model.metadata(),
            contract=copy.deepcopy(self._contract),
            initial_cursor=self._initial_cursor,
            cursor=self._cursor,
            initial_count=self._initial_count,
            horizon=self._horizon,
        )

    def _arrays(self):
        arrays = self._model.arrays()
        arrays.update(
            gram=np.asarray(self._gram),
            rhs=np.asarray(self._rhs),
            motion=np.asarray(self._motion),
            attitude=np.asarray(self._attitude),
            mean=np.asarray(self._mean),
            basis_linear=np.asarray(self._bases[0]),
            basis_nonlinear=np.asarray(self._bases[1]),
            scale=self._scale,
            tail_states=self._states,
            tail_inputs=self._inputs,
            rate_states=self._rate_states,
            rate_inputs=self._rate_inputs,
            rate_applied_start=self._rate_applied_start,
            rate_memory_start=self._rate_memory_start,
        )
        for role, windows in (("bootstrap", self._bootstrap), ("recent", self._recent)):
            arrays.update({f"{role}_{key}": value for key, value in windows.items()})
        return arrays

    def fingerprint(self):
        return array_fingerprint(self._metadata(), self._arrays())

    def save(self, path):
        self._validate()
        save_arrays(path, self._metadata(), self._arrays())

    @classmethod
    def load(cls, path):
        try:
            metadata, arrays = load_arrays(path)
            if (
                set(metadata)
                != {"format", "recipe", "model", "contract", "initial_cursor", "cursor", "initial_count", "horizon"}
                or metadata["format"] != _FORMAT
                or metadata["recipe"] != _RECIPE
            ):
                raise ValueError("unsupported online rate-memory archive")
            obj = cls.__new__(cls)
            obj._model = VehicleSequenceModel.from_arrays(
                metadata["model"],
                {key: value for key, value in arrays.items() if key.startswith(("param_", "norm_"))},
            )
            obj._contract = copy.deepcopy(metadata["contract"])
            obj._initial_cursor = metadata["initial_cursor"]
            obj._cursor = metadata["cursor"]
            obj._initial_count = metadata["initial_count"]
            obj._horizon = metadata["horizon"]
            obj._count = obj._cursor - obj._initial_cursor
            for attribute, key in (
                ("_scale", "scale"),
                ("_states", "tail_states"),
                ("_inputs", "tail_inputs"),
                ("_rate_states", "rate_states"),
                ("_rate_inputs", "rate_inputs"),
                ("_rate_applied_start", "rate_applied_start"),
                ("_rate_memory_start", "rate_memory_start"),
            ):
                setattr(obj, attribute, arrays[key].copy())
            for role in ("bootstrap", "recent"):
                setattr(obj, "_" + role, {
                    key: arrays[f"{role}_{key}"].copy() for key in _FIELDS
                })
            with jax.enable_x64(True):
                for attribute in ("gram", "rhs", "motion", "attitude", "mean"):
                    setattr(obj, "_" + attribute, jnp.asarray(arrays[attribute]))
                obj._bases = (
                    jnp.asarray(arrays["basis_linear"]),
                    jnp.asarray(arrays["basis_nonlinear"]),
                )
            if set(arrays) != set(obj._arrays()):
                raise ValueError("unexpected online rate-memory arrays")
            obj._validate()
            return obj
        except (KeyError, TypeError, AttributeError, IndexError) as error:
            raise ValueError("invalid online rate-memory archive") from error

    def _validate(self):
        from .learner import STATE_CHANNELS

        model = self._model
        history, channels = model.history_steps, len(model.norms["input_mean"])
        expected = steps_for(model.dt_s)
        contract = self._contract
        if (
            not isinstance(contract, dict)
            or set(contract) != {"configuration_id", "state_channels", "input_channels", "dt_s"}
            or not isinstance(contract["configuration_id"], str)
            or not contract["configuration_id"].strip()
            or tuple(contract["state_channels"]) != STATE_CHANNELS
            or not isinstance(contract["input_channels"], list)
            or len(contract["input_channels"]) != channels
            or any(not isinstance(item, str) or not item.strip() for item in contract["input_channels"])
            or len(set(contract["input_channels"])) != channels
            or contract["dt_s"] != model.dt_s
            or any(type(value) is not int for value in (
                self._cursor, self._initial_cursor, self._initial_count, self._horizon
            ))
            or self._initial_count != max(1, int(np.rint(0.25 / model.dt_s)))
            or self._horizon != max(1, int(np.rint(0.05 / model.dt_s)))
            or history != expected["history"]
            or model.delay_steps != expected["delay"]
            or self._initial_cursor < history + self._initial_count
            or self._cursor < self._initial_cursor
        ):
            raise ValueError("invalid online rate-memory timing or contract")
        arrays = self._arrays()
        if any(value.dtype != np.dtype("float64") or not np.isfinite(value).all() for value in arrays.values()):
            raise ValueError("online rate-memory arrays must be finite float64")
        size = len(readout_matrix(model.params))
        lag = 6 * (model.delay_steps + 1)
        if (
            self._gram.shape != (size, size)
            or self._rhs.shape != (size, 3)
            or self._motion.shape != (size, size)
            or self._attitude.shape != (size, size)
            or self._mean.shape != (size, 3)
            or self._bases[0].shape != (len(model.params["linear"]), lag)
            or self._bases[1].shape != (len(model.params["b1"]), lag)
            or not np.array_equal(np.asarray(self._mean), readout_matrix(model.params))
            or self._count != self._cursor - self._initial_cursor
        ):
            raise ValueError("online readout state differs from model")
        bases = sensitivity_bases(model.params, model.norms, model.delay_steps)
        if any(not np.array_equal(np.asarray(saved), actual) for saved, actual in zip(self._bases, bases)):
            raise ValueError("online sensitivity basis differs from model")
        if (
            self._states.shape != (history + self._horizon + 1, 15)
            or self._inputs.shape != (history + self._horizon, channels)
            or self._scale.shape != (self._horizon, 15)
            or not np.array_equal(self._scale, _scale(self._bootstrap))
            or self._rate_states.ndim != 2
            or self._rate_states.shape[-1] != 15
            or self._rate_inputs.ndim != 2
            or self._rate_inputs.shape[-1] != channels
            or self._rate_states.shape[0] != self._rate_inputs.shape[0] + 1
            or not 1 <= len(self._rate_inputs) <= 25
            or self._rate_applied_start.shape != (channels,)
            or self._rate_memory_start.shape != (3,)
            or not np.array_equal(
                self._states[-min(len(self._states), len(self._rate_states)):],
                self._rate_states[-min(len(self._states), len(self._rate_states)):],
            )
            or not np.array_equal(
                self._inputs[-min(len(self._inputs), len(self._rate_inputs)):],
                self._rate_inputs[-min(len(self._inputs), len(self._rate_inputs)):],
            )
        ):
            raise ValueError("online observed tail differs from rate window")
        _validate_rotations(self._states)
        _validate_rotations(self._rate_states)
        recovered = fit_rate(
            self._rate_states, self._rate_inputs, model.dt_s,
            initial_applied=self._rate_applied_start,
            initial_memory=self._rate_memory_start,
        )
        if not np.allclose(recovered, model.params["rate"], rtol=1e-10, atol=1e-10):
            raise ValueError("saved angular readout differs from retained observations")
        shapes = dict(
            past_states=(history + 1, 15),
            past_inputs=(history, channels),
            future_inputs=(self._horizon, channels),
            future_states=(self._horizon, 15),
        )
        for windows, count in (
            (self._bootstrap, min(32, self._initial_count - self._horizon + 1)),
            (self._recent, min(32, self._count)),
        ):
            if any(windows[key].shape != (count, *shape) for key, shape in shapes.items()):
                raise ValueError("online replay windows differ from recipe")
            if count:
                _validate_rotations(windows["past_states"])
                _validate_rotations(windows["future_states"])
