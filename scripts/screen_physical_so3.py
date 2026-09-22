"""One frozen physical SO(3) sensitivity term for the cold-start direct readout."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import cho_solve
from screen_cold_readout_curvature import (
    curvature_penalty,
    measurement,
    readout_features,
)
from screen_readout_sensitivity import (
    ETA,
    SensitivityReadout,
    feature_jacobian,
)
from verify_baseline import require

from glassbox import online
from glassbox._dynamics import (
    VehicleSequenceModel,
    _history,
    current_features,
    rotation_exp,
    time_constants,
)

ATTITUDE_ETA = 0.0001


def midpoint_context(params, norms, past, inputs, command, following, delay, dt_s):
    """Reproduce the already-frozen measured-midpoint feature context."""
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
    return midpoint, rotation, filtered, history[0], hidden[0]


def attitude_features(
    params, norms, past, inputs, command, following, theta, delay, dt_s
):
    """Readout features after a right physical attitude perturbation in radians."""
    midpoint, rotation, filtered, history, hidden = midpoint_context(
        params, norms, past, inputs, command, following, delay, dt_s
    )
    perturbed = midpoint.at[6:].set((rotation @ rotation_exp(theta)).reshape(9))
    current = current_features(perturbed, command, filtered, norms)
    return readout_features(params, norms, current, history, hidden)


def attitude_derivative(params, norms, past, inputs, command, following, delay, dt_s):
    theta = jnp.zeros(3, dtype=past.dtype)
    return jax.jacfwd(
        lambda value: attitude_features(
            params, norms, past, inputs, command, following, value, delay, dt_s
        )
    )(theta)


@partial(jax.jit, static_argnames=("delay", "dt_s"))
def physical_update(
    params,
    norms,
    gram,
    rhs,
    motion,
    attitude,
    bases,
    n,
    data,
    scale,
    weights,
    past,
    inputs,
    command,
    following,
    *,
    delay,
    dt_s,
):
    phi, target = measurement(
        params, norms, past, inputs, command, following, delay=delay, dt_s=dt_s
    )
    penalty = curvature_penalty(
        params, norms, data, scale, weights, n, delay=delay, dt_s=dt_s
    )
    d_motion = feature_jacobian(phi, norms, *bases)
    d_attitude = attitude_derivative(
        params, norms, past, inputs, command, following, delay, dt_s
    )
    gram = gram + jnp.outer(phi, phi)
    rhs = rhs + jnp.outer(phi, target)
    motion = motion + d_motion @ d_motion.T
    attitude = attitude + d_attitude @ d_attitude.T
    system = gram + jnp.diag(penalty) + ETA * motion + ATTITUDE_ETA * attitude
    root = jnp.sqrt(jnp.diag(system))
    lower = jnp.linalg.cholesky(system / root[:, None] / root[None, :])
    mean = cho_solve((lower, True), rhs / root[:, None]) / root[:, None]
    return gram, rhs, motion, attitude, mean, phi, target, penalty, d_attitude


class PhysicalSO3Readout(SensitivityReadout):
    def __init__(self, prefix):
        super().__init__(prefix)
        with jax.enable_x64(True):
            self.attitude_sensitivity = jnp.zeros_like(self.gram)
        jax.block_until_ready(self.attitude_sensitivity)

    def observe(self, row, command, following):
        require(row == self.session.cursor, "noncausal update")
        model = self.session.model
        states = np.concatenate((self.session._states[1:], following[None]))
        inputs = np.concatenate((self.session._inputs[1:], command[None]))
        new = online._windows(
            states,
            inputs,
            [model.history_steps],
            model.history_steps,
            self.session._horizon,
        )
        recent = {
            key: np.concatenate((self.session._recent[key], new[key]))[-32:]
            for key in online._FIELDS
        }
        data, weights = online._full_cache(self.session._bootstrap, recent)
        with jax.enable_x64(True):
            values = physical_update(
                model.params,
                model.norms,
                self.gram,
                self.rhs,
                self.sensitivity,
                self.attitude_sensitivity,
                self.bases,
                self.count + 1,
                tuple(jnp.asarray(value) for value in data),
                jnp.asarray(self.session._scale),
                jnp.asarray(weights),
                self.states,
                self.inputs,
                command,
                following,
                delay=model.delay_steps,
                dt_s=model.dt_s,
            )
            gram, rhs, motion, attitude, mean, phi, target, penalty, d_attitude = values
            saved = tuple(
                np.asarray(value) for value in (mean, phi, target, penalty, d_attitude)
            )
            require(
                all(np.isfinite(value).all() for value in saved),
                "nonfinite SO3 readout",
            )
        host = saved[0]
        f, q = len(model.params["linear"]), len(model.params["quadratic"])
        params = dict(
            model.params,
            linear=host[:f],
            quadratic=host[f : f + q],
            bias=host[f + q],
            w2=host[f + q + 1 :],
        )
        self.session._model = VehicleSequenceModel(
            model.dt_s, model.history_steps, model.delay_steps, params, model.norms
        )
        self.session._cursor += 1
        self.session._states, self.session._inputs, self.session._recent = (
            states,
            inputs,
            recent,
        )
        self.count += 1
        self.gram, self.rhs, self.sensitivity, self.attitude_sensitivity, self.mean = (
            gram,
            rhs,
            motion,
            attitude,
            mean,
        )
        self.states = np.concatenate((self.states[1:], following[None]))
        self.inputs = np.concatenate((self.inputs[1:], command[None]))
        return saved
