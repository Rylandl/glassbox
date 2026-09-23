"""One JAX rigid-body rollout with a causal positive-damping rate readout."""

import math
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from glassbox._dynamics import (
    GRAVITY,
    MAX_SUBSTEP_S,
    _acceleration,
    _history,
    _prepare_head,
    current_features,
    memory_step,
    rotation_exp,
    state_without_current_command,
    time_constants,
)
from screen_causal_rate import CausalRateFit


def joint_step(params, norms, state, command, force_applied, history, hidden,
               rate_applied, rate_memory, coefficient, rate_tau,
               memory_tau, dt_s):
    count = max(1, math.ceil(dt_s / MAX_SUBSTEP_S))
    duration = dt_s / count
    force_tau = time_constants(params)
    gravity = jnp.asarray(GRAVITY, dtype=state.dtype)
    m = command.shape[0]
    damping = coefficient[:, -2]
    memory_gain = coefficient[:, -1]
    start = current_features(state, command, force_applied, norms)
    start_state = state_without_current_command(start)
    base, effective = _prepare_head(params, norms, start_state, history, hidden)

    def force_acceleration(current_state, applied):
        current = current_features(current_state, command, applied, norms)
        projected = base + (state_without_current_command(current) - start_state) @ effective
        return _acceleration(params, norms, current, projected)[:3]

    def rate_acceleration(current_state, applied, filtered_rate):
        return (
            coefficient[:, 0]
            + coefficient[:, 1 : m + 1] @ command
            + coefficient[:, m + 1 : 2 * m + 1] @ applied
            - damping * current_state[3:6]
            - memory_gain * (current_state[3:6] - filtered_rate)
        )

    def substep(_, carry):
        current_state, applied_force, applied_rate, filtered_rate = carry
        velocity, rate = current_state[:3], current_state[3:6]
        rotation = current_state[6:].reshape(3, 3)
        first_force = force_acceleration(current_state, applied_force)
        first_rate = rate_acceleration(current_state, applied_rate, filtered_rate)
        world_first = gravity + rotation @ first_force
        half_rotation = rotation @ rotation_exp(0.5 * duration * rate)
        half_velocity = velocity + 0.5 * duration * world_first
        half_rate = rate + 0.5 * duration * first_rate
        half_force_applied = command + (applied_force - command) * jnp.exp(
            -0.5 * duration / force_tau
        )
        half_rate_applied = command + (applied_rate - command) * jnp.exp(
            -0.5 * duration / rate_tau
        )
        half_filtered_rate = rate + (filtered_rate - rate) * jnp.exp(
            -0.5 * duration / memory_tau
        )
        half_state = jnp.concatenate(
            (half_velocity, half_rate, half_rotation.reshape(9))
        )
        middle_force = force_acceleration(half_state, half_force_applied)
        middle_rate = rate_acceleration(
            half_state, half_rate_applied, half_filtered_rate
        )
        next_velocity = velocity + duration * (gravity + half_rotation @ middle_force)
        next_rate = rate + duration * middle_rate
        next_rotation = rotation @ rotation_exp(duration * half_rate)
        following = jnp.concatenate(
            (next_velocity, next_rate, next_rotation.reshape(9))
        )
        next_force_applied = command + (applied_force - command) * jnp.exp(
            -duration / force_tau
        )
        next_rate_applied = command + (applied_rate - command) * jnp.exp(
            -duration / rate_tau
        )
        next_filtered_rate = rate + (filtered_rate - rate) * jnp.exp(
            -duration / memory_tau
        )
        return following, next_force_applied, next_rate_applied, next_filtered_rate

    following, force_next, rate_next, memory_next = jax.lax.fori_loop(
        0, count, substep, (state, force_applied, rate_applied, rate_memory)
    )
    hidden_next = memory_step(params, norms, start, hidden, dt_s)
    history_next = jnp.concatenate((history[1:], start[None]), axis=0)
    return following, force_next, history_next, hidden_next, rate_next, memory_next


@partial(jax.jit, static_argnames=("delay", "dt_s"))
def joint_rollout(params, norms, past, past_inputs, future, rate_applied,
                  rate_memory, coefficient, rate_tau, memory_tau,
                  *, delay, dt_s):
    force_applied, history, hidden = _history(
        params, norms, past[None], past_inputs[None], delay, dt_s
    )

    def advance(carry, command):
        state, force_applied, history, hidden, rate_applied, rate_memory = carry
        following = joint_step(
            params, norms, state, command, force_applied, history, hidden,
            rate_applied, rate_memory, coefficient, rate_tau, memory_tau, dt_s
        )
        return following, following[0]

    initial = (
        past[-1], force_applied[0], history[0], hidden[0],
        rate_applied, rate_memory
    )
    _, prediction = jax.lax.scan(advance, initial, future)
    return prediction


class JaxRateSession:
    def __init__(self, owner):
        self.owner = owner

    @property
    def model(self):
        return self.owner.force.session.model

    def predict(self, past, inputs, future):
        model = self.model
        with jax.enable_x64(True):
            prediction = joint_rollout(
                {key: jnp.asarray(value) for key, value in model.params.items()},
                {key: jnp.asarray(value) for key, value in model.norms.items()},
                jnp.asarray(past),
                jnp.asarray(inputs),
                jnp.asarray(future),
                jnp.asarray(self.owner.applied[-1]),
                jnp.asarray(self.owner.rate_memory),
                jnp.asarray(self.owner.rate_coefficients),
                jnp.asarray(self.owner.tau),
                jnp.asarray(self.owner.memory_tau),
                delay=model.delay_steps,
                dt_s=model.dt_s,
            )
        return np.asarray(prediction)


class JaxHybrid(CausalRateFit):
    def __init__(self, prefix):
        super().__init__(prefix)
        self.session = JaxRateSession(self)
