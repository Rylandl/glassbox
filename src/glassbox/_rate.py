"""Episode-fitted command response and dissipative angular-rate memory."""

import jax
import jax.numpy as jnp
import numpy as np

COMMAND_TAU_S = 0.08
RATE_MEMORY_TAU_S = 0.1
RATE_FIT_TRANSITIONS = 25


def step_memories(applied, memory, command, rate, dt_s):
    """Advance the sufficient command/rate states over one observed interval."""
    return (
        command + (applied - command) * np.exp(-dt_s / COMMAND_TAU_S),
        rate + (memory - rate) * np.exp(-dt_s / RATE_MEMORY_TAU_S),
    )


def window_initial_memories(states, commands, dt_s, count):
    """Locate exact memory states at the start of a bounded fitting window."""
    applied = np.array(commands[0], copy=True)
    memory = np.array(states[0, 3:6], copy=True)
    for row in range(len(commands) - count):
        applied, memory = step_memories(
            applied, memory, commands[row], states[row, 3:6], dt_s
        )
    return applied, memory


def fit_rate(
    states, commands, dt_s, *, last=RATE_FIT_TRANSITIONS,
    initial_applied=None, initial_memory=None,
):
    """Fit physical angular acceleration from completed transitions only."""
    states, commands = np.asarray(states), np.asarray(commands)
    if states.ndim == 2:
        states, commands = states[None], commands[None]
    if (
        states.ndim != 3
        or commands.ndim != 3
        or states.shape[:2] != (len(commands), commands.shape[1] + 1)
        or states.shape[-1] != 15
        or commands.shape[-1] < 1
    ):
        raise ValueError("rate fitting needs aligned observed states and commands")
    count = commands.shape[1]
    rate = states[..., 3:6]
    applied = (
        commands[:, 0].copy()
        if initial_applied is None else np.broadcast_to(initial_applied, commands[:, 0].shape).copy()
    )
    memory = (
        rate[:, 0].copy()
        if initial_memory is None else np.broadcast_to(initial_memory, rate[:, 0].shape).copy()
    )
    midpoint_commands, midpoint_memory = [], []
    half_command = np.exp(-0.5 * dt_s / COMMAND_TAU_S)
    half_rate = np.exp(-0.5 * dt_s / RATE_MEMORY_TAU_S)
    command_decay = half_command**2
    rate_decay = half_rate**2
    for row in range(count):
        command = commands[:, row]
        observed_rate = rate[:, row]
        midpoint_commands.append(command + (applied - command) * half_command)
        midpoint_memory.append(observed_rate + (memory - observed_rate) * half_rate)
        applied = command + (applied - command) * command_decay
        memory = observed_rate + (memory - observed_rate) * rate_decay
    beginning = max(0, count - last) if last is not None else 0
    command = commands[:, beginning:].reshape(-1, commands.shape[-1])
    midpoint_command = np.stack(midpoint_commands, axis=1)[:, beginning:].reshape(
        command.shape
    )
    midpoint_memory = np.stack(midpoint_memory, axis=1)[:, beginning:].reshape(-1, 3)
    observed_rate = rate[:, beginning:-1].reshape(-1, 3)
    target = ((rate[:, beginning + 1 :] - rate[:, beginning:-1]) / dt_s).reshape(-1, 3)
    base = np.column_stack((np.ones(len(command)), command, midpoint_command))
    result = []
    for axis in range(3):
        design = np.column_stack(
            (base, -observed_rate[:, axis],
             -(observed_rate[:, axis] - midpoint_memory[:, axis]))
        )
        candidates = []
        for active in ((), (0,), (1,), (0, 1)):
            columns = list(range(base.shape[1])) + [base.shape[1] + index for index in active]
            fitted = np.linalg.lstsq(design[:, columns], target[:, axis], rcond=None)[0]
            if np.any(fitted[base.shape[1] :] < 0):
                continue
            coefficients = np.zeros(design.shape[1])
            coefficients[columns] = fitted
            error = np.linalg.norm(design @ coefficients - target[:, axis]) ** 2
            candidates.append((error, coefficients))
        result.append(min(candidates, key=lambda item: item[0])[1])
    return np.asarray(result)


def rate_history(past, inputs, dt_s):
    """Reconstruct initial rollout memories from the public observed history."""
    command_decay = jnp.exp(-dt_s / COMMAND_TAU_S)
    rate_decay = jnp.exp(-dt_s / RATE_MEMORY_TAU_S)

    def advance(carry, data):
        applied, memory = carry
        command, rate = data
        return (
            command + (applied - command) * command_decay,
            rate + (memory - rate) * rate_decay,
        ), None

    (applied, memory), _ = jax.lax.scan(
        advance,
        (inputs[:, 0], past[:, 0, 3:6]),
        (inputs.swapaxes(0, 1), past[:, :-1, 3:6].swapaxes(0, 1)),
    )
    return applied, memory
