"""Generic angular response and self-contained public online revisions."""

import jax
import numpy as np
import pytest

from glassbox import OnlineFit, STATE_CHANNELS, SequenceCollection, SequenceSegment
from glassbox._rate import (
    COMMAND_TAU_S,
    RATE_MEMORY_TAU_S,
    fit_rate,
    step_memories,
)


@pytest.mark.parametrize("channels", [1, 3, 4])
def test_episode_rate_solve_recovers_independent_command_and_passive_terms(channels):
    rng = np.random.default_rng(40 + channels)
    dt_s, count = 0.02, 100
    commands = rng.normal(size=(count, channels))
    coefficients = np.zeros((3, 2 * channels + 3))
    coefficients[:, 0] = [0.1, -0.2, 0.05]
    coefficients[:, 1 : channels + 1] = rng.normal(size=(3, channels))
    coefficients[:, channels + 1 : 2 * channels + 1] = rng.normal(size=(3, channels))
    coefficients[:, -2:] = [[0.7, 1.1], [0.4, 0.9], [0.6, 1.3]]
    states = np.zeros((count + 1, 15))
    states[:, 6:] = np.eye(3).ravel()
    states[0, 3:6] = [0.4, -0.6, 0.2]
    applied, memory = commands[0].copy(), states[0, 3:6].copy()
    for row, command in enumerate(commands):
        rate = states[row, 3:6]
        midpoint_command = command + (applied - command) * np.exp(
            -0.5 * dt_s / COMMAND_TAU_S
        )
        midpoint_memory = rate + (memory - rate) * np.exp(
            -0.5 * dt_s / RATE_MEMORY_TAU_S
        )
        states[row + 1, 3:6] = rate + dt_s * np.array(
            [
                np.r_[
                    1, command, midpoint_command, -rate[axis],
                    -(rate[axis] - midpoint_memory[axis]),
                ] @ coefficients[axis]
                for axis in range(3)
            ]
        )
        applied, memory = step_memories(applied, memory, command, rate, dt_s)
    fitted = fit_rate(states, commands, dt_s, last=None)
    np.testing.assert_allclose(fitted, coefficients, rtol=0, atol=1e-11)


def _stream():
    rng = np.random.default_rng(481)
    dt_s, count, channels = 0.01, 90, 4
    commands = rng.uniform(-0.3, 0.3, (count, channels))
    states = np.zeros((count + 1, 15))
    states[:, 6:] = np.eye(3).ravel()
    for row, command in enumerate(commands):
        states[row + 1, :3] = states[row, :3] + dt_s * np.array(
            [command[0], command[1], command[2] - 9.80665]
        )
        states[row + 1, 3:6] = states[row, 3:6] + dt_s * (
            np.array([command[1] - command[2], command[2] - command[3], command[0] - command[1]])
            - 0.8 * states[row, 3:6]
        )
    prefix = SequenceCollection(
        (SequenceSegment("analytic", "segment", states[:76], commands[:75], dt_s, 0),),
        "unlabeled-system", STATE_CHANNELS,
        tuple(f"input-{index} [1]" for index in range(channels)),
    )
    return states, commands, prefix


def test_online_snapshot_derivative_and_save_resume(tmp_path):
    states, commands, prefix = _stream()
    session = OnlineFit(prefix)
    history = session.model.history_steps
    past, inputs = states[75 - history : 76], commands[75 - history : 75]
    old = session.model
    original = np.asarray(old.rollout(past, inputs, commands[75:77]))
    with jax.enable_x64(True):
        response = jax.jacfwd(
            lambda command: old.rollout(past, inputs, command[None])[0, 3:6]
        )(commands[75])
        difference = np.column_stack([
            (
                np.asarray(old.rollout(past, inputs, (commands[75] + 1e-5 * np.eye(4)[axis])[None]))[0, 3:6]
                - np.asarray(old.rollout(past, inputs, (commands[75] - 1e-5 * np.eye(4)[axis])[None]))[0, 3:6]
            ) / 2e-5
            for axis in range(4)
        ])
    np.testing.assert_allclose(response, difference, rtol=1e-7, atol=1e-9)
    for row in range(75, 78):
        session.observe(session.cursor, commands[row], states[row + 1])
    np.testing.assert_array_equal(
        old.rollout(past, inputs, commands[75:77]), original
    )
    path = tmp_path / "online.npz"
    session.save(path)
    resumed = OnlineFit.load(path)
    assert resumed.fingerprint() == session.fingerprint()
    assert resumed.model.fingerprint == session.model.fingerprint
    for row in range(78, 80):
        session.observe(session.cursor, commands[row], states[row + 1])
        resumed.observe(resumed.cursor, commands[row], states[row + 1])
    assert resumed.fingerprint() == session.fingerprint()
