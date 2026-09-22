"""Analytic and causal checks for the bounded trajectory correction screen."""

import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from screen_cold_readout_curvature import coefficients
from screen_trajectory_path import (
    TrajectoryPathReadout,
    correction,
    trajectory_residual,
)
from verify_baseline import arrays, observed

from glassbox import STATE_CHANNELS, SequenceCollection, SequenceSegment
from glassbox import _dynamics as core
from glassbox._dynamics import VehicleSequenceModel

ROOT = Path("/Users/ryland/autonomy/glassbox/artifacts")


@pytest.mark.parametrize(
    "name,source,begin,first,dt,n_commands",
    [
        (
            "fixedwing-80",
            ROOT / "online-fit-v8/evaluation/inputs/fixedwing-80.npz",
            0,
            15,
            0.05,
            3,
        ),
        (
            "paired-quad-fine",
            ROOT / "paired-quad-sampling-v2/fine.npz",
            50,
            125,
            0.01,
            4,
        ),
    ],
)
def test_trajectory_jacobian_and_dual(name, source, begin, first, dt, n_commands):
    parent = ROOT / "readout-physical-so3-v1/evaluation" / name
    model = VehicleSequenceModel.from_arrays(
        json.loads((parent / "initial.json").read_text()),
        arrays(parent / "initial.npz"),
    )
    tape = arrays(source)
    states, commands = observed(tape["states"]), tape["commands"]
    horizon = round(0.25 / dt)
    origin = first - horizon
    history = model.history_steps
    with jax.enable_x64(True):
        params, norms = jax.tree.map(jnp.asarray, (model.params, model.norms))
        past = jnp.asarray(states[origin - history : origin + 1])
        inputs = jnp.asarray(commands[origin - history : origin])
        future = jnp.asarray(commands[origin:first])
        truth = jnp.asarray(states[origin + 1 : first + 1])
        gram = jnp.eye(len(coefficients(params)), dtype=jnp.float64)
        head, r, jacobian, scale, z, delta, trials, multiplier, baseline = correction(
            params,
            norms,
            gram,
            past,
            inputs,
            future,
            truth,
            delay=model.delay_steps,
            dt_s=dt,
        )
        mean = coefficients(params)
        rollout = np.asarray(
            core._rollout(
                params,
                norms,
                past[None],
                inputs[None],
                future[None],
                model.delay_steps,
                dt,
            )[0]
        )
        checkpoints = np.arange(1, 6) * round(0.05 / dt) - 1
        np.testing.assert_array_equal(
            checkpoints,
            [0, 1, 2, 3, 4] if dt == 0.05 else [4, 9, 14, 19, 24],
        )
        difference = rollout[checkpoints] - np.asarray(truth)[checkpoints]
        difference[:, 6:] /= np.sqrt(2)
        np.testing.assert_allclose(np.asarray(r), (difference / np.sqrt(5)).ravel())
        direction = np.random.default_rng(387).normal(size=mean.shape)
        direction /= np.linalg.norm(direction)
        epsilon = 1e-6
        plus = trajectory_residual(
            mean + epsilon * direction,
            params,
            norms,
            past,
            inputs,
            future,
            truth,
            delay=model.delay_steps,
            dt_s=dt,
        )
        minus = trajectory_residual(
            mean - epsilon * direction,
            params,
            norms,
            past,
            inputs,
            future,
            truth,
            delay=model.delay_steps,
            dt_s=dt,
        )
        measured = (np.asarray(plus) - np.asarray(minus)) / (2 * epsilon)
        predicted = np.asarray(jacobian) @ direction.ravel()
        np.testing.assert_allclose(predicted, measured, rtol=1e-4, atol=1e-5)
        whitened = np.asarray(jacobian) * np.asarray(scale)[None]
        assert np.asarray(r).shape == (75,)
        assert np.asarray(jacobian).shape == (75, mean.size)
        expected = -whitened.T @ np.linalg.solve(
            whitened @ whitened.T + np.eye(75), np.asarray(r)
        )
        expected *= min(1, 1 / max(np.linalg.norm(expected), 1e-12))
        np.testing.assert_allclose(np.asarray(z), expected, rtol=1e-10, atol=1e-9)
        np.testing.assert_allclose(
            np.asarray(delta).ravel(), np.asarray(scale) * expected
        )
        improving = np.flatnonzero(np.asarray(trials) < float(baseline))
        assert float(multiplier) == (
            0 if not len(improving) else (1, 0.5, 0.25, 0.125)[improving[0]]
        )
        np.testing.assert_allclose(
            np.asarray(head), np.asarray(mean + multiplier * delta)
        )
        assert np.linalg.norm(np.asarray(z)) <= 1 + 1e-10


def test_causal_cursor_rejects_duplicate_observation():
    name, begin, first, dt = "fixedwing-80", 0, 15, 0.05
    tape = arrays(ROOT / "online-fit-v8/evaluation/inputs/fixedwing-80.npz")
    states, commands = observed(tape["states"]), tape["commands"]
    prefix = SequenceCollection(
        (
            SequenceSegment(
                name,
                "prefix",
                states[begin : first + 1],
                commands[begin:first],
                dt,
                begin,
            ),
        ),
        name,
        STATE_CHANNELS,
        tuple(f"command_{i} [1]" for i in range(3)),
    )
    fit = TrajectoryPathReadout(prefix)
    fit.observe(first, commands[first], states[first + 1])
    assert fit.session.cursor == first + 1
    with pytest.raises(Exception, match="noncausal update"):
        fit.observe(first, commands[first], states[first + 1])
