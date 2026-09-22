"""Independent calculus checks for the one-off physical attitude experiment."""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from screen_cold_readout_curvature import readout_features
from screen_physical_so3 import (
    attitude_derivative,
    attitude_features,
    midpoint_context,
)
from verify_baseline import arrays, observed

from glassbox import (
    STATE_CHANNELS,
    OnlineFit,
    SequenceCollection,
    SequenceSegment,
)
from glassbox._dynamics import (
    current_features,
    rotation_exp,
    sampled_features,
)

ROOT = Path("/Users/ryland/autonomy/glassbox/artifacts")


@pytest.mark.parametrize(
    "case,recording,begin,first,dt,commands",
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
        (
            "paired-quad-coarse",
            ROOT / "paired-quad-sampling-v2/coarse.npz",
            10,
            25,
            0.05,
            4,
        ),
    ],
)
def test_physical_attitude_feature_derivative(
    case, recording, begin, first, dt, commands
):
    tape = arrays(recording)
    states = observed(tape["states"])
    issued = tape["commands"].astype(np.float64)
    prefix = SequenceCollection(
        (
            SequenceSegment(
                case,
                "prefix",
                states[begin : first + 1],
                issued[begin:first],
                dt,
                begin,
            ),
        ),
        case,
        STATE_CHANNELS,
        tuple(f"command_{i} [1]" for i in range(commands)),
    )
    model = OnlineFit(prefix).model
    assert model.delay_steps == (10 if dt == 0.01 else 2)
    h = model.history_steps
    row = first
    with jax.enable_x64(True):
        params, norms = jax.tree.map(jnp.asarray, (model.params, model.norms))
        past = jnp.asarray(states[row - h : row + 1])
        inputs = jnp.asarray(issued[row - h : row])
        command = jnp.asarray(issued[row])
        following = jnp.asarray(states[row + 1])
        derivative = np.asarray(
            attitude_derivative(
                params, norms, past, inputs, command, following, model.delay_steps, dt
            )
        )
        midpoint, rotation, filtered, history, hidden = midpoint_context(
            params, norms, past, inputs, command, following, model.delay_steps, dt
        )

        def current(theta):
            changed = midpoint.at[6:].set((rotation @ rotation_exp(theta)).reshape(9))
            return current_features(changed, command, filtered, norms)

        zero = jnp.zeros(3, dtype=past.dtype)
        dcurrent = np.asarray(jax.jacfwd(current)(zero))
        dhead = np.asarray(
            jax.jacfwd(
                lambda value: readout_features(params, norms, value, history, hidden)
            )(current(zero))
        )
        np.testing.assert_allclose(derivative, dhead @ dcurrent, rtol=2e-10, atol=2e-10)
        np.testing.assert_allclose(dcurrent[3:6], 0, atol=1e-12)
        np.testing.assert_allclose(dcurrent[9:], 0, atol=1e-12)
        assert np.linalg.norm(dcurrent[:3]) > 0
        assert np.linalg.norm(dcurrent[6:9]) > 0
        assert np.linalg.norm(dhead[:, 6:9] @ dcurrent[6:9]) > 0

        sampled = np.asarray(
            jax.jacfwd(lambda theta: sampled_features(current(theta), history, hidden))(
                zero
            )
        )
        current_width = len(current(zero))
        np.testing.assert_allclose(sampled[:current_width], dcurrent, atol=1e-12)
        np.testing.assert_allclose(
            sampled[current_width : (model.delay_steps + 1) * current_width].reshape(
                model.delay_steps, current_width, 3
            ),
            -np.broadcast_to(dcurrent, (model.delay_steps, current_width, 3)),
            atol=1e-12,
        )

        # The narrow frozen tanh features can change appreciably over 1e-5 rad.
        epsilon = 1e-7
        differences = np.stack(
            [
                (
                    np.asarray(
                        attitude_features(
                            params,
                            norms,
                            past,
                            inputs,
                            command,
                            following,
                            jnp.eye(3)[axis] * epsilon,
                            model.delay_steps,
                            dt,
                        )
                    )
                    - np.asarray(
                        attitude_features(
                            params,
                            norms,
                            past,
                            inputs,
                            command,
                            following,
                            -jnp.eye(3)[axis] * epsilon,
                            model.delay_steps,
                            dt,
                        )
                    )
                )
                / (2 * epsilon)
                for axis in range(3)
            ],
            axis=-1,
        )
        np.testing.assert_allclose(derivative, differences, rtol=2e-5, atol=2e-6)
