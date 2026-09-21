"""Physical signal geometry and the public consumer boundary."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from glassbox.core.dynamics import quaternion_multiply, quaternion_to_rotation
from glassbox.core.geometry import (
    motion_rollout,
    quaternion_to_rotation_batch,
    rotation_to_quaternion,
)


@pytest.mark.parametrize(
    "q",
    [
        [1.0, 0, 0, 0],
        [0, 1.0, 0, 0],
        [0, 0, 1.0, 0],
        [0, 0, 0, 1.0],
        [0.5, -0.5, 0.5, -0.5],
    ],
)
def test_rotation_roundtrip(q):
    q = jnp.asarray(q)
    r = quaternion_to_rotation(q)
    recovered = rotation_to_quaternion(r)
    np.testing.assert_allclose(quaternion_to_rotation(recovered), r, atol=1e-6)
    np.testing.assert_allclose(
        quaternion_to_rotation_batch((3 * q)[None])[0], r, atol=1e-6
    )
    np.testing.assert_allclose(
        quaternion_multiply(q, q * jnp.array([1.0, -1, -1, -1])),
        jnp.array([1.0, 0, 0, 0]),
        atol=1e-6,
    )


class PublicPrediction:
    dt_s = 0.1

    def predict(self, past_states, past_commands, commands):
        assert past_states.shape == (3, 15)
        assert past_commands.shape == (2, 2)
        velocity = jnp.pad(commands[:, :1], ((0, 0), (0, 2)))
        return jnp.concatenate(
            (
                velocity,
                jnp.zeros((len(commands), 3)),
                jnp.tile(jnp.eye(3).reshape(1, 9), (len(commands), 1)),
            ),
            axis=1,
        )


def test_motion_adapter_uses_public_prediction_and_trapezoidal_position():
    model = PublicPrediction()
    state = jnp.array([0.0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    past, issued = jnp.zeros((3, 15)), jnp.zeros((2, 2))
    command = jnp.array([[1.0, 0], [2.0, 0]])
    predicted = motion_rollout(model, state, past, issued, command)
    assert predicted.shape == (3, 13)
    np.testing.assert_array_equal(predicted[0], state)
    np.testing.assert_allclose(predicted[:, 0], [0, 0.05, 0.2])
    np.testing.assert_array_equal(predicted[:, 6:10], np.tile([1, 0, 0, 0], (3, 1)))
    derivative = jax.grad(
        lambda c: motion_rollout(model, state, past, issued, c)[-1, 0]
    )(command)
    np.testing.assert_allclose(derivative, [[0.1, 0], [0.05, 0]])


def test_motion_adapter_rejects_wrong_state_shape():
    with pytest.raises(ValueError, match="initial_state"):
        motion_rollout(
            PublicPrediction(),
            jnp.zeros(17),
            jnp.zeros((3, 15)),
            jnp.zeros((2, 2)),
            jnp.zeros((2, 2)),
        )


def test_motion_adapter_marks_invalid_rotation_nonfinite():
    class BadPrediction(PublicPrediction):
        def predict(self, *args):
            return super().predict(*args).at[:, 6].set(-1)

    state = jnp.array([0.0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    result = motion_rollout(
        BadPrediction(), state, jnp.zeros((3, 15)), jnp.zeros((2, 2)), jnp.zeros((2, 2))
    )
    assert np.isnan(result).all()
