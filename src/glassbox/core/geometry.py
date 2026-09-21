"""Rigid-body signal conversion and a public prediction-to-motion adapter."""

import jax
import jax.numpy as jnp
from jax import Array


def quaternion_to_rotation_batch(quaternion_wxyz: Array) -> Array:
    """Return body-to-world rotations for a batch of non-unit quaternions.

    The batched, normalizing JAX counterpart of
    :func:`glassbox.core.dynamics.quaternion_to_rotation`.  It is written as a
    single stacked expression rather than a ``vmap`` of the scalar version so
    the traced graph, and therefore every fitted number, is unchanged.
    """

    quaternion = quaternion_wxyz / jnp.linalg.norm(
        quaternion_wxyz, axis=-1, keepdims=True
    )
    w, x, y, z = jnp.moveaxis(quaternion, -1, 0)
    return jnp.stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape((-1, 3, 3))


def rotation_to_quaternion(rotation: Array) -> Array:
    """Return a unit WXYZ quaternion for one body-to-world rotation matrix.

    Shepperd's method: each of the four expressions below is proportional to
    the quaternion, and the one whose leading term is largest is the numerically
    best conditioned, so it is the branch taken. Sign is not fixed, because a
    quaternion and its negation represent the same rotation.
    """

    matrix = jnp.asarray(rotation)
    if matrix.shape != (3, 3):
        raise ValueError("a quaternion recovery needs one 3x3 rotation matrix")
    xx, yy, zz = matrix[0, 0], matrix[1, 1], matrix[2, 2]
    leading = jnp.stack(
        (1.0 + xx + yy + zz, 1.0 + xx - yy - zz, 1.0 - xx + yy - zz, 1.0 - xx - yy + zz)
    )
    symmetric = (
        matrix[1, 0] + matrix[0, 1],
        matrix[0, 2] + matrix[2, 0],
        matrix[2, 1] + matrix[1, 2],
    )
    antisymmetric = (
        matrix[2, 1] - matrix[1, 2],
        matrix[0, 2] - matrix[2, 0],
        matrix[1, 0] - matrix[0, 1],
    )
    candidates = jnp.stack(
        (
            jnp.stack(
                (leading[0], antisymmetric[0], antisymmetric[1], antisymmetric[2])
            ),
            jnp.stack((antisymmetric[0], leading[1], symmetric[0], symmetric[1])),
            jnp.stack((antisymmetric[1], symmetric[0], leading[2], symmetric[2])),
            jnp.stack((antisymmetric[2], symmetric[1], symmetric[2], leading[3])),
        )
    )
    quaternion = jnp.take(candidates, jnp.argmax(leading), axis=0)
    return quaternion / jnp.maximum(jnp.linalg.norm(quaternion), 1e-12)


def motion_rollout(model, initial_state, past_observations, past_commands, commands):
    """Integrate public predicted motion into position/WXYZ rigid-body states.

    ``initial_state`` is position (world), velocity (world), WXYZ quaternion,
    and angular velocity (body), with shape ``(13,)``. Predictions use Glassbox's
    canonical 15 signals: world velocity, body rate, body-to-world rotation.
    The returned ``(len(commands) + 1, 13)`` array includes the initial state.
    Position uses trapezoidal integration; rotations come directly from the
    model. This adapter does not add or fit a second dynamics model.
    """
    state = jnp.asarray(initial_state)
    if state.shape != (13,):
        raise ValueError("initial_state must have shape (13,)")
    predicted = model.predict(past_observations, past_commands, commands)
    rotations = predicted[:, 6:15].reshape((-1, 3, 3))
    proper = jnp.all(jnp.linalg.det(rotations) > 0) & jnp.all(jnp.isfinite(predicted))
    velocity = predicted[:, :3]
    speeds = jnp.concatenate((state[None, 3:6], velocity))
    positions = state[:3] + model.dt_s * jnp.cumsum(
        0.5 * (speeds[:-1] + speeds[1:]), axis=0
    )
    quaternions = jax.vmap(rotation_to_quaternion)(rotations)
    future = jnp.concatenate(
        (positions, velocity, quaternions, predicted[:, 3:6]), axis=1
    )
    states = jnp.concatenate((state[None], future))
    return jnp.where(proper, states, jnp.full_like(states, jnp.nan))
