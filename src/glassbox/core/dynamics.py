"""Differentiable quaternion operations for rigid-body consumers."""

import jax.numpy as jnp
from jax import Array


def quaternion_multiply(left: Array, right: Array) -> Array:
    """Multiply WXYZ quaternions."""

    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return jnp.asarray(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ]
    )


def quaternion_to_rotation(quaternion_wxyz: Array) -> Array:
    """Return the body-to-world rotation matrix for a unit quaternion."""

    w, x, y, z = quaternion_wxyz
    return jnp.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ]
    )
