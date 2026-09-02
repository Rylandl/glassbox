"""Local geometry for rigid-body prediction, control, and offline analysis.

The JAX entry points are traced inside differentiated code. The NumPy entry
points near the bottom of this module are the batched offline equivalents used
by identification, evaluation, and the benchmark workflows;
:func:`glassbox.core.dynamics.quaternion_to_rotation` remains the canonical
traced single-quaternion rotation.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from jax import Array

from glassbox.core.dynamics import quaternion_multiply


def quaternion_log_error(reference_wxyz: Array, actual_wxyz: Array) -> Array:
    """Return the shortest reference-to-actual rotation vector."""

    reference = reference_wxyz / jnp.maximum(jnp.linalg.norm(reference_wxyz), 1e-12)
    actual = actual_wxyz / jnp.maximum(jnp.linalg.norm(actual_wxyz), 1e-12)
    conjugate = reference * jnp.asarray([1.0, -1.0, -1.0, -1.0])
    relative = quaternion_multiply(conjugate, actual)
    relative = relative * jnp.where(relative[0] < 0.0, -1.0, 1.0)
    vector = relative[1:4]
    # The epsilon-smoothed norm preserves the small-angle limit and keeps
    # reverse-mode derivatives finite at the identity rotation.
    vector_norm = jnp.sqrt(jnp.sum(jnp.square(vector)) + 1e-16)
    angle_scale = (
        2.0 * jnp.arctan2(vector_norm, jnp.maximum(relative[0], 0.0)) / vector_norm
    )
    return angle_scale * vector


def rigid_body_local_error(reference: Array, actual: Array) -> Array:
    """Return position, velocity, attitude, and body-rate local error."""

    return jnp.concatenate(
        (
            actual[0:3] - reference[0:3],
            actual[3:6] - reference[3:6],
            quaternion_log_error(reference[6:10], actual[6:10]),
            actual[10:13] - reference[10:13],
        )
    )


def quaternion_to_rotation_matrices(
    quaternion_wxyz: npt.ArrayLike,
    *,
    normalize: bool = True,
) -> npt.NDArray[np.float64]:
    """Return batched body-to-world rotation matrices for WXYZ quaternions.

    ``quaternion_wxyz`` has shape ``(..., 4)`` and the result has shape
    ``(..., 3, 3)``. ``normalize`` divides by the quaternion norm first, which
    every current caller wants because logged and integrated attitudes drift
    off the unit sphere. Pass ``False`` only when the input is already unit
    length and the extra division would be pure rounding noise.

    Use :func:`glassbox.core.dynamics.quaternion_to_rotation` instead inside
    traced JAX code; this function is the offline NumPy equivalent.
    It computes in float64; ``glassbox.control._common.quaternion_to_rotation``
    keeps the input dtype because recorded closed-loop diagnostics depend on
    float32 arithmetic there.
    """

    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
    if quaternion.shape[-1] != 4:
        raise ValueError("quaternion arrays must have a trailing WXYZ axis")
    if normalize:
        quaternion = quaternion / np.linalg.norm(quaternion, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(quaternion, -1, 0)
    return np.stack(
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
    ).reshape(quaternion.shape[:-1] + (3, 3))


def quaternion_from_euler(
    roll: npt.ArrayLike,
    pitch: npt.ArrayLike,
    yaw: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Return the WXYZ quaternion of an intrinsic yaw-pitch-roll rotation.

    Scalar angles give a ``(4,)`` result and equal-shaped angle arrays give a
    ``(..., 4)`` result, so one implementation serves both the single reference
    attitudes and the whole-trajectory reference builders.
    """

    half_roll = 0.5 * np.asarray(roll, dtype=np.float64)
    half_pitch = 0.5 * np.asarray(pitch, dtype=np.float64)
    half_yaw = 0.5 * np.asarray(yaw, dtype=np.float64)
    cr, sr = np.cos(half_roll), np.sin(half_roll)
    cp, sp = np.cos(half_pitch), np.sin(half_pitch)
    cy, sy = np.cos(half_yaw), np.sin(half_yaw)
    return np.stack(
        (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ),
        axis=-1,
    )
