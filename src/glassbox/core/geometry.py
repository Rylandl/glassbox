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

TANGENT_STATE_SIZE = 12
TANGENT_STATE_ORDER = (
    "position_x",
    "position_y",
    "position_z",
    "velocity_x",
    "velocity_y",
    "velocity_z",
    "attitude_x",
    "attitude_y",
    "attitude_z",
    "angular_velocity_x",
    "angular_velocity_y",
    "angular_velocity_z",
)
TANGENT_GROUP_ORDER = (
    "position",
    "velocity",
    "attitude",
    "angular_velocity",
)
TANGENT_GROUP_INDICES = {
    "position": (0, 1, 2),
    "velocity": (3, 4, 5),
    "attitude": (6, 7, 8),
    "angular_velocity": (9, 10, 11),
}


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


def state_plus_tangent(state: Array, tangent: Array) -> Array:
    """Move one rigid-body state along a local twelve-vector.

    This is the retraction inverse to :func:`rigid_body_local_error`: the
    position, velocity and body-rate blocks add, and the attitude block is
    applied as a rotation about the body, so the quaternion stays on the unit
    sphere instead of acquiring a Euclidean displacement. Composing the two
    recovers the tangent exactly, which is what makes a tangent-space error, a
    tangent-space perturbation, and a tangent-space covariance describe the
    same thing.
    """

    angle = jnp.linalg.norm(tangent[6:9])
    quaternion_scale = 0.5 * jnp.sinc(angle / (2.0 * jnp.pi))
    delta_quaternion = jnp.concatenate(
        (
            jnp.cos(0.5 * angle)[None],
            quaternion_scale * tangent[6:9],
        )
    )
    quaternion = quaternion_multiply(state[6:10], delta_quaternion)
    quaternion /= jnp.maximum(jnp.linalg.norm(quaternion), 1e-12)
    return jnp.concatenate(
        (
            state[0:3] + tangent[0:3],
            state[3:6] + tangent[3:6],
            quaternion,
            state[10:13] + tangent[9:12],
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
    :func:`quaternion_to_rotation` in this module stays separate because
    it normalizes with ``np.linalg.norm(q)`` rather than an ``axis=-1``
    reduction; the last-ulp difference is amplified by the recorded closed-loop
    diagnostics, which are pinned at 1e-6.
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


def quaternion_to_rotation(quaternion_wxyz: np.ndarray) -> np.ndarray:
    """Return the NumPy body-to-world rotation for one non-unit quaternion.

    This is the NumPy mirror of :func:`glassbox.core.dynamics.quaternion_to_rotation`,
    with the normalization the JAX version leaves to its caller folded in. It
    is kept separate from
    :func:`glassbox.core.geometry.quaternion_to_rotation_matrices` on purpose:
    this version normalizes with ``np.linalg.norm(q)`` while the batched helper
    reduces along ``axis=-1``, and the two differ in the last ulp for roughly
    one quaternion in seven. That perturbation sits far below every test
    tolerance but the recorded closed-loop Crazyflow diagnostics amplify it over
    hundreds of steps, so swapping helpers silently moves pinned numbers.
    """

    quaternion = quaternion_wxyz / np.linalg.norm(quaternion_wxyz)
    w, x, y, z = quaternion
    return np.asarray(
        (
            (
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ),
            (
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ),
            (
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ),
        )
    )


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


def world_up_body(unit_quaternion_wxyz: np.ndarray) -> np.ndarray:
    """Return world up expressed in the body frame of a unit quaternion.

    This is the third row of the body-to-world rotation, written out directly
    so no full matrix is built for the one column that is needed.  The
    quaternion must already be normalized; callers own that step because they
    differ in how they reject a degenerate norm.
    """

    w, x, y, z = unit_quaternion_wxyz
    return np.asarray(
        (
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        )
    )
