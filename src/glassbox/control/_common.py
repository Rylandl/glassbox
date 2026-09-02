"""Shared, private helpers for the bootstrap and supervision controllers.

Every function here previously existed as a byte-for-byte copy in two or more
of :mod:`glassbox.control.bootstrap_identification`,
:mod:`glassbox.control.online_bootstrap` and
:mod:`glassbox.control.flight_supervisor`.  The implementations are kept
exactly as they were, in the same operation order, so consolidating them
changes no number anywhere.

Nothing here is part of the public API.  Two of these helpers are geometry
rather than control and belong in :mod:`glassbox.core.geometry` once that
module gains its NumPy rotation support: :func:`quaternion_to_rotation` and
:func:`quaternion_to_rotation_batch`, with :func:`world_up_body` a reasonable
third.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np
from jax import Array

from glassbox.core.dynamics import GRAVITY_M_S2


def finite_vector(
    name: str,
    values: float | Sequence[float],
    size: int,
) -> np.ndarray:
    """Broadcast a scalar or sequence to a validated float64 vector.

    A scalar is repeated to ``size`` entries.  Anything that is not exactly
    ``size`` finite values is refused by name, so a configuration mistake is
    reported where it is made rather than as a shape error deep in a solve.
    """

    if np.isscalar(values):
        result = np.full(size, float(values), dtype=np.float64)
    else:
        result = np.asarray(tuple(values), dtype=np.float64)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain {size} finite values")
    return result


def finite_tuple(
    name: str,
    values: float | Sequence[float],
    size: int,
) -> tuple[float, ...]:
    """Return :func:`finite_vector` as a hashable tuple of Python floats.

    Frozen configuration dataclasses store their validated vectors as tuples so
    they stay comparable and hashable; the arithmetic is identical.
    """

    return tuple(float(value) for value in finite_vector(name, values, size))


def immutable_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    """Return a validated, read-only float64 copy of one array field.

    Copying then freezing means a frozen result dataclass cannot be mutated
    through the array the caller handed it, and a wrong shape or a non-finite
    entry is refused by name at construction.
    """

    result = np.asarray(value, dtype=np.float64).copy()
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must have shape {shape} and contain finite values")
    result.flags.writeable = False
    return result


def quaternion_to_rotation(quaternion_wxyz: np.ndarray) -> np.ndarray:
    """Return the NumPy body-to-world rotation for one non-unit quaternion.

    This is the NumPy mirror of :func:`glassbox.core.dynamics.quaternion_to_rotation`,
    with the normalization the JAX version leaves to its caller folded in. It
    deliberately keeps the input dtype: simulator plants hand this float32
    attitudes, and the recorded closed-loop diagnostics are pinned to the
    float32 arithmetic. :func:`glassbox.core.geometry.quaternion_to_rotation_matrices`
    computes the same expressions in float64 and is the offline helper.
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


@dataclass(frozen=True)
class ThrustCascade:
    """One evaluation of the velocity-to-angular-acceleration cascade."""

    desired_world_acceleration_m_s2: np.ndarray
    desired_specific_force_m_s2: np.ndarray
    desired_force_magnitude_m_s2: float
    desired_thrust_direction_world: np.ndarray
    tilt_error_body: np.ndarray
    desired_angular_acceleration_rad_s2: np.ndarray


def thrust_cascade(
    *,
    world_velocity_m_s: np.ndarray,
    rotation: np.ndarray,
    angular_velocity_rad_s: np.ndarray,
    velocity_gain: np.ndarray,
    maximum_world_acceleration_m_s2: np.ndarray,
    maximum_tilt_rad: float,
    attitude_gain: np.ndarray,
    angular_rate_gain: np.ndarray,
    velocity_authority: float = 1.0,
) -> ThrustCascade:
    """Map a world velocity error to a desired body angular acceleration.

    The chain is the one both bootstrap controllers use: proportional velocity
    arrest, saturated to a bounded world acceleration; gravity added to get the
    specific force the vehicle must produce; the horizontal part squeezed until
    the implied tilt is within ``maximum_tilt_rad``; the resulting thrust
    direction rotated into the body frame; and the small-angle tilt error fed
    through attitude and rate gains, with yaw driven by rate damping alone
    because a thrust direction says nothing about heading.

    ``velocity_authority`` is the one switch between the two callers.  The
    batch identifier commands full velocity arrest and leaves it at the default
    ``1.0``; the online controller passes the belief's weakest relevant
    authority so an unsupported model cannot ask for a large arrest.  It scales
    the proportional gain before saturation, so the acceleration limit and the
    tilt limit still bound the result the same way.
    """

    desired_world_acceleration = np.clip(
        -velocity_authority * velocity_gain * world_velocity_m_s,
        -maximum_world_acceleration_m_s2,
        maximum_world_acceleration_m_s2,
    )
    desired_specific_force = desired_world_acceleration + np.asarray(
        (0.0, 0.0, GRAVITY_M_S2)
    )
    vertical_force = max(float(desired_specific_force[2]), 1e-3)
    maximum_horizontal_force = vertical_force * math.tan(maximum_tilt_rad)
    horizontal_norm = float(np.linalg.norm(desired_specific_force[:2]))
    if horizontal_norm > maximum_horizontal_force:
        desired_specific_force[:2] *= maximum_horizontal_force / horizontal_norm
    desired_force_magnitude = float(np.linalg.norm(desired_specific_force))
    desired_thrust_direction = desired_specific_force / desired_force_magnitude
    desired_thrust_body = rotation.T @ desired_thrust_direction
    tilt_error_body = np.cross(
        np.asarray((0.0, 0.0, 1.0)),
        desired_thrust_body,
    )
    desired_angular_acceleration = np.asarray(
        (
            attitude_gain[0] * tilt_error_body[0]
            - angular_rate_gain[0] * angular_velocity_rad_s[0],
            attitude_gain[1] * tilt_error_body[1]
            - angular_rate_gain[1] * angular_velocity_rad_s[1],
            -angular_rate_gain[2] * angular_velocity_rad_s[2],
        )
    )
    return ThrustCascade(
        desired_world_acceleration_m_s2=(
            desired_specific_force - np.asarray((0.0, 0.0, GRAVITY_M_S2))
        ),
        desired_specific_force_m_s2=desired_specific_force,
        desired_force_magnitude_m_s2=desired_force_magnitude,
        desired_thrust_direction_world=desired_thrust_direction,
        tilt_error_body=tilt_error_body,
        desired_angular_acceleration_rad_s2=desired_angular_acceleration,
    )
