"""Differentiable local geometry for rigid-body prediction and control."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from glassbox.dynamics import quaternion_multiply


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
