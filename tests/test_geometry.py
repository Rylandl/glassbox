"""Retraction values and derivatives in rigid-body local coordinates."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from glassbox.core.geometry import rigid_body_local_error, state_plus_tangent
from glassbox.core.synthetic import resting_state


@pytest.mark.parametrize("derivative", [jax.jacfwd, jax.jacrev])
@pytest.mark.parametrize("quaternion", [(1.0, 0.0, 0.0, 0.0), (0.5, 0.5, -0.5, 0.5)])
def test_retraction_has_the_expected_jacobian_at_zero(derivative, quaternion):
    state = jnp.asarray(resting_state()).at[6:10].set(jnp.asarray(quaternion))

    def retract(tangent):
        return state_plus_tangent(state, tangent)

    jacobian = np.asarray(jax.jit(derivative(retract))(jnp.zeros(12)))
    w, x, y, z = quaternion
    expected = np.zeros((13, 12))
    expected[:6, :6] = np.eye(6)
    expected[10:13, 9:12] = np.eye(3)
    expected[6:10, 6:9] = 0.5 * np.asarray(
        [[-x, -y, -z], [w, -z, y], [z, w, -x], [-y, x, w]]
    )
    np.testing.assert_allclose(jacobian, expected, atol=1e-7)
    np.testing.assert_allclose(retract(jnp.zeros(12)), state, atol=1e-7)


def test_retraction_has_finite_second_derivatives_at_zero():
    state = jnp.asarray(resting_state())
    hessian = jax.jit(
        jax.hessian(lambda tangent: state_plus_tangent(state, tangent)[6])
    )(jnp.zeros(12))
    expected = np.zeros((12, 12))
    expected[6:9, 6:9] = -0.25 * np.eye(3)
    np.testing.assert_allclose(hessian, expected, atol=1e-7)


@pytest.mark.parametrize("angle", [0.0, 1e-7, 0.000099, 0.000101, 0.4, 2.0])
def test_retraction_and_local_error_are_differentiable_inverses(angle):
    state = (
        jnp.asarray(resting_state()).at[6:10].set(jnp.asarray([0.5, 0.5, -0.5, 0.5]))
    )
    tangent = jnp.arange(12, dtype=jnp.float32) / 20.0
    tangent = tangent.at[6:9].set(jnp.asarray([0.6, 0.0, 0.8]) * angle)

    def round_trip(offset):
        return rigid_body_local_error(state, state_plus_tangent(state, offset))

    np.testing.assert_allclose(round_trip(tangent), tangent, atol=3e-7)
    np.testing.assert_allclose(jax.jacfwd(round_trip)(tangent), np.eye(12), atol=3e-7)
