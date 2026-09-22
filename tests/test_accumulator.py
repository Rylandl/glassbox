"""Independent sequential/closed-form memory and derivative qualification."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree

from glassbox._dynamics import accumulated_memory, memory_step


def fixture(channels=4, steps=40):
    rng = np.random.default_rng(916)
    current = 9 + 2 * channels
    params = dict(
        memory=rng.normal(0, 0.2, (current, 8)),
        memory_bias=rng.normal(0, 0.1, 8),
        raw_memory_tau=np.log(np.expm1(np.geomspace(0.01, 0.5, 8) - 0.001)),
    )
    norms = dict(feature_scale=np.exp(rng.normal(0, 0.4, current)))
    return rng, params, norms, rng.normal(size=(3, steps, current))


def sequential(params, norms, features, dt):
    tau = 0.001 + jnp.logaddexp(0, params["raw_memory_tau"])
    a = jnp.exp(-dt / tau)
    drive = jnp.tanh(
        (features / norms["feature_scale"]) @ params["memory"] + params["memory_bias"]
    )

    def advance(hidden, signal):
        return a * hidden + (1 - a) * signal, None

    return jax.lax.scan(
        advance, jnp.zeros((len(features), 8), features.dtype), drive.swapaxes(0, 1)
    )[0]


@pytest.mark.parametrize(
    "channels,steps,dt", [(1, 1, 0.2), (3, 8, 0.05), (4, 40, 0.01), (6, 0, 0.01)]
)
def test_accumulator_primal_parameter_and_input_jvp_vjp(channels, steps, dt):
    rng, p, n, x = fixture(channels, steps)
    with jax.enable_x64(True):
        theta, unpack = ravel_pytree((p, jnp.asarray(x)))

        def fn(value):
            params, features = unpack(value)
            return accumulated_memory(params, n, features, dt)

        def ref(value):
            params, features = unpack(value)
            return sequential(params, n, features, dt)

        direction = jnp.asarray(rng.normal(size=theta.shape))
        a = jax.jvp(jax.jit(fn), (theta,), (direction,))
        b = jax.jvp(ref, (theta,), (direction,))
        for actual, expected in zip(a, b, strict=True):
            np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-11)
        cotangent = jnp.asarray(rng.normal(size=(3, 8)))
        np.testing.assert_allclose(
            jax.vjp(fn, theta)[1](cotangent)[0],
            jax.vjp(ref, theta)[1](cotangent)[0],
            rtol=1e-10,
            atol=1e-11,
        )
        assert np.max(np.abs(a[0])) <= 1


def test_constant_drive_split_step_is_exact_and_stable_at_extreme_times():
    rng, p, n, x = fixture(3, 1)
    with jax.enable_x64(True):
        x = jnp.asarray(x[:, 0])
        hidden = jnp.asarray(rng.uniform(-1, 1, (3, 8)))
        split = memory_step(p, n, x, memory_step(p, n, x, hidden, 0.01), 0.04)
        whole = memory_step(p, n, x, hidden, 0.05)
        np.testing.assert_allclose(split, whole, rtol=1e-13, atol=1e-13)
        for raw in (-1000.0, 1000.0):
            changed = dict(p, raw_memory_tau=jnp.full(8, raw))
            prediction = accumulated_memory(
                changed, n, jnp.repeat(x[:, None], 40, axis=1), 0.01
            )
            grad = jax.grad(
                lambda tau, base=changed: jnp.sum(
                    accumulated_memory(
                        dict(base, raw_memory_tau=tau),
                        n,
                        jnp.repeat(x[:, None], 40, axis=1),
                        0.01,
                    )
                )
            )(changed["raw_memory_tau"])
            assert np.isfinite(prediction).all() and np.isfinite(grad).all()
            assert np.max(np.abs(prediction)) <= 1
