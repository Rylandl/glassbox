import jax
import jax.numpy as jnp
import numpy as np
import pytest
from screen_cold_readout_curvature import CurvedReadout, readout_features
from screen_readout_sensitivity import (
    ETA,
    SensitivityReadout,
    feature_jacobian,
    numpy_jacobian,
    sensitivity_bases,
    sensitivity_solve,
)
from test_dynamics import constant_model
from test_online import prefix, stream


@pytest.mark.parametrize("commands,delay", [(3, 2), (4, 10)])
def test_analytic_features_match_independent_autodiff_and_finite_difference(
    commands, delay
):
    model = constant_model(commands, history=delay + 1, delay=delay)
    rng = np.random.default_rng(41)
    params = {k: rng.normal(size=v.shape) * 0.1 for k, v in model.params.items()}
    norms = {k: v.copy() for k, v in model.norms.items()}
    for key in ("feature_scale", "quadratic_scale", "nonlinear_scale"):
        norms[key] = np.geomspace(0.3, 3, len(norms[key]))
    c = 9 + 2 * commands
    current, history, hidden = (
        rng.normal(size=c),
        rng.normal(size=(delay, c)),
        rng.normal(size=2),
    )
    with jax.enable_x64(True):
        current, history, hidden = map(jnp.asarray, (current, history, hidden))
        bases = tuple(jnp.asarray(v) for v in sensitivity_bases(params, norms, delay))

        def fun(delta):
            delta = delta.reshape(delay + 1, 6)
            now = current.at[:6].add(delta[0])
            past = history.at[:, :6].add(delta[1:])
            return readout_features(params, norms, now, past, hidden)

        zero = jnp.zeros(6 * (delay + 1))
        phi = fun(zero)
        actual = np.asarray(feature_jacobian(phi, norms, *bases))
        expected = np.asarray(jax.jacfwd(fun)(zero))
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
        initial = {
            **{"param_" + k: np.asarray(v) for k, v in params.items()},
            **{"norm_" + k: np.asarray(v) for k, v in norms.items()},
        }
        np.testing.assert_allclose(
            actual,
            numpy_jacobian(np.asarray(phi), initial, delay),
            atol=1e-12,
            rtol=1e-12,
        )
        direction = rng.normal(size=zero.shape)
        finite = np.asarray(
            (fun(zero + 1e-6 * direction) - fun(zero - 1e-6 * direction)) / 2e-6
        )
        np.testing.assert_allclose(actual @ direction, finite, atol=1e-8, rtol=1e-8)
        # A current-only perturbation enters every lag-difference row negatively.
        np.testing.assert_allclose(
            actual[c : c + 6, :6], -np.diag(1 / norms["feature_scale"][c : c + 6])
        )
        np.testing.assert_allclose(
            actual[c : c + 6, 6:12], np.diag(1 / norms["feature_scale"][c : c + 6])
        )


def test_sensitivity_solve_matches_augmented_batch_least_squares():
    rng = np.random.default_rng(128)
    size, steps = 12, 8
    phi = rng.normal(size=(steps, size)) * np.geomspace(0.1, 10, size)
    target = rng.normal(size=(steps, 6))
    derivatives = rng.normal(size=(steps, size, 5))
    mean0 = rng.normal(size=(size, 6))
    ridge = 0.25
    with jax.enable_x64(True):
        gram = jnp.eye(size) * ridge
        rhs = jnp.asarray(mean0) * ridge
        sensitivity = jnp.zeros_like(gram)
        for i, (x, y, d) in enumerate(zip(phi, target, derivatives), 1):
            penalty = np.arange(size) * 0.01 * i
            gram, rhs, sensitivity, actual = sensitivity_solve(
                gram, rhs, sensitivity, x, y, penalty, d
            )
            design = np.concatenate(
                (
                    phi[:i],
                    np.eye(size) * np.sqrt(ridge),
                    np.diag(np.sqrt(penalty)),
                    np.sqrt(ETA) * derivatives[:i].transpose(0, 2, 1).reshape(-1, size),
                )
            )
            targets = np.concatenate(
                (target[:i], np.sqrt(ridge) * mean0, np.zeros((size + i * 5, 6)))
            )
            expected = np.linalg.lstsq(design, targets, rcond=None)[0]
            np.testing.assert_allclose(actual, expected, atol=1e-11, rtol=1e-10)
        np.testing.assert_allclose(
            sensitivity, np.einsum("tpk,tqk->pq", derivatives, derivatives), atol=1e-12
        )
        np.testing.assert_allclose(gram, ridge * np.eye(size) + phi.T @ phi, atol=1e-12)


@pytest.mark.parametrize("commands,dt", [(3, 0.05), (4, 0.01)])
def test_causal_update_only_changes_readout_and_keeps_measurement_identical(
    commands, dt
):
    states, inputs = stream(commands, dt=dt)
    recording = prefix(states, inputs, offset=0, dt=dt)
    candidate, control = SensitivityReadout(recording), CurvedReadout(recording)
    before = candidate.session.model.arrays()
    row = candidate.session.cursor
    result = candidate.observe(row, inputs[row], states[row + 1])
    reference = control.observe(row, inputs[row], states[row + 1])
    for a, b in zip(result[1:], reference[1:]):
        np.testing.assert_array_equal(a, b)
    after = candidate.session.model.arrays()
    for key, value in before.items():
        if key not in ("param_linear", "param_quadratic", "param_bias", "param_w2"):
            np.testing.assert_array_equal(after[key], value)
    derivative = numpy_jacobian(result[1], before, candidate.session.model.delay_steps)
    np.testing.assert_allclose(
        candidate.sensitivity, derivative @ derivative.T, atol=1e-10, rtol=1e-12
    )
    assert not np.array_equal(result[0], reference[0])
    with pytest.raises(ValueError, match="noncausal"):
        candidate.observe(row, inputs[row], states[row + 1])
