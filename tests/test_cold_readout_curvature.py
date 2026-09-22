import jax
import jax.numpy as jnp
import numpy as np
import pytest
from screen_cold_readout_curvature import (
    CurvedReadout,
    RawReadout,
    coefficients,
    curvature_penalty,
    information_update,
    numpy_penalty,
    readout_features,
    recursive_update,
)
from test_dynamics import constant_model
from test_online import conditioning_fixture, prefix, stream

from glassbox import _dynamics as core
from glassbox import online


@pytest.mark.parametrize("commands,delay", [(3, 2), (4, 10)])
def test_readout_reconstructs_all_acceleration_paths(commands, delay):
    model = constant_model(commands, history=delay + 1, delay=delay)
    rng = np.random.default_rng(851)
    params = {k: rng.normal(size=v.shape) * 0.1 for k, v in model.params.items()}
    state = np.r_[rng.normal(size=6) * 0.2, np.eye(3).ravel()]
    command = rng.normal(size=commands) * 0.1
    history = rng.normal(size=(delay, 9 + 2 * commands)) * 0.1
    hidden = rng.normal(size=2) * 0.1
    with jax.enable_x64(True):
        physical, current, _ = core._head(
            params,
            model.norms,
            jnp.asarray(state),
            jnp.asarray(command),
            jnp.asarray(command),
            jnp.asarray(history),
            jnp.asarray(hidden),
        )
        actual = readout_features(
            params, model.norms, current, jnp.asarray(history), jnp.asarray(hidden)
        ) @ coefficients(params)
        np.testing.assert_allclose(
            actual * model.norms["output_scale"], physical, rtol=1e-12, atol=1e-12
        )


def test_recursive_estimator_matches_independent_regularized_batch_solution():
    rng = np.random.default_rng(12)
    x = rng.normal(size=(30, 12))
    y = rng.normal(size=(30, 6))
    prior = rng.normal(size=(12, 6))
    ridge = 0.25
    with jax.enable_x64(True):
        inverse, mean = jnp.eye(12) / ridge, jnp.asarray(prior)
        for row in range(len(x)):
            inverse, mean = recursive_update(inverse, mean, x[row], y[row])
        precision = ridge * np.eye(12) + x.T @ x
        expected = np.linalg.solve(precision, ridge * prior + x.T @ y)
        np.testing.assert_allclose(mean, expected, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(
            inverse, np.linalg.inv(precision), rtol=1e-10, atol=1e-10
        )


@pytest.mark.parametrize("commands,dt", [(3, 0.05), (4, 0.01)])
@pytest.mark.parametrize("kind", [RawReadout, CurvedReadout])
def test_fresh_update_preserves_features_and_rejects_duplicate_row(commands, dt, kind):
    states, inputs = stream(commands, dt=dt)
    candidate = kind(prefix(states, inputs, offset=0, dt=dt))
    before = candidate.session.model.arrays()
    row = candidate.session.cursor
    result = candidate.observe(row, inputs[row], states[row + 1])
    after = candidate.session.model.arrays()
    if kind is CurvedReadout:
        info = dict(
            first=row,
            initial_model_metadata=candidate.session.model.metadata(),
            baseline_report=candidate.session.report,
            feature_count=len(candidate.mean),
        )
        expected = numpy_penalty(before, info, states, inputs, row)
        np.testing.assert_allclose(result[-1], expected, rtol=2e-12, atol=1e-10)
    for key, value in before.items():
        if key not in ("param_linear", "param_quadratic", "param_bias", "param_w2"):
            np.testing.assert_array_equal(after[key], value)
    with pytest.raises(ValueError, match="noncausal"):
        candidate.observe(row, inputs[row], states[row + 1])


def test_equilibrated_information_solve_matches_augmented_least_squares():
    rng = np.random.default_rng(28)
    x = rng.normal(size=(25, 12)) * np.geomspace(0.001, 1000, 12)
    y, mean0 = rng.normal(size=(25, 6)), rng.normal(size=(12, 6))
    ridge = 0.25
    with jax.enable_x64(True):
        gram, rhs = jnp.eye(12) * ridge, jnp.asarray(mean0) * ridge
        for n, (phi, target) in enumerate(zip(x, y), 1):
            # Changing the penalty may not contaminate the retained data statistics.
            penalty = np.r_[
                np.zeros(4), np.arange(1, 5) * 0.015 * n * (1 + n % 3), np.zeros(4)
            ]
            gram, rhs, actual = information_update(gram, rhs, phi, target, penalty)
            design = np.vstack(
                (x[:n], np.eye(12) * np.sqrt(ridge), np.diag(np.sqrt(penalty)))
            )
            targets = np.vstack((y[:n], np.sqrt(ridge) * mean0, np.zeros_like(mean0)))
            expected = np.linalg.lstsq(design, targets, rcond=None)[0]
            np.testing.assert_allclose(actual, expected, rtol=2e-8, atol=1e-9)
        np.testing.assert_allclose(
            gram, ridge * np.eye(12) + x.T @ x, rtol=2e-13, atol=1e-9
        )
        np.testing.assert_allclose(rhs, ridge * mean0 + x.T @ y, rtol=2e-13, atol=1e-9)


def test_shared_penalty_is_the_existing_physical_hessian_penalty():
    model, bootstrap, recent = conditioning_fixture()
    data, weights = online._full_cache(bootstrap, recent)
    scale = online._scale(bootstrap)
    f, q, n = len(model.params["linear"]), len(model.params["quadratic"]), 19
    with jax.enable_x64(True):
        original = online._curvature_diagonal(
            {"quadratic": model.params["quadratic"]},
            model.norms,
            data,
            scale,
            weights,
            delay=model.delay_steps,
            dt_s=model.dt_s,
        ).reshape((-1, 6))
        penalty = np.asarray(
            curvature_penalty(
                model.params,
                model.norms,
                data,
                scale,
                weights,
                n,
                delay=model.delay_steps,
                dt_s=model.dt_s,
            )
        )
    beta = model.dt_s * model.norms["output_scale"] / scale[0, :6]
    np.testing.assert_allclose(
        penalty[f : f + q, None] * beta[None, :] ** 2 / (3 * n),
        original,
        rtol=2e-13,
        atol=1e-14,
    )
    assert np.count_nonzero(penalty[:f]) == np.count_nonzero(penalty[f + q :]) == 0
    head = model.params["quadratic"]
    candidate_value = np.sum(penalty[f : f + q, None] * (head * beta) ** 2) / (6 * n)
    original_value = 0.5 * np.sum(np.asarray(original) * head**2)
    assert candidate_value == pytest.approx(original_value, rel=2e-13)
