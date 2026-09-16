"""Numerical and data-contract checks for the experimental transition learner."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from glassbox.experimental.transition_gp import (
    GaussianTransition,
    TransitionSamples,
    _kernel,
    fit_transition_gp,
)


def repeated_observation_model(kernel):
    # Three independent noisy observations of a unit-variance function at zero.
    # This has a scalar closed-form posterior, independent of GP implementation.
    noise = 0.2
    covariance = np.ones((3, 3)) + (noise**2 + 1e-6) * np.eye(3)
    return GaussianTransition(
        kernel=kernel,
        dt_s=0.1,
        state_size=1,
        command_size=1,
        context_size=0,
        feature_mean=jnp.zeros(2),
        feature_scale=jnp.ones(2),
        target_mean=jnp.zeros(1),
        target_scale=jnp.ones(1),
        features=jnp.zeros((3, 2)),
        theta=jnp.log(
            jnp.array([1.0, 1.0] + ([0.4] if kernel == "rq" else []) + [1.0, noise])
        ),
        chol=jnp.asarray(np.linalg.cholesky(covariance)),
        alpha=jnp.zeros((3, 1)),
        fit_report={},
    )


@pytest.mark.parametrize("kernel", ["rbf", "matern52", "rq"])
def test_conditioning_matches_repeated_measurement_closed_form(kernel):
    model = repeated_observation_model(kernel)
    result = model.predict(jnp.array([0.0]), jnp.array([0.0]))
    expected = (0.2**2 + 1e-6) / (3 + 0.2**2 + 1e-6)
    np.testing.assert_allclose(result.function_variance, expected, atol=3e-7)
    np.testing.assert_allclose(
        result.observation_variance - result.function_variance, 0.2**2, atol=1e-7
    )
    # RQ correlations decay algebraically, more slowly than the other kernels.
    far = model.predict(jnp.array([1e6]), jnp.array([1e6]))
    np.testing.assert_allclose(far.function_variance, 1.0, atol=1e-6)
    local = model.local_response(jnp.array([0.0]), jnp.array([0.0]))
    assert all(np.all(np.isfinite(value)) for value in local)
    support = model.support(np.array([0.0]), np.array([0.0]))
    assert support.neighbor_count == 3
    # Function values at the origin do not constrain its gradient for either
    # stationary kernel. The Matern-5/2 prior slope variance is 5/3.
    slope_variance = 5.0 / 3.0 if kernel == "matern52" else 1.0
    np.testing.assert_allclose(
        local.jacobian_covariance[0], slope_variance * np.eye(2), atol=2e-5
    )


@pytest.fixture(scope="module", params=["rbf", "rq"])
def learned_model(request):
    grid = np.linspace(-1, 1, 6)
    x, u = np.meshgrid(grid, grid)
    x, u = x.ravel()[:, None], u.ravel()[:, None]
    targets = 0.6 * x + 0.3 * u + 0.15 * x * u
    return fit_transition_gp(
        TransitionSamples(x, u, targets, 0.1),
        kernel=request.param,
        steps=80,
        restarts=1,
    )


def test_learns_complete_transition_and_jit_rollout(learned_model):
    model = learned_model
    x, u = jnp.array([[0.17], [-0.43]]), jnp.array([[-0.23], [0.32]])
    expected = 0.6 * x + 0.3 * u + 0.15 * x * u
    prediction = jax.jit(model.predict)(x, u)
    np.testing.assert_allclose(prediction.mean, expected, atol=0.015)
    commands = jnp.array([[0.1], [-0.2], [0.3]])
    rollout = jax.jit(model.mean_rollout)(jnp.array([0.2]), commands)
    reference = [0.2]
    for command in np.asarray(commands).ravel():
        state = reference[-1]
        reference.append(0.6 * state + 0.3 * command + 0.15 * state * command)
    np.testing.assert_allclose(rollout[:, 0], reference, atol=0.02)


def test_local_slope_and_curvature_against_known_response(learned_model):
    x, u = 0.13, -0.2
    result = learned_model.local_response(jnp.array([x]), jnp.array([u]))
    np.testing.assert_allclose(
        result.jacobian, [[0.6 + 0.15 * u, 0.3 + 0.15 * x]], atol=0.025
    )
    np.testing.assert_allclose(result.hessian[0], [[0.0, 0.15], [0.15, 0.0]], atol=0.05)
    assert np.linalg.eigvalsh(result.jacobian_covariance).min() >= -1e-7


def test_round_trip_preserves_predictions_and_evidence(learned_model, tmp_path):
    path = tmp_path / "model.npz"
    learned_model.save(path)
    loaded = GaussianTransition.load(path)
    assert loaded.fit_report == learned_model.fit_report
    for left, right in zip(
        learned_model.predict(jnp.array([0.1]), jnp.array([0.2])),
        loaded.predict(jnp.array([0.1]), jnp.array([0.2])),
    ):
        np.testing.assert_array_equal(left, right)


def test_rational_quadratic_limit_and_long_range_correlation():
    points = jnp.array([[0.0, 0.0], [0.1, -0.2], [1.0, 0.4]])
    rbf = _kernel(points, points, jnp.log(jnp.array([0.8, 1.2, 1.0, 0.1])), "rbf")
    rq = _kernel(points, points, jnp.log(jnp.array([0.8, 1.2, 1e6, 1.0, 0.1])), "rq")
    np.testing.assert_allclose(rq, rbf, atol=1e-6)
    broad = _kernel(points, points, jnp.log(jnp.array([0.8, 1.2, 0.3, 1.0, 0.1])), "rq")
    assert float(broad[0, -1]) > float(rbf[0, -1])
    assert np.linalg.eigvalsh(np.asarray(broad)).min() > 0


def test_support_identifies_thin_direction_despite_nearby_samples():
    coordinate = np.linspace(-1, 1, 41)[:, None]
    model = fit_transition_gp(
        TransitionSamples(coordinate, coordinate, coordinate / 2, 0.1),
        steps=3,
        restarts=1,
    )
    report = model.support(np.array([[0.04]]), np.array([[-0.04]]), neighbors=12)
    assert report.nearest_distance[0] < 0.11
    assert report.eigenvalues[0, 0] < 1e-10
    assert report.eigenvalues[0, 1] > 0.01
    weak_direction = report.basis[0, :, 0]
    assert abs(weak_direction @ np.array([1.0, -1.0]) / np.sqrt(2)) > 0.999
    assert abs(report.query_offset_in_basis[0, 0]) > 0.08


def test_arbitrary_channel_counts_context_and_input_copy():
    rng = np.random.default_rng(19)
    x, u, context = rng.normal(size=(3, 24, 2))
    y = x * 0.4 + u * 0.2 + context * 0.3
    samples = TransitionSamples(x, u, y, 0.2, context)
    x[0, 0] = 1000
    assert samples.states[0, 0] != 1000
    model = fit_transition_gp(samples, steps=10, restarts=1)
    result = model.predict(samples.states[:2], u[:2], context=context[:2])
    assert result.mean.shape == (2, 2)
    with pytest.raises(ValueError, match="requires context"):
        model.predict(samples.states[0], u[0])
    with pytest.raises(ValueError, match="command shape"):
        model.predict(samples.states[0], u[:2], context=context[0])


@pytest.mark.parametrize("bad_dt", [0.0, -1.0, np.nan])
def test_rejects_invalid_timing(bad_dt):
    with pytest.raises(ValueError, match="dt_s"):
        TransitionSamples(np.zeros((3, 1)), np.zeros((3, 1)), np.zeros((3, 1)), bad_dt)


def test_rejects_nonfinite_or_misaligned_samples():
    with pytest.raises(ValueError, match="finite"):
        TransitionSamples(
            np.full((3, 1), np.nan), np.zeros((3, 1)), np.zeros((3, 1)), 0.1
        )
    with pytest.raises(ValueError, match="row counts"):
        TransitionSamples(np.zeros((3, 1)), np.zeros((4, 1)), np.zeros((3, 1)), 0.1)
    with pytest.raises(ValueError, match="same shape"):
        TransitionSamples(np.zeros((3, 1)), np.zeros((3, 1)), np.zeros((3, 2)), 0.1)
