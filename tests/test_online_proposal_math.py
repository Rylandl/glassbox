"""Independent explicit-J and dense-matrix oracles for proposal diagnostics."""

import sys
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import _online_proposal_diagnostics as diagnostic
import _online_proposal_verification as verification


def _module(*, nonlinear=False):
    def residual(params, norms, data, scale, delay, dt_s):
        del data, scale, delay, dt_s
        x = params["x"]
        value = norms["matrix"] @ x
        if nonlinear:
            value += jnp.sin(norms["wave"] @ x)
        return (value - norms["target"]).reshape(-1, 1, 15)

    def curvature(params, norms, data, scale, weights, *, delay, dt_s):
        del params, data, scale, weights, delay, dt_s
        return norms["prior"]

    return SimpleNamespace(_residual=residual, _curvature_diagonal=curvature)


def _raw_jacobian(theta, norms, nonlinear):
    value = norms["matrix"] @ theta
    jacobian = norms["matrix"].copy()
    if nonlinear:
        value += np.sin(norms["wave"] @ theta)
        jacobian += np.cos(norms["wave"] @ theta)[:, None] * norms["wave"]
    return (value - norms["target"]).reshape(-1, 1, 15), jacobian


def _data_loss(raw, weight):
    total = 0.0
    for row in range(len(raw)):
        for a, b in ((0, 3), (3, 6), (6, 15)):
            norm = np.linalg.norm(raw[row, 0, a:b])
            total += weight[row] / 3 * (0.5 * norm**2 if norm <= 1 else norm - 0.5)
    return total


def _dense_problem(theta, norms, weights, damping, nonlinear):
    raw, jac = _raw_jacobian(theta, norms, nonlinear)
    diagonal = np.empty_like(raw)
    for row in range(len(raw)):
        for a, b in ((0, 3), (3, 6), (6, 15)):
            diagonal[row, 0, a:b] = weights[row] / (
                3 * max(1.0, np.linalg.norm(raw[row, 0, a:b]))
            )
    prior = norms["prior"]
    gradient = jac.T @ (diagonal * raw).ravel() + prior * theta
    hessian = jac.T @ (diagonal.ravel()[:, None] * jac) + np.diag(prior + damping)
    return raw, jac, diagonal, gradient, hessian


def _production(theta, norms, weights, damping, nonlinear=False):
    raw, jac, diagonal, gradient, hessian = _dense_problem(
        theta, norms, weights, damping, nonlinear
    )
    prior = norms["prior"]
    delta, residual = np.zeros_like(theta), -gradient
    direction = residual / (prior + damping)
    rho = residual @ direction
    for _ in range(4):
        hp = hessian @ direction
        if rho == 0:
            continue
        alpha = rho / (direction @ hp)
        delta += alpha * direction
        residual -= alpha * hp
        z = residual / (prior + damping)
        new_rho = residual @ z
        direction = z + (new_rho / rho) * direction
        rho = new_rho
    projected = (jac @ delta).reshape(raw.shape)
    weight = weights[:, None, None] / 3
    radius = max(1.0, 0.5 * np.linalg.norm(np.sqrt(weight) * raw))
    delta *= min(1.0, radius / max(np.linalg.norm(np.sqrt(weight) * projected), 1e-30))
    projected = (jac @ delta).reshape(raw.shape)
    candidate = theta + delta
    trial_raw, _ = _raw_jacobian(candidate, norms, nonlinear)
    current = _data_loss(raw, weights) + 0.5 * np.dot(theta, prior * theta)
    trial = _data_loss(trial_raw, weights) + 0.5 * np.dot(candidate, prior * candidate)
    predicted = -gradient @ delta - 0.5 * (
        np.sum(diagonal * projected**2) + np.dot(delta, prior * delta)
    )
    return {"x": candidate}, current, trial, predicted, True


def _fixture(*, nonlinear=False, strong_prior=False):
    rng = np.random.default_rng(817)
    left, _ = np.linalg.qr(rng.normal(size=(30, 8)))
    right, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    matrix = left @ np.diag(np.geomspace(0.4, 12, 8)) @ right.T
    theta = np.linspace(-0.4, 0.6, 8)
    wave = rng.normal(size=(30, 8)) * 0.4
    value = matrix @ theta + (np.sin(wave @ theta) if nonlinear else 0)
    norms = dict(
        matrix=matrix,
        wave=wave,
        target=value - np.linspace(-0.3, 0.2, 30),
        prior=np.linspace(0.1, 0.6, 8) * (400 if strong_prior else 1),
    )
    return _module(nonlinear=nonlinear), {"x": theta}, norms, np.array([0.4, 0.6]), 0.2


def _run(module, params, norms, weights, damping, *, nonlinear=False):
    production = _production(params["x"], norms, weights, damping, nonlinear)
    args = (module, params, norms, (), np.ones((1, 15)), weights, damping)
    summary, arrays = diagnostic.diagnose_proposal(
        *args, delay=0, dt_s=0.05, production=production
    )
    return summary, arrays, args


def test_reference_matches_dense_nonzero_prior_and_four_is_incomplete():
    module, params, norms, weights, damping = _fixture()
    summary, arrays, _ = _run(module, params, norms, weights, damping)
    assert verification.verify_arrays(summary, arrays)["verified"]
    _, _, _, gradient, hessian = _dense_problem(
        params["x"], norms, weights, damping, False
    )
    exact = np.linalg.solve(hessian, -gradient)
    four, reference = arrays["probe_delta"][[0, 2]]
    assert np.linalg.norm(four - exact) > 0.01
    np.testing.assert_allclose(reference, exact, atol=2e-8, rtol=2e-6)
    assert summary["reference"]["status"] == "converged"
    assert summary["reference"]["iterations"] > 4
    assert summary["reference"]["relative_residual"] <= 1e-6
    for delta, gap, q in zip(
        arrays["checkpoint_delta"],
        arrays["checkpoint_gap_bound"],
        arrays["checkpoint_q"],
    ):
        exact_gap = 0.5 * (delta - exact) @ hessian @ (delta - exact)
        assert exact_gap <= gap + 1e-10
        np.testing.assert_allclose(
            q, gradient @ delta + 0.5 * delta @ hessian @ delta, atol=1e-10
        )


def test_nonlinear_symbolic_actions_trust_and_readonly_recomputation(monkeypatch):
    module, params, norms, weights, damping = _fixture(
        nonlinear=True, strong_prior=True
    )
    before = {key: value.copy() for key, value in norms.items()}
    summary, arrays, args = _run(
        module, params, norms, weights, damping, nonlinear=True
    )
    assert verification.verify_arrays(summary, arrays)["verified"]
    raw, jac, diagonal, gradient, hessian = _dense_problem(
        params["x"], norms, weights, damping, True
    )
    np.testing.assert_allclose(arrays["raw"], raw, atol=1e-12)
    np.testing.assert_allclose(
        arrays["gradient_data"], jac.T @ (diagonal * raw).ravel(), atol=1e-12
    )
    np.testing.assert_allclose(
        arrays["trace_projected"].reshape(len(arrays["trace_alpha"]), -1),
        arrays["trace_direction"] @ jac.T,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        arrays["trace_curvature"], arrays["trace_direction"] @ hessian.T, atol=1e-9
    )
    assert arrays["probe_shrink"][1] < 1
    for i, probe in enumerate(summary["probes"]):
        delta = arrays["probe_delta"][i]
        undamped = hessian - damping * np.eye(len(delta))
        np.testing.assert_allclose(
            probe["predicted_reduction"],
            -gradient @ delta - 0.5 * delta @ undamped @ delta,
            atol=1e-10,
        )
    # Directional gradient products covary under a diagonal parameter chart.
    change = np.geomspace(0.2, 4, len(params["x"]))
    direction = arrays["probe_delta"][0]
    changed_chart = dict(
        norms,
        matrix=norms["matrix"] * change,
        wave=norms["wave"] * change,
        prior=norms["prior"] * change**2,
    )
    with jax.enable_x64(True):
        transformed, _, transformed_action = diagnostic._system(
            module._residual,
            module._curvature_diagonal,
            {"x": params["x"] / change},
            changed_chart,
            (),
            np.ones((1, 15)),
            weights,
            damping,
            0,
            0.05,
        )
        projected = np.asarray(transformed_action(direction / change)[0])
    np.testing.assert_allclose(projected.ravel(), jac @ direction, atol=1e-11)
    for key in ("gradient_data", "gradient_prior"):
        np.testing.assert_allclose(transformed[key], arrays[key] * change, atol=1e-11)
        np.testing.assert_allclose(
            np.asarray(transformed[key]) @ (direction / change),
            arrays[key] @ direction,
            atol=1e-11,
        )
    epsilon = 1e-6
    plus, _ = _raw_jacobian(params["x"] + epsilon * direction, norms, True)
    minus, _ = _raw_jacobian(params["x"] - epsilon * direction, norms, True)
    np.testing.assert_allclose(
        ((plus - minus) / (2 * epsilon)).ravel(), jac @ direction, atol=1e-8
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("recomputation invoked PCG")

    monkeypatch.setattr(diagnostic, "_pcg_trace", forbidden)
    counts = diagnostic.recompute_actions(
        *args, delay=0, dt_s=0.05, summary=summary, arrays=arrays
    )
    assert (
        counts["cg_iterations"]
        == counts["observe_calls"]
        == counts["applied_updates"]
        == 0
    )
    changed = dict(arrays, trace_curvature=arrays["trace_curvature"].copy())
    changed["trace_curvature"][0, 0] += 0.1
    with pytest.raises(ValueError, match="recomputed trace_curvature"):
        diagnostic.recompute_actions(
            *args, delay=0, dt_s=0.05, summary=summary, arrays=changed
        )
    for key in before:
        np.testing.assert_array_equal(norms[key], before[key])


def _trace(gradient, matrix, preconditioner, *, iterations=128):
    with jax.enable_x64(True):
        gradient, matrix, preconditioner = map(
            jnp.asarray, (gradient, matrix, preconditioner)
        )

        def action(value):
            return value.reshape(1, 1, -1), matrix @ value

        result = jax.jit(
            lambda: diagnostic._pcg_trace(
                action,
                gradient,
                preconditioner,
                (1, 1, gradient.size),
                max_iterations=iterations,
                checkpoints=(4,) if iterations == 4 else diagnostic.CHECKPOINTS,
            )
        )()
        return jax.tree.map(np.asarray, result)


def test_zero_capped_breakdown_nonfinite_and_underflow_are_distinct():
    zero = _trace(np.zeros(6), np.eye(6), np.ones(6))
    assert int(zero["iteration"]) == 0
    assert diagnostic.STATUSES[int(zero["status"])] == "converged"
    assert zero["checkpoint_relative"][0] == 0
    capped = _trace(np.ones(6), np.diag([1, 2, 4, 8, 16, 32]), np.ones(6), iterations=4)
    assert int(capped["iteration"]) == 4
    assert diagnostic.STATUSES[int(capped["status"])] == "capped"
    broken = _trace(np.ones(6), -np.eye(6), np.ones(6))
    assert int(broken["iteration"]) == 1
    assert diagnostic.STATUSES[int(broken["status"])] == "breakdown"
    assert not broken["trace_active"][0]
    np.testing.assert_array_equal(broken["delta"], 0)
    nonfinite = _trace(np.ones(6), np.eye(6) * np.nan, np.ones(6))
    assert diagnostic.STATUSES[int(nonfinite["status"])] == "nonfinite"
    underflow = _trace(np.full(6, 1e-200), np.eye(6), np.ones(6))
    assert diagnostic.STATUSES[int(underflow["status"])] == "metric_underflow"
    assert np.isinf(underflow["checkpoint_relative"][0])


def test_zero_gradient_public_diagnostic_and_action_recompute():
    module, params, norms, weights, damping = _fixture()
    params["x"] = np.zeros(8)
    norms["target"] = np.zeros(30)
    summary, arrays, args = _run(module, params, norms, weights, damping)
    assert summary["reference"] == dict(
        status="converged", iterations=0, relative_residual=0.0
    )
    assert arrays["trace_direction"].shape == (0, 8)
    assert all(not p["would_accept"] for p in summary["probes"])
    diagnostic.recompute_actions(
        *args, delay=0, dt_s=0.05, summary=summary, arrays=arrays
    )


def test_near_cancelling_projection_scaling_uses_frozen_action_tolerance():
    # A tiny projected response can be the difference of much larger terms.
    # Recomputed J(s*delta) must remain consistent with saved s*(J*delta).
    module = _module()
    matrix = np.zeros((15, 2))
    matrix[:, 0] = np.linspace(1e4, 1e6, 15)
    matrix[:, 1] = matrix[:, 0]
    norms = dict(matrix=matrix, target=np.zeros(15), prior=np.ones(2))
    with jax.enable_x64(True):
        _, _, action = diagnostic._system(
            module._residual,
            module._curvature_diagonal,
            {"x": jnp.zeros(2)},
            norms,
            (),
            np.ones((1, 15)),
            jnp.ones(1),
            jnp.asarray(0.2),
            0,
            0.05,
        )
        direction = jnp.array([1.0, -1.0 + 1e-10])
        shrink = jnp.asarray(0.123456789)
        saved = np.asarray(shrink * action(direction)[0])
        recomputed = np.asarray(action(shrink * direction)[0])
    assert np.linalg.norm(saved) < 1e-4
    diagnostic._assert_close(recomputed, saved, "near-cancelling scaled action")
