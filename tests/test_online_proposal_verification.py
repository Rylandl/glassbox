"""Dense, independently assembled evidence for the saved-array arithmetic audit."""

from copy import deepcopy

import numpy as np
import pytest
from _online_proposal_diagnostics import summary_from_arrays
from _online_proposal_verification import verify_arrays


def dense_evidence(*, zero=False, breakdown=False):
    """Use an explicit dense Hessian, with no AD or Glassbox model calls."""
    rng = np.random.default_rng(981)
    size, shape = 9, (2, 2, 15)
    design = rng.normal(size=(np.prod(shape), size)) * np.geomspace(0.1, 7, size)
    theta = np.zeros(size) if zero else rng.normal(size=size)
    raw = (
        np.zeros(shape)
        if zero
        else (design @ theta + rng.normal(size=np.prod(shape))).reshape(shape)
    )
    weight = np.asarray([0.4, 0.6])[:, None, None] / 6
    norms = np.stack(
        [
            np.linalg.norm(raw[..., s], axis=-1)
            for s in (slice(0, 3), slice(3, 6), slice(6, 15))
        ],
        -1,
    )
    irls = np.repeat(1 / np.maximum(norms, 1), [3, 3, 9], axis=-1)
    prior = np.geomspace(0.01, 0.2, size)
    damping = 0.04
    metric = prior + damping
    w = (weight * irls).ravel()
    data_gradient = design.T @ (w * raw.ravel())
    gradient = data_gradient + prior * theta
    hessian = design.T @ (w[:, None] * design) + np.diag(metric)
    a = dict(
        theta=theta,
        raw=raw,
        irls=irls,
        weight=weight,
        prior=prior,
        damping=np.asarray(damping),
        preconditioner=metric,
        gradient_data=data_gradient,
        gradient_prior=prior * theta,
        gradient=gradient,
    )
    trace = {
        key: []
        for key in (
            "direction",
            "projected",
            "curvature",
            "alpha",
            "beta",
            "rho",
            "next_rho",
            "denominator",
            "active",
            "valid",
            "finite",
        )
    }
    checkpoints = {
        key: []
        for key in (
            "iterations",
            "delta",
            "recursive_residual",
            "true_residual",
            "curvature",
            "projected",
            "relative",
            "q",
            "gap_bound",
        )
    }
    delta = np.zeros(size)
    residual = -gradient.copy()
    direction = residual / metric
    rho = float(residual @ direction)
    initial_metric = rho
    four = None
    finite = True

    def checkpoint(iteration):
        hd, projected = hessian @ delta, (design @ delta).reshape(shape)
        true = -gradient - hd
        error = true @ (true / metric)
        relative = np.sqrt(error / initial_metric) if initial_metric else 0.0
        values = dict(
            iterations=iteration,
            delta=delta.copy(),
            recursive_residual=residual.copy(),
            true_residual=true,
            curvature=hd,
            projected=projected,
            relative=relative,
            q=gradient @ delta + 0.5 * delta @ hd,
            gap_bound=0.5 * error,
        )
        for key, value in values.items():
            checkpoints[key].append(value)
        return relative

    checkpoint(0)
    count, status = 0, 1 if zero else 0
    while status == 0:
        projected = (design @ direction).reshape(shape)
        hp = -direction if breakdown else hessian @ direction
        denominator = direction @ hp
        active = denominator > 0
        alpha = rho / denominator if active else 0.0
        p = direction.copy()
        delta = delta + alpha * p
        residual = residual - alpha * hp
        z = residual / metric
        next_rho = residual @ z
        beta = next_rho / rho if active else 0.0
        valid = active
        finite &= valid
        direction = z + beta * p if active else np.zeros_like(p)
        values = dict(
            direction=p,
            projected=projected,
            curvature=hp,
            alpha=alpha,
            beta=beta,
            rho=rho,
            next_rho=next_rho,
            denominator=denominator,
            active=active,
            valid=valid,
            finite=finite,
        )
        for key, value in values.items():
            trace[key].append(value)
        count += 1
        rho = next_rho
        if count == 4:
            four = delta.copy()
        if count in (4, 8, 16, 32, 64, 128) or not valid:
            relative = checkpoint(count)
            status = (
                1 if relative <= 1e-6 else 4 if not valid else 2 if count == 128 else 0
            )
    if four is None:
        four = delta.copy()
    a.update(
        iteration=np.asarray(count),
        status=np.asarray(status),
        delta=delta,
        residual=residual,
        direction=direction,
        rho=np.asarray(rho),
        finite=np.asarray(finite),
        four_delta=four,
        four_finite=np.asarray(not breakdown),
        checkpoint_count=np.asarray(len(checkpoints["iterations"])),
    )
    for key, value in trace.items():
        suffix = (
            shape
            if key == "projected"
            else (size,)
            if key in ("direction", "curvature")
            else ()
        )
        dtype = bool if key in ("active", "valid", "finite") else float
        a["trace_" + key] = np.asarray(value, dtype=dtype).reshape(count, *suffix)
    for key, value in checkpoints.items():
        a["checkpoint_" + key] = np.asarray(value)
    radius = max(1.0, 0.5 * np.linalg.norm(np.sqrt(weight) * raw))
    a["radius"] = np.asarray(radius)
    deltas, projections, shrinks = [], [], []
    for proposal in (four, delta):
        projected = (design @ proposal).reshape(shape)
        shrink = min(
            1.0, radius / max(np.linalg.norm(np.sqrt(weight) * projected), 1e-30)
        )
        deltas.extend((proposal, shrink * proposal))
        projections.extend((projected, shrink * projected))
        shrinks.extend((1.0, shrink))
    a.update(
        probe_delta=np.stack(deltas),
        probe_projected=np.stack(projections),
        probe_shrink=np.asarray(shrinks),
    )
    a["probe_trial_raw"] = raw[None] + a["probe_projected"]
    schema = [dict(path="['theta']", shape=[size], start=0, stop=size)]
    summary = summary_from_arrays(a, parameter_schema=schema)
    summary["production_comparison_passed"] = True
    a["production_theta"] = theta + a["probe_delta"][1]
    for key, value in (
        ("current", summary["current"]["total_loss"]),
        ("trial", summary["probes"][1]["total_loss"]),
        ("predicted", summary["probes"][1]["predicted_reduction"]),
        ("finite", summary["probes"][1]["finite"]),
    ):
        a["production_" + key] = np.asarray(value)
    return summary, a, hessian


def test_dense_system_checks_reference_solution_and_trust():
    summary, arrays, hessian = dense_evidence()
    result = verify_arrays(summary, arrays)
    np.testing.assert_allclose(
        arrays["delta"],
        np.linalg.solve(hessian, -arrays["gradient"]),
        rtol=2e-6,
        atol=1e-8,
    )
    assert arrays["checkpoint_relative"][1] > 1e-6
    assert arrays["probe_shrink"][1] < 1
    assert result["verified"] and result["krylov_directions"] > 4
    assert result["model_calls"] == result["optimizer_calls"] == 0


@pytest.mark.parametrize(
    "field,index",
    [
        ("gradient_prior", (0,)),
        ("irls", (0, 0, 0)),
        ("trace_alpha", (0,)),
        ("trace_direction", (0, 0)),
        ("trace_curvature", (0, 0)),
        ("trace_projected", (0, 0, 0, 0)),
        ("checkpoint_true_residual", (1, 0)),
        ("checkpoint_curvature", (1, 0)),
        ("probe_delta", (1, 0)),
        ("probe_trial_raw", (1, 0, 0, 0)),
        ("production_theta", (0,)),
    ],
)
def test_tampered_array_fails(field, index):
    summary, arrays, _ = dense_evidence()
    arrays[field][index] += 0.01
    with pytest.raises(ValueError):
        verify_arrays(summary, arrays)


def test_tampered_summary_or_status_fails():
    summary, arrays, _ = dense_evidence()
    damaged = deepcopy(summary)
    damaged["probes"][1]["gain"] += 0.1
    with pytest.raises(ValueError, match="gain"):
        verify_arrays(damaged, arrays)
    arrays["status"] = np.asarray(2)
    with pytest.raises(ValueError, match="status"):
        verify_arrays(summary, arrays)


def test_exact_zero_gradient_and_retained_breakdown_are_distinct():
    summary, arrays, _ = dense_evidence(zero=True)
    result = verify_arrays(summary, arrays)
    assert result["iterations"] == 0 and result["reference_status"] == "converged"
    assert not any(probe["would_accept"] for probe in summary["probes"])
    summary, arrays, _ = dense_evidence(breakdown=True)
    result = verify_arrays(summary, arrays)
    assert result["iterations"] == 1 and result["reference_status"] == "breakdown"
    assert not any(probe["finite"] for probe in summary["probes"])


def test_krylov_gradient_identity_catches_self_consistent_forged_gradient():
    summary, arrays, _ = dense_evidence()
    # Keep the total gradient and every recurrence scalar unchanged. Only the
    # data/prior split is forged; both simple sum and A*theta checks still pass.
    forged = 0.01 * np.ones_like(arrays["theta"])
    arrays["gradient_data"] += forged
    arrays["gradient_prior"] -= forged
    arrays["theta"] = arrays["gradient_prior"] / arrays["prior"]
    with pytest.raises(ValueError):
        verify_arrays(summary, arrays)


def test_nonzero_gradient_metric_underflow_is_not_convergence():
    _, arrays, _ = dense_evidence(zero=True)
    arrays["theta"][:] = 1e-200
    arrays["prior"][:] = 1.0
    arrays["damping"] = np.asarray(1e200)
    arrays["preconditioner"][:] = 1e200
    arrays["gradient_prior"][:] = 1e-200
    arrays["gradient"][:] = 1e-200
    arrays["residual"][:] = -1e-200
    arrays["direction"][:] = 0.0
    arrays["checkpoint_recursive_residual"][0] = arrays["residual"]
    arrays["checkpoint_true_residual"][0] = arrays["residual"]
    arrays["checkpoint_relative"][0] = np.inf
    arrays["status"] = np.asarray(6)
    arrays["production_theta"] = arrays["theta"].copy()
    summary = summary_from_arrays(
        arrays, parameter_schema=[dict(path="['theta']", shape=[9], start=0, stop=9)]
    )
    summary["production_comparison_passed"] = True
    result = verify_arrays(summary, arrays)
    assert result["reference_status"] == "metric_underflow"
    assert summary["reference"]["relative_residual"] is None
