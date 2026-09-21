"""Read-only matrix-free diagnostics for a frozen, conditioned online proposal.

Historical production is supplied by the caller. This module never initializes,
observes, applies a proposal, or changes a model/session. Reference iterations
solve only the same damped IRLS system; they are not a new online recipe.
"""

from functools import cache

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

CHECKPOINTS = (4, 8, 16, 32, 64, 128)
MAX_ITERATIONS = 128
RELATIVE_TOLERANCE = 1e-6
PROBE_LABELS = ("four_raw", "four_trusted", "reference_raw", "reference_trusted")
STATUSES = (
    "running",
    "converged",
    "capped",
    "stalled",
    "breakdown",
    "nonfinite",
    "metric_underflow",
)


def _system(
    residual, curvature, params, norms, data, scale, weights, damping, delay, dt_s
):
    theta, unpack = ravel_pytree(params)
    raw, push = jax.linearize(
        lambda flat: residual(unpack(flat), norms, data, scale, delay, dt_s), theta
    )
    pull = jax.linear_transpose(push, jnp.zeros_like(theta))
    prior = curvature(params, norms, data, scale, weights, delay=delay, dt_s=dt_s)
    weight = weights[:, None, None] / (raw.shape[1] * 3)
    group_norm = jnp.stack(
        [jnp.linalg.norm(raw[..., a:b], axis=-1) for a, b in ((0, 3), (3, 6), (6, 15))],
        axis=-1,
    )
    irls = jnp.repeat(
        1 / jnp.maximum(1.0, group_norm),
        np.array([3, 3, 9]),
        axis=-1,
        total_repeat_length=15,
    )
    gradient_data = pull(weight * irls * raw)[0]
    preconditioner = prior + damping
    base = dict(
        theta=theta,
        raw=raw,
        weight=weight,
        irls=irls,
        prior=prior,
        preconditioner=preconditioner,
        damping=jnp.asarray(damping),
        gradient_data=gradient_data,
        gradient_prior=prior * theta,
        gradient=gradient_data + prior * theta,
    )

    def action(direction):
        projected = push(direction)
        return projected, pull(weight * irls * projected)[
            0
        ] + preconditioner * direction

    return base, unpack, action


def _pcg_trace(
    action,
    gradient,
    preconditioner,
    raw_shape,
    *,
    max_iterations=MAX_ITERATIONS,
    checkpoints=CHECKPOINTS,
    tolerance=RELATIVE_TOLERANCE,
    initial_finite=True,
):
    """Instrument the production recurrence, without residual replacement.

    Private budget arguments permit small analytic fixtures; the public audit
    always uses the frozen constants above. Stored rows are padded internally.
    """
    size = gradient.size
    slots = len(checkpoints) + 2
    metric = jnp.sum(gradient * (gradient / preconditioner))
    exactly_zero = jnp.all(gradient == 0)
    initial_finite = (
        jnp.asarray(initial_finite)
        & jnp.all(jnp.isfinite(gradient))
        & jnp.all(jnp.isfinite(preconditioner))
        & jnp.all(preconditioner > 0)
    )
    metric_bad = (~jnp.isfinite(metric)) | (metric < 0)
    status = jnp.where(
        ~initial_finite | metric_bad,
        5,
        jnp.where(exactly_zero, 1, jnp.where(metric == 0, 6, 0)),
    )
    residual = -gradient
    direction = residual / preconditioner
    zero = jnp.zeros_like(gradient)
    trace = {
        name: jnp.zeros((max_iterations, size))
        for name in (
            "trace_direction",
            "trace_curvature",
        )
    }
    trace["trace_projected"] = jnp.zeros((max_iterations, *raw_shape))
    for name in ("alpha", "beta", "rho", "next_rho", "denominator"):
        trace[f"trace_{name}"] = jnp.zeros(max_iterations)
    for name in ("valid", "active", "finite"):
        trace[f"trace_{name}"] = jnp.zeros(max_iterations, dtype=bool)
    checkpoint = {
        f"checkpoint_{name}": jnp.zeros((slots, size))
        for name in (
            "delta",
            "recursive_residual",
            "true_residual",
            "curvature",
        )
    }
    checkpoint["checkpoint_projected"] = jnp.zeros((slots, *raw_shape))
    checkpoint["checkpoint_iterations"] = (
        jnp.full(slots, -1, dtype=jnp.int32).at[0].set(0)
    )
    checkpoint["checkpoint_recursive_residual"] = (
        checkpoint["checkpoint_recursive_residual"].at[0].set(residual)
    )
    checkpoint["checkpoint_true_residual"] = (
        checkpoint["checkpoint_true_residual"].at[0].set(residual)
    )
    for name in ("relative", "q", "gap_bound"):
        checkpoint[f"checkpoint_{name}"] = jnp.zeros(slots)
    checkpoint["checkpoint_relative"] = (
        checkpoint["checkpoint_relative"]
        .at[0]
        .set(jnp.where(exactly_zero, 0.0, jnp.where(metric > 0, 1.0, jnp.inf)))
    )
    checkpoint["checkpoint_gap_bound"] = (
        checkpoint["checkpoint_gap_bound"].at[0].set(0.5 * metric)
    )
    carry = dict(
        iteration=jnp.asarray(0, dtype=jnp.int32),
        status=status,
        delta=zero,
        residual=residual,
        direction=direction,
        rho=metric,
        finite=initial_finite,
        checkpoint_count=jnp.asarray(1, dtype=jnp.int32),
        four_delta=zero,
        four_finite=initial_finite,
        **trace,
        **checkpoint,
    )

    def step(state):
        i, p, rho = state["iteration"], state["direction"], state["rho"]
        projected, hp = action(p)
        denominator = jnp.vdot(p, hp)
        valid = jnp.all(jnp.isfinite(hp)) & jnp.all(jnp.isfinite(projected))
        valid &= jnp.isfinite(rho) & jnp.isfinite(denominator)
        valid &= (rho == 0) | ((rho > 0) & (denominator > 0))
        active = valid & (rho > 0)
        alpha = jnp.where(active, rho / jnp.where(active, denominator, 1.0), 0.0)
        delta = jnp.where(active, state["delta"] + alpha * p, state["delta"])
        following = jnp.where(active, state["residual"] - alpha * hp, state["residual"])
        z = following / preconditioner
        next_rho = jnp.vdot(following, z)
        valid &= jnp.all(jnp.isfinite(z)) & jnp.isfinite(next_rho) & (next_rho >= 0)
        beta = jnp.where(active, next_rho / jnp.where(active, rho, 1.0), 0.0)
        next_direction = jnp.where(active, z + beta * p, jnp.zeros_like(p))
        finite = state["finite"] & valid
        values = dict(
            direction=p,
            projected=projected,
            curvature=hp,
            alpha=alpha,
            beta=beta,
            rho=rho,
            next_rho=next_rho,
            denominator=denominator,
            valid=valid,
            active=active,
            finite=finite,
        )
        result = dict(state)
        for name, value in values.items():
            result[f"trace_{name}"] = state[f"trace_{name}"].at[i].set(value)
        result.update(
            iteration=i + 1,
            delta=delta,
            residual=following,
            direction=next_direction,
            rho=next_rho,
            finite=finite,
            four_delta=jnp.where(i == 3, delta, state["four_delta"]),
            four_finite=jnp.where(i == 3, finite, state["four_finite"]),
        )
        scheduled = jnp.any((i + 1) == jnp.asarray(checkpoints))
        check = scheduled | (~valid) | (next_rho == 0) | (i + 1 == max_iterations)

        def save_checkpoint(current):
            jp, hd = action(delta)
            true = -gradient - hd
            true_metric = jnp.sum(true * (true / preconditioner))
            relative = jnp.sqrt(true_metric / metric)
            action_finite = (
                jnp.all(jnp.isfinite(jp))
                & jnp.all(jnp.isfinite(hd))
                & jnp.isfinite(relative)
                & (true_metric >= 0)
            )
            converged = action_finite & (relative <= tolerance)
            numerical = (
                ~jnp.all(jnp.isfinite(hp))
                | ~jnp.all(jnp.isfinite(projected))
                | ~jnp.isfinite(next_rho)
                | ~jnp.all(jnp.isfinite(z))
            )
            next_status = jnp.where(
                ~action_finite | numerical,
                5,
                jnp.where(
                    converged,
                    1,
                    jnp.where(
                        ~valid,
                        4,
                        jnp.where(
                            next_rho == 0, 3, jnp.where(i + 1 == max_iterations, 2, 0)
                        ),
                    ),
                ),
            )
            slot = current["checkpoint_count"]
            out = dict(current)
            entries = dict(
                iterations=i + 1,
                delta=delta,
                recursive_residual=following,
                true_residual=true,
                curvature=hd,
                projected=jp,
                relative=relative,
                q=jnp.vdot(gradient, delta) + 0.5 * jnp.vdot(delta, hd),
                gap_bound=0.5 * true_metric,
            )
            for name, value in entries.items():
                out[f"checkpoint_{name}"] = (
                    current[f"checkpoint_{name}"].at[slot].set(value)
                )
            out.update(checkpoint_count=slot + 1, status=next_status)
            return out

        return jax.lax.cond(check, save_checkpoint, lambda x: x, result)

    carry = jax.lax.while_loop(
        lambda s: (s["iteration"] < max_iterations) & (s["status"] == 0), step, carry
    )
    carry["four_delta"] = jnp.where(
        carry["iteration"] < 4, carry["delta"], carry["four_delta"]
    )
    carry["four_finite"] = jnp.where(
        carry["iteration"] < 4, carry["finite"], carry["four_finite"]
    )
    return carry


@cache
def _kernels(residual, curvature, delay, dt_s):
    @jax.jit
    def diagnose(params, norms, data, scale, weights, damping):
        base, unpack, action = _system(
            residual,
            curvature,
            params,
            norms,
            data,
            scale,
            weights,
            damping,
            delay,
            dt_s,
        )
        trace = _pcg_trace(
            action,
            base["gradient"],
            base["preconditioner"],
            base["raw"].shape,
            initial_finite=jnp.all(jnp.isfinite(base["prior"]))
            & jnp.all(base["prior"] >= 0),
        )
        radius = jnp.maximum(
            1.0, 0.5 * jnp.linalg.norm(jnp.sqrt(base["weight"]) * base["raw"])
        )
        raw_directions = jnp.stack((trace["four_delta"], trace["delta"]))
        # Checkpoint projections also provide these actions, but two explicit
        # projections keep zero/early-breakdown handling uniform and are counted.
        # Only JVP results are used; dead-code elimination removes their VJPs.
        projections = jax.lax.map(lambda d: action(d)[0], raw_directions)
        sizes = jnp.linalg.norm(
            (jnp.sqrt(base["weight"])[None] * projections).reshape(2, -1), axis=1
        )
        shrink = jnp.minimum(1.0, radius / jnp.maximum(sizes, 1e-30))
        probe_delta = jnp.stack(
            (
                raw_directions[0],
                shrink[0] * raw_directions[0],
                raw_directions[1],
                shrink[1] * raw_directions[1],
            )
        )
        probe_projected = jnp.stack(
            (
                projections[0],
                shrink[0] * projections[0],
                projections[1],
                shrink[1] * projections[1],
            )
        )
        trial_raw = jax.lax.map(
            lambda d: residual(
                unpack(base["theta"] + d), norms, data, scale, delay, dt_s
            ),
            probe_delta,
        )
        return dict(
            **base,
            **trace,
            radius=radius,
            probe_delta=probe_delta,
            probe_projected=probe_projected,
            probe_trial_raw=trial_raw,
            probe_shrink=jnp.array([1.0, shrink[0], 1.0, shrink[1]]),
        )

    @jax.jit
    def recompute(
        params, norms, data, scale, weights, damping, directions, checkpoints, probes
    ):
        base, unpack, action = _system(
            residual,
            curvature,
            params,
            norms,
            data,
            scale,
            weights,
            damping,
            delay,
            dt_s,
        )
        projected, curvature_values = jax.lax.map(action, directions)
        checkpoint_projected, checkpoint_curvature = jax.lax.map(action, checkpoints)
        probe_projected = jax.lax.map(lambda d: action(d)[0], probes)
        trial_raw = jax.lax.map(
            lambda d: residual(
                unpack(base["theta"] + d), norms, data, scale, delay, dt_s
            ),
            probes,
        )
        return dict(
            **base,
            trace_projected=projected,
            trace_curvature=curvature_values,
            checkpoint_projected=checkpoint_projected,
            checkpoint_curvature=checkpoint_curvature,
            probe_projected=probe_projected,
            probe_trial_raw=trial_raw,
        )

    return diagnose, recompute


def _number(value):
    value = float(value)
    return value if np.isfinite(value) else None


def _numpy_data_loss(raw, weight):
    squared = np.stack(
        [np.sum(raw[..., a:b] ** 2, axis=-1) for a, b in ((0, 3), (3, 6), (6, 15))], -1
    )
    return float(
        np.sum(weight * np.where(squared <= 1, 0.5 * squared, np.sqrt(squared) - 0.5))
    )


def _gradient_metrics(data, prior, metric):
    a, b = np.sqrt(metric) * data, np.sqrt(metric) * prior
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return dict(
        data_norm=_number(na),
        prior_norm=_number(nb),
        total_norm=_number(np.linalg.norm(a + b)),
        cosine=_number(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else None,
        cancellation_ratio=_number(np.linalg.norm(a + b) / (na + nb))
        if na + nb > 0
        else None,
    )


def summary_from_arrays(arrays, *, parameter_schema):
    """JSON-safe reporting; the independent verifier implements its own algebra."""
    a = arrays
    current_data = _numpy_data_loss(a["raw"], a["weight"])
    current_prior = 0.5 * np.dot(a["theta"], a["prior"] * a["theta"])
    current = current_data + current_prior
    probes = []
    for i, name in enumerate(PROBE_LABELS):
        delta, jp, raw = (
            a["probe_delta"][i],
            a["probe_projected"][i],
            a["probe_trial_raw"][i],
        )
        theta = a["theta"] + delta
        data = _numpy_data_loss(raw, a["weight"])
        prior = 0.5 * np.dot(theta, a["prior"] * theta)
        predicted = -np.dot(a["gradient"], delta) - 0.5 * (
            np.sum(a["weight"] * a["irls"] * jp**2) + np.dot(delta, a["prior"] * delta)
        )
        total = data + prior
        finite = all(
            np.isfinite(v).all()
            for v in (
                delta,
                jp,
                raw,
                theta,
                a["gradient"],
                predicted,
                current,
                total,
            )
        )
        finite &= bool(a["four_finite"] if i < 2 else a["finite"])
        gain = (current - total) / predicted if finite and predicted > 0 else -np.inf
        damped_q = -predicted + 0.5 * float(a["damping"]) * np.dot(delta, delta)
        probes.append(
            dict(
                label=name,
                finite=bool(finite),
                shrink=_number(a["probe_shrink"][i]),
                projection_norm=_number(np.linalg.norm(np.sqrt(a["weight"]) * jp)),
                data_loss=_number(data),
                prior_loss=_number(prior),
                total_loss=_number(total),
                data_improvement=_number(current_data - data),
                prior_improvement=_number(current_prior - prior),
                improvement=_number(current - total),
                predicted_reduction=_number(predicted),
                damped_quadratic=_number(damped_q),
                damped_quadratic_decrease=_number(-damped_q),
                gain=_number(gain),
                would_accept=bool(finite and total < current and gain >= 0.1),
                gradient_data_dot=_number(np.dot(a["gradient_data"], delta)),
                gradient_prior_dot=_number(np.dot(a["gradient_prior"], delta)),
            )
        )
    count, checks = len(a["trace_alpha"]), len(a["checkpoint_iterations"])
    return dict(
        format="online-proposal-diagnostics-v1",
        parameter_schema=parameter_schema,
        reference=dict(
            status=STATUSES[int(a["status"])],
            iterations=count,
            relative_residual=_number(a["checkpoint_relative"][-1]),
        ),
        current=dict(
            data_loss=_number(current_data),
            prior_loss=_number(current_prior),
            total_loss=_number(current),
        ),
        radius=_number(a["radius"]),
        probes=probes,
        gradients=dict(
            coordinate=_gradient_metrics(
                a["gradient_data"], a["gradient_prior"], np.ones_like(a["theta"])
            ),
            preconditioned=_gradient_metrics(
                a["gradient_data"], a["gradient_prior"], 1 / a["preconditioner"]
            ),
        ),
        work=dict(
            residual_linearizations=1,
            gradient_vjp_calls=1,
            cg_iterations=count,
            cg_jvp_vjp_pairs=count,
            checkpoint_jvp_vjp_pairs=checks - 1,
            raw_probe_jvp_calls=2,
            trial_residual_calls=4,
            production_calls=0,
            applied_updates=0,
            observe_calls=0,
            initialization_calls=0,
        ),
    )


def _assert_close(actual, expected, name, *, production=False):
    atol, rtol = (1e-9, 1e-8) if production else (1e-10, 1e-8)
    try:
        np.testing.assert_allclose(
            actual, expected, atol=atol, rtol=rtol, equal_nan=True
        )
    except AssertionError as error:
        raise ValueError(f"{name} differs from authenticated equations") from error


def diagnose_proposal(
    module, params, norms, data, scale, weights, damping, *, delay, dt_s, production
):
    """Return JSON summary and numeric evidence on already-conditioned copies."""
    with jax.enable_x64(True):
        values = _kernels(module._residual, module._curvature_diagonal, delay, dt_s)[0](
            params, norms, data, scale, weights, damping
        )
        arrays = jax.tree.map(np.asarray, values)
        production_theta = np.asarray(ravel_pytree(production[0])[0])
    count, checks = int(arrays["iteration"]), int(arrays["checkpoint_count"])
    for key in arrays:
        if key.startswith("trace_"):
            arrays[key] = arrays[key][:count]
        elif key.startswith("checkpoint_") and key != "checkpoint_count":
            arrays[key] = arrays[key][:checks]
    arrays.update(production_theta=production_theta)
    for key, value in zip(("current", "trial", "predicted", "finite"), production[1:]):
        arrays[f"production_{key}"] = np.asarray(value)
    schema, start = [], 0
    leaves, _ = jax.tree_util.tree_flatten_with_path(params)
    for path, value in leaves:
        stop = start + np.size(value)
        schema.append(
            dict(
                path=jax.tree_util.keystr(path),
                shape=list(np.shape(value)),
                start=start,
                stop=stop,
            )
        )
        start = stop
    summary = summary_from_arrays(arrays, parameter_schema=schema)
    four = summary["probes"][1]
    _assert_close(
        arrays["theta"] + arrays["probe_delta"][1],
        production_theta,
        "four-step parameters",
        production=True,
    )
    for name, computed, original in (
        ("current", summary["current"]["total_loss"], production[1]),
        ("trial", four["total_loss"], production[2]),
        ("predicted", four["predicted_reduction"], production[3]),
    ):
        _assert_close(
            np.nan if computed is None else computed,
            original,
            f"four-step {name}",
            production=True,
        )
    if four["finite"] != bool(production[4]):
        raise ValueError("four-step finite flag differs from production")
    summary["production_comparison_passed"] = True
    return summary, arrays


def recompute_actions(
    module,
    params,
    norms,
    data,
    scale,
    weights,
    damping,
    *,
    delay,
    dt_s,
    summary,
    arrays,
):
    """Verify all saved derivative actions, without running a PCG recurrence."""
    with jax.enable_x64(True):
        values = _kernels(module._residual, module._curvature_diagonal, delay, dt_s)[1](
            params,
            norms,
            data,
            scale,
            weights,
            damping,
            arrays["trace_direction"],
            arrays["checkpoint_delta"],
            arrays["probe_delta"],
        )
        values = jax.tree.map(np.asarray, values)
    for name, value in values.items():
        _assert_close(value, arrays[name], f"recomputed {name}")
    if summary["format"] != "online-proposal-diagnostics-v1":
        raise ValueError("unsupported proposal diagnostic summary")
    return dict(
        residual_linearizations=1,
        gradient_vjp_calls=1,
        trace_jvp_vjp_pairs=len(arrays["trace_direction"]),
        checkpoint_jvp_vjp_pairs=len(arrays["checkpoint_delta"]),
        probe_jvp_calls=4,
        trial_residual_calls=4,
        cg_iterations=0,
        production_calls=0,
        applied_updates=0,
        observe_calls=0,
        initialization_calls=0,
    )
