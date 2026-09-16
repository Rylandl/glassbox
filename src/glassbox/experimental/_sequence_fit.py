"""Bounded full-batch fitting for the fixed generic sequence recipe.

Optimization termination and development selection are separate evidence. The
returned checkpoint need not be the optimizer's last or stationary point.
"""

from __future__ import annotations

import copy

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from scipy.optimize import fmin_l_bfgs_b

from .sequence_model import SequenceModel, _rollout

_BOUNDED_FIT = {
    "optimizer": "bounded-full-batch-lbfgs-v1",
    "max_function_evaluations": 160,
    "max_iterations": 160,
    "memory": 10,
    "max_line_search_steps": 20,
    "gradient_tolerance": 1e-7,
    "relative_loss_factr": 10000.0,
    "checkpoint_every_evaluations": 16,
    "max_development_passes": 12,
}


class _BudgetExhausted(Exception):
    pass


class _InvalidAcceptedPoint(Exception):
    pass


def _run_lbfgs(initial, value_and_gradient, development_scores, settings):
    """Internal solver seam, independently testable without sequence machinery."""
    evaluations, cache_hits, nonfinite_trials = 0, 0, 0
    cache = None
    trace, vectors = [], []
    development_passes, invalid_development = 0, 0
    last_checked = None
    accepted = None
    next_checkpoint = settings["checkpoint_every_evaluations"]

    def objective(vector):
        nonlocal evaluations, cache_hits, nonfinite_trials, cache
        vector = np.asarray(vector, dtype=np.float64)
        if cache is not None and np.array_equal(vector, cache["vector"]):
            cache_hits += 1
            return cache["value"], cache["gradient"].copy()
        if evaluations >= settings["max_function_evaluations"]:
            raise _BudgetExhausted
        evaluations += 1
        value, gradient = value_and_gradient(vector)
        value, gradient = float(value), np.asarray(gradient, dtype=np.float64)
        if gradient.shape != vector.shape:
            raise ValueError("objective gradient shape differs from parameters")
        finite = np.isfinite(value) and np.isfinite(gradient).all()
        if not finite:
            nonfinite_trials += 1
            # A rejected line-search trial must never become a selected model.
            value, gradient = float("inf"), np.zeros_like(vector)
        cache = dict(
            vector=vector.copy(),
            value=value,
            gradient=gradient.copy(),
            finite=bool(finite),
        )
        return value, gradient.copy()

    def record(point):
        nonlocal development_passes, invalid_development, last_checked
        if last_checked is not None and np.array_equal(last_checked, point["vector"]):
            return
        if development_passes >= settings["max_development_passes"]:
            raise RuntimeError("checkpoint schedule exceeded development budget")
        development_passes += 1
        last_checked = point["vector"].copy()
        scores = development_scores(point["vector"])
        values = [scores["mean"], *scores["recordings"].values()]
        if not scores["recordings"] or not np.isfinite(values).all():
            if not trace:
                raise ValueError("nonfinite or empty initial development evidence")
            invalid_development += 1
            return
        vectors.append(point["vector"].copy())
        trace.append(
            dict(
                step=point["iteration"],
                function_evaluations=evaluations,
                training_mse=point["value"],
                training_gradient_inf_norm=float(np.max(np.abs(point["gradient"]))),
                validation_rollout_mse=float(scores["mean"]),
                recording_losses={
                    str(k): float(v) for k, v in scores["recordings"].items()
                },
            )
        )

    objective(initial)
    if not cache["finite"]:
        raise ValueError("initial training objective or gradient is nonfinite")
    accepted = {**copy.deepcopy(cache), "iteration": 0}
    record(accepted)

    def callback(vector):
        nonlocal accepted, next_checkpoint
        objective(vector)
        if not cache["finite"]:
            raise _InvalidAcceptedPoint
        accepted = {**copy.deepcopy(cache), "iteration": accepted["iteration"] + 1}
        if evaluations >= next_checkpoint:
            record(accepted)
            next_checkpoint = (
                evaluations // settings["checkpoint_every_evaluations"] + 1
            ) * settings["checkpoint_every_evaluations"]

    message, reason = "", ""
    try:
        _, _, info = fmin_l_bfgs_b(
            objective,
            np.asarray(initial, dtype=np.float64),
            m=settings["memory"],
            factr=settings["relative_loss_factr"],
            pgtol=settings["gradient_tolerance"],
            maxfun=settings["max_function_evaluations"],
            maxiter=settings["max_iterations"],
            maxls=settings["max_line_search_steps"],
            callback=callback,
        )
        message = str(info["task"])
        if info["warnflag"] == 0:
            reason = (
                "gradient_tolerance"
                if np.max(np.abs(accepted["gradient"]))
                <= settings["gradient_tolerance"]
                else "relative_loss_tolerance"
            )
        elif evaluations >= settings["max_function_evaluations"]:
            reason = "evaluation_budget"
        elif accepted["iteration"] >= settings["max_iterations"]:
            reason = "iteration_budget"
        else:
            reason = "line_search_failed"
    except _BudgetExhausted:
        reason, message = (
            "evaluation_budget",
            "Hard objective/gradient call budget reached.",
        )
    except _InvalidAcceptedPoint:
        reason, message = (
            "nonfinite_accepted_point",
            "Solver proposed a nonfinite accepted point; retained earlier finite checkpoint.",
        )
    record(accepted)
    chosen = min(range(len(trace)), key=lambda i: trace[i]["validation_rollout_mse"])
    winners = {
        recording: min(
            range(len(trace)), key=lambda i: trace[i]["recording_losses"][recording]
        )
        for recording in trace[0]["recording_losses"]
    }
    ordered = sorted(t["validation_rollout_mse"] for t in trace)
    report = dict(
        settings=copy.deepcopy(settings),
        termination=dict(
            reason=reason,
            message=message,
            accepted_iterations=accepted["iteration"],
            final_training_mse=accepted["value"],
            final_training_gradient_inf_norm=float(
                np.max(np.abs(accepted["gradient"]))
            ),
            gradient_converged=bool(
                np.max(np.abs(accepted["gradient"])) <= settings["gradient_tolerance"]
            ),
        ),
        resource_usage=dict(
            training_value_gradient_calls=evaluations,
            cache_hits=cache_hits,
            development_passes=development_passes,
            nonfinite_trials=nonfinite_trials,
            nonfinite_development_checkpoints=invalid_development,
        ),
        selected_index=chosen,
        selected_step=trace[chosen]["step"],
        validation_rollout_mse=trace[chosen]["validation_rollout_mse"],
        selected_training_mse=trace[chosen]["training_mse"],
        selected_training_gradient_inf_norm=trace[chosen]["training_gradient_inf_norm"],
        selection=dict(
            recording_winner_indices=winners,
            recording_winner_disagreement=len(set(winners.values())) > 1,
            runner_up_loss_gap=None if len(ordered) < 2 else ordered[1] - ordered[0],
            meaning="Development-only checkpoint ranking; recording agreement is not calibrated confidence or proof of evaluation accuracy.",
        ),
        trace=trace,
    )
    return vectors[chosen], report


def _fit_bounded_sequence_model(
    train, development, initial, error_scale, recording_ids
):
    """Fit the supplied initialized recurrence with one maintained internal recipe."""
    scale = np.asarray(error_scale, dtype=float)
    ids = np.asarray(recording_ids)
    if (
        scale.shape != train.future_states.shape[1:]
        or not np.isfinite(scale).all()
        or np.any(scale <= 0)
    ):
        raise ValueError(
            "error_scale must contain positive finite horizon/channel values"
        )
    if train.dt_s != development.dt_s or any(
        getattr(train, k).shape[1:] != getattr(development, k).shape[1:]
        for k in ("past_states", "past_inputs", "future_inputs", "future_states")
    ):
        raise ValueError("training and development contracts differ")
    if (
        ids.shape != (len(development.past_states),)
        or ids.dtype.kind not in "US"
        or np.any(ids == "")
    ):
        raise ValueError("development recording identities must match windows")
    # Fitting precision is local; callers' JAX precision configuration is preserved.
    with jax.enable_x64(True):
        vector, unravel = ravel_pytree(jax.tree.map(jnp.asarray, initial.params))
        norms = jax.tree.map(jnp.asarray, initial.norms)
        batches = [
            tuple(
                jnp.asarray(getattr(batch, k))
                for k in (
                    "past_states",
                    "past_inputs",
                    "future_inputs",
                    "future_states",
                )
            )
            for batch in (train, development)
        ]

        def loss(parameters):
            x, up, uf, target = batches[0]
            prediction = _rollout(unravel(parameters), norms, initial.kind, x, up, uf)
            return jnp.mean(((prediction - target) / scale) ** 2)

        value_gradient = jax.jit(jax.value_and_grad(loss))

        @jax.jit
        def development_errors(parameters):
            x, up, uf, target = batches[1]
            prediction = _rollout(unravel(parameters), norms, initial.kind, x, up, uf)
            return jnp.mean(((prediction - target) / scale) ** 2, axis=(1, 2))

        def scores(parameters):
            errors = np.asarray(development_errors(parameters))
            return dict(
                mean=float(errors.mean()),
                recordings={
                    str(k): float(errors[ids == k].mean()) for k in sorted(set(ids))
                },
            )

        selected, report = _run_lbfgs(
            np.asarray(vector), value_gradient, scores, _BOUNDED_FIT
        )
        model = SequenceModel(
            initial.kind,
            train.dt_s,
            initial.history_steps,
            jax.tree.map(np.asarray, unravel(selected)),
            jax.tree.map(np.asarray, norms),
        )
    report.update(
        kind=initial.kind,
        objective="rollout",
        optimizer=_BOUNDED_FIT["optimizer"],
        precision="float64",
        parameter_count=sum(p.size for p in model.params.values()),
        loss_scale_mode="explicit_horizon_channel",
        error_scale=scale.tolist(),
    )
    report["resource_usage"]["training_window_gradient_evaluations"] = report[
        "resource_usage"
    ]["training_value_gradient_calls"] * len(train.past_states)
    report["resource_usage"]["development_window_evaluations"] = report[
        "resource_usage"
    ]["development_passes"] * len(development.past_states)
    return model, report
