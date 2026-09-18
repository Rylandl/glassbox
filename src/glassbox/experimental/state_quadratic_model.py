"""Frozen autonomous-state-quadratic-v1 research candidate, not a public recipe.

One direct autonomous quadratic output supplements the frozen bilinear output.
A private copy of its module supplies the unchanged optimizer, cache matching,
calibration and saved-candidate contract without mutating the historical module.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .._sequence_model import KIND, _features, _filter, initialize_sequence_model

_spec = importlib.util.spec_from_file_location(
    f"{__package__}._state_quadratic_shared",
    Path(__file__).with_name("state_input_model.py"),
)
_shared = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_shared)

EXPERIMENT = "autonomous-state-quadratic-v1"
RECIPE = {
    **_shared.RECIPE,
    "id": EXPERIMENT,
    "autonomous": "current_state_upper_triangular",
}
_MODEL_FORMAT = "glassbox-state-quadratic-sequence-v1"
_FORMAT = "glassbox-state-quadratic-candidate-v1"
CandidateFitError = _shared.CandidateFitError
interaction_features = _shared.interaction_features


def autonomous_features(x, xp=jnp):
    """Uncentered x_i*x_j, i<=j, in lexicographic order, including squares once."""
    i, j = xp.triu_indices(x.shape[-1])
    return x[..., i] * x[..., j]


def _rollout(params, norms, past, past_u, future_u, delay, memory=None):
    x = (past - norms["state_mean"]) / norms["state_scale"]
    up = (past_u - norms["input_mean"]) / norms["input_scale"]
    uf = (future_u - norms["input_mean"]) / norms["input_scale"]
    h = _filter(params, norms, x, up, delay, memory)
    x, up = x[:, -delay - 1 :], up[:, -delay:]

    def step(carry, command):
        current, history, inputs, hidden = carry
        z = _features(current, command, history, inputs, hidden)
        z = z / norms["feature_scale"]
        q = interaction_features(current, command) / norms["interaction_scale"]
        a = autonomous_features(current) / norms["autonomous_scale"]
        delta = z @ params["linear"] + q @ params["interaction"]
        delta = delta + a @ params["autonomous"] + params["bias"]
        delta = delta + jnp.tanh(z @ params["w1"] + params["b1"]) @ params["w2"]
        predicted = current + delta * norms["delta_scale"]
        hidden = jnp.tanh(z @ params["memory"] + params["memory_bias"])
        return (
            predicted,
            jnp.concatenate((history[:, 1:], current[:, None]), axis=1),
            jnp.concatenate((inputs[:, 1:], command[:, None]), axis=1),
            hidden,
        ), predicted

    _, predicted = jax.lax.scan(step, (x[:, -1], x[:, :-1], up, h), uf.swapaxes(0, 1))
    return predicted.swapaxes(0, 1) * norms["state_scale"] + norms["state_mean"]


class QuadraticSequenceModel(_shared.BilinearSequenceModel):
    """The unchanged bilinear sequence contract plus one autonomous output."""

    def __post_init__(self):
        super().__post_init__()
        d = len(self.norms["state_mean"])
        count = d * (d + 1) // 2
        coefficients = np.asarray(self.params.get("autonomous", []))
        if coefficients.shape != (count, d) or not np.isfinite(coefficients).all():
            raise ValueError(
                "autonomous coefficients must be finite [state*(state+1)/2,state]"
            )
        scale = np.asarray(self.norms.get("autonomous_scale", []))
        if (
            scale.shape != (count,)
            or not np.isfinite(scale).all()
            or np.any(scale <= 0)
        ):
            raise ValueError(
                "autonomous scales must be finite positive [state*(state+1)/2]"
            )


def initialize_candidate(
    batch, *, seed=0, width=32, memory=8, ridge=1.0, delay_steps=None
):
    """Joint ridge of base, bilinear and quadratic features with fixed draws."""
    baseline = initialize_sequence_model(
        batch,
        seed=seed,
        width=width,
        memory=memory,
        ridge=ridge,
        delay_steps=delay_steps,
    )
    params = {k: np.array(v, copy=True) for k, v in baseline.params.items()}
    norms = {k: np.array(v, copy=True) for k, v in baseline.norms.items()}
    context, p = baseline.history_steps, baseline.delay_steps
    physical = np.concatenate((batch.past_states, batch.future_states), axis=1)
    x = (physical - norms["state_mean"]) / norms["state_scale"]
    u = np.concatenate((batch.past_inputs, batch.future_inputs), axis=1)
    u = (u - norms["input_mean"]) / norms["input_scale"]
    current, commands = x[:, context:-1], u[:, context:]
    hidden = np.zeros((len(current), memory))
    features = np.stack(
        [
            _features(
                x[:, context + t],
                u[:, context + t],
                x[:, context + t - p : context + t],
                u[:, context + t - p : context + t],
                hidden,
                xp=np,
            )
            for t in range(current.shape[1])
        ],
        axis=1,
    ).reshape(-1, len(norms["feature_scale"]))
    products = interaction_features(current, commands, xp=np).reshape(len(features), -1)
    autonomous = autonomous_features(current, xp=np).reshape(len(features), -1)
    for name, values in (("interaction", products), ("autonomous", autonomous)):
        scale = values.std(axis=0)
        norms[f"{name}_scale"] = np.where(scale > 1e-8, scale, 1.0)
    design = np.column_stack(
        (
            features / norms["feature_scale"],
            products / norms["interaction_scale"],
            autonomous / norms["autonomous_scale"],
            np.ones(len(features)),
        )
    )
    penalty = np.diag(np.r_[np.full(design.shape[1] - 1, ridge), 0.0])
    # Match the frozen subtraction order before either normalization.
    delta = (batch.future_states - physical[:, context:-1]) / norms["state_scale"]
    target = (delta / norms["delta_scale"]).reshape(len(features), -1)
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ target)
    base_end = features.shape[-1]
    interaction_end = base_end + products.shape[-1]
    params.update(
        linear=coefficients[:base_end],
        interaction=coefficients[base_end:interaction_end],
        autonomous=coefficients[interaction_end:-1],
        bias=coefficients[-1],
    )
    return QuadraticSequenceModel(KIND, batch.dt_s, context, params, norms, p)


class CandidateDynamics(_shared.CandidateDynamics):
    """Saved quadratic research model; no public recipe or update qualification."""


# These bindings belong exclusively to the private module. Its existing methods
# resolve them dynamically, preserving the optimizer and wrapper implementations.
_shared.EXPERIMENT = EXPERIMENT
_shared.RECIPE = RECIPE
_shared._MODEL_FORMAT = _MODEL_FORMAT
_shared._FORMAT = _FORMAT
_shared._rollout = _rollout
_shared.BilinearSequenceModel = QuadraticSequenceModel
_shared.initialize_candidate = initialize_candidate
_shared.CandidateDynamics = CandidateDynamics
fit_candidate_sequence = _shared.fit_candidate_sequence
fit_candidate = _shared.fit_candidate
