"""A general Euclidean transition GP with separate empirical support diagnostics.

No force law, platform family, actuator response, or residual base is supplied.
The inputs are [state, command, optional context]; targets are complete next
states at one fixed interval. Context may contain causally available history.

This small-data reference implementation uses exact conditioning, independent
standardized outputs sharing one ARD kernel, and plug-in maximum-likelihood
hyperparameters. Its intervals are model-conditional, not distribution-free
coverage guarantees. It does not account for noisy inputs or hyperparameter
uncertainty. Exact fitting costs O(N^3); it is not a streaming implementation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import solve_triangular


def _matrix(value, name: str) -> np.ndarray:
    array = np.array(value, dtype=float, copy=True)
    if array.ndim != 2 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite two-dimensional array")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class TransitionSamples:
    """Observed transitions; rows may be drawn from separate trajectories.

    All rows must use the same channel ordering, units, timing and configuration.
    States are Euclidean coordinates. Angle/quaternion geometry and automatic
    construction of causal context from logs are intentionally not implemented.
    """

    states: np.ndarray
    commands: np.ndarray
    next_states: np.ndarray
    dt_s: float
    context: np.ndarray | None = None

    def __post_init__(self):
        for name in ("states", "commands", "next_states"):
            object.__setattr__(self, name, _matrix(getattr(self, name), name))
        n, state_size = self.states.shape
        context = np.empty((n, 0)) if self.context is None else self.context
        object.__setattr__(self, "context", _matrix(context, "context"))
        if n < 3 or state_size == 0:
            raise ValueError("at least three transitions and one state are required")
        if self.next_states.shape != self.states.shape:
            raise ValueError("next_states must have the same shape as states")
        if self.commands.shape[0] != n or self.context.shape[0] != n:
            raise ValueError("state, command and context row counts must match")
        if not np.isfinite(self.dt_s) or self.dt_s <= 0:
            raise ValueError("dt_s must be finite and positive")

    @property
    def features(self) -> np.ndarray:
        return np.concatenate((self.states, self.commands, self.context), axis=1)


class TransitionPrediction(NamedTuple):
    mean: jax.Array
    function_variance: jax.Array
    observation_variance: jax.Array


class LocalResponse(NamedTuple):
    """Derivatives in supplied input/output units; covariance is GP-conditional."""

    jacobian: jax.Array
    hessian: jax.Array
    jacobian_covariance: jax.Array


@dataclass(frozen=True)
class NeighborhoodSupport:
    """Geometry only, in training-standardized coordinates; no probability claim.

    Columns of basis are eigenvectors, ordered from least to most observed
    variation. Small eigenvalues flag thin directions, not zero model error.
    Neighbor selection is discrete and is not a differentiable planning cost.
    """

    nearest_distance: np.ndarray
    neighborhood_radius: np.ndarray
    eigenvalues: np.ndarray
    basis: np.ndarray
    query_offset_in_basis: np.ndarray
    neighbor_count: int


def _kernel(left, right, theta, kind: str):
    dimension = left.shape[1]
    length = jnp.exp(theta[:dimension])
    distance_sq = jnp.sum(
        ((left[:, None, :] - right[None, :, :]) / length) ** 2, axis=-1
    )
    amplitude_sq = jnp.exp(2 * theta[-2])
    if kind == "rbf":
        return amplitude_sq * jnp.exp(-0.5 * distance_sq)
    if kind == "rq":
        # Rational quadratic: a scale mixture of RBF kernels, with a fitted
        # generic shape parameter. It has no mechanism-specific basis terms.
        shape = jnp.exp(theta[dimension])
        return amplitude_sq * jnp.exp(-shape * jnp.log1p(distance_sq / (2 * shape)))
    if kind != "matern52":
        raise ValueError(f"unknown transition kernel: {kind}")
    # A tiny addition avoids undefined sqrt derivatives at coincident inputs.
    radius = jnp.sqrt(5 * distance_sq + 1e-12)
    return amplitude_sq * (1 + radius + radius**2 / 3) * jnp.exp(-radius)


def _numerical_jitter(features, theta):
    # Entry-wise kernel roundoff can accumulate across rows. In particular,
    # fused optimization and unfused conditioning can round differently. Keep
    # this numerical regularizer distinct from learned observation variance.
    return jnp.maximum(
        1e-6,
        2 * len(features) * jnp.finfo(features.dtype).eps * jnp.exp(2 * theta[-2]),
    )


def _factor(features, theta, kind: str):
    covariance = _kernel(features, features, theta, kind)
    diagonal = jnp.exp(2 * theta[-1]) + _numerical_jitter(features, theta)
    return jnp.linalg.cholesky(covariance + diagonal * jnp.eye(len(features)))


def _nll(theta, features, targets, kind: str):
    chol = _factor(features, theta, kind)
    whitened = solve_triangular(chol, targets, lower=True)
    outputs = targets.shape[1]
    return (
        0.5 * jnp.sum(whitened**2)
        + outputs * jnp.sum(jnp.log(jnp.diag(chol)))
        + 0.5 * targets.size * jnp.log(2 * jnp.pi)
    )


@partial(jax.jit, static_argnames=("kind", "steps"))
def _optimize(features, targets, initial, *, kind, steps):
    dimension = features.shape[1]
    shape_low, shape_high = ([0.02], [100.0]) if kind == "rq" else ([], [])
    low = jnp.log(jnp.array([0.05] * dimension + shape_low + [0.05, 0.001]))
    high = jnp.log(jnp.array([20.0] * dimension + shape_high + [20.0, 1.0]))
    value_grad = jax.value_and_grad(_nll)

    def update(carry, index):
        theta, first, second, best, best_loss = carry
        loss, gradient = value_grad(theta, features, targets, kind)
        finite = jnp.isfinite(loss) & jnp.all(jnp.isfinite(gradient))
        better = finite & (loss < best_loss)
        best = jnp.where(better, theta, best)
        best_loss = jnp.where(better, loss, best_loss)
        gradient = jnp.where(finite, gradient, jnp.zeros_like(gradient))
        first = 0.9 * first + 0.1 * gradient
        second = 0.999 * second + 0.001 * gradient**2
        corrected_first = first / (1 - 0.9 ** (index + 1))
        corrected_second = second / (1 - 0.999 ** (index + 1))
        theta = jnp.clip(
            theta - 0.04 * corrected_first / (jnp.sqrt(corrected_second) + 1e-8),
            low,
            high,
        )
        return (theta, first, second, best, best_loss), loss

    carry, trace = jax.lax.scan(
        update,
        (initial, jnp.zeros_like(initial), jnp.zeros_like(initial), initial, jnp.inf),
        jnp.arange(steps),
    )
    theta, _, _, best, loss = carry
    final_loss = _nll(theta, features, targets, kind)
    better = jnp.isfinite(final_loss) & (final_loss < loss)
    return jnp.where(better, theta, best), jnp.where(better, final_loss, loss), trace


@dataclass(frozen=True)
class GaussianTransition:
    """JAX-differentiable fixed-interval prediction, independently of a platform."""

    kernel: str
    dt_s: float
    state_size: int
    command_size: int
    context_size: int
    feature_mean: jax.Array
    feature_scale: jax.Array
    target_mean: jax.Array
    target_scale: jax.Array
    features: jax.Array
    theta: jax.Array
    chol: jax.Array
    alpha: jax.Array
    fit_report: dict
    mean_mode: str = "absolute"

    def fingerprint(self) -> str:
        """Content identity for invalidating residual calibration after refitting.

        Includes timing, dimensions, dtype and all predictive arrays. Human
        channel names/units remain a separate consumer-supplied input contract.
        """
        digest = hashlib.sha256()
        metadata = [
            self.kernel,
            self.dt_s,
            self.state_size,
            self.command_size,
            self.context_size,
        ]
        digest.update(json.dumps(metadata).encode())
        if self.mean_mode != "absolute":
            digest.update(self.mean_mode.encode())
        for name in (
            "feature_mean",
            "feature_scale",
            "target_mean",
            "target_scale",
            "features",
            "theta",
            "chol",
            "alpha",
        ):
            value = np.ascontiguousarray(getattr(self, name))
            digest.update(json.dumps([name, value.dtype.str, value.shape]).encode())
            digest.update(value.tobytes())
        return digest.hexdigest()

    def _features(self, states, commands, context):
        states, commands = jnp.asarray(states), jnp.asarray(commands)
        if states.ndim < 1 or states.shape[-1] != self.state_size:
            raise ValueError("state width does not match the fitted model")
        if commands.shape != (*states.shape[:-1], self.command_size):
            raise ValueError(
                "command shape does not match state batch and channel count"
            )
        if context is None:
            if self.context_size:
                raise ValueError("this model requires context at each prediction")
            context = jnp.empty((*states.shape[:-1], 0))
        context = jnp.asarray(context)
        if context.shape != (*states.shape[:-1], self.context_size):
            raise ValueError("context shape does not match the fitted model")
        return jnp.concatenate((states, commands, context), axis=-1)

    def _predict_features(self, features):
        features = jnp.asarray(features)
        batch_shape = features.shape[:-1]
        normalized = ((features - self.feature_mean) / self.feature_scale).reshape(
            -1, self.features.shape[1]
        )
        cross = _kernel(normalized, self.features, self.theta, self.kernel)
        mean = (cross @ self.alpha) * self.target_scale + self.target_mean
        if self.mean_mode == "increment":
            mean = mean + features.reshape(-1, features.shape[-1])[:, : self.state_size]
        projected = solve_triangular(self.chol, cross.T, lower=True)
        variance = jnp.maximum(
            jnp.exp(2 * self.theta[-2]) - jnp.sum(projected**2, axis=0), 0
        )
        function_variance = variance[:, None] * self.target_scale**2
        observation_variance = (
            variance[:, None] + jnp.exp(2 * self.theta[-1])
        ) * self.target_scale**2
        shape = (*batch_shape, self.state_size)
        return TransitionPrediction(
            mean.reshape(shape),
            function_variance.reshape(shape),
            observation_variance.reshape(shape),
        )

    def predict(self, states, commands, *, context=None) -> TransitionPrediction:
        """Conditional one-step marginals, not propagated trajectory uncertainty."""
        return self._predict_features(self._features(states, commands, context))

    def mean_rollout(self, initial_state, commands, *, context=None):
        """Compose the learned mean; deliberately returns no uncertainty band."""
        commands = jnp.asarray(commands)
        if commands.ndim != 2 or commands.shape[1] != self.command_size:
            raise ValueError("commands must have shape (steps, command_size)")
        if context is None:
            if self.context_size:
                raise ValueError("this model requires context for the rollout")
            context = jnp.empty((len(commands), 0))
        context = jnp.asarray(context)
        if context.shape != (len(commands), self.context_size):
            raise ValueError("context must have shape (steps, context_size)")
        initial_state = jnp.asarray(initial_state)
        if initial_state.shape != (self.state_size,):
            raise ValueError("initial_state must be a single state vector")

        def advance(state, item):
            command, auxiliary = item
            mean = self.predict(state, command, context=auxiliary).mean
            return mean, mean

        _, states = jax.lax.scan(advance, initial_state, (commands, context))
        return jnp.concatenate((initial_state[None, :], states), axis=0)

    def local_response(self, state, command, *, context=None) -> LocalResponse:
        """Mean slope/curvature and slope covariance at one query.

        Hessians are derivatives of the fitted mean, not independently measured
        curvature. Jacobian covariance omits hyperparameter/model uncertainty.
        """
        features = self._features(state, command, context)
        if features.ndim != 1:
            raise ValueError("local_response expects one state/command pair")

        def mean(point):
            return self._predict_features(point).mean

        jacobian = jax.jacfwd(mean)(features)
        hessian = jax.jacfwd(jax.jacrev(mean))(features)

        def cross(point):
            normalized = (point - self.feature_mean) / self.feature_scale
            return _kernel(normalized[None, :], self.features, self.theta, self.kernel)[
                0
            ]

        derivative = jax.jacfwd(cross)(features)
        projected = solve_triangular(self.chol, derivative, lower=True)
        factor = 5.0 / 3.0 if self.kernel == "matern52" else 1.0
        prior = jnp.diag(
            factor
            * jnp.exp(2 * self.theta[-2])
            / (jnp.exp(self.theta[: self.features.shape[1]]) * self.feature_scale) ** 2
        )
        covariance = prior - projected.T @ projected
        covariance = (covariance + covariance.T) / 2
        # Only remove numerical negative eigenvalues; no evidence is added.
        eigenvalues, basis = jnp.linalg.eigh(covariance)
        covariance = (basis * jnp.maximum(eigenvalues, 0)) @ basis.T
        return LocalResponse(
            jacobian,
            hessian,
            self.target_scale[:, None, None] ** 2 * covariance[None, :, :],
        )

    def support(self, states, commands, *, context=None, neighbors=32):
        """Host-side local input geometry, deliberately separate from GP variance."""
        query = np.asarray(self._features(states, commands, context))
        if query.ndim == 1:
            query = query[None, :]
        if query.ndim != 2 or len(query) == 0 or not np.all(np.isfinite(query)):
            raise ValueError("support expects finite vectors or a matrix of queries")
        if not isinstance(neighbors, int) or neighbors < 2:
            raise ValueError("neighbors must be at least 2")
        neighbors = min(neighbors, len(self.features))
        query = (query - np.asarray(self.feature_mean)) / np.asarray(self.feature_scale)
        reference = np.asarray(self.features)
        values = []
        for point in query:
            distance = np.linalg.norm(reference - point, axis=1)
            indices = np.argsort(distance, kind="stable")[:neighbors]
            local = reference[indices]
            center = local.mean(axis=0)
            centered = local - center
            eigenvalues, basis = np.linalg.eigh(centered.T @ centered / neighbors)
            values.append(
                (
                    distance[indices[0]],
                    distance[indices[-1]],
                    eigenvalues,
                    basis,
                    (point - center) @ basis,
                )
            )
        arrays = [np.asarray([value[index] for value in values]) for index in range(5)]
        return NeighborhoodSupport(*arrays, neighbors)

    def save(self, path: str | Path):
        """Save the experimental artifact without pickle or a platform decoder."""
        metadata = {
            "format": (
                "glassbox.experimental.transition_gp.v3"
                if self.mean_mode == "increment"
                else "glassbox.experimental.transition_gp.v2"
                if self.kernel == "rq"
                else "glassbox.experimental.transition_gp.v1"
            ),
            **{
                name: getattr(self, name)
                for name in (
                    "kernel",
                    "dt_s",
                    "state_size",
                    "command_size",
                    "context_size",
                    "fit_report",
                )
            },
        }
        if self.mean_mode != "absolute":
            metadata["mean_mode"] = self.mean_mode
        arrays = {
            name: np.asarray(getattr(self, name))
            for name in (
                "feature_mean",
                "feature_scale",
                "target_mean",
                "target_scale",
                "features",
                "theta",
                "chol",
                "alpha",
            )
        }
        with Path(path).open("wb") as stream:
            np.savez_compressed(stream, metadata=json.dumps(metadata), **arrays)

    @classmethod
    def load(cls, path: str | Path):
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"]))
            version = metadata.pop("format")
            if version not in (
                "glassbox.experimental.transition_gp.v1",
                "glassbox.experimental.transition_gp.v2",
                "glassbox.experimental.transition_gp.v3",
            ):
                raise ValueError("unsupported transition GP format")
            allowed = (
                ("rbf", "matern52")
                if version.endswith("v1")
                else ("rbf", "matern52", "rq")
            )
            if metadata["kernel"] not in allowed:
                raise ValueError("unsupported kernel for this transition GP format")
            if metadata.get("mean_mode", "absolute") not in ("absolute", "increment"):
                raise ValueError("unsupported GP mean mode")
            if metadata.get(
                "mean_mode", "absolute"
            ) != "absolute" and not version.endswith("v3"):
                raise ValueError("increment mean requires transition GP format v3")
            if (
                archive["features"].dtype == np.float64
                and not jax.config.jax_enable_x64
            ):
                raise ValueError("loading this float64 model requires JAX_ENABLE_X64=1")
            arrays = {
                name: jnp.asarray(archive[name])
                for name in archive.files
                if name != "metadata"
            }
        return cls(**metadata, **arrays)


def fit_transition_gp(
    samples: TransitionSamples,
    *,
    kernel="matern52",
    steps=160,
    restarts=2,
    mean_mode="absolute",
) -> GaussianTransition:
    """Fit only the supplied transitions; all scaling/hyperparameters use them.

    Restarts use deterministic initial standardized length scales spanning
    0.5--2.0; selection uses training marginal likelihood, never test labels.
    Numerical jitter is at least 1e-6 in standardized output variance and scales
    with matrix size, signal variance and dtype precision. It is reported
    separately from estimated observation noise.

    mean_mode="increment" fits next_state - state and adds the current state
    back in every prediction. This generic identity mean does not impose a
    force law, enforce a hard rate bound, or change the external next-state
    contract. Derivatives and rollouts include the identity term.
    """
    if kernel not in ("rbf", "matern52", "rq"):
        raise ValueError("kernel must be 'rbf', 'matern52', or 'rq'")
    if mean_mode not in ("absolute", "increment"):
        raise ValueError("mean_mode must be 'absolute' or 'increment'")
    if not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if not isinstance(restarts, int) or restarts < 1:
        raise ValueError("restarts must be a positive integer")
    raw = samples.features
    center, scale = raw.mean(axis=0), raw.std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    raw_targets = (
        samples.next_states - samples.states
        if mean_mode == "increment"
        else samples.next_states
    )
    target_mean = raw_targets.mean(axis=0)
    target_scale = raw_targets.std(axis=0)
    target_scale = np.where(target_scale > 1e-8, target_scale, 1.0)
    features = jnp.asarray((raw - center) / scale)
    targets = jnp.asarray((raw_targets - target_mean) / target_scale)
    trials = []
    candidates = []
    for length in np.geomspace(0.5, 2.0, restarts):
        shape = [1.0] if kernel == "rq" else []
        initial = jnp.log(jnp.array([length] * raw.shape[1] + shape + [1.0, 0.05]))
        theta, loss, trace = _optimize(
            features, targets, initial, kind=kernel, steps=steps
        )
        loss = float(loss)
        if not np.isfinite(loss) or not np.all(np.isfinite(np.asarray(theta))):
            raise FloatingPointError("nonfinite GP optimization result")
        trials.append(
            {
                "initial_length": float(length),
                "nll": loss,
                "loss_trace": np.asarray(trace).tolist(),
            }
        )
        candidates.append(theta)
    selected = int(np.argmin([trial["nll"] for trial in trials]))
    theta = candidates[selected]
    chol = _factor(features, theta, kernel)
    alpha = solve_triangular(
        chol.T, solve_triangular(chol, targets, lower=True), lower=False
    )
    if not np.all(np.isfinite(np.asarray(alpha))):
        raise FloatingPointError("nonfinite GP conditioning result")
    report = {
        "training_rows": len(raw),
        "steps": steps,
        "selected_restart": selected,
        "trials": trials,
        "nll": trials[selected]["nll"],
        "length_scales_standardized": np.exp(
            np.asarray(theta[: raw.shape[1]])
        ).tolist(),
        "signal_std_standardized": float(jnp.exp(theta[-2])),
        "noise_std_output_units": (jnp.exp(theta[-1]) * target_scale).tolist(),
        "numerical_jitter_standardized": float(_numerical_jitter(features, theta)),
        "dtype": str(features.dtype),
    }
    if kernel == "rq":
        report["rational_quadratic_alpha"] = float(jnp.exp(theta[raw.shape[1]]))
    report["mean_mode"] = mean_mode
    return GaussianTransition(
        kernel,
        samples.dt_s,
        samples.states.shape[1],
        samples.commands.shape[1],
        samples.context.shape[1],
        jnp.asarray(center),
        jnp.asarray(scale),
        jnp.asarray(target_mean),
        jnp.asarray(target_scale),
        features,
        theta,
        chol,
        alpha,
        report,
        mean_mode,
    )
