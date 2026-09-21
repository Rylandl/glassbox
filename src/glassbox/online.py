"""A bounded, causal fitting session for the same shared-physics dynamics model."""

from __future__ import annotations

import copy
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from ._dynamics import (
    GRAVITY,
    VehicleSequenceModel,
    _rollout,
    current_features,
    initialize,
    quadratic_features,
    time_constants,
)
from ._learner_arrays import array_fingerprint, load_arrays, save_arrays
from ._training import SequenceBatch, validate_window_consistency
from .learner import _contract, _validate_rotations, steps_for
from .recordings import WindowKey

_FORMAT = "glassbox-online-fit-v4"
_FIELDS = ("past_states", "past_inputs", "future_inputs", "future_states")
_RECIPE = dict(
    bootstrap_s=0.25,
    horizon_s=0.05,
    bootstrap_windows=32,
    recent_windows=32,
    batch_size=64,
    proposals=1,
    cg_iterations=4,
    huber_delta=1.0,
    loss=dict(
        groups=[3, 3, 9],
        weighting="equal velocity, body-rate and chordal rotation groups",
        penalty="Huber of normalized group L2 norm",
        scale="fixed bootstrap group hold-change RMS",
        floor="0.01 times max(raw group origin spread, 1e-4)",
        rotation_scale="sqrt(2) times chordal group scale",
    ),
    initial_damping=1.0,
    damping_bounds=[1e-8, 1e8],
    minimum_gain_ratio=0.1,
    trust_fraction=0.5,
    minimum_trust_radius=1.0,
    conditioning=dict(
        calls_per_observation=1,
        scales=["feature_scale", "quadratic_scale", "output_scale"],
        statistic="role-balanced uncentered measured-cache RMS",
        update="elementwise maximum with existing scales",
        hidden_scales="fixed at 1",
        compensation="exact feature/output row and column rescaling",
        acceptance="commit coordinates only with an accepted finite proposal",
    ),
)


@partial(jax.jit, static_argnames=("delay", "dt_s"))
def _predict(params, norms, past, inputs, future, *, delay, dt_s):
    return _rollout(params, norms, past, inputs, future, delay, dt_s)


def _residual(params, norms, data, scale, delay, dt_s):
    return (_rollout(params, norms, *data[:3], delay, dt_s) - data[3]) / scale


def _group_squared(residual):
    """Squared norms of velocity, body-rate and flattened rotation residuals."""
    return jnp.stack(
        [
            jnp.sum(residual[..., a:b] ** 2, axis=-1)
            for a, b in ((0, 3), (3, 6), (6, 15))
        ],
        axis=-1,
    )


def _huber(residual):
    """Radial group-Huber values, with finite derivatives at zero residual."""
    squared = _group_squared(residual)
    return jnp.where(
        squared <= 1, 0.5 * squared, jnp.sqrt(jnp.maximum(1.0, squared)) - 0.5
    )


@partial(jax.jit, static_argnames=("delay", "dt_s"))
def _recondition(params, norms, data, weights, *, delay, dt_s):
    """Grow three coordinate scales from completed observations, preserving the map.

    Measured histories and targets set feature statistics; no model trajectory or
    future observation enters. The known tanh bound keeps hidden scales fixed.
    An invalid transformation returns the original arrays and a false flag, so
    callers can retain the normal proposal accounting while forcing rejection.
    """
    past, past_inputs, future_inputs, targets = data
    states = jnp.concatenate((past, targets[:, :-1]), axis=1)
    commands = jnp.concatenate((past_inputs, future_inputs), axis=1)
    tau = time_constants(params)

    def filter_step(applied, command):
        return command + (applied - command) * jnp.exp(-dt_s / tau), applied

    _, filtered = jax.lax.scan(filter_step, commands[:, 0], commands.swapaxes(0, 1))
    features = current_features(states, commands, filtered.swapaxes(0, 1), norms)
    current = features[:, delay:]
    previous = jnp.stack(
        [features[:, t - delay : t] for t in range(delay, features.shape[1])],
        axis=1,
    )
    differences = (previous - current[:, :, None]).reshape(
        (*current.shape[:2], delay * current.shape[2])
    )
    sampled = jnp.concatenate((current, differences), axis=-1)

    def rms(values):
        return jnp.sqrt(
            jnp.sum(weights[:, None, None] * values**2, axis=(0, 1)) / values.shape[1]
        )

    feature_scale = jnp.concatenate(
        (
            jnp.maximum(norms["feature_scale"][: sampled.shape[-1]], rms(sampled)),
            norms["feature_scale"][sampled.shape[-1] :],
        )
    )
    quadratic_scale = jnp.maximum(
        norms["quadratic_scale"], rms(quadratic_features(current))
    )
    preceding = jnp.concatenate((past[:, -1:], targets[:, :-1]), axis=1)
    rotation = preceding[..., 6:].reshape((*preceding.shape[:-1], 3, 3))
    world_force = (targets[..., :3] - preceding[..., :3]) / dt_s - jnp.asarray(
        GRAVITY, dtype=states.dtype
    )
    body_force = jnp.einsum("...ji,...j->...i", rotation, world_force)
    angular = (targets[..., 3:6] - preceding[..., 3:6]) / dt_s
    output_scale = jnp.maximum(
        norms["output_scale"], rms(jnp.concatenate((body_force, angular), axis=-1))
    )
    sf = feature_scale / norms["feature_scale"]
    sq = quadratic_scale / norms["quadratic_scale"]
    so = norms["output_scale"] / output_scale
    changed_params = dict(
        params,
        linear=sf[:, None] * params["linear"] * so[None, :],
        quadratic=sq[:, None] * params["quadratic"] * so[None, :],
        bias=params["bias"] * so,
        w1=sf[:, None] * params["w1"],
        w2=params["w2"] * so[None, :],
        memory=sf[:, None] * params["memory"],
    )
    changed_norms = dict(
        norms,
        feature_scale=feature_scale,
        quadratic_scale=quadratic_scale,
        output_scale=output_scale,
    )
    finite = jnp.all(
        jnp.stack(
            [
                jnp.all(jnp.isfinite(value))
                for value in jax.tree.leaves((changed_params, changed_norms))
            ]
        )
    )
    conditioned = jax.tree.map(
        lambda new, old: jnp.where(finite, new, old),
        (changed_params, changed_norms),
        (params, norms),
    )
    return *conditioned, finite


@partial(jax.jit, static_argnames=("delay", "dt_s"))
def _proposal(params, norms, data, scale, weights, damping, *, delay, dt_s):
    """Four matrix-free CG iterations; all parameters remain differentiable.

    Each physical group's normalized norm sets its Huber threshold. Residual
    weights include the role weight and 1/(horizon*3); each group's IRLS factor
    is broadcast over its coordinates and stays fixed throughout the proposal.
    """
    flat, unpack = ravel_pytree(params)
    raw, push = jax.linearize(
        lambda value: _residual(unpack(value), norms, data, scale, delay, dt_s), flat
    )
    pull = jax.linear_transpose(push, jnp.zeros_like(flat))
    weight = weights[:, None, None] / (raw.shape[1] * 3)
    root_weight = jnp.sqrt(weight)
    factors = 1 / jnp.sqrt(jnp.maximum(1.0, _group_squared(raw)))
    irls = jnp.repeat(factors, np.array([3, 3, 9]), axis=-1, total_repeat_length=15)
    gradient = pull(weight * irls * raw)[0]

    def cg_step(_, carry):
        delta, residual, direction, squared, finite = carry
        projected = push(direction)
        curvature = pull(weight * irls * projected)[0] + damping * direction
        denominator = jnp.vdot(direction, curvature)
        valid = jnp.all(jnp.isfinite(curvature)) & jnp.all(jnp.isfinite(projected))
        valid &= jnp.isfinite(squared) & jnp.isfinite(denominator)
        valid &= (squared == 0) | (denominator > 0)
        active = valid & (squared > 0)
        alpha = jnp.where(active, squared / jnp.where(active, denominator, 1.0), 0.0)
        delta = jnp.where(active, delta + alpha * direction, delta)
        following = jnp.where(active, residual - alpha * curvature, residual)
        next_squared = jnp.vdot(following, following)
        beta = jnp.where(active, next_squared / jnp.where(active, squared, 1.0), 0.0)
        direction = jnp.where(
            active, following + beta * direction, jnp.zeros_like(direction)
        )
        return delta, following, direction, next_squared, finite & valid

    delta, _, _, _, finite = jax.lax.fori_loop(
        0,
        4,
        cg_step,
        (
            jnp.zeros_like(flat),
            -gradient,
            -gradient,
            jnp.vdot(gradient, gradient),
            jnp.asarray(True),
        ),
    )
    linearized = root_weight * push(delta)
    radius = jnp.maximum(1.0, 0.5 * jnp.linalg.norm(root_weight * raw))
    shrink = jnp.minimum(1.0, radius / jnp.maximum(jnp.linalg.norm(linearized), 1e-30))
    delta, linearized = delta * shrink, linearized * shrink
    predicted = -jnp.vdot(gradient, delta) - 0.5 * jnp.sum(irls * linearized**2)
    proposal = unpack(flat + delta)
    trial_raw = _residual(proposal, norms, data, scale, delay, dt_s)
    current_loss = jnp.sum(weight * _huber(raw))
    trial_loss = jnp.sum(weight * _huber(trial_raw))
    finite &= jnp.all(
        jnp.stack(
            [
                jnp.all(jnp.isfinite(v))
                for v in (
                    flat + delta,
                    raw,
                    trial_raw,
                    gradient,
                    predicted,
                    current_loss,
                    trial_loss,
                )
            ]
        )
    )
    return proposal, current_loss, trial_loss, predicted, finite


def _full_cache(bootstrap, recent):
    """Fixed slots duplicate actual rows; zero weights exclude unfilled slots."""
    blocks, role_weights = [], []
    for windows in (bootstrap, recent):
        count = len(windows["past_states"])
        if not count:
            raise ValueError("a proposal requires a real window in each role")
        indices = np.arange(32) % count
        blocks.append({k: windows[k][indices] for k in _FIELDS})
        role_weights.append(np.where(np.arange(32) < count, 0.5 / count, 0.0))
    return (
        tuple(np.concatenate((blocks[0][k], blocks[1][k])) for k in _FIELDS),
        np.concatenate(role_weights),
    )


def _windows(states, inputs, origins, history, horizon):
    return dict(
        zip(
            _FIELDS,
            (
                np.stack([states[i - history : i + 1] for i in origins]),
                np.stack([inputs[i - history : i] for i in origins]),
                np.stack([inputs[i : i + horizon] for i in origins]),
                np.stack([states[i + 1 : i + horizon + 1] for i in origins]),
            ),
        )
    )


def _scale(windows):
    """Fixed scalar per physical group, expressed in the 15 residual coordinates."""
    origins = windows["past_states"][:, -1:]
    targets = windows["future_states"]
    scales = []
    for a, b, factor in ((0, 3, 1.0), (3, 6, 1.0), (6, 15, 0.5)):
        origin = origins[..., a:b]
        change = np.sqrt(
            factor * np.mean(np.sum((origin - targets[..., a:b]) ** 2, axis=-1), axis=0)
        )
        spread = np.sqrt(
            factor * np.mean(np.sum((origin - origin.mean(axis=0)) ** 2, axis=-1))
        )
        group = np.maximum(change, 0.01 * max(float(spread), 1e-4))
        scales.append(np.repeat((group / np.sqrt(factor))[:, None], b - a, axis=1))
    return np.concatenate(scales, axis=1)


class OnlineFit:
    """Mutable prefix-only training session; predictions use issued commands.

    ``prefix`` is one real contiguous recording. ``cursor`` names the next
    absolute command row. ``observe(cursor, command, next_observation)`` consumes
    that transition once, after its prediction has been scored by the caller.
    Initialization retains 0.5 s context plus 0.25 s completed transitions.
    Numerical replay storage and optimization per observation are bounded.
    This session has no calibrated error envelope or control-readiness claim.
    """

    def __init__(self, prefix):
        contract = _contract(prefix)
        if len(prefix.segments) != 1:
            raise ValueError("online initialization requires one contiguous segment")
        segment = prefix.segments[0]
        history = steps_for(segment.dt_s)["history"]
        count = max(1, int(np.rint(0.25 / segment.dt_s)))
        horizon = max(1, int(np.rint(0.05 / segment.dt_s)))
        if count < 3 or len(segment.inputs) < history + count:
            raise ValueError(
                "online prefix needs real 0.5 s history and at least three 0.25 s transitions"
            )
        states = segment.states[-history - count - 1 :].copy()
        inputs = segment.inputs[-history - count :].copy()
        one_step = _windows(states, inputs, range(history, history + count), history, 1)
        with jax.enable_x64(True):
            self._model = initialize(SequenceBatch(**one_step, dt_s=segment.dt_s))
        self._bootstrap = _windows(
            states,
            inputs,
            list(range(history, history + count - horizon + 1))[-32:],
            history,
            horizon,
        )
        self._recent = {k: v[:0].copy() for k, v in self._bootstrap.items()}
        self._scale = _scale(self._bootstrap)
        self._states = states[-history - horizon - 1 :].copy()
        self._inputs = inputs[-history - horizon :].copy()
        self._contract = contract
        self._initial_cursor = segment.start_row + len(segment.inputs)
        self._cursor = self._initial_cursor
        self._initial_count, self._horizon = count, horizon
        self._damping = 1.0
        self._counts = dict(
            conditioning_calls=0,
            optimizer_steps=0,
            gradient_calls=0,
            objective_calls=0,
            accepted_proposals=0,
            cg_iterations=0,
            curvature_calls=0,
        )

    @property
    def cursor(self):
        return self._cursor

    @property
    def model(self):
        """Independent snapshot; editing its dictionaries cannot affect the session."""
        m = self._model
        return VehicleSequenceModel(
            m.dt_s, m.history_steps, m.delay_steps, m.params, m.norms
        )

    @property
    def report(self):
        return dict(
            recipe=copy.deepcopy(_RECIPE),
            cursor=self.cursor,
            observations=self.cursor - self._initial_cursor,
            initialized_transition_count=self._initial_count,
            initializers=1,
            ridge_solves=2,
            bootstrap_windows=len(self._bootstrap["past_states"]),
            recent_windows=len(self._recent["past_states"]),
            tail_rows=len(self._states),
            history_steps=self._model.history_steps,
            training_horizon_steps=self._horizon,
            **self._counts,
            damping=self._damping,
            envelope=dict(available=False),
        )

    def predict(self, past_states, past_inputs, future_inputs):
        """Canonical15 means; real history and the fitted sample grid are required.

        This method already uses a cached JIT with dynamic parameters. An outer
        JIT closing over this mutable session captures its weights at trace time,
        so it will not follow later observations. Compile an independent ``model``
        snapshot for a fixed forecast; live integration must pass changing weights
        as explicit traced arguments.
        """
        x, up, uf = map(jnp.asarray, (past_states, past_inputs, future_inputs))
        m, history = len(self._contract["input_channels"]), self._model.history_steps
        if (
            x.ndim not in (2, 3)
            or up.ndim != x.ndim
            or uf.ndim != x.ndim
            or x.shape[:-2] != up.shape[:-2]
            or x.shape[:-2] != uf.shape[:-2]
            or x.shape[-1] != 15
            or up.shape[-1] != m
            or uf.shape[-1] != m
            or x.shape[-2] != up.shape[-2] + 1
            or up.shape[-2] < history
            or not 1 <= uf.shape[-2] <= max(1, int(np.rint(1.2 / self._model.dt_s)))
        ):
            raise ValueError(
                "online prediction requires aligned real history and a valid forecast horizon"
            )
        if not any(isinstance(v, jax.core.Tracer) for v in (x, up, uf)):
            _validate_rotations(np.asarray(x[..., -history - 1 :, :]))
            if (
                not np.isfinite(np.asarray(up)).all()
                or not np.isfinite(np.asarray(uf)).all()
            ):
                raise ValueError("commands must be finite in the inference precision")
        single = x.ndim == 2
        if single:
            x, up, uf = x[None], up[None], uf[None]
        dtype = jnp.result_type(self._model.norms["state_scale"])
        params, norms = jax.tree.map(
            lambda a: jnp.asarray(a, dtype=dtype),
            (self._model.params, self._model.norms),
        )
        result = _predict(
            params,
            norms,
            jnp.asarray(x[:, -history - 1 :], dtype=dtype),
            jnp.asarray(up[:, -history:], dtype=dtype),
            jnp.asarray(uf, dtype=dtype),
            delay=self._model.delay_steps,
            dt_s=self._model.dt_s,
        )
        return result[0] if single else result

    def observe(self, index, command, next_observation):
        """Recondition from measured data, then attempt one curvature proposal.

        Invalid inputs leave the session untouched. Numerical rejection preserves
        the complete model, including its coordinates, increases damping, and
        still consumes the valid transition.
        """
        if (
            isinstance(index, (bool, np.bool_))
            or not isinstance(index, (int, np.integer))
            or index != self.cursor
        ):
            raise ValueError(
                "observation index must equal the next absolute command row"
            )
        command, state = (
            np.asarray(command, dtype=float),
            np.asarray(next_observation, dtype=float),
        )
        if (
            command.shape != (len(self._contract["input_channels"]),)
            or not np.isfinite(command).all()
            or state.shape != (15,)
        ):
            raise ValueError("invalid observed command/state shape or values")
        _validate_rotations(state)
        states = np.concatenate((self._states[1:], state[None]))
        inputs = np.concatenate((self._inputs[1:], command[None]))
        new = _windows(
            states,
            inputs,
            [self._model.history_steps],
            self._model.history_steps,
            self._horizon,
        )
        recent = {k: np.concatenate((self._recent[k], new[k]))[-32:] for k in _FIELDS}
        counts = self._counts.copy()
        with jax.enable_x64(True):
            params, norms = jax.tree.map(
                jnp.asarray, (self._model.params, self._model.norms)
            )
            data, weights = _full_cache(self._bootstrap, recent)
            data = tuple(jnp.asarray(v) for v in data)
            weights = jnp.asarray(weights)
            params, norms, conditioning_finite = _recondition(
                params,
                norms,
                data,
                weights,
                delay=self._model.delay_steps,
                dt_s=self._model.dt_s,
            )
            proposal, current, trial, predicted, finite = _proposal(
                params,
                norms,
                data,
                jnp.asarray(self._scale),
                weights,
                jnp.asarray(self._damping),
                delay=self._model.delay_steps,
                dt_s=self._model.dt_s,
            )
            finite = (
                bool(conditioning_finite)
                and bool(finite)
                and all(
                    np.isfinite(np.asarray(v)).all()
                    for v in jax.tree.leaves((proposal, current, trial, predicted))
                )
            )
            current, trial, predicted = map(float, (current, trial, predicted))
            gain = (
                (current - trial) / predicted if finite and predicted > 0 else -np.inf
            )
            accepted = bool(finite and trial < current and gain >= 0.1)
            damping = self._damping
            if not accepted or gain < 0.25:
                damping *= 4.0
            elif gain > 0.75:
                damping *= 0.5
            damping = float(np.clip(damping, 1e-8, 1e8))
            model = self._model
            if accepted:
                model = VehicleSequenceModel(
                    model.dt_s,
                    model.history_steps,
                    model.delay_steps,
                    jax.tree.map(np.asarray, proposal),
                    jax.tree.map(np.asarray, norms),
                )
        counts["conditioning_calls"] += 1
        counts["optimizer_steps"] += 1
        counts["gradient_calls"] += 1
        counts["objective_calls"] += 2
        counts["cg_iterations"] += 4
        counts["curvature_calls"] += 4
        counts["accepted_proposals"] += int(accepted)
        self._model, self._damping = model, damping
        self._states, self._inputs, self._recent = states, inputs, recent
        self._counts, self._cursor = counts, self.cursor + 1

    def _metadata(self):
        return dict(
            format=_FORMAT,
            recipe=copy.deepcopy(_RECIPE),
            model=self._model.metadata(),
            contract=copy.deepcopy(self._contract),
            initial_cursor=self._initial_cursor,
            cursor=self.cursor,
            initial_count=self._initial_count,
            horizon=self._horizon,
            counts=self._counts.copy(),
            damping=self._damping,
        )

    def _arrays(self):
        return {
            **self._model.arrays(),
            "scale": self._scale,
            "tail_states": self._states,
            "tail_inputs": self._inputs,
            **{f"bootstrap_{k}": v for k, v in self._bootstrap.items()},
            **{f"recent_{k}": v for k, v in self._recent.items()},
        }

    def fingerprint(self):
        return array_fingerprint(self._metadata(), self._arrays())

    def save(self, path):
        save_arrays(path, self._metadata(), self._arrays())

    @classmethod
    def load(cls, path):
        """Load a checksummed session without fitting or making a prediction."""
        try:
            meta, arrays = load_arrays(path)
            if (
                set(meta)
                != {
                    "format",
                    "recipe",
                    "model",
                    "contract",
                    "initial_cursor",
                    "cursor",
                    "initial_count",
                    "horizon",
                    "counts",
                    "damping",
                }
                or meta["format"] != _FORMAT
                or meta["recipe"] != _RECIPE
            ):
                raise ValueError("unsupported online session archive")
            obj = cls.__new__(cls)
            obj._model = VehicleSequenceModel.from_arrays(
                meta["model"],
                {k: v for k, v in arrays.items() if k.startswith(("param_", "norm_"))},
            )
            for key in (
                "contract",
                "initial_cursor",
                "cursor",
                "initial_count",
                "horizon",
                "counts",
                "damping",
            ):
                setattr(obj, "_" + key, copy.deepcopy(meta[key]))
            obj._scale, obj._states, obj._inputs = (
                arrays[k].copy() for k in ("scale", "tail_states", "tail_inputs")
            )
            for role in ("bootstrap", "recent"):
                setattr(
                    obj, "_" + role, {k: arrays[f"{role}_{k}"].copy() for k in _FIELDS}
                )
            if set(arrays) != set(obj._arrays()):
                raise ValueError("unexpected online session arrays")
            obj._validate()
            return obj
        except (KeyError, TypeError, AttributeError, IndexError) as error:
            raise ValueError("invalid online session archive") from error

    def _validate(self):
        from .learner import STATE_CHANNELS

        model, contract = self._model, self._contract
        p, h, m = model.history_steps, self._horizon, len(model.norms["input_mean"])
        if (
            not isinstance(contract, dict)
            or set(contract)
            != {"configuration_id", "state_channels", "input_channels", "dt_s"}
            or not isinstance(contract["configuration_id"], str)
            or not contract["configuration_id"].strip()
            or tuple(contract["state_channels"]) != STATE_CHANNELS
            or not isinstance(contract["input_channels"], list)
            or len(contract["input_channels"]) != m
            or any(
                not isinstance(c, str) or not c.strip()
                for c in contract["input_channels"]
            )
            or len(set(contract["input_channels"])) != m
            or contract["dt_s"] != model.dt_s
            or isinstance(contract["dt_s"], bool)
            or any(
                type(v) is not int
                for v in (self._cursor, self._initial_cursor, self._initial_count, h)
            )
            or self._initial_count != max(1, int(np.rint(0.25 / model.dt_s)))
            or self._initial_count < 3
            or h != max(1, int(np.rint(0.05 / model.dt_s)))
            or p != steps_for(model.dt_s)["history"]
            or model.delay_steps != steps_for(model.dt_s)["delay"]
            or model.params["b1"].shape != (32,)
            or model.params["memory_bias"].shape != (8,)
            or not np.array_equal(model.norms["feature_scale"][-8:], np.ones(8))
            or self._initial_cursor < p + self._initial_count
            or self.cursor < self._initial_cursor
        ):
            raise ValueError("invalid online timing or signal contract")
        observations = self.cursor - self._initial_cursor
        expected = dict(
            conditioning_calls=observations,
            optimizer_steps=observations,
            gradient_calls=observations,
            objective_calls=2 * observations,
            cg_iterations=4 * observations,
            curvature_calls=4 * observations,
        )
        if (
            not isinstance(self._counts, dict)
            or set(self._counts) != {*expected, "accepted_proposals"}
            or any(type(v) is not int or v < 0 for v in self._counts.values())
            or any(self._counts[k] != v for k, v in expected.items())
            or self._counts["accepted_proposals"] > observations
            or type(self._damping) is not float
            or not np.isfinite(self._damping)
            or not 1e-8 <= self._damping <= 1e8
            or (not observations and self._damping != 1.0)
        ):
            raise ValueError("invalid online optimizer accounting or damping")
        if any(
            v.dtype != np.dtype("float64") or not np.isfinite(v).all()
            for v in self._arrays().values()
        ):
            raise ValueError("online arrays must be finite float64")
        shapes = dict(
            past_states=(p + 1, 15),
            past_inputs=(p, m),
            future_inputs=(h, m),
            future_states=(h, 15),
        )
        for windows, count in (
            (self._bootstrap, min(32, self._initial_count - h + 1)),
            (self._recent, min(32, observations)),
        ):
            if any(windows[k].shape != (count, *shape) for k, shape in shapes.items()):
                raise ValueError("online replay storage differs from recipe")
            _validate_rotations(windows["past_states"])
            _validate_rotations(windows["future_states"])
        if (
            self._states.shape != (p + h + 1, 15)
            or self._inputs.shape != (p + h, m)
            or self._scale.shape != (h, 15)
            or not np.array_equal(self._scale, _scale(self._bootstrap))
        ):
            raise ValueError("online tail or fixed loss scale differs")
        _validate_rotations(self._states)
        tail = _windows(self._states, self._inputs, [p], p, h)
        windows = {
            k: np.concatenate((self._bootstrap[k], self._recent[k], tail[k]))
            for k in _FIELDS
        }
        nb, nr = len(self._bootstrap["past_states"]), len(self._recent["past_states"])
        origins = [
            *range(self._initial_cursor - h + 1 - nb, self._initial_cursor - h + 1),
            *range(self.cursor - h + 1 - nr, self.cursor - h + 1),
            self.cursor - h,
        ]
        keys = [WindowKey("stream", "contiguous", i) for i in origins]
        validate_window_consistency(
            SequenceBatch(**windows, dt_s=model.dt_s), keys, origins
        )
