"""A bounded, causal fitting session for the same shared-physics dynamics model."""

from __future__ import annotations

import copy
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from ._dynamics import SCALES, VehicleSequenceModel, _rollout, initialize
from ._learner_arrays import array_fingerprint, load_arrays, save_arrays
from ._training import (
    SequenceBatch,
    SequenceFitError,
    trial_parameters,
    validate_window_consistency,
)
from .learner import _contract, _validate_rotations, steps_for
from .recordings import WindowKey

_FORMAT = "glassbox-online-fit-v1"
_FIELDS = ("past_states", "past_inputs", "future_inputs", "future_states")
_RECIPE = dict(
    bootstrap_s=0.25,
    horizon_s=0.05,
    bootstrap_windows=32,
    recent_windows=32,
    batch_size=4,
    proposals=4,
    learning_rate=0.002,
    huber_delta=1.0,
)


@partial(jax.jit, static_argnames=("delay", "dt_s"))
def _predict(params, norms, past, inputs, future, *, delay, dt_s):
    return _rollout(params, norms, past, inputs, future, delay, dt_s)


def _loss(params, norms, data, scale, delay, dt_s):
    prediction = _rollout(params, norms, *data[:3], delay, dt_s)
    residual = jnp.abs((prediction - data[3]) / scale)
    return jnp.mean(jnp.where(residual <= 1, 0.5 * residual**2, residual - 0.5))


_objective = jax.jit(_loss, static_argnames=("delay", "dt_s"))


@partial(jax.jit, static_argnames=("delay", "dt_s"))
def _proposal(params, first, second, index, norms, data, scale, *, delay, dt_s):
    value, grad = jax.value_and_grad(_loss)(params, norms, data, scale, delay, dt_s)
    norm = jnp.sqrt(sum(jnp.sum(g * g) for g in jax.tree.leaves(grad)))
    grad = jax.tree.map(lambda g: g * jnp.minimum(1.0, 5.0 / (norm + 1e-12)), grad)
    first = jax.tree.map(lambda a, g: 0.9 * a + 0.1 * g, first, grad)
    second = jax.tree.map(lambda a, g: 0.999 * a + 0.001 * g * g, second, grad)
    proposal = jax.tree.map(
        lambda p, a, b: (
            p
            - 0.002 * (a / (1 - 0.9**index)) / (jnp.sqrt(b / (1 - 0.999**index)) + 1e-8)
        ),
        params,
        first,
        second,
    )
    finite = jnp.all(
        jnp.stack(
            [
                jnp.all(jnp.isfinite(v))
                for v in jax.tree.leaves((proposal, first, second, grad, value, norm))
            ]
        )
    )
    return proposal, first, second, value, norm, finite


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


def _scale(windows, norms):
    return np.maximum(
        np.sqrt(
            np.mean((windows["past_states"][:, -1:] - windows["future_states"]) ** 2, 0)
        ),
        0.01 * norms["state_scale"],
    )


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
        self._scale = _scale(self._bootstrap, self._model.norms)
        self._states = states[-history - horizon - 1 :].copy()
        self._inputs = inputs[-history - horizon :].copy()
        self._contract = contract
        self._initial_cursor = segment.start_row + len(segment.inputs)
        self._cursor = self._initial_cursor
        self._initial_count, self._horizon = count, horizon
        self._first = {k: np.zeros_like(v) for k, v in self._model.params.items()}
        self._second = {k: np.zeros_like(v) for k, v in self._model.params.items()}
        self._counts = dict(
            optimizer_steps=0, gradient_calls=0, objective_calls=0, accepted_proposals=0
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
        """Assimilate one completed transition atomically, with four proposals."""
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
            params, first, second, norms = jax.tree.map(
                jnp.asarray,
                (
                    self._model.params,
                    self._first,
                    self._second,
                    self._model.norms,
                ),
            )
            scale = jnp.asarray(self._scale)
            for _ in range(4):
                index = counts["optimizer_steps"]
                b = (2 * index + np.arange(2)) % len(self._bootstrap["past_states"])
                r = (2 * index + np.arange(2)) % len(recent["past_states"])
                data = tuple(
                    jnp.asarray(np.concatenate((self._bootstrap[k][b], recent[k][r])))
                    for k in _FIELDS
                )
                proposal, first, second, value, norm, finite = _proposal(
                    params,
                    first,
                    second,
                    jnp.asarray(index + 1),
                    norms,
                    data,
                    scale,
                    delay=self._model.delay_steps,
                    dt_s=self._model.dt_s,
                )
                if not bool(finite) or not all(
                    np.isfinite(np.asarray(v)).all()
                    for v in jax.tree.leaves((proposal, first, second, value, norm))
                ):
                    raise SequenceFitError(
                        f"nonfinite online Adam proposal at attempt {index + 1}"
                    )
                counts["optimizer_steps"] += 1
                counts["gradient_calls"] += 1
                for alpha in SCALES:
                    trial = trial_parameters(params, proposal, alpha)
                    loss = float(
                        _objective(
                            trial,
                            norms,
                            data,
                            scale,
                            self._model.delay_steps,
                            self._model.dt_s,
                        )
                    )
                    counts["objective_calls"] += 1
                    if np.isfinite(loss) and loss < float(value):
                        params = trial
                        counts["accepted_proposals"] += 1
                        break
            model = VehicleSequenceModel(
                self._model.dt_s,
                self._model.history_steps,
                self._model.delay_steps,
                jax.tree.map(np.asarray, params),
                self._model.norms,
            )
            first, second = jax.tree.map(np.asarray, (first, second))
        self._model, self._first, self._second = model, first, second
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
        )

    def _arrays(self):
        return {
            **self._model.arrays(),
            "scale": self._scale,
            "tail_states": self._states,
            "tail_inputs": self._inputs,
            **{f"bootstrap_{k}": v for k, v in self._bootstrap.items()},
            **{f"recent_{k}": v for k, v in self._recent.items()},
            **{f"first_{k}": v for k, v in self._first.items()},
            **{f"second_{k}": v for k, v in self._second.items()},
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
            ):
                setattr(obj, "_" + key, copy.deepcopy(meta[key]))
            obj._scale, obj._states, obj._inputs = (
                arrays[k].copy() for k in ("scale", "tail_states", "tail_inputs")
            )
            for role in ("bootstrap", "recent"):
                setattr(
                    obj, "_" + role, {k: arrays[f"{role}_{k}"].copy() for k in _FIELDS}
                )
            for role in ("first", "second"):
                setattr(
                    obj,
                    "_" + role,
                    {k: arrays[f"{role}_{k}"].copy() for k in obj._model.params},
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
            or self._initial_cursor < p + self._initial_count
            or self.cursor < self._initial_cursor
        ):
            raise ValueError("invalid online timing or signal contract")
        observations = self.cursor - self._initial_cursor
        expected = dict(
            optimizer_steps=4 * observations, gradient_calls=4 * observations
        )
        if (
            not isinstance(self._counts, dict)
            or set(self._counts) != {*expected, "objective_calls", "accepted_proposals"}
            or any(type(v) is not int or v < 0 for v in self._counts.values())
            or any(self._counts[k] != v for k, v in expected.items())
            or not expected["optimizer_steps"]
            <= self._counts["objective_calls"]
            <= 8 * expected["optimizer_steps"]
            or self._counts["accepted_proposals"] > expected["optimizer_steps"]
        ):
            raise ValueError("invalid online optimizer accounting")
        if any(
            v.dtype != np.dtype("float64") or not np.isfinite(v).all()
            for v in self._arrays().values()
        ):
            raise ValueError("online arrays must be finite float64")
        for role in (self._first, self._second):
            if any(role[k].shape != v.shape for k, v in model.params.items()):
                raise ValueError("optimizer moment shape differs")
        if any(np.any(v < 0) for v in self._second.values()):
            raise ValueError("negative optimizer second moment")
        if not observations and any(
            np.any(v) for v in (*self._first.values(), *self._second.values())
        ):
            raise ValueError("unobserved session has nonzero optimizer moments")
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
            or not np.array_equal(self._scale, _scale(self._bootstrap, model.norms))
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
