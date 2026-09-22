"""One supported shared-physics learner: fit, differentiable predict, immutable update."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict

import jax
import jax.numpy as jnp
import numpy as np

from ._dynamics import VehicleSequenceModel, fit_sequence
from ._learner_arrays import array_fingerprint, load_arrays, save_arrays
from ._training import SequenceBatch
from .recordings import SequenceCollection, SequenceWindows, WindowKey

ENVELOPE_COVERAGE = 0.9

STATE_CHANNELS = (
    "velocity_north [m/s,world_nwu]",
    "velocity_west [m/s,world_nwu]",
    "velocity_up [m/s,world_nwu]",
    "body_rate_x [rad/s,body_flu]",
    "body_rate_y [rad/s,body_flu]",
    "body_rate_z [rad/s,body_flu]",
    *(
        f"rotation_{i}{j} [unitless,body_flu_to_world_nwu]"
        for i in range(3)
        for j in range(3)
    ),
)
RECIPE = {
    "id": "shared-vehicle-accumulator-v1",
    "context_s": 0.5,
    "delay_s": 0.1,
    "horizon_s": 0.25,
    "prediction_limit_s": 1.2,
    "training_windows": 1536,
    "development_windows": 256,
    "steps": 1000,
    "width": 32,
    "memory": 8,
    "memory_dynamics": "stable_nonlinear_driven_accumulators",
    "memory_projection_initial_scale": "inverse_sqrt_full_head_feature_count",
    "memory_initial_time_range_s": [0.01, 0.5],
    "learning_rate": 0.002,
    "ridge_fraction": 0.01,
    "seed": 0,
    "gravity_world_m_s2": [0.0, 0.0, -9.80665],
    "max_substep_s": 0.025,
    "optimizer": "safeguarded_adam",
    "objective": "fixed_initial_training_channel_balance",
    "fitter_wall_time_limit_s": 7200,
}
_FORMAT = "glassbox-accumulator-dynamics-v1"
_ARRAYS = ("past_states", "past_inputs", "future_inputs", "future_states")


def _priority(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _recording_contract(recordings):
    if not isinstance(recordings, SequenceCollection):
        raise TypeError("fit and update require a SequenceCollection of recordings")
    if (
        not recordings.configuration_id
        or not recordings.state_channels
        or not recordings.input_channels
    ):
        raise ValueError(
            "recordings must declare configuration_id and ordered state/input channel identities, including units and frames"
        )
    return dict(
        configuration_id=recordings.configuration_id,
        state_channels=list(recordings.state_channels),
        input_channels=list(recordings.input_channels),
        dt_s=float(recordings.segments[0].dt_s),
    )


def _recording_content(recordings):
    result = {}
    for name in sorted({s.recording_id for s in recordings.segments}):
        segments = sorted(
            (s for s in recordings.segments if s.recording_id == name),
            key=lambda s: s.start_row,
        )
        arrays = {
            f"{i}_{key}": getattr(s, key)
            for i, s in enumerate(segments)
            for key in ("states", "inputs")
        }
        metadata = dict(dt_s=segments[0].dt_s, starts=[s.start_row for s in segments])
        result[name] = array_fingerprint(metadata, arrays)
    if len(set(result.values())) != len(result):
        raise ValueError(
            "duplicate recording content cannot establish a separate validation recording"
        )
    return result


def steps_for(dt_s):
    """Window history (the consumed context), forecast horizon, and explicit delay."""
    if not np.isscalar(dt_s) or not np.isfinite(dt_s) or dt_s <= 0:
        raise ValueError("dt_s must be finite and positive")

    def count(seconds):
        return max(1, int(np.rint(seconds / dt_s)))

    delay = count(RECIPE["delay_s"])
    return dict(
        history=max(count(RECIPE["context_s"]), delay + 1),
        horizon=count(RECIPE["horizon_s"]),
        delay=delay,
    )


def _indices(keys, origins, budget):
    by_record = {}
    for i, (key, _origin) in enumerate(zip(keys, origins, strict=True)):
        by_record.setdefault(key.recording_id, []).append(i)
    for indices in by_record.values():
        indices.sort(
            key=lambda i: _priority(
                [keys[i].recording_id, keys[i].segment_id, int(origins[i])]
            )
        )
    result = []
    # Round-robin gives each recording equal slots until it exhausts its data.
    for depth in range(max(map(len, by_record.values()), default=0)):
        for name in sorted(by_record):
            if depth < len(by_record[name]):
                result.append(by_record[name][depth])
            if len(result) == budget:
                return result
    return result


def _subset(windows, indices):
    return SequenceWindows(
        SequenceBatch(
            **{k: getattr(windows.batch, k)[indices] for k in _ARRAYS},
            dt_s=windows.batch.dt_s,
        ),
        tuple(windows.keys[i] for i in indices),
        tuple(windows.source_origins[i] for i in indices),
    )


def _extract(recordings, names, budget):
    steps = steps_for(recordings.segments[0].dt_s)
    p, h = steps["history"], steps["horizon"]
    selected = SequenceCollection(
        tuple(s for s in recordings.segments if s.recording_id in names)
    )
    keys = selected.window_keys(history_steps=p, horizon_steps=h)
    represented = {k.recording_id for k in keys}
    missing = set(names) - represented
    if missing:
        raise ValueError(
            f"recordings have no complete {p}-step history / {h}-step forecast windows: {sorted(missing)}"
        )
    origins = {(s.recording_id, s.segment_id): s.start_row for s in selected.segments}
    source_origins = [origins[k.recording_id, k.segment_id] + k.origin for k in keys]
    chosen = _indices(keys, source_origins, budget)
    if len(chosen) < 3:
        raise ValueError(
            "at least three complete windows are required in each data role"
        )
    return selected.extract([keys[i] for i in chosen], history_steps=p, horizon_steps=h)


def _merge_cache(old, fresh):
    merged = SequenceWindows(
        SequenceBatch(
            **{
                k: np.concatenate((getattr(old.batch, k), getattr(fresh.batch, k)))
                for k in _ARRAYS
            },
            dt_s=old.batch.dt_s,
        ),
        old.keys + fresh.keys,
        old.source_origins + fresh.source_origins,
    )
    return _subset(
        merged,
        _indices(merged.keys, merged.source_origins, RECIPE["training_windows"]),
    )


def _measure(model, windows):
    b = windows.batch
    prediction = np.asarray(
        model.rollout(b.past_states, b.past_inputs, b.future_inputs)
    )
    hold = np.repeat(b.past_states[:, -1:], b.future_states.shape[1], 1)
    result = {}
    ids = np.array([k.recording_id for k in windows.keys])
    for name in sorted(set(ids)):
        mask = ids == name
        error = np.sqrt(np.mean((prediction[mask] - b.future_states[mask]) ** 2, 0))
        reference = np.sqrt(np.mean((hold[mask] - b.future_states[mask]) ** 2, 0))
        result[str(name)] = dict(
            windows=int(mask.sum()),
            channel_rmse=error.tolist(),
            hold_channel_rmse=reference.tolist(),
            worse_than_hold=(error > reference * 1.05 + 1e-12).tolist(),
        )
    return result


def _calibrate(model, windows):
    """Split-conformal half-widths from the windows the fit already holds out.

    For each horizon step and channel, the finite-sample conformal quantile of
    the absolute development-window forecast error: the ``ceil((n+1) * level)``
    smallest of the ``n`` residuals, and the largest of them when that rank
    exceeds ``n``. The result is in the channel's own physical units and covers
    at least the nominal fraction of those residuals by construction.

    The development windows are held out of every gradient step, and they are
    also what selects the training checkpoint. That is a stated approximation of
    exchangeability rather than an independent calibration set, and it is one
    reason measured coverage on a further held-out recording can sit outside the
    envelope's nominal level.
    """
    b = windows.batch
    prediction = np.asarray(
        model.rollout(b.past_states, b.past_inputs, b.future_inputs)
    )
    residual = np.abs(prediction - b.future_states)
    if not np.isfinite(residual).all():
        raise ValueError(
            "a nonfinite development residual cannot calibrate an envelope"
        )
    count = len(residual)
    rank = min(int(np.ceil((count + 1) * ENVELOPE_COVERAGE)), count)
    return np.sort(residual, axis=0)[rank - 1], count, rank


def _contract(recordings):
    contract = _recording_contract(recordings)
    if tuple(contract["state_channels"]) != STATE_CHANNELS:
        raise ValueError(
            "vehicle recordings require canonical velocity/rate/rotation units and frames; use a telemetry adapter"
        )
    for segment in recordings.segments:
        _validate_rotations(segment.states)
    return contract


def _validate_rotations(states):
    x = np.asarray(states)
    if x.shape[-1] != 15 or not np.isfinite(x).all():
        raise ValueError(
            "vehicle observations must contain 15 finite canonical channels"
        )
    r = x[..., 6:].reshape(*x.shape[:-1], 3, 3)
    if not np.allclose(
        np.swapaxes(r, -1, -2) @ r, np.eye(3), atol=1e-5, rtol=0
    ) or not np.allclose(np.linalg.det(r), 1.0, atol=1e-5, rtol=0):
        raise ValueError(
            "observed body-to-world rotations must be proper orthonormal matrices"
        )


def _validate_contract(model, windows, contract, seen, report=None):
    if (
        not isinstance(contract, dict)
        or set(contract)
        != {"configuration_id", "state_channels", "input_channels", "dt_s"}
        or not isinstance(contract["configuration_id"], str)
        or not contract["configuration_id"].strip()
        or tuple(contract["state_channels"]) != STATE_CHANNELS
        or contract["dt_s"] != model.dt_s
    ):
        raise ValueError("saved vehicle signal contract differs from model")
    channels = contract["input_channels"]
    if (
        not isinstance(channels, list)
        or not channels
        or any(not isinstance(v, str) or not v.strip() for v in channels)
        or len(set(channels)) != len(channels)
        or len(channels) != len(model.norms["input_mean"])
    ):
        raise ValueError("saved command channels differ from model")
    if (
        not isinstance(seen, dict)
        or len(seen) < 2
        or any(not isinstance(k, str) or not k.strip() for k in seen)
        or any(
            not isinstance(v, str)
            or len(v) != 64
            or any(c not in "0123456789abcdef" for c in v)
            for v in seen.values()
        )
        or len(set(seen.values())) != len(seen)
    ):
        raise ValueError("invalid recording identity/content ledger")
    steps = steps_for(model.dt_s)
    if (
        model.history_steps != steps["history"]
        or model.delay_steps != steps["delay"]
        or len(model.params["b1"]) != RECIPE["width"]
        or len(model.params["memory_bias"]) != RECIPE["memory"]
    ):
        raise ValueError("model timing differs from the fixed recipe")
    roles = {}
    for role, cap in [("train", 1536), ("development", 256)]:
        w = windows[role]
        n = len(w.keys)
        if not 3 <= n <= cap:
            raise ValueError("window count outside recipe bounds")
        shapes = {
            "past_states": (n, steps["history"] + 1, 15),
            "past_inputs": (n, steps["history"], len(channels)),
            "future_states": (n, steps["horizon"], 15),
            "future_inputs": (n, steps["horizon"], len(channels)),
        }
        if w.batch.dt_s != model.dt_s or any(
            getattr(w.batch, k).shape != s for k, s in shapes.items()
        ):
            raise ValueError("retained window timing/shapes differ from recipe")
        _validate_rotations(w.batch.past_states)
        _validate_rotations(w.batch.future_states)
        starts = {}
        for key, origin in zip(w.keys, w.source_origins, strict=True):
            if (
                key.recording_id not in seen
                or not isinstance(key.segment_id, str)
                or not key.segment_id.strip()
                or type(key.origin) is not int
                or key.origin < steps["history"]
                or type(origin) is not int
                or origin < key.origin
            ):
                raise ValueError("invalid window provenance")
            identity, start = (key.recording_id, key.segment_id), origin - key.origin
            if identity in starts and starts[identity] != start:
                raise ValueError("inconsistent segment origins")
            starts[identity] = start
        roles[role] = {k.recording_id for k in w.keys}
        if report is not None:
            report_role = "training" if role == "train" else role
            if report.get(report_role) != w.coverage():
                raise ValueError("saved coverage differs from retained windows")
    if roles["train"] & roles["development"]:
        raise ValueError("training and development recordings overlap")
    if report is not None and (
        report.get("recipe") != RECIPE
        or report.get("history_steps") != steps["history"]
        or report.get("horizon_steps") != steps["horizon"]
        or report.get("prediction_limit_steps")
        != max(1, int(np.rint(1.2 / model.dt_s)))
        or report.get("delay_steps") != steps["delay"]
    ):
        raise ValueError("saved report timing or recipe differs from model")
    if (
        report is not None
        and report.get("optimization", {}).get("selected_fingerprint")
        != model.fingerprint
    ):
        raise ValueError("selected model fingerprint differs from optimization report")


def _train(
    train,
    development,
    contract,
    seen,
    *,
    previous=None,
):
    """Fit from the fixed training/development roles retained by the revision."""
    with jax.enable_x64(True):
        model, optimization = fit_sequence(
            train.batch,
            development.batch,
        )
        _validate_contract(
            model, {"train": train, "development": development}, contract, seen
        )
        envelope, count, rank = _calibrate(model, development)
        steps = steps_for(model.dt_s)
        report = dict(
            recipe=copy.deepcopy(RECIPE),
            previous_revision=previous,
            history_steps=model.history_steps,
            horizon_steps=steps["horizon"],
            prediction_limit_steps=max(1, int(np.rint(1.2 / model.dt_s))),
            delay_steps=model.delay_steps,
            training=train.coverage(),
            development=development.coverage(),
            optimization=optimization,
            precision=dict(
                fitting="float64", calibration="float64", prediction="ambient_jax"
            ),
            development_errors=_measure(model, development),
            envelope=dict(
                nominal_coverage=ENVELOPE_COVERAGE,
                method="split_conformal_absolute_error",
                calibrated_on="development",
                calibration_windows=count,
                quantile_rank=rank,
                units="physical, per horizon step and channel",
                half_width=envelope.tolist(),
            ),
            evidence_limits=[
                "Accuracy, calibration and task adequacy apply only to measured conditions.",
                "A rigid-body formulation is not proven for arbitrary articulated or deforming systems.",
                "Development selects the checkpoint and calibrates the envelope; calibration is not independent.",
                "Means beyond the fitted horizon are recursive extrapolations without calibrated envelopes.",
                "Computable derivatives do not establish physical response fidelity.",
                "Updates preserve the original development source; fresh evaluation remains necessary.",
            ],
        )
        result = LearnedDynamics(
            model, train, development, contract, seen, report, envelope
        )
        return result


class LearnedDynamics:
    """One immutable supported-physics revision with bounded recording caches."""

    def __init__(self, model, train, development, contract, seen, report, envelope):
        _validate_contract(
            model, {"train": train, "development": development}, contract, seen, report
        )
        self._model, self._train, self._development = model, train, development
        self._contract, self._seen, self._report = map(
            copy.deepcopy, (contract, seen, report)
        )
        self._envelope = None
        if envelope is not None:
            self._envelope = np.array(envelope, dtype=np.float64, copy=True)
            evidence = report["envelope"]
            count = len(development.keys)
            rank = min(int(np.ceil((count + 1) * ENVELOPE_COVERAGE)), count)
            if (
                self._envelope.shape != (self.horizon_steps, 15)
                or not np.isfinite(self._envelope).all()
                or np.any(self._envelope < 0)
                or not np.array_equal(
                    self._envelope, np.asarray(evidence["half_width"])
                )
                or evidence.get("nominal_coverage") != ENVELOPE_COVERAGE
                or evidence.get("method") != "split_conformal_absolute_error"
                or evidence.get("calibrated_on") != "development"
                or evidence.get("calibration_windows") != count
                or evidence.get("quantile_rank") != rank
            ):
                raise ValueError(
                    "calibration provenance differs from retained development windows"
                )
            self._envelope.setflags(write=False)
        elif report.get("envelope", {}).get("available") is not False:
            raise ValueError(
                "uncalibrated revision must explicitly declare absent envelope"
            )

    @property
    def contract(self):
        return copy.deepcopy(self._contract)

    @property
    def report(self):
        return copy.deepcopy(self._report)

    @property
    def recipe(self):
        return copy.deepcopy(RECIPE)

    @property
    def dt_s(self):
        return self._model.dt_s

    @property
    def history_steps(self):
        return self._model.history_steps

    @property
    def horizon_steps(self):
        """Fitted/calibrated horizon; longer recursive means are distinct."""
        return self._train.batch.future_states.shape[1]

    @property
    def prediction_limit_steps(self):
        return max(1, int(np.rint(1.2 / self._model.dt_s)))

    def predict(self, past_states, past_inputs, future_inputs):
        """Differentiable canonical15 means excluding x0, single or batched.

        Use real observed history on the fitted time grid. Means beyond
        ``horizon_steps`` are extrapolations; ``envelope`` refuses those horizons.
        Value checks occur on concrete calls, never through tracing callbacks.
        """
        x, up, uf = map(jnp.asarray, (past_states, past_inputs, future_inputs))
        if (
            x.ndim not in (2, 3)
            or up.ndim != x.ndim
            or uf.ndim != x.ndim
            or x.shape[:-2] != up.shape[:-2]
            or x.shape[:-2] != uf.shape[:-2]
            or x.shape[-1] != 15
            or up.shape[-1] != len(self._contract["input_channels"])
            or uf.shape[-1] != up.shape[-1]
        ):
            raise ValueError(
                "predict needs aligned [time,channel] or [batch,time,channel] arrays"
            )
        if x.shape[-2] != up.shape[-2] + 1 or up.shape[-2] < self.history_steps:
            raise ValueError(
                "predict needs aligned real observed history, without padding"
            )
        if not 1 <= uf.shape[-2] <= self.prediction_limit_steps:
            raise ValueError("forecast exceeds the prediction limit")
        if not any(isinstance(v, jax.core.Tracer) for v in (x, up, uf)):
            _validate_rotations(np.asarray(x[..., -self.history_steps - 1 :, :]))
            if (
                not np.isfinite(np.asarray(up)).all()
                or not np.isfinite(np.asarray(uf)).all()
            ):
                raise ValueError("commands must be finite in the inference precision")
        return self._model.rollout(
            x[..., -self.history_steps - 1 :, :], up[..., -self.history_steps :, :], uf
        )

    def envelope(self, horizon_steps=None):
        if self._envelope is None:
            raise ValueError("this revision has no calibrated envelope")
        steps = self.horizon_steps if horizon_steps is None else horizon_steps
        if (
            not isinstance(steps, (int, np.integer))
            or isinstance(steps, bool)
            or not 1 <= steps <= self.horizon_steps
        ):
            raise ValueError("no calibrated envelope beyond the fitted horizon")
        return np.array(self._envelope[:steps], copy=True)

    def update(self, recordings):
        """Refit using fresh whole recordings; return a new immutable revision."""
        if _contract(recordings) != self._contract:
            raise ValueError("update configuration, signals or timing differ")
        seen = _recording_content(recordings)
        if set(seen) & set(self._seen) or set(seen.values()) & set(self._seen.values()):
            raise ValueError("update reuses a known recording identity or content")
        fresh = _extract(recordings, set(seen), RECIPE["training_windows"])
        return _train(
            _merge_cache(self._train, fresh),
            self._development,
            self._contract,
            {**self._seen, **seen},
            previous=self.fingerprint(),
        )

    def _metadata(self):
        return dict(
            format=_FORMAT,
            recipe=copy.deepcopy(RECIPE),
            model=self._model.metadata(),
            contract=self.contract,
            seen=copy.deepcopy(self._seen),
            report=self.report,
            windows={
                role: dict(
                    keys=[asdict(k) for k in w.keys],
                    source_origins=list(w.source_origins),
                )
                for role, w in [
                    ("train", self._train),
                    ("development", self._development),
                ]
            },
        )

    def _arrays(self):
        arrays = self._model.arrays()
        if self._envelope is not None:
            arrays["envelope_half_width"] = self._envelope
        for role, w in [("train", self._train), ("development", self._development)]:
            arrays.update({f"{role}_{k}": getattr(w.batch, k) for k in _ARRAYS})
        return arrays

    def fingerprint(self):
        return array_fingerprint(self._metadata(), self._arrays())

    def save(self, path):
        save_arrays(path, self._metadata(), self._arrays())

    @classmethod
    def load(cls, path):
        try:
            meta, arrays = load_arrays(path)
            if (
                set(meta)
                != {
                    "format",
                    "recipe",
                    "model",
                    "contract",
                    "seen",
                    "report",
                    "windows",
                }
                or meta.get("format") != _FORMAT
                or meta.get("recipe") != RECIPE
                or set(meta["windows"]) != {"train", "development"}
            ):
                raise ValueError("unsupported shared vehicle recipe")
            model = VehicleSequenceModel.from_arrays(
                meta["model"],
                {k: v for k, v in arrays.items() if k.startswith(("param_", "norm_"))},
            )
            windows, expected = {}, set(model.arrays())
            if meta["report"]["envelope"].get("available") is not False:
                expected.add("envelope_half_width")
            for role in ("train", "development"):
                w = meta["windows"][role]
                if set(w) != {"keys", "source_origins"}:
                    raise ValueError("invalid retained-window metadata")
                keys = _ARRAYS
                expected.update(f"{role}_{k}" for k in keys)
                windows[role] = SequenceWindows(
                    SequenceBatch(
                        **{k: arrays[f"{role}_{k}"] for k in _ARRAYS}, dt_s=model.dt_s
                    ),
                    tuple(WindowKey(**k) for k in w["keys"]),
                    tuple(w["source_origins"]),
                )
            if set(arrays) != expected or any(
                v.dtype != np.dtype("float64") for v in arrays.values()
            ):
                raise ValueError("saved array roster or dtype differs from recipe")
            return cls(
                model,
                windows["train"],
                windows["development"],
                meta["contract"],
                meta["seen"],
                meta["report"],
                arrays.get("envelope_half_width"),
            )
        except (KeyError, TypeError, AttributeError, IndexError) as error:
            raise ValueError(
                "invalid saved vehicle revision metadata or arrays"
            ) from error


def fit(recordings):
    """Fit the one supported-physics recipe; no vehicle or tuning arguments."""
    contract = _contract(recordings)
    seen = _recording_content(recordings)
    names = sorted(seen, key=_priority)
    if len(names) < 2:
        raise ValueError("fit needs at least two distinct recordings")
    count = min(len(names) - 1, max(1, int(np.ceil(len(names) / 4))))
    development = _extract(recordings, names[:count], RECIPE["development_windows"])
    train = _extract(recordings, names[count:], RECIPE["training_windows"])
    return _train(train, development, contract, seen)
