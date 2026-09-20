"""Research vehicle learner: shared mechanics, learned configuration dynamics.

Recordings declare motion channels, units, frames, timestamps and ordered commands.
The caller supplies no vehicle parameters or learner tuning. Separate configurations
get separate immutable fitted revisions. This module is not yet the public recipe.
"""

from __future__ import annotations

import copy
from dataclasses import asdict

import jax
import jax.numpy as jnp
import numpy as np

from glassbox import learner
from glassbox._learner_arrays import array_fingerprint, load_arrays, save_arrays
from glassbox._sequence_model import SequenceBatch
from glassbox.recordings import SequenceWindows, WindowKey

from .shared_vehicle_core import VehicleSequenceModel, fit_sequence

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
    "id": "shared-vehicle-physics-v1-research",
    "context_s": 0.5,
    "delay_s": 0.1,
    "horizon_s": 0.25,
    "prediction_limit_s": 1.2,
    "training_windows": 1536,
    "development_windows": 256,
    "steps": 1000,
    "width": 32,
    "memory": 8,
    "learning_rate": 0.002,
    "ridge_fraction": 0.01,
    "seed": 0,
    "gravity_world_m_s2": [0.0, 0.0, -9.80665],
    "max_substep_s": 0.025,
    "optimizer": "safeguarded_adam",
    "objective": "fixed_initial_training_channel_balance",
    "fitter_wall_time_limit_s": 7200,
}
_FORMAT = "glassbox-shared-vehicle-revision-v1"
_ARRAYS = ("past_states", "past_inputs", "future_inputs", "future_states")


def _contract(recordings):
    contract = learner._contract(recordings)
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
    steps = learner.steps_for(model.dt_s)
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
    error_scale=None,
    channel_weights=None,
    observer=None,
):
    """Internal exact-cache harness entry, not a consumer tuning interface."""
    with jax.enable_x64(True):
        model, optimization = fit_sequence(
            train.batch,
            development.batch,
            error_scale=error_scale,
            channel_weights=channel_weights,
            observer=observer,
        )
        _validate_contract(
            model, {"train": train, "development": development}, contract, seen
        )
        envelope, count, rank = learner._calibrate(model, development)
        steps = learner.steps_for(model.dt_s)
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
            development_errors=learner._measure(model, development),
            envelope=dict(
                nominal_coverage=learner.ENVELOPE_COVERAGE,
                method="split_conformal_absolute_error",
                calibrated_on="development",
                calibration_windows=count,
                quantile_rank=rank,
                units="physical, per horizon step and channel",
                half_width=envelope.tolist(),
            ),
            evidence_limits=[
                "Research recipe; physical accuracy and Dart adequacy need separate frozen evaluations.",
                "A rigid-body formulation is not proven for arbitrary articulated or deforming systems.",
                "Development selects the checkpoint and calibrates the envelope; calibration is not independent.",
                "Means beyond the fitted horizon are recursive extrapolations without calibrated envelopes.",
                "Computable derivatives do not establish physical response fidelity.",
                "Updates preserve the original development source; fresh evaluation remains necessary.",
            ],
        )
        result = SharedVehicleDynamics(
            model, train, development, contract, seen, report, envelope
        )
        if observer is not None:
            observer(
                dict(
                    phase="calibrated",
                    envelope=np.array(envelope),
                    report=result.report,
                )
            )
        return result


class SharedVehicleDynamics:
    """Immutable research revision with bounded caches and explicit error scope."""

    def __init__(self, model, train, development, contract, seen, report, envelope):
        _validate_contract(
            model, {"train": train, "development": development}, contract, seen, report
        )
        self._model, self._train, self._development = model, train, development
        self._contract, self._seen, self._report = map(
            copy.deepcopy, (contract, seen, report)
        )
        self._envelope = np.array(envelope, dtype=np.float64, copy=True)
        if (
            self._envelope.shape != (self.horizon_steps, 15)
            or not np.isfinite(self._envelope).all()
            or np.any(self._envelope < 0)
            or not np.array_equal(
                self._envelope, np.asarray(report["envelope"]["half_width"])
            )
        ):
            raise ValueError(
                "finite calibrated envelope must match report and fitted horizon"
            )
        evidence = report["envelope"]
        count = len(development.keys)
        rank = min(int(np.ceil((count + 1) * learner.ENVELOPE_COVERAGE)), count)
        if (
            evidence.get("nominal_coverage") != learner.ENVELOPE_COVERAGE
            or evidence.get("method") != "split_conformal_absolute_error"
            or evidence.get("calibrated_on") != "development"
            or evidence.get("calibration_windows") != count
            or evidence.get("quantile_rank") != rank
        ):
            raise ValueError(
                "calibration provenance differs from retained development windows"
            )
        self._envelope.setflags(write=False)

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
    def history_steps(self):
        return self._model.history_steps

    @property
    def horizon_steps(self):
        """Fitted/calibrated horizon; longer research means are distinct."""
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
            raise ValueError("forecast exceeds the research prediction limit")
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
        seen = learner._recording_content(recordings)
        if set(seen) & set(self._seen) or set(seen.values()) & set(self._seen.values()):
            raise ValueError("update reuses a known recording identity or content")
        fresh = learner._extract(recordings, set(seen), RECIPE["training_windows"])
        return _train(
            learner._merge_cache(self._train, fresh),
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
                    excitation_declared=w.excitation_declared,
                )
                for role, w in [
                    ("train", self._train),
                    ("development", self._development),
                ]
            },
        )

    def _arrays(self):
        arrays = {"envelope_half_width": self._envelope, **self._model.arrays()}
        for role, w in [("train", self._train), ("development", self._development)]:
            arrays.update({f"{role}_{k}": getattr(w.batch, k) for k in _ARRAYS})
            arrays.update({f"{role}_{k}": v for k, v in learner._excitation(w).items()})
        return arrays

    def fingerprint(self):
        return array_fingerprint(self._metadata(), self._arrays())

    def save(self, path):
        save_arrays(path, self._metadata(), self._arrays())

    @classmethod
    def load(cls, path):
        try:
            meta, arrays = load_arrays(path)
            if meta.get("format") != _FORMAT or meta.get("recipe") != RECIPE:
                raise ValueError("unsupported shared vehicle recipe")
            model = VehicleSequenceModel.from_arrays(
                meta["model"],
                {k: v for k, v in arrays.items() if k.startswith(("param_", "norm_"))},
            )
            windows, expected = {}, {"envelope_half_width", *model.arrays()}
            for role in ("train", "development"):
                w = meta["windows"][role]
                if type(w["excitation_declared"]) is not bool:
                    raise ValueError("missing explicit window excitation declaration")
                keys = _ARRAYS + (
                    ("past_excitation", "future_excitation")
                    if w["excitation_declared"]
                    else ()
                )
                expected.update(f"{role}_{k}" for k in keys)
                windows[role] = SequenceWindows(
                    SequenceBatch(
                        **{k: arrays[f"{role}_{k}"] for k in _ARRAYS}, dt_s=model.dt_s
                    ),
                    tuple(WindowKey(**k) for k in w["keys"]),
                    tuple(w["source_origins"]),
                    **{k: arrays[f"{role}_{k}"] for k in keys if k not in _ARRAYS},
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
                arrays["envelope_half_width"],
            )
        except (KeyError, TypeError, AttributeError, IndexError) as error:
            raise ValueError(
                "invalid saved vehicle revision metadata or arrays"
            ) from error


def fit(recordings):
    """Fit the one research vehicle recipe; no vehicle or tuning arguments."""
    contract = _contract(recordings)
    seen = learner._recording_content(recordings)
    names = sorted(seen, key=learner._priority)
    if len(names) < 2:
        raise ValueError("fit needs at least two distinct recordings")
    count = min(len(names) - 1, max(1, int(np.ceil(len(names) / 4))))
    development = learner._extract(
        recordings, names[:count], RECIPE["development_windows"]
    )
    train = learner._extract(recordings, names[count:], RECIPE["training_windows"])
    return _train(train, development, contract, seen)
