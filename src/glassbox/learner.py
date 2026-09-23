"""One generic episode-fitted dynamics learner and immutable public revisions."""

from __future__ import annotations

import copy
from collections import defaultdict

import jax
import jax.numpy as jnp
import numpy as np

from ._causal_actuator import CausalActuatorModel
from ._causal_fit import fit_segments
from ._learner_arrays import array_fingerprint, load_arrays, save_arrays
from .recordings import SequenceCollection, SequenceSegment

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
    "id": "causal-actuator-v1",
    "history": "all issued commands since episode start",
    "horizon_s": 0.25,
    "prediction_limit_s": 1.2,
    "fit": "shared force and rigid-body torque from complete training segments",
    "actuator": "episode-fitted nonlinear target and relaxation",
    "initialization": "data-independent",
}
FORMAT = "glassbox-causal-vehicle-v1"


def _validate_rotations(states):
    x = np.asarray(states)
    if x.shape[-1] != 15 or not np.isfinite(x).all():
        raise ValueError("vehicle observations need 15 finite canonical channels")
    rotation = x[..., 6:].reshape(*x.shape[:-1], 3, 3)
    if not np.allclose(
        np.swapaxes(rotation, -1, -2) @ rotation,
        np.eye(3),
        atol=1e-5,
        rtol=0,
    ) or not np.allclose(np.linalg.det(rotation), 1.0, atol=1e-5, rtol=0):
        raise ValueError("observed rotations must be proper orthonormal matrices")


def _contract(recordings):
    if not isinstance(recordings, SequenceCollection):
        raise TypeError("fit and update require a SequenceCollection")
    if tuple(recordings.state_channels) != STATE_CHANNELS:
        raise ValueError("vehicle recordings require canonical units and frames")
    if not recordings.configuration_id or not recordings.input_channels:
        raise ValueError("recordings need configuration and command identities")
    for segment in recordings.segments:
        _validate_rotations(segment.states)
    return {
        "configuration_id": recordings.configuration_id,
        "state_channels": list(recordings.state_channels),
        "input_channels": list(recordings.input_channels),
        "dt_s": recordings.segments[0].dt_s,
    }


def _recording_content(segments):
    if isinstance(segments, SequenceCollection):
        segments = segments.segments
    grouped = defaultdict(list)
    for segment in segments:
        grouped[segment.recording_id].append(segment)
    result = {}
    for name, items in grouped.items():
        ordered = sorted(items, key=lambda segment: segment.start_row)
        metadata = {
            "dt_s": ordered[0].dt_s,
            "starts": [segment.start_row for segment in ordered],
        }
        arrays = {
            f"{i}_{key}": getattr(segment, key)
            for i, segment in enumerate(ordered)
            for key in ("states", "inputs")
        }
        result[name] = array_fingerprint(metadata, arrays)
    if len(set(result.values())) != len(result):
        raise ValueError("duplicate recording content cannot establish a new source")
    return result


def _fit(segments, contract, seen, previous=None):
    model, optimization = fit_segments(
        tuple((segment.states, segment.inputs, 0) for segment in segments),
        contract["dt_s"],
    )
    optimization["selected_fingerprint"] = model.fingerprint
    report = {
        "recipe": copy.deepcopy(RECIPE),
        "previous_revision": previous,
        "training_recordings": sorted({s.recording_id for s in segments}),
        "optimization": optimization,
        "envelope": {"available": False, "reason": "held-out coverage not qualified"},
        "evidence_limits": [
            "Fit data cannot establish independent forecast accuracy or coverage.",
            "A complete command history since episode start is required for prediction.",
            "Causal prefix fitting alone does not qualify live publication timing.",
        ],
    }
    return LearnedDynamics(model, segments, contract, seen, report)


class LearnedDynamics:
    """One immutable fitted equation with retained recording provenance."""

    def __init__(self, model, segments, contract, seen, report):
        if not isinstance(model, CausalActuatorModel):
            raise TypeError("invalid causal vehicle model")
        if (
            report.get("optimization", {}).get("selected_fingerprint")
            != model.fingerprint
        ):
            raise ValueError("selected model fingerprint differs from report")
        self._segments = tuple(segments)
        self._model = model
        self._contract = copy.deepcopy(contract)
        self._seen = copy.deepcopy(seen)
        self._report = copy.deepcopy(report)
        recordings = SequenceCollection(
            self._segments,
            contract.get("configuration_id"),
            tuple(contract.get("state_channels", ())),
            tuple(contract.get("input_channels", ())),
        )
        if (
            _contract(recordings) != contract
            or _recording_content(recordings.segments) != seen
            or model.dt_s != contract["dt_s"]
            or model.input_count != len(contract["input_channels"])
            or not self._segments
            or report.get("recipe") != RECIPE
            or report.get("training_recordings")
            != sorted({s.recording_id for s in self._segments})
            or report.get("envelope", {}).get("available") is not False
        ):
            raise ValueError("saved causal revision provenance differs")

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
    def horizon_steps(self):
        return max(1, round(RECIPE["horizon_s"] / self.dt_s))

    @property
    def prediction_limit_steps(self):
        return max(1, round(RECIPE["prediction_limit_s"] / self.dt_s))

    def predict(self, past_states, past_inputs, future_inputs):
        """Predict from the latest state and every issued command since reset."""
        x, past, future = map(jnp.asarray, (past_states, past_inputs, future_inputs))
        if x.ndim not in (2, 3) or past.ndim != x.ndim or future.ndim != x.ndim:
            raise ValueError("predict needs aligned [time,channel] arrays")
        if (
            x.shape[:-2] != past.shape[:-2]
            or x.shape[:-2] != future.shape[:-2]
            or x.shape[-1] != 15
            or past.shape[-1] != self._model.input_count
            or future.shape[-1] != self._model.input_count
            or x.shape[-2] != past.shape[-2] + 1
            or past.shape[-2] < 1
        ):
            raise ValueError("predict requires complete aligned episode history")
        if not 1 <= future.shape[-2] <= self.prediction_limit_steps:
            raise ValueError("forecast exceeds prediction limit")
        if not any(isinstance(v, jax.core.Tracer) for v in (x, past, future)):
            _validate_rotations(np.asarray(x[..., -1:, :]))
            if (
                not np.isfinite(np.asarray(past)).all()
                or not np.isfinite(np.asarray(future)).all()
            ):
                raise ValueError("issued commands must be finite")
        if x.ndim == 2:
            return self._model.forecast(x[-1], past, future)
        return jax.vmap(self._model.forecast)(x[:, -1], past, future)

    def envelope(self, horizon_steps=None):
        raise ValueError("this revision has no calibrated envelope")

    def update(self, recordings):
        contract = _contract(recordings)
        if contract != self._contract:
            raise ValueError("update configuration, signals or timing differ")
        fresh = _recording_content(recordings.segments)
        if set(fresh) & set(self._seen) or set(fresh.values()) & set(
            self._seen.values()
        ):
            raise ValueError("update reuses a known recording identity or content")
        return _fit(
            self._segments + tuple(recordings.segments),
            self._contract,
            {**self._seen, **fresh},
            previous=self.fingerprint(),
        )

    def _metadata(self):
        return {
            "format": FORMAT,
            "recipe": copy.deepcopy(RECIPE),
            "model": self._model.metadata(),
            "contract": self.contract,
            "seen": copy.deepcopy(self._seen),
            "report": self.report,
            "segments": [
                {
                    "recording_id": s.recording_id,
                    "segment_id": s.segment_id,
                    "dt_s": s.dt_s,
                    "start_row": s.start_row,
                }
                for s in self._segments
            ],
        }

    def _arrays(self):
        arrays = {f"model_{key}": value for key, value in self._model.arrays().items()}
        for i, segment in enumerate(self._segments):
            arrays[f"segment_{i}_states"] = segment.states.copy()
            arrays[f"segment_{i}_inputs"] = segment.inputs.copy()
        return arrays

    def fingerprint(self):
        return array_fingerprint(self._metadata(), self._arrays())

    def save(self, path):
        save_arrays(path, self._metadata(), self._arrays())

    @classmethod
    def load(cls, path):
        metadata, arrays = load_arrays(path)
        if (
            not isinstance(metadata, dict)
            or set(metadata)
            != {"format", "recipe", "model", "contract", "seen", "report", "segments"}
            or metadata["format"] != FORMAT
            or metadata["recipe"] != RECIPE
            or not isinstance(metadata["segments"], list)
        ):
            raise ValueError("invalid causal vehicle revision")
        model_keys = {"q", "coeff", "inertia", "force", "torque"}
        model = CausalActuatorModel.from_arrays(
            metadata["model"], {key: arrays[f"model_{key}"] for key in model_keys}
        )
        expected = {f"model_{key}" for key in model_keys}
        segments = []
        for i, item in enumerate(metadata["segments"]):
            keys = (f"segment_{i}_states", f"segment_{i}_inputs")
            expected.update(keys)
            segments.append(
                SequenceSegment(
                    item["recording_id"],
                    item["segment_id"],
                    arrays[keys[0]],
                    arrays[keys[1]],
                    item["dt_s"],
                    item["start_row"],
                )
            )
        if set(arrays) != expected:
            raise ValueError("saved causal array roster differs")
        return cls(
            model,
            segments,
            metadata["contract"],
            metadata["seen"],
            metadata["report"],
        )


def fit(recordings):
    """Fit one generic equation from every supplied recording."""
    contract = _contract(recordings)
    seen = _recording_content(recordings.segments)
    return _fit(recordings.segments, contract, seen)
