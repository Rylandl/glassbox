"""Causal episode collection with intermittent immutable model publication."""

from __future__ import annotations

import copy

import jax
import jax.numpy as jnp
import numpy as np

from ._causal_actuator import CausalActuatorModel
from ._causal_fit import BackgroundFit
from ._learner_arrays import array_fingerprint, load_arrays, save_arrays
from .learner import RECIPE, _contract, _validate_rotations
from .recordings import SequenceCollection, SequenceSegment

FORMAT = "glassbox-causal-episode-v1"


class OnlineFit:
    """Gather a segment while a single worker fits and publishes new revisions."""

    def __init__(self, prefix):
        contract = _contract(prefix)
        if len(prefix.segments) != 1:
            raise ValueError("episode initialization requires one contiguous segment")
        segment = prefix.segments[0]
        self._contract = contract
        self._recording_id = segment.recording_id
        self._segment_id = segment.segment_id
        self._start_row = segment.start_row
        self._fit = BackgroundFit(
            segment.states, segment.inputs, segment.dt_s, segment.start_row
        )

    @property
    def cursor(self):
        return self._fit.cursor

    @property
    def published_cursor(self):
        return self._fit.published_cursor

    @property
    def model(self):
        return self._fit.model

    @property
    def report(self):
        return self._fit.report

    def predict(self, past_states, past_inputs, future_inputs):
        """Predict using the latest published revision and full episode history."""
        x, past, future = map(jnp.asarray, (past_states, past_inputs, future_inputs))
        if (
            x.ndim != 2
            or past.ndim != 2
            or future.ndim != 2
            or x.shape != (len(past) + 1, 15)
            or past.shape[1] != self.model.input_count
            or future.shape[1] != self.model.input_count
            or len(past) != self.cursor - self._start_row
            or len(future) < 1
            or len(future)
            > max(1, round(RECIPE["prediction_limit_s"] / self.model.dt_s))
        ):
            raise ValueError("prediction requires complete current episode history")
        if not any(isinstance(v, jax.core.Tracer) for v in (x, past, future)):
            _validate_rotations(np.asarray(x[-1:]))
            if (
                not np.isfinite(np.asarray(past)).all()
                or not np.isfinite(np.asarray(future)).all()
            ):
                raise ValueError("issued commands must be finite")
        return self.model.forecast(x[-1], past, future)

    def observe(self, row, command, next_state):
        _validate_rotations(np.asarray(next_state)[None])
        self._fit.observe(row, command, next_state)

    def wait_for_publication(self, timeout=None):
        return self._fit.wait_for_publication(timeout)

    def close(self):
        self._fit.close()

    def _archive(self):
        states, commands, cursor, published, model, report = self._fit.snapshot()
        metadata = {
            "format": FORMAT,
            "recipe": copy.deepcopy(RECIPE),
            "contract": copy.deepcopy(self._contract),
            "recording_id": self._recording_id,
            "segment_id": self._segment_id,
            "start_row": self._start_row,
            "cursor": cursor,
            "published_cursor": published,
            "model": model.metadata(),
            "published_fingerprint": model.fingerprint,
            "report": report,
        }
        arrays = {
            "states": states,
            "commands": commands,
            **{f"model_{key}": value for key, value in model.arrays().items()},
        }
        return metadata, arrays

    def fingerprint(self):
        metadata, arrays = self._archive()
        return array_fingerprint(metadata, arrays)

    def save(self, path):
        metadata, arrays = self._archive()
        save_arrays(path, metadata, arrays)

    @classmethod
    def load(cls, path):
        metadata, arrays = load_arrays(path)
        if (
            not isinstance(metadata, dict)
            or set(metadata)
            != {
                "format",
                "recipe",
                "contract",
                "recording_id",
                "segment_id",
                "start_row",
                "cursor",
                "published_cursor",
                "model",
                "published_fingerprint",
                "report",
            }
            or metadata["format"] != FORMAT
            or metadata["recipe"] != RECIPE
            or set(arrays)
            != {
                "states",
                "commands",
                "model_q",
                "model_coeff",
                "model_inertia",
                "model_force",
                "model_torque",
            }
            or any(value.dtype != np.dtype("float64") for value in arrays.values())
        ):
            raise ValueError("invalid saved causal episode")
        model = CausalActuatorModel.from_arrays(
            metadata["model"],
            {
                key: arrays[f"model_{key}"]
                for key in ("q", "coeff", "inertia", "force", "torque")
            },
        )
        if model.fingerprint != metadata["published_fingerprint"]:
            raise ValueError("published model differs from episode report")
        segment = SequenceSegment(
            metadata["recording_id"],
            metadata["segment_id"],
            arrays["states"],
            arrays["commands"],
            metadata["contract"]["dt_s"],
            metadata["start_row"],
        )
        collection = SequenceCollection(
            (segment,),
            metadata["contract"]["configuration_id"],
            tuple(metadata["contract"]["state_channels"]),
            tuple(metadata["contract"]["input_channels"]),
        )
        if _contract(collection) != metadata["contract"] or metadata[
            "cursor"
        ] != segment.start_row + len(segment.inputs):
            raise ValueError("saved episode signals or cursor differ")
        obj = cls.__new__(cls)
        obj._contract = copy.deepcopy(metadata["contract"])
        obj._recording_id = segment.recording_id
        obj._segment_id = segment.segment_id
        obj._start_row = segment.start_row
        obj._fit = BackgroundFit.from_snapshot(
            segment.states,
            segment.inputs,
            segment.dt_s,
            segment.start_row,
            metadata["published_cursor"],
            model,
            metadata["report"],
        )
        return obj
