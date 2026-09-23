"""Validated recording windows and their shared-row consistency."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SequenceBatch:
    """Past x[-P:0], u[-P:-1]; future u[0:H-1], target x[1:H]."""

    past_states: np.ndarray
    past_inputs: np.ndarray
    future_inputs: np.ndarray
    future_states: np.ndarray
    dt_s: float

    def __post_init__(self):
        for key in ("past_states", "past_inputs", "future_inputs", "future_states"):
            value = np.array(getattr(self, key), dtype=float, copy=True)
            if value.ndim != 3 or not np.isfinite(value).all():
                raise ValueError(f"{key} must be a finite [batch,time,channel] array")
            value.setflags(write=False)
            object.__setattr__(self, key, value)
        n, p1, d = self.past_states.shape
        nf, h, u = self.future_inputs.shape
        if (
            n < 3
            or min(p1 - 1, d, h, u) < 1
            or nf != n
            or self.past_inputs.shape != (n, p1 - 1, u)
            or self.future_states.shape != (n, h, d)
            or not np.isfinite(self.dt_s)
            or self.dt_s <= 0
        ):
            raise ValueError("sequence shapes or dt_s are inconsistent")


def validate_window_consistency(batch, keys, source_origins):
    """Require one state/command value for each retained recording source row.

    Windows may overlap within a segment. Their history and forecast arrays must
    agree at every shared row, including the history/forecast boundary. Different
    segments in the same recording cannot claim overlapping source rows. The
    check uses only saved observations; it never runs the dynamics model.
    """
    history = batch.past_inputs.shape[1]
    horizon = batch.future_inputs.shape[1]
    records = {}
    for index, key in enumerate(keys):
        records.setdefault(key.recording_id, []).append(index)
    starts = np.asarray(source_origins) - history
    for indices in records.values():
        identities = {}
        segment_ids = np.array(
            [
                identities.setdefault(keys[i].segment_id, len(identities))
                for i in indices
            ]
        )
        for past, future, count in (
            (batch.past_states, batch.future_states, history + horizon + 1),
            (batch.past_inputs, batch.future_inputs, history + horizon),
        ):
            rows = (starts[indices, None] + np.arange(count)).reshape(-1)
            values = np.concatenate((past[indices], future[indices]), axis=1)
            values = values.reshape(-1, values.shape[-1])
            order = np.argsort(rows, kind="stable")
            ordered_rows = rows[order]
            repeats = ordered_rows[1:] == ordered_rows[:-1]
            segments = np.repeat(segment_ids, count)[order]
            if np.any(segments[1:][repeats] != segments[:-1][repeats]):
                raise ValueError("retained segments overlap within a recording")
            ordered_values = values[order]
            if np.any(ordered_values[1:][repeats] != ordered_values[:-1][repeats]):
                raise ValueError(
                    "overlapping windows disagree on recording source rows"
                )
