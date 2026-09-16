"""Contiguous observation segments and window provenance, without channel semantics.

Row identities refer to a caller's uniformly sampled recording, not raw sensor
events. Masking and clock alignment happen before learning. Distinct segments
cannot overlap within a recording; extraction never bridges their boundaries.
"""

from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from .sequence_model import SequenceBatch


def _positive_int(value):
    return (
        isinstance(value, (int, np.integer))
        and not isinstance(value, bool)
        and value > 0
    )


@dataclass(frozen=True)
class SequenceSegment:
    recording_id: str
    segment_id: str
    states: np.ndarray
    inputs: np.ndarray
    dt_s: float
    start_row: int = 0

    def __post_init__(self):
        x, u = (np.array(a, dtype=float, copy=True) for a in (self.states, self.inputs))
        if (
            not isinstance(self.recording_id, str)
            or not self.recording_id
            or not isinstance(self.segment_id, str)
            or not self.segment_id
            or x.ndim != 2
            or u.ndim != 2
            or min(x.shape) < 1
            or len(x) < 2
            or u.shape[1] < 1
            or len(u) != len(x) - 1
            or not np.isfinite(x).all()
            or not np.isfinite(u).all()
            or not np.isfinite(self.dt_s)
            or self.dt_s <= 0
            or not isinstance(self.start_row, (int, np.integer))
            or isinstance(self.start_row, bool)
            or self.start_row < 0
        ):
            raise ValueError("invalid segment identity, arrays, row offset or dt_s")
        x.setflags(write=False)
        u.setflags(write=False)
        object.__setattr__(self, "states", x)
        object.__setattr__(self, "inputs", u)


def segments_from_mask(recording_id, states, inputs, valid, *, dt_s):
    """Retain contiguous valid runs of at least two rows; no padding or imputation.

    The caller marks state and outgoing-input validity. Excluded rows may contain
    nonfinite observations; each retained segment is independently validated.
    """
    x, u, valid = map(np.asarray, (states, inputs, valid))
    if (
        x.ndim != 2
        or u.ndim != 2
        or len(u) != len(x) - 1
        or valid.shape != (len(x),)
        or valid.dtype != np.bool_
    ):
        raise ValueError("invalid recording arrays or boolean row mask")
    runs = np.flatnonzero(np.diff(np.r_[False, valid, False].astype(int))).reshape(
        -1, 2
    )
    return tuple(
        SequenceSegment(
            recording_id, f"rows-{a}-{b}", x[a:b], u[a : b - 1], dt_s, int(a)
        )
        for a, b in runs
        if b - a >= 2
    )


@dataclass(frozen=True)
class WindowKey:
    recording_id: str
    segment_id: str
    origin: int  # Row index within this segment; source row = start_row + origin.


@dataclass(frozen=True)
class SequenceWindows:
    batch: SequenceBatch
    keys: tuple[WindowKey, ...]
    source_origins: tuple[int, ...]

    def __post_init__(self):
        keys, origins = tuple(self.keys), tuple(self.source_origins)
        if (
            not isinstance(self.batch, SequenceBatch)
            or len(keys) != len(self.batch.past_states)
            or len(origins) != len(keys)
            or not all(isinstance(k, WindowKey) for k in keys)
            or len(set(keys)) != len(keys)
            or not all(_positive_int(i) for i in origins)
        ):
            raise ValueError("window provenance must match batch rows")
        object.__setattr__(self, "keys", keys)
        object.__setattr__(self, "source_origins", origins)

    def coverage(self):
        """Union of sampled grid rows/edges, so overlapping windows count once."""
        p = self.batch.past_inputs.shape[1]
        h = self.batch.future_inputs.shape[1]
        records = {}
        for key, origin in zip(self.keys, self.source_origins, strict=True):
            r = records.setdefault(
                key.recording_id,
                dict(states=set(), inputs=set(), segments=set(), windows=0),
            )
            r["states"].update(range(origin - p, origin + h + 1))
            r["inputs"].update(range(origin - p, origin + h))
            r["segments"].add(key.segment_id)
            r["windows"] += 1
        return {
            name: dict(
                windows=r["windows"],
                segments=len(r["segments"]),
                unique_state_rows=len(r["states"]),
                unique_input_rows=len(r["inputs"]),
                observed_transition_time_s=len(r["inputs"]) * self.batch.dt_s,
            )
            for name, r in records.items()
        }


@dataclass(frozen=True)
class SequenceCollection:
    segments: tuple[SequenceSegment, ...]
    configuration_id: str | None = None
    state_channels: tuple[str, ...] = ()
    input_channels: tuple[str, ...] = ()

    def __post_init__(self):
        segments = tuple(self.segments)
        if not segments or not all(isinstance(s, SequenceSegment) for s in segments):
            raise ValueError("a collection needs validated segments")
        identities = [(s.recording_id, s.segment_id) for s in segments]
        first = segments[0]
        if self.configuration_id is not None and (
            not isinstance(self.configuration_id, str)
            or not self.configuration_id.strip()
        ):
            raise ValueError("configuration identity must be a nonempty string")
        for name, width in (
            ("state_channels", first.states.shape[1]),
            ("input_channels", first.inputs.shape[1]),
        ):
            channels = tuple(getattr(self, name))
            if channels and (
                len(channels) != width
                or any(not isinstance(c, str) or not c.strip() for c in channels)
                or len(set(channels)) != len(channels)
            ):
                raise ValueError(
                    "channel identities must be nonempty, unique and match array columns"
                )
            object.__setattr__(self, name, channels)
        if len(set(identities)) != len(identities) or any(
            s.dt_s != first.dt_s
            or s.states.shape[1] != first.states.shape[1]
            or s.inputs.shape[1] != first.inputs.shape[1]
            for s in segments
        ):
            raise ValueError(
                "duplicate segments or inconsistent sample interval/channels"
            )
        for recording in {s.recording_id for s in segments}:
            selected = sorted(
                (s for s in segments if s.recording_id == recording),
                key=lambda s: s.start_row,
            )
            if any(
                a.start_row + len(a.states) > b.start_row for a, b in pairwise(selected)
            ):
                raise ValueError("segments overlap within a recording")
        object.__setattr__(self, "segments", segments)

    def window_keys(self, *, history_steps, horizon_steps, stride=1):
        if not all(_positive_int(v) for v in (history_steps, horizon_steps, stride)):
            raise ValueError("window lengths and stride must be positive integers")
        return tuple(
            WindowKey(s.recording_id, s.segment_id, i)
            for s in self.segments
            for i in range(history_steps, len(s.states) - horizon_steps, stride)
        )

    def extract(self, keys, *, history_steps, horizon_steps):
        """Extract explicitly chosen keys in order, retaining identities and coverage."""
        keys = tuple(keys)
        if not all(_positive_int(v) for v in (history_steps, horizon_steps)):
            raise ValueError("window lengths must be positive integers")
        if (
            len(keys) < 3
            or not all(isinstance(k, WindowKey) for k in keys)
            or len(set(keys)) != len(keys)
        ):
            raise ValueError("at least three distinct window keys are required")
        lookup = {(s.recording_id, s.segment_id): s for s in self.segments}
        arrays = {
            k: []
            for k in ("past_states", "past_inputs", "future_inputs", "future_states")
        }
        origins = []
        for key in keys:
            s = lookup.get((key.recording_id, key.segment_id))
            a = key.origin
            if (
                s is None
                or not _positive_int(a)
                or a < history_steps
                or a + horizon_steps >= len(s.states)
            ):
                raise ValueError("unknown segment or incomplete window")
            arrays["past_states"].append(s.states[a - history_steps : a + 1])
            arrays["past_inputs"].append(s.inputs[a - history_steps : a])
            arrays["future_inputs"].append(s.inputs[a : a + horizon_steps])
            arrays["future_states"].append(s.states[a + 1 : a + horizon_steps + 1])
            origins.append(int(s.start_row + a))
        return SequenceWindows(
            SequenceBatch(
                **{k: np.stack(v) for k, v in arrays.items()},
                dt_s=self.segments[0].dt_s,
            ),
            keys,
            tuple(origins),
        )
