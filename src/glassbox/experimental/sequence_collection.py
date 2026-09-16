"""Contiguous observation segments and window provenance, without channel semantics.

Row identities refer to a caller's uniformly sampled recording, not raw sensor
events. Masking and clock alignment happen before learning. Distinct segments
cannot overlap within a recording; extraction never bridges their boundaries.

A segment may also declare, per applied command, the exogenous component the
caller injected into it: a data fact about the recording with a declared
meaning, like a channel's units. It is optional, validated like every other
array, carried through masking and window extraction, and read by nothing here.
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
    """One contiguous block of observations and the commands applied across it.

    ``excitation`` is optional and, when supplied, is aligned row for row and
    column for column with ``inputs``: the exogenous component the caller
    injected into each applied command, zero where none was injected. It is a
    data fact about the recording in the same sense as the channel identities,
    declared by whoever applied it rather than inferred here, and it is
    validated exactly as the other arrays are.
    """

    recording_id: str
    segment_id: str
    states: np.ndarray
    inputs: np.ndarray
    dt_s: float
    start_row: int = 0
    excitation: np.ndarray | None = None

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
        if self.excitation is not None:
            e = np.array(self.excitation, dtype=float, copy=True)
            if e.shape != u.shape or not np.isfinite(e).all():
                raise ValueError(
                    "declared excitation must be finite and aligned with the inputs"
                )
            e.setflags(write=False)
            object.__setattr__(self, "excitation", e)


def segments_from_mask(recording_id, states, inputs, valid, *, dt_s, excitation=None):
    """Retain contiguous valid runs of at least two rows; no padding or imputation.

    The caller marks state and outgoing-input validity. Excluded rows may contain
    nonfinite observations; each retained segment is independently validated.
    ``excitation``, when supplied, is aligned with ``inputs`` and is cut the same
    way, so a retained segment carries the excitation of exactly its own rows.
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
    if excitation is not None:
        excitation = np.asarray(excitation, dtype=float)
        if excitation.shape != u.shape:
            raise ValueError("declared excitation must be aligned with the inputs")
    runs = np.flatnonzero(np.diff(np.r_[False, valid, False].astype(int))).reshape(
        -1, 2
    )
    return tuple(
        SequenceSegment(
            recording_id,
            f"rows-{a}-{b}",
            x[a:b],
            u[a : b - 1],
            dt_s,
            int(a),
            None if excitation is None else excitation[a : b - 1],
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
    """Extracted windows, their provenance, and any declared excitation beside them.

    ``past_excitation`` and ``future_excitation`` are optional and, when
    present, are aligned with ``batch.past_inputs`` and ``batch.future_inputs``:
    the exogenous component the caller injected into each of those applied
    commands. They are carried beside the batch rather than inside it, because
    the current recipe trains on the batch alone and ignores them.
    """

    batch: SequenceBatch
    keys: tuple[WindowKey, ...]
    source_origins: tuple[int, ...]
    past_excitation: np.ndarray | None = None
    future_excitation: np.ndarray | None = None

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
        if (self.past_excitation is None) != (self.future_excitation is None):
            raise ValueError(
                "window excitation covers the past and future inputs, or neither"
            )
        for name in ("past_excitation", "future_excitation"):
            value = getattr(self, name)
            if value is None:
                continue
            value = np.array(value, dtype=float, copy=True)
            aligned = getattr(self.batch, name.replace("excitation", "inputs"))
            if value.shape != aligned.shape or not np.isfinite(value).all():
                raise ValueError(
                    "declared excitation must be finite and aligned with the window inputs"
                )
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    @property
    def excitation_declared(self):
        """Whether these windows carry the excitation of their own inputs."""
        return self.past_excitation is not None

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
        # Excitation is declared for a whole collection or for none of it: a
        # half-declared collection would leave "what the caller injected"
        # ambiguous on the segments that said nothing.
        if len({s.excitation is None for s in segments}) != 1:
            raise ValueError(
                "every segment declares its excitation, or none of them does"
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

    @property
    def excitation_declared(self):
        """Whether every recording in this collection declares its excitation."""
        return self.segments[0].excitation is not None

    def window_keys(self, *, history_steps, horizon_steps, stride=1):
        if not all(_positive_int(v) for v in (history_steps, horizon_steps, stride)):
            raise ValueError("window lengths and stride must be positive integers")
        return tuple(
            WindowKey(s.recording_id, s.segment_id, i)
            for s in self.segments
            for i in range(history_steps, len(s.states) - horizon_steps, stride)
        )

    def extract(self, keys, *, history_steps, horizon_steps):
        """Extract explicitly chosen keys in order, retaining identities and coverage.

        A declared excitation is cut with the inputs it belongs to and returned
        beside them, so a window says what the caller injected into every
        command it holds. The current recipe reads the batch and ignores it.
        """
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
        declared = self.excitation_declared
        excitation = {k: [] for k in ("past_excitation", "future_excitation")}
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
            if declared:
                excitation["past_excitation"].append(
                    s.excitation[a - history_steps : a]
                )
                excitation["future_excitation"].append(
                    s.excitation[a : a + horizon_steps]
                )
            origins.append(int(s.start_row + a))
        return SequenceWindows(
            SequenceBatch(
                **{k: np.stack(v) for k, v in arrays.items()},
                dt_s=self.segments[0].dt_s,
            ),
            keys,
            tuple(origins),
            **({k: np.stack(v) for k, v in excitation.items()} if declared else {}),
        )
