"""Generic recording archives and an explicit canonical-telemetry adapter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path

import numpy as np

from glassbox._learner_arrays import load_arrays, save_arrays
from glassbox.learner import _contract
from glassbox.recordings import SequenceCollection, SequenceSegment, segments_from_mask

_FORMAT = "glassbox-recordings-v1"
_OBSERVED_CHANNELS = (
    "velocity_north [m/s,world_nwu]",
    "velocity_west [m/s,world_nwu]",
    "velocity_up [m/s,world_nwu]",
    "body_rate_x [rad/s,body_flu]",
    "body_rate_y [rad/s,body_flu]",
    "body_rate_z [rad/s,body_flu]",
) + tuple(
    f"rotation_{row}{column} [unitless,body_flu_to_world_nwu]"
    for row in range(3)
    for column in range(3)
)


def save_recordings(recordings: SequenceCollection, path: str | Path) -> None:
    """Write all recording facts and arrays in one fingerprinted NPZ archive."""
    contract = _contract(recordings)
    metadata = dict(format=_FORMAT, contract=contract, segments=[])
    arrays = {}
    for index, segment in enumerate(recordings.segments):
        metadata["segments"].append(
            dict(
                recording_id=segment.recording_id,
                segment_id=segment.segment_id,
                start_row=int(segment.start_row),
                dt_s=float(segment.dt_s),
                excitation_declared=segment.excitation is not None,
            )
        )
        for name in ("states", "inputs", "excitation"):
            value = getattr(segment, name)
            if value is not None:
                arrays[f"segment_{index}_{name}"] = value
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        save_arrays(handle, metadata, arrays)


def load_recordings(path: str | Path) -> SequenceCollection:
    """Load a generic archive, rejecting altered contents or an unknown format."""
    try:
        metadata, arrays = load_arrays(path)
        if metadata["format"] != _FORMAT or set(metadata) != {
            "format",
            "contract",
            "segments",
        }:
            raise ValueError("unsupported recording archive format")
        contract = metadata["contract"]
        segments, expected = [], set()
        for index, item in enumerate(metadata["segments"]):
            if set(item) != {
                "recording_id",
                "segment_id",
                "start_row",
                "dt_s",
                "excitation_declared",
            } or not isinstance(item["excitation_declared"], bool):
                raise ValueError("invalid recording segment metadata")
            names = ["states", "inputs"]
            if item["excitation_declared"]:
                names.append("excitation")
            values = {name: arrays[f"segment_{index}_{name}"] for name in names}
            expected.update(f"segment_{index}_{name}" for name in names)
            segments.append(
                SequenceSegment(
                    item["recording_id"],
                    item["segment_id"],
                    dt_s=item["dt_s"],
                    start_row=item["start_row"],
                    **values,
                )
            )
        if set(arrays) != expected:
            raise ValueError("recording archive arrays differ from its segments")
        recordings = SequenceCollection(
            tuple(segments),
            configuration_id=contract["configuration_id"],
            state_channels=tuple(contract["state_channels"]),
            input_channels=tuple(contract["input_channels"]),
        )
        if _contract(recordings) != contract:
            raise ValueError("recording archive contract differs from its segments")
        return recordings
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("invalid recording archive metadata or arrays") from error


def concatenate_recordings(
    collections: Sequence[SequenceCollection],
) -> SequenceCollection:
    """Combine compatible archives without renaming or merging their boundaries."""
    collections = tuple(collections)
    if not collections:
        raise ValueError("at least one recording collection is required")
    contract = _contract(collections[0])
    if any(_contract(item) != contract for item in collections[1:]):
        raise ValueError("recording configuration, channels or sample interval differ")
    return SequenceCollection(
        tuple(segment for item in collections for segment in item.segments),
        configuration_id=contract["configuration_id"],
        state_channels=tuple(contract["state_channels"]),
        input_channels=tuple(contract["input_channels"]),
    )


def from_trajectories(named_trajectories: Mapping, *, configuration_id: str):
    """Adapt named canonical telemetry to velocity, rates, rotation and commands.

    Mapping keys are the caller's whole-recording identities. The supplied
    configuration identity and each trajectory's channel contract remain data
    facts; no vehicle family is selected. The median interval of the first
    trajectory, rounded to twelve significant digits, establishes the grid.
    This removes timestamp-subtraction noise without changing explicit generic
    recording intervals. Gaps and invalid observed rows end a
    segment, preserving source row offsets and outgoing-command alignment.
    Position, training-only observations and pre-recording command prefixes
    are not predicted; forecasts require observed history inside a segment.
    Exogenous prediction inputs require an explicit application adapter.
    """
    from glassbox.core.data import RIGID_BODY_STATE_SCHEMA, Trajectory
    from glassbox.core.geometry import quaternion_to_rotation_matrices

    if not isinstance(named_trajectories, Mapping) or not named_trajectories:
        raise ValueError("supply a nonempty mapping of recording IDs to trajectories")
    first = next(iter(named_trajectories.values()))
    if not isinstance(first, Trajectory):
        raise TypeError("canonical recording values must be Trajectory objects")
    dt_s = float(f"{first.nominal_dt_s:.12g}")
    spec = first.spec.prediction_spec()
    channels = tuple(
        f"{c.name} [{c.unit},{c.frame or 'unframed'},{c.semantic},{c.role}]"
        for c in spec.controls
    )
    segments = []
    for name, trajectory in named_trajectories.items():
        if not isinstance(trajectory, Trajectory):
            raise TypeError("canonical recording values must be Trajectory objects")
        if trajectory.spec.prediction_spec() != spec:
            raise ValueError("canonical trajectory prediction contracts differ")
        if spec.state_schema != RIGID_BODY_STATE_SCHEMA or spec.exogenous:
            raise ValueError(
                "adapter requires canonical rigid-body states without exogenous inputs"
            )
        states = np.asarray(trajectory.states)
        rotation = quaternion_to_rotation_matrices(states[:, 6:10])
        observed = np.concatenate(
            (states[:, 3:6], states[:, 10:13], rotation.reshape(-1, 9)), axis=1
        )
        inputs = np.asarray(trajectory.controls)
        valid = np.isfinite(observed).all(axis=1)
        valid[:-1] &= np.isfinite(inputs).all(axis=1)
        uniform = np.isclose(np.diff(trajectory.time_s), dt_s, rtol=1e-6, atol=1e-9)
        boundaries = np.r_[0, np.flatnonzero(~uniform) + 1, len(states)]
        retained_before = len(segments)
        for start, stop in pairwise(boundaries):
            block = np.zeros(len(states), dtype=bool)
            block[start:stop] = True
            segments.extend(
                segments_from_mask(name, observed, inputs, valid & block, dt_s=dt_s)
            )
        if len(segments) == retained_before:
            raise ValueError(
                f"recording {name!r} has no valid segment on the sample grid"
            )
    result = SequenceCollection(
        tuple(segments),
        configuration_id=configuration_id,
        state_channels=_OBSERVED_CHANNELS,
        input_channels=channels,
    )
    _contract(result)
    return result
