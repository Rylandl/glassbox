"""Fingerprint-checked canonical motion recording archives."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from glassbox._learner_arrays import load_arrays, save_arrays
from glassbox.learner import _contract
from glassbox.recordings import SequenceCollection, SequenceSegment

_FORMAT = "glassbox-motion-recordings-v1"


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
            )
        )
        for name in ("states", "inputs"):
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
            }:
                raise ValueError("invalid recording segment metadata")
            names = ["states", "inputs"]
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
