"""Pinned PX4 ULogs from ARP Laboratory's system-identification release."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from glassbox.core.data import Trajectory, save_trajectory_npz
from glassbox.io.px4_ulog import PX4IngestConfig, inspect_ulog, load_px4_trajectory

ARP_REFERENCE_REPOSITORY = (
    "https://github.com/arplaboratory/data-driven-system-identification"
)
ARP_REFERENCE_COMMIT = "2d267dd07b4262f579ee223d20b26a6dc9d17147"
ARP_REFERENCE_PAPER = "https://arxiv.org/abs/2404.07837"
ARP_REFERENCE_MEDIA_ROOT = (
    "https://raw.githubusercontent.com/arplaboratory/"
    f"data-driven-system-identification/{ARP_REFERENCE_COMMIT}"
)
ARP_REFERENCE_LICENSE = "MIT"
ARP_REFERENCE_NAME = "arp_data_driven_system_identification"
ARP_CONFIGURATION_ID = "arp_iros_2024_large_quadrotor"
ARP_SAMPLE_RATE_HZ = 50.0


@dataclass(frozen=True)
class ARPRecording:
    """One immutable raw ULog in the pinned ARP reference snapshot."""

    relative_path: str
    sha256: str
    size_bytes: int
    replicate: int

    @property
    def filename(self) -> str:
        return Path(self.relative_path).name


ARP_RECORDINGS = (
    ARPRecording(
        "logs_large/log_63_2024-1-8-16-37-54.ulg",
        "887fb128983d767449112409143224874ff57c04b5efb8941582748a3aec9cf9",
        12_336_011,
        1,
    ),
    ARPRecording(
        "logs_large/log_64_2024-1-8-16-39-44.ulg",
        "f9517b747ef97a7f3d51b28043bcdd9fc50983c7e938d4b26ff22e8aff7a771d",
        13_568_823,
        2,
    ),
    ARPRecording(
        "logs_large/log_65_2024-1-8-16-40-52.ulg",
        "d2ce8a3c9ce5605cd8d73e9351b8a46640b11d5044b1746bbf15f63854178553",
        14_023_152,
        3,
    ),
    ARPRecording(
        "logs_large/log_66_2024-1-8-16-42-48.ulg",
        "8eafd19ecceeaf11812f94057bb8ee804f10adb457788d5d1889751a6a193ee6",
        18_518_753,
        4,
    ),
)

_RECORDING_BY_FILENAME = {recording.filename: recording for recording in ARP_RECORDINGS}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _recording_for_path(path: Path) -> ARPRecording:
    try:
        return _RECORDING_BY_FILENAME[path.name]
    except KeyError as error:
        raise ValueError(
            f"unrecognized ARP reference ULog filename {path.name!r}"
        ) from error


def _reference_metadata(recording: ARPRecording) -> dict[str, Any]:
    return {
        "name": ARP_REFERENCE_NAME,
        "repository": ARP_REFERENCE_REPOSITORY,
        "commit": ARP_REFERENCE_COMMIT,
        "paper": ARP_REFERENCE_PAPER,
        "license": ARP_REFERENCE_LICENSE,
        "relative_path": recording.relative_path,
        "expected_sha256": recording.sha256,
        "expected_size_bytes": recording.size_bytes,
    }


def _longest_powered_interval(
    trajectory: Trajectory,
    *,
    minimum_mean_motor_command: float = 0.1,
    minimum_duration_s: float = 0.5,
) -> Trajectory:
    """Select the longest powered interval when arming topics are unavailable."""

    if not 0.0 < minimum_mean_motor_command < 1.0:
        raise ValueError("minimum_mean_motor_command must lie in (0, 1)")
    if minimum_duration_s <= 0.0:
        raise ValueError("minimum_duration_s must be positive")
    powered = np.mean(trajectory.controls, axis=1) > minimum_mean_motor_command
    transitions = np.flatnonzero(
        np.diff(np.concatenate(([False], powered, [False])).astype(np.int8))
    )
    runs = [
        (int(start), int(stop))
        for start, stop in transitions.reshape(-1, 2)
        if (stop - start) * trajectory.nominal_dt_s >= minimum_duration_s
    ]
    if not runs:
        raise ValueError("ARP reference trajectory has no sustained powered interval")
    start, stop = max(runs, key=lambda run: run[1] - run[0])
    start_offset_s = float(trajectory.time_s[start] - trajectory.time_s[0])
    selected_time = trajectory.time_s[start : stop + 1]
    provenance = dict(trajectory.provenance)
    provenance["reference_powered_interval"] = {
        "selection": "longest_contiguous_mean_motor_command_above_threshold",
        "minimum_mean_motor_command": minimum_mean_motor_command,
        "minimum_duration_s": minimum_duration_s,
        "candidate_interval_count": len(runs),
        "start_offset_s": start_offset_s,
        "duration_s": float(selected_time[-1] - selected_time[0]),
        "discarded_duration_s": float(
            trajectory.time_s[-1]
            - trajectory.time_s[0]
            - (selected_time[-1] - selected_time[0])
        ),
    }
    return Trajectory(
        time_s=selected_time - selected_time[0],
        states=trajectory.states[start : stop + 1],
        controls=trajectory.controls[start:stop],
        spec=trajectory.spec,
        exogenous=trajectory.exogenous[start : stop + 1],
        observations=trajectory.observations[start : stop + 1],
        labels=trajectory.labels,
        provenance=provenance,
    )


@dataclass(frozen=True)
class ARPReferenceAdapter:
    """Convert one pinned ARP PX4 ULog into a canonical trajectory."""

    verify_checksum: bool = True
    name: str = "arp_px4_ulog_reference"

    def _identify(self, path: str | Path) -> tuple[Path, ARPRecording, str]:
        source_path = Path(path)
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        recording = _recording_for_path(source_path)
        checksum = _sha256(source_path)
        if self.verify_checksum and checksum != recording.sha256:
            raise ValueError(
                f"checksum mismatch for pinned ARP reference ULog {source_path}; "
                f"expected {recording.sha256}, got {checksum}"
            )
        return source_path, recording, checksum

    def inspect(self, path: str | Path) -> dict[str, Any]:
        """Validate and inventory one raw reference ULog."""

        source_path, recording, checksum = self._identify(path)
        inventory = inspect_ulog(source_path)
        inventory.update(
            {
                "adapter": {"name": self.name, "schema_version": 2},
                "reference_dataset": _reference_metadata(recording),
                "sha256": checksum,
                "checksum_matches_pinned_snapshot": checksum == recording.sha256,
            }
        )
        return inventory

    def load(self, path: str | Path) -> Trajectory:
        """Extract the longest telemetry-complete interval at the pinned rate."""

        source_path, recording, checksum = self._identify(path)
        trajectory = load_px4_trajectory(
            source_path,
            config=PX4IngestConfig(
                sample_rate_hz=ARP_SAMPLE_RATE_HZ,
                min_height_m=None,
                only_armed=False,
                only_in_air=False,
                # Pinned to the pre-auto-resolution actuator hold-age tolerance.
                # Recorded ARP fit and evaluation results depend on this exact
                # value; do not let it silently follow PX4IngestConfig's
                # automatic per-log resolution.
                actuator_hold_max_age_s=0.10,
                profile="published_sysid",
                condition="aggressive_real_flight",
                replicate=recording.replicate,
                vehicle_id=ARP_CONFIGURATION_ID,
            ),
        )
        trajectory = _longest_powered_interval(trajectory)
        provenance = dict(trajectory.provenance)
        provenance.update(
            {
                "source_sha256": checksum,
                "adapter": {"name": self.name, "schema_version": 2},
                "reference_dataset": _reference_metadata(recording),
            }
        )
        return Trajectory(
            time_s=trajectory.time_s,
            states=trajectory.states,
            controls=trajectory.controls,
            spec=trajectory.spec,
            exogenous=trajectory.exogenous,
            observations=trajectory.observations,
            labels={
                **trajectory.labels,
                "benchmark": ARP_REFERENCE_NAME,
                "recording_date": "2024-01-08",
                "source_group": (f"{ARP_REFERENCE_NAME}:{recording.relative_path}"),
            },
            provenance=provenance,
        )


def fetch_arp_reference(
    destination: str | Path,
    *,
    overwrite: bool = False,
    timeout_s: float = 60.0,
) -> tuple[Path, ...]:
    """Download and verify the four-ULog pinned ARP snapshot."""

    if timeout_s <= 0.0:
        raise ValueError("timeout_s must be positive")
    destination_root = Path(destination)
    fetched: list[Path] = []
    for recording in ARP_RECORDINGS:
        target = destination_root / recording.relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if _sha256(target) == recording.sha256:
                fetched.append(target)
                continue
            if not overwrite:
                raise FileExistsError(
                    f"existing file does not match pinned ARP reference: {target}"
                )

        request = urllib.request.Request(
            f"{ARP_REFERENCE_MEDIA_ROOT}/{recording.relative_path}",
            headers={"User-Agent": "glassbox-arp-reference-adapter/1"},
        )
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=target.parent,
                prefix=f".{target.name}.",
                suffix=".download",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                with urllib.request.urlopen(request, timeout=timeout_s) as response:
                    shutil.copyfileobj(response, temporary)
            if temporary_path.stat().st_size != recording.size_bytes:
                raise ValueError(
                    f"downloaded size mismatch for {recording.relative_path}"
                )
            checksum = _sha256(temporary_path)
            if checksum != recording.sha256:
                raise ValueError(
                    f"downloaded checksum mismatch for {recording.relative_path}"
                )
            os.replace(temporary_path, target)
            temporary_path = None
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        fetched.append(target)
    return tuple(fetched)


def extract_arp_reference(
    source_root: str | Path,
    output_root: str | Path,
    *,
    adapter: ARPReferenceAdapter | None = None,
) -> tuple[Path, ...]:
    """Convert the complete pinned ARP snapshot into canonical NPZ files."""

    source_directory = Path(source_root)
    output_directory = Path(output_root)
    selected_adapter = ARPReferenceAdapter() if adapter is None else adapter
    outputs: list[Path] = []
    for recording in ARP_RECORDINGS:
        source_path = source_directory / recording.relative_path
        output_path = output_directory / Path(recording.filename).with_suffix(".npz")
        save_trajectory_npz(selected_adapter.load(source_path), output_path)
        outputs.append(output_path)
    return tuple(outputs)
