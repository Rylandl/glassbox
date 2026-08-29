"""Pinned fixed-wing PX4 ULogs from the IDF-DS telemetry release."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
import zipfile
import zlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from pyulog import ULog

from glassbox.data import Trajectory, load_trajectory_npz, save_trajectory_npz
from glassbox.px4_ulog import (
    PX4IngestConfig,
    inspect_ulog,
    load_px4_trajectories,
)


IDF_REFERENCE_NAME = "idf_ds_fixedwing_telemetry"
IDF_REFERENCE_RECORD = "https://zenodo.org/records/16992976"
IDF_REFERENCE_DOI = "10.5281/zenodo.16992976"
IDF_REFERENCE_PAPER = "https://doi.org/10.1038/s41597-026-06716-3"
IDF_REFERENCE_LICENSE = "CC-BY-4.0"
IDF_ARCHIVE_URL = (
    "https://zenodo.org/api/records/16992976/files/"
    "Holybro%20Pixhawk.zip/content"
)
IDF_ARCHIVE_FILENAME = "Holybro Pixhawk.zip"
IDF_ARCHIVE_SIZE_BYTES = 2_121_943_653
IDF_ARCHIVE_MD5 = "8b990cc4c7ec1225a16e9a28225e5162"
IDF_ARCHIVE_ROOT = "Holybro Pixhawk/rawdata"
IDF_CONFIGURATION_ID = "idf_ds_holybro_pixhawk_fixedwing"
IDF_SAMPLE_RATE_HZ = 50.0
IDF_MAX_GAP_S = 0.20
IDF_MIN_SEGMENT_DURATION_S = 10.0
IDF_MISSING_RAW_FLIGHTS = (117, 118, 119, 120)


@dataclass(frozen=True)
class IDFRecording:
    """One raw PX4 session in the pinned Holybro archive."""

    filename: str
    size_bytes: int
    crc32: int
    session: int
    recording_date: str
    published_flights: tuple[int, ...]

    @property
    def archive_path(self) -> str:
        return f"{IDF_ARCHIVE_ROOT}/{self.filename}"


IDF_RECORDINGS = (
    IDFRecording(
        "log_67_2025-8-21-14-24-48.ulg",
        122_463_132,
        0x80F8898D,
        1,
        "2025-08-21",
        (1, 2, 3, 4, 5, 7, 8, 34),
    ),
    IDFRecording(
        "log_70_2025-8-21-15-57-48.ulg",
        165_692_958,
        0x3CD4FB07,
        2,
        "2025-08-21",
        tuple(range(10, 20)),
    ),
    IDFRecording(
        "log_74_2025-8-21-17-46-14.ulg",
        157_755_567,
        0xE570D71E,
        3,
        "2025-08-21",
        (9, 20, 21, 22, 23, 25, 26, 27, 28),
    ),
    IDFRecording(
        "log_84_2025-8-22-10-27-06.ulg",
        143_075_620,
        0x0E00AC0C,
        4,
        "2025-08-22",
        (29, 30, 31, 33, 35, 36, 37, 38),
    ),
    IDFRecording(
        "log_85_2025-8-22-11-19-58.ulg",
        141_563_433,
        0x8E4C82EA,
        5,
        "2025-08-22",
        tuple(range(39, 48)),
    ),
    IDFRecording(
        "log_86_2025-8-22-12-18-38.ulg",
        145_631_608,
        0xF1823B9B,
        6,
        "2025-08-22",
        (6, 24, 32, 48, 49, 50, 51, 52, 53),
    ),
    IDFRecording(
        "log_87_2025-8-22-13-09-40.ulg",
        142_772_399,
        0xD42D03B9,
        7,
        "2025-08-22",
        tuple(range(54, 63)),
    ),
    IDFRecording(
        "log_88_2025-8-22-14-04-34.ulg",
        153_743_888,
        0xC82F4C7E,
        8,
        "2025-08-22",
        tuple(range(63, 72)),
    ),
    IDFRecording(
        "log_92_2025-8-23-09-54-12.ulg",
        136_516_253,
        0xDA71D4F4,
        9,
        "2025-08-23",
        tuple(range(72, 81)),
    ),
    IDFRecording(
        "log_93_2025-8-23-10-43-56.ulg",
        138_820_627,
        0x25240D1E,
        10,
        "2025-08-23",
        tuple(range(81, 90)),
    ),
    IDFRecording(
        "log_112_2025-8-23-12-26-12.ulg",
        137_569_805,
        0x628AF7BD,
        11,
        "2025-08-23",
        tuple(range(90, 99)),
    ),
    IDFRecording(
        "log_113_2025-8-23-13-14-28.ulg",
        136_986_890,
        0x83ED389F,
        12,
        "2025-08-23",
        tuple(range(99, 108)),
    ),
    IDFRecording(
        "log_115_2025-8-23-14-06-08.ulg",
        137_877_626,
        0x29581259,
        13,
        "2025-08-23",
        tuple(range(108, 117)),
    ),
)

_RECORDING_BY_FILENAME = {
    recording.filename: recording for recording in IDF_RECORDINGS
}


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _crc32(path: Path) -> int:
    checksum = 0
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            checksum = zlib.crc32(chunk, checksum)
    return checksum & 0xFFFFFFFF


def _recording_for_path(path: Path) -> IDFRecording:
    try:
        return _RECORDING_BY_FILENAME[path.name]
    except KeyError as error:
        raise ValueError(f"unrecognized IDF-DS ULog filename {path.name!r}") from error


def _reference_metadata(recording: IDFRecording) -> dict[str, Any]:
    return {
        "name": IDF_REFERENCE_NAME,
        "record": IDF_REFERENCE_RECORD,
        "doi": IDF_REFERENCE_DOI,
        "paper": IDF_REFERENCE_PAPER,
        "license": IDF_REFERENCE_LICENSE,
        "archive": IDF_ARCHIVE_FILENAME,
        "archive_md5": IDF_ARCHIVE_MD5,
        "archive_path": recording.archive_path,
        "expected_size_bytes": recording.size_bytes,
        "expected_crc32": f"{recording.crc32:08x}",
        "published_flights": list(recording.published_flights),
        "missing_raw_flights": list(IDF_MISSING_RAW_FLIGHTS),
    }


@dataclass(frozen=True)
class IDFFixedWingAdapter:
    """Convert one IDF-DS PX4 session into canonical fixed-wing segments."""

    verify_checksum: bool = True
    name: str = "idf_ds_px4_ulog"

    def _identify(self, path: str | Path) -> tuple[Path, IDFRecording, int]:
        source_path = Path(path)
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        recording = _recording_for_path(source_path)
        size = source_path.stat().st_size
        if self.verify_checksum and size != recording.size_bytes:
            raise ValueError(
                f"size mismatch for pinned IDF-DS ULog {source_path}; "
                f"expected {recording.size_bytes}, got {size}"
            )
        checksum = _crc32(source_path)
        if self.verify_checksum and checksum != recording.crc32:
            raise ValueError(
                f"CRC32 mismatch for pinned IDF-DS ULog {source_path}; "
                f"expected {recording.crc32:08x}, got {checksum:08x}"
            )
        return source_path, recording, checksum

    def inspect(self, path: str | Path) -> dict[str, Any]:
        """Validate and inventory one raw IDF-DS ULog session."""

        source_path, recording, checksum = self._identify(path)
        inventory = inspect_ulog(source_path)
        inventory.update(
            {
                "adapter": {"name": self.name, "schema_version": 2},
                "reference_dataset": _reference_metadata(recording),
                "crc32": f"{checksum:08x}",
                "checksum_matches_pinned_snapshot": checksum == recording.crc32,
            }
        )
        return inventory

    def load_all(self, path: str | Path) -> tuple[Trajectory, ...]:
        """Return every useful interval without bridging telemetry dropouts."""

        source_path, recording, checksum = self._identify(path)
        extracted = load_px4_trajectories(
            source_path,
            config=PX4IngestConfig(
                platform="fixedwing",
                sample_rate_hz=IDF_SAMPLE_RATE_HZ,
                max_gap_s=IDF_MAX_GAP_S,
                min_duration_s=IDF_MIN_SEGMENT_DURATION_S,
                profile="waypoint_circuit",
                condition="outdoor_real_flight",
                replicate=recording.session,
                vehicle_id=IDF_CONFIGURATION_ID,
            ),
        )
        trajectories: list[Trajectory] = []
        for segment, trajectory in enumerate(extracted, start=1):
            provenance = dict(trajectory.provenance)
            px4 = dict(provenance.get("px4", {}))
            estimated_context = px4.pop("exogenous", None)
            if estimated_context is not None:
                px4["excluded_exogenous"] = {
                    **estimated_context,
                    "reason": (
                        "derived wind estimate is retained for audit but excluded "
                        "from the opinionated benchmark input contract"
                    ),
                }
            provenance["px4"] = px4
            provenance.update(
                {
                    "source_crc32": f"{checksum:08x}",
                    "adapter": {"name": self.name, "schema_version": 2},
                    "reference_dataset": _reference_metadata(recording),
                }
            )
            trajectories.append(
                Trajectory(
                    time_s=trajectory.time_s,
                    states=trajectory.states,
                    controls=trajectory.controls,
                    spec=replace(trajectory.spec, exogenous=()),
                    observations=trajectory.observations,
                    labels={
                        **trajectory.labels,
                        "benchmark": IDF_REFERENCE_NAME,
                        "recording_date": recording.recording_date,
                        "session": recording.session,
                        "segment": segment,
                        "source_group": f"idf_session_{recording.session:02d}",
                    },
                    provenance=provenance,
                )
            )
        return tuple(trajectories)

    def load(self, path: str | Path) -> Trajectory:
        """Return the longest valid interval for the one-trajectory protocol."""

        return max(self.load_all(path), key=lambda item: len(item.controls))


def fetch_idf_archive(
    destination: str | Path,
    *,
    overwrite: bool = False,
    timeout_s: float = 60.0,
) -> Path:
    """Download and verify the pinned 2.1 GB Holybro PX4 archive."""

    if timeout_s <= 0.0:
        raise ValueError("timeout_s must be positive")
    target = Path(destination) / IDF_ARCHIVE_FILENAME
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.stat().st_size == IDF_ARCHIVE_SIZE_BYTES and _md5(target) == IDF_ARCHIVE_MD5:
            return target
        if not overwrite:
            raise FileExistsError(
                f"existing file does not match pinned IDF-DS archive: {target}"
            )

    request = urllib.request.Request(
        IDF_ARCHIVE_URL,
        headers={"User-Agent": "glassbox-idf-reference-adapter/1"},
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
        if temporary_path.stat().st_size != IDF_ARCHIVE_SIZE_BYTES:
            raise ValueError("downloaded IDF-DS archive has the wrong size")
        if _md5(temporary_path) != IDF_ARCHIVE_MD5:
            raise ValueError("downloaded IDF-DS archive has the wrong MD5")
        os.replace(temporary_path, target)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return target


def extract_idf_ulogs(
    archive_path: str | Path,
    destination: str | Path,
    *,
    overwrite: bool = False,
) -> tuple[Path, ...]:
    """Extract only the 13 available raw PX4 sessions from the full archive."""

    archive = Path(archive_path)
    destination_root = Path(destination)
    extracted: list[Path] = []
    with zipfile.ZipFile(archive) as source_archive:
        for recording in IDF_RECORDINGS:
            info = source_archive.getinfo(recording.archive_path)
            if info.file_size != recording.size_bytes or info.CRC != recording.crc32:
                raise ValueError(
                    f"archive member metadata mismatch for {recording.archive_path}"
                )
            target = destination_root / recording.filename
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                valid = (
                    target.stat().st_size == recording.size_bytes
                    and _crc32(target) == recording.crc32
                )
                if valid:
                    extracted.append(target)
                    continue
                if not overwrite:
                    raise FileExistsError(
                        f"existing file does not match pinned IDF-DS ULog: {target}"
                    )

            temporary_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    dir=target.parent,
                    prefix=f".{target.name}.",
                    suffix=".extract",
                    delete=False,
                ) as temporary:
                    temporary_path = Path(temporary.name)
                    with source_archive.open(info) as member:
                        shutil.copyfileobj(member, temporary)
                if (
                    temporary_path.stat().st_size != recording.size_bytes
                    or _crc32(temporary_path) != recording.crc32
                ):
                    raise ValueError(
                        f"extracted ULog checksum mismatch for {recording.archive_path}"
                    )
                os.replace(temporary_path, target)
                temporary_path = None
            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)
            extracted.append(target)
    return tuple(extracted)


def extract_idf_reference(
    source_root: str | Path,
    output_root: str | Path,
    *,
    adapter: IDFFixedWingAdapter | None = None,
) -> tuple[Path, ...]:
    """Convert all raw sessions into dropout-safe canonical trajectories."""

    source_directory = Path(source_root)
    output_directory = Path(output_root)
    selected_adapter = IDFFixedWingAdapter() if adapter is None else adapter
    outputs: list[Path] = []
    for recording in IDF_RECORDINGS:
        source_path = source_directory / recording.filename
        for segment, trajectory in enumerate(
            selected_adapter.load_all(source_path), start=1
        ):
            output_path = output_directory / (
                f"idf_session_{recording.session:02d}_"
                f"{source_path.stem}_segment_{segment:02d}.npz"
            )
            save_trajectory_npz(trajectory, output_path)
            outputs.append(output_path)
    return tuple(outputs)


def idf_corpus_report(
    trajectory_paths: Sequence[str | Path],
    raw_ulog_root: str | Path,
) -> dict[str, Any]:
    """Audit the prepared fixed-wing corpus and its retained raw duration."""

    paths = tuple(Path(path) for path in trajectory_paths)
    if not paths:
        raise ValueError("at least one canonical IDF-DS trajectory is required")

    durations: list[float] = []
    intervals = 0
    path_length_m = 0.0
    maximum_speed_m_s = 0.0
    maximum_angular_speed_rad_s = 0.0
    control_count = 0
    control_sum: np.ndarray | None = None
    control_square_sum: np.ndarray | None = None
    control_minimum: np.ndarray | None = None
    control_maximum: np.ndarray | None = None
    spec_payload: dict[str, Any] | None = None
    canonical_by_session: dict[int, dict[str, float | int]] = {}

    for path in paths:
        trajectory = load_trajectory_npz(path)
        current_spec = trajectory.spec.to_dict()
        if spec_payload is None:
            spec_payload = current_spec
        elif current_spec != spec_payload:
            raise ValueError(f"inconsistent IDF-DS trajectory spec: {path}")

        duration_s = float(trajectory.time_s[-1])
        durations.append(duration_s)
        intervals += len(trajectory.controls)
        session = int(trajectory.labels["session"])
        session_summary = canonical_by_session.setdefault(
            session, {"segment_count": 0, "duration_s": 0.0}
        )
        session_summary["segment_count"] = int(session_summary["segment_count"]) + 1
        session_summary["duration_s"] = (
            float(session_summary["duration_s"]) + duration_s
        )

        position = trajectory.states[:, 0:3]
        velocity = trajectory.states[:, 3:6]
        angular_velocity = trajectory.states[:, 10:13]
        path_length_m += float(
            np.sum(np.linalg.norm(np.diff(position, axis=0), axis=1))
        )
        maximum_speed_m_s = max(
            maximum_speed_m_s,
            float(np.max(np.linalg.norm(velocity, axis=1))),
        )
        maximum_angular_speed_rad_s = max(
            maximum_angular_speed_rad_s,
            float(np.max(np.linalg.norm(angular_velocity, axis=1))),
        )

        controls = trajectory.controls
        if control_sum is None:
            control_sum = np.zeros(trajectory.control_size)
            control_square_sum = np.zeros(trajectory.control_size)
            control_minimum = np.full(trajectory.control_size, np.inf)
            control_maximum = np.full(trajectory.control_size, -np.inf)
        control_sum += np.sum(controls, axis=0)
        control_square_sum += np.sum(controls * controls, axis=0)
        control_minimum = np.minimum(control_minimum, np.min(controls, axis=0))
        control_maximum = np.maximum(control_maximum, np.max(controls, axis=0))
        control_count += len(controls)

    assert spec_payload is not None
    assert control_sum is not None
    assert control_square_sum is not None
    assert control_minimum is not None
    assert control_maximum is not None
    control_mean = control_sum / control_count
    control_variance = np.maximum(
        control_square_sum / control_count - control_mean * control_mean,
        0.0,
    )

    raw_root = Path(raw_ulog_root)
    raw_duration_s = 0.0
    sessions: list[dict[str, Any]] = []
    represented_flights: set[int] = set()
    for recording in IDF_RECORDINGS:
        ulog = ULog(
            str(raw_root / recording.filename),
            message_name_filter_list=["actuator_armed"],
        )
        session_raw_duration_s = float(
            (ulog.last_timestamp - ulog.start_timestamp) * 1e-6
        )
        raw_duration_s += session_raw_duration_s
        represented_flights.update(recording.published_flights)
        canonical = canonical_by_session.get(
            recording.session, {"segment_count": 0, "duration_s": 0.0}
        )
        canonical_duration_s = float(canonical["duration_s"])
        sessions.append(
            {
                "session": recording.session,
                "filename": recording.filename,
                "recording_date": recording.recording_date,
                "published_flights": list(recording.published_flights),
                "raw_duration_s": session_raw_duration_s,
                "canonical_duration_s": canonical_duration_s,
                "retained_duration_ratio": (
                    canonical_duration_s / session_raw_duration_s
                ),
                "segment_count": int(canonical["segment_count"]),
            }
        )

    quantile_levels = (0, 10, 25, 50, 75, 90, 100)
    quantiles = np.percentile(np.asarray(durations), quantile_levels)
    canonical_duration_s = float(sum(durations))
    return {
        "format_version": 1,
        "reference_dataset": {
            "name": IDF_REFERENCE_NAME,
            "record": IDF_REFERENCE_RECORD,
            "doi": IDF_REFERENCE_DOI,
            "license": IDF_REFERENCE_LICENSE,
            "archive": IDF_ARCHIVE_FILENAME,
            "archive_size_bytes": IDF_ARCHIVE_SIZE_BYTES,
            "archive_md5": IDF_ARCHIVE_MD5,
            "raw_ulog_count": len(IDF_RECORDINGS),
            "represented_processed_flight_count": len(represented_flights),
            "missing_raw_flights": list(IDF_MISSING_RAW_FLIGHTS),
        },
        "extraction_policy": {
            "sample_rate_hz": IDF_SAMPLE_RATE_HZ,
            "maximum_gap_s": IDF_MAX_GAP_S,
            "minimum_segment_duration_s": IDF_MIN_SEGMENT_DURATION_S,
            "armed_only": True,
            "in_air_only": True,
            "minimum_height_m": 0.2,
        },
        "canonical": {
            "trajectory_count": len(paths),
            "interval_count": intervals,
            "duration_s": canonical_duration_s,
            "raw_duration_s": raw_duration_s,
            "retained_duration_ratio": canonical_duration_s / raw_duration_s,
            "segment_duration_quantiles_s": {
                str(level): float(value)
                for level, value in zip(quantile_levels, quantiles)
            },
            "estimated_path_length_m": path_length_m,
            "maximum_speed_m_s": maximum_speed_m_s,
            "maximum_angular_speed_rad_s": maximum_angular_speed_rad_s,
            "control_statistics": {
                "names": [channel["name"] for channel in spec_payload["controls"]],
                "roles": [channel["role"] for channel in spec_payload["controls"]],
                "minimum": control_minimum.tolist(),
                "maximum": control_maximum.tolist(),
                "mean": control_mean.tolist(),
                "standard_deviation": np.sqrt(control_variance).tolist(),
            },
            "trajectory_spec": spec_payload,
        },
        "sessions": sessions,
    }


def save_idf_corpus_report(report: dict[str, Any], path: str | Path) -> None:
    """Write a deterministic JSON audit of a prepared IDF-DS corpus."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
