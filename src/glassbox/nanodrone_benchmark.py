"""Adapter for the IDSIA Nano-Quadrotor System Identification Benchmark."""

from __future__ import annotations

import csv
import hashlib
import os
import re
import shutil
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from glassbox.data import (
    ControlChannel,
    RIGID_BODY_STATE_SCHEMA,
    Trajectory,
    TrajectorySpec,
    VehicleConfigurationSpec,
    save_trajectory_npz,
    specific_force_observation_channels,
)
from glassbox.dynamics import QUADROTOR_CONTROL_NAMES


BENCHMARK_REPOSITORY = (
    "https://github.com/idsia-robotics/nanodrone-sysid-benchmark"
)
BENCHMARK_COMMIT = "2d921b57d166fe2debe08a5d39bd07297c5abc39"
BENCHMARK_DOI = "10.1016/j.conengprac.2026.106871"
BENCHMARK_MEDIA_ROOT = (
    "https://media.githubusercontent.com/media/idsia-robotics/"
    f"nanodrone-sysid-benchmark/{BENCHMARK_COMMIT}"
)
ROTOR_SPEED_REFERENCE_RAD_S = 2500.0
BENCHMARK_OBSERVATION_SOURCE = "processed_mocap_and_onboard_sensors"
BENCHMARK_CONFIGURATION_ID = (
    "idsia_crazyflie_2_1_brushless_flow_v2_ai_deck"
)

SOURCE_COLUMNS = (
    "t",
    "x",
    "y",
    "z",
    "qx",
    "qy",
    "qz",
    "qw",
    "vx",
    "vy",
    "vz",
    "wx",
    "wy",
    "wz",
    "m1_rads",
    "m2_rads",
    "m3_rads",
    "m4_rads",
    "ax_body",
    "ay_body",
    "az_body",
)
SOURCE_MOTOR_NAMES = ("m1_rads", "m2_rads", "m3_rads", "m4_rads")
# The paper's allocation matrix places m1/m2/m3/m4 at FR/RR/RL/FL.
CANONICAL_MOTOR_SOURCE_INDICES = (3, 0, 1, 2)


@dataclass(frozen=True)
class BenchmarkRecording:
    relative_path: str
    sha256: str
    size_bytes: int

    @property
    def filename(self) -> str:
        return Path(self.relative_path).name

    @property
    def split(self) -> str:
        return Path(self.relative_path).parts[-2]


BENCHMARK_RECORDINGS = (
    BenchmarkRecording(
        "data/test/melon_20251017_run1.csv",
        "41137fbe34e8f3ecec17e9c00c99145898677906c1ee662ecf8491afeb4c678e",
        2543109,
    ),
    BenchmarkRecording(
        "data/test/melon_20251017_run2.csv",
        "851771f39ad891428b097242a5c88eed48ae82e24dcb303b5b03b38acdb18dd1",
        2537432,
    ),
    BenchmarkRecording(
        "data/test/melon_20251017_run3.csv",
        "f7bc4e38086d8674d0f7ed8f08bb532ce90b1e7ef97fbb00fb51629e1d1c47ed",
        2549177,
    ),
    BenchmarkRecording(
        "data/train/chirp_20251017_run1.csv",
        "00e19738869e223c295232c44771c195f8c2f8c499e5497e27e520c7fc60ca23",
        2360840,
    ),
    BenchmarkRecording(
        "data/train/chirp_20251017_run2.csv",
        "6e5587da5055fc620677aeeb8e016a8e1d8d219464d7b2b82f2667174038c979",
        2352232,
    ),
    BenchmarkRecording(
        "data/train/chirp_20251017_run3.csv",
        "f151f009f23da67551e18b422bc7f2471d9df9dcf0a4e5279b469254bce86f91",
        2352015,
    ),
    BenchmarkRecording(
        "data/train/chirp_20251017_run4.csv",
        "f07baf341a2ee55c84de7867fd7bebccacace2fa4daa8672a7f881631122b30e",
        2348369,
    ),
    BenchmarkRecording(
        "data/train/random_20251017_run1.csv",
        "44b457f31ac71d6f18f8a680f69b909ac3cadbb270051430f2dd722acff4d890",
        2346029,
    ),
    BenchmarkRecording(
        "data/train/random_20251017_run2.csv",
        "64f382ee8c06f71397d6abc42b50b270dc4ba47df5f13ebc3515e43bbfaa0c75",
        2346472,
    ),
    BenchmarkRecording(
        "data/train/random_20251017_run3.csv",
        "716058325a802ea64c772961fd20f21bcf5c1dac6b6dc3d1591dcdf4e69c4ed6",
        2355450,
    ),
    BenchmarkRecording(
        "data/train/random_20251017_run4.csv",
        "15445bf88620a5e875a68e62583a4d77dc514de836d644be5a7e8107686e7d13",
        2353710,
    ),
    BenchmarkRecording(
        "data/train/square_20251017_run1.csv",
        "9b458d207d0a5e21933ba7ad16945bf505ab2b3abf6d151a1442cdd705d6c0cc",
        755523,
    ),
    BenchmarkRecording(
        "data/train/square_20251017_run2.csv",
        "3b47b3712915aff4fd301fd0a2f989bbdce0a5ca1af852c923f472ec7d05d084",
        744850,
    ),
    BenchmarkRecording(
        "data/train/square_20251017_run3.csv",
        "199f1de08d8da4830d79c4fff4e62f313af5ff54af9cac53c9d8341d4a73a2d8",
        744542,
    ),
    BenchmarkRecording(
        "data/train/square_20251017_run4.csv",
        "76b6a2e2400612681e65f041ca31e5c6ba6bea063a944d5d4a75481edd82a671",
        743660,
    ),
)

_RECORDING_BY_FILENAME = {
    recording.filename: recording for recording in BENCHMARK_RECORDINGS
}
_FILENAME_PATTERN = re.compile(
    r"^(?P<profile>chirp|random|square|melon)_"
    r"(?P<date>\d{8})_run(?P<replicate>\d+)\.csv$"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _recording_identity(path: Path) -> tuple[BenchmarkRecording, dict[str, Any]]:
    try:
        recording = _RECORDING_BY_FILENAME[path.name]
    except KeyError as error:
        raise ValueError(
            f"unrecognized Nano-Quadrotor benchmark filename {path.name!r}"
        ) from error
    match = _FILENAME_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"invalid benchmark recording name: {path.name}")
    identity = match.groupdict()
    split = recording.split
    if path.parent.name in {"train", "test"} and path.parent.name != split:
        raise ValueError(
            f"benchmark recording {path.name} belongs to {split!r}, not "
            f"{path.parent.name!r}"
        )
    return recording, {
        "profile": identity["profile"],
        "recording_date": (
            f"{identity['date'][:4]}-{identity['date'][4:6]}-"
            f"{identity['date'][6:]}"
        ),
        "replicate": int(identity["replicate"]),
        "benchmark_split": split,
    }


def _read_source_csv(path: Path) -> tuple[tuple[str, ...], np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", newline="") as source:
        header = tuple(next(csv.reader(source)))
    if header != SOURCE_COLUMNS:
        raise ValueError(
            "unexpected Nano-Quadrotor CSV schema; expected columns "
            f"{SOURCE_COLUMNS}, got {header}"
        )
    data = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    if data.shape[0] < 2 or data.shape[1] != len(SOURCE_COLUMNS):
        raise ValueError(
            "Nano-Quadrotor recording must contain at least two complete rows"
        )
    if not np.all(np.isfinite(data)):
        raise ValueError("Nano-Quadrotor recording contains non-finite values")
    return header, data


def _source_quality(data: np.ndarray) -> dict[str, Any]:
    time_s = data[:, 0]
    time_steps = np.diff(time_s)
    if not np.all(time_steps > 0.0):
        raise ValueError("Nano-Quadrotor timestamps must be strictly increasing")
    nominal_dt_s = float(np.median(time_steps))
    maximum_timing_error_s = float(np.max(np.abs(time_steps - nominal_dt_s)))
    if not np.isclose(nominal_dt_s, 0.01, atol=1e-8, rtol=0.0):
        raise ValueError(
            f"expected 100 Hz benchmark data, got dt={nominal_dt_s:g}s"
        )
    if maximum_timing_error_s > 1e-7:
        raise ValueError(
            "benchmark timestamps are not uniformly sampled: maximum interval "
            f"error is {maximum_timing_error_s:g}s"
        )

    quaternion = data[:, 4:8]
    quaternion_norm = np.linalg.norm(quaternion, axis=1)
    maximum_quaternion_norm_error = float(
        np.max(np.abs(quaternion_norm - 1.0))
    )
    if maximum_quaternion_norm_error > 1e-3:
        raise ValueError(
            "benchmark quaternion norm error exceeds 1e-3: "
            f"{maximum_quaternion_norm_error:g}"
        )
    motor_speeds = data[:, 14:18]
    if np.any(motor_speeds < 0.0):
        raise ValueError("benchmark motor angular velocities cannot be negative")
    return {
        "nominal_dt_s": nominal_dt_s,
        "sample_rate_hz": 1.0 / nominal_dt_s,
        "maximum_timing_error_s": maximum_timing_error_s,
        "maximum_quaternion_norm_error": maximum_quaternion_norm_error,
        "motor_speed_minimum_rad_s": np.min(motor_speeds, axis=0).tolist(),
        "motor_speed_maximum_rad_s": np.max(motor_speeds, axis=0).tolist(),
    }


def nanodrone_trajectory_spec(
    rotor_speed_reference_rad_s: float = ROTOR_SPEED_REFERENCE_RAD_S,
) -> TrajectorySpec:
    """Return the canonical contract emitted by the benchmark adapter."""

    if not np.isfinite(rotor_speed_reference_rad_s) or rotor_speed_reference_rad_s <= 0:
        raise ValueError("rotor_speed_reference_rad_s must be positive and finite")
    controls = tuple(
        ControlChannel(
            name=name,
            role=name,
            semantic="squared_rotor_speed_ratio",
            unit="1",
            minimum=0.0,
            maximum=None,
            frame="FLU",
        )
        for name in QUADROTOR_CONTROL_NAMES
    )
    return TrajectorySpec(
        state_schema=RIGID_BODY_STATE_SCHEMA,
        observation_source=BENCHMARK_OBSERVATION_SOURCE,
        controls=controls,
        vehicle=VehicleConfigurationSpec(
            family="multirotor",
            configuration_id=BENCHMARK_CONFIGURATION_ID,
            controlled_axes=("roll", "pitch", "yaw"),
            propulsion="quadrotor",
            fixed_states={
                "mass_kg": 0.045,
                "rotor_layout": "x",
                "rotor_speed_reference_rad_s": rotor_speed_reference_rad_s,
                "control_transform": "(rotor_speed_rad_s / reference_rad_s)^2",
            },
        ),
        observations=specific_force_observation_channels(
            "processed_onboard_accelerometer"
        ),
    )


@dataclass(frozen=True)
class NanoDroneBenchmarkAdapter:
    """Load one pinned benchmark CSV as a canonical Glassbox trajectory."""

    rotor_speed_reference_rad_s: float = ROTOR_SPEED_REFERENCE_RAD_S
    verify_checksum: bool = True
    name: str = "idsia_nanodrone_benchmark"

    def __post_init__(self) -> None:
        if (
            not np.isfinite(self.rotor_speed_reference_rad_s)
            or self.rotor_speed_reference_rad_s <= 0.0
        ):
            raise ValueError("rotor_speed_reference_rad_s must be positive and finite")

    def _read(
        self, path: str | Path
    ) -> tuple[Path, BenchmarkRecording, dict[str, Any], np.ndarray, dict[str, Any], str]:
        source_path = Path(path)
        recording, labels = _recording_identity(source_path)
        checksum = _sha256(source_path)
        if self.verify_checksum and checksum != recording.sha256:
            raise ValueError(
                f"checksum mismatch for pinned benchmark recording {source_path}; "
                f"expected {recording.sha256}, got {checksum}"
            )
        _, data = _read_source_csv(source_path)
        quality = _source_quality(data)
        return source_path, recording, labels, data, quality, checksum

    def inspect(self, path: str | Path) -> dict[str, Any]:
        """Validate and summarize one upstream benchmark CSV."""

        source_path, recording, labels, data, quality, checksum = self._read(path)
        return {
            "source": str(source_path),
            "adapter": {"name": self.name, "schema_version": 2},
            "benchmark": {
                "repository": BENCHMARK_REPOSITORY,
                "commit": BENCHMARK_COMMIT,
                "doi": BENCHMARK_DOI,
                "relative_path": recording.relative_path,
                "split": recording.split,
            },
            "rows": len(data),
            "intervals": len(data) - 1,
            "duration_s": float(data[-1, 0] - data[0, 0]),
            "columns": list(SOURCE_COLUMNS),
            "labels": labels,
            "quality": quality,
            "sha256": checksum,
            "checksum_matches_pinned_snapshot": checksum == recording.sha256,
            "spec": nanodrone_trajectory_spec(
                self.rotor_speed_reference_rad_s
            ).to_dict(),
        }

    def load(self, path: str | Path) -> Trajectory:
        """Convert one upstream benchmark CSV into NWU/FLU, wxyz form."""

        source_path, recording, labels, data, quality, checksum = self._read(path)
        time_s = data[:, 0] - data[0, 0]

        quaternion_xyzw = data[:, 4:8].copy()
        quaternion_xyzw /= np.linalg.norm(
            quaternion_xyzw, axis=1, keepdims=True
        )
        sign_flips = 0
        for index in range(1, len(quaternion_xyzw)):
            if np.dot(quaternion_xyzw[index - 1], quaternion_xyzw[index]) < 0.0:
                quaternion_xyzw[index] *= -1.0
                sign_flips += 1

        states = np.empty((len(data), 13), dtype=np.float64)
        states[:, 0:3] = data[:, 1:4]
        states[:, 3:6] = data[:, 8:11]
        states[:, 6] = quaternion_xyzw[:, 3]
        states[:, 7:10] = quaternion_xyzw[:, 0:3]
        states[:, 10:13] = data[:, 11:14]

        source_motor_speeds = data[:-1, 14:18]
        canonical_motor_speeds = source_motor_speeds[
            :, CANONICAL_MOTOR_SOURCE_INDICES
        ]
        controls = (
            canonical_motor_speeds / self.rotor_speed_reference_rad_s
        ) ** 2

        return Trajectory(
            time_s=time_s,
            states=states,
            controls=controls,
            spec=nanodrone_trajectory_spec(self.rotor_speed_reference_rad_s),
            observations=data[:, 18:21],
            labels={
                **labels,
                "benchmark": "idsia_nanodrone_sysid",
                "condition": "aggressive_real_flight",
            },
            provenance={
                "source": str(source_path),
                "source_sha256": checksum,
                "adapter": {"name": self.name, "schema_version": 2},
                "benchmark": {
                    "repository": BENCHMARK_REPOSITORY,
                    "commit": BENCHMARK_COMMIT,
                    "doi": BENCHMARK_DOI,
                    "relative_path": recording.relative_path,
                    "expected_sha256": recording.sha256,
                    "expected_size_bytes": recording.size_bytes,
                },
                "source_schema": {
                    "columns": list(SOURCE_COLUMNS),
                    "state_order": [
                        "position_world_xyz",
                        "quaternion_body_to_world_xyzw",
                        "velocity_world_xyz",
                        "angular_velocity_body_xyz",
                    ],
                    "motor_order": [
                        "front_right",
                        "rear_right",
                        "rear_left",
                        "front_left",
                    ],
                    "world_frame": "z_up_right_handed",
                    "body_frame": "FLU",
                },
                "transformations": [
                    {
                        "type": "state_column_reorder",
                        "target": "position_velocity_quaternion_angular_velocity",
                    },
                    {
                        "type": "quaternion_component_reorder",
                        "source": "xyzw",
                        "target": "wxyz",
                    },
                    {
                        "type": "quaternion_normalization_and_continuity",
                        "sign_flips": sign_flips,
                    },
                    {
                        "type": "motor_column_reorder",
                        "source_indices": list(CANONICAL_MOTOR_SOURCE_INDICES),
                        "target_order": list(QUADROTOR_CONTROL_NAMES),
                    },
                    {
                        "type": "motor_speed_to_squared_ratio",
                        "reference_rad_s": self.rotor_speed_reference_rad_s,
                        "formula": "(rotor_speed_rad_s / reference_rad_s)^2",
                    },
                    {
                        "type": "sample_to_interval_alignment",
                        "rule": "control row i applies from state row i to i+1",
                        "discarded_final_control_rows": 1,
                    },
                    {
                        "type": "observation_retention",
                        "source_columns": [
                            "ax_body",
                            "ay_body",
                            "az_body",
                        ],
                        "target_roles": [
                            "specific_force_x",
                            "specific_force_y",
                            "specific_force_z",
                        ],
                        "sample_alignment": "state_timestamp",
                    },
                ],
                "upstream_preprocessing": {
                    "sample_rate_hz": 100.0,
                    "time_alignment": "source timestamps aligned and uniformly retimed",
                    "motor_acceleration_alignment": "cross_correlation",
                    "filter": "zero_phase_fourth_order_butterworth",
                    "cutoff_hz": {
                        "position": 10.0,
                        "linear_and_angular_velocity": 18.0,
                        "motor_speed": 20.0,
                        "quaternion_log_map": 12.0,
                    },
                },
                "quality": quality,
            },
        )


def fetch_nanodrone_benchmark(
    destination: str | Path,
    *,
    overwrite: bool = False,
    timeout_s: float = 60.0,
) -> tuple[Path, ...]:
    """Download and verify the pinned 15-recording benchmark snapshot."""

    if timeout_s <= 0.0:
        raise ValueError("timeout_s must be positive")
    destination_root = Path(destination)
    fetched: list[Path] = []
    for recording in BENCHMARK_RECORDINGS:
        target = destination_root / recording.relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if _sha256(target) == recording.sha256:
                fetched.append(target)
                continue
            if not overwrite:
                raise FileExistsError(
                    f"existing file does not match pinned benchmark: {target}"
                )

        request = urllib.request.Request(
            f"{BENCHMARK_MEDIA_ROOT}/{recording.relative_path}",
            headers={"User-Agent": "glassbox-nanodrone-adapter/1"},
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


def extract_nanodrone_benchmark(
    source_root: str | Path,
    output_root: str | Path,
    *,
    adapter: NanoDroneBenchmarkAdapter | None = None,
) -> tuple[Path, ...]:
    """Convert all pinned benchmark recordings while preserving its split."""

    source_directory = Path(source_root)
    output_directory = Path(output_root)
    selected_adapter = NanoDroneBenchmarkAdapter() if adapter is None else adapter
    outputs: list[Path] = []
    for recording in BENCHMARK_RECORDINGS:
        source_path = source_directory / recording.relative_path
        output_path = (
            output_directory / recording.split / Path(recording.filename).with_suffix(".npz")
        )
        save_trajectory_npz(selected_adapter.load(source_path), output_path)
        outputs.append(output_path)
    return tuple(outputs)
