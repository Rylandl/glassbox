"""Pinned Skywalker X8 system-identification campaign adapter."""

from __future__ import annotations

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
    ExogenousChannel,
    RIGID_BODY_STATE_SCHEMA,
    Trajectory,
    TrajectorySpec,
    VehicleConfigurationSpec,
    save_trajectory_npz,
)


X8_REFERENCE_NAME = "ntnu_skywalker_x8_system_identification"
X8_REFERENCE_DOI = "10.18710/U4TLYV"
X8_REFERENCE_URL = "https://doi.org/10.18710/U4TLYV"
X8_REFERENCE_VERSION = "1.0"
X8_REFERENCE_LICENSE = "CC0-1.0"
X8_CONFIGURATION_ID = "ntnu_skywalker_x8_flying_wing"
X8_SAMPLE_RATE_HZ = 40.0
X8_COLUMN_COUNT = 41
X8_README_FILE_ID = 233052
X8_README_FILENAME = "00_README.txt"
X8_README_SIZE_BYTES = 9440
X8_README_MD5 = "4bc6c339fe46ad2ecac8d2d17d5eaa19"
_DATAVERSE_ACCESS_ROOT = "https://dataverse.no/api/access/datafile"


@dataclass(frozen=True)
class X8Recording:
    """One pinned maneuver from the Dataverse release."""

    filename: str
    split: str
    file_id: int
    size_bytes: int
    md5: str

    @property
    def profile(self) -> str:
        return re.sub(r"_\d+$", "", Path(self.filename).stem)

    @property
    def replicate(self) -> int:
        return int(Path(self.filename).stem.rsplit("_", 1)[1])

    @property
    def relative_path(self) -> str:
        return f"{self.split}/{self.filename}"


X8_RECORDINGS = (
    X8Recording("lateral_121_1.csv", "training", 233075, 276654, "d163a59d4da594f9c716990151159b3c"),
    X8Recording("lateral_121_2.csv", "training", 233078, 270694, "091298ff063747e010298ea3e61451be"),
    X8Recording("lateral_121_3.csv", "training", 233064, 269892, "92bdb4c97d16e05a0c0c4da868d72690"),
    X8Recording("lateral_doublet_1.csv", "training", 233067, 260832, "a120940bc294cff231b9dd4d24577d34"),
    X8Recording("lateral_doublet_2.csv", "training", 233073, 259119, "cec7f99412becc59ba5d2b6d36214912"),
    X8Recording("lateral_doublet_3.csv", "training", 233054, 258177, "338a13a232717c5f397fe14c53399242"),
    X8Recording("longitudinal_3211_1.csv", "training", 233065, 296095, "655edda35560ca299dac545ff7835dab"),
    X8Recording("longitudinal_3211_2.csv", "training", 233074, 294245, "caca49f8743caf59cc26bf756c914554"),
    X8Recording("longitudinal_3211_3.csv", "training", 233077, 293790, "f7fe10687cd6dac07a0d0387150fac9e"),
    X8Recording("longitudinal_3211_4.csv", "training", 233072, 292137, "52bff9a02a5df53b0f4a5b41625ff47f"),
    X8Recording("longitudinal_doublet_1.csv", "training", 233059, 327648, "9e9b1bd40bec64a96a30cfe44bf385c3"),
    X8Recording("longitudinal_doublet_2.csv", "training", 233082, 325829, "03ad563a6e491674c2b763b99a0ee63e"),
    X8Recording("longitudinal_doublet_3.csv", "training", 233068, 256717, "01f4a2fe2fa98937a090c79e117aba3e"),
    X8Recording("lateral_121_4.csv", "validation", 233061, 269112, "e9c734f9f7133d79fa568f6bc200d8f5"),
    X8Recording("lateral_doublet_4.csv", "validation", 233069, 261214, "2d2169e713e65f23e373db75b5122fe1"),
    X8Recording("longitudinal_3211_5.csv", "validation", 233056, 293225, "0d87838dc9de2e75e7a8db8beca0d3c8"),
    X8Recording("longitudinal_doublet_4.csv", "validation", 233084, 327017, "2920b5241fef83408fb1e48955372f2f"),
)

_RECORDING_BY_FILENAME = {
    recording.filename: recording for recording in X8_RECORDINGS
}


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _recording_for_path(path: Path) -> X8Recording:
    try:
        return _RECORDING_BY_FILENAME[path.name]
    except KeyError as error:
        raise ValueError(f"unrecognized Skywalker X8 filename {path.name!r}") from error


def _read_csv(path: Path) -> np.ndarray:
    data = np.loadtxt(path, delimiter=",", dtype=np.float64)
    if data.ndim == 1:
        data = data[None, :]
    if data.ndim != 2 or data.shape[1] != X8_COLUMN_COUNT:
        raise ValueError(
            f"Skywalker X8 CSV must contain {X8_COLUMN_COUNT} columns; "
            f"got {data.shape}"
        )
    if len(data) < 2:
        raise ValueError("Skywalker X8 maneuver needs at least two samples")
    if not np.all(np.isfinite(data)):
        raise ValueError("Skywalker X8 CSV contains non-finite values")
    return data


def _rotation_body_to_ned(phi: np.ndarray, theta: np.ndarray, psi: np.ndarray) -> np.ndarray:
    """Return vectorized aerospace 3-2-1 body-FRD to world-NED rotations."""

    cphi, sphi = np.cos(phi), np.sin(phi)
    ctheta, stheta = np.cos(theta), np.sin(theta)
    cpsi, spsi = np.cos(psi), np.sin(psi)
    rotation = np.empty((len(phi), 3, 3), dtype=np.float64)
    rotation[:, 0, 0] = cpsi * ctheta
    rotation[:, 0, 1] = cpsi * stheta * sphi - spsi * cphi
    rotation[:, 0, 2] = cpsi * stheta * cphi + spsi * sphi
    rotation[:, 1, 0] = spsi * ctheta
    rotation[:, 1, 1] = spsi * stheta * sphi + cpsi * cphi
    rotation[:, 1, 2] = spsi * stheta * cphi - cpsi * sphi
    rotation[:, 2, 0] = -stheta
    rotation[:, 2, 1] = ctheta * sphi
    rotation[:, 2, 2] = ctheta * cphi
    return rotation


def _euler_nwu_quaternion_wxyz(
    phi: np.ndarray, theta: np.ndarray, psi: np.ndarray
) -> np.ndarray:
    """Convert NED/FRD Euler angles to body-FLU -> world-NWU quaternions."""

    roll = phi
    pitch = -theta
    yaw = -psi
    cr, sr = np.cos(roll / 2.0), np.sin(roll / 2.0)
    cp, sp = np.cos(pitch / 2.0), np.sin(pitch / 2.0)
    cy, sy = np.cos(yaw / 2.0), np.sin(yaw / 2.0)
    quaternion = np.column_stack(
        (
            cy * cp * cr + sy * sp * sr,
            cy * cp * sr - sy * sp * cr,
            cy * sp * cr + sy * cp * sr,
            sy * cp * cr - cy * sp * sr,
        )
    )
    quaternion /= np.linalg.norm(quaternion, axis=1, keepdims=True)
    for index in range(1, len(quaternion)):
        if np.dot(quaternion[index - 1], quaternion[index]) < 0.0:
            quaternion[index] *= -1.0
    return quaternion


def _quality(data: np.ndarray) -> dict[str, Any]:
    time_s = data[:, 0]
    time_step_s = np.diff(time_s)
    nominal_dt_s = float(np.median(time_step_s))
    if np.any(time_step_s <= 0.0):
        raise ValueError("Skywalker X8 timestamps must be strictly increasing")
    maximum_timing_error_s = float(np.max(np.abs(time_step_s - nominal_dt_s)))
    if not np.isclose(nominal_dt_s, 1.0 / X8_SAMPLE_RATE_HZ, atol=1e-9, rtol=0.0):
        raise ValueError(
            f"Skywalker X8 sample interval must be {1.0 / X8_SAMPLE_RATE_HZ:g}s"
        )

    rotation = _rotation_body_to_ned(data[:, 10], data[:, 11], data[:, 12])
    reconstructed_body_velocity = np.einsum(
        "nji,nj->ni", rotation, data[:, 19:22]
    )
    body_velocity_error = reconstructed_body_velocity - data[:, 16:19]
    maximum_body_velocity_error_m_s = float(
        np.max(np.linalg.norm(body_velocity_error, axis=1))
    )
    if maximum_body_velocity_error_m_s > 1e-6:
        raise ValueError(
            "Skywalker X8 Euler, body-velocity, and inertial-velocity fields "
            "are inconsistent"
        )

    gps_position_ned = data[:, 32:35]
    integrated_displacement_ned = np.sum(
        0.5
        * (data[:-1, 19:22] + data[1:, 19:22])
        * time_step_s[:, None],
        axis=0,
    )
    gps_displacement_ned = gps_position_ned[-1] - gps_position_ned[0]
    gps_endpoint_discrepancy_m = float(
        np.linalg.norm(gps_displacement_ned - integrated_displacement_ned)
    )
    gps_stale_fraction = float(
        np.mean(np.linalg.norm(np.diff(gps_position_ned, axis=0), axis=1) < 1e-12)
    )

    return {
        "nominal_dt_s": nominal_dt_s,
        "sample_rate_hz": 1.0 / nominal_dt_s,
        "maximum_timing_error_s": maximum_timing_error_s,
        "maximum_body_velocity_consistency_error_m_s": (
            maximum_body_velocity_error_m_s
        ),
        "gps_position_stale_fraction": gps_stale_fraction,
        "gps_vs_integrated_velocity_endpoint_discrepancy_m": (
            gps_endpoint_discrepancy_m
        ),
        "ground_speed_range_m_s": [
            float(np.min(np.linalg.norm(data[:, 19:22], axis=1))),
            float(np.max(np.linalg.norm(data[:, 19:22], axis=1))),
        ],
        "estimated_wind_ned_mean_m_s": np.mean(data[:, 22:25], axis=0).tolist(),
        "control_ranges": {
            "throttle": [float(np.min(data[:, 3])), float(np.max(data[:, 3]))],
            "aileron_rad": [float(np.min(data[:, 2])), float(np.max(data[:, 2]))],
            "elevator_rad": [float(np.min(data[:, 1])), float(np.max(data[:, 1]))],
        },
    }


def x8_trajectory_spec(*, trusted_wind: bool = True) -> TrajectorySpec:
    """Return the typed three-control flying-wing contract."""

    controls = (
        ControlChannel(
            name="throttle",
            role="throttle",
            semantic="normalized_command",
            unit="1",
            minimum=0.0,
            maximum=1.0,
        ),
        ControlChannel(
            name="aileron",
            role="roll",
            semantic="generalized_surface_angle",
            unit="rad",
            frame="FLU",
        ),
        ControlChannel(
            name="elevator",
            role="pitch",
            semantic="generalized_surface_angle",
            unit="rad",
            frame="FLU",
        ),
    )
    exogenous = (
        tuple(
            ExogenousChannel(
                name=f"estimated_wind_{axis}",
                role=f"wind_{axis}",
                semantic="estimated_wind_velocity",
                unit="m/s",
                frame="NWU",
            )
            for axis in ("north", "west", "up")
        )
        if trusted_wind
        else ()
    )
    return TrajectorySpec(
        state_schema=RIGID_BODY_STATE_SCHEMA,
        observation_source="onboard_estimate",
        controls=controls,
        vehicle=VehicleConfigurationSpec(
            family="fixedwing",
            configuration_id=X8_CONFIGURATION_ID,
            controlled_axes=("roll", "pitch"),
            propulsion="single_propeller",
            fixed_states={
                "airframe_layout": "flying_wing",
                "surface_layout": "left_right_elevon",
                "generalized_surface_coordinates": "roll_pitch",
            },
        ),
        exogenous=exogenous,
    )


@dataclass(frozen=True)
class X8ReferenceAdapter:
    """Load one pinned Skywalker X8 CSV as a canonical trajectory."""

    verify_checksum: bool = True
    use_trusted_wind_estimate: bool = True
    name: str = X8_REFERENCE_NAME

    def _read(
        self, path: str | Path
    ) -> tuple[Path, X8Recording, np.ndarray, dict[str, Any], str]:
        source_path = Path(path)
        recording = _recording_for_path(source_path)
        checksum = _md5(source_path)
        if self.verify_checksum and checksum != recording.md5:
            raise ValueError(
                f"MD5 mismatch for pinned Skywalker X8 recording {source_path}; "
                f"expected {recording.md5}, got {checksum}"
            )
        data = _read_csv(source_path)
        return source_path, recording, data, _quality(data), checksum

    def inspect(self, path: str | Path) -> dict[str, Any]:
        """Validate and inventory one upstream maneuver CSV."""

        source_path, recording, data, quality, checksum = self._read(path)
        return {
            "source": str(source_path),
            "adapter": {"name": self.name, "schema_version": 1},
            "reference": {
                "name": X8_REFERENCE_NAME,
                "doi": X8_REFERENCE_DOI,
                "version": X8_REFERENCE_VERSION,
                "license": X8_REFERENCE_LICENSE,
                "relative_path": recording.relative_path,
            },
            "rows": len(data),
            "intervals": len(data) - 1,
            "duration_s": float(data[-1, 0] - data[0, 0]),
            "labels": {
                "profile": recording.profile,
                "replicate": recording.replicate,
                "benchmark_split": recording.split,
            },
            "quality": quality,
            "md5": checksum,
            "checksum_matches_pinned_snapshot": checksum == recording.md5,
            "spec": x8_trajectory_spec(
                trusted_wind=self.use_trusted_wind_estimate
            ).to_dict(),
        }

    def load(self, path: str | Path) -> Trajectory:
        """Convert one maneuver from NED/FRD Euler form to NWU/FLU wxyz."""

        source_path, recording, data, quality, checksum = self._read(path)
        time_s = data[:, 0] - data[0, 0]
        velocity_nwu = data[:, 19:22].copy()
        velocity_nwu[:, 1:] *= -1.0
        position_nwu = np.zeros_like(velocity_nwu)
        position_nwu[1:] = np.cumsum(
            0.5
            * (velocity_nwu[:-1] + velocity_nwu[1:])
            * np.diff(time_s)[:, None],
            axis=0,
        )
        quaternion_wxyz = _euler_nwu_quaternion_wxyz(
            data[:, 10], data[:, 11], data[:, 12]
        )
        angular_velocity_flu = data[:, 13:16].copy()
        angular_velocity_flu[:, 1:] *= -1.0

        states = np.column_stack(
            (
                position_nwu,
                velocity_nwu,
                quaternion_wxyz,
                angular_velocity_flu,
            )
        )
        controls = data[:-1][:, (3, 2, 1)]
        wind_ned = data[:, 22:25]
        wind_nwu = wind_ned.copy()
        wind_nwu[:, 1:] *= -1.0
        wind_provenance = {
            "source_columns": [
                "EstimatedStreamVelocity_x",
                "EstimatedStreamVelocity_y",
                "EstimatedStreamVelocity_z",
            ],
            "mean_ned_m_s": np.mean(wind_ned, axis=0).tolist(),
            "upstream_method": (
                "horizontal autopilot estimate plus vertical component inferred "
                "from airspeed and validated against an earlier five-hole-probe flight"
            ),
        }

        return Trajectory(
            time_s=time_s,
            states=states,
            controls=controls,
            exogenous=(wind_nwu if self.use_trusted_wind_estimate else None),
            spec=x8_trajectory_spec(
                trusted_wind=self.use_trusted_wind_estimate
            ),
            labels={
                "benchmark": X8_REFERENCE_NAME,
                "benchmark_split": recording.split,
                "profile": recording.profile,
                "replicate": recording.replicate,
                "source_group": Path(recording.filename).stem,
                "vehicle_id": X8_CONFIGURATION_ID,
                "condition": "strong_northwest_wind_real_flight",
            },
            provenance={
                "source": str(source_path),
                "source_md5": checksum,
                "adapter": {"name": self.name, "schema_version": 1},
                "reference": {
                    "doi": X8_REFERENCE_DOI,
                    "url": X8_REFERENCE_URL,
                    "version": X8_REFERENCE_VERSION,
                    "license": X8_REFERENCE_LICENSE,
                    "relative_path": recording.relative_path,
                    "expected_size_bytes": recording.size_bytes,
                    "expected_md5": recording.md5,
                },
                "source_schema": {
                    "column_count": X8_COLUMN_COUNT,
                    "world_frame": "NED",
                    "body_frame": "FRD",
                    "attitude": "aerospace_321_euler_rad",
                    "control_columns": ["elevator_rad", "aileron_rad", "throttle"],
                    "state_sources": {
                        "attitude_velocity_rate": "ArduPilot EKF",
                        "position": "integrated ArduPilot EKF inertial velocity",
                    },
                },
                "transformations": [
                    {
                        "type": "coordinate_frame",
                        "source_world": "NED",
                        "target_world": "NWU",
                        "source_body": "FRD",
                        "target_body": "FLU",
                    },
                    {
                        "type": "attitude_representation",
                        "source": "aerospace_321_euler_rad",
                        "target": "quaternion_wxyz",
                    },
                    {
                        "type": "local_position_reconstruction",
                        "method": "trapezoidal_integration_of_40hz_ekf_inertial_velocity",
                        "reason": "published GPS position is lower-rate upsampled telemetry",
                        "gps_endpoint_discrepancy_m": quality[
                            "gps_vs_integrated_velocity_endpoint_discrepancy_m"
                        ],
                    },
                    {
                        "type": "control_column_reorder",
                        "source": ["elevator", "aileron", "throttle"],
                        "target": ["throttle", "aileron", "elevator"],
                    },
                    {
                        "type": "sample_to_interval_alignment",
                        "rule": "control row i applies from state row i to i+1",
                        "discarded_final_control_rows": 1,
                    },
                ],
                (
                    "exogenous"
                    if self.use_trusted_wind_estimate
                    else "excluded_exogenous"
                ): {
                    **wind_provenance,
                    "prediction_policy": (
                        "sample_at_rollout_initialization_and_hold"
                        if self.use_trusted_wind_estimate
                        else "not_supplied_to_model"
                    ),
                    "reason": (
                        "explicit trusted-wind experiment"
                        if self.use_trusted_wind_estimate
                        else (
                            "retained for audit until matched validation shows that "
                            "treating the derived estimate as physics is beneficial"
                        )
                    ),
                },
                "upstream_preprocessing": {
                    "sample_rate_hz": X8_SAMPLE_RATE_HZ,
                    "synchronization": "manual_three_source_alignment",
                    "imu_downsampled_from_hz": 200.0,
                    "gps_and_propulsion_upsampled_from_hz": 10.0,
                },
                "quality": quality,
            },
        )


def _fetch_one(
    *,
    file_id: int,
    target: Path,
    size_bytes: int,
    md5: str,
    overwrite: bool,
    timeout_s: float,
) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.stat().st_size == size_bytes and _md5(target) == md5:
            return target
        if not overwrite:
            raise FileExistsError(
                f"existing file does not match pinned Skywalker X8 source: {target}"
            )

    request = urllib.request.Request(
        f"{_DATAVERSE_ACCESS_ROOT}/{file_id}",
        headers={"User-Agent": "glassbox-skywalker-x8-adapter/1"},
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
        if temporary_path.stat().st_size != size_bytes:
            raise ValueError(f"downloaded size mismatch for {target.name}")
        if _md5(temporary_path) != md5:
            raise ValueError(f"downloaded MD5 mismatch for {target.name}")
        os.replace(temporary_path, target)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return target


def fetch_x8_reference(
    destination: str | Path,
    *,
    overwrite: bool = False,
    timeout_s: float = 60.0,
) -> tuple[Path, ...]:
    """Download and verify the README and 17 canonical-source CSVs."""

    if timeout_s <= 0.0:
        raise ValueError("timeout_s must be positive")
    destination_root = Path(destination)
    _fetch_one(
        file_id=X8_README_FILE_ID,
        target=destination_root / X8_README_FILENAME,
        size_bytes=X8_README_SIZE_BYTES,
        md5=X8_README_MD5,
        overwrite=overwrite,
        timeout_s=timeout_s,
    )
    return tuple(
        _fetch_one(
            file_id=recording.file_id,
            target=destination_root / recording.relative_path,
            size_bytes=recording.size_bytes,
            md5=recording.md5,
            overwrite=overwrite,
            timeout_s=timeout_s,
        )
        for recording in X8_RECORDINGS
    )


def extract_x8_reference(
    source_root: str | Path,
    output_root: str | Path,
    *,
    adapter: X8ReferenceAdapter | None = None,
) -> tuple[Path, ...]:
    """Convert all pinned maneuvers while preserving the upstream split."""

    source_directory = Path(source_root)
    output_directory = Path(output_root)
    selected_adapter = X8ReferenceAdapter() if adapter is None else adapter
    outputs: list[Path] = []
    for recording in X8_RECORDINGS:
        source_path = source_directory / recording.relative_path
        output_path = (
            output_directory
            / recording.split
            / Path(recording.filename).with_suffix(".npz")
        )
        save_trajectory_npz(selected_adapter.load(source_path), output_path)
        outputs.append(output_path)
    return tuple(outputs)
