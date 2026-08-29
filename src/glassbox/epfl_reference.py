"""Pinned EPFL TOPOPlane2 fixed-wing flight-data adapter."""

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
from rosbags.highlevel import AnyReader

from glassbox.data import (
    ControlChannel,
    ExogenousChannel,
    RIGID_BODY_STATE_SCHEMA,
    Trajectory,
    TrajectorySpec,
    VehicleConfigurationSpec,
    save_trajectory_npz,
)


EPFL_REFERENCE_NAME = "epfl_vdm_navigation_flight_data"
EPFL_REFERENCE_DOI = "10.5281/zenodo.10337559"
EPFL_REFERENCE_URL = "https://zenodo.org/records/10337559"
EPFL_REFERENCE_VERSION = "v1"
EPFL_REFERENCE_LICENSE = "CC-BY-4.0"
EPFL_SOURCE_COMMIT = "0c5865cfb6f914a8e6cd5f0b6dd3cba02cb459a7"
EPFL_SOURCE_URL = (
    "https://gitlab.epfl.ch/laupre/vdm_c/-/tree/"
    f"{EPFL_SOURCE_COMMIT}"
)
TOPOPLANE_FILENAME = "TOPOPlane2_20221027_STIM14.bag"
TOPOPLANE_SIZE_BYTES = 298_658_823
TOPOPLANE_MD5 = "1050ed20a4ea78a5dbda37b79df0e788"
TOPOPLANE_DOWNLOAD_URL = (
    "https://zenodo.org/api/records/10337559/files/"
    f"{TOPOPLANE_FILENAME}/content"
)
TOPOPLANE_CONFIGURATION_ID = "epfl_topoplane2_conventional_fixedwing"
TOPOPLANE_SAMPLE_RATE_HZ = 5.0

_REQUIRED_TOPICS = ("/GIINAV_POSE", "/cc_tagged", "/airData")
_CONTROL_PWM_MIN = 800.0
_CONTROL_PWM_MAX = 2200.0
_MIN_FLIGHT_SPEED_M_S = 8.0
_MAX_FLIGHT_SPEED_M_S = 30.0
_MIN_THROTTLE_PWM = 1050.0
_BAROMETRIC_CONSISTENCY_TOLERANCE_M = 15.0
_MIN_SEGMENT_DURATION_S = 20.0
_BOUNDARY_MARGIN_S = 2.0


@dataclass(frozen=True)
class _TopoplaneStreams:
    nav_time_s: np.ndarray
    geodetic_deg_m: np.ndarray
    velocity_ned_m_s: np.ndarray
    quaternion_q0q1q2q3: np.ndarray
    published_angular_velocity: np.ndarray
    control_time_s: np.ndarray
    control_pwm: np.ndarray
    air_time_s: np.ndarray
    airspeed_m_s: np.ndarray
    barometric_altitude_m: np.ndarray


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _header_time_s(header: Any) -> float:
    return float(header.stamp.sec) + float(header.stamp.nanosec) * 1e-9


def _validate_time_series(name: str, time_s: np.ndarray) -> None:
    if len(time_s) < 2:
        raise ValueError(f"EPFL {name} topic needs at least two records")
    if not np.all(np.isfinite(time_s)) or np.any(np.diff(time_s) <= 0.0):
        raise ValueError(f"EPFL {name} timestamps must be finite and increasing")


def _keep_last_duplicate_timestamp(values: np.ndarray) -> np.ndarray:
    """Drop startup duplicates while preserving the newest actuator value."""

    if len(values) < 2:
        return values
    keep = np.r_[values[:-1, 0] != values[1:, 0], True]
    return values[keep]


def _read_topoplane_streams(path: Path) -> _TopoplaneStreams:
    nav: list[tuple[float, ...]] = []
    controls: list[tuple[float, ...]] = []
    air: list[tuple[float, ...]] = []
    with AnyReader([path]) as reader:
        by_topic = {connection.topic: connection for connection in reader.connections}
        missing = [topic for topic in _REQUIRED_TOPICS if topic not in by_topic]
        if missing:
            raise ValueError(f"EPFL TOPOPlane2 bag is missing topics {tuple(missing)}")
        selected = [by_topic[topic] for topic in _REQUIRED_TOPICS]
        for connection, _, rawdata in reader.messages(connections=selected):
            message = reader.deserialize(rawdata, connection.msgtype)
            if connection.topic == "/GIINAV_POSE":
                pose = message.pose.pose
                twist = message.twist.twist
                nav.append(
                    (
                        _header_time_s(message.header),
                        float(pose.position.x),
                        float(pose.position.y),
                        float(pose.position.z),
                        float(twist.linear.x),
                        float(twist.linear.y),
                        float(twist.linear.z),
                        float(pose.orientation.x),
                        float(pose.orientation.y),
                        float(pose.orientation.z),
                        float(pose.orientation.w),
                        float(twist.angular.x),
                        float(twist.angular.y),
                        float(twist.angular.z),
                    )
                )
            elif connection.topic == "/cc_tagged":
                if len(message.channels) < 4:
                    raise ValueError("EPFL cc_tagged record has fewer than four channels")
                controls.append(
                    (
                        _header_time_s(message.header),
                        *(float(value) for value in message.channels[:4]),
                    )
                )
            else:
                air.append(
                    (
                        float(message.time),
                        float(message.airSpeed),
                        float(message.baroAltitude),
                    )
                )

    nav_array = np.asarray(nav, dtype=np.float64)
    control_array = _keep_last_duplicate_timestamp(
        np.asarray(controls, dtype=np.float64)
    )
    air_array = np.asarray(air, dtype=np.float64)
    for name, values in (
        ("GIINAV_POSE", nav_array),
        ("cc_tagged", control_array),
        ("airData", air_array),
    ):
        if values.ndim != 2 or not np.all(np.isfinite(values)):
            raise ValueError(f"EPFL {name} topic contains invalid records")
        _validate_time_series(name, values[:, 0])

    return _TopoplaneStreams(
        nav_time_s=nav_array[:, 0],
        geodetic_deg_m=nav_array[:, 1:4],
        velocity_ned_m_s=nav_array[:, 4:7],
        quaternion_q0q1q2q3=nav_array[:, 7:11],
        published_angular_velocity=nav_array[:, 11:14],
        control_time_s=control_array[:, 0],
        control_pwm=control_array[:, 1:5],
        air_time_s=air_array[:, 0],
        airspeed_m_s=air_array[:, 1],
        barometric_altitude_m=air_array[:, 2],
    )


def topoplane_trajectory_spec() -> TrajectorySpec:
    """Return the typed conventional-airframe input contract."""

    return TrajectorySpec(
        state_schema=RIGID_BODY_STATE_SCHEMA,
        observation_source="offline_ins_gnss_solution",
        controls=(
            ControlChannel(
                name="throttle",
                role="throttle",
                semantic="normalized_actuator_output",
                unit="1",
                minimum=0.0,
                maximum=1.0,
            ),
            ControlChannel(
                name="aileron",
                role="roll",
                semantic="normalized_actuator_output",
                unit="1",
                minimum=-1.0,
                maximum=1.0,
                frame="FLU",
            ),
            ControlChannel(
                name="elevator",
                role="pitch",
                semantic="normalized_actuator_output",
                unit="1",
                minimum=-1.0,
                maximum=1.0,
                frame="FLU",
            ),
            ControlChannel(
                name="rudder",
                role="yaw",
                semantic="normalized_actuator_output",
                unit="1",
                minimum=-1.0,
                maximum=1.0,
                frame="FLU",
            ),
        ),
        vehicle=VehicleConfigurationSpec(
            family="fixedwing",
            configuration_id=TOPOPLANE_CONFIGURATION_ID,
            controlled_axes=("roll", "pitch", "yaw"),
            propulsion="single_propeller",
            fixed_states={
                "airframe_layout": "conventional_tail",
                "surface_layout": "aileron_elevator_rudder",
                "surface_mixing": "independent",
            },
        ),
        exogenous=(
            ExogenousChannel(
                name="pitot_airspeed",
                role="airspeed",
                semantic="measured_pitot_airspeed",
                unit="m/s",
            ),
        ),
    )


def _canonical_quaternion(source: np.ndarray) -> np.ndarray:
    """Convert scalar-first body-FRD -> NED quaternions to NWU/FLU wxyz."""

    quaternion = np.asarray(source, dtype=np.float64).copy()
    norms = np.linalg.norm(quaternion, axis=1, keepdims=True)
    if np.any(norms < 1e-9):
        raise ValueError("EPFL navigation solution contains a null quaternion")
    quaternion /= norms
    quaternion[:, 2:] *= -1.0
    for index in range(1, len(quaternion)):
        if np.dot(quaternion[index - 1], quaternion[index]) < 0.0:
            quaternion[index] *= -1.0
    return quaternion


def _quaternion_product(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.moveaxis(left, -1, 0)
    rw, rx, ry, rz = np.moveaxis(right, -1, 0)
    return np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )


def _angular_velocity_from_quaternion(
    time_s: np.ndarray, quaternion_wxyz: np.ndarray
) -> np.ndarray:
    edge_order = 2 if len(time_s) >= 3 else 1
    derivative = np.gradient(
        quaternion_wxyz,
        time_s,
        axis=0,
        edge_order=edge_order,
    )
    derivative -= quaternion_wxyz * np.sum(
        quaternion_wxyz * derivative, axis=1, keepdims=True
    )
    conjugate = quaternion_wxyz.copy()
    conjugate[:, 1:] *= -1.0
    return 2.0 * _quaternion_product(conjugate, derivative)[:, 1:]


def _geodetic_to_local_nwu(geodetic_deg_m: np.ndarray) -> np.ndarray:
    latitude = np.deg2rad(geodetic_deg_m[:, 0])
    longitude = np.deg2rad(geodetic_deg_m[:, 1])
    altitude = geodetic_deg_m[:, 2]
    semi_major_axis = 6_378_137.0
    eccentricity_squared = 6.69437999014e-3
    radius = semi_major_axis / np.sqrt(
        1.0 - eccentricity_squared * np.sin(latitude) ** 2
    )
    ecef = np.column_stack(
        (
            (radius + altitude) * np.cos(latitude) * np.cos(longitude),
            (radius + altitude) * np.cos(latitude) * np.sin(longitude),
            (radius * (1.0 - eccentricity_squared) + altitude) * np.sin(latitude),
        )
    )
    displacement = ecef - ecef[0]
    latitude_0 = latitude[0]
    longitude_0 = longitude[0]
    north = np.array(
        (
            -np.sin(latitude_0) * np.cos(longitude_0),
            -np.sin(latitude_0) * np.sin(longitude_0),
            np.cos(latitude_0),
        )
    )
    east = np.array((-np.sin(longitude_0), np.cos(longitude_0), 0.0))
    up = np.array(
        (
            np.cos(latitude_0) * np.cos(longitude_0),
            np.cos(latitude_0) * np.sin(longitude_0),
            np.sin(latitude_0),
        )
    )
    return np.column_stack((displacement @ north, -(displacement @ east), displacement @ up))


def _zero_order_hold(
    source_time_s: np.ndarray, source_values: np.ndarray, target_time_s: np.ndarray
) -> np.ndarray:
    indices = np.searchsorted(source_time_s, target_time_s, side="right") - 1
    if np.any(indices < 0) or np.any(indices >= len(source_time_s)):
        raise ValueError("EPFL control stream does not cover navigation timestamps")
    return source_values[indices]


def _normalized_controls(control_pwm: np.ndarray) -> np.ndarray:
    return np.column_stack(
        (
            np.clip((control_pwm[:, 2] - 1000.0) / 1000.0, 0.0, 1.0),
            np.clip((control_pwm[:, 0] - 1500.0) / 500.0, -1.0, 1.0),
            np.clip((control_pwm[:, 1] - 1500.0) / 500.0, -1.0, 1.0),
            np.clip((control_pwm[:, 3] - 1500.0) / 500.0, -1.0, 1.0),
        )
    )


def _dominant_offset(values: np.ndarray) -> float:
    bin_width_m = 5.0
    lower = np.floor(np.min(values) / bin_width_m) * bin_width_m
    upper = np.ceil(np.max(values) / bin_width_m) * bin_width_m + bin_width_m
    edges = np.arange(lower, upper + 0.5 * bin_width_m, bin_width_m)
    counts, _ = np.histogram(values, bins=edges)
    index = int(np.argmax(counts))
    in_mode = (values >= edges[index]) & (values < edges[index + 1])
    return float(np.median(values[in_mode]))


def _close_short_gaps(mask: np.ndarray, *, maximum_gap_samples: int) -> np.ndarray:
    closed = mask.copy()
    edges = np.diff(np.r_[True, mask, True].astype(np.int8))
    starts = np.flatnonzero(edges == -1)
    ends = np.flatnonzero(edges == 1)
    for start, end in zip(starts, ends):
        if start > 0 and end < len(mask) and end - start <= maximum_gap_samples:
            closed[start:end] = True
    return closed


def _healthy_ranges(
    streams: _TopoplaneStreams,
) -> tuple[list[tuple[int, int]], dict[str, Any], np.ndarray, np.ndarray]:
    time_s = streams.nav_time_s
    time_step_s = np.diff(time_s)
    nominal_dt_s = float(np.median(time_step_s))
    if not np.isclose(
        nominal_dt_s, 1.0 / TOPOPLANE_SAMPLE_RATE_HZ, atol=1e-6, rtol=0.0
    ):
        raise ValueError("EPFL GIINAV_POSE does not use the documented 5 Hz rate")
    maximum_timing_error_s = float(np.max(np.abs(time_step_s - nominal_dt_s)))
    if maximum_timing_error_s > 1e-4:
        raise ValueError("EPFL GIINAV_POSE timestamps are not uniformly sampled")

    control_pwm = _zero_order_hold(
        streams.control_time_s, streams.control_pwm, time_s
    )
    airspeed = np.interp(time_s, streams.air_time_s, streams.airspeed_m_s)
    barometric_altitude = np.interp(
        time_s, streams.air_time_s, streams.barometric_altitude_m
    )
    ground_speed = np.linalg.norm(streams.velocity_ned_m_s, axis=1)
    plausible_controls = np.all(
        (control_pwm >= _CONTROL_PWM_MIN) & (control_pwm <= _CONTROL_PWM_MAX),
        axis=1,
    )
    basic_flight = (
        (ground_speed >= _MIN_FLIGHT_SPEED_M_S)
        & (ground_speed <= _MAX_FLIGHT_SPEED_M_S)
        & (airspeed >= _MIN_FLIGHT_SPEED_M_S)
        & (airspeed <= _MAX_FLIGHT_SPEED_M_S)
        & (control_pwm[:, 2] >= _MIN_THROTTLE_PWM)
        & plausible_controls
    )
    if np.count_nonzero(basic_flight) < 2:
        raise ValueError("EPFL recording contains no telemetry-complete flight interval")

    altitude_offset = streams.geodetic_deg_m[:, 2] - barometric_altitude
    healthy_offset_m = _dominant_offset(altitude_offset[basic_flight])
    barometrically_consistent = (
        np.abs(altitude_offset - healthy_offset_m)
        <= _BAROMETRIC_CONSISTENCY_TOLERANCE_M
    )
    maximum_gap_samples = int(round(1.0 / nominal_dt_s))
    barometrically_consistent = _close_short_gaps(
        barometrically_consistent, maximum_gap_samples=maximum_gap_samples
    )
    healthy = basic_flight & barometrically_consistent

    margin_samples = int(round(_BOUNDARY_MARGIN_S / nominal_dt_s))
    minimum_samples = int(round(_MIN_SEGMENT_DURATION_S / nominal_dt_s)) + 1
    edges = np.diff(np.r_[False, healthy, False].astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    ranges = []
    for start, end in zip(starts, ends):
        start += margin_samples
        end -= margin_samples
        if end - start >= minimum_samples:
            ranges.append((int(start), int(end)))
    if not ranges:
        raise ValueError("EPFL recording has no navigation-healthy flight segment")

    quality = {
        "sample_rate_hz": 1.0 / nominal_dt_s,
        "maximum_timing_error_s": maximum_timing_error_s,
        "navigation_records": len(time_s),
        "control_records": len(streams.control_time_s),
        "air_data_records": len(streams.air_time_s),
        "navigation_healthy_segment_count": len(ranges),
        "navigation_healthy_duration_s": float(
            sum(time_s[end - 1] - time_s[start] for start, end in ranges)
        ),
        "dominant_giinav_minus_baro_altitude_m": healthy_offset_m,
        "barometric_consistency_tolerance_m": _BAROMETRIC_CONSISTENCY_TOLERANCE_M,
        "published_angular_velocity_max_abs": float(
            np.max(np.abs(streams.published_angular_velocity))
        ),
        "ground_speed_range_m_s": [
            float(np.min(ground_speed[healthy])),
            float(np.max(ground_speed[healthy])),
        ],
        "pitot_airspeed_range_m_s": [
            float(np.min(airspeed[healthy])),
            float(np.max(airspeed[healthy])),
        ],
        "selection_policy": (
            "5hz fused navigation with valid actuator PWM, flight speed, and "
            "GIINAV/barometric relative-altitude consistency; short gaps closed "
            "and two-second boundaries removed"
        ),
    }
    return ranges, quality, control_pwm, airspeed


def _forward_alignment(
    quaternion_wxyz: np.ndarray, velocity_nwu_m_s: np.ndarray
) -> np.ndarray:
    w, x, y, z = np.moveaxis(quaternion_wxyz, -1, 0)
    body_x_nwu = np.column_stack(
        (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y))
    )
    return np.sum(body_x_nwu * velocity_nwu_m_s, axis=1) / np.maximum(
        np.linalg.norm(velocity_nwu_m_s, axis=1), 1e-9
    )


def _build_trajectories(
    streams: _TopoplaneStreams,
    *,
    source_path: Path,
    checksum: str,
) -> tuple[Trajectory, ...]:
    ranges, shared_quality, control_pwm, airspeed = _healthy_ranges(streams)
    all_quaternion = _canonical_quaternion(streams.quaternion_q0q1q2q3)
    all_controls = _normalized_controls(control_pwm)
    velocity_nwu = streams.velocity_ned_m_s.copy()
    velocity_nwu[:, 1:] *= -1.0
    trajectories = []
    for segment, (start, end) in enumerate(ranges, start=1):
        segment_time = streams.nav_time_s[start:end]
        quaternion = all_quaternion[start:end]
        angular_velocity = _angular_velocity_from_quaternion(
            segment_time, quaternion
        )
        position_nwu = _geodetic_to_local_nwu(streams.geodetic_deg_m[start:end])
        states = np.column_stack(
            (
                position_nwu,
                velocity_nwu[start:end],
                quaternion,
                angular_velocity,
            )
        )
        alignment = _forward_alignment(quaternion, velocity_nwu[start:end])
        segment_quality = {
            **shared_quality,
            "segment": segment,
            "source_start_time_s": float(segment_time[0]),
            "source_end_time_s": float(segment_time[-1]),
            "duration_s": float(segment_time[-1] - segment_time[0]),
            "forward_velocity_alignment": {
                "median": float(np.median(alignment)),
                "p05": float(np.quantile(alignment, 0.05)),
            },
            "derived_body_rate_range_rad_s": [
                float(np.min(angular_velocity)),
                float(np.max(angular_velocity)),
            ],
            "control_ranges": {
                name: [float(np.min(values)), float(np.max(values))]
                for name, values in zip(
                    topoplane_trajectory_spec().control_names,
                    all_controls[start : end - 1].T,
                )
            },
        }
        trajectories.append(
            Trajectory(
                time_s=segment_time - segment_time[0],
                states=states,
                controls=all_controls[start : end - 1],
                exogenous=airspeed[start:end, None],
                spec=topoplane_trajectory_spec(),
                labels={
                    "benchmark": EPFL_REFERENCE_NAME,
                    "benchmark_split": "characterization_only",
                    "profile": "published_navigation_test_flight",
                    "condition": "outdoor_real_flight_navigation_healthy",
                    "replicate": segment,
                    "source_group": source_path.stem,
                    "vehicle_id": TOPOPLANE_CONFIGURATION_ID,
                },
                provenance={
                    "source": str(source_path),
                    "source_md5": checksum,
                    "adapter": {
                        "name": EPFLTopoplaneAdapter.name,
                        "schema_version": 1,
                    },
                    "reference": {
                        "name": EPFL_REFERENCE_NAME,
                        "doi": EPFL_REFERENCE_DOI,
                        "url": EPFL_REFERENCE_URL,
                        "version": EPFL_REFERENCE_VERSION,
                        "license": EPFL_REFERENCE_LICENSE,
                        "relative_path": TOPOPLANE_FILENAME,
                        "expected_size_bytes": TOPOPLANE_SIZE_BYTES,
                        "expected_md5": TOPOPLANE_MD5,
                    },
                    "source_implementation": {
                        "url": EPFL_SOURCE_URL,
                        "commit": EPFL_SOURCE_COMMIT,
                        "control_mapping": (
                            "channels 0..3 are aileron, elevator, throttle/RPM, rudder"
                        ),
                        "quaternion_storage": (
                            "scalar-first q0,q1,q2,q3 stored in ROS orientation x,y,z,w"
                        ),
                    },
                    "source_schema": {
                        "world_frame": "NED",
                        "body_frame": "FRD",
                        "geodetic_position": "WGS84 latitude/longitude degrees and altitude metres",
                        "navigation_rate_hz": TOPOPLANE_SAMPLE_RATE_HZ,
                        "control_source": "GNSS-tagged autopilot PWM output",
                        "control_channel_order": [
                            "aileron",
                            "elevator",
                            "throttle_rpm",
                            "rudder",
                        ],
                    },
                    "transformations": [
                        {
                            "type": "navigation_health_selection",
                            "method": shared_quality["selection_policy"],
                            "reason": (
                                "the flight intentionally includes navigation outages; "
                                "drifting fused position is not dynamics ground truth"
                            ),
                        },
                        {
                            "type": "geodetic_to_local_position",
                            "method": "WGS84 ECEF to segment-local NWU tangent frame",
                        },
                        {
                            "type": "coordinate_frame",
                            "source_world": "NED",
                            "target_world": "NWU",
                            "source_body": "FRD",
                            "target_body": "FLU",
                        },
                        {
                            "type": "quaternion_reorder",
                            "source": "q0_q1_q2_q3_scalar_first_in_xyzw_fields",
                            "target": "wxyz",
                        },
                        {
                            "type": "angular_velocity_reconstruction",
                            "method": "central quaternion derivative in the body frame",
                            "reason": (
                                "the published GIINAV_POSE angular field is identically zero"
                            ),
                        },
                        {
                            "type": "control_normalization",
                            "source": "autopilot PWM outputs",
                            "target_order": [
                                "throttle",
                                "aileron",
                                "elevator",
                                "rudder",
                            ],
                        },
                        {
                            "type": "sample_to_interval_alignment",
                            "rule": "latest GNSS-tagged PWM at state i applies to interval i",
                        },
                    ],
                    "exogenous": {
                        "source_topic": "/airData",
                        "channel": "airSpeed",
                        "prediction_policy": "sample_at_rollout_initialization_and_hold",
                    },
                    "evaluation_limitations": {
                        "single_flight": True,
                        "independent_validation_split": False,
                        "status": "characterization_only_not_promotion_evidence",
                    },
                    "quality": segment_quality,
                },
            )
        )
    return tuple(trajectories)


@dataclass(frozen=True)
class EPFLTopoplaneAdapter:
    """Convert the pinned TOPOPlane2 ROS1 bag into healthy flight segments."""

    verify_checksum: bool = True
    name: str = "epfl_topoplane2_rosbag"

    def _identify(self, path: str | Path) -> tuple[Path, str]:
        source_path = Path(path)
        if source_path.name != TOPOPLANE_FILENAME:
            raise ValueError(
                f"unrecognized EPFL recording {source_path.name!r}; "
                f"expected {TOPOPLANE_FILENAME!r}"
            )
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        if self.verify_checksum and source_path.stat().st_size != TOPOPLANE_SIZE_BYTES:
            raise ValueError(
                f"size mismatch for pinned EPFL recording {source_path}; "
                f"expected {TOPOPLANE_SIZE_BYTES}, got {source_path.stat().st_size}"
            )
        checksum = _md5(source_path)
        if self.verify_checksum and checksum != TOPOPLANE_MD5:
            raise ValueError(
                f"MD5 mismatch for pinned EPFL recording {source_path}; "
                f"expected {TOPOPLANE_MD5}, got {checksum}"
            )
        return source_path, checksum

    def inspect(self, path: str | Path) -> dict[str, Any]:
        """Validate the bag and report dynamics-grade usable coverage."""

        source_path, checksum = self._identify(path)
        streams = _read_topoplane_streams(source_path)
        ranges, quality, _, _ = _healthy_ranges(streams)
        return {
            "source": str(source_path),
            "adapter": {"name": self.name, "schema_version": 1},
            "reference": {
                "name": EPFL_REFERENCE_NAME,
                "doi": EPFL_REFERENCE_DOI,
                "version": EPFL_REFERENCE_VERSION,
                "license": EPFL_REFERENCE_LICENSE,
            },
            "md5": checksum,
            "checksum_matches_pinned_snapshot": checksum == TOPOPLANE_MD5,
            "dynamics_ready": True,
            "segments": [
                {
                    "start_time_s": float(streams.nav_time_s[start]),
                    "end_time_s": float(streams.nav_time_s[end - 1]),
                    "duration_s": float(
                        streams.nav_time_s[end - 1] - streams.nav_time_s[start]
                    ),
                }
                for start, end in ranges
            ],
            "quality": quality,
            "spec": topoplane_trajectory_spec().to_dict(),
        }

    def load_all(self, path: str | Path) -> tuple[Trajectory, ...]:
        """Return every navigation-healthy, telemetry-complete flight segment."""

        source_path, checksum = self._identify(path)
        return _build_trajectories(
            _read_topoplane_streams(source_path),
            source_path=source_path,
            checksum=checksum,
        )

    def load(self, path: str | Path) -> Trajectory:
        """Return the longest healthy segment for the minimal adapter protocol."""

        trajectories = self.load_all(path)
        return max(trajectories, key=lambda trajectory: trajectory.time_s[-1])


def fetch_epfl_topoplane_reference(
    destination: str | Path,
    *,
    overwrite: bool = False,
    timeout_s: float = 600.0,
) -> Path:
    """Download and verify the pinned TOPOPlane2 bag."""

    if timeout_s <= 0.0:
        raise ValueError("timeout_s must be positive")
    target = Path(destination) / TOPOPLANE_FILENAME
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.stat().st_size == TOPOPLANE_SIZE_BYTES and _md5(target) == TOPOPLANE_MD5:
            return target
        if not overwrite:
            raise FileExistsError(
                f"existing file does not match pinned EPFL reference: {target}"
            )

    request = urllib.request.Request(
        TOPOPLANE_DOWNLOAD_URL,
        headers={"User-Agent": "glassbox-epfl-topoplane-adapter/1"},
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
        if temporary_path.stat().st_size != TOPOPLANE_SIZE_BYTES:
            raise ValueError("downloaded size mismatch for EPFL TOPOPlane2 bag")
        if _md5(temporary_path) != TOPOPLANE_MD5:
            raise ValueError("downloaded MD5 mismatch for EPFL TOPOPlane2 bag")
        os.replace(temporary_path, target)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return target


def extract_epfl_topoplane_reference(
    source: str | Path,
    output_directory: str | Path,
    *,
    adapter: EPFLTopoplaneAdapter | None = None,
) -> tuple[Path, ...]:
    """Convert all healthy segments to canonical trajectory artifacts."""

    selected_adapter = EPFLTopoplaneAdapter() if adapter is None else adapter
    output_root = Path(output_directory)
    outputs = []
    for index, trajectory in enumerate(selected_adapter.load_all(source), start=1):
        output = output_root / f"topoplane2_segment_{index:02d}.npz"
        save_trajectory_npz(trajectory, output)
        outputs.append(output)
    return tuple(outputs)
