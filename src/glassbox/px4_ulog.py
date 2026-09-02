"""PX4 ULog discovery and conversion into canonical Glassbox trajectories."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
import numpy.typing as npt
from pyulog import ULog

from glassbox.data import (
    ExogenousChannel,
    Trajectory,
    angular_acceleration_observation_channels,
    make_trajectory_spec,
    specific_force_observation_channels,
)
from glassbox.dynamics import FIXED_WING_CONTROL_NAMES, QUADROTOR_CONTROL_NAMES
from glassbox.px4_frames import (
    PX4_FRD_TO_FLU_SIGNS,
    frd_to_flu,
    ned_frd_quaternion_to_nwu_flu,
    ned_to_nwu,
)


class ULogDataset(Protocol):
    """The subset of ``pyulog.ULog.Data`` used by the converter."""

    name: str
    multi_id: int
    data: Mapping[str, npt.NDArray[np.generic]]


class PX4ULogError(ValueError):
    """Raised when a ULog cannot produce a valid dynamics trajectory."""


PX4_WIND_TOPIC = "airspeed_wind"
PX4_SPECIFIC_FORCE_TOPIC = "vehicle_acceleration"
PX4_WIND_MAX_GAP_S = 2.5
WIND_EXOGENOUS_CHANNELS = (
    ExogenousChannel(
        name="wind_north_m_s",
        role="estimated_wind_north",
        semantic="estimated_environment_at_prediction_start",
        unit="m/s",
        frame="NWU",
    ),
    ExogenousChannel(
        name="wind_west_m_s",
        role="estimated_wind_west",
        semantic="estimated_environment_at_prediction_start",
        unit="m/s",
        frame="NWU",
    ),
    ExogenousChannel(
        name="wind_north_variance_m2_s2",
        role="estimated_wind_north_variance",
        semantic="estimated_environment_uncertainty_at_prediction_start",
        unit="(m/s)^2",
        frame="NWU",
    ),
    ExogenousChannel(
        name="wind_west_variance_m2_s2",
        role="estimated_wind_west_variance",
        semantic="estimated_environment_uncertainty_at_prediction_start",
        unit="(m/s)^2",
        frame="NWU",
    ),
)


@dataclass(frozen=True)
class PX4IngestConfig:
    """Configuration for converting PX4 topics into one fixed-rate trajectory."""

    platform: Literal["multirotor", "fixedwing"] = "multirotor"
    sample_rate_hz: float = 50.0
    state_source: Literal["estimated", "ground_truth"] = "estimated"
    motor_indices: tuple[int, int, int, int] | None = None
    motor_index: int | None = None
    surface_indices: tuple[int, ...] | None = None
    max_gap_s: float = 0.10
    actuator_hold_max_age_s: float | None = None
    min_duration_s: float = 0.50
    min_height_m: float | None = 0.20
    only_armed: bool = True
    only_in_air: bool = True
    position_topic: str | None = None
    attitude_topic: str | None = None
    angular_velocity_topic: str | None = None
    actuator_topic: str = "actuator_motors"
    actuator_field: str = "control"
    servo_topic: str = "actuator_servos"
    servo_field: str = "control"
    armed_topic: str = "actuator_armed"
    land_topic: str = "vehicle_land_detected"
    profile: str | None = None
    condition: str | None = None
    replicate: int | None = None
    initial_yaw_deg: float | None = None
    vehicle_id: str | None = None

    def __post_init__(self) -> None:
        if self.sample_rate_hz <= 0.0:
            raise ValueError("sample_rate_hz must be positive")
        if self.max_gap_s <= 0.0:
            raise ValueError("max_gap_s must be positive")
        if self.actuator_hold_max_age_s is not None and not (
            np.isfinite(self.actuator_hold_max_age_s)
            and self.actuator_hold_max_age_s > 0.0
        ):
            raise ValueError("actuator_hold_max_age_s must be positive and finite")
        if self.min_duration_s <= 0.0:
            raise ValueError("min_duration_s must be positive")
        if self.min_height_m is not None and self.min_height_m < 0.0:
            raise ValueError("min_height_m cannot be negative")
        if self.motor_indices is not None and (
            len(set(self.motor_indices)) != 4 or min(self.motor_indices) < 0
        ):
            raise ValueError(
                "motor_indices must contain four distinct nonnegative indices"
            )
        if self.motor_index is not None and self.motor_index < 0:
            raise ValueError("motor_index must be nonnegative")
        if self.surface_indices is not None and (
            len(self.surface_indices) != 3
            or len(set(self.surface_indices)) != 3
            or min(self.surface_indices) < 0
        ):
            raise ValueError(
                "explicit surface_indices must contain distinct nonnegative "
                "aileron,elevator,rudder indices"
            )
        if self.platform == "multirotor" and (
            self.motor_index is not None or self.surface_indices is not None
        ):
            raise ValueError(
                "motor_index and surface_indices are only valid for fixedwing ingestion"
            )
        if self.platform == "fixedwing" and self.motor_indices is not None:
            raise ValueError("motor_indices is only valid for multirotor ingestion")
        if self.profile is not None and not self.profile.strip():
            raise ValueError("profile cannot be empty")
        if self.condition is not None and not self.condition.strip():
            raise ValueError("condition cannot be empty")
        if self.replicate is not None and self.replicate < 1:
            raise ValueError("replicate must be positive")
        if self.initial_yaw_deg is not None and not np.isfinite(self.initial_yaw_deg):
            raise ValueError("initial_yaw_deg must be finite")
        if self.vehicle_id is not None and not self.vehicle_id.strip():
            raise ValueError("vehicle_id cannot be empty")

    def resolved_topics(self) -> dict[str, str]:
        suffix = "_groundtruth" if self.state_source == "ground_truth" else ""
        topics = {
            "position": self.position_topic or f"vehicle_local_position{suffix}",
            "attitude": self.attitude_topic or f"vehicle_attitude{suffix}",
            "angular_velocity": self.angular_velocity_topic
            or f"vehicle_angular_velocity{suffix}",
        }
        if self.platform == "fixedwing":
            topics["motor_actuator"] = self.actuator_topic
            topics["servo_actuator"] = self.servo_topic
        else:
            topics["actuator"] = self.actuator_topic
        if self.only_armed:
            topics["armed"] = self.armed_topic
        if self.only_in_air:
            topics["land"] = self.land_topic
        return topics


def _resolve_motor_indices(
    configured: tuple[int, int, int, int] | None,
    parameters: Mapping[str, object] | None,
) -> tuple[tuple[int, int, int, int], str]:
    if configured is not None:
        return configured, "explicit"
    if parameters is None:
        raise PX4ULogError(
            "motor order was not provided and PX4 parameters are unavailable; "
            "set motor_indices in front-left, front-right, rear-right, rear-left order"
        )

    rotor_count = int(parameters.get("CA_ROTOR_COUNT", 0))
    if rotor_count != 4:
        raise PX4ULogError(
            "automatic motor ordering currently requires CA_ROTOR_COUNT=4; "
            f"log reports {rotor_count}"
        )

    by_quadrant: dict[tuple[int, int], int] = {}
    for index in range(rotor_count):
        try:
            x = float(parameters[f"CA_ROTOR{index}_PX"])
            y = float(parameters[f"CA_ROTOR{index}_PY"])
        except KeyError as error:
            raise PX4ULogError(
                f"automatic motor ordering needs CA_ROTOR{index}_PX and _PY"
            ) from error
        if abs(x) < 1e-6 or abs(y) < 1e-6:
            raise PX4ULogError(
                "automatic motor ordering requires one rotor in each X-frame quadrant"
            )
        quadrant = (1 if x > 0.0 else -1, 1 if y > 0.0 else -1)
        if quadrant in by_quadrant:
            raise PX4ULogError("multiple rotors occupy the same position quadrant")
        by_quadrant[quadrant] = index

    # PX4 body coordinates are FRD (Y right). Glassbox's canonical motor order
    # is front-left, front-right, rear-right, rear-left.
    quadrants = ((1, -1), (1, 1), (-1, 1), (-1, -1))
    try:
        return tuple(by_quadrant[item] for item in quadrants), "px4_ca_rotor_geometry"  # type: ignore[return-value]
    except KeyError as error:
        raise PX4ULogError(
            "automatic motor ordering requires one rotor in each X-frame quadrant"
        ) from error


def _resolve_fixed_wing_actuators(
    configured_motor_index: int | None,
    configured_surface_indices: tuple[int, ...] | None,
    parameters: Mapping[str, object] | None,
) -> tuple[
    int,
    tuple[int, ...],
    np.ndarray,
    tuple[int, ...],
    tuple[str, ...],
    np.ndarray | None,
    str,
]:
    """Resolve throttle and aerodynamic-axis controls from PX4 allocation params.

    PX4 publishes normalized individual surface positions in ``actuator_servos``.
    Multiplying those positions by the logged control-effectiveness matrix
    reconstructs normalized roll, pitch, and yaw authority. This preserves the
    signs of paired surfaces such as left/right ailerons.
    """

    if configured_motor_index is not None and configured_surface_indices is not None:
        return (
            configured_motor_index,
            configured_surface_indices,
            np.eye(3, dtype=np.float64),
            (),
            ("roll", "pitch", "yaw"),
            None,
            "explicit",
        )
    if parameters is None:
        raise PX4ULogError(
            "fixed-wing actuator mapping was not fully provided and PX4 parameters "
            "are unavailable; set motor_index and surface_indices in "
            "aileron,elevator,rudder order"
        )

    if configured_motor_index is None:
        rotor_count = int(parameters.get("CA_ROTOR_COUNT", 0))
        if rotor_count != 1:
            raise PX4ULogError(
                "automatic fixed-wing throttle mapping requires CA_ROTOR_COUNT=1; "
                f"log reports {rotor_count}"
            )
        motor_index = 0
    else:
        motor_index = configured_motor_index

    if configured_surface_indices is not None:
        return (
            motor_index,
            configured_surface_indices,
            np.eye(3, dtype=np.float64),
            (),
            ("roll", "pitch", "yaw"),
            None,
            "explicit_surfaces",
        )

    surface_count = int(parameters.get("CA_SV_CS_COUNT", 0))
    if surface_count < 2:
        raise PX4ULogError(
            "automatic fixed-wing surface mapping requires at least two PX4 "
            f"control surfaces; log reports {surface_count}"
        )
    surface_indices = tuple(range(surface_count))
    surface_types = tuple(
        int(parameters.get(f"CA_SV_CS{index}_TYPE", 0)) for index in surface_indices
    )
    axes = ("R", "P", "Y")
    effectiveness = np.asarray(
        [
            [
                float(parameters.get(f"CA_SV_CS{index}_TRQ_{axis}", 0.0))
                for index in surface_indices
            ]
            for axis in axes
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(effectiveness)):
        raise PX4ULogError("PX4 CA_SV_CS*_TRQ_R/P/Y parameters must be finite")
    canonical_effectiveness = PX4_FRD_TO_FLU_SIGNS[:, np.newaxis] * effectiveness
    axis_names = ("roll", "pitch", "yaw")
    active_indices = tuple(
        index
        for index in range(3)
        if np.linalg.norm(canonical_effectiveness[index]) > 1e-8
    )
    controlled_axes = tuple(axis_names[index] for index in active_indices)
    if "roll" not in controlled_axes or "pitch" not in controlled_axes:
        raise PX4ULogError(
            "fixed-wing allocation must provide independent roll and pitch "
            f"authority; detected axes={controlled_axes}"
        )
    active_effectiveness = canonical_effectiveness[list(active_indices)]
    if np.linalg.matrix_rank(active_effectiveness, tol=1e-8) < len(active_indices):
        raise PX4ULogError(
            "PX4 control-surface effectiveness does not independently span "
            f"the detected axes {controlled_axes}"
        )
    flap_effectiveness = np.asarray(
        [
            float(parameters.get(f"CA_SV_CS{index}_FLAP", 0.0))
            for index in surface_indices
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(flap_effectiveness)):
        raise PX4ULogError("PX4 CA_SV_CS*_FLAP parameters must be finite")
    if np.linalg.norm(flap_effectiveness) <= 1e-8:
        flap_effectiveness_or_none = None
    else:
        flap_effectiveness_or_none = flap_effectiveness
    return (
        motor_index,
        surface_indices,
        effectiveness,
        surface_types,
        controlled_axes,
        flap_effectiveness_or_none,
        "px4_control_allocation",
    )


def _dataset_map(datasets: Sequence[ULogDataset]) -> dict[tuple[str, int], ULogDataset]:
    return {(dataset.name, dataset.multi_id): dataset for dataset in datasets}


def _required_dataset(
    datasets: Sequence[ULogDataset], name: str, *, multi_id: int = 0
) -> ULogDataset:
    by_name = _dataset_map(datasets)
    try:
        return by_name[(name, multi_id)]
    except KeyError as error:
        available = ", ".join(
            f"{topic}[{instance}]" for topic, instance in sorted(by_name)
        )
        raise PX4ULogError(
            f"required ULog topic {name}[{multi_id}] is missing; "
            f"available topics: {available or 'none'}"
        ) from error


def _sensor_aided_wind_dataset(
    datasets: Sequence[ULogDataset],
) -> tuple[ULogDataset, dict[str, object]] | None:
    """Select the lowest-variance wind estimate that fuses an airspeed sensor."""

    candidates: list[tuple[float, int, ULogDataset, int, float | None]] = []
    required_fields = {
        "timestamp",
        "windspeed_north",
        "windspeed_east",
        "variance_north",
        "variance_east",
        "source",
    }
    for dataset in datasets:
        if dataset.name != PX4_WIND_TOPIC or not required_fields.issubset(dataset.data):
            continue
        source_values = np.asarray(dataset.data["source"], dtype=np.int64)
        source_values = source_values[source_values > 0]
        if len(source_values) < 2:
            continue
        source = int(np.median(source_values))
        source_fraction = float(np.mean(source_values == source))
        if source_fraction < 0.95:
            continue
        variance = np.asarray(
            dataset.data["variance_north"], dtype=np.float64
        ) + np.asarray(dataset.data["variance_east"], dtype=np.float64)
        finite_variance = variance[np.isfinite(variance) & (variance >= 0.0)]
        if len(finite_variance) < 2:
            continue
        scale_median = None
        if "tas_scale_validated" in dataset.data:
            scale = np.asarray(dataset.data["tas_scale_validated"], dtype=np.float64)
            scale = scale[np.isfinite(scale) & (scale > 0.0)]
            if len(scale):
                scale_median = float(np.median(scale))
        candidates.append(
            (
                float(np.median(finite_variance)),
                int(dataset.multi_id),
                dataset,
                source,
                scale_median,
            )
        )
    if not candidates:
        return None
    variance, multi_id, dataset, source, scale_median = min(
        candidates, key=lambda item: (item[0], item[1])
    )
    return dataset, {
        "topic": PX4_WIND_TOPIC,
        "multi_id": multi_id,
        "source": source,
        "selection": "sensor_aided_lowest_median_variance",
        "median_horizontal_variance_m2_s2": variance,
        "median_validated_true_airspeed_scale": scale_median,
        "maximum_interpolation_gap_s": PX4_WIND_MAX_GAP_S,
        "prediction_policy": "sample_at_rollout_start_and_hold",
    }


def _field(dataset: ULogDataset, name: str) -> np.ndarray:
    try:
        return np.asarray(dataset.data[name])
    except KeyError as error:
        raise PX4ULogError(
            f"topic {dataset.name}[{dataset.multi_id}] is missing field {name!r}"
        ) from error


def _array_field(dataset: ULogDataset, name: str, size: int) -> np.ndarray:
    """Read an array field from pyulog's flattened or matrix representation."""

    if name in dataset.data:
        value = np.asarray(dataset.data[name])
        if value.ndim == 2 and value.shape[1] >= size:
            return value[:, :size]

    keys = [f"{name}[{index}]" for index in range(size)]
    missing = [key for key in keys if key not in dataset.data]
    if missing:
        raise PX4ULogError(
            f"topic {dataset.name}[{dataset.multi_id}] is missing fields "
            + ", ".join(missing)
        )
    return np.column_stack([np.asarray(dataset.data[key]) for key in keys])


def _timestamps_s(dataset: ULogDataset, *, sample_time: bool) -> np.ndarray:
    field_name = "timestamp"
    if sample_time and "timestamp_sample" in dataset.data:
        candidate = np.asarray(dataset.data["timestamp_sample"])
        if np.any(candidate > 0):
            field_name = "timestamp_sample"
    return np.asarray(_field(dataset, field_name), dtype=np.float64) * 1e-6


def _prepare_series(
    timestamps_s: np.ndarray,
    values: np.ndarray,
    valid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, np.newaxis]
    if len(timestamps_s) != len(values):
        raise PX4ULogError("timestamps and values have different lengths")

    mask = np.isfinite(timestamps_s) & np.all(np.isfinite(values), axis=1)
    if valid is not None:
        mask &= np.asarray(valid, dtype=bool)
    timestamps_s = np.asarray(timestamps_s[mask], dtype=np.float64)
    values = values[mask]
    if len(timestamps_s) < 2:
        raise PX4ULogError("a signal has fewer than two valid samples")

    order = np.argsort(timestamps_s, kind="stable")
    timestamps_s = timestamps_s[order]
    values = values[order]
    timestamps_s, unique_indices = np.unique(timestamps_s, return_index=True)
    return timestamps_s, values[unique_indices]


def _linear_resample(
    timestamps_s: np.ndarray,
    values: np.ndarray,
    target_s: np.ndarray,
    max_gap_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    result = np.column_stack(
        [
            np.interp(target_s, timestamps_s, values[:, index])
            for index in range(values.shape[1])
        ]
    )
    right = np.searchsorted(timestamps_s, target_s, side="left")
    exact = (right < len(timestamps_s)) & np.isclose(
        timestamps_s[np.minimum(right, len(timestamps_s) - 1)],
        target_s,
        atol=1e-9,
        rtol=0.0,
    )
    left = right - 1
    bracketed = (left >= 0) & (right < len(timestamps_s))
    gaps = np.full(len(target_s), np.inf)
    gaps[bracketed] = timestamps_s[right[bracketed]] - timestamps_s[left[bracketed]]
    valid = exact | (bracketed & (gaps <= max_gap_s + 1e-12))
    return result, valid


def _continuous_quaternions(quaternions: np.ndarray) -> np.ndarray:
    result = np.asarray(quaternions, dtype=np.float64).copy()
    result /= np.linalg.norm(result, axis=1, keepdims=True)
    for index in range(1, len(result)):
        if np.dot(result[index - 1], result[index]) < 0.0:
            result[index] *= -1.0
    return result


def _slerp_resample(
    timestamps_s: np.ndarray,
    quaternions: np.ndarray,
    target_s: np.ndarray,
    max_gap_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    quaternions = _continuous_quaternions(quaternions)
    left = np.searchsorted(timestamps_s, target_s, side="right") - 1
    left = np.clip(left, 0, len(timestamps_s) - 1)
    right = np.minimum(left + 1, len(timestamps_s) - 1)
    interval = timestamps_s[right] - timestamps_s[left]
    alpha = np.divide(
        target_s - timestamps_s[left],
        interval,
        out=np.zeros_like(target_s),
        where=interval > 0.0,
    )

    first = quaternions[left]
    second = quaternions[right]
    dot = np.clip(np.sum(first * second, axis=1), -1.0, 1.0)
    angle = np.arccos(dot)
    sine = np.sin(angle)
    first_weight = np.divide(
        np.sin((1.0 - alpha) * angle),
        sine,
        out=1.0 - alpha,
        where=np.abs(sine) > 1e-8,
    )
    second_weight = np.divide(
        np.sin(alpha * angle),
        sine,
        out=alpha.copy(),
        where=np.abs(sine) > 1e-8,
    )
    result = first_weight[:, np.newaxis] * first + second_weight[:, np.newaxis] * second
    result /= np.linalg.norm(result, axis=1, keepdims=True)

    in_range = (target_s >= timestamps_s[0] - 1e-12) & (
        target_s <= timestamps_s[-1] + 1e-12
    )
    valid = in_range & ((interval <= max_gap_s + 1e-12) | (interval == 0.0))
    return result, valid


def _hold_resample(
    timestamps_s: np.ndarray,
    values: np.ndarray,
    target_s: np.ndarray,
    max_gap_s: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.searchsorted(timestamps_s, target_s, side="right") - 1
    valid_index = indices >= 0
    safe_indices = np.clip(indices, 0, len(timestamps_s) - 1)
    age_s = target_s - timestamps_s[safe_indices]
    valid = valid_index & (age_s >= -1e-12)
    if max_gap_s is not None:
        valid &= age_s <= max_gap_s + 1e-12
    return values[safe_indices], valid


def _sample_rate_stats(timestamps_s: np.ndarray) -> dict[str, float | int]:
    """Median period, maximum gap, and sample count for a prepared time series."""

    diffs = np.diff(timestamps_s)
    return {
        "median_period_s": float(np.median(diffs)),
        "max_gap_s": float(np.max(diffs)),
        "sample_count": len(timestamps_s),
    }


def _resolve_actuator_hold_max_age_s(
    config: PX4IngestConfig,
    actuator_series: Sequence[tuple[np.ndarray, np.ndarray]],
) -> float:
    """Resolve the hold-age tolerance actually used for actuator validity.

    PX4's default logging profile publishes ``actuator_motors``/``actuator_servos``
    around every 100 ms with normal scheduling jitter. Reusing the tighter
    ``max_gap_s`` state-interpolation tolerance for that hold window makes
    ordinary jitter look like a telemetry dropout and needlessly fragments an
    otherwise continuous flight. When the caller has not pinned an explicit
    value, this widens the tolerance to 1.5x the median observed actuator
    sample period (never below ``max_gap_s``), measured from the actuator
    topic(s) actually selected for this log.
    """

    if config.actuator_hold_max_age_s is not None:
        return config.actuator_hold_max_age_s
    resolved = config.max_gap_s
    for series_time, _ in actuator_series:
        median_period_s = float(np.median(np.diff(series_time)))
        resolved = max(resolved, 1.5 * median_period_s)
    return resolved


def _longest_true_run(mask: np.ndarray) -> tuple[int, int]:
    padded = np.concatenate(([False], np.asarray(mask, dtype=bool), [False]))
    changes = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    if not len(starts):
        return 0, 0
    lengths = stops - starts
    selected = int(np.argmax(lengths))
    return int(starts[selected]), int(stops[selected])


def _true_runs(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    padded = np.concatenate(([False], np.asarray(mask, dtype=bool), [False]))
    changes = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return tuple((int(start), int(stop)) for start, stop in zip(starts, stops))


def trajectories_from_datasets(
    datasets: Sequence[ULogDataset],
    *,
    config: PX4IngestConfig | None = None,
    source: str = "ULog",
    parameters: Mapping[str, object] | None = None,
) -> tuple[Trajectory, ...]:
    """Convert every valid PX4 interval into fixed-rate NWU/FLU trajectories."""

    config = PX4IngestConfig() if config is None else config
    topics = config.resolved_topics()
    position_data = _required_dataset(datasets, topics["position"])
    attitude_data = _required_dataset(datasets, topics["attitude"])
    angular_data = _required_dataset(datasets, topics["angular_velocity"])
    dataset_by_name = _dataset_map(datasets)
    specific_force_data = dataset_by_name.get((PX4_SPECIFIC_FORCE_TOPIC, 0))
    armed_data = (
        _required_dataset(datasets, topics["armed"]) if config.only_armed else None
    )
    land_data = (
        _required_dataset(datasets, topics["land"]) if config.only_in_air else None
    )
    wind_selection = None
    wind_metadata = None
    if config.platform == "fixedwing":
        motor_actuator_data = _required_dataset(datasets, topics["motor_actuator"])
        servo_actuator_data = _required_dataset(datasets, topics["servo_actuator"])
        wind_selection = _sensor_aided_wind_dataset(datasets)
        (
            motor_index,
            surface_indices,
            surface_effectiveness,
            surface_types,
            controlled_axes,
            flap_effectiveness,
            actuator_mapping_source,
        ) = _resolve_fixed_wing_actuators(
            config.motor_index, config.surface_indices, parameters
        )
        if wind_selection is not None:
            wind_data, wind_metadata = wind_selection
            wind_time, wind_ned = _prepare_series(
                _timestamps_s(wind_data, sample_time=False),
                np.column_stack(
                    (
                        _field(wind_data, "windspeed_north"),
                        _field(wind_data, "windspeed_east"),
                        _field(wind_data, "variance_north"),
                        _field(wind_data, "variance_east"),
                    )
                ),
            )
        else:
            wind_time = np.asarray([], dtype=np.float64)
            wind_ned = np.empty((0, 4), dtype=np.float64)
            wind_metadata = None
    else:
        actuator_data = _required_dataset(datasets, topics["actuator"])
        motor_indices, motor_order_source = _resolve_motor_indices(
            config.motor_indices, parameters
        )

    position_valid = np.ones(len(_field(position_data, "x")), dtype=bool)
    for field_name in ("xy_valid", "z_valid", "v_xy_valid", "v_z_valid"):
        if field_name in position_data.data:
            position_valid &= np.asarray(position_data.data[field_name], dtype=bool)
    position_ned = np.column_stack(
        [
            _field(position_data, "x"),
            _field(position_data, "y"),
            _field(position_data, "z"),
            _field(position_data, "vx"),
            _field(position_data, "vy"),
            _field(position_data, "vz"),
        ]
    )
    position_time, position_ned = _prepare_series(
        _timestamps_s(position_data, sample_time=True), position_ned, position_valid
    )

    attitude_time, attitude_ned_frd = _prepare_series(
        _timestamps_s(attitude_data, sample_time=True),
        _array_field(attitude_data, "q", 4),
    )
    quaternion_norm = np.linalg.norm(attitude_ned_frd, axis=1)
    valid_quaternion = quaternion_norm > 0.5
    attitude_time, attitude_ned_frd = _prepare_series(
        attitude_time, attitude_ned_frd, valid_quaternion
    )

    angular_sample_time = _timestamps_s(angular_data, sample_time=True)
    angular_time, angular_velocity_frd = _prepare_series(
        angular_sample_time,
        _array_field(angular_data, "xyz", 3),
    )
    observation_series: list[tuple[np.ndarray, np.ndarray]] = []
    observation_channels = []
    observation_topic_metadata: dict[str, object] = {}
    if specific_force_data is not None:
        specific_force_time, specific_force_frd = _prepare_series(
            _timestamps_s(specific_force_data, sample_time=True),
            _array_field(specific_force_data, "xyz", 3),
        )
        observation_series.append((specific_force_time, specific_force_frd))
        observation_channels.extend(
            specific_force_observation_channels(
                "px4_vehicle_acceleration_bias_corrected"
            )
        )
        observation_topic_metadata["specific_force"] = {
            "topic": PX4_SPECIFIC_FORCE_TOPIC,
            "multi_id": 0,
            "timestamp_field": (
                "timestamp_sample"
                if "timestamp_sample" in specific_force_data.data
                else "timestamp"
            ),
            "source_frame": "FRD",
        }
    if "xyz_derivative[0]" in angular_data.data or (
        "xyz_derivative" in angular_data.data
        and np.asarray(angular_data.data["xyz_derivative"]).ndim == 2
    ):
        angular_acceleration_time, angular_acceleration_frd = _prepare_series(
            angular_sample_time,
            _array_field(angular_data, "xyz_derivative", 3),
        )
        observation_series.append((angular_acceleration_time, angular_acceleration_frd))
        observation_channels.extend(
            angular_acceleration_observation_channels(
                "px4_vehicle_angular_velocity_derivative"
            )
        )
        observation_topic_metadata["angular_acceleration"] = {
            "topic": topics["angular_velocity"],
            "multi_id": 0,
            "field": "xyz_derivative",
            "timestamp_field": (
                "timestamp_sample"
                if "timestamp_sample" in angular_data.data
                else "timestamp"
            ),
            "source_frame": "FRD",
        }

    if config.platform == "fixedwing":
        all_motor_controls = _array_field(
            motor_actuator_data, config.actuator_field, motor_index + 1
        )
        motor_time, selected_motor_control = _prepare_series(
            _timestamps_s(motor_actuator_data, sample_time=False),
            all_motor_controls[:, [motor_index]],
        )
        all_surface_controls = _array_field(
            servo_actuator_data, config.servo_field, max(surface_indices) + 1
        )
        surface_time, selected_surface_controls = _prepare_series(
            _timestamps_s(servo_actuator_data, sample_time=False),
            all_surface_controls[:, surface_indices],
        )
        actuator_series = (
            (motor_time, selected_motor_control),
            (surface_time, selected_surface_controls),
        )
    else:
        highest_motor_index = max(motor_indices)
        all_controls = _array_field(
            actuator_data, config.actuator_field, highest_motor_index + 1
        )
        selected_controls = all_controls[:, motor_indices]
        actuator_time, selected_controls = _prepare_series(
            _timestamps_s(actuator_data, sample_time=False), selected_controls
        )
        actuator_series = ((actuator_time, selected_controls),)

    if armed_data is not None:
        armed_time, armed_values = _prepare_series(
            _timestamps_s(armed_data, sample_time=False), _field(armed_data, "armed")
        )
    else:
        armed_time = np.asarray([], dtype=np.float64)
        armed_values = np.empty((0, 1), dtype=np.float64)

    if land_data is not None:
        land_time, land_status = _prepare_series(
            _timestamps_s(land_data, sample_time=False),
            np.column_stack(
                (
                    _field(land_data, "landed"),
                    _field(land_data, "ground_contact"),
                )
            ),
        )
    else:
        land_time = np.asarray([], dtype=np.float64)
        land_status = np.empty((0, 2), dtype=np.float64)

    start_candidates = [
        position_time[0],
        attitude_time[0],
        angular_time[0],
        *(series_time[0] for series_time, _ in actuator_series),
        *(series_time[0] for series_time, _ in observation_series),
    ]
    if armed_data is not None:
        start_candidates.append(armed_time[0])
    if land_data is not None:
        start_candidates.append(land_time[0])
    if wind_selection is not None:
        start_candidates.append(wind_time[0])
    start_s = max(start_candidates)
    stop_s = min(
        position_time[-1],
        attitude_time[-1],
        angular_time[-1],
        *(series_time[-1] for series_time, _ in actuator_series),
        *(series_time[-1] for series_time, _ in observation_series),
        *([wind_time[-1]] if wind_selection is not None else []),
    )
    dt_s = 1.0 / config.sample_rate_hz
    grid_start_s = np.ceil((start_s - 1e-12) / dt_s) * dt_s
    grid_stop_s = np.floor((stop_s + 1e-12) / dt_s) * dt_s
    state_count = round((grid_stop_s - grid_start_s) / dt_s) + 1
    if state_count < 2:
        raise PX4ULogError("the required topics have no overlapping time interval")
    grid_s = grid_start_s + np.arange(state_count, dtype=np.float64) * dt_s

    position_velocity_ned, position_mask = _linear_resample(
        position_time, position_ned, grid_s, config.max_gap_s
    )
    attitude_ned_frd, attitude_mask = _slerp_resample(
        attitude_time, attitude_ned_frd, grid_s, config.max_gap_s
    )
    angular_velocity_frd, angular_mask = _linear_resample(
        angular_time, angular_velocity_frd, grid_s, config.max_gap_s
    )
    resampled_observations = [
        _linear_resample(series_time, values, grid_s, config.max_gap_s)
        for series_time, values in observation_series
    ]
    if resampled_observations:
        observations_frd = np.column_stack(
            [values for values, _ in resampled_observations]
        )
        observation_mask = np.logical_and.reduce(
            [mask for _, mask in resampled_observations]
        )
        observations = observations_frd * np.tile(
            np.asarray([1.0, -1.0, -1.0]),
            len(resampled_observations),
        )
    else:
        observations = np.empty((len(grid_s), 0), dtype=np.float64)
        observation_mask = np.ones(len(grid_s), dtype=bool)
    if wind_selection is not None:
        wind_ned, wind_mask = _linear_resample(
            wind_time, wind_ned, grid_s, PX4_WIND_MAX_GAP_S
        )
        exogenous = wind_ned * np.asarray([1.0, -1.0, 1.0, 1.0])
    else:
        wind_mask = np.ones(len(grid_s), dtype=bool)
        exogenous = np.empty((len(grid_s), 0), dtype=np.float64)
    resolved_actuator_hold_max_age_s = _resolve_actuator_hold_max_age_s(
        config, actuator_series
    )
    resampled_actuators = [
        _hold_resample(
            series_time, series_values, grid_s[:-1], resolved_actuator_hold_max_age_s
        )
        for series_time, series_values in actuator_series
    ]
    if config.platform == "fixedwing":
        resampled_motor, motor_mask = resampled_actuators[0]
        resampled_surfaces, surface_mask = resampled_actuators[1]
        canonical_surface_mixing = (
            PX4_FRD_TO_FLU_SIGNS[:, np.newaxis] * surface_effectiveness
        )
        axis_indices = tuple(
            ("roll", "pitch", "yaw").index(axis) for axis in controlled_axes
        )
        aerodynamic_axes = (
            resampled_surfaces @ canonical_surface_mixing[list(axis_indices)].T
        )
        control_columns = [resampled_motor[:, 0], *aerodynamic_axes.T]
        if flap_effectiveness is not None:
            control_columns.append(resampled_surfaces @ flap_effectiveness)
        controls = np.column_stack(control_columns)
        control_mask = motor_mask & surface_mask
    else:
        controls, control_mask = resampled_actuators[0]
    if armed_data is not None:
        armed_values, armed_mask = _hold_resample(
            armed_time, armed_values, grid_s[:-1], None
        )
        armed_mask &= armed_values[:, 0] > 0.5
    else:
        armed_mask = np.ones(len(grid_s) - 1, dtype=bool)
    if land_data is not None:
        land_status, in_air_mask = _hold_resample(
            land_time, land_status, grid_s[:-1], None
        )
        in_air_mask &= np.all(land_status < 0.5, axis=1)
    else:
        in_air_mask = np.ones(len(grid_s) - 1, dtype=bool)

    # PX4 state is NED/FRD. Glassbox uses right-handed NWU/FLU so thrust is
    # positive body Z. Both frame conversions are 180-degree rotations about X.
    position_velocity_nwu = np.column_stack(
        (
            ned_to_nwu(position_velocity_ned[:, 0:3]),
            ned_to_nwu(position_velocity_ned[:, 3:6]),
        )
    )
    attitude_nwu_flu = ned_frd_quaternion_to_nwu_flu(attitude_ned_frd)
    attitude_nwu_flu = _continuous_quaternions(attitude_nwu_flu)
    angular_velocity_flu = frd_to_flu(angular_velocity_frd)
    states = np.column_stack(
        (
            position_velocity_nwu[:, 0:3],
            position_velocity_nwu[:, 3:6],
            attitude_nwu_flu,
            angular_velocity_flu,
        )
    )

    state_mask = (
        position_mask & attitude_mask & angular_mask & wind_mask & observation_mask
    )
    clearance_mask = (
        np.ones(len(states), dtype=bool)
        if config.min_height_m is None
        else states[:, 2] >= config.min_height_m
    )
    interval_mask = (
        state_mask[:-1]
        & state_mask[1:]
        & clearance_mask[:-1]
        & clearance_mask[1:]
        & control_mask
        & armed_mask
        & in_air_mask
    )
    all_runs = _true_runs(interval_mask)
    valid_runs = tuple(
        (start, stop)
        for start, stop in all_runs
        if (stop - start) * dt_s >= config.min_duration_s - 1e-12
    )
    if not valid_runs:
        longest_start, longest_stop = _longest_true_run(interval_mask)
        longest_duration_s = (longest_stop - longest_start) * dt_s
        raise PX4ULogError(
            "no contiguous valid interval meets the minimum duration: "
            f"best={longest_duration_s:.3f}s, required={config.min_duration_s:.3f}s"
        )

    armed_and_in_air_mask = armed_mask & in_air_mask
    total_armed_in_air_span_s = float(np.count_nonzero(armed_and_in_air_mask)) * dt_s
    valid_segment_count = len(valid_runs)

    source_rates: dict[str, object] = {
        "position": {**_sample_rate_stats(position_time), "method": "linear"},
        "attitude": {**_sample_rate_stats(attitude_time), "method": "slerp"},
        "angular_velocity": {**_sample_rate_stats(angular_time), "method": "linear"},
    }
    if config.platform == "fixedwing":
        source_rates["motor_actuator"] = {
            **_sample_rate_stats(motor_time),
            "method": "hold",
        }
        source_rates["servo_actuator"] = {
            **_sample_rate_stats(surface_time),
            "method": "hold",
        }
    else:
        source_rates["actuator"] = {
            **_sample_rate_stats(actuator_time),
            "method": "hold",
        }
    if armed_data is not None:
        source_rates["armed"] = {**_sample_rate_stats(armed_time), "method": "hold"}
    if land_data is not None:
        source_rates["land"] = {**_sample_rate_stats(land_time), "method": "hold"}
    if wind_selection is not None:
        source_rates["wind"] = {**_sample_rate_stats(wind_time), "method": "linear"}
    if specific_force_data is not None:
        source_rates["specific_force"] = {
            **_sample_rate_stats(specific_force_time),
            "method": "linear",
        }
    if "angular_acceleration" in observation_topic_metadata:
        source_rates["angular_acceleration"] = {
            **_sample_rate_stats(angular_acceleration_time),
            "method": "linear",
        }

    if config.platform == "fixedwing":
        actuator_metadata: dict[str, object] = {
            "actuator_fields": {
                "motor": config.actuator_field,
                "servo": config.servo_field,
            },
            "motor_index": motor_index,
            "surface_indices": list(surface_indices),
            "surface_types": list(surface_types),
            "surface_effectiveness_axes": ["roll", "pitch", "yaw"],
            "surface_effectiveness_matrix": surface_effectiveness.tolist(),
            "surface_effectiveness_frame": "PX4_FRD",
            "surface_axis_signs_frd_to_flu": (PX4_FRD_TO_FLU_SIGNS.tolist()),
            "canonical_surface_mixing_matrix": (canonical_surface_mixing.tolist()),
            "controlled_axes": list(controlled_axes),
            "flap_effectiveness": (
                None if flap_effectiveness is None else flap_effectiveness.tolist()
            ),
            "control_axis_frame": "FLU",
            "actuator_mapping_verified": True,
            "actuator_mapping_source": actuator_mapping_source,
        }
        if controlled_axes == ("roll", "pitch", "yaw"):
            axis_control_names = FIXED_WING_CONTROL_NAMES[1:]
        else:
            axis_control_names = controlled_axes
        control_names = (
            "throttle",
            *axis_control_names,
            *(("flap",) if flap_effectiveness is not None else ()),
        )
    else:
        actuator_metadata = {
            "actuator_field": config.actuator_field,
            "motor_indices": list(motor_indices),
            "motor_order_verified": True,
            "motor_order_source": motor_order_source,
        }
        control_names = QUADROTOR_CONTROL_NAMES

    common_labels = {
        **({"profile": config.profile} if config.profile is not None else {}),
        **({"condition": config.condition} if config.condition is not None else {}),
        **({"replicate": config.replicate} if config.replicate is not None else {}),
        **(
            {"initial_yaw_deg": config.initial_yaw_deg}
            if config.initial_yaw_deg is not None
            else {}
        ),
        **({"vehicle_id": config.vehicle_id} if config.vehicle_id is not None else {}),
    }
    spec = make_trajectory_spec(
        control_names,
        family=config.platform,
        observation_source=config.state_source,
        configuration_id=config.vehicle_id,
        exogenous=(WIND_EXOGENOUS_CHANNELS if wind_selection is not None else ()),
        observations=observation_channels,
    )
    trajectories: list[Trajectory] = []
    for segment_index, (interval_start, interval_stop) in enumerate(valid_runs):
        interval_count = interval_stop - interval_start
        selected_grid = grid_s[interval_start : interval_stop + 1]
        absolute_start_s = float(selected_grid[0])
        labels = {
            **common_labels,
            "source_group": source,
            **({"segment": segment_index + 1} if len(valid_runs) > 1 else {}),
        }
        segment_duration_s = interval_count * dt_s
        selected_segment_coverage = (
            segment_duration_s / total_armed_in_air_span_s
            if total_armed_in_air_span_s > 0.0
            else None
        )
        provenance = {
            "source": source,
            "adapter": {"name": "px4_ulog", "schema_version": 2},
            "px4": {
                "topics": {
                    **topics,
                    **({"wind": PX4_WIND_TOPIC} if wind_selection is not None else {}),
                },
                "actuator_mapping": actuator_metadata,
                "exogenous": wind_metadata,
                "observations": observation_topic_metadata,
                "source_start_time_s": absolute_start_s,
                "valid_interval_index": segment_index + 1,
                "valid_interval_count": len(valid_runs),
                "valid_segment_count": valid_segment_count,
                "selected_segment_coverage": selected_segment_coverage,
                "resolved_actuator_hold_max_age_s": resolved_actuator_hold_max_age_s,
                "source_rates": source_rates,
                "filters": {
                    "only_armed": config.only_armed,
                    "only_in_air": config.only_in_air,
                    "min_height_m": config.min_height_m,
                    "max_gap_s": config.max_gap_s,
                    "actuator_hold_max_age_s": config.actuator_hold_max_age_s,
                    "min_duration_s": config.min_duration_s,
                },
                "discarded_intervals": int(len(interval_mask) - interval_count),
            },
        }
        trajectories.append(
            Trajectory(
                time_s=selected_grid - absolute_start_s,
                states=states[interval_start : interval_stop + 1],
                controls=controls[interval_start:interval_stop],
                spec=spec,
                exogenous=exogenous[interval_start : interval_stop + 1],
                observations=observations[interval_start : interval_stop + 1],
                labels=labels,
                provenance=provenance,
            )
        )
    return tuple(trajectories)


def trajectory_from_datasets(
    datasets: Sequence[ULogDataset],
    *,
    config: PX4IngestConfig | None = None,
    source: str = "ULog",
    parameters: Mapping[str, object] | None = None,
) -> Trajectory:
    """Convert the longest valid PX4 interval into one canonical trajectory."""

    trajectories = trajectories_from_datasets(
        datasets,
        config=config,
        source=source,
        parameters=parameters,
    )
    return max(trajectories, key=lambda trajectory: len(trajectory.controls))


def load_px4_trajectory(
    path: str | Path, *, config: PX4IngestConfig | None = None
) -> Trajectory:
    """Parse a ULog file and convert its required PX4 topics."""

    config = PX4IngestConfig() if config is None else config
    topics = list(config.resolved_topics().values())
    topics.append(PX4_SPECIFIC_FORCE_TOPIC)
    if config.platform == "fixedwing":
        topics.append(PX4_WIND_TOPIC)
    ulog = ULog(str(path), message_name_filter_list=topics)
    return trajectory_from_datasets(
        ulog.data_list,
        config=config,
        source=str(path),
        parameters=ulog.initial_parameters,
    )


def load_px4_trajectories(
    path: str | Path, *, config: PX4IngestConfig | None = None
) -> tuple[Trajectory, ...]:
    """Parse a ULog once and convert every valid contiguous interval."""

    config = PX4IngestConfig() if config is None else config
    topics = list(config.resolved_topics().values())
    topics.append(PX4_SPECIFIC_FORCE_TOPIC)
    if config.platform == "fixedwing":
        topics.append(PX4_WIND_TOPIC)
    ulog = ULog(str(path), message_name_filter_list=topics)
    return trajectories_from_datasets(
        ulog.data_list,
        config=config,
        source=str(path),
        parameters=ulog.initial_parameters,
    )


def inspect_ulog(path: str | Path) -> dict[str, object]:
    """Return a JSON-compatible inventory of every dataset in a ULog."""

    ulog = ULog(str(path))
    topics: list[dict[str, object]] = []
    for dataset in sorted(ulog.data_list, key=lambda item: (item.name, item.multi_id)):
        timestamp = np.asarray(dataset.data.get("timestamp", []), dtype=np.float64)
        topics.append(
            {
                "name": dataset.name,
                "multi_id": int(dataset.multi_id),
                "samples": len(timestamp),
                "fields": sorted(dataset.data.keys()),
                "start_time_s": float(timestamp[0] * 1e-6) if len(timestamp) else None,
                "stop_time_s": float(timestamp[-1] * 1e-6) if len(timestamp) else None,
            }
        )
    return {
        "path": str(path),
        "start_timestamp_s": float(ulog.start_timestamp * 1e-6),
        "last_timestamp_s": float(ulog.last_timestamp * 1e-6),
        "file_corruption": bool(ulog.file_corruption),
        "dropout_count": len(ulog.dropouts),
        "topics": topics,
    }
