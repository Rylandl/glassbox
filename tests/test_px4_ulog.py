import sys
from dataclasses import dataclass

import numpy as np
import pytest

from glassbox.core.data import Trajectory, make_trajectory_spec
from glassbox.io.px4_ulog import (
    PX4IngestConfig,
    PX4ULogError,
    trajectories_from_datasets,
    trajectory_from_datasets,
)


@dataclass
class FakeDataset:
    name: str
    data: dict[str, np.ndarray]
    multi_id: int = 0


def array_fields(name: str, values: np.ndarray) -> dict[str, np.ndarray]:
    return {f"{name}[{index}]": values[:, index] for index in range(values.shape[1])}


def make_datasets(*, disarmed_until_s: float | None = None) -> list[FakeDataset]:
    start_us = 1_000_000

    position_timestamp = start_us + np.arange(11) * 100_000
    position_time = (position_timestamp - start_us) * 1e-6
    position = FakeDataset(
        "vehicle_local_position",
        {
            "timestamp": position_timestamp + 2_000,
            "timestamp_sample": position_timestamp,
            "x": position_time,
            "y": 2.0 * position_time,
            "z": -3.0 * position_time,
            "vx": np.full(11, 1.0),
            "vy": np.full(11, 2.0),
            "vz": np.full(11, -3.0),
            "xy_valid": np.ones(11, dtype=bool),
            "z_valid": np.ones(11, dtype=bool),
            "v_xy_valid": np.ones(11, dtype=bool),
            "v_z_valid": np.ones(11, dtype=bool),
        },
    )

    attitude_timestamp = start_us + np.arange(21) * 50_000
    half_angle = np.pi / 4.0
    attitude_values = np.tile(
        [np.cos(half_angle), 0.0, np.sin(half_angle), 0.0], (21, 1)
    )
    attitude = FakeDataset(
        "vehicle_attitude",
        {
            "timestamp": attitude_timestamp + 1_000,
            "timestamp_sample": attitude_timestamp,
            **array_fields("q", attitude_values),
        },
    )

    angular_timestamp = start_us + np.arange(26) * 40_000
    angular_values = np.tile([0.1, 0.2, 0.3], (26, 1))
    angular_velocity = FakeDataset(
        "vehicle_angular_velocity",
        {
            "timestamp": angular_timestamp + 1_000,
            "timestamp_sample": angular_timestamp,
            **array_fields("xyz", angular_values),
            **array_fields("xyz_derivative", np.tile([0.4, 0.5, 0.6], (26, 1))),
        },
    )

    acceleration_timestamp = start_us + np.arange(21) * 50_000
    acceleration = FakeDataset(
        "vehicle_acceleration",
        {
            "timestamp": acceleration_timestamp + 1_000,
            "timestamp_sample": acceleration_timestamp,
            **array_fields("xyz", np.tile([1.0, 2.0, -9.0], (21, 1))),
        },
    )

    actuator_timestamp = start_us + np.arange(51) * 20_000
    control_values = np.tile(np.arange(1, 13, dtype=float) / 10.0, (51, 1))
    if disarmed_until_s is not None:
        control_values[
            actuator_timestamp < start_us + int(disarmed_until_s * 1e6), 0:4
        ] = np.nan
    actuator = FakeDataset(
        "actuator_motors",
        {
            "timestamp": actuator_timestamp,
            "timestamp_sample": actuator_timestamp - 5_000,
            **array_fields("control", control_values),
        },
    )
    armed_timestamp = start_us + np.arange(11) * 100_000
    armed_values = np.ones(11, dtype=bool)
    if disarmed_until_s is not None:
        armed_values[armed_timestamp < start_us + int(disarmed_until_s * 1e6)] = False
    armed = FakeDataset(
        "actuator_armed",
        {"timestamp": armed_timestamp, "armed": armed_values},
    )
    land = FakeDataset(
        "vehicle_land_detected",
        {
            "timestamp": armed_timestamp,
            "landed": np.zeros(11, dtype=bool),
            "ground_contact": np.zeros(11, dtype=bool),
        },
    )
    return [
        position,
        attitude,
        angular_velocity,
        acceleration,
        actuator,
        armed,
        land,
    ]


def make_fixed_wing_datasets(*, split_ailerons: bool = False) -> list[FakeDataset]:
    datasets = make_datasets()
    actuator = next(item for item in datasets if item.name == "actuator_motors")
    timestamp = actuator.data["timestamp"]
    timestamp_sample = actuator.data["timestamp_sample"]
    if split_ailerons:
        surface_values = np.tile([-0.4, 0.4, -0.2, 0.1, 0.0, 0.0], (51, 1))
    else:
        surface_values = np.tile([0.4, -0.2, 0.1], (51, 1))
    datasets.append(
        FakeDataset(
            "actuator_servos",
            {
                "timestamp": timestamp,
                "timestamp_sample": timestamp_sample,
                **array_fields("control", surface_values),
            },
        )
    )
    return datasets


def add_sensor_aided_wind(datasets: list[FakeDataset]) -> None:
    timestamp = 1_000_000 + np.arange(11) * 100_000
    common = {
        "timestamp": timestamp,
        "timestamp_sample": np.zeros(11, dtype=np.int64),
        "variance_north": np.full(11, 0.08),
        "variance_east": np.full(11, 0.12),
        "tas_scale_validated": np.full(11, 0.95),
    }
    datasets.extend(
        (
            FakeDataset(
                "airspeed_wind",
                {
                    **common,
                    "windspeed_north": np.full(11, 1.0),
                    "windspeed_east": np.full(11, 1.5),
                    "source": np.zeros(11, dtype=np.uint8),
                },
                multi_id=0,
            ),
            FakeDataset(
                "airspeed_wind",
                {
                    **common,
                    "windspeed_north": np.full(11, 2.0),
                    "windspeed_east": np.full(11, 3.0),
                    "source": np.ones(11, dtype=np.uint8),
                },
                multi_id=1,
            ),
        )
    )


def test_fixed_wing_ingest_selects_typed_sensor_aided_wind() -> None:
    datasets = make_fixed_wing_datasets()
    add_sensor_aided_wind(datasets)

    trajectory = trajectory_from_datasets(
        datasets,
        config=PX4IngestConfig(
            platform="fixedwing",
            sample_rate_hz=10.0,
            motor_index=0,
            surface_indices=(0, 1, 2),
            max_gap_s=0.11,
            min_height_m=0.0,
        ),
    )

    assert trajectory.spec.exogenous_roles == (
        "estimated_wind_north",
        "estimated_wind_west",
        "estimated_wind_north_variance",
        "estimated_wind_west_variance",
    )
    np.testing.assert_allclose(
        trajectory.exogenous,
        np.tile([2.0, -3.0, 0.08, 0.12], (11, 1)),
    )
    metadata = trajectory.provenance["px4"]["exogenous"]
    assert metadata["multi_id"] == 1
    assert metadata["source"] == 1
    assert metadata["prediction_policy"] == "sample_at_rollout_start_and_hold"


def make_flying_wing_datasets(*, with_flap: bool = False) -> list[FakeDataset]:
    datasets = make_datasets()
    actuator = next(item for item in datasets if item.name == "actuator_motors")
    columns = [-0.2, 0.6, 0.3] if with_flap else [-0.2, 0.6]
    surface_values = np.tile(columns, (51, 1))
    datasets.append(
        FakeDataset(
            "actuator_servos",
            {
                "timestamp": actuator.data["timestamp"],
                "timestamp_sample": actuator.data["timestamp_sample"],
                **array_fields("control", surface_values),
            },
        )
    )
    return datasets


def test_px4_topics_are_aligned_and_converted_to_nwu_flu() -> None:
    trajectory = trajectory_from_datasets(
        make_datasets(),
        config=PX4IngestConfig(
            sample_rate_hz=10.0,
            motor_indices=(3, 2, 1, 0),
            max_gap_s=0.11,
            min_height_m=0.0,
            profile="lateral_steps",
            condition="high",
            replicate=2,
            initial_yaw_deg=45.0,
        ),
        source="fixture.ulg",
    )

    assert trajectory.states.shape == (11, 13)
    assert trajectory.controls.shape == (10, 4)
    np.testing.assert_allclose(trajectory.time_s, np.arange(11) * 0.1)
    np.testing.assert_allclose(trajectory.states[-1, 0:3], [1.0, -2.0, 3.0])
    np.testing.assert_allclose(
        trajectory.states[:, 3:6], np.tile([1.0, -2.0, 3.0], (11, 1))
    )
    np.testing.assert_allclose(
        trajectory.states[:, 6:10],
        np.tile([np.sqrt(0.5), 0.0, -np.sqrt(0.5), 0.0], (11, 1)),
        atol=1e-7,
    )
    np.testing.assert_allclose(
        trajectory.states[:, 10:13], np.tile([0.1, -0.2, -0.3], (11, 1))
    )
    np.testing.assert_allclose(
        trajectory.controls, np.tile([0.4, 0.3, 0.2, 0.1], (10, 1))
    )
    assert trajectory.spec.observation_roles == (
        "specific_force_x",
        "specific_force_y",
        "specific_force_z",
        "angular_acceleration_x",
        "angular_acceleration_y",
        "angular_acceleration_z",
    )
    np.testing.assert_allclose(
        trajectory.observations,
        np.tile([1.0, -2.0, 9.0, 0.4, -0.5, -0.6], (11, 1)),
    )
    px4 = trajectory.provenance["px4"]
    mapping = px4["actuator_mapping"]
    assert px4["source_start_time_s"] == pytest.approx(1.0)
    assert mapping["motor_order_verified"] is True
    assert mapping["motor_order_source"] == "explicit"
    assert trajectory.labels["profile"] == "lateral_steps"
    assert trajectory.labels["condition"] == "high"
    assert trajectory.labels["replicate"] == 2
    assert trajectory.labels["initial_yaw_deg"] == 45.0
    assert trajectory.labels["source_group"] == "fixture.ulg"
    assert trajectory.control_names == (
        "motor_front_left",
        "motor_front_right",
        "motor_rear_right",
        "motor_rear_left",
    )


def test_disarmed_nan_controls_are_trimmed() -> None:
    trajectory = trajectory_from_datasets(
        make_datasets(disarmed_until_s=0.2),
        config=PX4IngestConfig(
            sample_rate_hz=10.0,
            motor_indices=(0, 1, 2, 3),
            max_gap_s=0.11,
        ),
    )

    assert trajectory.provenance["px4"]["source_start_time_s"] == pytest.approx(1.2)
    assert trajectory.time_s[-1] == pytest.approx(0.8)
    assert np.all(np.isfinite(trajectory.controls))


def test_minimum_height_trims_ground_contact_transients() -> None:
    trajectory = trajectory_from_datasets(
        make_datasets(),
        config=PX4IngestConfig(
            sample_rate_hz=10.0,
            motor_indices=(0, 1, 2, 3),
            max_gap_s=0.11,
            min_height_m=0.5,
        ),
    )

    assert trajectory.provenance["px4"]["source_start_time_s"] == pytest.approx(1.2)
    assert trajectory.states[0, 2] == pytest.approx(0.6)


def test_disabled_height_gate_accepts_an_offset_local_origin() -> None:
    datasets = make_datasets()
    position = next(
        dataset for dataset in datasets if dataset.name == "vehicle_local_position"
    )
    position.data["z"] = position.data["z"] + 10.0

    trajectory = trajectory_from_datasets(
        datasets,
        config=PX4IngestConfig(
            sample_rate_hz=10.0,
            motor_indices=(0, 1, 2, 3),
            max_gap_s=0.11,
            min_height_m=None,
            only_armed=False,
            only_in_air=False,
        ),
    )

    assert trajectory.time_s[-1] == pytest.approx(1.0)
    assert np.all(trajectory.states[:, 2] < 0.0)
    assert trajectory.provenance["px4"]["filters"]["min_height_m"] is None


def test_all_valid_intervals_are_preserved_across_a_telemetry_gap() -> None:
    datasets = make_datasets()
    position = next(
        dataset for dataset in datasets if dataset.name == "vehicle_local_position"
    )
    retained = np.asarray([0, 1, 2, 3, 7, 8, 9, 10])
    position.data = {name: values[retained] for name, values in position.data.items()}

    trajectories = trajectories_from_datasets(
        datasets,
        config=PX4IngestConfig(
            sample_rate_hz=10.0,
            motor_indices=(0, 1, 2, 3),
            max_gap_s=0.11,
            min_duration_s=0.2,
            min_height_m=0.0,
        ),
    )

    assert len(trajectories) == 2
    assert [trajectory.time_s[-1] for trajectory in trajectories] == pytest.approx(
        [0.3, 0.3]
    )
    assert [trajectory.labels["segment"] for trajectory in trajectories] == [1, 2]
    assert [trajectory.labels["source_group"] for trajectory in trajectories] == [
        "ULog",
        "ULog",
    ]
    assert [
        trajectory.provenance["px4"]["source_start_time_s"]
        for trajectory in trajectories
    ] == pytest.approx([1.0, 1.7])


def test_missing_required_topic_reports_inventory() -> None:
    datasets = [
        dataset
        for dataset in make_datasets()
        if dataset.name != "vehicle_angular_velocity"
    ]

    with pytest.raises(
        PX4ULogError,
        match=r"vehicle_angular_velocity.*available topics",
    ):
        trajectory_from_datasets(datasets)


def test_motor_order_is_derived_from_px4_geometry() -> None:
    parameters = {
        "CA_ROTOR_COUNT": 4,
        "CA_ROTOR0_PX": 1.0,
        "CA_ROTOR0_PY": 1.0,
        "CA_ROTOR1_PX": -1.0,
        "CA_ROTOR1_PY": -1.0,
        "CA_ROTOR2_PX": 1.0,
        "CA_ROTOR2_PY": -1.0,
        "CA_ROTOR3_PX": -1.0,
        "CA_ROTOR3_PY": 1.0,
    }

    trajectory = trajectory_from_datasets(
        make_datasets(),
        config=PX4IngestConfig(
            sample_rate_hz=10.0,
            max_gap_s=0.11,
            min_height_m=0.0,
        ),
        parameters=parameters,
    )

    mapping = trajectory.provenance["px4"]["actuator_mapping"]
    assert mapping["motor_indices"] == [2, 0, 3, 1]
    assert mapping["motor_order_source"] == "px4_ca_rotor_geometry"
    np.testing.assert_allclose(
        trajectory.controls, np.tile([0.3, 0.1, 0.4, 0.2], (10, 1))
    )


def test_fixed_wing_topics_are_joined_using_px4_surface_effectiveness() -> None:
    parameters = {
        "CA_ROTOR_COUNT": 1,
        "CA_SV_CS_COUNT": 3,
        "CA_SV_CS0_TYPE": 15,
        "CA_SV_CS0_TRQ_R": 1.0,
        "CA_SV_CS1_TYPE": 3,
        "CA_SV_CS1_TRQ_P": 1.0,
        "CA_SV_CS2_TYPE": 4,
        "CA_SV_CS2_TRQ_Y": 1.0,
    }

    trajectory = trajectory_from_datasets(
        make_fixed_wing_datasets(),
        config=PX4IngestConfig(
            platform="fixedwing",
            sample_rate_hz=10.0,
            max_gap_s=0.11,
            min_height_m=0.0,
            vehicle_id="standard-plane-01",
        ),
        parameters=parameters,
    )

    mapping = trajectory.provenance["px4"]["actuator_mapping"]
    assert trajectory.spec.vehicle.family == "fixedwing"
    assert mapping["motor_index"] == 0
    assert mapping["surface_indices"] == [0, 1, 2]
    assert mapping["surface_types"] == [15, 3, 4]
    assert mapping["actuator_mapping_verified"] is True
    assert mapping["actuator_mapping_source"] == "px4_control_allocation"
    assert trajectory.control_names == ("throttle", "aileron", "elevator", "rudder")
    np.testing.assert_allclose(
        trajectory.controls, np.tile([0.1, 0.4, 0.2, -0.1], (10, 1))
    )
    assert mapping["surface_axis_signs_frd_to_flu"] == [1.0, -1.0, -1.0]
    assert mapping["control_axis_frame"] == "FLU"
    assert trajectory.spec.observation_source == "estimated"
    assert trajectory.spec.control_roles == ("throttle", "roll", "pitch", "yaw")
    assert trajectory.spec.vehicle.configuration_id == "standard-plane-01"
    assert trajectory.labels["vehicle_id"] == "standard-plane-01"
    assert trajectory.provenance["adapter"] == {
        "name": "px4_ulog",
        "schema_version": 2,
    }
    assert trajectory.provenance["px4"]["actuator_mapping"]["surface_indices"] == [
        0,
        1,
        2,
    ]


def test_split_ailerons_are_reconstructed_as_one_signed_roll_control() -> None:
    parameters = {
        "CA_ROTOR_COUNT": 1,
        "CA_SV_CS_COUNT": 6,
        "CA_SV_CS0_TYPE": 1,
        "CA_SV_CS0_TRQ_R": -0.5,
        "CA_SV_CS1_TYPE": 2,
        "CA_SV_CS1_TRQ_R": 0.5,
        "CA_SV_CS2_TYPE": 3,
        "CA_SV_CS2_TRQ_P": 1.0,
        "CA_SV_CS3_TYPE": 4,
        "CA_SV_CS3_TRQ_Y": 1.0,
        "CA_SV_CS4_TYPE": 9,
        "CA_SV_CS5_TYPE": 10,
    }

    trajectory = trajectory_from_datasets(
        make_fixed_wing_datasets(split_ailerons=True),
        config=PX4IngestConfig(
            platform="fixedwing",
            sample_rate_hz=10.0,
            max_gap_s=0.11,
            min_height_m=0.0,
        ),
        parameters=parameters,
    )

    np.testing.assert_allclose(
        trajectory.controls, np.tile([0.1, 0.4, 0.2, -0.1], (10, 1))
    )


def test_flying_wing_elevons_produce_roll_pitch_roles_without_fictional_yaw() -> None:
    parameters = {
        "CA_ROTOR_COUNT": 1,
        "CA_SV_CS_COUNT": 2,
        "CA_SV_CS0_TYPE": 5,
        "CA_SV_CS0_TRQ_R": -0.5,
        "CA_SV_CS0_TRQ_P": 0.5,
        "CA_SV_CS1_TYPE": 6,
        "CA_SV_CS1_TRQ_R": 0.5,
        "CA_SV_CS1_TRQ_P": 0.5,
    }

    trajectory = trajectory_from_datasets(
        make_flying_wing_datasets(),
        config=PX4IngestConfig(
            platform="fixedwing",
            sample_rate_hz=10.0,
            max_gap_s=0.11,
            min_height_m=0.0,
        ),
        parameters=parameters,
    )

    assert trajectory.control_names == ("throttle", "roll", "pitch")
    assert trajectory.spec is not None
    assert trajectory.spec.control_roles == ("throttle", "roll", "pitch")
    assert trajectory.spec.vehicle.controlled_axes == ("roll", "pitch")
    np.testing.assert_allclose(trajectory.controls, np.tile([0.1, 0.4, -0.2], (10, 1)))
    mapping = trajectory.provenance["px4"]["actuator_mapping"]
    assert mapping["surface_types"] == [5, 6]
    assert mapping["controlled_axes"] == ["roll", "pitch"]


def test_px4_flap_effectiveness_adds_a_typed_auxiliary_control() -> None:
    parameters = {
        "CA_ROTOR_COUNT": 1,
        "CA_SV_CS_COUNT": 3,
        "CA_SV_CS0_TYPE": 5,
        "CA_SV_CS0_TRQ_R": -0.5,
        "CA_SV_CS0_TRQ_P": 0.5,
        "CA_SV_CS1_TYPE": 6,
        "CA_SV_CS1_TRQ_R": 0.5,
        "CA_SV_CS1_TRQ_P": 0.5,
        "CA_SV_CS2_TYPE": 9,
        "CA_SV_CS2_FLAP": 1.0,
    }

    trajectory = trajectory_from_datasets(
        make_flying_wing_datasets(with_flap=True),
        config=PX4IngestConfig(
            platform="fixedwing",
            sample_rate_hz=10.0,
            max_gap_s=0.11,
            min_height_m=0.0,
        ),
        parameters=parameters,
    )

    assert trajectory.control_names == ("throttle", "roll", "pitch", "flap")
    assert trajectory.spec is not None
    assert trajectory.spec.control_roles == (
        "throttle",
        "roll",
        "pitch",
        "flap",
    )
    assert trajectory.spec.vehicle.auxiliary_controls == ("flap",)
    np.testing.assert_allclose(
        trajectory.controls, np.tile([0.1, 0.4, -0.2, 0.3], (10, 1))
    )


def make_jittered_actuator_datasets() -> list[FakeDataset]:
    """A ~4s flight with actuator_motors jittering +-10% around a 100ms period.

    Position, attitude, and angular velocity are sampled every 20ms (well
    inside ``max_gap_s``) so only the actuator hold-age tolerance can gate
    validity. The actuator series is phase-shifted from the state grid by a
    few milliseconds, matching how PX4's continuous microsecond timestamps
    actually fall relative to a fixed-rate output grid.
    """

    start_us = 1_000_000
    span_s = 4.0
    fine_count = int(span_s / 0.02) + 1
    fine_timestamp = start_us + np.arange(fine_count) * 20_000

    position = FakeDataset(
        "vehicle_local_position",
        {
            "timestamp": fine_timestamp,
            "timestamp_sample": fine_timestamp,
            "x": np.zeros(fine_count),
            "y": np.zeros(fine_count),
            "z": np.full(fine_count, -5.0),
            "vx": np.zeros(fine_count),
            "vy": np.zeros(fine_count),
            "vz": np.zeros(fine_count),
            "xy_valid": np.ones(fine_count, dtype=bool),
            "z_valid": np.ones(fine_count, dtype=bool),
            "v_xy_valid": np.ones(fine_count, dtype=bool),
            "v_z_valid": np.ones(fine_count, dtype=bool),
        },
    )
    attitude = FakeDataset(
        "vehicle_attitude",
        {
            "timestamp": fine_timestamp,
            "timestamp_sample": fine_timestamp,
            **array_fields("q", np.tile([1.0, 0.0, 0.0, 0.0], (fine_count, 1))),
        },
    )
    angular_velocity = FakeDataset(
        "vehicle_angular_velocity",
        {
            "timestamp": fine_timestamp,
            "timestamp_sample": fine_timestamp,
            **array_fields("xyz", np.tile([0.1, 0.2, 0.3], (fine_count, 1))),
        },
    )

    periods_us = np.tile([90_000, 110_000], 20)
    actuator_timestamp = start_us + 7_000 + np.concatenate(([0], np.cumsum(periods_us)))
    control_values = np.tile(
        np.arange(1, 5, dtype=float) / 10.0, (len(actuator_timestamp), 1)
    )
    actuator = FakeDataset(
        "actuator_motors",
        {
            "timestamp": actuator_timestamp,
            "timestamp_sample": actuator_timestamp,
            **array_fields("control", control_values),
        },
    )

    armed_timestamp = start_us + np.arange(int(span_s / 0.1) + 1) * 100_000
    armed = FakeDataset(
        "actuator_armed",
        {
            "timestamp": armed_timestamp,
            "armed": np.ones(len(armed_timestamp), dtype=bool),
        },
    )
    land = FakeDataset(
        "vehicle_land_detected",
        {
            "timestamp": armed_timestamp,
            "landed": np.zeros(len(armed_timestamp), dtype=bool),
            "ground_contact": np.zeros(len(armed_timestamp), dtype=bool),
        },
    )
    return [position, attitude, angular_velocity, actuator, armed, land]


def test_actuator_hold_age_resolves_from_publish_jitter_by_default() -> None:
    datasets = make_jittered_actuator_datasets()

    trajectories = trajectories_from_datasets(
        datasets,
        config=PX4IngestConfig(motor_indices=(0, 1, 2, 3)),
    )

    assert len(trajectories) == 1
    px4 = trajectories[0].provenance["px4"]
    assert px4["resolved_actuator_hold_max_age_s"] == pytest.approx(0.15, abs=1e-6)
    assert px4["valid_segment_count"] == 1
    assert px4["selected_segment_coverage"] == pytest.approx(1.0)
    assert trajectories[0].time_s[-1] == pytest.approx(3.98)


def test_explicit_actuator_hold_max_age_overrides_automatic_resolution() -> None:
    datasets = make_jittered_actuator_datasets()

    trajectories = trajectories_from_datasets(
        datasets,
        config=PX4IngestConfig(
            motor_indices=(0, 1, 2, 3),
            actuator_hold_max_age_s=0.10,
            min_duration_s=0.15,
        ),
    )

    assert len(trajectories) == 20
    px4 = trajectories[0].provenance["px4"]
    assert px4["resolved_actuator_hold_max_age_s"] == pytest.approx(0.10)
    assert px4["valid_segment_count"] == 20
    for trajectory in trajectories:
        assert trajectory.time_s[-1] == pytest.approx(0.18)


def test_source_rates_reports_period_method_and_sample_count() -> None:
    trajectory = trajectory_from_datasets(
        make_datasets(),
        config=PX4IngestConfig(
            sample_rate_hz=10.0,
            motor_indices=(0, 1, 2, 3),
            max_gap_s=0.11,
            min_height_m=0.0,
        ),
    )

    source_rates = trajectory.provenance["px4"]["source_rates"]

    position_rates = source_rates["position"]
    assert position_rates["method"] == "linear"
    assert position_rates["median_period_s"] == pytest.approx(0.1)
    assert position_rates["sample_count"] == 11

    attitude_rates = source_rates["attitude"]
    assert attitude_rates["method"] == "slerp"
    assert attitude_rates["median_period_s"] == pytest.approx(0.05)
    assert attitude_rates["sample_count"] == 21

    angular_rates = source_rates["angular_velocity"]
    assert angular_rates["method"] == "linear"
    assert angular_rates["median_period_s"] == pytest.approx(0.04)
    assert angular_rates["sample_count"] == 26

    actuator_rates = source_rates["actuator"]
    assert actuator_rates["method"] == "hold"
    assert actuator_rates["median_period_s"] == pytest.approx(0.02)
    assert actuator_rates["sample_count"] == 51


def test_cli_extract_warns_when_coverage_is_incomplete(
    tmp_path, monkeypatch, capsys
) -> None:
    import glassbox.cli.ulog as ulog_cli

    fragmented_trajectory = trajectory_from_datasets(
        make_jittered_actuator_datasets(),
        config=PX4IngestConfig(
            motor_indices=(0, 1, 2, 3),
            actuator_hold_max_age_s=0.10,
            min_duration_s=0.15,
        ),
    )

    def fake_load(path, *, config):
        return fragmented_trajectory

    monkeypatch.setattr(ulog_cli, "load_px4_trajectory", fake_load)
    output_path = tmp_path / "flight.npz"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "glassbox-ulog",
            "extract",
            str(tmp_path / "flight.ulg"),
            str(output_path),
        ],
    )

    ulog_cli.main()

    captured = capsys.readouterr()
    assert "segments: 20 valid" in captured.out
    assert "warning" in captured.err
    assert "--max-gap" in captured.err
    assert "--actuator-hold-max-age" in captured.err
    assert "0.100s" in captured.err


def test_arp_reference_pins_actuator_hold_max_age(tmp_path, monkeypatch) -> None:
    import glassbox.io.arp_reference as arp_module

    source = tmp_path / "log_63_2024-1-8-16-37-54.ulg"
    source.write_bytes(b"fixture")
    captured: dict[str, object] = {}

    states = np.zeros((31, 13), dtype=np.float64)
    states[:, 6] = 1.0
    powered_trajectory = Trajectory(
        time_s=np.arange(31, dtype=np.float64) * 0.02,
        states=states,
        controls=np.full((30, 4), 0.25),
        spec=make_trajectory_spec(
            (
                "motor_front_left",
                "motor_front_right",
                "motor_rear_right",
                "motor_rear_left",
            ),
            family="multirotor",
            observation_source="estimated",
        ),
        labels={"profile": "published_sysid", "replicate": 1},
        provenance={"source": "fixture.ulg", "px4": {"filters": {}}},
    )

    def fake_load(path, *, config):
        captured["config"] = config
        return powered_trajectory

    monkeypatch.setattr(arp_module, "load_px4_trajectory", fake_load)
    arp_module.ARPReferenceAdapter(verify_checksum=False).load(source)

    config = captured["config"]
    assert config.actuator_hold_max_age_s == pytest.approx(0.10)
    assert config.max_gap_s == pytest.approx(0.10)


def test_idf_reference_pins_actuator_hold_max_age_to_max_gap(
    tmp_path, monkeypatch
) -> None:
    import glassbox.io.idf_reference as idf_module

    source = tmp_path / "log_67_2025-8-21-14-24-48.ulg"
    source.write_bytes(b"fixture")
    captured: dict[str, object] = {}

    def fake_load(path, *, config):
        captured["config"] = config
        return ()

    monkeypatch.setattr(idf_module, "load_px4_trajectories", fake_load)
    idf_module.IDFFixedWingAdapter(verify_checksum=False).load_all(source)

    config = captured["config"]
    assert config.actuator_hold_max_age_s == idf_module.IDF_MAX_GAP_S
    assert config.max_gap_s == idf_module.IDF_MAX_GAP_S
