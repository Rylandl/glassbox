import numpy as np
import pytest

from glassbox.data import (
    ControlChannel,
    RIGID_BODY_STATE_SCHEMA,
    Trajectory,
    TrajectorySpec,
    VehicleConfigurationSpec,
    angular_acceleration_observation_channels,
    save_trajectory_npz,
    specific_force_observation_channels,
)
from glassbox.dynamics import MOTOR_MIXER, QUADROTOR_CONTROL_NAMES
from glassbox.observation_identification import (
    actuator_observation_alignment,
    fit_multirotor_observations,
)
from glassbox.fit_cli import fit_trajectory_artifact


def _trajectory(*, lag_steps: int = 0, intervals: int = 240) -> Trajectory:
    dt_s = 0.01
    time_s = np.arange(intervals + 1) * dt_s
    phase = np.arange(intervals) * dt_s
    controls = np.column_stack(
        (
            0.55 + 0.12 * np.sin(2.0 * np.pi * 1.1 * phase),
            0.52 + 0.10 * np.sin(2.0 * np.pi * 0.8 * phase + 0.4),
            0.50 + 0.09 * np.sin(2.0 * np.pi * 1.3 * phase + 0.9),
            0.53 + 0.11 * np.sin(2.0 * np.pi * 0.6 * phase + 1.2),
        )
    )
    states = np.zeros((intervals + 1, 13), dtype=np.float64)
    states[:, 6] = 1.0
    states[:, 3:6] = np.column_stack(
        (
            0.4 * np.sin(0.7 * time_s),
            0.3 * np.sin(0.9 * time_s + 0.2),
            0.2 * np.sin(1.1 * time_s + 0.5),
        )
    )
    states[:, 10:13] = np.column_stack(
        (
            0.2 * np.sin(1.2 * time_s),
            0.15 * np.sin(0.8 * time_s + 0.3),
            0.1 * np.sin(1.5 * time_s + 0.6),
        )
    )
    force_controls = controls
    if lag_steps:
        force_controls = np.vstack(
            (np.repeat(controls[:1], lag_steps, axis=0), controls[:-lag_steps])
        )
    specific_force = np.zeros((intervals + 1, 3), dtype=np.float64)
    specific_force[:-1] = -0.18 * states[:-1, 3:6]
    specific_force[:-1, 2] += 3.4 * np.sum(force_controls, axis=1)
    specific_force[-1] = specific_force[-2]
    mixer = force_controls @ np.asarray(MOTOR_MIXER).T
    angular_acceleration = np.zeros((intervals + 1, 3), dtype=np.float64)
    angular_acceleration[:-1] = (
        mixer * np.asarray([11.0, 12.0, 5.0])
        - states[:-1, 10:13] * np.asarray([0.3, 0.4, 0.2])
    )
    angular_acceleration[-1] = angular_acceleration[-2]
    observations = np.column_stack((specific_force, angular_acceleration))
    controls_spec = tuple(
        ControlChannel(
            name=name,
            role=name,
            semantic="squared_rotor_speed_ratio",
            unit="1",
            minimum=0.0,
            frame="FLU",
        )
        for name in QUADROTOR_CONTROL_NAMES
    )
    spec = TrajectorySpec(
        state_schema=RIGID_BODY_STATE_SCHEMA,
        observation_source="synthetic",
        controls=controls_spec,
        vehicle=VehicleConfigurationSpec(
            family="multirotor",
            propulsion="quadrotor",
            controlled_axes=("roll", "pitch", "yaw"),
        ),
        observations=(
            *specific_force_observation_channels("synthetic_imu"),
            *angular_acceleration_observation_channels("synthetic_imu"),
        ),
    )
    return Trajectory(
        time_s=time_s,
        states=states,
        controls=controls,
        spec=spec,
        observations=observations,
    )


def test_alignment_applies_only_to_confident_physical_actuator_lag() -> None:
    base = _trajectory()
    rng = np.random.default_rng(4)
    controls = rng.uniform(0.2, 0.9, size=base.controls.shape)
    observations = base.observations.copy()
    collective = np.sum(controls, axis=1)
    observations[:-1, 2] = 3.4 * np.concatenate(
        (np.repeat(collective[0], 3), collective[:-3])
    )
    delayed = Trajectory(
        time_s=base.time_s,
        states=base.states,
        controls=controls,
        spec=base.spec,
        observations=observations,
    )

    diagnostic = actuator_observation_alignment(delayed)

    assert diagnostic.lag_steps == 3
    assert diagnostic.lag_s == pytest.approx(0.03)
    assert diagnostic.alignment_applied is True
    assert diagnostic.correlation > 0.95


def test_observation_fit_recovers_structured_force_and_rotation() -> None:
    result = fit_multirotor_observations([_trajectory()])
    physical = result.params.physical()

    assert float(physical["thrust_accel"]) == pytest.approx(3.4, rel=0.02)
    assert float(physical["linear_drag"]) == pytest.approx(0.18, rel=0.1)
    np.testing.assert_allclose(
        physical["angular_accel"], [11.0, 12.0, 5.0], rtol=0.03
    )
    np.testing.assert_allclose(
        physical["angular_drag"], [0.3, 0.4, 0.2], rtol=0.15
    )
    assert result.report["force_fit"]["validation_rmse_m_s2"] < 1e-3
    assert result.report["selection_data"]["test_split_used"] is False


def test_rollout_fit_reports_typed_observation_fit_without_promoting_it(
    tmp_path,
) -> None:
    path = tmp_path / "trajectory.npz"
    save_trajectory_npz(_trajectory(intervals=80), path)

    _, report = fit_trajectory_artifact(path, horizon=8, steps=1)

    identification = report["observation_identification"]
    assert identification["policy"] == "typed_observation_identification_v1"
    assert identification["force_fit"]["passed_sensor_residual_gate"] is True
    assert identification["rollout_initializer"]["applied"] is False
    assert identification["rollout_initializer"]["status"] == "diagnostic_only"
    assert report["parameters"]["initial"]["thrust_accel"] == pytest.approx(4.5)
