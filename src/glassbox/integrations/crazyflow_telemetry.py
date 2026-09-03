"""Crazyflow plant telemetry: trajectory generation and closed-loop recording.

Every Crazyflow experiment in this package drives the same hidden simulator and
records the same state-aligned telemetry from it. This module owns that shared
boundary so the prototype, the no-prior bootstrap, the throw campaign, and the
animation renderers all read one definition of a recorded sample.
"""

from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from glassbox.control.nmpc.solver import SolverPolicy
from glassbox.core.data import (
    ObservationChannel,
    Trajectory,
    make_trajectory_spec,
)
from glassbox.core.dynamics import (
    MOTOR_MIXER,
    QUADROTOR_CONTROL_NAMES,
)
from glassbox.core.synthetic import resting_state
from glassbox.integrations.crazyflow import CrazyflowPlant, CrazyflowPlantConfig

PROTOTYPE_SCHEMA_VERSION = 2
DEFAULT_DURATION_S = 6.0
DEFAULT_ARM_LENGTH_RATIO = math.exp(0.20)
CONTROL_HORIZON_STEPS = 30


def initial_plant_state(
    *,
    world_velocity_m_s: tuple[float, float, float] = (0.0, 0.0, 0.0),
    angular_velocity_rad_s: tuple[float, float, float] = (0.0, 0.0, 0.0),
    roll_rad: float = 0.0,
    pitch_rad: float = 0.0,
) -> np.ndarray:
    """Return a canonical release state two metres up with the given attitude."""

    state = np.zeros(13, dtype=np.float64)
    state[2] = 2.0
    cos_roll = math.cos(0.5 * roll_rad)
    sin_roll = math.sin(0.5 * roll_rad)
    cos_pitch = math.cos(0.5 * pitch_rad)
    sin_pitch = math.sin(0.5 * pitch_rad)
    state[6:10] = (
        cos_roll * cos_pitch,
        sin_roll * cos_pitch,
        cos_roll * sin_pitch,
        -sin_roll * sin_pitch,
    )
    state[3:6] = world_velocity_m_s
    state[10:13] = angular_velocity_rad_s
    return state


def tilt_rad(state: np.ndarray) -> float:
    """Return the angle between the body up axis and world up."""

    quaternion = state[6:10] / np.linalg.norm(state[6:10])
    _, x, y, _ = quaternion
    world_up_body_z = 1.0 - 2.0 * (x * x + y * y)
    return math.acos(float(np.clip(world_up_body_z, -1.0, 1.0)))


class PlantTelemetryRecorder:
    """Accumulate the state-aligned trace every closed-loop experiment records.

    A recorder holds the sample times, canonical states, and measured applied
    motor state produced by one plant run, plus the trapezoidal average applied
    command over each interval, which is the input an identifier consumes.
    Recording starts from the plant's reset sample so the trace and the command
    list stay aligned: ``n + 1`` samples for ``n`` intervals.
    """

    def __init__(self, sample: Any) -> None:
        self.timestamps_s: list[float] = [sample.time_s]
        self.states: list[np.ndarray] = [sample.state]
        self.applied_motor_commands: list[np.ndarray] = [
            sample.applied_motor_thrust_fraction
        ]
        self.average_applied_commands: list[np.ndarray] = []

    @property
    def latest_applied(self) -> np.ndarray:
        return self.applied_motor_commands[-1]

    def record(
        self, sample: Any, *, previous_applied: np.ndarray | None = None
    ) -> None:
        """Append one plant sample and, when asked, its interval average input."""

        if previous_applied is not None:
            self.average_applied_commands.append(
                0.5 * (previous_applied + sample.applied_motor_thrust_fraction)
            )
        self.timestamps_s.append(sample.time_s)
        self.states.append(sample.state)
        self.applied_motor_commands.append(sample.applied_motor_thrust_fraction)

    def timestamp_array(self) -> np.ndarray:
        return np.asarray(self.timestamps_s)

    def state_array(self) -> np.ndarray:
        return np.asarray(self.states)

    def applied_array(self) -> np.ndarray:
        return np.asarray(self.applied_motor_commands)

    def average_applied_array(self) -> np.ndarray:
        return np.asarray(self.average_applied_commands)


def _applied_motor_observation_channels() -> tuple[ObservationChannel, ...]:
    return tuple(
        ObservationChannel(
            name=f"applied_{motor}_motor_thrust_fraction",
            role=f"applied_{motor}_motor_thrust_fraction",
            semantic="normalized_per_motor_thrust_fraction",
            unit="1",
            frame="FLU",
            source="crazyflow_first_principles_rotor_state",
        )
        for motor in (
            "front_left",
            "front_right",
            "rear_right",
            "rear_left",
        )
    )


def prewarm_controller(
    controller: Any,
    reference: Any,
    state: np.ndarray,
    hover: np.ndarray,
) -> tuple[Any, Any]:
    """Compile and warm-start one controller at a state, returning both solves.

    The first solve pays JAX compilation. The second reuses the first solve's
    warm start, which is the state the closed loop actually runs in. Both are
    blocked on, so a wall-time measurement taken around this call charges
    compilation here rather than to the loop that follows it.
    """

    cold = controller.solve(
        jnp.asarray(state),
        reference,
        jnp.asarray(hover),
        applied_command=jnp.asarray(hover),
    )
    jax.block_until_ready(cold.command)
    warm = controller.solve(
        jnp.asarray(state),
        reference,
        jnp.asarray(hover),
        applied_command=jnp.asarray(hover),
        warm_start=cold.warm_start,
    )
    jax.block_until_ready(warm.command)
    return cold, warm


def _crazyflow_solver_policy() -> SolverPolicy:
    """Return the fixed 50 Hz prototype policy with explicit timing headroom."""

    return SolverPolicy(
        horizon_steps=CONTROL_HORIZON_STEPS,
        block_count=10,
        maximum_iterations=6,
        line_search_steps=16,
    )


def _trajectory_statistics(trajectory: Trajectory) -> dict[str, Any]:
    return {
        "duration_s": float(trajectory.time_s[-1]),
        "sample_count": len(trajectory.time_s),
        "finite": bool(
            np.all(np.isfinite(trajectory.states))
            and np.all(np.isfinite(trajectory.controls))
        ),
        "position_minimum_m": np.min(trajectory.states[:, 0:3], axis=0).tolist(),
        "position_maximum_m": np.max(trajectory.states[:, 0:3], axis=0).tolist(),
        "velocity_maximum_absolute_m_s": np.max(
            np.abs(trajectory.states[:, 3:6]), axis=0
        ).tolist(),
        "angular_velocity_maximum_absolute_rad_s": np.max(
            np.abs(trajectory.states[:, 10:13]), axis=0
        ).tolist(),
        "command_minimum": np.min(trajectory.controls, axis=0).tolist(),
        "command_maximum": np.max(trajectory.controls, axis=0).tolist(),
    }


def generate_crazyflow_trajectory(
    *,
    seed: int,
    duration_s: float = DEFAULT_DURATION_S,
    arm_length_ratio: float = 1.0,
    source_group: str | None = None,
    configuration_id: str = "crazyflow_adjustable_arm_unknown",
    plant_config: CrazyflowPlantConfig | None = None,
) -> Trajectory:
    """Generate canonical command/state telemetry from a hidden Crazyflow plant."""

    if not np.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError("duration_s must be finite and positive")
    if not np.isfinite(arm_length_ratio) or arm_length_ratio <= 0.0:
        raise ValueError("arm_length_ratio must be finite and positive")
    config = CrazyflowPlantConfig() if plant_config is None else plant_config
    interval_count = round(duration_s / config.sample_period_s)
    if interval_count < 1:
        raise ValueError("duration is shorter than one control interval")

    rng = np.random.default_rng(seed)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=5)
    frequency_scale = rng.uniform(0.9, 1.1)
    plant = CrazyflowPlant(config)
    try:
        plant.set_arm_length_ratio(arm_length_ratio)
        initial_state = resting_state()
        initial_state[2] = 1.0
        hover = plant.hover_motor_thrust_fraction
        hover_command = np.full(4, hover, dtype=np.float64)
        initial = plant.reset(
            initial_state,
            applied_motor_thrust_fraction=hover_command,
        )

        states = np.empty((interval_count + 1, 13), dtype=np.float64)
        controls = np.empty((interval_count, 4), dtype=np.float64)
        applied = np.empty((interval_count + 1, 4), dtype=np.float64)
        states[0] = initial.state
        applied[0] = initial.applied_motor_thrust_fraction
        mixer_transpose = np.asarray(MOTOR_MIXER.T)
        for index in range(interval_count):
            time_s = index * config.sample_period_s
            state = states[index]
            excitation_ramp = min(time_s / 0.5, 1.0)
            attitude_vector = np.sign(state[6] or 1.0) * state[7:10]
            desired_angles = np.asarray(
                (
                    0.05 * state[1] + 0.08 * state[4],
                    -0.05 * state[0] - 0.08 * state[3],
                    0.0,
                )
            )
            desired_attitude_vector = 0.5 * desired_angles
            differential_excitation = excitation_ramp * np.asarray(
                (
                    0.006
                    * np.sin(frequency_scale * 2.0 * np.pi * 0.53 * time_s + phases[2]),
                    0.006 * np.sin(2.0 * np.pi * 0.67 * time_s + phases[3]),
                    0.004 * np.sin(2.0 * np.pi * 0.41 * time_s + phases[4]),
                )
            )
            desired_differential = (
                -0.28 * (attitude_vector - desired_attitude_vector)
                - 0.035 * state[10:13]
                + differential_excitation
            )
            collective = excitation_ramp * (
                0.004
                * np.sin(frequency_scale * 2.0 * np.pi * 0.37 * time_s + phases[0])
                + 0.002 * np.sin(2.0 * np.pi * 0.83 * time_s + phases[1])
            )
            collective += -0.12 * (state[2] - 1.0) - 0.08 * state[5]
            command = hover + collective + 0.25 * mixer_transpose @ desired_differential
            command = np.clip(command, 0.05, 0.95)
            controls[index] = command
            sample = plant.step(command)
            states[index + 1] = sample.state
            applied[index + 1] = sample.applied_motor_thrust_fraction

        group = f"crazyflow-trajectory-{seed}" if source_group is None else source_group
        spec = make_trajectory_spec(
            QUADROTOR_CONTROL_NAMES,
            family="multirotor",
            observation_source="simulator_truth",
            configuration_id=configuration_id,
            observations=_applied_motor_observation_channels(),
        )
        return Trajectory(
            time_s=np.arange(interval_count + 1, dtype=np.float64)
            * config.sample_period_s,
            states=states,
            controls=controls,
            observations=applied,
            spec=spec,
            labels={
                "seed": seed,
                "source_group": group,
            },
            provenance={
                "adapter": {
                    "name": "crazyflow_hidden_plant",
                    "schema_version": PROTOTYPE_SCHEMA_VERSION,
                    "crazyflow_version": plant.crazyflow_version,
                },
                "simulator_contract": {
                    "drone": config.drone,
                    "dynamics": "first_principles",
                    "control": "rotor_vel",
                    "simulation_frequency_hz": config.simulation_frequency_hz,
                    "control_frequency_hz": config.control_frequency_hz,
                    "canonical_motor_order": (
                        "front_left,front_right,rear_right,rear_left"
                    ),
                    "quaternion_storage": "wxyz",
                    "world_frame": "NWU",
                    "body_frame": "FLU",
                    "command_semantic": "normalized_per_motor_thrust_fraction",
                    "maximum_motor_thrust_n": config.maximum_motor_thrust_n,
                },
                "generator": {"seed": seed},
                "telemetry_only": {
                    "hidden_physical_parameters_supplied_to_glassbox": False,
                    "applied_actuator_state_retained_as_typed_observation": True,
                },
            },
        )
    finally:
        plant.close()
