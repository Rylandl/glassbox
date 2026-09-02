"""Synthetic closed-loop trajectories for the first recovery benchmark."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from glassbox.data import Trajectory, make_trajectory_spec
from glassbox.dynamics import (
    MOTOR_MIXER,
    QUADROTOR_CONTROL_NAMES,
    DynamicsParams,
    hover_control,
    step_with_latent,
)


def true_parameters() -> DynamicsParams:
    """Return the hidden parameters used by the synthetic vehicle."""

    return DynamicsParams.from_physical(
        thrust_accel=5.40,
        angular_accel=(18.0, 16.5, 7.5),
        linear_drag=0.18,
        angular_drag=(0.24, 0.21, 0.13),
        motor_time_constant=0.08,
    )


def initial_parameter_guess() -> DynamicsParams:
    """Return a deliberately imperfect identification starting point."""

    return DynamicsParams.from_physical(
        thrust_accel=4.50,
        angular_accel=(13.0, 13.0, 5.5),
        linear_drag=0.08,
        angular_drag=(0.10, 0.10, 0.07),
        motor_time_constant=0.02,
        angular_response_time_constant=(0.04, 0.04, 0.06),
    )


def resting_state() -> np.ndarray:
    """Return a level, motionless 13-element state."""

    state = np.zeros(13, dtype=np.float64)
    state[6] = 1.0
    return state


def generate_trajectory(
    *,
    seed: int,
    duration_s: float = 6.0,
    dt_s: float = 0.02,
    params: DynamicsParams | None = None,
) -> Trajectory:
    """Generate a bounded trajectory with stabilizing feedback and multisine inputs.

    The feedback keeps the open-loop synthetic plant from simply falling or
    tumbling while the seed-dependent multisine terms produce distinct state and
    actuator histories. Identification only receives the resulting states and
    motor commands, not the controller internals.
    """

    if duration_s <= 0.0 or dt_s <= 0.0:
        raise ValueError("duration_s and dt_s must be positive")

    params = true_parameters() if params is None else params
    interval_count = round(duration_s / dt_s)
    if interval_count < 1:
        raise ValueError("duration is shorter than one sample interval")

    rng = np.random.default_rng(seed)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=5)
    frequency_scale = rng.uniform(0.9, 1.1)
    base_motor_command = float(hover_control(params)[0])
    mixer_transpose = np.asarray(MOTOR_MIXER.T)

    states = np.empty((interval_count + 1, 13), dtype=np.float64)
    controls = np.empty((interval_count, 4), dtype=np.float64)
    states[0] = resting_state()
    motor_state = np.asarray(hover_control(params), dtype=np.float64)

    for index in range(interval_count):
        time_s = index * dt_s
        state = states[index]
        excitation_ramp = min(time_s / 0.5, 1.0)

        collective_excitation = excitation_ramp * (
            0.018 * np.sin(frequency_scale * 2.0 * np.pi * 0.37 * time_s + phases[0])
            + 0.009 * np.sin(2.0 * np.pi * 0.83 * time_s + phases[1])
        )
        collective_feedback = -0.035 * state[2] - 0.025 * state[5]

        attitude_error = np.sign(state[6] or 1.0) * state[7:10]
        angular_velocity = state[10:13]
        differential_excitation = excitation_ramp * np.asarray(
            [
                0.040 * np.sin(2.0 * np.pi * 0.53 * time_s + phases[2]),
                0.036 * np.sin(2.0 * np.pi * 0.67 * time_s + phases[3]),
                0.030 * np.sin(2.0 * np.pi * 0.41 * time_s + phases[4]),
            ]
        )
        desired_differential = (
            -0.30 * attitude_error - 0.075 * angular_velocity + differential_excitation
        )

        control = (
            base_motor_command
            + collective_excitation
            + collective_feedback
            + 0.25 * mixer_transpose @ desired_differential
        )
        control = np.clip(control, 0.05, 0.95)
        controls[index] = control
        next_state, next_motor_state = step_with_latent(
            params,
            jnp.asarray(state),
            jnp.asarray(motor_state),
            jnp.asarray(control),
            dt_s,
        )
        states[index + 1] = np.asarray(next_state)
        motor_state = np.asarray(next_motor_state)

    return Trajectory(
        time_s=np.arange(interval_count + 1, dtype=np.float64) * dt_s,
        states=states,
        controls=controls,
        spec=make_trajectory_spec(
            QUADROTOR_CONTROL_NAMES,
            family="multirotor",
            observation_source="simulator_truth",
            configuration_id="synthetic_quadrotor",
        ),
        labels={"seed": seed},
        provenance={
            "adapter": {"name": "synthetic", "schema_version": 1},
            "generator": {"seed": seed},
        },
    )
