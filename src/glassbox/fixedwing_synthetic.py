"""Synthetic fixed-wing telemetry for model-family and recovery benchmarks."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from glassbox.data import Trajectory, make_trajectory_spec
from glassbox.dynamics import (
    FIXED_WING_CONTROL_NAMES,
    GRAVITY_M_S2,
    FixedWingDynamicsParams,
    fixed_wing_trim_control,
    step_with_latent,
)

TRIM_AIRSPEED_M_S = 15.0


def true_fixed_wing_parameters() -> FixedWingDynamicsParams:
    """Return the hidden effective coefficients used by the synthetic plant."""

    return FixedWingDynamicsParams.from_physical(
        thrust_accel=3.0,
        lift_accel_per_speed_sq=GRAVITY_M_S2 / TRIM_AIRSPEED_M_S**2,
        lift_alpha_accel_per_speed_sq=1.0,
        drag_accel_per_speed_sq=0.003,
        side_force_accel_per_speed=0.040,
        surface_angular_accel_per_speed_sq=(0.040, 0.030, 0.020),
        pitch_stability_accel_per_speed_sq=1.0,
        lateral_stability_angular_accel_per_speed_sq=(0.3, 2.0),
        angular_drag_per_speed=(2.0 / 15.0, 1.8 / 15.0, 1.5 / 15.0),
        actuator_time_constant=0.08,
        surface_trim=(0.025, -0.040, 0.015),
    )


def initial_fixed_wing_parameter_guess() -> FixedWingDynamicsParams:
    """Return a deliberately imperfect fixed-wing identification start."""

    return FixedWingDynamicsParams.from_physical(
        thrust_accel=2.3,
        lift_accel_per_speed_sq=0.034,
        lift_alpha_accel_per_speed_sq=1.0,
        drag_accel_per_speed_sq=0.0042,
        side_force_accel_per_speed=0.025,
        surface_angular_accel_per_speed_sq=(0.030, 0.040, 0.014),
        pitch_stability_accel_per_speed_sq=1.0,
        lateral_stability_angular_accel_per_speed_sq=(0.3, 2.0),
        angular_drag_per_speed=(1.3 / 15.0, 2.4 / 15.0, 1.0 / 15.0),
        actuator_time_constant=0.03,
        surface_trim=(0.0, 0.0, 0.0),
    )


def fixed_wing_trim_state(airspeed_m_s: float = TRIM_AIRSPEED_M_S) -> np.ndarray:
    """Return a level NWU/FLU state moving north at the requested airspeed."""

    state = np.zeros(13, dtype=np.float64)
    state[3] = airspeed_m_s
    state[6] = 1.0
    return state


def generate_fixed_wing_trajectory(
    *,
    seed: int,
    duration_s: float = 6.0,
    dt_s: float = 0.02,
    params: FixedWingDynamicsParams | None = None,
) -> Trajectory:
    """Generate bounded cruise telemetry with throttle and surface excitation."""

    if duration_s <= 0.0 or dt_s <= 0.0:
        raise ValueError("duration_s and dt_s must be positive")
    interval_count = round(duration_s / dt_s)
    if interval_count < 1:
        raise ValueError("duration is shorter than one sample interval")

    params = true_fixed_wing_parameters() if params is None else params
    rng = np.random.default_rng(seed)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=4)
    frequency_scale = rng.uniform(0.9, 1.1)
    trim_control = np.asarray(
        fixed_wing_trim_control(params, TRIM_AIRSPEED_M_S), dtype=np.float64
    )

    states = np.empty((interval_count + 1, 13), dtype=np.float64)
    controls = np.empty((interval_count, 4), dtype=np.float64)
    states[0] = fixed_wing_trim_state()
    applied_control = trim_control.copy()

    for index in range(interval_count):
        time_s = index * dt_s
        state = states[index]
        ramp = min(time_s / 0.75, 1.0)
        attitude_vector = np.sign(state[6] or 1.0) * state[7:10]
        angular_velocity = state[10:13]

        control = trim_control.copy()
        control[0] += 0.035 * (TRIM_AIRSPEED_M_S - state[3]) + ramp * (
            0.035 * np.sin(frequency_scale * 2.0 * np.pi * 0.17 * time_s + phases[0])
            + 0.015 * np.sin(2.0 * np.pi * 0.43 * time_s + phases[1])
        )
        surface_excitation = ramp * np.asarray(
            [
                0.030 * np.sin(2.0 * np.pi * 0.31 * time_s + phases[1]),
                0.025 * np.sin(2.0 * np.pi * 0.27 * time_s + phases[2]),
                0.022 * np.sin(2.0 * np.pi * 0.23 * time_s + phases[3]),
            ]
        )
        control[1:4] = trim_control[1:4] + (
            -0.65 * attitude_vector - 0.12 * angular_velocity + surface_excitation
        )
        control[3] -= 0.012 * state[4]
        control[0] = np.clip(control[0], 0.05, 0.95)
        control[1:4] = np.clip(control[1:4], -0.25, 0.25)
        controls[index] = control

        next_state, next_applied_control = step_with_latent(
            params,
            jnp.asarray(state),
            jnp.asarray(applied_control),
            jnp.asarray(control),
            dt_s,
        )
        states[index + 1] = np.asarray(next_state)
        applied_control = np.asarray(next_applied_control)

    return Trajectory(
        time_s=np.arange(interval_count + 1, dtype=np.float64) * dt_s,
        states=states,
        controls=controls,
        spec=make_trajectory_spec(
            FIXED_WING_CONTROL_NAMES,
            family="fixedwing",
            observation_source="simulator_truth",
            configuration_id="synthetic_fixedwing",
            fixed_states={"wind_world_m_s": [0.0, 0.0, 0.0]},
        ),
        labels={
            "profile": f"multisine_{seed % 3}",
            "seed": seed,
        },
        provenance={
            "adapter": {"name": "synthetic_fixedwing", "schema_version": 1},
            "generator": {"seed": seed},
        },
    )
