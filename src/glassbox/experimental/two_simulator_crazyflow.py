"""Frozen Crazyflow flight fixture; no learner or controller-consumer imports.

The 13 public state coordinates are NWU/FLU/WXYZ. Four additional coordinates
are rotor RPM / 20000 in Crazyflow order. The excitation pilot is adapted from
the pinned Dart collector, while every parent and branch interval calls the
same separately compiled integrator. Validity/admission belongs to the runner.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import os
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from glassbox.core.data import make_trajectory_spec
from glassbox.core.dynamics import (
    MOTOR_MIXER,
    QUADROTOR_CONTROL_NAMES,
    quaternion_multiply,
    quaternion_to_rotation,
)
from glassbox.core.geometry import quaternion_log_error, state_plus_tangent

RPM_SCALE = 20_000.0
CF_FROM_GB = np.asarray([1, 2, 3, 0])
GB_FROM_CF = np.argsort(CF_FROM_GB)
CONFIGURATION = "two-simulator-crazyflow-cf2x_L250-v1"
CRAZYFLOW_COMMIT = "1142d2848320a63e5d404b21265348d78768f593"
FULL_STATE_NAMES = (
    "position_x",
    "position_y",
    "position_z",
    "velocity_x",
    "velocity_y",
    "velocity_z",
    "quaternion_w",
    "quaternion_x",
    "quaternion_y",
    "quaternion_z",
    "body_rate_x",
    "body_rate_y",
    "body_rate_z",
    "rotor_fr_rpm_over_20000",
    "rotor_rr_rpm_over_20000",
    "rotor_rl_rpm_over_20000",
    "rotor_fl_rpm_over_20000",
)
PILOT_TARGET_NAMES = (
    "desired_quaternion_w",
    "desired_quaternion_x",
    "desired_quaternion_y",
    "desired_quaternion_z",
    "desired_angle_vector_derivative_x",
    "desired_angle_vector_derivative_y",
    "desired_angle_vector_derivative_z",
)
WIND_MODES = {"calm": 0, "constant": 1, "gust": 2}


class FixtureSetupError(ValueError):
    """An explicit configuration/import failure, with serializable evidence."""

    def __init__(self, stage: str, reason: str):
        self.detail = {"stage": stage, "reason": reason}
        super().__init__(json.dumps(self.detail, sort_keys=True))


def wind_at_ticks(ticks, mode):
    """NWU wind at exact global integer millisecond RK-stage indices."""
    times = jnp.asarray(ticks, dtype=jnp.float64) / 1000.0
    gust = jnp.where(
        (times > 0.5) & (times < 2.5),
        2.0 * jnp.sin(jnp.pi * (times - 0.5) / 2.0) ** 2,
        0.0,
    )
    speed = jnp.where(mode == 1, 2.0, jnp.where(mode == 2, gust, 0.0))
    return jnp.stack((speed, jnp.zeros_like(speed), jnp.zeros_like(speed)), axis=-1)


def _yaw_quaternion(cell):
    yaw = np.deg2rad(float(cell["heading_nwu_deg"]))
    return jnp.asarray([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])


def _rotation_exp(angles):
    identity = jnp.zeros(13, dtype=jnp.float64).at[6].set(1.0)
    tangent = jnp.zeros(12, dtype=jnp.float64).at[6:9].set(angles)
    return state_plus_tangent(identity, tangent)[6:10]


class Fixture:
    """One physical Crazyflow configuration across the frozen condition grid."""

    def __init__(self, protocol: dict):
        if not jax.config.x64_enabled:
            raise FixtureSetupError(
                "runtime", "JAX float64 must be enabled before setup"
            )
        if os.environ.get("SCIPY_ARRAY_API") != "1":
            raise FixtureSetupError(
                "runtime", "SCIPY_ARRAY_API must be 1 before imports"
            )
        generation = protocol["generation"]
        setup = generation["crazyflow"]
        if (
            setup["drone"],
            setup["dt_s"],
            setup["integration_dt_s"],
            setup["substeps"],
        ) != (
            "cf2x_L250",
            0.01,
            0.002,
            5,
        ):
            raise FixtureSetupError(
                "configuration", "expected frozen cf2x_L250/100Hz/500Hz setup"
            )
        self.dt = float(setup["dt_s"])
        self.substeps = int(setup["substeps"])
        self.integration_dt = float(setup["integration_dt_s"])
        self.steps = round(generation["duration_s"] / self.dt)
        if self.steps < 1 or not np.isclose(
            self.steps * self.dt, generation["duration_s"], atol=1e-12, rtol=0
        ):
            raise FixtureSetupError(
                "configuration", "duration must contain complete recorded intervals"
            )
        self.mode_scale = copy.deepcopy(generation["mode_scale"])
        self.drone = setup["drone"]
        try:
            from crazyflow.drones import load_params as load_physical_params
            from crazyflow.dynamics import load_params
            from crazyflow.dynamics.first_principles import dynamics
        except ImportError as exc:
            raise FixtureSetupError("imports", str(exc)) from exc
        self._dynamics = dynamics
        try:
            self.params = load_params(dynamics, self.drone, xp=jnp)
            physical = load_physical_params(self.drone)
        except OSError as exc:
            raise FixtureSetupError("assets", str(exc)) from exc
        self.thrust_max = float(physical["thrust_max"])
        self.lower = np.full(4, float(physical["thrust_min"]) / self.thrust_max)
        self.upper = np.ones(4, dtype=np.float64)
        self.hover = float(self.params["mass"] * 9.81 / (4.0 * self.thrust_max))
        spec = make_trajectory_spec(
            QUADROTOR_CONTROL_NAMES,
            family="multirotor",
            observation_source="simulator_truth",
            configuration_id=CONFIGURATION,
        )
        self.spec = replace(
            spec,
            channels=tuple(
                replace(
                    channel, minimum=float(self.lower[i]), maximum=float(self.upper[i])
                )
                for i, channel in enumerate(spec.channels)
            ),
        )
        source_paths = {
            "dynamics": Path(inspect.getfile(dynamics)).resolve(),
            "physical_parameters": Path(inspect.getfile(load_physical_params))
            .resolve()
            .parent
            / "params.toml",
        }
        self.source_identity = {
            "declared_crazyflow_commit": CRAZYFLOW_COMMIT,
            "loaded_files": {
                name: {
                    "path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                for name, path in source_paths.items()
            },
        }
        # Materialized commands enter this exact kernel in both calling paths.
        self._advance = jax.jit(self._advance_impl)
        self._pilot = jax.jit(self._pilot_impl)

    def config(self) -> dict:
        return {
            "simulator": "crazyflow",
            "configuration_id": CONFIGURATION,
            "drone": self.drone,
            "dt_s": self.dt,
            "integration_dt_s": self.integration_dt,
            "substeps": self.substeps,
            "duration_s": self.steps * self.dt,
            "full_state_names": list(FULL_STATE_NAMES),
            "command_names": list(QUADROTOR_CONTROL_NAMES),
            "bounds": np.stack((self.lower, self.upper), axis=-1).tolist(),
            "actuator_output_names": [
                f"realized_{name}" for name in QUADROTOR_CONTROL_NAMES
            ],
            "pilot_target_names": list(PILOT_TARGET_NAMES),
            "pilot_rate_convention": "analytic angle-vector derivative used as desired body rate",
            "control_prefix": "50 constant hover commands; equilibrium initialization prior, no simulated history",
            "hover_command": self.hover,
            "rpm_scale": RPM_SCALE,
            "source_identity": copy.deepcopy(self.source_identity),
            "physical_parameters": {
                key: np.asarray(value).tolist() for key, value in self.params.items()
            },
            "wind_frame": "NWU",
            "integration_force_frame": "NWU",
            "integration_force_unit": "N",
        }

    def command_to_rpm(self, command):
        force = command[jnp.asarray(CF_FROM_GB)] * self.thrust_max
        k0, k1, k2 = (self.params["rpm2thrust"][..., i] for i in range(3))
        return (-k1 + jnp.sqrt(k1**2 + 4 * k2 * (force - k0))) / (2 * k2)

    def _derivative(self, state, command, wind):
        quaternion = state[6:10] / jnp.linalg.norm(state[6:10])
        dp, _, dv, dw, dr = self._dynamics(
            pos=state[:3],
            quat=quaternion[jnp.asarray([1, 2, 3, 0])],
            vel=state[3:6],
            ang_vel=state[10:13],
            cmd=self.command_to_rpm(command),
            rotor_vel=state[13:] * RPM_SCALE,
            **self.params,
        )
        rotation = quaternion_to_rotation(quaternion)
        force = -rotation @ self.params["drag_matrix"] @ rotation.T @ wind
        dv = dv + force / self.params["mass"]
        dq = 0.5 * quaternion_multiply(
            quaternion, jnp.concatenate((jnp.zeros(1), state[10:13]))
        )
        return jnp.concatenate((dp, dv, dq, dw, dr / RPM_SCALE)), force

    def _advance_impl(self, state, command, origin_index, wind_mode):
        ticks = (
            origin_index * 10 + 2 * jnp.arange(5)[:, None] + jnp.asarray([0, 1, 1, 2])
        )
        winds = wind_at_ticks(ticks, wind_mode)
        dt = self.integration_dt

        def integrate(current, stage_wind):
            k1, f1 = self._derivative(current, command, stage_wind[0])
            k2, f2 = self._derivative(current + dt * k1 / 2, command, stage_wind[1])
            k3, f3 = self._derivative(current + dt * k2 / 2, command, stage_wind[2])
            k4, f4 = self._derivative(current + dt * k3, command, stage_wind[3])
            following = current + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            following = following.at[6:10].set(
                following[6:10] / jnp.linalg.norm(following[6:10])
            )
            return following, jnp.stack((f1, f2, f3, f4))

        following, forces = jax.lax.scan(integrate, state, winds)
        return following, winds, forces

    def _initial(self, seed, cell):
        rng = np.random.Generator(np.random.PCG64(seed))
        velocity = rng.uniform(-0.5, 0.5, 3)
        attitude = rng.uniform(-0.2, 0.2, 3)
        rates = rng.uniform(-0.3, 0.3, 3)
        phases = rng.uniform(0, 2 * np.pi, 3)
        frequencies = rng.uniform(0.3, 0.65, 3)
        amplitude = rng.uniform((0.2, 0.5, 0.25), (0.55, 1.1, 0.65))
        scale = self.mode_scale[cell["maneuver"]]
        yaw = _yaw_quaternion(cell)
        state = jnp.zeros(17, dtype=jnp.float64).at[2].set(4.0)
        state = state.at[3:6].set(
            quaternion_to_rotation(yaw)
            @ jnp.asarray(velocity + np.asarray([cell["speed_m_s"], 0.0, 0.0]))
        )
        state = state.at[6:10].set(
            quaternion_multiply(yaw, _rotation_exp(jnp.asarray(attitude)))
        )
        state = state.at[10:13].set(jnp.asarray(rates))
        state = state.at[13:].set(
            self.command_to_rpm(jnp.full(4, self.hover)) / RPM_SCALE
        )
        draws = {
            "velocity_jitter": velocity.tolist(),
            "attitude_tangent": attitude.tolist(),
            "body_rates": rates.tolist(),
            "phases": phases.tolist(),
            "frequencies_hz": frequencies.tolist(),
            "base_amplitudes_rad": amplitude.tolist(),
            "amplitudes_rad": (amplitude * scale).tolist(),
            "mode_scale": scale,
        }
        return np.asarray(state, dtype=np.float64), draws

    def _pilot_impl(self, state, origin_index, yaw, phases, frequencies, amplitudes):
        time = (origin_index * 10).astype(jnp.float64) / 1000.0
        argument = 2 * jnp.pi * frequencies * time + phases
        angles = amplitudes * jnp.sin(argument)
        rates = amplitudes * 2 * jnp.pi * frequencies * jnp.cos(argument)
        desired = quaternion_multiply(yaw, _rotation_exp(angles))
        error = quaternion_log_error(desired, state[6:10])
        alpha = -18 * error - 7 * (state[10:13] - rates)
        collective = (
            self.hover
            - 0.10 * (state[2] - 4.0)
            - 0.12 * state[5]
            + 0.06 * jnp.sin(4 * time)
        )
        command = jnp.clip(
            collective
            + 0.25 * MOTOR_MIXER.T @ (alpha / jnp.asarray([232.0, 232.0, 30.0])),
            jnp.asarray(self.lower),
            jnp.asarray(self.upper),
        )
        return command, jnp.concatenate((desired, rates))

    def _arrays(self, full_states, commands, winds, forces, origin_index, wind_mode):
        count = len(commands)
        full_states = np.asarray(full_states, dtype=np.float64).reshape(count + 1, 17)
        # Host-only diagnostic reduction never feeds the integrator or pilot.
        rpm = full_states[:, 13:] * RPM_SCALE
        k0, k1, k2 = (np.asarray(self.params["rpm2thrust"])[..., i] for i in range(3))
        outputs = ((k0 + k1 * rpm + k2 * rpm**2) / self.thrust_max)[:, GB_FROM_CF]
        winds = np.asarray(winds, dtype=np.float64).reshape(count, 5, 4, 3)
        boundary_wind = (
            np.concatenate((winds[:, 0, 0], winds[-1:, -1, -1]))
            if count
            else np.asarray(
                wind_at_ticks(
                    jnp.asarray([origin_index * 10], dtype=jnp.int64), wind_mode
                )
            )
        )
        return {
            "time_s": (
                (origin_index + np.arange(count + 1, dtype=np.int64)) * 10
            ).astype(np.float64)
            / 1000.0,
            "states": full_states[:, :13].copy(),
            "full_states": full_states.copy(),
            "commands": np.asarray(commands, dtype=np.float64).reshape(count, 4),
            "actuator_outputs": np.asarray(outputs, dtype=np.float64),
            "wind": boundary_wind,
            "integration_wind": winds,
            "integration_force": np.asarray(forces, dtype=np.float64).reshape(
                count, 5, 4, 3
            ),
        }

    def generate(self, entry: dict, cell: dict) -> tuple[dict, dict]:
        if entry["simulator"] != "crazyflow" or entry["cell"] != cell["id"]:
            raise FixtureSetupError(
                "entry", "recording and Crazyflow condition must agree"
            )
        wind_mode = WIND_MODES[cell["wind"]]
        state, draws = self._initial(entry["seed"], cell)
        initial = state.copy()
        states, commands, winds, forces, targets = [state], [], [], [], []
        pilot_args = tuple(
            jnp.asarray(draws[key], dtype=jnp.float64)
            for key in (
                "phases",
                "frequencies_hz",
                "amplitudes_rad",
            )
        )
        yaw = _yaw_quaternion(cell)
        for index in range(self.steps):
            command, target = self._pilot(
                jnp.asarray(state),
                jnp.asarray(index, dtype=jnp.int64),
                yaw,
                *pilot_args,
            )
            command = np.asarray(command, dtype=np.float64)
            state, wind, force = self._advance(
                jnp.asarray(state),
                jnp.asarray(command),
                jnp.asarray(index, dtype=jnp.int64),
                jnp.asarray(wind_mode, dtype=jnp.int64),
            )
            state = np.asarray(state, dtype=np.float64)
            states.append(state)
            commands.append(command)
            winds.append(np.asarray(wind))
            forces.append(np.asarray(force))
            targets.append(np.asarray(target))
        arrays = self._arrays(states, commands, winds, forces, 0, wind_mode)
        arrays["pilot_targets"] = np.asarray(targets, dtype=np.float64)
        arrays["control_prefix"] = np.full((50, 4), self.hover, dtype=np.float64)
        metadata = {
            "seed": int(entry["seed"]),
            "recording_id": entry["id"],
            "condition": copy.deepcopy(cell),
            "initial_full_state": initial.tolist(),
            "random_draws": draws,
            "rng": "numpy.PCG64",
            "simulator_randomness": False,
            "feedback_pilot": True,
            "origin_index": 0,
            "configuration_id": CONFIGURATION,
        }
        return arrays, metadata

    def branch(
        self, full_state, commands, origin_index: int, cell: dict
    ) -> tuple[dict, dict]:
        state = np.asarray(full_state, dtype=np.float64)
        commands = np.asarray(commands, dtype=np.float64)
        if state.shape != (17,) or commands.ndim != 2 or commands.shape[1] != 4:
            raise ValueError("branch requires full state[17] and commands[T,4]")
        if not isinstance(origin_index, (int, np.integer)) or origin_index < 0:
            raise ValueError("branch origin_index must be a nonnegative integer")
        wind_mode = WIND_MODES[cell["wind"]]
        states, winds, forces = [state.copy()], [], []
        for offset, command in enumerate(commands):
            state, wind, force = self._advance(
                jnp.asarray(state),
                jnp.asarray(command),
                jnp.asarray(origin_index + offset, dtype=jnp.int64),
                jnp.asarray(wind_mode, dtype=jnp.int64),
            )
            state = np.asarray(state, dtype=np.float64)
            states.append(state)
            winds.append(np.asarray(wind))
            forces.append(np.asarray(force))
        return self._arrays(states, commands, winds, forces, origin_index, wind_mode), {
            "condition": copy.deepcopy(cell),
            "origin_index": int(origin_index),
            "configuration_id": CONFIGURATION,
            "feedback_pilot": False,
            "simulator_randomness": False,
        }
