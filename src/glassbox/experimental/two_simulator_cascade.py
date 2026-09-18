"""Pinned Cascade recording and exact-state response fixture.

This is a simulator-side harness, not a learner or consumer adapter. Recorded
commands are throttle/elevator/aileron; all native state leaves remain available
only to physical replay and counterfactual branching.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from glassbox.core.data import make_trajectory_spec

CONFIGURATION_ID = "two-simulator-cascade-skywalker_x8-v1"
COMMAND_NAMES = ("throttle", "elevator", "aileron")
PERMUTATION = (0, 2, 1)
WIND_CODES = {"calm": 0, "constant": 1, "gust": 2}


class FixtureSetupError(ValueError):
    """A declared initialization failed; detail is deterministic JSON data."""

    def __init__(self, detail):
        self.detail = _json_value(detail)
        super().__init__(json.dumps(self.detail, sort_keys=True, allow_nan=False))


def _json_value(value):
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (np.ndarray, jax.Array)):
        return _json_value(np.asarray(value).tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _wind_nwu(native_index, code):
    """Use one integer 400 Hz clock in parent, branches and saved schedules."""
    t = jnp.asarray(native_index, dtype=jnp.float64) / 400.0
    gust = jnp.where(
        (t > 0.5) & (t < 2.5), 2.0 * jnp.sin(jnp.pi * (t - 0.5) / 2.0) ** 2, 0.0
    )
    north = jnp.where(code == 1, 2.0, jnp.where(code == 2, gust, 0.0))
    return jnp.stack((north, jnp.zeros_like(north), jnp.zeros_like(north)), axis=-1)


def _pack(state):
    return jnp.concatenate(
        (
            state.rigid_body.position,
            state.rigid_body.attitude,
            state.rigid_body.velocity,
            state.rigid_body.angular_velocity,
            state.actuators.surface_deflection,
            state.actuators.propeller_speed,
            state.aero.separation,
        )
    )


def _targets(cell, mode_scale, phases, index):
    t = index / 20.0
    ramp = min(1.0, t)
    return np.asarray(
        [
            cell["speed_m_s"] + ramp * 0.8 * mode_scale * np.sin(0.7 * t + phases[0]),
            100.0 + ramp * 2.0 * mode_scale * np.sin(0.9 * t + phases[1]),
            -np.deg2rad(cell["heading_nwu_deg"])
            + ramp * 0.4 * mode_scale * np.sin(0.7 * t + phases[2]),
        ],
        dtype=np.float64,
    )


class Fixture:
    """One immutable X8 physical configuration with a cached declared trim."""

    def __init__(self, protocol: dict):
        import cascade
        from cascade.analysis import StraightFlightCondition, trim_straight_flight
        from cascade.canonical import rigid_body_to_canonical
        from cascade.control import (
            GuidanceSetpoint,
            cascade_step,
            initial_cascade_state,
            skywalker_x8_controller,
        )
        from cascade.initialization import control_from_array, control_to_array
        from cascade.integration import rk4_step
        from cascade.state import (
            ActuatorState,
            AeroState,
            AircraftState,
            Environment,
            RigidBodyState,
        )

        if not jax.config.x64_enabled:
            raise ValueError("Cascade fixture requires the frozen JAX float64 runtime")
        self.protocol = copy.deepcopy(protocol)
        declared = self.protocol["generation"]["cascade"]
        if (
            declared["dt_s"] != 0.05
            or declared["integration_dt_s"] != 0.0025
            or declared["substeps"] != 20
            or declared["aircraft"] != "skywalker_x8"
        ):
            raise ValueError(
                "Cascade timing or aircraft differs from the frozen fixture"
            )
        self.dt = float(declared["dt_s"])
        self.lower = np.asarray([0.0, -0.35, -0.35])
        self.upper = np.asarray([1.0, 0.35, 0.35])
        self.steps = round(protocol["generation"]["duration_s"] / self.dt)
        if (
            self.steps < 1
            or self.steps * self.dt != protocol["generation"]["duration_s"]
        ):
            raise ValueError("recording duration is not an integral positive grid")
        self.history_steps = round(
            protocol["fitting"]["generic_recipe"]["context_s"] / self.dt
        )
        self.spec = make_trajectory_spec(
            COMMAND_NAMES,
            family="fixedwing",
            observation_source="simulator_truth",
            configuration_id=CONFIGURATION_ID,
        )
        self.spec = replace(
            self.spec,
            channels=tuple(
                replace(
                    channel,
                    minimum=float(self.lower[index]),
                    maximum=float(self.upper[index]),
                    semantic="normalized_command"
                    if index == 0
                    else "generalized_surface_angle",
                    unit="1" if index == 0 else "rad",
                )
                for index, channel in enumerate(self.spec.channels)
            ),
        )
        source_root = Path(declared["source_root"]).resolve()
        imported = Path(cascade.__file__).resolve()
        if imported != source_root / "src/cascade/__init__.py":
            raise ValueError("Cascade must import from the frozen clean source archive")
        self.aircraft_spec = cascade.skywalker_x8_spec()
        native_names = tuple(
            prop.name for prop in self.aircraft_spec.propellers
        ) + tuple(self.aircraft_spec.control_channels)
        if native_names != ("throttle", "aileron", "elevator"):
            raise ValueError("native Cascade command layout differs")
        self.model = self.aircraft_spec.to_model()
        self._n_surface = len(self.aircraft_spec.surfaces)
        self._n_propeller = len(self.aircraft_spec.propellers)
        self._size = 13 + 2 * self._n_surface + self._n_propeller
        self._trim_function = trim_straight_flight
        self._condition_type = StraightFlightCondition
        self._initial_pilot = initial_cascade_state
        self._base_pilot = skywalker_x8_controller()._replace(
            rate_period=1, attitude_period=1, guidance_period=2
        )
        self._trim_cache = {}
        self._wind = jax.jit(_wind_nwu)

        def unpack(vector):
            end_surface = 13 + self._n_surface
            end_propeller = end_surface + self._n_propeller
            return AircraftState(
                RigidBodyState(vector[:3], vector[3:7], vector[7:10], vector[10:13]),
                ActuatorState(
                    vector[13:end_surface], vector[end_surface:end_propeller]
                ),
                AeroState(vector[end_propeller:]),
            )

        def environment(native_index, code):
            return Environment(
                jnp.asarray(1.225, dtype=jnp.float64),
                _wind_nwu(native_index, code) * jnp.asarray([1.0, -1.0, -1.0]),
                jnp.asarray([0.0, 0.0, 9.81], dtype=jnp.float64),
            )

        def advance(vector, command, index, code):
            control = control_from_array(self.model, command[jnp.asarray(PERMUTATION)])

            def substep(state, offset):
                native_index = index * 20 + offset
                state = rk4_step(
                    self.model,
                    state,
                    control,
                    environment(native_index, code),
                    1.0 / 400,
                )
                return state, _wind_nwu(native_index, code)

            state, winds = jax.lax.scan(substep, unpack(vector), jnp.arange(20))
            return _pack(state), winds

        def observe(vector):
            state = unpack(vector)
            return rigid_body_to_canonical(state.rigid_body), jnp.concatenate(
                (state.actuators.surface_deflection, state.actuators.propeller_speed)
            )

        def pilot_step(controller, pilot_state, vector, targets, index, code):
            control, updated = cascade_step(
                controller,
                pilot_state,
                GuidanceSetpoint(*targets),
                unpack(vector),
                environment(index * 20, code),
                self.dt,
            )
            return control_to_array(control)[jnp.asarray(PERMUTATION)], updated

        self._unpack = unpack
        self._environment = environment
        self._advance = jax.jit(advance)
        self._observe = jax.jit(observe)
        self._pilot_step = jax.jit(pilot_step)
        self._control_to_array = control_to_array
        runtime_assets = {
            str(path.relative_to(source_root)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in sorted((source_root / "src/cascade").rglob("*"))
            if path.is_file() and path.suffix in {".py", ".toml"}
        }
        self._source = {
            "declared_commit": declared["commit"],
            "imported_module": str(imported),
            "runtime_assets": runtime_assets,
            "runtime_assets_roster_sha256": hashlib.sha256(
                json.dumps(
                    runtime_assets, sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest(),
            "aircraft_spec_sha256": cascade.spec_hash(self.aircraft_spec),
        }

    def config(self):
        surface_names = [surface.name for surface in self.aircraft_spec.surfaces]
        propeller_names = [
            propeller.name for propeller in self.aircraft_spec.propellers
        ]
        full_names = (
            [f"position_ned_{axis}_m" for axis in "xyz"]
            + [f"attitude_xyzw_{axis}" for axis in "xyzw"]
            + [f"velocity_ned_{axis}_m_s" for axis in "xyz"]
            + [f"angular_velocity_frd_{axis}_rad_s" for axis in "xyz"]
            + [f"surface_{name}_rad" for name in surface_names]
            + [f"propeller_{name}_rad_s" for name in propeller_names]
            + [f"separation_{name}" for name in surface_names]
        )
        return {
            "configuration_id": CONFIGURATION_ID,
            "trajectory_spec": self.spec.to_dict(),
            "dt_s": self.dt,
            "integration_dt_s": 1 / 400,
            "integration_substeps": 20,
            "full_state_names": full_names,
            "actuator_output_names": full_names[
                13 : 13 + self._n_surface + self._n_propeller
            ],
            "command_names": list(COMMAND_NAMES),
            "native_command_names": ["throttle", "aileron", "elevator"],
            "command_permutation": list(PERMUTATION),
            "lower": self.lower.tolist(),
            "upper": self.upper.tolist(),
            "pilot_target_names": ["airspeed_m_s", "altitude_m", "heading_ned_rad"],
            "wind_frame": "NWU",
            "source_identity": copy.deepcopy(self._source),
            "simulator_rng_used": False,
            "control_prefix_semantics": "Declared equilibrium-command prior; not observed pre-reset motion",
        }

    def _cell(self, cell):
        if (
            cell["wind"] not in WIND_CODES
            or cell["maneuver"] not in self.protocol["generation"]["mode_scale"]
        ):
            raise ValueError("unknown Cascade condition")
        if (
            not np.isfinite([cell["speed_m_s"], cell["heading_nwu_deg"]]).all()
            or cell["speed_m_s"] <= 0
        ):
            raise ValueError("invalid Cascade initial speed or heading")
        return WIND_CODES[cell["wind"]]

    def _trim(self, cell, code):
        initial_wind = tuple(np.asarray(self._wind(0, code)).tolist())
        key = (float(cell["speed_m_s"]), float(cell["heading_nwu_deg"]), initial_wind)
        if key in self._trim_cache:
            value = self._trim_cache[key]
            if isinstance(value, dict):
                raise FixtureSetupError(value)
            return value
        condition = self._condition_type(
            airspeed_m_s=key[0],
            heading_rad=-np.deg2rad(key[1]),
            altitude_m=100.0,
            flight_path_angle_rad=0.0,
        )
        try:
            trim = self._trim_function(
                self.model,
                condition,
                self._environment(0, code),
                residual_tolerance=1e-4,
                max_evaluations=300,
            )
        except (ValueError, RuntimeError) as error:
            detail = {
                "stage": "trim",
                "reason": "trim_exception",
                "condition": list(key),
                "error_type": type(error).__name__,
                "error": str(error),
            }
            self._trim_cache[key] = detail
            raise FixtureSetupError(detail) from error
        vector = np.asarray(_pack(trim.state))
        command = np.asarray(self._control_to_array(trim.control))[list(PERMUTATION)]
        report = _json_value(
            {
                "success": bool(trim.success),
                "optimizer_success": bool(trim.optimizer_success),
                "decision": trim.decision,
                "residual": trim.residual,
                "scaled_residual": trim.scaled_residual,
                "cost": trim.cost,
                "optimality": trim.optimality,
                "evaluations": trim.evaluations,
                "message": trim.message,
                "angle_of_attack_rad": trim.angle_of_attack_rad,
                "sideslip_rad": trim.sideslip_rad,
                "command": command,
            }
        )
        reason = None
        if not np.isfinite(vector).all() or not np.isfinite(command).all():
            reason = "nonfinite_trim"
        elif not trim.success:
            reason = "unsuccessful_trim"
        elif np.any(command < self.lower) or np.any(command > self.upper):
            reason = "trim_outside_command_bounds"
        if reason:
            detail = {
                "stage": "trim",
                "reason": reason,
                "condition": list(key),
                "trim": report,
            }
            self._trim_cache[key] = detail
            raise FixtureSetupError(detail)
        value = (trim, command, report)
        self._trim_cache[key] = value
        return value

    def _arrays(self, full_states, commands, winds, origin_index, code):
        full_states = np.asarray(full_states, dtype=np.float64)
        observations = [self._observe(jnp.asarray(row)) for row in full_states]
        indices = origin_index + np.arange(len(full_states), dtype=np.int64)
        return {
            "time_s": indices.astype(np.float64) / 20.0,
            "states": np.stack([np.asarray(row[0]) for row in observations]),
            "full_states": full_states,
            "commands": np.asarray(commands, dtype=np.float64).reshape(-1, 3),
            "actuator_outputs": np.stack([np.asarray(row[1]) for row in observations]),
            "wind": np.asarray(self._wind(indices * 20, code)),
            "integration_wind": np.asarray(winds, dtype=np.float64).reshape(-1, 20, 3),
        }

    def generate(self, entry, cell):
        code = self._cell(cell)
        seed = entry["seed"]
        if (
            type(seed) is not int
            or seed < 0
            or entry["simulator"] != "cascade"
            or entry["cell"] != cell["id"]
        ):
            raise ValueError("recording identity does not match the Cascade condition")
        trim, initial_command, trim_report = self._trim(cell, code)
        rng = np.random.Generator(np.random.PCG64(seed))
        phases = rng.uniform(0.0, 2 * np.pi, 3)
        mode_scale = self.protocol["generation"]["mode_scale"][cell["maneuver"]]
        controller = self._base_pilot._replace(
            guidance=self._base_pilot.guidance._replace(
                pitch_trim=trim.decision[1], throttle_trim=trim.control.propeller[0]
            )
        )
        pilot_state = self._initial_pilot(controller, trim.state, trim.control)
        current = _pack(trim.state)
        full_states = [np.asarray(current)]
        commands, winds, targets, raw_commands, dither = [], [], [], [], []
        trim_offset = np.asarray([0.0, initial_command[1], initial_command[2]])
        for index in range(self.steps):
            target = _targets(cell, mode_scale, phases, index)
            raw, pilot_state = self._pilot_step(
                controller, pilot_state, current, jnp.asarray(target), index, code
            )
            t = index / 20.0
            requested = (
                min(1.0, t)
                * mode_scale
                * np.asarray([0.035, 0.025, 0.02])
                * np.sin(np.asarray([1.3, 2.1, 1.7]) * t + phases)
            )
            command = np.clip(
                np.asarray(raw) + trim_offset + requested, self.lower, self.upper
            )
            current, integration_wind = self._advance(
                current, jnp.asarray(command), index, code
            )
            full_states.append(np.asarray(current))
            commands.append(command)
            winds.append(np.asarray(integration_wind))
            targets.append(target)
            raw_commands.append(np.asarray(raw))
            dither.append(requested)
        arrays = self._arrays(full_states, commands, winds, 0, code)
        arrays.update(
            control_prefix=np.repeat(initial_command[None], self.history_steps, axis=0),
            pilot_targets=np.asarray(targets),
            pilot_raw_commands=np.asarray(raw_commands),
            requested_dither=np.asarray(dither),
        )
        return arrays, {
            "configuration_id": CONFIGURATION_ID,
            "seed": seed,
            "entry": copy.deepcopy(entry),
            "condition": copy.deepcopy(cell),
            "trim": copy.deepcopy(trim_report),
            "phases_rad": phases.tolist(),
            "pilot": {
                "rate_period": 1,
                "attitude_period": 1,
                "guidance_period": 2,
                "pitch_trim": float(trim.decision[1]),
                "throttle_trim": float(trim.control.propeller[0]),
                "recorded_trim_offset": trim_offset.tolist(),
                "mode_scale": mode_scale,
            },
            "simulator_rng_used": False,
        }

    def branch(self, full_state, commands, origin_index: int, cell):
        code = self._cell(cell)
        vector = np.asarray(full_state, dtype=np.float64)
        tape = np.asarray(commands, dtype=np.float64)
        if type(origin_index) is not int or origin_index < 0:
            raise ValueError(
                "branch origin must be a nonnegative recorded integer index"
            )
        if vector.shape != (self._size,) or not np.isfinite(vector).all():
            raise ValueError("branch needs an exact finite full physical state")
        if tape.ndim != 2 or tape.shape[1] != 3 or not np.isfinite(tape).all():
            raise ValueError("branch command tape must be finite with three columns")
        if np.any(tape < self.lower) or np.any(tape > self.upper):
            raise ValueError("branch command tape exceeds the declared bounds")
        current = jnp.asarray(vector)
        full_states, winds = [vector.copy()], []
        for local_index, command in enumerate(tape):
            current, integration_wind = self._advance(
                current, jnp.asarray(command), origin_index + local_index, code
            )
            full_states.append(np.asarray(current))
            winds.append(np.asarray(integration_wind))
        return self._arrays(full_states, tape, winds, origin_index, code), {
            "configuration_id": CONFIGURATION_ID,
            "origin_index": origin_index,
            "condition": copy.deepcopy(cell),
            "simulator_rng_used": False,
            "initialization": "exact saved full state; no reset or equilibration",
        }
