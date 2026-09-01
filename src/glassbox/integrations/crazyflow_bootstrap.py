"""No-airframe-prior bootstrap identification against the Crazyflow plant."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from glassbox.bootstrap_identification import (
    BootstrapExcitationConfig,
    BootstrapIdentificationConfig,
    BootstrapMultirotorIdentifier,
    plan_bootstrap_excitation,
)
from glassbox.integrations.crazyflow import CrazyflowPlant, CrazyflowPlantConfig


@dataclass(frozen=True)
class CrazyflowBootstrapTrace:
    """State-aligned telemetry retained for diagnostic replay and rendering."""

    sample_period_s: float
    provisional_interval_count: int
    evidence_timestamps_s: np.ndarray
    evidence_states: np.ndarray
    evidence_applied_motor_commands: np.ndarray
    recovery_timestamps_s: np.ndarray
    recovery_states: np.ndarray
    recovery_applied_motor_commands: np.ndarray

    def __post_init__(self) -> None:
        evidence_timestamps = np.asarray(
            self.evidence_timestamps_s,
            dtype=np.float64,
        )
        evidence_states = np.asarray(self.evidence_states, dtype=np.float64)
        evidence_commands = np.asarray(
            self.evidence_applied_motor_commands,
            dtype=np.float64,
        )
        recovery_timestamps = np.asarray(
            self.recovery_timestamps_s,
            dtype=np.float64,
        )
        recovery_states = np.asarray(self.recovery_states, dtype=np.float64)
        recovery_commands = np.asarray(
            self.recovery_applied_motor_commands,
            dtype=np.float64,
        )
        if not np.isfinite(self.sample_period_s) or self.sample_period_s <= 0.0:
            raise ValueError("sample_period_s must be finite and positive")
        for name, timestamps, states, commands in (
            (
                "evidence",
                evidence_timestamps,
                evidence_states,
                evidence_commands,
            ),
            (
                "recovery",
                recovery_timestamps,
                recovery_states,
                recovery_commands,
            ),
        ):
            if timestamps.ndim != 1 or len(timestamps) < 2:
                raise ValueError(f"{name} timestamps must be a nonempty vector")
            if states.shape != (len(timestamps), 13):
                raise ValueError(f"{name} states must align with timestamps")
            if commands.shape != (len(timestamps), 4):
                raise ValueError(f"{name} commands must align with timestamps")
            if (
                not np.all(np.isfinite(timestamps))
                or not np.all(np.diff(timestamps) > 0.0)
                or not np.all(np.isfinite(states))
                or not np.all(np.isfinite(commands))
            ):
                raise ValueError(f"{name} trace values must be finite and ordered")
        if not 0 < self.provisional_interval_count < len(evidence_timestamps):
            raise ValueError("provisional_interval_count must split the evidence")
        object.__setattr__(self, "evidence_timestamps_s", evidence_timestamps)
        object.__setattr__(self, "evidence_states", evidence_states)
        object.__setattr__(
            self,
            "evidence_applied_motor_commands",
            evidence_commands,
        )
        object.__setattr__(self, "recovery_timestamps_s", recovery_timestamps)
        object.__setattr__(self, "recovery_states", recovery_states)
        object.__setattr__(
            self,
            "recovery_applied_motor_commands",
            recovery_commands,
        )


@dataclass(frozen=True)
class CrazyflowBootstrapRun:
    """One report and its exact simulator replay trace."""

    report: dict[str, Any]
    trace: CrazyflowBootstrapTrace


def _bounded_excitation(
    config: BootstrapIdentificationConfig,
    *,
    seed: int,
) -> np.ndarray:
    """Generate deterministic full-rank inputs from command bounds alone."""

    generator = np.random.default_rng(seed)
    minimum = np.asarray(config.command_minimum)
    maximum = np.asarray(config.command_maximum)
    midpoint = 0.5 * (minimum + maximum)
    span = maximum - minimum
    normalized = np.clip(
        0.05 * generator.standard_normal((config.interval_count, 4)),
        -0.10,
        0.10,
    )
    return midpoint + span * normalized


def _initial_state(
    *,
    world_velocity_m_s: tuple[float, float, float] = (0.0, 0.0, 0.0),
    angular_velocity_rad_s: tuple[float, float, float] = (0.0, 0.0, 0.0),
    roll_rad: float = 0.0,
    pitch_rad: float = 0.0,
) -> np.ndarray:
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


def _tilt_rad(state: np.ndarray) -> float:
    quaternion = state[6:10] / np.linalg.norm(state[6:10])
    _, x, y, _ = quaternion
    world_up_body_z = 1.0 - 2.0 * (x * x + y * y)
    return math.acos(float(np.clip(world_up_body_z, -1.0, 1.0)))


def run_crazyflow_bootstrap_trial() -> CrazyflowBootstrapRun:
    """Fit direct motor effects and retain the independent arrest trace."""

    provisional_config = BootstrapIdentificationConfig(
        interval_count=24,
        minimum_normalized_command_rms=0.003,
    )
    identifier_config = BootstrapIdentificationConfig(
        interval_count=28,
        minimum_normalized_command_rms=0.003,
    )
    provisional_identifier = BootstrapMultirotorIdentifier(provisional_config)
    identifier = BootstrapMultirotorIdentifier(identifier_config)
    # Crazyflow establishes SciPy's array-API process contract. Import and
    # construct it before the JAX SVD prewarm can lazily import SciPy.
    plant = CrazyflowPlant(CrazyflowPlantConfig())
    try:
        provisional_prewarm_wall_time_s = provisional_identifier.prewarm()
        final_prewarm_wall_time_s = identifier.prewarm()
        plant.set_arm_length_ratio(1.25)
        minimum = np.asarray(identifier_config.command_minimum)
        maximum = np.asarray(identifier_config.command_maximum)
        midpoint = 0.5 * (minimum + maximum)
        excitation = _bounded_excitation(provisional_config, seed=11)
        sample = plant.reset(
            _initial_state(),
            applied_motor_thrust_fraction=midpoint,
        )
        timestamps = [sample.time_s]
        states = [sample.state]
        evidence_applied_motor_commands = [
            sample.applied_motor_thrust_fraction,
        ]
        applied_commands = []
        for command in excitation:
            previous_applied = sample.applied_motor_thrust_fraction
            sample = plant.step(command)
            timestamps.append(sample.time_s)
            states.append(sample.state)
            evidence_applied_motor_commands.append(sample.applied_motor_thrust_fraction)
            applied_commands.append(
                0.5 * (previous_applied + sample.applied_motor_thrust_fraction)
            )

        provisional = provisional_identifier.fit(
            np.asarray(timestamps),
            np.asarray(states),
            np.asarray(applied_commands),
        )
        excitation_plan = plan_bootstrap_excitation(
            provisional,
            BootstrapExcitationConfig(
                interval_count=(
                    identifier_config.interval_count - provisional_config.interval_count
                ),
                amplitude_fraction_of_command_span=0.08,
            ),
        )
        for command in excitation_plan.commands:
            previous_applied = sample.applied_motor_thrust_fraction
            sample = plant.step(command)
            timestamps.append(sample.time_s)
            states.append(sample.state)
            evidence_applied_motor_commands.append(sample.applied_motor_thrust_fraction)
            applied_commands.append(
                0.5 * (previous_applied + sample.applied_motor_thrust_fraction)
            )

        result = identifier.fit(
            np.asarray(timestamps),
            np.asarray(states),
            np.asarray(applied_commands),
        )
        if result.hover_command is None:
            raise RuntimeError("bootstrap evidence did not produce a hover command")

        recovery_state = _initial_state(
            world_velocity_m_s=(0.8, -0.5, 1.2),
            angular_velocity_rad_s=(1.2, -0.9, 0.55),
            roll_rad=0.20,
            pitch_rad=-0.15,
        )
        sample = plant.reset(
            recovery_state,
            applied_motor_thrust_fraction=result.hover_command,
        )
        recovery_timestamps = [sample.time_s]
        recovery_states = [sample.state]
        recovery_applied_motor_commands = [
            sample.applied_motor_thrust_fraction,
        ]
        recovery_commands = []
        previous_command = result.hover_command
        for _ in range(100):
            decision = result.velocity_attitude_rate_arrest_command(
                sample.state[3:6],
                sample.state[6:10],
                sample.state[10:13],
                velocity_gain=(1.5, 1.5, 1.5),
                maximum_world_acceleration_m_s2=(2.5, 2.5, 3.0),
                maximum_tilt_rad=0.5,
                previous_command=previous_command,
                maximum_motor_step=0.08,
            )
            previous_command = decision.command
            recovery_commands.append(decision.command)
            sample = plant.step(decision.command)
            recovery_timestamps.append(sample.time_s)
            recovery_states.append(sample.state)
            recovery_applied_motor_commands.append(sample.applied_motor_thrust_fraction)

        recovery_state_array = np.asarray(recovery_states)
        recovery_command_array = np.asarray(recovery_commands)
        velocity_norm = np.linalg.norm(recovery_state_array[:, 3:6], axis=1)
        rate_norm = np.linalg.norm(recovery_state_array[:, 10:13], axis=1)
        tilt = np.asarray([_tilt_rad(state) for state in recovery_state_array])
        hidden_hover = plant.hover_motor_thrust_fraction
        estimated_hover = float(np.mean(result.hover_command))
        hover_relative_error = abs(estimated_hover - hidden_hover) / hidden_hover
        finite = bool(
            np.all(np.isfinite(recovery_state_array))
            and np.all(np.isfinite(recovery_command_array))
        )
        bounded = bool(
            np.all(recovery_command_array >= minimum)
            and np.all(recovery_command_array <= maximum)
        )
        rate_reduction_ratio = float(rate_norm[-1] / rate_norm[0])
        velocity_reduction_ratio = float(velocity_norm[-1] / velocity_norm[0])
        altitude_excursion = float(
            np.max(np.abs(recovery_state_array[:, 2] - recovery_state_array[0, 2]))
        )
        report = {
            "artifact_type": "glassbox_crazyflow_no_prior_bootstrap_diagnostic",
            "schema_version": 3,
            "semantics": {
                "diagnostic_only": True,
                "flight_safety_claim": False,
                "throw_to_recover_claim": False,
                "airframe_parameter_prior_used": False,
                "canonical_motor_mixer_supplied_to_identifier": False,
                "hover_command_supplied_to_identifier": False,
                "command_bounds_and_channel_shape_known": True,
                "measured_applied_motor_state_used": True,
                "follow_up_excitation_selected_from_provisional_fit_only": True,
                "hidden_plant_values_used_only_for_post_fit_evaluation": True,
            },
            "configuration": {
                "hidden_arm_length_ratio": 1.25,
                "provisional_identifier": provisional_config.to_dict(),
                "identifier": identifier_config.to_dict(),
                "excitation_center_policy": "command_bounds_midpoint",
                "excitation_maximum_fraction_of_command_span": 0.10,
                "follow_up_excitation": excitation_plan.to_dict(),
            },
            "timing": {
                "provisional_prewarm_wall_time_s": (provisional_prewarm_wall_time_s),
                "final_prewarm_wall_time_s": final_prewarm_wall_time_s,
                "prewarm_wall_time_s": (
                    provisional_prewarm_wall_time_s + final_prewarm_wall_time_s
                ),
                "provisional_fit_wall_time_s": provisional.wall_time_s,
                "fit_wall_time_s": result.wall_time_s,
                "evidence_duration_s": result.evidence_duration_s,
            },
            "identification": result.to_dict(),
            "provisional_identification": provisional.to_dict(),
            "evaluation_only": {
                "hidden_hover_motor_command": hidden_hover,
                "estimated_hover_motor_command": estimated_hover,
                "hover_relative_error": hover_relative_error,
            },
            "velocity_attitude_rate_arrest": {
                "step_count": len(recovery_commands),
                "initial_velocity_norm_m_s": float(velocity_norm[0]),
                "terminal_velocity_norm_m_s": float(velocity_norm[-1]),
                "velocity_reduction_ratio": velocity_reduction_ratio,
                "maximum_velocity_norm_m_s": float(np.max(velocity_norm)),
                "initial_angular_rate_norm_rad_s": float(rate_norm[0]),
                "terminal_angular_rate_norm_rad_s": float(rate_norm[-1]),
                "rate_reduction_ratio": rate_reduction_ratio,
                "maximum_angular_rate_norm_rad_s": float(np.max(rate_norm)),
                "initial_tilt_rad": float(tilt[0]),
                "terminal_tilt_rad": float(tilt[-1]),
                "maximum_tilt_rad": float(np.max(tilt)),
                "initial_altitude_m": float(recovery_state_array[0, 2]),
                "terminal_altitude_m": float(recovery_state_array[-1, 2]),
                "maximum_altitude_excursion_m": altitude_excursion,
                "terminal_vertical_velocity_m_s": float(recovery_state_array[-1, 5]),
                "commands_finite": finite,
                "commands_bounded": bounded,
            },
            "observations": {
                "identifier_ready": result.ready,
                "provisional_identifier_not_ready": not provisional.ready,
                "hover_error_below_10_percent": hover_relative_error < 0.10,
                "velocity_reduced_by_80_percent": velocity_reduction_ratio < 0.20,
                "angular_rate_reduced_by_80_percent": rate_reduction_ratio < 0.20,
                "terminal_tilt_below_0_05_rad": bool(tilt[-1] < 0.05),
                "maximum_tilt_below_0_5_rad": bool(np.max(tilt) < 0.5),
                "altitude_excursion_below_0_75_m": altitude_excursion < 0.75,
                "all_recovery_values_finite": finite,
                "all_recovery_commands_bounded": bounded,
            },
            "limitations": [
                "The evidence starts at the midpoint of known normalized command bounds.",
                "Exact simulator state and measured applied rotor state are used.",
                "The post-fit controller regulates velocity, level attitude, and rates but not position.",
                "The excitation is bounded and airborne, not a ballistic throw.",
                "No sensor noise, estimator delay, packet loss, or firmware scheduling is included.",
            ],
        }
        report["observations"]["gate_passed"] = all(report["observations"].values())
        json.dumps(report, allow_nan=False)
        trace = CrazyflowBootstrapTrace(
            sample_period_s=plant.sample_period_s,
            provisional_interval_count=provisional_config.interval_count,
            evidence_timestamps_s=np.asarray(timestamps),
            evidence_states=np.asarray(states),
            evidence_applied_motor_commands=np.asarray(evidence_applied_motor_commands),
            recovery_timestamps_s=np.asarray(recovery_timestamps),
            recovery_states=recovery_state_array,
            recovery_applied_motor_commands=np.asarray(recovery_applied_motor_commands),
        )
        return CrazyflowBootstrapRun(report=report, trace=trace)
    finally:
        plant.close()


def run_crazyflow_bootstrap_benchmark() -> dict[str, Any]:
    """Fit direct motor effects, then independently arrest body rates."""

    return run_crazyflow_bootstrap_trial().report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the no-prior Crazyflow bootstrap diagnostic."
    )
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = run_crazyflow_bootstrap_benchmark()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
