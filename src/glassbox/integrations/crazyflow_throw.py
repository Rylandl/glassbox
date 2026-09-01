"""Unpowered throw followed by continuous online identification and arrest."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from glassbox.integrations.crazyflow import CrazyflowPlant, CrazyflowPlantConfig
from glassbox.integrations.crazyflow_bootstrap import _initial_state, _tilt_rad
from glassbox.online_bootstrap import (
    ProgressiveBootstrapController,
    RecursiveBootstrapIdentifier,
)


@dataclass(frozen=True)
class CrazyflowThrowTrace:
    """State-aligned telemetry for one uninterrupted release-to-hover trial."""

    sample_period_s: float
    model_enable_sample_index: int
    first_supported_control_sample_index: int
    certified_belief_sample_index: int
    timestamps_s: np.ndarray
    states: np.ndarray
    applied_motor_commands: np.ndarray
    requested_motor_commands: np.ndarray
    working_interval_counts: np.ndarray
    command_evidence_ranks: np.ndarray
    angular_effect_ranks: np.ndarray
    collective_authority: np.ndarray
    angular_axis_authority: np.ndarray
    certified_control_active: np.ndarray

    def __post_init__(self) -> None:
        timestamps = np.asarray(self.timestamps_s, dtype=np.float64)
        states = np.asarray(self.states, dtype=np.float64)
        applied = np.asarray(self.applied_motor_commands, dtype=np.float64)
        requested = np.asarray(self.requested_motor_commands, dtype=np.float64)
        interval_counts = np.asarray(self.working_interval_counts, dtype=np.int64)
        command_ranks = np.asarray(self.command_evidence_ranks, dtype=np.int64)
        effect_ranks = np.asarray(self.angular_effect_ranks, dtype=np.int64)
        collective_authority = np.asarray(self.collective_authority, dtype=np.float64)
        angular_authority = np.asarray(self.angular_axis_authority, dtype=np.float64)
        certified = np.asarray(self.certified_control_active, dtype=np.bool_)
        if not np.isfinite(self.sample_period_s) or self.sample_period_s <= 0.0:
            raise ValueError("sample_period_s must be finite and positive")
        if timestamps.ndim != 1 or len(timestamps) < 2:
            raise ValueError("timestamps_s must contain at least two samples")
        aligned_shapes = {
            "states": (len(timestamps), 13),
            "applied_motor_commands": (len(timestamps), 4),
            "working_interval_counts": (len(timestamps),),
            "command_evidence_ranks": (len(timestamps),),
            "angular_effect_ranks": (len(timestamps),),
            "collective_authority": (len(timestamps),),
            "angular_axis_authority": (len(timestamps), 3),
            "certified_control_active": (len(timestamps),),
        }
        values = {
            "states": states,
            "applied_motor_commands": applied,
            "working_interval_counts": interval_counts,
            "command_evidence_ranks": command_ranks,
            "angular_effect_ranks": effect_ranks,
            "collective_authority": collective_authority,
            "angular_axis_authority": angular_authority,
            "certified_control_active": certified,
        }
        for name, shape in aligned_shapes.items():
            if values[name].shape != shape:
                raise ValueError(f"{name} must have state-aligned shape {shape}")
        if requested.shape != (len(timestamps) - 1, 4):
            raise ValueError("requested_motor_commands must be interval-aligned")
        if (
            not np.all(np.isfinite(timestamps))
            or not np.all(np.diff(timestamps) > 0.0)
            or not np.all(np.isfinite(states))
            or not np.all(np.isfinite(applied))
            or not np.all(np.isfinite(requested))
            or not np.all(np.isfinite(collective_authority))
            or not np.all(np.isfinite(angular_authority))
        ):
            raise ValueError("throw trace values must be finite and ordered")
        if not (
            0
            < self.model_enable_sample_index
            < self.first_supported_control_sample_index
            < self.certified_belief_sample_index
            < len(timestamps)
        ):
            raise ValueError("throw phase indices must be strictly ordered")
        if np.any(interval_counts[: self.model_enable_sample_index + 1] != 0):
            raise ValueError("working belief must be disabled before model enable")
        object.__setattr__(self, "timestamps_s", timestamps)
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "applied_motor_commands", applied)
        object.__setattr__(self, "requested_motor_commands", requested)
        object.__setattr__(self, "working_interval_counts", interval_counts)
        object.__setattr__(self, "command_evidence_ranks", command_ranks)
        object.__setattr__(self, "angular_effect_ranks", effect_ranks)
        object.__setattr__(self, "collective_authority", collective_authority)
        object.__setattr__(self, "angular_axis_authority", angular_authority)
        object.__setattr__(self, "certified_control_active", certified)


@dataclass(frozen=True)
class CrazyflowThrowRun:
    """One continuous diagnostic report and its exact replay telemetry."""

    report: dict[str, Any]
    trace: CrazyflowThrowTrace


def _norm(values: np.ndarray) -> np.ndarray:
    return np.linalg.norm(values, axis=1)


def run_crazyflow_throw_trial() -> CrazyflowThrowRun:
    """Tumble unpowered for one second, then identify and stabilize online."""

    plant = CrazyflowPlant(CrazyflowPlantConfig(control_frequency_hz=100))
    try:
        plant.set_arm_length_ratio(1.25)
        identifier = RecursiveBootstrapIdentifier()
        controller = ProgressiveBootstrapController(identifier.config)
        minimum = np.asarray(identifier.config.command_minimum)
        maximum = np.asarray(identifier.config.command_maximum)
        release_state = _initial_state(
            world_velocity_m_s=(1.0, -0.6, 10.0),
            angular_velocity_rad_s=(0.8, -0.6, 0.4),
            roll_rad=0.20,
            pitch_rad=-0.15,
        )
        release_state[2] = 1.2
        sample = plant.reset(
            release_state,
            applied_motor_thrust_fraction=np.zeros(4),
        )
        timestamps = [sample.time_s]
        states = [sample.state]
        applied_state = [sample.applied_motor_thrust_fraction]
        requested_commands: list[np.ndarray] = []
        working_interval_counts = [0]
        command_evidence_ranks = [0]
        angular_effect_ranks = [0]
        collective_authority = [0.0]
        angular_axis_authority = [np.zeros(3)]
        certified_control_active = [False]

        model_enable_delay_s = 1.0
        model_enable_delay_step_count = round(
            model_enable_delay_s / plant.sample_period_s
        )
        zero_command = np.zeros(4, dtype=np.float64)
        for _ in range(model_enable_delay_step_count):
            requested_commands.append(zero_command)
            sample = plant.step(zero_command)
            timestamps.append(sample.time_s)
            states.append(sample.state)
            applied_state.append(sample.applied_motor_thrust_fraction)
            working_interval_counts.append(0)
            command_evidence_ranks.append(0)
            angular_effect_ranks.append(0)
            collective_authority.append(0.0)
            angular_axis_authority.append(np.zeros(3))
            certified_control_active.append(False)
        model_enable_sample_index = len(states) - 1

        target_trial_duration_s = 10.0
        online_step_count = round(
            (target_trial_duration_s - timestamps[-1]) / plant.sample_period_s
        )
        previous_command = zero_command
        first_supported_control_sample_index: int | None = None
        certified_belief_sample_index: int | None = None
        update_wall_times_s: list[float] = []
        for _ in range(online_step_count):
            control_belief = identifier.control_belief
            if (
                first_supported_control_sample_index is None
                and control_belief.has_any_control_authority
            ):
                first_supported_control_sample_index = len(states) - 1
            decision = controller.command(
                sample.state,
                control_belief,
                previous_command=previous_command,
            )
            previous_state = sample.state
            previous_applied = sample.applied_motor_thrust_fraction
            previous_command = decision.command
            requested_commands.append(decision.command)
            sample = plant.step(decision.command)
            average_applied = 0.5 * (
                previous_applied + sample.applied_motor_thrust_fraction
            )
            working_belief = identifier.update(
                previous_state,
                sample.state,
                average_applied,
                plant.sample_period_s,
            )
            if (
                certified_belief_sample_index is None
                and identifier.certified_belief is not None
            ):
                certified_belief_sample_index = len(states)
            update_wall_times_s.append(working_belief.update_wall_time_s)
            timestamps.append(sample.time_s)
            states.append(sample.state)
            applied_state.append(sample.applied_motor_thrust_fraction)
            working_interval_counts.append(working_belief.interval_count)
            command_evidence_ranks.append(working_belief.command_evidence_rank)
            angular_effect_ranks.append(working_belief.angular_effect_rank)
            collective_authority.append(working_belief.collective_authority)
            angular_axis_authority.append(working_belief.angular_axis_authority)
            certified_control_active.append(identifier.certified_belief is not None)

        if first_supported_control_sample_index is None:
            raise RuntimeError("online identifier never earned supported control")
        if certified_belief_sample_index is None or identifier.certified_belief is None:
            raise RuntimeError("online identifier never certified a control belief")
        certified_belief = identifier.certified_belief
        working_belief = identifier.belief
        state_array = np.asarray(states)
        applied_array = np.asarray(applied_state)
        requested_array = np.asarray(requested_commands)
        velocity_norm = _norm(state_array[:, 3:6])
        rate_norm = _norm(state_array[:, 10:13])
        tilt = np.asarray([_tilt_rad(state) for state in state_array])
        horizontal_excursion = _norm(state_array[:, 0:2] - state_array[0, 0:2])
        hidden_hover = plant.hover_motor_thrust_fraction
        if certified_belief.hover_command is None:
            raise RuntimeError("certified belief lost its hover command")
        estimated_hover = float(np.mean(certified_belief.hover_command))
        hover_relative_error = abs(estimated_hover - hidden_hover) / hidden_hover
        command_bound_tolerance = 1e-8
        commands_finite = bool(
            np.all(np.isfinite(requested_array)) and np.all(np.isfinite(applied_array))
        )
        commands_bounded = bool(
            np.all(requested_array >= minimum - command_bound_tolerance)
            and np.all(requested_array <= maximum + command_bound_tolerance)
            and np.all(applied_array >= minimum - command_bound_tolerance)
            and np.all(applied_array <= maximum + command_bound_tolerance)
        )
        pre_enable_commands_zero = bool(
            np.allclose(
                requested_array[:model_enable_delay_step_count],
                0.0,
                atol=2e-11,
                rtol=0.0,
            )
            and np.allclose(
                applied_array[: model_enable_sample_index + 1],
                0.0,
                atol=2e-11,
                rtol=0.0,
            )
        )
        release_velocity = float(velocity_norm[0])
        release_rate = float(rate_norm[0])
        terminal_velocity_ratio = float(velocity_norm[-1] / release_velocity)
        terminal_rate_ratio = float(rate_norm[-1] / release_rate)
        hover_mask = (
            (velocity_norm < 0.10)
            & (rate_norm < 0.10)
            & (np.abs(state_array[:, 5]) < 0.05)
            & (tilt < 0.05)
        )
        sustained_hover_mask = np.logical_and.accumulate(hover_mask[::-1])[::-1]
        sustained_hover_candidates = np.flatnonzero(
            sustained_hover_mask
            & (np.arange(len(state_array)) >= certified_belief_sample_index)
        )
        sustained_hover_start_index = (
            None
            if len(sustained_hover_candidates) == 0
            else int(sustained_hover_candidates[0])
        )
        sustained_hover_duration_s = (
            0.0
            if sustained_hover_start_index is None
            else float(timestamps[-1] - timestamps[sustained_hover_start_index])
        )
        first_authority_time_s = float(timestamps[first_supported_control_sample_index])
        certified_time_s = float(timestamps[certified_belief_sample_index])
        update_times = np.asarray(update_wall_times_s)
        report = {
            "artifact_type": "glassbox_crazyflow_online_throw_recovery_diagnostic",
            "schema_version": 2,
            "semantics": {
                "diagnostic_only": True,
                "flight_safety_claim": False,
                "physical_hand_contact_modeled": False,
                "release_state_injected_by_simulator": True,
                "continuous_after_release": True,
                "simulator_reset_after_release": False,
                "motors_cold_at_release": True,
                "motors_off_until_model_enable": True,
                "model_and_controller_disabled_before_model_enable": True,
                "continuous_identification_during_control": True,
                "separate_evidence_collection_phase": False,
                "progressive_supported_direction_authority": True,
                "working_belief_updated_every_actuated_interval": True,
                "unsupported_closed_loop_candidate_cannot_erase_certified_support": True,
                "initial_control_belief_uses_disjoint_predictive_validation": False,
                "post_admission_candidate_replacement_implemented": False,
                "airframe_parameter_prior_used": False,
                "canonical_motor_mixer_supplied_to_identifier": False,
                "hover_command_supplied_to_identifier": False,
                "command_bounds_and_channel_shape_known": True,
                "measured_applied_motor_state_used": True,
                "hidden_plant_values_used_only_for_post_run_evaluation": True,
            },
            "configuration": {
                "hidden_arm_length_ratio": 1.25,
                "release_height_m": float(release_state[2]),
                "release_world_velocity_m_s": release_state[3:6].tolist(),
                "release_angular_velocity_rad_s": release_state[10:13].tolist(),
                "release_tilt_rad": float(_tilt_rad(release_state)),
                "release_applied_motor_command": [0.0, 0.0, 0.0, 0.0],
                "command_bound_numerical_tolerance": command_bound_tolerance,
                "model_enable_delay_s": model_enable_delay_s,
                "model_enable_delay_step_count": model_enable_delay_step_count,
                "target_trial_duration_s": target_trial_duration_s,
                "online_step_count": online_step_count,
                "sample_period_s": plant.sample_period_s,
                "identifier": {
                    "forgetting_factor": identifier.config.forgetting_factor,
                    "command_rank_relative_tolerance": (
                        identifier.config.command_rank_relative_tolerance
                    ),
                    "minimum_normalized_command_rms": (
                        identifier.config.minimum_normalized_command_rms
                    ),
                    "output_rank_relative_tolerance": (
                        identifier.config.output_rank_relative_tolerance
                    ),
                    "authority_start_interval_count": (
                        identifier.config.authority_start_interval_count
                    ),
                    "authority_full_interval_count": (
                        identifier.config.authority_full_interval_count
                    ),
                    "minimum_certification_interval_count": (
                        identifier.config.minimum_certification_interval_count
                    ),
                },
                "controller": {
                    "velocity_gain": list(controller.config.velocity_gain),
                    "maximum_world_acceleration_m_s2": list(
                        controller.config.maximum_world_acceleration_m_s2
                    ),
                    "maximum_tilt_rad": controller.config.maximum_tilt_rad,
                    "attitude_gain": list(controller.config.attitude_gain),
                    "angular_rate_gain": list(controller.config.angular_rate_gain),
                    "initial_excitation_fraction": (
                        controller.config.initial_excitation_fraction
                    ),
                    "continuing_excitation_fraction": (
                        controller.config.continuing_excitation_fraction
                    ),
                    "maximum_feedback_delta": (
                        controller.config.maximum_feedback_delta
                    ),
                    "maximum_motor_step": controller.config.maximum_motor_step,
                },
            },
            "timing": {
                "model_enable_time_s": float(timestamps[model_enable_sample_index]),
                "first_supported_control_time_s": first_authority_time_s,
                "certified_belief_time_s": certified_time_s,
                "time_from_enable_to_first_supported_control_s": (
                    first_authority_time_s - model_enable_delay_s
                ),
                "time_from_enable_to_certified_belief_s": (
                    certified_time_s - model_enable_delay_s
                ),
                "total_trial_duration_s": float(timestamps[-1]),
                "median_recursive_update_wall_time_s": float(np.median(update_times)),
                "maximum_recursive_update_wall_time_s": float(np.max(update_times)),
            },
            "identification": {
                "certified_control_belief": certified_belief.to_dict(),
                "terminal_working_belief": working_belief.to_dict(),
                "working_update_count": working_belief.interval_count,
                "control_authority_was_progressive_before_certification": True,
                "control_belief_admission_contract": {
                    "minimum_interval_count": (
                        identifier.config.minimum_certification_interval_count
                    ),
                    "required_command_evidence_rank": 4,
                    "required_angular_effect_rank": 3,
                    "minimum_axis_authority": 0.95,
                    "feasible_hover_command_required": True,
                    "disjoint_predictive_validation_required": False,
                    "claim": "structural_support_and_feasibility_only",
                },
                "terminal_candidate_rejected_for_lost_independent_support": bool(
                    working_belief.command_evidence_rank
                    < certified_belief.command_evidence_rank
                    or working_belief.angular_effect_rank
                    < certified_belief.angular_effect_rank
                ),
            },
            "evaluation_only": {
                "hidden_hover_motor_command": hidden_hover,
                "estimated_hover_motor_command": estimated_hover,
                "hover_relative_error": hover_relative_error,
            },
            "continuous_throw": {
                "release_velocity_norm_m_s": release_velocity,
                "model_enable_velocity_norm_m_s": float(
                    velocity_norm[model_enable_sample_index]
                ),
                "first_supported_control_velocity_norm_m_s": float(
                    velocity_norm[first_supported_control_sample_index]
                ),
                "certified_belief_velocity_norm_m_s": float(
                    velocity_norm[certified_belief_sample_index]
                ),
                "terminal_velocity_norm_m_s": float(velocity_norm[-1]),
                "terminal_to_release_velocity_ratio": terminal_velocity_ratio,
                "release_angular_rate_norm_rad_s": release_rate,
                "model_enable_angular_rate_norm_rad_s": float(
                    rate_norm[model_enable_sample_index]
                ),
                "first_supported_control_angular_rate_norm_rad_s": float(
                    rate_norm[first_supported_control_sample_index]
                ),
                "certified_belief_angular_rate_norm_rad_s": float(
                    rate_norm[certified_belief_sample_index]
                ),
                "terminal_angular_rate_norm_rad_s": float(rate_norm[-1]),
                "terminal_to_release_rate_ratio": terminal_rate_ratio,
                "release_tilt_rad": float(tilt[0]),
                "model_enable_tilt_rad": float(tilt[model_enable_sample_index]),
                "certified_belief_tilt_rad": float(tilt[certified_belief_sample_index]),
                "terminal_tilt_rad": float(tilt[-1]),
                "maximum_pre_enable_tilt_rad": float(
                    np.max(tilt[: model_enable_sample_index + 1])
                ),
                "release_altitude_m": float(state_array[0, 2]),
                "model_enable_altitude_m": float(
                    state_array[model_enable_sample_index, 2]
                ),
                "certified_belief_altitude_m": float(
                    state_array[certified_belief_sample_index, 2]
                ),
                "terminal_altitude_m": float(state_array[-1, 2]),
                "minimum_altitude_m": float(np.min(state_array[:, 2])),
                "maximum_altitude_m": float(np.max(state_array[:, 2])),
                "maximum_horizontal_excursion_m": float(np.max(horizontal_excursion)),
                "terminal_vertical_velocity_m_s": float(state_array[-1, 5]),
                "sustained_hover_start_time_s": (
                    None
                    if sustained_hover_start_index is None
                    else float(timestamps[sustained_hover_start_index])
                ),
                "sustained_hover_duration_s": sustained_hover_duration_s,
                "sustained_hover_thresholds": {
                    "speed_m_s": 0.10,
                    "angular_rate_rad_s": 0.10,
                    "absolute_vertical_speed_m_s": 0.05,
                    "tilt_rad": 0.05,
                },
                "commands_finite": commands_finite,
                "commands_bounded": commands_bounded,
                "pre_enable_commands_exactly_zero": pre_enable_commands_zero,
            },
            "observations": {
                "pre_enable_commands_exactly_zero": pre_enable_commands_zero,
                "first_supported_control_began_before_certification": (
                    first_supported_control_sample_index < certified_belief_sample_index
                ),
                "working_belief_updated_for_every_post_enable_interval": (
                    working_belief.interval_count == online_step_count
                ),
                "certified_command_evidence_rank_is_four": (
                    certified_belief.command_evidence_rank == 4
                ),
                "certified_angular_effect_rank_is_three": (
                    certified_belief.angular_effect_rank == 3
                ),
                "hover_error_below_2_percent": hover_relative_error < 0.02,
                "terminal_velocity_below_1_percent_of_release": (
                    terminal_velocity_ratio < 0.01
                ),
                "terminal_rate_below_2_percent_of_release": (
                    terminal_rate_ratio < 0.02
                ),
                "terminal_speed_below_0_02_m_s": bool(velocity_norm[-1] < 0.02),
                "terminal_rate_below_0_03_rad_s": bool(rate_norm[-1] < 0.03),
                "terminal_vertical_speed_below_0_01_m_s": bool(
                    abs(state_array[-1, 5]) < 0.01
                ),
                "terminal_tilt_below_0_01_rad": bool(tilt[-1] < 0.01),
                "sustained_hover_exceeds_3_s": sustained_hover_duration_s >= 3.0,
                "minimum_altitude_above_1_m": bool(np.min(state_array[:, 2]) > 1.0),
                "all_values_finite": bool(
                    np.all(np.isfinite(state_array)) and commands_finite
                ),
                "all_commands_bounded": commands_bounded,
                "no_reset_after_release": True,
            },
            "limitations": [
                "The simulator injects a release state; hand contact and separation are not modeled.",
                "Exact simulator state and measured applied rotor state are used.",
                "The first post-enable command is centered at the known command-bounds midpoint.",
                "The controller regulates velocity, attitude, and rates but not position.",
                "Initial persistent-belief admission checks rank support, authority, sample count, and hover feasibility; it is not disjoint predictive validation.",
                "The admitted belief is retained after independent closed-loop excitation fades; later candidate replacement is disabled until independent predictive validation is implemented.",
                "No sensor noise, estimator delay, packet loss, or firmware scheduling is included.",
                "No real vehicle, propeller proximity, or human-release safety claim is made.",
            ],
        }
        report["observations"]["gate_passed"] = all(report["observations"].values())
        json.dumps(report, allow_nan=False)
        trace = CrazyflowThrowTrace(
            sample_period_s=plant.sample_period_s,
            model_enable_sample_index=model_enable_sample_index,
            first_supported_control_sample_index=(first_supported_control_sample_index),
            certified_belief_sample_index=certified_belief_sample_index,
            timestamps_s=np.asarray(timestamps),
            states=state_array,
            applied_motor_commands=applied_array,
            requested_motor_commands=requested_array,
            working_interval_counts=np.asarray(working_interval_counts),
            command_evidence_ranks=np.asarray(command_evidence_ranks),
            angular_effect_ranks=np.asarray(angular_effect_ranks),
            collective_authority=np.asarray(collective_authority),
            angular_axis_authority=np.asarray(angular_axis_authority),
            certified_control_active=np.asarray(certified_control_active),
        )
        return CrazyflowThrowRun(report=report, trace=trace)
    finally:
        plant.close()


def run_crazyflow_throw_benchmark() -> dict[str, Any]:
    """Return the fixed continuous throw diagnostic report."""

    return run_crazyflow_throw_trial().report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the unpowered-throw online Crazyflow diagnostic."
    )
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = run_crazyflow_throw_benchmark()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
