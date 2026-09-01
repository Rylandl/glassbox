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
class CrazyflowThrowScenario:
    """One deterministic hidden-airframe and release configuration."""

    name: str = "canonical"
    arm_length_ratio: float = 1.25
    release_height_m: float = 1.2
    world_velocity_m_s: tuple[float, float, float] = (1.0, -0.6, 10.0)
    angular_velocity_rad_s: tuple[float, float, float] = (0.8, -0.6, 0.4)
    roll_rad: float = 0.20
    pitch_rad: float = -0.15

    def __post_init__(self) -> None:
        vector = np.asarray(
            (
                self.arm_length_ratio,
                self.release_height_m,
                *self.world_velocity_m_s,
                *self.angular_velocity_rad_s,
                self.roll_rad,
                self.pitch_rad,
            ),
            dtype=np.float64,
        )
        if not self.name:
            raise ValueError("throw scenario name cannot be empty")
        if vector.shape != (10,) or not np.all(np.isfinite(vector)):
            raise ValueError("throw scenario values must be finite")
        if self.arm_length_ratio <= 0.0 or self.release_height_m <= 0.0:
            raise ValueError("throw arm ratio and release height must be positive")


CRAZYFLOW_THROW_CAMPAIGN_SCENARIOS = (
    CrazyflowThrowScenario(),
    CrazyflowThrowScenario(
        name="shorter_arms_high_release",
        arm_length_ratio=1.15,
        release_height_m=2.5,
    ),
    CrazyflowThrowScenario(
        name="long_arms_cross_axis_tumble",
        arm_length_ratio=1.35,
        release_height_m=2.5,
        world_velocity_m_s=(0.5, 0.8, 10.0),
        angular_velocity_rad_s=(0.5, 0.7, -0.3),
        roll_rad=0.15,
        pitch_rad=0.12,
    ),
    CrazyflowThrowScenario(
        name="milder_low_energy_release",
        release_height_m=2.5,
        world_velocity_m_s=(0.7, -0.4, 9.0),
        angular_velocity_rad_s=(0.6, -0.4, 0.3),
        roll_rad=0.15,
        pitch_rad=-0.10,
    ),
    CrazyflowThrowScenario(
        name="reversed_tumble",
        release_height_m=2.5,
        world_velocity_m_s=(-1.0, 0.6, 10.0),
        angular_velocity_rad_s=(-0.8, 0.6, -0.4),
        roll_rad=-0.20,
        pitch_rad=0.15,
    ),
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


def run_crazyflow_throw_trial(
    scenario: CrazyflowThrowScenario | None = None,
) -> CrazyflowThrowRun:
    """Tumble unpowered for one second, then identify and stabilize online."""

    scenario = CrazyflowThrowScenario() if scenario is None else scenario
    plant = CrazyflowPlant(CrazyflowPlantConfig(control_frequency_hz=100))
    try:
        plant.set_arm_length_ratio(scenario.arm_length_ratio)
        identifier = RecursiveBootstrapIdentifier()
        controller = ProgressiveBootstrapController(identifier.config)
        minimum = np.asarray(identifier.config.command_minimum)
        maximum = np.asarray(identifier.config.command_maximum)
        release_state = _initial_state(
            world_velocity_m_s=scenario.world_velocity_m_s,
            angular_velocity_rad_s=scenario.angular_velocity_rad_s,
            roll_rad=scenario.roll_rad,
            pitch_rad=scenario.pitch_rad,
        )
        release_state[2] = scenario.release_height_m
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
                exploration_belief=(
                    None if identifier.certified_belief is None else identifier.belief
                ),
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
        validation_reports = identifier.validation_history
        initial_admission = next(
            (
                validation
                for validation in validation_reports
                if validation.initial_admission and validation.accepted
            ),
            None,
        )
        if initial_admission is None:
            raise RuntimeError("certified belief has no accepted initial validation")
        replacement_commit_count = sum(
            validation.accepted and not validation.initial_admission
            for validation in validation_reports
        )
        report = {
            "artifact_type": "glassbox_crazyflow_online_throw_recovery_diagnostic",
            "schema_version": 3,
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
                "initial_control_belief_uses_disjoint_predictive_validation": True,
                "post_admission_candidate_replacement_implemented": True,
                "control_authority_uses_information_and_supported_covariance": True,
                "active_excitation_targets_weak_information_directions": True,
                "feedback_uses_committed_belief_after_admission": True,
                "exploration_uses_continuously_updated_working_belief": True,
                "airframe_parameter_prior_used": False,
                "canonical_motor_mixer_supplied_to_identifier": False,
                "hover_command_supplied_to_identifier": False,
                "command_bounds_and_channel_shape_known": True,
                "measured_applied_motor_state_used": True,
                "hidden_plant_values_used_only_for_post_run_evaluation": True,
            },
            "configuration": {
                "scenario_name": scenario.name,
                "hidden_arm_length_ratio": scenario.arm_length_ratio,
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
                    "minimum_information_singular_value": (
                        identifier.config.minimum_information_singular_value
                    ),
                    "full_authority_information_singular_value": (
                        identifier.config.full_authority_information_singular_value
                    ),
                    "minimum_effect_signal_to_noise": (
                        identifier.config.minimum_effect_signal_to_noise
                    ),
                    "full_authority_effect_signal_to_noise": (
                        identifier.config.full_authority_effect_signal_to_noise
                    ),
                    "collective_residual_std_floor_m_s2": (
                        identifier.config.collective_residual_std_floor_m_s2
                    ),
                    "angular_residual_std_floor_rad_s2": (
                        identifier.config.angular_residual_std_floor_rad_s2
                    ),
                    "minimum_certification_interval_count": (
                        identifier.config.minimum_certification_interval_count
                    ),
                    "validation_interval_count": (
                        identifier.config.validation_interval_count
                    ),
                    "minimum_validation_improvement": (
                        identifier.config.minimum_validation_improvement
                    ),
                    "maximum_model_movement_fraction": (
                        identifier.config.maximum_model_movement_fraction
                    ),
                    "proposal_cooldown_interval_count": (
                        identifier.config.proposal_cooldown_interval_count
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
                    "maximum_committed_excitation_fraction": (
                        controller.config.maximum_committed_excitation_fraction
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
                "pending_proposal_at_trial_end": identifier.pending_proposal,
                "accepted_update_count": identifier.accepted_update_count,
                "rejected_update_count": identifier.rejected_update_count,
                "accepted_replacement_count": replacement_commit_count,
                "initial_admission_validation": initial_admission.to_dict(),
                "last_validation": validation_reports[-1].to_dict(),
                "validation_history": [
                    validation.to_dict() for validation in validation_reports
                ],
                "control_belief_admission_contract": {
                    "minimum_interval_count": (
                        identifier.config.minimum_certification_interval_count
                    ),
                    "required_command_evidence_rank": 4,
                    "required_angular_effect_rank": 3,
                    "minimum_exploration_completion": 0.75,
                    "feasible_hover_command_required": True,
                    "future_validation_interval_count": (
                        identifier.config.validation_interval_count
                    ),
                    "minimum_validation_improvement": (
                        identifier.config.minimum_validation_improvement
                    ),
                    "maximum_replacement_model_movement_fraction": (
                        identifier.config.maximum_model_movement_fraction
                    ),
                    "disjoint_predictive_validation_required": True,
                    "claim": "frozen_candidate_prequential_future_validation",
                },
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
                "initial_admission_scored_on_future_intervals": (
                    initial_admission.validation_interval_count
                    == identifier.config.validation_interval_count
                    and initial_admission.candidate_interval_count
                    + initial_admission.validation_interval_count
                    == certified_belief_sample_index - model_enable_sample_index
                ),
                "at_least_one_belief_replacement_committed": (
                    replacement_commit_count >= 1
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
                "Candidate admission uses a short local future-prediction window; it is not a calibrated probability of flight safety.",
                "The nuisance-only initial reference and committed-model replacement reference are local one-step predictors, not full trajectory validators.",
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


def run_crazyflow_throw_campaign(
    scenarios: tuple[CrazyflowThrowScenario, ...] = CRAZYFLOW_THROW_CAMPAIGN_SCENARIOS,
) -> dict[str, Any]:
    """Run the fixed development campaign without hiding failed gates."""

    if not scenarios:
        raise ValueError("throw campaign needs at least one scenario")
    if len({scenario.name for scenario in scenarios}) != len(scenarios):
        raise ValueError("throw campaign scenario names must be unique")
    cases: list[dict[str, Any]] = []
    for scenario in scenarios:
        run = run_crazyflow_throw_trial(scenario)
        report = run.report
        recovery = report["continuous_throw"]
        observations = report["observations"]
        identification = report["identification"]
        cases.append(
            {
                "scenario": {
                    "name": scenario.name,
                    "hidden_arm_length_ratio": scenario.arm_length_ratio,
                    "release_height_m": scenario.release_height_m,
                    "release_world_velocity_m_s": list(scenario.world_velocity_m_s),
                    "release_angular_velocity_rad_s": list(
                        scenario.angular_velocity_rad_s
                    ),
                    "release_roll_rad": scenario.roll_rad,
                    "release_pitch_rad": scenario.pitch_rad,
                },
                "gate_passed": observations["gate_passed"],
                "failed_observations": [
                    name
                    for name, passed in observations.items()
                    if name != "gate_passed" and not passed
                ],
                "timing": {
                    "first_supported_control_time_s": report["timing"][
                        "first_supported_control_time_s"
                    ],
                    "initial_belief_admission_time_s": report["timing"][
                        "certified_belief_time_s"
                    ],
                },
                "identification": {
                    "accepted_update_count": identification["accepted_update_count"],
                    "rejected_update_count": identification["rejected_update_count"],
                    "accepted_replacement_count": identification[
                        "accepted_replacement_count"
                    ],
                    "terminal_working_command_evidence_rank": identification[
                        "terminal_working_belief"
                    ]["command_evidence_rank"],
                    "terminal_working_angular_effect_rank": identification[
                        "terminal_working_belief"
                    ]["angular_effect_rank"],
                },
                "recovery": {
                    name: recovery[name]
                    for name in (
                        "terminal_velocity_norm_m_s",
                        "terminal_angular_rate_norm_rad_s",
                        "terminal_tilt_rad",
                        "terminal_vertical_velocity_m_s",
                        "minimum_altitude_m",
                        "maximum_altitude_m",
                        "maximum_horizontal_excursion_m",
                        "sustained_hover_duration_s",
                        "commands_finite",
                        "commands_bounded",
                    )
                },
            }
        )
    passing_cases = [case for case in cases if case["gate_passed"]]
    recovery_cases = [case["recovery"] for case in cases]
    result = {
        "artifact_type": "glassbox_crazyflow_online_throw_development_campaign",
        "schema_version": 1,
        "semantics": {
            "diagnostic_only": True,
            "flight_safety_claim": False,
            "fixed_scenarios": scenarios == CRAZYFLOW_THROW_CAMPAIGN_SCENARIOS,
            "held_out_after_controller_tuning": False,
            "failed_gates_retained": True,
            "exact_simulator_state_used": True,
            "sensor_noise_or_delay_modeled": False,
        },
        "case_count": len(cases),
        "cases": cases,
        "aggregate": {
            "passing_case_count": len(passing_cases),
            "failing_case_count": len(cases) - len(passing_cases),
            "pass_fraction": len(passing_cases) / len(cases),
            "all_commands_finite_and_bounded": all(
                recovery["commands_finite"] and recovery["commands_bounded"]
                for recovery in recovery_cases
            ),
            "worst_terminal_velocity_norm_m_s": max(
                recovery["terminal_velocity_norm_m_s"] for recovery in recovery_cases
            ),
            "worst_terminal_angular_rate_norm_rad_s": max(
                recovery["terminal_angular_rate_norm_rad_s"]
                for recovery in recovery_cases
            ),
            "worst_terminal_tilt_rad": max(
                recovery["terminal_tilt_rad"] for recovery in recovery_cases
            ),
            "minimum_sustained_hover_duration_s": min(
                recovery["sustained_hover_duration_s"] for recovery in recovery_cases
            ),
            "minimum_altitude_m": min(
                recovery["minimum_altitude_m"] for recovery in recovery_cases
            ),
        },
        "limitations": [
            "This is a small deterministic development campaign, not held-out validation.",
            "The scenarios vary arm length and release state but not sensors, timing, motors, battery, or aerodynamics.",
            "A passed recovery gate is not a calibrated probability of flight safety.",
        ],
    }
    json.dumps(result, allow_nan=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the unpowered-throw online Crazyflow diagnostic."
    )
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--campaign",
        action="store_true",
        help="run the fixed multi-configuration development campaign",
    )
    args = parser.parse_args()
    report = (
        run_crazyflow_throw_campaign()
        if args.campaign
        else run_crazyflow_throw_benchmark()
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
