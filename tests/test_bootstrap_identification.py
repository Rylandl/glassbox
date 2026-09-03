from __future__ import annotations

from dataclasses import fields, replace

import numpy as np
import pytest

from glassbox.control.bootstrap_identification import (
    BootstrapExcitationConfig,
    BootstrapIdentificationConfig,
    BootstrapModelNotReadyError,
    BootstrapMultirotorIdentifier,
    plan_bootstrap_excitation,
)
from glassbox.control.online_bootstrap import (
    ProgressiveBootstrapController,
    RecursiveBootstrapBelief,
    RecursiveBootstrapConfig,
    RecursiveBootstrapIdentifier,
)
from glassbox.core.dynamics import GRAVITY_M_S2, MOTOR_MIXER


def _excitation(interval_count: int, *, collective_only: bool = False) -> np.ndarray:
    if collective_only:
        phase = np.arange(interval_count, dtype=np.float64)
        collective = 0.5 + 0.1 * np.sin(0.9 * phase + 0.2)
        return np.repeat(collective[:, None], 4, axis=1)
    generator = np.random.default_rng(7)
    return 0.5 + np.clip(
        0.09 * generator.standard_normal((interval_count, 4)),
        -0.16,
        0.16,
    )


def _linear_hidden_plant(
    commands: np.ndarray,
    *,
    sample_period_s: float = 0.02,
    effect_scales: np.ndarray | None = None,
    acceleration_noise_m_s2: float = 0.0,
    angular_noise_rad_s2: float = 0.0,
    noise_seed: int = 3,
    thrust_effect: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    thrust_effect = (
        np.asarray((4.8, 5.0, 5.2, 4.9))
        if thrust_effect is None
        else np.asarray(thrust_effect, dtype=np.float64)
    )
    angular_effect = np.diag((34.0, 31.0, 11.0)) @ np.asarray(MOTOR_MIXER)
    angular_rate_coefficient = -np.diag((0.4, 0.5, 0.25))
    thrust_intercept = 0.08
    states = np.zeros((len(commands) + 1, 13), dtype=np.float64)
    states[:, 6] = 1.0
    timestamps = np.arange(len(states), dtype=np.float64) * sample_period_s
    scales = (
        np.ones(len(commands), dtype=np.float64)
        if effect_scales is None
        else np.asarray(effect_scales, dtype=np.float64)
    )
    if scales.shape != (len(commands),):
        raise ValueError("effect_scales must be interval-aligned")
    noise = np.random.default_rng(noise_seed)
    for index, command in enumerate(commands):
        velocity = states[index, 3:6]
        angular_velocity = states[index, 10:13]
        body_specific_force_z = (
            scales[index] * (thrust_effect @ command)
            + thrust_intercept
            - 0.12 * velocity[2]
        )
        acceleration = np.asarray((0.0, 0.0, body_specific_force_z - GRAVITY_M_S2))
        acceleration = acceleration + acceleration_noise_m_s2 * noise.standard_normal(3)
        angular_acceleration = (
            scales[index] * (angular_effect @ command)
            + angular_rate_coefficient @ angular_velocity
            + angular_noise_rad_s2 * noise.standard_normal(3)
        )
        states[index + 1, 0:3] = (
            states[index, 0:3]
            + sample_period_s * velocity
            + 0.5 * sample_period_s**2 * acceleration
        )
        states[index + 1, 3:6] = velocity + sample_period_s * acceleration
        states[index + 1, 10:13] = (
            angular_velocity + sample_period_s * angular_acceleration
        )
    return timestamps, states, thrust_effect, angular_effect


def _update_recursive_identifier(
    commands: np.ndarray,
    config: RecursiveBootstrapConfig | None = None,
    **plant: float,
) -> tuple[RecursiveBootstrapIdentifier, np.ndarray, np.ndarray]:
    timestamps, states, thrust_effect, angular_effect = _linear_hidden_plant(
        commands,
        **plant,
    )
    identifier = RecursiveBootstrapIdentifier(config)
    for index, command in enumerate(commands):
        identifier.update(
            states[index],
            states[index + 1],
            command,
            timestamps[index + 1] - timestamps[index],
        )
    return identifier, thrust_effect, angular_effect


def _first_supported_interval(
    commands: np.ndarray,
    config: RecursiveBootstrapConfig | None = None,
    **plant: float,
) -> int | None:
    """Interval at which one identifier first meets its own support rule."""

    timestamps, states, _, _ = _linear_hidden_plant(commands, **plant)
    identifier = RecursiveBootstrapIdentifier(config)
    for index, command in enumerate(commands):
        identifier.update(
            states[index],
            states[index + 1],
            command,
            timestamps[index + 1] - timestamps[index],
        )
        if identifier.working_belief_supported:
            return index + 1
    return None


def test_recursive_bootstrap_updates_every_interval_and_certifies_support() -> None:
    commands = _excitation(80)

    identifier, thrust_effect, angular_effect = _update_recursive_identifier(commands)
    belief = identifier.belief
    certified = identifier.certified_belief

    assert belief.interval_count == len(commands)
    assert belief.command_evidence_rank == 4
    assert belief.angular_effect_rank == 3
    assert certified is not None
    assert certified.interval_count == 48
    assert identifier.predictive_belief is certified
    assert identifier.control_belief is identifier.predictive_belief
    assert len(identifier.validation_history) == 1
    validation = identifier.validation_history[0]
    assert validation.candidate_interval_count == 48
    assert validation.validation_interval_count == 16
    assert validation.initial_admission
    assert validation.accepted
    assert validation.reason == "initial_prequential_admission"
    assert identifier.accepted_update_count == 1
    assert identifier.rejected_update_count == 0
    np.testing.assert_allclose(
        belief.collective_acceleration_per_command,
        thrust_effect,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        belief.angular_acceleration_per_command,
        angular_effect,
        atol=1e-8,
    )
    assert belief.to_dict()["airframe_parameter_prior_used"] is False
    assert belief.to_dict()["canonical_motor_mixer_assumed"] is False


def test_recursive_candidate_is_frozen_then_scored_on_future_intervals() -> None:
    commands = _excitation(64)
    timestamps, states, _, _ = _linear_hidden_plant(commands)
    identifier = RecursiveBootstrapIdentifier()

    for index, command in enumerate(commands[:48]):
        identifier.update(
            states[index],
            states[index + 1],
            command,
            timestamps[index + 1] - timestamps[index],
        )
    assert identifier.pending_proposal
    assert identifier.certified_belief is None
    frozen_candidate = identifier.belief

    for index, command in enumerate(commands[48:63], start=48):
        identifier.update(
            states[index],
            states[index + 1],
            command,
            timestamps[index + 1] - timestamps[index],
        )
    assert identifier.pending_proposal
    assert identifier.certified_belief is None
    assert identifier.belief is not frozen_candidate

    identifier.update(
        states[63],
        states[64],
        commands[63],
        timestamps[64] - timestamps[63],
    )

    assert identifier.certified_belief is frozen_candidate
    report = identifier.validation_history[0]
    assert report.candidate_interval_count == 48
    assert report.validation_interval_count == 16
    assert report.accepted


def test_recursive_replacement_cannot_claim_sub_noise_information_gain() -> None:
    commands = _excitation(100)

    identifier, _, _ = _update_recursive_identifier(commands)

    replacements = [
        report
        for report in identifier.validation_history
        if not report.initial_admission
    ]
    assert replacements
    assert all(not report.accepted for report in replacements)
    assert all(
        report.reason == "prequential_improvement_not_demonstrated"
        for report in replacements
    )
    assert identifier.certified_belief is not None
    assert identifier.certified_belief.interval_count == 48


def test_recursive_replacement_rejects_excessive_model_movement() -> None:
    commands = _excitation(100)
    scales = np.ones(len(commands))
    scales[64:] = 4.0
    timestamps, states, _, _ = _linear_hidden_plant(
        commands,
        effect_scales=scales,
    )
    identifier = RecursiveBootstrapIdentifier()
    for index, command in enumerate(commands):
        identifier.update(
            states[index],
            states[index + 1],
            command,
            timestamps[index + 1] - timestamps[index],
        )

    movement_rejections = [
        report
        for report in identifier.validation_history
        if report.reason == "model_movement_exceeded"
    ]
    assert movement_rejections
    assert all(not report.accepted for report in movement_rejections)
    assert all(
        report.model_movement_fraction
        > identifier.config.maximum_model_movement_fraction
        for report in movement_rejections
    )


def test_recursive_belief_exposes_supported_covariance_and_information() -> None:
    identifier, _, _ = _update_recursive_identifier(_excitation(80))
    belief = identifier.belief

    assert belief.normalized_command_information.shape == (4, 4)
    assert belief.supported_collective_effect_covariance.shape == (4, 4)
    assert belief.supported_angular_effect_covariance.shape == (3, 4, 4)
    assert np.all(np.linalg.eigvalsh(belief.normalized_command_information) >= -1e-10)
    assert np.all(
        np.linalg.eigvalsh(belief.supported_collective_effect_covariance) >= -1e-10
    )
    assert np.all(
        np.linalg.eigvalsh(belief.supported_angular_effect_covariance) >= -1e-10
    )
    assert belief.minimum_supported_information_singular_value > 0.0
    assert 0.0 < belief.information_authority <= 1.0
    assert belief.collective_effect_signal_to_noise > 0.0
    assert np.all(belief.angular_effect_signal_to_noise > 0.0)
    assert belief.to_dict()["effect_covariance_scope"] == "supported_subspace_only"


def test_recursive_authority_tracks_information_not_elapsed_interval_count() -> None:
    generator = np.random.default_rng(3)
    patterns = generator.standard_normal((80, 4))
    config = RecursiveBootstrapConfig(
        minimum_normalized_command_rms=0.0001,
        minimum_information_singular_value=0.01,
        full_authority_information_singular_value=0.5,
    )
    beliefs = []
    for amplitude in (0.0035, 0.09):
        commands = 0.5 + amplitude * patterns
        timestamps, states, _, _ = _linear_hidden_plant(commands)
        identifier = RecursiveBootstrapIdentifier(config)
        for index, command in enumerate(commands):
            identifier.update(
                states[index],
                states[index + 1],
                command,
                timestamps[index + 1] - timestamps[index],
            )
        beliefs.append(identifier.belief)

    weak, strong = beliefs
    assert weak.interval_count == strong.interval_count == 80
    assert weak.command_evidence_rank == strong.command_evidence_rank == 4
    assert weak.information_authority < 0.05
    assert strong.information_authority == pytest.approx(1.0)
    assert weak.collective_authority < strong.collective_authority
    assert np.max(weak.angular_axis_authority) < np.min(strong.angular_axis_authority)


def test_recursive_rank_deficiency_never_certifies_unobserved_axes() -> None:
    commands = _excitation(80, collective_only=True)

    identifier, _, _ = _update_recursive_identifier(commands)
    belief = identifier.belief

    assert belief.command_evidence_rank <= 1
    assert belief.angular_effect_rank < 3
    assert identifier.certified_belief is None
    assert identifier.predictive_belief is belief
    assert np.max(belief.angular_axis_authority) < 0.26


def test_progressive_controller_optimizes_information_before_support() -> None:
    identifier = RecursiveBootstrapIdentifier()
    controller = ProgressiveBootstrapController(identifier.config)
    state = np.zeros(13, dtype=np.float64)
    state[6] = 1.0

    decision = controller.command(
        state,
        identifier.predictive_belief,
        previous_command=np.zeros(4),
    )

    assert decision.information_action_fraction > 0.0
    assert decision.information_reward > 0.0
    assert decision.estimated_information_gain > 0.0
    assert decision.objective_value == pytest.approx(
        decision.stabilization_cost
        - decision.information_reward
        + decision.uncertainty_cost
        + decision.altitude_risk_cost
    )
    assert decision.collective_authority == 0.0
    np.testing.assert_allclose(decision.angular_axis_authority, 0.0)
    assert decision.predicted_world_velocity_m_s.shape == (3,)
    assert decision.predicted_angular_velocity_rad_s.shape == (3,)
    assert np.all(decision.command >= 0.0)
    assert np.all(decision.command <= 1.0)


def test_progressive_controller_targets_weak_information_and_caps_live_probe() -> None:
    identifier, _, _ = _update_recursive_identifier(_excitation(80))
    controller = ProgressiveBootstrapController(identifier.config)
    state = np.zeros(13, dtype=np.float64)
    state[6] = 1.0
    feedback_belief = identifier.predictive_belief
    isotropic = replace(
        identifier.belief,
        normalized_command_information=np.eye(4),
        exploration_completion=0.0,
    )
    weak_first_channel = replace(
        isotropic,
        normalized_command_information=np.diag((2.5e-5, 1.0, 1.0, 1.0)),
    )

    no_information_decision = controller.command(
        state,
        feedback_belief,
        previous_command=np.full(4, 0.5),
        online_belief=replace(isotropic, exploration_completion=1.0),
    )
    isotropic_decision = controller.command(
        state,
        feedback_belief,
        previous_command=np.full(4, 0.5),
        online_belief=isotropic,
    )
    weak_decision = controller.command(
        state,
        feedback_belief,
        previous_command=np.full(4, 0.5),
        online_belief=weak_first_channel,
    )

    weak_delta = weak_decision.command - no_information_decision.command
    isotropic_delta = isotropic_decision.command - no_information_decision.command
    assert abs(weak_delta[0]) / np.linalg.norm(weak_delta[1:]) > abs(
        isotropic_delta[0]
    ) / np.linalg.norm(isotropic_delta[1:])
    assert weak_decision.information_action_fraction == pytest.approx(
        controller.config.maximum_committed_excitation_fraction
    )
    assert np.all(weak_decision.command >= 0.0)
    assert np.all(weak_decision.command <= 1.0)


def test_bootstrap_fit_recovers_supported_input_effects_without_mixer_prior() -> None:
    config = BootstrapIdentificationConfig(interval_count=32)
    commands = _excitation(config.interval_count)
    timestamps, states, thrust_effect, angular_effect = _linear_hidden_plant(commands)
    identifier = BootstrapMultirotorIdentifier(config)
    identifier.prewarm()

    result = identifier.fit(timestamps, states, commands)

    assert result.collective_command_evidence_rank == 4
    assert result.command_evidence_rank == 4
    assert result.angular_effect_rank == 3
    assert result.collective_support_fraction == pytest.approx(1.0, abs=1e-5)
    np.testing.assert_allclose(
        result.collective_acceleration_per_command,
        thrust_effect,
        rtol=0.02,
        atol=0.03,
    )
    np.testing.assert_allclose(
        result.angular_acceleration_per_command,
        angular_effect,
        rtol=0.03,
        atol=0.08,
    )
    assert result.hover_command is not None
    assert result.hover_command[0] == pytest.approx(
        (GRAVITY_M_S2 - 0.08) / np.sum(thrust_effect),
        rel=0.01,
    )
    assert result.ready_for_hover
    assert result.ready_for_rate_arrest
    assert result.ready
    assert result.wall_time_s < 0.1
    report = result.to_dict()
    assert not report["airframe_parameter_prior_used"]
    assert not report["canonical_motor_mixer_assumed"]
    assert report["applied_motor_state_required"]


def test_rate_arrest_allocation_reduces_predicted_angular_acceleration_error() -> None:
    config = BootstrapIdentificationConfig(interval_count=32)
    commands = _excitation(config.interval_count)
    timestamps, states, _, _ = _linear_hidden_plant(commands)
    result = BootstrapMultirotorIdentifier(config).fit(timestamps, states, commands)
    angular_velocity = np.asarray((0.8, -0.6, 0.4))
    desired = -np.asarray((2.0, 2.0, 1.0)) * angular_velocity
    arrest = result.rate_arrest_command(
        angular_velocity,
        angular_rate_gain=(2.0, 2.0, 1.0),
    )

    assert np.linalg.norm(
        arrest.predicted_control_angular_acceleration_rad_s2 - desired
    ) < np.linalg.norm(desired)
    assert np.all(arrest.command >= 0.0)
    assert np.all(arrest.command <= 1.0)
    assert not arrest.saturated


def test_attitude_rate_arrest_uses_identified_allocation_without_mixer() -> None:
    config = BootstrapIdentificationConfig(interval_count=32)
    commands = _excitation(config.interval_count)
    timestamps, states, _, _ = _linear_hidden_plant(commands)
    result = BootstrapMultirotorIdentifier(config).fit(timestamps, states, commands)
    roll_rad = 0.20
    quaternion = np.asarray((np.cos(roll_rad / 2.0), np.sin(roll_rad / 2.0), 0.0, 0.0))

    arrest = result.attitude_rate_arrest_command(
        quaternion,
        angular_velocity_rad_s=(0.0, 0.0, 0.0),
        attitude_gain=(10.0, 10.0),
    )

    assert arrest.desired_angular_acceleration_rad_s2[0] < 0.0
    assert arrest.predicted_control_angular_acceleration_rad_s2[0] < 0.0
    np.testing.assert_allclose(
        arrest.predicted_control_angular_acceleration_rad_s2,
        arrest.desired_angular_acceleration_rad_s2,
        atol=1e-4,
    )


def test_velocity_arrest_vectors_thrust_against_world_velocity() -> None:
    config = BootstrapIdentificationConfig(interval_count=32)
    commands = _excitation(config.interval_count)
    timestamps, states, _, _ = _linear_hidden_plant(commands)
    result = BootstrapMultirotorIdentifier(config).fit(timestamps, states, commands)

    arrest = result.velocity_attitude_rate_arrest_command(
        world_velocity_m_s=(1.0, -0.5, 0.4),
        quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        angular_velocity_rad_s=(0.0, 0.0, 0.0),
    )

    assert arrest.desired_world_acceleration_m_s2[0] < 0.0
    assert arrest.desired_world_acceleration_m_s2[1] > 0.0
    assert arrest.desired_world_acceleration_m_s2[2] < 0.0
    assert arrest.desired_thrust_direction_world[0] < 0.0
    assert np.all(arrest.command >= 0.0)
    assert np.all(arrest.command <= 1.0)


def test_rank_deficient_evidence_never_claims_three_axis_authority() -> None:
    config = BootstrapIdentificationConfig(interval_count=32)
    commands = _excitation(config.interval_count, collective_only=True)
    timestamps, states, _, _ = _linear_hidden_plant(commands)

    result = BootstrapMultirotorIdentifier(config).fit(timestamps, states, commands)

    assert result.collective_command_evidence_rank == 1
    assert result.command_evidence_rank <= 1
    assert result.angular_effect_rank < 3
    assert not result.ready_for_rate_arrest
    with pytest.raises(BootstrapModelNotReadyError, match="three-axis"):
        result.rate_arrest_command((0.1, 0.2, 0.3), reference_command=(0.5,) * 4)

    plan = plan_bootstrap_excitation(result)
    assert plan.selection_reason == "least_supported_command_direction"
    assert plan.target_angular_axis is None
    assert plan.commands.shape == (8, 4)
    np.testing.assert_allclose(
        plan.commands[0] + plan.commands[1],
        2.0 * plan.center_command,
    )


def test_follow_up_excitation_targets_weakest_validated_angular_output() -> None:
    config = BootstrapIdentificationConfig(interval_count=32)
    commands = _excitation(config.interval_count)
    timestamps, states, _, _ = _linear_hidden_plant(commands)
    result = BootstrapMultirotorIdentifier(config).fit(timestamps, states, commands)

    plan = plan_bootstrap_excitation(
        result,
        BootstrapExcitationConfig(
            interval_count=6,
            amplitude_fraction_of_command_span=0.05,
        ),
    )

    assert plan.selection_reason == "weakest_validated_angular_output"
    assert plan.target_angular_axis == int(
        np.argmin(result.angular_validation_improvement)
    )
    assert np.max(np.abs(plan.normalized_direction)) == pytest.approx(1.0)
    assert np.all(plan.commands >= result.command_minimum)
    assert np.all(plan.commands <= result.command_maximum)


def test_unexcited_evidence_returns_an_unready_result() -> None:
    config = BootstrapIdentificationConfig(interval_count=16)
    commands = np.full((config.interval_count, 4), 0.5)
    timestamps, states, _, _ = _linear_hidden_plant(commands)

    result = BootstrapMultirotorIdentifier(config).fit(timestamps, states, commands)

    assert result.collective_command_evidence_rank == 0
    assert result.command_evidence_rank == 0
    assert not result.ready_for_hover
    assert not result.ready_for_rate_arrest


def test_bootstrap_fit_rejects_requested_or_malformed_actuator_history() -> None:
    config = BootstrapIdentificationConfig(interval_count=16)
    commands = _excitation(config.interval_count)
    timestamps, states, _, _ = _linear_hidden_plant(commands)
    identifier = BootstrapMultirotorIdentifier(config)

    with pytest.raises(ValueError, match="shape"):
        identifier.fit(timestamps, states, commands[:-1])
    invalid_timestamps = timestamps.copy()
    invalid_timestamps[4] = invalid_timestamps[3]
    with pytest.raises(ValueError, match="strictly increasing"):
        identifier.fit(invalid_timestamps, states, commands)
    invalid_commands = commands.copy()
    invalid_commands[0, 0] = 1.1
    with pytest.raises(ValueError, match="bounds"):
        identifier.fit(timestamps, states, invalid_commands)


def test_unexcited_nuisance_directions_are_never_inverted() -> None:
    config = BootstrapIdentificationConfig(interval_count=32)
    commands = _excitation(config.interval_count)
    identifier = BootstrapMultirotorIdentifier(config)
    clean_timestamps, clean_states, _, _ = _linear_hidden_plant(commands)
    noisy_timestamps, noisy_states, _, _ = _linear_hidden_plant(
        commands,
        acceleration_noise_m_s2=0.02,
        angular_noise_rad_s2=0.02,
    )

    clean = identifier.fit(clean_timestamps, clean_states, commands)
    noisy = identifier.fit(noisy_timestamps, noisy_states, commands)

    # The plant only ever moves along its body vertical, so the two lateral
    # velocity columns carry no excitation and must not be inverted at all.
    assert clean.collective_nuisance_rank == 2
    assert noisy.collective_nuisance_rank == 2
    assert clean.angular_nuisance_rank == 7
    np.testing.assert_allclose(
        clean.collective_velocity_coefficient,
        (0.0, 0.0, -0.12),
        atol=1e-3,
    )
    np.testing.assert_allclose(
        clean.angular_rate_coefficient,
        -np.diag((0.4, 0.5, 0.25)),
        atol=1e-3,
    )
    np.testing.assert_allclose(clean.angular_rate_product_coefficient, 0.0, atol=1e-3)
    np.testing.assert_allclose(
        noisy.collective_velocity_coefficient,
        (0.0, 0.0, -0.12),
        atol=0.25,
    )
    assert noisy.ready_for_hover and noisy.ready_for_rate_arrest


def test_recursive_lateral_velocity_coefficient_stays_near_zero_under_noise() -> None:
    commands = _excitation(200)

    clean, _, _ = _update_recursive_identifier(commands)
    noisy, _, _ = _update_recursive_identifier(
        commands,
        acceleration_noise_m_s2=0.01,
        angular_noise_rad_s2=0.01,
    )

    assert clean.belief.collective_nuisance_rank == 2
    assert noisy.belief.collective_nuisance_rank == 2
    np.testing.assert_allclose(
        clean.belief.collective_velocity_coefficient,
        (0.0, 0.0, -0.12),
        atol=1e-9,
    )
    np.testing.assert_allclose(
        noisy.belief.collective_velocity_coefficient[:2],
        0.0,
        atol=0.05,
    )
    assert noisy.belief.collective_velocity_coefficient[2] == pytest.approx(
        -0.12,
        abs=0.05,
    )
    assert noisy.belief.to_dict()["collective_nuisance_rank"] == 2
    assert noisy.belief.to_dict()["angular_nuisance_rank"] == 7


def test_recursive_update_refuses_a_non_finite_sample_and_keeps_its_belief() -> None:
    commands = _excitation(60)
    identifier, _, _ = _update_recursive_identifier(commands)
    before = identifier.belief
    corrupted = np.zeros(13, dtype=np.float64)
    corrupted[6] = 1.0
    corrupted[3] = np.nan

    returned = identifier.update(corrupted, np.zeros(13), np.full(4, 0.5), 0.02)

    assert returned is before
    assert identifier.belief is before
    assert identifier.rejected_sample_count == 1
    report = identifier.last_sample_report
    assert not report.accepted
    assert report.reason == "previous_state_not_finite"
    assert report.interval_count == before.interval_count
    assert report.to_dict()["accepted"] is False

    healthy = identifier.update(
        np.concatenate((np.zeros(6), (1.0, 0.0, 0.0, 0.0), np.zeros(3))),
        np.concatenate((np.zeros(6), (1.0, 0.0, 0.0, 0.0), np.zeros(3))),
        np.full(4, 0.5),
        0.02,
    )

    assert healthy.interval_count == before.interval_count + 1
    assert identifier.last_sample_report.accepted


def test_recursive_update_accepts_rounding_width_bound_overshoot() -> None:
    identifier = RecursiveBootstrapIdentifier()
    state = np.concatenate((np.zeros(6), (1.0, 0.0, 0.0, 0.0), np.zeros(3)))

    belief = identifier.update(state, state, np.full(4, 1.0 + 1e-7), 0.02)

    assert belief.interval_count == 1
    assert identifier.last_sample_report.accepted

    refused = identifier.update(state, state, np.full(4, 1.01), 0.02)

    assert refused.interval_count == 1
    assert identifier.last_sample_report.reason == "applied_command_outside_bounds"


def test_progressive_command_returns_an_unusable_hold_for_a_broken_state() -> None:
    identifier = RecursiveBootstrapIdentifier()
    controller = ProgressiveBootstrapController(identifier.config)
    broken = np.zeros(13, dtype=np.float64)
    broken[6] = 1.0
    broken[10] = np.nan

    decision = controller.command(
        broken,
        identifier.predictive_belief,
        previous_command=np.full(4, 0.42),
    )

    assert not decision.command_usable
    assert decision.reason == "state_not_finite"
    np.testing.assert_allclose(decision.command, 0.42)

    unbounded = controller.command(
        broken,
        identifier.predictive_belief,
        previous_command=np.full(4, np.nan),
    )

    assert not unbounded.command_usable
    np.testing.assert_allclose(unbounded.command, 0.5)

    degenerate = np.zeros(13, dtype=np.float64)
    flat = controller.command(
        degenerate,
        identifier.predictive_belief,
        previous_command=np.full(4, 2.0),
    )

    assert flat.reason == "state_quaternion_degenerate"
    np.testing.assert_allclose(flat.command, 1.0)

    healthy = np.zeros(13, dtype=np.float64)
    healthy[6] = 1.0
    assert controller.command(
        healthy,
        identifier.predictive_belief,
        previous_command=np.full(4, 0.42),
    ).command_usable


def test_recursive_config_rejects_a_forgetting_factor_below_one() -> None:
    with pytest.raises(ValueError, match="never regain it"):
        RecursiveBootstrapConfig(forgetting_factor=0.95)

    assert RecursiveBootstrapConfig().forgetting_factor == 1.0


def test_working_control_model_flies_the_working_belief_without_a_transaction() -> None:
    commands = _excitation(80)
    timestamps, states, _, _ = _linear_hidden_plant(commands)
    config = RecursiveBootstrapConfig(control_model="working")
    identifier = RecursiveBootstrapIdentifier(config)

    supported_history: list[bool] = []
    for index, command in enumerate(commands):
        belief = identifier.update(
            states[index],
            states[index + 1],
            command,
            timestamps[index + 1] - timestamps[index],
        )
        supported_history.append(identifier.working_belief_supported)
        # The controller is always handed the belief that just assimilated the
        # newest interval, whether or not support holds yet.
        assert identifier.predictive_belief is belief
        assert identifier.control_belief is belief

    # Readiness is exactly the support conditions, never prequential evidence.
    assert identifier.working_belief_supported
    assert identifier.belief.command_evidence_rank == 4
    assert identifier.belief.angular_effect_rank == 3
    assert identifier.belief.hover_command is not None
    assert any(supported_history) and not all(supported_history)
    assert identifier.control_model_ready
    assert identifier.working_support_reached

    # Nothing the transaction does reaches control in this mode.
    assert identifier.certified_belief is None
    assert identifier.pending_proposal is False
    assert identifier.validation_history == ()
    assert identifier.accepted_update_count == 0
    assert identifier.rejected_update_count == 0


def test_working_control_model_still_records_a_shadow_transaction() -> None:
    commands = _excitation(80)
    timestamps, states, _, _ = _linear_hidden_plant(commands)
    certified = RecursiveBootstrapIdentifier()
    shadowed = RecursiveBootstrapIdentifier(
        RecursiveBootstrapConfig(control_model="working")
    )

    for index, command in enumerate(commands):
        for identifier in (certified, shadowed):
            identifier.update(
                states[index],
                states[index + 1],
                command,
                timestamps[index + 1] - timestamps[index],
            )

    # The two identifiers saw identical evidence, so the shadow record is the
    # certified record: working mode reports what the gate would have done.
    assert shadowed.shadow_certified_belief is not None
    assert (
        shadowed.shadow_certified_belief.interval_count
        == certified.certified_belief.interval_count
    )
    assert shadowed.shadow_accepted_update_count == certified.accepted_update_count
    assert shadowed.shadow_rejected_update_count == certified.rejected_update_count
    assert [report.reason for report in shadowed.shadow_validation_history] == [
        report.reason for report in certified.validation_history
    ]
    # The certified identifier reports no shadow, and neither reports both.
    assert certified.shadow_certified_belief is None
    assert certified.shadow_validation_history == ()
    assert certified.shadow_pending_proposal is False


def test_certified_control_model_is_the_unchanged_default() -> None:
    assert RecursiveBootstrapConfig().control_model == "certified"
    with pytest.raises(ValueError, match="control_model"):
        RecursiveBootstrapConfig(control_model="workin")

    commands = _excitation(80)
    identifier, _, _ = _update_recursive_identifier(commands)

    assert identifier.flies_working_belief is False
    assert identifier.certified_belief is not None
    assert identifier.predictive_belief is identifier.certified_belief
    assert identifier.control_model_ready


_STAGING_BOOKKEEPING = (
    "update_wall_time_s",
    "collective_nuisance_staged",
    "angular_nuisance_staged",
    "collective_staging_interval_count",
    "angular_staging_interval_count",
    "collective_sign_projection_count",
    "collective_sign_projection_magnitude",
)


def test_default_recursive_config_stages_every_regressor_and_enforces_no_sign() -> None:
    """Both pass-three switches are opt-in, so the shipped identifier moves."""

    config = RecursiveBootstrapConfig()

    assert config.staged_regressors is False
    assert config.enforce_collective_sign is False
    assert config.staging_sample_multiple == 4.0
    identifier = RecursiveBootstrapIdentifier()
    assert identifier.belief.collective_nuisance_staged
    assert identifier.belief.angular_nuisance_staged
    assert identifier.belief.collective_staging_interval_count is None


def test_staged_solve_equals_the_full_solve_bit_for_bit_once_fully_staged() -> None:
    """Staging chooses columns, never evidence.

    The Gram and right-hand side are accumulated over every regressor in both
    identifiers, so once the staged solve has admitted the nuisance block it is
    solving the same system on the same data and must return the same floats,
    not merely close ones.
    """

    commands = _excitation(80)

    plain, _, _ = _update_recursive_identifier(commands)
    staged, _, _ = _update_recursive_identifier(
        commands,
        RecursiveBootstrapConfig(staged_regressors=True),
    )

    assert staged.belief.collective_nuisance_staged
    assert staged.belief.angular_nuisance_staged
    # Four samples per column, on eight and eleven columns.
    assert staged.belief.collective_staging_interval_count == 32
    assert staged.belief.angular_staging_interval_count == 44
    expected = plain.belief.to_dict()
    actual = staged.belief.to_dict()
    assert set(expected) == set(actual)
    differing = [
        name
        for name in expected
        if name not in _STAGING_BOOKKEEPING
        and repr(expected[name]) != repr(actual[name])
    ]
    assert differing == []


def test_staged_support_arrives_before_unstaged_support() -> None:
    """Stage one resolves four command directions from five samples.

    Residualizing against the intercept alone is exact centering rather than a
    fitted projection, so the staged solve reports a supported model as soon as
    the design has spanned the command box, instead of waiting for more samples
    than the regression has columns.
    """

    commands = _excitation(80)

    unstaged = _first_supported_interval(commands)
    staged = _first_supported_interval(
        commands,
        RecursiveBootstrapConfig(staged_regressors=True),
    )

    assert unstaged is not None and staged is not None
    assert staged < unstaged
    # The unstaged fit cannot resolve eleven columns from fewer than eleven
    # samples; the staged one only ever has five.
    assert staged <= 5 < unstaged


def test_collective_sign_projection_clips_a_negative_coefficient_and_records_it() -> (
    None
):
    """A motor the fit says pushes down is clipped to no effect, and recorded.

    The hidden plant here really does have a negative fourth thrust
    coefficient, so this is the projection acting against the evidence rather
    than against noise: the constraint is a statement about what a thrust
    fraction means, and it is qualitative, so the clipped coefficient lands on
    exactly zero and carries no magnitude of its own.
    """

    commands = _excitation(80)
    reversed_motor = np.asarray((4.8, 5.0, 5.2, -3.1))

    plain, _, _ = _update_recursive_identifier(
        commands,
        thrust_effect=reversed_motor,
    )
    projected, _, _ = _update_recursive_identifier(
        commands,
        RecursiveBootstrapConfig(enforce_collective_sign=True),
        thrust_effect=reversed_motor,
    )

    assert plain.belief.collective_acceleration_per_command[3] < -1.0
    assert plain.belief.collective_sign_projection_count == 0
    assert projected.belief.collective_acceleration_per_command[3] == 0.0
    assert np.all(projected.belief.collective_acceleration_per_command >= 0.0)
    assert projected.belief.collective_sign_projection_count == 1
    assert projected.belief.collective_sign_projection_magnitude == pytest.approx(
        abs(float(plain.belief.collective_acceleration_per_command[3])),
        rel=1e-9,
    )


def test_collective_sign_projection_leaves_a_positive_fit_untouched() -> None:
    """Where the fit already respects the channel's sign, nothing moves."""

    commands = _excitation(80)

    plain, _, _ = _update_recursive_identifier(commands)
    projected, _, _ = _update_recursive_identifier(
        commands,
        RecursiveBootstrapConfig(enforce_collective_sign=True),
    )

    assert np.all(plain.belief.collective_acceleration_per_command > 0.0)
    assert projected.belief.collective_sign_projection_count == 0
    assert projected.belief.collective_sign_projection_magnitude == 0.0
    assert np.array_equal(
        projected.belief.collective_acceleration_per_command,
        plain.belief.collective_acceleration_per_command,
    )
    assert projected.belief.collective_intercept_m_s2 == (
        plain.belief.collective_intercept_m_s2
    )


def test_staging_sample_multiple_below_one_is_rejected() -> None:
    with pytest.raises(ValueError, match="staging_sample_multiple"):
        RecursiveBootstrapConfig(staging_sample_multiple=0.5)


def test_recursive_belief_exposes_the_accumulated_regression_grams() -> None:
    """The whole evidence, not only the summaries the fits reduce it to.

    A planner whose plan moves the nuisance regressors as well as the commands
    cannot work from the command-block summaries: it needs the Gram in the
    identifier's own feature order.  Exposing it is only honest if it is
    demonstrably the same evidence the reported command information is derived
    from, so the Schur complement is recomputed here through the identifier's
    own nuisance pseudo-inverse rule and compared, and the Loewner ordering the
    complement must satisfy is checked alongside it.
    """

    empty = RecursiveBootstrapIdentifier().belief
    assert empty.collective_information.shape == (8, 8)
    assert empty.angular_information.shape == (11, 11)
    assert np.all(empty.collective_information == 0.0)
    assert np.all(empty.angular_information == 0.0)
    assert not empty.collective_information.flags.writeable
    assert not empty.angular_information.flags.writeable

    identifier, _, _ = _update_recursive_identifier(_excitation(80))
    belief = identifier.belief
    collective = belief.collective_information
    angular = belief.angular_information

    assert np.allclose(collective, collective.T)
    assert np.allclose(angular, angular.T)
    assert np.all(np.linalg.eigvalsh(collective) >= -1e-8)
    assert np.all(np.linalg.eigvalsh(angular) >= -1e-8)
    # Both regressions read the same normalized command off the same samples,
    # so their command blocks are the same accumulated outer products.
    assert np.allclose(collective[:4, :4], angular[:4, :4])
    assert np.trace(angular[:4, :4]) > 0.0

    nuisance_inverse, _ = RecursiveBootstrapIdentifier._nuisance_inverse(
        angular[4:, 4:],
        relative_tolerance=identifier.config.nuisance_rank_relative_tolerance,
    )
    complement = (
        angular[:4, :4] - angular[:4, 4:] @ nuisance_inverse @ angular[:4, 4:].T
    )
    complement = 0.5 * (complement + complement.T)
    assert np.allclose(complement, belief.normalized_command_information, atol=1e-8)
    # Residualizing can only remove information, never add it.
    assert np.all(
        np.linalg.eigvalsh(angular[:4, :4] - belief.normalized_command_information)
        >= -1e-8
    )

    recorded = belief.to_dict()
    assert np.allclose(recorded["collective_information"], collective)
    assert np.allclose(recorded["angular_information"], angular)


def test_transition_aggregation_assimilates_window_means_weighted_by_the_window() -> (
    None
):
    """One sample per window, the window's mean, standing for its transitions.

    At a window of one the identifier is bit-for-bit the identifier as it
    was.  At a window of three, only every third transition changes the
    belief, the interval count still counts transitions, and the accumulated
    Gram is three times the outer product of the window's mean features, so
    the support thresholds and the residual floor stay per transition.
    """

    import numpy as np

    from glassbox.control.online_bootstrap import (
        RecursiveBootstrapConfig,
        RecursiveBootstrapIdentifier,
    )

    with pytest.raises(ValueError):
        RecursiveBootstrapConfig(transition_aggregation_steps=0)

    rng = np.random.default_rng(3)
    dt = 0.01

    def transition(k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        previous = np.zeros(13)
        previous[6] = 1.0
        previous[3:6] = rng.normal(scale=0.1, size=3)
        previous[10:13] = rng.normal(scale=0.2, size=3)
        current = previous.copy()
        current[3:6] += rng.normal(scale=0.02, size=3)
        current[10:13] += rng.normal(scale=0.05, size=3)
        command = np.clip(0.5 + 0.1 * rng.normal(size=4), 0.0, 1.0)
        return previous, current, command

    transitions = [transition(k) for k in range(9)]
    plain = RecursiveBootstrapIdentifier(
        RecursiveBootstrapConfig(control_model="working")
    )
    reference = RecursiveBootstrapIdentifier(
        RecursiveBootstrapConfig(
            control_model="working", transition_aggregation_steps=1
        )
    )
    windowed = RecursiveBootstrapIdentifier(
        RecursiveBootstrapConfig(
            control_model="working", transition_aggregation_steps=3
        )
    )
    for index, (previous, current, command) in enumerate(transitions):
        a = plain.update(previous, current, command, dt).to_dict()
        b = reference.update(previous, current, command, dt).to_dict()
        a.pop("update_wall_time_s")
        b.pop("update_wall_time_s")
        assert a == b
        before = windowed.belief
        after = windowed.update(previous, current, command, dt)
        if (index + 1) % 3:
            assert after is before
            assert windowed.last_sample_report.reason == "sample_buffered"
        else:
            assert after is not before
            assert after.interval_count == index + 1
    assert windowed.belief.interval_count == 9
    assert plain.belief.interval_count == 9

    # The aggregated Gram is the window length times the outer product of the
    # window's mean features: check the first window directly.
    features = [
        windowed._sample_features(previous, current, command, dt)
        for previous, current, command in transitions[:3]
    ]
    mean_force = np.mean([f.force_features for f in features], axis=0)
    single = RecursiveBootstrapIdentifier(
        RecursiveBootstrapConfig(
            control_model="working", transition_aggregation_steps=3
        )
    )
    for previous, current, command in transitions[:3]:
        single.update(previous, current, command, dt)
    assert np.allclose(
        single.belief.collective_information, 3.0 * np.outer(mean_force, mean_force)
    )


def test_the_prequential_residual_floors_the_scale_at_the_belief_error() -> None:
    """The residual scale cannot sit below what the belief actually gets wrong.

    Off, the identifier is bit-for-bit as it was.  On, each residual standard
    deviation is at least the exponentially weighted root-mean-square error
    the belief made predicting each transition before absorbing it, so a fit
    with as many samples as parameters no longer reads as certain.
    """

    import numpy as np

    from glassbox.control.online_bootstrap import (
        RecursiveBootstrapConfig,
        RecursiveBootstrapIdentifier,
    )

    rng = np.random.default_rng(5)
    dt = 0.01

    def transition() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        previous = np.zeros(13)
        previous[6] = 1.0
        previous[3:6] = rng.normal(scale=0.1, size=3)
        previous[10:13] = rng.normal(scale=0.3, size=3)
        current = previous.copy()
        # A large, command-independent angular kick the model cannot fit.
        current[3:6] += rng.normal(scale=0.02, size=3)
        current[10:13] += rng.normal(scale=0.4, size=3)
        command = np.clip(0.5 + 0.05 * rng.normal(size=4), 0.0, 1.0)
        return previous, current, command

    transitions = [transition() for _ in range(12)]
    plain = RecursiveBootstrapIdentifier(
        RecursiveBootstrapConfig(control_model="working")
    )
    off = RecursiveBootstrapIdentifier(
        RecursiveBootstrapConfig(control_model="working", prequential_residual=False)
    )
    on = RecursiveBootstrapIdentifier(
        RecursiveBootstrapConfig(control_model="working", prequential_residual=True)
    )
    for index, (previous, current, command) in enumerate(transitions):
        a = plain.update(previous, current, command, dt).to_dict()
        b = off.update(previous, current, command, dt).to_dict()
        a.pop("update_wall_time_s")
        b.pop("update_wall_time_s")
        assert a == b
        before = on.belief
        on.update(previous, current, command, dt)
        if index == 0:
            # The very first transition meets an empty belief: its error is the
            # target itself and is not recorded.
            assert before.command_evidence_rank == 0
            assert on._prequential_weight == 0.0
    assert (
        on.belief.collective_residual_std_m_s2
        >= off.belief.collective_residual_std_m_s2
    )
    assert np.all(
        on.belief.angular_residual_std_rad_s2 >= off.belief.angular_residual_std_rad_s2
    )
    # Twelve samples on eleven angular columns fit the kicks in-sample; the
    # prequential error does not, so the scale on the switch is well above
    # the declared floor.
    assert np.all(
        on.belief.angular_residual_std_rad_s2
        > 2.0 * RecursiveBootstrapConfig().angular_residual_std_floor_rad_s2
    )


def test_the_integrated_collective_fit_is_honest_under_velocity_noise() -> None:
    """The cumulative collective regression is the least-squares form for measurement noise.

    Off, the identifier is bit-for-bit as it was.  On, with white noise on the
    measured velocity, the collective map's coefficient error after a burst of
    transitions is smaller than the per-interval fit's and the collective
    authority is higher, because the integrated target carries the velocity
    noise once per row instead of divided by the interval.  The exported
    collective Gram is the equivalent per-transition one, scaled so that
    dividing it by the reported residual, the declared floor, gives the
    integrated precision.
    """

    import numpy as np

    from glassbox.control.online_bootstrap import (
        RecursiveBootstrapConfig,
        RecursiveBootstrapIdentifier,
    )
    from glassbox.core.dynamics import GRAVITY_M_S2

    dt = 0.01
    true_map = np.asarray((4.6, 4.7, 4.5, 4.6))
    intercept = -0.4
    noise = 0.02

    def flight(seed: int) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        local = np.random.default_rng(seed)
        velocity = np.zeros(3)
        transitions = []
        for k in range(60):
            command = np.clip(
                0.5 + 0.1 * (1 if (k // 5) % 2 else -1) + 0.05 * local.normal(size=4),
                0.0,
                1.0,
            )
            force = true_map @ command + intercept
            previous = np.zeros(13)
            previous[6] = 1.0
            previous[3:6] = velocity + noise * local.normal(size=3)
            velocity = velocity + dt * np.asarray((0.0, 0.0, force - GRAVITY_M_S2))
            current = np.zeros(13)
            current[6] = 1.0
            current[3:6] = velocity + noise * local.normal(size=3)
            transitions.append((previous, current, command))
        return transitions

    transitions = flight(3)
    plain = RecursiveBootstrapIdentifier(
        RecursiveBootstrapConfig(control_model="working")
    )
    off = RecursiveBootstrapIdentifier(
        RecursiveBootstrapConfig(control_model="working", integrated_collective=False)
    )
    on = RecursiveBootstrapIdentifier(
        RecursiveBootstrapConfig(control_model="working", integrated_collective=True)
    )
    first_half_authority: dict[str, int | None] = {"off": None, "on": None}
    for index, (previous, current, command) in enumerate(transitions):
        a = plain.update(previous, current, command, dt).to_dict()
        b = off.update(previous, current, command, dt).to_dict()
        a.pop("update_wall_time_s")
        b.pop("update_wall_time_s")
        assert a == b
        on.update(previous, current, command, dt)
        for name, identifier in (("off", off), ("on", on)):
            if (
                first_half_authority[name] is None
                and identifier.belief.collective_authority >= 0.5
            ):
                first_half_authority[name] = index

    per_step_error = abs(
        np.sum(off.belief.collective_acceleration_per_command) - true_map.sum()
    )
    integrated_error = abs(
        np.sum(on.belief.collective_acceleration_per_command) - true_map.sum()
    )
    assert integrated_error < per_step_error
    # The collective authority reaches half no later under the integrated fit.
    assert first_half_authority["on"] is not None
    assert first_half_authority["off"] is None or (
        first_half_authority["on"] <= first_half_authority["off"]
    )
    assert (
        on.belief.collective_residual_std_m_s2
        == RecursiveBootstrapConfig().collective_residual_std_floor_m_s2
    )
    assert not np.allclose(
        on.belief.collective_information, off.belief.collective_information
    )
    assert np.allclose(on.belief.angular_information, off.belief.angular_information)


# The dual-control controller in glassbox-throw reads this belief field by
# field, so the field list is a downstream contract: a rename or removal has to
# be made here on purpose, and mirrored there.
RECURSIVE_BOOTSTRAP_BELIEF_FIELDS = (
    "interval_count",
    "effective_interval_count",
    "collective_acceleration_per_command",
    "collective_velocity_coefficient",
    "collective_intercept_m_s2",
    "angular_acceleration_per_command",
    "angular_rate_coefficient",
    "angular_rate_product_coefficient",
    "angular_intercept_rad_s2",
    "normalized_command_support_projector",
    "normalized_command_singular_values",
    "normalized_command_information",
    "supported_collective_effect_covariance",
    "supported_angular_effect_covariance",
    "collective_information",
    "angular_information",
    "command_evidence_rank",
    "angular_effect_rank",
    "collective_nuisance_rank",
    "angular_nuisance_rank",
    "angular_output_support_projector",
    "collective_support_fraction",
    "minimum_supported_information_singular_value",
    "information_authority",
    "collective_effect_signal_to_noise",
    "angular_effect_signal_to_noise",
    "collective_residual_std_m_s2",
    "angular_residual_std_rad_s2",
    "exploration_completion",
    "collective_authority",
    "angular_axis_authority",
    "hover_command",
    "update_wall_time_s",
    "collective_nuisance_staged",
    "angular_nuisance_staged",
    "collective_staging_interval_count",
    "angular_staging_interval_count",
    "collective_sign_projection_count",
    "collective_sign_projection_magnitude",
)


def test_recursive_bootstrap_belief_fields_are_a_downstream_contract() -> None:
    names = tuple(field.name for field in fields(RecursiveBootstrapBelief))
    assert names == RECURSIVE_BOOTSTRAP_BELIEF_FIELDS
