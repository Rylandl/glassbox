from __future__ import annotations

from dataclasses import replace

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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    thrust_effect = np.asarray((4.8, 5.0, 5.2, 4.9))
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
    **plant: float,
) -> tuple[RecursiveBootstrapIdentifier, np.ndarray, np.ndarray]:
    timestamps, states, thrust_effect, angular_effect = _linear_hidden_plant(
        commands,
        **plant,
    )
    identifier = RecursiveBootstrapIdentifier()
    for index, command in enumerate(commands):
        identifier.update(
            states[index],
            states[index + 1],
            command,
            timestamps[index + 1] - timestamps[index],
        )
    return identifier, thrust_effect, angular_effect


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
