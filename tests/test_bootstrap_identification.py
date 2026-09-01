from __future__ import annotations

import numpy as np
import pytest

from glassbox.bootstrap_identification import (
    BootstrapExcitationConfig,
    BootstrapIdentificationConfig,
    BootstrapModelNotReadyError,
    BootstrapMultirotorIdentifier,
    plan_bootstrap_excitation,
)
from glassbox.dynamics import GRAVITY_M_S2, MOTOR_MIXER
from glassbox.online_bootstrap import (
    ProgressiveBootstrapController,
    RecursiveBootstrapIdentifier,
)


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    thrust_effect = np.asarray((4.8, 5.0, 5.2, 4.9))
    angular_effect = np.diag((34.0, 31.0, 11.0)) @ np.asarray(MOTOR_MIXER)
    angular_rate_coefficient = -np.diag((0.4, 0.5, 0.25))
    thrust_intercept = 0.08
    states = np.zeros((len(commands) + 1, 13), dtype=np.float64)
    states[:, 6] = 1.0
    timestamps = np.arange(len(states), dtype=np.float64) * sample_period_s
    for index, command in enumerate(commands):
        velocity = states[index, 3:6]
        angular_velocity = states[index, 10:13]
        body_specific_force_z = (
            thrust_effect @ command + thrust_intercept - 0.12 * velocity[2]
        )
        acceleration = np.asarray((0.0, 0.0, body_specific_force_z - GRAVITY_M_S2))
        angular_acceleration = (
            angular_effect @ command + angular_rate_coefficient @ angular_velocity
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
) -> tuple[RecursiveBootstrapIdentifier, np.ndarray, np.ndarray]:
    timestamps, states, thrust_effect, angular_effect = _linear_hidden_plant(commands)
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
    assert identifier.control_belief is certified
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


def test_recursive_rank_deficiency_never_certifies_unobserved_axes() -> None:
    commands = _excitation(80, collective_only=True)

    identifier, _, _ = _update_recursive_identifier(commands)
    belief = identifier.belief

    assert belief.command_evidence_rank <= 1
    assert belief.angular_effect_rank < 3
    assert identifier.certified_belief is None
    assert identifier.control_belief is belief
    assert np.max(belief.angular_axis_authority) < 0.26


def test_progressive_controller_explores_before_support_and_stays_bounded() -> None:
    identifier = RecursiveBootstrapIdentifier()
    controller = ProgressiveBootstrapController(identifier.config)
    state = np.zeros(13, dtype=np.float64)
    state[6] = 1.0

    decision = controller.command(
        state,
        identifier.control_belief,
        previous_command=np.zeros(4),
    )

    assert not np.allclose(decision.command, decision.feedback_command)
    assert decision.collective_authority == 0.0
    np.testing.assert_allclose(decision.angular_axis_authority, 0.0)
    assert np.all(decision.command >= 0.0)
    assert np.all(decision.command <= 1.0)


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
