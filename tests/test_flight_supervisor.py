import math

import numpy as np
import pytest

from glassbox.dynamics import MOTOR_MIXER
from glassbox.flight_supervisor import (
    MultirotorFlightSupervisor,
    MultirotorSupervisorConfig,
    SupervisorMode,
    SupervisorReason,
)
from glassbox.synthetic import resting_state


def _roll_quaternion(angle_rad: float) -> np.ndarray:
    return np.asarray(
        (
            math.cos(0.5 * angle_rad),
            math.sin(0.5 * angle_rad),
            0.0,
            0.0,
        )
    )


def _decision(
    supervisor: MultirotorFlightSupervisor,
    *,
    state: np.ndarray | None = None,
    candidate: np.ndarray | None = None,
    now_s: float = 1.0,
    state_received_at_s: float | None = None,
    command_generated_at_s: float | None = None,
    usable: bool = True,
):
    return supervisor.supervise(
        state=resting_state() if state is None else state,
        state_received_at_s=(
            now_s if state_received_at_s is None else state_received_at_s
        ),
        candidate_command=(np.full(4, 0.4) if candidate is None else candidate),
        command_generated_at_s=(
            now_s if command_generated_at_s is None else command_generated_at_s
        ),
        now_s=now_s,
        controller_command_usable=usable,
        previous_applied_command=np.full(4, 0.35),
    )


def test_nominal_fresh_command_passes_unchanged() -> None:
    supervisor = MultirotorFlightSupervisor(
        MultirotorSupervisorConfig(collective_hold_command=0.35)
    )
    candidate = np.asarray((0.31, 0.42, 0.39, 0.36))

    decision = _decision(
        supervisor,
        candidate=candidate,
        state_received_at_s=0.99,
        command_generated_at_s=0.995,
    )

    assert decision.mode == SupervisorMode.NOMINAL
    assert decision.nominal_command_accepted
    assert not decision.intervened
    assert decision.reasons == ()
    np.testing.assert_array_equal(decision.command, candidate)
    with pytest.raises(ValueError):
        decision.command[0] = 0.0


def test_stale_command_uses_bounded_attitude_and_rate_arrest() -> None:
    config = MultirotorSupervisorConfig(collective_hold_command=0.35)
    supervisor = MultirotorFlightSupervisor(config)
    state = resting_state()
    state[6:10] = _roll_quaternion(0.30)
    state[10:13] = (0.8, -0.2, 0.3)

    decision = _decision(
        supervisor,
        state=state,
        command_generated_at_s=0.90,
    )

    assert decision.mode == SupervisorMode.RATE_ARREST
    assert SupervisorReason.COMMAND_STALE in decision.reasons
    assert np.all(decision.command >= np.asarray(config.command_minimum))
    assert np.all(decision.command <= np.asarray(config.command_maximum))
    differentials = np.asarray(MOTOR_MIXER) @ (
        decision.command - np.asarray(config.collective_hold_command)
    )
    assert differentials[0] < 0.0
    assert differentials[2] < 0.0


def test_invalid_or_stale_state_uses_collective_hold() -> None:
    config = MultirotorSupervisorConfig(collective_hold_command=0.35)
    supervisor = MultirotorFlightSupervisor(config)
    invalid = resting_state()
    invalid[3] = np.nan

    nonfinite = _decision(supervisor, state=invalid)
    supervisor.reset()
    stale = _decision(supervisor, state_received_at_s=0.90)

    for decision, reason in (
        (nonfinite, SupervisorReason.STATE_INVALID),
        (stale, SupervisorReason.STATE_STALE),
    ):
        assert decision.mode == SupervisorMode.COLLECTIVE_HOLD
        assert reason in decision.reasons
        np.testing.assert_array_equal(
            decision.command,
            config.collective_hold_command,
        )


@pytest.mark.parametrize(
    ("state_change", "state_received_at_s", "reason"),
    (
        ("zero_quaternion", 1.0, SupervisorReason.QUATERNION_INVALID),
        ("none", math.nan, SupervisorReason.STATE_TIMESTAMP_INVALID),
        ("none", 1.01, SupervisorReason.STATE_FROM_FUTURE),
    ),
)
def test_remaining_telemetry_faults_select_collective_hold(
    state_change: str,
    state_received_at_s: float,
    reason: SupervisorReason,
) -> None:
    config = MultirotorSupervisorConfig(collective_hold_command=0.35)
    state = resting_state()
    if state_change == "zero_quaternion":
        state[6:10] = 0.0

    decision = _decision(
        MultirotorFlightSupervisor(config),
        state=state,
        state_received_at_s=state_received_at_s,
    )

    assert decision.mode == SupervisorMode.COLLECTIVE_HOLD
    assert reason in decision.reasons
    np.testing.assert_array_equal(decision.command, config.collective_hold_command)


@pytest.mark.parametrize(
    ("command_generated_at_s", "reason"),
    (
        (math.nan, SupervisorReason.COMMAND_TIMESTAMP_INVALID),
        (1.01, SupervisorReason.COMMAND_FROM_FUTURE),
    ),
)
def test_remaining_command_timestamp_faults_select_bounded_rate_arrest(
    command_generated_at_s: float,
    reason: SupervisorReason,
) -> None:
    config = MultirotorSupervisorConfig(collective_hold_command=0.35)
    decision = _decision(
        MultirotorFlightSupervisor(config),
        command_generated_at_s=command_generated_at_s,
    )

    assert decision.mode == SupervisorMode.RATE_ARREST
    assert reason in decision.reasons
    assert np.all(decision.command >= np.asarray(config.command_minimum))
    assert np.all(decision.command <= np.asarray(config.command_maximum))


def test_limits_and_unusable_or_invalid_commands_trigger_arrest() -> None:
    config = MultirotorSupervisorConfig(
        collective_hold_command=0.35,
        maximum_tilt_rad=0.5,
        release_tilt_rad=0.3,
        maximum_angular_rate_rad_s=2.0,
        release_angular_rate_rad_s=1.0,
    )
    state = resting_state()
    state[6:10] = _roll_quaternion(0.6)
    state[10] = 2.5

    limit = _decision(MultirotorFlightSupervisor(config), state=state)
    unusable = _decision(MultirotorFlightSupervisor(config), usable=False)
    invalid = _decision(
        MultirotorFlightSupervisor(config),
        candidate=np.full(4, np.nan),
    )
    unbounded = _decision(
        MultirotorFlightSupervisor(config),
        candidate=np.asarray((0.2, 0.3, 1.1, 0.4)),
    )

    assert SupervisorReason.TILT_LIMIT in limit.reasons
    assert SupervisorReason.ANGULAR_RATE_LIMIT in limit.reasons
    assert SupervisorReason.CONTROLLER_UNUSABLE in unusable.reasons
    assert SupervisorReason.COMMAND_INVALID in invalid.reasons
    assert SupervisorReason.COMMAND_OUT_OF_BOUNDS in unbounded.reasons
    assert all(
        decision.mode == SupervisorMode.RATE_ARREST
        for decision in (limit, unusable, invalid, unbounded)
    )


def test_arrest_latches_until_minimum_duration_and_release_thresholds() -> None:
    supervisor = MultirotorFlightSupervisor(
        MultirotorSupervisorConfig(
            collective_hold_command=0.35,
            minimum_arrest_duration_s=0.10,
        )
    )

    triggered = _decision(supervisor, now_s=1.0, usable=False)
    latched = _decision(supervisor, now_s=1.05)
    released = _decision(supervisor, now_s=1.11)

    assert triggered.mode == SupervisorMode.RATE_ARREST
    assert latched.mode == SupervisorMode.RATE_ARREST
    assert latched.reasons == (SupervisorReason.ARREST_LATCHED,)
    assert released.mode == SupervisorMode.NOMINAL


def test_time_regression_never_reuses_nominal_authority() -> None:
    supervisor = MultirotorFlightSupervisor(MultirotorSupervisorConfig())
    assert _decision(supervisor, now_s=1.0).mode == SupervisorMode.NOMINAL

    regressed = _decision(supervisor, now_s=0.5)

    assert regressed.mode == SupervisorMode.COLLECTIVE_HOLD
    assert SupervisorReason.TIME_REGRESSION in regressed.reasons


def test_supervisor_config_rejects_incoherent_release_and_command_bounds() -> None:
    with pytest.raises(ValueError, match="below maximum_tilt"):
        MultirotorSupervisorConfig(maximum_tilt_rad=0.5, release_tilt_rad=0.5)
    with pytest.raises(ValueError, match="inside command bounds"):
        MultirotorSupervisorConfig(collective_hold_command=1.1)
    with pytest.raises(ValueError, match="below maximum rates"):
        MultirotorSupervisorConfig(
            maximum_angular_rate_rad_s=2.0,
            release_angular_rate_rad_s=2.0,
        )


def _arrest_command(
    supervisor: MultirotorFlightSupervisor,
    roll_rad: float,
) -> np.ndarray:
    state = resting_state()
    state[6:10] = _roll_quaternion(roll_rad)
    decision = supervisor.supervise(
        state=state,
        state_received_at_s=1.0,
        candidate_command=np.full(4, 0.4),
        command_generated_at_s=1.0,
        now_s=1.0,
        controller_command_usable=False,
        previous_applied_command=np.full(4, 0.5),
    )
    assert decision.mode == SupervisorMode.RATE_ARREST
    return decision.command


def test_arrest_keeps_restoring_authority_near_inversion() -> None:
    config = MultirotorSupervisorConfig()
    upright = _arrest_command(MultirotorFlightSupervisor(config), 1.0)
    inverted = _arrest_command(MultirotorFlightSupervisor(config), 3.0)
    exactly_inverted = _arrest_command(MultirotorFlightSupervisor(config), math.pi)

    upright_spread = float(np.max(upright) - np.min(upright))
    inverted_spread = float(np.max(inverted) - np.min(inverted))
    exact_spread = float(np.max(exactly_inverted) - np.min(exactly_inverted))

    assert upright_spread > 0.1
    assert inverted_spread >= 0.9 * upright_spread
    assert exact_spread >= 0.9 * upright_spread
    assert np.sign(inverted[0] - inverted[1]) == np.sign(upright[0] - upright[1])


def test_small_tilt_arrest_matches_the_linearized_cross_product() -> None:
    config = MultirotorSupervisorConfig()
    hold = np.asarray(config.collective_hold_command)
    mixer = np.asarray(MOTOR_MIXER)

    for roll_rad in (1e-3, 5e-3, 1e-2):
        state = resting_state()
        state[6:10] = _roll_quaternion(roll_rad)
        _, x, _, _ = state[6:10]
        world_up_body = np.asarray((0.0, 2.0 * x * state[6], 1.0 - 2.0 * x * x))
        linearized = np.cross(np.asarray((0.0, 0.0, 1.0)), world_up_body)
        expected = hold + 0.25 * mixer.T @ np.asarray(
            (config.tilt_gain * linearized[0], config.tilt_gain * linearized[1], 0.0)
        )

        command = _arrest_command(MultirotorFlightSupervisor(config), roll_rad)

        np.testing.assert_allclose(command, expected, atol=1e-6)


def test_rounding_width_bound_overshoot_is_clipped_not_arrested() -> None:
    supervisor = MultirotorFlightSupervisor(MultirotorSupervisorConfig())

    decision = _decision(supervisor, candidate=np.asarray((1.0 + 1e-9, 0.4, 0.4, 0.4)))

    assert decision.mode == SupervisorMode.NOMINAL
    assert decision.reasons == ()
    assert decision.command[0] == pytest.approx(1.0)

    supervisor.reset()
    rejected = _decision(supervisor, candidate=np.asarray((1.01, 0.4, 0.4, 0.4)))

    assert rejected.mode == SupervisorMode.RATE_ARREST
    assert SupervisorReason.COMMAND_OUT_OF_BOUNDS in rejected.reasons


def test_time_regression_latch_still_serves_the_minimum_arrest_duration() -> None:
    supervisor = MultirotorFlightSupervisor(
        MultirotorSupervisorConfig(minimum_arrest_duration_s=0.10)
    )
    assert _decision(supervisor, now_s=1.0).mode == SupervisorMode.NOMINAL

    regressed = _decision(supervisor, now_s=0.5)
    early = _decision(supervisor, now_s=1.05)
    released = _decision(supervisor, now_s=1.11)

    assert regressed.mode == SupervisorMode.COLLECTIVE_HOLD
    assert early.mode == SupervisorMode.RATE_ARREST
    assert early.reasons == (SupervisorReason.ARREST_LATCHED,)
    assert released.mode == SupervisorMode.NOMINAL
