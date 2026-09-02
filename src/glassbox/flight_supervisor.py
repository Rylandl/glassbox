"""Independent command-freshness and attitude/rate-arrest supervision."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np

from glassbox.dynamics import MOTOR_MIXER


def _finite_vector(
    name: str,
    values: float | Sequence[float],
    size: int,
) -> tuple[float, ...]:
    if np.isscalar(values):
        result = np.full(size, float(values), dtype=np.float64)
    else:
        result = np.asarray(tuple(values), dtype=np.float64)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain {size} finite values")
    return tuple(float(value) for value in result)


def _tilt_error_body(world_up_body: np.ndarray) -> np.ndarray:
    """Return the geodesic body-frame rotation that would level the vehicle.

    The naive cross product of the body vertical with the measured world up has
    magnitude ``sin(tilt)``, so it collapses toward zero exactly when the vehicle
    is closest to inverted and the restoring torque matters most.  This returns
    the rotation vector instead: the same axis, but scaled to the true tilt angle,
    so the command keeps full authority across the whole ``[0, pi]`` range.  For
    small tilts it agrees with the cross product to third order in the angle, and
    it keeps the cross product's sign convention at every tilt below inversion.
    At exact inversion the axis is undefined and a fixed positive roll is chosen.
    """

    axis = np.cross(np.asarray((0.0, 0.0, 1.0)), world_up_body)
    sine = float(np.linalg.norm(axis))
    cosine = float(world_up_body[2])
    angle = math.atan2(sine, cosine)
    if sine < 1e-9:
        if cosine >= 0.0:
            return axis
        return np.asarray((angle, 0.0, 0.0))
    return (angle / sine) * axis


class SupervisorMode(StrEnum):
    """Authority selected by the independent flight supervisor."""

    NOMINAL = "nominal"
    RATE_ARREST = "rate_arrest"
    COLLECTIVE_HOLD = "collective_hold"


class SupervisorReason(StrEnum):
    """Auditable reasons for rejecting or retaining a nominal command."""

    TIME_REGRESSION = "time_regression"
    STATE_TIMESTAMP_INVALID = "state_timestamp_invalid"
    STATE_STALE = "state_stale"
    STATE_FROM_FUTURE = "state_from_future"
    STATE_INVALID = "state_invalid"
    QUATERNION_INVALID = "quaternion_invalid"
    COMMAND_TIMESTAMP_INVALID = "command_timestamp_invalid"
    COMMAND_STALE = "command_stale"
    COMMAND_FROM_FUTURE = "command_from_future"
    COMMAND_INVALID = "command_invalid"
    COMMAND_OUT_OF_BOUNDS = "command_out_of_bounds"
    CONTROLLER_UNUSABLE = "controller_unusable"
    TILT_LIMIT = "tilt_limit"
    ANGULAR_RATE_LIMIT = "angular_rate_limit"
    ARREST_LATCHED = "arrest_latched"


@dataclass(frozen=True)
class MultirotorSupervisorConfig:
    """Fixed, model-independent multirotor supervisor contract."""

    command_minimum: float | tuple[float, float, float, float] = 0.0
    command_maximum: float | tuple[float, float, float, float] = 1.0
    collective_hold_command: float | tuple[float, float, float, float] = 0.5
    maximum_state_age_s: float = 0.04
    maximum_command_age_s: float = 0.02
    future_timestamp_tolerance_s: float = 1e-6
    maximum_tilt_rad: float = 0.70
    release_tilt_rad: float = 0.45
    maximum_angular_rate_rad_s: float | tuple[float, float, float] = 6.0
    release_angular_rate_rad_s: float | tuple[float, float, float] = 3.0
    minimum_arrest_duration_s: float = 0.10
    tilt_gain: float = 0.8
    roll_pitch_rate_gain: float = 0.22
    yaw_rate_gain: float = 0.12
    maximum_axis_differential: float | tuple[float, float, float] = 0.8
    maximum_arrest_motor_step: float = 0.20

    def __post_init__(self) -> None:
        minimum = _finite_vector("command_minimum", self.command_minimum, 4)
        maximum = _finite_vector("command_maximum", self.command_maximum, 4)
        hold = _finite_vector(
            "collective_hold_command",
            self.collective_hold_command,
            4,
        )
        maximum_rate = _finite_vector(
            "maximum_angular_rate_rad_s",
            self.maximum_angular_rate_rad_s,
            3,
        )
        release_rate = _finite_vector(
            "release_angular_rate_rad_s",
            self.release_angular_rate_rad_s,
            3,
        )
        maximum_differential = _finite_vector(
            "maximum_axis_differential",
            self.maximum_axis_differential,
            3,
        )
        if np.any(np.asarray(minimum) >= np.asarray(maximum)):
            raise ValueError("command_minimum must be below command_maximum")
        if np.any(np.asarray(hold) < minimum) or np.any(np.asarray(hold) > maximum):
            raise ValueError("collective_hold_command must lie inside command bounds")
        if np.any(np.asarray(maximum_rate) <= 0.0):
            raise ValueError("maximum angular rates must be positive")
        if np.any(np.asarray(release_rate) <= 0.0) or np.any(
            np.asarray(release_rate) >= np.asarray(maximum_rate)
        ):
            raise ValueError(
                "release angular rates must be positive and below maximum rates"
            )
        if np.any(np.asarray(maximum_differential) <= 0.0):
            raise ValueError("maximum axis differentials must be positive")
        positive_fields = (
            "maximum_state_age_s",
            "maximum_command_age_s",
            "maximum_tilt_rad",
            "release_tilt_rad",
            "minimum_arrest_duration_s",
            "tilt_gain",
            "roll_pitch_rate_gain",
            "yaw_rate_gain",
            "maximum_arrest_motor_step",
        )
        for name in positive_fields:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.future_timestamp_tolerance_s) or (
            self.future_timestamp_tolerance_s < 0.0
        ):
            raise ValueError(
                "future_timestamp_tolerance_s must be finite and nonnegative"
            )
        if self.release_tilt_rad >= self.maximum_tilt_rad:
            raise ValueError("release_tilt_rad must be below maximum_tilt_rad")
        if self.maximum_tilt_rad >= math.pi:
            raise ValueError("maximum_tilt_rad must be below pi")
        object.__setattr__(self, "command_minimum", minimum)
        object.__setattr__(self, "command_maximum", maximum)
        object.__setattr__(self, "collective_hold_command", hold)
        object.__setattr__(self, "maximum_angular_rate_rad_s", maximum_rate)
        object.__setattr__(self, "release_angular_rate_rad_s", release_rate)
        object.__setattr__(self, "maximum_axis_differential", maximum_differential)

    def to_dict(self) -> dict[str, Any]:
        return {
            "command_minimum": list(self.command_minimum),
            "command_maximum": list(self.command_maximum),
            "collective_hold_command": list(self.collective_hold_command),
            "maximum_state_age_s": self.maximum_state_age_s,
            "maximum_command_age_s": self.maximum_command_age_s,
            "future_timestamp_tolerance_s": self.future_timestamp_tolerance_s,
            "maximum_tilt_rad": self.maximum_tilt_rad,
            "release_tilt_rad": self.release_tilt_rad,
            "maximum_angular_rate_rad_s": list(self.maximum_angular_rate_rad_s),
            "release_angular_rate_rad_s": list(self.release_angular_rate_rad_s),
            "minimum_arrest_duration_s": self.minimum_arrest_duration_s,
            "tilt_gain": self.tilt_gain,
            "roll_pitch_rate_gain": self.roll_pitch_rate_gain,
            "yaw_rate_gain": self.yaw_rate_gain,
            "maximum_axis_differential": list(self.maximum_axis_differential),
            "maximum_arrest_motor_step": self.maximum_arrest_motor_step,
        }


@dataclass(frozen=True)
class SupervisedCommand:
    """One immutable command-selection decision."""

    command: np.ndarray
    mode: SupervisorMode
    reasons: tuple[SupervisorReason, ...]
    state_age_s: float | None
    command_age_s: float | None
    tilt_rad: float | None
    maximum_angular_rate_rad_s: float | None
    nominal_command_accepted: bool

    def __post_init__(self) -> None:
        command = np.asarray(self.command, dtype=np.float64).copy()
        if command.shape != (4,) or not np.all(np.isfinite(command)):
            raise ValueError("supervised command must contain four finite values")
        command.flags.writeable = False
        object.__setattr__(self, "command", command)

    @property
    def intervened(self) -> bool:
        return self.mode != SupervisorMode.NOMINAL

    def to_dict(self) -> dict[str, Any]:
        return {
            "command": self.command.tolist(),
            "mode": self.mode.value,
            "reasons": [reason.value for reason in self.reasons],
            "state_age_s": self.state_age_s,
            "command_age_s": self.command_age_s,
            "tilt_rad": self.tilt_rad,
            "maximum_angular_rate_rad_s": self.maximum_angular_rate_rad_s,
            "nominal_command_accepted": self.nominal_command_accepted,
            "intervened": self.intervened,
        }


def _state_metrics(state: Any) -> tuple[np.ndarray | None, float | None, float | None]:
    try:
        values = np.asarray(state, dtype=np.float64)
    except (TypeError, ValueError):
        return None, None, None
    if values.shape != (13,) or not np.all(np.isfinite(values)):
        return None, None, None
    quaternion = values[6:10]
    norm = float(np.linalg.norm(quaternion))
    if not math.isfinite(norm) or norm < 1e-6:
        return values, None, None
    w, x, y, z = quaternion / norm
    world_up_body = np.asarray(
        (
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        )
    )
    tilt = math.acos(float(np.clip(world_up_body[2], -1.0, 1.0)))
    maximum_rate = float(np.max(np.abs(values[10:13])))
    return values, tilt, maximum_rate


class MultirotorFlightSupervisor:
    """Stateful, model-independent boundary around one multirotor controller."""

    #: Candidate commands are accepted this far outside the configured bounds
    #: and then clipped, so a rounding-width overshoot cannot latch an arrest.
    _BOUND_TOLERANCE_FRACTION = 1e-6

    def __init__(self, config: MultirotorSupervisorConfig) -> None:
        self.config = config
        self._arrest_started_at_s: float | None = None
        self._last_time_s: float | None = None

    def reset(self) -> None:
        self._arrest_started_at_s = None
        self._last_time_s = None

    def _rate_arrest_command(
        self,
        state: np.ndarray,
        previous_applied_command: Any,
    ) -> np.ndarray:
        quaternion = state[6:10] / np.linalg.norm(state[6:10])
        w, x, y, z = quaternion
        world_up_body = np.asarray(
            (
                2.0 * (x * z - w * y),
                2.0 * (y * z + w * x),
                1.0 - 2.0 * (x * x + y * y),
            )
        )
        tilt_error_body = _tilt_error_body(world_up_body)
        rates = state[10:13]
        differential = np.asarray(
            (
                self.config.tilt_gain * tilt_error_body[0]
                - self.config.roll_pitch_rate_gain * rates[0],
                self.config.tilt_gain * tilt_error_body[1]
                - self.config.roll_pitch_rate_gain * rates[1],
                -self.config.yaw_rate_gain * rates[2],
            )
        )
        maximum_differential = np.asarray(self.config.maximum_axis_differential)
        differential = np.clip(
            differential,
            -maximum_differential,
            maximum_differential,
        )
        command = np.asarray(self.config.collective_hold_command) + (
            0.25 * np.asarray(MOTOR_MIXER).T @ differential
        )
        try:
            previous = np.asarray(previous_applied_command, dtype=np.float64)
        except (TypeError, ValueError):
            previous = np.empty(0)
        if previous.shape == (4,) and np.all(np.isfinite(previous)):
            command = np.clip(
                command,
                previous - self.config.maximum_arrest_motor_step,
                previous + self.config.maximum_arrest_motor_step,
            )
        return np.clip(
            command,
            self.config.command_minimum,
            self.config.command_maximum,
        )

    def supervise(
        self,
        *,
        state: Any,
        state_received_at_s: float,
        candidate_command: Any,
        command_generated_at_s: float,
        now_s: float,
        controller_command_usable: bool,
        previous_applied_command: Any,
    ) -> SupervisedCommand:
        """Select a nominal, rate-arrest, or collective-hold command."""

        reasons: list[SupervisorReason] = []
        time_valid = math.isfinite(now_s)
        if not time_valid or (
            self._last_time_s is not None and now_s < self._last_time_s
        ):
            reasons.append(SupervisorReason.TIME_REGRESSION)
        if time_valid:
            self._last_time_s = (
                now_s if self._last_time_s is None else max(self._last_time_s, now_s)
            )

        state_age: float | None = None
        if not math.isfinite(state_received_at_s) or not time_valid:
            reasons.append(SupervisorReason.STATE_TIMESTAMP_INVALID)
        else:
            state_age = now_s - state_received_at_s
            if state_age < -self.config.future_timestamp_tolerance_s:
                reasons.append(SupervisorReason.STATE_FROM_FUTURE)
            elif state_age > self.config.maximum_state_age_s:
                reasons.append(SupervisorReason.STATE_STALE)

        state_values, tilt, maximum_rate = _state_metrics(state)
        if state_values is None:
            reasons.append(SupervisorReason.STATE_INVALID)
        elif tilt is None or maximum_rate is None:
            reasons.append(SupervisorReason.QUATERNION_INVALID)

        telemetry_reasons = {
            SupervisorReason.TIME_REGRESSION,
            SupervisorReason.STATE_TIMESTAMP_INVALID,
            SupervisorReason.STATE_STALE,
            SupervisorReason.STATE_FROM_FUTURE,
            SupervisorReason.STATE_INVALID,
            SupervisorReason.QUATERNION_INVALID,
        }
        telemetry_valid = not any(reason in telemetry_reasons for reason in reasons)
        if not telemetry_valid:
            # The latch starts from the newest time seen, never from a regressed
            # clock reading, so the next valid tick cannot claim that the minimum
            # arrest duration has already elapsed.
            if time_valid:
                self._arrest_started_at_s = self._last_time_s
            return SupervisedCommand(
                command=np.asarray(self.config.collective_hold_command),
                mode=SupervisorMode.COLLECTIVE_HOLD,
                reasons=tuple(dict.fromkeys(reasons)),
                state_age_s=state_age,
                command_age_s=None,
                tilt_rad=tilt,
                maximum_angular_rate_rad_s=maximum_rate,
                nominal_command_accepted=False,
            )

        command_age: float | None = None
        if not math.isfinite(command_generated_at_s) or not time_valid:
            reasons.append(SupervisorReason.COMMAND_TIMESTAMP_INVALID)
        else:
            command_age = now_s - command_generated_at_s
            if command_age < -self.config.future_timestamp_tolerance_s:
                reasons.append(SupervisorReason.COMMAND_FROM_FUTURE)
            elif command_age > self.config.maximum_command_age_s:
                reasons.append(SupervisorReason.COMMAND_STALE)

        try:
            candidate = np.asarray(candidate_command, dtype=np.float64)
        except (TypeError, ValueError):
            candidate = np.empty(0)
        minimum = np.asarray(self.config.command_minimum)
        maximum = np.asarray(self.config.command_maximum)
        tolerance = self._BOUND_TOLERANCE_FRACTION * (maximum - minimum)
        if candidate.shape != (4,) or not np.all(np.isfinite(candidate)):
            reasons.append(SupervisorReason.COMMAND_INVALID)
        elif np.any(candidate < minimum - tolerance) or np.any(
            candidate > maximum + tolerance
        ):
            reasons.append(SupervisorReason.COMMAND_OUT_OF_BOUNDS)
        if not controller_command_usable:
            reasons.append(SupervisorReason.CONTROLLER_UNUSABLE)
        assert tilt is not None
        assert maximum_rate is not None
        if tilt > self.config.maximum_tilt_rad:
            reasons.append(SupervisorReason.TILT_LIMIT)
        maximum_rates = np.asarray(self.config.maximum_angular_rate_rad_s)
        if np.any(np.abs(state_values[10:13]) > maximum_rates):
            reasons.append(SupervisorReason.ANGULAR_RATE_LIMIT)

        if reasons and time_valid and self._arrest_started_at_s is None:
            self._arrest_started_at_s = self._last_time_s
        latch_active = self._arrest_started_at_s is not None
        if latch_active and not reasons:
            elapsed = now_s - self._arrest_started_at_s
            below_release = tilt <= self.config.release_tilt_rad and np.all(
                np.abs(state_values[10:13])
                <= np.asarray(self.config.release_angular_rate_rad_s)
            )
            if elapsed >= self.config.minimum_arrest_duration_s and below_release:
                self._arrest_started_at_s = None
                latch_active = False
            else:
                reasons.append(SupervisorReason.ARREST_LATCHED)

        if reasons or latch_active:
            command = self._rate_arrest_command(
                state_values,
                previous_applied_command,
            )
            return SupervisedCommand(
                command=command,
                mode=SupervisorMode.RATE_ARREST,
                reasons=tuple(dict.fromkeys(reasons)),
                state_age_s=state_age,
                command_age_s=command_age,
                tilt_rad=tilt,
                maximum_angular_rate_rad_s=maximum_rate,
                nominal_command_accepted=False,
            )

        return SupervisedCommand(
            command=np.clip(candidate, minimum, maximum),
            mode=SupervisorMode.NOMINAL,
            reasons=(),
            state_age_s=state_age,
            command_age_s=command_age,
            tilt_rad=tilt,
            maximum_angular_rate_rad_s=maximum_rate,
            nominal_command_accepted=True,
        )
