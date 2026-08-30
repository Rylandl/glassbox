"""Read-only PX4 MAVLink telemetry integration for shadow-mode evaluation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from glassbox.px4_frames import (
    frd_to_flu,
    ned_frd_quaternion_to_nwu_flu,
    ned_to_nwu,
)

PX4_STATE_MESSAGE_TYPES = ("LOCAL_POSITION_NED", "ATTITUDE_QUATERNION")
_BOOT_TIME_MODULUS_MS = 2**32


class PX4TelemetryError(RuntimeError):
    """Raised when PX4 cannot provide a fresh canonical state sample."""


class _MAVLinkMessage(Protocol):
    time_boot_ms: int

    def get_type(self) -> str: ...

    def get_srcSystem(self) -> int: ...


class _MAVLinkConnection(Protocol):
    def wait_heartbeat(self, *, timeout: float) -> object | None: ...

    def recv_match(
        self,
        *,
        type: list[str],
        blocking: bool,
        timeout: float,
    ) -> _MAVLinkMessage | None: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class PX4StateSample:
    """One timestamp-audited canonical rigid-body state from PX4."""

    state: np.ndarray
    position_time_boot_ms: int
    attitude_time_boot_ms: int
    message_skew_s: float
    maximum_receive_age_s: float

    def __post_init__(self) -> None:
        state = np.asarray(self.state, dtype=np.float64)
        if state.shape != (13,) or not np.all(np.isfinite(state)):
            raise ValueError("PX4 canonical state must contain 13 finite values")
        if not np.isclose(np.linalg.norm(state[6:10]), 1.0, atol=1e-9):
            raise ValueError("PX4 canonical attitude must be a unit quaternion")
        if not 0 <= self.position_time_boot_ms < _BOOT_TIME_MODULUS_MS:
            raise ValueError("PX4 position boot timestamp is outside uint32 range")
        if not 0 <= self.attitude_time_boot_ms < _BOOT_TIME_MODULUS_MS:
            raise ValueError("PX4 attitude boot timestamp is outside uint32 range")
        timing = np.asarray([self.message_skew_s, self.maximum_receive_age_s])
        if not np.all(np.isfinite(timing)) or np.any(timing < 0.0):
            raise ValueError(
                "PX4 state timing diagnostics must be finite and nonnegative"
            )
        object.__setattr__(self, "state", state)


@dataclass(frozen=True)
class _ReceivedMessage:
    message: _MAVLinkMessage
    received_monotonic_s: float
    generation: int


def _boot_time_skew_s(first_ms: int, second_ms: int) -> float:
    difference = (
        (int(first_ms) - int(second_ms) + _BOOT_TIME_MODULUS_MS // 2)
        % _BOOT_TIME_MODULUS_MS
        - _BOOT_TIME_MODULUS_MS // 2
    )
    return abs(difference) * 1e-3


class PX4StateAssembler:
    """Pair asynchronous PX4 state messages into fresh canonical samples."""

    def __init__(
        self,
        *,
        maximum_message_skew_s: float = 0.10,
        maximum_receive_age_s: float = 0.25,
    ) -> None:
        if not np.isfinite(maximum_message_skew_s) or maximum_message_skew_s <= 0:
            raise ValueError("maximum_message_skew_s must be finite and positive")
        if not np.isfinite(maximum_receive_age_s) or maximum_receive_age_s <= 0:
            raise ValueError("maximum_receive_age_s must be finite and positive")
        self.maximum_message_skew_s = float(maximum_message_skew_s)
        self.maximum_receive_age_s = float(maximum_receive_age_s)
        self._position: _ReceivedMessage | None = None
        self._attitude: _ReceivedMessage | None = None
        self._generation = 0
        self._emitted_generations = (-1, -1)
        self._previous_quaternion: np.ndarray | None = None

    def ingest(
        self,
        message: _MAVLinkMessage,
        *,
        received_monotonic_s: float | None = None,
    ) -> PX4StateSample | None:
        """Consume one message and return a state once both streams advance."""

        message_type = message.get_type()
        if message_type not in PX4_STATE_MESSAGE_TYPES:
            return None
        received_at = (
            time.monotonic()
            if received_monotonic_s is None
            else float(received_monotonic_s)
        )
        if not np.isfinite(received_at):
            raise ValueError("received_monotonic_s must be finite")
        self._generation += 1
        received = _ReceivedMessage(message, received_at, self._generation)
        if message_type == "LOCAL_POSITION_NED":
            self._position = received
        else:
            self._attitude = received
        return self._assemble(received_at)

    def _assemble(self, now_s: float) -> PX4StateSample | None:
        position = self._position
        attitude = self._attitude
        if position is None or attitude is None:
            return None
        generations = (position.generation, attitude.generation)
        if any(
            current <= emitted
            for current, emitted in zip(generations, self._emitted_generations)
        ):
            return None

        ages = np.asarray(
            [
                now_s - position.received_monotonic_s,
                now_s - attitude.received_monotonic_s,
            ]
        )
        if np.any(ages < 0.0) or np.max(ages) > self.maximum_receive_age_s:
            return None

        position_boot_ms = int(position.message.time_boot_ms)
        attitude_boot_ms = int(attitude.message.time_boot_ms)
        skew_s = _boot_time_skew_s(position_boot_ms, attitude_boot_ms)
        if skew_s > self.maximum_message_skew_s:
            return None

        world_position = ned_to_nwu(
            [position.message.x, position.message.y, position.message.z]
        )
        world_velocity = ned_to_nwu(
            [position.message.vx, position.message.vy, position.message.vz]
        )
        quaternion = ned_frd_quaternion_to_nwu_flu(
            [
                attitude.message.q1,
                attitude.message.q2,
                attitude.message.q3,
                attitude.message.q4,
            ]
        )
        quaternion_norm = np.linalg.norm(quaternion)
        if not np.isfinite(quaternion_norm) or quaternion_norm < 1e-6:
            raise PX4TelemetryError("PX4 supplied an invalid attitude quaternion")
        quaternion = quaternion / quaternion_norm
        if (
            self._previous_quaternion is not None
            and np.dot(self._previous_quaternion, quaternion) < 0.0
        ):
            quaternion = -quaternion
        body_rates = frd_to_flu(
            [
                attitude.message.rollspeed,
                attitude.message.pitchspeed,
                attitude.message.yawspeed,
            ]
        )
        state = np.concatenate(
            (world_position, world_velocity, quaternion, body_rates)
        )
        if not np.all(np.isfinite(state)):
            raise PX4TelemetryError("PX4 supplied a non-finite state")

        self._previous_quaternion = quaternion
        self._emitted_generations = generations
        return PX4StateSample(
            state=state,
            position_time_boot_ms=position_boot_ms,
            attitude_time_boot_ms=attitude_boot_ms,
            message_skew_s=skew_s,
            maximum_receive_age_s=float(np.max(ages)),
        )


class PX4MavlinkStateSource:
    """Receive PX4 state estimates without transmitting MAVLink messages."""

    def __init__(
        self,
        connection: _MAVLinkConnection,
        *,
        assembler: PX4StateAssembler | None = None,
        source_system: int | None = None,
    ) -> None:
        self._connection = connection
        self._assembler = PX4StateAssembler() if assembler is None else assembler
        self._source_system = source_system

    @classmethod
    def connect(
        cls,
        connection_string: str = "udpin:0.0.0.0:14550",
        *,
        heartbeat_timeout_s: float = 10.0,
        assembler: PX4StateAssembler | None = None,
    ) -> PX4MavlinkStateSource:
        """Open a passive MAVLink listener and wait for a PX4 heartbeat."""

        if not np.isfinite(heartbeat_timeout_s) or heartbeat_timeout_s <= 0:
            raise ValueError("heartbeat_timeout_s must be finite and positive")
        from pymavlink import mavutil

        connection = mavutil.mavlink_connection(connection_string)
        heartbeat = connection.wait_heartbeat(timeout=float(heartbeat_timeout_s))
        if heartbeat is None:
            connection.close()
            raise PX4TelemetryError(
                f"no PX4 heartbeat received from {connection_string!r}"
            )
        if int(getattr(heartbeat, "autopilot", -1)) != int(
            mavutil.mavlink.MAV_AUTOPILOT_PX4
        ):
            connection.close()
            raise PX4TelemetryError(
                f"heartbeat from {connection_string!r} is not a PX4 autopilot"
            )
        return cls(
            connection,
            assembler=assembler,
            source_system=int(heartbeat.get_srcSystem()),
        )

    def next_sample(self, *, timeout_s: float = 1.0) -> PX4StateSample:
        """Wait for the next fresh, time-aligned canonical state sample."""

        if not np.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be finite and positive")
        deadline = time.monotonic() + timeout_s
        while True:
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0.0:
                raise PX4TelemetryError(
                    "timed out waiting for fresh LOCAL_POSITION_NED and "
                    "ATTITUDE_QUATERNION messages"
                )
            message = self._connection.recv_match(
                type=list(PX4_STATE_MESSAGE_TYPES),
                blocking=True,
                timeout=remaining_s,
            )
            if message is None:
                continue
            if (
                self._source_system is not None
                and int(message.get_srcSystem()) != self._source_system
            ):
                continue
            sample = self._assembler.ingest(message)
            if sample is not None:
                return sample

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> PX4MavlinkStateSource:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
