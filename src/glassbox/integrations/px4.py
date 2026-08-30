"""Read-only PX4 MAVLink telemetry integration for shadow-mode evaluation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from threading import Condition, Event, Thread
from typing import Protocol

import numpy as np

from glassbox.px4_frames import (
    frd_to_flu,
    ned_frd_quaternion_to_nwu_flu,
    ned_to_nwu,
)

PX4_STATE_MESSAGE_TYPES = ("LOCAL_POSITION_NED", "ATTITUDE_QUATERNION")
_BOOT_TIME_MODULUS_MS = 2**32
_RECEIVE_POLL_TIMEOUT_S = 0.10
_MAXIMUM_DRAIN_MESSAGES = 4096


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
    estimated_source_clock_lag_s: float = 0.0

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
        timing = np.asarray(
            [
                self.message_skew_s,
                self.maximum_receive_age_s,
                self.estimated_source_clock_lag_s,
            ]
        )
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
        self._last_position_boot_ms: int | None = None
        self._unwrapped_position_boot_ms = 0
        self._minimum_clock_offset_s: float | None = None

    def _source_clock_lag_s(self, boot_ms: int, received_at_s: float) -> float:
        previous_boot_ms = self._last_position_boot_ms
        if previous_boot_ms is None:
            self._unwrapped_position_boot_ms = boot_ms
            self._minimum_clock_offset_s = None
        else:
            advance_ms = (
                (boot_ms - previous_boot_ms + _BOOT_TIME_MODULUS_MS // 2)
                % _BOOT_TIME_MODULUS_MS
                - _BOOT_TIME_MODULUS_MS // 2
            )
            if advance_ms < 0:
                # A negative source-time jump indicates a PX4 restart. Re-anchor
                # rather than treating the new boot as years of transport lag.
                self._unwrapped_position_boot_ms = boot_ms
                self._minimum_clock_offset_s = None
            else:
                self._unwrapped_position_boot_ms += advance_ms
        self._last_position_boot_ms = boot_ms

        clock_offset_s = received_at_s - self._unwrapped_position_boot_ms * 1e-3
        if (
            self._minimum_clock_offset_s is None
            or clock_offset_s < self._minimum_clock_offset_s
        ):
            self._minimum_clock_offset_s = clock_offset_s
        return max(0.0, clock_offset_s - self._minimum_clock_offset_s)

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
            estimated_source_clock_lag_s=self._source_clock_lag_s(
                position_boot_ms, now_s
            ),
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
        self._condition = Condition()
        self._stop_requested = Event()
        self._latest_sample: PX4StateSample | None = None
        self._latest_sequence = 0
        self._delivered_sequence = 0
        self._receiver_error: Exception | None = None
        self._closed = False
        self._receiver = Thread(
            target=self._receive_forever,
            name="glassbox-px4-telemetry",
            daemon=True,
        )
        self._receiver.start()

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

    def _ingest_selected(
        self, message: _MAVLinkMessage
    ) -> PX4StateSample | None:
        if (
            self._source_system is not None
            and int(message.get_srcSystem()) != self._source_system
        ):
            return None
        return self._assembler.ingest(message)

    def _receive_forever(self) -> None:
        try:
            while not self._stop_requested.is_set():
                message = self._connection.recv_match(
                    type=list(PX4_STATE_MESSAGE_TYPES),
                    blocking=True,
                    timeout=_RECEIVE_POLL_TIMEOUT_S,
                )
                if message is None:
                    continue
                latest = self._ingest_selected(message)
                for _ in range(_MAXIMUM_DRAIN_MESSAGES):
                    pending = self._connection.recv_match(
                        type=list(PX4_STATE_MESSAGE_TYPES),
                        blocking=False,
                        timeout=0.0,
                    )
                    if pending is None:
                        break
                    sample = self._ingest_selected(pending)
                    if sample is not None:
                        latest = sample
                if latest is None:
                    continue
                with self._condition:
                    self._latest_sample = latest
                    self._latest_sequence += 1
                    self._condition.notify_all()
        except Exception as error:
            with self._condition:
                if not self._closed:
                    self._receiver_error = error
                    self._condition.notify_all()

    def next_sample(self, *, timeout_s: float = 1.0) -> PX4StateSample:
        """Wait for the next fresh, time-aligned canonical state sample."""

        if not np.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be finite and positive")
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while True:
                if self._receiver_error is not None:
                    raise PX4TelemetryError(
                        "PX4 telemetry receiver failed"
                    ) from self._receiver_error
                if self._closed:
                    raise PX4TelemetryError("PX4 telemetry source is closed")
                if self._latest_sequence > self._delivered_sequence:
                    sample = self._latest_sample
                    if sample is None:
                        raise PX4TelemetryError(
                            "PX4 telemetry receiver published an empty sample"
                        )
                    self._delivered_sequence = self._latest_sequence
                    return sample
                remaining_s = deadline - time.monotonic()
                if remaining_s <= 0.0:
                    raise PX4TelemetryError(
                        "timed out waiting for fresh LOCAL_POSITION_NED and "
                        "ATTITUDE_QUATERNION messages"
                    )
                self._condition.wait(timeout=remaining_s)

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._stop_requested.set()
            self._condition.notify_all()
        self._connection.close()
        self._receiver.join(timeout=1.0)

    def __enter__(self) -> PX4MavlinkStateSource:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
