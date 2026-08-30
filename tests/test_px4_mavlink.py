from collections import deque
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

from glassbox.integrations.px4 import (
    PX4HILActuatorSource,
    PX4MavlinkStateSource,
    PX4StateAssembler,
    PX4TelemetryError,
)
from glassbox.px4_frames import (
    frd_to_flu,
    ned_frd_quaternion_to_nwu_flu,
    ned_to_nwu,
)


class Message(SimpleNamespace):
    def get_type(self) -> str:
        return self.message_type

    def get_srcSystem(self) -> int:
        return getattr(self, "source_system", 1)


def position_message(
    time_boot_ms: int,
    *,
    source_system: int = 1,
) -> Message:
    return Message(
        message_type="LOCAL_POSITION_NED",
        source_system=source_system,
        time_boot_ms=time_boot_ms,
        x=1.0,
        y=2.0,
        z=-3.0,
        vx=4.0,
        vy=5.0,
        vz=-6.0,
    )


def attitude_message(
    time_boot_ms: int,
    *,
    sign: float = 1.0,
    source_system: int = 1,
) -> Message:
    return Message(
        message_type="ATTITUDE_QUATERNION",
        source_system=source_system,
        time_boot_ms=time_boot_ms,
        q1=0.5 * sign,
        q2=0.5 * sign,
        q3=0.5 * sign,
        q4=0.5 * sign,
        rollspeed=0.1,
        pitchspeed=0.2,
        yawspeed=0.3,
    )


def actuator_message(
    time_usec: int,
    controls: tuple[float, ...],
    *,
    mode: int = 145,
    source_system: int = 1,
) -> Message:
    return Message(
        message_type="HIL_ACTUATOR_CONTROLS",
        source_system=source_system,
        time_usec=time_usec,
        controls=controls,
        mode=mode,
    )


def test_shared_px4_frame_conversions_support_vectors_and_batches() -> None:
    np.testing.assert_allclose(ned_to_nwu([1.0, 2.0, 3.0]), [1.0, -2.0, -3.0])
    np.testing.assert_allclose(
        frd_to_flu([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]]),
        [[1.0, -2.0, -3.0], [-1.0, 2.0, 3.0]],
    )
    np.testing.assert_allclose(
        ned_frd_quaternion_to_nwu_flu([0.5, 0.5, 0.5, 0.5]),
        [0.5, 0.5, -0.5, -0.5],
    )

    with pytest.raises(ValueError, match="final dimension 3"):
        ned_to_nwu([1.0, 2.0])


def test_assembler_builds_canonical_state_only_after_both_streams_advance() -> None:
    assembler = PX4StateAssembler()

    assert assembler.ingest(position_message(1_000), received_monotonic_s=5.0) is None
    sample = assembler.ingest(attitude_message(1_020), received_monotonic_s=5.01)
    assert sample is not None
    np.testing.assert_allclose(
        sample.state,
        [
            1.0,
            -2.0,
            3.0,
            4.0,
            -5.0,
            6.0,
            0.5,
            0.5,
            -0.5,
            -0.5,
            0.1,
            -0.2,
            -0.3,
        ],
    )
    assert sample.message_skew_s == pytest.approx(0.02)
    assert sample.maximum_receive_age_s == pytest.approx(0.01)
    assert sample.estimated_source_clock_lag_s == 0.0

    assert assembler.ingest(position_message(1_040), received_monotonic_s=5.02) is None
    next_sample = assembler.ingest(
        attitude_message(1_040, sign=-1.0), received_monotonic_s=5.03
    )
    assert next_sample is not None
    np.testing.assert_allclose(next_sample.state[6:10], sample.state[6:10])


def test_assembler_detects_source_clock_lag_and_recovers_after_catchup() -> None:
    assembler = PX4StateAssembler()
    assert assembler.ingest(position_message(1_000), received_monotonic_s=5.0) is None
    first = assembler.ingest(attitude_message(1_000), received_monotonic_s=5.01)
    assert first is not None

    assert assembler.ingest(position_message(1_020), received_monotonic_s=5.10) is None
    delayed = assembler.ingest(attitude_message(1_020), received_monotonic_s=5.11)
    assert delayed is not None
    assert delayed.estimated_source_clock_lag_s == pytest.approx(0.08)

    assert assembler.ingest(position_message(1_200), received_monotonic_s=5.12) is None
    caught_up = assembler.ingest(attitude_message(1_200), received_monotonic_s=5.13)
    assert caught_up is not None
    assert caught_up.estimated_source_clock_lag_s == 0.0


def test_assembler_rejects_stale_or_misaligned_pairs_until_both_refresh() -> None:
    assembler = PX4StateAssembler(
        maximum_message_skew_s=0.05,
        maximum_receive_age_s=0.10,
    )
    assert assembler.ingest(position_message(1_000), received_monotonic_s=0.0) is None
    assert assembler.ingest(attitude_message(1_000), received_monotonic_s=0.11) is None
    assert assembler.ingest(position_message(1_200), received_monotonic_s=0.12) is None
    sample = assembler.ingest(attitude_message(1_220), received_monotonic_s=0.13)
    assert sample is not None

    assert assembler.ingest(position_message(2_000), received_monotonic_s=0.14) is None
    assert assembler.ingest(attitude_message(2_100), received_monotonic_s=0.15) is None


def test_assembler_handles_px4_boot_timestamp_wraparound() -> None:
    assembler = PX4StateAssembler(maximum_message_skew_s=0.05)
    assert (
        assembler.ingest(position_message(2**32 - 10), received_monotonic_s=1.0) is None
    )
    sample = assembler.ingest(attitude_message(10), received_monotonic_s=1.01)
    assert sample is not None
    assert sample.message_skew_s == pytest.approx(0.02)


def test_assembler_rejects_invalid_quaternion() -> None:
    assembler = PX4StateAssembler()
    invalid = attitude_message(1_000)
    invalid.q1 = invalid.q2 = invalid.q3 = invalid.q4 = 0.0
    assert assembler.ingest(position_message(1_000), received_monotonic_s=1.0) is None
    with pytest.raises(PX4TelemetryError, match="invalid attitude quaternion"):
        assembler.ingest(invalid, received_monotonic_s=1.01)


def test_state_source_filters_systems_and_has_no_transmit_surface() -> None:
    class Connection:
        def __init__(self) -> None:
            self.messages = deque(
                (
                    position_message(1_000, source_system=2),
                    attitude_message(1_000, source_system=2),
                    position_message(1_000),
                    attitude_message(1_000),
                    position_message(2_000),
                    attitude_message(2_000),
                )
            )
            self.received_types: list[list[str]] = []
            self.blocking_modes: list[bool] = []
            self.closed = False
            self.closed_event = Event()

        def recv_match(
            self,
            *,
            type: list[str],
            blocking: bool,
            timeout: float,
        ) -> Message | None:
            assert timeout >= 0.0
            self.received_types.append(type)
            self.blocking_modes.append(blocking)
            if self.messages:
                return self.messages.popleft()
            if blocking:
                self.closed_event.wait(timeout)
            return None

        def close(self) -> None:
            self.closed = True
            self.closed_event.set()

    connection = Connection()
    with PX4MavlinkStateSource(connection, source_system=1) as source:
        sample = source.next_sample(timeout_s=0.1)

    assert sample.position_time_boot_ms == 2_000
    assert connection.received_types
    assert connection.blocking_modes[0]
    assert False in connection.blocking_modes
    assert connection.closed
    assert not hasattr(source, "send")


def test_hil_actuator_source_maps_fresh_commands_without_transmission() -> None:
    class Connection:
        def __init__(self) -> None:
            self.messages = deque(
                (
                    actuator_message(
                        900_000,
                        (0.9, 0.9, 0.9, 0.9),
                        source_system=2,
                    ),
                    actuator_message(1_000_000, (0.1, 0.2, 0.3, 0.4)),
                    actuator_message(1_020_000, (0.2, 0.3, 0.4, 0.5)),
                )
            )
            self.closed = False
            self.closed_event = Event()

        def recv_match(
            self,
            *,
            type: list[str],
            blocking: bool,
            timeout: float,
        ) -> Message | None:
            assert type == ["HIL_ACTUATOR_CONTROLS"]
            assert timeout >= 0.0
            if self.messages:
                return self.messages.popleft()
            if blocking:
                self.closed_event.wait(timeout)
            return None

        def close(self) -> None:
            self.closed = True
            self.closed_event.set()

    connection = Connection()
    with PX4HILActuatorSource(
        connection,
        command_indices=(2, 0, 3, 1),
        source_system=1,
    ) as source:
        sample = source.next_sample(timeout_s=0.1)
        nearest = source.sample_nearest(1_001, timeout_s=0.1)

    np.testing.assert_allclose(sample.command, [0.4, 0.2, 0.5, 0.3])
    assert sample.source_time_us == 1_020_000
    assert sample.armed
    assert sample.mav_mode == 145
    assert 0.0 <= sample.receive_age_s <= 0.25
    np.testing.assert_allclose(nearest.command, [0.3, 0.1, 0.4, 0.2])
    assert nearest.source_time_us == 1_000_000
    assert connection.closed
    assert not hasattr(source, "send")


def test_hil_actuator_source_rejects_invalid_mapping() -> None:
    with pytest.raises(ValueError, match="distinct nonnegative"):
        PX4HILActuatorSource(object(), command_indices=(0, 0))  # type: ignore[arg-type]
