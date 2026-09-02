"""Fly bounded PX4 fixed-wing attitude/throttle profiles over MAVLink."""

from __future__ import annotations

import argparse
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

try:
    from pymavlink import mavutil
except ImportError as error:  # pragma: no cover - exercised without the extra
    raise ImportError(
        "PX4 SITL recording needs the 'px4' extra: "
        "uv sync --extra px4, or pip install 'glassbox[px4]'"
    ) from error

from glassbox.core.geometry import quaternion_from_euler


@dataclass(frozen=True)
class AttitudeTarget:
    roll_deg: float
    pitch_deg: float
    throttle: float
    duration_s: float


@dataclass(frozen=True)
class ExcitationCondition:
    attitude_scale: float
    throttle_scale: float
    dwell_s: float


CONDITIONS: dict[str, ExcitationCondition] = {
    "low": ExcitationCondition(0.55, 0.55, 3.0),
    "medium": ExcitationCondition(1.0, 1.0, 3.0),
    "high": ExcitationCondition(1.35, 1.25, 2.5),
}

TRIM_PITCH_DEG = 5.0
TRIM_THROTTLE = 0.88

PROFILES: dict[str, tuple[AttitudeTarget, ...]] = {
    "throttle_steps": (
        AttitudeTarget(0.0, 5.0, 0.78, 3.0),
        AttitudeTarget(0.0, 5.0, 0.96, 3.0),
        AttitudeTarget(0.0, 5.0, 0.82, 3.0),
        AttitudeTarget(0.0, 5.0, 1.00, 3.0),
        AttitudeTarget(0.0, 5.0, 0.88, 3.0),
    ),
    "roll_steps": (
        AttitudeTarget(-10.0, 5.0, 0.90, 3.0),
        AttitudeTarget(10.0, 5.0, 0.90, 3.0),
        AttitudeTarget(-16.0, 5.0, 0.92, 3.0),
        AttitudeTarget(16.0, 5.0, 0.92, 3.0),
        AttitudeTarget(0.0, 5.0, 0.88, 3.0),
    ),
    "pitch_steps": (
        AttitudeTarget(0.0, 1.0, 0.90, 3.0),
        AttitudeTarget(0.0, 9.0, 0.94, 3.0),
        AttitudeTarget(0.0, -2.0, 0.86, 3.0),
        AttitudeTarget(0.0, 11.0, 0.96, 3.0),
        AttitudeTarget(0.0, 5.0, 0.88, 3.0),
    ),
    "combined": (
        AttitudeTarget(-10.0, 8.0, 0.94, 3.0),
        AttitudeTarget(12.0, 2.0, 0.82, 3.0),
        AttitudeTarget(-16.0, 0.0, 0.96, 3.0),
        AttitudeTarget(16.0, 10.0, 0.90, 3.0),
        AttitudeTarget(0.0, 5.0, 0.88, 3.0),
    ),
}


def profile_targets(
    profile: str, *, condition: str = "medium"
) -> tuple[AttitudeTarget, ...]:
    """Scale one profile around the fixed-wing trim operating point."""

    excitation = CONDITIONS[condition]
    return tuple(
        AttitudeTarget(
            roll_deg=target.roll_deg * excitation.attitude_scale,
            pitch_deg=TRIM_PITCH_DEG
            + (target.pitch_deg - TRIM_PITCH_DEG) * excitation.attitude_scale,
            throttle=float(
                min(
                    1.0,
                    max(
                        0.0,
                        TRIM_THROTTLE
                        + (target.throttle - TRIM_THROTTLE) * excitation.throttle_scale,
                    ),
                )
            ),
            duration_s=excitation.dwell_s,
        )
        for target in PROFILES[profile]
    )


def _quaternion_from_euler(
    roll_rad: float, pitch_rad: float, yaw_rad: float
) -> tuple[float, float, float, float]:
    """Return the shared WXYZ attitude quaternion as a MAVLink-ready tuple."""

    w, x, y, z = quaternion_from_euler(roll_rad, pitch_rad, yaw_rad)
    return (float(w), float(x), float(y), float(z))


def _attitude_mask() -> int:
    return int(
        mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_BODY_ROLL_RATE_IGNORE
        | mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_BODY_PITCH_RATE_IGNORE
        | mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_BODY_YAW_RATE_IGNORE
    )


def _send_target(connection: Any, target: AttitudeTarget, yaw_rad: float) -> None:
    quaternion = _quaternion_from_euler(
        math.radians(target.roll_deg), math.radians(target.pitch_deg), yaw_rad
    )
    connection.mav.set_attitude_target_send(
        int(time.monotonic() * 1000.0) & 0xFFFFFFFF,
        connection.target_system,
        connection.target_component,
        _attitude_mask(),
        quaternion,
        0.0,
        0.0,
        0.0,
        target.throttle,
    )


def _stream_target(
    connection: Any, target: AttitudeTarget, yaw_rad: float, rate_hz: float
) -> None:
    period_s = 1.0 / rate_hz
    stop_time = time.monotonic() + target.duration_s
    while time.monotonic() < stop_time:
        started = time.monotonic()
        _send_target(connection, target, yaw_rad)
        connection.recv_match(blocking=False)
        time.sleep(max(0.0, period_s - (time.monotonic() - started)))


def _current_yaw_rad(connection: Any, timeout_s: float) -> float:
    attitude = connection.recv_match(type="ATTITUDE", blocking=True, timeout=timeout_s)
    if attitude is None:
        raise RuntimeError("PX4 did not publish ATTITUDE before profile start")
    return float(attitude.yaw)


def fly_profile(
    profile: str,
    *,
    condition: str = "medium",
    connection_string: str = "udpin:0.0.0.0:14550",
    rate_hz: float = 20.0,
    heartbeat_timeout_s: float = 30.0,
) -> None:
    """Take over an already-airborne plane and stream one bounded profile."""

    targets = profile_targets(profile, condition=condition)
    connection = mavutil.mavlink_connection(connection_string)
    heartbeat = connection.wait_heartbeat(timeout=heartbeat_timeout_s)
    if heartbeat is None:
        raise RuntimeError(f"no PX4 heartbeat on {connection_string}")
    if not connection.motors_armed():
        raise RuntimeError("fixed-wing profile requires an already-armed airplane")
    yaw_rad = _current_yaw_rad(connection, heartbeat_timeout_s)
    print(
        f"connected to system {connection.target_system}, "
        f"component {connection.target_component}"
    )
    print(f"profile={profile} condition={condition}")

    warmup = AttitudeTarget(0.0, TRIM_PITCH_DEG, TRIM_THROTTLE, 1.5)
    _stream_target(connection, warmup, yaw_rad, rate_hz)
    modes = connection.mode_mapping()
    if not modes or "OFFBOARD" not in modes:
        raise RuntimeError("PX4 did not advertise OFFBOARD mode")
    offboard_mode = modes["OFFBOARD"]
    if isinstance(offboard_mode, tuple):
        connection.set_mode_px4(*offboard_mode)
    else:
        connection.set_mode(offboard_mode)

    for index, target in enumerate(targets, start=1):
        print(
            f"target {index}/{len(targets)}: roll={target.roll_deg:g}deg "
            f"pitch={target.pitch_deg:g}deg throttle={target.throttle:g}"
        )
        _stream_target(connection, target, yaw_rad, rate_hz)

    print("profile complete; holding trim")
    _stream_target(connection, warmup, yaw_rad, rate_hz)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=tuple(PROFILES))
    parser.add_argument("--condition", choices=tuple(CONDITIONS), default="medium")
    parser.add_argument(
        "--connection", default="udpin:0.0.0.0:14550", help="pymavlink endpoint"
    )
    parser.add_argument("--rate", type=float, default=20.0)
    args = parser.parse_args(argv)
    fly_profile(
        args.profile,
        condition=args.condition,
        connection_string=args.connection,
        rate_hz=args.rate,
    )


if __name__ == "__main__":
    main()
