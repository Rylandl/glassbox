"""Fly bounded PX4 SITL position/yaw profiles over MAVLink."""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from typing import Any

from pymavlink import mavutil


@dataclass(frozen=True)
class PositionTarget:
    north_m: float
    east_m: float
    down_m: float
    yaw_deg: float
    duration_s: float


@dataclass(frozen=True)
class ExcitationCondition:
    translation_scale: float
    yaw_scale: float
    dwell_s: float


CONDITIONS: dict[str, ExcitationCondition] = {
    "low": ExcitationCondition(translation_scale=0.6, yaw_scale=0.5, dwell_s=4.0),
    "medium": ExcitationCondition(
        translation_scale=1.0, yaw_scale=1.0, dwell_s=3.0
    ),
    "high": ExcitationCondition(
        translation_scale=1.4, yaw_scale=1.0, dwell_s=2.0
    ),
}


PROFILES: dict[str, tuple[PositionTarget, ...]] = {
    "vertical_steps": (
        PositionTarget(0.0, 0.0, -1.0, 0.0, 3.0),
        PositionTarget(0.0, 0.0, -2.0, 0.0, 3.0),
        PositionTarget(0.0, 0.0, -1.2, 0.0, 3.0),
        PositionTarget(0.0, 0.0, -2.2, 0.0, 3.0),
        PositionTarget(0.0, 0.0, -1.5, 0.0, 3.0),
    ),
    "lateral_steps": (
        PositionTarget(0.0, 0.0, -1.5, 0.0, 3.0),
        PositionTarget(2.0, 0.0, -1.5, 0.0, 3.0),
        PositionTarget(-2.0, 0.0, -1.5, 0.0, 3.0),
        PositionTarget(0.0, 2.0, -1.5, 0.0, 3.0),
        PositionTarget(0.0, -2.0, -1.5, 0.0, 3.0),
        PositionTarget(0.0, 0.0, -1.5, 0.0, 3.0),
    ),
    "yaw_steps": (
        PositionTarget(0.0, 0.0, -1.5, 0.0, 3.0),
        PositionTarget(0.0, 0.0, -1.5, 90.0, 3.0),
        PositionTarget(0.0, 0.0, -1.5, -90.0, 3.0),
        PositionTarget(0.0, 0.0, -1.5, 180.0, 3.0),
        PositionTarget(0.0, 0.0, -1.5, 0.0, 3.0),
    ),
    "combined": (
        PositionTarget(0.0, 0.0, -1.5, 0.0, 3.0),
        PositionTarget(1.5, 1.5, -1.2, 45.0, 3.0),
        PositionTarget(-1.5, 1.5, -2.0, 135.0, 3.0),
        PositionTarget(-1.5, -1.5, -1.2, -135.0, 3.0),
        PositionTarget(1.5, -1.5, -2.0, -45.0, 3.0),
        PositionTarget(0.0, 0.0, -1.5, 0.0, 3.0),
    ),
}


def _wrapped_yaw_deg(yaw_deg: float) -> float:
    return (yaw_deg + 180.0) % 360.0 - 180.0


def profile_targets(
    profile: str,
    *,
    condition: str = "medium",
    initial_yaw_deg: float = 0.0,
) -> tuple[PositionTarget, ...]:
    """Scale and rotate one base profile into an excitation condition."""

    excitation = CONDITIONS[condition]
    yaw_rad = math.radians(initial_yaw_deg)
    cosine = math.cos(yaw_rad)
    sine = math.sin(yaw_rad)
    targets = []
    for target in PROFILES[profile]:
        north = excitation.translation_scale * target.north_m
        east = excitation.translation_scale * target.east_m
        rotated_north = cosine * north - sine * east
        rotated_east = sine * north + cosine * east
        nominal_down_m = -1.5
        scaled_down = nominal_down_m + excitation.translation_scale * (
            target.down_m - nominal_down_m
        )
        targets.append(
            PositionTarget(
                north_m=rotated_north,
                east_m=rotated_east,
                down_m=scaled_down,
                yaw_deg=_wrapped_yaw_deg(
                    initial_yaw_deg + excitation.yaw_scale * target.yaw_deg
                ),
                duration_s=excitation.dwell_s,
            )
        )
    return tuple(targets)


def _position_mask() -> int:
    ignored = (
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_VX_IGNORE,
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_VY_IGNORE,
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_VZ_IGNORE,
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE,
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE,
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE,
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE,
    )
    return int(sum(ignored))


def _send_target(connection: Any, target: PositionTarget) -> None:
    connection.mav.set_position_target_local_ned_send(
        int(time.monotonic() * 1000.0) & 0xFFFFFFFF,
        connection.target_system,
        connection.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        _position_mask(),
        target.north_m,
        target.east_m,
        target.down_m,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        math.radians(target.yaw_deg),
        0.0,
    )


def _stream_target(connection: Any, target: PositionTarget, rate_hz: float) -> None:
    period_s = 1.0 / rate_hz
    stop_time = time.monotonic() + target.duration_s
    while time.monotonic() < stop_time:
        started = time.monotonic()
        _send_target(connection, target)
        connection.recv_match(blocking=False)
        time.sleep(max(0.0, period_s - (time.monotonic() - started)))


def _command_arm(connection: Any, arm: bool) -> None:
    connection.mav.command_long_send(
        connection.target_system,
        connection.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0,
        1.0 if arm else 0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )


def _command_land(connection: Any) -> None:
    connection.mav.command_long_send(
        connection.target_system,
        connection.target_component,
        mavutil.mavlink.MAV_CMD_NAV_LAND,
        0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )


def fly_profile(
    profile: str,
    *,
    condition: str = "medium",
    initial_yaw_deg: float = 0.0,
    connection_string: str = "udpin:0.0.0.0:14550",
    rate_hz: float = 20.0,
    heartbeat_timeout_s: float = 30.0,
    landing_timeout_s: float = 20.0,
) -> None:
    """Connect to one PX4 SITL instance, fly a profile, land, and disarm."""

    targets = profile_targets(
        profile, condition=condition, initial_yaw_deg=initial_yaw_deg
    )
    connection = mavutil.mavlink_connection(connection_string)
    heartbeat = connection.wait_heartbeat(timeout=heartbeat_timeout_s)
    if heartbeat is None:
        raise RuntimeError(f"no PX4 heartbeat on {connection_string}")
    print(
        f"connected to system {connection.target_system}, "
        f"component {connection.target_component}"
    )
    print(
        f"profile={profile} condition={condition} "
        f"initial_yaw={initial_yaw_deg:g}deg"
    )

    # PX4 requires a setpoint stream before it will enter offboard mode.
    warmup_target = PositionTarget(
        targets[0].north_m,
        targets[0].east_m,
        targets[0].down_m,
        targets[0].yaw_deg,
        1.5,
    )
    _stream_target(connection, warmup_target, rate_hz)
    modes = connection.mode_mapping()
    if not modes or "OFFBOARD" not in modes:
        raise RuntimeError("PX4 did not advertise OFFBOARD mode")
    offboard_mode = modes["OFFBOARD"]
    if isinstance(offboard_mode, tuple):
        connection.set_mode_px4(*offboard_mode)
    else:
        connection.set_mode(offboard_mode)
    if not connection.motors_armed():
        _command_arm(connection, True)

    armed = connection.motors_armed()
    arm_deadline = time.monotonic() + 8.0
    while time.monotonic() < arm_deadline:
        _send_target(connection, targets[0])
        message = connection.recv_match(type="HEARTBEAT", blocking=False)
        if message is not None and message.get_srcSystem() == connection.target_system:
            armed = bool(
                message.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
            )
            if armed:
                break
        time.sleep(1.0 / rate_hz)
    if not armed:
        raise RuntimeError("PX4 did not arm in OFFBOARD mode")

    try:
        for index, target in enumerate(targets, start=1):
            print(
                f"target {index}/{len(targets)}: "
                f"NED=({target.north_m:g}, {target.east_m:g}, "
                f"{target.down_m:g})m yaw={target.yaw_deg:g}deg"
            )
            _stream_target(connection, target, rate_hz)
    finally:
        print("landing")
        _command_land(connection)
        landing_deadline = time.monotonic() + landing_timeout_s
        while time.monotonic() < landing_deadline:
            message = connection.recv_match(type="HEARTBEAT", blocking=True, timeout=1.0)
            if message is not None and not (
                message.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
            ):
                print("landed and disarmed")
                return
        _command_arm(connection, False)
        raise RuntimeError("landing timed out; sent disarm command")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=tuple(PROFILES))
    parser.add_argument("--condition", choices=tuple(CONDITIONS), default="medium")
    parser.add_argument("--initial-yaw", type=float, default=0.0)
    parser.add_argument(
        "--connection", default="udpin:0.0.0.0:14550", help="pymavlink endpoint"
    )
    parser.add_argument("--rate", type=float, default=20.0)
    args = parser.parse_args()
    fly_profile(
        args.profile,
        condition=args.condition,
        initial_yaw_deg=args.initial_yaw,
        connection_string=args.connection,
        rate_hz=args.rate,
    )


if __name__ == "__main__":
    main()
