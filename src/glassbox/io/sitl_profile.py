"""One PX4 SITL recorder for both vehicle families.

Recording a scripted maneuver profile is the same job on a multirotor and on
an airplane: connect to one SITL instance, stream a bounded setpoint at a
fixed rate through a table of dwell targets, and leave the vehicle where the
script found it. The two families differ in the setpoint message and in what
the excitation condition scales, so each declares its own target type, profile
table and condition table, and the loop is written once.

The multirotor family streams local NED position and yaw setpoints; it takes
off into OFFBOARD, flies the profile, then lands and disarms even when the
profile itself failed. The fixed-wing family streams attitude and throttle
setpoints and requires an already-airborne, already-armed airplane, because a
plane cannot hold station while the stream is established; it hands the plane
back holding trim.

Every profile is bounded: the targets are a fixed table, the excitation
condition only scales them, and neither family sends anything outside the
declared envelope.
"""

from __future__ import annotations

import math
import time
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

FAMILIES = ("multirotor", "fixedwing")


@dataclass(frozen=True)
class PositionTarget:
    """One multirotor dwell target in local NED plus a heading."""

    north_m: float
    east_m: float
    down_m: float
    yaw_deg: float
    duration_s: float


@dataclass(frozen=True)
class AttitudeTarget:
    """One fixed-wing dwell target as an attitude and a throttle."""

    roll_deg: float
    pitch_deg: float
    throttle: float
    duration_s: float


@dataclass(frozen=True)
class PositionExcitation:
    translation_scale: float
    yaw_scale: float
    dwell_s: float


@dataclass(frozen=True)
class AttitudeExcitation:
    attitude_scale: float
    throttle_scale: float
    dwell_s: float


NOMINAL_DOWN_M = -1.5
TRIM_PITCH_DEG = 5.0
TRIM_THROTTLE = 0.88

CONDITIONS: dict[str, dict[str, Any]] = {
    "multirotor": {
        "low": PositionExcitation(translation_scale=0.6, yaw_scale=0.5, dwell_s=4.0),
        "medium": PositionExcitation(translation_scale=1.0, yaw_scale=1.0, dwell_s=3.0),
        "high": PositionExcitation(translation_scale=1.4, yaw_scale=1.0, dwell_s=2.0),
    },
    "fixedwing": {
        "low": AttitudeExcitation(0.55, 0.55, 3.0),
        "medium": AttitudeExcitation(1.0, 1.0, 3.0),
        "high": AttitudeExcitation(1.35, 1.25, 2.5),
    },
}

PROFILES: dict[str, dict[str, tuple[Any, ...]]] = {
    "multirotor": {
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
    },
    "fixedwing": {
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
    },
}


def family_of(profile: str) -> str:
    """Return the one family that declares ``profile``."""

    families = [name for name, table in PROFILES.items() if profile in table]
    if len(families) == 1:
        return families[0]
    if not families:
        raise ValueError(f"unknown profile {profile!r}")
    raise ValueError(f"profile {profile!r} is declared by more than one family")


def _wrapped_yaw_deg(yaw_deg: float) -> float:
    return (yaw_deg + 180.0) % 360.0 - 180.0


def profile_targets(
    profile: str,
    *,
    family: str = "multirotor",
    condition: str = "medium",
    initial_yaw_deg: float = 0.0,
) -> tuple[Any, ...]:
    """Scale one base profile into an excitation condition.

    The multirotor profiles are also rotated by ``initial_yaw_deg``, so one
    table of translations is flown along several headings; the fixed-wing
    profiles are scaled around the trim operating point and ignore it.
    """

    if family not in PROFILES:
        raise ValueError(f"family must be one of {FAMILIES}")
    excitation = CONDITIONS[family][condition]
    targets = PROFILES[family][profile]
    if family == "fixedwing":
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
                            + (target.throttle - TRIM_THROTTLE)
                            * excitation.throttle_scale,
                        ),
                    )
                ),
                duration_s=excitation.dwell_s,
            )
            for target in targets
        )

    yaw_rad = math.radians(initial_yaw_deg)
    cosine = math.cos(yaw_rad)
    sine = math.sin(yaw_rad)
    scaled = []
    for target in targets:
        north = excitation.translation_scale * target.north_m
        east = excitation.translation_scale * target.east_m
        scaled.append(
            PositionTarget(
                north_m=cosine * north - sine * east,
                east_m=sine * north + cosine * east,
                down_m=NOMINAL_DOWN_M
                + excitation.translation_scale * (target.down_m - NOMINAL_DOWN_M),
                yaw_deg=_wrapped_yaw_deg(
                    initial_yaw_deg + excitation.yaw_scale * target.yaw_deg
                ),
                duration_s=excitation.dwell_s,
            )
        )
    return tuple(scaled)


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


def _attitude_mask() -> int:
    return int(
        mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_BODY_ROLL_RATE_IGNORE
        | mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_BODY_PITCH_RATE_IGNORE
        | mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_BODY_YAW_RATE_IGNORE
    )


def _quaternion_from_euler(
    roll_rad: float, pitch_rad: float, yaw_rad: float
) -> tuple[float, float, float, float]:
    """Return the shared WXYZ attitude quaternion as a MAVLink-ready tuple."""

    w, x, y, z = quaternion_from_euler(roll_rad, pitch_rad, yaw_rad)
    return (float(w), float(x), float(y), float(z))


def _send_target(connection: Any, target: Any, yaw_rad: float = 0.0) -> None:
    if isinstance(target, AttitudeTarget):
        connection.mav.set_attitude_target_send(
            int(time.monotonic() * 1000.0) & 0xFFFFFFFF,
            connection.target_system,
            connection.target_component,
            _attitude_mask(),
            _quaternion_from_euler(
                math.radians(target.roll_deg), math.radians(target.pitch_deg), yaw_rad
            ),
            0.0,
            0.0,
            0.0,
            target.throttle,
        )
        return
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


def _stream_target(
    connection: Any, target: Any, rate_hz: float, yaw_rad: float = 0.0
) -> None:
    period_s = 1.0 / rate_hz
    stop_time = time.monotonic() + target.duration_s
    while time.monotonic() < stop_time:
        started = time.monotonic()
        _send_target(connection, target, yaw_rad)
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


def _land_and_disarm(connection: Any, timeout_s: float) -> None:
    """Land after a profile without suppressing a preceding flight error."""

    print("landing")
    _command_land(connection)
    landing_deadline = time.monotonic() + timeout_s
    while time.monotonic() < landing_deadline:
        message = connection.recv_match(type="HEARTBEAT", blocking=True, timeout=1.0)
        if message is not None and not (
            message.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
        ):
            print("landed and disarmed")
            return
    _command_arm(connection, False)
    raise RuntimeError("landing timed out; sent disarm command")


def _enter_offboard(connection: Any) -> None:
    modes = connection.mode_mapping()
    if not modes or "OFFBOARD" not in modes:
        raise RuntimeError("PX4 did not advertise OFFBOARD mode")
    offboard_mode = modes["OFFBOARD"]
    if isinstance(offboard_mode, tuple):
        connection.set_mode_px4(*offboard_mode)
    else:
        connection.set_mode(offboard_mode)


def _connect(connection_string: str, heartbeat_timeout_s: float) -> Any:
    connection = mavutil.mavlink_connection(connection_string)
    if connection.wait_heartbeat(timeout=heartbeat_timeout_s) is None:
        raise RuntimeError(f"no PX4 heartbeat on {connection_string}")
    print(
        f"connected to system {connection.target_system}, "
        f"component {connection.target_component}"
    )
    return connection


def fly_profile(
    profile: str,
    *,
    family: str | None = None,
    condition: str = "medium",
    initial_yaw_deg: float = 0.0,
    connection_string: str = "udpin:0.0.0.0:14550",
    rate_hz: float = 20.0,
    heartbeat_timeout_s: float = 30.0,
    landing_timeout_s: float = 30.0,
) -> None:
    """Fly one bounded profile on one PX4 SITL instance."""

    resolved_family = family_of(profile) if family is None else family
    targets = profile_targets(
        profile,
        family=resolved_family,
        condition=condition,
        initial_yaw_deg=initial_yaw_deg,
    )
    connection = _connect(connection_string, heartbeat_timeout_s)
    print(
        f"family={resolved_family} profile={profile} condition={condition} "
        f"initial_yaw={initial_yaw_deg:g}deg"
    )
    if resolved_family == "fixedwing":
        _fly_fixedwing(connection, targets, rate_hz, heartbeat_timeout_s)
        return
    _fly_multirotor(connection, targets, rate_hz, landing_timeout_s)


def _fly_multirotor(
    connection: Any,
    targets: tuple[PositionTarget, ...],
    rate_hz: float,
    landing_timeout_s: float,
) -> None:
    # PX4 requires a setpoint stream before it will enter offboard mode.
    warmup = PositionTarget(
        targets[0].north_m,
        targets[0].east_m,
        targets[0].down_m,
        targets[0].yaw_deg,
        1.5,
    )
    _stream_target(connection, warmup, rate_hz)
    _enter_offboard(connection)
    if not connection.motors_armed():
        _command_arm(connection, True)

    armed = connection.motors_armed()
    arm_deadline = time.monotonic() + 8.0
    while not armed and time.monotonic() < arm_deadline:
        _send_target(connection, targets[0])
        message = connection.recv_match(type="HEARTBEAT", blocking=False)
        if message is not None and message.get_srcSystem() == connection.target_system:
            armed = bool(message.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
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
        _land_and_disarm(connection, landing_timeout_s)


def _fly_fixedwing(
    connection: Any,
    targets: tuple[AttitudeTarget, ...],
    rate_hz: float,
    heartbeat_timeout_s: float,
) -> None:
    if not connection.motors_armed():
        raise RuntimeError("fixed-wing profile requires an already-armed airplane")
    attitude = connection.recv_match(
        type="ATTITUDE", blocking=True, timeout=heartbeat_timeout_s
    )
    if attitude is None:
        raise RuntimeError("PX4 did not publish ATTITUDE before profile start")
    yaw_rad = float(attitude.yaw)

    warmup = AttitudeTarget(0.0, TRIM_PITCH_DEG, TRIM_THROTTLE, 1.5)
    _stream_target(connection, warmup, rate_hz, yaw_rad)
    _enter_offboard(connection)

    for index, target in enumerate(targets, start=1):
        print(
            f"target {index}/{len(targets)}: roll={target.roll_deg:g}deg "
            f"pitch={target.pitch_deg:g}deg throttle={target.throttle:g}"
        )
        _stream_target(connection, target, rate_hz, yaw_rad)

    print("profile complete; holding trim")
    _stream_target(connection, warmup, rate_hz, yaw_rad)
