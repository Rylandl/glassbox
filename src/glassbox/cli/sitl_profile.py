"""Fly one bounded scripted maneuver profile on a PX4 SITL instance.

``--family multirotor`` streams local NED position and yaw setpoints: the
command takes off into OFFBOARD, flies the profile, then lands and disarms
even when the profile failed. ``--family fixedwing`` streams attitude and
throttle setpoints and requires an already-airborne, already-armed airplane;
it hands the plane back holding trim.

``--condition`` scales one profile into a low, medium or high excitation
condition, which is also the trajectory label the extracted flight carries.
``--initial-yaw`` rotates the multirotor translations so one table of
maneuvers is flown along several headings.

This command only transmits setpoints inside the declared profile table. It
records nothing itself: the flight is logged by PX4, and
``glassbox extract`` converts that log.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from glassbox.io.sitl_profile import CONDITIONS, FAMILIES, PROFILES, fly_profile


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "profile",
        help="profile name; "
        + "; ".join(f"{family}: {', '.join(PROFILES[family])}" for family in FAMILIES),
    )
    parser.add_argument("--family", choices=FAMILIES, default="multirotor")
    parser.add_argument(
        "--condition", choices=tuple(CONDITIONS["multirotor"]), default="medium"
    )
    parser.add_argument(
        "--initial-yaw",
        type=float,
        default=0.0,
        help="heading the multirotor translations are rotated by, in degrees",
    )
    parser.add_argument(
        "--connection", default="udpin:0.0.0.0:14550", help="pymavlink endpoint"
    )
    parser.add_argument("--rate", type=float, default=20.0)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.profile not in PROFILES[args.family]:
        parser.error(
            f"{args.family} has no profile {args.profile!r}; "
            f"choose one of {', '.join(PROFILES[args.family])}"
        )
    fly_profile(
        args.profile,
        family=args.family,
        condition=args.condition,
        initial_yaw_deg=args.initial_yaw,
        connection_string=args.connection,
        rate_hz=args.rate,
    )


if __name__ == "__main__":
    main()
