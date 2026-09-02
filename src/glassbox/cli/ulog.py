"""Inspect PX4 ULogs and extract canonical Glassbox trajectories."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from glassbox.core.data import load_trajectory_npz, save_trajectory_npz
from glassbox.io.arp_reference import (
    ARP_REFERENCE_COMMIT,
    extract_arp_reference,
    fetch_arp_reference,
)
from glassbox.io.idf_reference import (
    IDF_ARCHIVE_SIZE_BYTES,
    extract_idf_reference,
    extract_idf_ulogs,
    fetch_idf_archive,
    idf_corpus_report,
    save_idf_corpus_report,
)
from glassbox.io.px4_ulog import PX4IngestConfig, inspect_ulog, load_px4_trajectory


def _motor_indices(value: str) -> tuple[int, int, int, int]:
    try:
        result = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("motor indices must be integers") from error
    if len(result) != 4:
        raise argparse.ArgumentTypeError("exactly four motor indices are required")
    return result  # type: ignore[return-value]


def _surface_indices(value: str) -> tuple[int, int, int]:
    try:
        result = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("surface indices must be integers") from error
    if len(result) != 3:
        raise argparse.ArgumentTypeError("exactly three surface indices are required")
    return result  # type: ignore[return-value]


def _inspect(args: argparse.Namespace) -> None:
    inventory = inspect_ulog(args.log)
    if args.json:
        print(json.dumps(inventory, indent=2))
        return

    print(f"ULog: {args.log}")
    print(
        f"duration: {inventory['last_timestamp_s'] - inventory['start_timestamp_s']:.3f}s  "
        f"topics: {len(inventory['topics'])}  dropouts: {inventory['dropout_count']}"
    )
    for topic in inventory["topics"]:
        print(
            f"  {topic['name']}[{topic['multi_id']}]  "
            f"samples={topic['samples']}  fields={len(topic['fields'])}"
        )


def _report_segments(trajectory) -> None:
    """Print segment coverage and warn when telemetry gaps discarded flight time."""

    px4 = trajectory.provenance["px4"]
    segment_count = px4.get("valid_segment_count", 1)
    coverage = px4.get("selected_segment_coverage")
    coverage_text = f"{coverage:.1%}" if coverage is not None else "unknown"
    print(f"segments: {segment_count} valid, coverage: {coverage_text}")
    if segment_count > 1 or (coverage is not None and coverage < 0.5):
        resolved_hold_age = px4.get("resolved_actuator_hold_max_age_s")
        hold_age_text = (
            f"{resolved_hold_age:.3f}s" if resolved_hold_age is not None else "unknown"
        )
        print(
            "warning: extraction kept only "
            f"{segment_count} valid segment(s) covering {coverage_text} of the "
            "armed/in-air span this log offered; only the longest segment was "
            "written. Telemetry gaps wider than the resolved actuator hold "
            f"age ({hold_age_text}) split the flight into separate segments. "
            "Widen --max-gap or set --actuator-hold-max-age explicitly to "
            "recover more flight time.",
            file=sys.stderr,
        )


def _extract(args: argparse.Namespace) -> None:
    config = PX4IngestConfig(
        sample_rate_hz=args.rate,
        state_source=args.state_source,
        motor_indices=args.motor_indices,
        actuator_topic=args.actuator_topic,
        actuator_field=args.actuator_field,
        max_gap_s=args.max_gap,
        actuator_hold_max_age_s=args.actuator_hold_max_age,
        min_duration_s=args.min_duration,
        min_height_m=None if args.include_ground else args.min_height,
        only_armed=not args.include_disarmed,
        only_in_air=not args.include_ground,
        profile=args.profile,
        condition=args.condition,
        replicate=args.replicate,
        initial_yaw_deg=args.initial_yaw,
        vehicle_id=args.vehicle_id,
    )
    trajectory = load_px4_trajectory(args.log, config=config)
    save_trajectory_npz(trajectory, args.output)
    mapping = trajectory.provenance["px4"]["actuator_mapping"]
    print(
        f"wrote {args.output}: {len(trajectory.controls)} intervals, "
        f"{trajectory.time_s[-1]:.3f}s at {config.sample_rate_hz:g} Hz"
    )
    print(f"motor order: {mapping['motor_indices']} ({mapping['motor_order_source']})")
    _report_segments(trajectory)


def _extract_fixedwing(args: argparse.Namespace) -> None:
    config = PX4IngestConfig(
        platform="fixedwing",
        sample_rate_hz=args.rate,
        state_source=args.state_source,
        motor_index=args.motor_index,
        surface_indices=args.surface_indices,
        actuator_topic=args.motor_topic,
        actuator_field=args.motor_field,
        servo_topic=args.servo_topic,
        servo_field=args.servo_field,
        max_gap_s=args.max_gap,
        actuator_hold_max_age_s=args.actuator_hold_max_age,
        min_duration_s=args.min_duration,
        min_height_m=None if args.include_ground else args.min_height,
        only_armed=not args.include_disarmed,
        only_in_air=not args.include_ground,
        profile=args.profile,
        condition=args.condition,
        replicate=args.replicate,
        initial_yaw_deg=args.initial_yaw,
        vehicle_id=args.vehicle_id,
    )
    trajectory = load_px4_trajectory(args.log, config=config)
    save_trajectory_npz(trajectory, args.output)
    mapping = trajectory.provenance["px4"]["actuator_mapping"]
    print(
        f"wrote {args.output}: {len(trajectory.controls)} intervals, "
        f"{trajectory.time_s[-1]:.3f}s at {config.sample_rate_hz:g} Hz"
    )
    print(
        "actuator mapping: "
        f"motor={mapping['motor_index']} "
        f"surfaces={mapping['surface_indices']} "
        f"({mapping['actuator_mapping_source']})"
    )
    _report_segments(trajectory)


def _prepare_arp(args: argparse.Namespace) -> None:
    raw_root = args.destination / "raw"
    canonical_root = args.destination / "canonical"
    paths = fetch_arp_reference(raw_root, overwrite=args.overwrite)
    outputs = extract_arp_reference(raw_root, canonical_root)
    total_duration_s = sum(
        float(load_trajectory_npz(output).time_s[-1]) for output in outputs
    )
    print(
        f"prepared pinned ARP reference commit {ARP_REFERENCE_COMMIT}: "
        f"{len(paths)} verified ULogs in {raw_root}, "
        f"{len(outputs)} trajectories ({total_duration_s:.1f}s) in {canonical_root}"
    )


def _prepare_idf(args: argparse.Namespace) -> None:
    raw_root = args.destination / "raw"
    ulog_root = raw_root / "ulogs"
    canonical_root = args.destination / "canonical"
    print(
        "preparing the pinned IDF-DS Holybro archive "
        f"({IDF_ARCHIVE_SIZE_BYTES / 1_000_000_000:.2f} GB)"
    )
    archive = fetch_idf_archive(raw_root, overwrite=args.overwrite)
    ulogs = extract_idf_ulogs(archive, ulog_root, overwrite=args.overwrite)
    outputs = extract_idf_reference(ulog_root, canonical_root)
    report_path = args.destination / "corpus_report.json"
    report = idf_corpus_report(outputs, ulog_root)
    save_idf_corpus_report(report, report_path)
    total_duration_s = float(report["canonical"]["duration_s"])
    print(
        f"prepared IDF-DS fixed-wing reference: {len(ulogs)} verified ULogs, "
        f"{len(outputs)} dropout-safe trajectories "
        f"({total_duration_s / 3600.0:.2f}h) in {canonical_root}; "
        f"audit in {report_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="list topics and fields")
    inspect_parser.add_argument("log", type=Path)
    inspect_parser.add_argument("--json", action="store_true")
    inspect_parser.set_defaults(handler=_inspect)

    extract_parser = subparsers.add_parser(
        "extract", help="convert state and motor topics to a trajectory NPZ"
    )
    extract_parser.add_argument("log", type=Path)
    extract_parser.add_argument("output", type=Path)
    extract_parser.add_argument("--rate", type=float, default=50.0)
    extract_parser.add_argument(
        "--state-source", choices=("estimated", "ground_truth"), default="estimated"
    )
    extract_parser.add_argument(
        "--motor-indices",
        type=_motor_indices,
        help="FL,FR,RR,RL channel indices; default derives them from CA_ROTOR geometry",
    )
    extract_parser.add_argument(
        "--actuator-topic",
        default="actuator_motors",
        help="normalized actuator input topic",
    )
    extract_parser.add_argument(
        "--actuator-field",
        default="control",
        help="actuator array field prefix, for example control or output",
    )
    extract_parser.add_argument("--max-gap", type=float, default=0.10)
    extract_parser.add_argument(
        "--actuator-hold-max-age",
        type=float,
        default=None,
        help=(
            "maximum age in seconds for holding the last actuator sample "
            "valid; default resolves per log as max(--max-gap, 1.5x the "
            "median actuator sample period) so ordinary publish jitter does "
            "not fragment a flight into short segments"
        ),
    )
    extract_parser.add_argument("--min-duration", type=float, default=0.50)
    extract_parser.add_argument(
        "--min-height",
        type=float,
        default=0.20,
        help="minimum local height above the takeoff origin in metres",
    )
    extract_parser.add_argument(
        "--include-disarmed",
        action="store_true",
        help="do not gate samples using actuator_armed",
    )
    extract_parser.add_argument(
        "--include-ground",
        action="store_true",
        help="do not gate samples using vehicle_land_detected or local height",
    )
    extract_parser.add_argument(
        "--profile",
        help="maneuver-family trajectory label",
    )
    extract_parser.add_argument(
        "--condition",
        help="excitation-condition trajectory label",
    )
    extract_parser.add_argument(
        "--replicate", type=int, help="positive replicate trajectory label"
    )
    extract_parser.add_argument(
        "--initial-yaw",
        type=float,
        help="initial profile yaw trajectory label in degrees",
    )
    extract_parser.add_argument(
        "--vehicle-id",
        help="stable physical vehicle identity used to validate dataset pooling",
    )
    extract_parser.set_defaults(handler=_extract)

    fixedwing_parser = subparsers.add_parser(
        "extract-fixedwing",
        help="join motor and servo topics into a fixed-wing trajectory NPZ",
    )
    fixedwing_parser.add_argument("log", type=Path)
    fixedwing_parser.add_argument("output", type=Path)
    fixedwing_parser.add_argument("--rate", type=float, default=50.0)
    fixedwing_parser.add_argument(
        "--state-source", choices=("estimated", "ground_truth"), default="estimated"
    )
    fixedwing_parser.add_argument(
        "--motor-index",
        type=int,
        help="throttle slot; default requires a single CA_ROTOR entry",
    )
    fixedwing_parser.add_argument(
        "--surface-indices",
        type=_surface_indices,
        help=(
            "explicit aileron,elevator,rudder slots; default reconstructs axes "
            "from CA_SV_CS allocation parameters"
        ),
    )
    fixedwing_parser.add_argument("--motor-topic", default="actuator_motors")
    fixedwing_parser.add_argument("--motor-field", default="control")
    fixedwing_parser.add_argument("--servo-topic", default="actuator_servos")
    fixedwing_parser.add_argument("--servo-field", default="control")
    fixedwing_parser.add_argument("--max-gap", type=float, default=0.10)
    fixedwing_parser.add_argument(
        "--actuator-hold-max-age",
        type=float,
        default=None,
        help=(
            "maximum age in seconds for holding the last motor/servo sample "
            "valid; default resolves per log as max(--max-gap, 1.5x the "
            "median actuator sample period) so ordinary publish jitter does "
            "not fragment a flight into short segments"
        ),
    )
    fixedwing_parser.add_argument("--min-duration", type=float, default=0.50)
    fixedwing_parser.add_argument(
        "--min-height",
        type=float,
        default=0.20,
        help="minimum local height above the takeoff origin in metres",
    )
    fixedwing_parser.add_argument("--include-disarmed", action="store_true")
    fixedwing_parser.add_argument(
        "--include-ground",
        action="store_true",
        help="do not gate samples using vehicle_land_detected or local height",
    )
    fixedwing_parser.add_argument("--profile")
    fixedwing_parser.add_argument("--condition")
    fixedwing_parser.add_argument("--replicate", type=int)
    fixedwing_parser.add_argument("--initial-yaw", type=float)
    fixedwing_parser.add_argument(
        "--vehicle-id",
        help="stable physical vehicle identity used to validate dataset pooling",
    )
    fixedwing_parser.set_defaults(handler=_extract_fixedwing)

    arp_parser = subparsers.add_parser(
        "prepare-arp",
        help="fetch and convert the pinned ARP quadrotor system-ID ULogs",
    )
    arp_parser.add_argument("destination", type=Path)
    arp_parser.add_argument("--overwrite", action="store_true")
    arp_parser.set_defaults(handler=_prepare_arp)

    idf_parser = subparsers.add_parser(
        "prepare-idf",
        help="fetch and convert the pinned IDF-DS fixed-wing PX4 ULogs",
    )
    idf_parser.add_argument("destination", type=Path)
    idf_parser.add_argument("--overwrite", action="store_true")
    idf_parser.set_defaults(handler=_prepare_idf)

    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
