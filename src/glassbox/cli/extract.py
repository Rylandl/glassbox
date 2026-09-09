"""Turn PX4 ULogs into canonical Glassbox trajectory NPZ files.

``glassbox extract LOG OUT.npz`` converts one log. ``glassbox extract LOG...
OUTDIR`` converts several: each log becomes ``OUTDIR/<log stem>_<state
source>.npz``, so estimated and ground-truth extractions of the same corpus sit
side by side in one directory and a later ``fit`` can glob either.

``--family`` selects the actuator contract. A multirotor log is joined against
four normalized motor channels, whose order is derived from the log's own
CA_ROTOR geometry unless ``--motor-indices`` overrides it. A fixed-wing log is
joined against one throttle channel and three control surfaces, reconstructed
from the CA_SV_CS allocation parameters unless ``--motor-index`` and
``--surface-indices`` override them.

``--inspect`` lists each log's topics, fields, sample counts and dropouts and
writes nothing; every path on the command line is then a log.

Extraction keeps the longest contiguous armed, airborne, telemetry-complete
interval and reports what fraction of the armed span that was. A telemetry gap
wider than the resolved actuator hold age splits a flight into separate
segments rather than being bridged into a rollout, so a low coverage number is
a fact about the log, not a fitting knob.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from glassbox.core.data import save_trajectory_npz
from glassbox.io.px4_ulog import PX4IngestConfig, inspect_ulog, load_px4_trajectory

FAMILIES = ("multirotor", "fixedwing")


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


def _inspect(log: Path, *, as_json: bool) -> None:
    inventory = inspect_ulog(log)
    if as_json:
        print(json.dumps(inventory, indent=2))
        return
    duration_s = inventory["last_timestamp_s"] - inventory["start_timestamp_s"]
    print(f"ULog: {log}")
    print(
        f"duration: {duration_s:.3f}s  topics: {len(inventory['topics'])}  "
        f"dropouts: {inventory['dropout_count']}"
    )
    for topic in inventory["topics"]:
        print(
            f"  {topic['name']}[{topic['multi_id']}]  "
            f"samples={topic['samples']}  fields={len(topic['fields'])}"
        )


def _config(args: argparse.Namespace) -> PX4IngestConfig:
    """Build the one ingest configuration this family's flags describe."""

    shared = {
        "sample_rate_hz": args.rate,
        "state_source": args.state_source,
        "actuator_topic": args.actuator_topic,
        "actuator_field": args.actuator_field,
        "max_gap_s": args.max_gap,
        "actuator_hold_max_age_s": args.actuator_hold_max_age,
        "min_duration_s": args.min_duration,
        "min_height_m": None if args.include_ground else args.min_height,
        "only_armed": not args.include_disarmed,
        "only_in_air": not args.include_ground,
        "profile": args.profile,
        "condition": args.condition,
        "replicate": args.replicate,
        "initial_yaw_deg": args.initial_yaw,
        "vehicle_id": args.vehicle_id,
    }
    if args.family == "multirotor":
        return PX4IngestConfig(motor_indices=args.motor_indices, **shared)
    return PX4IngestConfig(
        platform="fixedwing",
        motor_index=args.motor_index,
        surface_indices=args.surface_indices,
        servo_topic=args.servo_topic,
        servo_field=args.servo_field,
        **shared,
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


def _extract(log: Path, output: Path, args: argparse.Namespace) -> None:
    config = _config(args)
    trajectory = load_px4_trajectory(log, config=config)
    save_trajectory_npz(trajectory, output)
    mapping = trajectory.provenance["px4"]["actuator_mapping"]
    print(
        f"wrote {output}: {len(trajectory.controls)} intervals, "
        f"{trajectory.time_s[-1]:.3f}s at {config.sample_rate_hz:g} Hz"
    )
    if args.family == "multirotor":
        print(
            f"motor order: {mapping['motor_indices']} ({mapping['motor_order_source']})"
        )
    else:
        print(
            "actuator mapping: "
            f"motor={mapping['motor_index']} "
            f"surfaces={mapping['surface_indices']} "
            f"({mapping['actuator_mapping_source']})"
        )
    _report_segments(trajectory)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "path",
        nargs="+",
        type=Path,
        metavar="PATH",
        help="one or more ULogs then the output NPZ or directory; with "
        "--inspect every path is a ULog",
    )
    parser.add_argument("--family", choices=FAMILIES, default="multirotor")
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="list topics, fields and dropouts instead of writing a trajectory",
    )
    parser.add_argument("--json", action="store_true", help="print inspection as JSON")
    parser.add_argument("--rate", type=float, default=50.0)
    parser.add_argument(
        "--state-source", choices=("estimated", "ground_truth"), default="estimated"
    )
    parser.add_argument(
        "--motor-indices",
        type=_motor_indices,
        help="multirotor FL,FR,RR,RL channel indices; default derives them "
        "from CA_ROTOR geometry",
    )
    parser.add_argument(
        "--motor-index",
        type=int,
        help="fixed-wing throttle slot; default requires a single CA_ROTOR entry",
    )
    parser.add_argument(
        "--surface-indices",
        type=_surface_indices,
        help="fixed-wing aileron,elevator,rudder slots; default reconstructs "
        "the axes from CA_SV_CS allocation parameters",
    )
    parser.add_argument(
        "--actuator-topic",
        default="actuator_motors",
        help="normalized motor input topic",
    )
    parser.add_argument(
        "--actuator-field",
        default="control",
        help="motor array field prefix, for example control or output",
    )
    parser.add_argument("--servo-topic", default="actuator_servos")
    parser.add_argument("--servo-field", default="control")
    parser.add_argument("--max-gap", type=float, default=0.10)
    parser.add_argument(
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
    parser.add_argument("--min-duration", type=float, default=0.50)
    parser.add_argument(
        "--min-height",
        type=float,
        default=0.20,
        help="minimum local height above the takeoff origin in metres",
    )
    parser.add_argument(
        "--include-disarmed",
        action="store_true",
        help="do not gate samples using actuator_armed",
    )
    parser.add_argument(
        "--include-ground",
        action="store_true",
        help="do not gate samples using vehicle_land_detected or local height",
    )
    parser.add_argument("--profile", help="maneuver-family trajectory label")
    parser.add_argument("--condition", help="excitation-condition trajectory label")
    parser.add_argument(
        "--replicate", type=int, help="positive replicate trajectory label"
    )
    parser.add_argument(
        "--initial-yaw",
        type=float,
        help="initial profile yaw trajectory label in degrees",
    )
    parser.add_argument(
        "--vehicle-id",
        help="stable physical vehicle identity used to validate dataset pooling",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.family == "multirotor" and (
        args.motor_index is not None or args.surface_indices is not None
    ):
        parser.error(
            "--motor-index and --surface-indices describe a fixed-wing allocation"
        )
    if args.family == "fixedwing" and args.motor_indices is not None:
        parser.error("--motor-indices describes a multirotor allocation")

    if args.inspect:
        for log in args.path:
            _inspect(log, as_json=args.json)
        return

    if len(args.path) < 2:
        parser.error("a ULog and an output path are required")
    *logs, output = args.path
    if len(logs) == 1:
        _extract(logs[0], output, args)
        return
    for log in logs:
        _extract(log, output / f"{log.stem}_{args.state_source}.npz", args)


if __name__ == "__main__":
    main()
