"""Fetch and convert the pinned EPFL TOPOPlane2 flight-data release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from glassbox.io.epfl_reference import (
    EPFL_REFERENCE_DOI,
    EPFLTopoplaneAdapter,
    extract_epfl_topoplane_reference,
    fetch_epfl_topoplane_reference,
)
from glassbox.workflows.epfl_evaluation import (
    evaluate_epfl_characterization,
    save_epfl_characterization,
)


def _adapter(args: argparse.Namespace) -> EPFLTopoplaneAdapter:
    return EPFLTopoplaneAdapter(verify_checksum=not args.skip_checksum)


def _inspect(args: argparse.Namespace) -> None:
    inventory = _adapter(args).inspect(args.bag)
    if args.json:
        print(json.dumps(inventory, indent=2))
        return
    quality = inventory["quality"]
    print(f"recording: {args.bag}")
    print(
        f"{len(inventory['segments'])} navigation-healthy segments, "
        f"{quality['navigation_healthy_duration_s']:.1f}s at "
        f"{quality['sample_rate_hz']:.1f} Hz"
    )
    print(f"ground-speed range [m/s]: {quality['ground_speed_range_m_s']}")
    print(f"pitot-airspeed range [m/s]: {quality['pitot_airspeed_range_m_s']}")
    print(
        "published angular-rate field: "
        f"max abs {quality['published_angular_velocity_max_abs']:.3g} "
        "(adapter reconstructs it from attitude)"
    )
    print(
        "pinned checksum: "
        f"{'verified' if inventory['checksum_matches_pinned_snapshot'] else 'different'}"
    )


def _extract(args: argparse.Namespace) -> None:
    outputs = extract_epfl_topoplane_reference(
        args.bag,
        args.output,
        adapter=_adapter(args),
    )
    print(f"wrote {len(outputs)} canonical TOPOPlane2 segments under {args.output}")


def _fetch(args: argparse.Namespace) -> None:
    path = fetch_epfl_topoplane_reference(args.destination, overwrite=args.overwrite)
    print(f"verified {path} from {EPFL_REFERENCE_DOI}")


def _prepare(args: argparse.Namespace) -> None:
    raw_directory = args.destination / "raw"
    canonical_directory = args.destination / "canonical"
    source = fetch_epfl_topoplane_reference(raw_directory, overwrite=args.overwrite)
    outputs = extract_epfl_topoplane_reference(source, canonical_directory)
    print(
        f"prepared EPFL TOPOPlane2: verified {source}, wrote {len(outputs)} "
        f"canonical segments under {canonical_directory}"
    )


def _evaluate(args: argparse.Namespace) -> None:
    report = evaluate_epfl_characterization(
        args.structured_report,
        args.residual_report,
    )
    save_epfl_characterization(report, args.output)
    for name, model in report["models"].items():
        metrics = model["aggregate_horizon_rollouts"]["2s"]
        print(
            f"{name}: score/persistence="
            f"{model['score_vs_kinematic_persistence']:.3f}  "
            f"2s position={metrics['position_rmse_m']:.3f}m  "
            f"attitude={metrics['attitude_rmse_deg']:.2f}deg"
        )
    print(
        f"selected={report['selected_model']} promotion={report['can_promote_model']}"
    )
    print(f"wrote {args.output}")


def _add_checksum_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--skip-checksum",
        action="store_true",
        help="accept a modified bag with the pinned filename",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser(
        "inspect", help="validate and summarize usable flight coverage"
    )
    inspect_parser.add_argument("bag", type=Path)
    inspect_parser.add_argument("--json", action="store_true")
    _add_checksum_option(inspect_parser)
    inspect_parser.set_defaults(handler=_inspect)

    extract_parser = subparsers.add_parser(
        "extract", help="convert all navigation-healthy flight segments"
    )
    extract_parser.add_argument("bag", type=Path)
    extract_parser.add_argument("output", type=Path)
    _add_checksum_option(extract_parser)
    extract_parser.set_defaults(handler=_extract)

    fetch_parser = subparsers.add_parser(
        "fetch", help="download and verify the pinned TOPOPlane2 bag"
    )
    fetch_parser.add_argument("destination", type=Path)
    fetch_parser.add_argument("--overwrite", action="store_true")
    fetch_parser.set_defaults(handler=_fetch)

    prepare_parser = subparsers.add_parser(
        "prepare", help="fetch and convert the pinned TOPOPlane2 recording"
    )
    prepare_parser.add_argument("destination", type=Path)
    prepare_parser.add_argument("--overwrite", action="store_true")
    prepare_parser.set_defaults(handler=_prepare)

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="compare maintained models on the same-flight holdout"
    )
    evaluate_parser.add_argument("--structured-report", type=Path, required=True)
    evaluate_parser.add_argument("--residual-report", type=Path, required=True)
    evaluate_parser.add_argument("--output", type=Path, required=True)
    evaluate_parser.set_defaults(handler=_evaluate)

    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
