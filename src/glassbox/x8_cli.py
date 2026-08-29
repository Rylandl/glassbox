"""Fetch and convert the pinned NTNU Skywalker X8 SysID campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from glassbox.data import save_trajectory_npz
from glassbox.x8_reference import (
    X8_RECORDINGS,
    X8_REFERENCE_DOI,
    X8ReferenceAdapter,
    extract_x8_reference,
    fetch_x8_reference,
)
from glassbox.x8_evaluation import (
    evaluate_x8_reference_models,
    save_x8_reference_report,
)


def _adapter(args: argparse.Namespace) -> X8ReferenceAdapter:
    return X8ReferenceAdapter(verify_checksum=not args.skip_checksum)


def _inspect(args: argparse.Namespace) -> None:
    inventory = _adapter(args).inspect(args.csv)
    if args.json:
        print(json.dumps(inventory, indent=2))
        return
    labels = inventory["labels"]
    quality = inventory["quality"]
    print(f"recording: {args.csv}")
    print(
        f"profile={labels['profile']}  split={labels['benchmark_split']}  "
        f"replicate={labels['replicate']}"
    )
    print(
        f"{inventory['intervals']} intervals, {inventory['duration_s']:.3f}s "
        f"at {quality['sample_rate_hz']:.3f} Hz"
    )
    print(f"ground-speed range [m/s]: {quality['ground_speed_range_m_s']}")
    print(
        "GPS/integrated-velocity endpoint discrepancy [m]: "
        f"{quality['gps_vs_integrated_velocity_endpoint_discrepancy_m']:.3f}"
    )
    print(
        "pinned checksum: "
        f"{'verified' if inventory['checksum_matches_pinned_snapshot'] else 'different'}"
    )


def _extract(args: argparse.Namespace) -> None:
    trajectory = _adapter(args).load(args.csv)
    save_trajectory_npz(trajectory, args.output)
    print(
        f"wrote {args.output}: {len(trajectory.controls)} intervals, "
        f"{trajectory.time_s[-1]:.3f}s at "
        f"{1.0 / trajectory.nominal_dt_s:.3f} Hz"
    )


def _fetch(args: argparse.Namespace) -> None:
    paths = fetch_x8_reference(args.destination, overwrite=args.overwrite)
    print(
        f"verified {len(paths)} maneuvers from {X8_REFERENCE_DOI} "
        f"under {args.destination}"
    )


def _extract_dataset(args: argparse.Namespace) -> None:
    outputs = extract_x8_reference(
        args.source,
        args.output,
        adapter=_adapter(args),
    )
    print(
        f"wrote {len(outputs)} canonical trajectories under {args.output} "
        f"({sum(item.split == 'training' for item in X8_RECORDINGS)} training, "
        f"{sum(item.split == 'validation' for item in X8_RECORDINGS)} validation)"
    )


def _prepare(args: argparse.Namespace) -> None:
    raw_root = args.destination / "raw"
    canonical_root = args.destination / "canonical"
    paths = fetch_x8_reference(raw_root, overwrite=args.overwrite)
    outputs = extract_x8_reference(raw_root, canonical_root)
    print(
        f"prepared pinned Skywalker X8 campaign: {len(paths)} verified CSVs in "
        f"{raw_root}, {len(outputs)} trajectories in {canonical_root}"
    )


def _evaluate(args: argparse.Namespace) -> None:
    validation_paths = tuple(
        sorted((args.destination / "canonical" / "validation").glob("*.npz"))
    )
    report = evaluate_x8_reference_models(
        {
            "structured": args.structured_model,
            "structured_residual": args.residual_model,
        },
        validation_paths,
    )
    save_x8_reference_report(report, args.report)
    for name, model in report["models"].items():
        metrics = model["aggregate"]["horizon_rollouts"]["2s"]
        print(
            f"{name}: 2s position={metrics['position_rmse_m']:.3f}m  "
            f"attitude={metrics['attitude_rmse_deg']:.2f}deg  "
            "score/persistence="
            f"{model['score_vs_kinematic_persistence']:.3f}"
        )
    print(f"wrote benchmark report {args.report}")


def _add_checksum_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--skip-checksum",
        action="store_true",
        help="accept a modified or future CSV with a pinned filename",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser(
        "inspect", help="validate and summarize one maneuver CSV"
    )
    inspect_parser.add_argument("csv", type=Path)
    inspect_parser.add_argument("--json", action="store_true")
    _add_checksum_option(inspect_parser)
    inspect_parser.set_defaults(handler=_inspect)

    extract_parser = subparsers.add_parser(
        "extract", help="convert one maneuver CSV to a trajectory NPZ"
    )
    extract_parser.add_argument("csv", type=Path)
    extract_parser.add_argument("output", type=Path)
    _add_checksum_option(extract_parser)
    extract_parser.set_defaults(handler=_extract)

    fetch_parser = subparsers.add_parser(
        "fetch", help="download and verify the pinned upstream CSVs"
    )
    fetch_parser.add_argument("destination", type=Path)
    fetch_parser.add_argument("--overwrite", action="store_true")
    fetch_parser.set_defaults(handler=_fetch)

    dataset_parser = subparsers.add_parser(
        "extract-dataset", help="convert all maneuvers and preserve the split"
    )
    dataset_parser.add_argument("source", type=Path)
    dataset_parser.add_argument("output", type=Path)
    _add_checksum_option(dataset_parser)
    dataset_parser.set_defaults(handler=_extract_dataset)

    prepare_parser = subparsers.add_parser(
        "prepare", help="fetch and convert the complete pinned campaign"
    )
    prepare_parser.add_argument("destination", type=Path)
    prepare_parser.add_argument("--overwrite", action="store_true")
    prepare_parser.set_defaults(handler=_prepare)

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="score structured models on the untouched validation split"
    )
    evaluate_parser.add_argument("destination", type=Path)
    evaluate_parser.add_argument("--structured-model", type=Path, required=True)
    evaluate_parser.add_argument("--residual-model", type=Path, required=True)
    evaluate_parser.add_argument("--report", type=Path, required=True)
    evaluate_parser.set_defaults(handler=_evaluate)

    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
