"""Fetch and convert the IDSIA Nano-Quadrotor SysID benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from glassbox.data import save_trajectory_npz
from glassbox.nanodrone_benchmark import (
    BENCHMARK_COMMIT,
    BENCHMARK_RECORDINGS,
    NanoDroneBenchmarkAdapter,
    extract_nanodrone_benchmark,
    fetch_nanodrone_benchmark,
)
from glassbox.nanodrone_evaluation import (
    BENCHMARK_MAX_HORIZON_STEPS,
    evaluate_nanodrone_model_artifact,
    save_nanodrone_benchmark_report,
)


def _adapter(args: argparse.Namespace) -> NanoDroneBenchmarkAdapter:
    return NanoDroneBenchmarkAdapter(verify_checksum=not args.skip_checksum)


def _inspect(args: argparse.Namespace) -> None:
    inventory = _adapter(args).inspect(args.csv)
    if args.json:
        print(json.dumps(inventory, indent=2))
        return
    quality = inventory["quality"]
    labels = inventory["labels"]
    print(f"recording: {args.csv}")
    print(
        f"profile={labels['profile']}  split={labels['benchmark_split']}  "
        f"replicate={labels['replicate']}"
    )
    print(
        f"{inventory['intervals']} intervals, {inventory['duration_s']:.3f}s "
        f"at {quality['sample_rate_hz']:.3f} Hz"
    )
    print(
        "motor speed range [rad/s]: "
        f"{quality['motor_speed_minimum_rad_s']} -> "
        f"{quality['motor_speed_maximum_rad_s']}"
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
    paths = fetch_nanodrone_benchmark(
        args.destination,
        overwrite=args.overwrite,
    )
    print(
        f"verified {len(paths)} recordings at benchmark commit {BENCHMARK_COMMIT} "
        f"under {args.destination}"
    )


def _extract_dataset(args: argparse.Namespace) -> None:
    outputs = extract_nanodrone_benchmark(
        args.source,
        args.output,
        adapter=_adapter(args),
    )
    print(
        f"wrote {len(outputs)} canonical trajectories under {args.output} "
        f"({sum(recording.split == 'train' for recording in BENCHMARK_RECORDINGS)} "
        "train, "
        f"{sum(recording.split == 'test' for recording in BENCHMARK_RECORDINGS)} test)"
    )


def _prepare(args: argparse.Namespace) -> None:
    raw_root = args.destination / "raw"
    canonical_root = args.destination / "canonical"
    paths = fetch_nanodrone_benchmark(raw_root, overwrite=args.overwrite)
    outputs = extract_nanodrone_benchmark(raw_root, canonical_root)
    print(
        f"prepared pinned benchmark commit {BENCHMARK_COMMIT}: "
        f"{len(paths)} verified CSVs in {raw_root}, "
        f"{len(outputs)} trajectories in {canonical_root}"
    )


def _evaluate(args: argparse.Namespace) -> None:
    report = evaluate_nanodrone_model_artifact(
        args.model,
        args.trajectory,
        max_horizon_steps=args.max_horizon,
    )
    selected = report["model"]["selected_horizons"]
    for step in selected:
        metrics = selected[step]
        print(
            f"h={step} ({metrics['time_s']:.3f}s): "
            f"position={metrics['position_mae_m']:.5f}m  "
            f"velocity={metrics['velocity_mae_m_s']:.5f}m/s  "
            f"attitude={metrics['attitude_mae_rad']:.5f}rad  "
            "angular_velocity="
            f"{metrics['angular_velocity_mae_rad_s']:.5f}rad/s"
        )
    if args.report is not None:
        save_nanodrone_benchmark_report(report, args.report)
        print(f"wrote benchmark report {args.report}")
    elif args.json:
        print(json.dumps(report, indent=2))


def _add_checksum_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--skip-checksum",
        action="store_true",
        help="accept a modified or future CSV with the pinned filename",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser(
        "inspect", help="validate and summarize one benchmark CSV"
    )
    inspect_parser.add_argument("csv", type=Path)
    inspect_parser.add_argument("--json", action="store_true")
    _add_checksum_option(inspect_parser)
    inspect_parser.set_defaults(handler=_inspect)

    extract_parser = subparsers.add_parser(
        "extract", help="convert one benchmark CSV to a trajectory NPZ"
    )
    extract_parser.add_argument("csv", type=Path)
    extract_parser.add_argument("output", type=Path)
    _add_checksum_option(extract_parser)
    extract_parser.set_defaults(handler=_extract)

    fetch_parser = subparsers.add_parser(
        "fetch", help="download and verify the pinned upstream dataset"
    )
    fetch_parser.add_argument("destination", type=Path)
    fetch_parser.add_argument("--overwrite", action="store_true")
    fetch_parser.set_defaults(handler=_fetch)

    dataset_parser = subparsers.add_parser(
        "extract-dataset", help="convert all 15 recordings and preserve the split"
    )
    dataset_parser.add_argument("source", type=Path)
    dataset_parser.add_argument("output", type=Path)
    _add_checksum_option(dataset_parser)
    dataset_parser.set_defaults(handler=_extract_dataset)

    prepare_parser = subparsers.add_parser(
        "prepare", help="fetch and convert the complete pinned benchmark"
    )
    prepare_parser.add_argument("destination", type=Path)
    prepare_parser.add_argument("--overwrite", action="store_true")
    prepare_parser.set_defaults(handler=_prepare)

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="score a saved model with the published rolling-horizon protocol",
    )
    evaluate_parser.add_argument("model", type=Path)
    evaluate_parser.add_argument("trajectory", type=Path, nargs="+")
    evaluate_parser.add_argument(
        "--max-horizon", type=int, default=BENCHMARK_MAX_HORIZON_STEPS
    )
    evaluate_parser.add_argument("--report", type=Path)
    evaluate_parser.add_argument("--json", action="store_true")
    evaluate_parser.set_defaults(handler=_evaluate)

    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
