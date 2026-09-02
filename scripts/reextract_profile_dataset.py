#!/usr/bin/env python3
"""Re-extract one recorded PX4 profile corpus into the current trajectory format."""

from __future__ import annotations

import argparse
from pathlib import Path

from glassbox.core.data import save_trajectory_npz
from glassbox.io.px4_ulog import PX4IngestConfig, load_px4_trajectory


def _recorded_runs(root: Path) -> list[tuple[str, str, int, Path]]:
    runs: list[tuple[str, str, int, Path]] = []
    for run_dir in sorted(root.glob("*/*/run_*")):
        if not run_dir.is_dir():
            continue
        try:
            replicate = int(run_dir.name.removeprefix("run_"))
        except ValueError as error:
            raise ValueError(f"invalid run directory: {run_dir}") from error
        logs = sorted(run_dir.glob("px4/**/*.ulg"))
        if not logs:
            raise ValueError(f"no ULog found under {run_dir}")
        runs.append(
            (
                run_dir.parent.parent.name,
                run_dir.parent.name,
                replicate,
                logs[-1],
            )
        )
    if not runs:
        raise ValueError(f"no profile runs found under {root}")
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--platform", choices=("multirotor", "fixedwing"), required=True
    )
    parser.add_argument("--rate", type=float, default=50.0)
    parser.add_argument("--vehicle-id", required=True)
    parser.add_argument(
        "--multirotor-initial-yaws",
        default="0,45",
        help="comma-separated yaw labels cycled by replicate",
    )
    args = parser.parse_args()

    initial_yaws = tuple(
        float(value.strip()) for value in args.multirotor_initial_yaws.split(",")
    )
    if not initial_yaws:
        parser.error("--multirotor-initial-yaws cannot be empty")

    for profile, condition, replicate, log_path in _recorded_runs(args.root):
        run_dir = log_path.parents[4]
        stem = run_dir / f"{profile}_{condition}_{replicate}"
        for state_source in ("estimated", "ground_truth"):
            config = PX4IngestConfig(
                platform=args.platform,
                sample_rate_hz=args.rate,
                state_source=state_source,
                min_height_m=0.5 if args.platform == "fixedwing" else 0.2,
                profile=profile,
                condition=condition,
                replicate=replicate,
                initial_yaw_deg=(
                    initial_yaws[(replicate - 1) % len(initial_yaws)]
                    if args.platform == "multirotor"
                    else None
                ),
                vehicle_id=args.vehicle_id,
            )
            trajectory = load_px4_trajectory(log_path, config=config)
            output = stem.with_name(f"{stem.name}_{state_source}.npz")
            save_trajectory_npz(trajectory, output)
            print(
                f"wrote {output}: {trajectory.time_s[-1]:.3f}s, "
                f"controls={trajectory.control_names}"
            )


if __name__ == "__main__":
    main()
