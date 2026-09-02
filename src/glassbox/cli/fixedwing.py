"""Generate canonical synthetic fixed-wing trajectories for a smoke benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

from glassbox.core.data import save_trajectory_npz
from glassbox.core.fixedwing_synthetic import generate_fixed_wing_trajectory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--flights", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--dt", type=float, default=0.02)
    args = parser.parse_args()
    if args.flights < 1:
        parser.error("--flights must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for offset in range(args.flights):
        seed = args.seed + offset
        trajectory = generate_fixed_wing_trajectory(
            seed=seed, duration_s=args.duration, dt_s=args.dt
        )
        path = args.output_dir / f"fixedwing_synthetic_{seed}.npz"
        save_trajectory_npz(trajectory, path)
        print(
            f"wrote {path}: {trajectory.time_s[-1]:.3f}s, "
            f"profile={trajectory.labels['profile']}"
        )


if __name__ == "__main__":
    main()
