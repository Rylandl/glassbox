"""Generate canonical synthetic trajectories for either vehicle family.

The synthetic flights are a closed-world smoke corpus: the generator's own
parameters are the truth, so a fit on them measures the fitting path rather
than a vehicle. ``--family multirotor`` writes quadrotor flights and
``--family fixedwing`` writes fixed-wing flights, both in the same canonical
NPZ format every other command reads.

Each flight is named ``<family>_synthetic_<seed>.npz``, and the seed is the
only thing that distinguishes two flights of one family, so a corpus is
reproducible from ``--seed`` and ``--flights`` alone.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from glassbox.core.data import save_trajectory_npz
from glassbox.core.fixedwing_synthetic import generate_fixed_wing_trajectory
from glassbox.core.synthetic import generate_trajectory

FAMILIES = ("multirotor", "fixedwing")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--family", choices=FAMILIES, default="multirotor")
    parser.add_argument("--flights", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--dt", type=float, default=0.02)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.flights < 1:
        parser.error("--flights must be positive")

    generate = (
        generate_trajectory
        if args.family == "multirotor"
        else generate_fixed_wing_trajectory
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for offset in range(args.flights):
        seed = args.seed + offset
        trajectory = generate(seed=seed, duration_s=args.duration, dt_s=args.dt)
        path = args.output_dir / f"{args.family}_synthetic_{seed}.npz"
        save_trajectory_npz(trajectory, path)
        labels = " ".join(f"{key}={value}" for key, value in trajectory.labels.items())
        print(f"wrote {path}: {trajectory.time_s[-1]:.3f}s, {labels}")


if __name__ == "__main__":
    main()
