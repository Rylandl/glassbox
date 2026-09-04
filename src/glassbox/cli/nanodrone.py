"""Score a model on the IDSIA Nano-Quadrotor benchmark test split."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from glassbox.io.corpus import REFERENCE_CORPORA
from glassbox.io.nanodrone_reference import (
    BENCHMARK_COMMIT,
    BENCHMARK_DOI,
    BENCHMARK_REPOSITORY,
)
from glassbox.workflows.evaluate import PROTOCOLS, evaluate, save_report

BENCHMARK_MAX_HORIZON_STEPS = PROTOCOLS["nanodrone"].maximum_horizon_steps


def _evaluate(args: argparse.Namespace) -> None:
    _, trajectories = REFERENCE_CORPORA["nanodrone"].load_evaluation_trajectories(
        args.trajectory
    )
    report = evaluate(
        args.model,
        trajectories,
        protocol="nanodrone",
        maximum_horizon_steps=args.max_horizon,
    )
    report["benchmark"] = {
        "repository": BENCHMARK_REPOSITORY,
        "commit": BENCHMARK_COMMIT,
        "doi": BENCHMARK_DOI,
        "test_profile": "melon",
    }
    report["test_artifacts"] = [str(path) for path in args.trajectory]
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
        save_report(report, args.report)
        print(f"wrote benchmark report {args.report}")
    elif args.json:
        print(json.dumps(report, indent=2))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

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

    args = parser.parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
