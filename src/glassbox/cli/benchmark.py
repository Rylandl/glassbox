"""Run one of the maintained closed-loop and corpus benchmarks.

``nmpc`` is the acceptance and timing suite: a fixed set of scenarios flown
against a fitted model, with the solve statuses, the bounded holds, and the
per-interval timings its recorded artifact reports.

``recovery`` is the prewarmed synthetic diagnostic: a configuration change the
belief has not seen, then adaptation through an NMPC recovery, scored against
the stale belief on independent telemetry.

``cascade-x8`` scores the published Cascade Skywalker X8 model, unfitted, on
the campaign's untouched validation maneuvers, under the same protocol and
against the same baseline the fitted models are scored against. Rows other
than the primary one are an environment and parameter sensitivity, never a
fit. ``--diagnose`` instead regresses the one-step force and moment residuals
on the flight variables, so a losing variant says which coefficient it is
losing on.

Every benchmark writes JSON. ``nmpc`` exits non-zero when its acceptance
summary fails.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

X8_AIRCRAFT = ("skywalker_x8", "skywalker_x8_panels")


def _floats(value: str) -> tuple[float, ...]:
    try:
        return tuple(float(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected a comma-separated list of numbers"
        ) from error


def _nmpc(args: argparse.Namespace) -> None:
    from glassbox.workflows.benchmarks.nmpc import run_nmpc_benchmark

    report = run_nmpc_benchmark()
    encoded = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
    print(encoded)
    if not report["summary"]["passed"]:
        raise SystemExit(1)


def _recovery(args: argparse.Namespace) -> None:
    from glassbox.workflows.benchmarks.recovery import run_adaptive_recovery_benchmark

    report = run_adaptive_recovery_benchmark()
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
        print(f"wrote {args.output}")
    else:
        print(payload, end="")
    prediction = report["evidence"]["independent_prediction"]
    comparisons = report["comparisons"]
    print(
        "independent 0.6 s prediction RMS: "
        f"{prediction['normalized_rms_before']:.6f} -> "
        f"{prediction['normalized_rms_after']:.6f}"
    )
    print(
        "adapted/stale recovery-tail ratios: "
        f"tracking={comparisons['adapted_vs_stale_tail_tracking_rms_ratio']:.3f}, "
        "attitude+rate="
        f"{comparisons['adapted_vs_stale_tail_attitude_rate_rms_ratio']:.3f}"
    )


def _cascade_x8(args: argparse.Namespace) -> None:
    if args.diagnose:
        _diagnose_cascade_x8(args)
        return
    from glassbox.workflows.benchmarks.cascade_x8 import (
        evaluate_x8_cascade,
        save_x8_cascade_report,
    )

    reference = args.reference_report
    if reference is None:
        candidate = args.destination / "benchmark_report.json"
        reference = candidate if candidate.exists() else None
    report = evaluate_x8_cascade(
        args.destination,
        aircraft=args.aircraft,
        vertical_wind_fractions=args.vertical_wind_fractions,
        cg_shifts_forward_m=args.cg_shifts,
        inertia_scales=args.inertia_scales,
        reference_report=reference,
        simulation_substeps=args.substeps,
    )
    save_x8_cascade_report(report, args.output)
    persistence = report["kinematic_persistence"]["aggregate"]["horizon_rollouts"]["2s"]
    print(
        f"kinematic persistence: 2s position={persistence['position_rmse_m']:.3f}m  "
        f"attitude={persistence['attitude_rmse_deg']:.2f}deg"
    )
    ordered = sorted(
        report["models"].items(),
        key=lambda item: item[1]["score_vs_kinematic_persistence"],
    )
    for name, model in ordered:
        metrics = model["aggregate"]["horizon_rollouts"]["2s"]
        flag = " (primary)" if model["primary"] else ""
        print(
            f"{name}{flag}: 2s position={metrics['position_rmse_m']:.3f}m  "
            f"velocity={metrics['velocity_rmse_m_s']:.3f}m/s  "
            f"attitude={metrics['attitude_rmse_deg']:.2f}deg  "
            f"rate={metrics['angular_velocity_rmse_rad_s']:.3f}rad/s  "
            f"score/persistence={model['score_vs_kinematic_persistence']:.3f}"
        )
    for name, comparison in report["comparisons"].items():
        if report["primary_model"] in name or report["best_model"] in name:
            print(f"{name}: {comparison['score']:.3f}")
    print(f"wrote Cascade report {args.output}")


def _diagnose_cascade_x8(args: argparse.Namespace) -> None:
    from glassbox.workflows.benchmarks.cascade_x8 import diagnose_x8_cascade

    report = diagnose_x8_cascade(
        args.destination,
        aircraft=args.aircraft,
        split=args.split,
        cg_shift_forward_m=args.cg_shift,
        mass_kg=args.mass,
        inertia_scale=args.inertia_scale,
        vertical_wind_fraction=args.vertical_wind_fraction,
    )
    print(f"{args.aircraft} on {args.split}: configuration {report['configuration']}")
    for channel, item in report["channels"].items():
        terms = ", ".join(
            f"{name}:{value:+.4f}"
            for name, value in item["corrections"].items()
            if abs(value) > 0.003 or name == "1"
        )
        print(
            f"  {channel:5s} mean={item['mean']:+7.2f} rms={item['rms']:6.2f} "
            f"{item['unit']:3s} R2={item['r_squared']:.2f}  {terms}"
        )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as output:
            json.dump(report, output, indent=2, sort_keys=True)
            output.write("\n")
        print(f"wrote residual diagnostic {args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="benchmark", required=True)

    nmpc_parser = subparsers.add_parser(
        "nmpc", help="closed-loop NMPC acceptance and timing suite"
    )
    nmpc_parser.add_argument(
        "--output", type=Path, help="JSON evidence path; stdout is always populated"
    )
    nmpc_parser.set_defaults(handler=_nmpc)

    recovery_parser = subparsers.add_parser(
        "recovery", help="prewarmed synthetic recovery after a configuration change"
    )
    recovery_parser.add_argument("--output", type=Path)
    recovery_parser.set_defaults(handler=_recovery)

    cascade_parser = subparsers.add_parser(
        "cascade-x8",
        help="score the unfitted published Cascade X8 on the validation split",
    )
    cascade_parser.add_argument("destination", type=Path)
    cascade_parser.add_argument("--output", type=Path)
    cascade_parser.add_argument(
        "--diagnose",
        action="store_true",
        help="regress one-step force and moment residuals on the flight variables",
    )
    cascade_parser.add_argument(
        "--aircraft",
        default="skywalker_x8",
        choices=X8_AIRCRAFT,
        help="Cascade X8 backend: published coefficient table or component panels",
    )
    cascade_parser.add_argument(
        "--reference-report",
        type=Path,
        default=None,
        help="benchmark report to compare against "
        "(default: DESTINATION/benchmark_report.json)",
    )
    cascade_parser.add_argument(
        "--vertical-wind-fractions",
        type=_floats,
        default=(0.0, 0.25, 0.5, 0.75, 1.0),
        help="fractions of the campaign's vertical wind estimate to evaluate",
    )
    cascade_parser.add_argument(
        "--cg-shifts",
        type=_floats,
        default=(0.0, 0.02, 0.03, 0.05),
        help="forward CG shifts in metres applied to the pitch triple",
    )
    cascade_parser.add_argument(
        "--inertia-scales",
        type=_floats,
        default=(1.0, 2.0, 3.5),
        help="uniform scale factors applied to the published inertia tensor",
    )
    cascade_parser.add_argument("--substeps", type=int, default=10)
    cascade_parser.add_argument(
        "--split", default="validation", choices=("training", "validation", "all")
    )
    cascade_parser.add_argument("--cg-shift", type=float, default=0.0)
    cascade_parser.add_argument("--mass", type=float, default=None)
    cascade_parser.add_argument("--inertia-scale", type=float, default=1.0)
    cascade_parser.add_argument("--vertical-wind-fraction", type=float, default=1.0)
    cascade_parser.set_defaults(handler=_cascade_x8)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.benchmark == "cascade-x8" and not args.diagnose and args.output is None:
        parser.error("cascade-x8 requires --output")
    args.handler(args)


if __name__ == "__main__":
    main()
