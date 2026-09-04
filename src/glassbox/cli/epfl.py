"""Characterize maintained models on the EPFL TOPOPlane2 same-flight holdout."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from glassbox.workflows.evaluate import evaluate_fit_reports, save_report

# The retained TOPOPlane2 segments all come from one published flight, so the
# 0.2-second horizon the campaign reports is one sample at 5 Hz and the score
# is taken over the three longer horizons only.
EPFL_CHARACTERIZATION_HORIZONS_S = (0.2, 0.5, 1.0, 2.0)
EPFL_SCORE_HORIZONS_S = (0.5, 1.0, 2.0)


def _evaluate(args: argparse.Namespace) -> None:
    report = evaluate_fit_reports(
        {
            "structured": args.structured_report,
            "structured_residual": args.residual_report,
        },
        protocol="windowed",
        horizons_s=EPFL_CHARACTERIZATION_HORIZONS_S,
        score_horizons_s=EPFL_SCORE_HORIZONS_S,
    )
    report["evaluation"] = "epfl_topoplane2_same_flight_characterization"
    report["split"] = "chronological_segments_within_one_source_flight"
    report["interpretation"] = (
        "useful same-flight airframe characterization; independent flights "
        "are required before this result can enter the promotion gate"
    )
    report["limitations"] = [
        "all retained segments come from one published flight",
        "angular velocity is derived from attitude at 5 Hz",
        "the requested 0.5-second horizon resolves to 0.4 seconds at 5 Hz",
        "complete-segment open-loop errors are diagnostic, not an operational claim",
    ]
    save_report(report, args.output)
    for name, model in report["models"].items():
        metrics = model["aggregate_horizon_rollouts"]["2s"]
        print(
            f"{name}: score/persistence={model['score_vs_baseline']:.3f}  "
            f"2s position={metrics['position_rmse_m']:.3f}m  "
            f"attitude={metrics['attitude_rmse_deg']:.2f}deg"
        )
    print(
        f"selected={report['selected_model']} promotion={report['can_promote_model']}"
    )
    print(f"wrote {args.output}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="compare maintained models on the same-flight holdout"
    )
    evaluate_parser.add_argument("--structured-report", type=Path, required=True)
    evaluate_parser.add_argument("--residual-report", type=Path, required=True)
    evaluate_parser.add_argument("--output", type=Path, required=True)
    evaluate_parser.set_defaults(handler=_evaluate)

    args = parser.parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
