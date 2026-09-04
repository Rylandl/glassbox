"""Fit an effective differentiable dynamics model from a trajectory artifact."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from glassbox.belief.belief_io import save_dynamics_belief
from glassbox.fitting import FitSpec, Holdout, LossPolicy, WeightingPolicy, fit


def _evaluation_horizons(value: str) -> tuple[float, ...]:
    try:
        horizons = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "evaluation horizons must be comma-separated numbers"
        ) from error
    if not horizons or any(item <= 0.0 for item in horizons):
        raise argparse.ArgumentTypeError("evaluation horizons must be positive")
    return tuple(dict.fromkeys(horizons))


def _holdout_label(value: str) -> tuple[str, str]:
    key, separator, label = value.partition("=")
    if not separator or not key.strip() or not label.strip():
        raise argparse.ArgumentTypeError("a holdout label must be given as KEY=VALUE")
    return key.strip(), label.strip()


def _no_lag_model_path(model_path: Path) -> Path:
    return model_path.with_name(f"{model_path.stem}_no_motor_lag{model_path.suffix}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path, nargs="+")
    parser.add_argument("--model", type=Path, help="output dynamics-belief JSON")
    parser.add_argument(
        "--baseline-model",
        type=Path,
        help="output no-lag dynamics-belief JSON; defaults beside --model",
    )
    parser.add_argument("--report", type=Path, help="output fit report JSON")
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.70,
        help="training share of the one flight a single-trajectory fit splits",
    )
    parser.add_argument(
        "--holdout-count",
        type=int,
        help=(
            "number of final source groups reserved completely for validation, "
            "one by default; falls back to input trajectories in argument order "
            "when the source_group label separates nothing"
        ),
    )
    parser.add_argument(
        "--holdout-profile",
        action="append",
        help=("maneuver profile to reserve completely; repeat for multiple profiles"),
    )
    parser.add_argument(
        "--holdout-label",
        action="append",
        type=_holdout_label,
        metavar="KEY=VALUE",
        help=(
            "reserve every flight whose KEY label is VALUE; repeat for several "
            "values of one key, for example benchmark_split=validation"
        ),
    )
    parser.add_argument("--horizon", type=int, default=25)
    parser.add_argument("--stride", type=int)
    parser.add_argument(
        "--training-horizons",
        type=_evaluation_horizons,
        help=(
            "comma-separated rollout horizons in seconds; combines normalized "
            "losses and supersedes --horizon"
        ),
    )
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument(
        "--endpoint-weight",
        type=float,
        default=3.0,
        help="relative loss weight on the final rollout step; must be at least one",
    )
    parser.add_argument(
        "--stability-regularization",
        type=float,
        default=0.01,
        help=(
            "penalty on predicted body velocity/rates outside the robust "
            "training envelope"
        ),
    )
    parser.add_argument(
        "--model-class",
        choices=("structured", "structured_residual"),
        default="structured",
        help="dynamics parameterization to fit",
    )
    parser.add_argument(
        "--evaluation-horizons",
        type=_evaluation_horizons,
        default=(0.1, 0.5, 1.0, 2.0),
        help="comma-separated held-out rollout horizons in seconds",
    )
    parser.add_argument(
        "--ablation",
        action="append",
        choices=("no-lag",),
        default=[],
        help=(
            "also fit this ablation and write it beside --model; repeatable. "
            "no-lag pins a near-zero applied-control response and reports the "
            "learned-lag improvement over it"
        ),
    )
    parser.add_argument(
        "--duration-weighted-training",
        action="store_true",
        help=(
            "weight training by extracted window count instead of giving each "
            "complete flight equal total weight"
        ),
    )
    parser.add_argument(
        "--fixed-response-time-constant",
        "--fixed-motor-time-constant",
        dest="fixed_motor_time_constant",
        type=float,
        help=(
            "pin the family-specific applied-control response time instead of "
            "learning it; incompatible with --ablation no-lag"
        ),
    )
    return parser


def _resolve_holdout(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> Holdout:
    """Pick the one holdout rule the requested flags and inputs describe."""

    requested = [
        name
        for name, value in (
            ("--holdout-label", args.holdout_label),
            ("--holdout-profile", args.holdout_profile),
            ("--holdout-count", args.holdout_count),
        )
        if value is not None
    ]
    if len(requested) > 1:
        parser.error(f"{' and '.join(requested)} select different holdouts")
    if args.holdout_label:
        keys = {key for key, _ in args.holdout_label}
        if len(keys) > 1:
            parser.error("every --holdout-label must name the same key")
        return Holdout.by_label(
            args.holdout_label[0][0], tuple(value for _, value in args.holdout_label)
        )
    if args.holdout_profile:
        return Holdout.by_label("profile", tuple(args.holdout_profile))
    if len(args.trajectory) == 1:
        if args.holdout_count is not None:
            parser.error(
                "one trajectory is split chronologically by --train-fraction; "
                "--holdout-count needs at least two"
            )
        return Holdout.temporal(args.train_fraction)
    return Holdout.by_group(1 if args.holdout_count is None else args.holdout_count)


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    ablations = tuple(name.replace("-", "_") for name in args.ablation)
    if args.baseline_model is not None and args.model is None:
        parser.error("--baseline-model requires --model")
    if args.baseline_model is not None and "no_lag" not in ablations:
        parser.error("--baseline-model requires --ablation no-lag")
    if args.fixed_motor_time_constant is not None and "no_lag" in ablations:
        parser.error(
            "--fixed-response-time-constant cannot be used with --ablation no-lag; "
            "the ablation pins the same response time"
        )

    outcome = fit(
        args.trajectory,
        FitSpec(
            holdout=_resolve_holdout(parser, args),
            horizons_s=args.training_horizons,
            horizon_steps=args.horizon,
            stride=args.stride,
            steps=args.steps,
            learning_rate=args.learning_rate,
            evaluation_horizons_s=args.evaluation_horizons,
            model_class=args.model_class,
            ablations=ablations,
            parameter_evidence=args.model is not None,
            fixed_response_time_constant_s=args.fixed_motor_time_constant,
            loss=LossPolicy(
                endpoint_weight=args.endpoint_weight,
                stability_regularization=args.stability_regularization,
            ),
            weighting=WeightingPolicy(balanced=not args.duration_weighted_training),
        ),
    )
    report = outcome.report
    learned = report["models"]["learned_lag"]
    learned_fit = learned["fit"]
    learned_full = learned["validation"]["aggregate"]["full_rollout"]
    validation_label = (
        "held-out temporal rollout"
        if report["split"]["mode"] == "temporal_within_flight"
        else "held-out complete-source rollout"
    )
    print(
        f"learned-lag loss: {learned_fit['initial_loss']:.6g} -> "
        f"{learned_fit['final_loss']:.6g} "
        f"({learned_fit['loss_reduction']:.1f}x reduction)"
    )
    print(
        f"{validation_label}: "
        f"position={learned_full['position_rmse_m']:.4f} m  "
        f"attitude={learned_full['attitude_rmse_deg']:.3f} deg"
    )
    if "no_lag" in outcome.ablations:
        baseline_full = report["models"]["no_lag"]["validation"]["aggregate"][
            "full_rollout"
        ]
        ratios = report["comparison"]["aggregate_full_rollout"]
        print(
            "no-lag ablation: "
            f"position={baseline_full['position_rmse_m']:.4f} m  "
            f"attitude={baseline_full['attitude_rmse_deg']:.3f} deg"
        )
        print(
            "learned-lag improvement: "
            f"position={ratios['position_rmse_m']:.2f}x  "
            f"attitude={ratios['attitude_rmse_deg']:.2f}x"
        )

    if args.model is not None:
        report_path = str(args.report) if args.report else None

        def stamped(belief):
            return replace(
                belief, provenance={**belief.provenance, "fit_report": report_path}
            )

        save_dynamics_belief(stamped(outcome.belief), args.model)
        print(f"wrote dynamics belief {args.model}")
        if "no_lag" in outcome.ablations:
            baseline_path = args.baseline_model or _no_lag_model_path(args.model)
            save_dynamics_belief(stamped(outcome.ablations["no_lag"]), baseline_path)
            print(f"wrote no-lag dynamics belief {baseline_path}")
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote report {args.report}")


if __name__ == "__main__":
    main()
