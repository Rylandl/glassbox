"""Fit an effective differentiable dynamics model from a trajectory artifact."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from glassbox.belief.belief import (
    DynamicsBelief,
    UnavailableParameterEvidence,
    UnavailablePredictiveError,
    parameter_evidence_from_dict,
    predictive_error_from_dict,
)
from glassbox.belief.belief_io import save_dynamics_belief
from glassbox.core.data import TrajectorySpec
from glassbox.core.runtime import runtime_spec_from_fit_report
from glassbox.workflows.fitting import (
    BenchmarkSplitHoldoutConflict,
    fit_trajectory_artifact,
    fit_trajectory_artifacts,
)

_COMPLETE_HOLDOUT_MODES = {
    "leave_complete_flights_out",
    "leave_profiles_out",
    "leave_source_groups_out",
    "benchmark_split_holdout",
}


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
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument(
        "--holdout-count",
        type=int,
        default=1,
        help=(
            "number of final source groups reserved completely for validation; "
            "falls back to input trajectories when groups are unlabeled; "
            "rejected when every trajectory carries a benchmark_split label, "
            "which determines the holdout instead"
        ),
    )
    parser.add_argument(
        "--holdout-profile",
        action="append",
        help=(
            "maneuver profile to reserve completely; repeat for multiple profiles "
            "and supersedes --holdout-count; rejected when every trajectory "
            "carries a benchmark_split label, which determines the holdout instead"
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
        "--skip-no-lag-ablation",
        action="store_true",
        help="fit only the learned-lag model",
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
        help="single-flight mode with a fixed family-specific control response time",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.baseline_model is not None and args.model is None:
        parser.error("--baseline-model requires --model")
    if args.baseline_model is not None and args.skip_no_lag_ablation:
        parser.error("--baseline-model cannot be used when the ablation is skipped")

    if args.fixed_motor_time_constant is not None:
        if len(args.trajectory) != 1:
            parser.error(
                "--fixed-response-time-constant requires exactly one trajectory"
            )
        if args.baseline_model is not None:
            parser.error("--baseline-model is not used in fixed-response mode")
        if args.training_horizons is not None:
            parser.error("--training-horizons is not used in fixed-response mode")
        if args.model_class != "structured":
            parser.error(
                "--fixed-response-time-constant only supports --model-class structured"
            )
        params, report = fit_trajectory_artifact(
            args.trajectory[0],
            train_fraction=args.train_fraction,
            horizon=args.horizon,
            stride=args.stride,
            steps=args.steps,
            learning_rate=args.learning_rate,
            fixed_motor_time_constant_s=args.fixed_motor_time_constant,
            endpoint_weight=args.endpoint_weight,
            stability_regularization=args.stability_regularization,
        )
        baseline_params = None
        fit = report["fit"]
        validation = report["validation_rollout"]["fitted"]
        print(
            f"loss: {fit['initial_loss']:.6g} -> {fit['final_loss']:.6g} "
            f"({fit['loss_reduction']:.1f}x reduction)"
        )
        print(
            "held-out validation: "
            f"position={validation['position_rmse_m']:.4f} m  "
            f"attitude={validation['attitude_rmse_deg']:.3f} deg"
        )
    else:
        try:
            params, baseline_params, report = fit_trajectory_artifacts(
                args.trajectory,
                train_fraction=args.train_fraction,
                holdout_count=args.holdout_count,
                horizon=args.horizon,
                stride=args.stride,
                training_horizons_s=args.training_horizons,
                steps=args.steps,
                learning_rate=args.learning_rate,
                evaluation_horizons_s=args.evaluation_horizons,
                run_no_lag_ablation=not args.skip_no_lag_ablation,
                balance_training_flights=not args.duration_weighted_training,
                holdout_profiles=args.holdout_profile,
                model_class=args.model_class,
                endpoint_weight=args.endpoint_weight,
                stability_regularization=args.stability_regularization,
                build_parameter_evidence=args.model is not None,
            )
        except BenchmarkSplitHoldoutConflict as error:
            parser.error(str(error))
        learned = report["models"]["learned_lag"]
        learned_fit = learned["fit"]
        learned_full = learned["validation"]["aggregate"]["full_rollout"]
        validation_label = (
            "held-out complete-source rollout"
            if report["split"]["mode"] in _COMPLETE_HOLDOUT_MODES
            else "held-out temporal rollout"
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
        if baseline_params is not None:
            baseline = report["models"]["no_lag"]
            baseline_full = baseline["validation"]["aggregate"]["full_rollout"]
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

    if "split" in report:
        training_paths = [item["path"] for item in report["split"]["training_flights"]]
        validation_paths = [
            item["path"] for item in report["split"]["validation_flights"]
        ]
    else:
        training_paths = [str(path) for path in args.trajectory]
        validation_paths = []

    if args.model is not None:
        input_spec = TrajectorySpec.from_dict(
            report["dataset"]["trajectory_spec"]
            if "dataset" in report
            else report["source"]["spec"]
        )
        provenance = {
            "training_trajectories": training_paths,
            "validation_trajectories": validation_paths,
            "fit_report": str(args.report) if args.report else None,
        }
        predictive_error = (
            predictive_error_from_dict(
                report["models"]["learned_lag"]["validation"]["predictive_error"]
            )
            if "models" in report
            else UnavailablePredictiveError(
                "single-trajectory fixed-response fitting does not produce "
                "a fixed-horizon held-out error profile"
            )
        )
        parameter_evidence = (
            parameter_evidence_from_dict(
                report["models"]["learned_lag"]["parameter_evidence"]
            )
            if "models" in report
            else UnavailableParameterEvidence(
                "single-trajectory fixed-response fitting does not evaluate "
                "grouped local parameter information"
            )
        )
        save_dynamics_belief(
            DynamicsBelief(
                params=params,
                input_spec=input_spec,
                runtime_spec=runtime_spec_from_fit_report(report),
                predictive_error=predictive_error,
                parameter_evidence=parameter_evidence,
                provenance=provenance,
            ),
            args.model,
        )
        print(f"wrote dynamics belief {args.model}")
        if baseline_params is not None:
            baseline_path = args.baseline_model or _no_lag_model_path(args.model)
            baseline_provenance = {
                "training_trajectories": training_paths,
                "validation_trajectories": validation_paths,
                "fit_report": str(args.report) if args.report else None,
                "ablation": "fixed near-zero applied-control response",
            }
            save_dynamics_belief(
                DynamicsBelief(
                    params=baseline_params,
                    input_spec=input_spec,
                    runtime_spec=runtime_spec_from_fit_report(
                        report, model_name="no_lag"
                    ),
                    predictive_error=predictive_error_from_dict(
                        report["models"]["no_lag"]["validation"]["predictive_error"]
                    ),
                    parameter_evidence=parameter_evidence_from_dict(
                        report["models"]["no_lag"]["parameter_evidence"]
                    ),
                    provenance=baseline_provenance,
                ),
                baseline_path,
            )
            print(f"wrote no-lag dynamics belief {baseline_path}")
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote report {args.report}")


if __name__ == "__main__":
    main()
