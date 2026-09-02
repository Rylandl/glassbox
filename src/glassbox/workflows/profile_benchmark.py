"""Run leave-one-maneuver-profile-out dynamics identification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from glassbox.core.data import load_trajectory_npz
from glassbox.core.evaluation import aggregate_rollout_metrics
from glassbox.core.model_io import save_dynamics_model
from glassbox.core.runtime import runtime_spec_from_fit_report
from glassbox.workflows.acceptance import evaluate_multirotor_accuracy
from glassbox.workflows.fitting import fit_trajectory_artifacts


def _horizons(value: str) -> tuple[float, ...]:
    try:
        result = tuple(dict.fromkeys(float(item.strip()) for item in value.split(",")))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "horizons must be comma-separated numbers"
        ) from error
    if not result or any(item <= 0.0 for item in result):
        raise argparse.ArgumentTypeError("horizons must be positive")
    return result


def benchmark_profiles(
    trajectory_paths: list[str | Path] | tuple[str | Path, ...],
    output_dir: str | Path,
    *,
    training_horizons_s: tuple[float, ...] = (0.1, 0.5, 2.0),
    evaluation_horizons_s: tuple[float, ...] = (0.1, 0.5, 1.0, 2.0),
    steps: int = 2500,
    learning_rate: float = 0.01,
    run_no_lag_ablation: bool = False,
    model_class: str = "structured",
    endpoint_weight: float = 3.0,
    stability_regularization: float = 0.01,
    instantaneous_rotational_response: bool = True,
    diagonal_angular_control: bool = True,
) -> dict[str, Any]:
    """Fit one fold per profile and return an equal-profile macro summary."""

    paths = [Path(path) for path in trajectory_paths]
    if len(paths) < 2:
        raise ValueError("profile benchmark requires at least two trajectories")
    trajectories = [load_trajectory_npz(path) for path in paths]
    profile_values = [trajectory.labels.get("profile") for trajectory in trajectories]
    if any(profile is None for profile in profile_values):
        unlabeled = [
            str(path) for path, profile in zip(paths, profile_values) if profile is None
        ]
        raise ValueError(
            "profile benchmark requires every trajectory to have a profile; "
            f"unlabeled: {', '.join(unlabeled)}"
        )
    profiles = tuple(sorted({str(profile) for profile in profile_values}))
    if len(profiles) < 2:
        raise ValueError("profile benchmark requires at least two distinct profiles")
    state_sources = {
        trajectory.spec.observation_source
        for trajectory in trajectories
        if trajectory.spec is not None
    }
    if len(state_sources) != 1:
        raise ValueError("profile benchmark requires one consistent state_source")
    state_source = next(iter(state_sources))
    platforms = {
        trajectory.spec.vehicle.family
        for trajectory in trajectories
        if trajectory.spec is not None
    }
    if len(platforms) != 1:
        raise ValueError("profile benchmark requires one consistent platform")
    platform = str(next(iter(platforms)))
    control_names = trajectories[0].control_names
    if any(trajectory.control_names != control_names for trajectory in trajectories):
        raise ValueError(
            "profile benchmark requires one consistent ordered control schema"
        )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    per_profile: dict[str, Any] = {}
    fold_full_metrics: list[dict[str, Any]] = []
    fold_horizon_metrics: dict[str, list[dict[str, Any]]] = {
        f"{seconds:g}s": [] for seconds in evaluation_horizons_s
    }

    for profile in profiles:
        print(f"holding out profile: {profile}")
        learned, baseline, report = fit_trajectory_artifacts(
            paths,
            holdout_profiles=(profile,),
            training_horizons_s=training_horizons_s,
            steps=steps,
            learning_rate=learning_rate,
            evaluation_horizons_s=evaluation_horizons_s,
            run_no_lag_ablation=run_no_lag_ablation,
            model_class=model_class,
            endpoint_weight=endpoint_weight,
            stability_regularization=stability_regularization,
            instantaneous_rotational_response=instantaneous_rotational_response,
            diagonal_angular_control=diagonal_angular_control,
        )
        report_path = destination / f"holdout_{profile}_report.json"
        model_path = destination / f"holdout_{profile}_model.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        save_dynamics_model(
            learned,
            model_path,
            input_spec=trajectories[0].spec,
            runtime_spec=runtime_spec_from_fit_report(report),
            provenance={
                "held_out_profile": profile,
                "fit_report": str(report_path),
            },
        )
        baseline_path = None
        if baseline is not None:
            lag_label = "no_motor_lag" if platform == "multirotor" else "no_control_lag"
            baseline_path = destination / f"holdout_{profile}_{lag_label}.json"
            save_dynamics_model(
                baseline,
                baseline_path,
                input_spec=trajectories[0].spec,
                runtime_spec=runtime_spec_from_fit_report(report, model_name="no_lag"),
                provenance={
                    "held_out_profile": profile,
                    "fit_report": str(report_path),
                    "ablation": lag_label,
                },
            )

        validation = report["models"]["learned_lag"]["validation"]
        validation_flights = report["split"]["validation_flights"]
        path_length_m = sum(
            float(flight["characteristics"]["path_length_m"])
            for flight in validation_flights
        ) / len(validation_flights)
        full_metrics = validation["aggregate"]["full_rollout"]
        fold_full_metrics.append(full_metrics)
        for label, metrics in validation["aggregate"]["horizon_rollouts"].items():
            fold_horizon_metrics[label].append(metrics)
        per_profile[profile] = {
            "validation_flight_count": len(report["split"]["validation_flights"]),
            "path_length_m": path_length_m,
            "full_rollout": full_metrics,
            "horizon_rollouts": validation["aggregate"]["horizon_rollouts"],
            "fit": report["models"]["learned_lag"]["fit"],
            "model": str(model_path),
            "baseline_model": str(baseline_path) if baseline_path else None,
            "report": str(report_path),
        }
        print(
            f"  position={full_metrics['position_rmse_m']:.4f}m "
            f"attitude={full_metrics['attitude_rmse_deg']:.3f}deg"
        )

    aggregate_horizons = {
        label: aggregate_rollout_metrics(metrics, weighting="equal")
        for label, metrics in fold_horizon_metrics.items()
        if metrics
    }
    summary = {
        "format_version": 1,
        "evaluation": "leave_one_profile_out",
        "platform": platform,
        "state_source": state_source,
        "trajectory_count": len(paths),
        "profile_count": len(profiles),
        "profiles": list(profiles),
        "configuration": {
            "training_horizons_s": list(training_horizons_s),
            "evaluation_horizons_s": list(evaluation_horizons_s),
            "optimization_steps_per_fold": steps,
            "learning_rate": learning_rate,
            "no_lag_ablation": run_no_lag_ablation,
            "model_class": model_class,
            "endpoint_weight": endpoint_weight,
            "stability_regularization": stability_regularization,
            "rotational_response": (
                "not_applicable_fixedwing"
                if platform != "multirotor"
                else "instantaneous_diagonal_reference"
                if instantaneous_rotational_response
                else "learned_latent_diagonal"
                if diagonal_angular_control
                else "learned_latent_cross_coupled"
            ),
            "control_size": trajectories[0].control_size,
            "control_names": list(control_names),
        },
        "aggregate": {
            "weighting": "equal_profile",
            "full_rollout": aggregate_rollout_metrics(
                fold_full_metrics, weighting="equal"
            ),
            "horizon_rollouts": aggregate_horizons,
        },
        "per_profile": per_profile,
    }
    summary["acceptance"] = (
        evaluate_multirotor_accuracy(
            state_source=state_source,
            aggregate_full_rollout=summary["aggregate"]["full_rollout"],
            aggregate_horizon_rollouts=summary["aggregate"]["horizon_rollouts"],
            per_profile=per_profile,
        )
        if platform == "multirotor"
        else {
            "status": "not_scored",
            "passed": None,
            "platform": platform,
            "reason": "no versioned fixed-wing accuracy contract is defined yet",
        }
    )
    summary_path = destination / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {summary_path}")
    print(f"accuracy contract: {summary['acceptance']['status']}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--training-horizons", type=_horizons, default=(0.1, 0.5, 2.0))
    parser.add_argument(
        "--evaluation-horizons", type=_horizons, default=(0.1, 0.5, 1.0, 2.0)
    )
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--endpoint-weight", type=float, default=3.0)
    parser.add_argument("--stability-regularization", type=float, default=0.01)
    parser.add_argument("--run-no-lag-ablation", action="store_true")
    parser.add_argument(
        "--model-class",
        choices=("structured", "structured_residual"),
        default="structured",
    )
    args = parser.parse_args()
    benchmark_profiles(
        args.trajectory,
        args.output_dir,
        training_horizons_s=args.training_horizons,
        evaluation_horizons_s=args.evaluation_horizons,
        steps=args.steps,
        learning_rate=args.learning_rate,
        run_no_lag_ablation=args.run_no_lag_ablation,
        model_class=args.model_class,
        endpoint_weight=args.endpoint_weight,
        stability_regularization=args.stability_regularization,
    )


if __name__ == "__main__":
    main()
