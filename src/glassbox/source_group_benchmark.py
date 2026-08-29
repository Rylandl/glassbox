"""Run leave-one-source-group-out dynamics identification."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from glassbox.data import Trajectory, duration_to_steps, load_trajectory_npz
from glassbox.evaluation import (
    METRIC_FLOORS,
    ROLLOUT_METRICS,
    aggregate_rollout_metrics,
    kinematic_persistence_windowed_metrics,
)
from glassbox.fit_cli import fit_trajectory_artifacts
from glassbox.identification import (
    MAX_OPTIMIZATION_WINDOWS_PER_HORIZON,
    OPTIMIZATION_POLICY_VERSION,
)
from glassbox.model_io import (
    FIXED_WING_MODEL_TYPE,
    MODEL_TYPE,
    RESIDUAL_MODEL_TYPE,
    save_dynamics_model,
)


_DISTRIBUTION_METRICS = (
    "position_rmse_m",
    "velocity_rmse_m_s",
    "attitude_rmse_deg",
    "angular_velocity_rmse_rad_s",
    "final_position_error_m",
)


def _safe_fold_name(value: str | int) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value)).strip("_")
    return slug or "group"


def _file_record(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _request_digest(request: Mapping[str, Any]) -> str:
    payload = json.dumps(
        request, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _metric_distribution(
    metrics: list[Mapping[str, Any]],
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for name in _DISTRIBUTION_METRICS:
        values = np.asarray([float(item[name]) for item in metrics], dtype=np.float64)
        result[name] = {
            "minimum": float(np.min(values)),
            "median": float(np.median(values)),
            "mean": float(np.mean(values)),
            "p90": float(np.quantile(values, 0.9)),
            "maximum": float(np.max(values)),
        }
    return result


def _source_groups(
    paths: list[Path], trajectories: list[Trajectory]
) -> tuple[str | int, ...]:
    values = [trajectory.labels.get("source_group") for trajectory in trajectories]
    if any(value is None for value in values):
        missing = [
            str(path) for path, value in zip(paths, values) if value is None
        ]
        raise ValueError(
            "source-group benchmark requires every trajectory to have a "
            f"source_group label; unlabeled: {', '.join(missing)}"
        )
    if any(
        not isinstance(value, (str, int))
        or (isinstance(value, str) and not value.strip())
        for value in values
    ):
        raise ValueError(
            "source-group benchmark labels must be non-empty strings or integers"
        )
    return tuple(value for value in values if isinstance(value, (str, int)))


def benchmark_source_groups(
    trajectory_paths: list[str | Path] | tuple[str | Path, ...],
    output_dir: str | Path,
    *,
    training_horizons_s: tuple[float, ...] = (0.1, 0.5, 2.0),
    evaluation_horizons_s: tuple[float, ...] = (0.1, 0.5, 1.0, 2.0),
    steps: int = 400,
    learning_rate: float = 0.02,
    run_no_lag_ablation: bool = False,
    model_class: str = "structured",
    endpoint_weight: float = 3.0,
    stability_regularization: float = 0.01,
    learn_thrust_command_offset: bool = False,
    instantaneous_rotational_response: bool = True,
    diagonal_angular_control: bool = True,
) -> dict[str, Any]:
    """Fit one fold per independent source group and summarize robustness."""

    paths = [Path(path).resolve() for path in trajectory_paths]
    if len(paths) < 2:
        raise ValueError(
            "source-group benchmark requires at least two trajectories"
        )
    trajectories = [load_trajectory_npz(path) for path in paths]
    groups_by_trajectory = _source_groups(paths, trajectories)
    groups = tuple(dict.fromkeys(groups_by_trajectory))
    if len(groups) < 2:
        raise ValueError(
            "source-group benchmark requires at least two distinct source groups"
        )
    display_names = tuple(str(group) for group in groups)
    if len(set(display_names)) != len(display_names):
        raise ValueError(
            "source-group benchmark labels must have unique string representations"
        )

    reference_spec = trajectories[0].spec
    if any(trajectory.spec != reference_spec for trajectory in trajectories[1:]):
        raise ValueError(
            "source-group benchmark requires one consistent trajectory spec"
        )
    base_model_type = (
        FIXED_WING_MODEL_TYPE
        if reference_spec.vehicle.family == "fixedwing"
        else MODEL_TYPE
    )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    summary_path = destination / "summary.json"
    request_path = destination / "request.json"
    request = {
        "format_version": 1,
        "evaluation": "leave_one_source_group_out",
        "files": [_file_record(path) for path in paths],
        "source_groups": list(groups),
        "configuration": {
            "training_horizons_s": list(training_horizons_s),
            "evaluation_horizons_s": list(evaluation_horizons_s),
            "optimization_steps_per_fold": steps,
            "learning_rate": learning_rate,
            "no_lag_ablation": run_no_lag_ablation,
            "model_class": model_class,
            "base_model_type": base_model_type,
            "residual_model_type": (
                RESIDUAL_MODEL_TYPE
                if model_class == "structured_residual"
                else None
            ),
            "endpoint_weight": endpoint_weight,
            "stability_regularization": stability_regularization,
            "multirotor_thrust_command_offset": (
                "not_applicable_fixedwing"
                if reference_spec.vehicle.family != "multirotor"
                else "learned"
                if learn_thrust_command_offset
                else "fixed_zero_reference"
            ),
            "rotational_response": (
                "not_applicable_fixedwing"
                if reference_spec.vehicle.family != "multirotor"
                else "instantaneous_diagonal_reference"
                if instantaneous_rotational_response
                else "learned_latent_diagonal"
                if diagonal_angular_control
                else "learned_latent_cross_coupled"
            ),
            "optimization_policy": OPTIMIZATION_POLICY_VERSION,
            "maximum_optimization_windows_per_horizon": (
                MAX_OPTIMIZATION_WINDOWS_PER_HORIZON
            ),
        },
    }
    if request_path.exists() and summary_path.exists():
        recorded_request = json.loads(request_path.read_text())
        if recorded_request == request:
            print(f"resume complete source-group benchmark: {summary_path}")
            return json.loads(summary_path.read_text())
    request_path.write_text(json.dumps(request, indent=2) + "\n")
    dataset_request_digest = _request_digest(request)

    per_source_group: dict[str, Any] = {}
    fold_full_metrics: list[dict[str, Any]] = []
    fold_horizon_metrics: dict[str, list[dict[str, Any]]] = {
        f"{seconds:g}s": [] for seconds in evaluation_horizons_s
    }
    fold_persistence_metrics: dict[str, list[dict[str, Any]]] = {
        f"{seconds:g}s": [] for seconds in evaluation_horizons_s
    }

    for fold_index, (group, display_name) in enumerate(
        zip(groups, display_names), start=1
    ):
        print(f"holding out source group: {display_name}")
        training_paths = [
            path
            for path, path_group in zip(paths, groups_by_trajectory)
            if path_group != group
        ]
        validation_paths = [
            path
            for path, path_group in zip(paths, groups_by_trajectory)
            if path_group == group
        ]
        validation_trajectories = [
            trajectory
            for trajectory, path_group in zip(
                trajectories, groups_by_trajectory
            )
            if path_group == group
        ]
        fold_paths = [*training_paths, *validation_paths]
        prefix = f"fold_{fold_index:02d}_{_safe_fold_name(group)}"
        report_path = destination / f"{prefix}_report.json"
        model_path = destination / f"{prefix}_model.json"
        fold_request_path = destination / f"{prefix}_request.json"
        fold_request = {
            "format_version": 1,
            "dataset_request_sha256": dataset_request_digest,
            "held_out_source_group": group,
        }
        lag_label = (
            "no_motor_lag"
            if reference_spec.vehicle.family == "multirotor"
            else "no_control_lag"
        )
        expected_baseline_path = destination / f"{prefix}_{lag_label}.json"
        reusable = (
            fold_request_path.exists()
            and report_path.exists()
            and model_path.exists()
            and (
                not run_no_lag_ablation
                or expected_baseline_path.exists()
            )
            and json.loads(fold_request_path.read_text()) == fold_request
        )
        if reusable:
            print(f"  resume fold: {report_path}")
            report = json.loads(report_path.read_text())
            baseline_path = (
                expected_baseline_path if run_no_lag_ablation else None
            )
        else:
            learned, baseline, report = fit_trajectory_artifacts(
                fold_paths,
                holdout_count=1,
                training_horizons_s=training_horizons_s,
                steps=steps,
                learning_rate=learning_rate,
                evaluation_horizons_s=evaluation_horizons_s,
                run_no_lag_ablation=run_no_lag_ablation,
                model_class=model_class,
                endpoint_weight=endpoint_weight,
                stability_regularization=stability_regularization,
                learn_thrust_command_offset=learn_thrust_command_offset,
                instantaneous_rotational_response=(
                    instantaneous_rotational_response
                ),
                diagonal_angular_control=diagonal_angular_control,
            )
            if report["split"]["validation_source_groups"] != [group]:
                raise RuntimeError(
                    "source-group fold did not preserve its holdout boundary"
                )
            report_path.write_text(json.dumps(report, indent=2) + "\n")
            save_dynamics_model(
                learned,
                model_path,
                input_spec=reference_spec,
                provenance={
                    "evaluation": "leave_one_source_group_out",
                    "held_out_source_group": group,
                    "fit_report": str(report_path),
                },
            )
            baseline_path = None
            if baseline is not None:
                baseline_path = expected_baseline_path
                save_dynamics_model(
                    baseline,
                    baseline_path,
                    input_spec=reference_spec,
                    provenance={
                        "evaluation": "leave_one_source_group_out",
                        "held_out_source_group": group,
                        "fit_report": str(report_path),
                        "ablation": lag_label,
                    },
                )
            fold_request_path.write_text(
                json.dumps(fold_request, indent=2) + "\n"
            )

        validation = report["models"]["learned_lag"]["validation"]
        full_metrics = validation["aggregate"]["full_rollout"]
        horizon_rollouts = validation["aggregate"]["horizon_rollouts"]
        fold_full_metrics.append(full_metrics)
        for label, metrics in horizon_rollouts.items():
            fold_horizon_metrics[label].append(metrics)
        persistence_horizons = {}
        model_over_persistence = {}
        for seconds in evaluation_horizons_s:
            label = f"{seconds:g}s"
            persistence_metrics = []
            for trajectory in validation_trajectories:
                horizon_steps = duration_to_steps(
                    seconds, trajectory.nominal_dt_s
                )
                persistence_metrics.append(
                    kinematic_persistence_windowed_metrics(
                        trajectory,
                        horizon_steps=horizon_steps,
                        stride_steps=horizon_steps,
                    )
                )
            persistence = aggregate_rollout_metrics(
                persistence_metrics,
                weighting="equal",
            )
            persistence_horizons[label] = persistence
            fold_persistence_metrics[label].append(persistence)
            model_over_persistence[label] = {
                metric: max(
                    float(horizon_rollouts[label][metric]),
                    METRIC_FLOORS[metric],
                )
                / max(float(persistence[metric]), METRIC_FLOORS[metric])
                for metric in ROLLOUT_METRICS
            }
        per_source_group[display_name] = {
            "source_group": group,
            "validation_trajectory_count": len(validation_paths),
            "validation_duration_s": sum(
                float(item["duration_s"])
                for item in report["split"]["validation_flights"]
            ),
            "full_rollout": full_metrics,
            "horizon_rollouts": horizon_rollouts,
            "kinematic_persistence_horizon_rollouts": persistence_horizons,
            "model_over_kinematic_persistence": model_over_persistence,
            "fit": report["models"]["learned_lag"]["fit"],
            "training_window_selection": report["configuration"][
                "training_window_selection"
            ],
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
    aggregate_persistence_horizons = {
        label: aggregate_rollout_metrics(metrics, weighting="equal")
        for label, metrics in fold_persistence_metrics.items()
        if metrics
    }
    summary = {
        "format_version": 1,
        "evaluation": "leave_one_source_group_out",
        "platform": reference_spec.vehicle.family,
        "state_source": reference_spec.observation_source,
        "trajectory_count": len(paths),
        "source_group_count": len(groups),
        "source_groups": list(groups),
        "configuration": {
            "training_horizons_s": list(training_horizons_s),
            "evaluation_horizons_s": list(evaluation_horizons_s),
            "optimization_steps_per_fold": steps,
            "learning_rate": learning_rate,
            "no_lag_ablation": run_no_lag_ablation,
            "model_class": model_class,
            "base_model_type": base_model_type,
            "residual_model_type": (
                RESIDUAL_MODEL_TYPE
                if model_class == "structured_residual"
                else None
            ),
            "endpoint_weight": endpoint_weight,
            "stability_regularization": stability_regularization,
            "multirotor_thrust_command_offset": (
                "not_applicable_fixedwing"
                if reference_spec.vehicle.family != "multirotor"
                else "learned"
                if learn_thrust_command_offset
                else "fixed_zero_reference"
            ),
            "rotational_response": (
                "not_applicable_fixedwing"
                if reference_spec.vehicle.family != "multirotor"
                else "instantaneous_diagonal_reference"
                if instantaneous_rotational_response
                else "learned_latent_diagonal"
                if diagonal_angular_control
                else "learned_latent_cross_coupled"
            ),
            "control_size": trajectories[0].control_size,
            "control_names": list(reference_spec.control_names),
            "control_semantics": list(reference_spec.control_semantics),
            "exogenous_size": trajectories[0].exogenous_size,
            "exogenous_names": list(reference_spec.exogenous_names),
            "exogenous_roles": list(reference_spec.exogenous_roles),
            "fold_selection": "all_source_groups",
        },
        "aggregate": {
            "weighting": "equal_source_group",
            "full_rollout": aggregate_rollout_metrics(
                fold_full_metrics, weighting="equal"
            ),
            "horizon_rollouts": aggregate_horizons,
            "kinematic_persistence_horizon_rollouts": (
                aggregate_persistence_horizons
            ),
            "model_over_kinematic_persistence": {
                label: {
                    metric: max(
                        float(aggregate_horizons[label][metric]),
                        METRIC_FLOORS[metric],
                    )
                    / max(float(persistence[metric]), METRIC_FLOORS[metric])
                    for metric in ROLLOUT_METRICS
                }
                for label, persistence in aggregate_persistence_horizons.items()
            },
        },
        "distribution": {
            "full_rollout": _metric_distribution(fold_full_metrics),
            "horizon_rollouts": {
                label: _metric_distribution(metrics)
                for label, metrics in fold_horizon_metrics.items()
                if metrics
            },
            "kinematic_persistence_horizon_rollouts": {
                label: _metric_distribution(metrics)
                for label, metrics in fold_persistence_metrics.items()
                if metrics
            },
        },
        "per_source_group": per_source_group,
        "acceptance": {
            "status": "not_scored",
            "passed": None,
            "reason": (
                "source-group accuracy thresholds require empirical fold "
                "distributions before they can be versioned"
            ),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {summary_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model-class",
        choices=("structured", "structured_residual"),
        default="structured",
    )
    args = parser.parse_args()
    benchmark_source_groups(
        args.trajectory,
        args.output_dir,
        model_class=args.model_class,
    )


if __name__ == "__main__":
    main()
