"""One leave-one-label-out runner over any holdout label.

Leaving one maneuver profile out and leaving one independent source group out
were two copies of the same loop: fit every fold, write each fold's belief and
report, and summarize the folds equally weighted. The only difference was the
label the folds are keyed by, so that label is the argument.

Every fold is scored against the ``windowed`` protocol's kinematic-persistence
baseline, and the summary carries both the equal-fold aggregate and the
distribution across folds, because one badly predicted fold is the result that
matters and an average hides it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from glassbox.belief.belief_io import save_dynamics_belief
from glassbox.core.data import Trajectory, load_trajectory_npz
from glassbox.core.identification import (
    MAX_OPTIMIZATION_WINDOWS_PER_HORIZON,
    OPTIMIZATION_POLICY_VERSION,
)
from glassbox.core.metrics import (
    METRIC_FLOORS,
    ROLLOUT_METRICS,
    aggregate_rollout_metrics,
)
from glassbox.core.model_io import (
    FIXED_WING_MODEL_TYPE,
    MODEL_TYPE,
    RESIDUAL_MODEL_TYPE,
)
from glassbox.fitting import DEFAULT_FIT_SPEC, FitSpec, Holdout, LossPolicy, fit
from glassbox.workflows.evaluate import baseline_horizon_rollouts

_DISTRIBUTION_METRICS = (
    "position_rmse_m",
    "velocity_rmse_m_s",
    "attitude_rmse_deg",
    "angular_velocity_rmse_rad_s",
    "final_position_error_m",
)


def _safe_fold_name(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")
    return slug or "fold"


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
    metrics: Sequence[Mapping[str, Any]],
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


def _holdout_key(hold_out: Holdout | str) -> str:
    if isinstance(hold_out, str):
        key = hold_out
    elif hold_out.rule == "temporal":
        raise ValueError(
            "a temporal holdout splits one flight and has no folds; "
            "hold out a label instead"
        )
    else:
        key = hold_out.key
    if not key.strip():
        raise ValueError("the holdout label key cannot be empty")
    return key


def _fold_values(
    key: str, paths: Sequence[Path], trajectories: Sequence[Trajectory]
) -> tuple[str, ...]:
    values = [trajectory.labels.get(key) for trajectory in trajectories]
    missing = [str(path) for path, value in zip(paths, values) if value is None]
    if missing:
        raise ValueError(
            f"a {key} holdout requires every trajectory to have a {key} label; "
            f"unlabeled: {', '.join(missing)}"
        )
    if any(
        not isinstance(value, (str, int))
        or (isinstance(value, str) and not value.strip())
        for value in values
    ):
        raise ValueError(f"{key} labels must be non-empty strings or integers")
    labels = tuple(str(value) for value in values)
    if len(set(labels)) != len(set(values)):
        raise ValueError(f"{key} labels must have unique string representations")
    return labels


def evaluate_holdout(
    trajectories: Sequence[str | Path],
    *,
    hold_out: Holdout | str = "source_group",
    spec: FitSpec = DEFAULT_FIT_SPEC,
    output_dir: str | Path,
    resume: bool = True,
) -> dict[str, Any]:
    """Fit one fold per distinct value of a holdout label and summarize them.

    ``hold_out`` names the label the folds are keyed by, either as the key
    itself or as the :class:`~glassbox.Holdout` rule that carries it; every
    distinct value of that label becomes one fold that reserves every flight
    carrying it. ``spec`` shapes each fold's fit, except for its holdout rule,
    which the fold supplies.

    With ``resume``, a fold whose request, report, belief and any requested
    ablation are already on disk is reused instead of refitted, and a complete
    run whose request matches returns its recorded summary untouched.
    """

    key = _holdout_key(hold_out)
    paths = [Path(path).resolve() for path in trajectories]
    if len(paths) < 2:
        raise ValueError(f"a {key} holdout requires at least two trajectories")
    flights = [load_trajectory_npz(path) for path in paths]
    values = _fold_values(key, paths, flights)
    folds = tuple(dict.fromkeys(values))
    if len(folds) < 2:
        raise ValueError(f"a {key} holdout requires at least two distinct {key} values")

    reference_spec = flights[0].spec
    if any(flight.spec != reference_spec for flight in flights[1:]):
        raise ValueError(f"a {key} holdout requires one consistent trajectory spec")
    family = reference_spec.vehicle.family
    base_model_type = FIXED_WING_MODEL_TYPE if family == "fixedwing" else MODEL_TYPE
    lag_label = "no_motor_lag" if family == "multirotor" else "no_control_lag"

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    summary_path = destination / "summary.json"
    request_path = destination / "request.json"
    configuration = {
        "holdout_label": key,
        "training_horizons_s": (
            None if spec.horizons_s is None else list(spec.horizons_s)
        ),
        "training_horizon_steps": spec.horizon_steps,
        "evaluation_horizons_s": list(spec.evaluation_horizons_s),
        "optimization_steps_per_fold": spec.steps,
        "learning_rate": spec.learning_rate,
        "ablations": list(spec.ablations),
        "model_class": spec.model_class,
        "base_model_type": base_model_type,
        "residual_model_type": (
            RESIDUAL_MODEL_TYPE if spec.model_class == "structured_residual" else None
        ),
        "endpoint_weight": spec.loss.endpoint_weight,
        "stability_regularization": spec.loss.stability_regularization,
        "learn_thrust_command_offset": spec.loss.learn_thrust_command_offset,
        "instantaneous_rotational_response": (
            spec.loss.instantaneous_rotational_response
        ),
        "diagonal_angular_control": spec.loss.diagonal_angular_control,
        "optimization_policy": OPTIMIZATION_POLICY_VERSION,
        "maximum_optimization_windows_per_horizon": (
            MAX_OPTIMIZATION_WINDOWS_PER_HORIZON
        ),
        "control_size": flights[0].control_size,
        "control_names": list(reference_spec.control_names),
        "control_semantics": list(reference_spec.control_semantics),
        "exogenous_size": flights[0].exogenous_size,
        "exogenous_names": list(reference_spec.exogenous_names),
        "exogenous_roles": list(reference_spec.exogenous_roles),
        "fold_selection": f"all_{key}_values",
    }
    request = {
        "format_version": 1,
        "evaluation": f"leave_one_{key}_out",
        "files": [_file_record(path) for path in paths],
        "folds": list(folds),
        "configuration": configuration,
    }
    if resume and request_path.exists() and summary_path.exists():
        if json.loads(request_path.read_text()) == request:
            print(f"resume complete {key} holdout: {summary_path}")
            return json.loads(summary_path.read_text())
    request_path.write_text(json.dumps(request, indent=2) + "\n")
    dataset_request_digest = _request_digest(request)

    per_fold: dict[str, Any] = {}
    fold_full_metrics: list[dict[str, Any]] = []
    horizon_labels = [f"{seconds:g}s" for seconds in spec.evaluation_horizons_s]
    fold_horizon_metrics: dict[str, list[dict[str, Any]]] = {
        label: [] for label in horizon_labels
    }
    fold_baseline_metrics: dict[str, list[dict[str, Any]]] = {
        label: [] for label in horizon_labels
    }

    for index, value in enumerate(folds, start=1):
        print(f"holding out {key}: {value}")
        held_out = [
            (path, flight)
            for path, flight, label in zip(paths, flights, values)
            if label == value
        ]
        prefix = f"fold_{index:02d}_{_safe_fold_name(value)}"
        report_path = destination / f"{prefix}_report.json"
        model_path = destination / f"{prefix}_model.json"
        fold_request_path = destination / f"{prefix}_request.json"
        baseline_path = destination / f"{prefix}_{lag_label}.json"
        fold_request = {
            "format_version": 1,
            "dataset_request_sha256": dataset_request_digest,
            "held_out": {"key": key, "value": value},
        }
        wants_ablation = "no_lag" in spec.ablations
        reusable = (
            resume
            and fold_request_path.exists()
            and report_path.exists()
            and model_path.exists()
            and (not wants_ablation or baseline_path.exists())
            and json.loads(fold_request_path.read_text()) == fold_request
        )
        if reusable:
            print(f"  resume fold: {report_path}")
            report = json.loads(report_path.read_text())
        else:
            outcome = fit(paths, replace(spec, holdout=Holdout.by_label(key, (value,))))
            report = outcome.report
            report_path.write_text(json.dumps(report, indent=2) + "\n")
            provenance = {
                "evaluation": f"leave_one_{key}_out",
                "held_out_label": key,
                "held_out_value": value,
                "fit_report": str(report_path),
            }
            save_dynamics_belief(
                replace(outcome.belief, provenance=provenance), model_path
            )
            if "no_lag" in outcome.ablations:
                save_dynamics_belief(
                    replace(
                        outcome.ablations["no_lag"],
                        provenance={**provenance, "ablation": lag_label},
                    ),
                    baseline_path,
                )
            fold_request_path.write_text(json.dumps(fold_request, indent=2) + "\n")

        validation = report["models"]["learned_lag"]["validation"]
        full_metrics = validation["aggregate"]["full_rollout"]
        horizon_rollouts = validation["aggregate"]["horizon_rollouts"]
        fold_full_metrics.append(full_metrics)
        for label, metrics in horizon_rollouts.items():
            fold_horizon_metrics[label].append(metrics)
        baseline = baseline_horizon_rollouts(
            [flight for _, flight in held_out],
            protocol="windowed",
            horizons_s=spec.evaluation_horizons_s,
        )
        for label, metrics in baseline.items():
            fold_baseline_metrics[label].append(metrics)
        validation_flights = report["split"]["validation_flights"]
        per_fold[value] = {
            "held_out_value": value,
            "validation_trajectory_count": len(held_out),
            "validation_duration_s": sum(
                float(item["duration_s"]) for item in validation_flights
            ),
            "path_length_m": sum(
                float(item["characteristics"]["path_length_m"])
                for item in validation_flights
            )
            / len(validation_flights),
            "full_rollout": full_metrics,
            "horizon_rollouts": horizon_rollouts,
            "baseline_horizon_rollouts": baseline,
            "model_over_baseline": {
                label: {
                    metric: max(
                        float(horizon_rollouts[label][metric]), METRIC_FLOORS[metric]
                    )
                    / max(float(baseline[label][metric]), METRIC_FLOORS[metric])
                    for metric in ROLLOUT_METRICS
                }
                for label in baseline
            },
            "fit": report["models"]["learned_lag"]["fit"],
            "training_window_selection": report["configuration"][
                "training_window_selection"
            ],
            "model": str(model_path),
            "baseline_model": str(baseline_path) if wants_ablation else None,
            "report": str(report_path),
        }
        print(
            f"  position={full_metrics['position_rmse_m']:.4f}m "
            f"attitude={full_metrics['attitude_rmse_deg']:.3f}deg"
        )

    aggregate_horizons = _aggregate_by_label(fold_horizon_metrics)
    aggregate_baseline = _aggregate_by_label(fold_baseline_metrics)
    summary = {
        "format_version": 1,
        "evaluation": f"leave_one_{key}_out",
        "holdout_label": key,
        "protocol": "windowed",
        "baseline": "kinematic_persistence",
        "floors": dict(METRIC_FLOORS),
        "independent_holdout": True,
        "platform": family,
        "state_source": reference_spec.observation_source,
        "trajectory_count": len(paths),
        "fold_count": len(folds),
        "folds": list(folds),
        "configuration": configuration,
        "aggregate": {
            "weighting": f"equal_{key}",
            "full_rollout": _aggregate_equal(fold_full_metrics),
            "horizon_rollouts": aggregate_horizons,
            "baseline_horizon_rollouts": aggregate_baseline,
            "model_over_baseline": {
                label: {
                    metric: max(
                        float(aggregate_horizons[label][metric]), METRIC_FLOORS[metric]
                    )
                    / max(float(baseline[metric]), METRIC_FLOORS[metric])
                    for metric in ROLLOUT_METRICS
                }
                for label, baseline in aggregate_baseline.items()
            },
        },
        "distribution": {
            "full_rollout": _metric_distribution(fold_full_metrics),
            "horizon_rollouts": {
                label: _metric_distribution(metrics)
                for label, metrics in fold_horizon_metrics.items()
                if metrics
            },
            "baseline_horizon_rollouts": {
                label: _metric_distribution(metrics)
                for label, metrics in fold_baseline_metrics.items()
                if metrics
            },
        },
        "per_fold": per_fold,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {summary_path}")
    return summary


def _aggregate_equal(metrics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return aggregate_rollout_metrics(list(metrics), weighting="equal")


def _aggregate_by_label(
    by_label: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    return {
        label: _aggregate_equal(metrics)
        for label, metrics in by_label.items()
        if metrics
    }


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


def build_parser(
    *,
    label: str,
    description: str,
    steps: int,
    learning_rate: float,
) -> argparse.ArgumentParser:
    """Build the shared front end for one leave-one-label-out command."""

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("trajectory", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hold-out", default=label, help="trajectory label to fold on")
    parser.add_argument("--training-horizons", type=_horizons, default=(0.1, 0.5, 2.0))
    parser.add_argument(
        "--evaluation-horizons", type=_horizons, default=(0.1, 0.5, 1.0, 2.0)
    )
    parser.add_argument("--steps", type=int, default=steps)
    parser.add_argument("--learning-rate", type=float, default=learning_rate)
    parser.add_argument("--endpoint-weight", type=float, default=3.0)
    parser.add_argument("--stability-regularization", type=float, default=0.01)
    parser.add_argument("--run-no-lag-ablation", action="store_true")
    parser.add_argument(
        "--model-class",
        choices=("structured", "structured_residual"),
        default="structured",
    )
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    return parser


def _run(parser: argparse.ArgumentParser, argv: Sequence[str] | None) -> None:
    args = parser.parse_args(argv)
    evaluate_holdout(
        args.trajectory,
        hold_out=args.hold_out,
        spec=FitSpec(
            horizons_s=args.training_horizons,
            evaluation_horizons_s=args.evaluation_horizons,
            steps=args.steps,
            learning_rate=args.learning_rate,
            model_class=args.model_class,
            ablations=("no_lag",) if args.run_no_lag_ablation else (),
            loss=LossPolicy(
                endpoint_weight=args.endpoint_weight,
                stability_regularization=args.stability_regularization,
            ),
        ),
        output_dir=args.output_dir,
        resume=args.resume,
    )


def profile_main(argv: Sequence[str] | None = None) -> None:
    """Run leave-one-maneuver-profile-out dynamics identification."""

    _run(
        build_parser(
            label="profile",
            description=profile_main.__doc__ or "",
            steps=2500,
            learning_rate=0.01,
        ),
        argv,
    )


def source_group_main(argv: Sequence[str] | None = None) -> None:
    """Run leave-one-source-group-out dynamics identification."""

    _run(
        build_parser(
            label="source_group",
            description=source_group_main.__doc__ or "",
            steps=400,
            learning_rate=0.02,
        ),
        argv,
    )
