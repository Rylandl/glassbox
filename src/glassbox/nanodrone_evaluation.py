"""Published rolling-horizon protocol for the IDSIA Nano-drone benchmark."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from glassbox.data import Trajectory, load_trajectory_npz
from glassbox.dynamics import ModelParams, with_constant_angular_rate
from glassbox.evaluation import parameter_dict, windowed_rollout_predictions
from glassbox.model_io import load_dynamics_model
from glassbox.nanodrone_benchmark import (
    BENCHMARK_COMMIT,
    BENCHMARK_DOI,
    BENCHMARK_REPOSITORY,
    nanodrone_trajectory_spec,
)


BENCHMARK_MAX_HORIZON_STEPS = 50
_METRIC_NAMES = (
    "position_mae_m",
    "velocity_mae_m_s",
    "attitude_mae_rad",
    "angular_velocity_mae_rad_s",
)

# Table 6 of Busetto et al., Control Engineering Practice 172 (2026), 106871.
# The upstream implementation concatenates the three Melon recordings before
# shifting targets, whereas Glassbox keeps windows inside recording boundaries.
PUBLISHED_PHYS_PLUS_RES = {
    "selected_horizons": {
        "1": {
            "position_mae_m": 0.0016,
            "velocity_mae_m_s": 0.0092,
            "attitude_mae_rad": 0.0027,
            "angular_velocity_mae_rad_s": 0.0912,
        },
        "10": {
            "position_mae_m": 0.0166,
            "velocity_mae_m_s": 0.0613,
            "attitude_mae_rad": 0.0376,
            "angular_velocity_mae_rad_s": 0.4880,
        },
        "50": {
            "position_mae_m": 0.1119,
            "velocity_mae_m_s": 0.5556,
            "attitude_mae_rad": 0.2306,
            "angular_velocity_mae_rad_s": 0.5979,
        },
    },
    "cumulative_simulation_error": {
        "position_mae_m": 2.3625,
        "velocity_mae_m_s": 10.4033,
        "attitude_mae_rad": 6.1534,
        "angular_velocity_mae_rad_s": 28.9873,
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _per_horizon_errors(
    predicted: np.ndarray,
    target: np.ndarray,
) -> dict[str, np.ndarray]:
    if predicted.shape != target.shape:
        raise ValueError("predicted and target windows must have identical shapes")
    if predicted.ndim != 3 or predicted.shape[-1] != 13:
        raise ValueError("benchmark state windows must have shape (window, step, 13)")
    if predicted.shape[1] < 2:
        raise ValueError("benchmark state windows need at least one predicted step")

    predicted_future = predicted[:, 1:, :]
    target_future = target[:, 1:, :]
    position_error = np.linalg.norm(
        predicted_future[..., 0:3] - target_future[..., 0:3], axis=-1
    )
    velocity_error = np.linalg.norm(
        predicted_future[..., 3:6] - target_future[..., 3:6], axis=-1
    )
    angular_velocity_error = np.linalg.norm(
        predicted_future[..., 10:13] - target_future[..., 10:13], axis=-1
    )

    predicted_quaternion = predicted_future[..., 6:10] / np.linalg.norm(
        predicted_future[..., 6:10], axis=-1, keepdims=True
    )
    target_quaternion = target_future[..., 6:10] / np.linalg.norm(
        target_future[..., 6:10], axis=-1, keepdims=True
    )
    quaternion_dot = np.clip(
        np.abs(np.sum(predicted_quaternion * target_quaternion, axis=-1)),
        0.0,
        1.0,
    )
    attitude_error = 2.0 * np.arccos(quaternion_dot)
    return {
        "position_mae_m": np.mean(position_error, axis=0),
        "velocity_mae_m_s": np.mean(velocity_error, axis=0),
        "attitude_mae_rad": np.mean(attitude_error, axis=0),
        "angular_velocity_mae_rad_s": np.mean(
            angular_velocity_error, axis=0
        ),
    }


def _metric_summary(
    values: dict[str, np.ndarray],
    *,
    dt_s: float,
    window_count: int,
) -> dict[str, Any]:
    horizon_count = len(values[_METRIC_NAMES[0]])
    for name in _METRIC_NAMES:
        if values[name].shape != (horizon_count,):
            raise ValueError("benchmark metric horizon lengths must match")
    selected_steps = tuple(
        dict.fromkeys(
            step for step in (1, 10, horizon_count) if step <= horizon_count
        )
    )
    return {
        "window_count": window_count,
        "horizon_steps": list(range(1, horizon_count + 1)),
        "horizon_time_s": [
            step * dt_s for step in range(1, horizon_count + 1)
        ],
        "per_horizon": {
            name: values[name].tolist() for name in _METRIC_NAMES
        },
        "selected_horizons": {
            str(step): {
                "time_s": step * dt_s,
                **{
                    name: float(values[name][step - 1])
                    for name in _METRIC_NAMES
                },
            }
            for step in selected_steps
        },
        "cumulative_simulation_error": {
            name: float(np.sum(values[name])) for name in _METRIC_NAMES
        },
    }


def _aggregate_flights(
    flight_metrics: list[tuple[int, dict[str, np.ndarray]]],
    *,
    dt_s: float,
) -> dict[str, Any]:
    total_windows = sum(count for count, _ in flight_metrics)
    values = {
        name: sum(
            metrics[name] * count for count, metrics in flight_metrics
        )
        / total_windows
        for name in _METRIC_NAMES
    }
    summary = _metric_summary(
        values, dt_s=dt_s, window_count=total_windows
    )
    summary["weighting"] = "prediction_window"
    return summary


def _naive_predictions(target: np.ndarray) -> np.ndarray:
    return np.repeat(target[:, 0:1, :], target.shape[1], axis=1)


def _ratio_summary(
    model_summary: dict[str, Any], naive_summary: dict[str, Any]
) -> dict[str, Any]:
    selected = {}
    for step, model_values in model_summary["selected_horizons"].items():
        naive_values = naive_summary["selected_horizons"][step]
        selected[step] = {
            name: (
                float(naive_values[name]) / float(model_values[name])
                if float(model_values[name]) > 0.0
                else None
            )
            for name in _METRIC_NAMES
        }
    return {
        "definition": "naive_error / model_error; values above one favor model",
        "selected_horizons": selected,
        "cumulative_simulation_error": {
            name: (
                float(naive_summary["cumulative_simulation_error"][name])
                / float(model_summary["cumulative_simulation_error"][name])
                if float(model_summary["cumulative_simulation_error"][name]) > 0.0
                else None
            )
            for name in _METRIC_NAMES
        },
    }


def _published_reference_comparison(
    model_summary: dict[str, Any],
) -> dict[str, Any] | None:
    """Compare a complete 50-step result with the paper's Phys+Res table."""

    if len(model_summary["horizon_steps"]) != 50:
        return None
    selected_ratios = {
        step: {
            name: (
                float(model_summary["selected_horizons"][step][name])
                / float(reference_values[name])
            )
            for name in _METRIC_NAMES
        }
        for step, reference_values in PUBLISHED_PHYS_PLUS_RES[
            "selected_horizons"
        ].items()
    }
    cumulative_ratios = {
        name: (
            float(model_summary["cumulative_simulation_error"][name])
            / float(
                PUBLISHED_PHYS_PLUS_RES["cumulative_simulation_error"][name]
            )
        )
        for name in _METRIC_NAMES
    }
    all_cumulative_ratios = list(cumulative_ratios.values())
    geometric_ratio = float(
        np.exp(np.mean(np.log(np.asarray(all_cumulative_ratios))))
    )
    return {
        "definition": "Glassbox error / published Phys+Res error; below one wins",
        "comparability": "near_comparable_boundary_safe_vs_concatenated_upstream",
        "source": {
            "paper": "Nonlinear system identification for a nano-drone benchmark",
            "doi": BENCHMARK_DOI,
            "table": 6,
        },
        "published_phys_plus_res": PUBLISHED_PHYS_PLUS_RES,
        "selected_horizon_ratio": selected_ratios,
        "cumulative_ratio": cumulative_ratios,
        "cumulative_equal_metric_geometric_ratio": geometric_ratio,
        "beats_every_published_cumulative_metric": all(
            ratio <= 1.0 for ratio in all_cumulative_ratios
        ),
        "beats_every_published_50_step_metric": all(
            ratio <= 1.0 for ratio in selected_ratios["50"].values()
        ),
    }


def evaluate_nanodrone_benchmark(
    params: ModelParams,
    trajectories: Sequence[Trajectory],
    *,
    max_horizon_steps: int = BENCHMARK_MAX_HORIZON_STEPS,
) -> dict[str, Any]:
    """Evaluate a model on all rolling starts in official test trajectories."""

    if not trajectories:
        raise ValueError("at least one benchmark test trajectory is required")
    if max_horizon_steps < 1:
        raise ValueError("max_horizon_steps must be positive")

    expected_spec = nanodrone_trajectory_spec()
    dt_s = trajectories[0].nominal_dt_s
    model_flights: list[tuple[int, dict[str, np.ndarray]]] = []
    constant_rate_flights: list[tuple[int, dict[str, np.ndarray]]] = []
    naive_flights: list[tuple[int, dict[str, np.ndarray]]] = []
    per_flight: list[dict[str, Any]] = []
    constant_rate_params = with_constant_angular_rate(params)
    for trajectory in trajectories:
        if trajectory.spec != expected_spec:
            raise ValueError(
                "trajectory does not use the pinned Nano-drone benchmark spec"
            )
        if trajectory.labels.get("benchmark_split") != "test":
            raise ValueError("benchmark evaluation requires test-split trajectories")
        if trajectory.labels.get("profile") != "melon":
            raise ValueError("benchmark evaluation requires Melon trajectories")
        if not np.isclose(
            trajectory.nominal_dt_s, dt_s, atol=1e-7, rtol=0.0
        ):
            raise ValueError("benchmark trajectories must share one sample interval")
        if max_horizon_steps > len(trajectory.controls):
            raise ValueError("benchmark horizon exceeds a trajectory length")

        predicted, target, prediction_dt_s = windowed_rollout_predictions(
            params,
            trajectory,
            horizon_steps=max_horizon_steps,
            stride_steps=1,
        )
        if not np.isclose(prediction_dt_s, dt_s, atol=1e-7, rtol=0.0):
            raise ValueError("prediction sample interval changed unexpectedly")
        model_values = _per_horizon_errors(predicted, target)
        constant_rate_predicted, _, constant_rate_dt_s = (
            windowed_rollout_predictions(
                constant_rate_params,
                trajectory,
                horizon_steps=max_horizon_steps,
                stride_steps=1,
            )
        )
        if not np.isclose(constant_rate_dt_s, dt_s, atol=1e-7, rtol=0.0):
            raise ValueError("constant-rate diagnostic sample interval changed")
        constant_rate_values = _per_horizon_errors(
            constant_rate_predicted, target
        )
        naive_values = _per_horizon_errors(_naive_predictions(target), target)
        window_count = len(predicted)
        model_flights.append((window_count, model_values))
        constant_rate_flights.append((window_count, constant_rate_values))
        naive_flights.append((window_count, naive_values))
        per_flight.append(
            {
                "profile": trajectory.labels["profile"],
                "replicate": trajectory.labels.get("replicate"),
                "model": _metric_summary(
                    model_values, dt_s=dt_s, window_count=window_count
                ),
                "constant_angular_rate_diagnostic": _metric_summary(
                    constant_rate_values,
                    dt_s=dt_s,
                    window_count=window_count,
                ),
                "naive": _metric_summary(
                    naive_values, dt_s=dt_s, window_count=window_count
                ),
            }
        )

    model_summary = _aggregate_flights(model_flights, dt_s=dt_s)
    constant_rate_summary = _aggregate_flights(
        constant_rate_flights, dt_s=dt_s
    )
    naive_summary = _aggregate_flights(naive_flights, dt_s=dt_s)
    report = {
        "format_version": 1,
        "protocol": {
            "name": "idsia_nanodrone_rolling_multi_horizon_v1",
            "benchmark_repository": BENCHMARK_REPOSITORY,
            "benchmark_commit": BENCHMARK_COMMIT,
            "benchmark_doi": BENCHMARK_DOI,
            "test_profile": "melon",
            "test_flight_count": len(trajectories),
            "sample_rate_hz": 1.0 / dt_s,
            "maximum_horizon_steps": max_horizon_steps,
            "maximum_horizon_s": max_horizon_steps * dt_s,
            "start_stride_steps": 1,
            "flight_boundaries_crossed": False,
            "state_correction_during_rollout": False,
            "aggregation": "mean Euclidean error over every prediction window",
            "attitude_error": "shortest quaternion geodesic distance in radians",
            "cumulative_error": "sum of per-horizon MAE from step 1 through maximum",
        },
        "model": model_summary,
        "constant_angular_rate_diagnostic": {
            **constant_rate_summary,
            "definition": (
                "same translational model with zero angular acceleration; "
                "attitude integrates the measured initial body rate"
            ),
        },
        "naive": naive_summary,
        "model_vs_naive": _ratio_summary(model_summary, naive_summary),
        "per_flight": per_flight,
    }
    published_comparison = _published_reference_comparison(model_summary)
    if published_comparison is not None:
        report["published_reference_comparison"] = published_comparison
    return report


def evaluate_nanodrone_model_artifact(
    model_path: str | Path,
    trajectory_paths: Sequence[str | Path],
    *,
    max_horizon_steps: int = BENCHMARK_MAX_HORIZON_STEPS,
) -> dict[str, Any]:
    """Load a saved Glassbox model and evaluate the official test artifacts."""

    source_model_path = Path(model_path)
    params, payload = load_dynamics_model(source_model_path)
    trajectories = [load_trajectory_npz(path) for path in trajectory_paths]
    model_spec = payload["input_spec"]
    for path, trajectory in zip(trajectory_paths, trajectories):
        if trajectory.spec.prediction_spec().to_dict() != model_spec:
            raise ValueError(
                f"model input spec does not match benchmark trajectory {path}"
            )
    report = evaluate_nanodrone_benchmark(
        params,
        trajectories,
        max_horizon_steps=max_horizon_steps,
    )
    report["model_artifact"] = {
        "path": str(source_model_path),
        "sha256": _sha256(source_model_path),
        "model_type": payload["model_type"],
        "model_family": payload["model_family"],
        "parameters": parameter_dict(params),
        "input_spec": model_spec,
        "provenance": payload.get("provenance", {}),
    }
    report["test_artifacts"] = [str(path) for path in trajectory_paths]
    return report


def save_nanodrone_benchmark_report(
    report: dict[str, Any], path: str | Path
) -> None:
    """Write a benchmark report as readable JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n")
