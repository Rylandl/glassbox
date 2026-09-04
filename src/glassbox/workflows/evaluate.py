"""One evaluation, three named scoring policies.

The same question, "how well does this model predict held-out flight", was
answered three different ways by three corpus modules, and their numbers were
quoted side by side as if they meant the same thing. They do not: they differ
in the error statistic, in how often a prediction is initialized, in what they
compare against, and in the floor below which two numbers count as equal.

Those conventions are kept, and named. :data:`PROTOCOLS` maps a name to the
:class:`ScoringPolicy` that spells the convention out, :func:`evaluate` applies
one, and every report it writes carries the protocol, the baseline, the stride,
the floors and whether the flights were an independent holdout, so two numbers
produced under different conventions can be told apart on sight.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from glassbox.core.data import Trajectory, duration_to_steps, load_trajectory_npz
from glassbox.core.dynamics import ModelParams
from glassbox.core.metrics import (
    METRIC_FLOORS,
    NEGLIGIBLE_METRIC_FLOORS,
    ROLLOUT_METRICS,
    SEQUENTIAL_LOG_MEAN,
    VECTORIZED_LOG_MEAN,
    aggregate_rollout_metrics,
    kinematic_persistence_windowed_metrics,
    persistence_score,
    predict_windows,
    rollout_metrics,
)
from glassbox.core.model_io import load_dynamics_model, parameter_dict

WINDOWED_RMSE = "windowed_rmse"
ROLLING_STEP_MAE = "rolling_step_mae"
KINEMATIC_PERSISTENCE = "kinematic_persistence"
HOLD_STATE = "hold_state"
ONE_HORIZON = "one_horizon"
ONE_SAMPLE = "one_sample"

_MAE_METRICS = (
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


@dataclass(frozen=True)
class ScoringPolicy:
    """One named convention for scoring predictions against held-out flight.

    ``statistic`` is either :data:`WINDOWED_RMSE`, the four rigid-body RMSE
    metrics over fixed-horizon windows, or :data:`ROLLING_STEP_MAE`, the mean
    Euclidean error as a function of prediction step. ``stride`` says how often
    a prediction is initialized: :data:`ONE_HORIZON` gives back-to-back windows,
    :data:`ONE_SAMPLE` starts one at every admissible sample. ``baseline`` names
    what the model is scored against, ``floors`` the per-metric value below
    which two numbers count as equal, and ``score_reduction`` which
    floating-point reduction produces the geometric mean, because recorded
    reports pin the one that produced them.
    """

    name: str
    statistic: str
    stride: str
    baseline: str
    aggregation: str
    definition: str
    horizons_s: tuple[float, ...] = ()
    maximum_horizon_steps: int | None = None
    floors: Mapping[str, float] | None = None
    score_reduction: str | None = None
    exact_horizons: bool = False
    published_reference: Mapping[str, Any] | None = None

    def stride_steps(self, horizon_steps: int) -> int:
        """Return the sample stride this policy uses at one horizon."""

        return horizon_steps if self.stride == ONE_HORIZON else 1

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("published_reference")
        payload["horizons_s"] = list(self.horizons_s)
        payload["floors"] = None if self.floors is None else dict(self.floors)
        return payload


PROTOCOLS: dict[str, ScoringPolicy] = {
    "windowed": ScoringPolicy(
        name="windowed",
        statistic=WINDOWED_RMSE,
        stride=ONE_HORIZON,
        baseline=KINEMATIC_PERSISTENCE,
        aggregation="equal_flight",
        horizons_s=(0.1, 0.5, 1.0, 2.0),
        floors=METRIC_FLOORS,
        score_reduction=SEQUENTIAL_LOG_MEAN,
        definition=(
            "back-to-back fixed-horizon rollouts scored as rigid-body RMSE, "
            "against a constant-world-velocity and constant-body-rate baseline, "
            "with the physically meaningful equality floors"
        ),
    ),
    "x8": ScoringPolicy(
        name="x8",
        statistic=WINDOWED_RMSE,
        stride=ONE_SAMPLE,
        baseline=KINEMATIC_PERSISTENCE,
        aggregation="equal_flight",
        horizons_s=(0.1, 0.5, 1.0, 2.0),
        floors=NEGLIGIBLE_METRIC_FLOORS,
        score_reduction=VECTORIZED_LOG_MEAN,
        exact_horizons=True,
        definition=(
            "the NTNU Skywalker X8 campaign convention: a rollout initialized "
            "at every admissible sample, scored as rigid-body RMSE against the "
            "same kinematic baseline, floored only against log(0) because both "
            "sides are real flight errors"
        ),
    ),
    "nanodrone": ScoringPolicy(
        name="nanodrone",
        statistic=ROLLING_STEP_MAE,
        stride=ONE_SAMPLE,
        baseline=HOLD_STATE,
        aggregation="prediction_window",
        maximum_horizon_steps=50,
        published_reference=PUBLISHED_PHYS_PLUS_RES,
        definition=(
            "the IDSIA Nano-drone benchmark convention: mean Euclidean error "
            "at every prediction step from one to the maximum horizon, over "
            "rollouts started at every sample, against holding the measured "
            "initial state, and against the published Phys+Res table"
        ),
    ),
}


def policy_for(protocol: str) -> ScoringPolicy:
    """Return the named scoring policy, or say which names exist."""

    try:
        return PROTOCOLS[protocol]
    except KeyError:
        raise ValueError(
            f"unknown evaluation protocol {protocol!r}; "
            f"choose one of {', '.join(sorted(PROTOCOLS))}"
        ) from None


def horizon_steps_for_duration(trajectory: Trajectory, horizon_s: float) -> int:
    """Return the integer step count for one horizon at a trajectory's rate."""

    steps = duration_to_steps(horizon_s, trajectory.nominal_dt_s)
    if not np.isclose(steps * trajectory.nominal_dt_s, horizon_s, atol=1e-9, rtol=0.0):
        raise ValueError(
            f"horizon {horizon_s:g}s is not representable at the sample rate"
        )
    return steps


def aggregate_horizon_rollouts(
    per_trajectory: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Average each horizon's rollout metrics across flights, equally weighted."""

    labels = tuple(per_trajectory[0]["horizon_rollouts"])
    return {
        label: aggregate_rollout_metrics(
            [item["horizon_rollouts"][label] for item in per_trajectory],
            weighting="equal",
        )
        for label in labels
    }


def score_against_baseline(
    candidate: Mapping[str, Mapping[str, Any]],
    baseline: Mapping[str, Mapping[str, Any]],
    *,
    protocol: str = "windowed",
    horizons_s: Sequence[float] | None = None,
) -> float:
    """Score one horizon table against another under a policy's reduction.

    ``horizons_s`` restricts the score to a subset of the evaluated horizons;
    by default every horizon the candidate carries contributes. Values below
    one favor the candidate.
    """

    policy = policy_for(protocol)
    if policy.score_reduction is None:
        raise ValueError(f"the {policy.name} protocol does not score a horizon table")
    return persistence_score(
        candidate,
        baseline,
        horizons=horizons_s,
        floors=policy.floors,
        metrics=ROLLOUT_METRICS,
        aggregation=policy.score_reduction,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_model(
    belief_or_params: Any,
) -> tuple[ModelParams, dict[str, Any] | None]:
    """Return the parameters to roll out and, for a saved belief, its record."""

    if isinstance(belief_or_params, (str, Path)):
        path = Path(belief_or_params)
        params, payload = load_dynamics_model(path)
        return params, {
            "path": str(path),
            "sha256": _sha256(path),
            "model_type": payload["model_type"],
            "model_family": payload["model_family"],
            "parameters": parameter_dict(params),
            "input_spec": payload["input_spec"],
            "provenance": payload.get("provenance", {}),
        }
    params = getattr(getattr(belief_or_params, "model", None), "params", None)
    if params is not None:
        return params, None
    return belief_or_params, None


def _resolve_trajectories(
    trajectories: Sequence[Trajectory | str | Path],
) -> tuple[list[str], list[Trajectory]]:
    if not trajectories:
        raise ValueError("at least one trajectory is required")
    labels: list[str] = []
    resolved: list[Trajectory] = []
    for index, item in enumerate(trajectories):
        if isinstance(item, Trajectory):
            labels.append(str(item.provenance.get("path", f"trajectory_{index}")))
            resolved.append(item)
        else:
            path = Path(item).resolve()
            labels.append(str(path))
            resolved.append(load_trajectory_npz(path))
    return labels, resolved


def _windowed_arm(
    params: ModelParams,
    labels: Sequence[str],
    trajectories: Sequence[Trajectory],
    *,
    policy: ScoringPolicy,
    horizon_steps: Mapping[str, int],
) -> dict[str, Any]:
    per_trajectory = [
        {
            "path": path,
            "horizon_rollouts": {
                label: rollout_metrics(
                    predict_windows(
                        params,
                        trajectory,
                        horizon_steps=steps,
                        stride=policy.stride_steps(steps),
                    )
                )
                for label, steps in horizon_steps.items()
            },
        }
        for path, trajectory in zip(labels, trajectories)
    ]
    return {
        "aggregate": {"horizon_rollouts": aggregate_horizon_rollouts(per_trajectory)},
        "per_trajectory": per_trajectory,
    }


def _windowed_baseline_arm(
    labels: Sequence[str],
    trajectories: Sequence[Trajectory],
    *,
    policy: ScoringPolicy,
    horizon_steps: Mapping[str, int],
) -> dict[str, Any]:
    per_trajectory = [
        {
            "path": path,
            "horizon_rollouts": {
                label: kinematic_persistence_windowed_metrics(
                    trajectory,
                    horizon_steps=steps,
                    stride=policy.stride_steps(steps),
                )
                for label, steps in horizon_steps.items()
            },
        }
        for path, trajectory in zip(labels, trajectories)
    ]
    return {
        "aggregate": {"horizon_rollouts": aggregate_horizon_rollouts(per_trajectory)},
        "per_trajectory": per_trajectory,
    }


def baseline_horizon_rollouts(
    trajectories: Sequence[Trajectory | str | Path],
    *,
    protocol: str = "windowed",
    horizons_s: Sequence[float] | None = None,
) -> dict[str, dict[str, Any]]:
    """Return one protocol's baseline horizon table over a set of flights.

    Callers that already hold a model's held-out metrics, such as the
    comparison of two fit reports over one split, need the baseline half of the
    policy without repeating the rollouts.
    """

    policy = policy_for(protocol)
    if policy.baseline != KINEMATIC_PERSISTENCE:
        raise ValueError(f"the {policy.name} protocol has no horizon baseline")
    labels, resolved = _resolve_trajectories(trajectories)
    horizon_steps, _ = _horizon_steps(resolved, policy, horizons_s)
    arm = _windowed_baseline_arm(
        labels, resolved, policy=policy, horizon_steps=horizon_steps
    )
    return arm["aggregate"]["horizon_rollouts"]


def _horizon_steps(
    trajectories: Sequence[Trajectory],
    policy: ScoringPolicy,
    horizons_s: Sequence[float] | None,
) -> tuple[dict[str, int], dict[str, dict[str, float | int]]]:
    """Resolve requested horizons to step counts, and record what they became.

    A policy with ``exact_horizons`` rejects a horizon the sample rate cannot
    represent; the others round to the nearest step and report the horizon they
    actually scored, which is how a 0.5-second request becomes 0.4 seconds at
    5 Hz.
    """

    requested = tuple(policy.horizons_s if horizons_s is None else horizons_s)
    if not requested or any(seconds <= 0.0 for seconds in requested):
        raise ValueError("evaluation horizons must be positive")
    steps: dict[str, int] = {}
    effective: dict[str, dict[str, float | int]] = {}
    for seconds in requested:
        counts = {
            horizon_steps_for_duration(trajectory, seconds)
            if policy.exact_horizons
            else duration_to_steps(seconds, trajectory.nominal_dt_s)
            for trajectory in trajectories
        }
        if len(counts) != 1:
            raise ValueError("evaluated flights must share one sample interval")
        label = f"{seconds:g}s"
        steps[label] = counts.pop()
        effective[label] = {
            "requested_s": float(seconds),
            "steps": steps[label],
            "effective_s": steps[label] * trajectories[0].nominal_dt_s,
        }
    return steps, effective


def _step_errors(predicted: np.ndarray, target: np.ndarray) -> dict[str, np.ndarray]:
    """Return the mean Euclidean error at each prediction step over windows."""

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
        "angular_velocity_mae_rad_s": np.mean(angular_velocity_error, axis=0),
    }


def _step_summary(
    values: Mapping[str, np.ndarray],
    *,
    dt_s: float,
    window_count: int,
) -> dict[str, Any]:
    horizon_count = len(values[_MAE_METRICS[0]])
    for name in _MAE_METRICS:
        if values[name].shape != (horizon_count,):
            raise ValueError("benchmark metric horizon lengths must match")
    selected_steps = tuple(
        dict.fromkeys(step for step in (1, 10, horizon_count) if step <= horizon_count)
    )
    return {
        "window_count": window_count,
        "horizon_steps": list(range(1, horizon_count + 1)),
        "horizon_time_s": [step * dt_s for step in range(1, horizon_count + 1)],
        "per_horizon": {name: values[name].tolist() for name in _MAE_METRICS},
        "selected_horizons": {
            str(step): {
                "time_s": step * dt_s,
                **{name: float(values[name][step - 1]) for name in _MAE_METRICS},
            }
            for step in selected_steps
        },
        "cumulative_simulation_error": {
            name: float(np.sum(values[name])) for name in _MAE_METRICS
        },
    }


def _aggregate_step_flights(
    flight_metrics: Sequence[tuple[int, Mapping[str, np.ndarray]]],
    *,
    dt_s: float,
) -> dict[str, Any]:
    total_windows = sum(count for count, _ in flight_metrics)
    values = {
        name: sum(metrics[name] * count for count, metrics in flight_metrics)
        / total_windows
        for name in _MAE_METRICS
    }
    summary = _step_summary(values, dt_s=dt_s, window_count=total_windows)
    summary["weighting"] = "prediction_window"
    return summary


def _hold_state_predictions(target: np.ndarray) -> np.ndarray:
    return np.repeat(target[:, 0:1, :], target.shape[1], axis=1)


def _step_ratio_summary(
    model: Mapping[str, Any], baseline: Mapping[str, Any]
) -> dict[str, Any]:
    selected = {}
    for step, model_values in model["selected_horizons"].items():
        baseline_values = baseline["selected_horizons"][step]
        selected[step] = {
            name: (
                float(baseline_values[name]) / float(model_values[name])
                if float(model_values[name]) > 0.0
                else None
            )
            for name in _MAE_METRICS
        }
    return {
        "definition": "baseline_error / model_error; values above one favor model",
        "selected_horizons": selected,
        "cumulative_simulation_error": {
            name: (
                float(baseline["cumulative_simulation_error"][name])
                / float(model["cumulative_simulation_error"][name])
                if float(model["cumulative_simulation_error"][name]) > 0.0
                else None
            )
            for name in _MAE_METRICS
        },
    }


def _published_reference_comparison(
    model: Mapping[str, Any], reference: Mapping[str, Any]
) -> dict[str, Any] | None:
    """Compare a complete 50-step result with the published Phys+Res table."""

    if len(model["horizon_steps"]) != 50:
        return None
    selected_ratios = {
        step: {
            name: float(model["selected_horizons"][step][name])
            / float(reference_values[name])
            for name in _MAE_METRICS
        }
        for step, reference_values in reference["selected_horizons"].items()
    }
    cumulative_ratios = {
        name: float(model["cumulative_simulation_error"][name])
        / float(reference["cumulative_simulation_error"][name])
        for name in _MAE_METRICS
    }
    all_cumulative_ratios = list(cumulative_ratios.values())
    return {
        "definition": "Glassbox error / published Phys+Res error; below one wins",
        "comparability": "near_comparable_boundary_safe_vs_concatenated_upstream",
        "published_phys_plus_res": dict(reference),
        "selected_horizon_ratio": selected_ratios,
        "cumulative_ratio": cumulative_ratios,
        "cumulative_equal_metric_geometric_ratio": float(
            np.exp(np.mean(np.log(np.asarray(all_cumulative_ratios))))
        ),
        "beats_every_published_cumulative_metric": all(
            ratio <= 1.0 for ratio in all_cumulative_ratios
        ),
        "beats_every_published_50_step_metric": all(
            ratio <= 1.0 for ratio in selected_ratios["50"].values()
        ),
    }


def _rolling_step_report(
    params: ModelParams,
    labels: Sequence[str],
    trajectories: Sequence[Trajectory],
    *,
    policy: ScoringPolicy,
    maximum_horizon_steps: int,
) -> dict[str, Any]:
    dt_s = trajectories[0].nominal_dt_s
    model_flights: list[tuple[int, dict[str, np.ndarray]]] = []
    baseline_flights: list[tuple[int, dict[str, np.ndarray]]] = []
    per_trajectory: list[dict[str, Any]] = []
    for path, trajectory in zip(labels, trajectories):
        if maximum_horizon_steps > len(trajectory.controls):
            raise ValueError("benchmark horizon exceeds a trajectory length")
        prediction = predict_windows(
            params,
            trajectory,
            horizon_steps=maximum_horizon_steps,
            stride=1,
        )
        if not np.isclose(prediction.dt_s, dt_s, atol=1e-7, rtol=0.0):
            raise ValueError("evaluated flights must share one sample interval")
        model_values = _step_errors(prediction.predicted, prediction.target)
        baseline_values = _step_errors(
            _hold_state_predictions(prediction.target), prediction.target
        )
        window_count = len(prediction.predicted)
        model_flights.append((window_count, model_values))
        baseline_flights.append((window_count, baseline_values))
        per_trajectory.append(
            {
                "path": path,
                "profile": trajectory.labels.get("profile"),
                "replicate": trajectory.labels.get("replicate"),
                "model": _step_summary(
                    model_values, dt_s=dt_s, window_count=window_count
                ),
                "baseline": _step_summary(
                    baseline_values, dt_s=dt_s, window_count=window_count
                ),
            }
        )
    model = _aggregate_step_flights(model_flights, dt_s=dt_s)
    baseline = _aggregate_step_flights(baseline_flights, dt_s=dt_s)
    report: dict[str, Any] = {
        "model": model,
        "baseline_metrics": baseline,
        "model_vs_baseline": _step_ratio_summary(model, baseline),
        "per_trajectory": per_trajectory,
        "score_vs_baseline": None,
        "sample_rate_hz": 1.0 / dt_s,
        "maximum_horizon_steps": maximum_horizon_steps,
        "maximum_horizon_s": maximum_horizon_steps * dt_s,
    }
    if policy.published_reference is not None:
        comparison = _published_reference_comparison(model, policy.published_reference)
        if comparison is not None:
            report["published_reference_comparison"] = comparison
    return report


def evaluate(
    belief_or_params: Any,
    trajectories: Sequence[Trajectory | str | Path],
    *,
    protocol: str = "windowed",
    horizons_s: Sequence[float] | None = None,
    maximum_horizon_steps: int | None = None,
    independent_holdout: bool = True,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Score one model on held-out flight under a named scoring policy.

    ``belief_or_params`` is a :class:`~glassbox.DynamicsBelief`, bare model
    parameters, or a path to a saved belief, in which case the report records
    which artifact produced its numbers. ``trajectories`` are canonical
    trajectories or paths to them.

    ``independent_holdout`` is the caller's declaration that these flights were
    withheld from the fit. Passing ``False`` marks the report as a same-flight
    characterization: useful for airframe characterization, but its
    ``can_promote_model`` is false, because a model cannot be promoted on
    evidence it was fitted to.
    """

    policy = policy_for(protocol)
    params, artifact = _resolve_model(belief_or_params)
    labels, resolved = _resolve_trajectories(trajectories)

    body: dict[str, Any]
    effective: dict[str, Any] | None = None
    if policy.statistic == ROLLING_STEP_MAE:
        if horizons_s is not None:
            raise ValueError(
                f"the {policy.name} protocol scores prediction steps, not "
                "horizons in seconds; use maximum_horizon_steps"
            )
        steps = (
            policy.maximum_horizon_steps
            if maximum_horizon_steps is None
            else maximum_horizon_steps
        )
        if steps is None or steps < 1:
            raise ValueError("maximum_horizon_steps must be positive")
        body = _rolling_step_report(
            params,
            labels,
            resolved,
            policy=policy,
            maximum_horizon_steps=steps,
        )
    else:
        if maximum_horizon_steps is not None:
            raise ValueError(
                f"the {policy.name} protocol scores horizons in seconds, not "
                "prediction steps"
            )
        horizon_steps, effective = _horizon_steps(resolved, policy, horizons_s)
        model = _windowed_arm(
            params, labels, resolved, policy=policy, horizon_steps=horizon_steps
        )
        baseline = _windowed_baseline_arm(
            labels, resolved, policy=policy, horizon_steps=horizon_steps
        )
        body = {
            "model": model["aggregate"],
            "baseline_metrics": baseline["aggregate"],
            "per_trajectory": model["per_trajectory"],
            "baseline_per_trajectory": baseline["per_trajectory"],
            "score_vs_baseline": score_against_baseline(
                model["aggregate"]["horizon_rollouts"],
                baseline["aggregate"]["horizon_rollouts"],
                protocol=protocol,
            ),
        }

    report: dict[str, Any] = {
        "format_version": 1,
        "protocol": policy.name,
        "baseline": policy.baseline,
        "stride": policy.stride,
        "floors": None if policy.floors is None else dict(policy.floors),
        "independent_holdout": independent_holdout,
        "can_promote_model": independent_holdout,
        "scoring": policy.to_dict(),
        "dataset": {
            "trajectory_count": len(resolved),
            "duration_s": float(
                sum(
                    trajectory.time_s[-1] - trajectory.time_s[0]
                    for trajectory in resolved
                )
            ),
            "trajectory_spec": resolved[0].spec.to_dict(),
            "flight_boundaries_crossed": False,
            "state_correction_during_rollout": False,
            "trajectories": list(labels),
        },
        **body,
    }
    if effective is not None:
        report["scoring"]["requested_and_effective_horizons"] = effective
    if artifact is not None:
        report["model_artifact"] = artifact
    if report_path is not None:
        save_report(report, report_path)
    return report


def evaluate_models(
    models: Mapping[str, Any],
    trajectories: Sequence[Trajectory | str | Path],
    *,
    protocol: str = "windowed",
    horizons_s: Sequence[float] | None = None,
    independent_holdout: bool = True,
    expected_spec: Mapping[str, Any] | None = None,
    benchmark: Mapping[str, Any] | None = None,
    split: str | None = None,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Score several named models on one held-out set under one policy.

    Every model is scored by :func:`evaluate` against the same flights and the
    same baseline, so the models can be compared to the baseline and to each
    other: ``comparisons`` holds the ordered pair scores. ``expected_spec``, if
    given, is the trajectory spec every model must declare, which is how a
    model fitted to a different vehicle is caught before it is scored.
    """

    policy = policy_for(protocol)
    if not models:
        raise ValueError("at least one named model is required")
    labels, resolved = _resolve_trajectories(trajectories)
    scored_models: dict[str, Any] = {}
    baseline: dict[str, Any] | None = None
    for name, model in models.items():
        scored = evaluate(
            model,
            resolved,
            protocol=protocol,
            horizons_s=horizons_s,
            independent_holdout=independent_holdout,
        )
        artifact = scored.get("model_artifact")
        if expected_spec is not None and artifact is not None:
            if artifact["input_spec"] != dict(expected_spec):
                raise ValueError(f"model input spec does not match the corpus: {model}")
        baseline = scored["baseline_metrics"]
        scored_models[name] = {
            "path": None if artifact is None else artifact["path"],
            "model_type": None if artifact is None else artifact["model_type"],
            "aggregate": scored["model"],
            "per_trajectory": scored["per_trajectory"],
            f"score_vs_{policy.baseline}": scored["score_vs_baseline"],
            "score_horizons_s": scored["scoring"]["horizons_s"],
        }

    report: dict[str, Any] = {
        "format_version": 1,
        "protocol": policy.name,
        "baseline": policy.baseline,
        "stride": policy.stride,
        "independent_holdout": independent_holdout,
        "can_promote_model": independent_holdout,
        "dataset": {
            "validation_trajectory_count": len(resolved),
            "validation_duration_s": float(
                sum(trajectory.time_s[-1] for trajectory in resolved)
            ),
            "trajectory_spec": (
                resolved[0].spec.to_dict()
                if expected_spec is None
                else dict(expected_spec)
            ),
            "trajectories": list(labels),
        },
        policy.baseline: baseline,
        "models": scored_models,
        "comparisons": {
            f"{candidate}_vs_{reference}": {
                "ratio_definition": (
                    "candidate/reference geometric mean over four state metrics "
                    "and every horizon; values below one favor the candidate"
                ),
                "score": score_against_baseline(
                    scored_models[candidate]["aggregate"]["horizon_rollouts"],
                    scored_models[reference]["aggregate"]["horizon_rollouts"],
                    protocol=protocol,
                ),
            }
            for candidate in scored_models
            for reference in scored_models
            if candidate != reference
        },
    }
    if benchmark is not None:
        report["benchmark"] = dict(benchmark)
    if split is not None:
        report["split"] = split
    if report_path is not None:
        save_report(report, report_path)
    return report


def evaluate_fit_reports(
    fit_reports: Mapping[str, str | Path],
    *,
    protocol: str = "windowed",
    horizons_s: Sequence[float] | None = None,
    score_horizons_s: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Score models whose held-out metrics one fit already computed.

    Each named fit report supplies its own validation flights and the horizon
    table the fit measured on them; this adds the policy's baseline over those
    same flights and scores every model against it. The reports must name the
    same split, because a score is only comparable within one.
    """

    policy = policy_for(protocol)
    paths = {name: Path(path) for name, path in fit_reports.items()}
    if not paths:
        raise ValueError("at least one fit report is required")
    reports = {name: json.loads(path.read_text()) for name, path in paths.items()}

    splits = {
        name: [item["path"] for item in report["split"]["validation_flights"]]
        for name, report in reports.items()
    }
    reference_split = next(iter(splits.values()))
    if any(split != reference_split for split in splits.values()):
        raise ValueError("fit reports must evaluate the same validation flights")
    training = {
        name: [item["path"] for item in report["split"]["training_flights"]]
        for name, report in reports.items()
    }
    if any(item != next(iter(training.values())) for item in training.values()):
        raise ValueError("fit reports must fit the same training flights")

    anchor = next(iter(paths.values()))
    validation_paths = [
        _resolve_relative(value, anchor=anchor) for value in reference_split
    ]
    labels, resolved = _resolve_trajectories(validation_paths)
    horizon_steps, effective = _horizon_steps(resolved, policy, horizons_s)
    baseline = _windowed_baseline_arm(
        labels, resolved, policy=policy, horizon_steps=horizon_steps
    )["aggregate"]["horizon_rollouts"]

    independent_holdout = all(
        bool(report["split"]["independent_source_group_holdout"])
        for report in reports.values()
    )
    models: dict[str, Any] = {}
    for name, report in reports.items():
        learned = report["models"]["learned_lag"]
        horizons = learned["validation"]["aggregate"]["horizon_rollouts"]
        missing = [label for label in horizon_steps if label not in horizons]
        if missing:
            raise ValueError(
                f"{name} fit report is missing horizon(s): {', '.join(missing)}"
            )
        models[name] = {
            "fit_report": {
                "path": str(paths[name].resolve()),
                "size_bytes": paths[name].stat().st_size,
                "sha256": _sha256(paths[name]),
            },
            "fit": learned["fit"],
            "model_class": report["configuration"]["model_class"],
            "aggregate_horizon_rollouts": horizons,
            "aggregate_full_rollout": learned["validation"]["aggregate"][
                "full_rollout"
            ],
            "score_vs_baseline": score_against_baseline(
                horizons,
                baseline,
                protocol=protocol,
                horizons_s=score_horizons_s,
            ),
            "score_horizons_s": (
                None if score_horizons_s is None else list(score_horizons_s)
            ),
        }
    scoring = policy.to_dict()
    scoring["requested_and_effective_horizons"] = effective
    return {
        "format_version": 1,
        "protocol": policy.name,
        "baseline": policy.baseline,
        "stride": policy.stride,
        "floors": None if policy.floors is None else dict(policy.floors),
        "independent_holdout": independent_holdout,
        "can_promote_model": independent_holdout,
        "scoring": scoring,
        "dataset": {
            "trajectory_count": len(resolved),
            "duration_s": float(
                sum(
                    trajectory.time_s[-1] - trajectory.time_s[0]
                    for trajectory in resolved
                )
            ),
            "training_flight_count": len(next(iter(training.values()))),
            "trajectories": list(labels),
        },
        "baseline_metrics": {"horizon_rollouts": baseline},
        "models": models,
        "selected_model": min(
            models, key=lambda name: models[name]["score_vs_baseline"]
        ),
    }


def _resolve_relative(value: str, *, anchor: Path) -> Path:
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    anchored = anchor.parent / path
    if anchored.exists():
        return anchored
    raise FileNotFoundError(path)


def save_report(report: Mapping[str, Any], path: str | Path) -> None:
    """Write one evaluation report as deterministic JSON."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
