"""Versioned accuracy targets for telemetry-driven dynamics benchmarks."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

CONTRACT_ID = "multirotor_prediction_v1"
REQUIRED_HORIZONS_S = (0.1, 0.5, 1.0, 2.0)
FIXED_WING_CONTRACT_ID = "fixedwing_prediction_development_v1"
FIXED_WING_REQUIRED_HORIZONS_S = (0.5, 1.0, 2.0)
FIXED_WING_MAXIMUM_PERSISTENCE_SCORE = 0.95

_FIXED_WING_HORIZON_THRESHOLDS = {
    0.5: (0.25, 6.0),
    1.0: (0.60, 9.0),
    2.0: (1.20, 13.0),
}

_HORIZON_THRESHOLDS = {
    "ground_truth": {
        0.1: (0.001, 0.25),
        0.5: (0.01, 1.0),
        1.0: (0.05, 2.0),
        2.0: (0.20, 5.0),
        5.0: (0.75, 10.0),
    },
    "estimated": {
        0.1: (0.02, 0.5),
        0.5: (0.08, 2.0),
        1.0: (0.15, 4.0),
        2.0: (0.30, 7.0),
        5.0: (1.00, 12.0),
    },
}

_FULL_FLIGHT_THRESHOLDS = {
    "ground_truth": (0.10, 10.0),
    "estimated": (0.15, 15.0),
}


def _horizon_seconds(label: str) -> float | None:
    if not label.endswith("s"):
        return None
    try:
        return float(label[:-1])
    except ValueError:
        return None


def _threshold_for_horizon(
    thresholds: Mapping[float, tuple[float, float]], label: str
) -> tuple[float, float] | None:
    seconds = _horizon_seconds(label)
    if seconds is None:
        return None
    for threshold_seconds, threshold in thresholds.items():
        if math.isclose(seconds, threshold_seconds, abs_tol=1e-9, rel_tol=0.0):
            return threshold
    return None


def _metric_gate(value: float, maximum: float) -> dict[str, Any]:
    return {
        "value": value,
        "maximum": maximum,
        "margin": maximum - value,
        "passed": math.isfinite(value) and value <= maximum,
    }


def _worst_profile_metric(
    per_profile: Mapping[str, Mapping[str, Any]],
    *,
    section: str,
    metric: str,
    horizon_label: str | None = None,
) -> tuple[str, float]:
    values: list[tuple[str, float]] = []
    for profile, result in per_profile.items():
        metrics = result[section]
        if horizon_label is not None:
            metrics = metrics[horizon_label]
        values.append((profile, float(metrics[metric])))
    return max(values, key=lambda item: item[1])


def evaluate_multirotor_accuracy(
    *,
    state_source: str | None,
    aggregate_full_rollout: Mapping[str, Any],
    aggregate_horizon_rollouts: Mapping[str, Mapping[str, Any]],
    per_profile: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Evaluate macro and worst-profile errors against a versioned contract.

    These are development targets for held-out predictive accuracy, not flight
    safety or certification limits. A horizon passes only when both the
    equal-profile aggregate and every held-out profile meet both error limits.
    """

    base: dict[str, Any] = {
        "contract_id": CONTRACT_ID,
        "platform": "multirotor",
        "state_source": state_source,
        "scope": "held-out logged-input prediction",
        "policy": "aggregate_and_worst_profile_must_pass",
        "required_horizons_s": list(REQUIRED_HORIZONS_S),
        "is_safety_or_certification_limit": False,
    }
    if state_source not in _HORIZON_THRESHOLDS:
        return {
            **base,
            "status": "not_scored",
            "passed": None,
            "reason": (
                "state_source must be 'ground_truth' or 'estimated' to select "
                "a threshold set"
            ),
        }
    if not per_profile:
        raise ValueError("accuracy evaluation requires per-profile metrics")

    horizon_thresholds = _HORIZON_THRESHOLDS[state_source]
    horizon_results: dict[str, Any] = {}
    scored_seconds: list[float] = []
    for label, aggregate_metrics in aggregate_horizon_rollouts.items():
        threshold = _threshold_for_horizon(horizon_thresholds, label)
        if threshold is None:
            continue
        position_limit, attitude_limit = threshold
        aggregate_position = _metric_gate(
            float(aggregate_metrics["position_rmse_m"]), position_limit
        )
        aggregate_attitude = _metric_gate(
            float(aggregate_metrics["attitude_rmse_deg"]), attitude_limit
        )
        worst_position_profile, worst_position_value = _worst_profile_metric(
            per_profile,
            section="horizon_rollouts",
            horizon_label=label,
            metric="position_rmse_m",
        )
        worst_attitude_profile, worst_attitude_value = _worst_profile_metric(
            per_profile,
            section="horizon_rollouts",
            horizon_label=label,
            metric="attitude_rmse_deg",
        )
        worst_position = _metric_gate(worst_position_value, position_limit)
        worst_position["profile"] = worst_position_profile
        worst_attitude = _metric_gate(worst_attitude_value, attitude_limit)
        worst_attitude["profile"] = worst_attitude_profile
        passed = all(
            result["passed"]
            for result in (
                aggregate_position,
                aggregate_attitude,
                worst_position,
                worst_attitude,
            )
        )
        horizon_results[label] = {
            "passed": passed,
            "aggregate": {
                "position_rmse_m": aggregate_position,
                "attitude_rmse_deg": aggregate_attitude,
            },
            "worst_profile": {
                "position_rmse_m": worst_position,
                "attitude_rmse_deg": worst_attitude,
            },
        }
        seconds = _horizon_seconds(label)
        if seconds is not None:
            scored_seconds.append(seconds)

    profile_position_fractions: dict[str, float] = {}
    for profile, result in per_profile.items():
        path_length_m = float(result["path_length_m"])
        if path_length_m <= 0.0:
            raise ValueError(
                f"profile {profile!r} needs positive path_length_m for full-flight scoring"
            )
        profile_position_fractions[profile] = (
            float(result["full_rollout"]["position_rmse_m"]) / path_length_m
        )

    full_position_limit, full_attitude_limit = _FULL_FLIGHT_THRESHOLDS[state_source]
    aggregate_position_fraction = math.sqrt(
        sum(value**2 for value in profile_position_fractions.values())
        / len(profile_position_fractions)
    )
    aggregate_position = _metric_gate(aggregate_position_fraction, full_position_limit)
    aggregate_attitude = _metric_gate(
        float(aggregate_full_rollout["attitude_rmse_deg"]), full_attitude_limit
    )
    worst_position_profile, worst_position_fraction = max(
        profile_position_fractions.items(), key=lambda item: item[1]
    )
    worst_attitude_profile, worst_attitude_value = _worst_profile_metric(
        per_profile,
        section="full_rollout",
        metric="attitude_rmse_deg",
    )
    worst_position = _metric_gate(worst_position_fraction, full_position_limit)
    worst_position["profile"] = worst_position_profile
    worst_attitude = _metric_gate(worst_attitude_value, full_attitude_limit)
    worst_attitude["profile"] = worst_attitude_profile
    full_passed = all(
        result["passed"]
        for result in (
            aggregate_position,
            aggregate_attitude,
            worst_position,
            worst_attitude,
        )
    )
    full_result = {
        "passed": full_passed,
        "aggregate": {
            "position_rmse_fraction_of_path": aggregate_position,
            "attitude_rmse_deg": aggregate_attitude,
        },
        "worst_profile": {
            "position_rmse_fraction_of_path": worst_position,
            "attitude_rmse_deg": worst_attitude,
        },
    }

    missing_required = [
        seconds
        for seconds in REQUIRED_HORIZONS_S
        if not any(
            math.isclose(seconds, scored, abs_tol=1e-9, rel_tol=0.0)
            for scored in scored_seconds
        )
    ]
    scored_passes = [result["passed"] for result in horizon_results.values()] + [
        full_passed
    ]
    if not all(scored_passes):
        status = "fail"
        passed: bool | None = False
    elif missing_required:
        status = "incomplete"
        passed = None
    else:
        status = "pass"
        passed = True

    return {
        **base,
        "status": status,
        "passed": passed,
        "missing_required_horizons_s": missing_required,
        "horizon_rollouts": horizon_results,
        "full_rollout": full_result,
    }


def evaluate_fixedwing_accuracy(
    airframes: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Evaluate every fixed-wing configuration against a development contract.

    Each airframe must independently satisfy aggregate and p90 position and
    attitude limits, improve the complete metric grid over kinematic
    persistence, and keep every full logged-input rollout finite. These are
    predictive development targets, not safety or certification limits.
    """

    if not airframes:
        raise ValueError("fixed-wing accuracy evaluation requires an airframe")
    result: dict[str, Any] = {
        "contract_id": FIXED_WING_CONTRACT_ID,
        "platform": "fixedwing",
        "scope": "held-out logged-input prediction",
        "policy": "every_airframe_aggregate_and_p90_must_pass",
        "required_horizons_s": list(FIXED_WING_REQUIRED_HORIZONS_S),
        "maximum_persistence_score": FIXED_WING_MAXIMUM_PERSISTENCE_SCORE,
        "is_safety_or_certification_limit": False,
        "airframes": {},
    }
    any_failure = False
    any_incomplete = False
    for name, airframe in airframes.items():
        aggregate = airframe["aggregate_horizon_rollouts"]
        p90 = airframe["p90_horizon_rollouts"]
        horizons: dict[str, Any] = {}
        missing: list[float] = []
        for seconds, (
            position_limit,
            attitude_limit,
        ) in _FIXED_WING_HORIZON_THRESHOLDS.items():
            label = f"{seconds:g}s"
            if label not in aggregate or label not in p90:
                missing.append(seconds)
                continue
            aggregate_position = _metric_gate(
                float(aggregate[label]["position_rmse_m"]), position_limit
            )
            aggregate_attitude = _metric_gate(
                float(aggregate[label]["attitude_rmse_deg"]), attitude_limit
            )
            p90_position = _metric_gate(
                float(p90[label]["position_rmse_m"]), position_limit
            )
            p90_attitude = _metric_gate(
                float(p90[label]["attitude_rmse_deg"]), attitude_limit
            )
            passed = all(
                gate["passed"]
                for gate in (
                    aggregate_position,
                    aggregate_attitude,
                    p90_position,
                    p90_attitude,
                )
            )
            horizons[label] = {
                "passed": passed,
                "aggregate": {
                    "position_rmse_m": aggregate_position,
                    "attitude_rmse_deg": aggregate_attitude,
                },
                "p90": {
                    "position_rmse_m": p90_position,
                    "attitude_rmse_deg": p90_attitude,
                },
            }

        persistence = _metric_gate(
            float(airframe["score_vs_kinematic_persistence"]),
            FIXED_WING_MAXIMUM_PERSISTENCE_SCORE,
        )
        finite_fraction = float(airframe["full_rollout_finite_fraction"])
        full_finite = _metric_gate(1.0 - finite_fraction, 0.0)
        full_finite.update(
            {
                "finite_fraction": finite_fraction,
                "required_fraction": 1.0,
            }
        )
        scored_passes = [item["passed"] for item in horizons.values()]
        if (
            not all(scored_passes)
            or not persistence["passed"]
            or not full_finite["passed"]
        ):
            status = "fail"
            passed: bool | None = False
            any_failure = True
        elif missing:
            status = "incomplete"
            passed = None
            any_incomplete = True
        else:
            status = "pass"
            passed = True
        result["airframes"][name] = {
            "status": status,
            "passed": passed,
            "missing_required_horizons_s": missing,
            "horizon_rollouts": horizons,
            "persistence": persistence,
            "full_rollout_finiteness": full_finite,
        }

    if any_failure:
        result["status"] = "fail"
        result["passed"] = False
    elif any_incomplete:
        result["status"] = "incomplete"
        result["passed"] = None
    else:
        result["status"] = "pass"
        result["passed"] = True
    return result
