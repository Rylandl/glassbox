"""Compact synthetic evidence for fleet-prior live adaptation."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from glassbox.belief import (
    DynamicsBelief,
    EmpiricalErrorSample,
    EmpiricalHorizonPredictiveError,
    LocalGaussianParameterBelief,
    structured_parameter_names,
    structured_parameter_vector,
    with_structured_parameter_vector,
)
from glassbox.data import Trajectory
from glassbox.dynamics import ModelParams
from glassbox.evaluation import windowed_rollout_evaluation
from glassbox.fixedwing_synthetic import (
    generate_fixed_wing_trajectory,
    true_fixed_wing_parameters,
)
from glassbox.nmpc import TrackingTolerances
from glassbox.parameter_prior import StructuredParameterPrior
from glassbox.runtime import runtime_spec_from_trajectory
from glassbox.synthetic import generate_trajectory, true_parameters

ADAPTATION_HORIZON_STEPS = 5
EVALUATION_HORIZON_STEPS = 10
SAMPLE_DT_S = 0.02
FLEET_TRAJECTORY_DURATION_S = 0.8
ADAPTATION_DURATION_S = 0.8
EVALUATION_DURATION_S = 1.6

TrajectoryGenerator = Callable[..., Trajectory]


@dataclass(frozen=True)
class _Scenario:
    family: str
    base_params: ModelParams
    generate: TrajectoryGenerator
    member_shifts: tuple[Mapping[str, float], ...]
    target_shift: Mapping[str, float]


def _scenarios() -> tuple[_Scenario, ...]:
    return (
        _Scenario(
            family="multirotor",
            base_params=true_parameters(),
            generate=generate_trajectory,
            member_shifts=(
                {
                    "log_thrust_accel": -0.25,
                    "log_angular_accel[0]": -0.20,
                    "log_motor_time_constant": -0.15,
                },
                {
                    "log_thrust_accel": -0.10,
                    "log_angular_accel[0]": 0.15,
                    "log_motor_time_constant": 0.10,
                },
                {
                    "log_thrust_accel": 0.15,
                    "log_angular_accel[0]": -0.10,
                    "log_motor_time_constant": 0.20,
                },
                {
                    "log_thrust_accel": 0.30,
                    "log_angular_accel[0]": 0.20,
                    "log_motor_time_constant": -0.10,
                },
            ),
            target_shift={
                "log_thrust_accel": 0.22,
                "log_angular_accel[0]": -0.18,
                "log_motor_time_constant": 0.25,
            },
        ),
        _Scenario(
            family="fixedwing",
            base_params=true_fixed_wing_parameters(),
            generate=generate_fixed_wing_trajectory,
            member_shifts=(
                {
                    "log_thrust_accel": -0.20,
                    "log_surface_angular_accel_per_speed_sq[0]": -0.15,
                    "log_surface_angular_accel_per_speed_sq[1]": -0.10,
                    "log_surface_angular_accel_per_speed_sq[2]": -0.20,
                    "log_actuator_time_constant": -0.20,
                    "surface_trim_unconstrained[0]": -0.03,
                    "surface_trim_unconstrained[1]": 0.02,
                    "surface_trim_unconstrained[2]": -0.01,
                },
                {
                    "log_thrust_accel": -0.10,
                    "log_surface_angular_accel_per_speed_sq[0]": 0.20,
                    "log_surface_angular_accel_per_speed_sq[1]": 0.15,
                    "log_surface_angular_accel_per_speed_sq[2]": 0.10,
                    "log_actuator_time_constant": 0.10,
                    "surface_trim_unconstrained[0]": 0.02,
                    "surface_trim_unconstrained[1]": -0.02,
                    "surface_trim_unconstrained[2]": 0.015,
                },
                {
                    "log_thrust_accel": 0.10,
                    "log_surface_angular_accel_per_speed_sq[0]": -0.10,
                    "log_surface_angular_accel_per_speed_sq[1]": 0.20,
                    "log_surface_angular_accel_per_speed_sq[2]": 0.15,
                    "log_actuator_time_constant": 0.20,
                    "surface_trim_unconstrained[0]": -0.01,
                    "surface_trim_unconstrained[1]": 0.03,
                    "surface_trim_unconstrained[2]": 0.02,
                },
                {
                    "log_thrust_accel": 0.20,
                    "log_surface_angular_accel_per_speed_sq[0]": 0.15,
                    "log_surface_angular_accel_per_speed_sq[1]": -0.20,
                    "log_surface_angular_accel_per_speed_sq[2]": -0.10,
                    "log_actuator_time_constant": -0.10,
                    "surface_trim_unconstrained[0]": 0.03,
                    "surface_trim_unconstrained[1]": 0.01,
                    "surface_trim_unconstrained[2]": -0.02,
                },
                {
                    "log_surface_angular_accel_per_speed_sq[0]": 0.05,
                    "log_surface_angular_accel_per_speed_sq[2]": 0.05,
                    "surface_trim_unconstrained[0]": -0.02,
                    "surface_trim_unconstrained[1]": -0.01,
                },
            ),
            target_shift={
                "log_thrust_accel": 0.18,
                "log_surface_angular_accel_per_speed_sq[0]": -0.12,
                "log_surface_angular_accel_per_speed_sq[1]": 0.18,
                "log_surface_angular_accel_per_speed_sq[2]": -0.15,
                "log_actuator_time_constant": 0.22,
                "surface_trim_unconstrained[0]": 0.025,
                "surface_trim_unconstrained[1]": -0.025,
                "surface_trim_unconstrained[2]": 0.02,
            },
        ),
    )


def _shift_parameters(
    params: ModelParams,
    shifts: Mapping[str, float],
) -> ModelParams:
    names = structured_parameter_names(params)
    unknown = set(shifts) - set(names)
    if unknown:
        raise ValueError(f"unknown structured parameter shifts: {sorted(unknown)}")
    vector = np.asarray(structured_parameter_vector(params), dtype=np.float64)
    indices = {name: index for index, name in enumerate(names)}
    for name, value in shifts.items():
        vector[indices[name]] += float(value)
    return with_structured_parameter_vector(params, jnp.asarray(vector))


def _normalized_prediction_rms(errors: np.ndarray, family: str) -> float:
    scale = np.asarray(TrackingTolerances.for_platform(family).local_state_scale)
    return float(np.sqrt(np.mean(np.square(errors / scale))))


def _normalized_parameter_rms(
    params: ModelParams,
    target: ModelParams,
    prior: StructuredParameterPrior,
) -> float:
    delta = np.asarray(structured_parameter_vector(params)) - np.asarray(
        structured_parameter_vector(target)
    )
    return float(np.sqrt(np.mean(np.square(delta / prior.natural_scale))))


def _normalized_covariance_trace(
    belief: LocalGaussianParameterBelief,
    prior: StructuredParameterPrior,
) -> float:
    return float(
        np.sum(np.diag(belief.covariance) / np.square(prior.natural_scale))
    )


def _run_scenario(scenario: _Scenario) -> dict[str, Any]:
    reference = scenario.generate(seed=0, duration_s=0.2)
    runtime_spec = runtime_spec_from_trajectory(reference)
    member_params = tuple(
        _shift_parameters(scenario.base_params, shifts)
        for shifts in scenario.member_shifts
    )
    member_labels = tuple(
        f"{scenario.family}-configuration-{index}"
        for index in range(len(member_params))
    )
    members = tuple(
        DynamicsBelief(
            params=params,
            input_spec=reference.spec,
            runtime_spec=runtime_spec,
        )
        for params in member_params
    )
    prior = StructuredParameterPrior.from_beliefs(
        members,
        source=f"synthetic_{scenario.family}_configuration_fleet",
        member_labels=member_labels,
    )
    nominal_params = with_structured_parameter_vector(
        scenario.base_params,
        jnp.asarray(prior.mean),
    )

    error_samples = []
    fleet_trajectories = []
    for index, (label, params) in enumerate(zip(member_labels, member_params)):
        trajectory = scenario.generate(
            seed=100 + index,
            duration_s=FLEET_TRAJECTORY_DURATION_S,
            params=params,
        )
        fleet_trajectories.append(trajectory)
        _, errors = windowed_rollout_evaluation(
            nominal_params,
            trajectory,
            horizon_steps=ADAPTATION_HORIZON_STEPS,
            stride_steps=ADAPTATION_HORIZON_STEPS,
        )
        error_samples.append(
            EmpiricalErrorSample(
                errors=errors,
                source_group=label,
                trajectory_id=f"{label}-held-out-profile",
            )
        )
    predictive_error = replace(
        EmpiricalHorizonPredictiveError.from_samples(
            {
                ADAPTATION_HORIZON_STEPS * SAMPLE_DT_S: tuple(error_samples),
            }
        ),
        source="synthetic_fleet_member_rollout_endpoints",
    )

    target_params = _shift_parameters(scenario.base_params, scenario.target_shift)
    adaptation_telemetry = replace(
        scenario.generate(
            seed=21,
            duration_s=ADAPTATION_DURATION_S,
            params=target_params,
        ),
        labels={"source_group": f"{scenario.family}-target-adaptation"},
    )
    evaluation_telemetry = scenario.generate(
        seed=22,
        duration_s=EVALUATION_DURATION_S,
        params=target_params,
    )
    shell = DynamicsBelief(
        params=scenario.base_params,
        input_spec=adaptation_telemetry.spec,
        runtime_spec=runtime_spec_from_trajectory(fleet_trajectories[0]),
        predictive_error=predictive_error,
        provenance={"benchmark_role": "unseen_target_vehicle_shell"},
    )
    belief = prior.initialize_belief(shell)
    before_metrics, before_errors = windowed_rollout_evaluation(
        belief.params,
        evaluation_telemetry,
        horizon_steps=EVALUATION_HORIZON_STEPS,
        stride_steps=EVALUATION_HORIZON_STEPS,
    )
    updated, update = belief.update(adaptation_telemetry)
    after_metrics, after_errors = windowed_rollout_evaluation(
        updated.params,
        evaluation_telemetry,
        horizon_steps=EVALUATION_HORIZON_STEPS,
        stride_steps=EVALUATION_HORIZON_STEPS,
    )
    if not isinstance(updated.parameter_belief, LocalGaussianParameterBelief):
        raise TypeError("applied adaptation did not preserve parameter covariance")
    before_rms = _normalized_prediction_rms(before_errors, scenario.family)
    after_rms = _normalized_prediction_rms(after_errors, scenario.family)
    return {
        "family": scenario.family,
        "fleet_evidence": {
            "member_count": prior.member_count,
            "parameter_count": len(prior.parameter_names),
            "empirical_rank": prior.empirical_rank,
            "completion_fraction_in_natural_coordinates": (
                prior.completion_fraction_in_natural_coordinates
            ),
            "predictive_error_group_count": (
                predictive_error.independent_group_count[0]
            ),
            "predictive_error_endpoint_count": predictive_error.raw_sample_count[0],
        },
        "adaptation": update.to_dict(),
        "independent_evaluation": {
            "horizon_s": EVALUATION_HORIZON_STEPS * SAMPLE_DT_S,
            "normalized_prediction_rms_before": before_rms,
            "normalized_prediction_rms_after": after_rms,
            "normalized_prediction_rms_ratio": after_rms / before_rms,
            "metrics_before": before_metrics,
            "metrics_after": after_metrics,
        },
        "parameter_diagnostics": {
            "interpretation": (
                "Generator-parameter distance is diagnostic only; predictive "
                "equivalence does not imply coefficient identifiability."
            ),
            "normalized_distance_to_generator_before": (
                _normalized_parameter_rms(belief.params, target_params, prior)
            ),
            "normalized_distance_to_generator_after": (
                _normalized_parameter_rms(updated.params, target_params, prior)
            ),
            "normalized_covariance_trace_before": (
                _normalized_covariance_trace(belief.parameter_belief, prior)
            ),
            "normalized_covariance_trace_after": (
                _normalized_covariance_trace(updated.parameter_belief, prior)
            ),
        },
    }


def run_adaptation_benchmark() -> dict[str, Any]:
    """Run both vehicle-family cases without fitting or acceptance thresholds."""

    scenarios = tuple(_run_scenario(scenario) for scenario in _scenarios())
    return {
        "format_version": 1,
        "artifact_type": "glassbox_synthetic_fleet_adaptation_diagnostic",
        "semantics": {
            "diagnostic_only": True,
            "acceptance_gate": False,
            "synthetic": True,
            "posterior_calibration_claim": False,
            "physical_parameter_recovery_required": False,
            "evaluation_telemetry_is_disjoint_from_adaptation_telemetry": True,
            "runtime_envelope_uses_target_telemetry": False,
        },
        "scenarios": list(scenarios),
        "observations": {
            "all_updates_applied": all(
                item["adaptation"]["applied"] for item in scenarios
            ),
            "all_independent_predictions_improved": all(
                item["independent_evaluation"][
                    "normalized_prediction_rms_after"
                ]
                < item["independent_evaluation"][
                    "normalized_prediction_rms_before"
                ]
                for item in scenarios
            ),
            "all_parameter_covariances_contracted": all(
                item["parameter_diagnostics"][
                    "normalized_covariance_trace_after"
                ]
                < item["parameter_diagnostics"][
                    "normalized_covariance_trace_before"
                ]
                for item in scenarios
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_adaptation_benchmark()
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output is None:
        print(payload, end="")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload)
    print(f"wrote {args.output}")
    for scenario in report["scenarios"]:
        evaluation = scenario["independent_evaluation"]
        print(
            f"{scenario['family']}: normalized held-out prediction RMS "
            f"{evaluation['normalized_prediction_rms_before']:.6f} -> "
            f"{evaluation['normalized_prediction_rms_after']:.6f} "
            f"(ratio {evaluation['normalized_prediction_rms_ratio']:.3f})"
        )


if __name__ == "__main__":
    main()
