"""Crazyflow fleet identification and the configuration-change prior.

The prototype learns what an arm-length change does to a multirotor by fitting
one effective model per fleet member and reading the between-member spread as a
directional prior. That machinery lives here, separate from the online
adaptation loop that consumes it.
"""

from __future__ import annotations

import math
import time
from dataclasses import replace
from typing import Any

import numpy as np

from glassbox.belief.belief import (
    DynamicsBelief,
    EmpiricalErrorSample,
    EmpiricalHorizonPredictiveError,
    LocalGaussianParameterBelief,
    structured_parameter_names,
    structured_parameter_vector,
)
from glassbox.belief.parameter_prior import StructuredParameterPrior
from glassbox.control.nmpc import TrackingTolerances
from glassbox.core.data import Trajectory
from glassbox.core.dynamics import ModelParams, physics_parameters
from glassbox.core.evaluation import windowed_rollout_evaluation
from glassbox.core.identification import fit_dynamics
from glassbox.core.runtime import (
    DirectActuationMap,
    RuntimeDynamicsModel,
    runtime_spec_from_trajectory,
)
from glassbox.core.synthetic import initial_parameter_guess
from glassbox.integrations.crazyflow_telemetry import (
    CONTROL_HORIZON_STEPS,
    DEFAULT_ARM_LENGTH_RATIO,
    DEFAULT_DURATION_S,
    generate_crazyflow_trajectory,
)

ADAPTATION_HORIZON_STEPS = 5
FLEET_LOG_ARM_LENGTH_RATIOS = (-0.25, -0.125, 0.0, 0.125, 0.25)
FLEET_PROFILE_COUNT = 2


def _fit_effective_model(
    trajectories: list[Trajectory],
    initial_params: ModelParams,
    *,
    steps: int,
    learning_rate: float,
) -> tuple[ModelParams, dict[str, Any]]:
    from glassbox.core.data import trajectory_windows

    windows = trajectory_windows(
        trajectories,
        horizon=ADAPTATION_HORIZON_STEPS,
        stride=2,
        maximum_windows=1_000,
    )
    started_at = time.perf_counter()
    result = fit_dynamics(
        windows,
        initial_params,
        steps=steps,
        learning_rate=learning_rate,
        instantaneous_rotational_response=True,
        diagonal_angular_control=True,
    )
    elapsed_s = time.perf_counter() - started_at
    physical = physics_parameters(result.params).physical()
    return result.params, {
        "trajectory_count": len(trajectories),
        "window_count": len(windows.initial_states),
        "horizon_s": ADAPTATION_HORIZON_STEPS * windows.dt_s,
        "optimization_steps": steps,
        "initial_loss": result.initial_loss,
        "final_loss": result.final_loss,
        "wall_time_s": elapsed_s,
        "effective_parameters": {
            "thrust_accel": float(physical["thrust_accel"]),
            "angular_accel": np.asarray(physical["angular_accel"]).tolist(),
            "linear_drag": float(physical["linear_drag"]),
            "angular_drag": np.asarray(physical["angular_drag"]).tolist(),
            "motor_time_constant": float(physical["motor_time_constant"]),
        },
    }


def _configuration_direction_prior(
    members: list[DynamicsBelief],
    configuration_coordinates: tuple[float, ...],
) -> tuple[StructuredParameterPrior, int]:
    """Keep only fleet variation explained by one known configuration axis."""

    if len(members) != len(configuration_coordinates) or len(members) < 2:
        raise ValueError("configuration coordinates must identify every fleet member")
    raw = StructuredParameterPrior.from_beliefs(
        tuple(members),
        source="independent_crazyflow_configuration_fits",
        member_labels=tuple(
            f"fleet-configuration-{index}" for index in range(len(members))
        ),
    )
    coordinates = np.asarray(configuration_coordinates, dtype=np.float64)
    if not np.all(np.isfinite(coordinates)) or np.std(coordinates) <= 0.0:
        raise ValueError("configuration coordinates must contain finite variation")
    centered_coordinate = coordinates - np.mean(coordinates)
    values = np.stack(
        [
            np.asarray(structured_parameter_vector(member.params), dtype=np.float64)
            for member in members
        ]
    )
    centered_values = values - np.mean(values, axis=0)
    slope = (
        centered_coordinate @ centered_values / np.sum(np.square(centered_coordinate))
    )
    coordinate_variance = float(np.var(coordinates, ddof=1))
    between = np.outer(slope, slope) * coordinate_variance

    inverse_scale = 1.0 / raw.natural_scale
    normalized = between * inverse_scale[:, None] * inverse_scale[None, :]
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (normalized + normalized.T))
    tolerance = (
        np.finfo(np.float64).eps
        * len(eigenvalues)
        * max(float(np.max(np.abs(eigenvalues))), 1.0)
    )
    rank = int(np.count_nonzero(eigenvalues > tolerance))
    if rank != 1:
        raise ValueError("one-dimensional configuration regression must have rank one")
    unresolved = eigenvectors[:, : len(eigenvalues) - rank]
    completion_normalized = unresolved @ unresolved.T
    completion = (
        completion_normalized * raw.natural_scale[:, None] * raw.natural_scale[None, :]
    )
    prior = replace(
        raw,
        between_member_covariance=between,
        completion_covariance=completion,
        source="one_dimensional_arm_coordinate_regression_from_crazyflow_fits",
    )
    return prior, raw.empirical_rank


def _build_crazyflow_beliefs() -> tuple[
    DynamicsBelief,
    DynamicsBelief,
    RuntimeDynamicsModel,
    dict[str, Any],
]:
    baseline_trajectories = [
        generate_crazyflow_trajectory(
            seed=101 + index,
            duration_s=DEFAULT_DURATION_S,
            arm_length_ratio=1.0,
            source_group=f"prechange-flight-{index}",
            configuration_id="prechange_vehicle",
        )
        for index in range(2)
    ]
    baseline_params, baseline_fit = _fit_effective_model(
        baseline_trajectories,
        initial_parameter_guess(),
        steps=250,
        learning_rate=0.02,
    )
    runtime_spec = runtime_spec_from_trajectory(baseline_trajectories[0])

    members: list[DynamicsBelief] = []
    member_reports: list[dict[str, Any]] = []
    error_samples: dict[int, list[EmpiricalErrorSample]] = {
        ADAPTATION_HORIZON_STEPS: [],
        CONTROL_HORIZON_STEPS: [],
    }
    for member_index, log_ratio in enumerate(FLEET_LOG_ARM_LENGTH_RATIOS):
        ratio = math.exp(log_ratio)
        label = f"fleet-configuration-{member_index}"
        trajectories = [
            generate_crazyflow_trajectory(
                seed=200 + 10 * member_index + profile_index,
                duration_s=4.0,
                arm_length_ratio=ratio,
                source_group=label,
                configuration_id=label,
            )
            for profile_index in range(FLEET_PROFILE_COUNT)
        ]
        member_params, fit_report = _fit_effective_model(
            trajectories,
            baseline_params,
            steps=100,
            learning_rate=0.01,
        )
        members.append(
            DynamicsBelief(
                member_params,
                trajectories[0].spec,
                runtime_spec,
                provenance={"role": "independently_fitted_fleet_member"},
            )
        )
        member_reports.append(
            {
                "member": label,
                "experiment_arm_length_ratio": ratio,
                "fit": fit_report,
            }
        )
        for profile_index, trajectory in enumerate(trajectories):
            for horizon_steps in error_samples:
                _, errors = windowed_rollout_evaluation(
                    baseline_params,
                    trajectory,
                    horizon_steps=horizon_steps,
                    stride_steps=horizon_steps,
                )
                error_samples[horizon_steps].append(
                    EmpiricalErrorSample(
                        errors=errors,
                        source_group=label,
                        trajectory_id=f"{label}-profile-{profile_index}",
                    )
                )

    fleet_prior, unprojected_fit_scatter_rank = _configuration_direction_prior(
        members,
        FLEET_LOG_ARM_LENGTH_RATIOS,
    )
    sample_period_s = baseline_trajectories[0].nominal_dt_s
    predictive_error = EmpiricalHorizonPredictiveError.from_samples(
        {
            horizon_steps * sample_period_s: tuple(samples)
            for horizon_steps, samples in error_samples.items()
        }
    )
    parameter_belief = LocalGaussianParameterBelief(
        parameter_names=structured_parameter_names(baseline_params),
        covariance=fleet_prior.between_member_covariance,
        source="independent_crazyflow_configuration_fits",
        evidence_count=fleet_prior.member_count,
        effective_sample_count=float(fleet_prior.member_count),
    )

    adaptation = generate_crazyflow_trajectory(
        seed=21,
        duration_s=0.8,
        arm_length_ratio=DEFAULT_ARM_LENGTH_RATIO,
        source_group="unknown-target-adaptation",
        configuration_id="unknown_target_configuration",
    )
    belief = DynamicsBelief(
        baseline_params,
        adaptation.spec,
        runtime_spec,
        predictive_error=predictive_error,
        parameter_belief=parameter_belief,
        provenance={
            "role": "prechange_vehicle_plus_configuration_fleet",
            "hidden_target_configuration_supplied": False,
        },
    )
    updated, update_report = belief.update(adaptation)

    oracle_trajectories = [
        generate_crazyflow_trajectory(
            seed=31 + index,
            duration_s=5.0,
            arm_length_ratio=DEFAULT_ARM_LENGTH_RATIO,
            source_group=f"target-oracle-flight-{index}",
            configuration_id="target_oracle_only",
        )
        for index in range(2)
    ]
    oracle_params, oracle_fit = _fit_effective_model(
        oracle_trajectories,
        baseline_params,
        steps=150,
        learning_rate=0.01,
    )
    oracle_runtime = RuntimeDynamicsModel(
        oracle_params,
        belief.input_spec,
        replace(
            runtime_spec,
            certified_prediction_horizon_s=(CONTROL_HORIZON_STEPS * sample_period_s),
            certification_source="Crazyflow prototype equal-horizon comparison",
        ),
        DirectActuationMap(belief.input_spec.controls),
    )

    evaluation = generate_crazyflow_trajectory(
        seed=22,
        duration_s=1.6,
        arm_length_ratio=DEFAULT_ARM_LENGTH_RATIO,
        source_group="unknown-target-independent-evaluation",
        configuration_id="unknown_target_configuration",
    )
    _, before_errors = windowed_rollout_evaluation(
        belief.params,
        evaluation,
        horizon_steps=CONTROL_HORIZON_STEPS,
        stride_steps=CONTROL_HORIZON_STEPS,
    )
    _, after_errors = windowed_rollout_evaluation(
        updated.params,
        evaluation,
        horizon_steps=CONTROL_HORIZON_STEPS,
        stride_steps=CONTROL_HORIZON_STEPS,
    )
    tolerances = np.asarray(
        TrackingTolerances.for_platform("multirotor").local_state_scale
    )
    before_rms = float(np.sqrt(np.mean(np.square(before_errors / tolerances))))
    after_rms = float(np.sqrt(np.mean(np.square(after_errors / tolerances))))
    evidence = {
        "baseline_fit": baseline_fit,
        "fleet": {
            "member_count": fleet_prior.member_count,
            "empirical_rank": fleet_prior.empirical_rank,
            "unprojected_fit_scatter_rank": unprojected_fit_scatter_rank,
            "configuration_delta_covariance_rank": int(
                np.linalg.matrix_rank(fleet_prior.between_member_covariance)
            ),
            "known_configuration_coordinate_used_for_fleet_only": True,
            "completion_fraction_in_natural_coordinates": (
                fleet_prior.completion_fraction_in_natural_coordinates
            ),
            "members": member_reports,
            "predictive_error_horizons_s": list(predictive_error.horizons_s),
            "predictive_error_group_count": list(
                predictive_error.independent_group_count
            ),
        },
        "adaptation": update_report.to_dict(),
        "oracle_fit": oracle_fit,
        "independent_prediction": {
            "horizon_s": CONTROL_HORIZON_STEPS * sample_period_s,
            "normalized_rms_before": before_rms,
            "normalized_rms_after": after_rms,
            "normalized_rms_ratio": after_rms / before_rms,
        },
    }
    return belief, updated, oracle_runtime, evidence
