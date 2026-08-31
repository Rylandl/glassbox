"""Fit an effective differentiable dynamics model from a trajectory artifact."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from glassbox.belief import (
    DynamicsBelief,
    EmpiricalErrorSample,
    EmpiricalHorizonPredictiveError,
    UnavailableParameterEvidence,
    UnavailablePredictiveError,
    parameter_evidence_from_dict,
    predictive_error_from_dict,
)
from glassbox.belief_io import save_dynamics_belief
from glassbox.data import (
    Trajectory,
    TrajectorySpec,
    TrajectoryWindows,
    duration_to_steps,
    load_trajectory_npz,
    split_trajectory,
    trajectory_windows,
)
from glassbox.dynamics import (
    MOTOR_MIXER,
    ModelParams,
    initial_residual_parameters,
    model_family,
    with_response_time_constant,
)
from glassbox.evaluation import (
    aggregate_innovation_diagnostics,
    aggregate_rollout_metrics,
    one_step_innovation_diagnostics,
    parameter_dict,
    rollout_metrics,
    windowed_rollout_evaluation,
)
from glassbox.fixedwing_synthetic import initial_fixed_wing_parameter_guess
from glassbox.identification import (
    MAX_OPTIMIZATION_TRANSITIONS_PER_HORIZON,
    MAX_OPTIMIZATION_WINDOWS_PER_HORIZON,
    OPTIMIZATION_POLICY_VERSION,
    fit_dynamics,
    fit_dynamics_multi_horizon,
    residual_initialization_statistics,
    rollout_loss_configuration,
)
from glassbox.model_family import MULTIROTOR_FAMILY, family_for_platform
from glassbox.observation_identification import (
    ObservationFitResult,
    fit_multirotor_observations,
)
from glassbox.parameter_evidence import (
    MAX_PARAMETER_EVIDENCE_WINDOWS_PER_HORIZON,
    estimate_local_parameter_information,
    fitted_structured_parameter_mask,
)
from glassbox.runtime import runtime_spec_from_fit_report
from glassbox.synthetic import initial_parameter_guess

_MAX_TRAINING_WINDOWS_PER_HORIZON = 8_192
_MAX_TRAINING_TRANSITIONS_PER_HORIZON = 524_288


def _automatic_training_window_budget(
    *, horizon_steps: int, source_group_count: int
) -> int:
    """Use all available windows up to fixed memory and transition bounds.

    ``trajectory_windows`` applies this ceiling after candidate construction, so
    small and medium corpora retain every valid window.  Large corpora are still
    bounded by both the number of windows and total unrolled transitions.
    """

    transition_limit = max(
        source_group_count,
        _MAX_TRAINING_TRANSITIONS_PER_HORIZON // horizon_steps,
    )
    return max(
        source_group_count,
        min(
            _MAX_TRAINING_WINDOWS_PER_HORIZON,
            transition_limit,
        ),
    )


def _excitation_diagnostics(
    controls: np.ndarray,
    control_names: tuple[str, ...],
    control_roles: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    roles = control_names if control_roles is None else control_roles
    result = {
        "control_names": list(control_names),
        "control_roles": list(roles),
        "control_standard_deviation": np.std(controls, axis=0).tolist(),
        "control_minimum": np.min(controls, axis=0).tolist(),
        "control_maximum": np.max(controls, axis=0).tolist(),
    }
    if roles == MULTIROTOR_FAMILY.control_roles:
        mixer_channels = controls @ np.asarray(MOTOR_MIXER).T
        collective = np.sum(controls, axis=1)
        result.update(
            {
                "motor_standard_deviation": np.std(controls, axis=0).tolist(),
                "collective_standard_deviation": float(np.std(collective)),
                "roll_pitch_yaw_standard_deviation": np.std(
                    mixer_channels, axis=0
                ).tolist(),
                "motor_minimum": np.min(controls, axis=0).tolist(),
                "motor_maximum": np.max(controls, axis=0).tolist(),
            }
        )
    else:
        throttle_index = roles.index("throttle")
        aerodynamic_indices = [
            index
            for index, role in enumerate(roles)
            if role in {"roll", "pitch", "yaw"}
        ]
        result.update(
            {
                "throttle_standard_deviation": float(
                    np.std(controls[:, throttle_index])
                ),
                "surface_standard_deviation": np.std(
                    controls[:, aerodynamic_indices], axis=0
                ).tolist(),
                "aerodynamic_role_standard_deviation": {
                    roles[index]: float(np.std(controls[:, index]))
                    for index in aerodynamic_indices
                },
                "auxiliary_role_standard_deviation": {
                    role: float(np.std(controls[:, index]))
                    for index, role in enumerate(roles)
                    if role not in {"throttle", "roll", "pitch", "yaw"}
                },
            }
        )
    return result


def _configured_initial_params(
    fixed_motor_time_constant_s: float | None,
    *,
    platform: str,
    observation_initializer: ModelParams | None = None,
) -> ModelParams:
    params: ModelParams
    if observation_initializer is not None:
        if model_family(observation_initializer).platform != platform:
            raise ValueError(
                "observation initializer does not match the requested platform"
            )
        params = observation_initializer
    else:
        params = (
            initial_fixed_wing_parameter_guess()
            if platform == "fixedwing"
            else initial_parameter_guess()
        )
    if fixed_motor_time_constant_s is not None:
        params = with_response_time_constant(params, fixed_motor_time_constant_s)
    return params


def _trajectory_platform(trajectory: Trajectory) -> str:
    family = family_for_platform(trajectory.spec.vehicle.family)
    family.validate_control_schema(
        trajectory.control_names,
        trajectory.spec.control_roles,
    )
    return family.platform


def _fit_on_windows(
    windows: TrajectoryWindows | tuple[TrajectoryWindows, ...],
    *,
    steps: int,
    learning_rate: float,
    fixed_motor_time_constant_s: float | None = None,
    horizon_labels: tuple[str, ...] | None = None,
    model_class: str = "structured",
    platform: str = "multirotor",
    endpoint_weight: float = 3.0,
    stability_regularization: float = 0.01,
    learn_thrust_command_offset: bool = False,
    instantaneous_rotational_response: bool = True,
    diagonal_angular_control: bool = True,
    observation_initializer: ModelParams | None = None,
    normalization_windows: (
        TrajectoryWindows | tuple[TrajectoryWindows, ...] | None
    ) = None,
) -> tuple[ModelParams, dict[str, Any]]:
    physics_params = _configured_initial_params(
        fixed_motor_time_constant_s,
        platform=platform,
        observation_initializer=observation_initializer,
    )
    normalization_physics_params = _configured_initial_params(
        fixed_motor_time_constant_s,
        platform=platform,
    )
    instantaneous_rotational_response = (
        instantaneous_rotational_response and platform == "multirotor"
    )
    diagonal_angular_control = (
        diagonal_angular_control and platform == "multirotor"
    )
    window_sets = windows if isinstance(windows, tuple) else (windows,)
    normalization_window_sets = (
        window_sets
        if normalization_windows is None
        else normalization_windows
        if isinstance(normalization_windows, tuple)
        else (normalization_windows,)
    )
    if len(normalization_window_sets) != len(window_sets):
        raise ValueError("normalization_windows must match fitting horizons")
    loss_configuration = rollout_loss_configuration(
        normalization_window_sets,
        endpoint_weight=endpoint_weight,
        stability_regularization=stability_regularization,
    )
    if model_class == "structured_residual":
        residual_statistics = residual_initialization_statistics(
            normalization_window_sets
        )
        initial_params: ModelParams = initial_residual_parameters(
            physics_params,
            control_size=window_sets[0].control_size,
            exogenous_size=window_sets[0].initial_exogenous.shape[1],
            **residual_statistics,
        )
        normalization_params: ModelParams = initial_residual_parameters(
            normalization_physics_params,
            control_size=window_sets[0].control_size,
            exogenous_size=window_sets[0].initial_exogenous.shape[1],
            **residual_statistics,
        )
    else:
        initial_params = physics_params
        normalization_params = normalization_physics_params
    if horizon_labels is None:
        horizon_labels = tuple(
            f"{item.controls.shape[1] * item.dt_s:g}s" for item in window_sets
        )
    if len(horizon_labels) != len(window_sets):
        raise ValueError("horizon_labels must match the supplied window sets")

    start = perf_counter()
    if len(window_sets) == 1:
        fit = fit_dynamics(
            window_sets[0],
            initial_params,
            steps=steps,
            learning_rate=learning_rate,
            fixed_motor_time_constant_s=fixed_motor_time_constant_s,
            learn_thrust_command_offset=learn_thrust_command_offset,
            instantaneous_rotational_response=instantaneous_rotational_response,
            diagonal_angular_control=diagonal_angular_control,
            loss_configuration=loss_configuration,
        )
    else:
        fit = fit_dynamics_multi_horizon(
            window_sets,
            initial_params,
            steps=steps,
            learning_rate=learning_rate,
            fixed_motor_time_constant_s=fixed_motor_time_constant_s,
            learn_thrust_command_offset=learn_thrust_command_offset,
            instantaneous_rotational_response=instantaneous_rotational_response,
            diagonal_angular_control=diagonal_angular_control,
            loss_configuration=loss_configuration,
            loss_normalization_params=normalization_params,
            loss_normalization_window_sets=normalization_window_sets,
        )
    wall_time_s = perf_counter() - start
    component_losses = {}
    if (
        fit.component_initial_losses is not None
        and fit.component_final_losses is not None
    ):
        component_losses = {
            label: {
                "initial_loss": float(initial_loss),
                "final_loss": float(final_loss),
                "loss_reduction": float(initial_loss / final_loss),
            }
            for label, initial_loss, final_loss in zip(
                horizon_labels,
                fit.component_initial_losses,
                fit.component_final_losses,
            )
        }
    return fit.params, {
        "fit": {
            "initial_loss": fit.initial_loss,
            "final_loss": fit.final_loss,
            "loss_reduction": fit.initial_loss / fit.final_loss,
            "wall_time_s": wall_time_s,
            "component_losses": component_losses,
            "multi_horizon_loss_normalizers": (
                None
                if fit.component_loss_normalizers is None
                else {
                    label: float(value)
                    for label, value in zip(
                        horizon_labels,
                        fit.component_loss_normalizers,
                    )
                }
            ),
            "rollout_loss": (
                fit.loss_configuration.to_dict()
                if fit.loss_configuration is not None
                else None
            ),
            "statistics_source": (
                "member_training_windows"
                if normalization_windows is None
                else "shared_outer_training_windows"
            ),
            "optimization_data_policy": {
                "policy": fit.optimization_policy,
                "automatic_policy": OPTIMIZATION_POLICY_VERSION,
                "maximum_windows_per_horizon_per_step": (
                    MAX_OPTIMIZATION_WINDOWS_PER_HORIZON
                ),
                "maximum_transitions_per_horizon_per_step": (
                    MAX_OPTIMIZATION_TRANSITIONS_PER_HORIZON
                ),
                "batch_size_by_horizon": {
                    label: size
                    for label, size in zip(
                        horizon_labels,
                        (
                            fit.batch_sizes
                            if fit.batch_sizes
                            else tuple(
                                len(item.initial_states) for item in window_sets
                            )
                        ),
                    )
                },
                "window_coverage_by_horizon": {
                    label: coverage
                    for label, coverage in zip(
                        horizon_labels,
                        (
                            fit.window_coverage
                            if fit.window_coverage
                            else tuple(1.0 for _ in window_sets)
                        ),
                    )
                },
                "full_window_initial_and_final_scoring": True,
            },
        },
        "parameters": {
            "initial": parameter_dict(initial_params),
            "fitted": parameter_dict(fit.params),
        },
    }


def _observation_fit(
    trajectories: list[Trajectory] | tuple[Trajectory, ...],
    *,
    platform: str,
) -> ObservationFitResult | None:
    """Return the automatic typed-observation stage when the data supports it."""

    if platform != "multirotor" or not trajectories:
        return None
    required = {f"specific_force_{axis}" for axis in "xyz"}
    if any(
        not required.issubset(trajectory.spec.observation_roles)
        for trajectory in trajectories
    ):
        return None
    result = fit_multirotor_observations(trajectories)
    return ObservationFitResult(
        params=result.params,
        report={
            **result.report,
            "rollout_initializer": {
                "applied": False,
                "status": "diagnostic_only",
                "reason": (
                    "direct sensor residuals have not passed the maintained "
                    "cross-platform rollout promotion gate"
                ),
            },
        },
    )


def fit_trajectory_artifact(
    trajectory_path: str | Path,
    *,
    train_fraction: float = 0.70,
    horizon: int = 25,
    stride: int | None = None,
    steps: int = 400,
    learning_rate: float = 0.02,
    fixed_motor_time_constant_s: float | None = None,
    endpoint_weight: float = 3.0,
    stability_regularization: float = 0.01,
) -> tuple[ModelParams, dict[str, Any]]:
    """Fit one trajectory and return parameters plus a JSON-compatible report."""

    trajectory = load_trajectory_npz(trajectory_path)
    platform = _trajectory_platform(trajectory)
    training, validation = split_trajectory(
        trajectory, train_fraction=train_fraction
    )
    maximum_windows = _automatic_training_window_budget(
        horizon_steps=horizon,
        source_group_count=1,
    )
    windows = trajectory_windows(
        [training],
        horizon=horizon,
        stride=horizon if stride is None else stride,
        maximum_windows=maximum_windows,
    )
    observation_fit = _observation_fit(
        [training], platform=platform
    )
    fitted_params, model_report = _fit_on_windows(
        windows,
        steps=steps,
        learning_rate=learning_rate,
        fixed_motor_time_constant_s=fixed_motor_time_constant_s,
        platform=platform,
        endpoint_weight=endpoint_weight,
        stability_regularization=stability_regularization,
    )
    initial_params = _configured_initial_params(
        fixed_motor_time_constant_s,
        platform=platform,
    )

    report = {
        "trajectory": str(trajectory_path),
        "source": {
            "spec": trajectory.spec.to_dict(),
            "labels": dict(trajectory.labels),
            "provenance": dict(trajectory.provenance),
        },
        "configuration": {
            "train_fraction": train_fraction,
            "training_duration_s": float(training.time_s[-1]),
            "validation_duration_s": float(validation.time_s[-1]),
            "horizon_steps": horizon,
            "horizon_duration_s": horizon * windows.dt_s,
            "stride_steps": horizon if stride is None else stride,
            "control_history_duration_s": (
                windows.control_histories.shape[1] * windows.dt_s
            ),
            "motor_history_duration_s": (
                windows.control_histories.shape[1] * windows.dt_s
                if platform == "multirotor"
                else None
            ),
            "optimization_steps": steps,
            "learning_rate": learning_rate,
            "endpoint_weight": endpoint_weight,
            "stability_regularization": stability_regularization,
            "fixed_motor_time_constant_s": fixed_motor_time_constant_s,
            "fixed_response_time_constant_s": fixed_motor_time_constant_s,
            "platform": platform,
            "training_windows": len(windows.initial_states),
            "training_window_selection": {
                "policy": windows.selection_policy,
                "maximum_windows": maximum_windows,
                "candidate_windows": windows.candidate_window_count,
                "selected_windows": len(windows.initial_states),
                "selection_fraction": (
                    len(windows.initial_states) / windows.candidate_window_count
                ),
            },
        },
        "fit": model_report["fit"],
        "observation_identification": (
            None if observation_fit is None else observation_fit.report
        ),
        "parameters": model_report["parameters"],
        "validation_rollout": {
            "initial": rollout_metrics(
                initial_params,
                validation,
                control_history=training.controls,
            ),
            "fitted": rollout_metrics(
                fitted_params,
                validation,
                control_history=training.controls,
            ),
        },
        "validation_innovation": {
            "initial": one_step_innovation_diagnostics(
                initial_params,
                validation,
                control_history=training.controls,
            ),
            "fitted": one_step_innovation_diagnostics(
                fitted_params,
                validation,
                control_history=training.controls,
            ),
        },
        "excitation": _excitation_diagnostics(
            training.controls,
            training.control_names,
            training.spec.control_roles,
        ),
        "interpretation": (
            "Parameters are effective predictive coefficients. They are not "
            "independently verified physical mass, inertia, aerodynamic, or "
            "actuator constants."
        ),
    }
    return fitted_params, report


@dataclass(frozen=True)
class _EvaluationFlight:
    path: str
    trajectory: Trajectory
    control_history: np.ndarray | None = None
    source_group: str | int | None = None


def _source_groups(
    paths: list[Path], trajectories: list[Trajectory]
) -> list[str | int] | None:
    values = [trajectory.labels.get("source_group") for trajectory in trajectories]
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        missing = [str(path) for path, value in zip(paths, values) if value is None]
        raise ValueError(
            "source-group splitting requires every trajectory to have a "
            f"source_group label; unlabeled: {', '.join(missing)}"
        )
    if any(
        not isinstance(value, (str, int))
        or (isinstance(value, str) and not value.strip())
        for value in values
    ):
        raise ValueError("source_group labels must be non-empty strings or integers")
    source_groups = [
        value for value in values if isinstance(value, (str, int))
    ]
    unique_groups = tuple(dict.fromkeys(source_groups))
    if len({str(group) for group in unique_groups}) != len(unique_groups):
        raise ValueError(
            "source_group labels must have unique string representations"
        )
    return source_groups


def _trajectory_summary(path: str, trajectory: Trajectory) -> dict[str, Any]:
    position = trajectory.states[:, 0:3]
    velocity = trajectory.states[:, 3:6]
    angular_velocity = trajectory.states[:, 10:13]
    return {
        "path": path,
        "duration_s": float(trajectory.time_s[-1]),
        "intervals": len(trajectory.controls),
        "control_size": trajectory.control_size,
        "control_names": list(trajectory.control_names),
        "sample_rate_hz": 1.0 / trajectory.nominal_dt_s,
        "characteristics": {
            "path_length_m": float(
                np.sum(np.linalg.norm(np.diff(position, axis=0), axis=1))
            ),
            "net_displacement_m": float(np.linalg.norm(position[-1] - position[0])),
            "position_range_xyz_m": np.ptp(position, axis=0).tolist(),
            "maximum_speed_m_s": float(
                np.max(np.linalg.norm(velocity, axis=1))
            ),
            "maximum_angular_speed_rad_s": float(
                np.max(np.linalg.norm(angular_velocity, axis=1))
            ),
            "excitation": _excitation_diagnostics(
                trajectory.controls,
                trajectory.control_names,
                trajectory.spec.control_roles,
            ),
        },
        "spec": trajectory.spec.to_dict(),
        "labels": dict(trajectory.labels),
        "provenance": dict(trajectory.provenance),
    }


def _dataset_contract(
    paths: list[Path], trajectories: list[Trajectory]
) -> dict[str, Any]:
    """Validate canonical semantics, independent of source adapter details."""

    def consistent_value(
        name: str, values: list[Any], *, serialize: bool = False
    ) -> Any:
        comparable = [
            json.dumps(value, sort_keys=True) if serialize else value
            for value in values
        ]
        unique = list(dict.fromkeys(comparable))
        if len(unique) > 1:
            details = ", ".join(
                f"{path}={value!r}" for path, value in zip(paths, values)
            )
            raise ValueError(f"inconsistent dataset {name}: {details}")
        return values[0]

    sample_rates = [1.0 / trajectory.nominal_dt_s for trajectory in trajectories]
    reference_rate = sample_rates[0]
    if not np.allclose(sample_rates, reference_rate, atol=1e-6, rtol=0.0):
        details = ", ".join(
            f"{path}={rate:g}Hz" for path, rate in zip(paths, sample_rates)
        )
        raise ValueError(f"inconsistent dataset sample_rate_hz: {details}")

    spec_payloads = [trajectory.spec.to_dict() for trajectory in trajectories]
    spec_payload = consistent_value(
        "trajectory_spec", spec_payloads, serialize=True
    )

    profiles = [trajectory.labels.get("profile") for trajectory in trajectories]
    profile_counts = {
        str(profile): profiles.count(profile)
        for profile in dict.fromkeys(profiles)
        if profile is not None
    }
    conditions = [trajectory.labels.get("condition") for trajectory in trajectories]
    condition_counts = {
        str(condition): conditions.count(condition)
        for condition in dict.fromkeys(conditions)
        if condition is not None
    }
    source_types = []
    for trajectory in trajectories:
        adapter = trajectory.provenance.get("adapter", {})
        adapter_name = adapter.get("name") if isinstance(adapter, Mapping) else None
        source_types.append(str(adapter_name or "unknown"))
    source_type_counts = {
        source_type: source_types.count(source_type)
        for source_type in dict.fromkeys(source_types)
    }
    platform = str(spec_payload["vehicle"]["family"])

    if platform == "fixedwing":
        for path, trajectory, source_type in zip(paths, trajectories, source_types):
            if source_type != "px4_ulog":
                continue
            px4 = trajectory.provenance.get("px4", {})
            mapping = (
                px4.get("actuator_mapping", {})
                if isinstance(px4, Mapping)
                else {}
            )
            mapping_verified = (
                mapping.get("actuator_mapping_verified")
                if isinstance(mapping, Mapping)
                else None
            )
            if mapping_verified is not True:
                raise ValueError(
                    f"fixed-wing PX4 trajectory {path} lacks a verified "
                    "canonical actuator mapping"
                )

    contract = {
        "pooling_basis": "canonical_trajectory_spec",
        "flight_count": len(trajectories),
        "total_duration_s": float(
            sum(trajectory.time_s[-1] for trajectory in trajectories)
        ),
        "sample_rate_hz": reference_rate,
        "control_size": len(spec_payload["controls"]),
        "control_names": [
            channel["name"] for channel in spec_payload["controls"]
        ],
        "control_roles": [
            channel["role"] for channel in spec_payload["controls"]
        ],
        "control_semantics": [
            channel["semantic"] for channel in spec_payload["controls"]
        ],
        "exogenous_size": len(spec_payload["exogenous"]),
        "exogenous_names": [
            channel["name"] for channel in spec_payload["exogenous"]
        ],
        "exogenous_roles": [
            channel["role"] for channel in spec_payload["exogenous"]
        ],
        "observation_size": len(spec_payload["observations"]),
        "observation_names": [
            channel["name"] for channel in spec_payload["observations"]
        ],
        "observation_roles": [
            channel["role"] for channel in spec_payload["observations"]
        ],
        "platform": platform,
        "source_type": (
            source_types[0] if len(source_type_counts) == 1 else "mixed"
        ),
        "source_type_counts": source_type_counts,
        "state_source": spec_payload["observation_source"],
        "state_schema": spec_payload["state_schema"],
        "coordinate_frames": {"world": "NWU", "body": "FLU"},
        "vehicle_configuration": spec_payload["vehicle"],
        "trajectory_spec": spec_payload,
        "profile_counts": profile_counts,
        "unlabeled_flight_count": profiles.count(None),
        "condition_counts": condition_counts,
        "unlabeled_condition_count": conditions.count(None),
    }
    return contract


def _evaluate_model(
    params: ModelParams,
    flights: list[_EvaluationFlight],
    *,
    horizon_seconds: tuple[float, ...],
) -> dict[str, Any]:
    per_flight: list[dict[str, Any]] = []
    full_metrics: list[dict[str, Any]] = []
    innovation_diagnostics: list[dict[str, Any]] = []
    horizon_metrics: dict[str, list[dict[str, Any]]] = {
        f"{seconds:g}s": [] for seconds in horizon_seconds
    }
    error_samples: dict[float, list[EmpiricalErrorSample]] = {
        seconds: [] for seconds in horizon_seconds
    }

    for flight in flights:
        trajectory = flight.trajectory
        full = rollout_metrics(
            params,
            trajectory,
            control_history=flight.control_history,
        )
        full_metrics.append(full)
        innovation = one_step_innovation_diagnostics(
            params,
            trajectory,
            control_history=flight.control_history,
        )
        innovation_diagnostics.append(innovation)
        per_horizon: dict[str, Any] = {}
        for seconds in horizon_seconds:
            label = f"{seconds:g}s"
            steps = duration_to_steps(seconds, trajectory.nominal_dt_s)
            if steps > len(trajectory.controls):
                continue
            metrics, endpoint_errors = windowed_rollout_evaluation(
                params,
                trajectory,
                horizon_steps=steps,
                stride_steps=steps,
            )
            metrics["requested_horizon_s"] = seconds
            metrics["horizon_steps"] = steps
            per_horizon[label] = metrics
            horizon_metrics[label].append(metrics)
            error_samples[seconds].append(
                EmpiricalErrorSample(
                    endpoint_errors,
                    source_group=str(
                        flight.source_group
                        if flight.source_group is not None
                        else flight.path
                    ),
                    trajectory_id=flight.path,
                )
            )

        per_flight.append(
            {
                "path": flight.path,
                "duration_s": float(trajectory.time_s[-1]),
                "full_rollout": full,
                "horizon_rollouts": per_horizon,
                "one_step_innovation": innovation,
            }
        )

    aggregate_horizons: dict[str, Any] = {}
    for label, items in horizon_metrics.items():
        if items:
            aggregate_horizons[label] = aggregate_rollout_metrics(items)

    available_error_samples = {
        horizon: samples
        for horizon, samples in error_samples.items()
        if samples
    }
    predictive_error = (
        EmpiricalHorizonPredictiveError.from_samples(available_error_samples)
        if available_error_samples
        else UnavailablePredictiveError(
            "held-out trajectories were shorter than every evaluation horizon"
        )
    )

    return {
        "aggregate": {
            "flight_count": len(flights),
            "weighting": "equal_flight",
            "full_rollout": aggregate_rollout_metrics(
                full_metrics, weighting="equal"
            ),
            "horizon_rollouts": {
                label: aggregate_rollout_metrics(items, weighting="equal")
                for label, items in horizon_metrics.items()
                if items
            },
            "one_step_innovation": aggregate_innovation_diagnostics(
                innovation_diagnostics
            ),
        },
        "sample_weighted_aggregate": {
            "flight_count": len(flights),
            "full_rollout": aggregate_rollout_metrics(full_metrics),
            "horizon_rollouts": aggregate_horizons,
        },
        "per_flight": per_flight,
        "predictive_error": predictive_error.to_dict(),
    }


def _comparison_report(
    learned: dict[str, Any], baseline: dict[str, Any]
) -> dict[str, Any]:
    metric_names = (
        "position_rmse_m",
        "velocity_rmse_m_s",
        "attitude_rmse_deg",
        "angular_velocity_rmse_rad_s",
        "final_position_error_m",
    )

    def ratios(
        learned_metrics: dict[str, Any], baseline_metrics: dict[str, Any]
    ) -> dict[str, float | None]:
        result: dict[str, float | None] = {}
        for name in metric_names:
            learned_value = float(learned_metrics[name])
            result[name] = (
                float(baseline_metrics[name]) / learned_value
                if learned_value > 0.0
                else None
            )
        return result

    learned_aggregate = learned["aggregate"]
    baseline_aggregate = baseline["aggregate"]
    horizon_ratios = {}
    for label, learned_metrics in learned_aggregate["horizon_rollouts"].items():
        if label in baseline_aggregate["horizon_rollouts"]:
            horizon_ratios[label] = ratios(
                learned_metrics,
                baseline_aggregate["horizon_rollouts"][label],
            )
    return {
        "ratio_definition": (
            "no_lag_rmse / learned_lag_rmse; values above one favor learned lag"
        ),
        "aggregate_full_rollout": ratios(
            learned_aggregate["full_rollout"],
            baseline_aggregate["full_rollout"],
        ),
        "aggregate_horizon_rollouts": horizon_ratios,
    }


def fit_trajectory_artifacts(
    trajectory_paths: list[str | Path] | tuple[str | Path, ...],
    *,
    train_fraction: float = 0.70,
    holdout_count: int = 1,
    horizon: int = 25,
    stride: int | None = None,
    training_horizons_s: tuple[float, ...] | None = None,
    steps: int = 400,
    learning_rate: float = 0.02,
    evaluation_horizons_s: tuple[float, ...] = (0.1, 0.5, 1.0, 2.0),
    run_no_lag_ablation: bool = True,
    balance_training_flights: bool = True,
    holdout_profiles: tuple[str, ...] | list[str] | None = None,
    training_source_group_weights: Mapping[str | int, float] | None = None,
    normalization_source_group_weights: (
        Mapping[str | int, float] | None
    ) = None,
    model_class: str = "structured",
    endpoint_weight: float = 3.0,
    stability_regularization: float = 0.01,
    learn_thrust_command_offset: bool = False,
    instantaneous_rotational_response: bool = True,
    diagonal_angular_control: bool = True,
    build_parameter_evidence: bool = False,
) -> tuple[ModelParams, ModelParams | None, dict[str, Any]]:
    """Fit across flights and reserve complete flights when multiple are given."""

    if not trajectory_paths:
        raise ValueError("at least one trajectory path is required")
    if any(seconds <= 0.0 for seconds in evaluation_horizons_s):
        raise ValueError("evaluation horizons must be positive")
    if training_horizons_s is not None and any(
        seconds <= 0.0 for seconds in training_horizons_s
    ):
        raise ValueError("training horizons must be positive")
    if model_class not in {"structured", "structured_residual"}:
        raise ValueError("model_class must be structured or structured_residual")
    if endpoint_weight < 1.0:
        raise ValueError("endpoint_weight must be at least one")
    if stability_regularization < 0.0:
        raise ValueError("stability_regularization must be nonnegative")
    if training_source_group_weights is not None:
        weights = np.asarray(
            list(training_source_group_weights.values()), dtype=np.float64
        )
        if (
            not np.all(np.isfinite(weights))
            or np.any(weights < 0.0)
            or not np.any(weights > 0.0)
        ):
            raise ValueError(
                "training_source_group_weights values must be finite and "
                "nonnegative with at least one positive group"
            )
    if normalization_source_group_weights is not None and any(
        not np.isfinite(weight) or weight <= 0.0
        for weight in normalization_source_group_weights.values()
    ):
        raise ValueError(
            "normalization_source_group_weights values must be finite and positive"
        )
    if (
        normalization_source_group_weights is not None
        and training_source_group_weights is None
    ):
        raise ValueError(
            "normalization_source_group_weights requires explicit training "
            "source-group weights"
        )

    paths = [Path(path) for path in trajectory_paths]
    trajectories = [load_trajectory_npz(path) for path in paths]
    source_groups = _source_groups(paths, trajectories)
    dataset_contract = _dataset_contract(paths, trajectories)
    dataset_contract["source_group_count"] = (
        len(dict.fromkeys(source_groups))
        if source_groups is not None
        else len(trajectories)
    )
    dataset_contract["source_grouping"] = (
        "trajectory_label:source_group"
        if source_groups is not None
        else "one_group_per_trajectory"
    )
    platform = _trajectory_platform(trajectories[0])
    if dataset_contract["platform"] is not None:
        platform = str(dataset_contract["platform"])
    family = family_for_platform(platform)
    family.validate_control_schema(
        trajectories[0].control_names,
        trajectories[0].spec.control_roles,
    )
    dataset_contract["platform"] = platform
    dataset_contract["model_family"] = family.key
    if model_class == "structured_residual" and not family.supports_residual:
        raise ValueError(
            f"structured_residual is not supported for platform {platform!r}"
        )
    if holdout_profiles and len(trajectories) == 1:
        raise ValueError("profile holdout requires multiple trajectories")
    if len(trajectories) == 1:
        training_segment, validation_segment = split_trajectory(
            trajectories[0], train_fraction=train_fraction
        )
        training = [training_segment]
        training_labels = [f"{paths[0]}#training"]
        validation = [
            _EvaluationFlight(
                path=f"{paths[0]}#validation",
                trajectory=validation_segment,
                control_history=training_segment.controls,
                source_group=(source_groups[0] if source_groups else None),
            )
        ]
        training_source_groups = None
        split_mode = "temporal_within_flight"
    elif holdout_profiles:
        selected_profiles = tuple(dict.fromkeys(holdout_profiles))
        profile_by_flight = [
            trajectory.labels.get("profile") for trajectory in trajectories
        ]
        if any(profile is None for profile in profile_by_flight):
            unlabeled = [
                str(path)
                for path, profile in zip(paths, profile_by_flight)
                if profile is None
            ]
            raise ValueError(
                "profile holdout requires every trajectory to have a profile; "
                f"unlabeled: {', '.join(unlabeled)}"
            )
        missing = [
            profile for profile in selected_profiles if profile not in profile_by_flight
        ]
        if missing:
            raise ValueError(f"holdout profiles are absent: {', '.join(missing)}")
        training_indices = [
            index
            for index, profile in enumerate(profile_by_flight)
            if profile not in selected_profiles
        ]
        validation_indices = [
            index
            for index, profile in enumerate(profile_by_flight)
            if profile in selected_profiles
        ]
        if not training_indices:
            raise ValueError("profile holdout cannot reserve every trajectory")
        training = [trajectories[index] for index in training_indices]
        training_labels = [str(paths[index]) for index in training_indices]
        training_source_groups = (
            [source_groups[index] for index in training_indices]
            if source_groups is not None
            else None
        )
        validation = [
            _EvaluationFlight(
                path=str(paths[index]),
                trajectory=trajectories[index],
                source_group=(source_groups[index] if source_groups else None),
            )
            for index in validation_indices
        ]
        split_mode = "leave_profiles_out"
    elif source_groups is not None:
        group_order = list(dict.fromkeys(source_groups))
        characterization_only = all(
            trajectory.labels.get("benchmark_split") == "characterization_only"
            for trajectory in trajectories
        )
        if len(group_order) == 1 and characterization_only:
            if not 1 <= holdout_count < len(trajectories):
                raise ValueError(
                    "holdout_count must reserve at least one but not all "
                    "characterization segments"
                )
            training_indices = list(range(len(trajectories) - holdout_count))
            validation_indices = list(
                range(len(trajectories) - holdout_count, len(trajectories))
            )
            split_mode = "chronological_segments_within_source_group_characterization"
        else:
            if not 1 <= holdout_count < len(group_order):
                raise ValueError(
                    "holdout_count must reserve at least one but not all source groups"
                )
            held_out_groups = set(group_order[-holdout_count:])
            training_indices = [
                index
                for index, group in enumerate(source_groups)
                if group not in held_out_groups
            ]
            validation_indices = [
                index
                for index, group in enumerate(source_groups)
                if group in held_out_groups
            ]
            split_mode = "leave_source_groups_out"
        training = [trajectories[index] for index in training_indices]
        training_labels = [str(paths[index]) for index in training_indices]
        training_source_groups = [source_groups[index] for index in training_indices]
        validation = [
            _EvaluationFlight(
                path=str(paths[index]),
                trajectory=trajectories[index],
                source_group=source_groups[index],
            )
            for index in validation_indices
        ]
    else:
        if not 1 <= holdout_count < len(trajectories):
            raise ValueError(
                "holdout_count must reserve at least one but not all flights"
            )
        training = trajectories[:-holdout_count]
        training_labels = [str(path) for path in paths[:-holdout_count]]
        validation = [
            _EvaluationFlight(path=str(path), trajectory=trajectory)
            for path, trajectory in zip(
                paths[-holdout_count:], trajectories[-holdout_count:]
            )
        ]
        training_source_groups = None
        split_mode = "leave_complete_flights_out"

    dt_s = training[0].nominal_dt_s
    if training_horizons_s is None:
        training_horizon_steps = (horizon,)
    else:
        training_horizon_steps = tuple(
            dict.fromkeys(
                duration_to_steps(seconds, dt_s)
                for seconds in training_horizons_s
            )
        )
    training_horizon_labels = tuple(
        f"{steps_at_horizon * dt_s:g}s"
        for steps_at_horizon in training_horizon_steps
    )
    training_profiles = [
        trajectory.labels.get("profile") for trajectory in training
    ]
    training_group_order = (
        list(dict.fromkeys(training_source_groups))
        if training_source_groups is not None
        else []
    )
    if training_source_group_weights is not None:
        if training_source_groups is None:
            raise ValueError(
                "training_source_group_weights requires source_group labels"
            )
        if set(training_source_group_weights) != set(training_group_order):
            raise ValueError(
                "training_source_group_weights must contain exactly the "
                "training source groups"
            )
    if normalization_source_group_weights is not None:
        if training_source_groups is None:
            raise ValueError(
                "normalization_source_group_weights requires source_group labels"
            )
        if set(normalization_source_group_weights) != set(training_group_order):
            raise ValueError(
                "normalization_source_group_weights must contain exactly the "
                "training source groups"
            )
    group_balanced = (
        balance_training_flights
        and (
            len(training) > 1
            or training_source_group_weights is not None
        )
        and training_source_groups is not None
        and (
            not holdout_profiles
            or training_source_group_weights is not None
        )
    )
    profile_balanced_weights = None
    if (
        not group_balanced
        and
        balance_training_flights
        and len(training) > 1
        and all(profile is not None for profile in training_profiles)
    ):
        profile_counts = {
            profile: training_profiles.count(profile)
            for profile in dict.fromkeys(training_profiles)
        }
        profile_balanced_weights = tuple(
            1.0 / profile_counts[profile] for profile in training_profiles
        )
    training_diversity_count = (
        len(dict.fromkeys(training_source_groups))
        if training_source_groups is not None
        else len(training)
    )
    maximum_windows_by_horizon = tuple(
        _automatic_training_window_budget(
            horizon_steps=steps_at_horizon,
            source_group_count=training_diversity_count,
        )
        for steps_at_horizon in training_horizon_steps
    )
    window_sets = tuple(
        trajectory_windows(
            training,
            horizon=steps_at_horizon,
            stride=steps_at_horizon if stride is None else stride,
            balance_trajectories=(
                balance_training_flights
                and len(training) > 1
                and not group_balanced
                and profile_balanced_weights is None
            ),
            trajectory_weights=profile_balanced_weights,
            trajectory_groups=(
                training_source_groups if group_balanced else None
            ),
            trajectory_group_weights=(
                training_source_group_weights if group_balanced else None
            ),
            maximum_windows=maximum_windows,
        )
        for steps_at_horizon, maximum_windows in zip(
            training_horizon_steps, maximum_windows_by_horizon
        )
    )
    normalization_window_sets = (
        None
        if normalization_source_group_weights is None
        else tuple(
            trajectory_windows(
                training,
                horizon=steps_at_horizon,
                stride=steps_at_horizon if stride is None else stride,
                trajectory_groups=training_source_groups,
                trajectory_group_weights=normalization_source_group_weights,
                maximum_windows=maximum_windows,
            )
            for steps_at_horizon, maximum_windows in zip(
                training_horizon_steps, maximum_windows_by_horizon
            )
        )
    )
    fitting_windows: TrajectoryWindows | tuple[TrajectoryWindows, ...]
    fitting_windows = window_sets[0] if len(window_sets) == 1 else window_sets
    normalization_fitting_windows = (
        None
        if normalization_window_sets is None
        else normalization_window_sets[0]
        if len(normalization_window_sets) == 1
        else normalization_window_sets
    )
    observation_fit = _observation_fit(training, platform=platform)
    learned_params, learned_report = _fit_on_windows(
        fitting_windows,
        steps=steps,
        learning_rate=learning_rate,
        horizon_labels=training_horizon_labels,
        model_class=model_class,
        platform=platform,
        endpoint_weight=endpoint_weight,
        stability_regularization=stability_regularization,
        learn_thrust_command_offset=learn_thrust_command_offset,
        instantaneous_rotational_response=instantaneous_rotational_response,
        diagonal_angular_control=diagonal_angular_control,
        normalization_windows=normalization_fitting_windows,
    )
    learned_report["observation_identification"] = (
        None if observation_fit is None else observation_fit.report
    )
    learned_report["validation"] = _evaluate_model(
        learned_params,
        validation,
        horizon_seconds=evaluation_horizons_s,
    )

    evidence_groups = (
        training_source_groups
        if training_source_groups is not None
        else training_labels
    )
    evidence_independence_unit = (
        "source_group"
        if training_source_groups is not None
        else "temporal_training_segment"
        if split_mode == "temporal_within_flight"
        else "complete_trajectory"
    )

    def attach_parameter_evidence(
        selected_params: ModelParams,
        model_report: dict[str, Any],
        *,
        fixed_response_time: bool,
    ) -> None:
        if not build_parameter_evidence:
            return
        predictive_error = predictive_error_from_dict(
            model_report["validation"]["predictive_error"]
        )
        if not isinstance(
            predictive_error,
            EmpiricalHorizonPredictiveError,
        ):
            evidence = UnavailableParameterEvidence(
                "held-out fixed-horizon residual covariance is unavailable"
            )
        else:
            fitted_mask = fitted_structured_parameter_mask(
                selected_params,
                fixed_response_time=fixed_response_time,
                learn_thrust_command_offset=learn_thrust_command_offset,
                instantaneous_rotational_response=(
                    instantaneous_rotational_response
                ),
                diagonal_angular_control=diagonal_angular_control,
            )
            evidence = estimate_local_parameter_information(
                selected_params,
                window_sets,
                predictive_error,
                evidence_groups,
                fitted_parameter_mask=fitted_mask,
                independence_unit=evidence_independence_unit,
            )
        model_report["parameter_evidence"] = evidence.to_dict()

    attach_parameter_evidence(
        learned_params,
        learned_report,
        fixed_response_time=False,
    )

    baseline_params = None
    baseline_report = None
    comparison = None
    if run_no_lag_ablation:
        baseline_params, baseline_report = _fit_on_windows(
            fitting_windows,
            steps=steps,
            learning_rate=learning_rate,
            fixed_motor_time_constant_s=0.0001,
            horizon_labels=training_horizon_labels,
            model_class=model_class,
            platform=platform,
            endpoint_weight=endpoint_weight,
            stability_regularization=stability_regularization,
            learn_thrust_command_offset=learn_thrust_command_offset,
            instantaneous_rotational_response=instantaneous_rotational_response,
            diagonal_angular_control=diagonal_angular_control,
            normalization_windows=normalization_fitting_windows,
        )
        baseline_report["observation_identification"] = (
            None if observation_fit is None else observation_fit.report
        )
        baseline_report["validation"] = _evaluate_model(
            baseline_params,
            validation,
            horizon_seconds=evaluation_horizons_s,
        )
        attach_parameter_evidence(
            baseline_params,
            baseline_report,
            fixed_response_time=True,
        )
        comparison = _comparison_report(
            learned_report["validation"], baseline_report["validation"]
        )

    controls = np.concatenate([trajectory.controls for trajectory in training])

    def group_weight_shares(windows: TrajectoryWindows) -> dict[str, float]:
        if training_source_groups is None:
            return {}
        weights = (
            windows.window_weights
            if windows.window_weights is not None
            else np.ones(len(windows.initial_states))
        )
        total = float(np.sum(weights))
        return {
            str(group): float(
                np.sum(
                    weights[
                        np.isin(
                            windows.trajectory_indices,
                            [
                                index
                                for index, value in enumerate(training_source_groups)
                                if value == group
                            ],
                        )
                    ]
                )
                / total
            )
            for group in training_group_order
        }

    models: dict[str, Any] = {"learned_lag": learned_report}
    if baseline_report is not None:
        models["no_lag"] = baseline_report
    report = {
        "format_version": 1,
        "dataset": dataset_contract,
        "split": {
            "mode": split_mode,
            "independent_source_group_holdout": bool(training_source_groups)
            and set(training_source_groups).isdisjoint(
                flight.source_group
                for flight in validation
                if flight.source_group is not None
            ),
            "held_out_profiles": (
                list(dict.fromkeys(holdout_profiles)) if holdout_profiles else []
            ),
            "training_source_groups": training_group_order,
            "validation_source_groups": list(
                dict.fromkeys(
                    flight.source_group
                    for flight in validation
                    if flight.source_group is not None
                )
            ),
            "training_flights": [
                _trajectory_summary(label, trajectory)
                for label, trajectory in zip(training_labels, training)
            ],
            "validation_flights": [
                _trajectory_summary(flight.path, flight.trajectory)
                for flight in validation
            ],
        },
        "configuration": {
            "train_fraction_for_single_flight": train_fraction,
            "holdout_count": len(validation),
            "holdout_source_group_count": len(
                {
                    flight.source_group
                    for flight in validation
                    if flight.source_group is not None
                }
            ),
            "horizon_steps": max(training_horizon_steps),
            "horizon_duration_s": max(training_horizon_steps) * dt_s,
            "training_horizon_steps": list(training_horizon_steps),
            "training_horizons_s": [
                steps_at_horizon * dt_s
                for steps_at_horizon in training_horizon_steps
            ],
            "stride_steps_by_horizon": {
                label: steps_at_horizon if stride is None else stride
                for label, steps_at_horizon in zip(
                    training_horizon_labels, training_horizon_steps
                )
            },
            "control_history_duration_s": (
                window_sets[0].control_histories.shape[1] * dt_s
            ),
            "motor_history_duration_s": (
                window_sets[0].control_histories.shape[1] * dt_s
                if platform == "multirotor"
                else None
            ),
            "optimization_steps_per_model": steps,
            "learning_rate": learning_rate,
            "endpoint_weight": endpoint_weight,
            "stability_regularization": stability_regularization,
            "multirotor_thrust_command_offset": (
                "not_applicable_fixedwing"
                if platform != "multirotor"
                else "learned"
                if learn_thrust_command_offset
                else "fixed_zero_reference"
            ),
            "rotational_response": (
                "not_applicable_fixedwing"
                if platform != "multirotor"
                else "instantaneous_diagonal_reference"
                if instantaneous_rotational_response
                else "learned_latent_diagonal"
                if diagonal_angular_control
                else "learned_latent_cross_coupled"
            ),
            "training_windows": sum(
                len(windows.initial_states) for windows in window_sets
            ),
            "training_windows_by_horizon": {
                label: len(windows.initial_states)
                for label, windows in zip(training_horizon_labels, window_sets)
            },
            "training_window_selection": {
                "budget_policy": "automatic_corpus_and_horizon",
                "selection_policy_by_horizon": {
                    label: windows.selection_policy
                    for label, windows in zip(training_horizon_labels, window_sets)
                },
                "maximum_windows_by_horizon": {
                    label: maximum_windows
                    for label, maximum_windows in zip(
                        training_horizon_labels, maximum_windows_by_horizon
                    )
                },
                "candidate_windows_by_horizon": {
                    label: windows.candidate_window_count
                    for label, windows in zip(training_horizon_labels, window_sets)
                },
                "selected_windows_by_horizon": {
                    label: len(windows.initial_states)
                    for label, windows in zip(training_horizon_labels, window_sets)
                },
                "selection_fraction_by_horizon": {
                    label: len(windows.initial_states)
                    / windows.candidate_window_count
                    for label, windows in zip(training_horizon_labels, window_sets)
                },
                "source_group_count": training_diversity_count,
                "stratification": (
                    "weighted_source_group"
                    if group_balanced
                    and training_source_group_weights is not None
                    else "source_group"
                    if group_balanced
                    else "weighted_trajectory"
                    if profile_balanced_weights is not None
                    else "trajectory"
                    if balance_training_flights and len(training) > 1
                    else "global_timeline"
                ),
            },
            "evaluation_horizons_s": list(evaluation_horizons_s),
            "no_lag_ablation": run_no_lag_ablation,
            "model_class": model_class,
            "platform": platform,
            "model_family": family.key,
            "training_flight_weighting": (
                "weighted_source_group_then_equal_window"
                if group_balanced
                and training_source_group_weights is not None
                else "equal_source_group_then_equal_window"
                if group_balanced
                else "equal_profile_then_equal_flight"
                if profile_balanced_weights is not None
                else "equal_flight"
                if balance_training_flights
                else "window_count"
            ),
            "training_windows_per_flight_by_horizon": {
                label: {
                    training_labels[index]: int(
                        np.sum(windows.trajectory_indices == index)
                    )
                    for index in range(len(training))
                }
                for label, windows in zip(training_horizon_labels, window_sets)
            },
            "candidate_training_windows_per_flight_by_horizon": {
                label: {
                    training_labels[index]: int(
                        windows.candidate_window_counts[index]
                    )
                    for index in range(len(training))
                }
                for label, windows in zip(training_horizon_labels, window_sets)
            },
            "training_weight_share_per_flight_by_horizon": {
                label: {
                    training_labels[index]: float(
                        np.sum(
                            (
                                windows.window_weights
                                if windows.window_weights is not None
                                else np.ones(len(windows.initial_states))
                            )[windows.trajectory_indices == index]
                        )
                        / np.sum(
                            windows.window_weights
                            if windows.window_weights is not None
                            else np.ones(len(windows.initial_states))
                        )
                    )
                    for index in range(len(training))
                }
                for label, windows in zip(training_horizon_labels, window_sets)
            },
            "training_weight_share_per_source_group_by_horizon": {
                label: group_weight_shares(windows)
                for label, windows in zip(training_horizon_labels, window_sets)
            },
            "training_source_group_weights": (
                None
                if training_source_group_weights is None
                else {
                    str(group): float(training_source_group_weights[group])
                    for group in training_group_order
                }
            ),
            "fit_statistics": {
                "policy": (
                    "member_training_windows_v1"
                    if normalization_window_sets is None
                    else "shared_outer_training_windows_v1"
                ),
                "shared_across_resampled_members": bool(
                    normalization_window_sets is not None
                ),
                "normalization_source_group_weights": (
                    None
                    if normalization_source_group_weights is None
                    else {
                        str(group): float(
                            normalization_source_group_weights[group]
                        )
                        for group in training_group_order
                    }
                ),
                "selected_windows_by_horizon": (
                    None
                    if normalization_window_sets is None
                    else {
                        label: len(windows.initial_states)
                        for label, windows in zip(
                            training_horizon_labels,
                            normalization_window_sets,
                        )
                    }
                ),
                "data_derived_values": [
                    "state_error_scales",
                    "dynamic_envelope",
                    "multi_horizon_initial_loss_normalizers",
                    *(
                        [
                            "residual_feature_center_and_scale",
                            "residual_correction_scale",
                        ]
                        if model_class == "structured_residual"
                        else []
                    ),
                ],
            },
            "parameter_evidence": {
                "requested": build_parameter_evidence,
                "method": "grouped_local_rollout_information_v1",
                "maximum_windows_per_horizon": (
                    MAX_PARAMETER_EVIDENCE_WINDOWS_PER_HORIZON
                ),
                "independence_unit": evidence_independence_unit,
                "residual_scale_source": "held_out_tangent_covariance",
            },
        },
        "models": models,
        "comparison": comparison,
        "training_excitation": _excitation_diagnostics(
            controls,
            training[0].control_names,
            training[0].spec.control_roles,
        ),
        "interpretation": (
            "Parameters are effective predictive coefficients. Complete-flight "
            "holdout results test cross-flight generalization; no-lag ratios above "
            "one indicate that latent applied-control response improves "
            "prediction. When "
            "multiple training horizons are supplied, their losses are normalized "
            "by their initial values before being combined with equal weight. "
            "By default each labeled source group contributes equal total loss "
            "weight with uniform window weight inside the group; without source "
            "groups, each training flight contributes equally. Explicit source-"
            "group weights represent complete-group resampling multiplicities. "
            "When shared fit-statistics weights are supplied, state scales, the "
            "stability envelope, multi-horizon loss normalization, and residual "
            "normalization are derived from that fixed outer-training reference "
            "rather than each resampled empirical loss. "
            "Large candidate "
            "sets are deterministically thinned across every group's timeline "
            "using an automatic corpus- and horizon-aware compute budget. For a "
            "structured "
            "residual, frame-invariant feature normalization and six-axis "
            "correction bounds are derived only from the training windows, "
            "kept fixed during fitting, and serialized with the model. Every "
            "model class uses equal semantic state-group loss after scaling by "
            "training-window motion, linearly emphasizes later rollout steps, "
            "and softly penalizes velocity/rate escape beyond a generous "
            "training-derived body-frame envelope. When requested for a model "
            "artifact, local structured-parameter information uses bounded "
            "rollout Jacobians, gives each independent training group one unit "
            "of evidence, averages correlated horizons, and whitens only the "
            "held-out residual subspace supported numerically. Its rank and "
            "group scores are diagnostics, not an inferred parameter covariance."
        ),
    }
    return learned_params, baseline_params, report


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
    return model_path.with_name(
        f"{model_path.stem}_no_motor_lag{model_path.suffix}"
    )


def main() -> None:
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
            "falls back to input trajectories when groups are unlabeled"
        ),
    )
    parser.add_argument(
        "--holdout-profile",
        action="append",
        help=(
            "maneuver profile to reserve completely; repeat for multiple profiles "
            "and supersedes --holdout-count"
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
    args = parser.parse_args()
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
                "--fixed-response-time-constant only supports "
                "--model-class structured"
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
        learned = report["models"]["learned_lag"]
        learned_fit = learned["fit"]
        learned_full = learned["validation"]["aggregate"]["full_rollout"]
        validation_label = (
            "held-out complete-source rollout"
            if report["split"]["mode"]
            in {
                "leave_complete_flights_out",
                "leave_profiles_out",
                "leave_source_groups_out",
            }
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
        training_paths = [
            item["path"] for item in report["split"]["training_flights"]
        ]
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
                report["models"]["learned_lag"]["validation"][
                    "predictive_error"
                ]
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
                        report["models"]["no_lag"]["validation"][
                            "predictive_error"
                        ]
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
