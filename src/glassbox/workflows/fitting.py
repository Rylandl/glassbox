"""Fit an effective differentiable dynamics model from a trajectory artifact."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from glassbox.belief.adaptation import endpoint_error_evidence_by_horizon
from glassbox.belief.belief import (
    EmpiricalErrorSample,
    EmpiricalHorizonPredictiveError,
    UnavailableParameterEvidence,
    UnavailablePredictiveError,
    predictive_error_from_dict,
)
from glassbox.belief.parameter_evidence import (
    MAX_PARAMETER_EVIDENCE_WINDOWS_PER_HORIZON,
    estimate_local_parameter_information,
    fitted_structured_parameter_mask,
)
from glassbox.core.data import (
    Trajectory,
    TrajectoryWindows,
    duration_to_steps,
    load_trajectory_npz,
    split_trajectory,
    trajectory_windows,
)
from glassbox.core.dynamics import (
    MOTOR_MIXER,
    ModelParams,
    initial_residual_parameters,
    model_family,
    with_response_time_constant,
)
from glassbox.core.evaluation import (
    aggregate_innovation_diagnostics,
    aggregate_rollout_metrics,
    one_step_innovation_diagnostics,
    parameter_dict,
    rollout_metrics,
)
from glassbox.core.families import (
    MULTIROTOR_FAMILY,
    DynamicsModelFamily,
    family_for_platform,
)
from glassbox.core.fixedwing_synthetic import initial_fixed_wing_parameter_guess
from glassbox.core.identification import (
    MAX_OPTIMIZATION_TRANSITIONS_PER_HORIZON,
    MAX_OPTIMIZATION_WINDOWS_PER_HORIZON,
    OPTIMIZATION_POLICY_VERSION,
    fit_dynamics,
    fit_dynamics_multi_horizon,
    residual_initialization_statistics,
    rollout_loss_configuration,
)
from glassbox.core.synthetic import initial_parameter_guess
from glassbox.workflows.observation_identification import (
    ObservationFitResult,
    fit_multirotor_observations,
)

_MAX_TRAINING_WINDOWS_PER_HORIZON = 8_192
_MAX_TRAINING_TRANSITIONS_PER_HORIZON = 524_288

_MODEL_CLASSES = ("structured", "structured_residual")

INTERPRETATION = (
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
)

SINGLE_FLIGHT_INTERPRETATION = (
    "Parameters are effective predictive coefficients. They are not "
    "independently verified physical mass, inertia, aerodynamic, or "
    "actuator constants."
)


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
    diagonal_angular_control = diagonal_angular_control and platform == "multirotor"
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
                "diverged": fit.diverged,
                "completed_steps": fit.completed_steps,
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
                            else tuple(len(item.initial_states) for item in window_sets)
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
    training, validation = split_trajectory(trajectory, train_fraction=train_fraction)
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
    observation_fit = _observation_fit([training], platform=platform)
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
        "interpretation": SINGLE_FLIGHT_INTERPRETATION,
    }
    return fitted_params, report


@dataclass(frozen=True)
class FitRequest:
    """Every knob that shapes a multi-flight fit, validated on construction.

    The defaults are the library defaults of :func:`fit_trajectory_artifacts`,
    which builds one of these from its keyword arguments.
    """

    train_fraction: float = 0.70
    holdout_count: int = 1
    horizon: int = 25
    stride: int | None = None
    training_horizons_s: tuple[float, ...] | None = None
    steps: int = 400
    learning_rate: float = 0.02
    evaluation_horizons_s: tuple[float, ...] = (0.1, 0.5, 1.0, 2.0)
    run_no_lag_ablation: bool = True
    balance_training_flights: bool = True
    holdout_profiles: tuple[str, ...] | None = None
    training_source_group_weights: Mapping[str | int, float] | None = None
    normalization_source_group_weights: Mapping[str | int, float] | None = None
    model_class: str = "structured"
    endpoint_weight: float = 3.0
    stability_regularization: float = 0.01
    learn_thrust_command_offset: bool = False
    instantaneous_rotational_response: bool = True
    diagonal_angular_control: bool = True
    build_parameter_evidence: bool = False
    respect_benchmark_split: bool = True

    def __post_init__(self) -> None:
        for name in ("evaluation_horizons_s", "training_horizons_s"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, tuple(value))
        if self.holdout_profiles is not None:
            object.__setattr__(self, "holdout_profiles", tuple(self.holdout_profiles))

        if any(seconds <= 0.0 for seconds in self.evaluation_horizons_s):
            raise ValueError("evaluation horizons must be positive")
        if self.training_horizons_s is not None and any(
            seconds <= 0.0 for seconds in self.training_horizons_s
        ):
            raise ValueError("training horizons must be positive")
        if self.model_class not in _MODEL_CLASSES:
            raise ValueError("model_class must be structured or structured_residual")
        if self.endpoint_weight < 1.0:
            raise ValueError("endpoint_weight must be at least one")
        if self.stability_regularization < 0.0:
            raise ValueError("stability_regularization must be nonnegative")
        if self.training_source_group_weights is not None:
            weights = np.asarray(
                list(self.training_source_group_weights.values()), dtype=np.float64
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
        if self.normalization_source_group_weights is not None and any(
            not np.isfinite(weight) or weight <= 0.0
            for weight in self.normalization_source_group_weights.values()
        ):
            raise ValueError(
                "normalization_source_group_weights values must be finite and positive"
            )
        if (
            self.normalization_source_group_weights is not None
            and self.training_source_group_weights is None
        ):
            raise ValueError(
                "normalization_source_group_weights requires explicit training "
                "source-group weights"
            )

    def stride_for(self, horizon_steps: int) -> int:
        """Return the window stride used at one training horizon."""

        return horizon_steps if self.stride is None else self.stride


@dataclass(frozen=True)
class EvaluationFlight:
    """One held-out flight and the control history that precedes it."""

    path: str
    trajectory: Trajectory
    control_history: np.ndarray | None = None
    source_group: str | int | None = None


class BenchmarkSplitHoldoutConflict(ValueError):
    """Raised when --holdout-count/--holdout-profile conflict with labels.

    Every input trajectory carries a ``labels["benchmark_split"]`` of
    ``"training"`` or ``"validation"``, so the holdout is derived from that
    label rather than from argument order or an explicit holdout request.
    """


_BENCHMARK_SPLIT_HOLDOUT_VALUES = ("training", "validation")


def _benchmark_split_holdout_indices(
    trajectories: Sequence[Trajectory],
) -> tuple[list[int], list[int]] | None:
    """Return (training, validation) indices from upstream split labels.

    Returns ``None`` unless every trajectory carries a
    ``labels["benchmark_split"]`` of exactly ``"training"`` or
    ``"validation"`` and at least one trajectory has each value; callers fall
    back to argument-order-based splitting in every other case.
    """

    splits = [trajectory.labels.get("benchmark_split") for trajectory in trajectories]
    if any(split not in _BENCHMARK_SPLIT_HOLDOUT_VALUES for split in splits):
        return None
    training_indices = [
        index for index, split in enumerate(splits) if split == "training"
    ]
    validation_indices = [
        index for index, split in enumerate(splits) if split == "validation"
    ]
    if not training_indices or not validation_indices:
        return None
    return training_indices, validation_indices


def _source_groups(
    paths: Sequence[Path], trajectories: Sequence[Trajectory]
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
    source_groups = [value for value in values if isinstance(value, (str, int))]
    unique_groups = tuple(dict.fromkeys(source_groups))
    if len({str(group) for group in unique_groups}) != len(unique_groups):
        raise ValueError("source_group labels must have unique string representations")
    return source_groups


@dataclass(frozen=True)
class HoldoutPlan:
    """Which flights train and which are reserved, and why.

    ``mode`` records the rule that produced the split and is copied verbatim
    into ``report["split"]["mode"]``.
    """

    mode: str
    training: tuple[Trajectory, ...]
    training_labels: tuple[str, ...]
    validation: tuple[EvaluationFlight, ...]
    training_source_groups: tuple[str | int, ...] | None = None

    @property
    def training_group_order(self) -> list[str | int]:
        """Distinct training source groups in first-appearance order."""

        if self.training_source_groups is None:
            return []
        return list(dict.fromkeys(self.training_source_groups))

    @property
    def validation_group_order(self) -> list[str | int]:
        """Distinct held-out source groups in first-appearance order."""

        return list(
            dict.fromkeys(
                flight.source_group
                for flight in self.validation
                if flight.source_group is not None
            )
        )


def _default_holdout_paths(count: int) -> list[Path]:
    return [Path(f"trajectory_{index}") for index in range(count)]


def plan_holdout(
    trajectories: Sequence[Trajectory],
    request: FitRequest,
    paths: Sequence[str | Path] | None = None,
) -> HoldoutPlan:
    """Decide the training/validation split without loading or fitting anything.

    Four rules are tried in order, and the first that applies wins:

    1. a single trajectory is split temporally by ``train_fraction``;
    2. ``labels["benchmark_split"]`` on every flight reserves the flights
       labeled ``"validation"``, independent of argument order;
    3. ``holdout_profiles`` reserves every flight in the named profiles;
    4. ``labels["source_group"]`` reserves the last ``holdout_count`` groups;
    5. otherwise the last ``holdout_count`` flights are reserved positionally.

    ``paths`` only supplies the labels used in the report and error messages;
    it defaults to positional placeholders so the planner can be exercised on
    in-memory trajectories.
    """

    if not trajectories:
        raise ValueError("at least one trajectory is required")
    resolved_paths = (
        _default_holdout_paths(len(trajectories))
        if paths is None
        else [Path(path) for path in paths]
    )
    if len(resolved_paths) != len(trajectories):
        raise ValueError("paths must label every trajectory")
    source_groups = _source_groups(resolved_paths, trajectories)

    if request.holdout_profiles and len(trajectories) == 1:
        raise ValueError("profile holdout requires multiple trajectories")

    if len(trajectories) == 1:
        return _temporal_holdout(
            trajectories[0], resolved_paths[0], request, source_groups
        )

    benchmark_split_indices = (
        _benchmark_split_holdout_indices(trajectories)
        if request.respect_benchmark_split
        else None
    )
    if benchmark_split_indices is not None:
        _reject_explicit_holdout_with_benchmark_split(request)
        training_indices, validation_indices = benchmark_split_indices
        mode = "benchmark_split_holdout"
    elif request.holdout_profiles:
        training_indices, validation_indices = _profile_holdout_indices(
            trajectories, resolved_paths, request.holdout_profiles
        )
        mode = "leave_profiles_out"
    elif source_groups is not None:
        training_indices, validation_indices, mode = _source_group_holdout_indices(
            trajectories, source_groups, request.holdout_count
        )
    else:
        if not 1 <= request.holdout_count < len(trajectories):
            raise ValueError(
                "holdout_count must reserve at least one but not all flights"
            )
        training_indices = list(range(len(trajectories) - request.holdout_count))
        validation_indices = list(
            range(len(trajectories) - request.holdout_count, len(trajectories))
        )
        mode = "leave_complete_flights_out"

    def group_of(index: int) -> str | int | None:
        return None if source_groups is None else source_groups[index]

    return HoldoutPlan(
        mode=mode,
        training=tuple(trajectories[index] for index in training_indices),
        training_labels=tuple(str(resolved_paths[index]) for index in training_indices),
        training_source_groups=(
            None
            if source_groups is None
            else tuple(source_groups[index] for index in training_indices)
        ),
        validation=tuple(
            EvaluationFlight(
                path=str(resolved_paths[index]),
                trajectory=trajectories[index],
                source_group=group_of(index),
            )
            for index in validation_indices
        ),
    )


def _temporal_holdout(
    trajectory: Trajectory,
    path: Path,
    request: FitRequest,
    source_groups: list[str | int] | None,
) -> HoldoutPlan:
    training_segment, validation_segment = split_trajectory(
        trajectory, train_fraction=request.train_fraction
    )
    return HoldoutPlan(
        mode="temporal_within_flight",
        training=(training_segment,),
        training_labels=(f"{path}#training",),
        training_source_groups=None,
        validation=(
            EvaluationFlight(
                path=f"{path}#validation",
                trajectory=validation_segment,
                control_history=training_segment.controls,
                source_group=(source_groups[0] if source_groups else None),
            ),
        ),
    )


def _reject_explicit_holdout_with_benchmark_split(request: FitRequest) -> None:
    preamble = (
        "every trajectory carries a benchmark_split label of 'training'/'validation'; "
    )
    suffix = (
        " because the validation split is determined by the label "
        "(pass respect_benchmark_split=False to override)"
    )
    if request.holdout_profiles:
        raise BenchmarkSplitHoldoutConflict(
            f"{preamble}holdout_profiles is not applicable{suffix}"
        )
    if request.holdout_count != 1:
        raise BenchmarkSplitHoldoutConflict(
            f"{preamble}holdout_count is not applicable{suffix}"
        )


def _profile_holdout_indices(
    trajectories: Sequence[Trajectory],
    paths: Sequence[Path],
    holdout_profiles: Sequence[str],
) -> tuple[list[int], list[int]]:
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
    return training_indices, validation_indices


def _source_group_holdout_indices(
    trajectories: Sequence[Trajectory],
    source_groups: Sequence[str | int],
    holdout_count: int,
) -> tuple[list[int], list[int], str]:
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
        return (
            list(range(len(trajectories) - holdout_count)),
            list(range(len(trajectories) - holdout_count, len(trajectories))),
            "chronological_segments_within_source_group_characterization",
        )
    if not 1 <= holdout_count < len(group_order):
        raise ValueError(
            "holdout_count must reserve at least one but not all source groups"
        )
    held_out_groups = set(group_order[-holdout_count:])
    return (
        [
            index
            for index, group in enumerate(source_groups)
            if group not in held_out_groups
        ],
        [
            index
            for index, group in enumerate(source_groups)
            if group in held_out_groups
        ],
        "leave_source_groups_out",
    )


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
            "maximum_speed_m_s": float(np.max(np.linalg.norm(velocity, axis=1))),
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
    spec_payload = consistent_value("trajectory_spec", spec_payloads, serialize=True)

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
                px4.get("actuator_mapping", {}) if isinstance(px4, Mapping) else {}
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
        "control_names": [channel["name"] for channel in spec_payload["controls"]],
        "control_roles": [channel["role"] for channel in spec_payload["controls"]],
        "control_semantics": [
            channel["semantic"] for channel in spec_payload["controls"]
        ],
        "exogenous_size": len(spec_payload["exogenous"]),
        "exogenous_names": [channel["name"] for channel in spec_payload["exogenous"]],
        "exogenous_roles": [channel["role"] for channel in spec_payload["exogenous"]],
        "observation_size": len(spec_payload["observations"]),
        "observation_names": [
            channel["name"] for channel in spec_payload["observations"]
        ],
        "observation_roles": [
            channel["role"] for channel in spec_payload["observations"]
        ],
        "platform": platform,
        "source_type": (source_types[0] if len(source_type_counts) == 1 else "mixed"),
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


@dataclass(frozen=True)
class DatasetResolution:
    """The validated dataset contract plus the family it selects."""

    contract: dict[str, Any]
    platform: str
    family: DynamicsModelFamily
    source_groups: list[str | int] | None


def resolve_dataset(
    paths: list[Path], trajectories: list[Trajectory], request: FitRequest
) -> DatasetResolution:
    """Validate the pooled dataset and resolve the model family it supports."""

    source_groups = _source_groups(paths, trajectories)
    contract = _dataset_contract(paths, trajectories)
    contract["source_group_count"] = (
        len(dict.fromkeys(source_groups))
        if source_groups is not None
        else len(trajectories)
    )
    contract["source_grouping"] = (
        "trajectory_label:source_group"
        if source_groups is not None
        else "one_group_per_trajectory"
    )
    platform = _trajectory_platform(trajectories[0])
    if contract["platform"] is not None:
        platform = str(contract["platform"])
    family = family_for_platform(platform)
    family.validate_control_schema(
        trajectories[0].control_names,
        trajectories[0].spec.control_roles,
    )
    contract["platform"] = platform
    contract["model_family"] = family.key
    if request.model_class == "structured_residual" and not family.supports_residual:
        raise ValueError(
            f"structured_residual is not supported for platform {platform!r}"
        )
    return DatasetResolution(
        contract=contract,
        platform=platform,
        family=family,
        source_groups=source_groups,
    )


def _evaluate_model(
    params: ModelParams,
    flights: Sequence[EvaluationFlight],
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
        for evidence in endpoint_error_evidence_by_horizon(
            params,
            trajectory,
            horizons_s=horizon_seconds,
            source_group=str(
                flight.source_group if flight.source_group is not None else flight.path
            ),
            trajectory_id=flight.path,
        ):
            label = f"{evidence.horizon_s:g}s"
            per_horizon[label] = evidence.window_metrics
            horizon_metrics[label].append(evidence.window_metrics)
            error_samples[evidence.horizon_s].append(evidence.sample)

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
        horizon: samples for horizon, samples in error_samples.items() if samples
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
            "full_rollout": aggregate_rollout_metrics(full_metrics, weighting="equal"),
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


@dataclass(frozen=True)
class TrainingWindows:
    """Rollout windows for every training horizon, plus how they were weighted."""

    dt_s: float
    horizon_steps: tuple[int, ...]
    horizon_labels: tuple[str, ...]
    maximum_windows_by_horizon: tuple[int, ...]
    diversity_count: int
    group_balanced: bool
    profile_balanced_weights: tuple[float, ...] | None
    window_sets: tuple[TrajectoryWindows, ...]
    normalization_window_sets: tuple[TrajectoryWindows, ...] | None = None

    @property
    def fitting_windows(self) -> TrajectoryWindows | tuple[TrajectoryWindows, ...]:
        return self.window_sets[0] if len(self.window_sets) == 1 else self.window_sets

    @property
    def normalization_fitting_windows(
        self,
    ) -> TrajectoryWindows | tuple[TrajectoryWindows, ...] | None:
        sets = self.normalization_window_sets
        if sets is None:
            return None
        return sets[0] if len(sets) == 1 else sets


def _validate_group_weights(plan: HoldoutPlan, request: FitRequest) -> None:
    group_order = plan.training_group_order
    if request.training_source_group_weights is not None:
        if plan.training_source_groups is None:
            raise ValueError(
                "training_source_group_weights requires source_group labels"
            )
        if set(request.training_source_group_weights) != set(group_order):
            raise ValueError(
                "training_source_group_weights must contain exactly the "
                "training source groups"
            )
    if request.normalization_source_group_weights is not None:
        if plan.training_source_groups is None:
            raise ValueError(
                "normalization_source_group_weights requires source_group labels"
            )
        if set(request.normalization_source_group_weights) != set(group_order):
            raise ValueError(
                "normalization_source_group_weights must contain exactly the "
                "training source groups"
            )


def build_training_windows(plan: HoldoutPlan, request: FitRequest) -> TrainingWindows:
    """Extract and weight the rollout windows the optimizer will consume."""

    training = list(plan.training)
    dt_s = training[0].nominal_dt_s
    if request.training_horizons_s is None:
        horizon_steps = (request.horizon,)
    else:
        horizon_steps = tuple(
            dict.fromkeys(
                duration_to_steps(seconds, dt_s)
                for seconds in request.training_horizons_s
            )
        )
    horizon_labels = tuple(f"{steps * dt_s:g}s" for steps in horizon_steps)

    _validate_group_weights(plan, request)

    training_source_groups = (
        None
        if plan.training_source_groups is None
        else list(plan.training_source_groups)
    )
    group_balanced = (
        request.balance_training_flights
        and (len(training) > 1 or request.training_source_group_weights is not None)
        and training_source_groups is not None
        and (
            not request.holdout_profiles
            or request.training_source_group_weights is not None
        )
    )
    training_profiles = [trajectory.labels.get("profile") for trajectory in training]
    profile_balanced_weights = None
    if (
        not group_balanced
        and request.balance_training_flights
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
    diversity_count = (
        len(dict.fromkeys(training_source_groups))
        if training_source_groups is not None
        else len(training)
    )
    maximum_windows_by_horizon = tuple(
        _automatic_training_window_budget(
            horizon_steps=steps,
            source_group_count=diversity_count,
        )
        for steps in horizon_steps
    )
    window_sets = tuple(
        trajectory_windows(
            training,
            horizon=steps,
            stride=request.stride_for(steps),
            balance_trajectories=(
                request.balance_training_flights
                and len(training) > 1
                and not group_balanced
                and profile_balanced_weights is None
            ),
            trajectory_weights=profile_balanced_weights,
            trajectory_groups=(training_source_groups if group_balanced else None),
            trajectory_group_weights=(
                request.training_source_group_weights if group_balanced else None
            ),
            maximum_windows=maximum_windows,
        )
        for steps, maximum_windows in zip(horizon_steps, maximum_windows_by_horizon)
    )
    normalization_window_sets = (
        None
        if request.normalization_source_group_weights is None
        else tuple(
            trajectory_windows(
                training,
                horizon=steps,
                stride=request.stride_for(steps),
                trajectory_groups=training_source_groups,
                trajectory_group_weights=request.normalization_source_group_weights,
                maximum_windows=maximum_windows,
            )
            for steps, maximum_windows in zip(horizon_steps, maximum_windows_by_horizon)
        )
    )
    return TrainingWindows(
        dt_s=dt_s,
        horizon_steps=horizon_steps,
        horizon_labels=horizon_labels,
        maximum_windows_by_horizon=maximum_windows_by_horizon,
        diversity_count=diversity_count,
        group_balanced=group_balanced,
        profile_balanced_weights=profile_balanced_weights,
        window_sets=window_sets,
        normalization_window_sets=normalization_window_sets,
    )


def evidence_independence_unit(plan: HoldoutPlan) -> str:
    """Name the unit each parameter-evidence group represents."""

    if plan.training_source_groups is not None:
        return "source_group"
    if plan.mode == "temporal_within_flight":
        return "temporal_training_segment"
    return "complete_trajectory"


def _window_weights(window_set: TrajectoryWindows) -> np.ndarray:
    if window_set.window_weights is not None:
        return window_set.window_weights
    return np.ones(len(window_set.initial_states))


def _by_horizon(windows: TrainingWindows, render: Any) -> dict[str, Any]:
    """Map every training-horizon label to ``render(window_set)``."""

    return {
        label: render(window_set)
        for label, window_set in zip(windows.horizon_labels, windows.window_sets)
    }


def _split_section(plan: HoldoutPlan, request: FitRequest) -> dict[str, Any]:
    return {
        "mode": plan.mode,
        "independent_source_group_holdout": bool(plan.training_source_groups)
        and set(plan.training_source_groups).isdisjoint(plan.validation_group_order),
        "held_out_profiles": (
            list(dict.fromkeys(request.holdout_profiles))
            if request.holdout_profiles
            else []
        ),
        "training_source_groups": plan.training_group_order,
        "validation_source_groups": plan.validation_group_order,
        "training_flights": [
            _trajectory_summary(label, trajectory)
            for label, trajectory in zip(plan.training_labels, plan.training)
        ],
        "validation_flights": [
            _trajectory_summary(flight.path, flight.trajectory)
            for flight in plan.validation
        ],
        "benchmark_split_holdout": plan.mode == "benchmark_split_holdout",
        "benchmark_split_training": [
            trajectory.labels.get("benchmark_split") for trajectory in plan.training
        ],
        "benchmark_split_validation": [
            flight.trajectory.labels.get("benchmark_split")
            for flight in plan.validation
        ],
    }


def _training_weight_sections(
    plan: HoldoutPlan, windows: TrainingWindows
) -> dict[str, Any]:
    """Per-flight and per-source-group shares of the total training weight."""

    labels = plan.training_labels
    group_order = plan.training_group_order
    training_source_groups = plan.training_source_groups

    def per_flight(window_set: TrajectoryWindows) -> dict[str, int]:
        return {
            labels[index]: int(np.sum(window_set.trajectory_indices == index))
            for index in range(len(labels))
        }

    def candidates(window_set: TrajectoryWindows) -> dict[str, int]:
        return {
            labels[index]: int(window_set.candidate_window_counts[index])
            for index in range(len(labels))
        }

    def flight_shares(window_set: TrajectoryWindows) -> dict[str, float]:
        weights = _window_weights(window_set)
        total = np.sum(weights)
        return {
            labels[index]: float(
                np.sum(weights[window_set.trajectory_indices == index]) / total
            )
            for index in range(len(labels))
        }

    def group_shares(window_set: TrajectoryWindows) -> dict[str, float]:
        if training_source_groups is None:
            return {}
        weights = _window_weights(window_set)
        total = float(np.sum(weights))
        return {
            str(group): float(
                np.sum(
                    weights[
                        np.isin(
                            window_set.trajectory_indices,
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
            for group in group_order
        }

    return {
        "training_windows_per_flight_by_horizon": _by_horizon(windows, per_flight),
        "candidate_training_windows_per_flight_by_horizon": _by_horizon(
            windows, candidates
        ),
        "training_weight_share_per_flight_by_horizon": _by_horizon(
            windows, flight_shares
        ),
        "training_weight_share_per_source_group_by_horizon": _by_horizon(
            windows, group_shares
        ),
    }


def _window_selection_section(
    plan: HoldoutPlan, windows: TrainingWindows, request: FitRequest
) -> dict[str, Any]:
    return {
        "budget_policy": "automatic_corpus_and_horizon",
        "selection_policy_by_horizon": _by_horizon(
            windows, lambda window_set: window_set.selection_policy
        ),
        "maximum_windows_by_horizon": dict(
            zip(windows.horizon_labels, windows.maximum_windows_by_horizon)
        ),
        "candidate_windows_by_horizon": _by_horizon(
            windows, lambda window_set: window_set.candidate_window_count
        ),
        "selected_windows_by_horizon": _by_horizon(
            windows, lambda window_set: len(window_set.initial_states)
        ),
        "selection_fraction_by_horizon": _by_horizon(
            windows,
            lambda window_set: (
                len(window_set.initial_states) / window_set.candidate_window_count
            ),
        ),
        "source_group_count": windows.diversity_count,
        "stratification": (
            "weighted_source_group"
            if windows.group_balanced
            and request.training_source_group_weights is not None
            else "source_group"
            if windows.group_balanced
            else "weighted_trajectory"
            if windows.profile_balanced_weights is not None
            else "trajectory"
            if request.balance_training_flights and len(plan.training) > 1
            else "global_timeline"
        ),
    }


def _fit_statistics_section(
    plan: HoldoutPlan, windows: TrainingWindows, request: FitRequest
) -> dict[str, Any]:
    normalization_window_sets = windows.normalization_window_sets
    group_order = plan.training_group_order
    weights = request.normalization_source_group_weights
    return {
        "policy": (
            "member_training_windows_v1"
            if normalization_window_sets is None
            else "shared_outer_training_windows_v1"
        ),
        "shared_across_resampled_members": bool(normalization_window_sets is not None),
        "normalization_source_group_weights": (
            None
            if weights is None
            else {str(group): float(weights[group]) for group in group_order}
        ),
        "selected_windows_by_horizon": (
            None
            if normalization_window_sets is None
            else {
                label: len(window_set.initial_states)
                for label, window_set in zip(
                    windows.horizon_labels, normalization_window_sets
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
                if request.model_class == "structured_residual"
                else []
            ),
        ],
    }


def _configuration_section(
    *,
    request: FitRequest,
    dataset: DatasetResolution,
    plan: HoldoutPlan,
    windows: TrainingWindows,
    independence_unit: str,
) -> dict[str, Any]:
    dt_s = windows.dt_s
    platform = dataset.platform
    group_weights = request.training_source_group_weights
    return {
        "train_fraction_for_single_flight": request.train_fraction,
        "holdout_count": len(plan.validation),
        "holdout_source_group_count": len(
            {
                flight.source_group
                for flight in plan.validation
                if flight.source_group is not None
            }
        ),
        "horizon_steps": max(windows.horizon_steps),
        "horizon_duration_s": max(windows.horizon_steps) * dt_s,
        "training_horizon_steps": list(windows.horizon_steps),
        "training_horizons_s": [steps * dt_s for steps in windows.horizon_steps],
        "stride_steps_by_horizon": {
            label: request.stride_for(steps)
            for label, steps in zip(windows.horizon_labels, windows.horizon_steps)
        },
        "control_history_duration_s": (
            windows.window_sets[0].control_histories.shape[1] * dt_s
        ),
        "motor_history_duration_s": (
            windows.window_sets[0].control_histories.shape[1] * dt_s
            if platform == "multirotor"
            else None
        ),
        "optimization_steps_per_model": request.steps,
        "learning_rate": request.learning_rate,
        "endpoint_weight": request.endpoint_weight,
        "stability_regularization": request.stability_regularization,
        "multirotor_thrust_command_offset": (
            "not_applicable_fixedwing"
            if platform != "multirotor"
            else "learned"
            if request.learn_thrust_command_offset
            else "fixed_zero_reference"
        ),
        "rotational_response": (
            "not_applicable_fixedwing"
            if platform != "multirotor"
            else "instantaneous_diagonal_reference"
            if request.instantaneous_rotational_response
            else "learned_latent_diagonal"
            if request.diagonal_angular_control
            else "learned_latent_cross_coupled"
        ),
        "training_windows": sum(
            len(window_set.initial_states) for window_set in windows.window_sets
        ),
        "training_windows_by_horizon": _by_horizon(
            windows, lambda window_set: len(window_set.initial_states)
        ),
        "training_window_selection": _window_selection_section(plan, windows, request),
        "evaluation_horizons_s": list(request.evaluation_horizons_s),
        "no_lag_ablation": request.run_no_lag_ablation,
        "model_class": request.model_class,
        "platform": platform,
        "model_family": dataset.family.key,
        "training_flight_weighting": (
            "weighted_source_group_then_equal_window"
            if windows.group_balanced and group_weights is not None
            else "equal_source_group_then_equal_window"
            if windows.group_balanced
            else "equal_profile_then_equal_flight"
            if windows.profile_balanced_weights is not None
            else "equal_flight"
            if request.balance_training_flights
            else "window_count"
        ),
        **_training_weight_sections(plan, windows),
        "training_source_group_weights": (
            None
            if group_weights is None
            else {
                str(group): float(group_weights[group])
                for group in plan.training_group_order
            }
        ),
        "fit_statistics": _fit_statistics_section(plan, windows, request),
        "parameter_evidence": {
            "requested": request.build_parameter_evidence,
            "method": "grouped_local_rollout_information_v1",
            "maximum_windows_per_horizon": (MAX_PARAMETER_EVIDENCE_WINDOWS_PER_HORIZON),
            "independence_unit": independence_unit,
            "residual_scale_source": "held_out_tangent_covariance",
        },
    }


def build_fit_report(
    *,
    request: FitRequest,
    dataset: DatasetResolution,
    plan: HoldoutPlan,
    windows: TrainingWindows,
    models: dict[str, Any],
    comparison: dict[str, Any] | None,
) -> dict[str, Any]:
    """Assemble the JSON-compatible multi-flight fit report."""

    controls = np.concatenate([trajectory.controls for trajectory in plan.training])
    return {
        "format_version": 1,
        "dataset": dataset.contract,
        "split": _split_section(plan, request),
        "configuration": _configuration_section(
            request=request,
            dataset=dataset,
            plan=plan,
            windows=windows,
            independence_unit=evidence_independence_unit(plan),
        ),
        "models": models,
        "comparison": comparison,
        "training_excitation": _excitation_diagnostics(
            controls,
            plan.training[0].control_names,
            plan.training[0].spec.control_roles,
        ),
        "interpretation": INTERPRETATION,
    }


def _parameter_evidence(
    params: ModelParams,
    model_report: dict[str, Any],
    *,
    plan: HoldoutPlan,
    windows: TrainingWindows,
    request: FitRequest,
    fixed_response_time: bool,
) -> dict[str, Any]:
    predictive_error = predictive_error_from_dict(
        model_report["validation"]["predictive_error"]
    )
    if not isinstance(predictive_error, EmpiricalHorizonPredictiveError):
        return UnavailableParameterEvidence(
            "held-out fixed-horizon residual covariance is unavailable"
        ).to_dict()
    fitted_mask = fitted_structured_parameter_mask(
        params,
        fixed_response_time=fixed_response_time,
        learn_thrust_command_offset=request.learn_thrust_command_offset,
        instantaneous_rotational_response=request.instantaneous_rotational_response,
        diagonal_angular_control=request.diagonal_angular_control,
    )
    groups = (
        list(plan.training_source_groups)
        if plan.training_source_groups is not None
        else list(plan.training_labels)
    )
    return estimate_local_parameter_information(
        params,
        windows.window_sets,
        predictive_error,
        groups,
        fitted_parameter_mask=fitted_mask,
        independence_unit=evidence_independence_unit(plan),
    ).to_dict()


def fit_from_request(
    trajectory_paths: Sequence[str | Path],
    request: FitRequest,
) -> tuple[ModelParams, ModelParams | None, dict[str, Any]]:
    """Fit across flights for an explicit :class:`FitRequest`.

    This is the coordinator: it loads, validates the pooled dataset, plans the
    holdout, extracts training windows, fits the learned-lag model and its
    optional no-lag ablation, and hands the pieces to ``build_fit_report``.
    """

    if not trajectory_paths:
        raise ValueError("at least one trajectory path is required")
    paths = [Path(path) for path in trajectory_paths]
    trajectories = [load_trajectory_npz(path) for path in paths]
    dataset = resolve_dataset(paths, trajectories, request)
    plan = plan_holdout(trajectories, request, paths)
    windows = build_training_windows(plan, request)
    observation_fit = _observation_fit(list(plan.training), platform=dataset.platform)

    def fit_model(
        *, fixed_motor_time_constant_s: float | None, fixed_response_time: bool
    ) -> tuple[ModelParams, dict[str, Any]]:
        params, model_report = _fit_on_windows(
            windows.fitting_windows,
            steps=request.steps,
            learning_rate=request.learning_rate,
            fixed_motor_time_constant_s=fixed_motor_time_constant_s,
            horizon_labels=windows.horizon_labels,
            model_class=request.model_class,
            platform=dataset.platform,
            endpoint_weight=request.endpoint_weight,
            stability_regularization=request.stability_regularization,
            learn_thrust_command_offset=request.learn_thrust_command_offset,
            instantaneous_rotational_response=request.instantaneous_rotational_response,
            diagonal_angular_control=request.diagonal_angular_control,
            normalization_windows=windows.normalization_fitting_windows,
        )
        model_report["observation_identification"] = (
            None if observation_fit is None else observation_fit.report
        )
        model_report["validation"] = _evaluate_model(
            params,
            plan.validation,
            horizon_seconds=request.evaluation_horizons_s,
        )
        if request.build_parameter_evidence:
            model_report["parameter_evidence"] = _parameter_evidence(
                params,
                model_report,
                plan=plan,
                windows=windows,
                request=request,
                fixed_response_time=fixed_response_time,
            )
        return params, model_report

    learned_params, learned_report = fit_model(
        fixed_motor_time_constant_s=None, fixed_response_time=False
    )
    models: dict[str, Any] = {"learned_lag": learned_report}

    baseline_params: ModelParams | None = None
    comparison = None
    if request.run_no_lag_ablation:
        baseline_params, baseline_report = fit_model(
            fixed_motor_time_constant_s=0.0001, fixed_response_time=True
        )
        models["no_lag"] = baseline_report
        comparison = _comparison_report(
            learned_report["validation"], baseline_report["validation"]
        )

    report = build_fit_report(
        request=request,
        dataset=dataset,
        plan=plan,
        windows=windows,
        models=models,
        comparison=comparison,
    )
    return learned_params, baseline_params, report


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
    normalization_source_group_weights: (Mapping[str | int, float] | None) = None,
    model_class: str = "structured",
    endpoint_weight: float = 3.0,
    stability_regularization: float = 0.01,
    learn_thrust_command_offset: bool = False,
    instantaneous_rotational_response: bool = True,
    diagonal_angular_control: bool = True,
    build_parameter_evidence: bool = False,
    respect_benchmark_split: bool = True,
) -> tuple[ModelParams, ModelParams | None, dict[str, Any]]:
    """Fit across flights and reserve complete flights when multiple are given.

    When ``respect_benchmark_split`` is true (the default) and every input
    trajectory carries a ``labels["benchmark_split"]`` of ``"training"`` or
    ``"validation"``, the flights labeled ``"validation"`` are reserved
    regardless of ``trajectory_paths`` order or ``holdout_count``/
    ``holdout_profiles``; passing either of those explicitly in that case
    raises :class:`BenchmarkSplitHoldoutConflict`. Positional,
    argument-order-based holdout selection applies only when the label is
    absent from at least one trajectory.

    Every keyword is a field of :class:`FitRequest`; callers holding a request
    already can use :func:`fit_from_request` instead.
    """

    if not trajectory_paths:
        raise ValueError("at least one trajectory path is required")
    request = FitRequest(
        train_fraction=train_fraction,
        holdout_count=holdout_count,
        horizon=horizon,
        stride=stride,
        training_horizons_s=training_horizons_s,
        steps=steps,
        learning_rate=learning_rate,
        evaluation_horizons_s=evaluation_horizons_s,
        run_no_lag_ablation=run_no_lag_ablation,
        balance_training_flights=balance_training_flights,
        holdout_profiles=holdout_profiles,
        training_source_group_weights=training_source_group_weights,
        normalization_source_group_weights=normalization_source_group_weights,
        model_class=model_class,
        endpoint_weight=endpoint_weight,
        stability_regularization=stability_regularization,
        learn_thrust_command_offset=learn_thrust_command_offset,
        instantaneous_rotational_response=instantaneous_rotational_response,
        diagonal_angular_control=diagonal_angular_control,
        build_parameter_evidence=build_parameter_evidence,
        respect_benchmark_split=respect_benchmark_split,
    )
    return fit_from_request(trajectory_paths, request)
