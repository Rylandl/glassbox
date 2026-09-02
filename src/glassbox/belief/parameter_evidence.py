"""Rank-aware local parameter information from grouped rollout sensitivities."""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
import numpy as np

from glassbox.belief.belief import (
    EmpiricalHorizonPredictiveError,
    LocalParameterInformation,
    ParameterEvidence,
    UnavailableParameterEvidence,
    structured_parameter_names,
    structured_parameter_vector,
)
from glassbox.core.covariance import supported_covariance
from glassbox.core.data import TrajectoryWindows
from glassbox.core.dynamics import (
    FixedWingDynamicsParams,
    ModelParams,
    has_instantaneous_rotational_response,
    model_family,
    structured_parameters,
)
from glassbox.core.linearization import compiled_batched_endpoint_tangent_linearization

MAX_PARAMETER_EVIDENCE_WINDOWS_PER_HORIZON = 96
_FLOAT32_EPSILON = float(np.finfo(np.float32).eps)


def fitted_structured_parameter_mask(
    params: ModelParams,
    *,
    fixed_response_time: bool = False,
    learn_thrust_command_offset: bool = False,
    instantaneous_rotational_response: bool | None = None,
    diagonal_angular_control: bool = False,
) -> np.ndarray:
    """Return the structured coordinates actually varied by the fitter."""

    names = structured_parameter_names(params)
    fitted = np.ones(len(names), dtype=bool)
    platform = model_family(params).platform
    if instantaneous_rotational_response is None:
        # A model whose rotational-response leaves sit at the memoryless
        # sentinel has an exactly zero gradient there, so the fitter could not
        # have varied them regardless of what the caller intended.
        instantaneous_rotational_response = has_instantaneous_rotational_response(
            params
        )
    for index, name in enumerate(names):
        if fixed_response_time and name in {
            "log_motor_time_constant",
            "log_actuator_time_constant",
        }:
            fitted[index] = False
        if platform != "multirotor":
            continue
        if name == "thrust_command_offset_unconstrained":
            fitted[index] = learn_thrust_command_offset
        if name.startswith("angular_control_cross_coupling_unconstrained["):
            location = name.removeprefix(
                "angular_control_cross_coupling_unconstrained["
            ).removesuffix("]")
            row, column = (int(value) for value in location.split(","))
            if row == column:
                fitted[index] = False
            if instantaneous_rotational_response or diagonal_angular_control:
                fitted[index] = False
        if instantaneous_rotational_response and name.startswith(
            "log_angular_response_time_constant["
        ):
            fitted[index] = False
    return fitted


def structured_parameter_scale(params: ModelParams) -> np.ndarray:
    """Return natural perturbation scales for numerical-rank diagnostics."""

    names = structured_parameter_names(params)
    center = np.asarray(structured_parameter_vector(params), dtype=np.float64)
    scale = np.ones(len(names), dtype=np.float64)
    base = structured_parameters(params)
    if not isinstance(base, FixedWingDynamicsParams):
        return scale
    surface_authority = np.asarray(
        base.physical()["surface_angular_accel_per_speed_sq"],
        dtype=np.float64,
    )
    direct_scales = {
        "lateral_surface_cross_angular_accel_per_speed_sq[0]": (surface_authority[0]),
        "lateral_surface_cross_angular_accel_per_speed_sq[1]": (surface_authority[2]),
        "flap_pitch_angular_accel_per_speed_sq": surface_authority[1],
    }
    for index, name in enumerate(names):
        if name in direct_scales:
            scale[index] = max(
                abs(center[index]),
                float(direct_scales[name]),
                1e-6,
            )
    return scale


def _balanced_window_indices(
    groups: np.ndarray,
    *,
    maximum_windows: int,
) -> np.ndarray:
    group_order = tuple(dict.fromkeys(groups.tolist()))
    budget = min(len(groups), max(maximum_windows, len(group_order)))
    members = [np.flatnonzero(groups == group) for group in group_order]
    allocation = np.zeros(len(members), dtype=np.int64)
    remaining = budget
    while remaining > 0:
        progressed = False
        for index, locations in enumerate(members):
            if allocation[index] >= len(locations):
                continue
            allocation[index] += 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            break
    selected: list[int] = []
    for locations, count in zip(members, allocation):
        if count == 0:
            continue
        ordinals = ((2 * np.arange(count, dtype=np.int64) + 1) * len(locations)) // (
            2 * count
        )
        selected.extend(int(locations[ordinal]) for ordinal in ordinals)
    return np.asarray(sorted(selected), dtype=np.int64)


def estimate_local_parameter_information(
    params: ModelParams,
    window_sets: Sequence[TrajectoryWindows],
    predictive_error: EmpiricalHorizonPredictiveError,
    trajectory_groups: Sequence[str | int],
    *,
    fitted_parameter_mask: np.ndarray | None = None,
    independence_unit: str = "source_group",
) -> ParameterEvidence:
    """Estimate conservative local information without refitting the model.

    Every independent group contributes one unit of information. Windows are
    averaged within horizon and horizons are averaged within group, preventing
    denser logs or longer trajectories from creating fictitious evidence.
    Held-out tangent covariance is inverted only on its numerically supported
    subspace; unresolved residual directions never become near-infinite weight.
    """

    if not window_sets:
        return UnavailableParameterEvidence("no training rollout windows")
    names = structured_parameter_names(params)
    center = np.asarray(structured_parameter_vector(params), dtype=np.float64)
    fitted = (
        np.ones(len(names), dtype=bool)
        if fitted_parameter_mask is None
        else np.asarray(fitted_parameter_mask, dtype=bool)
    )
    if fitted.shape != (len(names),) or not np.any(fitted):
        raise ValueError("fitted parameter mask must select structured coordinates")
    trajectory_groups = tuple(trajectory_groups)
    trajectory_count = len(trajectory_groups)
    if trajectory_count < 1:
        raise ValueError("parameter evidence requires trajectory groups")
    string_groups = tuple(str(group) for group in trajectory_groups)
    if any(not group.strip() for group in string_groups):
        raise ValueError("parameter-evidence groups must be nonempty")
    if len(set(string_groups)) != len(set(trajectory_groups)):
        raise ValueError("parameter-evidence groups need unique string labels")
    if not independence_unit.strip():
        raise ValueError("parameter-evidence independence unit is required")

    group_information: dict[str, list[np.ndarray]] = {}
    group_scores: dict[str, list[np.ndarray]] = {}
    included_horizons: list[float] = []
    window_counts: list[int] = []
    precision_ranks: list[int] = []
    total_observation_rows = 0
    for windows in window_sets:
        if windows.trajectory_indices is None:
            raise ValueError("parameter evidence requires window trajectory indices")
        if (
            len(windows.trajectory_indices)
            and int(np.max(windows.trajectory_indices)) >= trajectory_count
        ):
            raise ValueError("trajectory groups do not cover every window")
        horizon_s = float(windows.controls.shape[1] * windows.dt_s)
        if horizon_s > predictive_error.maximum_horizon_s * (1.0 + 1e-9):
            continue
        bias, covariance = predictive_error.moments(horizon_s)
        supported = supported_covariance(np.asarray(covariance, dtype=np.float64))
        if supported.rank == 0:
            continue
        precision = supported.precision
        precision_rank = supported.rank
        window_groups = np.asarray(
            [string_groups[index] for index in windows.trajectory_indices],
            dtype=object,
        )
        selected = _balanced_window_indices(
            window_groups,
            maximum_windows=MAX_PARAMETER_EVIDENCE_WINDOWS_PER_HORIZON,
        )
        if len(selected) == 0:
            continue
        horizon_steps = windows.controls.shape[1]
        contexts = np.broadcast_to(
            np.asarray(windows.initial_exogenous)[selected, None, :],
            (
                len(selected),
                horizon_steps,
                windows.initial_exogenous.shape[1],
            ),
        )
        biases = np.broadcast_to(
            np.asarray(bias, dtype=np.float64),
            (len(selected), len(bias)),
        )
        errors, jacobians = compiled_batched_endpoint_tangent_linearization(
            jnp.asarray(center),
            params,
            jnp.asarray(windows.initial_states[selected]),
            jnp.asarray(windows.control_histories[selected]),
            jnp.asarray(windows.controls[selected]),
            jnp.asarray(windows.target_states[selected, -1]),
            jnp.asarray(contexts),
            jnp.asarray(biases),
            dt_s=windows.dt_s,
            control_roles=windows.control_roles,
            exogenous_roles=windows.exogenous_roles,
        )
        errors_np = np.asarray(errors, dtype=np.float64)
        jacobians_np = np.asarray(jacobians, dtype=np.float64)
        if not (np.all(np.isfinite(errors_np)) and np.all(np.isfinite(jacobians_np))):
            return UnavailableParameterEvidence(
                f"non-finite rollout linearization at {horizon_s:g}s"
            )
        jacobians_np[..., ~fitted] = 0.0
        per_window_information = np.einsum(
            "wip,ij,wjq->wpq",
            jacobians_np,
            precision,
            jacobians_np,
            optimize=True,
        )
        per_window_scores = np.einsum(
            "wip,ij,wj->wp",
            jacobians_np,
            precision,
            errors_np,
            optimize=True,
        )
        selected_groups = window_groups[selected]
        for group in dict.fromkeys(selected_groups.tolist()):
            group_mask = selected_groups == group
            group_information.setdefault(str(group), []).append(
                np.mean(per_window_information[group_mask], axis=0)
            )
            group_scores.setdefault(str(group), []).append(
                np.mean(per_window_scores[group_mask], axis=0)
            )
        included_horizons.append(horizon_s)
        window_counts.append(len(selected))
        precision_ranks.append(precision_rank)
        total_observation_rows += len(selected) * precision_rank

    if not group_information:
        return UnavailableParameterEvidence(
            "no training horizon had supported held-out predictive-error covariance"
        )
    information = np.sum(
        [np.mean(per_horizon, axis=0) for per_horizon in group_information.values()],
        axis=0,
    )
    information = 0.5 * (information + information.T)
    eigenvalues, eigenvectors = np.linalg.eigh(information)
    information = (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T
    information[~fitted, :] = 0.0
    information[:, ~fitted] = 0.0
    group_labels = tuple(group_information)
    score_vectors = np.asarray(
        [np.mean(group_scores[group], axis=0) for group in group_labels]
    )
    score_vectors[:, ~fitted] = 0.0
    scale = structured_parameter_scale(params)
    rank_relative_tolerance = min(
        0.01,
        max(len(names), total_observation_rows) * _FLOAT32_EPSILON,
    )
    ordered = np.argsort(np.asarray(included_horizons), kind="stable")
    return LocalParameterInformation(
        parameter_names=names,
        center=center,
        information_matrix=information,
        parameter_scale=scale,
        fitted_parameter_mask=fitted,
        horizons_s=tuple(included_horizons[index] for index in ordered),
        window_count_by_horizon=tuple(window_counts[index] for index in ordered),
        residual_precision_rank_by_horizon=tuple(
            precision_ranks[index] for index in ordered
        ),
        group_labels=group_labels,
        group_score_vectors=score_vectors,
        independent_group_count=len(group_information),
        trajectory_count=trajectory_count,
        rank_relative_tolerance=rank_relative_tolerance,
        covariance_scope=predictive_error.covariance_scope,
        source=f"grouped_rollout_jacobians:{independence_unit}",
    )
