"""Gradient-based identification through batches of multi-step rollouts."""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from jax import Array

from glassbox.core.data import (
    NORMALIZED_MOTOR_COMMAND_SEMANTICS,
    PHYSICAL_MOTOR_THRUST_SEMANTICS,
    TrajectoryWindows,
)
from glassbox.core.dynamics import (
    ModelParams,
    ResidualDynamicsParams,
    control_state_after_history,
    has_instantaneous_rotational_response,
    model_family,
    quaternion_to_rotation,
    rollout_with_latent,
    validate_control_schema,
    with_diagonal_angular_control,
    with_instantaneous_rotational_response,
    with_response_time_constant,
    with_thrust_command_offset,
    zero_angular_cross_coupling_gradient,
    zero_residual_configuration_gradient,
    zero_response_time_gradient,
    zero_rotational_response_gradient,
    zero_thrust_command_offset_gradient,
)

OPTIMIZATION_POLICY_VERSION = "deterministic_weighted_minibatch_v3"
MAX_OPTIMIZATION_WINDOWS_PER_HORIZON = 8_192
MAX_OPTIMIZATION_TRANSITIONS_PER_HORIZON = 65_536


@dataclass(frozen=True)
class RolloutLossConfiguration:
    """Training-only scales and stability envelope for rigid-body rollouts."""

    position_scale_m: npt.NDArray[np.float64]
    velocity_scale_m_s: npt.NDArray[np.float64]
    attitude_scale_rad: float
    angular_velocity_scale_rad_s: npt.NDArray[np.float64]
    body_velocity_center_m_s: npt.NDArray[np.float64]
    body_velocity_bound_m_s: npt.NDArray[np.float64]
    angular_velocity_center_rad_s: npt.NDArray[np.float64]
    angular_velocity_bound_rad_s: npt.NDArray[np.float64]
    endpoint_weight: float = 3.0
    stability_regularization: float = 0.01

    def __post_init__(self) -> None:
        for name in (
            "position_scale_m",
            "velocity_scale_m_s",
            "angular_velocity_scale_rad_s",
            "body_velocity_center_m_s",
            "body_velocity_bound_m_s",
            "angular_velocity_center_rad_s",
            "angular_velocity_bound_rad_s",
        ):
            values = np.asarray(getattr(self, name), dtype=np.float64)
            if values.shape != (3,) or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must contain three finite values")
            object.__setattr__(self, name, values)
        for name in (
            "position_scale_m",
            "velocity_scale_m_s",
            "angular_velocity_scale_rad_s",
            "body_velocity_bound_m_s",
            "angular_velocity_bound_rad_s",
        ):
            if np.any(getattr(self, name) <= 0.0):
                raise ValueError(f"{name} must be positive")
        if not np.isfinite(self.attitude_scale_rad) or self.attitude_scale_rad <= 0.0:
            raise ValueError("attitude_scale_rad must be finite and positive")
        if not np.isfinite(self.endpoint_weight) or self.endpoint_weight < 1.0:
            raise ValueError("endpoint_weight must be finite and at least one")
        if (
            not np.isfinite(self.stability_regularization)
            or self.stability_regularization < 0.0
        ):
            raise ValueError("stability_regularization must be finite and nonnegative")

    def to_dict(self) -> dict[str, object]:
        return {
            "state_error_scales": {
                "position_xyz_m": self.position_scale_m.tolist(),
                "velocity_xyz_m_s": self.velocity_scale_m_s.tolist(),
                "attitude_rad": self.attitude_scale_rad,
                "angular_velocity_xyz_rad_s": (
                    self.angular_velocity_scale_rad_s.tolist()
                ),
            },
            "dynamic_envelope": {
                "body_velocity_center_m_s": (self.body_velocity_center_m_s.tolist()),
                "body_velocity_half_width_m_s": (self.body_velocity_bound_m_s.tolist()),
                "angular_velocity_center_rad_s": (
                    self.angular_velocity_center_rad_s.tolist()
                ),
                "angular_velocity_half_width_rad_s": (
                    self.angular_velocity_bound_rad_s.tolist()
                ),
            },
            "endpoint_weight": self.endpoint_weight,
            "stability_regularization": self.stability_regularization,
            "state_group_weighting": "equal_semantic_groups",
            "time_weighting": "linear_from_one_to_endpoint_weight",
        }


@dataclass(frozen=True)
class FitResult:
    params: ModelParams
    loss_history: npt.NDArray[np.float64]
    component_initial_losses: npt.NDArray[np.float64] | None = None
    component_final_losses: npt.NDArray[np.float64] | None = None
    component_loss_normalizers: npt.NDArray[np.float64] | None = None
    loss_configuration: RolloutLossConfiguration | None = None
    optimization_policy: str = "full_batch_v1"
    batch_sizes: tuple[int, ...] = ()
    window_coverage: tuple[float, ...] = ()
    diverged: bool = False
    completed_steps: int | None = None

    @property
    def initial_loss(self) -> float:
        return float(self.loss_history[0])

    @property
    def final_loss(self) -> float:
        return float(self.loss_history[-1])


def _all_leaves_finite(tree: ModelParams) -> bool:
    """Return whether every leaf of a parameter or gradient tree is finite."""

    return bool(
        jnp.all(
            jnp.stack([jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(tree)])
        )
    )


def _warn_if_rotational_response_frozen(params: ModelParams) -> None:
    """Warn when a fit starts at the memoryless sentinel it can never leave."""

    if has_instantaneous_rotational_response(params):
        warnings.warn(
            "the rotational-response time constants start at the memoryless "
            "sentinel, where their loss gradient is exactly zero, so this fit "
            "cannot learn rotational lag; pass instantaneous_rotational_response="
            "True to declare that mode or start from a positive time constant",
            stacklevel=3,
        )


def deterministic_weighted_batch_schedule(
    window_weights: npt.NDArray[np.float64] | None,
    *,
    window_count: int,
    steps: int,
    maximum_batch_size: int = MAX_OPTIMIZATION_WINDOWS_PER_HORIZON,
) -> npt.NDArray[np.int64]:
    """Return reproducible systematic samples spanning the weighted window set."""

    if window_count < 1:
        raise ValueError("window_count must be positive")
    if steps < 1:
        raise ValueError("steps must be positive")
    if maximum_batch_size < 1:
        raise ValueError("maximum_batch_size must be positive")
    batch_size = min(window_count, maximum_batch_size)
    if window_weights is None:
        probabilities = np.full(window_count, 1.0 / window_count)
    else:
        weights = np.asarray(window_weights, dtype=np.float64)
        if weights.shape != (window_count,):
            raise ValueError("window_weights must match window_count")
        if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
            raise ValueError("window_weights must be finite and positive")
        probabilities = weights / np.sum(weights)
    cumulative = np.cumsum(probabilities)
    cumulative[-1] = 1.0
    base_positions = (np.arange(batch_size, dtype=np.float64) + 0.5) / batch_size
    golden_fraction = 0.6180339887498949
    phases = np.mod(np.arange(steps, dtype=np.float64) * golden_fraction, 1.0)
    positions = np.mod(phases[:, np.newaxis] + base_positions[np.newaxis, :], 1.0)
    return np.searchsorted(cumulative, positions, side="right").astype(np.int64)


def _optimization_batch_schedules(
    window_sets: tuple[TrajectoryWindows, ...] | list[TrajectoryWindows],
    *,
    steps: int,
) -> tuple[npt.NDArray[np.int64], ...] | None:
    schedules = tuple(
        deterministic_weighted_batch_schedule(
            windows.window_weights,
            window_count=len(windows.initial_states),
            steps=steps,
            maximum_batch_size=min(
                MAX_OPTIMIZATION_WINDOWS_PER_HORIZON,
                max(
                    1,
                    MAX_OPTIMIZATION_TRANSITIONS_PER_HORIZON
                    // windows.controls.shape[1],
                ),
            ),
        )
        for windows in window_sets
    )
    if all(
        schedule.shape[1] == len(windows.initial_states)
        for schedule, windows in zip(schedules, window_sets)
    ):
        return None
    return schedules


def _rotation_matrices(
    quaternion_wxyz: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Return vectorized body-to-world rotations for normalized WXYZ quaternions."""

    quaternion = quaternion_wxyz / np.linalg.norm(
        quaternion_wxyz, axis=-1, keepdims=True
    )
    w, x, y, z = np.moveaxis(quaternion, -1, 0)
    return np.stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape(quaternion.shape[:-1] + (3, 3))


def _robust_axis_scale(
    values: npt.NDArray[np.float64],
    *,
    floor: float,
) -> npt.NDArray[np.float64]:
    """Estimate a Gaussian-equivalent scale without letting quiet axes explode."""

    scale = np.quantile(np.abs(values), 0.90, axis=0) / 1.6448536269514722
    return np.maximum(scale, floor)


def _window_wind_world(windows: TrajectoryWindows) -> npt.NDArray[np.float64]:
    """Return one NWU wind vector for each rollout window."""

    values = np.asarray(windows.initial_exogenous, dtype=np.float64)
    roles = windows.exogenous_roles
    wind = np.zeros((len(values), 3), dtype=np.float64)
    for axis, role in enumerate(("wind_north", "wind_west", "wind_up")):
        if role in roles:
            wind[:, axis] = values[:, roles.index(role)]
    return wind


def _batch_wind_world(exogenous: Array, exogenous_roles: tuple[str, ...]) -> Array:
    """JAX equivalent of :func:`_window_wind_world`."""

    return jnp.stack(
        tuple(
            exogenous[:, exogenous_roles.index(role)]
            if role in exogenous_roles
            else jnp.zeros(exogenous.shape[0])
            for role in ("wind_north", "wind_west", "wind_up")
        ),
        axis=-1,
    )


def rollout_loss_configuration(
    window_sets: tuple[TrajectoryWindows, ...] | list[TrajectoryWindows],
    *,
    endpoint_weight: float = 3.0,
    stability_regularization: float = 0.01,
) -> RolloutLossConfiguration:
    """Derive semantic error scales and a generous dynamic envelope.

    Error scales use within-window state changes, avoiding sensitivity to world
    origins or steady cruise offsets. The stability envelope is expressed in
    body velocity and body angular velocity, making it invariant to world frame
    orientation and usable by every rigid-body vehicle family.
    """

    if not window_sets:
        raise ValueError("at least one window set is required")
    position_changes = []
    velocity_changes = []
    attitude_changes = []
    angular_velocity_changes = []
    body_velocities = []
    angular_velocities = []
    for windows in window_sets:
        states = np.asarray(windows.target_states[:, 1:], dtype=np.float64)
        initial = np.asarray(windows.initial_states, dtype=np.float64)[:, None, :]
        position_changes.append((states[..., 0:3] - initial[..., 0:3]).reshape(-1, 3))
        velocity_changes.append((states[..., 3:6] - initial[..., 3:6]).reshape(-1, 3))
        angular_velocity_changes.append(
            (states[..., 10:13] - initial[..., 10:13]).reshape(-1, 3)
        )
        quaternion = states[..., 6:10]
        initial_quaternion = initial[..., 6:10]
        quaternion = quaternion / np.linalg.norm(quaternion, axis=-1, keepdims=True)
        initial_quaternion = initial_quaternion / np.linalg.norm(
            initial_quaternion, axis=-1, keepdims=True
        )
        quaternion_dot = np.clip(
            np.abs(np.sum(quaternion * initial_quaternion, axis=-1)),
            0.0,
            1.0,
        )
        attitude_changes.append((2.0 * np.arccos(quaternion_dot)).reshape(-1))
        rotations = _rotation_matrices(quaternion)
        wind_world = _window_wind_world(windows)[:, None, :]
        body_velocities.append(
            np.einsum(
                "...ji,...j->...i",
                rotations,
                states[..., 3:6] - wind_world,
            ).reshape(-1, 3)
        )
        angular_velocities.append(states[..., 10:13].reshape(-1, 3))

    position_change = np.concatenate(position_changes, axis=0)
    velocity_change = np.concatenate(velocity_changes, axis=0)
    attitude_change = np.concatenate(attitude_changes, axis=0)
    angular_velocity_change = np.concatenate(angular_velocity_changes, axis=0)
    body_velocity = np.concatenate(body_velocities, axis=0)
    angular_velocity = np.concatenate(angular_velocities, axis=0)
    body_velocity_center = np.median(body_velocity, axis=0)
    angular_velocity_center = np.median(angular_velocity, axis=0)
    body_velocity_deviation = body_velocity - body_velocity_center
    angular_velocity_deviation = angular_velocity - angular_velocity_center
    body_velocity_scale = _robust_axis_scale(body_velocity_deviation, floor=0.1)
    angular_velocity_envelope_scale = _robust_axis_scale(
        angular_velocity_deviation, floor=0.1
    )
    body_velocity_bound = np.maximum(
        np.quantile(np.abs(body_velocity_deviation), 0.995, axis=0),
        4.0 * body_velocity_scale,
    )
    angular_velocity_bound = np.maximum(
        np.quantile(np.abs(angular_velocity_deviation), 0.995, axis=0),
        4.0 * angular_velocity_envelope_scale,
    )
    attitude_scale = max(
        float(np.quantile(np.abs(attitude_change), 0.90) / 1.6448536269514722),
        0.05,
    )
    return RolloutLossConfiguration(
        position_scale_m=_robust_axis_scale(position_change, floor=0.05),
        velocity_scale_m_s=_robust_axis_scale(velocity_change, floor=0.1),
        attitude_scale_rad=attitude_scale,
        angular_velocity_scale_rad_s=_robust_axis_scale(
            angular_velocity_change, floor=0.1
        ),
        body_velocity_center_m_s=body_velocity_center,
        body_velocity_bound_m_s=body_velocity_bound,
        angular_velocity_center_rad_s=angular_velocity_center,
        angular_velocity_bound_rad_s=angular_velocity_bound,
        endpoint_weight=endpoint_weight,
        stability_regularization=stability_regularization,
    )


def residual_initialization_statistics(
    window_sets: tuple[TrajectoryWindows, ...] | list[TrajectoryWindows],
) -> dict[str, npt.NDArray[np.float64]]:
    """Derive platform-neutral residual normalization from training windows.

    The feature basis is invariant across vehicle families: body velocity,
    body angular velocity, canonical applied controls, then optional typed
    exogenous observations. Correction bounds use robust observed linear and
    angular acceleration ranges. These values are model configuration captured
    in the artifact, not learned benchmark lore.
    """

    if not window_sets:
        raise ValueError("at least one window set is required")
    features = []
    accelerations = []
    expected_control_size = window_sets[0].control_size
    expected_roles = window_sets[0].control_roles
    expected_exogenous_size = window_sets[0].initial_exogenous.shape[1]
    expected_exogenous_roles = window_sets[0].exogenous_roles
    for windows in window_sets:
        if windows.control_size != expected_control_size:
            raise ValueError("residual window sets must share a control size")
        if windows.control_roles != expected_roles:
            raise ValueError("residual window sets must share control roles")
        if windows.initial_exogenous.shape[1] != expected_exogenous_size:
            raise ValueError("residual window sets must share an exogenous size")
        if windows.exogenous_roles != expected_exogenous_roles:
            raise ValueError("residual window sets must share exogenous roles")
        states = np.asarray(windows.target_states[:, :-1], dtype=np.float64)
        next_states = np.asarray(windows.target_states[:, 1:], dtype=np.float64)
        controls = np.asarray(windows.controls, dtype=np.float64)
        rotations = _rotation_matrices(states[..., 6:10])
        wind_world = _window_wind_world(windows)[:, None, :]
        body_velocity = np.einsum(
            "...ji,...j->...i",
            rotations,
            states[..., 3:6] - wind_world,
        )
        exogenous = np.broadcast_to(
            np.asarray(windows.initial_exogenous, dtype=np.float64)[:, None, :],
            states.shape[:2] + (expected_exogenous_size,),
        )
        features.append(
            np.concatenate(
                (body_velocity, states[..., 10:13], controls, exogenous),
                axis=-1,
            ).reshape(-1, 6 + expected_control_size + expected_exogenous_size)
        )
        world_acceleration = (next_states[..., 3:6] - states[..., 3:6]) / windows.dt_s
        body_acceleration = np.einsum("...ji,...j->...i", rotations, world_acceleration)
        angular_acceleration = (
            next_states[..., 10:13] - states[..., 10:13]
        ) / windows.dt_s
        accelerations.append(
            np.concatenate((body_acceleration, angular_acceleration), axis=-1).reshape(
                -1, 6
            )
        )

    feature_values = np.concatenate(features, axis=0)
    acceleration_values = np.concatenate(accelerations, axis=0)
    feature_mean = np.median(feature_values, axis=0)
    feature_scale = np.std(feature_values, axis=0)
    feature_scale = np.where(feature_scale > 1e-6, feature_scale, 1.0)
    acceleration_center = np.median(acceleration_values, axis=0)
    correction_scale = 1.5 * np.quantile(
        np.abs(acceleration_values - acceleration_center), 0.95, axis=0
    )
    correction_scale = np.maximum(correction_scale, 1.0)
    return {
        "feature_mean": feature_mean,
        "feature_scale": feature_scale,
        "correction_scale": correction_scale,
    }


def batch_rollout_loss(
    params: ModelParams,
    initial_states: Array,
    control_histories: Array,
    controls: Array,
    target_states: Array,
    dt_s: float,
    loss_configuration: RolloutLossConfiguration,
    window_weights: Array | None = None,
    control_roles: tuple[str, ...] | None = None,
    initial_exogenous: Array | None = None,
    exogenous_roles: tuple[str, ...] | None = None,
) -> Array:
    """Calculate semantic rollout error plus a soft dynamic-envelope penalty."""

    initial_motor_states = jax.vmap(
        lambda history: control_state_after_history(
            params, history, dt_s, control_roles
        )
    )(control_histories)
    if initial_exogenous is None:
        initial_exogenous = jnp.empty((len(initial_states), 0))
    if exogenous_roles is None:
        exogenous_roles = ()
    predicted, _ = jax.vmap(
        lambda initial, control_sequence, initial_control, context: rollout_with_latent(
            params,
            initial,
            control_sequence,
            dt_s,
            initial_control,
            control_roles,
            context,
            exogenous_roles,
        )
    )(
        initial_states,
        controls,
        initial_motor_states,
        initial_exogenous,
    )
    predicted = predicted[:, 1:, :]
    target_states = target_states[:, 1:, :]
    position_error = (predicted[..., 0:3] - target_states[..., 0:3]) / jnp.asarray(
        loss_configuration.position_scale_m
    )
    velocity_error = (predicted[..., 3:6] - target_states[..., 3:6]) / jnp.asarray(
        loss_configuration.velocity_scale_m_s
    )
    angular_velocity_error = (
        predicted[..., 10:13] - target_states[..., 10:13]
    ) / jnp.asarray(loss_configuration.angular_velocity_scale_rad_s)
    predicted_quaternion = predicted[..., 6:10]
    target_quaternion = target_states[..., 6:10]
    predicted_quaternion = predicted_quaternion / jnp.linalg.norm(
        predicted_quaternion, axis=-1, keepdims=True
    )
    target_quaternion = target_quaternion / jnp.linalg.norm(
        target_quaternion, axis=-1, keepdims=True
    )
    quaternion_dot = jnp.clip(
        jnp.abs(jnp.sum(predicted_quaternion * target_quaternion, axis=-1)),
        0.0,
        1.0,
    )
    attitude_squared_error = (
        8.0 * (1.0 - quaternion_dot) / loss_configuration.attitude_scale_rad**2
    )
    per_step_error = 0.25 * (
        jnp.mean(jnp.square(position_error), axis=-1)
        + jnp.mean(jnp.square(velocity_error), axis=-1)
        + attitude_squared_error
        + jnp.mean(jnp.square(angular_velocity_error), axis=-1)
    )
    time_weights = jnp.linspace(
        1.0,
        loss_configuration.endpoint_weight,
        per_step_error.shape[1],
    )
    per_window_error = jnp.sum(
        per_step_error * time_weights[jnp.newaxis, :], axis=1
    ) / jnp.sum(time_weights)

    per_step_stability = dynamic_envelope_penalty(
        predicted,
        loss_configuration,
        initial_exogenous,
        exogenous_roles,
    )
    per_window_stability = jnp.sum(
        per_step_stability * time_weights[jnp.newaxis, :], axis=1
    ) / jnp.sum(time_weights)
    per_window_loss = per_window_error + (
        loss_configuration.stability_regularization * per_window_stability
    )
    if window_weights is None:
        return jnp.mean(per_window_loss)
    return jnp.sum(per_window_loss * window_weights) / jnp.sum(window_weights)


def dynamic_envelope_penalty(
    predicted_states: Array,
    loss_configuration: RolloutLossConfiguration,
    initial_exogenous: Array | None = None,
    exogenous_roles: tuple[str, ...] | None = None,
) -> Array:
    """Return per-step soft penalties outside the training dynamic envelope."""

    predicted_quaternion = predicted_states[..., 6:10]
    predicted_quaternion = predicted_quaternion / jnp.linalg.norm(
        predicted_quaternion, axis=-1, keepdims=True
    )
    rotations = jax.vmap(jax.vmap(quaternion_to_rotation))(predicted_quaternion)
    if initial_exogenous is None:
        initial_exogenous = jnp.empty((predicted_states.shape[0], 0))
    if exogenous_roles is None:
        exogenous_roles = ()
    wind_world = _batch_wind_world(initial_exogenous, exogenous_roles)
    predicted_body_velocity = jnp.einsum(
        "...ji,...j->...i",
        rotations,
        predicted_states[..., 3:6] - wind_world[:, None, :],
    )
    body_velocity_bound = jnp.asarray(loss_configuration.body_velocity_bound_m_s)
    angular_velocity_bound = jnp.asarray(
        loss_configuration.angular_velocity_bound_rad_s
    )
    body_velocity_excess = (
        jax.nn.relu(
            jnp.abs(
                predicted_body_velocity
                - jnp.asarray(loss_configuration.body_velocity_center_m_s)
            )
            - body_velocity_bound
        )
        / body_velocity_bound
    )
    angular_velocity_excess = (
        jax.nn.relu(
            jnp.abs(
                predicted_states[..., 10:13]
                - jnp.asarray(loss_configuration.angular_velocity_center_rad_s)
            )
            - angular_velocity_bound
        )
        / angular_velocity_bound
    )
    per_step_stability = 0.5 * (
        jnp.mean(jnp.square(body_velocity_excess), axis=-1)
        + jnp.mean(jnp.square(angular_velocity_excess), axis=-1)
    )
    return per_step_stability


def _fit_objective(
    objective: Callable[[ModelParams], Array],
    component_objective: Callable[[ModelParams], Array],
    initial_params: ModelParams,
    *,
    steps: int,
    learning_rate: float,
    gradient_clip_norm: float,
    fixed_motor_time_constant: bool,
    fixed_thrust_command_offset: bool,
    fixed_rotational_response: bool = False,
    fixed_angular_cross_coupling: bool = False,
    loss_configuration: RolloutLossConfiguration,
    batch_objective: Callable[[ModelParams, tuple[Array, ...]], Array] | None = None,
    batch_schedules: tuple[npt.NDArray[np.int64], ...] | None = None,
    batch_window_counts: tuple[int, ...] | None = None,
) -> FitResult:
    if (batch_objective is None) != (batch_schedules is None):
        raise ValueError(
            "batch_objective and batch_schedules must be supplied together"
        )
    if batch_schedules is not None and (
        batch_window_counts is None or len(batch_window_counts) != len(batch_schedules)
    ):
        raise ValueError("batch_window_counts must match supplied batch schedules")
    value_and_grad = jax.jit(
        jax.value_and_grad(objective if batch_objective is None else batch_objective)
    )
    component_values = jax.jit(component_objective)
    params = initial_params
    first_moment = jax.tree.map(jnp.zeros_like, params)
    second_moment = jax.tree.map(jnp.zeros_like, params)
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    history = np.empty(steps + 1, dtype=np.float64)
    history[0] = float(jax.jit(objective)(params))
    initial_components = np.asarray(component_values(params), dtype=np.float64)
    best_loss = history[0]
    best_params = params
    diverged = False
    completed_steps = 0

    for index in range(1, steps + 1):
        if batch_schedules is None:
            loss, gradients = value_and_grad(params)
        else:
            batch_indices = tuple(
                jnp.asarray(schedule[index - 1]) for schedule in batch_schedules
            )
            loss, gradients = value_and_grad(params, batch_indices)
        loss_value = float(loss)
        if not math.isfinite(loss_value) or not _all_leaves_finite(gradients):
            # Stop rather than propagate NaN into the parameters; the caller
            # receives the best finite iterate and an explicit flag.
            diverged = True
            break
        if loss_value < best_loss:
            best_loss = loss_value
            best_params = params
        gradients = zero_residual_configuration_gradient(gradients)
        if fixed_motor_time_constant:
            gradients = zero_response_time_gradient(gradients)
        if fixed_thrust_command_offset:
            gradients = zero_thrust_command_offset_gradient(gradients)
        if fixed_rotational_response:
            gradients = zero_rotational_response_gradient(gradients)
        elif fixed_angular_cross_coupling:
            gradients = zero_angular_cross_coupling_gradient(gradients)
        gradient_norm = jnp.sqrt(
            sum(jnp.sum(jnp.square(leaf)) for leaf in jax.tree.leaves(gradients))
        )
        clip_scale = jnp.minimum(1.0, gradient_clip_norm / (gradient_norm + 1e-12))
        gradients = jax.tree.map(
            lambda value, scale=clip_scale: value * scale,
            gradients,
        )

        first_moment = jax.tree.map(
            lambda moment, gradient: beta1 * moment + (1.0 - beta1) * gradient,
            first_moment,
            gradients,
        )
        second_moment = jax.tree.map(
            lambda moment, gradient: (
                beta2 * moment + (1.0 - beta2) * jnp.square(gradient)
            ),
            second_moment,
            gradients,
        )
        corrected_first = jax.tree.map(
            lambda moment, step=index: moment / (1.0 - beta1**step),
            first_moment,
        )
        corrected_second = jax.tree.map(
            lambda moment, step=index: moment / (1.0 - beta2**step),
            second_moment,
        )
        params = jax.tree.map(
            lambda parameter, first, second: (
                parameter - learning_rate * first / (jnp.sqrt(second) + epsilon)
            ),
            params,
            corrected_first,
            corrected_second,
        )
        history[index] = loss_value
        completed_steps = index

    if diverged:
        params = best_params
        history = history[: completed_steps + 1]
    final_loss = float(jax.jit(objective)(params))
    if not math.isfinite(final_loss):
        diverged = True
        params = best_params
        final_loss = float(jax.jit(objective)(params))
        history = history[: completed_steps + 1]
    if diverged:
        history = np.append(history, final_loss)
    else:
        history[-1] = final_loss
    final_components = np.asarray(component_values(params), dtype=np.float64)
    return FitResult(
        params=params,
        loss_history=history,
        diverged=diverged,
        completed_steps=completed_steps,
        component_initial_losses=initial_components,
        component_final_losses=final_components,
        loss_configuration=loss_configuration,
        optimization_policy=(
            "full_batch_v1" if batch_schedules is None else OPTIMIZATION_POLICY_VERSION
        ),
        batch_sizes=(
            ()
            if batch_schedules is None
            else tuple(schedule.shape[1] for schedule in batch_schedules)
        ),
        window_coverage=(
            ()
            if batch_schedules is None
            else tuple(
                len(np.unique(schedule)) / window_count
                for schedule, window_count in zip(
                    batch_schedules,
                    batch_window_counts,
                )
            )
        ),
    )


def _configured_initial_params(
    initial_params: ModelParams,
    fixed_motor_time_constant_s: float | None,
) -> ModelParams:
    if fixed_motor_time_constant_s is not None and fixed_motor_time_constant_s <= 0.0:
        raise ValueError("fixed_motor_time_constant_s must be positive")
    if fixed_motor_time_constant_s is None:
        return initial_params
    return with_response_time_constant(initial_params, fixed_motor_time_constant_s)


def _validate_window_schema(params: ModelParams, windows: TrajectoryWindows) -> None:
    validate_control_schema(
        params,
        windows.control_names,
        windows.control_roles,
    )


def supports_multirotor_thrust_command_offset(
    params: ModelParams,
    windows: TrajectoryWindows,
) -> bool:
    """Resolve the thrust-map policy from typed control semantics."""

    if model_family(params).platform != "multirotor":
        return False
    semantics = frozenset(windows.control_semantics)
    if semantics <= NORMALIZED_MOTOR_COMMAND_SEMANTICS:
        return True
    if semantics <= PHYSICAL_MOTOR_THRUST_SEMANTICS:
        return False
    raise ValueError(
        "multirotor controls must all be normalized motor commands or "
        "physical squared-rotor-speed thrust proxies; got "
        + ", ".join(sorted(semantics))
    )


def _resolved_thrust_command_offset_policy(
    params: ModelParams,
    windows: tuple[TrajectoryWindows, ...],
    requested: bool,
) -> bool:
    semantic_policies = {
        supports_multirotor_thrust_command_offset(params, item) for item in windows
    }
    if len(semantic_policies) != 1:
        raise ValueError(
            "all training horizons must use one motor-control semantic policy"
        )
    semantic_policy = semantic_policies.pop()
    if requested is True and not semantic_policy:
        raise ValueError(
            "a thrust command offset can only be learned from normalized "
            "multirotor motor commands"
        )
    return requested


def _residual_regularization(params: ModelParams) -> Array:
    if not isinstance(params, ResidualDynamicsParams):
        return jnp.asarray(0.0)
    return 1e-3 * (
        jnp.mean(jnp.square(params.hidden_weights))
        + jnp.mean(jnp.square(params.output_weights))
        + jnp.mean(jnp.square(params.hidden_bias))
    )


def _window_loss(
    params: ModelParams,
    windows: TrajectoryWindows,
    loss_configuration: RolloutLossConfiguration,
    indices: Array | None = None,
) -> Array:
    initial_states = jnp.asarray(windows.initial_states)
    control_histories = jnp.asarray(windows.control_histories)
    controls = jnp.asarray(windows.controls)
    target_states = jnp.asarray(windows.target_states)
    initial_exogenous = jnp.asarray(windows.initial_exogenous)
    window_weights = (
        None if windows.window_weights is None else jnp.asarray(windows.window_weights)
    )
    if indices is not None:
        initial_states = initial_states[indices]
        control_histories = control_histories[indices]
        controls = controls[indices]
        target_states = target_states[indices]
        initial_exogenous = initial_exogenous[indices]
        # ``deterministic_weighted_batch_schedule`` already draws windows in
        # proportion to their weights, so the in-batch mean must be uniform.
        # Weighting the sampled windows again would square every weight.
        window_weights = None
    return batch_rollout_loss(
        params,
        initial_states,
        control_histories,
        controls,
        target_states,
        windows.dt_s,
        loss_configuration,
        window_weights,
        windows.control_roles,
        initial_exogenous,
        windows.exogenous_roles,
    )


def fit_dynamics(
    windows: TrajectoryWindows,
    initial_params: ModelParams,
    *,
    steps: int = 400,
    learning_rate: float = 0.03,
    gradient_clip_norm: float = 10.0,
    fixed_motor_time_constant_s: float | None = None,
    learn_thrust_command_offset: bool = False,
    instantaneous_rotational_response: bool = False,
    diagonal_angular_control: bool = False,
    loss_configuration: RolloutLossConfiguration | None = None,
    endpoint_weight: float = 3.0,
    stability_regularization: float = 0.01,
) -> FitResult:
    """Fit dynamics parameters with Adam and return the complete loss history."""

    if steps < 1:
        raise ValueError("steps must be positive")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    _validate_window_schema(initial_params, windows)
    learn_thrust_command_offset = _resolved_thrust_command_offset_policy(
        initial_params, (windows,), learn_thrust_command_offset
    )
    initial_params = _configured_initial_params(
        initial_params, fixed_motor_time_constant_s
    )
    if (
        not learn_thrust_command_offset
        and model_family(initial_params).platform == "multirotor"
    ):
        initial_params = with_thrust_command_offset(initial_params, 0.0)
    if instantaneous_rotational_response:
        initial_params = with_instantaneous_rotational_response(initial_params)
    else:
        if diagonal_angular_control:
            initial_params = with_diagonal_angular_control(initial_params)
        _warn_if_rotational_response_frozen(initial_params)
    if loss_configuration is None:
        loss_configuration = rollout_loss_configuration(
            [windows],
            endpoint_weight=endpoint_weight,
            stability_regularization=stability_regularization,
        )

    def component_objective(params: ModelParams) -> Array:
        return jnp.asarray([_window_loss(params, windows, loss_configuration)])

    def objective(params: ModelParams) -> Array:
        return component_objective(params)[0] + _residual_regularization(params)

    batch_schedules = _optimization_batch_schedules([windows], steps=steps)

    def batch_objective(params: ModelParams, batch_indices: tuple[Array, ...]) -> Array:
        return _window_loss(
            params,
            windows,
            loss_configuration,
            indices=batch_indices[0],
        ) + _residual_regularization(params)

    return _fit_objective(
        objective,
        component_objective,
        initial_params,
        steps=steps,
        learning_rate=learning_rate,
        gradient_clip_norm=gradient_clip_norm,
        fixed_motor_time_constant=fixed_motor_time_constant_s is not None,
        fixed_thrust_command_offset=not learn_thrust_command_offset,
        fixed_rotational_response=instantaneous_rotational_response,
        fixed_angular_cross_coupling=diagonal_angular_control,
        loss_configuration=loss_configuration,
        batch_objective=(None if batch_schedules is None else batch_objective),
        batch_schedules=batch_schedules,
        batch_window_counts=(
            None if batch_schedules is None else (len(windows.initial_states),)
        ),
    )


def fit_dynamics_multi_horizon(
    window_sets: tuple[TrajectoryWindows, ...] | list[TrajectoryWindows],
    initial_params: ModelParams,
    *,
    steps: int = 400,
    learning_rate: float = 0.03,
    gradient_clip_norm: float = 10.0,
    fixed_motor_time_constant_s: float | None = None,
    learn_thrust_command_offset: bool = False,
    instantaneous_rotational_response: bool = False,
    diagonal_angular_control: bool = False,
    horizon_weights: tuple[float, ...] | list[float] | None = None,
    loss_configuration: RolloutLossConfiguration | None = None,
    endpoint_weight: float = 3.0,
    stability_regularization: float = 0.01,
    loss_normalization_params: ModelParams | None = None,
    loss_normalization_window_sets: (
        tuple[TrajectoryWindows, ...] | list[TrajectoryWindows] | None
    ) = None,
) -> FitResult:
    """Fit one model to normalized rollout losses at several horizons."""

    if not window_sets:
        raise ValueError("at least one window set is required")
    if steps < 1:
        raise ValueError("steps must be positive")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    for windows in window_sets:
        _validate_window_schema(initial_params, windows)
    if loss_normalization_window_sets is not None:
        if len(loss_normalization_window_sets) != len(window_sets):
            raise ValueError("loss_normalization_window_sets must match window_sets")
        for windows in loss_normalization_window_sets:
            _validate_window_schema(initial_params, windows)
    learn_thrust_command_offset = _resolved_thrust_command_offset_policy(
        initial_params, tuple(window_sets), learn_thrust_command_offset
    )
    if horizon_weights is None:
        weights = jnp.full((len(window_sets),), 1.0 / len(window_sets))
    else:
        if len(horizon_weights) != len(window_sets):
            raise ValueError("horizon_weights must match window_sets")
        if any(weight <= 0.0 for weight in horizon_weights):
            raise ValueError("horizon weights must be positive")
        weights = jnp.asarray(horizon_weights, dtype=jnp.float32)
        weights = weights / jnp.sum(weights)

    initial_params = _configured_initial_params(
        initial_params, fixed_motor_time_constant_s
    )
    if (
        not learn_thrust_command_offset
        and model_family(initial_params).platform == "multirotor"
    ):
        initial_params = with_thrust_command_offset(initial_params, 0.0)
    if instantaneous_rotational_response:
        initial_params = with_instantaneous_rotational_response(initial_params)
    else:
        if diagonal_angular_control:
            initial_params = with_diagonal_angular_control(initial_params)
        _warn_if_rotational_response_frozen(initial_params)
    if loss_normalization_params is not None:
        loss_normalization_params = _configured_initial_params(
            loss_normalization_params, fixed_motor_time_constant_s
        )
        if (
            not learn_thrust_command_offset
            and model_family(loss_normalization_params).platform == "multirotor"
        ):
            loss_normalization_params = with_thrust_command_offset(
                loss_normalization_params, 0.0
            )
        if instantaneous_rotational_response:
            loss_normalization_params = with_instantaneous_rotational_response(
                loss_normalization_params
            )
        elif diagonal_angular_control:
            loss_normalization_params = with_diagonal_angular_control(
                loss_normalization_params
            )
    if loss_configuration is None:
        loss_configuration = rollout_loss_configuration(
            (
                window_sets
                if loss_normalization_window_sets is None
                else loss_normalization_window_sets
            ),
            endpoint_weight=endpoint_weight,
            stability_regularization=stability_regularization,
        )

    def component_objective(params: ModelParams) -> Array:
        return jnp.stack(
            [
                _window_loss(params, windows, loss_configuration)
                for windows in window_sets
            ]
        )

    normalization_params = (
        initial_params
        if loss_normalization_params is None
        else loss_normalization_params
    )
    initial_component_losses = (
        component_objective(normalization_params)
        if loss_normalization_window_sets is None
        else jnp.stack(
            [
                _window_loss(
                    normalization_params,
                    windows,
                    loss_configuration,
                )
                for windows in loss_normalization_window_sets
            ]
        )
    )
    normalizers = jax.lax.stop_gradient(jnp.maximum(initial_component_losses, 1e-12))

    def objective(params: ModelParams) -> Array:
        return jnp.sum(
            weights * component_objective(params) / normalizers
        ) + _residual_regularization(params)

    batch_schedules = _optimization_batch_schedules(window_sets, steps=steps)

    def batch_objective(params: ModelParams, batch_indices: tuple[Array, ...]) -> Array:
        component_losses = jnp.stack(
            [
                _window_loss(
                    params,
                    windows,
                    loss_configuration,
                    indices=indices,
                )
                for windows, indices in zip(window_sets, batch_indices)
            ]
        )
        return jnp.sum(
            weights * component_losses / normalizers
        ) + _residual_regularization(params)

    result = _fit_objective(
        objective,
        component_objective,
        initial_params,
        steps=steps,
        learning_rate=learning_rate,
        gradient_clip_norm=gradient_clip_norm,
        fixed_motor_time_constant=fixed_motor_time_constant_s is not None,
        fixed_thrust_command_offset=not learn_thrust_command_offset,
        fixed_rotational_response=instantaneous_rotational_response,
        fixed_angular_cross_coupling=diagonal_angular_control,
        loss_configuration=loss_configuration,
        batch_objective=(None if batch_schedules is None else batch_objective),
        batch_schedules=batch_schedules,
        batch_window_counts=(
            None
            if batch_schedules is None
            else tuple(len(windows.initial_states) for windows in window_sets)
        ),
    )
    return replace(
        result,
        component_loss_normalizers=np.asarray(normalizers, dtype=np.float64),
    )
