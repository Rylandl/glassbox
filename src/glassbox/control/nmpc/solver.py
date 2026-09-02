"""Warm-started bounded direct-shooting NMPC backend."""

from __future__ import annotations

import math
import time
from copy import copy
from dataclasses import dataclass, replace
from typing import Protocol

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from glassbox.belief.belief import DynamicsBelief, RuntimeDynamicsBelief
from glassbox.control.nmpc.types import (
    NMPCDiagnostics,
    NMPCResult,
    NMPCWarmStart,
    ReferenceTrajectory,
    SafetyEnvelope,
    SolveStatus,
    SupportFilterMode,
    TrackingTolerances,
)
from glassbox.core.data import duration_to_steps
from glassbox.core.dynamics import ModelParams, quaternion_to_rotation
from glassbox.core.geometry import rigid_body_local_error
from glassbox.core.runtime import RuntimeDynamicsModel

_MINIMUM_SUPPORT_HORIZON_S = 0.1
_MAXIMUM_SUPPORT_HORIZON_S = 0.3
_ACTUATOR_TIME_CONSTANT_MULTIPLIER = 2.0
_SUPPORT_INTERPOLATION_FRACTIONS = (0.75, 0.5, 0.25, 0.0)
_MAXIMUM_COMMAND_BLOCKS = 10
_COMMAND_BOUND_RELATIVE_TOLERANCE = 1e-6

# iteration, blocks, value, gradient, step size, converged, stalled,
# line-search failure, and whether any outer iteration was accepted.
_OuterCarry = tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array]


def _block_steps_for(horizon_steps: int, block_count: int) -> int:
    """Return the model steps each command block is held for."""

    return math.ceil(horizon_steps / block_count)


def _blocks_cover_horizon(horizon_steps: int, block_count: int) -> bool:
    """Whether every block of this layout drives at least one model step."""

    block_steps = _block_steps_for(horizon_steps, block_count)
    return (block_count - 1) * block_steps < horizon_steps


def _maintained_block_count(horizon_steps: int) -> int:
    """Choose the maintained command-block layout for one horizon length.

    The layout is the largest divisor of ``horizon_steps`` no greater than
    ``_MAXIMUM_COMMAND_BLOCKS``, so every block is held for the same number of
    model steps and the expansion covers the horizon exactly. A horizon of ten
    steps or fewer therefore uses one block per step. When the horizon is a
    prime longer than that cap the only divisor available is one, which would
    throw away nearly all command authority; in that case the largest block
    count whose last block is merely truncated, and never empty, is used
    instead.
    """

    limit = min(_MAXIMUM_COMMAND_BLOCKS, horizon_steps)
    for count in range(limit, 1, -1):
        if horizon_steps % count == 0:
            return count
    for count in range(limit, 0, -1):
        if _blocks_cover_horizon(horizon_steps, count):
            return count
    raise AssertionError("a single block always covers the horizon")


def _projected_gradient_norm(blocks: Array, gradient: Array) -> Array:
    """Return the infinity norm of the bound-projected gradient.

    Command blocks are normalized to ``[-1, 1]`` and every iterate is projected
    back into that box, so a raw gradient component that points outward at an
    active bound never shrinks no matter how optimal the iterate is. The
    projected step ``blocks - clip(blocks - gradient)`` is the honest
    first-order residual for this bounded problem: it vanishes exactly when no
    feasible descent direction remains.
    """

    return jnp.max(jnp.abs(blocks - jnp.clip(blocks - gradient, -1.0, 1.0)))


@dataclass(frozen=True)
class _SolverPolicy:
    horizon_steps: int
    block_count: int
    maximum_iterations: int = 8
    line_search_steps: int = 8
    initial_step_size: float = 0.2
    gradient_tolerance: float = 2e-3
    relative_improvement_tolerance: float = 1e-5
    armijo_fraction: float = 1e-4
    command_change_fraction: float = 0.25
    command_change_weight: float = 0.03
    validity_weight: float = 20.0
    safety_weight: float = 40.0
    terminal_weight: float = 2.0

    @property
    def block_steps(self) -> int:
        """Model steps each command block is held for."""

        return _block_steps_for(self.horizon_steps, self.block_count)

    def __post_init__(self) -> None:
        if self.horizon_steps < 1:
            raise ValueError("horizon_steps must be positive")
        if not 1 <= self.block_count <= self.horizon_steps:
            raise ValueError("block_count must be within the prediction horizon")
        if not _blocks_cover_horizon(self.horizon_steps, self.block_count):
            raise ValueError(
                "block_count leaves trailing command blocks that drive no "
                "prediction step"
            )
        if self.maximum_iterations < 1 or self.line_search_steps < 1:
            raise ValueError("solver iteration counts must be positive")
        positive = (
            self.initial_step_size,
            self.gradient_tolerance,
            self.relative_improvement_tolerance,
            self.armijo_fraction,
            self.command_change_fraction,
            self.validity_weight,
            self.safety_weight,
            self.terminal_weight,
        )
        if not np.all(np.isfinite(positive)) or np.any(np.asarray(positive) <= 0):
            raise ValueError("solver policy values must be finite and positive")
        if (
            not np.isfinite(self.command_change_weight)
            or self.command_change_weight < 0
        ):
            raise ValueError("command_change_weight must be finite and nonnegative")


@dataclass(frozen=True)
class _SupportDecision:
    command: np.ndarray
    mode: SupportFilterMode
    applied: bool
    nominal_fraction: float
    current_validity: float
    next_mean_validity: float
    next_robust_validity: float
    current_rate_energy: float
    next_rate_energy: float
    support_horizon_s: float
    support_horizon_maximum_robust_validity: float
    support_horizon_terminal_robust_validity: float
    support_horizon_terminal_rate_energy: float


@dataclass(frozen=True)
class _SupportMetrics:
    """Per-candidate support metrics, non-finite entries masked to infinity.

    Masking rather than dropping keeps every array aligned with the candidate
    list, and infinity makes a candidate whose forecast went non-finite lose
    every comparison without a separate branch.
    """

    next_mean_validity: np.ndarray
    next_robust_validity: np.ndarray
    next_rate_energy: np.ndarray
    horizon_maximum_robust_validity: np.ndarray
    horizon_terminal_robust_validity: np.ndarray
    horizon_terminal_rate_energy: np.ndarray

    def at(self, index: int) -> tuple[float, float, float, float, float, float]:
        """Return one candidate's six metrics in decision-record order."""

        return (
            float(self.next_mean_validity[index]),
            float(self.next_robust_validity[index]),
            float(self.next_rate_energy[index]),
            float(self.horizon_maximum_robust_validity[index]),
            float(self.horizon_terminal_robust_validity[index]),
            float(self.horizon_terminal_rate_energy[index]),
        )


@dataclass(frozen=True)
class _PlanEvaluation:
    """One command plan with its objective, gradient norms, and prediction."""

    blocks: Array
    value: Array
    gradient: Array
    value_float: float
    gradient_inf_norm: float
    projected_gradient_inf_norm: float
    states: Array
    latent_states: Array
    commands: Array
    states_np: np.ndarray
    latent_np: np.ndarray
    commands_np: np.ndarray

    @property
    def prediction_finite(self) -> bool:
        """Whether the rolled-out plan and its objective are all finite."""

        return bool(
            np.all(np.isfinite(self.states_np))
            and np.all(np.isfinite(self.latent_np))
            and np.all(np.isfinite(self.commands_np))
            and np.isfinite(self.value_float)
        )


@dataclass
class _OptimizerOutcome:
    """What the bounded optimizer produced and how it terminated.

    ``converged`` and ``stalled`` are cleared by any later edit to the command
    blocks, because a plan that was changed after the line search no longer
    satisfies the criterion the line search reported.
    """

    blocks: Array
    value: Array
    gradient: Array
    iterations: int
    converged: bool
    stalled: bool
    line_search_failed: bool
    progressed: bool
    finite: bool
    stall_message: str


@dataclass(frozen=True)
class _PredictionDiagnostics:
    """Bound, validity, and safety measurements of one predicted horizon."""

    maximum_command_bound_violation: float
    maximum_validity_utilization: float
    maximum_normalized_safety_violation: float


@dataclass
class _SolveProgress:
    """Failure-report context accumulated as one solve advances."""

    started_at: float
    previous_command: Array
    initial_objective: float = math.inf
    iterations: int = 0
    warm_start_used: bool = False


class _SolveAbort(Exception):
    """Internal signal that one solve must return a bounded fallback command."""

    def __init__(self, status: SolveStatus, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def _default_policy(model: RuntimeDynamicsModel) -> _SolverPolicy:
    dt_s = model.runtime_spec.sample_period_s
    target_horizon_s = 0.6 if model.input_spec.vehicle.family == "multirotor" else 1.0
    certified = model.runtime_spec.certified_prediction_horizon_s
    if certified is not None:
        target_horizon_s = min(target_horizon_s, certified)
    maximum_steps = 40 if model.input_spec.vehicle.family == "multirotor" else 50
    steps = min(maximum_steps, max(2, duration_to_steps(target_horizon_s, dt_s)))
    if certified is not None and steps * dt_s > certified + 1e-12:
        steps = duration_to_steps(certified, dt_s)
    if steps < 1:
        raise ValueError("certified prediction horizon is shorter than one model step")
    return _SolverPolicy(
        horizon_steps=steps,
        block_count=_maintained_block_count(steps),
    )


def _runtime_belief(
    model: RuntimeDynamicsModel | RuntimeDynamicsBelief | DynamicsBelief,
) -> RuntimeDynamicsBelief:
    return (
        model.compile_for_nmpc()
        if isinstance(model, DynamicsBelief)
        else model
        if isinstance(model, RuntimeDynamicsBelief)
        else RuntimeDynamicsBelief.from_nominal(model)
    )


def _parameter_tree_signature(params: ModelParams) -> tuple[object, tuple[tuple, ...]]:
    leaves, structure = jax.tree_util.tree_flatten(params)
    signature = tuple(
        (np.asarray(leaf).shape, np.asarray(leaf).dtype.str) for leaf in leaves
    )
    return structure, signature


class _SolverBackend(Protocol):
    """Internal boundary allowing solver replacement without API changes."""

    prediction_steps: int
    prediction_horizon_s: float

    def solve(
        self,
        state: Array,
        reference: ReferenceTrajectory,
        previous_command: Array,
        *,
        applied_command: Array | None = None,
        latent_state: Array | None = None,
        warm_start: NMPCWarmStart | None = None,
        deadline_s: float | None = None,
    ) -> NMPCResult: ...

    def rebind(self, belief: RuntimeDynamicsBelief) -> _SolverBackend: ...


class _DirectShootingBackend:
    """Maintained bounded direct-shooting implementation."""

    def __init__(
        self,
        belief: RuntimeDynamicsBelief,
        tolerances: TrackingTolerances,
        safety_envelope: SafetyEnvelope,
        *,
        _policy: _SolverPolicy | None = None,
    ) -> None:
        self.belief = belief
        self.model = belief.nominal
        self._active_parameters = self.model.params
        self.tolerances = tolerances
        self.safety_envelope = safety_envelope
        self._policy = _default_policy(self.model) if _policy is None else _policy
        if _policy is None and belief.maximum_error_horizon_s is not None:
            supported_steps = math.floor(
                belief.maximum_error_horizon_s / self.model.runtime_spec.sample_period_s
                + 1e-9
            )
            if supported_steps < 1:
                raise ValueError(
                    "predictive-error evidence is shorter than one model step"
                )
            if supported_steps < self._policy.horizon_steps:
                self._policy = replace(
                    self._policy,
                    horizon_steps=supported_steps,
                    block_count=_maintained_block_count(supported_steps),
                )
        horizon_s = self._policy.horizon_steps * self.model.runtime_spec.sample_period_s
        certified = self.model.runtime_spec.certified_prediction_horizon_s
        if certified is not None and horizon_s > certified + 1e-12:
            raise ValueError("solver horizon exceeds the model's certified horizon")
        self._block_steps = self._policy.block_steps
        response_time_constants = np.asarray(
            self.model.latent_response_time_constants_s,
            dtype=np.float64,
        )
        slowest_actuator_s = float(np.max(response_time_constants))
        support_horizon_s = min(
            _MAXIMUM_SUPPORT_HORIZON_S,
            max(
                _MINIMUM_SUPPORT_HORIZON_S,
                _ACTUATOR_TIME_CONSTANT_MULTIPLIER * slowest_actuator_s,
            ),
        )
        self._support_horizon_steps = min(
            self._policy.horizon_steps,
            max(
                1,
                math.ceil(
                    support_horizon_s / self.model.runtime_spec.sample_period_s - 1e-9
                ),
            ),
        )
        self._objective_gradient = jax.value_and_grad(self._objective)
        self._objective_and_gradient = jax.jit(self._objective_gradient)
        self._initial_latent_compiled = jax.jit(
            self.model.initial_latent_state_with_parameters
        )
        self._optimize_compiled = jax.jit(self._optimize)
        self._rollout_compiled = jax.jit(self._rollout)
        self._validity_compiled = jax.jit(self._maximum_validity_utilization)
        self._safety_compiled = jax.jit(self._maximum_safety_violation)
        self._uncertainty_support_compiled = jax.jit(
            self._model_uncertainty_and_support_metrics
        )
        self._support_metric_compiled = jax.jit(self._support_horizon_metrics)
        self._support_metrics_compiled = jax.jit(
            jax.vmap(
                self._support_horizon_metrics,
                in_axes=(None, None, 0, None, None),
            )
        )
        self._support_batch_warmed = False

    @property
    def prediction_steps(self) -> int:
        return self._policy.horizon_steps

    @property
    def prediction_horizon_s(self) -> float:
        return self.prediction_steps * self.model.runtime_spec.sample_period_s

    @property
    def command_block_count(self) -> int:
        return self._policy.block_count

    def rebind(self, belief: RuntimeDynamicsBelief) -> _DirectShootingBackend:
        """Return a handle sharing compiled kernels with compatible new numerics."""

        candidate = belief.nominal
        if candidate.input_spec != self.model.input_spec:
            raise ValueError("rebound belief input specification changed")
        if candidate.runtime_spec != self.model.runtime_spec:
            raise ValueError("rebound belief runtime specification changed")
        same_actuation = candidate.actuation is self.model.actuation
        if not same_actuation:
            try:
                same_actuation = bool(candidate.actuation == self.model.actuation)
            except (TypeError, ValueError):
                same_actuation = False
        if not same_actuation:
            raise ValueError("rebound belief actuation map changed")
        if _parameter_tree_signature(candidate.params) != _parameter_tree_signature(
            self.model.params
        ):
            raise ValueError("rebound belief parameter structure changed")
        candidate_leaves = jax.tree_util.tree_leaves(candidate.params)
        if not all(np.all(np.isfinite(np.asarray(leaf))) for leaf in candidate_leaves):
            raise ValueError("rebound belief parameters must be finite")
        if (
            belief.parameter_belief is not self.belief.parameter_belief
            and belief.parameter_belief.to_dict()
            != self.belief.parameter_belief.to_dict()
        ):
            raise ValueError("rebound belief parameter uncertainty changed")
        belief_flags = (
            belief.predictive_error_available,
            belief.predictive_error_current,
            belief.parameter_uncertainty_available,
            belief.maximum_error_horizon_s,
        )
        template_flags = (
            self.belief.predictive_error_available,
            self.belief.predictive_error_current,
            self.belief.parameter_uncertainty_available,
            self.belief.maximum_error_horizon_s,
        )
        if belief_flags != template_flags:
            raise ValueError("rebound belief uncertainty availability changed")
        if (
            belief.predictive_error_current
            and belief.predictive_error.to_dict()
            != self.belief.predictive_error.to_dict()
        ):
            raise ValueError("rebound belief predictive-error numerics changed")
        response_time_constants = np.asarray(
            candidate.latent_response_time_constants_s,
            dtype=np.float64,
        )
        support_horizon_s = min(
            _MAXIMUM_SUPPORT_HORIZON_S,
            max(
                _MINIMUM_SUPPORT_HORIZON_S,
                _ACTUATOR_TIME_CONSTANT_MULTIPLIER
                * float(np.max(response_time_constants)),
            ),
        )
        support_steps = min(
            self._policy.horizon_steps,
            max(
                1,
                math.ceil(
                    support_horizon_s / candidate.runtime_spec.sample_period_s - 1e-9
                ),
            ),
        )
        if support_steps != self._support_horizon_steps:
            raise ValueError("rebound belief changes the compiled support horizon")

        rebound = copy(self)
        rebound.belief = belief
        rebound.model = candidate
        rebound._active_parameters = candidate.params
        return rebound

    def _expand_normalized_blocks(self, blocks: Array) -> Array:
        """Hold each block over its model steps, covering the whole horizon.

        The maintained layout divides the horizon exactly, so the trailing
        slice is a no-op. It only ever shortens the final block of a horizon
        whose length admits no usable divisor, and every block still drives at
        least one prediction step.
        """

        expanded = jnp.repeat(blocks, self._block_steps, axis=0)
        return expanded[: self.prediction_steps]

    def _commands_from_normalized(self, normalized: Array) -> Array:
        minimum = self.model.command_minimum
        command_range = self.model.command_maximum - minimum
        return minimum + 0.5 * (jnp.clip(normalized, -1.0, 1.0) + 1.0) * command_range

    def _normalized_from_commands(self, commands: Array) -> Array:
        minimum = self.model.command_minimum
        command_range = self.model.command_maximum - minimum
        return 2.0 * (commands - minimum) / command_range - 1.0

    def _rollout(
        self,
        blocks: Array,
        initial_state: Array,
        initial_latent: Array,
        exogenous: Array,
        model_parameters: ModelParams,
    ) -> tuple[Array, Array, Array]:
        normalized_commands = self._expand_normalized_blocks(blocks)
        commands = self._commands_from_normalized(normalized_commands)

        def transition(
            carry: tuple[Array, Array], inputs: tuple[Array, Array]
        ) -> tuple[tuple[Array, Array], tuple[Array, Array]]:
            state, latent = carry
            command, context = inputs
            next_state, next_latent = self.model.transition_at_interval_with_parameters(
                model_parameters,
                state,
                latent,
                command,
                self.model.runtime_spec.sample_period_s,
                context,
            )
            return (next_state, next_latent), (next_state, next_latent)

        _, (future_states, future_latent) = jax.lax.scan(
            transition,
            (initial_state, initial_latent),
            (commands, exogenous),
        )
        horizons = self.model.runtime_spec.sample_period_s * jnp.arange(
            1, self.prediction_steps + 1
        )
        corrected_states, _, _ = jax.vmap(self.belief.corrected_state)(
            future_states,
            horizons,
            commands,
            exogenous,
        )
        states = jnp.concatenate((initial_state[None, :], corrected_states), axis=0)
        latent = jnp.concatenate((initial_latent[None, :], future_latent), axis=0)
        return states, latent, commands

    def _validity_utilization(self, state: Array, exogenous: Array) -> Array:
        return self.model.validity_utilization(state, exogenous)

    def _safety_violation(self, state: Array) -> Array:
        envelope = self.safety_envelope
        violations: list[Array] = []
        position_scale = self.tolerances.local_state_scale[0:3]
        if envelope.minimum_position_m is not None:
            violations.append(
                jax.nn.relu(jnp.asarray(envelope.minimum_position_m) - state[0:3])
                / position_scale
            )
        if envelope.maximum_position_m is not None:
            violations.append(
                jax.nn.relu(state[0:3] - jnp.asarray(envelope.maximum_position_m))
                / position_scale
            )
        if envelope.maximum_speed_m_s is not None:
            violations.append(
                jnp.atleast_1d(
                    jax.nn.relu(
                        jnp.sqrt(jnp.sum(jnp.square(state[3:6])) + 1e-12)
                        - envelope.maximum_speed_m_s
                    )
                    / envelope.maximum_speed_m_s
                )
            )
        if envelope.maximum_angular_velocity_rad_s is not None:
            violations.append(
                jnp.atleast_1d(
                    jax.nn.relu(
                        jnp.sqrt(jnp.sum(jnp.square(state[10:13])) + 1e-12)
                        - envelope.maximum_angular_velocity_rad_s
                    )
                    / envelope.maximum_angular_velocity_rad_s
                )
            )
        return jnp.concatenate(violations) if violations else jnp.zeros(1)

    def _objective(
        self,
        blocks: Array,
        initial_state: Array,
        initial_latent: Array,
        reference_states: Array,
        previous_command: Array,
        exogenous: Array,
        model_parameters: ModelParams,
    ) -> Array:
        states, _, commands = self._rollout(
            blocks,
            initial_state,
            initial_latent,
            exogenous,
            model_parameters,
        )
        local_error = jax.vmap(rigid_body_local_error)(reference_states[1:], states[1:])
        normalized_error = local_error / self.tolerances.local_state_scale
        tracking_cost = jnp.mean(jnp.sum(jnp.square(normalized_error), axis=1))
        terminal_cost = self._policy.terminal_weight * jnp.sum(
            jnp.square(normalized_error[-1])
        )

        command_range = self.model.command_maximum - self.model.command_minimum
        command_delta = jnp.diff(
            jnp.concatenate((previous_command[None, :], commands), axis=0), axis=0
        )
        normalized_delta = command_delta / (
            self._policy.command_change_fraction * command_range
        )
        smoothness_cost = self._policy.command_change_weight * jnp.mean(
            jnp.square(normalized_delta)
        )

        utilization = jax.vmap(self._validity_utilization)(states[1:], exogenous)
        validity_cost = self._policy.validity_weight * jnp.mean(
            jnp.square(jax.nn.relu(utilization - 1.0))
        )
        safety_violation = jax.vmap(self._safety_violation)(states[1:])
        safety_cost = self._policy.safety_weight * jnp.mean(
            jnp.square(safety_violation)
        )
        return (
            tracking_cost
            + terminal_cost
            + smoothness_cost
            + validity_cost
            + safety_cost
        )

    def _maximum_validity_utilization(self, states: Array, exogenous: Array) -> Array:
        return jnp.max(jax.vmap(self._validity_utilization)(states[1:], exogenous))

    def _maximum_safety_violation(self, states: Array) -> Array:
        return jnp.max(jax.vmap(self._safety_violation)(states[1:]))

    def _model_uncertainty_and_support_metrics(
        self,
        initial_state: Array,
        initial_latent: Array,
        commands: Array,
        exogenous: Array,
        model_parameters: ModelParams,
    ) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
        forecast = self.belief.rollout(
            initial_state,
            commands,
            model_parameters=model_parameters,
            initial_latent_state=initial_latent,
            exogenous=exogenous,
        )
        maximum_uncertainty = jnp.max(
            forecast.tangent_standard_deviation[1:]
            / self.tolerances.local_state_scale[None, :]
        )
        support_steps = min(self._support_horizon_steps, len(commands))
        mean_validity, robust_validity, rate_energy = jax.vmap(
            self._support_state_metrics
        )(
            forecast.mean_states[1 : support_steps + 1],
            forecast.tangent_covariance[1 : support_steps + 1],
            exogenous[:support_steps],
        )
        return (
            maximum_uncertainty,
            mean_validity[0],
            robust_validity[0],
            rate_energy[0],
            jnp.max(robust_validity),
            robust_validity[-1],
            rate_energy[-1],
        )

    def _support_state_metrics(
        self,
        mean_state: Array,
        covariance: Array,
        exogenous: Array,
    ) -> tuple[Array, Array, Array]:
        """Return mean/marginal-radius validity and normalized rate energy."""

        envelope = self.model.runtime_spec.validity_envelope
        mean_utilization = self.model.validity_utilization(mean_state, exogenous)
        mean_validity = jnp.max(mean_utilization)

        roles = self.model.input_spec.exogenous_roles
        wind = jnp.stack(
            tuple(
                exogenous[roles.index(role)] if role in roles else jnp.asarray(0.0)
                for role in ("wind_north", "wind_west", "wind_up")
            )
        )
        rotation = quaternion_to_rotation(mean_state[6:10])
        body_velocity = rotation.T @ (mean_state[3:6] - wind)
        body_velocity_cross = jnp.asarray(
            (
                (0.0, -body_velocity[2], body_velocity[1]),
                (body_velocity[2], 0.0, -body_velocity[0]),
                (-body_velocity[1], body_velocity[0], 0.0),
            )
        )
        feature_jacobian = jnp.zeros((6, 12))
        feature_jacobian = feature_jacobian.at[0:3, 3:6].set(rotation.T)
        feature_jacobian = feature_jacobian.at[0:3, 6:9].set(body_velocity_cross)
        feature_jacobian = feature_jacobian.at[3:6, 9:12].set(jnp.eye(3))
        feature_covariance = feature_jacobian @ covariance @ feature_jacobian.T
        feature_half_width = jnp.concatenate(
            (
                jnp.asarray(envelope.body_velocity_half_width_m_s),
                jnp.asarray(envelope.angular_velocity_half_width_rad_s),
            )
        )
        marginal_radius = (
            jnp.sqrt(jnp.maximum(jnp.diag(feature_covariance), 0.0))
            / feature_half_width
        )
        robust_validity = jnp.max(mean_utilization + marginal_radius)
        normalized_rate = (
            mean_state[10:13] - jnp.asarray(envelope.angular_velocity_center_rad_s)
        ) / jnp.asarray(envelope.angular_velocity_half_width_rad_s)
        rate_energy = jnp.sum(jnp.square(normalized_rate))
        return mean_validity, robust_validity, rate_energy

    def _support_horizon_metrics(
        self,
        state: Array,
        latent: Array,
        command: Array,
        exogenous: Array,
        model_parameters: ModelParams,
    ) -> tuple[Array, Array, Array, Array, Array, Array]:
        """Evaluate one held command over the actuator reaction horizon."""

        commands = jnp.repeat(
            command[None, :],
            self._support_horizon_steps,
            axis=0,
        )
        exogenous_forecast = jnp.repeat(
            exogenous[None, :], self._support_horizon_steps, axis=0
        )
        forecast = self.belief.rollout(
            state,
            commands,
            model_parameters=model_parameters,
            initial_latent_state=latent,
            exogenous=exogenous_forecast,
        )
        mean_validity, robust_validity, rate_energy = jax.vmap(
            self._support_state_metrics
        )(
            forecast.mean_states[1:],
            forecast.tangent_covariance[1:],
            exogenous_forecast,
        )
        return (
            mean_validity[0],
            robust_validity[0],
            rate_energy[0],
            jnp.max(robust_validity),
            robust_validity[-1],
            rate_energy[-1],
        )

    def _current_support_metrics(
        self,
        state: np.ndarray,
        exogenous: np.ndarray,
    ) -> tuple[float, float]:
        current_validity = float(
            np.max(
                np.asarray(
                    self.model.validity_utilization(
                        jnp.asarray(state),
                        jnp.asarray(exogenous),
                    )
                )
            )
        )
        envelope = self.model.runtime_spec.validity_envelope
        normalized_rate = (
            state[10:13] - np.asarray(envelope.angular_velocity_center_rad_s)
        ) / np.asarray(envelope.angular_velocity_half_width_rad_s)
        return current_validity, float(normalized_rate @ normalized_rate)

    def _uncertainty_support_values(
        self,
        state: Array,
        latent: Array,
        commands: Array,
        exogenous: Array,
    ) -> tuple[float, tuple[float, float, float, float, float, float]]:
        values = tuple(
            float(np.asarray(value))
            for value in self._uncertainty_support_compiled(
                state,
                latent,
                commands,
                exogenous,
                self._active_parameters,
            )
        )
        return values[0], values[1:]  # type: ignore[return-value]

    def _support_candidates(
        self,
        nominal_command: np.ndarray,
        previous_command: np.ndarray,
    ) -> tuple[list[np.ndarray], list[float]]:
        minimum = np.asarray(self.model.command_minimum, dtype=np.float64)
        maximum = np.asarray(self.model.command_maximum, dtype=np.float64)
        candidates = [np.clip(nominal_command, minimum, maximum)]
        nominal_fractions = [1.0]
        anchor = np.clip(previous_command, minimum, maximum)

        for fraction in _SUPPORT_INTERPOLATION_FRACTIONS:
            candidate = np.clip(
                anchor + fraction * (nominal_command - anchor),
                minimum,
                maximum,
            )
            candidates.append(candidate)
            nominal_fractions.append(fraction)
        return candidates, nominal_fractions

    def _warm_support_batch(
        self,
        state: Array,
        latent: Array,
        candidates: list[np.ndarray],
        exogenous: Array,
    ) -> None:
        """Compile the batched candidate kernel on a solve that does not need it.

        The recovery and boundary paths must not pay a first-call compile at the
        moment they are reached, so the first nominal-safe solve warms the same
        kernel with the candidates it would otherwise have evaluated.
        """

        if self._support_batch_warmed:
            return
        warmed = self._support_metrics_compiled(
            state,
            latent,
            jnp.asarray(np.asarray(candidates[1:])),
            exogenous,
            self._active_parameters,
        )
        jax.block_until_ready(warmed)
        self._support_batch_warmed = True

    def _candidate_support_metrics(
        self,
        state: Array,
        latent: Array,
        candidates: list[np.ndarray],
        nominal_metrics: tuple[float, float, float, float, float, float],
        exogenous: Array,
    ) -> _SupportMetrics:
        """Evaluate every candidate over the actuator reaction horizon.

        The nominal candidate's metrics are already known, so only the
        interpolated alternatives go through the batched kernel; the two are
        then concatenated back into candidate order.
        """

        alternative_metrics = (
            self._support_metrics_compiled(
                state,
                latent,
                jnp.asarray(np.asarray(candidates[1:])),
                exogenous,
                self._active_parameters,
            )
            if len(candidates) > 1
            else tuple(np.empty(0) for _ in range(6))
        )
        self._support_batch_warmed = True
        columns = [
            np.asarray(
                np.concatenate(
                    (
                        np.asarray((nominal_metrics[index],), dtype=np.float64),
                        np.asarray(alternative_metrics[index], dtype=np.float64),
                    )
                ),
                dtype=np.float64,
            )
            for index in range(6)
        ]
        finite = np.logical_and.reduce([np.isfinite(column) for column in columns])
        return _SupportMetrics(
            *(np.where(finite, column, np.inf) for column in columns)
        )

    def _acceptable_support(
        self,
        metrics: _SupportMetrics,
        *,
        inside_support: bool,
        current_validity: float,
        current_rate_energy: float,
    ) -> np.ndarray:
        """Mark the candidates whose support-horizon forecast is acceptable.

        Inside the validity envelope a candidate is acceptable when it never
        leaves the envelope over the whole support horizon.  Outside it no
        candidate can already be safe, so acceptability becomes strict progress
        instead: terminal robust validity and terminal rate energy must both
        fall by more than the rounding width of the value they are measured
        against, which stops a numerically flat candidate from being mistaken
        for a recovering one.
        """

        if inside_support:
            return metrics.horizon_maximum_robust_validity <= 1.0 + 1e-6
        validity_tolerance = (
            32.0
            * np.finfo(np.float64).eps
            * max(
                1.0,
                current_validity,
            )
        )
        energy_tolerance = (
            32.0
            * np.finfo(np.float64).eps
            * max(
                1.0,
                current_rate_energy,
            )
        )
        validity_progress = (
            metrics.horizon_terminal_robust_validity
            < current_validity - validity_tolerance
        )
        rate_progress = np.where(
            current_rate_energy > energy_tolerance,
            metrics.horizon_terminal_rate_energy
            < current_rate_energy - energy_tolerance,
            metrics.horizon_terminal_rate_energy <= energy_tolerance,
        )
        return validity_progress & rate_progress

    def _selected_support_candidate(
        self,
        metrics: _SupportMetrics,
        acceptable: np.ndarray,
        nominal_fractions: list[float],
        *,
        inside_support: bool,
        current_validity: float,
        current_rate_energy: float,
    ) -> tuple[int, SupportFilterMode]:
        """Pick one candidate and name the mode that picked it.

        With acceptable candidates available the choice is the most nominal one
        inside the envelope, and the one that recovers hardest outside it.  With
        none, the same orderings are applied as a best effort so a command is
        still returned, and the mode records that no candidate met the bar.
        """

        acceptable_indices = np.flatnonzero(acceptable)
        if len(acceptable_indices):
            selected_index = (
                int(acceptable_indices[0])
                if inside_support
                else min(
                    (int(index) for index in acceptable_indices),
                    key=lambda index: (
                        metrics.horizon_maximum_robust_validity[index],
                        metrics.horizon_terminal_robust_validity[index]
                        / max(current_validity, 1e-12),
                        metrics.horizon_terminal_rate_energy[index]
                        / max(current_rate_energy, 1e-12),
                        -nominal_fractions[index],
                    ),
                )
            )
            return selected_index, (
                SupportFilterMode.NOMINAL_SAFE
                if inside_support and selected_index == 0
                else SupportFilterMode.BOUNDARY_FILTERED
                if inside_support
                else SupportFilterMode.RECOVERY_FILTERED
            )

        if inside_support:
            scores = np.stack(
                (
                    metrics.horizon_maximum_robust_validity,
                    metrics.horizon_terminal_robust_validity,
                    metrics.horizon_terminal_rate_energy,
                ),
                axis=1,
            )
        else:
            validity_scale = max(current_validity, 1e-12)
            rate_scale = max(current_rate_energy, 1e-12)
            maximum_validity_ratio = (
                metrics.horizon_maximum_robust_validity / validity_scale
            )
            terminal_validity_ratio = (
                metrics.horizon_terminal_robust_validity / validity_scale
            )
            terminal_rate_ratio = metrics.horizon_terminal_rate_energy / rate_scale
            scores = np.stack(
                (
                    maximum_validity_ratio,
                    np.maximum(terminal_validity_ratio, terminal_rate_ratio),
                    terminal_validity_ratio + terminal_rate_ratio,
                ),
                axis=1,
            )
        selected_index = min(
            range(len(nominal_fractions)),
            key=lambda index: (
                scores[index, 0],
                scores[index, 1],
                scores[index, 2],
                -nominal_fractions[index],
            ),
        )
        return selected_index, (
            SupportFilterMode.BOUNDARY_BEST_EFFORT
            if inside_support
            else SupportFilterMode.RECOVERY_BEST_EFFORT
        )

    def _support_decision(
        self,
        command: np.ndarray,
        mode: SupportFilterMode,
        *,
        applied: bool,
        nominal_fraction: float,
        current_validity: float,
        current_rate_energy: float,
        metrics: tuple[float, float, float, float, float, float],
    ) -> _SupportDecision:
        """Assemble one auditable support-filter decision record."""

        return _SupportDecision(
            command=command,
            mode=mode,
            applied=applied,
            nominal_fraction=nominal_fraction,
            current_validity=current_validity,
            next_mean_validity=metrics[0],
            next_robust_validity=metrics[1],
            current_rate_energy=current_rate_energy,
            next_rate_energy=metrics[2],
            support_horizon_s=(
                self._support_horizon_steps * self.model.runtime_spec.sample_period_s
            ),
            support_horizon_maximum_robust_validity=metrics[3],
            support_horizon_terminal_robust_validity=metrics[4],
            support_horizon_terminal_rate_energy=metrics[5],
        )

    def _select_support_command(
        self,
        state: Array,
        latent: Array,
        nominal_command: Array,
        previous_command: Array,
        exogenous: Array,
        nominal_support_metrics: tuple[float, float, float, float, float, float]
        | None = None,
    ) -> _SupportDecision:
        """Choose the first command the belief can actually support.

        The nominal command is kept untouched whenever the vehicle is inside its
        validity envelope and the nominal plan keeps it there across the whole
        actuator reaction horizon.  Otherwise the interpolation toward the
        previous command is searched, and the mode records whether the result
        was safe, filtered, or the best available with nothing acceptable.
        """

        state_np = np.asarray(state, dtype=np.float64)
        nominal_np = np.asarray(nominal_command, dtype=np.float64)
        previous_np = np.asarray(previous_command, dtype=np.float64)
        exogenous_np = np.asarray(exogenous, dtype=np.float64)
        candidates, nominal_fractions = self._support_candidates(
            nominal_np, previous_np
        )
        current_validity, current_rate_energy = self._current_support_metrics(
            state_np,
            exogenous_np,
        )
        inside_support = current_validity <= 1.0 + 1e-6
        nominal_metrics = (
            tuple(
                float(np.asarray(value))
                for value in self._support_metric_compiled(
                    state,
                    latent,
                    jnp.asarray(candidates[0]),
                    exogenous,
                    self._active_parameters,
                )
            )
            if nominal_support_metrics is None
            else nominal_support_metrics
        )
        nominal_finite = bool(np.all(np.isfinite(nominal_metrics)))
        if nominal_finite and inside_support and nominal_metrics[3] <= 1.0 + 1e-6:
            self._warm_support_batch(state, latent, candidates, exogenous)
            return self._support_decision(
                candidates[0],
                SupportFilterMode.NOMINAL_SAFE,
                applied=False,
                nominal_fraction=1.0,
                current_validity=current_validity,
                current_rate_energy=current_rate_energy,
                metrics=nominal_metrics,
            )

        metrics = self._candidate_support_metrics(
            state,
            latent,
            candidates,
            nominal_metrics,
            exogenous,
        )
        selected_index, mode = self._selected_support_candidate(
            metrics,
            self._acceptable_support(
                metrics,
                inside_support=inside_support,
                current_validity=current_validity,
                current_rate_energy=current_rate_energy,
            ),
            nominal_fractions,
            inside_support=inside_support,
            current_validity=current_validity,
            current_rate_energy=current_rate_energy,
        )
        selected = candidates[selected_index]
        return self._support_decision(
            selected,
            mode,
            applied=not np.allclose(selected, nominal_np, rtol=0.0, atol=1e-12),
            nominal_fraction=nominal_fractions[selected_index],
            current_validity=current_validity,
            current_rate_energy=current_rate_energy,
            metrics=metrics.at(selected_index),
        )

    def _optimize(
        self,
        initial_blocks: Array,
        initial_value: Array,
        initial_gradient: Array,
        initial_state: Array,
        initial_latent: Array,
        reference_states: Array,
        previous_command: Array,
        exogenous: Array,
        model_parameters: ModelParams,
    ) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array]:
        """Run the fixed maintained policy as one compiled JAX operation."""

        def finite(value: Array, gradient: Array) -> Array:
            return jnp.isfinite(value) & jnp.all(jnp.isfinite(gradient))

        def continue_outer(carry: _OuterCarry) -> Array:
            (
                iteration,
                _,
                value,
                gradient,
                _,
                converged,
                stalled,
                line_search_failed,
                _,
            ) = carry
            return (
                (iteration < self._policy.maximum_iterations)
                & ~converged
                & ~stalled
                & ~line_search_failed
                & finite(value, gradient)
            )

        def outer_step(carry: _OuterCarry) -> _OuterCarry:
            iteration, blocks, value, gradient, step_size, _, _, _, progressed = carry
            gradient_norm = _projected_gradient_norm(blocks, gradient)
            gradient_converged = gradient_norm <= self._policy.gradient_tolerance

            def continue_line_search(
                line_carry: tuple[Array, Array, Array, Array, Array, Array],
            ) -> Array:
                line_iteration, accepted, _, _, _, _ = line_carry
                return (line_iteration < self._policy.line_search_steps) & ~accepted

            def line_search_step(
                line_carry: tuple[Array, Array, Array, Array, Array, Array],
            ) -> tuple[Array, Array, Array, Array, Array, Array]:
                (
                    line_iteration,
                    accepted,
                    best_blocks,
                    best_value,
                    best_gradient,
                    accepted_step_size,
                ) = line_carry
                candidate_step_size = step_size * jnp.power(0.5, line_iteration)
                candidate = jnp.clip(blocks - candidate_step_size * gradient, -1.0, 1.0)
                candidate_value, candidate_gradient = self._objective_gradient(
                    candidate,
                    initial_state,
                    initial_latent,
                    reference_states,
                    previous_command,
                    exogenous,
                    model_parameters,
                )
                projected_decrease = jnp.sum(gradient * (blocks - candidate))
                candidate_accepted = finite(candidate_value, candidate_gradient) & (
                    candidate_value
                    <= value
                    - self._policy.armijo_fraction
                    * jnp.maximum(projected_decrease, 0.0)
                )
                return (
                    line_iteration + 1,
                    accepted | candidate_accepted,
                    jnp.where(candidate_accepted, candidate, best_blocks),
                    jnp.where(candidate_accepted, candidate_value, best_value),
                    jnp.where(candidate_accepted, candidate_gradient, best_gradient),
                    jnp.where(
                        candidate_accepted,
                        candidate_step_size,
                        accepted_step_size,
                    ),
                )

            line_initial = (
                jnp.asarray(0),
                gradient_converged,
                blocks,
                value,
                gradient,
                step_size,
            )
            (
                _,
                accepted,
                next_blocks,
                next_value,
                next_gradient,
                accepted_step_size,
            ) = jax.lax.while_loop(continue_line_search, line_search_step, line_initial)
            relative_improvement = (value - next_value) / jnp.maximum(
                jnp.abs(value), 1.0
            )
            improvement_stalled = (
                accepted
                & ~gradient_converged
                & (relative_improvement <= self._policy.relative_improvement_tolerance)
            )
            return (
                iteration + 1,
                next_blocks,
                next_value,
                next_gradient,
                jnp.minimum(
                    self._policy.initial_step_size,
                    2.0 * accepted_step_size,
                ),
                gradient_converged,
                improvement_stalled,
                ~accepted,
                progressed | (accepted & ~gradient_converged),
            )

        initial_carry = (
            jnp.asarray(0),
            initial_blocks,
            initial_value,
            initial_gradient,
            jnp.asarray(self._policy.initial_step_size),
            jnp.asarray(False),
            jnp.asarray(False),
            jnp.asarray(False),
            jnp.asarray(False),
        )
        (
            iteration,
            blocks,
            value,
            gradient,
            _,
            converged,
            stalled,
            line_search_failed,
            progressed,
        ) = jax.lax.while_loop(continue_outer, outer_step, initial_carry)
        return (
            blocks,
            value,
            gradient,
            iteration,
            converged,
            stalled,
            line_search_failed,
            progressed,
            finite(value, gradient),
        )

    def _exogenous_forecast(self, reference: ReferenceTrajectory) -> Array:
        if reference.exogenous is None:
            return jnp.zeros((self.prediction_steps, self.model.exogenous_size))
        return reference.exogenous

    def _cold_blocks(self, previous_command: Array) -> Array:
        normalized = self._normalized_from_commands(previous_command)
        return jnp.repeat(normalized[None, :], self.command_block_count, axis=0)

    def _warm_blocks(self, warm_start: NMPCWarmStart) -> Array | None:
        """Recover the previous plan's blocks and advance them by one block.

        The plan is parameterized at block granularity, so the seed is shifted
        at that granularity too: its first block is the previous plan's second
        block, and its final block repeats the previous plan's last block.
        Shifting the expanded command sequence by a single model step instead
        would land back inside the same old block whenever a block spans more
        than one step, which reproduces the previous plan unshifted.
        """

        commands = warm_start.commands
        if commands.shape != (self.prediction_steps, self.model.command_size):
            return None
        indices = jnp.minimum(
            jnp.arange(self.command_block_count) * self._block_steps,
            self.prediction_steps - 1,
        )
        blocks = commands[indices]
        shifted = jnp.concatenate((blocks[1:], blocks[-1:]), axis=0)
        return jnp.clip(self._normalized_from_commands(shifted), -1.0, 1.0)

    def _input_error(
        self,
        state: Array,
        reference: ReferenceTrajectory,
        previous_command: Array,
        applied_command: Array | None,
        latent_state: Array | None,
    ) -> str | None:
        state_array = np.asarray(state)
        previous_array = np.asarray(previous_command)
        if state_array.shape != (13,) or not np.all(np.isfinite(state_array)):
            return "state estimate must be a finite 13-element vector"
        if np.linalg.norm(state_array[6:10]) < 1e-6:
            return "state estimate quaternion has zero norm"
        if reference.states.shape != (self.prediction_steps + 1, 13):
            return "reference length does not match the controller horizon"
        if previous_array.shape != (self.model.command_size,) or not np.all(
            np.isfinite(previous_array)
        ):
            return "previous command has the wrong shape or non-finite values"
        minimum = np.asarray(self.model.command_minimum)
        maximum = np.asarray(self.model.command_maximum)
        # A measured or saturated command routinely lands a rounding step
        # outside its own bound. Accept that band and clip it rather than
        # rejecting a command the vehicle actually holds.
        tolerance = _COMMAND_BOUND_RELATIVE_TOLERANCE * (maximum - minimum)
        if np.any(previous_array < minimum - tolerance) or np.any(
            previous_array > maximum + tolerance
        ):
            return "previous command lies outside the command bounds"
        if reference.exogenous is not None and reference.exogenous.shape != (
            self.prediction_steps,
            self.model.exogenous_size,
        ):
            return "exogenous forecast does not match the runtime model"
        if applied_command is not None:
            applied_array = np.asarray(applied_command)
            if applied_array.shape != (self.model.command_size,) or not np.all(
                np.isfinite(applied_array)
            ):
                return "applied command has the wrong shape or non-finite values"
            if np.any(applied_array < minimum - tolerance) or np.any(
                applied_array > maximum + tolerance
            ):
                return "applied command lies outside the command bounds"
        if applied_command is not None and latent_state is not None:
            return "provide either applied_command or latent_state, not both"
        if latent_state is not None:
            latent_array = np.asarray(latent_state)
            if latent_array.shape != (self.model.latent_size,) or not np.all(
                np.isfinite(latent_array)
            ):
                return "latent applied-control state is invalid"
        return None

    def _fallback_command(self, previous_command: Array) -> Array:
        previous = np.asarray(previous_command)
        if previous.shape != (self.model.command_size,) or not np.all(
            np.isfinite(previous)
        ):
            previous = 0.5 * (
                np.asarray(self.model.command_minimum)
                + np.asarray(self.model.command_maximum)
            )
        return jnp.clip(
            jnp.asarray(previous),
            self.model.command_minimum,
            self.model.command_maximum,
        )

    def _failure_result(
        self,
        status: SolveStatus,
        message: str,
        previous_command: Array,
        started_at: float,
        *,
        initial_objective: float = math.inf,
        iterations: int = 0,
        warm_start_used: bool = False,
    ) -> NMPCResult:
        fallback = self._fallback_command(previous_command)
        commands = jnp.repeat(fallback[None, :], self.prediction_steps, axis=0)
        certified = self.model.runtime_spec.certified_prediction_horizon_s
        return NMPCResult(
            status=status,
            command=fallback,
            predicted_states=jnp.empty((0, 13)),
            predicted_latent_states=jnp.empty((0, self.model.latent_size)),
            predicted_commands=commands,
            warm_start=None,
            diagnostics=NMPCDiagnostics(
                iterations=iterations,
                solve_time_s=time.perf_counter() - started_at,
                initial_objective=initial_objective,
                final_objective=math.inf,
                final_gradient_inf_norm=math.inf,
                final_projected_gradient_inf_norm=math.inf,
                maximum_command_bound_violation=0.0,
                maximum_validity_utilization=math.inf,
                maximum_normalized_safety_violation=math.inf,
                maximum_normalized_model_uncertainty_standard_deviation=(
                    math.inf if self.belief.uncertainty_available else 0.0
                ),
                command_authority_fraction=0.0,
                uncertainty_aware_command_selection=(self.belief.uncertainty_available),
                model_uncertainty_available=self.belief.uncertainty_available,
                prediction_error_model_available=(
                    self.belief.predictive_error_available
                ),
                prediction_error_model_current=self.belief.predictive_error_current,
                prediction_error_horizon_supported=False,
                parameter_uncertainty_available=(
                    self.belief.parameter_uncertainty_available
                ),
                warm_start_used=warm_start_used,
                prediction_horizon_s=self.prediction_horizon_s,
                prediction_horizon_certified=(
                    certified is not None
                    and self.prediction_horizon_s <= certified + 1e-12
                ),
                support_filter_mode=SupportFilterMode.SOLVER_FALLBACK,
                support_filter_applied=False,
                support_command_fraction=0.0,
                current_validity_utilization=math.inf,
                next_step_mean_validity_utilization=math.inf,
                next_step_robust_validity_utilization=math.inf,
                current_angular_rate_energy=math.inf,
                next_step_angular_rate_energy=math.inf,
                support_horizon_s=(
                    self._support_horizon_steps
                    * self.model.runtime_spec.sample_period_s
                ),
                support_horizon_maximum_robust_validity_utilization=math.inf,
                support_horizon_terminal_robust_validity_utilization=math.inf,
                support_horizon_terminal_angular_rate_energy=math.inf,
            ),
            used_fallback=True,
            message=message,
        )

    def _reject_invalid_request(
        self,
        state: Array,
        reference: ReferenceTrajectory,
        previous_command: Array,
        applied_command: Array | None,
        latent_state: Array | None,
        deadline_s: float | None,
    ) -> None:
        """Refuse a request the bounded solver cannot act on at all."""

        input_error = self._input_error(
            state,
            reference,
            previous_command,
            applied_command,
            latent_state,
        )
        if input_error is not None:
            raise _SolveAbort(SolveStatus.INVALID_INPUT, input_error)
        if deadline_s is not None and (
            not np.isfinite(deadline_s) or deadline_s <= 0.0
        ):
            raise _SolveAbort(
                SolveStatus.DEADLINE_EXCEEDED,
                "deadline must be finite and positive",
            )

    def _require_deadline(
        self,
        deadline_s: float | None,
        progress: _SolveProgress,
        message: str,
    ) -> None:
        """Abort the solve when it has already spent its deadline."""

        if (
            deadline_s is not None
            and time.perf_counter() - progress.started_at >= deadline_s
        ):
            raise _SolveAbort(SolveStatus.DEADLINE_EXCEEDED, message)

    def _initial_latent(
        self,
        previous_command: Array,
        applied_command: Array | None,
        latent_state: Array | None,
    ) -> Array:
        """Return the actuator state the predicted horizon starts from.

        A caller that knows the actuator state supplies it; otherwise it is
        reconstructed from the command the vehicle is actually holding, which is
        the applied command when one was measured and the previous command
        when it was not.
        """

        if latent_state is not None:
            return jnp.asarray(latent_state)
        return self._initial_latent_compiled(
            self._active_parameters,
            previous_command
            if applied_command is None
            else jnp.clip(
                jnp.asarray(applied_command),
                self.model.command_minimum,
                self.model.command_maximum,
            ),
        )

    def _seed_plan(
        self,
        cold_blocks: Array,
        warm_start: NMPCWarmStart | None,
        state: Array,
        latent: Array,
        reference: ReferenceTrajectory,
        previous_command: Array,
        exogenous: Array,
    ) -> tuple[Array, Array, Array, float, bool]:
        """Score the cold seed and adopt a warm seed only when it is no worse.

        A warm start that is non-finite, the wrong shape, or simply worse than
        holding the previous command is discarded rather than trusted, so a
        stale plan can never make this solve start behind a cold start.
        """

        value, gradient = self._objective_and_gradient(
            cold_blocks,
            state,
            latent,
            reference.states,
            previous_command,
            exogenous,
            self._active_parameters,
        )
        value_float = float(np.asarray(value))
        if not np.isfinite(value_float) or not np.all(
            np.isfinite(np.asarray(gradient))
        ):
            raise _SolveAbort(
                SolveStatus.NONFINITE_OBJECTIVE,
                "cold-start objective or gradient is non-finite",
            )
        blocks = cold_blocks
        used_warm_start = False
        if warm_start is not None:
            warm_blocks = self._warm_blocks(warm_start)
            if warm_blocks is not None:
                warm_value, warm_gradient = self._objective_and_gradient(
                    warm_blocks,
                    state,
                    latent,
                    reference.states,
                    previous_command,
                    exogenous,
                    self._active_parameters,
                )
                warm_value_float = float(np.asarray(warm_value))
                if (
                    np.isfinite(warm_value_float)
                    and np.all(np.isfinite(np.asarray(warm_gradient)))
                    and warm_value_float <= value_float
                ):
                    blocks = warm_blocks
                    value = warm_value
                    value_float = warm_value_float
                    gradient = warm_gradient
                    used_warm_start = True
        return blocks, value, gradient, value_float, used_warm_start

    def _optimize_plan(
        self,
        blocks: Array,
        value: Array,
        gradient: Array,
        state: Array,
        latent: Array,
        reference: ReferenceTrajectory,
        previous_command: Array,
        exogenous: Array,
    ) -> _OptimizerOutcome:
        """Run the bounded projected line search and report what it found."""

        (
            blocks,
            value,
            gradient,
            iteration_array,
            converged_array,
            stalled_array,
            line_search_failed_array,
            progressed_array,
            finite_array,
        ) = self._optimize_compiled(
            blocks,
            value,
            gradient,
            state,
            latent,
            reference.states,
            previous_command,
            exogenous,
            self._active_parameters,
        )
        return _OptimizerOutcome(
            blocks=blocks,
            value=value,
            gradient=gradient,
            iterations=int(np.asarray(iteration_array)),
            converged=bool(np.asarray(converged_array)),
            stalled=bool(np.asarray(stalled_array)),
            line_search_failed=bool(np.asarray(line_search_failed_array)),
            progressed=bool(np.asarray(progressed_array)),
            finite=bool(np.asarray(finite_array)),
            stall_message=(
                "finite best plan returned after improvement stalled short of "
                "the first-order criterion"
            ),
        )

    def _accept_optimizer_outcome(self, outcome: _OptimizerOutcome) -> None:
        """Refuse an unusable optimizer result, or record a line-search stall.

        A line search that failed only after an earlier outer iteration was
        accepted still leaves a finite improvement on the seed, so that progress
        is reported as a stall instead of being discarded for a hold.
        """

        if not outcome.finite:
            raise _SolveAbort(
                SolveStatus.NONFINITE_OBJECTIVE,
                "objective or gradient became non-finite during optimization",
            )
        if outcome.line_search_failed and not outcome.progressed:
            raise _SolveAbort(
                SolveStatus.LINE_SEARCH_FAILED,
                "bounded line search could not find a finite descent step",
            )
        if outcome.line_search_failed:
            outcome.stalled = True
            outcome.stall_message = (
                "finite best plan returned after the bounded line search "
                "stalled short of the first-order criterion"
            )

    def _evaluate_blocks(
        self,
        blocks: Array,
        state: Array,
        latent: Array,
        reference: ReferenceTrajectory,
        previous_command: Array,
        exogenous: Array,
        *,
        value: Array | None = None,
        gradient: Array | None = None,
    ) -> _PlanEvaluation:
        """Score one command plan and roll it out across the whole horizon.

        Every site that edits the command blocks after optimization, namely the
        uncertainty-bounded authority scaling and the support filter, has to
        recompute the objective, the gradient norms, and the prediction that go
        with the edited plan.  ``value`` and ``gradient`` are passed only by the
        caller that already holds them for exactly these blocks, which is how
        the optimizer's own result is carried through without re-evaluating it.
        """

        if value is None or gradient is None:
            value, gradient = self._objective_and_gradient(
                blocks,
                state,
                latent,
                reference.states,
                previous_command,
                exogenous,
                self._active_parameters,
            )
        states, latent_states, commands = self._rollout_compiled(
            blocks,
            state,
            latent,
            exogenous,
            self._active_parameters,
        )
        return _PlanEvaluation(
            blocks=blocks,
            value=value,
            gradient=gradient,
            value_float=float(np.asarray(value)),
            gradient_inf_norm=float(np.max(np.abs(np.asarray(gradient)))),
            projected_gradient_inf_norm=float(
                np.asarray(_projected_gradient_norm(blocks, gradient))
            ),
            states=states,
            latent_states=latent_states,
            commands=commands,
            states_np=np.asarray(states),
            latent_np=np.asarray(latent_states),
            commands_np=np.asarray(commands),
        )

    def _bounded_authority_plan(
        self,
        plan: _PlanEvaluation,
        cold_blocks: Array,
        maximum_model_uncertainty: float,
        support_metrics: tuple[float, float, float, float, float, float],
        outcome: _OptimizerOutcome,
        state: Array,
        latent: Array,
        reference: ReferenceTrajectory,
        previous_command: Array,
        exogenous: Array,
    ) -> tuple[
        float,
        _PlanEvaluation,
        float,
        tuple[float, float, float, float, float, float],
    ]:
        """Pull the plan toward the previous command when the belief is weak.

        Command authority is the reciprocal of the largest normalized predicted
        uncertainty, so a forecast whose spread already fills the tracking
        tolerance keeps only the fraction of the optimized departure the belief
        can still stand behind.  The scaled plan is a different plan, so it is
        rescored and its uncertainty forecast is recomputed.
        """

        command_authority = (
            min(1.0, 1.0 / maximum_model_uncertainty)
            if self.belief.uncertainty_available and maximum_model_uncertainty > 0.0
            else 1.0
        )
        if command_authority < 1.0:
            plan = self._evaluate_blocks(
                jnp.clip(
                    cold_blocks + command_authority * (plan.blocks - cold_blocks),
                    -1.0,
                    1.0,
                ),
                state,
                latent,
                reference,
                previous_command,
                exogenous,
            )
            (
                maximum_model_uncertainty,
                support_metrics,
            ) = self._uncertainty_support_values(
                state,
                latent,
                plan.commands,
                exogenous,
            )
            outcome.converged = False
            outcome.stalled = False
        if not (plan.prediction_finite and np.isfinite(maximum_model_uncertainty)):
            raise _SolveAbort(
                SolveStatus.NONFINITE_OBJECTIVE,
                "uncertainty-bounded prediction is non-finite",
            )
        return command_authority, plan, maximum_model_uncertainty, support_metrics

    def _support_filtered_plan(
        self,
        plan: _PlanEvaluation,
        support_metrics: tuple[float, float, float, float, float, float],
        maximum_model_uncertainty: float,
        outcome: _OptimizerOutcome,
        state: Array,
        latent: Array,
        reference: ReferenceTrajectory,
        previous_command: Array,
        exogenous: Array,
    ) -> tuple[_SupportDecision, _PlanEvaluation, float]:
        """Project the first command onto the support the belief actually has.

        Only the first block is edited, because only the first command is
        applied; the rest of the horizon is left for the next solve to revisit.
        As with authority scaling, an edited plan is rescored before it is
        reported.
        """

        support_decision = self._select_support_command(
            state,
            latent,
            plan.commands[0],
            previous_command,
            exogenous[0],
            support_metrics,
        )
        if support_decision.applied:
            plan = self._evaluate_blocks(
                plan.blocks.at[0].set(
                    jnp.clip(
                        self._normalized_from_commands(
                            jnp.asarray(support_decision.command)
                        ),
                        -1.0,
                        1.0,
                    )
                ),
                state,
                latent,
                reference,
                previous_command,
                exogenous,
            )
            maximum_model_uncertainty, _ = self._uncertainty_support_values(
                state,
                latent,
                plan.commands,
                exogenous,
            )
            outcome.converged = False
            outcome.stalled = False
        if not (plan.prediction_finite and np.isfinite(maximum_model_uncertainty)):
            raise _SolveAbort(
                SolveStatus.NONFINITE_OBJECTIVE,
                "support-filtered prediction is non-finite",
            )
        return support_decision, plan, maximum_model_uncertainty

    def _prediction_diagnostics(
        self,
        plan: _PlanEvaluation,
        exogenous: Array,
    ) -> _PredictionDiagnostics:
        """Measure the bound, validity, and safety margins of the final plan."""

        minimum = np.asarray(self.model.command_minimum)
        maximum = np.asarray(self.model.command_maximum)
        return _PredictionDiagnostics(
            maximum_command_bound_violation=float(
                max(
                    np.max(minimum - plan.commands_np),
                    np.max(plan.commands_np - maximum),
                    0.0,
                )
            ),
            maximum_validity_utilization=float(
                np.asarray(self._validity_compiled(plan.states, exogenous))
            ),
            maximum_normalized_safety_violation=float(
                np.asarray(self._safety_compiled(plan.states))
            ),
        )

    def _solved_result(
        self,
        plan: _PlanEvaluation,
        outcome: _OptimizerOutcome,
        support_decision: _SupportDecision,
        prediction: _PredictionDiagnostics,
        maximum_model_uncertainty: float,
        command_authority: float,
        progress: _SolveProgress,
    ) -> NMPCResult:
        """Assemble the auditable result for one finite, bounded solve."""

        certified = self.model.runtime_spec.certified_prediction_horizon_s
        status = (
            SolveStatus.CONVERGED
            if outcome.converged
            else SolveStatus.STALLED
            if outcome.stalled
            else SolveStatus.ITERATION_LIMIT
        )
        return NMPCResult(
            status=status,
            command=plan.commands[0],
            predicted_states=plan.states,
            predicted_latent_states=plan.latent_states,
            predicted_commands=plan.commands,
            warm_start=NMPCWarmStart(plan.commands),
            diagnostics=NMPCDiagnostics(
                iterations=outcome.iterations,
                solve_time_s=time.perf_counter() - progress.started_at,
                initial_objective=progress.initial_objective,
                final_objective=plan.value_float,
                final_gradient_inf_norm=plan.gradient_inf_norm,
                final_projected_gradient_inf_norm=(plan.projected_gradient_inf_norm),
                maximum_command_bound_violation=(
                    prediction.maximum_command_bound_violation
                ),
                maximum_validity_utilization=(prediction.maximum_validity_utilization),
                maximum_normalized_safety_violation=(
                    prediction.maximum_normalized_safety_violation
                ),
                maximum_normalized_model_uncertainty_standard_deviation=(
                    maximum_model_uncertainty
                ),
                command_authority_fraction=command_authority,
                uncertainty_aware_command_selection=(self.belief.uncertainty_available),
                model_uncertainty_available=self.belief.uncertainty_available,
                prediction_error_model_available=(
                    self.belief.predictive_error_available
                ),
                prediction_error_model_current=self.belief.predictive_error_current,
                prediction_error_horizon_supported=(
                    self.belief.maximum_error_horizon_s is not None
                    and self.prediction_horizon_s
                    <= self.belief.maximum_error_horizon_s + 1e-12
                ),
                parameter_uncertainty_available=(
                    self.belief.parameter_uncertainty_available
                ),
                warm_start_used=progress.warm_start_used,
                prediction_horizon_s=self.prediction_horizon_s,
                prediction_horizon_certified=(
                    certified is not None
                    and self.prediction_horizon_s <= certified + 1e-12
                ),
                support_filter_mode=support_decision.mode,
                support_filter_applied=support_decision.applied,
                support_command_fraction=support_decision.nominal_fraction,
                current_validity_utilization=support_decision.current_validity,
                next_step_mean_validity_utilization=(
                    support_decision.next_mean_validity
                ),
                next_step_robust_validity_utilization=(
                    support_decision.next_robust_validity
                ),
                current_angular_rate_energy=(support_decision.current_rate_energy),
                next_step_angular_rate_energy=support_decision.next_rate_energy,
                support_horizon_s=support_decision.support_horizon_s,
                support_horizon_maximum_robust_validity_utilization=(
                    support_decision.support_horizon_maximum_robust_validity
                ),
                support_horizon_terminal_robust_validity_utilization=(
                    support_decision.support_horizon_terminal_robust_validity
                ),
                support_horizon_terminal_angular_rate_energy=(
                    support_decision.support_horizon_terminal_rate_energy
                ),
            ),
            used_fallback=False,
            message=(
                f"finite plan returned with {support_decision.mode.value} projection"
                if support_decision.applied
                or support_decision.mode
                in {
                    SupportFilterMode.RECOVERY_FILTERED,
                    SupportFilterMode.RECOVERY_BEST_EFFORT,
                    SupportFilterMode.BOUNDARY_BEST_EFFORT,
                }
                else "finite plan returned with belief-bounded command authority"
                if command_authority < 1.0
                else "first-order convergence criterion satisfied"
                if outcome.converged
                else outcome.stall_message
                if outcome.stalled
                else "finite best plan returned at the maintained iteration limit"
            ),
        )

    def solve(
        self,
        state: Array,
        reference: ReferenceTrajectory,
        previous_command: Array,
        *,
        applied_command: Array | None = None,
        latent_state: Array | None = None,
        warm_start: NMPCWarmStart | None = None,
        deadline_s: float | None = None,
    ) -> NMPCResult:
        """Optimize one bounded command and return an auditable receding horizon.

        The solve is a fixed sequence of steps: refuse an unusable request, seed
        the plan, optimize it, roll it out, bound its authority by the belief's
        own uncertainty, project the first command onto supported ground, and
        measure the result.  Any step may abort, and every abort returns the
        same bounded previous-command hold rather than raising.
        """

        started_at = time.perf_counter()
        progress = _SolveProgress(
            started_at=started_at,
            previous_command=previous_command,
        )
        try:
            self._reject_invalid_request(
                state,
                reference,
                previous_command,
                applied_command,
                latent_state,
                deadline_s,
            )
            state = jnp.asarray(state)
            # Validation accepts a rounding step outside the bounds; everything
            # downstream sees a strictly bounded command.
            previous_command = jnp.clip(
                jnp.asarray(previous_command),
                self.model.command_minimum,
                self.model.command_maximum,
            )
            progress.previous_command = previous_command
            latent = self._initial_latent(
                previous_command,
                applied_command,
                latent_state,
            )
            exogenous = self._exogenous_forecast(reference)
            cold_blocks = self._cold_blocks(previous_command)

            blocks, value, gradient, value_float, used_warm_start = self._seed_plan(
                cold_blocks,
                warm_start,
                state,
                latent,
                reference,
                previous_command,
                exogenous,
            )
            progress.initial_objective = value_float
            progress.warm_start_used = used_warm_start
            self._require_deadline(
                deadline_s,
                progress,
                "solver deadline expired before optimization",
            )

            outcome = self._optimize_plan(
                blocks,
                value,
                gradient,
                state,
                latent,
                reference,
                previous_command,
                exogenous,
            )
            progress.iterations = outcome.iterations
            self._require_deadline(
                deadline_s,
                progress,
                "solver deadline expired during optimization",
            )
            self._accept_optimizer_outcome(outcome)

            plan = self._evaluate_blocks(
                outcome.blocks,
                state,
                latent,
                reference,
                previous_command,
                exogenous,
                value=outcome.value,
                gradient=outcome.gradient,
            )
            if not plan.prediction_finite:
                raise _SolveAbort(
                    SolveStatus.NONFINITE_OBJECTIVE,
                    "optimized prediction is non-finite",
                )
            (
                maximum_model_uncertainty,
                support_metrics,
            ) = self._uncertainty_support_values(
                state,
                latent,
                plan.commands,
                exogenous,
            )
            if not np.isfinite(maximum_model_uncertainty):
                raise _SolveAbort(
                    SolveStatus.NONFINITE_OBJECTIVE,
                    "model-uncertainty forecast is non-finite",
                )

            (
                command_authority,
                plan,
                maximum_model_uncertainty,
                support_metrics,
            ) = self._bounded_authority_plan(
                plan,
                cold_blocks,
                maximum_model_uncertainty,
                support_metrics,
                outcome,
                state,
                latent,
                reference,
                previous_command,
                exogenous,
            )
            (
                support_decision,
                plan,
                maximum_model_uncertainty,
            ) = self._support_filtered_plan(
                plan,
                support_metrics,
                maximum_model_uncertainty,
                outcome,
                state,
                latent,
                reference,
                previous_command,
                exogenous,
            )

            prediction = self._prediction_diagnostics(plan, exogenous)
            self._require_deadline(
                deadline_s,
                progress,
                "solver deadline expired during prediction diagnostics",
            )
        except _SolveAbort as abort:
            return self._failure_result(
                abort.status,
                abort.message,
                progress.previous_command,
                progress.started_at,
                initial_objective=progress.initial_objective,
                iterations=progress.iterations,
                warm_start_used=progress.warm_start_used,
            )
        return self._solved_result(
            plan,
            outcome,
            support_decision,
            prediction,
            maximum_model_uncertainty,
            command_authority,
            progress,
        )


class NMPCController:
    """Opinionated NMPC facade over one actionable Glassbox runtime model."""

    def __init__(
        self,
        model: RuntimeDynamicsModel | RuntimeDynamicsBelief | DynamicsBelief,
        tolerances: TrackingTolerances | None = None,
        safety_envelope: SafetyEnvelope | None = None,
        *,
        _policy: _SolverPolicy | None = None,
    ) -> None:
        belief = _runtime_belief(model)
        self.belief = belief
        self.model = belief.nominal
        self.tolerances = (
            TrackingTolerances.for_platform(self.model.input_spec.vehicle.family)
            if tolerances is None
            else tolerances
        )
        self.safety_envelope = (
            SafetyEnvelope() if safety_envelope is None else safety_envelope
        )
        self._backend: _SolverBackend = _DirectShootingBackend(
            belief,
            self.tolerances,
            self.safety_envelope,
            _policy=_policy,
        )

    def rebind_belief(
        self,
        model: RuntimeDynamicsModel | RuntimeDynamicsBelief | DynamicsBelief,
    ) -> NMPCController:
        """Share precompiled kernels with a structurally compatible belief.

        Only dynamic model-parameter values may change. Static runtime,
        actuation, uncertainty, solver-horizon, and support-horizon semantics
        remain those that were compiled and validated on this controller.
        """

        belief = _runtime_belief(model)
        rebound = object.__new__(NMPCController)
        rebound.belief = belief
        rebound.model = belief.nominal
        rebound.tolerances = self.tolerances
        rebound.safety_envelope = self.safety_envelope
        rebound._backend = self._backend.rebind(belief)
        return rebound

    @property
    def prediction_steps(self) -> int:
        return self._backend.prediction_steps

    @property
    def prediction_horizon_s(self) -> float:
        return self._backend.prediction_horizon_s

    def hold_reference(
        self, state: Array, *, exogenous: Array | None = None
    ) -> ReferenceTrajectory:
        """Build the common regulation reference for this controller."""

        return ReferenceTrajectory.hold(
            state, self.prediction_steps, exogenous=exogenous
        )

    def solve(
        self,
        state: Array,
        reference: ReferenceTrajectory,
        previous_command: Array,
        *,
        applied_command: Array | None = None,
        latent_state: Array | None = None,
        warm_start: NMPCWarmStart | None = None,
        deadline_s: float | None = None,
    ) -> NMPCResult:
        """Optimize one bounded command and return an auditable receding horizon."""

        return self._backend.solve(
            state,
            reference,
            previous_command,
            applied_command=applied_command,
            latent_state=latent_state,
            warm_start=warm_start,
            deadline_s=deadline_s,
        )
