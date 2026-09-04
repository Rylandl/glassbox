"""Warm-started bounded direct-shooting NMPC backend."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from glassbox.belief.belief import (
    DynamicsBelief,
    EmpiricalHorizonPredictiveError,
    ErrorCovarianceScope,
    structured_parameter_vector,
    with_structured_parameter_vector,
)
from glassbox.control.nmpc.types import (
    NMPCDiagnostics,
    NMPCResult,
    NMPCWarmStart,
    ReferenceTrajectory,
    SafetyEnvelope,
    SolveStatus,
    TrackingTolerances,
)
from glassbox.core.data import duration_to_steps
from glassbox.core.dynamics import ModelParams, quaternion_to_rotation
from glassbox.core.geometry import rigid_body_local_error
from glassbox.core.model import ExecutableModel, NonActionableModelError

_MAXIMUM_COMMAND_BLOCKS = 10
_COMMAND_BOUND_RELATIVE_TOLERANCE = 1e-6
_MINIMUM_COVARIANCE_EIGENVALUE_FRACTION = 1e-10

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


def _marginal_standard_deviation(variance: Array) -> Array:
    """Square root of a variance whose gradient is finite where it vanishes.

    The square root has an infinite slope at zero, and a point model predicts
    exactly zero spread on every axis at every stage. Differentiating the naive
    expression therefore hands the line search a non-finite gradient on the one
    case that must reduce to the point objective. The masked form returns zero
    with a zero derivative there, which is the derivative of the constant the
    function actually is when the belief carries no covariance.
    """

    positive = variance > 0.0
    return jnp.where(positive, jnp.sqrt(jnp.where(positive, variance, 1.0)), 0.0)


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
class SolverPolicy:
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
class _PlanEvaluation:
    """One command plan with its objective, gradient norm, and prediction."""

    blocks: Array
    value: Array
    gradient: Array
    value_float: float
    projected_gradient_inf_norm: float
    maximum_normalized_uncertainty: float
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
    """What the bounded optimizer produced and how it terminated."""

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


def _default_policy(model: ExecutableModel) -> SolverPolicy:
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
    return SolverPolicy(
        horizon_steps=steps,
        block_count=_maintained_block_count(steps),
    )


def _parameter_covariance_factor(belief: DynamicsBelief) -> np.ndarray | None:
    """Return a factor ``L`` with ``L @ L.T`` equal to the parameter covariance.

    The objective charges predicted spread, and the parameter contribution to
    that spread is ``J C J.T`` for the plan's parameter Jacobian ``J``. Written
    through a factor it becomes a sum of ``len(L.T)`` directional derivatives,
    each one forward-mode rollout, so a covariance the evidence resolved along
    one direction costs one extra rollout instead of a full Jacobian.

    ``None`` means the belief contributes no parameter spread, either because
    it carries no parameter uncertainty or because its forecast-error evidence
    already scopes the total forecast error, which is what
    :attr:`PredictiveTrajectory.tangent_covariance` reports in that case.
    """

    if not belief.parameter_uncertainty_available:
        return None
    scope = (
        belief.predictive_error.covariance_scope
        if isinstance(belief.predictive_error, EmpiricalHorizonPredictiveError)
        and belief.predictive_error_current
        else None
    )
    if scope == ErrorCovarianceScope.TOTAL_FORECAST:
        return None
    covariance = np.asarray(belief.parameter_belief.covariance, dtype=np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (covariance + covariance.T))
    floor = _MINIMUM_COVARIANCE_EIGENVALUE_FRACTION * max(float(eigenvalues[-1]), 0.0)
    retained = eigenvalues > max(floor, 0.0)
    if not np.any(retained):
        return None
    return eigenvectors[:, retained] * np.sqrt(eigenvalues[retained])


def _controlled_belief(model: ExecutableModel | DynamicsBelief) -> DynamicsBelief:
    """Accept either a bare executable model or the belief that carries one."""

    belief = model if isinstance(model, DynamicsBelief) else DynamicsBelief(model)
    if belief.model.actuation is None:
        # Fail closed here rather than at the first solve: a model whose inputs
        # are observations of actuation has no command space to optimize over.
        raise NonActionableModelError(
            "NMPC needs a command space; this model has no actuation map"
        )
    return belief


class _DirectShootingBackend:
    """The bounded direct-shooting implementation NMPCController drives."""

    def __init__(
        self,
        belief: DynamicsBelief,
        tolerances: TrackingTolerances,
        safety_envelope: SafetyEnvelope,
        *,
        policy: SolverPolicy | None = None,
    ) -> None:
        self.belief = belief
        self.model = belief.model
        self._active_parameters = self.model.params
        self.tolerances = tolerances
        self.safety_envelope = safety_envelope
        self._policy = _default_policy(self.model) if policy is None else policy
        if policy is None and belief.maximum_error_horizon_s is not None:
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
        self._covariance_factor = _parameter_covariance_factor(belief)
        self._objective_gradient = jax.value_and_grad(self._objective)
        self._objective_and_gradient = jax.jit(self._objective_gradient)
        self._initial_latent_compiled = jax.jit(
            self.model.initial_latent_state_with_parameters
        )
        self._optimize_compiled = jax.jit(self._optimize)
        self._rollout_compiled = jax.jit(self._rollout)
        self._validity_compiled = jax.jit(self._maximum_validity_utilization)
        self._safety_compiled = jax.jit(self._maximum_safety_violation)
        self._uncertainty_compiled = jax.jit(self._maximum_normalized_uncertainty)

    @property
    def prediction_steps(self) -> int:
        return self._policy.horizon_steps

    @property
    def prediction_horizon_s(self) -> float:
        return self.prediction_steps * self.model.runtime_spec.sample_period_s

    @property
    def command_block_count(self) -> int:
        return self._policy.block_count

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
    ) -> tuple[Array, Array, Array, Array]:
        """Predict the horizon and the forecast-error covariance along it.

        The returned covariance is the belief's own forecast-error covariance
        at each predicted stage, which the correction step already evaluates;
        the parameter contribution is added separately by
        :meth:`_tangent_covariance`, because it depends on the plan.
        """

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
        corrected_states, _, error_covariance = jax.vmap(self.belief.corrected_state)(
            future_states,
            horizons,
            commands,
            exogenous,
        )
        states = jnp.concatenate((initial_state[None, :], corrected_states), axis=0)
        latent = jnp.concatenate((initial_latent[None, :], future_latent), axis=0)
        return states, latent, commands, error_covariance

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
        """Score one command plan as the expected cost of its own forecast.

        Tracking charges ``E[l] = l(mean) + trace(W Sigma)`` at every predicted
        stage, where ``W`` is the diagonal tracking weight built from the
        declared tolerances and ``Sigma`` is the belief's predicted tangent
        covariance. Validity charges the robust utilization, the mean
        utilization plus the belief's own marginal radius on the same envelope
        features. A belief that carries no covariance contributes exactly zero
        to both, so a point model is scored by the point objective.
        """

        states, _, commands, error_covariance = self._rollout(
            blocks,
            initial_state,
            initial_latent,
            exogenous,
            model_parameters,
        )
        covariance = self._tangent_covariance(
            blocks,
            initial_state,
            initial_latent,
            exogenous,
            model_parameters,
            states,
            error_covariance,
        )
        local_error = jax.vmap(rigid_body_local_error)(reference_states[1:], states[1:])
        normalized_error = local_error / self.tolerances.local_state_scale
        normalized_spread = jnp.sum(
            jnp.diagonal(covariance, axis1=-2, axis2=-1)
            / jnp.square(self.tolerances.local_state_scale),
            axis=1,
        )
        tracking_cost = jnp.mean(
            jnp.sum(jnp.square(normalized_error), axis=1) + normalized_spread
        )
        terminal_cost = self._policy.terminal_weight * (
            jnp.sum(jnp.square(normalized_error[-1])) + normalized_spread[-1]
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

        utilization = jax.vmap(self._robust_validity_utilization)(
            states[1:], covariance, exogenous
        )
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

    def _tangent_covariance(
        self,
        blocks: Array,
        initial_state: Array,
        initial_latent: Array,
        exogenous: Array,
        model_parameters: ModelParams,
        states: Array,
        error_covariance: Array,
    ) -> Array:
        """Add the plan's parameter spread to the forecast-error covariance.

        The parameter contribution is ``J C J.T`` for the Jacobian of the
        predicted tangent error with respect to the fitted parameters. Written
        through the covariance factor it is the sum of the outer products of
        one directional derivative per retained direction, and each of those is
        a single forward-mode rollout rather than a column of a full Jacobian.
        """

        if self._covariance_factor is None:
            return error_covariance
        center = structured_parameter_vector(model_parameters)

        def varied_error(vector: Array) -> Array:
            varied_parameters = with_structured_parameter_vector(
                model_parameters, vector
            )
            varied_states, _, _, _ = self._rollout(
                blocks,
                initial_state,
                initial_latent,
                exogenous,
                varied_parameters,
            )
            return jax.vmap(rigid_body_local_error)(states[1:], varied_states[1:])

        def direction(column: Array) -> Array:
            return jax.jvp(varied_error, (center,), (column,))[1]

        directions = jax.vmap(direction)(jnp.asarray(self._covariance_factor.T))
        return error_covariance + jnp.einsum("kti,ktj->tij", directions, directions)

    def _maximum_normalized_uncertainty(
        self,
        blocks: Array,
        initial_state: Array,
        initial_latent: Array,
        exogenous: Array,
        model_parameters: ModelParams,
    ) -> Array:
        """Largest predicted tangent spread in tracking-tolerance units."""

        states, _, _, error_covariance = self._rollout(
            blocks,
            initial_state,
            initial_latent,
            exogenous,
            model_parameters,
        )
        covariance = self._tangent_covariance(
            blocks,
            initial_state,
            initial_latent,
            exogenous,
            model_parameters,
            states,
            error_covariance,
        )
        standard_deviation = _marginal_standard_deviation(
            jnp.diagonal(covariance, axis1=-2, axis2=-1)
        )
        return jnp.max(standard_deviation / self.tolerances.local_state_scale[None, :])

    def _robust_validity_utilization(
        self,
        mean_state: Array,
        covariance: Array,
        exogenous: Array,
    ) -> Array:
        """Return per-axis validity utilization widened by its own spread.

        The envelope is stated on body velocity and body rates, so the tangent
        covariance is mapped onto those six features and the marginal standard
        deviation of each is added to that feature's mean utilization. A belief
        with no covariance adds exactly zero and the metric is the mean
        utilization it has always been.
        """

        envelope = self.model.runtime_spec.validity_envelope
        mean_utilization = self.model.validity_utilization(mean_state, exogenous)

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
            _marginal_standard_deviation(jnp.diag(feature_covariance))
            / feature_half_width
        )
        return mean_utilization + marginal_radius

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
                final_projected_gradient_inf_norm=math.inf,
                maximum_command_bound_violation=0.0,
                maximum_validity_utilization=math.inf,
                maximum_normalized_safety_violation=math.inf,
                maximum_normalized_model_uncertainty_standard_deviation=(
                    math.inf if self.belief.uncertainty_available else 0.0
                ),
                warm_start_used=warm_start_used,
                prediction_horizon_s=self.prediction_horizon_s,
                prediction_horizon_certified=(
                    certified is not None
                    and self.prediction_horizon_s <= certified + 1e-12
                ),
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
        exogenous: Array,
        value: Array,
        gradient: Array,
    ) -> _PlanEvaluation:
        """Roll out the optimized plan and measure what it forecasts.

        The optimizer already holds this plan's objective and gradient, so they
        are carried through rather than recomputed; nothing edits the command
        blocks after optimization, so no plan is ever scored twice.
        """

        states, latent_states, commands, _ = self._rollout_compiled(
            blocks,
            state,
            latent,
            exogenous,
            self._active_parameters,
        )
        maximum_uncertainty = float(
            np.asarray(
                self._uncertainty_compiled(
                    blocks,
                    state,
                    latent,
                    exogenous,
                    self._active_parameters,
                )
            )
        )
        return _PlanEvaluation(
            blocks=blocks,
            value=value,
            gradient=gradient,
            value_float=float(np.asarray(value)),
            projected_gradient_inf_norm=float(
                np.asarray(_projected_gradient_norm(blocks, gradient))
            ),
            maximum_normalized_uncertainty=maximum_uncertainty,
            states=states,
            latent_states=latent_states,
            commands=commands,
            states_np=np.asarray(states),
            latent_np=np.asarray(latent_states),
            commands_np=np.asarray(commands),
        )

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
        prediction: _PredictionDiagnostics,
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
                final_projected_gradient_inf_norm=(plan.projected_gradient_inf_norm),
                maximum_command_bound_violation=(
                    prediction.maximum_command_bound_violation
                ),
                maximum_validity_utilization=(prediction.maximum_validity_utilization),
                maximum_normalized_safety_violation=(
                    prediction.maximum_normalized_safety_violation
                ),
                maximum_normalized_model_uncertainty_standard_deviation=(
                    plan.maximum_normalized_uncertainty
                ),
                warm_start_used=progress.warm_start_used,
                prediction_horizon_s=self.prediction_horizon_s,
                prediction_horizon_certified=(
                    certified is not None
                    and self.prediction_horizon_s <= certified + 1e-12
                ),
            ),
            used_fallback=False,
            message=(
                "first-order convergence criterion satisfied"
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
        the plan, optimize it, roll it out, and measure the result. Any step may
        abort, and every abort returns the same bounded previous-command hold
        rather than raising.
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
                exogenous,
                outcome.value,
                outcome.gradient,
            )
            if not (
                plan.prediction_finite
                and np.isfinite(plan.maximum_normalized_uncertainty)
            ):
                raise _SolveAbort(
                    SolveStatus.NONFINITE_OBJECTIVE,
                    "optimized prediction is non-finite",
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
        return self._solved_result(plan, outcome, prediction, progress)


class NMPCController:
    """Opinionated NMPC facade over one actionable Glassbox runtime model."""

    def __init__(
        self,
        model: ExecutableModel | DynamicsBelief,
        tolerances: TrackingTolerances | None = None,
        safety_envelope: SafetyEnvelope | None = None,
        *,
        policy: SolverPolicy | None = None,
    ) -> None:
        belief = _controlled_belief(model)
        self.belief = belief
        self.model = belief.model
        self.tolerances = (
            TrackingTolerances.for_platform(self.model.input_spec.vehicle.family)
            if tolerances is None
            else tolerances
        )
        self.safety_envelope = (
            SafetyEnvelope() if safety_envelope is None else safety_envelope
        )
        self._backend = _DirectShootingBackend(
            belief,
            self.tolerances,
            self.safety_envelope,
            policy=policy,
        )

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
