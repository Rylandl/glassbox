"""Warm-started bounded direct-shooting NMPC backend."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from typing import Protocol

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from glassbox.belief import DynamicsBelief, RuntimeDynamicsBelief
from glassbox.geometry import rigid_body_local_error
from glassbox.nmpc.types import (
    NMPCDiagnostics,
    NMPCResult,
    NMPCWarmStart,
    ReferenceTrajectory,
    SafetyEnvelope,
    SolveStatus,
    TrackingTolerances,
)
from glassbox.runtime import RuntimeDynamicsModel


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

    def __post_init__(self) -> None:
        if self.horizon_steps < 1:
            raise ValueError("horizon_steps must be positive")
        if not 1 <= self.block_count <= self.horizon_steps:
            raise ValueError("block_count must be within the prediction horizon")
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


def _default_policy(model: RuntimeDynamicsModel) -> _SolverPolicy:
    dt_s = model.runtime_spec.sample_period_s
    target_horizon_s = 0.6 if model.input_spec.vehicle.family == "multirotor" else 1.0
    certified = model.runtime_spec.certified_prediction_horizon_s
    if certified is not None:
        target_horizon_s = min(target_horizon_s, certified)
    maximum_steps = 40 if model.input_spec.vehicle.family == "multirotor" else 50
    steps = min(maximum_steps, max(2, math.floor(target_horizon_s / dt_s)))
    if certified is not None and steps * dt_s > certified + 1e-12:
        steps = math.floor(certified / dt_s)
    if steps < 1:
        raise ValueError("certified prediction horizon is shorter than one model step")
    return _SolverPolicy(horizon_steps=steps, block_count=min(10, steps))


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
        self.tolerances = tolerances
        self.safety_envelope = safety_envelope
        self._policy = _default_policy(self.model) if _policy is None else _policy
        if _policy is None and belief.maximum_error_horizon_s is not None:
            supported_steps = math.floor(
                belief.maximum_error_horizon_s
                / self.model.runtime_spec.sample_period_s
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
                    block_count=min(self._policy.block_count, supported_steps),
                )
        horizon_s = (
            self._policy.horizon_steps * self.model.runtime_spec.sample_period_s
        )
        certified = self.model.runtime_spec.certified_prediction_horizon_s
        if certified is not None and horizon_s > certified + 1e-12:
            raise ValueError("solver horizon exceeds the model's certified horizon")
        self._block_steps = math.ceil(
            self._policy.horizon_steps / self._policy.block_count
        )
        self._objective_gradient = jax.value_and_grad(self._objective)
        self._objective_and_gradient = jax.jit(self._objective_gradient)
        self._initial_latent_compiled = jax.jit(self.model.initial_latent_state)
        self._optimize_compiled = jax.jit(self._optimize)
        self._rollout_compiled = jax.jit(self._rollout)
        self._validity_compiled = jax.jit(self._maximum_validity_utilization)
        self._safety_compiled = jax.jit(self._maximum_safety_violation)
        self._uncertainty_compiled = jax.jit(
            self._maximum_normalized_model_uncertainty
        )

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
    ) -> tuple[Array, Array, Array]:
        normalized_commands = self._expand_normalized_blocks(blocks)
        commands = self._commands_from_normalized(normalized_commands)

        def transition(
            carry: tuple[Array, Array], inputs: tuple[Array, Array]
        ) -> tuple[tuple[Array, Array], tuple[Array, Array]]:
            state, latent = carry
            command, context = inputs
            next_state, next_latent = self.model.transition(
                state, latent, command, context
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
        states = jnp.concatenate(
            (initial_state[None, :], corrected_states), axis=0
        )
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
    ) -> Array:
        states, _, commands = self._rollout(
            blocks, initial_state, initial_latent, exogenous
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

    def _maximum_normalized_model_uncertainty(
        self,
        initial_state: Array,
        initial_latent: Array,
        commands: Array,
        exogenous: Array,
    ) -> Array:
        forecast = self.belief.rollout(
            initial_state,
            commands,
            initial_latent_state=initial_latent,
            exogenous=exogenous,
        )
        return jnp.max(
            forecast.tangent_standard_deviation[1:]
            / self.tolerances.local_state_scale[None, :]
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
    ) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
        """Run the fixed maintained policy as one compiled JAX operation."""

        def finite(value: Array, gradient: Array) -> Array:
            return jnp.isfinite(value) & jnp.all(jnp.isfinite(gradient))

        def continue_outer(
            carry: tuple[Array, Array, Array, Array, Array, Array, Array],
        ) -> Array:
            iteration, _, value, gradient, _, converged, line_search_failed = carry
            return (
                (iteration < self._policy.maximum_iterations)
                & ~converged
                & ~line_search_failed
                & finite(value, gradient)
            )

        def outer_step(
            carry: tuple[Array, Array, Array, Array, Array, Array, Array],
        ) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
            iteration, blocks, value, gradient, step_size, _, _ = carry
            gradient_norm = jnp.max(jnp.abs(gradient))
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
                candidate = jnp.clip(
                    blocks - candidate_step_size * gradient, -1.0, 1.0
                )
                candidate_value, candidate_gradient = self._objective_gradient(
                    candidate,
                    initial_state,
                    initial_latent,
                    reference_states,
                    previous_command,
                    exogenous,
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
            ) = jax.lax.while_loop(
                continue_line_search, line_search_step, line_initial
            )
            relative_improvement = (value - next_value) / jnp.maximum(
                jnp.abs(value), 1.0
            )
            improvement_converged = (
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
                gradient_converged | improvement_converged,
                ~accepted,
            )

        initial_carry = (
            jnp.asarray(0),
            initial_blocks,
            initial_value,
            initial_gradient,
            jnp.asarray(self._policy.initial_step_size),
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
            line_search_failed,
        ) = jax.lax.while_loop(continue_outer, outer_step, initial_carry)
        return (
            blocks,
            value,
            gradient,
            iteration,
            converged,
            line_search_failed,
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
        commands = warm_start.commands
        if commands.shape != (self.prediction_steps, self.model.command_size):
            return None
        shifted = jnp.concatenate((commands[1:], commands[-1:]), axis=0)
        indices = jnp.minimum(
            jnp.arange(self.command_block_count) * self._block_steps,
            self.prediction_steps - 1,
        )
        return jnp.clip(self._normalized_from_commands(shifted[indices]), -1.0, 1.0)

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
        if np.any(previous_array < minimum) or np.any(previous_array > maximum):
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
            if np.any(applied_array < minimum) or np.any(applied_array > maximum):
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
                maximum_command_bound_violation=0.0,
                maximum_validity_utilization=math.inf,
                maximum_normalized_safety_violation=math.inf,
                maximum_normalized_model_uncertainty_standard_deviation=(
                    math.inf if self.belief.uncertainty_available else 0.0
                ),
                command_authority_fraction=0.0,
                uncertainty_aware_command_selection=(
                    self.belief.uncertainty_available
                ),
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
            ),
            used_fallback=True,
            message=message,
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

        started_at = time.perf_counter()
        input_error = self._input_error(
            state,
            reference,
            previous_command,
            applied_command,
            latent_state,
        )
        if input_error is not None:
            return self._failure_result(
                SolveStatus.INVALID_INPUT,
                input_error,
                previous_command,
                started_at,
            )
        if deadline_s is not None and (
            not np.isfinite(deadline_s) or deadline_s <= 0.0
        ):
            return self._failure_result(
                SolveStatus.DEADLINE_EXCEEDED,
                "deadline must be finite and positive",
                previous_command,
                started_at,
            )

        state = jnp.asarray(state)
        previous_command = jnp.asarray(previous_command)
        latent = (
            self._initial_latent_compiled(
                previous_command
                if applied_command is None
                else jnp.asarray(applied_command)
            )
            if latent_state is None
            else jnp.asarray(latent_state)
        )
        exogenous = self._exogenous_forecast(reference)
        cold_blocks = self._cold_blocks(previous_command)
        value, gradient = self._objective_and_gradient(
            cold_blocks,
            state,
            latent,
            reference.states,
            previous_command,
            exogenous,
        )
        current_value_float = float(np.asarray(value))
        current_value = value
        current_gradient = gradient
        blocks = cold_blocks
        used_warm_start = False
        if not np.isfinite(current_value_float) or not np.all(
            np.isfinite(np.asarray(current_gradient))
        ):
            return self._failure_result(
                SolveStatus.NONFINITE_OBJECTIVE,
                "cold-start objective or gradient is non-finite",
                previous_command,
                started_at,
            )

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
                )
                warm_value_float = float(np.asarray(warm_value))
                if (
                    np.isfinite(warm_value_float)
                    and np.all(np.isfinite(np.asarray(warm_gradient)))
                    and warm_value_float <= current_value_float
                ):
                    blocks = warm_blocks
                    current_value = warm_value
                    current_value_float = warm_value_float
                    current_gradient = warm_gradient
                    used_warm_start = True

        initial_objective = current_value_float
        if deadline_s is not None and time.perf_counter() - started_at >= deadline_s:
            return self._failure_result(
                SolveStatus.DEADLINE_EXCEEDED,
                "solver deadline expired before optimization",
                previous_command,
                started_at,
                initial_objective=initial_objective,
                warm_start_used=used_warm_start,
            )
        (
            blocks,
            current_value,
            current_gradient,
            iteration_array,
            converged_array,
            line_search_failed_array,
            finite_array,
        ) = self._optimize_compiled(
            blocks,
            current_value,
            current_gradient,
            state,
            latent,
            reference.states,
            previous_command,
            exogenous,
        )
        current_value_float = float(np.asarray(current_value))
        iteration = int(np.asarray(iteration_array))
        converged = bool(np.asarray(converged_array))
        line_search_failed = bool(np.asarray(line_search_failed_array))
        finite_optimization = bool(np.asarray(finite_array))
        gradient_norm = float(np.max(np.abs(np.asarray(current_gradient))))
        if deadline_s is not None and time.perf_counter() - started_at >= deadline_s:
            return self._failure_result(
                SolveStatus.DEADLINE_EXCEEDED,
                "solver deadline expired during optimization",
                previous_command,
                started_at,
                initial_objective=initial_objective,
                iterations=iteration,
                warm_start_used=used_warm_start,
            )
        if not finite_optimization:
            return self._failure_result(
                SolveStatus.NONFINITE_OBJECTIVE,
                "objective or gradient became non-finite during optimization",
                previous_command,
                started_at,
                initial_objective=initial_objective,
                iterations=iteration,
                warm_start_used=used_warm_start,
            )
        if line_search_failed:
            return self._failure_result(
                SolveStatus.LINE_SEARCH_FAILED,
                "bounded line search could not find a finite descent step",
                previous_command,
                started_at,
                initial_objective=initial_objective,
                iterations=iteration,
                warm_start_used=used_warm_start,
            )

        states, latent_states, commands = self._rollout_compiled(
            blocks, state, latent, exogenous
        )
        states_np = np.asarray(states)
        latent_np = np.asarray(latent_states)
        commands_np = np.asarray(commands)
        if not (
            np.all(np.isfinite(states_np))
            and np.all(np.isfinite(latent_np))
            and np.all(np.isfinite(commands_np))
            and np.isfinite(current_value_float)
        ):
            return self._failure_result(
                SolveStatus.NONFINITE_OBJECTIVE,
                "optimized prediction is non-finite",
                previous_command,
                started_at,
                initial_objective=initial_objective,
                iterations=iteration,
                warm_start_used=used_warm_start,
            )

        maximum_model_uncertainty = float(
            np.asarray(
                self._uncertainty_compiled(
                    state,
                    latent,
                    commands,
                    exogenous,
                )
            )
        )
        if not np.isfinite(maximum_model_uncertainty):
            return self._failure_result(
                SolveStatus.NONFINITE_OBJECTIVE,
                "model-uncertainty forecast is non-finite",
                previous_command,
                started_at,
                initial_objective=initial_objective,
                iterations=iteration,
                warm_start_used=used_warm_start,
            )
        command_authority = (
            min(1.0, 1.0 / maximum_model_uncertainty)
            if self.belief.uncertainty_available
            and maximum_model_uncertainty > 0.0
            else 1.0
        )
        if command_authority < 1.0:
            blocks = jnp.clip(
                cold_blocks + command_authority * (blocks - cold_blocks),
                -1.0,
                1.0,
            )
            current_value, current_gradient = self._objective_and_gradient(
                blocks,
                state,
                latent,
                reference.states,
                previous_command,
                exogenous,
            )
            current_value_float = float(np.asarray(current_value))
            gradient_norm = float(np.max(np.abs(np.asarray(current_gradient))))
            states, latent_states, commands = self._rollout_compiled(
                blocks,
                state,
                latent,
                exogenous,
            )
            states_np = np.asarray(states)
            latent_np = np.asarray(latent_states)
            commands_np = np.asarray(commands)
            maximum_model_uncertainty = float(
                np.asarray(
                    self._uncertainty_compiled(
                        state,
                        latent,
                        commands,
                        exogenous,
                    )
                )
            )
            converged = False
        if not (
            np.all(np.isfinite(states_np))
            and np.all(np.isfinite(latent_np))
            and np.all(np.isfinite(commands_np))
            and np.isfinite(current_value_float)
            and np.isfinite(maximum_model_uncertainty)
        ):
            return self._failure_result(
                SolveStatus.NONFINITE_OBJECTIVE,
                "uncertainty-bounded prediction is non-finite",
                previous_command,
                started_at,
                initial_objective=initial_objective,
                iterations=iteration,
                warm_start_used=used_warm_start,
            )

        minimum = np.asarray(self.model.command_minimum)
        maximum = np.asarray(self.model.command_maximum)
        bound_violation = float(
            max(
                np.max(minimum - commands_np),
                np.max(commands_np - maximum),
                0.0,
            )
        )
        maximum_validity = float(np.asarray(self._validity_compiled(states, exogenous)))
        maximum_safety = float(np.asarray(self._safety_compiled(states)))
        if deadline_s is not None and time.perf_counter() - started_at >= deadline_s:
            return self._failure_result(
                SolveStatus.DEADLINE_EXCEEDED,
                "solver deadline expired during prediction diagnostics",
                previous_command,
                started_at,
                initial_objective=initial_objective,
                iterations=iteration,
                warm_start_used=used_warm_start,
            )
        certified = self.model.runtime_spec.certified_prediction_horizon_s
        status = SolveStatus.CONVERGED if converged else SolveStatus.ITERATION_LIMIT
        return NMPCResult(
            status=status,
            command=commands[0],
            predicted_states=states,
            predicted_latent_states=latent_states,
            predicted_commands=commands,
            warm_start=NMPCWarmStart(commands),
            diagnostics=NMPCDiagnostics(
                iterations=iteration,
                solve_time_s=time.perf_counter() - started_at,
                initial_objective=initial_objective,
                final_objective=current_value_float,
                final_gradient_inf_norm=gradient_norm,
                maximum_command_bound_violation=bound_violation,
                maximum_validity_utilization=maximum_validity,
                maximum_normalized_safety_violation=maximum_safety,
                maximum_normalized_model_uncertainty_standard_deviation=(
                    maximum_model_uncertainty
                ),
                command_authority_fraction=command_authority,
                uncertainty_aware_command_selection=(
                    self.belief.uncertainty_available
                ),
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
                warm_start_used=used_warm_start,
                prediction_horizon_s=self.prediction_horizon_s,
                prediction_horizon_certified=(
                    certified is not None
                    and self.prediction_horizon_s <= certified + 1e-12
                ),
            ),
            used_fallback=False,
            message=(
                "finite plan returned with belief-bounded command authority"
                if command_authority < 1.0
                else "first-order convergence criterion satisfied"
                if converged
                else "finite best plan returned at the maintained iteration limit"
            ),
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
        belief = (
            model.compile_for_nmpc()
            if isinstance(model, DynamicsBelief)
            else model
            if isinstance(model, RuntimeDynamicsBelief)
            else RuntimeDynamicsBelief.from_nominal(model)
        )
        self.belief = belief
        self.model = belief.nominal
        self.tolerances = (
            TrackingTolerances.for_platform(
                self.model.input_spec.vehicle.family
            )
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
