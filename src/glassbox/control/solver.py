"""Warm-started bounded direct shooting over any :class:`PlanModel`.

Nothing here knows what a belief is. The solver moves normalized command
blocks inside their box, asks the model to roll them out and price them, and
returns an auditable result with a bounded hold on every failure.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from glassbox.control.plan import (
    NMPCDiagnostics,
    NMPCWarmStart,
    PlanMeasurements,
    PlanModel,
    Prediction,
    ReferenceTrajectory,
    SolveResult,
    SolverPolicy,
    SolveStatus,
)

_COMMAND_BOUND_RELATIVE_TOLERANCE = 1e-6

# iteration, blocks, value, gradient, step size, converged, stalled,
# line-search failure, and whether any outer iteration was accepted.
_OuterCarry = tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array]


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
class _PlanEvaluation:
    """One command plan with its objective, gradient norm, and prediction."""

    blocks: Array
    value: Array
    gradient: Array
    value_float: float
    projected_gradient_inf_norm: float
    measurements: PlanMeasurements
    maximum_normalized_uncertainty: float
    prediction: Prediction
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
            and np.isfinite(self.maximum_normalized_uncertainty)
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


@dataclass(frozen=True)
class _Kernels:
    """The compiled operations one solver configuration needs."""

    objective_and_gradient: object
    optimize: object
    rollout: object
    measure: object
    initial_latent: object


_KERNEL_CACHE: dict[tuple[str, SolverPolicy], _Kernels] = {}


def compiled_kernel_count() -> int:
    """How many distinct solver configurations have compiled kernels."""

    return len(_KERNEL_CACHE)


def _objective(
    model: PlanModel,
    policy: SolverPolicy,
    blocks: Array,
    initial_state: Array,
    initial_latent: Array,
    reference_states: Array,
    previous_command: Array,
    exogenous: Array,
    parameters: object,
) -> Array:
    prediction = model.rollout(
        blocks,
        initial_state,
        initial_latent,
        exogenous,
        parameters,
    )
    return model.stage_cost(prediction, reference_states, previous_command, policy)


def _optimize_step(
    policy: SolverPolicy,
    objective_gradient: object,
    initial_blocks: Array,
    initial_value: Array,
    initial_gradient: Array,
    initial_state: Array,
    initial_latent: Array,
    reference_states: Array,
    previous_command: Array,
    exogenous: Array,
    parameters: object,
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
            (iteration < policy.maximum_iterations)
            & ~converged
            & ~stalled
            & ~line_search_failed
            & finite(value, gradient)
        )

    def outer_step(carry: _OuterCarry) -> _OuterCarry:
        iteration, blocks, value, gradient, step_size, _, _, _, progressed = carry
        gradient_norm = _projected_gradient_norm(blocks, gradient)
        gradient_converged = gradient_norm <= policy.gradient_tolerance

        def continue_line_search(
            line_carry: tuple[Array, Array, Array, Array, Array, Array],
        ) -> Array:
            line_iteration, accepted, _, _, _, _ = line_carry
            return (line_iteration < policy.line_search_steps) & ~accepted

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
            candidate_value, candidate_gradient = objective_gradient(
                candidate,
                initial_state,
                initial_latent,
                reference_states,
                previous_command,
                exogenous,
                parameters,
            )
            projected_decrease = jnp.sum(gradient * (blocks - candidate))
            candidate_accepted = finite(candidate_value, candidate_gradient) & (
                candidate_value
                <= value - policy.armijo_fraction * jnp.maximum(projected_decrease, 0.0)
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
        relative_improvement = (value - next_value) / jnp.maximum(jnp.abs(value), 1.0)
        improvement_stalled = (
            accepted
            & ~gradient_converged
            & (relative_improvement <= policy.relative_improvement_tolerance)
        )
        return (
            iteration + 1,
            next_blocks,
            next_value,
            next_gradient,
            jnp.minimum(policy.initial_step_size, 2.0 * accepted_step_size),
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
        jnp.asarray(policy.initial_step_size),
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


def _build_kernels(model: PlanModel, policy: SolverPolicy) -> _Kernels:
    def objective(
        blocks, initial_state, initial_latent, reference_states, previous, exogenous, p
    ):
        return _objective(
            model,
            policy,
            blocks,
            initial_state,
            initial_latent,
            reference_states,
            previous,
            exogenous,
            p,
        )

    objective_gradient = jax.value_and_grad(objective)

    def optimize(
        blocks,
        value,
        gradient,
        initial_state,
        initial_latent,
        reference_states,
        previous,
        exogenous,
        p,
    ):
        return _optimize_step(
            policy,
            objective_gradient,
            blocks,
            value,
            gradient,
            initial_state,
            initial_latent,
            reference_states,
            previous,
            exogenous,
            p,
        )

    return _Kernels(
        objective_and_gradient=jax.jit(objective_gradient),
        optimize=jax.jit(optimize),
        rollout=jax.jit(model.rollout),
        measure=jax.jit(model.measure),
        initial_latent=jax.jit(model.initial_latent),
    )


def solver_kernels(model: PlanModel, policy: SolverPolicy) -> _Kernels:
    """Return the compiled kernels for one solver configuration.

    Compiling a controller costs seconds and every kernel is a pure function of
    the model's static signature and the policy, so two solvers built from the
    same configuration share one set. The fitted parameters are an argument to
    every kernel rather than part of the signature, which is what lets a
    re-fitted or re-adapted belief reuse the compiled code of the belief it
    came from.
    """

    key = (model.compile_signature, policy)
    cached = _KERNEL_CACHE.get(key)
    if cached is None:
        cached = _build_kernels(model, policy)
        _KERNEL_CACHE[key] = cached
    return cached


class BoundedShootingSolver:
    """Warm-started bounded direct shooting over one plan model."""

    def __init__(self, model: PlanModel, policy: SolverPolicy) -> None:
        if (
            policy.horizon_steps != model.horizon_steps
            or policy.block_count != model.block_count
        ):
            raise ValueError("solver policy and plan model disagree on the horizon")
        self.model = model
        self.policy = policy
        self._kernels = solver_kernels(model, policy)

    @property
    def prediction_steps(self) -> int:
        return self.model.horizon_steps

    @property
    def prediction_horizon_s(self) -> float:
        return self.prediction_steps * self.model.sample_period_s

    def hold_reference(
        self, state: Array, *, exogenous: Array | None = None
    ) -> ReferenceTrajectory:
        """Build the common regulation reference for this solver."""

        return ReferenceTrajectory.hold(
            state, self.prediction_steps, exogenous=exogenous
        )

    def _normalized_from_commands(self, commands: Array) -> Array:
        minimum = self.model.command_minimum
        command_range = self.model.command_maximum - minimum
        return 2.0 * (commands - minimum) / command_range - 1.0

    def _exogenous_forecast(self, reference: ReferenceTrajectory) -> Array:
        if reference.exogenous is None:
            return jnp.zeros((self.prediction_steps, self.model.exogenous_size))
        return reference.exogenous

    def _cold_blocks(self, previous_command: Array) -> Array:
        normalized = self._normalized_from_commands(previous_command)
        return jnp.repeat(normalized[None, :], self.model.block_count, axis=0)

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
            jnp.arange(self.model.block_count) * self.policy.block_steps,
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

    def _certified(self) -> bool:
        certified = self.model.certified_horizon_s
        return certified is not None and self.prediction_horizon_s <= certified + 1e-12

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
    ) -> SolveResult:
        fallback = self._fallback_command(previous_command)
        commands = jnp.repeat(fallback[None, :], self.prediction_steps, axis=0)
        return SolveResult(
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
                    math.inf if self.model.uncertainty_available else 0.0
                ),
                warm_start_used=warm_start_used,
                prediction_horizon_s=self.prediction_horizon_s,
                prediction_horizon_certified=self._certified(),
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
        return self._kernels.initial_latent(
            previous_command
            if applied_command is None
            else jnp.clip(
                jnp.asarray(applied_command),
                self.model.command_minimum,
                self.model.command_maximum,
            ),
            self.model.parameters,
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

        value, gradient = self._kernels.objective_and_gradient(
            cold_blocks,
            state,
            latent,
            reference.states,
            previous_command,
            exogenous,
            self.model.parameters,
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
                warm_value, warm_gradient = self._kernels.objective_and_gradient(
                    warm_blocks,
                    state,
                    latent,
                    reference.states,
                    previous_command,
                    exogenous,
                    self.model.parameters,
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
        ) = self._kernels.optimize(
            blocks,
            value,
            gradient,
            state,
            latent,
            reference.states,
            previous_command,
            exogenous,
            self.model.parameters,
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

        prediction = self._kernels.rollout(
            blocks,
            state,
            latent,
            exogenous,
            self.model.parameters,
        )
        measurements = self._kernels.measure(prediction)
        return _PlanEvaluation(
            blocks=blocks,
            value=value,
            gradient=gradient,
            value_float=float(np.asarray(value)),
            projected_gradient_inf_norm=float(
                np.asarray(_projected_gradient_norm(blocks, gradient))
            ),
            measurements=measurements,
            maximum_normalized_uncertainty=float(
                np.asarray(measurements.maximum_normalized_uncertainty)
            ),
            prediction=prediction,
            states_np=np.asarray(prediction.mean_states),
            latent_np=np.asarray(prediction.latent_states),
            commands_np=np.asarray(prediction.commands),
        )

    def _maximum_command_bound_violation(self, plan: _PlanEvaluation) -> float:
        minimum = np.asarray(self.model.command_minimum)
        maximum = np.asarray(self.model.command_maximum)
        return float(
            max(
                np.max(minimum - plan.commands_np),
                np.max(plan.commands_np - maximum),
                0.0,
            )
        )

    def _solved_result(
        self,
        plan: _PlanEvaluation,
        outcome: _OptimizerOutcome,
        progress: _SolveProgress,
    ) -> SolveResult:
        """Assemble the auditable result for one finite, bounded solve."""

        status = (
            SolveStatus.CONVERGED
            if outcome.converged
            else SolveStatus.STALLED
            if outcome.stalled
            else SolveStatus.ITERATION_LIMIT
        )
        return SolveResult(
            status=status,
            command=plan.prediction.commands[0],
            predicted_states=plan.prediction.mean_states,
            predicted_latent_states=plan.prediction.latent_states,
            predicted_commands=plan.prediction.commands,
            warm_start=NMPCWarmStart(plan.prediction.commands),
            diagnostics=NMPCDiagnostics(
                iterations=outcome.iterations,
                solve_time_s=time.perf_counter() - progress.started_at,
                initial_objective=progress.initial_objective,
                final_objective=plan.value_float,
                final_projected_gradient_inf_norm=(plan.projected_gradient_inf_norm),
                maximum_command_bound_violation=(
                    self._maximum_command_bound_violation(plan)
                ),
                maximum_validity_utilization=float(
                    np.asarray(plan.measurements.maximum_validity_utilization)
                ),
                maximum_normalized_safety_violation=float(
                    np.asarray(plan.measurements.maximum_normalized_safety_violation)
                ),
                maximum_normalized_model_uncertainty_standard_deviation=(
                    plan.maximum_normalized_uncertainty
                ),
                warm_start_used=progress.warm_start_used,
                prediction_horizon_s=self.prediction_horizon_s,
                prediction_horizon_certified=self._certified(),
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
    ) -> SolveResult:
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
            if not plan.prediction_finite:
                raise _SolveAbort(
                    SolveStatus.NONFINITE_OBJECTIVE,
                    "optimized prediction is non-finite",
                )
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
        return self._solved_result(plan, outcome, progress)
