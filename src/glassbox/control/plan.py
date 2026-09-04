"""What a bounded solver plans over: the model contract and its data types.

Nothing here knows how a plan is obtained. :class:`PlanModel` is the whole
interface between a solver and whatever supplies its dynamics, so one solver
drives a fitted belief, a recursively identified belief, or any other model
that can roll a command plan out and price it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import NamedTuple, Protocol, runtime_checkable

import jax.numpy as jnp
import numpy as np
from jax import Array

MAXIMUM_COMMAND_BLOCKS = 10


def block_steps_for(horizon_steps: int, block_count: int) -> int:
    """Return the model steps each command block is held for."""

    return math.ceil(horizon_steps / block_count)


def blocks_cover_horizon(horizon_steps: int, block_count: int) -> bool:
    """Whether every block of this layout drives at least one model step."""

    return (block_count - 1) * block_steps_for(horizon_steps, block_count) < (
        horizon_steps
    )


def maintained_block_count(horizon_steps: int) -> int:
    """Choose the maintained command-block layout for one horizon length.

    The layout is the largest divisor of ``horizon_steps`` no greater than
    ``MAXIMUM_COMMAND_BLOCKS``, so every block is held for the same number of
    model steps and the expansion covers the horizon exactly. A horizon of ten
    steps or fewer therefore uses one block per step. When the horizon is a
    prime longer than that cap the only divisor available is one, which would
    throw away nearly all command authority; in that case the largest block
    count whose last block is merely truncated, and never empty, is used
    instead.
    """

    limit = min(MAXIMUM_COMMAND_BLOCKS, horizon_steps)
    for count in range(limit, 1, -1):
        if horizon_steps % count == 0:
            return count
    for count in range(limit, 0, -1):
        if blocks_cover_horizon(horizon_steps, count):
            return count
    raise AssertionError("a single block always covers the horizon")


def _positive_triplet(
    name: str, values: float | Sequence[float]
) -> tuple[float, float, float]:
    if np.isscalar(values):
        result = (float(values),) * 3
    else:
        result = tuple(float(value) for value in values)
    if len(result) != 3:
        raise ValueError(f"{name} must be a scalar or three values")
    if not np.all(np.isfinite(result)) or np.any(np.asarray(result) <= 0.0):
        raise ValueError(f"{name} must contain finite positive values")
    return result  # type: ignore[return-value]


@dataclass(frozen=True)
class TrackingTolerances:
    """Physical errors that should each contribute unit normalized error."""

    position_m: float | tuple[float, float, float]
    velocity_m_s: float | tuple[float, float, float]
    attitude_rad: float | tuple[float, float, float]
    angular_velocity_rad_s: float | tuple[float, float, float]

    def __post_init__(self) -> None:
        for name in (
            "position_m",
            "velocity_m_s",
            "attitude_rad",
            "angular_velocity_rad_s",
        ):
            object.__setattr__(self, name, _positive_triplet(name, getattr(self, name)))

    @classmethod
    def for_platform(cls, platform: str) -> TrackingTolerances:
        """Return the maintained physical defaults for a vehicle family.

        Both multirotor families share one set: the bootstrap parameterization
        describes the same vehicle as a fitted multirotor model and is tracked
        to the same physical errors.
        """

        if platform in {"multirotor", "multirotor_bootstrap"}:
            return cls(
                position_m=0.4,
                velocity_m_s=0.5,
                attitude_rad=0.15,
                angular_velocity_rad_s=0.5,
            )
        if platform == "fixedwing":
            return cls(
                position_m=(5.0, 5.0, 3.0),
                velocity_m_s=2.0,
                attitude_rad=0.2,
                angular_velocity_rad_s=0.5,
            )
        raise ValueError(f"no NMPC defaults for platform {platform!r}")

    @property
    def local_state_scale(self) -> Array:
        return jnp.asarray(
            self.position_m
            + self.velocity_m_s
            + self.attitude_rad
            + self.angular_velocity_rad_s
        )


def _optional_triplet(
    name: str, values: Sequence[float] | None
) -> tuple[float, float, float] | None:
    if values is None:
        return None
    result = tuple(float(value) for value in values)
    if len(result) != 3 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain three finite values")
    return result  # type: ignore[return-value]


@dataclass(frozen=True)
class SafetyEnvelope:
    """Optional, physically stated soft state limits for prediction."""

    minimum_position_m: tuple[float, float, float] | None = None
    maximum_position_m: tuple[float, float, float] | None = None
    maximum_speed_m_s: float | None = None
    maximum_angular_velocity_rad_s: float | None = None

    def __post_init__(self) -> None:
        minimum = _optional_triplet("minimum_position_m", self.minimum_position_m)
        maximum = _optional_triplet("maximum_position_m", self.maximum_position_m)
        if minimum is not None and maximum is not None:
            if np.any(np.asarray(minimum) >= np.asarray(maximum)):
                raise ValueError("minimum_position_m must be below maximum_position_m")
        for name in ("maximum_speed_m_s", "maximum_angular_velocity_rad_s"):
            value = getattr(self, name)
            if value is not None and (not np.isfinite(value) or value <= 0.0):
                raise ValueError(f"{name} must be finite and positive")
        object.__setattr__(self, "minimum_position_m", minimum)
        object.__setattr__(self, "maximum_position_m", maximum)


@dataclass(frozen=True)
class ReferenceTrajectory:
    """A rigid-body state reference and optional known exogenous forecast."""

    states: Array
    exogenous: Array | None = None

    def __post_init__(self) -> None:
        states = np.asarray(self.states)
        if states.ndim != 2 or states.shape[1] != 13:
            raise ValueError("reference states must have shape (time, 13)")
        if len(states) < 2 or not np.all(np.isfinite(states)):
            raise ValueError("reference states must be finite and nonempty")
        quaternion_norm = np.linalg.norm(states[:, 6:10], axis=1)
        if np.any(quaternion_norm < 1e-6):
            raise ValueError("reference quaternions must have nonzero norm")
        object.__setattr__(self, "states", jnp.asarray(states))

        if self.exogenous is not None:
            exogenous = np.asarray(self.exogenous)
            if exogenous.ndim != 2 or exogenous.shape[0] != len(states) - 1:
                raise ValueError(
                    "reference exogenous forecast must have one row per interval"
                )
            if not np.all(np.isfinite(exogenous)):
                raise ValueError("reference exogenous forecast must be finite")
            object.__setattr__(self, "exogenous", jnp.asarray(exogenous))

    @classmethod
    def hold(
        cls,
        state: Array,
        horizon_steps: int,
        *,
        exogenous: Array | None = None,
    ) -> ReferenceTrajectory:
        """Hold one desired state and optional exogenous vector over a horizon."""

        if horizon_steps < 1:
            raise ValueError("horizon_steps must be positive")
        state_array = np.asarray(state)
        if state_array.shape != (13,):
            raise ValueError("held reference state must have shape (13,)")
        states = np.repeat(state_array[None, :], horizon_steps + 1, axis=0)
        forecast = None
        if exogenous is not None:
            exogenous_array = np.asarray(exogenous)
            if exogenous_array.ndim != 1:
                raise ValueError("held exogenous input must be a vector")
            forecast = np.repeat(exogenous_array[None, :], horizon_steps, axis=0)
        return cls(
            jnp.asarray(states), None if forecast is None else jnp.asarray(forecast)
        )


class SolveStatus(StrEnum):
    """Controller outcome with explicit degraded and failure states.

    The first three members return a finite optimized plan and a usable
    command. ``CONVERGED`` is reserved for the first-order criterion, which
    tests the bound-projected gradient against the maintained tolerance.
    ``STALLED`` means the bounded line search stopped making progress while
    that criterion was still unmet, so the returned plan is the best finite
    iterate rather than a stationary point. The remaining members are
    failures: they set ``used_fallback`` and return a bounded hold.
    """

    CONVERGED = "converged"
    ITERATION_LIMIT = "iteration_limit"
    STALLED = "stalled"
    LINE_SEARCH_FAILED = "line_search_failed"
    INVALID_INPUT = "invalid_input"
    NONFINITE_OBJECTIVE = "nonfinite_objective"
    DEADLINE_EXCEEDED = "deadline_exceeded"


@dataclass(frozen=True)
class NMPCWarmStart:
    """Opaque receding-horizon seed returned by a previous solve."""

    commands: Array

    def __post_init__(self) -> None:
        commands = np.asarray(self.commands)
        if commands.ndim != 2 or not np.all(np.isfinite(commands)):
            raise ValueError("warm-start commands must be a finite matrix")
        object.__setattr__(self, "commands", jnp.asarray(commands))


@dataclass(frozen=True)
class NMPCDiagnostics:
    """Auditable numerical and constraint diagnostics from one solve.

    Every field is a measurement of the plan that was returned. The objective
    values and the bound-projected gradient norm describe the optimization,
    the three maxima describe the predicted horizon, and the last two describe
    the horizon the plan covers.
    """

    iterations: int
    solve_time_s: float
    initial_objective: float
    final_objective: float
    final_projected_gradient_inf_norm: float
    maximum_command_bound_violation: float
    maximum_validity_utilization: float
    maximum_normalized_safety_violation: float
    maximum_normalized_model_uncertainty_standard_deviation: float
    warm_start_used: bool
    prediction_horizon_s: float
    prediction_horizon_certified: bool


@dataclass(frozen=True)
class SolveResult:
    """Command, prediction, warm start, and explicit solver outcome."""

    status: SolveStatus
    command: Array
    predicted_states: Array
    predicted_latent_states: Array
    predicted_commands: Array
    warm_start: NMPCWarmStart | None
    diagnostics: NMPCDiagnostics
    used_fallback: bool
    message: str

    @property
    def command_usable(self) -> bool:
        """Whether the returned command came from a finite optimized plan.

        ``CONVERGED``, ``ITERATION_LIMIT``, and ``STALLED`` all return an
        optimized bounded plan and are usable. ``STALLED`` carries exactly the
        same usability as ``ITERATION_LIMIT``: it reports that optimization
        stopped early, not that the command is a fallback. Only the explicit
        failure statuses set ``used_fallback`` and make the command a hold.
        """

        return not self.used_fallback


@dataclass(frozen=True)
class SolverPolicy:
    """Horizon, command blocking, line search, and objective weights."""

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

        return block_steps_for(self.horizon_steps, self.block_count)

    def __post_init__(self) -> None:
        if self.horizon_steps < 1:
            raise ValueError("horizon_steps must be positive")
        if not 1 <= self.block_count <= self.horizon_steps:
            raise ValueError("block_count must be within the prediction horizon")
        if not blocks_cover_horizon(self.horizon_steps, self.block_count):
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


class Prediction(NamedTuple):
    """One rolled-out command plan and what the model believes about it.

    The exogenous forecast the plan was rolled out under travels with it,
    because pricing the plan and measuring it both read the same context and
    reading a different one would price a different plan.
    """

    mean_states: Array
    tangent_covariance: Array
    commands: Array
    latent_states: Array
    exogenous: Array


class PlanMeasurements(NamedTuple):
    """Constraint and uncertainty margins of one predicted horizon."""

    maximum_validity_utilization: Array
    maximum_normalized_safety_violation: Array
    maximum_normalized_uncertainty: Array


class PlanValues(NamedTuple):
    """Everything a compiled kernel reads as an argument, not as a constant.

    A kernel is compiled once per static signature and then serves every model
    that shares it, so every number a belief holds travels here: the model
    parameters the horizon is rolled out through, the factor of the resolved
    parameter covariance the plan is charged spread for, and the
    forecast-error covariance at each predicted stage. A belief that absorbs
    telemetry every control interval moves all three every interval and still
    reuses the code compiled for the belief it came from.

    ``covariance_factor`` is ``None`` for a belief that resolves no parameter
    direction. That is a different traced structure rather than a different
    set of values, because such a belief really is priced by the point
    objective, and the signature records it as such.
    """

    parameters: object
    covariance_factor: Array | None
    forecast_error_covariance: Array


@runtime_checkable
class PlanModel(Protocol):
    """Everything a bounded shooting solver needs from a model.

    A plan is a sequence of normalized command blocks in ``[-1, 1]``. The model
    expands them, rolls them out, and prices them; the solver moves them and
    projects them back into the box. Everything the model believes numerically
    travels as :class:`PlanValues` rather than as part of the model, so one
    compiled kernel serves every model that shares this model's static
    signature.
    """

    horizon_steps: int
    block_count: int
    sample_period_s: float
    command_size: int
    exogenous_size: int
    latent_size: int
    certified_horizon_s: float | None
    uncertainty_available: bool
    command_minimum: Array
    command_maximum: Array
    values: PlanValues
    compile_signature: str

    def initial_latent(self, command_history: Array, values: PlanValues) -> Array:
        """Infer the actuator state a horizon starting now would begin from."""

    def rollout(
        self,
        blocks: Array,
        initial_state: Array,
        initial_latent: Array,
        exogenous: Array,
        values: PlanValues,
    ) -> Prediction:
        """Predict the horizon this plan drives, with its tangent covariance."""

    def stage_cost(
        self,
        prediction: Prediction,
        reference_states: Array,
        previous_command: Array,
        policy: SolverPolicy,
    ) -> Array:
        """Price one prediction against a reference under one policy."""

    def measure(self, prediction: Prediction) -> PlanMeasurements:
        """Measure the margins the result reports for one finished plan."""
