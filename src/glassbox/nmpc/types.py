"""Public data contracts for nonlinear model-predictive control."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

import jax.numpy as jnp
import numpy as np
from jax import Array


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
        """Return the maintained physical defaults for a vehicle family."""

        if platform == "multirotor":
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
    """Controller outcome with explicit degraded and failure states."""

    CONVERGED = "converged"
    ITERATION_LIMIT = "iteration_limit"
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
    """Auditable numerical and constraint diagnostics from one solve."""

    iterations: int
    solve_time_s: float
    initial_objective: float
    final_objective: float
    final_gradient_inf_norm: float
    maximum_command_bound_violation: float
    maximum_validity_utilization: float
    maximum_normalized_safety_violation: float
    maximum_normalized_model_uncertainty_standard_deviation: float
    model_uncertainty_available: bool
    prediction_error_model_available: bool
    prediction_error_model_current: bool
    prediction_error_horizon_supported: bool
    parameter_uncertainty_available: bool
    warm_start_used: bool
    prediction_horizon_s: float
    prediction_horizon_certified: bool


@dataclass(frozen=True)
class NMPCResult:
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
        """Whether the returned command came from a finite optimized plan."""

        return not self.used_fallback
