"""Fast, support-aware multirotor identification without an airframe prior.

The bootstrap path deliberately estimates only the input/output relationships
needed to establish bounded collective and angular-rate authority.  It does not
construct a complete flight model and it never fills unexcited motor directions
from a nominal mixer or fleet prior.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from glassbox.control._common import (
    finite_tuple,
    immutable_array,
    thrust_cascade,
)
from glassbox.core.dynamics import GRAVITY_M_S2
from glassbox.core.geometry import (
    quaternion_to_rotation,
    quaternion_to_rotation_batch,
    world_up_body,
)


@dataclass(frozen=True)
class BootstrapIdentificationConfig:
    """Fixed evidence and acceptance contract for one precompiled fit shape."""

    interval_count: int = 32
    validation_fraction: float = 0.25
    command_minimum: float | tuple[float, float, float, float] = 0.0
    command_maximum: float | tuple[float, float, float, float] = 1.0
    command_rank_relative_tolerance: float = 0.02
    minimum_normalized_command_rms: float = 0.015
    nuisance_rank_relative_tolerance: float = 0.002
    output_rank_relative_tolerance: float = 0.03
    minimum_validation_improvement: float = 0.05
    minimum_collective_support_fraction: float = 0.95
    maximum_sample_period_deviation_fraction: float = 0.05

    def __post_init__(self) -> None:
        minimum = finite_tuple("command_minimum", self.command_minimum, 4)
        maximum = finite_tuple("command_maximum", self.command_maximum, 4)
        if np.any(np.asarray(minimum) >= np.asarray(maximum)):
            raise ValueError("command_minimum must be below command_maximum")
        if self.interval_count < 16:
            raise ValueError("bootstrap identification needs at least 16 intervals")
        if not 0.15 <= self.validation_fraction <= 0.5:
            raise ValueError("validation_fraction must lie between 0.15 and 0.5")
        unit_interval_fields = (
            "command_rank_relative_tolerance",
            "minimum_normalized_command_rms",
            "nuisance_rank_relative_tolerance",
            "output_rank_relative_tolerance",
            "minimum_validation_improvement",
            "minimum_collective_support_fraction",
            "maximum_sample_period_deviation_fraction",
        )
        for name in unit_interval_fields:
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 < value < 1.0:
                raise ValueError(f"{name} must lie strictly between zero and one")
        if self.training_interval_count < 8 or self.validation_interval_count < 4:
            raise ValueError(
                "bootstrap split needs eight train and four validation intervals"
            )
        object.__setattr__(self, "command_minimum", minimum)
        object.__setattr__(self, "command_maximum", maximum)

    @property
    def validation_interval_count(self) -> int:
        return max(4, round(self.interval_count * self.validation_fraction))

    @property
    def training_interval_count(self) -> int:
        return self.interval_count - self.validation_interval_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "interval_count": self.interval_count,
            "training_interval_count": self.training_interval_count,
            "validation_interval_count": self.validation_interval_count,
            "validation_fraction": self.validation_fraction,
            "command_minimum": list(self.command_minimum),
            "command_maximum": list(self.command_maximum),
            "command_rank_relative_tolerance": self.command_rank_relative_tolerance,
            "minimum_normalized_command_rms": self.minimum_normalized_command_rms,
            "nuisance_rank_relative_tolerance": (self.nuisance_rank_relative_tolerance),
            "output_rank_relative_tolerance": self.output_rank_relative_tolerance,
            "minimum_validation_improvement": self.minimum_validation_improvement,
            "minimum_collective_support_fraction": (
                self.minimum_collective_support_fraction
            ),
            "maximum_sample_period_deviation_fraction": (
                self.maximum_sample_period_deviation_fraction
            ),
        }


class BootstrapModelNotReadyError(RuntimeError):
    """Raised when unsupported bootstrap evidence is used for control."""


@dataclass(frozen=True)
class BootstrapExcitationConfig:
    """Bounded follow-up excitation policy for a provisional bootstrap fit."""

    interval_count: int = 8
    amplitude_fraction_of_command_span: float = 0.08

    def __post_init__(self) -> None:
        if self.interval_count < 4 or self.interval_count % 2:
            raise ValueError("excitation interval_count must be even and at least four")
        if not math.isfinite(self.amplitude_fraction_of_command_span) or not (
            0.0 < self.amplitude_fraction_of_command_span <= 0.25
        ):
            raise ValueError("amplitude_fraction_of_command_span must lie in (0, 0.25]")


@dataclass(frozen=True)
class BootstrapExcitationPlan:
    """Symmetric motor commands selected from unresolved fit evidence."""

    commands: np.ndarray
    normalized_direction: np.ndarray
    center_command: np.ndarray
    target_angular_axis: int | None
    selection_reason: str

    def __post_init__(self) -> None:
        commands = np.asarray(self.commands, dtype=np.float64).copy()
        if (
            commands.ndim != 2
            or commands.shape[1] != 4
            or not np.all(np.isfinite(commands))
        ):
            raise ValueError("excitation commands must have shape (intervals, 4)")
        commands.flags.writeable = False
        object.__setattr__(self, "commands", commands)
        object.__setattr__(
            self,
            "normalized_direction",
            immutable_array(
                self.normalized_direction,
                (4,),
                "normalized_direction",
            ),
        )
        object.__setattr__(
            self,
            "center_command",
            immutable_array(self.center_command, (4,), "center_command"),
        )
        if self.target_angular_axis is not None and self.target_angular_axis not in (
            0,
            1,
            2,
        ):
            raise ValueError("target_angular_axis must be 0, 1, 2, or None")
        if not self.selection_reason:
            raise ValueError("selection_reason cannot be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "commands": self.commands.tolist(),
            "normalized_direction": self.normalized_direction.tolist(),
            "center_command": self.center_command.tolist(),
            "target_angular_axis": self.target_angular_axis,
            "selection_reason": self.selection_reason,
        }


@dataclass(frozen=True)
class BootstrapArrestCommand:
    """Bounded motor command with its locally predicted angular effect."""

    command: np.ndarray
    desired_angular_acceleration_rad_s2: np.ndarray
    predicted_control_angular_acceleration_rad_s2: np.ndarray
    residual_norm_rad_s2: float
    saturated: bool
    slew_limited: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "command", immutable_array(self.command, (4,), "command")
        )
        object.__setattr__(
            self,
            "desired_angular_acceleration_rad_s2",
            immutable_array(
                self.desired_angular_acceleration_rad_s2,
                (3,),
                "desired angular acceleration",
            ),
        )
        object.__setattr__(
            self,
            "predicted_control_angular_acceleration_rad_s2",
            immutable_array(
                self.predicted_control_angular_acceleration_rad_s2,
                (3,),
                "predicted control angular acceleration",
            ),
        )
        if not math.isfinite(self.residual_norm_rad_s2):
            raise ValueError("residual norm must be finite")

    def to_dict(self) -> dict[str, Any]:
        return {
            "command": self.command.tolist(),
            "desired_angular_acceleration_rad_s2": (
                self.desired_angular_acceleration_rad_s2.tolist()
            ),
            "predicted_control_angular_acceleration_rad_s2": (
                self.predicted_control_angular_acceleration_rad_s2.tolist()
            ),
            "residual_norm_rad_s2": self.residual_norm_rad_s2,
            "saturated": self.saturated,
            "slew_limited": self.slew_limited,
        }


@dataclass(frozen=True)
class BootstrapVelocityArrestCommand:
    """Velocity-vector objective and its bounded attitude/motor allocation."""

    motor: BootstrapArrestCommand
    desired_world_acceleration_m_s2: np.ndarray
    desired_thrust_direction_world: np.ndarray
    collective_reference_command: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.motor, BootstrapArrestCommand):
            raise TypeError("motor must be a BootstrapArrestCommand")
        object.__setattr__(
            self,
            "desired_world_acceleration_m_s2",
            immutable_array(
                self.desired_world_acceleration_m_s2,
                (3,),
                "desired_world_acceleration_m_s2",
            ),
        )
        object.__setattr__(
            self,
            "desired_thrust_direction_world",
            immutable_array(
                self.desired_thrust_direction_world,
                (3,),
                "desired_thrust_direction_world",
            ),
        )
        object.__setattr__(
            self,
            "collective_reference_command",
            immutable_array(
                self.collective_reference_command,
                (4,),
                "collective_reference_command",
            ),
        )

    @property
    def command(self) -> np.ndarray:
        return self.motor.command

    def to_dict(self) -> dict[str, Any]:
        return {
            "motor": self.motor.to_dict(),
            "desired_world_acceleration_m_s2": (
                self.desired_world_acceleration_m_s2.tolist()
            ),
            "desired_thrust_direction_world": (
                self.desired_thrust_direction_world.tolist()
            ),
            "collective_reference_command": (
                self.collective_reference_command.tolist()
            ),
        }


@dataclass(frozen=True)
class BootstrapIdentificationResult:
    """Auditable local input/output model and its evidence support."""

    collective_acceleration_per_command: np.ndarray
    collective_velocity_coefficient: np.ndarray
    collective_intercept_m_s2: float
    angular_acceleration_per_command: np.ndarray
    angular_rate_coefficient: np.ndarray
    angular_rate_product_coefficient: np.ndarray
    angular_intercept_rad_s2: np.ndarray
    normalized_command_support_projector: np.ndarray
    normalized_command_singular_values: np.ndarray
    collective_command_evidence_rank: int
    command_evidence_rank: int
    angular_effect_rank: int
    collective_nuisance_rank: int
    angular_nuisance_rank: int
    collective_support_fraction: float
    hover_command: np.ndarray | None
    collective_validation_rmse_m_s2: float
    collective_baseline_validation_rmse_m_s2: float
    collective_validation_improvement: float
    angular_validation_rmse_rad_s2: np.ndarray
    angular_baseline_validation_rmse_rad_s2: np.ndarray
    angular_validation_improvement: np.ndarray
    ready_for_hover: bool
    ready_for_rate_arrest: bool
    wall_time_s: float
    sample_period_s: float
    evidence_duration_s: float
    command_center: np.ndarray
    command_minimum: np.ndarray
    command_maximum: np.ndarray

    def __post_init__(self) -> None:
        array_fields = {
            "collective_acceleration_per_command": (4,),
            "collective_velocity_coefficient": (3,),
            "angular_acceleration_per_command": (3, 4),
            "angular_rate_coefficient": (3, 3),
            "angular_rate_product_coefficient": (3, 3),
            "angular_intercept_rad_s2": (3,),
            "normalized_command_support_projector": (4, 4),
            "normalized_command_singular_values": (4,),
            "angular_validation_rmse_rad_s2": (3,),
            "angular_baseline_validation_rmse_rad_s2": (3,),
            "angular_validation_improvement": (3,),
            "command_center": (4,),
            "command_minimum": (4,),
            "command_maximum": (4,),
        }
        for name, shape in array_fields.items():
            object.__setattr__(
                self,
                name,
                immutable_array(getattr(self, name), shape, name),
            )
        if self.hover_command is not None:
            object.__setattr__(
                self,
                "hover_command",
                immutable_array(self.hover_command, (4,), "hover_command"),
            )
        scalar_fields = (
            "collective_intercept_m_s2",
            "collective_support_fraction",
            "collective_validation_rmse_m_s2",
            "collective_baseline_validation_rmse_m_s2",
            "collective_validation_improvement",
            "wall_time_s",
            "sample_period_s",
            "evidence_duration_s",
        )
        if any(not math.isfinite(float(getattr(self, name))) for name in scalar_fields):
            raise ValueError("bootstrap result scalar fields must be finite")

    @property
    def ready(self) -> bool:
        return self.ready_for_hover and self.ready_for_rate_arrest

    def predict_angular_acceleration(
        self,
        command: Sequence[float],
        angular_velocity_rad_s: Sequence[float],
    ) -> np.ndarray:
        command_array = np.asarray(command, dtype=np.float64)
        angular_velocity = np.asarray(angular_velocity_rad_s, dtype=np.float64)
        if command_array.shape != (4,) or angular_velocity.shape != (3,):
            raise ValueError(
                "command and angular velocity must have shapes (4,) and (3,)"
            )
        return (
            self.angular_acceleration_per_command @ command_array
            + self.angular_rate_coefficient @ angular_velocity
            + self.angular_rate_product_coefficient
            @ np.asarray(
                (
                    angular_velocity[0] * angular_velocity[1],
                    angular_velocity[0] * angular_velocity[2],
                    angular_velocity[1] * angular_velocity[2],
                )
            )
            + self.angular_intercept_rad_s2
        )

    def rate_arrest_command(
        self,
        angular_velocity_rad_s: Sequence[float],
        *,
        angular_rate_gain: float | Sequence[float] = (4.0, 4.0, 2.0),
        desired_angular_acceleration_rad_s2: Sequence[float] | None = None,
        reference_command: Sequence[float] | None = None,
        previous_command: Sequence[float] | None = None,
        maximum_motor_step: float | None = None,
    ) -> BootstrapArrestCommand:
        """Allocate a directly explained angular-rate-arrest motor command."""

        if not self.ready_for_rate_arrest:
            raise BootstrapModelNotReadyError(
                "angular input/output evidence does not support three-axis arrest"
            )
        angular_velocity = np.asarray(angular_velocity_rad_s, dtype=np.float64)
        if angular_velocity.shape != (3,) or not np.all(np.isfinite(angular_velocity)):
            raise ValueError("angular_velocity_rad_s must contain three finite values")
        if desired_angular_acceleration_rad_s2 is None:
            gains = np.asarray(
                finite_tuple("angular_rate_gain", angular_rate_gain, 3),
                dtype=np.float64,
            )
            if np.any(gains <= 0.0):
                raise ValueError("angular_rate_gain must be positive")
            desired = -gains * angular_velocity
        else:
            desired = np.asarray(
                finite_tuple(
                    "desired_angular_acceleration_rad_s2",
                    desired_angular_acceleration_rad_s2,
                    3,
                ),
                dtype=np.float64,
            )
        if reference_command is None:
            reference = self.hover_command
            if reference is None:
                raise BootstrapModelNotReadyError(
                    "rate arrest needs an explicit collective reference until hover is identified"
                )
        else:
            reference = np.asarray(reference_command, dtype=np.float64)
            if reference.shape != (4,) or not np.all(np.isfinite(reference)):
                raise ValueError("reference_command must contain four finite values")

        # Allocate only the directly identified incremental control effect.
        # Nuisance rate terms improve evidence validation but are not extrapolated
        # into this first bounded controller.
        delta = (
            np.linalg.pinv(
                self.angular_acceleration_per_command,
                rcond=1e-6,
            )
            @ desired
        )
        unconstrained = np.asarray(reference) + delta
        bounded = np.clip(unconstrained, self.command_minimum, self.command_maximum)
        saturated = not np.allclose(bounded, unconstrained, atol=1e-12, rtol=0.0)
        slew_limited = False
        if maximum_motor_step is not None:
            if not math.isfinite(maximum_motor_step) or maximum_motor_step <= 0.0:
                raise ValueError("maximum_motor_step must be finite and positive")
            if previous_command is None:
                raise ValueError("previous_command is required with maximum_motor_step")
            previous = np.asarray(previous_command, dtype=np.float64)
            if previous.shape != (4,) or not np.all(np.isfinite(previous)):
                raise ValueError("previous_command must contain four finite values")
            limited = np.clip(
                bounded,
                previous - maximum_motor_step,
                previous + maximum_motor_step,
            )
            limited = np.clip(limited, self.command_minimum, self.command_maximum)
            slew_limited = not np.allclose(limited, bounded, atol=1e-12, rtol=0.0)
            bounded = limited
        predicted = self.angular_acceleration_per_command @ (
            bounded - np.asarray(reference)
        )
        return BootstrapArrestCommand(
            command=bounded,
            desired_angular_acceleration_rad_s2=desired,
            predicted_control_angular_acceleration_rad_s2=predicted,
            residual_norm_rad_s2=float(np.linalg.norm(predicted - desired)),
            saturated=saturated,
            slew_limited=slew_limited,
        )

    def attitude_rate_arrest_command(
        self,
        quaternion_wxyz: Sequence[float],
        angular_velocity_rad_s: Sequence[float],
        *,
        attitude_gain: float | Sequence[float] = (14.0, 14.0),
        angular_rate_gain: float | Sequence[float] = (6.0, 6.0, 3.0),
        reference_command: Sequence[float] | None = None,
        previous_command: Sequence[float] | None = None,
        maximum_motor_step: float | None = None,
    ) -> BootstrapArrestCommand:
        """Allocate level-attitude and body-rate arrest without a known mixer."""

        quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
        angular_velocity = np.asarray(angular_velocity_rad_s, dtype=np.float64)
        if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
            raise ValueError("quaternion_wxyz must contain four finite values")
        quaternion_norm = float(np.linalg.norm(quaternion))
        if quaternion_norm < 1e-6:
            raise ValueError("quaternion_wxyz must have nonzero norm")
        if angular_velocity.shape != (3,) or not np.all(np.isfinite(angular_velocity)):
            raise ValueError("angular_velocity_rad_s must contain three finite values")
        attitude_gains = np.asarray(
            finite_tuple("attitude_gain", attitude_gain, 2),
            dtype=np.float64,
        )
        rate_gains = np.asarray(
            finite_tuple("angular_rate_gain", angular_rate_gain, 3),
            dtype=np.float64,
        )
        if np.any(attitude_gains <= 0.0) or np.any(rate_gains <= 0.0):
            raise ValueError("attitude and angular-rate gains must be positive")
        tilt_error_body = np.cross(
            np.asarray((0.0, 0.0, 1.0)),
            world_up_body(quaternion / quaternion_norm),
        )
        desired = np.asarray(
            (
                attitude_gains[0] * tilt_error_body[0]
                - rate_gains[0] * angular_velocity[0],
                attitude_gains[1] * tilt_error_body[1]
                - rate_gains[1] * angular_velocity[1],
                -rate_gains[2] * angular_velocity[2],
            )
        )
        return self.rate_arrest_command(
            angular_velocity,
            desired_angular_acceleration_rad_s2=desired,
            reference_command=reference_command,
            previous_command=previous_command,
            maximum_motor_step=maximum_motor_step,
        )

    def velocity_attitude_rate_arrest_command(
        self,
        world_velocity_m_s: Sequence[float],
        quaternion_wxyz: Sequence[float],
        angular_velocity_rad_s: Sequence[float],
        *,
        velocity_gain: float | Sequence[float] = (1.5, 1.5, 1.5),
        maximum_world_acceleration_m_s2: float | Sequence[float] = (
            2.5,
            2.5,
            3.0,
        ),
        maximum_tilt_rad: float = 0.50,
        attitude_gain: float | Sequence[float] = (14.0, 14.0),
        angular_rate_gain: float | Sequence[float] = (6.0, 6.0, 3.0),
        previous_command: Sequence[float] | None = None,
        maximum_motor_step: float | None = None,
    ) -> BootstrapVelocityArrestCommand:
        """Arrest world velocity by vectoring identified collective authority."""

        if not self.ready:
            raise BootstrapModelNotReadyError(
                "velocity arrest requires supported hover and three-axis authority"
            )
        velocity = np.asarray(world_velocity_m_s, dtype=np.float64)
        quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
        angular_velocity = np.asarray(angular_velocity_rad_s, dtype=np.float64)
        if velocity.shape != (3,) or not np.all(np.isfinite(velocity)):
            raise ValueError("world_velocity_m_s must contain three finite values")
        if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
            raise ValueError("quaternion_wxyz must contain four finite values")
        quaternion_norm = float(np.linalg.norm(quaternion))
        if quaternion_norm < 1e-6:
            raise ValueError("quaternion_wxyz must have nonzero norm")
        if angular_velocity.shape != (3,) or not np.all(np.isfinite(angular_velocity)):
            raise ValueError("angular_velocity_rad_s must contain three finite values")
        velocity_gains = np.asarray(
            finite_tuple("velocity_gain", velocity_gain, 3),
            dtype=np.float64,
        )
        maximum_acceleration = np.asarray(
            finite_tuple(
                "maximum_world_acceleration_m_s2",
                maximum_world_acceleration_m_s2,
                3,
            ),
            dtype=np.float64,
        )
        attitude_gains = np.asarray(
            finite_tuple("attitude_gain", attitude_gain, 2),
            dtype=np.float64,
        )
        rate_gains = np.asarray(
            finite_tuple("angular_rate_gain", angular_rate_gain, 3),
            dtype=np.float64,
        )
        if (
            np.any(velocity_gains <= 0.0)
            or np.any(maximum_acceleration <= 0.0)
            or np.any(attitude_gains <= 0.0)
            or np.any(rate_gains <= 0.0)
        ):
            raise ValueError(
                "velocity, acceleration, attitude, and rate gains must be positive"
            )
        if not math.isfinite(maximum_tilt_rad) or not 0.0 < maximum_tilt_rad < 1.0:
            raise ValueError("maximum_tilt_rad must lie between zero and one radian")

        cascade = thrust_cascade(
            world_velocity_m_s=velocity,
            rotation=quaternion_to_rotation(quaternion),
            angular_velocity_rad_s=angular_velocity,
            velocity_gain=velocity_gains,
            maximum_world_acceleration_m_s2=maximum_acceleration,
            maximum_tilt_rad=maximum_tilt_rad,
            attitude_gain=attitude_gains,
            angular_rate_gain=rate_gains,
        )
        collective_effect = float(np.sum(self.collective_acceleration_per_command))
        if collective_effect <= 1e-8:
            raise BootstrapModelNotReadyError(
                "identified collective effect is not positive"
            )
        collective_scalar = (
            cascade.desired_force_magnitude_m_s2 - self.collective_intercept_m_s2
        ) / collective_effect
        collective_reference = np.clip(
            np.full(4, collective_scalar),
            self.command_minimum,
            self.command_maximum,
        )
        motor = self.rate_arrest_command(
            angular_velocity,
            desired_angular_acceleration_rad_s2=(
                cascade.desired_angular_acceleration_rad_s2
            ),
            reference_command=collective_reference,
            previous_command=previous_command,
            maximum_motor_step=maximum_motor_step,
        )
        return BootstrapVelocityArrestCommand(
            motor=motor,
            desired_world_acceleration_m_s2=cascade.desired_world_acceleration_m_s2,
            desired_thrust_direction_world=cascade.desired_thrust_direction_world,
            collective_reference_command=collective_reference,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": "rank_supported_multirotor_bootstrap_v1",
            "airframe_parameter_prior_used": False,
            "canonical_motor_mixer_assumed": False,
            "applied_motor_state_required": True,
            "collective_command_evidence_rank": (self.collective_command_evidence_rank),
            "command_evidence_rank": self.command_evidence_rank,
            "angular_effect_rank": self.angular_effect_rank,
            "collective_nuisance_rank": self.collective_nuisance_rank,
            "angular_nuisance_rank": self.angular_nuisance_rank,
            "collective_support_fraction": self.collective_support_fraction,
            "collective_acceleration_per_command": (
                self.collective_acceleration_per_command.tolist()
            ),
            "collective_velocity_coefficient": (
                self.collective_velocity_coefficient.tolist()
            ),
            "collective_intercept_m_s2": self.collective_intercept_m_s2,
            "angular_acceleration_per_command": (
                self.angular_acceleration_per_command.tolist()
            ),
            "angular_rate_coefficient": self.angular_rate_coefficient.tolist(),
            "angular_rate_product_coefficient": (
                self.angular_rate_product_coefficient.tolist()
            ),
            "angular_intercept_rad_s2": self.angular_intercept_rad_s2.tolist(),
            "normalized_command_support_projector": (
                self.normalized_command_support_projector.tolist()
            ),
            "normalized_command_singular_values": (
                self.normalized_command_singular_values.tolist()
            ),
            "hover_command": (
                None if self.hover_command is None else self.hover_command.tolist()
            ),
            "validation": {
                "collective_rmse_m_s2": self.collective_validation_rmse_m_s2,
                "collective_baseline_rmse_m_s2": (
                    self.collective_baseline_validation_rmse_m_s2
                ),
                "collective_improvement": self.collective_validation_improvement,
                "angular_rmse_rad_s2": self.angular_validation_rmse_rad_s2.tolist(),
                "angular_baseline_rmse_rad_s2": (
                    self.angular_baseline_validation_rmse_rad_s2.tolist()
                ),
                "angular_improvement": self.angular_validation_improvement.tolist(),
            },
            "ready_for_hover": self.ready_for_hover,
            "ready_for_rate_arrest": self.ready_for_rate_arrest,
            "ready": self.ready,
            "wall_time_s": self.wall_time_s,
            "sample_period_s": self.sample_period_s,
            "evidence_duration_s": self.evidence_duration_s,
        }


def plan_bootstrap_excitation(
    provisional: BootstrapIdentificationResult,
    config: BootstrapExcitationConfig | None = None,
) -> BootstrapExcitationPlan:
    """Target the least validated output or an unsupported command direction."""

    config = BootstrapExcitationConfig() if config is None else config
    minimum = provisional.command_minimum
    maximum = provisional.command_maximum
    span = maximum - minimum
    center = (
        provisional.hover_command
        if provisional.ready_for_hover and provisional.hover_command is not None
        else 0.5 * (minimum + maximum)
    )
    if provisional.command_evidence_rank < 4:
        unsupported = np.eye(4) - provisional.normalized_command_support_projector
        eigenvalues, eigenvectors = np.linalg.eigh(unsupported)
        direction = eigenvectors[:, int(np.argmax(eigenvalues))]
        direction /= max(float(np.max(np.abs(direction))), 1e-9)
        target_axis = None
        reason = "least_supported_command_direction"
    else:
        target_axis = int(np.argmin(provisional.angular_validation_improvement))
        direction = provisional.angular_acceleration_per_command[target_axis].copy()
        maximum_effect = float(np.max(np.abs(direction)))
        if maximum_effect <= 1e-8:
            raise BootstrapModelNotReadyError(
                "full-rank commands produced no measurable angular effect"
            )
        direction /= maximum_effect
        reason = "weakest_validated_angular_output"
    commands = np.empty((config.interval_count, 4), dtype=np.float64)
    delta = config.amplitude_fraction_of_command_span * span * direction
    for index in range(config.interval_count):
        sign = 1.0 if index % 2 == 0 else -1.0
        commands[index] = np.clip(center + sign * delta, minimum, maximum)
    return BootstrapExcitationPlan(
        commands=commands,
        normalized_direction=direction,
        center_command=center,
        target_angular_axis=target_axis,
        selection_reason=reason,
    )


def _nuisance_pinv(matrix: Array, relative_tolerance: float) -> tuple[Array, Array]:
    """Invert only the nuisance directions the evidence window actually excited.

    Nuisance features carry their own physical units, so no normalized span is
    available.  The constant intercept column supplies the missing unit scale:
    its root-mean-square is exactly one, so the leading nuisance direction always
    has at least unit root-mean-square and a threshold relative to it is also an
    absolute floor in the features' own units.  Directions below the threshold
    are dropped rather than inverted, which is the same rank-support rule the
    command directions already follow.
    """

    left, singular_values, right = jnp.linalg.svd(matrix, full_matrices=False)
    threshold = jnp.maximum(singular_values[0] * relative_tolerance, 1e-12)
    supported = singular_values >= threshold
    inverse = jnp.where(
        supported,
        1.0 / jnp.maximum(singular_values, threshold),
        0.0,
    )
    return (right.T * inverse) @ left.T, jnp.sum(supported)


def _residualized_fit(
    command: Array,
    nuisance: Array,
    target: Array,
    *,
    rank_relative_tolerance: float,
    minimum_command_rms: float,
    nuisance_rank_relative_tolerance: float,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    nuisance_pinv, nuisance_rank = _nuisance_pinv(
        nuisance,
        nuisance_rank_relative_tolerance,
    )
    residual_command = command - nuisance @ (nuisance_pinv @ command)
    residual_target = target - nuisance @ (nuisance_pinv @ target)
    left, singular_values, right = jnp.linalg.svd(
        residual_command,
        full_matrices=False,
    )
    threshold = jnp.maximum(
        singular_values[0] * rank_relative_tolerance,
        minimum_command_rms * jnp.sqrt(residual_command.shape[0]),
    )
    supported = singular_values >= threshold
    inverse = jnp.where(
        supported,
        1.0 / jnp.maximum(singular_values, threshold),
        0.0,
    )
    command_pinv = (right.T * inverse) @ left.T
    command_coefficient = command_pinv @ residual_target
    nuisance_coefficient = nuisance_pinv @ (target - command @ command_coefficient)
    support_projector = (right.T * supported.astype(command.dtype)) @ right
    return (
        command_coefficient,
        nuisance_coefficient,
        support_projector,
        singular_values,
        jnp.sum(supported),
        nuisance_rank,
    )


def _rmse(target: Array, prediction: Array) -> Array:
    return jnp.sqrt(jnp.mean(jnp.square(target - prediction), axis=0))


@partial(
    jax.jit,
    static_argnames=(
        "training_count",
        "rank_relative_tolerance",
        "minimum_command_rms",
        "nuisance_rank_relative_tolerance",
    ),
)
def _bootstrap_fit_core(
    timestamps_s: Array,
    states: Array,
    applied_commands: Array,
    command_minimum: Array,
    command_maximum: Array,
    *,
    training_count: int,
    rank_relative_tolerance: float,
    minimum_command_rms: float,
    nuisance_rank_relative_tolerance: float,
) -> tuple[Array, ...]:
    intervals = jnp.diff(timestamps_s)
    rotations = quaternion_to_rotation_batch(states[:-1, 6:10])
    world_acceleration = jnp.diff(states[:, 3:6], axis=0) / intervals[:, None]
    gravity_compensated = world_acceleration + jnp.asarray((0.0, 0.0, GRAVITY_M_S2))
    body_specific_force = jnp.einsum(
        "nij,nj->ni",
        jnp.swapaxes(rotations, 1, 2),
        gravity_compensated,
    )
    body_velocity = jnp.einsum(
        "nij,nj->ni",
        jnp.swapaxes(rotations, 1, 2),
        states[:-1, 3:6],
    )
    angular_acceleration = jnp.diff(states[:, 10:13], axis=0) / intervals[:, None]

    span = command_maximum - command_minimum
    midpoint = 0.5 * (command_maximum + command_minimum)
    normalized_command = (applied_commands - midpoint) / span
    train_command = normalized_command[:training_count]
    validation_command = normalized_command[training_count:]

    train_force_nuisance = jnp.column_stack(
        (body_velocity[:training_count], jnp.ones(training_count))
    )
    validation_force_nuisance = jnp.column_stack(
        (
            body_velocity[training_count:],
            jnp.ones(body_velocity.shape[0] - training_count),
        )
    )
    train_force_target = body_specific_force[:training_count, 2:3]
    validation_force_target = body_specific_force[training_count:, 2:3]
    (
        force_command_coefficient,
        force_nuisance_coefficient,
        force_support_projector,
        _,
        force_command_rank,
        force_nuisance_rank,
    ) = _residualized_fit(
        train_command,
        train_force_nuisance,
        train_force_target,
        rank_relative_tolerance=rank_relative_tolerance,
        minimum_command_rms=minimum_command_rms,
        nuisance_rank_relative_tolerance=nuisance_rank_relative_tolerance,
    )
    force_prediction = (
        validation_command @ force_command_coefficient
        + validation_force_nuisance @ force_nuisance_coefficient
    )
    force_baseline_coefficient = (
        _nuisance_pinv(train_force_nuisance, nuisance_rank_relative_tolerance)[0]
        @ train_force_target
    )
    force_baseline_prediction = validation_force_nuisance @ force_baseline_coefficient

    angular_velocity = states[:-1, 10:13]
    angular_rate_products = jnp.column_stack(
        (
            angular_velocity[:, 0] * angular_velocity[:, 1],
            angular_velocity[:, 0] * angular_velocity[:, 2],
            angular_velocity[:, 1] * angular_velocity[:, 2],
        )
    )
    train_angular_nuisance = jnp.column_stack(
        (
            angular_velocity[:training_count],
            angular_rate_products[:training_count],
            jnp.ones(training_count),
        )
    )
    validation_angular_nuisance = jnp.column_stack(
        (
            angular_velocity[training_count:],
            angular_rate_products[training_count:],
            jnp.ones(states.shape[0] - 1 - training_count),
        )
    )
    train_angular_target = angular_acceleration[:training_count]
    validation_angular_target = angular_acceleration[training_count:]
    (
        angular_command_coefficient,
        angular_nuisance_coefficient,
        angular_support_projector,
        angular_command_singular_values,
        angular_command_rank,
        angular_nuisance_rank,
    ) = _residualized_fit(
        train_command,
        train_angular_nuisance,
        train_angular_target,
        rank_relative_tolerance=rank_relative_tolerance,
        minimum_command_rms=minimum_command_rms,
        nuisance_rank_relative_tolerance=nuisance_rank_relative_tolerance,
    )
    angular_prediction = (
        validation_command @ angular_command_coefficient
        + validation_angular_nuisance @ angular_nuisance_coefficient
    )
    angular_baseline_coefficient = (
        _nuisance_pinv(train_angular_nuisance, nuisance_rank_relative_tolerance)[0]
        @ train_angular_target
    )
    angular_baseline_prediction = (
        validation_angular_nuisance @ angular_baseline_coefficient
    )

    return (
        force_command_coefficient,
        force_nuisance_coefficient,
        angular_command_coefficient,
        angular_nuisance_coefficient,
        force_support_projector,
        force_command_rank,
        force_nuisance_rank,
        angular_support_projector,
        angular_command_singular_values,
        angular_command_rank,
        angular_nuisance_rank,
        _rmse(validation_force_target, force_prediction),
        _rmse(validation_force_target, force_baseline_prediction),
        _rmse(validation_angular_target, angular_prediction),
        _rmse(validation_angular_target, angular_baseline_prediction),
    )


class BootstrapMultirotorIdentifier:
    """Prewarmed fixed-shape bootstrap identifier."""

    def __init__(self, config: BootstrapIdentificationConfig | None = None) -> None:
        self.config = BootstrapIdentificationConfig() if config is None else config

    def prewarm(self) -> float:
        """Compile the exact fit shape without using airframe information."""

        count = self.config.interval_count
        timestamps = np.arange(count + 1, dtype=np.float32) * 0.02
        states = np.zeros((count + 1, 13), dtype=np.float32)
        states[:, 6] = 1.0
        phase = np.arange(count, dtype=np.float32)[:, None]
        motor = np.arange(4, dtype=np.float32)[None, :]
        command = 0.5 + 0.1 * np.sin(phase * 1.7 + motor * 1.3)
        started_at = time.perf_counter()
        result = self._run_core(timestamps, states, command)
        jax.block_until_ready(result)
        return time.perf_counter() - started_at

    def _run_core(
        self,
        timestamps_s: Any,
        states: Any,
        applied_commands: Any,
    ) -> tuple[Array, ...]:
        return _bootstrap_fit_core(
            jnp.asarray(timestamps_s),
            jnp.asarray(states),
            jnp.asarray(applied_commands),
            jnp.asarray(self.config.command_minimum),
            jnp.asarray(self.config.command_maximum),
            training_count=self.config.training_interval_count,
            rank_relative_tolerance=self.config.command_rank_relative_tolerance,
            minimum_command_rms=self.config.minimum_normalized_command_rms,
            nuisance_rank_relative_tolerance=(
                self.config.nuisance_rank_relative_tolerance
            ),
        )

    def fit(
        self,
        timestamps_s: Any,
        states: Any,
        applied_motor_commands: Any,
    ) -> BootstrapIdentificationResult:
        """Fit collective and angular input effects from measured applied inputs."""

        count = self.config.interval_count
        timestamps = np.asarray(timestamps_s, dtype=np.float64)
        state_values = np.asarray(states, dtype=np.float64)
        commands = np.asarray(applied_motor_commands, dtype=np.float64)
        if timestamps.shape != (count + 1,) or not np.all(np.isfinite(timestamps)):
            raise ValueError(f"timestamps_s must contain {count + 1} finite samples")
        if state_values.shape != (count + 1, 13) or not np.all(
            np.isfinite(state_values)
        ):
            raise ValueError(f"states must have shape ({count + 1}, 13) and be finite")
        if commands.shape != (count, 4) or not np.all(np.isfinite(commands)):
            raise ValueError(f"applied_motor_commands must have shape ({count}, 4)")
        minimum = np.asarray(self.config.command_minimum)
        maximum = np.asarray(self.config.command_maximum)
        if np.any(commands < minimum - 1e-9) or np.any(commands > maximum + 1e-9):
            raise ValueError("applied motor commands must lie within configured bounds")
        intervals = np.diff(timestamps)
        if np.any(intervals <= 0.0):
            raise ValueError("timestamps_s must be strictly increasing")
        sample_period = float(np.median(intervals))
        maximum_deviation = (
            self.config.maximum_sample_period_deviation_fraction * sample_period
        )
        if np.any(np.abs(intervals - sample_period) > maximum_deviation):
            raise ValueError("sample periods exceed the configured deviation")
        quaternion_norm = np.linalg.norm(state_values[:, 6:10], axis=1)
        if np.any(quaternion_norm < 1e-6):
            raise ValueError("state quaternions must have nonzero norm")

        started_at = time.perf_counter()
        raw = self._run_core(timestamps, state_values, commands)
        raw = tuple(np.asarray(item) for item in jax.block_until_ready(raw))
        wall_time_s = time.perf_counter() - started_at
        (
            normalized_force_effect,
            force_nuisance,
            normalized_angular_effect,
            angular_nuisance,
            force_support_projector,
            force_command_rank,
            force_nuisance_rank,
            angular_support_projector,
            angular_command_singular_values,
            angular_command_rank,
            angular_nuisance_rank,
            force_rmse,
            force_baseline_rmse,
            angular_rmse,
            angular_baseline_rmse,
        ) = raw
        span = maximum - minimum
        midpoint = 0.5 * (minimum + maximum)
        force_effect = normalized_force_effect[:, 0] / span
        angular_effect = (normalized_angular_effect / span[:, None]).T
        force_intercept = float(force_nuisance[-1, 0] - force_effect @ midpoint)
        angular_intercept = angular_nuisance[-1] - angular_effect @ midpoint
        force_rmse_value = float(force_rmse[0])
        force_baseline_value = float(force_baseline_rmse[0])
        angular_rmse_values = np.asarray(angular_rmse, dtype=np.float64)
        angular_baseline_values = np.asarray(
            angular_baseline_rmse,
            dtype=np.float64,
        )
        force_improvement = 1.0 - force_rmse_value / max(
            force_baseline_value,
            1e-9,
        )
        angular_improvement = 1.0 - angular_rmse_values / np.maximum(
            angular_baseline_values,
            1e-9,
        )

        collective_direction = np.ones(4, dtype=np.float64)
        supported_collective = force_support_projector @ collective_direction
        collective_support = float(
            np.dot(supported_collective, supported_collective)
            / np.dot(collective_direction, collective_direction)
        )
        effect_singular_values = np.linalg.svd(
            angular_effect @ angular_support_projector,
            compute_uv=False,
        )
        effect_threshold = max(
            effect_singular_values[0] * self.config.output_rank_relative_tolerance,
            1e-8,
        )
        angular_effect_rank = int(np.sum(effect_singular_values >= effect_threshold))

        hover_command: np.ndarray | None = None
        collective_sum = float(np.sum(force_effect))
        if collective_sum > 1e-8:
            hover_scalar = (GRAVITY_M_S2 - force_intercept) / collective_sum
            candidate_hover = np.full(4, hover_scalar, dtype=np.float64)
            if np.all(candidate_hover >= minimum) and np.all(
                candidate_hover <= maximum
            ):
                hover_command = candidate_hover
        ready_for_hover = bool(
            int(force_command_rank) >= 1
            and collective_support >= self.config.minimum_collective_support_fraction
            and force_improvement >= self.config.minimum_validation_improvement
            and hover_command is not None
        )
        ready_for_rate_arrest = bool(
            int(angular_command_rank) >= 3
            and angular_effect_rank == 3
            and np.all(
                angular_improvement >= self.config.minimum_validation_improvement
            )
        )
        return BootstrapIdentificationResult(
            collective_acceleration_per_command=force_effect,
            collective_velocity_coefficient=force_nuisance[:3, 0],
            collective_intercept_m_s2=force_intercept,
            angular_acceleration_per_command=angular_effect,
            angular_rate_coefficient=angular_nuisance[:3].T,
            angular_rate_product_coefficient=angular_nuisance[3:6].T,
            angular_intercept_rad_s2=angular_intercept,
            normalized_command_support_projector=angular_support_projector,
            normalized_command_singular_values=angular_command_singular_values,
            collective_command_evidence_rank=int(force_command_rank),
            command_evidence_rank=int(angular_command_rank),
            angular_effect_rank=angular_effect_rank,
            collective_nuisance_rank=int(force_nuisance_rank),
            angular_nuisance_rank=int(angular_nuisance_rank),
            collective_support_fraction=collective_support,
            hover_command=hover_command,
            collective_validation_rmse_m_s2=force_rmse_value,
            collective_baseline_validation_rmse_m_s2=force_baseline_value,
            collective_validation_improvement=force_improvement,
            angular_validation_rmse_rad_s2=angular_rmse_values,
            angular_baseline_validation_rmse_rad_s2=angular_baseline_values,
            angular_validation_improvement=angular_improvement,
            ready_for_hover=ready_for_hover,
            ready_for_rate_arrest=ready_for_rate_arrest,
            wall_time_s=wall_time_s,
            sample_period_s=sample_period,
            evidence_duration_s=float(timestamps[-1] - timestamps[0]),
            command_center=np.mean(
                commands[: self.config.training_interval_count], axis=0
            ),
            command_minimum=minimum,
            command_maximum=maximum,
        )
