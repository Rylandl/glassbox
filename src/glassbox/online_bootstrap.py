"""Recursive, support-aware identification and control from motor I/O alone.

This module is the continuous counterpart to :mod:`bootstrap_identification`.
It keeps a working local belief updated after every measured actuation interval
and gives control authority only to output directions supported by that belief.
There is deliberately no evidence-collection/model-running phase boundary.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from glassbox.dynamics import GRAVITY_M_S2


def _finite_vector(
    name: str,
    value: float | Sequence[float],
    size: int,
) -> np.ndarray:
    if np.isscalar(value):
        result = np.full(size, float(value), dtype=np.float64)
    else:
        result = np.asarray(tuple(value), dtype=np.float64)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain {size} finite values")
    return result


def _immutable_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).copy()
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must have shape {shape} and contain finite values")
    result.flags.writeable = False
    return result


def _quaternion_to_rotation(quaternion_wxyz: np.ndarray) -> np.ndarray:
    quaternion = quaternion_wxyz / np.linalg.norm(quaternion_wxyz)
    w, x, y, z = quaternion
    return np.asarray(
        (
            (
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ),
            (
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ),
            (
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ),
        )
    )


@dataclass(frozen=True)
class RecursiveBootstrapConfig:
    """Known I/O contract and evidence thresholds for the working belief."""

    command_minimum: float | tuple[float, float, float, float] = 0.0
    command_maximum: float | tuple[float, float, float, float] = 1.0
    forgetting_factor: float = 1.0
    command_rank_relative_tolerance: float = 0.025
    minimum_normalized_command_rms: float = 0.003
    output_rank_relative_tolerance: float = 0.04
    authority_start_interval_count: int = 8
    authority_full_interval_count: int = 32
    minimum_certification_interval_count: int = 48

    def __post_init__(self) -> None:
        minimum = _finite_vector("command_minimum", self.command_minimum, 4)
        maximum = _finite_vector("command_maximum", self.command_maximum, 4)
        if np.any(minimum >= maximum):
            raise ValueError("command_minimum must be below command_maximum")
        if not 0.9 <= self.forgetting_factor <= 1.0:
            raise ValueError("forgetting_factor must lie in [0.9, 1.0]")
        for name in (
            "command_rank_relative_tolerance",
            "minimum_normalized_command_rms",
            "output_rank_relative_tolerance",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 < value < 1.0:
                raise ValueError(f"{name} must lie strictly between zero and one")
        if self.authority_start_interval_count < 4:
            raise ValueError("authority_start_interval_count must be at least four")
        if self.authority_full_interval_count <= self.authority_start_interval_count:
            raise ValueError("full authority must follow authority start")
        if self.minimum_certification_interval_count < (
            self.authority_full_interval_count
        ):
            raise ValueError("certification cannot precede full authority")
        object.__setattr__(self, "command_minimum", tuple(minimum))
        object.__setattr__(self, "command_maximum", tuple(maximum))


@dataclass(frozen=True)
class RecursiveBootstrapBelief:
    """One auditable snapshot of the continuously updated local model."""

    interval_count: int
    effective_interval_count: float
    collective_acceleration_per_command: np.ndarray
    collective_velocity_coefficient: np.ndarray
    collective_intercept_m_s2: float
    angular_acceleration_per_command: np.ndarray
    angular_rate_coefficient: np.ndarray
    angular_rate_product_coefficient: np.ndarray
    angular_intercept_rad_s2: np.ndarray
    normalized_command_support_projector: np.ndarray
    normalized_command_singular_values: np.ndarray
    command_evidence_rank: int
    angular_effect_rank: int
    angular_output_support_projector: np.ndarray
    collective_support_fraction: float
    collective_authority: float
    angular_axis_authority: np.ndarray
    hover_command: np.ndarray | None
    update_wall_time_s: float

    def __post_init__(self) -> None:
        arrays = {
            "collective_acceleration_per_command": (4,),
            "collective_velocity_coefficient": (3,),
            "angular_acceleration_per_command": (3, 4),
            "angular_rate_coefficient": (3, 3),
            "angular_rate_product_coefficient": (3, 3),
            "angular_intercept_rad_s2": (3,),
            "normalized_command_support_projector": (4, 4),
            "normalized_command_singular_values": (4,),
            "angular_output_support_projector": (3, 3),
            "angular_axis_authority": (3,),
        }
        for name, shape in arrays.items():
            object.__setattr__(
                self,
                name,
                _immutable_array(getattr(self, name), shape, name),
            )
        if self.hover_command is not None:
            object.__setattr__(
                self,
                "hover_command",
                _immutable_array(self.hover_command, (4,), "hover_command"),
            )
        scalars = (
            self.effective_interval_count,
            self.collective_intercept_m_s2,
            self.collective_support_fraction,
            self.collective_authority,
            self.update_wall_time_s,
        )
        if self.interval_count < 0 or not np.all(np.isfinite(scalars)):
            raise ValueError("recursive belief counts and scalars must be finite")
        if np.any(self.angular_axis_authority < 0.0) or np.any(
            self.angular_axis_authority > 1.0
        ):
            raise ValueError("angular authority must lie inside [0, 1]")

    @property
    def has_any_control_authority(self) -> bool:
        return bool(
            self.collective_authority > 0.0 or np.any(self.angular_axis_authority > 0.0)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": "recursive_rank_supported_multirotor_bootstrap_v1",
            "airframe_parameter_prior_used": False,
            "canonical_motor_mixer_assumed": False,
            "interval_count": self.interval_count,
            "effective_interval_count": self.effective_interval_count,
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
            "command_evidence_rank": self.command_evidence_rank,
            "angular_effect_rank": self.angular_effect_rank,
            "angular_output_support_projector": (
                self.angular_output_support_projector.tolist()
            ),
            "collective_support_fraction": self.collective_support_fraction,
            "collective_authority": self.collective_authority,
            "angular_axis_authority": self.angular_axis_authority.tolist(),
            "hover_command": (
                None if self.hover_command is None else self.hover_command.tolist()
            ),
            "update_wall_time_s": self.update_wall_time_s,
        }


@dataclass(frozen=True)
class ProgressiveBootstrapCommand:
    """One bounded command combining supported feedback and exploration."""

    command: np.ndarray
    feedback_command: np.ndarray
    excitation_command_delta: np.ndarray
    desired_world_acceleration_m_s2: np.ndarray
    desired_angular_acceleration_rad_s2: np.ndarray
    collective_authority: float
    angular_axis_authority: np.ndarray

    def __post_init__(self) -> None:
        for name, shape in {
            "command": (4,),
            "feedback_command": (4,),
            "excitation_command_delta": (4,),
            "desired_world_acceleration_m_s2": (3,),
            "desired_angular_acceleration_rad_s2": (3,),
            "angular_axis_authority": (3,),
        }.items():
            object.__setattr__(
                self,
                name,
                _immutable_array(getattr(self, name), shape, name),
            )
        if not math.isfinite(self.collective_authority):
            raise ValueError("collective_authority must be finite")


class RecursiveBootstrapIdentifier:
    """Update direct motor effects after every observed actuation interval."""

    _FORCE_NUISANCE_SIZE = 4
    _ANGULAR_NUISANCE_SIZE = 7

    def __init__(self, config: RecursiveBootstrapConfig | None = None) -> None:
        self.config = RecursiveBootstrapConfig() if config is None else config
        self._minimum = np.asarray(self.config.command_minimum)
        self._maximum = np.asarray(self.config.command_maximum)
        self._span = self._maximum - self._minimum
        self._midpoint = 0.5 * (self._minimum + self._maximum)
        self._interval_count = 0
        self._weight = 0.0
        self._force_gram = np.zeros((8, 8), dtype=np.float64)
        self._force_rhs = np.zeros((8, 1), dtype=np.float64)
        self._angular_gram = np.zeros((11, 11), dtype=np.float64)
        self._angular_rhs = np.zeros((11, 3), dtype=np.float64)
        self._belief = self._empty_belief()
        self._certified_belief: RecursiveBootstrapBelief | None = None

    def _empty_belief(self) -> RecursiveBootstrapBelief:
        return RecursiveBootstrapBelief(
            interval_count=0,
            effective_interval_count=0.0,
            collective_acceleration_per_command=np.zeros(4),
            collective_velocity_coefficient=np.zeros(3),
            collective_intercept_m_s2=0.0,
            angular_acceleration_per_command=np.zeros((3, 4)),
            angular_rate_coefficient=np.zeros((3, 3)),
            angular_rate_product_coefficient=np.zeros((3, 3)),
            angular_intercept_rad_s2=np.zeros(3),
            normalized_command_support_projector=np.zeros((4, 4)),
            normalized_command_singular_values=np.zeros(4),
            command_evidence_rank=0,
            angular_effect_rank=0,
            angular_output_support_projector=np.zeros((3, 3)),
            collective_support_fraction=0.0,
            collective_authority=0.0,
            angular_axis_authority=np.zeros(3),
            hover_command=None,
            update_wall_time_s=0.0,
        )

    @property
    def belief(self) -> RecursiveBootstrapBelief:
        return self._belief

    @property
    def certified_belief(self) -> RecursiveBootstrapBelief | None:
        """Last fully supported belief admitted for persistent control use."""

        return self._certified_belief

    @property
    def control_belief(self) -> RecursiveBootstrapBelief:
        """Progressive working belief, then the last supported commit."""

        return (
            self._belief if self._certified_belief is None else self._certified_belief
        )

    @staticmethod
    def _supported_fit(
        gram: np.ndarray,
        rhs: np.ndarray,
        *,
        nuisance_size: int,
        effective_count: float,
        relative_tolerance: float,
        minimum_rms: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        command_gram = gram[:4, :4]
        cross_gram = gram[:4, 4:]
        nuisance_gram = gram[4:, 4:]
        nuisance_inverse = np.linalg.pinv(nuisance_gram, rcond=1e-8)
        residual_gram = command_gram - (cross_gram @ nuisance_inverse @ cross_gram.T)
        residual_gram = 0.5 * (residual_gram + residual_gram.T)
        eigenvalues, eigenvectors = np.linalg.eigh(residual_gram)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues[order], 0.0)
        right = eigenvectors[:, order]
        singular_values = np.sqrt(eigenvalues)
        leading = float(singular_values[0]) if len(singular_values) else 0.0
        threshold = max(
            leading * relative_tolerance,
            minimum_rms * math.sqrt(max(effective_count, 1.0)),
        )
        supported = singular_values >= threshold
        inverse_eigenvalues = np.where(
            supported,
            1.0 / np.maximum(eigenvalues, threshold * threshold),
            0.0,
        )
        residual_inverse = (right * inverse_eigenvalues) @ right.T
        residual_rhs = rhs[:4] - cross_gram @ nuisance_inverse @ rhs[4:]
        command_coefficient = residual_inverse @ residual_rhs
        nuisance_coefficient = nuisance_inverse @ (
            rhs[4:] - cross_gram.T @ command_coefficient
        )
        support_projector = (right * supported.astype(np.float64)) @ right.T
        if nuisance_coefficient.shape[0] != nuisance_size:
            raise RuntimeError("recursive nuisance feature shape changed")
        return (
            command_coefficient,
            nuisance_coefficient,
            support_projector,
            singular_values,
            int(np.sum(supported)),
        )

    def update(
        self,
        previous_state: Sequence[float],
        current_state: Sequence[float],
        average_applied_motor_command: Sequence[float],
        sample_period_s: float,
    ) -> RecursiveBootstrapBelief:
        """Assimilate one measured transition and return the new belief."""

        started_at = time.perf_counter()
        previous = _finite_vector("previous_state", previous_state, 13)
        current = _finite_vector("current_state", current_state, 13)
        command = _finite_vector(
            "average_applied_motor_command",
            average_applied_motor_command,
            4,
        )
        if np.any(command < self._minimum - 1e-9) or np.any(
            command > self._maximum + 1e-9
        ):
            raise ValueError("applied motor command lies outside configured bounds")
        if not math.isfinite(sample_period_s) or sample_period_s <= 0.0:
            raise ValueError("sample_period_s must be finite and positive")
        rotation = _quaternion_to_rotation(previous[6:10])
        world_acceleration = (current[3:6] - previous[3:6]) / sample_period_s
        body_specific_force = rotation.T @ (
            world_acceleration + np.asarray((0.0, 0.0, GRAVITY_M_S2))
        )
        body_velocity = rotation.T @ previous[3:6]
        angular_velocity = previous[10:13]
        angular_acceleration = (current[10:13] - previous[10:13]) / sample_period_s
        rate_products = np.asarray(
            (
                angular_velocity[0] * angular_velocity[1],
                angular_velocity[0] * angular_velocity[2],
                angular_velocity[1] * angular_velocity[2],
            )
        )
        normalized_command = (command - self._midpoint) / self._span
        force_features = np.concatenate((normalized_command, body_velocity, np.ones(1)))
        angular_features = np.concatenate(
            (normalized_command, angular_velocity, rate_products, np.ones(1))
        )
        forgetting = self.config.forgetting_factor
        self._force_gram *= forgetting
        self._force_rhs *= forgetting
        self._angular_gram *= forgetting
        self._angular_rhs *= forgetting
        self._force_gram += np.outer(force_features, force_features)
        self._force_rhs += np.outer(force_features, body_specific_force[2:3])
        self._angular_gram += np.outer(angular_features, angular_features)
        self._angular_rhs += np.outer(angular_features, angular_acceleration)
        self._weight = forgetting * self._weight + 1.0
        self._interval_count += 1

        force = self._supported_fit(
            self._force_gram,
            self._force_rhs,
            nuisance_size=self._FORCE_NUISANCE_SIZE,
            effective_count=self._weight,
            relative_tolerance=self.config.command_rank_relative_tolerance,
            minimum_rms=self.config.minimum_normalized_command_rms,
        )
        angular = self._supported_fit(
            self._angular_gram,
            self._angular_rhs,
            nuisance_size=self._ANGULAR_NUISANCE_SIZE,
            effective_count=self._weight,
            relative_tolerance=self.config.command_rank_relative_tolerance,
            minimum_rms=self.config.minimum_normalized_command_rms,
        )
        (
            normalized_force_effect,
            force_nuisance,
            force_support,
            _,
            force_rank,
        ) = force
        (
            normalized_angular_effect,
            angular_nuisance,
            angular_support,
            angular_singular_values,
            angular_command_rank,
        ) = angular
        force_effect = normalized_force_effect[:, 0] / self._span
        angular_effect = (normalized_angular_effect / self._span[:, None]).T
        force_intercept = float(force_nuisance[-1, 0] - force_effect @ self._midpoint)
        angular_intercept = angular_nuisance[-1] - angular_effect @ self._midpoint

        collective_direction = np.ones(4, dtype=np.float64)
        collective_support = float(
            np.linalg.norm(force_support @ collective_direction) ** 2 / 4.0
        )
        supported_effect = angular_effect @ angular_support
        left, effect_singular_values, _ = np.linalg.svd(
            supported_effect,
            full_matrices=False,
        )
        effect_threshold = max(
            (
                float(effect_singular_values[0])
                * self.config.output_rank_relative_tolerance
            ),
            1e-8,
        )
        effect_supported = effect_singular_values >= effect_threshold
        angular_effect_rank = int(np.sum(effect_supported))
        angular_output_support = (left * effect_supported.astype(np.float64)) @ left.T
        evidence_authority = float(
            np.clip(
                (self._interval_count - self.config.authority_start_interval_count)
                / (
                    self.config.authority_full_interval_count
                    - self.config.authority_start_interval_count
                ),
                0.0,
                1.0,
            )
        )
        # Output support is conditional on the explored motor subspace.  A
        # one-dimensional effect may be real, but it should not receive the
        # same authority as a result supported across all four command
        # directions.  This factor grows smoothly instead of creating a ready
        # switch.
        command_coverage = angular_command_rank / 4.0
        angular_axis_authority = np.clip(
            evidence_authority * command_coverage * np.diag(angular_output_support),
            0.0,
            1.0,
        )
        hover_command: np.ndarray | None = None
        collective_sum = float(np.sum(force_effect))
        if collective_sum > 1e-8:
            hover_scalar = (GRAVITY_M_S2 - force_intercept) / collective_sum
            candidate = np.full(4, hover_scalar)
            if np.all(candidate >= self._minimum) and np.all(
                candidate <= self._maximum
            ):
                hover_command = candidate
        collective_authority = 0.0
        if force_rank >= 1 and hover_command is not None:
            collective_authority = float(
                np.clip(evidence_authority * collective_support, 0.0, 1.0)
            )
        self._belief = RecursiveBootstrapBelief(
            interval_count=self._interval_count,
            effective_interval_count=self._weight,
            collective_acceleration_per_command=force_effect,
            collective_velocity_coefficient=force_nuisance[:3, 0],
            collective_intercept_m_s2=force_intercept,
            angular_acceleration_per_command=angular_effect,
            angular_rate_coefficient=angular_nuisance[:3].T,
            angular_rate_product_coefficient=angular_nuisance[3:6].T,
            angular_intercept_rad_s2=angular_intercept,
            normalized_command_support_projector=angular_support,
            normalized_command_singular_values=angular_singular_values,
            command_evidence_rank=angular_command_rank,
            angular_effect_rank=angular_effect_rank,
            angular_output_support_projector=angular_output_support,
            collective_support_fraction=collective_support,
            collective_authority=collective_authority,
            angular_axis_authority=angular_axis_authority,
            hover_command=hover_command,
            update_wall_time_s=time.perf_counter() - started_at,
        )
        if (
            self._certified_belief is None
            and self._belief.interval_count
            >= self.config.minimum_certification_interval_count
            and self._belief.command_evidence_rank == 4
            and self._belief.angular_effect_rank == 3
            and self._belief.hover_command is not None
            and self._belief.collective_authority >= 0.95
            and np.all(self._belief.angular_axis_authority >= 0.95)
        ):
            # Later candidates continue to be computed on every interval.  A
            # replacement needs independent predictive validation; absent that
            # evidence, loss of excitation must not erase established support.
            self._certified_belief = self._belief
        return self._belief


@dataclass(frozen=True)
class ProgressiveBootstrapControllerConfig:
    """Bounded dual-control policy for identification during stabilization."""

    velocity_gain: tuple[float, float, float] = (1.5, 1.5, 4.0)
    maximum_world_acceleration_m_s2: tuple[float, float, float] = (2.5, 2.5, 4.0)
    maximum_tilt_rad: float = 0.50
    attitude_gain: tuple[float, float] = (14.0, 14.0)
    angular_rate_gain: tuple[float, float, float] = (6.0, 6.0, 3.0)
    initial_excitation_fraction: float = 0.12
    continuing_excitation_fraction: float = 0.0
    maximum_feedback_delta: float = 0.35
    maximum_motor_step: float = 0.10

    def __post_init__(self) -> None:
        for name, size in (
            ("velocity_gain", 3),
            ("maximum_world_acceleration_m_s2", 3),
            ("attitude_gain", 2),
            ("angular_rate_gain", 3),
        ):
            values = _finite_vector(name, getattr(self, name), size)
            if np.any(values <= 0.0):
                raise ValueError(f"{name} must be positive")
        if not 0.0 < self.maximum_tilt_rad < 1.0:
            raise ValueError("maximum_tilt_rad must lie inside (0, 1)")
        if (
            not 0.0
            <= self.continuing_excitation_fraction
            <= (self.initial_excitation_fraction)
        ):
            raise ValueError("continuing excitation must not exceed initial excitation")
        if not 0.0 < self.maximum_feedback_delta <= 0.5:
            raise ValueError("maximum_feedback_delta must lie inside (0, 0.5]")
        if not 0.0 < self.maximum_motor_step <= 1.0:
            raise ValueError("maximum_motor_step must lie inside (0, 1]")


class ProgressiveBootstrapController:
    """Blend exploration with feedback only in currently supported directions."""

    def __init__(
        self,
        identifier_config: RecursiveBootstrapConfig | None = None,
        config: ProgressiveBootstrapControllerConfig | None = None,
    ) -> None:
        self.identifier_config = (
            RecursiveBootstrapConfig()
            if identifier_config is None
            else identifier_config
        )
        self.config = (
            ProgressiveBootstrapControllerConfig() if config is None else config
        )
        self._minimum = np.asarray(self.identifier_config.command_minimum)
        self._maximum = np.asarray(self.identifier_config.command_maximum)
        self._span = self._maximum - self._minimum
        self._midpoint = 0.5 * (self._minimum + self._maximum)
        generator = np.random.default_rng(11)
        patterns = generator.standard_normal((4096, 4))
        self._patterns = patterns / np.maximum(
            np.max(np.abs(patterns), axis=1, keepdims=True),
            1e-9,
        )

    def command(
        self,
        state: Sequence[float],
        belief: RecursiveBootstrapBelief,
        *,
        previous_command: Sequence[float],
    ) -> ProgressiveBootstrapCommand:
        state_array = _finite_vector("state", state, 13)
        previous = _finite_vector("previous_command", previous_command, 4)
        quaternion = state_array[6:10]
        rotation = _quaternion_to_rotation(quaternion)
        velocity = state_array[3:6]
        angular_velocity = state_array[10:13]
        velocity_authority = float(
            min(
                belief.collective_authority,
                belief.angular_axis_authority[0],
                belief.angular_axis_authority[1],
            )
        )
        velocity_gain = np.asarray(self.config.velocity_gain)
        maximum_acceleration = np.asarray(self.config.maximum_world_acceleration_m_s2)
        desired_world_acceleration = np.clip(
            -velocity_authority * velocity_gain * velocity,
            -maximum_acceleration,
            maximum_acceleration,
        )
        desired_specific_force = desired_world_acceleration + np.asarray(
            (0.0, 0.0, GRAVITY_M_S2)
        )
        vertical_force = max(float(desired_specific_force[2]), 1e-3)
        maximum_horizontal_force = vertical_force * math.tan(
            self.config.maximum_tilt_rad
        )
        horizontal_norm = float(np.linalg.norm(desired_specific_force[:2]))
        if horizontal_norm > maximum_horizontal_force:
            desired_specific_force[:2] *= maximum_horizontal_force / horizontal_norm
        desired_force_magnitude = float(np.linalg.norm(desired_specific_force))
        desired_thrust_world = desired_specific_force / desired_force_magnitude
        desired_thrust_body = rotation.T @ desired_thrust_world
        tilt_error_body = np.cross(
            np.asarray((0.0, 0.0, 1.0)),
            desired_thrust_body,
        )
        attitude_gain = np.asarray(self.config.attitude_gain)
        angular_rate_gain = np.asarray(self.config.angular_rate_gain)
        desired_angular_acceleration = np.asarray(
            (
                attitude_gain[0] * tilt_error_body[0]
                - angular_rate_gain[0] * angular_velocity[0],
                attitude_gain[1] * tilt_error_body[1]
                - angular_rate_gain[1] * angular_velocity[1],
                -angular_rate_gain[2] * angular_velocity[2],
            )
        )

        collective_reference = self._midpoint.copy()
        collective_sum = float(np.sum(belief.collective_acceleration_per_command))
        if belief.hover_command is not None and collective_sum > 1e-8:
            estimated_scalar = (
                desired_force_magnitude - belief.collective_intercept_m_s2
            ) / collective_sum
            estimated_reference = np.full(4, estimated_scalar)
            collective_reference = (
                1.0 - belief.collective_authority
            ) * self._midpoint + belief.collective_authority * estimated_reference
        collective_reference = np.clip(
            collective_reference,
            self._minimum,
            self._maximum,
        )

        angular_effect = (
            belief.angular_acceleration_per_command
            @ belief.normalized_command_support_projector
        )
        supported_desired = belief.angular_output_support_projector @ (
            belief.angular_axis_authority * desired_angular_acceleration
        )
        if belief.angular_effect_rank:
            delta = np.linalg.pinv(angular_effect, rcond=1e-5) @ supported_desired
        else:
            delta = np.zeros(4)
        delta_limit = self.config.maximum_feedback_delta * float(
            np.max(belief.angular_axis_authority)
        )
        delta = np.clip(delta, -delta_limit, delta_limit)
        feedback = np.clip(
            collective_reference + delta,
            self._minimum,
            self._maximum,
        )

        # Collective is often the first identified direction.  Do not let that
        # prematurely extinguish the differential excitation still needed for
        # roll, pitch, and yaw authority.
        evidence_authority = float(
            min(
                belief.collective_authority,
                belief.command_evidence_rank / 4.0,
                belief.angular_effect_rank / 3.0,
            )
        )
        excitation_amplitude = self.config.continuing_excitation_fraction + (
            self.config.initial_excitation_fraction
            - self.config.continuing_excitation_fraction
        ) * (1.0 - evidence_authority)
        pattern = self._patterns[belief.interval_count % len(self._patterns)]
        excitation = excitation_amplitude * self._span * pattern
        unconstrained = np.clip(
            feedback + excitation,
            self._minimum,
            self._maximum,
        )
        # The first post-release command may jump to the bounds-derived midpoint;
        # subsequent requested commands are slew bounded. Rotor lag remains in the
        # hidden plant and the identifier consumes measured applied motor state.
        if belief.interval_count == 0:
            bounded = unconstrained
        else:
            bounded = np.clip(
                unconstrained,
                previous - self.config.maximum_motor_step,
                previous + self.config.maximum_motor_step,
            )
            bounded = np.clip(bounded, self._minimum, self._maximum)
        return ProgressiveBootstrapCommand(
            command=bounded,
            feedback_command=feedback,
            excitation_command_delta=bounded - feedback,
            desired_world_acceleration_m_s2=(
                desired_specific_force - np.asarray((0.0, 0.0, GRAVITY_M_S2))
            ),
            desired_angular_acceleration_rad_s2=desired_angular_acceleration,
            collective_authority=belief.collective_authority,
            angular_axis_authority=belief.angular_axis_authority,
        )
