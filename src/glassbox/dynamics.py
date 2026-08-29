"""Compact differentiable multirotor and fixed-wing dynamics families."""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from glassbox.model_family import (
    FIXED_WING_FAMILY,
    MULTIROTOR_FAMILY,
    DynamicsModelFamily,
)


GRAVITY_M_S2 = 9.80665
QUADROTOR_CONTROL_SIZE = MULTIROTOR_FAMILY.control_size
QUADROTOR_CONTROL_NAMES = MULTIROTOR_FAMILY.control_names
FIXED_WING_CONTROL_NAMES = FIXED_WING_FAMILY.control_names
FIXED_WING_CONTROL_ROLES = FIXED_WING_FAMILY.control_roles
WIND_EXOGENOUS_ROLES = ("wind_north", "wind_west")
MAX_INTERNAL_INTEGRATION_STEP_S = 0.025
MULTIROTOR_ROTATIONAL_STATE_SIZE = 3
MAX_ANGULAR_CONTROL_CROSS_COUPLING = 0.5
MAX_THRUST_COMMAND_OFFSET = 0.3

# Motor order: front-left, front-right, rear-right, rear-left.
# Each row maps motor commands to a roll, pitch, or yaw differential.
MOTOR_MIXER = jnp.asarray(
    [
        [1.0, -1.0, -1.0, 1.0],
        [-1.0, -1.0, 1.0, 1.0],
        [1.0, -1.0, 1.0, -1.0],
    ]
)


class DynamicsParams(NamedTuple):
    """Unconstrained parameters for the effective vehicle dynamics.

    Positive physical values are stored in log space so gradient-based fitting
    cannot produce negative thrust, acceleration, or damping coefficients. The
    shared normalized-command offset uses a bounded signed parameterization.
    """

    log_thrust_accel: Array
    thrust_command_offset_unconstrained: Array
    log_angular_accel: Array
    log_linear_drag: Array
    log_angular_drag: Array
    log_motor_time_constant: Array
    log_angular_response_time_constant: Array
    angular_control_cross_coupling_unconstrained: Array

    @classmethod
    def from_physical(
        cls,
        *,
        thrust_accel: float,
        thrust_command_offset: float = 0.0,
        angular_accel: tuple[float, float, float],
        linear_drag: float,
        angular_drag: tuple[float, float, float],
        motor_time_constant: float,
        angular_response_time_constant: tuple[float, float, float] = (
            1e-4,
            1e-4,
            1e-4,
        ),
        angular_control_cross_coupling: tuple[
            tuple[float, float, float],
            tuple[float, float, float],
            tuple[float, float, float],
        ] = (
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
        ),
    ) -> "DynamicsParams":
        if (
            not math.isfinite(thrust_command_offset)
            or abs(thrust_command_offset) >= MAX_THRUST_COMMAND_OFFSET
        ):
            raise ValueError(
                "thrust_command_offset must be finite and strictly within "
                f"{-MAX_THRUST_COMMAND_OFFSET:g} and "
                f"{MAX_THRUST_COMMAND_OFFSET:g}"
            )
        cross_coupling = jnp.asarray(angular_control_cross_coupling)
        if cross_coupling.shape != (3, 3):
            raise ValueError(
                "angular_control_cross_coupling must have shape (3, 3)"
            )
        cross_coupling = cross_coupling.at[jnp.diag_indices(3)].set(0.0)
        normalized_cross_coupling = jnp.clip(
            cross_coupling / MAX_ANGULAR_CONTROL_CROSS_COUPLING,
            -0.999,
            0.999,
        )
        return cls(
            log_thrust_accel=jnp.log(jnp.asarray(thrust_accel)),
            thrust_command_offset_unconstrained=jnp.arctanh(
                jnp.asarray(
                    thrust_command_offset / MAX_THRUST_COMMAND_OFFSET
                )
            ),
            log_angular_accel=jnp.log(jnp.asarray(angular_accel)),
            log_linear_drag=jnp.log(jnp.asarray(linear_drag)),
            log_angular_drag=jnp.log(jnp.asarray(angular_drag)),
            log_motor_time_constant=jnp.log(jnp.asarray(motor_time_constant)),
            log_angular_response_time_constant=jnp.log(
                jnp.asarray(angular_response_time_constant)
            ),
            angular_control_cross_coupling_unconstrained=jnp.arctanh(
                normalized_cross_coupling
            ),
        )

    def physical(self) -> dict[str, Array]:
        angular_accel = jnp.exp(self.log_angular_accel)
        cross_coupling = MAX_ANGULAR_CONTROL_CROSS_COUPLING * jnp.tanh(
            self.angular_control_cross_coupling_unconstrained
        )
        cross_coupling = cross_coupling.at[jnp.diag_indices(3)].set(0.0)
        return {
            "thrust_accel": jnp.exp(self.log_thrust_accel),
            "thrust_command_offset": MAX_THRUST_COMMAND_OFFSET * jnp.tanh(
                self.thrust_command_offset_unconstrained
            ),
            "angular_accel": angular_accel,
            "linear_drag": jnp.exp(self.log_linear_drag),
            "angular_drag": jnp.exp(self.log_angular_drag),
            "motor_time_constant": jnp.exp(self.log_motor_time_constant),
            "angular_response_time_constant": jnp.exp(
                self.log_angular_response_time_constant
            ),
            "angular_control_cross_coupling": cross_coupling,
            "angular_control_matrix": jnp.diag(angular_accel) @ (
                jnp.eye(3) + cross_coupling
            ),
        }


class FixedWingDynamicsParams(NamedTuple):
    """Positive effective coefficients for a low-angle fixed-wing model.

    Lift, drag, lateral stability, and surface moments are represented as
    acceleration coefficients, so mass, reference area, air density, and
    inertia are absorbed into values that can be identified from telemetry.
    """

    log_thrust_accel: Array
    log_lift_accel_per_speed_sq: Array
    log_lift_alpha_accel_per_speed_sq: Array
    log_drag_accel_per_speed_sq: Array
    log_side_force_accel_per_speed: Array
    log_surface_angular_accel_per_speed_sq: Array
    lateral_surface_cross_angular_accel_per_speed_sq: Array
    log_pitch_stability_accel_per_speed_sq: Array
    log_lateral_stability_angular_accel_per_speed_sq: Array
    log_angular_drag_per_speed: Array
    log_actuator_time_constant: Array
    surface_trim_unconstrained: Array
    log_flap_lift_accel_per_speed_sq: Array
    log_flap_drag_accel_per_speed_sq: Array
    flap_pitch_angular_accel_per_speed_sq: Array
    flap_trim_unconstrained: Array

    @classmethod
    def from_physical(
        cls,
        *,
        thrust_accel: float,
        lift_accel_per_speed_sq: float,
        lift_alpha_accel_per_speed_sq: float,
        drag_accel_per_speed_sq: float,
        side_force_accel_per_speed: float,
        surface_angular_accel_per_speed_sq: tuple[float, float, float],
        pitch_stability_accel_per_speed_sq: float,
        lateral_stability_angular_accel_per_speed_sq: tuple[float, float],
        angular_drag_per_speed: tuple[float, float, float],
        actuator_time_constant: float,
        lateral_surface_cross_angular_accel_per_speed_sq: tuple[float, float] = (
            0.0,
            0.0,
        ),
        surface_trim: tuple[float, float, float] = (0.0, 0.0, 0.0),
        flap_lift_accel_per_speed_sq: float = 1e-6,
        flap_drag_accel_per_speed_sq: float = 1e-6,
        flap_pitch_angular_accel_per_speed_sq: float = 0.0,
        flap_trim: float = 0.0,
    ) -> "FixedWingDynamicsParams":
        return cls(
            log_thrust_accel=jnp.log(jnp.asarray(thrust_accel)),
            log_lift_accel_per_speed_sq=jnp.log(
                jnp.asarray(lift_accel_per_speed_sq)
            ),
            log_lift_alpha_accel_per_speed_sq=jnp.log(
                jnp.asarray(lift_alpha_accel_per_speed_sq)
            ),
            log_drag_accel_per_speed_sq=jnp.log(
                jnp.asarray(drag_accel_per_speed_sq)
            ),
            log_side_force_accel_per_speed=jnp.log(
                jnp.asarray(side_force_accel_per_speed)
            ),
            log_surface_angular_accel_per_speed_sq=jnp.log(
                jnp.asarray(surface_angular_accel_per_speed_sq)
            ),
            lateral_surface_cross_angular_accel_per_speed_sq=jnp.asarray(
                lateral_surface_cross_angular_accel_per_speed_sq
            ),
            log_pitch_stability_accel_per_speed_sq=jnp.log(
                jnp.asarray(pitch_stability_accel_per_speed_sq)
            ),
            log_lateral_stability_angular_accel_per_speed_sq=jnp.log(
                jnp.asarray(lateral_stability_angular_accel_per_speed_sq)
            ),
            log_angular_drag_per_speed=jnp.log(
                jnp.asarray(angular_drag_per_speed)
            ),
            log_actuator_time_constant=jnp.log(
                jnp.asarray(actuator_time_constant)
            ),
            surface_trim_unconstrained=jnp.arctanh(
                jnp.clip(jnp.asarray(surface_trim), -0.999, 0.999)
            ),
            log_flap_lift_accel_per_speed_sq=jnp.log(
                jnp.asarray(flap_lift_accel_per_speed_sq)
            ),
            log_flap_drag_accel_per_speed_sq=jnp.log(
                jnp.asarray(flap_drag_accel_per_speed_sq)
            ),
            flap_pitch_angular_accel_per_speed_sq=jnp.asarray(
                flap_pitch_angular_accel_per_speed_sq
            ),
            flap_trim_unconstrained=jnp.arctanh(
                jnp.clip(jnp.asarray(flap_trim), -0.999, 0.999)
            ),
        )

    def physical(self) -> dict[str, Array]:
        return {
            "thrust_accel": jnp.exp(self.log_thrust_accel),
            "lift_accel_per_speed_sq": jnp.exp(
                self.log_lift_accel_per_speed_sq
            ),
            "lift_alpha_accel_per_speed_sq": jnp.exp(
                self.log_lift_alpha_accel_per_speed_sq
            ),
            "drag_accel_per_speed_sq": jnp.exp(
                self.log_drag_accel_per_speed_sq
            ),
            "side_force_accel_per_speed": jnp.exp(
                self.log_side_force_accel_per_speed
            ),
            "surface_angular_accel_per_speed_sq": jnp.exp(
                self.log_surface_angular_accel_per_speed_sq
            ),
            "lateral_surface_cross_angular_accel_per_speed_sq": (
                self.lateral_surface_cross_angular_accel_per_speed_sq
            ),
            "pitch_stability_accel_per_speed_sq": jnp.exp(
                self.log_pitch_stability_accel_per_speed_sq
            ),
            "lateral_stability_angular_accel_per_speed_sq": jnp.exp(
                self.log_lateral_stability_angular_accel_per_speed_sq
            ),
            "angular_drag_per_speed": jnp.exp(
                self.log_angular_drag_per_speed
            ),
            "actuator_time_constant": jnp.exp(
                self.log_actuator_time_constant
            ),
            "surface_trim": jnp.tanh(self.surface_trim_unconstrained),
            "flap_lift_accel_per_speed_sq": jnp.exp(
                self.log_flap_lift_accel_per_speed_sq
            ),
            "flap_drag_accel_per_speed_sq": jnp.exp(
                self.log_flap_drag_accel_per_speed_sq
            ),
            "flap_pitch_angular_accel_per_speed_sq": (
                self.flap_pitch_angular_accel_per_speed_sq
            ),
            "flap_trim": jnp.tanh(self.flap_trim_unconstrained),
        }


BaseDynamicsParams = DynamicsParams | FixedWingDynamicsParams


class ResidualDynamicsParams(NamedTuple):
    """A structured vehicle model plus a frame-invariant acceleration residual.

    The network sees body-frame velocity, body angular velocity, and the
    canonical applied-control channels, and optional typed exogenous
    observations. It can therefore wrap any registered rigid-body vehicle
    family without learning position or attitude kinematics.
    Feature normalization and correction bounds are stored with the fitted
    model rather than being tied to one platform's expected operating range.
    """

    base: BaseDynamicsParams
    hidden_weights: Array
    hidden_bias: Array
    output_weights: Array
    feature_mean: Array
    feature_scale: Array
    correction_scale: Array


ModelParams = BaseDynamicsParams | ResidualDynamicsParams


def model_family(params: ModelParams) -> DynamicsModelFamily:
    """Return the static vehicle-family contract for a parameter tree."""

    base = structured_parameters(params)
    return (
        FIXED_WING_FAMILY
        if isinstance(base, FixedWingDynamicsParams)
        else MULTIROTOR_FAMILY
    )


def validate_control_schema(
    params: ModelParams,
    control_names: tuple[str, ...],
    control_roles: tuple[str, ...] | None = None,
) -> None:
    """Validate channel count, names, and order for a model family."""

    model_family(params).validate_control_schema(
        control_names,
        control_roles,
    )


def _response_time_constant(params: ModelParams) -> Array:
    base = structured_parameters(params)
    if isinstance(base, FixedWingDynamicsParams):
        return jnp.exp(base.log_actuator_time_constant)
    return jnp.exp(base.log_motor_time_constant)


def _angular_control_target(params: ModelParams, applied_control: Array) -> Array:
    """Return the multirotor control-generated angular acceleration target."""

    physical = physics_parameters(params).physical()
    return physical["angular_control_matrix"] @ (MOTOR_MIXER @ applied_control)


def _initial_latent_state(params: ModelParams, applied_control: Array) -> Array:
    """Return a steady latent state for one observed applied-control vector."""

    if isinstance(structured_parameters(params), FixedWingDynamicsParams):
        return applied_control
    return jnp.concatenate(
        (applied_control, _angular_control_target(params, applied_control))
    )


def _split_latent_state(
    params: ModelParams,
    latent_state: Array,
    control_size: int,
) -> tuple[Array, Array | None]:
    """Split canonical applied controls from optional multirotor torque state."""

    base = structured_parameters(params)
    if isinstance(base, FixedWingDynamicsParams):
        if latent_state.shape[-1] != control_size:
            raise ValueError(
                "fixed-wing latent state must contain one applied value per control"
            )
        return latent_state, None
    if latent_state.shape[-1] == control_size:
        applied_control = latent_state
        return applied_control, _angular_control_target(params, applied_control)
    expected_size = control_size + MULTIROTOR_ROTATIONAL_STATE_SIZE
    if latent_state.shape[-1] != expected_size:
        raise ValueError(
            "multirotor latent state must contain applied controls and three "
            f"rotational-response states; expected {expected_size}, got "
            f"{latent_state.shape[-1]}"
        )
    return latent_state[:control_size], latent_state[control_size:]


def _angular_response_at(
    params: ModelParams,
    initial_applied_control: Array,
    initial_angular_response: Array,
    commanded_control: Array,
    time_s: float,
) -> Array:
    """Analytically propagate cascaded control and rotational response lags."""

    physical = physics_parameters(params).physical()
    motor_time_constant = physical["motor_time_constant"]
    response_time_constant = physical["angular_response_time_constant"]
    initial_target = _angular_control_target(params, initial_applied_control)
    commanded_target = _angular_control_target(params, commanded_control)
    motor_decay = jnp.exp(-time_s / motor_time_constant)
    response_decay = jnp.exp(-time_s / response_time_constant)
    denominator = motor_time_constant - response_time_constant
    close_time_constants = (
        jnp.abs(denominator)
        <= 1e-6 * jnp.maximum(motor_time_constant, response_time_constant)
    )
    safe_denominator = jnp.where(
        close_time_constants,
        jnp.ones_like(denominator),
        denominator,
    )
    distinct_factor = (
        motor_time_constant
        / safe_denominator
        * (motor_decay - response_decay)
    )
    equal_factor = (
        time_s / response_time_constant * response_decay
    )
    forcing_factor = jnp.where(
        close_time_constants,
        equal_factor,
        distinct_factor,
    )
    lagged_response = (
        commanded_target
        + (initial_angular_response - commanded_target) * response_decay
        + (initial_target - commanded_target) * forcing_factor
    )
    instantaneous_target = _angular_control_target(
        params,
        commanded_control
        + (initial_applied_control - commanded_control) * motor_decay,
    )
    # A 0.1 ms value is the serialized sentinel for the exact memoryless
    # reference model. This makes the simpler model a true nested ablation
    # rather than an approximation using another fast latent state.
    instantaneous = response_time_constant <= 1.00001e-4
    return jnp.where(instantaneous, instantaneous_target, lagged_response)


def with_response_time_constant(
    params: ModelParams, response_time_constant_s: float
) -> ModelParams:
    """Return parameters with the family-specific control lag fixed."""

    log_value = jnp.log(jnp.asarray(response_time_constant_s))
    base = structured_parameters(params)
    if isinstance(base, FixedWingDynamicsParams):
        updated = base._replace(log_actuator_time_constant=log_value)
    else:
        updated = base._replace(log_motor_time_constant=log_value)
    if isinstance(params, ResidualDynamicsParams):
        return params._replace(base=updated)
    return updated


def with_thrust_command_offset(
    params: ModelParams, thrust_command_offset: float
) -> ModelParams:
    """Return a multirotor model with one shared collective command offset."""

    if (
        not math.isfinite(thrust_command_offset)
        or abs(thrust_command_offset) >= MAX_THRUST_COMMAND_OFFSET
    ):
        raise ValueError(
            "thrust_command_offset must be finite and strictly within "
            f"{-MAX_THRUST_COMMAND_OFFSET:g} and "
            f"{MAX_THRUST_COMMAND_OFFSET:g}"
        )
    base = structured_parameters(params)
    if isinstance(base, FixedWingDynamicsParams):
        raise TypeError("fixed-wing models do not have a motor command offset")
    updated = base._replace(
        thrust_command_offset_unconstrained=jnp.arctanh(
            jnp.asarray(
                thrust_command_offset / MAX_THRUST_COMMAND_OFFSET
            )
        )
    )
    if isinstance(params, ResidualDynamicsParams):
        return params._replace(base=updated)
    return updated


def zero_response_time_gradient(params: ModelParams) -> ModelParams:
    """Zero only the family-specific response-time leaf in a gradient tree."""

    base = structured_parameters(params)
    if isinstance(base, FixedWingDynamicsParams):
        updated = base._replace(
            log_actuator_time_constant=jnp.zeros_like(
                base.log_actuator_time_constant
            )
        )
    else:
        updated = base._replace(
            log_motor_time_constant=jnp.zeros_like(
                base.log_motor_time_constant
            )
        )
    if isinstance(params, ResidualDynamicsParams):
        return params._replace(base=updated)
    return updated


def zero_thrust_command_offset_gradient(params: ModelParams) -> ModelParams:
    """Freeze the multirotor command offset for physical thrust-proxy inputs."""

    base = structured_parameters(params)
    if isinstance(base, FixedWingDynamicsParams):
        return params
    updated = base._replace(
        thrust_command_offset_unconstrained=jnp.zeros_like(
            base.thrust_command_offset_unconstrained
        )
    )
    if isinstance(params, ResidualDynamicsParams):
        return params._replace(base=updated)
    return updated


def with_instantaneous_rotational_response(params: ModelParams) -> ModelParams:
    """Return a multirotor model with the exact diagonal memoryless torque map."""

    base = structured_parameters(params)
    if isinstance(base, FixedWingDynamicsParams):
        raise TypeError("fixed-wing models do not have multirotor torque response")
    updated = base._replace(
        log_angular_response_time_constant=jnp.log(jnp.full((3,), 1e-4)),
        angular_control_cross_coupling_unconstrained=jnp.zeros((3, 3)),
    )
    if isinstance(params, ResidualDynamicsParams):
        return params._replace(base=updated)
    return updated


def with_diagonal_angular_control(params: ModelParams) -> ModelParams:
    """Return a multirotor model using only the canonical mixer axes."""

    base = structured_parameters(params)
    if isinstance(base, FixedWingDynamicsParams):
        raise TypeError("fixed-wing models do not have a multirotor mixer")
    updated = base._replace(
        angular_control_cross_coupling_unconstrained=jnp.zeros((3, 3))
    )
    if isinstance(params, ResidualDynamicsParams):
        return params._replace(base=updated)
    return updated


def with_constant_angular_rate(params: ModelParams) -> ModelParams:
    """Return a diagnostic model that holds measured body rate constant.

    Translational structured and residual behavior is retained. The rotational
    control, damping, and residual-acceleration terms are disabled so attitude
    evolves only by integrating the rollout's initial measured angular rate.
    """

    base = structured_parameters(params)
    if isinstance(base, FixedWingDynamicsParams):
        raise TypeError("constant-rate diagnostic is currently multirotor-only")
    updated = base._replace(
        log_angular_accel=jnp.log(jnp.full((3,), 1e-9)),
        log_angular_drag=jnp.log(jnp.full((3,), 1e-9)),
        log_angular_response_time_constant=jnp.log(jnp.full((3,), 1e-4)),
        angular_control_cross_coupling_unconstrained=jnp.zeros((3, 3)),
    )
    if not isinstance(params, ResidualDynamicsParams):
        return updated
    return params._replace(
        base=updated,
        output_weights=params.output_weights.at[3:6].set(0.0),
    )


def with_angular_dynamics_authority(
    params: ModelParams,
    authority: float | tuple[float, float, float],
) -> ModelParams:
    """Scale total multirotor angular acceleration by a bounded authority.

    This is a model-selection transform, not a runtime tuning parameter. One
    reproduces the fitted model and zero approaches constant measured body rate,
    while intermediate values retain a conservative fraction of structured and
    residual angular acceleration. Translation is unchanged directly.
    """

    values = np.asarray(authority, dtype=np.float64)
    if values.ndim == 0:
        values = np.full(3, float(values))
    if values.shape != (3,) or not np.all(np.isfinite(values)):
        raise ValueError(
            "angular dynamics authority must be one or three finite values"
        )
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("angular dynamics authority must lie in [0, 1]")

    base = structured_parameters(params)
    if isinstance(base, FixedWingDynamicsParams):
        raise TypeError("angular dynamics authority is currently multirotor-only")
    authority_array = jnp.asarray(values)
    positive_scale = jnp.maximum(authority_array, 1e-9)
    updated = base._replace(
        log_angular_accel=base.log_angular_accel + jnp.log(positive_scale),
        log_angular_drag=base.log_angular_drag + jnp.log(positive_scale),
    )
    if not isinstance(params, ResidualDynamicsParams):
        return updated
    return params._replace(
        base=updated,
        output_weights=params.output_weights.at[3:6].multiply(
            authority_array[:, jnp.newaxis]
        ),
    )


def zero_rotational_response_gradient(params: ModelParams) -> ModelParams:
    """Freeze multirotor rotational-memory and cross-coupling parameters."""

    base = structured_parameters(params)
    if isinstance(base, FixedWingDynamicsParams):
        return params
    updated = base._replace(
        log_angular_response_time_constant=jnp.zeros_like(
            base.log_angular_response_time_constant
        ),
        angular_control_cross_coupling_unconstrained=jnp.zeros_like(
            base.angular_control_cross_coupling_unconstrained
        ),
    )
    if isinstance(params, ResidualDynamicsParams):
        return params._replace(base=updated)
    return updated


def zero_angular_cross_coupling_gradient(params: ModelParams) -> ModelParams:
    """Freeze only multirotor cross-axis control coupling."""

    base = structured_parameters(params)
    if isinstance(base, FixedWingDynamicsParams):
        return params
    updated = base._replace(
        angular_control_cross_coupling_unconstrained=jnp.zeros_like(
            base.angular_control_cross_coupling_unconstrained
        )
    )
    if isinstance(params, ResidualDynamicsParams):
        return params._replace(base=updated)
    return updated


def zero_residual_configuration_gradient(params: ModelParams) -> ModelParams:
    """Keep data-derived residual normalization fixed during optimization."""

    if not isinstance(params, ResidualDynamicsParams):
        return params
    return params._replace(
        feature_mean=jnp.zeros_like(params.feature_mean),
        feature_scale=jnp.zeros_like(params.feature_scale),
        correction_scale=jnp.zeros_like(params.correction_scale),
    )


def require_quadrotor_control_size(control_size: int) -> None:
    """Reject data that cannot be consumed by the quadrotor model family."""

    if control_size != QUADROTOR_CONTROL_SIZE:
        raise ValueError(
            "quadrotor dynamics require exactly "
            f"{QUADROTOR_CONTROL_SIZE} control channels, got {control_size}"
        )


def require_model_control_size(
    params: ModelParams,
    control_size: int,
    control_roles: tuple[str, ...] | None = None,
) -> None:
    """Reject arrays whose final dimension does not match the model family."""

    family = model_family(params)
    if control_roles is not None:
        if len(control_roles) != control_size:
            raise ValueError(
                "control_roles must contain one role per control channel"
            )
        family.validate_control_roles(control_roles)
        return
    if control_size != family.control_size:
        raise ValueError(
            f"{family.key} dynamics require exactly "
            f"{family.control_size} control channels, got {control_size}"
        )


def _resolved_control_roles(
    params: ModelParams,
    control_size: int,
    control_roles: tuple[str, ...] | None,
) -> tuple[str, ...]:
    roles = model_family(params).control_roles if control_roles is None else tuple(control_roles)
    require_model_control_size(params, control_size, roles)
    return roles


def initial_residual_parameters(
    base: BaseDynamicsParams,
    *,
    control_size: int | None = None,
    exogenous_size: int = 0,
    hidden_units: int = 16,
    seed: int = 0,
    feature_mean: Array | None = None,
    feature_scale: Array | None = None,
    correction_scale: Array | None = None,
) -> ResidualDynamicsParams:
    """Return a residual model whose initial predictions equal its base model."""

    if hidden_units < 1:
        raise ValueError("hidden_units must be positive")
    if control_size is None:
        control_size = model_family(base).control_size
    if control_size < 1:
        raise ValueError("control_size must be positive")
    if exogenous_size < 0:
        raise ValueError("exogenous_size cannot be negative")
    feature_size = 6 + control_size + exogenous_size
    if feature_mean is None:
        feature_mean = jnp.zeros(feature_size)
    if feature_scale is None:
        feature_scale = jnp.ones(feature_size)
    if correction_scale is None:
        correction_scale = jnp.ones(6)
    feature_mean = jnp.asarray(feature_mean)
    feature_scale = jnp.asarray(feature_scale)
    correction_scale = jnp.asarray(correction_scale)
    if feature_mean.shape != (feature_size,):
        raise ValueError(
            "feature_mean must match state, control, and exogenous features"
        )
    if feature_scale.shape != (feature_size,):
        raise ValueError(
            "feature_scale must match state, control, and exogenous features"
        )
    if correction_scale.shape != (6,):
        raise ValueError("correction_scale must contain six acceleration bounds")
    if bool(jnp.any(feature_scale <= 0.0)):
        raise ValueError("feature_scale must be positive")
    if bool(jnp.any(correction_scale <= 0.0)):
        raise ValueError("correction_scale must be positive")
    key = jax.random.key(seed)
    hidden_weights = 0.05 * jax.random.normal(
        key, (hidden_units, feature_size)
    )
    return ResidualDynamicsParams(
        base=base,
        hidden_weights=hidden_weights,
        hidden_bias=jnp.zeros(hidden_units),
        output_weights=jnp.zeros((6, hidden_units)),
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        correction_scale=correction_scale,
    )


def structured_parameters(params: ModelParams) -> BaseDynamicsParams:
    """Return the structured base parameter block from any model class."""

    return params.base if isinstance(params, ResidualDynamicsParams) else params


def physics_parameters(params: ModelParams) -> DynamicsParams:
    """Return multirotor physics, rejecting other structured families."""

    base = structured_parameters(params)
    if isinstance(base, FixedWingDynamicsParams):
        raise TypeError("fixed-wing parameters do not contain quadrotor physics")
    return base


def _residual_acceleration(
    params: ResidualDynamicsParams,
    state: Array,
    applied_motor_state: Array,
    exogenous: Array | None = None,
    exogenous_roles: tuple[str, ...] | None = None,
) -> tuple[Array, Array]:
    """Predict body-linear and body-angular acceleration corrections."""

    rotation = quaternion_to_rotation(state[6:10])
    body_velocity = rotation.T @ (
        state[3:6] - _wind_world(exogenous, exogenous_roles)
    )
    exogenous_features = (
        jnp.empty((0,)) if exogenous is None else exogenous
    )
    features = jnp.concatenate(
        (
            body_velocity,
            state[10:13],
            applied_motor_state,
            exogenous_features,
        )
    )
    normalized_features = (features - params.feature_mean) / params.feature_scale
    hidden = jnp.tanh(
        params.hidden_weights @ normalized_features + params.hidden_bias
    )
    if exogenous is None or exogenous.shape[-1] == 0:
        angular_hidden = hidden
    else:
        roles = () if exogenous_roles is None else exogenous_roles
        estimated_wind_mask = jnp.asarray(
            [role.startswith("estimated_wind_") for role in roles]
        )
        angular_exogenous = jnp.where(
            estimated_wind_mask,
            jnp.zeros_like(normalized_features[-exogenous.shape[-1] :]),
            normalized_features[-exogenous.shape[-1] :],
        )
        angular_features = normalized_features.at[
            -exogenous.shape[-1] :
        ].set(angular_exogenous)
        angular_hidden = jnp.tanh(
            params.hidden_weights @ angular_features + params.hidden_bias
        )
    normalized_correction = jnp.concatenate(
        (
            params.output_weights[0:3] @ hidden,
            params.output_weights[3:6] @ angular_hidden,
        )
    )
    correction = params.correction_scale * jnp.tanh(normalized_correction)
    return rotation @ correction[0:3], correction[3:6]


def quaternion_multiply(left: Array, right: Array) -> Array:
    """Multiply WXYZ quaternions."""

    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return jnp.asarray(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ]
    )


def quaternion_to_rotation(quaternion_wxyz: Array) -> Array:
    """Return the body-to-world rotation matrix for a unit quaternion."""

    w, x, y, z = quaternion_wxyz
    return jnp.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ]
    )


def _wind_world(
    exogenous: Array | None,
    exogenous_roles: tuple[str, ...] | None,
) -> Array:
    """Return world-frame NWU wind from optional typed rollout context."""

    if exogenous is None or exogenous.shape[-1] == 0:
        return jnp.zeros(3)
    if exogenous_roles is None or len(exogenous_roles) != exogenous.shape[-1]:
        raise ValueError(
            "exogenous_roles must identify every supplied exogenous channel"
        )
    north = (
        exogenous[exogenous_roles.index("wind_north")]
        if "wind_north" in exogenous_roles
        else jnp.asarray(0.0)
    )
    west = (
        exogenous[exogenous_roles.index("wind_west")]
        if "wind_west" in exogenous_roles
        else jnp.asarray(0.0)
    )
    up = (
        exogenous[exogenous_roles.index("wind_up")]
        if "wind_up" in exogenous_roles
        else jnp.asarray(0.0)
    )
    return jnp.stack((north, west, up))


def state_derivative(
    params: ModelParams,
    state: Array,
    applied_motor_state: Array,
    control_roles: tuple[str, ...] | None = None,
    exogenous: Array | None = None,
    exogenous_roles: tuple[str, ...] | None = None,
    rotational_response_state: Array | None = None,
) -> Array:
    """Calculate the vehicle derivative from latent applied controls."""

    roles = _resolved_control_roles(
        params, applied_motor_state.shape[-1], control_roles
    )
    base = structured_parameters(params)
    if isinstance(base, FixedWingDynamicsParams):
        physical = base.physical()
        velocity = state[3:6]
        quaternion = state[6:10]
        angular_velocity = state[10:13]
        rotation = quaternion_to_rotation(quaternion)
        body_velocity = rotation.T @ (
            velocity - _wind_world(exogenous, exogenous_roles)
        )
        airspeed = jnp.sqrt(jnp.sum(jnp.square(body_velocity)) + 1e-9)
        forward_speed = jnp.maximum(body_velocity[0], 0.0)
        throttle = jnp.clip(
            applied_motor_state[roles.index("throttle")], 0.0, 1.0
        )
        surface_commands = jnp.stack(
            tuple(
                applied_motor_state[roles.index(axis)]
                if axis in roles
                else physical["surface_trim"][index]
                for index, axis in enumerate(("roll", "pitch", "yaw"))
            )
        )
        surface_authority = jnp.asarray(
            [axis in roles for axis in ("roll", "pitch", "yaw")]
        )
        surfaces = (
            surface_commands - physical["surface_trim"]
        ) * surface_authority
        flap = (
            applied_motor_state[roles.index("flap")] - physical["flap_trim"]
            if "flap" in roles
            else jnp.asarray(0.0)
        )
        flap_drag = physical["flap_drag_accel_per_speed_sq"] * (
            jnp.sqrt(flap * flap + 1e-9) - jnp.sqrt(jnp.asarray(1e-9))
        )

        body_acceleration = jnp.asarray(
            [
                physical["thrust_accel"] * throttle,
                -physical["side_force_accel_per_speed"]
                * airspeed
                * body_velocity[1],
                physical["lift_accel_per_speed_sq"] * forward_speed**2
                - physical["lift_alpha_accel_per_speed_sq"]
                * forward_speed
                * body_velocity[2],
            ]
        )
        body_acceleration = body_acceleration.at[2].add(
            physical["flap_lift_accel_per_speed_sq"]
            * forward_speed**2
            * flap
        )
        body_acceleration = body_acceleration - (
            physical["drag_accel_per_speed_sq"] + flap_drag
        ) * airspeed * body_velocity
        world_acceleration = (
            jnp.asarray([0.0, 0.0, -GRAVITY_M_S2])
            + rotation @ body_acceleration
        )
        angular_acceleration = (
            physical["surface_angular_accel_per_speed_sq"]
            * forward_speed**2
            * surfaces
            - physical["angular_drag_per_speed"]
            * airspeed
            * angular_velocity
        )
        angular_acceleration = angular_acceleration + forward_speed**2 * jnp.asarray(
            [
                physical[
                    "lateral_surface_cross_angular_accel_per_speed_sq"
                ][0]
                * surfaces[2],
                0.0,
                physical[
                    "lateral_surface_cross_angular_accel_per_speed_sq"
                ][1]
                * surfaces[0],
            ]
        )
        angular_acceleration = angular_acceleration.at[1].add(
            -physical["pitch_stability_accel_per_speed_sq"]
            * forward_speed
            * body_velocity[2]
            + physical["flap_pitch_angular_accel_per_speed_sq"]
            * forward_speed**2
            * flap
        )
        lateral_stability = (
            physical["lateral_stability_angular_accel_per_speed_sq"]
            * forward_speed
            * body_velocity[1]
        )
        angular_acceleration = angular_acceleration.at[0].add(
            lateral_stability[0]
        )
        angular_acceleration = angular_acceleration.at[2].add(
            lateral_stability[1]
        )
        quaternion_rate = 0.5 * quaternion_multiply(
            quaternion, jnp.concatenate((jnp.zeros(1), angular_velocity))
        )
        derivative = jnp.concatenate(
            (
                velocity,
                world_acceleration,
                quaternion_rate,
                angular_acceleration,
            )
        )
    else:
        require_quadrotor_control_size(applied_motor_state.shape[-1])

        physical = base.physical()
        velocity = state[3:6]
        quaternion = state[6:10]
        angular_velocity = state[10:13]

        effective_motor_thrust = jnp.maximum(
            applied_motor_state - physical["thrust_command_offset"], 0.0
        )
        body_thrust = jnp.asarray(
            [
                0.0,
                0.0,
                physical["thrust_accel"] * jnp.sum(effective_motor_thrust),
            ]
        )
        world_acceleration = (
            jnp.asarray([0.0, 0.0, -GRAVITY_M_S2])
            + quaternion_to_rotation(quaternion) @ body_thrust
            - physical["linear_drag"]
            * (velocity - _wind_world(exogenous, exogenous_roles))
        )

        control_generated_angular_acceleration = (
            _angular_control_target(params, applied_motor_state)
            if rotational_response_state is None
            else rotational_response_state
        )
        angular_acceleration = (
            control_generated_angular_acceleration
            - physical["angular_drag"] * angular_velocity
        )
        quaternion_rate = 0.5 * quaternion_multiply(
            quaternion, jnp.concatenate((jnp.zeros(1), angular_velocity))
        )

        derivative = jnp.concatenate(
            (velocity, world_acceleration, quaternion_rate, angular_acceleration)
        )
    if isinstance(params, ResidualDynamicsParams):
        linear_residual, angular_residual = _residual_acceleration(
            params,
            state,
            applied_motor_state,
            exogenous,
            exogenous_roles,
        )
        derivative = derivative.at[3:6].add(linear_residual)
        derivative = derivative.at[10:13].add(angular_residual)
    return derivative


def _normalized_state(state: Array) -> Array:
    quaternion = state[6:10]
    quaternion = quaternion / jnp.linalg.norm(quaternion)
    return state.at[6:10].set(quaternion)


def step_with_latent(
    params: ModelParams,
    state: Array,
    latent_state: Array,
    control: Array,
    dt_s: float,
    control_roles: tuple[str, ...] | None = None,
    exogenous: Array | None = None,
    exogenous_roles: tuple[str, ...] | None = None,
) -> tuple[Array, Array]:
    """Advance vehicle and latent actuator states with bounded RK4 steps.

    Motor response is integrated analytically for the piecewise-constant input.
    Multirotors additionally carry three learned control-generated angular
    acceleration states. Their analytic first-order response can represent
    slow rotor/aerodynamic torque dynamics without delaying collective thrust.
    This keeps the rollout stable even while optimization explores time constants
    much shorter than the telemetry sample interval. Telemetry intervals above
    25 ms are integrated with deterministic internal substeps so low-rate state
    estimates do not destabilize otherwise unchanged continuous-time dynamics.
    """

    roles = _resolved_control_roles(params, control.shape[-1], control_roles)
    control_size = control.shape[-1]
    applied_control_state, rotational_response_state = _split_latent_state(
        params, latent_state, control_size
    )
    require_model_control_size(params, applied_control_state.shape[-1], roles)

    response_time_constant = _response_time_constant(params)

    def motor_at(time_s: float) -> Array:
        decay = jnp.exp(-time_s / response_time_constant)
        return control + (applied_control_state - control) * decay

    def angular_response_at(time_s: float) -> Array | None:
        if rotational_response_state is None:
            return None
        return _angular_response_at(
            params,
            applied_control_state,
            rotational_response_state,
            control,
            time_s,
        )

    substep_count = max(
        1, math.ceil(dt_s / MAX_INTERNAL_INTEGRATION_STEP_S)
    )
    integration_dt_s = dt_s / substep_count
    half_integration_dt_s = 0.5 * integration_dt_s
    next_vehicle = state
    for index in range(substep_count):
        start_time_s = index * integration_dt_s
        middle_time_s = start_time_s + half_integration_dt_s
        end_time_s = start_time_s + integration_dt_s
        start_motor_state = motor_at(start_time_s)
        middle_motor_state = motor_at(middle_time_s)
        end_motor_state = motor_at(end_time_s)
        start_angular_response = angular_response_at(start_time_s)
        middle_angular_response = angular_response_at(middle_time_s)
        end_angular_response = angular_response_at(end_time_s)
        k1 = state_derivative(
            params,
            next_vehicle,
            start_motor_state,
            roles,
            exogenous,
            exogenous_roles,
            start_angular_response,
        )
        k2 = state_derivative(
            params,
            next_vehicle + half_integration_dt_s * k1,
            middle_motor_state,
            roles,
            exogenous,
            exogenous_roles,
            middle_angular_response,
        )
        k3 = state_derivative(
            params,
            next_vehicle + half_integration_dt_s * k2,
            middle_motor_state,
            roles,
            exogenous,
            exogenous_roles,
            middle_angular_response,
        )
        k4 = state_derivative(
            params,
            next_vehicle + integration_dt_s * k3,
            end_motor_state,
            roles,
            exogenous,
            exogenous_roles,
            end_angular_response,
        )
        next_vehicle = _normalized_state(
            next_vehicle
            + (integration_dt_s / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        )
    next_motor_state = motor_at(dt_s)
    next_angular_response = angular_response_at(dt_s)
    next_latent_state = (
        next_motor_state
        if next_angular_response is None
        else jnp.concatenate((next_motor_state, next_angular_response))
    )
    return next_vehicle, next_latent_state


def step(
    params: ModelParams,
    state: Array,
    control: Array,
    dt_s: float,
    control_roles: tuple[str, ...] | None = None,
    exogenous: Array | None = None,
    exogenous_roles: tuple[str, ...] | None = None,
) -> Array:
    """Advance one step, assuming the initial applied motors equal the command.

    Use :func:`step_with_latent` when stepping repeatedly so motor state is
    carried between calls. This convenience wrapper is useful for equilibrium
    checks and isolated transitions.
    """

    next_state, _ = step_with_latent(
        params,
        state,
        control,
        control,
        dt_s,
        control_roles,
        exogenous,
        exogenous_roles,
    )
    return next_state


def rollout_with_latent(
    params: ModelParams,
    initial_state: Array,
    controls: Array,
    dt_s: float,
    initial_motor_state: Array | None = None,
    control_roles: tuple[str, ...] | None = None,
    exogenous: Array | None = None,
    exogenous_roles: tuple[str, ...] | None = None,
) -> tuple[Array, Array]:
    """Roll out vehicle and latent actuator states.

    When no prior motor state is available, the first recorded command is used
    as the initial applied state. This is exact at steady state and becomes an
    approximation when a telemetry window begins during a fast command change.
    The returned latent trace remains the canonical applied-control trace; the
    multirotor rotational-response state is carried internally.
    """

    if controls.ndim != 2:
        raise ValueError("rollout controls must be two-dimensional")
    roles = _resolved_control_roles(params, controls.shape[-1], control_roles)
    if initial_motor_state is None:
        initial_latent_state = _initial_latent_state(params, controls[0])
    else:
        initial_applied_control, initial_rotational_response = _split_latent_state(
            params, initial_motor_state, controls.shape[-1]
        )
        initial_latent_state = (
            initial_applied_control
            if initial_rotational_response is None
            else jnp.concatenate(
                (initial_applied_control, initial_rotational_response)
            )
        )
    initial_combined = jnp.concatenate((initial_state, initial_latent_state))
    control_size = controls.shape[-1]
    latent_size = initial_latent_state.shape[-1]

    def scan_step(combined: Array, control: Array) -> tuple[Array, Array]:
        next_state, next_latent_state = step_with_latent(
            params,
            combined[:13],
            combined[13 : 13 + latent_size],
            control,
            dt_s,
            roles,
            exogenous,
            exogenous_roles,
        )
        next_combined = jnp.concatenate((next_state, next_latent_state))
        return next_combined, next_combined

    _, combined_states = jax.lax.scan(scan_step, initial_combined, controls)
    combined_states = jnp.concatenate(
        (initial_combined[jnp.newaxis, :], combined_states), axis=0
    )
    return combined_states[:, :13], combined_states[:, 13 : 13 + control_size]


def control_state_after_history(
    params: ModelParams,
    control_history: Array,
    dt_s: float,
    control_roles: tuple[str, ...] | None = None,
) -> Array:
    """Infer the complete actuator state after a nonempty command history.

    The first command is treated as the steady state before the history begins.
    With a history several time constants long, the influence of that boundary
    assumption decays away.
    """

    if control_history.ndim != 2:
        raise ValueError("control history must be two-dimensional")
    _resolved_control_roles(params, control_history.shape[-1], control_roles)

    initial_latent_state = _initial_latent_state(params, control_history[0])
    decay = jnp.exp(-dt_s / _response_time_constant(params))

    def scan_step(latent_state: Array, control: Array) -> tuple[Array, None]:
        applied_control, rotational_response = _split_latent_state(
            params, latent_state, control_history.shape[-1]
        )
        next_applied_control = control + (applied_control - control) * decay
        if rotational_response is None:
            next_latent_state = next_applied_control
        else:
            next_rotational_response = _angular_response_at(
                params,
                applied_control,
                rotational_response,
                control,
                dt_s,
            )
            next_latent_state = jnp.concatenate(
                (next_applied_control, next_rotational_response)
            )
        return next_latent_state, None

    final_latent_state, _ = jax.lax.scan(
        scan_step, initial_latent_state, control_history
    )
    return final_latent_state


def rollout(
    params: ModelParams,
    initial_state: Array,
    controls: Array,
    dt_s: float,
    control_roles: tuple[str, ...] | None = None,
    exogenous: Array | None = None,
    exogenous_roles: tuple[str, ...] | None = None,
) -> Array:
    """Roll out observed vehicle states while carrying latent motor response."""

    states, _ = rollout_with_latent(
        params,
        initial_state,
        controls,
        dt_s,
        control_roles=control_roles,
        exogenous=exogenous,
        exogenous_roles=exogenous_roles,
    )
    return states


def hover_control(params: ModelParams) -> Array:
    """Return equal motor commands that balance gravity at level attitude."""

    if isinstance(params, FixedWingDynamicsParams):
        raise TypeError("fixed-wing models do not have a hover control")
    motor_command = GRAVITY_M_S2 / (
        4.0 * jnp.exp(physics_parameters(params).log_thrust_accel)
    )
    motor_command += physics_parameters(params).physical()[
        "thrust_command_offset"
    ]
    return jnp.full((QUADROTOR_CONTROL_SIZE,), motor_command)


def fixed_wing_trim_control(
    params: FixedWingDynamicsParams,
    airspeed_m_s: float,
    control_roles: tuple[str, ...] | None = None,
) -> Array:
    """Return level-flight throttle/surface commands for a supplied airspeed."""

    physical = params.physical()
    trim_throttle = (
        physical["drag_accel_per_speed_sq"] * airspeed_m_s**2
        / physical["thrust_accel"]
    )
    roles = FIXED_WING_CONTROL_ROLES if control_roles is None else control_roles
    FIXED_WING_FAMILY.validate_control_roles(tuple(roles))
    values = {
        "throttle": trim_throttle,
        "roll": physical["surface_trim"][0],
        "pitch": physical["surface_trim"][1],
        "yaw": physical["surface_trim"][2],
        "flap": physical["flap_trim"],
    }
    return jnp.stack(tuple(values[role] for role in roles))
