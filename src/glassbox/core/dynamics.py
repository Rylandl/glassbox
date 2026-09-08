"""Compact differentiable multirotor, fixed-wing and bootstrap families."""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.flatten_util import ravel_pytree

from glassbox.core.families import (
    BOOTSTRAP_MULTIROTOR_FAMILY,
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


def _require_positive(name: str, value: object) -> None:
    values = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError(f"{name} must be finite and strictly positive")


def _require_finite(name: str, value: object) -> None:
    if not np.all(np.isfinite(np.asarray(value, dtype=np.float64))):
        raise ValueError(f"{name} must be finite")


def _require_open_unit_interval(name: str, value: object) -> None:
    values = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(np.abs(values) >= 1.0):
        raise ValueError(f"{name} must lie strictly within (-1, 1)")


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
        angular_control_cross_coupling: tuple[
            tuple[float, float, float],
            tuple[float, float, float],
            tuple[float, float, float],
        ] = (
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
        ),
    ) -> DynamicsParams:
        if (
            not math.isfinite(thrust_command_offset)
            or abs(thrust_command_offset) >= MAX_THRUST_COMMAND_OFFSET
        ):
            raise ValueError(
                "thrust_command_offset must be finite and strictly within "
                f"{-MAX_THRUST_COMMAND_OFFSET:g} and "
                f"{MAX_THRUST_COMMAND_OFFSET:g}"
            )
        _require_positive("thrust_accel", thrust_accel)
        _require_positive("angular_accel", angular_accel)
        _require_positive("linear_drag", linear_drag)
        _require_positive("angular_drag", angular_drag)
        _require_positive("motor_time_constant", motor_time_constant)
        cross_coupling = jnp.asarray(angular_control_cross_coupling)
        if cross_coupling.shape != (3, 3):
            raise ValueError("angular_control_cross_coupling must have shape (3, 3)")
        cross_coupling = cross_coupling.at[jnp.diag_indices(3)].set(0.0)
        normalized_cross_coupling = cross_coupling / MAX_ANGULAR_CONTROL_CROSS_COUPLING
        _require_open_unit_interval(
            f"angular_control_cross_coupling / {MAX_ANGULAR_CONTROL_CROSS_COUPLING:g}",
            normalized_cross_coupling,
        )
        return cls(
            log_thrust_accel=jnp.log(jnp.asarray(thrust_accel)),
            thrust_command_offset_unconstrained=jnp.arctanh(
                jnp.asarray(thrust_command_offset / MAX_THRUST_COMMAND_OFFSET)
            ),
            log_angular_accel=jnp.log(jnp.asarray(angular_accel)),
            log_linear_drag=jnp.log(jnp.asarray(linear_drag)),
            log_angular_drag=jnp.log(jnp.asarray(angular_drag)),
            log_motor_time_constant=jnp.log(jnp.asarray(motor_time_constant)),
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
            "thrust_command_offset": MAX_THRUST_COMMAND_OFFSET
            * jnp.tanh(self.thrust_command_offset_unconstrained),
            "angular_accel": angular_accel,
            "linear_drag": jnp.exp(self.log_linear_drag),
            "angular_drag": jnp.exp(self.log_angular_drag),
            "motor_time_constant": jnp.exp(self.log_motor_time_constant),
            "angular_control_cross_coupling": cross_coupling,
            "angular_control_matrix": jnp.diag(angular_accel)
            @ (jnp.eye(3) + cross_coupling),
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
    ) -> FixedWingDynamicsParams:
        for name, value in (
            ("thrust_accel", thrust_accel),
            ("lift_accel_per_speed_sq", lift_accel_per_speed_sq),
            ("lift_alpha_accel_per_speed_sq", lift_alpha_accel_per_speed_sq),
            ("drag_accel_per_speed_sq", drag_accel_per_speed_sq),
            ("side_force_accel_per_speed", side_force_accel_per_speed),
            ("surface_angular_accel_per_speed_sq", surface_angular_accel_per_speed_sq),
            ("pitch_stability_accel_per_speed_sq", pitch_stability_accel_per_speed_sq),
            (
                "lateral_stability_angular_accel_per_speed_sq",
                lateral_stability_angular_accel_per_speed_sq,
            ),
            ("angular_drag_per_speed", angular_drag_per_speed),
            ("actuator_time_constant", actuator_time_constant),
            ("flap_lift_accel_per_speed_sq", flap_lift_accel_per_speed_sq),
            ("flap_drag_accel_per_speed_sq", flap_drag_accel_per_speed_sq),
        ):
            _require_positive(name, value)
        _require_finite(
            "lateral_surface_cross_angular_accel_per_speed_sq",
            lateral_surface_cross_angular_accel_per_speed_sq,
        )
        _require_finite(
            "flap_pitch_angular_accel_per_speed_sq",
            flap_pitch_angular_accel_per_speed_sq,
        )
        _require_open_unit_interval("surface_trim", surface_trim)
        _require_open_unit_interval("flap_trim", flap_trim)
        return cls(
            log_thrust_accel=jnp.log(jnp.asarray(thrust_accel)),
            log_lift_accel_per_speed_sq=jnp.log(jnp.asarray(lift_accel_per_speed_sq)),
            log_lift_alpha_accel_per_speed_sq=jnp.log(
                jnp.asarray(lift_alpha_accel_per_speed_sq)
            ),
            log_drag_accel_per_speed_sq=jnp.log(jnp.asarray(drag_accel_per_speed_sq)),
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
            log_angular_drag_per_speed=jnp.log(jnp.asarray(angular_drag_per_speed)),
            log_actuator_time_constant=jnp.log(jnp.asarray(actuator_time_constant)),
            surface_trim_unconstrained=jnp.arctanh(jnp.asarray(surface_trim)),
            log_flap_lift_accel_per_speed_sq=jnp.log(
                jnp.asarray(flap_lift_accel_per_speed_sq)
            ),
            log_flap_drag_accel_per_speed_sq=jnp.log(
                jnp.asarray(flap_drag_accel_per_speed_sq)
            ),
            flap_pitch_angular_accel_per_speed_sq=jnp.asarray(
                flap_pitch_angular_accel_per_speed_sq
            ),
            flap_trim_unconstrained=jnp.arctanh(jnp.asarray(flap_trim)),
        )

    def physical(self) -> dict[str, Array]:
        return {
            "thrust_accel": jnp.exp(self.log_thrust_accel),
            "lift_accel_per_speed_sq": jnp.exp(self.log_lift_accel_per_speed_sq),
            "lift_alpha_accel_per_speed_sq": jnp.exp(
                self.log_lift_alpha_accel_per_speed_sq
            ),
            "drag_accel_per_speed_sq": jnp.exp(self.log_drag_accel_per_speed_sq),
            "side_force_accel_per_speed": jnp.exp(self.log_side_force_accel_per_speed),
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
            "angular_drag_per_speed": jnp.exp(self.log_angular_drag_per_speed),
            "actuator_time_constant": jnp.exp(self.log_actuator_time_constant),
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


class BootstrapMultirotorParams(NamedTuple):
    """The bootstrap parameterization: direct command effects, no airframe.

    This is what an in-flight identifier can learn about a multirotor it has
    never seen, and nothing more. Body-``z`` specific force is affine in the
    motor command and the body velocity; body angular acceleration is affine
    in the motor command, the body rate, and the three body-rate products. No
    mixer, mass, inertia, arm length, or thrust coefficient appears, and there
    is no actuator lag, because the identifier regresses on the applied
    command it measured rather than on a requested one.

    Every coefficient is stated in the raw command units of the box the
    vehicle is flown in, so the parameters are directly readable and the
    structured parameter vector needs no rescaling. The order of the fields is
    the order of :func:`structured_parameter_names`, and it is a contract: an
    information state accumulated over these coordinates is stated in it.
    """

    #: Body-``z`` specific force per unit of each motor command, m/s^2.
    collective_acceleration_per_command: Array
    #: Body-``z`` specific force per unit of body velocity, 1/s.
    collective_velocity_coefficient: Array
    #: Body-``z`` specific force at zero command and zero velocity, m/s^2.
    collective_intercept_m_s2: Array
    #: Body angular acceleration per unit of each motor command, rad/s^2.
    angular_acceleration_per_command: Array
    #: Body angular acceleration per unit of body rate, 1/s.
    angular_rate_coefficient: Array
    #: Body angular acceleration per unit of the three body-rate products, 1/s.
    angular_rate_product_coefficient: Array
    #: Body angular acceleration at zero command and zero rate, rad/s^2.
    angular_intercept_rad_s2: Array

    def hover_command(self) -> Array:
        """Return the equal motor command this map says holds a level hover.

        It is a derived quantity rather than a parameter: the collective map
        and its intercept determine it. It is finite only when the four
        command effects sum to something positive, and whether it lies inside
        the command box is a question for the evidence that produced the map,
        not for the map itself. Like every other model computation it is
        evaluated in the execution precision, so a caller that needs the
        estimator's own double-precision value reads it from the evidence that
        produced the map.
        """

        collective_sum = jnp.sum(self.collective_acceleration_per_command)
        return jnp.full(
            (QUADROTOR_CONTROL_SIZE,),
            (GRAVITY_M_S2 - self.collective_intercept_m_s2) / collective_sum,
        )


BaseDynamicsParams = (
    DynamicsParams | FixedWingDynamicsParams | BootstrapMultirotorParams
)


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
    if isinstance(base, FixedWingDynamicsParams):
        return FIXED_WING_FAMILY
    if isinstance(base, BootstrapMultirotorParams):
        return BOOTSTRAP_MULTIROTOR_FAMILY
    return MULTIROTOR_FAMILY


def models_actuator_lag(params: ModelParams) -> bool:
    """Whether this family carries a first-order latent actuator response.

    The bootstrap parameterization does not: its latent applied command is the
    command itself, because the identifier that produces it regresses on the
    applied command it measured. Every fitted family does.
    """

    return not isinstance(structured_parameters(params), BootstrapMultirotorParams)


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
    if isinstance(base, BootstrapMultirotorParams):
        raise TypeError("the bootstrap parameterization fits no actuator lag")
    return jnp.exp(base.log_motor_time_constant)


def latent_response_time_constants(params: ModelParams) -> Array:
    """Return every fitted first-order latent-response time constant in seconds.

    A family that models no actuator lag returns an empty array rather than a
    zero, because it fitted no time constant at all.
    """

    if not models_actuator_lag(params):
        return jnp.zeros(0)
    return jnp.atleast_1d(_response_time_constant(params))


def _angular_control_target(params: ModelParams, applied_control: Array) -> Array:
    """Return the multirotor control-generated angular acceleration target."""

    physical = physics_parameters(params).physical()
    return physical["angular_control_matrix"] @ (MOTOR_MIXER @ applied_control)


def _validated_latent_state(latent_state: Array, control_size: int) -> Array:
    """Reject a latent state that is not one applied value per control channel."""

    if latent_state.shape[-1] != control_size:
        raise ValueError(
            "the latent state must contain one applied value per control channel; "
            f"expected {control_size}, got {latent_state.shape[-1]}"
        )
    return latent_state


def with_response_time_constant(
    params: ModelParams, response_time_constant_s: float
) -> ModelParams:
    """Return parameters with the family-specific control lag fixed."""

    log_value = jnp.log(jnp.asarray(response_time_constant_s))
    base = structured_parameters(params)
    if isinstance(base, BootstrapMultirotorParams):
        raise TypeError("the bootstrap parameterization fits no actuator lag")
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
    if not isinstance(base, DynamicsParams):
        raise TypeError("only the fitted multirotor model has a command offset")
    updated = base._replace(
        thrust_command_offset_unconstrained=jnp.arctanh(
            jnp.asarray(thrust_command_offset / MAX_THRUST_COMMAND_OFFSET)
        )
    )
    if isinstance(params, ResidualDynamicsParams):
        return params._replace(base=updated)
    return updated


def zero_response_time_gradient(params: ModelParams) -> ModelParams:
    """Zero only the family-specific response-time leaf in a gradient tree."""

    base = structured_parameters(params)
    if isinstance(base, BootstrapMultirotorParams):
        return params
    if isinstance(base, FixedWingDynamicsParams):
        updated = base._replace(
            log_actuator_time_constant=jnp.zeros_like(base.log_actuator_time_constant)
        )
    else:
        updated = base._replace(
            log_motor_time_constant=jnp.zeros_like(base.log_motor_time_constant)
        )
    if isinstance(params, ResidualDynamicsParams):
        return params._replace(base=updated)
    return updated


def zero_thrust_command_offset_gradient(params: ModelParams) -> ModelParams:
    """Freeze the multirotor command offset for physical thrust-proxy inputs."""

    base = structured_parameters(params)
    if not isinstance(base, DynamicsParams):
        return params
    updated = base._replace(
        thrust_command_offset_unconstrained=jnp.zeros_like(
            base.thrust_command_offset_unconstrained
        )
    )
    if isinstance(params, ResidualDynamicsParams):
        return params._replace(base=updated)
    return updated


def with_diagonal_angular_control(params: ModelParams) -> ModelParams:
    """Return a multirotor model using only the canonical mixer axes."""

    base = structured_parameters(params)
    if not isinstance(base, DynamicsParams):
        raise TypeError("only the fitted multirotor model has a mixer")
    updated = base._replace(
        angular_control_cross_coupling_unconstrained=jnp.zeros((3, 3))
    )
    if isinstance(params, ResidualDynamicsParams):
        return params._replace(base=updated)
    return updated


def zero_angular_cross_coupling_gradient(params: ModelParams) -> ModelParams:
    """Freeze only multirotor cross-axis control coupling."""

    base = structured_parameters(params)
    if not isinstance(base, DynamicsParams):
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
            raise ValueError("control_roles must contain one role per control channel")
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
    roles = (
        model_family(params).control_roles
        if control_roles is None
        else tuple(control_roles)
    )
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
    hidden_weights = 0.05 * jax.random.normal(key, (hidden_units, feature_size))
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


def structured_parameter_names(params: ModelParams) -> tuple[str, ...]:
    """Return stable scalar names in JAX's structured-parameter leaf order."""

    base = structured_parameters(params)
    names: list[str] = []
    for field_name, value in base._asdict().items():
        array = np.asarray(value)
        if array.ndim == 0:
            names.append(field_name)
            continue
        names.extend(
            f"{field_name}[{','.join(str(index) for index in location)}]"
            for location in np.ndindex(array.shape)
        )
    return tuple(names)


def structured_parameter_vector(params: ModelParams) -> Array:
    """Flatten only the interpretable structured coefficient block."""

    vector, _ = ravel_pytree(structured_parameters(params))
    return vector


def with_structured_parameter_vector(params: ModelParams, vector: Array) -> ModelParams:
    """Replace the structured block while leaving any residual network fixed."""

    expected, unravel = ravel_pytree(structured_parameters(params))
    vector = jnp.asarray(vector)
    if vector.shape != expected.shape:
        raise ValueError(
            f"structured parameter vector has shape {vector.shape}, "
            f"expected {expected.shape}"
        )
    updated_base = unravel(vector)
    return (
        params._replace(base=updated_base)
        if isinstance(params, ResidualDynamicsParams)
        else updated_base
    )


def physics_parameters(params: ModelParams) -> DynamicsParams:
    """Return multirotor physics, rejecting other structured families."""

    base = structured_parameters(params)
    if not isinstance(base, DynamicsParams):
        raise TypeError("only the fitted multirotor model carries airframe physics")
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
    body_velocity = rotation.T @ (state[3:6] - _wind_world(exogenous, exogenous_roles))
    exogenous_features = jnp.empty((0,)) if exogenous is None else exogenous
    features = jnp.concatenate(
        (
            body_velocity,
            state[10:13],
            applied_motor_state,
            exogenous_features,
        )
    )
    normalized_features = (features - params.feature_mean) / params.feature_scale
    hidden = jnp.tanh(params.hidden_weights @ normalized_features + params.hidden_bias)
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
        angular_features = normalized_features.at[-exogenous.shape[-1] :].set(
            angular_exogenous
        )
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
        throttle = jnp.clip(applied_motor_state[roles.index("throttle")], 0.0, 1.0)
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
        surfaces = (surface_commands - physical["surface_trim"]) * surface_authority
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
                -physical["side_force_accel_per_speed"] * airspeed * body_velocity[1],
                physical["lift_accel_per_speed_sq"] * forward_speed**2
                - physical["lift_alpha_accel_per_speed_sq"]
                * forward_speed
                * body_velocity[2],
            ]
        )
        body_acceleration = body_acceleration.at[2].add(
            physical["flap_lift_accel_per_speed_sq"] * forward_speed**2 * flap
        )
        body_acceleration = (
            body_acceleration
            - (physical["drag_accel_per_speed_sq"] + flap_drag)
            * airspeed
            * body_velocity
        )
        world_acceleration = (
            jnp.asarray([0.0, 0.0, -GRAVITY_M_S2]) + rotation @ body_acceleration
        )
        angular_acceleration = (
            physical["surface_angular_accel_per_speed_sq"] * forward_speed**2 * surfaces
            - physical["angular_drag_per_speed"] * airspeed * angular_velocity
        )
        angular_acceleration = angular_acceleration + forward_speed**2 * jnp.asarray(
            [
                physical["lateral_surface_cross_angular_accel_per_speed_sq"][0]
                * surfaces[2],
                0.0,
                physical["lateral_surface_cross_angular_accel_per_speed_sq"][1]
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
        angular_acceleration = angular_acceleration.at[0].add(lateral_stability[0])
        angular_acceleration = angular_acceleration.at[2].add(lateral_stability[1])
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
    elif isinstance(base, BootstrapMultirotorParams):
        require_quadrotor_control_size(applied_motor_state.shape[-1])

        velocity = state[3:6]
        quaternion = state[6:10]
        angular_velocity = state[10:13]
        rotation = quaternion_to_rotation(quaternion)
        body_velocity = rotation.T @ velocity
        # The identifier explains the body-z specific force and nothing else,
        # so the other two body axes carry exactly zero rather than an
        # invented coefficient.
        specific_force_z = (
            base.collective_acceleration_per_command @ applied_motor_state
            + base.collective_velocity_coefficient @ body_velocity
            + base.collective_intercept_m_s2
        )
        body_specific_force = jnp.stack(
            (jnp.zeros(()), jnp.zeros(()), specific_force_z)
        )
        world_acceleration = (
            jnp.asarray([0.0, 0.0, -GRAVITY_M_S2]) + rotation @ body_specific_force
        )
        rate_products = jnp.stack(
            (
                angular_velocity[0] * angular_velocity[1],
                angular_velocity[0] * angular_velocity[2],
                angular_velocity[1] * angular_velocity[2],
            )
        )
        angular_acceleration = (
            base.angular_acceleration_per_command @ applied_motor_state
            + base.angular_rate_coefficient @ angular_velocity
            + base.angular_rate_product_coefficient @ rate_products
            + base.angular_intercept_rad_s2
        )
        quaternion_rate = 0.5 * quaternion_multiply(
            quaternion, jnp.concatenate((jnp.zeros(1), angular_velocity))
        )
        derivative = jnp.concatenate(
            (velocity, world_acceleration, quaternion_rate, angular_acceleration)
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

        angular_acceleration = (
            _angular_control_target(params, applied_motor_state)
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


def _actuator_quadrature_decay(ratio: Array) -> tuple[Array, Array, Array]:
    """Fit RK stage controls to the first two exponential response moments.

    For x=h/tau, A=int_0^1 exp(-xs) ds and B=int_0^1 (1-s)exp(-xs) ds.
    Keeping the middle-stage decay m=exp(-x/2), the stage decays a,b satisfy
    (a+4m+b)/6=A and (a+2m)/6=B. Thus affine actuator forcing gives the exact
    velocity AND position increment, including the instantaneous-response
    limit. Small-x series avoid cancellation. At fixed tau the endpoint
    changes are opposite O(h^3), preserving the RK4 order for smooth forces.
    """

    small = jnp.minimum(ratio, 0.5)
    large = jnp.maximum(ratio, 0.5)
    mean = jnp.where(
        ratio < 0.5,
        1
        - small / 2
        + small**2 / 6
        - small**3 / 24
        + small**4 / 120
        - small**5 / 720
        + small**6 / 5040
        - small**7 / 40320
        + small**8 / 362880,
        -jnp.expm1(-large) / large,
    )
    first_moment = jnp.where(
        ratio < 0.5,
        0.5
        - small / 6
        + small**2 / 24
        - small**3 / 120
        + small**4 / 720
        - small**5 / 5040
        + small**6 / 40320
        - small**7 / 362880
        + small**8 / 3628800,
        (1 - mean) / large,
    )
    middle = jnp.exp(-ratio / 2)
    return (
        jnp.clip(6 * first_moment - 2 * middle, 0.0, 1.0),
        middle,
        jnp.clip(6 * (mean - first_moment) - 2 * middle, 0.0, 1.0),
    )


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
    """Advance vehicle and actuator states with exponential-fitted RK4 steps.

    The latent state is the applied-control vector, one value per control
    channel. Actuator response is integrated analytically for the
    piecewise-constant input. RK stage controls match its first two integral
    moments, so a fast actuator cannot retain a spurious old-command impulse.
    The control-generated torque follows that
    applied control with no memory of its own. This keeps the rollout stable
    even while optimization explores time constants much shorter than the
    telemetry sample interval. Telemetry intervals above 25 ms are integrated
    with deterministic internal substeps so low-rate state estimates do not
    destabilize otherwise unchanged continuous-time dynamics.
    """

    roles = _resolved_control_roles(params, control.shape[-1], control_roles)
    control_size = control.shape[-1]
    applied_control_state = _validated_latent_state(latent_state, control_size)
    require_model_control_size(params, applied_control_state.shape[-1], roles)

    if models_actuator_lag(params):
        response_time_constant = _response_time_constant(params)

        def motor_at(time_s: float) -> Array:
            decay = jnp.exp(-time_s / response_time_constant)
            return control + (applied_control_state - control) * decay
    else:

        def motor_at(time_s: float) -> Array:
            del time_s
            return control

    substep_count = max(1, math.ceil(dt_s / MAX_INTERNAL_INTEGRATION_STEP_S))
    integration_dt_s = dt_s / substep_count
    half_integration_dt_s = 0.5 * integration_dt_s
    if models_actuator_lag(params):
        decay_start, decay_middle, decay_end = _actuator_quadrature_decay(
            integration_dt_s / response_time_constant
        )
    next_vehicle = state
    for index in range(substep_count):
        start_time_s = index * integration_dt_s
        if models_actuator_lag(params):
            amplitude = (applied_control_state - control) * jnp.exp(
                -start_time_s / response_time_constant
            )
            start_motor_state = control + amplitude * decay_start
            middle_motor_state = control + amplitude * decay_middle
            end_motor_state = control + amplitude * decay_end
        else:
            start_motor_state = middle_motor_state = end_motor_state = control
        k1 = state_derivative(
            params,
            next_vehicle,
            start_motor_state,
            roles,
            exogenous,
            exogenous_roles,
        )
        k2 = state_derivative(
            params,
            next_vehicle + half_integration_dt_s * k1,
            middle_motor_state,
            roles,
            exogenous,
            exogenous_roles,
        )
        k3 = state_derivative(
            params,
            next_vehicle + half_integration_dt_s * k2,
            middle_motor_state,
            roles,
            exogenous,
            exogenous_roles,
        )
        k4 = state_derivative(
            params,
            next_vehicle + integration_dt_s * k3,
            end_motor_state,
            roles,
            exogenous,
            exogenous_roles,
        )
        next_vehicle = _normalized_state(
            next_vehicle + (integration_dt_s / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        )
    return next_vehicle, motor_at(dt_s)


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
    The returned latent trace is the canonical applied-control trace.

    ``exogenous`` may be one vector, held for every step, or one row per control
    step so that logged wind or other context varies along the rollout.
    """

    if controls.ndim != 2:
        raise ValueError("rollout controls must be two-dimensional")
    roles = _resolved_control_roles(params, controls.shape[-1], control_roles)
    control_size = controls.shape[-1]
    initial_latent_state = (
        controls[0]
        if initial_motor_state is None
        else _validated_latent_state(initial_motor_state, control_size)
    )
    initial_combined = jnp.concatenate((initial_state, initial_latent_state))
    per_step_exogenous = _per_step_exogenous(exogenous, controls.shape[0])

    def scan_step(
        combined: Array, inputs: tuple[Array, Array | None]
    ) -> tuple[Array, Array]:
        control, step_exogenous = inputs
        next_state, next_latent_state = step_with_latent(
            params,
            combined[:13],
            combined[13 : 13 + control_size],
            control,
            dt_s,
            roles,
            step_exogenous,
            exogenous_roles,
        )
        next_combined = jnp.concatenate((next_state, next_latent_state))
        return next_combined, next_combined

    _, combined_states = jax.lax.scan(
        scan_step, initial_combined, (controls, per_step_exogenous)
    )
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

    return control_state_trace(params, control_history, dt_s, control_roles)[-1]


def control_state_trace(
    params: ModelParams,
    controls: Array,
    dt_s: float,
    control_roles: tuple[str, ...] | None = None,
    initial_state: Array | None = None,
) -> Array:
    """Return applied-control states at every command boundary, including zero."""

    if controls.ndim != 2 or controls.shape[0] == 0:
        raise ValueError("control history must be a nonempty two-dimensional array")
    _resolved_control_roles(params, controls.shape[-1], control_roles)
    initial = (
        controls[0]
        if initial_state is None
        else _validated_latent_state(initial_state, controls.shape[-1])
    )
    if not models_actuator_lag(params):
        return jnp.concatenate((initial[jnp.newaxis, :], controls))

    decay = jnp.exp(-dt_s / _response_time_constant(params))

    def scan_step(applied_control: Array, control: Array) -> tuple[Array, Array]:
        applied = control + (applied_control - control) * decay
        return applied, applied

    _, trace = jax.lax.scan(scan_step, initial, controls)
    return jnp.concatenate((initial[jnp.newaxis, :], trace))


def _per_step_exogenous(exogenous: Array | None, step_count: int) -> Array | None:
    """Broadcast one exogenous vector, or validate one row per control step."""

    if exogenous is None:
        return None
    values = jnp.asarray(exogenous)
    if values.ndim == 1:
        return jnp.broadcast_to(values, (step_count, values.shape[0]))
    if values.ndim == 2:
        if values.shape[0] != step_count:
            raise ValueError(
                "per-step exogenous inputs need exactly one row per control step"
            )
        return values
    raise ValueError("exogenous inputs must be one vector or one row per step")


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

    base = structured_parameters(params)
    if isinstance(base, BootstrapMultirotorParams):
        return base.hover_command()
    if isinstance(base, FixedWingDynamicsParams):
        raise TypeError("fixed-wing models do not have a hover control")
    motor_command = GRAVITY_M_S2 / (
        4.0 * jnp.exp(physics_parameters(params).log_thrust_accel)
    )
    motor_command += physics_parameters(params).physical()["thrust_command_offset"]
    return jnp.full((QUADROTOR_CONTROL_SIZE,), motor_command)


def fixed_wing_trim_control(
    params: FixedWingDynamicsParams,
    airspeed_m_s: float,
    control_roles: tuple[str, ...] | None = None,
) -> Array:
    """Return level-flight throttle/surface commands for a supplied airspeed."""

    physical = params.physical()
    trim_throttle = (
        physical["drag_accel_per_speed_sq"] * airspeed_m_s**2 / physical["thrust_accel"]
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
