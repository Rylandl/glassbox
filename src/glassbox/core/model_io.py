"""Serialization for fitted differentiable dynamics models."""

from __future__ import annotations

import json
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from glassbox.core.data import (
    NORMALIZED_MOTOR_COMMAND_SEMANTICS,
    PHYSICAL_MOTOR_THRUST_SEMANTICS,
    TrajectorySpec,
)
from glassbox.core.dynamics import (
    BootstrapMultirotorParams,
    DynamicsParams,
    FixedWingDynamicsParams,
    ModelParams,
    ResidualDynamicsParams,
    initial_residual_parameters,
    model_family,
    physics_parameters,
    structured_parameters,
)
from glassbox.core.model import RuntimeModelSpec

MODEL_FORMAT_VERSION = 4
MODEL_TYPE = "effective_quadrotor_command_offset_v4"
RESIDUAL_MODEL_TYPE = "structured_acceleration_residual_v1"
FIXED_WING_MODEL_TYPE = "effective_fixedwing_role_aerodynamic_lag_v3"
BOOTSTRAP_MODEL_TYPE = "recursive_bootstrap_multirotor_command_effects_v1"

# Format 3 is every model and belief written before the multirotor
# rotational-response branch was deleted. Its multirotor payloads carry one
# parameter this model no longer has, and its type string names that branch.
LEGACY_MODEL_FORMAT_VERSION = 3
LEGACY_MULTIROTOR_MODEL_TYPE = (
    "effective_quadrotor_command_offset_rotational_response_v3"
)
DROPPED_MULTIROTOR_PARAMETERS = ("angular_response_time_constant",)

MULTIROTOR_PARAMETER_NAMES = (
    "thrust_accel",
    "thrust_command_offset",
    "angular_accel",
    "linear_drag",
    "angular_drag",
    "motor_time_constant",
    "angular_control_cross_coupling",
)
FIXED_WING_PARAMETER_NAMES = (
    "thrust_accel",
    "lift_accel_per_speed_sq",
    "lift_alpha_accel_per_speed_sq",
    "drag_accel_per_speed_sq",
    "side_force_accel_per_speed",
    "surface_angular_accel_per_speed_sq",
    "lateral_surface_cross_angular_accel_per_speed_sq",
    "pitch_stability_accel_per_speed_sq",
    "lateral_stability_angular_accel_per_speed_sq",
    "angular_drag_per_speed",
    "surface_trim",
    "flap_lift_accel_per_speed_sq",
    "flap_drag_accel_per_speed_sq",
    "flap_pitch_angular_accel_per_speed_sq",
    "flap_trim",
    "actuator_time_constant",
)
BOOTSTRAP_PARAMETER_NAMES = (
    "collective_acceleration_per_command",
    "collective_velocity_coefficient",
    "collective_intercept_m_s2",
    "angular_acceleration_per_command",
    "angular_rate_coefficient",
    "angular_rate_product_coefficient",
    "angular_intercept_rad_s2",
)
RESIDUAL_ARRAY_NAMES = (
    "hidden_weights",
    "hidden_bias",
    "output_weights",
    "feature_mean",
    "feature_scale",
    "correction_scale",
)
_RESIDUAL_ANNOTATION_NAMES = (
    "feature_order",
    "correction_order",
    "bounded_output",
    "estimated_wind_correction_target",
)
_MULTIROTOR_MODEL_TYPES = (MODEL_TYPE, LEGACY_MULTIROTOR_MODEL_TYPE)


def parameter_dict(params: ModelParams) -> dict[str, Any]:
    """Convert physical parameter arrays to JSON-compatible values."""

    base = structured_parameters(params)
    if isinstance(base, BootstrapMultirotorParams):
        # The bootstrap parameterization is already stated in the units it is
        # read in, so the payload is the parameters themselves.
        return {
            name: np.asarray(getattr(base, name), dtype=np.float64).tolist()
            for name in BOOTSTRAP_PARAMETER_NAMES
        }
    if isinstance(base, FixedWingDynamicsParams):
        physical = base.physical()
        result: dict[str, Any] = {
            "thrust_accel": float(physical["thrust_accel"]),
            "lift_accel_per_speed_sq": float(physical["lift_accel_per_speed_sq"]),
            "lift_alpha_accel_per_speed_sq": float(
                physical["lift_alpha_accel_per_speed_sq"]
            ),
            "drag_accel_per_speed_sq": float(physical["drag_accel_per_speed_sq"]),
            "side_force_accel_per_speed": float(physical["side_force_accel_per_speed"]),
            "surface_angular_accel_per_speed_sq": np.asarray(
                physical["surface_angular_accel_per_speed_sq"]
            ).tolist(),
            "lateral_surface_cross_angular_accel_per_speed_sq": np.asarray(
                physical["lateral_surface_cross_angular_accel_per_speed_sq"]
            ).tolist(),
            "pitch_stability_accel_per_speed_sq": float(
                physical["pitch_stability_accel_per_speed_sq"]
            ),
            "lateral_stability_angular_accel_per_speed_sq": np.asarray(
                physical["lateral_stability_angular_accel_per_speed_sq"]
            ).tolist(),
            "angular_drag_per_speed": np.asarray(
                physical["angular_drag_per_speed"]
            ).tolist(),
            "surface_trim": np.asarray(physical["surface_trim"]).tolist(),
            "flap_lift_accel_per_speed_sq": float(
                physical["flap_lift_accel_per_speed_sq"]
            ),
            "flap_drag_accel_per_speed_sq": float(
                physical["flap_drag_accel_per_speed_sq"]
            ),
            "flap_pitch_angular_accel_per_speed_sq": float(
                physical["flap_pitch_angular_accel_per_speed_sq"]
            ),
            "flap_trim": float(physical["flap_trim"]),
            "actuator_time_constant": float(physical["actuator_time_constant"]),
        }
    else:
        physical = physics_parameters(params).physical()
        result = {
            "thrust_accel": float(physical["thrust_accel"]),
            "thrust_command_offset": float(physical["thrust_command_offset"]),
            "angular_accel": np.asarray(physical["angular_accel"]).tolist(),
            "linear_drag": float(physical["linear_drag"]),
            "angular_drag": np.asarray(physical["angular_drag"]).tolist(),
            "motor_time_constant": float(physical["motor_time_constant"]),
            "angular_control_cross_coupling": np.asarray(
                physical["angular_control_cross_coupling"]
            ).tolist(),
        }
    if isinstance(params, ResidualDynamicsParams):
        result["residual"] = {
            "input_features": int(params.feature_mean.shape[0]),
            "hidden_units": int(params.hidden_weights.shape[0]),
            "output_accelerations": 6,
            "hidden_weight_norm": float(np.linalg.norm(params.hidden_weights)),
            "output_weight_norm": float(np.linalg.norm(params.output_weights)),
            "feature_mean": np.asarray(params.feature_mean).tolist(),
            "feature_scale": np.asarray(params.feature_scale).tolist(),
            "correction_scale": np.asarray(params.correction_scale).tolist(),
            "frame": "body",
            "bounded_output": True,
        }
    return result


def model_payload(
    params: ModelParams,
    *,
    input_spec: TrajectorySpec,
    runtime_spec: RuntimeModelSpec,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a JSON-compatible differentiable-model artifact."""

    residual = isinstance(params, ResidualDynamicsParams)
    identification_observations = input_spec.observations
    input_spec = input_spec.prediction_spec()
    base = structured_parameters(params)
    fixed_wing = isinstance(base, FixedWingDynamicsParams)
    bootstrap = isinstance(base, BootstrapMultirotorParams)
    family = model_family(params)
    if input_spec.vehicle.family != family.platform:
        raise ValueError(
            f"model family {family.platform!r} cannot bind to vehicle family "
            f"{input_spec.vehicle.family!r}"
        )
    family.validate_control_schema(input_spec.control_names, input_spec.control_roles)
    thrust_mapping = None
    if not fixed_wing and not bootstrap:
        semantics = frozenset(input_spec.control_semantics)
        if semantics <= NORMALIZED_MOTOR_COMMAND_SEMANTICS:
            thrust_mapping = "shared_normalized_command_offset"
        elif semantics <= PHYSICAL_MOTOR_THRUST_SEMANTICS:
            thrust_mapping = "identity_physical_thrust_proxy"
            offset = float(base.physical()["thrust_command_offset"])
            if abs(offset) > 1e-7:
                raise ValueError(
                    "physical multirotor thrust proxies require zero command offset"
                )
        else:
            raise ValueError(
                "unsupported multirotor control semantics: "
                + ", ".join(sorted(semantics))
            )
    if residual:
        expected_feature_size = 6 + len(input_spec.controls) + len(input_spec.exogenous)
        if params.feature_mean.shape != (expected_feature_size,):
            raise ValueError("residual feature configuration does not match input spec")
    payload = {
        "format_version": MODEL_FORMAT_VERSION,
        "model_type": (
            RESIDUAL_MODEL_TYPE
            if residual
            else BOOTSTRAP_MODEL_TYPE
            if bootstrap
            else FIXED_WING_MODEL_TYPE
            if fixed_wing
            else MODEL_TYPE
        ),
        "model_family": family.key,
        "platform": family.platform,
        "parameterization": (
            "structured_base_plus_body_acceleration_residual"
            if residual
            else "direct_command_effect_maps_without_airframe_constants"
            if bootstrap
            else "effective_quadratic_aerodynamics"
            if fixed_wing
            else "effective_positive_coefficients_with_bounded_command_offset"
        ),
        "coordinate_frames": {"world": "NWU", "body": "FLU"},
        "state_order": [
            "position_xyz",
            "velocity_xyz",
            "quaternion_wxyz",
            "angular_velocity_xyz",
        ],
        "input_spec": input_spec.to_dict(),
        "runtime_spec": runtime_spec.to_dict(),
        "identification_observations": [
            channel.to_dict() for channel in identification_observations
        ],
        "latent_state_order": [
            f"applied_{channel.name}" for channel in input_spec.controls
        ],
        "control_order": list(input_spec.control_names),
        "control_roles": list(input_spec.control_roles),
        "control_capability": {
            "required_roles": list(family.required_control_roles),
            "optional_roles": list(family.optional_control_roles),
        },
        "multirotor_thrust_mapping": thrust_mapping,
        "provenance": dict(provenance or {}),
    }
    if residual:
        payload["parameters"] = {
            "base_model_type": (FIXED_WING_MODEL_TYPE if fixed_wing else MODEL_TYPE),
            "base": parameter_dict(params.base),
            "residual": {
                "hidden_weights": params.hidden_weights.tolist(),
                "hidden_bias": params.hidden_bias.tolist(),
                "output_weights": params.output_weights.tolist(),
                "feature_mean": params.feature_mean.tolist(),
                "feature_scale": params.feature_scale.tolist(),
                "correction_scale": params.correction_scale.tolist(),
                "feature_order": [
                    "body_velocity_x",
                    "body_velocity_y",
                    "body_velocity_z",
                    "body_angular_velocity_x",
                    "body_angular_velocity_y",
                    "body_angular_velocity_z",
                    *[f"applied_control:{role}" for role in input_spec.control_roles],
                    *[f"exogenous:{role}" for role in input_spec.exogenous_roles],
                ],
                "correction_order": [
                    "body_linear_acceleration_x",
                    "body_linear_acceleration_y",
                    "body_linear_acceleration_z",
                    "body_angular_acceleration_x",
                    "body_angular_acceleration_y",
                    "body_angular_acceleration_z",
                ],
                "bounded_output": True,
                "estimated_wind_correction_target": "body_linear_acceleration_only",
            },
        }
    else:
        payload["parameters"] = parameter_dict(params)
    return payload


def _decoded_parameters(
    parameters: Mapping[str, Any],
    expected: tuple[str, ...],
    *,
    kind: str,
    dropped: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Return exactly the expected parameters, refusing every other name.

    A payload is only as trustworthy as the parameter set it declares, so a
    missing name and an unrecognized name are both errors rather than a
    silently different model. The one tolerated exception is a multirotor
    payload written under format 3, which carries the deleted
    ``angular_response_time_constant``: that entry is dropped with a warning so
    beliefs and fitted models recorded before the rotational-response branch
    was removed still load. The tolerance goes away when those artifacts are
    re-recorded.
    """

    if not isinstance(parameters, Mapping):
        raise ValueError(f"{kind} parameters must be a mapping")
    supplied = set(parameters)
    missing = sorted(set(expected) - supplied)
    if missing:
        raise ValueError(
            f"{kind} payload is missing parameter(s): {', '.join(missing)}"
        )
    legacy = sorted(supplied.intersection(dropped))
    unknown = sorted(supplied - set(expected) - set(legacy))
    if unknown:
        raise ValueError(
            f"{kind} payload declares unknown parameter(s): {', '.join(unknown)}"
        )
    if legacy:
        warnings.warn(
            f"dropping {', '.join(legacy)} from a {kind} payload written under "
            f"format {LEGACY_MODEL_FORMAT_VERSION}; the multirotor model no "
            "longer has a lagged rotational response, so re-record the "
            "artifact to keep its parameter set exact",
            stacklevel=3,
        )
    return {name: parameters[name] for name in expected}


def _physics_from_payload(parameters: Mapping[str, Any]) -> DynamicsParams:
    values = _decoded_parameters(
        parameters,
        MULTIROTOR_PARAMETER_NAMES,
        kind="multirotor",
        dropped=DROPPED_MULTIROTOR_PARAMETERS,
    )
    return DynamicsParams.from_physical(
        thrust_accel=float(values["thrust_accel"]),
        thrust_command_offset=float(values["thrust_command_offset"]),
        angular_accel=tuple(values["angular_accel"]),
        linear_drag=float(values["linear_drag"]),
        angular_drag=tuple(values["angular_drag"]),
        motor_time_constant=float(values["motor_time_constant"]),
        angular_control_cross_coupling=tuple(
            tuple(row) for row in values["angular_control_cross_coupling"]
        ),
    )


def _fixed_wing_from_payload(
    parameters: Mapping[str, Any],
) -> FixedWingDynamicsParams:
    parameters = _decoded_parameters(
        parameters, FIXED_WING_PARAMETER_NAMES, kind="fixed-wing"
    )
    return FixedWingDynamicsParams.from_physical(
        thrust_accel=float(parameters["thrust_accel"]),
        lift_accel_per_speed_sq=float(parameters["lift_accel_per_speed_sq"]),
        lift_alpha_accel_per_speed_sq=float(
            parameters["lift_alpha_accel_per_speed_sq"]
        ),
        drag_accel_per_speed_sq=float(parameters["drag_accel_per_speed_sq"]),
        side_force_accel_per_speed=float(parameters["side_force_accel_per_speed"]),
        surface_angular_accel_per_speed_sq=tuple(
            parameters["surface_angular_accel_per_speed_sq"]
        ),
        lateral_surface_cross_angular_accel_per_speed_sq=tuple(
            parameters["lateral_surface_cross_angular_accel_per_speed_sq"]
        ),
        pitch_stability_accel_per_speed_sq=float(
            parameters["pitch_stability_accel_per_speed_sq"]
        ),
        lateral_stability_angular_accel_per_speed_sq=tuple(
            parameters["lateral_stability_angular_accel_per_speed_sq"]
        ),
        angular_drag_per_speed=tuple(parameters["angular_drag_per_speed"]),
        actuator_time_constant=float(parameters["actuator_time_constant"]),
        surface_trim=tuple(parameters["surface_trim"]),
        flap_lift_accel_per_speed_sq=float(parameters["flap_lift_accel_per_speed_sq"]),
        flap_drag_accel_per_speed_sq=float(parameters["flap_drag_accel_per_speed_sq"]),
        flap_pitch_angular_accel_per_speed_sq=float(
            parameters["flap_pitch_angular_accel_per_speed_sq"]
        ),
        flap_trim=float(parameters["flap_trim"]),
    )


def _bootstrap_from_payload(
    parameters: Mapping[str, Any],
) -> BootstrapMultirotorParams:
    values = _decoded_parameters(
        parameters, BOOTSTRAP_PARAMETER_NAMES, kind="bootstrap multirotor"
    )
    shapes = {
        "collective_acceleration_per_command": (4,),
        "collective_velocity_coefficient": (3,),
        "collective_intercept_m_s2": (),
        "angular_acceleration_per_command": (3, 4),
        "angular_rate_coefficient": (3, 3),
        "angular_rate_product_coefficient": (3, 3),
        "angular_intercept_rad_s2": (3,),
    }
    decoded: dict[str, Any] = {}
    for name, shape in shapes.items():
        array = np.asarray(values[name], dtype=np.float64)
        if array.shape != shape or not np.all(np.isfinite(array)):
            raise ValueError(
                f"bootstrap multirotor payload parameter {name} must be a "
                f"finite array of shape {shape}"
            )
        # Kept in double precision rather than converted: these are direct
        # estimates in physical units, and the payload records them exactly.
        decoded[name] = array
    return BootstrapMultirotorParams(**decoded)


def dynamics_model_from_payload(
    payload: Mapping[str, Any],
) -> tuple[ModelParams, dict[str, Any]]:
    """Restore model parameters from an already decoded artifact payload."""

    payload = dict(payload)
    version = payload.get("format_version")
    model_type = payload.get("model_type")
    input_spec = TrajectorySpec.from_dict(payload["input_spec"])
    RuntimeModelSpec.from_dict(payload["runtime_spec"])
    if version not in (MODEL_FORMAT_VERSION, LEGACY_MODEL_FORMAT_VERSION):
        raise ValueError(f"unsupported model format version: {version}")
    if model_type in _MULTIROTOR_MODEL_TYPES:
        params: ModelParams = _physics_from_payload(payload["parameters"])
    elif model_type == FIXED_WING_MODEL_TYPE:
        params = _fixed_wing_from_payload(payload["parameters"])
    elif model_type == BOOTSTRAP_MODEL_TYPE:
        params = _bootstrap_from_payload(payload["parameters"])
    elif model_type == RESIDUAL_MODEL_TYPE:
        parameters = _decoded_parameters(
            payload["parameters"],
            ("base_model_type", "base", "residual"),
            kind="structured residual",
        )
        residual = _decoded_parameters(
            parameters["residual"],
            RESIDUAL_ARRAY_NAMES + _RESIDUAL_ANNOTATION_NAMES,
            kind="residual network",
        )
        base_model_type = parameters["base_model_type"]
        if base_model_type in _MULTIROTOR_MODEL_TYPES:
            base = _physics_from_payload(parameters["base"])
        elif base_model_type == FIXED_WING_MODEL_TYPE:
            base = _fixed_wing_from_payload(parameters["base"])
        else:
            raise ValueError(
                f"unsupported structured residual base type: {base_model_type}"
            )
        hidden_bias = jnp.asarray(residual["hidden_bias"])
        if hidden_bias.ndim != 1 or hidden_bias.shape[0] < 1:
            raise ValueError("residual hidden_bias must be a nonempty vector")
        params = initial_residual_parameters(
            base,
            control_size=len(input_spec.controls),
            exogenous_size=len(input_spec.exogenous),
            hidden_units=int(hidden_bias.shape[0]),
            feature_mean=jnp.asarray(residual["feature_mean"]),
            feature_scale=jnp.asarray(residual["feature_scale"]),
            correction_scale=jnp.asarray(residual["correction_scale"]),
        )._replace(
            hidden_weights=jnp.asarray(residual["hidden_weights"]),
            hidden_bias=hidden_bias,
            output_weights=jnp.asarray(residual["output_weights"]),
        )
        expected_feature_size = 6 + len(input_spec.controls) + len(input_spec.exogenous)
        if params.hidden_weights.shape != (
            hidden_bias.shape[0],
            expected_feature_size,
        ):
            raise ValueError(
                "residual hidden_weights do not match hidden and feature sizes"
            )
        if params.output_weights.shape != (6, hidden_bias.shape[0]):
            raise ValueError(
                "residual output_weights must map hidden units to six accelerations"
            )
    else:
        raise ValueError(f"unsupported model type: {model_type}")
    family = model_family(params)
    if input_spec.vehicle.family != family.platform:
        raise ValueError("model input_spec vehicle family does not match model type")
    family.validate_control_schema(input_spec.control_names, input_spec.control_roles)
    return params, payload


def load_dynamics_model(path: str | Path) -> tuple[ModelParams, dict[str, Any]]:
    """Load a nominal model from either a model or dynamics-belief artifact."""

    payload = json.loads(Path(path).read_text())
    from glassbox.belief.belief_io import (
        BELIEF_ARTIFACT_TYPE,
        dynamics_belief_from_payload,
    )

    if payload.get("artifact_type") == BELIEF_ARTIFACT_TYPE:
        belief = dynamics_belief_from_payload(payload)
        return belief.params, dict(payload["nominal_model"])
    return dynamics_model_from_payload(payload)
