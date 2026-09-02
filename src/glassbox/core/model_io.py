"""Serialization for fitted differentiable dynamics models."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import jax.numpy as jnp

from glassbox.core.data import (
    NORMALIZED_MOTOR_COMMAND_SEMANTICS,
    PHYSICAL_MOTOR_THRUST_SEMANTICS,
    TrajectorySpec,
)
from glassbox.core.dynamics import (
    DynamicsParams,
    FixedWingDynamicsParams,
    ModelParams,
    ResidualDynamicsParams,
    initial_residual_parameters,
    model_family,
    structured_parameters,
)
from glassbox.core.evaluation import parameter_dict
from glassbox.core.runtime import RuntimeModelSpec

MODEL_FORMAT_VERSION = 3
MODEL_TYPE = "effective_quadrotor_command_offset_rotational_response_v3"
RESIDUAL_MODEL_TYPE = "structured_acceleration_residual_v1"
FIXED_WING_MODEL_TYPE = "effective_fixedwing_role_aerodynamic_lag_v3"


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
    family = model_family(params)
    if input_spec.vehicle.family != family.platform:
        raise ValueError(
            f"model family {family.platform!r} cannot bind to vehicle family "
            f"{input_spec.vehicle.family!r}"
        )
    family.validate_control_schema(input_spec.control_names, input_spec.control_roles)
    thrust_mapping = None
    if not fixed_wing:
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
            else FIXED_WING_MODEL_TYPE
            if fixed_wing
            else MODEL_TYPE
        ),
        "model_family": family.key,
        "platform": family.platform,
        "parameterization": (
            "structured_base_plus_body_acceleration_residual"
            if residual
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
            *[f"applied_{channel.name}" for channel in input_spec.controls],
            *(
                []
                if fixed_wing
                else [
                    "control_generated_roll_angular_acceleration",
                    "control_generated_pitch_angular_acceleration",
                    "control_generated_yaw_angular_acceleration",
                ]
            ),
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


def save_dynamics_model(
    params: ModelParams,
    path: str | Path,
    *,
    input_spec: TrajectorySpec,
    runtime_spec: RuntimeModelSpec,
    provenance: Mapping[str, Any] | None = None,
) -> None:
    """Write a fitted dynamics model as readable JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            model_payload(
                params,
                input_spec=input_spec,
                runtime_spec=runtime_spec,
                provenance=provenance,
            ),
            indent=2,
        )
        + "\n"
    )


def _physics_from_payload(parameters: Mapping[str, Any]) -> DynamicsParams:
    return DynamicsParams.from_physical(
        thrust_accel=float(parameters["thrust_accel"]),
        thrust_command_offset=float(parameters["thrust_command_offset"]),
        angular_accel=tuple(parameters["angular_accel"]),
        linear_drag=float(parameters["linear_drag"]),
        angular_drag=tuple(parameters["angular_drag"]),
        motor_time_constant=float(parameters["motor_time_constant"]),
        angular_response_time_constant=tuple(
            parameters["angular_response_time_constant"]
        ),
        angular_control_cross_coupling=tuple(
            tuple(row) for row in parameters["angular_control_cross_coupling"]
        ),
    )


def _fixed_wing_from_payload(
    parameters: Mapping[str, Any],
) -> FixedWingDynamicsParams:
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


def dynamics_model_from_payload(
    payload: Mapping[str, Any],
) -> tuple[ModelParams, dict[str, Any]]:
    """Restore model parameters from an already decoded artifact payload."""

    payload = dict(payload)
    version = payload.get("format_version")
    model_type = payload.get("model_type")
    input_spec = TrajectorySpec.from_dict(payload["input_spec"])
    RuntimeModelSpec.from_dict(payload["runtime_spec"])
    if version == MODEL_FORMAT_VERSION and model_type == MODEL_TYPE:
        params: ModelParams = _physics_from_payload(payload["parameters"])
    elif version == MODEL_FORMAT_VERSION and model_type == FIXED_WING_MODEL_TYPE:
        params = _fixed_wing_from_payload(payload["parameters"])
    elif version == MODEL_FORMAT_VERSION and model_type == RESIDUAL_MODEL_TYPE:
        parameters = payload["parameters"]
        residual = parameters["residual"]
        base_model_type = parameters["base_model_type"]
        if base_model_type == MODEL_TYPE:
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
        raise ValueError(
            f"unsupported model format/type: version={version}, type={model_type}"
        )
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
