import json

import numpy as np
import pytest

from glassbox.belief.belief import DynamicsBelief
from glassbox.core.data import ExogenousChannel, make_trajectory_spec
from glassbox.core.dynamics import (
    initial_residual_parameters,
    with_thrust_command_offset,
)
from glassbox.core.fixedwing_synthetic import (
    generate_fixed_wing_trajectory,
    true_fixed_wing_parameters,
)
from glassbox.core.model_io import (
    load_dynamics_model,
    model_payload,
    save_dynamics_model,
)
from glassbox.core.runtime import ModelValidityEnvelope, RuntimeModelSpec
from glassbox.core.synthetic import true_parameters
from glassbox.io.nanodrone_reference import nanodrone_trajectory_spec


def _runtime_spec() -> RuntimeModelSpec:
    return RuntimeModelSpec(
        sample_period_s=0.01,
        validity_envelope=ModelValidityEnvelope(
            body_velocity_center_m_s=(0.0, 0.0, 0.0),
            body_velocity_half_width_m_s=(5.0, 5.0, 5.0),
            angular_velocity_center_rad_s=(0.0, 0.0, 0.0),
            angular_velocity_half_width_rad_s=(2.0, 2.0, 2.0),
        ),
    )


def test_model_json_round_trip(tmp_path, quadrotor_trajectory_seed0_dur0_1s) -> None:
    path = tmp_path / "model.json"
    original = with_thrust_command_offset(true_parameters(), -0.12)

    input_spec = quadrotor_trajectory_seed0_dur0_1s.spec
    save_dynamics_model(
        original,
        path,
        input_spec=input_spec,
        runtime_spec=_runtime_spec(),
        provenance={"flight": "fixture"},
    )
    restored, payload = load_dynamics_model(path)

    for original_leaf, restored_leaf in zip(original, restored, strict=True):
        np.testing.assert_allclose(restored_leaf, original_leaf, rtol=1e-6)
    assert payload["model_type"] == (
        "effective_quadrotor_command_offset_rotational_response_v3"
    )
    assert payload["multirotor_thrust_mapping"] == ("shared_normalized_command_offset")
    assert payload["parameters"]["thrust_command_offset"] == pytest.approx(-0.12)
    assert payload["format_version"] == 3
    assert payload["provenance"] == {"flight": "fixture"}
    assert payload["input_spec"] == input_spec.prediction_spec().to_dict()


def test_nominal_loader_unwraps_dynamics_belief(
    tmp_path, quadrotor_trajectory_seed0_dur0_1s
) -> None:
    path = tmp_path / "belief.json"
    input_spec = quadrotor_trajectory_seed0_dur0_1s.spec
    DynamicsBelief(
        params=true_parameters(),
        input_spec=input_spec,
        runtime_spec=_runtime_spec(),
        provenance={"flight": "fixture"},
    ).save(path)

    restored, payload = load_dynamics_model(path)

    for expected_leaf, restored_leaf in zip(true_parameters(), restored, strict=True):
        np.testing.assert_allclose(restored_leaf, expected_leaf)
    assert payload["model_type"] == (
        "effective_quadrotor_command_offset_rotational_response_v3"
    )
    assert payload["provenance"] == {"flight": "fixture"}


def test_residual_model_json_round_trip(
    tmp_path, quadrotor_trajectory_seed0_dur0_1s
) -> None:
    path = tmp_path / "residual_model.json"
    original = initial_residual_parameters(true_parameters(), hidden_units=5)

    save_dynamics_model(
        original,
        path,
        input_spec=quadrotor_trajectory_seed0_dur0_1s.spec,
        runtime_spec=_runtime_spec(),
    )
    restored, payload = load_dynamics_model(path)

    for original_leaf, restored_leaf in zip(original.base, restored.base, strict=True):
        np.testing.assert_allclose(restored_leaf, original_leaf, rtol=1e-6)
    np.testing.assert_allclose(restored.hidden_weights, original.hidden_weights)
    np.testing.assert_allclose(restored.output_weights, original.output_weights)
    np.testing.assert_allclose(restored.feature_scale, original.feature_scale)
    assert payload["model_type"] == "structured_acceleration_residual_v1"
    assert payload["parameters"]["base_model_type"] == (
        "effective_quadrotor_command_offset_rotational_response_v3"
    )
    assert payload["format_version"] == 3


def test_physical_rotor_thrust_proxy_requires_identity_offset() -> None:
    input_spec = nanodrone_trajectory_spec()

    payload = model_payload(
        true_parameters(), input_spec=input_spec, runtime_spec=_runtime_spec()
    )

    assert payload["multirotor_thrust_mapping"] == ("identity_physical_thrust_proxy")
    assert payload["input_spec"]["observations"] == []
    assert [channel["role"] for channel in payload["identification_observations"]] == [
        "specific_force_x",
        "specific_force_y",
        "specific_force_z",
    ]
    with pytest.raises(ValueError, match="require zero command offset"):
        model_payload(
            with_thrust_command_offset(true_parameters(), -0.1),
            input_spec=input_spec,
            runtime_spec=_runtime_spec(),
        )


def test_residual_model_serializes_typed_exogenous_features(tmp_path) -> None:
    path = tmp_path / "context_residual_model.json"
    channels = tuple(
        ExogenousChannel(
            name=f"wind_{axis}_m_s",
            role=f"estimated_wind_{axis}",
            semantic="estimated_environment_at_prediction_start",
            unit="m/s",
            frame="NWU",
        )
        for axis in ("north", "west")
    )
    input_spec = make_trajectory_spec(
        (
            "motor_front_left",
            "motor_front_right",
            "motor_rear_right",
            "motor_rear_left",
        ),
        family="multirotor",
        observation_source="estimated",
        exogenous=channels,
    )
    original = initial_residual_parameters(
        true_parameters(), hidden_units=4, exogenous_size=2
    )

    save_dynamics_model(
        original, path, input_spec=input_spec, runtime_spec=_runtime_spec()
    )
    restored, payload = load_dynamics_model(path)

    assert restored.feature_mean.shape == (12,)
    assert payload["parameters"]["residual"]["feature_order"][-2:] == [
        "exogenous:estimated_wind_north",
        "exogenous:estimated_wind_west",
    ]


def test_fixed_wing_residual_model_json_round_trip(tmp_path) -> None:
    path = tmp_path / "fixed_wing_residual_model.json"
    base = true_fixed_wing_parameters()
    original = initial_residual_parameters(base, hidden_units=4)
    input_spec = generate_fixed_wing_trajectory(seed=1, duration_s=0.1).spec

    save_dynamics_model(
        original, path, input_spec=input_spec, runtime_spec=_runtime_spec()
    )
    restored, payload = load_dynamics_model(path)

    assert restored.base.__class__ is base.__class__
    for original_leaf, restored_leaf in zip(original.base, restored.base, strict=True):
        np.testing.assert_allclose(restored_leaf, original_leaf, rtol=1e-6)
    assert payload["model_type"] == "structured_acceleration_residual_v1"
    assert payload["parameters"]["base_model_type"] == (
        "effective_fixedwing_role_aerodynamic_lag_v3"
    )
    assert payload["platform"] == "fixedwing"


def test_fixed_wing_model_json_round_trip(
    tmp_path, fixedwing_trajectory_seed0_dur0_1s
) -> None:
    path = tmp_path / "fixed_wing_model.json"
    original = true_fixed_wing_parameters()

    input_spec = fixedwing_trajectory_seed0_dur0_1s.spec
    save_dynamics_model(
        original, path, input_spec=input_spec, runtime_spec=_runtime_spec()
    )
    restored, payload = load_dynamics_model(path)

    for original_leaf, restored_leaf in zip(original, restored, strict=True):
        np.testing.assert_allclose(restored_leaf, original_leaf, rtol=1e-6)
    assert payload["model_type"] == "effective_fixedwing_role_aerodynamic_lag_v3"
    assert payload["format_version"] == 3
    assert payload["platform"] == "fixedwing"
    assert payload["control_order"] == [
        "throttle",
        "aileron",
        "elevator",
        "rudder",
    ]
    assert payload["control_capability"] == {
        "required_roles": ["throttle", "roll", "pitch"],
        "optional_roles": ["yaw", "flap"],
    }


def test_rejects_noncurrent_model_format(
    tmp_path, fixedwing_trajectory_seed0_dur0_1s
) -> None:
    path = tmp_path / "old_model.json"
    payload = model_payload(
        true_fixed_wing_parameters(),
        input_spec=fixedwing_trajectory_seed0_dur0_1s.spec,
        runtime_spec=_runtime_spec(),
    )
    payload["format_version"] = 1
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="unsupported model format/type"):
        load_dynamics_model(path)
