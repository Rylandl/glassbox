import json

import numpy as np
import pytest

from glassbox.belief.belief import (
    DynamicsBelief,
    structured_parameter_names,
    structured_parameter_vector,
)
from glassbox.belief.belief_io import belief_payload, load_dynamics_belief
from glassbox.core.data import Channel, make_trajectory_spec
from glassbox.core.dynamics import (
    initial_residual_parameters,
    with_thrust_command_offset,
)
from glassbox.core.fixedwing_synthetic import (
    true_fixed_wing_parameters,
)
from glassbox.core.model import (
    ExecutableModel,
    ModelValidityEnvelope,
    RuntimeModelSpec,
)
from glassbox.core.model_io import load_dynamics_model, model_payload
from glassbox.core.synthetic import true_parameters
from glassbox.io.nanodrone_reference import nanodrone_trajectory_spec


def _write_model_payload(
    params,
    path,
    *,
    input_spec,
    runtime_spec,
    provenance=None,
) -> None:
    """Write a bare nominal-model artifact.

    The library itself only writes beliefs. Model-only payloads still exist on
    disk from before that fold, so both loaders have to keep reading them; this
    helper is how those payloads are produced for the tests that pin it.
    """

    path.write_text(
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
    _write_model_payload(
        original,
        path,
        input_spec=input_spec,
        runtime_spec=_runtime_spec(),
        provenance={"flight": "fixture"},
    )
    restored, payload = load_dynamics_model(path)

    for original_leaf, restored_leaf in zip(original, restored, strict=True):
        np.testing.assert_allclose(restored_leaf, original_leaf, rtol=1e-6)
    assert payload["model_type"] == "effective_quadrotor_command_offset_v4"
    assert payload["multirotor_thrust_mapping"] == ("shared_normalized_command_offset")
    assert payload["parameters"]["thrust_command_offset"] == pytest.approx(-0.12)
    assert payload["format_version"] == 4
    assert payload["provenance"] == {"flight": "fixture"}
    assert payload["input_spec"] == input_spec.prediction_spec().to_dict()


def test_nominal_loader_unwraps_dynamics_belief(
    tmp_path, quadrotor_trajectory_seed0_dur0_1s
) -> None:
    path = tmp_path / "belief.json"
    input_spec = quadrotor_trajectory_seed0_dur0_1s.spec
    DynamicsBelief(
        model=ExecutableModel(true_parameters(), input_spec, _runtime_spec()),
        provenance={"flight": "fixture"},
    ).save(path)

    restored, payload = load_dynamics_model(path)

    for expected_leaf, restored_leaf in zip(true_parameters(), restored, strict=True):
        np.testing.assert_allclose(restored_leaf, expected_leaf)
    assert payload["model_type"] == "effective_quadrotor_command_offset_v4"
    assert payload["provenance"] == {"flight": "fixture"}


def test_residual_model_json_round_trip(
    tmp_path, quadrotor_trajectory_seed0_dur0_1s
) -> None:
    path = tmp_path / "residual_model.json"
    original = initial_residual_parameters(true_parameters(), hidden_units=5)

    _write_model_payload(
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
        "effective_quadrotor_command_offset_v4"
    )
    assert payload["format_version"] == 4


def test_physical_rotor_thrust_proxy_requires_identity_offset() -> None:
    input_spec = nanodrone_trajectory_spec()

    payload = model_payload(
        true_parameters(), input_spec=input_spec, runtime_spec=_runtime_spec()
    )

    assert payload["multirotor_thrust_mapping"] == ("identity_physical_thrust_proxy")
    assert all(
        channel["kind"] != "observation"
        for channel in payload["input_spec"]["channels"]
    )
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
        Channel(
            name=f"wind_{axis}_m_s",
            role=f"estimated_wind_{axis}",
            semantic="estimated_environment_at_prediction_start",
            unit="m/s",
            kind="exogenous",
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

    _write_model_payload(
        original, path, input_spec=input_spec, runtime_spec=_runtime_spec()
    )
    restored, payload = load_dynamics_model(path)

    assert restored.feature_mean.shape == (12,)
    assert payload["parameters"]["residual"]["feature_order"][-2:] == [
        "exogenous:estimated_wind_north",
        "exogenous:estimated_wind_west",
    ]


def test_fixed_wing_residual_model_json_round_trip(tmp_path, fixedwing_flight) -> None:
    path = tmp_path / "fixed_wing_residual_model.json"
    base = true_fixed_wing_parameters()
    original = initial_residual_parameters(base, hidden_units=4)
    input_spec = fixedwing_flight(1, 0.1).spec

    _write_model_payload(
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
    _write_model_payload(
        original, path, input_spec=input_spec, runtime_spec=_runtime_spec()
    )
    restored, payload = load_dynamics_model(path)

    for original_leaf, restored_leaf in zip(original, restored, strict=True):
        np.testing.assert_allclose(restored_leaf, original_leaf, rtol=1e-6)
    assert payload["model_type"] == "effective_fixedwing_role_aerodynamic_lag_v3"
    assert payload["format_version"] == 4
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

    with pytest.raises(ValueError, match="unsupported model format version"):
        load_dynamics_model(path)


def test_strict_decoder_refuses_unknown_and_missing_parameters(
    tmp_path, quadrotor_trajectory_seed0_dur0_1s
) -> None:
    path = tmp_path / "model.json"
    payload = model_payload(
        true_parameters(),
        input_spec=quadrotor_trajectory_seed0_dur0_1s.spec,
        runtime_spec=_runtime_spec(),
    )

    unknown = json.loads(json.dumps(payload))
    unknown["parameters"]["rotor_inertia"] = 0.1
    path.write_text(json.dumps(unknown))
    with pytest.raises(ValueError, match="unknown parameter"):
        load_dynamics_model(path)

    missing = json.loads(json.dumps(payload))
    del missing["parameters"]["linear_drag"]
    path.write_text(json.dumps(missing))
    with pytest.raises(ValueError, match="missing parameter"):
        load_dynamics_model(path)


def test_format_three_multirotor_payload_drops_the_rotational_response(
    tmp_path, quadrotor_trajectory_seed0_dur0_1s
) -> None:
    path = tmp_path / "legacy_model.json"
    payload = model_payload(
        true_parameters(),
        input_spec=quadrotor_trajectory_seed0_dur0_1s.spec,
        runtime_spec=_runtime_spec(),
    )
    payload["format_version"] = 3
    payload["model_type"] = "effective_quadrotor_command_offset_rotational_response_v3"
    payload["parameters"]["angular_response_time_constant"] = [0.04, 0.04, 0.06]
    path.write_text(json.dumps(payload))

    with pytest.warns(UserWarning, match="angular_response_time_constant"):
        restored, _ = load_dynamics_model(path)

    for expected_leaf, restored_leaf in zip(true_parameters(), restored, strict=True):
        np.testing.assert_allclose(restored_leaf, expected_leaf, rtol=1e-6)
    assert not any("angular_response" in name for name in restored._asdict())


def _legacy_belief_payload(
    belief: DynamicsBelief,
    *,
    parameter_belief: dict | None = None,
) -> dict:
    """Reshape a current payload into the format-4 one a belief used to write."""

    payload = belief_payload(belief)
    payload.pop("information")
    payload.pop("forecast_error")
    payload["format_version"] = 4
    names = list(structured_parameter_names(true_parameters()))
    payload["parameter_belief"] = parameter_belief or {
        "format_version": 1,
        "kind": "point_estimate",
        "uncertainty_available": False,
        "scenario_count": 1,
        "update_count": 0,
    }
    payload["parameter_evidence"] = {
        "format_version": 2,
        "kind": "local_structured_parameter_information",
        "coordinate_system": "unconstrained_structured_parameter_vector",
        "parameter_scale_semantics": (
            "one_transformed_unit_or_same_axis_effective_authority"
        ),
        "parameter_names": names,
        "center": np.asarray(
            structured_parameter_vector(true_parameters()), dtype=float
        ).tolist(),
        "information_matrix": np.eye(len(names)).tolist(),
        "parameter_scale": np.ones(len(names)).tolist(),
        "fitted_parameter_mask": np.ones(len(names), dtype=bool).tolist(),
        "rank_relative_tolerance": 1e-6,
        "group_score_vectors": np.zeros((1, len(names))).tolist(),
    }
    payload["predictive_error"] = {
        "format_version": 3,
        "kind": "empirical_horizon_tangent_moments",
        "horizons_s": [0.1],
        "tangent_bias": [[0.1] + [0.0] * 11],
        "tangent_covariance": [np.diag([0.04] + [0.01] * 11).tolist()],
        "quantile_levels": [0.5],
        "raw_sample_count": [8],
        "effective_sample_count": [8.0],
        "independent_group_count": [2],
        "source": "held_out_rollout_endpoints",
        "weighting": "equal_source_group_then_trajectory_then_endpoint",
    }
    payload["predictive_error_parameter_update_count"] = 0
    return payload


def test_legacy_belief_folds_the_forecast_bias_into_the_envelope(
    tmp_path, quadrotor_trajectory_seed0_dur0_1s
) -> None:
    path = tmp_path / "legacy_belief.json"
    input_spec = quadrotor_trajectory_seed0_dur0_1s.spec
    belief = DynamicsBelief(
        model=ExecutableModel(true_parameters(), input_spec, _runtime_spec()),
    )
    path.write_text(json.dumps(_legacy_belief_payload(belief)))

    with pytest.warns(UserWarning, match="format 4"):
        restored = load_dynamics_belief(path)

    # The old model corrected the bias and reported the covariance about it.
    # Nothing corrects it now, so the envelope is the uncentered second moment
    # and the conversion is exact rather than lossy.
    expected = np.diag([0.04] + [0.01] * 11)
    expected[0, 0] += 0.01
    np.testing.assert_allclose(
        restored.forecast_error.tangent_covariance[0], expected, atol=1e-12
    )


def test_legacy_belief_without_a_parameter_covariance_loads_at_rank_zero(
    tmp_path, quadrotor_trajectory_seed0_dur0_1s
) -> None:
    path = tmp_path / "legacy_belief.json"
    input_spec = quadrotor_trajectory_seed0_dur0_1s.spec
    belief = DynamicsBelief(
        model=ExecutableModel(true_parameters(), input_spec, _runtime_spec()),
    )
    path.write_text(json.dumps(_legacy_belief_payload(belief)))

    with pytest.warns(UserWarning, match="cannot be converted"):
        restored = load_dynamics_belief(path)

    assert restored.information.resolved_rank() == 0
    assert restored.information.names == structured_parameter_names(true_parameters())
    assert restored.information.source == "legacy_artifact"


def test_legacy_parameter_covariance_converts_to_precision_exactly(
    tmp_path, quadrotor_trajectory_seed0_dur0_1s
) -> None:
    path = tmp_path / "legacy_belief.json"
    input_spec = quadrotor_trajectory_seed0_dur0_1s.spec
    names = list(structured_parameter_names(true_parameters()))
    covariance = np.zeros((len(names), len(names)))
    covariance[0, 0] = 0.04
    belief = DynamicsBelief(
        model=ExecutableModel(true_parameters(), input_spec, _runtime_spec()),
    )
    payload = _legacy_belief_payload(
        belief,
        parameter_belief={
            "format_version": 1,
            "kind": "local_gaussian_structured_parameters",
            "coordinate_system": "unconstrained_structured_parameter_vector",
            "parameter_names": names,
            "covariance": covariance.tolist(),
            "source": "configuration_members",
            "evidence_count": 5,
            "effective_sample_count": 5.0,
            "update_count": 0,
        },
    )
    path.write_text(json.dumps(payload))

    with pytest.warns(UserWarning, match="format 4"):
        restored = load_dynamics_belief(path)

    assert restored.information.resolved_rank() == 1
    np.testing.assert_allclose(
        restored.information.covariance(), covariance, atol=1e-12
    )


def test_format_three_belief_drops_the_rotational_response_coordinates(
    tmp_path, quadrotor_trajectory_seed0_dur0_1s
) -> None:
    path = tmp_path / "legacy_belief.json"
    input_spec = quadrotor_trajectory_seed0_dur0_1s.spec
    belief = DynamicsBelief(
        model=ExecutableModel(true_parameters(), input_spec, _runtime_spec()),
    )
    payload = _legacy_belief_payload(belief)
    payload["format_version"] = 3
    payload["nominal_model"]["format_version"] = 3
    payload["nominal_model"]["model_type"] = (
        "effective_quadrotor_command_offset_rotational_response_v3"
    )
    payload["nominal_model"]["parameters"]["angular_response_time_constant"] = [
        1e-4,
        1e-4,
        1e-4,
    ]
    evidence = payload["parameter_evidence"]
    names = list(evidence["parameter_names"])
    names[10:10] = [f"log_angular_response_time_constant[{axis}]" for axis in range(3)]
    evidence["parameter_names"] = names
    for key in ("center", "parameter_scale", "fitted_parameter_mask"):
        values = list(evidence[key])
        values[10:10] = [0.0, 0.0, 0.0]
        evidence[key] = values
    matrix = np.asarray(evidence["information_matrix"], dtype=float)
    matrix = np.insert(np.insert(matrix, [10] * 3, 0.0, axis=0), [10] * 3, 0.0, axis=1)
    evidence["information_matrix"] = matrix.tolist()
    path.write_text(json.dumps(payload))

    with pytest.warns(UserWarning, match="format 3"):
        restored = load_dynamics_belief(path)

    for expected_leaf, restored_leaf in zip(
        true_parameters(), restored.params, strict=True
    ):
        np.testing.assert_allclose(restored_leaf, expected_leaf, rtol=1e-6)
    assert restored.information.names == structured_parameter_names(true_parameters())
    assert restored.information.resolved_rank() == 0


def test_format_three_belief_refuses_information_on_a_deleted_coordinate(
    tmp_path, quadrotor_trajectory_seed0_dur0_1s
) -> None:
    from glassbox.belief.belief_io import _without_dropped_parameters

    names = list(structured_parameter_names(true_parameters()))
    names[10:10] = [f"log_angular_response_time_constant[{axis}]" for axis in range(3)]
    matrix = np.eye(22)

    with pytest.raises(ValueError, match="cannot be read by the memoryless"):
        _without_dropped_parameters(
            {"parameter_names": names, "information_matrix": matrix.tolist()},
            matrix_keys=("information_matrix",),
        )


def test_belief_loader_reads_a_bare_model_as_a_point_belief(
    tmp_path, quadrotor_trajectory_seed0_dur0_1s
) -> None:
    path = tmp_path / "model.json"
    params = with_thrust_command_offset(true_parameters(), -0.05)
    input_spec = quadrotor_trajectory_seed0_dur0_1s.spec
    _write_model_payload(
        params,
        path,
        input_spec=input_spec,
        runtime_spec=_runtime_spec(),
        provenance={"flight": "fixture"},
    )

    belief = load_dynamics_belief(path)

    for original_leaf, restored_leaf in zip(params, belief.params, strict=True):
        np.testing.assert_allclose(restored_leaf, original_leaf, rtol=1e-6)
    assert belief.input_spec == input_spec.prediction_spec()
    assert belief.runtime_spec == _runtime_spec()
    assert belief.forecast_error is None
    assert belief.information.resolved_rank() == 0
    assert belief.provenance == {"flight": "fixture"}
