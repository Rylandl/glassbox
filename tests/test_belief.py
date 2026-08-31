from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from glassbox.belief import (
    DynamicsBelief,
    EmpiricalErrorSample,
    EmpiricalHorizonPredictiveError,
    LocalGaussianParameterBelief,
    apply_tangent_correction,
    structured_parameter_vector,
    with_structured_parameter_vector,
)
from glassbox.dynamics import ResidualDynamicsParams, initial_residual_parameters
from glassbox.evaluation import rigid_body_tangent_errors
from glassbox.fixedwing_synthetic import true_fixed_wing_parameters
from glassbox.runtime import RuntimeDynamicsModel, runtime_spec_from_trajectory
from glassbox.synthetic import generate_trajectory, true_parameters


def _error_model(value: float = 0.0) -> EmpiricalHorizonPredictiveError:
    first = np.full((3, 12), value, dtype=np.float64)
    second = np.full((2, 12), 2.0 * value, dtype=np.float64)
    return EmpiricalHorizonPredictiveError.from_samples(
        {
            0.1: (
                EmpiricalErrorSample(first, "group-a", "flight-a"),
                EmpiricalErrorSample(second, "group-b", "flight-b"),
            ),
            0.2: (
                EmpiricalErrorSample(first, "group-a", "flight-a"),
                EmpiricalErrorSample(second, "group-b", "flight-b"),
            ),
        }
    )


def _nonsingular_error_model(scale: float = 0.02) -> EmpiricalHorizonPredictiveError:
    errors = scale * np.concatenate((np.eye(12), -np.eye(12)), axis=0)
    samples = (
        EmpiricalErrorSample(errors, "group-a", "flight-a"),
        EmpiricalErrorSample(errors, "group-b", "flight-b"),
    )
    return EmpiricalHorizonPredictiveError.from_samples({0.1: samples, 0.2: samples})


def _parameter_belief(params, *, spread: float = 0.2):
    center = np.asarray(structured_parameter_vector(params))
    positive = center.copy()
    negative = center.copy()
    positive[0] += spread
    negative[0] -= spread
    return LocalGaussianParameterBelief.from_members(
        params,
        (
            with_structured_parameter_vector(params, jnp.asarray(positive)),
            with_structured_parameter_vector(params, jnp.asarray(negative)),
        ),
        source="independent_vehicle_members",
    )


def test_tangent_error_and_correction_use_quaternion_geometry() -> None:
    state = np.asarray(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )
    correction = np.asarray(
        [1.0, -2.0, 3.0, 0.1, 0.2, 0.3, 0.0, 0.0, 0.2, 0.4, 0.5, 0.6]
    )

    corrected = np.asarray(
        apply_tangent_correction(jnp.asarray(state), jnp.asarray(correction))
    )
    recovered = rigid_body_tangent_errors(state[None, :], corrected[None, :])[0]
    sign_equivalent = corrected.copy()
    sign_equivalent[6:10] *= -1.0

    np.testing.assert_allclose(recovered, correction, atol=1e-7)
    np.testing.assert_allclose(
        rigid_body_tangent_errors(corrected[None, :], sign_equivalent[None, :]),
        0.0,
        atol=1e-7,
    )


def test_empirical_error_model_balances_complete_source_groups() -> None:
    model = _error_model(1.0)

    assert model.available
    assert model.raw_sample_count == (5, 5)
    np.testing.assert_allclose(model.tangent_bias, 1.5)
    assert model.effective_sample_count[0] == pytest.approx(4.8)
    assert model.independent_group_count == (2, 2)
    assert np.min(np.linalg.eigvalsh(model.tangent_covariance[0])) >= -1e-10
    bias, covariance = model.moments(0.05)
    np.testing.assert_allclose(bias, 0.75)
    np.testing.assert_allclose(
        covariance,
        0.5 * model.tangent_covariance[0],
        atol=1e-7,
    )


def test_structured_parameter_belief_is_generic_and_leaves_residual_fixed() -> None:
    residual = initial_residual_parameters(
        true_fixed_wing_parameters(), hidden_units=3
    )
    vector = np.asarray(structured_parameter_vector(residual))
    changed = vector.copy()
    changed[0] += 0.1

    updated = with_structured_parameter_vector(residual, jnp.asarray(changed))

    assert isinstance(updated, ResidualDynamicsParams)
    np.testing.assert_allclose(structured_parameter_vector(updated), changed)
    np.testing.assert_allclose(updated.hidden_weights, residual.hidden_weights)
    np.testing.assert_allclose(updated.output_weights, residual.output_weights)


def test_belief_round_trip_and_runtime_forecast(tmp_path) -> None:
    trajectory = generate_trajectory(seed=4, duration_s=0.3)
    error_model = _error_model(0.01)
    belief = DynamicsBelief(
        params=true_parameters(),
        input_spec=trajectory.spec,
        runtime_spec=runtime_spec_from_trajectory(trajectory),
        predictive_error=error_model,
        provenance={"fixture": True},
    )
    path = tmp_path / "belief.json"

    belief.save(path)
    restored = DynamicsBelief.load(path)
    runtime = restored.compile_for_nmpc()
    commands = jnp.asarray(trajectory.controls[:5])
    forecast = runtime.rollout(
        jnp.asarray(trajectory.states[0]),
        commands,
    )
    nominal_from_legacy_loader = RuntimeDynamicsModel.load(path)

    assert restored.provenance == {"fixture": True}
    assert restored.predictive_error.available
    assert forecast.uncertainty_available
    assert forecast.uncertainty_horizon_supported
    assert forecast.nominal_states.shape == (6, 13)
    assert forecast.mean_states.shape == (6, 13)
    assert forecast.tangent_covariance.shape == (6, 12, 12)
    assert forecast.quantile_levels == (0.5, 0.8, 0.9)
    assert forecast.group_radius_quantiles.shape == (6, 3, 4)
    assert forecast.validity_utilization.shape == (6, 6)
    assert not np.allclose(forecast.mean_states[1:], forecast.nominal_states[1:])
    assert nominal_from_legacy_loader.command_size == runtime.nominal.command_size


def test_parameter_belief_propagates_and_scores_candidate_information(tmp_path) -> None:
    trajectory = generate_trajectory(seed=7, duration_s=0.3)
    params = true_parameters()
    belief = DynamicsBelief(
        params=params,
        input_spec=trajectory.spec,
        runtime_spec=runtime_spec_from_trajectory(trajectory),
        predictive_error=_nonsingular_error_model(),
        parameter_belief=_parameter_belief(params),
    )
    path = tmp_path / "parameter-belief.json"
    belief.save(path)

    restored = DynamicsBelief.load(path)
    runtime = restored.compile_for_nmpc()
    commands = jnp.asarray(trajectory.controls[:5])
    assessment = runtime.assess_plan(
        jnp.asarray(trajectory.states[0]),
        commands,
    )

    assert isinstance(restored.parameter_belief, LocalGaussianParameterBelief)
    assert assessment.information_available
    assert assessment.expected_parameter_information_gain_nats > 0.0
    assert assessment.expected_parameter_covariance is not None
    assert np.trace(assessment.expected_parameter_covariance) < np.trace(
        restored.parameter_belief.covariance
    )
    assert assessment.prediction.parameter_uncertainty_available
    assert np.max(assessment.prediction.parameter_tangent_covariance) > 0.0


def test_live_update_moves_structured_parameters_and_preserves_error_provenance() -> (
    None
):
    telemetry = generate_trajectory(seed=11, duration_s=0.4)
    true_vector = np.asarray(structured_parameter_vector(true_parameters()))
    nominal_vector = true_vector.copy()
    nominal_vector[0] += 0.25
    nominal = with_structured_parameter_vector(
        true_parameters(), jnp.asarray(nominal_vector)
    )
    belief = DynamicsBelief(
        params=nominal,
        input_spec=telemetry.spec,
        runtime_spec=runtime_spec_from_trajectory(telemetry),
        predictive_error=_nonsingular_error_model(scale=0.05),
        parameter_belief=_parameter_belief(nominal, spread=0.4),
    )

    updated, report = belief.update(telemetry)

    assert report.applied
    assert report.used_window_count == 4
    assert report.normalized_innovation_rms_after < (
        report.normalized_innovation_rms_before
    )
    assert report.posterior_covariance_trace < report.prior_covariance_trace
    assert updated.parameter_belief.update_count == 1
    assert not updated.predictive_error_current
    assert updated.predictive_error_parameter_update_count == 0
    assert not np.allclose(
        structured_parameter_vector(updated.params),
        structured_parameter_vector(belief.params),
    )
    assert updated.provenance["online_adaptation"]["last_update"]["applied"]

    updated_again, second_report = updated.update(telemetry)
    assert second_report.applied
    assert updated_again.parameter_belief.update_count == 2
    assert updated_again.predictive_error_parameter_update_count == 0


def test_live_update_does_not_require_actionable_control_semantics() -> None:
    telemetry = generate_trajectory(seed=3, duration_s=0.2)
    physical_spec = replace(
        telemetry.spec,
        controls=tuple(
            replace(
                channel,
                semantic="squared_rotor_speed_ratio",
                minimum=None,
                maximum=None,
            )
            for channel in telemetry.spec.controls
        ),
    )
    telemetry = replace(telemetry, spec=physical_spec)
    params = true_parameters()
    belief = DynamicsBelief(
        params=params,
        input_spec=physical_spec,
        runtime_spec=runtime_spec_from_trajectory(telemetry),
        predictive_error=_nonsingular_error_model(scale=0.05),
        parameter_belief=_parameter_belief(params),
    )

    with pytest.raises(ValueError, match="direct NMPC actuation"):
        belief.compile_for_nmpc()
    _, report = belief.update(telemetry)

    assert report.applied
