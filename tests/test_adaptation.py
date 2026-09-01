from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp
import numpy as np

from glassbox.belief import (
    DynamicsBelief,
    EmpiricalErrorSample,
    EmpiricalHorizonPredictiveError,
    ErrorCovarianceScope,
    LocalGaussianParameterBelief,
    structured_parameter_names,
    structured_parameter_vector,
    with_structured_parameter_vector,
)
from glassbox.runtime import ModelValidityEnvelope, runtime_spec_from_trajectory
from glassbox.synthetic import generate_trajectory, true_parameters


def _error_model(
    horizon_s: float,
    *,
    covariance_scope: ErrorCovarianceScope,
) -> EmpiricalHorizonPredictiveError:
    errors = 0.05 * np.concatenate((np.eye(12), -np.eye(12)), axis=0)
    samples = (
        EmpiricalErrorSample(errors, "group-a", "flight-a"),
        EmpiricalErrorSample(errors, "group-b", "flight-b"),
    )
    return EmpiricalHorizonPredictiveError.from_samples(
        {horizon_s: samples},
        covariance_scope=covariance_scope,
    )


def _shifted_belief(
    telemetry,
    *,
    covariance_scope: ErrorCovarianceScope,
    runtime_spec=None,
) -> DynamicsBelief:
    params = true_parameters()
    vector = np.asarray(structured_parameter_vector(params)).copy()
    vector[0] += 0.25
    nominal = with_structured_parameter_vector(params, jnp.asarray(vector))
    covariance = np.zeros((len(vector), len(vector)))
    covariance[0, 0] = 0.16
    return DynamicsBelief(
        params=nominal,
        input_spec=telemetry.spec,
        runtime_spec=(
            runtime_spec_from_trajectory(telemetry)
            if runtime_spec is None
            else runtime_spec
        ),
        predictive_error=_error_model(
            0.1,
            covariance_scope=covariance_scope,
        ),
        parameter_belief=LocalGaussianParameterBelief(
            parameter_names=structured_parameter_names(nominal),
            covariance=covariance,
            source="test_parameter_evidence",
            evidence_count=2,
            effective_sample_count=2.0,
        ),
    )


def test_total_forecast_error_does_not_double_count_parameter_spread() -> None:
    telemetry = generate_trajectory(seed=30, duration_s=0.3)
    belief = _shifted_belief(
        telemetry,
        covariance_scope=ErrorCovarianceScope.TOTAL_FORECAST,
    )

    assessment = belief.compile_for_nmpc().assess_plan(
        jnp.asarray(telemetry.states[0]),
        jnp.asarray(telemetry.controls[:5]),
    )
    prediction = assessment.prediction

    np.testing.assert_allclose(
        prediction.tangent_covariance,
        prediction.empirical_error_tangent_covariance,
    )
    assert np.max(prediction.parameter_tangent_covariance) > 0.0
    assert not prediction.parameter_covariance_combined_with_empirical_error
    assert not assessment.information_available
    assert "conditional innovation" in assessment.information_unavailable_reason


def test_total_forecast_error_can_move_mean_without_contracting_covariance() -> None:
    telemetry = generate_trajectory(seed=31, duration_s=0.4)
    belief = _shifted_belief(
        telemetry,
        covariance_scope=ErrorCovarianceScope.TOTAL_FORECAST,
    )

    updated, report = belief.update(telemetry)

    assert report.applied
    assert not report.covariance_updated
    assert report.realized_local_information_gain_nats is None
    np.testing.assert_array_equal(
        updated.parameter_belief.covariance,
        belief.parameter_belief.covariance,
    )
    assert not np.array_equal(
        structured_parameter_vector(updated.params),
        structured_parameter_vector(belief.params),
    )


def test_update_rejects_horizon_longer_than_predictive_error_support() -> None:
    telemetry = generate_trajectory(
        seed=32,
        duration_s=0.4,
        dt_s=0.1,
    )
    belief = _shifted_belief(
        telemetry,
        covariance_scope=ErrorCovarianceScope.CONDITIONAL_INNOVATION,
    )
    belief = replace(
        belief,
        predictive_error=_error_model(
            0.05,
            covariance_scope=ErrorCovarianceScope.CONDITIONAL_INNOVATION,
        ),
    )

    updated, report = belief.update(telemetry)

    assert updated is belief
    assert not report.applied
    assert "longer than the shortest supported horizon" in report.reason
    assert report.update_horizon_s == 0.1


def test_update_rejects_telemetry_outside_validity_support() -> None:
    telemetry = generate_trajectory(seed=33, duration_s=0.4)
    belief = _shifted_belief(
        telemetry,
        covariance_scope=ErrorCovarianceScope.CONDITIONAL_INNOVATION,
    )
    states = telemetry.states.copy()
    states[:, 3] += 100.0
    unsupported = replace(telemetry, states=states)

    updated, report = belief.update(unsupported)

    assert updated is belief
    assert not report.applied
    assert report.maximum_validity_utilization > 1.0
    assert "outside the learned validity envelope" in report.reason


def test_failed_disjoint_validation_rolls_back_proposal() -> None:
    proposal_telemetry = generate_trajectory(seed=34, duration_s=0.2)
    broad_runtime = replace(
        runtime_spec_from_trajectory(proposal_telemetry),
        validity_envelope=ModelValidityEnvelope(
            body_velocity_center_m_s=(0.0, 0.0, 0.0),
            body_velocity_half_width_m_s=(100.0, 100.0, 100.0),
            angular_velocity_center_rad_s=(0.0, 0.0, 0.0),
            angular_velocity_half_width_rad_s=(100.0, 100.0, 100.0),
        ),
    )
    belief = _shifted_belief(
        proposal_telemetry,
        covariance_scope=ErrorCovarianceScope.CONDITIONAL_INNOVATION,
        runtime_spec=broad_runtime,
    )
    validation = generate_trajectory(
        seed=35,
        duration_s=0.2,
        params=belief.params,
    )

    proposal, proposal_report = belief.propose_update(proposal_telemetry)

    assert proposal is not None
    assert proposal_report.proposal_available
    unchanged, overlap_report = belief.commit_update(
        proposal,
        proposal_telemetry,
    )
    assert unchanged is belief
    assert not overlap_report.validation_performed
    assert "overlaps proposal transitions" in overlap_report.reason
    updated, report = belief.commit_update(proposal, validation)
    assert updated is belief
    assert not report.applied
    assert report.validation_performed
    assert "did not improve disjoint validation" in report.reason


def test_rank_zero_conditional_error_cannot_create_information() -> None:
    telemetry = generate_trajectory(seed=36, duration_s=0.3)
    belief = _shifted_belief(
        telemetry,
        covariance_scope=ErrorCovarianceScope.CONDITIONAL_INNOVATION,
    )
    zero_errors = np.zeros((2, 12))
    belief = replace(
        belief,
        predictive_error=EmpiricalHorizonPredictiveError.from_samples(
            {
                0.1: (
                    EmpiricalErrorSample(zero_errors, "group-a", "flight-a"),
                    EmpiricalErrorSample(zero_errors, "group-b", "flight-b"),
                )
            },
            covariance_scope=ErrorCovarianceScope.CONDITIONAL_INNOVATION,
        ),
    )

    assessment = belief.compile_for_nmpc().assess_plan(
        jnp.asarray(telemetry.states[0]),
        jnp.asarray(telemetry.controls[:5]),
    )
    updated, report = belief.update(telemetry)

    assert not assessment.information_available
    assert "rank zero" in assessment.information_unavailable_reason
    assert updated is belief
    assert not report.applied
    assert "no supported direction" in report.reason
