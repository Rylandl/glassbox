from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

import glassbox.adaptation as adaptation_module
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


def _split_telemetry(telemetry, split: int):
    proposal = replace(
        telemetry,
        time_s=telemetry.time_s[: split + 1],
        states=telemetry.states[: split + 1],
        controls=telemetry.controls[:split],
        exogenous=telemetry.exogenous[: split + 1],
        observations=telemetry.observations[: split + 1],
    )
    validation = replace(
        telemetry,
        time_s=telemetry.time_s[split:],
        states=telemetry.states[split:],
        controls=telemetry.controls[split:],
        exogenous=telemetry.exogenous[split:],
        observations=telemetry.observations[split:],
    )
    return proposal, validation


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


def test_update_uses_runtime_period_after_timestamp_period_validation() -> None:
    telemetry = generate_trajectory(seed=31, duration_s=0.4)
    belief = _shifted_belief(
        telemetry,
        covariance_scope=ErrorCovarianceScope.TOTAL_FORECAST,
    )
    perturbed_period_s = belief.runtime_spec.sample_period_s + 5e-13
    perturbed_time = np.arange(len(telemetry.time_s)) * perturbed_period_s
    perturbed = replace(telemetry, time_s=perturbed_time)

    reference_updated, reference_report = belief.update(telemetry)
    perturbed_updated, perturbed_report = belief.update(perturbed)

    assert reference_report.applied
    assert perturbed_report.applied
    assert (
        perturbed_report.update_horizon_s
        == perturbed_report.update_horizon_steps * belief.runtime_spec.sample_period_s
    )
    assert perturbed_report.update_horizon_s == reference_report.update_horizon_s
    np.testing.assert_array_equal(
        structured_parameter_vector(perturbed_updated.params),
        structured_parameter_vector(reference_updated.params),
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
    updated, report = belief.commit_update(
        proposal,
        validation,
        validation_control_history=validation.controls[0:1],
    )
    assert updated is belief
    assert not report.applied
    assert report.validation_performed
    assert "did not improve disjoint validation" in report.reason


def test_validation_without_actuator_context_fails_closed() -> None:
    telemetry = generate_trajectory(seed=11, duration_s=0.4)
    belief = _shifted_belief(
        telemetry,
        covariance_scope=ErrorCovarianceScope.TOTAL_FORECAST,
    )
    proposal_telemetry, validation_telemetry = _split_telemetry(telemetry, 10)
    proposal, _ = belief.propose_update(proposal_telemetry)

    assert proposal is not None
    unchanged, report = belief.commit_update(proposal, validation_telemetry)

    assert unchanged is belief
    assert not report.applied
    assert not report.validation_performed
    assert "actuator command context" in report.reason


def test_transaction_carries_actuator_history_across_validation_split() -> None:
    telemetry = generate_trajectory(seed=11, duration_s=0.4)
    belief = _shifted_belief(
        telemetry,
        covariance_scope=ErrorCovarianceScope.TOTAL_FORECAST,
    )
    proposal_telemetry, validation_telemetry = _split_telemetry(telemetry, 10)
    proposal, _ = belief.propose_update(proposal_telemetry)

    assert proposal is not None
    explicit, explicit_report = belief.commit_update(
        proposal,
        validation_telemetry,
        validation_control_history=proposal_telemetry.controls,
    )
    automatic, automatic_report = belief.update(telemetry)
    context, rejected = adaptation_module._preflight(
        belief,
        validation_telemetry,
        preceding_control_history=proposal_telemetry.controls,
    )

    assert rejected is None
    assert context is not None
    np.testing.assert_allclose(
        context.windows[0].control_history[-len(proposal_telemetry.controls) :],
        proposal_telemetry.controls,
        rtol=1e-6,
        atol=1e-7,
    )
    assert explicit_report.applied
    assert automatic_report.applied
    np.testing.assert_allclose(
        structured_parameter_vector(automatic.params),
        structured_parameter_vector(explicit.params),
    )
    assert automatic_report.normalized_validation_rms_before == pytest.approx(
        explicit_report.normalized_validation_rms_before
    )
    assert automatic_report.normalized_validation_rms_after == pytest.approx(
        explicit_report.normalized_validation_rms_after
    )
    assert automatic_report.actuator_context_sample_count == len(
        proposal_telemetry.controls
    )
    assert automatic_report.actuator_context_fingerprint is not None


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


def test_proposal_owns_immutable_arrays_and_verifies_its_trust_step() -> None:
    telemetry = generate_trajectory(seed=37, duration_s=0.2)
    belief = _shifted_belief(
        telemetry,
        covariance_scope=ErrorCovarianceScope.CONDITIONAL_INNOVATION,
    )

    proposal, _ = belief.propose_update(telemetry)

    assert proposal is not None
    assert not np.shares_memory(
        proposal.base_parameter_covariance,
        belief.parameter_belief.covariance,
    )
    assert not proposal.base_parameter_vector.flags.writeable
    assert not proposal.base_parameter_covariance.flags.writeable
    assert not proposal.candidate_parameter_vector.flags.writeable
    assert not proposal.normalized_information_matrix.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        proposal.candidate_parameter_vector[0] = proposal.base_parameter_vector[0]
    with pytest.raises(ValueError, match="trust-region evidence is inconsistent"):
        replace(
            proposal,
            candidate_parameter_vector=(
                proposal.base_parameter_vector
                + 0.5
                * (proposal.candidate_parameter_vector - proposal.base_parameter_vector)
            ),
        )


def test_shifted_timestamp_replay_is_not_disjoint_validation() -> None:
    telemetry = generate_trajectory(seed=38, duration_s=0.2)
    belief = _shifted_belief(
        telemetry,
        covariance_scope=ErrorCovarianceScope.CONDITIONAL_INNOVATION,
    )
    proposal, _ = belief.propose_update(telemetry)
    replay = replace(telemetry, time_s=telemetry.time_s + 100.0)

    assert proposal is not None
    unchanged, report = belief.commit_update(proposal, replay)

    assert unchanged is belief
    assert not report.validation_performed
    assert "overlaps proposal transitions" in report.reason


def test_validation_rechecks_candidate_rollout_support(monkeypatch) -> None:
    telemetry = generate_trajectory(seed=39, duration_s=0.4)
    belief = _shifted_belief(
        telemetry,
        covariance_scope=ErrorCovarianceScope.CONDITIONAL_INNOVATION,
    )
    proposal_telemetry, validation_telemetry = _split_telemetry(telemetry, 10)
    proposal, _ = belief.propose_update(proposal_telemetry)

    assert proposal is not None
    monkeypatch.setattr(
        adaptation_module,
        "_candidate_rollouts_supported",
        lambda *_args, **_kwargs: False,
    )
    unchanged, report = belief.commit_update(
        proposal,
        validation_telemetry,
        validation_control_history=proposal_telemetry.controls,
    )

    assert unchanged is belief
    assert report.validation_performed
    assert "validation rollouts left" in report.reason


def test_proposal_is_tied_to_full_belief_and_target_specification() -> None:
    telemetry = generate_trajectory(seed=40, duration_s=0.4)
    belief = _shifted_belief(
        telemetry,
        covariance_scope=ErrorCovarianceScope.CONDITIONAL_INNOVATION,
    )
    proposal_telemetry, validation_telemetry = _split_telemetry(telemetry, 10)
    proposal, _ = belief.propose_update(proposal_telemetry)

    assert proposal is not None
    recalibrated = replace(
        belief,
        predictive_error=replace(
            belief.predictive_error,
            source="independently_recalibrated",
        ),
    )
    unchanged, revision_report = recalibrated.commit_update(
        proposal,
        validation_telemetry,
    )
    assert unchanged is recalibrated
    assert "no longer matches" in revision_report.reason

    changed_vehicle = replace(
        validation_telemetry,
        spec=replace(
            validation_telemetry.spec,
            vehicle=replace(
                validation_telemetry.spec.vehicle,
                configuration_id="different-validation-target",
            ),
        ),
    )
    unchanged, target_report = belief.commit_update(proposal, changed_vehicle)
    assert unchanged is belief
    assert "target specification changed" in target_report.reason


def test_commit_derives_covariance_information_from_validation_evidence() -> None:
    telemetry = generate_trajectory(seed=41, duration_s=0.4)
    belief = _shifted_belief(
        telemetry,
        covariance_scope=ErrorCovarianceScope.CONDITIONAL_INNOVATION,
    )
    proposal_telemetry, validation_telemetry = _split_telemetry(telemetry, 10)
    proposal, _ = belief.propose_update(proposal_telemetry)

    assert proposal is not None
    exaggerated = replace(
        proposal,
        normalized_information_matrix=(
            1e12 * np.eye(len(proposal.normalized_information_matrix))
        ),
    )
    expected, expected_report = belief.commit_update(
        proposal,
        validation_telemetry,
        validation_control_history=proposal_telemetry.controls,
    )
    actual, actual_report = belief.commit_update(
        exaggerated,
        validation_telemetry,
        validation_control_history=proposal_telemetry.controls,
    )

    assert expected_report.applied
    assert actual_report.applied
    np.testing.assert_array_equal(
        actual.parameter_belief.covariance,
        expected.parameter_belief.covariance,
    )
    assert (
        actual_report.realized_local_information_gain_nats
        == expected_report.realized_local_information_gain_nats
    )
