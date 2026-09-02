from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

import glassbox.belief.adaptation as adaptation_module
from glassbox.belief.belief import (
    DynamicsBelief,
    EmpiricalErrorSample,
    EmpiricalHorizonPredictiveError,
    ErrorCovarianceScope,
    LocalGaussianParameterBelief,
    structured_parameter_names,
    structured_parameter_vector,
    with_structured_parameter_vector,
)
from glassbox.core.data import duration_to_steps
from glassbox.core.evaluation import (
    rigid_body_tangent_errors,
    windowed_rollout_evaluation,
)
from glassbox.core.runtime import ModelValidityEnvelope, runtime_spec_from_trajectory
from glassbox.core.synthetic import generate_trajectory, true_parameters


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


def test_validation_without_actuator_context_fails_closed(
    quadrotor_trajectory_seed11_dur0_4s,
) -> None:
    telemetry = quadrotor_trajectory_seed11_dur0_4s
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


def test_transaction_carries_actuator_history_across_validation_split(
    quadrotor_trajectory_seed11_dur0_4s,
) -> None:
    telemetry = quadrotor_trajectory_seed11_dur0_4s
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


def _concentrated_proposal(
    *,
    rank: int,
    sigma: float,
) -> adaptation_module.BeliefUpdateProposal:
    """Build a proposal that moves exactly one whitened prior coordinate."""

    candidate = np.zeros(rank)
    candidate[0] = sigma
    return adaptation_module.BeliefUpdateProposal(
        base_parameter_vector=np.zeros(rank),
        base_parameter_covariance=np.eye(rank),
        candidate_parameter_vector=candidate,
        normalized_information_matrix=np.eye(rank),
        covariance_scope=ErrorCovarianceScope.CONDITIONAL_INNOVATION,
        base_update_count=0,
        update_horizon_s=0.1,
        update_horizon_steps=1,
        proposal_window_count=4,
        normalized_innovation_rms_before=2.0,
        normalized_innovation_rms_after=1.0,
        normalized_innovation_improvement=10.0,
        normalized_innovation_improvement_margin=1.0,
        prior_standardized_step_rms=float(np.sqrt(np.mean(np.square(candidate)))),
        prior_standardized_step_max=sigma,
        proposal_step_fraction=1.0,
        maximum_validity_utilization=0.5,
        source_group="test-trust-region",
        evidence_transition_hashes=("a" * 64,),
        base_belief_fingerprint="b" * 64,
        target_spec_fingerprint="c" * 64,
    )


def test_trust_region_bounds_each_prior_coordinate_not_their_mean() -> None:
    concentrated = np.zeros(22)
    concentrated[7] = 4.69
    diffuse = np.full(22, 0.2)

    concentrated_fraction = adaptation_module._bounded_local_step_fraction(concentrated)
    diffuse_fraction = adaptation_module._bounded_local_step_fraction(diffuse)
    bounded = concentrated_fraction * concentrated

    # The step keeps its direction and lands exactly on the one-sigma bound.
    assert np.max(np.abs(bounded)) == pytest.approx(1.0)
    assert bounded[7] == pytest.approx(1.0)
    assert np.count_nonzero(bounded) == 1
    # A diffuse step already inside the bound is untouched, even though its
    # movement is spread across every supported direction.
    assert diffuse_fraction == 1.0
    np.testing.assert_array_equal(diffuse_fraction * diffuse, diffuse)

    # A rank-22 step of 4.69 sigma in one direction has an RMS just under one,
    # so the retired root-mean-square bound accepted it.
    with pytest.raises(ValueError, match="exceeds the local trust region"):
        _concentrated_proposal(rank=22, sigma=4.69)
    accepted = _concentrated_proposal(rank=22, sigma=1.0)
    assert accepted.prior_standardized_step_max == pytest.approx(1.0)
    assert accepted.prior_standardized_step_rms == pytest.approx(1.0 / np.sqrt(22.0))


def test_proposal_reports_the_bounded_coordinate_and_its_spread() -> None:
    telemetry = generate_trajectory(seed=37, duration_s=0.4)
    belief = _shifted_belief(
        telemetry,
        covariance_scope=ErrorCovarianceScope.CONDITIONAL_INNOVATION,
    )
    proposal_telemetry, validation_telemetry = _split_telemetry(telemetry, 10)

    proposal, proposal_report = belief.propose_update(proposal_telemetry)

    assert proposal is not None
    assert 0.0 < proposal.prior_standardized_step_max <= 1.0
    assert proposal.prior_standardized_step_rms <= proposal.prior_standardized_step_max
    assert proposal_report.prior_standardized_step_max == pytest.approx(
        proposal.prior_standardized_step_max
    )

    _, commit_report = belief.commit_update(
        proposal,
        validation_telemetry,
        validation_control_history=proposal_telemetry.controls,
    )

    assert commit_report.applied
    assert commit_report.to_dict()["prior_standardized_step_max"] == pytest.approx(
        commit_report.prior_standardized_step_max
    )
    assert 0.0 < commit_report.prior_standardized_step_max <= 1.0


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


def _self_calibrated_belief(
    *,
    offset: float,
    calibration_seed: int,
    calibration_duration_s: float = 1.0,
) -> DynamicsBelief:
    """Build a belief whose error evidence was measured around its own model.

    The fitted bias then absorbs the parameter offset, which is exactly the
    situation in which a commit that stales the bias can make the runtime
    forecast worse than doing nothing.
    """

    params = true_parameters()
    vector = np.asarray(structured_parameter_vector(params)).copy()
    vector[0] += offset
    nominal = with_structured_parameter_vector(params, jnp.asarray(vector))
    covariance = np.zeros((len(vector), len(vector)))
    covariance[0, 0] = 0.16
    calibration = generate_trajectory(
        seed=calibration_seed,
        duration_s=calibration_duration_s,
    )
    shell = DynamicsBelief(
        params=nominal,
        input_spec=calibration.spec,
        runtime_spec=runtime_spec_from_trajectory(calibration),
        parameter_belief=LocalGaussianParameterBelief(
            parameter_names=structured_parameter_names(nominal),
            covariance=covariance,
            source="test_parameter_evidence",
            evidence_count=2,
            effective_sample_count=2.0,
        ),
    )
    return adaptation_module.recalibrate_predictive_error(
        shell,
        calibration,
        horizons_s=(0.1,),
    )


def _runtime_endpoint_errors(
    belief: DynamicsBelief,
    trajectory,
    *,
    horizon_steps: int = 5,
    window_count: int = 4,
) -> np.ndarray:
    """Score the forecast the controller actually consumes, bias included."""

    runtime = belief.compile_for_nmpc()
    errors = []
    for start in range(0, horizon_steps * window_count, horizon_steps):
        forecast = runtime.rollout(
            jnp.asarray(trajectory.states[start]),
            jnp.asarray(trajectory.controls[start : start + horizon_steps]),
            command_history=jnp.asarray(trajectory.controls[: start + 1]),
        )
        errors.append(
            float(
                np.linalg.norm(
                    rigid_body_tangent_errors(
                        np.asarray(forecast.mean_states[-1]),
                        trajectory.states[start + horizon_steps],
                    )
                )
            )
        )
    return np.asarray(errors)


def _noisy(trajectory, seed: int):
    """Add i.i.d. observation noise without leaving the quaternion manifold."""

    rng = np.random.default_rng(10_000 + seed)
    states = np.array(trajectory.states, dtype=np.float64, copy=True)
    count = len(states)
    states[:, 0:3] += rng.normal(0.0, 0.002, size=(count, 3))
    states[:, 3:6] += rng.normal(0.0, 0.010, size=(count, 3))
    states[:, 10:13] += rng.normal(0.0, 0.010, size=(count, 3))
    rotation = rng.normal(0.0, 0.002, size=(count, 3))
    angle = np.linalg.norm(rotation, axis=1, keepdims=True)
    axis = np.where(angle > 0.0, rotation / np.maximum(angle, 1e-30), 0.0)
    delta = np.concatenate((np.cos(angle / 2.0), axis * np.sin(angle / 2.0)), axis=1)
    left = states[:, 6:10]
    w1, x1, y1, z1 = left.T
    w2, x2, y2, z2 = delta.T
    product = np.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        axis=1,
    )
    states[:, 6:10] = product / np.linalg.norm(product, axis=1, keepdims=True)
    return replace(trajectory, states=states)


def test_committed_update_never_worsens_the_runtime_forecast(
    quadrotor_trajectory_seed11_dur0_4s,
) -> None:
    belief = _self_calibrated_belief(offset=0.25, calibration_seed=1)
    telemetry = quadrotor_trajectory_seed11_dur0_4s
    evaluation = generate_trajectory(seed=2, duration_s=0.5)

    updated, report = belief.update(telemetry)
    incumbent = _runtime_endpoint_errors(belief, evaluation)
    committed = _runtime_endpoint_errors(updated, evaluation)

    assert report.validation_scoring == (
        "candidate_uncorrected_vs_nominal_bias_corrected"
    )
    # The acceptance criterion scores the candidate the way the runtime will
    # fly it, so a commit can never trade a good corrected forecast for a worse
    # uncorrected one.
    assert np.all(committed <= incumbent + 1e-12)
    # This specific belief is corrected by evidence fitted to its own offset:
    # no bounded parameter move beats it, so the transaction fails closed.
    assert not report.applied
    assert updated is belief
    np.testing.assert_array_equal(committed, incumbent)


def test_null_acceptance_rate_stays_within_one_in_twenty() -> None:
    truth = true_parameters()
    names = structured_parameter_names(truth)
    covariance = np.zeros((len(names), len(names)))
    covariance[0, 0] = 0.16
    calibration = _noisy(generate_trajectory(seed=901, duration_s=2.0), 901)
    permissive = replace(
        runtime_spec_from_trajectory(calibration),
        validity_envelope=ModelValidityEnvelope(
            body_velocity_center_m_s=(0.0, 0.0, 0.0),
            body_velocity_half_width_m_s=(100.0, 100.0, 100.0),
            angular_velocity_center_rad_s=(0.0, 0.0, 0.0),
            angular_velocity_half_width_rad_s=(100.0, 100.0, 100.0),
        ),
    )
    shell = DynamicsBelief(
        params=truth,
        input_spec=calibration.spec,
        runtime_spec=permissive,
        parameter_belief=LocalGaussianParameterBelief(
            parameter_names=names,
            covariance=covariance,
            source="test_parameter_evidence",
            evidence_count=2,
            effective_sample_count=2.0,
        ),
    )
    belief = adaptation_module.recalibrate_predictive_error(
        shell,
        calibration,
        horizons_s=(0.1,),
    )

    seeds = tuple(range(20))
    commits = 0
    for seed in seeds:
        telemetry = _noisy(generate_trajectory(seed=seed, duration_s=0.4), seed)
        updated, report = belief.update(telemetry)
        commits += bool(report.applied)
        if not report.applied:
            assert updated is belief

    assert commits <= 0.05 * len(seeds)


def test_endpoint_error_evidence_matches_the_previous_inline_recipe() -> None:
    params = true_parameters()
    horizons = (0.1, 0.2, 5.0)
    flights = ((101, "group-a"), (102, "group-b"))
    expected: dict[float, list[EmpiricalErrorSample]] = {}
    expected_metrics: list[dict] = []
    actual: dict[float, list[EmpiricalErrorSample]] = {}
    actual_metrics: list[dict] = []
    for seed, group in flights:
        trajectory = generate_trajectory(seed=seed, duration_s=0.6)
        path = f"synthetic-{seed}.npz"
        for seconds in horizons:
            steps = duration_to_steps(seconds, trajectory.nominal_dt_s)
            if steps > len(trajectory.controls):
                continue
            metrics, endpoint_errors = windowed_rollout_evaluation(
                params,
                trajectory,
                horizon_steps=steps,
                stride_steps=steps,
            )
            metrics["requested_horizon_s"] = seconds
            metrics["horizon_steps"] = steps
            expected_metrics.append(metrics)
            expected.setdefault(seconds, []).append(
                EmpiricalErrorSample(
                    endpoint_errors,
                    source_group=group,
                    trajectory_id=path,
                )
            )
        for evidence in adaptation_module.endpoint_error_evidence_by_horizon(
            params,
            trajectory,
            horizons_s=horizons,
            source_group=group,
            trajectory_id=path,
        ):
            actual_metrics.append(evidence.window_metrics)
            actual.setdefault(evidence.horizon_s, []).append(evidence.sample)

    assert actual_metrics == expected_metrics
    assert sorted(actual) == sorted(expected) == [0.1, 0.2]
    assert (
        EmpiricalHorizonPredictiveError.from_samples(actual).to_dict()
        == EmpiricalHorizonPredictiveError.from_samples(expected).to_dict()
    )
