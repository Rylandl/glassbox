from __future__ import annotations

import json
from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from glassbox.belief import (
    DynamicsBelief,
    EmpiricalErrorSample,
    EmpiricalHorizonPredictiveError,
    ErrorCovarianceScope,
    LocalGaussianParameterBelief,
    LocalParameterInformation,
    apply_tangent_correction,
    structured_parameter_names,
    structured_parameter_vector,
    with_structured_parameter_vector,
)
from glassbox.data import trajectory_windows
from glassbox.dynamics import ResidualDynamicsParams, initial_residual_parameters
from glassbox.evaluation import rigid_body_tangent_errors
from glassbox.fixedwing_synthetic import (
    generate_fixed_wing_trajectory,
    true_fixed_wing_parameters,
)
from glassbox.parameter_evidence import (
    estimate_local_parameter_information,
    fitted_structured_parameter_mask,
)
from glassbox.parameter_prior import StructuredParameterPrior
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


def _nonsingular_error_model(
    scale: float = 0.02,
    *,
    covariance_scope: ErrorCovarianceScope = ErrorCovarianceScope.TOTAL_FORECAST,
) -> EmpiricalHorizonPredictiveError:
    errors = scale * np.concatenate((np.eye(12), -np.eye(12)), axis=0)
    samples = (
        EmpiricalErrorSample(errors, "group-a", "flight-a"),
        EmpiricalErrorSample(errors, "group-b", "flight-b"),
    )
    return EmpiricalHorizonPredictiveError.from_samples(
        {0.1: samples, 0.2: samples},
        covariance_scope=covariance_scope,
    )


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
    residual = initial_residual_parameters(true_fixed_wing_parameters(), hidden_units=3)
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
        predictive_error=_nonsingular_error_model(
            covariance_scope=ErrorCovarianceScope.CONDITIONAL_INNOVATION
        ),
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


def test_local_parameter_information_preserves_unresolved_directions(tmp_path) -> None:
    trajectory = generate_trajectory(seed=9, duration_s=0.3)
    params = true_parameters()
    names = structured_parameter_names(params)
    center = np.asarray(structured_parameter_vector(params))
    information = np.zeros((len(names), len(names)))
    information[0, 0] = 4.0
    group_scores = np.zeros((2, len(names)))
    group_scores[:, 0] = 0.1
    evidence = LocalParameterInformation(
        parameter_names=names,
        center=center,
        information_matrix=information,
        parameter_scale=np.maximum(np.abs(center), 1.0),
        fitted_parameter_mask=np.ones(len(names), dtype=bool),
        horizons_s=(0.1,),
        window_count_by_horizon=(8,),
        residual_precision_rank_by_horizon=(12,),
        group_labels=("group-a", "group-b"),
        group_score_vectors=group_scores,
        independent_group_count=2,
        trajectory_count=2,
        rank_relative_tolerance=1e-5,
    )
    belief = DynamicsBelief(
        params=params,
        input_spec=trajectory.spec,
        runtime_spec=runtime_spec_from_trajectory(trajectory),
        predictive_error=_nonsingular_error_model(),
        parameter_evidence=evidence,
    )
    path = tmp_path / "parameter-evidence.json"

    belief.save(path)
    restored = DynamicsBelief.load(path)

    assert isinstance(restored.parameter_evidence, LocalParameterInformation)
    assert restored.parameter_evidence.numerical_rank == 1
    assert restored.parameter_evidence.unresolved_fitted_direction_count == (
        len(names) - 1
    )
    assert restored.parameter_evidence.unresolved_direction_basis.shape == (
        len(names),
        len(names) - 1,
    )
    assert restored.parameter_evidence.score_vector[0] == pytest.approx(0.2)
    assert not restored.parameter_belief.uncertainty_available


def test_local_information_conditions_complete_prior_without_collapsing_nullspace() -> (
    None
):
    trajectory = generate_trajectory(seed=12, duration_s=0.3)
    params = true_parameters()
    names = structured_parameter_names(params)
    local_center = np.asarray(structured_parameter_vector(params))
    information = np.zeros((len(names), len(names)))
    information[0, 0] = 4.0
    group_scores = np.zeros((2, len(names)))
    group_scores[:, 0] = 0.1
    evidence = LocalParameterInformation(
        parameter_names=names,
        center=local_center,
        information_matrix=information,
        parameter_scale=np.maximum(np.abs(local_center), 1.0),
        fitted_parameter_mask=np.ones(len(names), dtype=bool),
        horizons_s=(0.1,),
        window_count_by_horizon=(8,),
        residual_precision_rank_by_horizon=(12,),
        group_labels=("group-a", "group-b"),
        group_score_vectors=group_scores,
        independent_group_count=2,
        trajectory_count=2,
        rank_relative_tolerance=1e-5,
        covariance_scope=ErrorCovarianceScope.CONDITIONAL_INNOVATION,
    )
    fitted = DynamicsBelief(
        params=params,
        input_spec=trajectory.spec,
        runtime_spec=runtime_spec_from_trajectory(trajectory),
        predictive_error=_nonsingular_error_model(),
        parameter_evidence=evidence,
    )
    prior_center = local_center.copy()
    prior_center[0] += 0.5
    prior_center[1] += 0.75
    contracts = tuple(
        (channel.role, channel.semantic, channel.unit, channel.frame)
        for channel in trajectory.spec.controls
    )
    prior = StructuredParameterPrior(
        parameter_names=names,
        mean=prior_center,
        between_member_covariance=np.zeros((len(names), len(names))),
        within_member_covariance=np.zeros((len(names), len(names))),
        completion_covariance=np.eye(len(names)),
        natural_scale=np.ones(len(names)),
        parameter_control_dependencies=(None,) * len(names),
        parameter_member_counts=(4,) * len(names),
        member_labels=("fleet-a", "fleet-b", "fleet-c", "fleet-d"),
        member_control_roles=(trajectory.spec.control_roles,) * 4,
        within_member_covariance_count=0,
        state_schema=trajectory.spec.state_schema,
        vehicle_family=trajectory.spec.vehicle.family,
        control_contracts=contracts,
        source="fleet_hierarchical_prior",
    )

    conditioned = fitted.condition_parameter_prior(prior)
    posterior_center = np.asarray(structured_parameter_vector(conditioned.params))

    assert isinstance(conditioned.parameter_belief, LocalGaussianParameterBelief)
    assert posterior_center[0] < prior_center[0]
    assert posterior_center[0] > local_center[0]
    assert posterior_center[0] == pytest.approx(
        local_center[0] + (0.5 - 0.2) / 5.0,
        abs=1e-6,
    )
    assert posterior_center[1] == pytest.approx(prior_center[1])
    assert conditioned.parameter_belief.covariance[0, 0] < 1.0
    assert conditioned.parameter_belief.covariance[1, 1] == pytest.approx(1.0)
    assert not conditioned.predictive_error_current
    assert conditioned.parameter_evidence is evidence

    total_error_evidence = replace(
        evidence,
        covariance_scope=ErrorCovarianceScope.TOTAL_FORECAST,
    )
    mean_only = replace(
        fitted,
        parameter_evidence=total_error_evidence,
    ).condition_parameter_prior(prior)
    np.testing.assert_allclose(
        structured_parameter_vector(mean_only.params),
        structured_parameter_vector(conditioned.params),
    )
    np.testing.assert_allclose(
        mean_only.parameter_belief.covariance,
        prior.covariance,
    )
    assert not mean_only.provenance["parameter_prior_conditioning"][
        "parameter_covariance_updated"
    ]

    incompatible_contracts = (
        (contracts[0][0], contracts[0][1], "rad", contracts[0][3]),
        *contracts[1:],
    )
    with pytest.raises(ValueError, match="incompatible control semantics"):
        fitted.condition_parameter_prior(
            replace(prior, control_contracts=incompatible_contracts)
        )


def test_grouped_rollout_information_uses_only_fitted_structured_coordinates() -> None:
    trajectories = tuple(
        replace(
            generate_trajectory(seed=seed, duration_s=0.3),
            labels={"source_group": group},
        )
        for seed, group in ((1, "group-a"), (2, "group-b"))
    )
    windows = trajectory_windows(
        trajectories,
        horizon=5,
        stride=5,
        trajectory_groups=("group-a", "group-b"),
    )
    params = true_parameters()
    fitted_mask = fitted_structured_parameter_mask(
        params,
        instantaneous_rotational_response=True,
        diagonal_angular_control=True,
    )

    evidence = estimate_local_parameter_information(
        params,
        (windows,),
        _nonsingular_error_model(),
        ("group-a", "group-b"),
        fitted_parameter_mask=fitted_mask,
    )

    assert isinstance(evidence, LocalParameterInformation)
    assert evidence.independent_group_count == 2
    assert evidence.window_count_by_horizon == (6,)
    assert evidence.fitted_parameter_count == 9
    assert 0 < evidence.numerical_rank <= evidence.fitted_parameter_count
    assert np.all(np.isfinite(evidence.information_matrix))
    assert evidence.group_score_vectors.shape == (2, len(evidence.parameter_names))
    assert np.allclose(evidence.information_matrix[~fitted_mask], 0.0)


def test_grouped_rollout_information_is_vehicle_family_generic() -> None:
    trajectories = tuple(
        generate_fixed_wing_trajectory(seed=seed, duration_s=0.3) for seed in (1, 2)
    )
    windows = trajectory_windows(
        trajectories,
        horizon=5,
        stride=5,
        trajectory_groups=("fixedwing-a", "fixedwing-b"),
    )
    params = true_fixed_wing_parameters()
    fitted_mask = fitted_structured_parameter_mask(params)
    fixed_response_mask = fitted_structured_parameter_mask(
        params,
        fixed_response_time=True,
    )

    evidence = estimate_local_parameter_information(
        params,
        (windows,),
        _nonsingular_error_model(),
        ("fixedwing-a", "fixedwing-b"),
        fitted_parameter_mask=fitted_mask,
    )

    assert isinstance(evidence, LocalParameterInformation)
    assert evidence.fitted_parameter_count == len(structured_parameter_names(params))
    assert 0 < evidence.numerical_rank <= evidence.fitted_parameter_count
    assert np.count_nonzero(fixed_response_mask) == (
        evidence.fitted_parameter_count - 1
    )


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
        predictive_error=_nonsingular_error_model(
            scale=0.05,
            covariance_scope=ErrorCovarianceScope.CONDITIONAL_INNOVATION,
        ),
        parameter_belief=_parameter_belief(nominal, spread=0.4),
    )

    updated, report = belief.update(telemetry)

    assert report.applied
    assert report.used_window_count == 4
    assert report.proposal_window_count == 2
    assert report.validation_window_count == 2
    assert report.validation_performed
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
    stale_assessment = updated.compile_for_nmpc().assess_plan(
        jnp.asarray(telemetry.states[0]),
        jnp.asarray(telemetry.controls[:5]),
    )
    assert not stale_assessment.information_available
    assert "stale" in stale_assessment.information_unavailable_reason
    stale_prediction = stale_assessment.prediction
    assert updated.compile_for_nmpc().maximum_error_horizon_s is None
    np.testing.assert_allclose(
        stale_prediction.mean_states,
        stale_prediction.nominal_states,
    )
    np.testing.assert_array_equal(
        stale_prediction.empirical_error_tangent_covariance,
        np.zeros_like(stale_prediction.empirical_error_tangent_covariance),
    )
    assert stale_prediction.group_radius_quantiles is None
    assert stale_prediction.empirical_error_covariance_scope is None
    assert stale_prediction.parameter_uncertainty_available
    assert stale_prediction.uncertainty_available
    assert np.max(stale_prediction.parameter_tangent_covariance) > 0.0
    np.testing.assert_allclose(
        stale_prediction.tangent_covariance,
        stale_prediction.parameter_tangent_covariance,
    )

    updated_again, second_report = updated.update(telemetry)
    assert not second_report.applied
    assert "stale" in second_report.reason
    assert updated_again.parameter_belief.update_count == 1
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
    vector = np.asarray(structured_parameter_vector(params)).copy()
    vector[0] += 0.25
    nominal = with_structured_parameter_vector(params, jnp.asarray(vector))
    belief = DynamicsBelief(
        params=nominal,
        input_spec=physical_spec,
        runtime_spec=runtime_spec_from_trajectory(telemetry),
        predictive_error=_nonsingular_error_model(
            scale=0.05,
            covariance_scope=ErrorCovarianceScope.CONDITIONAL_INNOVATION,
        ),
        parameter_belief=_parameter_belief(nominal, spread=0.4),
    )

    with pytest.raises(ValueError, match="direct NMPC actuation"):
        belief.compile_for_nmpc()
    _, report = belief.update(telemetry)

    assert report.applied


def test_parameter_evidence_coerces_numpy_scalar_tolerance_to_json_native_float() -> None:
    params = true_parameters()
    names = structured_parameter_names(params)
    center = np.asarray(structured_parameter_vector(params))
    information = np.zeros((len(names), len(names)))
    information[0, 0] = 4.0
    group_scores = np.zeros((2, len(names)))
    group_scores[:, 0] = 0.1
    evidence = LocalParameterInformation(
        parameter_names=names,
        center=center,
        information_matrix=information,
        parameter_scale=np.maximum(np.abs(center), 1.0),
        fitted_parameter_mask=np.ones(len(names), dtype=bool),
        horizons_s=(0.1,),
        window_count_by_horizon=(8,),
        residual_precision_rank_by_horizon=(12,),
        group_labels=("group-a", "group-b"),
        group_score_vectors=group_scores,
        independent_group_count=2,
        trajectory_count=2,
        rank_relative_tolerance=np.float32(1e-5),
    )

    assert type(evidence.rank_relative_tolerance) is float
    payload = json.loads(json.dumps(evidence.to_dict()))
    assert payload["rank_relative_tolerance"] == pytest.approx(1e-5)

