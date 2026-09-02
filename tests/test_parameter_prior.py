from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from glassbox.belief.belief import (
    DynamicsBelief,
    LocalGaussianParameterBelief,
    structured_parameter_names,
    structured_parameter_vector,
    with_structured_parameter_vector,
)
from glassbox.belief.parameter_prior import StructuredParameterPrior
from glassbox.core.data import make_trajectory_spec
from glassbox.core.dynamics import FIXED_WING_CONTROL_NAMES
from glassbox.core.fixedwing_synthetic import (
    generate_fixed_wing_trajectory,
    true_fixed_wing_parameters,
)
from glassbox.core.runtime import runtime_spec_from_trajectory
from glassbox.core.synthetic import generate_trajectory, true_parameters


def _member_belief(trajectory, params, offset: np.ndarray) -> DynamicsBelief:
    center = np.asarray(structured_parameter_vector(params), dtype=np.float64)
    return DynamicsBelief(
        params=with_structured_parameter_vector(
            params,
            jnp.asarray(center + offset),
        ),
        input_spec=trajectory.spec,
        runtime_spec=runtime_spec_from_trajectory(trajectory),
    )


# Resolved lazily from the session-scoped conftest fixtures at test setup
# time (keyed by a plain string id) instead of building both rollouts
# eagerly inside `@pytest.mark.parametrize`, which used to run at import.
@pytest.fixture(params=("quadrotor", "fixedwing"))
def fleet_reference_case(request):
    if request.param == "quadrotor":
        trajectory = request.getfixturevalue("quadrotor_trajectory_seed1_dur0_2s")
        return trajectory, true_parameters()
    trajectory = request.getfixturevalue("fixedwing_trajectory_seed1_dur0_2s")
    return trajectory, true_fixed_wing_parameters()


def test_parameter_prior_separates_fleet_spread_from_full_rank_completion(
    fleet_reference_case,
    tmp_path,
) -> None:
    trajectory, params = fleet_reference_case
    size = len(np.asarray(structured_parameter_vector(params)))
    offsets = []
    for first, second in ((-0.2, 0.1), (0.0, -0.2), (0.2, 0.1)):
        offset = np.zeros(size)
        offset[:2] = (first, second)
        offsets.append(offset)
    members = tuple(_member_belief(trajectory, params, offset) for offset in offsets)

    prior = StructuredParameterPrior.from_beliefs(
        members,
        source="three_vehicle_reference",
        member_labels=("vehicle-a", "vehicle-b", "vehicle-c"),
    )

    assert prior.member_count == 3
    assert prior.empirical_rank == 2
    assert np.linalg.matrix_rank(prior.between_member_covariance) == 2
    assert np.linalg.matrix_rank(prior.completion_covariance) == size - 2
    normalized_total = (
        prior.covariance / prior.natural_scale[:, None] / prior.natural_scale[None, :]
    )
    assert np.min(np.linalg.eigvalsh(normalized_total)) > 0.0
    assert 0.0 < prior.completion_fraction_in_natural_coordinates < 1.0

    path = tmp_path / "fleet-prior.json"
    prior.save(path)
    restored = StructuredParameterPrior.load(path)
    np.testing.assert_allclose(restored.mean, prior.mean)
    np.testing.assert_allclose(
        restored.between_member_covariance,
        prior.between_member_covariance,
    )
    np.testing.assert_allclose(
        restored.completion_covariance,
        prior.completion_covariance,
    )
    payload = restored.to_dict()
    assert payload["semantics"]["posterior"] is False
    assert payload["semantics"]["calibrated_distribution"] is False
    assert payload["completion_policy"]["completed_dimension"] == size - 2

    initialized = restored.initialize_belief(members[0])
    np.testing.assert_allclose(
        structured_parameter_vector(initialized.params),
        restored.mean,
    )
    assert isinstance(initialized.parameter_belief, LocalGaussianParameterBelief)
    assert not initialized.parameter_evidence.available
    assert (
        initialized.provenance["parameter_prior_initialization"]["prior_empirical_rank"]
        == 2
    )


def test_parameter_prior_rejects_mixed_member_covariance_semantics(
    quadrotor_trajectory_seed2_dur0_2s,
) -> None:
    trajectory = quadrotor_trajectory_seed2_dur0_2s
    params = true_parameters()
    size = len(np.asarray(structured_parameter_vector(params)))
    first = _member_belief(trajectory, params, np.zeros(size))
    second = _member_belief(trajectory, params, np.full(size, 0.01))
    second = replace(
        second,
        parameter_belief=LocalGaussianParameterBelief(
            parameter_names=structured_parameter_names(params),
            covariance=np.eye(size),
            source="member_covariance",
            evidence_count=2,
            effective_sample_count=2.0,
        ),
    )

    with pytest.raises(ValueError, match="either all provide parameter covariance"):
        StructuredParameterPrior.from_beliefs(
            (first, second),
            source="mixed_reference",
        )


def test_parameter_prior_requires_fleet_coverage_for_target_controls() -> None:
    trajectory = generate_fixed_wing_trajectory(seed=3, duration_s=0.2)
    params = true_fixed_wing_parameters()
    size = len(np.asarray(structured_parameter_vector(params)))
    prior = StructuredParameterPrior.from_beliefs(
        (
            _member_belief(trajectory, params, np.zeros(size)),
            _member_belief(trajectory, params, np.full(size, 0.01)),
        ),
        source="flapless_reference",
    )
    flap_spec = make_trajectory_spec(
        FIXED_WING_CONTROL_NAMES + ("flap",),
        family="fixedwing",
        observation_source="simulator_truth",
        configuration_id="flap_equipped_target",
    )

    with pytest.raises(ValueError, match=r"no fleet evidence.*flap"):
        prior.validate_input_spec(flap_spec)


def test_configuration_specific_parameters_ignore_inapplicable_members() -> None:
    trajectory = generate_fixed_wing_trajectory(seed=4, duration_s=0.2)
    flap_spec = make_trajectory_spec(
        FIXED_WING_CONTROL_NAMES + ("flap",),
        family="fixedwing",
        observation_source="simulator_truth",
        configuration_id="flap_equipped_reference",
    )
    flap_trajectory = replace(
        trajectory,
        spec=flap_spec,
        controls=np.column_stack(
            (trajectory.controls, np.zeros(len(trajectory.controls)))
        ),
    )
    params = true_fixed_wing_parameters()
    names = structured_parameter_names(params)
    size = len(names)
    flap_indices = np.asarray(
        [
            index
            for index, name in enumerate(names)
            if name
            in {
                "log_flap_lift_accel_per_speed_sq",
                "log_flap_drag_accel_per_speed_sq",
                "flap_pitch_angular_accel_per_speed_sq",
                "flap_trim_unconstrained",
            }
        ]
    )
    flapless_first = np.zeros(size)
    flapless_second = np.zeros(size)
    flap_equipped = np.zeros(size)
    flapless_first[flap_indices] = -10.0
    flapless_second[flap_indices] = 10.0
    flap_equipped[flap_indices] = (0.2, 0.3, 0.4, 0.5)

    prior = StructuredParameterPrior.from_beliefs(
        (
            _member_belief(trajectory, params, flapless_first),
            _member_belief(trajectory, params, flapless_second),
            _member_belief(flap_trajectory, params, flap_equipped),
        ),
        source="mixed_flap_configuration_reference",
    )
    base = np.asarray(structured_parameter_vector(params))

    np.testing.assert_allclose(
        prior.mean[flap_indices],
        base[flap_indices] + flap_equipped[flap_indices],
    )
    assert set(np.asarray(prior.parameter_member_counts)[flap_indices]) == {1}
    core_indices = np.setdiff1d(np.arange(size), flap_indices)
    assert set(np.asarray(prior.parameter_member_counts)[core_indices]) == {3}
    np.testing.assert_allclose(
        prior.between_member_covariance[np.ix_(flap_indices, core_indices)],
        0.0,
    )
    prior.validate_input_spec(flap_spec)


def test_initialize_belief_stales_inherited_error_evidence_unless_asserted() -> None:
    from glassbox.belief.belief import (
        EmpiricalErrorSample,
        EmpiricalHorizonPredictiveError,
    )

    trajectory = generate_trajectory(seed=4, duration_s=0.2)
    params = true_parameters()
    size = len(np.asarray(structured_parameter_vector(params)))
    members = [
        _member_belief(trajectory, params, np.full(size, offset))
        for offset in (0.0, 0.02, -0.02)
    ]
    prior = StructuredParameterPrior.from_beliefs(members, source="test")
    errors = np.random.default_rng(0).normal(scale=0.02, size=(40, 12))
    shell = replace(
        members[0],
        predictive_error=EmpiricalHorizonPredictiveError.from_samples(
            {
                0.1: (
                    EmpiricalErrorSample(
                        errors=errors, source_group="fleet", trajectory_id="shell"
                    ),
                )
            }
        ),
    )
    assert shell.predictive_error_current

    stale = prior.initialize_belief(shell)
    assert not stale.predictive_error_current
    assert stale.compile_for_nmpc().maximum_error_horizon_s is None
    assert stale.provenance["parameter_prior_initialization"][
        "predictive_error_marked_stale"
    ]

    asserted = prior.initialize_belief(shell, predictive_error_valid_at_prior_mean=True)
    assert asserted.predictive_error_current
    assert asserted.provenance["parameter_prior_initialization"][
        "predictive_error_valid_at_prior_mean"
    ]
