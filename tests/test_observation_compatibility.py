from dataclasses import replace

import numpy as np
import pytest

from glassbox.observation_compatibility import (
    MAXIMUM_TIME_CONSTANT_S,
    MAXIMUM_SCALE,
    FirstOrderObservationFilter,
    StateObservationCorrection,
    apply_state_observation_correction,
    evaluate_first_order_observation_filter,
    evaluate_state_observation_correction,
    fit_first_order_observation_filter,
    fit_state_observation_correction,
)
from glassbox.synthetic import generate_trajectory


def _affine_corrupted_trajectory():
    trajectory = generate_trajectory(seed=4, duration_s=8.0)
    velocity_scale = np.asarray([1.08, 0.94, 1.12])
    velocity_bias = np.asarray([0.10, -0.05, 0.08])
    angular_scale = np.asarray([1.05, 0.92, 1.10])
    angular_bias = np.asarray([0.03, -0.02, 0.04])
    states = trajectory.states.copy()
    states[:, 3:6] = (states[:, 3:6] - velocity_bias) / velocity_scale
    states[:, 10:13] = (states[:, 10:13] - angular_bias) / angular_scale
    return (
        replace(trajectory, states=states),
        velocity_scale,
        velocity_bias,
        angular_scale,
        angular_bias,
    )


def _low_pass(values: np.ndarray, dt_s: float, time_constant_s: float):
    filtered = np.empty_like(values)
    filtered[0] = values[0]
    decay = np.exp(-dt_s / time_constant_s)
    for index in range(1, len(values)):
        filtered[index] = (
            decay * filtered[index - 1] + (1.0 - decay) * values[index]
        )
    return filtered


def _temporally_filtered_trajectory(
    *, seed: int, time_constant_s: float, duration_s: float = 3.0
):
    dt_s = 0.02
    trajectory = generate_trajectory(
        seed=seed, duration_s=duration_s, dt_s=dt_s
    )
    states = trajectory.states.copy()
    states[:, 3:6] = _low_pass(states[:, 3:6], dt_s, time_constant_s)
    states[:, 10:13] = _low_pass(
        states[:, 10:13], dt_s, time_constant_s
    )
    return replace(trajectory, states=states)


def test_affine_correction_recovers_known_state_measurement_errors() -> None:
    (
        trajectory,
        velocity_scale,
        velocity_bias,
        angular_scale,
        angular_bias,
    ) = _affine_corrupted_trajectory()

    result = fit_state_observation_correction([trajectory])

    np.testing.assert_allclose(
        result.correction.velocity_scale, velocity_scale, atol=1e-3
    )
    np.testing.assert_allclose(
        result.correction.velocity_bias_m_s, velocity_bias, atol=1e-3
    )
    np.testing.assert_allclose(
        result.correction.angular_rate_scale, angular_scale, atol=1e-3
    )
    np.testing.assert_allclose(
        result.correction.angular_rate_bias_rad_s, angular_bias, atol=1e-3
    )
    ratios = result.report["training_compatibility"]["after_over_before"]
    assert all(value < 0.01 for value in ratios.values())


def test_held_out_evaluation_applies_material_improvement_gate() -> None:
    trajectory, *_ = _affine_corrupted_trajectory()
    result = fit_state_observation_correction([trajectory])

    evaluation = evaluate_state_observation_correction(
        result.correction, [trajectory]
    )

    assert evaluation["gate"]["all_groups_improve_materially"] is True
    assert evaluation["gate"]["no_group_regresses"] is True
    assert len(evaluation["per_trajectory"]) == 1


def test_correction_is_bounded_and_rejects_protected_fit_data() -> None:
    trajectory = generate_trajectory(seed=6, duration_s=4.0)
    states = trajectory.states.copy()
    states[:, 3:6] *= 0.5
    states[:, 10:13] *= 0.5
    distorted = replace(trajectory, states=states)

    result = fit_state_observation_correction([distorted])

    assert np.all(result.correction.velocity_scale <= MAXIMUM_SCALE)
    assert np.all(result.correction.angular_rate_scale <= MAXIMUM_SCALE)
    assert max(result.correction.velocity_scale) == pytest.approx(MAXIMUM_SCALE)
    protected = replace(
        trajectory,
        labels={**trajectory.labels, "benchmark_split": "test"},
    )
    with pytest.raises(ValueError, match="protected benchmark splits"):
        fit_state_observation_correction([protected])


def test_fit_rejects_mixed_vehicle_configurations() -> None:
    first = generate_trajectory(seed=1, duration_s=1.0)
    second = replace(
        generate_trajectory(seed=2, duration_s=1.0),
        spec=replace(
            first.spec,
            vehicle=replace(
                first.spec.vehicle,
                configuration_id="different_vehicle",
            ),
        ),
    )

    with pytest.raises(ValueError, match="one vehicle configuration"):
        fit_state_observation_correction([first, second])


def test_application_preserves_arrays_and_records_observation_semantics() -> None:
    trajectory = generate_trajectory(seed=2, duration_s=1.0)
    correction = StateObservationCorrection(
        velocity_scale=np.asarray([1.01, 0.99, 1.02]),
        velocity_bias_m_s=np.asarray([0.1, 0.0, -0.1]),
        angular_rate_scale=np.asarray([1.0, 1.01, 0.98]),
        angular_rate_bias_rad_s=np.asarray([0.01, -0.01, 0.0]),
    )

    corrected = apply_state_observation_correction(trajectory, correction)

    np.testing.assert_allclose(corrected.time_s, trajectory.time_s)
    np.testing.assert_allclose(corrected.controls, trajectory.controls)
    np.testing.assert_allclose(corrected.observations, trajectory.observations)
    assert corrected.spec.observation_source.endswith(
        "bounded_affine_state_compatibility_v1"
    )
    assert corrected.provenance["state_observation_correction"]["policy"] == (
        "bounded_affine_state_compatibility_v1"
    )
    with pytest.raises(ValueError, match="already has"):
        apply_state_observation_correction(corrected, correction)


def test_first_order_filter_recovers_temporal_observation_response() -> None:
    time_constant_s = 0.08
    training = _temporally_filtered_trajectory(
        seed=8, time_constant_s=time_constant_s
    )
    held_out = _temporally_filtered_trajectory(
        seed=9, time_constant_s=time_constant_s
    )

    result = fit_first_order_observation_filter([training])
    evaluation = evaluate_first_order_observation_filter(
        result.candidate,
        result.instantaneous_reference,
        [held_out],
    )

    np.testing.assert_allclose(
        result.candidate.velocity_time_constant_s,
        time_constant_s,
        atol=0.02,
    )
    np.testing.assert_allclose(
        result.candidate.angular_rate_time_constant_s,
        time_constant_s,
        atol=0.02,
    )
    assert all(
        ratio < 0.1
        for ratio in evaluation["candidate_over_reference"].values()
    )
    assert evaluation["gate"]["passes"] is True


def test_temporal_fit_rejects_protected_data_and_boundary_is_not_promoted() -> None:
    slow = _temporally_filtered_trajectory(
        seed=10, time_constant_s=2.0, duration_s=4.0
    )

    result = fit_first_order_observation_filter([slow])

    np.testing.assert_allclose(
        result.candidate.velocity_time_constant_s, MAXIMUM_TIME_CONSTANT_S
    )
    np.testing.assert_allclose(
        result.candidate.angular_rate_time_constant_s,
        MAXIMUM_TIME_CONSTANT_S,
    )
    assert result.report["training_comparison"]["gate"][
        "time_constants_interior"
    ] is False
    assert result.report["training_comparison"]["gate"]["passes"] is False
    protected = replace(
        slow,
        labels={**slow.labels, "benchmark_split": "validation"},
    )
    with pytest.raises(ValueError, match="protected benchmark splits"):
        fit_first_order_observation_filter([protected])


def test_temporal_filter_validates_maintained_time_constant_bounds() -> None:
    with pytest.raises(ValueError, match="maintained bounds"):
        FirstOrderObservationFilter(
            velocity_time_constant_s=np.asarray([0.0, 0.1, 0.6]),
            velocity_scale=np.ones(3),
            velocity_bias_m_s=np.zeros(3),
            angular_rate_time_constant_s=np.zeros(3),
            angular_rate_scale=np.ones(3),
            angular_rate_bias_rad_s=np.zeros(3),
        )


def test_temporal_fit_weights_source_groups_instead_of_segment_count() -> None:
    first = replace(
        _temporally_filtered_trajectory(seed=11, time_constant_s=0.04),
        labels={"source_group": "first"},
    )
    second = replace(
        _temporally_filtered_trajectory(seed=12, time_constant_s=0.15),
        labels={"source_group": "second"},
    )

    balanced = fit_first_order_observation_filter([first, second])
    duplicated_segment = fit_first_order_observation_filter(
        [first, first, second]
    )

    for name in (
        "velocity_time_constant_s",
        "velocity_scale",
        "velocity_bias_m_s",
        "angular_rate_time_constant_s",
        "angular_rate_scale",
        "angular_rate_bias_rad_s",
    ):
        np.testing.assert_allclose(
            getattr(balanced.candidate, name),
            getattr(duplicated_segment.candidate, name),
            atol=1e-12,
        )
    assert duplicated_segment.report["source_group_count"] == 2
    assert duplicated_segment.report["fit_weighting"] == "equal_source_group"
