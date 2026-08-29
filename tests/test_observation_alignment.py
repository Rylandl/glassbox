from dataclasses import replace

import numpy as np
import pytest

from glassbox.observation_alignment import (
    MAXIMUM_ABSOLUTE_ALIGNMENT_S,
    StateObservationAlignment,
    evaluate_state_observation_alignment,
    fit_state_observation_alignment,
)
from glassbox.synthetic import generate_trajectory


def _shift(values: np.ndarray, time_s: np.ndarray, delay_s: float) -> np.ndarray:
    query_time_s = time_s - delay_s
    return np.column_stack(
        [
            np.interp(query_time_s, time_s, values[:, axis])
            for axis in range(3)
        ]
    )


def _shifted_trajectory(*, seed: int, delay_s: float, duration_s: float = 3.0):
    trajectory = generate_trajectory(
        seed=seed,
        duration_s=duration_s,
        dt_s=0.02,
    )
    states = trajectory.states.copy()
    states[:, 3:6] = _shift(
        states[:, 3:6], trajectory.time_s, delay_s
    )
    states[:, 10:13] = _shift(
        states[:, 10:13], trajectory.time_s, delay_s
    )
    return replace(trajectory, states=states)


@pytest.mark.parametrize("delay_s", [0.06, -0.04])
def test_alignment_recovers_signed_state_channel_delay(delay_s: float) -> None:
    training = _shifted_trajectory(seed=1, delay_s=delay_s)
    held_out = _shifted_trajectory(seed=2, delay_s=delay_s)

    result = fit_state_observation_alignment([training])
    evaluation = evaluate_state_observation_alignment(
        result.candidate,
        result.instantaneous_reference,
        [held_out],
    )

    assert result.candidate.velocity_delay_s == pytest.approx(delay_s, abs=0.01)
    assert result.candidate.angular_rate_delay_s == pytest.approx(
        delay_s, abs=0.01
    )
    assert all(
        ratio < 0.01
        for ratio in evaluation["candidate_over_reference"].values()
    )
    assert evaluation["gate"]["conditional_transfer_passes"] is True
    assert evaluation["gate"]["blanket_transfer_passes"] is True


def test_alignment_rejects_protected_fit_and_boundary_is_not_promoted() -> None:
    slow = _shifted_trajectory(seed=3, delay_s=0.2, duration_s=4.0)

    result = fit_state_observation_alignment([slow])

    assert result.candidate.velocity_delay_s == pytest.approx(
        MAXIMUM_ABSOLUTE_ALIGNMENT_S
    )
    assert result.candidate.angular_rate_delay_s == pytest.approx(
        MAXIMUM_ABSOLUTE_ALIGNMENT_S
    )
    decisions = result.report["training_comparison"]["gate"][
        "channel_decisions"
    ]
    assert all(not decision["identifiable"] for decision in decisions.values())
    assert result.report["training_comparison"]["gate"][
        "conditional_transfer_passes"
    ] is False
    protected = replace(
        slow,
        labels={**slow.labels, "benchmark_split": "holdout"},
    )
    with pytest.raises(ValueError, match="protected benchmark splits"):
        fit_state_observation_alignment([protected])


def test_alignment_validates_maintained_delay_bounds() -> None:
    with pytest.raises(ValueError, match="maintained bounds"):
        StateObservationAlignment(
            velocity_delay_s=0.11,
            velocity_scale=np.ones(3),
            velocity_bias_m_s=np.zeros(3),
            angular_rate_delay_s=0.0,
            angular_rate_scale=np.ones(3),
            angular_rate_bias_rad_s=np.zeros(3),
        )


def test_alignment_weights_source_groups_instead_of_segment_count() -> None:
    first = replace(
        _shifted_trajectory(seed=4, delay_s=0.03),
        labels={"source_group": "first"},
    )
    second = replace(
        _shifted_trajectory(seed=5, delay_s=0.08),
        labels={"source_group": "second"},
    )

    balanced = fit_state_observation_alignment([first, second])
    duplicated_segment = fit_state_observation_alignment(
        [first, first, second]
    )

    assert duplicated_segment.candidate.velocity_delay_s == pytest.approx(
        balanced.candidate.velocity_delay_s
    )
    assert duplicated_segment.candidate.angular_rate_delay_s == pytest.approx(
        balanced.candidate.angular_rate_delay_s
    )
    np.testing.assert_allclose(
        duplicated_segment.candidate.velocity_scale,
        balanced.candidate.velocity_scale,
    )
    np.testing.assert_allclose(
        duplicated_segment.candidate.angular_rate_scale,
        balanced.candidate.angular_rate_scale,
    )
    assert duplicated_segment.report["source_group_count"] == 2
    assert duplicated_segment.report["fit_weighting"] == "equal_source_group"
