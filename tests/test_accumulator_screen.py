"""Pure checks for non-overlapping paired timing and accuracy guards."""

import copy

import pytest
from screen_accumulator import aggregate, schedule


def intervals():
    cases = [dict(id="one", count=26)]
    labels = ["startup", "one:0", "one:25"]
    baseline = [
        dict(label=k, start_ns=i * 4 + 1, end_ns=i * 4 + 2)
        for i, k in enumerate(labels)
    ]
    candidate = [
        dict(label=k, start_ns=i * 4 + 3, end_ns=i * 4 + 4)
        for i, k in enumerate(labels)
    ]
    return cases, baseline, candidate


def test_paired_schedule_rejects_overlap_reordering_and_missing_blocks():
    cases, a, b = intervals()
    assert schedule(a, b, cases) == dict(blocks_per_arm=3, overlapping_intervals=0)
    bad = copy.deepcopy(b)
    bad[0]["start_ns"] = 1
    with pytest.raises(ValueError, match="overlap"):
        schedule(a, bad, cases)
    with pytest.raises(ValueError, match="roster"):
        schedule(a, b[:-1], cases)
    with pytest.raises(ValueError, match="order"):
        schedule(b, a, cases)


def test_velocity_gain_cannot_hide_rate_regression():
    cases = []
    for family in ("quad", "fixedwing"):
        cases.append(
            dict(
                id=family,
                family=family,
                ratios=dict(
                    velocity_rmse_m_s=0.8,
                    body_rate_rmse_rad_s=1.1,
                    primary=(0.8 * 1.1) ** 0.5,
                    tail=0.9,
                    orientation_rmse_rad=1.0,
                    rotation_rate=1.0,
                ),
            )
        )
    result = aggregate(cases)
    assert result["gates"]["primary"]
    assert not result["gates"]["body_rate"]
    assert not result["screen_passed"]
    assert not result["adopted"]
