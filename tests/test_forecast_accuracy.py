"""Requirements remain empirical and cannot hide failed forecasts or records."""

import json

import numpy as np
import pytest

from glassbox.experimental.forecast_accuracy import assess_forecasts


def assess(errors, ids=None, times=None, limits=None):
    a = next(iter(errors.values()))
    return assess_forecasts(
        errors,
        ["record"] * len(a) if ids is None else ids,
        times_s=np.arange(1, a.shape[1] + 1) * 0.1 if times is None else times,
        limits=dict.fromkeys(errors, 1.0) if limits is None else limits,
    )


def test_prefix_failure_cannot_recover_at_endpoint():
    r = assess({"error [m]": np.array([[0.5, 2.0, 0.1]])})
    metric = r["recordings"]["record"]["metrics"]["error [m]"]
    assert metric["endpoint_empirical_p95"] == [0.5, 2.0, 0.1]
    assert metric["prefix_empirical_p95"] == [0.5, 2.0, 2.0]
    assert r["minimum_recording_joint_fraction"] == [1.0, 0.0, 0.0]
    assert r["observed_95pct_horizon_s"] == 0.1


def test_poor_short_recording_cannot_be_hidden_by_pooling():
    errors = np.r_[np.zeros(100), 2.0][:, None]
    r = assess({"e": errors}, ids=["long"] * 100 + ["short"])
    assert np.mean(errors <= 1) > 0.95
    assert r["minimum_recording_joint_fraction"] == [0.0]
    assert r["required_marginal_tolerance"]["e"] == [2.0]
    assert r["observed_95pct_horizon_s"] is None


def test_marginal_95_percent_does_not_claim_joint_95_percent():
    a, b = np.zeros((20, 1)), np.zeros((20, 1))
    a[0], b[1] = 2.0, 2.0
    r = assess({"a": a, "b": b})
    assert r["required_marginal_tolerance"] == {"a": [0.0], "b": [0.0]}
    assert r["minimum_recording_joint_fraction"] == [0.9]
    assert r["observed_95pct_horizon_s"] is None


def test_nearest_rank_is_empirical_not_a_conformal_bound():
    r = assess({"e": np.arange(1, 21, dtype=float)[:, None]}, limits={"e": 19})
    assert r["required_marginal_tolerance"]["e"] == [19.0]
    assert r["minimum_recording_joint_fraction"] == [0.95]
    assert r["observed_95pct_horizon_s"] == 0.1
    # One sample still has a finite descriptive maximum, without a population claim.
    assert assess({"e": np.array([[2.0]])})["required_marginal_tolerance"] == {
        "e": [2.0]
    }


def test_nonfinite_forecasts_fail_without_dropping_windows():
    r = assess({"e": np.array([[0.0, np.nan, 0.0], [0.0, np.inf, 0.0]])})
    metric = r["recordings"]["record"]["metrics"]["e"]
    assert metric["prefix_unavailable_count"] == [0, 2, 2]
    assert metric["prefix_empirical_p95"] == [0.0, None, None]
    assert r["minimum_recording_joint_fraction"] == [1.0, 0.0, 0.0]
    json.dumps(r, allow_nan=False)


def test_units_and_window_order_do_not_change_joint_fraction():
    rng = np.random.default_rng(7)
    errors = rng.uniform(0, 2, (30, 3))
    ids = np.array(["a"] * 10 + ["b"] * 20)
    before = errors.copy()
    a = assess({"e [m]": errors}, ids=ids)
    b = assess({"e [cm]": errors[::-1] * 100}, ids=ids[::-1], limits={"e [cm]": 100})
    assert (
        a["minimum_recording_joint_fraction"] == b["minimum_recording_joint_fraction"]
    )
    np.testing.assert_allclose(
        np.array(a["required_marginal_tolerance"]["e [m]"]) * 100,
        b["required_marginal_tolerance"]["e [cm]"],
    )
    np.testing.assert_array_equal(errors, before)


@pytest.mark.parametrize("limit", [0, -1, np.inf, np.nan, True, [1, 2], "invalid"])
def test_invalid_allowance_rejected(limit):
    with pytest.raises(ValueError):
        assess({"e": np.zeros((2, 1))}, limits={"e": limit})


@pytest.mark.parametrize(
    "errors,ids,times,limits",
    [
        ({"e": [[-1.0]]}, ["a"], [0.1], {"e": 1}),
        ({"e": [[-np.inf]]}, ["a"], [0.1], {"e": 1}),
        ({"e": [[1.0, 1.0]]}, ["a"], [0.1, 0.1], {"e": 1}),
        ({"e": [[1.0]]}, ["a"], [0.0], {"e": 1}),
        ({"e": [[1.0]]}, [" "], [0.1], {"e": 1}),
        ({"e": [[1.0]]}, ["a", "b"], [0.1], {"e": 1}),
        ({"e": [[1.0]]}, ["a"], [0.1], {"unknown": 1}),
        ({"e": [[1.0 + 1.0j]]}, ["a"], [0.1], {"e": 1}),
        ({"e": [[1.0]]}, ["a"], [0.1 + 1.0j], {"e": 1}),
        ({"e": [[1.0]]}, [b"a"], [0.1], {"e": 1}),
        ({}, ["a"], [0.1], {}),
    ],
)
def test_invalid_contract_rejected(errors, ids, times, limits):
    with pytest.raises(ValueError):
        assess_forecasts(errors, ids, times_s=times, limits=limits)
