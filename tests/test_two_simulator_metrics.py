"""Hand-calculated checks independent of simulator execution and model fitting."""

import json

import numpy as np
import pytest

from glassbox.experimental.two_simulator_metrics import (
    aggregate,
    score_forecast,
    score_response_directions,
)


def _row(rows, horizon=0.05, group="velocity_m_s", statistic="endpoint"):
    return next(
        r
        for r in rows
        if (r["horizon_s"], r["group"], r["statistic"]) == (horizon, group, statistic)
    )


def _query(error=1, *, cell="a", parent="p", query="q", arm="generic", eligible=True):
    prediction = np.zeros((1, 15))
    prediction[:, :3] = error
    row = _row(
        score_forecast(
            prediction,
            np.zeros_like(prediction),
            np.array([eligible]),
            dt_s=0.05,
            horizons_s=[],
        )
    )
    return dict(
        row,
        scope="primary",
        simulator="cascade",
        kind="factual",
        arm=arm,
        cell=cell,
        parent=parent,
        query=query,
    )


def test_endpoint_cumulative_component_mse_and_first_sample_are_distinct():
    prediction = np.zeros((5, 15))
    prediction[:, :3] = np.arange(1, 6)[:, None]
    rows = score_forecast(
        prediction,
        np.zeros_like(prediction),
        np.ones(5, bool),
        dt_s=0.01,
        horizons_s=[0.05, 0.01, 0.05],
    )
    assert len(rows) == 12  # Native first step appears once, not once per declaration.
    assert _row(rows)["mse"] == 25
    assert _row(rows, statistic="cumulative")["mse"] == 11
    assert _row(rows, horizon=0.01)["mse"] == 1
    assert _row(rows, group="body_rate_rad_s")["mse"] == 0


def test_bad_truth_censors_entire_prefix_and_missing_horizons_remain_planned():
    prediction = np.zeros((3, 15))
    rows = score_forecast(
        prediction,
        prediction.copy(),
        np.array([True, False, True]),
        dt_s=0.05,
        horizons_s=[0.15, 0.25],
    )
    assert len(rows) == 18
    assert _row(rows)["truth_eligible"]
    assert not _row(rows, horizon=0.15)["truth_eligible"]
    assert not _row(rows, horizon=0.25)["truth_eligible"]
    assert _row(rows, horizon=0.15)["mse"] is None
    assert _row(rows, horizon=0.15)["finite_subset_mse"] is None


def test_nonfinite_predictions_fail_score_and_count_uncovered_components():
    prediction = np.ones((1, 15))
    prediction[0, 1] = np.nan
    rows = score_forecast(
        prediction,
        np.zeros_like(prediction),
        np.ones(1, bool),
        dt_s=0.05,
        horizons_s=[],
        envelope=np.ones_like(prediction),
    )
    velocity = _row(rows)
    assert velocity["mse"] is None
    assert not velocity["prediction_finite"]
    assert velocity["finite_subset_mse"] == 1
    assert velocity["finite_components"] == 2
    assert velocity["coverage"] == pytest.approx(2 / 3)
    assert velocity["coverage_components"] == 3
    assert _row(rows, group="body_rate_rad_s")["mse"] == 1


def test_endpoint_can_be_finite_when_earlier_cumulative_prediction_failed():
    prediction = np.zeros((3, 15))
    prediction[0, 0] = np.inf
    rows = score_forecast(
        prediction,
        np.zeros_like(prediction),
        np.ones(3, bool),
        dt_s=0.05,
        horizons_s=[0.15],
    )
    assert _row(rows, horizon=0.15)["mse"] == 0
    assert _row(rows, horizon=0.15, statistic="cumulative")["mse"] is None


def test_rotation_geometry_is_raw_and_can_be_disabled_for_responses():
    prediction = np.zeros((1, 15))
    prediction[0, 6:] = np.diag([1, 1, -1]).ravel()
    rows = score_forecast(
        prediction,
        np.zeros_like(prediction),
        np.ones(1, bool),
        dt_s=0.05,
        horizons_s=[],
    )
    geometry = _row(rows, group="rotation_entries")["geometry"]
    assert geometry["orthogonality_frobenius_mean"] == 0
    assert geometry["determinant_absolute_error_mean"] == 2
    assert geometry["nonpositive_determinants"] == 1
    response = score_forecast(
        prediction,
        np.zeros_like(prediction),
        np.ones(1, bool),
        dt_s=0.05,
        horizons_s=[],
        rotation_geometry=False,
    )
    assert "geometry" not in _row(response, group="rotation_entries")


def test_aggregate_averages_origins_then_parents_then_cells_before_sqrt():
    rows = [_query(1, parent="a1", query=str(i)) for i in range(9)]
    rows += [_query(3, parent="a2"), _query(5, cell="b", parent="b1")]
    result = aggregate(rows)
    summary = result["summaries"][0]
    # Cell A = (1 + 9)/2 = 5, cell B = 25. Never pool the eleven origins.
    assert summary["mse"] == 15
    assert summary["rmse"] == pytest.approx(np.sqrt(15))
    assert summary["planned"] == 11
    assert len(result["parents"]) == 3
    assert len(result["cells"]) == 2


def test_response_branches_average_within_origin_before_origin_weighting():
    rows = [
        dict(_query(1, query="lo"), origin=10),
        dict(_query(1, query="hi"), origin=10),
    ]
    rows += [dict(_query(3, query="later-lo"), origin=20)]
    rows += [dict(_query(9, query="later-hi", eligible=False), origin=20)]
    summary = aggregate(rows)["summaries"][0]
    assert summary["rmse"] is None
    assert summary["available_truth_mse"] == 5
    assert summary["planned_origins"] == summary["eligible_origins"] == 2
    assert summary["empty_origins"] == 0


def test_empty_parents_and_cells_remain_in_counts_and_block_headline():
    rows = [
        _query(2),
        _query(3, parent="empty", eligible=False),
        _query(7, cell="empty", parent="empty", eligible=False),
    ]
    summary = aggregate(rows)["summaries"][0]
    assert summary["planned"] == 3
    assert summary["truth_eligible"] == 1
    assert summary["empty_parents"] == 2
    assert summary["empty_cells"] == 1
    assert summary["status"] == "incomplete"
    assert summary["rmse"] is None
    assert summary["available_truth_rmse"] == 2


def test_one_prediction_failure_cannot_be_hidden_by_finite_subset_reduction():
    good, bad = _query(2), _query(4, query="bad")
    bad.update(prediction_finite=False, mse=None, finite_subset_mse=16)
    summary = aggregate([good, bad])["summaries"][0]
    assert summary["failed"] == 1
    assert summary["predicted_finite"] == 1
    assert summary["status"] == "failed"
    assert summary["rmse"] is None
    assert summary["available_truth_rmse"] is None
    assert summary["finite_query_subset_rmse"] == 2
    assert summary["finite_component_subset_rmse"] == pytest.approx(np.sqrt(10))


def test_ratios_require_same_query_roster_and_eligibility():
    generic, reference = _query(2), _query(4, arm="structured")
    ratio = aggregate([generic, reference])["paired_ratios"][0]
    assert ratio["comparable_cohort"]
    assert ratio["ratio"] == 0.5
    assert ratio["improvement_fraction"] == 0.5
    reference["query"] = "different"
    mismatch = aggregate([generic, reference])["paired_ratios"][0]
    assert not mismatch["comparable_cohort"]
    assert mismatch["unmatched_queries"] == 2
    assert mismatch["available_truth_ratio"] is None
    reference = _query(4, arm="structured", eligible=False)
    mismatch = aggregate([generic, reference])["paired_ratios"][0]
    assert mismatch["truth_eligibility_mismatches"] == 1
    assert mismatch["ratio"] is None
    generic, reference = (
        dict(_query(), origin=10),
        dict(_query(arm="structured"), origin=20),
    )
    mismatch = aggregate([generic, reference])["paired_ratios"][0]
    assert mismatch["origin_mismatches"] == 1
    assert not mismatch["comparable_cohort"]


def test_zero_reference_is_explicit_and_incomplete_ratios_are_conditional():
    zero = aggregate([_query(2), _query(0, arm="hold")])["paired_ratios"][0]
    assert zero["zero_reference"]
    assert zero["ratio"] is None
    rows = [_query(2), _query(4, arm="structured")]
    rows += [
        _query(query="missing", eligible=False),
        _query(query="missing", arm="structured", eligible=False),
    ]
    ratio = aggregate(rows)["paired_ratios"][0]
    assert ratio["comparable_cohort"]
    assert ratio["ratio"] is None
    assert ratio["available_truth_ratio"] == 0.5


def test_response_ratios_filter_weak_pairs_without_filtering_raw_error():
    rows = []
    for arm, error in (("generic", 2), ("structured", 4)):
        rows.append(dict(_query(error, arm=arm), kind="response", pair_nonweak=True))
        rows.append(
            dict(
                _query(100, arm=arm, query="weak"), kind="response", pair_nonweak=False
            )
        )
    result = aggregate(rows)
    ratio = result["paired_ratios"][0]
    assert ratio["ratio"] == 0.5
    assert ratio["cohort"] == "nonweak_signed_pairs"
    assert ratio["cohort_queries"] == ratio["excluded_weak_queries"] == 1
    generic = next(r for r in result["summaries"] if r["arm"] == "generic")
    assert generic["mse"] == 5002
    assert generic["planned"] == 2
    with pytest.raises(ValueError, match="pair_nonweak"):
        aggregate([dict(_query(), kind="response")])
    rows[-1]["pair_nonweak"] = True
    assert not aggregate(rows)["paired_ratios"][0]["comparable_cohort"]


def test_duplicate_slots_and_invalid_contracts_raise_instead_of_changing_weights():
    with pytest.raises(ValueError, match="duplicate"):
        aggregate([_query(), _query()])
    with pytest.raises(ValueError, match="observation grid"):
        score_forecast(
            np.zeros((3, 15)),
            np.zeros((3, 15)),
            np.ones(3, bool),
            dt_s=0.05,
            horizons_s=[0.12],
        )
    with pytest.raises(ValueError, match="Boolean"):
        score_forecast(
            np.zeros((3, 15)), np.zeros((3, 15)), np.ones(3), dt_s=0.05, horizons_s=[]
        )
    target = np.zeros((1, 15))
    target[0, 0] = np.nan
    with pytest.raises(ValueError, match="truth marked valid"):
        score_forecast(
            np.zeros_like(target), target, np.ones(1, bool), dt_s=0.05, horizons_s=[]
        )


def test_empty_query_and_arithmetic_overflow_are_json_safe():
    rows = score_forecast(
        np.empty((0, 15)),
        np.empty((0, 15)),
        np.empty(0, bool),
        dt_s=0.05,
        horizons_s=[0.25],
    )
    assert len(rows) == 12
    assert not any(r["truth_eligible"] for r in rows)
    huge = _query(1e200)
    assert huge["prediction_finite"]
    assert huge["mse"] is None
    assert huge["failure"] == "nonfinite_squared_error"
    json.dumps(aggregate([huge]), allow_nan=False)
    ratio = aggregate([_query(1e154), _query(1e-160, arm="hold")])["paired_ratios"][0]
    assert ratio["nonfinite_ratio"]
    assert ratio["ratio"] is None
    json.dumps(ratio, allow_nan=False)


def test_response_direction_requires_both_signs_nonweak_and_handles_zero_hold():
    truth = np.zeros((2, 1, 15))
    truth[0, 0, 0] = -2
    truth[1, 0, 0] = 3
    prediction = -truth
    thresholds = {
        "velocity_m_s": 1e-4,
        "body_rate_rad_s": 1e-4,
        "rotation_entries": 1e-5,
    }
    kwargs = dict(dt_s=0.05, horizons_s=[], thresholds=thresholds)
    rows = score_response_directions(prediction, truth, np.ones((2, 1), bool), **kwargs)
    velocity = [r for r in rows if r["group"] == "velocity_m_s"]
    assert all(r["pair_nonweak"] and r["cosine"] == -1 for r in velocity)
    assert all(r["sign_components"] == 1 and r["sign_agreement"] == 0 for r in velocity)
    hold = score_response_directions(
        np.zeros_like(truth), truth, np.ones((2, 1), bool), **kwargs
    )
    assert all(
        r["zero_prediction"] and r["cosine"] is None
        for r in hold
        if r["group"] == "velocity_m_s"
    )
    truth[0, 0, 0] = 0
    weak = score_response_directions(prediction, truth, np.ones((2, 1), bool), **kwargs)
    assert not any(r["pair_nonweak"] for r in weak)
    assert all(r["cosine"] is None for r in weak)


def test_response_nonfinite_prediction_and_invalid_truth_keep_planned_rows():
    truth = np.ones((2, 3, 15))
    prediction = truth.copy()
    prediction[0, 0, 0] = np.nan
    valid = np.ones((2, 3), bool)
    valid[1, 1] = False
    rows = score_response_directions(
        prediction,
        truth,
        valid,
        dt_s=0.05,
        horizons_s=[0.15],
        thresholds={
            key: 1e-4 for key in ("velocity_m_s", "body_rate_rad_s", "rotation_entries")
        },
    )
    assert len(rows) == 12
    assert not next(
        r for r in rows if r["sign"] == "lower" and r["group"] == "velocity_m_s"
    )["prediction_finite"]
    assert not any(r["truth_eligible"] for r in rows if r["horizon_s"] == 0.15)
