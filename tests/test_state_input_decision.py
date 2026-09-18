"""Residual decision semantics independent of simulator or learner execution."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from glassbox.experimental.state_input_decision import reduce
from glassbox.experimental.two_simulator_metrics import aggregate


def fixture(response=0.8, factual=0.99):
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads(
        (root / "docs/harness/state-input-interaction-v1.json").read_text()
    )
    policy = protocol["decision"]["aggregation"]
    policy["horizons_s"] = [0.25]
    policy["scope_weights"] = {"primary": 1.0}
    protocol["cells"] = {
        s: [{"id": "p", "group": "primary"}] for s in ("crazyflow", "cascade")
    }
    protocol["recordings"] = [
        dict(id=f"{s}/p/{n}", simulator=s, role="test", cell="p")
        for s in protocol["cells"]
        for n in range(2)
    ]
    rows = {}
    for sim in protocol["cells"]:
        rows[sim] = []
        for n in range(2):
            for kind in ("factual", "response"):
                for group in policy["groups"]:
                    for arm in ("baseline", "candidate"):
                        scale = (
                            1
                            if arm == "baseline"
                            else response
                            if kind == "response"
                            else factual
                        )
                        mse = ((n + 1) * scale) ** 2
                        rows[sim].append(
                            dict(
                                simulator=sim,
                                scope="primary",
                                cell="p",
                                parent=f"{sim}/p/{n}",
                                query=f"q-{kind}",
                                origin=1,
                                kind=kind,
                                arm=arm,
                                horizon_s=0.25,
                                horizon_steps=5,
                                group=group,
                                statistic="endpoint",
                                truth_eligible=True,
                                prediction_finite=True,
                                mse=mse,
                                finite_subset_mse=mse,
                                components=3,
                                pair_nonweak=False,
                            )
                        )
    return protocol, rows


def run(protocol, rows):
    return reduce({sim: aggregate(r) for sim, r in rows.items()}, protocol, rows=rows)


def test_raw_weak_responses_contribute_to_progress_and_paired_bootstrap():
    protocol, rows = fixture()
    result = run(protocol, rows)
    assert result["residual_criteria_pass"]
    assert result["weighted_geometric_mean_ratios"] == pytest.approx(
        {"factual": 0.99, "response": 0.8}
    )
    assert result["public_promotion"] is False
    bootstrap = result["bootstrap"]["aggregates"]["response"]
    assert bootstrap["available_draws"] == 1000
    np.testing.assert_allclose(bootstrap["draws"], 0.8)
    assert result == run(protocol, rows)


def test_response_win_does_not_hide_factual_regression():
    protocol, rows = fixture(response=0.5, factual=1.1)
    result = run(protocol, rows)
    assert result["checks"]["response_gain"]
    assert not result["checks"]["factual_regression"]
    assert not result["residual_criteria_pass"]


def test_identical_counts_cannot_hide_different_query_identity_or_truth():
    protocol, rows = fixture()
    changed = copy.deepcopy(rows)
    changed["crazyflow"][1]["query"] = "other-query"
    with pytest.raises(ValueError, match="unmatched"):
        run(protocol, changed)
    changed = copy.deepcopy(rows)
    row = changed["crazyflow"][1]
    row.update(truth_eligible=False, mse=None, finite_subset_mse=None)
    with pytest.raises(ValueError, match="truth cohort"):
        run(protocol, changed)


def test_parent_missing_from_both_arms_is_rejected():
    protocol, rows = fixture()
    rows["crazyflow"] = [r for r in rows["crazyflow"] if r["parent"] != "crazyflow/p/1"]
    with pytest.raises(ValueError, match="parent roster"):
        run(protocol, rows)


def test_nonfinite_prediction_cannot_improve_by_dropping_failed_parent():
    protocol, rows = fixture()
    rows["crazyflow"][1].update(
        prediction_finite=False, mse=None, finite_subset_mse=None
    )
    result = run(protocol, rows)
    assert not result["checks"]["finite_eligible_predictions"]
    assert not result["residual_criteria_pass"]
    assert result["bootstrap"]["status"] == "unavailable_due_to_prediction_failure"


def test_small_physical_floor_is_fixed_and_zero_reference_is_not_a_win():
    protocol, rows = fixture()
    for values in rows.values():
        for row in values:
            row.update(mse=0.0, finite_subset_mse=0.0)
    result = run(protocol, rows)
    assert result["weighted_geometric_mean_ratios"] == {"factual": 1.0, "response": 1.0}
    assert not result["checks"]["response_gain"]


def test_missing_truth_remains_conditional_without_changing_paired_cohort():
    protocol, rows = fixture()
    for row in rows["crazyflow"]:
        if row["parent"] == "crazyflow/p/1":
            row.update(
                truth_eligible=False,
                prediction_finite=False,
                mse=None,
                finite_subset_mse=None,
                components=0,
            )
    result = run(protocol, rows)
    assert result["residual_criteria_pass"]
    assert all(
        c["conditional_on_incomplete_truth"]
        for c in result["physical_comparisons"]
        if c["simulator"] == "crazyflow"
    )
    assert result["bootstrap"]["aggregates"]["response"]["unavailable_draws"] > 0
    assert all(
        r["missing_parents"] == 1
        for r in result["parent_error_tails"]
        if r["simulator"] == "crazyflow"
    )
