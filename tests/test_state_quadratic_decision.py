"""Three-arm acceptance must test the named mechanism and preserve cohorts."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from glassbox.experimental.state_quadratic_decision import reduce
from glassbox.experimental.two_simulator_metrics import aggregate


def fixture(*, candidate=0.7, bilinear=1.0):
    protocol = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "docs/harness/autonomous-state-quadratic-v1.json"
        ).read_text()
    )
    policy = protocol["decision"]["aggregation"]
    policy["horizons_s"] = [0.25]
    policy["scope_weights"] = {"primary": 1.0}
    protocol["cells"] = {
        sim: [{"id": "p", "group": "primary"}] for sim in ("crazyflow", "cascade")
    }
    protocol["recordings"] = [
        dict(id=f"{sim}/p/{n}", simulator=sim, role="test", cell="p")
        for sim in protocol["cells"]
        for n in range(2)
    ]
    rows = {sim: [] for sim in protocol["cells"]}
    for sim in rows:
        for n in range(2):
            for kind in ("factual", "response"):
                for group in policy["groups"]:
                    for arm, scale in (
                        ("baseline", 1.0),
                        ("bilinear", bilinear),
                        ("candidate", candidate),
                        ("hold", 1.5),
                    ):
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


def test_both_comparisons_use_same_draws_without_mutating_rows():
    protocol, rows = fixture(bilinear=1.1)
    original = copy.deepcopy(rows)
    result = run(protocol, rows)
    assert result["residual_criteria_pass"]
    assert result["public_promotion"] is False
    assert result["shared_bootstrap_parent_draws_verified"]
    assert result["weighted_geometric_mean_ratios"]["response"] == pytest.approx(0.7)
    mechanism = result["mechanism_comparison"]
    assert mechanism["reference_arm"] == "bilinear"
    assert mechanism["weighted_geometric_mean_ratios"]["response"] == pytest.approx(
        0.7 / 1.1
    )
    np.testing.assert_allclose(
        mechanism["bootstrap"]["aggregates"]["response"]["draws"], 0.7 / 1.1
    )
    assert rows == original
    assert result == run(protocol, rows)


def test_broad_public_gain_does_not_replace_named_angular_repair():
    protocol, rows = fixture(candidate=0.85)
    result = run(protocol, rows)
    assert result["checks"]["response_gain"]
    assert not result["checks"]["mechanism_angular_response_gain"]
    assert not result["residual_criteria_pass"]


@pytest.mark.parametrize("kind", ["factual", "response"])
def test_mechanism_cannot_discard_prior_broad_gains(kind):
    protocol, rows = fixture()
    for values in rows.values():
        for row in values:
            if row["arm"] == "bilinear" and row["kind"] == kind:
                row["mse"] *= 0.25
                row["finite_subset_mse"] *= 0.25
    result = run(protocol, rows)
    assert not result["checks"][f"mechanism_{kind}_regression"]
    assert not result["residual_criteria_pass"]


def test_bilinear_cohort_mismatch_and_prediction_failure_remain_visible():
    protocol, rows = fixture()
    row = next(r for r in rows["crazyflow"] if r["arm"] == "bilinear")
    row["query"] = "changed"
    with pytest.raises(ValueError, match="unmatched"):
        run(protocol, rows)
    row["query"] = f"q-{row['kind']}"
    row.update(prediction_finite=False, mse=None, finite_subset_mse=None)
    result = run(protocol, rows)
    assert not result["checks"]["bilinear_finite_eligible_predictions"]
    assert not result["residual_criteria_pass"]
    assert not result["shared_bootstrap_parent_draws_verified"]


def test_missing_truth_uses_identical_conditional_draws_in_both_comparisons():
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
    assert result["shared_bootstrap_parent_draws_verified"]
    public = result["bootstrap"]["aggregates"]["response"]
    mechanism = result["mechanism_comparison"]["bootstrap"]["aggregates"]["response"]
    assert public["unavailable_draws"] == mechanism["unavailable_draws"] > 0


def test_stale_summary_is_rejected():
    protocol, rows = fixture()
    summaries = {sim: aggregate(r) for sim, r in rows.items()}
    rows["crazyflow"][0]["mse"] *= 2
    with pytest.raises(ValueError, match="summary differs"):
        reduce(summaries, protocol, rows=rows)
