"""Collection-effect acceptance must preserve all three fitted-arm cohorts."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from glassbox.experimental.command_excitation_decision import reduce
from glassbox.experimental.two_simulator_metrics import aggregate


def fixture(*, candidate=0.7, original=1.0):
    protocol = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "docs/harness/independent-command-excitation-v1.json"
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
                        ("original", original),
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
    protocol, rows = fixture(original=1.1)
    before = copy.deepcopy(rows)
    result = run(protocol, rows)
    assert result["residual_criteria_pass"]
    assert result["public_promotion"] is False
    assert result["shared_bootstrap_parent_draws_verified"]
    assert result["weighted_geometric_mean_ratios"]["response"] == pytest.approx(0.7)
    mechanism = result["mechanism_comparison"]
    assert mechanism["reference_arm"] == "original"
    assert mechanism["weighted_geometric_mean_ratios"]["response"] == pytest.approx(
        0.7 / 1.1
    )
    np.testing.assert_allclose(
        mechanism["bootstrap"]["aggregates"]["response"]["draws"], 0.7 / 1.1
    )
    assert rows == before
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
            if row["arm"] == "original" and row["kind"] == kind:
                row["mse"] *= 0.25
                row["finite_subset_mse"] *= 0.25
    result = run(protocol, rows)
    assert not result["checks"][f"mechanism_{kind}_regression"]
    assert not result["residual_criteria_pass"]


def test_original_cohort_mismatch_and_prediction_failure_remain_visible():
    protocol, rows = fixture()
    row = next(r for r in rows["crazyflow"] if r["arm"] == "original")
    row["query"] = "changed"
    with pytest.raises(ValueError, match="unmatched"):
        run(protocol, rows)
    row["query"] = f"q-{row['kind']}"
    row.update(prediction_finite=False, mse=None, finite_subset_mse=None)
    result = run(protocol, rows)
    assert not result["checks"]["original_finite_eligible_predictions"]
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


def test_weak_probe_errors_are_in_the_raw_objective():
    protocol, rows = fixture()
    for values in rows.values():
        for row in values:
            if row["kind"] == "response":
                row["pair_nonweak"] = row["group"] == "velocity_m_s"
                if row["arm"] == "candidate" and not row["pair_nonweak"]:
                    row["mse"] *= (1.2 / 0.7) ** 2
                    row["finite_subset_mse"] = row["mse"]
    result = run(protocol, rows)
    # Two velocity groups have ratio .7; all four weak angular/rotation groups
    # have ratio 1.2. Filtering weak probes would incorrectly produce .7.
    expected = (0.7**2 * 1.2**4) ** (1 / 6)
    assert result["weighted_geometric_mean_ratios"]["response"] == pytest.approx(
        expected
    )
    assert result["mechanism_comparison"]["weighted_geometric_mean_ratios"][
        "response"
    ] == pytest.approx(expected)
    assert not result["checks"]["response_gain"]
    assert not result["checks"]["mechanism_response_regression"]


def test_local_loss_does_not_impose_an_all_case_win_gate():
    protocol, rows = fixture()
    for row in rows["crazyflow"]:
        if (
            row["arm"] == "candidate"
            and row["kind"] == "factual"
            and row["group"] == "rotation_entries"
        ):
            row["mse"] *= (1.1 / 0.7) ** 2
            row["finite_subset_mse"] = row["mse"]
    result = run(protocol, rows)
    assert result["residual_criteria_pass"]
    assert result["weighted_geometric_mean_ratios"]["factual"] == pytest.approx(
        (1.1 * 0.7**5) ** (1 / 6)
    )
    assert any(row["ratio"] > 1 for row in result["physical_comparisons"])
    assert any(
        row["ratio"] > 1
        for row in result["mechanism_comparison"]["physical_comparisons"]
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("truth_eligible", False),
        ("pair_nonweak", True),
        ("components", 9),
        ("horizon_steps", 25),
    ],
)
def test_original_truth_cohort_cannot_differ(field, value):
    protocol, rows = fixture()
    row = next(r for r in rows["crazyflow"] if r["arm"] == "original")
    row[field] = value
    if field == "truth_eligible":
        row.update(mse=None, finite_subset_mse=None, components=0)
    with pytest.raises(ValueError, match="cohort differs"):
        run(protocol, rows)


def test_missing_original_metric_slot_is_rejected():
    protocol, rows = fixture()
    row = next(r for r in rows["crazyflow"] if r["arm"] == "original")
    rows["crazyflow"].remove(row)
    with pytest.raises(ValueError, match="unmatched"):
        run(protocol, rows)


def test_original_comparison_uses_its_frozen_response_limit():
    protocol, rows = fixture(candidate=0.85)
    # The angular target has a separate 20% gain, but the remaining responses
    # improve only 7.5% against original. Public's broad gain still passes.
    for values in rows.values():
        for row in values:
            if row["arm"] == "original" and row["kind"] == "response":
                scale = 0.85 / 0.925
                if (
                    row["simulator"] == "crazyflow"
                    and row["group"] == "body_rate_rad_s"
                ):
                    scale = 0.85 / 0.8
                row["mse"] *= scale**2
                row["finite_subset_mse"] = row["mse"]
    result = run(protocol, rows)
    mechanism_ratio = (0.925**5 * 0.8) ** (1 / 6)
    assert result["mechanism_comparison"]["weighted_geometric_mean_ratios"][
        "response"
    ] == pytest.approx(mechanism_ratio)
    assert result["checks"]["response_gain"]
    assert mechanism_ratio > 0.9
    assert not result["checks"]["mechanism_response_regression"]
    assert not result["residual_criteria_pass"]


def test_reference_is_frozen_and_historical_reducer_is_unchanged():
    from glassbox.experimental import state_input_decision

    protocol, rows = fixture()
    original_policy = copy.deepcopy(protocol)
    original_arms = state_input_decision.ARMS
    run(protocol, rows)
    assert protocol == original_policy
    assert state_input_decision.ARMS is original_arms
    assert original_arms == ("baseline", "candidate")
    protocol["decision"]["mechanism_comparison"]["reference_arm"] = "baseline"
    with pytest.raises(ValueError, match="reference must be original"):
        run(protocol, rows)
