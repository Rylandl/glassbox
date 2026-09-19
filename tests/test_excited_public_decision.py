"""Independent ratios and failure isolation for four frozen paired comparisons."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from glassbox.experimental.excited_public_decision import reduce
from glassbox.experimental.two_simulator_metrics import aggregate


def fixture(*, candidate=0.7, quadratic=0.7):
    protocol = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "docs/harness/excited-public-architecture-v1.json"
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
    for sim, values in rows.items():
        for n in range(2):
            for kind in ("factual", "response"):
                for group in policy["groups"]:
                    for arm, scale in (
                        ("baseline", 1.0),
                        ("candidate", candidate),
                        ("quadratic", quadratic),
                        ("hold", 1.5),
                    ):
                        mse = ((n + 1) * scale) ** 2
                        values.append(
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
    return reduce(
        {sim: aggregate(values) for sim, values in rows.items()}, protocol, rows=rows
    )


def fail_arm(rows, arm):
    for values in rows.values():
        for row in values:
            if row["arm"] == arm:
                row.update(prediction_finite=False, mse=None, finite_subset_mse=None)


def test_all_four_directions_are_explicit_and_unchanged_inputs():
    protocol, rows = fixture(candidate=0.8, quadratic=0.65)
    before = copy.deepcopy((protocol, rows))
    result = run(protocol, rows)
    ratios = {
        "public_collection": 0.8,
        "quadratic_workflow": 0.65,
        "simpler_recipe_comparable": 0.8 / 0.65,
        "quadratic_material_value": 0.65 / 0.8,
    }
    for name, expected in ratios.items():
        comparison = result["comparisons"][name]
        for kind in ("response", "factual"):
            assert comparison["weighted_geometric_mean_ratios"][kind] == pytest.approx(
                expected
            )
            np.testing.assert_allclose(
                comparison["bootstrap"]["aggregates"][kind]["draws"], expected
            )
        assert len(comparison["tail_checks"]) == 12
        assert comparison["parent_error_tails"]
        assert (
            comparison["remaining_promotion_requirements"]
            == protocol["decision"]["public_promotion"]
        )
    assert result["public_collection_progress"]
    assert result["quadratic_workflow_progress"]
    assert result["quadratic_material_value"]
    assert not result["simpler_recipe_comparable"]
    assert result["preference"] == "quadratic_for_separate_qualification"
    assert result["public_promotion"] is False
    assert result["shared_bootstrap_parent_draws_verified"]
    assert len(set(result["bootstrap_comparison_hashes"].values())) == 1
    assert (protocol, rows) == before
    assert result == run(protocol, rows)


def test_equal_excited_errors_prefer_existing_public_recipe():
    result = run(*fixture())
    assert result["simpler_recipe_comparable"]
    assert not result["quadratic_material_value"]
    assert result["preference"] == "public_recipe"
    assert not result["unresolved_tradeoff"]


def test_unavailable_quadratic_does_not_erase_public_collection_gain():
    protocol, rows = fixture()
    fail_arm(rows, "quadratic")
    result = run(protocol, rows)
    assert result["public_collection_progress"]
    assert not result["quadratic_workflow_progress"]
    assert not result["simpler_recipe_comparable"]
    assert not result["quadratic_material_value"]
    assert result["unresolved_tradeoff"]
    assert result["available_bootstrap_parent_draws_match"]
    assert not result["shared_bootstrap_parent_draws_verified"]
    for name, comparison in result["comparisons"].items():
        if name != "public_collection":
            assert not comparison["pairwise_predictions_available"]
            assert (
                comparison["bootstrap"]["status"]
                == "unavailable_due_to_prediction_failure"
            )
    assert (
        result["comparisons"]["public_collection"]["cohorts"]["crazyflow"][
            "test_parents"
        ]
        == 2
    )


def test_unavailable_public_excited_preserves_quadratic_workflow_evidence():
    protocol, rows = fixture()
    fail_arm(rows, "candidate")
    result = run(protocol, rows)
    assert result["quadratic_workflow_progress"]
    assert not result["public_collection_progress"]
    assert not result["quadratic_material_value"]
    assert result["unresolved_tradeoff"]


@pytest.mark.parametrize(
    "candidate,expected", [(0.7 * 1.049999, True), (0.7 * 1.050001, False)]
)
def test_comparability_uses_five_percent_margin(candidate, expected):
    result = run(*fixture(candidate=candidate))
    assert result["public_collection_progress"]
    assert result["simpler_recipe_comparable"] is expected


@pytest.mark.parametrize(
    "quadratic,expected", [(0.8 * 0.899999, True), (0.8 * 0.900001, False)]
)
def test_quadratic_needs_ten_percent_response_value(quadratic, expected):
    result = run(*fixture(candidate=0.8, quadratic=quadratic))
    assert result["quadratic_workflow_progress"]
    assert result["quadratic_material_value"] is expected


def test_unresolved_tradeoff_preserves_both_collection_workflow_gains():
    result = run(*fixture(candidate=0.8, quadratic=0.75))
    assert (
        result["public_collection_progress"] and result["quadratic_workflow_progress"]
    )
    assert (
        not result["simpler_recipe_comparable"]
        and not result["quadratic_material_value"]
    )
    assert result["preference"] == "unresolved_tradeoff"


def test_comparability_requires_public_collection_progress():
    result = run(*fixture(candidate=0.95, quadratic=0.95))
    assert result["comparisons"]["simpler_recipe_comparable"]["residual_criteria_pass"]
    assert not result["public_collection_progress"]
    assert not result["simpler_recipe_comparable"]


def test_quadratic_value_requires_its_workflow_progress():
    result = run(*fixture(candidate=1.3, quadratic=1.1))
    assert result["comparisons"]["quadratic_material_value"]["residual_criteria_pass"]
    assert not result["quadratic_workflow_progress"]
    assert not result["quadratic_material_value"]


def test_small_individual_loss_is_allowed_without_legacy_angular_gate():
    protocol, rows = fixture(candidate=0.7, quadratic=0.75)
    for row in rows["crazyflow"]:
        if (
            row["arm"] == "candidate"
            and row["kind"] == "response"
            and row["group"] == "body_rate_rad_s"
        ):
            row["mse"] *= (1.1 / 0.7) ** 2
            row["finite_subset_mse"] = row["mse"]
    result = run(protocol, rows)
    assert result["public_collection_progress"]
    ratio = result["comparisons"]["public_collection"][
        "weighted_geometric_mean_ratios"
    ]["response"]
    assert ratio == pytest.approx((1.1 * 0.7**5) ** (1 / 6))
    assert (
        "mechanism_angular_response_gain"
        not in result["comparisons"]["public_collection"]["checks"]
    )


def test_weak_probes_stay_in_objective():
    protocol, rows = fixture()
    for values in rows.values():
        for row in values:
            if row["kind"] == "response":
                row["pair_nonweak"] = row["group"] == "velocity_m_s"
                if row["arm"] == "candidate" and not row["pair_nonweak"]:
                    row["mse"] *= (1.2 / 0.7) ** 2
                    row["finite_subset_mse"] = row["mse"]
    result = run(protocol, rows)
    expected = (0.7**2 * 1.2**4) ** (1 / 6)
    assert result["comparisons"]["public_collection"]["weighted_geometric_mean_ratios"][
        "response"
    ] == pytest.approx(expected)
    assert not result["public_collection_progress"]


def test_primary_and_tail_guards_are_not_replaced_by_broad_gain():
    protocol, rows = fixture(candidate=0.3, quadratic=0.3)
    for row in rows["crazyflow"]:
        if row["arm"] == "candidate" and row["kind"] == "response":
            row["mse"] *= (1.6 / 0.3) ** 2
            row["finite_subset_mse"] = row["mse"]
    result = run(protocol, rows)
    public = result["comparisons"]["public_collection"]
    assert public["checks"]["response_gain"]
    assert not public["checks"]["primary_simulator_regressions"]
    assert not public["checks"]["scope_regressions"]
    assert not public["checks"]["primary_parent_tail_regressions"]
    assert not result["public_collection_progress"]


@pytest.mark.parametrize("arm", ["baseline", "candidate", "quadratic"])
def test_missing_slot_is_integrity_failure_not_unavailable_model(arm):
    protocol, rows = fixture()
    row = next(r for r in rows["crazyflow"] if r["arm"] == arm)
    rows["crazyflow"].remove(row)
    with pytest.raises(ValueError, match="unmatched"):
        run(protocol, rows)


def test_truth_masks_must_match_across_each_pair():
    protocol, rows = fixture()
    row = next(r for r in rows["crazyflow"] if r["arm"] == "quadratic")
    row["pair_nonweak"] = True
    with pytest.raises(ValueError, match="cohort differs"):
        run(protocol, rows)


def test_conditional_missing_truth_has_identical_parent_draws():
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
    assert result["simpler_recipe_comparable"]
    assert result["shared_bootstrap_parent_draws_verified"]
    assert all(
        c["bootstrap"]["aggregates"]["response"]["unavailable_draws"] > 0
        for c in result["comparisons"].values()
    )


def test_stale_full_summary_is_rejected_even_for_hold():
    protocol, rows = fixture()
    summaries = {sim: aggregate(values) for sim, values in rows.items()}
    row = next(r for r in rows["crazyflow"] if r["arm"] == "hold")
    row["mse"] *= 2
    with pytest.raises(ValueError, match="summary differs"):
        reduce(summaries, protocol, rows=rows)


def test_historical_reducer_globals_remain_unchanged():
    from glassbox.experimental import state_input_decision

    arms = state_input_decision.ARMS
    run(*fixture())
    assert state_input_decision.ARMS is arms
    assert arms == ("baseline", "candidate")


@pytest.mark.parametrize(
    "comparison,arm,scale",
    [
        ("simpler_recipe_comparable", "candidate", 0.7 * 1.06),
        ("quadratic_material_value", "quadratic", 0.8 * 1.06),
    ],
)
def test_factual_margin_applies_separately_from_response(comparison, arm, scale):
    base = 0.7 if comparison == "simpler_recipe_comparable" else 0.8
    quadratic = 0.7 if comparison == "simpler_recipe_comparable" else 0.65
    protocol, rows = fixture(candidate=base, quadratic=quadratic)
    original = base if arm == "candidate" else quadratic
    for values in rows.values():
        for row in values:
            if row["arm"] == arm and row["kind"] == "factual":
                row["mse"] *= (scale / original) ** 2
                row["finite_subset_mse"] = row["mse"]
    result = run(protocol, rows)
    checks = result["comparisons"][comparison]["checks"]
    assert checks["response_gain"]
    assert not checks["factual_regression"]
    assert result["public_collection_progress"]
    assert result["quadratic_workflow_progress"]
    assert not result[comparison]


def test_architecture_tail_limit_can_veto_comparability_despite_aggregate_pass():
    protocol, rows = fixture()
    for row in rows["crazyflow"]:
        if row["kind"] != "factual":
            continue
        scale = None
        if row["arm"] == "quadratic" and row["group"] == "body_rate_rad_s":
            scale = 0.4
        elif row["arm"] == "candidate" and row["group"] == "rotation_entries":
            scale = 0.3
        if scale is not None:
            row["mse"] *= (scale / 0.7) ** 2
            row["finite_subset_mse"] = row["mse"]
    result = run(protocol, rows)
    checks = result["comparisons"]["simpler_recipe_comparable"]["checks"]
    assert checks["response_gain"] and checks["factual_regression"]
    assert checks["primary_simulator_regressions"] and checks["scope_regressions"]
    assert not checks["primary_parent_tail_regressions"]
    assert result["public_collection_progress"]
    assert not result["simpler_recipe_comparable"]


def test_mismatched_bootstrap_draw_hashes_abort(monkeypatch):
    from glassbox.experimental import excited_public_decision

    original = excited_public_decision.paired_reduce
    calls = 0

    def altered(*args, **kwargs):
        nonlocal calls
        result = original(*args, **kwargs)
        calls += 1
        if calls == 2:
            result["bootstrap"]["parent_draws_sha256"] = "changed"
        return result

    monkeypatch.setattr(excited_public_decision, "paired_reduce", altered)
    with pytest.raises(ValueError, match="parent draws differ"):
        run(*fixture())
