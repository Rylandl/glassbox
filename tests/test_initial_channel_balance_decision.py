"""Independent ratio, cohort and availability checks for channel balancing."""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from test_excited_bilinear_decision import fail_arm, set_scale
from test_excited_bilinear_decision import fixture as prior_fixture

from glassbox.experimental import excited_bilinear_decision as pinned
from glassbox.experimental import initial_channel_balance_decision as decision
from glassbox.experimental import state_input_decision
from glassbox.experimental.two_simulator_metrics import aggregate


def fixture(*, candidate=0.6, quadratic=0.8, anchored=0.85):
    old, rows = prior_fixture(candidate=candidate, quadratic=quadratic)
    protocol = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "docs/harness/initial-channel-balance-v1.json"
        ).read_text()
    )
    protocol["cells"] = old["cells"]
    protocol["recordings"] = old["recordings"]
    protocol["decision"]["aggregation"] = old["decision"]["aggregation"]
    for values in rows.values():
        values.extend(
            dict(row, arm="anchored")
            for row in list(values)
            if row["arm"] == "baseline"
        )
    set_scale(rows, anchored, arm="anchored")
    return protocol, rows


def run(protocol, rows):
    return decision.reduce(
        {sim: aggregate(values) for sim, values in rows.items()}, protocol, rows=rows
    )


def test_all_five_directions_have_independent_expected_ratios_and_full_reports():
    protocol, rows = fixture()
    before = copy.deepcopy((protocol, rows, pinned._PAIRS, state_input_decision.ARMS))
    result = run(protocol, rows)
    for name, numerator, denominator, expected in (
        ("balanced_public_progress", "candidate", "baseline", 0.6),
        ("anchored_gain_retention", "candidate", "anchored", 0.6 / 0.85),
        ("quadratic_gain_retention", "candidate", "quadratic", 0.75),
        ("anchored_public_context", "anchored", "baseline", 0.85),
        ("quadratic_public_context", "quadratic", "baseline", 0.8),
    ):
        comparison = result["comparisons"][name]
        assert (
            comparison["numerator_arm"]
            == comparison["candidate_fields_mean"]
            == numerator
        )
        assert (
            comparison["denominator_arm"]
            == comparison["baseline_fields_mean"]
            == denominator
        )
        assert result[name]
        assert len(comparison["physical_comparisons"]) == 12
        assert len(comparison["parent_error_tails"]) == 24
        assert len(comparison["tail_checks"]) == 12
        for kind in ("factual", "response"):
            assert comparison["weighted_geometric_mean_ratios"][kind] == pytest.approx(
                expected
            )
            np.testing.assert_allclose(
                comparison["bootstrap"]["aggregates"][kind]["draws"], expected
            )
    repair = result["angular_repair_comparison"]
    assert repair["denominator_arm"] == repair["baseline_fields_mean"] == "anchored"
    assert repair["policy"]["denominator"] == "anchored"
    for kind in ("factual", "response"):
        physical = repair["physical_comparisons"][kind]
        assert physical["baseline_rmse"] == pytest.approx(np.sqrt(2.5) * 0.85)
        assert physical["candidate_rmse"] == pytest.approx(np.sqrt(2.5) * 0.6)
        assert repair["parent_tail_comparisons"][kind]["baseline_p95"] == pytest.approx(
            1.95 * 0.85
        )
        assert repair["parent_tail_comparisons"][kind]["ratio"] == pytest.approx(
            0.6 / 0.85
        )
    assert result["anchored_mechanism_progress"] and result["residual_criteria_pass"]
    assert result["diagnostic_mechanism_success"] is None
    assert (
        result["verification_status"]
        == "requires_integrity_replay_tamper_and_focused_tests"
    )
    assert result["public_promotion"] is False
    assert (protocol, rows, pinned._PAIRS, state_input_decision.ARMS) == before
    assert result == run(protocol, rows)
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    "anchored,quadratic,expected", [(1.0, 0.7, True), (0.7, 1.0, False)]
)
def test_angular_gate_uses_actual_anchored_not_quadratic_denominator(
    anchored, quadratic, expected
):
    result = run(*fixture(anchored=anchored, quadratic=quadratic))
    assert result["angular_repair"] is expected
    assert result["anchored_mechanism_progress"] is expected
    for kind in ("factual", "response"):
        assert result["angular_repair_comparison"]["physical_comparisons"][kind][
            "ratio"
        ] == pytest.approx(0.6 / anchored)
    for distribution in result["angular_repair_comparison"]["bootstrap"][
        "statistics"
    ].values():
        np.testing.assert_allclose(distribution["draws"], 0.6 / anchored)


@pytest.mark.parametrize("reference", ["anchored", "quadratic"])
def test_contextual_reference_losses_cannot_veto_candidate_success(reference):
    protocol, rows = fixture()
    set_scale(rows, 1.6, simulator="crazyflow", arm=reference, group="body_rate_rad_s")
    result = run(protocol, rows)
    assert not result[f"{reference}_public_context"]
    assert result["residual_criteria_pass"]


@pytest.mark.parametrize("arm", ["baseline", "anchored", "quadratic", "candidate"])
def test_failed_arm_only_invalidates_affected_pairs(arm):
    protocol, rows = fixture()
    fail_arm(rows, arm)
    result = run(protocol, rows)
    for name, comparison in result["comparisons"].items():
        affected = arm in (comparison["numerator_arm"], comparison["denominator_arm"])
        assert comparison["pairwise_predictions_available"] is not affected
        assert result[name] is not affected
        if affected:
            assert (
                comparison["bootstrap"]["status"]
                == "unavailable_due_to_prediction_failure"
            )
    assert result["angular_repair"] is (arm not in ("candidate", "anchored"))
    assert not result["residual_criteria_pass"]
    assert not result["shared_bootstrap_parent_draws_verified"]
    assert result["available_bootstrap_parent_draws_match"]


def test_hold_failure_is_visible_but_not_an_extra_acceptance_gate():
    protocol, rows = fixture()
    fail_arm(rows, "hold")
    result = run(protocol, rows)
    assert result["residual_criteria_pass"]
    assert all(
        cohort["arms"]["hold"]["failed_rows"] == 12
        for cohort in result["cohorts"].values()
    )


@pytest.mark.parametrize(
    "arm", ["baseline", "anchored", "quadratic", "candidate", "hold"]
)
def test_missing_any_arm_slot_is_integrity_error(arm):
    protocol, rows = fixture()
    rows["crazyflow"].remove(
        next(row for row in rows["crazyflow"] if row["arm"] == arm)
    )
    with pytest.raises(ValueError, match="unmatched"):
        run(protocol, rows)


@pytest.mark.parametrize(
    "field,value",
    [
        ("truth_eligible", False),
        ("pair_nonweak", True),
        ("components", 2),
        ("horizon_steps", 4),
    ],
)
def test_hold_masks_are_part_of_full_five_arm_cohort(field, value):
    protocol, rows = fixture()
    row = next(row for row in rows["crazyflow"] if row["arm"] == "hold")
    row[field] = value
    if field == "truth_eligible":
        row.update(mse=None, finite_subset_mse=None, components=0)
    with pytest.raises(ValueError, match="cohort differs"):
        run(protocol, rows)


def test_unknown_arm_and_extra_simulator_are_rejected():
    protocol, rows = fixture()
    rows["crazyflow"].append(dict(rows["crazyflow"][0], arm="unexpected"))
    with pytest.raises(ValueError, match="five-arm roster"):
        run(protocol, rows)
    protocol, rows = fixture()
    rows["unexpected"] = []
    with pytest.raises(ValueError, match="simulator roster"):
        run(protocol, rows)


@pytest.mark.parametrize("kind,limit", [("factual", 0.8), ("response", 1.05)])
@pytest.mark.parametrize("offset,expected", [(-1e-6, True), (0.0, True), (1e-6, False)])
def test_angular_point_thresholds(kind, limit, offset, expected):
    protocol, rows = fixture()
    for row in rows["crazyflow"]:
        if (
            row["kind"] == kind
            and row["group"] == "body_rate_rad_s"
            and row["arm"] in ("candidate", "anchored")
        ):
            value = limit + offset if row["arm"] == "candidate" else 1.0
            row["mse"] = row["finite_subset_mse"] = value**2
    result = run(protocol, rows)
    checks = result["angular_repair_comparison"]["checks"]
    assert checks[f"{kind}_aggregate_rmse"] is expected
    assert checks[f"{kind}_parent_rmse_p95"] is expected
    assert result["angular_repair"] is expected


@pytest.mark.parametrize("reference", ["anchored", "quadratic"])
@pytest.mark.parametrize("kind", ["factual", "response"])
@pytest.mark.parametrize("ratio,expected", [(1.049999, True), (1.050001, False)])
def test_both_retention_ratios_allow_five_percent_each_kind(
    reference, kind, ratio, expected
):
    protocol, rows = fixture(anchored=0.8)
    set_scale(rows, 0.8 * ratio, kind=kind)
    result = run(protocol, rows)
    assert result[f"{reference}_gain_retention"] is expected
    assert result["comparisons"][f"{reference}_gain_retention"][
        "weighted_geometric_mean_ratios"
    ][kind] == pytest.approx(ratio)


@pytest.mark.parametrize("kind,limit", [("factual", 1.05), ("response", 0.9)])
@pytest.mark.parametrize("offset,expected", [(-1e-6, True), (1e-6, False)])
def test_public_progress_limits(kind, limit, offset, expected):
    protocol, rows = fixture()
    set_scale(rows, limit + offset, kind=kind)
    assert run(protocol, rows)["balanced_public_progress"] is expected


def test_anchored_retention_does_not_require_an_extra_ten_percent_response_gain():
    protocol, rows = fixture()
    set_scale(rows, 0.85, kind="response")
    result = run(protocol, rows)
    assert result["anchored_gain_retention"]
    assert result["angular_repair"]
    assert result["anchored_mechanism_progress"]
    assert (
        result["comparisons"]["anchored_gain_retention"][
            "weighted_geometric_mean_ratios"
        ]["response"]
        == 1.0
    )


def test_broad_primary_scope_and_tail_guards_are_preserved():
    protocol, rows = fixture(candidate=0.1)
    set_scale(rows, 1.6, simulator="crazyflow", kind="response")
    result = run(protocol, rows)
    comparison = result["comparisons"]["balanced_public_progress"]
    assert comparison["checks"]["response_gain"]
    for check in (
        "primary_simulator_regressions",
        "scope_regressions",
        "primary_parent_tail_regressions",
    ):
        assert not comparison["checks"][check]
    assert not result["balanced_public_progress"]


def test_local_loss_is_not_an_all_case_veto_and_weak_probes_remain():
    protocol, rows = fixture()
    assert all(not row["pair_nonweak"] for values in rows.values() for row in values)
    set_scale(rows, 1.1, simulator="crazyflow", group="velocity_m_s")
    result = run(protocol, rows)
    assert result["residual_criteria_pass"]
    assert result["comparisons"]["balanced_public_progress"][
        "weighted_geometric_mean_ratios"
    ]["response"] == pytest.approx((1.1 * 0.6**5) ** (1 / 6))
    set_scale(
        rows, 0.9, simulator="crazyflow", kind="response", group="body_rate_rad_s"
    )
    assert not run(protocol, rows)["angular_repair"]


def test_angular_physical_floors_prevent_tiny_ratio_claims():
    protocol, rows = fixture()
    for arm, value in (("anchored", 1e-4), ("candidate", 1e-5)):
        set_scale(rows, value, simulator="crazyflow", arm=arm, group="body_rate_rad_s")
    result = run(protocol, rows)
    assert not result["angular_repair"]
    repair = result["angular_repair_comparison"]
    for kind in ("factual", "response"):
        assert repair["physical_comparisons"][kind]["ratio"] == 1.0
        assert repair["parent_tail_comparisons"][kind]["ratio"] == 1.0
    for distribution in repair["bootstrap"]["statistics"].values():
        assert set(distribution["draws"]) == {1.0}


@pytest.mark.parametrize("all_missing", [False, True])
def test_missing_truth_stays_in_shared_draws_and_never_creates_a_pass(all_missing):
    protocol, rows = fixture()
    for row in rows["crazyflow"]:
        if all_missing or row["parent"] == "crazyflow/p/1":
            row.update(
                truth_eligible=False,
                prediction_finite=False,
                mse=None,
                finite_subset_mse=None,
                components=0,
            )
    result = run(protocol, rows)
    assert result["residual_criteria_pass"] is not all_missing
    physical = result["angular_repair_comparison"]["physical_comparisons"]["factual"]
    assert physical["planned"] == 2
    assert physical["truth_eligible"] == (0 if all_missing else 1)
    assert physical["conditional_on_incomplete_truth"]
    if not all_missing:
        for distribution in result["angular_repair_comparison"]["bootstrap"][
            "statistics"
        ].values():
            assert 0 < distribution["unavailable_draws"] < 1000
            assert distribution["percentile_95_interval"] == pytest.approx(
                [0.6 / 0.85] * 2
            )


def test_all_six_bootstrap_hashes_and_angular_draws_match_independent_regeneration():
    protocol, rows = fixture()
    candidate_values = np.array([0.2, 1.0])
    anchored_values = np.array([0.85, 1.7])
    for row in rows["crazyflow"]:
        if (
            row["arm"] == "candidate"
            and row["kind"] == "factual"
            and row["group"] == "body_rate_rad_s"
        ):
            row["mse"] = row["finite_subset_mse"] = (
                candidate_values[int(row["parent"].rsplit("/", 1)[1])] ** 2
            )
    result = run(protocol, rows)
    rng = np.random.default_rng(20260918)
    digest = hashlib.sha256()
    for sim in ("cascade", "crazyflow"):
        parents = [f"{sim}/p/0", f"{sim}/p/1"]
        chosen = rng.integers(2, size=(1000, 2), dtype=np.int64)
        digest.update(
            json.dumps([sim, "primary", "p", parents], separators=(",", ":")).encode()
        )
        digest.update(chosen.astype("<i8").tobytes())
    assert len(result["bootstrap_comparison_hashes"]) == 6
    assert set(result["bootstrap_comparison_hashes"].values()) == {digest.hexdigest()}
    assert result["shared_bootstrap_parent_draws_verified"]
    numerator, denominator = candidate_values[chosen], anchored_values[chosen]
    statistics = result["angular_repair_comparison"]["bootstrap"]["statistics"]
    np.testing.assert_allclose(
        statistics["factual_aggregate_rmse"]["draws"],
        np.sqrt(np.mean(numerator**2, axis=1) / np.mean(denominator**2, axis=1)),
    )
    np.testing.assert_allclose(
        statistics["factual_parent_rmse_p95"]["draws"],
        np.quantile(numerator, 0.95, axis=1, method="linear")
        / np.quantile(denominator, 0.95, axis=1, method="linear"),
    )


def test_bootstrap_hash_disagreement_is_integrity_error(monkeypatch):
    original = decision._anchored_angular_repair

    def changed(*args):
        result = original(*args)
        result["bootstrap"]["parent_draws_sha256"] = "altered"
        return result

    monkeypatch.setattr(decision, "_anchored_angular_repair", changed)
    with pytest.raises(ValueError, match="bootstrap parent draws differ"):
        run(*fixture())


def test_full_summary_and_raw_rows_required():
    protocol, rows = fixture()
    summaries = {sim: aggregate(values) for sim, values in rows.items()}
    with pytest.raises(ValueError, match="full metric rows"):
        decision.reduce(summaries, protocol)
    next(row for row in rows["crazyflow"] if row["arm"] == "hold")["mse"] *= 2
    with pytest.raises(ValueError, match="summary differs"):
        decision.reduce(summaries, protocol, rows)


@pytest.mark.parametrize(
    "change", ["angular_denominator", "comparison_direction", "comparison_roster"]
)
def test_frozen_arms_and_comparison_roster_are_enforced(change):
    protocol, rows = fixture()
    if change == "angular_denominator":
        protocol["decision"]["angular_repair"]["denominator"] = "quadratic"
    elif change == "comparison_direction":
        protocol["decision"]["comparisons"]["anchored_gain_retention"]["numerator"] = (
            "anchored"
        )
    else:
        del protocol["decision"]["comparisons"]["anchored_public_context"]
    with pytest.raises(ValueError, match="frozen"):
        run(protocol, rows)


def test_frozen_protocol_and_used_reducer_sources_are_pinned():
    root = Path(__file__).resolve().parents[1]
    path = root / "docs/harness/initial-channel-balance-v1.json"
    assert (
        hashlib.sha256(path.read_bytes()).hexdigest()
        == "822a28d5a20596026851ec9e5f64f9c304e09a68b67045b44bc3ee3633d056dc"
    )
    protocol = json.loads(path.read_text())
    for name in ("excited_bilinear_decision", "state_input_decision"):
        relative = f"src/glassbox/experimental/{name}.py"
        pins = {
            **protocol["sources"],
            **protocol["parent_source_sha256"],
            **protocol["inherited_source_sha256"],
        }
        assert (
            hashlib.sha256((root / relative).read_bytes()).hexdigest() == pins[relative]
        )
