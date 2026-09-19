"""Independent expectations for frozen bilinear retention and angular repair."""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from glassbox.experimental import excited_bilinear_decision as decision
from glassbox.experimental.two_simulator_metrics import aggregate


def fixture(*, candidate=0.6, quadratic=0.8):
    path = (
        Path(__file__).resolve().parents[1]
        / "docs/harness/excited-bilinear-ablation-v1.json"
    )
    protocol = json.loads(path.read_text())
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
    return decision.reduce(
        {sim: aggregate(values) for sim, values in rows.items()}, protocol, rows
    )


def set_scale(rows, scale, *, simulator=None, arm="candidate", kind=None, group=None):
    for sim, values in rows.items():
        for row in values:
            if (
                (simulator is None or sim == simulator)
                and row["arm"] == arm
                and (kind is None or row["kind"] == kind)
                and (group is None or row["group"] == group)
            ):
                parent = int(row["parent"].rsplit("/", 1)[1])
                row["mse"] = ((parent + 1) * scale) ** 2
                row["finite_subset_mse"] = row["mse"]


def fail_arm(rows, arm):
    for values in rows.values():
        for row in values:
            if row["arm"] == arm:
                row.update(prediction_finite=False, mse=None, finite_subset_mse=None)


def test_three_directions_and_independent_physical_expectations():
    protocol, rows = fixture()
    before = copy.deepcopy((protocol, rows))
    result = run(protocol, rows)
    for name, expected, numerator, denominator in (
        ("bilinear_public_progress", 0.6, "candidate", "baseline"),
        ("quadratic_gain_retention", 0.75, "candidate", "quadratic"),
        ("quadratic_public_context", 0.8, "quadratic", "baseline"),
    ):
        comparison = result["comparisons"][name]
        assert comparison["numerator_arm"] == numerator
        assert comparison["denominator_arm"] == denominator
        assert comparison["baseline_fields_mean"] == denominator
        assert comparison["candidate_fields_mean"] == numerator
        assert (
            comparison["remaining_promotion_requirements"]
            == protocol["decision"]["public_promotion"]
        )
        for kind in ("factual", "response"):
            assert comparison["weighted_geometric_mean_ratios"][kind] == pytest.approx(
                expected
            )
            np.testing.assert_allclose(
                comparison["bootstrap"]["aggregates"][kind]["draws"], expected
            )
        assert len(comparison["tail_checks"]) == 12
        assert len(comparison["parent_error_tails"]) == 24
        assert len(comparison["physical_comparisons"]) == 12
    repair = result["angular_repair_comparison"]
    for kind in ("factual", "response"):
        assert repair["physical_comparisons"][kind]["candidate_rmse"] == pytest.approx(
            np.sqrt(2.5) * 0.6
        )
        assert repair["parent_tail_comparisons"][kind][
            "candidate_p95"
        ] == pytest.approx(1.95 * 0.6)
        assert repair["parent_tail_comparisons"][kind]["ratio"] == pytest.approx(0.75)
    assert result["residual_criteria_pass"]
    assert result["diagnostic_mechanism_success"] is None
    assert (
        result["verification_status"]
        == "requires_integrity_replay_tamper_and_focused_tests"
    )
    assert result["public_promotion"] is False
    assert (protocol, rows) == before
    assert result == run(protocol, rows)
    json.dumps(result, allow_nan=False)


def test_context_failure_is_not_a_combined_veto():
    protocol, rows = fixture()
    set_scale(
        rows, 1.6, simulator="crazyflow", arm="quadratic", group="body_rate_rad_s"
    )
    result = run(protocol, rows)
    assert not result["quadratic_public_context"]
    assert (
        result["bilinear_public_progress"]
        and result["quadratic_gain_retention"]
        and result["angular_repair"]
    )
    assert result["residual_criteria_pass"]


@pytest.mark.parametrize("kind,limit", [("factual", 0.8), ("response", 1.05)])
@pytest.mark.parametrize("offset,expected", [(-1e-6, True), (0, True), (1e-6, False)])
def test_targeted_angular_thresholds(kind, limit, offset, expected):
    protocol, rows = fixture()
    # Unit denominator and equal parents make the exact boundary representable;
    # acceptance retains the pinned reducer's literal <= comparison.
    for row in rows["crazyflow"]:
        if row["kind"] == kind and row["group"] == "body_rate_rad_s":
            if row["arm"] in ("candidate", "quadratic"):
                value = limit + offset if row["arm"] == "candidate" else 1.0
                row["mse"] = row["finite_subset_mse"] = value**2
    result = run(protocol, rows)
    repair = result["angular_repair_comparison"]
    assert repair["checks"][f"{kind}_aggregate_rmse"] is expected
    assert repair["checks"][f"{kind}_parent_rmse_p95"] is expected
    assert result["angular_repair"] is expected
    assert result["bilinear_public_progress"]


@pytest.mark.parametrize(
    "values,mean_pass,tail_pass", [([0.0, 1.4], True, False), ([1.2, 1.2], False, True)]
)
def test_angular_mean_and_tail_are_separate(values, mean_pass, tail_pass):
    protocol, rows = fixture()
    for row in rows["crazyflow"]:
        if (
            row["arm"] == "candidate"
            and row["kind"] == "factual"
            and row["group"] == "body_rate_rad_s"
        ):
            n = int(row["parent"].rsplit("/", 1)[1])
            row["mse"] = row["finite_subset_mse"] = values[n] ** 2
    checks = run(protocol, rows)["angular_repair_comparison"]["checks"]
    assert checks["factual_aggregate_rmse"] is mean_pass
    assert checks["factual_parent_rmse_p95"] is tail_pass


@pytest.mark.parametrize("kind", ["factual", "response"])
@pytest.mark.parametrize("ratio,expected", [(1.049999, True), (1.050001, False)])
def test_retention_five_percent_limit_for_each_kind(kind, ratio, expected):
    protocol, rows = fixture()
    set_scale(rows, 0.8 * ratio, kind=kind)
    result = run(protocol, rows)
    assert result["quadratic_gain_retention"] is expected
    assert result["comparisons"]["quadratic_gain_retention"][
        "weighted_geometric_mean_ratios"
    ][kind] == pytest.approx(ratio)


@pytest.mark.parametrize("kind,limit", [("response", 0.9), ("factual", 1.05)])
@pytest.mark.parametrize("offset,expected", [(-1e-6, True), (1e-6, False)])
def test_public_progress_uses_unchanged_limits(kind, limit, offset, expected):
    protocol, rows = fixture()
    set_scale(rows, limit + offset, kind=kind)
    assert run(protocol, rows)["bilinear_public_progress"] is expected


@pytest.mark.parametrize(
    "arm,preserved",
    [
        ("quadratic", "bilinear_public_progress"),
        ("baseline", "quadratic_gain_retention"),
        ("candidate", "quadratic_public_context"),
    ],
)
def test_prediction_failure_is_pairwise_and_keeps_other_evidence(arm, preserved):
    protocol, rows = fixture()
    fail_arm(rows, arm)
    result = run(protocol, rows)
    assert result[preserved]
    assert not result["residual_criteria_pass"]
    assert not result["shared_bootstrap_parent_draws_verified"]
    assert result["available_bootstrap_parent_draws_match"]
    for name, comparison in result["comparisons"].items():
        if name != preserved:
            assert not comparison["pairwise_predictions_available"]
            assert (
                comparison["bootstrap"]["status"]
                == "unavailable_due_to_prediction_failure"
            )
    assert result["angular_repair"] is (arm == "baseline")


def test_one_failed_eligible_prediction_invalidates_pair_not_finite_subset():
    protocol, rows = fixture()
    row = next(row for row in rows["crazyflow"] if row["arm"] == "candidate")
    row.update(prediction_finite=False, mse=None, finite_subset_mse=0.0)
    result = run(protocol, rows)
    assert not result["bilinear_public_progress"]
    assert not result["quadratic_gain_retention"]
    assert not result["angular_repair"]
    assert result["quadratic_public_context"]


@pytest.mark.parametrize("arm", ["baseline", "candidate", "quadratic"])
def test_missing_slot_is_integrity_failure(arm):
    protocol, rows = fixture()
    rows["crazyflow"].remove(
        next(row for row in rows["crazyflow"] if row["arm"] == arm)
    )
    with pytest.raises(ValueError, match="unmatched"):
        run(protocol, rows)


def test_truth_and_probe_masks_must_match():
    protocol, rows = fixture()
    next(
        row
        for row in rows["crazyflow"]
        if row["arm"] == "quadratic" and row["kind"] == "response"
    )["pair_nonweak"] = True
    with pytest.raises(ValueError, match="cohort differs"):
        run(protocol, rows)


@pytest.mark.parametrize("all_missing", [False, True])
def test_missing_truth_is_retained_and_target_bootstrap_matches(all_missing):
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
    repair = result["angular_repair_comparison"]
    physical = repair["physical_comparisons"]["factual"]
    assert physical["planned"] == 2
    assert physical["truth_eligible"] == (0 if all_missing else 1)
    assert physical["conditional_on_incomplete_truth"]
    assert result["angular_repair"] is not all_missing
    if all_missing:
        assert repair["bootstrap"]["status"] == "unavailable_due_to_prediction_failure"
    else:
        assert result["shared_bootstrap_parent_draws_verified"]
        for values in repair["bootstrap"]["statistics"].values():
            assert 0 < values["unavailable_draws"] < 1000
            assert values["percentile_95_interval"] == pytest.approx([0.75, 0.75])


def test_all_shared_bootstrap_indices_match_independent_regeneration():
    protocol, rows = fixture()
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
    assert result["bootstrap_parent_draws_sha256"] == digest.hexdigest()
    assert set(result["bootstrap_comparison_hashes"].values()) == {digest.hexdigest()}
    for values in result["angular_repair_comparison"]["bootstrap"][
        "statistics"
    ].values():
        assert values["available_draws"] == 1000
        np.testing.assert_allclose(values["draws"], 0.75)


def test_common_physical_floor_prevents_tiny_residual_ratio_claim():
    protocol, rows = fixture()
    set_scale(rows, 1e-5, simulator="crazyflow", group="body_rate_rad_s")
    set_scale(
        rows, 1e-4, simulator="crazyflow", arm="quadratic", group="body_rate_rad_s"
    )
    result = run(protocol, rows)
    repair = result["angular_repair_comparison"]
    assert not result["angular_repair"]
    for kind in ("factual", "response"):
        assert repair["physical_comparisons"][kind]["ratio"] == 1.0
        assert repair["parent_tail_comparisons"][kind]["ratio"] == 1.0
    for values in repair["bootstrap"]["statistics"].values():
        assert set(values["draws"]) == {1.0}


def test_weak_responses_remain_in_angular_objective():
    protocol, rows = fixture()
    assert all(not row["pair_nonweak"] for values in rows.values() for row in values)
    set_scale(
        rows, 0.85, simulator="crazyflow", kind="response", group="body_rate_rad_s"
    )
    result = run(protocol, rows)
    assert result["bilinear_public_progress"]
    assert not result["angular_repair"]
    assert result["angular_repair_comparison"]["physical_comparisons"]["response"][
        "ratio"
    ] == pytest.approx(0.85 / 0.8)


def test_local_channel_loss_is_not_an_all_case_veto():
    protocol, rows = fixture()
    set_scale(rows, 1.1, simulator="crazyflow", group="velocity_m_s")
    result = run(protocol, rows)
    assert result["residual_criteria_pass"]
    assert result["comparisons"]["bilinear_public_progress"][
        "weighted_geometric_mean_ratios"
    ]["factual"] == pytest.approx((1.1 * 0.6**5) ** (1 / 6))


def test_primary_and_scope_guards_survive_broad_gain():
    protocol, rows = fixture(candidate=0.1)
    set_scale(rows, 1.6, simulator="crazyflow", kind="response")
    comparison = run(protocol, rows)["comparisons"]["bilinear_public_progress"]
    assert comparison["checks"]["response_gain"]
    for key in (
        "primary_simulator_regressions",
        "scope_regressions",
        "primary_parent_tail_regressions",
    ):
        assert not comparison["checks"][key]


def test_full_summary_consistency_includes_hold():
    protocol, rows = fixture()
    summaries = {sim: aggregate(values) for sim, values in rows.items()}
    row = next(row for row in rows["crazyflow"] if row["arm"] == "hold")
    row["mse"] *= 2
    with pytest.raises(ValueError, match="summary differs"):
        decision.reduce(summaries, protocol, rows)


def test_raw_rows_are_mandatory():
    protocol, rows = fixture()
    with pytest.raises(ValueError, match="full metric rows"):
        decision.reduce(
            {sim: aggregate(values) for sim, values in rows.items()}, protocol
        )


def test_historical_reducer_globals_are_unchanged():
    from glassbox.experimental import state_input_decision

    arms = state_input_decision.ARMS
    run(*fixture())
    assert state_input_decision.ARMS is arms
    assert arms == ("baseline", "candidate")


def test_mismatched_bootstrap_hash_is_rejected(monkeypatch):
    original = decision._angular_bootstrap

    def changed(summaries, protocol):
        result = original(summaries, protocol)
        result["parent_draws_sha256"] = "altered"
        return result

    monkeypatch.setattr(decision, "_angular_bootstrap", changed)
    with pytest.raises(ValueError, match="bootstrap parent draws differ"):
        run(*fixture())


def test_target_bootstrap_mean_and_quantile_match_independent_draw_reduction():
    protocol, rows = fixture()
    candidate_values = np.array([0.2, 1.0])
    quadratic_values = np.array([0.8, 1.6])
    for row in rows["crazyflow"]:
        if (
            row["kind"] == "factual"
            and row["group"] == "body_rate_rad_s"
            and row["arm"] == "candidate"
        ):
            n = int(row["parent"].rsplit("/", 1)[1])
            row["mse"] = row["finite_subset_mse"] = candidate_values[n] ** 2
    result = run(protocol, rows)
    rng = np.random.default_rng(20260918)
    rng.integers(2, size=(1000, 2), dtype=np.int64)  # Cascade first, as frozen.
    chosen = rng.integers(2, size=(1000, 2), dtype=np.int64)
    numerator, denominator = candidate_values[chosen], quadratic_values[chosen]
    expected_mean = np.sqrt(
        np.mean(numerator**2, axis=1) / np.mean(denominator**2, axis=1)
    )
    expected_p95 = np.quantile(numerator, 0.95, axis=1, method="linear") / np.quantile(
        denominator, 0.95, axis=1, method="linear"
    )
    statistics = result["angular_repair_comparison"]["bootstrap"]["statistics"]
    np.testing.assert_allclose(
        statistics["factual_aggregate_rmse"]["draws"], expected_mean
    )
    np.testing.assert_allclose(
        statistics["factual_parent_rmse_p95"]["draws"], expected_p95
    )
    assert not np.allclose(expected_mean, expected_p95)


def test_missing_target_truth_does_not_drop_other_eligible_queries():
    protocol, rows = fixture()
    for row in rows["crazyflow"]:
        if row["group"] == "body_rate_rad_s":
            row.update(
                truth_eligible=False,
                prediction_finite=False,
                mse=None,
                finite_subset_mse=None,
                components=0,
            )
    result = run(protocol, rows)
    repair = result["angular_repair_comparison"]
    assert repair["checks"]["finite_eligible_predictions"]
    assert not result["angular_repair"]
    assert result["shared_bootstrap_parent_draws_verified"]
    for value in repair["bootstrap"]["statistics"].values():
        assert value["available_draws"] == 0
        assert value["unavailable_draws"] == 1000
        assert value["percentile_95_interval"] is None
        assert set(value["draws"]) == {None}


def test_shifted_and_cumulative_tails_remain_descriptive():
    protocol, rows = fixture()
    protocol["decision"]["aggregation"]["scope_weights"] = {
        "primary": 0.5,
        "wind_shift": 0.5,
    }
    for sim, values in rows.items():
        protocol["cells"][sim].append({"id": "wind", "group": "wind_shift"})
        protocol["recordings"].extend(
            dict(id=f"{sim}/wind/{n}", simulator=sim, role="test", cell="wind")
            for n in range(2)
        )
        values.extend(
            dict(
                row,
                scope="wind_shift",
                cell="wind",
                parent=row["parent"].replace("/p/", "/wind/"),
            )
            for row in list(values)
        )
        values.extend(dict(row, statistic="cumulative") for row in list(values))
    result = run(protocol, rows)
    assert result["shared_bootstrap_parent_draws_verified"]
    for comparison in result["comparisons"].values():
        tails = comparison["parent_error_tails"]
        assert {tail["scope"] for tail in tails} == {"primary", "wind_shift"}
        assert {tail["statistic"] for tail in tails} == {"endpoint", "cumulative"}
        assert len(tails) == 96
        assert len(comparison["tail_checks"]) == 12


def test_angular_uncertainty_has_no_significance_gate(monkeypatch):
    original = decision._angular_bootstrap

    def wider(summaries, protocol):
        result = original(summaries, protocol)
        for statistic in result["statistics"].values():
            statistic["percentile_95_interval"] = [0.2, 2.0]
        return result

    monkeypatch.setattr(decision, "_angular_bootstrap", wider)
    result = run(*fixture())
    assert result["angular_repair"]
    assert result["residual_criteria_pass"]
