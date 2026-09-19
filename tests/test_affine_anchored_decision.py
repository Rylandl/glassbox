"""Numerical parity and labels for the frozen affine-anchor reducer view."""

import copy
import hashlib
import json
from pathlib import Path

import pytest
from test_excited_bilinear_decision import fail_arm, set_scale
from test_excited_bilinear_decision import fixture as previous_fixture

from glassbox.experimental import affine_anchored_decision as decision
from glassbox.experimental import excited_bilinear_decision as pinned
from glassbox.experimental.two_simulator_metrics import aggregate

PROGRESS = "anchored_public_progress"
OLD_PROGRESS = "bilinear_public_progress"


def fixture():
    old, rows = previous_fixture()
    protocol = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "docs/harness/affine-anchored-quadratic-v1.json"
        ).read_text()
    )
    protocol["cells"] = old["cells"]
    protocol["recordings"] = old["recordings"]
    protocol["decision"]["aggregation"] = old["decision"]["aggregation"]
    return protocol, rows


def run(protocol, rows):
    return decision.reduce(
        {sim: aggregate(values) for sim, values in rows.items()}, protocol, rows=rows
    )


def old_view(protocol):
    view = copy.deepcopy(protocol)
    comparisons = view["decision"]["comparisons"]
    comparisons[OLD_PROGRESS] = comparisons.pop(PROGRESS)
    return view


def old_labels(value):
    if isinstance(value, dict):
        return {
            (OLD_PROGRESS if key == PROGRESS else key): old_labels(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [old_labels(item) for item in value]
    return value


@pytest.mark.parametrize(
    "mode",
    [
        "complete",
        "missing_baseline",
        "missing_quadratic",
        "missing_candidate",
        "missing_truth",
        "angular_loss",
        "scope_loss",
        "below_floor",
    ],
)
def test_every_result_and_bootstrap_draw_matches_pinned_reducer(mode):
    protocol, rows = fixture()
    if mode.startswith("missing_") and mode != "missing_truth":
        fail_arm(rows, mode.removeprefix("missing_"))
    elif mode == "missing_truth":
        for row in rows["crazyflow"]:
            if row["parent"] == "crazyflow/p/1":
                row.update(
                    truth_eligible=False,
                    prediction_finite=False,
                    mse=None,
                    finite_subset_mse=None,
                    components=0,
                )
    elif mode == "angular_loss":
        set_scale(rows, 0.9, simulator="crazyflow", group="body_rate_rad_s")
    elif mode == "scope_loss":
        set_scale(rows, 1.6, simulator="crazyflow", kind="response")
    elif mode == "below_floor":
        set_scale(rows, 1e-5, simulator="crazyflow", group="body_rate_rad_s")
        set_scale(
            rows, 1e-4, simulator="crazyflow", arm="quadratic", group="body_rate_rad_s"
        )
    before = copy.deepcopy((protocol, rows, pinned._PAIRS))
    summaries = {sim: aggregate(values) for sim, values in rows.items()}
    actual = decision.reduce(summaries, protocol, rows)
    expected = pinned.reduce(summaries, old_view(protocol), rows)
    assert old_labels(actual) == expected
    assert (protocol, rows, pinned._PAIRS) == before
    assert actual["protocol"] == protocol["id"] == "affine-anchored-quadratic-v1"
    assert actual["qualification"] == protocol["decision"]["public_promotion"]
    assert actual["diagnostic_mechanism_success"] is None
    assert actual["public_promotion"] is False
    assert (
        actual["verification_status"]
        == "requires_integrity_replay_tamper_and_focused_tests"
    )
    assert OLD_PROGRESS not in json.dumps(actual, allow_nan=False)


def test_independent_ratios_and_report_labels():
    result = run(*fixture())
    expected = {
        PROGRESS: 0.6,
        "quadratic_gain_retention": 0.75,
        "quadratic_public_context": 0.8,
    }
    for name, ratio in expected.items():
        comparison = result["comparisons"][name]
        assert comparison["weighted_geometric_mean_ratios"] == pytest.approx(
            {"factual": ratio, "response": ratio}
        )
        assert len(comparison["physical_comparisons"]) == 12
        assert len(comparison["parent_error_tails"]) == 24
        assert len(comparison["tail_checks"]) == 12
    assert (
        result[PROGRESS]
        and result["quadratic_gain_retention"]
        and result["angular_repair"]
    )
    assert result["residual_criteria_pass"]
    assert result["comparisons"][PROGRESS]["baseline_fields_mean"] == "baseline"
    assert result["comparisons"][PROGRESS]["candidate_fields_mean"] == "candidate"
    assert result["shared_bootstrap_parent_draws_verified"]
    assert len(set(result["bootstrap_comparison_hashes"].values())) == 1
    assert result["angular_repair_comparison"]["bootstrap"][
        "point_estimates_only_for_acceptance"
    ]


def test_contextual_quadratic_failure_cannot_veto_anchor_success():
    protocol, rows = fixture()
    set_scale(
        rows, 1.6, simulator="crazyflow", arm="quadratic", group="body_rate_rad_s"
    )
    result = run(protocol, rows)
    assert not result["quadratic_public_context"]
    assert result["residual_criteria_pass"]


def test_unavailable_quadratic_preserves_public_relative_gain():
    protocol, rows = fixture()
    fail_arm(rows, "quadratic")
    result = run(protocol, rows)
    assert result[PROGRESS]
    assert not result["quadratic_gain_retention"]
    assert not result["angular_repair"]
    assert not result["residual_criteria_pass"]


@pytest.mark.parametrize("offset,expected", [(0, True), (1e-6, False)])
def test_angular_factual_eighty_percent_boundary_is_unchanged(offset, expected):
    protocol, rows = fixture()
    for row in rows["crazyflow"]:
        if (
            row["kind"] == "factual"
            and row["group"] == "body_rate_rad_s"
            and row["arm"] in ("candidate", "quadratic")
        ):
            value = 0.8 + offset if row["arm"] == "candidate" else 1.0
            row["mse"] = row["finite_subset_mse"] = value**2
    result = run(protocol, rows)
    assert result["angular_repair"] is expected
    assert result[PROGRESS]


def test_raw_rows_and_full_summary_consistency_remain_required():
    protocol, rows = fixture()
    summaries = {sim: aggregate(values) for sim, values in rows.items()}
    with pytest.raises(ValueError, match="full metric rows"):
        decision.reduce(summaries, protocol)
    row = next(row for row in rows["crazyflow"] if row["arm"] == "hold")
    row["mse"] *= 2
    with pytest.raises(ValueError, match="summary differs"):
        decision.reduce(summaries, protocol, rows)


def test_mismatched_query_roster_is_still_an_integrity_failure():
    protocol, rows = fixture()
    rows["crazyflow"].remove(
        next(row for row in rows["crazyflow"] if row["arm"] == "candidate")
    )
    with pytest.raises(ValueError, match="unmatched"):
        run(protocol, rows)


def test_old_progress_name_is_rejected_in_input_protocol():
    protocol, rows = fixture()
    with pytest.raises(ValueError, match="anchored comparison roster"):
        run(old_view(protocol), rows)


def test_inherited_reducer_source_pin_and_frozen_protocol_identity():
    root = Path(__file__).resolve().parents[1]
    path = root / "docs/harness/affine-anchored-quadratic-v1.json"
    assert (
        hashlib.sha256(path.read_bytes()).hexdigest()
        == "d0ec7e1af8ee5877de1a84fb49f2e484070c9d790b5860b2aa2c558b204989f9"
    )
    protocol = json.loads(path.read_text())
    name = "src/glassbox/experimental/excited_bilinear_decision.py"
    assert (
        hashlib.sha256((root / name).read_bytes()).hexdigest()
        == protocol["inherited_source_sha256"][name]
    )
