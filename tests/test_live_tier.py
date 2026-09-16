"""Live-tier contracts: the swap gate, the segmentation, the rule, the replay.

Nothing above the Cascade marker at the bottom of this module needs the
simulator. The swap gate is exercised on fabricated block scores, the
before/after segmentation on a fabricated tracking trace, the decision on
fabricated trial results, and the digest gate on the committed manifest.
"""

import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest

from glassbox.experimental import harness

MANIFEST = Path(__file__).resolve().parents[1] / "docs/harness/live-v1.json"
RMSE_KEYS = (
    "position_rmse_m",
    "velocity_rmse_m_s",
    "attitude_rmse_deg",
    "angular_velocity_rmse_rad_s",
)


def _manifest(**overrides):
    manifest = harness.frozen_live_manifest(MANIFEST)
    for section, values in overrides.items():
        manifest[section] = {**manifest[section], **values}
    return manifest


def _tracking(position, attitude):
    return {
        "position_rmse_m": position,
        "velocity_rmse_m_s": 0.5 * position,
        "attitude_rmse_deg": attitude,
        "angular_velocity_rmse_rad_s": 0.1 * attitude,
    }


def _blocks(passing_index=2, count=7):
    return [
        {
            "index": index,
            "generic": {"velocity_rmse_m_s": 0.1, "body_rate_rmse_rad_s": 0.2},
            "structured": {"velocity_rmse_m_s": 0.15, "body_rate_rmse_rad_s": 0.1},
            "gate": harness.live_swap_gate(
                {"velocity_rmse_m_s": 0.1, "body_rate_rmse_rad_s": 0.2},
                {"velocity_rmse_m_s": 0.15, "body_rate_rmse_rad_s": 0.1},
            )
            if index != passing_index
            else harness.live_swap_gate(
                {"velocity_rmse_m_s": 0.1, "body_rate_rmse_rad_s": 0.05},
                {"velocity_rmse_m_s": 0.15, "body_rate_rmse_rad_s": 0.1},
            ),
            "within_budget": True,
        }
        for index in range(count)
    ]


def _row(
    repetition,
    arm,
    *,
    before=(1.0, 1.0),
    after=(1.0, 1.0),
    whole=(1.0, 1.0),
    swapped=True,
    swap_interval=120,
    terminated=False,
    completed=320,
    worker_error=None,
):
    return {
        "repetition": repetition,
        "arm": arm,
        "completed_intervals": completed,
        "requested_intervals": 320,
        "terminated": terminated,
        "failure": None,
        "worker_error": worker_error,
        "swapped": swapped,
        "swap_interval": swap_interval if swapped else None,
        "swap_time_s": None if not swapped else swap_interval * 0.05,
        "swap_revision": "live-0:3" if swapped else None,
        "swap_scored_block": 2 if swapped else None,
        "candidate_revisions": 7,
        "submitted_blocks": 7,
        "dropped_blocks": 0,
        "deadline_misses": 0,
        "solve_deadline_misses": 0,
        "budget_overruns": 0,
        "maximum_refit_wall_seconds": 1.2,
        "blocks": _blocks(),
        "segments": {
            "before": None if before is None else _tracking(*before),
            "after": None if after is None else _tracking(*after),
            "whole": None if whole is None else _tracking(*whole),
        },
        "pass_criterion": {"met": True},
    }


def _rows(overrides=None):
    """Four trial rows, with per-(repetition, arm) overrides applied by key."""
    overrides = overrides or {}
    return [
        _row(repetition, arm, **overrides.get((repetition, arm), {}))
        for repetition in range(2)
        for arm in harness.LIVE_ARMS
    ]


def _reference():
    return {
        "swapped": {"0": True, "1": True},
        "tracking_rmse": {
            str(repetition): {
                "adopting": {
                    "after": {"position_rmse_m": 1.0, "attitude_rmse_deg": 1.0}
                }
            }
            for repetition in range(2)
        },
    }


# --- the digest gate --------------------------------------------------------


def test_the_live_manifest_digest_is_the_gate(tmp_path):
    manifest = harness.frozen_live_manifest(MANIFEST)
    assert manifest["id"] == "live-v1"
    assert harness.sha256(MANIFEST) == harness.LIVE_MANIFEST_SHA256
    altered = tmp_path / "live.json"
    edited = copy.deepcopy(manifest)
    edited["decision"]["enforced"] = True
    altered.write_text(json.dumps(edited, indent=2) + "\n")
    with pytest.raises(ValueError, match="digest"):
        harness.frozen_live_manifest(altered)


def test_the_committed_live_manifest_is_the_one_in_the_checkout():
    assert harness.COMMITTED_LIVE_MANIFEST == MANIFEST
    assert (
        harness.frozen_live_manifest(harness.COMMITTED_LIVE_MANIFEST)["live"][
            "transport"
        ]["block_steps"]
        == 40
    )


def test_the_live_manifest_declares_a_plan_the_recipe_can_be_cut_to():
    manifest = harness.frozen_live_manifest(MANIFEST)
    steps = harness.steps_for(manifest["information_budget"]["sample_interval_s"])
    transport = manifest["live"]["transport"]
    # Three whole windows is what the recipe demands of any new recording.
    windows = transport["block_steps"] + 1 - steps["history"] - steps["horizon"]
    assert windows >= 3
    assert (
        transport["first_block_start_interval"]
        + transport["block_steps"] * transport["blocks_per_trial"]
        <= manifest["trial"]["intervals"]
    )
    assert manifest["decision"]["enforced"] is False


# --- the swap gate ----------------------------------------------------------


def test_the_swap_gate_passes_only_when_both_metrics_are_at_or_below():
    comparator = {"velocity_rmse_m_s": 0.20, "body_rate_rmse_rad_s": 0.10}
    assert harness.live_swap_gate(
        {"velocity_rmse_m_s": 0.19, "body_rate_rmse_rad_s": 0.09}, comparator
    )["passed"]
    # Equality is at or below, so it passes.
    assert harness.live_swap_gate(dict(comparator), comparator)["passed"]
    for worse in ({"velocity_rmse_m_s": 0.21}, {"body_rate_rmse_rad_s": 0.11}):
        candidate = {**comparator, **worse}
        gate = harness.live_swap_gate(candidate, comparator)
        assert not gate["passed"]
        assert sum(entry["met"] is False for entry in gate["metrics"].values()) == 1


def test_the_swap_gate_reports_the_margin_on_every_metric():
    gate = harness.live_swap_gate(
        {"velocity_rmse_m_s": 0.10, "body_rate_rmse_rad_s": 0.30},
        {"velocity_rmse_m_s": 0.15, "body_rate_rmse_rad_s": 0.10},
    )
    assert gate["metrics"]["velocity_rmse_m_s"]["margin"] == pytest.approx(0.05)
    assert gate["metrics"]["body_rate_rmse_rad_s"]["margin"] == pytest.approx(-0.20)
    assert not gate["passed"]


@pytest.mark.parametrize(
    "candidate",
    [
        {"velocity_rmse_m_s": float("nan"), "body_rate_rmse_rad_s": 0.01},
        {"velocity_rmse_m_s": float("inf"), "body_rate_rmse_rad_s": 0.01},
        {"velocity_rmse_m_s": None, "body_rate_rmse_rad_s": 0.01},
        {"body_rate_rmse_rad_s": 0.01},
        {"velocity_rmse_m_s": "0.01", "body_rate_rmse_rad_s": 0.01},
        {"velocity_rmse_m_s": True, "body_rate_rmse_rad_s": 0.01},
        None,
    ],
)
def test_an_unreadable_candidate_score_never_swaps(candidate):
    gate = harness.live_swap_gate(
        candidate, {"velocity_rmse_m_s": 0.5, "body_rate_rmse_rad_s": 0.5}
    )
    assert not gate["passed"]
    assert gate["metrics"]["velocity_rmse_m_s"]["met"] is None


def test_an_unreadable_comparator_score_never_swaps():
    gate = harness.live_swap_gate(
        {"velocity_rmse_m_s": 0.01, "body_rate_rmse_rad_s": 0.01},
        {"velocity_rmse_m_s": float("nan"), "body_rate_rmse_rad_s": 0.5},
    )
    assert not gate["passed"]


def test_the_first_passing_block_inside_its_budget_is_the_one_offered():
    blocks = _blocks(passing_index=3)
    assert harness.live_first_gate_block(blocks) == 3
    blocks[3]["within_budget"] = False
    assert harness.live_first_gate_block(blocks) is None
    assert harness.live_first_gate_block(_blocks(passing_index=99)) is None
    assert harness.live_first_gate_block([]) is None
    assert harness.live_first_gate_block(None) is None


# --- the before/after segmentation -----------------------------------------


def _trace(rows, seed=0):
    generator = np.random.default_rng(seed)
    states = np.zeros((rows, 13))
    states[:, 6] = 1.0
    reference = states.copy()
    states[:, 0:3] = generator.normal(scale=0.3, size=(rows, 3))
    return states, reference


def test_the_segments_partition_the_trial_at_the_swap_interval():
    states, reference = _trace(101)
    segments = harness.live_segments(states, reference, 40)
    whole = harness.live_segments(states, reference, None)["whole"]
    assert segments["whole"] == whole
    # Forty intervals before, sixty after, and the pooled square error of the
    # two segments is the whole trial's, which is what partitioning means.
    pooled = math.sqrt(
        (
            40 * segments["before"]["position_rmse_m"] ** 2
            + 60 * segments["after"]["position_rmse_m"] ** 2
        )
        / 100
    )
    assert pooled == pytest.approx(whole["position_rmse_m"])


def test_a_swap_at_the_trial_edges_leaves_one_segment_empty():
    states, reference = _trace(21)
    assert harness.live_segments(states, reference, 0)["before"] is None
    assert harness.live_segments(states, reference, 20)["after"] is None
    assert harness.live_segments(states, reference, None)["before"] is None
    assert harness.live_segments(states, reference, None)["after"] is None


def test_the_segmentation_scores_the_state_each_interval_ends_on():
    from glassbox.core.metrics import state_rmse_metrics

    states, reference = _trace(31, seed=3)
    segments = harness.live_segments(states, reference, 12)
    assert segments["before"] == state_rmse_metrics(states[1:13], reference[1:13])
    assert segments["after"] == state_rmse_metrics(states[13:31], reference[13:31])


def test_a_swap_outside_the_completed_trial_is_refused():
    states, reference = _trace(11)
    with pytest.raises(ValueError, match="outside the completed trial"):
        harness.live_segments(states, reference, 11)
    with pytest.raises(ValueError, match="outside the completed trial"):
        harness.live_segments(states, reference, -1)


# --- the decision -----------------------------------------------------------


def test_a_clean_run_is_accepted_and_meets_the_rule():
    decision = harness.live_decide(_manifest(), _rows())
    assert decision["accepted"] and decision["rule_met"]
    assert decision["decision"] == "accept"
    assert decision["rule_enforced"] is False
    assert decision["reference_compared"] is False
    assert set(decision["tracking_rmse"]) == {"0", "1"}
    entry = decision["tracking_rmse"]["0"]["position_rmse_m"]
    assert entry["adopting_before"] == 1.0 and entry["frozen_after"] == 1.0


def _no_swap_rows(repetition):
    return _rows(
        {
            (repetition, "adopting"): dict(
                swapped=False, before=None, after=None, swap_interval=None
            ),
            (repetition, "frozen"): dict(before=None, after=None),
        }
    )


@pytest.mark.parametrize("enforced", [False, True])
def test_a_trial_that_never_swapped_is_reported_where_nothing_gates(enforced):
    # No reference means no case the reference already meets, so the rule is
    # reported everywhere and gates nowhere -- on this tier as on the others.
    decision = harness.live_decide(
        _manifest(decision=dict(enforced=enforced)), _no_swap_rows(1)
    )
    assert decision["accepted"] and not decision["rule_met"]
    assert decision["gating_rule_breaches"] == 0
    assert [
        b["trial"] for b in decision["rule_breaches"] if b["gate"] == "swap_occurred"
    ] == ["1-adopting"]


@pytest.mark.parametrize("enforced", [False, True])
def test_a_trial_that_never_swapped_is_rejected_against_a_reference_that_swapped(
    enforced,
):
    decision = harness.live_decide(
        _manifest(decision=dict(enforced=enforced)), _no_swap_rows(0), _reference()
    )
    assert not decision["rule_met"] and not decision["accepted"]
    gates = [b for b in decision["rule_breaches"] if b["gate"] == "swap_occurred"]
    assert [b["trial"] for b in gates] == ["0-adopting"]
    assert gates[0]["gating"] is True
    # A trial with no after-segment also has no number for the regression gate.
    assert {r["trial"] for r in decision["reference_regressions"]} == {"0-adopting"}


@pytest.mark.parametrize("enforced", [False, True])
def test_tracking_that_regresses_after_the_swap_breaches_the_rule(enforced):
    # The reference's own after-segment sits just inside its own before-limit,
    # so the rule gates here; this run's after-segment breaks its own limit
    # while staying inside the reference's, which isolates the rule from the
    # regression gate.
    reference = _reference()
    for repetition in ("0", "1"):
        reference["tracking_rmse"][repetition]["adopting"]["after"][
            "position_rmse_m"
        ] = 1.05
    rows = _rows({(0, "adopting"): dict(before=(1.0, 1.0), after=(1.08, 1.0))})
    decision = harness.live_decide(
        _manifest(decision=dict(enforced=enforced)), rows, reference
    )
    breach = [
        b for b in decision["rule_breaches"] if b["gate"] == "tracking_after_swap"
    ]
    assert [(b["trial"], b["metric"]) for b in breach] == [
        ("0-adopting", "position_rmse_m")
    ]
    assert breach[0]["limit"] == pytest.approx(1.0 * 1.05 + 0.005)
    assert breach[0]["gating"] is True
    assert not decision["rule_met"]
    assert decision["reference_regressions"] == []
    assert decision["accepted"] is not enforced


def test_tracking_inside_the_declared_allowance_after_the_swap_holds_the_rule():
    rows = _rows({(0, "adopting"): dict(before=(1.0, 1.0), after=(1.05, 1.0))})
    decision = harness.live_decide(_manifest(), rows)
    assert decision["rule_met"] and decision["accepted"]


@pytest.mark.parametrize("enforced", [False, True])
def test_a_terminated_trial_always_fails_closed(enforced):
    rows = _rows({(1, "adopting"): dict(terminated=True, completed=200)})
    decision = harness.live_decide(_manifest(decision=dict(enforced=enforced)), rows)
    assert not decision["accepted"] and not decision["rule_met"]
    assert [b["gate"] for b in decision["gate_breaches"]] == ["trial_complete"]


@pytest.mark.parametrize("enforced", [False, True])
def test_a_short_trial_always_fails_closed(enforced):
    rows = _rows({(0, "frozen"): dict(completed=319)})
    decision = harness.live_decide(_manifest(decision=dict(enforced=enforced)), rows)
    assert not decision["accepted"]
    assert [b["gate"] for b in decision["gate_breaches"]] == ["declared_intervals"]


@pytest.mark.parametrize("enforced", [False, True])
def test_a_worker_error_always_fails_closed(enforced):
    rows = _rows({(1, "frozen"): dict(worker_error="ValueError: telemetry")})
    decision = harness.live_decide(_manifest(decision=dict(enforced=enforced)), rows)
    assert not decision["accepted"]
    assert [b["gate"] for b in decision["gate_breaches"]] == ["worker_error"]


@pytest.mark.parametrize("enforced", [False, True])
def test_a_missing_trial_always_fails_closed(enforced):
    rows = [row for row in _rows() if (row["repetition"], row["arm"]) != (1, "frozen")]
    decision = harness.live_decide(_manifest(decision=dict(enforced=enforced)), rows)
    assert not decision["accepted"]
    assert decision["gate_breaches"] == [{"trial": "1-frozen", "gate": "trial_present"}]


@pytest.mark.parametrize("enforced", [False, True])
def test_a_duplicated_or_undeclared_trial_always_fails_closed(enforced):
    rows = _rows() + [_row(0, "adopting")]
    decision = harness.live_decide(_manifest(decision=dict(enforced=enforced)), rows)
    assert not decision["accepted"]
    assert [b["gate"] for b in decision["gate_breaches"]] == ["trial_unique"]
    rows = _rows() + [_row(0, "shadow")]
    decision = harness.live_decide(_manifest(decision=dict(enforced=enforced)), rows)
    assert not decision["accepted"]
    assert [b["gate"] for b in decision["gate_breaches"]] == ["trial_declared"]


@pytest.mark.parametrize("enforced", [False, True])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), None, "1.0"])
def test_a_nonfinite_whole_trial_metric_always_fails_closed(enforced, value):
    rows = _rows()
    for row in rows:
        if (row["repetition"], row["arm"]) == (0, "adopting"):
            row["segments"]["whole"]["position_rmse_m"] = value
    decision = harness.live_decide(_manifest(decision=dict(enforced=enforced)), rows)
    assert not decision["accepted"]
    assert [b["gate"] for b in decision["gate_breaches"]] == ["finite_rmse"]


def test_a_negative_tracking_metric_is_not_a_number_this_gate_accepts():
    rows = _rows()
    for row in rows:
        if (row["repetition"], row["arm"]) == (1, "adopting"):
            row["segments"]["whole"]["attitude_rmse_deg"] = -1.0
    decision = harness.live_decide(_manifest(), rows)
    assert not decision["accepted"]
    assert [b["gate"] for b in decision["gate_breaches"]] == ["finite_rmse"]


def test_a_swap_with_no_after_segment_breaches_the_rule():
    rows = _rows({(0, "adopting"): dict(after=None)})
    decision = harness.live_decide(_manifest(), rows)
    assert not decision["rule_met"]
    assert {b["gate"] for b in decision["rule_breaches"]} == {"segments_present"}


def test_a_missing_reference_row_is_a_regression_when_a_reference_is_compared():
    reference = _reference()
    del reference["tracking_rmse"]["1"]
    decision = harness.live_decide(_manifest(), _rows(), reference, "abc")
    assert not decision["accepted"]
    assert decision["reference_sha256"] == "abc"
    assert {r["gate"] for r in decision["reference_regressions"]} == {
        "reference_present"
    }


def test_a_reference_regression_rejects_whatever_the_rule_says():
    reference = _reference()
    reference["tracking_rmse"]["0"]["adopting"]["after"]["position_rmse_m"] = 0.5
    decision = harness.live_decide(_manifest(), _rows(), reference, "abc")
    assert decision["rule_met"] and not decision["accepted"]
    assert [r["gate"] for r in decision["reference_regressions"]] == [
        "reference_tracking_rmse"
    ]


def test_the_rule_gates_nothing_a_reference_does_not_already_meet():
    reference = _reference()
    # The incumbent's own after-segment is worse than its before-segment, so
    # this is a case the reference already fails: it is reported, not gated.
    reference["tracking_rmse"]["0"]["adopting"]["after"]["position_rmse_m"] = 9.0
    rows = _rows({(0, "adopting"): dict(before=(1.0, 1.0), after=(1.5, 1.0))})
    decision = harness.live_decide(
        _manifest(decision=dict(enforced=True)), rows, reference, "abc"
    )
    breach = [
        b for b in decision["rule_breaches"] if b["gate"] == "tracking_after_swap"
    ]
    assert breach and breach[0]["gating"] is False
    assert decision["gating_rule_breaches"] == 0
