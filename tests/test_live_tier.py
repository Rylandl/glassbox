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

import jax
import numpy as np
import pytest

from glassbox.experimental import harness

MANIFEST = Path(__file__).resolve().parents[1] / "docs/harness/live-v2.json"
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
        "swap_scored_stop_interval": 140 if swapped else None,
        "swap_release_interval": swap_interval if swapped else None,
        "offer_deferred_past_trial": False,
        "candidate_revisions": 7,
        "submitted_blocks": 7,
        "dropped_blocks": 0,
        "budget_overruns": 0,
        "wall": {"intervals_over_sample_interval": 0, "solves_over_deadline": 0},
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
    assert manifest["id"] == "live-v2"
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


def test_the_live_manifest_declares_what_makes_the_trajectory_deterministic():
    manifest = harness.frozen_live_manifest(MANIFEST)
    assert manifest["trial"]["solver_deadline_applied"] is False
    assert manifest["live"]["transport"]["drive"] == "synchronous"
    assert manifest["live"]["transport"]["offer_release_offset_intervals"] == 40
    assert "maximum_offer_age_intervals" not in manifest["live"]["transport"]
    assert manifest["live"]["determinism"]["claim"]


@pytest.mark.parametrize(
    "edit",
    [
        lambda m: m["live"]["transport"].__setitem__("drive", "threaded"),
        lambda m: m["trial"].__setitem__("solver_deadline_applied", True),
        lambda m: m["live"]["transport"].__setitem__(
            "offer_release_offset_intervals", 0
        ),
        lambda m: m["live"]["transport"].__setitem__(
            "offer_release_offset_intervals", True
        ),
    ],
)
def test_a_manifest_that_lets_a_clock_in_is_refused(tmp_path, edit, monkeypatch):
    manifest = copy.deepcopy(harness.frozen_live_manifest(MANIFEST))
    edit(manifest)
    altered = tmp_path / "live.json"
    altered.write_text(json.dumps(manifest, indent=2) + "\n")
    # The digest gate fires first, so check the structural refusal on its own.
    monkeypatch.setattr(
        harness, "LIVE_MANIFEST_SHA256", harness.sha256(altered), raising=True
    )
    with pytest.raises(ValueError):
        harness.frozen_live_manifest(altered)


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


def test_the_first_passing_block_is_the_one_offered():
    blocks = _blocks(passing_index=3)
    assert harness.live_first_gate_block(blocks) == 3
    # A refit that ran over its wall budget is reported and still offered: the
    # host's scheduling must not decide where the aircraft flew.
    blocks[3]["within_budget"] = False
    assert harness.live_first_gate_block(blocks) == 3
    assert harness.live_first_gate_block(_blocks(passing_index=99)) is None
    assert harness.live_first_gate_block([]) is None
    assert harness.live_first_gate_block(None) is None


def test_the_swap_interval_is_the_gate_block_plus_the_declared_offset():
    blocks = [
        dict(entry, start_interval=20 + 40 * i, stop_interval=60 + 40 * i)
        for i, entry in enumerate(_blocks(passing_index=2))
    ]
    assert harness.live_expected_swap(blocks, 40, 320) == (2, 180)
    assert harness.live_expected_swap(blocks, 10, 320) == (2, 150)
    # A gate that passes too late to be released inside the trial never swaps.
    assert harness.live_expected_swap(blocks, 40, 180) == (2, None)
    assert harness.live_expected_swap(_blocks(passing_index=99), 40, 320) == (
        None,
        None,
    )


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


def test_a_refit_over_its_budget_is_reported_and_gates_nothing():
    rows = _rows()
    for row in rows:
        if (row["repetition"], row["arm"]) == (0, "adopting"):
            row["blocks"][0]["within_budget"] = False
            row["blocks"][0]["refit_wall_seconds"] = 9.0
            row["blocks"][0]["wall_budget_seconds"] = 4.0
            row["budget_overruns"] = 1
    decision = harness.live_decide(_manifest(decision=dict(enforced=True)), rows)
    assert decision["accepted"] and decision["rule_met"]
    assert [b["gate"] for b in decision["budget_breaches"]] == ["refit_wall_budget"]
    assert decision["budget_breaches"][0]["trial"] == "0-adopting"
    assert decision["gating_rule_breaches"] == 0


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


# --- the block rows, the refiner and the replay -----------------------------

DT_S = 0.05
MINIMUM = np.array([0.0, -0.35, -0.35])
MAXIMUM = np.array([1.0, 0.35, 0.35])
LEVEL = np.array([1.0, -2.0, 100.0, 18.0, 0.3, -0.2, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def _recording(seed, rows=140):
    """One synthetic recording whose rotation entries are real rotations."""
    from glassbox.core.geometry import quaternion_from_euler

    generator = np.random.default_rng(seed)
    commands = generator.uniform(MINIMUM, MAXIMUM, size=(rows - 1, 3))
    commands[::2] += np.array([0.25, 0.2, 0.2])
    commands = np.clip(commands, MINIMUM, MAXIMUM)
    states = np.zeros((rows, 13))
    states[0] = LEVEL
    angles = np.zeros(3)
    for index, command in enumerate(commands):
        angles = 0.98 * angles + 0.05 * np.r_[command[1], command[2], 0.01 * index]
        states[index + 1, 0:3] = states[index, 0:3] + DT_S * states[index, 3:6]
        states[index + 1, 3:6] = (
            0.97 * states[index, 3:6]
            + np.r_[0.4 * command[0], 0.1 * angles[0], 0.1 * angles[1]]
        )
        states[index + 1, 6:10] = quaternion_from_euler(*angles)
        states[index + 1, 10:13] = 0.9 * states[index, 10:13] + 0.2 * angles
    return states, commands


def _trajectory(seed, rows=140):
    """One fabricated calibration recording on the tier's declared contract."""
    from glassbox.core.data import Trajectory

    states, commands = _recording(seed, rows)
    return Trajectory(
        time_s=np.arange(len(states)) * DT_S,
        states=states,
        controls=commands,
        control_prefix=np.repeat(commands[:1], 20, axis=0),
        spec=harness.control_telemetry_spec(harness.frozen_live_manifest(MANIFEST)),
        labels={"source_group": f"cascade-calibration-{seed}"},
    )


@pytest.fixture(scope="module")
def learned():
    """One really fitted generic learner on the contract this tier requires."""
    from glassbox.experimental.default_model import fit

    manifest = harness.frozen_live_manifest(MANIFEST)
    with jax.enable_x64(True):
        return fit(
            harness.control_collection(
                manifest, [(f"recording-{seed}", _trajectory(seed)) for seed in (0, 1)]
            )
        )


@pytest.fixture(scope="module")
def belief():
    """One really fitted structured belief, the comparator every gate reads."""
    from glassbox.fitting import FitSpec, Holdout
    from glassbox.fitting import fit as structured_fit

    with jax.enable_x64(True):
        return structured_fit(
            [_trajectory(seed) for seed in (0, 1)],
            FitSpec(
                holdout=Holdout.by_group(),
                steps=2,
                horizons_s=(0.1, 0.4),
                evaluation_horizons_s=(0.1, 0.4),
            ),
        ).belief


def _block(seed, start=20, block_steps=40):
    """One streamed block, shaped the way the transition buffer emits them."""
    from glassbox.core.data import Trajectory

    states, commands = _recording(seed)
    return Trajectory(
        time_s=np.arange(block_steps + 1) * DT_S,
        states=states[start : start + block_steps + 1],
        controls=commands[start : start + block_steps],
        control_prefix=commands[start - 20 : start],
        spec=harness.control_telemetry_spec(harness.frozen_live_manifest(MANIFEST)),
        labels={"source_group": f"block-{seed}-{start}"},
    )


def _rows_for(manifest, block, name="block"):
    steps = harness.steps_for(manifest["plant"]["sample_interval_s"])
    return harness.live_block_rows(
        manifest, block, harness.control_collection(manifest, [(name, block)]), steps
    )


def test_a_block_carries_the_same_origins_for_both_models():
    manifest = harness.frozen_live_manifest(MANIFEST)
    steps = harness.steps_for(manifest["plant"]["sample_interval_s"])
    block = _block(7)
    arrays = _rows_for(manifest, block)
    context, horizon = steps["history"], steps["horizon"]
    expected = 40 + 1 - context - horizon
    assert len(arrays["targets"]) == expected >= 3
    assert arrays["past_states"].shape[1] == context + 1
    assert arrays["future_inputs"].shape[1] == horizon
    assert arrays["control_histories"].shape[1] == 20
    # The structured rollout starts at the same origin the generic context ends
    # at, in the two representations of one state.
    np.testing.assert_allclose(
        harness.observed_from_states(arrays["initial_states"]),
        arrays["past_states"][:, -1],
        rtol=0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        arrays["controls"], arrays["future_inputs"], rtol=0, atol=0
    )


def test_a_block_too_short_for_one_whole_window_is_refused():
    manifest = harness.frozen_live_manifest(MANIFEST)
    with pytest.raises(ValueError, match="no origin with a complete context"):
        _rows_for(manifest, _block(7, block_steps=14))


def test_both_models_are_scored_on_the_same_block_rows(learned, belief):
    manifest = harness.frozen_live_manifest(MANIFEST)
    arrays = _rows_for(manifest, _block(7))
    generic, structured, generic_score, structured_score = harness.live_block_scores(
        manifest, learned, belief, arrays
    )
    assert generic.shape == structured.shape == arrays["targets"].shape
    assert sorted(generic_score) == sorted(harness.METRICS)
    gate = harness.live_swap_gate(generic_score, structured_score)
    assert gate["passed"] in (True, False)
    # The recorded score is a function of the saved arrays alone, which is what
    # lets a replay recompute it without rerunning anything.
    np.testing.assert_allclose(
        harness.platform_measure(generic, arrays["targets"])["final_step"][
            "velocity_rmse_m_s"
        ],
        generic_score["velocity_rmse_m_s"],
        rtol=0,
        atol=0,
    )


def _refiner(manifest, learned, belief, session="t"):
    return harness._LiveRefiner(
        manifest, learned, belief, recording_id=session, session=session
    )


def test_the_refiner_scores_a_block_before_it_fits_on_it(learned, belief):
    manifest = harness.frozen_live_manifest(MANIFEST)
    refiner = _refiner(manifest, learned, belief)
    assert refiner.active.revision_id == "t:structured"
    assert refiner.candidate.revision_id == "t:0"
    result = refiner.observe(_block(7), recording_id="t", start_interval=0)
    record = result.record
    # The revision offered is the one the block was held out from, not the one
    # that has just learned from it.
    assert result.candidate_score.revision.revision_id == "t:0"
    assert record["scored_revision"] == "t:0"
    assert record["revision"] == "t:1" == refiner.candidate.revision_id
    assert record["rows"] == 26
    assert record["fit_steps"] == 1000
    assert refiner.candidate.learned.fingerprint() != learned.fingerprint()
    assert record["scored_fingerprint"] == learned.fingerprint()
    assert json.dumps(result.to_dict(), allow_nan=False)


def test_the_refiner_requires_blocks_in_order_on_one_recording(learned, belief):
    manifest = harness.frozen_live_manifest(MANIFEST)
    refiner = _refiner(manifest, learned, belief)
    with pytest.raises(ValueError, match="in order"):
        refiner.observe(_block(7), recording_id="t", start_interval=40)
    with pytest.raises(ValueError, match="in order"):
        refiner.observe(_block(7), recording_id="other", start_interval=0)
    refiner.skip(recording_id="t", start_interval=0, stop_interval=20, reason="history")
    assert refiner.skipped_interval_count == 20
    with pytest.raises(ValueError, match="advance this recording's cursor"):
        refiner.skip(recording_id="t", start_interval=0, stop_interval=5, reason="x")


def test_only_a_scored_revision_can_be_held_and_adopted(learned, belief):
    manifest = harness.frozen_live_manifest(MANIFEST)
    refiner = _refiner(manifest, learned, belief)
    with pytest.raises(ValueError, match="evaluated revision"):
        refiner.hold_for_adoption("t:0")
    refiner.skip(recording_id="t", start_interval=0, stop_interval=20, reason="history")
    refiner.observe(_block(7), recording_id="t", start_interval=20)
    refiner.hold_for_adoption("t:0")
    with pytest.raises(ValueError, match="already outstanding"):
        refiner.hold_for_adoption("t:0")
    with pytest.raises(ValueError, match="active revision changed"):
        refiner.adopt("t:0", expected_active_revision="t:9", reason="r")
    with pytest.raises(ValueError, match="scored on a subsequent block"):
        refiner.adopt("t:1", expected_active_revision="t:structured", reason="r")
    adoption = refiner.adopt("t:0", expected_active_revision="t:structured", reason="r")
    assert refiner.active.revision_id == "t:0"
    assert json.dumps(adoption.to_dict(), allow_nan=False)
    refiner.release_adoption_hold()


# --- the replay -------------------------------------------------------------


def _anchor(directory):
    return Path(directory).parent / "live-reference.json"


def _fabricate_run(directory, manifest, learned, belief, offsets):
    """A live run directory that was never flown, for the replay to check."""
    import shutil

    from glassbox.core.data import save_trajectory_npz, trajectory_content_digest
    from glassbox.core.metrics import state_rmse_metrics

    directory.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(MANIFEST, directory / "manifest.json")
    names, digests = [], {}
    for seed in manifest["calibration"]["seeds"]:
        flight = _trajectory(seed)
        save_trajectory_npz(flight, directory / f"recording-{seed}.npz")
        names.append(f"recording-{seed}.npz")
        digests[f"recording-{seed}"] = trajectory_content_digest(flight)
    learned.save(directory / "generic.npz")
    belief.save(directory / "structured.json")
    (directory / "structured_report.json").write_text('{"report": "fabricated"}\n')

    reserved = [f"recording-{s}" for s in manifest["calibration"]["reserved_seeds"]]
    evidence_arrays = harness.control_evidence_arrays(
        manifest, [(name, _trajectory(int(name.split("-")[1]))) for name in reserved]
    )
    with jax.enable_x64(True):
        prediction, half_width, coverage = harness.control_evidence(
            learned, evidence_arrays
        )
    np.savez_compressed(
        directory / "evidence.npz",
        **evidence_arrays,
        prediction=prediction,
        envelope_half_width=half_width,
    )
    names.append("evidence.npz")

    anchor_state = LEVEL.copy()
    declared = manifest["tracking_reference"]
    intervals = manifest["trial"]["intervals"]
    times = np.arange(intervals + 1) * DT_S
    reference = harness.control_reference(anchor_state, times, declared)
    blocks_by_arm = {}
    for arm in harness.LIVE_ARMS:
        records, files = [], []
        revision = learned
        for index in range(2):
            block = _block(30 + index, start=20 + 40 * index)
            arrays = _rows_for(manifest, block, name=f"{arm}-{index}")
            generic, structured, generic_score, structured_score = (
                harness.live_block_scores(manifest, revision, belief, arrays)
            )
            np.savez_compressed(
                directory / f"{arm}-block-{index:03d}.npz",
                **arrays,
                generic_prediction=generic,
                structured_prediction=structured,
            )
            revision.save(directory / f"{arm}-block-{index:03d}-scored.npz")
            records.append(
                dict(
                    index=index,
                    recording_id=f"{arm}-block-{index:03d}",
                    start_interval=20 + 40 * index,
                    stop_interval=60 + 40 * index,
                    rows=len(arrays["targets"]),
                    scored_revision=f"live-{arm}:{index}",
                    scored_fingerprint=revision.fingerprint(),
                    revision=f"live-{arm}:{index + 1}",
                    generic=generic_score,
                    structured=structured_score,
                    gate=harness.live_swap_gate(generic_score, structured_score),
                    score_wall_seconds=0.5,
                    refit_wall_seconds=1.0,
                    fit_steps=1000,
                    wall_budget_seconds=4.0,
                    within_budget=True,
                    training_recordings=2,
                    training_windows=300,
                    arrays=f"{arm}-block-{index:03d}.npz",
                    scored_model=f"{arm}-block-{index:03d}-scored.npz",
                )
            )
            files.extend([records[-1]["arrays"], records[-1]["scored_model"]])
        blocks_by_arm[arm] = (records, files)

    offset = manifest["live"]["transport"]["offer_release_offset_intervals"]
    first, swap_interval = harness.live_expected_swap(
        blocks_by_arm["adopting"][0], offset, manifest["trial"]["intervals"]
    )
    rows = []
    for repetition in range(manifest["trial"]["repetitions"]):
        seed = manifest["trial"]["initial_state_seeds"][repetition]
        initial_state = harness.control_initial_state(manifest, anchor_state, seed)
        for arm in harness.LIVE_ARMS:
            case = directory / f"trial-{repetition}" / arm
            case.mkdir(parents=True, exist_ok=True)
            states = reference.copy()
            states[:, 0] += offsets[arm]
            states[0] = initial_state
            swapped = arm == "adopting" and swap_interval is not None
            actives = np.full(intervals, f"live-{repetition}-{arm}:structured")
            if swapped:
                actives[swap_interval:] = f"live-{arm}:{first}"
            np.savez_compressed(
                case / "tracking.npz",
                time_s=times,
                states=states,
                reference_states=reference,
                commands=np.zeros((intervals, 3)),
                solver_used=np.ones(intervals, dtype=bool),
                used_fallback=np.zeros(intervals, dtype=bool),
                active_revisions=actives,
                initial_state=initial_state,
                reference_anchor_state=anchor_state,
            )
            np.savez_compressed(
                case / "timing.npz",
                tick_times_s=np.full(intervals, 0.01),
                solve_times_s=np.full(intervals, 0.002),
                telemetry_times_s=np.zeros(intervals),
            )
            records, files = blocks_by_arm[arm]
            for name in files:
                shutil.copyfile(directory / name, case / name)
            (case / "events.jsonl").write_text("")
            row = dict(
                arm=arm,
                session=f"live-{repetition}-{arm}",
                repetition=repetition,
                initial_state_seed=seed,
                completed_intervals=intervals,
                requested_intervals=intervals,
                terminated=False,
                failure=None,
                worker_error=None,
                swapped=swapped,
                swap_interval=swap_interval if swapped else None,
                swap_time_s=swap_interval * DT_S if swapped else None,
                swap_revision=(records[first]["scored_revision"] if swapped else None),
                swap_scored_block=first if swapped else None,
                swap_scored_stop_interval=(
                    records[first]["stop_interval"] if swapped else None
                ),
                swap_release_interval=swap_interval if swapped else None,
                offer_deferred_past_trial=False,
                offers=int(swapped),
                rejected_offers=0,
                blocks=records,
                candidate_revisions=len(records),
                submitted_blocks=len(records),
                dropped_blocks=0,
                budget_overruns=0,
                wall={
                    "intervals_over_sample_interval": 0,
                    "solves_over_deadline": 0,
                    "maximum_refit_seconds": 1.0,
                },
                tracking_rmse=state_rmse_metrics(states[1:], reference[1:]),
                pass_criterion=harness.control_pass_criterion(
                    states, anchor_state, manifest
                ),
                segment_swap_interval=swap_interval,
                segments=harness.live_segments(states, reference, swap_interval),
                directory=f"trial-{repetition}/{arm}",
                files=harness._files(
                    case, ["tracking.npz", "timing.npz", "events.jsonl", *files]
                ),
            )
            harness.write(case / "trial.json", row)
            rows.append(row)
    harness.write(directory / "results.json", rows)
    harness.write(
        directory / "calibration.json",
        dict(
            recordings=digests,
            training=[
                f"recording-{s}" for s in manifest["calibration"]["training_seeds"]
            ],
            reserved=reserved,
            evidence=coverage,
            generic_fingerprint=learned.fingerprint(),
            command_excitation=harness.control_excitation(
                manifest,
                [
                    (f"recording-{seed}", _trajectory(seed))
                    for seed in manifest["calibration"]["seeds"]
                ],
            ),
            files=harness._files(
                directory,
                names + ["structured.json", "structured_report.json", "generic.npz"],
            ),
        ),
    )
    harness.write(
        directory / "decision.json", harness.live_decide(manifest, rows, None, None)
    )
    return rows


@pytest.fixture(scope="module")
def flown(tmp_path_factory, learned, belief):
    """One fabricated run directory, built once for every replay test."""
    manifest = harness.frozen_live_manifest(MANIFEST)
    directory = tmp_path_factory.mktemp("live") / "live-run"
    _fabricate_run(directory, manifest, learned, belief, dict(adopting=0.4, frozen=0.5))
    return directory


@pytest.fixture
def run(flown, tmp_path):
    """A private copy of it, so a test may alter what it is checking."""
    import shutil

    directory = tmp_path / "live-run"
    shutil.copytree(flown, directory)
    return directory


def test_the_replay_recomputes_every_metric_and_the_decision(run):
    manifest = harness.frozen_live_manifest(MANIFEST)
    result = harness.verify_live(run, manifest)
    assert result["tier"] == "live"
    assert result["verified_trials"] == 4
    assert result["replays"] == 16
    assert result["maximum_replay_difference"] < 1e-6
    assert "not rerun" in result["meaning"]
    assert result["decision"]["reference_compared"] is False
    # The tier is chosen by the digest of the manifest the run copied.
    assert harness.verify(run)["tier"] == "live"


def test_the_replay_rejects_an_altered_tracking_array(run):
    manifest = harness.frozen_live_manifest(MANIFEST)
    case = run / "trial-0" / "adopting"
    with np.load(case / "tracking.npz", allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    arrays["states"] = arrays["reference_states"].copy()
    np.savez_compressed(case / "tracking.npz", **arrays)
    with pytest.raises(ValueError, match="altered artifact"):
        harness.verify_live(run, manifest)


def test_the_replay_rejects_an_altered_block_array(run):
    manifest = harness.frozen_live_manifest(MANIFEST)
    case = run / "trial-0" / "adopting"
    row = harness.read(case / "trial.json")
    name = row["blocks"][0]["arrays"]
    with np.load(case / name, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    arrays["generic_prediction"] = arrays["targets"].copy()
    np.savez_compressed(case / name, **arrays)
    row["files"][name] = harness.sha256(case / name)
    harness.write(case / "trial.json", row)
    rows = harness.read(run / "results.json")
    rows[[r["directory"] for r in rows].index(row["directory"])] = row
    harness.write(run / "results.json", rows)
    with pytest.raises(AssertionError):
        harness.verify_live(run, manifest)


def test_the_replay_rejects_a_forged_block_score(run):
    manifest = harness.frozen_live_manifest(MANIFEST)
    rows = harness.read(run / "results.json")
    row = next(r for r in rows if r["directory"] == "trial-0/adopting")
    row["blocks"][0]["generic"]["velocity_rmse_m_s"] = 0.0
    harness.write(run / row["directory"] / "trial.json", row)
    harness.write(run / "results.json", rows)
    with pytest.raises(AssertionError):
        harness.verify_live(run, manifest)


def test_the_replay_rejects_a_forged_gate_outcome(run):
    manifest = harness.frozen_live_manifest(MANIFEST)
    rows = harness.read(run / "results.json")
    row = next(r for r in rows if r["directory"] == "trial-0/adopting")
    entry = row["blocks"][0]
    entry["gate"]["passed"] = not entry["gate"]["passed"]
    harness.write(run / row["directory"] / "trial.json", row)
    harness.write(run / "results.json", rows)
    with pytest.raises(ValueError, match="recomputed swap gate differs"):
        harness.verify_live(run, manifest)


def test_the_replay_rejects_a_swap_at_an_interval_the_offset_does_not_give(run):
    manifest = harness.frozen_live_manifest(MANIFEST)
    rows = harness.read(run / "results.json")
    row = next(r for r in rows if r["directory"] == "trial-0/adopting")
    if not row["swapped"]:
        pytest.skip("this fabricated run never swapped")
    # The swap interval is the gate block's stop interval plus the declared
    # offset. Move it by one and move the recorded active revisions with it, so
    # the run is internally consistent and still disagrees with the manifest.
    case = run / row["directory"]
    with np.load(case / "tracking.npz", allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    moved = row["swap_interval"] + 1
    arrays["active_revisions"][row["swap_interval"]] = f"{row['session']}:structured"
    np.savez_compressed(case / "tracking.npz", **arrays)
    row["swap_interval"] = moved
    row["files"]["tracking.npz"] = harness.sha256(case / "tracking.npz")
    harness.write(case / "trial.json", row)
    harness.write(run / "results.json", rows)
    with pytest.raises(ValueError, match=r"plus\s+the declared offset"):
        harness.verify_live(run, manifest)


def test_the_replay_rejects_a_trial_that_should_have_swapped_and_did_not(run):
    manifest = harness.frozen_live_manifest(MANIFEST)
    rows = harness.read(run / "results.json")
    row = next(r for r in rows if r["directory"] == "trial-0/adopting")
    if not row["swapped"]:
        pytest.skip("this fabricated run never swapped")
    row.update(swapped=False, swap_interval=None, swap_revision=None)
    harness.write(run / row["directory"] / "trial.json", row)
    harness.write(run / "results.json", rows)
    with pytest.raises(ValueError, match=r"the declared|active revisions"):
        harness.verify_live(run, manifest)


def test_the_replay_rejects_a_dropped_block_under_a_synchronous_drive(run):
    manifest = harness.frozen_live_manifest(MANIFEST)
    rows = harness.read(run / "results.json")
    row = next(r for r in rows if r["directory"] == "trial-0/frozen")
    row["dropped_blocks"] = 1
    harness.write(run / row["directory"] / "trial.json", row)
    harness.write(run / "results.json", rows)
    with pytest.raises(ValueError, match="dropped a block"):
        harness.verify_live(run, manifest)


def test_the_replay_rejects_a_recorded_active_revision_the_swap_denies(run):
    manifest = harness.frozen_live_manifest(MANIFEST)
    case = run / "trial-0" / "frozen"
    with np.load(case / "tracking.npz", allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    arrays["active_revisions"] = np.full(len(arrays["commands"]), "live-0-frozen:9")
    np.savez_compressed(case / "tracking.npz", **arrays)
    row = harness.read(case / "trial.json")
    row["files"]["tracking.npz"] = harness.sha256(case / "tracking.npz")
    harness.write(case / "trial.json", row)
    rows = harness.read(run / "results.json")
    rows[[r["directory"] for r in rows].index(row["directory"])] = row
    harness.write(run / "results.json", rows)
    with pytest.raises(ValueError, match="active revisions"):
        harness.verify_live(run, manifest)


def test_the_replay_rejects_segments_recomputed_from_the_saved_arrays(run):
    manifest = harness.frozen_live_manifest(MANIFEST)
    rows = harness.read(run / "results.json")
    row = next(r for r in rows if r["directory"] == "trial-0/adopting")
    if row["segments"]["after"] is None:
        pytest.skip("this fabricated run never swapped, so it has no after segment")
    row["segments"]["after"]["position_rmse_m"] = 0.0
    harness.write(run / row["directory"] / "trial.json", row)
    harness.write(run / "results.json", rows)
    with pytest.raises(AssertionError):
        harness.verify_live(run, manifest)


def test_the_replay_rejects_a_reference_it_cannot_anchor(run):
    manifest = harness.frozen_live_manifest(MANIFEST)
    harness.write(run / "reference.json", {"tracking_rmse": {}})
    with pytest.raises(ValueError, match="reference mismatch"):
        harness.verify_live(run, manifest, _anchor(run))


# --- the tests that drive Cascade itself ------------------------------------


@pytest.mark.cascade
def test_one_short_live_trial_streams_refits_and_may_swap(tmp_path):
    """One short paced trial end to end: the transport, the refits, the metrics."""
    pytest.importorskip("cascade")
    from glassbox.belief.belief_io import load_dynamics_belief  # noqa: F401
    from glassbox.experimental.default_model import fit
    from glassbox.fitting import FitSpec, Holdout
    from glassbox.fitting import fit as structured_fit

    manifest = copy.deepcopy(harness.frozen_live_manifest(MANIFEST))
    manifest["calibration"]["duration_s"] = 3.0
    manifest["arms"]["structured"]["optimization_steps"] = 3
    manifest["trial"]["intervals"] = 70
    manifest["trial"]["duration_s"] = 3.5
    manifest["live"]["transport"]["block_steps"] = 20
    manifest["live"]["transport"]["block_duration_s"] = 1.0
    manifest["live"]["transport"]["blocks_per_trial"] = 2
    manifest["metrics"]["pass_criterion"]["settled_after_s"] = 0.5

    spec, model, trim, state, command = harness.control_fixture(manifest)
    plant = harness._control_plant(manifest, spec, model)
    recordings = [
        (
            f"recording-{seed}",
            harness.control_recording(manifest, plant, trim, state, command, seed),
        )
        for seed in (0, 1)
    ]
    with jax.enable_x64(True):
        artifacts = dict(
            generic=fit(harness.control_collection(manifest, recordings)),
            structured=structured_fit(
                [flight for _, flight in recordings],
                FitSpec(
                    holdout=Holdout.by_group(),
                    steps=manifest["arms"]["structured"]["optimization_steps"],
                    horizons_s=(0.1, 0.4),
                    evaluation_horizons_s=(0.1, 0.4),
                ),
            ).belief,
        )

    def reference_fn(times):
        return harness.control_reference(state, times, manifest["tracking_reference"])

    start = harness.control_initial_state(manifest, state, 101)
    row = harness._live_trial(
        manifest,
        adopting=True,
        artifacts=artifacts,
        plant=harness._control_tracking_plant(manifest, start, command),
        warmup=recordings[0][1],
        reference_fn=reference_fn,
        anchor_state=state,
        session="live-test",
        directory=tmp_path / "t",
    )
    assert row["terminated"] is False
    assert row["worker_error"] is None
    assert row["completed_intervals"] == 70
    assert row["maximum_command_bound_violation"] == 0.0
    assert row["arm"] == "adopting"
    assert row["dropped_blocks"] == 0
    # The wall measurements exist and decide nothing.
    assert row["wall"]["maximum_refit_seconds"] > 0.0
    assert set(row["wall"]) >= {
        "intervals_over_sample_interval",
        "solves_over_deadline",
        "maximum_refit_seconds",
    }
    # Twenty intervals fill the command history, then two whole blocks.
    assert row["submitted_blocks"] == 2
    assert row["candidate_revisions"] == len(row["blocks"]) >= 1
    for entry in row["blocks"]:
        assert entry["rows"] == 20 + 1 - 10 - 5
        assert sorted(entry["generic"]) == sorted(harness.METRICS)
        assert entry["gate"] == harness.live_swap_gate(
            entry["generic"], entry["structured"]
        )
    assert math.isfinite(row["tracking_rmse"]["position_rmse_m"])
    assert json.dumps(row, allow_nan=False)
    assert (tmp_path / "t" / "block-000.npz").exists()
    assert (tmp_path / "t" / "block-000-scored.npz").exists()
    assert (tmp_path / "t" / "timing.npz").exists()
    # The swap, if there was one, is exactly where the manifest puts it.
    _, expected = harness.live_expected_swap(
        row["blocks"],
        manifest["live"]["transport"]["offer_release_offset_intervals"],
        manifest["trial"]["intervals"],
    )
    assert row["swap_interval"] == expected


@pytest.mark.cascade
def test_a_frozen_arm_runs_the_same_learner_and_never_swaps(tmp_path):
    """The reference arm pays the same compute and is offered nothing."""
    pytest.importorskip("cascade")
    from glassbox.experimental.default_model import fit
    from glassbox.fitting import FitSpec, Holdout
    from glassbox.fitting import fit as structured_fit

    manifest = copy.deepcopy(harness.frozen_live_manifest(MANIFEST))
    manifest["calibration"]["duration_s"] = 3.0
    manifest["arms"]["structured"]["optimization_steps"] = 3
    manifest["trial"]["intervals"] = 45
    manifest["trial"]["duration_s"] = 2.25
    manifest["live"]["transport"]["block_steps"] = 20
    manifest["live"]["transport"]["block_duration_s"] = 1.0
    manifest["live"]["transport"]["blocks_per_trial"] = 1
    manifest["metrics"]["pass_criterion"]["settled_after_s"] = 0.5

    spec, model, trim, state, command = harness.control_fixture(manifest)
    plant = harness._control_plant(manifest, spec, model)
    recordings = [
        (
            f"recording-{seed}",
            harness.control_recording(manifest, plant, trim, state, command, seed),
        )
        for seed in (0, 1)
    ]
    with jax.enable_x64(True):
        artifacts = dict(
            generic=fit(harness.control_collection(manifest, recordings)),
            structured=structured_fit(
                [flight for _, flight in recordings],
                FitSpec(holdout=Holdout.by_group(), steps=3, horizons_s=(0.1, 0.4)),
            ).belief,
        )

    def reference_fn(times):
        return harness.control_reference(state, times, manifest["tracking_reference"])

    row = harness._live_trial(
        manifest,
        adopting=False,
        artifacts=artifacts,
        plant=harness._control_tracking_plant(
            manifest, harness.control_initial_state(manifest, state, 101), command
        ),
        warmup=recordings[0][1],
        reference_fn=reference_fn,
        anchor_state=state,
        session="live-frozen",
        directory=tmp_path / "f",
    )
    assert row["arm"] == "frozen"
    assert row["swapped"] is False and row["offers"] == 0
    assert row["terminated"] is False and row["worker_error"] is None
    # It still scores and refits every block: the two arms differ in the swap.
    assert len(row["blocks"]) == 1
    assert row["blocks"][0]["refit_wall_seconds"] > 0.0


# --- the commands themselves, end to end ------------------------------------


def _shortened(path, manifest, monkeypatch, constant):
    """Write a shortened manifest and let the harness run that one instead.

    The frozen digest is the gate on which manifest a *measurement* may use.
    The two tests below are not measurements: they run the commands end to end
    to prove those paths still work, on a trial short enough to belong in a test
    suite, so they point the digest constant at the manifest they wrote. Every
    other constant these manifests declare is the frozen one, and no committed
    file is touched.
    """
    path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    monkeypatch.setattr(harness, constant, harness.sha256(path), raising=True)
    return path


def _smoke_manifest(source, **trial):
    manifest = copy.deepcopy(source)
    manifest["arms"]["structured"]["optimization_steps"] = 3
    manifest["trial"].update(trial)
    manifest["metrics"]["pass_criterion"]["settled_after_s"] = 0.2
    return manifest


@pytest.mark.cascade
def test_the_control_command_runs_and_replays_end_to_end(tmp_path, monkeypatch):
    """The whole `control` command, including the row every trial writes."""
    pytest.importorskip("cascade")

    source = harness.frozen_control_manifest(
        Path(__file__).resolve().parents[1] / "docs/harness/control-v4.json"
    )
    path = _shortened(
        tmp_path / "control.json",
        _smoke_manifest(source, intervals=20, duration_s=1.0),
        monkeypatch,
        "CONTROL_MANIFEST_SHA256",
    )
    decision = harness.control(path, tmp_path / "run")
    assert decision["manifest"] == "control-v4"
    assert decision["trials"] == 4
    rows = harness.read(tmp_path / "run" / "results.json")
    assert len(rows) == 4
    assert {(row["repetition"], row["arm"]) for row in rows} == {
        (repetition, arm) for repetition in range(2) for arm in harness.CONTROL_ARMS
    }
    for row in rows:
        assert row["completed_intervals"] == 20
        assert row["terminated"] is False
        assert math.isfinite(row["wall"]["trial_seconds"])
        assert math.isfinite(row["tracking_rmse"]["position_rmse_m"])
    replay = harness.verify(tmp_path / "run", tmp_path / "control-reference.json")
    assert replay["tier"] == "control"
    assert replay["verified_trials"] == 4


@pytest.mark.cascade
def test_the_live_command_runs_and_replays_end_to_end(tmp_path, monkeypatch):
    """The whole `live` command, including the row every trial writes."""
    pytest.importorskip("cascade")

    manifest = _smoke_manifest(
        harness.frozen_live_manifest(MANIFEST), intervals=70, duration_s=3.5
    )
    manifest["live"]["transport"].update(
        block_steps=20,
        block_duration_s=1.0,
        blocks_per_trial=2,
        offer_release_offset_intervals=20,
    )
    path = _shortened(
        tmp_path / "live.json", manifest, monkeypatch, "LIVE_MANIFEST_SHA256"
    )
    decision = harness.live(path, tmp_path / "run")
    assert decision["manifest"] == "live-v2"
    assert decision["trials"] == 4
    rows = harness.read(tmp_path / "run" / "results.json")
    assert len(rows) == 4
    assert {(row["repetition"], row["arm"]) for row in rows} == {
        (repetition, arm) for repetition in range(2) for arm in harness.LIVE_ARMS
    }
    offset = manifest["live"]["transport"]["offer_release_offset_intervals"]
    for row in rows:
        assert row["completed_intervals"] == 70
        assert row["terminated"] is False and row["worker_error"] is None
        assert row["dropped_blocks"] == 0
        assert math.isfinite(row["wall"]["trial_seconds"])
        assert math.isfinite(row["tracking_rmse"]["position_rmse_m"])
        # The swap is where the gates and the declared offset put it, or absent.
        _, expected = harness.live_expected_swap(
            row["blocks"], offset, manifest["trial"]["intervals"]
        )
        assert row["swap_interval"] == (expected if row["arm"] == "adopting" else None)
    replay = harness.verify(tmp_path / "run", tmp_path / "live-reference.json")
    assert replay["tier"] == "live"
    assert replay["verified_trials"] == 4
