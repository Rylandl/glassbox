"""The evidence tier's contracts: the frozen manifest, the band, the decision.

These tests never fit anything. They exercise the manifest digest, the declared
channel groups, the coverage arithmetic on saved arrays, and every way the band
decision fails closed or reports. Coverage itself is measured inside the
synthetic, platform and control runs, on the rows those tiers already score;
what is decided from it is decided here.
"""

import copy
import math
from pathlib import Path

import numpy as np
import pytest

from glassbox.experimental import harness

MANIFEST = Path(__file__).resolve().parents[1] / "docs/harness/evidence-v2.json"
GROUPS = ("world_velocity", "body_rate", "rotation_entries")
HORIZONS = 3


@pytest.fixture(scope="module")
def manifest():
    return harness.frozen_evidence_manifest(MANIFEST)


def platform_rows(manifest, coverage=0.9):
    """One comfortably passing row per declared platform corpus."""
    return [
        dict(
            case=case,
            regime="held_out",
            scored_rows=64,
            horizon_steps=HORIZONS,
            channels=15,
            coverage={group: [coverage] * HORIZONS for group in GROUPS},
            pooled_coverage={group: coverage for group in GROUPS},
        )
        for case in manifest["tiers"]["platform"]["cases"]
    ]


def expected_for(manifest, tier, regime):
    return {(case, regime) for case in manifest["tiers"][tier]["cases"]}


def reference_from(rows, tier="platform"):
    return dict(
        coverage={
            tier: {row["case"]: {row["regime"]: dict(row["coverage"])} for row in rows}
        }
    )


def decide(
    manifest,
    rows,
    *,
    tier="platform",
    regime="held_out",
    reference=None,
    enforced=False,
):
    """Decide these rows under either reading of the band.

    ``enforced`` is the one thing that separates the committed manifest, which
    gates, from the reported-only reading it replaced, so every failure below
    is measured against both.
    """
    manifest = copy.deepcopy(manifest)
    manifest["decision"]["enforced"] = enforced
    return harness.evidence_decide(
        manifest, tier, rows, expected_for(manifest, tier, regime), reference
    )


def gates(decision):
    return {breach["gate"] for breach in decision["gate_breaches"]} | {
        regression["gate"] for regression in decision["reference_regressions"]
    }


# --- the frozen manifest ----------------------------------------------------


def test_the_manifest_digest_gates_the_evidence_contract(tmp_path, manifest):
    assert harness.sha256(MANIFEST) == harness.EVIDENCE_MANIFEST_SHA256
    assert harness.COMMITTED_EVIDENCE_MANIFEST == MANIFEST
    altered = tmp_path / "evidence-v2.json"
    widened = copy.deepcopy(manifest)
    widened["band"]["minimum"] = 0.5
    harness.write(altered, widened)
    with pytest.raises(ValueError, match="evidence manifest digest"):
        harness.frozen_evidence_manifest(altered)


def test_the_manifest_declares_the_band_the_nominal_level_and_the_plan(manifest):
    from glassbox.experimental.default_model import RECIPE

    assert manifest["band"] == {"minimum": 0.85, "maximum": 0.95, **manifest["band"]}
    assert (manifest["band"]["minimum"], manifest["band"]["maximum"]) == (0.85, 0.95)
    assert manifest["envelope"]["nominal_coverage"] == 0.9
    assert manifest["decision"]["enforced"] is True
    assert set(manifest["recipe"]) == set(RECIPE)
    # The plan is pinned even though the rest of the recipe is not, exactly as
    # the other three manifests pin theirs.
    assert harness.declared_plan(manifest) == manifest["recipe"]
    assert sorted(manifest["tiers"]) == ["control", "platform", "synthetic"]
    assert manifest["tiers"]["control"]["cases"] == ["recording-3"]
    assert len(manifest["tiers"]["synthetic"]["cases"]) == 27


def test_a_manifest_declaring_another_evaluation_plan_is_refused(manifest):
    for name in harness.PLAN_CONSTANTS:
        other = copy.deepcopy(manifest)
        other["recipe"][name] = manifest["recipe"][name] * 2
        with pytest.raises(ValueError, match="evaluation plan"):
            harness.declared_plan(other)


# --- the declared channel groups --------------------------------------------


def test_the_rigid_body_groups_partition_the_fifteen_declared_channels():
    groups = harness.evidence_groups("rigid_body_15", 15)
    assert tuple(groups) == GROUPS
    covered = [index for indices in groups.values() for index in indices]
    assert sorted(covered) == list(range(15))
    with pytest.raises(ValueError, match="fifteen observed channels"):
        harness.evidence_groups("rigid_body_15", 14)


def test_a_synthetic_family_groups_every_channel_on_its_own():
    assert harness.evidence_groups("per_channel", 3) == {
        "channel_0": (0,),
        "channel_1": (1,),
        "channel_2": (2,),
    }
    with pytest.raises(ValueError, match="at least one channel"):
        harness.evidence_groups("per_channel", 0)
    with pytest.raises(ValueError, match="undeclared channel grouping"):
        harness.evidence_groups("whatever", 15)


# --- coverage from saved arrays ---------------------------------------------


def test_coverage_counts_row_and_channel_pairs_inside_the_half_width():
    targets = np.zeros((4, 2, 3))
    prediction = np.array([[[0.0, 2.0, 0.0], [0.0, 0.0, 0.0]]] * 4)
    half = np.ones((4, 2, 3))
    groups = harness.evidence_groups("per_channel", 3)
    measured = harness.envelope_coverage(prediction, targets, half, groups)
    assert measured["scored_rows"] == 4
    assert measured["horizon_steps"] == 2
    assert measured["coverage"]["channel_1"] == [0.0, 1.0]
    assert measured["coverage"]["channel_0"] == [1.0, 1.0]
    assert measured["pooled_coverage"]["channel_1"] == 0.5
    # A boundary error is covered: the half-width is a closed interval.
    edge = harness.envelope_coverage(half, targets, half, groups)
    assert edge["pooled_coverage"]["channel_2"] == 1.0


def test_coverage_refuses_half_widths_that_do_not_describe_the_predictions():
    groups = harness.evidence_groups("per_channel", 3)
    targets = np.zeros((4, 2, 3))
    with pytest.raises(ValueError, match="shaped like the predictions"):
        harness.envelope_coverage(targets, targets, np.ones((2, 3)), groups)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        harness.envelope_coverage(targets, targets, -np.ones((4, 2, 3)), groups)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        harness.envelope_coverage(targets, targets, np.full((4, 2, 3), np.nan), groups)


# --- the band decision ------------------------------------------------------


def test_coverage_inside_the_band_is_accepted_and_json_serializable(manifest):
    decision = decide(manifest, platform_rows(manifest))
    assert decision["accepted"] and decision["band_met"]
    assert decision["band_enforced"] is False
    assert decision["gating_band_breaches"] == 0
    assert decision["band"] == {"minimum": 0.85, "maximum": 0.95}
    assert not decision["band_breaches"] and not gates(decision)
    assert len(decision["coverage"]) == 5
    assert len(decision["coverage"]["arp/held_out"]["body_rate"]) == HORIZONS
    harness.json.dumps(decision, allow_nan=False)


@pytest.mark.parametrize(
    ("coverage", "below"), [(0.80, True), (0.99, False), (0.8499, True)]
)
def test_a_corpus_outside_the_band_is_reported_and_gates_when_enforced(
    manifest, coverage, below
):
    rows = platform_rows(manifest)
    rows[2]["coverage"]["body_rate"] = [coverage] * HORIZONS
    reference = reference_from(platform_rows(manifest))
    unenforced = decide(manifest, rows, reference=reference)
    # Unenforced: measured, reported, and it does not decide the run.
    assert unenforced["band_breaches"] and not unenforced["band_met"]
    assert unenforced["accepted"] is True
    breach = unenforced["band_breaches"][0]
    assert (breach["case"], breach["group"], breach["below"]) == (
        "arp",
        "body_rate",
        below,
    )
    assert breach["gating"] is True
    assert unenforced["gating_band_breaches"] == HORIZONS
    rejected = decide(manifest, rows, reference=reference, enforced=True)
    assert rejected["accepted"] is False and rejected["decision"] == "reject"


def test_a_band_breach_the_reference_already_fails_is_reported_not_gated(manifest):
    rows = platform_rows(manifest)
    rows[2]["coverage"]["body_rate"] = [0.70] * HORIZONS
    decision = decide(manifest, rows, reference=reference_from(rows), enforced=True)
    assert decision["band_breaches"] and not decision["band_met"]
    assert all(not breach["gating"] for breach in decision["band_breaches"])
    assert decision["gating_band_breaches"] == 0
    assert decision["accepted"] is True


def test_a_case_that_moves_further_outside_the_band_is_a_regression(manifest):
    reference = reference_from(platform_rows(manifest, coverage=0.80))
    rows = platform_rows(manifest)
    rows[1]["coverage"]["world_velocity"] = [0.70] * HORIZONS
    decision = decide(manifest, rows, reference=reference, enforced=True)
    assert gates(decision) == {"reference_band_excess"}
    assert decision["accepted"] is False
    regression = decision["reference_regressions"][0]
    assert regression["case"] == "x8"
    assert regression["value"] == pytest.approx(0.15)
    assert regression["reference"] == pytest.approx(0.05)
    # Unenforced the same regression is recorded and decides nothing.
    assert decide(manifest, rows, reference=reference)["accepted"] is True
    # Moving back inside the band from a reference that was outside is not one.
    assert decide(
        manifest, platform_rows(manifest), reference=reference, enforced=True
    )["accepted"]


def test_a_reference_without_the_case_fails_closed(manifest):
    rows = platform_rows(manifest)
    reference = reference_from(rows)
    del reference["coverage"]["platform"]["idf"]
    decision = decide(manifest, rows, reference=reference, enforced=True)
    assert gates(decision) == {"reference_present"}
    assert decision["accepted"] is False
    assert decide(manifest, rows, reference=reference)["accepted"] is True
    short = reference_from(rows)
    short["coverage"]["platform"]["epfl"]["held_out"]["body_rate"] = [0.9]
    assert gates(decide(manifest, rows, reference=short)) == {"reference_present"}


@pytest.mark.parametrize("problem", ["missing", "duplicate", "undeclared"])
def test_missing_duplicate_and_undeclared_corpora_fail_closed(manifest, problem):
    rows = platform_rows(manifest)
    if problem == "missing":
        rows = rows[:-1]
        expected = "case_present"
    elif problem == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
        expected = "case_unique"
    else:
        extra = copy.deepcopy(rows[0])
        extra["case"] = "another-corpus"
        rows.append(extra)
        expected = "case_declared"
    decision = decide(manifest, rows, enforced=True)
    assert expected in gates(decision)
    assert decision["accepted"] is False and not decision["band_met"]
    reported = decide(manifest, rows)
    assert expected in gates(reported) and reported["accepted"] is True


def test_a_tier_measuring_a_different_case_set_than_the_manifest_fails_closed(
    manifest,
):
    rows = platform_rows(manifest)
    expected = expected_for(manifest, "platform", "held_out") - {("epfl", "held_out")}
    enforced = copy.deepcopy(manifest)
    enforced["decision"]["enforced"] = True
    decision = harness.evidence_decide(enforced, "platform", rows, expected)
    assert "cases_declared" in gates(decision)
    assert decision["accepted"] is False


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.5, "0.9", None])
def test_a_coverage_that_is_not_a_fraction_fails_closed(manifest, value):
    rows = platform_rows(manifest)
    rows[0]["coverage"]["body_rate"] = [0.9, value, 0.9]
    decision = decide(manifest, rows, enforced=True)
    assert gates(decision) == {"finite_coverage"}
    assert decision["accepted"] is False and not decision["band_met"]
    # A nonfinite coverage never reaches the decision as a JSON number.
    recorded = decision["gate_breaches"][0]["value"]
    assert recorded is None or math.isfinite(recorded)
    harness.json.dumps(decision, allow_nan=False)
    reported = decide(manifest, rows)
    assert gates(reported) == {"finite_coverage"} and reported["accepted"] is True
    harness.json.dumps(reported, allow_nan=False)


@pytest.mark.parametrize(
    "problem", ["groups", "horizons", "channels", "scored_rows", "steps"]
)
def test_a_malformed_coverage_table_fails_closed(manifest, problem):
    rows = platform_rows(manifest)
    row, expected = rows[0], f"{problem}_declared"
    if problem == "groups":
        del row["coverage"]["body_rate"]
        expected = "groups_declared"
    elif problem == "horizons":
        row["coverage"]["body_rate"] = [0.9] * (HORIZONS + 1)
        expected = "group_horizons"
    elif problem == "channels":
        row["channels"] = 12
        expected = "channels_declared"
    elif problem == "scored_rows":
        row["scored_rows"] = 0
        expected = "scored_rows"
    else:
        row["horizon_steps"] = 0
        expected = "horizon_steps"
    decision = decide(manifest, rows, enforced=True)
    assert expected in gates(decision)
    assert decision["accepted"] is False
    assert expected in gates(decide(manifest, rows))


def test_the_synthetic_and_control_tiers_use_their_own_declared_scope(manifest):
    control = [
        dict(
            case="recording-3",
            regime="reserved",
            scored_rows=100,
            horizon_steps=5,
            channels=15,
            coverage={group: [0.9] * 5 for group in GROUPS},
            pooled_coverage={group: 0.9 for group in GROUPS},
        )
    ]
    decision = decide(manifest, control, tier="control", regime="reserved")
    assert (
        decision["accepted"] and decision["band_met"] and decision["tier"] == "control"
    )
    synthetic = [
        dict(
            case=case,
            regime=regime,
            scored_rows=48,
            horizon_steps=5,
            channels=1,
            coverage={"channel_0": [0.9] * 5},
            pooled_coverage={"channel_0": 0.9},
        )
        for case in manifest["tiers"]["synthetic"]["cases"]
        for regime in ("matched",)
    ]
    expected = {(row["case"], row["regime"]) for row in synthetic}
    replayed = harness.evidence_decide(manifest, "synthetic", synthetic, expected)
    assert replayed["accepted"] and replayed["band_met"]
    # A synthetic case that names a rigid-body group fails closed: per_channel
    # is what this tier declares.
    synthetic[0]["coverage"] = {"body_rate": [0.9] * 5}
    broken = harness.evidence_decide(manifest, "synthetic", synthetic, expected)
    assert "groups_declared" in gates(broken)
