"""Platform-tier contracts: the corpus adapter, the origins, the decision, the replay.

None of these tests reads a pinned corpus. The adapter is exercised on synthetic
canonical trajectories, the origins on fabricated segments, the decision on
fabricated scores, and the replay on a model that was never fitted.
"""

import copy
import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np
import pytest
from conftest import seal_evidence

from glassbox.core.data import Trajectory
from glassbox.experimental import harness
from glassbox.experimental.default_model import RECIPE, steps_for
from glassbox.experimental.sequence_collection import SequenceSegment
from glassbox.experimental.sequence_model import (
    initialize_sequence_model,
    sequence_windows,
)

MANIFEST = Path(__file__).resolve().parents[1] / "docs/harness/platform-v3.json"
REFERENCE = MANIFEST.parent / "platform-reference.json"
DT_S = 0.02
TOLERANCE = 0.001


@pytest.fixture(scope="module")
def manifest():
    return harness.frozen_platform_manifest(MANIFEST)


@pytest.fixture(scope="module")
def flight(quadrotor_flight):
    """One synthetic canonical quadrotor recording, long enough to cut up."""

    return quadrotor_flight(3, 1.2, DT_S)


def adapt(trajectory, dt_s=DT_S):
    return harness.trajectory_segments(
        "flight", trajectory, dt_s=dt_s, tolerance_fraction=TOLERANCE
    )


# --- the corpus adapter -----------------------------------------------------


def test_the_adapter_produces_the_declared_observed_channels(flight):
    segments = adapt(flight)
    assert len(segments) == 1
    segment = segments[0]
    assert segment.recording_id == "flight" and segment.start_row == 0
    assert segment.dt_s == DT_S
    assert len(harness.OBSERVED_CHANNELS) == 15
    assert segment.states.shape == (len(flight.time_s), 15)
    assert segment.inputs.shape == (len(flight.time_s) - 1, flight.control_size)
    np.testing.assert_array_equal(segment.inputs, flight.controls)
    # Velocity, then body rate, then the rotation entries. Position is absent.
    np.testing.assert_allclose(segment.states[:, 0:3], flight.states[:, 3:6])
    np.testing.assert_allclose(segment.states[:, 3:6], flight.states[:, 10:13])
    assert not np.allclose(segment.states[:, 0:3], flight.states[:, 0:3])


def test_the_adapted_rotation_entries_are_orthonormal_and_body_to_world(flight):
    rotation = adapt(flight)[0].states[:, 6:].reshape(-1, 3, 3)
    products = np.einsum("nij,nkj->nik", rotation, rotation)
    identity = np.broadcast_to(np.eye(3), products.shape)
    np.testing.assert_allclose(products, identity, atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(rotation), 1.0, atol=1e-12)
    # Row-major body-to-world: the same matrix the geometry helper returns.
    from glassbox.core.geometry import quaternion_to_rotation_matrices

    np.testing.assert_allclose(
        rotation, quaternion_to_rotation_matrices(flight.states[:, 6:10])
    )


def test_every_observed_channel_name_carries_a_unit_and_a_frame():
    for channel in harness.OBSERVED_CHANNELS:
        name, _, tail = channel.partition(" [")
        assert name and tail.endswith("]")
        unit, _, frame = tail[:-1].partition(",")
        assert unit and frame
    assert len(set(harness.OBSERVED_CHANNELS)) == 15


def test_input_channel_names_come_from_the_trajectory_spec(flight):
    channels = harness.command_channels(flight.spec)
    assert len(channels) == flight.control_size
    assert len(set(channels)) == len(channels)
    for channel, declared in zip(channels, flight.spec.controls, strict=True):
        assert channel == f"{declared.name} [{declared.unit},{declared.role}]"


def test_a_timing_gap_splits_the_recording_and_pads_nothing(flight):
    time_s = np.array(flight.time_s)
    time_s[25:] += 5 * DT_S
    segments = adapt(replace(flight, time_s=time_s))
    assert [segment.start_row for segment in segments] == [0, 25]
    assert [len(segment.states) for segment in segments] == [25, len(time_s) - 25]
    # Every row survives; the split drops the interval, not the samples.
    assert sum(len(segment.states) for segment in segments) == len(time_s)
    rows = harness.observed_rows(flight)
    for segment in segments:
        start = segment.start_row
        np.testing.assert_array_equal(
            segment.states, rows[start : start + len(segment.states)]
        )
        assert len(segment.inputs) == len(segment.states) - 1


def test_a_nonfinite_observed_row_splits_the_recording(flight):
    states = np.array(flight.states)
    states[30, 6:10] = 0.0  # an attitude that cannot be normalized
    broken = replace(flight, states=states)
    assert not np.isfinite(harness.observed_rows(broken)[30]).all()
    segments = adapt(broken)
    assert [segment.start_row for segment in segments] == [0, 31]
    assert all(np.isfinite(segment.states).all() for segment in segments)
    assert sum(len(segment.states) for segment in segments) == len(states) - 1


def test_a_sample_period_the_recording_does_not_share_yields_no_segments(flight):
    assert adapt(flight, dt_s=5 * DT_S) == ()
    with pytest.raises(ValueError, match="sample interval"):
        adapt(flight, dt_s=0.0)


def test_the_adapter_refuses_an_unknown_state_schema(flight):
    assert isinstance(flight, Trajectory)
    assert flight.spec.state_schema == "rigid_body_13_nwu_flu_wxyz_v1"
    # A canonical TrajectorySpec cannot carry another schema, so the guard is
    # exercised on a stand-in that supplies the two fields the adapter reads.
    stand_in = SimpleNamespace(
        spec=SimpleNamespace(state_schema="something_else"), states=flight.states
    )
    with pytest.raises(ValueError, match="state schema"):
        harness.observed_rows(stand_in)
    narrow = SimpleNamespace(
        spec=SimpleNamespace(state_schema=flight.spec.state_schema),
        states=flight.states[:, :12],
    )
    with pytest.raises(ValueError, match="thirteen rows"):
        harness.observed_rows(narrow)


# --- which recordings a corpus holds ----------------------------------------


def _tree(root, names):
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


def test_the_declared_recording_counts_gate_the_tree_on_disk(tmp_path):
    entry = dict(
        name="demo",
        directory="demo",
        file_pattern="**/*.npz",
        held_out_patterns=["test/*.npz"],
        recordings=dict(total=3, held_out=1, training=2),
    )
    _tree(tmp_path, ["demo/train/a.npz", "demo/train/b.npz", "demo/test/c.npz"])
    training, held_out = harness.corpus_paths(entry, tmp_path)
    assert [path.stem for path in training] == ["a", "b"]
    assert [path.stem for path in held_out] == ["c"]
    _tree(tmp_path, ["demo/train/d.npz"])
    with pytest.raises(ValueError, match="frozen manifest"):
        harness.corpus_paths(entry, tmp_path)


def test_recording_identities_must_be_unique_stems(tmp_path):
    entry = dict(
        name="demo",
        directory="demo",
        file_pattern="**/*.npz",
        held_out_patterns=["test/*.npz"],
        recordings=dict(total=2, held_out=1, training=1),
    )
    _tree(tmp_path, ["demo/train/a.npz", "demo/test/a.npz"])
    with pytest.raises(ValueError, match="unique stems"):
        harness.corpus_paths(entry, tmp_path)


# --- origin selection -------------------------------------------------------


def segment(name, rows, start_row):
    return SequenceSegment(
        "r", name, np.zeros((rows, 15)), np.zeros((rows - 1, 2)), 0.1, start_row
    )


def test_origin_selection_respects_context_horizon_stride_and_boundaries():
    steps = dict(history=4, horizon=3, delay=1)
    segments = (segment("a", 20, 0), segment("b", 12, 40), segment("short", 7, 80))
    picked = harness.platform_origins(segments, steps, 3)
    assert [(s.segment_id, row) for s, row in picked] == [
        ("a", 4),
        ("a", 7),
        ("a", 10),
        ("a", 13),
        ("a", 16),
        ("b", 4),
        ("b", 7),
    ]
    for chosen, row in picked:
        assert row >= steps["history"]
        assert row + steps["horizon"] <= len(chosen.states) - 1
    # A segment with no room for one whole window contributes nothing; one with
    # room for exactly one contributes exactly one.
    assert harness.platform_origins((segment("short", 7, 0),), steps, 1) == []
    exact = harness.platform_origins((segment("exact", 8, 0),), steps, 1)
    assert [(s.segment_id, row) for s, row in exact] == [("exact", 4)]


def test_stride_one_selects_every_admissible_origin():
    steps = dict(history=2, horizon=2, delay=1)
    picked = harness.platform_origins((segment("a", 10, 3),), steps, 1)
    assert [row for _, row in picked] == [2, 3, 4, 5, 6, 7]


# --- scoring ----------------------------------------------------------------


def test_the_metric_is_one_scalar_over_rows_and_components():
    targets = np.zeros((4, 3, 15))
    prediction = np.zeros((4, 3, 15))
    prediction[:, -1, 0] = 2.0
    prediction[:, :, 3] = 1.0
    measured = harness.platform_measure(prediction, targets)
    # One component off by 2 at the final step: sqrt(mean of 4/3).
    assert measured["final_step"]["velocity_rmse_m_s"] == pytest.approx(
        np.sqrt(4.0 / 3.0)
    )
    assert measured["horizon_prefix"]["body_rate_rmse_rad_s"] == pytest.approx(
        np.sqrt(1.0 / 3.0)
    )
    assert measured["final_step"]["body_rate_rmse_rad_s"] == pytest.approx(
        np.sqrt(1.0 / 3.0)
    )
    assert measured["horizon_prefix"]["velocity_rmse_m_s"] == pytest.approx(
        np.sqrt(4.0 / 9.0)
    )


def test_the_hold_baseline_repeats_the_last_observed_row():
    past = np.arange(2 * 4 * 15, dtype=float).reshape(2, 4, 15)
    hold = harness.hold_current(past, 3)
    assert hold.shape == (2, 3, 15)
    for step in range(3):
        np.testing.assert_array_equal(hold[:, step], past[:, -1])


def test_scores_are_reported_per_recording_as_well_as_pooled():
    targets = np.zeros((4, 2, 15))
    prediction = np.zeros((4, 2, 15))
    prediction[2:, -1, 0] = 3.0
    ids = np.array(["one", "one", "two", "two"])
    scored = harness.platform_score(prediction, targets, ids)
    assert scored["rows"] == 4 and sorted(scored["recordings"]) == ["one", "two"]
    assert scored["recordings"]["one"]["final_step"][
        "velocity_rmse_m_s"
    ] == pytest.approx(0.0)
    assert scored["recordings"]["two"]["final_step"][
        "velocity_rmse_m_s"
    ] == pytest.approx(np.sqrt(3.0))


# --- the decision -----------------------------------------------------------


def one_score(values):
    return dict(
        rows=100, final_step=dict(values), horizon_prefix=dict(values), recordings={}
    )


def passing_rows(manifest):
    """One complete, comfortably passing row per declared corpus."""

    rows = []
    for entry in manifest["corpora"]:
        arms = entry["structured"]["arms"]
        rows.append(
            dict(
                corpus=entry["name"],
                status="complete",
                evaluation_rows=100,
                generic=one_score({metric: 0.1 for metric in harness.METRICS}),
                hold_current=one_score({metric: 0.9 for metric in harness.METRICS}),
                structured={
                    arm: one_score(
                        {metric: 0.2 + 0.1 * index for metric in harness.METRICS}
                    )
                    for index, arm in enumerate(arms)
                },
            )
        )
    return rows


def gates(decision):
    return {breach["gate"] for breach in decision["gate_breaches"]} | {
        breach["gate"] for breach in decision["rule_breaches"]
    }


def met_reference(manifest, **moved):
    """A reference that met the rule everywhere, with named cases moved.

    The rule gates a case only where the reference already meets it, so a test
    of the gate has to say what the reference measured. ``moved`` names a
    corpus and the reference values to use for it.
    """
    reference = reference_from(passing_rows(manifest))
    for corpus, values in moved.items():
        reference["final_step_rmse"][corpus].update(values)
    return reference


def test_a_complete_measurement_is_accepted_and_json_serializable(manifest):
    decision = harness.platform_decide(manifest, passing_rows(manifest))
    assert decision["accepted"] and decision["decision"] == "accept"
    assert decision["rule_met"] and decision["rule_enforced"] is True
    assert decision["corpora"] == 5 and not gates(decision)
    assert not decision["reference_compared"] and not decision["reference_regressions"]
    summary = decision["final_step_rmse"]["arp"]["velocity_rmse_m_s"]
    assert summary["generic"] == 0.1 and summary["comparator"] == 0.2
    assert summary["comparator_arm"] == "structured"
    assert summary["allowance"] == 0.709 and summary["hold_current"] == 0.9
    # With no reference compared there is no case the reference already meets,
    # so the rule is reported on every case and gates none of them.
    assert summary["reference"] is None and summary["reference_meets_rule"] is None
    assert decision["gating_rule_breaches"] == 0
    json.dumps(decision, allow_nan=False)


def test_the_rule_gates_only_a_case_the_reference_already_met(manifest):
    """The whole semantics, isolated: no regression, one case gated, one not."""

    reference = met_reference(
        manifest,
        x8={"velocity_rmse_m_s": 0.19},
        arp={"body_rate_rmse_rad_s": 0.25},
    )
    rows = passing_rows(manifest)
    # x8 velocity: the reference met the rule at 0.19 and this run does not.
    # Its regression limit is 0.19 * 1.05 + 0.005 = 0.2045, so only the rule
    # can trip.
    next(r for r in rows if r["corpus"] == "x8")["generic"]["final_step"][
        "velocity_rmse_m_s"
    ] = 0.2001
    # arp body rate: the reference already failed the rule at 0.25, above the
    # 0.2 comparator, and this run still fails it inside the regression limit.
    next(r for r in rows if r["corpus"] == "arp")["generic"]["final_step"][
        "body_rate_rmse_rad_s"
    ] = 0.26
    decision = harness.platform_decide(manifest, rows, reference)
    assert decision["reference_regressions"] == []
    assert not decision["rule_met"]
    assert not decision["accepted"] and decision["decision"] == "reject"
    assert decision["gating_rule_breaches"] == 1
    gated = {b["corpus"] for b in decision["rule_breaches"] if b["gating"]}
    reported = {b["corpus"] for b in decision["rule_breaches"] if not b["gating"]}
    assert gated == {"x8"} and reported == {"arp"}
    arp = decision["final_step_rmse"]["arp"]["body_rate_rmse_rad_s"]
    assert arp["reference"] == 0.25 and arp["reference_meets_rule"] is False
    x8 = decision["final_step_rmse"]["x8"]["velocity_rmse_m_s"]
    assert x8["reference"] == 0.19 and x8["reference_meets_rule"] is True


def test_a_case_the_reference_already_failed_is_reported_and_accepted(manifest):
    """The defect this manifest fixes: an unmet case must not block a change."""

    reference = met_reference(manifest, arp={"body_rate_rmse_rad_s": 0.25})
    rows = passing_rows(manifest)
    next(r for r in rows if r["corpus"] == "arp")["generic"]["final_step"][
        "body_rate_rmse_rad_s"
    ] = 0.26
    decision = harness.platform_decide(manifest, rows, reference)
    assert decision["accepted"] and decision["decision"] == "accept"
    # Accepted is not met: the status table's row reads rule_met, not accepted.
    assert not decision["rule_met"]
    assert decision["gating_rule_breaches"] == 0
    assert [b["gating"] for b in decision["rule_breaches"]] == [False]
    json.dumps(decision, allow_nan=False)


def test_a_regression_rejects_a_run_that_meets_the_rule_everywhere(manifest):
    """Condition (a) stands on its own: the rule held and the run still fails."""

    reference = met_reference(manifest, idf={"velocity_rmse_m_s": 0.1})
    rows = passing_rows(manifest)
    next(r for r in rows if r["corpus"] == "idf")["generic"]["final_step"][
        "velocity_rmse_m_s"
    ] = 0.1 * 1.05 + 0.005 + 1e-9
    decision = harness.platform_decide(manifest, rows, reference)
    assert decision["rule_met"] and not decision["accepted"]
    assert decision["rule_breaches"] == []
    assert [r["corpus"] for r in decision["reference_regressions"]] == ["idf"]


@pytest.mark.parametrize(
    "problem", ["missing", "unfinished", "nonfinite", "duplicate", "undeclared"]
)
def test_a_structural_failure_rejects_whatever_the_reference_says(manifest, problem):
    """Condition (c) never waits for the reference: structure always closes."""

    # A reference that fails the rule on every case, so nothing the rule says
    # can be what rejects the run.
    reference = reference_from(passing_rows(manifest))
    for scores in reference["final_step_rmse"].values():
        for metric in harness.METRICS:
            scores[metric] = 9.0
    rows = passing_rows(manifest)
    expected = {
        "missing": "corpus_present",
        "unfinished": "fit_complete",
        "nonfinite": "finite_final_step_rmse",
        "duplicate": "corpus_unique",
        "undeclared": "corpus_declared",
    }[problem]
    if problem == "missing":
        rows = [row for row in rows if row["corpus"] != "idf"]
    elif problem == "unfinished":
        rows[0]["status"] = "failed"
    elif problem == "nonfinite":
        rows[0]["generic"]["final_step"]["velocity_rmse_m_s"] = float("nan")
    elif problem == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    else:
        rows.append({**copy.deepcopy(rows[0]), "corpus": "crazyflie"})
    decision = harness.platform_decide(manifest, rows, reference)
    assert not decision["accepted"] and not decision["rule_met"]
    assert expected in gates(decision)
    json.dumps(decision, allow_nan=False)


def test_a_corpus_above_the_comparator_fails(manifest):
    reference = met_reference(manifest, x8={"velocity_rmse_m_s": 0.19})
    rows = passing_rows(manifest)
    row = next(r for r in rows if r["corpus"] == "x8")
    row["generic"]["final_step"]["velocity_rmse_m_s"] = 0.2001
    decision = harness.platform_decide(manifest, rows, reference)
    assert not decision["rule_met"] and gates(decision) == {
        "comparator_final_step_rmse"
    }
    breach = decision["rule_breaches"][0]
    assert breach["corpus"] == "x8" and breach["arm"] == "structured"
    assert breach["limit"] == 0.2 and breach["value"] == 0.2001
    assert breach["gating"] is True
    # The frozen manifest enforces the rule, so one corpus the reference met
    # and this run does not rejects the whole run.
    assert not decision["accepted"] and decision["decision"] == "reject"
    reporting = copy.deepcopy(manifest)
    reporting["decision"]["enforced"] = False
    assert harness.platform_decide(reporting, rows, reference)["accepted"]


def test_the_comparator_is_the_better_arm_on_those_rows(manifest):
    rows = passing_rows(manifest)
    row = next(r for r in rows if r["corpus"] == "epfl")
    row["structured"]["structured_residual"]["final_step"]["velocity_rmse_m_s"] = 0.05
    row["generic"]["final_step"]["velocity_rmse_m_s"] = 0.1
    decision = harness.platform_decide(manifest, rows)
    breach = decision["rule_breaches"][0]
    assert breach["corpus"] == "epfl" and breach["arm"] == "structured_residual"
    assert breach["limit"] == 0.05
    summary = decision["final_step_rmse"]["epfl"]
    assert summary["body_rate_rmse_rad_s"]["comparator_arm"] == "structured"


def test_the_frozen_manifest_enforces_the_rule_on_every_corpus(manifest):
    """Enforcement is per corpus: four passing corpora do not carry a fifth."""
    assert manifest["decision"]["enforced"] is True
    reference = reference_from(passing_rows(manifest))
    for entry in manifest["corpora"]:
        rows = passing_rows(manifest)
        row = next(r for r in rows if r["corpus"] == entry["name"])
        # Above every arm on those rows, below every declared allowance, so
        # only the comparator gate can trip and only on this corpus.
        row["generic"]["final_step"]["body_rate_rmse_rad_s"] = 0.5
        decision = harness.platform_decide(manifest, rows, reference)
        assert not decision["accepted"] and not decision["rule_met"]
        assert [b["corpus"] for b in decision["rule_breaches"]] == [entry["name"]]
        assert all(breach["gating"] for breach in decision["rule_breaches"])
        assert gates(decision) == {"comparator_final_step_rmse"}


def test_a_corpus_above_the_allowance_fails(manifest):
    rows = passing_rows(manifest)
    row = next(r for r in rows if r["corpus"] == "nanodrone")
    # Above the 0.696 allowance while still below every structured arm, so only
    # the allowance gate can trip.
    row["generic"]["final_step"]["velocity_rmse_m_s"] = 0.8
    row["structured"]["structured_residual"]["final_step"]["velocity_rmse_m_s"] = 0.9
    reference = reference_from(passing_rows(manifest))
    decision = harness.platform_decide(manifest, rows, reference)
    assert not decision["rule_met"] and gates(decision) == {"allowance_final_step_rmse"}
    breach = decision["rule_breaches"][0]
    assert breach["corpus"] == "nanodrone" and breach["limit"] == 0.696
    assert breach["gating"] is True and not decision["accepted"]


def test_a_corpus_with_no_allowance_is_only_read_against_the_comparator(manifest):
    rows = passing_rows(manifest)
    for name in ("idf", "epfl"):
        row = next(r for r in rows if r["corpus"] == name)
        row["generic"]["final_step"]["velocity_rmse_m_s"] = 0.15
    decision = harness.platform_decide(manifest, rows)
    assert decision["rule_met"]
    assert decision["final_step_rmse"]["idf"]["velocity_rmse_m_s"]["allowance"] is None


@pytest.mark.parametrize("problem", ["missing", "duplicate", "undeclared"])
def test_missing_duplicate_and_undeclared_corpora_fail_closed(manifest, problem):
    rows = passing_rows(manifest)
    expected = {
        "missing": "corpus_present",
        "duplicate": "corpus_unique",
        "undeclared": "corpus_declared",
    }[problem]
    if problem == "missing":
        rows = [row for row in rows if row["corpus"] != "idf"]
    elif problem == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    else:
        rows.append({**copy.deepcopy(rows[0]), "corpus": "crazyflie"})
    decision = harness.platform_decide(manifest, rows)
    assert not decision["accepted"] and not decision["rule_met"]
    assert expected in gates(decision)
    json.dumps(decision, allow_nan=False)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), None, "0.1"])
def test_nonfinite_scores_fail_closed_and_stay_serializable(manifest, value):
    rows = passing_rows(manifest)
    rows[0]["generic"]["final_step"]["velocity_rmse_m_s"] = value
    rows[1]["structured"][manifest["corpora"][1]["structured"]["arms"][0]][
        "final_step"
    ]["body_rate_rmse_rad_s"] = value
    decision = harness.platform_decide(manifest, rows)
    assert not decision["accepted"] and gates(decision) == {"finite_final_step_rmse"}
    first = manifest["corpora"][0]["name"]
    assert decision["final_step_rmse"][first]["velocity_rmse_m_s"]["generic"] is None
    json.dumps(decision, allow_nan=False)


def test_an_unfinished_corpus_fails_closed(manifest):
    rows = passing_rows(manifest)
    rows[0]["status"] = "failed"
    assert gates(harness.platform_decide(manifest, rows)) == {"fit_complete"}


@pytest.mark.parametrize("scored", [0, 2001, None, 12.5])
def test_an_evaluation_row_count_outside_the_budget_fails_closed(manifest, scored):
    rows = passing_rows(manifest)
    rows[0]["evaluation_rows"] = scored
    decision = harness.platform_decide(manifest, rows)
    assert not decision["accepted"] and "evaluation_rows" in gates(decision)
    json.dumps(decision, allow_nan=False)


def test_an_undeclared_structured_arm_set_fails_closed(manifest):
    rows = passing_rows(manifest)
    rows[0]["structured"]["extra_arm"] = one_score(
        {metric: 0.2 for metric in harness.METRICS}
    )
    assert gates(harness.platform_decide(manifest, rows)) == {"arms_declared"}


# --- the regression reference -----------------------------------------------


def reference_from(rows):
    """The reference a run of these rows would have been frozen from."""
    return dict(
        final_step_rmse={
            row["corpus"]: {
                metric: row["generic"]["final_step"][metric]
                for metric in harness.METRICS
            }
            for row in rows
        }
    )


def test_the_committed_platform_reference_carries_the_merged_numbers(manifest):
    """The reference is platform-v2's, forward unchanged but for its manifest id."""

    reference = harness.read(REFERENCE)
    assert reference["manifest"] == manifest["id"] == "glassbox-harness-platform-v3"
    assert reference["final_step_rmse"]["arp"]["body_rate_rmse_rad_s"] == pytest.approx(
        0.7154644403799479
    )
    assert reference["final_step_rmse"]["x8"]["velocity_rmse_m_s"] == pytest.approx(
        0.22234692231595657
    )


def test_the_committed_platform_reference_covers_every_corpus(manifest):
    reference = harness.read(REFERENCE)
    assert reference["manifest"] == manifest["id"]
    assert reference["manifest_sha256"] == harness.sha256(MANIFEST)
    recorded = reference["final_step_rmse"]
    assert sorted(recorded) == sorted(entry["name"] for entry in manifest["corpora"])
    for scores in recorded.values():
        assert sorted(scores) == sorted(harness.METRICS)
        assert all(harness._number(value) is not None for value in scores.values())


def test_a_platform_reference_regression_fails_closed(manifest):
    rows = passing_rows(manifest)
    reference = reference_from(copy.deepcopy(rows))
    allowance = manifest["reference"]
    base = reference["final_step_rmse"]["arp"]["body_rate_rmse_rad_s"]
    limit = (
        base * (1 + allowance["relative_tolerance"]) + allowance["absolute_tolerance"]
    )
    row = next(r for r in rows if r["corpus"] == "arp")
    row["generic"]["final_step"]["body_rate_rmse_rad_s"] = limit
    decision = harness.platform_decide(manifest, rows, reference, "a" * 64)
    assert decision["accepted"] and decision["reference_compared"]
    assert decision["reference_sha256"] == "a" * 64

    row["generic"]["final_step"]["body_rate_rmse_rad_s"] = limit * 1.000001
    decision = harness.platform_decide(manifest, rows, reference)
    assert not decision["accepted"] and decision["rule_met"]
    assert decision["gating_rule_breaches"] == 0
    regression = decision["reference_regressions"][0]
    assert regression["corpus"] == "arp" and regression["reference"] == base
    assert regression["gate"] == "reference_final_step_rmse"
    json.dumps(decision, allow_nan=False)


def test_a_platform_reference_without_the_corpus_fails_closed(manifest):
    rows = passing_rows(manifest)
    reference = reference_from(copy.deepcopy(rows))
    reference["final_step_rmse"].pop("idf")
    decision = harness.platform_decide(manifest, rows, reference)
    assert not decision["accepted"]
    assert {r["gate"] for r in decision["reference_regressions"]} == {
        "reference_present"
    }


@pytest.mark.parametrize("value", [float("nan"), float("inf"), None, "0.1", True])
def test_a_platform_reference_value_that_is_not_a_number_fails_closed(manifest, value):
    rows = passing_rows(manifest)
    reference = reference_from(copy.deepcopy(rows))
    reference["final_step_rmse"]["x8"]["velocity_rmse_m_s"] = value
    decision = harness.platform_decide(manifest, rows, reference)
    assert not decision["accepted"]
    assert decision["reference_regressions"][0] == dict(
        corpus="x8", metric="velocity_rmse_m_s", gate="reference_present"
    )
    json.dumps(decision, allow_nan=False)


PLATFORM_V2_SHA256 = "f4796e1a1bc2120aa0c00067853f13bf2004cf00ca9c5b34322daf0b7ce2f3ac"
"""The digest of the deleted platform-v2.json, whose protocol v3 carries."""

PLATFORM_V3_EDITS = (
    (
        '  "id": "glassbox-harness-platform-v3",\n',
        '  "id": "glassbox-harness-platform-v2",\n',
    ),
    (
        '    "gates_merges_from": "this manifest. A run is accepted only when no metric'
        " regresses past its reference value times 1.05 plus 0.005, the rule holds on"
        " every case the reference already meets it on, and nothing structural fails."
        " A case the reference already fails is reported, not gated: an enforced rule"
        " the incumbent does not meet would block every change rather than measure"
        ' one.",\n'
        '    "rule_met": "Reported separately from the decision, on every run. The'
        " accuracy row of docs/status.md is met only when the rule holds on every case,"
        ' which is a stronger statement than an accepted run.",\n',
        '    "gates_merges_from": "this manifest. The rule is a gate: a run is accepted'
        " only when every declared corpus is present once, fitted, and scored with"
        " finite numbers on rows inside the declared budget, and every corpus meets the"
        ' rule.",\n',
    ),
    (
        '    "gates_the_rule": "The same recorded value decides which cases the rule'
        " gates. A corpus and metric whose reference value is itself at or below this"
        " run's comparator and the declared allowance is a case the reference meets, and"
        " the rule is a gate there; a case the reference already fails is reported only."
        " With no reference compared there is no case the reference meets, and the rule"
        ' is reported everywhere."\n',
        "",
    ),
    (
        'believing it.",\n',
        'believing it."\n',
    ),
)


def test_the_committed_manifest_carries_platform_v2s_protocol_byte_for_byte():
    """Undoing the decision edits reproduces the deleted platform-v2 exactly.

    The corpora, the holdouts, the strides, the allowances and the arms cannot
    have moved, because reversing the id and the two decision blocks recovers
    v2's own digest from v3's bytes.
    """

    text = MANIFEST.read_text()
    for new_text, old_text in PLATFORM_V3_EDITS:
        assert text.count(new_text) == 1, new_text[:60]
        text = text.replace(new_text, old_text, 1)
    assert hashlib.sha256(text.encode()).hexdigest() == PLATFORM_V2_SHA256


# --- the frozen manifest ----------------------------------------------------


def test_the_platform_manifest_carries_the_recipe_and_the_frozen_plan(manifest):
    assert manifest["id"] == "glassbox-harness-platform-v3"
    # The manifest records the recipe it was frozen against in full and pins
    # the evaluation plan; it does not pin the candidate under measurement.
    assert set(manifest["recipe"]) == set(RECIPE)
    assert harness.declared_plan(manifest) == manifest["recipe"]
    assert manifest["decision"]["enforced"] is True
    assert manifest["reference"] == {
        "file": "platform-reference.json",
        "relative_tolerance": 0.05,
        "absolute_tolerance": 0.005,
        "meaning": manifest["reference"]["meaning"],
        "gates_the_rule": manifest["reference"]["gates_the_rule"],
    }
    assert manifest["evaluation_rows_maximum"] == 2000
    assert manifest["motor_history_s"] == 1.0
    contract = manifest["observed_contract"]
    assert contract["channels"] == list(harness.OBSERVED_CHANNELS)
    assert contract["velocity_channels"] == list(harness.VELOCITY_CHANNELS)
    assert contract["body_rate_channels"] == list(harness.BODY_RATE_CHANNELS)
    assert [entry["name"] for entry in manifest["corpora"]] == [
        "nanodrone",
        "x8",
        "arp",
        "idf",
        "epfl",
    ]
    allowances = {entry["name"]: entry["allowance"] for entry in manifest["corpora"]}
    assert allowances["nanodrone"] == {
        "velocity_rmse_m_s": 0.696,
        "body_rate_rmse_rad_s": 3.706,
    }
    assert allowances["x8"] == {
        "velocity_rmse_m_s": 1.601,
        "body_rate_rmse_rad_s": 0.764,
    }
    assert allowances["arp"] == {
        "velocity_rmse_m_s": 0.709,
        "body_rate_rmse_rad_s": 2.864,
    }
    assert allowances["idf"] is None and allowances["epfl"] is None


def test_every_corpus_declares_the_budget_the_recipe_resolves_on_its_grid(manifest):
    horizons = {}
    for entry in manifest["corpora"]:
        steps = steps_for(entry["sample_interval_s"])
        assert entry["forecast_horizon_s"] == RECIPE["horizon_s"]
        assert entry["information_budget"] == {
            "context_steps": steps["history"],
            "delay_steps": steps["delay"],
            "horizon_steps": steps["horizon"],
        }
        horizons[entry["name"]] = entry["realized_forecast_horizon_s"]
        assert horizons[entry["name"]] == pytest.approx(
            steps["horizon"] * entry["sample_interval_s"]
        )
        assert entry["evaluation_origin_stride_rows"] >= 1
        assert entry["structured"]["optimization_steps"] == 400
        assert entry["structured"]["holdout_count"] == 1
        assert set(entry["structured"]["arms"]) <= {
            "structured",
            "structured_residual",
        }
    assert horizons["nanodrone"] == pytest.approx(0.25)
    assert horizons["x8"] == pytest.approx(0.25)
    assert horizons["arp"] == pytest.approx(0.24)
    assert horizons["idf"] == pytest.approx(0.24)
    assert horizons["epfl"] == pytest.approx(0.2)


def test_the_manifest_digest_gates_platform_and_verify(tmp_path, manifest):
    assert harness.sha256(MANIFEST) == harness.PLATFORM_MANIFEST_SHA256
    assert MANIFEST == harness.COMMITTED_PLATFORM_MANIFEST
    altered = tmp_path / "platform-v3.json"
    tampered = copy.deepcopy(manifest)
    tampered["corpora"][0]["allowance"]["velocity_rmse_m_s"] = 99.0
    harness.write(altered, tampered)
    with pytest.raises(ValueError, match="manifest digest"):
        harness.frozen_platform_manifest(altered)
    with pytest.raises(ValueError, match="manifest digest"):
        harness.platform(altered, tmp_path / "corpora", tmp_path / "out")
    assert not (tmp_path / "out").exists()
    directory = tmp_path / "saved"
    directory.mkdir()
    harness.write(directory / "manifest.json", tampered)
    with pytest.raises(ValueError, match="manifest digest"):
        harness.verify(directory)


def test_verify_detects_which_tier_a_directory_holds(tmp_path, manifest):
    platform = tmp_path / "platform"
    platform.mkdir()
    shutil.copyfile(harness.COMMITTED_PLATFORM_MANIFEST, platform / "manifest.json")
    shutil.copyfile(harness.COMMITTED_PLATFORM_REFERENCE, platform / "reference.json")
    harness.write(platform / "results.json", [])
    harness.write(
        platform / "decision.json",
        seal_evidence(
            platform,
            harness.platform_decide(
                manifest,
                [],
                harness.read(REFERENCE),
                harness.sha256(REFERENCE),
            ),
            "platform",
            manifest,
        ),
    )
    replayed = harness.verify(platform)
    assert replayed["tier"] == "platform" and replayed["verified_corpora"] == 0
    assert not replayed["decision"]["accepted"]

    synthetic = tmp_path / "synthetic"
    synthetic.mkdir()
    shutil.copyfile(harness.COMMITTED_MANIFEST, synthetic / "manifest.json")
    shutil.copyfile(harness.COMMITTED_REFERENCE, synthetic / "reference.json")
    plan = harness.frozen_manifest(harness.COMMITTED_MANIFEST)
    harness.write(synthetic / "results.json", [])
    harness.write(
        synthetic / "decision.json",
        seal_evidence(
            synthetic,
            harness.decide(
                plan,
                [],
                harness.read(harness.COMMITTED_REFERENCE),
                harness.sha256(harness.COMMITTED_REFERENCE),
            ),
            "synthetic",
            plan,
        ),
    )
    replayed = harness.verify(synthetic)
    assert "tier" not in replayed and replayed["verified_cases"] == 0


# --- the replay -------------------------------------------------------------


def saved_platform_run(tmp_path, manifest):
    """A run directory whose one corpus holds one recorded artifact."""

    directory = tmp_path / "run"
    case = directory / "nanodrone"
    case.mkdir(parents=True)
    (case / "generic.npz").write_bytes(b"the bytes the run recorded")
    row = dict(
        corpus="nanodrone",
        status="complete",
        evaluation_rows=100,
        files={"generic.npz": harness.sha256(case / "generic.npz")},
    )
    harness.write(case / "result.json", row)
    harness.write(directory / "results.json", [row])
    shutil.copyfile(harness.COMMITTED_PLATFORM_MANIFEST, directory / "manifest.json")
    shutil.copyfile(REFERENCE, directory / "reference.json")
    harness.write(
        directory / "decision.json",
        harness.platform_decide(
            manifest, [row], harness.read(REFERENCE), harness.sha256(REFERENCE)
        ),
    )
    return directory, case, row


def test_verify_rejects_an_altered_platform_artifact(tmp_path, manifest):
    directory, case, _ = saved_platform_run(tmp_path, manifest)
    (case / "generic.npz").write_bytes(b"not the bytes the run recorded")
    with pytest.raises(ValueError, match="altered artifact"):
        harness.verify(directory)


def test_verify_rejects_a_case_result_edited_in_only_one_place(tmp_path, manifest):
    directory, case, row = saved_platform_run(tmp_path, manifest)
    harness.write(case / "result.json", {**row, "evaluation_rows": 99})
    with pytest.raises(ValueError, match="case result mismatch"):
        harness.verify(directory)


def test_numpy_replay_matches_the_jax_rollout_on_adapted_rows(flight):
    context, horizon, delay = 6, 3, 2
    adapted = adapt(flight)[0]
    batch = sequence_windows(
        adapted.states,
        adapted.inputs,
        np.arange(context, 40, 4),
        history_steps=context,
        horizon_steps=horizon,
        dt_s=DT_S,
    )
    model = initialize_sequence_model(batch, width=4, memory=3, delay_steps=delay)
    # An affine start reads the memory out as zero; give it weight so the
    # replayed filter has to agree, not merely cancel.
    model.params["linear"][-3:] = 0.15
    model.params["w2"][:] = 0.05
    x, up, uf = batch.past_states, batch.past_inputs, batch.future_inputs
    with jax.enable_x64(True):
        rollout = np.asarray(model.rollout(x, up, uf))
    replayed = harness.replay(model, x, up, uf)
    assert replayed.shape == rollout.shape == (len(x), horizon, 15)
    np.testing.assert_allclose(replayed, rollout, **harness.REPLAY_TOLERANCE)
    # The replay runs the filter, not just the explicit delay history: moving a
    # command outside that history still moves every forecast.
    moved = np.array(up)
    moved[:, context - delay - 1] += 5.0
    difference = np.abs(harness.replay(model, x, moved, uf) - replayed)
    assert difference.max() > 1e-6


# --- anchoring the platform reference ---------------------------------------


def scored_platform_run(tmp_path, manifest, *, reference=REFERENCE):
    """A run directory with no corpora, so a replay is only the anchoring check.

    Every gate that needs a fit is covered above on recorded rows. What this
    exercises is the part a saved run can lie about: the reference its own
    regression gate was measured against.
    """
    directory = tmp_path / "run"
    directory.mkdir(parents=True)
    shutil.copyfile(MANIFEST, directory / "manifest.json")
    harness.write(directory / "results.json", [])
    anchor, digest = None, None
    if reference is not None:
        shutil.copyfile(reference, directory / "reference.json")
        anchor, digest = harness.read(reference), harness.sha256(reference)
    harness.write(
        directory / "decision.json",
        seal_evidence(
            directory,
            harness.platform_decide(manifest, [], anchor, digest),
            "platform",
            manifest,
        ),
    )
    return directory


def loosen(value, factor=10):
    if isinstance(value, float):
        return value * factor
    if isinstance(value, dict):
        return {k: loosen(v, factor) for k, v in value.items()}
    if isinstance(value, list):
        return [loosen(v, factor) for v in value]
    return value


def test_the_committed_platform_reference_anchors_an_honest_replay(tmp_path, manifest):
    directory = scored_platform_run(tmp_path, manifest)
    result = harness.verify(directory)
    assert result["tier"] == "platform"
    assert result["decision"]["reference_compared"]
    assert result["decision"]["reference_sha256"] == harness.sha256(REFERENCE)
    assert harness.COMMITTED_PLATFORM_REFERENCE == REFERENCE


def test_a_platform_run_cannot_relax_its_own_regression_gate(tmp_path, manifest):
    """The defect: a replay must not read the threshold the run saved itself."""
    directory = scored_platform_run(tmp_path, manifest)
    copied = directory / "reference.json"
    harness.write(copied, loosen(harness.read(copied)))
    assert harness.sha256(copied) != harness.sha256(REFERENCE)
    with pytest.raises(ValueError, match="cannot relax its own regression gate"):
        harness.verify(directory)


def test_a_mismatched_committed_platform_reference_is_rejected(tmp_path, manifest):
    directory = scored_platform_run(tmp_path, manifest)
    other = tmp_path / "other-platform-reference.json"
    harness.write(other, loosen(harness.read(REFERENCE)))
    with pytest.raises(ValueError, match="cannot relax its own regression gate"):
        harness.verify(directory, other)


def test_a_platform_run_and_its_reference_must_agree_about_existing(tmp_path, manifest):
    compared = scored_platform_run(tmp_path, manifest)
    missing = tmp_path / "absent-platform-reference.json"
    with pytest.raises(ValueError, match="cannot be anchored"):
        harness.verify(compared, missing)
    uncompared = scored_platform_run(tmp_path / "second", manifest, reference=None)
    assert not harness.read(uncompared / "decision.json")["reference_compared"]
    with pytest.raises(ValueError, match="cannot be anchored"):
        harness.verify(uncompared)
    assert harness.verify(uncompared, missing)["decision"]["reference_sha256"] is None


def test_a_forged_platform_reference_digest_in_the_decision_is_rejected(
    tmp_path, manifest
):
    directory = scored_platform_run(tmp_path, manifest)
    decision = harness.read(directory / "decision.json")
    decision["reference_sha256"] = "0" * 64
    harness.write(directory / "decision.json", decision)
    with pytest.raises(ValueError, match="reference_sha256"):
        harness.verify(directory)


def test_a_platform_replay_recomputes_the_reference_regressions(tmp_path, manifest):
    """A saved decision that hid a regression does not survive the replay."""
    directory = scored_platform_run(tmp_path, manifest)
    decision = harness.read(directory / "decision.json")
    decision["reference_regressions"] = [dict(corpus="arp", gate="invented")]
    harness.write(directory / "decision.json", decision)
    with pytest.raises(ValueError, match="reference_regressions"):
        harness.verify(directory)
