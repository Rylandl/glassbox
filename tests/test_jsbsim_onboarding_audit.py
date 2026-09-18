"""Pure synthetic evidence tests: no JSBSim execution or model measurement."""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from glassbox.experimental.jsbsim_onboarding_audit import (
    ARRAY_NAMES,
    bitwise_equal,
    expected_branches,
    summarize,
    validate_case,
)


@pytest.fixture
def evidence():
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads((root / "docs/harness/jsbsim-onboarding-v1.json").read_text())
    entry = {"id": "example/plane.xml", "selected_initialization": "aircraft/example/reset00.xml"}
    bounds = np.array([[-1, 1]] * 3 + [[0, 1]], dtype=np.float64)
    initial = np.array([0.1, -0.2, 0, 0.6], dtype=np.float64)
    seed = int.from_bytes(hashlib.sha256(b"jsbsim-onboarding-v1:example/plane.xml").digest()[:8], "little")
    random = np.random.Generator(np.random.PCG64(seed))
    tape = np.repeat(np.clip(initial + .05 * (bounds[:, 1] - bounds[:, 0]) * random.uniform(-1, 1, (8, 4)),
                             bounds[:, 0], bounds[:, 1]), 5, axis=0)
    names = ["fcs/aileron-cmd-norm", "fcs/elevator-cmd-norm", "fcs/rudder-cmd-norm", "fcs/throttle-cmd-norm"]
    roster = expected_branches(4)
    case = {"id": entry["id"], "runs": {}, "branches": roster, "elapsed_s": 0.2}
    arrays = {}
    commands = {"parent": tape, "replay": tape.copy(), "factual": tape[:25].copy()}
    for branch in roster[1:]:
        copied = tape[:25].copy()
        column = branch["command_index"]
        bound = bounds[column, int(branch["direction"] == "upper")]
        copied[20:, column] += .1 * (bound - copied[20:, column])
        commands[branch["label"]] = copied
    for label, command in commands.items():
        n = len(command)
        native = np.zeros((n + 1, 16), dtype=np.float64)
        native[:, 0] = 20 + np.r_[0, np.cumsum(command[:, 0])]
        native[:, 4] = np.r_[0, np.cumsum(command[:, 1])]
        native[:, 15] = 500 + np.r_[0, np.cumsum(command[:, 3])]
        c, s = np.cos(.2), np.sin(.2)
        native[:, 6:15] = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]]).ravel()
        observed = native.copy()
        observed[:, :3] *= .3048
        observed[:, 6:15] = native[:, 6:15].reshape(-1, 3, 3).transpose(0, 2, 1).reshape(-1, 9)
        observed[:, 15] *= .3048
        data = dict(
            commands=command.copy(), observations=observed, native_observations=native,
            time_s=np.arange(n + 1, dtype=np.float64) * .05,
            immediate_readbacks=np.repeat(command[:, None, :], 6, axis=1),
            end_readbacks=np.repeat(command[:, None, :], 6, axis=1),
            native_attempted=np.ones((n, 6)), native_completed=np.ones((n, 6)),
            failed_native_observation=np.empty((0, 16)), failed_observation=np.empty((0, 16)),
            failed_time_s=np.empty(0),
        )
        arrays.update({f"{label}__{key}": value for key, value in data.items()})
        case["runs"][label] = dict(
            status="completed", stage="completed", error=None, command_names=names.copy(),
            bounds=bounds.tolist(), diagnostics=dict(engine_count=1, initial_commands=initial.tolist()),
            completed_intervals=n, failed_interval=None, failed_substep=None,
        )
    return case, arrays, protocol, entry


def _compute(evidence):
    return validate_case(*evidence)


def _set_raw(evidence, label, index, column, value):
    arrays = evidence[1]
    arrays[f"{label}__native_observations"][index, column] = value
    arrays[f"{label}__observations"][index, column] = value * (.3048 if column in (0, 1, 2, 15) else 1)


def _make_failed(evidence, label, completed=22):
    case, arrays, _, _ = evidence
    case["runs"][label].update(status="recording_failure", stage="run", error="run returned false",
                                 completed_intervals=completed, failed_interval=completed, failed_substep=2)
    for name in ("observations", "native_observations", "time_s"):
        arrays[f"{label}__{name}"] = arrays[f"{label}__{name}"][:completed + 1]
    for name in ("immediate_readbacks", "end_readbacks", "native_attempted", "native_completed"):
        arrays[f"{label}__{name}"] = arrays[f"{label}__{name}"][:completed + 1].copy()
    for name in ("immediate_readbacks", "end_readbacks"):
        arrays[f"{label}__{name}"][-1, 3:] = np.nan
    arrays[f"{label}__native_attempted"][-1, 3:] = 0
    arrays[f"{label}__native_completed"][-1, 2:] = 0


def test_complete_accounting_and_physical_activity(evidence):
    result = _compute(evidence)
    assert result["outcome"] == "completed"
    assert result["run_counts"] == dict(planned=11, attempted=11, completed=11, unattempted=0)
    assert result["branch_counts"] == dict(planned=9, attempted=9, completed=9, unattempted=0)
    assert result["parent_replay_equal"] and result["factual_equal"]
    assert result["active_commands"] == 3
    assert result["weak_commands"] == 1
    assert result["unavailable_commands"] == 0
    delta = evidence[1]["c000_lower__observations"][21:26, 0] - evidence[1]["factual__observations"][21:26, 0]
    assert result["activity"][0]["lower"]["velocity_m_s"] == np.abs(delta).tolist()
    assert result["activity"][2]["active"] is False


@pytest.mark.parametrize("mutate,match", [
    (lambda c, a: c["branches"][1].update(command_index=1), "branch roster"),
    (lambda c, a: c["branches"].pop(), "branch roster"),
    (lambda c, a: a["parent__commands"].__setitem__((0, 0), .8), "deterministic command"),
    (lambda c, a: a["c000_lower__commands"].__setitem__((20, 0), .8), "intended tape"),
    (lambda c, a: a["parent__observations"].__setitem__((0, 0), 123), "conversion"),
    (lambda c, a: a["parent__observations"].__setitem__((0, 7), 0), "conversion"),
    (lambda c, a: a["parent__time_s"].__setitem__(3, .17), "timing"),
    (lambda c, a: c["runs"]["parent"].update(completed_intervals=39), "interval count"),
    (lambda c, a: a["parent__native_completed"].__setitem__((1, 1), 0), "nonchronological"),
    (lambda c, a: a["parent__native_attempted"].__setitem__((0, 0), 2), "native masks"),
    (lambda c, a: c["runs"]["parent"]["command_names"].reverse(), "command names"),
    (lambda c, a: a.update(extra=np.ones(1)), "array roster"),
    (lambda c, a: a.update(parent__observations=a["parent__observations"].astype(np.float32)), "float64"),
])
def test_malformed_evidence_rejected(evidence, mutate, match):
    mutate(evidence[0], evidence[1])
    with pytest.raises(ValueError, match=match):
        _compute(evidence)


def test_replay_mismatch_is_retained_not_relaxed(evidence):
    _set_raw(evidence, "replay", 30, 0, 77)
    result = _compute(evidence)
    assert result["outcome"] == "replay_failure"
    assert result["parent_replay_equal"] is False
    # Independently matching response prefixes remain descriptive evidence.
    assert result["active_commands"] == 3


def test_prefix_mismatch_makes_affected_command_unavailable(evidence):
    _set_raw(evidence, "c000_upper", 10, 0, 77)
    result = _compute(evidence)
    assert result["outcome"] == "replay_failure"
    assert result["unavailable_commands"] == 1
    row = result["activity"][0]
    assert row["available"] is False and row["active"] is None
    assert row["lower"] is None and row["upper"] is None
    assert result["weak_commands"] == 1


def test_signed_zero_replay_mismatch_is_detected(evidence):
    evidence[1]["replay__native_observations"][0, 2] = -0.
    evidence[1]["replay__observations"][0, 2] = -0.
    assert _compute(evidence)["parent_replay_equal"] is False
    assert not bitwise_equal(np.array([0.]), np.array([-0.]))
    assert not bitwise_equal(np.array([1.], dtype=np.float32), np.array([1.], dtype=np.float64))


def test_failed_branch_is_valid_measured_unavailability(evidence):
    _make_failed(evidence, "c000_lower")
    result = _compute(evidence)
    assert result["outcome"] == "branch_failure"
    assert result["run_counts"]["completed"] == 10
    assert result["unavailable_commands"] == 1
    assert result["weak_commands"] == 1
    assert result["readbacks"]["c000_lower"]["attempted_native_substeps"] == 135
    assert result["readbacks"]["c000_lower"]["completed_native_substeps"] == 134


def test_failed_observation_preserves_invalid_snapshot(evidence):
    _make_failed(evidence, "c000_lower")
    case, arrays, _, _ = evidence
    case["runs"]["c000_lower"].update(stage="observation", failed_substep=5)
    arrays["c000_lower__native_attempted"][-1] = 1
    arrays["c000_lower__native_completed"][-1] = 1
    arrays["c000_lower__immediate_readbacks"][-1] = arrays["c000_lower__commands"][22]
    arrays["c000_lower__end_readbacks"][-1] = arrays["c000_lower__commands"][22]
    arrays["c000_lower__failed_native_observation"] = np.full((1, 16), np.nan)
    arrays["c000_lower__failed_observation"] = np.full((1, 16), np.nan)
    arrays["c000_lower__failed_time_s"] = np.array([1.15])
    assert _compute(evidence)["outcome"] == "branch_failure"


def test_finite_overwrites_counted_without_replacing_commands(evidence):
    for label in evidence[0]["runs"]:
        evidence[1][f"{label}__end_readbacks"][0, 0, 0] = .77
    result = _compute(evidence)
    assert result["outcome"] == "completed"
    assert result["readbacks"]["parent"]["end_overwrites"] == 1
    assert result["readbacks"]["parent"]["immediate_overwrites"] == 0


def test_nonfinite_readback_is_a_diagnostic_not_an_added_failure_gate(evidence):
    for label in evidence[0]["runs"]:
        evidence[1][f"{label}__end_readbacks"][0, 0, 0] = np.nan
    result = _compute(evidence)
    assert result["outcome"] == "completed"
    assert result["parent_replay_equal"] is True
    assert result["readbacks"]["parent"]["end_nonfinite_attempted_readbacks"] == 1
    assert result["readbacks"]["parent"]["end_overwrites"] == 0


def _setup_failure(evidence, unknown=False):
    case, arrays, _, _ = evidence
    for label in list(case["runs"]):
        if label == "parent":
            continue
        case["runs"][label] = {"status": "unattempted"}
        for name in ARRAY_NAMES:
            del arrays[f"{label}__{name}"]
    parent = case["runs"]["parent"]
    parent.update(status="missing_initialization", stage="initialization_selection", completed_intervals=0)
    parent["diagnostics"].pop("initial_commands")
    width = 4
    if unknown:
        parent.update(status="load_failure", stage="model_load", command_names=[], bounds=[], diagnostics={})
        case["branches"] = expected_branches(0)
        case["runs"] = {key: value for key, value in case["runs"].items() if key in ("parent", "replay", "factual")}
        width = 0
    for name in ARRAY_NAMES:
        old = arrays[f"parent__{name}"]
        arrays[f"parent__{name}"] = old[:0].copy()
    arrays["parent__commands"] = np.empty((0, width))
    for name in ("immediate_readbacks", "end_readbacks"):
        arrays[f"parent__{name}"] = np.empty((0, 6, width))


@pytest.mark.parametrize("unknown", [False, True])
def test_setup_failure_retains_planned_or_unknown_counts(evidence, unknown):
    _setup_failure(evidence, unknown)
    result = _compute(evidence)
    assert result["outcome"] == "setup_failure"
    assert result["command_count"] == (None if unknown else 4)
    assert result["run_counts"]["planned"] == (None if unknown else 11)
    assert result["unavailable_commands"] == (None if unknown else 4)
    assert result["active_commands"] == result["weak_commands"] == 0


def test_summary_preserves_empty_directories_root_template_and_failures(evidence):
    case = evidence[0]
    case["computed"] = _compute(evidence)
    template = copy.deepcopy(case)
    template["id"] = "aircraft_template.xml"
    timeout = {"id": "slow/a.xml", "execution_status": "timeout", "runs": {}, "branches": [],
               "timeout_s": 120, "returncode": None}
    timeout["computed"] = validate_case(timeout, {}, evidence[2], {"id": timeout["id"]})
    inventory = {"directories": ["example", "LM", "slow"],
                 "entries": [{"id": row["id"]} for row in (case, template, timeout)]}
    result = summarize([case, template, timeout], inventory)
    assert result["candidate_count"] == result["directory_count"] == 3
    assert result["directories"][1]["status"] == "no_config"
    assert result["root_candidates"][0]["id"] == "aircraft_template.xml"
    assert result["outcome_counts"] == {"completed": 2, "timeout": 1}
    assert result["command_activity"]["unknown_command_count_cases"] == 1
    assert result["run_counts"]["unknown_planned_cases"] == 1
    with pytest.raises(ValueError, match="inventory accounting"):
        summarize([case, case, timeout], inventory)


def test_saved_computed_is_not_trusted(evidence):
    expected = _compute(evidence)
    evidence[0]["computed"] = {"outcome": "forged", "active_commands": 999}
    assert _compute(evidence) == expected


def test_coherent_trajectory_change_requires_fresh_simulator_replay(evidence):
    # Pure arrays cannot authenticate physics. This deliberately remains internally
    # consistent and root's fresh pinned-simulator replay must reject the change.
    for label in evidence[0]["runs"]:
        raw = evidence[1][f"{label}__native_observations"]
        raw[:, 0] += 1
        evidence[1][f"{label}__observations"][:, 0] = raw[:, 0] * .3048
    assert _compute(evidence)["outcome"] == "completed"


def _interrupt_at(evidence, label, *, observations=11, metadata_only=False):
    case, arrays, _, _ = evidence
    labels = list(case["runs"])
    for later in labels[labels.index(label) + 1:]:
        case["runs"][later] = {"status": "unattempted"}
        for name in ARRAY_NAMES:
            del arrays[f"{later}__{name}"]
    case.update(execution_status="timeout", timeout_s=120, returncode=None)
    if metadata_only:
        case["runs"][label] = {"status": "running", "stage": "construct"}
        for name in ARRAY_NAMES:
            del arrays[f"{label}__{name}"]
        if label == "parent":
            case["branches"] = []
            case["runs"] = {"parent": case["runs"]["parent"]}
        return
    completed = observations - 1
    case["runs"][label].update(status="running", stage="observation", completed_intervals=completed,
                                 failed_interval=completed - 1, failed_substep=5)
    for name in ("observations", "native_observations", "time_s"):
        arrays[f"{label}__{name}"] = arrays[f"{label}__{name}"][:observations]
    for name in ("immediate_readbacks", "end_readbacks", "native_attempted", "native_completed"):
        arrays[f"{label}__{name}"] = arrays[f"{label}__{name}"][:completed]


@pytest.mark.parametrize("label", ["parent", "replay", "factual", "c003_upper"])
def test_interrupted_boundary_preserves_evidence_and_roster(evidence, label):
    _interrupt_at(evidence, label)
    result = _compute(evidence)
    assert result["outcome"] == "timeout"
    assert result["run_counts"]["planned"] == 11
    assert result["run_counts"]["attempted"] + result["run_counts"]["unattempted"] == 11
    assert result["active_commands"] == result["weak_commands"] == 0
    assert result["unavailable_commands"] == 4
    assert result["readbacks"][label]["completed_native_substeps"] == 60


@pytest.mark.parametrize("label", ["parent", "replay", "c003_upper"])
def test_interrupted_constructor_without_snapshot_is_explicit(evidence, label):
    _interrupt_at(evidence, label, metadata_only=True)
    result = _compute(evidence)
    assert result["outcome"] == "timeout"
    assert result["command_count"] == (None if label == "parent" else 4)
    assert result["run_counts"]["attempted"] >= 1


def test_nonterminal_evidence_cannot_masquerade_as_completed_execution(evidence):
    _interrupt_at(evidence, "c003_upper")
    del evidence[0]["execution_status"]
    with pytest.raises(ValueError, match="nonterminal"):
        _compute(evidence)


def test_unknown_failure_preserves_supplied_branch_tape(evidence):
    case, arrays, _, _ = evidence
    label = "c000_lower"
    tape = arrays[f"{label}__commands"].copy()
    case["runs"][label] = dict(status="load_failure", stage="model_load", command_names=[], bounds=[],
                                diagnostics={}, completed_intervals=0, failed_interval=None, failed_substep=None)
    for name in ARRAY_NAMES:
        arrays[f"{label}__{name}"] = arrays[f"{label}__{name}"][:0].copy()
    arrays[f"{label}__commands"] = tape
    for name in ("immediate_readbacks", "end_readbacks"):
        arrays[f"{label}__{name}"] = np.empty((0, 6, 0))
    result = _compute(evidence)
    assert result["outcome"] == "branch_failure"
    assert result["unavailable_commands"] == 1


def test_bad_command_is_rejected_even_in_interrupted_case(evidence):
    _interrupt_at(evidence, "parent")
    evidence[1]["parent__commands"][0, 0] += .001
    with pytest.raises(ValueError, match="deterministic command"):
        _compute(evidence)


def test_unknown_execution_cannot_hide_full_evidence(evidence):
    evidence[0].update(execution_status="timeout", timeout_s=120, returncode=0)
    with pytest.raises(ValueError, match="timeout cannot"):
        _compute(evidence)
