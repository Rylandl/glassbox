"""Excitation runner boundaries with fake workers, never a simulator trial."""

import copy
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from glassbox.experimental import jsbsim_excitation as harness
from glassbox.experimental import jsbsim_onboarding as onboarding
from glassbox.experimental import jsbsim_onboarding_audit as onboarding_audit


@pytest.fixture
def inputs():
    protocol = json.loads(
        (onboarding.REPO / "docs/harness/jsbsim-excitation-v1.json").read_text()
    )
    inherited, inventory = onboarding.load_spec()
    entry = next(row for row in inventory["entries"] if row["id"] == "ball/ball.xml")
    bounds = np.tile([-1.0, 1.0], (3, 1))
    parent = {
        "status": "completed",
        "command_names": list(onboarding_audit.SURFACES),
        "bounds": bounds.tolist(),
        "diagnostics": {"initial_commands": [0.0] * 3, "engine_count": 0},
    }
    reference_case = {"id": entry["id"], "runs": {"parent": parent}}
    reference_arrays = {
        "parent__commands": onboarding_audit.expected_tape(
            inherited, entry, np.zeros(3), bounds
        )
    }
    return protocol, inherited, entry, reference_case, reference_arrays


@pytest.fixture
def boundaries(monkeypatch):
    audit = ModuleType("glassbox.experimental.jsbsim_excitation_audit")
    audited = []

    def validate_arm(case, arrays, *args, **kwargs):
        audited.append(copy.deepcopy(case))
        return {"mock_audit": True}

    audit.validate_arm = validate_arm
    audit.summarize = lambda cases, *args, **kwargs: {"cases": cases}
    adapter = ModuleType("glassbox.experimental.jsbsim_excitation_adapter")
    monkeypatch.setitem(sys.modules, audit.__name__, audit)
    monkeypatch.setitem(sys.modules, adapter.__name__, adapter)
    return adapter, audit, audited


def _fake_result(commands, completed, *, status="completed"):
    observations = np.zeros((completed + 1, 16), dtype=np.float64)
    observations[:, 6:15] = np.eye(3).reshape(9)
    return {
        "status": status,
        "stage": "completed" if status == "completed" else "observation",
        "completed_intervals": completed,
        "command_names": list(onboarding_audit.SURFACES),
        "bounds": np.tile([-1.0, 1.0], (3, 1)).tolist(),
        "diagnostics": {"initial_commands": [0.0] * 3, "engine_count": 0},
        "arrays": {
            "commands": commands.copy(),
            "observations": observations,
            "native_observations": observations.copy(),
            "time_s": np.arange(completed + 1, dtype=np.float64) * 0.05,
        },
    }


def test_common_tape_extends_the_pinned_baseline_without_changing_old_rows(inputs):
    protocol, inherited, entry, reference_case, reference_arrays = inputs
    tape = harness.common_tape(
        entry, protocol, inherited, reference_case, reference_arrays
    )
    assert tape.shape == (120, 3)
    assert onboarding_audit.bitwise_equal(tape[:40], reference_arrays["parent__commands"])
    expected = onboarding_audit.expected_tape(
        {**inherited, "recording": {**inherited["recording"], "transitions": 120}},
        entry, np.zeros(3), np.tile([-1.0, 1.0], (3, 1)),
    )
    assert onboarding_audit.bitwise_equal(tape, expected)


def test_common_tape_rejects_changed_reference_commands(inputs):
    protocol, inherited, entry, reference_case, reference_arrays = inputs
    reference_arrays["parent__commands"][3, 1] += 0.01
    with pytest.raises(ValueError):
        harness.common_tape(entry, protocol, inherited, reference_case, reference_arrays)


def test_unknown_reference_baseline_is_not_invented(inputs):
    protocol, inherited, entry, reference_case, reference_arrays = inputs
    reference_case["runs"]["parent"]["diagnostics"].pop("initial_commands")
    reference_arrays["parent__commands"] = np.empty((0, 3), dtype=np.float64)
    assert harness.common_tape(
        entry, protocol, inherited, reference_case, reference_arrays
    ) is None


@pytest.mark.parametrize("completed", [19, 20, 29])
def test_branch_admission_depends_on_origin_prefix_not_terminal_parent_status(
    tmp_path, inputs, boundaries, completed
):
    protocol, inherited, entry, reference_case, reference_arrays = inputs
    adapter, _, audited = boundaries
    tapes = []

    def simulate(*args, **kwargs):
        command = next(value for value in args if isinstance(value, np.ndarray))
        tapes.append(command.copy())
        return _fake_result(
            command, completed if len(tapes) == 1 else 120,
            status="recording_failure" if len(tapes) == 1 else "completed",
        )

    adapter.simulate = simulate
    case, arrays = harness.execute_case(
        tmp_path, entry, "as_shipped", protocol, inherited,
        reference_case, reference_arrays,
    )
    assert case["runs"]["parent"]["status"] == "recording_failure"
    assert len(arrays["parent__observations"]) == completed + 1
    assert len(case["branches"]) == 7
    branch_labels = [row["label"] for row in case["branches"]]
    if completed < 20:
        assert all(case["runs"][label]["status"] == "unattempted" for label in branch_labels)
    else:
        assert len(tapes) == 9
        assert all(case["runs"][label]["status"] == "completed" for label in branch_labels)
    assert audited[-1]["runs"]["parent"]["status"] == "recording_failure"


def test_both_arms_receive_identical_tapes_with_one_sustained_signed_intervention(
    tmp_path, inputs, boundaries
):
    protocol, inherited, entry, reference_case, reference_arrays = inputs
    adapter, _, _ = boundaries
    arm_tapes = {}

    for arm in protocol["arms"]:
        captured = []

        def simulate(*args, captured=captured, arm=arm, **kwargs):
            command = next(value for value in args if isinstance(value, np.ndarray))
            captured.append(command.copy())
            result = _fake_result(command, 120)
            # A changed bootstrap baseline must not regenerate commands.
            if arm == "engine_bootstrap":
                result["diagnostics"]["initial_commands"] = [0.25] * 3
            return result

        adapter.simulate = simulate
        harness.execute_case(
            tmp_path, entry, arm, protocol, inherited, reference_case, reference_arrays
        )
        arm_tapes[arm] = captured

    assert len(arm_tapes["as_shipped"]) == len(arm_tapes["engine_bootstrap"]) == 9
    for shipped, bootstrapped in zip(
        arm_tapes["as_shipped"], arm_tapes["engine_bootstrap"], strict=True
    ):
        assert onboarding_audit.bitwise_equal(shipped, bootstrapped)
    tapes = arm_tapes["as_shipped"]
    for tape in tapes[:3]:
        assert onboarding_audit.bitwise_equal(tape, tapes[0])
    for channel in range(3):
        for direction, bound in enumerate([-1.0, 1.0]):
            actual = tapes[3 + 2 * channel + direction]
            expected = tapes[0].copy()
            expected[20:, channel] += 0.1 * (bound - expected[20:, channel])
            assert onboarding_audit.bitwise_equal(actual, expected)
            assert actual.shape == (120, 3)


def test_adapter_programming_errors_abort_instead_of_becoming_arm_outcomes(
    tmp_path, inputs, boundaries
):
    protocol, inherited, entry, reference_case, reference_arrays = inputs

    def broken(*args, **kwargs):
        raise KeyError("deliberate harness bug")

    boundaries[0].simulate = broken
    with pytest.raises(KeyError, match="deliberate harness bug"):
        harness.execute_case(
            tmp_path, entry, "as_shipped", protocol, inherited,
            reference_case, reference_arrays,
        )


@pytest.fixture
def worker_boundary(inputs, boundaries, monkeypatch):
    protocol, inherited, entry, reference_case, reference_arrays = inputs
    _, inventory = onboarding.load_spec()
    inventory = {**inventory, "entries": [entry]}
    monkeypatch.setattr(harness, "load_spec", lambda: (protocol, inherited, inventory))
    monkeypatch.setattr(onboarding, "check_assets", lambda *args: None)
    monkeypatch.setattr(harness, "runtime_identity", lambda: {"mocked_runtime": True})
    monkeypatch.setattr(harness, "validate_reference", lambda *args: None)
    monkeypatch.setattr(
        harness, "reference_case_at", lambda *args: (reference_case, reference_arrays)
    )
    return protocol, entry


@pytest.mark.parametrize("outcome", ["timeout", "worker_failure"])
def test_interrupted_arm_keeps_checkpoint_and_does_not_prevent_other_arm(
    tmp_path, worker_boundary, boundaries, monkeypatch, outcome
):
    protocol, entry = worker_boundary
    _, audit, _ = boundaries
    branches = onboarding.branch_roster(3)
    saved_arrays = {"parent__observations": np.array([[0.0, -0.0, np.nan]])}
    started_arms = []
    breadcrumb = {"label": "c000_lower", "stage": "run", "failed_interval": 33,
                  "failed_substep": 2, "progress_only": True}

    def worker(command, **kwargs):
        arm = command[command.index("--arm") + 1]
        started_arms.append(arm)
        assert kwargs["timeout"] == 180
        assert Path(kwargs["cwd"]).is_dir()
        directory = Path(command[command.index("--out") + 1])
        directory.mkdir()
        case = {"id": entry["id"], "arm": arm, "branches": branches,
                "runs": {"parent": {"status": "completed"},
                         "c000_lower": {"status": "running"}}}
        if arm == "as_shipped":
            onboarding.checkpoint(directory, case, saved_arrays)
            onboarding.write_json(directory / "progress.json", breadcrumb)
            (directory / "checkpoint.tmp").write_bytes(b"unfinished next checkpoint")
            if outcome == "timeout":
                raise subprocess.TimeoutExpired(command, kwargs["timeout"])
            return SimpleNamespace(returncode=-11)
        case["computed"] = audit.validate_arm(case, saved_arrays)
        harness._finish_case(directory, case, saved_arrays)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(harness.subprocess, "run", worker)
    output = tmp_path / "evidence"
    harness.run_all(tmp_path / "assets", output, tmp_path / "reference", workers=1)
    assert started_arms == protocol["arms"]
    interrupted = harness.case_path(output, entry["id"], "as_shipped")
    case, arrays = onboarding.read_case(interrupted)
    assert case["execution_status"] == outcome
    assert case["returncode"] == (None if outcome == "timeout" else -11)
    assert case["unsaved_tail"] == "unknown"
    assert case["branches"] == branches
    assert case["interrupted_progress"] == breadcrumb
    assert onboarding_audit.bitwise_equal(
        arrays["parent__observations"], saved_arrays["parent__observations"]
    )
    sibling, _ = onboarding.read_case(harness.case_path(output, entry["id"], "engine_bootstrap"))
    assert "execution_status" not in sibling
    assert (output / "run.json").is_file()
    assert not any(interrupted.glob("checkpoint*"))

    def reject_interrupted(case, *args, **kwargs):
        if "execution_status" in case:
            raise ValueError("interrupted evidence must still be audited")
        return {"mock_audit": True}

    monkeypatch.setattr(audit, "validate_arm", reject_interrupted)
    with pytest.raises(ValueError, match="interrupted evidence must still be audited"):
        harness.validate_bundle(tmp_path / "assets", output, tmp_path / "reference")


@pytest.mark.parametrize("code", [0, 1])
def test_bad_worker_exit_aborts_before_report_or_seal(
    tmp_path, worker_boundary, monkeypatch, code
):
    def worker(command, **kwargs):
        if code:
            directory = Path(command[command.index("--out") + 1])
            directory.mkdir()
            onboarding.write_json(directory / "case.json", {"execution_status": "timeout"})
        return SimpleNamespace(returncode=code)

    monkeypatch.setattr(harness.subprocess, "run", worker)
    output = tmp_path / "evidence"
    with pytest.raises(RuntimeError, match="harness worker failed"):
        harness.run_all(tmp_path, output, tmp_path / "reference", workers=1)
    assert not (output / "report.json").exists()
    assert not (output / "run.json").exists()


@pytest.fixture
def reference_tree(tmp_path):
    reference = tmp_path / "reference"
    reference.mkdir()
    onboarding.write_json(reference / "protocol.json", {"id": "historical"})
    onboarding.write_json(reference / "inventory.json", {"entries": []})
    identity = {
        "python": "pinned interpreter", "python_binary_sha256": "python hash",
        "numpy": "pinned numpy", "numpy_native_sha256": "numpy native hash",
        "jsbsim": "1.3.1", "jsbsim_files": {"module.so": "native hash"},
        "platform": "pinned platform", "machine": "pinned machine",
        "dispersion": "0", "simulator_seed": 0,
        "inventory_sha256": onboarding.digest(reference / "inventory.json"),
    }
    inherited_sources = {"src/frozen.py": "frozen source hash"}
    onboarding.write_json(
        reference / "runtime.json", {**identity, "source_files": inherited_sources}
    )
    seal = {
        "protocol_sha256": onboarding.digest(reference / "protocol.json"),
        "inventory_sha256": identity["inventory_sha256"],
        "files": harness._sealed_files(reference),
    }
    onboarding.write_json(reference / "run.json", seal)
    protocol = {"reference": {
        "protocol_sha256": seal["protocol_sha256"],
        "inventory_sha256": seal["inventory_sha256"],
        "run_sha256": onboarding.digest(reference / "run.json"),
        "inherited_source_sha256": inherited_sources,
    }}
    return reference, protocol, identity


def test_reference_payload_mutation_cannot_hide_behind_original_seal(reference_tree):
    reference, protocol, identity = reference_tree
    harness.validate_reference(reference, protocol, identity)
    onboarding.write_json(reference / "runtime.json", {"tampered": True})
    with pytest.raises(ValueError, match="old reference payload changed"):
        harness.validate_reference(reference, protocol, identity)


def test_rehashed_reference_still_requires_frozen_external_anchor(reference_tree):
    reference, protocol, identity = reference_tree
    onboarding.write_json(reference / "extra.json", {"added": True})
    seal = json.loads((reference / "run.json").read_text())
    seal["files"] = harness._sealed_files(reference)
    onboarding.write_json(reference / "run.json", seal)
    with pytest.raises(ValueError, match="old reference external seal changed"):
        harness.validate_reference(reference, protocol, identity)


def test_reference_source_roster_is_independently_pinned(reference_tree):
    reference, protocol, identity = reference_tree
    runtime = json.loads((reference / "runtime.json").read_text())
    runtime["source_files"]["src/frozen.py"] = "changed source hash"
    onboarding.write_json(reference / "runtime.json", runtime)
    seal = json.loads((reference / "run.json").read_text())
    seal["files"] = harness._sealed_files(reference)
    onboarding.write_json(reference / "run.json", seal)
    # Even a caller supplying this new seal cannot change inherited source pins.
    protocol["reference"]["run_sha256"] = onboarding.digest(reference / "run.json")
    with pytest.raises(ValueError, match="old reference source pins changed"):
        harness.validate_reference(reference, protocol, identity)


def test_current_runtime_must_match_the_reference_native_runtime(reference_tree):
    reference, protocol, identity = reference_tree
    identity["numpy_native_sha256"] = "different native build"
    with pytest.raises(ValueError, match="old reference runtime differs: numpy_native"):
        harness.validate_reference(reference, protocol, identity)


def test_changed_inherited_source_is_rejected_before_running(monkeypatch):
    actual_digest = onboarding.digest

    def changed_digest(path):
        if Path(path).name == "jsbsim_adapter.py":
            return "0" * 64
        return actual_digest(path)

    monkeypatch.setattr(onboarding, "digest", changed_digest)
    with pytest.raises(ValueError, match="inherited source changed"):
        harness.load_spec()


@pytest.mark.parametrize("trusted", [None, "0" * 64])
def test_verify_rejects_untrusted_seal_before_validation_or_simulation(
    tmp_path, monkeypatch, trusted
):
    onboarding.write_json(tmp_path / "run.json", {"files": {}})

    def forbidden(*args, **kwargs):
        pytest.fail("untrusted evidence cannot trigger validation or simulation")

    monkeypatch.setattr(harness, "validate_bundle", forbidden)
    monkeypatch.setattr(harness, "run_all", forbidden)
    with pytest.raises(ValueError, match="trusted external run seal"):
        harness.verify(tmp_path, tmp_path, tmp_path, expected_run_sha256=trusted)


@pytest.mark.parametrize("alteration", ["signed_zero", "dtype", "telemetry"])
def test_fresh_replay_rejects_coherent_array_alterations(tmp_path, alteration):
    saved_root, fresh_root = tmp_path / "saved", tmp_path / "fresh"
    case = {"id": "ball/ball.xml", "arm": "as_shipped", "computed": {"available": True}}
    original = {"parent__observations": np.array([0.0, 1.0]),
                "parent__telemetry": np.array([0.0, 2.0])}
    changed = {key: value.copy() for key, value in original.items()}
    if alteration == "signed_zero":
        changed["parent__observations"][0] = -0.0
    elif alteration == "dtype":
        changed["parent__observations"] = changed["parent__observations"].astype(np.float32)
    else:
        changed["parent__telemetry"][1] += 1.0
    for root, arrays in ((saved_root, changed), (fresh_root, original)):
        directory = harness.case_path(root, case["id"], case["arm"])
        directory.mkdir(parents=True)
        harness._finish_case(directory, case, arrays)
    with pytest.raises(ValueError, match="fresh simulator replay differs"):
        harness.compare_fresh(saved_root, fresh_root, [case], [case])


@pytest.mark.parametrize("fresh_status", [None, "worker_failure", "timeout"])
def test_timeout_replay_never_qualifies_physics_or_integrity(tmp_path, fresh_status):
    saved = {"id": "ball/ball.xml", "arm": "as_shipped", "execution_status": "timeout"}
    fresh = {"id": saved["id"], "arm": saved["arm"]}
    if fresh_status is not None:
        fresh["execution_status"] = fresh_status
    if fresh_status != "timeout":
        with pytest.raises(ValueError, match="execution outcome changed"):
            harness.compare_fresh(tmp_path, tmp_path, [saved], [fresh])
    else:
        result = harness.compare_fresh(tmp_path, tmp_path, [saved], [fresh])
        assert result["integrity_verified"] is False
        assert result["cases"][0]["exact_recorded_outcome"] is False
        assert result["model_qualification"] is False
        assert result["flight_qualification"] is False
