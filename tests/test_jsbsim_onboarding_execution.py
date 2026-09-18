"""Runner failure boundaries with mocked workers; no simulator is imported."""

import copy
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from glassbox.experimental import jsbsim_onboarding as harness


@pytest.fixture
def audit_boundary(monkeypatch):
    """Isolate process orchestration from the separately tested metric audit."""
    audit = ModuleType("glassbox.experimental.jsbsim_onboarding_audit")
    audited = []

    def validate_case(case, arrays, protocol, entry):
        audited.append(copy.deepcopy(case))
        return {"execution_status": case.get("execution_status"), "arrays": sorted(arrays)}

    def summarize(cases, inventory):
        return {
            "cases": {case["id"]: case.get("execution_status") for case in cases},
            "candidate_count": len(inventory["entries"]),
        }

    audit.validate_case = validate_case
    audit.summarize = summarize
    audit.bitwise_equal = lambda left, right: (
        left.dtype == right.dtype
        and left.shape == right.shape
        and left.tobytes() == right.tobytes()
    )
    monkeypatch.setitem(sys.modules, audit.__name__, audit)
    return audit, audited


@pytest.fixture
def runner_boundary(monkeypatch, audit_boundary):
    protocol, inventory = harness.load_spec()
    inventory = {**inventory, "entries": inventory["entries"][:1]}
    monkeypatch.setattr(harness, "load_spec", lambda: (protocol, inventory))
    monkeypatch.setattr(harness, "check_assets", lambda root, entries: None)
    monkeypatch.setattr(harness, "runtime_identity", lambda: {"mocked_runtime": True})
    return protocol, inventory["entries"][0]


def _partial_case(entry):
    roster = harness.branch_roster(1)
    case = {
        "id": entry["id"],
        "branches": roster,
        "runs": {
            "parent": {"status": "completed", "command_names": ["command"]},
            "replay": {"status": "completed"},
            "factual": {"status": "completed"},
            "c000_lower": {"status": "running", "completed_intervals": 2},
            "c000_upper": {"status": "unattempted"},
        },
    }
    arrays = {
        "parent__observations": np.arange(12, dtype=np.float64).reshape(4, 3),
        "c000_lower__observations": np.array([[0.0, -0.0, 2.0]], dtype=np.float64),
    }
    return case, arrays


@pytest.mark.parametrize("outcome", ["timeout", "worker_failure"])
def test_interrupted_worker_retains_last_atomic_checkpoint(
    tmp_path, monkeypatch, runner_boundary, audit_boundary, outcome
):
    protocol, entry = runner_boundary
    saved_case, saved_arrays = _partial_case(entry)
    progress = {
        "label": "c000_lower", "progress_only": True, "stage": "run",
        "failed_interval": 2, "failed_substep": 3,
    }

    def worker(command, **kwargs):
        assert kwargs["timeout"] == protocol["recording"]["case_timeout_s"]
        directory = Path(command[command.index("--out") + 1])
        directory.mkdir()
        harness.checkpoint(directory, saved_case, saved_arrays)
        harness.write_json(directory / "progress.json", progress)
        # A kill during the next write must not invalidate the last full snapshot.
        (directory / "checkpoint.tmp").write_bytes(b"incomplete next archive")
        if outcome == "timeout":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return SimpleNamespace(returncode=-11)

    monkeypatch.setattr(harness.subprocess, "run", worker)
    output = tmp_path / "evidence"
    report = harness.execute(tmp_path / "assets", output, workers=1)
    case, arrays = harness.read_case(output / "cases" / harness.case_slug(entry["id"]))
    assert case["execution_status"] == outcome
    assert case["returncode"] == (None if outcome == "timeout" else -11)
    assert case["runs"] == saved_case["runs"]
    assert case["branches"] == saved_case["branches"]
    assert case["interrupted_progress"] == progress
    assert set(arrays) == set(saved_arrays)
    for name, expected in saved_arrays.items():
        assert arrays[name].dtype == expected.dtype
        assert arrays[name].shape == expected.shape
        assert arrays[name].tobytes() == expected.tobytes()
    assert audit_boundary[1][0]["execution_status"] == outcome
    assert report["cases"][entry["id"]] == outcome
    assert (output / "run.json").exists()


@pytest.mark.parametrize("returncode", [0, 1])
def test_python_worker_failure_aborts_without_sealing(
    tmp_path, monkeypatch, runner_boundary, returncode
):
    def worker(command, **kwargs):
        # A positive Python exit is fatal even if it leaves a plausible result.
        if returncode:
            directory = Path(command[command.index("--out") + 1])
            directory.mkdir()
            harness.write_json(directory / "case.json", {"execution_status": "timeout"})
        return SimpleNamespace(returncode=returncode)

    monkeypatch.setattr(harness.subprocess, "run", worker)
    output = tmp_path / "evidence"
    with pytest.raises(RuntimeError, match="harness worker failed"):
        harness.execute(tmp_path / "assets", output, workers=1)
    assert not (output / "report.json").exists()
    assert not (output / "run.json").exists()


def test_execution_status_does_not_bypass_case_validation(
    tmp_path, monkeypatch, runner_boundary, audit_boundary
):
    def worker(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(harness.subprocess, "run", worker)
    output = tmp_path / "evidence"
    root = tmp_path / "assets"
    harness.execute(root, output, workers=1)

    def reject_case(case, arrays, protocol, entry):
        assert case["execution_status"] == "timeout"
        raise ValueError("malformed interrupted case")

    monkeypatch.setattr(audit_boundary[0], "validate_case", reject_case)
    with pytest.raises(ValueError, match="malformed interrupted case"):
        harness.validate_bundle(root, output)


@pytest.mark.parametrize("seal", [None, "0" * 64])
def test_verify_requires_external_seal_before_validation_or_replay(
    tmp_path, monkeypatch, audit_boundary, seal
):
    harness.write_json(tmp_path / "run.json", {"files": {}})

    def unexpected(*args, **kwargs):
        pytest.fail("an untrusted bundle must not trigger validation or a fresh run")

    monkeypatch.setattr(harness, "validate_bundle", unexpected)
    monkeypatch.setattr(harness, "execute", unexpected)
    with pytest.raises(ValueError, match="trusted external run seal"):
        harness.verify(tmp_path, tmp_path, expected_run_sha256=seal)


def test_rehashed_mutation_does_not_replace_the_external_anchor(
    tmp_path, monkeypatch, audit_boundary
):
    seal_path = tmp_path / "run.json"
    harness.write_json(seal_path, {"files": {"case.json": "original hash"}})
    trusted_sha = harness.digest(seal_path)
    # The attacker may rewrite every internal hash, but not the published digest.
    seal = json.loads(seal_path.read_text())
    seal["files"]["case.json"] = "coherently rehashed altered outcome"
    harness.write_json(seal_path, seal)
    monkeypatch.setattr(
        harness, "validate_bundle", lambda *args: pytest.fail("seal must reject first")
    )
    with pytest.raises(ValueError, match="trusted external run seal"):
        harness.verify(tmp_path, tmp_path, expected_run_sha256=trusted_sha)


@pytest.mark.parametrize(
    ("original_status", "fresh_status"),
    [("timeout", None), (None, "timeout"), ("timeout", "worker_failure")],
)
def test_changed_execution_outcome_cannot_verify(
    tmp_path, monkeypatch, audit_boundary, original_status, fresh_status
):
    harness.write_json(tmp_path / "run.json", {"files": {}})

    def cases(root, output):
        status = original_status if output == tmp_path else fresh_status
        case = {"id": "demo/demo.xml"}
        if status is not None:
            case["execution_status"] = status
        return [case]

    monkeypatch.setattr(harness, "validate_bundle", cases)
    monkeypatch.setattr(harness, "execute", lambda *args: None)
    with pytest.raises(ValueError, match="execution outcome changed"):
        harness.verify(
            tmp_path, tmp_path, expected_run_sha256=harness.digest(tmp_path / "run.json")
        )


@pytest.mark.parametrize("status", ["timeout", "worker_failure"])
def test_repeated_execution_failure_is_not_integrity_or_physics_evidence(
    tmp_path, monkeypatch, audit_boundary, status
):
    harness.write_json(tmp_path / "run.json", {"files": {}})
    monkeypatch.setattr(
        harness, "validate_bundle",
        lambda *args: [{"id": "demo/demo.xml", "execution_status": status}],
    )
    monkeypatch.setattr(harness, "execute", lambda *args: None)
    result = harness.verify(
        tmp_path, tmp_path, expected_run_sha256=harness.digest(tmp_path / "run.json")
    )
    assert result["integrity_verified"] is False
    assert result["exact_physical_cases"] == 0
    assert result["cases"][0]["integrity_unverified"] is True
    assert result["model_qualification"] is False
