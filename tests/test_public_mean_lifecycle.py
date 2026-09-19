"""Cheap lifecycle harness tests; never execute its three scientific fits.

These tests are written before the implementation commit and run only after that
commit. The real M0/M1/MC operations are distinct supervised qualification stages.
"""

import hashlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from glassbox._learner_arrays import array_fingerprint
from glassbox.experimental import public_mean_lifecycle as lifecycle


def test_frozen_fixture_has_three_distinct_unpadded_origins_per_recording():
    for operation in lifecycle.OPERATIONS:
        supplied = lifecycle.fixture(operation)
        for segment in supplied.segments:
            assert segment.states.shape == (6, 3)
            assert segment.inputs.shape == (5, 2)
            assert segment.states.dtype == segment.inputs.dtype == np.float64
            assert segment.dt_s == 0.25 and segment.start_row == 0
        keys = supplied.window_keys(history_steps=2, horizon_steps=1)
        for segment in supplied.segments:
            assert [
                k.origin for k in keys if k.recording_id == segment.recording_id
            ] == [2, 3, 4]
        assert supplied.configuration_id == "public-v4-small-v1"


def test_small_fixture_equations_and_constant_fixture_contents_are_fixed():
    for r in range(3):
        segment = lifecycle.fixture_segment("test", r)
        np.testing.assert_array_equal(segment.states[0], [0.05 * r, 2, 0.03 * r])
        for t in range(5):
            expected_u = 0.25 * np.sin(0.7 * (t + 1) + r)
            assert segment.inputs[t, 0] == expected_u
            assert segment.inputs[t, 1] == -0.5
            assert (
                segment.states[t + 1, 0]
                == 0.8 * segment.states[t, 0] + 0.1 * expected_u
            )
            assert segment.states[t + 1, 1] == 2
            assert (
                segment.states[t + 1, 2]
                == 0.6 * segment.states[t, 2]
                + 0.05 * segment.states[t, 0]
                + 0.02 * expected_u**2
            )
    a, b = lifecycle.fixture("MC").segments
    np.testing.assert_array_equal(a.states, np.tile([1.0, 2.0, 3.0], (6, 1)))
    np.testing.assert_array_equal(b.states, np.tile([1.125, 2.25, 3.375], (6, 1)))
    assert not np.array_equal(a.states, b.states)


def test_expected_roles_use_declared_hash_and_update_keeps_reserved_parent():
    def priority(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

    names = sorted(["small-a", "small-b"], key=priority)
    assert lifecycle.expected_roles("M0") == {
        "development": names[:1],
        "train": names[1:],
    }
    assert lifecycle.expected_roles("M1") == {
        "development": names[:1],
        "train": sorted(names[1:] + ["small-c"]),
    }
    assert set(lifecycle._expected_ledger("M1")) == {"small-a", "small-b", "small-c"}
    assert (
        lifecycle._expected_ledger("M0").items()
        <= lifecycle._expected_ledger("M1").items()
    )


def test_expected_update_cache_interleaves_old_and_fresh_without_changing_dev():
    old_keys, old = lifecycle._expected_windows("M0", "development")
    new_keys, new = lifecycle._expected_windows("M1", "development")
    assert old_keys == new_keys
    for name in lifecycle.ARRAYS:
        np.testing.assert_array_equal(old[name], new[name])
    keys, arrays = lifecycle._expected_windows("M1", "train")
    names = sorted({name for name, _ in keys})
    assert [name for name, _ in keys] == names * 3
    assert set(keys) == {(name, origin) for name in names for origin in (2, 3, 4)}
    assert all(len(v) == 6 for v in arrays.values())


def test_constructor_gap_and_defensive_cases_execute_without_fitting(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("constructor-only checks reached numerical fitting")

    monkeypatch.setattr(lifecycle.learner, "_train", forbidden)
    rows = lifecycle._constructor_checks()
    names = {r["check"] for r in rows}
    assert len(names) == len(rows) == 17
    assert {
        "mask_gap",
        "segment_defensive_arrays",
        "nan_states",
        "inf_inputs",
        "overlap",
        "mixed_intervals",
    } <= names


def test_rejection_helper_never_counts_unexpected_fitter_sentinel_as_validation():
    def ordinary():
        raise ValueError("expected malformed input")

    assert lifecycle._reject(ordinary, "invalid")["status"] == "rejected"
    with pytest.raises(lifecycle.LifecycleError, match="unexpected InjectedFailure"):
        lifecycle._reject(lifecycle._forbidden, "invalid")
    with pytest.raises(lifecycle.LifecycleError, match="accepted invalid"):
        lifecycle._reject(lambda: None, "invalid")


def test_exact_array_checks_include_shape_dtype_and_signed_zero():
    lifecycle._same(np.array([1.0]), np.array([1.0]), "same")
    for value in (np.array([1.0], np.float32), np.array([[1.0]]), np.array([2.0])):
        with pytest.raises(lifecycle.LifecycleError):
            lifecycle._same(value, np.array([1.0]), "changed")
    with pytest.raises(lifecycle.LifecycleError):
        lifecycle._same(np.array([-0.0]), np.array([0.0]), "signed zero")


def test_json_evidence_is_exclusive_and_rejects_nonfinite_scalars(tmp_path):
    path = tmp_path / "report.json"
    lifecycle._json(path, {"status": "failed", "ratio": None})
    with pytest.raises(FileExistsError):
        lifecycle._json(path, {})
    with pytest.raises(ValueError):
        lifecycle._json(tmp_path / "nonfinite.json", {"bad": np.nan})


def test_event_capture_preserves_actual_index_dtype_and_nonfinite_trial(tmp_path):
    stream = io.StringIO()
    lifecycle._capture_event(
        {"phase": "started", "attempt": 1, "indices": np.arange(3, dtype=np.int64)},
        tmp_path,
        stream,
    )
    lifecycle._capture_event(
        {
            "phase": "trial",
            "attempt": 1,
            "trial_index": 0,
            "scale": 1.0,
            "loss": float("inf"),
        },
        tmp_path,
        stream,
    )
    rows = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert rows[0] == {
        "phase": "started",
        "attempt": 1,
        "indices": [0, 1, 2],
        "indices_dtype": np.dtype(np.int64).str,
    }
    assert rows[1]["loss"] == "inf"
    assert "Infinity" not in stream.getvalue()


def test_checkpoint_capture_saves_actual_tree_and_refuses_duplicate_step(tmp_path):
    stream = io.StringIO()
    state = {
        "phase": "checkpoint",
        "step": 0,
        "validation_rollout_mse": 0.25,
        "params": {"linear": np.arange(6.0).reshape(3, 2)},
        "norms": {"state_scale": np.ones(2)},
    }
    lifecycle._capture_event(state, tmp_path, stream)
    row = json.loads(stream.getvalue())
    with np.load(tmp_path / "checkpoint-0000.npz", allow_pickle=False) as saved:
        assert row["array_fingerprint"] == array_fingerprint(
            {}, {k: saved[k] for k in saved.files}
        )
        np.testing.assert_array_equal(saved["param_linear"], state["params"]["linear"])
    with pytest.raises(lifecycle.LifecycleError, match="overwritten"):
        lifecycle._capture_event(state, tmp_path, stream)


def test_failed_prefix_reports_only_returned_work_and_retains_pending_gradient(
    tmp_path,
):
    rows = [
        {"phase": "initialized"},
        {"phase": "initial_objective", "loss": 1.0},
        {"phase": "started", "attempt": 1, "indices": [0, 1, 2]},
        {"phase": "proposed", "attempt": 1},
        {"phase": "trial", "attempt": 1, "loss": 0.5},
        {"phase": "completed", "attempt": 1},
        {"phase": "started", "attempt": 2, "indices": [0, 1, 2]},
    ]
    (tmp_path / "work.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    prefix = lifecycle._work_prefix(tmp_path)
    assert prefix["known_gradient_window_visits"] == 3
    assert prefix["incomplete_gradient_work_unknown"]
    assert prefix["full_training_objective_calls_returned"] == 2
    assert prefix["phase_counts"]["completed"] == 1


def test_missing_prefix_reports_zero_without_claiming_success(tmp_path):
    prefix = lifecycle._work_prefix(tmp_path)
    assert prefix["phase_counts"] == {}
    assert prefix["known_gradient_window_visits"] == 0
    assert "unpersisted" in prefix["scope"]


def test_archive_challenges_distinguish_raw_and_coherent_defects(tmp_path, monkeypatch):
    # Only test challenge construction here; actual loader refusals use saved M0.
    arrays = {
        "param_linear": np.ones((3, 3)),
        "param_autonomous": np.ones((6, 3)),
        "train_future_states": np.ones((3, 1, 3)),
        "norm_state_scale": np.ones(3),
        "envelope_half_width": np.zeros((1, 3)),
    }
    meta = {
        "format": "glassbox-default-recipe-v4",
        "recipe": {"id": "generic-memory-v4-prototype", "steps": 1000},
        "report": {"recipe": {"id": "generic-memory-v4-prototype", "steps": 1000}},
        "contract": {"configuration_id": "fixture"},
        "model": {"format": "glassbox-sequence-v2"},
    }
    original = tmp_path / "model.npz"
    lifecycle._write_archive(original, meta, arrays, coherent=True)
    digest = lifecycle._sha(original)
    loaded = []

    def refuse(path):
        changed_meta, changed_arrays = lifecycle._archive(path)
        fingerprint = changed_meta.pop("fingerprint")
        coherent = fingerprint == array_fingerprint(changed_meta, changed_arrays)
        loaded.append((Path(path).stem, coherent))
        raise ValueError("test validator")

    monkeypatch.setattr(lifecycle.learner.LearnedDynamics, "load", refuse)
    rows = lifecycle.archive_defects(original, tmp_path / "defects")
    assert len(rows) == len(loaded) == 13
    assert all(coherent == (not name.startswith("raw_")) for name, coherent in loaded)
    assert lifecycle._sha(original) == digest
    assert len({row["payload_fingerprint"] for row in rows}) == 13


def test_configuration_does_not_persist_environment_secrets(monkeypatch):
    monkeypatch.setenv("GLASSBOX_TEST_PRIVATE_VALUE", "must-not-appear-in-artifacts")
    before = lifecycle._configuration()
    assert "must-not-appear-in-artifacts" not in json.dumps(before)
    monkeypatch.setenv("GLASSBOX_TEST_PRIVATE_VALUE", "changed")
    assert (
        lifecycle._configuration()["environment_sha256"] != before["environment_sha256"]
    )


def test_thread_scope_uses_initializer_sentinel_and_restores_false():
    # Entry raises before initialization; no fitting, gradient or forecast runs.
    environment = dict(os.environ)
    environment.pop("JAX_ENABLE_X64", None)
    code = "from glassbox.experimental.public_mean_lifecycle import _thread_check; import json; print(json.dumps(_thread_check()))"
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    assert json.loads(result.stdout)["observed"] == {
        "before": False,
        "inside": True,
        "independent": False,
        "sentinel": True,
        "after": False,
    }


def test_run_refuses_existing_attempt_before_any_worker(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("resumed an existing attempt")

    monkeypatch.setattr(lifecycle, "_launch", forbidden)
    monkeypatch.setattr(lifecycle, "_identity", forbidden)
    with pytest.raises(FileExistsError):
        lifecycle.run(tmp_path)


def test_run_dispatches_exact_three_operations_then_two_no_fit_workers(
    tmp_path, monkeypatch
):
    commands = []
    monkeypatch.setattr(lifecycle, "_identity", lambda: {"test": "source-identity"})

    def launch(command, *args):
        commands.append(command)
        return {"returncode": 0}

    monkeypatch.setattr(lifecycle, "_launch", launch)
    monkeypatch.setattr(lifecycle, "_boundary", lambda *args: [])
    directory = tmp_path / "new-attempt"
    result = lifecycle.run(directory)
    assert result["status"] == "complete" and result["successful_fit_budget"] == 3
    assert [c[c.index("--operation") + 1] for c in commands[:3]] == ["M0", "M1", "MC"]
    assert all(c[2] == lifecycle.MODULE for c in commands)
    assert [c[3] for c in commands] == ["_worker"] * 3 + ["_checks"] * 2
    assert [c[c.index("--ambient") + 1] for c in commands[3:]] == ["false", "true"]


def test_run_failure_is_sealed_without_retry_or_later_operations(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(lifecycle, "_identity", lambda: {"test": "identity"})

    def fail(command, *args):
        calls.append(command)
        raise lifecycle.LifecycleError("failed M0")

    monkeypatch.setattr(lifecycle, "_launch", fail)
    directory = tmp_path / "failed"
    with pytest.raises(lifecycle.LifecycleError, match="failed M0"):
        lifecycle.run(directory)
    saved = lifecycle._read(directory / "run.json")
    assert saved["status"] == "failed" and len(calls) == 1
    assert "intent.json" in saved["files"]


def test_hard_timeout_retains_command_log_and_incomplete_status(tmp_path, monkeypatch):
    def timeout(command, **kwargs):
        kwargs["stdout"].write("worker entered\n")
        raise subprocess.TimeoutExpired(command, 14400)

    monkeypatch.setattr(lifecycle.subprocess, "run", timeout)
    with pytest.raises(lifecycle.LifecycleError, match="incomplete"):
        lifecycle._launch(
            ["test-worker"], tmp_path / "stage.log", tmp_path / "stage.command.json"
        )
    saved = lifecycle._read(tmp_path / "stage.command.exit.json")
    assert saved["returncode"] is None and saved["status"] == "hard_timeout_incomplete"
    assert saved["log_sha256"] == lifecycle._sha(tmp_path / "stage.log")


def test_failed_child_is_never_silently_retried(tmp_path, monkeypatch):
    calls = []

    def failed(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=3)

    monkeypatch.setattr(lifecycle.subprocess, "run", failed)
    with pytest.raises(lifecycle.LifecycleError, match="worker failed"):
        lifecycle._launch(
            ["test-worker"], tmp_path / "stage.log", tmp_path / "stage.command.json"
        )
    assert len(calls) == 1
    assert lifecycle._read(tmp_path / "stage.command.exit.json")["returncode"] == 3


def test_replay_refuses_nested_output_without_touching_sealed_directory(tmp_path):
    original = tmp_path / "sealed"
    original.mkdir()
    with pytest.raises(lifecycle.LifecycleError, match="outside sealed"):
        lifecycle.replay(original, "0" * 64, original / "replay")
    assert list(original.iterdir()) == []


def test_replay_wrong_anchor_preserves_failure_before_any_numerical_worker(
    tmp_path, monkeypatch
):
    original = tmp_path / "sealed"
    original.mkdir()
    lifecycle._json(original / "run.json", {"status": "complete"})

    def forbidden(*args, **kwargs):
        pytest.fail("unauthenticated replay reached a worker")

    monkeypatch.setattr(lifecycle, "_launch", forbidden)
    output = tmp_path / "replay"
    with pytest.raises(lifecycle.LifecycleError, match="external lifecycle anchor"):
        lifecycle.replay(original, "0" * 64, output)
    assert lifecycle._read(output / "replay.json")["status"] == "failed"


def test_replay_rejects_extra_payload_before_dispatch(tmp_path, monkeypatch):
    original = tmp_path / "sealed"
    original.mkdir()
    identity = {"test": "identity"}
    lifecycle._json(
        original / "run.json",
        {
            "status": "complete",
            "identity": identity,
            "operations": list(lifecycle.OPERATIONS),
            "successful_fit_budget": 3,
            "files": {},
        },
    )
    (original / "undeclared.bin").write_bytes(b"extra")
    monkeypatch.setattr(lifecycle, "_identity", lambda: identity)
    monkeypatch.setattr(
        lifecycle, "_launch", lambda *args: pytest.fail("bad roster reached worker")
    )
    with pytest.raises(lifecycle.LifecycleError, match="payload hashes/roster"):
        lifecycle.replay(
            original, lifecycle._sha(original / "run.json"), tmp_path / "replay"
        )


@pytest.mark.parametrize(
    "argv",
    [
        ["run"],
        ["replay"],
        ["_worker", "--directory", "x"],
        ["_checks", "--directory", "x", "--output", "y"],
    ],
)
def test_cli_refuses_incomplete_dispatch_before_fixture_execution(argv):
    with pytest.raises(SystemExit) as error:
        lifecycle.main(argv)
    assert error.value.code == 2
