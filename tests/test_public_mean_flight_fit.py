"""Qualification control-flow/evidence tests; these tests perform no fitting."""

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from glassbox._learner_arrays import array_fingerprint, load_arrays
from glassbox.experimental import public_mean_flight_fit as subject
from glassbox.recordings import WindowKey


def _windows(count, *, offset=0):
    batch = SimpleNamespace(
        dt_s=0.02,
        past_states=np.arange(count * 2, dtype=np.float64).reshape(count, 2, 1),
        past_inputs=np.zeros((count, 1, 1), dtype=np.float64),
        future_inputs=np.zeros((count, 1, 1), dtype=np.float64),
        future_states=np.ones((count, 1, 1), dtype=np.float64),
    )
    return SimpleNamespace(
        batch=batch,
        keys=tuple(WindowKey("parent", "segment", i + offset) for i in range(count)),
        source_origins=tuple(range(offset, offset + count)),
        excitation_declared=False,
    )


def _reference():
    return SimpleNamespace(
        train=_windows(1536),
        development=_windows(256, offset=10000),
        contract={"configuration_id": "fixture", "dt_s": 0.02},
        seen={"parent": "content"},
        collection=object(),
        roles={
            "training": [str(i) for i in range(72)],
            "development": [str(i) for i in range(24)],
        },
        ridge=15.36,
        delay=1,
        source={"reference_bundle_sha256": subject.REFERENCE_SHA256},
    )


def test_preparation_compares_actual_commands_and_identities():
    reference = _reference()
    metadata, arrays = subject._expected_preparation(reference)
    subject._compare_preparation((metadata, arrays), reference)
    changed = {**arrays, "train_future_inputs": arrays["train_future_inputs"].copy()}
    changed["train_future_inputs"][1, 0, 0] = 0.25
    with pytest.raises(subject.FlightFitError, match="byte-identical"):
        subject._compare_preparation((metadata, changed), reference)
    metadata = deepcopy(metadata)
    metadata["windows"]["development"]["source_origins"][0] += 1
    with pytest.raises(subject.FlightFitError, match="contract/ledger/windows"):
        subject._compare_preparation((metadata, arrays), reference)


def test_array_equality_checks_dtype_shape_and_signed_zero():
    for altered in (
        np.array([0], dtype=np.float32),
        np.array([[0.0]]),
        np.array([-0.0]),
    ):
        with pytest.raises(subject.FlightFitError, match="byte-identical"):
            subject._same(altered, np.array([0.0]), "fixture")


@pytest.fixture(scope="module")
def work_evidence(tmp_path_factory):
    directory = tmp_path_factory.mktemp("flight-evidence")
    initial = {f"param_p{i}": np.array([float(i)]) for i in range(9)}
    initial.update({f"norm_n{i}": np.array([float(i + 1)]) for i in range(8)})
    selected = {k: v + 1 for k, v in initial.items()}
    weights = {name: np.array([1.0]) for name in subject.WEIGHTS}
    reference = SimpleNamespace(
        initial_arrays=initial,
        selected_arrays=selected,
        objective_arrays=weights,
        ridge=15.36,
        delay=1,
    )
    np.savez_compressed(directory / "initializer-return.npz", **initial)
    np.savez_compressed(directory / "weight-initial.npz", **initial)
    np.savez_compressed(directory / "weights.npz", **weights)
    subject._write(
        directory / "initializer-entered.json",
        {
            "jax_enable_x64": True,
            "training_windows": 1536,
            "ridge": 15.36,
            "settings": {"seed": 0, "width": 32, "memory": 8, "delay_steps": 1},
        },
    )
    rows = [{"phase": "initialized"}, {"phase": "weights"}]
    checkpoints = []

    def checkpoint(step, loss):
        arrays = initial if step == 0 else selected
        np.savez_compressed(directory / f"checkpoint-{step:04d}.npz", **arrays)
        row = dict(
            phase="checkpoint",
            step=step,
            validation_rollout_mse=1.0 - step / 2000,
            unweighted_validation_rollout_mse=2.0 - step / 2000,
            array_fingerprint=array_fingerprint({}, arrays),
        )
        if step:
            row.update(full_training_loss=loss, selected_step=step)
        rows.append(row)
        checkpoints.append(row)

    checkpoint(0, 1.0)
    rows.append(dict(phase="initial_objective", loss=1.0))
    losses = [dict(step=0, full_training_loss=1.0)]
    scales = []
    objective_calls = 1
    current = 1.0
    for index in range(1, 1001):
        rows.append(
            dict(
                phase="started",
                attempt=index,
                indices=list(range(1536)),
                indices_dtype=np.dtype("int64").str,
            )
        )
        rows.append(
            dict(
                phase="proposed",
                attempt=index,
                gradient_finite=True,
                minibatch_loss=current,
                gradient_norm=0.1,
            )
        )
        if index == 2:
            for trial, scale in enumerate(subject.core.SCALES):
                rows.append(
                    dict(
                        phase="trial",
                        attempt=index,
                        trial_index=trial,
                        scale=scale,
                        loss=current,
                    )
                )
            objective_calls += 8
            scale, trial = 0.0, None
        else:
            if index == 1:
                rows.append(
                    dict(
                        phase="trial",
                        attempt=index,
                        trial_index=0,
                        scale=1.0,
                        loss="nan",
                    )
                )
                objective_calls += 1
                trial, scale = 1, 0.5
            else:
                trial, scale = 0, 1.0
            current -= 0.0005
            rows.append(
                dict(
                    phase="trial",
                    attempt=index,
                    trial_index=trial,
                    scale=scale,
                    loss=current,
                )
            )
            objective_calls += 1
        rows.append(
            dict(
                phase="completed",
                attempt=index,
                accepted_scale=scale,
                accepted_trial_index=trial,
                loss=current,
            )
        )
        scales.append(scale)
        if index % 100 == 0:
            checkpoint(index, current)
            losses.append(dict(step=index, full_training_loss=current))
    report = dict(
        optimization=dict(
            selected_step=1000,
            trace=[
                {k: row[k] for k in ("step", "validation_rollout_mse")}
                for row in checkpoints
            ],
            unweighted_development_trace=[
                dict(
                    step=row["step"],
                    validation_rollout_mse=row["unweighted_validation_rollout_mse"],
                )
                for row in checkpoints
            ],
            safeguard=subject.core.safeguard_metadata(
                steps=1000,
                accepted_scales=scales,
                objective_calls=objective_calls,
                initial_loss=1.0,
                final_loss=current,
                checkpoint_losses=losses,
                selected_step=1000,
            ),
            gradient=subject.core.gradient_metadata(
                1536,
                attempts_started=1000,
                gradient_proposal_calls_returned=1000,
                completed_acceptance_attempts=1000,
            ),
        )
    )
    return directory, rows, SimpleNamespace(report=report), reference


def test_scalar_chain_accepts_nonfinite_backtrack_and_strict_rejection(
    work_evidence, monkeypatch
):
    directory, rows, model, reference = work_evidence
    monkeypatch.setattr(subject, "_rows", lambda *args, **kwargs: (rows, None))

    def forbidden(*args, **kwargs):
        raise AssertionError("verification must not perform numerical model work")

    for name in (
        "initialize_sequence_model",
        "fit_sequence_model",
        "initial_training_forecast",
        "make_training_objective",
    ):
        monkeypatch.setattr(subject.core, name, forbidden)
    monkeypatch.setattr(np.linalg, "solve", forbidden)
    result = subject._work(directory, model, reference)
    assert result["known_gradient_window_visits"] == 1536000
    assert result["full_training_objective_calls_returned"] == 1009
    assert result["selected_step"] == 1000


def test_selected_witness_must_match_authenticated_research(work_evidence, monkeypatch):
    directory, rows, model, reference = work_evidence
    reference = deepcopy(reference)
    reference.selected_arrays["param_p0"][0] += 0.01
    monkeypatch.setattr(subject, "_rows", lambda *args, **kwargs: (rows, None))
    with pytest.raises(subject.FlightFitError, match="observed checkpoint"):
        subject._work(directory, model, reference)


@pytest.mark.parametrize(
    "mutation",
    [
        "index_order",
        "dtype",
        "integer",
        "omission",
        "late_accept",
        "scale",
        "selected",
        "counter",
        "trace",
    ],
)
def test_scalar_chain_rejects_coherent_semantic_changes(
    work_evidence, monkeypatch, mutation
):
    directory, original_rows, original_model, reference = work_evidence
    # Scalar rows are copied; the common ordered index lists stay immutable.
    rows = [dict(row) for row in original_rows]
    model = SimpleNamespace(report=deepcopy(original_model.report))
    started = next(row for row in rows if row["phase"] == "started")
    trial = next(row for row in rows if row["phase"] == "trial")
    if mutation == "index_order":
        started["indices"] = started["indices"][::-1]
    elif mutation == "dtype":
        started["indices_dtype"] = "<i4"
    elif mutation == "integer":
        started["indices"] = [float(i) for i in started["indices"]]
    elif mutation == "omission":
        rows.pop(
            next(i for i, row in enumerate(rows) if row.get("phase") == "completed")
        )
    elif mutation == "late_accept":
        trial["loss"] = 0.25
    elif mutation == "scale":
        trial["scale"] = 0.5
    elif mutation == "selected":
        model.report["optimization"]["selected_step"] = 900
    elif mutation == "counter":
        model.report["optimization"]["gradient"]["known_gradient_window_visits"] -= 1
    else:
        model.report["optimization"]["unweighted_development_trace"][0][
            "validation_rollout_mse"
        ] += 1
    monkeypatch.setattr(subject, "_rows", lambda *args, **kwargs: (rows, None))
    with pytest.raises(subject.FlightFitError):
        subject._work(directory, model, reference)


def test_prefix_preserves_unreturned_gradient_and_interrupted_write(tmp_path):
    path = tmp_path / "work.jsonl"
    path.write_text(json.dumps(dict(phase="started", indices=[0, 1])) + '\n{"phase":')
    evidence = subject._prefix(tmp_path, incomplete=True)
    assert evidence["known_gradient_window_visits"] == 0
    assert evidence["incomplete_gradient_work_unknown"] is True
    assert evidence["incomplete_tail"]["bytes"] == 9
    with pytest.raises(subject.FlightFitError, match="invalid persisted"):
        subject._prefix(tmp_path)


@pytest.mark.parametrize("failure", [None, "initializer", "calibration"])
def test_worker_captures_preparation_before_one_actual_call_and_keeps_failures(
    tmp_path, monkeypatch, failure
):
    reference = _reference()
    arrays = {"param_fixture": np.array([1.0])}
    core_model = SimpleNamespace(arrays=lambda: arrays)
    model = SimpleNamespace(
        report={"fixture": "mocked-control-flow"}, fingerprint=lambda: "model"
    )
    model.save = lambda path: np.savez_compressed(path, **arrays)
    monkeypatch.setattr(
        subject,
        "_configuration",
        lambda: {"jax_enable_x64": False, "environment_sha256": "env"},
    )
    monkeypatch.setattr(
        subject, "_binding", lambda *args: {"implementation_commit": "commit"}
    )
    monkeypatch.setattr(subject, "_prepare", lambda *args: reference)
    monkeypatch.setattr(
        subject,
        "save_recordings",
        lambda collection, path: Path(path).write_bytes(b"recordings"),
    )
    monkeypatch.setattr(subject, "_compare_model", lambda *args: {"mocked": True})
    monkeypatch.setattr(subject, "_work", lambda *args: {"mocked": True})
    calls = []

    def initializer(batch, **kwargs):
        calls.append("initializer")
        assert (tmp_path / "recordings.npz").is_file()
        assert (tmp_path / "initializer-entered.json").is_file()
        subject._compare_preparation(
            load_arrays(tmp_path / "actual-preparation.npz"), reference
        )
        if failure == "initializer":
            raise ArithmeticError("retained initializer failure")
        return core_model

    def fitter(train, development, **kwargs):
        calls.append("fitter")
        result = subject.core.initialize_sequence_model(train, **kwargs)
        subject.core._observe_attempt(dict(phase="initialized", model=result))
        return result, {}

    def calibration(*args):
        calls.append("calibration")
        if failure == "calibration":
            raise ArithmeticError("retained calibration failure")

    def training(train, development, contract, seen, **kwargs):
        result, _ = subject.learner.fit_sequence_model(
            train.batch,
            development.batch,
            seed=0,
            width=32,
            memory=8,
            ridge=reference.ridge,
            delay_steps=reference.delay,
        )
        subject.learner._calibrate(result, development)
        return model

    def public_fit(collection):
        calls.append("public")
        assert collection is reference.collection
        return subject.learner._train(
            reference.train, reference.development, reference.contract, reference.seen
        )

    monkeypatch.setattr(subject.core, "initialize_sequence_model", initializer)
    monkeypatch.setattr(subject.learner, "fit_sequence_model", fitter)
    monkeypatch.setattr(subject.learner, "_calibrate", calibration)
    monkeypatch.setattr(subject.learner, "_train", training)
    monkeypatch.setattr(subject.glassbox, "fit", public_fit)
    result = subject._worker(
        "crazyflow",
        tmp_path,
        implementation_manifest="binding",
        implementation_sha256="sha",
        reference_root="reference",
    )
    assert calls == ["public", "fitter", "initializer"] + (
        [] if failure == "initializer" else ["calibration"]
    )
    assert result["status"] == ("failed" if failure else "complete")
    assert (
        result["timing"]["public_fit_calls"]
        == result["timing"]["fitter_calls"]
        == result["timing"]["initializer_calls"]
        == 1
    )
    assert (tmp_path / "outcome.json").is_file()
    assert subject.core.initialize_sequence_model is initializer
    assert subject.learner._train is training
    if failure:
        assert result["error_type"] == "ArithmeticError"
        assert not (tmp_path / "model.npz").exists()


def test_binding_failure_never_reaches_preparation_or_public_fit(tmp_path, monkeypatch):
    monkeypatch.setattr(subject, "_configuration", lambda: {"jax_enable_x64": False})

    def fail(*args):
        raise subject.FlightFitError("source mismatch")

    monkeypatch.setattr(subject, "_binding", fail)
    monkeypatch.setattr(
        subject,
        "_prepare",
        lambda *args: pytest.fail("preparation must follow binding"),
    )
    monkeypatch.setattr(
        subject.glassbox,
        "fit",
        lambda *args: pytest.fail("fitting must follow binding"),
    )
    result = subject._worker(
        "cascade",
        tmp_path,
        implementation_manifest="binding",
        implementation_sha256="sha",
        reference_root="reference",
    )
    assert result["status"] == "failed" and result["stage"] == "binding"
    assert result["timing"]["public_fit_calls"] == 0


def test_binding_rejects_an_actual_import_missing_from_external_source_seal(
    monkeypatch,
):
    bound = {
        "public_root": str(Path(subject.__file__).resolve().parents[3]),
        "public_source_sha256": {},
    }
    monkeypatch.setattr(subject.implementation, "verify", lambda *args: bound)
    with pytest.raises(subject.FlightFitError, match="committed implementation seal"):
        subject._binding("manifest", "external-sha")


@pytest.mark.parametrize("timeout", [False, True])
def test_supervisor_seals_incomplete_launch_or_hard_timeout_once(
    tmp_path, monkeypatch, timeout
):
    output = tmp_path / "output"
    monkeypatch.setattr(
        subject,
        "_binding",
        lambda *args: {
            "interpreter": "python",
            "public_root": str(tmp_path),
            "implementation_commit": "commit",
        },
    )
    calls = []

    class Process:
        pid = 123

        def wait(self, *, timeout=None):
            calls.append(("wait", timeout))
            if timeout is not None:
                (output / "outcome.json").write_text('{"status":')
                (output / "work.jsonl").write_text('{"phase":')
                raise subject.subprocess.TimeoutExpired("worker", timeout)
            return -9

    def launch(*args, **kwargs):
        calls.append(("launch", kwargs["env"].get("JAX_ENABLE_X64")))
        if not timeout:
            raise OSError("cannot launch")
        return Process()

    monkeypatch.setattr(subject.subprocess, "Popen", launch)
    monkeypatch.setattr(
        subject.os, "killpg", lambda *args: calls.append(("kill", args))
    )
    result = subject.run(
        "cascade",
        output,
        implementation_manifest="binding",
        implementation_sha256="sha",
        reference_root="reference",
    )
    assert result["status"] == (
        "incomplete_hard_timeout" if timeout else "incomplete_worker_exit"
    )
    assert len([row for row in calls if row[0] == "launch"]) == 1
    replay = subject.verify(
        output,
        expected_manifest_sha256=result["manifest_sha256"],
        implementation_manifest="binding",
        implementation_sha256="sha",
        reference_root="reference",
    )
    assert replay["status"] == result["status"]


def test_inventory_includes_nested_manifest_and_rejects_symlink(tmp_path):
    subject._write(tmp_path / "manifest.json", {"root": True})
    (tmp_path / "nested").mkdir()
    subject._write(tmp_path / "nested/manifest.json", {"nested": True})
    assert set(subject._inventory(tmp_path)) == {"nested/manifest.json"}
    (tmp_path / "link").symlink_to(tmp_path / "nested/manifest.json")
    with pytest.raises(subject.FlightFitError, match="symlink"):
        subject._inventory(tmp_path)


def test_status_cannot_downgrade_completed_zero_exit(tmp_path):
    subject._write(tmp_path / "outcome.json", {"status": "complete"})
    assert (
        subject._stage_status(tmp_path, {"exit_code": 0, "hard_timeout": False})
        == "complete"
    )
    assert (
        subject._stage_status(tmp_path, {"exit_code": 1, "hard_timeout": False})
        == "failed"
    )
