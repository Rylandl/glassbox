"""One observed public fit per frozen flight calibration collection.

This qualification adapter reconstructs authenticated recordings, observes the
single public call without replacing numerical operations, and compares its
actual arrays with the retained research witnesses. Replay checks saved evidence;
it does not initialize, fit, differentiate, simulate, or forecast.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import signal
import subprocess
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import jax
import numpy as np

import glassbox
from glassbox import _sequence_model as core
from glassbox import learner
from glassbox._learner_arrays import array_fingerprint, load_arrays, save_arrays
from glassbox.io.recordings import load_recordings, save_recordings

from . import public_mean_implementation as implementation
from . import public_mean_lifecycle as lifecycle
from . import public_mean_saved_port as saved_port
from .public_mean_scoring import PROTOCOL_SHA256

MODULE = "glassbox.experimental.public_mean_flight_fit"
FORMAT = "glassbox-public-mean-flight-fit-v1"
REFERENCE_SHA256 = "88b59d60be28ffed1e93fa6e6ecaa026b1d49950a23f7f2b1e0f9b65b3b9d99d"
HARD_WALL_TIME_S = 14400
ARRAYS = ("past_states", "past_inputs", "future_inputs", "future_states")
WEIGHTS = (
    "normalization",
    "initial_training_prediction",
    "initial_channel_mse",
    "raw_channel_weights",
    "channel_weights",
    "fixed_weights",
    "weight_floor",
    "weight_normalizer",
)


class FlightFitError(ValueError):
    """A violated qualification contract, not an alternative model choice."""


def _require(condition, message):
    if not condition:
        raise FlightFitError(message)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path):
    return json.loads(Path(path).read_text())


def _write(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _same(actual, expected, label):
    actual, expected = np.asarray(actual), np.asarray(expected)
    _require(
        actual.shape == expected.shape
        and actual.dtype.str == expected.dtype.str
        and actual.tobytes(order="C") == expected.tobytes(order="C"),
        label + " is not byte-identical",
    )


def _same_tree(actual, expected, label):
    _require(set(actual) == set(expected), label + " array roster differs")
    for name in expected:
        _same(actual[name], expected[name], label + ":" + name)


def _same_recordings(actual, expected):
    _require(
        learner._contract(actual) == learner._contract(expected),
        "saved recording contract differs",
    )
    _require(
        len(actual.segments) == len(expected.segments),
        "saved recording segment roster differs",
    )
    for left, right in zip(actual.segments, expected.segments, strict=True):
        for key in ("recording_id", "segment_id", "start_row", "dt_s"):
            _require(
                getattr(left, key) == getattr(right, key),
                "saved recording " + key + " differs",
            )
        for key in ("states", "inputs", "excitation"):
            a, b = getattr(left, key), getattr(right, key)
            if a is None or b is None:
                _require(a is b, "saved recording " + key + " declaration differs")
            else:
                _same(a, b, "saved recording " + key)


def _npz(path):
    with np.load(path, allow_pickle=False) as archive:
        _require(len(archive.files) == len(set(archive.files)), "duplicate NPZ arrays")
        return {name: archive[name] for name in archive.files}


def _tree(params, norms):
    return {
        **{"param_" + k: np.asarray(v) for k, v in params.items()},
        **{"norm_" + k: np.asarray(v) for k, v in norms.items()},
    }


def _cache_payload(train, development, contract, seen):
    metadata = dict(format=FORMAT, contract=contract, seen=seen, windows={})
    arrays = {}
    for role, windows in (("train", train), ("development", development)):
        metadata["windows"][role] = dict(
            dt_s=windows.batch.dt_s,
            keys=[
                dict(
                    recording_id=k.recording_id,
                    segment_id=k.segment_id,
                    origin=int(k.origin),
                )
                for k in windows.keys
            ],
            source_origins=[int(x) for x in windows.source_origins],
            excitation_declared=windows.excitation_declared,
        )
        arrays.update({role + "_" + key: getattr(windows.batch, key) for key in ARRAYS})
        if windows.excitation_declared:
            arrays.update(
                {
                    role + "_" + key: getattr(windows, key)
                    for key in ("past_excitation", "future_excitation")
                }
            )
    return metadata, arrays


def _expected_preparation(reference):
    return _cache_payload(
        reference.train, reference.development, reference.contract, reference.seen
    )


def _compare_preparation(actual, reference):
    metadata, arrays = actual
    expected_meta, expected_arrays = _expected_preparation(reference)
    _require(
        metadata == expected_meta, "actual preparation contract/ledger/windows differ"
    )
    _same_tree(arrays, expected_arrays, "actual preparation")


def _configuration():
    return dict(
        jax_enable_x64=bool(jax.config.x64_enabled),
        environment_sha256=hashlib.sha256(
            json.dumps(dict(os.environ), sort_keys=True).encode()
        ).hexdigest(),
    )


def _binding(path, expected_sha256):
    bound = implementation.verify(path, expected_sha256)
    root = Path(bound["public_root"])
    modules = (
        glassbox,
        core,
        learner,
        implementation,
        lifecycle,
        saved_port,
        sys.modules[load_arrays.__module__],
        sys.modules[load_recordings.__module__],
        sys.modules[glassbox.SequenceCollection.__module__],
    )
    paths = [Path(module.__file__).resolve() for module in modules]
    paths.append(Path(__file__).resolve())
    for source in paths:
        _require(
            source.is_relative_to(root / "src/glassbox"),
            "worker imports another public checkout",
        )
        relative = source.relative_to(root).as_posix()
        _require(
            bound["public_source_sha256"].get(relative) == _sha(source),
            "worker source is not in its committed implementation seal",
        )
    _require(core.FIT_WALL_TIME_LIMIT_S == 7200, "internal fitter wall budget differs")
    return bound


def _prepare(reference_root, simulator):
    _require(simulator in ("crazyflow", "cascade"), "unknown simulator")
    reference = saved_port.prepare_reference(
        reference_root, simulator, expected_bundle_sha256=REFERENCE_SHA256
    )
    _require(
        len(reference.train.keys) == 1536 and len(reference.development.keys) == 256,
        "frozen cache counts differ",
    )
    _require(
        len(reference.seen) == 96
        and len(reference.roles["training"]) == 72
        and len(reference.roles["development"]) == 24,
        "frozen recording roles differ",
    )
    return reference


def _compare_model(model, reference):
    _require(
        model.contract == reference.contract and model._seen == reference.seen,
        "public model contract/ledger differs",
    )
    _require(
        model.report["previous_revision"] is None,
        "flight fit unexpectedly updates a model",
    )
    _require(model.recipe == learner.RECIPE, "public recipe differs")
    _compare_preparation(
        _cache_payload(model._train, model._development, model.contract, model._seen),
        reference,
    )
    _same_tree(model._model.arrays(), reference.selected_arrays, "selected mean")
    opt = model.report["optimization"]
    _require(
        opt["selected_step"]
        == reference.source_report["optimization"]["selected_step"],
        "selected step differs from research",
    )
    _require(
        opt["steps"] == 1000
        and opt["batch_size"] == 1536
        and opt["ridge"] == reference.ridge,
        "actual fit budget/ridge differs",
    )
    _require(opt["delay_steps"] == reference.delay, "actual delay differs")
    _same(np.asarray(opt["error_scale"]), reference.error_scale, "hold scales")
    for name in (
        "initial_channel_mse",
        "raw_channel_weights",
        "channel_weights",
        "weight_floor",
        "weight_normalizer",
    ):
        _same(
            np.asarray(opt["objective"][name]),
            reference.objective_arrays[name],
            "fixed objective " + name,
        )
    _require(
        opt["objective"]["initial_parameters_and_norms_fingerprint"]
        == array_fingerprint({}, reference.initial_arrays),
        "reported initial array fingerprint differs",
    )
    _require(
        opt["objective"]["initial_training_prediction_fingerprint"]
        == array_fingerprint(
            {},
            {"prediction": reference.objective_arrays["initial_training_prediction"]},
        ),
        "reported initial forecast fingerprint differs",
    )
    _require(
        opt["selection_objective"] == core.OBJECTIVE, "selection objective differs"
    )
    return dict(
        selected_step=opt["selected_step"],
        ridge=opt["ridge"],
        training_windows=1536,
        development_windows=256,
        recording_roles={k: len(v) for k, v in reference.roles.items()},
        initial_arrays_fingerprint=array_fingerprint({}, reference.initial_arrays),
        selected_arrays_fingerprint=array_fingerprint({}, reference.selected_arrays),
        error_scale_fingerprint=array_fingerprint(
            {}, {"error_scale": reference.error_scale}
        ),
        scope="Exact scientific array/role/selection identity; new format, recipe labels and implementation metadata differ intentionally.",
    )


def _rows(path, *, incomplete=False):
    if not Path(path).exists():
        return [], None
    lines = Path(path).read_bytes().splitlines(keepends=True)
    result = []
    for index, raw in enumerate(lines):
        try:
            result.append(json.loads(raw))
        except (ValueError, UnicodeDecodeError):
            if incomplete and index == len(lines) - 1 and not raw.endswith(b"\n"):
                return result, dict(
                    bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()
                )
            raise FlightFitError("invalid persisted work event") from None
    return result, None


def _prefix(directory, *, incomplete=False):
    rows, tail = _rows(Path(directory) / "work.jsonl", incomplete=incomplete)
    counts = Counter(row["phase"] for row in rows)
    started = [row for row in rows if row["phase"] == "started"]
    returned = counts["proposed"]
    _require(returned <= len(started), "returned gradients exceed entered attempts")
    return dict(
        phase_counts=dict(sorted(counts.items())),
        known_gradient_window_visits=sum(
            len(row["indices"]) for row in started[:returned]
        ),
        incomplete_gradient_work_unknown=len(started) > returned,
        full_training_objective_calls_returned=counts["initial_objective"]
        + counts["trial"],
        incomplete_tail=tail,
        scope="Persisted observations only; no inference about unreturned or unwritten work.",
    )


def _work(directory, model, reference):
    """Check the actual scalar acceptance chain without replaying an optimizer."""
    rows, _ = _rows(Path(directory) / "work.jsonl")
    position = 0

    def take(phase):
        nonlocal position
        _require(
            position < len(rows) and rows[position].get("phase") == phase,
            "work event order differs: " + phase,
        )
        row = rows[position]
        position += 1
        return row

    take("initialized")
    take("weights")
    checkpoints = [take("checkpoint")]
    _require(checkpoints[0]["step"] == 0, "missing initial development checkpoint")
    current = take("initial_objective")["loss"]
    _require(
        type(current) in (int, float) and np.isfinite(current) and current >= 0,
        "invalid initial acceptance loss",
    )
    initial, calls, scales = current, 1, []
    for index in range(1, 1001):
        start, proposal = take("started"), take("proposed")
        _require(
            type(start["attempt"]) is type(proposal["attempt"]) is int
            and start["attempt"] == proposal["attempt"] == index,
            "attempt indices differ",
        )
        _require(
            start["indices_dtype"] == np.dtype(np.int64).str
            and start["indices"] == list(range(1536))
            and all(type(x) is int for x in start["indices"]),
            "actual full-gradient rows differ",
        )
        _require(
            proposal["gradient_finite"] is True
            and np.isfinite(
                [proposal["minibatch_loss"], proposal["gradient_norm"]]
            ).all(),
            "nonfinite completed proposal",
        )
        accepted, accepted_index = 0.0, None
        for trial_index, alpha in enumerate(core.SCALES):
            row = take("trial")
            _require(
                type(row["attempt"]) is type(row["trial_index"]) is int
                and row["attempt"] == index
                and row["trial_index"] == trial_index
                and row["scale"] == alpha,
                "trial scale order differs",
            )
            calls += 1
            loss = row["loss"]
            if isinstance(loss, str):
                _require(
                    loss in ("nan", "inf", "-inf"), "invalid nonfinite trial encoding"
                )
            else:
                _require(
                    type(loss) in (int, float) and np.isfinite(loss) and loss >= 0,
                    "invalid finite trial loss",
                )
                if loss < current:
                    current, accepted, accepted_index = loss, alpha, trial_index
                    break
        done = take("completed")
        _require(
            type(done["attempt"]) is int
            and done["attempt"] == index
            and done["accepted_scale"] == accepted
            and done["accepted_trial_index"] == accepted_index
            and done["loss"] == current,
            "first strict acceptable trial or rejection differs",
        )
        scales.append(accepted)
        if index % 100 == 0:
            checkpoint = take("checkpoint")
            _require(
                checkpoint["step"] == index
                and checkpoint["full_training_loss"] == current,
                "checkpoint acceptance scalar differs",
            )
            checkpoints.append(checkpoint)
    _require(position == len(rows), "extra work events")
    opt = model.report["optimization"]
    trace = [
        dict(step=row["step"], validation_rollout_mse=row["validation_rollout_mse"])
        for row in checkpoints
    ]
    original = [
        dict(
            step=row["step"],
            validation_rollout_mse=row["unweighted_validation_rollout_mse"],
        )
        for row in checkpoints
    ]
    _require(
        all(
            np.isfinite(
                [r["validation_rollout_mse"], r["unweighted_validation_rollout_mse"]]
            ).all()
            for r in checkpoints
        ),
        "nonfinite observed development loss",
    )
    _require(
        trace
        == [
            {k: row[k] for k in ("step", "validation_rollout_mse")}
            for row in opt["trace"]
        ]
        and original == opt["unweighted_development_trace"],
        "observed development traces differ",
    )
    selected = min(trace, key=lambda row: row["validation_rollout_mse"])["step"]
    _require(
        selected == opt["selected_step"],
        "selected checkpoint is not the first strict development minimum",
    )
    losses = [dict(step=0, full_training_loss=initial)] + [
        dict(step=row["step"], full_training_loss=row["full_training_loss"])
        for row in checkpoints[1:]
    ]
    expected_safeguard = core.safeguard_metadata(
        steps=1000,
        accepted_scales=scales,
        objective_calls=calls,
        initial_loss=initial,
        final_loss=current,
        checkpoint_losses=losses,
        selected_step=selected,
    )
    _require(
        opt["safeguard"] == expected_safeguard, "observed safeguard report differs"
    )
    _require(
        opt["gradient"]
        == core.gradient_metadata(
            1536,
            attempts_started=1000,
            gradient_proposal_calls_returned=1000,
            completed_acceptance_attempts=1000,
        ),
        "observed gradient work report differs",
    )
    for row in checkpoints:
        arrays = _npz(Path(directory) / f"checkpoint-{row['step']:04d}.npz")
        _require(
            array_fingerprint({}, arrays) == row["array_fingerprint"],
            "checkpoint fingerprint differs",
        )
        _require(
            set(arrays) == set(reference.initial_arrays)
            and all(
                v.dtype == np.dtype("float64")
                and v.shape == reference.initial_arrays[k].shape
                and np.isfinite(v).all()
                for k, v in arrays.items()
            ),
            "checkpoint array schema differs",
        )
        if row["step"] in (0, selected):
            _same_tree(
                arrays,
                reference.initial_arrays
                if row["step"] == 0
                else reference.selected_arrays,
                "observed checkpoint",
            )
    initial_arrays = _npz(Path(directory) / "initializer-return.npz")
    _same_tree(initial_arrays, reference.initial_arrays, "actual initializer return")
    _same_tree(
        _npz(Path(directory) / "weight-initial.npz"),
        initial_arrays,
        "actual weighting initial tree",
    )
    weights = _npz(Path(directory) / "weights.npz")
    _same_tree(
        weights,
        {name: reference.objective_arrays[name] for name in WEIGHTS},
        "actual weighting witness",
    )
    entered = _read(Path(directory) / "initializer-entered.json")
    _require(
        entered
        == dict(
            jax_enable_x64=True,
            training_windows=1536,
            ridge=reference.ridge,
            settings=dict(seed=0, width=32, memory=8, delay_steps=reference.delay),
        ),
        "actual initializer scope/settings differ",
    )
    return {
        **_prefix(directory),
        "selected_step": selected,
        "scope": "Actual source-bound arrays, rows and scalar acceptance/selection chain; no replay of gradients, Adam, initialization or forecasts.",
    }


def _worker(
    simulator,
    directory,
    *,
    implementation_manifest,
    implementation_sha256,
    reference_root,
):
    directory = Path(directory)
    started = time.perf_counter()
    before = _configuration()
    outcome = dict(
        format=FORMAT,
        simulator=simulator,
        status="failed",
        stage="binding",
        configuration_before=before,
    )
    timing = dict(
        public_fit_calls=0,
        training_entry_calls=0,
        fitter_calls=0,
        calibration_calls=0,
        initializer_calls=0,
        scope="Measured wall times include read-only capture; JIT compilation remains included in fitter time, not separately estimated.",
    )
    try:
        _require(
            not before["jax_enable_x64"],
            "public flight worker must begin in ambient default32",
        )
        bound = _binding(implementation_manifest, implementation_sha256)
        outcome["implementation_commit"] = bound["implementation_commit"]
        outcome["stage"] = "preparation"
        preparation_started = time.perf_counter()
        reference = _prepare(reference_root, simulator)
        save_recordings(reference.collection, directory / "recordings.npz")
        expected_meta, expected_arrays = _expected_preparation(reference)
        save_arrays(
            directory / "reference-preparation.npz", expected_meta, expected_arrays
        )
        _write(directory / "reference.json", reference.source)
        timing["reference_preparation_wall_s"] = (
            time.perf_counter() - preparation_started
        )
        original_train, original_fit = learner._train, learner.fit_sequence_model
        original_init, original_calibration = (
            core.initialize_sequence_model,
            learner._calibrate,
        )

        def observed_train(train, development, contract, seen, **kwargs):
            timing["training_entry_calls"] += 1
            _require(
                timing["training_entry_calls"] == 1, "multiple public training entries"
            )
            actual = _cache_payload(train, development, contract, seen)
            save_arrays(directory / "actual-preparation.npz", *actual)
            _compare_preparation(actual, reference)
            return original_train(train, development, contract, seen, **kwargs)

        def observed_init(batch, **kwargs):
            timing["initializer_calls"] += 1
            _require(timing["initializer_calls"] == 1, "multiple initializers")
            _write(
                directory / "initializer-entered.json",
                dict(
                    jax_enable_x64=bool(jax.config.x64_enabled),
                    training_windows=len(batch.past_states),
                    ridge=kwargs["ridge"],
                    settings={
                        k: kwargs[k] for k in ("seed", "width", "memory", "delay_steps")
                    },
                ),
            )
            model = original_init(batch, **kwargs)
            np.savez_compressed(directory / "initializer-return.npz", **model.arrays())
            return model

        def observed_fit(*args, **kwargs):
            timing["fitter_calls"] += 1
            _require(timing["fitter_calls"] == 1, "multiple numerical fitters")
            entered = time.perf_counter()
            timing["public_preparation_and_capture_wall_s"] = entered - public_started
            try:
                return original_fit(*args, **kwargs)
            finally:
                timing["fitter_including_compile_and_capture_wall_s"] = (
                    time.perf_counter() - entered
                )

        def observed_calibration(*args, **kwargs):
            timing["calibration_calls"] += 1
            _require(timing["calibration_calls"] == 1, "multiple calibrations")
            entered = time.perf_counter()
            try:
                return original_calibration(*args, **kwargs)
            finally:
                timing["calibration_wall_s"] = time.perf_counter() - entered

        outcome["stage"] = "public_fit"
        public_started = time.perf_counter()
        with (directory / "work.jsonl").open("x") as work:

            def observe(state):
                if state["phase"] == "weights":
                    path = directory / "weight-initial.npz"
                    _require(not path.exists(), "weight initial witness already exists")
                    np.savez_compressed(path, **_tree(state["params"], state["norms"]))
                lifecycle._capture_event(state, directory, work)

            with (
                patch.object(learner, "_train", observed_train),
                patch.object(learner, "fit_sequence_model", observed_fit),
                patch.object(core, "initialize_sequence_model", observed_init),
                patch.object(core, "_observe_attempt", observe),
                patch.object(learner, "_calibrate", observed_calibration),
            ):
                timing["public_fit_calls"] += 1
                model = glassbox.fit(reference.collection)
        timing["public_call_wall_s"] = time.perf_counter() - public_started
        outcome["public_fit_returned"] = True
        model.save(directory / "model.npz")
        _write(directory / "report.json", model.report)
        outcome["stage"] = "verification"
        parity = _compare_model(model, reference)
        observed = _work(directory, model, reference)
        _write(directory / "parity.json", dict(model=parity, work=observed))
        _require(
            _configuration() == before,
            "public fit changed ambient precision/environment",
        )
        _binding(implementation_manifest, implementation_sha256)
        outcome.update(
            status="complete", stage="complete", model_fingerprint=model.fingerprint()
        )
    except BaseException as error:
        outcome.update(
            error_type=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc(),
        )
    finally:
        if "public_started" in locals():
            timing.setdefault(
                "public_call_wall_s", time.perf_counter() - public_started
            )
        outcome.update(
            configuration_after=_configuration(),
            elapsed_s=time.perf_counter() - started,
            timing=timing,
            observed_prefix=_prefix(directory, incomplete=True),
        )
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        outcome["peak_rss"] = dict(
            native=peak,
            native_units="bytes" if sys.platform == "darwin" else "KiB",
            bytes=int(peak if sys.platform == "darwin" else 1024 * peak),
            scope="whole worker, including reference preparation, one fit/capture/calibration and verification",
        )
        _write(directory / "outcome.json", outcome)
    return outcome


def _inventory(directory):
    directory = Path(directory)
    files = {}
    for path in sorted(directory.rglob("*")):
        _require(not path.is_symlink(), "evidence contains a symlink")
        if path == directory / "manifest.json":
            _require(path.is_file(), "manifest is not a regular file")
        elif path.is_file():
            files[path.relative_to(directory).as_posix()] = _sha(path)
        else:
            _require(path.is_dir(), "nonregular evidence file")
    return files


def _stage_status(directory, execution):
    if execution["hard_timeout"]:
        return "incomplete_hard_timeout"
    path = Path(directory) / "outcome.json"
    if not path.exists():
        return "incomplete_worker_exit"
    try:
        outcome = _read(path)
    except (ValueError, UnicodeDecodeError):
        return "incomplete_worker_exit"
    _require(outcome["status"] in ("complete", "failed"), "invalid worker outcome")
    return (
        "complete"
        if execution["exit_code"] == 0 and outcome["status"] == "complete"
        else "failed"
    )


def run(
    simulator, output, *, implementation_manifest, implementation_sha256, reference_root
):
    """Supervise exactly one fresh worker; never retry or reuse its output."""
    _require(simulator in ("crazyflow", "cascade"), "unknown simulator")
    bound = _binding(implementation_manifest, implementation_sha256)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    implementation_manifest, reference_root = (
        str(Path(implementation_manifest).resolve()),
        str(Path(reference_root).resolve()),
    )
    command = [
        bound["interpreter"],
        "-m",
        MODULE,
        "worker",
        "--simulator",
        simulator,
        "--output",
        str(output),
        "--implementation-manifest",
        implementation_manifest,
        "--implementation-sha256",
        implementation_sha256,
        "--reference-root",
        reference_root,
    ]
    env = dict(os.environ)
    env.pop("JAX_ENABLE_X64", None)
    env.update(SCIPY_ARRAY_API="1", PYTHONPATH=str(Path(bound["public_root"]) / "src"))
    _write(
        output / "command.json",
        dict(
            argv=command,
            cwd=bound["public_root"],
            environment={
                k: env.get(k)
                for k in ("PYTHONPATH", "SCIPY_ARRAY_API", "JAX_ENABLE_X64", "PATH")
            },
            hard_wall_time_s=HARD_WALL_TIME_S,
            implementation_manifest_sha256=implementation_sha256,
        ),
    )
    started = time.perf_counter()
    timed_out = False
    launch_error = None
    exit_code = None
    with (output / "stdout-stderr.log").open("x") as log:
        try:
            process = subprocess.Popen(
                command,
                cwd=bound["public_root"],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                exit_code = process.wait(timeout=HARD_WALL_TIME_S)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(process.pid, signal.SIGKILL)
                exit_code = process.wait()
        except OSError as error:
            launch_error = dict(type=type(error).__name__, message=str(error))
    execution = dict(
        exit_code=exit_code,
        hard_timeout=timed_out,
        elapsed_s=time.perf_counter() - started,
        hard_wall_time_s=HARD_WALL_TIME_S,
        launch_error=launch_error,
    )
    _write(output / "execution.json", execution)
    status = _stage_status(output, execution)
    manifest = dict(
        format=FORMAT,
        protocol_sha256=PROTOCOL_SHA256,
        simulator=simulator,
        status=status,
        implementation_manifest_sha256=implementation_sha256,
        implementation_commit=bound["implementation_commit"],
        reference_bundle_sha256=REFERENCE_SHA256,
        observed_prefix=_prefix(output, incomplete=status != "complete"),
        files=_inventory(output),
    )
    _write(output / "manifest.json", manifest)
    return dict(
        status=status,
        manifest=str(output / "manifest.json"),
        manifest_sha256=_sha(output / "manifest.json"),
    )


def verify(
    output,
    *,
    expected_manifest_sha256,
    implementation_manifest,
    implementation_sha256,
    reference_root,
):
    """Authenticate and reduce saved evidence without any new numerical forecast."""
    bound = _binding(implementation_manifest, implementation_sha256)
    output = Path(output)
    _require(
        _sha(output / "manifest.json") == expected_manifest_sha256,
        "flight evidence external SHA differs",
    )
    manifest = _read(output / "manifest.json")
    _require(
        manifest["format"] == FORMAT
        and manifest["protocol_sha256"] == PROTOCOL_SHA256
        and manifest["implementation_manifest_sha256"] == implementation_sha256
        and manifest["implementation_commit"] == bound["implementation_commit"]
        and manifest["reference_bundle_sha256"] == REFERENCE_SHA256,
        "flight evidence identity differs",
    )
    _require(
        manifest["files"] == _inventory(output),
        "flight evidence file integrity differs",
    )
    incomplete = manifest["status"] != "complete"
    prefix = _prefix(output, incomplete=incomplete)
    _require(prefix == manifest["observed_prefix"], "flight observed prefix differs")
    execution = _read(output / "execution.json")
    _require(
        execution["hard_wall_time_s"] == HARD_WALL_TIME_S,
        "hard supervisor budget differs",
    )
    _require(
        manifest["status"] == _stage_status(output, execution),
        "worker outcome classification differs",
    )
    if manifest["status"] != "complete":
        _require(
            manifest["status"]
            in ("failed", "incomplete_hard_timeout", "incomplete_worker_exit"),
            "invalid failure status",
        )
        _require(
            (manifest["status"] == "incomplete_hard_timeout")
            == execution["hard_timeout"],
            "hard timeout classification differs",
        )
        return dict(
            status=manifest["status"],
            observed_prefix=prefix,
            scope="Authenticated failed/incomplete prefix only; no qualified returned model is claimed.",
        )
    _require(
        execution["exit_code"] == 0 and execution["hard_timeout"] is False,
        "completed worker exit differs",
    )
    outcome = _read(output / "outcome.json")
    _require(
        outcome["status"] == outcome["stage"] == "complete"
        and outcome["configuration_before"] == outcome["configuration_after"]
        and outcome["configuration_before"]["jax_enable_x64"] is False,
        "completed outcome/precision differs",
    )
    _require(
        all(
            outcome["timing"][key] == 1
            for key in (
                "public_fit_calls",
                "training_entry_calls",
                "fitter_calls",
                "initializer_calls",
                "calibration_calls",
            )
        ),
        "single actual fit/initializer/calibration count differs",
    )
    reference = _prepare(reference_root, manifest["simulator"])
    _require(
        _read(output / "reference.json") == reference.source,
        "reference source links differ",
    )
    for name in ("reference-preparation.npz", "actual-preparation.npz"):
        _compare_preparation(load_arrays(output / name), reference)
    recordings = load_recordings(output / "recordings.npz")
    _same_recordings(recordings, reference.collection)
    model = glassbox.LearnedDynamics.load(output / "model.npz")
    _require(
        model.fingerprint() == outcome["model_fingerprint"]
        and model.report == _read(output / "report.json"),
        "saved public revision/report differs",
    )
    parity = dict(
        model=_compare_model(model, reference), work=_work(output, model, reference)
    )
    _require(parity == _read(output / "parity.json"), "saved parity evidence differs")
    return dict(
        status="complete",
        simulator=manifest["simulator"],
        model_fingerprint=model.fingerprint(),
        parity=parity,
        scope="Read-only source/data/witness replay; no initialization, solve, fit, gradient or model forecast.",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("run", "worker", "verify"))
    parser.add_argument("--simulator", choices=("crazyflow", "cascade"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--implementation-manifest", required=True)
    parser.add_argument("--implementation-sha256", required=True)
    parser.add_argument("--reference-root", required=True)
    parser.add_argument("--expected-manifest-sha256")
    args = parser.parse_args(argv)
    shared = dict(
        implementation_manifest=args.implementation_manifest,
        implementation_sha256=args.implementation_sha256,
        reference_root=args.reference_root,
    )
    if args.operation == "verify":
        _require(
            args.expected_manifest_sha256 is not None,
            "verification requires an external manifest SHA",
        )
        result = verify(
            args.output,
            expected_manifest_sha256=args.expected_manifest_sha256,
            **shared,
        )
    else:
        _require(args.simulator is not None, "fitting requires a simulator routing ID")
        result = (run if args.operation == "run" else _worker)(
            args.simulator, args.output, **shared
        )
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    if result["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
