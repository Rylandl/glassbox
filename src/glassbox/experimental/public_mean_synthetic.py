"""The fixed synthetic tier of public-mean-qualification-v1.

Historical generation and literal legacy flags run only in the bound old worker.
The current worker fits each of the 27 cases once through ``glassbox.fit``. Saved
default32 forecasts use fixed old384 units for acceptance; x64 and the old
candidate-scaled flags remain separate diagnostics. Replay never refits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import jax
import numpy as np

import glassbox
from glassbox import _sequence_model as core
from glassbox import learner
from glassbox._learner_arrays import array_fingerprint
from glassbox.io.recordings import load_recordings

from . import public_mean_implementation as implementation
from . import public_mean_scoring as scoring
from .public_mean_lifecycle import _capture_event, _configuration, _work_prefix

MODULE = "glassbox.experimental.public_mean_synthetic"
ARRAYS = ("past_states", "past_inputs", "future_inputs", "future_states")
QUERY_ARRAYS = (*ARRAYS[:3], "targets", "recording_ids", "source_origins")


def _require(value, message):
    if not value:
        raise ValueError(message)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text())


def _json(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def _same(a, b, label):
    a, b = np.asarray(a), np.asarray(b)
    _require(
        a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(), label
    )


def _payloads(root, *, exclude=()):
    root = Path(root)
    result = {}
    for path in sorted(root.rglob("*")):
        _require(not path.is_symlink(), "synthetic evidence cannot use symlinks")
        if path.is_file() and str(path.relative_to(root)) not in exclude:
            result[str(path.relative_to(root))] = _sha(path)
    return result


def _bound(binding_path, expected_sha256):
    binding = implementation.verify(binding_path, expected_sha256)
    actual = Path(__file__).resolve()
    expected = (
        Path(binding["public_root"])
        / "src/glassbox/experimental/public_mean_synthetic.py"
    )
    _require(actual == expected, "synthetic runner imported from another checkout")
    _require(
        binding["public_source_sha256"].get(
            str(actual.relative_to(binding["public_root"]))
        )
        == _sha(actual),
        "synthetic runner absent from committed binding",
    )
    return binding


# Execute this script, never this module, in the old worker. Nothing imports new
# learner code into that interpreter. Every actual Glassbox import is attested.
_HISTORICAL_WORKER = r"""
import hashlib, importlib.metadata, json, pathlib, sys
import numpy as np

def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()

request_path, expected_sha = sys.argv[1:]
assert sha(request_path) == expected_sha, "historical request anchor"
request = json.loads(pathlib.Path(request_path).read_text())
assert sha(request["binding_path"]) == request["binding_sha256"], "implementation anchor"
binding = json.loads(pathlib.Path(request["binding_path"]).read_text())
assert binding["oracle_commit"] == "8b61830c9c353dbb25edc6e63a76886d0ce9b9d3"
root = pathlib.Path(binding["oracle_root"]).resolve()
assert pathlib.Path(sys.executable).resolve() == pathlib.Path(binding["interpreter"]).resolve()
assert sha(pathlib.Path(sys.executable).resolve()) == binding["interpreter_sha256"]
runtime = {"python":".".join(map(str,sys.version_info[:3])), **{n:importlib.metadata.version(n) for n in ("jax","jaxlib","numpy","scipy")}}
assert runtime == binding["runtime"], "historical runtime"
import glassbox
from glassbox import learner, _sequence_model
from glassbox.experimental import harness
from glassbox.io.recordings import save_recordings

def forbidden(*args, **kwargs):
    raise AssertionError("historical numerical fitting is forbidden")

glassbox.fit = learner.fit = harness.fit = forbidden
learner.fit_sequence_model = _sequence_model.fit_sequence_model = forbidden
learner.initialize_sequence_model = _sequence_model.initialize_sequence_model = forbidden

def imports():
    result = {}
    for name, module in sorted(sys.modules.items()):
        if name != "glassbox" and not name.startswith("glassbox."):
            continue
        raw = getattr(module, "__file__", None)
        if raw is None:
            continue
        path = pathlib.Path(raw).resolve()
        assert path.is_relative_to(root), (name, "historical import escaped source root")
        relative = str(path.relative_to(root))
        assert relative in binding["oracle_source_sha256"], (name, "unbound historical module")
        digest = sha(path)
        assert digest == binding["oracle_source_sha256"][relative], (name, "historical source differs")
        result[name] = {"path":str(path),"sha256":digest}
    return result

observed = imports()
artifact_root = pathlib.Path(request["artifact_root"])
anchors = request["reference_files"]
for relative, digest in anchors.items():
    assert sha(artifact_root/relative) == digest, (relative, "historical artifact anchor")
manifest = harness.frozen_manifest(artifact_root/"slow-sampling/manifest.json")
output = pathlib.Path(request["output"])
output.mkdir(parents=True,exist_ok=False)
def write(path, value):
    with path.open("x") as handle:
        handle.write(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n")

if request["stage"] == "generate":
    cases=[]
    for case in request["cases"]:
        family, seed, name = case["family"], case["data_seed"], case["name"]
        directory=output/name
        directory.mkdir()
        witness=family==harness.WITNESS
        calibration=harness.witness_recordings(manifest["dataset"],seed) if witness else harness.generate(manifest["dataset"],family,seed,"calibration")
        save_recordings(calibration,directory/"calibration.npz")
        for regime in case["regimes"]:
            supplied=harness.witness_recordings(manifest["dataset"],seed,evaluation=True) if witness else harness.generate(manifest["dataset"],family,seed,regime)
            save_recordings(supplied,directory/(regime+"-recordings.npz"))
            arrays=harness.evaluation_rows(supplied,manifest["dataset"],dict(history=10,horizon=5,delay=2))
            np.savez_compressed(directory/(regime+"-queries.npz"),**arrays)
        if witness:
            x,up,uf,target=harness.witness_probe(10,5)
            np.savez_compressed(directory/"probe-queries.npz",past_states=x,past_inputs=up,future_inputs=uf,targets=target)
        cases.append(name)
    result={"status":"complete","cases":cases,"numerical_fits":0,"generation":"unchanged source-pinned historical generators"}
elif request["stage"] == "decide":
    rows_by_precision=json.loads(pathlib.Path(request["rows_path"]).read_text())
    assert sha(request["rows_path"])==request["rows_sha256"]
    reference_path=artifact_root/"slow-sampling/reference.json"
    evidence_path=artifact_root/"slow-sampling/evidence-manifest.json"
    evidence_reference_path=artifact_root/"slow-sampling/evidence-reference.json"
    reference=harness.read(reference_path)
    evidence_manifest=harness.frozen_evidence_manifest(evidence_path)
    evidence_reference=harness.read(evidence_reference_path)
    result={"status":"complete","precisions":{}}
    for precision, rows in rows_by_precision.items():
        mean=harness.decide(manifest,rows,reference,sha(reference_path))
        coverage=harness.evidence_decide(evidence_manifest,"synthetic",harness.evidence_table(rows,"name"),harness.synthetic_evidence_cases(manifest),evidence_reference,sha(evidence_reference_path))
        combined=harness.with_evidence(json.loads(json.dumps(mean)),coverage)
        result["precisions"][precision]={"legacy_candidate_scaled_mean":mean,"historical_coverage":coverage,"historical_with_evidence":combined,"required_new_capability_gate":False}
else:
    raise ValueError("unknown historical stage")
result["imports"]=imports()
assert result["imports"]==observed
result["runtime"]=runtime
result["oracle_commit"]=binding["oracle_commit"]
result["binding_sha256"]=request["binding_sha256"]
result["request_sha256"]=expected_sha
result["files"]={str(p.relative_to(output)):sha(p) for p in sorted(output.rglob("*")) if p.is_file()}
write(output/"result.json",result)
"""


def _launch(argv, cwd, environment, stem):
    stem = Path(stem)
    _json(
        Path(str(stem) + ".command.json"),
        {
            "argv": argv,
            "cwd": str(cwd),
            "environment": {
                k: environment.get(k)
                for k in ("PYTHONPATH", "JAX_ENABLE_X64", "SCIPY_ARRAY_API")
            },
            "hard_timeout_s": 14400,
        },
    )
    started = time.perf_counter()
    with Path(str(stem) + ".log").open("x") as log:
        try:
            process = subprocess.run(
                argv,
                cwd=cwd,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=14400,
                check=False,
            )
            result = {
                "returncode": process.returncode,
                "status": "complete" if process.returncode == 0 else "worker_failed",
            }
        except subprocess.TimeoutExpired:
            result = {"returncode": None, "status": "hard_timeout_incomplete"}
    result.update(
        elapsed_s=time.perf_counter() - started,
        log_sha256=_sha(Path(str(stem) + ".log")),
    )
    _json(Path(str(stem) + ".exit.json"), result)
    return result


def _historical(
    stage,
    destination,
    *,
    binding_path,
    binding_sha256,
    artifact_root,
    cases=None,
    rows_path=None,
):
    binding = _bound(binding_path, binding_sha256)
    _, ledger = scoring.specification(binding["public_root"])
    destination = Path(destination)
    request = {
        "stage": stage,
        "output": str(destination.resolve()),
        "binding_path": str(Path(binding_path).resolve()),
        "binding_sha256": binding_sha256,
        "artifact_root": str(Path(artifact_root).resolve()),
        "reference_files": ledger["source_sha256"],
    }
    if cases is not None:
        request["cases"] = cases
    if rows_path is not None:
        request.update(
            rows_path=str(Path(rows_path).resolve()), rows_sha256=_sha(rows_path)
        )
    stem = destination.parent / (destination.name + "-worker")
    script, request_path = Path(str(stem) + ".py"), Path(str(stem) + ".request.json")
    with script.open("x") as handle:
        handle.write(_HISTORICAL_WORKER)
    _json(request_path, request)
    environment = dict(os.environ)
    environment.update(
        PYTHONPATH=str(Path(binding["oracle_root"]) / "src"),
        JAX_ENABLE_X64="1",
        SCIPY_ARRAY_API="1",
    )
    argv = [
        binding["interpreter"],
        str(script.resolve()),
        str(request_path.resolve()),
        _sha(request_path),
    ]
    status = _launch(argv, binding["oracle_root"], environment, stem)
    _require(status["returncode"] == 0, "historical worker failed; attempt retained")
    result = _read(destination / "result.json")
    _require(
        result["status"] == "complete"
        and result["binding_sha256"] == binding_sha256
        and result["request_sha256"] == _sha(request_path),
        "historical output provenance",
    )
    _require(
        _payloads(destination, exclude=("result.json",)) == result["files"],
        "historical output payloads",
    )
    _bound(binding_path, binding_sha256)
    return result


def _window_state(windows):
    return {
        "keys": [asdict(k) for k in windows.keys],
        "source_origins": list(windows.source_origins),
        "dt_s": windows.batch.dt_s,
        "arrays_fingerprint": array_fingerprint(
            {}, {k: getattr(windows.batch, k) for k in ARRAYS}
        ),
    }


def _verify_data(directory, expected_sha256, binding_sha256, cases):
    directory = Path(directory)
    _require(
        _sha(directory / "result.json") == expected_sha256,
        "external generated-data anchor",
    )
    result = _read(directory / "result.json")
    _require(
        result["status"] == "complete" and result["binding_sha256"] == binding_sha256,
        "generated-data source binding",
    )
    _require(
        result["cases"] == [case["name"] for case in cases]
        and result["numerical_fits"] == 0,
        "generated-data case/fit roster",
    )
    _require(
        _payloads(directory, exclude=("result.json",)) == result["files"],
        "generated-data payload integrity",
    )


def preparation(calibration_path, historical_archive):
    """Prove the new extraction preserves old units/support before any fit."""
    supplied = load_recordings(calibration_path)
    _require(
        len(supplied.segments) == 8
        and len({s.recording_id for s in supplied.segments}) == 8,
        "exact eight calibration parents",
    )
    _require(
        all(
            len(s.states) == 161
            and len(s.inputs) == 160
            and s.dt_s == 0.05
            and s.start_row == 0
            and s.segment_id == "whole"
            for s in supplied.segments
        ),
        "unchanged synthetic collection budget",
    )
    seen = learner._recording_content(supplied)
    names = sorted(
        seen,
        key=lambda s: hashlib.sha256(
            json.dumps(s, sort_keys=True).encode()
        ).hexdigest(),
    )
    roles = {"development": names[:2], "train": names[2:]}
    train = learner._extract(supplied, roles["train"], 1536)
    development = learner._extract(supplied, roles["development"], 256)
    _require(
        len(train.keys) == 876 and len(development.keys) == 256,
        "actual876/dev256 support",
    )
    saved = _arrays(historical_archive)
    metadata = json.loads(str(saved.pop("metadata")))
    _require(
        metadata["seen"] == seen
        and metadata["contract"] == learner._contract(supplied),
        "historical generation/ledger/contract identity",
    )
    _require(metadata["model"]["dt_s"] == 0.05, "historical sampling interval")
    for role, windows, count in (
        ("train", train, 384),
        ("development", development, 256),
    ):
        declared = metadata["windows"][role]
        _require(
            [asdict(k) for k in windows.keys[:count]] == declared["keys"],
            "old384/dev key identity",
        )
        _require(
            list(windows.source_origins[:count]) == declared["source_origins"],
            "old384/dev source origins",
        )
        for key in ARRAYS:
            _same(
                getattr(windows.batch, key)[:count],
                saved[f"{role}_{key}"],
                "old384/dev cache array " + key,
            )
    return (
        supplied,
        train,
        development,
        {
            "training_windows": 876,
            "development_windows": 256,
            "calibration_recordings": 8,
            "roles": roles,
            "seen": seen,
            "train": _window_state(train),
            "development": _window_state(development),
            "old_model_sha256": _sha(historical_archive),
            "old_training_prefix": 384,
            "old_reference_state_scale": saved["norm_state_scale"].tolist(),
        },
    )


def _check_model(model, train, development):
    for role, expected in (("train", train), ("development", development)):
        actual = getattr(model, f"_{role}")
        _require(
            _window_state(actual) == _window_state(expected),
            "public fitted cache differs from checked preparation",
        )
        for name in ARRAYS:
            _same(
                getattr(actual.batch, name),
                getattr(expected.batch, name),
                "fitted ordered cache " + name,
            )
    opt = model.report["optimization"]
    _require(
        model.recipe["id"] == "generic-memory-v4-prototype" and opt["steps"] == 1000,
        "one frozen public recipe/budget",
    )
    _require(
        opt["batch_size"] == 876 and opt["ridge"] == 0.01 * 876 * 5,
        "actual synthetic batch/ridge",
    )
    gradient = opt["gradient"]
    for name in (
        "attempts_started",
        "gradient_proposal_calls_returned",
        "completed_acceptance_attempts",
    ):
        _require(gradient[name] == 1000, "complete synthetic gradient calls")
    _require(
        gradient["training_windows"] == 876
        and gradient["known_gradient_window_visits"] == 876000
        and not gradient["incomplete_gradient_work_unknown"],
        "actual synthetic gradient work",
    )
    _require(
        gradient["policy"] == "ordered_full_cache"
        and gradient["sampling_seed"] is None,
        "fixed full-cache policy",
    )
    _require(
        [r["step"] for r in opt["trace"]] == list(range(0, 1001, 100)),
        "synthetic checkpoint roster",
    )
    _require(
        opt["selected_step"]
        == min(opt["trace"], key=lambda r: r["validation_rollout_mse"])["step"],
        "first strict development selector",
    )
    _require(
        opt["safeguard"]["completed_attempts"] == 1000
        and 1 <= opt["safeguard"]["full_training_objective_calls"] <= 8001,
        "synthetic acceptance work",
    )


def _verify_work(directory, model):
    """Link returned call counts and selected arrays, without optimizer replay."""
    directory = Path(directory)
    prefix = _work_prefix(directory)
    counts = prefix["phase_counts"]
    _require(
        counts.get("initialized")
        == counts.get("weights")
        == counts.get("initial_objective")
        == 1,
        "one actual synthetic initialization/objective",
    )
    _require(
        all(counts.get(name) == 1000 for name in ("started", "proposed", "completed"))
        and counts.get("checkpoint") == 11,
        "actual synthetic work/checkpoint roster",
    )
    _require(
        prefix["known_gradient_window_visits"] == 876000
        and not prefix["incomplete_gradient_work_unknown"],
        "observed synthetic gradient work",
    )
    opt = model.report["optimization"]
    _require(
        prefix["full_training_objective_calls_returned"]
        == opt["safeguard"]["full_training_objective_calls"],
        "actual synthetic objective call count",
    )
    events = [
        json.loads(line) for line in (directory / "work.jsonl").read_text().splitlines()
    ]
    checkpoints = [row for row in events if row["phase"] == "checkpoint"]
    _require(
        [{k: r[k] for k in ("step", "validation_rollout_mse")} for r in checkpoints]
        == [
            {k: r[k] for k in ("step", "validation_rollout_mse")} for r in opt["trace"]
        ],
        "actual synthetic development trace",
    )
    for row in events:
        if row["phase"] == "started":
            _require(
                row["indices_dtype"] == np.dtype(np.int64).str
                and row["indices"] == list(range(876))
                and all(type(i) is int for i in row["indices"]),
                "actual ordered synthetic gradient indices",
            )
    initial = _arrays(directory / "checkpoint-0000.npz")
    _require(
        array_fingerprint({}, initial)
        == opt["objective"]["initial_parameters_and_norms_fingerprint"],
        "actual initial synthetic tree",
    )
    selected = _arrays(directory / f"checkpoint-{opt['selected_step']:04d}.npz")
    _require(
        set(selected) == set(model._model.arrays()), "selected synthetic tree roster"
    )
    for name, values in model._model.arrays().items():
        _same(selected[name], values, "selected synthetic mean " + name)
    weights = _arrays(directory / "weights.npz")
    for name in (
        "initial_channel_mse",
        "raw_channel_weights",
        "channel_weights",
        "weight_floor",
        "weight_normalizer",
    ):
        _same(
            weights[name],
            np.asarray(opt["objective"][name]),
            "actual synthetic objective " + name,
        )
    _same(
        weights["fixed_weights"],
        weights["channel_weights"],
        "fixed synthetic optimizer weights",
    )
    _same(
        weights["normalization"],
        np.asarray(opt["error_scale"]),
        "actual synthetic hold normalization",
    )
    return prefix


def _coverage(prediction, target, half_width):
    prediction, target, half_width = (
        np.asarray(a, dtype=np.float64) for a in (prediction, target, half_width)
    )
    _require(
        prediction.shape == target.shape == half_width.shape
        and np.isfinite(half_width).all()
        and np.all(half_width >= 0),
        "coverage envelope shape/finiteness",
    )
    covered = np.abs(prediction - target) <= half_width
    return {
        "scored_rows": len(prediction),
        "horizon_steps": prediction.shape[1],
        "channels": prediction.shape[2],
        "coverage": {
            f"channel_{i}": covered[:, :, i].mean(axis=0).tolist()
            for i in range(prediction.shape[2])
        },
        "pooled_coverage": {
            f"channel_{i}": float(covered[:, :, i].mean())
            for i in range(prediction.shape[2])
        },
    }


def _legacy_metrics(scores, count):
    def convert(row):
        return {
            "horizon_scaled_rmse": row["horizon_rmse"],
            "overall_scaled_rmse": row["overall_rmse"],
            "channel_rmse": row["channel_rmse"],
        }

    return {
        "windows": count,
        **convert(scores["legacy_candidate_scaled"]),
        "recordings": {
            name: convert(value["legacy_candidate_scaled"])
            for name, value in scores["recordings"].items()
        },
    }


def _validate_queries(arrays, reference_path):
    _require(set(arrays) == set(QUERY_ARRAYS), "complete synthetic query roster")
    previous = _arrays(reference_path)
    for name in QUERY_ARRAYS:
        _same(
            arrays[name], previous[name], "unchanged historical evaluation tape " + name
        )
    _require(
        arrays["past_states"].shape[1] == 11 and arrays["future_inputs"].shape[1] == 5,
        "fixed synthetic information budget",
    )


def _evaluate(
    model, case, data_directory, output, reference_scale, artifact_root, precision
):
    output = Path(output)
    output.mkdir()
    candidate_scale = np.asarray(model._model.norms["state_scale"])
    rows, evidence, probes = [], {}, []
    legacy = {
        "name": case["name"],
        "family": case["family"],
        "data_seed": case["data_seed"],
        "status": "complete",
        "regimes": {},
        "evidence": evidence,
    }
    with jax.enable_x64(precision == "float64"):
        for regime in case["regimes"]:
            arrays = _arrays(Path(data_directory) / f"{regime}-queries.npz")
            _validate_queries(
                arrays,
                Path(artifact_root) / "slow-sampling" / case["name"] / f"{regime}.npz",
            )
            _require(
                not set(arrays["recording_ids"]) & set(model._seen),
                "evaluation/fit recording identities disjoint",
            )
            prediction = np.asarray(model.predict(*(arrays[k] for k in ARRAYS[:3])))
            half_width = np.broadcast_to(model.envelope(), prediction.shape).copy()
            np.savez_compressed(
                output / f"{regime}.npz",
                **arrays,
                prediction=prediction,
                envelope_half_width=half_width,
            )
            _require(
                prediction.dtype == np.dtype(precision),
                "actual public inference precision",
            )
            score = scoring.forecast_scores(
                prediction,
                arrays["targets"],
                reference_scale,
                candidate_scale,
                arrays["recording_ids"].tolist(),
            )
            rows.append({"case": case["name"], "regime": regime, "scores": score})
            evidence[regime] = _coverage(prediction, arrays["targets"], half_width)
            legacy["regimes"][regime] = _legacy_metrics(score, len(prediction))
        if case["family"] == "hidden_input_delay":
            arrays = _arrays(Path(data_directory) / "probe-queries.npz")
            previous = _arrays(
                Path(artifact_root) / "slow-sampling" / case["name"] / "probe.npz"
            )
            _require(
                set(arrays) == {*ARRAYS[:3], "targets"}, "complete physical probe tape"
            )
            for name, values in arrays.items():
                _same(values, previous[name], "unchanged historical probe " + name)
            predicted = np.asarray(model.predict(*(arrays[k] for k in ARRAYS[:3])))
            np.savez_compressed(output / "probe.npz", **arrays, prediction=predicted)
            _require(predicted.dtype == np.dtype(precision), "actual probe precision")
            score = scoring.probe_scores(predicted, arrays["targets"], reference_scale)
            probes.append({"seed": case["data_seed"], "scores": score})
            legacy["probe"] = {
                "paired_rmse": score["physical_rmse"],
                "blind_floor": (
                    np.abs(arrays["targets"][1, :, 0] - arrays["targets"][0, :, 0]) / 2
                ).tolist(),
                "branches_identical": bool(np.array_equal(predicted[0], predicted[1])),
            }
    result = {
        "status": "complete",
        "precision": precision,
        "rows": rows,
        "probes": probes,
        "legacy": legacy,
        "coverage_is_diagnostic": True,
    }
    _json(output / "result.json", result)
    return result


def run_case(
    case_name,
    output,
    data_root,
    *,
    data_sha256,
    binding_path,
    binding_sha256,
    artifact_root,
):
    """Exactly one public fit; failures are records, never retry instructions."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "case": case_name,
        "status": "failed",
        "fit_complete": False,
        "fit_calls": 0,
        "precisions": {},
        "data_sha256": data_sha256,
    }
    started = time.perf_counter()
    before = _configuration()
    try:
        binding = _bound(binding_path, binding_sha256)
        _require(not jax.config.x64_enabled, "synthetic public worker starts default32")
        cases, ledger = scoring.specification(binding["public_root"])
        _verify_data(data_root, data_sha256, binding_sha256, cases)
        case = next(c for c in cases if c["name"] == case_name)
        scales = scoring.authenticate_reference_scales(
            binding["public_root"], artifact_root
        )
        historical = Path(artifact_root) / ledger["cases"][case_name]["source_path"]
        supplied, train, development, prep = preparation(
            Path(data_root) / case_name / "calibration.npz", historical
        )
        _json(output / "preparation.json", prep)
        timing = {}
        original_fit, original_calibrate = (
            learner.fit_sequence_model,
            learner._calibrate,
        )

        def timed_fit(*args, **kwargs):
            mark = time.perf_counter()
            timing["public_preparation_wall_s"] = mark - public_started
            try:
                return original_fit(*args, **kwargs)
            finally:
                timing["fit_including_compile_wall_s"] = time.perf_counter() - mark

        def timed_calibration(*args, **kwargs):
            mark = time.perf_counter()
            try:
                return original_calibrate(*args, **kwargs)
            finally:
                timing["calibration_wall_s"] = time.perf_counter() - mark

        with (output / "work.jsonl").open("x") as work:

            def observe(state):
                _capture_event(state, output, work)

            with (
                patch.object(core, "_observe_attempt", observe),
                patch.object(learner, "fit_sequence_model", timed_fit),
                patch.object(learner, "_calibrate", timed_calibration),
            ):
                result["fit_calls"] = 1
                public_started = time.perf_counter()
                try:
                    model = glassbox.fit(supplied)
                finally:
                    timing["public_call_wall_s"] = time.perf_counter() - public_started
                    result["timing"] = timing
        model.save(output / "model.npz")
        _json(output / "public-report.json", model.report)
        result.update(fit_complete=True, model_fingerprint=model.fingerprint())
        _check_model(model, train, development)
        _verify_work(output, model)
        for precision in ("float32", "float64"):
            try:
                result["precisions"][precision] = _evaluate(
                    model,
                    case,
                    Path(data_root) / case_name,
                    output / precision,
                    scales[case_name],
                    artifact_root,
                    precision,
                )
            except Exception as error:
                result["precisions"][precision] = {
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                }
                _json(
                    output / precision / "failure.json", result["precisions"][precision]
                )
        _require(
            _configuration() == before,
            "synthetic public precision/environment restoration",
        )
        _bound(binding_path, binding_sha256)
        _verify_data(data_root, data_sha256, binding_sha256, cases)
        result["status"] = (
            "complete"
            if all(r["status"] == "complete" for r in result["precisions"].values())
            else "prediction_failed"
        )
    except Exception as error:
        result.update(
            error_type=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc(),
        )
    finally:
        result.update(
            elapsed_s=time.perf_counter() - started,
            observed_prefix=_work_prefix(output),
            configuration_before=before,
            configuration_after=_configuration(),
            peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if sys.platform == "darwin" else 1024),
            peak_process_rss_scope="whole isolated synthetic case process including imports, preparation, fitting and both inference precisions",
            files=_payloads(output),
        )
        _json(output / "result.json", result)
    return result


def aggregate(repository, cases, results):
    """Keep the full case roster; missing/failed evidence cannot shrink a gate."""
    expected = [c["name"] for c in cases]
    _require(
        len(results) == 27 and [r["case"] for r in results] == expected,
        "exact ordered synthetic result roster",
    )
    _require(
        all(type(r.get("fit_complete")) is bool for r in results),
        "explicit synthetic fit completion status",
    )
    completed = [r["case"] for r in results if r.get("fit_complete")]
    legacy, decisions = {}, {}
    for precision in ("float32", "float64"):
        rows, probes, legacy_rows, unavailable = [], [], [], []
        for case, result in zip(cases, results, strict=True):
            scored = result.get("precisions", {}).get(precision, {})
            if scored.get("status") == "complete":
                rows.extend(scored["rows"])
                probes.extend(scored["probes"])
                legacy_rows.append(scored["legacy"])
            else:
                unavailable.append(case["name"])
                legacy_rows.append(
                    {
                        "name": case["name"],
                        "family": case["family"],
                        "data_seed": case["data_seed"],
                        "status": "failed",
                    }
                )
        try:
            decision = scoring.capability_decision(repository, completed, rows, probes)
        except ValueError as error:
            decision = {
                "fixed_reference_absolute_capability_pass": False,
                "incomplete_or_invalid_roster": str(error),
                "completed_fits": len(completed),
                "regime_count": len(rows),
                "probe_count": len(probes),
            }
        decision["unavailable_cases"] = unavailable
        decisions[precision] = decision
        legacy[precision] = legacy_rows
    return {
        "fixed_reference_absolute_capability_pass": decisions["float32"][
            "fixed_reference_absolute_capability_pass"
        ],
        "required_precision": "float32",
        "decisions": decisions,
        "completed_fits": completed,
        "failed_cases": [r["case"] for r in results if r["status"] != "complete"],
        "legacy_flags_diagnostic_only": True,
        "float64_is_a_diagnostic_not_a_fallback": True,
    }, legacy


def _request(binding_path, binding_sha256, artifact_root):
    return [
        "--binding",
        str(Path(binding_path).resolve()),
        "--binding-sha256",
        binding_sha256,
        "--artifact-root",
        str(Path(artifact_root).resolve()),
    ]


def run(output, *, binding_path, binding_sha256, artifact_root):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "status": "failed",
        "binding_sha256": binding_sha256,
        "case_results": [],
        "scientific_fit_budget": 27,
    }
    try:
        binding = _bound(binding_path, binding_sha256)
        cases, _ = scoring.specification(binding["public_root"])
        result["planned_case_names"] = [c["name"] for c in cases]
        scales = scoring.authenticate_reference_scales(
            binding["public_root"], artifact_root
        )
        _json(
            output / "intent.json",
            {
                "cases": cases,
                "fit_budget": 27,
                "case_order": [c["name"] for c in cases],
                "reference_scales": {k: v.tolist() for k, v in scales.items()},
                "binding_sha256": binding_sha256,
            },
        )
        _historical(
            "generate",
            output / "data",
            binding_path=binding_path,
            binding_sha256=binding_sha256,
            artifact_root=artifact_root,
            cases=cases,
        )
        data_sha256 = _sha(output / "data/result.json")
        result["generated_data_sha256"] = data_sha256
        (output / "cases").mkdir()
        environment = dict(os.environ)
        environment.pop("JAX_ENABLE_X64", None)
        environment.update(
            PYTHONPATH=str(Path(binding["public_root"]) / "src"), SCIPY_ARRAY_API="1"
        )
        for case in cases:
            name = case["name"]
            command = [
                binding["interpreter"],
                "-m",
                MODULE,
                "case",
                "--case",
                name,
                "--output",
                str(output / "cases" / name),
                "--data-root",
                str(output / "data"),
                "--data-sha256",
                data_sha256,
                *_request(binding_path, binding_sha256, artifact_root),
            ]
            status = _launch(
                command, binding["public_root"], environment, output / name
            )
            report = output / "cases" / name / "result.json"
            if status["returncode"] == 0 and report.is_file():
                row = _read(report)
                _require(row["case"] == name, "case worker identity")
            else:
                row = {
                    "case": name,
                    "status": status["status"],
                    "fit_complete": False,
                    "precisions": {},
                    "returned_result_present": report.is_file(),
                    "meaning": "Incomplete case; no retry or replacement. Partial files and worker status retained.",
                }
            result["case_results"].append(row)
            print(
                json.dumps(
                    {
                        "case": name,
                        "status": row["status"],
                        "fit_complete": row.get("fit_complete", False),
                    }
                ),
                flush=True,
            )
        result["capability"], legacy = aggregate(
            binding["public_root"], cases, result["case_results"]
        )
        _json(output / "legacy-rows.json", legacy)
        result["legacy_diagnostics"] = _historical(
            "decide",
            output / "legacy",
            binding_path=binding_path,
            binding_sha256=binding_sha256,
            artifact_root=artifact_root,
            rows_path=output / "legacy-rows.json",
        )
        _bound(binding_path, binding_sha256)
        result["status"] = "complete"
    except Exception as error:
        result.update(
            error_type=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc(),
        )
        raise
    finally:
        result["not_attempted_cases"] = [
            name
            for name in result.get("planned_case_names", [])
            if name not in {r["case"] for r in result["case_results"]}
        ]
        if result["not_attempted_cases"]:
            actual = {r["case"]: r for r in result["case_results"]}
            result["case_results"] = [
                actual.get(
                    name,
                    {
                        "case": name,
                        "status": "not_attempted",
                        "fit_complete": False,
                        "fit_calls": 0,
                        "precisions": {},
                        "reason": "prior enclosing stage failed; no replacement or retry",
                    },
                )
                for name in result["planned_case_names"]
            ]
        result["completed_fit_count"] = sum(
            r.get("fit_complete", False) for r in result["case_results"]
        )
        result["known_public_fit_calls"] = sum(
            r.get("fit_calls", 0) for r in result["case_results"]
        )
        result["incomplete_process_case_count"] = sum(
            r["status"] in ("hard_timeout_incomplete", "worker_failed")
            for r in result["case_results"]
        )
        result["files"] = _payloads(output, exclude=("run.json",))
        _json(output / "run.json", result)
    return result


def _verify_case(
    original, destination, case, data_root, artifact_root, scale, expected_result
):
    row = _read(original / "result.json")
    _require(
        row == expected_result and row["case"] == case["name"],
        "root/per-case result mirror",
    )
    _require(
        row["files"] == _payloads(original, exclude=("result.json",)),
        "case payload hashes",
    )
    if not (original / "preparation.json").exists():
        _require(
            not row.get("fit_complete") and row.get("fit_calls") == 0,
            "failed preparation cannot claim a fit",
        )
        return {
            "case": case["name"],
            "scope": "failed pre-fit preparation retained under external anchor; no fit or automatic retry",
        }
    supplied, train, development, prep = preparation(
        Path(data_root) / case["name"] / "calibration.npz",
        Path(artifact_root) / "slow-sampling" / case["name"] / "model.npz",
    )
    _require(_read(original / "preparation.json") == prep, "reconstructed preparation")
    if not row.get("fit_complete"):
        return {
            "case": case["name"],
            "scope": "failed fit retained under external anchor; not reproduced by refitting",
        }
    model = glassbox.LearnedDynamics.load(original / "model.npz")
    _check_model(model, train, development)
    _require(
        _verify_work(original, model) == row["observed_prefix"],
        "observed work report mirror",
    )
    _require(
        model.fingerprint() == row["model_fingerprint"]
        and model.report == _read(original / "public-report.json"),
        "saved synthetic revision/report",
    )
    _require(
        model._seen == learner._recording_content(supplied),
        "fitted synthetic recording ledger",
    )
    destination.mkdir()
    comparisons = []
    for precision in ("float32", "float64"):
        old = row["precisions"].get(precision, {})
        if old.get("status") != "complete":
            comparisons.append(
                {
                    "precision": precision,
                    "scope": "failed prediction evidence retained; not erased or used as fallback",
                }
            )
            continue
        fresh = _evaluate(
            model,
            case,
            Path(data_root) / case["name"],
            destination / precision,
            scale,
            artifact_root,
            precision,
        )
        _require(fresh == old, "synthetic scores/legacy/coverage replay")
        for name in [
            *case["regimes"],
            *(["probe"] if case["family"] == "hidden_input_delay" else []),
        ]:
            saved, replayed = (
                _arrays(original / precision / f"{name}.npz"),
                _arrays(destination / precision / f"{name}.npz"),
            )
            _require(set(saved) == set(replayed), "saved forecast array roster")
            for key in saved:
                _same(
                    saved[key], replayed[key], "same-path public forecast replay " + key
                )
        comparisons.append(
            {"precision": precision, "status": "exact_saved_prediction_replay"}
        )
    return {"case": case["name"], "comparisons": comparisons}


def replay(
    directory, expected_sha256, output, *, binding_path, binding_sha256, artifact_root
):
    directory, output = Path(directory).resolve(), Path(output).resolve()
    _require(
        output != directory and directory not in output.parents,
        "replay output must be outside the sealed attempt",
    )
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "status": "failed",
        "expected_sha256": expected_sha256,
        "numerical_fits": 0,
        "initializers": 0,
        "gradient_or_optimizer_replay": False,
    }
    try:
        _require(
            _sha(directory / "run.json") == expected_sha256,
            "external synthetic run anchor",
        )
        saved = _read(directory / "run.json")
        _require(
            saved["status"] == "complete" and saved["binding_sha256"] == binding_sha256,
            "complete bound synthetic attempt",
        )
        _require(
            saved["files"] == _payloads(directory, exclude=("run.json",)),
            "synthetic sealed payloads",
        )
        binding = _bound(binding_path, binding_sha256)
        cases, _ = scoring.specification(binding["public_root"])
        _verify_data(
            directory / "data", saved["generated_data_sha256"], binding_sha256, cases
        )
        scales = scoring.authenticate_reference_scales(
            binding["public_root"], artifact_root
        )
        _historical(
            "generate",
            output / "data",
            binding_path=binding_path,
            binding_sha256=binding_sha256,
            artifact_root=artifact_root,
            cases=cases,
        )
        for case in cases:
            for path in sorted((output / "data" / case["name"]).glob("*.npz")):
                old, fresh = (
                    _arrays(directory / "data" / case["name"] / path.name),
                    _arrays(path),
                )
                _require(set(old) == set(fresh), "regenerated synthetic data roster")
                for name in old:
                    _same(
                        old[name],
                        fresh[name],
                        "regenerated fixed synthetic data " + name,
                    )
        (output / "cases").mkdir()
        result["cases"] = []
        for case, old in zip(cases, saved["case_results"], strict=True):
            original = directory / "cases" / case["name"]
            if old.get("status") in (
                "worker_failed",
                "hard_timeout_incomplete",
                "not_attempted",
            ):
                result["cases"].append(
                    {
                        "case": case["name"],
                        "scope": "incomplete process evidence authenticated, no restart",
                    }
                )
            else:
                _require(
                    old["data_sha256"] == saved["generated_data_sha256"],
                    "case generation anchor mirror",
                )
                result["cases"].append(
                    _verify_case(
                        original,
                        output / "cases" / case["name"],
                        case,
                        output / "data",
                        artifact_root,
                        scales[case["name"]],
                        old,
                    )
                )
        capability, legacy = aggregate(
            binding["public_root"], cases, saved["case_results"]
        )
        _require(
            capability == saved["capability"], "synthetic capability decision replay"
        )
        _json(output / "legacy-rows.json", legacy)
        diagnostics = _historical(
            "decide",
            output / "legacy",
            binding_path=binding_path,
            binding_sha256=binding_sha256,
            artifact_root=artifact_root,
            rows_path=output / "legacy-rows.json",
        )
        _require(
            diagnostics["precisions"] == saved["legacy_diagnostics"]["precisions"],
            "literal historical diagnostic decisions replay",
        )
        _require(
            saved["files"] == _payloads(directory, exclude=("run.json",)),
            "sealed synthetic attempt unchanged",
        )
        result.update(
            status="complete", capability=capability, historical_flags_replayed=True
        )
    except Exception as error:
        result.update(
            error_type=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc(),
        )
        raise
    finally:
        _json(output / "replay.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("run", "case", "replay"))
    for argument in ("output", "binding", "binding-sha256", "artifact-root"):
        parser.add_argument("--" + argument, required=True)
    parser.add_argument("--directory")
    parser.add_argument("--expected-sha256")
    parser.add_argument("--case")
    parser.add_argument("--data-root")
    parser.add_argument("--data-sha256")
    args = parser.parse_args(argv)
    kwargs = dict(
        binding_path=args.binding,
        binding_sha256=args.binding_sha256,
        artifact_root=args.artifact_root,
    )
    if args.stage == "run":
        run(args.output, **kwargs)
    elif args.stage == "case":
        if not args.case or not args.data_root or not args.data_sha256:
            parser.error("case requires --case, --data-root and --data-sha256")
        run_case(
            args.case,
            args.output,
            args.data_root,
            data_sha256=args.data_sha256,
            **kwargs,
        )
    else:
        if not args.directory or not args.expected_sha256:
            parser.error("replay requires --directory and --expected-sha256")
        replay(args.directory, args.expected_sha256, args.output, **kwargs)


if __name__ == "__main__":
    main()
