"""Frozen public-mean lifecycle qualification; never a consumer fitting option.

``run`` owns exactly M0, M1 and MC, each in a fresh worker. ``replay`` authenticates
the saved attempt and repeats only read-only and deliberately failing control-flow
checks. Both refuse output reuse. Numerical execution requires the separately
committed implementation; importing this module does not construct a fixture.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import resource
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

import glassbox
from glassbox import _sequence_model as core
from glassbox import learner
from glassbox._learner_arrays import array_fingerprint
from glassbox.io.recordings import save_recordings
from glassbox.recordings import SequenceCollection, SequenceSegment, segments_from_mask

PROTOCOL = "public-mean-qualification-v1"
MODULE = "glassbox.experimental.public_mean_lifecycle"
PROTOCOL_SHA256 = "d3a8d0eeeeab2aac923cad7f1f5fe1c8a49fc0ffa765e84b3d7c6bc80cec8f99"
SPEC_SHA256 = "366fd718b490dc2bc44e4a8724f74c3d3b2eda89495fa8eb4a3788f7f2980d08"
OPERATIONS = ("M0", "M1", "MC")
ARRAYS = ("past_states", "past_inputs", "future_inputs", "future_states")
STATE_CHANNELS = (
    "x0 [unitless,fixture]",
    "constant_x [unitless,fixture]",
    "x2 [unitless,fixture]",
)
INPUT_CHANNELS = ("u0 [unitless,fixture]", "constant_u [unitless,fixture]")
SOURCE_FILES = (
    "__init__.py",
    "learner.py",
    "_sequence_model.py",
    "_learner_arrays.py",
    "recordings.py",
    "io/recordings.py",
    "experimental/public_mean_lifecycle.py",
)


class LifecycleError(ValueError):
    """A named qualification check failed, without an automatic retry."""


class InjectedFailure(RuntimeError):
    """Expected test-only failure before numerical initialization/calibration."""


def _require(value, check):
    if not value:
        raise LifecycleError(check)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def _read(path):
    return json.loads(Path(path).read_text())


def _archive(path):
    with np.load(path, allow_pickle=False) as data:
        return json.loads(str(data["metadata"])), {
            name: np.array(data[name], copy=True)
            for name in data.files
            if name != "metadata"
        }


def _same(actual, expected, check):
    a, b = np.asarray(actual), np.asarray(expected)
    _require(
        a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(),
        check,
    )


def _sources():
    package = Path(glassbox.__file__).resolve().parent
    return {name: _sha(package / name) for name in SOURCE_FILES}


def _identity():
    root = Path(__file__).resolve().parents[3]
    protocol = root / "docs/harness/public-mean-qualification-v1.json"
    spec = (
        root
        / "docs/harness/public-mean-qualification-v1/public-v4-lifecycle-fixtures.md"
    )
    _require(_sha(protocol) == PROTOCOL_SHA256, "frozen protocol identity")
    _require(_sha(spec) == SPEC_SHA256, "frozen lifecycle specification identity")
    return {
        "protocol": PROTOCOL,
        "protocol_sha256": PROTOCOL_SHA256,
        "specification_sha256": SPEC_SHA256,
        "sources": _sources(),
        "package_path": str(Path(glassbox.__file__).resolve().parent),
        "python": sys.version,
        "executable": sys.executable,
        "runtime": {
            name: importlib.metadata.version(name)
            for name in ("jax", "jaxlib", "numpy", "scipy")
        },
    }


def _configuration():
    return {
        "jax_enable_x64": bool(jax.config.x64_enabled),
        "environment_sha256": hashlib.sha256(
            json.dumps(dict(os.environ), sort_keys=True).encode()
        ).hexdigest(),
    }


def fixture_segment(name, r, *, constant=False):
    """The frozen six-observation fixture, generated without RNG."""
    if constant:
        x = np.tile(np.array([1 + r / 8, 2 + r / 4, 3 + 3 * r / 8]), (6, 1))
        u = np.tile(np.array([1 / 4 + r / 8, -1 / 2]), (5, 1))
    else:
        u = np.column_stack(
            (0.25 * np.sin(0.7 * (np.arange(5) + 1) + r), np.full(5, -0.5))
        )
        x = np.zeros((6, 3), dtype=np.float64)
        x[0] = [0.05 * r, 2, 0.03 * r]
        for t in range(5):
            x[t + 1] = [
                0.8 * x[t, 0] + 0.1 * u[t, 0],
                2,
                0.6 * x[t, 2] + 0.05 * x[t, 0] + 0.02 * u[t, 0] ** 2,
            ]
    return SequenceSegment(name, "whole", x, u, 0.25)


def collection(*segments):
    return SequenceCollection(
        segments,
        configuration_id="public-v4-small-v1",
        state_channels=STATE_CHANNELS,
        input_channels=INPUT_CHANNELS,
    )


def fixture(operation):
    if operation == "M0":
        return collection(fixture_segment("small-a", 0), fixture_segment("small-b", 1))
    if operation == "M1":
        return collection(fixture_segment("small-c", 2))
    if operation == "MC":
        return collection(
            fixture_segment("constant-a", 0, constant=True),
            fixture_segment("constant-b", 1, constant=True),
        )
    raise ValueError("unknown lifecycle operation")


def _priority(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def expected_roles(operation):
    base = fixture("MC" if operation == "MC" else "M0")
    names = sorted((s.recording_id for s in base.segments), key=_priority)
    return {
        "development": names[:1],
        "train": sorted(names[1:] + (["small-c"] if operation == "M1" else [])),
    }


def _expected_windows(operation, role):
    segments = {
        s.recording_id: s for s in fixture("MC" if operation == "MC" else "M0").segments
    }
    if operation == "M1":
        segments["small-c"] = fixture("M1").segments[0]
    names = expected_roles(operation)[role]
    order = {
        name: sorted(range(2, 5), key=lambda i: _priority([name, "whole", i]))
        for name in names
    }
    keys = [(name, order[name][depth]) for depth in range(3) for name in sorted(names)]
    arrays = {name: [] for name in ARRAYS}
    for name, origin in keys:
        s = segments[name]
        for key, value in zip(
            ARRAYS,
            (
                s.states[origin - 2 : origin + 1],
                s.inputs[origin - 2 : origin],
                s.inputs[origin : origin + 1],
                s.states[origin + 1 : origin + 2],
            ),
            strict=True,
        ):
            arrays[key].append(value)
    return keys, {name: np.stack(values) for name, values in arrays.items()}


def _expected_ledger(operation):
    segments = list(fixture("MC" if operation == "MC" else "M0").segments)
    if operation == "M1":
        segments.extend(fixture("M1").segments)
    return {
        s.recording_id: array_fingerprint(
            {"dt_s": 0.25, "starts": [0]}, {"0_states": s.states, "0_inputs": s.inputs}
        )
        for s in segments
    }


def _verify_cache(model, operation):
    for role in ("train", "development"):
        windows = getattr(model, f"_{role}")
        keys, arrays = _expected_windows(operation, role)
        _require(
            [asdict(k) for k in windows.keys]
            == [
                {"recording_id": n, "segment_id": "whole", "origin": o} for n, o in keys
            ],
            f"{operation} {role} ordered keys",
        )
        _require(windows.source_origins == tuple(o for _, o in keys), f"{role} origins")
        for name in ARRAYS:
            _same(
                getattr(windows.batch, name), arrays[name], f"{operation} {role} {name}"
            )
            _require(
                not getattr(windows.batch, name).flags.writeable, "read-only cache"
            )
        _require(not windows.excitation_declared, "undeclared lifecycle excitation")
    _require(model._seen == _expected_ledger(operation), "complete content ledger")


def _verify_model(path, operation):
    model = learner.LearnedDynamics.load(path)
    _verify_cache(model, operation)
    n = 6 if operation == "M1" else 3
    _require(
        model._metadata()["format"] == "glassbox-default-recipe-v4", "public format"
    )
    _require(model._model.metadata()["format"] == "glassbox-sequence-v2", "mean format")
    _require(model.recipe["id"] == "generic-memory-v4-prototype", "public recipe")
    _require(
        (model.history_steps, model.horizon_steps, model.report["delay_steps"])
        == (2, 1, 1),
        "timing",
    )
    _require(
        model.contract
        == {
            "configuration_id": "public-v4-small-v1",
            "state_channels": list(STATE_CHANNELS),
            "input_channels": list(INPUT_CHANNELS),
            "dt_s": 0.25,
        },
        "recording contract",
    )
    opt = model.report["optimization"]
    gradient, safeguard, objective = (
        opt[name] for name in ("gradient", "safeguard", "objective")
    )
    _require(
        opt["steps"] == 1000 and opt["selected_step"] in range(0, 1001, 100),
        "optimizer budget/selection",
    )
    _require(gradient["training_windows"] == n, "actual gradient batch")
    _require(gradient["policy"] == "ordered_full_cache", "full-cache gradient policy")
    for field in (
        "attempts_started",
        "gradient_proposal_calls_returned",
        "completed_acceptance_attempts",
    ):
        _require(gradient[field] == 1000, f"completed gradient {field}")
    _require(
        gradient["known_gradient_window_visits"] == 1000 * n, "gradient work count"
    )
    _require(not gradient["incomplete_gradient_work_unknown"], "complete gradient work")
    _require(opt["ridge"] == 0.01 * n, "actual ridge")
    _require(safeguard["completed_attempts"] == 1000, "acceptance budget")
    _require(
        safeguard["accepted_attempts"] + safeguard["rejected_attempts"] == 1000,
        "acceptance accounting",
    )
    _require(
        1 <= safeguard["full_training_objective_calls"] <= 8001,
        "acceptance objective budget",
    )
    trace = opt["trace"]
    _require(
        [r["step"] for r in trace] == list(range(0, 1001, 100)), "checkpoint roster"
    )
    _require(
        opt["selected_step"]
        == min(trace, key=lambda r: r["validation_rollout_mse"])["step"],
        "first strict development minimum",
    )
    for values in model._arrays().values():
        _require(np.isfinite(values).all(), "finite saved arrays")
    for name in ("initial_channel_mse", "raw_channel_weights", "channel_weights"):
        _require(np.isfinite(objective[name]).all(), "finite fixed weights")
    norms = model._model.norms
    for name in (
        "state_scale",
        "input_scale",
        "feature_scale",
        "delta_scale",
        "interaction_scale",
        "autonomous_scale",
    ):
        _require(np.all(norms[name] > 0), "positive normalization scales")
    _require(
        norms["state_scale"][1] == norms["input_scale"][1] == 1,
        "constant channel fallback",
    )
    _require(norms["delta_scale"][1] == 1e-4, "constant delta floor")
    # All products involving the centered constant state or command are zero.
    interaction = norms["interaction_scale"].reshape(3, 2)
    _require(
        np.all(interaction[1] == 1) and np.all(interaction[:, 1] == 1),
        "zero interaction columns",
    )
    i, j = np.triu_indices(3)
    _require(
        np.all(norms["autonomous_scale"][(i == 1) | (j == 1)] == 1),
        "zero autonomous columns",
    )
    with jax.enable_x64(True):
        b = model._development.batch
        predicted = np.asarray(
            model.predict(b.past_states, b.past_inputs, b.future_inputs)
        )
    residual = np.abs(predicted - b.future_states)
    expected = np.sort(residual, axis=0)[2]
    _same(model.envelope(), expected, "independent calibration order statistic")
    calibration = model.report["envelope"]
    _require(
        calibration["calibration_windows"] == calibration["quantile_rank"] == 3,
        "calibration count/rank",
    )
    _require(
        calibration["nominal_coverage"] == 0.9
        and calibration["calibrated_on"] == "development",
        "calibration provenance",
    )
    _same(
        model.envelope(),
        np.array(calibration["half_width"]),
        "calibration report mirror",
    )
    if operation == "MC":
        for name in (
            "state_scale",
            "input_scale",
            "feature_scale",
            "interaction_scale",
            "autonomous_scale",
        ):
            _require(np.all(norms[name] == 1), f"MC constant {name}")
        _require(np.all(norms["delta_scale"] == 1e-4), "MC delta floors")
        for name, value in (
            ("initial_channel_mse", 0),
            ("raw_channel_weights", 10000),
            ("channel_weights", 1),
        ):
            _require(np.all(np.asarray(objective[name]) == value), f"MC {name}")
        _require(opt["selected_step"] == 0, "MC selected initial checkpoint")
        _require(
            safeguard["accepted_attempts"] == 0
            and safeguard["rejected_attempts"] == 1000,
            "MC strict zero rejection",
        )
        _require(
            safeguard["full_training_objective_calls"] == 8001,
            "MC all trial evaluations",
        )
        for name in (
            "initial_full_training_loss",
            "final_full_training_loss",
            "selected_full_training_loss",
        ):
            _require(safeguard[name] == 0, f"MC zero {name}")
        _require(
            all(r["validation_rollout_mse"] == 0 for r in trace),
            "MC zero development losses",
        )
    return model, {
        "operation": operation,
        "fingerprint": model.fingerprint(),
        "training_windows": n,
        "development_windows": 3,
        "roles": expected_roles(operation),
        "ridge": opt["ridge"],
        "selected_step": opt["selected_step"],
        "gradient": gradient,
        "safeguard": safeguard,
        "objective": objective,
        "envelope": calibration,
        "zero_denominator_ratios": {
            "value": None,
            "reason": "no ratio is used to qualify a zero objective",
        }
        if operation == "MC"
        else None,
    }


def _capture_event(state, directory, handle):
    """Persist actual source-bound scalar work and selected-array witnesses."""
    phase = state["phase"]
    row = {"phase": phase}
    for name in (
        "attempt",
        "step",
        "trial_index",
        "scale",
        "loss",
        "accepted_scale",
        "accepted_trial_index",
        "minibatch_loss",
        "gradient_norm",
        "gradient_finite",
        "validation_rollout_mse",
        "unweighted_validation_rollout_mse",
        "full_training_loss",
        "selected_step",
    ):
        if name in state:
            value = state[name]
            row[name] = (
                str(value)
                if isinstance(value, float) and not np.isfinite(value)
                else value
            )
    if phase == "started":
        indices = np.asarray(state["indices"])
        row["indices"] = indices.tolist()
        row["indices_dtype"] = indices.dtype.str
    if phase == "checkpoint":
        arrays = {
            **{f"param_{k}": np.asarray(v) for k, v in state["params"].items()},
            **{f"norm_{k}": np.asarray(v) for k, v in state["norms"].items()},
        }
        path = directory / f"checkpoint-{state['step']:04d}.npz"
        _require(not path.exists(), "checkpoint witness cannot be overwritten")
        np.savez_compressed(path, **arrays)
        row["array_fingerprint"] = array_fingerprint({}, arrays)
    if phase == "weights":
        path = directory / "weights.npz"
        _require(not path.exists(), "weight witness cannot be overwritten")
        np.savez_compressed(
            path,
            **{
                name: np.asarray(state[name])
                for name in (
                    "normalization",
                    "initial_training_prediction",
                    "initial_channel_mse",
                    "raw_channel_weights",
                    "channel_weights",
                    "fixed_weights",
                    "weight_floor",
                    "weight_normalizer",
                )
            },
        )
    handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    handle.flush()


def _observed_work(directory, model):
    """Reduce saved observations, without replaying any gradient or optimizer."""
    rows = [
        json.loads(line)
        for line in (Path(directory) / "work.jsonl").read_text().splitlines()
    ]
    phases = {
        name: [r for r in rows if r["phase"] == name]
        for name in (
            "initialized",
            "weights",
            "checkpoint",
            "initial_objective",
            "started",
            "proposed",
            "trial",
            "completed",
        )
    }
    _require(len(rows) == sum(map(len, phases.values())), "known observed phases")
    for phase in ("initialized", "weights", "initial_objective"):
        _require(len(phases[phase]) == 1, "one actual " + phase)
    for phase in ("started", "proposed", "completed"):
        _require(
            [r["attempt"] for r in phases[phase]] == list(range(1, 1001)),
            "actual attempt roster " + phase,
        )
    n = len(model._train.keys)
    for row in phases["started"]:
        _require(
            row["indices"] == list(range(n))
            and all(type(v) is int for v in row["indices"]),
            "actual full-cache indices",
        )
        _require(
            row["indices_dtype"] == np.dtype(np.int64).str,
            "actual full-cache index dtype",
        )
    opt = model.report["optimization"]
    sg = opt["safeguard"]
    _require(
        1 + len(phases["trial"]) == sg["full_training_objective_calls"],
        "actual objective call count",
    )
    accepted = sum(r["accepted_scale"] > 0 for r in phases["completed"])
    _require(
        accepted == sg["accepted_attempts"]
        and 1000 - accepted == sg["rejected_attempts"],
        "actual accepted/rejected counts",
    )
    checkpoints = phases["checkpoint"]
    _require(
        [r["step"] for r in checkpoints] == list(range(0, 1001, 100)),
        "actual checkpoint roster",
    )
    _require(
        [
            {"step": r["step"], "validation_rollout_mse": r["validation_rollout_mse"]}
            for r in checkpoints
        ]
        == [
            {k: r[k] for k in ("step", "validation_rollout_mse")} for r in opt["trace"]
        ],
        "actual checkpoint scalar trace",
    )
    selected = min(checkpoints, key=lambda r: r["validation_rollout_mse"])
    _require(selected["step"] == opt["selected_step"], "actual selected checkpoint")
    for row in checkpoints:
        with np.load(
            Path(directory) / f"checkpoint-{row['step']:04d}.npz", allow_pickle=False
        ) as data:
            arrays = {k: data[k] for k in data.files}
            _require(
                array_fingerprint({}, arrays) == row["array_fingerprint"],
                "checkpoint array witness",
            )
            if row["step"] == selected["step"]:
                _require(
                    set(arrays) == set(model._model.arrays()),
                    "selected parameter/norm roster",
                )
                for name, value in model._model.arrays().items():
                    _same(arrays[name], value, "selected checkpoint " + name)
    with np.load(
        Path(directory) / "initializer-1-returned.npz", allow_pickle=False
    ) as initial:
        with np.load(
            Path(directory) / "checkpoint-0000.npz", allow_pickle=False
        ) as checkpoint:
            _require(
                set(initial.files) == set(checkpoint.files), "initial witness roster"
            )
            for name in initial.files:
                _same(
                    initial[name], checkpoint[name], "actual initialized tree " + name
                )
    objective = opt["objective"]
    with np.load(Path(directory) / "weights.npz", allow_pickle=False) as data:
        for name in (
            "initial_channel_mse",
            "raw_channel_weights",
            "channel_weights",
            "weight_floor",
            "weight_normalizer",
        ):
            _same(
                data[name], np.asarray(objective[name]), "actual weight report " + name
            )
        _same(data["channel_weights"], data["fixed_weights"], "fixed optimizer weights")
        _same(
            data["normalization"], np.asarray(opt["error_scale"]), "actual hold scales"
        )
        _require(
            array_fingerprint({}, {"prediction": data["initial_training_prediction"]})
            == objective["initial_training_prediction_fingerprint"],
            "actual initial prediction fingerprint",
        )
    entered = _read(Path(directory) / "initializer-1-entered.json")
    _require(
        entered["jax_enable_x64"]
        and entered["training_windows"] == n
        and entered["ridge"] == 0.01 * n,
        "actual initializer context",
    )
    return {
        "phase_counts": {k: len(v) for k, v in phases.items()},
        "known_gradient_window_visits": n * len(phases["proposed"]),
        "selected_step": selected["step"],
        "scope": "actual call/scalar/array witnesses; no replay of gradients, Adam or solves",
    }


def _work_prefix(directory):
    path = Path(directory) / "work.jsonl"
    rows = (
        []
        if not path.exists()
        else [json.loads(line) for line in path.read_text().splitlines()]
    )
    counts = {}
    for row in rows:
        counts[row["phase"]] = counts.get(row["phase"], 0) + 1
    started = [r for r in rows if r["phase"] == "started"]
    returned = counts.get("proposed", 0)
    return {
        "phase_counts": counts,
        "known_gradient_window_visits": sum(
            len(r["indices"]) for r in started[:returned]
        ),
        "incomplete_gradient_work_unknown": len(started) > returned,
        "full_training_objective_calls_returned": counts.get("initial_objective", 0)
        + counts.get("trial", 0),
        "scope": "persisted observer prefix; unpersisted or interrupted work is not counted as complete",
    }


def _operation(output, operation):
    """One actual operation; save returned evidence before testing its claims."""
    directory = Path(output) / operation
    directory.mkdir()
    before, identity = _configuration(), _identity()
    _require(
        before["jax_enable_x64"] == (operation == "MC"), "operation ambient precision"
    )
    recordings = fixture(operation)
    save_recordings(recordings, directory / "recordings.npz")
    calls = []
    original = core.initialize_sequence_model

    def observed(*args, **kwargs):
        row = {
            "index": len(calls) + 1,
            "jax_enable_x64": bool(jax.config.x64_enabled),
            "training_windows": len(args[0].past_states),
            "ridge": kwargs["ridge"],
            "settings": {
                k: kwargs[k] for k in ("seed", "width", "memory", "delay_steps")
            },
        }
        calls.append(row)
        _json(directory / f"initializer-{len(calls)}-entered.json", row)
        result = original(*args, **kwargs)
        np.savez_compressed(
            directory / f"initializer-{len(calls)}-returned.npz", **result.arrays()
        )
        return result

    started = time.perf_counter()
    timing = {
        "fit_calls": 0,
        "calibration_calls": 0,
        "scope": "wall times around the one actual public call, fitter and calibration; compilation is included in fitting, not independently isolated",
    }
    original_fit, original_calibrate = learner.fit_sequence_model, learner._calibrate

    def timed_fit(*args, **kwargs):
        timing["fit_calls"] += 1
        fit_started = time.perf_counter()
        timing["public_preparation_wall_s"] = fit_started - public_started
        try:
            return original_fit(*args, **kwargs)
        finally:
            timing["fit_including_compile_wall_s"] = time.perf_counter() - fit_started

    def timed_calibrate(*args, **kwargs):
        timing["calibration_calls"] += 1
        calibration_started = time.perf_counter()
        try:
            return original_calibrate(*args, **kwargs)
        finally:
            timing["calibration_wall_s"] = time.perf_counter() - calibration_started

    outcome = {
        "operation": operation,
        "identity": identity,
        "configuration_before": before,
    }
    try:
        previous = (
            learner.LearnedDynamics.load(Path(output) / "M0/model.npz")
            if operation == "M1"
            else None
        )
        predecessor = previous.fingerprint() if previous is not None else None
        if previous is not None:
            predecessor_file = _sha(Path(output) / "M0/model.npz")
            args = tuple(getattr(previous._development.batch, k) for k in ARRAYS[:3])
            predecessor_prediction = np.asarray(previous.predict(*args))
            np.savez_compressed(
                directory / "predecessor-prediction.npz",
                prediction=predecessor_prediction,
            )
        with (directory / "work.jsonl").open("x") as work:

            def observe(state):
                _capture_event(state, directory, work)

            with (
                patch.object(core, "initialize_sequence_model", observed),
                patch.object(core, "_observe_attempt", observe),
                patch.object(learner, "fit_sequence_model", timed_fit),
                patch.object(learner, "_calibrate", timed_calibrate),
            ):
                public_started = time.perf_counter()
                try:
                    model = (
                        previous.update(recordings)
                        if previous is not None
                        else glassbox.fit(recordings)
                    )
                finally:
                    timing["public_call_wall_s"] = time.perf_counter() - public_started
        _require(
            timing["fit_calls"] == timing["calibration_calls"] == 1,
            "one actual fitter/calibration call",
        )
        model.save(directory / "model.npz")
        _json(directory / "public-report.json", model.report)
        b = model._development.batch
        np.savez_compressed(
            directory / "query.npz", **{k: getattr(b, k) for k in ARRAYS[:3]}
        )
        _require(
            len(calls) == 1 and calls[0]["jax_enable_x64"],
            "one owned float64 initializer",
        )
        _require(
            calls[0]["ridge"] == 0.01 * (6 if operation == "M1" else 3),
            "captured ridge",
        )
        _, verified = _verify_model(directory / "model.npz", operation)
        verified["observed_work"] = _observed_work(directory, model)
        if previous is not None:
            _require(
                previous.fingerprint() == predecessor,
                "immutable predecessor after update",
            )
            _require(
                model.report["previous_revision"] == predecessor
                and model.fingerprint() != predecessor,
                "new revision linkage",
            )
            _require(
                _sha(Path(output) / "M0/model.npz") == predecessor_file,
                "predecessor archive unchanged",
            )
            _same(
                previous.predict(*args),
                predecessor_prediction,
                "predecessor prediction after update",
            )
        _require(_configuration() == before, "operation configuration restoration")
        _require(_identity() == identity, "operation source/runtime stable")
        outcome.update(status="complete", verification=verified)
    except BaseException as error:
        outcome.update(
            status="failed",
            error_type=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc(),
        )
        raise
    finally:
        outcome.update(
            elapsed_s=time.perf_counter() - started,
            initializer_calls=calls,
            configuration_after=_configuration(),
            observed_prefix=_work_prefix(directory),
            timing=timing,
            peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if sys.platform == "darwin" else 1024),
            peak_process_rss_scope="whole fresh operation worker including imports, fitting, calibration and saved-evidence verification",
        )
        _json(directory / "outcome.json", outcome)
    return outcome


def _reject(call, check, *, kinds=(ValueError, TypeError)):
    try:
        call()
    except kinds as error:
        return {
            "check": check,
            "status": "rejected",
            "exception_type": type(error).__name__,
        }
    except BaseException as error:
        raise LifecycleError(f"{check}: unexpected {type(error).__name__}") from error
    raise LifecycleError(f"{check}: accepted invalid evidence")


def _forbidden(*args, **kwargs):
    raise InjectedFailure("unexpected numerical fitting")


def _negative_calls(m0, m1):
    base, fresh = fixture("M0"), fixture("M1")
    a, b = base.segments
    c = fresh.segments[0]
    cases = {
        "list_not_collection": lambda: glassbox.fit(list(base.segments)),
        "one_recording": lambda: glassbox.fit(collection(a)),
        "two_origins_per_role": lambda: glassbox.fit(
            collection(
                *(replace(s, states=s.states[:5], inputs=s.inputs[:4]) for s in (a, b))
            )
        ),
        "duplicate_content": lambda: glassbox.fit(
            collection(a, replace(a, recording_id="same-content"))
        ),
        "reused_id_new_content": lambda: m0.update(
            collection(replace(c, recording_id="small-a"))
        ),
        "renamed_old_content": lambda: m0.update(
            collection(replace(a, recording_id="fresh-name"))
        ),
        "persisted_fresh_content": lambda: m1.update(
            collection(replace(c, recording_id="again"))
        ),
        "configuration": lambda: m0.update(replace(fresh, configuration_id="changed")),
        "state_order": lambda: m0.update(
            replace(fresh, state_channels=tuple(reversed(STATE_CHANNELS)))
        ),
        "input_order": lambda: m0.update(
            replace(fresh, input_channels=tuple(reversed(INPUT_CHANNELS)))
        ),
        "units": lambda: m0.update(
            replace(fresh, state_channels=("x0 [metres,fixture]", *STATE_CHANNELS[1:]))
        ),
        "interval": lambda: m0.update(collection(replace(c, dt_s=0.2))),
        "missing_configuration": lambda: glassbox.fit(
            replace(base, configuration_id=None)
        ),
        "missing_state_channels": lambda: glassbox.fit(
            replace(base, state_channels=())
        ),
        "missing_input_channels": lambda: glassbox.fit(
            replace(base, input_channels=())
        ),
    }
    with patch.object(learner, "_train", _forbidden):
        return [_reject(call, name) for name, call in cases.items()]


def _constructor_checks():
    a, b = fixture("M0").segments
    cases = {
        "nan_states": lambda: replace(a, states=np.full_like(a.states, np.nan)),
        "inf_inputs": lambda: replace(a, inputs=np.full_like(a.inputs, np.inf)),
        "row_count": lambda: replace(a, inputs=a.inputs[:-1]),
        "duplicate_segment": lambda: collection(a, a),
        "overlap": lambda: collection(a, replace(a, segment_id="overlap", start_row=1)),
        "mixed_intervals": lambda: collection(a, replace(b, dt_s=0.2)),
        "duplicate_channel_names": lambda: replace(
            collection(a, b), state_channels=("x", "x", "z")
        ),
        "state_width_names": lambda: replace(
            collection(a, b), state_channels=("x", "z")
        ),
        "input_width_names": lambda: replace(collection(a, b), input_channels=("u",)),
    }
    for label, value in (
        ("zero", 0),
        ("negative", -1),
        ("nan", np.nan),
        ("inf", np.inf),
    ):
        cases[f"invalid_dt_{label}"] = lambda value=value: replace(a, dt_s=value)
    for label, value in (("negative", -1), ("boolean", True)):
        cases[f"invalid_start_{label}"] = lambda value=value: replace(
            a, start_row=value
        )
    rows = [_reject(call, name) for name, call in cases.items()]
    x, u = np.array(a.states), np.array(a.inputs)
    owned = SequenceSegment("owned", "whole", x, u, 0.25)
    x[:] = 99
    u[:] = 99
    _same(owned.states, a.states, "segment copied observations")
    _same(owned.inputs, a.inputs, "segment copied commands")
    _require(
        not owned.states.flags.writeable and not owned.inputs.flags.writeable,
        "read-only segment",
    )
    x = np.arange(16.0)[:, None]
    u = (np.arange(15.0) + 100)[:, None]
    valid = np.ones(16, dtype=bool)
    valid[6:8] = False
    x[6:8] = np.nan
    masked = SequenceCollection(segments_from_mask("r", x, u, valid, dt_s=0.1))
    keys = masked.window_keys(history_steps=1, horizon_steps=2)
    _require(len(keys) == 8, "mask complete-window count")
    windows = masked.extract(
        [keys[0], keys[1], keys[3]], history_steps=1, horizon_steps=2
    )
    _require(windows.source_origins == (1, 2, 9), "mask source origins")
    _same(
        windows.batch.past_states[..., 0],
        np.array([[0.0, 1.0], [1.0, 2.0], [8.0, 9.0]]),
        "history never bridges gap",
    )
    _same(
        windows.batch.future_states[..., 0],
        np.array([[2.0, 3.0], [3.0, 4.0], [10.0, 11.0]]),
        "targets never bridge gap",
    )
    _same(
        windows.batch.future_inputs[..., 0],
        np.array([[101.0, 102.0], [102.0, 103.0], [109.0, 110.0]]),
        "commands never bridge gap",
    )
    rows.extend(
        [
            {"check": "segment_defensive_arrays", "status": "passed"},
            {"check": "mask_gap", "status": "passed"},
        ]
    )
    return rows


def _shape_checks(model):
    b = model._development.batch
    x, up, uf = (np.array(getattr(b, k)[0]) for k in ARRAYS[:3])
    cases = {
        "rank_one": (x[:, 0], up, uf),
        "mismatched_ranks": (x[None], up, uf),
        "state_width": (x[:, :2], up, uf),
        "past_input_width": (x, up[:, :1], uf),
        "future_input_width": (x, up, uf[:, :1]),
        "batch_counts": (np.stack([x, x]), up[None], uf[None]),
        "insufficient_context": (x[1:], up[1:], uf),
        "unaligned_history": (x, up[:1], uf),
        "empty_future": (x, up, uf[:0]),
        "unsupported_horizon": (x, up, np.concatenate([uf, uf])),
    }
    rows = []
    for mode, predict in (
        ("eager", model.predict),
        ("outer_jit", jax.jit(model.predict)),
    ):
        for name, values in cases.items():
            rows.append(
                _reject(
                    lambda values=values, predict=predict: predict(*values),
                    f"{mode}:{name}",
                )
            )
    return rows


def _write_archive(path, metadata, arrays, *, coherent):
    meta = copy.deepcopy(metadata)
    if coherent:
        meta.pop("fingerprint", None)
        meta["fingerprint"] = array_fingerprint(meta, arrays)
    np.savez_compressed(path, metadata=json.dumps(meta), **arrays)


def archive_defects(path, directory):
    """Fixed raw and coherently resealed defects; originals remain untouched."""
    directory = Path(directory)
    directory.mkdir()
    metadata, arrays = _archive(path)
    cases = (
        "raw_parameter",
        "raw_cache",
        "raw_contract",
        "old_format_recipe",
        "wrong_recipe",
        "missing_envelope",
        "negative_envelope",
        "envelope_shape",
        "missing_quadratic",
        "parameter_shape",
        "nonfinite_parameter",
        "nonpositive_scale",
        "unsupported_model_format",
    )
    result = []
    for name in cases:
        meta, values = copy.deepcopy(metadata), {k: v.copy() for k, v in arrays.items()}
        if name == "raw_parameter":
            values["param_linear"].flat[0] += 1
        elif name == "raw_cache":
            values["train_future_states"].flat[0] += 1
        elif name == "raw_contract":
            meta["contract"]["configuration_id"] = "changed"
        elif name == "old_format_recipe":
            meta["format"] = "glassbox-default-recipe-v3"
            meta["recipe"]["id"] = "generic-memory-v3-prototype"
            meta["report"]["recipe"]["id"] = "generic-memory-v3-prototype"
        elif name == "wrong_recipe":
            meta["recipe"]["steps"] = meta["report"]["recipe"]["steps"] = 999
        elif name == "missing_envelope":
            del values["envelope_half_width"]
        elif name == "negative_envelope":
            values["envelope_half_width"].flat[0] = -1
        elif name == "envelope_shape":
            values["envelope_half_width"] = values["envelope_half_width"].ravel()
        elif name == "missing_quadratic":
            del values["param_autonomous"]
        elif name == "parameter_shape":
            values["param_autonomous"] = values["param_autonomous"].ravel()
        elif name == "nonfinite_parameter":
            values["param_autonomous"].flat[0] = np.nan
        elif name == "nonpositive_scale":
            values["norm_state_scale"].flat[0] = 0
        else:
            meta["model"]["format"] = "unsupported-sequence-format"
        changed = directory / f"{name}.npz"
        _write_archive(changed, meta, values, coherent=not name.startswith("raw_"))
        row = _reject(
            lambda changed=changed: learner.LearnedDynamics.load(changed), name
        )
        row.update(
            sha256=_sha(changed),
            payload_fingerprint=array_fingerprint(meta, values),
            coherent=not name.startswith("raw_"),
        )
        result.append(row)
    return result


def _defensive(model):
    before = model.fingerprint()
    report, contract, recipe, envelope = (
        model.report,
        model.contract,
        model.recipe,
        model.envelope(),
    )
    report["recipe"]["steps"] = 0
    contract["state_channels"][0] = "changed"
    recipe["id"] = "changed"
    envelope[:] = 999
    _require(model.fingerprint() == before, "defensive public copies")
    return {"check": "defensive_public_copies", "status": "passed"}


def _scope_checks(m0, m1):
    rows = []
    models = (m0, m1)
    before_models = [m.fingerprint() for m in models]
    for ambient in (False, True):
        for operation in ("fit", "update"):
            for site in ("initializer", "calibration"):
                observed = []

                def fail(*args, observed=observed, site=site, **kwargs):
                    observed.append(bool(jax.config.x64_enabled))
                    if not jax.config.x64_enabled:
                        raise LifecycleError(
                            "injected failure outside owned float64 scope"
                        )
                    raise InjectedFailure(site)

                def stub(*args, **kwargs):
                    _require(
                        jax.config.x64_enabled, "stub fitter inside owned float64 scope"
                    )
                    return m0._model, m0.report["optimization"]

                with jax.enable_x64(ambient):
                    before = _configuration()
                    if site == "initializer":
                        with patch.object(core, "initialize_sequence_model", fail):
                            row = _reject(
                                lambda operation=operation: (
                                    glassbox.fit(fixture("M0"))
                                    if operation == "fit"
                                    else m0.update(fixture("M1"))
                                ),
                                f"scope:{ambient}:{operation}:{site}",
                                kinds=(InjectedFailure,),
                            )
                    else:
                        with (
                            patch.object(learner, "fit_sequence_model", stub),
                            patch.object(learner, "_calibrate", fail),
                            patch.object(core, "initialize_sequence_model", _forbidden),
                        ):
                            row = _reject(
                                lambda operation=operation: (
                                    glassbox.fit(fixture("M0"))
                                    if operation == "fit"
                                    else m0.update(fixture("M1"))
                                ),
                                f"scope:{ambient}:{operation}:{site}",
                                kinds=(InjectedFailure,),
                            )
                    _require(observed == [True], "exact injected owned-scope entry")
                    _require(
                        _configuration() == before,
                        "error precision/environment restoration",
                    )
                rows.append(row)
    _require(
        [m.fingerprint() for m in models] == before_models,
        "immutable revisions after failed updates",
    )
    return rows


def _thread_check():
    _require(not jax.config.x64_enabled, "thread test default false")
    entered, release = threading.Event(), threading.Event()
    observed = {}

    def fail(*args, **kwargs):
        observed["inside"] = bool(jax.config.x64_enabled)
        entered.set()
        if not release.wait(5):
            raise LifecycleError("thread scope release timed out")
        raise InjectedFailure("thread")

    def fit_thread():
        observed["before"] = bool(jax.config.x64_enabled)
        try:
            glassbox.fit(fixture("M0"))
        except InjectedFailure:
            observed["sentinel"] = True
        except BaseException as error:
            observed["error"] = type(error).__name__
        finally:
            observed["after"] = bool(jax.config.x64_enabled)

    with patch.object(core, "initialize_sequence_model", fail):
        thread = threading.Thread(target=fit_thread, daemon=True)
        thread.start()
        try:
            _require(entered.wait(5), "thread initializer entry timed out")
            observed["independent"] = bool(jax.config.x64_enabled)
        finally:
            release.set()
            thread.join(5)
        _require(not thread.is_alive(), "thread worker did not stop")
    _require(
        observed
        == {
            "before": False,
            "inside": True,
            "independent": False,
            "sentinel": True,
            "after": False,
        },
        "thread-local float64 isolation",
    )
    return {"check": "thread_scope", "status": "passed", "observed": observed}


def _close(actual, expected, *, atol, rtol, check):
    a, b = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    _require(
        a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(),
        check + " shape/finite",
    )
    delta = np.abs(a - b)
    _require(np.all(delta <= atol + rtol * np.abs(b)), check)
    return {
        "check": check,
        "maximum_absolute_difference": float(delta.max(initial=0)),
        "atol": atol,
        "rtol": rtol,
    }


def _transforms(model):
    """Same-function lifecycle transformations; oracle parity is a separate tier."""
    b = model._development.batch
    values = tuple(jnp.asarray(getattr(b, k)) for k in ARRAYS[:3])
    single = tuple(v[0] for v in values)
    ambient = bool(jax.config.x64_enabled)
    atol, rtol = (1e-12, 1e-10) if ambient else (2e-6, 2e-5)
    predictions = {
        "eager": model.predict(*values),
        "jit": jax.jit(model.predict)(*values),
        "vmap": jax.vmap(model.predict)(*values),
        "single": model.predict(*single),
    }
    rows = [
        _close(
            predictions[name],
            predictions["eager"],
            atol=atol,
            rtol=rtol,
            check=f"prediction:{name}",
        )
        for name in ("jit", "vmap")
    ]
    rows.append(
        _close(
            predictions["single"],
            predictions["eager"][0],
            atol=atol,
            rtol=rtol,
            check="single/batch",
        )
    )
    expected_dtype = np.dtype("float64" if ambient else "float32")
    _require(
        all(np.asarray(v).dtype == expected_dtype for v in predictions.values()),
        "ambient prediction dtype",
    )
    scale = jnp.asarray(model._model.norms["state_scale"])
    mean = jnp.asarray(model._model.norms["state_mean"])
    coefficients = jnp.sin(jnp.arange(3) + 1)[None] / 3

    def loss(*args):
        return jnp.sum(coefficients * (model.predict(*args) - mean) / scale)

    gradients = jax.grad(loss, argnums=(0, 1, 2))(*single)
    compiled_gradients = jax.jit(jax.grad(loss, argnums=(0, 1, 2)))(*single)
    through_jit = jax.grad(jax.jit(loss), argnums=(0, 1, 2))(*single)
    direction_scales = (
        scale,
        jnp.asarray(model._model.norms["input_scale"]),
        jnp.asarray(model._model.norms["input_scale"]),
    )
    for i, (value, g, cg, gj, units) in enumerate(
        zip(
            single,
            gradients,
            compiled_gradients,
            through_jit,
            direction_scales,
            strict=True,
        )
    ):
        rows.append(
            _close(
                g * units,
                cg * units,
                atol=1e-11 if ambient else 2e-6,
                rtol=1e-9 if ambient else 2e-3,
                check=f"jit_grad:{i}",
            )
        )
        rows.append(
            _close(
                g * units,
                gj * units,
                atol=1e-11 if ambient else 2e-6,
                rtol=1e-9 if ambient else 2e-3,
                check=f"grad_jit:{i}",
            )
        )
        direction = jnp.sin(jnp.arange(value.size).reshape(value.shape) + 1)
        direction = direction / jnp.sqrt(jnp.sum(direction**2)) * units
        tangents = tuple(
            direction if j == i else jnp.zeros_like(v) for j, v in enumerate(single)
        )
        _, jvp = jax.jvp(model.predict, single, tangents)
        vjp = jnp.sum(g * direction)
        projected = jnp.sum(coefficients * jvp / scale)
        rows.append(
            _close(
                projected,
                vjp,
                atol=1e-11 if ambient else 2e-6,
                rtol=1e-9 if ambient else 2e-4,
                check=f"duality:{i}",
            )
        )
        h = 2.0 ** (-16 if ambient else -7)
        plus = tuple(v + h * t for v, t in zip(single, tangents, strict=True))
        minus = tuple(v - h * t for v, t in zip(single, tangents, strict=True))
        _require(
            np.any(np.asarray(plus[i]) != np.asarray(minus[i])),
            "finite-difference direction resolved",
        )
        fd = (loss(*plus) - loss(*minus)) / (2 * h)
        rows.append(
            _close(
                fd,
                vjp,
                atol=2e-8 if ambient else 2e-5,
                rtol=2e-5 if ambient else 1e-2,
                check=f"finite_difference:{i}",
            )
        )
    return {
        "ambient_x64": ambient,
        "prediction_fingerprint": array_fingerprint(
            {}, {k: np.asarray(v) for k, v in predictions.items()}
        ),
        "checks": rows,
        "oracle_scope": "same-function lifecycle checks; independent pinned quadratic oracle is qualified separately",
    }


def _checks(output, destination, ambient):
    destination = Path(destination)
    destination.mkdir()
    before, identity = _configuration(), _identity()
    _require(before["jax_enable_x64"] == ambient, "checks ambient precision")
    outcome = {
        "ambient_x64": ambient,
        "identity": identity,
        "configuration_before": before,
    }
    try:
        models, verification, transforms = {}, {}, {}
        for operation in OPERATIONS:
            model, verification[operation] = _verify_model(
                Path(output) / operation / "model.npz", operation
            )
            models[operation] = model
            verification[operation]["observed_work"] = _observed_work(
                Path(output) / operation, model
            )
            with np.load(
                Path(output) / operation / "query.npz", allow_pickle=False
            ) as query:
                _require(set(query.files) == set(ARRAYS[:3]), "lifecycle query roster")
                for name in ARRAYS[:3]:
                    _same(
                        query[name],
                        getattr(model._development.batch, name),
                        "saved lifecycle query " + name,
                    )
            source_meta, source_arrays = _archive(
                Path(output) / operation / "model.npz"
            )
            roundtrip = destination / f"{operation}-roundtrip.npz"
            model.save(roundtrip)
            saved_meta, saved_arrays = _archive(roundtrip)
            _require(
                source_meta == saved_meta and set(source_arrays) == set(saved_arrays),
                "roundtrip metadata/roster",
            )
            for name in source_arrays:
                _same(
                    saved_arrays[name], source_arrays[name], "roundtrip array " + name
                )
            restored = learner.LearnedDynamics.load(roundtrip)
            _require(
                restored.fingerprint() == model.fingerprint(), "roundtrip fingerprint"
            )
            b = model._development.batch
            args = tuple(getattr(b, k) for k in ARRAYS[:3])
            _same(
                restored.predict(*args),
                model.predict(*args),
                "same-path roundtrip prediction",
            )
            transforms[operation] = _transforms(restored)
        m0, m1 = models["M0"], models["M1"]
        before_revisions = {k: m.fingerprint() for k, m in models.items()}
        m0_prediction = np.asarray(
            m0.predict(*(getattr(m0._development.batch, k) for k in ARRAYS[:3]))
        )
        _require(
            m1.report["previous_revision"] == m0.fingerprint()
            and m1.fingerprint() != m0.fingerprint(),
            "revision predecessor",
        )
        for name in ARRAYS:
            _same(
                getattr(m0._development.batch, name),
                getattr(m1._development.batch, name),
                "reserved development " + name,
            )
        checks = [
            *(_defensive(m) for m in models.values()),
            *_negative_calls(m0, m1),
            *_constructor_checks(),
            *_shape_checks(m0),
            *_scope_checks(m0, m1),
        ]
        if not ambient:
            checks.append(_thread_check())
        checks.extend(
            archive_defects(
                Path(output) / "M0/model.npz", destination / "archive-defects"
            )
        )
        _require(
            {k: m.fingerprint() for k, m in models.items()} == before_revisions,
            "immutable revisions after checks",
        )
        _same(
            m0.predict(*(getattr(m0._development.batch, k) for k in ARRAYS[:3])),
            m0_prediction,
            "predecessor prediction unchanged",
        )
        _require(_configuration() == before, "checks configuration restoration")
        _require(_identity() == identity, "checks source/runtime stable")
        outcome.update(
            status="complete",
            checks=checks,
            verification=verification,
            transforms=transforms,
        )
    except BaseException as error:
        outcome.update(
            status="failed",
            error_type=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc(),
        )
        raise
    finally:
        outcome["configuration_after"] = _configuration()
        _json(destination / "outcome.json", outcome)
    return outcome


_PUBLIC_BOUNDARY_CODE = """
import hashlib,inspect,json,os,pathlib,sys
import jax
import numpy as np
before=dict(os.environ)
with jax.enable_x64(sys.argv[3] == "true"):
    expected=bool(jax.config.x64_enabled)
    import glassbox
    assert tuple(glassbox.__all__)==("fit","LearnedDynamics","SequenceCollection","SequenceSegment","segments_from_mask")
    assert list(inspect.signature(glassbox.fit).parameters)==["recordings"]
    assert list(inspect.signature(glassbox.LearnedDynamics.update).parameters)==["self","recordings"]
    assert list(inspect.signature(glassbox.LearnedDynamics.predict).parameters)==["self","past_states","past_inputs","future_inputs"]
    model=glassbox.LearnedDynamics.load(sys.argv[1])
    with np.load(sys.argv[2],allow_pickle=False) as data:
        args=tuple(data[k] for k in ("past_states","past_inputs","future_inputs"))
    prediction=np.asarray(jax.jit(model.predict)(*args))
    forbidden={"cli","integrations","io","workflows","experimental","belief","fitting","control","core"}
    loaded=sorted(n for n in sys.modules if n.startswith("glassbox.") and n.split(".")[1] in forbidden)
    assert not loaded,loaded
    assert bool(jax.config.x64_enabled)==expected
    assert dict(os.environ)==before
    package=pathlib.Path(glassbox.__file__).resolve().parent
    sources={name:hashlib.sha256((package/name).read_bytes()).hexdigest() for name in ("__init__.py","learner.py","_sequence_model.py","_learner_arrays.py","recordings.py")}
    print(json.dumps({"status":"complete","ambient_x64":expected,"loaded_deferred":loaded,"shape":list(prediction.shape),"dtype":prediction.dtype.str,"finite":bool(np.isfinite(prediction).all()),"package_path":glassbox.__file__,"sources":sources,"fingerprint":model.fingerprint()},sort_keys=True))
"""


def _launch(command, log_path, command_path):
    environment = dict(os.environ)
    environment.pop("JAX_ENABLE_X64", None)
    _json(
        command_path,
        {
            "argv": command,
            "environment_overrides": {"JAX_ENABLE_X64": None},
            "hard_timeout_s": 14400,
        },
    )
    started = time.perf_counter()
    with Path(log_path).open("x") as handle:
        try:
            process = subprocess.run(
                command,
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=environment,
                timeout=14400,
                check=False,
            )
        except subprocess.TimeoutExpired:
            handle.flush()
            _json(
                Path(command_path).with_suffix(".exit.json"),
                {
                    "returncode": None,
                    "status": "hard_timeout_incomplete",
                    "elapsed_s": time.perf_counter() - started,
                    "log_sha256": _sha(log_path),
                },
            )
            raise LifecycleError(
                "worker hard time limit: incomplete, no retry"
            ) from None
    result = {
        "returncode": process.returncode,
        "elapsed_s": time.perf_counter() - started,
        "log_sha256": _sha(log_path),
    }
    _json(Path(command_path).with_suffix(".exit.json"), result)
    _require(process.returncode == 0, "lifecycle worker failed; attempt retained")
    return result


def _boundary(output, destination):
    rows = []
    for ambient in (False, True):
        for operation in OPERATIONS:
            stem = (
                Path(destination)
                / f"public-boundary-{operation}-{str(ambient).lower()}"
            )
            command = [
                sys.executable,
                "-c",
                _PUBLIC_BOUNDARY_CODE,
                str(Path(output) / operation / "model.npz"),
                str(Path(output) / operation / "query.npz"),
                str(ambient).lower(),
            ]
            _launch(
                command, stem.with_suffix(".log"), stem.with_suffix(".command.json")
            )
            row = json.loads(stem.with_suffix(".log").read_text().splitlines()[-1])
            _require(row["finite"], "finite public boundary forecast")
            sources = _sources()
            _require(
                all(sources[name] == digest for name, digest in row["sources"].items()),
                "fresh public module identity",
            )
            row["operation"] = operation
            rows.append(row)
    return rows


def _payloads(directory):
    return {
        str(p.relative_to(directory)): _sha(p)
        for p in sorted(directory.rglob("*"))
        if p.is_file() and p.name != "run.json"
    }


def run(output):
    """Create one immutable attempt; an existing directory is never resumed."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    identity = _identity()
    _json(
        output / "intent.json",
        {
            "identity": identity,
            "operations": list(OPERATIONS),
            "successful_fit_budget": 3,
        },
    )
    result = {
        "identity": identity,
        "operations": list(OPERATIONS),
        "successful_fit_budget": 3,
        "stages": [],
    }
    try:
        for operation in OPERATIONS:
            command = [
                sys.executable,
                "-m",
                MODULE,
                "_worker",
                "--directory",
                str(output),
                "--operation",
                operation,
            ]
            result["stages"].append(
                _launch(
                    command,
                    output / f"{operation}.log",
                    output / f"{operation}.command.json",
                )
            )
        for ambient in (False, True):
            name = "true" if ambient else "false"
            command = [
                sys.executable,
                "-m",
                MODULE,
                "_checks",
                "--directory",
                str(output),
                "--output",
                str(output / f"checks-{name}"),
                "--ambient",
                name,
            ]
            result["stages"].append(
                _launch(
                    command,
                    output / f"checks-{name}.log",
                    output / f"checks-{name}.command.json",
                )
            )
        result["public_boundary"] = _boundary(output, output)
        _require(_identity() == identity, "run source/runtime stable")
        result["status"] = "complete"
    except BaseException as error:
        result.update(
            status="failed", error_type=type(error).__name__, error=str(error)
        )
        raise
    finally:
        result["files"] = _payloads(output)
        _json(output / "run.json", result)
    return result


def replay(directory, expected_sha256, output):
    """Recheck with no numerical fits, initialization or solves.

    Invalid and deliberately stubbed public fit/update calls repeat the frozen
    control-flow checks, never a fourth completed fitting operation.
    """
    directory, output = Path(directory).resolve(), Path(output).resolve()
    _require(
        output != directory and directory not in output.parents,
        "replay output outside sealed attempt",
    )
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "expected_sha256": expected_sha256,
        "no_numerical_fitting": True,
        "stages": [],
    }
    try:
        _require(
            _sha(directory / "run.json") == expected_sha256, "external lifecycle anchor"
        )
        saved = _read(directory / "run.json")
        _require(
            saved["status"] == "complete",
            "only complete lifecycle attempt can pass replay",
        )
        result["identity"] = _identity()
        _require(
            saved["identity"] == result["identity"], "replay source/runtime identity"
        )
        _require(
            saved["operations"] == list(OPERATIONS)
            and saved["successful_fit_budget"] == 3,
            "exact lifecycle operation roster",
        )
        _require(
            _payloads(directory) == saved["files"], "lifecycle payload hashes/roster"
        )
        for ambient in (False, True):
            name = "true" if ambient else "false"
            command = [
                sys.executable,
                "-m",
                MODULE,
                "_checks",
                "--directory",
                str(directory),
                "--output",
                str(output / f"checks-{name}"),
                "--ambient",
                name,
            ]
            result["stages"].append(
                _launch(
                    command,
                    output / f"checks-{name}.log",
                    output / f"checks-{name}.command.json",
                )
            )
            fresh = _read(output / f"checks-{name}/outcome.json")
            old = _read(directory / f"checks-{name}/outcome.json")
            for key in ("verification", "checks", "transforms"):
                # Zip-container timestamps are not semantic lifecycle evidence.
                def normalize(value, key=key):
                    if key == "checks":
                        return [
                            {k: v for k, v in row.items() if k != "sha256"}
                            for row in value
                        ]
                    return value

                _require(
                    normalize(fresh[key]) == normalize(old[key]),
                    f"replayed lifecycle {key}",
                )
        result["public_boundary"] = _boundary(directory, output)
        _require(
            result["public_boundary"] == saved["public_boundary"],
            "public boundary replay",
        )
        _require(
            _payloads(directory) == saved["files"],
            "sealed lifecycle original unchanged",
        )
        result["status"] = "complete"
    except BaseException as error:
        result.update(
            status="failed", error_type=type(error).__name__, error=str(error)
        )
        raise
    finally:
        _json(output / "replay.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("run", "replay", "_worker", "_checks"))
    parser.add_argument("--output")
    parser.add_argument("--directory")
    parser.add_argument("--expected-sha256")
    parser.add_argument("--operation", choices=OPERATIONS)
    parser.add_argument("--ambient", choices=("true", "false"))
    args = parser.parse_args(argv)
    if args.stage == "run":
        if not args.output:
            parser.error("run requires --output")
        run(args.output)
    elif args.stage == "replay":
        if not all((args.directory, args.expected_sha256, args.output)):
            parser.error("replay requires --directory, --expected-sha256 and --output")
        replay(args.directory, args.expected_sha256, args.output)
    else:
        if (
            not args.directory
            or (args.stage == "_worker" and not args.operation)
            or (args.stage == "_checks" and not (args.output and args.ambient))
        ):
            parser.error("worker arguments are incomplete")
        ambient = (
            args.operation == "MC"
            if args.stage == "_worker"
            else args.ambient == "true"
        )
        before = _configuration()
        _require(
            not before["jax_enable_x64"], "worker must start with default x64 false"
        )
        with jax.enable_x64(ambient):
            if args.stage == "_worker":
                _operation(args.directory, args.operation)
            else:
                _checks(args.directory, args.output, ambient)
        _require(_configuration() == before, "caller-owned scope restoration")


if __name__ == "__main__":
    main()
