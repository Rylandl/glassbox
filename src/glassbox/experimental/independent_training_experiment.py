"""Frozen independent-recording intervention through ordinary public update.

No learner, optimizer, public role policy or simulator is replaced. Preparation
checks are read-only; exactly one public update supplies the candidate. Raw
queries and prediction arrays remain the source of every physical reduction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
import traceback
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import jax
import numpy as np

from glassbox import LearnedDynamics, learner
from glassbox import _sequence_model as core
from glassbox._learner_arrays import load_arrays, save_arrays
from glassbox.io.recordings import (
    concatenate_recordings,
    load_recordings,
    save_recordings,
)

from . import public_mean_flight_fit as capture
from . import public_mean_lifecycle as lifecycle
from . import public_mean_physical_evaluation as physical
from . import public_mean_saved_port as saved
from . import state_input_decision as paired
from .excited_bilinear_decision import _angular_repair
from .two_simulator_metrics import aggregate, score_forecast, score_response_directions

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_PATH = "docs/harness/independent-training-recordings-v1.json"
PROTOCOL_SHA256 = "b5fae12bf2b25c88d7a1758214d1faf11749e7a88d8ce4d4e397b04265272adf"
FORMAT = "glassbox-independent-training-candidate-v1"
ARMS = ("baseline", "candidate", "hold")
SIMULATORS = ("crazyflow", "cascade")
ARRAYS = learner._ARRAYS
INITIAL_SUBSET = ("w1", "b1", "w2", "memory", "memory_bias")


class IntegrityError(ValueError):
    """An evidence/source/contract error aborts, rather than losing an arm."""


class PreparationUnavailable(ValueError):
    def __init__(self, details):
        super().__init__("not every frozen training parent has complete window support")
        self.details = details


def require(condition, label):
    if not condition:
        raise IntegrityError(label)


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def inventory(directory, excluding="run.json"):
    directory = Path(directory)
    result = {}
    for path in sorted(directory.rglob("*")):
        require(not path.is_symlink(), "symlink in evidence inventory")
        if path.is_file() and path != directory / excluding:
            result[str(path.relative_to(directory))] = digest(path)
    return result


def sealed(path, expected):
    path = Path(path)
    require(digest(path) == expected, "external stage SHA differs")
    value = read(path)
    require(
        value["files"] == inventory(path.parent, path.name),
        "stage payload inventory differs",
    )
    return value


def _anchor(entry):
    path = Path(entry["path"])
    require(digest(path) == entry["sha256"], "external input SHA differs: " + str(path))
    return path


def roster(protocol):
    item = protocol["dependencies"]["roster"]
    path = ROOT / item["path"]
    require(digest(path) == item["sha256"], "literal roster changed")
    return read(path)


def resolved(protocol):
    """A fresh descriptive view; never mutate a historical module or recipe."""
    root = Path(protocol["imported_evidence"]["expanded_bundle"]["root"])
    run = read(root / "run.json")
    require(
        digest(root / "run.json")
        == protocol["imported_evidence"]["expanded_bundle"]["run_sha256"],
        "expanded source root changed",
    )
    path = root / "resolved-protocol.json"
    require(
        digest(path) == run["files"]["resolved-protocol.json"],
        "resolved source protocol changed",
    )
    result = read(path)
    result["id"] = protocol["id"]
    result["recordings"] = deepcopy(roster(protocol)["confirmation"])
    result["decision"] = deepcopy(protocol["decision"])
    return result


def authenticate(protocol_path, protocol_sha256, binding_path, binding_sha256):
    from .independent_training_data import read_protocol
    from .public_mean_implementation import verify

    require(protocol_sha256 == PROTOCOL_SHA256, "wrong frozen experiment protocol")
    protocol = read_protocol(protocol_path, protocol_sha256)
    binding = verify(binding_path, binding_sha256)
    require(
        Path(binding["public_root"]).resolve() == ROOT,
        "current imported source root differs",
    )
    for name, module in (
        ("learner", learner),
        ("core", core),
        ("capture", capture),
        ("lifecycle", lifecycle),
        ("physical", physical),
        ("saved", saved),
        ("paired", paired),
    ):
        path = Path(module.__file__).resolve()
        require(path.is_relative_to(ROOT), "cross-source import: " + name)
        require(
            digest(path)
            == binding["public_source_sha256"][str(path.relative_to(ROOT))],
            "imported source bytes: " + name,
        )
    for key in (
        "corrected_physical_consumer_replay",
        "corrected_synthetic_lifecycle_replay",
    ):
        stage = protocol["imported_evidence"][key]
        report = sealed(_anchor(stage), stage["sha256"])
        require(report["status"] == "complete", "required corrected replay incomplete")
        required = (
            "physical_and_consumer_regression_passed"
            if "physical" in key
            else "inference_regression_passed"
        )
        require(report[required] is True, "required corrected replay failed")
    return protocol, binding


def import_control(simulator, protocol):
    require(simulator in SIMULATORS, "unknown simulator")
    source = protocol["imported_evidence"]["controls"][simulator]
    manifest = sealed(_anchor(source["manifest"]), source["manifest"]["sha256"])
    for name in ("model", "recordings", "cache", "checkpoint0", "report"):
        path = _anchor(source[name])
        require(
            path.parent == Path(source["directory"])
            and manifest["files"][path.name] == source[name]["sha256"],
            "original fit payload association",
        )
    model = LearnedDynamics.load(source["model"]["path"])
    require(
        model.report == read(source["report"]["path"]), "original report mirror differs"
    )
    require(
        model.report["optimization"]["selected_step"] == source["selected_step"],
        "original selected step changed",
    )
    require(
        model.report["recipe"] == learner.RECIPE == protocol["fitting"]["recipe"],
        "unchanged public recipe",
    )
    cache_meta, cache_arrays = load_arrays(source["cache"]["path"])
    expected_meta, expected_arrays = capture._cache_payload(
        model._train, model._development, model._contract, model._seen
    )
    require(cache_meta == expected_meta, "original actual cache metadata differs")
    capture._same_tree(cache_arrays, expected_arrays, "original actual cache")
    return model, load_recordings(source["recordings"]["path"])


@dataclass(frozen=True)
class PreparedUpdate:
    expected_train: object
    development: object
    contract: dict
    seen: dict
    new_recordings: object
    provenance: dict


def _support(collection, ids):
    steps = learner.steps_for(collection.segments[0].dt_s)
    keys = collection.window_keys(
        history_steps=steps["history"], horizon_steps=steps["horizon"]
    )
    counts = Counter(k.recording_id for k in keys)
    return {name: counts[name] for name in sorted(ids)}


def _overlap(windows):
    targets = Counter(
        (key.recording_id, origin + t)
        for key, origin in zip(windows.keys, windows.source_origins, strict=True)
        for t in range(windows.batch.future_states.shape[1])
    )
    return dict(
        target_transition_visits=sum(targets.values()),
        unique_target_transitions=len(targets),
        repeated_target_transition_visits=sum(targets.values()) - len(targets),
    )


def prepare_update(simulator, control, old_recordings, new_recordings, protocol):
    """Prove actual public merge equals the complete144-parent extraction."""
    plan = roster(protocol)
    old_roles = plan["original_roles"][simulator]
    old_train, development_ids = (
        set(old_roles["training"]),
        set(old_roles["development"]),
    )
    new_ids = {e["id"] for e in plan["added_training"] if e["simulator"] == simulator}
    require(
        len(old_train) == len(new_ids) == 72 and len(development_ids) == 24,
        "fixed role counts",
    )
    require(
        old_train.isdisjoint(development_ids)
        and new_ids.isdisjoint(old_train | development_ids),
        "role overlap",
    )
    contract = learner._contract(old_recordings)
    require(
        contract == control._contract == learner._contract(new_recordings),
        "recording/update contract differs",
    )
    old_seen, new_seen = (
        learner._recording_content(old_recordings),
        learner._recording_content(new_recordings),
    )
    require(
        old_seen == control._seen and set(old_seen) == old_train | development_ids,
        "original full96 ledger differs",
    )
    require(set(new_seen) <= new_ids, "unplanned added recording")
    require(
        set(old_seen.values()).isdisjoint(new_seen.values()),
        "new recording duplicates original content",
    )
    saved.equal_windows(
        learner._extract(old_recordings, old_train, 1536),
        control._train,
        "exact original72 cache",
    )
    saved.equal_windows(
        learner._extract(old_recordings, development_ids, 256),
        control._development,
        "exact original24 development cache",
    )
    support = _support(new_recordings, new_ids)
    if set(new_seen) != new_ids or any(n == 0 for n in support.values()):
        raise PreparationUnavailable(
            dict(
                planned_new_parents=sorted(new_ids),
                loaded_new_parents=sorted(new_seen),
                legal_windows_per_new_parent=support,
                missing_support=sorted(name for name in new_ids if support[name] == 0),
            )
        )
    combined = concatenate_recordings((old_recordings, new_recordings))
    seen = learner._recording_content(combined)
    require(
        seen == {**old_seen, **new_seen} and len(seen) == 168,
        "combined content ledger differs",
    )
    expected = learner._extract(combined, old_train | new_ids, 1536)
    fresh = learner._extract(new_recordings, new_ids, 1536)
    merged = learner._merge_cache(control._train, fresh)
    saved.equal_windows(
        merged, expected, "public update merge versus full144 extraction"
    )
    require(
        len(expected.keys) == 1536
        and {k.recording_id for k in expected.keys} == old_train | new_ids,
        "all144 represented in1536 cache",
    )
    require(len(control._development.keys) == 256, "fixed256 development windows")
    require(
        not expected.excitation_declared
        and not control._development.excitation_declared,
        "physical adapter carries actual commands, no declared sidecars",
    )
    cells = {}
    for entry in plan["added_training"]:
        if entry["simulator"] == simulator:
            cells[entry["id"]] = entry["cell"]
            cells[entry["source_condition_parent"]] = entry["cell"]
    require(set(cells) == old_train | new_ids, "condition-source association differs")
    selected = Counter(k.recording_id for k in expected.keys)
    legal = _support(combined, old_train | new_ids)
    provenance = dict(
        condition_parent_counts={
            "old": dict(sorted(Counter(cells[k] for k in old_train).items())),
            "new": dict(sorted(Counter(cells[k] for k in new_ids).items())),
        },
        selected_condition_windows={
            cohort: dict(
                sorted(
                    Counter(
                        cells[k.recording_id]
                        for k in expected.keys
                        if k.recording_id in ids
                    ).items()
                )
            )
            for cohort, ids in (("old", old_train), ("new", new_ids))
        },
        old_roles=old_roles,
        new_training_ids=sorted(new_ids),
        training_ids=sorted(old_train | new_ids),
        control_fingerprint=control.fingerprint(),
        contract=contract,
        seen=seen,
        actual_merge_equals_full_pool=True,
        training_windows=1536,
        development_windows=256,
        legal_windows_per_training_parent=legal,
        selected_windows_per_training_parent=dict(sorted(selected.items())),
        selected_old_windows=sum(selected[k] for k in old_train),
        selected_new_windows=sum(selected[k] for k in new_ids),
        old_cache_key_overlap=len(set(expected.keys) & set(control._train.keys)),
        training_overlap=_overlap(expected),
        development_overlap=_overlap(control._development),
        cache_fingerprint=saved.window_fingerprint(expected),
        development_fingerprint=saved.window_fingerprint(control._development),
        scope="Complete real parent pool; additional within-window overlap is not independent trajectory evidence.",
    )
    return PreparedUpdate(
        expected, control._development, contract, seen, new_recordings, provenance
    )


def save_preparation(directory, prepared):
    save_arrays(
        Path(directory) / "expected-preparation.npz",
        *capture._cache_payload(
            prepared.expected_train,
            prepared.development,
            prepared.contract,
            prepared.seen,
        ),
    )
    write(Path(directory) / "preparation.json", prepared.provenance)
    save_recordings(prepared.new_recordings, Path(directory) / "added-recordings.npz")


def _check_preparation_file(directory, prepared, name):
    meta, arrays = load_arrays(Path(directory) / name)
    expected_meta, expected_arrays = capture._cache_payload(
        prepared.expected_train, prepared.development, prepared.contract, prepared.seen
    )
    require(meta == expected_meta, "saved preparation metadata differs")
    capture._same_tree(arrays, expected_arrays, "saved actual preparation")


def _initial_model(model, values):
    metadata = model._model.metadata().copy()
    metadata.pop("format")
    return core.SequenceModel(
        **metadata,
        params={k[6:]: v for k, v in values.items() if k.startswith("param_")},
        norms={k[5:]: v for k, v in values.items() if k.startswith("norm_")},
    )


def check_work(directory, model, prepared, control, protocol, simulator):
    """Within-fit links and independent arithmetic reductions; no solves/gradients."""
    directory = Path(directory)
    _check_preparation_file(directory, prepared, "expected-preparation.npz")
    _check_preparation_file(directory, prepared, "actual-preparation.npz")
    require(
        read(directory / "preparation.json") == prepared.provenance,
        "preparation provenance changed",
    )
    saved.equal_windows(
        model._train, prepared.expected_train, "selected training cache"
    )
    saved.equal_windows(
        model._development, prepared.development, "selected development cache"
    )
    require(
        model._seen == prepared.seen and model._contract == prepared.contract,
        "selected model input ledger differs",
    )
    require(
        model.report["previous_revision"] == control.fingerprint(),
        "previous revision differs from exact imported control",
    )
    require(
        model.report["recipe"] == protocol["fitting"]["recipe"] == learner.RECIPE,
        "candidate recipe differs",
    )
    initial = capture._npz(directory / "initializer-return.npz")
    initial_model = _initial_model(model, initial)
    base_initial = capture._npz(
        _anchor(protocol["imported_evidence"]["controls"][simulator]["checkpoint0"])
    )
    for name in INITIAL_SUBSET:
        capture._same(
            initial["param_" + name],
            base_initial["param_" + name],
            "data-independent initializer " + name,
        )
    norms = saved.training_norms(prepared.expected_train.batch)
    for name, value in norms.items():
        capture._same(initial["norm_" + name], value, "training norm " + name)
        capture._same(model._model.norms[name], value, "selected norm " + name)
    b = prepared.expected_train.batch
    weights = capture._npz(directory / "weights.npz")
    require(set(weights) == set(capture.WEIGHTS), "weight witness array roster")
    hold = np.repeat(b.past_states[:, -1:], b.future_states.shape[1], 1)
    scale = np.maximum(
        np.sqrt(np.mean((hold - b.future_states) ** 2, 0)), 0.01 * norms["state_scale"]
    )
    opt = model.report["optimization"]
    require(
        type(opt["steps"]) is int
        and opt["steps"] == 1000
        and type(opt["batch_size"]) is int
        and opt["batch_size"] == len(b.past_states) == 1536,
        "actual optimization budget metadata differs",
    )
    require(
        opt["ridge"] == 0.01 * len(b.past_states) * b.future_states.shape[1]
        and opt["delay_steps"] == learner.steps_for(b.dt_s)["delay"],
        "actual optimization ridge/delay differs",
    )
    capture._same(np.asarray(opt["error_scale"]), scale, "reported hold scales")
    require(
        opt["selection_objective"] == core.OBJECTIVE,
        "reported selection objective differs",
    )
    prediction = weights["initial_training_prediction"]
    require(
        prediction.dtype == np.float64
        and prediction.shape == b.future_states.shape
        and np.isfinite(prediction).all(),
        "initial weighting forecast schema",
    )
    mse = np.mean(((prediction - b.future_states) / scale) ** 2, axis=(0, 1))
    floor = 0.01**2
    raw = 1 / np.maximum(mse, floor)
    normalizer = np.mean(raw)
    channel_weights = raw / normalizer
    actual = dict(
        normalization=scale,
        initial_training_prediction=prediction,
        initial_channel_mse=mse,
        raw_channel_weights=raw,
        channel_weights=channel_weights,
        fixed_weights=channel_weights,
        weight_floor=np.asarray(floor),
        weight_normalizer=np.asarray(normalizer),
    )
    capture._same_tree(weights, actual, "arithmetic objective reduction")
    require(
        model.report["optimization"]["objective"]
        == core.weighting_metadata(
            initial_model, prediction, mse, raw, channel_weights, floor, normalizer
        ),
        "objective report differs",
    )
    witness = SimpleNamespace(
        initial_arrays=initial,
        selected_arrays=model._model.arrays(),
        objective_arrays=weights,
        ridge=0.01 * len(b.past_states) * b.future_states.shape[1],
        delay=learner.steps_for(b.dt_s)["delay"],
    )
    return capture._work(directory, model, witness)


def _cache_diagnostics(directory, model, prepared):
    """Only initializer/selected states; reuse actual initial training forecast."""
    directory = Path(directory)
    initial = _initial_model(model, capture._npz(directory / "initializer-return.npz"))
    weights = capture._npz(directory / "weights.npz")
    output = {}
    arrays = {}
    new_ids = set(prepared.provenance["new_training_ids"])
    with jax.enable_x64(True):
        for step, predictor in (("initial", initial), ("selected", model._model)):
            for role, windows in (
                ("training", prepared.expected_train),
                ("development", prepared.development),
            ):
                b = windows.batch
                prediction = (
                    weights["initial_training_prediction"]
                    if step == "initial" and role == "training"
                    else np.asarray(
                        predictor.rollout(b.past_states, b.past_inputs, b.future_inputs)
                    )
                )
                require(np.isfinite(prediction).all(), "nonfinite cache diagnostic")
                arrays[step + "_" + role + "_prediction"] = prediction
                partitions = {"all": np.ones(len(windows.keys), dtype=bool)}
                if role == "training":
                    partitions.update(
                        old=np.asarray(
                            [k.recording_id not in new_ids for k in windows.keys]
                        ),
                        new=np.asarray(
                            [k.recording_id in new_ids for k in windows.keys]
                        ),
                    )
                for partition, mask in partitions.items():
                    residual = prediction[mask] - b.future_states[mask]
                    sq = (residual / weights["normalization"]) ** 2
                    groups = {
                        name: np.sqrt(
                            np.mean(residual[:, :, sl] ** 2, axis=(0, 2))
                        ).tolist()
                        for name, sl in (
                            ("velocity_m_s", slice(0, 3)),
                            ("body_rate_rad_s", slice(3, 6)),
                            ("rotation_entries", slice(6, 15)),
                        )
                    }
                    output[f"{step}/{role}/{partition}"] = dict(
                        windows=int(mask.sum()),
                        rmse_by_native_horizon=groups,
                        weighted_loss=float(np.mean(sq * weights["channel_weights"])),
                        original_loss=float(np.mean(sq)),
                        channel_original_contribution=np.mean(sq, axis=(0, 1)).tolist(),
                        channel_weighted_contribution=(
                            np.mean(sq, axis=(0, 1)) * weights["channel_weights"]
                        ).tolist(),
                    )
    return arrays, output


def perform_update(simulator, directory, control, prepared, protocol):
    """Observe the one actual public call; do not replace numerical arithmetic."""
    directory = Path(directory)
    save_preparation(directory, prepared)
    before = control.fingerprint()
    configuration = capture._configuration()
    require(not configuration["jax_enable_x64"], "update begins in ambient default32")
    original_train, original_fit = learner._train, learner.fit_sequence_model
    original_init, original_calibrate = (
        core.initialize_sequence_model,
        learner._calibrate,
    )
    timing = dict(
        public_update_calls=0,
        training_entry_calls=0,
        initializer_calls=0,
        fitter_calls=0,
        calibration_calls=0,
        configuration_before=configuration,
    )
    started = time.perf_counter()

    def train(t, d, c, s, **kw):
        timing["training_entry_calls"] += 1
        require(timing["training_entry_calls"] == 1, "multiple actual training entries")
        save_arrays(
            directory / "actual-preparation.npz", *capture._cache_payload(t, d, c, s)
        )
        _check_preparation_file(directory, prepared, "actual-preparation.npz")
        require(
            d is control._development and kw.get("previous") == before,
            "actual public update parent/development differs",
        )
        return original_train(t, d, c, s, **kw)

    def initialize(batch, **kw):
        timing["initializer_calls"] += 1
        require(timing["initializer_calls"] == 1, "multiple actual initializers")
        write(
            directory / "initializer-entered.json",
            dict(
                jax_enable_x64=bool(jax.config.x64_enabled),
                training_windows=len(batch.past_states),
                ridge=kw["ridge"],
                settings={k: kw[k] for k in ("seed", "width", "memory", "delay_steps")},
            ),
        )
        value = original_init(batch, **kw)
        np.savez_compressed(directory / "initializer-return.npz", **value.arrays())
        return value

    def fit(*a, **kw):
        timing["fitter_calls"] += 1
        require(timing["fitter_calls"] == 1, "multiple numerical fitters")
        start = time.perf_counter()
        try:
            return original_fit(*a, **kw)
        finally:
            timing["fitter_including_capture_wall_s"] = time.perf_counter() - start

    def calibrate(*a, **kw):
        timing["calibration_calls"] += 1
        require(timing["calibration_calls"] == 1, "multiple calibrations")
        start = time.perf_counter()
        try:
            return original_calibrate(*a, **kw)
        finally:
            timing["calibration_wall_s"] = time.perf_counter() - start

    try:
        with (directory / "work.jsonl").open("x") as handle:

            def observe(state):
                if state["phase"] == "weights":
                    np.savez_compressed(
                        directory / "weight-initial.npz",
                        **capture._tree(state["params"], state["norms"]),
                    )
                lifecycle._capture_event(state, directory, handle)

            with (
                patch.object(learner, "_train", train),
                patch.object(learner, "fit_sequence_model", fit),
                patch.object(core, "initialize_sequence_model", initialize),
                patch.object(learner, "_calibrate", calibrate),
                patch.object(core, "_observe_attempt", observe),
            ):
                timing["public_update_calls"] += 1
                model = control.update(prepared.new_recordings)
        require(
            control.fingerprint() == before, "public update mutated original revision"
        )
        require(
            capture._configuration() == configuration,
            "public update changed ambient precision",
        )
        model.save(directory / "model.npz")
        write(directory / "report.json", model.report)
        write(
            directory / "work-verification.json",
            check_work(directory, model, prepared, control, protocol, simulator),
        )
        diagnostic_arrays, diagnostics = _cache_diagnostics(directory, model, prepared)
        np.savez_compressed(directory / "cache-diagnostics.npz", **diagnostic_arrays)
        write(directory / "cache-diagnostics.json", diagnostics)
        return model
    finally:
        timing["public_update_and_verification_wall_s"] = time.perf_counter() - started
        timing["original_unchanged"] = control.fingerprint() == before
        timing["configuration_after"] = capture._configuration()
        write(directory / "timing.json", timing)


def _load_update_inputs(simulator, protocol, training_root, training_sha256):
    from . import independent_training_data as data

    data.verify_data(
        training_root,
        training_sha256,
        kind="training",
        simulator=simulator,
        protocol=protocol,
    )
    control, old = import_control(simulator, protocol)
    try:
        new = data.load_added_recordings(
            training_root, training_sha256, control.contract
        )
    except data.PreparationUnavailable as error:
        raise PreparationUnavailable(error.details) from error
    return control, prepare_update(simulator, control, old, new, protocol)


def _update_worker(request, protocol, binding):
    directory = Path(request["output"])
    simulator = request["simulator"]
    outcome = dict(simulator=simulator, status="integrity_failure", stage="preparation")
    try:
        control, old = import_control(simulator, protocol)
        shutil.copy2(
            protocol["imported_evidence"]["controls"][simulator]["model"]["path"],
            directory / "control.npz",
        )
        from . import independent_training_data as data

        data.verify_data(
            request["training_root"],
            request["training_sha256"],
            kind="training",
            simulator=simulator,
            protocol=protocol,
        )
        start = time.perf_counter()
        try:
            new = data.load_added_recordings(
                request["training_root"], request["training_sha256"], control.contract
            )
        except data.PreparationUnavailable as error:
            raise PreparationUnavailable(error.details) from error
        prepared = prepare_update(simulator, control, old, new, protocol)
        outcome["preparation_wall_s"] = time.perf_counter() - start
        outcome["stage"] = "public_update"
        model = perform_update(simulator, directory, control, prepared, protocol)
        outcome.update(
            status="complete",
            stage="complete",
            model_fingerprint=model.fingerprint(),
            previous_revision=model.report["previous_revision"],
            selected_step=model.report["optimization"]["selected_step"],
        )
    except PreparationUnavailable as error:
        outcome.update(status="preparation_unavailable", details=error.details)
    except core.SequenceFitError as error:
        outcome.update(
            status="fit_failed",
            error_type=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc(),
        )
    except (np.linalg.LinAlgError, FloatingPointError) as error:
        outcome.update(
            status="fit_failed",
            error_type=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc(),
        )
    except BaseException as error:
        outcome.update(
            error_type=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc(),
        )
        raise
    finally:
        outcome["observed_prefix"] = capture._prefix(directory, incomplete=True)
        write(directory / "outcome.json", outcome)
    return outcome


def _supervise(
    kind,
    simulator,
    output,
    *,
    protocol_path,
    protocol_sha256,
    binding_path,
    binding_sha256,
    **inputs,
):
    _protocol, binding = authenticate(
        protocol_path, protocol_sha256, binding_path, binding_sha256
    )
    require(simulator in SIMULATORS, "unknown simulator")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    request = dict(
        kind=kind,
        simulator=simulator,
        output=str(output),
        protocol_path=str(Path(protocol_path).resolve()),
        protocol_sha256=protocol_sha256,
        binding_path=str(Path(binding_path).resolve()),
        binding_sha256=binding_sha256,
        **inputs,
    )
    write(output / "request.json", request)
    env = dict(os.environ)
    env.pop("JAX_ENABLE_X64", None)
    env.update(PYTHONPATH=str(ROOT / "src"), SCIPY_ARRAY_API="1")
    command = [
        binding["interpreter"],
        "-m",
        "glassbox.experimental.independent_training_experiment",
        "--worker",
        str(output / "request.json"),
    ]
    write(
        output / "command.json",
        dict(
            argv=command,
            hard_timeout_s=14400,
            environment={
                k: env.get(k)
                for k in ("PYTHONPATH", "SCIPY_ARRAY_API", "JAX_ENABLE_X64")
            },
        ),
    )
    started = time.perf_counter()
    result = dict(
        format=FORMAT
        if kind == "update"
        else "glassbox-independent-training-evaluation-v1",
        simulator=simulator,
        status="incomplete",
        binding_sha256=binding_sha256,
        implementation_sha256=binding_sha256,
        protocol_sha256=protocol_sha256,
    )
    try:
        with (output / "worker.log").open("x") as log:
            process = subprocess.run(
                command,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=14400,
                check=False,
            )
        result["returncode"] = process.returncode
        require(process.returncode == 0, "worker failed; retained without retry")
        outcome = read(output / "outcome.json")
        require(
            outcome["status"] in ("complete", "preparation_unavailable", "fit_failed"),
            "invalid terminal scientific outcome",
        )
        result["status"] = outcome["status"]
        authenticate(protocol_path, protocol_sha256, binding_path, binding_sha256)
    except BaseException as error:
        result.update(
            status="hard_timeout_incomplete"
            if isinstance(error, subprocess.TimeoutExpired)
            else "integrity_failure",
            error_type=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc(),
        )
        raise
    finally:
        result.update(elapsed_s=time.perf_counter() - started, files=inventory(output))
        write(output / "run.json", result)
    return result


def update_candidate(simulator, output, *, training_root, training_sha256, **context):
    return _supervise(
        "update",
        simulator,
        output,
        training_root=str(Path(training_root).resolve()),
        training_sha256=training_sha256,
        **context,
    )


def _check_timing(timing):
    require(
        all(
            type(timing[key]) is int and timing[key] == 1
            for key in (
                "public_update_calls",
                "training_entry_calls",
                "initializer_calls",
                "fitter_calls",
                "calibration_calls",
            )
        ),
        "single actual update/initializer/calibration count differs",
    )
    require(
        timing["original_unchanged"] is True, "original revision immutability witness"
    )
    require(
        timing["configuration_before"] == timing["configuration_after"]
        and timing["configuration_before"]["jax_enable_x64"] is False,
        "update ambient precision restoration witness",
    )


def verify_candidate(output, expected_sha256, *, protocol, replay_diagnostics=False):
    output = Path(output)
    run = sealed(output / "run.json", expected_sha256)
    require(
        run["format"] == FORMAT and run["protocol_sha256"] == PROTOCOL_SHA256,
        "candidate stage format/protocol",
    )
    require(
        run["status"]
        in (
            "complete",
            "preparation_unavailable",
            "fit_failed",
            "hard_timeout_incomplete",
        ),
        "unusable candidate stage",
    )
    request = read(output / "request.json")
    simulator = request["simulator"]
    require(simulator == run["simulator"], "candidate simulator differs")
    outcome = (
        read(output / "outcome.json") if (output / "outcome.json").exists() else None
    )
    if run["status"] == "hard_timeout_incomplete":
        require(
            run.get("error_type") == "TimeoutExpired", "hard timeout reason differs"
        )
        return None, dict(
            status=run["status"], prefix=capture._prefix(output, incomplete=True)
        )
    require(
        outcome is not None and outcome["status"] == run["status"],
        "candidate outcome mirror differs",
    )
    try:
        control, prepared = _load_update_inputs(
            simulator, protocol, request["training_root"], request["training_sha256"]
        )
    except PreparationUnavailable as error:
        require(
            run["status"] == "preparation_unavailable"
            and outcome["details"] == error.details,
            "preparation unavailable reason changed",
        )
        require(
            not any(
                (output / p).exists()
                for p in (
                    "model.npz",
                    "work.jsonl",
                    "actual-preparation.npz",
                    "initializer-entered.json",
                )
            ),
            "unavailable preparation contains fit artifacts",
        )
        return None, dict(status="preparation_unavailable", details=error.details)
    require(
        run["status"] != "preparation_unavailable", "preparation is actually available"
    )
    if run["status"] == "fit_failed":
        require(
            outcome.get("error_type")
            in ("SequenceFitError", "LinAlgError", "FloatingPointError"),
            "undeclared numerical failure",
        )
        _check_preparation_file(output, prepared, "expected-preparation.npz")
        return None, dict(
            status="fit_failed", prefix=capture._prefix(output, incomplete=True)
        )
    _check_timing(read(output / "timing.json"))
    _anchor(
        dict(
            path=str(output / "control.npz"),
            sha256=protocol["imported_evidence"]["controls"][simulator]["model"][
                "sha256"
            ],
        )
    )
    model = LearnedDynamics.load(output / "model.npz")
    require(model.report == read(output / "report.json"), "candidate report mirror")
    require(
        model.fingerprint() == outcome["model_fingerprint"]
        and outcome["stage"] == "complete"
        and outcome["previous_revision"] == model.report["previous_revision"]
        and outcome["selected_step"] == model.report["optimization"]["selected_step"],
        "candidate revision/selection mirrors differ",
    )
    work = check_work(output, model, prepared, control, protocol, simulator)
    require(
        read(output / "work-verification.json") == work,
        "work verification mirror differs",
    )
    if replay_diagnostics:
        arrays, report = _cache_diagnostics(output, model, prepared)
        capture._same_tree(
            arrays,
            capture._npz(output / "cache-diagnostics.npz"),
            "cache diagnostic replay",
        )
        require(
            report == read(output / "cache-diagnostics.json"),
            "cache diagnostic report replay",
        )
        physical.calibration_evidence(model)
    return model, dict(
        status="complete", model_fingerprint=model.fingerprint(), work=work
    )


def _prediction_arrays(model, query, values):
    if model is None:
        return np.full(values["target"].shape, np.nan, dtype=np.float32)
    return physical.predict_query(model, query, values, dtype=np.float32)


def _evaluation_worker(request, protocol, binding):
    from . import independent_training_data as data

    require(not jax.config.x64_enabled, "physical forecast worker must be default32")
    directory = Path(request["output"])
    simulator = request["simulator"]
    data.verify_data(
        request["data_root"],
        request["data_sha256"],
        kind="confirmation",
        simulator=simulator,
        protocol=protocol,
    )
    model, verification = verify_candidate(
        request["candidate_root"], request["candidate_sha256"], protocol=protocol
    )
    control, _ = import_control(simulator, protocol)
    source = resolved(protocol)
    queries = read(Path(request["data_root"]) / "queries.json")
    with jax.enable_x64(True):
        physical.reconstruct_truth(request["data_root"], simulator, source)
    require(not jax.config.x64_enabled, "telemetry scope must restore default32")
    functions = {
        arm: jax.jit(m.predict) if m is not None else None
        for arm, m in (("baseline", control), ("candidate", model))
    }
    envelopes = {
        arm: np.asarray(m.envelope())
        for arm, m in (("baseline", control), ("candidate", model))
        if m is not None
    }
    np.savez_compressed(directory / "envelopes.npz", **envelopes)
    calls = 0
    finite = True
    for query in queries:
        values = physical.arrays(physical.under(request["data_root"], query["path"]))
        predictions = {}
        for arm, function in functions.items():
            predictions[arm] = _prediction_arrays(function, query, values)
            if function is not None and query["history_eligible"]:
                n = physical.finite_command_prefix(values["future_inputs"])
                calls += bool(n)
                finite = finite and bool(np.isfinite(predictions[arm][:n]).all())
            if query["kind"] == "response":
                predictions[arm + "_factual"] = (
                    np.full(values["target"].shape, np.nan, dtype=np.float32)
                    if function is None
                    else physical.predict_query(
                        function, query, values, dtype=np.float32, factual=True
                    )
                )
                if function is not None and query["history_eligible"]:
                    n = physical.finite_command_prefix(values["factual_inputs"])
                    calls += bool(n)
                    finite = finite and bool(
                        np.isfinite(predictions[arm + "_factual"][:n]).all()
                    )
        path = directory / physical.query_path(query)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **predictions)
    scored = score(request["data_root"], directory, simulator, source)
    write(directory / "metrics.json", scored)
    outcome = dict(
        status="complete",
        simulator=simulator,
        queries=len(queries),
        native_forecast_calls=calls,
        all_input_eligible_predictions_finite=finite,
        candidate_available=model is not None,
        candidate_verification=verification,
        model_fingerprints=dict(
            baseline=control.fingerprint(),
            candidate=model.fingerprint() if model is not None else None,
        ),
        fits=0,
        initializers=0,
    )
    write(directory / "outcome.json", outcome)
    return outcome


def evaluate(
    simulator,
    output,
    *,
    data_root,
    data_sha256,
    candidate_root,
    candidate_sha256,
    **context,
):
    return _supervise(
        "evaluate",
        simulator,
        output,
        data_root=str(Path(data_root).resolve()),
        data_sha256=data_sha256,
        candidate_root=str(Path(candidate_root).resolve()),
        candidate_sha256=candidate_sha256,
        **context,
    )


def validate_rows(rows, queries, p, simulator):
    """Exact planned slot multiplicity, independent of predictions or truth success."""
    from .two_simulator_metrics import _horizons

    expected = set()
    for q in queries:
        for _, horizon in _horizons(
            p["generation"][simulator]["dt_s"], p["evaluation"]["horizons_s"]
        ):
            for group in p["decision"]["aggregation"]["groups"]:
                for statistic in ("endpoint", "cumulative"):
                    for arm in ARMS:
                        expected.add(
                            (
                                q["parent"],
                                q["id"],
                                q["kind"],
                                arm,
                                horizon,
                                group,
                                statistic,
                            )
                        )
    actual = [
        (
            r["parent"],
            r["query"],
            r["kind"],
            r["arm"],
            r["horizon_s"],
            r["group"],
            r["statistic"],
        )
        for r in rows
    ]
    require(
        len(actual) == len(set(actual)) and set(actual) == expected,
        "complete metric slots differ",
    )
    require(all(r["simulator"] == simulator for r in rows), "metric simulator differs")
    planned = {(q["parent"], q["id"]): q for q in queries}
    require(len(planned) == len(queries), "duplicate planned query identity")
    for row in rows:
        query = planned[row["parent"], row["query"]]
        require(
            all(
                row[key] == query[key]
                for key in ("parent", "scope", "cell", "origin", "kind")
            )
            and row.get("channel") == query.get("channel")
            and row.get("sign") == query.get("sign"),
            "metric identity differs from planned query",
        )
    for arm in ("candidate", "hold"):
        view = {
            simulator: [
                dict(r, arm="candidate" if r["arm"] == arm else "baseline")
                for r in rows
                if r["arm"] in ("baseline", arm)
            ]
        }
        local = deepcopy(p)
        local["cells"] = {simulator: p["cells"][simulator]}
        paired.validate_cohorts(view, local)


def score(data_root, evaluation_root, simulator, p):
    """Pinned physical operators over exactly baseline/candidate/hold raw arrays."""
    _, queries, _ = physical._planned(data_root, simulator, p)
    evaluation_root = Path(evaluation_root)
    envelopes = physical.arrays(evaluation_root / "envelopes.npz")
    require(
        set(envelopes) in ({"baseline"}, {"baseline", "candidate"}),
        "envelope arm roster differs",
    )
    rows, directions, pairs = [], [], {}
    dt, horizons = p["generation"][simulator]["dt_s"], p["evaluation"]["horizons_s"]
    for query in queries:
        values = physical.arrays(physical.under(data_root, query["path"]))
        predictions = physical.arrays(evaluation_root / physical.query_path(query))
        wanted = {"baseline", "candidate"} | (
            {"baseline_factual", "candidate_factual"}
            if query["kind"] == "response"
            else set()
        )
        require(set(predictions) == wanted, "prediction arm roster differs")
        for value in predictions.values():
            require(
                value.dtype == np.float32 and value.shape == values["target"].shape,
                "native32 prediction shape/dtype",
            )

        def hold(x, u, f):
            return np.repeat(x[-1:], len(f), axis=0)

        predictions["hold"] = physical.predict_query(
            hold, query, values, dtype=np.float64
        )
        target = values["target"]
        if query["kind"] == "response":
            predictions["hold_factual"] = physical.predict_query(
                hold, query, values, dtype=np.float64, factual=True
            )
            predictions = {
                arm: np.asarray(predictions[arm], dtype=float)
                - np.asarray(predictions[arm + "_factual"], dtype=float)
                for arm in ARMS
            }
            target = target - values["factual_target"]
            key = (query["parent"], query["origin"], query["channel"])
            if query["sign"] < 0:
                require(key not in pairs, "duplicate lower response")
                pairs[key] = (predictions, target, values["valid"])
            else:
                require(key in pairs, "missing lower response")
                lower, truth, valid = pairs.pop(key)
                for arm in ARMS:
                    scored = score_response_directions(
                        np.stack((lower[arm], predictions[arm])),
                        np.stack((truth, target)),
                        np.stack((valid, values["valid"])),
                        dt_s=dt,
                        horizons_s=horizons,
                        thresholds=p["evaluation"]["responses"][
                            "weak_endpoint_thresholds"
                        ],
                    )
                    directions.extend(
                        dict(
                            row,
                            simulator=simulator,
                            scope=query["scope"],
                            cell=query["cell"],
                            parent=query["parent"],
                            origin=query["origin"],
                            channel=query["channel"],
                            arm=arm,
                        )
                        for row in scored
                    )
        for arm in ARMS:
            scored = score_forecast(
                predictions[arm],
                target,
                values["valid"],
                dt_s=dt,
                horizons_s=horizons,
                envelope=envelopes.get(arm) if query["kind"] == "factual" else None,
                rotation_geometry=query["kind"] == "factual",
            )
            rows.extend(
                dict(
                    row,
                    simulator=simulator,
                    scope=query["scope"],
                    cell=query["cell"],
                    parent=query["parent"],
                    query=query["id"],
                    origin=query["origin"],
                    channel=query.get("channel"),
                    sign=query.get("sign"),
                    kind=query["kind"],
                    arm=arm,
                )
                for row in scored
            )
    require(not pairs, "unpaired response")
    nonweak = {
        (r["parent"], r["origin"], r["channel"], r["horizon_s"], r["group"]): r[
            "pair_nonweak"
        ]
        for r in directions
    }
    for row in rows:
        if row["kind"] == "response":
            row["pair_nonweak"] = nonweak[
                row["parent"],
                row["origin"],
                row["channel"],
                row["horizon_s"],
                row["group"],
            ]
    validate_rows(rows, queries, p, simulator)
    return dict(rows=rows, directions=directions, summary=aggregate(rows))


def decide(rows, protocol, *, all_available=True, all_input_finite=True):
    p = resolved(protocol)
    require(set(rows) == set(SIMULATORS), "decision simulator roster differs")
    for values in rows.values():
        require(
            set(r["arm"] for r in values) == set(ARMS), "decision arm roster differs"
        )
    summaries = {sim: aggregate(values) for sim, values in rows.items()}
    for arm in ("candidate", "hold"):
        paired.validate_cohorts(
            {
                sim: [
                    dict(r, arm="candidate" if r["arm"] == arm else "baseline")
                    for r in values
                    if r["arm"] in ("baseline", arm)
                ]
                for sim, values in rows.items()
            },
            p,
        )
    comparison = paired.reduce(summaries, p, rows=rows)
    angular_protocol = deepcopy(p)
    angular_protocol["decision"]["angular_repair"] = deepcopy(
        p["decision"]["angular_retention"]
    )
    angular_protocol["decision"]["angular_repair"]["denominator"] = "quadratic"
    angular_summaries = {
        sim: aggregate(
            [
                dict(r, arm="quadratic" if r["arm"] == "baseline" else r["arm"])
                for r in values
                if r["arm"] in ("baseline", "candidate")
            ]
        )
        for sim, values in rows.items()
    }
    angular = _angular_repair(angular_summaries, angular_protocol, comparison)
    angular.update(
        denominator_arm="baseline",
        baseline_fields_mean="baseline",
        policy=deepcopy(p["decision"]["angular_retention"]),
    )
    require(
        type(all_available) is bool and type(all_input_finite) is bool,
        "availability/finiteness must be booleans",
    )
    checks = dict(
        broad_residual_progress=comparison["residual_criteria_pass"],
        angular_retention=angular["residual_criteria_pass"],
        all_models_available=all_available,
        all_input_eligible_predictions_finite=all_input_finite,
    )
    return dict(
        comparison=comparison,
        angular_retention=angular,
        checks=checks,
        residual_criteria_pass=all(checks.values()),
        public_promotion=False,
        scope="Bounded offline update accuracy on this fixed collection intervention; no universal improvement, safe live swap, uncertainty or controller qualification.",
    )


def _stage_source(record, binding_sha256, label):
    require(
        record["protocol_sha256"] == PROTOCOL_SHA256
        and record["implementation_sha256"] == binding_sha256
        and record.get("binding_sha256", binding_sha256) == binding_sha256,
        label + " source association",
    )


def _confirmation_associations(actual, stages):
    expected = {}
    for simulator in SIMULATORS:
        entry = stages[simulator]["candidate"]
        path = Path(entry["root"]).resolve() / "run.json"
        candidate = sealed(path, entry["sha256"])
        expected[simulator] = dict(
            path=str(path), sha256=entry["sha256"], status=candidate["status"]
        )
    require(actual == expected, "confirmation exact candidate outcome associations")


def _checked_stages(stages, protocol, binding_sha256):
    from . import independent_training_data as data

    require(set(stages) == set(SIMULATORS), "stage simulator roster")
    rows = {}
    available = True
    finite = True
    for simulator, entries in stages.items():
        require(
            set(entries) == {"training", "candidate", "confirmation", "evaluation"},
            "stage kind roster",
        )
        training, confirmation = entries["training"], entries["confirmation"]
        for kind, entry in (("training", training), ("confirmation", confirmation)):
            data_seal = data.verify_data(
                entry["root"],
                entry["sha256"],
                kind=kind,
                simulator=simulator,
                protocol=protocol,
            )
            _stage_source(data_seal, binding_sha256, kind + " data")
            data_root = Path(entry["root"]).resolve()
            stage_path = data_root.parent / "run.json"
            data_stage = sealed(stage_path, digest(stage_path))
            _stage_source(data_stage, binding_sha256, kind + " stage")
            require(
                data_stage["format"] == data.STAGE_FORMAT
                and data_stage["status"] == "complete"
                and data_stage["kind"] == kind
                and data_stage["simulator"] == simulator
                and data_stage["data_sha256"] == entry["sha256"],
                "data stage identity/seal association",
            )
            data_request = read(data_root.parent / "request.json")
            require(
                Path(data_request["directory"]).resolve() == data_root
                and data_request["binding_sha256"] == binding_sha256
                and data_request["protocol_sha256"] == PROTOCOL_SHA256,
                "data stage request association",
            )
            if kind == "confirmation":
                _confirmation_associations(data_seal["candidate_outcomes"], stages)
        candidate = entries["candidate"]
        candidate_stage = sealed(
            Path(candidate["root"]) / "run.json", candidate["sha256"]
        )
        _stage_source(candidate_stage, binding_sha256, "candidate")
        model, _ = verify_candidate(
            candidate["root"], candidate["sha256"], protocol=protocol
        )
        available = available and model is not None
        evaluation = entries["evaluation"]
        root = Path(evaluation["root"])
        stage = sealed(root / "run.json", evaluation["sha256"])
        _stage_source(stage, binding_sha256, "evaluation")
        require(
            stage["status"] == "complete" and stage["simulator"] == simulator,
            "evaluation stage incomplete",
        )
        request = read(root / "request.json")
        require(
            request["candidate_sha256"] == candidate["sha256"]
            and Path(request["candidate_root"]).resolve()
            == Path(candidate["root"]).resolve(),
            "evaluation candidate association",
        )
        require(
            request["data_sha256"] == confirmation["sha256"]
            and Path(request["data_root"]).resolve()
            == Path(confirmation["root"]).resolve(),
            "evaluation confirmation association",
        )
        candidate_request = read(Path(candidate["root"]) / "request.json")
        for stage_request in (request, candidate_request):
            require(
                stage_request["binding_sha256"] == binding_sha256
                and stage_request["protocol_sha256"] == PROTOCOL_SHA256,
                "candidate/evaluation request source association",
            )
        require(
            candidate_request["training_sha256"] == training["sha256"]
            and Path(candidate_request["training_root"]).resolve()
            == Path(training["root"]).resolve(),
            "candidate training association",
        )
        reduced = score(confirmation["root"], root, simulator, resolved(protocol))
        require(
            reduced == read(root / "metrics.json"),
            "raw prediction metric reduction differs",
        )
        outcome = read(root / "outcome.json")
        require(
            outcome["candidate_available"] == (model is not None),
            "candidate availability report differs",
        )
        # Independent of truth masks: every available learned arm must be finite
        # on every finite command prefix. Later missing truth never hides NaNs.
        actual_finite = True
        calls = 0
        for query in read(Path(confirmation["root"]) / "queries.json"):
            values = physical.arrays(
                physical.under(confirmation["root"], query["path"])
            )
            predictions = physical.arrays(root / physical.query_path(query))
            for arm in ("baseline", "candidate"):
                if arm == "candidate" and model is None:
                    for suffix in (
                        ("", "_factual") if query["kind"] == "response" else ("",)
                    ):
                        require(
                            np.isnan(predictions[arm + suffix]).all(),
                            "unavailable candidate has predictions",
                        )
                    continue
                for suffix, commands in (("", values["future_inputs"]),) + (
                    (("_factual", values["factual_inputs"]),)
                    if query["kind"] == "response"
                    else ()
                ):
                    n = (
                        physical.finite_command_prefix(commands)
                        if query["history_eligible"]
                        else 0
                    )
                    calls += bool(n)
                    actual_finite = actual_finite and bool(
                        np.isfinite(predictions[arm + suffix][:n]).all()
                    )
                    require(
                        np.isnan(predictions[arm + suffix][n:]).all(),
                        "prediction padding differs from input eligibility",
                    )
        require(
            outcome["native_forecast_calls"] == calls
            and outcome["all_input_eligible_predictions_finite"] == actual_finite,
            "prediction availability/finiteness summary differs",
        )
        finite = finite and actual_finite
        rows[simulator] = reduced["rows"]
    return rows, available, finite


def finalize(
    output,
    *,
    stages,
    protocol_path,
    binding_path,
    protocol_sha256=PROTOCOL_SHA256,
    binding_sha256,
):
    """Seal the complete experiment after reducing its saved raw evidence."""
    protocol, binding = authenticate(
        protocol_path, protocol_sha256, binding_path, binding_sha256
    )
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    require(not (output / "run.json").exists(), "experiment already finalized")
    for entries in stages.values():
        for entry in entries.values():
            require(
                Path(entry["root"]).resolve().is_relative_to(output),
                "final bundle must include every stage payload",
            )
    rows, available, finite = _checked_stages(stages, protocol, binding_sha256)
    decision = decide(rows, protocol, all_available=available, all_input_finite=finite)
    write(output / "stage-links.json", stages)
    write(output / "decision.json", decision)
    result = dict(
        format="glassbox-independent-training-experiment-v1",
        protocol_sha256=protocol_sha256,
        binding_sha256=binding_sha256,
        implementation_commit=binding["implementation_commit"],
        status="complete",
        stages=stages,
        decision_sha256=digest(output / "decision.json"),
        residual_criteria_pass=decision["residual_criteria_pass"],
        public_promotion=False,
        files=inventory(output),
    )
    write(output / "run.json", result)
    return result


def verify_bundle(
    output,
    expected_sha256,
    *,
    protocol_path,
    binding_path,
    protocol_sha256=PROTOCOL_SHA256,
    binding_sha256,
):
    protocol, binding = authenticate(
        protocol_path, protocol_sha256, binding_path, binding_sha256
    )
    output = Path(output)
    result = sealed(output / "run.json", expected_sha256)
    require(
        result["format"] == "glassbox-independent-training-experiment-v1"
        and result["status"] == "complete"
        and result["protocol_sha256"] == protocol_sha256
        and result["binding_sha256"] == binding_sha256
        and result["implementation_commit"] == binding["implementation_commit"],
        "experiment source association",
    )
    stages = read(output / "stage-links.json")
    require(stages == result["stages"], "stage links mirror differs")
    rows, available, finite = _checked_stages(stages, protocol, binding_sha256)
    decision = decide(rows, protocol, all_available=available, all_input_finite=finite)
    require(
        decision == read(output / "decision.json")
        and digest(output / "decision.json") == result["decision_sha256"],
        "saved decision reduction differs",
    )
    require(
        result["residual_criteria_pass"] == decision["residual_criteria_pass"]
        and result["public_promotion"] is False,
        "outer decision mirror differs",
    )
    return result


def replay(
    simulator,
    output,
    expected_sha256,
    *,
    replay_output,
    protocol_path,
    binding_path,
    protocol_sha256=PROTOCOL_SHA256,
    binding_sha256,
):
    """Regenerate new physics and current predictions, never the update fit."""
    from . import independent_training_data as data

    context = dict(
        protocol_path=protocol_path,
        protocol_sha256=protocol_sha256,
        binding_path=binding_path,
        binding_sha256=binding_sha256,
    )
    result = verify_bundle(output, expected_sha256, **context)
    protocol, _ = authenticate(**context)
    destination = Path(replay_output).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    entries = result["stages"][simulator]
    prerequisites = {
        sim: dict(
            path=str(Path(result["stages"][sim]["candidate"]["root"]) / "run.json"),
            sha256=result["stages"][sim]["candidate"]["sha256"],
        )
        for sim in SIMULATORS
    }
    data.collect_training(
        simulator,
        destination / "training",
        replay_data=entries["training"]["root"],
        expected_data_sha256=entries["training"]["sha256"],
        **context,
    )
    data.collect_confirmation(
        simulator,
        destination / "confirmation",
        replay_data=entries["confirmation"]["root"],
        expected_data_sha256=entries["confirmation"]["sha256"],
        candidate_outcomes=prerequisites,
        **context,
    )
    _, work = verify_candidate(
        entries["candidate"]["root"],
        entries["candidate"]["sha256"],
        protocol=protocol,
        replay_diagnostics=True,
    )
    evaluate(
        simulator,
        destination / "evaluation",
        data_root=entries["confirmation"]["root"],
        data_sha256=entries["confirmation"]["sha256"],
        candidate_root=entries["candidate"]["root"],
        candidate_sha256=entries["candidate"]["sha256"],
        **context,
    )
    original = Path(entries["evaluation"]["root"])
    fresh = destination / "evaluation"
    original_npz = {str(p.relative_to(original)) for p in original.rglob("*.npz")}
    fresh_npz = {str(p.relative_to(fresh)) for p in fresh.rglob("*.npz")}
    require(original_npz == fresh_npz, "replay prediction file roster")
    arrays = 0
    for relative in sorted(original_npz):
        before, after = (
            physical.arrays(original / relative),
            physical.arrays(fresh / relative),
        )
        capture._same_tree(after, before, "fresh prediction replay " + relative)
        arrays += len(before)
    require(
        read(original / "metrics.json") == read(fresh / "metrics.json"),
        "fresh physical reduction differs",
    )
    report = dict(
        format="glassbox-independent-training-replay-v1",
        status="complete",
        simulator=simulator,
        source_run_sha256=expected_sha256,
        exact_arrays=arrays,
        exact_metrics=True,
        work=work,
        fits=0,
        initializers=0,
        files=inventory(destination),
    )
    write(destination / "run.json", report)
    return report


def _worker(request_path):
    request = read(request_path)
    protocol, binding = authenticate(
        *(
            request[k]
            for k in (
                "protocol_path",
                "protocol_sha256",
                "binding_path",
                "binding_sha256",
            )
        )
    )
    require(
        Path(request["output"]).resolve() == Path(request_path).resolve().parent,
        "worker output/request association",
    )
    if request["kind"] == "update":
        outcome = _update_worker(request, protocol, binding)
    elif request["kind"] == "evaluate":
        outcome = _evaluation_worker(request, protocol, binding)
    else:
        raise IntegrityError("unknown worker kind")
    authenticate(
        *(
            request[k]
            for k in (
                "protocol_path",
                "protocol_sha256",
                "binding_path",
                "binding_sha256",
            )
        )
    )
    return outcome


# Append before the experiment CLI. Uses the experiment module's existing imports.
# These audits run only after the implementation is committed and science sealed.


def _audit_close(actual, expected, label):
    require(
        (actual is None and expected is None)
        or (
            actual is not None
            and expected is not None
            and bool(np.isclose(actual, expected, rtol=1e-12, atol=1e-15))
        ),
        "independent reduction differs: " + label,
    )


def _audit_mean(values):
    values = list(values)
    if not values:
        return None
    with np.errstate(over="ignore", invalid="ignore"):
        result = float(np.sum(np.asarray(values, dtype=np.float64) / len(values)))
    return result if np.isfinite(result) else None


def _audit_hierarchy(rows):
    """All eligible errors, branches→origin→parent→cell; no finite-subset fallback."""
    eligible = [r for r in rows if r["truth_eligible"]]
    if not eligible or any(r["mse"] is None for r in eligible):
        return None
    cells = {}
    for r in eligible:
        cells.setdefault(r["cell"], {}).setdefault(r["parent"], {}).setdefault(
            r["origin"], []
        ).append(r["mse"])
    value = _audit_mean(
        _audit_mean(
            _audit_mean(_audit_mean(v) for v in origins.values())
            for origins in parents.values()
        )
        for parents in cells.values()
    )
    return None if value is None else float(np.sqrt(value))


def _audit_scalar(prediction, target, valid, step, columns, statistic):
    eligible = step <= len(valid) and bool(np.asarray(valid[:step]).all())
    if not eligible:
        return False, False, None
    times = slice(step - 1, step) if statistic == "endpoint" else slice(0, step)
    predicted = np.asarray(prediction, dtype=np.float64)[times, columns]
    finite = bool(np.isfinite(predicted).all())
    with np.errstate(over="ignore", invalid="ignore"):
        squared = np.square(
            predicted - np.asarray(target, dtype=np.float64)[times, columns]
        )
    mse = (
        _audit_mean(squared.ravel()) if finite and np.isfinite(squared).all() else None
    )
    return True, finite, mse


def _audit_hold(query, values, factual=False):
    commands = values["factual_inputs" if factual else "future_inputs"]
    prediction = np.full(values["target"].shape, np.nan, dtype=np.float64)
    if query["history_eligible"]:
        finite = np.isfinite(commands).all(axis=1)
        invalid = np.flatnonzero(~finite)
        n = int(invalid[0]) if len(invalid) else len(commands)
        prediction[:n] = values["past_states"][-1]
    return prediction


def independent_reduce(output, expected_sha256, *, audit_output, **context):
    """NumPy-only scientific reduction after ordinary source/inventory validation.

    Recomputes every raw endpoint/cumulative MSE and all hierarchical means/tails.
    Bootstrap draws, response-direction diagnostics and policy Boolean logic retain
    their explicitly attributed frozen canonical validators, not a second policy.
    """
    source = verify_bundle(output, expected_sha256, **context)
    protocol, _ = authenticate(**context)
    p = resolved(protocol)
    destination = Path(audit_output).resolve()
    require(
        not destination.is_relative_to(Path(output).resolve()),
        "audit must be outside original bundle",
    )
    destination.mkdir(parents=True, exist_ok=False)
    group_columns = {
        "velocity_m_s": slice(0, 3),
        "body_rate_rad_s": slice(3, 6),
        "rotation_entries": slice(6, 15),
    }
    counts = dict(raw_rows=0, summaries=0, parents=0, cells=0, tails=0)
    independent = {}
    for simulator, stages in source["stages"].items():
        data_root, evaluation = (
            Path(stages[k]["root"]) for k in ("confirmation", "evaluation")
        )
        scored = read(evaluation / "metrics.json")
        indexed = {}
        for row in scored["rows"]:
            indexed.setdefault((row["parent"], row["query"]), []).append(row)
        rows = []
        for query in read(data_root / "queries.json"):
            values = physical.arrays(physical.under(data_root, query["path"]))
            predictions = physical.arrays(evaluation / physical.query_path(query))
            predictions["hold"] = _audit_hold(query, values)
            target = np.asarray(values["target"], dtype=np.float64)
            if query["kind"] == "response":
                predictions["hold_factual"] = _audit_hold(query, values, True)
                predictions = {
                    a: np.asarray(predictions[a], dtype=np.float64)
                    - np.asarray(predictions[a + "_factual"], dtype=np.float64)
                    for a in ARMS
                }
                target = target - np.asarray(values["factual_target"], dtype=np.float64)
            for row in indexed[query["parent"], query["id"]]:
                eligible, finite, mse = _audit_scalar(
                    predictions[row["arm"]],
                    target,
                    values["valid"],
                    row["horizon_steps"],
                    group_columns[row["group"]],
                    row["statistic"],
                )
                require(
                    (eligible, finite)
                    == (row["truth_eligible"], row["prediction_finite"]),
                    "independent truth/finiteness masks differ",
                )
                _audit_close(mse, row["mse"], "raw " + query["id"])
                rows.append(
                    dict(
                        row, mse=mse, truth_eligible=eligible, prediction_finite=finite
                    )
                )
                counts["raw_rows"] += 1
        independent[simulator] = rows
        grouping = (
            "scope",
            "simulator",
            "kind",
            "arm",
            "horizon_s",
            "group",
            "statistic",
        )
        for kind, extra in (
            ("summaries", ()),
            ("parents", ("cell", "parent")),
            ("cells", ("cell",)),
        ):
            keys = grouping + extra
            groups = {}
            for row in rows:
                groups.setdefault(tuple(row[k] for k in keys), []).append(row)
            require(
                len(groups) == len(scored["summary"][kind]),
                "independent aggregate roster",
            )
            for summary in scored["summary"][kind]:
                selected = groups[tuple(summary[k] for k in keys)]
                _audit_close(
                    _audit_hierarchy(selected), summary["available_truth_rmse"], kind
                )
                require(
                    summary["planned"] == len(selected)
                    and summary["truth_eligible"]
                    == sum(r["truth_eligible"] for r in selected)
                    and summary["failed"]
                    == sum(r["truth_eligible"] and r["mse"] is None for r in selected),
                    "independent aggregate availability differs",
                )
                counts[kind] += 1
    decision = read(Path(output) / "decision.json")
    comparison = decision["comparison"]
    policy = p["decision"]["aggregation"]
    all_rows = [r for rows in independent.values() for r in rows]
    comparison_keys = ("simulator", "scope", "kind", "horizon_s", "group")
    tail_keys = ("simulator", "scope", "kind", "arm", "horizon_s", "group", "statistic")
    comparison_groups, tail_groups = {}, {}
    for row in all_rows:
        tail_groups.setdefault(tuple(row[k] for k in tail_keys), []).append(row)
        if row["statistic"] == "endpoint":
            comparison_groups.setdefault(
                tuple(row[k] for k in comparison_keys), []
            ).append(row)
    ratios = []
    for comparison_row in comparison["physical_comparisons"]:
        selected = comparison_groups[tuple(comparison_row[k] for k in comparison_keys)]
        b, c = (
            _audit_hierarchy([r for r in selected if r["arm"] == arm])
            for arm in ("baseline", "candidate")
        )
        floor = policy["floors"][comparison_row["group"]]
        ratio = None if b is None or c is None else max(c, floor) / max(b, floor)
        for name, value in (
            ("baseline_rmse", b),
            ("candidate_rmse", c),
            ("ratio", ratio),
        ):
            _audit_close(value, comparison_row[name], name)
        ratios.append(dict(comparison_row, ratio=ratio))
    for kind, original in comparison["weighted_geometric_mean_ratios"].items():
        selected = [r for r in ratios if r["kind"] == kind]
        weights = np.asarray(
            [
                policy["simulator_weights"][r["simulator"]]
                * policy["scope_weights"][r["scope"]]
                for r in selected
            ]
        )
        value = (
            None
            if any(r["ratio"] is None for r in selected)
            else float(
                np.exp(
                    np.sum(
                        weights / weights.sum() * np.log([r["ratio"] for r in selected])
                    )
                )
            )
        )
        _audit_close(value, original, "weighted " + kind)
    for tail in comparison["parent_error_tails"]:
        selected = tail_groups[tuple(tail[k] for k in tail_keys)]
        parents = {}
        for row in selected:
            parents.setdefault((row["cell"], row["parent"]), []).append(row)
        values = [_audit_hierarchy(rows) for rows in parents.values()]
        available = [v for v in values if v is not None]
        for name, value in (
            ("p50", float(np.quantile(available, 0.5)) if available else None),
            ("p95", float(np.quantile(available, 0.95)) if available else None),
            ("max", max(available) if available else None),
        ):
            _audit_close(value, tail[name], "parent " + name)
        require(
            tail["planned_parents"] == len(parents)
            and tail["available_parents"] == len(available),
            "independent parent availability",
        )
        counts["tails"] += 1
    result = dict(
        source_run_sha256=expected_sha256,
        passed=True,
        counts=counts,
        comparison_ratios=comparison["weighted_geometric_mean_ratios"],
        fits=0,
        predictions=0,
        initializers=0,
        scope="Independent NumPy raw MSE, matched masks, hierarchy, parent quantiles and aggregate ratios. Frozen canonical validators separately check directions, bootstrap and policy.",
    )
    write(destination / "result.json", result)
    write(destination / "run.json", dict(result, files=inventory(destination)))
    sealed(Path(output) / "run.json", expected_sha256)
    return result


def _audit_replace(path, value):
    path = Path(path)
    path.unlink()
    write(path, value)


def _audit_reseal(directory):
    """Repair local payload inventories only; immutable external anchors stay fixed."""
    from . import independent_training_data as data

    directory = Path(directory)
    manifests = [
        p for p in directory.rglob("*.json") if p.name in ("seal.json", "run.json")
    ]
    for path in sorted(manifests, key=lambda p: len(p.parts), reverse=True):
        value = read(path)
        if not isinstance(value, dict) or "files" not in value:
            continue
        if path.name == "seal.json" and "arrays" in value:
            value["arrays"] = data._array_count(path.parent)
        value["files"] = inventory(path.parent, path.name)
        _audit_replace(path, value)
    return digest(directory / "run.json")


def _audit_rejection(call, expected_text):
    try:
        call()
    except ValueError as error:
        require(expected_text in str(error), "wrong semantic rejection: " + str(error))
        return str(error)
    raise IntegrityError("alteration was accepted: " + expected_text)


def alteration_audit(output, expected_sha256, *, audit_output, **context):
    """Exactly ITR-A01..08; disposable copies, fixed external root, named checks.

    Case A03 performs native training replay. A07 makes one selected-control
    query prediction. Neither case fits or initializes a model. Other cases
    reduce or validate saved evidence. Missing candidate material is unavailable,
    never a successful integrity challenge. Source/root associations in copies
    remain historical: semantic checks target the explicitly named copied stage,
    not a fabricated newly executed full experiment.
    """
    from . import independent_training_data as data

    original = Path(output).resolve()
    source = verify_bundle(original, expected_sha256, **context)
    protocol, _ = authenticate(**context)
    destination = Path(audit_output).resolve()
    require(
        not destination.is_relative_to(original),
        "audit must be outside original bundle",
    )
    destination.mkdir(parents=True, exist_ok=False)
    outcomes = []
    for number in range(1, 9):
        identity = f"ITR-A{number:02d}"
        case = destination / identity
        case.mkdir()
        write(
            case / "command.json",
            dict(
                entrypoint="alteration_audit",
                case=identity,
                source_run_sha256=expected_sha256,
                context={k: str(v) for k, v in context.items()},
            ),
        )
        copied = case / "mutant"
        shutil.copytree(original, copied, copy_function=shutil.copy2)
        simulator = next(
            (
                s
                for s in SIMULATORS
                if source["stages"][s]["candidate"]
                and read(Path(source["stages"][s]["candidate"]["root"]) / "run.json")[
                    "status"
                ]
                == "complete"
            ),
            None,
        )
        if number not in (1, 4):
            simulator = SIMULATORS[0]
        started = time.monotonic()
        record = dict(
            case=identity,
            status="failed",
            passed=False,
            fits=0,
            initializers=0,
            scope="Locally resealed disposable payload, unchanged frozen external root, named semantic validation.",
        )
        try:
            if simulator is None:
                record.update(
                    status="unavailable",
                    reason="No complete candidate carries the required cache/role evidence.",
                )
                continue
            stages = source["stages"][simulator]
            paths = {
                k: copied / Path(v["root"]).resolve().relative_to(original)
                for k, v in stages.items()
            }
            validator = None
            expected_error = None
            if number in (1, 4):
                verify_candidate(
                    paths["candidate"],
                    digest(paths["candidate"] / "run.json"),
                    protocol=protocol,
                )
            elif number in (2, 3, 6):
                clean_root = paths["confirmation" if number == 6 else "training"]
                data.verify_data(
                    clean_root, digest(clean_root / "seal.json"), protocol=protocol
                )
            record["clean_semantic_precheck_passed"] = True
            # A05 is additionally checked against the pinned imported archive below;
            # A07 executes fresh equality below; A08 uses the verified original decision.
            if number == 1:
                path = paths["candidate"] / "preparation.json"
                value = read(path)
                roles = value["old_roles"]
                training, development = roles["training"], roles["development"]
                old, moved = training[0], development[0]
                training[0], development[0] = moved, old
                value["training_ids"] = sorted(
                    moved if k == old else k for k in value["training_ids"]
                )
                _audit_replace(path, value)

                def validator(paths=paths):
                    return verify_candidate(
                        paths["candidate"],
                        digest(paths["candidate"] / "run.json"),
                        protocol=protocol,
                    )

                expected_error = "preparation provenance changed"
            elif number == 2:
                root = paths["training"]
                records = read(root / "records.json")
                removed = records.pop()
                _audit_replace(root / "records.json", records)
                planned = read(root / "planned-records.json")
                _audit_replace(root / "planned-records.json", planned[:-1])
                for relative in (
                    removed["prefix"] + ".json",
                    removed["prefix"] + ".npz",
                    "recordings/" + removed["id"] + ".npz",
                ):
                    (root / relative).unlink(missing_ok=True)
                seal = read(root / "seal.json")
                seal["parents"] = len(records)
                seal["support"] = data.support(
                    records, read(root / "recording-contract.json")["dt_s"]
                )
                _audit_replace(root / "seal.json", seal)

                def validator(root=root):
                    return data.verify_data(
                        root, digest(root / "seal.json"), protocol=protocol
                    )

                expected_error = "literal data roster"
            elif number == 3:
                root = paths["training"]
                records = read(root / "records.json")
                row = next(
                    (r for r in records if (root / (r["prefix"] + ".npz")).exists()),
                    None,
                )
                if row is None:
                    record.update(
                        status="unavailable",
                        reason="No actual native training trajectory exists.",
                    )
                    continue
                path = root / (row["prefix"] + ".npz")
                values = data._load_npz(path)
                config = read(root / "configuration.json")
                channel = next(
                    c for c in config["spec"]["channels"] if c["kind"] == "control"
                )
                old = values["excitation_pilot_commands"][0, 0]
                new = (channel["minimum"] + channel["maximum"]) / 2
                if new == old:
                    new = channel["minimum"] + 0.25 * (
                        channel["maximum"] - channel["minimum"]
                    )
                values["excitation_pilot_commands"][0, 0] = new
                values["commands"][0, 0] = new
                # Before .75s the assignment and realized perturbation are exactly zero.
                np.savez_compressed(path, **values)
                ordinary = root / ("recordings/" + row["id"] + ".npz")
                if ordinary.exists():
                    obs = data._load_npz(ordinary)
                    obs["inputs"][0, 0] = new
                    np.savez_compressed(ordinary, **obs)

                def native_rejection(
                    root=root, simulator=simulator, case=case, row=row
                ):
                    data.verify_data(
                        root, digest(root / "seal.json"), protocol=protocol
                    )
                    try:
                        data.collect_training(
                            simulator,
                            case / "native-replay",
                            replay_data=root,
                            expected_data_sha256=digest(root / "seal.json"),
                            **context,
                        )
                    except ValueError:
                        logs = "\n".join(
                            p.read_text(errors="replace")
                            for p in (case / "native-replay").rglob("*.log")
                        )
                        require(
                            "fresh array bytes differ" in logs and row["id"] in logs,
                            "native replay failed for an unrelated reason",
                        )
                        raise IntegrityError(
                            "native issued-command replay rejected"
                        ) from None

                validator, expected_error = (
                    native_rejection,
                    "native issued-command replay rejected",
                )
                record["scope"] += (
                    " Assigned/clipped command algebra must pass first; regenerated pilot/physics must reject. Native replay performs no fitting."
                )
            elif number == 4:
                path = paths["candidate"] / "actual-preparation.npz"
                metadata, arrays = load_arrays(path)
                key = next(k for k in arrays if "future_inputs" in k and "train" in k)
                arrays[key].flat[0] += 1.0
                save_arrays(path, metadata, arrays)

                def validator(paths=paths):
                    return verify_candidate(
                        paths["candidate"],
                        digest(paths["candidate"] / "run.json"),
                        protocol=protocol,
                    )

                expected_error = (
                    "saved actual preparation:train_future_inputs is not byte-identical"
                )
            elif number == 5:
                entry = protocol["imported_evidence"]["controls"][simulator]["model"]
                path = paths["candidate"] / "control.npz"
                if not path.exists():
                    record.update(
                        status="unavailable",
                        reason="No saved imported-control payload exists.",
                    )
                    continue
                require(
                    digest(path) == entry["sha256"],
                    "copied control does not match frozen original",
                )
                metadata, arrays = load_arrays(path)
                key = next(
                    k
                    for k, v in arrays.items()
                    if np.issubdtype(v.dtype, np.floating) and v.size
                )
                arrays[key].flat[0] += 1.0
                save_arrays(path, metadata, arrays)

                def validator(entry=entry, path=path):
                    return _anchor(dict(path=str(path), sha256=entry["sha256"]))

                expected_error = "external input SHA differs"
                record["scope"] = (
                    "Declared immutable external control identity case; internal archive fingerprint repaired, frozen model SHA intentionally retained."
                )
            elif number == 6:
                root = paths["confirmation"]
                records = read(root / "records.json")
                training = read(paths["training"] / "planned-records.json")[0]
                records[0].update(id=training["id"], seed=training["seed"])
                _audit_replace(root / "records.json", records)
                planned = read(root / "planned-records.json")
                planned[0].update(id=training["id"], seed=training["seed"])
                _audit_replace(root / "planned-records.json", planned)
                _audit_replace(root / (records[0]["prefix"] + ".json"), records[0])

                def validator(root=root):
                    return data.verify_data(
                        root, digest(root / "seal.json"), protocol=protocol
                    )

                expected_error = "literal data roster"
            elif number == 7:
                queries = read(paths["confirmation"] / "queries.json")
                selected = None
                for query in queries:
                    if query["kind"] != "factual" or not query["history_eligible"]:
                        continue
                    values = physical.arrays(
                        physical.under(paths["confirmation"], query["path"])
                    )
                    if physical.finite_command_prefix(values["future_inputs"]):
                        selected = (query, values)
                        break
                if selected is None:
                    record.update(
                        status="unavailable",
                        reason="No input-eligible selected-control factual query.",
                    )
                    continue
                query, values = selected
                control, _ = import_control(simulator, protocol)
                require(not jax.config.x64_enabled, "A07 requires default32 inference")
                fresh = physical.predict_query(
                    jax.jit(control.predict), query, values, dtype=np.float32
                )
                path = paths["evaluation"] / physical.query_path(query)
                arrays = physical.arrays(path)
                require(
                    np.array_equal(fresh, arrays["baseline"], equal_nan=True),
                    "A07 unchanged selected prediction mismatch",
                )
                before = arrays["baseline"][0, 0]
                arrays["baseline"][0, 0] += np.float32(1.0)
                if arrays["baseline"][0, 0] == before:
                    arrays["baseline"][0, 0] = np.nextafter(before, np.float32(0.0))
                require(
                    arrays["baseline"][0, 0] != before, "A07 mutation made no change"
                )
                np.savez_compressed(path, **arrays)
                _audit_replace(
                    paths["evaluation"] / "metrics.json",
                    score(
                        paths["confirmation"],
                        paths["evaluation"],
                        simulator,
                        resolved(protocol),
                    ),
                )

                def validator(fresh=fresh, path=path):
                    return require(
                        np.array_equal(
                            fresh, physical.arrays(path)["baseline"], equal_nan=True
                        ),
                        "selected query prediction replay differs",
                    )

                expected_error = "selected query prediction replay differs"
                record.update(predictions=1, query=query["id"], arm="baseline")
            else:
                path = copied / "decision.json"
                value = read(path)
                value["residual_criteria_pass"] = not value["residual_criteria_pass"]
                _audit_replace(path, value)
                outer = read(copied / "run.json")
                outer.update(
                    decision_sha256=digest(path),
                    residual_criteria_pass=value["residual_criteria_pass"],
                )
                _audit_replace(copied / "run.json", outer)

                def decision_rejection(path=path):
                    rows, available, finite = _checked_stages(
                        source["stages"], protocol, context["binding_sha256"]
                    )
                    require(
                        decide(
                            rows,
                            protocol,
                            all_available=available,
                            all_input_finite=finite,
                        )
                        == read(path),
                        "saved acceptance reduction differs",
                    )

                validator, expected_error = (
                    decision_rejection,
                    "saved acceptance reduction differs",
                )
            new_sha = _audit_reseal(copied)
            record.update(
                mutant_run_sha256=new_sha,
                external_rejection=_audit_rejection(
                    lambda copied=copied: sealed(copied / "run.json", expected_sha256),
                    "external stage SHA differs",
                ),
            )
            record["semantic_rejection"] = _audit_rejection(validator, expected_error)
            record.update(status="detected", passed=True)
        except Exception as error:
            record.update(
                status="audit_error",
                error_type=type(error).__name__,
                error=str(error),
                traceback=traceback.format_exc(),
            )
        finally:
            record["elapsed_s"] = time.monotonic() - started
            record["mutated_files"] = {
                k: v
                for k, v in inventory(copied).items()
                if source["files"].get(k) != v
            }
            write(case / "result.json", record)
            write(
                case / "exit.json",
                dict(returncode=0 if record["passed"] else 1, status=record["status"]),
            )
            (case / "audit.log").write_text(
                json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + "\n"
            )
            outcomes.append(record)
            shutil.rmtree(copied)
    require(
        digest(original / "run.json") == expected_sha256, "original bundle root changed"
    )
    sealed(original / "run.json", expected_sha256)
    result = dict(
        source_run_sha256=expected_sha256,
        cases=outcomes,
        required_cases=8,
        passed=len(outcomes) == 8 and all(r["passed"] for r in outcomes),
        fits=0,
        initializers=0,
    )
    write(destination / "result.json", result)
    write(destination / "run.json", dict(result, files=inventory(destination)))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker")
    parser.add_argument(
        "--stage",
        choices=(
            "update",
            "evaluate",
            "finalize",
            "verify",
            "replay",
            "independent",
            "alterations",
        ),
    )
    parser.add_argument("--simulator", choices=SIMULATORS)
    parser.add_argument("--output")
    parser.add_argument("--protocol-path", default=str(ROOT / PROTOCOL_PATH))
    parser.add_argument("--protocol-sha256", default=PROTOCOL_SHA256)
    parser.add_argument("--binding-path")
    parser.add_argument("--binding-sha256")
    parser.add_argument("--training-root")
    parser.add_argument("--training-sha256")
    parser.add_argument("--data-root")
    parser.add_argument("--data-sha256")
    parser.add_argument("--candidate-root")
    parser.add_argument("--candidate-sha256")
    parser.add_argument("--stages-json")
    parser.add_argument("--expected-sha256")
    parser.add_argument("--replay-output")
    parser.add_argument("--audit-output")
    args = parser.parse_args(argv)
    if args.worker:
        return _worker(args.worker)
    require(
        args.stage and args.output and args.binding_path and args.binding_sha256,
        "stage output and external binding required",
    )
    context = {
        k: getattr(args, k)
        for k in ("protocol_path", "protocol_sha256", "binding_path", "binding_sha256")
    }
    if args.stage == "update":
        return update_candidate(
            args.simulator,
            args.output,
            training_root=args.training_root,
            training_sha256=args.training_sha256,
            **context,
        )
    if args.stage == "evaluate":
        return evaluate(
            args.simulator,
            args.output,
            data_root=args.data_root,
            data_sha256=args.data_sha256,
            candidate_root=args.candidate_root,
            candidate_sha256=args.candidate_sha256,
            **context,
        )
    if args.stage == "finalize":
        return finalize(args.output, stages=read(args.stages_json), **context)
    if args.stage == "verify":
        return verify_bundle(args.output, args.expected_sha256, **context)
    if args.stage in ("independent", "alterations"):
        require(args.audit_output, "exclusive external audit output required")
        function = (
            independent_reduce if args.stage == "independent" else alteration_audit
        )
        return function(
            args.output, args.expected_sha256, audit_output=args.audit_output, **context
        )
    return replay(
        args.simulator,
        args.output,
        args.expected_sha256,
        replay_output=args.replay_output,
        **context,
    )


if __name__ == "__main__":
    main()
