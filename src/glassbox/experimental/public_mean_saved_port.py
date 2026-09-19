"""Authenticated, no-fit research-to-public fixtures and preparation witnesses.

This qualification adapter is not a public archive migration path. It preserves
the historical report separately and never initializes, fits, calibrates or
predicts. Quaternion conversion is the original recording adapter arithmetic;
all learned arrays are read from externally authenticated archives.
"""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

import jax
import jax.numpy as jnp
import numpy as np

from glassbox._learner_arrays import array_fingerprint, load_arrays
from glassbox._sequence_model import (
    SequenceBatch,
    SequenceModel,
    _features,
    autonomous_features,
    interaction_features,
)
from glassbox.core.data import TrajectorySpec
from glassbox.core.geometry import quaternion_to_rotation_batch
from glassbox.io.recordings import load_recordings, save_recordings
from glassbox.learner import (
    RECIPE,
    LearnedDynamics,
    _contract,
    _extract,
    _priority,
    _recording_content,
    _subset,
    steps_for,
)
from glassbox.recordings import (
    SequenceCollection,
    SequenceSegment,
    SequenceWindows,
    WindowKey,
)

from .public_mean_scoring import PROTOCOL_SHA256

BUNDLE_SHA256 = "88b59d60be28ffed1e93fa6e6ecaa026b1d49950a23f7f2b1e0f9b65b3b9d99d"
SIMULATORS = ("crazyflow", "cascade")
UNTRUSTED_REFERENCE_REPORTS = ("crazyflow/replay.json", "cascade/replay.json")
FORMAT = "glassbox-public-mean-saved-port-v1"
ARRAYS = ("past_states", "past_inputs", "future_inputs", "future_states")
OBSERVED_CHANNELS = (
    "velocity_north [m/s,world_nwu]",
    "velocity_west [m/s,world_nwu]",
    "velocity_up [m/s,world_nwu]",
    "body_rate_x [rad/s,body_flu]",
    "body_rate_y [rad/s,body_flu]",
    "body_rate_z [rad/s,body_flu]",
) + tuple(
    f"rotation_{i}{j} [unitless,body_flu_to_world_nwu]"
    for i in range(3)
    for j in range(3)
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path):
    return json.loads(Path(path).read_text())


def _write(path, value):
    with Path(path).open("x") as handle:
        handle.write(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _under(root, relative):
    _require(
        isinstance(relative, str)
        and relative
        and "\\" not in relative
        and not relative.startswith("/")
        and all(part not in ("", ".", "..") for part in relative.split("/")),
        "invalid bundle-relative path",
    )
    current = Path(root)
    for part in PurePosixPath(relative).parts:
        current /= part
        _require(not current.is_symlink(), "bundle paths cannot contain symlinks")
    _require(current.is_file(), "bundle payload is not a regular file")
    return current


def _inventory(root, manifest_name, expected_sha256, *, ignored_auxiliary=()):
    root = Path(root)
    _require(
        digest(_under(root, manifest_name)) == expected_sha256,
        "manifest differs from external anchor",
    )
    manifest = _json(root / manifest_name)
    files = manifest["files"]
    _require(isinstance(files, dict) and files, "empty payload inventory")
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
    _require(
        not set(files).intersection(ignored_auxiliary),
        "science payload cannot be excluded",
    )
    _require(
        actual - set(ignored_auxiliary) == set(files) | {manifest_name},
        "bundle payload roster differs",
    )
    for relative, expected in files.items():
        _require(
            digest(_under(root, relative)) == expected,
            "bundle payload changed: " + relative,
        )
    return manifest


def authenticate_reference(reference_root, *, expected_bundle_sha256):
    """Authenticate the frozen protocol, dependencies and complete research bundle."""
    _require(expected_bundle_sha256 == BUNDLE_SHA256, "wrong research bundle anchor")
    repository = Path(__file__).resolve().parents[3]
    protocol_path = repository / "docs/harness/public-mean-qualification-v1.json"
    _require(digest(protocol_path) == PROTOCOL_SHA256, "qualification protocol changed")
    protocol = _json(protocol_path)
    for dependency in protocol["protocol_dependencies"]:
        _require(
            digest(_under(repository, dependency["path"])) == dependency["sha256"],
            "qualification dependency changed: " + dependency["path"],
        )
    _require(
        protocol["candidate_source"]["manifest_sha256"] == BUNDLE_SHA256,
        "frozen candidate source differs",
    )
    # These two reports were written after the scientific run was finalized.
    # Their contents are not read, authenticated, or relied upon by this stage.
    return _inventory(
        reference_root,
        "run.json",
        expected_bundle_sha256,
        ignored_auxiliary=UNTRUSTED_REFERENCE_REPORTS,
    )


def array_equal(actual, expected, label):
    """Scientific equality includes array names, dtype, shape and exact bytes."""
    _require(set(actual) == set(expected), label + ": array roster differs")
    for key, value in actual.items():
        wanted = expected[key]
        _require(
            value.dtype == wanted.dtype
            and value.shape == wanted.shape
            and value.tobytes() == wanted.tobytes(),
            label + ": " + key,
        )


def _window_metadata(windows):
    return {
        "keys": [asdict(k) for k in windows.keys],
        "source_origins": list(windows.source_origins),
    }


def window_arrays(windows):
    result = {key: getattr(windows.batch, key) for key in ARRAYS}
    if windows.excitation_declared:
        result.update(
            {
                key: getattr(windows, key)
                for key in ("past_excitation", "future_excitation")
            }
        )
    return result


def window_fingerprint(windows):
    return array_fingerprint(
        {**_window_metadata(windows), "dt_s": windows.batch.dt_s},
        {key: getattr(windows.batch, key) for key in ARRAYS},
    )


def excitation_fingerprint(windows):
    return array_fingerprint(
        {"declared": windows.excitation_declared},
        {key: getattr(windows, key) for key in ("past_excitation", "future_excitation")}
        if windows.excitation_declared
        else {},
    )


def equal_windows(actual, expected, label):
    _require(
        _window_metadata(actual) == _window_metadata(expected)
        and actual.batch.dt_s == expected.batch.dt_s
        and actual.excitation_declared == expected.excitation_declared,
        label + ": identities, timing or excitation declaration differ",
    )
    array_equal(window_arrays(actual), window_arrays(expected), label)


def _windows(metadata, arrays, role, dt_s):
    item = metadata["windows"][role]
    declared = item.get("excitation_declared", False)  # old public archive omits it
    _require(type(declared) is bool, "invalid excitation declaration")
    return SequenceWindows(
        SequenceBatch(**{key: arrays[f"{role}_{key}"] for key in ARRAYS}, dt_s=dt_s),
        tuple(WindowKey(**key) for key in item["keys"]),
        tuple(item["source_origins"]),
        **(
            {
                key: arrays[f"{role}_{key}"]
                for key in ("past_excitation", "future_excitation")
            }
            if declared
            else {}
        ),
    )


def _observe(state):
    # Keep the historical single-row expression and outer jit(vmap) unchanged.
    state = jnp.asarray(state)
    rotation = quaternion_to_rotation_batch(state[6:10][None, :])[0]
    return jnp.concatenate((state[3:6], state[10:13], rotation.reshape(9)))


_OBSERVE = jax.jit(jax.vmap(_observe))


def _collection(root, simulator, manifest):
    relative = f"{simulator}/excitation"
    directory = root / relative
    seal = _inventory(
        directory, "seal.json", manifest["files"][relative + "/seal.json"]
    )
    config = _json(directory / "configuration.json")
    spec = TrajectorySpec.from_dict(config["spec"])
    records = _json(directory / "records.json")
    roles = _json(directory / "roles.json")
    _require(roles == seal["roles"], "recording roles disagree with seal")
    ids = [row["id"] for row in records]
    _require(
        len(ids) == 96 and len(set(ids)) == 96 and set(ids) == set(seal["admitted"]),
        "calibration admission roster differs",
    )
    segments = []
    with jax.enable_x64(True):
        for row in records:
            with np.load(
                _under(directory, row["prefix"] + ".npz"), allow_pickle=False
            ) as archive:
                n = row["validity"]["valid_transitions"]
                observed = np.asarray(_OBSERVE(archive["states"][: n + 1]))
                dt = float(archive["time_s"][1] - archive["time_s"][0])
                segments.append(
                    SequenceSegment(
                        row["id"], "valid-prefix", observed, archive["commands"][:n], dt
                    )
                )
    return SequenceCollection(
        tuple(segments),
        configuration_id=spec.vehicle.configuration_id,
        state_channels=OBSERVED_CHANNELS,
        input_channels=tuple(
            json.dumps(c.to_dict(), sort_keys=True) for c in spec.controls
        ),
    ), roles


def training_norms(batch):
    """Reconstruct the eight train-derived moments without draws or solves."""
    context, delay = batch.past_inputs.shape[1], steps_for(batch.dt_s)["delay"]
    physical = np.concatenate((batch.past_states, batch.future_states), axis=1)
    commands = np.concatenate((batch.past_inputs, batch.future_inputs), axis=1)
    current = physical[:, context:-1]
    xm, xs = current.mean((0, 1)), current.std((0, 1))
    um, us = batch.future_inputs.mean((0, 1)), batch.future_inputs.std((0, 1))
    xs, us = np.where(xs > 1e-8, xs, 1), np.where(us > 1e-8, us, 1)
    x, u = (physical - xm) / xs, (commands - um) / us
    ds = np.maximum(((batch.future_states - current) / xs).std((0, 1)), 1e-4)
    hidden = np.zeros((len(current), RECIPE["memory"]))
    features = np.stack(
        [
            _features(
                x[:, context + t],
                u[:, context + t],
                x[:, context + t - delay : context + t],
                u[:, context + t - delay : context + t],
                hidden,
                xp=np,
            )
            for t in range(current.shape[1])
        ],
        axis=1,
    ).reshape(
        -1, (delay + 1) * (current.shape[-1] + commands.shape[-1]) + RECIPE["memory"]
    )
    fs = features.std(axis=0)
    norms = dict(
        state_mean=xm,
        state_scale=xs,
        input_mean=um,
        input_scale=us,
        feature_scale=np.where(fs > 1e-8, fs, 1.0),
        delta_scale=ds,
    )
    for name, values in (
        ("interaction", interaction_features(x[:, context:-1], u[:, context:], xp=np)),
        ("autonomous", autonomous_features(x[:, context:-1], xp=np)),
    ):
        scale = values.reshape(len(features), -1).std(axis=0)
        norms[name + "_scale"] = np.where(scale > 1e-8, scale, 1.0)
    return norms


def _model_arrays(arrays):
    return {
        key: value
        for key, value in arrays.items()
        if key.startswith(("param_", "norm_"))
    }


@dataclass(frozen=True)
class PreparedReference:
    collection: SequenceCollection
    train: SequenceWindows
    development: SequenceWindows
    contract: dict
    seen: dict
    roles: dict
    norms: dict
    error_scale: np.ndarray
    ridge: float
    delay: int
    source_report: dict
    initial_arrays: dict
    selected_arrays: dict
    objective_arrays: dict
    envelope: np.ndarray
    source: dict


def _prepare(root, simulator, manifest):
    _require(simulator in SIMULATORS, "unknown frozen simulator")
    directory = root / simulator
    meta, arrays = load_arrays(directory / "candidate/model.npz")
    baseline, base_arrays = load_arrays(directory / "baseline/model.npz")
    preparation, prep_arrays = load_arrays(directory / "candidate/preparation.npz")
    report = _json(directory / "candidate/report.json")
    _require(meta["report"] == report, "saved research report differs")
    _require(
        report["preparation"]["public_reference_fingerprint"]
        == array_fingerprint(baseline, base_arrays),
        "historical public reference differs",
    )
    for field, path in (
        ("common_data_seal_sha256", "data/seal.json"),
        ("excitation_seal_sha256", "excitation/seal.json"),
        ("excitation_reuse_sha256", "reuse.json"),
        ("reference_reuse_sha256", "reference-reuse.json"),
    ):
        _require(
            report["preparation"]["data_seal"][field]
            == manifest["files"][f"{simulator}/{path}"],
            "preparation data seal differs",
        )
    collection, roles = _collection(root, simulator, manifest)
    contract, seen = _contract(collection), _recording_content(collection)
    _require(
        contract == meta["contract"] == baseline["contract"] == preparation["contract"],
        "source signal contract differs",
    )
    _require(
        seen
        == baseline["seen"]
        == report["preparation"]["recording_content_fingerprints"],
        "source recording ledger differs",
    )
    names = sorted(seen, key=_priority)
    _require(
        len(names) == 96
        and roles == {"development": names[:24], "training": names[24:]},
        "automatic recording roles differ",
    )
    _require(roles == report["preparation"]["roles"], "preparation roles differ")
    train = _extract(collection, roles["training"], RECIPE["training_windows"])
    development = _extract(
        collection, roles["development"], RECIPE["development_windows"]
    )
    original = _extract(collection, roles["training"], 384)
    _require(
        len(train.keys) == 1536
        and len(development.keys) == 256
        and len(original.keys) == 384,
        "frozen cache budget differs",
    )
    for role, value in (("train", train), ("development", development)):
        equal_windows(
            value, _windows(meta, arrays, role, contract["dt_s"]), "candidate " + role
        )
        equal_windows(
            value,
            _windows(preparation, prep_arrays, role, contract["dt_s"]),
            "preparation " + role,
        )
    equal_windows(
        original,
        _windows(baseline, base_arrays, "train", contract["dt_s"]),
        "original384",
    )
    equal_windows(_subset(train, list(range(384))), original, "training384 prefix")
    equal_windows(
        development,
        _windows(baseline, base_arrays, "development", contract["dt_s"]),
        "original development",
    )
    norms = training_norms(train.batch)
    array_equal(
        {"norm_" + k: v for k, v in norms.items()},
        {k: v for k, v in prep_arrays.items() if k.startswith("norm_")},
        "training moments",
    )
    hold = np.repeat(
        train.batch.past_states[:, -1:], train.batch.future_states.shape[1], axis=1
    )
    error_scale = np.maximum(
        np.sqrt(np.mean((hold - train.batch.future_states) ** 2, axis=0)),
        RECIPE["hold_scale_floor"] * norms["state_scale"],
    )
    array_equal(
        {"error_scale": error_scale},
        {"error_scale": prep_arrays["error_scale"]},
        "hold scale",
    )
    ridge = (
        RECIPE["ridge_fraction"] * len(train.keys) * train.batch.future_states.shape[1]
    )
    delay = steps_for(contract["dt_s"])["delay"]
    _require(
        meta["model"]["dt_s"] == contract["dt_s"]
        and meta["model"]["history_steps"] == train.batch.past_inputs.shape[1]
        and meta["model"]["delay_steps"] == delay,
        "research model timing differs",
    )
    _require(
        preparation["ridge"] == report["preparation"]["ridge"] == ridge
        and preparation["delay"] == report["preparation"]["delay"] == delay,
        "ridge or delay differs",
    )
    _require(
        preparation["provenance"] == report["preparation"],
        "preparation provenance differs",
    )
    for key, value in (
        ("training_windows_fingerprint", window_fingerprint(train)),
        ("development_windows_fingerprint", window_fingerprint(development)),
        ("reference_training_windows_fingerprint", window_fingerprint(original)),
        ("training_excitation_fingerprint", excitation_fingerprint(train)),
        ("development_excitation_fingerprint", excitation_fingerprint(development)),
        ("training_norms_fingerprint", array_fingerprint({}, norms)),
        (
            "error_scale_fingerprint",
            array_fingerprint({}, {"error_scale": error_scale}),
        ),
    ):
        _require(
            report["preparation"][key] == value,
            "preparation fingerprint differs: " + key,
        )
    checkpoints = directory / "candidate/checkpoints"
    trajectory = checkpoints / "weighted/anchored/trajectory"
    _, initial = load_arrays(trajectory / "step-0000.npz")
    selected_step = report["optimization"]["selected_step"]
    _require(selected_step == 1000, "historical selected checkpoint differs")
    _, selected = load_arrays(trajectory / f"step-{selected_step:04d}.npz")
    _, returned = load_arrays(checkpoints / "initializer-return.npz")
    _, objective = load_arrays(checkpoints / "weighted/objective-witness.npz")
    array_equal(_model_arrays(initial), returned, "actual initializer return")
    array_equal(
        _model_arrays(initial),
        _model_arrays(objective),
        "weight-boundary initialization",
    )
    array_equal(
        _model_arrays(selected), _model_arrays(arrays), "selected research mean"
    )
    array_equal(
        {"norm_" + k: v for k, v in norms.items()},
        {k: v for k, v in initial.items() if k.startswith("norm_")},
        "initial norms",
    )
    for label, scale in (
        ("initial", initial["error_scale"]),
        ("selected", selected["error_scale"]),
        ("objective", objective["normalization"]),
    ):
        array_equal(
            {"error_scale": scale}, {"error_scale": error_scale}, label + " hold scale"
        )
    e0 = np.mean(
        (
            (objective["initial_training_prediction"] - train.batch.future_states)
            / error_scale
        )
        ** 2,
        axis=(0, 1),
    )
    floor = np.asarray(RECIPE["initial_channel_mse_floor"], dtype=np.float64)
    raw = 1 / np.maximum(e0, floor)
    normalizer = np.asarray(raw.mean())
    weights = raw / normalizer
    array_equal(
        {
            "initial_channel_mse": e0,
            "raw_channel_weights": raw,
            "channel_weights": weights,
            "fixed_weights": weights,
            "weight_floor": floor,
            "weight_normalizer": normalizer,
        },
        {
            key: objective[key]
            for key in (
                "initial_channel_mse",
                "raw_channel_weights",
                "channel_weights",
                "fixed_weights",
                "weight_floor",
                "weight_normalizer",
            )
        },
        "saved objective reductions",
    )
    array_equal(
        {"envelope": arrays["envelope_half_width"]},
        {"envelope": np.asarray(report["envelope"]["half_width"], dtype=np.float64)},
        "original calibration envelope",
    )
    source_paths = [
        "candidate/model.npz",
        "baseline/model.npz",
        "candidate/preparation.npz",
        "candidate/report.json",
        "excitation/seal.json",
        "excitation/roles.json",
        "candidate/checkpoints/initializer-return.npz",
        "candidate/checkpoints/weighted/objective-witness.npz",
        "candidate/checkpoints/weighted/anchored/trajectory/step-0000.npz",
        f"candidate/checkpoints/weighted/anchored/trajectory/step-{selected_step:04d}.npz",
    ]
    source = {
        "bundle_sha256": BUNDLE_SHA256,
        "protocol_sha256": PROTOCOL_SHA256,
        "simulator": simulator,
        "selected_step": selected_step,
        "files": {
            f"{simulator}/{name}": manifest["files"][f"{simulator}/{name}"]
            for name in source_paths
        },
    }
    for values in (norms, initial, selected, objective):
        for value in values.values():
            _require(
                value.dtype == np.dtype("float64") and np.isfinite(value).all(),
                "research numerical witness must be finite float64",
            )
            value.setflags(write=False)
    error_scale.setflags(write=False)
    arrays["envelope_half_width"].setflags(write=False)
    return PreparedReference(
        collection,
        train,
        development,
        contract,
        seen,
        roles,
        norms,
        error_scale,
        ridge,
        delay,
        report,
        _model_arrays(initial),
        _model_arrays(selected),
        objective,
        arrays["envelope_half_width"],
        source,
    )


def prepare_reference(reference_root, simulator, *, expected_bundle_sha256):
    """Return exact prepared inputs and old actual-fit witnesses; no fitted calls."""
    manifest = authenticate_reference(
        reference_root, expected_bundle_sha256=expected_bundle_sha256
    )
    return _prepare(Path(reference_root), simulator, manifest)


def public_fixture(prepared):
    """Wrap immutable saved research arrays, explicitly declaring port provenance."""
    model = SequenceModel(
        kind=RECIPE["kind"],
        dt_s=prepared.contract["dt_s"],
        history_steps=prepared.train.batch.past_inputs.shape[1],
        delay_steps=prepared.delay,
        params={
            k[6:]: v
            for k, v in prepared.selected_arrays.items()
            if k.startswith("param_")
        },
        norms={
            k[5:]: v
            for k, v in prepared.selected_arrays.items()
            if k.startswith("norm_")
        },
    )
    original = prepared.source_report
    report = {
        "recipe": copy.deepcopy(RECIPE),
        "previous_revision": None,
        "history_steps": model.history_steps,
        "horizon_steps": prepared.train.batch.future_states.shape[1],
        "delay_steps": prepared.delay,
        "training": prepared.train.coverage(),
        "development": prepared.development.coverage(),
        "envelope": copy.deepcopy(original["envelope"]),
        "development_errors": copy.deepcopy(original["development_errors"]),
        "evidence_limits": copy.deepcopy(original["evidence_limits"])
        + [
            "qualification fixture ported from saved research arrays; no public fit or recalibration was performed",
        ],
        "saved_research_port": {
            "source": copy.deepcopy(prepared.source),
            "original_recipe": copy.deepcopy(original["recipe"]),
            "calibration": "unchanged original development envelope",
            "fitting_operations": 0,
            "initialization_calls": 0,
            "prediction_calls": 0,
        },
    }
    return LearnedDynamics(
        model,
        prepared.train,
        prepared.development,
        prepared.contract,
        prepared.seen,
        report,
        prepared.envelope,
    )


def _evidence(prepared, fixture):
    return {
        "source": prepared.source,
        "model_fingerprint": fixture.fingerprint(),
        "training_windows": len(prepared.train.keys),
        "development_windows": len(prepared.development.keys),
        "training_recordings": len(prepared.roles["training"]),
        "development_recordings": len(prepared.roles["development"]),
        "training_window_fingerprint": window_fingerprint(prepared.train),
        "development_window_fingerprint": window_fingerprint(prepared.development),
        "training_excitation_fingerprint": excitation_fingerprint(prepared.train),
        "development_excitation_fingerprint": excitation_fingerprint(
            prepared.development
        ),
        "recording_ledger": prepared.seen,
        "contract": prepared.contract,
        "ridge": prepared.ridge,
        "delay": prepared.delay,
        "initial_arrays_fingerprint": array_fingerprint({}, prepared.initial_arrays),
        "selected_arrays_fingerprint": array_fingerprint({}, prepared.selected_arrays),
        "objective_arrays_fingerprint": array_fingerprint(
            {}, prepared.objective_arrays
        ),
        "exact_preparation": True,
        "original_384_prefix_exact": True,
        "original_calibration_preserved": True,
        "fitting_operations": 0,
        "initialization_calls": 0,
        "prediction_calls": 0,
    }


def _attempt(implementation_sha256):
    return {
        "format": FORMAT + "-attempt",
        "protocol_sha256": PROTOCOL_SHA256,
        "implementation_sha256": implementation_sha256,
        "reference_bundle_sha256": BUNDLE_SHA256,
        "simulators": list(SIMULATORS),
        "untrusted_reference_reports_not_used": list(UNTRUSTED_REFERENCE_REPORTS),
        "fitting_operations": 0,
        "initialization_calls": 0,
        "prediction_calls": 0,
    }


def _payloads(output):
    return {
        str(p.relative_to(output)): digest(p)
        for p in sorted(output.rglob("*"))
        if p.is_file()
    }


def run(output, *, reference_root, implementation_manifest, implementation_sha256):
    """Create an exclusive attempt; retain and seal failures without retrying."""
    from .public_mean_implementation import verify as verify_implementation

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    intent = _attempt(implementation_sha256)
    _write(output / "attempt.json", intent)
    results, completed = {}, []
    active = None
    try:
        binding = verify_implementation(implementation_manifest, implementation_sha256)
        _require(
            Path(__file__).resolve()
            == Path(binding["public_root"])
            / "src/glassbox/experimental/public_mean_saved_port.py",
            "saved-port module imported from another checkout",
        )
        manifest = authenticate_reference(
            reference_root, expected_bundle_sha256=BUNDLE_SHA256
        )
        for simulator in SIMULATORS:
            active = simulator
            prepared = _prepare(Path(reference_root), simulator, manifest)
            directory = output / simulator
            directory.mkdir()
            fixture = public_fixture(prepared)
            fixture.save(directory / "model.npz")
            loaded = LearnedDynamics.load(directory / "model.npz")
            _require(
                loaded.fingerprint() == fixture.fingerprint(),
                "public fixture persistence changed",
            )
            save_recordings(prepared.collection, directory / "recordings.npz")
            shutil.copyfile(
                Path(reference_root) / simulator / "candidate/report.json",
                directory / "source-report.json",
            )
            _write(directory / "source.json", prepared.source)
            results[simulator] = _evidence(prepared, fixture)
            _write(directory / "preparation-parity.json", results[simulator])
            completed.append(simulator)
        active = None
        _write(
            output / "outcome.json",
            {"status": "complete", "completed_simulators": completed},
        )
        result = {
            "format": FORMAT,
            "protocol_sha256": PROTOCOL_SHA256,
            "implementation_sha256": implementation_sha256,
            "reference_bundle_sha256": BUNDLE_SHA256,
            "results": results,
            "fitting_operations": 0,
            "initialization_calls": 0,
            "prediction_calls": 0,
            "files": _payloads(output),
        }
        _write(output / "manifest.json", result)
    except BaseException as error:
        failure = {
            "status": "failed",
            "completed_simulators": completed,
            "active_simulator": active,
            "exception_type": type(error).__name__,
            "reason": str(error),
            "fitting_operations": 0,
            "initialization_calls": 0,
            "prediction_calls": 0,
        }
        if not (output / "outcome.json").exists():
            _write(output / "outcome.json", failure)
        _write(
            output / "failure.json",
            {
                "format": FORMAT + "-failure",
                "attempt": intent,
                "outcome": failure,
                "files": _payloads(output),
            },
        )
        raise
    return {
        "manifest": str(output / "manifest.json"),
        "manifest_sha256": digest(output / "manifest.json"),
        "simulators": len(results),
        "fitting_operations": 0,
        "prediction_calls": 0,
    }


def verify(
    output,
    *,
    expected_manifest_sha256,
    reference_root,
    implementation_manifest,
    implementation_sha256,
):
    """Freshly reconstruct preparation/port identities; never replay an optimizer."""
    from .public_mean_implementation import verify as verify_implementation

    binding = verify_implementation(implementation_manifest, implementation_sha256)
    _require(
        Path(__file__).resolve()
        == Path(binding["public_root"])
        / "src/glassbox/experimental/public_mean_saved_port.py",
        "saved-port module imported from another checkout",
    )
    output = Path(output)
    saved = _inventory(output, "manifest.json", expected_manifest_sha256)
    _require(
        saved["format"] == FORMAT
        and saved["protocol_sha256"] == PROTOCOL_SHA256
        and saved["implementation_sha256"] == implementation_sha256
        and saved["reference_bundle_sha256"] == BUNDLE_SHA256,
        "saved port provenance differs",
    )
    _require(set(saved["results"]) == set(SIMULATORS), "saved simulator roster differs")
    _require(
        _json(output / "attempt.json") == _attempt(implementation_sha256),
        "saved attempt intent differs",
    )
    _require(
        _json(output / "outcome.json")
        == {"status": "complete", "completed_simulators": list(SIMULATORS)},
        "saved attempt did not complete",
    )
    expected_files = {"attempt.json", "outcome.json"} | {
        f"{s}/{name}"
        for s in SIMULATORS
        for name in (
            "model.npz",
            "recordings.npz",
            "source-report.json",
            "source.json",
            "preparation-parity.json",
        )
    }
    _require(set(saved["files"]) == expected_files, "saved port file roster differs")
    manifest = authenticate_reference(
        reference_root, expected_bundle_sha256=BUNDLE_SHA256
    )
    for simulator in SIMULATORS:
        prepared = _prepare(Path(reference_root), simulator, manifest)
        expected = public_fixture(prepared)
        directory = output / simulator
        actual = LearnedDynamics.load(directory / "model.npz")
        _require(
            actual.fingerprint() == expected.fingerprint(),
            "ported mean or provenance changed",
        )
        equal_windows(actual._train, prepared.train, "saved port training")
        equal_windows(
            actual._development, prepared.development, "saved port development"
        )
        recordings = load_recordings(directory / "recordings.npz")
        _require(
            _contract(recordings) == prepared.contract
            and _recording_content(recordings) == prepared.seen,
            "saved recording ledger or contract differs",
        )
        _require(
            len(recordings.segments) == len(prepared.collection.segments),
            "saved recording segments differ",
        )
        for a, b in zip(recordings.segments, prepared.collection.segments, strict=True):
            _require(
                (
                    a.recording_id,
                    a.segment_id,
                    a.start_row,
                    a.dt_s,
                    a.excitation is None,
                )
                == (
                    b.recording_id,
                    b.segment_id,
                    b.start_row,
                    b.dt_s,
                    b.excitation is None,
                ),
                "saved recording provenance or sidecars differ",
            )
            array_equal(
                {"states": a.states, "inputs": a.inputs},
                {"states": b.states, "inputs": b.inputs},
                "saved recording bytes",
            )
        _require(
            (directory / "source-report.json").read_bytes()
            == (
                Path(reference_root) / simulator / "candidate/report.json"
            ).read_bytes(),
            "original research report changed",
        )
        _require(
            _json(directory / "source.json") == prepared.source,
            "source anchors changed",
        )
        _require(
            saved["results"][simulator]
            == _json(directory / "preparation-parity.json")
            == _evidence(prepared, expected),
            "preparation evidence does not replay",
        )
    _require(
        all(
            saved[key] == 0
            for key in (
                "fitting_operations",
                "initialization_calls",
                "prediction_calls",
            )
        ),
        "port operation count differs",
    )
    return {
        "passed": True,
        "manifest_sha256": expected_manifest_sha256,
        "simulators": len(SIMULATORS),
        "fitting_operations": 0,
        "initialization_calls": 0,
        "prediction_calls": 0,
    }


replay = verify
