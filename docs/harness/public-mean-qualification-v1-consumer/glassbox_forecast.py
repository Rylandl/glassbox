"""Inspect public Glassbox forecasts and command sensitivities from sealed inputs.

This Dart-owned consumer fits nothing and reads no model implementation details.
The qualification launcher independently authenticates this executable and its
observed public import map before launching the fixed command below.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import zipfile
from collections import Counter
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from glassbox import LearnedDynamics
from glassbox.io.recordings import load_recordings

FORMAT = "glassbox-public-consumer-input-v1"
RESULT_FORMAT = "crazydart-glassbox-forecast-v1"
PROTOCOL_SHA256 = "d3a8d0eeeeab2aac923cad7f1f5fe1c8a49fc0ffa765e84b3d7c6bc80cec8f99"
SELECTION_SHA256 = "366fd718b490dc2bc44e4a8724f74c3d3b2eda89495fa8eb4a3788f7f2980d08"
EXPORTER_MODULE = "glassbox.experimental.public_mean_consumer_export"
MODEL_IDS = ("crazyflow", "cascade")
SCOPES = frozenset(
    ("primary", "heading_shift", "maneuver_shift", "speed_shift", "wind_shift")
)
PUBLIC_MODULES = frozenset(
    (
        "glassbox",
        "glassbox.learner",
        "glassbox._sequence_model",
        "glassbox._learner_arrays",
        "glassbox.recordings",
        "glassbox.io",
        "glassbox.io.recordings",
    )
)


class PacketError(ValueError):
    """The sealed input packet or public consumer contract is invalid."""


def _require(condition, message):
    if not condition:
        raise PacketError(message)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def _invalid_constant(value):
    raise PacketError("nonfinite JSON constant: " + value)


def _keys(value, keys):
    _require(type(value) is dict and set(value) == set(keys), "unexpected object keys")


def _text(value):
    _require(type(value) is str and bool(value.strip()), "expected nonempty string")
    return value


def _integer(value, minimum=0):
    _require(type(value) is int and value >= minimum, "invalid integer")
    return value


def _digest(value, length=64):
    _require(
        type(value) is str and re.fullmatch(r"[0-9a-f]{" + str(length) + "}", value),
        "invalid lowercase hexadecimal digest",
    )
    return value


def _interval(value):
    _require(
        type(value) in (int, float) and math.isfinite(value) and value > 0,
        "invalid sample interval",
    )


def _contract(value):
    _keys(value, ("configuration_id", "state_channels", "input_channels", "dt_s"))
    _text(value["configuration_id"])
    _interval(value["dt_s"])
    for name in ("state_channels", "input_channels"):
        channels = value[name]
        _require(type(channels) is list and bool(channels), "empty channel list")
        for channel in channels:
            _text(channel)
        _require(len(set(channels)) == len(channels), "duplicate channels")


def _payload_path(directory, relative):
    _text(relative)
    parts = relative.split("/")
    _require(
        not relative.startswith("/")
        and "\\" not in relative
        and all(part not in ("", ".", "..") for part in parts),
        "payload path is not a normalized relative POSIX path",
    )
    value = directory
    for part in parts:
        value = value / part
        _require(not value.is_symlink(), "payload path contains a symlink")
    _require(value.suffix == ".npz" and value.is_file(), "payload is not a regular NPZ")
    _require(value.resolve().is_relative_to(directory), "payload escapes packet")
    return value


def _packet_files(directory):
    files = set()
    for path in directory.rglob("*"):
        _require(not path.is_symlink(), "packet contains a symlink")
        if path.is_file():
            files.add(path.relative_to(directory).as_posix())
        else:
            _require(path.is_dir(), "packet contains a special file")
    return files


def _zip_members(path):
    with zipfile.ZipFile(path) as archive:
        members = archive.namelist()
        _require(len(set(members)) == len(members), "duplicate ZIP members")
        _require(
            all(name.endswith(".npy") and "/" not in name for name in members),
            "invalid NPZ member",
        )
    return members


def _exact_array(left, right, label):
    _require(
        left.shape == right.shape
        and left.dtype.str == right.dtype.str
        and left.tobytes(order="C") == right.tobytes(order="C"),
        label + " differs from recording bytes",
    )


def _query_arrays(path, query, descriptor):
    names = {"past_states", "past_inputs", "future_inputs"}
    if query["kind"] == "response":
        names.add("factual_inputs")
    _require(
        set(_zip_members(path)) == {name + ".npy" for name in names},
        "query array roster differs",
    )
    c, h = descriptor["history_steps"], descriptor["horizon_steps"]
    d, m = (
        len(descriptor["contract"][name])
        for name in ("state_channels", "input_channels")
    )
    shapes = dict(
        past_states=(c + 1, d),
        past_inputs=(c, m),
        future_inputs=(h, m),
        factual_inputs=(h, m),
    )
    result = {}
    with np.load(path, allow_pickle=False) as archive:
        _require(
            len(archive.files) == len(names) and set(archive.files) == names,
            "query arrays differ",
        )
        for name in names:
            value = archive[name]
            _require(
                value.dtype.str == "<f8"
                and value.shape == shapes[name]
                and value.flags.c_contiguous
                and np.isfinite(value).all(),
                "query arrays must have exact shapes and finite C-contiguous little-endian float64 values",
            )
            result[name] = np.array(value, copy=True)
            result[name].setflags(write=False)
    return result


def _history(query, arrays, collection, history_steps):
    origin = query["source_origin"]
    segments = [s for s in collection.segments if s.recording_id == query["parent"]]
    matches = [
        s
        for s in segments
        if s.start_row <= origin - history_steps
        and origin < s.start_row + len(s.states)
    ]
    _require(
        len(matches) == 1 and matches[0].segment_id == query["segment_id"],
        "history crosses or misidentifies a segment boundary",
    )
    segment = matches[0]
    local = origin - segment.start_row
    _exact_array(
        segment.states[local - history_steps : local + 1],
        arrays["past_states"],
        "past states",
    )
    _exact_array(
        segment.inputs[local - history_steps : local],
        arrays["past_inputs"],
        "past commands",
    )
    factual = (
        arrays["factual_inputs"]
        if query["kind"] == "response"
        else arrays["future_inputs"]
    )
    checked = 0
    for segment in segments:
        start = max(origin, segment.start_row)
        stop = min(origin + len(factual), segment.start_row + len(segment.inputs))
        if start < stop:
            _exact_array(
                segment.inputs[start - segment.start_row : stop - segment.start_row],
                factual[start - origin : stop - origin],
                "available factual future commands",
            )
            checked += stop - start
    return checked


@dataclass(frozen=True)
class ValidatedPacket:
    manifest_path: Path
    manifest_sha256: str
    manifest: dict
    models: dict
    queries: tuple
    payloads: dict


def validate_packet(manifest, expected_manifest_sha256):
    """Validate the entire packet and histories before any model prediction."""
    _digest(expected_manifest_sha256)
    path = Path(manifest).absolute()
    _require(
        path.name == "INPUT.json" and path.is_file() and not path.is_symlink(),
        "manifest must be regular INPUT.json",
    )
    _require(_sha(path) == expected_manifest_sha256, "external manifest SHA mismatch")
    directory = path.parent.resolve()
    data = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_pairs,
        parse_constant=_invalid_constant,
    )
    _keys(data, ("format", "qualification", "models", "recordings", "queries"))
    _require(data["format"] == FORMAT, "unsupported packet format")
    q = data["qualification"]
    _keys(
        q,
        (
            "protocol_sha256",
            "implementation_commit",
            "exporter_module",
            "exporter_source_sha256",
            "query_selection_spec_sha256",
            "test_seed_shift_from_original",
        ),
    )
    for key in (
        "protocol_sha256",
        "exporter_source_sha256",
        "query_selection_spec_sha256",
    ):
        _digest(q[key])
    _digest(q["implementation_commit"], 40)
    _integer(q["test_seed_shift_from_original"])
    _require(
        q["protocol_sha256"] == PROTOCOL_SHA256
        and q["query_selection_spec_sha256"] == SELECTION_SHA256
        and q["exporter_module"] == EXPORTER_MODULE
        and q["test_seed_shift_from_original"] == 11000000,
        "qualification identity differs",
    )
    for key, count in (("models", 2), ("recordings", 2), ("queries", 16)):
        _require(
            type(data[key]) is list and len(data[key]) == count,
            "packet roster count differs",
        )
    models, recording_rows, paths = {}, {}, {}
    for row, identity in zip(data["models"], MODEL_IDS, strict=True):
        _keys(
            row,
            (
                "id",
                "path",
                "sha256",
                "fingerprint",
                "contract",
                "history_steps",
                "horizon_steps",
            ),
        )
        _require(row["id"] == identity, "model routing order differs")
        _digest(row["fingerprint"])
        _integer(row["history_steps"], 1)
        _integer(row["horizon_steps"], 1)
        _contract(row["contract"])
        models[identity] = row
    for row, identity in zip(data["recordings"], MODEL_IDS, strict=True):
        _keys(row, ("id", "model_id", "path", "sha256", "parents"))
        _require(
            row["id"] == identity + "-recordings" and row["model_id"] == identity,
            "recording routing order differs",
        )
        _require(type(row["parents"]) is list and bool(row["parents"]), "empty parents")
        parent_ids = []
        for parent in row["parents"]:
            _keys(
                parent,
                (
                    "recording_id",
                    "source_parent_sha256",
                    "source_record_entry_sha256",
                    "segments",
                ),
            )
            parent_ids.append(_text(parent["recording_id"]))
            _digest(parent["source_parent_sha256"])
            _digest(parent["source_record_entry_sha256"])
            _require(
                type(parent["segments"]) is list and bool(parent["segments"]),
                "empty segments",
            )
            order = []
            for segment in parent["segments"]:
                _keys(segment, ("segment_id", "start_row", "state_rows", "input_rows"))
                _text(segment["segment_id"])
                _integer(segment["start_row"])
                _integer(segment["state_rows"], 2)
                _integer(segment["input_rows"], 1)
                _require(
                    segment["input_rows"] == segment["state_rows"] - 1,
                    "segment row counts differ",
                )
                order.append((segment["start_row"], segment["segment_id"]))
            _require(
                order == sorted(set(order))
                and len({x[1] for x in order}) == len(order),
                "segment order or identity differs",
            )
        _require(
            parent_ids == sorted(set(parent_ids)), "parent order or identity differs"
        )
        recording_rows[identity] = row
    identities, counts = [], Counter()
    for row in data["queries"]:
        _keys(
            row,
            (
                "model_id",
                "recordings_id",
                "parent",
                "id",
                "segment_id",
                "source_origin",
                "kind",
                "scope",
                "dt_s",
                "horizon_steps",
                "branches",
                "path",
                "sha256",
                "source_query_sha256",
            ),
        )
        for key in ("model_id", "recordings_id", "parent", "id", "segment_id"):
            _text(row[key])
        _require(row["model_id"] in models, "unknown model routing ID")
        descriptor = models[row["model_id"]]
        _require(
            row["recordings_id"] == recording_rows[row["model_id"]]["id"],
            "query recordings differ",
        )
        _require(
            type(row["scope"]) is str
            and row["scope"] in SCOPES
            and row["kind"] in ("factual", "response"),
            "unknown query scope or kind",
        )
        _require(
            row["branches"]
            == (["factual"] if row["kind"] == "factual" else ["intervened", "factual"]),
            "query branch roster differs",
        )
        _integer(row["source_origin"])
        _integer(row["horizon_steps"], 1)
        _interval(row["dt_s"])
        _digest(row["source_query_sha256"])
        _require(
            row["dt_s"] == descriptor["contract"]["dt_s"]
            and row["horizon_steps"] == descriptor["horizon_steps"],
            "query timing differs",
        )
        identities.append((row["model_id"], row["parent"], row["id"]))
        counts[
            row["model_id"],
            "primary" if row["scope"] == "primary" else "shifted",
            row["kind"],
        ] += 1
    _require(identities == sorted(set(identities)), "query identities or order differ")
    _require(
        counts
        == Counter(
            {
                (model, scope, kind): 2
                for model in MODEL_IDS
                for scope in ("primary", "shifted")
                for kind in ("factual", "response")
            }
        ),
        "query strata differ",
    )
    for identity, row in recording_rows.items():
        parents = sorted(
            {q["parent"] for q in data["queries"] if q["model_id"] == identity}
        )
        _require(
            [p["recording_id"] for p in row["parents"]] == parents,
            "recording parent roster differs from queries",
        )
    for row in (*data["models"], *data["recordings"], *data["queries"]):
        _digest(row["sha256"])
        payload = _payload_path(directory, row["path"])
        _require(row["path"] not in paths, "duplicate payload path")
        _require(_sha(payload) == row["sha256"], "payload SHA mismatch")
        _zip_members(payload)
        paths[row["path"]] = row["sha256"]
    _require(
        _packet_files(directory) == {"INPUT.json", *paths},
        "undeclared or missing packet files",
    )
    loaded = {}
    for identity, row in models.items():
        model = LearnedDynamics.load(directory / row["path"])
        _require(
            model.fingerprint() == row["fingerprint"]
            and model.contract == row["contract"]
            and model.history_steps == row["history_steps"]
            and model.horizon_steps == row["horizon_steps"],
            "loaded public model identity/contract/timing differs",
        )
        _require(
            model.recipe.get("id") == "generic-memory-v4-prototype",
            "loaded public recipe differs",
        )
        loaded[identity] = model
    collections = {}
    for identity, row in recording_rows.items():
        collection = load_recordings(directory / row["path"])
        contract = models[identity]["contract"]
        _require(
            collection.configuration_id == contract["configuration_id"]
            and list(collection.state_channels) == contract["state_channels"]
            and list(collection.input_channels) == contract["input_channels"]
            and all(s.dt_s == contract["dt_s"] for s in collection.segments),
            "recording semantic contract differs",
        )
        actual = {}
        for segment in collection.segments:
            actual.setdefault(segment.recording_id, []).append(
                dict(
                    segment_id=segment.segment_id,
                    start_row=int(segment.start_row),
                    state_rows=len(segment.states),
                    input_rows=len(segment.inputs),
                )
            )
        actual = {
            key: sorted(value, key=lambda s: (s["start_row"], s["segment_id"]))
            for key, value in actual.items()
        }
        _require(
            actual == {p["recording_id"]: p["segments"] for p in row["parents"]},
            "loaded recording segment roster differs",
        )
        collections[identity] = collection
    queries = []
    for row in data["queries"]:
        arrays = _query_arrays(directory / row["path"], row, models[row["model_id"]])
        checked = _history(
            row,
            arrays,
            collections[row["model_id"]],
            models[row["model_id"]]["history_steps"],
        )
        queries.append((row, arrays, checked))
    packet = ValidatedPacket(
        directory / "INPUT.json",
        expected_manifest_sha256,
        data,
        loaded,
        tuple(queries),
        paths,
    )
    _unchanged(packet)
    return packet


def _unchanged(packet):
    _require(
        _sha(packet.manifest_path) == packet.manifest_sha256,
        "manifest changed during consumption",
    )
    root = packet.manifest_path.parent
    _require(
        _packet_files(root) == {"INPUT.json", *packet.payloads}, "packet roster changed"
    )
    for path, expected in packet.payloads.items():
        _require(
            _sha(_payload_path(root, path)) == expected,
            "payload changed during consumption",
        )
    for row in packet.manifest["models"]:
        _require(
            packet.models[row["id"]].fingerprint() == row["fingerprint"],
            "public model changed during consumption",
        )


def observed_runtime():
    """Record source identity; the launcher checks its independent expected map."""
    _require(
        not jax.config.x64_enabled,
        "consumer must begin and remain in ambient default32",
    )
    own = Path(__file__).resolve()
    package = Path(sys.modules["crazydart"].__file__).resolve()
    _require(
        own == package.with_name("glassbox_forecast.py"),
        "consumer did not execute from its Dart package",
    )
    dart_root = package.parents[2]
    modules = {}
    for name, module in tuple(sys.modules.items()):
        if name == "glassbox" or name.startswith("glassbox."):
            _require(
                name in PUBLIC_MODULES, "forbidden Glassbox module imported: " + name
            )
            path = Path(module.__file__).resolve()
            _require(
                not path.is_relative_to(dart_root / "glassbox"),
                "Dart nested Glassbox dependency is not the qualified public implementation",
            )
            modules[name] = dict(path=str(path), sha256=_sha(path))
    for name, path in (("crazydart", package), ("crazydart.glassbox_forecast", own)):
        modules[name] = dict(path=str(path), sha256=_sha(path))
    return dict(
        interpreter=str(Path(sys.executable).resolve()),
        interpreter_sha256=_sha(Path(sys.executable).resolve()),
        python=sys.version,
        versions={
            name: metadata.version(name)
            for name in ("numpy", "jax", "jaxlib", "glassbox", "crazydart")
        },
        imports=dict(sorted(modules.items())),
        jax_enable_x64=bool(jax.config.x64_enabled),
        environment={
            name: os.environ.get(name) for name in ("JAX_ENABLE_X64", "SCIPY_ARRAY_API")
        },
    )


def _array_sha(value):
    digest = hashlib.sha256()
    digest.update(
        json.dumps([value.dtype.str, list(value.shape)], separators=(",", ":")).encode()
    )
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _future_sensitivity(model, past_states, past_inputs):
    @jax.jit
    def sensitivity(command, direction):
        return jax.jvp(
            lambda future: model.predict(past_states, past_inputs, future),
            (command,),
            (direction,),
        )[1]

    return sensitivity


def run(manifest, expected_manifest_sha256, output):
    """Produce the fixed public forecasts/JVPs after complete input validation."""
    runtime = observed_runtime()
    packet = validate_packet(manifest, expected_manifest_sha256)
    destination = Path(output).absolute()
    _require(
        not destination.exists() and not destination.is_symlink(),
        "output must be a new directory",
    )
    resolved_output = destination.resolve()
    _require(
        not resolved_output.is_relative_to(packet.manifest_path.parent)
        and not packet.manifest_path.parent.is_relative_to(resolved_output),
        "output must be separate from the input packet",
    )
    _require(
        observed_runtime() == runtime, "runtime/import state changed during validation"
    )
    arrays, means, responses = {}, [], []
    for query, data, checked in packet.queries:
        model = packet.models[query["model_id"]]
        contract = model.contract
        x, up = data["past_states"], data["past_inputs"]
        compiled = jax.jit(model.predict)
        sensitivity = _future_sensitivity(model, x, up)
        branch_results = {}
        for branch in query["branches"]:
            tape = (
                data["factual_inputs"]
                if query["kind"] == "response" and branch == "factual"
                else data["future_inputs"]
            )
            tangent = np.zeros_like(tape)
            tangent[0, 0] = 1.0
            eager = np.asarray(model.predict(x, up, tape))
            predicted = np.asarray(compiled(x, up, tape))
            derivative = np.asarray(
                sensitivity(jnp.asarray(tape), jnp.asarray(tangent))
            )
            envelope = np.asarray(model.envelope(query["horizon_steps"]))
            shape = (query["horizon_steps"], len(contract["state_channels"]))
            _require(
                all(
                    value.shape == shape and np.isfinite(value).all()
                    for value in (eager, predicted, derivative, envelope)
                ),
                "nonfinite or malformed public output",
            )
            _require(
                all(
                    value.dtype == np.dtype("float32")
                    for value in (eager, predicted, derivative)
                ),
                "public inference did not follow default32",
            )
            _require(np.all(envelope >= 0), "negative public envelope")
            index = len(means)
            prefix = f"mean_{index:03d}"
            values = dict(
                eager=eager,
                compiled=predicted,
                jvp=derivative,
                envelope=envelope,
                tangent=tangent,
                offsets_s=np.arange(1, query["horizon_steps"] + 1) * query["dt_s"],
            )
            arrays.update({prefix + "_" + key: value for key, value in values.items()})
            row = dict(
                index=index,
                model_id=query["model_id"],
                parent=query["parent"],
                query_id=query["id"],
                kind=query["kind"],
                scope=query["scope"],
                branch=branch,
                segment_id=query["segment_id"],
                source_origin=query["source_origin"],
                dt_s=query["dt_s"],
                query_payload_sha256=query["sha256"],
                source_query_sha256=query["source_query_sha256"],
                future_tape_sha256=_array_sha(tape),
                verified_factual_command_rows=checked,
                arrays={key: prefix + "_" + key for key in values},
                shapes={key: list(value.shape) for key, value in values.items()},
                dtypes={key: value.dtype.str for key, value in values.items()},
                sensitivity_command_channel=contract["input_channels"][0],
                sensitivity_direction="one native command unit at future row 0, channel 0",
                final_channels=[
                    dict(
                        channel=name,
                        forecast=float(eager[-1, i]),
                        envelope_half_width=float(envelope[-1, i]),
                        first_command_sensitivity=float(derivative[-1, i]),
                        sensitivity_per_command_channel=contract["input_channels"][0],
                    )
                    for i, name in enumerate(contract["state_channels"])
                ],
            )
            means.append(row)
            branch_results[branch] = (row, eager, predicted)
        if query["kind"] == "response":
            a, b = branch_results["intervened"], branch_results["factual"]
            index = len(responses)
            prefix = f"response_{index:03d}"
            values = dict(
                eager=a[1] - b[1],
                compiled=a[2] - b[2],
                command_delta=data["future_inputs"] - data["factual_inputs"],
            )
            _require(
                all(np.isfinite(value).all() for value in values.values()),
                "nonfinite response subtraction",
            )
            arrays.update({prefix + "_" + key: value for key, value in values.items()})
            responses.append(
                dict(
                    index=index,
                    model_id=query["model_id"],
                    parent=query["parent"],
                    query_id=query["id"],
                    intervened_mean=a[0]["index"],
                    factual_mean=b[0]["index"],
                    branch_tape_sha256={
                        "intervened": a[0]["future_tape_sha256"],
                        "factual": b[0]["future_tape_sha256"],
                    },
                    arrays={key: prefix + "_" + key for key in values},
                )
            )
    _require(len(means) == 24 and len(responses) == 8, "consumer output roster differs")
    _unchanged(packet)
    _require(
        observed_runtime() == runtime, "runtime/import state changed during inference"
    )
    destination.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(destination / "arrays.npz", **arrays)
    result = dict(
        format=RESULT_FORMAT,
        input_manifest_sha256=packet.manifest_sha256,
        qualification=packet.manifest["qualification"],
        runtime=runtime,
        models=[
            dict(
                id=row["id"],
                fingerprint=packet.models[row["id"]].fingerprint(),
                contract=packet.models[row["id"]].contract,
                recipe=packet.models[row["id"]].recipe,
            )
            for row in packet.manifest["models"]
        ],
        counts=dict(
            models=2,
            queries=16,
            means=24,
            responses=8,
            fits=0,
            updates=0,
            initializers=0,
            optimizer_steps=0,
            simulator_calls=0,
        ),
        means=means,
        responses=responses,
        arrays_file="arrays.npz",
        arrays_sha256=_sha(destination / "arrays.npz"),
        array_hash_encoding="SHA256(compact JSON [dtype.str,shape] UTF-8 + C-order bytes)",
        claim="Public saved-model forecast and computational sensitivity integration; no physical Jacobian or controller qualification.",
    )
    (destination / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = run(args.manifest, args.expected_manifest_sha256, args.output)
    print(
        json.dumps(
            dict(
                format=result["format"],
                counts=result["counts"],
                result=str(Path(args.output) / "result.json"),
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
