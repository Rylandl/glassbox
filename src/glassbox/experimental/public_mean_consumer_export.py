"""Input-only packet exporter for the independent Dart forecast consumer.

Selection uses input completeness and the frozen first/last rule. Truth, prediction
errors and derivative magnitudes are never selection inputs. No fitting occurs.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path, PurePosixPath

import numpy as np

from glassbox import LearnedDynamics, SequenceCollection, SequenceSegment
from glassbox.io.recordings import save_recordings

from .public_mean_scoring import PROTOCOL_SHA256

SIMULATORS = ("crazyflow", "cascade")
SCOPES = {"primary", "heading_shift", "maneuver_shift", "speed_shift", "wind_shift"}
SELECTION_SHA256 = "366fd718b490dc2bc44e4a8724f74c3d3b2eda89495fa8eb4a3788f7f2980d08"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path):
    return json.loads(Path(path).read_text())


def _under(root, relative):
    if (
        not isinstance(relative, str)
        or "\\" in relative
        or relative.startswith("/")
        or any(part in ("", ".", "..") for part in relative.split("/"))
    ):
        raise ValueError("source path is not a normalized packet-relative path")
    parts = PurePosixPath(relative).parts
    current = Path(root)
    for part in parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("source paths cannot contain symlinks")
    if not current.is_file() or not current.resolve().is_relative_to(
        Path(root).resolve()
    ):
        raise ValueError("source payload is not a regular file below its root")
    return current


def verify_data(root, expected_seal):
    root = Path(root)
    if sha256(root / "seal.json") != expected_seal:
        raise ValueError("data seal differs from the external anchor")
    seal = _json(root / "seal.json")
    for relative, expected in seal["files"].items():
        if sha256(_under(root, relative)) != expected:
            raise ValueError(f"sealed data payload differs: {relative}")
    return seal


def _query_arrays(path, kind):
    keys = ["past_states", "past_inputs", "future_inputs"]
    if kind == "response":
        keys.append("factual_inputs")
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in keys}


def select_queries(rows, data_root, model):
    """Eight exact input-complete identities per simulator, before prediction."""
    contract = model.contract
    c, h = model.history_steps, model.horizon_steps
    d, m = len(contract["state_channels"]), len(contract["input_channels"])
    shapes = {
        "past_states": (c + 1, d),
        "past_inputs": (c, m),
        "future_inputs": (h, m),
        "factual_inputs": (h, m),
    }
    strata = {
        (scope, kind): []
        for scope in ("primary", "shifted")
        for kind in ("factual", "response")
    }
    identities = set()
    for row in rows:
        if row["scope"] not in SCOPES or row["kind"] not in ("factual", "response"):
            raise ValueError("unknown query kind or scope")
        identity = (row["parent"], row["id"])
        if identity in identities:
            raise ValueError("duplicate source query identity")
        identities.add(identity)
        if type(row["history_eligible"]) is not bool:
            raise ValueError("history eligibility must be explicit")
        if not row["history_eligible"]:
            continue
        path = _under(data_root, row["path"])
        arrays = _query_arrays(path, row["kind"])
        if any(
            value.shape != shapes[key] or not np.isfinite(value).all()
            for key, value in arrays.items()
        ):
            continue
        # Source layout is a provenance check, not another selection predicate.
        if any(
            value.dtype.str != "<f8" or not value.flags.c_contiguous
            for value in arrays.values()
        ):
            raise ValueError("source query layout differs from frozen float64 arrays")
        scope = "primary" if row["scope"] == "primary" else "shifted"
        strata[scope, row["kind"]].append((identity, row, arrays))
    selected = []
    for key in sorted(strata):
        choices = sorted(strata[key], key=lambda item: item[0])
        if len(choices) < 2:
            raise ValueError(f"insufficient input-complete evidence in stratum {key}")
        selected.extend((choices[0], choices[-1]))
    return [
        (row, arrays) for _, row, arrays in sorted(selected, key=lambda item: item[0])
    ]


def _observed_parent(data_root, record, contract):
    """Use the unchanged telemetry observation map under its original precision.

    The simulator's validity protocol defines a single retained prefix. Preserve
    that entire prefix, including rows beyond the selected query histories.
    """
    import jax

    from .two_simulator_flight import OBSERVE, OBSERVED_CHANNELS

    path = _under(data_root, record["prefix"] + ".npz")
    validity = record["validity"]
    n = validity["valid_transitions"]
    if not validity["valid_initial"] or type(n) is not int or n < 1:
        raise ValueError("selected parent has no retained observed segment")
    with np.load(path, allow_pickle=False) as archive:
        states = archive["states"]
        commands = archive["commands"]
        times = archive["time_s"]
    if tuple(contract["state_channels"]) != OBSERVED_CHANNELS:
        raise ValueError("model observation contract differs from telemetry map")
    if n >= len(states) or n > len(commands) or len(times) != len(states):
        raise ValueError("parent prefix is not complete")
    dt = float(times[1] - times[0])
    if dt != contract["dt_s"] or not np.allclose(
        np.diff(times[: n + 1]), dt, rtol=0, atol=1e-12
    ):
        raise ValueError("parent sampling grid differs from the public contract")
    with jax.enable_x64(True):
        # The original query generator maps the whole parent in one call.
        observed = np.asarray(OBSERVE(states))[: n + 1]
    segment = SequenceSegment(record["id"], "valid-prefix", observed, commands[:n], dt)
    entry_hash = hashlib.sha256(
        json.dumps(
            record, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    description = {
        "recording_id": record["id"],
        "source_parent_sha256": sha256(path),
        "source_record_entry_sha256": entry_hash,
        "segments": [
            {
                "segment_id": segment.segment_id,
                "start_row": segment.start_row,
                "state_rows": len(segment.states),
                "input_rows": len(segment.inputs),
            }
        ],
    }
    return segment, description


def _same(actual, expected, label):
    if (
        actual.shape != expected.shape
        or actual.dtype != expected.dtype
        or actual.tobytes() != expected.tobytes()
    ):
        raise ValueError(f"exported {label} differs from sealed source bytes")


def export_packet(output, *, models, data_roots, data_seals, implementation_commit):
    """Create a new 20-payload input-only packet, never overwrite an attempt.

    The qualification launcher authenticates the implementation/source manifest
    before calling this function and supplies externally anchored data seals.
    """
    if any(
        set(mapping) != set(SIMULATORS) for mapping in (models, data_roots, data_seals)
    ):
        raise ValueError("consumer export requires exactly the two declared simulators")
    if (
        not isinstance(implementation_commit, str)
        or len(implementation_commit) != 40
        or any(c not in "0123456789abcdef" for c in implementation_commit)
    ):
        raise ValueError(
            "consumer export requires the committed implementation identity"
        )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "format": "glassbox-public-consumer-input-v1",
        "qualification": {
            "protocol_sha256": PROTOCOL_SHA256,
            "implementation_commit": implementation_commit,
            "exporter_module": "glassbox.experimental.public_mean_consumer_export",
            "exporter_source_sha256": sha256(__file__),
            "query_selection_spec_sha256": SELECTION_SHA256,
            "test_seed_shift_from_original": 11000000,
        },
        "models": [],
        "recordings": [],
        "queries": [],
    }
    for simulator in SIMULATORS:
        root = Path(data_roots[simulator])
        verify_data(root, data_seals[simulator])
        model_path = Path(models[simulator])
        model = LearnedDynamics.load(model_path)
        contract = model.contract
        selected = select_queries(_json(root / "queries.json"), root, model)
        records = _json(root / "records.json")
        record_map = {row["id"]: row for row in records}
        if len(record_map) != len(records):
            raise ValueError("duplicate generation-record identity")
        parent_ids = sorted({row["parent"] for row, _ in selected})
        segments, descriptions = [], []
        for parent in parent_ids:
            record = record_map[parent]
            if record["role"] != "test":
                raise ValueError("consumer selected a calibration parent")
            segment, description = _observed_parent(root, record, contract)
            segments.append(segment)
            descriptions.append(description)
        collection = SequenceCollection(
            tuple(segments),
            configuration_id=contract["configuration_id"],
            state_channels=tuple(contract["state_channels"]),
            input_channels=tuple(contract["input_channels"]),
        )
        recording_name = simulator + "-recordings.npz"
        save_recordings(collection, output / recording_name)
        model_name = simulator + "-model.npz"
        shutil.copyfile(model_path, output / model_name)
        manifest["models"].append(
            {
                "id": simulator,
                "path": model_name,
                "sha256": sha256(output / model_name),
                "fingerprint": model.fingerprint(),
                "contract": contract,
                "history_steps": model.history_steps,
                "horizon_steps": model.horizon_steps,
            }
        )
        recordings_id = simulator + "-recordings"
        manifest["recordings"].append(
            {
                "id": recordings_id,
                "model_id": simulator,
                "path": recording_name,
                "sha256": sha256(output / recording_name),
                "parents": descriptions,
            }
        )
        for index, (row, arrays) in enumerate(selected):
            matches = [
                s
                for s in segments
                if s.recording_id == row["parent"]
                and row["origin"] - model.history_steps >= s.start_row
                and row["origin"] < s.start_row + len(s.states)
            ]
            if len(matches) != 1:
                raise ValueError("selected history crosses or misses retained segment")
            segment = matches[0]
            j, c, h = (
                row["origin"] - segment.start_row,
                model.history_steps,
                model.horizon_steps,
            )
            _same(
                segment.states[j - c : j + 1],
                arrays["past_states"],
                "observation history",
            )
            _same(segment.inputs[j - c : j], arrays["past_inputs"], "command history")
            factual = (
                arrays["future_inputs"]
                if row["kind"] == "factual"
                else arrays["factual_inputs"]
            )
            if j + h <= len(segment.inputs):
                _same(segment.inputs[j : j + h], factual, "factual future commands")
            filename = f"{simulator}-query-{index:02d}.npz"
            np.savez_compressed(output / filename, **arrays)
            manifest["queries"].append(
                {
                    "model_id": simulator,
                    "recordings_id": recordings_id,
                    "parent": row["parent"],
                    "id": row["id"],
                    "segment_id": segment.segment_id,
                    "source_origin": row["origin"],
                    "kind": row["kind"],
                    "scope": row["scope"],
                    "dt_s": contract["dt_s"],
                    "horizon_steps": h,
                    "branches": ["factual"]
                    if row["kind"] == "factual"
                    else ["intervened", "factual"],
                    "path": filename,
                    "sha256": sha256(output / filename),
                    "source_query_sha256": sha256(_under(root, row["path"])),
                }
            )
    manifest["queries"].sort(
        key=lambda row: (row["model_id"], row["parent"], row["id"])
    )
    path = output / "INPUT.json"
    path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n"
    )
    if len(list(output.iterdir())) != 21 or len(manifest["queries"]) != 16:
        raise ValueError("consumer packet payload roster is incomplete")
    return {"path": str(path), "sha256": sha256(path)}
