"""External adjudication, same-source replay and five Dart boundary challenges.

The original input packet and consumer retain their original source binding.
This later evaluator may add source files, but may not relabel or change any
previously bound source. No fitting, initialization or simulator work occurs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np

from glassbox import LearnedDynamics

from . import public_mean_implementation as implementation
from . import public_v4_numerics as numerics
from .public_mean_consumer_export import _under, export_packet

NEGATIVES = (
    "wrong_fingerprint",
    "swapped_channels",
    "changed_dt",
    "history_boundary",
    "future_command",
)
EXPECTED_COUNTS = dict(
    models=2,
    queries=16,
    means=24,
    responses=8,
    fits=0,
    updates=0,
    initializers=0,
    optimizer_steps=0,
    simulator_calls=0,
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text())


def _write(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {k: np.array(archive[k], copy=True) for k in archive.files}


def _same(a, b, label):
    a, b = np.asarray(a), np.asarray(b)
    _require(
        a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes(), label
    )


def _array_sha(value):
    value = np.asarray(value)
    return hashlib.sha256(
        json.dumps([value.dtype.str, list(value.shape)], separators=(",", ":")).encode()
        + value.tobytes(order="C")
    ).hexdigest()


def _files(root):
    result = {}
    for path in sorted(Path(root).rglob("*")):
        _require(
            not path.is_symlink(), "qualification evidence cannot contain symlinks"
        )
        if path.is_file():
            result[str(path.relative_to(root))] = _sha(path)
    return result


def associate_bindings(old, current):
    """Authenticate a source extension without rewriting the original identity."""
    for key in (
        "protocol_sha256",
        "oracle_root",
        "oracle_commit",
        "oracle_source_sha256",
        "dart_root",
        "consumer_source_sha256",
        "consumer_distribution_versions",
        "consumer_exporter_sha256",
        "consumer_prefreeze_inventory_sha256",
        "interpreter_sha256",
        "runtime",
        "machine",
    ):
        _require(
            old[key] == current[key], "evaluator changed original binding field " + key
        )
    _require(
        Path(old["interpreter"]).resolve() == Path(current["interpreter"]).resolve(),
        "evaluator interpreter identity",
    )
    for name, sha in old["public_source_sha256"].items():
        _require(
            current["public_source_sha256"].get(name) == sha,
            "evaluator changed original source " + name,
        )
    return {
        "original_root": old["public_root"],
        "original_commit": old["implementation_commit"],
        "evaluator_root": current["public_root"],
        "evaluator_commit": current["implementation_commit"],
        "unchanged_original_sources": len(old["public_source_sha256"]),
        "added_sources": sorted(
            set(current["public_source_sha256"]) - set(old["public_source_sha256"])
        ),
        "consumer_replay_uses_original_root": True,
    }


def _bindings(old_path, old_sha, current_path, current_sha):
    old = implementation.verify(old_path, old_sha)
    current = implementation.verify(current_path, current_sha)
    association = associate_bindings(old, current)
    relative = "src/glassbox/experimental/public_mean_consumer_qualification.py"
    _require(
        Path(__file__).resolve() == Path(current["public_root"]) / relative,
        "evaluator imported from another root",
    )
    _require(
        current["public_source_sha256"].get(relative) == _sha(__file__),
        "evaluator source absent from committed binding",
    )
    return old, current, association


def packet_manifest(path, expected_sha):
    path = Path(path)
    _require(
        path.name == "INPUT.json" and _sha(path) == expected_sha,
        "external consumer input anchor",
    )
    packet = _read(path)
    rows = [*packet["models"], *packet["recordings"], *packet["queries"]]
    _require(
        len(rows) == 20 and len({r["path"] for r in rows}) == 20,
        "input-only payload roster",
    )
    expected = {"INPUT.json": expected_sha}
    for row in rows:
        expected[row["path"]] = row["sha256"]
        _require(
            _sha(_under(path.parent, row["path"])) == row["sha256"],
            "consumer payload anchor",
        )
    _require(_files(path.parent) == expected, "complete input-only packet files")
    return packet


def _mean_key(row):
    return row["model_id"], row["parent"], row["query_id"], row["branch"]


def _query_key(row, branch):
    return row["model_id"], row["parent"], row["id"], branch


def map_numeric_cases(packet, cases, *, input_sha256, old_binding_sha256, data_seals):
    """The existing packet fixes identities and branch routing, never errors."""
    actual = [c for c in cases if c["source"] == "public_archive"]
    expected = [(q, branch) for q in packet["queries"] for branch in q["branches"]]
    _require(len(expected) == len(actual) == 24, "exact 24 actual numeric cases")
    models = {r["id"]: r for r in packet["models"]}
    recordings = {r["id"]: r for r in packet["recordings"]}
    result = []
    for case, (query, branch) in zip(actual, expected, strict=True):
        source = case["source_identity"]
        model, recording = models[query["model_id"]], recordings[query["recordings_id"]]
        parent = next(
            p for p in recording["parents"] if p["recording_id"] == query["parent"]
        )
        future = (
            "factual_inputs"
            if query["kind"] == "response" and branch == "factual"
            else "future_inputs"
        )
        _require(
            case["id"] == "/".join(_query_key(query, branch))
            and source["query"] == query
            and source["model"] == model
            and source["branch"] == branch
            and source["future_array_key"] == future
            and source["consumer_packet_manifest_sha256"] == input_sha256
            and source["implementation_manifest_sha256"] == old_binding_sha256
            and source["data_seal_sha256"] == data_seals[query["model_id"]]
            and source["recording_parent"] == parent
            and source["recordings_archive"]
            == {k: v for k, v in recording.items() if k != "parents"}
            and case["model_sha256"] == model["sha256"]
            and case["model_fingerprint"] == model["fingerprint"]
            and case["dt_s"] == query["dt_s"]
            and case["history_steps"] == model["history_steps"]
            and case["horizon"] == case["maximum_horizon"] == query["horizon_steps"],
            "numeric case differs from fixed consumer input/branch provenance",
        )
        result.append((case, query, branch, future))
    return result


def mean_checks(actual, public, oracle, mean, scale):
    """Unbatched same-path identity; frozen normalized oracle bounds."""
    _same(
        actual["eager"], public["actual__singles"][0], "consumer eager same-path output"
    )
    _same(
        actual["compiled"],
        public["actual__jit_singles"][0],
        "consumer compiled same-path output",
    )
    checks = {}
    for name, reference in (("eager", "singles"), ("compiled", "jit_singles")):
        checks[name + "_oracle"] = numerics.element_check(
            (actual[name] - mean) / scale,
            (oracle["quantized_inputs__" + reference][0] - mean) / scale,
            numerics.TOLERANCES["forecast32"],
        )
    checks["eager_compiled"] = numerics.element_check(
        (actual["compiled"] - mean) / scale,
        (actual["eager"] - mean) / scale,
        numerics.TOLERANCES["paths32"],
    )
    for label, reference in (
        ("jvp_public", public["actual__first_command_jvp"]),
        ("jvp_oracle", oracle["quantized_inputs__first_command_jvp"]),
    ):
        checks[label] = numerics.element_check(
            actual["jvp"] / scale,
            reference / scale,
            numerics.TOLERANCES["derivative32"],
        )
    return checks


def adjudicate(
    packet_path, packet, consumer_directory, numeric_manifest, numeric_directory, mapped
):
    """Read saved arrays only; this reduction makes no model forecasts."""
    consumer_directory, numeric_manifest, numeric_directory = map(
        Path, (consumer_directory, numeric_manifest, numeric_directory)
    )
    report = _read(consumer_directory / "result.json")
    values = _arrays(consumer_directory / report["arrays_file"])
    _require(
        report["counts"] == EXPECTED_COUNTS
        and len(report["means"]) == 24
        and len(report["responses"]) == 8,
        "consumer complete output roster",
    )
    _require(
        [_mean_key(r) for r in report["means"]]
        == [_query_key(q, b) for _, q, b, _ in mapped],
        "consumer mean identity/order",
    )
    model_rows = {m["id"]: m for m in packet["models"]}
    model_arrays, models = {}, {}
    for identity, row in model_rows.items():
        path = Path(packet_path).parent / row["path"]
        model_arrays[identity] = _arrays(path)
        models[identity] = LearnedDynamics.load(path)
        _require(
            models[identity].fingerprint() == row["fingerprint"]
            and models[identity].contract == row["contract"],
            "public consumer model identity",
        )
    _require(
        report["models"]
        == [
            dict(
                id=r["id"],
                fingerprint=models[r["id"]].fingerprint(),
                contract=models[r["id"]].contract,
                recipe=models[r["id"]].recipe,
            )
            for r in packet["models"]
        ],
        "consumer output model descriptors",
    )
    workers = {}
    for mode in ("public32", "oracle64"):
        rows = _read(numeric_directory / mode / "result.json")["cases"]
        _require(
            len({r["id"] for r in rows}) == len(rows), "unique numeric worker cases"
        )
        workers[mode] = {r["id"]: r for r in rows}
    means, used, branches = [], set(), {}
    for index, (entry, query, branch, future) in enumerate(mapped):
        row = report["means"][index]
        identity = query["model_id"]
        source = _arrays(numeric_manifest.parent / entry["path"])
        tape = _arrays(Path(packet_path).parent / query["path"])
        for name in numerics.INPUTS:
            _same(
                source[name],
                tape[future if name == "future_inputs" else name][None],
                "numeric source input bytes",
            )
        for name, array in model_arrays[identity].items():
            if name.startswith(("param_", "norm_")):
                _same(source[name], array, "numeric source model tree")
        selected = {k: values[v] for k, v in row["arrays"].items()}
        _require(
            set(selected)
            == {"eager", "compiled", "jvp", "envelope", "tangent", "offsets_s"},
            "consumer mean array roster",
        )
        _require(row["index"] == index, "consumer mean index")
        for name, array in selected.items():
            _require(
                row["shapes"][name] == list(array.shape)
                and row["dtypes"][name] == array.dtype.str
                and np.isfinite(array).all(),
                "consumer array shape/dtype/finiteness",
            )
            _require(row["arrays"][name] not in used, "consumer array alias")
            used.add(row["arrays"][name])
        for name in ("eager", "compiled", "jvp"):
            _require(
                selected[name].dtype == np.dtype("float32"), "consumer actual default32"
            )
        tangent = np.zeros_like(tape[future])
        tangent[0, 0] = 1.0
        _same(selected["tangent"], tangent, "explicit native first-command tangent")
        _same(
            selected["offsets_s"],
            np.arange(1, query["horizon_steps"] + 1) * query["dt_s"],
            "consumer forecast offsets",
        )
        _same(
            selected["envelope"],
            models[identity].envelope(query["horizon_steps"]),
            "public envelope provenance",
        )
        for name, value in {
            "kind": query["kind"],
            "scope": query["scope"],
            "segment_id": query["segment_id"],
            "source_origin": query["source_origin"],
            "dt_s": query["dt_s"],
            "query_payload_sha256": query["sha256"],
            "source_query_sha256": query["source_query_sha256"],
            "sensitivity_command_channel": model_rows[identity]["contract"][
                "input_channels"
            ][0],
            "sensitivity_direction": "one native command unit at future row 0, channel 0",
            "future_tape_sha256": _array_sha(tape[future]),
        }.items():
            _require(row[name] == value, "consumer semantic mean descriptor " + name)
        command_channel = model_rows[identity]["contract"]["input_channels"][0]
        _require(
            row["final_channels"]
            == [
                dict(
                    channel=name,
                    forecast=float(selected["eager"][-1, i]),
                    envelope_half_width=float(selected["envelope"][-1, i]),
                    first_command_sensitivity=float(selected["jvp"][-1, i]),
                    sensitivity_per_command_channel=command_channel,
                )
                for i, name in enumerate(
                    model_rows[identity]["contract"]["state_channels"]
                )
            ],
            "consumer physical channel summary",
        )
        transforms = {
            mode: _arrays(numeric_directory / mode / workers[mode][entry["id"]]["path"])
            for mode in workers
        }
        checks = mean_checks(
            selected,
            transforms["public32"],
            transforms["oracle64"],
            source["norm_state_mean"],
            source["norm_state_scale"],
        )
        means.append(
            {
                "id": entry["id"],
                "checks": checks,
                "passed": all(c["passed"] for c in checks.values()),
            }
        )
        branches[_query_key(query, branch)] = (row, selected, transforms, source)
    responses = []
    expected_responses = [q for q in packet["queries"] if q["kind"] == "response"]
    for index, (row, query) in enumerate(
        zip(report["responses"], expected_responses, strict=True)
    ):
        a, b = (
            branches[_query_key(query, branch)] for branch in ("intervened", "factual")
        )
        _require(
            row["index"] == index
            and (row["model_id"], row["parent"], row["query_id"])
            == _query_key(query, "")[:3]
            and row["intervened_mean"] == a[0]["index"]
            and row["factual_mean"] == b[0]["index"],
            "consumer response branch identity",
        )
        _require(
            set(row["arrays"]) == {"eager", "compiled", "command_delta"},
            "consumer response array roster",
        )
        _require(
            row["branch_tape_sha256"]
            == {
                "intervened": a[0]["future_tape_sha256"],
                "factual": b[0]["future_tape_sha256"],
            },
            "consumer branch tape hash links",
        )
        tape = _arrays(Path(packet_path).parent / query["path"])
        checks = {}
        for name in ("eager", "compiled", "command_delta"):
            key = row["arrays"][name]
            _require(key not in used, "consumer response array alias")
            used.add(key)
            expected = (
                tape["future_inputs"] - tape["factual_inputs"]
                if name == "command_delta"
                else a[1][name] - b[1][name]
            )
            _same(values[key], expected, "consumer response exact subtraction " + name)
            if name != "command_delta":
                path = "singles" if name == "eager" else "jit_singles"
                oracle = (
                    a[2]["oracle64"]["quantized_inputs__" + path][0]
                    - b[2]["oracle64"]["quantized_inputs__" + path][0]
                )
                checks[name + "_oracle"] = numerics.element_check(
                    values[key] / a[3]["norm_state_scale"],
                    oracle / a[3]["norm_state_scale"],
                    numerics.TOLERANCES["forecast32"],
                )
        responses.append(
            {
                "query": list(_query_key(query, "")[:3]),
                "checks": checks,
                "passed": all(c["passed"] for c in checks.values()),
            }
        )
    _require(used == set(values), "complete consumer output array roster")
    return {
        "passed": all(r["passed"] for r in means + responses),
        "means": means,
        "responses": responses,
        "forecasts_executed": 0,
        "envelopes_are_provenance_not_coverage_qualification": True,
        "physical_derivative_fidelity_qualified": False,
    }


_NEGATIVE_WORKER = r"""
import json, pathlib, sys, traceback
from crazydart import glassbox_forecast as consumer
manifest, expected, output = sys.argv[1:]
runtime = consumer.observed_runtime()
calls = []
def forbidden(*args, **kwargs):
    calls.append("predict")
    raise AssertionError("negative validation reached prediction")
consumer.LearnedDynamics.predict = forbidden
result = {"rejected":False,"prediction_calls":0,"runtime":runtime}
try:
    consumer.validate_packet(manifest, expected)
except consumer.PacketError as error:
    result.update(rejected=True,error_type=type(error).__name__,error=str(error),frames=[f.name for f in traceback.extract_tb(error.__traceback__)])
result["prediction_calls"] = len(calls)
assert consumer.observed_runtime() == runtime
with pathlib.Path(output).open("x") as handle:
    json.dump(result,handle,sort_keys=True,indent=2)
    handle.write("\n")
assert result["rejected"] and not calls
"""


def _launch(command, binding, stem, *, timeout=7200):
    environment = implementation.consumer_environment(binding)
    stem = Path(stem)
    _write(
        Path(str(stem) + ".command.json"),
        {
            "argv": command,
            "cwd": binding["dart_root"],
            "environment": {
                k: environment.get(k)
                for k in ("PYTHONPATH", "JAX_ENABLE_X64", "SCIPY_ARRAY_API")
            },
            "timeout_s": timeout,
        },
    )
    started = time.perf_counter()
    with (
        Path(str(stem) + ".stdout.log").open("x") as out,
        Path(str(stem) + ".stderr.log").open("x") as err,
    ):
        try:
            child = subprocess.run(
                command,
                cwd=binding["dart_root"],
                env=environment,
                stdout=out,
                stderr=err,
                check=False,
                timeout=timeout,
            )
            result = {"returncode": child.returncode, "timed_out": False}
        except subprocess.TimeoutExpired:
            result = {"returncode": None, "timed_out": True}
    result.update(
        elapsed_s=time.perf_counter() - started,
        stdout_sha256=_sha(Path(str(stem) + ".stdout.log")),
        stderr_sha256=_sha(Path(str(stem) + ".stderr.log")),
    )
    _write(Path(str(stem) + ".exit.json"), result)
    _require(result["returncode"] == 0, "consumer child failed; attempt retained")
    return result


def mutate_packet(directory, case):
    """One fixed defect; repair local hashes, never an external trust anchor."""
    _require(case in NEGATIVES, "unknown consumer challenge")
    directory = Path(directory)
    manifest_path = directory / "INPUT.json"
    packet = _read(manifest_path)
    original_sha = _sha(manifest_path)
    if case == "wrong_fingerprint":
        value = packet["models"][0]["fingerprint"]
        packet["models"][0]["fingerprint"] = ("0" if value[0] != "0" else "1") + value[
            1:
        ]
        category = "loaded public model identity/contract/timing differs"
    elif case == "swapped_channels":
        channels = packet["models"][0]["contract"]["state_channels"]
        _require(len(channels) >= 2, "challenge needs two semantic channels")
        channels[0], channels[1] = channels[1], channels[0]
        category = "loaded public model identity/contract/timing differs"
    elif case == "changed_dt":
        packet["queries"][0]["dt_s"] *= 2
        category = "query timing differs"
    elif case == "history_boundary":
        query = packet["queries"][0]
        model = next(m for m in packet["models"] if m["id"] == query["model_id"])
        recording = next(
            r for r in packet["recordings"] if r["id"] == query["recordings_id"]
        )
        parent = next(
            p for p in recording["parents"] if p["recording_id"] == query["parent"]
        )
        segment = next(
            s for s in parent["segments"] if s["segment_id"] == query["segment_id"]
        )
        query["source_origin"] = segment["start_row"] + model["history_steps"] - 1
        category = "history crosses or misidentifies a segment boundary"
    else:
        query = next(q for q in packet["queries"] if q["kind"] == "factual")
        recording = next(
            r for r in packet["recordings"] if r["id"] == query["recordings_id"]
        )
        parent = next(
            p for p in recording["parents"] if p["recording_id"] == query["parent"]
        )
        _require(
            any(
                s["start_row"]
                <= query["source_origin"]
                < s["start_row"] + s["input_rows"]
                for s in parent["segments"]
            ),
            "command challenge requires authenticated outgoing command",
        )
        path = directory / query["path"]
        values = _arrays(path)
        values["future_inputs"][0, 0] += 0.125
        _require(
            np.isfinite(values["future_inputs"]).all(), "finite command alteration"
        )
        np.savez_compressed(path, **values)
        query["sha256"] = _sha(path)
        category = "available factual future commands differs from recording bytes"
    # The caller can challenge a raw payload before this separate resealing step.
    return {
        "case": case,
        "original_manifest_sha256": original_sha,
        "packet": packet,
        "expected_semantic_error": category,
    }


def _negative_validation(manifest, expected, binding, runtime, script, stem, category):
    result_path = Path(str(stem) + ".json")
    _launch(
        [
            binding["interpreter"],
            str(script),
            str(manifest),
            expected,
            str(result_path),
        ],
        binding,
        stem,
        timeout=120,
    )
    report = _read(result_path)
    _require(
        report["runtime"] == runtime
        and report["rejected"]
        and report["prediction_calls"] == 0,
        "negative did not preserve public validation boundary",
    )
    _require(
        report["error"] == category,
        "negative failed at an unintended validation boundary",
    )
    return {
        "report_sha256": _sha(result_path),
        "error": report["error"],
        "prediction_calls": 0,
        "frames": report["frames"],
    }


def run(
    output,
    *,
    old_binding_path,
    old_binding_sha256,
    evaluator_binding_path,
    evaluator_binding_sha256,
    input_manifest,
    input_sha256,
    consumer_directory,
    consumer_result_sha256,
    numeric_manifest,
    numeric_manifest_sha256,
    numeric_directory,
    numeric_run_sha256,
    data_roots,
    data_seals,
):
    output = Path(output).resolve()
    _require(
        all(
            not output.is_relative_to(Path(p).resolve())
            for p in (
                Path(input_manifest).parent,
                consumer_directory,
                numeric_directory,
                Path(numeric_manifest).parent,
            )
        ),
        "qualification output must be separate from original evidence",
    )
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "status": "failed",
        "fits": 0,
        "updates": 0,
        "initializers": 0,
        "simulator_calls": 0,
        "request": {
            "old_binding_sha256": old_binding_sha256,
            "evaluator_binding_sha256": evaluator_binding_sha256,
            "input_sha256": input_sha256,
            "consumer_result_sha256": consumer_result_sha256,
            "numeric_manifest_sha256": numeric_manifest_sha256,
            "numeric_run_sha256": numeric_run_sha256,
            "data_seals": data_seals,
        },
    }
    try:
        old, _current, association = _bindings(
            old_binding_path,
            old_binding_sha256,
            evaluator_binding_path,
            evaluator_binding_sha256,
        )
        result["source_association"] = association
        packet = packet_manifest(input_manifest, input_sha256)
        original_files = _files(Path(input_manifest).parent)
        _require(
            packet["qualification"]["implementation_commit"]
            == old["implementation_commit"]
            and packet["qualification"]["exporter_source_sha256"]
            == old["consumer_exporter_sha256"],
            "original packet producer identity",
        )
        original_result_path = Path(consumer_directory) / "result.json"
        _require(
            _sha(original_result_path) == consumer_result_sha256,
            "external original consumer result anchor",
        )
        report = _read(original_result_path)
        _require(
            report["input_manifest_sha256"] == input_sha256
            and report["qualification"] == packet["qualification"],
            "consumer original packet identity",
        )
        _require(
            report["arrays_file"] == "arrays.npz"
            and _sha(Path(consumer_directory) / "arrays.npz")
            == report["arrays_sha256"],
            "consumer output arrays anchor",
        )
        original_consumer_files = _files(consumer_directory)
        runtime = implementation.consumer_preflight(
            old_binding_path, old_binding_sha256, output / "preflight"
        )
        _require(
            report["runtime"] == runtime, "original consumer runtime/import identity"
        )
        numeric_result = numerics.verify_saved(
            numeric_manifest,
            numeric_directory,
            expected_manifest_sha256=numeric_manifest_sha256,
            expected_run_sha256=numeric_run_sha256,
        )
        _require(
            "cases" in numeric_result,
            "numeric workers did not complete; no complete oracle evidence",
        )
        numerical_rows = _read(Path(numeric_directory) / "result.json")["results"]
        result["separate_numeric_qualification"] = {
            "overall_passed": numeric_result["passed"],
            "is_consumer_specific_parity_gate": False,
            "failed_required_checks": [
                {"case": row["id"], "check": name}
                for row in numerical_rows
                for name, check in row["checks"].items()
                if not check["passed"]
                and (row["stress"] is None or name.endswith(("/causality", "/dtype")))
            ],
            "failed_diagnostic_stress_checks": sum(
                not check["passed"]
                for row in numerical_rows
                if row["stress"] is not None
                for name, check in row["checks"].items()
                if not name.endswith(("/causality", "/dtype"))
            ),
        }
        numeric_packet = _read(numeric_manifest)
        provenance = numeric_packet["provenance"]
        for key in (
            "implementation_commit",
            "public_source_sha256",
            "oracle_source_sha256",
            "runtime",
            "protocol_sha256",
        ):
            _require(
                provenance[key] == old[key],
                "actual numerics original source identity " + key,
            )
        _require(
            provenance["implementation_manifest_sha256"] == old_binding_sha256
            and provenance["consumer_packet_manifest_sha256"] == input_sha256,
            "actual numerics external packet identity",
        )
        mapped = map_numeric_cases(
            packet,
            numeric_packet["cases"],
            input_sha256=input_sha256,
            old_binding_sha256=old_binding_sha256,
            data_seals=data_seals,
        )
        reproduced = export_packet(
            output / "source-reconstruction",
            models={
                r["id"]: Path(input_manifest).parent / r["path"]
                for r in packet["models"]
            },
            data_roots=data_roots,
            data_seals=data_seals,
            implementation_commit=old["implementation_commit"],
        )
        _require(
            reproduced["sha256"] == input_sha256
            and _files(output / "source-reconstruction") == original_files,
            "full source roster/observation/input-only export replay",
        )
        result["adjudication"] = adjudicate(
            input_manifest,
            packet,
            consumer_directory,
            numeric_manifest,
            numeric_directory,
            mapped,
        )
        fresh = output / "consumer-replay"
        result["replay_stage"] = _launch(
            [
                old["interpreter"],
                "-m",
                "crazydart.glassbox_forecast",
                "--manifest",
                str(Path(input_manifest).resolve()),
                "--expected-manifest-sha256",
                input_sha256,
                "--output",
                str(fresh),
            ],
            old,
            output / "consumer-replay-stage",
        )
        replayed = _read(fresh / "result.json")
        _require(replayed == report, "fresh same-source consumer report differs")
        prior_arrays, fresh_arrays = (
            _arrays(Path(consumer_directory) / "arrays.npz"),
            _arrays(fresh / "arrays.npz"),
        )
        _require(set(prior_arrays) == set(fresh_arrays), "fresh consumer array roster")
        for name, values in prior_arrays.items():
            _same(fresh_arrays[name], values, "fresh same-path consumer array " + name)
        result["fresh_consumer_replay"] = {
            "passed": True,
            "result_sha256": _sha(fresh / "result.json"),
            "arrays": len(fresh_arrays),
            "original_source_root": old["public_root"],
        }
        script = output / "negative-worker.py"
        script.write_text(_NEGATIVE_WORKER)
        (output / "negatives").mkdir()
        result["negatives"] = []
        for case in NEGATIVES:
            directory = output / "negatives" / case
            directory.mkdir()
            copied = directory / "input"
            shutil.copytree(Path(input_manifest).parent, copied)
            mutation = mutate_packet(copied, case)
            checks = {}
            if case == "future_command":
                checks["raw_payload"] = _negative_validation(
                    copied / "INPUT.json",
                    input_sha256,
                    old,
                    runtime,
                    script,
                    directory / "raw-payload",
                    "payload SHA mismatch",
                )
            (copied / "INPUT.json").write_text(
                json.dumps(
                    mutation["packet"], indent=2, sort_keys=True, allow_nan=False
                )
                + "\n"
            )
            changed_sha = _sha(copied / "INPUT.json")
            _require(changed_sha != input_sha256, "negative manifest is not altered")
            checks["external_anchor"] = _negative_validation(
                copied / "INPUT.json",
                input_sha256,
                old,
                runtime,
                script,
                directory / "external-anchor",
                "external manifest SHA mismatch",
            )
            checks["coherently_resealed_semantic"] = _negative_validation(
                copied / "INPUT.json",
                changed_sha,
                old,
                runtime,
                script,
                directory / "semantic",
                mutation["expected_semantic_error"],
            )
            evidence = {
                "case": case,
                "passed": True,
                "checks": checks,
                "original_manifest_sha256": input_sha256,
                "altered_manifest_sha256": changed_sha,
                "altered_files": _files(copied),
                "repaired_local_hashes_are_not_external_authentication": True,
            }
            _write(directory / "result.json", evidence)
            result["negatives"].append(evidence)
        _bindings(
            old_binding_path,
            old_binding_sha256,
            evaluator_binding_path,
            evaluator_binding_sha256,
        )
        _require(
            _files(Path(input_manifest).parent) == original_files
            and _files(consumer_directory) == original_consumer_files,
            "original consumer evidence changed",
        )
        numerics.verify_saved(
            numeric_manifest,
            numeric_directory,
            expected_manifest_sha256=numeric_manifest_sha256,
            expected_run_sha256=numeric_run_sha256,
        )
        result.update(
            status="complete",
            passed=result["adjudication"]["passed"] and len(result["negatives"]) == 5,
            claim="Dart-owned separately launched public saved-model forecast/sensitivity integration on the fixed input roster. No physical Jacobian, uncertainty, controller, update-quality, production-adoption or arbitrary-system qualification.",
        )
    except Exception as error:
        result.update(
            error_type=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc(),
        )
        raise
    finally:
        result["files"] = _files(output)
        _write(output / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--expected-request-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    _require(
        _sha(args.request) == args.expected_request_sha256,
        "external consumer qualification request anchor",
    )
    run(args.output, **_read(args.request))


if __name__ == "__main__":
    main()
