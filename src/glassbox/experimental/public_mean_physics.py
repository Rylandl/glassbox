"""Fresh +11M physical cohort, with isolated pinned historical simulators.

This file is also a standalone worker. No Glassbox module is imported at module
load: the worker authenticates its requested historical checkout before imports.
It collects only test parents; calibration recordings are reused elsewhere.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

PROTOCOL_SHA256 = "d3a8d0eeeeab2aac923cad7f1f5fe1c8a49fc0ffa765e84b3d7c6bc80cec8f99"
ORACLE_COMMIT = "8b61830c9c353dbb25edc6e63a76886d0ce9b9d3"
SIMULATORS = ("crazyflow", "cascade")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def _require(value, label):
    if not value:
        raise ValueError(label)


def fresh_test_roster(original, *, shift=11000000):
    """Exact original order, shifted identities/seeds, and no calibration."""
    _require(shift == 11000000, "frozen physical seed shift")
    rows = []
    old = [r for r in original["recordings"] if r["role"] == "test"]
    for source in old:
        row = copy.deepcopy(source)
        _require(
            row["id"].rsplit("/", 1)[-1] == "test-" + str(row["seed"]),
            "original physical identity",
        )
        row["seed"] += shift
        row["id"] = row["id"].rsplit("-", 1)[0] + "-" + str(row["seed"])
        rows.append(row)
    identities = {r["id"] for r in rows}
    pairs = {(r["simulator"], r["seed"]) for r in rows}
    _require(len(rows) == len(identities) == len(pairs) == 168, "168 fresh parents")
    for simulator in SIMULATORS:
        _require(
            sum(r["simulator"] == simulator for r in rows) == 84,
            "84 parents per simulator",
        )
    for old_shift in range(0, 11000000, 1000000):
        _require(
            not pairs.intersection(
                (r["simulator"], r["seed"] + old_shift) for r in old
            ),
            "physical seed overlap with inspected cohort",
        )
    return rows


def _inventory(directory):
    return {
        str(p.relative_to(directory)): digest(p)
        for p in sorted(Path(directory).rglob("*"))
        if p.is_file() and p.name != "seal.json"
    }


def verify_data(directory, expected_sha256):
    directory = Path(directory).resolve()
    _require(digest(directory / "seal.json") == expected_sha256, "external data seal")
    seal = read(directory / "seal.json")
    _require(seal["format"] == "glassbox-public-mean-physics-v1", "physical format")
    _require(seal["protocol_sha256"] == PROTOCOL_SHA256, "physical protocol")
    _require(_inventory(directory) == seal["files"], "physical payload inventory")
    _require(len(read(directory / "records.json")) == 84, "physical record count")
    return seal


def _worker_identity(request):
    """Authenticate both the executing file and every historical source."""
    binding_path = Path(request["binding_path"])
    _require(
        digest(binding_path) == request["binding_sha256"],
        "external implementation seal",
    )
    binding = read(binding_path)
    _require(binding["protocol_sha256"] == PROTOCOL_SHA256, "implementation protocol")
    root = Path(binding["oracle_root"]).resolve()
    commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    _require(commit == ORACLE_COMMIT, "historical physical commit")
    for relative, expected in binding["oracle_source_sha256"].items():
        _require(digest(root / relative) == expected, "historical source: " + relative)
    relative = "src/glassbox/experimental/public_mean_physics.py"
    _require(
        digest(__file__) == binding["public_source_sha256"][relative]
        and Path(__file__).resolve() == Path(binding["public_root"]) / relative,
        "physical worker source association",
    )
    protocol_file = (
        Path(binding["public_root"]) / "docs/harness/public-mean-qualification-v1.json"
    )
    _require(digest(protocol_file) == PROTOCOL_SHA256, "frozen physical protocol")
    protocol = read(protocol_file)
    for dependency in protocol["protocol_dependencies"]:
        _require(
            digest(Path(binding["public_root"]) / dependency["path"])
            == dependency["sha256"],
            "protocol dependency",
        )
    _require(
        Path(sys.executable).resolve() == Path(binding["interpreter"]).resolve(),
        "physical interpreter",
    )
    _require(
        digest(Path(sys.executable).resolve()) == binding["interpreter_sha256"],
        "physical interpreter bytes",
    )
    versions = {
        name: importlib.metadata.version(name)
        for name in ("jax", "jaxlib", "numpy", "scipy")
    }
    _require(
        all(versions[k] == protocol["runtime"][k] for k in versions),
        "physical numerical runtime",
    )
    _require(
        platform.python_version() == protocol["runtime"]["python"],
        "physical Python runtime",
    )
    import jax

    import glassbox

    _require(
        Path(glassbox.__file__).resolve() == root / "src/glassbox/__init__.py",
        "historical physical import location",
    )
    _require(
        jax.config.x64_enabled and jax.default_backend() == "cpu",
        "physical float64 CPU",
    )
    _require(os.environ.get("SCIPY_ARRAY_API") == "1", "physical SciPy array API")
    return binding, protocol


def _imports(binding):
    root = Path(binding["oracle_root"]).resolve()
    imported = {}
    for name, module in tuple(sys.modules.items()):
        if name != "glassbox" and not name.startswith("glassbox."):
            continue
        file = getattr(module, "__file__", None)
        if file is None:
            continue
        path = Path(file).resolve()
        _require(
            path.is_relative_to(root), "mixed Glassbox versions in physical worker"
        )
        relative = str(path.relative_to(root))
        _require(
            binding["oracle_source_sha256"].get(relative) == digest(path),
            "unbound physical import",
        )
        imported[name] = {"path": str(path), "sha256": digest(path)}
    return imported


def _worker(request_path):
    request = read(request_path)
    binding, qualification = _worker_identity(request)
    from glassbox.experimental import expanded_training_cache_experiment as historical

    historical_runtime = historical.check_sources()
    p = historical.protocol()
    original_entry = qualification["fresh_confirmation"]["original_protocol"]
    original_path = Path(binding["oracle_root"]) / original_entry["path"]
    _require(
        digest(original_path) == original_entry["sha256"], "original cohort protocol"
    )
    original = read(original_path)
    # The retained resolved protocol updates descriptive scoring/admission prose.
    # Collection uses its exact pinned operators; only the original test roster
    # supplies the new +11M identities. Do not compare obsolete policy prose.
    _require(p["cells"] == original["cells"], "unchanged physical conditions")
    for simulator in SIMULATORS:
        _require(
            p["generation"][simulator] == original["generation"][simulator],
            "unchanged simulator generation: " + simulator,
        )
    for key in ("factual_origins_s", "horizons_s"):
        _require(
            p["evaluation"][key] == original["evaluation"][key],
            "unchanged physical query grid",
        )
    _require(
        p["evaluation"]["responses"]["origins_s"]
        == original["evaluation"]["responses"]["origins_s"],
        "unchanged physical intervention origins",
    )
    entries = [
        r for r in fresh_test_roster(original) if r["simulator"] == request["simulator"]
    ]
    flight = historical.flight()
    fixture = flight.fixture_for(request["simulator"], p)
    directory = Path(request["directory"])
    replaying = request["stage"] == "replay"
    if replaying:
        seal = verify_data(directory, request["expected_data_sha256"])
        _require(
            seal["implementation_sha256"] == request["binding_sha256"],
            "physical implementation association",
        )
        _require(
            seal["historical_runtime"] == historical_runtime, "physical runtime replay"
        )
    else:
        directory.mkdir(parents=True, exist_ok=False)
    array_count = 0

    def json_payload(relative, value):
        path = directory / relative
        if replaying:
            _require(read(path) == value, "physical JSON replay: " + relative)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            write(path, value)

    def array_payload(relative, value):
        nonlocal array_count
        path = directory / relative
        if replaying:
            flight.array_equal(value, flight.load_arrays(path), relative)
        else:
            flight.save_arrays(path, value)
        array_count += len(value)

    json_payload(
        "configuration.json", dict(fixture.config(), spec=fixture.spec.to_dict())
    )
    json_payload("planned-records.json", entries)
    cells = {c["id"]: c for c in p["cells"][request["simulator"]]}
    records, all_queries = [], []
    for entry in entries:
        cell = cells[entry["cell"]]
        print(("replay" if replaying else "collect") + " " + entry["id"], flush=True)
        try:
            arrays, metadata = fixture.generate(entry, cell)
        except Exception as error:
            if type(error).__name__ != "FixtureSetupError":
                raise
            arrays = None
            metadata = {"setup_failure": str(error), "exception": type(error).__name__}
        prefix = str(Path("parents") / entry["id"])
        record = dict(entry, cell_facts=cell, prefix=prefix, metadata=metadata)
        if arrays is not None:
            array_payload(prefix + ".npz", arrays)
            record.update(
                validity=flight.validity(arrays), achieved=flight.achieved(arrays)
            )
        else:
            record["validity"] = dict(
                valid_transitions=0,
                valid_initial=False,
                failure_index=0,
                reason="setup_failure",
            )
        json_payload(prefix + ".json", record)
        records.append(record)
        queries, values, branches = flight.make_queries(fixture, arrays, entry, cell, p)
        for query in queries:
            query["scope"] = cell["group"]
            query["path"] = str(Path("queries") / entry["id"] / (query["id"] + ".npz"))
            array_payload(query["path"], values[query["id"]])
        for name, (values, detail) in branches.items():
            prefix = str(Path("branches") / entry["id"] / name)
            array_payload(prefix + ".npz", values)
            json_payload(prefix + ".json", detail)
        all_queries.extend(queries)
    json_payload("records.json", records)
    json_payload("queries.json", all_queries)
    imported = _imports(binding)
    result = {
        "format": "glassbox-public-mean-physics-v1",
        "protocol_sha256": PROTOCOL_SHA256,
        "implementation_sha256": request["binding_sha256"],
        "historical_runtime": historical_runtime,
        "simulator": request["simulator"],
        "test_seed_shift_from_original": 11000000,
        "parents": len(records),
        "queries": len(all_queries),
        "arrays": array_count,
        "imported_sources": imported,
        "calibration_collected": 0,
        "fits": 0,
        "files": _inventory(directory),
    }
    _worker_identity(request)
    if replaying:
        _require(seal == result, "complete physical replay evidence")
        write(
            request["result_path"],
            dict(
                exact=True,
                parents=len(records),
                queries=len(all_queries),
                arrays=array_count,
                fits=0,
                data_sha256=request["expected_data_sha256"],
            ),
        )
    else:
        write(directory / "seal.json", result)
    return result


def run(
    simulator,
    output,
    *,
    binding_path,
    binding_sha256,
    replay_data=None,
    expected_data_sha256=None,
):
    """One bounded stage; never restart a failed or existing attempt."""
    from .public_mean_implementation import verify

    _require(simulator in SIMULATORS, "physical simulator")
    binding = verify(binding_path, binding_sha256)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    if replay_data is not None:
        _require(expected_data_sha256 is not None, "physical replay external anchor")
        verify_data(replay_data, expected_data_sha256)
    request = dict(
        simulator=simulator,
        stage="replay" if replay_data is not None else "generate",
        binding_path=str(Path(binding_path).resolve()),
        binding_sha256=binding_sha256,
        directory=str(
            Path(replay_data).resolve() if replay_data is not None else output / "data"
        ),
        expected_data_sha256=expected_data_sha256,
        result_path=str(output / "replay.json"),
    )
    write(output / "request.json", request)
    env = dict(os.environ)
    env.update(
        PYTHONPATH=str(Path(binding["oracle_root"]) / "src")
        + os.pathsep
        + "/private/tmp/glassbox-cascade-e8f6ba6/src",
        JAX_ENABLE_X64="1",
        SCIPY_ARRAY_API="1",
    )
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        str(output / "request.json"),
    ]
    write(
        output / "command.json",
        dict(
            argv=command,
            hard_timeout_s=14400,
            environment_overrides={
                k: env[k] for k in ("PYTHONPATH", "JAX_ENABLE_X64", "SCIPY_ARRAY_API")
            },
        ),
    )
    started = time.monotonic()
    result = dict(
        status="incomplete",
        request_sha256=digest(output / "request.json"),
        implementation_sha256=binding_sha256,
    )
    try:
        with (output / "worker.log").open("x") as log:
            process = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                timeout=14400,
                check=False,
            )
        result["returncode"] = process.returncode
        _require(
            process.returncode == 0, "physical worker failed; retained without retry"
        )
        data_sha = (
            expected_data_sha256
            if replay_data is not None
            else digest(output / "data/seal.json")
        )
        verify_data(request["directory"], data_sha)
        result.update(status="complete", data_sha256=data_sha)
    except BaseException as error:
        result.update(
            status="hard_timeout_incomplete"
            if isinstance(error, subprocess.TimeoutExpired)
            else "failed",
            error_type=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc(),
        )
        raise
    finally:
        result.update(elapsed_s=time.monotonic() - started, files=_inventory(output))
        write(output / "run.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", required=True)
    _worker(parser.parse_args().worker)
