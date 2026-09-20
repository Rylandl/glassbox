"""Frozen shared-vehicle data, source bindings and isolated native replay.

Only standard-library imports run before authentication. The same committed file
runs under the historical simulator source without importing the new learner.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_PATH = ROOT / "docs/harness/shared-vehicle-physics-v1.json"
PROTOCOL_SHA256 = "90f0c418379c62a6fc1b9a4f1ae094f74de8868a6e77f7370138cc6533dd0053"
SIMULATORS = ("crazyflow", "cascade")
MODULE = "src/glassbox/experimental/shared_vehicle_data.py"
FORMAT = "glassbox-shared-vehicle-data-v1"
STAGE_FORMAT = "glassbox-shared-vehicle-data-stage-v1"
CANDIDATE_FORMAT = "glassbox-shared-vehicle-candidate-v1"


class IntegrityError(ValueError):
    """Corrupted evidence must not be treated as a numerical failure."""


def require(value, label):
    if not value:
        raise IntegrityError(label)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def inventory(directory, excluding="run.json"):
    directory = Path(directory)
    values = {}
    for path in sorted(directory.rglob("*")):
        require(not path.is_symlink(), "symlink in evidence")
        if path.is_file() and path != directory / excluding:
            values[str(path.relative_to(directory))] = digest(path)
    return values


def sealed(path, expected):
    path = Path(path)
    require(digest(path) == expected, "external stage SHA differs")
    value = read(path)
    require(
        value["files"] == inventory(path.parent, path.name), "stage inventory differs"
    )
    return value


def anchor(entry, *, base=None):
    path = Path(entry["path"])
    path = path if path.is_absolute() else (ROOT if base is None else Path(base)) / path
    require(digest(path) == entry["sha256"], "external input SHA differs: " + str(path))
    return path


def _source_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def roster(protocol):
    original = read(anchor(protocol["dependencies"]["original_flight"]))
    prior = read(anchor(protocol["dependencies"]["roster"]))
    require(
        protocol["generation"]["confirmation"]["seed_shift_from_original"] == 14000000,
        "frozen shift",
    )
    original_test = [r for r in original["recordings"] if r["role"] == "test"]
    result = []
    for source in original_test:
        row = copy.deepcopy(source)
        require(
            row["id"].rsplit("/", 1)[-1] == "test-" + str(row["seed"]),
            "original identity",
        )
        row["seed"] += 14000000
        row["id"] = row["id"].rsplit("-", 1)[0] + "-" + str(row["seed"])
        result.append(row)
    identities = {(r["simulator"], r["seed"]) for r in result}
    require(
        len(result) == len(identities) == len({r["id"] for r in result}) == 168,
        "unique fresh parents",
    )
    excluded = {
        (r["simulator"], r["seed"])
        for r in original["recordings"] + prior["additional_training"]
    }
    for shift in range(0, 14000000, 1000000):
        excluded.update((r["simulator"], r["seed"] + shift) for r in original_test)
    require(
        not identities.intersection(excluded),
        "fresh roster overlaps inspected/train/development",
    )
    require(
        all(sum(r["simulator"] == s for r in result) == 84 for s in SIMULATORS),
        "84 parents per simulator",
    )
    return result


def read_protocol(path=PROTOCOL_PATH, expected_sha256=PROTOCOL_SHA256):
    require(
        expected_sha256 == PROTOCOL_SHA256 and digest(path) == expected_sha256,
        "frozen protocol bytes",
    )
    p = read(path)
    require(p["id"] == "shared-vehicle-physics-v1", "protocol identity")
    for entry in p["dependencies"].values():
        anchor(entry)
    roster(p)
    imported = p["imported_evidence"]
    root = Path(imported["updated_experiment"]["root"])
    require(
        digest(root / "run.json") == imported["updated_experiment"]["run_sha256"],
        "retained root",
    )
    run = read(root / "run.json")
    require(
        run["status"] == "complete" and run["residual_criteria_pass"],
        "retained outcome",
    )
    anchor(imported["updated_binding"])
    require(set(imported["controls"]) == set(SIMULATORS), "retained control roster")
    for sim, control in imported["controls"].items():
        stage = root / sim / "candidate"
        require(Path(control["root"]) == stage, "retained stage location")
        require(
            digest(stage / "run.json")
            == control["run_sha256"]
            == run["files"][sim + "/candidate/run.json"],
            "retained stage anchor",
        )
        saved = read(stage / "run.json")
        require(
            saved["status"] == "complete" and saved["simulator"] == sim,
            "retained model outcome",
        )
        require(
            control["model"] == control["files"]["model.npz"], "model anchor mirror"
        )
        for name, entry in control["files"].items():
            require(
                anchor(entry) == stage / name
                and entry["sha256"] == saved["files"][name],
                "retained payload anchor",
            )
    return p


def physical_view(protocol):
    p = copy.deepcopy(read(anchor(protocol["dependencies"]["original_flight"])))
    p["recordings"] = roster(protocol)
    return p


def resolved(protocol):
    entry = protocol["imported_evidence"]["updated_experiment"]
    root = Path(entry["root"])
    run = read(root / "run.json")
    require(digest(root / "run.json") == entry["run_sha256"], "resolved original root")
    require(
        digest(root / "resolved-protocol.json")
        == run["files"]["resolved-protocol.json"],
        "resolved original bytes",
    )
    p = read(root / "resolved-protocol.json")
    p.update(
        id=protocol["id"],
        recordings=roster(protocol),
        decision=copy.deepcopy(protocol["decision"]),
    )
    return p


def _runtime():
    return dict(
        python=platform.python_version(),
        machine=platform.machine(),
        interpreter=str(Path(sys.executable).resolve()),
        interpreter_sha256=digest(Path(sys.executable).resolve()),
        versions={
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "jax", "jaxlib")
        },
    )


def _source_identity(root, commit=None, files=None):
    root = Path(root).resolve()
    current = _git(root, "rev-parse", "HEAD")
    require(commit is None or current == commit, "source commit")
    require(
        not _git(root, "status", "--porcelain", "--untracked-files=normal"),
        "source checkout clean",
    )
    names = (
        files
        if files is not None
        else [
            name
            for name in _git(root, "ls-files").splitlines()
            if name.startswith(("src/", "tests/"))
            or name == "pyproject.toml"
            or name.startswith("docs/harness/shared-vehicle")
        ]
    )
    hashes = {name: digest(root / name) for name in names}
    if isinstance(files, dict):
        require(hashes == files, "bound source bytes")
    return dict(root=str(root), commit=current, files=hashes)


def simulator_identity(protocol):
    """Hash frozen native sources/assets without importing or constructing physics."""
    entry = protocol["dependencies"]["simulator_sources"]
    pins = read(anchor(entry))
    cf, ca = pins["crazyflow"], pins["cascade"]
    cf_files = {
        k: v["sha256"] if isinstance(v, dict) else v
        for k, v in cf["official_source_roster"]["files"].items()
    }
    cf_root = Path(cf["installed_package"]["path"])
    result = dict(
        crazyflow=dict(
            package_root=str(cf_root),
            source_root=cf["source_collection"]["path"],
            files=cf_files,
        ),
        cascade=dict(
            package_root=str(Path(ca["unpacked_root"]) / "src/cascade"),
            source_root=ca["unpacked_root"],
            files=ca["python_toml_assets"],
        ),
        manifest=entry,
    )
    for root in (Path(cf["source_collection"]["path"]), cf_root.parent):
        require(
            {name: digest(root / name) for name in cf_files} == cf_files,
            "frozen Crazyflow source/assets",
        )
    require(
        {
            name: digest(Path(ca["unpacked_root"]) / name)
            for name in ca["python_toml_assets"]
        }
        == ca["python_toml_assets"],
        "frozen Cascade source/assets",
    )
    return result


def create_binding(
    output, *, protocol_path=PROTOCOL_PATH, protocol_sha256=PROTOCOL_SHA256
):
    p = read_protocol(protocol_path, protocol_sha256)
    old = read(anchor(p["imported_evidence"]["updated_binding"]))
    source = p["imported_evidence"]["updated_source"]
    require(
        source["root"] == old["public_root"]
        and source["commit"] == old["implementation_commit"],
        "retained binding source",
    )
    result = dict(
        format="glassbox-shared-vehicle-binding-v1",
        protocol_sha256=protocol_sha256,
        current=_source_identity(ROOT),
        baseline=_source_identity(
            source["root"], source["commit"], old["public_source_sha256"]
        ),
        oracle=_source_identity(
            old["oracle_root"], old["oracle_commit"], old["oracle_source_sha256"]
        ),
        runtime=_runtime(),
        prior_binding=p["imported_evidence"]["updated_binding"],
        inputs={
            "protocol": dict(
                path=str(Path(protocol_path).resolve()), sha256=protocol_sha256
            )
        },
        interpreter=str(Path(sys.executable).resolve()),
        implementation_commit=_git(ROOT, "rev-parse", "HEAD"),
        oracle_root=old["oracle_root"],
    )
    result["simulator_sources"] = simulator_identity(p)
    result["dart_root"] = "/Users/ryland/autonomy/dart"
    result["dart_source_sha256"] = {
        str(path.relative_to(result["dart_root"])): digest(path)
        for path in sorted((Path(result["dart_root"]) / "src/crazydart").rglob("*.py"))
    }
    write(output, result)
    return result


def authenticate(protocol_path, protocol_sha256, binding_path, binding_sha256):
    p = read_protocol(protocol_path, protocol_sha256)
    require(digest(binding_path) == binding_sha256, "external binding SHA")
    binding = read(binding_path)
    require(
        binding["format"] == "glassbox-shared-vehicle-binding-v1"
        and binding["protocol_sha256"] == protocol_sha256,
        "binding identity",
    )
    require(binding["runtime"] == _runtime(), "bound runtime")
    require(
        binding["simulator_sources"] == simulator_identity(p), "bound simulator sources"
    )
    require(
        binding["dart_source_sha256"]
        == {
            str(path.relative_to(binding["dart_root"])): digest(path)
            for path in sorted(
                (Path(binding["dart_root"]) / "src/crazydart").rglob("*.py")
            )
        },
        "bound Dart sources",
    )
    require(Path(binding["current"]["root"]) == ROOT, "executing harness root")
    require(
        binding["prior_binding"] == p["imported_evidence"]["updated_binding"],
        "retained binding link",
    )
    old = read(anchor(binding["prior_binding"]))
    require(
        binding["baseline"]["root"] == old["public_root"]
        and binding["baseline"]["commit"] == old["implementation_commit"]
        and binding["baseline"]["files"] == old["public_source_sha256"],
        "baseline source binding",
    )
    require(
        binding["oracle"]["root"] == old["oracle_root"]
        and binding["oracle"]["commit"] == old["oracle_commit"]
        and binding["oracle"]["files"] == old["oracle_source_sha256"],
        "native source binding",
    )
    for key in ("current", "baseline", "oracle"):
        entry = binding[key]
        require(
            _source_identity(entry["root"], entry["commit"], entry["files"]) == entry,
            "source identity",
        )
    require(
        binding["implementation_commit"] == binding["current"]["commit"]
        and binding["oracle_root"] == binding["oracle"]["root"]
        and binding["interpreter"] == binding["runtime"]["interpreter"],
        "binding mirrors",
    )
    return p, binding


def _prior_module():
    return _source_module(
        Path(__file__).with_name("independent_training_data.py"), "_svp_prior_data"
    )


def _array_count(directory):
    return _prior_module()._array_count(directory)


def _confirmation_queries(records, p, config):
    return _prior_module()._confirmation_queries(records, p, config)


def candidate_prerequisites(anchors, *, binding_sha256):
    """Only completed fits or preserved numerical failures permit confirmation."""
    require(set(anchors) == set(SIMULATORS), "both candidate outcomes required")
    result = {}
    for simulator in SIMULATORS:
        entry = anchors[simulator]
        path = Path(entry["path"])
        require(path.is_absolute() and path.name == "run.json", "candidate run path")
        require(digest(path) == entry["sha256"], "external candidate outcome SHA")
        outcome = read(path)
        require(
            outcome["format"] == CANDIDATE_FORMAT and outcome["simulator"] == simulator,
            "candidate outcome identity",
        )
        require(
            outcome["protocol_sha256"] == PROTOCOL_SHA256
            and outcome["binding_sha256"] == binding_sha256,
            "candidate outcome source association",
        )
        require(
            outcome["status"] in ("complete", "fit_failed")
            and entry["status"] == outcome["status"],
            "candidate outcome must be complete or fit_failed",
        )
        require(
            inventory(path.parent) == outcome["files"], "candidate payload inventory"
        )
        result[simulator] = dict(
            path=str(path.resolve()), sha256=entry["sha256"], status=outcome["status"]
        )
    return result


def verify_data(
    directory,
    expected_sha256,
    *,
    simulator=None,
    protocol=None,
    binding_sha256=None,
    candidate_outcomes=None,
):
    """Verify saved collection identities; native truth is checked by replay."""
    directory = Path(directory).resolve()
    require(digest(directory / "seal.json") == expected_sha256, "external data seal")
    seal = read(directory / "seal.json")
    require(
        seal["format"] == FORMAT and seal["protocol_sha256"] == PROTOCOL_SHA256,
        "data format/protocol",
    )
    require(seal["simulator"] in SIMULATORS, "data simulator")
    require(simulator is None or simulator == seal["simulator"], "requested simulator")
    require(
        binding_sha256 is None or seal["binding_sha256"] == binding_sha256,
        "data implementation association",
    )
    require(inventory(directory, "seal.json") == seal["files"], "data inventory")
    require(seal["arrays"] == _array_count(directory), "data array count")
    protocol = read_protocol() if protocol is None else protocol
    p = physical_view(protocol)
    require(read(directory / "resolved-physical.json") == p, "resolved physical view")
    entries = [r for r in p["recordings"] if r["simulator"] == seal["simulator"]]
    require(
        len(entries) == 84 and read(directory / "planned-records.json") == entries,
        "literal confirmation roster",
    )
    records = read(directory / "records.json")
    require(len(records) == seal["parents"] == 84, "all planned parent outcomes")
    cells = {c["id"]: c for c in p["cells"][seal["simulator"]]}
    for row, entry in zip(records, entries, strict=True):
        require(
            {k: row[k] for k in entry} == entry
            and row["cell_facts"] == cells[entry["cell"]]
            and row["prefix"] == str(Path("parents") / entry["id"]),
            "ordered parent identity/cell/path",
        )
        require(read(directory / (row["prefix"] + ".json")) == row, "parent mirror")
    config = read(directory / "configuration.json")
    require(
        config["dt_s"] == p["generation"][seal["simulator"]]["dt_s"],
        "native confirmation timing",
    )
    queries = read(directory / "queries.json")
    require(
        queries == _confirmation_queries(records, p, config)
        and len(queries) == seal["queries"],
        "ordered complete query roster",
    )
    anchors = seal["candidate_outcomes"]
    require(
        anchors
        == candidate_prerequisites(anchors, binding_sha256=seal["binding_sha256"]),
        "confirmation candidate associations",
    )
    if candidate_outcomes is not None:
        require(anchors == candidate_outcomes, "requested candidate associations")
    require(
        seal["training_collected"] == seal["fits"] == 0
        and seal["test_seed_shift_from_original"] == 14000000,
        "confirmation-only scope",
    )
    return seal


def _worker_identity(request):
    protocol, binding = authenticate(
        request["protocol_path"],
        request["protocol_sha256"],
        request["binding_path"],
        request["binding_sha256"],
    )
    require(
        digest(__file__) == binding["current"]["files"][MODULE], "actual worker file"
    )
    import glassbox

    require(
        Path(glassbox.__file__).resolve()
        == Path(binding["oracle_root"]) / "src/glassbox/__init__.py",
        "historical native imports",
    )
    import jax

    require(jax.config.x64_enabled and jax.default_backend() == "cpu", "native x64 CPU")
    return protocol, binding


def _worker(request_path, expected_request_sha256):
    require(digest(request_path) == expected_request_sha256, "external worker request")
    request = read(request_path)
    protocol, binding = _worker_identity(request)
    require(request["simulator"] in SIMULATORS, "worker simulator")
    require(request["stage"] in ("collect", "replay"), "worker stage")
    from glassbox.experimental import expanded_training_cache_experiment as historical

    runtime = historical.check_sources()
    p, old = physical_view(protocol), historical.protocol()
    require(p["cells"] == old["cells"], "unchanged fixture cells")
    for simulator in SIMULATORS:
        require(
            p["generation"][simulator] == old["generation"][simulator],
            "unchanged fixture",
        )
    prerequisites = candidate_prerequisites(
        request["candidate_outcomes"], binding_sha256=request["binding_sha256"]
    )
    flight = historical.flight()
    fixture = flight.fixture_for(request["simulator"], p)
    entries = [r for r in p["recordings"] if r["simulator"] == request["simulator"]]
    directory = Path(request["directory"])
    replaying = request["stage"] == "replay"
    if replaying:
        old_seal = verify_data(
            directory,
            request["expected_data_sha256"],
            simulator=request["simulator"],
            protocol=protocol,
            binding_sha256=request["binding_sha256"],
            candidate_outcomes=prerequisites,
        )
    else:
        directory.mkdir(parents=True, exist_ok=False)
    arrays_count = 0

    def json_payload(relative, value):
        path = directory / relative
        if replaying:
            require(read(path) == value, "native JSON replay: " + relative)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            write(path, value)

    def array_payload(relative, values):
        nonlocal arrays_count
        path = directory / relative
        if replaying:
            flight.array_equal(values, flight.load_arrays(path), relative)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            flight.save_arrays(path, values)
        arrays_count += len(values)

    json_payload("resolved-physical.json", p)
    json_payload("planned-records.json", entries)
    json_payload(
        "configuration.json", dict(fixture.config(), spec=fixture.spec.to_dict())
    )
    cells = {c["id"]: c for c in p["cells"][request["simulator"]]}
    records, queries = [], []
    for entry in entries:
        print(("replay " if replaying else "collect ") + entry["id"], flush=True)
        cell = cells[entry["cell"]]
        try:
            values, metadata = fixture.generate(entry, cell)
        except Exception as error:
            if type(error).__name__ != "FixtureSetupError":
                raise
            values, metadata = (
                None,
                dict(setup_failure=str(error), exception=type(error).__name__),
            )
        prefix = str(Path("parents") / entry["id"])
        record = dict(entry, prefix=prefix, cell_facts=cell, metadata=metadata)
        if values is None:
            record["validity"] = dict(
                valid_transitions=0,
                valid_initial=False,
                failure_index=0,
                reason="setup_failure",
            )
        else:
            array_payload(prefix + ".npz", values)
            record.update(
                validity=flight.validity(values), achieved=flight.achieved(values)
            )
        json_payload(prefix + ".json", record)
        records.append(record)
        rows, query_arrays, branches = flight.make_queries(
            fixture, values, entry, cell, p
        )
        for row in rows:
            row["scope"] = cell["group"]
            row["path"] = str(Path("queries") / entry["id"] / (row["id"] + ".npz"))
            array_payload(row["path"], query_arrays[row["id"]])
        for name, (branch_arrays, detail) in branches.items():
            prefix = str(Path("branches") / entry["id"] / name)
            array_payload(prefix + ".npz", branch_arrays)
            json_payload(prefix + ".json", detail)
        queries.extend(rows)
    json_payload("records.json", records)
    json_payload("queries.json", queries)
    result = dict(
        format=FORMAT,
        protocol_sha256=PROTOCOL_SHA256,
        binding_sha256=request["binding_sha256"],
        simulator=request["simulator"],
        test_seed_shift_from_original=14000000,
        parents=len(records),
        queries=len(queries),
        arrays=arrays_count,
        candidate_outcomes=prerequisites,
        historical_runtime=runtime,
        imported_sources=binding["oracle"],
        training_collected=0,
        fits=0,
        files=inventory(directory, "seal.json"),
    )
    _worker_identity(request)
    require(
        candidate_prerequisites(
            request["candidate_outcomes"], binding_sha256=request["binding_sha256"]
        )
        == prerequisites,
        "candidate anchors unchanged during collection",
    )
    if replaying:
        require(old_seal == result, "complete native replay evidence")
        write(
            request["result_path"],
            dict(
                exact=True,
                parents=len(records),
                queries=len(queries),
                arrays=arrays_count,
                fits=0,
                training_collected=0,
                data_sha256=request["expected_data_sha256"],
            ),
        )
    else:
        write(directory / "seal.json", result)
    return result


def collect_confirmation(
    simulator,
    output,
    *,
    candidate_outcomes,
    protocol_path=PROTOCOL_PATH,
    protocol_sha256=PROTOCOL_SHA256,
    binding_path,
    binding_sha256,
    replay_data=None,
    expected_data_sha256=None,
):
    """One exclusive bounded collect/replay attempt; never retry a failed worker."""
    require(simulator in SIMULATORS, "confirmation simulator")
    protocol, binding = authenticate(
        protocol_path, protocol_sha256, binding_path, binding_sha256
    )
    prerequisites = candidate_prerequisites(
        candidate_outcomes, binding_sha256=binding_sha256
    )
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = dict(
        format=STAGE_FORMAT,
        simulator=simulator,
        status="incomplete",
        protocol_sha256=protocol_sha256,
        binding_sha256=binding_sha256,
        fits=0,
        training_collected=0,
    )
    try:
        require(
            (replay_data is None) == (expected_data_sha256 is None),
            "replay requires a data path and external SHA together",
        )
        if replay_data is not None:
            verify_data(
                replay_data,
                expected_data_sha256,
                simulator=simulator,
                protocol=protocol,
                binding_sha256=binding_sha256,
                candidate_outcomes=prerequisites,
            )
        request = dict(
            simulator=simulator,
            stage="replay" if replay_data is not None else "collect",
            protocol_path=str(Path(protocol_path).resolve()),
            protocol_sha256=protocol_sha256,
            binding_path=str(Path(binding_path).resolve()),
            binding_sha256=binding_sha256,
            candidate_outcomes=prerequisites,
            directory=str(
                Path(replay_data).resolve()
                if replay_data is not None
                else output / "data"
            ),
            expected_data_sha256=expected_data_sha256,
            result_path=str(output / "replay.json"),
        )
        write(output / "request.json", request)
        result["request_sha256"] = digest(output / "request.json")
        environment = dict(os.environ)
        environment.update(
            PYTHONPATH=os.pathsep.join(
                (
                    str(Path(binding["oracle_root"]) / "src"),
                    "/private/tmp/glassbox-cascade-e8f6ba6/src",
                )
            ),
            JAX_ENABLE_X64="1",
            SCIPY_ARRAY_API="1",
        )
        argv = [
            binding["interpreter"],
            str(Path(__file__).resolve()),
            "--worker",
            str(output / "request.json"),
            "--request-sha256",
            result["request_sha256"],
        ]
        write(
            output / "command.json",
            dict(
                argv=argv,
                cwd=str(ROOT),
                hard_timeout_s=14400,
                environment_overrides={
                    k: environment[k]
                    for k in ("PYTHONPATH", "JAX_ENABLE_X64", "SCIPY_ARRAY_API")
                },
            ),
        )
        with (output / "worker.log").open("x") as log:
            process = subprocess.run(
                argv,
                cwd=ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=14400,
                check=False,
            )
        result["returncode"] = process.returncode
        require(
            process.returncode == 0,
            "confirmation worker failed; retained without retry",
        )
        data_sha = (
            expected_data_sha256
            if replay_data is not None
            else digest(output / "data/seal.json")
        )
        verify_data(
            request["directory"],
            data_sha,
            simulator=simulator,
            protocol=protocol,
            binding_sha256=binding_sha256,
            candidate_outcomes=prerequisites,
        )
        authenticate(protocol_path, protocol_sha256, binding_path, binding_sha256)
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
        result["elapsed_s"] = time.monotonic() - started
        write(
            output / "exit.json",
            {
                k: result.get(k)
                for k in ("status", "returncode", "elapsed_s", "error_type", "error")
            },
        )
        result["files"] = inventory(output)
        write(output / "run.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", required=True)
    parser.add_argument("--request-sha256", required=True)
    args = parser.parse_args()
    _worker(args.worker, args.request_sha256)
