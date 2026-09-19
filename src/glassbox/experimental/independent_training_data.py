"""Frozen independent training/confirmation data in an isolated old simulator.

The standalone worker loads no current Glassbox package. Only literal rosters
and the unchanged excitation operator are supplied to the pinned simulator.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import time
import traceback
import zipfile
from collections import Counter
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_PATH = ROOT / "docs/harness/independent-training-recordings-v1.json"
PROTOCOL_SHA256 = "b5fae12bf2b25c88d7a1758214d1faf11749e7a88d8ce4d4e397b04265272adf"
SIMULATORS = ("crazyflow", "cascade")
FORMAT = "glassbox-independent-training-data-v1"
STAGE_FORMAT = "glassbox-independent-training-data-stage-v1"
MODULE = "src/glassbox/experimental/independent_training_data.py"


class PreparationUnavailable(ValueError):
    def __init__(self, support):
        self.support = support
        self.details = support
        super().__init__("no added parent has a usable valid prefix")


def require(condition, label):
    if not condition:
        raise ValueError(label)


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


def _array_count(directory):
    count = 0
    for path in Path(directory).rglob("*.npz"):
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
        require(
            len(names) == len(set(names))
            and all(name.endswith(".npy") for name in names),
            "NPZ array member roster",
        )
        count += len(names)
    return count


def _anchored(entry):
    path = Path(entry["path"])
    require(digest(path) == entry["sha256"], "external input SHA: " + str(path))
    return path


def read_protocol(path=PROTOCOL_PATH, expected_sha256=PROTOCOL_SHA256):
    require(expected_sha256 == PROTOCOL_SHA256, "frozen independent protocol anchor")
    path = Path(path).resolve()
    require(path == PROTOCOL_PATH, "protocol source association")
    require(digest(path) == expected_sha256, "external protocol SHA")
    protocol = read(path)
    require(protocol["id"] == "independent-training-recordings-v1", "protocol identity")
    for entry in protocol["dependencies"].values():
        require(digest(ROOT / entry["path"]) == entry["sha256"], "protocol dependency")
    for name, expected in protocol["verification"][
        "inherited_current_source_sha256"
    ].items():
        require(digest(ROOT / name) == expected, "unchanged inherited source: " + name)
    validate_roster(protocol)
    return protocol


def _dependency(protocol, name):
    entry = protocol["dependencies"][name]
    require(
        digest(ROOT / entry["path"]) == entry["sha256"], "protocol dependency: " + name
    )
    return read(ROOT / entry["path"])


def validate_roster(protocol):
    """Reconstruct every literal entry and all previously inspected exclusions."""
    original = _dependency(protocol, "original_flight")
    excitation = _dependency(protocol, "excitation")
    roster = _dependency(protocol, "roster")
    roles = excitation["planned_automatic_roles"]
    require(roster["original_roles"] == roles, "original training/development roles")
    indexed = {row["id"]: row for row in original["recordings"]}
    require(len(indexed) == len(original["recordings"]), "original unique parents")
    added, confirmation = [], []
    for simulator in SIMULATORS:
        role = roles[simulator]
        require(set(role) == {"training", "development"}, "exact original role names")
        require(
            len(role["training"]) == 72 and len(role["development"]) == 24,
            "original 72/24 roles",
        )
        require(
            len(set(role["training"] + role["development"])) == 96,
            "original disjoint roles",
        )
        for name in role["training"]:
            source = indexed[name]
            require(
                source["simulator"] == simulator
                and source["role"] == "calibration_pool",
                "original training identity",
            )
            row = deepcopy(source)
            row["seed"] += 20000000
            row["id"] = row["id"].rsplit("-", 1)[0] + "-" + str(row["seed"])
            row["source_condition_parent"] = name
            added.append(row)
        for source in original["recordings"]:
            if source["simulator"] != simulator or source["role"] != "test":
                continue
            row = deepcopy(source)
            row["seed"] += 12000000
            row["id"] = row["id"].rsplit("-", 1)[0] + "-" + str(row["seed"])
            confirmation.append(row)
        counts = dict(Counter(indexed[name]["cell"] for name in role["training"]))
        require(
            counts
            == roster["per_simulator_original_and_added_training_cell_counts"][
                simulator
            ],
            "realized training allocation",
        )
        require(
            sum(row["simulator"] == simulator for row in confirmation) == 84,
            "84 confirmation parents",
        )
    require(roster["added_training"] == added, "literal ordered added training roster")
    require(
        roster["confirmation"] == confirmation, "literal ordered confirmation roster"
    )
    require(
        roster["excluded_prior_test_shifts"] == list(range(0, 12000000, 1000000)),
        "all prior seed shifts",
    )
    fresh = added + confirmation
    ids = {row["id"] for row in fresh}
    seeds = {(row["simulator"], row["seed"]) for row in fresh}
    require(
        len(ids) == len(seeds) == len(fresh) == 312, "312 unique independent parents"
    )
    excluded_ids, excluded_seeds = (
        set(indexed),
        {(r["simulator"], r["seed"]) for r in original["recordings"]},
    )
    for row in original["recordings"]:
        if row["role"] == "test":
            for shift in roster["excluded_prior_test_shifts"]:
                seed = row["seed"] + shift
                excluded_seeds.add((row["simulator"], seed))
                excluded_ids.add(row["id"].rsplit("-", 1)[0] + "-" + str(seed))
    require(
        not ids.intersection(excluded_ids) and not seeds.intersection(excluded_seeds),
        "new/prior data overlap",
    )
    return roster


def _imported_evidence(protocol):
    evidence = protocol["imported_evidence"]
    for simulator in SIMULATORS:
        for value in evidence["controls"][simulator].values():
            if isinstance(value, dict) and "path" in value:
                _anchored(value)
    for key, flag in (
        (
            "corrected_physical_consumer_replay",
            "physical_and_consumer_regression_passed",
        ),
        ("corrected_synthetic_lifecycle_replay", "inference_regression_passed"),
    ):
        report = read(_anchored(evidence[key]))
        require(
            report["status"] == "complete" and report[flag] is True,
            "completed corrected inference evidence",
        )
    _anchored(evidence["original_fit_binding"])


def authenticate(protocol_path, protocol_sha256, binding_path, binding_sha256):
    """Current-process preflight; never call this inside the old-source worker."""
    from .public_mean_implementation import verify

    protocol = read_protocol(protocol_path, protocol_sha256)
    binding = verify(binding_path, binding_sha256)
    require(Path(binding["public_root"]).resolve() == ROOT, "current data source root")
    require(
        binding["public_source_sha256"][MODULE] == digest(__file__),
        "data driver source",
    )
    _imported_evidence(protocol)
    return protocol, binding


def candidate_prerequisites(anchors, *, binding_sha256=None):
    require(
        set(anchors) == set(SIMULATORS),
        "both candidate outcomes required before confirmation",
    )
    result = {}
    for simulator in SIMULATORS:
        path = _anchored(anchors[simulator])
        outcome = read(path)
        require(
            outcome["format"] == "glassbox-independent-training-candidate-v1"
            and outcome["simulator"] == simulator,
            "candidate outcome identity",
        )
        require(
            outcome["protocol_sha256"] == PROTOCOL_SHA256, "candidate outcome protocol"
        )
        require(
            binding_sha256 is None
            or outcome["implementation_sha256"] == binding_sha256,
            "candidate outcome implementation",
        )
        require(
            outcome["status"]
            in (
                "complete",
                "preparation_unavailable",
                "fit_failed",
                "hard_timeout_incomplete",
            ),
            "terminal candidate outcome",
        )
        require(
            outcome["files"] == inventory(path.parent),
            "candidate outcome payload inventory",
        )
        result[simulator] = {
            "path": str(path.resolve()),
            "sha256": anchors[simulator]["sha256"],
            "status": outcome["status"],
        }
    return result


def physical_view(protocol, kind):
    require(kind in ("training", "confirmation"), "collection kind")
    p = deepcopy(_dependency(protocol, "original_flight"))
    roster = validate_roster(protocol)
    p["recordings"] = deepcopy(
        roster["added_training" if kind == "training" else "confirmation"]
    )
    if kind == "training":
        p["collection_intervention"] = deepcopy(
            _dependency(protocol, "excitation")["collection_intervention"]
        )
        p["planned_automatic_roles"] = {
            simulator: {
                "training": [
                    row["id"]
                    for row in p["recordings"]
                    if row["simulator"] == simulator
                ],
                "development": [],
            }
            for simulator in SIMULATORS
        }
        p["counts"] = {
            "excited_training_parents_per_simulator": 72,
            "unchanged_development_copies_per_simulator": 0,
        }
    return p


def support(records, dt_s):
    minimum = round(0.75 / dt_s)
    admitted, unavailable = [], []
    for row in records:
        validity = row["validity"]
        count = validity["valid_transitions"]
        require(type(count) is int and count >= 0, "valid transition count")
        if validity["valid_initial"] and count >= minimum:
            admitted.append(row["id"])
        else:
            unavailable.append(
                {
                    "id": row["id"],
                    "valid_transitions": count,
                    "required_transitions": minimum,
                    "valid_initial": validity["valid_initial"],
                    "failure_reason": validity.get("reason"),
                    "setup_failure": row["metadata"].get("setup_failure"),
                }
            )
    return {
        "planned": [row["id"] for row in records],
        "admitted": admitted,
        "unavailable": unavailable,
        "minimum_transitions": minimum,
        "all_planned_usable": not unavailable,
    }


def _load_npz(path):
    import numpy as np

    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _same(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def _recording_contract(configuration):
    from glassbox.experimental.two_simulator_flight import OBSERVED_CHANNELS

    spec = configuration["spec"]
    return dict(
        configuration_id=spec["vehicle"]["configuration_id"],
        state_channels=list(OBSERVED_CHANNELS),
        input_channels=[
            json.dumps(c, sort_keys=True)
            for c in spec["channels"]
            if c["kind"] == "control"
        ],
        dt_s=float(configuration["dt_s"]),
    )


def _recording_values(raw, transitions):
    import jax
    import numpy as np

    from glassbox.experimental.two_simulator_flight import OBSERVE

    # Telemetry projection only: never initialize or forecast a learned model.
    with jax.enable_x64(True):
        states = np.asarray(OBSERVE(raw["states"][: transitions + 1]))
    return {"states": states, "inputs": raw["commands"][:transitions]}


def _excitation_check(arrays, record, rule, configuration):
    """Arithmetic-only check of actual issued commands and PCG64 assignment."""
    import numpy as np

    detail = record["metadata"]["command_excitation"]
    lower, upper = np.asarray(detail["lower"]), np.asarray(detail["upper"])
    controls = [c for c in configuration["spec"]["channels"] if c["kind"] == "control"]
    require(len(controls) == len(lower), "excitation control width")
    require(
        np.array_equal(lower, [c["minimum"] for c in controls])
        and np.array_equal(upper, [c["maximum"] for c in controls]),
        "excitation fixture bounds",
    )
    require(detail["dt_s"] == configuration["dt_s"], "excitation fixture timing")
    # The actual schedule metadata is bound to the fixture in fresh replay;
    # the deterministic assignment/issued-command algebra is checked here too.
    material = json.dumps(
        {"namespace": rule["id"], "parent": record["id"], "seed": record["seed"]},
        sort_keys=True,
        separators=(",", ":"),
    )
    seed_digest = hashlib.sha256(material.encode()).digest()
    require(
        detail["seed_material"] == material
        and detail["seed_sha256"] == seed_digest.hex()
        and detail["seed_integer"] == int.from_bytes(seed_digest, "big"),
        "excitation seed identity",
    )
    dt = detail["dt_s"]
    steps = len(arrays["commands"])
    start, block = round(rule["start_s"] / dt), round(rule["block_s"] / dt)
    require(
        detail["steps"] == steps
        and detail["start_index"] == start
        and detail["block_steps"] == block
        and detail["blocks"] == 23,
        "excitation schedule dimensions",
    )
    require(
        detail["start_s"] == rule["start_s"]
        and detail["block_s"] == rule["block_s"]
        and detail["range_fraction"] == rule["range_fraction"],
        "excitation amplitude/timing",
    )
    signs = (
        2
        * np.random.Generator(
            np.random.PCG64(int.from_bytes(seed_digest, "big"))
        ).integers(0, 2, size=(23, len(lower)), dtype=np.int64)
        - 1
    )
    require(np.array_equal(signs, detail["signs"]), "excitation deterministic signs")
    indices = np.full(steps, -1, dtype=np.int64)
    indices[start:] = np.arange(steps - start, dtype=np.int64) // block
    assigned = np.zeros((steps, len(lower)), dtype=np.float64)
    assigned[start:] = rule["range_fraction"] * (upper - lower) * signs[indices[start:]]
    pilot = arrays["excitation_pilot_commands"]
    issued = pilot.copy()
    issued[start:] = np.clip(pilot[start:] + assigned[start:], lower, upper)
    clipped = (pilot + assigned < lower) | (pilot + assigned > upper)
    clipped[:start] = False
    expected = {
        "commands": issued,
        "excitation_block_index": indices,
        "excitation_assigned_delta": assigned,
        "excitation_realized_delta": issued - pilot,
        "excitation_clipped": clipped,
        "excitation_clipping_residual": issued - (pilot + assigned),
    }
    for key, value in expected.items():
        require(_same(arrays[key], value), "excitation issued-command algebra: " + key)


def _confirmation_queries(records, p, configuration):
    from glassbox.experimental.two_simulator_flight import query_roster

    width = sum(
        channel["kind"] == "control" for channel in configuration["spec"]["channels"]
    )
    dt = configuration["dt_s"]
    cells = {cell["id"]: cell for cell in p["cells"][records[0]["simulator"]]}
    result = []
    for row in records:
        for query in query_roster(row, p, dt, width):
            query["history_eligible"] = bool(
                row["validity"]["valid_initial"]
                and query["origin"] <= row["validity"]["valid_transitions"]
            )
            query["scope"] = cells[row["cell"]]["group"]
            query["path"] = str(Path("queries") / row["id"] / (query["id"] + ".npz"))
            result.append(query)
    return result


def verify_data(
    directory, expected_sha256, *, kind=None, simulator=None, protocol=None
):
    directory = Path(directory).resolve()
    require(digest(directory / "seal.json") == expected_sha256, "external data seal")
    seal = read(directory / "seal.json")
    require(
        seal["format"] == FORMAT and seal["protocol_sha256"] == PROTOCOL_SHA256,
        "data format/protocol",
    )
    require(
        seal["kind"] in ("training", "confirmation")
        and seal["simulator"] in SIMULATORS,
        "data kind/simulator",
    )
    require(kind is None or seal["kind"] == kind, "requested data kind")
    require(
        simulator is None or seal["simulator"] == simulator, "requested data simulator"
    )
    require(
        inventory(directory, "seal.json") == seal["files"], "data payload inventory"
    )
    require(seal["arrays"] == _array_count(directory), "data array count")
    protocol = read_protocol() if protocol is None else protocol
    p = physical_view(protocol, seal["kind"])
    require(
        read(directory / "resolved-physical.json") == p, "frozen resolved physical view"
    )
    planned = [row for row in p["recordings"] if row["simulator"] == seal["simulator"]]
    require(read(directory / "planned-records.json") == planned, "literal data roster")
    records = read(directory / "records.json")
    require(len(records) == len(planned), "complete planned data outcomes")
    for row, entry in zip(records, planned, strict=True):
        require({key: row[key] for key in entry} == entry, "ordered record identity")
        require(
            row["prefix"] == str(Path("parents") / entry["id"]), "record archive path"
        )
        require(
            read(directory / (row["prefix"] + ".json")) == row, "record detail mirror"
        )
    require(seal["parents"] == len(records), "data parent count")
    queries = read(directory / "queries.json")
    require(seal["queries"] == len(queries), "data query count")
    if seal["kind"] == "training":
        require(
            not queries and not seal["candidate_outcomes"],
            "training precedes confirmation",
        )
        contract = read(directory / "recording-contract.json")
        require(
            seal["support"] == support(records, contract["dt_s"]),
            "training support evidence",
        )
        configuration = read(directory / "configuration.json")
        require(
            contract == _recording_contract(configuration),
            "recording contract derives from fixture",
        )
        from glassbox.experimental.two_simulator_flight import validity

        for row in records:
            raw_path = directory / (row["prefix"] + ".npz")
            if not raw_path.exists():
                require(
                    row["validity"]["reason"] == "setup_failure"
                    and not row["validity"]["valid_initial"],
                    "absent raw trajectory reason",
                )
                continue
            raw = _load_npz(raw_path)
            require(row["validity"] == validity(raw), "raw trajectory validity")
            require(
                float(raw["time_s"][1] - raw["time_s"][0]) == contract["dt_s"],
                "raw recording timing",
            )
            _excitation_check(raw, row, p["collection_intervention"], configuration)
            if row["id"] in seal["support"]["admitted"]:
                observations = _load_npz(
                    directory / ("recordings/" + row["id"] + ".npz")
                )
                n = row["validity"]["valid_transitions"]
                require(
                    set(observations) == {"states", "inputs"},
                    "ordinary learner channel roster",
                )
                require(
                    observations["states"].shape == (n + 1, 15),
                    "ordinary observation shape",
                )
                expected = _recording_values(raw, n)
                require(
                    _same(observations["states"], expected["states"]),
                    "recordings use actual observed states",
                )
                require(
                    _same(observations["inputs"], expected["inputs"]),
                    "recordings use actual issued commands",
                )
    else:
        require(
            seal["support"] is None, "confirmation support is descriptive truth only"
        )
        require(
            set(seal["candidate_outcomes"]) == set(SIMULATORS),
            "confirmation candidate outcome associations",
        )
        require(
            queries
            == _confirmation_queries(
                records, p, read(directory / "configuration.json")
            ),
            "complete ordered confirmation query roster",
        )
    return seal


def load_added_recordings(directory, expected_sha256, contract=None):
    from glassbox.recordings import SequenceCollection, SequenceSegment

    directory = Path(directory)
    seal = verify_data(directory, expected_sha256, kind="training")
    actual = read(directory / "recording-contract.json")
    require(
        contract is None or actual == contract, "added recording signal/timing contract"
    )
    if not seal["support"]["admitted"]:
        raise PreparationUnavailable(seal["support"])
    segments = []
    for name in seal["support"]["admitted"]:
        values = _load_npz(directory / ("recordings/" + name + ".npz"))
        segments.append(
            SequenceSegment(
                name, "valid-prefix", values["states"], values["inputs"], actual["dt_s"]
            )
        )
    return SequenceCollection(
        tuple(segments),
        configuration_id=actual["configuration_id"],
        state_channels=tuple(actual["state_channels"]),
        input_channels=tuple(actual["input_channels"]),
    )


def _source_module(path):
    spec = importlib.util.spec_from_file_location(
        "_independent_training_physics_source", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _worker_identity(request):
    protocol = read_protocol(request["protocol_path"], request["protocol_sha256"])
    binding = read(
        _anchored(
            {"path": request["binding_path"], "sha256": request["binding_sha256"]}
        )
    )
    root = Path(binding["public_root"]).resolve()
    require(
        root == ROOT and digest(__file__) == binding["public_source_sha256"][MODULE],
        "standalone worker source association",
    )
    require(
        not subprocess.check_output(
            [
                "git",
                "-C",
                str(root),
                "status",
                "--porcelain",
                "--untracked-files=normal",
            ],
            text=True,
        ).strip(),
        "current worker source checkout clean",
    )
    require(
        subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
        == binding["implementation_commit"],
        "current worker source commit",
    )
    for relative, expected in binding["public_source_sha256"].items():
        require(digest(root / relative) == expected, "current source pin: " + relative)
    physics = _source_module(root / "src/glassbox/experimental/public_mean_physics.py")
    old_binding, _ = physics._worker_identity(request)
    require(old_binding == binding, "worker binding agreement")
    _imported_evidence(protocol)
    return protocol, binding, physics


def _worker(request_path, expected_request_sha256):
    require(digest(request_path) == expected_request_sha256, "external worker request")
    request = read(request_path)
    protocol, binding, physics = _worker_identity(request)
    from glassbox.experimental import command_excitation_data as excitation
    from glassbox.experimental import expanded_training_cache_experiment as historical

    historical_runtime = historical.check_sources()
    p = physical_view(protocol, request["kind"])
    inherited = historical.protocol()
    require(p["cells"] == inherited["cells"], "unchanged fixture cells")
    for simulator in SIMULATORS:
        require(
            p["generation"][simulator] == inherited["generation"][simulator],
            "unchanged fixture generation",
        )
    entries = [
        row for row in p["recordings"] if row["simulator"] == request["simulator"]
    ]
    prerequisites = (
        candidate_prerequisites(
            request["candidate_outcomes"], binding_sha256=request["binding_sha256"]
        )
        if request["kind"] == "confirmation"
        else {}
    )
    require(
        not request["candidate_outcomes"] or request["kind"] == "confirmation",
        "no training candidate prerequisites",
    )
    flight = historical.flight()
    fixture = flight.fixture_for(request["simulator"], p)
    directory = Path(request["directory"])
    replaying = request["stage"] == "replay"
    if replaying:
        old = verify_data(
            directory,
            request["expected_data_sha256"],
            kind=request["kind"],
            simulator=request["simulator"],
            protocol=protocol,
        )
    else:
        require(request["stage"] == "collect", "data worker stage")
        directory.mkdir(parents=True, exist_ok=False)
    array_count = 0

    def json_payload(relative, value):
        path = directory / relative
        if replaying:
            require(read(path) == value, "data JSON replay: " + relative)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            write(path, value)

    def array_payload(relative, values):
        nonlocal array_count
        path = directory / relative
        if replaying:
            flight.array_equal(values, flight.load_arrays(path), relative)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            flight.save_arrays(path, values)
        array_count += len(values)

    json_payload("resolved-physical.json", p)
    json_payload("planned-records.json", entries)
    json_payload(
        "configuration.json", dict(fixture.config(), spec=fixture.spec.to_dict())
    )
    contract = dict(
        configuration_id=fixture.spec.vehicle.configuration_id,
        state_channels=list(flight.OBSERVED_CHANNELS),
        input_channels=[
            json.dumps(c.to_dict(), sort_keys=True) for c in fixture.spec.controls
        ],
        dt_s=float(fixture.dt),
    )
    if request["kind"] == "training":
        json_payload("recording-contract.json", contract)
    cells = {cell["id"]: cell for cell in p["cells"][request["simulator"]]}
    records, all_queries = [], []
    for entry in entries:
        print(("replay " if replaying else "collect ") + entry["id"], flush=True)
        cell = cells[entry["cell"]]
        try:
            arrays, metadata = (
                excitation.excite(fixture, entry, cell, p)
                if request["kind"] == "training"
                else fixture.generate(entry, cell)
            )
        except Exception as error:
            if type(error).__name__ != "FixtureSetupError":
                raise
            arrays, metadata = (
                None,
                {"setup_failure": str(error), "exception": type(error).__name__},
            )
        prefix = str(Path("parents") / entry["id"])
        record = dict(entry, cell_facts=cell, prefix=prefix, metadata=metadata)
        if arrays is None:
            record["validity"] = dict(
                valid_transitions=0,
                valid_initial=False,
                failure_index=0,
                reason="setup_failure",
            )
        else:
            array_payload(prefix + ".npz", arrays)
            record.update(
                validity=flight.validity(arrays), achieved=flight.achieved(arrays)
            )
        json_payload(prefix + ".json", record)
        records.append(record)
        if request["kind"] == "training":
            if support([record], fixture.dt)["admitted"]:
                n = record["validity"]["valid_transitions"]
                array_payload(
                    "recordings/" + entry["id"] + ".npz",
                    {
                        "states": flight.np.asarray(
                            flight.OBSERVE(arrays["states"][: n + 1])
                        ),
                        "inputs": arrays["commands"][:n],
                    },
                )
            continue
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
    result = dict(
        format=FORMAT,
        protocol_sha256=request["protocol_sha256"],
        implementation_sha256=request["binding_sha256"],
        historical_runtime=historical_runtime,
        kind=request["kind"],
        simulator=request["simulator"],
        parents=len(records),
        queries=len(all_queries),
        arrays=array_count,
        candidate_outcomes=prerequisites,
        support=support(records, fixture.dt) if request["kind"] == "training" else None,
        imported_sources=physics._imports(binding),
        fits=0,
        original_parents_regenerated=0,
        files=inventory(directory, "seal.json"),
    )
    _worker_identity(request)
    if replaying:
        require(old == result, "complete data replay evidence")
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


def _run(
    simulator,
    output,
    *,
    kind,
    protocol_path=PROTOCOL_PATH,
    protocol_sha256=PROTOCOL_SHA256,
    binding_path,
    binding_sha256,
    candidate_outcomes=None,
    replay_data=None,
    expected_data_sha256=None,
):
    require(
        simulator in SIMULATORS and kind in ("training", "confirmation"),
        "collection simulator/kind",
    )
    protocol, binding = authenticate(
        protocol_path, protocol_sha256, binding_path, binding_sha256
    )
    prerequisites = candidate_outcomes or {}
    if kind == "confirmation":
        candidate_prerequisites(prerequisites, binding_sha256=binding_sha256)
    else:
        require(not prerequisites, "training precedes candidate outcomes")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    result = dict(
        format=STAGE_FORMAT,
        simulator=simulator,
        kind=kind,
        status="incomplete",
        protocol_sha256=protocol_sha256,
        implementation_sha256=binding_sha256,
        fits=0,
    )
    started = time.monotonic()
    try:
        if replay_data is not None:
            require(expected_data_sha256 is not None, "data replay external anchor")
            verify_data(
                replay_data,
                expected_data_sha256,
                kind=kind,
                simulator=simulator,
                protocol=protocol,
            )
        request = dict(
            simulator=simulator,
            kind=kind,
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
        command = [
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
                argv=command,
                cwd=str(ROOT),
                hard_timeout_s=14400,
                environment_overrides={
                    key: environment[key]
                    for key in ("PYTHONPATH", "JAX_ENABLE_X64", "SCIPY_ARRAY_API")
                },
            ),
        )
        with (output / "worker.log").open("x") as log:
            process = subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=14400,
                check=False,
            )
        result["returncode"] = process.returncode
        require(process.returncode == 0, "data worker failed; retained without retry")
        data_sha = (
            expected_data_sha256
            if replay_data is not None
            else digest(output / "data/seal.json")
        )
        verify_data(
            request["directory"],
            data_sha,
            kind=kind,
            simulator=simulator,
            protocol=protocol,
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
                key: result.get(key)
                for key in ("status", "returncode", "elapsed_s", "error_type", "error")
            },
        )
        result["files"] = inventory(output)
        write(output / "run.json", result)
    return result


def collect_training(simulator, output, **kwargs):
    return _run(simulator, output, kind="training", **kwargs)


def collect_confirmation(simulator, output, *, candidate_outcomes, **kwargs):
    return _run(
        simulator,
        output,
        kind="confirmation",
        candidate_outcomes=candidate_outcomes,
        **kwargs,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", required=True)
    parser.add_argument("--request-sha256", required=True)
    args = parser.parse_args()
    _worker(args.worker, args.request_sha256)
