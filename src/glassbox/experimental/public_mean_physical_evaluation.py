"""Frozen public-v4 physical predictions, raw-array replay and promotion policy.

This file doubles as a standalone inference worker. It imports no Glassbox/JAX
module before authenticating the requested public or historical source role.
There is no fitting, initialization or simulator execution here. A read-only
calibration check forecasts the selected mean on its 256 development windows
and independently reconstructs the saved envelope.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import platform
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

FORMAT = "glassbox-public-mean-physical-evaluation-v1"
PROTOCOL_SHA256 = "d3a8d0eeeeab2aac923cad7f1f5fe1c8a49fc0ffa765e84b3d7c6bc80cec8f99"
REFERENCE_SHA256 = "88b59d60be28ffed1e93fa6e6ecaa026b1d49950a23f7f2b1e0f9b65b3b9d99d"
ORACLE_COMMIT = "8b61830c9c353dbb25edc6e63a76886d0ce9b9d3"
RELATIVE = "src/glassbox/experimental/public_mean_physical_evaluation.py"
ARMS = ("public_v3", "research", "public_v4", "public_v4_float64", "hold")
ROLES = {
    "historical": ("public_v3", "research"),
    "public": ("public_v4", "public_v4_float64"),
}
HARD_TIMEOUT_S = 14400
OPERATORS = tuple(
    "src/glassbox/experimental/" + name + ".py"
    for name in (
        "two_simulator_metrics",
        "state_input_decision",
        "two_simulator_flight",
    )
)


def operator_continuity(binding):
    result = {}
    for path in OPERATORS:
        public = binding["public_source_sha256"].get(path)
        require(
            public is not None and public == binding["oracle_source_sha256"].get(path),
            "worker_source_identity",
            "unchanged physical operator " + path,
        )
        result[path] = public
    return result


class EvaluationError(ValueError):
    """A named qualification integrity failure, distinct from physical losses."""

    def __init__(self, check_id, detail):
        self.check_id = check_id
        super().__init__(check_id + ": " + detail)


def require(value, check_id, detail):
    if not value:
        raise EvaluationError(check_id, detail)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def under(root, relative):
    require(
        isinstance(relative, str)
        and relative
        and "\\" not in relative
        and all(p not in ("", ".", "..") for p in relative.split("/")),
        "payload_integrity",
        "invalid relative path",
    )
    path = Path(root)
    for part in relative.split("/"):
        path /= part
        require(not path.is_symlink(), "payload_integrity", "symlink payload")
    require(path.is_file(), "payload_integrity", "missing payload " + relative)
    return path


def inventory(root, *, excluding="seal.json"):
    root = Path(root)
    result = {}
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(), "payload_integrity", "symlink evidence")
        if path.is_file() and path != root / excluding:
            result[path.relative_to(root).as_posix()] = digest(path)
        elif not path.is_file():
            require(path.is_dir(), "payload_integrity", "nonregular evidence")
    return result


def sealed(root, name, expected):
    root = Path(root)
    require(
        digest(under(root, name)) == expected,
        "payload_integrity",
        "external anchor differs",
    )
    seal = read(root / name)
    for relative, sha in seal["files"].items():
        require(digest(under(root, relative)) == sha, "payload_integrity", relative)
    return seal


def stage_binding(path, expected_sha256, current):
    """Authenticate an old stage without pretending its commit is current HEAD."""
    require(
        digest(path) == expected_sha256,
        "worker_source_identity",
        "external stage binding",
    )
    old = read(path)
    require(
        old["format"] == "glassbox-public-mean-implementation-v1"
        and old["protocol_sha256"] == PROTOCOL_SHA256,
        "worker_source_identity",
        "stage binding format/protocol",
    )
    for key in (
        "interpreter_sha256",
        "runtime",
        "machine",
        "oracle_commit",
        "oracle_source_sha256",
        "consumer_source_sha256",
    ):
        require(
            old[key] == current[key],
            "worker_source_identity",
            "stage continuity " + key,
        )
    require(
        Path(old["interpreter"]).resolve() == Path(current["interpreter"]).resolve(),
        "worker_source_identity",
        "stage interpreter path",
    )
    require(
        old["public_source_sha256"]
        and all(
            current["public_source_sha256"].get(path) == sha
            for path, sha in old["public_source_sha256"].items()
        ),
        "worker_source_identity",
        "current implementation is not an exact source superset of stage",
    )
    return old


def _continuity(request, binding):
    result = {}
    for stage in ("data", "flight"):
        old = stage_binding(
            request[stage + "_binding_path"],
            request[stage + "_binding_sha256"],
            binding,
        )
        result[stage] = dict(
            binding_sha256=request[stage + "_binding_sha256"],
            implementation_commit=old["implementation_commit"],
            original_public_root=old["public_root"],
            unchanged_source_files=len(old["public_source_sha256"]),
            scope="Every old bound public source is byte-identical in the current implementation; new evaluator files may be added. Original stage association is retained.",
        )
    return result


def arrays(path):
    import numpy as np

    with np.load(path, allow_pickle=False) as archive:
        require(
            len(archive.files) == len(set(archive.files)),
            "payload_integrity",
            "duplicate array",
        )
        return {k: archive[k] for k in archive.files}


def equal(actual, expected, check_id):
    require(set(actual) == set(expected), check_id, "array roster")
    for key in expected:
        a, b = actual[key], expected[key]
        require(
            a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes(),
            check_id,
            key,
        )


def query_path(query):
    return "predictions/" + query["parent"] + "/" + query["id"] + ".npz"


def _source_identity(request):
    require(
        digest(request["binding_path"]) == request["binding_sha256"],
        "worker_source_identity",
        "external implementation anchor",
    )
    binding = read(request["binding_path"])
    operator_continuity(binding)
    require(
        binding["protocol_sha256"] == PROTOCOL_SHA256,
        "worker_source_identity",
        "protocol binding",
    )
    public = Path(binding["public_root"])
    require(
        Path(__file__).resolve() == public / RELATIVE
        and digest(__file__) == binding["public_source_sha256"][RELATIVE],
        "worker_source_identity",
        "actual worker file",
    )
    for role, root_key, hashes_key, commit in (
        (
            "public",
            "public_root",
            "public_source_sha256",
            binding["implementation_commit"],
        ),
        ("historical", "oracle_root", "oracle_source_sha256", ORACLE_COMMIT),
    ):
        root = Path(binding[root_key])
        actual = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
        require(actual == commit, "worker_source_identity", role + " commit")
        for relative, sha in binding[hashes_key].items():
            require(
                digest(root / relative) == sha,
                "worker_source_identity",
                role + ":" + relative,
            )
    require(
        Path(sys.executable).resolve() == Path(binding["interpreter"]).resolve()
        and digest(Path(sys.executable).resolve()) == binding["interpreter_sha256"],
        "worker_source_identity",
        "interpreter",
    )
    runtime = {
        "python": platform.python_version(),
        **{
            k: importlib.metadata.version(k)
            for k in ("jax", "jaxlib", "numpy", "scipy")
        },
    }
    require(
        runtime == binding["runtime"] and platform.machine() == binding["machine"],
        "worker_source_identity",
        "numerical runtime",
    )
    protocol_path = public / "docs/harness/public-mean-qualification-v1.json"
    require(
        digest(protocol_path) == PROTOCOL_SHA256,
        "worker_source_identity",
        "protocol file",
    )
    qualification = read(protocol_path)
    for entry in qualification["protocol_dependencies"]:
        require(
            digest(public / entry["path"]) == entry["sha256"],
            "worker_source_identity",
            "protocol dependency",
        )
    import jax

    import glassbox

    role = request["role"]
    root = Path(binding["oracle_root" if role == "historical" else "public_root"])
    require(
        Path(glassbox.__file__).resolve() == root / "src/glassbox/__init__.py",
        "worker_source_identity",
        "actual package import",
    )
    require(
        jax.default_backend() == "cpu"
        and bool(jax.config.x64_enabled) == (role == "historical")
        and os.environ.get("SCIPY_ARRAY_API") == "1",
        "worker_source_identity",
        "ambient precision/backend",
    )
    return binding, qualification


def _imports(binding, role):
    root = Path(binding["oracle_root" if role == "historical" else "public_root"])
    pins = binding[
        "oracle_source_sha256" if role == "historical" else "public_source_sha256"
    ]
    result = {}
    for name, module in tuple(sys.modules.items()):
        if name != "glassbox" and not name.startswith("glassbox."):
            continue
        if not getattr(module, "__file__", None):
            continue
        path = Path(module.__file__).resolve()
        require(
            path.is_relative_to(root),
            "worker_source_identity",
            "mixed imported versions",
        )
        relative = path.relative_to(root).as_posix()
        require(
            pins.get(relative) == digest(path),
            "worker_source_identity",
            "unbound import " + name,
        )
        result[name] = {"path": str(path), "sha256": digest(path)}
    return result


def recorded_sources(result, binding, role):
    root = Path(binding["oracle_root" if role == "historical" else "public_root"])
    pins = binding[
        "oracle_source_sha256" if role == "historical" else "public_source_sha256"
    ]
    values = result["imported_sources"]
    require(
        {"glassbox", "glassbox.learner", "glassbox._sequence_model"} <= set(values),
        "worker_source_identity",
        "required actual imports",
    )
    for name, item in values.items():
        path = Path(item["path"])
        require(
            name == "glassbox" or name.startswith("glassbox."),
            "worker_source_identity",
            "recorded package name",
        )
        require(
            path.is_relative_to(root)
            and pins.get(path.relative_to(root).as_posix())
            == item["sha256"]
            == digest(path),
            "worker_source_identity",
            "recorded source role/path/hash",
        )


def resolved(reference_root, qualification, public_root):
    """Keep all retained operators; replace only the prospectively frozen roster."""
    p = read(Path(reference_root) / "resolved-protocol.json")
    entry = qualification["fresh_confirmation"]["original_protocol"]
    path = Path(public_root) / entry["path"]
    require(
        digest(path) == entry["sha256"],
        "truth_roster_and_masks",
        "original query protocol",
    )
    original = read(path)
    roster = []
    for source in original["recordings"]:
        if source["role"] != "test":
            continue
        row = copy.deepcopy(source)
        require(
            row["id"].endswith("test-" + str(row["seed"])),
            "truth_roster_and_masks",
            "original parent identity",
        )
        row["seed"] += 11000000
        row["id"] = row["id"].rsplit("-", 1)[0] + "-" + str(row["seed"])
        roster.append(row)
    require(
        len(roster) == len({r["id"] for r in roster}) == 168,
        "truth_roster_and_masks",
        "complete fresh parent roster",
    )
    p["recordings"] = roster
    return p


def _planned(data_root, simulator, p):
    from glassbox.experimental.two_simulator_flight import query_roster

    records, queries = (
        read(Path(data_root) / "records.json"),
        read(Path(data_root) / "queries.json"),
    )
    entries = [r for r in p["recordings"] if r["simulator"] == simulator]
    require(
        len(entries) == 84
        and read(Path(data_root) / "planned-records.json") == entries,
        "truth_roster_and_masks",
        "planned parents",
    )
    require(
        [r["id"] for r in records] == [r["id"] for r in entries],
        "truth_roster_and_masks",
        "record roster",
    )
    config = read(Path(data_root) / "configuration.json")
    width = sum(channel["kind"] == "control" for channel in config["spec"]["channels"])
    cells = {c["id"]: c for c in p["cells"][simulator]}
    wanted = []
    for record, entry in zip(records, entries, strict=True):
        require(
            all(record[k] == v for k, v in entry.items())
            and record["cell_facts"] == cells[entry["cell"]],
            "truth_roster_and_masks",
            "record facts",
        )
        for row in query_roster(entry, p, p["generation"][simulator]["dt_s"], width):
            row.update(
                scope=cells[entry["cell"]]["group"],
                path="queries/" + entry["id"] + "/" + row["id"] + ".npz",
            )
            wanted.append(row)
    require(len(queries) == len(wanted), "truth_roster_and_masks", "query count")
    for actual, expected in zip(queries, wanted, strict=True):
        require(
            {k: v for k, v in actual.items() if k != "history_eligible"} == expected
            and type(actual.get("history_eligible")) is bool,
            "truth_roster_and_masks",
            "ordered query identity",
        )
    return records, queries, config


def reconstruct_truth(data_root, simulator, p):
    """Rebuild input, truth and mask bytes from saved raw parents/branches only."""
    import numpy as np

    from glassbox.experimental.two_simulator_flight import OBSERVE, validity

    records, queries, config = _planned(data_root, simulator, p)
    width = sum(channel["kind"] == "control" for channel in config["spec"]["channels"])
    bounds = (
        np.asarray(config["bounds"])
        if "bounds" in config
        else np.stack((config["lower"], config["upper"]), axis=1)
    )
    require(bounds.shape == (width, 2), "truth_roster_and_masks", "command bounds")
    dt = p["generation"][simulator]["dt_s"]
    history, horizon = round(0.5 / dt), round(0.25 / dt)
    indexed = {}
    for record in records:
        path = Path(data_root) / (record["prefix"] + ".npz")
        parent = arrays(path) if path.exists() else None
        valid = (
            validity(parent)
            if parent is not None
            else dict(
                valid_transitions=0,
                valid_initial=False,
                failure_index=0,
                reason="setup_failure",
            )
        )
        require(
            valid == record["validity"], "truth_roster_and_masks", "raw parent validity"
        )
        observed = np.asarray(OBSERVE(parent["states"])) if parent is not None else None
        indexed[record["id"]] = (record, parent, valid, observed)
    for query in queries:
        _, parent, valid, observed = indexed[query["parent"]]
        origin = query["origin"]
        has_history = bool(
            valid["valid_initial"] and origin <= valid["valid_transitions"]
        )
        require(
            query["history_eligible"] == has_history,
            "truth_roster_and_masks",
            "history eligibility",
        )
        value = dict(
            past_states=np.full((history + 1, 15), np.nan),
            past_inputs=np.full((history, width), np.nan),
            future_inputs=np.full((horizon, width), np.nan),
            factual_inputs=np.full((horizon, width), np.nan),
            target=np.full((horizon, 15), np.nan),
            factual_target=np.full((horizon, 15), np.nan),
            valid=np.zeros(horizon, dtype=bool),
        )
        if has_history:
            value.update(
                past_states=observed[origin - history : origin + 1].copy(),
                past_inputs=parent["commands"][origin - history : origin].copy(),
                factual_inputs=parent["commands"][origin : origin + horizon].copy(),
                factual_target=observed[origin + 1 : origin + horizon + 1].copy(),
            )
            value["future_inputs"], value["target"] = (
                value["factual_inputs"].copy(),
                value["factual_target"].copy(),
            )
            value["valid"] = (
                np.arange(1, horizon + 1) + origin <= valid["valid_transitions"]
            )
            if query["kind"] == "response":
                length = finite_command_prefix(value["factual_inputs"])
                channel, sign = query["channel"], query["sign"]
                bound = bounds[channel, 0 if sign < 0 else 1]
                value["future_inputs"][:, channel] += 0.1 * (
                    bound - value["future_inputs"][:, channel]
                )
                value["target"] = np.full((horizon, 15), np.nan)
                if length:
                    branch = arrays(
                        under(
                            data_root,
                            "branches/" + query["parent"] + "/" + query["id"] + ".npz",
                        )
                    )
                    equal(
                        {"commands": branch["commands"]},
                        {"commands": value["future_inputs"][:length]},
                        "truth_roster_and_masks",
                    )
                    value["target"][:length] = np.asarray(OBSERVE(branch["states"]))[1:]
                    branch_valid = validity(branch)
                    value["valid"] &= branch_valid["valid_initial"] & (
                        np.arange(1, horizon + 1) <= branch_valid["valid_transitions"]
                    )
                else:
                    value["valid"][:] = False
            value["command_delta"] = value["future_inputs"] - value["factual_inputs"]
        else:
            value["command_delta"] = np.full((horizon, width), np.nan)
        equal(arrays(under(data_root, query["path"])), value, "truth_roster_and_masks")
    return dict(
        parents=len(records),
        queries=len(queries),
        exact_input_truth_mask_arrays=True,
        simulation_calls=0,
    )


def finite_command_prefix(commands):
    import numpy as np

    bad = np.flatnonzero(~np.isfinite(commands).all(axis=1))
    return int(bad[0]) if len(bad) else len(commands)


def predict_query(function, query, values, *, dtype, factual=False):
    """Infer every finite command prefix even when later physical truth is absent."""
    import numpy as np

    commands = values["factual_inputs" if factual else "future_inputs"]
    result = np.full((len(commands), 15), np.nan, dtype=dtype)
    if not query["history_eligible"]:
        return result
    length = finite_command_prefix(commands)
    if length:
        predicted = np.asarray(
            function(values["past_states"], values["past_inputs"], commands[:length])
        )
        require(
            predicted.shape == (length, 15) and predicted.dtype == np.dtype(dtype),
            "prediction_contract",
            "returned shape/dtype",
        )
        result[:length] = predicted
    return result


def calibration_evidence(model):
    """Independent split-conformal reduction from selected mean + actual dev256."""
    import jax
    import numpy as np

    before = bool(jax.config.x64_enabled)
    batch = model._development.batch
    require(
        len(batch.past_states) == 256,
        "calibration_envelope_reconstruction",
        "development count",
    )
    with jax.enable_x64(True):
        prediction = np.asarray(
            model._model.rollout(
                batch.past_states, batch.past_inputs, batch.future_inputs
            )
        )
        residual = np.abs(prediction - batch.future_states)
        require(
            np.isfinite(residual).all() and prediction.dtype == np.dtype("float64"),
            "calibration_envelope_reconstruction",
            "finite float64 residuals",
        )
        rank = min(int(np.ceil((len(residual) + 1) * 0.9)), len(residual))
        half_width = np.sort(residual, axis=0)[rank - 1]
    require(
        rank == 232 and bool(jax.config.x64_enabled) == before,
        "calibration_envelope_reconstruction",
        "rank/precision restoration",
    )
    equal(
        {"half_width": np.asarray(model.envelope())},
        {"half_width": half_width},
        "calibration_envelope_reconstruction",
    )
    expected = dict(
        nominal_coverage=0.9,
        method="split_conformal_absolute_error",
        calibrated_on="development",
        calibration_windows=256,
        quantile_rank=232,
        units="physical, per horizon step and per channel, aligned with predict",
        half_width=half_width.tolist(),
    )
    require(
        model.report["envelope"] == expected,
        "calibration_envelope_reconstruction",
        "report method/source/count/rank/width",
    )
    return dict(
        prediction=prediction, absolute_residual=residual, half_width=half_width
    ), dict(
        nominal_coverage=0.9,
        calibration_windows=256,
        quantile_rank=232,
        selected_step=model.report["optimization"]["selected_step"],
        actual_selected_mean_development_forecasts=1,
        calibration_source="development",
        heldout_coverage_qualified=False,
        scope="Fresh selected-mean development forecasts and independent NumPy quantile; no fit or initializer, no nominal held-out coverage claim.",
    )


def _worker(request_path):
    request = read(request_path)
    binding, qualification = _source_identity(request)
    import jax
    import numpy as np

    from glassbox import LearnedDynamics

    continuity = _continuity(request, binding)
    reference = sealed(request["reference_root"], "run.json", REFERENCE_SHA256)
    data_seal = sealed(request["data_root"], "seal.json", request["data_sha256"])
    require(
        data_seal["simulator"] == request["simulator"]
        and data_seal["protocol_sha256"] == PROTOCOL_SHA256
        and data_seal["implementation_sha256"] == request["data_binding_sha256"],
        "payload_integrity",
        "data association",
    )
    p = resolved(request["reference_root"], qualification, binding["public_root"])
    _, queries, _ = _planned(request["data_root"], request["simulator"], p)
    root = Path(request["directory"])
    replay = request["replay"]
    if not replay:
        root.mkdir(parents=True, exist_ok=False)
    role = request["role"]
    models, paths = {}, {}
    if role == "historical":
        from glassbox.experimental.expanded_training_cache_model import (
            CandidateDynamics,
        )

        for arm, subdirectory, cls in (
            ("public_v3", "baseline", LearnedDynamics),
            ("research", "candidate", CandidateDynamics),
        ):
            relative = request["simulator"] + "/" + subdirectory + "/model.npz"
            path = under(request["reference_root"], relative)
            require(
                digest(path) == reference["files"][relative],
                "payload_integrity",
                "reference model",
            )
            models[arm], paths[arm] = cls.load(path), digest(path)
        truth = reconstruct_truth(request["data_root"], request["simulator"], p)
    else:
        fit_seal = sealed(
            request["flight_fit_root"], "manifest.json", request["flight_fit_sha256"]
        )
        require(
            fit_seal["status"] == "complete"
            and fit_seal["simulator"] == request["simulator"]
            and fit_seal["implementation_manifest_sha256"]
            == request["flight_binding_sha256"]
            and fit_seal["implementation_commit"]
            == continuity["flight"]["implementation_commit"],
            "payload_integrity",
            "public fit association",
        )
        path = under(request["flight_fit_root"], "model.npz")
        model = LearnedDynamics.load(path)
        models = {arm: model for arm in ROLES[role]}
        paths = {arm: digest(path) for arm in models}
        truth = None
    counters = {
        arm: dict(branch_queries=0, factual_queries=0, inferred_prefixes=0)
        for arm in models
    }
    envelopes = {arm: np.asarray(model.envelope()) for arm, model in models.items()}

    def emit(relative, value):
        path = root / relative
        if replay:
            equal(arrays(path), value, "saved_prediction_replay")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, **value)

    emit("envelopes.npz", envelopes)
    calibration = None
    if role == "public":
        witness, calibration = calibration_evidence(models["public_v4"])
        emit("calibration.npz", witness)
    # Separate JIT wrappers are first traced under their specified ambient mode.
    functions = {arm: jax.jit(model.predict) for arm, model in models.items()}
    for query in queries:
        values = arrays(under(request["data_root"], query["path"]))
        output = {}
        for arm in models:
            use64 = arm != "public_v4"
            with jax.enable_x64(use64):
                for factual in (
                    (False, True) if query["kind"] == "response" else (False,)
                ):
                    name = arm + ("_factual" if factual else "")
                    output[name] = predict_query(
                        functions[arm],
                        query,
                        values,
                        dtype=np.float64 if use64 else np.float32,
                        factual=factual,
                    )
                    counters[arm][
                        "factual_queries" if factual else "branch_queries"
                    ] += 1
                    command = values["factual_inputs" if factual else "future_inputs"]
                    counters[arm]["inferred_prefixes"] += int(
                        query["history_eligible"] and finite_command_prefix(command) > 0
                    )
        emit(query_path(query), output)
    imported = _imports(binding, role)
    result = dict(
        format=FORMAT,
        role=role,
        simulator=request["simulator"],
        data_sha256=request["data_sha256"],
        flight_fit_sha256=request["flight_fit_sha256"],
        reference_sha256=REFERENCE_SHA256,
        implementation_sha256=request["binding_sha256"],
        unchanged_physical_operators=operator_continuity(binding),
        stage_continuity=continuity,
        imported_sources=imported,
        model_sha256=paths,
        arms=list(models),
        counters=counters,
        truth_reconstruction=truth,
        calibration_reconstruction=calibration,
        fits=0,
        simulations=0,
        ambient_x64_restored=bool(jax.config.x64_enabled) == (role == "historical"),
    )
    require(
        result["ambient_x64_restored"], "worker_source_identity", "precision leakage"
    )
    _source_identity(request)
    if replay:
        saved = read(root / "result.json")
        recorded_sources(saved, binding, role)
        require(
            saved["imported_sources"] == imported,
            "worker_source_identity",
            "actual fresh imported-source witness",
        )
        require(
            saved == result,
            "saved_prediction_replay",
            "actual source/model/query witness",
        )
        write(request["replay_result"], dict(exact=True, **result))
    else:
        write(root / "result.json", result)
    return result


def validate_rows(rows, queries, p, simulator):
    from glassbox.experimental.two_simulator_metrics import GROUPS, _horizons

    horizons = _horizons(
        p["generation"][simulator]["dt_s"], p["evaluation"]["horizons_s"]
    )
    expected = {
        (q["parent"], q["id"], arm, h, group, stat): q
        for q in queries
        for arm in ARMS
        for _, h in horizons
        for group in GROUPS
        for stat in ("endpoint", "cumulative")
    }
    require(
        len(rows) == len(expected),
        "truth_roster_and_masks",
        "complete metric-slot count",
    )
    seen = set()
    for row in rows:
        key = tuple(
            row[k]
            for k in ("parent", "query", "arm", "horizon_s", "group", "statistic")
        )
        require(
            key in expected and key not in seen,
            "truth_roster_and_masks",
            "duplicate/missing/unplanned metric slot",
        )
        seen.add(key)
        q = expected[key]
        require(
            all(
                row[k] == q[k]
                for k in ("parent", "scope", "cell", "origin", "kind", "simulator")
            )
            and row.get("channel") == q.get("channel")
            and row.get("sign") == q.get("sign"),
            "truth_roster_and_masks",
            "metric identity",
        )


def score(data_root, evaluation_root, simulator, p):
    """Recompute all metric rows from sealed raw queries and saved mean arrays."""
    import numpy as np

    from glassbox.experimental.two_simulator_metrics import (
        aggregate,
        score_forecast,
        score_response_directions,
    )

    _, queries, _ = _planned(data_root, simulator, p)
    evaluation_root = Path(evaluation_root)
    envelopes = {}
    for role in ROLES:
        loaded = arrays(evaluation_root / role / "envelopes.npz")
        require(
            set(loaded) == set(ROLES[role]), "prediction_contract", "envelope roster"
        )
        envelopes.update(loaded)
    rows, directions, lower_pairs = [], [], {}
    dt, horizons = p["generation"][simulator]["dt_s"], p["evaluation"]["horizons_s"]
    for query in queries:
        values = arrays(under(data_root, query["path"]))
        predictions = {}
        for role, arms in ROLES.items():
            loaded = arrays(evaluation_root / role / query_path(query))
            expected = set(arms) | (
                {arm + "_factual" for arm in arms}
                if query["kind"] == "response"
                else set()
            )
            require(
                set(loaded) == expected,
                "prediction_contract",
                "complete per-query arm roster",
            )
            for key, value in loaded.items():
                dtype = (
                    np.float32
                    if key in ("public_v4", "public_v4_factual")
                    else np.float64
                )
                require(
                    value.shape == values["target"].shape
                    and value.dtype == np.dtype(dtype),
                    "prediction_contract",
                    "saved prediction shape/dtype",
                )
            predictions.update(loaded)
        predictions["hold"] = predict_query(
            lambda x, u, f: np.repeat(x[-1:], len(f), axis=0),
            query,
            values,
            dtype=np.float64,
        )
        target = values["target"]
        if query["kind"] == "response":
            predictions["hold_factual"] = predict_query(
                lambda x, u, f: np.repeat(x[-1:], len(f), axis=0),
                query,
                values,
                dtype=np.float64,
                factual=True,
            )
            # The inherited evaluator stores means in float64 before physical
            # response subtraction. Preserve native32 outputs on disk, then use
            # that same reduction arithmetic without extra float32 subtraction.
            predictions = {
                arm: np.asarray(predictions[arm], dtype=float)
                - np.asarray(predictions[arm + "_factual"], dtype=float)
                for arm in ARMS
            }
            target = target - values["factual_target"]
            key = (query["parent"], query["origin"], query["channel"])
            if query["sign"] < 0:
                require(
                    key not in lower_pairs,
                    "truth_roster_and_masks",
                    "duplicate lower response",
                )
                lower_pairs[key] = (predictions, target, values["valid"])
            else:
                require(
                    key in lower_pairs,
                    "truth_roster_and_masks",
                    "missing lower response",
                )
                lower, lower_truth, lower_valid = lower_pairs.pop(key)
                for arm in ARMS:
                    scored = score_response_directions(
                        np.stack((lower[arm], predictions[arm])),
                        np.stack((lower_truth, target)),
                        np.stack((lower_valid, values["valid"])),
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
    require(not lower_pairs, "truth_roster_and_masks", "unpaired response")
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


def reduce_decision(rows, p, qualification, *, queries):
    """Reuse frozen operators with only the prospectively declared gate sets."""
    from glassbox.experimental.state_input_decision import _number
    from glassbox.experimental.state_input_decision import reduce as paired_reduce
    from glassbox.experimental.two_simulator_metrics import aggregate

    require(
        set(rows) == set(queries) == {"crazyflow", "cascade"},
        "decision_reduction",
        "both simulators",
    )
    for simulator in rows:
        validate_rows(rows[simulator], queries[simulator], p, simulator)
        masks = {}
        for row in rows[simulator]:
            identity = tuple(
                row[k] for k in ("parent", "query", "horizon_s", "group", "statistic")
            )
            mask = tuple(
                row.get(k)
                for k in (
                    "truth_eligible",
                    "pair_nonweak",
                    "components",
                    "horizon_steps",
                )
            )
            require(
                identity not in masks or masks[identity] == mask,
                "truth_roster_and_masks",
                "common-arm truth masks",
            )
            masks[identity] = mask
    # The frozen integrity rule requires every eligible prediction to be finite,
    # including public64 diagnostics. Only their residual magnitudes are nongating;
    # a diagnostic label does not exempt an output from this integrity contract.
    all_finite = all(
        not r["truth_eligible"] or (r["prediction_finite"] and _number(r["mse"]))
        for values in rows.values()
        for r in values
    )
    policy = qualification["fresh_confirmation"]["decision"]
    research_limits = policy["vs_expanded_research"]
    comparisons = {}
    for reference, policy_key in (
        ("public_v3", "vs_public_v3"),
        ("research", "vs_expanded_research"),
    ):
        pair = {
            sim: [
                dict(r, arm="candidate" if r["arm"] == "public_v4" else "baseline")
                for r in values
                if r["arm"] in ("public_v4", reference)
            ]
            for sim, values in rows.items()
        }
        view = copy.deepcopy(p)
        # Inherited reducer computes all physical diagnostics. Its context guard
        # booleans are discarded for v3, never folded into v3 acceptance.
        view["decision"]["accept_research_candidate"] = dict(
            research_limits,
            tail_guard_groups=policy["tail_groups"],
            tail_guard_kinds=policy["tail_kinds"],
        )
        result = paired_reduce(
            {sim: aggregate(values) for sim, values in pair.items()}, view, rows=pair
        )
        ratios = result["weighted_geometric_mean_ratios"]
        limits = policy[policy_key]
        checks = {
            key: result["checks"][key]
            for key in (
                "matching_planned_queries_and_truth",
                "finite_eligible_predictions",
            )
        }
        checks.update(
            {
                kind + "_aggregate": ratios[kind] is not None
                and ratios[kind] <= limits[kind + "_weighted_geometric_mean_ratio_max"]
                for kind in ("factual", "response")
            }
        )
        if reference == "research":
            checks.update(
                {
                    key: result["checks"][key]
                    for key in (
                        "primary_simulator_regressions",
                        "scope_regressions",
                        "primary_parent_tail_regressions",
                    )
                }
            )
        else:
            result["tail_checks"] = [
                {k: v for k, v in r.items() if k != "pass"}
                for r in result["tail_checks"]
            ]
        result.update(
            checks=checks,
            residual_criteria_pass=all(checks.values()),
            numerator_arm="public_v4",
            denominator_arm=reference,
            limits=copy.deepcopy(limits),
            public_promotion=False,
            remaining_promotion_requirements=qualification["public_mean_promotion"][
                "all_required"
            ],
        )
        comparisons[reference] = result
    hashes = {
        name: value["bootstrap"].get("parent_draws_sha256")
        for name, value in comparisons.items()
    }
    available = {x for x in hashes.values() if x is not None}
    require(len(available) <= 1, "decision_reduction", "paired bootstrap schedule")
    return dict(
        format=FORMAT,
        comparisons=comparisons,
        all_arm_eligible_predictions_finite=all_finite,
        fresh_default32_physical_comparisons_pass=all_finite
        and all(r["residual_criteria_pass"] for r in comparisons.values()),
        bootstrap_parent_draws_sha256=next(iter(available), None),
        shared_bootstrap_parent_draws_verified=all(hashes.values())
        and len(available) == 1,
        public_mean_adopted=False,
        coverage_is_separate=True,
        float64_is_diagnostic=True,
        float64_diagnostic_scope="Residual magnitudes and coverage are nongating; the universal finite-output and matched-cohort integrity rules still apply.",
        not_implied=qualification["public_mean_promotion"]["not_implied"],
    )


def promotion(physical, qualifications, qualification):
    required = qualification["public_mean_promotion"]["all_required"]
    require(
        set(qualifications) == set(required)
        and all(type(v) is bool for v in qualifications.values()),
        "decision_reduction",
        "exact Boolean qualification roster",
    )
    require(
        qualifications["fresh_default32_physical_comparisons_pass"]
        is physical["fresh_default32_physical_comparisons_pass"],
        "decision_reduction",
        "physical qualification mirror",
    )
    return dict(
        checks=dict(qualifications),
        public_mean_adopted=all(qualifications.values()),
        not_implied=qualification["public_mean_promotion"]["not_implied"],
        evidence_scope="Inputs must come from separately authenticated qualification stages; this pure policy reducer does not authenticate those stages.",
    )


def _verify_prior_fit(
    directory, expected_sha256, old_binding_sha256, continuity, reference_root
):
    """Use unchanged read-only validators, preserving the old stage's binding."""
    from glassbox import LearnedDynamics
    from glassbox._learner_arrays import load_arrays
    from glassbox.experimental import public_mean_flight_fit as fit
    from glassbox.io.recordings import load_recordings

    directory = Path(directory)
    manifest = sealed(directory, "manifest.json", expected_sha256)
    require(
        manifest["files"] == fit._inventory(directory),
        "payload_integrity",
        "flight payload roster",
    )
    require(
        manifest["format"] == fit.FORMAT
        and manifest["protocol_sha256"] == PROTOCOL_SHA256
        and manifest["reference_bundle_sha256"] == REFERENCE_SHA256
        and manifest["status"] == "complete"
        and manifest["implementation_manifest_sha256"] == old_binding_sha256
        and manifest["implementation_commit"] == continuity["implementation_commit"],
        "payload_integrity",
        "original flight identity",
    )
    execution, outcome = (
        read(directory / "execution.json"),
        read(directory / "outcome.json"),
    )
    require(
        fit._stage_status(directory, execution) == "complete"
        and execution["hard_wall_time_s"] == 14400,
        "payload_integrity",
        "completed flight execution",
    )
    require(
        outcome["status"] == outcome["stage"] == "complete"
        and outcome["configuration_before"] == outcome["configuration_after"]
        and outcome["configuration_before"]["jax_enable_x64"] is False
        and all(
            outcome["timing"][key] == 1
            for key in (
                "public_fit_calls",
                "training_entry_calls",
                "fitter_calls",
                "initializer_calls",
                "calibration_calls",
            )
        ),
        "payload_integrity",
        "single flight fit outcome",
    )
    reference = fit._prepare(reference_root, manifest["simulator"])
    require(
        read(directory / "reference.json") == reference.source,
        "payload_integrity",
        "flight reference links",
    )
    for name in ("reference-preparation.npz", "actual-preparation.npz"):
        fit._compare_preparation(load_arrays(directory / name), reference)
    fit._same_recordings(
        load_recordings(directory / "recordings.npz"), reference.collection
    )
    model = LearnedDynamics.load(directory / "model.npz")
    require(
        model.fingerprint() == outcome["model_fingerprint"]
        and model.report == read(directory / "report.json"),
        "payload_integrity",
        "flight saved revision",
    )
    parity = dict(
        model=fit._compare_model(model, reference),
        work=fit._work(directory, model, reference),
    )
    require(
        parity == read(directory / "parity.json")
        and fit._prefix(directory) == manifest["observed_prefix"],
        "payload_integrity",
        "flight observed parity",
    )
    return manifest


def _prepare(
    simulator,
    *,
    data_root,
    data_sha256,
    flight_fit_root,
    flight_fit_sha256,
    reference_root,
    binding_path,
    binding_sha256,
    data_binding_path,
    data_binding_sha256,
    flight_binding_path,
    flight_binding_sha256,
):
    from glassbox.experimental.public_mean_implementation import verify
    from glassbox.experimental.public_mean_physics import verify_data
    from glassbox.experimental.public_mean_saved_port import authenticate_reference

    binding = verify(binding_path, binding_sha256)
    operator_continuity(binding)
    require(
        Path(__file__).resolve() == Path(binding["public_root"]) / RELATIVE
        and binding["public_source_sha256"].get(RELATIVE) == digest(__file__),
        "worker_source_identity",
        "committed evaluation source",
    )
    authenticate_reference(reference_root, expected_bundle_sha256=REFERENCE_SHA256)
    continuity = _continuity(
        dict(
            data_binding_path=data_binding_path,
            data_binding_sha256=data_binding_sha256,
            flight_binding_path=flight_binding_path,
            flight_binding_sha256=flight_binding_sha256,
        ),
        binding,
    )
    data = verify_data(data_root, data_sha256)
    require(
        data["simulator"] == simulator
        and data["implementation_sha256"] == data_binding_sha256,
        "payload_integrity",
        "data stage association",
    )
    fitted = _verify_prior_fit(
        flight_fit_root,
        flight_fit_sha256,
        flight_binding_sha256,
        continuity["flight"],
        reference_root,
    )
    require(
        fitted["status"] == "complete" and fitted["simulator"] == simulator,
        "payload_integrity",
        "qualified public fit",
    )
    qualification = read(
        Path(binding["public_root"]) / "docs/harness/public-mean-qualification-v1.json"
    )
    p = resolved(reference_root, qualification, binding["public_root"])
    _planned(data_root, simulator, p)
    return binding, qualification, p


def _launch(request, directory, binding):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    write(directory / "request.json", request)
    environment = dict(os.environ)
    historical = request["role"] == "historical"
    environment["PYTHONPATH"] = str(
        Path(binding["oracle_root" if historical else "public_root"]) / "src"
    )
    environment["SCIPY_ARRAY_API"] = "1"
    if historical:
        environment["JAX_ENABLE_X64"] = "1"
    else:
        environment.pop("JAX_ENABLE_X64", None)
    command = [
        binding["interpreter"],
        str(Path(__file__).resolve()),
        "--worker",
        str(directory / "request.json"),
    ]
    write(
        directory / "command.json",
        dict(
            argv=command,
            environment={
                k: environment.get(k)
                for k in ("PYTHONPATH", "JAX_ENABLE_X64", "SCIPY_ARRAY_API", "PATH")
            },
            hard_timeout_s=HARD_TIMEOUT_S,
        ),
    )
    started = time.perf_counter()
    result = dict(status="failed", returncode=None, hard_timeout=False)
    try:
        with (directory / "worker.log").open("x") as log:
            process = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=environment,
                start_new_session=True,
            )
            try:
                result["returncode"] = process.wait(timeout=HARD_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                result["hard_timeout"] = True
                os.killpg(process.pid, signal.SIGKILL)
                result["returncode"] = process.wait()
        require(
            not result["hard_timeout"] and result["returncode"] == 0,
            "worker_execution",
            "worker failed; no retry",
        )
        result["status"] = "complete"
    except BaseException as error:
        result.update(
            error_type=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc(),
        )
        raise
    finally:
        result["elapsed_s"] = time.perf_counter() - started
        write(directory / "execution.json", result)
    return result


def run(simulator, output, **inputs):
    """Two isolated inference processes, then one saved-array metric reduction."""
    binding, _, p = _prepare(simulator, **inputs)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    request = dict(inputs, simulator=simulator, replay=False)
    for key in (
        "data_root",
        "flight_fit_root",
        "reference_root",
        "binding_path",
        "data_binding_path",
        "flight_binding_path",
    ):
        request[key] = str(Path(request[key]).resolve())
    status = "failed"
    try:
        for role in ROLES:
            _launch(
                dict(request, role=role, directory=str(output / role)),
                output / (role + "-execution"),
                binding,
            )
        for key, value in score(inputs["data_root"], output, simulator, p).items():
            write(output / (key + ".json"), value)
        status = "complete"
    finally:
        seal = dict(
            format=FORMAT,
            status=status,
            simulator=simulator,
            protocol_sha256=PROTOCOL_SHA256,
            inputs=request,
            files=inventory(output),
        )
        write(output / "seal.json", seal)
    return dict(
        status=status, seal_sha256=digest(output / "seal.json"), directory=str(output)
    )


def verify(output, *, expected_sha256, replay_output=None, **inputs):
    """Recompute raw reductions; optional fresh worker forecasts never refit."""
    output = Path(output).resolve()
    seal = sealed(output, "seal.json", expected_sha256)
    require(
        seal["files"] == inventory(output),
        "payload_integrity",
        "evaluation payload roster",
    )
    require(
        seal["format"] == FORMAT
        and seal["status"] == "complete"
        and seal["protocol_sha256"] == PROTOCOL_SHA256,
        "payload_integrity",
        "completed evaluation",
    )
    simulator = seal["simulator"]
    binding, _, p = _prepare(simulator, **inputs)
    expected = dict(inputs, simulator=simulator, replay=False)
    for key in (
        "data_root",
        "flight_fit_root",
        "reference_root",
        "binding_path",
        "data_binding_path",
        "flight_binding_path",
    ):
        expected[key] = str(Path(expected[key]).resolve())
    require(seal["inputs"] == expected, "payload_integrity", "evaluation source links")
    for role in ROLES:
        result = read(output / role / "result.json")
        recorded_sources(result, binding, role)
        require(
            result["role"] == role
            and result["simulator"] == simulator
            and result["data_sha256"] == inputs["data_sha256"]
            and result["flight_fit_sha256"] == inputs["flight_fit_sha256"]
            and result["reference_sha256"] == REFERENCE_SHA256
            and result["implementation_sha256"] == inputs["binding_sha256"]
            and result["stage_continuity"] == _continuity(expected, binding)
            and result["unchanged_physical_operators"] == operator_continuity(binding),
            "payload_integrity",
            "worker stage associations",
        )
    reduced = score(inputs["data_root"], output, simulator, p)
    for key, value in reduced.items():
        require(read(output / (key + ".json")) == value, "physical_reduction", key)
    if replay_output is not None:
        replay_output = Path(replay_output).resolve()
        replay_output.mkdir(parents=True, exist_ok=False)
        for role in ROLES:
            request = dict(
                expected,
                role=role,
                replay=True,
                directory=str(output / role),
                replay_result=str(replay_output / (role + "-replay.json")),
            )
            _launch(request, replay_output / (role + "-execution"), binding)
    return dict(
        exact_saved_reduction=True,
        fresh_prediction_replay=replay_output is not None,
        rows=len(reduced["rows"]),
        fits=0,
        simulations=0,
        simulator=simulator,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", required=True)
    _worker(parser.parse_args().worker)
