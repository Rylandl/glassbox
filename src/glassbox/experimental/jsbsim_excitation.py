"""Frozen, no-fit paired JSBSim startup and response-timescale experiment."""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from . import jsbsim_onboarding as old
from .jsbsim_onboarding_audit import bitwise_equal, expected_tape

REPO = Path(__file__).resolve().parents[3]
PROTOCOL = REPO / "docs/harness/jsbsim-excitation-v1.json"
PROTOCOL_SHA = "1b63e73d4afb7b09811140f4fc21b84a66beb2aeff3e6e463b567aad07e34786"
CASE_TIMEOUT_S = 180


def load_spec():
    if old.digest(PROTOCOL) != PROTOCOL_SHA:
        raise ValueError("frozen excitation protocol changed")
    protocol = json.loads(PROTOCOL.read_text())
    inherited, inventory = old.load_spec()
    for name, sha in protocol["reference"]["inherited_source_sha256"].items():
        if old.digest(REPO / name) != sha:
            raise ValueError(f"inherited source changed: {name}")
    return protocol, inherited, inventory


def runtime_identity():
    identity = old.runtime_identity()
    identity.update(protocol_sha256=PROTOCOL_SHA,
                    inherited_protocol_sha256=old.PROTOCOL_SHA)
    return identity


def _sealed_files(output):
    actual = {}
    for path in sorted(output.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"evidence symlink: {path}")
        if path.is_file() and path != output / "run.json":
            actual[str(path.relative_to(output))] = old.digest(path)
    return actual


def validate_reference(reference, protocol, identity=None):
    """Authenticate historical bytes without changing their full-source identity."""
    ref = protocol["reference"]
    if old.digest(reference / "run.json") != ref["run_sha256"]:
        raise ValueError("old reference external seal changed")
    seal = json.loads((reference / "run.json").read_text())
    if _sealed_files(reference) != seal["files"]:
        raise ValueError("old reference payload changed")
    for key, name in (("protocol_sha256", "protocol.json"),
                      ("inventory_sha256", "inventory.json")):
        if old.digest(reference / name) != ref[key] or seal[key] != ref[key]:
            raise ValueError("old reference frozen input changed")
    saved = json.loads((reference / "runtime.json").read_text())
    if saved["source_files"] != ref["inherited_source_sha256"]:
        raise ValueError("old reference source pins changed")
    identity = runtime_identity() if identity is None else identity
    for key in ("python", "python_binary_sha256", "numpy", "numpy_native_sha256",
                "jsbsim", "jsbsim_files", "platform", "machine", "dispersion",
                "simulator_seed", "inventory_sha256"):
        if saved[key] != identity[key]:
            raise ValueError(f"old reference runtime differs: {key}")


def inherited_long(protocol, inherited):
    extended = copy.deepcopy(inherited)
    extended["recording"]["transitions"] = protocol["recording"]["transitions"]
    extended["replay"]["horizon_steps"] = max(protocol["recording"]["horizons_steps"])
    return extended


def common_tape(entry, protocol, inherited, reference_case, reference_arrays):
    parent = reference_case["runs"]["parent"]
    initial = parent.get("diagnostics", {}).get("initial_commands")
    if initial is None:
        return None
    initial = np.asarray(initial, dtype=np.float64)
    bounds = np.asarray(parent["bounds"], dtype=np.float64)
    if (initial.ndim != 1 or bounds.shape != (len(initial), 2)
            or not np.isfinite(initial).all()):
        raise ValueError("invalid saved reference baseline")
    tape = expected_tape(inherited_long(protocol, inherited), entry, initial, bounds)
    previous = reference_arrays["parent__commands"]
    if not bitwise_equal(tape[:len(previous)], previous):
        raise ValueError("extended tape differs from old command prefix")
    return tape


def case_path(output, entry_id, arm):
    return output / "cases" / old.case_slug(entry_id + "::" + arm)


def reference_case_at(reference, entry):
    return old.read_case(reference / "cases" / old.case_slug(entry["id"]))


def execute_case(root, entry, arm, protocol, inherited, reference_case,
                 reference_arrays, checkpoint_dir=None):
    from .jsbsim_excitation_adapter import simulate
    from .jsbsim_excitation_audit import validate_arm

    if arm not in protocol["arms"]:
        raise ValueError("unknown excitation arm")
    started = time.perf_counter()
    tape = common_tape(entry, protocol, inherited, reference_case, reference_arrays)
    arrays, runs = {}, {}
    names = reference_case["runs"]["parent"].get("command_names", [])
    roster = old.branch_roster(len(names))
    case = dict(id=entry["id"], arm=arm, runs=runs, branches=roster)
    for label in ["parent", "replay", *(x["label"] for x in roster)]:
        runs[label] = dict(status="unattempted")
    output = checkpoint_dir
    if output is not None:
        output.mkdir(parents=True, exist_ok=False)

    def checkpoint():
        if output is not None:
            old.checkpoint(output, case, arrays)

    def retain(label, result):
        metadata = {k: v for k, v in result.items() if k != "arrays"}
        for name, values in result["arrays"].items():
            arrays[f"{label}__{name}"] = np.asarray(values)
        runs[label] = metadata
        if label == "parent" and metadata.get("command_names"):
            case["branches"] = old.branch_roster(len(metadata["command_names"]))
            for row in case["branches"]:
                runs.setdefault(row["label"], dict(status="unattempted"))
        return metadata

    def invoke(label, commands):
        runs[label] = dict(status="running", stage="construct")
        checkpoint()

        def progress(value):
            if output is None:
                return
            if value.get("progress_only"):
                temporary = output / "progress.tmp"
                old.write_json(temporary, dict(label=label, **value))
                temporary.replace(output / "progress.json")
            elif (value["status"] != "running"
                  or value.get("completed_intervals", 0) % 10 == 0):
                retain(label, value)
                checkpoint()
                (output / "progress.json").unlink(missing_ok=True)

        result = simulate(root, entry, commands, inherited_long(protocol, inherited),
                          arm, protocol, progress=progress)
        metadata = retain(label, result)
        checkpoint()
        if output is not None:
            (output / "progress.json").unlink(missing_ok=True)
        return metadata

    checkpoint()
    parent = invoke("parent", tape)
    if tape is not None:
        invoke("replay", tape)
    origin = protocol["recording"]["origin"]
    if len(arrays["parent__observations"]) > origin:
        if tape is None:
            raise ValueError("unfrozen baseline became available")
        bounds = np.asarray(parent["bounds"], dtype=np.float64)
        for branch in case["branches"]:
            commands = tape.copy()
            channel = branch["command_index"]
            if channel is not None:
                bound = bounds[channel, int(branch["direction"] == "upper")]
                commands[origin:, channel] += 0.1 * (bound - commands[origin:, channel])
            invoke(branch["label"], commands)
    case["elapsed_s"] = time.perf_counter() - started
    case["computed"] = validate_arm(case, arrays, protocol, inherited, entry,
                                    reference_case, reference_arrays)
    if output is not None:
        _finish_case(output, case, arrays)
    return case, arrays


def _finish_case(output, case, arrays):
    np.savez_compressed(output / "arrays.npz", **arrays)
    old.write_json(output / "case.json", case)
    for name in ("checkpoint.npz", "checkpoint.tmp", "progress.json", "progress.tmp"):
        (output / name).unlink(missing_ok=True)


def run_all(root, output, reference, workers=4):
    from .jsbsim_excitation_audit import summarize, validate_arm

    protocol, inherited, inventory = load_spec()
    old.check_assets(root, inventory)
    identity = runtime_identity()
    validate_reference(reference, protocol, identity)
    output.mkdir(parents=True, exist_ok=False)
    (output / "cases").mkdir()
    shutil.copyfile(PROTOCOL, output / "protocol.json")
    shutil.copyfile(old.PROTOCOL, output / "inherited-protocol.json")
    shutil.copyfile(old.INVENTORY, output / "inventory.json")
    old.write_json(output / "runtime.json", identity)
    started = time.perf_counter()

    def launch(item):
        entry, arm = item
        directory = case_path(output, entry["id"], arm)
        log = directory.with_suffix(".log")
        command = [sys.executable, "-m", "glassbox.experimental.jsbsim_excitation",
                   "case", "--root", str(root), "--out", str(directory),
                   "--reference", str(reference), "--entry", entry["id"], "--arm", arm]
        env = dict(os.environ, JSBSIM_DISPERSE="0", OMP_NUM_THREADS="1",
                   OPENBLAS_NUM_THREADS="1", PYTHONPATH=str(REPO / "src")
                   + os.pathsep + os.environ.get("PYTHONPATH", ""))
        code, timeout = None, False
        with (log.open("wb") as stream,
              tempfile.TemporaryDirectory(prefix="glassbox-excitation-worker-") as cwd):
            try:
                result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                        timeout=CASE_TIMEOUT_S, env=env, cwd=cwd, check=False)
                code = result.returncode
            except subprocess.TimeoutExpired:
                timeout = True
        if not timeout and (code is None or code > 0
                            or (code == 0 and not (directory / "case.json").exists())):
            raise RuntimeError(f"harness worker failed: {entry['id']} / {arm}; inspect {log}")
        if timeout or code < 0:
            directory.mkdir(exist_ok=True)
            case, arrays = old.restore_checkpoint(directory, entry)
            case.update(arm=arm, execution_status="timeout" if timeout else "worker_failure",
                        returncode=code, timeout_s=CASE_TIMEOUT_S, unsaved_tail="unknown")
            refcase, refarrays = reference_case_at(reference, entry)
            if not case["branches"]:
                names = refcase["runs"]["parent"].get("command_names", [])
                case["branches"] = old.branch_roster(len(names))
            case["computed"] = validate_arm(case, arrays, protocol, inherited, entry,
                                            refcase, refarrays)
            _finish_case(directory, case, arrays)
            return case
        return json.loads((directory / "case.json").read_text())

    items = [(entry, arm) for entry in inventory["entries"] for arm in protocol["arms"]]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        cases = list(pool.map(launch, items))
    report = summarize(cases, inventory, protocol)
    old.write_json(output / "report.json", report)
    old.write_json(output / "execution.json", dict(elapsed_s=time.perf_counter() - started,
                   workers=workers, instrumentation="unpaced diagnostic"))
    old.write_json(output / "run.json", dict(protocol_sha256=PROTOCOL_SHA,
                   inventory_sha256=old.INVENTORY_SHA,
                   reference_run_sha256=protocol["reference"]["run_sha256"],
                   files=_sealed_files(output)))
    return report


def validate_bundle(root, output, reference):
    from .jsbsim_excitation_audit import summarize, validate_arm

    protocol, inherited, inventory = load_spec()
    old.check_assets(root, inventory)
    identity = runtime_identity()
    validate_reference(reference, protocol, identity)
    seal = json.loads((output / "run.json").read_text())
    if (seal["protocol_sha256"] != PROTOCOL_SHA
            or seal["inventory_sha256"] != old.INVENTORY_SHA
            or seal["reference_run_sha256"] != protocol["reference"]["run_sha256"]):
        raise ValueError("evidence protocol/reference identity changed")
    if _sealed_files(output) != seal["files"]:
        raise ValueError("sealed evidence files differ")
    for name, sha in (("protocol.json", PROTOCOL_SHA),
                      ("inherited-protocol.json", old.PROTOCOL_SHA),
                      ("inventory.json", old.INVENTORY_SHA)):
        if old.digest(output / name) != sha:
            raise ValueError("copied frozen input changed")
    if json.loads((output / "runtime.json").read_text()) != identity:
        raise ValueError("runtime/source identity changed")
    expected = {case_path(output, e["id"], a).name
                for e in inventory["entries"] for a in protocol["arms"]}
    if {p.name for p in (output / "cases").iterdir() if p.is_dir()} != expected:
        raise ValueError("case directory roster changed")
    cases = []
    for entry in inventory["entries"]:
        refcase, refarrays = reference_case_at(reference, entry)
        for arm in protocol["arms"]:
            case, arrays = old.read_case(case_path(output, entry["id"], arm))
            if case["id"] != entry["id"] or case["arm"] != arm:
                raise ValueError("case identity changed")
            computed = validate_arm(case, arrays, protocol, inherited, entry, refcase, refarrays)
            if computed != case["computed"]:
                raise ValueError("saved case metrics differ from recomputation")
            cases.append(case)
    if summarize(cases, inventory, protocol) != json.loads((output / "report.json").read_text()):
        raise ValueError("saved summary differs from recomputation")
    return cases


def compare_fresh(output, replay, cases, replayed):
    comparisons = []
    for saved, fresh in zip(cases, replayed, strict=True):
        entry_id, arm = saved["id"], saved["arm"]
        record = dict(id=entry_id, arm=arm)
        if "execution_status" in saved or "execution_status" in fresh:
            if saved.get("execution_status") != fresh.get("execution_status"):
                raise ValueError(f"execution outcome changed: {entry_id}/{arm}")
            record.update(integrity_unverified=True, exact_recorded_outcome=False,
                          original=saved.get("execution_status"), fresh=fresh.get("execution_status"))
        else:
            _, a = old.read_case(case_path(output, entry_id, arm))
            _, b = old.read_case(case_path(replay, entry_id, arm))
            if set(a) != set(b) or any(not bitwise_equal(a[k], b[k]) for k in a):
                raise ValueError(f"fresh simulator replay differs: {entry_id}/{arm}")
            if ({k: v for k, v in saved.items() if k != "elapsed_s"}
                    != {k: v for k, v in fresh.items() if k != "elapsed_s"}):
                raise ValueError(f"fresh metadata/diagnostic replay differs: {entry_id}/{arm}")
            record.update(exact_recorded_outcome=True, arrays_checked=len(a))
        comparisons.append(record)
    return dict(integrity_verified=not any(x.get("integrity_unverified") for x in comparisons),
                cases=comparisons, model_qualification=False, flight_qualification=False)


def verify(root, output, reference, workers=4, expected_run_sha256=None):
    if expected_run_sha256 is None or old.digest(output / "run.json") != expected_run_sha256:
        raise ValueError("trusted external run seal required and must match")
    cases = validate_bundle(root, output, reference)
    with tempfile.TemporaryDirectory(prefix="glassbox-excitation-replay-") as temporary:
        replay = Path(temporary) / "replay"
        run_all(root, replay, reference, workers)
        replayed = validate_bundle(root, replay, reference)
        return compare_fresh(output, replay, cases, replayed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("case", "run", "validate", "verify"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--entry")
    parser.add_argument("--arm", choices=("as_shipped", "engine_bootstrap"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seal-sha256")
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error("workers must be between1 and16")
    root, output, reference = args.root.resolve(), args.out.resolve(), args.reference.resolve()
    if args.mode == "case":
        protocol, inherited, inventory = load_spec()
        entry = next(e for e in inventory["entries"] if e["id"] == args.entry)
        refcase, refarrays = reference_case_at(reference, entry)
        case, _ = execute_case(root, entry, args.arm, protocol, inherited, refcase, refarrays, output)
        result = dict(id=case["id"], arm=case["arm"], completed=True)
    elif args.mode == "run":
        result = run_all(root, output, reference, args.workers)
    elif args.mode == "validate":
        result = dict(validated_cases=len(validate_bundle(root, output, reference)))
    else:
        result = verify(root, output, reference, args.workers, args.seal_sha256)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
