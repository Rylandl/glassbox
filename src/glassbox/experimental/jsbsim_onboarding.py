"""Frozen no-fit JSBSim onboarding experiment; not part of the learner API.

Install JSBSim 1.3.1 separately and supply its pinned release data directory.
Run and verify produce sealed evidence, preserving every configuration failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

PROTOCOL_SHA = "0680272443996f13e6d433d1e7ebd689697c126d7ce483466fd78636b59152be"
INVENTORY_SHA = "26c96553dff5ace218599b525208e0fe42abafa935350d4dbc37215ebd929d9e"
REPO = Path(__file__).resolve().parents[3]
PROTOCOL = REPO / "docs/harness/jsbsim-onboarding-v1.json"
INVENTORY = REPO / "docs/harness/jsbsim-onboarding-v1-inventory.json"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def load_spec():
    if digest(PROTOCOL) != PROTOCOL_SHA or digest(INVENTORY) != INVENTORY_SHA:
        raise ValueError("frozen protocol or inventory changed")
    return json.loads(PROTOCOL.read_text()), json.loads(INVENTORY.read_text())


def discover(root):
    """Rediscover configurations and IC choices rather than trusting their manifest."""
    entries = []
    for path in sorted((root / "aircraft").rglob("*.xml")):
        node = ET.parse(path).getroot()
        if node.tag.lower() != "fdm_config":
            continue
        initializations = []
        for candidate in sorted(path.parent.glob("*.xml")):
            initial = ET.parse(candidate).getroot()
            if initial.tag.lower() == "initialize":
                initializations.append(
                    dict(
                        path=str(candidate.relative_to(root)),
                        requests_trim=any(
                            e.tag.lower() == "trim" for e in initial.iter()
                        ),
                        engine_running_tags=[
                            e.text.strip() if e.text else ""
                            for e in initial.iter()
                            if e.tag.lower() == "running"
                        ],
                    )
                )
        usable = sorted(
            (x for x in initializations if not x["requests_trim"]),
            key=lambda x: (Path(x["path"]).name != "reset00.xml", x["path"]),
        )
        entries.append(
            dict(
                id=str(path.relative_to(root / "aircraft")),
                path=str(path.relative_to(root)),
                root_tag=node.tag,
                initializations=initializations,
                selected_initialization=usable[0]["path"] if usable else None,
            )
        )
    return entries


def check_assets(root, inventory):
    actual = {}
    for part in ("aircraft", "engine", "systems"):
        for path in sorted((root / part).rglob("*")):
            if path.is_symlink():
                raise ValueError(f"unpinned asset symlink: {path}")
            if path.is_file():
                actual[str(path.relative_to(root))] = digest(path)
    if actual != inventory["assets"]:
        raise ValueError("release assets differ from frozen exact file roster")
    directories = [p.name for p in sorted((root / "aircraft").iterdir()) if p.is_dir()]
    if (
        directories != inventory["directories"]
        or discover(root) != inventory["entries"]
    ):
        raise ValueError("rediscovered inventory/initialization differs from freeze")


def runtime_identity():
    import jsbsim
    import numpy._core._multiarray_umath as native_numpy

    if jsbsim.__version__ != "1.3.1":
        raise ValueError("JSBSim 1.3.1 is required")
    package = Path(jsbsim.__file__).parent
    files = {
        str(path.relative_to(package)): digest(path)
        for path in sorted(package.rglob("*"))
        if path.is_file() and path.suffix in (".py", ".so", ".pyd", ".dylib")
    }
    sources = {
        str(path.relative_to(REPO)): digest(path)
        for path in sorted((REPO / "src/glassbox").rglob("*.py"))
    }
    return dict(
        python=sys.version,
        python_binary_sha256=digest(Path(sys.executable).resolve()),
        numpy=np.__version__,
        numpy_native_sha256=digest(native_numpy.__file__),
        jsbsim=jsbsim.__version__,
        jsbsim_files=files,
        source_files=sources,
        platform=platform.platform(),
        machine=platform.machine(),
        protocol_sha256=PROTOCOL_SHA,
        inventory_sha256=INVENTORY_SHA,
        dispersion="0",
        simulator_seed=0,
    )


def case_slug(entry_id):
    return hashlib.sha256(entry_id.encode()).hexdigest()[:16]


def branch_roster(count):
    rows = [dict(label="factual", command_index=None, direction=None)]
    for channel in range(count):
        for direction in ("lower", "upper"):
            rows.append(
                dict(
                    label=f"c{channel:03d}_{direction}",
                    command_index=channel,
                    direction=direction,
                )
            )
    return rows


def checkpoint(output, case, arrays):
    """One atomic archive binds partial arrays to their matching metadata."""
    temporary = output / "checkpoint.tmp"
    metadata = np.frombuffer(
        json.dumps(case, sort_keys=True, allow_nan=False).encode(), dtype=np.uint8
    )
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, __case_json__=metadata, **arrays)
    temporary.replace(output / "checkpoint.npz")


def restore_checkpoint(output, entry):
    path = output / "checkpoint.npz"
    if not path.exists():
        return dict(id=entry["id"], runs={}, branches=[]), {}
    with np.load(path, allow_pickle=False) as saved:
        case = json.loads(saved["__case_json__"].tobytes())
        arrays = {k: saved[k] for k in saved.files if k != "__case_json__"}
    if (output / "progress.json").exists():
        case["interrupted_progress"] = json.loads(
            (output / "progress.json").read_text()
        )
    return case, arrays


def simulate_case(root, entry, protocol, output):
    from .jsbsim_adapter import simulate
    from .jsbsim_onboarding_audit import validate_case

    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    arrays, runs = {}, {}
    case = dict(id=entry["id"], runs=runs, branches=[])
    checkpoint(output, case, arrays)

    def retain(label, result):
        result = dict(result)
        for name, values in result.pop("arrays").items():
            arrays[f"{label}__{name}"] = np.asarray(values)
        runs[label] = result
        if label == "parent":
            roster = branch_roster(len(result.get("command_names", [])))
            case["branches"] = roster
            for planned in ["replay", *(x["label"] for x in roster)]:
                runs.setdefault(planned, dict(status="unattempted"))
        return result

    def invoke(label, commands):
        (output / "progress.json").unlink(missing_ok=True)
        runs[label] = dict(status="running", stage="construct")
        checkpoint(output, case, arrays)

        def progress(value):
            if value.get("progress_only"):
                temporary = output / "progress.tmp"
                write_json(temporary, dict(label=label, **value))
                temporary.replace(output / "progress.json")
            else:
                (output / "progress.json").unlink(missing_ok=True)
                retain(label, value)
                checkpoint(output, case, arrays)

        result = simulate(root, entry, commands, protocol, progress=progress)
        (output / "progress.json").unlink(missing_ok=True)
        result = retain(label, result)
        checkpoint(output, case, arrays)
        return result

    parent = invoke("parent", None)
    if parent["status"] == "completed":
        tape = arrays["parent__commands"]
        invoke("replay", tape)
        origin = protocol["replay"]["branch_origin"]
        horizon = protocol["replay"]["horizon_steps"]
        base = tape[: origin + horizon]
        bounds = np.asarray(parent["bounds"])
        for branch in case["branches"]:
            commands = base.copy()
            channel = branch["command_index"]
            if channel is not None:
                bound = bounds[channel, int(branch["direction"] == "upper")]
                commands[origin:, channel] += 0.1 * (bound - commands[origin:, channel])
            invoke(branch["label"], commands)
    case["elapsed_s"] = time.perf_counter() - started
    case["computed"] = validate_case(case, arrays, protocol, entry)
    np.savez_compressed(output / "arrays.npz", **arrays)
    write_json(output / "case.json", case)
    for name in ("checkpoint.npz", "checkpoint.tmp", "progress.json", "progress.tmp"):
        (output / name).unlink(missing_ok=True)
    return case


def execute(root, output, workers=4):
    from .jsbsim_onboarding_audit import summarize, validate_case

    protocol, inventory = load_spec()
    check_assets(root, inventory)
    identity = runtime_identity()
    output.mkdir(parents=True, exist_ok=False)
    cases_dir = output / "cases"
    cases_dir.mkdir()
    shutil.copyfile(PROTOCOL, output / "protocol.json")
    shutil.copyfile(INVENTORY, output / "inventory.json")
    write_json(output / "runtime.json", identity)
    started = time.perf_counter()

    def launch(entry):
        slug = case_slug(entry["id"])
        case_dir = cases_dir / slug
        log = cases_dir / f"{slug}.log"
        command = [
            sys.executable,
            "-m",
            "glassbox.experimental.jsbsim_onboarding",
            "case",
            "--root",
            str(root),
            "--out",
            str(case_dir),
            "--entry",
            entry["id"],
        ]
        env = dict(
            os.environ,
            JSBSIM_DISPERSE="0",
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            PYTHONPATH=str(REPO / "src")
            + os.pathsep
            + os.environ.get("PYTHONPATH", ""),
        )
        code = None
        timeout = False
        with (
            log.open("wb") as stream,
            tempfile.TemporaryDirectory(
                prefix="glassbox-jsbsim-worker-"
            ) as worker_directory,
        ):
            try:
                completed = subprocess.run(
                    command,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    timeout=protocol["recording"]["case_timeout_s"],
                    env=env,
                    cwd=worker_directory,
                    check=False,
                )
                code = completed.returncode
            except subprocess.TimeoutExpired:
                timeout = True
        result_path = case_dir / "case.json"
        if not timeout and (
            code is None or code > 0 or (code == 0 and not result_path.exists())
        ):
            raise RuntimeError(
                f"harness worker failed for {entry['id']}; inspect {log}"
            )
        if timeout or code < 0:
            # A crashed native engine must not erase the inventory entry or poison siblings.
            case_dir.mkdir(exist_ok=True)
            case, arrays = restore_checkpoint(case_dir, entry)
            case.update(
                execution_status="timeout" if timeout else "worker_failure",
                returncode=code,
                timeout_s=protocol["recording"]["case_timeout_s"],
            )
            case["computed"] = validate_case(case, arrays, protocol, entry)
            np.savez_compressed(case_dir / "arrays.npz", **arrays)
            write_json(result_path, case)
            return case
        return json.loads(result_path.read_text())

    with ThreadPoolExecutor(max_workers=workers) as pool:
        cases = list(pool.map(launch, inventory["entries"]))
    report = summarize(cases, inventory)
    write_json(output / "report.json", report)
    write_json(
        output / "execution.json",
        dict(
            elapsed_s=time.perf_counter() - started,
            workers=workers,
            instrumentation="unpaced diagnostic",
        ),
    )
    files = {
        str(p.relative_to(output)): digest(p)
        for p in sorted(output.rglob("*"))
        if p.is_file()
    }
    write_json(
        output / "run.json",
        dict(protocol_sha256=PROTOCOL_SHA, inventory_sha256=INVENTORY_SHA, files=files),
    )
    return report


def read_case(directory):
    case = json.loads((directory / "case.json").read_text())
    arrays = {}
    if (directory / "arrays.npz").exists():
        with np.load(directory / "arrays.npz", allow_pickle=False) as saved:
            arrays = {k: saved[k] for k in saved.files}
    return case, arrays


def validate_bundle(root, output):
    from .jsbsim_onboarding_audit import summarize, validate_case

    protocol, inventory = load_spec()
    check_assets(root, inventory)
    seal = json.loads((output / "run.json").read_text())
    if (
        seal["protocol_sha256"] != PROTOCOL_SHA
        or seal["inventory_sha256"] != INVENTORY_SHA
    ):
        raise ValueError("evidence protocol identity changed")
    actual = {
        str(p.relative_to(output)): digest(p)
        for p in sorted(output.rglob("*"))
        if p.is_file() and p != output / "run.json"
    }
    if actual != seal["files"]:
        raise ValueError("sealed evidence files differ")
    if (
        digest(output / "protocol.json") != PROTOCOL_SHA
        or digest(output / "inventory.json") != INVENTORY_SHA
    ):
        raise ValueError("copied frozen inputs changed")
    if json.loads((output / "runtime.json").read_text()) != runtime_identity():
        raise ValueError("runtime/source identity changed")
    cases = []
    expected_dirs = {case_slug(e["id"]) for e in inventory["entries"]}
    if {p.name for p in (output / "cases").iterdir() if p.is_dir()} != expected_dirs:
        raise ValueError("case directory roster changed")
    for entry in inventory["entries"]:
        case, arrays = read_case(output / "cases" / case_slug(entry["id"]))
        if case["id"] != entry["id"]:
            raise ValueError("case identity changed")
        computed = validate_case(case, arrays, protocol, entry)
        if case["computed"] != computed:
            raise ValueError("saved case metrics differ from independent recomputation")
        cases.append(case)
    recomputed = summarize(cases, inventory)
    if json.loads((output / "report.json").read_text()) != recomputed:
        raise ValueError("saved summary differs from independent recomputation")
    return cases


def verify(root, output, workers=4, expected_run_sha256=None):
    from .jsbsim_onboarding_audit import bitwise_equal

    if (
        expected_run_sha256 is None
        or digest(output / "run.json") != expected_run_sha256
    ):
        raise ValueError("trusted external run seal required and must match")
    cases = validate_bundle(root, output)
    comparisons = []
    with tempfile.TemporaryDirectory(prefix="glassbox-jsbsim-replay-") as temporary:
        replay = Path(temporary) / "replay"
        execute(root, replay, workers)
        replayed = validate_bundle(root, replay)
        for old, new in zip(cases, replayed, strict=True):
            record = dict(id=old["id"])
            if "execution_status" in old or "execution_status" in new:
                record.update(
                    original=old.get("execution_status"),
                    fresh=new.get("execution_status"),
                    exact_physical_replay=False,
                )
                # Execution failures remain unavailable; equality of timeouts is not physics evidence.
                if old.get("execution_status") != new.get("execution_status"):
                    raise ValueError(
                        f"execution outcome changed and cannot be verified: {old['id']}"
                    )
                record["integrity_unverified"] = True
            else:
                slug = case_slug(old["id"])
                _, a = read_case(output / "cases" / slug)
                _, b = read_case(replay / "cases" / slug)
                if set(a) != set(b) or any(not bitwise_equal(a[k], b[k]) for k in a):
                    raise ValueError(f"fresh simulator replay differs: {old['id']}")
                # Time counters are descriptive and not replay evidence.
                old_core = {k: v for k, v in old.items() if k != "elapsed_s"}
                new_core = {k: v for k, v in new.items() if k != "elapsed_s"}
                if old_core != new_core:
                    raise ValueError(
                        f"fresh metadata/diagnostic replay differs: {old['id']}"
                    )
                record.update(
                    exact_physical_replay=old["runs"]["parent"]["status"]
                    == "completed",
                    exact_recorded_outcome=True,
                    arrays_checked=len(a),
                )
            comparisons.append(record)
    return dict(
        integrity_verified=not any(x.get("integrity_unverified") for x in comparisons),
        cases=comparisons,
        exact_physical_cases=sum(x["exact_physical_replay"] for x in comparisons),
        model_qualification=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("case", "run", "verify", "validate"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--entry")
    parser.add_argument(
        "--seal-sha256",
        help="Trusted published run.json digest required for full verify",
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.out.resolve()
    if not 1 <= args.workers <= 16:
        parser.error("workers must be between1 and16")
    if args.mode == "case":
        protocol, inventory = load_spec()
        entry = next(x for x in inventory["entries"] if x["id"] == args.entry)
        result = simulate_case(root, entry, protocol, output)
        print(
            json.dumps(
                dict(id=result["id"], computed=result["computed"]), allow_nan=False
            )
        )
    elif args.mode == "run":
        print(
            json.dumps(execute(root, output, args.workers), indent=2, allow_nan=False)
        )
    elif args.mode == "verify":
        print(
            json.dumps(
                verify(root, output, args.workers, args.seal_sha256),
                indent=2,
                allow_nan=False,
            )
        )
    else:
        print(
            json.dumps(
                dict(validated_cases=len(validate_bundle(root, output))),
                allow_nan=False,
            )
        )


if __name__ == "__main__":
    main()
