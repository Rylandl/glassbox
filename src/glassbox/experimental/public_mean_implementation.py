"""Bind clean source revisions before public-mean qualification execution.

This is a source/runtime inventory, not a numerical or scientific result. Both
the public worker and isolated historical worker consume the same external seal.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from .public_mean_scoring import PROTOCOL_SHA256

ORACLE_COMMIT = "8b61830c9c353dbb25edc6e63a76886d0ce9b9d3"
PROTOCOL_PATH = "docs/harness/public-mean-qualification-v1.json"
CONSUMER_SNAPSHOT = "docs/harness/public-mean-qualification-v1-consumer"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def _clean(root):
    if _git(root, "status", "--porcelain", "--untracked-files=normal"):
        raise ValueError(f"qualification source checkout is not clean: {root}")
    return _git(root, "rev-parse", "HEAD")


def _sources(root):
    tracked = _git(
        root, "ls-files", "src/glassbox", "pyproject.toml", "tests"
    ).splitlines()
    return {
        name: digest(root / name) for name in tracked if name.endswith((".py", ".toml"))
    }


def bind(output, *, public_root, oracle_root, dart_root):
    """Write an exclusive source/runtime seal from committed, immutable inputs."""
    public_root, oracle_root, dart_root = (
        Path(p).resolve() for p in (public_root, oracle_root, dart_root)
    )
    implementation_commit = _clean(public_root)
    if (
        Path(__file__).resolve()
        != public_root / "src/glassbox/experimental/public_mean_implementation.py"
    ):
        raise ValueError("implementation binder was imported from another checkout")
    if _clean(oracle_root) != ORACLE_COMMIT:
        raise ValueError("historical oracle checkout is not the frozen source commit")
    protocol_file = public_root / PROTOCOL_PATH
    if digest(protocol_file) != PROTOCOL_SHA256:
        raise ValueError("qualification protocol changed")
    protocol = json.loads(protocol_file.read_text())
    for entry in protocol["protocol_dependencies"]:
        if digest(public_root / entry["path"]) != entry["sha256"]:
            raise ValueError("qualification dependency changed: " + entry["path"])
    runtime = {
        "python": ".".join(map(str, sys.version_info[:3])),
        **{
            name: importlib.metadata.version(name)
            for name in ("jax", "jaxlib", "numpy", "scipy")
        },
    }
    if any(runtime[name] != protocol["runtime"][name] for name in runtime):
        raise ValueError(
            "qualification runtime versions differ from the frozen protocol"
        )
    if platform.machine() != "arm64":
        raise ValueError("qualification requires its declared arm64 CPU host")
    consumer_sources = {}
    snapshot_root = public_root / CONSUMER_SNAPSHOT
    snapshot_files = sorted(snapshot_root.rglob("*.py"))
    if {str(path.relative_to(snapshot_root)) for path in snapshot_files} != {
        "glassbox_forecast.py",
        "test_glassbox_forecast.py",
    }:
        raise ValueError("committed Dart consumer source/test snapshot roster differs")
    tracked = set(_git(public_root, "ls-files", CONSUMER_SNAPSHOT).splitlines())
    for snapshot in snapshot_files:
        if str(snapshot.relative_to(public_root)) not in tracked:
            raise ValueError("consumer snapshot is not committed")
        relative = {
            "glassbox_forecast.py": "src/crazydart/glassbox_forecast.py",
            "test_glassbox_forecast.py": "tests/test_glassbox_forecast.py",
        }[str(snapshot.relative_to(snapshot_root))]
        live = dart_root / relative
        if not live.is_file() or digest(live) != digest(snapshot):
            raise ValueError(
                "Dart consumer differs from its committed snapshot: " + relative
            )
        consumer_sources[relative] = digest(live)
    if "src/crazydart/glassbox_forecast.py" not in consumer_sources:
        raise ValueError("Dart consumer implementation is absent from its snapshot")
    # Preserve the already inspected consumer project: unrelated source changes
    # are an explicit provenance finding, not silently accepted as the old state.
    base_path = (
        public_root
        / "docs/harness/public-mean-qualification-v1/public-v4-consumer-source-anchors.json"
    )
    for relative, expected in json.loads(base_path.read_text())["dart_sha256"].items():
        if digest(dart_root / relative) != expected:
            raise ValueError("pre-existing Dart source changed: " + relative)
        consumer_sources[relative] = expected
    result = {
        "format": "glassbox-public-mean-implementation-v1",
        "protocol_sha256": PROTOCOL_SHA256,
        "implementation_commit": implementation_commit,
        "oracle_commit": ORACLE_COMMIT,
        "public_root": str(public_root),
        "oracle_root": str(oracle_root),
        "dart_root": str(dart_root),
        "interpreter": sys.executable,
        "interpreter_sha256": digest(Path(sys.executable).resolve()),
        "runtime": runtime,
        "machine": platform.machine(),
        "platform": platform.platform(),
        "public_source_sha256": _sources(public_root),
        "oracle_source_sha256": _sources(oracle_root),
        "consumer_source_sha256": consumer_sources,
        "consumer_distribution_versions": {
            name: importlib.metadata.version(name) for name in ("glassbox", "crazydart")
        },
        "consumer_prefreeze_inventory_sha256": digest(base_path),
        "consumer_exporter_sha256": digest(
            public_root / "src/glassbox/experimental/public_mean_consumer_export.py"
        ),
        "scientific_operations_executed": 0,
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        handle.write(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
    return {"path": str(output), "sha256": digest(output)}


def verify(path, expected_sha256):
    """Authenticate all bound executable files before any qualification stage."""
    path = Path(path)
    if digest(path) != expected_sha256:
        raise ValueError("implementation seal differs from its external anchor")
    value = json.loads(path.read_text())
    if (
        value["format"] != "glassbox-public-mean-implementation-v1"
        or value["protocol_sha256"] != PROTOCOL_SHA256
    ):
        raise ValueError("unsupported implementation seal")
    for root_key, sources_key, commit_key in (
        ("public_root", "public_source_sha256", "implementation_commit"),
        ("oracle_root", "oracle_source_sha256", "oracle_commit"),
    ):
        root = Path(value[root_key])
        if _clean(root) != value[commit_key]:
            raise ValueError("implementation commit changed")
        for relative, sha in value[sources_key].items():
            if digest(root / relative) != sha:
                raise ValueError("bound implementation source changed: " + relative)
    for relative, sha in value["consumer_source_sha256"].items():
        if digest(Path(value["dart_root"]) / relative) != sha:
            raise ValueError("bound Dart consumer source changed: " + relative)
    if (
        Path(sys.executable).resolve() != Path(value["interpreter"]).resolve()
        or digest(Path(sys.executable).resolve()) != value["interpreter_sha256"]
    ):
        raise ValueError("bound qualification interpreter changed")
    runtime = {
        "python": ".".join(map(str, sys.version_info[:3])),
        **{
            name: importlib.metadata.version(name)
            for name in ("jax", "jaxlib", "numpy", "scipy")
        },
    }
    if runtime != value["runtime"] or platform.machine() != value["machine"]:
        raise ValueError("bound qualification runtime changed")
    return value


def consumer_environment(binding):
    """Explicit version boundary for a fresh Dart process, with default precision."""
    environment = os.environ.copy()
    environment.pop("JAX_ENABLE_X64", None)
    environment["SCIPY_ARRAY_API"] = "1"
    environment["PYTHONPATH"] = os.pathsep.join(
        (
            str(Path(binding["public_root"]) / "src"),
            str(Path(binding["dart_root"]) / "src"),
        )
    )
    return environment


def consumer_preflight(binding_path, expected_sha256, output):
    """Verify actual public imports in a fresh Dart process before any forecast."""
    binding = verify(binding_path, expected_sha256)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    environment = consumer_environment(binding)
    code = "import json; from crazydart.glassbox_forecast import observed_runtime; print(json.dumps(observed_runtime(), sort_keys=True))"
    command = [binding["interpreter"], "-c", code]
    (output / "command.json").write_text(
        json.dumps(
            {
                "command": command,
                "cwd": binding["dart_root"],
                "implementation_sha256": expected_sha256,
                "environment": {
                    key: environment.get(key)
                    for key in ("PYTHONPATH", "JAX_ENABLE_X64", "SCIPY_ARRAY_API")
                },
            },
            indent=2,
        )
        + "\n"
    )
    process = subprocess.run(
        command,
        cwd=binding["dart_root"],
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    (output / "stdout.txt").write_text(process.stdout)
    (output / "stderr.txt").write_text(process.stderr)
    (output / "exit.json").write_text(
        json.dumps({"returncode": process.returncode}) + "\n"
    )
    if process.returncode:
        raise ValueError("consumer source preflight failed; output retained")
    actual = json.loads(process.stdout)
    if (
        actual["jax_enable_x64"]
        or Path(actual["interpreter"]).resolve()
        != Path(binding["interpreter"]).resolve()
        or actual["interpreter_sha256"] != binding["interpreter_sha256"]
    ):
        raise ValueError("consumer interpreter/precision does not match bound runtime")
    versions = {name: binding["runtime"][name] for name in ("numpy", "jax", "jaxlib")}
    versions.update(binding["consumer_distribution_versions"])
    if actual["versions"] != versions:
        raise ValueError("consumer distribution versions differ")
    for name, identity in actual["imports"].items():
        if name == "glassbox" or name.startswith("glassbox."):
            root, sources = (
                Path(binding["public_root"]),
                binding["public_source_sha256"],
            )
        elif name in ("crazydart", "crazydart.glassbox_forecast"):
            root, sources = (
                Path(binding["dart_root"]),
                binding["consumer_source_sha256"],
            )
        else:
            raise ValueError("consumer preflight returned an undeclared import")
        path = Path(identity["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("consumer import escaped its bound source root")
        relative = str(path.relative_to(root))
        if (
            relative not in sources
            or identity["sha256"] != sources[relative]
            or digest(path) != sources[relative]
        ):
            raise ValueError("consumer import does not match committed source")
    expected_modules = {
        "glassbox",
        "glassbox.learner",
        "glassbox._sequence_model",
        "glassbox._learner_arrays",
        "glassbox.recordings",
        "glassbox.io",
        "glassbox.io.recordings",
        "crazydart",
        "crazydart.glassbox_forecast",
    }
    if set(actual["imports"]) != expected_modules:
        raise ValueError("consumer preflight import roster differs")
    verify(binding_path, expected_sha256)
    (output / "verified.json").write_text(
        json.dumps({"status": "passed", "runtime": actual}, indent=2, sort_keys=True)
        + "\n"
    )
    return actual
