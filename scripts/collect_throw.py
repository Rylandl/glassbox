"""Collect the four frozen Throw tapes in the original, pinned behavior runtime."""

import argparse
import importlib.metadata
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from run_dart import ROOT, write
from verify_baseline import digest, read, require

PROTOCOL = ROOT / "docs/harness/online-fit-v1.json"
GIT = "git"


def git(root, *arguments):
    return subprocess.check_output(
        [GIT, "-C", str(root), *arguments], text=True
    ).strip()


def source_binding(root, *paths):
    require(
        not git(root, "diff", "HEAD", "--name-only", "--", *paths),
        "commit source before running",
    )
    require(
        not git(root, "ls-files", "--others", "--exclude-standard", "--", *paths),
        "track source before running",
    )
    names = git(root, "ls-files", "--", *paths).splitlines()
    require(bool(names), "empty source inventory")
    return dict(
        commit=git(root, "rev-parse", "HEAD"),
        files={name: digest(root / name) for name in names},
    )


def binding(protocol):
    import jax

    require(not jax.config.x64_enabled, "start in ambient float32")
    relative = str(protocol.relative_to(ROOT))
    git(ROOT, "ls-files", "--error-unmatch", relative)
    return dict(
        source=source_binding(ROOT, "src", "scripts", "pyproject.toml", relative),
        protocol_sha256=digest(protocol),
        runtime=dict(
            python=platform.python_version(),
            machine=platform.machine(),
            executable=str(Path(sys.executable).resolve()),
            interpreter_sha256=digest(Path(sys.executable).resolve()),
            backend=jax.default_backend(),
            x64_enabled=jax.config.x64_enabled,
            versions={
                n: importlib.metadata.version(n)
                for n in ("jax", "jaxlib", "numpy", "scipy")
            },
        ),
    )


def check_source(bound):
    require(
        all(
            digest(ROOT / name) == wanted
            for name, wanted in bound["source"]["files"].items()
        ),
        "source changed during execution",
    )


def seal(root, kind):
    write(
        root / "manifest.json",
        dict(
            format=kind,
            files={
                str(p.relative_to(root)): digest(p)
                for p in sorted(root.rglob("*"))
                if p.is_file() and p != root / "manifest.json"
            },
        ),
    )
    return digest(root / "manifest.json")


def authenticate(root, authority):
    require(digest(root / "manifest.json") == authority, "manifest authority differs")
    manifest = read(root / "manifest.json")
    require(
        set(manifest["files"])
        == {
            str(p.relative_to(root))
            for p in root.rglob("*")
            if p.is_file() and p != root / "manifest.json"
        },
        "artifact inventory differs",
    )
    for name, wanted in manifest["files"].items():
        relative = Path(name)
        require(
            not relative.is_absolute() and ".." not in relative.parts,
            "unsafe artifact path",
        )
        require(digest(root / name) == wanted, f"artifact hash differs: {name}")
    return manifest


def checkpoint(path, **values):
    """Replace only this attempt's own checkpoint, atomically after flushing it."""
    temporary = path.with_suffix(".partial")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **values)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def collect_case(case, output, plants, study):
    output.mkdir()
    print(json.dumps(dict(case=case["id"], status="collecting")), flush=True)
    write(output / "attempt.json", case)
    states, commands, timestamps = [], [], []
    plant, error, record = None, None, None
    started = time.perf_counter()

    def save():
        checkpoint(
            output / "stream.npz",
            states=np.asarray(states).reshape(-1, 13),
            commands=np.asarray(commands).reshape(-1, 4),
            time_s=np.asarray(timestamps),
        )

    try:
        plant = plants.CrazyflowPlant(
            plants.CrazyflowPlantConfig(control_frequency_hz=100)
        )
        reset, step = plant.reset, plant.step

        def recorded_reset(*args, **kwargs):
            sample = reset(*args, **kwargs)
            states.append(sample.state.copy())
            timestamps.append(sample.time_s)
            save()
            return sample

        def recorded_step(command):
            sample = step(command)
            commands.append(np.asarray(command).copy())
            states.append(sample.state.copy())
            timestamps.append(sample.time_s)
            if len(commands) % 25 == 0:
                save()
            if len(commands) % 100 == 0:
                print(
                    json.dumps(dict(case=case["id"], native_intervals=len(commands))),
                    flush=True,
                )
            return sample

        plant.reset, plant.step = recorded_reset, recorded_step
        canonical = study.CRAZYFLOW_THROW_STUDY_CASES[0]
        change = (
            study.ThrowStudyConfigurationChange(
                case["change_at_s"], case["changed_arm_ratio"]
            )
            if "change_at_s" in case
            else None
        )
        chosen = replace(
            canonical,
            name=case["id"],
            scenario=replace(canonical.scenario, arm_length_ratio=case["arm_ratio"]),
            configuration_change=change,
        )
        record, _telemetry, requested, _identification, _trace = study._fly_trial(
            chosen, study.DUAL_CONTROL_PASS6_MODEL, plant
        )
        require(np.array_equal(np.asarray(commands), requested), "issued tape differs")
    except Exception as caught:
        error = repr(caught)
    finally:
        save()
        if plant is not None:
            try:
                plant.close()
            except Exception as caught:
                error = f"{error or ''} shutdown: {caught!r}"
    result = dict(
        id=case["id"],
        status="complete" if error is None else "failed",
        error=error,
        intervals=len(commands),
        duration_s=timestamps[-1] if timestamps else 0,
        floor_contact_step=None if record is None else record.floor_contact_step,
        change_step=None if record is None else record.configuration_change_step,
        stop="exception"
        if error is not None
        else "floor_contact"
        if record.floor_contact_step is not None
        else "duration",
        truncated=bool(not timestamps or timestamps[-1] < 10.0 - 1e-10),
        wall_time_s=time.perf_counter() - started,
        ordered_commands=[f"command_{i} [1]" for i in range(4)],
        candidate_fields=["states", "commands", "time_s"],
    )
    write(output / "report.json", result)
    print(json.dumps(dict(case=case["id"], result=result)), flush=True)
    return result


def run(output, protocol=PROTOCOL):
    output.mkdir(parents=True, exist_ok=False)
    write(output / "attempt.json", dict(protocol=str(protocol)))
    try:
        p = read(protocol)
        bound = binding(protocol)
        q = p["streams"]["quad"]
        repository = Path(q["source_repository"])
        require(
            Path(sys.prefix).resolve() == (repository / ".venv").resolve(),
            "collect in the pinned original Throw virtualenv",
        )
        old = importlib.metadata.distribution("glassbox")
        direct = json.loads(old.read_text("direct_url.json"))
        require(
            direct["vcs_info"]["commit_id"] == q["behavior_glassbox_commit"],
            "behavior Glassbox revision differs",
        )
        require(
            importlib.metadata.version("crazyflow") == "0.3.2", "Crazyflow pin differs"
        )
        external = source_binding(
            repository, "src", "scripts", "pyproject.toml", "uv.lock"
        )
        require(
            external["commit"].startswith(q["source_commit"]), "Throw revision differs"
        )
        bound.update(throw_source=external, behavior_glassbox=direct)
        bound["packages"] = {}
        for name in ("glassbox", "crazyflow"):
            package = Path(importlib.util.find_spec(name).origin).parent
            bound["packages"][name] = dict(
                root=str(package),
                files={
                    str(path.relative_to(package)): digest(path)
                    for path in sorted(package.rglob("*"))
                    if path.suffix in (".py", ".toml", ".yaml", ".yml", ".json", ".xml")
                },
            )
        write(output / "binding.json", bound)
        import glassbox_throw.plant as plants
        import glassbox_throw.throw_study as study

        require(
            Path(plants.__file__).resolve()
            == repository / "src/glassbox_throw/plant.py",
            "unexpected Throw import",
        )
        results = [
            collect_case(case, output / case["id"], plants, study)
            for case in q["cases"]
        ]
        check_source(bound)
        require(
            all(
                digest(repository / name) == wanted
                for name, wanted in external["files"].items()
            ),
            "Throw source changed",
        )
        require(
            all(
                digest(Path(package["root"]) / name) == wanted
                for package in bound["packages"].values()
                for name, wanted in package["files"].items()
            ),
            "behavior package changed",
        )
        write(
            output / "report.json",
            dict(cases=results, candidate_fits=0, candidate_controls_plant=False),
        )
    except BaseException as error:
        write(output / "failure.json", dict(error=repr(error)))
        raise
    finally:
        authority = seal(output, "glassbox-online-collection-v1")
        print(
            json.dumps(dict(output=str(output), manifest_sha256=authority)), flush=True
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    args = parser.parse_args()
    run(args.output.resolve(), args.protocol.resolve())
