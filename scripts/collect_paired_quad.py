"""Collect one 100 Hz Crazyflow tape with physically held 50 ms commands."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/harness/paired-quad-sampling-v1.json"
GIT = "/opt/homebrew/Caskroom/miniconda/base/bin/git"


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def save(path, **values):
    temporary = Path(str(path) + ".partial")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **values)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def seal(root):
    write(root / "manifest.json", {
        "format": "glassbox-paired-quad-sampling-v1",
        "files": {
            str(path.relative_to(root)): digest(path)
            for path in sorted(root.rglob("*")) if path.is_file() and path != root / "manifest.json"
        },
    })
    return digest(root / "manifest.json")


class HeldPlant:
    """Present a 20 Hz policy interface while observing the 100 Hz physical plant."""

    def __init__(self, plant, deadline):
        self.plant = plant
        self.deadline = deadline
        self.states = []
        self.times = []
        self.commands = []
        self.applied = []
        self.requested_macro = []

    def __getattr__(self, name):
        return getattr(self.plant, name)

    @property
    def sample_period_s(self):
        return 0.05

    def reset(self, *args, **kwargs):
        sample = self.plant.reset(*args, **kwargs)
        self.states = [sample.state.copy()]
        self.times = [sample.time_s]
        self.commands = []
        self.applied = [sample.applied_motor_thrust_fraction.copy()]
        self.requested_macro = []
        return sample

    def step(self, command):
        if time.monotonic() > self.deadline:
            raise TimeoutError("frozen collection wall limit exceeded")
        held = np.asarray(command, dtype=np.float64)
        assert held.shape == (4,) and np.isfinite(held).all()
        self.requested_macro.append(held.copy())
        for _ in range(5):
            sample = self.plant.step(held)
            self.commands.append(held.copy())
            self.states.append(sample.state.copy())
            self.times.append(sample.time_s)
            self.applied.append(sample.applied_motor_thrust_fraction.copy())
        return sample

    def arrays(self):
        return dict(
            states=np.asarray(self.states),
            time_s=np.asarray(self.times),
            commands=np.asarray(self.commands),
            applied=np.asarray(self.applied),
            requested_macro=np.asarray(self.requested_macro).reshape(-1, 4),
        )


def source_binding(spec):
    repository = Path(spec["behavior_source"]["repository"])
    commit = subprocess.check_output([GIT, "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    assert commit == spec["behavior_source"]["commit"]
    changed = subprocess.check_output(
        [GIT, "-C", str(repository), "diff", "HEAD", "--name-only", "--", "src", "pyproject.toml", "uv.lock"],
        text=True,
    ).splitlines()
    assert not changed, changed
    assert importlib.metadata.version("crazyflow") == spec["behavior_source"]["crazyflow_version"]
    return dict(
        repository=str(repository), commit=commit,
        crazyflow_version=importlib.metadata.version("crazyflow"),
        script_sha256=digest(Path(__file__)), protocol_sha256=digest(PROTOCOL),
        python=platform.python_version(),
    )


def paired(fine):
    states, commands, times = (fine[key] for key in ("states", "commands", "time_s"))
    assert len(commands) % 5 == 0 and len(states) == len(commands) + 1
    assert np.array_equal(commands.reshape(-1, 5, 4), np.broadcast_to(
        commands[::5, None], (len(commands) // 5, 5, 4)
    ))
    assert np.array_equal(commands[::5], fine["requested_macro"])
    assert np.allclose(times, np.arange(len(times)) * 0.01, rtol=0, atol=1e-7)
    coarse = dict(states=states[::5], commands=commands[::5], time_s=times[::5])
    assert np.array_equal(coarse["states"], fine["states"][::5])
    assert np.allclose(coarse["time_s"], np.arange(len(coarse["time_s"])) * 0.05, rtol=0, atol=1e-7)
    return coarse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = read(PROTOCOL)
    bound = source_binding(spec)
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(PROTOCOL, args.output / "protocol.json")
    write(args.output / "binding.json", bound)
    started = time.monotonic()
    held = None
    plant = None
    try:
        from glassbox_throw import plant as plants, throw_study as study

        assert Path(plants.__file__).resolve() == Path(bound["repository"]) / "src/glassbox_throw/plant.py"
        plant = plants.CrazyflowPlant(plants.CrazyflowPlantConfig(control_frequency_hz=100))
        held = HeldPlant(plant, started + spec["budget"]["maximum_wall_s_collection"])
        original = study.CRAZYFLOW_THROW_STUDY_CASES[0]
        case = replace(
            original, name="paired_held_50ms_arm125",
            scenario=replace(original.scenario, arm_length_ratio=1.25),
        )
        record, _telemetry, requested, identification, _trace = study._fly_trial(
            case, study.DUAL_CONTROL_PASS6_MODEL, held
        )
        fine = held.arrays()
        assert np.array_equal(fine["requested_macro"], requested)
        coarse = paired(fine)
        save(args.output / "fine.npz", **fine)
        save(args.output / "coarse.npz", **coarse)
        write(args.output / "result.json", {
            "status": "complete", "fine_intervals": len(fine["commands"]),
            "coarse_intervals": len(coarse["commands"]),
            "duration_s": float(fine["time_s"][-1]),
            "floor_contact_step_20hz": record.floor_contact_step,
            "configuration_change_step_20hz": record.configuration_change_step,
            "working_identifier_intervals": identification["working_interval_count"],
            "wall_s": time.monotonic() - started,
            "hover_motor_thrust_fraction": plant.hover_motor_thrust_fraction,
        })
        print("complete", read(args.output / "result.json"), flush=True)
    except BaseException as error:
        if held is not None and held.states:
            save(args.output / "partial.npz", **held.arrays())
        write(args.output / "failure.json", {"error": repr(error), "wall_s": time.monotonic() - started})
        raise
    finally:
        if plant is not None:
            plant.close()
        print("manifest_sha256", seal(args.output), flush=True)


if __name__ == "__main__":
    main()
