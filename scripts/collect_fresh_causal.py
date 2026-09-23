"""Collect two new Crazyflow flights using the pinned, independent Throw policy.

This script fixes the aircraft/release roster before any flight is generated.
Glassbox's candidate learner is never imported by this process.
"""

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from glassbox_throw import throw_study as study
from glassbox_throw.plant import CrazyflowPlant, CrazyflowPlantConfig

THROW_ROOT = Path("/Users/ryland/projects/glassbox-throw")
GLASSBOX_ROOT = Path(__file__).resolve().parents[1]
GIT = "/opt/homebrew/Caskroom/miniconda/base/bin/git"
CASES = (
    ("quad-arm-095-spin", 0.95, 2.4, (2.4, -1.8, 1.2)),
    ("quad-arm-155-canonical", 1.55, 1.2, (0.8, -0.6, 0.4)),
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(root, *arguments):
    return subprocess.check_output(
        [GIT, "-C", str(root), *arguments], text=True
    ).strip()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def binding():
    if not git(THROW_ROOT, "rev-parse", "HEAD").startswith("9bd74c1"):
        raise ValueError("Throw behavior source revision differs")
    if git(
        GLASSBOX_ROOT,
        "diff",
        "HEAD",
        "--name-only",
        "--",
        "scripts/collect_fresh_causal.py",
    ):
        raise ValueError("commit the collector before generating flights")
    direct = json.loads(
        importlib.metadata.distribution("glassbox").read_text("direct_url.json")
    )
    if direct["vcs_info"]["commit_id"] != "d10bb24e4d30f3be212d0101d1f433c0004ecedb":
        raise ValueError("Throw behavior Glassbox revision differs")
    if importlib.metadata.version("crazyflow") != "0.3.2":
        raise ValueError("Crazyflow version differs")
    return {
        "collector_commit": git(GLASSBOX_ROOT, "rev-parse", "HEAD"),
        "throw_commit": git(THROW_ROOT, "rev-parse", "HEAD"),
        "behavior_glassbox_commit": direct["vcs_info"]["commit_id"],
        "crazyflow_version": importlib.metadata.version("crazyflow"),
        "cases": [
            {
                "name": name,
                "arm_ratio": ratio,
                "release_height_m": height,
                "angular_velocity_rad_s": angular,
            }
            for name, ratio, height, angular in CASES
        ],
    }


def collect(name, ratio, height, angular, output):
    plant = CrazyflowPlant(CrazyflowPlantConfig(control_frequency_hz=100))
    states, commands, times = [], [], []
    record, error = None, None
    reset, step = plant.reset, plant.step

    def recorded_reset(*args, **kwargs):
        sample = reset(*args, **kwargs)
        states.append(np.asarray(sample.state).copy())
        times.append(sample.time_s)
        return sample

    def recorded_step(command):
        sample = step(command)
        commands.append(np.asarray(command).copy())
        states.append(np.asarray(sample.state).copy())
        times.append(sample.time_s)
        return sample

    plant.reset, plant.step = recorded_reset, recorded_step
    started = time.perf_counter()
    try:
        canonical = study.CRAZYFLOW_THROW_STUDY_CASES[0]
        scenario = replace(
            canonical.scenario,
            arm_length_ratio=ratio,
            release_height_m=height,
            angular_velocity_rad_s=angular,
        )
        case = replace(canonical, name=name, scenario=scenario)
        record, _telemetry, requested, _identification, _trace = study._fly_trial(
            case, study.DUAL_CONTROL_PASS6_MODEL, plant
        )
        if not np.array_equal(np.asarray(commands), requested):
            raise ValueError("recorded commands differ from behavior policy")
    except Exception as caught:
        error = repr(caught)
    finally:
        try:
            plant.close()
        except Exception as caught:
            error = f"{error or ''} close: {caught!r}"
    np.savez_compressed(
        output / f"{name}.npz",
        states=np.asarray(states, dtype=np.float64).reshape(-1, 13),
        commands=np.asarray(commands, dtype=np.float64).reshape(-1, 4),
        time_s=np.asarray(times, dtype=np.float64),
    )
    result = {
        "status": "complete" if error is None else "failed",
        "error": error,
        "intervals": len(commands),
        "duration_s": float(times[-1]) if times else 0.0,
        "floor_contact_step": None if record is None else record.floor_contact_step,
        "wall_time_s": time.perf_counter() - started,
    }
    write_json(output / f"{name}.json", result)
    print(name, result, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bound = binding()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "binding.json", bound)
    for case in CASES:
        collect(*case, args.output)
    manifest = {
        "files": {
            path.name: digest(path)
            for path in sorted(args.output.iterdir())
            if path.is_file()
        }
    }
    write_json(args.output / "manifest.json", manifest)
    print("source_manifest_sha256", digest(args.output / "manifest.json"))


if __name__ == "__main__":
    main()
