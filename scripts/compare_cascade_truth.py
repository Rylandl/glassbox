"""Compare saved Glassbox physical recurrence with exactly replayed Cascade truth."""

import argparse
import platform
import shutil
import subprocess
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import cascade
from cascade.canonical import frd_to_flu, ned_to_nwu, rigid_body_to_canonical
from cascade.initialization import control_from_array
from cascade.math import (
    quaternion_conjugate,
    quaternion_from_rotvec,
    quaternion_multiply,
    quaternion_to_rotvec,
)

from attribute_readout_recurrence import authenticate, digest, gain, load, read, seal, write


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/harness/readout-attribution-truth-v1.json"
GIT = "/opt/homebrew/Caskroom/miniconda/base/bin/git"
SIGNS = jnp.asarray([1.0, -1.0, -1.0])
ARMS = ("baseline", "curvature", "candidate")


def source_check(spec):
    root = Path(spec["path"])
    commit = subprocess.check_output([GIT, "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    assert commit == spec["commit"]
    changed = subprocess.check_output(
        [GIT, "-C", str(root), "diff", "HEAD", "--name-only", "--", "src/cascade"], text=True
    ).splitlines()
    assert set(changed) <= {"src/cascade/viz/geometry.py"}, changed
    untracked = subprocess.check_output(
        [GIT, "-C", str(root), "ls-files", "--others", "--exclude-standard", "--", "src/cascade"], text=True
    ).splitlines()
    assert not untracked, untracked
    return {"commit": commit, "changed_source_files": changed}


def perturb(state, tangent):
    body = state.rigid_body
    changed = body._replace(
        velocity=body.velocity + SIGNS * tangent[:3],
        angular_velocity=body.angular_velocity + SIGNS * tangent[3:6],
        attitude=quaternion_multiply(
            body.attitude, quaternion_from_rotvec(SIGNS * tangent[6:9])
        ),
    )
    return state._replace(rigid_body=changed)


def physical_difference(state, reference):
    actual, nominal = state.rigid_body, reference.rigid_body
    relative = quaternion_multiply(quaternion_conjugate(nominal.attitude), actual.attitude)
    return jnp.concatenate((
        ned_to_nwu(actual.velocity - nominal.velocity),
        frd_to_flu(actual.angular_velocity - nominal.angular_velocity),
        frd_to_flu(quaternion_to_rotvec(relative)),
    ))


def simulator_functions(plant):
    def advance(state, commands):
        for command in commands:
            control = control_from_array(plant.model, command)
            state = plant._advance(state, control, plant._environment)
        return state

    @jax.jit
    def derivative(state, commands):
        nominal = advance(state, commands)
        return jax.jacfwd(
            lambda tangent: physical_difference(
                advance(perturb(state, tangent), commands), nominal
            )
        )(jnp.zeros(9, dtype=state.rigid_body.velocity.dtype))

    @jax.jit
    def forecast(state, commands):
        return advance(state, commands)

    return derivative, forecast


def max_rigid_difference(a, b):
    a = np.asarray(rigid_body_to_canonical(a.rigid_body))
    b = np.asarray(rigid_body_to_canonical(b.rigid_body))
    return float(np.max(np.abs(a - b)))


def finite_difference(derivative, forecast, state, commands, *, direction_seed, epsilon):
    direction = np.random.default_rng(direction_seed).normal(size=9)
    direction /= np.linalg.norm(direction)
    nominal = forecast(state, commands)
    plus = forecast(perturb(state, jnp.asarray(epsilon * direction)), commands)
    minus = forecast(perturb(state, jnp.asarray(-epsilon * direction)), commands)
    measured = np.asarray((physical_difference(plus, nominal) - physical_difference(minus, nominal)) / (2 * epsilon))
    prediction = np.asarray(derivative @ direction)
    return float(np.linalg.norm(prediction - measured) / max(1, np.linalg.norm(measured)))


def product(sequence):
    value = np.eye(sequence.shape[-1])
    for matrix in sequence:
        value = matrix @ value
    return value


def model_matrix(model, arm, query, path="measured"):
    key = "baseline_measured_full" if arm == "baseline" else f"{arm}_{path}_full"
    return product(model[key][query])[3:6, 6:9]


def summarize(plant_data, model):
    truth_one, truth_five = plant_data["truth_one_step"], plant_data["truth_five_step"]
    report = {
        "replay_max_abs": float(plant_data["replay_abs"].max()),
        "replay_final_max_abs": float(plant_data["replay_abs"][-1]),
        "branch_max_abs": float(plant_data["branch_abs"].max()),
        "finite_difference_max_relative": float(plant_data["fd_relative"].max()),
        "truth": {
            "median_local_attitude_to_rate_norm": float(np.median(
                np.linalg.norm(truth_one[:, :, 3:6, 6:9], axis=(-2, -1))
            )),
            "median_five_step_attitude_to_rate_gain": float(np.median(
                [np.linalg.svd(matrix[3:6, 6:9], compute_uv=False)[0] for matrix in truth_five]
            )),
        },
        "models": {},
    }
    for arm in ARMS:
        matrices = model["baseline_measured_full"] if arm == "baseline" else model[f"{arm}_measured_full"]
        local = matrices[:, :, 3:6, 6:9]
        final = np.asarray([model_matrix(model, arm, query) for query in range(len(truth_five))])
        true_final = truth_five[:, 3:6, 6:9]
        truth_norm = np.linalg.norm(true_final, axis=(-2, -1))
        difference = np.linalg.norm(final - true_final, axis=(-2, -1))
        alignment = np.sum(final * true_final, axis=(-2, -1)) / np.maximum(
            np.linalg.norm(final, axis=(-2, -1)) * truth_norm, 1e-12
        )
        report["models"][arm] = {
            "median_local_attitude_to_rate_norm": float(np.median(np.linalg.norm(local, axis=(-2, -1)))),
            "median_local_attitude_to_rate_error_norm": float(np.median(np.linalg.norm(
                local - truth_one[:, :, 3:6, 6:9], axis=(-2, -1)
            ))),
            "median_five_step_attitude_to_rate_gain": float(np.median(
                [np.linalg.svd(matrix, compute_uv=False)[0] for matrix in final]
            )),
            "median_five_step_attitude_to_rate_error_norm": float(np.median(difference)),
            "median_five_step_attitude_alignment": float(np.median(alignment)),
            "per_origin_gain": [float(np.linalg.svd(matrix, compute_uv=False)[0]) for matrix in final],
            "per_origin_error_norm": difference.tolist(),
        }
    return report


def analyze_case(case, spec, deadline):
    path = Path(spec["sources"]["recording_80" if case.endswith("80") else "recording_81"]["path"])
    with np.load(path, allow_pickle=False) as recording:
        states, commands, prefix = (recording[name].copy() for name in ("states", "controls", "control_prefix"))
    model = load(Path(spec["sources"]["model_attribution"]["path"]) / f"{case}.npz")
    rows = model["rows"]
    plant = cascade.Plant(cascade.skywalker_x8_spec(), cascade.PlantConfig(control_frequency_hz=20))
    sample = plant.reset(states[0], applied_control=prefix[0])
    replay = [plant._state]
    replay_abs = [float(np.max(np.abs(sample.state - states[0])))]
    for index, command in enumerate(commands):
        sample = plant.step(command)
        replay.append(plant._state)
        replay_abs.append(float(np.max(np.abs(sample.state - states[index + 1]))))
    replay_abs = np.asarray(replay_abs)
    assert replay_abs.max() < spec["replay_gate_max_abs"]
    derivative, forecast = simulator_functions(plant)
    one_step, five_step, branch_abs, fd_relative = [], [], [], []
    for query, row in enumerate(rows):
        if time.monotonic() > deadline:
            raise TimeoutError("frozen 600 s truth-audit budget exceeded")
        row = int(row)
        measured = []
        for horizon in range(5):
            t = row + horizon
            segment = jnp.asarray(commands[t : t + 1])
            measured.append(np.asarray(derivative(replay[t], segment)))
            branch_abs.append(max_rigid_difference(forecast(replay[t], segment), replay[t + 1]))
        segment = jnp.asarray(commands[row : row + 5])
        five_step.append(np.asarray(derivative(replay[row], segment)))
        branch_abs.append(max_rigid_difference(forecast(replay[row], segment), replay[row + 5]))
        if query in (0, 7):
            fd_relative.append(finite_difference(
                measured[0], forecast, replay[row], jnp.asarray(commands[row : row + 1]),
                direction_seed=391 + query, epsilon=spec["fd_epsilon"],
            ))
            fd_relative.append(finite_difference(
                five_step[-1], forecast, replay[row], segment, direction_seed=421 + query,
                epsilon=spec["fd_epsilon"],
            ))
        one_step.append(measured)
        print(case, query + 1, "/", len(rows), flush=True)
    result = {
        "rows": rows,
        "truth_one_step": np.asarray(one_step),
        "truth_five_step": np.asarray(five_step),
        "replay_abs": replay_abs,
        "branch_abs": np.asarray(branch_abs),
        "fd_relative": np.asarray(fd_relative),
    }
    assert all(np.isfinite(value).all() for value in result.values())
    assert result["branch_abs"].max() < spec["branch_max_abs"]
    assert result["fd_relative"].max() < spec["fd_relative_max"]
    return result, model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "verify"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-sha256")
    args = parser.parse_args()
    if args.mode == "verify":
        authenticate(args.output, args.manifest_sha256)
        spec = read(args.output / "protocol.json")
        authenticate(spec["sources"]["model_attribution"]["path"],
                     spec["sources"]["model_attribution"]["manifest_sha256"])
        for case in ("fixedwing-80", "fixedwing-81"):
            truth = load(args.output / f"{case}.npz")
            model = load(Path(spec["sources"]["model_attribution"]["path"]) / f"{case}.npz")
            assert summarize(truth, model) == read(args.output / f"{case}.json")
        print("Verified saved plant-truth comparison without simulation or fitting.")
        return
    spec = read(PROTOCOL)
    sources = spec["sources"]
    authenticate(sources["model_attribution"]["path"], sources["model_attribution"]["manifest_sha256"])
    for name in ("recording_80", "recording_81"):
        assert digest(sources[name]["path"]) == sources[name]["sha256"]
    cascade_binding = source_check(sources["cascade_source"])
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(PROTOCOL, args.output / "protocol.json")
    write(args.output / "binding.json", {
        "script_sha256": digest(Path(__file__)),
        "protocol_sha256": digest(PROTOCOL),
        "cascade": cascade_binding,
        "runtime": {"python": platform.python_version(), "jax": jax.__version__,
                    "numpy": np.__version__, "backend": jax.default_backend(),
                    "x64_ambient": bool(jax.config.x64_enabled)},
    })
    deadline = time.monotonic() + spec["budget"]["maximum_wall_s"]
    for case in ("fixedwing-80", "fixedwing-81"):
        truth, model = analyze_case(case, spec, deadline)
        with (args.output / f"{case}.npz").open("wb") as stream:
            np.savez_compressed(stream, **truth)
        report = summarize(truth, model)
        write(args.output / f"{case}.json", report)
        print(case, report["truth"], flush=True)
    print("manifest_sha256", seal(args.output), flush=True)


if __name__ == "__main__":
    main()
