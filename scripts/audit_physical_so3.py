"""Compare the saved SO(3) candidate recurrence with authenticated plant tangents."""

import argparse
import json
import platform
import shutil
import subprocess
import time
from dataclasses import replace
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from collect_throw import authenticate, seal
from verify_baseline import arrays, observed

from glassbox import _dynamics as core

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/harness/readout-physical-so3-attribution-v1.json"
GIT = "/opt/homebrew/Caskroom/miniconda/base/bin/git"


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def source_binding():
    dirty = subprocess.check_output(
        [GIT, "-C", str(ROOT), "status", "--porcelain", "--untracked-files=normal"],
        text=True,
    ).strip()
    assert not dirty, dirty
    return dict(
        source_commit=subprocess.check_output(
            [GIT, "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        runtime=dict(
            python=platform.python_version(),
            jax=jax.__version__,
            numpy=np.__version__,
            backend=jax.default_backend(),
        ),
    )


def with_mean(model, mean):
    f, q = len(model.params["linear"]), len(model.params["quadratic"])
    return replace(
        model,
        params=dict(
            model.params,
            linear=mean[:f],
            quadratic=mean[f : f + q],
            bias=mean[f + q],
            w2=mean[f + q + 1 :],
        ),
    )


def endpoint(params, norms, past, inputs, future, theta, *, delay, dt_s):
    rotation = past[-1, 6:].reshape(3, 3) @ core.rotation_exp(theta)
    changed = past.at[-1, 6:].set(rotation.reshape(9))
    return core._rollout(
        params, norms, changed[None], inputs[None], future[None], delay, dt_s
    )[0, -1, 3:6]


@partial(jax.jit, static_argnames=("delay", "dt_s"))
def tangent(params, norms, past, inputs, future, *, delay, dt_s):
    return jax.jacfwd(
        lambda theta: endpoint(
            params, norms, past, inputs, future, theta, delay=delay, dt_s=dt_s
        )
    )(jnp.zeros(3, dtype=past.dtype))


def finite_difference(params, norms, past, inputs, future, expected, *, delay, dt_s):
    epsilon = 1e-6
    measured = np.stack(
        [
            (
                np.asarray(
                    endpoint(
                        params,
                        norms,
                        past,
                        inputs,
                        future,
                        jnp.eye(3)[axis] * epsilon,
                        delay=delay,
                        dt_s=dt_s,
                    )
                )
                - np.asarray(
                    endpoint(
                        params,
                        norms,
                        past,
                        inputs,
                        future,
                        -jnp.eye(3)[axis] * epsilon,
                        delay=delay,
                        dt_s=dt_s,
                    )
                )
            )
            / (2 * epsilon)
            for axis in range(3)
        ],
        axis=-1,
    )
    return float(
        np.linalg.norm(measured - expected) / max(1.0, np.linalg.norm(measured))
    )


def summarize(data):
    local, plant_local = data["local"], data["plant_local"]
    five, plant_five = data["five"], data["plant_five"]
    local_norm = np.linalg.norm(local, axis=(-2, -1))
    plant_local_norm = np.linalg.norm(plant_local, axis=(-2, -1))
    local_error = np.linalg.norm(local - plant_local, axis=(-2, -1))
    five_gain = np.linalg.svd(five, compute_uv=False)[:, 0]
    plant_gain = np.linalg.svd(plant_five, compute_uv=False)[:, 0]
    five_error = np.linalg.norm(five - plant_five, axis=(-2, -1))
    alignment = np.sum(five * plant_five, axis=(-2, -1)) / (
        np.linalg.norm(five, axis=(-2, -1)) * np.linalg.norm(plant_five, axis=(-2, -1))
        + 1e-12
    )
    return dict(
        origins=len(data["rows"]),
        measured_steps=local_norm.size,
        median_local_norm=float(np.median(local_norm)),
        median_plant_local_norm=float(np.median(plant_local_norm)),
        median_local_error_norm=float(np.median(local_error)),
        median_five_gain=float(np.median(five_gain)),
        median_plant_five_gain=float(np.median(plant_gain)),
        median_five_error_norm=float(np.median(five_error)),
        median_five_alignment=float(np.median(alignment)),
        per_origin_five_gain=five_gain.tolist(),
        per_origin_plant_gain=plant_gain.tolist(),
        maximum_fd_relative=float(np.max(data["fd_relative"])),
    )


def run_case(spec, case, output, deadline):
    sources = {name: Path(value["path"]) for name, value in spec["sources"].items()}
    candidate = sources["candidate"] / case
    info = read(candidate / "result.json")
    prediction = arrays(candidate / "predictions.npz")
    plant = arrays(sources["plant"] / f"{case}.npz")
    tape = arrays(sources["recordings"] / "inputs" / f"{case}.npz")
    states, commands = observed(tape["states"]), tape["commands"].astype(np.float64)
    model = core.VehicleSequenceModel.from_arrays(
        read(candidate / "initial.json"), arrays(candidate / "initial.npz")
    )
    rows = prediction["conditional_rows"]
    assert len(rows) == 14 and np.array_equal(rows, plant["rows"])
    saved = dict(
        rows=rows,
        local=[],
        plant_local=plant["truth_one_step"][:, :, 3:6, 6:9],
        five=[],
        plant_five=plant["truth_five_step"][:, 3:6, 6:9],
        fd_relative=[],
    )
    h = model.history_steps
    with jax.enable_x64(True):
        for origin, row in enumerate(rows):
            if time.monotonic() > deadline:
                raise TimeoutError("frozen no-fit audit wall limit exceeded")
            row = int(row)
            mean = (
                None
                if row == info["first_prediction_row"]
                else prediction["mean"][row - info["first_prediction_row"] - 1]
            )
            fitted = model if mean is None else with_mean(model, mean)
            params, norms = jax.tree.map(jnp.asarray, (fitted.params, fitted.norms))
            matrices = []
            for step in range(5):
                t = row + step
                past = jnp.asarray(states[t - h : t + 1])
                inputs = jnp.asarray(commands[t - h : t])
                future = jnp.asarray(commands[t : t + 1])
                value = np.asarray(
                    tangent(
                        params,
                        norms,
                        past,
                        inputs,
                        future,
                        delay=model.delay_steps,
                        dt_s=model.dt_s,
                    )
                )
                matrices.append(value)
                if origin in (0, 7) and step == 0:
                    saved["fd_relative"].append(
                        finite_difference(
                            params,
                            norms,
                            past,
                            inputs,
                            future,
                            value,
                            delay=model.delay_steps,
                            dt_s=model.dt_s,
                        )
                    )
            saved["local"].append(matrices)
            past = jnp.asarray(states[row - h : row + 1])
            inputs = jnp.asarray(commands[row - h : row])
            future = jnp.asarray(commands[row : row + 5])
            value = np.asarray(
                tangent(
                    params,
                    norms,
                    past,
                    inputs,
                    future,
                    delay=model.delay_steps,
                    dt_s=model.dt_s,
                )
            )
            saved["five"].append(value)
            if origin in (0, 7):
                saved["fd_relative"].append(
                    finite_difference(
                        params,
                        norms,
                        past,
                        inputs,
                        future,
                        value,
                        delay=model.delay_steps,
                        dt_s=model.dt_s,
                    )
                )
    data = {key: np.asarray(value) for key, value in saved.items()}
    assert all(np.isfinite(value).all() for value in data.values())
    summary = summarize(data)
    assert summary["maximum_fd_relative"] <= 0.0001
    with output.with_suffix(".npz").open("wb") as stream:
        np.savez_compressed(stream, **data)
    write(output.with_suffix(".json"), summary)
    print(case, summary, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "verify"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-sha256")
    args = parser.parse_args()
    if args.mode == "verify":
        authenticate(args.output, args.manifest_sha256)
        spec = read(args.output / "protocol.json")
        for value in spec["sources"].values():
            authenticate(Path(value["path"]), value["manifest_sha256"])
        for case in spec["cases"]:
            data = arrays(args.output / f"{case}.npz")
            assert summarize(data) == read(args.output / f"{case}.json")
            plant = arrays(Path(spec["sources"]["plant"]["path"]) / f"{case}.npz")
            assert np.array_equal(data["rows"], plant["rows"])
            assert np.array_equal(
                data["plant_local"], plant["truth_one_step"][:, :, 3:6, 6:9]
            )
            assert np.array_equal(
                data["plant_five"], plant["truth_five_step"][:, 3:6, 6:9]
            )
        print(
            "Verified saved candidate and plant tangent summaries without model calls."
        )
        return
    spec = read(PROTOCOL)
    for value in spec["sources"].values():
        authenticate(Path(value["path"]), value["manifest_sha256"])
    bound = source_binding()
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(PROTOCOL, args.output / "protocol.json")
    write(args.output / "binding.json", bound)
    deadline = time.monotonic() + spec["budget"]["maximum_wall_s"]
    try:
        for case in spec["cases"]:
            run_case(spec, case, args.output / case, deadline)
    except BaseException as error:
        write(args.output / "failure.json", {"error": repr(error)})
        raise
    finally:
        print(
            "manifest_sha256",
            seal(args.output, "glassbox-readout-physical-so3-attribution-v1"),
            flush=True,
        )


if __name__ == "__main__":
    main()
