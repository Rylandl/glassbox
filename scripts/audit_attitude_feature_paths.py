"""Decompose saved readout attitude response through its physical input features."""

import argparse
import platform
import shutil
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from glassbox import _dynamics as core
from verify_baseline import observed

from attribute_readout_recurrence import (
    authenticate,
    carry_from,
    digest,
    load,
    read,
    seal,
    with_mean,
    write,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/harness/readout-attitude-paths-v1.json"
ARMS = ("curvature", "candidate")


@jax.jit
def path_derivatives(params, norms, carry, command):
    state, applied, history, hidden = carry
    command = command[None]
    current = core.current_features(state, command, applied, norms)[0]

    def head(features):
        projected, _ = core._prepare_head(params, norms, features[None], history, hidden)
        return core._acceleration(params, norms, features[None], projected)[0, 3:6]

    def changed_state(theta):
        rotation = state[0, 6:].reshape(3, 3) @ core.rotation_exp(theta)
        return state.at[0, 6:].set(rotation.reshape(9))

    def changed_features(theta):
        return core.current_features(changed_state(theta), command, applied, norms)[0]

    derivative_head = jax.jacfwd(head)(current)
    derivative_features = jax.jacfwd(changed_features)(jnp.zeros(3, dtype=state.dtype))
    body_velocity = derivative_head[:, :3] @ derivative_features[:3]
    gravity_direction = derivative_head[:, 6:9] @ derivative_features[6:9]
    direct = jax.jacfwd(
        lambda theta: core._head(
            params, norms, changed_state(theta), command, applied, history, hidden
        )[0][0, 3:6]
    )(jnp.zeros(3, dtype=state.dtype))
    return body_velocity, gravity_direction, direct, derivative_head, derivative_features


def summarize(data):
    result = {}
    for arm in ARMS:
        velocity = data[f"{arm}_velocity"]
        gravity = data[f"{arm}_gravity"]
        direct = data[f"{arm}_direct"]
        vnorm = np.linalg.norm(velocity, axis=(-2, -1))
        gnorm = np.linalg.norm(gravity, axis=(-2, -1))
        total = np.linalg.norm(direct, axis=(-2, -1))
        result[arm] = {
            "median_body_velocity_norm": float(np.median(vnorm)),
            "median_gravity_direction_norm": float(np.median(gnorm)),
            "median_total_norm": float(np.median(total)),
            "median_velocity_to_total_ratio": float(np.median(vnorm / np.maximum(total, 1e-12))),
            "median_gravity_to_total_ratio": float(np.median(gnorm / np.maximum(total, 1e-12))),
            "velocity_larger_count": int(np.sum(vnorm > gnorm)),
            "samples": int(total.size),
            "maximum_chain_difference": float(np.max(np.abs(velocity + gravity - direct))),
            "median_component_cancellation": float(np.median(
                (vnorm + gnorm) / np.maximum(total, 1e-12)
            )),
            "per_origin_median_velocity_norm": np.median(vnorm, axis=1).tolist(),
            "per_origin_median_gravity_norm": np.median(gnorm, axis=1).tolist(),
        }
    result["maximum_attitude_invariant_feature_derivative"] = float(
        data["invariant_feature_difference"].max()
    )
    return result


def analyze_case(case, spec, deadline):
    sources = {key: Path(value["path"]) for key, value in spec["sources"].items()}
    source = sources["evaluation"] / case
    info = read(source / "result.json")
    predictions = load(source / "predictions.npz")
    tape = load(sources["tapes"] / "inputs" / f"{case}.npz")
    states, commands = observed(tape["states"]), tape["commands"].astype(np.float64)
    model = core.VehicleSequenceModel.from_arrays(info["initial_model_metadata"], load(source / "initial.npz"))
    rows = predictions["conditional_rows"]
    saved = {f"{arm}_{field}": [] for arm in ARMS for field in ("velocity", "gravity", "direct")}
    saved.update(rows=rows, invariant_feature_difference=[])
    with jax.enable_x64(True):
        for query, row in enumerate(rows):
            if time.monotonic() > deadline:
                raise TimeoutError("frozen 120 s feature-path budget exceeded")
            row = int(row)
            offset = row - info["first"]
            for arm in ARMS:
                mean = None if offset == 0 else predictions[f"{arm}_mean"][offset - 1]
                fitted = model if mean is None else with_mean(model, mean)
                params, norms = jax.tree.map(jnp.asarray, (fitted.params, fitted.norms))
                h = fitted.history_steps
                values = {field: [] for field in ("velocity", "gravity", "direct")}
                for horizon in range(5):
                    t = row + horizon
                    carry = carry_from(
                        params, norms, jnp.asarray(states[t - h : t + 1]),
                        jnp.asarray(commands[t - h : t]), delay=fitted.delay_steps,
                        dt_s=fitted.dt_s,
                    )
                    velocity, gravity, direct, _, derivative_features = path_derivatives(
                        params, norms, carry, jnp.asarray(commands[t])
                    )
                    values["velocity"].append(np.asarray(velocity))
                    values["gravity"].append(np.asarray(gravity))
                    values["direct"].append(np.asarray(direct))
                    saved["invariant_feature_difference"].append(float(np.max(np.abs(
                        np.asarray(derivative_features)[3:6]
                    ))))
                    saved["invariant_feature_difference"].append(float(np.max(np.abs(
                        np.asarray(derivative_features)[9:]
                    ))))
                for field, matrices in values.items():
                    saved[f"{arm}_{field}"].append(matrices)
            print(case, query + 1, "/", len(rows), flush=True)
    data = {key: np.asarray(value) for key, value in saved.items()}
    assert all(np.isfinite(value).all() for value in data.values())
    summary = summarize(data)
    assert summary["maximum_attitude_invariant_feature_derivative"] < 1e-10
    assert all(summary[arm]["maximum_chain_difference"] < 1e-8 for arm in ARMS)
    return data, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "verify"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-sha256")
    args = parser.parse_args()
    if args.mode == "verify":
        authenticate(args.output, args.manifest_sha256)
        for case in read(args.output / "protocol.json")["cases"]:
            assert summarize(load(args.output / f"{case}.npz")) == read(args.output / f"{case}.json")
        print("Verified saved attitude-path decomposition without model calls or fitting.")
        return
    spec = read(PROTOCOL)
    for source in spec["sources"].values():
        authenticate(source["path"], source["manifest_sha256"])
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(PROTOCOL, args.output / "protocol.json")
    write(args.output / "binding.json", {
        "script_sha256": digest(Path(__file__)), "protocol_sha256": digest(PROTOCOL),
        "runtime": {"python": platform.python_version(), "jax": jax.__version__,
                    "numpy": np.__version__, "backend": jax.default_backend(),
                    "x64_ambient": bool(jax.config.x64_enabled)},
    })
    deadline = time.monotonic() + spec["budget"]["maximum_wall_s"]
    for case in spec["cases"]:
        data, summary = analyze_case(case, spec, deadline)
        with (args.output / f"{case}.npz").open("wb") as stream:
            np.savez_compressed(stream, **data)
        write(args.output / f"{case}.json", summary)
        print(case, summary, flush=True)
    print("manifest_sha256", seal(args.output), flush=True)


if __name__ == "__main__":
    main()
