"""Attribute saved fast-readout recurrence without fitting another model."""

import argparse
import hashlib
import json
import math
import platform
import shutil
import time
from dataclasses import replace
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from glassbox import _dynamics as core
from verify_baseline import observed


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/harness/readout-attribution-v1.json"
GROUPS = {"velocity": slice(0, 3), "rate": slice(3, 6), "attitude": slice(6, 9)}
ARMS = ("curvature", "candidate")
PATHS = ("measured", "rolled")
PERTURBATIONS = ("original", "known_attitude", "known_attitude_rate", "known_attitude_force", "no_latent_return")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def authenticate(root, expected):
    root = Path(root)
    assert digest(root / "manifest.json") == expected, root
    manifest = read(root / "manifest.json")
    files = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and p != root / "manifest.json"}
    assert files == set(manifest["files"]), root
    for name, wanted in manifest["files"].items():
        assert digest(root / name) == wanted, root / name


def load(path):
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def with_mean(model, mean):
    linear, quadratic = len(model.params["linear"]), len(model.params["quadratic"])
    return replace(
        model,
        params=dict(
            model.params,
            linear=mean[:linear],
            quadratic=mean[linear : linear + quadratic],
            bias=mean[linear + quadratic],
            w2=mean[linear + quadratic + 1 :],
        ),
    )


@partial(jax.jit, static_argnames=("delay", "dt_s"))
def carry_from(params, norms, past, inputs, *, delay, dt_s):
    applied, history, hidden = core._history(params, norms, past[None], inputs[None], delay, dt_s)
    return past[None, -1], applied, history, hidden


@partial(jax.jit, static_argnames=("dt_s",))
def step(params, norms, carry, command, *, dt_s):
    return core.physical_step(params, norms, carry[0], command[None], *carry[1:], dt_s)


def frozen_acceleration_step(params, norms, states, commands, filtered, history, hidden, dt_s):
    """Identical nominal step with all learned acceleration derivatives stopped."""
    count = max(1, math.ceil(dt_s / core.MAX_SUBSTEP_S))
    duration = dt_s / count
    tau = core.time_constants(params)
    gravity = jnp.asarray(core.GRAVITY, dtype=states.dtype)
    start = core.current_features(states, commands, filtered, norms)
    base, effective = core._prepare_head(params, norms, start, history, hidden)

    def acceleration(state, applied):
        current = core.current_features(state, commands, applied, norms)
        projection = base + (current - start) @ effective
        return jax.lax.stop_gradient(core._acceleration(params, norms, current, projection))

    def substep(_, carry):
        state, applied = carry
        velocity, rate = state[..., :3], state[..., 3:6]
        rotation = state[..., 6:].reshape((*state.shape[:-1], 3, 3))
        first = acceleration(state, applied)
        world_first = gravity + jnp.einsum("...ij,...j->...i", rotation, first[..., :3])
        rotation_half = rotation @ core.rotation_exp(0.5 * duration * rate)
        velocity_half = velocity + 0.5 * duration * world_first
        rate_half = rate + 0.5 * duration * first[..., 3:]
        applied_half = commands + (applied - commands) * jnp.exp(-0.5 * duration / tau)
        state_half = jnp.concatenate(
            (velocity_half, rate_half, rotation_half.reshape((*state.shape[:-1], 9))), -1
        )
        middle = acceleration(state_half, applied_half)
        velocity_next = velocity + duration * (
            gravity + jnp.einsum("...ij,...j->...i", rotation_half, middle[..., :3])
        )
        rate_next = rate + duration * middle[..., 3:]
        rotation_next = rotation @ core.rotation_exp(duration * rate_half)
        state_next = jnp.concatenate(
            (velocity_next, rate_next, rotation_next.reshape((*state.shape[:-1], 9))), -1
        )
        applied_next = commands + (applied - commands) * jnp.exp(-duration / tau)
        return state_next, applied_next

    state, applied = jax.lax.fori_loop(0, count, substep, (states, filtered))
    hidden_next = core.memory_step(params, norms, start, hidden, dt_s)
    history_next = jnp.concatenate((history[:, 1:], start[:, None]), axis=1)
    return state, applied, history_next, hidden_next


@partial(jax.jit, static_argnames=("dt_s",))
def known_step(params, norms, carry, command, *, dt_s):
    return frozen_acceleration_step(params, norms, carry[0], command[None], *carry[1:], dt_s)


def perturb(carry, delta):
    state, *latent = carry
    rotation = state[:, 6:].reshape((-1, 3, 3)) @ core.rotation_exp(delta[6:9])[None]
    changed = jnp.concatenate((state[:, :6] + delta[None, :6], rotation.reshape((-1, 9))), axis=-1)
    result, index = [changed], 9
    for value in latent:
        result.append(value + delta[index : index + value.size].reshape(value.shape))
        index += value.size
    return tuple(result)


def difference(value, reference):
    state, origin = value[0], reference[0]
    relative = origin[:, 6:].reshape((-1, 3, 3)).swapaxes(-1, -2) @ state[:, 6:].reshape((-1, 3, 3))
    skew = 0.5 * (relative - relative.swapaxes(-1, -2))
    angle = jnp.stack((skew[0, 2, 1], skew[0, 0, 2], skew[0, 1, 0]))
    return jnp.concatenate(
        ((state[:, :6] - origin[:, :6]).ravel(), angle,
         *[(a - b).ravel() for a, b in zip(value[1:], reference[1:])])
    )


@partial(jax.jit, static_argnames=("dt_s", "known"))
def jacobian(params, norms, carry, command, *, dt_s, known):
    fn = known_step if known else step
    nominal = fn(params, norms, carry, command, dt_s=dt_s)
    size = 9 + sum(value.size for value in carry[1:])
    return jax.jacfwd(
        lambda delta: difference(
            fn(params, norms, perturb(carry, delta), command, dt_s=dt_s), nominal
        )
    )(jnp.zeros(size, dtype=carry[0].dtype))


def gain(sequence, start):
    product = np.eye(sequence.shape[-1])
    for matrix in sequence:
        product = matrix @ product
    return float(np.linalg.svd(product[3:6, start], compute_uv=False)[0])


def intervention(full, known, label):
    adjusted = full.copy()
    if label == "known_attitude":
        adjusted[:6, 6:9] = known[:6, 6:9]
    elif label == "known_attitude_rate":
        adjusted[3:6, 6:9] = known[3:6, 6:9]
    elif label == "known_attitude_force":
        adjusted[:3, 6:9] = known[:3, 6:9]
    elif label == "no_latent_return":
        adjusted[:9, 9:] = 0
    else:
        assert label == "original"
    return adjusted


def summarize(data):
    result = {}
    baseline = data["baseline_measured_full"]
    result["baseline_measured"] = {
        "original": {
            group: [gain(query, section) for query in baseline]
            for group, section in GROUPS.items()
        },
        "median_full_radius": float(np.median(
            np.max(np.abs(np.linalg.eigvals(baseline)), axis=-1)
        )),
    }
    for arm in ARMS:
        result[arm] = {}
        for path in PATHS:
            full = data[f"{arm}_{path}_full"]
            known = data[f"{arm}_{path}_known"]
            path_result = {}
            for label in PERTURBATIONS:
                matrices = np.asarray([
                    [intervention(a, b, label) for a, b in zip(q, k)]
                    for q, k in zip(full, known)
                ])
                path_result[label] = {
                    group: [gain(q, section) for q in matrices]
                    for group, section in GROUPS.items()
                }
            learned = full - known
            path_result["median_learned_attitude_to_rate_jacobian_norm"] = float(
                np.median(np.linalg.norm(learned[:, :, 3:6, 6:9], axis=(-2, -1)))
            )
            path_result["median_learned_attitude_to_force_jacobian_norm"] = float(
                np.median(np.linalg.norm(learned[:, :, :3, 6:9], axis=(-2, -1)))
            )
            path_result["median_known_attitude_to_force_jacobian_norm"] = float(
                np.median(np.linalg.norm(known[:, :, :3, 6:9], axis=(-2, -1)))
            )
            path_result["median_latent_to_rate_jacobian_norm"] = float(
                np.median(np.linalg.norm(full[:, :, 3:6, 9:], axis=(-2, -1)))
            )
            path_result["median_physical_radius"] = float(np.median(
                np.max(np.abs(np.linalg.eigvals(full[:, :, :9, :9])), axis=-1)
            ))
            path_result["median_full_radius"] = float(np.median(
                np.max(np.abs(np.linalg.eigvals(full)), axis=-1)
            ))
            result[arm][path] = path_result
        result[arm]["rate_error_250_per_origin"] = np.linalg.norm(
            data[f"{arm}_prediction"][:, -1, 3:6] - data["truth"][:, -1, 3:6], axis=-1
        ).tolist()
        result[arm]["rate_rmse_250"] = float(np.sqrt(np.mean(np.square(
            result[arm]["rate_error_250_per_origin"]
        ))))
    result["checks"] = {
        "maximum_saved_jacobian_difference": float(data["saved_jacobian_difference"].max()),
        "maximum_nominal_step_difference": float(data["nominal_step_difference"].max()),
        "maximum_fd_relative_error": float(data["fd_relative_error"].max()),
        "maximum_known_rate_identity_difference": float(data["known_rate_identity_difference"].max()),
        "maximum_known_velocity_identity_difference": float(data["known_velocity_identity_difference"].max()),
        "maximum_forecast_relative_difference": float(data["forecast_relative_difference"].max()),
        "maximum_previous_curvature_jacobian_difference": float(
            data["previous_curvature_jacobian_difference"].max()
        ),
    }
    return result


def finite_difference(params, norms, carry, command, *, dt_s, known, expected):
    direction = np.random.default_rng(391).normal(size=expected.shape[-1])
    direction /= np.linalg.norm(direction)
    epsilon = 1e-7
    fn = known_step if known else step
    nominal = fn(params, norms, carry, command, dt_s=dt_s)
    plus = fn(params, norms, perturb(carry, jnp.asarray(epsilon * direction)), command, dt_s=dt_s)
    minus = fn(params, norms, perturb(carry, jnp.asarray(-epsilon * direction)), command, dt_s=dt_s)
    measured = np.asarray((difference(plus, nominal) - difference(minus, nominal)) / (2 * epsilon))
    return float(np.linalg.norm(expected @ direction - measured) / max(1, np.linalg.norm(measured)))


def analyze_case(case, spec, deadline):
    roots = {key: Path(value["path"]) for key, value in spec["sources"].items()}
    source = roots["evaluation"] / case
    info = read(source / "result.json")
    predicted = load(source / "predictions.npz")
    earlier = load(roots["recurrence"] / f"{case}.npz")
    previous = load(roots["previous_diagnosis"] / case / "diagnostics.npz")
    tape = load(roots["tapes"] / "inputs" / f"{case}.npz")
    states, commands = observed(tape["states"]), tape["commands"].astype(np.float64)
    model = core.VehicleSequenceModel.from_arrays(info["initial_model_metadata"], load(source / "initial.npz"))
    assert model.dt_s == 0.05
    rows = predicted["conditional_rows"]
    assert len(rows) == 14
    saved = {f"{arm}_{path}_{kind}": [] for arm in ARMS for path in PATHS for kind in ("full", "known")}
    saved.update(rows=rows, truth=predicted["conditional_truth"], saved_jacobian_difference=[],
                 nominal_step_difference=[], fd_relative_error=[], forecast_relative_difference=[],
                 baseline_measured_full=previous["baseline_measured_jacobian"],
                 previous_curvature_jacobian_difference=[], known_rate_identity_difference=[],
                 known_velocity_identity_difference=[])
    for arm in ARMS:
        saved[f"{arm}_prediction"] = []
    with jax.enable_x64(True):
        for query, row in enumerate(rows):
            if time.monotonic() > deadline:
                raise TimeoutError("frozen 600 s diagnostic budget exceeded")
            row = int(row)
            offset = row - info["first"]
            assert offset >= 0
            for arm in ARMS:
                mean = None if offset == 0 else predicted[f"{arm}_mean"][offset - 1]
                fitted = model if mean is None else with_mean(model, mean)
                params, norms = jax.tree.map(jnp.asarray, (fitted.params, fitted.norms))
                h = fitted.history_steps
                carry = carry_from(params, norms, jnp.asarray(states[row - h : row + 1]),
                                   jnp.asarray(commands[row - h : row]), delay=fitted.delay_steps,
                                   dt_s=fitted.dt_s)
                arrays = {f"{path}_{kind}": [] for path in PATHS for kind in ("full", "known")}
                forecast = []
                for horizon in range(5):
                    t = row + horizon
                    command = jnp.asarray(commands[t])
                    measured = carry_from(params, norms, jnp.asarray(states[t - h : t + 1]),
                                          jnp.asarray(commands[t - h : t]), delay=fitted.delay_steps,
                                          dt_s=fitted.dt_s)
                    for path, at in (("measured", measured), ("rolled", carry)):
                        full = np.asarray(jacobian(params, norms, at, command, dt_s=fitted.dt_s, known=False))
                        known = np.asarray(jacobian(params, norms, at, command, dt_s=fitted.dt_s, known=True))
                        arrays[f"{path}_full"].append(full)
                        arrays[f"{path}_known"].append(known)
                        expected_rate = np.zeros_like(known[3:6])
                        expected_rate[:, 3:6] = np.eye(3)
                        saved["known_rate_identity_difference"].append(float(np.max(np.abs(
                            known[3:6] - expected_rate
                        ))))
                        saved["known_velocity_identity_difference"].append(float(np.max(np.abs(
                            known[:3, :3] - np.eye(3)
                        ))))
                        nominal_full = step(params, norms, at, command, dt_s=fitted.dt_s)
                        nominal_known = known_step(params, norms, at, command, dt_s=fitted.dt_s)
                        saved["nominal_step_difference"].append(max(
                            float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
                            for a, b in zip(nominal_full, nominal_known)
                        ))
                        if path == "measured":
                            saved["saved_jacobian_difference"].append(float(np.max(np.abs(
                                full - earlier[f"{arm}_full"][query, horizon]
                            ))))
                            if arm == "curvature":
                                saved["previous_curvature_jacobian_difference"].append(
                                    float(np.max(np.abs(
                                        full - previous["candidate_measured_jacobian"][query, horizon]
                                    )))
                                )
                        if path == "measured" and query in (0, 7) and horizon == 0:
                            saved["fd_relative_error"].append(finite_difference(
                                params, norms, at, command, dt_s=fitted.dt_s, known=False, expected=full
                            ))
                    carry = step(params, norms, carry, command, dt_s=fitted.dt_s)
                    forecast.append(np.asarray(carry[0])[0])
                forecast = np.asarray(forecast)
                reference = predicted[f"{arm}_conditional"][query]
                saved["forecast_relative_difference"].append(float(
                    np.linalg.norm(forecast - reference) / max(1, np.linalg.norm(reference))
                ))
                saved[f"{arm}_prediction"].append(forecast)
                for key, values in arrays.items():
                    saved[f"{arm}_{key}"].append(values)
            print(case, query + 1, "/", len(rows), flush=True)
    result = {key: np.asarray(value) for key, value in saved.items()}
    assert all(np.isfinite(value).all() for value in result.values())
    assert result["saved_jacobian_difference"].max() < 1e-8
    assert result["previous_curvature_jacobian_difference"].max() < 1e-8
    assert result["nominal_step_difference"].max() < 1e-11
    assert result["fd_relative_error"].max() < 2e-5
    assert result["known_rate_identity_difference"].max() < 1e-10
    assert result["known_velocity_identity_difference"].max() < 1e-10
    assert result["forecast_relative_difference"].max() < 1e-3
    return result


def seal(output):
    write(output / "manifest.json", {
        "format": "glassbox-readout-attribution-v1",
        "files": {
            str(path.relative_to(output)): digest(path)
            for path in sorted(output.rglob("*")) if path.is_file() and path != output / "manifest.json"
        },
    })
    return digest(output / "manifest.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "verify"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-sha256")
    args = parser.parse_args()
    if args.mode == "verify":
        authenticate(args.output, args.manifest_sha256)
        for case in read(args.output / "protocol.json")["cases"]:
            recalculated = summarize(load(args.output / f"{case}.npz"))
            assert recalculated == read(args.output / f"{case}.json")
        print("Verified saved attribution without fitting or JAX recurrence.")
        return
    spec = read(PROTOCOL)
    for source in spec["sources"].values():
        authenticate(source["path"], source["manifest_sha256"])
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(PROTOCOL, args.output / "protocol.json")
    write(args.output / "binding.json", {
        "script_sha256": digest(Path(__file__)),
        "protocol_sha256": digest(PROTOCOL),
        "runtime": {"python": platform.python_version(), "jax": jax.__version__,
                    "numpy": np.__version__, "backend": jax.default_backend(),
                    "x64_ambient": bool(jax.config.x64_enabled)},
    })
    deadline = time.monotonic() + spec["budget"]["maximum_wall_s"]
    for case in spec["cases"]:
        data = analyze_case(case, spec, deadline)
        with (args.output / f"{case}.npz").open("wb") as stream:
            np.savez_compressed(stream, **data)
        summary = summarize(data)
        write(args.output / f"{case}.json", summary)
        print(case, "rate RMSE", summary["candidate"]["rate_rmse_250"], flush=True)
    print("manifest_sha256", seal(args.output), flush=True)


if __name__ == "__main__":
    main()
