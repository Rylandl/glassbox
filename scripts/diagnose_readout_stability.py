"""Frozen diagnostics; only the unchanged reference learner is replayed."""

import argparse
import math
import time
from dataclasses import replace
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from collect_throw import authenticate, binding, check_source, checkpoint, seal
from evaluate_online import metrics
from run_dart import ROOT, write
from scipy.linalg import solve_triangular
from scipy.stats import spearmanr
from verify_baseline import arrays, exact, observed, read, require

from glassbox import (
    STATE_CHANNELS,
    OnlineFit,
    SequenceCollection,
    SequenceSegment,
    online,
)
from glassbox import _dynamics as core

PROTOCOL = ROOT / "docs/harness/readout-stability-diagnosis-v1.json"
VARIANTS = (
    "baseline",
    "candidate",
    "no_quadratic_rate",
    "no_nonlinear_rate",
    "no_lag_rate",
    "candidate_force_baseline_rate",
    "baseline_force_candidate_rate",
)


def model_from(info, values):
    return core.VehicleSequenceModel.from_arrays(
        info["initial_model_metadata"],
        {k: v for k, v in values.items() if k.startswith(("param_", "norm_"))},
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


def ablate(model, name):
    params = {k: v.copy() for k, v in model.params.items()}
    if name == "no_quadratic_rate":
        params["quadratic"][:, 3:] = 0
    elif name == "no_nonlinear_rate":
        params["w2"][:, 3:] = 0
    elif name == "no_lag_rate":
        c = 9 + 2 * len(model.norms["input_mean"])
        params["linear"][c : (model.delay_steps + 1) * c, 3:] = 0
    else:
        raise ValueError(name)
    return replace(model, params=params)


@partial(jax.jit, static_argnames=("delay", "dt_s"))
def carry_from(params, norms, past, inputs, *, delay, dt_s):
    return (
        past[None, -1],
        *core._history(params, norms, past[None], inputs[None], delay, dt_s),
    )


@partial(jax.jit, static_argnames=("dt_s",))
def step(params, norms, carry, command, *, dt_s):
    return core.physical_step(params, norms, carry[0], command[None], *carry[1:], dt_s)


def perturb(carry, delta):
    state, *latent = carry
    rotation = state[:, 6:].reshape((-1, 3, 3)) @ core.rotation_exp(delta[6:9])[None]
    state = jnp.concatenate(
        (state[:, :6] + delta[None, :6], rotation.reshape((-1, 9))), axis=-1
    )
    result, cursor = [state], 9
    for value in latent:
        result.append(value + delta[cursor : cursor + value.size].reshape(value.shape))
        cursor += value.size
    return tuple(result)


def difference(value, reference):
    state, origin = value[0], reference[0]
    relative = origin[:, 6:].reshape((-1, 3, 3)).swapaxes(-1, -2) @ state[
        :, 6:
    ].reshape((-1, 3, 3))
    skew = (relative - relative.swapaxes(-1, -2)) * 0.5
    theta = jnp.stack((skew[0, 2, 1], skew[0, 0, 2], skew[0, 1, 0]))
    return jnp.concatenate(
        (
            (state[:, :6] - origin[:, :6]).ravel(),
            theta,
            *[(a - b).ravel() for a, b in zip(value[1:], reference[1:])],
        )
    )


@partial(jax.jit, static_argnames=("dt_s",))
def jacobian(params, norms, carry, command, *, dt_s):
    nominal = step(params, norms, carry, command, dt_s=dt_s)
    size = 9 + sum(v.size for v in carry[1:])
    return jax.jacfwd(
        lambda d: difference(
            step(params, norms, perturb(carry, d), command, dt_s=dt_s), nominal
        )
    )(jnp.zeros(size, dtype=carry[0].dtype))


@jax.jit
def features(params, norms, carry, command):
    current = core.current_features(carry[0], command[None], carry[1], norms)
    sampled = core.sampled_features(current, carry[2], carry[3])
    compact = core.nonlinear_features(sampled, current.shape[-1], carry[2].shape[-2])
    return jnp.concatenate(
        (
            sampled / norms["feature_scale"],
            core.quadratic_features(current) / norms["quadratic_scale"],
            jnp.ones((1, 1), dtype=current.dtype),
            jnp.tanh(compact / norms["nonlinear_scale"] @ params["w1"] + params["b1"]),
        ),
        axis=-1,
    )[0]


def precision_root(phi, ridge, penalty=None):
    blocks = [np.eye(phi.shape[1]) * np.sqrt(ridge), phi]
    if penalty is not None:
        blocks.append(np.diag(np.sqrt(penalty)))
    return np.linalg.qr(np.concatenate(blocks), mode="r")


def leverage(root, phi):
    projected = solve_triangular(root.T, np.asarray(phi).T, lower=True)
    return np.sum(projected**2, axis=0)


@partial(jax.jit, static_argnames=("dt_s",))
def dual_step(lp, ln, rp, rn, left, right, command, *, dt_s):
    """Left force + right angular acceleration; preserve each branch's latent state."""
    count = max(1, math.ceil(dt_s / core.MAX_SUBSTEP_S))
    duration = dt_s / count
    command = command[None]
    anchors = [
        core.current_features(c[0], command, c[1], n)
        for c, n in ((left, ln), (right, rn))
    ]
    projections = [
        core._prepare_head(p, n, a, c[2], c[3])
        for p, n, a, c in ((lp, ln, anchors[0], left), (rp, rn, anchors[1], right))
    ]

    def acceleration(state, applied, p, n, anchor, projection):
        current = core.current_features(state, command, applied, n)
        return core._acceleration(
            p, n, current, projection[0] + (current - anchor) @ projection[1]
        )

    def combined(state, la, ra):
        force = acceleration(state, la, lp, ln, anchors[0], projections[0])
        angular = acceleration(state, ra, rp, rn, anchors[1], projections[1])
        return jnp.concatenate((force[:, :3], angular[:, 3:]), axis=-1)

    def advance(_, value):
        state, la, ra = value
        v, w, R = state[:, :3], state[:, 3:6], state[:, 6:].reshape((-1, 3, 3))
        first = combined(state, la, ra)
        Rhalf = R @ core.rotation_exp(0.5 * duration * w)
        vhalf = v + 0.5 * duration * (
            jnp.asarray(core.GRAVITY) + jnp.einsum("bij,bj->bi", R, first[:, :3])
        )
        whalf = w + 0.5 * duration * first[:, 3:]
        lh = command + (la - command) * jnp.exp(
            -0.5 * duration / core.time_constants(lp)
        )
        rh = command + (ra - command) * jnp.exp(
            -0.5 * duration / core.time_constants(rp)
        )
        middle = combined(
            jnp.concatenate((vhalf, whalf, Rhalf.reshape((-1, 9))), axis=-1), lh, rh
        )
        vn = v + duration * (
            jnp.asarray(core.GRAVITY) + jnp.einsum("bij,bj->bi", Rhalf, middle[:, :3])
        )
        wn = w + duration * middle[:, 3:]
        Rn = R @ core.rotation_exp(duration * whalf)
        return (
            jnp.concatenate((vn, wn, Rn.reshape((-1, 9))), axis=-1),
            command + (la - command) * jnp.exp(-duration / core.time_constants(lp)),
            command + (ra - command) * jnp.exp(-duration / core.time_constants(rp)),
        )

    state, la, ra = jax.lax.fori_loop(0, count, advance, (left[0], left[1], right[1]))

    def finish(c, p, n, anchor, applied):
        return (
            state,
            applied,
            jnp.concatenate((c[2][:, 1:], anchor[:, None]), axis=1),
            core.memory_step(p, n, anchor, c[3], dt_s),
        )

    return finish(left, lp, ln, anchors[0], la), finish(right, rp, rn, anchors[1], ra)


def predict(model, past, inputs, future):
    with jax.enable_x64(False):
        params, norms = jax.tree.map(
            lambda v: jnp.asarray(v, dtype=jnp.float32), (model.params, model.norms)
        )
        return np.asarray(
            online._predict(
                params,
                norms,
                jnp.asarray(past[None], dtype=jnp.float32),
                jnp.asarray(inputs[None], dtype=jnp.float32),
                jnp.asarray(future[None], dtype=jnp.float32),
                delay=model.delay_steps,
                dt_s=model.dt_s,
            )
        )[0]


def recover_reference(name, parent, source, output):
    info = read(source / name / "result.json")
    original = read(parent / name / "case.json")
    tape = arrays(parent / "inputs" / (name + ".npz"))
    data = arrays(source / name / "predictions.npz")
    states, commands = observed(tape["states"]), tape["commands"].astype(float)
    first, begin, dt = (info[k] for k in ("first", "begin", "dt_s"))
    prefix = SequenceCollection(
        (
            SequenceSegment(
                original["opaque_id"],
                "prefix",
                states[begin : first + 1],
                commands[begin:first],
                dt,
                begin,
            ),
        ),
        original["opaque_id"],
        STATE_CHANNELS,
        tuple(original["ordered_commands"]),
    )
    session = OnlineFit(prefix)
    h, snapshots = session.model.history_steps, {}
    for index, row in enumerate(data["row"]):
        past, inputs = states[row - h : row + 1], commands[row - h : row]
        exact(
            np.asarray(session.predict(past, inputs, commands[row : row + 1]))[0],
            data["baseline"][index],
            "reference one-step replay",
        )
        where = np.flatnonzero(data["conditional_rows"] == row)
        if len(where):
            exact(
                np.asarray(session.predict(past, inputs, commands[row : row + 5])),
                data["baseline_conditional"][where[0]],
                "reference conditional replay",
            )
            snapshots[int(row)] = session.model
            checkpoint(output / f"baseline-{row}.npz", **session.model.arrays())
        session.observe(int(row), commands[row], states[row + 1])
    final = arrays(source / name / "baseline-final.npz")
    for key, value in session.model.arrays().items():
        exact(value, final[key], "reference final replay")
    return info, data, states, commands, snapshots


def summarize(data):
    result = {
        name: {
            str((step + 1) * 50): metrics(data[name][:, step], data["truth"][:, step])
            for step in range(5)
        }
        for name in VARIANTS
    }
    jac = {}
    for arm in ("baseline", "candidate"):
        jac[arm] = {}
        for path in ("measured", "rolled"):
            value = data[f"{arm}_{path}_jacobian"]
            spectral = np.max(np.abs(np.linalg.eigvals(value)), axis=-1)
            partial_rho = np.max(
                np.abs(np.linalg.eigvals(value[:, :, :9, :9])), axis=-1
            )
            singular = np.linalg.svd(value, compute_uv=False)[..., 0]
            products = []
            for query in value:
                product = np.eye(value.shape[-1])
                for a in query:
                    product = a @ product
                products.append(np.linalg.svd(product, compute_uv=False)[0])
            jac[arm][path] = dict(
                spectral_radius=spectral.tolist(),
                physical_partial_radius=partial_rho.tolist(),
                singular_value=singular.tolist(),
                five_step_product_gain=products,
                median_radius=float(np.median(spectral)),
                maximum_radius=float(spectral.max()),
            )
    ratio = data["rolled_leverage"] / np.maximum(data["measured_leverage"], 1e-300)
    rate_error = np.linalg.norm(
        data["candidate"][:, :, 3:6] - data["truth"][:, :, 3:6], axis=-1
    )
    association = float(
        spearmanr(
            np.log10(np.maximum(data["rolled_leverage"], 1e-300)).ravel(),
            rate_error.ravel(),
        ).statistic
    )
    return dict(
        scores=result,
        jacobians=jac,
        leverage=dict(
            rolled_to_measured=ratio.tolist(),
            median_ratio=float(np.median(ratio)),
            maximum_ratio=float(np.max(ratio)),
            spearman_log_leverage_rate_error=association,
        ),
        maximum_fd_relative_error=float(data["fd_error"].max()),
    )


def diagnose_case(name, spec, output):
    parent, source = (Path(spec["sources"][k]["path"]) for k in ("tapes", "full"))
    output.mkdir()
    info, data, states, commands, references = recover_reference(
        name, parent, source, output
    )
    initial = model_from(info, arrays(source / name / "initial.npz"))
    result = {
        k: []
        for k in (
            *VARIANTS,
            "truth",
            "rows",
            "fd_error",
            "measured_leverage",
            "rolled_leverage",
            "measured_regularized_leverage",
            "rolled_regularized_leverage",
            "measured_support",
            "rolled_support",
            *[
                f"{a}_{p}_jacobian"
                for a in ("baseline", "candidate")
                for p in ("measured", "rolled")
            ],
        )
    }
    fd_errors = []
    with jax.enable_x64(True):
        for query, row in enumerate(data["conditional_rows"]):
            row = int(row)
            n = row - info["first"]
            candidate = initial if n == 0 else with_mean(initial, data["mean"][n - 1])
            baseline = references[row]
            h = candidate.history_steps
            past, inputs, future = (
                states[row - h : row + 1],
                commands[row - h : row],
                commands[row : row + 5],
            )
            exact(
                predict(candidate, past, inputs, future),
                data["candidate_conditional"][query],
                "candidate saved forecast replay",
            )
            root = precision_root(data["phi"][:n], info["ridge"])
            penalty = np.zeros(len(root)) if n == 0 else data["penalty"][n - 1]
            regularized = precision_root(data["phi"][:n], info["ridge"], penalty)
            for name_arm, model in [("baseline", baseline), ("candidate", candidate)]:
                p, norm = jax.tree.map(jnp.asarray, (model.params, model.norms))
                carry = carry_from(
                    p,
                    norm,
                    jnp.asarray(past),
                    jnp.asarray(inputs),
                    delay=model.delay_steps,
                    dt_s=model.dt_s,
                )
                forecasts, jacs, teacher_jacs, phir, phit, supr, supt = (
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                )
                for j, command in enumerate(future):
                    teacher = carry_from(
                        p,
                        norm,
                        jnp.asarray(states[row + j - h : row + j + 1]),
                        jnp.asarray(commands[row + j - h : row + j]),
                        delay=model.delay_steps,
                        dt_s=model.dt_s,
                    )
                    J = np.asarray(
                        jacobian(p, norm, carry, jnp.asarray(command), dt_s=model.dt_s)
                    )
                    teacher_jacs.append(
                        np.asarray(
                            jacobian(
                                p, norm, teacher, jnp.asarray(command), dt_s=model.dt_s
                            )
                        )
                    )
                    jacs.append(J)
                    if query in (0, 7) and j == 0:
                        direction = np.random.default_rng(391).normal(size=J.shape[0])
                        direction /= np.linalg.norm(direction)
                        nominal = step(
                            p, norm, carry, jnp.asarray(command), dt_s=model.dt_s
                        )
                        plus = step(
                            p,
                            norm,
                            perturb(carry, jnp.asarray(direction * 1e-7)),
                            jnp.asarray(command),
                            dt_s=model.dt_s,
                        )
                        minus = step(
                            p,
                            norm,
                            perturb(carry, jnp.asarray(-direction * 1e-7)),
                            jnp.asarray(command),
                            dt_s=model.dt_s,
                        )
                        fd = np.asarray(
                            (difference(plus, nominal) - difference(minus, nominal))
                            / (2e-7)
                        )
                        error = np.linalg.norm(J @ direction - fd) / max(
                            1, np.linalg.norm(fd)
                        )
                        require(
                            error < 2e-5, "actual snapshot finite difference failed"
                        )
                        fd_errors.append(float(error))
                    if name_arm == "candidate":
                        phir.append(
                            np.asarray(features(p, norm, carry, jnp.asarray(command)))
                        )
                        phit.append(
                            np.asarray(features(p, norm, teacher, jnp.asarray(command)))
                        )
                        for value, dest in ((carry, supr), (teacher, supt)):
                            body = (
                                np.asarray(core._body_features(value[0]))[0, :6]
                                - model.norms["body_mean"][:6]
                            ) / model.norms["body_scale"][:6]
                            dest.append(
                                float(
                                    np.max(
                                        np.abs(body)
                                        / (model.norms["motion_bound_scale"] / 4)
                                    )
                                )
                            )
                    carry = step(p, norm, carry, jnp.asarray(command), dt_s=model.dt_s)
                    forecasts.append(np.asarray(carry[0])[0])
                result[name_arm].append(forecasts)
                result[f"{name_arm}_rolled_jacobian"].append(jacs)
                result[f"{name_arm}_measured_jacobian"].append(teacher_jacs)
                reference = data[name_arm + "_conditional"][query]
                require(
                    np.linalg.norm(np.asarray(forecasts) - reference)
                    / max(1, np.linalg.norm(reference))
                    < 1e-4,
                    "float64 diagnostic differs materially from saved forecast",
                )
                if name_arm == "candidate":
                    for label, ph in (("rolled", phir), ("measured", phit)):
                        result[label + "_leverage"].append(leverage(root, ph))
                        result[label + "_regularized_leverage"].append(
                            leverage(regularized, ph)
                        )
                    result["rolled_support"].append(supr)
                    result["measured_support"].append(supt)
            for variant in VARIANTS[2:5]:
                model = ablate(candidate, variant)
                result[variant].append(
                    np.asarray(
                        online._predict(
                            model.params,
                            model.norms,
                            jnp.asarray(past[None]),
                            jnp.asarray(inputs[None]),
                            jnp.asarray(future[None]),
                            delay=model.delay_steps,
                            dt_s=model.dt_s,
                        )
                    )[0]
                )
            for variant, left_model, right_model in (
                (VARIANTS[5], candidate, baseline),
                (VARIANTS[6], baseline, candidate),
            ):
                lp, ln, rp, rn = jax.tree.map(
                    jnp.asarray,
                    (
                        left_model.params,
                        left_model.norms,
                        right_model.params,
                        right_model.norms,
                    ),
                )
                left = carry_from(
                    lp, ln, jnp.asarray(past), jnp.asarray(inputs), delay=2, dt_s=0.05
                )
                right = carry_from(
                    rp, rn, jnp.asarray(past), jnp.asarray(inputs), delay=2, dt_s=0.05
                )
                forecast = []
                for command in future:
                    left, right = dual_step(
                        lp, ln, rp, rn, left, right, jnp.asarray(command), dt_s=0.05
                    )
                    forecast.append(np.asarray(left[0])[0])
                result[variant].append(forecast)
            result["truth"].append(data["conditional_truth"][query])
            result["rows"].append(row)
    result["fd_error"] = fd_errors
    result = {k: np.asarray(v) for k, v in result.items()}
    require(all(np.isfinite(v).all() for v in result.values()), "nonfinite diagnostic")
    checkpoint(output / "diagnostics.npz", **result)
    report = summarize(result)
    report["baseline_replay_updates"] = len(data["row"])
    report["candidate_fits"] = 0
    write(output / "result.json", report)
    print(name, "complete", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "verify"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-sha256")
    args = parser.parse_args()
    if args.mode == "verify":
        authenticate(args.output, args.manifest_sha256)
        spec = read(args.output / "protocol.json")
        for name in spec["cases"]:
            data = arrays(args.output / name / "diagnostics.npz")
            report = read(args.output / name / "result.json")
            recomputed = summarize(data)
            require(
                all(report[k] == v for k, v in recomputed.items()),
                "diagnostic scores differ",
            )
        print(
            "Saved diagnostic metrics and payloads verified without replay or fitting."
        )
        return
    spec, bound = read(PROTOCOL), binding(PROTOCOL)
    for source in spec["sources"].values():
        authenticate(Path(source["path"]), source["manifest_sha256"])
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "protocol.json", spec)
    write(args.output / "binding.json", bound)
    started = time.perf_counter()
    try:
        for name in spec["cases"]:
            require(
                time.perf_counter() - started < spec["budget"]["wall_limit_s"],
                "wall budget exceeded",
            )
            diagnose_case(name, spec, args.output / name)
    finally:
        check_source(bound)
        print(
            "manifest_sha256",
            seal(args.output, "glassbox-readout-stability-diagnosis-v1"),
            flush=True,
        )


if __name__ == "__main__":
    main()
