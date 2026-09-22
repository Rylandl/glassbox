"""Bounded no-pretraining diagnostic; the public learner is unchanged."""

import argparse
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from collect_throw import authenticate, binding, check_source, checkpoint, seal
from evaluate_online import geometric, metrics
from run_dart import ROOT, write
from verify_baseline import arrays, exact, observed, read, require

from glassbox import STATE_CHANNELS, OnlineFit, SequenceCollection, SequenceSegment
from glassbox._dynamics import (
    GRAVITY,
    VehicleSequenceModel,
    _history,
    current_features,
    nonlinear_features,
    quadratic_features,
    rotation_exp,
    sampled_features,
    time_constants,
)

PROTOCOL = ROOT / "docs/harness/cold-readout-v1.json"
ARMS = ("baseline", "candidate", "frozen")


def coefficients(params):
    return jnp.concatenate(
        (params["linear"], params["quadratic"], params["bias"][None], params["w2"])
    )


def readout_features(params, norms, current, history, hidden):
    sampled = sampled_features(current, history, hidden)
    compact = nonlinear_features(sampled, current.shape[-1], history.shape[-2])
    return jnp.concatenate(
        (
            sampled / norms["feature_scale"],
            quadratic_features(current) / norms["quadratic_scale"],
            jnp.ones((*current.shape[:-1], 1), dtype=current.dtype),
            jnp.tanh(
                (compact / norms["nonlinear_scale"]) @ params["w1"] + params["b1"]
            ),
        ),
        axis=-1,
    )


def recursive_update(inverse, mean, phi, target):
    projected = inverse @ phi
    denominator = 1 + phi @ projected
    mean = mean + jnp.outer(projected / denominator, target - phi @ mean)
    inverse = inverse - jnp.outer(projected, projected) / denominator
    return (inverse + inverse.T) * 0.5, mean


@partial(jax.jit, static_argnames=("delay", "dt_s"))
def update(
    params, norms, inverse, mean, past, inputs, command, following, *, delay, dt_s
):
    applied, history, hidden = _history(
        params, norms, past[None], inputs[None], delay, dt_s
    )
    start = past[-1]
    rotation = start[6:].reshape(3, 3) @ rotation_exp(
        dt_s * 0.25 * (start[3:6] + following[3:6])
    )
    midpoint = jnp.concatenate(((start[:6] + following[:6]) * 0.5, rotation.reshape(9)))
    filtered = command + (applied[0] - command) * jnp.exp(
        -0.5 * dt_s / time_constants(params)
    )
    current = current_features(midpoint, command, filtered, norms)
    phi = readout_features(params, norms, current, history[0], hidden[0])
    target = (
        jnp.concatenate(
            (
                rotation.T
                @ ((following[:3] - start[:3]) / dt_s - jnp.asarray(GRAVITY)),
                (following[3:6] - start[3:6]) / dt_s,
            )
        )
        / norms["output_scale"]
    )
    return *recursive_update(inverse, mean, phi, target), phi, target


class Readout:
    def __init__(self, prefix):
        self.session = OnlineFit(prefix)
        m = self.session.model
        with jax.enable_x64(True):
            self.mean = coefficients(m.params)
            self.inverse = jnp.eye(len(self.mean), dtype=jnp.float64) / (
                0.01 * self.session.report["initialized_transition_count"]
            )
        self.states = prefix.segments[0].states[-m.history_steps - 1 :].copy()
        self.inputs = prefix.segments[0].inputs[-m.history_steps :].copy()
        jax.block_until_ready((self.mean, self.inverse))

    def observe(self, row, command, following):
        require(row == self.session.cursor, "noncausal update")
        model = self.session.model
        with jax.enable_x64(True):
            inverse, mean, phi, target = update(
                model.params,
                model.norms,
                self.inverse,
                self.mean,
                self.states,
                self.inputs,
                command,
                following,
                delay=model.delay_steps,
                dt_s=model.dt_s,
            )
            host_mean = np.asarray(mean)
            require(
                np.isfinite(host_mean).all() and np.isfinite(np.asarray(inverse)).all(),
                "nonfinite readout",
            )
        f, q = model.params["linear"].shape[0], model.params["quadratic"].shape[0]
        params = dict(
            model.params,
            linear=host_mean[:f],
            quadratic=host_mean[f : f + q],
            bias=host_mean[f + q],
            w2=host_mean[f + q + 1 :],
        )
        self.session._model = VehicleSequenceModel(
            model.dt_s, model.history_steps, model.delay_steps, params, model.norms
        )
        self.session._cursor += 1
        self.inverse, self.mean = inverse, mean
        self.states = np.concatenate((self.states[1:], following[None]))
        self.inputs = np.concatenate((self.inputs[1:], command[None]))
        return np.asarray(phi), np.asarray(target)


def score(data, dt):
    result = {}
    for name in ARMS:
        result[name] = {
            label: metrics(data[name][a:b], data["truth"][a:b])
            for label, a, b in [
                ("all", 0, None),
                ("first16", 0, 16),
                ("after16", 16, None),
            ]
        }
        result[name]["conditional"] = {
            str(h): metrics(
                data[name + "_conditional"][:, h - 1],
                data["conditional_truth"][:, h - 1],
            )
            for h in (max(1, round(0.05 / dt)), round(0.25 / dt))
        }
    return result


def run_case(parent, output, name, spec, started):
    info = read(parent / name / "case.json")
    tape = arrays(parent / "inputs" / (name + ".npz"))
    states, commands = observed(tape["states"]), tape["commands"].astype(float)
    first, begin, dt = (info[k] for k in ("first", "begin", "dt_s"))
    stop = min(first + spec["budget"]["updates_per_case_max"], len(commands))
    prefix = SequenceCollection(
        (
            SequenceSegment(
                info["opaque_id"],
                "prefix",
                states[begin : first + 1],
                commands[begin:first],
                dt,
                begin,
            ),
        ),
        info["opaque_id"],
        STATE_CHANNELS,
        tuple(info["ordered_commands"]),
    )
    output.mkdir()
    initial_times = {}
    tick = time.perf_counter()
    baseline = OnlineFit(prefix)
    initial_times["baseline"] = time.perf_counter() - tick
    tick = time.perf_counter()
    candidate = Readout(prefix)
    initial_times["candidate"] = time.perf_counter() - tick
    frozen = baseline.model
    for key, value in frozen.arrays().items():
        exact(value, candidate.session.model.arrays()[key], "same fresh initialization")
    checkpoint(output / "initial.npz", **frozen.arrays())
    frozen_predict = jax.jit(lambda x, u, future: frozen.rollout(x, u, future))
    h, horizon = frozen.history_steps, round(0.25 / dt)
    data = {
        key: []
        for key in (
            "row",
            "truth",
            "origin",
            *ARMS,
            "baseline_update_s",
            "candidate_update_s",
            "baseline_predict_s",
            "candidate_predict_s",
            "conditional_rows",
            "conditional_truth",
            "baseline_conditional",
            "candidate_conditional",
            "frozen_conditional",
            "phi",
            "target",
        )
    }
    status, error = "complete", None
    try:
        for offset, row in enumerate(range(first, stop)):
            require(
                time.perf_counter() - started < spec["budget"]["wall_limit_s"],
                "wall budget exceeded",
            )
            x, u, future = (
                states[row - h : row + 1],
                commands[row - h : row],
                commands[row : row + 1],
            )
            # Only currently available history/issued command enters the one-step prediction.
            for arm, session in [
                ("baseline", baseline),
                ("candidate", candidate.session),
            ]:
                tick = time.perf_counter()
                value = np.asarray(session.predict(x, u, future))[0]
                data[arm + "_predict_s"].append(time.perf_counter() - tick)
                require(np.isfinite(value).all(), "nonfinite forecast")
                data[arm].append(value)
            data["frozen"].append(np.asarray(frozen_predict(x, u, future))[0])
            if offset % 16 == 0 and row + horizon <= len(commands):
                data["conditional_rows"].append(row)
                future = commands[row : row + horizon]
                for arm, session in [
                    ("baseline", baseline),
                    ("candidate", candidate.session),
                ]:
                    value = np.asarray(session.predict(x, u, future))
                    require(np.isfinite(value).all(), "nonfinite conditional forecast")
                    data[arm + "_conditional"].append(value)
                data["frozen_conditional"].append(
                    np.asarray(frozen_predict(x, u, future))
                )
                data["conditional_truth"].append(states[row + 1 : row + horizon + 1])
            # Reveal the next observation only after every prediction above.
            following = states[row + 1]
            data["row"].append(row)
            data["truth"].append(following)
            data["origin"].append(states[row])
            order = (
                ("baseline", "candidate")
                if offset % 2 == 0
                else ("candidate", "baseline")
            )
            for arm in order:
                tick = time.perf_counter()
                if arm == "baseline":
                    baseline.observe(row, commands[row], following)
                    jax.block_until_ready(baseline.model.params)
                else:
                    phi, target = candidate.observe(row, commands[row], following)
                    jax.block_until_ready(candidate.session.model.params)
                    data["phi"].append(phi)
                    data["target"].append(target)
                data[arm + "_update_s"].append(time.perf_counter() - tick)
    except Exception as exc:
        status, error = "failed", repr(exc)
    finally:
        data = {k: np.asarray(v) for k, v in data.items()}
        checkpoint(output / "predictions.npz", **data)
        checkpoint(output / "baseline-final.npz", **baseline.model.arrays())
        checkpoint(
            output / "candidate-final.npz",
            **candidate.session.model.arrays(),
            inverse=np.asarray(candidate.inverse),
        )
        record = dict(
            id=name,
            family=info["family"],
            dt_s=dt,
            begin=begin,
            first=first,
            stop=stop,
            status=status,
            error=error,
            initialization_s=initial_times,
            prefix_transitions=first - begin,
            first_prediction_tape_time_s=float(tape["time_s"][first]),
            baseline_report=baseline.report,
            initial_model_metadata=frozen.metadata(),
            feature_count=len(candidate.mean),
            readout_weights=int(candidate.mean.size),
        )
        if status == "complete":
            record["scores"] = score(data, dt)
        write(output / "result.json", record)
    require(status == "complete", error)
    print(name, status, flush=True)


def verify(output, authority):
    authenticate(output, authority)
    spec = read(output / "protocol.json")
    parent = Path(spec["parent"]["path"])
    authenticate(parent, spec["parent"]["manifest_sha256"])
    cases = []
    for name in spec["cases"]:
        info = read(output / name / "result.json")
        require(info["status"] == "complete", "incomplete roster")
        data = arrays(output / name / "predictions.npz")
        tape = arrays(parent / "inputs" / (name + ".npz"))
        truth = observed(tape["states"])
        exact(data["row"], np.arange(info["first"], info["stop"]), "row roster")
        exact(data["truth"], truth[data["row"] + 1], "scored truth")
        for i, row in enumerate(data["conditional_rows"]):
            exact(
                data["conditional_truth"][i],
                truth[row + 1 : row + 1 + data["conditional_truth"].shape[1]],
                "conditional truth",
            )
        require(
            all(np.isfinite(a).all() for a in data.values()), "nonfinite saved data"
        )
        require(score(data, info["dt_s"]) == info["scores"], "scores differ")
        initial = arrays(output / name / "initial.npz")
        final = arrays(output / name / "candidate-final.npz")
        for key, value in initial.items():
            if key not in ("param_linear", "param_quadratic", "param_bias", "param_w2"):
                exact(final[key], value, "frozen feature/normalizer")
        s = info["scores"]
        ratios = {}
        for phase in ("all", "first16", "after16"):
            ratios[phase] = {
                k: s["candidate"][phase][k] / max(s["baseline"][phase][k], 1e-9)
                for k in s["candidate"][phase]
            }
            ratios[phase]["primary"] = geometric(
                [
                    ratios[phase][k]
                    for k in ("velocity_rmse_m_s", "body_rate_rmse_rad_s")
                ]
            )
        for horizon in s["candidate"]["conditional"]:
            c, b = (
                s["candidate"]["conditional"][horizon],
                s["baseline"]["conditional"][horizon],
            )
            ratios["horizon_" + horizon] = {k: c[k] / max(b[k], 1e-9) for k in c}
            ratios["horizon_" + horizon]["primary"] = geometric(
                [
                    ratios["horizon_" + horizon][k]
                    for k in ("velocity_rmse_m_s", "body_rate_rmse_rad_s")
                ]
            )
        times = {
            arm: {
                "median_s": float(np.median(data[arm + "_update_s"][1:])),
                "p95_s": float(np.quantile(data[arm + "_update_s"][1:], 0.95)),
                "first_update_s": float(data[arm + "_update_s"][0]),
                "compute_s": info["initialization_s"][arm]
                + float(data[arm + "_update_s"].sum() + data[arm + "_predict_s"].sum()),
            }
            for arm in ("baseline", "candidate")
        }
        cases.append(
            dict(
                id=name,
                family=info["family"],
                count=len(data["row"]),
                scores=s,
                ratios=ratios,
                timing=times,
                median_ratio=times["candidate"]["median_s"]
                / times["baseline"]["median_s"],
                primary=ratios["all"]["primary"],
                horizon250=ratios["horizon_" + str(round(0.25 / info["dt_s"]))][
                    "primary"
                ],
            )
        )
    families = {
        f: {
            key: geometric([c[key] for c in cases if c["family"] == f])
            for key in ("primary", "horizon250", "median_ratio")
        }
        for f in ("quad", "fixedwing")
    }
    totals = {
        key: geometric([f[key] for f in families.values()])
        for key in next(iter(families.values()))
    }
    checks = dict(
        speed=totals["median_ratio"] <= 0.5,
        one_step=totals["primary"] <= 1.15,
        forecast250=totals["horizon250"] <= 1.25,
        families=all(f["primary"] <= 1.5 for f in families.values()),
    )
    return dict(
        cases=cases, families=families, aggregate=totals, checks=checks, adopted=False
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "verify"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-sha256")
    args = parser.parse_args()
    if args.mode == "verify":
        result = verify(args.output, args.manifest_sha256)
        write(args.output.parent / "verified.json", result)
        print({k: v for k, v in result.items() if k != "cases"})
        return
    spec, bound = read(PROTOCOL), binding(PROTOCOL)
    parent = Path(spec["parent"]["path"])
    authenticate(parent, spec["parent"]["manifest_sha256"])
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "protocol.json", spec)
    write(args.output / "binding.json", bound)
    started = time.perf_counter()
    try:
        for name in spec["cases"]:
            run_case(parent, args.output / name, name, spec, started)
    finally:
        check_source(bound)
        print(
            "manifest_sha256",
            seal(args.output, "glassbox-cold-readout-screen-v1"),
            flush=True,
        )


if __name__ == "__main__":
    main()
