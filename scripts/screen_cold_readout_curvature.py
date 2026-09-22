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
from jax.scipy.linalg import cho_solve
from run_dart import ROOT, write
from verify_baseline import arrays, exact, observed, read, require

from glassbox import (
    STATE_CHANNELS,
    OnlineFit,
    SequenceCollection,
    SequenceSegment,
    online,
)
from glassbox._dynamics import (
    GRAVITY,
    VehicleSequenceModel,
    _history,
    additive_readout_features,
    current_features,
    nonlinear_features,
    quadratic_features,
    rotation_exp,
    sampled_features,
    time_constants,
)

PROTOCOL = ROOT / "docs/harness/cold-readout-curvature-v1.json"
ARMS = ("baseline", "raw", "candidate", "frozen")
UPDATING = ARMS[:-1]


def coefficients(params):
    return jnp.concatenate(
        (params["linear"], params["quadratic"], params["bias"][None], params["w2"])
    )


def readout_features(params, norms, current, history, hidden):
    sampled, quadratic, compact = additive_readout_features(
        current, history, hidden
    )
    return jnp.concatenate(
        (
            sampled / norms["feature_scale"],
            quadratic / norms["quadratic_scale"],
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


class RawReadout:
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


def measurement(params, norms, past, inputs, command, following, *, delay, dt_s):
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
    return phi, target


def curvature_penalty(params, norms, data, scale, weights, n, *, delay, dt_s):
    diagonal = online._curvature_diagonal(
        {"quadratic": params["quadratic"]},
        norms,
        data,
        scale,
        weights,
        delay=delay,
        dt_s=dt_s,
    ).reshape((-1, 6))
    beta = dt_s * norms["output_scale"] / scale[0, :6]
    f, q = params["linear"].shape[0], params["quadratic"].shape[0]
    size = f + q + 1 + params["w2"].shape[0]
    return (
        jnp.zeros(size, dtype=diagonal.dtype)
        .at[f : f + q]
        .set(3 * n * diagonal[:, 0] / beta[0] ** 2)
    )


def information_update(gram, rhs, phi, target, penalty):
    gram = gram + jnp.outer(phi, phi)
    rhs = rhs + jnp.outer(phi, target)
    system = gram + jnp.diag(penalty)
    root = jnp.sqrt(jnp.diag(system))
    lower = jnp.linalg.cholesky(system / root[:, None] / root[None, :])
    mean = cho_solve((lower, True), rhs / root[:, None]) / root[:, None]
    return gram, rhs, mean


@partial(jax.jit, static_argnames=("delay", "dt_s"))
def penalized_update(
    params,
    norms,
    gram,
    rhs,
    n,
    data,
    scale,
    weights,
    past,
    inputs,
    command,
    following,
    *,
    delay,
    dt_s,
):
    phi, target = measurement(
        params, norms, past, inputs, command, following, delay=delay, dt_s=dt_s
    )
    penalty = curvature_penalty(
        params, norms, data, scale, weights, n, delay=delay, dt_s=dt_s
    )
    return *information_update(gram, rhs, phi, target, penalty), phi, target, penalty


class CurvedReadout:
    def __init__(self, prefix):
        self.session = OnlineFit(prefix)
        m = self.session.model
        self.count = 0
        with jax.enable_x64(True):
            self.mean = coefficients(m.params)
            self.ridge = 0.01 * self.session.report["initialized_transition_count"]
            self.gram = self.ridge * jnp.eye(len(self.mean), dtype=jnp.float64)
            self.rhs = self.ridge * self.mean
        self.states = prefix.segments[0].states[-m.history_steps - 1 :].copy()
        self.inputs = prefix.segments[0].inputs[-m.history_steps :].copy()
        jax.block_until_ready((self.mean, self.gram, self.rhs))

    def observe(self, row, command, following):
        require(row == self.session.cursor, "noncausal update")
        model = self.session.model
        states = np.concatenate((self.session._states[1:], following[None]))
        inputs = np.concatenate((self.session._inputs[1:], command[None]))
        new = online._windows(
            states,
            inputs,
            [model.history_steps],
            model.history_steps,
            self.session._horizon,
        )
        recent = {
            k: np.concatenate((self.session._recent[k], new[k]))[-32:]
            for k in online._FIELDS
        }
        data, weights = online._full_cache(self.session._bootstrap, recent)
        with jax.enable_x64(True):
            gram, rhs, mean, phi, target, penalty = penalized_update(
                model.params,
                model.norms,
                self.gram,
                self.rhs,
                self.count + 1,
                tuple(jnp.asarray(v) for v in data),
                jnp.asarray(self.session._scale),
                jnp.asarray(weights),
                self.states,
                self.inputs,
                command,
                following,
                delay=model.delay_steps,
                dt_s=model.dt_s,
            )
            saved = tuple(np.asarray(v) for v in (mean, phi, target, penalty))
            require(
                all(np.isfinite(v).all() for v in saved),
                "nonfinite regularized readout",
            )
        host_mean = saved[0]
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
        self.session._states, self.session._inputs, self.session._recent = (
            states,
            inputs,
            recent,
        )
        self.count += 1
        self.gram, self.rhs, self.mean = gram, rhs, mean
        self.states = np.concatenate((self.states[1:], following[None]))
        self.inputs = np.concatenate((self.inputs[1:], command[None]))
        return saved


def score(data, dt):
    result = {}
    for name in ARMS:
        result[name] = {
            label: metrics(data[name][a:b], data["truth"][a:b])
            for label, a, b in [
                ("all", 0, None),
                ("first16", 0, 16),
                ("after16", 16, None),
                *(
                    [("first100", 0, 100), ("after100", 100, None)]
                    if len(data["row"]) > 100
                    else []
                ),
                *(
                    [
                        ("pre4s", 0, int(np.searchsorted(data["time_s"], 4.0))),
                        ("post4s", int(np.searchsorted(data["time_s"], 4.0)), None),
                    ]
                    if "time_s" in data and data["time_s"][0] < 4 <= data["time_s"][-1]
                    else []
                ),
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
    raw = RawReadout(prefix) if "raw" in UPDATING else None
    if raw is not None:
        initial_times["raw"] = time.perf_counter() - tick
    tick = time.perf_counter()
    candidate = CurvedReadout(prefix)
    initial_times["candidate"] = time.perf_counter() - tick
    frozen = baseline.model
    sessions = {"baseline": baseline, "candidate": candidate.session}
    if raw is not None:
        sessions["raw"] = raw.session
    for key, value in frozen.arrays().items():
        for session in sessions.values():
            exact(value, session.model.arrays()[key], "same fresh initialization")
    checkpoint(output / "initial.npz", **frozen.arrays())
    frozen_predict = jax.jit(lambda x, u, future: frozen.rollout(x, u, future))
    h, horizon = frozen.history_steps, round(0.25 / dt)
    data = {
        key: []
        for key in (
            "row",
            "time_s",
            "truth",
            "origin",
            *ARMS,
            *(arm + "_update_s" for arm in UPDATING),
            *(arm + "_predict_s" for arm in UPDATING),
            "conditional_rows",
            "conditional_truth",
            *(arm + "_conditional" for arm in ARMS),
            *(("raw_phi", "raw_target") if raw is not None else ()),
            "phi",
            "target",
            "mean",
            "penalty",
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
            for arm in UPDATING:
                session = sessions[arm]
                tick = time.perf_counter()
                value = np.asarray(session.predict(x, u, future))[0]
                data[arm + "_predict_s"].append(time.perf_counter() - tick)
                require(np.isfinite(value).all(), "nonfinite forecast")
                data[arm].append(value)
            data["frozen"].append(np.asarray(frozen_predict(x, u, future))[0])
            if offset % 16 == 0 and row + horizon <= len(commands):
                data["conditional_rows"].append(row)
                future = commands[row : row + horizon]
                for arm in UPDATING:
                    session = sessions[arm]
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
            data["time_s"].append(tape["time_s"][row])
            data["truth"].append(following)
            data["origin"].append(states[row])
            shift = offset % len(UPDATING)
            order = UPDATING[shift:] + UPDATING[:shift]
            for arm in order:
                tick = time.perf_counter()
                if arm == "baseline":
                    baseline.observe(row, commands[row], following)
                    jax.block_until_ready(baseline.model.params)
                elif arm == "raw":
                    phi, target = raw.observe(row, commands[row], following)
                    jax.block_until_ready(raw.session.model.params)
                else:
                    mean, phi, target, penalty = candidate.observe(
                        row, commands[row], following
                    )
                    jax.block_until_ready(candidate.session.model.params)
                data[arm + "_update_s"].append(time.perf_counter() - tick)
                if arm == "raw":
                    data["raw_phi"].append(phi)
                    data["raw_target"].append(target)
                elif arm == "candidate":
                    for key, value in zip(
                        ("mean", "phi", "target", "penalty"),
                        (mean, phi, target, penalty),
                    ):
                        data[key].append(value)
    except Exception as exc:
        status, error = "failed", repr(exc)
    finally:
        data = {k: np.asarray(v) for k, v in data.items()}
        checkpoint(output / "predictions.npz", **data)
        checkpoint(output / "baseline-final.npz", **baseline.model.arrays())
        if raw is not None:
            checkpoint(
                output / "raw-final.npz",
                **raw.session.model.arrays(),
                inverse=np.asarray(raw.inverse),
            )
        checkpoint(
            output / "candidate-final.npz",
            **candidate.session.model.arrays(),
            gram=np.asarray(candidate.gram),
            rhs=np.asarray(candidate.rhs),
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
            candidate_work=dict(
                observations=candidate.count,
                domain_calls=candidate.count,
                cholesky_solves=candidate.count,
            ),
            initial_model_metadata=frozen.metadata(),
            feature_count=len(candidate.mean),
            readout_weights=int(candidate.mean.size),
            loss_scale=baseline._scale.tolist(),
            ridge=candidate.ridge,
        )
        if status == "complete":
            record["scores"] = score(data, dt)
        write(output / "result.json", record)
    require(status == "complete", error)
    print(name, status, flush=True)


def numpy_penalty(initial, info, states, commands, row):
    """Rebuild measured domains from tape indices, without session/cache code."""
    norms = {
        k.removeprefix("norm_"): v for k, v in initial.items() if k.startswith("norm_")
    }
    h, delay = (
        info["initial_model_metadata"][k] for k in ("history_steps", "delay_steps")
    )
    horizon = info["baseline_report"]["training_horizon_steps"]
    count = info["baseline_report"]["initialized_transition_count"]
    first = info["first"]
    roles = (
        list(range(first - count, first - horizon + 1))[-32:],
        list(range(first + 1 - horizon, row + 2 - horizon))[-32:],
    )
    energies = []
    for origins in roles:
        energy = []
        for origin in origins:
            x = states[origin - h + delay : origin + horizon]
            u = commands[origin - h + delay : origin + horizon]
            rotation = x[:, 6:].reshape((-1, 3, 3))
            motion = (
                np.c_[np.einsum("nji,nj->ni", rotation, x[:, :3]), x[:, 3:6]]
                - norms["body_mean"][:6]
            ) / norms["body_scale"][:6]
            support = norms["motion_bound_scale"] / 4
            excess = np.maximum(np.abs(motion) - support, 0)
            motion = np.where(
                np.abs(motion) <= support,
                motion,
                np.sign(motion)
                * (support + 3 * support * np.tanh(excess / (3 * support))),
            )
            issued = (u - norms["input_mean"]) / norms["input_scale"]
            energy.append(np.mean(np.c_[motion, issued] ** 2, axis=0))
        energies.append(np.mean(energy, axis=0))
    rms = np.maximum(1, np.sqrt(np.mean(energies, axis=0)))
    domain = np.r_[rms[:6], 1 / norms["body_scale"][6:9], rms[6:], rms[6:]]
    i, j = np.triu_indices(len(domain))
    factors = np.where(i == j, 2, np.sqrt(2)) * domain[i] * domain[j]
    penalty = np.zeros(info["feature_count"])
    f, q = len(initial["param_linear"]), len(initial["param_quadratic"])
    penalty[f : f + q] = (
        0.015 * (row - first + 1) * (factors / norms["quadratic_scale"]) ** 2
    )
    return penalty


def saved_coefficients(values):
    return np.concatenate(
        (
            values["param_linear"],
            values["param_quadratic"],
            values["param_bias"][None],
            values["param_w2"],
        )
    )


def verify_case(output, previous, parent, name, info, data, initial, final):
    old = arrays(previous / name / "predictions.npz")
    if "raw" not in UPDATING:
        count = len(old["row"])
        for key in (
            "row",
            "truth",
            "baseline",
            "candidate",
            "frozen",
            "phi",
            "target",
            "penalty",
            "mean",
        ):
            exact(data[key][:count], old[key], "short prefix replay " + key)
        query_count = len(old["conditional_rows"])
        for key in (
            "conditional_rows",
            "conditional_truth",
            "baseline_conditional",
            "candidate_conditional",
            "frozen_conditional",
        ):
            exact(data[key][:query_count], old[key], "short conditional replay " + key)
        before = arrays(previous / name / "initial.npz")
        for key in initial:
            exact(initial[key], before[key], "same initialization")
    else:
        for arm, old_arm in (
            ("baseline", "baseline"),
            ("raw", "candidate"),
            ("frozen", "frozen"),
        ):
            for suffix in ("", "_conditional"):
                exact(
                    data[arm + suffix],
                    old[old_arm + suffix],
                    "control prediction replay",
                )
        for key in ("phi", "target"):
            exact(data["raw_" + key], old[key], "raw measurement replay")
            np.testing.assert_allclose(
                data[key], data["raw_" + key], rtol=2e-12, atol=1e-10
            )
        for filename, old_filename in (
            ("initial.npz", "initial.npz"),
            ("baseline-final.npz", "baseline-final.npz"),
            ("raw-final.npz", "candidate-final.npz"),
        ):
            before, after = (
                arrays(previous / name / old_filename),
                arrays(output / name / filename),
            )
            require(set(before) == set(after), "control model keys differ")
            for key in before:
                exact(after[key], before[key], "control model replay")
    tape = arrays(parent / "inputs" / (name + ".npz"))
    states, commands = observed(tape["states"]), tape["commands"]
    start = saved_coefficients(initial)
    gram, rhs = info["ridge"] * np.eye(len(start)), info["ridge"] * start
    maximum_error = componentwise_error = 0.0
    for n, row in enumerate(data["row"]):
        penalty = numpy_penalty(initial, info, states, commands, row)
        np.testing.assert_allclose(data["penalty"][n], penalty, rtol=2e-12, atol=1e-10)
        phi, target, mean = data["phi"][n], data["target"][n], data["mean"][n]
        gram += np.outer(phi, phi)
        rhs += np.outer(phi, target)
        system = gram + np.diag(penalty)
        backward = np.linalg.norm(system @ mean - rhs, ord=np.inf) / max(
            np.linalg.norm(system, ord=np.inf) * np.linalg.norm(mean, ord=np.inf)
            + np.linalg.norm(rhs, ord=np.inf),
            1e-300,
        )
        require(backward <= 1e-10, "regularized normal equations not solved")
        maximum_error = max(maximum_error, float(backward))
        componentwise = np.max(
            np.abs(system @ mean - rhs)
            / np.maximum(np.abs(system) @ np.abs(mean) + np.abs(rhs), 1e-300)
        )
        require(componentwise <= 1e-10, "componentwise normal-equation error")
        componentwise_error = max(componentwise_error, float(componentwise))
    np.testing.assert_allclose(final["gram"], gram, rtol=2e-12, atol=1e-8)
    np.testing.assert_allclose(final["rhs"], rhs, rtol=2e-12, atol=1e-8)
    exact(saved_coefficients(final), data["mean"][-1], "final readout")
    lower = np.linalg.cholesky(system)
    require(
        np.isfinite(lower).all() and np.all(np.diag(lower) > 0),
        "nonpositive final precision",
    )
    require(
        info["candidate_work"]
        == dict.fromkeys(
            ("observations", "domain_calls", "cholesky_solves"), len(data["row"])
        ),
        "candidate work differs",
    )
    # Evaluate the same final penalized quadratic objective at three saved heads.
    beta = (
        info["dt_s"]
        * initial["norm_output_scale"]
        / np.array(info["loss_scale"])[0, :6]
    )

    def objective(mean):
        errors = data["phi"] @ mean - data["target"]
        terms = np.sum(errors**2, axis=0) + info["ridge"] * np.sum(
            (mean - start) ** 2, axis=0
        )
        terms += np.sum(penalty[:, None] * mean**2, axis=0)
        return float(np.sum(beta**2 * terms) / (6 * len(data["row"])))

    return dict(
        maximum_normal_equation_backward_error=maximum_error,
        maximum_componentwise_backward_error=componentwise_error,
        final_penalized_objective={
            "initial": objective(start),
            **(
                {
                    "raw": objective(
                        saved_coefficients(arrays(output / name / "raw-final.npz"))
                    )
                }
                if "raw" in UPDATING
                else {}
            ),
            "candidate": objective(data["mean"][-1]),
        },
        controls_exact_replay=True,
        penalty_numpy_reconstruction=True,
    )


def verify(output, authority):
    authenticate(output, authority)
    spec = read(output / "protocol.json")
    parent = Path(spec["parent"]["path"])
    authenticate(parent, spec["parent"]["manifest_sha256"])
    previous = Path(spec["previous_screen"]["path"])
    authenticate(previous, spec["previous_screen"]["manifest_sha256"])
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
        diagnosis = verify_case(
            output, previous, parent, name, info, data, initial, final
        )
        s = info["scores"]
        ratios = {}
        for phase in (k for k in s["candidate"] if k != "conditional"):
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
            for arm in UPDATING
        }
        cases.append(
            dict(
                id=name,
                family=info["family"],
                count=len(data["row"]),
                scores=s,
                diagnosis=diagnosis,
                **(
                    dict(
                        primary_to_raw=geometric(
                            [
                                s["candidate"]["all"][k] / max(s["raw"]["all"][k], 1e-9)
                                for k in ("velocity_rmse_m_s", "body_rate_rmse_rad_s")
                            ]
                        ),
                        horizon250_to_raw=geometric(
                            [
                                s["candidate"]["conditional"][
                                    str(round(0.25 / info["dt_s"]))
                                ][k]
                                / max(
                                    s["raw"]["conditional"][
                                        str(round(0.25 / info["dt_s"]))
                                    ][k],
                                    1e-9,
                                )
                                for k in ("velocity_rmse_m_s", "body_rate_rmse_rad_s")
                            ]
                        ),
                    )
                    if "raw" in UPDATING
                    else {}
                ),
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
            for key in (
                "primary",
                "horizon250",
                "median_ratio",
                *(("primary_to_raw", "horizon250_to_raw") if "raw" in UPDATING else ()),
            )
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
    global PROTOCOL, ARMS, UPDATING
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "verify"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-sha256")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run the separately frozen full-recording qualification",
    )
    args = parser.parse_args()
    if args.full:
        PROTOCOL = ROOT / "docs/harness/cold-readout-curvature-full-v1.json"
        ARMS = ("baseline", "candidate", "frozen")
        UPDATING = ARMS[:-1]
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
            seal(args.output, "glassbox-cold-readout-curvature-screen-v1"),
            flush=True,
        )


if __name__ == "__main__":
    main()
