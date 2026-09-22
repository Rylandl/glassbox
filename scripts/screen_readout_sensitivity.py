"""One frozen first-derivative penalty experiment; no production changes."""

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
from screen_cold_readout_curvature import (
    CurvedReadout,
    curvature_penalty,
    measurement,
    numpy_penalty,
)
from verify_baseline import arrays, exact, observed, read, require

from glassbox import (
    STATE_CHANNELS,
    OnlineFit,
    SequenceCollection,
    SequenceSegment,
    online,
)
from glassbox._dynamics import VehicleSequenceModel, nonlinear_features

PROTOCOL = ROOT / "docs/harness/readout-sensitivity-v1.json"
ETA = 0.01
ARMS = ("baseline", "curvature", "candidate", "frozen")
UPDATING = ARMS[:-1]


def sensitivity_bases(params, norms, delay):
    """Derivatives with respect to independent current and lag motion coordinates."""
    current = 9 + 2 * len(norms["input_mean"])
    size = len(params["linear"])
    basis = np.zeros((size, 6 * (delay + 1)))
    basis[:6, :6] = np.eye(6)
    for lag in range(delay):
        start = (lag + 1) * current
        basis[start : start + 6, :6] = -np.eye(6)
        basis[start : start + 6, (lag + 1) * 6 : (lag + 2) * 6] = np.eye(6)
    compact = nonlinear_features(basis.T, current, delay, np).T
    return (
        basis / norms["feature_scale"][:, None],
        params["w1"].T @ (compact / norms["nonlinear_scale"][:, None]),
    )


def feature_jacobian(phi, norms, linear, nonlinear):
    current = 9 + 2 * norms["input_mean"].shape[0]
    x = phi[:current] * norms["feature_scale"][:current]
    i, j = jnp.triu_indices(current)
    axes = jnp.arange(6)
    retained = ((i < 9) & (j < 9)) | ((i < 3) & (j >= 9))
    quadratic = (
        (i[:, None] == axes) * x[j, None] + (j[:, None] == axes) * x[i, None]
    ) * retained[:, None] / norms["quadratic_scale"][:, None]
    quadratic = jnp.pad(quadratic, ((0, 0), (0, linear.shape[1] - 6)))
    hidden = phi[-nonlinear.shape[0] :]
    return jnp.concatenate(
        (
            linear,
            quadratic,
            jnp.zeros((1, linear.shape[1])),
            (1 - hidden[:, None] ** 2) * nonlinear,
        )
    )


def numpy_jacobian(phi, initial, delay):
    """Independent NumPy calculus, including explicit temporal projection loops."""
    norms = {k[5:]: v for k, v in initial.items() if k.startswith("norm_")}
    params = {k[6:]: v for k, v in initial.items() if k.startswith("param_")}
    current = 9 + 2 * len(norms["input_mean"])
    f, width = len(params["linear"]), params["w1"].shape[1]
    k = 6 * (delay + 1)
    sampled = np.zeros((f, k))
    for axis in range(6):
        sampled[axis, axis] = 1
        for lag in range(delay):
            sampled[(lag + 1) * current + axis, axis] = -1
            sampled[(lag + 1) * current + axis, (lag + 1) * 6 + axis] = 1
    if delay <= 4:
        compact = sampled
    else:
        time_axis = np.linspace(-1, 1, delay)
        temporal, r = np.linalg.qr(np.stack([time_axis**i for i in range(4)], axis=1))
        temporal *= np.sign(np.diag(r))
        compact = np.zeros((len(norms["nonlinear_scale"]), k))
        compact[:current] = sampled[:current]
        for row in range(4):
            for lag in range(delay):
                compact[(row + 1) * current : (row + 2) * current] += (
                    temporal[lag, row]
                    * sampled[(lag + 1) * current : (lag + 2) * current]
                )
        compact[5 * current :] = sampled[(delay + 1) * current :]
    x = phi[:current] * norms["feature_scale"][:current]
    quadratic = np.zeros((len(params["quadratic"]), k))
    row = 0
    for i in range(current):
        for j in range(i, current):
            if i < 6:
                quadratic[row, i] += x[j] / norms["quadratic_scale"][row]
            if j < 6:
                quadratic[row, j] += x[i] / norms["quadratic_scale"][row]
            row += 1
    network = (1 - phi[-width:] ** 2)[:, None] * (
        params["w1"].T @ (compact / norms["nonlinear_scale"][:, None])
    )
    return np.concatenate(
        (
            sampled / norms["feature_scale"][:, None],
            quadratic,
            np.zeros((1, k)),
            network,
        )
    )


def sensitivity_solve(gram, rhs, sensitivity, phi, target, penalty, derivative):
    gram = gram + jnp.outer(phi, phi)
    rhs = rhs + jnp.outer(phi, target)
    sensitivity = sensitivity + derivative @ derivative.T
    system = gram + jnp.diag(penalty) + ETA * sensitivity
    root = jnp.sqrt(jnp.diag(system))
    lower = jnp.linalg.cholesky(system / root[:, None] / root[None, :])
    mean = cho_solve((lower, True), rhs / root[:, None]) / root[:, None]
    return gram, rhs, sensitivity, mean


@partial(jax.jit, static_argnames=("delay", "dt_s"))
def sensitivity_update(
    params,
    norms,
    gram,
    rhs,
    sensitivity,
    bases,
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
    derivative = feature_jacobian(phi, norms, *bases)
    return (
        *sensitivity_solve(gram, rhs, sensitivity, phi, target, penalty, derivative),
        phi,
        target,
        penalty,
    )


class SensitivityReadout(CurvedReadout):
    def __init__(self, prefix):
        super().__init__(prefix)
        model = self.session.model
        with jax.enable_x64(True):
            self.bases = tuple(
                jnp.asarray(x)
                for x in sensitivity_bases(model.params, model.norms, model.delay_steps)
            )
            self.sensitivity = jnp.zeros_like(self.gram)
        jax.block_until_ready((self.bases, self.sensitivity))

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
            gram, rhs, sensitivity, mean, phi, target, penalty = sensitivity_update(
                model.params,
                model.norms,
                self.gram,
                self.rhs,
                self.sensitivity,
                self.bases,
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
                "nonfinite sensitivity readout",
            )
        host = saved[0]
        f, q = len(model.params["linear"]), len(model.params["quadratic"])
        params = dict(
            model.params,
            linear=host[:f],
            quadratic=host[f : f + q],
            bias=host[f + q],
            w2=host[f + q + 1 :],
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
        self.gram, self.rhs, self.sensitivity, self.mean = gram, rhs, sensitivity, mean
        self.states = np.concatenate((self.states[1:], following[None]))
        self.inputs = np.concatenate((self.inputs[1:], command[None]))
        return saved


def score(data, dt):
    groups = [("all", 0, None), ("first16", 0, 16), ("after16", 16, None)]
    if len(data["row"]) > 100:
        groups += [("first100", 0, 100), ("after100", 100, None)]
    if data["time_s"][0] < 4 <= data["time_s"][-1]:
        split = int(np.searchsorted(data["time_s"], 4))
        groups += [("pre4s", 0, split), ("post4s", split, None)]
    result = {}
    for arm in ARMS:
        result[arm] = {
            label: metrics(data[arm][a:b], data["truth"][a:b]) for label, a, b in groups
        }
        result[arm]["conditional"] = {
            str(ms): metrics(
                data[arm + "_conditional"][:, round(ms / 1000 / dt) - 1],
                data["conditional_truth"][:, round(ms / 1000 / dt) - 1],
            )
            for ms in (50, 100, 150, 200, 250)
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
    objects, sessions, initialization = {}, {}, {}
    for arm, constructor in (
        ("baseline", OnlineFit),
        ("curvature", CurvedReadout),
        ("candidate", SensitivityReadout),
    ):
        tick = time.perf_counter()
        obj = constructor(prefix)
        session = obj if arm == "baseline" else obj.session
        jax.block_until_ready(session.model.params)
        initialization[arm] = time.perf_counter() - tick
        objects[arm], sessions[arm] = obj, session
    frozen = sessions["baseline"].model
    for session in sessions.values():
        for key, value in frozen.arrays().items():
            exact(value, session.model.arrays()[key], "fresh initialization differs")
    checkpoint(output / "initial.npz", **frozen.arrays())
    frozen_predict = jax.jit(lambda x, u, future: frozen.rollout(x, u, future))
    h, horizon = frozen.history_steps, round(0.25 / dt)
    keys = [
        "row",
        "time_s",
        "truth",
        "conditional_rows",
        "conditional_truth",
        *ARMS,
        *(arm + s for arm in UPDATING for s in ("_update_s", "_predict_s")),
        *(arm + "_conditional" for arm in ARMS),
        *(
            arm + "_" + s
            for arm in ("curvature", "candidate")
            for s in ("mean", "phi", "target", "penalty")
        ),
    ]
    data = {k: [] for k in keys}
    status, error = "complete", None
    try:
        for offset, row in enumerate(range(first, stop)):
            require(
                time.perf_counter() - started < spec["budget"]["wall_limit_s"],
                "wall budget exceeded",
            )
            x, u = states[row - h : row + 1], commands[row - h : row]
            for arm, session in sessions.items():
                tick = time.perf_counter()
                value = np.asarray(session.predict(x, u, commands[row : row + 1]))[0]
                data[arm + "_predict_s"].append(time.perf_counter() - tick)
                require(np.isfinite(value).all(), "nonfinite one-step prediction")
                data[arm].append(value)
            data["frozen"].append(
                np.asarray(frozen_predict(x, u, commands[row : row + 1]))[0]
            )
            if offset % 16 == 0 and row + horizon <= len(commands):
                data["conditional_rows"].append(row)
                future = commands[row : row + horizon]
                for arm, session in sessions.items():
                    value = np.asarray(session.predict(x, u, future))
                    require(np.isfinite(value).all(), "nonfinite conditional forecast")
                    data[arm + "_conditional"].append(value)
                data["frozen_conditional"].append(
                    np.asarray(frozen_predict(x, u, future))
                )
                data["conditional_truth"].append(states[row + 1 : row + horizon + 1])
            data["row"].append(row)
            data["time_s"].append(tape["time_s"][row])
            data["truth"].append(states[row + 1])
            shift = offset % len(UPDATING)
            for arm in UPDATING[shift:] + UPDATING[:shift]:
                tick = time.perf_counter()
                values = objects[arm].observe(row, commands[row], states[row + 1])
                jax.block_until_ready(sessions[arm].model.params)
                data[arm + "_update_s"].append(time.perf_counter() - tick)
                if arm != "baseline":
                    for key, value in zip(("mean", "phi", "target", "penalty"), values):
                        data[arm + "_" + key].append(value)
    except Exception as exc:
        status, error = "failed", repr(exc)
    finally:
        data = {k: np.asarray(v) for k, v in data.items()}
        checkpoint(output / "predictions.npz", **data)
        for arm, session in sessions.items():
            extra = (
                {}
                if arm == "baseline"
                else dict(
                    gram=np.asarray(objects[arm].gram), rhs=np.asarray(objects[arm].rhs)
                )
            )
            if arm == "candidate":
                extra["sensitivity"] = np.asarray(objects[arm].sensitivity)
            checkpoint(output / (arm + "-final.npz"), **session.model.arrays(), **extra)
        record = dict(
            id=name,
            family=info["family"],
            dt_s=dt,
            begin=begin,
            first=first,
            stop=stop,
            status=status,
            error=error,
            initialization_s=initialization,
            prefix_transitions=first - begin,
            first_prediction_tape_time_s=float(tape["time_s"][first]),
            baseline_report=sessions["baseline"].report,
            initial_model_metadata=frozen.metadata(),
            feature_count=len(objects["candidate"].mean),
            readout_weights=int(objects["candidate"].mean.size),
            ridge=objects["candidate"].ridge,
            loss_scale=sessions["baseline"]._scale.tolist(),
            candidate_work=dict(
                observations=objects["candidate"].count,
                derivative_evaluations=objects["candidate"].count,
                cholesky_solves=objects["candidate"].count,
            ),
        )
        if status == "complete":
            record["scores"] = score(data, dt)
        write(output / "result.json", record)
    require(status == "complete", error)
    print(
        name,
        status,
        {a: record["scores"][a]["conditional"]["250"] for a in UPDATING},
        flush=True,
    )


def aggregate(records):
    families = {}
    cases = []
    for info, data in records:
        s = score(data, info["dt_s"])
        row = dict(id=info["id"], family=info["family"], scores=s, timing={})
        for arm in UPDATING:
            times = data[arm + "_update_s"]
            row["timing"][arm] = dict(
                first_s=float(times[0]),
                median_s=float(np.median(times[1:])),
                p95_s=float(np.percentile(times[1:], 95)),
                total_s=float(times.sum()),
            )
        for control in ("baseline", "curvature"):
            ratios = {}
            for label in ("all", "250"):

                def select(arm, s=s, label=label):
                    return (
                        s[arm]["all"]
                        if label == "all"
                        else s[arm]["conditional"]["250"]
                    )

                for short, key in (
                    ("velocity", "velocity_rmse_m_s"),
                    ("rate", "body_rate_rmse_rad_s"),
                ):
                    ratios[label + "_" + short] = select("candidate")[key] / max(
                        select(control)[key], 1e-9
                    )
                ratios[label + "_primary"] = geometric(
                    [ratios[label + "_velocity"], ratios[label + "_rate"]]
                )
            ratios["time"] = (
                row["timing"]["candidate"]["median_s"]
                / row["timing"][control]["median_s"]
            )
            row[control + "_ratios"] = ratios
        cases.append(row)
    for family in ("quad", "fixedwing"):
        selected = [c for c in cases if c["family"] == family]
        if selected:
            families[family] = {
                control: {
                    k: geometric([c[control + "_ratios"][k] for c in selected])
                    for k in selected[0][control + "_ratios"]
                }
                for control in ("baseline", "curvature")
            }
    totals = {
        control: {
            k: geometric([f[control][k] for f in families.values()])
            for k in next(iter(families.values()))[control]
        }
        for control in ("baseline", "curvature")
    }
    b = totals["baseline"]
    checks = dict(
        one_step=b["all_primary"] <= 1.15,
        forecast250=b["250_primary"] <= 1.25,
        speed=b["time"] <= 0.5,
        family_primary=all(
            f["baseline"]["250_primary"] <= 1.5 for f in families.values()
        ),
        family_velocity=all(
            f["baseline"]["250_velocity"] <= 1.5 for f in families.values()
        ),
        family_rate=all(f["baseline"]["250_rate"] <= 1.5 for f in families.values()),
    )
    return dict(
        cases=cases, families=families, aggregate=totals, checks=checks, adopted=False
    )


def verify(root, authority):
    authenticate(root, authority)
    spec = read(root / "protocol.json")
    for source in spec["sources"].values():
        authenticate(Path(source["path"]), source["manifest_sha256"])
    previous = Path(spec["sources"]["control"]["path"])
    tapes = Path(spec["sources"]["tapes"]["path"])
    records, audits = [], {}
    for name in spec["cases"]:
        info = read(root / name / "result.json")
        require(info["status"] == "complete", "incomplete run")
        data = arrays(root / name / "predictions.npz")
        old = arrays(previous / name / "predictions.npz")
        initial = arrays(root / name / "initial.npz")
        for key, value in initial.items():
            exact(
                value,
                arrays(previous / name / "initial.npz")[key],
                "initial control replay",
            )
        for current, prior in (
            ("baseline", "baseline"),
            ("curvature", "candidate"),
            ("frozen", "frozen"),
        ):
            for suffix in ("", "_conditional"):
                exact(
                    data[current + suffix], old[prior + suffix], "control predictions"
                )
            if current != "frozen":
                actual = arrays(root / name / (current + "-final.npz"))
                for key, value in arrays(
                    previous / name / (prior + "-final.npz")
                ).items():
                    exact(actual[key], value, "control final replay")
        for key in ("phi", "target", "penalty", "mean"):
            exact(data["curvature_" + key], old[key], "curvature saved solve replay")
            if key != "mean":
                exact(data["candidate_" + key], old[key], "candidate design unchanged")
        phi, target, means, penalties = (
            data["candidate_" + k] for k in ("phi", "target", "mean", "penalty")
        )
        p = {k[6:]: v for k, v in initial.items() if k.startswith("param_")}
        mean0 = np.concatenate((p["linear"], p["quadratic"], p["bias"][None], p["w2"]))
        gram = np.eye(len(mean0)) * info["ridge"]
        rhs = mean0 * info["ridge"]
        sensitivity = np.zeros_like(gram)
        maximum = 0.0
        tape = arrays(tapes / "inputs" / (name + ".npz"))
        states, commands = observed(tape["states"]), tape["commands"].astype(float)
        for i, (x, y, m, penalty) in enumerate(zip(phi, target, means, penalties)):
            derivative = numpy_jacobian(
                x, initial, info["initial_model_metadata"]["delay_steps"]
            )
            sensitivity += derivative @ derivative.T
            gram += np.outer(x, x)
            rhs += np.outer(x, y)
            expected = numpy_penalty(
                initial, info, states, commands, int(data["row"][i])
            )
            np.testing.assert_allclose(penalty, expected, rtol=2e-12, atol=1e-10)
            system = gram + np.diag(penalty) + ETA * sensitivity
            residual = np.abs(system @ m - rhs)
            error = float(
                np.max(
                    residual
                    / np.maximum(np.abs(system) @ np.abs(m) + np.abs(rhs), 1e-300)
                )
            )
            require(error <= 1e-10, "stationarity backward error")
            maximum = max(maximum, error)
        final = arrays(root / name / "candidate-final.npz")
        for k, expected in (("gram", gram), ("rhs", rhs), ("sensitivity", sensitivity)):
            np.testing.assert_allclose(final[k], expected, rtol=2e-11, atol=1e-9)
        np.linalg.cholesky(gram + np.diag(penalties[-1]) + ETA * sensitivity)
        for key, value in initial.items():
            if key not in ("param_linear", "param_quadratic", "param_bias", "param_w2"):
                exact(final[key], value, "frozen feature changed")
        last = np.concatenate(
            (
                final["param_linear"],
                final["param_quadratic"],
                final["param_bias"][None],
                final["param_w2"],
            )
        )
        exact(last, means[-1], "final readout differs")
        require(score(data, info["dt_s"]) == info["scores"], "saved scores differ")
        require(all(np.isfinite(v).all() for v in data.values()), "nonfinite evidence")
        records.append((info, data))
        audits[name] = dict(
            count=len(phi),
            maximum_componentwise_backward_error=maximum,
            control_replay_exact=True,
        )
        print(name, "verified", maximum, flush=True)
    result = aggregate(records)
    result["audits"] = audits
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "verify"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-sha256")
    args = parser.parse_args()
    if args.mode == "verify":
        result = verify(args.output, args.manifest_sha256)
        write(args.output.parent / "verified.json", result)
        print({k: v for k, v in result.items() if k not in ("cases", "audits")})
        return
    spec, bound = read(PROTOCOL), binding(PROTOCOL)
    require(spec["penalty"]["strength"] == ETA, "strength differs from frozen contract")
    for source in spec["sources"].values():
        authenticate(Path(source["path"]), source["manifest_sha256"])
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "protocol.json", spec)
    write(args.output / "binding.json", bound)
    started = time.perf_counter()
    try:
        for name in spec["cases"]:
            run_case(
                Path(spec["sources"]["tapes"]["path"]),
                args.output / name,
                name,
                spec,
                started,
            )
    finally:
        check_source(bound)
        print(
            "manifest_sha256",
            seal(args.output, "glassbox-readout-sensitivity-v1"),
            flush=True,
        )


if __name__ == "__main__":
    main()
