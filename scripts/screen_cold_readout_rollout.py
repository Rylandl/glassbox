"""Bounded no-pretraining diagnostic; the public learner is unchanged."""

import argparse
import time
from pathlib import Path

import jax
import numpy as np
from collect_throw import authenticate, binding, check_source, checkpoint, seal
from evaluate_online import geometric, metrics
from run_dart import ROOT, write
from verify_baseline import arrays, exact, observed, read, require

from glassbox import (
    STATE_CHANNELS,
    OnlineFit,
    SequenceCollection,
    SequenceSegment,
    online,
)

PROTOCOL = ROOT / "docs/harness/cold-readout-rollout-v1.json"
ARMS = ("baseline", "candidate", "frozen")


READOUT = ("linear", "quadratic", "bias", "w2")


class Readout(OnlineFit):
    """Experimental parameter subset; all public observation machinery is shared."""

    def _propose(self, *args, **kwargs):
        return online._proposal(*args, **kwargs, readout_only=True)


def verify_feature_functions(initial, final):
    """Coordinate compensation may change weights but never their function."""
    changing = {"feature_scale", "nonlinear_scale", "quadratic_scale", "output_scale"}
    for key, value in initial.items():
        if key in ("param_w1", "param_memory"):
            scale_key = (
                "norm_nonlinear_scale" if key == "param_w1" else "norm_feature_scale"
            )
            a = value / initial[scale_key][: len(value), None]
            b = final[key] / final[scale_key][: len(value), None]
            np.testing.assert_allclose(a, b, rtol=2e-13, atol=2e-13)
        elif key.removeprefix("param_") in READOUT:
            continue
        elif key.removeprefix("norm_") in changing:
            require(np.all(final[key] >= value), "scale shrank")
        else:
            exact(final[key], value, "frozen feature function")
    for key in ("norm_feature_scale", "norm_nonlinear_scale"):
        exact(initial[key][-8:], final[key][-8:], "fixed hidden coordinate scales")


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
        exact(value, candidate.model.arrays()[key], "same fresh initialization")
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
        )
    }
    work = {arm: [] for arm in ("baseline", "candidate")}
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
                ("candidate", candidate),
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
                    ("candidate", candidate),
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
                session = baseline if arm == "baseline" else candidate
                session.observe(row, commands[row], following)
                jax.block_until_ready(session.model.params)
                data[arm + "_update_s"].append(time.perf_counter() - tick)
                work[arm].append(session.report["last_proposal"])
    except Exception as exc:
        status, error = "failed", repr(exc)
    finally:
        data = {k: np.asarray(v) for k, v in data.items()}
        checkpoint(output / "predictions.npz", **data)
        checkpoint(output / "baseline-final.npz", **baseline.model.arrays())
        checkpoint(
            output / "candidate-final.npz",
            **candidate.model.arrays(),
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
            candidate_report=candidate.report,
            proposal_records=work,
            initial_model_metadata=frozen.metadata(),
            readout_weights=sum(frozen.params[k].size for k in READOUT),
            total_weights=sum(v.size for v in frozen.params.values()),
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
        verify_feature_functions(initial, final)
        old = arrays(previous / name / "predictions.npz")
        for key in ("baseline", "baseline_conditional", "frozen", "frozen_conditional"):
            exact(data[key], old[key], "baseline replay " + key)
        for filename in ("initial.npz", "baseline-final.npz"):
            before, after = (
                arrays(previous / name / filename),
                arrays(output / name / filename),
            )
            require(set(before) == set(after), "model keys differ")
            for key in before:
                exact(before[key], after[key], "baseline model replay " + key)
        for arm in ("baseline", "candidate"):
            records = info["proposal_records"][arm]
            report = info[arm + "_report"]
            require(len(records) == len(data["row"]), "proposal count")
            for record in records:
                online._validate_proposal_report(record)
            require(report["observations"] == len(records), "observations differ")
            require(
                report["cg_iterations"] == 16 * len(records), "solver budget differs"
            )
            require(
                report["objective_calls"]
                == sum(1 + r["trial_evaluations"] for r in records),
                "objective work differs",
            )
            require(
                report["accepted_proposals"]
                == sum(r["selected_alpha"] > 0 for r in records),
                "acceptance count differs",
            )
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
        speed=totals["median_ratio"] <= 0.8,
        one_step=totals["primary"] <= 1.15,
        forecast250=totals["horizon250"] <= 1.25,
        families=all(f["primary"] <= 1.5 for f in families.values()),
    )
    return dict(
        cases=cases,
        families=families,
        aggregate=totals,
        checks=checks,
        adopted=False,
        baseline_exact_replay=True,
        frozen_feature_functions=True,
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
            seal(args.output, "glassbox-cold-readout-rollout-screen-v1"),
            flush=True,
        )


if __name__ == "__main__":
    main()
