"""Paired nonlinear-only temporal architecture screen; no-fit saved-data audit."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from collect_throw import authenticate, binding, check_source, checkpoint, seal
from evaluate_online import fingerprint, geometric
from run_dart import ROOT, write
from screen_accumulator import (
    ENVIRONMENT,
    INTERVALS,
    aggregate,
    finish_interval,
    pair,
    pause,
    schedule,
    summarize,
)
from verify_baseline import arrays, exact, observed, read, require
from verify_online_v8 import check_cache, session, verify_transition

PROTOCOL = ROOT / "docs/harness/nonlinear-temporal-v1.json"
CHANGED = {
    "src/glassbox/_dynamics.py",
    "src/glassbox/online.py",
    "src/glassbox/learner.py",
}


def learning_phases(data, info, reference):
    result = {}
    for label, sl in (
        ("first25", slice(0, 25)),
        ("25to100", slice(25, 100)),
        ("100toend", slice(100, None)),
    ):
        result[label] = (
            summarize({k: v[sl] for k, v in data.items()}, info, reference[sl])
            if len(reference[sl])
            else None
        )
    return result


def initial_activation(model, batch):
    import jax
    import jax.numpy as jnp

    from glassbox import _dynamics as core

    with jax.enable_x64(True):
        params, norms = jax.tree.map(jnp.asarray, (model.params, model.norms))
        past, issued, future = (
            jnp.asarray(batch[k])
            for k in ("past_states", "past_inputs", "future_inputs")
        )
        applied, history, hidden = core._history(
            params, norms, past, issued, model.delay_steps, model.dt_s
        )
        current = core.current_features(past[:, -1], future[:, 0], applied, norms)
        features = core.sampled_features(current, history, hidden)
        if params["w1"].shape[0] != features.shape[-1]:
            features = core.nonlinear_features(
                features, current.shape[-1], model.delay_steps
            )
        values = np.asarray(
            (features / norms.get("nonlinear_scale", norms["feature_scale"]))
            @ params["w1"]
            + params["b1"]
        )
    return dict(
        count=values.size,
        rms=float(np.sqrt(np.mean(values**2))),
        saturated_fraction=float(np.mean(abs(np.tanh(values)) > 0.95)),
    )


def run_case(spec, output, name, arm):
    import jax

    from glassbox import STATE_CHANNELS, OnlineFit, SequenceCollection, SequenceSegment

    parent = Path(spec["parent"]["path"])
    original = read(parent / name / "case.json")
    tape = arrays(parent / "inputs" / (name + ".npz"))
    begin, first, dt = (original[k] for k in ("begin", "first", "dt_s"))
    output.mkdir()
    prefix = SequenceCollection(
        (
            SequenceSegment(
                original["opaque_id"],
                "prefix",
                observed(tape["states"][begin : first + 1]),
                tape["commands"][begin:first].copy(),
                dt,
                begin,
            ),
        ),
        original["opaque_id"],
        STATE_CHANNELS,
        tuple(original["ordered_commands"]),
    )
    start = time.perf_counter()
    online = (
        OnlineFit.load(
            Path(spec["initial_sessions"]["path"])
            / "candidate"
            / name
            / "initial-online.npz"
        )
        if arm == "baseline"
        else OnlineFit(prefix)
    )
    jax.block_until_ready(online.model.params)
    info = {k: original[k] for k in ("id", "family", "dt_s", "first", "begin")}
    info.update(
        initialization_s=time.perf_counter() - start,
        parameters=sum(x.size for x in online.model.params.values()),
        initial_report=online.report,
        activation=initial_activation(online.model, online._bootstrap),
    )
    online.save(output / "initial-online.npz")
    h = online.model.history_steps
    history = observed(tape["states"][first - h : first + 1])
    issued = tape["commands"][first - h : first].copy()
    data = {
        k: [] for k in ("row", "prediction", "truth", "origin", "predict_s", "update_s")
    }
    try:
        with (output / "events.jsonl").open("x") as journal:
            for i, row in enumerate(range(first, len(tape["commands"]))):
                if i % 25 == 0:
                    pause(f"{name}:{i}")
                require(online.cursor == row, "cursor differs")
                before, identity = online.report, fingerprint(online.model)
                start = time.perf_counter()
                prediction = np.asarray(
                    online.predict(history, issued, tape["commands"][row : row + 1])
                )[0]
                predict_s = time.perf_counter() - start
                truth = observed(tape["states"][row + 1 : row + 2])[0]
                start = time.perf_counter()
                online.observe(row, tape["commands"][row], truth)
                updated = online.model
                jax.block_until_ready(updated.params)
                update_s = time.perf_counter() - start
                after = online.report
                verify_transition(before, after)
                journal.write(
                    json.dumps(
                        dict(
                            row=row,
                            before=identity,
                            after=fingerprint(updated),
                            report=after,
                        ),
                        allow_nan=False,
                    )
                    + "\n"
                )
                journal.flush()
                for key, value in dict(
                    row=row,
                    prediction=prediction,
                    truth=truth,
                    origin=history[-1],
                    predict_s=predict_s,
                    update_s=update_s,
                ).items():
                    data[key].append(value)
                history = np.concatenate((history[1:], truth[None]))
                issued = np.concatenate((issued[1:], tape["commands"][row : row + 1]))
        info["status"] = "complete"
    except Exception as error:
        info.update(status="failed", error=repr(error))
        raise
    finally:
        checkpoint(
            output / "predictions.npz", **{k: np.asarray(v) for k, v in data.items()}
        )
        online.save(output / "final-online.npz")
        info["final_report"] = online.report
        write(output / "case.json", info)


def worker(output, arm):
    import os

    pause("startup")
    spec, bound = read(PROTOCOL), binding(PROTOCOL)
    bound["environment"] = {k: os.environ.get(k) for k in ENVIRONMENT}
    for key in ("parent", "initial_sessions"):
        authenticate(Path(spec[key]["path"]), spec[key]["manifest_sha256"])
    for name, wanted in spec["package_files"].items():
        if arm == "baseline" or name not in CHANGED:
            require(
                bound["source"]["files"][name] == wanted,
                "unexpected package change: " + name,
            )
    output.mkdir(parents=True, exist_ok=False)
    write(output / "binding.json", bound)
    write(output / "protocol.json", spec)
    try:
        for name in spec["cases"]:
            run_case(spec, output / name, name, arm)
        check_source(bound)
    except Exception as error:
        write(output / "failure.json", dict(error=repr(error)))
        raise
    finally:
        finish_interval()
        write(output / "intervals.json", INTERVALS)
        print(
            json.dumps(
                dict(
                    event="done",
                    manifest_sha256=seal(output, "glassbox-temporal-worker-v1"),
                )
            ),
            flush=True,
        )


def verify(output, authority):
    authenticate(output, authority)
    spec = read(PROTOCOL)
    for key in ("parent", "initial_sessions"):
        authenticate(Path(spec[key]["path"]), spec[key]["manifest_sha256"])
    authorities = read(output / "workers.json")
    bounds = {}
    for arm in ("baseline", "candidate"):
        folder = output / arm
        authenticate(folder, authorities[arm]["manifest_sha256"])
        require(read(folder / "protocol.json") == spec, "protocol differs")
        bounds[arm] = read(folder / "binding.json")
        for name, wanted in spec["package_files"].items():
            if arm == "baseline" or name not in CHANGED:
                require(
                    bounds[arm]["source"]["files"][name] == wanted, "package differs"
                )
    for key in ("runtime", "environment", "protocol_sha256"):
        require(
            bounds["baseline"][key] == bounds["candidate"][key],
            "worker context differs",
        )
    for name in (
        "screen_temporal.py",
        "screen_accumulator.py",
        "collect_throw.py",
        "evaluate_online.py",
        "verify_baseline.py",
        "verify_online_v8.py",
        "run_dart.py",
    ):
        key = "scripts/" + name
        require(
            bounds["baseline"]["source"]["files"][key]
            == bounds["candidate"]["source"]["files"][key],
            "evaluator differs",
        )
    cases, timing, progress, initialization, work = [], [], {}, [], []
    for name in spec["cases"]:
        tape = arrays(Path(spec["parent"]["path"]) / "inputs" / (name + ".npz"))
        original = read(Path(spec["parent"]["path"]) / name / "case.json")
        runs, infos, initials = {}, {}, {}
        for arm in ("baseline", "candidate"):
            folder = output / arm / name
            info, data = read(folder / "case.json"), arrays(folder / "predictions.npz")
            require(info["status"] == "complete", "incomplete case")
            for key in ("id", "family", "dt_s", "first", "begin"):
                require(info[key] == original[key], "identity differs")
            rows = np.arange(info["first"], len(tape["commands"]))
            exact(data["row"], rows, "row roster")
            exact(data["truth"], observed(tape["states"][rows + 1]), "truth")
            exact(data["origin"], observed(tape["states"][rows]), "origin")
            require(all(np.isfinite(v).all() for v in data.values()), "nonfinite data")
            require(
                all((data[k] > 0).all() for k in ("predict_s", "update_s")),
                "invalid timing",
            )
            initial, iv, _, model = session(folder / "initial-online.npz")
            final, fv, _, final_model = session(folder / "final-online.npz")
            require(
                sum(v.size for k, v in iv.items() if k.startswith("param_"))
                == info["parameters"],
                "parameter count",
            )
            for meta, values in ((initial, iv), (final, fv)):
                check_cache(values, meta, tape, iv)
            events = [
                json.loads(s)
                for s in (folder / "events.jsonl").read_text().splitlines()
            ]
            require(len(events) == len(rows), "incomplete journal")
            previous = info["initial_report"]
            for row, event in zip(rows, events, strict=True):
                require(
                    event["row"] == row and event["before"] == model, "causal chain"
                )
                decision = verify_transition(previous, event["report"])
                if not decision["accepted"]:
                    require(event["after"] == model, "rejection changed model")
                previous, model = event["report"], event["after"]
            require(
                model == final_model
                and previous == info["final_report"]
                and final["cursor"] == int(rows[-1]) + 1,
                "final state",
            )
            runs[arm], infos[arm], initials[arm] = data, info, iv
        a, b = initials["candidate"], initials["baseline"]
        require(set(a) == set(b) | {"norm_nonlinear_scale"}, "initial array roster")
        for key in b:
            if key != "param_w1":
                exact(a[key], b[key], "unchanged initialization: " + key)
        c = 9 + 2 * len(b["norm_input_mean"])
        d = round(0.1 / original["dt_s"])
        expected = b["param_w1"]
        if d > 4:
            t = np.linspace(-1, 1, d)
            q, r = np.linalg.qr(np.stack([t**i for i in range(4)], axis=1))
            q *= np.sign(np.diag(r))
            physical = (
                expected[c : (d + 1) * c]
                / b["norm_feature_scale"][c : (d + 1) * c, None]
            ).reshape(d, c, -1)
            projected = np.einsum("dr,dch->rch", q, physical).reshape(4 * c, -1)
            expected = np.concatenate(
                (
                    expected[:c],
                    projected * a["norm_nonlinear_scale"][c : 5 * c, None],
                    expected[(d + 1) * c :],
                )
            )
        require(
            np.allclose(a["param_w1"], expected, rtol=1e-12, atol=1e-12),
            "initial projection differs",
        )
        require(
            np.allclose(
                runs["candidate"]["prediction"][0],
                runs["baseline"]["prediction"][0],
                rtol=2e-5,
                atol=2e-5,
            ),
            "initial physical prediction differs",
        )
        initialization.append(
            dict(
                id=name,
                unchanged_initial_arrays=True,
                projected_w1=True,
                activations={arm: info["activation"] for arm, info in infos.items()},
            )
        )
        info = dict(
            infos["candidate"], reference_parameters=infos["baseline"]["parameters"]
        )
        scores = summarize(runs["candidate"], info, runs["baseline"]["prediction"])
        cases.append(scores)
        work.append(
            dict(
                id=name,
                **{
                    arm: {
                        key: (
                            record["final_report"][key] - record["initial_report"][key]
                        )
                        / scores["count"]
                        for key in (
                            "optimizer_steps",
                            "cg_iterations",
                            "objective_calls",
                            "accepted_proposals",
                        )
                    }
                    for arm, record in infos.items()
                },
            )
        )
        baseline_scores = summarize(
            runs["baseline"],
            dict(
                infos["baseline"], reference_parameters=infos["baseline"]["parameters"]
            ),
            runs["baseline"]["prediction"],
        )
        timing.append(
            dict(
                id=name,
                family=info["family"],
                baseline=baseline_scores["update"],
                candidate=scores["update"],
                median_ratio=scores["update"]["warm_median_s"]
                / baseline_scores["update"]["warm_median_s"],
                p95_ratio=scores["update"]["warm_p95_s"]
                / baseline_scores["update"]["warm_p95_s"],
            )
        )
        for label, score in learning_phases(
            runs["candidate"], info, runs["baseline"]["prediction"]
        ).items():
            progress.setdefault(label, {})[name] = score
    require(
        sum(c["count"] for c in cases) == spec["online"]["rows"], "incomplete roster"
    )
    accuracy = aggregate(cases)
    # Shared helper's old acceptance limits do not define this new experiment.
    for key in ("gates", "screen_passed", "adopted", "claim"):
        accuracy.pop(key)
    families = {
        f: {
            k: geometric([r[k] for r in timing if r["family"] == f])
            for k in ("median_ratio", "p95_ratio")
        }
        for f in ("quad", "fixedwing")
    }
    limits, totals = spec["diagnostic_limits"], accuracy["aggregate"]
    checks = {
        k: totals[metric] <= limits[k]
        for k, metric in (
            ("equal_family_primary_ratio_max", "primary"),
            ("equal_family_velocity_ratio_max", "velocity_rmse_m_s"),
            ("equal_family_rate_ratio_max", "body_rate_rmse_rad_s"),
            ("equal_family_tail_ratio_max", "tail"),
            ("equal_family_orientation_ratio_max", "orientation_rmse_rad"),
        )
    }
    checks.update(
        each_family_primary_ratio_max=all(
            f["primary"] <= limits["each_family_primary_ratio_max"]
            for f in accuracy["families"].values()
        ),
        quad_parameter_ratio_max=all(
            c["parameter_ratio"] <= limits["quad_parameter_ratio_max"]
            for c in cases
            if c["family"] == "quad"
        ),
        quad_median_time_ratio_max=families["quad"]["median_ratio"]
        <= limits["quad_median_time_ratio_max"],
        fixedwing_median_time_ratio_max=families["fixedwing"]["median_ratio"]
        <= limits["fixedwing_median_time_ratio_max"],
    )
    return dict(
        accuracy=accuracy,
        timing=timing,
        timing_families=families,
        initialization=initialization,
        mean_work_per_observation=work,
        progress={
            label: dict(
                unavailable_cases=[
                    name for name, score in rows.items() if score is None
                ],
                **{
                    k: v
                    for k, v in aggregate(
                        [score for score in rows.values() if score is not None]
                    ).items()
                    if k in ("cases", "families", "aggregate")
                },
            )
            for label, rows in progress.items()
        },
        checks=checks,
        schedule=schedule(
            read(output / "baseline/intervals.json"),
            read(output / "candidate/intervals.json"),
            cases,
        ),
        adopted=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("worker", "pair", "verify"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=("baseline", "candidate"))
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--manifest-sha256")
    args = parser.parse_args()
    if args.mode == "worker":
        worker(args.output, args.arm)
    else:
        result = (
            pair(
                args.output,
                args.baseline_root,
                runner="screen_temporal.py",
                verifier=verify,
            )
            if args.mode == "pair"
            else verify(args.output, args.manifest_sha256)
        )
        if args.mode == "verify":
            write(args.output.parent / (args.output.name + "-verified.json"), result)
        print(
            json.dumps(
                {
                    k: v
                    for k, v in result.items()
                    if k in ("checks", "timing_families", "schedule", "adopted")
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
