"""One frozen architecture screen; saved verification makes no model calls."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from itertools import pairwise
from pathlib import Path

import numpy as np
from collect_throw import authenticate, binding, check_source, checkpoint, seal
from evaluate_online import (
    fingerprint,
    forecast_diagnostics,
    geometric,
    latency,
    metrics,
)
from run_dart import ROOT, write
from verify_baseline import arrays, digest, exact, observed, read, require
from verify_online_v8 import check_cache, session, verify_transition

PROTOCOL = ROOT / "docs/harness/accumulator-qualification-online-v1.json"
PRIMARY = ("velocity_rmse_m_s", "body_rate_rmse_rad_s")


def summarize(data, info, reference):
    scores = [metrics(a, data["truth"]) for a in (data["prediction"], reference)]
    require(all(scores), "nonfinite or empty predictions")
    details = [
        forecast_diagnostics(a, data["truth"], data["origin"], info["dt_s"])
        for a in (data["prediction"], reference)
    ]
    ratios = {key: scores[0][key] / max(scores[1][key], 1e-9) for key in scores[0]}
    ratios["primary"] = geometric([ratios[key] for key in PRIMARY])
    ratios["tail"] = geometric(
        [
            details[0]["tail"][key]["upper_decile_rmse"]
            / max(details[1]["tail"][key]["upper_decile_rmse"], 1e-9)
            for key in ("velocity_m_s", "body_rate_rad_s")
        ]
    )
    ratios["rotation_rate"] = details[0]["rotation_rate"][
        "truth_relative_rmse_rad_s"
    ] / max(details[1]["rotation_rate"]["truth_relative_rmse_rad_s"], 1e-9)
    return dict(
        id=info["id"],
        family=info["family"],
        count=len(reference),
        candidate=scores[0],
        reference=scores[1],
        ratios=ratios,
        diagnostics=details[0],
        parameters=info["parameters"],
        parameter_ratio=info["parameters"] / info["reference_parameters"],
        predict=latency(data["predict_s"]),
        update=latency(data["update_s"]),
    )


def aggregate(cases):
    keys = (*PRIMARY, "primary", "tail", "orientation_rmse_rad", "rotation_rate")
    families = {
        f: {
            key: geometric([c["ratios"][key] for c in cases if c["family"] == f])
            for key in keys
        }
        for f in ("quad", "fixedwing")
    }
    totals = {key: geometric([f[key] for f in families.values()]) for key in keys}
    gates = dict(
        primary=totals["primary"] <= 1.01,
        family_primary=all(f["primary"] <= 1.02 for f in families.values()),
        tails=totals["tail"] <= 1.02,
        orientation=totals["orientation_rmse_rad"] <= 1.02,
        rotation_rate=totals["rotation_rate"] <= 1.02,
        velocity=totals["velocity_rmse_m_s"] <= 1.02,
        body_rate=totals["body_rate_rmse_rad_s"] <= 1.02,
    )
    return dict(
        cases=cases,
        families=families,
        aggregate=totals,
        gates=gates,
        screen_passed=all(gates.values()),
        adopted=False,
        claim="Online accuracy only; paired timing evaluated separately; no offline or Dart qualification.",
    )


def run_case(parent, output, name, arm):
    import jax

    from glassbox import STATE_CHANNELS, OnlineFit, SequenceCollection, SequenceSegment

    ref = parent / name
    info = read(ref / "case.json")
    tape = arrays(parent / "inputs" / (name + ".npz"))
    reference = arrays(ref / "predictions.npz")
    parent_events = [
        json.loads(line)
        for line in (ref / "events.jsonl").read_text().splitlines()
        if json.loads(line)["phase"] == "assimilated"
    ]
    begin, first, dt = info["begin"], info["first"], info["dt_s"]
    output.mkdir()
    data = {
        k: [] for k in ("row", "prediction", "truth", "origin", "predict_s", "update_s")
    }
    online = None
    try:
        prefix = SequenceCollection(
            (
                SequenceSegment(
                    info["opaque_id"],
                    "prefix",
                    observed(tape["states"][begin : first + 1]),
                    tape["commands"][begin:first].copy(),
                    dt,
                    begin,
                ),
            ),
            info["opaque_id"],
            STATE_CHANNELS,
            tuple(info["ordered_commands"]),
        )
        tick = time.perf_counter()
        online = (
            OnlineFit.load(ref / "initial-online.npz")
            if arm == "baseline"
            else OnlineFit(prefix)
        )
        jax.block_until_ready(online.model.params)
        info = {k: info[k] for k in ("id", "family", "dt_s", "first", "begin")}
        info["initialization_s"] = time.perf_counter() - tick
        info["parameters"] = sum(v.size for v in online.model.params.values())
        initial_reference = arrays(ref / "initial-online.npz")
        info["reference_parameters"] = sum(
            v.size for k, v in initial_reference.items() if k.startswith("param_")
        )
        info["initial_report"] = online.report
        online.save(output / "initial-online.npz")
        h = online.model.history_steps
        history = observed(tape["states"][first - h : first + 1])
        issued = tape["commands"][first - h : first].copy()
        with (output / "events.jsonl").open("x") as journal:
            for i, k in enumerate(range(first, len(tape["commands"]))):
                if i % 25 == 0:
                    pause(f"{name}:{i}")
                require(online.cursor == k, "cursor mismatch")
                before, before_model = online.report, fingerprint(online.model)
                tick = time.perf_counter()
                prediction = np.asarray(
                    online.predict(history, issued, tape["commands"][k : k + 1])
                )[0]
                predict_s = time.perf_counter() - tick
                # The learner has not received the next observation before this prediction.
                truth = observed(tape["states"][k + 1 : k + 2])[0]
                tick = time.perf_counter()
                online.observe(k, tape["commands"][k], truth)
                updated = online.model
                jax.block_until_ready(updated.params)
                update_s = time.perf_counter() - tick
                after = online.report
                if arm == "baseline":
                    exact(prediction, reference["candidate"][i], "baseline prediction")
                    require(
                        fingerprint(updated) == parent_events[i]["model"]
                        and after == parent_events[i]["report"],
                        "baseline update differs",
                    )
                verify_transition(before, after)
                record = dict(
                    row=k, before=before_model, after=fingerprint(updated), report=after
                )
                journal.write(json.dumps(record, allow_nan=False) + "\n")
                journal.flush()
                for key, value in dict(
                    row=k,
                    prediction=prediction,
                    truth=truth,
                    origin=history[-1],
                    predict_s=predict_s,
                    update_s=update_s,
                ).items():
                    data[key].append(value)
                history = np.concatenate((history[1:], truth[None]))
                issued = np.concatenate((issued[1:], tape["commands"][k : k + 1]))
        info["status"] = "complete"
    except Exception as error:
        info.update(status="failed", error=repr(error))
    finally:
        checkpoint(
            output / "predictions.npz", **{k: np.asarray(v) for k, v in data.items()}
        )
        if online is not None:
            online.save(output / "final-online.npz")
            info["final_report"] = online.report
            if arm == "candidate":
                info["memory_times_s"] = (
                    0.001 + np.logaddexp(0, online.model.params["raw_memory_tau"])
                ).tolist()
        write(output / "case.json", info)
    require(info["status"] == "complete", info.get("error", "incomplete case"))
    return summarize(arrays(output / "predictions.npz"), info, reference["candidate"])


def run(output, arm):
    pause("startup")
    spec, bound = read(PROTOCOL), binding(PROTOCOL)
    bound["environment"] = {key: os.environ.get(key) for key in ENVIRONMENT}
    hashes = read(ROOT / spec["parent"]["package_hashes_from"])["parent"][
        "package_files"
    ]
    for name, wanted in hashes.items():
        if arm == "baseline" or name not in CHANGED:
            require(
                bound["source"]["files"][name] == wanted,
                "unexpected package change: " + name,
            )
    parent = Path(spec["parent"]["path"])
    authenticate(parent, spec["parent"]["manifest_sha256"])
    require(
        bound["runtime"] == read(parent / "binding.json")["runtime"],
        "runtime differs from v8",
    )
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(PROTOCOL, output / "protocol.json")
    write(output / "binding.json", bound)
    write(output / "arm.json", dict(arm=arm))
    cases, failures = [], []
    try:
        for name in spec["cases"]:
            try:
                result = run_case(parent, output / name, name, arm)
                cases.append(result)
            except Exception as error:
                failures.append(dict(case=name, error=repr(error)))
        check_source(bound)
        authenticate(parent, spec["parent"]["manifest_sha256"])
        result = (
            aggregate(cases)
            if not failures
            else dict(
                failures=failures, cases=cases, screen_passed=False, adopted=False
            )
        )
        write(output / "summary.json", result)
    finally:
        finish_interval()
        write(output / "intervals.json", INTERVALS)
        authority = seal(output, "glassbox-accumulator-worker-v1")
        print(
            json.dumps(
                dict(event="done", output=str(output), manifest_sha256=authority)
            ),
            flush=True,
        )
    return result


def verify_worker(output, authority):
    authenticate(output, authority)
    spec = read(output / "protocol.json")
    require(spec == read(PROTOCOL), "protocol differs")
    require(
        read(output / "binding.json")["protocol_sha256"] == digest(PROTOCOL),
        "protocol binding differs",
    )
    parent = Path(spec["parent"]["path"])
    authenticate(parent, spec["parent"]["manifest_sha256"])
    cases = []
    for name in spec["cases"]:
        case = output / name
        data, info = arrays(case / "predictions.npz"), read(case / "case.json")
        original = read(parent / name / "case.json")
        for key in ("id", "family", "dt_s", "first", "begin"):
            require(info[key] == original[key], "case identity differs")
        reference = arrays(parent / name / "predictions.npz")
        tape = arrays(parent / "inputs" / (name + ".npz"))
        rows = reference["origin"]
        exact(data["row"], rows, "row roster")
        exact(data["truth"], reference["truth"], "paired truth")
        exact(data["truth"], observed(tape["states"][rows + 1]), "source truth")
        exact(data["origin"], observed(tape["states"][rows]), "source origin")
        require(all(np.isfinite(v).all() for v in data.values()), "nonfinite data")
        require(
            all((data[k] > 0).all() for k in ("predict_s", "update_s")),
            "invalid timing",
        )
        initial, iv, _, model = session(case / "initial-online.npz")
        final, fv, _, final_model = session(case / "final-online.npz")
        old = arrays(parent / name / "initial-online.npz")
        for key in old:
            if key.startswith(("bootstrap_", "recent_")) or key in (
                "scale",
                "states",
                "inputs",
            ):
                exact(iv[key], old[key], "initial observed cache " + key)
        for meta, values in ((initial, iv), (final, fv)):
            check_cache(values, meta, tape, iv)
        require(
            sum(v.size for k, v in iv.items() if k.startswith("param_"))
            == info["parameters"],
            "parameter count differs",
        )
        require(
            sum(v.size for k, v in old.items() if k.startswith("param_"))
            == info["reference_parameters"],
            "reference parameter count differs",
        )
        events = [
            json.loads(s) for s in (case / "events.jsonl").read_text().splitlines()
        ]
        require(len(events) == len(rows), "incomplete events")
        previous = info["initial_report"]
        for row, event in zip(rows, events, strict=True):
            require(
                event["row"] == row and event["before"] == model, "causal chain differs"
            )
            decision = verify_transition(previous, event["report"])
            if not decision["accepted"]:
                require(event["after"] == model, "rejected update changed model")
            previous, model = event["report"], event["after"]
        require(
            model == final_model and previous == info["final_report"],
            "final state differs",
        )
        arm = read(output / "arm.json")["arm"]
        if arm == "candidate":
            exact(
                0.001 + np.logaddexp(0, fv["param_raw_memory_tau"]),
                np.asarray(info["memory_times_s"]),
                "memory times",
            )
        else:
            exact(data["prediction"], reference["candidate"], "baseline predictions")
            for endpoint in ("initial", "final"):
                expected = arrays(parent / name / (endpoint + "-online.npz"))
                actual = arrays(case / (endpoint + "-online.npz"))
                require(set(actual) == set(expected), "baseline session keys")
                for key in actual:
                    if key == "metadata":
                        require(
                            actual[key].item() == expected[key].item(),
                            "baseline metadata differs",
                        )
                    else:
                        exact(actual[key], expected[key], "baseline session " + key)
        require(final["cursor"] == int(rows[-1]) + 1, "final cursor differs")
        cases.append(summarize(data, info, reference["candidate"]))
    result = aggregate(cases)
    require(sum(c["count"] for c in cases) == 3137, "incomplete roster")
    require(result == read(output / "summary.json"), "summary differs")
    return result


ENVIRONMENT = (
    "JAX_ENABLE_X64",
    "JAX_PLATFORMS",
    "JAX_COMPILATION_CACHE_DIR",
    "JAX_ENABLE_COMPILATION_CACHE",
    "XLA_FLAGS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
CHANGED = {
    "src/glassbox/_dynamics.py",
    "src/glassbox/online.py",
    "src/glassbox/learner.py",
}
INTERVALS = []
START = None
LABEL = None


def finish_interval():
    global START
    if START is not None:
        INTERVALS.append(dict(label=LABEL, start_ns=START, end_ns=time.monotonic_ns()))
        START = None


def pause(label):
    global START, LABEL
    finish_interval()
    print(json.dumps(dict(event="ready", label=label)), flush=True)
    require(sys.stdin.readline().strip() == "go", "worker grant missing")
    START, LABEL = time.monotonic_ns(), label


def schedule(baseline, candidate, cases):
    labels = ["startup"] + [
        f"{case['id']}:{start}"
        for case in cases
        for start in range(0, case["count"], 25)
    ]
    all_intervals = []
    for arm, intervals in (("baseline", baseline), ("candidate", candidate)):
        require([x["label"] for x in intervals] == labels, "block roster differs")
        require(all(x["end_ns"] > x["start_ns"] for x in intervals), "invalid interval")
        all_intervals.extend(dict(x, arm=arm) for x in intervals)
    ordered = sorted(all_intervals, key=lambda x: x["start_ns"])
    require(
        all(a["end_ns"] <= b["start_ns"] for a, b in pairwise(ordered)),
        "workers overlap",
    )
    require(
        [(x["arm"], x["label"]) for x in ordered]
        == [(arm, label) for label in labels for arm in ("baseline", "candidate")],
        "grant order differs",
    )
    return dict(blocks_per_arm=len(labels), overlapping_intervals=0)


def verify_pair(output, authority):
    authenticate(output, authority)
    attempts = read(output / "workers.json")
    workers = {
        arm: verify_worker(output / arm, attempts[arm]["manifest_sha256"])
        for arm in ("baseline", "candidate")
    }
    bounds = {arm: read(output / arm / "binding.json") for arm in workers}
    require(
        bounds["baseline"]["runtime"] == bounds["candidate"]["runtime"],
        "runtime differs",
    )
    require(
        bounds["baseline"]["environment"] == bounds["candidate"]["environment"],
        "environment differs",
    )
    for name in (
        "scripts/screen_accumulator.py",
        "scripts/collect_throw.py",
        "scripts/evaluate_online.py",
        "scripts/run_dart.py",
        "scripts/verify_baseline.py",
        "scripts/verify_online_v8.py",
    ):
        require(
            bounds["baseline"]["source"]["files"][name]
            == bounds["candidate"]["source"]["files"][name],
            "shared evaluator differs",
        )
    hashes = read(ROOT / read(PROTOCOL)["parent"]["package_hashes_from"])["parent"][
        "package_files"
    ]
    for name, wanted in hashes.items():
        require(
            bounds["baseline"]["source"]["files"][name] == wanted,
            "baseline package changed",
        )
        if name not in CHANGED:
            require(
                bounds["candidate"]["source"]["files"][name] == wanted,
                "extra candidate package change",
            )
    verified_schedule = schedule(
        read(output / "baseline/intervals.json"),
        read(output / "candidate/intervals.json"),
        workers["baseline"]["cases"],
    )
    candidate = workers["candidate"]
    timing = []
    for a, b in zip(candidate["cases"], workers["baseline"]["cases"], strict=True):
        require(a["id"] == b["id"], "paired case order differs")
        timing.append(
            dict(
                id=a["id"],
                family=a["family"],
                median_ratio=a["update"]["warm_median_s"]
                / b["update"]["warm_median_s"],
                p95_ratio=a["update"]["warm_p95_s"] / b["update"]["warm_p95_s"],
            )
        )
    families = {
        f: {
            key: geometric([x[key] for x in timing if x["family"] == f])
            for key in ("median_ratio", "p95_ratio")
        }
        for f in ("quad", "fixedwing")
    }
    speed = dict(
        quad_median=families["quad"]["median_ratio"] <= 0.75,
        quad_p95=families["quad"]["p95_ratio"] <= 0.75,
        fixedwing_p95=families["fixedwing"]["p95_ratio"] <= 1.10,
    )
    return dict(
        accuracy=candidate,
        baseline=workers["baseline"],
        timing=timing,
        timing_families=families,
        speed_gates=speed,
        schedule=verified_schedule,
        qualified=candidate["screen_passed"] and all(speed.values()),
        adopted=False,
    )


def pair(output, baseline_root):
    output.mkdir(parents=True, exist_ok=False)
    processes, logs, results = {}, [], {}
    try:
        for arm, root in (("baseline", baseline_root), ("candidate", ROOT)):
            log = (output / (arm + ".log")).open("x")
            logs.append(log)
            env = dict(
                os.environ, PYTHONPATH="src:scripts", PYTHONDONTWRITEBYTECODE="1"
            )
            processes[arm] = subprocess.Popen(
                [
                    sys.executable,
                    str(root / "scripts/screen_accumulator.py"),
                    "worker",
                    "--arm",
                    arm,
                    "--output",
                    str(output / arm),
                ],
                cwd=root,
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=log,
                text=True,
                bufsize=1,
            )
        pending = {arm: json.loads(p.stdout.readline()) for arm, p in processes.items()}
        require(
            all(
                p["event"] == "ready" and p["label"] == "startup"
                for p in pending.values()
            ),
            "workers not ready",
        )
        while pending:
            for arm, p in processes.items():
                if arm not in pending:
                    continue
                p.stdin.write("go\n")
                p.stdin.flush()
                line = p.stdout.readline()
                require(bool(line), arm + " worker exited before sealing; see log")
                event = json.loads(line)
                if event["event"] == "done":
                    require(p.wait() == 0, arm + " worker failed")
                    results[arm] = event
                    del pending[arm]
                else:
                    require(event["event"] == "ready", "unexpected worker event")
                    pending[arm] = event
                    if arm == "candidate" and event["label"].endswith(":0"):
                        print(json.dumps(dict(next_block=event["label"])), flush=True)
        write(output / "workers.json", results)
    except Exception as error:
        write(output / "failure.json", dict(error=repr(error)))
        raise
    finally:
        for p in processes.values():
            if p.poll() is None:
                p.terminate()
                p.wait()
        for log in logs:
            log.close()
        authority = seal(output, "glassbox-accumulator-pair-v1")
        print(
            json.dumps(dict(output=str(output), manifest_sha256=authority)), flush=True
        )
    result = verify_pair(output, authority)
    # Keep the completed sealed attempt immutable; publish reduction next to it.
    write(output.parent / (output.name + "-verified.json"), result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("worker", "pair", "verify"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=("baseline", "candidate"))
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--manifest-sha256")
    args = parser.parse_args()
    if args.mode == "worker":
        run(args.output, args.arm)
    else:
        result = (
            pair(args.output, args.baseline_root)
            if args.mode == "pair"
            else verify_pair(args.output, args.manifest_sha256)
        )
        print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
