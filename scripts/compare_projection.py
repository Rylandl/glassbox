"""Focused saved-model equivalence and paired whole-update timing."""

import argparse
import json
import os
import subprocess
import sys
import time
from itertools import pairwise
from pathlib import Path

import numpy as np
from collect_throw import authenticate, binding, check_source, checkpoint, seal
from evaluate_online import geometric
from run_dart import ROOT, write
from screen_accumulator import INTERVALS, finish_interval, pause
from verify_baseline import arrays, digest, observed, read, require

PROTOCOL = ROOT / "docs/harness/history-projection-reuse-v1.json"


def capture(output):
    from glassbox import OnlineFit

    spec, bound = read(PROTOCOL), binding(PROTOCOL)
    require(
        {k: digest(ROOT / k) for k in spec["package_files"]} == spec["package_files"],
        "capture needs unchanged baseline",
    )
    output.mkdir(parents=True, exist_ok=False)
    write(output / "binding.json", bound)
    points = []
    try:
        for key in ("parent", "tapes", "fits"):
            authenticate(Path(spec[key]["path"]), spec[key]["manifest_sha256"])
        parent = Path(spec["parent"]["path"]) / "candidate"
        tapes = Path(spec["tapes"]["path"]) / "inputs"
        for case in spec["cases"]:
            original = parent / case
            session = OnlineFit.load(original / "initial-online.npz")
            events = [
                json.loads(x)
                for x in (original / "events.jsonl").read_text().splitlines()
            ]
            tape = arrays(tapes / f"{case}.npz")
            info = read(original / "case.json")
            for offset in range(max(spec["capture_offsets"]) + 1):
                row = session.cursor
                if offset in spec["capture_offsets"]:
                    name = f"{case}-{offset}"
                    folder = output / name
                    folder.mkdir()
                    session.save(folder / "session.npz")
                    h = session.model.history_steps
                    checkpoint(
                        folder / "context.npz",
                        row=np.asarray(row),
                        past=observed(tape["states"][row - h : row + 1]),
                        issued=tape["commands"][row - h : row],
                        command=tape["commands"][row],
                        truth=observed(tape["states"][row + 1 : row + 2])[0],
                    )
                    write(folder / "expected.json", events[offset])
                    points.append(dict(id=name, family=info["family"], offset=offset))
                if offset < max(spec["capture_offsets"]):
                    session.observe(
                        row,
                        tape["commands"][row],
                        observed(tape["states"][row + 1 : row + 2])[0],
                    )
                    require(
                        session.model.fingerprint == events[offset]["after"]
                        and session.report == events[offset]["report"],
                        "capture replay differs",
                    )
            print(json.dumps(dict(captured=case)), flush=True)
        check_source(bound)
        write(output / "points.json", points)
    except Exception as error:
        write(output / "failure.json", dict(error=repr(error)))
        raise
    finally:
        print(
            json.dumps(
                dict(manifest_sha256=seal(output, "glassbox-projection-captures-v1"))
            ),
            flush=True,
        )


def offline(output, spec):
    import jax
    import jax.numpy as jnp
    from jax.flatten_util import ravel_pytree

    from glassbox import LearnedDynamics
    from glassbox._dynamics import _rollout

    rows = []
    for name in ("dart", "crazyflow", "cascade"):
        source = Path(spec["fits"]["path"]) / "candidate" / name / "model.npz"
        revision = LearnedDynamics.load(source)
        raw = arrays(source)
        core = revision._model
        for enabled in (False, True):
            with jax.enable_x64(enabled):
                dtype = jnp.float64 if enabled else jnp.float32
                par, norms = jax.tree.map(
                    lambda x, dtype=dtype: jnp.asarray(x, dtype),
                    (core.params, core.norms),
                )
                past, issued, future = [
                    jnp.asarray(raw["development_" + k][:8], dtype)
                    for k in ("past_states", "past_inputs", "future_inputs")
                ]
                theta, unpack = ravel_pytree((par, future))

                def prediction(
                    value,
                    unpack=unpack,
                    norms=norms,
                    past=past,
                    issued=issued,
                    core=core,
                ):
                    params, commands = unpack(value)
                    return _rollout(
                        params,
                        norms,
                        past,
                        issued,
                        commands,
                        core.delay_steps,
                        core.dt_s,
                    )

                fn = jax.jit(prediction)
                rng = np.random.default_rng(917)
                direction = jnp.asarray(rng.normal(size=theta.shape), dtype)
                direction /= jnp.linalg.norm(direction)
                primal, tangent = jax.jvp(fn, (theta,), (direction,))
                cotangent = jnp.asarray(rng.normal(size=primal.shape), dtype)
                cotangent /= jnp.linalg.norm(cotangent)
                reverse = jax.vjp(fn, theta)[1](cotangent)[0]
                label = f"{name}-{'float64' if enabled else 'float32'}"
                values = dict(
                    primal=np.asarray(primal),
                    jvp=np.asarray(tangent),
                    vjp=np.asarray(reverse),
                )
                require(
                    all(np.isfinite(v).all() for v in values.values()),
                    "nonfinite offline value",
                )
                checkpoint(output / f"{label}.npz", **values)
                rows.append(
                    dict(
                        id=label, model=revision.fingerprint(), dtype=str(primal.dtype)
                    )
                )
    write(output / "offline.json", rows)


def updated(path, context):
    import jax

    from glassbox import OnlineFit

    session = OnlineFit.load(path)
    start = time.perf_counter()
    session.observe(int(context["row"]), context["command"], context["truth"])
    model = session.model
    jax.block_until_ready(model.params)
    elapsed = time.perf_counter() - start
    return session, model, elapsed


def worker(output, arm, captures, authority):
    pause("startup")
    output.mkdir(parents=True, exist_ok=False)
    spec, bound = read(PROTOCOL), binding(PROTOCOL)
    write(output / "binding.json", bound)
    write(output / "protocol.json", spec)
    authenticate(captures, authority)
    authenticate(Path(spec["fits"]["path"]), spec["fits"]["manifest_sha256"])
    write(output / "captures.json", dict(path=str(captures), authority=authority))
    try:
        offline(output, spec)
        from glassbox import OnlineFit

        for point in read(captures / "points.json"):
            name = point["id"]
            folder = captures / name
            context = arrays(folder / "context.npz")
            path = folder / "session.npz"
            before = OnlineFit.load(path)
            prediction = np.asarray(
                before.predict(
                    context["past"], context["issued"], context["command"][None]
                )
            )
            pause(name + ":warmup")
            warmups = []
            identity = None
            for _ in range(1 + spec["timing"]["warmups"]):
                session, model, elapsed = updated(path, context)
                warmups.append(elapsed)
                if identity is None:
                    identity = model.fingerprint
                    report = session.report
                    expected = read(folder / "expected.json")
                    if arm == "baseline":
                        require(
                            identity == expected["after"]
                            and report == expected["report"],
                            "baseline update differs",
                        )
                    post = np.asarray(
                        session.predict(
                            context["past"], context["issued"], context["command"][None]
                        )
                    )
                    checkpoint(
                        output / f"{name}.npz",
                        prediction=prediction,
                        post_prediction=post,
                        **model.arrays(),
                    )
                require(
                    model.fingerprint == identity and session.report == report,
                    "repeated update differs",
                )
            timings = []
            for block in range(spec["timing"]["blocks"]):
                pause(f"{name}:{block}")
                for _ in range(spec["timing"]["repeats_per_block"]):
                    session, model, elapsed = updated(path, context)
                    timings.append(elapsed)
                    require(
                        model.fingerprint == identity and session.report == report,
                        "timed replay differs",
                    )
            write(
                output / f"{name}.json",
                dict(
                    point,
                    report=report,
                    warmup_s=warmups,
                    update_s=timings,
                    fingerprint=identity,
                ),
            )
        check_source(bound)
    except Exception as error:
        write(output / "failure.json", dict(error=repr(error)))
        raise
    finally:
        finish_interval()
        write(output / "intervals.json", INTERVALS)
        authority = seal(output, "glassbox-projection-worker-v1")
        print(json.dumps(dict(event="done", manifest_sha256=authority)), flush=True)


def discrepancy(actual, reference, tolerance):
    require(
        actual.shape == reference.shape and actual.dtype == reference.dtype,
        "array contract differs",
    )
    require(
        np.isfinite(actual).all() and np.isfinite(reference).all(),
        "nonfinite comparison",
    )
    error = abs(actual - reference)
    limit = tolerance["atol"] + tolerance["rtol"] * abs(reference)
    return dict(
        passed=bool(np.all(error <= limit)),
        max_abs=float(error.max()),
        max_scaled=float((error / limit).max()),
    )


def verify(output, authority):
    authenticate(output, authority)
    worker_hashes = read(output / "workers.json")
    folders = {arm: output / arm for arm in ("baseline", "candidate")}
    baseline, candidate = folders.values()
    spec = read(PROTOCOL)
    for arm, folder in folders.items():
        authenticate(folder, worker_hashes[arm]["manifest_sha256"])
        require(read(folder / "protocol.json") == spec, "protocol differs")
        source = read(folder / "binding.json")["source"]["files"]
        for name, expected in spec["package_files"].items():
            if arm == "baseline" or name != "src/glassbox/_dynamics.py":
                require(source[name] == expected, "unexpected package change: " + name)
    require(
        read(baseline / "binding.json")["runtime"]
        == read(candidate / "binding.json")["runtime"],
        "runtime differs",
    )
    capture_spec = read(baseline / "captures.json")
    require(capture_spec == read(candidate / "captures.json"), "capture inputs differ")
    captures = Path(capture_spec["path"])
    authenticate(captures, capture_spec["authority"])
    points = read(captures / "points.json")
    require(
        [x["id"] for x in points]
        == [f"{c}-{o}" for c in spec["cases"] for o in spec["capture_offsets"]],
        "point roster differs",
    )
    off = read(baseline / "offline.json")
    require(
        [r["id"] for r in off]
        == [
            f"{name}-{dtype}"
            for name in ("dart", "crazyflow", "cascade")
            for dtype in ("float32", "float64")
        ],
        "offline query roster differs",
    )
    require(off == read(candidate / "offline.json"), "offline identities differ")
    comparisons = {}
    for row in off:
        name = row["id"]
        a, b = arrays(candidate / f"{name}.npz"), arrays(baseline / f"{name}.npz")
        comparisons[name] = {
            key: discrepancy(
                a[key],
                b[key],
                spec["tolerance"][
                    row["dtype"] + ("_prediction" if key == "primal" else "_derivative")
                ],
            )
            for key in b
        }
    rows = []
    for point in points:
        name = point["id"]
        a, b = arrays(candidate / f"{name}.npz"), arrays(baseline / f"{name}.npz")
        comparison = {}
        require(set(a) == set(b), "post-update array roster differs")
        for key in b:
            tolerance = spec["tolerance"][
                "float32_prediction"
                if key.endswith("prediction")
                else "post_parameters"
            ]
            comparison[key] = discrepancy(a[key], b[key], tolerance)
        records = {
            arm: read(folder / f"{name}.json") for arm, folder in folders.items()
        }
        ratios = {}
        for arm, record in records.items():
            times = np.asarray(record["update_s"])
            require(
                len(times)
                == spec["timing"]["blocks"] * spec["timing"]["repeats_per_block"]
                and np.isfinite(times).all()
                and np.all(times > 0),
                "timing roster differs",
            )
            ratios[arm] = dict(
                median_s=float(np.median(times)), p95_s=float(np.quantile(times, 0.95))
            )
        decisions = {
            arm: dict(
                accepted_proposals=record["report"]["accepted_proposals"],
                objective_calls=record["report"]["objective_calls"],
                selected_alpha=record["report"]["last_proposal"]["selected_alpha"],
                damping=record["report"]["damping"],
            )
            for arm, record in records.items()
        }
        rows.append(
            dict(
                point,
                comparison=comparison,
                timing=ratios,
                decisions=decisions,
                same_decision=decisions["candidate"] == decisions["baseline"],
                median_ratio=ratios["candidate"]["median_s"]
                / ratios["baseline"]["median_s"],
                p95_ratio=ratios["candidate"]["p95_s"] / ratios["baseline"]["p95_s"],
            )
        )
    labels = ["startup"] + [
        label
        for point in points
        for label in [point["id"] + ":warmup"]
        + [f"{point['id']}:{b}" for b in range(spec["timing"]["blocks"])]
    ]
    intervals = []
    for arm, folder in folders.items():
        items = read(folder / "intervals.json")
        require([x["label"] for x in items] == labels, "interval roster differs")
        intervals.extend(dict(x, arm=arm) for x in items)
    intervals.sort(key=lambda x: x["start_ns"])
    require(all(x["end_ns"] > x["start_ns"] for x in intervals), "invalid interval")
    require(
        all(a["end_ns"] <= b["start_ns"] for a, b in pairwise(intervals)),
        "workers overlap",
    )
    require(
        [(x["label"], x["arm"]) for x in intervals]
        == [(label, arm) for label in labels for arm in ("baseline", "candidate")],
        "grant order differs",
    )
    families = {
        family: {
            key: geometric([r[key] for r in rows if r["family"] == family])
            for key in ("median_ratio", "p95_ratio")
        }
        for family in sorted({r["family"] for r in rows})
    }
    median = geometric([v["median_ratio"] for v in families.values()])
    return dict(
        offline=comparisons,
        points=rows,
        families=families,
        equal_family_median_ratio=median,
        timing_target_met=median <= 0.95
        and all(v["median_ratio"] <= 1.05 for v in families.values()),
        numerical_checks_pass=all(
            x["passed"] for group in comparisons.values() for x in group.values()
        )
        and all(x["passed"] for row in rows for x in row["comparison"].values()),
        same_decisions=all(r["same_decision"] for r in rows),
        exclusive_blocks_per_arm=len(labels),
    )


def pair(output, baseline_root, captures, authority):
    output.mkdir(parents=True, exist_ok=False)
    processes, logs, results = {}, [], {}
    try:
        for arm, root in (("baseline", baseline_root), ("candidate", ROOT)):
            log = (output / (arm + ".log")).open("x")
            logs.append(log)
            processes[arm] = subprocess.Popen(
                [
                    sys.executable,
                    str(root / "scripts/compare_projection.py"),
                    "worker",
                    "--arm",
                    arm,
                    "--output",
                    str(output / arm),
                    "--captures",
                    str(captures),
                    "--manifest-sha256",
                    authority,
                ],
                cwd=root,
                env=dict(
                    os.environ, PYTHONPATH="src:scripts", PYTHONDONTWRITEBYTECODE="1"
                ),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=log,
                text=True,
                bufsize=1,
            )
        pending = {arm: json.loads(p.stdout.readline()) for arm, p in processes.items()}
        require(
            all(e == dict(event="ready", label="startup") for e in pending.values()),
            "workers not ready",
        )
        while pending:
            for arm, p in processes.items():
                if arm not in pending:
                    continue
                p.stdin.write("go\n")
                p.stdin.flush()
                line = p.stdout.readline()
                require(bool(line), arm + " worker stopped; see log")
                event = json.loads(line)
                if event["event"] == "done":
                    require(p.wait() == 0, arm + " worker failed")
                    results[arm] = event
                    del pending[arm]
                else:
                    require(event["event"] == "ready", "unexpected worker event")
                    pending[arm] = event
                    if arm == "candidate" and event["label"].endswith(":warmup"):
                        print(json.dumps(dict(point=event["label"])), flush=True)
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
        result_authority = seal(output, "glassbox-projection-pair-v1")
        print(json.dumps(dict(manifest_sha256=result_authority)), flush=True)
    result = verify(output, result_authority)
    write(output.parent / (output.name + "-verified.json"), result)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("offline", "points")},
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("capture", "pair", "worker", "verify"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm")
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--captures", type=Path)
    parser.add_argument("--manifest-sha256")
    args = parser.parse_args()
    if args.mode == "capture":
        capture(args.output)
    elif args.mode == "pair":
        pair(args.output, args.baseline_root, args.captures, args.manifest_sha256)
    elif args.mode == "worker":
        worker(args.output, args.arm, args.captures, args.manifest_sha256)
    else:
        print(json.dumps(verify(args.output, args.manifest_sha256), indent=2))


if __name__ == "__main__":
    main()
