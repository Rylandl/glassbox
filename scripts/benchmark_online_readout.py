"""Run one reusable, causal two-origin online-readout benchmark."""

import argparse
import hashlib
import importlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
from collect_throw import authenticate, seal
from evaluate_online import metrics
from verify_baseline import arrays, observed, require

from glassbox import STATE_CHANNELS, SequenceCollection, SequenceSegment

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "docs/online-readout-benchmark.json"
GIT = "/opt/homebrew/Caskroom/miniconda/base/bin/git"
HORIZONS = (50, 100, 150, 200, 250)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_case(spec, name):
    roots = {key: Path(value["path"]) for key, value in spec["sources"].items()}
    direct_root = roots["direct"] / name
    previous = arrays(direct_root / "predictions.npz")
    result = read(direct_root / "result.json")
    origins = previous["conditional_rows"][:2].astype(int)
    begin, first, dt = (
        result["prefix_begin_row"],
        result["first_prediction_row"],
        result["dt_s"],
    )
    require(first == origins[0] and origins[1] > first, "control origin mismatch")
    if name.startswith("paired-quad-"):
        grid = name.removeprefix("paired-quad-")
        tape = arrays(roots["paired_recording"] / f"{grid}.npz")
        full = arrays(roots["paired_full"] / grid / "predictions.npz")
        full_rows = full["origin_rows"]
        full_forecasts = full["conditional_baseline"]
        identity = "paired-quad-v2"
        names = tuple(f"motor_{i} [1]" for i in range(4))
    else:
        base = roots["recordings"]
        tape = arrays(base / "inputs" / f"{name}.npz")
        info = read(base / name / "case.json")
        full = arrays(roots["full"] / name / "predictions.npz")
        full_rows = full["conditional_rows"]
        full_forecasts = full["baseline_conditional"]
        identity, names = info["opaque_id"], tuple(info["ordered_commands"])
    require(np.array_equal(full_rows[:2], origins), "full control origin mismatch")
    states, commands = observed(tape["states"]), tape["commands"].astype(float)
    horizon = round(0.25 / dt)
    require(origins[1] + horizon < len(states), "missing 250 ms truth")
    truth = np.stack(
        [states[row + 1 : row + horizon + 1] for row in origins]
    )
    return dict(
        name=name,
        identity=identity,
        names=names,
        states=states,
        commands=commands,
        begin=begin,
        origins=origins,
        dt=dt,
        horizon=horizon,
        truth=truth,
        direct=previous["conditional"][:2],
        direct_one=previous["one"][: origins[1] - first],
        full=full_forecasts[:2],
    )


def scores(case, forecast, one):
    one_step = None
    if one is not None:
        actual_one = case["states"][case["origins"][0] + 1 : case["origins"][1] + 1]
        one_step = metrics(one, actual_one)
    result = {"one_step": one_step, "origins": {}}
    for index in range(2):
        result["origins"][str(index)] = {}
        for ms in HORIZONS:
            step = round(ms / 1000 / case["dt"]) - 1
            result["origins"][str(index)][str(ms)] = metrics(
                forecast[index, step : step + 1],
                case["truth"][index, step : step + 1],
            )
    return result


def run_case(case, factory, output):
    first, last = case["origins"]
    prefix = SequenceCollection(
        (
            SequenceSegment(
                case["identity"],
                "prefix",
                case["states"][case["begin"] : first + 1],
                case["commands"][case["begin"] : first],
                case["dt"],
                case["begin"],
            ),
        ),
        case["identity"],
        STATE_CHANNELS,
        case["names"],
    )
    started = time.perf_counter()
    fit = factory(prefix)
    initialization_s = time.perf_counter() - started
    predictions, one, update_s = [], [], []
    for row in range(first, last + 1):
        model = fit.session.model
        h = model.history_steps
        past = case["states"][row - h : row + 1]
        inputs = case["commands"][row - h : row]
        if row in (first, last):
            forecast = np.asarray(
                fit.session.predict(
                    past,
                    inputs,
                    case["commands"][row : row + case["horizon"]],
                )
            )
            require(
                forecast.shape == (case["horizon"], 15)
                and np.isfinite(forecast).all(),
                "invalid conditional forecast",
            )
            predictions.append(forecast)
        if row == last:
            break
        step = np.asarray(
            fit.session.predict(past, inputs, case["commands"][row : row + 1])
        )[0]
        require(step.shape == (15,) and np.isfinite(step).all(), "invalid one-step")
        one.append(step)
        started = time.perf_counter()
        fit.observe(row, case["commands"][row], case["states"][row + 1])
        update_s.append(time.perf_counter() - started)
    forecast, one, update_s = map(np.asarray, (predictions, one, update_s))
    np.savez_compressed(
        output / f"{case['name']}.npz",
        origins=case["origins"],
        forecast=forecast,
        one=one,
        update_s=update_s,
    )
    return dict(
        candidate=scores(case, forecast, one),
        direct=scores(case, case["direct"], case["direct_one"]),
        full=scores(case, case["full"], None),
        initialization_s=initialization_s,
        warm_update_median_s=float(np.median(update_s[1:])),
        updates=len(one),
    )


def verify_case(case, output, result):
    saved = arrays(output / f"{case['name']}.npz")
    require(np.array_equal(saved["origins"], case["origins"]), "origin mismatch")
    require(saved["forecast"].shape == (2, case["horizon"], 15), "forecast shape")
    require(
        saved["one"].shape == (case["origins"][1] - case["origins"][0], 15),
        "one-step shape",
    )
    require(result["candidate"] == scores(case, saved["forecast"], saved["one"]),
            "candidate score mismatch")
    require(result["direct"] == scores(case, case["direct"], case["direct_one"]),
            "direct score mismatch")
    require(result["full"] == scores(case, case["full"], None),
            "full score mismatch")
    require(result["updates"] == len(saved["update_s"]), "update count mismatch")
    require(
        result["warm_update_median_s"] == float(np.median(saved["update_s"][1:])),
        "timing mismatch",
    )


def table(cases, results):
    lines = [
        "| Case | First 250 ms rate: candidate / direct / full (rad/s) | First velocity: candidate / direct / full (m/s) | Next 250 ms rate: candidate / direct / full (rad/s) | Warm update (ms) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name in cases:
        result = results[name]
        candidate = result["candidate"]["origins"]
        direct = result["direct"]["origins"]
        full = result["full"]["origins"]
        def value(arm, origin, field):
            return arm[str(origin)]["250"][field]
        lines.append(
            f"| {name} | "
            f"{value(candidate, 0, 'body_rate_rmse_rad_s'):.3f} / {value(direct, 0, 'body_rate_rmse_rad_s'):.3f} / {value(full, 0, 'body_rate_rmse_rad_s'):.3f} | "
            f"{value(candidate, 0, 'velocity_rmse_m_s'):.3f} / {value(direct, 0, 'velocity_rmse_m_s'):.3f} / {value(full, 0, 'velocity_rmse_m_s'):.3f} | "
            f"{value(candidate, 1, 'body_rate_rmse_rad_s'):.3f} / {value(direct, 1, 'body_rate_rmse_rad_s'):.3f} / {value(full, 1, 'body_rate_rmse_rad_s'):.3f} | "
            f"{result['warm_update_median_s'] * 1000:.2f} |"
        )
    return "\n".join(lines) + "\n"


def authenticate_sources(spec):
    for source in spec["sources"].values():
        authenticate(Path(source["path"]), source["manifest_sha256"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "verify"))
    parser.add_argument("--candidate", help="module:factory with session.predict and observe")
    parser.add_argument("--suite", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-sha256")
    args = parser.parse_args()
    if args.mode == "verify":
        authenticate(args.output, args.manifest_sha256)
        spec = read(args.output / "benchmark.json")
        authenticate_sources(spec)
        summary = read(args.output / "summary.json")
        for name in summary["cases"]:
            verify_case(source_case(spec, name), args.output, summary["results"][name])
        require((args.output / "table.md").read_text() == table(summary["cases"], summary["results"]), "table mismatch")
        print("Verified saved forecasts, controls, scores and table without fitting.")
        return
    require(args.candidate and ":" in args.candidate, "supply module:factory")
    spec = read(SPEC)
    authenticate_sources(spec)
    module_name, factory_name = args.candidate.split(":", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, factory_name)
    source_path = Path(module.__file__).resolve()
    cases = spec[f"{args.suite}_cases"]
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "benchmark.json").write_bytes(SPEC.read_bytes())
    write(args.output / "binding.json", dict(
        candidate=args.candidate,
        candidate_file=str(source_path),
        candidate_sha256=sha(source_path),
        benchmark_sha256=sha(SPEC),
        runner_sha256=sha(__file__),
        git_head=subprocess.check_output([GIT, "rev-parse", "HEAD"], text=True).strip(),
    ))
    try:
        results = {name: run_case(source_case(spec, name), factory, args.output)
                   for name in cases}
        summary = dict(suite=args.suite, cases=cases, results=results)
        write(args.output / "summary.json", summary)
        (args.output / "table.md").write_text(table(cases, results))
        for name in cases:
            verify_case(source_case(spec, name), args.output, results[name])
    except BaseException as error:
        write(args.output / "failure.json", dict(error=repr(error)))
        raise
    finally:
        print("manifest_sha256", seal(args.output, "glassbox-online-readout-benchmark-v1"))
    print((args.output / "table.md").read_text())


if __name__ == "__main__":
    main()
