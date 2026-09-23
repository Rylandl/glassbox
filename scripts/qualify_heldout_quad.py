"""Score fresh Crazyflow configurations with the public episode-only learner."""

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
from collect_throw import authenticate, seal
from verify_baseline import observed, require

from glassbox import STATE_CHANNELS, OnlineFit, SequenceCollection, SequenceSegment

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/harness/heldout-quad-v1.json"
GIT = "/opt/homebrew/Caskroom/miniconda/base/bin/git"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def case_data(source, item):
    report = read(source / item["id"] / "report.json")
    with np.load(source / item["id"] / "stream.npz") as tape:
        raw, commands, times = (
            tape[key].copy() for key in ("states", "commands", "time_s")
        )
    require(raw.shape == (len(commands) + 1, 13), "source motion/command alignment")
    require(
        times.shape == (len(raw),) and np.isfinite(times).all(), "invalid source clock"
    )
    require(np.allclose(np.diff(times), 0.01, rtol=0, atol=1e-6), "source sample grid")
    states = observed(raw)
    require(
        np.isfinite(states).all() and np.isfinite(commands).all(), "nonfinite source"
    )
    return report, states, commands


def rmse(predicted, actual, start, end):
    error = predicted[..., start:end] - actual[..., start:end]
    return float(np.sqrt(np.mean(np.sum(error**2, axis=-1))))


def score(states, commands, origins, forecast, one):
    actual = np.stack([states[row + 1 : row + 26] for row in origins])
    require(forecast.shape == actual.shape == (len(origins), 25, 15), "forecast shape")
    require(one.shape == (origins[-1] - origins[0], 15), "one-step shape")
    require(
        np.isfinite(forecast).all() and np.isfinite(one).all(), "nonfinite forecast"
    )
    output = {"origins": len(origins), "horizons": {}, "one_step": {}}
    for ms in (50, 100, 150, 200, 250):
        step = ms // 10 - 1
        output["horizons"][str(ms)] = {
            "rate_rad_s": rmse(forecast[:, step], actual[:, step], 3, 6),
            "velocity_m_s": rmse(forecast[:, step], actual[:, step], 0, 3),
            "hold_rate_rad_s": rmse(states[origins], actual[:, step], 3, 6),
            "hold_velocity_m_s": rmse(states[origins], actual[:, step], 0, 3),
        }
    truth_one = states[origins[0] + 1 : origins[-1] + 1]
    output["one_step"] = {
        "rate_rad_s": rmse(one, truth_one, 3, 6),
        "velocity_m_s": rmse(one, truth_one, 0, 3),
        "hold_rate_rad_s": rmse(states[origins[0] : origins[-1]], truth_one, 3, 6),
        "hold_velocity_m_s": rmse(states[origins[0] : origins[-1]], truth_one, 0, 3),
    }
    output["per_origin_250"] = [
        {
            "row": int(row),
            "rate_rad_s": float(
                np.linalg.norm(forecast[index, -1, 3:6] - actual[index, -1, 3:6])
            ),
            "velocity_m_s": float(
                np.linalg.norm(forecast[index, -1, :3] - actual[index, -1, :3])
            ),
        }
        for index, row in enumerate(origins)
    ]
    return output


def schedule(commands, spec):
    measure = spec["measurement"]
    first = measure["first_prediction_row"]
    last = len(commands) - max(measure["horizon_ms"]) // 10
    return np.arange(first, last + 1, measure["forecast_stride_rows"], dtype=int)


def run_case(source, item, spec, output):
    report, states, commands = case_data(source, item)
    origins = schedule(commands, spec)
    if len(origins) < 2:
        return dict(
            source_status=report["status"],
            intervals=len(commands),
            qualification="insufficient recorded horizon",
        )
    first = int(origins[0])
    begin = spec["measurement"]["prefix_begin_row"]
    prefix = SequenceCollection(
        (
            SequenceSegment(
                item["id"],
                "prefix",
                states[begin : first + 1],
                commands[begin:first],
                0.01,
                begin,
            ),
        ),
        item["id"],
        STATE_CHANNELS,
        tuple(f"motor_{index} [1]" for index in range(4)),
    )
    started = time.perf_counter()
    session = OnlineFit(prefix)
    initialization_s = time.perf_counter() - started
    forecasts, one, update_s = [], [], []
    first_forecast_s = None
    chosen = set(origins.tolist())
    for row in range(first, int(origins[-1]) + 1):
        h = session.model.history_steps
        past, prior = states[row - h : row + 1], commands[row - h : row]
        if row in chosen:
            started = time.perf_counter()
            predicted = np.asarray(
                session.predict(past, prior, commands[row : row + 25])
            )
            if first_forecast_s is None:
                first_forecast_s = time.perf_counter() - started
            forecasts.append(predicted)
        if row == origins[-1]:
            break
        one.append(np.asarray(session.predict(past, prior, commands[row : row + 1]))[0])
        started = time.perf_counter()
        session.observe(row, commands[row], states[row + 1])
        update_s.append(time.perf_counter() - started)
    forecasts, one, update_s = map(np.asarray, (forecasts, one, update_s))
    require(len(update_s) > 1, "insufficient update timings")
    np.savez_compressed(
        output / f"{item['id']}.npz",
        origins=origins,
        forecast=forecasts,
        one=one,
        update_s=update_s,
    )
    return dict(
        source_status=report["status"],
        intervals=len(commands),
        qualification="scored",
        scores=score(states, commands, origins, forecasts, one),
        initialization_s=initialization_s,
        first_forecast_s=first_forecast_s,
        first_update_s=float(update_s[0]),
        warm_update_median_s=float(np.median(update_s[1:])),
    )


def verify(source, authority, output):
    authenticate(source, authority)
    require(
        (output / "benchmark.json").read_bytes() == PROTOCOL.read_bytes(),
        "protocol changed",
    )
    bound = read(output / "binding.json")
    require(
        bound["source_manifest_sha256"] == authority
        and Path(bound["source_path"]).resolve() == source.resolve(),
        "source binding differs",
    )
    spec = read(PROTOCOL)
    summary = read(output / "summary.json")
    require(
        set(summary) == {item["id"] for item in spec["streams"]["quad"]["cases"]},
        "case inventory differs",
    )
    for item in spec["streams"]["quad"]["cases"]:
        name = item["id"]
        report, states, commands = case_data(source, item)
        result = summary[name]
        require(
            result["source_status"] == report["status"]
            and result["intervals"] == len(commands),
            "source report differs",
        )
        origins = schedule(commands, spec)
        if len(origins) < 2:
            require(
                result["qualification"] == "insufficient recorded horizon",
                "missing failure",
            )
            continue
        with np.load(output / f"{name}.npz") as saved:
            actual_origins = saved["origins"].copy()
            forecast, one, update_s = (
                saved[key].copy() for key in ("forecast", "one", "update_s")
            )
        require(np.array_equal(actual_origins, origins), "origin schedule differs")
        require(
            result["scores"] == score(states, commands, origins, forecast, one),
            "saved scores differ",
        )
        require(
            result["first_update_s"] == float(update_s[0])
            and result["warm_update_median_s"] == float(np.median(update_s[1:])),
            "saved timing differs",
        )
    print("Verified held-out forecasts and physical errors without fitting.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "verify"))
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest-sha256")
    args = parser.parse_args()
    source = args.source.resolve()
    if args.mode == "verify":
        require(args.manifest_sha256, "result authority required")
        authenticate(args.output, args.manifest_sha256)
        verify(source, args.source_manifest_sha256, args.output)
        return
    authenticate(source, args.source_manifest_sha256)
    spec = read(PROTOCOL)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "benchmark.json").write_bytes(PROTOCOL.read_bytes())
    write(
        args.output / "binding.json",
        dict(
            source_path=str(source),
            source_manifest_sha256=args.source_manifest_sha256,
            source_commit=subprocess.check_output(
                [GIT, "rev-parse", "HEAD"], text=True
            ).strip(),
            learner_sha256=digest(ROOT / "src/glassbox/online.py"),
            runner_sha256=digest(__file__),
        ),
    )
    try:
        results = {
            item["id"]: run_case(source, item, spec, args.output)
            for item in spec["streams"]["quad"]["cases"]
        }
        write(args.output / "summary.json", results)
    finally:
        print("manifest_sha256", seal(args.output, spec["id"]))


if __name__ == "__main__":
    main()
