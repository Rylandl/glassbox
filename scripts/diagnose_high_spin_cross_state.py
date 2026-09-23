"""Score models learned on different excitation prefixes at one common state."""

import argparse
import json
from pathlib import Path

import numpy as np
from collect_throw import authenticate, seal
from verify_baseline import observed, require

from glassbox import STATE_CHANNELS, OnlineFit, SequenceCollection, SequenceSegment

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/harness/high-spin-cross-state-v1.json"


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def arrays(path):
    with np.load(path) as pack:
        return {key: pack[key].copy() for key in pack.files}


def fit_and_predict(training, target, row, begin, delta):
    trained = observed(training["states"])
    commands = training["commands"]
    prefix = SequenceCollection(
        (SequenceSegment("excitation-diagnosis", "prefix", trained[begin : row + 1], commands[begin:row], .01, begin),),
        "excitation-diagnosis", STATE_CHANNELS, tuple(f"motor_{i} [1]" for i in range(4)),
    )
    session = OnlineFit(prefix)
    h = session.model.history_steps
    target_states = observed(target["states"])
    target_commands = target["commands"]
    past, prior = target_states[row - h : row + 1], target_commands[row - h : row]
    future = target_commands[row : row + 25]
    forecast = np.asarray(session.predict(past, prior, future))
    columns = []
    for channel in range(4):
        low, high = future.copy(), future.copy()
        low[0, channel] = np.clip(low[0, channel] - delta, 0, 1)
        high[0, channel] = np.clip(high[0, channel] + delta, 0, 1)
        minus = np.asarray(session.predict(past, prior, low))[:, 3:6]
        plus = np.asarray(session.predict(past, prior, high))[:, 3:6]
        columns.append((plus - minus) / (high[0, channel] - low[0, channel]))
    return forecast, np.stack(columns, axis=-1)


def scores(target, forecast, response, row):
    actual = observed(target["states"])[row + 1 : row + 26]
    true_response = target["response_truth"]
    output = {}
    for ms in (50, 100, 250):
        step = ms // 10 - 1
        output[str(ms)] = dict(
            rate_error_rad_s=float(np.linalg.norm(forecast[step, 3:6] - actual[step, 3:6])),
            response_relative_error=float(np.linalg.norm(response[step] - true_response[step]) / np.linalg.norm(true_response[step])),
        )
    return output


def run(output):
    spec = read(PROTOCOL)
    for source in spec["sources"].values():
        authenticate(Path(source["path"]), source["manifest_sha256"])
    measure = spec["measurement"]
    row, begin, delta = (measure[key] for key in ("target_row", "prefix_begin_row", "probe_delta"))
    output.mkdir(parents=True, exist_ok=False)
    (output / "protocol.json").write_bytes(PROTOCOL.read_bytes())
    summary = {}
    for case in spec["cases"]:
        target = arrays(Path(spec["sources"]["fast"]["path"]) / f"{case}-control.npz")
        for kind, source in spec["sources"].items():
            for branch in ("control", "orthogonal"):
                training = arrays(Path(source["path"]) / f"{case}-{branch}.npz")
                forecast, response = fit_and_predict(training, target, row, begin, delta)
                name = f"{case}-{kind}-{branch}"
                np.savez_compressed(output / f"{name}.npz", forecast=forecast, response=response)
                summary[name] = scores(target, forecast, response, row)
                print(name, json.dumps(summary[name], sort_keys=True), flush=True)
    write(output / "summary.json", summary)
    print("manifest_sha256", seal(output, spec["id"]), flush=True)


def verify(output, expected):
    authenticate(output, expected)
    spec = read(PROTOCOL)
    for source in spec["sources"].values():
        authenticate(Path(source["path"]), source["manifest_sha256"])
    require((output / "protocol.json").read_bytes() == PROTOCOL.read_bytes(), "protocol differs")
    summary = read(output / "summary.json")
    names = {f"{case}-{kind}-{branch}" for case in spec["cases"] for kind in spec["sources"] for branch in ("control", "orthogonal")}
    require(set(summary) == names, "result roster differs")
    row = spec["measurement"]["target_row"]
    for case in spec["cases"]:
        target = arrays(Path(spec["sources"]["fast"]["path"]) / f"{case}-control.npz")
        for kind in spec["sources"]:
            for branch in ("control", "orthogonal"):
                name = f"{case}-{kind}-{branch}"
                saved = arrays(output / f"{name}.npz")
                forecast, response = saved["forecast"], saved["response"]
                require(forecast.shape == (25, 15) and response.shape == (25, 3, 4), "prediction shape differs")
                require(np.isfinite(forecast).all() and np.isfinite(response).all(), "nonfinite prediction")
                require(summary[name] == scores(target, forecast, response, row), "saved score differs")
    print("Verified matched-state forecasts and responses without fitting.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "verify"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-sha256")
    args = parser.parse_args()
    if args.mode == "run":
        run(args.output)
    else:
        require(args.manifest_sha256 is not None, "expected manifest required")
        verify(args.output, args.manifest_sha256)
