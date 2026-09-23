"""Freeze and score high-spin command-response truth from two recorded flights."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/harness/high-spin-response-v1.json"
HORIZONS = (10, 50, 100, 150, 250)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seal(path, kind):
    write(path / "manifest.json", dict(
        format=kind,
        files={str(p.relative_to(path)): digest(p) for p in sorted(path.rglob("*")) if p.is_file()},
    ))
    return digest(path / "manifest.json")


def authenticate(path, expected):
    if digest(path / "manifest.json") != expected:
        raise ValueError("manifest authority differs")
    manifest = read(path / "manifest.json")
    names = {str(p.relative_to(path)) for p in path.rglob("*") if p.is_file() and p.name != "manifest.json"}
    if names != set(manifest["files"]):
        raise ValueError("artifact inventory differs")
    for name, wanted in manifest["files"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or digest(path / relative) != wanted:
            raise ValueError(f"artifact hash differs: {name}")


def source_case(source, name):
    with np.load(source / name / "stream.npz") as tape:
        states, commands = tape["states"].copy(), tape["commands"].copy()
    if states.shape != (len(commands) + 1, 13) or commands.shape[1:] != (4,):
        raise ValueError("invalid source dimensions")
    return states, commands


def branch(plant_type, config, ratio, snapshot, commands, row, command):
    plant = plant_type(config)
    try:
        plant.set_arm_length_ratio(ratio)
        plant.reset(snapshot.state, applied_motor_thrust_fraction=snapshot.applied_motor_thrust_fraction)
        return np.asarray([
            plant.step(command if index == row else commands[index]).state
            for index in range(row, row + 25)
        ])
    finally:
        plant.close()


def collect(source, spec, output):
    from glassbox_throw.plant import CrazyflowPlant, CrazyflowPlantConfig

    config = CrazyflowPlantConfig(control_frequency_hz=100)
    rows = spec["measurement"]["origins"]
    epsilon = spec["measurement"]["command_delta"]
    output.mkdir(parents=True, exist_ok=False)
    (output / "benchmark.json").write_bytes(PROTOCOL.read_bytes())
    write(output / "binding.json", dict(source_manifest_sha256=spec["source"]["manifest_sha256"]))
    for item in spec["source"]["cases"]:
        name, ratio = item["id"], item["arm_ratio"]
        states, commands = source_case(source, name)
        replay = CrazyflowPlant(config)
        try:
            replay.set_arm_length_ratio(ratio)
            replay.reset(states[0], applied_motor_thrust_fraction=np.zeros(4))
            snapshots = {}
            for index, command in enumerate(commands):
                if index in rows:
                    snapshots[index] = replay.snapshot()
                if np.max(np.abs(replay.step(command).state - states[index + 1])) > 1e-5:
                    raise ValueError(f"source replay differs at row {index}")
        finally:
            replay.close()
        jacobians, replay_errors = [], []
        for row in rows:
            snapshot = snapshots[row]
            nominal = branch(CrazyflowPlant, config, ratio, snapshot, commands, row, commands[row])
            replay_errors.append(float(np.max(np.abs(nominal - states[row + 1 : row + 26]))))
            columns = []
            for channel in range(4):
                low, high = commands[row].copy(), commands[row].copy()
                low[channel] = np.clip(low[channel] - epsilon, 0, 1)
                high[channel] = np.clip(high[channel] + epsilon, 0, 1)
                minus = branch(CrazyflowPlant, config, ratio, snapshot, commands, row, low)
                plus = branch(CrazyflowPlant, config, ratio, snapshot, commands, row, high)
                columns.append((plus[:, 10:13] - minus[:, 10:13]) / (high[channel] - low[channel]))
            jacobians.append(np.stack(columns, axis=-1))
        np.savez_compressed(
            output / f"{name}.npz", rows=np.asarray(rows), commands=commands[rows],
            jacobian=np.asarray(jacobians), replay_error=np.asarray(replay_errors),
        )
    print("truth_manifest_sha256", seal(output, spec["id"] + "-truth"))


def scores(rows, states, forecast, candidate, truth):
    result = {}
    for index, row in enumerate(rows):
        response = {}
        for ms in HORIZONS:
            step = ms // 10 - 1
            observed = truth[index, step]
            predicted = candidate[index, step]
            response[str(ms)] = dict(
                relative_error=float(np.linalg.norm(predicted - observed) / np.linalg.norm(observed)),
                truth_norm=float(np.linalg.norm(observed)),
                model_norm=float(np.linalg.norm(predicted)),
            )
        result[str(row)] = dict(
            rate_250_error=float(np.linalg.norm(forecast[index, -1, 3:6] - states[row + 25, 3:6])),
            hold_rate_250_error=float(np.linalg.norm(states[row, 3:6] - states[row + 25, 3:6])),
            response=response,
        )
    return result


def score(source, truth_root, spec, output):
    from verify_baseline import observed

    from glassbox import STATE_CHANNELS, OnlineFit, SequenceCollection, SequenceSegment

    rows = np.asarray(spec["measurement"]["origins"])
    epsilon = spec["measurement"]["command_delta"]
    output.mkdir(parents=True, exist_ok=False)
    (output / "benchmark.json").write_bytes(PROTOCOL.read_bytes())
    write(output / "binding.json", dict(
        source_manifest_sha256=spec["source"]["manifest_sha256"],
        truth_manifest_sha256=digest(truth_root / "manifest.json"),
        rate_sha256=digest(ROOT / "src/glassbox/_rate.py"),
        dynamics_sha256=digest(ROOT / "src/glassbox/_dynamics.py"),
    ))
    summary = {}
    for item in spec["source"]["cases"]:
        name = item["id"]
        raw, commands = source_case(source, name)
        states = observed(raw)
        with np.load(truth_root / f"{name}.npz") as saved:
            truth_rows, truth_commands, truth = (saved[key].copy() for key in ("rows", "commands", "jacobian"))
        if not np.array_equal(truth_rows, rows) or not np.array_equal(truth_commands, commands[rows]):
            raise ValueError("counterfactual origins differ from source")
        prefix = SequenceCollection((SequenceSegment(
            name, "prefix", states[50:126], commands[50:125], .01, 50,
        ),), name, STATE_CHANNELS, tuple(f"motor_{i} [1]" for i in range(4)))
        session = OnlineFit(prefix)
        forecasts, responses = [], []
        for row in rows:
            for index in range(session.cursor, int(row)):
                session.observe(index, commands[index], states[index + 1])
            h = session.model.history_steps
            past, previous = states[row - h : row + 1], commands[row - h : row]
            future = commands[row : row + 25].copy()
            forecasts.append(np.asarray(session.predict(past, previous, future)))
            columns = []
            for channel in range(4):
                low, high = future.copy(), future.copy()
                low[0, channel] = np.clip(low[0, channel] - epsilon, 0, 1)
                high[0, channel] = np.clip(high[0, channel] + epsilon, 0, 1)
                minus = np.asarray(session.predict(past, previous, low))[:, 3:6]
                plus = np.asarray(session.predict(past, previous, high))[:, 3:6]
                columns.append((plus - minus) / (high[0, channel] - low[0, channel]))
            responses.append(np.stack(columns, axis=-1))
        forecasts, responses = np.asarray(forecasts), np.asarray(responses)
        np.savez_compressed(output / f"{name}.npz", rows=rows, forecast=forecasts, response=responses)
        summary[name] = scores(rows, states, forecasts, responses, truth)
    write(output / "summary.json", summary)
    verify(source, truth_root, spec, output)
    print("evaluation_manifest_sha256", seal(output, spec["id"] + "-evaluation"))


def verify(source, truth_root, spec, output):
    from verify_baseline import observed

    if (output / "benchmark.json").read_bytes() != PROTOCOL.read_bytes():
        raise ValueError("protocol changed")
    bound = read(output / "binding.json")
    if bound["source_manifest_sha256"] != spec["source"]["manifest_sha256"] or bound["truth_manifest_sha256"] != digest(truth_root / "manifest.json"):
        raise ValueError("source/truth binding differs")
    summary = read(output / "summary.json")
    if set(summary) != {item["id"] for item in spec["source"]["cases"]}:
        raise ValueError("case inventory differs")
    rows = np.asarray(spec["measurement"]["origins"])
    for name in summary:
        raw, commands = source_case(source, name)
        states = observed(raw)
        with np.load(truth_root / f"{name}.npz") as saved:
            truth = saved["jacobian"].copy()
            if not np.array_equal(saved["rows"], rows) or not np.array_equal(saved["commands"], commands[rows]) or np.max(saved["replay_error"]) > 1e-5:
                raise ValueError("invalid counterfactual truth")
        with np.load(output / f"{name}.npz") as saved:
            forecast, response = saved["forecast"].copy(), saved["response"].copy()
            if not np.array_equal(saved["rows"], rows):
                raise ValueError("candidate origin mismatch")
        if forecast.shape != (2, 25, 15) or response.shape != truth.shape != (2, 25, 3, 4) or not np.isfinite(forecast).all() or not np.isfinite(response).all():
            raise ValueError("invalid candidate shape or value")
        if summary[name] != scores(rows, states, forecast, response, truth):
            raise ValueError("saved physical scores differ")
    print("Verified high-spin responses and forecasts from saved arrays without fitting.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("collect", "score", "verify"))
    parser.add_argument("--truth", type=Path)
    parser.add_argument("--truth-manifest-sha256")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest-sha256")
    args = parser.parse_args()
    spec = read(PROTOCOL)
    source = Path(spec["source"]["path"])
    authenticate(source, spec["source"]["manifest_sha256"])
    if args.mode == "collect":
        collect(source, spec, args.output)
    else:
        if args.truth is None or args.truth_manifest_sha256 is None:
            parser.error("scoring and verification require sealed truth")
        authenticate(args.truth, args.truth_manifest_sha256)
        if args.mode == "score":
            score(source, args.truth, spec, args.output)
        else:
            if args.manifest_sha256 is None:
                parser.error("verification requires the evaluation manifest")
            authenticate(args.output, args.manifest_sha256)
            verify(source, args.truth, spec, args.output)


if __name__ == "__main__":
    main()
