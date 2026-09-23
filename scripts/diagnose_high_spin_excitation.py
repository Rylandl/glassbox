"""Replay high-spin tapes with a frozen independent-input excitation pattern."""

import argparse
import json
from pathlib import Path

import numpy as np
from collect_throw import authenticate, seal
from verify_baseline import observed, require

from glassbox import STATE_CHANNELS, OnlineFit, SequenceCollection, SequenceSegment

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/harness/high-spin-excitation-v1.json"
PATTERN = np.array([[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]])


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def source_case(source, name):
    with np.load(source / name / "stream.npz") as tape:
        return tape["states"].copy(), tape["commands"].copy()


def replay(plant_type, config, ratio, raw, issued, amplitude, measure):
    plant = plant_type(config)
    try:
        plant.set_arm_length_ratio(ratio)
        sample = plant.reset(raw[0], applied_motor_thrust_fraction=np.zeros(4))
        states = [sample.state]
        commands = []
        snapshot = None
        for row in range(measure["forecast_end_row"]):
            command = issued[row].copy()
            if measure["replay_through_row"] <= row < measure["excitation_end_row"]:
                index = row - measure["replay_through_row"]
                sign = 1 if index % 8 < 4 else -1
                command = np.clip(command + amplitude * sign * PATTERN[index % 4], 0, 1)
            if row == measure["excitation_end_row"]:
                snapshot = plant.snapshot()
            commands.append(command)
            states.append(plant.step(command).state)
            if row < measure["replay_through_row"]:
                require(np.max(np.abs(states[-1] - raw[row + 1])) < 1e-5, "original prefix replay differs")
        states, commands = np.asarray(states), np.asarray(commands)
        require(snapshot is not None, "missing forecast snapshot")
        if amplitude == 0:
            require(np.max(np.abs(states - raw[: len(states)])) < 1e-5, "unexcited replay differs")
        return states, commands, snapshot
    finally:
        plant.close()


def truth_response(plant_type, config, ratio, snapshot, commands, row, delta):
    from qualify_high_spin_response import branch

    columns = []
    for channel in range(commands.shape[1]):
        low, high = commands[row].copy(), commands[row].copy()
        low[channel] = np.clip(low[channel] - delta, 0, 1)
        high[channel] = np.clip(high[channel] + delta, 0, 1)
        minus = branch(plant_type, config, ratio, snapshot, commands, row, low)
        plus = branch(plant_type, config, ratio, snapshot, commands, row, high)
        columns.append((plus[:, 10:13] - minus[:, 10:13]) / (high[channel] - low[channel]))
    return np.stack(columns, axis=-1)


def predict(states, commands, row, begin, delta):
    canon = observed(states)
    prefix = SequenceCollection(
        (SequenceSegment("excitation-diagnosis", "prefix", canon[begin : row + 1], commands[begin:row], .01, begin),),
        "excitation-diagnosis", STATE_CHANNELS, tuple(f"motor_{i} [1]" for i in range(4)),
    )
    session = OnlineFit(prefix)
    h = session.model.history_steps
    past, prior = canon[row - h : row + 1], commands[row - h : row]
    future = commands[row : row + 25]
    factual = np.asarray(session.predict(past, prior, future))
    columns = []
    for channel in range(4):
        low, high = future.copy(), future.copy()
        low[0, channel] = np.clip(low[0, channel] - delta, 0, 1)
        high[0, channel] = np.clip(high[0, channel] + delta, 0, 1)
        minus = np.asarray(session.predict(past, prior, low))[:, 3:6]
        plus = np.asarray(session.predict(past, prior, high))[:, 3:6]
        columns.append((plus - minus) / (high[0, channel] - low[0, channel]))
    return factual, np.stack(columns, axis=-1), canon


def score(states, factual, response, response_truth, row):
    actual = observed(states)[row + 1 : row + 26]
    start_rate = observed(states)[row, 3:6]
    output = {}
    for ms in (50, 100, 250):
        step = ms // 10 - 1
        output[str(ms)] = dict(
            rate_error_rad_s=float(np.linalg.norm(factual[step, 3:6] - actual[step, 3:6])),
            hold_rate_error_rad_s=float(np.linalg.norm(start_rate - actual[step, 3:6])),
            response_relative_error=float(np.linalg.norm(response[step] - response_truth[step]) / np.linalg.norm(response_truth[step])),
        )
    return output


def run(output):
    from glassbox_throw.plant import CrazyflowPlant, CrazyflowPlantConfig

    spec = read(PROTOCOL)
    source = Path(spec["source"]["path"])
    authenticate(source, spec["source"]["manifest_sha256"])
    measure = spec["measurement"]
    config = CrazyflowPlantConfig(control_frequency_hz=100)
    output.mkdir(parents=True, exist_ok=False)
    (output / "protocol.json").write_bytes(PROTOCOL.read_bytes())
    summary = {}
    for item in spec["cases"]:
        raw, issued = source_case(source, item["id"])
        for amplitude in measure["amplitudes"]:
            name = item["id"] + ("-control" if amplitude == 0 else "-orthogonal")
            states, commands, snapshot = replay(CrazyflowPlant, config, item["arm_ratio"], raw, issued, amplitude, measure)
            row = measure["excitation_end_row"]
            true_response = truth_response(CrazyflowPlant, config, item["arm_ratio"], snapshot, commands, row, measure["probe_delta"])
            factual, response, _ = predict(states, commands, row, measure["prefix_begin_row"], measure["probe_delta"])
            np.savez_compressed(output / f"{name}.npz", states=states, commands=commands, factual=factual, response=response, response_truth=true_response)
            summary[name] = score(states, factual, response, true_response, row)
            print(name, json.dumps(summary[name], sort_keys=True), flush=True)
    write(output / "summary.json", summary)
    print("manifest_sha256", seal(output, spec["id"]), flush=True)


def verify(output, expected):
    spec = read(PROTOCOL)
    authenticate(output, expected)
    require((output / "protocol.json").read_bytes() == PROTOCOL.read_bytes(), "protocol differs")
    summary = read(output / "summary.json")
    expected_names = {item["id"] + suffix for item in spec["cases"] for suffix in ("-control", "-orthogonal")}
    require(set(summary) == expected_names, "case roster differs")
    row = spec["measurement"]["excitation_end_row"]
    for name in summary:
        with np.load(output / f"{name}.npz") as saved:
            states, commands, factual, response, truth = (saved[key].copy() for key in ("states", "commands", "factual", "response", "response_truth"))
        require(states.shape == (176, 13) and commands.shape == (175, 4), "recording shape differs")
        require(factual.shape == (25, 15) and response.shape == truth.shape == (25, 3, 4), "prediction shape differs")
        require(np.isfinite(states).all() and np.isfinite(commands).all() and np.isfinite(factual).all() and np.isfinite(response).all() and np.isfinite(truth).all(), "nonfinite evidence")
        require(summary[name] == score(states, factual, response, truth, row), "saved score differs")
    print("Verified excitation diagnosis from saved arrays without fitting.")


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
