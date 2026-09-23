"""Replay frozen observations at recorded cadence and score published revisions.

Prediction is deferred until after the timed replay so its JAX compilation does
not change which background fit would have been available at each origin.
"""

import argparse
import json
import time
from pathlib import Path

import jax
import numpy as np
from benchmark_online_readout import SPEC, read, source_case
from qualify_causal_public import PUBLIC, _score

from glassbox import STATE_CHANNELS, OnlineFit, SequenceCollection, SequenceSegment
from glassbox._causal_actuator import CausalActuatorModel

PARAMETERS = ("q", "coeff", "inertia", "force", "torque")


def _prefix(case):
    first = int(case["origins"][0])
    return SequenceCollection(
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


def _forecast(case, origin, model):
    with jax.enable_x64(True):
        return np.asarray(
            model.forecast(
                case["states"][origin],
                case["commands"][case["begin"] : origin],
                case["commands"][origin : origin + case["horizon"]],
            )
        )


def _summary(case, saved, public):
    report = _score(case, saved["forecast"], public)
    lag = saved["origins"] - saved["published_cursor"]
    report["publication"] = {
        "initial_fit_s": float(saved["initial_fit_s"]),
        "replay_wall_s": float(saved["replay_wall_s"]),
        "recorded_duration_s": float(saved["recorded_duration_s"]),
        "maximum_schedule_lateness_s": float(saved["maximum_schedule_lateness_s"]),
        "revisions_seen": len(np.unique(saved["published_cursor"])),
        "mean_model_age_s": float(np.mean(lag) * case["dt"]),
        "maximum_model_age_s": float(np.max(lag) * case["dt"]),
        "publication_cursors": saved["published_cursor"].tolist(),
    }
    return report


def evaluate_case(name, suite, output):
    case = source_case(read(SPEC), name, suite)
    origin_set = set(case["origins"])
    started = time.perf_counter()
    session = OnlineFit(_prefix(case))
    initial_fit_s = time.perf_counter() - started
    captures, lateness = [], []
    first, end = int(case["origins"][0]), case["end_row"]
    replay_start = time.perf_counter()
    try:
        for row in range(first, end + 1):
            due = replay_start + (row - first) * case["dt"]
            remaining = due - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
            lateness.append(max(0.0, time.perf_counter() - due))
            if row in origin_set:
                _, _, _, published, model, report = session._fit.snapshot()
                captures.append((row, published, model, report["fit_seconds"]))
            if row < end:
                session.observe(row, case["commands"][row], case["states"][row + 1])
        replay_wall_s = time.perf_counter() - replay_start
    finally:
        session.close()
    assert np.array_equal([c[0] for c in captures], case["origins"])
    forecasts = np.stack([_forecast(case, row, model) for row, _, model, _ in captures])
    assert forecasts.shape == case["truth"].shape and np.isfinite(forecasts).all()
    arrays = {
        "origins": case["origins"],
        "published_cursor": np.asarray([c[1] for c in captures]),
        "forecast": forecasts,
        "fit_seconds": np.asarray([c[3] for c in captures]),
        "initial_fit_s": np.asarray(initial_fit_s),
        "replay_wall_s": np.asarray(replay_wall_s),
        "recorded_duration_s": np.asarray((end - first) * case["dt"]),
        "maximum_schedule_lateness_s": np.asarray(max(lateness)),
    }
    arrays.update(
        {key: np.stack([getattr(c[2], key) for c in captures]) for key in PARAMETERS}
    )
    np.savez_compressed(output / f"{name}.npz", **arrays)
    return _summary(
        case, arrays, np.load(PUBLIC / f"{name}.npz")["forecast"][: len(captures)]
    )


def verify_case(name, suite, output):
    case = source_case(read(SPEC), name, suite)
    with np.load(output / f"{name}.npz") as stored:
        saved = {key: stored[key] for key in stored.files}
    assert set(saved) == {
        "origins",
        "published_cursor",
        "forecast",
        "fit_seconds",
        "initial_fit_s",
        "replay_wall_s",
        "recorded_duration_s",
        "maximum_schedule_lateness_s",
        *PARAMETERS,
    }
    assert np.array_equal(saved["origins"], case["origins"])
    assert saved["forecast"].shape == case["truth"].shape
    assert len(saved["published_cursor"]) == len(case["origins"])
    assert np.all(saved["published_cursor"] <= saved["origins"])
    assert np.all(np.diff(saved["published_cursor"]) >= 0)
    for index, origin in enumerate(case["origins"]):
        model = CausalActuatorModel(
            case["dt"], **{key: saved[key][index] for key in PARAMETERS}
        )
        np.testing.assert_allclose(
            saved["forecast"][index],
            _forecast(case, int(origin), model),
            rtol=0,
            atol=2e-11,
        )
    public = np.load(PUBLIC / f"{name}.npz")["forecast"][: len(case["origins"])]
    return _summary(case, saved, public)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=("smoke", "full"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", action="append")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    names = args.name or read(SPEC)[f"{args.suite}_cases"]
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.verify:
        for name in names:
            result = evaluate_case(name, args.suite, args.output)
            print(name, result["publication"], result["candidate"]["250"], flush=True)
    result = {
        "suite": args.suite,
        "cases": {name: verify_case(name, args.suite, args.output) for name in names},
    }
    path = args.output / "summary.json"
    if args.verify:
        assert result == json.loads(path.read_text())
        print(f"verified {len(names)} saved revision forecasts without fitting")
    else:
        path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
