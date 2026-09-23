"""Compare the public causal equation with frozen forecasts on matched prefixes.

The first state and command visible to a candidate are exactly those passed to
the original online benchmark. Fitting, latent reconstruction and prediction
all start at ``case['begin']``. This is identification evidence, not a timed
background-publication replay.
"""

import argparse
import json
from pathlib import Path

import jax
import numpy as np
from benchmark_online_readout import SPEC, read, source_case
from scipy.spatial.transform import Rotation

from glassbox._causal_fit import fit_episode

PUBLIC = (
    Path(read(SPEC)["sources"]["recordings"]["path"]).parents[1]
    / "readout-public-rate-v1/public-rate-lean-full"
)
HORIZONS_MS = (50, 100, 150, 200, 250)


def _errors(prediction, truth, dt):
    result = {}
    for ms in HORIZONS_MS:
        step = round(ms / (1000 * dt)) - 1
        a, b = prediction[:, step], truth[:, step]
        orientation = [
            Rotation.from_matrix(
                x[6:].reshape(3, 3).T @ y[6:].reshape(3, 3)
            ).magnitude()
            for x, y in zip(b, a, strict=True)
        ]
        result[str(ms)] = {
            "velocity_m_s": float(
                np.sqrt(np.mean(np.sum((a[:, :3] - b[:, :3]) ** 2, axis=1)))
            ),
            "body_rate_rad_s": float(
                np.sqrt(np.mean(np.sum((a[:, 3:6] - b[:, 3:6]) ** 2, axis=1)))
            ),
            "orientation_rad": float(np.sqrt(np.mean(np.square(orientation)))),
        }
    return result


def _score(case, candidate, public):
    return {
        "origins": len(case["origins"]),
        "candidate": _errors(candidate, case["truth"], case["dt"]),
        "public": _errors(public, case["truth"], case["dt"]),
    }


def _fit(case, origin):
    begin = case["begin"]
    model, report = fit_episode(
        case["states"][begin : origin + 1],
        case["commands"][begin:origin],
        case["dt"],
    )
    return model, report


def _forecast(case, origin, model, future):
    with jax.enable_x64(True):
        return np.asarray(
            model.forecast(
                case["states"][origin],
                case["commands"][case["begin"] : origin],
                future,
            )
        )


def _response(case, origin, model):
    command = case["commands"][origin]
    epsilon = case["response"]["epsilon"]
    columns = []
    for channel in range(len(command)):
        low, high = command.copy(), command.copy()
        low[channel] = max(0.0, low[channel] - epsilon)
        high[channel] = min(1.0, high[channel] + epsilon)
        minus = _forecast(case, origin, model, low[None])[0, 3:6]
        plus = _forecast(case, origin, model, high[None])[0, 3:6]
        columns.append((plus - minus) / (high[channel] - low[channel]))
    return np.stack(columns, axis=1)


def evaluate_case(name, suite, output):
    case = source_case(read(SPEC), name, suite)
    frozen = np.load(PUBLIC / f"{name}.npz")
    assert np.array_equal(frozen["origins"][: len(case["origins"])], case["origins"])
    forecasts, responses, fit_seconds = [], [], []
    response_rows = set(case["response"]["rows"]) if case["response"] else set()
    for origin in case["origins"]:
        origin = int(origin)
        model, report = _fit(case, origin)
        forecast = _forecast(
            case,
            origin,
            model,
            case["commands"][origin : origin + case["horizon"]],
        )
        assert forecast.shape == (case["horizon"], 15) and np.isfinite(forecast).all()
        forecasts.append(forecast)
        fit_seconds.append(report["fit_seconds"])
        if origin in response_rows:
            responses.append(_response(case, origin, model))
    prediction = np.stack(forecasts)
    np.savez_compressed(
        output / f"{name}.npz",
        origins=case["origins"],
        forecast=prediction,
        fit_seconds=np.asarray(fit_seconds),
        response=np.asarray(responses),
    )
    return _score(case, prediction, frozen["forecast"][: len(case["origins"])])


def verify_case(name, suite, output):
    case = source_case(read(SPEC), name, suite)
    frozen = np.load(PUBLIC / f"{name}.npz")
    saved = np.load(output / f"{name}.npz")
    assert np.array_equal(saved["origins"], case["origins"])
    assert saved["forecast"].shape == case["truth"].shape
    assert len(saved["fit_seconds"]) == len(case["origins"])
    assert np.isfinite(saved["forecast"]).all()
    if case["response"] is not None:
        truth = case["response"]["truth"]
        candidate = saved["response"]
        public = frozen["response"][: len(truth)]
        assert candidate.shape == public.shape == truth.shape

        def relative(a):
            return (
                np.linalg.norm(a - truth, axis=(1, 2))
                / np.linalg.norm(truth, axis=(1, 2))
            ).tolist()

        response = {
            "origins": case["response"]["rows"].tolist(),
            "candidate": relative(candidate),
            "public": relative(public),
        }
    else:
        assert saved["response"].shape == (0,)
        response = None
    return (
        _score(case, saved["forecast"], frozen["forecast"][: len(case["origins"])]),
        response,
    )


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
            print(name, evaluate_case(name, args.suite, args.output), flush=True)
    summary, response = {}, {}
    for name in names:
        summary[name], found = verify_case(name, args.suite, args.output)
        if found is not None:
            response[name] = found
    result = {
        "suite": args.suite,
        "prefix": "matched",
        "cases": summary,
        "response": response,
    }
    path = args.output / "summary.json"
    if args.verify:
        assert result == json.loads(path.read_text())
        print(f"verified {len(names)} cases from saved forecasts without fitting")
    else:
        path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
