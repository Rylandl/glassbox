"""Score the predeclared fresh Crazyflow roster with causal prefix fits."""

import argparse
import json
from pathlib import Path

import jax
import numpy as np
from benchmark_online_readout import observed
from qualify_heldout_causal import authenticate, digest, errors

from glassbox._causal_actuator import CausalActuatorModel
from glassbox._causal_fit import fit_episode

CASES = ("quad-arm-095-spin", "quad-arm-155-canonical")
PARAMETERS = ("q", "coeff", "inertia", "force", "torque")
BEGIN = 50
FIRST = 125
STRIDE = 25
HORIZON = 25


def load_case(source, name):
    report = json.loads((source / f"{name}.json").read_text())
    with np.load(source / f"{name}.npz", allow_pickle=False) as data:
        raw = data["states"]
        commands = data["commands"]
        time_s = data["time_s"]
    if (
        raw.ndim != 2
        or raw.shape[1] != 13
        or commands.ndim != 2
        or commands.shape[1] != 4
        or len(raw) != len(commands) + 1
        or len(time_s) != len(raw)
        or not np.isfinite(raw).all()
        or not np.isfinite(commands).all()
        or not np.isfinite(time_s).all()
        or not np.allclose(np.diff(time_s), 0.01, rtol=0, atol=1e-8)
    ):
        raise ValueError(f"invalid source stream: {name}")
    origins = np.arange(FIRST, len(commands) - HORIZON + 1, STRIDE, dtype=int)
    if report["intervals"] != len(commands):
        raise ValueError("source report and stream length disagree")
    if len(origins) < 2 or report["status"] != "complete":
        return dict(report=report, qualification="insufficient complete flight")
    states = observed(raw)
    truth = np.stack([states[row + 1 : row + HORIZON + 1] for row in origins])
    hold = np.repeat(states[origins, None, :], HORIZON, axis=1)
    return dict(
        report=report,
        qualification="scored",
        states=states,
        commands=commands,
        origins=origins,
        truth=truth,
        hold=hold,
    )


def score(case, forecast, archive):
    truth = case["truth"]
    return {
        "source_report": case["report"],
        "qualification": "scored",
        "origins": case["origins"].tolist(),
        "max_recorded_body_rate_rad_s": float(
            np.max(np.linalg.norm(case["states"][:, 3:6], axis=1))
        ),
        "candidate": errors(forecast, truth),
        "hold": errors(case["hold"], truth),
        "candidate_250_rate_by_origin": np.linalg.norm(
            forecast[:, -1, 3:6] - truth[:, -1, 3:6], axis=1
        ).tolist(),
        "archive_sha256": digest(archive),
    }


def fit_case(case, output, name):
    models, forecasts, seconds = [], [], []
    for origin in case["origins"]:
        model, report = fit_episode(
            case["states"][BEGIN : origin + 1],
            case["commands"][BEGIN:origin],
            0.01,
        )
        with jax.enable_x64(True):
            forecast = np.asarray(
                model.forecast(
                    case["states"][origin],
                    case["commands"][BEGIN:origin],
                    case["commands"][origin : origin + HORIZON],
                )
            )
        if forecast.shape != (HORIZON, 15) or not np.isfinite(forecast).all():
            raise ValueError(f"invalid forecast: {name} row {origin}")
        models.append(model)
        forecasts.append(forecast)
        seconds.append(report["fit_seconds"])
    archive = output / f"{name}.npz"
    np.savez_compressed(
        archive,
        origins=case["origins"],
        forecast=np.stack(forecasts),
        fit_seconds=np.asarray(seconds),
        **{
            key: np.stack([getattr(model, key) for model in models])
            for key in PARAMETERS
        },
    )
    return score(case, np.stack(forecasts), archive)


def verify_case(case, output, name):
    archive = output / f"{name}.npz"
    with np.load(archive, allow_pickle=False) as data:
        saved = {key: data[key] for key in data.files}
    if (
        set(saved) != {"origins", "forecast", "fit_seconds", *PARAMETERS}
        or not np.array_equal(saved["origins"], case["origins"])
        or saved["forecast"].shape != case["truth"].shape
    ):
        raise ValueError("saved roster or forecast differs")
    for i, origin in enumerate(case["origins"]):
        model = CausalActuatorModel(0.01, **{key: saved[key][i] for key in PARAMETERS})
        with jax.enable_x64(True):
            replay = np.asarray(
                model.forecast(
                    case["states"][origin],
                    case["commands"][BEGIN:origin],
                    case["commands"][origin : origin + HORIZON],
                )
            )
        np.testing.assert_allclose(saved["forecast"][i], replay, rtol=0, atol=2e-11)
    return score(case, saved["forecast"], archive)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    authenticate(args.source, args.source_manifest_sha256)
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {"source_manifest_sha256": args.source_manifest_sha256, "cases": {}}
    for name in CASES:
        case = load_case(args.source, name)
        if case["qualification"] != "scored":
            summary["cases"][name] = {
                "qualification": case["qualification"],
                "source_report": case["report"],
            }
            continue
        summary["cases"][name] = (
            verify_case(case, args.output, name)
            if args.verify
            else fit_case(case, args.output, name)
        )
        print(name, summary["cases"][name]["candidate"]["250"], flush=True)
    path = args.output / "summary.json"
    if args.verify:
        if summary != json.loads(path.read_text()):
            raise ValueError("saved fresh-flight scores differ")
        print("verified fresh flights without fitting")
    else:
        path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
