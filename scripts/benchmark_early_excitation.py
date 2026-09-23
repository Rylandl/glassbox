"""Compare causal early fits on original and separately excited quad episodes."""

import argparse
import hashlib
import importlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from benchmark_online_readout import (
    GIT,
    SPEC as ONLINE_SPEC,
    authenticate_sources,
    read,
    response_jacobian,
    source_case,
    write,
)
from collect_throw import authenticate, seal
from evaluate_online import metrics
from glassbox import STATE_CHANNELS, SequenceCollection, SequenceSegment
from verify_baseline import observed, require

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "docs/early-excitation-benchmark.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def branches(spec):
    require(sha(ONLINE_SPEC) == spec["online_benchmark_sha256"], "benchmark changed")
    authenticate_sources(read(ONLINE_SPEC))
    authenticate(Path(spec["dither_path"]), spec["dither_manifest_sha256"])
    case = source_case(read(ONLINE_SPEC), spec["recording"], "full")
    row = spec["origin_row"]
    require(row == case["origins"][0], "origin mismatch")
    require(spec["prefix_begin_row"] == case["begin"], "prefix mismatch")
    require(spec["dt_s"] == case["dt"], "sample interval mismatch")
    require(spec["horizon_ms"] == round(1000 * case["horizon"] * case["dt"]),
            "horizon mismatch")
    saved = np.load(Path(spec["dither_path"]) / "glassbox-arm125-dither.npz")
    dither_states = observed(saved["states"])
    dither_commands = np.asarray(saved["commands"], dtype=float)
    require(dither_states.shape == case["states"][:151].shape and
            dither_commands.shape == case["commands"][:150].shape,
            "dither trajectory shape mismatch")
    require(np.array_equal(dither_commands[row:], case["commands"][row:150]),
            "commands after excitation differ")
    return case, {
        "original": (case["states"], case["commands"], case["response"]["truth"][0]),
        "dither": (dither_states, dither_commands, saved["truth_jacobian"]),
    }


def score(spec, states, commands, truth_response, forecast, response):
    row = spec["origin_row"]
    horizon = round(spec["horizon_ms"] / 1000 / spec["dt_s"])
    actual = states[row + 1 : row + horizon + 1]
    require(forecast.shape == actual.shape == (horizon, 15), "forecast shape")
    require(response.shape == truth_response.shape == (3, 4), "response shape")
    require(np.isfinite(forecast).all() and np.isfinite(response).all(),
            "nonfinite prediction")
    window = commands[row - 25 : row]
    singular = np.linalg.svd(window - window.mean(axis=0), compute_uv=False)
    return dict(
        rate_250_rad_s=metrics(forecast[-1:], actual[-1:])["body_rate_rmse_rad_s"],
        velocity_250_m_s=metrics(forecast[-1:], actual[-1:])["velocity_rmse_m_s"],
        rate_250_endpoint_norm_rad_s=float(
            np.linalg.norm(forecast[-1, 3:6] - actual[-1, 3:6])
        ),
        response_relative_error=float(
            np.linalg.norm(response - truth_response) / np.linalg.norm(truth_response)
        ),
        command_singular_values=singular.tolist(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "verify"))
    parser.add_argument("--candidate", help="module:factory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-sha256")
    args = parser.parse_args()
    spec = read(SPEC if args.mode == "run" else args.output / "benchmark.json")
    case, branch_data = branches(spec)
    if args.mode == "verify":
        authenticate(args.output, args.manifest_sha256)
        summary = read(args.output / "summary.json")
        for name in spec["branches"]:
            saved = np.load(args.output / f"{name}.npz")
            states, commands, truth_response = branch_data[name]
            require(summary["branches"][name]["scores"] == score(
                spec, states, commands, truth_response,
                saved["forecast"], saved["response"]
            ), "saved score mismatch")
        print("Verified early forecasts and responses from sealed data without fitting.")
        return
    require(args.candidate and ":" in args.candidate, "supply module:factory")
    module_name, factory_name = args.candidate.split(":", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, factory_name)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "benchmark.json").write_bytes(SPEC.read_bytes())
    write(args.output / "binding.json", dict(
        candidate=args.candidate,
        candidate_sha256=sha(module.__file__),
        runner_sha256=sha(__file__),
        git_head=subprocess.check_output([GIT, "rev-parse", "HEAD"], text=True).strip(),
    ))
    results = {}
    try:
        row = spec["origin_row"]
        for name in spec["branches"]:
            states, commands, truth_response = branch_data[name]
            prefix = SequenceCollection((SequenceSegment(
                case["identity"], "prefix", states[case["begin"] : row + 1],
                commands[case["begin"] : row], case["dt"], case["begin"]
            ),), case["identity"], STATE_CHANNELS, case["names"])
            started = time.perf_counter()
            fit = factory(prefix)
            initialization_s = time.perf_counter() - started
            h = fit.session.model.history_steps
            past = states[row - h : row + 1]
            inputs = commands[row - h : row]
            started = time.perf_counter()
            forecast = np.asarray(fit.session.predict(
                past, inputs, commands[row : row + case["horizon"]]
            ))
            first_prediction_s = time.perf_counter() - started
            response = response_jacobian(
                fit.session, past, inputs, commands[row], spec["response_epsilon"]
            )
            np.savez_compressed(
                args.output / f"{name}.npz", forecast=forecast, response=response
            )
            results[name] = dict(
                scores=score(spec, states, commands, truth_response, forecast, response),
                initialization_s=initialization_s,
                first_prediction_s=first_prediction_s,
            )
        write(args.output / "summary.json", dict(branches=results))
    except BaseException as error:
        write(args.output / "failure.json", dict(error=repr(error)))
        raise
    finally:
        print("manifest_sha256", seal(args.output, spec["id"]))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
