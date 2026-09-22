"""Score three unchanged cold-start learners on paired 10/50 ms quad tapes."""

import argparse
import platform
import shutil
import subprocess
import time
from pathlib import Path

import jax
import numpy as np

from collect_throw import authenticate
from collect_paired_quad import GIT, ROOT, digest, read, save, seal, write
from evaluate_online import metrics, residuals
from screen_cold_readout_curvature import CurvedReadout
from screen_readout_sensitivity import SensitivityReadout
from verify_baseline import observed

from glassbox import STATE_CHANNELS, OnlineFit, SequenceCollection, SequenceSegment
from glassbox._learner_arrays import array_fingerprint


PROTOCOL = ROOT / "docs/harness/paired-quad-evaluation-v2.json"
ARMS = ("baseline", "curvature", "candidate")
HORIZONS = (50, 100, 150, 200, 250)


def npz(path):
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def source_binding(spec):
    current = subprocess.check_output([GIT, "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(
        [GIT, "-C", str(ROOT), "status", "--porcelain", "--untracked-files=normal"], text=True
    ).strip()
    assert not dirty, dirty
    assert subprocess.check_output(
        [GIT, "-C", str(ROOT), "merge-base", "--is-ancestor", spec["source"]["maintained_source_commit"], "HEAD"]
    ) == b""
    for name in ("screen_cold_readout_curvature.py", "screen_readout_sensitivity.py"):
        original = subprocess.check_output(
            [GIT, "-C", str(ROOT), "show", f'{spec["source"]["fast_readout_source_commit"]}:scripts/{name}']
        )
        assert digest(ROOT / "scripts" / name) == __import__("hashlib").sha256(original).hexdigest()
    return dict(
        commit=current,
        protocol_sha256=digest(PROTOCOL),
        script_sha256=digest(Path(__file__)),
        fast_source_commit=spec["source"]["fast_readout_source_commit"],
        runtime={"python": platform.python_version(), "jax": jax.__version__,
                 "numpy": np.__version__, "backend": jax.default_backend(),
                 "x64_ambient": bool(jax.config.x64_enabled)},
    )


def session(arm):
    return arm if isinstance(arm, OnlineFit) else arm.session


def latency(values):
    values = np.asarray(values)
    warm = values[1:]
    return dict(
        first_s=float(values[0]),
        warm_median_s=float(np.median(warm)),
        warm_p95_s=float(np.quantile(warm, 0.95)),
        total_s=float(values.sum()),
    )


def summarize(data, tape, spec):
    dt = spec["dt_s"]
    result = dict(
        schedule=spec["id"], dt_s=dt,
        elapsed_prefix_s=spec["first_prediction_row"] * dt,
        updates=len(data["rows"]), conditional_origins=len(data["origin_rows"]),
        recording_duration_s=float(tape["time_s"][-1]),
        command_span=(np.max(tape["commands"], axis=0) - np.min(tape["commands"], axis=0)).tolist(),
        maximum_speed_m_s=float(np.max(np.linalg.norm(tape["states"][:, 3:6], axis=-1))),
        maximum_rate_rad_s=float(np.max(np.linalg.norm(tape["states"][:, 10:13], axis=-1))),
        arms={}, ratios={},
    )
    for arm in ARMS:
        horizon = {
            str(ms): metrics(
                data[f"conditional_{arm}"][:, ms // round(dt * 1000) - 1],
                data["conditional_truth"][:, ms // round(dt * 1000) - 1],
            ) for ms in HORIZONS
        }
        result["arms"][arm] = dict(
            one_step=metrics(data[f"one_step_{arm}"], data["one_step_truth"]),
            horizons=horizon,
            initialization_s=float(data[f"initialization_s_{arm}"][0]),
            one_step_prediction=latency(data[f"predict_s_{arm}"]),
            complete_update=latency(data[f"update_s_{arm}"]),
            complete_update_cpu_per_recorded_second_s=float(
                data[f"update_s_{arm}"].sum() / (data["rows"][-1] * dt - data["rows"][0] * dt + dt)
            ),
            first_conditional_prediction_s=float(data[f"conditional_predict_s_{arm}"][0]),
        )
        one = data[f"conditional_{arm}"][:, -1]
        truth = data["conditional_truth"][:, -1]
        physical, angle = residuals(one, truth)
        assert np.array_equal(np.linalg.norm(physical[:, :3], axis=-1), data[f"error_250_velocity_{arm}"])
        assert np.array_equal(np.linalg.norm(physical[:, 3:6], axis=-1), data[f"error_250_rate_{arm}"])
        assert np.array_equal(angle, data[f"error_250_orientation_{arm}"])
    baseline = result["arms"]["baseline"]["horizons"]
    for arm in ("curvature", "candidate"):
        result["ratios"][arm] = {
            str(ms): {
                channel: result["arms"][arm]["horizons"][str(ms)][channel] /
                max(1e-9, baseline[str(ms)][channel])
                for channel in ("velocity_rmse_m_s", "body_rate_rmse_rad_s", "orientation_rmse_rad")
            } for ms in HORIZONS
        }
    return result


def run_schedule(root, spec, tape, deadline):
    name, dt = spec["id"], spec["dt_s"]
    output = root / name
    output.mkdir()
    states, commands = observed(tape["states"]), tape["commands"].astype(np.float64)
    begin, first, stride, horizon = (
        spec[key] for key in ("prefix_begin_row", "first_prediction_row", "common_origin_stride_rows", "forecast_steps")
    )
    assert first * dt == 1.25 and begin * dt == 0.5
    prefix = SequenceCollection(
        (SequenceSegment(
            "paired-quad-v2", "prefix", states[begin : first + 1], commands[begin:first],
            dt, begin,
        ),),
        "paired-quad-v2", STATE_CHANNELS,
        tuple(f"motor_{index} [1]" for index in range(4)),
    )
    constructor = {"baseline": OnlineFit, "curvature": CurvedReadout,
                   "candidate": SensitivityReadout}
    arms, initialization = {}, {}
    for arm in ARMS:
        started = time.perf_counter()
        arms[arm] = constructor[arm](prefix)
        initialization[arm] = time.perf_counter() - started
        initial_model = session(arms[arm]).model
        save(output / f"initial-{arm}.npz", **initial_model.arrays())
        write(output / f"initial-{arm}.json", initial_model.metadata())
    initial = npz(output / "initial-baseline.npz")
    assert all(
        np.array_equal(value, npz(output / f"initial-{arm}.npz")[key])
        for arm in ARMS[1:] for key, value in initial.items()
    )
    assert all(session(arms[arm]).cursor == first for arm in ARMS)
    origins = set(range(first, len(commands) - horizon + 1, stride))
    assert len(origins) == 35
    recorded = dict(rows=[], origin_rows=[], one_step_truth=[], conditional_truth=[])
    for arm in ARMS:
        recorded.update({
            f"one_step_{arm}": [], f"conditional_{arm}": [],
            f"predict_s_{arm}": [], f"conditional_predict_s_{arm}": [],
            f"update_s_{arm}": [], f"initialization_s_{arm}": [initialization[arm]],
            f"model_fingerprint_{arm}": [],
            f"error_250_velocity_{arm}": [], f"error_250_rate_{arm}": [],
            f"error_250_orientation_{arm}": [],
        })
        if arm != "baseline":
            for field in ("mean", "phi", "target", "penalty"):
                recorded[f"direct_{arm}_{field}"] = []
    for offset, row in enumerate(range(first, len(commands))):
        if time.monotonic() > deadline:
            raise TimeoutError("frozen 900 s online-evaluation budget exceeded")
        recorded["rows"].append(row)
        recorded["one_step_truth"].append(states[row + 1])
        conditional = row in origins
        if conditional:
            recorded["origin_rows"].append(row)
            recorded["conditional_truth"].append(states[row + 1 : row + horizon + 1])
        for arm in ARMS:
            fit = session(arms[arm])
            h = fit.model.history_steps
            past, inputs = states[row - h : row + 1], commands[row - h : row]
            started = time.perf_counter()
            one = np.asarray(fit.predict(past, inputs, commands[row : row + 1]))[0]
            recorded[f"predict_s_{arm}"].append(time.perf_counter() - started)
            assert np.isfinite(one).all()
            recorded[f"one_step_{arm}"].append(one)
            if conditional:
                started = time.perf_counter()
                forecast = np.asarray(fit.predict(
                    past, inputs, commands[row : row + horizon]
                ))
                recorded[f"conditional_predict_s_{arm}"].append(time.perf_counter() - started)
                assert forecast.shape == (horizon, 15) and np.isfinite(forecast).all()
                recorded[f"conditional_{arm}"].append(forecast)
                error, angle = residuals(forecast[-1:], states[row + horizon : row + horizon + 1])
                recorded[f"error_250_velocity_{arm}"].append(np.linalg.norm(error[0, :3]))
                recorded[f"error_250_rate_{arm}"].append(np.linalg.norm(error[0, 3:6]))
                recorded[f"error_250_orientation_{arm}"].append(angle[0])
        for arm in ARMS[offset % 3 :] + ARMS[: offset % 3]:
            started = time.perf_counter()
            result = arms[arm].observe(row, commands[row], states[row + 1])
            model = session(arms[arm]).model
            snapshot = model.arrays()
            recorded[f"update_s_{arm}"].append(time.perf_counter() - started)
            assert session(arms[arm]).cursor == row + 1
            assert all(np.isfinite(value).all() for value in snapshot.values())
            recorded[f"model_fingerprint_{arm}"].append(
                array_fingerprint(model.metadata(), snapshot)
            )
            if arm != "baseline":
                assert len(result) == 4
                for field, value in zip(("mean", "phi", "target", "penalty"), result):
                    recorded[f"direct_{arm}_{field}"].append(value)
        if (offset + 1) % max(1, round(1 / dt)) == 0:
            print(name, "elapsed_s", round((row + 1) * dt, 2), flush=True)
    for arm in ARMS:
        assert session(arms[arm]).report["observations"] == spec["expected_updates"]
        final_model = session(arms[arm]).model
        save(output / f"final-{arm}.npz", **final_model.arrays())
        write(output / f"final-{arm}.json", final_model.metadata())
        assert recorded[f"model_fingerprint_{arm}"][-1] == final_model.fingerprint
    assert all(len(recorded[f"update_s_{arm}"]) == spec["expected_updates"] for arm in ARMS)
    stats = {
        "curvature_gram": np.asarray(arms["curvature"].gram),
        "curvature_rhs": np.asarray(arms["curvature"].rhs),
        "curvature_mean": np.asarray(arms["curvature"].mean),
        "candidate_gram": np.asarray(arms["candidate"].gram),
        "candidate_rhs": np.asarray(arms["candidate"].rhs),
        "candidate_mean": np.asarray(arms["candidate"].mean),
        "candidate_sensitivity": np.asarray(arms["candidate"].sensitivity),
    }
    save(output / "terminal-statistics.npz", **stats)
    data = {key: np.asarray(value) for key, value in recorded.items()}
    assert all(np.isfinite(value).all() for value in data.values() if value.dtype.kind in "biufc")
    save(output / "predictions.npz", **data)
    summary = summarize(data, tape, spec)
    write(output / "result.json", summary)
    print(name, "250 ms", {arm: summary["arms"][arm]["horizons"]["250"] for arm in ARMS}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "verify"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-sha256")
    args = parser.parse_args()
    if args.mode == "verify":
        authenticate(args.output, args.manifest_sha256)
        spec = read(args.output / "protocol.json")
        source = Path(spec["source"]["collection_path"])
        authenticate(source, spec["source"]["collection_manifest_sha256"])
        fine, coarse = npz(source / "fine.npz"), npz(source / "coarse.npz")
        assert np.array_equal(fine["states"][::5], coarse["states"])
        assert np.array_equal(fine["commands"][::5], coarse["commands"])
        for schedule, tape in zip(spec["schedules"], (fine, coarse)):
            data = npz(args.output / schedule["id"] / "predictions.npz")
            rows = np.arange(schedule["first_prediction_row"], len(tape["commands"]))
            origins = np.arange(
                schedule["first_prediction_row"],
                len(tape["commands"]) - schedule["forecast_steps"] + 1,
                schedule["common_origin_stride_rows"],
            )
            observed_states = observed(tape["states"])
            assert len(rows) == schedule["expected_updates"]
            assert len(origins) == 35
            assert np.array_equal(data["rows"], rows)
            assert np.array_equal(data["origin_rows"], origins)
            assert np.array_equal(data["one_step_truth"], observed_states[rows + 1])
            assert np.array_equal(data["conditional_truth"], np.stack([
                observed_states[row + 1 : row + schedule["forecast_steps"] + 1]
                for row in origins
            ]))
            for arm in ARMS:
                assert len(data[f"model_fingerprint_{arm}"]) == len(rows)
                assert all(len(value) == 64 for value in data[f"model_fingerprint_{arm}"])
                for endpoint, expected in (("initial", None), ("final", data[f"model_fingerprint_{arm}"][-1])):
                    prefix = args.output / schedule["id"] / f"{endpoint}-{arm}"
                    fingerprint = array_fingerprint(read(prefix.with_suffix(".json")), npz(prefix.with_suffix(".npz")))
                    if expected is not None:
                        assert fingerprint == expected
            assert summarize(data, tape, schedule) == read(args.output / schedule["id"] / "result.json")
        print("Verified paired physical metrics and timings without fitting.")
        return
    spec = read(PROTOCOL)
    source = Path(spec["source"]["collection_path"])
    authenticate(source, spec["source"]["collection_manifest_sha256"])
    bound = source_binding(spec)
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(PROTOCOL, args.output / "protocol.json")
    write(args.output / "binding.json", bound)
    started = time.monotonic()
    try:
        fine, coarse = npz(source / "fine.npz"), npz(source / "coarse.npz")
        assert np.array_equal(fine["states"][::5], coarse["states"])
        assert np.array_equal(fine["commands"][::5], coarse["commands"])
        assert np.array_equal(fine["commands"].reshape(-1, 5, 4),
                              np.broadcast_to(coarse["commands"][:, None],
                                              (len(coarse["commands"]), 5, 4)))
        deadline = time.monotonic() + spec["budget"]["maximum_wall_s"]
        for schedule, tape in zip(spec["schedules"], (fine, coarse)):
            run_schedule(args.output, schedule, tape, deadline)
        write(args.output / "report.json", {"status": "complete", "wall_s": time.monotonic() - started})
    except BaseException as error:
        write(args.output / "failure.json", {"error": repr(error), "wall_s": time.monotonic() - started})
        raise
    finally:
        print("manifest_sha256", seal(args.output), flush=True)


if __name__ == "__main__":
    main()
