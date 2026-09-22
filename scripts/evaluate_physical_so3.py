"""Run and verify one frozen cold-start physical SO(3) readout experiment."""

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import time
from pathlib import Path

import jax
import numpy as np
from collect_throw import authenticate, seal
from evaluate_online import metrics
from screen_physical_so3 import ATTITUDE_ETA, PhysicalSO3Readout
from screen_readout_sensitivity import ETA, numpy_jacobian
from verify_baseline import arrays, observed, require

from glassbox import STATE_CHANNELS, SequenceCollection, SequenceSegment
from glassbox._learner_arrays import array_fingerprint

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/harness/readout-physical-so3-v1.json"
GIT = "/opt/homebrew/Caskroom/miniconda/base/bin/git"
CONTROLS = ("baseline", "curvature", "sensitivity")
ARMS = CONTROLS + ("candidate",)
HORIZONS = (50, 100, 150, 200, 250)
FIELDS = ("mean", "phi", "target", "penalty", "d_attitude")


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def save(path, **values):
    temporary = Path(str(path) + ".partial")
    with temporary.open("wb") as target:
        np.savez_compressed(target, **values)
    temporary.replace(path)


def source_binding(spec):
    require(not jax.config.x64_enabled, "ambient float64 differs")
    dirty = subprocess.check_output(
        [GIT, "-C", str(ROOT), "status", "--porcelain", "--untracked-files=normal"],
        text=True,
    ).strip()
    require(not dirty, f"scientific source is not committed: {dirty}")
    require(
        subprocess.run(
            [
                GIT,
                "-C",
                str(ROOT),
                "diff",
                "--quiet",
                spec["source"]["maintained_learner_commit"],
                "HEAD",
                "--",
                "src/glassbox",
            ],
            check=False,
        ).returncode
        == 0,
        "maintained learner differs from authenticated controls",
    )
    for name in ("screen_cold_readout_curvature.py", "screen_readout_sensitivity.py"):
        prior = subprocess.check_output(
            [
                GIT,
                "-C",
                str(ROOT),
                "show",
                f"{spec['source']['previous_readout_source_commit']}:scripts/{name}",
            ]
        )
        require(
            hashlib.sha256(prior).hexdigest() == digest(ROOT / "scripts" / name),
            "prior readout control source differs",
        )
    return dict(
        source_commit=subprocess.check_output(
            [GIT, "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        script_sha256=digest(Path(__file__)),
        candidate_sha256=digest(ROOT / "scripts/screen_physical_so3.py"),
        protocol_sha256=digest(PROTOCOL),
        runtime=dict(
            python=platform.python_version(),
            jax=jax.__version__,
            numpy=np.__version__,
            backend=jax.default_backend(),
            ambient_x64=bool(jax.config.x64_enabled),
        ),
    )


def source_case(spec, name):
    old = not name.startswith("paired-quad-")
    if old:
        case = Path(spec["source"]["previous_evaluation"]["path"]) / name
        info = read(case / "result.json")
        saved = arrays(case / "predictions.npz")
        tape = arrays(
            Path(spec["source"]["recordings"]["path"]) / "inputs" / f"{name}.npz"
        )
        begin, first, dt, stop = (
            info[key] for key in ("begin", "first", "dt_s", "stop")
        )
        truth = saved["truth"]
        rows = saved["row"]
        origin_rows = saved["conditional_rows"]
        conditional_truth = saved["conditional_truth"]
        initial = arrays(case / "initial.npz")
        metadata = info["initial_model_metadata"]
        control = {
            arm: dict(
                one=saved["candidate" if arm == "sensitivity" else arm],
                conditional=saved[
                    ("candidate" if arm == "sensitivity" else arm) + "_conditional"
                ],
            )
            for arm in CONTROLS
        }
        direct = {field: saved[f"candidate_{field}"] for field in FIELDS[:-1]}
        old_timing = {
            arm: saved[("candidate" if arm == "sensitivity" else arm) + "_update_s"]
            for arm in CONTROLS
        }
        ordered_commands = tuple(
            info["initial_model_metadata"].get("ordered_commands", ())
        )
        if not ordered_commands:
            source_info = read(
                Path(spec["source"]["recordings"]["path"]) / name / "case.json"
            )
            ordered_commands = tuple(source_info["ordered_commands"])
        family = info["family"]
        opaque_id = read(
            Path(spec["source"]["recordings"]["path"]) / name / "case.json"
        )["opaque_id"]
    else:
        schedule = "fine" if name.endswith("fine") else "coarse"
        case = Path(spec["source"]["paired_evaluation"]["path"]) / schedule
        saved = arrays(case / "predictions.npz")
        tape = arrays(
            Path(spec["source"]["paired_recording"]["path"]) / f"{schedule}.npz"
        )
        schedule_spec = next(
            value
            for value in read(
                Path(spec["source"]["paired_evaluation"]["path"]) / "protocol.json"
            )["schedules"]
            if value["id"] == schedule
        )
        begin, first, dt, stop = (
            schedule_spec["prefix_begin_row"],
            schedule_spec["first_prediction_row"],
            schedule_spec["dt_s"],
            len(tape["commands"]),
        )
        truth, rows, origin_rows = (
            saved["one_step_truth"],
            saved["rows"],
            saved["origin_rows"],
        )
        conditional_truth = saved["conditional_truth"]
        initial = arrays(case / "initial-baseline.npz")
        metadata = read(case / "initial-baseline.json")
        control = {
            arm: dict(
                one=saved[f"one_step_{'candidate' if arm == 'sensitivity' else arm}"],
                conditional=saved[
                    f"conditional_{'candidate' if arm == 'sensitivity' else arm}"
                ],
            )
            for arm in CONTROLS
        }
        direct = {field: saved[f"direct_candidate_{field}"] for field in FIELDS[:-1]}
        old_timing = {
            arm: saved[f"update_s_{'candidate' if arm == 'sensitivity' else arm}"]
            for arm in CONTROLS
        }
        ordered_commands = tuple(f"motor_{index} [1]" for index in range(4))
        family, opaque_id = "quad", "paired-quad-v2"
    states, commands = observed(tape["states"]), tape["commands"].astype(np.float64)
    require(np.array_equal(rows, np.arange(first, stop)), "source row schedule differs")
    require(np.array_equal(truth, states[rows + 1]), "source one-step truth differs")
    horizon = round(0.25 / dt)
    require(
        np.array_equal(
            conditional_truth,
            np.stack([states[row + 1 : row + horizon + 1] for row in origin_rows]),
        ),
        "source conditional truth differs",
    )
    return dict(
        name=name,
        family=family,
        begin=begin,
        first=first,
        stop=stop,
        dt_s=dt,
        opaque_id=opaque_id,
        ordered_commands=ordered_commands,
        states=states,
        commands=commands,
        rows=rows,
        origin_rows=origin_rows,
        truth=truth,
        conditional_truth=conditional_truth,
        initial=initial,
        metadata=metadata,
        controls=control,
        direct=direct,
        old_timing=old_timing,
    )


def latency(values):
    values = np.asarray(values)
    return dict(
        first_s=float(values[0]),
        warm_median_s=float(np.median(values[1:])),
        warm_p95_s=float(np.quantile(values[1:], 0.95)),
        total_s=float(values.sum()),
    )


def case_scores(source, candidate):
    dt = source["dt_s"]
    result = {}
    for arm in ARMS:
        one = candidate["one"] if arm == "candidate" else source["controls"][arm]["one"]
        conditional = (
            candidate["conditional"]
            if arm == "candidate"
            else source["controls"][arm]["conditional"]
        )
        result[arm] = dict(
            one_step=metrics(one, source["truth"]),
            horizons={
                str(ms): metrics(
                    conditional[:, round(ms / 1000 / dt) - 1],
                    source["conditional_truth"][:, round(ms / 1000 / dt) - 1],
                )
                for ms in HORIZONS
            },
            first16=metrics(one[:16], source["truth"][:16]),
            after16=metrics(one[16:], source["truth"][16:]),
            first100=metrics(one[:100], source["truth"][:100]),
            after100=metrics(one[100:], source["truth"][100:]),
        )
        if source["name"] == "quad-change":
            # Original source tape timestamps define the configuration change.
            tape = arrays(
                Path(read(PROTOCOL)["source"]["recordings"]["path"])
                / "inputs"
                / f"{source['name']}.npz"
            )
            split = int(np.searchsorted(tape["time_s"][source["rows"]], 4.0))
            result[arm]["pre4s"] = metrics(one[:split], source["truth"][:split])
            result[arm]["post4s"] = metrics(one[split:], source["truth"][split:])
    return result


def audit_solve(source, saved):
    initial, metadata = source["initial"], source["metadata"]
    mean0 = np.concatenate(
        (
            initial["param_linear"],
            initial["param_quadratic"],
            initial["param_bias"][None],
            initial["param_w2"],
        )
    )
    ridge = 0.01 * round(0.25 / source["dt_s"])
    size = len(mean0)
    gram = ridge * np.eye(size)
    rhs = ridge * mean0
    motion = np.zeros((size, size))
    attitude = np.zeros((size, size))
    maximum = 0.0
    for index, (phi, target, penalty, d_attitude, mean) in enumerate(
        zip(
            saved["phi"],
            saved["target"],
            saved["penalty"],
            saved["d_attitude"],
            saved["mean"],
        )
    ):
        np.testing.assert_array_equal(phi, source["direct"]["phi"][index])
        np.testing.assert_array_equal(target, source["direct"]["target"][index])
        np.testing.assert_array_equal(penalty, source["direct"]["penalty"][index])
        d_motion = numpy_jacobian(phi, initial, metadata["delay_steps"])
        gram += np.outer(phi, phi)
        rhs += np.outer(phi, target)
        motion += d_motion @ d_motion.T
        attitude += d_attitude @ d_attitude.T
        system = gram + np.diag(penalty) + ETA * motion + ATTITUDE_ETA * attitude
        residual = system @ mean - rhs
        denominator = np.abs(system) @ np.abs(mean) + np.abs(rhs) + 1e-12
        maximum = max(maximum, float(np.max(np.abs(residual) / denominator)))
    assert maximum <= 1e-10, maximum
    for key, value in (
        ("gram", gram),
        ("rhs", rhs),
        ("motion", motion),
        ("attitude", attitude),
        ("mean", saved["mean"][-1]),
    ):
        np.testing.assert_allclose(
            saved[f"terminal_{key}"], value, rtol=2e-11, atol=2e-8
        )
    return maximum


def run_case(spec, output, source, deadline):
    output.mkdir()
    prefix = SequenceCollection(
        (
            SequenceSegment(
                source["opaque_id"],
                "prefix",
                source["states"][source["begin"] : source["first"] + 1],
                source["commands"][source["begin"] : source["first"]],
                source["dt_s"],
                source["begin"],
            ),
        ),
        source["opaque_id"],
        STATE_CHANNELS,
        source["ordered_commands"],
    )
    started = time.perf_counter()
    fit = PhysicalSO3Readout(prefix)
    initialization_s = time.perf_counter() - started
    initial_model = fit.session.model
    require(
        all(
            np.array_equal(value, source["initial"][key])
            for key, value in initial_model.arrays().items()
        ),
        "candidate fresh initial model differs from prior controls",
    )
    save(output / "initial.npz", **initial_model.arrays())
    write(output / "initial.json", initial_model.metadata())
    require(
        ETA == spec["penalty"]["eta_motion"]
        and ATTITUDE_ETA == spec["penalty"]["eta_attitude_rad2"],
        "strength differs from frozen protocol",
    )
    h, horizon = initial_model.history_steps, round(0.25 / source["dt_s"])
    origins = set(int(value) for value in source["origin_rows"])
    recorded = {
        key: []
        for key in (
            "rows",
            "one",
            "conditional_rows",
            "conditional",
            "predict_s",
            "conditional_predict_s",
            "update_s",
            "fingerprint",
            *FIELDS,
        )
    }
    try:
        for row in source["rows"]:
            if time.monotonic() > deadline:
                raise TimeoutError("frozen wall limit exceeded")
            row = int(row)
            past = source["states"][row - h : row + 1]
            inputs = source["commands"][row - h : row]
            started = time.perf_counter()
            one = np.asarray(
                fit.session.predict(past, inputs, source["commands"][row : row + 1])
            )[0]
            recorded["predict_s"].append(time.perf_counter() - started)
            require(np.isfinite(one).all(), "nonfinite one-step prediction")
            recorded["rows"].append(row)
            recorded["one"].append(one)
            if row in origins:
                started = time.perf_counter()
                forecast = np.asarray(
                    fit.session.predict(
                        past, inputs, source["commands"][row : row + horizon]
                    )
                )
                recorded["conditional_predict_s"].append(time.perf_counter() - started)
                require(
                    forecast.shape == (horizon, 15) and np.isfinite(forecast).all(),
                    "nonfinite conditional forecast",
                )
                recorded["conditional_rows"].append(row)
                recorded["conditional"].append(forecast)
            started = time.perf_counter()
            values = fit.observe(
                row, source["commands"][row], source["states"][row + 1]
            )
            model = fit.session.model
            snapshot = model.arrays()
            recorded["update_s"].append(time.perf_counter() - started)
            require(
                all(np.isfinite(value).all() for value in snapshot.values()),
                "nonfinite model",
            )
            recorded["fingerprint"].append(
                array_fingerprint(model.metadata(), snapshot)
            )
            for key, value in zip(FIELDS, values):
                recorded[key].append(value)
        require(
            fit.session.report["observations"] == len(source["rows"]),
            "missing observation",
        )
    finally:
        data = {key: np.asarray(value) for key, value in recorded.items()}
        if len(data["mean"]):
            data.update(
                terminal_gram=np.asarray(fit.gram),
                terminal_rhs=np.asarray(fit.rhs),
                terminal_motion=np.asarray(fit.sensitivity),
                terminal_attitude=np.asarray(fit.attitude_sensitivity),
                terminal_mean=np.asarray(fit.mean),
            )
        save(output / "predictions.npz", **data)
        final_model = fit.session.model
        save(output / "final.npz", **final_model.arrays())
        write(output / "final.json", final_model.metadata())
        write(
            output / "progress.json",
            dict(
                completed_updates=len(data["mean"]),
                expected_updates=len(source["rows"]),
                initialization_s=initialization_s,
            ),
        )
    assert np.array_equal(data["rows"], source["rows"])
    assert np.array_equal(data["conditional_rows"], source["origin_rows"])
    maximum = audit_solve(source, data)
    result = dict(
        id=source["name"],
        family=source["family"],
        dt_s=source["dt_s"],
        updates=len(source["rows"]),
        conditional_origins=len(source["origin_rows"]),
        prefix_begin_row=source["begin"],
        first_prediction_row=source["first"],
        initialization_s=initialization_s,
        candidate_timing=latency(data["update_s"]),
        historical_control_timing={
            arm: latency(source["old_timing"][arm]) for arm in CONTROLS
        },
        maximum_solve_backward_error=maximum,
        scores=case_scores(source, data),
    )
    write(output / "result.json", result)
    print(
        source["name"],
        "250 ms",
        {arm: result["scores"][arm]["horizons"]["250"] for arm in ARMS},
        flush=True,
    )
    return result


def geometric(values):
    return float(np.exp(np.mean(np.log(np.maximum(1e-12, values)))))


def aggregate(results):
    families = {}
    for family in ("fixedwing", "quad"):
        selected = [value for value in results if value["family"] == family]
        families[family] = {}
        for control in CONTROLS:
            families[family][control] = {}
            for horizon in ("one_step", *(str(ms) for ms in HORIZONS)):
                label = "horizons" if horizon != "one_step" else None
                families[family][control][horizon] = {
                    channel: geometric(
                        [
                            (
                                value["scores"]["candidate"][label][horizon]
                                if label
                                else value["scores"]["candidate"][horizon]
                            )[channel]
                            / max(
                                1e-9,
                                (
                                    value["scores"][control][label][horizon]
                                    if label
                                    else value["scores"][control][horizon]
                                )[channel],
                            )
                            for value in selected
                        ]
                    )
                    for channel in (
                        "velocity_rmse_m_s",
                        "body_rate_rmse_rad_s",
                        "orientation_rmse_rad",
                    )
                }
    equal_family = {
        control: {
            horizon: {
                channel: geometric(
                    [families[family][control][horizon][channel] for family in families]
                )
                for channel in (
                    "velocity_rmse_m_s",
                    "body_rate_rmse_rad_s",
                    "orientation_rmse_rad",
                )
            }
            for horizon in ("one_step", *(str(ms) for ms in HORIZONS))
        }
        for control in CONTROLS
    }
    return dict(families=families, equal_family=equal_family)


def verify_case(root, source):
    case = root / source["name"]
    data = arrays(case / "predictions.npz")
    require(np.array_equal(data["rows"], source["rows"]), "candidate rows differ")
    require(
        np.array_equal(data["conditional_rows"], source["origin_rows"]),
        "candidate origins differ",
    )
    require(len(data["one"]) == len(source["truth"]), "prediction count differs")
    initial = arrays(case / "initial.npz")
    require(
        all(
            np.array_equal(value, source["initial"][key])
            for key, value in initial.items()
        ),
        "saved initial model differs from controls",
    )
    final = arrays(case / "final.npz")
    require(
        array_fingerprint(read(case / "final.json"), final) == data["fingerprint"][-1],
        "final fingerprint differs",
    )
    maximum = audit_solve(source, data)
    expected = read(case / "result.json")
    require(expected["maximum_solve_backward_error"] == maximum, "solve audit differs")
    require(expected["scores"] == case_scores(source, data), "physical metrics differ")
    require(expected["candidate_timing"] == latency(data["update_s"]), "timing differs")
    require(
        expected["updates"] == len(source["rows"])
        and expected["conditional_origins"] == len(source["origin_rows"]),
        "counts differ",
    )
    return expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "verify"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-sha256")
    args = parser.parse_args()
    if args.mode == "verify":
        authenticate(args.output, args.manifest_sha256)
        spec = read(args.output / "protocol.json")
        for value in spec["source"].values():
            if isinstance(value, dict) and "path" in value:
                authenticate(value["path"], value["manifest_sha256"])
        results = [
            verify_case(args.output, source_case(spec, name)) for name in spec["cases"]
        ]
        require(
            aggregate(results) == read(args.output / "aggregate.json"),
            "aggregate differs",
        )
        print(
            "Verified all saved forecasts, physical metrics, timings and direct solves without fitting."
        )
        return
    spec = read(PROTOCOL)
    for value in spec["source"].values():
        if isinstance(value, dict) and "path" in value:
            authenticate(value["path"], value["manifest_sha256"])
    bound = source_binding(spec)
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(PROTOCOL, args.output / "protocol.json")
    write(args.output / "binding.json", bound)
    deadline = time.monotonic() + spec["budget"]["maximum_wall_s"]
    try:
        results = [
            run_case(spec, args.output / name, source_case(spec, name), deadline)
            for name in spec["cases"]
        ]
        write(args.output / "aggregate.json", aggregate(results))
        write(
            args.output / "report.json",
            {
                "status": "complete",
                "candidate_fits": len(results),
                "candidate_updates": sum(value["updates"] for value in results),
            },
        )
    except BaseException as error:
        write(args.output / "failure.json", {"error": repr(error)})
        raise
    finally:
        print(
            "manifest_sha256",
            seal(args.output, "glassbox-readout-physical-so3-v1"),
            flush=True,
        )


if __name__ == "__main__":
    main()
