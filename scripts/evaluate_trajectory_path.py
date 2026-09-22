"""Frozen causal screen for one low-rank physical path correction."""

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
from screen_trajectory_path import TrajectoryPathReadout
from verify_baseline import arrays, observed, require

from glassbox import STATE_CHANNELS, SequenceCollection, SequenceSegment
from glassbox._learner_arrays import array_fingerprint

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/harness/readout-trajectory-path-screen-v1.json"
GIT = "/opt/homebrew/Caskroom/miniconda/base/bin/git"
HORIZONS = (50, 100, 150, 200, 250)
CORRECTION_FIELDS = (
    "head",
    "r",
    "jacobian",
    "column_scale",
    "z",
    "delta",
    "trial_norms",
    "multiplier",
    "baseline_norm",
)
DIRECT_FIELDS = ("mean", "phi", "target", "penalty", "d_attitude")


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
    dirty = subprocess.check_output(
        [GIT, "-C", str(ROOT), "status", "--porcelain", "--untracked-files=normal"],
        text=True,
    ).strip()
    require(not dirty, f"uncommitted experiment source: {dirty}")
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
        "maintained learner changed",
    )
    for name in (
        "screen_cold_readout_curvature.py",
        "screen_readout_sensitivity.py",
        "screen_physical_so3.py",
    ):
        previous = subprocess.check_output(
            [
                GIT,
                "-C",
                str(ROOT),
                "show",
                f"{spec['source']['previous_readout_source_commit']}:scripts/{name}",
            ]
        )
        require(
            hashlib.sha256(previous).hexdigest() == digest(ROOT / "scripts" / name),
            "restored readout source differs",
        )
    return dict(
        commit=subprocess.check_output(
            [GIT, "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        protocol_sha256=digest(PROTOCOL),
        script_sha256=digest(Path(__file__)),
        candidate_sha256=digest(ROOT / "scripts/screen_trajectory_path.py"),
        runtime=dict(
            python=platform.python_version(),
            jax=jax.__version__,
            numpy=np.__version__,
            backend=jax.default_backend(),
            ambient_x64=bool(jax.config.x64_enabled),
        ),
    )


def source_case(spec, name):
    prior_root = Path(spec["source"]["previous_evaluation"]["path"])
    prior_result = read(prior_root / name / "result.json")
    prior = arrays(prior_root / name / "predictions.npz")
    endpoint_root = Path(spec["source"]["endpoint_screen"]["path"])
    endpoint = arrays(endpoint_root / name / "predictions.npz")
    if name.startswith("paired-quad-"):
        grid = "fine" if name.endswith("fine") else "coarse"
        tape = arrays(Path(spec["source"]["paired_recording"]["path"]) / f"{grid}.npz")
        controls = arrays(
            Path(spec["source"]["paired_evaluation"]["path"]) / grid / "predictions.npz"
        )
        full_rows, full_forecasts = (
            controls["origin_rows"],
            controls["conditional_baseline"],
        )
        opaque_id = "paired-quad-v2"
        ordered_commands = tuple(f"motor_{i} [1]" for i in range(4))
    else:
        parent = Path(spec["source"]["recordings"]["path"])
        tape = arrays(parent / "inputs" / f"{name}.npz")
        info = read(parent / name / "case.json")
        controls = arrays(
            Path(spec["source"]["previous_controls"]["path"]) / name / "predictions.npz"
        )
        full_rows, full_forecasts = (
            controls["conditional_rows"],
            controls["baseline_conditional"],
        )
        opaque_id, ordered_commands = info["opaque_id"], tuple(info["ordered_commands"])
    begin = prior_result["prefix_begin_row"]
    first = prior_result["first_prediction_row"]
    dt = prior_result["dt_s"]
    rows = prior["conditional_rows"]
    require(
        rows[0] == first and np.array_equal(rows[:2], full_rows[:2]),
        "candidate/full control origins differ",
    )
    require(
        np.array_equal(endpoint["origin_rows"], rows[:2]),
        "endpoint control origins differ",
    )
    next_row = int(rows[1])
    updates = next_row - first
    require(
        updates
        == (
            25
            if name == "paired-quad-fine"
            else 5
            if name == "paired-quad-coarse"
            else 16
        ),
        "screen update count differs",
    )
    states, issued = observed(tape["states"]), tape["commands"].astype(np.float64)
    h = round(0.25 / dt)
    require(next_row + h <= len(issued), "incomplete scoring truth")
    return dict(
        name=name,
        opaque_id=opaque_id,
        ordered_commands=ordered_commands,
        begin=begin,
        first=first,
        next_row=next_row,
        updates=updates,
        dt_s=dt,
        states=states,
        commands=issued,
        initial=arrays(prior_root / name / "initial.npz"),
        prior=prior,
        endpoint=endpoint,
        full_forecasts=full_forecasts[:2],
    )


def latency(values):
    values = np.asarray(values)
    return dict(
        first_s=float(values[0]),
        warm_median_s=float(np.median(values[1:])),
        warm_p95_s=float(np.quantile(values[1:], 0.95)),
        total_s=float(values.sum()),
    )


def case_scores(source, saved):
    origins = saved["origin_rows"]
    h = round(0.25 / source["dt_s"])
    truth = np.stack(
        [source["states"][int(row) + 1 : int(row) + h + 1] for row in origins]
    )
    result = {"one_step": {}, "origins": {}}
    old = source["prior"]["one"][: source["updates"]]
    actual = source["states"][source["first"] + 1 : source["next_row"] + 1]
    for arm, one in (("candidate", saved["one"]), ("previous", old)):
        result["one_step"][arm] = metrics(one, actual)
    for arm, forecast in (
        ("candidate", saved["conditional"]),
        ("previous", source["prior"]["conditional"][:2]),
        ("endpoint", source["endpoint"]["conditional"]),
        ("full", source["full_forecasts"]),
    ):
        result["origins"][arm] = {
            str(which): {
                str(ms): metrics(
                    forecast[
                        which,
                        round(ms / 1000 / source["dt_s"]) - 1 : round(
                            ms / 1000 / source["dt_s"]
                        ),
                    ],
                    truth[
                        which,
                        round(ms / 1000 / source["dt_s"]) - 1 : round(
                            ms / 1000 / source["dt_s"]
                        ),
                    ],
                )
                for ms in HORIZONS
            }
            for which in range(2)
        }
    return result


def audit_correction(saved):
    maximum = 0.0
    for index in range(len(saved["r"])):
        r, jacobian, scale = (
            saved["r"][index],
            saved["jacobian"][index],
            saved["column_scale"][index],
        )
        b = jacobian * scale[None]
        require(r.shape == (75,) and jacobian.shape[0] == 75, "path dimensions differ")
        dual = b @ b.T + np.eye(75)
        z = -b.T @ np.linalg.solve(dual, r)
        z *= min(1.0, 1.0 / max(np.linalg.norm(z), 1e-12))
        delta = (scale * z).reshape(saved["delta"][index].shape)
        maximum = max(
            maximum,
            float(np.max(np.abs(z - saved["z"][index]))),
            float(np.max(np.abs(delta - saved["delta"][index]))),
        )
        assert np.linalg.norm(saved["z"][index]) <= 1 + 1e-10
        trials = saved["trial_norms"][index]
        improving = np.flatnonzero(
            np.isfinite(trials) & (trials < saved["baseline_norm"][index])
        )
        expected = 0.0 if not len(improving) else (1.0, 0.5, 0.25, 0.125)[improving[0]]
        assert saved["multiplier"][index] == expected
        assert np.allclose(
            saved["head"][index],
            saved["correction_base_mean"][index] + expected * delta,
            rtol=1e-12,
            atol=1e-10,
        )
    assert maximum < 1e-8, maximum
    return maximum


def run_case(root, source, deadline):
    output = root / source["name"]
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
    fit = TrajectoryPathReadout(prefix)
    initialization_s = time.perf_counter() - started
    require(
        all(
            np.array_equal(value, source["initial"][key])
            for key, value in fit.initial_direct_model.arrays().items()
        ),
        "fresh direct initial model differs",
    )
    save(output / "initial-direct.npz", **fit.initial_direct_model.arrays())
    save(output / "initial-corrected.npz", **fit.session.model.arrays())
    write(output / "metadata.json", fit.session.model.metadata())
    recorded = {
        key: []
        for key in (
            "origin_rows",
            "conditional",
            "rows",
            "one",
            "predict_s",
            "conditional_predict_s",
            "update_s",
            "fingerprint",
            *(f"direct_{field}" for field in DIRECT_FIELDS),
            *(f"correction_{field}" for field in CORRECTION_FIELDS),
            "correction_direct_mean",
        )
    }
    initial_params = fit.initial_direct_model.params
    prefix_direct = np.concatenate(
        (
            initial_params["linear"],
            initial_params["quadratic"],
            initial_params["bias"][None],
            initial_params["w2"],
        )
    )
    corrections = [fit.initial_correction]
    direct_means = [prefix_direct]
    h, horizon = fit.session.model.history_steps, fit.horizon
    try:
        for row in range(source["first"], source["next_row"] + 1):
            if time.monotonic() > deadline:
                raise TimeoutError("frozen screen wall limit exceeded")
            past = source["states"][row - h : row + 1]
            inputs = source["commands"][row - h : row]
            if row in (source["first"], source["next_row"]):
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
                recorded["origin_rows"].append(row)
                recorded["conditional"].append(forecast)
            if row == source["next_row"]:
                break
            started = time.perf_counter()
            one = np.asarray(
                fit.session.predict(past, inputs, source["commands"][row : row + 1])
            )[0]
            recorded["predict_s"].append(time.perf_counter() - started)
            require(np.isfinite(one).all(), "nonfinite one-step forecast")
            recorded["one"].append(one)
            recorded["rows"].append(row)
            started = time.perf_counter()
            direct, correction = fit.observe(
                row, source["commands"][row], source["states"][row + 1]
            )
            snapshot = fit.session.model.arrays()
            recorded["update_s"].append(time.perf_counter() - started)
            require(
                all(np.isfinite(value).all() for value in snapshot.values()),
                "nonfinite model",
            )
            recorded["fingerprint"].append(
                array_fingerprint(fit.session.model.metadata(), snapshot)
            )
            for field, value in zip(DIRECT_FIELDS, direct):
                recorded[f"direct_{field}"].append(value)
            for field, value in zip(CORRECTION_FIELDS, correction):
                recorded[f"correction_{field}"].append(value)
            recorded["correction_direct_mean"].append(direct[0])
            direct_means.append(direct[0])
            corrections.append(correction)
        require(
            fit.session.report["observations"] == source["updates"], "missing update"
        )
    finally:
        data = {key: np.asarray(value) for key, value in recorded.items()}
        for field, values in zip(CORRECTION_FIELDS, zip(*corrections)):
            data[field] = np.asarray(values)
        data["correction_base_mean"] = np.asarray(direct_means)
        save(output / "predictions.npz", **data)
        save(output / "final.npz", **fit.session.model.arrays())
        write(
            output / "progress.json",
            dict(
                completed_updates=len(data["rows"]),
                expected_updates=source["updates"],
                initialization_s=initialization_s,
            ),
        )
    assert np.array_equal(data["origin_rows"], (source["first"], source["next_row"]))
    assert np.array_equal(data["rows"], np.arange(source["first"], source["next_row"]))
    audit = audit_correction(data)
    for field in DIRECT_FIELDS:
        np.testing.assert_array_equal(
            data[f"direct_{field}"], source["prior"][field][: source["updates"]]
        )
    result = dict(
        id=source["name"],
        dt_s=source["dt_s"],
        updates=source["updates"],
        initialization_s=initialization_s,
        timing=latency(data["update_s"]),
        accepted_corrections=int(np.count_nonzero(data["multiplier"])),
        maximum_dual_difference=audit,
        scores=case_scores(source, data),
    )
    write(output / "result.json", result)
    print(
        source["name"],
        "250 ms",
        {
            arm: [result["scores"]["origins"][arm][str(i)]["250"] for i in (0, 1)]
            for arm in ("candidate", "previous", "endpoint", "full")
        },
        "warm_ms",
        result["timing"]["warm_median_s"] * 1000,
        flush=True,
    )
    return result


def verify_case(root, source):
    output = root / source["name"]
    saved = arrays(output / "predictions.npz")
    result = read(output / "result.json")
    require(
        np.array_equal(saved["rows"], np.arange(source["first"], source["next_row"])),
        "saved update rows differ",
    )
    require(
        np.array_equal(saved["origin_rows"], (source["first"], source["next_row"])),
        "saved origins differ",
    )
    require(
        all(
            np.array_equal(value, source["initial"][key])
            for key, value in arrays(output / "initial-direct.npz").items()
        ),
        "saved prefix direct model differs",
    )
    require(result["scores"] == case_scores(source, saved), "physical scores differ")
    require(result["timing"] == latency(saved["update_s"]), "timing differs")
    require(
        result["maximum_dual_difference"] == audit_correction(saved),
        "dual audit differs",
    )
    require(
        result["accepted_corrections"] == int(np.count_nonzero(saved["multiplier"])),
        "accepted count differs",
    )
    for field in DIRECT_FIELDS:
        np.testing.assert_array_equal(
            saved[f"direct_{field}"], source["prior"][field][: source["updates"]]
        )
    final = arrays(output / "final.npz")
    require(
        array_fingerprint(read(output / "metadata.json"), final)
        == saved["fingerprint"][-1],
        "final fingerprint differs",
    )
    return result


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
                authenticate(Path(value["path"]), value["manifest_sha256"])
        results = [
            verify_case(args.output, source_case(spec, name)) for name in spec["cases"]
        ]
        require(
            sum(value["updates"] for value in results) == 126,
            "total update count differs",
        )
        print("Verified saved physical scores and dual corrections without fitting.")
        return
    spec = read(PROTOCOL)
    for value in spec["source"].values():
        if isinstance(value, dict) and "path" in value:
            authenticate(Path(value["path"]), value["manifest_sha256"])
    bound = source_binding(spec)
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(PROTOCOL, args.output / "protocol.json")
    write(args.output / "binding.json", bound)
    deadline = time.monotonic() + 600
    try:
        results = [
            run_case(args.output, source_case(spec, name), deadline)
            for name in spec["cases"]
        ]
        write(
            args.output / "report.json",
            dict(
                status="complete",
                fits=len(results),
                candidate_updates=sum(value["updates"] for value in results),
            ),
        )
    except BaseException as error:
        write(args.output / "failure.json", dict(error=repr(error)))
        raise
    finally:
        print(
            "manifest_sha256",
            seal(args.output, "glassbox-trajectory-path-screen-v1"),
            flush=True,
        )


if __name__ == "__main__":
    main()
