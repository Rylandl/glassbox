"""Real-flight, offline conditional forecasts with generic temporal structure.

The held-out Melon recordings are evaluated only in confirmation. Future motor
speeds condition each forecast; future states/attitudes never become features.
Released signals were processed offline with zero-phase filters upstream.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
import zipfile
from pathlib import Path
from statistics import NormalDist

import jax
import numpy as np

from glassbox.core.data import load_trajectory_npz
from glassbox.core.geometry import quaternion_to_rotation_matrices
from glassbox.experimental.error_calibration import (
    PredictionContract,
    ResidualSamples,
    conformal_quantile,
    fit_error_calibration,
)
from glassbox.experimental.temporal import forecast_windows
from glassbox.experimental.transition_gp import TransitionSamples, fit_transition_gp
from glassbox.io.nanodrone_reference import BENCHMARK_COMMIT, NanoDroneBenchmarkAdapter

MODES = {
    "absolute": ("absolute", ()),
    "increment": ("increment", ()),
    "history": ("increment", (5, 10)),
}
ALLOCATIONS = {
    "more-evidence": {"train": 192, "scale": 192, "calibration": 192},
    "more-model": {"train": 384, "scale": 96, "calibration": 96},
}
OUTPUT_NAMES = (
    "velocity_x [m/s]",
    "velocity_y [m/s]",
    "velocity_z [m/s]",
    "rate_x [rad/s]",
    "rate_y [rad/s]",
    "rate_z [rad/s]",
)
HORIZONS = (1, 10, 25, 50)


def feature_names(mode, horizon):
    actuators = tuple(f"measured_actuation_{i} [rad/s]" for i in range(4))
    names = [*OUTPUT_NAMES, *actuators]
    names.extend(
        f"rotation_body_to_world_{i}{j} [1]" for i in range(3) for j in range(3)
    )
    names.extend(
        f"future_bin_{b}/{name}" for b in range(min(10, horizon)) for name in actuators
    )
    for lag in MODES[mode][1]:
        names.extend(
            f"past_minus_current_{lag * 10}ms/{name}"
            for name in (*OUTPUT_NAMES, *actuators)
        )
    return tuple(names)


def ridge_prediction(train, test, path):
    """Generic linear increment baseline, same windows/features as its GP."""
    x = train["features"]
    center, scale = x.mean(axis=0), x.std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    z = (x - center) / scale
    delta = train["next_states"] - train["states"]
    ymean, yscale = delta.mean(axis=0), delta.std(axis=0)
    yscale = np.where(yscale > 1e-8, yscale, 1.0)
    weights = np.linalg.solve(
        z.T @ z + np.eye(z.shape[1]), z.T @ ((delta - ymean) / yscale)
    )
    np.savez_compressed(
        path,
        feature_mean=center,
        feature_scale=scale,
        target_mean=ymean,
        target_scale=yscale,
        weights=weights,
        ridge_penalty=1.0,
    )
    return (
        test["states"]
        + (((test["features"] - center) / scale) @ weights) * yscale
        + ymean
    )


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_corpus(corpus, phase):
    records = []
    for path in sorted((corpus / "canonical").rglob("*.npz")):
        if phase == "development" and path.parent.name == "test":
            continue
        flight = load_trajectory_npz(path)
        raw = corpus / "raw" / flight.provenance["benchmark"]["relative_path"]
        inspection = NanoDroneBenchmarkAdapter().inspect(raw)
        if phase == "development":
            if flight.labels["replicate"] > 2:
                continue
            role = "train" if flight.labels["replicate"] == 1 else "test"
        else:
            role = (
                "test"
                if flight.labels["benchmark_split"] == "test"
                else {1: "train", 2: "train", 3: "scale", 4: "calibration"}[
                    flight.labels["replicate"]
                ]
            )
        # Decode the adapter's squared-speed representation. The generic
        # learner receives measured speeds, never a prescribed thrust feature.
        speed = np.sqrt(flight.controls) * 2500.0
        records.append(
            {
                "name": path.stem,
                "flight": flight,
                "role": role,
                "inspection": inspection,
                "states": np.concatenate(
                    (flight.states[:, 3:6], flight.states[:, 10:13]), axis=1
                ),
                "controls": speed,
                "context": quaternion_to_rotation_matrices(
                    flight.states[:, 6:10]
                ).reshape(-1, 9),
            }
        )
    return records


def choose_anchors(record, count, seed):
    eligible = np.arange(10, len(record["states"]) - max(HORIZONS))
    # Role/flight-local permutations retain smaller allocations as prefixes.
    identity = int(hashlib.sha256(record["name"].encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(np.random.SeedSequence([seed, identity]))
    if count is None:
        return eligible[::10]
    return rng.permutation(eligible)[:count]


def assemble(records, role, budget, seed, horizon, mode):
    selected = [r for r in records if r["role"] == role]
    if budget is not None and budget % len(selected):
        raise ValueError("budget must divide evenly among recordings")
    per_flight = None if budget is None else budget // len(selected)
    arrays = {
        "states": [],
        "commands": [],
        "next_states": [],
        "context": [],
        "anchors": [],
        "groups": [],
        "ids": [],
        "trend": [],
        "full_initial": [],
        "full_controls": [],
    }
    usage = []
    for record in selected:
        anchors = choose_anchors(record, per_flight, seed)
        windows = forecast_windows(
            record["states"],
            record["controls"],
            recording_id=record["name"],
            dt_s=0.01,
            horizon_steps=horizon,
            anchors=anchors,
            context=record["context"],
            history_lags=MODES[mode][1],
            control_bins=min(10, horizon),
        )
        for name in ("states", "commands", "next_states", "context"):
            arrays[name].append(getattr(windows.samples, name))
        arrays["anchors"].append(anchors)
        arrays["groups"].append(np.full(len(anchors), record["name"]))
        arrays["ids"].append(np.asarray(windows.sample_ids))
        arrays["trend"].append(
            record["states"][anchors]
            + horizon / 5 * (record["states"][anchors] - record["states"][anchors - 5])
        )
        arrays["full_initial"].append(record["flight"].states[anchors])
        arrays["full_controls"].append(
            record["flight"].controls[anchors[:, None] + np.arange(horizon)]
        )
        state_rows = np.unique(
            np.concatenate(
                [anchors, anchors + horizon, *[anchors - lag for lag in MODES[mode][1]]]
            )
        )
        control_rows = np.unique(
            np.concatenate(
                [
                    (anchors[:, None] + np.arange(horizon)).ravel(),
                    *[anchors - lag for lag in MODES[mode][1]],
                ]
            )
        )
        usage.append(
            {
                "recording": record["name"],
                "windows": len(anchors),
                "unique_state_rows": len(state_rows),
                "unique_control_intervals": len(control_rows),
                "control_interval_union_s": len(control_rows) * 0.01,
                "available_recording_s": float(record["flight"].time_s[-1]),
            }
        )
    result = {name: np.concatenate(parts) for name, parts in arrays.items()}
    result["features"] = np.concatenate(
        (result["states"], result["commands"], result["context"]), axis=1
    )
    return result, usage


def prediction(model, data):
    # Chunk queries to bound intermediate GP memory at high feature dimension.
    outputs = []
    for start in range(0, len(data["states"]), 128):
        sl = slice(start, start + 128)
        outputs.append(
            tuple(
                np.asarray(x)
                for x in model.predict(
                    data["states"][sl],
                    data["commands"][sl],
                    context=data["context"][sl],
                )
            )
        )
    return tuple(np.concatenate([x[i] for x in outputs]) for i in range(3))


def score(mean, data, bounds=None, support=None, support_threshold=None):
    masks = {
        "all": np.ones(len(mean), bool),
        **{name: data["groups"] == name for name in np.unique(data["groups"])},
    }
    if support is not None:
        masks.update(
            {
                "nearer-support": support <= support_threshold,
                "farther-support": support > support_threshold,
            }
        )
    result = {}
    for name, mask in masks.items():
        if not mask.any():
            continue
        error = mean[mask] - data["next_states"][mask]
        metrics = {
            "count": int(mask.sum()),
            "velocity_rmse_m_s": float(
                np.sqrt(np.mean(np.sum(error[:, :3] ** 2, axis=1)))
            ),
            "rate_rmse_rad_s": float(
                np.sqrt(np.mean(np.sum(error[:, 3:] ** 2, axis=1)))
            ),
            "per_channel_rmse": np.sqrt(np.mean(error**2, axis=0)).tolist(),
        }
        if bounds is not None:
            contained = (data["next_states"][mask] >= bounds[0][mask]) & (
                data["next_states"][mask] <= bounds[1][mask]
            )
            width = bounds[1][mask] - bounds[0][mask]
            metrics.update(
                joint_coverage=float(np.mean(np.all(contained, axis=1))),
                marginal_coverage=np.mean(contained, axis=0).tolist(),
                velocity_mean_width_m_s=float(np.mean(width[:, :3])),
                rate_mean_width_rad_s=float(np.mean(width[:, 3:])),
            )
        result[name] = metrics
    return result


def nearest(model, data):
    q = (data["features"] - np.asarray(model.feature_mean)) / np.asarray(
        model.feature_scale
    )
    ref = np.asarray(model.features)
    return np.sqrt(
        np.maximum(
            np.sum(q * q, axis=1)[:, None]
            + np.sum(ref * ref, axis=1)[None]
            - 2 * q @ ref.T,
            0,
        ).min(axis=1)
    )


def structured_predictions(belief, data, records, horizon):
    # These are observed actuator inputs, not a declared command interface.
    # Preserve the contract while using Glassbox's existing logged-input replay.
    from glassbox.core.metrics import predict_windows

    output = np.empty_like(data["states"])
    for group in np.unique(data["groups"]):
        flight = next(r["flight"] for r in records if r["name"] == group)
        if flight.spec.prediction_spec() != belief.model.input_spec:
            raise ValueError("structured replay input contract changed")
        replay = predict_windows(
            belief.model.params, flight, horizon_steps=horizon, stride=10
        )
        mask = data["groups"] == group
        indices = data["anchors"][mask] // 10
        expected = replay.target[indices, -1][:, [3, 4, 5, 10, 11, 12]]
        np.testing.assert_array_equal(expected, data["next_states"][mask])
        output[mask] = replay.predicted[indices, -1][:, [3, 4, 5, 10, 11, 12]]
    return output


def run(output, corpus, phase, seeds, steps, structured):
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    sources = [
        Path(__file__).resolve(),
        *[
            root / "src/glassbox/experimental" / name
            for name in ("temporal.py", "transition_gp.py", "error_calibration.py")
        ],
        root / "tests/test_temporal_transition.py",
    ]
    hashes = {}
    with zipfile.ZipFile(
        output / "executed-sources.zip", "x", zipfile.ZIP_DEFLATED
    ) as archive:
        for path in sources:
            name = str(path.relative_to(root))
            archive.write(path, name)
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(output / "sources.json", hashes)
    records = load_corpus(corpus, phase)
    plan = {
        "phase": phase,
        "seeds": seeds,
        "horizon_steps": HORIZONS,
        "sample_period_s": 0.01,
        "modes": MODES,
        "allocations": ALLOCATIONS,
        "kernel": "rq",
        "fit_steps": steps,
        "restarts": 2,
        "control_bins": "min(10, horizon_steps)",
        "ridge_reference": "linear state increment, unit ridge penalty, same input features and mean-fit windows as its GP",
        "test_stride_s": 0.1,
        "split": {
            role: [r["name"] for r in records if r["role"] == role]
            for role in ("train", "scale", "calibration", "test")
        },
        "source_commit": BENCHMARK_COMMIT,
        "outputs": OUTPUT_NAMES,
        "prediction_task": "direct horizon forecasts of velocity and body rate conditional on observed future actuation; no future state or attitude input",
        "control_encoding": "current measured speeds plus 10 sequential means over the supplied future actuation window; no squared-speed prior",
        "history": "state and control differences from 50 and 100 ms before the origin",
        "model_scope": "six Euclidean outputs, not a complete recursively executable rigid-body model",
        "coverage": "empirical six-output joint coverage; correlated samples, only three calibration flights, and maneuver shift invalidate an IID 95% guarantee",
        "physical_bounds": "no hard bounds inferred from finite telemetry",
        "upstream_processing": "100 Hz offline retiming, cross-correlation alignment, zero-phase Butterworth filtering; not causal raw sensor data",
        "budget_unit": "selected windows, not independent samples or flight seconds; unique supporting rows and input interval unions reported",
        "structured_reference": "same six mean-fit flights; existing full-state multi-step objective with its own larger window budget and full actuation sequence; contextual reference, not a matched label-budget comparison",
    }
    write_json(output / "plan.json", plan)
    write_json(
        output / "environment.json",
        {
            "python": sys.version,
            "jax": jax.__version__,
            "numpy": np.__version__,
            "x64": bool(jax.config.jax_enable_x64),
            "platform": platform.platform(),
        },
    )
    write_json(
        output / "data-audit.json",
        {
            "recordings": [{"role": r["role"], **r["inspection"]} for r in records],
            "verified_recordings": len(records),
            "raw_filter_provenance": records[0]["flight"].provenance[
                "upstream_preprocessing"
            ],
        },
    )
    belief = None
    if structured is not None:
        from glassbox import DynamicsBelief

        belief = DynamicsBelief.load(structured)
        write_json(
            output / "structured-reference.json",
            {
                "path": str(structured.resolve()),
                "sha256": hashlib.sha256(structured.read_bytes()).hexdigest(),
            },
        )
    reports = []
    baseline_cache = {}
    for seed in seeds:
        for allocation, budgets in ALLOCATIONS.items():
            for horizon in HORIZONS:
                for mode, (mean_mode, _) in MODES.items():
                    start = time.monotonic()
                    folder = output / f"{allocation}-{mode}-h{horizon}-seed{seed}"
                    folder.mkdir()
                    roles = (
                        ("train", "test")
                        if phase == "development"
                        else ("train", "scale", "calibration", "test")
                    )
                    data = {}
                    usage = {}
                    for role in roles:
                        data[role], usage[role] = assemble(
                            records, role, budgets.get(role), seed, horizon, mode
                        )
                    model = fit_transition_gp(
                        TransitionSamples(
                            data["train"]["states"],
                            data["train"]["commands"],
                            data["train"]["next_states"],
                            horizon * 0.01,
                            data["train"]["context"],
                        ),
                        kernel="rq",
                        mean_mode=mean_mode,
                        steps=steps,
                        restarts=2,
                    )
                    model.save(folder / "model.npz")
                    report = {
                        "seed": seed,
                        "allocation": allocation,
                        "horizon_s": horizon * 0.01,
                        "horizon_steps": horizon,
                        "mode": mode,
                        "usage": usage,
                        "methods": {},
                    }
                    mean, fvar, ovar = prediction(model, data["test"])
                    if not np.all(np.isfinite(mean)):
                        raise FloatingPointError("nonfinite test forecast")
                    predictions = {
                        "mean": mean,
                        "function_variance": fvar,
                        "observation_variance": ovar,
                    }
                    test = data["test"]
                    report["methods"]["hold"] = score(test["states"], test)
                    report["methods"]["trend"] = score(test["trend"], test)
                    predictions["hold"] = test["states"]
                    predictions["trend"] = test["trend"]
                    predictions["ridge"] = ridge_prediction(
                        data["train"], test, folder / "ridge.npz"
                    )
                    report["methods"]["ridge"] = score(predictions["ridge"], test)
                    if belief is not None:
                        if horizon not in baseline_cache:
                            baseline_cache[horizon] = structured_predictions(
                                belief, test, records, horizon
                            )
                        predictions["structured"] = baseline_cache[horizon]
                        report["methods"]["structured"] = score(
                            predictions["structured"], test
                        )
                    z = NormalDist().inv_cdf(1 - 0.05 / (2 * 6))
                    gp_bounds = mean - z * np.sqrt(ovar), mean + z * np.sqrt(ovar)
                    report["methods"]["gp"] = score(mean, test, gp_bounds)
                    predictions["gp_lower"], predictions["gp_upper"] = gp_bounds
                    if phase == "confirmation":
                        contract = PredictionContract(
                            model.fingerprint(),
                            feature_names(mode, horizon),
                            OUTPUT_NAMES,
                            horizon * 0.01,
                        )
                        residuals = []
                        for role in ("scale", "calibration"):
                            mu, _, _ = prediction(model, data[role])
                            data[role]["mean"] = mu
                            residuals.append(
                                ResidualSamples(
                                    contract,
                                    data[role]["features"],
                                    data[role]["next_states"] - mu,
                                    tuple(data[role]["ids"]),
                                )
                            )
                        train_support = nearest(model, data["scale"])
                        threshold = float(np.quantile(train_support, 0.9))
                        test_support = nearest(model, test)
                        predictions["nearest_training_distance"] = test_support
                        report["support_threshold_from_scale_inputs"] = threshold
                        report["test_fraction_farther_support"] = float(
                            np.mean(test_support > threshold)
                        )
                        for kind in ("global", "local"):
                            cal = fit_error_calibration(
                                *residuals,
                                training_sample_ids=tuple(data["train"]["ids"]),
                                mode=kind,
                                neighbors=24,
                            )
                            cal.save(folder / f"{kind}-calibration.npz")
                            bounds = cal.interval(
                                test["features"], mean, contract=contract
                            )
                            (
                                predictions[f"{kind}_lower"],
                                predictions[f"{kind}_upper"],
                            ) = bounds
                            report["methods"][kind] = score(
                                mean, test, bounds, test_support, threshold
                            )
                            groups = data["calibration"]["groups"]
                            flight_scores = np.array(
                                [
                                    cal.calibration_scores[groups == g].max()
                                    for g in np.unique(groups)
                                ]
                            )
                            report[f"{kind}_flight_rank_diagnostic"] = {
                                "flight_count": len(flight_scores),
                                "max_scores": flight_scores.tolist(),
                                "finite_95_percent_multiplier": bool(
                                    np.isfinite(conformal_quantile(flight_scores, 0.05))
                                ),
                                "meaning": "Only three groups: a 95% flight-level split-conformal rank would be infinite even if future flights were exchangeable",
                            }
                    # Source arrays include full actuation windows for the separate
                    # structured reference and for checking bin compression.
                    np.savez_compressed(
                        folder / "samples.npz",
                        **{
                            f"{role}_{key}": value
                            for role, items in data.items()
                            for key, value in items.items()
                        },
                    )
                    np.savez_compressed(folder / "predictions.npz", **predictions)
                    report["elapsed_s"] = time.monotonic() - start
                    write_json(folder / "report.json", report)
                    reports.append(report)
                    print(
                        json.dumps(
                            {
                                "seed": seed,
                                "allocation": allocation,
                                "horizon": horizon,
                                "mode": mode,
                                "velocity": report["methods"]["gp"]["all"][
                                    "velocity_rmse_m_s"
                                ],
                                "rates": report["methods"]["gp"]["all"][
                                    "rate_rmse_rad_s"
                                ],
                                "elapsed_s": report["elapsed_s"],
                            }
                        ),
                        flush=True,
                    )
    write_json(output / "summary.json", {"models": len(reports), "reports": reports})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument(
        "--phase", choices=("development", "confirmation"), required=True
    )
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--structured", type=Path)
    args = parser.parse_args()
    if not jax.config.jax_enable_x64:
        parser.error("Use JAX_ENABLE_X64=1 for recorded numerical experiments")
    run(args.output, args.corpus, args.phase, args.seeds, args.steps, args.structured)
