"""Reconstruct real-flight windows, replay predictions, and render results."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import zipfile
from pathlib import Path
from statistics import NormalDist

import numpy as np
from report_error_calibration import replay_scale, rq


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def gp_replay(model, metadata, features):
    q = (features - model["feature_mean"]) / model["feature_scale"]
    cross = np.concatenate(
        [
            rq(q[start : start + 128], model["features"], model["theta"])
            for start in range(0, len(q), 128)
        ]
    )
    mean = cross @ model["alpha"] * model["target_scale"] + model["target_mean"]
    if metadata.get("mean_mode", "absolute") == "increment":
        mean += features[:, : metadata["state_size"]]
    projected = np.linalg.solve(model["chol"], cross.T)
    variance = (
        np.maximum(np.exp(2 * model["theta"][-2]) - np.sum(projected**2, axis=0), 0)[
            :, None
        ]
        * model["target_scale"] ** 2
    )
    return (
        mean,
        variance,
        variance + np.exp(2 * model["theta"][-1]) * model["target_scale"] ** 2,
    )


def model_fingerprint(model, metadata):
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            [
                metadata[k]
                for k in (
                    "kernel",
                    "dt_s",
                    "state_size",
                    "command_size",
                    "context_size",
                )
            ]
        ).encode()
    )
    if metadata.get("mean_mode", "absolute") != "absolute":
        digest.update(metadata["mean_mode"].encode())
    for name in (
        "feature_mean",
        "feature_scale",
        "target_mean",
        "target_scale",
        "features",
        "theta",
        "chol",
        "alpha",
    ):
        v = np.ascontiguousarray(model[name])
        digest.update(json.dumps([name, v.dtype.str, v.shape]).encode())
        digest.update(v.tobytes())
    return digest.hexdigest()


def rotations(raw):
    # Independent quaternion-to-matrix reconstruction from upstream XYZW data.
    q = raw[:, 4:8] / np.linalg.norm(raw[:, 4:8], axis=1)[:, None]
    x, y, z, w = q.T
    return np.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        axis=1,
    )


def check_windows(samples, prefix, raw_by_name, horizon, history, usage):
    all_groups = samples[f"{prefix}_groups"]
    usage_by_name = {item["recording"]: item for item in usage}
    assert set(usage_by_name) == set(all_groups)
    for group in np.unique(all_groups):
        mask = all_groups == group
        anchors = samples[f"{prefix}_anchors"][mask]
        raw = raw_by_name[group]
        states = raw[:, [8, 9, 10, 11, 12, 13]]
        controls = raw[:, [17, 14, 15, 16]]
        assert (
            len(anchors) == len(set(anchors))
            and anchors.min() >= 10
            and (anchors + horizon).max() < len(raw)
        )
        np.testing.assert_allclose(
            samples[f"{prefix}_states"][mask], states[anchors], atol=1e-12
        )
        np.testing.assert_allclose(
            samples[f"{prefix}_next_states"][mask],
            states[anchors + horizon],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            samples[f"{prefix}_commands"][mask], controls[anchors], atol=1e-9
        )
        edges = np.linspace(0, horizon, min(10, horizon) + 1, dtype=int)
        bins = np.stack(
            [
                np.mean(
                    controls[anchors[:, None] + np.arange(edges[i], edges[i + 1])],
                    axis=1,
                )
                for i in range(len(edges) - 1)
            ],
            axis=1,
        )
        context = [rotations(raw)[anchors], bins.reshape(len(anchors), -1)]
        for lag in history:
            context.extend(
                (
                    states[anchors - lag] - states[anchors],
                    controls[anchors - lag] - controls[anchors],
                )
            )
        expected_context = np.concatenate(context, axis=1)
        np.testing.assert_allclose(
            samples[f"{prefix}_context"][mask], expected_context, atol=1e-9
        )
        np.testing.assert_allclose(
            samples[f"{prefix}_features"][mask],
            np.concatenate(
                (states[anchors], controls[anchors], expected_context), axis=1
            ),
            atol=1e-9,
        )
        np.testing.assert_array_equal(
            samples[f"{prefix}_ids"][mask],
            [f"{group}/anchor{a}/h{horizon}" for a in anchors],
        )
        np.testing.assert_allclose(
            samples[f"{prefix}_trend"][mask],
            states[anchors] + horizon / 5 * (states[anchors] - states[anchors - 5]),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            samples[f"{prefix}_full_controls"][mask],
            (controls[anchors[:, None] + np.arange(horizon)] / 2500.0) ** 2,
            atol=1e-12,
        )
        initial = samples[f"{prefix}_full_initial"][mask]
        np.testing.assert_allclose(initial[:, :6], raw[anchors][:, [1, 2, 3, 8, 9, 10]])
        np.testing.assert_allclose(initial[:, 10:13], states[anchors, 3:])
        quaternion = raw[anchors][:, [7, 4, 5, 6]]
        quaternion /= np.linalg.norm(quaternion, axis=1, keepdims=True)
        # Quaternion signs can differ without changing the represented rotation.
        np.testing.assert_allclose(
            abs(np.sum(quaternion * initial[:, 6:10], axis=1)), 1
        )
        state_rows = set(anchors) | set(anchors + horizon)
        control_rows = {a + step for a in anchors for step in range(horizon)}
        for lag in history:
            state_rows.update(anchors - lag)
            control_rows.update(anchors - lag)
        item = usage_by_name[group]
        assert item["windows"] == len(anchors)
        assert item["unique_state_rows"] == len(state_rows)
        assert item["unique_control_intervals"] == len(control_rows)
        np.testing.assert_allclose(
            item["control_interval_union_s"], len(control_rows) * 0.01
        )
        np.testing.assert_allclose(
            item["available_recording_s"], raw[-1, 0] - raw[0, 0]
        )


def scale_replay(artifact, metadata, features):
    return np.concatenate(
        [
            replay_scale(artifact, metadata, features[start : start + 128])
            for start in range(0, len(features), 128)
        ]
    )


def audit(run, corpus, output):
    plan = json.loads((run / "plan.json").read_text())
    source_hashes = json.loads((run / "sources.json").read_text())
    with zipfile.ZipFile(run / "executed-sources.zip") as archive:
        assert archive.testzip() is None
        for name, digest in source_hashes.items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest
    data_audit = json.loads((run / "data-audit.json").read_text())
    raw_by_name = {}
    for recording in data_audit["recordings"]:
        path = corpus / "raw" / recording["benchmark"]["relative_path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == recording["sha256"]
        raw_by_name[path.stem] = np.loadtxt(path, delimiter=",", skiprows=1)
    assert len(raw_by_name) == 15
    role_groups = [
        set(plan["split"][role]) for role in ("train", "scale", "calibration", "test")
    ]
    assert sum(map(len, role_groups)) == len(set.union(*role_groups)) == 15
    for group in plan["split"]["test"]:
        assert group.startswith("melon_")
    ref = json.loads((run / "structured-reference.json").read_text())
    assert hashlib.sha256(Path(ref["path"]).read_bytes()).hexdigest() == ref["sha256"]
    reference_fit = json.loads((Path(ref["path"]).parent / "fit.json").read_text())
    assert set(reference_fit["split"]["training_source_groups"]) == role_groups[0]
    assert set(reference_fit["split"]["validation_source_groups"]) == role_groups[1]
    found = set()
    reports = []
    maximum_difference = 0.0
    calibrations = 0
    baseline = {}
    test_identity = {}
    for file in sorted(run.glob("*-seed*/report.json")):
        folder = file.parent
        report = json.loads(file.read_text())
        reports.append(report)
        seed, allocation, mode, h = (
            report["seed"],
            report["allocation"],
            report["mode"],
            report["horizon_steps"],
        )
        found.add((seed, allocation, mode, h))
        with (
            np.load(folder / "samples.npz", allow_pickle=False) as data,
            np.load(folder / "model.npz", allow_pickle=False) as model,
            np.load(folder / "predictions.npz", allow_pickle=False) as predictions,
        ):
            metadata = json.loads(str(model["metadata"]))
            history = plan["modes"][mode][1]
            used = set()
            for role in ("train", "scale", "calibration", "test"):
                assert set(data[f"{role}_groups"]) == set(plan["split"][role])
                ids = list(data[f"{role}_ids"])
                assert len(ids) == len(set(ids)) and used.isdisjoint(ids)
                used.update(ids)
                if role != "test":
                    assert len(ids) == plan["allocations"][allocation][role]
                check_windows(
                    data, role, raw_by_name, h, history, report["usage"][role]
                )
            # Same forecast origins across methods, allocations and sampling seeds.
            if h in test_identity:
                np.testing.assert_array_equal(test_identity[h], data["test_ids"])
            else:
                test_identity[h] = data["test_ids"].copy()
            np.testing.assert_array_equal(predictions["hold"], data["test_states"])
            np.testing.assert_array_equal(predictions["trend"], data["test_trend"])
            if h in baseline:
                np.testing.assert_array_equal(baseline[h], predictions["structured"])
            else:
                baseline[h] = predictions["structured"].copy()
            x = data["train_features"]
            target = data["train_next_states"] - (
                data["train_states"] if metadata.get("mean_mode") == "increment" else 0
            )
            for name, values in (("feature", x), ("target", target)):
                center = values.mean(axis=0)
                spread = values.std(axis=0)
                spread = np.where(spread > 1e-8, spread, 1.0)
                np.testing.assert_allclose(model[f"{name}_mean"], center, atol=1e-10)
                np.testing.assert_allclose(model[f"{name}_scale"], spread, atol=1e-10)
            z = (x - model["feature_mean"]) / model["feature_scale"]
            np.testing.assert_allclose(z, model["features"], atol=1e-10)
            covariance = rq(z, z, model["theta"]) + (
                np.exp(2 * model["theta"][-1]) + 1e-6
            ) * np.eye(len(z))
            assert metadata["fit_report"]["numerical_jitter_standardized"] == 1e-6
            np.testing.assert_allclose(
                model["chol"] @ model["chol"].T, covariance, atol=1e-9
            )
            np.testing.assert_allclose(
                covariance @ model["alpha"],
                (target - model["target_mean"]) / model["target_scale"],
                atol=1e-8,
            )
            means = {}
            for role in ("scale", "calibration", "test"):
                mean, fvar, ovar = gp_replay(model, metadata, data[f"{role}_features"])
                means[role] = mean
                expected = (
                    predictions["mean"] if role == "test" else data[f"{role}_mean"]
                )
                maximum_difference = max(
                    maximum_difference, float(np.max(abs(mean - expected)))
                )
                np.testing.assert_allclose(mean, expected, atol=1e-8)
            np.testing.assert_allclose(
                fvar, predictions["function_variance"], atol=1e-8
            )
            np.testing.assert_allclose(
                ovar, predictions["observation_variance"], atol=1e-8
            )
            with np.load(folder / "ridge.npz", allow_pickle=False) as ridge:
                normalized = (x - ridge["feature_mean"]) / ridge["feature_scale"]
                targets = (
                    data["train_next_states"]
                    - data["train_states"]
                    - ridge["target_mean"]
                ) / ridge["target_scale"]
                np.testing.assert_allclose(
                    (normalized.T @ normalized + np.eye(x.shape[1])) @ ridge["weights"],
                    normalized.T @ targets,
                    atol=1e-7,
                )
                value = (
                    data["test_states"]
                    + (
                        (data["test_features"] - ridge["feature_mean"])
                        / ridge["feature_scale"]
                        @ ridge["weights"]
                    )
                    * ridge["target_scale"]
                    + ridge["target_mean"]
                )
                np.testing.assert_allclose(value, predictions["ridge"], atol=1e-9)
            bounds_by_method = {}
            critical = NormalDist().inv_cdf(1 - 0.05 / 12)
            bounds_by_method["gp"] = (
                mean - critical * np.sqrt(ovar),
                mean + critical * np.sqrt(ovar),
            )
            for kind in ("global", "local"):
                with np.load(
                    folder / f"{kind}-calibration.npz", allow_pickle=False
                ) as artifact:
                    metadata_cal = json.loads(str(artifact["metadata"]))
                    assert metadata_cal["contract"]["model_id"] == model_fingerprint(
                        model, metadata
                    )
                    assert metadata_cal["contract"]["dt_s"] == h * 0.01
                    assert len(metadata_cal["contract"]["input_names"]) == x.shape[1]
                    for calrole, role in (
                        ("training", "train"),
                        ("scale", "scale"),
                        ("calibration", "calibration"),
                    ):
                        assert metadata_cal[f"{calrole}_sample_ids"] == list(
                            data[f"{role}_ids"]
                        )
                    scale_features = data["scale_features"]
                    np.testing.assert_allclose(
                        artifact["feature_mean"],
                        scale_features.mean(axis=0),
                        atol=1e-10,
                    )
                    scale_std = scale_features.std(axis=0)
                    scale_std = np.where(scale_std > 1e-8, scale_std, 1.0)
                    np.testing.assert_allclose(
                        artifact["feature_scale"], scale_std, atol=1e-10
                    )
                    np.testing.assert_allclose(
                        artifact["reference_features"],
                        (scale_features - artifact["feature_mean"])
                        / artifact["feature_scale"],
                        atol=1e-10,
                    )
                    np.testing.assert_allclose(
                        artifact["reference_residuals"],
                        data["scale_next_states"] - means["scale"],
                        atol=1e-8,
                    )
                    scores = np.max(
                        abs(data["calibration_next_states"] - means["calibration"])
                        / scale_replay(
                            artifact, metadata_cal, data["calibration_features"]
                        ),
                        axis=1,
                    )
                    np.testing.assert_allclose(
                        scores, artifact["calibration_scores"], atol=1e-7
                    )
                    quantile = np.sort(np.r_[scores, np.inf])[
                        math.ceil((len(scores) + 1) * 0.95) - 1
                    ]
                    width = quantile * scale_replay(
                        artifact, metadata_cal, data["test_features"]
                    )
                    bounds_by_method[kind] = (mean - width, mean + width)
                    diagnostic = report[f"{kind}_flight_rank_diagnostic"]
                    assert (
                        diagnostic["flight_count"] == 3
                        and not diagnostic["finite_95_percent_multiplier"]
                    )
                    np.testing.assert_allclose(
                        diagnostic["max_scores"],
                        [
                            max(scores[data["calibration_groups"] == g])
                            for g in sorted(set(data["calibration_groups"]))
                        ],
                        atol=1e-7,
                    )
                    calibrations += 1
            normalized_test = (data["test_features"] - model["feature_mean"]) / model[
                "feature_scale"
            ]
            nearest_distance = np.array(
                [np.linalg.norm(z - q, axis=1).min() for q in normalized_test]
            )
            np.testing.assert_allclose(
                nearest_distance, predictions["nearest_training_distance"], atol=1e-7
            )
            scale_q = (data["scale_features"] - model["feature_mean"]) / model[
                "feature_scale"
            ]
            threshold = np.quantile(
                [np.linalg.norm(z - q, axis=1).min() for q in scale_q], 0.9
            )
            np.testing.assert_allclose(
                threshold, report["support_threshold_from_scale_inputs"], atol=1e-7
            )
            masks = {
                "all": np.ones(len(mean), bool),
                **{g: data["test_groups"] == g for g in np.unique(data["test_groups"])},
                "nearer-support": nearest_distance <= threshold,
                "farther-support": nearest_distance > threshold,
            }
            for method, regions in report["methods"].items():
                mu = mean if method in bounds_by_method else predictions[method]
                if method in bounds_by_method:
                    lower, upper = bounds_by_method[method]
                    np.testing.assert_allclose(
                        lower, predictions[f"{method}_lower"], atol=1e-7
                    )
                    np.testing.assert_allclose(
                        upper, predictions[f"{method}_upper"], atol=1e-7
                    )
                for region, metrics in regions.items():
                    mask = masks[region]
                    assert int(mask.sum()) == metrics["count"]
                    errors = mu[mask] - data["test_next_states"][mask]
                    np.testing.assert_allclose(
                        np.sqrt(np.mean(errors**2, axis=0)),
                        metrics["per_channel_rmse"],
                        atol=1e-8,
                    )
                    for col, key in (
                        (slice(0, 3), "velocity_rmse_m_s"),
                        (slice(3, 6), "rate_rmse_rad_s"),
                    ):
                        np.testing.assert_allclose(
                            np.sqrt(np.mean(np.sum(errors[:, col] ** 2, axis=1))),
                            metrics[key],
                            atol=1e-8,
                        )
                    if method in bounds_by_method:
                        contained = (data["test_next_states"][mask] >= lower[mask]) & (
                            data["test_next_states"][mask] <= upper[mask]
                        )
                        np.testing.assert_allclose(
                            np.mean(np.all(contained, axis=1)),
                            metrics["joint_coverage"],
                            atol=1e-10,
                        )
                        np.testing.assert_allclose(
                            np.mean(contained, axis=0),
                            metrics["marginal_coverage"],
                            atol=1e-10,
                        )
                        width = upper[mask] - lower[mask]
                        np.testing.assert_allclose(
                            [width[:, :3].mean(), width[:, 3:].mean()],
                            [
                                metrics["velocity_mean_width_m_s"],
                                metrics["rate_mean_width_rad_s"],
                            ],
                            atol=1e-8,
                        )
    expected = {
        (seed, allocation, mode, h)
        for seed in plan["seeds"]
        for allocation in plan["allocations"]
        for mode in plan["modes"]
        for h in plan["horizon_steps"]
    }
    assert found == expected
    saved = json.loads((run / "summary.json").read_text())
    assert saved["models"] == len(reports)
    assert {json.dumps(r, sort_keys=True) for r in saved["reports"]} == {
        json.dumps(r, sort_keys=True) for r in reports
    }
    summary = {}
    for allocation in plan["allocations"]:
        summary[allocation] = {}
        for mode in plan["modes"]:
            summary[allocation][mode] = {}
            for h in plan["horizon_steps"]:
                selected = [
                    r
                    for r in reports
                    if r["allocation"] == allocation
                    and r["mode"] == mode
                    and r["horizon_steps"] == h
                ]
                methods = {}
                for method in selected[0]["methods"]:
                    keys = [
                        k
                        for k, v in selected[0]["methods"][method]["all"].items()
                        if np.isscalar(v)
                    ]
                    methods[method] = {
                        k: {
                            "mean": float(
                                np.mean(
                                    [r["methods"][method]["all"][k] for r in selected]
                                )
                            ),
                            "min": float(
                                np.min(
                                    [r["methods"][method]["all"][k] for r in selected]
                                )
                            ),
                            "max": float(
                                np.max(
                                    [r["methods"][method]["all"][k] for r in selected]
                                )
                            ),
                        }
                        for k in keys
                    }
                summary[allocation][mode][str(h)] = {
                    "methods": methods,
                    "support_regions": {
                        region: {
                            key: {
                                name: float(
                                    reducer(
                                        [
                                            r["methods"]["local"][region][key]
                                            for r in selected
                                        ]
                                    )
                                )
                                for name, reducer in (
                                    ("mean", np.mean),
                                    ("min", np.min),
                                    ("max", np.max),
                                )
                            }
                            for key in (
                                "velocity_rmse_m_s",
                                "rate_rmse_rad_s",
                                "joint_coverage",
                                "velocity_mean_width_m_s",
                                "rate_mean_width_rad_s",
                            )
                        }
                        for region in ("nearer-support", "farther-support")
                    },
                    "fraction_farther_support": {
                        key: float(
                            reducer(
                                [r["test_fraction_farther_support"] for r in selected]
                            )
                        )
                        for key, reducer in (
                            ("mean", np.mean),
                            ("min", np.min),
                            ("max", np.max),
                        )
                    },
                }
    write_json(output / "summary.json", summary)
    write_json(
        output / "audit.json",
        {
            "status": "passed",
            "raw_recordings_verified": len(raw_by_name),
            "gp_models": len(reports),
            "ridge_models": len(reports),
            "calibrations": calibrations,
            "maximum_gp_numpy_replay_difference": maximum_difference,
            "checks": [
                "source CRC/SHA256 and raw corpus pins",
                "complete planned cases and whole-flight roles",
                "every feature and target reconstructed from source sample indices",
                "supporting row/interval budgets and structured initial states",
                "future-state exclusion and causal lag indexing",
                "measured speed decoding and future control bins",
                "same test origins across all cases",
                "training-only GP normalization and covariance conditioning",
                "independent GP/ridge/residual-scale/quantile replay",
                "input contracts and model revision fingerprint",
                "all per-flight, pooled and support-region metrics",
                "three-flight rank diagnostic",
                "structured reference source groups and repeated prediction alignment",
            ],
            "scope": "Structured physics integration uses the existing evaluator; the independent NumPy replay covers the generic GP, ridge, and calibration calculations. Three seeds reuse the same three held-out flights.",
        },
    )
    return plan, summary


def render(run, output, plan, summary):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullLocator

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    colors = {
        "absolute": "#d99233",
        "increment": "#328db2",
        "history": "#148b82",
        "ridge": "#6e5798",
        "hold": "#8c99a6",
        "structured": "#1b293a",
    }
    horizons = np.asarray(plan["horizon_steps"]) * 0.01
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.4))
    fig.subplots_adjust(
        left=0.08, right=0.98, bottom=0.17, top=0.90, wspace=0.26, hspace=0.43
    )
    for row, allocation in enumerate(plan["allocations"]):
        for col, key in enumerate(("velocity_rmse_m_s", "rate_rmse_rad_s")):
            ax = axes[row, col]
            for mode in plan["modes"]:
                vals = [
                    summary[allocation][mode][str(h)]["methods"]["gp"][key]
                    for h in plan["horizon_steps"]
                ]
                ax.errorbar(
                    horizons,
                    [v["mean"] for v in vals],
                    yerr=[
                        [v["mean"] - v["min"] for v in vals],
                        [v["max"] - v["mean"] for v in vals],
                    ],
                    marker="o",
                    capsize=3,
                    color=colors[mode],
                    label={
                        "absolute": "GP: future state",
                        "increment": "GP: state change",
                        "history": "GP: state change + history",
                    }[mode],
                )
            for method, label, style in (
                ("hold", "Hold current value", ":"),
                ("ridge", "Linear change + history", "--"),
                ("structured", "Structured reference", "-."),
            ):
                vals = [
                    summary[allocation]["history"][str(h)]["methods"][method][key][
                        "mean"
                    ]
                    for h in plan["horizon_steps"]
                ]
                ax.plot(horizons, vals, color=colors[method], ls=style, label=label)
            budget = plan["allocations"][allocation]
            ax.set(
                xscale="log",
                yscale="log",
                xlabel="Forecast horizon [s]",
                ylabel="Velocity vector RMSE [m/s]"
                if col == 0
                else "Body-rate vector RMSE [rad/s]",
                title=f"{budget['train']} fit + {budget['scale'] + budget['calibration']} error-evidence windows",
                xticks=horizons,
                xticklabels=[f"{h:g}" for h in horizons],
            )
            ax.xaxis.set_minor_locator(NullLocator())
            ax.grid(alpha=0.15)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.045),
        ncol=3,
        fontsize=9,
    )
    fig.suptitle("Temporal structure on three held-out Crazyflie flights", fontsize=16)
    fig.text(
        0.5,
        0.02,
        "Offline processed measurements · supplied future motor speeds · 3 training-window draws, whiskers show range · structured reference has more supervision",
        fontsize=7.5,
        ha="center",
    )
    for ext in ("png", "svg"):
        fig.savefig(output / f"real-forecast-errors.{ext}", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.7))
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.29, top=0.80, wspace=0.33)
    for allocation, style in (("more-evidence", "-"), ("more-model", "--")):
        for method, color in (
            ("gp", "#8191a6"),
            ("global", "#d99233"),
            ("local", "#148b82"),
        ):
            for col, key in enumerate(
                ("joint_coverage", "velocity_mean_width_m_s", "rate_mean_width_rad_s")
            ):
                vals = [
                    summary[allocation]["history"][str(h)]["methods"][method][key][
                        "mean"
                    ]
                    for h in plan["horizon_steps"]
                ]
                axes[col].plot(
                    horizons,
                    vals,
                    marker="o",
                    color=color,
                    ls=style,
                    label=f"{'GP' if method == 'gp' else method.title()} / {allocation}",
                )
                axes[col].set(
                    xscale="log",
                    xlabel="Forecast horizon [s]",
                    xticks=horizons,
                    xticklabels=[f"{h:g}" for h in horizons],
                )
                axes[col].xaxis.set_minor_locator(NullLocator())
                axes[col].grid(alpha=0.15)
    axes[0].axhline(0.95, color="#1b293a", ls=":", lw=1)
    axes[0].set(
        ylim=(0, 1.02),
        ylabel="Six-output joint empirical coverage",
        title="Nominal 95% is not a guarantee",
    )
    axes[1].set(
        ylabel="Mean channel interval width [m/s]",
        title="Velocity uncertainty",
        yscale="log",
    )
    axes[2].set(
        ylabel="Mean channel interval width [rad/s]",
        title="Body-rate uncertainty",
        yscale="log",
    )
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.06),
        ncol=3,
        fontsize=8,
    )
    fig.suptitle("Calibration under a held-out maneuver", fontsize=16)
    fig.text(
        0.5,
        0.025,
        "History GP · three correlated calibration recordings · solid: more evidence, dashed: more mean fitting · empirical coverage only",
        fontsize=8,
        ha="center",
    )
    for ext in ("png", "svg"):
        fig.savefig(output / f"real-forecast-coverage.{ext}", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    plan, summary = audit(args.run, args.corpus, args.output)
    render(args.run, args.output, plan, summary)
    dependencies = [
        Path(__file__).resolve(),
        Path(__file__).resolve().with_name("report_error_calibration.py"),
    ]
    for source in dependencies:
        shutil.copy2(source, args.output / source.name)
    write_json(
        args.output / "renderer-sources.json",
        {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in dependencies},
    )
    print((args.output / "audit.json").read_text())
