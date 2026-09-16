"""Independent NumPy replay, data-role audit, and scientific figures."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import zipfile
from pathlib import Path

import numpy as np


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def rq(left, right, theta):
    dimension = left.shape[1]
    distance = np.sum(
        ((left[:, None] - right[None]) / np.exp(theta[:dimension])) ** 2, axis=-1
    )
    shape = np.exp(theta[dimension])
    return np.exp(2 * theta[-2] - shape * np.log1p(distance / (2 * shape)))


def replay_gp(model, features):
    normalized = (features - model["feature_mean"]) / model["feature_scale"]
    cross = rq(normalized, model["features"], model["theta"])
    mean = cross @ model["alpha"] * model["target_scale"] + model["target_mean"]
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


def fingerprint(model, metadata):
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            [
                metadata[name]
                for name in (
                    "kernel",
                    "dt_s",
                    "state_size",
                    "command_size",
                    "context_size",
                )
            ]
        ).encode()
    )
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
        value = np.ascontiguousarray(model[name])
        digest.update(json.dumps([name, value.dtype.str, value.shape]).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def replay_scale(artifact, metadata, features):
    rms = np.sqrt(np.mean(artifact["reference_residuals"] ** 2, axis=0))
    floor = np.where(rms > 0, 0.05 * rms, 1.0)
    if metadata["mode"] == "global":
        return np.broadcast_to(np.maximum(rms, floor), (len(features), len(rms))).copy()
    normalized = (features - artifact["feature_mean"]) / artifact["feature_scale"]
    distance = np.sum(
        (normalized[:, None] - artifact["reference_features"][None]) ** 2, axis=-1
    )
    k = min(metadata["neighbors"], len(artifact["reference_features"]))
    indices = np.argsort(distance, axis=1, kind="stable")[:, :k]
    selected = np.take_along_axis(distance, indices, axis=1)
    radius = selected[:, -1:]
    weights = np.exp(
        -2 * np.divide(selected, radius, out=np.zeros_like(selected), where=radius > 0)
    )
    squared_error = artifact["reference_residuals"][indices] ** 2
    scale = np.sqrt(
        np.sum(weights[..., None] * squared_error, axis=1)
        / weights.sum(axis=1)[:, None]
    )
    return np.maximum(scale, floor)


def audit(run, output):
    plan = json.loads((run / "plan.json").read_text())
    manifest = json.loads((run / "sources.json").read_text())
    with zipfile.ZipFile(run / "executed-sources.zip") as archive:
        assert archive.testzip() is None
        for name, expected in manifest.items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == expected
    found = set()
    model_count, calibration_count, maximum_difference = 0, 0, 0.0
    reports = []
    diagnosis = {
        "scope": "Oracle analysis of the synthetic memory benchmark, not an inferred diagnosis from telemetry",
        "cases": {},
    }
    for report_path in sorted(run.glob("*-seed*/report.json")):
        directory = report_path.parent
        report = json.loads(report_path.read_text())
        reports.append(report)
        case, seed = report["case"], report["seed"]
        found.add((case, seed))
        with (
            np.load(directory / "samples.npz", allow_pickle=False) as samples,
            np.load(directory / "model.npz", allow_pickle=False) as model,
            np.load(directory / "predictions.npz", allow_pickle=False) as prediction,
        ):
            metadata = json.loads(str(model["metadata"]))
            assert fingerprint(model, metadata) == report["contract"]["model_id"]
            used = set()
            for role, count in plan["budgets"].items():
                ids = list(samples[f"{role}_ids"])
                assert len(ids) == len(set(ids)) == count
                assert used.isdisjoint(ids)
                used.update(ids)
            for prefix, raw in (
                ("feature", samples["train_features"]),
                ("target", samples["train_observed"]),
            ):
                np.testing.assert_allclose(
                    model[f"{prefix}_mean"], raw.mean(axis=0), atol=1e-12
                )
                np.testing.assert_allclose(
                    model[f"{prefix}_scale"], raw.std(axis=0), atol=1e-12
                )
            normalized_train = (
                samples["train_features"] - model["feature_mean"]
            ) / model["feature_scale"]
            np.testing.assert_allclose(model["features"], normalized_train, atol=1e-12)
            jitter = max(
                1e-6,
                2
                * len(normalized_train)
                * np.finfo(model["features"].dtype).eps
                * np.exp(2 * model["theta"][-2]),
            )
            covariance = rq(normalized_train, normalized_train, model["theta"]) + (
                np.exp(2 * model["theta"][-1]) + jitter
            ) * np.eye(len(normalized_train))
            np.testing.assert_allclose(
                model["chol"] @ model["chol"].T, covariance, atol=1e-10
            )
            standardized_targets = (
                samples["train_observed"] - model["target_mean"]
            ) / model["target_scale"]
            np.testing.assert_allclose(
                covariance @ model["alpha"], standardized_targets, atol=1e-8
            )
            mean, fvar, ovar = replay_gp(model, samples["test_features"])
            for actual, key in (
                (mean, "mean"),
                (fvar, "function_variance"),
                (ovar, "observation_variance"),
            ):
                np.testing.assert_allclose(
                    actual, prediction[key], atol=1e-9, rtol=1e-9
                )
                maximum_difference = max(
                    maximum_difference, float(np.max(np.abs(actual - prediction[key])))
                )
            probe_mean = replay_gp(model, prediction["probe_features"])[0]
            np.testing.assert_allclose(probe_mean, prediction["probe_mean"], atol=1e-9)
            for method, regions in report["methods"].items():
                if method == "gp":
                    lower, upper = (
                        mean - 1.95996398454 * np.sqrt(ovar),
                        mean + 1.95996398454 * np.sqrt(ovar),
                    )
                else:
                    with np.load(
                        directory / f"{method}.npz", allow_pickle=False
                    ) as artifact:
                        evidence = json.loads(str(artifact["metadata"]))
                        assert evidence["contract"]["model_id"] == fingerprint(
                            model, metadata
                        )
                        assert evidence["contract"]["dt_s"] == metadata["dt_s"]
                        assert (
                            evidence["contract"]["input_names"]
                            == report["contract"]["input_names"]
                        )
                        assert evidence["training_sample_ids"] == list(
                            samples["train_ids"]
                        )
                        n = len(evidence["scale_sample_ids"])
                        assert n in plan["evidence_rows_per_role"] and n == len(
                            evidence["calibration_sample_ids"]
                        )
                        prefix = "_outer" if method.startswith("outer") else ""
                        roles = [
                            set(evidence[f"{role}_sample_ids"])
                            for role in ("training", "scale", "calibration")
                        ]
                        assert (
                            roles[0].isdisjoint(roles[1])
                            and roles[0].isdisjoint(roles[2])
                            and roles[1].isdisjoint(roles[2])
                        )
                        assert set.union(*roles).isdisjoint(samples["test_ids"])
                        for role in ("scale", "calibration"):
                            assert evidence[f"{role}_sample_ids"] == list(
                                samples[f"{role}{prefix}_ids"][:n]
                            )
                        scale_features = samples[f"scale{prefix}_features"][:n]
                        scale_mean = replay_gp(model, scale_features)[0]
                        scale_errors = (
                            samples[f"scale{prefix}_observed"][:n] - scale_mean
                        )
                        np.testing.assert_allclose(
                            artifact["reference_residuals"], scale_errors, atol=1e-9
                        )
                        np.testing.assert_allclose(
                            artifact["feature_mean"],
                            scale_features.mean(axis=0),
                            atol=1e-12,
                        )
                        np.testing.assert_allclose(
                            artifact["feature_scale"],
                            scale_features.std(axis=0),
                            atol=1e-12,
                        )
                        np.testing.assert_allclose(
                            artifact["reference_features"],
                            (scale_features - artifact["feature_mean"])
                            / artifact["feature_scale"],
                            atol=1e-12,
                        )
                        cal_features = samples[f"calibration{prefix}_features"][:n]
                        cal_errors = (
                            samples[f"calibration{prefix}_observed"][:n]
                            - replay_gp(model, cal_features)[0]
                        )
                        scores = np.max(
                            np.abs(cal_errors)
                            / replay_scale(artifact, evidence, cal_features),
                            axis=1,
                        )
                        np.testing.assert_allclose(
                            scores, artifact["calibration_scores"], atol=1e-7
                        )
                        quantile = np.sort(np.r_[scores, np.inf])[
                            math.ceil((n + 1) * (1 - evidence["miscoverage"])) - 1
                        ]
                        width = quantile * replay_scale(
                            artifact, evidence, samples["test_features"]
                        )
                        lower, upper = mean - width, mean + width
                        probe_width = quantile * replay_scale(
                            artifact, evidence, prediction["probe_features"]
                        )
                        np.testing.assert_allclose(
                            probe_width, prediction[f"probe_{method}_width"], atol=1e-8
                        )
                        calibration_count += 1
                np.testing.assert_allclose(
                    lower, prediction[f"{method}_lower"], atol=1e-8
                )
                np.testing.assert_allclose(
                    upper, prediction[f"{method}_upper"], atol=1e-8
                )
                masks = {
                    "all": np.ones(len(mean), dtype=bool),
                    "feature-interior": np.linalg.norm(
                        samples["test_raw"][:, :2], axis=1
                    )
                    < 0.28,
                    "outer": abs(samples["test_raw"][:, 1]) >= 0.55,
                }
                for region, mask in masks.items():
                    assert int(mask.sum()) == regions[region]["count"]
                    measured = {
                        "latent_rmse": np.sqrt(
                            np.mean((mean[mask] - samples["test_latent"][mask]) ** 2)
                        ),
                        "observation_rmse": np.sqrt(
                            np.mean((mean[mask] - samples["test_observed"][mask]) ** 2)
                        ),
                        "joint_coverage": np.mean(
                            np.all(
                                (samples["test_observed"][mask] >= lower[mask])
                                & (samples["test_observed"][mask] <= upper[mask]),
                                axis=1,
                            )
                        ),
                        "mean_width": np.mean(upper[mask] - lower[mask]),
                    }
                    for key, value in measured.items():
                        np.testing.assert_allclose(
                            value, regions[region][key], atol=1e-8
                        )
            # These arms differ only in the available explanatory information.
            if case.startswith("memory"):
                raw = samples["test_raw"]
                if case == "memory-history":
                    conditional_mean = samples["test_latent"]
                    hidden_mse = 0.0
                else:
                    conditional_mean = 0.6 * raw[:, :1] + 0.3 * np.sin(2 * raw[:, 1:2])
                    hidden_mse = float(np.mean((0.35 + 0.15 * raw[:, 1]) ** 2 / 3))
                mean_fit_mse = float(np.mean((mean - conditional_mean) ** 2))
                diagnosis["cases"].setdefault(case, []).append(
                    {
                        "seed": seed,
                        "missing_history_mse": hidden_mse,
                        "conditional_mean_fit_mse": mean_fit_mse,
                        "expected_latent_mse": hidden_mse + mean_fit_mse,
                    }
                )
                with np.load(
                    run / f"memory-omitted-seed{seed}" / "samples.npz",
                    allow_pickle=False,
                ) as base:
                    for role in plan["budgets"]:
                        np.testing.assert_array_equal(
                            samples[f"{role}_observed"], base[f"{role}_observed"]
                        )
                        np.testing.assert_array_equal(
                            samples[f"{role}_features"][:, :2], base[f"{role}_features"]
                        )
                        if case == "memory-history":
                            np.testing.assert_array_equal(
                                samples[f"{role}_features"][:, 2],
                                samples[f"{role}_raw"][:, 2],
                            )
                        if case == "memory-unrelated":
                            assert not np.array_equal(
                                samples[f"{role}_features"][:, 2],
                                samples[f"{role}_raw"][:, 2],
                            )
            model_count += 1
    assert found == {(case, seed) for case in plan["cases"] for seed in plan["seeds"]}
    summary = json.loads((run / "summary.json").read_text())
    assert summary["models"] == model_count
    for case, methods in summary["cases"].items():
        selected = [r for r in reports if r["case"] == case]
        for method, regions in methods.items():
            for region, metrics in regions.items():
                for metric, aggregate in metrics.items():
                    values = [r["methods"][method][region][metric] for r in selected]
                    for key, reducer in (
                        ("mean", np.mean),
                        ("min", np.min),
                        ("max", np.max),
                    ):
                        np.testing.assert_allclose(
                            aggregate[key], reducer(values), atol=1e-12
                        )
    result = {
        "status": "passed",
        "models": model_count,
        "calibrations": calibration_count,
        "maximum_gp_replay_difference": maximum_difference,
        "checks": [
            "source CRC and SHA256",
            "complete planned cases and role budgets",
            "independent mean/scale/calibration/test IDs",
            "mean normalization and conditioning from training only",
            "GP prediction replay in NumPy",
            "content fingerprint",
            "scale normalization and residual replay",
            "score rank and all interval bounds",
            "paired history ablation observations",
            "all metrics and seed aggregations",
        ],
    }
    write_json(output / "audit.json", result)
    write_json(output / "memory-diagnosis.json", diagnosis)
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
    colors = ("#8191a6", "#df9b41", "#148b82")
    seed = plan["seeds"][0]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.2), layout="constrained")
    with np.load(
        run / f"localized-seed{seed}" / "predictions.npz", allow_pickle=False
    ) as data:
        for col, (method, title, color) in enumerate(
            zip(
                ("gp", "broad-global", "broad-local"),
                ("GP interval", "Global error calibration", "Local error calibration"),
                colors,
            )
        ):
            ax = axes[0, col]
            x, mean = data["probe_raw"][:, 1], data["probe_mean"][:, 0]
            width = data[f"probe_{method}_width"][:, 0]
            ax.fill_between(x, mean - width, mean + width, color=color, alpha=0.22)
            ax.plot(
                x,
                data["probe_latent"][:, 0],
                color="#152b40",
                ls="--",
                label="True response",
            )
            ax.plot(x, mean, color=color, lw=2, label="Frozen learned mean")
            ax.set(
                title=title, xlabel="Command [a.u.]", ylim=(-0.48, 0.88), xlim=(-1, 1)
            )
            ax.grid(alpha=0.15)
        axes[0, 0].set_ylabel("Next state at state = 0 [a.u.]")
        axes[0, 0].legend(fontsize=8, loc="upper left")
    methods = ("gp", "broad-global", "broad-local")
    case = summary["cases"]["localized"]
    x = np.arange(2)
    for offset, method, color in zip((-0.24, 0, 0.24), methods, colors):
        for col, metric in enumerate(("joint_coverage", "mean_width")):
            values = [
                case[method][region][metric] for region in ("all", "feature-interior")
            ]
            means = np.array([v["mean"] for v in values])
            error = np.array(
                [
                    [v["mean"] - v["min"] for v in values],
                    [v["max"] - v["mean"] for v in values],
                ]
            )
            axes[1, col].bar(
                x + offset,
                means,
                width=0.22,
                color=color,
                yerr=error,
                capsize=3,
                label={
                    "gp": "GP",
                    "broad-global": "Global calibration",
                    "broad-local": "Local calibration",
                }[method],
            )
    axes[1, 0].axhline(0.95, color="#152b40", ls="--", lw=1)
    axes[1, 0].legend(fontsize=8, loc="lower left")
    axes[1, 0].set(
        title="Coverage can hide local failures",
        ylabel="Observed coverage",
        ylim=(0, 1.04),
        xticks=x,
        xticklabels=("Whole domain", "Localized feature"),
    )
    axes[1, 1].set(
        title="Put interval width where errors occur",
        ylabel="Mean full interval width [a.u.]",
        xticks=x,
        xticklabels=("Whole domain", "Localized feature"),
    )
    for region, color, label in (
        ("all", "#8191a6", "Whole domain"),
        ("feature-interior", "#148b82", "Localized feature"),
    ):
        values = [
            case[method][region]["joint_coverage"]
            for method in ("broad64-local", "broad128-local", "broad-local")
        ]
        axes[1, 2].errorbar(
            [128, 256, 1024],
            [v["mean"] for v in values],
            yerr=[
                [v["mean"] - v["min"] for v in values],
                [v["max"] - v["mean"] for v in values],
            ],
            marker="o",
            capsize=3,
            color=color,
            label=label,
        )
    axes[1, 2].axhline(0.95, color="#152b40", ls="--", lw=1)
    axes[1, 2].set(
        title="Additional calibration observations",
        xlabel="Scale-fit + score-calibration rows",
        ylabel="Observed coverage",
        xscale="log",
        ylim=(0, 1.04),
        xticks=[128, 256, 1024],
        xticklabels=[128, 256, 1024],
    )
    axes[1, 2].legend(fontsize=8)
    axes[1, 2].xaxis.set_minor_locator(NullLocator())
    fig.suptitle(
        "A better error model without changing the predicted response", fontsize=16
    )
    fig.supxlabel(
        f"Synthetic independent resets · slice: seed {seed} · bars: {len(plan['seeds'])} fresh seeds, whiskers show range · nominal 95% observation intervals",
        fontsize=9,
    )
    for ext in ("png", "svg"):
        fig.savefig(output / f"calibrated-errors.{ext}", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.3), layout="constrained")
    memory_cases = ("memory-omitted", "memory-unrelated", "memory-history")
    labels = ("Current inputs", "+ unrelated input", "+ previous command")
    for case_name, label, color in zip(memory_cases, labels, colors):
        with np.load(
            run / f"{case_name}-seed{seed}" / "predictions.npz", allow_pickle=False
        ) as data:
            x, mean = data["probe_raw"][:, 2], data["probe_mean"][:, 0]
            width = data["probe_broad-local_width"][:, 0]
            axes[0].fill_between(x, mean - width, mean + width, color=color, alpha=0.12)
            axes[0].plot(x, mean, color=color, lw=2, label=label)
            if case_name == "memory-history":
                axes[0].plot(
                    x,
                    data["probe_latent"][:, 0],
                    color="#152b40",
                    ls="--",
                    label="True response",
                )
    axes[0].set(
        title="Same current state and command",
        xlabel="Previous command [a.u.]",
        ylabel="Next state [a.u.]",
    )
    axes[0].legend(fontsize=8)
    for col, metric, title in (
        (1, "latent_rmse", "History explains the missing variation"),
        (2, "mean_width", "Honest intervals become more useful"),
    ):
        values = [
            summary["cases"][name]["broad-local"]["all"][metric]
            for name in memory_cases
        ]
        axes[col].bar(
            np.arange(3),
            [v["mean"] for v in values],
            color=colors,
            yerr=[
                [v["mean"] - v["min"] for v in values],
                [v["max"] - v["mean"] for v in values],
            ],
            capsize=4,
        )
        axes[col].set(
            title=title,
            xticks=np.arange(3),
            xticklabels=(
                "Current\ninputs",
                "+ unrelated\ninput",
                "+ previous\ncommand",
            ),
            ylabel="RMSE [a.u.]"
            if col == 1
            else "Full observation interval width [a.u.]",
        )
    axes[1].axhline(
        plan["memory_irreducible_latent_rmse_without_history"],
        color="#152b40",
        ls="--",
        lw=1,
        label="Best possible without history",
    )
    axes[1].legend(fontsize=8)
    fig.suptitle("Causal history resolves an ambiguous response", fontsize=16)
    fig.supxlabel(
        f"Synthetic memory case · identical observations and label budgets · 192 mean-fit + 1,024 calibration observations · {len(plan['seeds'])} fresh seeds",
        fontsize=9,
    )
    for ext in ("png", "svg"):
        fig.savefig(output / f"history-and-error.{ext}", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    plan, summary = audit(args.run, args.output)
    render(args.run, args.output, plan, summary)
    source = Path(__file__).resolve()
    shutil.copy2(source, args.output / source.name)
    write_json(
        args.output / "renderer-source.json",
        {source.name: hashlib.sha256(source.read_bytes()).hexdigest()},
    )
    print((args.output / "audit.json").read_text())
