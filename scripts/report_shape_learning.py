"""Independently audit and render a completed shape-learning experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def verify(run, output):
    plan = json.loads((run / "plan.json").read_text())
    manifest = json.loads((run / "sources.json").read_text())
    with zipfile.ZipFile(run / "executed-sources.zip") as archive:
        assert archive.testzip() is None
        for name, expected in manifest.items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == expected
    expected_cases = {
        (system, layout, seed)
        for system in plan["systems"]
        for layout in plan["layouts"]
        for seed in plan["seeds"]
    }
    cases = set()
    models = 0
    maximum_difference = 0.0
    for path in sorted(run.glob("*-seed*/report.json")):
        report = json.loads(path.read_text())
        directory = path.parent
        cases.add((report["system"], report["layout"], report["seed"]))
        with np.load(directory / "samples.npz", allow_pickle=False) as data:
            ids = data["train_ids"]
            assert (
                len(ids) == len(set(ids)) == plan["training_budgets"][report["layout"]]
            )
            assert not np.any(
                np.all(data["train_features"][:, None] == data["query"][None], axis=-1)
            )
            if report["layout"] in ("random-fill", "coverage-fill"):
                assert np.all(ids[128:] >= 3000)
                parent = (
                    run
                    / f"{report['system']}-outer-seed{report['seed']}"
                    / "samples.npz"
                )
                with np.load(parent, allow_pickle=False) as base:
                    np.testing.assert_array_equal(ids[:128], base["train_ids"])
                    np.testing.assert_array_equal(
                        data["train_targets"][:128], base["train_targets"]
                    )
            for kernel in plan["kernels"]:
                with (
                    np.load(
                        directory / f"{kernel}-model.npz", allow_pickle=False
                    ) as model,
                    np.load(
                        directory / f"{kernel}-predictions.npz", allow_pickle=False
                    ) as prediction,
                ):
                    raw = data["train_features"]
                    np.testing.assert_allclose(
                        model["feature_mean"], raw.mean(axis=0), atol=1e-12
                    )
                    np.testing.assert_allclose(
                        model["feature_scale"], raw.std(axis=0), atol=1e-12
                    )
                    np.testing.assert_allclose(
                        model["target_mean"],
                        data["train_targets"].mean(axis=0),
                        atol=1e-12,
                    )
                    np.testing.assert_allclose(
                        model["target_scale"],
                        data["train_targets"].std(axis=0),
                        atol=1e-12,
                    )
                    train = (raw - model["feature_mean"]) / model["feature_scale"]
                    query = (data["query"] - model["feature_mean"]) / model[
                        "feature_scale"
                    ]
                    theta = model["theta"]
                    distance_sq = np.sum(
                        ((query[:, None] - train[None]) / np.exp(theta[: raw.shape[1]]))
                        ** 2,
                        axis=-1,
                    )
                    if kernel == "rbf":
                        cross = np.exp(2 * theta[-2] - 0.5 * distance_sq)
                    elif kernel == "rq":
                        shape = np.exp(theta[raw.shape[1]])
                        cross = np.exp(
                            2 * theta[-2] - shape * np.log1p(distance_sq / (2 * shape))
                        )
                    else:
                        radius = np.sqrt(5 * distance_sq + 1e-12)
                        cross = (
                            np.exp(2 * theta[-2])
                            * (1 + radius + radius**2 / 3)
                            * np.exp(-radius)
                        )
                    mean = (
                        cross @ model["alpha"] * model["target_scale"]
                        + model["target_mean"]
                    )
                    solved = np.linalg.solve(model["chol"], cross.T)
                    variance = (
                        np.maximum(
                            np.exp(2 * theta[-2]) - np.sum(solved**2, axis=0), 0
                        )[:, None]
                        * model["target_scale"] ** 2
                    )
                    np.testing.assert_allclose(
                        mean, prediction["mean"], atol=1e-9, rtol=1e-9
                    )
                    np.testing.assert_allclose(
                        variance, prediction["function_variance"], atol=1e-9, rtol=1e-9
                    )
                    np.testing.assert_allclose(
                        prediction["observation_variance"]
                        - prediction["function_variance"],
                        np.broadcast_to(
                            np.exp(2 * theta[-1]) * model["target_scale"] ** 2,
                            mean.shape,
                        ),
                        atol=1e-12,
                    )
                    assert np.all(np.isfinite(mean)) and np.all(variance >= 0)
                    maximum_difference = max(
                        maximum_difference,
                        float(np.max(abs(mean - prediction["mean"]))),
                    )
                    for region, metrics in report["models"][kernel]["metrics"].items():
                        if region == "slice-plateau":
                            continue
                        mask = data[f"region_{region}"]
                        error = prediction["mean"][mask] - data["latent"][mask]
                        rmse = float(np.sqrt(np.mean(error**2)))
                        assert abs(rmse - metrics["rmse"]) < 1e-12
                        observed_error = (
                            prediction["mean"][mask] - data["observed"][mask]
                        )
                        coverage = np.mean(
                            abs(observed_error)
                            <= 1.95996398454
                            * np.sqrt(prediction["observation_variance"][mask])
                        )
                        assert (
                            abs(coverage - metrics["observation_coverage_95"]) < 1e-12
                        )
                models += 1
    assert cases == expected_cases
    audit = {
        "passed": True,
        "models_replayed_with_numpy": models,
        "maximum_prediction_difference": maximum_difference,
        "checks": [
            "source hashes and archive CRC",
            "complete planned cases",
            "unique training rows with declared budgets",
            "retained base labels in acquisition arms",
            "training/test rows disjoint",
            "training-only normalization",
            "independent NumPy predictive means and variances",
            "observation noise separated from function uncertainty",
            "reported RMSE and observation coverage match saved arrays",
        ],
    }
    write_json(output / "audit.json", audit)
    return audit


def render(run, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = json.loads((run / "summary.json").read_text())
    plan = json.loads((run / "plan.json").read_text())
    kernels = plan["kernels"]
    colors = {"rbf": "#4477aa", "matern52": "#228833", "rq": "#cc6677"}
    names = {"rbf": "RBF", "matern52": "Matérn 5/2", "rq": "Multiple scales (RQ)"}
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.right": False, "axes.spines.top": False}
    )
    seed = plan["seeds"][0]
    layouts = ["outer", "random-fill", "coverage-fill"]
    titles = [
        "128 observations outside the gap",
        "Add 64 random observations",
        "Add 64 to fill input coverage",
    ]
    figure, axes = plt.subplots(2, 3, figsize=(12, 7), layout="constrained")
    for column, layout in enumerate(layouts):
        directory = run / f"dead-zone-{layout}-seed{seed}"
        for kernel in kernels:
            with np.load(directory / f"{kernel}-predictions.npz") as data:
                command = data["slice_command"][:, 0]
                mean = data["slice_mean"][:, 0]
                sigma = np.sqrt(data["slice_variance"][:, 0])
                axes[0, column].plot(
                    command, mean, color=colors[kernel], label=names[kernel]
                )
                axes[0, column].fill_between(
                    command,
                    mean - 1.96 * sigma,
                    mean + 1.96 * sigma,
                    color=colors[kernel],
                    alpha=0.09,
                )
                if kernel == kernels[0]:
                    axes[0, column].plot(
                        command,
                        data["slice_truth"][:, 0],
                        color="#222222",
                        linestyle="--",
                        label="True response",
                        linewidth=1.5,
                    )
        with np.load(directory / "samples.npz") as data:
            inputs = data["train_features"]
            axes[1, column].scatter(
                inputs[:128, 0],
                inputs[:128, 1],
                s=9,
                color="#777777",
                alpha=0.65,
                label="Original",
            )
            if len(inputs) > 128:
                axes[1, column].scatter(
                    inputs[128:, 0],
                    inputs[128:, 1],
                    s=14,
                    color="#dd9933",
                    label="Additional",
                )
        state = np.linspace(-1, 1, 200)
        width = 0.25 + 0.08 * np.sin(2 * state)
        axes[1, column].plot(state, width, color="#333333", linestyle="--", linewidth=1)
        axes[1, column].plot(
            state, -width, color="#333333", linestyle="--", linewidth=1
        )
        axes[1, column].set(
            xlabel="State", ylabel="Command", xlim=(-1, 1), ylim=(-1, 1)
        )
        axes[0, column].set(
            title=titles[column],
            xlabel="Command",
            ylabel="Next state at state = 0",
            ylim=(-0.66, 0.66),
        )
        axes[0, column].axvspan(-0.55, 0.55, color="#777777", alpha=0.05)
        for row in range(2):
            axes[row, column].grid(alpha=0.12)
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[1, 1].legend(frameon=False, fontsize=8)
    figure.suptitle(
        f"A dead zone learned from observations, without a dead-zone model\n"
        f"Synthetic state-dependent width · confirmation seed {seed} · bands: conditional 95% function intervals"
    )
    figure.savefig(output / "dead-zone-learning.png", dpi=180)
    figure.savefig(output / "dead-zone-learning.svg")
    plt.close(figure)

    layouts += ["broad"]
    labels = ["Outer\n128", "+ random\n192", "+ coverage\n192", "Broad\n192"]
    figure, axes = plt.subplots(2, 4, figsize=(14, 6.8), layout="constrained")
    for column, system in enumerate(plan["systems"]):
        for kernel in kernels:
            items = [
                next(
                    item
                    for item in summary
                    if item["system"] == system
                    and item["kernel"] == kernel
                    and item["layout"] == layout
                    and item["region"] == "shape-interior"
                )
                for layout in layouts
            ]
            for row, metric in enumerate(("rmse", "observation_coverage_95")):
                axes[row, column].plot(
                    range(4),
                    [item["mean"][metric] for item in items],
                    marker="o",
                    color=colors[kernel],
                    label=names[kernel],
                )
                axes[row, column].fill_between(
                    range(4),
                    [item["min"][metric] for item in items],
                    [item["max"][metric] for item in items],
                    color=colors[kernel],
                    alpha=0.08,
                )
        axes[0, column].set_title(system.capitalize())
        axes[0, column].set_ylim(bottom=0)
        axes[1, column].axhline(0.95, color="#444444", linestyle="--", linewidth=1)
        axes[1, column].set_ylim(0, 1.04)
        for row in range(2):
            axes[row, column].set_xticks(range(4), labels, fontsize=8)
            axes[row, column].grid(alpha=0.12)
    axes[0, 0].set_ylabel("RMSE in the shape interior")
    axes[1, 0].set_ylabel("Coverage of nominal 95% observation intervals")
    axes[0, 0].legend(frameon=False, fontsize=8)
    figure.suptitle(
        f"General shape learning on {len(plan['seeds'])} fresh seeds\n"
        "Lines: seed means · shading: seed range · each acquisition arm adds 64 counted observations"
    )
    figure.savefig(output / "shape-results.png", dpi=180)
    figure.savefig(output / "shape-results.svg")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, args.output / "report.source.py")
    audit = verify(args.run, args.output)
    render(args.run, args.output)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
