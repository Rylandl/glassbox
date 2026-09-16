"""Replay saved means independently in NumPy, verify data, and summarize results."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from experiment_model_structures import sequence_batches
from experiment_transition_diagnosis import load_records

from glassbox.experimental.sequence_model import SequenceModel
from glassbox.experimental.structured_regression import StructuredRegressor


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def kernel(left, right, kind, length):
    # Per-row vectorized implementation, independent of the training helper.
    result = []
    for query in left:
        distance = ((right - query) / length) ** 2
        components = np.exp(-0.5 * distance)
        if kind == "rbf":
            value = np.exp(-0.5 * distance.mean(1))
        elif kind == "additive" or right.shape[1] == 1:
            value = components.mean(1)
        else:
            # Cumulative products enumerate every unordered feature pair once.
            pairs = np.sum(components[:, 1:] * np.cumsum(components, axis=1)[:, :-1], 1)
            value = (
                components.mean(1) + pairs / (right.shape[1] * (right.shape[1] - 1) / 2)
            ) / 2
        result.append(value)
    return np.asarray(result)


def regression(model, x):
    a = model.arrays
    z = (x - a["feature_mean"]) / a["feature_scale"]
    mean = z @ a["linear"]
    if model.kind != "linear":
        mean += kernel(z, a["features"], model.kind, model.length_scale) @ a["alpha"]
    return mean * a["target_scale"] + a["target_mean"]


def recurrence(model, x, up, uf):
    """Loop over time; no training rollout, JAX scan, or future states used."""
    n, p = model.norms, model.params
    history = (x - n["state_mean"]) / n["state_scale"]
    commands = (up - n["input_mean"]) / n["input_scale"]
    future = (uf - n["input_mean"]) / n["input_scale"]
    hidden = np.empty((len(x), 0))
    if model.kind == "latent":
        encoded = np.column_stack(
            (history.reshape(len(x), -1), commands.reshape(len(x), -1))
        )
        hidden = np.tanh(encoded @ p["encoder"] + p["encoder_bias"])
    output = []
    for t in range(future.shape[1]):
        current, command = history[:, -1], future[:, t]
        features = [current, command]
        if model.kind in ("delay", "delay_mlp"):
            features.extend(
                (
                    (history[:, :-1] - current[:, None]).reshape(len(x), -1),
                    (commands - command[:, None]).reshape(len(x), -1),
                )
            )
        if model.kind == "latent":
            features.append(hidden)
        features = np.column_stack(features) / n["feature_scale"]
        delta = features @ p["linear"] + p["bias"]
        if model.kind in ("mlp", "latent", "delay_mlp"):
            delta += np.tanh(features @ p["w1"] + p["b1"]) @ p["w2"]
        predicted = current + n["delta_scale"] * delta
        if model.kind == "latent":
            hidden = np.tanh(features @ p["memory"] + p["memory_bias"])
        output.append(predicted * n["state_scale"] + n["state_mean"])
        history = np.concatenate((history[:, 1:], predicted[:, None]), axis=1)
        commands = np.concatenate((commands[:, 1:], command[:, None]), axis=1)
    return np.stack(output, axis=1)


def stats(values):
    return dict(
        mean=float(np.mean(values)),
        minimum=float(np.min(values)),
        maximum=float(np.max(values)),
    )


def main():
    # Plotting is only needed for the report; importing the audited helpers
    # such as recurrence() must not require Matplotlib.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--followup", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--corpus", type=Path, default=Path("../artifacts/real-transition/corpus")
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    maximum = 0.0
    comparisons = 0

    def check(a, b):
        nonlocal maximum, comparisons
        a, b = np.asarray(a), np.asarray(b)
        assert a.shape == b.shape, (a.shape, b.shape)
        if a.size:
            difference = float(np.max(np.abs(a - b)))
            maximum = max(maximum, difference)
        np.testing.assert_allclose(a, b, atol=2e-9, rtol=2e-9)
        comparisons += 1

    for run in (args.run, args.followup):
        with zipfile.ZipFile(run / "executed-sources.zip") as archive:
            for name, expected in read(run / "sources.json").items():
                assert hashlib.sha256(archive.read(name)).hexdigest() == expected
    records = load_records(args.corpus)
    assert all(r["inspection"]["checksum_matches_pinned_snapshot"] for r in records)
    raw_hashes = {r["name"]: r["inspection"]["sha256"] for r in records}
    direct, sequence, synthetic = (
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
    )
    model_count = 0
    for case in sorted((args.run / "direct").iterdir()):
        if not case.is_dir():
            continue
        report = read(case / "report.json")
        origin = Path(report["source"])
        for name, expected in report["hashes"].items():
            assert hashlib.sha256((origin / name).read_bytes()).hexdigest() == expected
        with (
            np.load(origin / "samples.npz") as samples,
            np.load(case / "predictions.npz") as saved,
        ):
            x = samples["train_features"]
            target = samples["train_next_states"] - samples["train_states"]
            for name, method in report["methods"].items():
                if name not in ("shared", "ridge", "hold"):
                    model = StructuredRegressor.load(case / f"{name}.npz")
                    assert model.fingerprint() == method["fingerprint"]
                    a = model.arrays
                    check(a["feature_mean"], x.mean(0))
                    check(a["target_mean"], target.mean(0))
                    z = (x - a["feature_mean"]) / a["feature_scale"]
                    y = (target - a["target_mean"]) / a["target_scale"]
                    check(a["features"], z)
                    check(
                        a["linear"],
                        np.linalg.solve(z.T @ z + np.eye(z.shape[1]), z.T @ y),
                    )
                    if name != "linear":
                        k = kernel(z, z, name, model.length_scale)
                        check(
                            (k + model.regularization * np.eye(len(x))) @ a["alpha"],
                            y - z @ a["linear"],
                        )
                        selected = min(
                            method["trials"],
                            key=lambda t: t["development_standardized_mse"],
                        )
                        assert selected["length_scale"] == model.length_scale
                        assert selected["regularization"] == model.regularization
                    model_count += 1
                for role in ("development", "replication"):
                    mean = saved[f"{name}_{role}"]
                    if name not in ("shared", "ridge", "hold"):
                        replay = samples[f"{role}_states"] + regression(
                            model, samples[f"{role}_features"]
                        )
                        check(replay, mean)
                        if name != "linear" and role == "development":
                            loss = np.mean(
                                (
                                    (mean - samples[f"{role}_next_states"])
                                    / a["target_scale"]
                                )
                                ** 2
                            )
                            check(loss, selected["development_standardized_mse"])
                    error = mean - samples[f"{role}_next_states"]
                    for field, slc in (
                        ("velocity_rmse_m_s", slice(0, 3)),
                        ("rate_rmse_rad_s", slice(3, 6)),
                    ):
                        metric = float(
                            np.sqrt(np.mean(np.sum(error[:, slc] ** 2, axis=1)))
                        )
                        check(metric, method["scores"][role]["all"][field])
                        key = f"{report['history']}/{report['horizon_steps']}/{name}/{role}/{field}"
                        direct[key].append(metric)

    for run in (args.run, args.followup):
        source = Path(read(run / "plan.json")["source"])
        for folder in sorted((run / "sequence").glob("seed*")):
            seed = int(folder.name.removeprefix("seed"))
            rebuilt, usage = sequence_batches(records, source, seed, False)
            assert usage == read(folder / "usage.json")
            with np.load(folder / "samples.npz") as saved_samples:
                for role, batch in rebuilt.items():
                    for name in (
                        "past_states",
                        "past_inputs",
                        "future_inputs",
                        "future_states",
                    ):
                        check(getattr(batch, name), saved_samples[f"{role}_{name}"])
            training = rebuilt["train"]
            current = np.concatenate(
                (training.past_states[:, -1:], training.future_states[:, :-1]), axis=1
            )
            for report_path in sorted(folder.glob("*-report.json")):
                report = read(report_path)
                name = report["name"]
                model = SequenceModel.load(folder / f"{name}.npz")
                assert model.fingerprint() == report["fingerprint"]
                check(model.norms["state_mean"], current.mean((0, 1)))
                check(
                    model.norms["state_scale"],
                    np.where(current.std((0, 1)) > 1e-8, current.std((0, 1)), 1),
                )
                check(model.norms["input_mean"], training.future_inputs.mean((0, 1)))
                selected = min(
                    report["trace"], key=lambda row: row["validation_rollout_mse"]
                )
                assert selected["step"] == report["selected_step"]
                with np.load(folder / f"{name}-predictions.npz") as saved:
                    for role in ("development", "replication"):
                        batch = rebuilt[role]
                        replay = recurrence(
                            model,
                            batch.past_states,
                            batch.past_inputs,
                            batch.future_inputs,
                        )
                        check(replay, saved[role])
                        if role == "development":
                            check(
                                np.mean(
                                    (
                                        (replay - batch.future_states)
                                        / model.norms["state_scale"]
                                    )
                                    ** 2
                                ),
                                report["validation_rollout_mse"],
                            )
                        for horizon in (1, 10, 25):
                            error = (
                                replay[:, horizon - 1, :6]
                                - batch.future_states[:, horizon - 1, :6]
                            )
                            for field, slc in (
                                ("velocity_rmse_m_s", slice(0, 3)),
                                ("rate_rmse_rad_s", slice(3, 6)),
                            ):
                                metric = float(
                                    np.sqrt(np.mean(np.sum(error[:, slc] ** 2, axis=1)))
                                )
                                check(
                                    metric,
                                    report["scores"][role][str(horizon)]["all"][field],
                                )
                                sequence[f"{horizon}/{name}/{role}/{field}"].append(
                                    metric
                                )
                model_count += 1

    for folder in sorted((args.run / "synthetic").glob("seed*")):
        report = read(folder / "report.json")
        with (
            np.load(folder / "samples.npz") as samples,
            np.load(folder / "predictions.npz") as saved,
        ):
            train = samples["train_features"]
            assert not np.any((train[:, 0] > 0) & (train[:, 1] > 0))
            combo = samples["combination_features"]
            assert np.all((combo[:, 0] > 0) & (combo[:, 1] > 0))
            assert np.all(samples["outside_features"][:, 1] > 2)
            for name, method in report["methods"].items():
                model = StructuredRegressor.load(folder / f"{name}.npz")
                for role in ("familiar", "combination", "outside"):
                    mean = regression(model, samples[f"{role}_features"])
                    check(mean, saved[f"{name}_{role}"])
                    metric = float(
                        np.sqrt(np.mean((mean - samples[f"{role}_targets"]) ** 2))
                    )
                    check(metric, method["rmse"][role])
                    synthetic[f"{name}/{role}"].append(metric)
                model_count += 1

    summary = {
        "direct": {k: stats(v) for k, v in direct.items()},
        "sequence": {k: stats(v) for k, v in sequence.items()},
        "synthetic": {k: stats(v) for k, v in synthetic.items()},
    }
    write(args.output / "summary.json", summary)
    write(
        args.output / "audit.json",
        dict(
            models_replayed=model_count,
            numeric_comparisons=comparisons,
            maximum_absolute_difference=maximum,
            raw_flight_hashes=raw_hashes,
            source_archives_verified=True,
            source_baseline_hashes_verified=True,
            sequence_windows_rebuilt=True,
            development_selection_replayed=True,
            caveat="This audits saved predictions and selection; it is not an independent rerun of optimization.",
        ),
    )
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    fields = ("velocity_rmse_m_s", "rate_rmse_rad_s")
    methods = (
        ("shared", "Previous GP"),
        ("linear", "Linear"),
        ("rbf", "Linear + RBF"),
        ("pairwise", "Linear + pairwise"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for row, history in enumerate(("20ms", "100ms")):
        for col, field in enumerate(fields):
            ax = axes[row, col]
            for method, label in methods:
                points = [
                    summary["direct"][f"{history}/{h}/{method}/replication/{field}"]
                    for h in (1, 10, 25)
                ]
                ax.errorbar(
                    [10, 100, 250],
                    [p["mean"] for p in points],
                    yerr=[
                        [p["mean"] - p["minimum"] for p in points],
                        [p["maximum"] - p["mean"] for p in points],
                    ],
                    marker="o",
                    capsize=3,
                    label=label,
                )
            ax.set(
                xscale="log",
                xticks=[10, 100, 250],
                xticklabels=[10, 100, 250],
                xlabel="Forecast horizon [ms]",
                ylabel="Velocity RMSE [m/s]" if col == 0 else "Body-rate RMSE [rad/s]",
                title=f"{history} history",
            )
            ax.set_ylim(bottom=0)
            ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=9)
    fig.suptitle(
        "Direct forecasts · identical features and endpoint labels\nRun-4 flights; means and sampling-seed ranges"
    )
    for extension in ("png", "svg"):
        fig.savefig(args.output / f"direct-structures.{extension}", dpi=160)
    plt.close(fig)

    methods = (
        ("mlp-teacher", "MLP · one-step fit"),
        ("mlp-rollout", "MLP · trajectory fit"),
        ("latent-rollout", "Latent memory · trajectory fit"),
        ("delay-teacher", "Linear history · both objectives"),
        ("delay_mlp-rollout", "Nonlinear history · trajectory fit"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), constrained_layout=True)
    for ax, field in zip(axes, fields):
        for method, label in methods:
            points = [
                summary["sequence"][f"{h}/{method}/replication/{field}"]
                for h in (1, 10, 25)
            ]
            ax.errorbar(
                [10, 100, 250],
                [p["mean"] for p in points],
                yerr=[
                    [p["mean"] - p["minimum"] for p in points],
                    [p["maximum"] - p["mean"] for p in points],
                ],
                marker="o",
                capsize=3,
                label=label,
            )
        ax.set(
            xscale="log",
            xticks=[10, 100, 250],
            xticklabels=[10, 100, 250],
            xlabel="Recursive forecast horizon [ms]",
            ylabel="Velocity RMSE [m/s]"
            if field == fields[0]
            else "Body-rate RMSE [rad/s]",
        )
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle(
        "Recursive forecasts · predict velocity, body rate and orientation\nRun-4 flights; means and sampling-seed ranges; includes adaptive history follow-up"
    )
    for extension in ("png", "svg"):
        fig.savefig(args.output / f"sequence-structures.{extension}", dpi=160)
    plt.close(fig)
    print(json.dumps(read(args.output / "audit.json")), flush=True)


if __name__ == "__main__":
    main()
