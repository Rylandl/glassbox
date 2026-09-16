"""Independent saved-model replay and descriptive support/age error analysis."""

import argparse
import hashlib
import json
import zipfile
from itertools import pairwise
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from report_model_structures import recurrence
from scipy.stats import spearmanr
from sequence_transfer_data import load_prepared, write_json

from glassbox.experimental.direct_forecast import DirectForecast
from glassbox.experimental.sequence_model import SequenceModel

GROUPS = (tuple(range(3)), tuple(range(3, 6)), tuple(range(6, 15)))
CHANNELS = ("velocity", "body_rate", "rotation")
ROLES = ("train", "development", "evaluation")
ARRAYS = ("past_states", "past_inputs", "future_inputs", "future_states")


def read(p):
    return json.loads(p.read_text())


def direct_features(batch, h, history):
    x, u, v = batch["past_states"], batch["past_inputs"], batch["future_inputs"]
    columns = [x[:, -1], v[:, 0]]
    if history:
        columns += [
            np.column_stack([x[:, t] - x[:, -1] for t in range(x.shape[1] - 1)]),
            np.column_stack([u[:, t] - v[:, 0] for t in range(u.shape[1])]),
        ]
    columns += [v[:, t] - v[:, 0] for t in range(1, h)]
    return np.column_stack(columns)


def direct_prediction(model, batch):
    result = []
    for h in model.horizons:
        a = model.arrays
        result.append(
            batch["past_states"][:, -1]
            + (
                (direct_features(batch, h, model.use_history) - a[f"mean_{h}"])
                / a[f"scale_{h}"]
            )
            @ a[f"weight_{h}"]
            + a[f"bias_{h}"]
        )
    return np.stack(result, 1)


def nearest(query, reference, exclude=None):
    result = []
    for i, q in enumerate(query):
        distances = np.sqrt(np.mean((reference - q) ** 2, 1))
        if exclude is not None:
            distances = np.where(exclude[i], np.inf, distances)
        result.append(distances.min())
    return np.array(result)


def squared_errors(prediction, target):
    return np.stack(
        [
            np.sum((prediction[:, :, list(g)] - target[:, :, list(g)]) ** 2, -1)
            for g in GROUPS
        ],
        -1,
    )


def correlation(x, y):
    if len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return None
    return float(spearmanr(x, y).statistic)


def audit_and_report(args):
    args.output.mkdir(parents=True, exist_ok=False)
    checks, maximum, models, heads = 0, 0.0, 0, 0

    def check(a, b):
        nonlocal checks, maximum
        a, b = np.asarray(a), np.asarray(b)
        assert a.shape == b.shape
        maximum = max(maximum, float(np.max(np.abs(a - b))))
        np.testing.assert_allclose(a, b, atol=3e-8, rtol=3e-8)
        checks += 1

    with zipfile.ZipFile(args.source / "executed-sources.zip") as z:
        for p, h in read(args.source / "sources.json").items():
            assert hashlib.sha256(z.read(p)).hexdigest() == h
    records = {
        d: {r["name"]: r for r in load_prepared(args.prepared, d)}
        for d in ("x8", "arp")
    }
    for by_name in records.values():
        for r in by_name.values():
            assert (
                hashlib.sha256(Path(r["metadata"]["source"]).read_bytes()).hexdigest()
                == r["metadata"]["sha256"]
            )
    rows, diagnostics, composition = [], [], []
    for folder in sorted(args.source.glob("*/*")):
        if not (folder / "report.json").exists():
            continue
        report = read(folder / "report.json")
        dataset, fold, seed = report["dataset"], report["fold"], report["seed"]
        horizons = report["horizons"]
        indices = np.array(horizons) - 1
        usage = read(folder / "usage.json")
        with np.load(folder / "samples.npz") as z:
            batches = {r: {k: z[f"{r}_{k}"] for k in ARRAYS} for r in ROLES}
        p = batches["train"]["past_states"].shape[1] - 1
        role_names = [set(u["recording"] for u in usage[r]) for r in ROLES]
        assert not any(
            role_names[i] & role_names[j] for i in range(3) for j in range(i)
        )
        if dataset == "arp":
            names = sorted(records["arp"])
            assert role_names[2] == {names[fold]}
            assert role_names[1] == {names[(fold - 1) % 4]}
        if dataset == "nano":
            with np.load(args.previous / dataset / f"seed{seed}" / "samples.npz") as z:
                for role, b in batches.items():
                    for k, v in b.items():
                        check(v, z[f"{role}_{k}"])
        else:
            for role, b in batches.items():
                offset = 0
                for u in usage[role]:
                    r = records[dataset][u["recording"]]
                    a = np.array(u["origins"])
                    count = len(a)
                    for key, field, times in [
                        ("past_states", "states", np.arange(-p, 1)),
                        ("past_inputs", "inputs", np.arange(-p, 0)),
                        (
                            "future_inputs",
                            "inputs",
                            np.arange(b["future_inputs"].shape[1]),
                        ),
                        (
                            "future_states",
                            "states",
                            np.arange(1, b["future_states"].shape[1] + 1),
                        ),
                    ]:
                        check(
                            b[key][offset : offset + count],
                            r[field][a[:, None] + times],
                        )
                    offset += count
                assert offset == len(b["past_states"])
        train = batches["train"]
        current = np.concatenate(
            (train["past_states"][:, -1:], train["future_states"][:, :-1]), 1
        )
        state_scale = np.where(current.std((0, 1)) > 1e-8, current.std((0, 1)), 1)
        check(state_scale, report["state_scale"])
        with np.load(folder / "support.npz") as z:
            support = {k: z[k] for k in z.files}
        features = direct_features(train, 1, True)
        fm, fs = features.mean(0), features.std(0)
        fs = np.where(fs > 1e-8, fs, 1)
        check(fm, support["feature_mean"])
        check(fs, support["feature_scale"])
        features = (features - fm) / fs
        check(features, support["training_features"])
        groups = np.concatenate(
            [np.repeat(u["recording"], u["windows"]) for u in usage["train"]]
        )
        cross = nearest(features, features, groups[:, None] == groups[None, :])
        check(cross, support["training_cross_record_distance"])
        check(np.quantile(cross, [0.5, 0.9]), support["thresholds"])
        with np.load(folder / "predictions.npz") as z:
            saved = {
                r: {k.split("__")[1]: z[k] for k in z.files if k.startswith(r + "__")}
                for r in ("development", "evaluation")
            }
        target = {r: batches[r]["future_states"][:, indices] for r in saved}
        for candidate in report["candidates"]:
            name = candidate["name"]
            if name.startswith("direct"):
                model = DirectForecast.load(folder / f"{name}.npz")
                for h in horizons:
                    raw = direct_features(train, h, model.use_history)
                    a = model.arrays
                    check(a[f"mean_{h}"], raw.mean(0))
                    scale = np.where(raw.std(0) > 1e-8, raw.std(0), 1)
                    check(a[f"scale_{h}"], scale)
                    z = (raw - raw.mean(0)) / scale
                    delta = (
                        train["future_states"][:, h - 1] - train["past_states"][:, -1]
                    )
                    check(a[f"bias_{h}"], delta.mean(0))
                    normal = (
                        z.T @ (z @ a[f"weight_{h}"] + a[f"bias_{h}"] - delta) / len(z)
                        + candidate["ridge_fraction"] * a[f"weight_{h}"]
                    )
                    check(normal, np.zeros_like(normal))
                    heads += 1
                predicted = {r: direct_prediction(model, batches[r]) for r in saved}
            else:
                model = SequenceModel.load(folder / f"{name}.npz")
                predicted = {
                    r: recurrence(
                        model,
                        batches[r]["past_states"],
                        batches[r]["past_inputs"],
                        batches[r]["future_inputs"],
                    )[:, indices]
                    for r in saved
                }
            assert model.fingerprint() == candidate["fingerprint"]
            for role, pred in predicted.items():
                check(pred, saved[role][name])
            models += 1
        selection = report["selection"]
        all_names = list(selection["losses"])
        losses = {
            name: float(
                np.mean(
                    ((saved["development"][name] - target["development"]) / state_scale)
                    ** 2
                )
            )
            for name in all_names
        }
        for name, v in losses.items():
            check(v, selection["losses"][name])
        assert min(losses, key=losses.get) == selection["global_name"]
        matrix = np.array(
            [
                squared_errors(saved["development"][name], target["development"]).mean(
                    0
                )
                for name in all_names
            ]
        )
        cell_names = np.array(all_names)[matrix.argmin(0)]
        np.testing.assert_array_equal(cell_names, selection["cell_names"])
        for family, name in report["selected_families"].items():
            assert name == min(
                [c["name"] for c in report["candidates"] if c["family"] == family],
                key=losses.get,
            )
        for role, predictions in saved.items():
            b = batches[role]
            observed = b["past_states"]
            hold = np.repeat(observed[:, -1:], len(horizons), 1)
            t = np.arange(p + 1) - (p / 2)
            trend = hold + np.array(horizons)[None, :, None] * np.sum(
                observed * t[None, :, None], 1
            )[:, None] / np.sum(t * t)
            check(hold, predictions["hold"])
            check(trend, predictions["trend"])
            check(predictions[selection["global_name"]], predictions["selected_global"])
            combination = np.empty_like(hold)
            for i in range(len(horizons)):
                for j, g in enumerate(GROUPS):
                    combination[:, i, list(g)] = predictions[cell_names[i, j]][
                        :, i, list(g)
                    ]
            check(combination, predictions["selected_cells"])
            dist = nearest((direct_features(b, 1, True) - fm) / fs, features)
            check(dist, support[f"{role}_distance"])
            record_ids = np.concatenate(
                [np.repeat(u["recording"], u["windows"]) for u in usage[role]]
            )
            np.testing.assert_array_equal(record_ids, support[f"{role}_recordings"])
            if dataset == "arp":
                for kind, field in [
                    ("sample_age", "sample_s"),
                    ("publication_age", "publication_s"),
                ]:
                    ages = np.concatenate(
                        [
                            records["arp"][u["recording"]]["absolute_grid_s"][
                                u["origins"], None
                            ]
                            - records["arp"][u["recording"]][field][u["origins"]]
                            for u in usage[role]
                        ]
                    )
                    check(ages, support[f"{role}_{kind}"])
            for name, pred in predictions.items():
                check(
                    np.sqrt(squared_errors(pred, target[role]).mean(0)),
                    report["scores"][role][name]["rmse"],
                )
                check(
                    np.mean(((pred - target[role]) / state_scale) ** 2),
                    report["scores"][role][name]["loss"],
                )
            if role != "evaluation":
                continue
            query = (direct_features(b, 1, True) - fm) / fs
            closest = np.array(
                [
                    features[np.argmin(np.sum((features - q) ** 2, axis=1))]
                    for q in query
                ]
            )
            squared = (query - closest) ** 2
            width = b["future_inputs"].shape[2]
            boundaries = (
                0,
                3,
                6,
                15,
                15 + width,
                15 + width + p * 15,
                squared.shape[1],
            )
            blocks = (
                "current_velocity",
                "current_body_rate",
                "current_rotation",
                "current_input",
                "observation_history",
                "input_history",
            )
            mass = np.array(
                [squared[:, start:stop].sum() for start, stop in pairwise(boundaries)]
            )
            composition.append(
                dict(
                    dataset=dataset,
                    fold=fold,
                    seed=seed,
                    far_fraction=float(np.mean(dist > support["thresholds"][1])),
                    fraction_of_total_squared_distance=dict(
                        zip(blocks, (mass / mass.sum()).tolist())
                    ),
                    interpretation="Contributions to standardized nearest-neighbor distance, not contributions to prediction error.",
                )
            )
            arms = {
                "hold": "hold",
                "trend": "trend",
                **report["selected_families"],
                "selected_global": "selected_global",
                "selected_cells": "selected_cells",
            }
            for arm, name in arms.items():
                error = squared_errors(predictions[name], target[role])
                base = squared_errors(hold, target[role])
                rows.append(
                    dict(
                        dataset=dataset,
                        fold=fold,
                        seed=seed,
                        arm=arm,
                        candidate=name,
                        horizons=horizons,
                        rmse=np.sqrt(error.mean(0)).tolist(),
                        recordings=sorted(set(record_ids)),
                    )
                )
                for i, h in enumerate(horizons):
                    for j, channel in enumerate(CHANNELS):
                        values = dict(
                            dataset=dataset,
                            fold=fold,
                            seed=seed,
                            arm=arm,
                            horizon=h,
                            channel=channel,
                            count=len(error),
                            support_far_fraction=float(
                                np.mean(dist > support["thresholds"][1])
                            ),
                            support_error_rank_correlation=correlation(
                                dist, error[:, i, j]
                            ),
                            support_excess_error_rank_correlation=correlation(
                                dist, error[:, i, j] - base[:, i, j]
                            ),
                            bins=[],
                        )
                        for label, mask in [
                            ("near", dist <= support["thresholds"][0]),
                            (
                                "middle",
                                (dist > support["thresholds"][0])
                                & (dist <= support["thresholds"][1]),
                            ),
                            ("far", dist > support["thresholds"][1]),
                        ]:
                            if mask.any():
                                values["bins"].append(
                                    dict(
                                        label=label,
                                        count=int(mask.sum()),
                                        rmse=float(np.sqrt(error[mask, i, j].mean())),
                                        hold_rmse=float(
                                            np.sqrt(base[mask, i, j].mean())
                                        ),
                                    )
                                )
                        if dataset == "arp":
                            age = support[f"{role}_sample_age"][:, j]
                            motion = np.linalg.norm(
                                observed[:, -1, list(GROUPS[j])]
                                - observed[:, 0, list(GROUPS[j])],
                                axis=1,
                            )
                            values.update(
                                sample_age_ms=np.quantile(age, [0, 0.5, 1])
                                .__mul__(1000)
                                .tolist(),
                                age_error_rank_correlation=correlation(
                                    age, error[:, i, j]
                                ),
                                age_excess_error_rank_correlation=correlation(
                                    age, error[:, i, j] - base[:, i, j]
                                ),
                                motion_error_rank_correlation=correlation(
                                    motion, error[:, i, j]
                                ),
                            )
                        diagnostics.append(values)
    summary = []
    for dataset in ("nano", "x8", "arp"):
        for fold in sorted({r["fold"] for r in rows if r["dataset"] == dataset}):
            for arm in sorted({r["arm"] for r in rows}):
                group = [
                    r
                    for r in rows
                    if r["dataset"] == dataset and r["fold"] == fold and r["arm"] == arm
                ]
                a = np.array([r["rmse"] for r in group])
                summary.append(
                    dict(
                        dataset=dataset,
                        fold=fold,
                        arm=arm,
                        horizons=group[0]["horizons"],
                        mean=a.mean(0).tolist(),
                        minimum=a.min(0).tolist(),
                        maximum=a.max(0).tolist(),
                    )
                )
    write_json(args.output / "summary.json", summary)
    write_json(args.output / "case-metrics.json", rows)
    write_json(args.output / "diagnostics.json", diagnostics)
    write_json(args.output / "support-composition.json", composition)
    write_json(
        args.output / "audit.json",
        dict(
            models_replayed=models,
            direct_heads_normal_equations_checked=heads,
            numerical_checks=checks,
            maximum_absolute_difference=maximum,
            source_archive_verified=True,
            recording_splits_verified=True,
            source_hashes_verified=True,
            support_rebuilt_with_independent_distance_loops=True,
            selection_uses_development_only=True,
            interpretation="Saved prediction and selection replay, direct normal equations, extraction against prepared records. Prior sequence-transfer audit validates the raw-clock extraction; this is not independent physical truth.",
        ),
    )
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    arms = ("recursive_history", "direct_history", "hold", "selected_cells")
    labels = (
        "Recursive history",
        "Direct history",
        "Hold current",
        "Select per group/horizon",
    )
    for j, ax in enumerate(axes):
        for k, (arm, label) in enumerate(zip(arms, labels)):
            points = [
                next(
                    s
                    for s in summary
                    if s["dataset"] == "arp" and s["fold"] == f and s["arm"] == arm
                )
                for f in range(4)
            ]
            mean = np.array([s["mean"][-1][j] for s in points])
            lo = np.array([s["minimum"][-1][j] for s in points])
            hi = np.array([s["maximum"][-1][j] for s in points])
            ax.bar(
                np.arange(4) + (k - 1.5) * 0.2,
                mean,
                0.2,
                label=label,
                yerr=[mean - lo, hi - mean],
                capsize=2,
            )
        ax.set(
            xticks=np.arange(4),
            xticklabels=["Log 63", "Log 64", "Log 65", "Log 66"],
            xlabel="Held-out recording",
            ylabel=["Velocity RMSE [m/s]", "Body-rate RMSE [rad/s]"][j],
            ylim=(0, None),
        )
        ax.grid(axis="y", alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.suptitle(
        "Causal ARP observations · 240 ms conditional forecasts\nSame 384 training origins per fit; means and sampling-seed ranges",
        fontsize=12,
    )
    for ext in ("png", "svg"):
        fig.savefig(args.output / f"arp-forecast.{ext}", dpi=170)
    plt.close(fig)
    print(json.dumps(read(args.output / "audit.json")), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--prepared",
        type=Path,
        default=Path("../artifacts/sequence-transfer/prepared-02"),
    )
    parser.add_argument(
        "--previous",
        type=Path,
        default=Path("../artifacts/sequence-transfer/comparison-01"),
    )
    parser.add_argument("--output", type=Path, required=True)
    audit_and_report(parser.parse_args())
