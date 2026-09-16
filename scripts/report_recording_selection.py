"""Independent replay of recording choices, row budgets, and new fitted forecasts."""

import argparse
import hashlib
import json
import runpy
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from report_forecast_representation import direct, raw_features
from report_model_structures import recurrence
from sequence_transfer_data import write_json

from glassbox.experimental.direct_forecast import DirectForecast
from glassbox.experimental.sequence_model import SequenceModel

ARRAYS = ("past_states", "past_inputs", "future_inputs", "future_states")


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rmse(pred, target, groups):
    return np.array(
        [
            [
                np.sqrt(np.mean(np.sum((pred[:, h, g] - target[:, h, g]) ** 2, -1)))
                for g in groups
            ]
            for h in range(pred.shape[1])
        ]
    )


def main(a):
    a.output.mkdir(parents=True, exist_ok=False)
    checks, largest, models, heads, decisions = 0, 0.0, 0, 0, 0

    def check(x, y):
        nonlocal checks, largest
        x, y = np.asarray(x), np.asarray(y)
        assert x.shape == y.shape
        largest = max(largest, float(np.max(np.abs(x - y))))
        np.testing.assert_allclose(x, y, atol=5e-8, rtol=5e-8)
        checks += 1

    with zipfile.ZipFile(a.run / "executed-sources.zip") as z:
        for path, h in read(a.run / "sources.json").items():
            assert hashlib.sha256(z.read(path)).hexdigest() == h
    decision_rows = []
    for folder in sorted((a.run / "selection").glob("*/*")):
        report = read(folder / "report.json")
        source_files = read(folder / "source-files.json")
        for path, h in source_files.items():
            assert sha(Path(path)) == h
        old_report = next(Path(p) for p in source_files if p.endswith("report.json"))
        source = old_report.parent
        old = read(old_report)
        usage = read(next(Path(p) for p in source_files if p.endswith("usage.json")))
        names, groups = report["names"], report["groups"]
        ids = {
            role: np.array(
                [item["recording"] for item in usage[role] for _ in item["origins"]]
            )
            for role in ("development", "evaluation")
        }
        with np.load(source / "samples.npz") as z:
            target = {
                role: z[f"{role}_future_states"][:, np.array(old["horizons"]) - 1]
                for role in ids
            }
        with np.load(source / "predictions.npz") as z:
            pred = {role: {n: z[f"{role}__{n}"] for n in names} for role in ids}
        records = list(report["recording_counts"])
        scale = np.array(old["state_scale"])
        mse = np.array(
            [
                [
                    [
                        [
                            (
                                np.square(
                                    (
                                        pred["development"][n][
                                            ids["development"] == r, h
                                        ][:, g]
                                        - target["development"][
                                            ids["development"] == r, h
                                        ][:, g]
                                    )
                                    / scale[g]
                                )
                            ).mean()
                            for g in groups
                        ]
                        for h in range(len(old["horizons"]))
                    ]
                    for r in records
                ]
                for n in names
            ]
        )
        counts = np.array([np.sum(ids["development"] == r) for r in records])
        widths = np.array([len(g) for g in groups])
        with np.load(folder / "evidence.npz") as z:
            check(mse, z["mse"])
            check(counts, z["counts"])
            check(widths, z["group_widths"])
        saved_metrics = read(folder / "recording-scores.json")
        for role in ids:
            for n in names:
                for r in dict.fromkeys(ids[role].tolist()):
                    mask = ids[role] == r
                    check(
                        rmse(pred[role][n][mask], target[role][mask], groups),
                        saved_metrics[role][n][r],
                    )
        for d in report["decisions"]:
            rec = [i for i, r in enumerate(records) if r != d["excluded"]]
            selected_mse = mse[:, rec]
            means = (selected_mse * (widths / widths.sum())).sum(-1).mean(-1)
            equal = means.mean(-1)
            worst = (
                selected_mse / np.maximum(selected_mse[names.index("hold")], 1e-12)
            ).max((1, 2, 3))
            scores = (
                np.average(means, axis=1, weights=counts[rec])
                if d["policy"] == "pooled"
                else worst
                if d["policy"] == "minimax_hold"
                else equal
            )
            eligible = (
                worst <= 1.05**2
                if d["policy"] == "guarded_hold"
                else np.ones(len(names), dtype=bool)
            )
            chosen = names[int(np.argmin(np.where(eligible, scores, np.inf)))]
            assert chosen == d["selected"]
            assert d["recordings"] == [records[i] for i in rec]
            assert d["eligible"] == [n for n, ok in zip(names, eligible) if ok]
            check(scores, [d["scores"][n] for n in names])
            check(worst, [d["worst_reference_mse_ratio"][n] for n in names])
            for r, metrics in d["tested"].items():
                check(metrics, saved_metrics[d["role"]][chosen][r])
                check(d["hold"][r], saved_metrics[d["role"]]["hold"][r])
            decisions += 1
            decision_rows.append(
                dict(dataset=report["dataset"], seed=report["seed"], **d)
            )
    # Rebuild source rows with the separately pinned official binary decoder.
    raw_root = a.previous / "new-corpus"
    decoder_path = raw_root / "cfusdlog.py"
    assert sha(decoder_path) == read(raw_root / "decoder-source.json")["sha256"]
    assert (
        sha(raw_root / "log10")
        == "da77c31ecf18a65159b7f973cf3246ec80ea2fc3e60183812416fad47f32b402"
    )
    record = runpy.run_path(str(decoder_path))["decode"](str(raw_root / "log10"))[
        "fixedFrequency"
    ]
    t = record["timestamp"] / 1000
    grid = np.arange(np.ceil(t[0] / 0.02), np.floor(t[-1] / 0.02) + 1) * 0.02
    indices = np.searchsorted(t, grid, side="right") - 1
    x = np.column_stack(
        [record[f"acc.{c}"][indices] for c in "xyz"]
        + [record[f"gyro.{c}"][indices] * np.pi / 180 for c in "xyz"]
    )
    u = np.column_stack([record[f"motor.m{i}"][indices] / 65536 for i in range(1, 5)])
    valid = (grid >= t[indices]) & (grid - t[indices] <= 0.01) & (u.mean(1) > 0.1)
    intervals = np.flatnonzero(np.diff(np.r_[False, valid, False].astype(int))).reshape(
        -1, 2
    )
    coverage_rows = []
    for folder in sorted((a.run / "coverage").glob("seed*")):
        usage = read(folder / "usage.json")
        report = read(folder / "fitted/report.json")
        origin = np.array(usage["source_origins"])
        expected_origins = []
        state_rows, input_rows = set(), set()
        for crop in usage["crops"]:
            start, stop = crop["crop_start"], crop["crop_stop"]
            assert [crop["eligible_start"], crop["eligible_stop"]] in intervals.tolist()
            assert crop["eligible_start"] <= start < stop <= crop["eligible_stop"]
            assert np.all(valid[start:stop])
            expected_origins.extend(range(start + 5, stop - 12))
        np.testing.assert_array_equal(origin, expected_origins)
        for key, row in zip(usage["keys"], origin):
            crop = next(
                c
                for c in usage["crops"]
                if key["segment_id"] == f"crop-{c['crop_start']}-{c['crop_stop']}"
            )
            assert (
                key["recording_id"] == "log10"
                and row == crop["crop_start"] + key["origin"]
            )
            state_rows.update(range(row - 5, row + 13))
            input_rows.update(range(row - 5, row + 12))
        assert len(origin) == len(np.unique(origin)) == 167
        assert len(state_rows) == usage["coverage"]["unique_state_rows"] == 337
        assert len(input_rows) == usage["coverage"]["unique_input_rows"] == 327
        check(usage["coverage"]["observed_transition_time_s"], 327 * 0.02)
        with np.load(folder / "fitted/samples.npz") as z:
            batches = {
                r: {k: z[f"{r}_{k}"] for k in ARRAYS}
                for r in ("train", "development", "evaluation")
            }
        train = batches["train"]
        for k, values, offset in [
            ("past_states", x, np.arange(-5, 1)),
            ("past_inputs", u, np.arange(-5, 0)),
            ("future_states", x, np.arange(1, 13)),
            ("future_inputs", u, np.arange(12)),
        ]:
            check(train[k], values[origin[:, None] + offset])
        with np.load(a.previous / "cf-frozen-01/fitted/samples.npz") as z:
            for k in ARRAYS:
                check(batches["development"][k], z[f"development_{k}"])
        with np.load(a.previous / "cf-evaluation-01/samples.npz") as z:
            for k in ARRAYS:
                check(batches["evaluation"][k], z[k])
        with np.load(folder / "fitted/predictions.npz") as z:
            predictions = {
                r: {k.split("__")[1]: z[k] for k in z.files if k.startswith(r + "__")}
                for r in ("development", "evaluation")
            }
        hidx = np.array(report["horizons"]) - 1
        for candidate in report["candidates"]:
            n = candidate["name"]
            cls = SequenceModel if n.startswith("recursive") else DirectForecast
            model = cls.load(folder / "fitted" / f"{n}.npz")
            assert model.fingerprint() == candidate["fingerprint"]
            if cls == DirectForecast:
                for h in report["horizons"]:
                    raw = raw_features(train, h, model.representation)
                    mean, std = raw.mean(0), raw.std(0)
                    std = np.where(std > 1e-8, std, 1)
                    check(mean, model.arrays[f"mean_{h}"])
                    check(std, model.arrays[f"scale_{h}"])
                    z = (raw - mean) / std
                    if f"projection_{h}" in model.arrays:
                        p = model.arrays[f"projection_{h}"]
                        s = model.arrays[f"singular_values_{h}"]
                        check(p.T @ p, np.eye(p.shape[1]))
                        check((z.T @ z) @ p, p * s[: p.shape[1]] ** 2)
                        check(np.sum(s**2), np.sum(z**2))
                        fraction = 0.95 if model.representation == "pca95" else 0.99
                        assert (
                            np.sum(s[: p.shape[1]] ** 2)
                            >= fraction * np.sum(s**2) - 1e-7
                        )
                        assert (
                            p.shape[1] == 1
                            or np.sum(s[: p.shape[1] - 1] ** 2)
                            < fraction * np.sum(s**2) + 1e-7
                        )
                        z = z @ p
                    weight, bias = (
                        model.arrays[f"weight_{h}"],
                        model.arrays[f"bias_{h}"],
                    )
                    y = train["future_states"][:, h - 1] - train["past_states"][:, -1]
                    check(bias, y.mean(0))
                    check(
                        z.T @ (z @ weight + bias - y) / len(z)
                        + model.ridge_fraction * weight,
                        np.zeros_like(weight),
                    )
                    heads += 1
            for role in ("development", "evaluation"):
                b = batches[role]
                p = (
                    direct(model, b)
                    if cls == DirectForecast
                    else recurrence(
                        model, b["past_states"], b["past_inputs"], b["future_inputs"]
                    )[:, hidx]
                )
                check(p, predictions[role][n])
            models += 1
        current = np.concatenate(
            (train["past_states"][:, -1:], train["future_states"][:, :-1]), 1
        )
        scale = current.std((0, 1))
        check(report["state_scale"], np.where(scale > 1e-8, scale, 1))
        for role, ps in predictions.items():
            past = batches[role]["past_states"]
            hold = np.repeat(past[:, -1:], len(hidx), 1)
            times = np.arange(past.shape[1], dtype=float)
            times -= times.mean()
            slope = (past * times[None, :, None]).sum(1) / (times @ times)
            check(hold, ps["hold"])
            check(
                hold + np.array(report["horizons"])[None, :, None] * slope[:, None],
                ps["trend"],
            )
            truth = batches[role]["future_states"][:, hidx]
            for name, p in ps.items():
                check(
                    rmse(p, truth, report["groups"]),
                    report["scores"][role][name]["rmse"],
                )
                check(
                    np.mean(((p - truth) / report["state_scale"]) ** 2),
                    report["scores"][role][name]["loss"],
                )
        dev = report["scores"]["development"]
        assert report["selected"] == min(dev, key=lambda n: dev[n]["loss"])
        for family, name in report["selected_families"].items():
            assert name == min(
                [c["name"] for c in report["candidates"] if c["family"] == family],
                key=lambda n: dev[n]["loss"],
            )
        coverage_rows.append(
            dict(
                seed=report["seed"],
                coverage=usage["coverage"],
                selected=report["selected"],
                arms={
                    arm: report["scores"]["evaluation"][n]["rmse"]
                    for arm, n in {
                        **report["selected_families"],
                        "selected": report["selected"],
                        "hold": "hold",
                    }.items()
                },
            )
        )
    summary = []
    for dataset in ("nano", "x8"):
        for policy in ("pooled", "equal_record", "minimax_hold", "guarded_hold"):
            all_rows = [
                r
                for r in decision_rows
                if r["dataset"] == dataset and r["policy"] == policy
            ]
            evaluation = [r for r in all_rows if r["excluded"] is None]
            loo = [r for r in all_rows if r["excluded"] is not None]
            item = dict(
                dataset=dataset,
                policy=policy,
                selected=[r["selected"] for r in evaluation],
                loo_choices_changed=sum(
                    r["selected"]
                    != next(x["selected"] for x in evaluation if x["seed"] == r["seed"])
                    for r in loo
                ),
                loo_choices=len(loo),
            )
            for label, rows in (
                ("evaluation", evaluation),
                ("leave_one_development_out", loo),
            ):
                values = np.array([v for r in rows for v in r["tested"].values()])
                hold = np.array([v for r in rows for v in r["hold"].values()])
                item[label] = dict(
                    mean_recording_rmse=values.mean(0).tolist(),
                    hold_mean_recording_rmse=hold.mean(0).tolist(),
                    physical_rmse_over_hold_count=(values > hold * 1.05)
                    .sum(0)
                    .tolist(),
                    recording_seed_pairs=len(values),
                )
            summary.append(item)
    write_json(a.output / "selection-summary.json", summary)
    write_json(a.output / "coverage-summary.json", coverage_rows)
    audit = dict(
        models_replayed=models,
        direct_heads_checked=heads,
        recording_choices_replayed=decisions,
        numerical_checks=checks,
        maximum_absolute_difference=largest,
        fixed_row_windows_checked=3 * 167,
        source_archives_verified=True,
        source_hashes_verified=True,
        interpretation="Independent numerical/selection replay and official-decoder row extraction; existing predictions retain prior audits. No untouched new recording, independent physical truth or calibrated uncertainty claim.",
    )
    write_json(a.output / "audit.json", audit)
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    policies = ("pooled", "equal_record", "minimax_hold", "guarded_hold")
    for ax, dataset in zip(axes, ("nano", "x8")):
        rows = [
            r
            for r in decision_rows
            if r["dataset"] == dataset and r["excluded"] is None
        ]
        records = list(rows[0]["tested"])
        for i, record_name in enumerate(records):
            ratio = [
                np.mean(
                    [
                        r["tested"][record_name][-1][1] / r["hold"][record_name][-1][1]
                        for r in rows
                        if r["policy"] == p
                    ]
                )
                for p in policies
            ]
            ax.plot(range(4), ratio, marker="o", label=f"Recording {i + 1}")
        ax.axhline(1, color="black", linestyle="--", label="Hold current")
        ax.set(
            title=dataset.upper(),
            xticks=range(4),
            xticklabels=["Pooled", "Equal recording", "Worst ratio", "Guarded"],
            ylabel="250 ms body-rate RMSE / hold-current RMSE",
            ylim=(0, None),
        )
        ax.tick_params(axis="x", labelrotation=15)
        ax.grid(axis="y", alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle(
        "Selection on development recordings · evaluation on separate recordings\nEach line is one recording, averaged over three reused training seeds",
        fontsize=11,
    )
    for ext in ("png", "svg"):
        fig.savefig(a.output / f"recording-selection.{ext}", dpi=170)
    plt.close(fig)
    print(json.dumps(audit), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run",
        type=Path,
        default=Path("../artifacts/recording-selection/comparison-01"),
    )
    p.add_argument(
        "--previous", type=Path, default=Path("../artifacts/representation-study")
    )
    p.add_argument("--output", type=Path, required=True)
    main(p.parse_args())
