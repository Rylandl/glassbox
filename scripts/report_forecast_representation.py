"""Replay representation fits, reserved selection, and raw Crazyflie observations."""

import argparse
import hashlib
import json
import runpy
import zipfile
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from report_model_structures import recurrence
from sequence_transfer_data import write_json

from glassbox.experimental.direct_forecast import DirectForecast
from glassbox.experimental.sequence_model import SequenceModel

ARRAYS = ("past_states", "past_inputs", "future_inputs", "future_states")


def read(path):
    return json.loads(path.read_text())


def raw_features(b, h, mode):
    x, up, uf = b["past_states"], b["past_inputs"], b["future_inputs"]
    n, p, d = x.shape
    p -= 1
    parts = [] if mode == "relative" else [x[:, -1]]
    parts.append(uf[:, 0])
    dx = x[:, :-1] - x[:, -1:]
    du = up - uf[:, :1]
    if mode in ("linear_history", "quadratic_history"):
        order = 1 if mode == "linear_history" else 2
        t = np.arange(-p, 0, dtype=float) / p
        basis = np.array([t**k for k in range(1, order + 1)]).T
        operator = np.linalg.solve(basis.T @ basis, basis.T)
        dx = np.stack(
            [
                np.stack([operator @ history[:, j] for j in range(d)], 1)
                for history in dx
            ]
        )
        du = np.stack(
            [
                np.stack([operator @ history[:, j] for j in range(up.shape[2])], 1)
                for history in du
            ]
        )
    parts.extend([dx.reshape(n, -1), du.reshape(n, -1)])
    parts.extend(uf[:, t] - uf[:, 0] for t in range(1, h))
    return np.column_stack(parts)


def direct(model, b):
    result = []
    for h in model.horizons:
        a = model.arrays
        z = (raw_features(b, h, model.representation) - a[f"mean_{h}"]) / a[
            f"scale_{h}"
        ]
        if f"projection_{h}" in a:
            z = z @ a[f"projection_{h}"]
        result.append(b["past_states"][:, -1] + z @ a[f"weight_{h}"] + a[f"bias_{h}"])
    return np.stack(result, 1)


def vector_error(p, y, groups):
    return np.stack(
        [np.sum((p[:, :, list(g)] - y[:, :, list(g)]) ** 2, -1) for g in groups], -1
    )


def local_replay(train, dev, dev_errors, query, predictions, groups):
    mean, std = train.mean(0), train.std(0)
    std = np.where(std > 1e-8, std, 1)
    calibrated = (dev - mean) / std
    query = (query - mean) / std
    result = np.empty_like(predictions[0])
    indices = []
    estimated = []
    choices = []
    for i, q in enumerate(query):
        neighbors = np.argsort(np.sum((calibrated - q) ** 2, 1), kind="stable")[
            : min(32, len(dev))
        ]
        cost = dev_errors[:, neighbors].mean(1)
        selected = cost.argmin(0)
        for h in range(result.shape[1]):
            for g, columns in enumerate(groups):
                result[i, h, list(columns)] = predictions[
                    selected[h, g], i, h, list(columns)
                ]
        indices.append(neighbors)
        estimated.append(cost)
        choices.append(selected)
    return result, np.array(indices), np.array(estimated), np.array(choices)


def main(a):
    a.output.mkdir(parents=True, exist_ok=False)
    checks, maximum, models, heads = 0, 0.0, 0, 0

    def check(x, y):
        nonlocal checks, maximum
        x, y = np.asarray(x), np.asarray(y)
        assert x.shape == y.shape
        maximum = max(maximum, float(np.max(np.abs(x - y))) if x.size else 0)
        np.testing.assert_allclose(x, y, atol=5e-8, rtol=5e-8)
        checks += 1

    root = a.root
    for run in ["ablation-01", "cf-frozen-01", "cf-expanded-01"]:
        with zipfile.ZipFile(root / run / "executed-sources.zip") as z:
            for name, digest in read(root / run / "sources.json").items():
                assert hashlib.sha256(z.read(name)).hexdigest() == digest
    frozen = read(root / "cf-frozen-01/frozen.json")
    evaluation = read(root / "cf-evaluation-01/report.json")
    for p, h in frozen["files"].items():
        assert hashlib.sha256((root / "cf-frozen-01" / p).read_bytes()).hexdigest() == h
    assert datetime.fromisoformat(frozen["created_utc"]) < datetime.fromisoformat(
        evaluation["evaluated_utc"]
    )
    assert (
        evaluation["frozen_sha256"]
        == hashlib.sha256((root / "cf-frozen-01/frozen.json").read_bytes()).hexdigest()
    )
    # Independent official decoder, previously inspected; pin its complete bytes.
    decoder = root / "new-corpus/cfusdlog.py"
    assert (
        hashlib.sha256(decoder.read_bytes()).hexdigest()
        == read(root / "new-corpus/decoder-source.json")["sha256"]
    )
    decode = runpy.run_path(str(decoder))["decode"]
    independent_decode = runpy.run_path(
        str(Path(__file__).with_name("prepare_crazyflie_reference.py"))
    )["decode"]
    raw_records = {}
    raw_rows = 0
    source_hashes = {
        Path(r["path"]).name: r["sha256"]
        for r in read(root / "new-corpus/sources.json")
    }
    for name in ("log10", "log15", "log16"):
        rawpath = root / "new-corpus" / name
        assert hashlib.sha256(rawpath.read_bytes()).hexdigest() == source_hashes[name]
        r = decode(str(rawpath))["fixedFrequency"]
        t = r["timestamp"] / 1000
        prepared_raw = independent_decode(rawpath)["fixedFrequency"]
        check(t, prepared_raw["timestamp_s"])
        for field in r:
            if field != "timestamp":
                check(r[field], prepared_raw[field])
        raw_rows += len(t)
        values = np.column_stack(
            [r[f"acc.{c}"] for c in "xyz"]
            + [r[f"gyro.{c}"] for c in "xyz"]
            + [r[f"motor.m{i}"] for i in range(1, 5)]
        )
        grid = np.arange(np.ceil(t[0] / 0.02), np.floor(t[-1] / 0.02) + 1) * 0.02
        index = np.searchsorted(t, grid, side="right") - 1
        raw_records[name] = (grid, values[index], grid - t[index])
        with np.load(root / "cf-complete" / f"{name}.npz") as z:
            idx = z["source_indices"]
            g = z["absolute_grid_s"]
            np.testing.assert_array_equal(idx, np.searchsorted(t, g, side="right") - 1)
            assert np.all(t[idx] <= g) and np.all(g - t[idx] <= 0.01)
            check(z["event_s"], t[idx])
            check(
                z["states"],
                np.column_stack((values[idx, :3], values[idx, 3:6] * np.pi / 180)),
            )
            check(z["inputs"], values[idx[:-1], 6:] / 65536)
    cases = sorted((root / "ablation-01").glob("*/*")) + [
        root / "cf-frozen-01/fitted",
        root / "cf-expanded-01/fitted",
    ]
    summaries = []
    for case in cases:
        report = read(case / "report.json")
        horizons = report["horizons"]
        hidx = np.array(horizons) - 1
        groups = report["groups"]
        with np.load(case / "samples.npz") as z:
            batches = {
                role: {k: z[f"{role}_{k}"] for k in ARRAYS}
                for role in ("train", "development", "evaluation")
                if f"{role}_past_states" in z
            }
        with np.load(case / "predictions.npz") as z:
            preds = {
                role: {
                    k.split("__")[1]: z[k] for k in z.files if k.startswith(role + "__")
                }
                for role in ("development", "evaluation")
                if f"{role}__hold" in z
            }
        if report["dataset"] in ("arp", "nano", "x8"):
            prior = a.previous / report["dataset"] / case.name
            with np.load(prior / "samples.npz") as z:
                for role, b in batches.items():
                    for k, v in b.items():
                        check(v, z[f"{role}_{k}"])
            with np.load(prior / "predictions.npz") as z:
                for role, p in preds.items():
                    for n, v in p.items():
                        if n.startswith("full-"):
                            check(
                                v,
                                z[role + "__" + n.replace("full-", "direct_history-")],
                            )
                        elif n.startswith("recursive-"):
                            check(
                                v,
                                z[
                                    role
                                    + "__"
                                    + n.replace("recursive-", "recursive_history-")
                                ],
                            )
        else:
            usage = read(root / "cf-frozen-01/usage.json")
            if report["dataset"].endswith("expanded"):
                expanded = read(root / "cf-expanded-01/usage.json")
                chosen = np.array(expanded["segment_and_origin"])
                assert len(chosen) == usage["train"]["windows"] == 320
                assert np.all(chosen[:, 2] - 5 >= chosen[:, 0]) and np.all(
                    chosen[:, 2] + 12 < chosen[:, 1]
                )
                grid, data, age = raw_records["log10"]
                origin = chosen[:, 2]
                fullx = np.column_stack((data[:, :3], data[:, 3:6] * np.pi / 180))
                fullu = data[:, 6:] / 65536
                valid = (fullu.mean(1) > 0.1) & (age >= 0) & (age <= 0.01)
                for start, stop, _ in chosen:
                    assert np.all(valid[start:stop])
                for k, x, offset in [
                    ("past_states", fullx, np.arange(-5, 1)),
                    ("past_inputs", fullu, np.arange(-5, 0)),
                    ("future_states", fullx, np.arange(1, 13)),
                    ("future_inputs", fullu, np.arange(12)),
                ]:
                    check(batches["train"][k], x[origin[:, None] + offset])
            for role, b in batches.items():
                if role == "train" and report["dataset"].endswith("expanded"):
                    continue
                u = (
                    usage[role]
                    if role in usage
                    else read(root / "cf-evaluation-01/usage.json")
                )
                with np.load(root / "cf-complete" / f"{u['recording']}.npz") as z:
                    origin = np.array(u["origins"])
                    for k, field, offset in [
                        ("past_states", "states", np.arange(-5, 1)),
                        ("past_inputs", "inputs", np.arange(-5, 0)),
                        ("future_states", "states", np.arange(1, 13)),
                        ("future_inputs", "inputs", np.arange(12)),
                    ]:
                        check(b[k], z[field][origin[:, None] + offset])
        train = batches["train"]
        x0 = train["past_states"][:, -1]
        for candidate in report["candidates"]:
            name = candidate["name"]
            ridge = float(name.split("-r")[1])
            if name.startswith("recursive"):
                m = SequenceModel.load(case / f"{name}.npz")
                prediction = {
                    role: recurrence(
                        m, b["past_states"], b["past_inputs"], b["future_inputs"]
                    )[:, hidx]
                    for role, b in batches.items()
                    if role in preds
                }
            else:
                m = DirectForecast.load(case / f"{name}.npz")
                for h in horizons:
                    raw = raw_features(train, h, m.representation)
                    mean = raw.mean(0)
                    std = np.where(raw.std(0) > 1e-8, raw.std(0), 1)
                    check(mean, m.arrays[f"mean_{h}"])
                    check(std, m.arrays[f"scale_{h}"])
                    z = (raw - mean) / std
                    if f"projection_{h}" in m.arrays:
                        p = m.arrays[f"projection_{h}"]
                        s = m.arrays[f"singular_values_{h}"]
                        rank = p.shape[1]
                        check(p.T @ p, np.eye(rank))
                        check((z.T @ z) @ p, p * (s[:rank] ** 2))
                        fraction = 0.95 if m.representation == "pca95" else 0.99
                        assert np.sum(s[:rank] ** 2) >= fraction * np.sum(s**2) - 1e-7
                        assert (
                            rank == 1
                            or np.sum(s[: rank - 1] ** 2)
                            < fraction * np.sum(s**2) + 1e-7
                        )
                        z = z @ p
                    target = train["future_states"][:, h - 1] - x0
                    weight = m.arrays[f"weight_{h}"]
                    bias = m.arrays[f"bias_{h}"]
                    check(bias, target.mean(0))
                    normal = (
                        z.T @ (z @ weight + bias - target) / len(z) + ridge * weight
                    )
                    check(normal, np.zeros_like(normal))
                    heads += 1
                prediction = {
                    role: direct(m, b) for role, b in batches.items() if role in preds
                }
            assert m.fingerprint() == candidate["fingerprint"]
            for role, p in prediction.items():
                check(p, preds[role][name])
            models += 1
        scale = np.array(report["state_scale"])
        current = np.concatenate(
            (train["past_states"][:, -1:], train["future_states"][:, :-1]), 1
        )
        check(scale, np.where(current.std((0, 1)) > 1e-8, current.std((0, 1)), 1))
        losses = {}
        for role, predictions in preds.items():
            b = batches[role]
            target = b["future_states"][:, hidx]
            hold = np.repeat(b["past_states"][:, -1:], len(horizons), 1)
            t = np.arange(b["past_states"].shape[1], dtype=float)
            t -= t.mean()
            trend = (
                hold
                + np.array(horizons)[None, :, None]
                * (np.sum(b["past_states"] * t[None, :, None], 1) / np.sum(t * t))[
                    :, None
                ]
            )
            check(hold, predictions["hold"])
            check(trend, predictions["trend"])
            for name, pred in predictions.items():
                loss = float(np.mean(((pred - target) / scale) ** 2))
                rmse = np.sqrt(vector_error(pred, target, groups).mean(0))
                check(loss, report["scores"][role][name]["loss"])
                check(rmse, report["scores"][role][name]["rmse"])
                if role == "development":
                    losses[name] = loss
        assert report["selected"] == min(losses, key=losses.get)
        for family, name in report["selected_families"].items():
            assert name == min(
                [c["name"] for c in report["candidates"] if c["family"] == family],
                key=losses.get,
            )
        with np.load(case / "local-reference.npz") as z:
            reference = {k: z[k] for k in z.files}
        check(reference["training_features"], raw_features(train, 1, "full"))
        check(
            reference["development_features"],
            raw_features(batches["development"], 1, "full"),
        )
        pool = report["local_pool"]
        devtarget = batches["development"]["future_states"][:, hidx]
        check(
            reference["development_errors"],
            np.array(
                [vector_error(preds["development"][n], devtarget, groups) for n in pool]
            ),
        )
        if "evaluation" in batches:
            expected, idx, cost, chosen = local_replay(
                reference["training_features"],
                reference["development_features"],
                reference["development_errors"],
                raw_features(batches["evaluation"], 1, "full"),
                np.array([preds["evaluation"][n] for n in pool]),
                groups,
            )
            with np.load(case / "local-predictions.npz") as z:
                check(expected, z["prediction"])
                check(cost, z["estimated_squared_error"])
                np.testing.assert_array_equal(idx, z["indices"])
                np.testing.assert_array_equal(chosen, z["choices"])
            check(
                np.sqrt(
                    vector_error(
                        expected,
                        batches["evaluation"]["future_states"][:, hidx],
                        groups,
                    ).mean(0)
                ),
                report["scores"]["evaluation"]["local_selector"]["rmse"],
            )
            check(
                np.mean(
                    (
                        (expected - batches["evaluation"]["future_states"][:, hidx])
                        / scale
                    )
                    ** 2
                ),
                report["scores"]["evaluation"]["local_selector"]["loss"],
            )
            arms = {
                **report["selected_families"],
                "hold": "hold",
                "selected": report["selected"],
                "local_selector": "local_selector",
            }
            summaries.append(
                dict(
                    dataset=report["dataset"],
                    fold=report.get("fold"),
                    seed=report["seed"],
                    selected=report["selected"],
                    arms={
                        arm: report["scores"]["evaluation"][name]["rmse"]
                        for arm, name in arms.items()
                    },
                )
            )
    # Reserved log16 inference from the frozen artifacts, without reselecting.
    report = read(root / "cf-frozen-01/fitted/report.json")
    groups = report["groups"]
    hidx = np.array(report["horizons"]) - 1
    with np.load(root / "cf-evaluation-01/samples.npz") as z:
        b = {k: z[k] for k in ARRAYS}
    usage = read(root / "cf-evaluation-01/usage.json")
    with np.load(root / "cf-complete/log16.npz") as z:
        origin = np.array(usage["origins"])
        for k, field, offset in [
            ("past_states", "states", np.arange(-5, 1)),
            ("past_inputs", "inputs", np.arange(-5, 0)),
            ("future_states", "states", np.arange(1, 13)),
            ("future_inputs", "inputs", np.arange(12)),
        ]:
            check(b[k], z[field][origin[:, None] + offset])
    with np.load(root / "cf-evaluation-01/predictions.npz") as z:
        actual = {k: z[k] for k in z.files}
    hold = np.repeat(b["past_states"][:, -1:], len(hidx), 1)
    t = np.arange(b["past_states"].shape[1], dtype=float)
    t -= t.mean()
    slope = np.sum(b["past_states"] * t[None, :, None], 1) / np.sum(t * t)
    check(hold, actual["hold"])
    check(
        hold + np.array(report["horizons"])[None, :, None] * slope[:, None],
        actual["trend"],
    )
    for arm, name in evaluation["names"].items():
        if name in ("hold", "trend"):
            continue
        cls = SequenceModel if name.startswith("recursive") else DirectForecast
        m = cls.load(root / "cf-frozen-01/fitted" / f"{name}.npz")
        predicted = (
            direct(m, b)
            if cls == DirectForecast
            else recurrence(m, b["past_states"], b["past_inputs"], b["future_inputs"])[
                :, hidx
            ]
        )
        check(predicted, actual[arm])
    pool = report["local_pool"]
    by_name = {name: actual[arm] for arm, name in evaluation["names"].items()}
    with np.load(root / "cf-frozen-01/fitted/local-reference.npz") as ref:
        pred, idx, cost, chosen = local_replay(
            ref["training_features"],
            ref["development_features"],
            ref["development_errors"],
            raw_features(b, 1, "full"),
            np.array([by_name[n] for n in pool]),
            groups,
        )
    check(pred, actual["local_selector"])
    with np.load(root / "cf-evaluation-01/local-selection.npz") as z:
        check(cost, z["estimated_squared_error"])
        np.testing.assert_array_equal(idx, z["indices"])
        np.testing.assert_array_equal(chosen, z["choices"])
    for arm, p in actual.items():
        check(
            np.mean(
                ((p - b["future_states"][:, hidx]) / np.array(report["state_scale"]))
                ** 2
            ),
            evaluation["scores"][arm]["loss"],
        )
        check(
            np.sqrt(vector_error(p, b["future_states"][:, hidx], groups).mean(0)),
            evaluation["scores"][arm]["rmse"],
        )
    write_json(a.output / "case-metrics.json", summaries)
    aggregated = []
    for dataset in ("arp", "nano", "x8"):
        for fold in sorted({r["fold"] for r in summaries if r["dataset"] == dataset}):
            cases = [
                r for r in summaries if r["dataset"] == dataset and r["fold"] == fold
            ]
            aggregated.append(
                dict(
                    dataset=dataset,
                    fold=fold,
                    arms={
                        arm: np.mean([r["arms"][arm] for r in cases], 0).tolist()
                        for arm in cases[0]["arms"]
                    },
                )
            )
    write_json(a.output / "summary.json", aggregated)
    write_json(a.output / "reserved-evaluation.json", evaluation)
    write_json(
        a.output / "audit.json",
        dict(
            models_replayed=models,
            direct_heads_checked=heads,
            numerical_checks=checks,
            maximum_absolute_difference=maximum,
            official_decoder_rows_checked=raw_rows,
            baseline_predictions_reproduced=True,
            frozen_artifacts_verified=True,
            expanded_training_origins_checked=320,
            interpretation="Independent saved-prediction and direct normal-equation replay; PCA training covariance checks; official decoder comparison and as-of row checks. No independent physical ground truth or uncertainty calibration.",
        ),
    )
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    expanded = next(r for r in summaries if r["dataset"] == "crazyflie_sensor_expanded")
    labels = [
        "Original full",
        "Original selected",
        "Broader full",
        "Broader selected",
        "Broader local",
    ]
    points = [
        evaluation["scores"]["full"]["rmse"][-1],
        evaluation["scores"]["selected"]["rmse"][-1],
        expanded["arms"]["full"][-1],
        expanded["arms"]["selected"][-1],
        expanded["arms"]["local_selector"][-1],
    ]
    for j, ax in enumerate(axes):
        ax.bar(
            np.arange(5),
            np.array(points)[:, j],
            color=["#4477aa", "#66aacc", "#cc6677", "#dd8899", "#aa4477"],
        )
        ax.axhline(
            evaluation["scores"]["hold"]["rmse"][-1][j],
            color="black",
            ls="--",
            label="Hold current",
        )
        ax.set(
            xticks=np.arange(5),
            xticklabels=labels,
            ylabel=["Accelerometer vector RMSE [g]", "Gyro vector RMSE [rad/s]"][j],
            ylim=(0, None),
        )
        ax.tick_params(axis="x", labelrotation=25, labelsize=8)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle(
        "Crazyflie sensor forecast · 240 ms · one 3.22 s evaluation interval\nOriginal fit and selection were frozen; broader training is an adaptive follow-up",
        fontsize=11,
    )
    for ext in ("png", "svg"):
        fig.savefig(a.output / f"crazyflie-representation.{ext}", dpi=170)
    plt.close(fig)
    print(json.dumps(read(a.output / "audit.json")), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root", type=Path, default=Path("../artifacts/representation-study")
    )
    p.add_argument(
        "--previous",
        type=Path,
        default=Path("../artifacts/forecast-diagnosis/comparison-01"),
    )
    p.add_argument("--output", type=Path, required=True)
    main(p.parse_args())
