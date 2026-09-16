"""Audit and summarize the frozen synthetic model-qualification experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from experiment_model_qualification import collection, encoding, recordings
from report_model_structures import recurrence

from glassbox.experimental.default_model import LearnedDynamics


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def errors(prediction, target):
    squared = np.sum(np.square(prediction - target), axis=-1)
    return np.sqrt(squared.mean(axis=0))


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    plan, rows = read(args.run / "plan.json"), read(args.run / "results.json")
    expected = {
        (family, seed, variant)
        for family, variants in (
            ("command", ("0.0", "0.02", "0.25")),
            ("memory", ("hidden",)),
            ("encoding", tuple(plan["experiments"]["encoding"]["variants"])),
        )
        for seed in plan["dataset_seeds"]
        for variant in variants
    }
    assert (
        len(rows) == 27
        and {(r["family"], r["seed"], r["variant"]) for r in rows} == expected
    )
    with zipfile.ZipFile(args.run / "executed-sources.zip") as z:
        for name, digest in read(args.run / "sources.json").items():
            assert hashlib.sha256(z.read(name)).hexdigest() == digest
    count, maximum, cache_count = 0, 0.0, 0

    def check(a, b):
        nonlocal count, maximum
        a, b = np.asarray(a), np.asarray(b)
        assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
        maximum = max(maximum, float(np.max(np.abs(a - b))))
        np.testing.assert_allclose(a, b, rtol=2e-9, atol=2e-9)
        count += 1

    encodings, memories = {}, []
    for r in rows:
        directory = args.run / f"{r['family']}-{r['seed']}-{r['variant']}"
        assert read(directory / "result.json") == r
        if r["status"] != "complete":
            continue
        model = LearnedDynamics.load(directory / "model.npz")
        assert model.fingerprint() == r["fingerprint"]
        assert model.report == read(directory / "fit-report.json")
        assert model.report["recipe"] == plan["recipe"]
        assert model.report["optimization"]["selected_step"] == r["selected_step"]
        source = recordings(
            r["family"],
            r["seed"],
            excitation=float(r["variant"]) if r["family"] == "command" else 0,
        )
        transform = (
            encoding(r["variant"], 2)
            if r["family"] == "encoding"
            else encoding("identity", 1)
        )
        transformed = collection(source, transform, r["family"])
        source_by_id = {s.recording_id: s for s in transformed.segments}
        for role in ("train", "development"):
            cache = getattr(model, f"_{role}")
            key_name = "training_keys" if role == "train" else "development_keys"
            assert [[k.recording_id, k.segment_id, k.origin] for k in cache.keys] == r[
                key_name
            ]
            for i, key in enumerate(cache.keys):
                s, t = source_by_id[key.recording_id], key.origin
                check(cache.batch.past_states[i], s.states[t - 2 : t + 1])
                check(cache.batch.past_inputs[i], s.inputs[t - 2 : t])
                check(cache.batch.future_states[i], s.states[t + 1 : t + 6])
                check(cache.batch.future_inputs[i], s.inputs[t : t + 5])
                cache_count += 1
        with np.load(directory / "evaluation.npz") as z:
            canonical = recordings(
                r["family"],
                r["seed"],
                evaluation=True,
                excitation=float(r["variant"]) if r["family"] == "command" else 0,
            )
            # Independently index every evaluation window from regenerated data.
            for name, attribute, offsets in (
                ("past_states", "states", np.arange(-2, 1)),
                ("past_inputs", "inputs", np.arange(-2, 0)),
                ("future_inputs", "inputs", np.arange(5)),
                ("targets", "states", np.arange(1, 6)),
            ):
                expected_array = np.stack(
                    [
                        getattr(s, attribute)[t + offsets]
                        for s in canonical
                        for t in range(2, len(s.states) - 5, 5)
                    ]
                )
                check(z[name], expected_array)
            e, d, o, f = (
                z[k] for k in ("state_map", "state_inverse", "offset", "input_map")
            )
            check(e @ d, np.eye(e.shape[0]))
            raw = recurrence(
                model._model,
                z["past_states"] @ e + o,
                z["past_inputs"] @ f,
                z["future_inputs"] @ f,
            )
            check(raw, z["encoded_prediction"])
            decoded = (raw - o) @ d
            check(decoded, z["prediction"])
            check(errors(decoded, z["targets"]), r["forecast_rmse"])
            if r["family"] == "command":
                u, px, pu = (
                    z["query_future_inputs"],
                    z["query_past_states"],
                    z["query_past_inputs"],
                )
                predictions = np.stack(
                    [recurrence(model._model, px, pu, branch) for branch in u]
                )
                check(predictions, z["query_prediction"])
                # Closed-form forced response, rather than the generator's recurrence.
                a, b = 1.08, 0.2
                targets = np.stack(
                    [
                        np.stack(
                            [
                                a ** (h + 1) * px[:, -1]
                                + b
                                * sum(a ** (h - j) * branch[:, j] for j in range(h + 1))
                                for h in range(5)
                            ],
                            axis=1,
                        )
                        for branch in u
                    ]
                )
                check(targets, z["query_targets"])
                exact_slope = np.broadcast_to(
                    (b * a ** np.arange(5))[None, :, None], predictions.shape[1:]
                )
                slope = (predictions[1] - predictions[0]) / 0.1
                check((targets[1] - targets[0]) / 0.1, exact_slope)
                ratio = errors(slope, exact_slope) / np.sqrt(
                    np.mean(exact_slope**2, axis=(0, 2))
                )
                check(ratio, r["command_slope_relative_rmse"])
                check(
                    errors(predictions.reshape(-1, 5, 1), targets.reshape(-1, 5, 1)),
                    r["perturbed_forecast_rmse"],
                )
                nominal = recurrence(model._model, px, pu, u.mean(0))
                check(nominal, z["nominal_prediction"])
                check(
                    errors(nominal, z["nominal_targets"]),
                    r["common_history_forecast_rmse"],
                )
                lo, hi = (
                    model._train.batch.future_inputs.min((0, 1)),
                    model._train.batch.future_inputs.max((0, 1)),
                )
                fraction = np.mean(
                    np.all((u[:, :, 0] >= lo) & (u[:, :, 0] <= hi), axis=-1)
                )
                check(
                    fraction,
                    r["perturbed_commands_in_marginal_training_range_fraction"],
                )
            elif r["family"] == "memory":
                px, pu, uf = (
                    z["probe_past_states"],
                    z["probe_past_inputs"],
                    z["probe_future_inputs"],
                )
                np.testing.assert_array_equal(px[0, -3:], px[1, -3:])
                np.testing.assert_array_equal(pu[0, -2:], pu[1, -2:])
                np.testing.assert_array_equal(uf[0], uf[1])
                half_gap = 0.2 * 0.8 ** np.arange(5)
                targets = np.stack((-half_gap, half_gap))[:, :, None]
                check(targets, z["probe_targets"])
                prediction = recurrence(model._model, px[:, -3:], pu[:, -2:], uf)
                check(prediction, z["probe_prediction"])
                check(prediction[0], prediction[1])
                check(errors(prediction, targets), r["paired_probe_rmse"])
                check(half_gap, r["paired_probe_minimum_possible_rmse"])
                # Exact paired squared-error identity for any common forecast.
                check(
                    errors(prediction, targets) ** 2,
                    half_gap**2 + prediction[0, :, 0] ** 2,
                )
                memories.append(
                    dict(seed=r["seed"], prediction=prediction[0, :, 0].tolist())
                )
            else:
                encodings.setdefault(r["seed"], {})[r["variant"]] = dict(
                    prediction=decoded.copy(), target=z["targets"].copy(), result=r
                )
    encoding_summary = []
    for seed, variants in encodings.items():
        if "identity" not in variants:
            continue
        base = variants["identity"]
        for name, item in variants.items():
            check(base["target"], item["target"])
            assert item["result"]["training_keys"] == base["result"]["training_keys"]
            assert (
                item["result"]["development_keys"] == base["result"]["development_keys"]
            )
            ratio = (
                np.array(item["result"]["forecast_rmse"])
                / base["result"]["forecast_rmse"]
            )
            encoding_summary.append(
                dict(
                    seed=seed,
                    variant=name,
                    endpoint_rmse_ratio=float(ratio[-1]),
                    rmse_ratio=ratio.tolist(),
                    maximum_decoded_prediction_difference=float(
                        np.max(np.abs(item["prediction"] - base["prediction"]))
                    ),
                )
            )
    witnesses = read(args.run / "identifiability-witnesses.json")
    assert len(witnesses) == 9
    for witness in witnesses:
        a, b = witness["a"], witness["b"]
        check(a - 0.5 * b, 0.98)
        assert witness["maximum_recorded_transition_difference"] < 1e-12
    summary = dict(
        command=[
            {
                k: v
                for k, v in r.items()
                if k not in ("training_keys", "development_keys")
            }
            for r in rows
            if r["family"] == "command"
        ],
        memory=[
            {
                k: v
                for k, v in r.items()
                if k not in ("training_keys", "development_keys")
            }
            for r in rows
            if r["family"] == "memory"
        ],
        encoding=encoding_summary,
    )
    write(args.output / "summary.json", summary)
    audit = dict(
        planned_fits=27,
        completed_fits=sum(r["status"] == "complete" for r in rows),
        failed_fits=sum(r["status"] != "complete" for r in rows),
        numerical_checks=count,
        cached_windows_reconstructed=cache_count,
        maximum_absolute_difference=maximum,
        library_predictions_replayed_in_numpy=True,
        sources_verified=True,
        limitation="Synthetic generators and selected input distributions; not vehicle validation. No uncertainty or task-admission guarantee inferred.",
    )
    write(args.output / "audit.json", audit)

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.5), layout="constrained")
    colors = ("#2563a6", "#bd6b15", "#4f8b65")
    for color, seed in zip(colors, plan["dataset_seeds"], strict=True):
        selected = [
            r
            for r in summary["command"]
            if r["seed"] == seed and r["status"] == "complete"
        ]
        axes[0].plot(
            [("0.0", "0.02", "0.25").index(r["variant"]) for r in selected],
            [100 * r["command_slope_relative_rmse"][0] for r in selected],
            "o-",
            color=color,
            label=f"Data seed {seed}",
        )
    axes[0].set(
        yscale="log",
        xticks=[0, 1, 2],
        xticklabels=["None", "±0.02", "±0.25"],
        xlabel="Independent calibration input variation",
        ylabel="50 ms response-slope error [%]",
        title="Recorded fit can hide a wrong response",
    )
    axes[0].legend(fontsize=8)
    t, half = np.arange(1, 6) * 50, 0.2 * 0.8 ** np.arange(5)
    axes[1].plot(t, half, "o-", color="#bd6b15", label="Possible future A")
    axes[1].plot(t, -half, "o-", color="#2563a6", label="Possible future B")
    for i, m in enumerate(memories):
        axes[1].plot(
            t,
            m["prediction"],
            color="#52545b",
            alpha=0.8,
            label="Model forecast (each data seed)" if i == 0 else None,
        )
    axes[1].set(
        xlabel="Forecast horizon [ms]",
        ylabel="Observation [arbitrary units]",
        title="Identical short histories, two futures",
    )
    axes[1].legend(fontsize=8)
    names = ["units_offset", "orthogonal", "permuted", "duplicate_first"]
    for index, (color, seed) in enumerate(
        zip(colors, plan["dataset_seeds"], strict=True)
    ):
        chosen = [
            r for r in encoding_summary if r["seed"] == seed and r["variant"] in names
        ]
        axes[2].scatter(
            np.array([names.index(r["variant"]) for r in chosen]) + (index - 1) * 0.08,
            [r["endpoint_rmse_ratio"] for r in chosen],
            color=color,
            s=33,
        )
    axes[2].axhline(1, color="#52545b", linestyle="--", linewidth=1)
    axes[2].set(
        xticks=np.arange(4),
        xticklabels=[
            "Units +\noffset",
            "Rotated\naxes",
            "Swapped\nchannels",
            "Repeated\nchannel",
        ],
        ylabel="250 ms RMSE / original encoding RMSE",
        title="Equivalent information, changed fit",
    )
    for ax in axes:
        ax.grid(alpha=0.15)
    fig.suptitle(
        "Frozen generic recipe · 27 synthetic fits · no vehicle or controller trials",
        fontsize=13,
    )
    for ext in ("png", "svg"):
        fig.savefig(args.output / f"model-qualification.{ext}", dpi=170)
    plt.close(fig)
    print(json.dumps(audit), flush=True)
    print(json.dumps(encoding_summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
