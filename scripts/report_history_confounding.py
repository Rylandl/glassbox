"""Audit the history-confounding experiment without its regression fit helper."""

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
from experiment_history_confounding import observed_cycle
from experiment_model_qualification import collection, encoding, recordings
from experiment_sequence_diagnostics import fresh_recordings
from report_model_structures import recurrence
from report_sequence_diagnostics import regress

from glassbox.experimental.default_model import LearnedDynamics
from glassbox.experimental.sequence_model import SequenceModel


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def score(residual, predictions, subset):
    error = {
        name: np.mean((value[subset] - residual[subset]) ** 2, axis=0)
        for name, value in predictions.items()
    }
    return dict(
        rms={name: np.sqrt(value).tolist() for name, value in error.items()},
        extra_history_mse_reduction={
            name: [
                1 - float(x / y) if y > 0 else None
                for x, y in zip(error[name], error["short"], strict=True)
            ]
            for name in ("extended", "replacement")
        },
    )


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    plan, results = read(args.run / "plan.json"), read(args.run / "results.json")
    assert len(results) == 12
    assert {(r["case"], r["seed"]) for r in results} == {
        (c, s) for c in plan["cases"] for s in plan["seeds"]
    }
    with zipfile.ZipFile(args.run / "executed-sources.zip") as z:
        for path, expected in read(args.run / "sources.json").items():
            assert hashlib.sha256(z.read(path)).hexdigest() == expected
    count, maximum, windows, regressions, transition_checks = 0, 0.0, 0, 0, 0

    def check(left, right):
        nonlocal count, maximum
        left, right = np.asarray(left), np.asarray(right)
        assert (
            left.shape == right.shape
            and np.isfinite(left).all()
            and np.isfinite(right).all()
        )
        maximum = max(maximum, float(np.max(np.abs(left - right))))
        np.testing.assert_allclose(left, right, rtol=1e-7, atol=1e-9)
        count += 1

    def compare(left, right):
        if isinstance(left, dict):
            assert left.keys() == right.keys()
            for key in left:
                compare(left[key], right[key])
        elif isinstance(left, list) and any(v is None for v in left):
            assert left == right
        else:
            check(left, right)

    for r in results:
        directory = args.run / f"{r['case']}-{r['seed']}"
        assert read(directory / "result.json") == r
        path = Path(r["model_source"])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == r["model_sha256"]
        model = LearnedDynamics.load(path)
        assert model.fingerprint() == r["fitted_model_fingerprint"]
        assert model.report["recipe"] == plan["learner_recipe"]
        diagnostic, comparison = (
            read(directory / "diagnostic.json"),
            read(directory / "comparison.json"),
        )
        assert diagnostic["recipe"] == plan["diagnostic_recipe"]
        if r["case"].startswith("observed_cycle"):
            supplied = observed_cycle(r["seed"], evaluation=True)
            for segment in supplied.segments:
                x = segment.states
                check(x[1:, 0], x[:-1, 1] ** 3)
                check(x[1:, 1:4], x[:-1, 2:5])
                check(x[1:, 4] ** 3, x[:-1, 0])
                check(x[5:], x[:-5])
                transition_checks += len(x) - 1
        elif r["case"] == "hidden_delay-learned":
            supplied = collection(
                recordings("memory", r["seed"], evaluation=True),
                encoding("identity", 1),
                "memory",
            )
        else:
            supplied = collection(
                fresh_recordings("process_noise", r["seed"], evaluation=True),
                encoding("identity", 1),
                "process_noise",
            )
        assert diagnostic["contract"] == model.contract
        assert not set(diagnostic["recording_content"]) & set(model._seen)
        assert not set(diagnostic["recording_content"].values()) & set(
            model._seen.values()
        )
        with np.load(directory / "evidence.npz") as z:
            a = {key: z[key] for key in z.files}
        source = sorted(supplied.segments, key=lambda s: (s.recording_id, s.start_row))
        names = sorted({s.recording_id for s in source})
        assert a["record_names"].tolist() == names
        index = [(s, t) for s in source for t in range(4, len(s.states) - 1)]
        windows += len(index)
        assert len(index) == diagnostic["windows"] == r["windows"]
        for key, expected in (
            ("past_states", [s.states[t + np.arange(-2, 1)] for s, t in index]),
            ("past_inputs", [s.inputs[t + np.arange(-2, 0)] for s, t in index]),
            ("current_inputs", [s.inputs[t] for s, t in index]),
            ("targets", [s.states[t + 1] for s, t in index]),
            (
                "older_features",
                [
                    np.concatenate(
                        (
                            s.states[t + np.arange(-4, -2)].ravel(),
                            s.inputs[t + np.arange(-4, -2)].ravel(),
                        )
                    )
                    for s, t in index
                ],
            ),
            ("source_origins", [s.start_row + t for s, t in index]),
            ("record_index", [names.index(s.recording_id) for s, _ in index]),
        ):
            check(a[key], expected)
        assert a["segment_ids"].tolist() == [s.segment_id for s, _ in index]
        short = np.column_stack(
            (
                a["past_states"].reshape(len(index), -1),
                a["past_inputs"].reshape(len(index), -1),
            )
        )
        check(short, a["short_features"])
        if r["case"].endswith("hold"):
            expected_fingerprint = hashlib.sha256(
                ("hold-observation-witness-v1:" + model.fingerprint()).encode()
            ).hexdigest()
            predicted = a["past_states"][:, -1]
        else:
            expected_fingerprint = model.fingerprint()
            predicted = recurrence(
                model._model,
                a["past_states"],
                a["past_inputs"],
                a["current_inputs"][:, None],
            )[:, 0]
        assert (
            expected_fingerprint
            == r["predictor_fingerprint"]
            == diagnostic["model_fingerprint"]
        )
        check(a["model_prediction"], predicted)
        check(a["model_residual"], a["targets"] - predicted)
        check(np.sqrt(np.mean(a["model_residual"] ** 2, axis=0)), r["model_rms"])
        residual, older, ids = (
            a["model_residual"],
            a["older_features"],
            a["record_index"],
        )
        context = np.column_stack((short, a["current_inputs"]))
        for held in range(len(names)):
            old_prefix = f"fold_{held}_"
            train, test = a[old_prefix + "train"], a[old_prefix + "test"]
            check(train, np.flatnonzero(ids != held))
            check(test, np.flatnonzero(ids == held))
            dt, dq = a[old_prefix + "donor_train"], a[old_prefix + "donor_test"]
            assert set(dt) <= set(train) and set(dq) <= set(train)
            assert np.all(ids[dt] != ids[train])
            for degree in (1, 2, 3):
                prefix = f"degree_{degree}_{old_prefix}"
                mean, scale = context[train].mean(0), context[train].std(0)
                scale[scale == 0] = 1
                check(mean, a[prefix + "short_mean"])
                check(scale, a[prefix + "short_scale"])
                normalized = (context - mean) / scale
                # Explicit per-column powers instead of the experiment's helper.
                basis = np.column_stack(
                    [
                        normalized[:, col] ** power
                        for power in range(1, degree + 1)
                        for col in range(normalized.shape[1])
                    ]
                )
                for name, left, right in (
                    ("short", basis[train], basis[test]),
                    (
                        "extended",
                        np.column_stack((basis[train], older[train])),
                        np.column_stack((basis[test], older[test])),
                    ),
                    (
                        "replacement",
                        np.column_stack((basis[train], older[dt])),
                        np.column_stack((basis[test], older[dq])),
                    ),
                ):
                    output, fitted = regress(
                        left,
                        residual[train],
                        right,
                        False,
                        plan["comparison"]["ridge_fraction"],
                    )
                    regressions += 1
                    assert not bool(a[prefix + name + "_quadratic"])
                    for key, value in fitted.items():
                        check(a[prefix + name + "_" + key], value)
                    check(output, a[f"degree_{degree}_{name}"][test])
                    if degree == 1:
                        check(output, a["residual_" + name][test])
        for degree in (1, 2, 3):
            predictions = {
                name: a[f"degree_{degree}_{name}"]
                for name in ("short", "extended", "replacement")
            }
            aggregate = score(residual, predictions, np.arange(len(ids)))
            compare(aggregate, r["degrees"][str(degree)])
            compare(aggregate, comparison[str(degree)]["aggregate"])
            for i, name in enumerate(names):
                compare(
                    score(residual, predictions, np.flatnonzero(ids == i)),
                    comparison[str(degree)]["recordings"][name],
                )
    audit = dict(
        cases=len(results),
        windows=windows,
        auxiliary_refits=regressions,
        markov_transition_witness_checks=transition_checks,
        numeric_comparisons=count,
        max_absolute_difference=maximum,
        relative_tolerance=1e-7,
        absolute_tolerance=1e-9,
        method="Source/model hashes; regenerated and independently indexed windows; analytic Markov identities; independent NumPy prediction replay; whole-recording folds/donor exclusions; train-only normalization and additive powers; augmented least-squares fits and per-channel/per-recording scoring.",
        limits="Shares data generators, artifact loader, and the prior independent NumPy replay/least-squares code. Audits the new residual-basis comparisons; v1 input-predictability regressions are unchanged and were audited in the earlier investigation.",
    )
    write(args.output / "audit.json", audit)
    summary = {}
    for case in plan["cases"]:
        rows = [r for r in results if r["case"] == case]
        summary[case] = {}
        for degree in (1, 2, 3):
            fields = {
                **{
                    f"{name}_rms": [
                        r["degrees"][str(degree)]["rms"][name] for r in rows
                    ]
                    for name in ("short", "extended", "replacement")
                },
                **{
                    f"{name}_mse_reduction": [
                        r["degrees"][str(degree)]["extra_history_mse_reduction"][name]
                        for r in rows
                    ]
                    for name in ("extended", "replacement")
                },
                "short_mse_reduction_vs_affine": [
                    1
                    - (
                        np.array(r["degrees"][str(degree)]["rms"]["short"])
                        / r["degrees"]["1"]["rms"]["short"]
                    )
                    ** 2
                    for r in rows
                ],
                "extra_history_mse_gain_in_affine_units": [
                    (
                        np.array(r["degrees"][str(degree)]["rms"]["short"]) ** 2
                        - np.array(r["degrees"][str(degree)]["rms"]["extended"]) ** 2
                    )
                    / np.array(r["degrees"]["1"]["rms"]["short"]) ** 2
                    for r in rows
                ],
            }
            summary[case][str(degree)] = {
                name: dict(
                    minimum=np.min(value, axis=0).tolist(),
                    maximum=np.max(value, axis=0).tolist(),
                )
                for name, value in fields.items()
            }
    write(args.output / "summary.json", summary)
    plot(results, args.output / "history-confounding.png")
    if args.ablation is not None:
        audit_ablation(args.ablation, args.output)
    print(json.dumps(audit, indent=2))


def audit_ablation(run, output):
    results, plan = read(run / "results.json"), read(run / "plan.json")
    assert len(results) == 9
    assert {(r["case"], r["seed"]) for r in results} == {
        (c, s) for c in plan["cases"] for s in plan["seeds"]
    }
    with zipfile.ZipFile(run / "executed-sources.zip") as z:
        for name, digest in read(run / "sources.json").items():
            assert hashlib.sha256(z.read(name)).hexdigest() == digest
    count, maximum, windows = 0, 0.0, 0

    def check(a, b):
        nonlocal count, maximum
        a, b = np.asarray(a), np.asarray(b)
        assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
        maximum = max(maximum, float(np.max(np.abs(a - b))))
        np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-9)
        count += 1

    summary = {}
    for r in results:
        directory = run / f"{r['case']}-{r['seed']}"
        assert r["status"] == "complete" and read(directory / "result.json") == r
        path = Path(r["baseline_model_source"])
        assert (
            hashlib.sha256(path.read_bytes()).hexdigest() == r["baseline_model_sha256"]
        )
        base, candidate = (
            LearnedDynamics.load(path),
            SequenceModel.load(directory / "candidate.npz"),
        )
        assert base.fingerprint() == r["baseline_fingerprint"]
        assert candidate.fingerprint() == r["candidate_fingerprint"]
        for key in candidate.norms:
            check(candidate.norms[key], base._model.norms[key])
        if r["case"].startswith("observed_cycle"):
            supplied = observed_cycle(r["seed"], evaluation=True)
        elif r["case"] == "hidden_delay-learned":
            supplied = collection(
                recordings("memory", r["seed"], evaluation=True),
                encoding("identity", 1),
                "memory",
            )
        else:
            supplied = collection(
                fresh_recordings("process_noise", r["seed"], evaluation=True),
                encoding("identity", 1),
                "process_noise",
            )
        index = [(s, t) for s in supplied.segments for t in range(2, len(s.states) - 5)]
        windows += len(index)
        assert len(index) == r["evaluation_windows"]
        with np.load(directory / "evaluation.npz") as z:
            a = {key: z[key] for key in z.files}
        for key, expected in (
            ("past_states", [s.states[t + np.arange(-2, 1)] for s, t in index]),
            ("past_inputs", [s.inputs[t + np.arange(-2, 0)] for s, t in index]),
            ("future_inputs", [s.inputs[t + np.arange(5)] for s, t in index]),
            ("targets", [s.states[t + np.arange(1, 6)] for s, t in index]),
            ("source_origins", [s.start_row + t for s, t in index]),
        ):
            check(a[key], expected)
        assert a["recording_ids"].tolist() == [s.recording_id for s, _ in index]
        assert a["segment_ids"].tolist() == [s.segment_id for s, _ in index]
        b, dev = base._train.batch, base._development.batch
        assert (
            len(b.past_states) == r["training_windows"]
            and len(dev.past_states) == r["development_windows"]
        )
        # Pool the already window-averaged squared reference error over horizons.
        by_horizon = np.mean((b.future_states - b.past_states[:, -1:]) ** 2, axis=0)
        pooled = np.maximum(
            np.sqrt(np.mean(by_horizon, axis=0)),
            0.01 * base._model.norms["state_scale"],
        )
        scale = np.tile(pooled, (5, 1))
        check(a["loss_scale"], scale)
        check(r["loss_scale"], scale)
        report = read(directory / "fit-report.json")
        check(report["error_scale"], scale)
        assert report["selected_step"] == r["selected_step"]
        trace_best = min(report["trace"], key=lambda t: t["validation_rollout_mse"])
        assert trace_best["step"] == report["selected_step"]
        predicted = recurrence(
            candidate, dev.past_states, dev.past_inputs, dev.future_inputs
        )
        check(
            np.mean(((predicted - dev.future_states) / scale) ** 2),
            report["validation_rollout_mse"],
        )
        for label, model in (("baseline", base._model), ("pooled", candidate)):
            predicted = recurrence(
                model, a["past_states"], a["past_inputs"], a["future_inputs"]
            )
            check(predicted, a[label])
            squared = np.mean((predicted - a["targets"]) ** 2, axis=0)
            check(np.sqrt(squared), r["channel_rmse"][label])
            check(np.sqrt(squared.sum(axis=1)), r["horizon_vector_rmse"][label])
        relative = (
            np.array(r["horizon_vector_rmse"]["pooled"])
            / r["horizon_vector_rmse"]["baseline"]
            - 1
        )
        summary.setdefault(r["case"], []).append(relative)
    summary = {
        case: dict(
            minimum_relative_vector_rmse_change=np.min(values, axis=0).tolist(),
            maximum_relative_vector_rmse_change=np.max(values, axis=0).tolist(),
        )
        for case, values in summary.items()
    }
    write(
        output / "objective-comparison-audit.json",
        dict(
            cases=len(results),
            evaluation_windows=windows,
            numeric_comparisons=count,
            max_absolute_difference=maximum,
            checks="Regenerated evaluation histories/targets/provenance, paired NumPy model replay, all horizon/channel scores, train-only pooled scales, preserved model normalization, baseline hashes, candidate fingerprints, development loss and checkpoint selection.",
            limits="Shares frozen generators and prior NumPy rollout. Does not independently repeat the optimizer; the experiment uses identical saved calibration caches and changes only error scaling.",
        ),
    )
    write(output / "objective-comparison-summary.json", summary)
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.6))
    for ax, case, title in zip(
        axes,
        plan["cases"],
        ("Nonlinear cycle", "Hidden input delay", "Process noise"),
        strict=True,
    ):
        for label, color in (("baseline", "#8e989e"), ("pooled", "#306d99")):
            for i, r in enumerate(row for row in results if row["case"] == case):
                ax.plot(
                    np.arange(1, 6) * 50,
                    r["horizon_vector_rmse"][label],
                    color=color,
                    alpha=0.8,
                    marker="o",
                    markersize=3,
                    label=label.capitalize() if i == 0 else None,
                )
        ax.set(
            title=title,
            xlabel="Forecast horizon (ms)",
            ylabel="State-vector RMSE (unitless)",
            xticks=[50, 150, 250],
        )
        ax.grid(alpha=0.15)
    axes[0].legend(frameon=False)
    fig.suptitle(
        "Pooling loss scales improves the cycle's early forecasts, with tradeoffs",
        fontsize=14,
    )
    fig.text(
        0.5,
        0.02,
        "Matched data, architecture and optimizer; only loss normalization changes. Three data seeds per case.",
        ha="center",
        color=".35",
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.94))
    fig.savefig(output / "horizon-scaling.png", dpi=180)
    plt.close(fig)


def plot(results, path):
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.2), sharey=True)
    for ax, case, title in zip(
        axes,
        ("observed_cycle-learned", "hidden_delay-learned", "process_noise-learned"),
        (
            "Fully observed nonlinear cycle",
            "Genuinely omitted input delay",
            "Unpredictable process noise",
        ),
        strict=True,
    ):
        for name, color, offset in (
            ("short", "#376b93", -0.16),
            ("extended", "#d26439", 0.0),
            ("replacement", "#8f979d", 0.16),
        ):
            for degree in (1, 2, 3):
                values = [
                    r["degrees"][str(degree)]["rms"][name][0]
                    for r in results
                    if r["case"] == case
                ]
                ax.scatter(
                    np.full(3, degree + offset),
                    values,
                    color=color,
                    s=30,
                    label={
                        "short": "Short history",
                        "extended": "+ Older history",
                        "replacement": "+ Foreign-history control",
                    }[name]
                    if degree == 1
                    else None,
                )
        ax.set(
            title=title,
            yscale="log",
            ylim=(1e-3, 0.2),
            xticks=[1, 2, 3],
            xticklabels=["Affine", "Quadratic", "Cubic"],
            xlabel="Fixed additive short-context basis",
        )
        ax.grid(axis="y", alpha=0.15)
    axes[0].set_ylabel("Remaining residual RMS, first output (unitless)")
    axes[1].legend(loc="lower left", frameon=False, fontsize=9)
    fig.suptitle(
        "Older observations can compensate for missing nonlinear features",
        fontsize=15,
        y=0.98,
    )
    fig.text(
        0.5,
        0.025,
        "Three data seeds; whole-recording evaluation; fitted model unchanged. Synthetic examples, not a cause classifier.",
        ha="center",
        color="0.35",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ablation", type=Path)
    main(parser.parse_args())
