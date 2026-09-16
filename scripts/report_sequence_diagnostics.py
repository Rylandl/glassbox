"""Independently replay the fixed sequence diagnostics and summarize witnesses."""

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
from experiment_sequence_diagnostics import fresh_recordings
from report_model_structures import recurrence

from glassbox.experimental.default_model import LearnedDynamics


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def basis(x, quadratic):
    if not quadratic:
        return x
    return np.column_stack(
        [x]
        + [x[:, i] * x[:, j] for i in range(x.shape[1]) for j in range(i, x.shape[1])]
    )


def regress(x, y, query, quadratic, penalty_fraction):
    """Augmented least squares instead of the implementation's normal equations."""
    values = {}
    for name, data in (("input", x), ("target", y)):
        values[name + "_mean"] = data.mean(0)
        scale = data.std(0)
        values[name + "_scale"] = np.where(scale == 0, 1.0, scale)
    raw = basis((x - values["input_mean"]) / values["input_scale"], quadratic)
    values["feature_mean"] = raw.mean(0)
    scale = raw.std(0)
    values["feature_scale"] = np.where(scale == 0, 1.0, scale)
    z = (raw - values["feature_mean"]) / values["feature_scale"]
    target = (y - values["target_mean"]) / values["target_scale"]
    augmented = np.vstack((z, np.sqrt(penalty_fraction * len(x)) * np.eye(z.shape[1])))
    rhs = np.vstack((target, np.zeros((z.shape[1], y.shape[1]))))
    values["coefficients"] = np.linalg.lstsq(augmented, rhs, rcond=None)[0]
    features = basis((query - values["input_mean"]) / values["input_scale"], quadratic)
    prediction = (
        (features - values["feature_mean"])
        / values["feature_scale"]
        @ values["coefficients"]
    ) * values["target_scale"] + values["target_mean"]
    return prediction, values


def metrics(a, subset):
    def mse(left, right):
        return np.mean((left[subset] - right[subset]) ** 2, axis=0)

    def ratio(a, b):
        return [float(x / y) if y > 0 else None for x, y in zip(a, b, strict=True)]

    inputs, current = {}, a["current_inputs"]
    reference = np.sqrt(mse(current, a["input_reference"]))
    for kind in ("affine", "quadratic"):
        error = np.sqrt(mse(current, a["input_" + kind]))
        inputs[kind] = dict(
            rms=error.tolist(),
            reference_rms=reference.tolist(),
            remaining_rms_fraction=ratio(error, reference),
        )
    residual = a["model_residual"]
    errors = {
        k: mse(residual, a["residual_" + k])
        for k in ("short", "extended", "replacement")
    }
    reductions = {
        k: [None if x is None else 1 - x for x in ratio(errors[k], errors["short"])]
        for k in ("extended", "replacement")
    }
    return dict(
        input_predictability=inputs,
        forecast_error=dict(
            model_rms=np.sqrt(np.mean(residual[subset] ** 2, axis=0)).tolist(),
            **{f"after_{k}_rms": np.sqrt(v).tolist() for k, v in errors.items()},
            extra_history_mse_reduction=reductions,
        ),
    )


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    plan, rows = read(args.run / "plan.json"), read(args.run / "results.json")
    expected = {
        (case, seed)
        for cases in plan["cases"].values()
        for case in cases
        for seed in plan["data_seeds"]
    }
    assert len(rows) == 24 and {(r["case"], r["seed"]) for r in rows} == expected
    with zipfile.ZipFile(args.run / "executed-sources.zip") as z:
        for name, digest in read(args.run / "sources.json").items():
            assert hashlib.sha256(z.read(name)).hexdigest() == digest
    count, maximum, window_count, regression_count = 0, 0.0, 0, 0

    def check(left, right):
        nonlocal count, maximum
        left, right = np.asarray(left), np.asarray(right)
        assert left.shape == right.shape
        assert np.isfinite(left).all() and np.isfinite(right).all()
        maximum = max(maximum, float(np.max(np.abs(left - right))))
        np.testing.assert_allclose(left, right, rtol=1e-7, atol=1e-9)
        count += 1

    def compare(left, right):
        if isinstance(left, dict):
            assert left.keys() == right.keys()
            for key in left:
                compare(left[key], right[key])
        else:
            check(left, right)

    for r in rows:
        directory = args.run / f"{r['case']}-{r['seed']}"
        assert read(directory / "result.json") == r
        model_path = Path(r["model_source"])
        assert hashlib.sha256(model_path.read_bytes()).hexdigest() == r["model_sha256"]
        model = LearnedDynamics.load(model_path)
        assert model.fingerprint() == r["model_fingerprint"]
        report = read(directory / "report.json")
        assert report["model_fingerprint"] == r["model_fingerprint"]
        assert report["recipe"] == plan["diagnostic_recipe"]
        assert model.report["recipe"] == plan["learner_recipe"]
        if r["case"] in plan["cases"]["existing"]:
            family, variant = r["case"].split("-", 1)
            segments = recordings(
                family,
                r["seed"],
                evaluation=True,
                excitation=float(variant) if family == "command" else 0,
            )
        else:
            family = r["case"]
            segments = fresh_recordings(family, r["seed"], evaluation=True)
            if family in ("quadratic_policy", "cubic_policy"):
                power = 2 if family == "quadratic_policy" else 3
                for s in segments:
                    check(s.inputs, s.states[:-1] ** power)
        supplied = collection(
            segments, encoding("identity", len(segments[0].states[0])), family
        )
        assert report["contract"] == model.contract
        assert supplied.configuration_id == model.contract["configuration_id"]
        assert not set(report["recording_content"]) & set(model._seen)
        assert not set(report["recording_content"].values()) & set(model._seen.values())
        assert report["model_history_steps"] == model.history_steps == 2
        assert report["extended_history_steps"] == 4 and report["forecast_steps"] == 1
        with np.load(directory / "evidence.npz") as z:
            a = {k: z[k] for k in z.files}
        names = sorted({s.recording_id for s in supplied.segments})
        assert a["record_names"].tolist() == names
        source = sorted(supplied.segments, key=lambda s: s.recording_id)
        index = [(s, t) for s in source for t in range(4, len(s.states) - 1)]
        assert len(index) == report["windows"] == 624
        window_count += len(index)
        for key, expected_array in (
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
            check(a[key], expected_array)
        assert a["segment_ids"].tolist() == [s.segment_id for s, _ in index]
        short = np.concatenate(
            (
                a["past_states"].reshape(len(index), -1),
                a["past_inputs"].reshape(len(index), -1),
            ),
            axis=1,
        )
        check(a["short_features"], short)
        prediction = recurrence(
            model._model,
            a["past_states"],
            a["past_inputs"],
            a["current_inputs"][:, None],
        )[:, 0]
        check(a["model_prediction"], prediction)
        check(a["model_residual"], a["targets"] - prediction)
        residual = a["model_residual"]
        short_error = np.column_stack((short, a["current_inputs"]))
        older = a["older_features"]
        extended = np.column_stack((short_error, older))
        for held in range(len(names)):
            prefix = f"fold_{held}_"
            train, test = a[prefix + "train"], a[prefix + "test"]
            check(train, np.flatnonzero(a["record_index"] != held))
            check(test, np.flatnonzero(a["record_index"] == held))
            dtrain, dtest = a[prefix + "donor_train"], a[prefix + "donor_test"]
            assert set(dtrain) <= set(train) and set(dtest) <= set(train)
            assert np.all(a["record_index"][dtrain] != a["record_index"][train])
            check(
                a["input_reference"][test],
                np.broadcast_to(
                    a["current_inputs"][train].mean(0), a["current_inputs"][test].shape
                ),
            )
            for label, x, y, query, quadratic in (
                (
                    "input_affine",
                    short[train],
                    a["current_inputs"][train],
                    short[test],
                    False,
                ),
                (
                    "input_quadratic",
                    short[train],
                    a["current_inputs"][train],
                    short[test],
                    True,
                ),
                (
                    "residual_short",
                    short_error[train],
                    residual[train],
                    short_error[test],
                    False,
                ),
                (
                    "residual_extended",
                    extended[train],
                    residual[train],
                    extended[test],
                    False,
                ),
                (
                    "residual_replacement",
                    np.column_stack((short_error[train], older[dtrain])),
                    residual[train],
                    np.column_stack((short_error[test], older[dtest])),
                    False,
                ),
            ):
                predicted, fitted = regress(
                    x, y, query, quadratic, plan["diagnostic_recipe"]["ridge_fraction"]
                )
                regression_count += 1
                assert bool(a[prefix + label + "_quadratic"]) == quadratic
                for key, value in fitted.items():
                    check(a[prefix + label + "_" + key], value)
                check(a[label][test], predicted)
        aggregate = metrics(a, np.arange(len(index)))
        compare(report["aggregate"], aggregate)
        for key in aggregate:
            compare(r[key], aggregate[key])
        for i, name in enumerate(names):
            mask = np.flatnonzero(a["record_index"] == i)
            compare(
                report["recordings"][name], dict(windows=len(mask), **metrics(a, mask))
            )
    audit = dict(
        cases=len(rows),
        diagnostic_windows=window_count,
        auxiliary_regressions=regression_count,
        numeric_checks=count,
        maximum_absolute_difference=maximum,
        relative_tolerance=1e-7,
        absolute_tolerance=1e-9,
        method="Regenerated synthetic data; independent window indexing and NumPy dynamics replay; whole-recording folds and foreign donor checks; independent train-only scaling and augmented least squares; independent per-channel/per-recording scores; archived-source and model hashes.",
        limitations="Shares the frozen data generators and model artifact loader. This verifies arithmetic and evidence separation, not causal interpretation or external validity.",
    )
    write(args.output / "audit.json", audit)
    summary = {}
    for case in sorted({r["case"] for r in rows}):
        subset = [r for r in rows if r["case"] == case]
        fields = {
            **{
                f"input_{k}_remaining_rms_fraction": [
                    r["input_predictability"][k]["remaining_rms_fraction"]
                    for r in subset
                ]
                for k in ("affine", "quadratic")
            },
            **{
                f"{k}_history_mse_reduction": [
                    r["forecast_error"]["extra_history_mse_reduction"][k]
                    for r in subset
                ]
                for k in ("extended", "replacement")
            },
            **{
                k: [r["forecast_error"][k] for r in subset]
                for k in ("model_rms", "after_short_rms", "after_extended_rms")
            },
        }
        summary[case] = {
            k: dict(
                minimum=np.min(v, axis=0).tolist(), maximum=np.max(v, axis=0).tolist()
            )
            for k, v in fields.items()
        }
    write(args.output / "summary.json", summary)
    plot(rows, args.output / "sequence-diagnostics.png")
    print(json.dumps(audit, indent=2))


def plot(rows, path):
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(
        1, 2, figsize=(12.8, 5.7), gridspec_kw={"width_ratios": [1.35, 1]}
    )
    ax = axes[0]
    cases = [
        "command-0.0",
        "command-0.02",
        "command-0.25",
        "quadratic_policy",
        "cubic_policy",
        "process_noise",
    ]
    labels = [
        "Linear policy\nno added input",
        "Linear policy\n±0.02 noise",
        "Linear policy\n±0.25 noise",
        "u = x²\ndeterministic",
        "u = x³\ndeterministic",
        "Independent\nrandom input",
    ]
    for k, color, offset in (
        ("affine", "#6e8baa", -0.12),
        ("quadratic", "#cb6235", 0.12),
    ):
        for i, case in enumerate(cases):
            values = [
                100 * r["input_predictability"][k]["remaining_rms_fraction"][0]
                for r in rows
                if r["case"] == case
            ]
            ax.scatter(
                np.full(3, i + offset),
                values,
                color=color,
                s=30,
                label=k.capitalize() if i == 0 else None,
            )
    ax.set(
        yscale="log",
        ylim=(0.01, 200),
        ylabel="Unexplained input RMS / reference RMS (%)",
        title="Predictor limitations can resemble input variation",
    )
    ax.set_xticks(range(len(cases)), labels, fontsize=9)
    ax.axhline(100, color="0.7", linestyle=":", linewidth=1)
    ax.legend(loc="lower right", frameon=False)
    ax.grid(axis="y", alpha=0.15)
    ax = axes[1]
    for k, color, offset in (
        ("extended", "#cb6235", -0.12),
        ("replacement", "#6e8baa", 0.12),
    ):
        for i, case in enumerate(
            ("memory-hidden", "process_noise", "encoding-identity")
        ):
            values = [
                100 * v
                for r in rows
                if r["case"] == case
                for v in r["forecast_error"]["extra_history_mse_reduction"][k]
            ]
            ax.scatter(
                np.full(len(values), i + offset),
                values,
                color=color,
                s=30,
                label="Older history"
                if k == "extended" and i == 0
                else "Foreign-history control"
                if i == 0
                else None,
            )
    ax.set(
        ylim=(-12, 106),
        ylabel="Residual MSE reduction beyond short context (%)",
        title="Older history helps the stipulated memory case",
    )
    ax.set_xticks(
        range(3),
        [
            "Hidden\ninput delay",
            "Unpredictable\nprocess noise",
            "Nonlinear\nMarkov system",
        ],
    )
    ax.axhline(0, color="0.6", linestyle=":", linewidth=1)
    ax.legend(loc="center right", frameon=False)
    ax.grid(axis="y", alpha=0.15)
    fig.suptitle(
        "Synthetic diagnostics: useful evidence, with a demonstrated blind spot",
        fontsize=15,
        y=0.99,
    )
    fig.text(
        0.5,
        0.015,
        "Points: three data seeds; nonlinear Markov case has two output channels. Ratios are descriptive, not acceptance scores.",
        ha="center",
        fontsize=10,
        color="0.35",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96), w_pad=3)
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
