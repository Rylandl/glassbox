"""Independently audit new-system horizon scaling comparisons and their limits."""

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
from experiment_horizon_generalization import generate
from report_model_structures import recurrence

from glassbox.experimental.default_model import LearnedDynamics
from glassbox.experimental.sequence_model import SequenceModel


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def scores(predicted, target, scale):
    squared = (predicted - target) ** 2
    normalized = squared / scale**2
    return dict(
        channel_rmse=np.sqrt(squared.mean(axis=0)).tolist(),
        horizon_scaled_rmse=np.sqrt(normalized.mean(axis=(0, 2))).tolist(),
        overall_scaled_rmse=float(np.sqrt(normalized.mean())),
    )


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    plan, rows = read(args.run / "plan.json"), read(args.run / "results.json")
    key = plan.get("candidate_key", "pooled")
    assert len(rows) == 24
    assert {(r["family"], r["seed"]) for r in rows} == {
        (family, seed) for family in plan["families"] for seed in plan["data_seeds"]
    }
    with zipfile.ZipFile(args.run / "executed-sources.zip") as z:
        for name, digest in read(args.run / "sources.json").items():
            assert hashlib.sha256(z.read(name)).hexdigest() == digest
    count, maximum, cache_windows, evaluation_windows = 0, 0.0, 0, 0

    def check(a, b):
        nonlocal count, maximum
        a, b = np.asarray(a), np.asarray(b)
        assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
        maximum = max(maximum, float(np.max(np.abs(a - b))))
        np.testing.assert_allclose(a, b, rtol=1e-8, atol=1e-9)
        count += 1

    def compare(a, b):
        if isinstance(a, dict):
            assert a.keys() == b.keys()
            for name in a:
                compare(a[name], b[name])
        else:
            check(a, b)

    for r in rows:
        directory = args.run / f"{r['family']}-{r['seed']}"
        assert read(directory / "result.json") == r
        if r["status"] != "complete":
            continue
        baseline = LearnedDynamics.load(directory / "baseline.npz")
        candidate = SequenceModel.load(directory / f"{key}.npz")
        assert baseline.fingerprint() == r["baseline_fingerprint"]
        assert candidate.fingerprint() == r.get(
            "candidate_fingerprint", r.get("pooled_fingerprint")
        )
        assert baseline.report == read(directory / "baseline-report.json")
        assert baseline.report["recipe"] == plan["recipe"]
        assert (
            candidate.kind == baseline._model.kind
            and candidate.history_steps == baseline.history_steps == 2
        )
        assert candidate.dt_s == baseline._model.dt_s == 0.05
        for name in candidate.norms:
            check(candidate.norms[name], baseline._model.norms[name])
        calibration, _ = generate(plan, r["family"], r["seed"], "calibration")
        source = {(s.recording_id, s.segment_id): s for s in calibration.segments}
        names = sorted(
            {s.recording_id for s in calibration.segments},
            key=lambda name: hashlib.sha256(
                json.dumps(name, sort_keys=True).encode()
            ).hexdigest(),
        )
        expected_roles = dict(development=set(names[:2]), train=set(names[2:]))
        for role in ("train", "development"):
            cache = getattr(baseline, "_" + role)
            expected_result_key = (
                "training_keys" if role == "train" else "development_keys"
            )
            assert [[k.recording_id, k.segment_id, k.origin] for k in cache.keys] == r[
                expected_result_key
            ]
            assert {k.recording_id for k in cache.keys} == expected_roles[role]
            assert len(cache.keys) == (384 if role == "train" else 256)
            cache_windows += len(cache.keys)
            for name, attribute, offsets in (
                ("past_states", "states", np.arange(-2, 1)),
                ("past_inputs", "inputs", np.arange(-2, 0)),
                ("future_states", "states", np.arange(1, 6)),
                ("future_inputs", "inputs", np.arange(5)),
            ):
                expected = np.stack(
                    [
                        getattr(source[(k.recording_id, k.segment_id)], attribute)[
                            k.origin + offsets
                        ]
                        for k in cache.keys
                    ]
                )
                check(getattr(cache.batch, name), expected)
        train, dev = baseline._train.batch, baseline._development.batch
        current = np.concatenate((train.past_states, train.future_states), axis=1)[
            :, 2:-1
        ]
        standard = current.std(axis=(0, 1))
        standard = np.where(standard > 1e-8, standard, 1.0)
        check(standard, baseline._model.norms["state_scale"])
        check(current.mean(axis=(0, 1)), baseline._model.norms["state_mean"])
        reference = np.mean(
            (train.future_states - train.past_states[:, -1:]) ** 2, axis=0
        )
        original_scale = np.maximum(np.sqrt(reference), 0.01 * standard)
        check(original_scale, baseline.report["optimization"]["error_scale"])
        if key == "pooled":
            expected_scale = np.tile(
                np.maximum(np.sqrt(reference.mean(axis=0)), 0.01 * standard), (5, 1)
            )
        else:
            # Express the cap in squared-error weights, independently of max(scales).
            weights = 1 / original_scale**2
            expected_scale = 1 / np.sqrt(np.minimum(weights, weights[0]))
            changed = original_scale < original_scale[0]
            assert int(changed.sum()) == r["changed_scale_entries"]
            assert bool(not changed.any()) == r["identical_objective_reused"]
            if not changed.any():
                assert candidate.fingerprint() == baseline._model.fingerprint()
        candidate_report = read(directory / f"{key}-report.json")
        check(expected_scale, candidate_report["error_scale"])
        for model, report in (
            (baseline._model, baseline.report["optimization"]),
            (candidate, candidate_report),
        ):
            predicted = recurrence(
                model, dev.past_states, dev.past_inputs, dev.future_inputs
            )
            value = np.mean(
                ((predicted - dev.future_states) / np.asarray(report["error_scale"]))
                ** 2
            )
            check(value, report["validation_rollout_mse"])
            assert (
                report["selected_step"]
                == min(report["trace"], key=lambda t: t["validation_rollout_mse"])[
                    "step"
                ]
            )
        training_u = train.future_inputs.reshape(-1, train.future_inputs.shape[-1])
        for regime, recorded in r["regimes"].items():
            supplied, _ = generate(plan, r["family"], r["seed"], regime)
            assert not {s.recording_id for s in supplied.segments} & set(baseline._seen)
            index = [
                (s, t)
                for s in supplied.segments
                for t in range(2, len(s.states) - 5, 5)
            ]
            assert len(index) == recorded["windows"] == 124
            evaluation_windows += len(index)
            with np.load(directory / f"{regime}.npz") as z:
                arrays = {name: z[name] for name in z.files}
            for name, expected in (
                ("past_states", [s.states[t + np.arange(-2, 1)] for s, t in index]),
                ("past_inputs", [s.inputs[t + np.arange(-2, 0)] for s, t in index]),
                ("future_inputs", [s.inputs[t + np.arange(5)] for s, t in index]),
                ("targets", [s.states[t + np.arange(1, 6)] for s, t in index]),
                ("source_origins", [s.start_row + t for s, t in index]),
            ):
                check(arrays[name], expected)
            assert arrays["recording_ids"].tolist() == [
                s.recording_id for s, _ in index
            ]
            check(arrays["state_scale"], standard)
            for label, model in (
                ("baseline", baseline._model),
                (key, candidate),
                ("hold", None),
            ):
                if model is None:
                    predicted = np.repeat(arrays["past_states"][:, -1:], 5, axis=1)
                else:
                    predicted = recurrence(
                        model,
                        arrays["past_states"],
                        arrays["past_inputs"],
                        arrays["future_inputs"],
                    )
                check(arrays[label], predicted)
                compare(
                    scores(predicted, arrays["targets"], standard),
                    recorded["metrics"][label],
                )
                for name in recorded["recordings"]:
                    selected = arrays["recording_ids"] == name
                    compare(
                        scores(
                            predicted[selected], arrays["targets"][selected], standard
                        ),
                        recorded["recordings"][name][label],
                    )
            inside = (arrays["future_inputs"] >= training_u.min(0)) & (
                arrays["future_inputs"] <= training_u.max(0)
            )
            check(
                inside.mean(axis=(0, 1)),
                recorded["future_input_in_training_marginal_range_fraction"],
            )
            for metric in ("first_step", "overall"):
                base, other = recorded["metrics"]["baseline"], recorded["metrics"][key]
                a, b = (
                    (base["horizon_scaled_rmse"][0], other["horizon_scaled_rmse"][0])
                    if metric == "first_step"
                    else (base["overall_scaled_rmse"], other["overall_scaled_rmse"])
                )
                flag = (
                    b - a > plan["screening"]["absolute_scaled_rmse_increase"]
                ) and (b / a - 1 > plan["screening"]["relative_rmse_increase"])
                assert bool(flag) == recorded["material_regression"][metric]
    audit = dict(
        cases=len(rows),
        complete=sum(r["status"] == "complete" for r in rows),
        cached_windows=cache_windows,
        evaluation_windows=evaluation_windows,
        numeric_comparisons=count,
        max_absolute_difference=maximum,
        relative_tolerance=1e-8,
        absolute_tolerance=1e-9,
        method="Archived-source/model fingerprints; calibration windows regenerated by saved identities; independent whole-recording split and train-only state scales; both loss definitions; NumPy evaluation/development replay; all aggregate/per-recording scores, marginal input ranges, and regression flags. Exact parameter identity verified when first-step floor leaves weights unchanged.",
        limits="Shares frozen generators, artifact loaders, and earlier independent NumPy recurrence. Does not repeat optimization. Synthetic evidence and research screening constants do not establish deployment readiness.",
    )
    write(args.output / "audit.json", audit)
    summary = summarize(rows, plan, key)
    write(args.output / "summary.json", summary)
    plot(summary, args.output / "horizon-generalization.png")
    print(json.dumps(audit, indent=2))
    print(json.dumps(summary["counts"], indent=2))


def summarize(rows, plan, key):
    families = {}
    counts = {
        regime: dict(
            cases=0,
            first_improved=0,
            overall_improved=0,
            first_worse=0,
            overall_worse=0,
            first_unchanged=0,
            overall_unchanged=0,
            first_material_regressions=0,
            overall_material_regressions=0,
        )
        for regime in ("matched", "shifted")
    }
    for family in plan["families"]:
        selected = [
            r for r in rows if r["family"] == family and r["status"] == "complete"
        ]
        families[family] = {}
        for regime in counts:
            values = []
            for r in selected:
                metrics = r["regimes"][regime]["metrics"]
                baseline, candidate = metrics["baseline"], metrics[key]
                first = (
                    baseline["horizon_scaled_rmse"][0],
                    candidate["horizon_scaled_rmse"][0],
                )
                overall = (
                    baseline["overall_scaled_rmse"],
                    candidate["overall_scaled_rmse"],
                )
                counts[regime]["cases"] += 1
                for name, (a, b) in (("first", first), ("overall", overall)):
                    category = (
                        "unchanged"
                        if abs(a - b) <= 1e-12
                        else "improved"
                        if b < a
                        else "worse"
                    )
                    counts[regime][name + "_" + category] += 1
                flags = r["regimes"][regime]["material_regression"]
                counts[regime]["first_material_regressions"] += int(flags["first_step"])
                counts[regime]["overall_material_regressions"] += int(flags["overall"])
                values.append(
                    dict(
                        first_relative_change=first[1] / first[0] - 1,
                        overall_relative_change=overall[1] / overall[0] - 1,
                        first_absolute_scaled_change=first[1] - first[0],
                        overall_absolute_scaled_change=overall[1] - overall[0],
                        first_baseline=first[0],
                        first_candidate=first[1],
                        overall_baseline=overall[0],
                        overall_candidate=overall[1],
                        horizon_relative_change=(
                            np.array(candidate["horizon_scaled_rmse"])
                            / baseline["horizon_scaled_rmse"]
                            - 1
                        ).tolist(),
                    )
                )
            families[family][regime] = (
                {
                    name: dict(
                        minimum=np.min([v[name] for v in values], axis=0).tolist(),
                        median=np.median([v[name] for v in values], axis=0).tolist(),
                        maximum=np.max([v[name] for v in values], axis=0).tolist(),
                    )
                    for name in values[0]
                }
                if values
                else {}
            )
    return dict(
        candidate_key=key,
        phase=plan.get("phase", "pooled_screen"),
        families=families,
        counts=counts,
        reused_models=sum(r.get("identical_objective_reused", False) for r in rows),
        candidate_refits=sum(
            r["status"] == "complete" and not r.get("identical_objective_reused", False)
            for r in rows
        ),
        limits="Three data seeds, not confidence intervals. Relative changes can look large when physical errors are tiny; absolute scaled changes and all per-channel scores are retained. Confirmation shares equation families with development.",
    )


def plot(summary, path):
    labels = [name.replace("_", " ") for name in summary["families"]]
    values = np.array(
        [
            [
                100 * summary["families"][family][regime][metric]["median"]
                for regime in ("matched", "shifted")
                for metric in ("first_relative_change", "overall_relative_change")
            ]
            for family in summary["families"]
        ]
    )
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    limit = max(10, float(np.max(np.abs(values))))
    im = ax.imshow(values, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xticks(
        range(4),
        [
            "Matched\nFirst step",
            "Matched\nOverall",
            "Shifted\nFirst step",
            "Shifted\nOverall",
        ],
    )
    for row in range(len(labels)):
        for col in range(4):
            ax.text(
                col,
                row,
                f"{values[row, col]:+.1f}%",
                ha="center",
                va="center",
                color="white" if abs(values[row, col]) > limit * 0.65 else "black",
            )
    title = (
        "Pooled normalization"
        if summary["candidate_key"] == "pooled"
        else "First-step floor: " + summary["phase"]
    )
    ax.set_title(title + " — median RMSE change across three seeds", pad=15)
    fig.colorbar(im, ax=ax, label="RMSE change (%); positive is worse")
    fig.text(
        0.5,
        0.025,
        "Synthetic, paired model comparisons. Relative percentages need the accompanying absolute errors.",
        ha="center",
        fontsize=9,
        color=".35",
    )
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    main(p.parse_args())
