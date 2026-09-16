"""Independently replay checkpoint paths, cross-selection, and paired contrasts."""

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

from glassbox.experimental.sequence_model import SequenceModel

OBJECTIVES = ("original", "first_step_floor")


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def metric(predicted, target, scale):
    squared = (predicted - target) ** 2
    normalized = squared / scale**2
    return dict(
        channel_rmse=np.sqrt(squared.mean(axis=0)).tolist(),
        horizon_scaled_rmse=np.sqrt(normalized.mean(axis=(0, 2))).tolist(),
        overall_scaled_rmse=float(np.sqrt(normalized.mean())),
    )


def contrasts(cells):
    a, b, c, d = (np.asarray(cells[k]) for k in ("00", "01", "10", "11"))
    return {
        key: value.tolist()
        for key, value in dict(
            optimization_original_selector=c - a,
            selection_floor_optimizer=d - c,
            selection_original_optimizer=b - a,
            optimization_floor_selector=d - b,
            interaction=d - c - b + a,
            combined=d - a,
        ).items()
    }


def score_vector(scores):
    return [*scores["horizon_scaled_rmse"], scores["overall_scaled_rmse"]]


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    plan, rows = read(args.run / "plan.json"), read(args.run / "results.json")
    stage = rows[0]["stage"]
    specification = plan["stages"][stage]
    assert {(r["family"], r["data_seed"], r["fit_seed"]) for r in rows} == {
        (f, d, s)
        for f in specification["families"]
        for d in specification["data_seeds"]
        for s in specification["fit_seeds"]
    }
    assert len(rows) == len(specification["families"]) * len(
        specification["data_seeds"]
    ) * len(specification["fit_seeds"])
    with zipfile.ZipFile(args.run / "executed-sources.zip") as archive:
        for name, expected in read(args.run / "sources.json").items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == expected
    comparisons, maximum, cached_windows, eval_windows, snapshots = 0, 0.0, 0, 0, 0

    def check(actual, expected):
        nonlocal comparisons, maximum
        a, b = np.asarray(actual), np.asarray(expected)
        assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
        maximum = max(maximum, float(np.max(np.abs(a - b))))
        np.testing.assert_allclose(a, b, atol=1e-9, rtol=1e-8)
        comparisons += 1

    def compare(actual, expected):
        if isinstance(actual, dict):
            assert actual.keys() == expected.keys()
            for key in actual:
                compare(actual[key], expected[key])
        elif isinstance(actual, list) and actual and isinstance(actual[0], dict):
            assert len(actual) == len(expected)
            for a, b in zip(actual, expected, strict=True):
                compare(a, b)
        else:
            check(actual, expected)

    summaries = []
    for row in rows:
        name = f"{row['family']}-{row['data_seed']}-fit{row['fit_seed']}"
        directory = args.run / name
        assert row == read(directory / "result.json")
        assert row["selected"] == read(directory / "selection.json")
        supplied, _ = generate(
            plan["dataset"], row["family"], row["data_seed"], "calibration"
        )
        records = {s.recording_id: s for s in supplied.segments}

        def hashed(value):
            return hashlib.sha256(
                json.dumps(value, sort_keys=True).encode()
            ).hexdigest()

        ordered = sorted(records, key=hashed)
        roles = dict(train=ordered[2:], development=ordered[:2])
        caches = {}
        for role, names in roles.items():
            with np.load(directory / f"{role}.npz", allow_pickle=False) as a:
                cache = {k: a[k] for k in a.files}
            budget = 384 if role == "train" else 256
            available = {
                record: sorted(
                    range(2, len(records[record].states) - 5),
                    key=lambda t: hashed([record, "whole", t]),
                )
                for record in names
            }
            chosen = [
                (record, available[record][depth])
                for depth in range(min(map(len, available.values())))
                for record in sorted(names)
            ][:budget]
            assert (
                list(zip(cache["recording_ids"], cache["origins"], strict=True))
                == chosen
            )
            assert set(cache["segment_ids"]) == {"whole"}
            for key, attribute, offsets in (
                ("past_states", "states", np.arange(-2, 1)),
                ("past_inputs", "inputs", np.arange(-2, 0)),
                ("future_inputs", "inputs", np.arange(5)),
                ("future_states", "states", np.arange(1, 6)),
            ):
                check(
                    cache[key],
                    np.stack(
                        [
                            getattr(records[record], attribute)[t + offsets]
                            for record, t in chosen
                        ]
                    ),
                )
            caches[role] = cache
            cached_windows += len(chosen)
        train, dev = caches["train"], caches["development"]
        current = np.concatenate(
            (train["past_states"], train["future_states"]), axis=1
        )[:, 2:-1]
        state_scale = current.std(axis=(0, 1))
        state_scale = np.where(state_scale > 1e-8, state_scale, 1.0)
        original = np.maximum(
            np.sqrt(
                np.mean(
                    (train["future_states"] - train["past_states"][:, -1:]) ** 2, axis=0
                )
            ),
            0.01 * state_scale,
        )
        floor = 1 / np.sqrt(np.minimum(1 / original**2, 1 / original[0] ** 2))
        scales = dict(original=original, first_step_floor=floor)
        compare(row["loss_scales"], {k: v.tolist() for k, v in scales.items()})
        unchanged = bool(np.all(original >= original[0]))
        assert unchanged == row["identical_objective_reused"]
        traces = read(directory / "traces.json")
        with np.load(
            directory / "development-predictions.npz", allow_pickle=False
        ) as p:
            saved_dev = {k: p[k] for k in p.files}
        models = {}
        for objective in OBJECTIVES:
            models[objective] = [
                SequenceModel.load(directory / f"{objective}-{step:04d}.npz")
                for step in plan["checkpoints"]
            ]
            replay = []
            for model in models[objective]:
                assert (
                    model.kind == "delay_mlp"
                    and model.dt_s == 0.05
                    and model.history_steps == 2
                )
                check(model.norms["state_scale"], state_scale)
                check(model.norms["state_mean"], current.mean(axis=(0, 1)))
                replay.append(
                    recurrence(
                        model,
                        dev["past_states"],
                        dev["past_inputs"],
                        dev["future_inputs"],
                    )
                )
                snapshots += 1
            replay = np.stack(replay)
            check(saved_dev[objective], replay)
            squared = (replay - dev["future_states"]) ** 2
            for criterion, loss_scale in scales.items():
                # Divide mean squared errors by squared scales after averaging windows.
                losses = (squared.mean(axis=1) / loss_scale**2).mean(axis=(1, 2))
                expected = row["selected"][objective][criterion]
                check(expected["losses"], losses)
                assert expected["selected_index"] == int(np.argmin(losses))
                for record in sorted(set(dev["recording_ids"])):
                    mask = dev["recording_ids"] == record
                    record_losses = (
                        squared[:, mask].mean(axis=1) / loss_scale**2
                    ).mean(axis=(1, 2))
                    check(expected["recording_losses"][record], record_losses)
                    excluded_losses = (
                        squared[:, ~mask].mean(axis=1) / loss_scale**2
                    ).mean(axis=(1, 2))
                    assert expected["leave_one_recording_out_indices"][record] == int(
                        np.argmin(excluded_losses)
                    )
            check(
                [t["validation_rollout_mse"] for t in traces[objective]],
                row["selected"][objective][objective]["losses"],
            )
        assert (
            models["original"][0].fingerprint()
            == models["first_step_floor"][0].fingerprint()
        )
        if unchanged:
            assert all(
                a.fingerprint() == b.fingerprint()
                for a, b in zip(*models.values(), strict=True)
            )
        parity = row["parity"]
        if stage == "known" and row["fit_seed"] == 0:
            assert parity.keys() == set(OBJECTIVES)
            for objective in OBJECTIVES:
                index = row["selected"][objective][objective]["selected_index"]
                assert (
                    models[objective][index].fingerprint()
                    == parity[objective]["fingerprint"]
                )
                assert plan["checkpoints"][index] == parity[objective]["selected_step"]
        else:
            assert not parity
        case_summary = dict(
            name=name,
            family=row["family"],
            data_seed=row["data_seed"],
            fit_seed=row["fit_seed"],
            identical_objective_reused=unchanged,
            selected_steps={
                o: {
                    c: plan["checkpoints"][v["selected_index"]]
                    for c, v in selected.items()
                }
                for o, selected in row["selected"].items()
            },
            leave_one_recording_out_steps={
                o: {
                    c: {
                        r: plan["checkpoints"][i]
                        for r, i in v["leave_one_recording_out_indices"].items()
                    }
                    for c, v in selected.items()
                }
                for o, selected in row["selected"].items()
            },
            regimes={},
        )
        for regime, report in row["regimes"].items():
            evaluation, _ = generate(
                plan["dataset"], row["family"], row["data_seed"], regime
            )
            assert not set(records) & {s.recording_id for s in evaluation.segments}
            index = [
                (s, t)
                for s in evaluation.segments
                for t in range(2, len(s.states) - 5, 5)
            ]
            with np.load(directory / f"{regime}.npz", allow_pickle=False) as a:
                arrays = {k: a[k] for k in a.files}
            assert report["windows"] == len(index) == 124
            assert arrays["recording_ids"].tolist() == [
                s.recording_id for s, _ in index
            ]
            check(arrays["source_origins"], [s.start_row + t for s, t in index])
            check(arrays["state_scale"], state_scale)
            for key, attribute, offsets in (
                ("past_states", "states", np.arange(-2, 1)),
                ("past_inputs", "inputs", np.arange(-2, 0)),
                ("future_inputs", "inputs", np.arange(5)),
                ("targets", "states", np.arange(1, 6)),
            ):
                check(
                    arrays[key],
                    np.stack([getattr(s, attribute)[t + offsets] for s, t in index]),
                )
            eval_windows += len(index)
            for objective, path in models.items():
                values = np.stack(
                    [
                        recurrence(
                            m,
                            arrays["past_states"],
                            arrays["past_inputs"],
                            arrays["future_inputs"],
                        )
                        for m in path
                    ]
                )
                check(arrays[objective], values)
                expected = dict(
                    checkpoints=[
                        metric(p, arrays["targets"], state_scale) for p in values
                    ],
                    recordings={
                        str(r): [
                            metric(
                                p[arrays["recording_ids"] == r],
                                arrays["targets"][arrays["recording_ids"] == r],
                                state_scale,
                            )
                            for p in values
                        ]
                        for r in sorted(set(arrays["recording_ids"]))
                    },
                )
                compare(report["metrics"][objective], expected)
            cells = {
                f"{i}{j}": score_vector(
                    report["metrics"][o]["checkpoints"][
                        row["selected"][o][c]["selected_index"]
                    ]
                )
                for i, o in enumerate(OBJECTIVES)
                for j, c in enumerate(OBJECTIVES)
            }
            final = {
                o: score_vector(report["metrics"][o]["checkpoints"][-1])
                for o in OBJECTIVES
            }
            effects = contrasts(cells)
            check(
                np.array(effects["optimization_original_selector"])
                + effects["selection_floor_optimizer"],
                effects["combined"],
            )
            check(
                np.array(effects["selection_original_optimizer"])
                + effects["optimization_floor_selector"],
                effects["combined"],
            )
            case_summary["regimes"][regime] = dict(
                cells=cells, fixed_final=final, contrasts=effects
            )
        summaries.append(case_summary)

    counts = {}
    for regime in ("matched", "shifted"):
        counts[regime] = {}
        for label, left, right in (
            ("combined", "00", "11"),
            ("optimization_original_selector", "00", "10"),
            ("selection_original_optimizer", "00", "01"),
            ("selection_floor_optimizer", "10", "11"),
        ):
            counts[regime][label] = {}
            for metric_name, index in (("first", 0), ("overall", 5)):
                pairs = np.array(
                    [
                        [r["regimes"][regime]["cells"][k][index] for k in (left, right)]
                        for r in summaries
                    ]
                )
                delta = pairs[:, 1] - pairs[:, 0]
                screen = (
                    pairs[:, 1]
                    > pairs[:, 0] * (1 + plan["screening"]["relative_rmse_increase"])
                ) & (delta > plan["screening"]["absolute_scaled_rmse_increase"])
                counts[regime][label][metric_name] = dict(
                    improved=int(np.sum(delta < -1e-12)),
                    worse=int(np.sum(delta > 1e-12)),
                    unchanged=int(np.sum(np.abs(delta) <= 1e-12)),
                    screen_regressions=int(screen.sum()),
                )
    summary = dict(
        stage=stage,
        cases=summaries,
        counts=counts,
        score_order=["h1", "h2", "h3", "h4", "h5", "overall"],
        optimizer_runs=sum(1 if r["identical_objective_reused"] else 2 for r in rows),
        identical_trajectories_reused=sum(
            r["identical_objective_reused"] for r in rows
        ),
        limits=plan["limits"],
    )
    audit = dict(
        cases=len(rows),
        checkpoint_models_checked=snapshots,
        cached_windows=cached_windows,
        evaluation_windows=eval_windows,
        numeric_comparisons=comparisons,
        max_absolute_difference=maximum,
        atol=1e-9,
        rtol=1e-8,
        method="Archived sources, independent recording split/window selection and time indexing; train-only loss scales; NumPy replay of every saved checkpoint on development and evaluation data; every horizon/channel/recording score, crossed and leave-one-recording-out selectors, and attribution identities. Shared frozen generator and artifact loader; no optimizer rerun.",
    )
    write(args.output / "summary.json", summary)
    write(args.output / "audit.json", audit)
    if stage == "known":
        focus = next(r for r in rows if r["data_seed"] == 5101 and r["fit_seed"] == 0)
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
        for i, criterion in enumerate(OBJECTIVES):
            ax = axes[0, i]
            for objective, color in zip(
                OBJECTIVES, ("#3465a4", "#c05a27"), strict=True
            ):
                chosen = focus["selected"][objective][criterion]
                ax.semilogy(
                    plan["checkpoints"],
                    chosen["losses"],
                    color=color,
                    label=objective.replace("_", " "),
                )
                step = chosen["selected_index"]
                ax.scatter(
                    plan["checkpoints"][step],
                    chosen["losses"][step],
                    color=color,
                    s=45,
                    zorder=3,
                )
            ax.set_title(f"Development criterion: {criterion.replace('_', ' ')}")
            ax.set_ylabel("Development normalized MSE")
        for i, regime in enumerate(("matched", "shifted")):
            ax = axes[1, i]
            for objective, color in zip(
                OBJECTIVES, ("#3465a4", "#c05a27"), strict=True
            ):
                values = [
                    v["overall_scaled_rmse"]
                    for v in focus["regimes"][regime]["metrics"][objective][
                        "checkpoints"
                    ]
                ]
                ax.semilogy(
                    plan["checkpoints"],
                    values,
                    color=color,
                    label=objective.replace("_", " "),
                )
                step = focus["selected"][objective][objective]["selected_index"]
                ax.scatter(
                    plan["checkpoints"][step], values[step], color=color, s=45, zorder=3
                )
            ax.set_title(f"{regime.capitalize()} evaluation, never used to select")
            ax.set_ylabel("Overall scaled RMSE")
        for ax in axes.flat:
            ax.set_xlabel("Optimizer step")
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
        fig.suptitle(
            "Known regression: near-periodic data seed 5101, fit seed 0\nDots mark selected checkpoints; both criteria choose the same step on each path"
        )
        fig.savefig(args.output / "known-regression.png", dpi=160)
        plt.close(fig)
    print(
        json.dumps(
            dict(audit=audit, counts=counts, optimizer_runs=summary["optimizer_runs"])
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
