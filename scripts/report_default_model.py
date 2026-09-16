"""Independent sampling, recursive prediction, and revision audit for the fixed recipe."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from report_model_structures import recurrence
from sequence_transfer_data import write_json

from glassbox.core.data import load_trajectory_npz
from glassbox.core.geometry import quaternion_to_rotation_matrices
from glassbox.experimental.default_model import LearnedDynamics
from glassbox.experimental.sequence_model import SequenceModel

ARRAYS = ("past_states", "past_inputs", "future_inputs", "future_states")


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def priority(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def select(pool, count):
    records = sorted({r for r, _ in pool})
    ordered = {
        r: sorted(
            [row for rr, row in pool if rr == r],
            key=lambda row: priority([r, "prepared", row]),
        )
        for r in records
    }
    chosen = []
    for depth in range(max(map(len, ordered.values()))):
        for r in records:
            if depth < len(ordered[r]):
                chosen.append((r, ordered[r][depth]))
            if len(chosen) >= count:
                return chosen
    return chosen


def group_rmse(pred, target, groups):
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
    checks, maximum, snapshots, windows_checked, normal_equations = 0, 0.0, 0, 0, 0

    def check(x, y):
        nonlocal checks, maximum
        x, y = np.asarray(x), np.asarray(y)
        assert x.shape == y.shape
        maximum = max(maximum, float(np.max(np.abs(x - y))))
        np.testing.assert_allclose(x, y, atol=5e-8, rtol=5e-8)
        checks += 1

    with zipfile.ZipFile(a.run / "executed-sources.zip") as z:
        for n, h in read(a.run / "sources.json").items():
            assert hashlib.sha256(z.read(n)).hexdigest() == h
    recipe = read(a.run / "recipe.json")
    summaries = []
    for folder in sorted(a.run.iterdir()):
        if not folder.is_dir():
            continue
        report = read(folder / "report.json")
        records = read(folder / "records.json")
        data = {}
        with np.load(folder / "records.npz") as z:
            for i, record in enumerate(records):
                assert sha(Path(record["source"])) == record["sha256"]
                data[record["name"]] = {k: z[f"{i}_{k}"] for k in ("states", "inputs")}
                if report["case"] == "nano":
                    flight = load_trajectory_npz(record["source"])
                    expected = np.column_stack(
                        (
                            flight.states[:, 3:6],
                            flight.states[:, 10:13],
                            quaternion_to_rotation_matrices(
                                flight.states[:, 6:10]
                            ).reshape(-1, 9),
                        )
                    )
                    check(data[record["name"]]["states"], expected)
                    check(
                        data[record["name"]]["inputs"], np.sqrt(flight.controls) * 2500
                    )
                else:
                    with np.load(record["source"]) as raw:
                        for k in ("states", "inputs"):
                            check(data[record["name"]][k], raw[k])
        original = [r["name"] for r in records if r["role"] == "calibration"]
        new = [r["name"] for r in records if r["role"] == "update"]
        evaluated = [r["name"] for r in records if r["role"] == "evaluation"]
        if new:
            assert new == [max(original + new)]
        ndev = min(len(original) - 1, max(1, int(np.ceil(len(original) / 4))))
        dev_names = sorted(original, key=priority)[:ndev]
        train_names = [r for r in original if r not in dev_names]
        previous_model, previous_train = None, None
        for revision in ("initial", "updated"):
            if revision not in report["models"]:
                continue
            model = LearnedDynamics.load(folder / f"{revision}.npz")
            assert model.fingerprint() == report["models"][revision]["fingerprint"]
            assert model.report == report["models"][revision]["report"]
            assert model.contract == report["contract"]
            assert set(model._seen) == set(
                original if revision == "initial" else original + new
            )
            assert model.report["recipe"] == recipe
            assert (
                model._model.kind == "delay_mlp"
                and model._model.params["w1"].shape[1] == 32
            )
            assert model._model.dt_s == report["dt_s"]
            p, h = model.history_steps, model.horizon_steps
            assert p == max(1, int(np.rint(0.1 / report["dt_s"]))) and h == max(
                1, int(np.rint(0.25 / report["dt_s"]))
            )

            def pool(names, p=p, h=h, data=data):
                return [
                    (r, i) for r in names for i in range(p, len(data[r]["states"]) - h)
                ]

            expected_train = (
                select(pool(train_names), 384)
                if revision == "initial"
                else select(previous_train + select(pool(new), 384), 384)
            )
            for role, windows, expected in (
                ("training", model._train, expected_train),
                ("development", model._development, select(pool(dev_names), 256)),
            ):
                actual = [
                    (k.recording_id, row)
                    for k, row in zip(windows.keys, windows.source_origins)
                ]
                assert actual == expected
                ids = {r for r, _ in actual}
                assert not ids & set(evaluated)
                for key, (_, row) in zip(windows.keys, actual):
                    assert key.origin == row and key.segment_id == "prepared"
                for k, field, offset in [
                    ("past_states", "states", np.arange(-p, 1)),
                    ("past_inputs", "inputs", np.arange(-p, 0)),
                    ("future_states", "states", np.arange(1, h + 1)),
                    ("future_inputs", "inputs", np.arange(h)),
                ]:
                    check(
                        getattr(windows.batch, k),
                        np.array([data[r][field][row + offset] for r, row in actual]),
                    )
                for r in ids:
                    rows = [row for rr, row in actual if rr == r]
                    state_rows = set(
                        v for row in rows for v in range(row - p, row + h + 1)
                    )
                    input_rows = set(v for row in rows for v in range(row - p, row + h))
                    evidence = model.report[role][r]
                    assert (
                        evidence["windows"] == len(rows) and evidence["segments"] == 1
                    )
                    assert evidence["unique_state_rows"] == len(
                        state_rows
                    ) and evidence["unique_input_rows"] == len(input_rows)
                    check(
                        evidence["observed_transition_time_s"],
                        len(input_rows) * report["dt_s"],
                    )
                windows_checked += len(actual)
            if previous_model is not None:
                assert model.report["previous_revision"] == previous_model.fingerprint()
                assert model._development.keys == previous_model._development.keys
                for k in ARRAYS:
                    check(
                        getattr(model._development.batch, k),
                        getattr(previous_model._development.batch, k),
                    )
            train, dev = model._train.batch, model._development.batch
            current = np.concatenate(
                (train.past_states[:, -1:], train.future_states[:, :-1]), 1
            )
            std = current.std((0, 1))
            check(model._model.norms["state_scale"], np.where(std > 1e-8, std, 1))
            scale = np.maximum(
                np.sqrt(
                    np.mean((train.past_states[:, -1:] - train.future_states) ** 2, 0)
                ),
                0.01 * model._model.norms["state_scale"],
            )
            dev_pred = recurrence(
                model._model, dev.past_states, dev.past_inputs, dev.future_inputs
            )
            loss = np.mean(((dev_pred - dev.future_states) / scale) ** 2)
            optimization = model.report["optimization"]
            check(loss, optimization["validation_rollout_mse"])
            assert optimization["steps"] == 1000 and optimization["seed"] == 0
            best = min(optimization["trace"], key=lambda r: r["validation_rollout_mse"])
            assert best["step"] == optimization["selected_step"]
            check(best["validation_rollout_mse"], loss)
            ids = np.array([k.recording_id for k in model._development.keys])
            for r, evidence in model.report["development_errors"].items():
                mask = ids == r
                err = np.sqrt(
                    np.mean((dev_pred[mask] - dev.future_states[mask]) ** 2, 0)
                )
                hold = np.sqrt(
                    np.mean(
                        (dev.past_states[mask, -1:] - dev.future_states[mask]) ** 2, 0
                    )
                )
                check(err, evidence["channel_rmse"])
                check(hold, evidence["hold_channel_rmse"])
                np.testing.assert_array_equal(
                    err > hold * 1.05 + 1e-12, evidence["worse_than_hold"]
                )
            previous_model, previous_train = model, expected_train
            snapshots += 1
        initial = LearnedDynamics.load(folder / "initial.npz")
        base = SequenceModel.load(folder / "initialization.npz")
        b = initial._train.batch
        norm = base.norms
        xs = np.concatenate((b.past_states, b.future_states), 1)
        us = np.concatenate((b.past_inputs, b.future_inputs), 1)
        x = (xs - norm["state_mean"]) / norm["state_scale"]
        u = (us - norm["input_mean"]) / norm["input_scale"]
        features = np.stack(
            [
                np.concatenate(
                    (
                        x[:, p + t],
                        u[:, p + t],
                        (x[:, t : t + p] - x[:, p + t, None]).reshape(len(x), -1),
                        (u[:, t : t + p] - u[:, p + t, None]).reshape(len(x), -1),
                    ),
                    -1,
                )
                for t in range(h)
            ],
            1,
        ).reshape(-1, len(norm["feature_scale"]))
        feature_scale = features.std(0)
        check(norm["feature_scale"], np.where(feature_scale > 1e-8, feature_scale, 1))
        design = np.column_stack(
            (features / norm["feature_scale"], np.ones(len(features)))
        )
        target = (
            (xs[:, p + 1 :] - xs[:, p:-1]) / norm["state_scale"] / norm["delta_scale"]
        ).reshape(len(features), -1)
        coef = np.vstack((base.params["linear"], base.params["bias"]))
        penalty = np.r_[np.full(len(coef) - 1, 0.01), 0]
        check(
            design.T @ (design @ coef - target) / len(features)
            + penalty[:, None] * coef,
            np.zeros_like(coef),
        )
        normal_equations += 1
        usage = read(folder / "evaluation-usage.json")
        with np.load(folder / "evaluation.npz") as z:
            evaluation = {k: z[k] for k in z.files}
        rows = [
            (k["recording_id"], row)
            for k, row in zip(usage["keys"], usage["source_origins"])
        ]
        assert {r for r, _ in rows} == set(evaluated)
        stride = max(1, int(np.rint(0.1 / report["dt_s"])))
        assert rows == [
            (r, row)
            for r in evaluated
            for row in range(p, len(data[r]["states"]) - h, stride)
        ]
        assert all(
            k["segment_id"] == "prepared" and k["origin"] == row
            for k, (_, row) in zip(usage["keys"], rows)
        )
        for k, field, offset in [
            ("past_states", "states", np.arange(-p, 1)),
            ("past_inputs", "inputs", np.arange(-p, 0)),
            ("future_states", "states", np.arange(1, h + 1)),
            ("future_inputs", "inputs", np.arange(h)),
        ]:
            check(
                evaluation[k],
                np.array([data[r][field][row + offset] for r, row in rows]),
            )
        for name, scores in report["rmse"].items():
            if name == "hold":
                pred = np.repeat(evaluation["past_states"][:, -1:], h, 1)
            else:
                m = (
                    base
                    if name == "initialization"
                    else LearnedDynamics.load(folder / f"{name}.npz")._model
                )
                pred = recurrence(
                    m,
                    evaluation["past_states"],
                    evaluation["past_inputs"],
                    evaluation["future_inputs"],
                )
            check(pred, evaluation[f"prediction_{name}"])
            check(
                group_rmse(pred, evaluation["future_states"], report["groups"]), scores
            )
        summaries.append(
            dict(
                case=report["case"],
                horizon_s=h * report["dt_s"],
                evaluation_windows=len(rows),
                final_rmse={n: v[-1] for n, v in report["rmse"].items()},
                selected_steps={
                    n: m["report"]["optimization"]["selected_step"]
                    for n, m in report["models"].items()
                },
                training={
                    n: m["report"]["training"] for n, m in report["models"].items()
                },
                seconds=dict(
                    initial=report["initial_seconds"], update=report["update_seconds"]
                ),
            )
        )
    audit = dict(
        serialized_revisions_replayed=snapshots,
        automatic_role_and_cache_windows_checked=windows_checked,
        affine_initializations_checked=normal_equations,
        numerical_checks=checks,
        maximum_absolute_difference=maximum,
        same_recipe_all_cases=True,
        immutable_update_and_reserved_roles_verified=True,
        source_archive_verified=True,
        interpretation="Independent automatic sampling, cached-row extraction, prediction, selected-loss and initial-normal-equation replay. Trace minima are checked; every optimizer step is not independently retrained. Prepared sources retain prior raw-clock audits, not independent physical ground truth.",
    )
    write_json(a.output / "audit.json", audit)
    write_json(a.output / "summary.json", summaries)
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), constrained_layout=True)
    for r in [r for r in summaries if r["case"].startswith("arp")]:
        for j, ax in enumerate(axes):
            baseline = r["final_rmse"]["hold"][j]
            ax.plot(
                [0, 1],
                [r["final_rmse"][n][j] / baseline for n in ("initial", "updated")],
                marker="o",
                label="log" + str(63 + int(r["case"][-1])),
            )
    for j, ax in enumerate(axes):
        ax.axhline(1, ls="--", color="black", label="Hold current")
        ax.set(
            yscale="log",
            xticks=[0, 1],
            xticklabels=["Initial fit", "After one added recording"],
            ylabel=["Velocity RMSE / hold RMSE", "Body-rate RMSE / hold RMSE"][j],
        )
        ax.grid(axis="y", which="both", alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle(
        "One fixed recipe · ARP · 240 ms\nSame evaluation recordings before and after a batch update; lower is better",
        fontsize=11,
    )
    for ext in ("png", "svg"):
        fig.savefig(a.output / f"default-recipe.{ext}", dpi=170)
    plt.close(fig)
    print(json.dumps(audit), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run", type=Path, default=Path("../artifacts/default-recipe/comparison-01")
    )
    p.add_argument("--output", type=Path, required=True)
    main(p.parse_args())
