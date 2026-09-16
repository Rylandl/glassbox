"""Independent prediction replay, source-row causality checks, and final figures."""

import argparse
import hashlib
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from experiment_sequence_transfer import ARRAYS, GROUPS
from report_model_structures import recurrence
from sequence_transfer_data import load_prepared, write_json

from glassbox.experimental.sequence_model import SequenceModel
from glassbox.experimental.sequence_objective import SequenceGuard


def read(path):
    return json.loads(path.read_text())


def rotation(q):
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        1,
    ).reshape(-1, 3, 3)


def operator_from_perturbations(model):
    """Rebuild the affine history map from basis perturbations of predictions."""
    p, d = model.history_steps, len(model.norms["state_mean"])
    width = (p + 1) * d
    past = np.broadcast_to(model.norms["state_mean"], (width + 1, p + 1, d)).copy()
    # Operator coordinates: current state, then history oldest to newest.
    for column in range(width):
        slot = p if column < d else (column - d) // d
        channel = column % d
        past[column + 1, slot, channel] += model.norms["state_scale"][channel]
    inputs = np.broadcast_to(
        model.norms["input_mean"], (width + 1, p + 1, len(model.norms["input_mean"]))
    )
    future = recurrence(model, past, inputs[:, :p], inputs[:, p:])
    advanced = np.concatenate((future, past[:, 1:]), axis=1)
    return (
        ((advanced[1:] - advanced[:1]) / model.norms["state_scale"])
        .reshape(width, width)
        .T
    )


def refreshed_predictions(model, batch):
    """Independent NumPy one-step replay with actual observations refreshed."""
    states = np.concatenate((batch["past_states"], batch["future_states"]), axis=1)
    inputs = np.concatenate((batch["past_inputs"], batch["future_inputs"]), axis=1)
    p = model.history_steps
    return np.concatenate(
        [
            recurrence(
                model,
                states[:, t : t + p + 1],
                inputs[:, t : t + p],
                inputs[:, t + p : t + p + 1],
            )
            for t in range(batch["future_states"].shape[1])
        ],
        axis=1,
    )


def main(args):
    from pyulog import ULog

    args.output.mkdir(parents=True, exist_ok=False)
    maximum, checks, models = 0.0, 0, 0

    def check(a, b):
        nonlocal maximum, checks
        a, b = np.asarray(a), np.asarray(b)
        assert a.shape == b.shape
        maximum = max(maximum, float(np.max(np.abs(a - b))) if a.size else 0)
        np.testing.assert_allclose(a, b, atol=2e-8, rtol=2e-9)
        checks += 1

    prepared = {d: load_prepared(args.prepared, d) for d in ("x8", "arp")}
    source_hashes, maximum_lookahead, source_rows = {}, -np.inf, 0
    for dataset, records in prepared.items():
        for r in records:
            meta = r["metadata"]
            raw = Path(meta["source"])
            assert hashlib.sha256(raw.read_bytes()).hexdigest() == meta["sha256"]
            source_hashes[r["name"]] = meta["sha256"]
            if dataset == "x8":
                assert hashlib.md5(raw.read_bytes()).hexdigest() == meta["expected_md5"]
                data = np.loadtxt(raw, delimiter=",")
                roll, pitch, yaw = data[:, 10], -data[:, 11], -data[:, 12]
                cr, sr, cp, sp, cy, sy = (
                    np.cos(roll / 2),
                    np.sin(roll / 2),
                    np.cos(pitch / 2),
                    np.sin(pitch / 2),
                    np.cos(yaw / 2),
                    np.sin(yaw / 2),
                )
                q = np.column_stack(
                    (
                        cy * cp * cr + sy * sp * sr,
                        cy * cp * sr - sy * sp * cr,
                        cy * sp * cr + sy * cp * sr,
                        sy * cp * cr - cy * sp * sr,
                    )
                )
                expected = np.column_stack(
                    (
                        data[:, 19:22] * [1, -1, -1],
                        data[:, 13:16] * [1, -1, -1],
                        rotation(q).reshape(-1, 9),
                    )
                )
                check(expected, r["states"])
                check(data[:-1][:, (3, 2, 1)], r["inputs"])
            else:
                log = ULog(str(raw))
                topics = {(d.name, d.multi_id): d.data for d in log.data_list}
                signals = []
                grid = r["absolute_grid_s"]
                for i, (topic, fields) in enumerate(
                    zip(meta["topics"], meta["fields"])
                ):
                    data = topics[(topic, 0)]
                    index = r["source_indices"][:, i]
                    publication = data["timestamp"].astype(float) * 1e-6
                    sample = data["timestamp_sample"].astype(float) * 1e-6
                    check(publication[index], r["publication_s"][:, i])
                    check(sample[index], r["sample_s"][:, i])
                    # Every selected source row is already published. No sample-clock retiming.
                    assert np.all(publication[index] <= grid)
                    assert np.all(sample[index] <= publication[index])
                    assert np.all(grid - publication[index] <= 0.05)
                    maximum_lookahead = max(
                        maximum_lookahead, float(np.max(publication[index] - grid))
                    )
                    values = np.column_stack([data[f] for f in fields]).astype(float)
                    valid = (
                        np.isfinite(values).all(1)
                        & (publication > 0)
                        & (sample > 0)
                        & (sample <= publication)
                    )
                    if topic == "vehicle_local_position":
                        valid &= data["v_xy_valid"].astype(bool) & data[
                            "v_z_valid"
                        ].astype(bool)
                    if topic == "vehicle_attitude":
                        valid &= np.linalg.norm(values, axis=1) > 0.5
                    available = np.flatnonzero(valid)
                    latest = available[
                        np.searchsorted(publication[available], grid, side="right") - 1
                    ]
                    np.testing.assert_array_equal(index, latest)
                    signals.append(values[index])
                    source_rows += len(index)
                transform = np.array([1, -1, -1])
                matrices = (
                    rotation(signals[2])
                    * transform[None, :, None]
                    * transform[None, None, :]
                )
                expected = np.column_stack(
                    (
                        signals[0] * transform,
                        signals[1] * transform,
                        matrices.reshape(-1, 9),
                    )
                )
                check(expected, r["states"])
                check(signals[3][:-1], r["inputs"])

    for run in (args.source, args.selection, args.refinement):
        with zipfile.ZipFile(run / "executed-sources.zip") as archive:
            if (run / "sources.json").exists():
                for name, digest in read(run / "sources.json").items():
                    assert hashlib.sha256(archive.read(name)).hexdigest() == digest

    summary = defaultdict(list)
    guard_details, selected_details = [], []

    def record_metrics(dataset, arm, role, horizons, mean, target, expected=None):
        for h in horizons:
            e = mean[:, h - 1] - target[:, h - 1]
            for label, group in zip(
                ("velocity_rmse", "rate_rmse", "rotation_rmse"), GROUPS
            ):
                metric = float(np.sqrt(np.mean(np.sum(e[:, group] ** 2, 1))))
                if expected is not None:
                    check(metric, expected["all"][str(h)][label])
                summary[f"{dataset}/{arm}/{role}/{h}/{label}"].append(metric)

    for dataset in ("nano", "x8", "arp"):
        for seed in (60, 61, 62):
            folder = args.source / dataset / f"seed{seed}"
            with np.load(folder / "samples.npz") as saved:
                batches = {
                    role: {k: saved[f"{role}_{k}"] for k in ARRAYS}
                    for role in ("train", "development", "evaluation")
                }
            usage = read(folder / "usage.json")
            if dataset != "nano":
                by_name = {r["name"]: r for r in prepared[dataset]}
                p = batches["train"]["past_inputs"].shape[1]
                h = batches["train"]["future_states"].shape[1]
                role_names = [{u["recording"] for u in usage[role]} for role in batches]
                assert not any(
                    role_names[i] & role_names[j] for i in range(3) for j in range(i)
                )
                for role, b in batches.items():
                    offset = 0
                    for u in usage[role]:
                        r = by_name[u["recording"]]
                        a = np.array(u["origins"])
                        stop = offset + len(a)
                        assert r["role"] == role
                        check(
                            b["past_states"][offset:stop],
                            r["states"][a[:, None] + np.arange(-p, 1)],
                        )
                        check(
                            b["past_inputs"][offset:stop],
                            r["inputs"][a[:, None] + np.arange(-p, 0)],
                        )
                        check(
                            b["future_inputs"][offset:stop],
                            r["inputs"][a[:, None] + np.arange(h)],
                        )
                        check(
                            b["future_states"][offset:stop],
                            r["states"][a[:, None] + np.arange(1, h + 1)],
                        )
                        offset = stop
                    assert offset == len(b["past_states"])
            else:
                prior = args.previous / f"seed{seed}" / "samples.npz"
                with np.load(prior) as saved:
                    for role, b in batches.items():
                        oldrole = "replication" if role == "evaluation" else role
                        for k in ARRAYS:
                            check(b[k], saved[f"{oldrole}_{k}"])
            all_runs = ((args.source, "initial"), (args.refinement, "refined"))
            for run, prefix in all_runs:
                case = run / dataset / f"seed{seed}"
                for path in sorted(case.glob("*-report.json")):
                    report = read(path)
                    arm = report["arm"]
                    model = SequenceModel.load(case / f"{arm}.npz")
                    assert model.fingerprint() == report["fingerprint"]
                    train = batches["train"]
                    current = np.concatenate(
                        (train["past_states"][:, -1:], train["future_states"][:, :-1]),
                        1,
                    )
                    check(model.norms["state_mean"], current.mean((0, 1)))
                    check(
                        model.norms["state_scale"],
                        np.where(current.std((0, 1)) > 1e-8, current.std((0, 1)), 1),
                    )
                    trace = report["trace"]
                    selected = min(
                        [r for r in trace if r.get("guard_accepted", True)],
                        key=lambda r: r["validation_rollout_mse"],
                    )
                    assert selected["step"] == report["selected_step"]
                    if report["selection_guard"]:
                        reference = np.asarray(trace[0]["guard_errors"])
                        for row in trace:
                            accepted = bool(
                                np.all(
                                    np.asarray(row["guard_errors"])
                                    <= reference * 1.05 + 1e-12
                                )
                            )
                            assert accepted == row["guard_accepted"]
                            assert (
                                SequenceGuard(
                                    tuple(report["horizons"]), GROUPS
                                ).accepts(np.asarray(row["guard_errors"]), reference)
                                == accepted
                            )
                    with np.load(case / f"{arm}-predictions.npz") as saved:
                        for role in ("development", "evaluation"):
                            b = batches[role]
                            mean = recurrence(
                                model,
                                b["past_states"],
                                b["past_inputs"],
                                b["future_inputs"],
                            )
                            check(mean, saved[role])
                            if role == "development":
                                check(
                                    np.mean(
                                        (
                                            (mean - b["future_states"])
                                            / np.asarray(report["error_scale"])
                                        )
                                        ** 2
                                    ),
                                    report["validation_rollout_mse"],
                                )
                                if report["selection_guard"]:
                                    errors = np.array(
                                        [
                                            [
                                                np.sqrt(
                                                    np.mean(
                                                        np.sum(
                                                            (
                                                                mean[:, h - 1, list(g)]
                                                                - b["future_states"][
                                                                    :, h - 1, list(g)
                                                                ]
                                                            )
                                                            ** 2,
                                                            1,
                                                        )
                                                    )
                                                )
                                                for g in GROUPS
                                            ]
                                            for h in report["horizons"]
                                        ]
                                    )
                                    check(errors, selected["guard_errors"])
                            record_metrics(
                                dataset,
                                f"{prefix}_{arm}",
                                role,
                                report["horizons"],
                                mean,
                                b["future_states"],
                                report["scores"][role],
                            )
                    if report["selection_guard"]:
                        guard_details.append(
                            dict(
                                dataset=dataset,
                                seed=seed,
                                phase=prefix,
                                step=report["selected_step"],
                                accepted_checkpoints=sum(
                                    t["guard_accepted"] for t in trace
                                ),
                            )
                        )
                    if dataset == "nano" and prefix == "initial" and arm == "standard":
                        old = (
                            args.previous
                            / f"seed{seed}"
                            / "delay_mlp-rollout-predictions.npz"
                        )
                        with (
                            np.load(old) as previous,
                            np.load(case / f"{arm}-predictions.npz") as actual,
                        ):
                            check(previous["replication"], actual["evaluation"])
                    models += 1
            chosen_folder = args.selection / dataset / f"seed{seed}"
            chosen = read(chosen_folder / "report.json")
            assert (
                chosen["selected"]["name"]
                == min(chosen["candidates"], key=lambda c: c["development_loss"])[
                    "name"
                ]
            )
            dev = batches["development"]
            for candidate in chosen["candidates"]:
                model = SequenceModel.load(chosen_folder / f"{candidate['name']}.npz")
                mean = recurrence(
                    model, dev["past_states"], dev["past_inputs"], dev["future_inputs"]
                )
                check(
                    np.mean(
                        ((mean - dev["future_states"]) / model.norms["state_scale"])
                        ** 2
                    ),
                    candidate["development_loss"],
                )
                operator = operator_from_perturbations(model)
                check(
                    np.max(np.abs(np.linalg.eigvals(operator))),
                    candidate["spectral_radius"],
                )
                check(
                    np.linalg.norm(
                        np.linalg.matrix_power(operator, dev["future_states"].shape[1]),
                        2,
                    ),
                    candidate["finite_horizon_operator_norm"],
                )
                models += 1
            model = SequenceModel.load(
                chosen_folder / f"{chosen['selected']['name']}.npz"
            )
            selected_details.append(
                dict(dataset=dataset, seed=seed, **chosen["selected"])
            )
            with np.load(chosen_folder / "predictions.npz") as saved:
                for role in ("development", "evaluation"):
                    b = batches[role]
                    mean = recurrence(
                        model, b["past_states"], b["past_inputs"], b["future_inputs"]
                    )
                    check(mean, saved[f"{role}_selected"])
                    check(refreshed_predictions(model, b), saved[f"{role}_teacher"])
                    original = SequenceModel.load(chosen_folder / "delay-ridge1.npz")
                    check(
                        refreshed_predictions(original, b),
                        saved[f"{role}_original_teacher"],
                    )
                    check(
                        np.repeat(
                            b["past_states"][:, -1:],
                            b["future_states"].shape[1],
                            axis=1,
                        ),
                        saved[f"{role}_hold"],
                    )
                    for arm in ("selected", "hold", "teacher", "original_teacher"):
                        record_metrics(
                            dataset,
                            arm,
                            role,
                            chosen["horizons"],
                            saved[f"{role}_{arm}"],
                            b["future_states"],
                            chosen["scores"][role][arm],
                        )
            # Independently verify both training-derived balancing scales.
            for phase, reference_folder in (
                (args.source, folder),
                (args.refinement, chosen_folder),
            ):
                if phase == args.source:
                    reference_model = SequenceModel.load(
                        reference_folder / "linear_history.npz"
                    )
                else:
                    reference_model = model
                b = batches["train"]
                predicted = recurrence(
                    reference_model,
                    b["past_states"],
                    b["past_inputs"],
                    b["future_inputs"],
                )
                expected = np.maximum(
                    np.sqrt(np.mean((predicted - b["future_states"]) ** 2, 0)),
                    0.001 * reference_model.norms["state_scale"],
                )
                with np.load(
                    phase / dataset / f"seed{seed}" / "objective.npz"
                ) as objective:
                    check(expected, objective["error_scale"])
    stats = {
        k: dict(
            mean=float(np.mean(v)), minimum=float(np.min(v)), maximum=float(np.max(v))
        )
        for k, v in summary.items()
    }
    write_json(args.output / "summary.json", stats)
    write_json(
        args.output / "selection.json",
        dict(guards=guard_details, affine_selection=selected_details),
    )
    write_json(
        args.output / "audit.json",
        dict(
            models_replayed=models,
            numerical_checks=checks,
            maximum_absolute_difference=maximum,
            causal_source_rows_checked=source_rows,
            maximum_source_lookahead_s=maximum_lookahead,
            source_hashes=source_hashes,
            nano_previous_prediction_reproduction=True,
            teacher_refresh_predictions_replayed=True,
            affine_operators_rebuilt_from_prediction_perturbations=True,
            interpretation="Prediction/data/selection replay; not independent re-optimization or physical-truth validation.",
        ),
    )
    plt.rcParams.update(
        {"font.size": 9, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
    arms = (
        "initial_linear_history",
        "initial_standard",
        "selected",
        "refined_standard",
        "refined_guarded",
        "hold",
    )
    labels = (
        "Initial linear",
        "Initial nonlinear",
        "Selected linear",
        "Refined nonlinear",
        "Guarded refinement",
        "Hold current",
    )
    colors = ("#4477aa", "#ee9944", "#8877bb", "#cc6677", "#228866", "#999999")
    for row, (dataset, h, display) in enumerate(
        (
            ("nano", 25, "Processed Nano · 250 ms"),
            ("x8", 10, "Skywalker X8 · 250 ms"),
            ("arp", 12, "Causal ARP estimator logs · 240 ms"),
        )
    ):
        for col, (metric, label) in enumerate(
            (
                ("velocity_rmse", "Velocity RMSE [m/s]"),
                ("rate_rmse", "Body-rate RMSE [rad/s]"),
            )
        ):
            ax = axes[row, col]
            points = [stats[f"{dataset}/{a}/evaluation/{h}/{metric}"] for a in arms]
            ax.bar(
                np.arange(len(arms)),
                [p["mean"] for p in points],
                color=colors,
                yerr=[
                    [p["mean"] - p["minimum"] for p in points],
                    [p["maximum"] - p["mean"] for p in points],
                ],
                capsize=3,
            )
            ax.set(
                xticks=np.arange(len(arms)),
                xticklabels=labels,
                ylabel=label,
                title=display,
                ylim=(0, None),
            )
            ax.tick_params(axis="x", labelrotation=35, labelsize=8)
            for tick in ax.get_xticklabels():
                tick.set_ha("right")
            ax.grid(axis="y", alpha=0.2)
    fig.suptitle(
        "One learning recipe across three telemetry datasets\nMeans and sampling-seed ranges; includes adaptive model-selection follow-ups",
        fontsize=13,
    )
    for extension in ("png", "svg"):
        fig.savefig(args.output / f"sequence-transfer.{extension}", dpi=160)
    plt.close(fig)
    print(json.dumps(read(args.output / "audit.json")), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--refinement", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument(
        "--previous",
        type=Path,
        default=Path("../artifacts/model-structures/followup-01/sequence"),
    )
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
