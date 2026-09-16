"""Two distinct evidence tracks: matched direct forecasts and recursive sequences.

Development flights select hyperparameters/checkpoints. Replication flights do
not. Neither group is pristine: both appeared in previous investigations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
import zipfile
from dataclasses import replace
from pathlib import Path

import jax
import numpy as np
from experiment_real_transition import score
from experiment_transition_diagnosis import load_records

from glassbox.experimental.sequence_model import (
    SequenceBatch,
    fit_sequence_model,
    sequence_windows,
)
from glassbox.experimental.structured_regression import (
    covariance,
    fit_structured_regressor,
)

ROLES = ("train", "development", "replication")
LENGTHS = (0.5, 1.0, 2.0)
REGULARIZATIONS = (0.01, 0.1, 1.0)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def numpy_predict(model, features):
    a = model.arrays
    z = (features - a["feature_mean"]) / a["feature_scale"]
    y = z @ a["linear"]
    if model.kind != "linear":
        y += covariance(z, a["features"], model.kind, model.length_scale) @ a["alpha"]
    return y * a["target_scale"] + a["target_mean"]


def select_regressor(train_x, train_y, dev_x, dev_y, kind):
    """All normalizers and fits use train rows; development only selects a grid."""
    base = fit_structured_regressor(train_x, train_y, kind="linear")
    if kind == "linear":
        return base, []
    a = base.arrays
    z = (dev_x - a["feature_mean"]) / a["feature_scale"]
    residual = (train_y - a["target_mean"]) / a["target_scale"] - a["features"] @ a[
        "linear"
    ]
    dev_target = (dev_y - a["target_mean"]) / a["target_scale"]
    trend = z @ a["linear"]
    best, best_loss, trials = None, np.inf, []
    for length in LENGTHS:
        kernel = covariance(a["features"], a["features"], kind, length)
        cross = covariance(z, a["features"], kind, length)
        for regularization in REGULARIZATIONS:
            alpha = np.linalg.solve(
                kernel + regularization * np.eye(len(train_x)), residual
            )
            error = trend + cross @ alpha - dev_target
            loss = float(np.mean(error**2))
            trials.append(
                dict(
                    length_scale=length,
                    regularization=regularization,
                    development_standardized_mse=loss,
                )
            )
            if loss < best_loss:
                best_loss = loss
                best = replace(
                    base,
                    kind=kind,
                    length_scale=length,
                    regularization=regularization,
                    arrays={**a, "alpha": alpha},
                )
    return best, trials


def run_direct(source, output, seeds, smoke):
    output.mkdir()
    reports = []
    horizons, histories = (
        ((1,), ("20ms",)) if smoke else ((1, 10, 25), ("20ms", "100ms"))
    )
    for seed in seeds:
        for horizon in horizons:
            for history in histories:
                name = f"h{horizon}-{history}-n384-seed{seed}"
                folder = output / name
                folder.mkdir()
                origin = source / name
                with np.load(origin / "samples.npz") as data:
                    batches = {
                        role: {
                            k.removeprefix(role + "_"): data[k]
                            for k in data.files
                            if k.startswith(role + "_")
                        }
                        for role in ROLES
                    }
                train, dev = batches["train"], batches["development"]
                train_y = train["next_states"] - train["states"]
                dev_y = dev["next_states"] - dev["states"]
                report = dict(
                    seed=seed,
                    horizon_steps=horizon,
                    history=history,
                    source=str(origin),
                    hashes={
                        key: hashlib.sha256((origin / key).read_bytes()).hexdigest()
                        for key in ("samples.npz", "predictions.npz", "report.json")
                    },
                    methods={},
                )
                predictions = {}
                for kind in ("linear", "rbf", "additive", "pairwise"):
                    start = time.monotonic()
                    model, trials = select_regressor(
                        train["features"], train_y, dev["features"], dev_y, kind
                    )
                    model.save(folder / f"{kind}.npz")
                    method = dict(
                        trials=trials,
                        length_scale=model.length_scale,
                        regularization=model.regularization,
                        scores={},
                        fit_selection_seconds=time.monotonic() - start,
                        fingerprint=model.fingerprint(),
                    )
                    for role in ROLES[1:]:
                        batch = batches[role]
                        mean = batch["states"] + numpy_predict(model, batch["features"])
                        predictions[f"{kind}_{role}"] = mean
                        method["scores"][role] = score(mean, batch)
                    report["methods"][kind] = method
                old_report = json.loads((origin / "report.json").read_text())
                with np.load(origin / "predictions.npz") as old:
                    for kind in ("shared", "ridge", "hold"):
                        report["methods"][kind] = old_report["methods"][kind]
                        for role in ROLES[1:]:
                            predictions[f"{kind}_{role}"] = old[f"{kind}_{role}_mean"]
                    np.testing.assert_allclose(
                        predictions["linear_replication"],
                        predictions["ridge_replication"],
                        atol=1e-12,
                    )
                np.savez_compressed(folder / "predictions.npz", **predictions)
                write_json(folder / "report.json", report)
                reports.append(report)
                write_json(output / "summary.json", reports)
                print(
                    json.dumps(
                        dict(
                            track="direct",
                            case=name,
                            rates={
                                k: round(
                                    m["scores"]["replication"]["all"][
                                        "rate_rmse_rad_s"
                                    ],
                                    5,
                                )
                                for k, m in report["methods"].items()
                            },
                        )
                    ),
                    flush=True,
                )


def sequence_batches(records, source, seed, smoke):
    """Same origins as h25/100ms diagnosis; dense labels, full input sequence."""
    path = source / f"h25-100ms-n384-seed{seed}" / "samples.npz"
    result, details = {}, {}
    with np.load(path) as archived:
        for role in ROLES:
            groups, anchors = archived[f"{role}_groups"], archived[f"{role}_anchors"]
            collected = {
                k: []
                for k in (
                    "past_states",
                    "past_inputs",
                    "future_inputs",
                    "future_states",
                )
            }
            usage = []
            for record in records:
                if record["role"] != role:
                    continue
                origins = anchors[groups == record["name"]]
                if smoke:
                    origins = origins[:8]
                states = np.column_stack((record["states"], record["context"]))
                window = sequence_windows(
                    states,
                    record["controls"],
                    origins,
                    history_steps=10,
                    horizon_steps=25,
                    dt_s=0.01,
                )
                for k in collected:
                    collected[k].append(getattr(window, k))
                state_rows = np.unique(origins[:, None] + np.arange(-10, 26))
                input_rows = np.unique(origins[:, None] + np.arange(-10, 25))
                usage.append(
                    dict(
                        recording=record["name"],
                        origins=origins.tolist(),
                        windows=len(origins),
                        unique_state_rows=len(state_rows),
                        unique_input_rows=len(input_rows),
                    )
                )
            result[role] = SequenceBatch(
                **{k: np.concatenate(v) for k, v in collected.items()}, dt_s=0.01
            )
            details[role] = usage
    return result, details


def run_sequence(corpus, source, output, seeds, smoke, followup=False):
    output.mkdir()
    records = load_records(corpus)
    write_json(output / "data-audit.json", [r["inspection"] for r in records])
    reports = []
    arms = (
        ("linear", "teacher"),
        ("delay", "teacher"),
        ("mlp", "teacher"),
        ("mlp", "rollout"),
        ("latent", "teacher"),
        ("latent", "rollout"),
    )
    if followup:
        arms = (
            ("delay", "rollout"),
            ("delay_mlp", "teacher"),
            ("delay_mlp", "rollout"),
        )
    for seed in seeds:
        batches, usage = sequence_batches(records, source, seed, smoke)
        seed_folder = output / f"seed{seed}"
        seed_folder.mkdir()
        write_json(seed_folder / "usage.json", usage)
        np.savez_compressed(
            seed_folder / "samples.npz",
            **{
                f"{role}_{k}": getattr(batch, k)
                for role, batch in batches.items()
                for k in (
                    "past_states",
                    "past_inputs",
                    "future_inputs",
                    "future_states",
                )
            },
        )
        for kind, objective in arms:
            name = f"{kind}-{objective}"
            start = time.monotonic()
            model, report = fit_sequence_model(
                batches["train"],
                batches["development"],
                kind=kind,
                objective=objective,
                seed=seed,
                steps=0
                if kind in ("linear", "delay") and objective == "teacher"
                else (5 if smoke else 1000),
                check_every=5 if smoke else 100,
            )
            report.update(
                seed=seed,
                name=name,
                scores={},
                fingerprint=model.fingerprint(),
                fit_selection_seconds=time.monotonic() - start,
            )
            model.save(seed_folder / f"{name}.npz")
            predicted = {}
            rollout = jax.jit(model.rollout)
            for role in ROLES[1:]:
                batch = batches[role]
                mean = np.asarray(
                    rollout(batch.past_states, batch.past_inputs, batch.future_inputs)
                )
                predicted[role] = mean
                groups = np.concatenate(
                    [
                        np.full(item["windows"], item["recording"])
                        for item in usage[role]
                    ]
                )
                report["scores"][role] = {}
                for horizon in (1, 10, 25):
                    data = dict(
                        next_states=batch.future_states[:, horizon - 1, :6],
                        groups=groups,
                    )
                    metrics = score(mean[:, horizon - 1, :6], data)
                    rotation = mean[:, horizon - 1, 6:].reshape(-1, 3, 3)
                    metrics["rotation"] = dict(
                        entry_rmse=float(
                            np.sqrt(
                                np.mean(
                                    (
                                        mean[:, horizon - 1, 6:]
                                        - batch.future_states[:, horizon - 1, 6:]
                                    )
                                    ** 2
                                )
                            )
                        ),
                        orthogonality_rms=float(
                            np.sqrt(
                                np.mean(
                                    np.sum(
                                        (
                                            rotation.swapaxes(-1, -2) @ rotation
                                            - np.eye(3)
                                        )
                                        ** 2,
                                        axis=(1, 2),
                                    )
                                )
                            )
                        ),
                        determinant_min=float(np.linalg.det(rotation).min()),
                    )
                    report["scores"][role][str(horizon)] = metrics
            np.savez_compressed(seed_folder / f"{name}-predictions.npz", **predicted)
            write_json(seed_folder / f"{name}-report.json", report)
            reports.append(report)
            write_json(output / "summary.json", reports)
            print(
                json.dumps(
                    dict(
                        track="sequence",
                        seed=seed,
                        method=name,
                        step=report["selected_step"],
                        seconds=round(report["fit_selection_seconds"], 1),
                        rates={
                            h: round(s["all"]["rate_rmse_rad_s"], 5)
                            for h, s in report["scores"]["replication"].items()
                        },
                    )
                ),
                flush=True,
            )
            del model, rollout
            jax.clear_caches()


def run_synthetic(output, seeds):
    """Known compositional surface; withheld combinations and unseen value ranges."""
    output.mkdir()

    def target(x):
        deadzone = np.sign(x[:, 2]) * np.maximum(np.abs(x[:, 2]) - 0.4, 0)
        return np.column_stack(
            (
                0.8 * x[:, 0]
                + 1.2 * np.sin(1.5 * x[:, 1])
                + deadzone
                + 0.5 * x[:, 3] * x[:, 4],
                -0.4 * np.cos(2 * x[:, 1]) + 0.6 * x[:, 2] * x[:, 5],
                0.4 * x[:, 0] * x[:, 1] + 0.5 * np.tanh(2 * x[:, 6]) + 0.3 * x[:, 7],
            )
        )

    reports = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        pool = rng.uniform(-2, 2, (6000, 8))
        omitted = (pool[:, 0] > 0) & (pool[:, 1] > 0)
        known = pool[~omitted]
        train, dev, familiar = known[:384], known[384:896], known[896:1408]
        combo = pool[omitted][:512]
        outside = rng.uniform(-2, 2, (512, 8))
        outside[:, 1] = rng.uniform(2.5, 4, 512)
        observed = target(train) + rng.normal(0, 0.03, (len(train), 3))
        case = output / f"seed{seed}"
        case.mkdir()
        samples = dict(
            train_features=train,
            train_targets=observed,
            development_features=dev,
            development_targets=target(dev),
        )
        for role, x in (
            ("familiar", familiar),
            ("combination", combo),
            ("outside", outside),
        ):
            samples[f"{role}_features"], samples[f"{role}_targets"] = x, target(x)
        np.savez_compressed(case / "samples.npz", **samples)
        report = dict(seed=seed, methods={})
        predictions = {}
        for kind in ("linear", "rbf", "additive", "pairwise"):
            model, trials = select_regressor(train, observed, dev, target(dev), kind)
            model.save(case / f"{kind}.npz")
            metrics = {}
            for role in ("familiar", "combination", "outside"):
                mean = numpy_predict(model, samples[f"{role}_features"])
                predictions[f"{kind}_{role}"] = mean
                metrics[role] = float(
                    np.sqrt(np.mean((mean - samples[f"{role}_targets"]) ** 2))
                )
            report["methods"][kind] = dict(
                rmse=metrics,
                trials=trials,
                length_scale=model.length_scale,
                regularization=model.regularization,
            )
        np.savez_compressed(case / "predictions.npz", **predictions)
        write_json(case / "report.json", report)
        reports.append(report)
        print(
            json.dumps(
                dict(
                    track="synthetic",
                    seed=seed,
                    metrics={k: v["rmse"] for k, v in report["methods"].items()},
                )
            ),
            flush=True,
        )
    write_json(output / "summary.json", reports)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("../artifacts/transition-diagnosis/diagnosis-01"),
    )
    parser.add_argument(
        "--corpus", type=Path, default=Path("../artifacts/real-transition/corpus")
    )
    parser.add_argument(
        "--track",
        choices=("all", "direct", "sequence", "synthetic", "sequence-followup"),
        default="all",
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not jax.config.jax_enable_x64:
        raise ValueError("recorded structure experiments require JAX_ENABLE_X64=1")
    args.output.mkdir(parents=True, exist_ok=False)
    seeds = (60,) if args.smoke else (60, 61, 62)
    root = Path(__file__).resolve().parents[1]
    paths = [
        Path(__file__),
        root / "scripts/experiment_transition_diagnosis.py",
        root / "scripts/experiment_real_transition.py",
        root / "tests/test_model_structures.py",
        *sorted((root / "src/glassbox/experimental").glob("*.py")),
    ]
    with zipfile.ZipFile(
        args.output / "executed-sources.zip", "x", zipfile.ZIP_DEFLATED
    ) as archive:
        for p in paths:
            archive.write(p, p.relative_to(root))
    write_json(
        args.output / "sources.json",
        {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths
        },
    )
    write_json(
        args.output / "plan.json",
        dict(
            seeds=seeds,
            track=args.track,
            smoke=args.smoke,
            adaptive_followup=args.track == "sequence-followup",
            source=str(args.source.resolve()),
            direct=dict(
                horizons=[1, 10, 25],
                histories=["20ms", "100ms"],
                windows=384,
                lengths=LENGTHS,
                regularizations=REGULARIZATIONS,
                selection="minimum six-output training-increment-standardized development MSE per family",
            ),
            sequence=dict(
                history_steps=10,
                horizon_steps=25,
                windows=384,
                state_width=15,
                input_width=4,
                width=32,
                memory=8,
                steps=1000,
                batch_size=64,
                learning_rate=0.002,
                gradient_clip=5.0,
                ridge=1.0,
                selection="minimum 15-output training-state-standardized development rollout MSE, every 100 updates including initialization",
            ),
            caveats=[
                "Development selects; replication does not. Both groups used in prior work.",
                "Direct matches old features and endpoint labels; sequence uses dense targets, past orientation, full future input sequence.",
                "All real measurements were processed offline upstream; future measured motor speeds condition forecasts.",
                "Rotation matrix entries are Euclidean outputs; no SO(3) projection or physical bounds.",
                "Synthetic surface intentionally contains additive and pairwise structure; a mechanism demonstration, not an unbiased architecture ranking.",
                "No learned uncertainty or support claims; no streaming, control, or cross-platform result.",
            ],
        ),
    )
    write_json(
        args.output / "environment.json",
        dict(
            python=sys.version,
            numpy=np.__version__,
            jax=jax.__version__,
            platform=platform.platform(),
            x64=True,
        ),
    )
    if args.track in ("all", "direct"):
        run_direct(args.source, args.output / "direct", seeds, args.smoke)
    if args.track in ("all", "sequence", "sequence-followup"):
        run_sequence(
            args.corpus,
            args.source,
            args.output / "sequence",
            seeds,
            args.smoke,
            followup=args.track == "sequence-followup",
        )
    if args.track in ("all", "synthetic"):
        run_synthetic(args.output / "synthetic", seeds)


if __name__ == "__main__":
    main()
