"""Freeze one sequence-learning recipe across Nano, X8 and causal ARP streams."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
import zipfile
from pathlib import Path

import jax
import numpy as np
from experiment_model_structures import sequence_batches
from experiment_transition_diagnosis import load_records
from sequence_transfer_data import load_prepared, write_json

from glassbox.experimental.sequence_model import (
    SequenceBatch,
    fit_sequence_model,
    initialize_sequence_model,
    sequence_windows,
)
from glassbox.experimental.sequence_objective import SequenceGuard, relative_error_scale

GROUPS = ((0, 1, 2), (3, 4, 5), tuple(range(6, 15)))
ROLES = ("train", "development", "evaluation")
ARRAYS = ("past_states", "past_inputs", "future_inputs", "future_states")


def assemble(records, seed, *, smoke=False):
    dt = records[0]["dt_s"]
    history, horizon = int(np.rint(0.1 / dt)), int(np.rint(0.25 / dt))
    batches, usage = {}, {}
    for role in ROLES:
        selected = [r for r in records if r["role"] == role]
        if not selected:
            raise ValueError(f"missing {role} recordings")
        parts = {k: [] for k in ARRAYS}
        details = []
        for i, r in enumerate(selected):
            if r["dt_s"] != dt:
                raise ValueError("inconsistent sample interval")
            pool = np.arange(history, len(r["states"]) - horizon)
            if role == "train":
                count = (48 if smoke else 384) // len(selected) + (
                    i < (48 if smoke else 384) % len(selected)
                )
                identity = int(hashlib.sha256(r["name"].encode()).hexdigest()[:8], 16)
                rng = np.random.default_rng(np.random.SeedSequence([seed, identity]))
                anchors = rng.permutation(pool)[:count]
                if len(anchors) != count:
                    raise ValueError("insufficient complete training windows")
            else:
                anchors = pool[:: int(np.rint(0.1 / dt))]
                if smoke:
                    anchors = anchors[:8]
            window = sequence_windows(
                r["states"],
                r["inputs"],
                anchors,
                history_steps=history,
                horizon_steps=horizon,
                dt_s=dt,
            )
            for key in ARRAYS:
                parts[key].append(getattr(window, key))
            details.append(
                dict(
                    recording=r["name"],
                    windows=len(anchors),
                    origins=anchors.tolist(),
                    unique_state_rows=len(
                        np.unique(anchors[:, None] + np.arange(-history, horizon + 1))
                    ),
                    unique_input_rows=len(
                        np.unique(anchors[:, None] + np.arange(-history, horizon))
                    ),
                    source_sha256=r["metadata"]["sha256"],
                )
            )
        batches[role] = SequenceBatch(
            **{k: np.concatenate(v) for k, v in parts.items()}, dt_s=dt
        )
        usage[role] = details
    return batches, usage


def metrics(mean, target, horizons, groups=None):
    result = {}
    masks = {"all": np.ones(len(mean), dtype=bool)}
    if groups is not None:
        masks.update({str(g): groups == g for g in np.unique(groups)})
    for label, mask in masks.items():
        result[label] = {}
        for h in horizons:
            e = mean[mask, h - 1] - target[mask, h - 1]
            r = mean[mask, h - 1, 6:].reshape(-1, 3, 3)
            result[label][str(h)] = dict(
                count=int(mask.sum()),
                velocity_rmse=float(np.sqrt(np.mean(np.sum(e[:, :3] ** 2, 1)))),
                rate_rmse=float(np.sqrt(np.mean(np.sum(e[:, 3:6] ** 2, 1)))),
                rotation_rmse=float(np.sqrt(np.mean(np.sum(e[:, 6:] ** 2, 1)))),
                per_channel_rmse=np.sqrt(np.mean(e**2, 0)).tolist(),
                rotation_orthogonality_rms=float(
                    np.sqrt(
                        np.mean(np.sum((r.swapaxes(1, 2) @ r - np.eye(3)) ** 2, (1, 2)))
                    )
                ),
            )
    return result


def run(args):
    if not jax.config.jax_enable_x64:
        raise ValueError("recorded transfer experiments require JAX_ENABLE_X64=1")
    args.output.mkdir(parents=True, exist_ok=False)
    seeds = (60,) if args.smoke else (60, 61, 62)
    recipe = dict(
        datasets=args.datasets,
        seeds=seeds,
        smoke=args.smoke,
        widths=dict(nonlinear=32),
        history_s=0.1,
        horizon_s=0.25,
        timing="nearest integer samples (ties-to-even); ARP has a 240 ms final horizon",
        training_windows=384,
        updates=1000,
        batch_size=64,
        learning_rate=0.002,
        ridge=1.0,
        checkpoint_interval=100,
        gradient_clip=5.0,
        balanced_scale="training linear-history recursive RMSE per horizon/channel; floor 0.001 times training channel standard deviation",
        selection_guard=dict(
            maximum_rmse_ratio=1.05,
            groups=GROUPS,
            horizons="one step, 100 ms, final horizon",
        ),
        arms=["linear_history", "standard", "balanced", "guarded"],
        transfer="same training recipe; fit new parameters on each dataset; no transferred weights",
        evaluation="whole recordings excluded from training and checkpoint selection",
        limits=[
            "Nano and X8 retain upstream offline preprocessing.",
            "ARP holds already-published estimator observations; it is not independent physical truth.",
            "Future logged inputs condition every forecast; their availability at launch is not established.",
            "Guard checks development only and is not an unseen-data guarantee.",
            "Rotation entries remain unconstrained Euclidean outputs.",
            "X8 segments share a campaign; ARP has one evaluation recording; seeds reuse evaluation data.",
        ],
    )
    write_json(args.output / "plan.json", recipe)
    root = Path(__file__).resolve().parents[1]
    sources = [
        Path(__file__),
        root / "scripts/sequence_transfer_data.py",
        root / "scripts/experiment_model_structures.py",
        root / "scripts/experiment_transition_diagnosis.py",
        root / "scripts/experiment_real_transition.py",
        root / "tests/test_sequence_transfer.py",
        root / "tests/test_model_structures.py",
        *sorted((root / "src/glassbox/experimental").glob("*.py")),
        root / "src/glassbox/io/x8_reference.py",
        root / "src/glassbox/io/arp_reference.py",
        root / "src/glassbox/core/geometry.py",
    ]
    with zipfile.ZipFile(
        args.output / "executed-sources.zip", "x", zipfile.ZIP_DEFLATED
    ) as archive:
        for path in sources:
            archive.write(path, path.relative_to(root))
    write_json(
        args.output / "sources.json",
        {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources
        },
    )
    write_json(
        args.output / "environment.json",
        dict(
            python=sys.version,
            jax=jax.__version__,
            numpy=np.__version__,
            platform=platform.platform(),
        ),
    )
    reports = []
    for dataset in args.datasets:
        records = (
            load_records(args.nano)
            if dataset == "nano"
            else load_prepared(args.prepared, dataset)
        )
        folder = args.output / dataset
        folder.mkdir()
        for seed in seeds:
            if dataset == "nano":
                original, detail = sequence_batches(
                    records, args.previous, seed, args.smoke
                )
                batches = {
                    ("evaluation" if role == "replication" else role): b
                    for role, b in original.items()
                }
                usage = {
                    ("evaluation" if role == "replication" else role): u
                    for role, u in detail.items()
                }
            else:
                batches, usage = assemble(records, seed, smoke=args.smoke)
            case = folder / f"seed{seed}"
            case.mkdir()
            write_json(case / "usage.json", usage)
            np.savez_compressed(
                case / "samples.npz",
                **{
                    f"{role}_{k}": getattr(b, k)
                    for role, b in batches.items()
                    for k in ARRAYS
                },
            )
            train, dev = batches["train"], batches["development"]
            horizon = train.future_states.shape[1]
            horizons = (1, int(np.rint(0.1 / train.dt_s)), horizon)
            baseline = initialize_sequence_model(train, kind="delay", seed=seed)
            baseline_train = np.asarray(
                jax.jit(baseline.rollout)(
                    train.past_states, train.past_inputs, train.future_inputs
                )
            )
            scale = relative_error_scale(
                baseline_train, train.future_states, baseline.norms["state_scale"]
            )
            np.savez_compressed(
                case / "objective.npz", error_scale=scale, baseline_train=baseline_train
            )
            for arm in recipe["arms"]:
                start = time.monotonic()
                model, report = fit_sequence_model(
                    train,
                    dev,
                    kind="delay" if arm == "linear_history" else "delay_mlp",
                    steps=0 if arm == "linear_history" else (5 if args.smoke else 1000),
                    check_every=5 if args.smoke else 100,
                    objective="rollout",
                    seed=seed,
                    error_scale=scale if arm in ("balanced", "guarded") else None,
                    selection_guard=SequenceGuard(horizons, GROUPS)
                    if arm == "guarded"
                    else None,
                )
                report.update(
                    dataset=dataset,
                    arm=arm,
                    dt_s=train.dt_s,
                    horizons=horizons,
                    elapsed_fit_s=time.monotonic() - start,
                    fingerprint=model.fingerprint(),
                    scores={},
                )
                model.save(case / f"{arm}.npz")
                means = {}
                predictor = jax.jit(model.rollout)
                for role in ("development", "evaluation"):
                    b = batches[role]
                    mean = np.asarray(
                        predictor(b.past_states, b.past_inputs, b.future_inputs)
                    )
                    means[role] = mean
                    groups = np.concatenate(
                        [np.full(u["windows"], u["recording"]) for u in usage[role]]
                    )
                    report["scores"][role] = metrics(
                        mean, b.future_states, horizons, groups
                    )
                np.savez_compressed(case / f"{arm}-predictions.npz", **means)
                write_json(case / f"{arm}-report.json", report)
                reports.append(report)
                write_json(args.output / "summary.json", reports)
                final = report["scores"]["evaluation"]["all"][str(horizon)]
                print(
                    json.dumps(
                        dict(
                            dataset=dataset,
                            seed=seed,
                            arm=arm,
                            selected=report["selected_step"],
                            rate=round(final["rate_rmse"], 6),
                            velocity=round(final["velocity_rmse"], 6),
                            seconds=round(report["elapsed_fit_s"], 2),
                        )
                    ),
                    flush=True,
                )
                del model, predictor
                jax.clear_caches()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--prepared",
        type=Path,
        default=Path("../artifacts/sequence-transfer/prepared-02"),
    )
    parser.add_argument(
        "--nano", type=Path, default=Path("../artifacts/real-transition/corpus")
    )
    parser.add_argument(
        "--previous",
        type=Path,
        default=Path("../artifacts/transition-diagnosis/diagnosis-01"),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("nano", "x8", "arp"),
        default=["nano", "x8", "arp"],
    )
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
