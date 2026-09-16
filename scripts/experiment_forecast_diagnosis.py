"""Matched direct/recursive prediction with whole-recording and support diagnostics."""

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
from experiment_sequence_transfer import ARRAYS, GROUPS, assemble
from sequence_transfer_data import load_prepared, write_json

from glassbox.experimental.direct_forecast import fit_direct_forecast, prefix_features
from glassbox.experimental.sequence_model import (
    SequenceBatch,
    _rollout,
    initialize_sequence_model,
)

RIDGES = (0.0001, 0.001, 0.01, 0.1, 1.0, 10.0)
ROLES = ("train", "development", "evaluation")


def loss(prediction, target, scale):
    return float(np.mean(((prediction - target) / scale) ** 2))


def vector_errors(prediction, target):
    return np.stack(
        [
            np.sum((prediction[:, :, list(g)] - target[:, :, list(g)]) ** 2, -1)
            for g in GROUPS
        ],
        -1,
    )


def scores(prediction, target):
    return np.sqrt(vector_errors(prediction, target).mean(0)).tolist()


def references(batch, horizons):
    current = batch.past_states[:, -1]
    hold = np.repeat(current[:, None], len(horizons), 1)
    t = np.arange(batch.past_states.shape[1], dtype=float)
    t -= t.mean()
    slope = np.einsum("t,ntd->nd", t, batch.past_states) / (t @ t)
    return dict(
        hold=hold, trend=hold + np.array(horizons)[None, :, None] * slope[:, None]
    )


def select_candidates(predictions, target, scale):
    """Only the explicitly supplied development target enters selection."""
    names = list(predictions)
    losses = {name: loss(p, target, scale) for name, p in predictions.items()}
    cell_losses = np.array(
        [vector_errors(p, target).mean(0) for p in predictions.values()]
    )
    return dict(
        global_name=min(names, key=lambda n: losses[n]),
        cell_names=np.array(names)[cell_losses.argmin(0)].tolist(),
        losses=losses,
    )


def selected_cells(predictions, names):
    first = next(iter(predictions.values()))
    result = np.empty_like(first)
    for i, row in enumerate(names):
        for group, name in zip(GROUPS, row):
            result[:, i, list(group)] = predictions[name][:, i, list(group)]
    return result


def distance_matrix(left, right):
    squared = (
        (left * left).sum(1)[:, None]
        + (right * right).sum(1)[None, :]
        - 2 * left @ right.T
    )
    return np.sqrt(np.maximum(squared / left.shape[1], 0))


def support_diagnostics(batches, usage):
    features = {
        r: prefix_features(
            b.past_states, b.past_inputs, b.future_inputs, 1, use_history=True
        )
        for r, b in batches.items()
    }
    train = features["train"]
    mean, scale = train.mean(0), train.std(0)
    scale = np.where(scale > 1e-8, scale, 1)
    z = (train - mean) / scale
    groups = np.concatenate(
        [np.repeat(u["recording"], u["windows"]) for u in usage["train"]]
    )
    distances = distance_matrix(z, z)
    distances[groups[:, None] == groups[None, :]] = np.inf
    cross_record = distances.min(1)
    if not np.isfinite(cross_record).all():
        raise ValueError("support reference requires two or more training recordings")
    threshold = np.quantile(cross_record, [0.5, 0.9])
    arrays = dict(
        feature_mean=mean,
        feature_scale=scale,
        training_features=z,
        training_cross_record_distance=cross_record,
        thresholds=threshold,
    )
    for role in ("development", "evaluation"):
        arrays[f"{role}_distance"] = distance_matrix(
            (features[role] - mean) / scale, z
        ).min(1)
    return arrays


def origin_metadata(usage, records):
    by_name = {r["name"]: r for r in records}
    result = {}
    for role, items in usage.items():
        ids = np.concatenate([np.repeat(u["recording"], u["windows"]) for u in items])
        result[f"{role}_recordings"] = ids
        if records and "publication_s" in records[0]:
            for kind, field in (
                ("publication_age", "publication_s"),
                ("sample_age", "sample_s"),
            ):
                result[f"{role}_{kind}"] = np.concatenate(
                    [
                        by_name[u["recording"]]["absolute_grid_s"][u["origins"], None]
                        - by_name[u["recording"]][field][u["origins"]]
                        for u in items
                    ]
                )
    return result


def run(args):
    if not jax.config.jax_enable_x64:
        raise ValueError("recorded experiments require float64")
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "plan.json",
        dict(
            question="Does avoiding recursive prediction improve conditional forecasts, and do error/support/age associations repeat across held-out flights?",
            datasets=args.datasets,
            seeds=[60] if args.smoke else [60, 61, 62],
            families=[
                "recursive_current",
                "recursive_history",
                "direct_current",
                "direct_history",
            ],
            ridge_fractions=RIDGES,
            penalty="mean squared residual + ridge fraction times squared standardized feature coefficients",
            references=[
                "hold current observation",
                "least-squares trend over the observed history",
            ],
            horizons="one step, 100 ms, final 250 ms (ARP 240 ms)",
            direct="separate change-from-current heads; only command prefixes through each target; no future observations",
            arp_folds="each of four recordings evaluates once; preceding recording cyclically develops; other two train",
            budget="384 matched training origins per case; existing Nano/X8 windows; recursive fit uses every one-step transition inside them",
            selector="whole-model train-state-standardized development loss over the three horizons; separate diagnostic selector minimizes each development horizon/group error",
            support="nearest standardized history/current-command reference; training thresholds use neighbors from another recording only; 50th/90th percentiles",
            age="publication/sample age at prediction origin; no future observation ages in predictor or selector",
            limits=[
                "Adaptively motivated by prior inspected results; these are reused flight recordings, not pristine new holdouts.",
                "Overlapping windows and repeated seeds are not independent flights.",
                "All predictions condition on future logged inputs.",
                "Support/age error associations are descriptive, not causal attribution or calibrated uncertainty.",
                "Cell-wise selection may combine incompatible heads; it is an offline diagnostic, not a physical trajectory model.",
            ],
            smoke=args.smoke,
        ),
    )
    root = Path(__file__).resolve().parents[1]
    files = [
        Path(__file__),
        root / "scripts/experiment_sequence_transfer.py",
        root / "scripts/sequence_transfer_data.py",
        *sorted((root / "src/glassbox/experimental").glob("*.py")),
    ]
    with zipfile.ZipFile(
        args.output / "executed-sources.zip", "x", zipfile.ZIP_DEFLATED
    ) as archive:
        for p in files:
            archive.write(p, p.relative_to(root))
    write_json(
        args.output / "sources.json",
        {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files
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
    predict_recursive = jax.jit(_rollout, static_argnames=("kind",))
    reports = []
    for dataset in args.datasets:
        records = load_prepared(args.prepared, dataset) if dataset != "nano" else []
        folds = range(1) if args.smoke else range(4) if dataset == "arp" else range(1)
        for fold in folds:
            for seed in [60] if args.smoke else [60, 61, 62]:
                start = time.monotonic()
                folder = args.output / dataset / f"fold{fold}-seed{seed}"
                folder.mkdir(parents=True)
                if dataset == "arp":
                    roles = [
                        "evaluation"
                        if i == fold
                        else "development"
                        if i == (fold - 1) % 4
                        else "train"
                        for i in range(4)
                    ]
                    fold_records = [
                        {**r, "role": role} for r, role in zip(records, roles)
                    ]
                    batches, usage = assemble(fold_records, seed, smoke=args.smoke)
                else:
                    old = args.previous / dataset / f"seed{seed}"
                    usage = json.loads((old / "usage.json").read_text())
                    dt = json.loads((old / "linear_history-report.json").read_text())[
                        "dt_s"
                    ]
                    with np.load(old / "samples.npz") as archive:
                        batches = {
                            r: SequenceBatch(
                                **{k: archive[f"{r}_{k}"] for k in ARRAYS}, dt_s=dt
                            )
                            for r in ROLES
                        }
                train = batches["train"]
                horizons = (
                    1,
                    int(np.rint(0.1 / train.dt_s)),
                    train.future_states.shape[1],
                )
                indices = np.array(horizons) - 1
                targets = {r: b.future_states[:, indices] for r, b in batches.items()}
                norm = initialize_sequence_model(train, kind="linear")
                scale = norm.norms["state_scale"]
                np.savez_compressed(
                    folder / "samples.npz",
                    **{
                        f"{r}_{k}": getattr(b, k)
                        for r, b in batches.items()
                        for k in ARRAYS
                    },
                )
                write_json(folder / "usage.json", usage)
                evidence = support_diagnostics(batches, usage)
                evidence.update(origin_metadata(usage, records))
                np.savez_compressed(folder / "support.npz", **evidence)
                predictions = {
                    r: references(b, horizons)
                    for r, b in batches.items()
                    if r != "train"
                }
                candidates = []
                for family in (
                    "recursive_current",
                    "recursive_history",
                    "direct_current",
                    "direct_history",
                ):
                    for ridge in RIDGES:
                        name = f"{family}-r{ridge:g}"
                        history = family.endswith("history")
                        if family.startswith("direct"):
                            model = fit_direct_forecast(
                                train,
                                horizons,
                                use_history=history,
                                ridge_fraction=ridge,
                            )
                            for role in predictions:
                                b = batches[role]
                                predictions[role][name] = np.asarray(
                                    model.predict(
                                        b.past_states, b.past_inputs, b.future_inputs
                                    )
                                )
                        else:
                            model = initialize_sequence_model(
                                train,
                                kind="delay" if history else "linear",
                                ridge=ridge
                                * len(train.past_states)
                                * train.future_states.shape[1],
                            )
                            for role in predictions:
                                b = batches[role]
                                predictions[role][name] = np.asarray(
                                    predict_recursive(
                                        model.params,
                                        model.norms,
                                        model.kind,
                                        b.past_states,
                                        b.past_inputs,
                                        b.future_inputs,
                                    )
                                )[:, indices]
                        model.save(folder / f"{name}.npz")
                        candidates.append(
                            dict(
                                name=name,
                                family=family,
                                ridge_fraction=ridge,
                                fingerprint=model.fingerprint(),
                            )
                        )
                selection = select_candidates(
                    predictions["development"], targets["development"], scale
                )
                selected_families = {
                    family: min(
                        [c["name"] for c in candidates if c["family"] == family],
                        key=lambda n: selection["losses"][n],
                    )
                    for family in (
                        "recursive_current",
                        "recursive_history",
                        "direct_current",
                        "direct_history",
                    )
                }
                report = dict(
                    dataset=dataset,
                    fold=fold,
                    seed=seed,
                    dt_s=train.dt_s,
                    horizons=horizons,
                    state_scale=scale.tolist(),
                    candidates=candidates,
                    selection=selection,
                    selected_families=selected_families,
                    scores={},
                )
                for role, pred in predictions.items():
                    report["scores"][role] = {
                        name: dict(
                            loss=loss(p, targets[role], scale),
                            rmse=scores(p, targets[role]),
                        )
                        for name, p in pred.items()
                    }
                    pred["selected_global"] = pred[selection["global_name"]]
                    pred["selected_cells"] = selected_cells(
                        pred, selection["cell_names"]
                    )
                    report["scores"][role].update(
                        {
                            name: dict(
                                loss=loss(pred[name], targets[role], scale),
                                rmse=scores(pred[name], targets[role]),
                            )
                            for name in ("selected_global", "selected_cells")
                        }
                    )
                np.savez_compressed(
                    folder / "predictions.npz",
                    **{
                        f"{r}__{k}": v
                        for r, pred in predictions.items()
                        for k, v in pred.items()
                    },
                )
                write_json(folder / "report.json", report)
                reports.append(report)
                write_json(args.output / "summary.json", reports)
                print(
                    json.dumps(
                        dict(
                            dataset=dataset,
                            fold=fold,
                            seed=seed,
                            selected=selection["global_name"],
                            final={
                                n: report["scores"]["evaluation"][
                                    selected_families.get(n, n)
                                ]["rmse"][-1][:2]
                                for n in (
                                    "hold",
                                    "trend",
                                    "recursive_history",
                                    "direct_history",
                                    "selected_global",
                                    "selected_cells",
                                )
                            },
                            seconds=round(time.monotonic() - start, 2),
                        )
                    ),
                    flush=True,
                )
    return reports


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--previous",
        type=Path,
        default=Path("../artifacts/sequence-transfer/comparison-01"),
    )
    parser.add_argument(
        "--prepared",
        type=Path,
        default=Path("../artifacts/sequence-transfer/prepared-02"),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("nano", "x8", "arp"),
        default=["nano", "x8", "arp"],
    )
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
