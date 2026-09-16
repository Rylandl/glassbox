"""Adaptive neural refinement after development-only affine model selection."""

import argparse
import json
import zipfile
from pathlib import Path

import jax
import numpy as np
from experiment_sequence_transfer import ARRAYS, GROUPS, metrics
from report_model_structures import recurrence
from sequence_transfer_data import write_json

from glassbox.experimental.sequence_model import (
    SequenceBatch,
    SequenceModel,
    fit_sequence_model,
)
from glassbox.experimental.sequence_objective import SequenceGuard, relative_error_scale


def run(source, selection, output):
    if not jax.config.jax_enable_x64:
        raise ValueError("requires float64")
    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output / "plan.json",
        dict(
            adaptive=True,
            source=str(source.resolve()),
            selection=str(selection.resolve()),
            arms=["standard", "guarded"],
            steps=1000,
            check_every=100,
            initialization="family and ridge chosen by prior development-only affine selection",
            guarded="training reference-error scaling and 5% development regression limit",
            other_hyperparameters="unchanged from initial transfer plan",
        ),
    )
    root = Path(__file__).resolve().parents[1]
    with zipfile.ZipFile(
        output / "executed-sources.zip", "x", zipfile.ZIP_DEFLATED
    ) as archive:
        for p in [
            Path(__file__),
            root / "scripts/experiment_sequence_transfer.py",
            root / "scripts/report_model_structures.py",
            *sorted((root / "src/glassbox/experimental").glob("*.py")),
        ]:
            archive.write(p, p.relative_to(root))
    reports = []
    for dataset in ("nano", "x8", "arp"):
        (output / dataset).mkdir()
        for seed in (60, 61, 62):
            chosen_folder = selection / dataset / f"seed{seed}"
            chosen = json.loads((chosen_folder / "report.json").read_text())
            baseline = SequenceModel.load(
                chosen_folder / f"{chosen['selected']['name']}.npz"
            )
            with np.load(source / dataset / f"seed{seed}" / "samples.npz") as data:
                batches = {
                    role: SequenceBatch(
                        **{k: data[f"{role}_{k}"] for k in ARRAYS}, dt_s=baseline.dt_s
                    )
                    for role in ("train", "development", "evaluation")
                }
            train = batches["train"]
            reference = recurrence(
                baseline, train.past_states, train.past_inputs, train.future_inputs
            )
            scale = relative_error_scale(
                reference, train.future_states, baseline.norms["state_scale"]
            )
            folder = output / dataset / f"seed{seed}"
            folder.mkdir()
            np.savez_compressed(folder / "objective.npz", error_scale=scale)
            for arm in ("standard", "guarded"):
                model, report = fit_sequence_model(
                    train,
                    batches["development"],
                    kind="delay_mlp" if baseline.kind == "delay" else "mlp",
                    ridge=chosen["selected"]["ridge"],
                    seed=seed,
                    error_scale=scale if arm == "guarded" else None,
                    selection_guard=SequenceGuard(tuple(chosen["horizons"]), GROUPS)
                    if arm == "guarded"
                    else None,
                )
                report.update(
                    dataset=dataset,
                    arm=arm,
                    seed=seed,
                    dt_s=baseline.dt_s,
                    horizons=chosen["horizons"],
                    initialization=chosen["selected"],
                    fingerprint=model.fingerprint(),
                    scores={},
                )
                means = {}
                for role in ("development", "evaluation"):
                    b = batches[role]
                    mean = recurrence(
                        model, b.past_states, b.past_inputs, b.future_inputs
                    )
                    means[role] = mean
                    report["scores"][role] = metrics(
                        mean, b.future_states, chosen["horizons"]
                    )
                model.save(folder / f"{arm}.npz")
                np.savez_compressed(folder / f"{arm}-predictions.npz", **means)
                write_json(folder / f"{arm}-report.json", report)
                reports.append(report)
                write_json(output / "summary.json", reports)
                s = report["scores"]["evaluation"]["all"][str(chosen["horizons"][-1])]
                print(
                    json.dumps(
                        dict(
                            dataset=dataset,
                            seed=seed,
                            arm=arm,
                            step=report["selected_step"],
                            velocity=s["velocity_rmse"],
                            rate=s["rate_rmse"],
                        )
                    ),
                    flush=True,
                )
                jax.clear_caches()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.selection, args.output)
