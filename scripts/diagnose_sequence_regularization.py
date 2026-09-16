"""Adaptive investigation of affine initialization, conditioning and rollout drift.

All candidates use the same saved windows. Only development trajectory loss
selects a candidate; this study was designed after the initial transfer results.
"""

import argparse
import json
import time
import zipfile
from pathlib import Path

import jax
import numpy as np
from experiment_sequence_transfer import ARRAYS, metrics
from report_model_structures import recurrence
from sequence_transfer_data import write_json

from glassbox.experimental.sequence_model import (
    SequenceBatch,
    initialize_sequence_model,
)


def history_operator(model):
    """Linearized homogeneous state/history update in normalized observation units.

    Commands are fixed. This is a diagnostic of the fitted linear recurrence,
    not a physical stability constraint. History is oldest-to-newest.
    """
    d, p = len(model.norms["state_mean"]), model.history_steps
    u = len(model.norms["input_mean"])
    b = (
        model.params["linear"]
        / model.norms["feature_scale"][:, None]
        * model.norms["delta_scale"][None, :]
    )
    a = np.zeros(((p + 1) * d, (p + 1) * d))
    if model.kind == "delay":
        delayed = b[d + u : d + u + p * d].reshape(p, d, d)
        a[:d, :d] = np.eye(d) + (b[:d] - delayed.sum(0)).T
        for i in range(p):
            a[:d, d + i * d : d + (i + 1) * d] = delayed[i].T
    else:
        a[:d, :d] = np.eye(d) + b[:d].T
    a[d : p * d, 2 * d :] = np.eye((p - 1) * d)
    a[p * d :, :d] = np.eye(d)
    return a


def teacher_predictions(model, batch):
    complete = np.concatenate((batch.past_states, batch.future_states), 1)
    inputs = np.concatenate((batch.past_inputs, batch.future_inputs), 1)
    p = model.history_steps
    predict = jax.jit(model.rollout)
    return np.concatenate(
        [
            np.asarray(
                predict(
                    complete[:, t : t + p + 1],
                    inputs[:, t : t + p],
                    inputs[:, t + p : t + p + 1],
                )
            )
            for t in range(batch.future_states.shape[1])
        ],
        axis=1,
    )


def run(source, output):
    if not jax.config.jax_enable_x64:
        raise ValueError("requires float64")
    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output / "plan.json",
        dict(
            adaptive=True,
            based_on=str(source.resolve()),
            families=["linear", "delay"],
            ridge_penalties=[1, 10, 100, 1000, 10000],
            selection="minimum development recursive MSE in train-state-standardized units",
            hold="reported separately; no candidate parameter fitting or selection uses evaluation targets",
            diagnostics=[
                "one-step refreshed observations versus open-loop recursion",
                "homogeneous state/history operator eigenvalues and finite-horizon norm",
            ],
            interpretation="Regularization can remove unsupported fit directions; no global stability constraint imposed.",
        ),
    )
    root = Path(__file__).resolve().parents[1]
    with zipfile.ZipFile(
        output / "executed-sources.zip", "x", zipfile.ZIP_DEFLATED
    ) as archive:
        for p in [
            Path(__file__),
            root / "scripts/report_model_structures.py",
            root / "scripts/experiment_sequence_transfer.py",
            root / "scripts/sequence_transfer_data.py",
            *sorted((root / "src/glassbox/experimental").glob("*.py")),
        ]:
            archive.write(p, p.relative_to(root))
    reports = []
    for dataset in ("nano", "x8", "arp"):
        (output / dataset).mkdir()
        for seed in (60, 61, 62):
            start = time.monotonic()
            origin = source / dataset / f"seed{seed}"
            folder = output / dataset / f"seed{seed}"
            folder.mkdir()
            with np.load(origin / "samples.npz") as arrays:
                dt = json.loads((origin / "linear_history-report.json").read_text())[
                    "dt_s"
                ]
                batches = {
                    role: SequenceBatch(
                        **{k: arrays[f"{role}_{k}"] for k in ARRAYS}, dt_s=dt
                    )
                    for role in ("train", "development", "evaluation")
                }
            train, dev = batches["train"], batches["development"]
            h = train.future_states.shape[1]
            horizons = (1, int(np.rint(0.1 / dt)), h)
            candidates = []
            best, best_loss = None, np.inf
            for kind in ("linear", "delay"):
                for ridge in (1, 10, 100, 1000, 10000):
                    model = initialize_sequence_model(
                        train, kind=kind, ridge=ridge, seed=seed
                    )
                    mean = recurrence(
                        model, dev.past_states, dev.past_inputs, dev.future_inputs
                    )
                    loss = float(
                        np.mean(
                            ((mean - dev.future_states) / model.norms["state_scale"])
                            ** 2
                        )
                    )
                    name = f"{kind}-ridge{ridge}"
                    operator = history_operator(model)
                    item = dict(
                        name=name,
                        kind=kind,
                        ridge=ridge,
                        development_loss=loss,
                        spectral_radius=float(
                            np.max(np.abs(np.linalg.eigvals(operator)))
                        ),
                        finite_horizon_operator_norm=float(
                            np.linalg.norm(np.linalg.matrix_power(operator, h), 2)
                        ),
                    )
                    model.save(folder / f"{name}.npz")
                    candidates.append(item)
                    if loss < best_loss:
                        best, best_loss = (model, item), loss
            selected, item = best
            selection = dict(
                dataset=dataset,
                seed=seed,
                dt_s=dt,
                horizons=horizons,
                selected=item,
                candidates=candidates,
                scores={},
                fingerprint=selected.fingerprint(),
            )
            saved = {}
            for role in ("development", "evaluation"):
                batch = batches[role]
                mean = recurrence(
                    selected, batch.past_states, batch.past_inputs, batch.future_inputs
                )
                teacher = teacher_predictions(selected, batch)
                hold = np.repeat(batch.past_states[:, -1:], h, axis=1)
                saved.update(
                    {
                        f"{role}_selected": mean,
                        f"{role}_teacher": teacher,
                        f"{role}_hold": hold,
                    }
                )
                selection["scores"][role] = {
                    k: metrics(v, batch.future_states, horizons)
                    for k, v in (
                        ("selected", mean),
                        ("teacher", teacher),
                        ("hold", hold),
                    )
                }
                # Refresh observed history for the original ridge-one baseline too.
                original = initialize_sequence_model(
                    train, kind="delay", ridge=1, seed=seed
                )
                original_teacher = teacher_predictions(original, batch)
                saved[f"{role}_original_teacher"] = original_teacher
                selection["scores"][role]["original_teacher"] = metrics(
                    original_teacher, batch.future_states, horizons
                )
            np.savez_compressed(folder / "predictions.npz", **saved)
            write_json(folder / "report.json", selection)
            reports.append(selection)
            write_json(output / "summary.json", reports)
            print(
                json.dumps(
                    dict(
                        dataset=dataset,
                        seed=seed,
                        selected=item["name"],
                        original_spectral_radius=candidates[5]["spectral_radius"],
                        original_amplification=candidates[5][
                            "finite_horizon_operator_norm"
                        ],
                        selected_amplification=item["finite_horizon_operator_norm"],
                        final=selection["scores"]["evaluation"]["selected"]["all"][
                            str(h)
                        ],
                        seconds=round(time.monotonic() - start, 2),
                    )
                ),
                flush=True,
            )
            jax.clear_caches()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.output)
