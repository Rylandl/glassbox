"""One matched objective ablation on synthetic data, without a recipe update."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import zipfile
from pathlib import Path

import jax
import numpy as np
from experiment_history_confounding import observed_cycle
from experiment_model_qualification import collection, encoding, recordings
from experiment_sequence_diagnostics import fresh_recordings
from report_model_structures import recurrence

from glassbox.experimental.default_model import _HISTORY_RECIPE as _RECIPE
from glassbox.experimental.default_model import LearnedDynamics
from glassbox.experimental.sequence_model import fit_sequence_model


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def pooled_scale(batch, state_scale):
    deviation = batch.future_states - batch.past_states[:, -1:]
    scale = np.maximum(np.sqrt(np.mean(deviation**2, axis=(0, 1))), 0.01 * state_scale)
    return np.broadcast_to(scale, batch.future_states.shape[1:]).copy()


def main(args):
    assert jax.config.jax_enable_x64
    plan = json.loads(args.plan.read_text())
    assert plan["cases"] == [
        "observed_cycle-learned",
        "hidden_delay-learned",
        "process_noise-learned",
    ]
    assert plan["seeds"] == [101, 202, 303]
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "plan.json", plan)
    write(
        args.output / "environment.json",
        dict(
            python=sys.version,
            platform=platform.platform(),
            jax=jax.__version__,
            numpy=np.__version__,
            x64=True,
        ),
    )
    root = Path(__file__).resolve().parents[1]
    sources = [
        *sorted((root / "src/glassbox").rglob("*.py")),
        Path(__file__),
        root / "scripts/experiment_history_confounding.py",
        root / "scripts/experiment_model_qualification.py",
        root / "scripts/experiment_sequence_diagnostics.py",
        root / "scripts/report_model_structures.py",
        root / "tests/test_history_confounding.py",
    ]
    with zipfile.ZipFile(
        args.output / "executed-sources.zip", "x", zipfile.ZIP_DEFLATED
    ) as z:
        for p in sources:
            z.write(p, str(p.relative_to(root)))
    write(
        args.output / "sources.json",
        {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources
        },
    )
    results, max_replay_difference = [], 0.0
    for case in plan["cases"]:
        for seed in plan["seeds"]:
            name = f"{case}-{seed}"
            directory = args.output / name
            directory.mkdir()
            print(json.dumps(dict(starting=name)), flush=True)
            prior = json.loads((args.previous / name / "result.json").read_text())
            path = Path(prior["model_source"])
            assert (
                hashlib.sha256(path.read_bytes()).hexdigest() == prior["model_sha256"]
            )
            baseline = LearnedDynamics.load(path)
            fingerprint = baseline.fingerprint()
            assert baseline.report["recipe"] == _RECIPE
            b, dev = baseline._train.batch, baseline._development.batch
            scale = pooled_scale(b, baseline._model.norms["state_scale"])
            candidate, report = fit_sequence_model(
                b,
                dev,
                kind=_RECIPE["kind"],
                objective="rollout",
                width=_RECIPE["width"],
                ridge=_RECIPE["ridge_fraction"]
                * len(b.past_states)
                * b.future_states.shape[1],
                seed=_RECIPE["seed"],
                steps=_RECIPE["steps"],
                batch_size=_RECIPE["batch_size"],
                learning_rate=_RECIPE["learning_rate"],
                check_every=_RECIPE["check_every"],
                error_scale=scale,
            )
            candidate.save(directory / "candidate.npz")
            write(directory / "fit-report.json", report)
            for key in candidate.norms:
                np.testing.assert_array_equal(
                    candidate.norms[key], baseline._model.norms[key]
                )
            if case.startswith("observed_cycle"):
                supplied = observed_cycle(seed, evaluation=True)
            elif case == "hidden_delay-learned":
                supplied = collection(
                    recordings("memory", seed, evaluation=True),
                    encoding("identity", 1),
                    "memory",
                )
            else:
                supplied = collection(
                    fresh_recordings("process_noise", seed, evaluation=True),
                    encoding("identity", 1),
                    "process_noise",
                )
            origins = [
                (s, t) for s in supplied.segments for t in range(2, len(s.states) - 5)
            ]
            x = np.stack([s.states[t - 2 : t + 1] for s, t in origins])
            up = np.stack([s.inputs[t - 2 : t] for s, t in origins])
            uf = np.stack([s.inputs[t : t + 5] for s, t in origins])
            targets = np.stack([s.states[t + 1 : t + 6] for s, t in origins])
            predictions = {
                label: np.asarray(model.rollout(x, up, uf))
                for label, model in (
                    ("baseline", baseline._model),
                    ("pooled", candidate),
                )
            }
            for label, model in (("baseline", baseline._model), ("pooled", candidate)):
                replay = recurrence(model, x, up, uf)
                max_replay_difference = max(
                    max_replay_difference,
                    float(np.max(np.abs(replay - predictions[label]))),
                )
                np.testing.assert_allclose(
                    replay, predictions[label], atol=1e-9, rtol=1e-9
                )
            development_prediction = recurrence(
                candidate, dev.past_states, dev.past_inputs, dev.future_inputs
            )
            reproduced_loss = float(
                np.mean(((development_prediction - dev.future_states) / scale) ** 2)
            )
            np.testing.assert_allclose(
                reproduced_loss, report["validation_rollout_mse"], atol=1e-9, rtol=1e-9
            )
            assert baseline.fingerprint() == fingerprint
            np.savez_compressed(
                directory / "evaluation.npz",
                past_states=x,
                past_inputs=up,
                future_inputs=uf,
                targets=targets,
                loss_scale=scale,
                baseline=predictions["baseline"],
                pooled=predictions["pooled"],
                recording_ids=np.array([s.recording_id for s, _ in origins]),
                segment_ids=np.array([s.segment_id for s, _ in origins]),
                source_origins=np.array([s.start_row + t for s, t in origins]),
            )
            rms = {
                label: np.sqrt(np.mean((predicted - targets) ** 2, axis=0))
                for label, predicted in predictions.items()
            }
            result = dict(
                case=case,
                seed=seed,
                status="complete",
                baseline_model_source=str(path),
                baseline_model_sha256=prior["model_sha256"],
                baseline_fingerprint=fingerprint,
                candidate_fingerprint=candidate.fingerprint(),
                evaluation_windows=len(x),
                training_windows=len(b.past_states),
                development_windows=len(dev.past_states),
                selected_step=report["selected_step"],
                baseline_selected_step=baseline.report["optimization"]["selected_step"],
                channel_rmse={label: value.tolist() for label, value in rms.items()},
                horizon_vector_rmse={
                    label: np.sqrt(np.sum(value**2, axis=1)).tolist()
                    for label, value in rms.items()
                },
                loss_scale=scale.tolist(),
                selection_loss_replay=reproduced_loss,
            )
            write(directory / "result.json", result)
            results.append(result)
            write(args.output / "results.json", results)
            print(
                json.dumps(
                    dict(
                        case=case,
                        seed=seed,
                        baseline=result["horizon_vector_rmse"]["baseline"],
                        pooled=result["horizon_vector_rmse"]["pooled"],
                    )
                ),
                flush=True,
            )
    assert len(results) == 9
    write(
        args.output / "replay.json",
        dict(
            cases=len(results),
            max_prediction_difference=max_replay_difference,
            verified="NumPy baseline/candidate prediction replay, selected-model development loss, identical normalization constants, baseline immutability. Evaluation indexing and all score arrays are independently checked by the follow-up auditor.",
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--previous", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    main(p.parse_args())
