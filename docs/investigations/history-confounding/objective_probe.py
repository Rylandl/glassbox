"""Post-hoc, read-only objective decomposition for the exact five-step cycle."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Invoke from the Glassbox checkout, including when using the archived copy.
ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "scripts"))
from report_model_structures import recurrence

from glassbox.experimental.default_model import LearnedDynamics


def main(run, output):
    results = []
    for seed in (101, 202, 303):
        path = run / f"observed_cycle-learned-{seed}/model.npz"
        model = LearnedDynamics.load(path)
        before = model.fingerprint()
        train, dev = model._train.batch, model._development.batch
        scale = np.asarray(model.report["optimization"]["error_scale"])
        reference = np.sqrt(np.mean((train.future_states - train.past_states[:, -1:]) ** 2, axis=0))
        floor = 0.01 * model._model.norms["state_scale"]
        np.testing.assert_allclose(scale, np.maximum(reference, floor), rtol=1e-12)
        np.testing.assert_array_equal(train.future_states[:, -1], train.past_states[:, -1])
        predicted = recurrence(model._model, dev.past_states, dev.past_inputs, dev.future_inputs)
        mse = np.mean((predicted - dev.future_states) ** 2, axis=0)
        normalized = mse / scale ** 2
        reported = model.report["optimization"]["validation_rollout_mse"]
        np.testing.assert_allclose(np.mean(normalized), reported, rtol=1e-9)
        assert model.fingerprint() == before
        results.append(dict(
            seed=seed, fingerprint=before,
            training_windows=len(train.past_states), development_windows=len(dev.past_states),
            selected_step=model.report["optimization"]["selected_step"],
            training_hold_rmse=reference.tolist(), loss_scale=scale.tolist(),
            fifth_vs_first_squared_error_weight=(scale[0] / scale[-1]) ** 2,
            development_channel_rmse=np.sqrt(mse).tolist(),
            normalized_loss_by_horizon_channel=normalized.tolist(),
            loss_fraction_by_horizon=(normalized.sum(axis=1) / normalized.sum()).tolist(),
            physical_mse_fraction_by_horizon=(mse.sum(axis=1) / mse.sum()).tolist(),
            reconstructed_selection_loss=float(np.mean(normalized)),
            reported_selection_loss=reported,
        ))
        results[-1]["fifth_vs_first_squared_error_weight"] = results[-1]["fifth_vs_first_squared_error_weight"].tolist()
    report = dict(
        status="post-hoc read-only measurement; no retraining or causal ablation",
        reason="The analytical witness repeats exactly at the five-step training horizon, making hold-current error zero there. Inspect whether the fixed per-horizon loss scaling magnifies this horizon.",
        cases=results,
        limits="Weights and selected-model loss shares are verified, but do not isolate the effect of objective scaling from capacity, optimization, representation, or data size. Cross-channel physical MSE shares are descriptive in this synthetic unitless encoding.",
    )
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for r in results:
        print(json.dumps(dict(seed=r["seed"], fifth_vs_first_weight=r["fifth_vs_first_squared_error_weight"], fifth_loss_fraction=r["loss_fraction_by_horizon"][-1], fifth_physical_mse_fraction=r["physical_mse_fraction_by_horizon"][-1])))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    main(args.run, args.output)
