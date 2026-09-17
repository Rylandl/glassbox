# Fit, predict and update from recordings

Run the public generic workflow:

```bash
uv run python examples/platform_onboarding.py --output artifacts/onboarding
```

The [example](../../examples/platform_onboarding.py) generates six recordings
of one small synthetic system with two observed signals and one command. It
uses the public recipe unchanged, with no family or training-budget selection.
The fixture demonstrates the API; it is not evidence of performance on an
unseen physical system.

## Data and fit

Each recording contains 161 observations and 160 commands at 50 ms intervals.
The command at row `k` drives the transition from observation `k` to
`k + 1`. The two observation channels and the command are declared unitless.

```python
from glassbox import SequenceCollection, SequenceSegment, fit

recordings = SequenceCollection(
    tuple(
        SequenceSegment(name, "whole", states, inputs, dt_s=0.05)
        for name, states, inputs in rows
    ),
    configuration_id="onboarding-demo-v1",
    state_channels=("signal_a [unitless]", "signal_b [unitless]"),
    input_channels=("command [unitless,requested]",),
)
model = fit(recordings)
```

Replace `rows` with your aligned recordings. Each fit describes one
configuration and signal contract. Retain distinct source recording identities;
multiple segments from the same recording share one ID.

The example keeps data roles separate:

| Recording | Role |
| --- | --- |
| 0–2 | Automatic training and development split |
| 3 | Untouched initial evaluation and prediction example |
| 4 | Fresh update data |
| 5 | Common untouched evaluation of the original and updated revisions |

The recipe deterministically reserves approximately one quarter of the fit
recordings for development, at least one. `model.report["training"]` and
`["development"]` identify the actual split; it is not chosen by input order.
Development windows select the checkpoint and calibrate the envelope.

## Predict with real history

```python
p, h = model.history_steps, model.horizon_steps
past_states = recording.states[: p + 1]
past_inputs = recording.inputs[:p]
future_inputs = recording.inputs[p : p + h]
future = model.predict(past_states, past_inputs, future_inputs)
half_width = model.envelope(h)
```

Here `recording` is an untouched `SequenceSegment`. The returned means have
shape `(h, number_of_observed_channels)` and exclude the current observation.
The future commands are known recording inputs for this replay. In an
application they are the command sequence whose consequences you want to
predict.

Use the model's fitted history and horizon lengths. Short history is rejected;
gaps are separate segments, not padding. The [learner contract](../learner.md)
also describes batch queries and sample-grid behavior.

## Save, update and compare

```python
from glassbox import LearnedDynamics
from glassbox.workflows.forecast import evaluate

model.save("model.npz")
loaded = LearnedDynamics.load("model.npz")
revision = loaded.update(new_recordings)
revision.save("updated-model.npz")

before = evaluate(loaded, evaluation_recordings)
after = evaluate(revision, evaluation_recordings)
```

The update uses only fresh recording 4. Both evaluations use recording 5, which
neither model has seen. The example saves both reports even when an error gets
worse. Updating does not select the better revision or switch a controller.

Evaluation reports per-horizon, per-channel RMSE, hold-current RMSE and
envelope coverage over every complete window, with per-recording results
beside the pooled values. Errors stay in each channel's declared units.
Neighboring windows overlap; their count is not a count of independent
experiments.

The original model remains unchanged. The revision records its predecessor's
fingerprint and refits against the retained development cache. New evaluation
data are still needed to assess whether the revised model helps.

## Recording files and the CLI

The same example writes generic recording NPZs using:

```python
from glassbox.io.recordings import load_recordings, save_recordings

save_recordings(recordings, "calibration.npz")
restored = load_recordings("calibration.npz")
```

These files preserve the configuration, ordered channels, segment identities,
source offsets and optional excitation arrays. They are different from the
canonical flight-trajectory files produced by telemetry extraction. The
[PX4 guide](px4-ulog.md#fit-from-what-came-out) shows explicit conversion.

You can fit and evaluate the example's files directly:

```bash
uv run glassbox fit artifacts/onboarding/calibration.npz \
  --model artifacts/onboarding/cli-model.npz \
  --report artifacts/onboarding/cli-fit.json
uv run glassbox evaluate artifacts/onboarding/cli-model.npz \
  artifacts/onboarding/evaluation.npz \
  --report artifacts/onboarding/cli-evaluation.json
```

## Outputs

The output directory contains:

- `calibration.npz`, `initial-evaluation.npz`, `new-recordings.npz` and
  `evaluation.npz`: generic recordings with the roles above.
- `model.npz`, `fit.json`, `updated-model.npz` and `update-fit.json`: both
  revisions and their reports.
- `prediction.npz`: observed history, future commands, targets, forecast and
  envelope from the initial evaluation recording.
- `initial-evaluation.json`, `before-update.json` and `after-update.json`:
  untouched evaluation results.
- `summary.json`: recording roles, revision identities and the common
  evaluation's final-step per-channel RMSEs.

`--output` changes only the destination. Repeating the example replaces its
named output files. The recipe and synthetic data seeds remain fixed.

## Evidence boundaries

The walkthrough checks usable interfaces, saved replay and immutable updates.
It does not certify control or prove calibrated uncertainty. Current platform,
control and coverage evidence is recorded in [status](../status.md). The
structured streaming and live-control examples are retained research
consumers described in [streaming refinement](../streaming-refinement.md),
separate from this generic onboarding path.
