# The generic learner

One recipe, one module, three calls. `glassbox.experimental.default_model`
learns a differentiable dynamics model of any uniformly sampled system from
recordings of its observed signals and commands. The caller supplies signals,
units, timing and recording boundaries; nothing else. This is not the stable
structured `glassbox.fit`. See the [charter](charter.md) for what it has to
reach and [status](status.md) for the current gap.

## Consumer workflow

```python
from glassbox.experimental.default_model import LearnedDynamics, fit
from glassbox.experimental.sequence_collection import (
    SequenceCollection,
    SequenceSegment,
)

recordings = SequenceCollection(
    tuple(
        SequenceSegment(name, "whole", states, inputs, dt_s=0.05)
        for name, states, inputs in rows
    ),
    configuration_id="my-vehicle-rev-c",
    state_channels=("vx [m/s,body]", "vy [m/s,body]"),
    input_channels=("throttle [normalized,requested]",),
)

model = fit(recordings)                       # no options
future = model.predict(past_states, past_inputs, future_inputs)
model.save(path)
revision = LearnedDynamics.load(path).update(more_recordings)
```

`fit(recordings)` takes the recordings and nothing else. It needs at least two
distinct recordings, holds a quarter of them out for development, samples
windows round-robin per recording, and returns a `LearnedDynamics`.
`predict` returns JAX-compatible means in the declared observation
coordinates, excluding the initial observation. `update(recordings)` requires
whole new recordings, refits the same recipe on the merged window cache, and
leaves the current revision untouched; the new revision records its
predecessor's fingerprint. `diagnose(recordings)` is a read-only report on
input predictability and extra-history error evidence; it never modifies the
model and never admits it for control.

The report carries the recipe, the window coverage of each role, the
optimization trace, per-recording development errors against a hold-current
reference, and the evidence limits that apply to all of it. Development
targets select checkpoints; they are not independent error calibration.

## The recipe

`generic-memory-v2-prototype` is the only recipe, and a saved model that
carries anything else is rejected rather than migrated.

| Constant | Value |
| --- | --- |
| `kind` | `filter_mlp` |
| `width` | 32 |
| `memory` | 8 |
| `delay_s` | 0.1 |
| `context_s` | 0.5 |
| `horizon_s` | 0.25 |
| `training_windows` | 384 |
| `development_windows` | 256 |
| `steps` | 1000 |
| `batch_size` | 64 |
| `learning_rate` | 0.002 |
| `ridge_fraction` | 0.01 |
| `seed` | 0 |
| `check_every` | 100 |
| `hold_scale_floor` | 0.01 |

The model is a recursive affine term plus a tanh residual over explicit
`delay_s` differences and a memory readout. It is initialized from a ridge
solution of the one-step affine model with a zero memory readout and zero
residual, then trained with Adam under gradient clipping, with the
hold-current loss scales and development-rollout checkpoint selection. At
50 ms sampling the constants mean a 10-step consumed context, a 2-step
explicit history and a 5-step forecast horizon.

## The memory contract

The model consumes a fixed number of consecutive observed transitions before
the forecast origin. Its memory starts at rest at the first consumed
observation, advances once per observed transition inside one recording
segment, and continues from predicted observations during the forecast. The
consumed context is the information budget: nothing before it is implied, and
a recording boundary means starting again from rest. A caller may carry the
memory within a recording through `SequenceModel.memory_state` and
`rollout(..., memory=...)`, which is exactly equivalent to consuming the whole
context at once.

`predict` needs `history_steps + 1` aligned observations and `history_steps`
aligned inputs on the fitted time grid; a longer past is truncated to the most
recent context and a shorter one is rejected, never padded. Horizons beyond
the fitted range are rejected, not extrapolated.

Forecasts are Euclidean. They do not enforce manifold constraints, establish
support outside observed conditions, or carry a calibrated error envelope.

## The harness

[`harness/v1.json`](harness/v1.json) is the frozen manifest: eight synthetic
families over three seeds plus a delayed-input witness over three more, the
dataset plan, the absolute per-family error caps, and the paired-probe limit.
Its digest is a constant in `glassbox.experimental.harness`; a manifest whose
bytes differ is refused by both commands.

```sh
uv run python -m glassbox.experimental.harness run \
  --manifest docs/harness/v1.json --output /tmp/harness-run
uv run python -m glassbox.experimental.harness verify /tmp/harness-run
```

`run` fits `fit(recordings)` end to end on each case's calibration recordings,
scores its forecasts on independent matched and shifted recordings at every
origin with a complete consumed context, scales errors by the fitted model's
training state scale, and writes the model artifact, the evaluation arrays,
the predictions, the per-horizon and per-recording scores, the fit report and
the decision. It takes about half a minute on a CPU and exits nonzero when the
decision is a rejection.

A case is accepted when every horizon scaled RMSE is below its family's cap
and, for the witness, the paired-probe first step is below 0.05. When
[`harness/reference.json`](harness/reference.json) exists beside the manifest,
no case's overall scaled RMSE may exceed its reference value by more than 5%
plus 0.005 in either regime. Missing, duplicated, undeclared or nonfinite
scores fail closed.

`verify` refits nothing. It checks the manifest digest, the recorded artifact
hashes and every model fingerprint, replays every saved prediction with an
independent NumPy recurrence that shares no code with the fitted rollout,
recomputes the scores and the decision from the saved arrays, and rejects any
artifact that has changed.

A run copies the manifest and the reference it compared into its output, and
records the reference's sha256 in `decision.json`. Those copies are artifacts,
not authority: `verify` anchors both to the committed files, so a saved run
cannot loosen its own thresholds after the fact. It reads the committed
reference, refuses when the copy differs from it by a byte, and refuses when
one exists and the other does not. Pass `--reference PATH` to say where the
committed reference is when the replay does not run inside a checkout.

Synthetic results are a fast regression guard, not a place to win. They are
not platform readiness, control adequacy, or calibrated uncertainty.

## The platform tier

[`harness/platform-v1.json`](harness/platform-v1.json) is the second frozen
manifest, with its own digest constant. It is the accuracy tier: for each of
the five pinned corpora it declares the directory below a root, which
recordings are held out by name pattern, how many recordings each side must
hold, the evaluation origin stride, the task allowance from
[accuracy-requirements](accuracy-requirements.md) where one was derived, and
the structured comparator's arm set and fit options as that corpus's recorded
validation chain uses them. The corpus root is a command-line argument and
never a fact in the manifest.

```sh
uv run python -m glassbox.experimental.harness platform \
  --manifest docs/harness/platform-v1.json \
  --corpora /path/to/corpora --output /tmp/platform-run
uv run python -m glassbox.experimental.harness verify /tmp/platform-run
```

One adapter turns any canonical trajectory into `SequenceSegment`s, with no
per-corpus branch. The observed channels are world-frame velocity, body rates
and the nine body-to-world rotation-matrix entries from the quaternion, in that
order, with units and frames in their names; position is not modeled. Inputs
are the recorded controls, named from the trajectory's own spec. Recording
identity is the file stem, and the sample period is the corpus's declared one,
checked against every interval. A timing gap or a nonfinite observed row splits
the recording into segments through `segments_from_mask`, one uniformly sampled
block at a time, so every retained sample keeps its source row and nothing is
padded.

`platform` fits `fit(recordings)` on a corpus's training recordings and fits
the structured model, through `glassbox.fitting.fit` rather than the CLI, on
exactly those same recordings; the held-out recordings are never passed to
either fit in any role. Both models then forecast from the same origins with
the same recorded future commands: the generic model through `predict`, the
structured model through the library's rollout from that origin's full
canonical state, initialized from the real command history before it. Every
origin at the declared stride carries the recipe's whole consumed context
inside one segment and the whole horizon after it. Velocity and body-rate RMSE
are reported at the final horizon row and over the whole prefix, for the
generic model, each structured arm and a hold-current baseline, pooled per
corpus and per recording.

The recipe's 0.25 s horizon resolves on each corpus's own sample grid, so the
realized horizon is 0.24 s on the 50 Hz corpora and 0.2 s on the 5 Hz one; each
run reports the horizon it actually measured. The run saves, per corpus, the
generic model artifact, each structured belief and fit report, the evaluation
arrays with both predictions and the hold baseline, the recording identities
and source origins, a sha256 of every artifact, the wall time of every fit, and
the decision. Nothing is cached across runs.

`verify` decides which tier a directory holds from the digest of the manifest
it copied, not from what the copy says about itself, and refuses a manifest
that matches neither frozen contract. On a platform run it replays the generic
predictions with the same independent NumPy recurrence the synthetic tier uses,
replays each structured arm from its saved artifact, recomputes every score and
the decision from the saved arrays, and rejects any artifact whose bytes moved.

The declared rule is that, on every corpus, the generic model's final-step
velocity and body-rate RMSE are at or below the structured comparator's on the
same rows and at or below the allowance where one exists; the comparator is the
better structured arm on those rows, metric by metric, and both arms are
reported. The manifest currently carries `"enforced": false`: the rule is
measured and reported, a run is accepted when the measurement itself is
complete, and the rule gates merges from the first iteration that changes the
recipe. Structural problems — a corpus missing, duplicated, undeclared or
unfinished, a nonfinite score, a row count outside the declared budget, an
undeclared arm set — always fail closed.
