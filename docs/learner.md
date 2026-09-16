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

[`harness/platform-v2.json`](harness/platform-v2.json) is the second frozen
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
  --manifest docs/harness/platform-v2.json \
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
reported. The manifest carries `"enforced": true`: the rule is a gate, and one
corpus above its comparator or its allowance rejects the run. Structural
problems — a corpus missing, duplicated, undeclared or unfinished, a nonfinite
score, a row count outside the declared budget, an undeclared arm set — always
fail closed.

[`harness/platform-reference.json`](harness/platform-reference.json) is the
regression reference beside it, holding the generic model's final-step velocity
and body-rate RMSE per corpus from the merged run that first measured them. No
corpus may exceed its reference value by more than 5% plus 0.005 on either
metric, and a missing, null or non-numeric reference value fails closed. It is
anchored exactly the way the synthetic reference is: a run copies the reference
it compared into its output and records the sha256 in `decision.json`, and
`verify` reads the committed file, refuses when the copy differs from it by a
byte, and refuses when one exists and the other does not. Pass `--reference
PATH` to say where the committed platform reference is when the replay does not
run inside a checkout.

Each manifest records, in full, the recipe it was frozen against, and pins the
evaluation plan that recipe is cut from: the `context_s`, `delay_s` and
`horizon_s` the information budget of every window comes from. It does not pin
the rest of the recipe, because a gate only the recipe that froze it could pass
would never measure a change. Every run records the recipe it actually fitted.

## The control tier

[`harness/control-v2.json`](harness/control-v2.json) is the third frozen
manifest, with its own digest constant. It is the control tier: one Cascade X8
trial set, driven once by the structured belief and once by the generic
learner, through the same plant and the same NMPC seam.

```sh
uv run --group cascade python -m glassbox.experimental.harness control \
  --manifest docs/harness/control-v2.json --output /tmp/control-run
uv run python -m glassbox.experimental.harness verify /tmp/control-run
```

The manifest pins the Cascade X8 specification hash and the pinned source
revision the run checks its installed simulator against, the cruise reference
with its small altitude variation, the calibration protocol — three eight-second
recordings with the published X8 stabilizer, simulator-derived trim feedforward
and the declared excitation seeds, and a fourth recording reserved and never
fitted in any role — twelve-second trials repeated twice with alternating arm
order, the controller policy both arms share, how each arm is fitted, and the
metrics. The recording protocol is the one
[`examples/cascade_refinement.py`](../examples/cascade_refinement.py) collects
the structured belief's calibration with, constant for constant, and a
Cascade-marked test asserts that the two produce the same arrays byte for byte.

The declared rule is that the generic arm's position and attitude tracking RMSE
are at or below the structured arm's on the same trial, on every trial, with no
terminated trial. The manifest carries `"enforced": true`: the rule is a gate,
and one trial whose generic position or attitude RMSE is above the structured
arm's rejects the run. Structural problems never wait for that. A trial missing,
duplicated, undeclared, terminated, short of its declared intervals, or
carrying a metric that is not a finite number fails closed either way.

`control-v2` carries `control-v1`'s protocol constant for constant — the plant
hash and pinned revision, the calibration recordings and their seeds, the
reference, the trial duration and repetitions, both arms and the controller
policy they share. Only the rule's standing changed, and it changed before any
candidate was fitted. `control-v1` is deleted, not kept beside it.

`verify` recognizes the tier from the digest of the manifest a run copied. It
recomputes every metric from the saved per-interval tracking arrays with the
library's own metric code, rebuilds the reference rows from the saved initial
state so a run cannot score itself against a reference it invented, recomputes
the hashes of the tracking arrays, the calibration recordings and both fitted
artifacts, and replays the decision. It does not rerun the plant or the solver,
and says so in what it returns: neither is deterministic under a wall clock, so
rerunning either would be a new measurement rather than a check of this one.
The tier declares no regression reference, because this is its first
measurement and there is nothing yet to anchor.

## The plan model

`glassbox.experimental.learned_plan` presents a fitted `LearnedDynamics` to
`glassbox.control.plan.PlanModel`, the seam a bounded shooting solver plans
over. It is the control boundary for the learner exactly as
`glassbox.control.fitted` is for a dynamics belief, and nothing on the solver's
side learns which one it is driving.

Coordinates are bridged both ways. The controller's rigid-body state becomes the
learner's fifteen observed channels — world velocity, body rates and the
rotation entries from the quaternion, exactly as the platform tier's adapter
builds them. Coming back, velocity and body rates are predicted channels,
position is integrated trapezoidally from the predicted world velocity starting
at the supplied state's own position, and the predicted rotation entries are
projected onto the nearest rotation and read back as a quaternion. The
projection is Higham's Newton iteration for the orthogonal polar factor, whose
derivative is well conditioned at a rotation, where a singular-value
decomposition's is not; both it and the quaternion recovery live with the
library's other geometry helpers.

History is the other half. A recursive learner's forecast means nothing without
the observed transitions before its origin, and a control loop has them:
`LearnedPlanController` carries them interval by interval from the states it
observes and the commands it applies, and resets them at a trial start, which
is a recording boundary and where the memory starts again at rest. Nothing is
padded. Until the loop has observed the explicit-difference window there is no
forecast, `ready` is false, and the trial holds the command it is already
applying; on this 50 ms grid that is the first two intervals of a trial, and
they are scored like every other. Once the consumed context is full the carried
history reproduces what `predict` would compute from the same rows. Carrying it
needs one small, general extension to the seam: `PlanValues` gains an optional
`observed_history` slot, defaulted to `None`, so a model with its own memory
moves it every control interval without paying a recompile. Nothing about how a
belief is presented changes.

Command bounds are the caller's declared telemetry contract, required as
arguments. The learner observed commands; it was never told what the actuators
accept, and it is not asked. The horizon is the recipe's own fitted `horizon_s`
and a longer one is refused rather than rolled past, because `predict` refuses
it: a plan model that extrapolated where `predict` will not would be claiming
evidence the fit never produced. No uncertainty is claimed either.
`uncertainty_available` and `uncertainty_complete` are both false, both
robustness terms are exactly zero, and a solve runs only under the seam's
explicit no-evidence override, which the run records. Validity utilization is
reported as zero because the learner declares no support envelope, not because
one was checked and found clear.

### The measurement

The gate was frozen and committed at `367530e` before any trial was run, and
re-frozen as `control-v2` at `3c149fa`, with the rule enforced, before any
candidate was fitted. The run below is `control-v2`'s, and it reproduces the
earlier one digit for digit: the plant, both fits and both arms are
deterministic, so the same protocol measures the same numbers.

| Repetition | Position RMSE, generic / structured (m) | Attitude RMSE, generic / structured (deg) | Terminated | Deadline misses, generic / structured |
| --- | --- | --- | --- | --- |
| 0 | 60.347 / 1.208 | 121.874 / 1.022 | none | 0 / 1 |
| 1 | 60.347 / 1.208 | 121.874 / 1.022 | none | 0 / 0 |

**The rule is not met and the run is rejected.** No trial terminated: all four
completed 240 intervals with finite states, bounded commands and no solver
fallback, so there is no structural breach; the four rule breaches are the
generic arm's two metrics on each of the two repetitions. The two
repetitions are identical to every digit because the plant and both arms are
deterministic and the loop is paced; only the deadline misses, which are
host-specific and informational, differ between them. The structured arm plans
a 0.80 s horizon, capped by its own forecast-error evidence; the generic arm
plans the recipe's fitted 0.25 s.

The generic arm does not diverge numerically — it flies the aircraft into the
ground. Position error is 0.005 m at 0.5 s, better than the structured arm's
0.176 m there, and attitude error is 0.59 degrees against 3.41. It then loses
the aircraft: 9.3 degrees of attitude error at 1 s, 117 at 3 s, and an altitude
of 100 m at the start, 91 m at 3 s and -26 m at 12 s. Its commands saturate:
mean applied throttle 0.103 against a trim of 0.437, with roll and pitch resting
near their +0.35 bounds.

The mechanism is measured, and it is not the fit's forecast quality. On the
reserved recording the learner's own 0.25 s forecast beats hold-current on every
channel group — world velocity 0.168 against 0.249 m/s, body rate 0.139 against
0.356 rad/s, rotation entries 0.0149 against 0.0212. What it does not have is a
usable command Jacobian in the one direction the controller reaches for first.
Comparing each arm's final-step response to a +0.05 command step against the
plant's own, averaged over three held-out origins:

| Command | Direction cosine, generic / structured | Magnitude ratio, generic / structured |
| --- | --- | --- |
| throttle | -0.056 / 0.968 | 26.9 / 1.39 |
| roll | 0.594 / 0.798 | 0.58 / 0.60 |
| pitch | 0.969 / 0.951 | 1.12 / 1.03 |

Pitch is as good as the structured model's and roll is comparable, but throttle
is uncorrelated with the plant's response and 27 times too large. The
calibration explains the arithmetic: across the three recordings throttle moves
with a standard deviation of 0.0225 to 0.0312 over a declared range of 1.0,
while roll moves 0.1117 to 0.1727 and pitch 0.2598 to 0.2671 over ranges of 0.7.
Because the affine start standardizes each command by that sample standard
deviation, the recipe's ridge charges the throttle level column 0.0127 per full
declared-range move where it charges pitch 2.126, 168 times weaker, and the
throttle difference columns 2.0e-4 and 6.6e-5. Open-loop scoring never charges
for the result either, because throttle barely moves in the evaluation data. An
optimizer charges for it immediately: it drives throttle to a bound, on 79% of
this trial's intervals.

**The horizon is not the difference.** Replanned at the generic arm's own 0.25 s
horizon, the structured arm still tracks, at 1.738 m and 1.371 degrees: 44% and
34% worse than at 0.80 s, and 35 and 89 times better than the generic arm. Its
applied throttle stays within [0.437, 0.553] and its roll within
[+0.004, +0.007], never at a bound.

**Restating the scale or the penalty in physical units does not fix it.** Both
of those changes are the same solve, and both do to the affine start what the
arithmetic predicts, moving the throttle column from -0.048 / 25.17 to
0.259 / 0.127; neither survives the fit, because the ridge lives only in the
initializer and the training objective never charges for the command Jacobian.
Nor would carrying it into the objective help. Sweeping the throttle columns'
ridge over twelve decades drives the magnitude ratio from 25.42 to zero and
never lifts the direction cosine above -0.044, while roll's best, 0.764, and
pitch's, 0.965, are already reached at the recipe's own ridge. The throttle
column is not identified by these recordings at any penalty; it can only be made
small, and small is its own hazard, because a bounded solver that believes a
command is weak spends more of it. [`status.md`](status.md) carries the trial
ladder that measures each of these.

This is a measurement of one cruise trial set on one simulated plant, not
hardware readiness, a real-time claim, or calibrated uncertainty. On this
reference a model with no command authority at all scores 0.109 m and 0.000
degrees, better than either arm, which is worth knowing about what the rule can
and cannot certify.
