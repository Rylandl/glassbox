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
envelope = model.envelope(len(future))        # half-widths beside those means
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
reference, the envelope, and the evidence limits that apply to all of it.

## The envelope

Every forecast carries a measured error envelope. There is no caller option
and no way to turn it off. `envelope(horizon_steps)` returns one half-width
per horizon step and per declared observation channel, in that channel's own
physical units, aligned row for row and column for column with what `predict`
returns over the same horizon; a longer horizon is rejected exactly as
`predict` rejects it, and the returned array is a copy.

It is a split-conformal quantile at a nominal 90%: for each horizon step and
channel, the `ceil((n + 1) * 0.9)`-smallest of the `n` absolute forecast errors
on the development windows, and the largest of them when that rank exceeds `n`.
The calibration data is exactly the windows the recipe already holds out. No
training window, no scored row and no held-out recording of any tier enters it
in any role, and `update` recalibrates on the same pinned development cache it
refits against, so a revision never carries its predecessor's envelope forward.

Two things it is not. The development windows also select the training
checkpoint, so they are held out of every gradient step but not of model
selection: exchangeability is a stated approximation rather than an
independent calibration set. And the nominal level holds on those windows by
construction; whether it holds anywhere else is measured, not claimed. The
[evidence tier](#the-evidence-tier) is where it is measured.

## The recipe

`generic-memory-v3-prototype` is the only recipe, and a saved model that
carries anything else is rejected rather than migrated. A `v2` artifact, which
carried no envelope, is refused on load rather than migrated: a forecast
without a measured envelope is not a forecast this recipe makes, and inventing
one on load would be the opposite of measuring it.

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

Forecasts are Euclidean. They do not enforce manifold constraints and they do
not establish support outside observed conditions. They do carry the measured
error envelope above, whose coverage is a measurement on the rows it was
measured on rather than a guarantee.

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

[`harness/platform-v3.json`](harness/platform-v3.json) is the second frozen
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
  --manifest docs/harness/platform-v3.json \
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
reported. The rule is reported on every corpus and metric, and `rule_met` in the
decision says whether it holds on all ten of those cases. That is the statement
the accuracy row of [`status.md`](status.md) is read from, and it is stronger
than an accepted run.

The decision itself is the one both this tier and the control tier make. A run
is accepted when three things hold. No metric regressed: no corpus's generic
final-step RMSE exceeds its reference value by more than 5% plus 0.005 on either
metric. The rule holds wherever the reference already meets it: a corpus and
metric whose recorded reference value is itself at or below this run's
comparator and allowance is a gate, and one this run fails rejects it; a case
the reference already fails is reported with `"gating": false` and does not.
And nothing structural failed — a corpus missing, duplicated, undeclared or
unfinished, a nonfinite score, a row count outside the declared budget, an
undeclared arm set — which never waits for what the reference says.

That last distinction is the whole of `platform-v3`. The incumbent does not meet
the rule on arp, so an unconditional gate would reject every candidate for a
case no candidate had changed, and the Evidence and Live rows could never be
measured at all. A candidate that gives ground on arp still cannot hide: it is
rejected by the regression reference, which is arp's own recorded number.

[`harness/platform-reference.json`](harness/platform-reference.json) is the
regression reference beside it, holding the generic model's final-step velocity
and body-rate RMSE per corpus from the merged run that first measured them. It
carries `platform-v2`'s numbers forward unchanged apart from the manifest id it
names, because the protocol did not change. A missing, null or non-numeric
reference value fails closed, both as a regression and as a case the rule cannot
be shown to gate. It is anchored exactly the way the synthetic reference is: a run copies the reference
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

[`harness/control-v4.json`](harness/control-v4.json) is the third frozen
manifest, with its own digest constant. It is the control tier: one Cascade X8
trial set, driven once by the structured belief and once by the generic
learner, through the same plant and the same NMPC seam, computed in simulated
time.

```sh
uv run --group cascade python -m glassbox.experimental.harness control \
  --manifest docs/harness/control-v4.json --output /tmp/control-run
uv run python -m glassbox.experimental.harness verify /tmp/control-run
```

The manifest pins the Cascade X8 specification hash and the pinned source
revision the run checks its installed simulator against, the tracking task, the
calibration protocol — three eight-second recordings with the published X8
stabilizer, simulator-derived trim feedforward and the declared excitation
seeds, and a fourth recording reserved and never fitted in any role —
sixteen-second trials repeated twice with alternating arm order, the controller
policy both arms share, how each arm is fitted, and the metrics.

The task is the one [`cascade-accuracy`](cascade-accuracy.md) declares: the trim
state carried forward at 18 m/s and 100 m, with lateral position `sin(0.35 t)`
metres and altitude `100 + 0.75 sin(0.3 t)` metres and the world velocities that
match them. Each repetition starts from its own perturbed state, drawn from a
declared seed as that page draws it — up to 0.15 m of lateral and vertical
position, 0.05 m/s of lateral and vertical velocity, 0.01 rad of attitude and
0.02 rad/s of body rate, applied through the library's retraction so the
quaternion stays on the unit sphere. It is the only disturbance: wind is zero,
sensing is simulator truth, and nothing is added during a trial. Both arms of a
repetition fly from the same perturbed state, and the reference is anchored to
the unperturbed trim state and does not move with it, which is what `verify`
rebuilds it from.

`control-v2`'s cruise reference is deleted rather than kept beside it. On that
reference a model with no command authority at all scored better than either
arm, because the reference was the trim trajectory the plant was already on: the
trial certified claiming nothing. Under `control-v3`, holding the trim command
for the whole trial scores 1.217 m and 0.753 degrees on the first repetition and
1.728 m and 1.896 degrees on the second, against the structured arm's 1.179 m
and 1.320 degrees, and meets the page's tolerance on none of the 281 scored
samples of either trial.

### The trajectory is computed in simulated time

`control-v4` replaces `control-v3`, which is deleted, and changes nothing about
the protocol: the same plant hash and pinned source revision, the same task, the
same calibration and its excitation requirement, the same seeds, the same
controller policy, the same two arms, the same metric definitions, the same RMSE
rule and the same pass criterion. What it changes is the clock the trajectory is
computed on.

`control-v3` paced its loop against a wall clock and held the solver to a 50 ms
deadline, which meant a busy host could change where the aircraft flew. Reviewing
the rejected evidence candidate, a `control-v3` run made while the platform tier
and the test suite loaded the machine put the structured arm at 1.155 m on the
first repetition against its reference's 1.179 m, and the same run made alone
reproduced the reference to every digit. A 5% regression allowance on a number
that moves with machine load gates on scheduling noise.

`control-v4` adopts what `live-v2` already does, and
[says it the same way](#nothing-a-clock-measured-reaches-the-trajectory).
Interval `k` is the state at `k` times the sample interval: the loop is not
paced and reads no clock to decide anything, and the solver is given no
deadline, so it never returns a fallback command for want of time and the
command it solved is the command the plant is stepped with. `solve_deadline_s`
survives in the manifest only as the threshold solve times are reported against,
and it is the sample interval.

The wall measurements are all still taken, in the two files the live tier splits
them into. `tracking.npz` carries the trajectory alone -- times, observed states,
reference rows, applied commands, the solver-used and fallback flags, the
perturbed start and the anchor -- and `timing.npz` and each trial's `wall` block
carry interval times, solve times, the two counts over the sample interval, and
whether a deadline was assessed on each interval. Nothing reads the second group
back.

Two consecutive runs made while four concurrent synthetic-tier runs and the fast
test suite loaded the machine produced byte-identical `tracking.npz` on all four
trials -- `58e0aa06`, `aab97c7d`, `77df70f1` and `13736903` -- and identical
metrics, pass statistics and solver-status counts, while their recorded clocks
differed: the structured arm's solves over the 50 ms threshold were 1 and 3 in
the first run against 1 and 1 in the second. Under `control-v3` each of those
would have been a deadline the solve was cut short by.

Both arms are fitted on the same calibration, and the manifest now says what
that calibration has to contain: every command channel's standard deviation, in
each of the three recordings that fit the arms, is at least 10% of the channel's
declared range. The run measures it per channel and per recording, records it,
and fails closed before either fit when a channel falls short. One setpoint
constant carries it — the pilot's airspeed amplitude, 0.4 m/s under `control-v2`
and 2.5 m/s here — and a Cascade-marked test puts that one constant back and
asserts that the tier's recording is byte for byte the one
[`examples/cascade_refinement.py`](../examples/cascade_refinement.py) collects.
Excitation is a property of the recordings the caller supplies, applied
identically to both arms. Nothing about it reaches the learner, which is still
told the channels and nothing else, and the learner still does not report its
absence: that is the Evidence row.

The declared rule is that the generic arm's position and attitude tracking RMSE
are at or below the structured arm's on the same trial, on every trial. It is
decided exactly as the platform tier's is: a run is accepted when no metric
regressed past its reference value by more than 5% plus 0.005, the rule holds on
every trial and metric the reference already meets it on, and nothing structural
failed. A trial missing, duplicated, undeclared, terminated, short of its
declared intervals, or carrying a metric that is not a finite number fails
closed whatever the reference says. `rule_met` is reported beside `accepted`,
and the control row of [`status.md`](status.md) is read from `rule_met`.

Beside the rule the run records the page's own pass criterion, per trial and per
arm, and reports rather than gates it: both absolute lateral and altitude errors
at most 0.5 m in at least 95% of the 20 Hz samples at `t >= 2 s`, with no
terminated trial. There are 281 scored samples in a complete trial. Every
unexecuted interval counts as outside tolerance and a failed trial is not
discarded; its scored error is unbounded and is serialized as `null`, never as
zero. It is a provisional application requirement for one task and one
controller, not a learned or universal flight tolerance, and it does not decide
this run.

[`harness/control-reference.json`](harness/control-reference.json) is the
regression reference, both arms' position and attitude RMSE per trial and the
pass statistic beside them, from the incumbent measurement below. It is anchored
the way the other two are, and `--reference PATH` says where the committed file
is when the replay does not run inside a checkout.

`verify` recognizes the tier from the digest of the manifest a run copied. It
recomputes every metric from the saved per-interval tracking arrays with the
library's own metric code, rebuilds the reference rows from the saved anchor
state so a run cannot score itself against a reference it invented, rederives
each trial's perturbed start from its declared seed so a run cannot invent where
it started either, remeasures the calibration's command excitation from the
saved recordings and refuses one that falls short, recomputes the pass criterion
and the hashes of the tracking arrays, the calibration recordings and both
fitted artifacts, and replays the decision. It also checks that the run was
computed in simulated time as far as the run's own artifacts can say: it
recomputes the two recorded counts from the saved solve and interval times, and
rejects a run whose row claims a deadline was applied, whose recorded count of
deadline-assessed intervals is not zero or disagrees with the per-interval array
it came from, or whose solver statuses report a deadline expiring. None of that
proves a deadline was absent; each of them rejects a run whose own record says
one decided something. It does not rerun the plant or the solver, and says so in
what it returns: rerunning either would be a second measurement rather than a
check of this one, and the run whose determinism the tier claims is the one that
was saved.

## The evidence tier

[`harness/evidence-v2.json`](harness/evidence-v2.json) is the fourth frozen
manifest, with its own digest constant. It is not a fourth command. Coverage is
measured inside the synthetic, platform and control runs, on exactly the rows
those tiers already score, so each of the three copies this manifest into its
output as `evidence-manifest.json` and folds one band decision into its own.
`verify` checks that copy against the digest constant rather than against the
file, which is the authority the three tier manifests are held to.

What it declares is the envelope above — a nominal 90% half-width per horizon
step and per channel, calibrated only on the development windows the fit
already holds out — the channel groups coverage is reported over, and the band
that coverage has to land in, 85% to 95%. The groups are world velocity, body
rates and the nine rotation entries for the fifteen-channel contract the
platform and control tiers share, and one group per channel for the synthetic
families, whose channels are coordinates of unrelated systems.

Coverage is the fraction of scored `(row, channel)` pairs in a group whose
absolute forecast error at that horizon step is at or below the envelope's
half-width for that step and channel. It is measured on every platform corpus's
held-out rows at the recipe's horizon, on every synthetic case's evaluation
rows in every declared regime, and on the control tier's reserved recording, at
every origin there that carries the whole consumed context and the whole
horizon. Every run saves an envelope half-width array beside every prediction
array it already saves, and `verify` rebuilds those half-widths from the saved
model artifact, refuses an array that is not the model's own, and recomputes
every coverage number from the replayed prediction and the saved targets.

The band is enforced, under the semantics the platform and control tiers
already use: the metric is `band_excess`, the distance a coverage lies outside
the band and zero inside it; a run is accepted only when no case's
`band_excess` regresses past its reference value times 1.05 plus 0.005, the
band holds on every case the reference already meets it on, and nothing
structural fails. A coverage above the band breaches it exactly as one below it
does: an envelope wider than the errors it carries is as far from the declared
level as one narrower than them, and a candidate may not trade a case that
holds the band for one that does not.

`evidence-v1` is deleted rather than kept beside it. It declared the same
envelope, groups, band and decision and differed in one field, `enforced`,
which it left false for its own first measurement and said would gate from the
first candidate after it. That candidate was
[`status.md`](status.md)'s Evidence attempt 1, and the flag was still false when
it was measured, so this manifest states the rule that was always meant to
apply to it.

Structural problems always fail closed once the band is enforced: a declared
case missing, duplicated or undeclared, a tier measuring a different case set
than the manifest declares, a declared channel group missing, a horizon-step
count that differs from the tier's own, a coverage that is not a finite number
in `[0, 1]`, or a scored-row count of zero.

Coverage is a measurement on the rows it was measured on. It is not a
probabilistic guarantee, not a support envelope, not a claim about conditions no
recording covers, and not control adequacy.

## The live improvement tier

[`harness/live-v2.json`](harness/live-v2.json) is the fifth frozen manifest,
with its own digest constant and the harness's fourth command. It is the live
improvement tier: the same Cascade X8 trial set, flown from the first interval
by the frozen structured belief, while the generic learner refits on the
trial's own streamed transitions and is allowed to take the controller when a
predeclared held-out gate passes. The evidence tier is not folded into it:
coverage is measured on the rows the synthetic, platform and control runs
already score, and this tier measures the live improvement row and nothing
else.

```sh
uv run --group cascade python -m glassbox.experimental.harness live \
  --manifest docs/harness/live-v2.json --output /tmp/live-run
uv run python -m glassbox.experimental.harness verify /tmp/live-run
```

The plant, the tracking task, the calibration protocol, the seeds, the
controller policy and both fitted arms are `control-v3`'s, constant for
constant. What `control-v3`'s two model arms become here are two live arms. The
**frozen** arm flies the structured belief for all 320 intervals and is never
offered anything; the **adopting** arm flies the same structured belief and may
be handed the generic plan model once. Both run the same transport and the same
refits, so they pay the same compute and differ in the swap and in nothing else,
and the frozen arm is what says where "before" would have gone. Two repetitions,
arm order alternating as [`examples/live_refinement.py`](../examples/live_refinement.py)
alternates its own.

### Nothing a clock measured reaches the trajectory

This is the one thing `live-v1` got wrong and the only reason it was replaced.
That manifest paced its loop against a wall clock and gave the solver a 50 ms
deadline while a background worker loaded the same CPU. Rerun on the same
checkout, every tracking number moved: deadlines were missed nondeterministically,
fallback commands changed the flight, and refit completion times moved the swap
by an interval or two. A regression reference and a 5% allowance on that would
have gated on scheduling noise, and the frozen arm would have failed its own
reference. `control-v3`, which flew the same plant and the same solver without a
worker, reproduced digit for digit on an idle host, which is what isolated the
cause; it did not reproduce under load either, and `control-v4` is the same fix
applied to it.

`live-v2` computes the trajectory in simulated time. Interval `k` is the state
at `k` times the sample interval, and four things make that true. The loop is
not paced and reads no clock to decide anything. The solver is given no
deadline, so it never returns a fallback command for want of time and its
commands are a function of the model and the state alone; the measured solve
times and the count of solves over 50 ms are still recorded, and decide
nothing. The refinement worker is driven synchronously -- the producer's own
thread runs the score, the refit and any controller preparation inside the
`submit` that hands a block over -- so no thread schedule can decide when a
block was learned from, nothing queues and nothing drops. And a candidate is
released at a declared offset of one block period after the block it was scored
on, rather than whenever a refit happened to finish, which makes the swap
interval a manifest constant.

A refit that runs over its wall budget is reported as a budget breach and
offered anyway. Withholding it would put the host's scheduling back into the
commands, which is the thing this manifest exists to prevent, so the breach
gates nothing for the same reason a deadline miss gates nothing.

The wall measurements are all still taken. `tracking.npz` carries the
trajectory alone -- times, observed states, reference rows, applied commands,
the active revision per interval, the perturbed start and the anchor -- and
`timing.npz` and each trial's `wall` block carry interval times, solve times,
telemetry times and every refit's and every score's wall seconds. Nothing reads
the second group back.

### The transport and the gate

`TransitionBuffer` takes each interval's aligned transition -- the observed
state it started at, the observed state it ended at, and the command the loop
actually applied over it -- and emits whole 40-interval blocks. Forty intervals
is two seconds at this cadence: the smallest block that clears the recipe's
demand of at least three complete windows in any new recording -- a ten-step
consumed context and a five-step horizon leave 26 origins in a 40-step block --
and long enough that one score, one refit and one controller preparation fit
inside the interval that produces the next block. The buffer emits nothing until
it holds the declared 20 intervals of known commands, so the first block covers
intervals 20 to 59 and seven whole blocks fit in a trial; the last 20 intervals
are a partial block and are never submitted.

`RefinementWorker` now accepts either a belief, which it wraps in a
`ModelRefiner` as it always did, or a refiner of the caller's own keeping the
`Refiner` protocol the module declares, and it takes a `synchronous` option that
runs the same learning on the producer's thread and carries out an acknowledged
decision inside `acknowledge`. The structured path and both examples are
untouched and still run the worker thread; this tier supplies its own refiner
over the generic recipe and drives it synchronously. Each block is scored before
anything is fitted on it, by the current candidate and by the frozen structured
belief, on identical rows: every origin of the block carrying the whole consumed
context and the whole horizon, with the recorded commands, the same canonical
state and the same 20-command actuator history. Then the block is absorbed with
`update(recordings)` as one new recording with its own identity. The recipe's
holdout and window sampling are untouched -- the development windows the
calibration fit reserved stay reserved and keep selecting every checkpoint, and
the fresh block's windows enter the training cache under the recipe's own
round-robin, which caps it at 384 windows. The declared budget is the recipe's
own 1000 fit steps and two block periods of wall time, which is the rate a
stream of this block size has to be kept up with.

**The swap gate reads no tracking error.** The candidate is offered as the
active plan model when its final-step forecast error on the most recent block it
did not fit on is at or below the structured belief's on the same rows, for
world velocity and for body rate, both. A score that cannot be read leaves that
metric unknown and an unknown metric never passes. The first candidate that
passes is prepared -- the worker builds and warms a bounded solver over it, so
no compilation lands in a timed interval -- and one block period later the
control owner seeds that controller with the trial's own retained observed
states and applied commands and installs it. The observed history is a property
of the flight, not of the revision, so it is carried across and nothing is
padded. One swap per trial; after it the worker keeps scoring and refitting
every block, so every block's forecast errors are recorded, and offers nothing
further.

### The rule and the replay

The declared rule is that the swap happens in every adopting trial, no trial
terminates, and the trial's position and attitude RMSE over the intervals after
the swap are at or below its own values over the intervals before it, within the
same 5% plus 0.005 allowance the other tiers use. Interval `i` is scored at the
state it ends on, so the two segments partition the trial and neither counts the
other's rows; the frozen arm of the same repetition is segmented at the same
interval and reported beside them. A trial missing, duplicated, undeclared,
terminated, short of its intervals, reporting a refinement-worker error,
dropping a streamed block, recording a swap the gates and the declared offset do
not require, or carrying a whole-trial metric that is not a finite number fails
closed whatever the reference says.

The rule is not enforced for this first measurement. Nothing has measured this
row under this manifest, so this run measures it and the gate binds from the
next candidate, under the semantics the platform, control and evidence tiers
already use. Every number and every breach is measured, recorded and printed
exactly as it would be enforced; only `accepted` ignores them.

`verify` recognizes the tier from the digest of the manifest a run copied. It
reruns neither the plant, nor the solver, nor a refit -- rerunning any of them
would be a new run rather than a check of this one -- and everything else it
recomputes. It rechecks every artifact hash, including every block's evaluation
arrays and every scored revision; rebuilds the tracking reference from the saved
anchor and each trial's perturbed start from its declared seed; remeasures the
calibration's command excitation; recomputes both models' block forecasts from
the saved arrays and the saved revisions, the generic one through the same
independent NumPy recurrence the synthetic tier replays with and the structured
one through the library's own rollout; recomputes every block score and every
swap gate from those forecasts; recomputes the swap interval from the recorded
gates and the declared offset and checks the per-interval active revision
against it; and recomputes the whole-trial, before and after metrics from the
saved per-interval arrays.

### The measurement

`live-v2` was frozen and committed at 15:12:17Z; the run below started
collecting its calibration at 15:12:23Z, so the gate is older than every number
under it. Both arms are the two `control-v3` fits, on the same calibration,
whose measured command excitation is unchanged: the generic arm's fingerprint is
`9f7e7b17`, the same artifact `control-v3`'s own run produced, so this tier
starts from that measurement rather than beside it.

**It reproduces.** Two consecutive runs of this manifest produced byte-identical
`tracking.npz` in all four trials -- `a40f9b00`, `5a3e6b14`, `f99e79ea` and
`64c52e1e` -- byte-identical block evaluation arrays and forecasts, and
byte-identical scored revisions: 64 of 72 artifacts match to the byte, and the
eight that differ are the four `timing.npz` and the four worker event journals,
which carry nothing but host measurements. Every trial and block field the
decision reads -- 196 of them, including all four swap intervals, every block
score and gate, every tracking metric and every segment -- is identical.

| Repetition (seed) | Arm | Swap | Position RMSE before / after (m) | Attitude RMSE before / after (deg) | Whole trial (m / deg) | Terminated |
| --- | --- | --- | --- | --- | --- | --- |
| 0 (101) | adopting | interval 140, 7.00 s | 0.868 / **19.355** | 1.676 / **86.309** | 14.528 / 64.742 | none |
| 0 (101) | frozen | none | 0.868 / 1.373 | 1.676 / 0.956 | 1.179 / 1.320 | none |
| 1 (102) | adopting | interval 140, 7.00 s | 0.896 / **5.697** | 1.620 / **14.885** | 4.314 / 11.215 | none |
| 1 (102) | frozen | none | 0.896 / 1.358 | 1.620 / 0.941 | 1.179 / 1.283 | none |

Both frozen arms are segmented at the adopting arm's own swap interval, which is
what they are flown for, and both arms of a repetition agree to every printed
digit before the swap, because up to interval 140 they are the same flight.
Under `live-v1` they did not, and the difference was the clock.

Every held-out block, final-step forecast error as generic / structured, in m/s
of world velocity and rad/s of body rate. Each block is 26 rows neither model
had fitted on; a bold pair is one where the gate passed. The swap was decided on
block 1 of each adopting trial, and released at interval 100 plus the declared
40.

| Block (intervals) | 0-adopting | 0-frozen | 1-adopting | 1-frozen |
| --- | --- | --- | --- | --- |
| 0 (20-60) | 0.284/0.184 vs 0.161/0.085 | 0.284/0.184 vs 0.161/0.085 | 0.309/0.208 vs 0.161/0.085 | 0.309/0.208 vs 0.161/0.085 |
| 1 (60-100) | **0.032/0.024** vs 0.158/0.080 | **0.053/0.029** vs 0.158/0.080 | **0.046/0.023** vs 0.158/0.080 | **0.059/0.037** vs 0.158/0.080 |
| 2 (100-140) | **0.014/0.023** vs 0.153/0.077 | 0.060/0.078 vs 0.153/0.077 | **0.045/0.071** vs 0.153/0.077 | **0.039/0.067** vs 0.153/0.077 |
| 3 (140-180) | 0.172/0.192 vs 0.173/0.089 | **0.021/0.073** vs 0.154/0.077 | 0.253/0.286 vs 0.161/0.075 | 0.027/0.132 vs 0.154/0.077 |
| 4 (180-220) | 5.162/12.962 vs 1.394/0.594 | **0.021/0.023** vs 0.156/0.078 | 0.156/0.167 vs 0.124/0.078 | **0.017/0.016** vs 0.156/0.078 |
| 5 (220-260) | 3.774/1.282 vs 1.075/0.625 | **0.015/0.028** vs 0.160/0.081 | 0.386/0.328 vs 0.179/0.101 | **0.025/0.039** vs 0.160/0.081 |
| 6 (260-300) | **1.295/0.473** vs 1.918/0.598 | **0.042/0.058** vs 0.164/0.083 | 0.140/0.135 vs 0.196/0.106 | **0.035/0.043** vs 0.164/0.083 |

Blocks 0, 1 and 2 of an adopting arm are the same rows as its frozen twin's up
to the swap; from block 3 the adopting arm is scoring a flight the generic model
is already losing, which is why both models' errors grow there.

**The swap happens, and tracking after it is not no worse.** The rule's first
half holds: both adopting trials swapped, at the same interval in both, and no
trial terminated. Its second half fails on all four metrics, by a factor of 22
and 51 on the first repetition and 6.4 and 9.2 on the second, against an
allowance of 5% plus 0.005. Nothing structural breached: every trial completed
its 320 intervals with finite states, commands inside the declared box, no
dropped block and, now that no deadline is applied, no solver fallback at all.
The rule is not enforced for this measurement, so the run is accepted and
`rule_met` is false.

**The gate passed on evidence, and the evidence was real.** Each trial produced
seven candidate revisions -- the recipe's 1000 fit steps at 0.855 s to 0.896 s
of wall time against the declared 4.0 s, with block scoring another 0.52 s to
0.91 s -- with no budget overrun and no block dropped. On the block the swap was
decided on, intervals 60 to 100 of each trial, the candidate's final-step
forecast error was 0.032 m/s and 0.024 rad/s against the structured belief's
0.158 and 0.080 on the first repetition, and 0.046 and 0.023 against the same
0.158 and 0.080 on the second: three to five times better, on 26 rows it had
never fitted on, one refit after a block it lost on 0.284 against 0.161. The
frozen arms show it was not a fluke of one block. Left flying the structured
belief for the whole trial, their refits pass the same gate on five of seven
blocks each and hold forecast errors of 0.015 to 0.060 m/s across the trial
while the structured belief stays at 0.15 to 0.17. **The generic learner does
learn the regime it is flying in, quickly, from two seconds of it.**

**What it does not learn is command authority, and the gate cannot see that.**
Hold-current -- predicting that nothing changes -- scores 0.021 m/s and
0.0014 rad/s on that same block. The comparator the gate is written against is
seven times worse than claiming nothing there, and the candidate that beat it is
still 1.5 times worse than claiming nothing on velocity and seventeen times
worse on body rate. A held-out forecast comparison in a regime the structured
arm is holding almost still certifies a model that has learned that almost
nothing happens. The solver then spends commands on it: mean applied throttle
moves from 0.494 before the swap to 0.791 after it on the first repetition and
0.492 to 0.770 on the second, mean roll from +0.006 to +0.222 on the first
against a +0.35 bound, and altitude from 102.6 m at the swap to 64.9 m at 16 s.
The second repetition is the milder of the two and still loses the task: it ends
110.6 m high against a reference of about 100 m.

The swap changes the planning horizon too, from the structured belief's 0.80 s
to the recipe's own fitted 0.25 s, because each model plans exactly as far as
its own evidence supports. That is not the difference: replanned at 0.25 s on
`control-v2`'s reference the structured arm still tracked at 1.738 m, 35 times
better than the generic arm did.

That is Control attempt 2's finding arriving from the other side. There the
calibration excited the plant an order of magnitude harder than the task flew
it, so the fit's error floor exceeded the motion the controller had to resolve.
Here the learner is given exactly the regime being flown and closes that floor --
and the arm still cannot fly, because forecast error on a quiet block is not a
measurement of the command response, which is what
[`status.md`](status.md)'s ladder measured directly: the ceiling of a perfect
command response on this task is 9.04 m and the ceiling of claiming nothing is
1.22 m. Nothing in this tier's gate reads a command Jacobian, and the swap it
authorizes is the first thing that has ever needed one.

Host measurements inform and gate nothing, and now that none of them reaches a
command they are only that: each trial's interval work exceeded the 50 ms sample
interval on 0 to 2 of its 320 intervals, no solve exceeded the recorded 50 ms
threshold, and a whole trial costs 16 s to 26 s of wall time against 16 s of
simulated time. Two runs differ in every one of those numbers and in none of the
others.

This is a measurement of one live-refinement trial set on one simulated plant,
not hardware readiness, a real-time claim, or calibrated uncertainty.

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
evidence the fit never produced.

Uncertainty is the third thing bridged. `uncertainty_available` is true and the
seam's two robustness terms — the predicted spread charged at every tracking
stage, and the terminal stage's spread charged again — consume the learner's
envelope exactly as they consume a belief's forecast-error covariance.
`tangent_error_covariance` is the whole mapping, read once when the plan model
is built and carried through every kernel as a value, so a recalibrated
envelope of the same length costs no recompile. Each half-width becomes a
standard deviation by dividing by `Phi^-1(0.95)`, which reads a
distribution-free half-width as a Gaussian central interval at the envelope's
own nominal level; that is the first approximation. Velocity and body rates are
predicted channels and pass straight through. Position is integrated from the
velocity half-widths on the same grid, by the same trapezoidal rule the mean
uses, from a zero half-width at the observed origin: integrating half-widths
rather than variances treats one rollout's per-step velocity errors as moving
together, which is the conservative reading and the one a recursive forecast's
compounding error argues for. Attitude comes through the derivative of the same
polar projection the mean uses, which sends an entry perturbation `dR` to the
body-frame tangent `vee(skew(R.T @ dR))`; taking the nine entry errors to be
independent with one common scale — the root mean square of their nine
half-widths — makes each tangent axis's standard deviation that scale over
`sqrt(2)`, uncorrelated between axes and independent of the attitude, which is
what lets one covariance stand for the whole horizon as a belief's does. The
isotropy is the second approximation. Only the diagonal is filled: position and
velocity are genuinely correlated, but the seam reads the diagonal and nothing
else, so an off-diagonal term would be an unmeasured claim that changes no
number.

What the learner still does not have is a resolved parameter direction. There
is no covariance factor and therefore no plan-dependent `J C J.T`, so the
spread it charges is the same at every plan and shifts the objective without
moving its minimizer; `uncertainty_complete` stays false and a solve still runs
only under the seam's explicit no-evidence override, which the run records.
Validity utilization is still reported as zero because the learner declares no
support envelope, not because one was checked and found clear, so the seam's
validity-side robustness term has nothing to widen.

### The measurement

`control-v4` was frozen and committed before this run, which is its incumbent
measurement and the file `control-reference.json` is written from. Every number
below is `control-v3`'s to every printed digit, along with the pass statistics,
the command excitation and the generic fingerprint `9f7e7b17`: the runs that
measured `control-v3` were made on an idle host where every solve finished
inside its deadline, so every solved command was applied there too. Both arms
were fitted on the same three calibration recordings, whose measured command
excitation is 12.1%, 20.3% and 38.1% of declared range on the first, 14.2%,
13.7% and 37.5% on the second and 12.3%, 16.4% and 37.5% on the third, against
the declared 10%. Under `control-v2`'s setpoints the same three recordings moved
throttle 2.7%, 3.1% and 2.3%.

| Repetition (seed) | Position RMSE, generic / structured (m) | Attitude RMSE, generic / structured (deg) | 0.5 m pass fraction, generic / structured | Terminated | Solves over 50 ms, generic / structured |
| --- | --- | --- | --- | --- | --- |
| 0 (101) | 60.797 / 1.179 | 97.180 / 1.320 | 0.000 / 0.000 | none | 0 / 1 |
| 1 (102) | 49.308 / 1.179 | 86.514 / 1.283 | 0.007 / 0.000 | none | 0 / 3 |

The last column is reported and decides nothing; it differed between the two
runs that produced identical trajectories.

**The rule is not met.** All four trials completed their 320 intervals with
finite states, bounded commands and no solver fallback, so nothing structural
breached; the four rule breaches are the generic arm's two metrics on each
repetition. This is the incumbent measurement, so there was no reference to
compare against and the rule only reported. From `control-reference.json`
onward it gates every case the incumbent met, which is none of these four.

**Neither arm meets the page's criterion under this controller.** The structured
arm holds lateral position to 0.27 m but settles about 2 m high, so its altitude
error leaves the 0.5 m band on every scored sample. The page's own 3-for-3 pass
was measured with a different controller, developed against an oracle on a
separate seed; this tier flies the bounded shooting solver both arms share.
That is why the criterion is recorded and reported rather than gated.

**A calibration that moves throttle does not fix the generic arm.** With
throttle excitation four to six times `control-v2`'s, the generic arm still
loses the aircraft: mean applied throttle 0.137 against a trim of 0.437, roll
resting at +0.269 of a +0.35 bound and pitch at -0.236, and an altitude of 100 m
at the start, 79 m at 8 s and -27 m at 16 s. Its first two seconds are
comparable to the structured arm's — 0.14 m of lateral error and 0.00 m of
altitude error at 1 s — and it diverges after. Insufficient throttle excitation
was the arithmetic behind the unidentified throttle column; it is not by itself
the reason the arm cannot fly.

### What control-v2 measured, and what survives its calibration

Everything below was measured under `control-v2`: its cruise reference and, in
particular, its calibration, the one whose throttle excitation `control-v3`
replaced. The arithmetic of the mechanism is stated against those recordings and
no longer describes the ones the tier now collects. The outcome it was offered
to explain does survive them, which is the measurement above.

The mechanism was not the fit's forecast quality. On `control-v2`'s reserved
recording the learner's own 0.25 s forecast beat hold-current on every
channel group — world velocity 0.168 against 0.249 m/s, body rate 0.139 against
0.356 rad/s, rotation entries 0.0149 against 0.0212. What it did not have was a
usable command Jacobian in the one direction the controller reaches for first.
Comparing each arm's final-step response to a +0.05 command step against the
plant's own, averaged over three held-out origins:

| Command | Direction cosine, generic / structured | Magnitude ratio, generic / structured |
| --- | --- | --- |
| throttle | -0.056 / 0.968 | 26.9 / 1.39 |
| roll | 0.594 / 0.798 | 0.58 / 0.60 |
| pitch | 0.969 / 0.951 | 1.12 / 1.03 |

Pitch was as good as the structured model's and roll comparable, but throttle
was uncorrelated with the plant's response and 27 times too large. `control-v2`'s
calibration explained the arithmetic: across its three recordings throttle moved
with a standard deviation of 0.0225 to 0.0312 over a declared range of 1.0,
while roll moved 0.1117 to 0.1727 and pitch 0.2598 to 0.2671 over ranges of 0.7.
Because the affine start standardizes each command by that sample standard
deviation, the recipe's ridge charged the throttle level column 0.0127 per full
declared-range move where it charged pitch 2.126, 168 times weaker, and the
throttle difference columns 2.0e-4 and 6.6e-5. Open-loop scoring never charged
for the result either, because throttle barely moved in the evaluation data. An
optimizer charged for it immediately: it drove throttle to a bound, on 79% of
that trial's intervals. `control-v3`'s calibration moves throttle 12.1% to 14.2%
of the same range, which is four to six times as much and removes that
arithmetic; the generic arm still loses the aircraft.

**The horizon was not the difference.** Replanned at the generic arm's own
0.25 s horizon on `control-v2`'s reference, the structured arm still tracked, at
1.738 m and 1.371 degrees: 44% and 34% worse than at 0.80 s, and 35 and 89 times
better than the generic arm. Its
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

This is a measurement of one tracking trial set on one simulated plant, not
hardware readiness, a real-time claim, or calibrated uncertainty.
