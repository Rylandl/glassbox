# The generic learner

`glassbox.fit` learns differentiable dynamics from uniformly sampled
observations and commands. It returns a `LearnedDynamics` with `predict` and
`update`. The implementation lives in `glassbox.learner`; recording types live
in `glassbox.recordings`. There is one maintained recipe and no caller-selected
model family, optimizer, seed or representation.

The generic approach is the adopted baseline. Its current strengths and
deficits are recorded in [status](status.md); the [charter](charter.md)
distinguishes adoption from completing the project.

## Consumer workflow

```python
from glassbox import LearnedDynamics, SequenceCollection, SequenceSegment, fit

# rows contains distinct recordings supplied by your application.
recordings = SequenceCollection(
    tuple(
        SequenceSegment(name, "whole", states, inputs, dt_s=0.05)
        for name, states, inputs in rows
    ),
    configuration_id="system-revision-c",
    state_channels=("signal_a [m/s,world]", "signal_b [rad/s,body]"),
    input_channels=("command [normalized,applied]",),
)

model = fit(recordings)
future = model.predict(past_states, past_inputs, future_inputs)
half_width = model.envelope(len(future))
model.save("model.npz")
revision = LearnedDynamics.load("model.npz").update(more_recordings)
```

The [onboarding walkthrough](guides/platform-onboarding.md) supplies a runnable
example, including independent evaluation and generic recording files.

## Recording contract

A `SequenceSegment` describes one contiguous block:

| Field | Meaning |
| --- | --- |
| `recording_id` | Identity of the source recording, shared by its retained segments. |
| `segment_id` | Unique segment name within that recording. |
| `states` | Finite `(N + 1, D)` observed-signal array. |
| `inputs` | Finite `(N, U)` applied-command array; row `k` drives the transition from state row `k` to `k + 1`. |
| `dt_s` | Positive uniform sample interval in seconds. |
| `start_row` | Original recording row offset, default zero. |
| `excitation` | Optional injected command component, aligned with `inputs`. |

The observations need not be a complete physical state. Memory can represent
some hidden dynamics, but sufficient observability and excitation remain
properties of the data and system, not guarantees of this interface.

A `SequenceCollection` requires a consistent sample interval and channel
counts. Fitting also requires a nonempty `configuration_id` and unique ordered
`state_channels` and `input_channels` describing units and frames. Use the same
contract for prediction, evaluation and updates. One fitted model describes one
configuration; sharing a recipe does not mean mixing unrelated systems in one
fit.

The canonical telemetry adapter `glassbox.io.recordings.from_trajectories`
encodes command names, units, frames, semantics and roles in its input-channel
identities. Converted recordings must match a loaded model's stored contract.
Earlier generic artifacts keep their original channel identities; use their
declared contract, without relabeling different signals to force acceptance.

Use `segments_from_mask(recording_id, states, inputs, valid, dt_s=...)` to
retain contiguous valid runs. The boolean mask describes state and outgoing
command validity. The helper preserves source offsets and omits runs shorter
than two observations. Windows never cross a gap or recording boundary, and
the learner does not pad missing history or infer clock alignment.

`fit` needs at least two distinct recordings, with enough complete windows
for both automatic roles. It orders recording identities deterministically,
reserves approximately a quarter for development (at least one), and samples
windows round-robin across recordings within each role. The report names the
actual roles. Exact duplicate content cannot establish an independent recording,
even under a different name.

## Prediction and memory

`predict(past_states, past_inputs, future_inputs)` accepts either one query or
a batch:

| Array | One query | Batch |
| --- | --- | --- |
| Past observations | `(P + 1, D)` | `(B, P + 1, D)` |
| Past commands | `(P, U)` | `(B, P, U)` |
| Future commands | `(H, U)` | `(B, H, U)` |
| Returned means | `(H, D)` | `(B, H, D)` |

Supply at least `model.history_steps` past transitions and between one and
`model.horizon_steps` future commands. A longer past is truncated to the most
recent context. A shorter past or longer forecast is rejected. The returned
means exclude the current observed state and use the declared observation
coordinates. Forecasts are conditional on the commands supplied, not
predictions of what a controller will choose.

The model's memory starts at rest at the first consumed observation, advances
through that context, and continues recursively during the forecast. Earlier
history is not implied. At 50 ms sampling the current recipe consumes ten
transitions and predicts at most five. Always read the fitted properties
instead of hard-coding those lengths.

Forecasts support JAX batching, JIT and differentiation with respect to observed
history, past commands and future commands. Fitting and envelope calibration run
in a local float64 scope. Prediction follows the caller's ambient JAX precision;
it does not change global configuration. Inputs must be finite and representable
in that precision. Large coordinate offsets can erase small variations in
float32, so measured numerical support matters when choosing application units.
The output coordinates are Euclidean: the learner does not enforce rotation
manifolds or other physical constraints.

## Using a model in another application

Consumers need no Glassbox controller. Load a saved `LearnedDynamics`, inspect
its `contract`, `history_steps` and `horizon_steps`, and query `predict` with
the declared observation history and candidate commands. Batched queries serve
candidate comparison; JAX automatic differentiation can form command sensitivities
for an optimizer or local analysis. Correctly computing a derivative of the
learned function does not establish physical response accuracy; that requires
separate held-out response evidence.

This artifact represents dynamics in the supplied observation coordinates,
conditioned on the required history. It is not necessarily a Markov model of
the latest observation alone, a continuous-time differential equation, or a set
of identified physical parameters. Consumers own any coordinate conversion,
state estimator, objective, constraints and scheduling. Those choices do not
select a different fitting recipe.

The current artifact/runtime contract is Python/JAX with the supported recipe
format. `save` and `load` preserve model identity and evidence; they do not
currently promise an export for every external solver or deployment runtime.

## Immutable updates and saved revisions

`revision = model.update(new_recordings)` refits the same recipe using fresh
training windows and its bounded retained cache. The original development cache
continues to select the checkpoint and calibrate the new envelope. The original
model stays unchanged, and `revision.report["previous_revision"]` records its
fingerprint.

Updates require the exact fitted signal/configuration/time contract and whole
new recording identities. Previously seen identities or exact content are
rejected. Distinct hashes cannot prove that differently segmented data are
independent; preserving source identities remains the caller's responsibility.

An update may worsen predictions. Use separate untouched recordings to compare
both revisions on identical forecast queries before claiming improvement.
Repeatedly selecting revisions on the same evaluation set makes that set part
of model selection.

`model.save(path)` stores the recipe, parameters, normalizers, envelope,
recording ledger and update caches together. `LearnedDynamics.load(path)`
checks artifact integrity and the supported recipe/format. The loader supports
`glassbox-default-recipe-v4` archives. Earlier recipes are rejected and must be
replayed with their pinned historical source; loading them does not silently
convert their model or evidence.

## The envelope

`model.envelope(H)` returns a copy of the measured per-step, per-channel
half-widths in physical units, aligned with `predict`. Omit `H` for the whole
fitted horizon. Unsupported horizons are rejected.

For each horizon step and channel, the nominal 90% half-width is the
`ceil((n + 1) * 0.9)`-ranked absolute error on the development windows, capped
at their largest error when that rank exceeds the sample count. Those windows
also select the training checkpoint, so this is not an independent calibration
set. Coverage on other recordings must be measured.

`glassbox.workflows.forecast.evaluate(model, recordings)` reports per-channel and
per-horizon RMSE, hold-current RMSE and measured envelope coverage, both pooled
over complete windows and separately by recording. It checks the fitted
contract and rejects previously fitted identities or content. It changes no
model and makes no acceptance decision. A nominal interval or small average
forecast error does not establish downstream task success.

## Declared excitation and diagnostics

A caller may declare the exogenous component it injected into each applied
command as `SequenceSegment.excitation`. Every segment in a collection must
declare it, or none may. Masking preserves its alignment with the commands.
The current recipe records declaration and measured excitation fractions in
its report but does not use the excitation array in its fitting objective.

`model.diagnose(recordings)` is a separate read-only report on input
predictability and extra-history error evidence. It requires at least three
distinct contract-matching recordings and does not qualify the model for
control.

## The recipe

The recipe ID is `generic-memory-v4-prototype`. These are
implementation constants, not consumer options.

| Constant | Value |
| --- | --- |
| Model | Affine, observation-quadratic and observation-command terms plus tanh residual and recurrent memory |
| Hidden width / memory size | 32 / 8 |
| Explicit delay / consumed context / forecast | 0.1 s / 0.5 s / 0.25 s |
| Training / development window budgets | 1,536 / 256 |
| Safeguarded Adam steps / gradient batch / learning rate | 1,000 / complete training cache / 0.002 |
| Ridge fraction / fixed seed | 0.01 / 0 |
| Checkpoint interval / hold-scale floor | 100 / 0.01 |

Initialization fits an affine model, then solves jointly for affine and
quadratic coefficients with regularization toward that affine solution. The
neural residual and memory readout begin at zero. Training balances channels using
fixed weights derived from the initial training errors. Each Adam proposal uses
the complete training cache and is accepted only when a bounded backtracking
search finds a finite decrease in that training objective. Development rollouts
select the saved checkpoint. The larger cache adds computation and does not, by
itself, add independent recordings.

Durations round to the nearest sample, with at least one delay and forecast
step and at least one memory step beyond the explicit delay in the consumed
context. At 500 ms sampling, this means two context steps, one delay step and
one forecast step. The fitted properties expose the resulting lengths.

## Harness and control consumers

The frozen harness separates synthetic capability, platform forecasts,
control, live improvement and error-envelope coverage. Its historical
acceptance flags retain their original meaning; current adoption and
promotion policy is stated in the [charter](charter.md).

```bash
uv run python -m glassbox.experimental.harness run \
  --manifest docs/harness/v1.json --output /tmp/glassbox-synthetic
uv run python -m glassbox.experimental.harness verify /tmp/glassbox-synthetic
```

The platform, control and live plans are
[platform-v4](harness/platform-v4.json),
[control-v5](harness/control-v5.json) and [live-v3](harness/live-v3.json).
The [evidence plan](harness/evidence-v2.json) measures coverage.
[Status](status.md) identifies saved evidence and the source
versions needed for replay. Synthetic passes are regression evidence, not
readiness for a new system.

The experimental `learned_plan` adapter supplies the generic learner to the
rigid-body NMPC seam. It converts world velocities, body rates and rotation
entries into the controller's state coordinates and carries observed history.
Its uncertainty cost currently uses a fixed forecast-error offset: that cannot
rank command plans, but can change finite-iteration stopping. The qualified
anticipatory oracle and isolated research learners in [status](status.md) now
meet the declared task. Those results do not automatically qualify the public
model or other controller integrations. Model accuracy remains a directly
measured product property, and each consumer has its own task requirements.
