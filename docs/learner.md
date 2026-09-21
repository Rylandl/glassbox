# Learner contract

Glassbox fits one shared rigid-body formulation separately to each configuration.
It combines known gravity, coordinate transforms and rotation integration with
learned effective accelerations and latent command response. No vehicle-family
selector, physical parameter guesses, actuator assignment or tuning options are
required. Input-channel count need not equal actuator count.

This is a model of observed rigid-body motion. Coverage across articulated,
flexible or arbitrary nonmechanical systems has not been demonstrated.

## Recordings

Construct a `SequenceCollection` of `SequenceSegment` objects. Each segment holds
`states` with shape `[N, 15]`, `inputs` with shape `[N-1, U]`, a positive sample
interval `dt_s`, and its whole-recording ID, segment ID and source `start_row`.
`inputs[t]` is the issued command during the transition from `states[t]` to
`states[t+1]`. Arrays must be finite and uniformly sampled within each segment.
Different segments never share a prediction window.

Import `STATE_CHANNELS` from `glassbox` for the required ordered channel names:

| Columns | Meaning | Units and frame |
| --- | --- | --- |
| 0–2 | Linear velocity | m/s, world north-west-up |
| 3–5 | Angular velocity | rad/s, body forward-left-up |
| 6–14 | Rotation matrix, flattened by rows | body forward-left-up to world north-west-up |

Rotations must be proper orthonormal matrices. Position is not part of the learned
state; a consumer can integrate the predicted world velocities when needed.
Specify ordered `input_channels` describing actual issued command meanings and
units, plus an opaque `configuration_id`. All recordings in a fit or update must
share that identity, channel order and sample interval. For offline revisions, a known change to physical configuration calls for a
separate fit and identity. The streaming procedure below can instead be tested
on an unannounced change within an ongoing acquisition; that does not establish
that arbitrary changes can be identified or safely controlled.

Whole-recording IDs identify independent data acquisitions. Do not rename or split
one acquisition to suggest independence. Use segment boundaries for gaps and
invalid observations; `segments_from_mask` helps select contiguous valid runs.
Clock alignment, coordinate conversion and telemetry decoding belong to the
application. Do not substitute measured actuator states for issued commands.

`glassbox.io.recordings.save_recordings` and `load_recordings` store these facts and
arrays in a fingerprinted NPZ archive. `concatenate_recordings` combines compatible
archives without merging or renaming their boundaries.

## Fit and predict

`fit(recordings) -> LearnedDynamics` uses one deterministic, bounded recipe.
It reserves development recordings, selects a checkpoint using them and retains
bounded training/development windows for future updates. Supply at least two
independent whole recordings with complete history and forecast windows. More
varied observed conditions and command excitation determine what the fit can
identify; the API does not create missing information.

The recipe uses approximately 500 ms of observed history, a 250 ms fitting
horizon and a 1.2 s prediction limit. Read the integer `history_steps`,
`horizon_steps` and `prediction_limit_steps` from each model; they account for its
sample interval.

```python
prediction = model.predict(past_states, past_inputs, future_inputs)
```

For history length `P`, horizon `H` and command count `U`, pass:

- `past_states`: `[P+1, 15]` observed states, including the current state.
- `past_inputs`: `[P, U]` issued commands connecting those states.
- `future_inputs`: `[H, U]` commands starting at the current state.

`P` must be at least `history_steps`; excess history is trimmed. `H` must be from
1 through `prediction_limit_steps`. The returned JAX array is `[H, 15]` and
excludes the current state. Add the same leading batch dimension to all three
inputs for batched prediction. Padding missing history is not valid.

Predictions are differentiable through JAX. They use the caller's JAX inference
precision. Fitting uses float64. Mathematical derivatives and finite optimizer
callbacks do not establish physical command-response fidelity. Means beyond
`horizon_steps` are recursive extrapolations, without calibrated error envelopes.

## Evidence and persistence

`model.contract` describes signals and timing; `model.report` contains fitting,
provenance and evidence limits. `model.fingerprint()` identifies the revision.
`model.save(path)` and `LearnedDynamics.load(path)` persist a self-contained model
and reject altered archive contents.

When a revision has measured calibration evidence, `model.envelope(H)` returns
per-channel absolute-error half widths for each of its first `H` steps. Calibration
uses development windows that also select the checkpoint; coverage on untouched
recordings and shifts remains unqualified. These widths are not a safety bound.
No envelope is extrapolated beyond the fitting horizon.

The adopted saved revisions retain their fitting caches and usable predictions,
but do not carry newly measured calibration for the supported formulation.
`envelope` raises an explicit error when calibration is unavailable. Do not treat
absent uncertainty as zero uncertainty.

## Update and evaluate

`revision = model.update(new_recordings)` returns a new immutable revision using
fresh recordings plus retained bounded training windows. It preserves the
original development split and rejects known recording IDs or duplicate content.
The old model remains unchanged. This is an offline revision mechanism; improvement
and safe replacement in a running controller need separate evaluation.

```python
from glassbox.workflows import evaluate

scores = evaluate(model, held_out_recordings)
```

Evaluation never fits or updates. It scores every complete context/horizon window
within each segment, returning per-recording and aggregate RMSE in each channel's
units, hold-current-state RMSE and measured envelope coverage when available.
Coverage is absent when the model has no calibration. Known training/development
recording IDs and exact contents are rejected; the caller remains responsible
for independence when data have been transformed or resegmented.

For CLI equivalents use `glassbox fit --help` and `glassbox evaluate --help`.
Current physical results and their limits are documented in [status](status.md).

## Streaming identification

`OnlineFit(prefix)` is a stateful fitting session for an ongoing, contiguous
recording. It uses the same shared-physics dynamics engine as offline revisions.
It does not require a previously fitted model or a vehicle-family label.

Supply a `SequenceCollection` containing one segment with the normal channel and
configuration metadata. Startup needs 0.5 seconds of real observed context plus
0.25 seconds of completed transitions (rounded to the sampling grid, with at
least three training transitions). The fitter uses the last such prefix when
more data are supplied. It cannot predict from observations that have not arrived.

```python
from glassbox import OnlineFit

session = OnlineFit(prefix)
prediction = session.predict(past_states, past_inputs, next_command[None])
# Only after receiving the next observation:
session.observe(session.cursor, next_command, next_observation)
session.save("stream-session.npz")
session = OnlineFit.load("stream-session.npz")
```

The cursor is the absolute source row of the next issued command, including the
segment's `start_row`. An observation consumes exactly that transition; duplicate,
backward and skipped indices are rejected. Start a new session across a gap.
Prediction uses the same observed-history and issued-command meaning as above.
Its numerical kernel is already compiled with model parameters passed as data.
Wrapping a mutable session in an outer `jax.jit` closure captures its parameters
at trace time; controllers that compile their own objective must pass changing
model parameters as traced arguments, or deliberately use a fixed snapshot.
The session is mutable; it does not rewrite previously saved sessions or model
snapshots. Saves include the optimizer state, bounded replay data and cursor
needed to continue deterministically.

Startup initializes the existing model from one-step ridge windows. Thereafter
each observation permits one damped Gauss-Newton proposal on 50 ms recursive
prediction windows. Four conjugate-gradient iterations use matrix-free curvature
products; acceptance checks the complete bounded startup and recent replay
caches, with equal weight to each role. A fixed prediction-change bound and an
actual-versus-predicted improvement check control the step. Normalization stays
fixed. Rejected proposals retain the previous parameters and increase damping.
There is no controller, actuator-telemetry input, simulator coefficient access,
platform dispatch or user-selected learning budget.

This procedure has no development split or calibrated error envelope. Training
loss is not an independent accuracy measure. The active
[online-fitting protocol](harness/online-fit-v2.json) measures predictions before
assimilating their targets, against an identical session frozen after startup.
It separately measures update latency; a bounded proposal count alone does not
establish real-time fitting. Initial evidence and remaining limits belong in
[status](status.md), not in the API's guarantees.
