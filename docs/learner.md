# Learner contract

Glassbox fits one rigid-body dynamics equation to each configuration. Gravity,
body/world transforms and rotation integration are analytic. The force, torque,
inertia and hidden command response are inferred from that configuration's
recordings. No pretrained weights, vehicle family, mixer, actuator layout or
consumer tuning are required. Input count comes from the data and need not equal
actuator count. This formulation has evidence for rigid-body flight; it is not a
claim about arbitrary articulated systems.

## Recordings

Supply a `SequenceCollection` of `SequenceSegment` objects. Each segment has
`states` shaped `[N+1, 15]`, `inputs` shaped `[N, U]` and a positive, uniform
`dt_s`. `inputs[t]` is the **issued** command during the transition from
`states[t]` to `states[t+1]`. Segment boundaries mark resets or gaps; command
history is never carried across them. Each segment needs at least five completed
transitions to fit.

Use `STATE_CHANNELS` from `glassbox` for the required state channel names:

| Columns | Signal | Units and frame |
| --- | --- | --- |
| 0–2 | Linear velocity | m/s, world north-west-up |
| 3–5 | Angular velocity | rad/s, body forward-left-up |
| 6–14 | Row-flattened rotation matrix | Body forward-left-up to world north-west-up |

Observed rotations must be proper orthonormal matrices. Position is outside the
learned state; a consumer can integrate predicted velocity. Supply ordered
`input_channels` with interpretable names and units and an opaque
`configuration_id`. The names establish identity; they do not choose equations.
All segments in a fit or update must have the same configuration, channel order
and sample interval. Recordings from a changed physical configuration need a
fresh fit and identity unless the change is deliberately being studied within
one streaming episode.

Whole-recording IDs identify independent acquisitions. Use segment IDs and
`start_row` to preserve gaps within a recording; do not rename one acquisition
to claim an independent test. `segments_from_mask` selects contiguous valid
runs. Coordinate conversion, telemetry decoding and clock alignment belong to
the caller. `glassbox.io.recordings` saves and loads fingerprinted archives.

## Fit, predict and update

```python
from glassbox import LearnedDynamics, fit

model = fit(recordings)
prediction = model.predict(past_states, past_inputs, future_inputs)
model.save("model.npz")
revision = LearnedDynamics.load("model.npz").update(new_recordings)
```

`fit` uses **all** supplied segments. It does not silently reserve one for
development or claim calibration from training data. Varied conditions and
command excitation determine what is identifiable. `update` refits with the
original and new recordings, returns an immutable revision, and rejects reused
recording identity or content. The original revision remains unchanged.

For command count `U`, pass aligned arrays:

- `past_states`: `[P+1, 15]` observations since the start of this segment.
- `past_inputs`: `[P, U]` issued commands connecting those observations.
- `future_inputs`: `[H, U]` candidate commands beginning at the current state.

At least one past command is required. The command history must begin at the
episode or segment start because it reconstructs hidden applied-command state;
arbitrary recent windows are not equivalent. `H` is 1 through
`model.prediction_limit_steps` (1.2 s on the recording grid). The returned JAX
array is `[H, 15]`, excluding the current state. Add the same leading batch
dimension to all three arrays for batched prediction. The nominal evaluation
horizon is `model.horizon_steps` (250 ms). Longer outputs are recursive
extrapolations.

Predictions have JAX command derivatives. Fitting uses float64; prediction uses
the caller's JAX precision. A finite derivative is a software contract, not
proof of physical command-response accuracy.

The fitted command state has one shared nonlinear target and rising/falling
relaxation law across input channels. Its magnitude and lag come from the
current episode. A single force readout uses constant, applied-command,
command-squared, body-velocity and velocity-times-speed terms. The torque
equation uses a fitted inertia tensor, command effect, angular damping and an
actuator angular-momentum term. The rigid-body cross products couple axes;
coefficients may fit to zero where coupling is absent. A weak, data-independent
zero-force prior prevents an underexcited short prefix from explaining motion
with cancelling enormous intercept and slope terms. There is no class branch or
actuator metadata prior.

`model.contract` reports signals and timing. `model.report` records fit
provenance, residuals, elapsed fit time and evidence limits.
`model.fingerprint()` identifies the complete revision. Save/load retains the
fitted equation and source recordings so later updates can use the same data.
The current recipe is `causal-actuator-v1`; earlier archive formats require a
fresh fit or their original checkout. No calibrated error envelope is provided:
`model.envelope()` raises explicitly. Missing coverage is not zero uncertainty.

## Held-out evaluation

`glassbox.workflows.evaluate(model, held_out_recordings)` scores every complete
causal-prefix/250 ms window within each segment without fitting or updating.
It reports per-channel RMSE, hold-current-state RMSE and counts per recording.
Known fitting IDs and exact contents are rejected. The caller remains
responsible for independence after transformation or resegmentation. Coverage
is absent until an independently tested calibration method exists. The CLI
provides `glassbox fit` and `glassbox evaluate` for the same workflow.

## Streaming identification

`OnlineFit(prefix)` starts from one contiguous current-episode segment with at
least five completed transitions. Initialization itself takes fit time. After
each new observation, `session.observe(session.cursor, issued_command,
next_state)` collects the completed transition. A background worker refits the
whole causal prefix and publishes immutable revisions when ready. Fitting may
lag observations; `session.published_cursor` names the last transition seen by
the currently published model. `session.predict(...)` uses that revision and
the full issued-command history through `session.cursor`. No per-observation
update deadline is assumed.

`session.save(path)` and `OnlineFit.load(path)` persist the collected observations,
published model and cursor. A pending worker computation is not serialized; a
resumed session starts new background work after the next observation. Use
`session.close()` to finish the worker. A controller compiling a JAX objective
should deliberately capture an immutable published model rather than close over
a mutable session. Current direct prediction and publication measurements are
reported in [status](status.md); neither alone establishes closed-loop recovery.
