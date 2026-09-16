# Platform onboarding from memory

Run one workflow for both supported dynamics families:

```bash
uv run python examples/platform_onboarding.py
```

The [example source](../../examples/platform_onboarding.py) uses the public
fit, evaluation, belief rollout, and update interfaces. Only the telemetry
generator changes between multirotor and fixed-wing runs. The example assumes
the package is installed as described in the [README](../../README.md).

The fixtures are synthetic and use simulator-truth rigid-body states. Their
generators use stabilizing feedback and known plant parameters to collect
telemetry; the fitter receives the resulting states and commanded inputs.
The example checks interface composition. It is not a measurement of hardware
onboarding time, flight readiness, or performance on an unseen model family.

## Data and fit

`fit(sources, spec)` accepts a sequence of `Trajectory` objects, file paths,
or both. All sources use the same dataset compatibility checks, holdout rules,
training-window selection, optimization, and evidence calculation. A raw
state/control array needs a `Trajectory` with its signal contract before fitting.
See [scope](../scope.md#inputs-and-models) for those semantics.

The example makes six distinct synthetic recordings per family:

| Recording | Role |
| --- | --- |
| 0 and 1 | Fit parameters and parameter information |
| 2 | Calibrate empirical forecast error and innovation noise |
| 3 | Evaluate the fitted model and inspect a short prediction |
| 4 | Supply fresh telemetry to `absorb` |
| 5 | Compare the original and updated models on the same fresh recording |

`Holdout.by_group()` reserves the final source group among recordings 0 to 2.
The fit report still calls that section `validation_flights` for compatibility;
its `validation_role` explicitly says `forecast_error_calibration`. Those
measurements are part of constructing the belief. Recordings 3 and 5 are not
used for fitting, tuning, or selecting the update.

No calibration NPZ files are needed. Replace the generators with your adapter's
canonical trajectories to use the same calls. Keep a consistent configuration
identity, sample rate, and signal contract within one fit. Family differences
are carried by that contract, rather than an airframe-name switch in the fitting
or prediction workflow.

## Names and evidence identity

Reports preserve supplied file paths. An in-memory trajectory uses its
`provenance["path"]` as a display label when present, otherwise `trajectory_N`.
Colliding in-memory labels are disambiguated so per-flight report entries are
not overwritten. These labels are display names, not required filesystem paths.
Saving an in-memory fit report does not save its telemetry. Evaluate the belief
with explicit trajectories or persist the data separately; the file-oriented
`evaluate --fit-reports` workflow requires resolvable source files.

Each fit training/calibration summary includes `content_sha256`. The saved
belief's `provenance["data_identity"]` retains the algorithm and the identities
assigned to each role. The `trajectory_sha256_v1` digest includes canonical
time, states, controls, exogenous inputs, observations, command prefix, and
signal contract. It excludes labels, provenance, and storage details. Saving
equivalent canonical content to NPZ preserves the identity.

Identical content can therefore be recognized even when renamed. This delivery
does not automatically detect all overlap or enforce split independence.
Different hashes can describe overlapping segments, and
`evaluate(..., independent_holdout=True)` remains the caller's declaration.
Source grouping and reserved-data discipline are still required.

## Predict and inspect

The example saves and reloads the belief, evaluates it, and calls
`belief.rollout` on a short command sequence from recording 3. It supplies
the real preceding commands using `trajectory_segment`, along with any
declared exogenous inputs. Actuator state is reconstructed from commands;
the example does not read the simulator's latent actuator state.

History-based initialization remains an estimate with an initial-history
assumption. For external actuator-state estimates or measured-actuation models,
see [dynamics beliefs](../concepts/dynamics-beliefs.md) and the explicit command
mapping contract. The loaded model must have a command map before a
command-based rollout is actionable.

The summary keeps parameter rank/completeness, error-data availability,
measured-horizon support, and operating-range utilization separate. A finite
prediction is not a claim of sufficient accuracy, complete information, or
admission by a controller.

## Update and evaluate again

`belief.absorb(recording_4)` returns an updated belief and an `UpdateResult`.
The example records whether the data were absorbed, any refusal reason, and
parameter movement since the forecast-error measurements. It saves the revised
belief separately and compares it with the original on recording 5.

An absorbed update need not improve held-out prediction. The example records
either outcome. An update also retains the earlier forecast-error measurements;
evaluating the revised model writes a new evaluation report but does not
recalibrate that envelope in the belief. See [updates](../concepts/dynamics-beliefs.md#updates).

## Outputs and repeatability

Each family writes a directory under `artifacts/onboarding/` containing:

- `fit.json` and `belief.json`: the fit report and serialized model/evidence.
- `evaluation.json` and `prediction.npz`: independent prediction metrics and
  the observed/predicted trace with its commands and history.
- `updated-belief.json`, `updated-evaluation.json`, and
  `prior-on-updated-test.json`: the revised artifact and paired evaluation.
- `summary.json`: data roles, content identities, assumptions, and outcomes.

Use `--family multirotor` or `--family fixedwing` for one family, `--output`
for a separate artifact directory, and `--fit-steps` to change the optimizer
budget. The default is a small demonstration budget. It does not establish
that this amount of calibration or optimization is sufficient for another
platform. Reusing an output directory overwrites that example's artifacts.

## Streaming refinement

The experimental [replay example](../../examples/streaming_refinement.py) uses
the same model operations in a session coordinator:

```bash
uv run python examples/streaming_refinement.py --adopt-between-recordings
```

This fits both families from synthetic calibration recordings 0 to 2, saves
fresh recordings from seeds 6 and 7, and reloads them in contiguous blocks.
It uses the same simulator-truth and stabilizing-feedback assumptions as the
first walkthrough. The default block contains 20 control intervals. Only the
candidate learns; the active revision stays fixed throughout each recording.

`ModelRefiner` lives in `glassbox.workflows.refinement`, outside the stable root
API. The application supplies a belief and canonical telemetry. This call
sequence is illustrative; the executable example above supplies those inputs:

```text
refiner = ModelRefiner(belief)
pinned = refiner.active  # A consumer can retain pinned.belief for one solve.
result = refiner.observe(block, recording_id="flight-001", start_interval=0)

# After inspecting scores from subsequent blocks, an application may decide:
refiner.adopt(
    result.candidate_score.revision.revision_id,
    expected_active_revision=result.active_score.revision.revision_id,
    reason="application decision with its own requirements",
)
```

Each `observe` forecasts the entire block from its first observed state using
the active and incoming candidate snapshots. The four RMSE metrics exclude
that shared initial state and use geodesic attitude error. The forecasts use
commands and exogenous inputs observed during the block, so this is conditional
model replay rather than a prospective prediction of a closed-loop flight.
Nonfinite state forecasts have no RMSE and remain visible in the report.

Only after both forecasts are scored does `absorb` process the block. The
`candidate_score.revision` is the evaluated model. `candidate_after` includes
the new information and, when absorption succeeds, has a new identity even
if its mean parameters did not move. It has not yet been scored on subsequent
data. Adoption of that unscored revision is rejected. A previously scored
revision remains selectable after learning advances, including for rollback.
The expected active identity rejects a decision based on a superseded active
model; it is not a thread synchronization primitive or a performance gate.

The optional example flag exercises adoption at recording boundaries without
a performance threshold. It selects the last scored candidate even if some
prediction metrics got worse. Omit the flag to keep the original active model
throughout. Neither path certifies a model for a controller or a vehicle.

### Recording and model contracts

Supply a stable `recording_id` and an absolute control-interval index within
that recording. Each recording starts at interval zero, and blocks must arrive
in order without gaps or overlaps. Two adjacent blocks share one endpoint
state but no control intervals. Rebasing a segment's timestamps does not
reset its interval index. `trajectory_segment` preserves the full preceding
command history; the coordinator verifies it and the shared endpoint.
History initializes actuator response and is not absorbed again as fresh data.
At recording start, missing history retains the model's first-command
initialization assumption; it does not represent measured actuator state.

The session requires the same prediction contract, observation source, sample
period and vehicle configuration. A missing configuration ID remains missing;
the coordinator cannot infer platform identity. Direct command models are
supported. Measured-actuation models and custom actuation maps are rejected
because their training inputs may differ from their runtime command coordinates.

Exact content duplicates are rejected across names, including exact matches
to fit-reserved source hashes when available. This does not discover every
overlap under renamed recordings or different segmentation. Stable source
identity and fresh-data declarations remain the application's responsibility.
Later blocks in the same flight are temporally fresh but correlated, and are
not an independent-flight holdout for final performance claims.

An update refusal is reported and consumes the scored block. An exception
before completion leaves the candidate, cursors and ledger unchanged, allowing
a corrected retry. All session methods have one owning worker. The replay
defaults retain every revision, forecast and preceding command. Optional
`history_steps`, `retained_blocks` and `max_recordings` cap these independently;
set all three for bounded session retention. Only retained or explicitly held
evaluated revisions remain available for adoption. Exact-content duplicate
checks then cover retained blocks and fit-reserved hashes, while recording
cursors still reject old intervals. There is no durable exactly-once restart
protocol. Do not mutate nested belief metadata after handing it to the session.

### Replay artifacts and existing recordings

Each family writes `artifacts/refinement/<family>/summary.json`, separate
`revision-*.json` beliefs and `block-*.npz` observed/predicted traces. The summary
maps session revision IDs to artifacts, records source hashes and interval
ranges, and retains update diagnostics and adoption decisions. Scores report
parameter rank, horizon support, operating-range utilization, update count and
parameter movement since forecast-error measurement separately.

The following replays the multirotor artifacts produced above through the
file-input path, keeping the active model fixed:

```bash
uv run python examples/streaming_refinement.py \
  --belief artifacts/refinement/multirotor/initial-belief.json \
  --recordings artifacts/refinement/multirotor/recording-seed-6.npz \
    artifacts/refinement/multirotor/recording-seed-7.npz \
  --output artifacts/refinement/replay-existing
```

Use the same flags with your own compatible canonical recordings. The outputs
are session reports and artifacts, not resumable checkpoints. Output paths are
overwritten on repeat runs. A new replay removes the previous summary before
writing traces, so a failed run leaves partial artifacts without a completed
summary. The original operating and forecast-error envelopes
remain unchanged as information accumulates. There is no forgetting or
recalibration, and update count/parameter movement do not measure wall-clock
evidence age.

## Learning during simulated tracking

The [live refinement example](../../examples/live_refinement.py) adds a bounded
transition buffer, a background learner, and explicit controller handoff:

```bash
uv run python examples/live_refinement.py
```

It compares frozen and adopting controllers for two synthetic platform variants
in each family. Both arms continue tracking while the learner receives delayed
and interrupted telemetry. `TransitionBuffer` consumes already aligned state
transitions and applied commands; it does not align raw sensor messages.
`RefinementWorker` owns learning and controller preparation. The control owner
applies an offered revision at a solve boundary and acknowledges its decision.

The [experiment and interface notes](../streaming-refinement.md) describe the
input contract, retention and gap policies, adoption gate, and recorded results.
The trials exercise this composition but miss many control deadlines. Raw
telemetry alignment, estimator semantics, independent execution scheduling and
durable restart remain separate work before a hardware integration.

## An independent fixed-wing simulator

The [Cascade example](../../examples/cascade_refinement.py) uses the same tracking
and refinement loop with Cascade's Skywalker X8 as the plant:

```bash
uv run --group cascade python examples/cascade_refinement.py
uv run python scripts/audit_live_refinement.py artifacts/cascade-refinement
```

The plant supplies canonical state observations and receives bounded commands.
Its coefficients and internal actuator/separation states are not inputs to
Glassbox's fitter, learner or controller. The calibration pilot does use
Cascade's published stabilizer and a simulator-derived trim; that dependency
is recorded. Requested surface angles use `surface_angle_command` in radians,
so their command meaning survives fitting and serialization.

Use `--calibration-only` to stop after fit and held-out prediction, or
`--reuse-calibration` to reuse the saved calibration before tracking. Reuse
checks recording and belief identities. `--repeats` changes the number of paired
tracking trials, with order alternating between pairs. The
[experiment report](../cascade-refinement.md) records the source snapshot,
data roles, initial prediction results, later adoption outcomes and limitations.
