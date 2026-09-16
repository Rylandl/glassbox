# Learning during simulated tracking

The workflow now fits a platform, tracks a reference while learning in a separate
thread, and explicitly hands an evaluated model/controller pair to the control
owner. The same composition runs on synthetic multirotor and fixed-wing variants.
The recorded trials demonstrate that composition. Their deadline misses and
mixed tracking outcomes do not establish real-time control or consistently
beneficial adoption.

The [Cascade follow-up](cascade-refinement.md) replaces the synthetic fixed-wing
plant with independently implemented X8 dynamics, using the same learner and
handoff loop. It includes separate calibration and held-out prediction records.

## Run and inspect

From the repository with Glassbox installed:

```bash
uv run python examples/live_refinement.py
uv run python scripts/audit_live_refinement.py artifacts/live-refinement
```

The [example](../examples/live_refinement.py) accepts `--family multirotor` or
`--family fixedwing`, `--variants 1`, `--duration-s`, `--fit-steps`,
`--learner-delay-s`, and `--output`. Reusing an output directory overwrites its
reports. A run removes the previous summary before writing new trial data.
The audit reads saved arrays and events without rerunning identification or
control. It checks tracking metrics, command bounds, interval accounting,
prediction gates, actual revision changes, acknowledgement order, and source
hashes when an executed-source archive is present.

Each platform directory contains calibration recordings, the fit report and
initial belief. Each trial contains `tracking.npz`, an `events.jsonl` journal,
serialized beliefs, and `summary.json`. The trace records the model revision
actually used for each control interval. The root `comparison.json` collects
both controller arms for every platform.

## The interface boundary

The experimental types live in `glassbox.workflows.streaming`; they are not
exported from the stable package root. The application supplies already aligned
transitions in the model's declared coordinates and sample period. Each command
must be the one actually applied between the two endpoint states. Source time
and monotonic host reception time remain separate. The buffer validates the
source clock, rebases each block to a local time grid, and retains the source
origin in provenance.

| Owner | Responsibility |
| --- | --- |
| Producer / `TransitionBuffer` | Validate transitions, retain a fixed command history, emit fixed-size blocks, discard incomplete blocks across missing state observations |
| `RefinementWorker` | Own `ModelRefiner`, score before learning, account for skipped intervals, prepare controllers and publish offers |
| Control owner | Pin a model/controller for a solve, inspect an offer, apply or reject it at a solve boundary, acknowledge that decision |
| Application journal | Persist block scores, gaps, revision artifacts, offers and adoption acknowledgements |

Queue overflow drops the oldest pending block. The worker advances the recording
cursor through an explicit `skip` event before observing the next accepted block;
skipped intervals contribute no information. Known command history is still
available after a state-telemetry interruption. Missing commands require an
upstream policy; this buffer cannot infer them or align raw sensor messages.

The worker fixes both block size and history length and caps queued blocks,
retained results and recording cursors. Revision retention includes active and
candidate models, recent scores, and one outstanding adoption offer. Revision
IDs remain unique after pruning. The replay coordinator's defaults still retain
full history and results; bounded mode is explicit. Retired exact-content hashes
are pruned, while recording cursors continue to reject reused intervals. Neither
mode proves independence under renamed or differently segmented recordings.

Finite command history estimates actuator state. Reports expose its length,
whether earlier commands were truncated, and the remaining initial-offset
fraction implied by each model's first-order lag. That fraction is not a
probability or a bound on mismatch with the actual actuator.

An offer contains the previously scored candidate and its prepared controller,
plus the expected active revision and scoring interval. Preparation runs on the
worker. The active learner revision changes only after the control owner applies
the pair and acknowledges it. Failed preparation leaves that active revision
unchanged and exposes a worker error. Callbacks and nested belief metadata remain
subject to the single-owner contract. Shutdown has a bounded join; a thread
cannot forcibly cancel an in-progress JAX kernel. These limits cover queued
telemetry and Python retention, not compiler caches or process-wide memory.

## Recorded experiment

The 12 September 2026 snapshot used the example's default command. The
[comparison](investigations/live-refinement/comparison.json) contains the original
reports; the [audit](investigations/live-refinement/audit.json) records successful
checks for all eight trials. The [trial archive](investigations/live-refinement/trials.zip)
contains their arrays, journals, beliefs, calibration data and 68 executed source
files with hashes. It also retains the earlier negative smoke run. The
[audit source](investigations/live-refinement/audit.source.py) preserves the
postprocessing used for this snapshot.

Each family has two platforms: thrust/lag scales of 0.90/0.80 and 1.10/1.25
relative to the synthetic base. Parameters stay fixed during each trial. Each
platform supplies three calibration recordings of three seconds each; two fit
the parameters and one measures forecast error. The fit uses ten optimization
steps. These are demonstration budgets, not measured minimum requirements.

Tracking uses a 20 Hz source clock for 240 intervals, four-interval learner
blocks, 20 preceding commands, two pending queue slots and two retained results.
The first learner block is deliberately delayed. Six consecutive intervals of
state observations are withheld from learning during tracking; the controller
continues receiving the state. The generator and observations use simulator
truth, and calibration uses a stabilizer with known plant parameters.

Both frozen and adopting arms run a learner. Only the adopting arm prepares
replacement controllers. Trial order alternates across variants. Each controller
uses the fitted mean and explicitly allows unresolved parameters; the ordinary
controller defaults are unchanged. The learner retains parameter information
and empirical error evidence but skips propagated parameter covariance while
scoring. This exposed a useful generic API improvement:
`belief.rollout(..., propagate_parameter_covariance=False)` preserves mean
predictions and returns `None` for the omitted covariance component.

The application gate requires at least 2% improvement in a normalized four-metric
prediction norm, no individual metric regression above 5% plus a small numerical
floor, and predicted operation within the observed envelope. Normalization scales
are explicit in `comparison.json:gate`. The control owner also checks the
expected active identity and a maximum score age of 100 source intervals.
The selected revision was scored on a subsequent block before being offered;
the just-updated successor is not eligible on that block. This remains correlated
online evidence, not an independent-flight holdout or proof of better control.

| Platform | Position RMSE, frozen / adopting (m) | Whole-tick deadline misses, frozen / adopting | Applied revisions |
| --- | --- | --- | --- |
| Multirotor 0 | 0.01255 / 0.01192 | 194 / 55 | 5 |
| Multirotor 1 | 0.17722 / 0.17783 | 54 / 58 | 6 |
| Fixed wing 0 | 0.93504 / 0.70682 | 42 / 39 | 5 |
| Fixed wing 1 | 0.94786 / 0.78592 | 43 / 40 | 4 |

These values come from `comparisons[].trials.{frozen,adopting}` in the linked
comparison, using `tracking_rmse.position_rmse_m`, `deadline_misses` and
`applied_adoptions`. Position RMSE averages squared coordinate errors. All
eight trials completed with finite states, bounded commands and no worker error.
The fixed-wing runs show lower position and attitude errors after adoption;
the quad results are mixed, including higher errors on the second variant.

Only 11 to 16 of 53 submitted blocks were processed per trial, with the remaining
37 to 42 dropped from the pending queue. Every final queue/retention report was
within its configured bounds. The first frozen quad run has a large timing
outlier; it is retained. These host-scheduled runs cannot isolate the causal
effect of learning on performance because scheduling changes which blocks are
processed and when fallbacks occur. The earlier smoke run completed without
adopting a revision. Its different duration and scoring path do not form a
controlled timing comparison with this snapshot.

## What this leaves unresolved

The experiment is paced fixed-step simulation: when computation is late, plant
simulation time slows. It does not measure a real-time plant evolving under a
held stale command. A bounded queue prevents backlog growth but gives the
controller no scheduling isolation from learning, compilation or preparation.
The observed deadline misses make that distinction material.

Both families share their synthetic plant equations with the model family.
There is no sensor noise, unknown structural dynamics, hardware adapter, or
within-trial parameter change. Learning has no forgetting and does not refresh
forecast-error measurements. A short prediction gate can accept a model whose
closed-loop tracking is worse, as this comparison illustrates.

The next experiment should isolate the control executor from learning, keep
the plant clock advancing through delayed commands, and repeat paired trials
with predetermined telemetry/fault schedules. Longer evaluation windows and
explicit behavior acceptance criteria can then test adoption separately from
transport correctness. Raw telemetry alignment, state-estimator semantics and
restart accounting remain necessary integration work.
