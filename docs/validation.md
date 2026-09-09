# Validation

Recorded comparisons are linked below. `glassbox record-results` regenerates
the artifacts in [`results/`](results/): the local tier runs in CI; the corpus
tier needs downloaded datasets and longer fits. See
[Contributing](../CONTRIBUTING.md#recorded-results) for the procedure.

## Corpus evidence

The corpus registry pins upstream files and evaluation splits. Each artifact
records its input pins, fit configuration and scoring protocol.

| Corpus | Airframe | Pinned version | Protocol | Holdout | Score against the baseline | Artifact |
| --- | --- | --- | --- | --- | --- | --- |
| Nano-Quadrotor | quadrotor, measured rotor speeds | commit `2d921b57` | `nanodrone`, rolling 1 to 50 steps against hold-state | the three protected Melon flights, by label | no single score; per metric below | [`validation-nanodrone-results.json`](results/validation-nanodrone-results.json) |
| ARP | 3.35 kg PX4 quadrotor, normalized motor commands | commit `2d267dd0` | `windowed`, against kinematic persistence | log 66 reserved, logs 63 to 65 trained | `1.220`, worse than persistence | [`validation-arp-results.json`](results/validation-arp-results.json) |
| IDF-DS | conventional fixed wing, one motor and three surfaces | Zenodo record `16992976` | `windowed`, leave one source group out over 13 sessions | every session held out in turn | no single score; beats persistence at 1 s and 2 s, loses at 0.1 s | [`validation-idf-results.json`](results/validation-idf-results.json) |
| Skywalker X8 | flying wing, throttle and generalized elevons | Dataverse `1.0` | `x8`, boundary-safe rolling windows | the four upstream validation maneuvers, by label | `0.436` residual, `0.507` structured | [`validation-x8-results.json`](results/validation-x8-results.json) |
| EPFL TOPOPlane2 | conventional fixed wing, 5 Hz fused state | Zenodo `v1` | `windowed` over two fit reports | the last two chronological segments of one flight | `0.846` residual, `1.123` structured | [`validation-epfl-results.json`](results/validation-epfl-results.json) |

The abbreviated pins expand to `corpus.citation.pinned_version` in each artifact.

## Diagnostics

Three closed-loop and simulator diagnostics. None of them is a pass/fail gate
on the library, and none is a flight-safety claim.

| Diagnostic | What it measures | Tier | Headline | Artifact |
| --- | --- | --- | --- | --- |
| NMPC acceptance | the solver on 16 synthetic scenarios, 8 nominal and 8 under parameter mismatch | local | tracking ratio `0.644` nominal and `0.541` under mismatch | [`nmpc-acceptance-results.json`](results/nmpc-acceptance-results.json) |
| Adaptive recovery | the whole belief-to-control path across a configuration change | local | resolved rank `1` to `15`, independent prediction ratio `0.868` | [`adaptive-recovery-results.json`](results/adaptive-recovery-results.json) |
| Cascade X8 | an unfitted published physics model against the same X8 campaign | corpus | `2.760` as published, `0.679` at its best documented variant | [`cascade-x8-validation-results.json`](results/cascade-x8-validation-results.json) |

## Nano-Quadrotor

The [IDSIA Nano-Quadrotor benchmark](https://github.com/idsia-robotics/nanodrone-sysid-benchmark)
is 15 recordings of a quadrotor at 100 Hz, with motion-capture position
and attitude, onboard velocity and rate, and measured rotor speeds as the
control input rather than motor commands. Twelve Square, Random and Chirp
flights train and the three Melon recordings are the published protected test
split. The benchmark's own metric initializes a half-second open-loop
prediction at every admissible sample and reports mean Euclidean error at
every step from one to fifty. Glassbox never lets a window cross a recording
boundary; the released upstream metric script concatenates the three Melon
runs before shifting, so the comparison is near comparable rather than exact
and the artifact says so in `results.published_reference_comparison.comparability`.

The recorded run fits the structured residual on the twelve training flights
with the Melon profile reserved by label
(`fit.structured_residual.split.holdout`), across 7,192 training windows
(`fit.structured_residual.spec.training_windows`). Against the boundary-safe
hold-state baseline the cumulative 1-to-50-step ratios are `7.661` on
position, `3.044` on velocity, `1.385` on attitude and `0.918` on angular
velocity (`results.model_vs_baseline.cumulative_simulation_error`, defined as
baseline error over model error, above one favouring the model). Against the
published Physics plus Residual reference the equal-metric geometric
cumulative ratio is `1.058` and both
`beats_every_published_cumulative_metric` and
`beats_every_published_50_step_metric` are false
(`results.published_reference_comparison`). The `nanodrone` policy scores
prediction steps rather than a horizon table, so `results.score_vs_baseline`
is null by construction.

The boundary is the rotational channel. The angular-velocity ratio below one
says the compact structured angular model is still worse than holding the
measured rate constant over the cumulative metric, which is the clearest open
weakness this benchmark exposes. The result is competitive benchmark
performance next to a published reference, not a state-of-the-art claim, and
it says nothing about complete-flight open-loop stability.

## ARP

`glassbox corpus prepare arp` converts four ARP Laboratory quadrotor ULogs to
50 Hz trajectories, selecting each recording's longest sustained powered
interval and preserving its preceding actuator commands. These logs omit
arming and land-detection streams, and their local-position origin is not the
takeoff point. A stable `source_group` keeps each recording in one fit split.

The recorded chain trains on logs 63 to 65 and evaluates log 66 against
constant-velocity, constant-body-rate persistence at 0.1, 0.5, 1 and 2 seconds.
The structured model scores `1.220` (`results.score_vs_baseline`, above one
being worse). Position and velocity lose to persistence at every scored
horizon; attitude and body rate beat it. These are repeat measurements on one
vehicle, and log 66 has already been used in model-development experiments.

## IDF-DS

The [IDF-DS fixed-wing telemetry release](https://zenodo.org/records/16992976)
is 13 independent recording sessions of one conventional airframe, one motor
with split ailerons, elevator and rudder. Preparation verifies the published
MD5 of a 2.12 GB archive, extracts the 13 raw ULogs, verifies each member's
size and CRC32, and converts every telemetry-complete interval to 50 Hz
canonical trajectories, keeping only contiguous segments of at least ten
seconds. Every dropout-separated segment of a session keeps that session's
group identity.

The recorded chain fits the structured residual in 13 leave-one-session-out
folds over 119 trajectories. Each session receives equal weight in the
aggregate. `results.aggregate.model_over_baseline` reports model error over
kinematic-persistence error for each horizon and metric; below one favours
the model.

At 0.1 seconds the model loses to persistence on all four state groups. At
0.5 seconds it improves angular velocity only; at one and two seconds it
improves all four. The two-second ratios are `0.493` for position, `0.468`
for velocity, `0.461` for attitude and `0.537` for angular velocity.

At two seconds, the median held-out-session position error is `0.926` m,
p90 is `1.144` m and the maximum is `1.216` m
(`results.distribution.horizon_rollouts["2s"].position_rmse_m`). Full-session
open-loop errors remain large (`results.aggregate.full_rollout`).

## Skywalker X8

NTNU's pinned [Skywalker X8 campaign](https://doi.org/10.18710/U4TLYV) is 17
dedicated maneuvers of about ten seconds at 40 Hz, 13 upstream training and
four untouched validation. It is the first materially different fixed-wing
configuration in the matrix: a flying wing with no yaw control, no
conventional tail, and normalized throttle plus generalized elevon roll and
pitch. The campaign provides an author-validated wind estimate that Glassbox
types as trusted context and holds through each prediction, and the adapter
reconstructs local position by trapezoidally integrating the internally
consistent 40 Hz velocity because the upsampled GPS position is visibly
staircase-like.

The recorded chain fits both maintained model classes on the exact upstream
split, reserving the four flights labeled `benchmark_split=validation`, then
runs the boundary-safe rolling comparison against constant-velocity,
constant-body-rate persistence. The structured residual scores `0.4359` and
the structured model `0.5069`
(`results.models[<name>].score_vs_kinematic_persistence`), and the residual is
`0.8600` of the structured model over every horizon and state metric
(`results.comparisons.structured_residual_vs_structured.score`).

The boundary is that this is characterization evidence for one flying-wing
airframe under a trusted wind estimate; it does not claim that every
estimator-derived wind source is trustworthy. Complete-maneuver open-loop
rollout on this campaign is fragile to residual initialization, so the
windowed table is the claim and a complete-maneuver figure is one draw.

## EPFL TOPOPlane2

EPFL's pinned [TOPOPlane2 navigation flight](https://zenodo.org/records/10337559)
adds a second conventional configuration and a third fixed-wing dataset
family. Its 5 Hz fused state is paired with GNSS-tagged autopilot outputs and a
measured pitot channel. The published flight intentionally exercises
navigation outages, so the adapter compares fused altitude against the
independent barometric signal, keeps only the dominant navigation-consistent
mode, and removes two seconds around every boundary.

The recorded chain fits both model classes on the canonical segments with the
last two chronological segments reserved, then combines the two fit reports
into one characterization. The structured residual scores `0.8462` against
kinematic persistence and the structured model `1.1228`
(`results.models[<name>].score_vs_baseline`), and `results.selected_model` is
the residual.

All segments come from one flight; `protocol.independent_holdout` is false.
Body rates are derived from attitude at 5 Hz rather than measured.

## NMPC acceptance

`glassbox benchmark nmpc` runs the solver over a fixed synthetic scenario
suite: multirotor hover, translation and attitude, and fixed-wing trim,
altitude, path, turn and flap, each once nominally and once with the plant
parameters perturbed away from the model. It is the one recorded diagnostic
with fixed thresholds, declared before the solver was tuned and recorded in
`thresholds` beside every result.

On the recorded run all 16 scenarios were finite with no fallback and no
command-bound violation, every mismatch case stayed inside the fitted validity
envelope, and every one of the nine checks in `summary.checks` passed. The
equal-scenario geometric mean of normalized tracking RMS was `0.6440` of the
non-optimizing trim baseline nominally and `0.5407` under parameter mismatch
(`summary.nominal_geometric_mean_tracking_ratio` and
`summary.mismatch_geometric_mean_tracking_ratio`), against a declared maximum
of `0.80` nominally. No individual scenario hid inside those aggregates, which
is what `summary.checks.nominal_individual` and `mismatch_individual` assert.

Post-JIT solve time is recorded per scenario as median, p90 and maximum under
`summary.post_jit_solve_time_s` and per scenario in `scenarios[*]`. Those
figures depend on the host and its load, so they live in the artifact and not
in this prose. This benchmark applies no elapsed-time acceptance gate.
Passing the functional gates is
not a real-time claim and not a flight-safety claim.

## Adaptive recovery

`glassbox benchmark recovery` is the one diagnostic that exercises the whole
path from evidence to command. A quadrotor's arm-length ratio changes from
`1.0` to `1.2214` (`configuration.arm_length_ratio_after`), which the
synthetic plant maps into inverse roll and pitch angular authority. The belief
is seeded from five sibling configurations whose sample covariance inverts, on
its rank-one supported subspace, to precision; it then absorbs 0.8 seconds of
telemetry from the changed vehicle, is evaluated on independent 0.6-second
prediction windows, and drives four prewarmed closed-loop recoveries from one
bounded initial disturbance.

The update raised the resolved rank from `1` to `15` of the `15` estimable
coordinates and reported an information gain of `1.698` nats over 40 windows
(`evidence.fleet.seed_resolved_rank`, `posterior_resolved_rank`,
`estimable_parameter_count`, and `evidence.adaptation`). The whitened one-step
innovation fell from `0.2514` to `0.2111` and the step was `0.0084` of the
prior precision's own standard deviation. Independent normalized 0.6-second
prediction RMS fell from `0.033394` to `0.028970`, a ratio of `0.868`
(`evidence.independent_prediction`).

Better prediction did not produce better recovery. Against the stale arm, the
adapted belief produced `1.604` recovery-tail tracking RMS and `1.076`
recovery-tail attitude and rate RMS; against the hidden oracle point model
those ratios were `1.823` and `1.506` (`comparisons`). Every trace was finite,
all commands stayed within bounds, and no solve fell back
(`observations.all_recovery_traces_finite`, `all_commands_within_bounds`,
`all_recovery_traces_without_fallback`). The negative tracking comparison is
part of the evidence, not an acceptance threshold.

Every trace also left the validity envelope. Maximum actual utilization was
`1.347`, `1.056`, `1.336` and `1.423` for the `stale_belief`,
`adapted_belief`, `adapted_mean_point` and `oracle_mean_point` arms
(`recovery[*].maximum_actual_validity_utilization`), so
`observations.all_actual_recovery_within_validity_support` is false.

The [recovery investigation](recovery-investigation.md) compares uncertainty
terms, optimizer choices and further telemetry on this scenario.

## Cascade X8

`glassbox benchmark cascade-x8` evaluates an unfitted published physics model
against the same four X8 validation maneuvers, under exactly the
protocol, horizons, metrics and persistence baseline of `glassbox evaluate
--protocol x8`. [Cascade](https://github.com/Rylandl/cascade) assembles its
Skywalker X8 from the published NTNU model: wind-tunnel statics, XFLR5 rate
derivatives, bifilar-pendulum inertia, the pyfly stall blend and an exact map
of the NTNU propulsion law. The nominal configuration uses published parameters. The
sweep rows vary documented uncertainties of the published model, applied
identically to every window and never tuned per window: the pitching-moment
reference point as a forward CG shift, the mass, a uniform inertia scale, and
the fraction of the campaign's inferred vertical wind that is applied.

As published the model scores `2.760` against kinematic persistence and its
best documented variant, a 50 mm forward reference point at 4 kg with the
inertia doubled and half the inferred vertical wind, scores `0.679`
(`models[<name>].score_vs_kinematic_persistence`, with `primary_model` and
`best_model` naming the two rows). The artifact records that best row as
`1.339` and `1.558` times the fitted structured and structured-residual
reference arms it was recorded beside (`comparisons`). The from-geometry
component-panel variant of the same airframe scores `2.185` as published and
`0.661` at its own best variant (`component_panels`), within `0.02` of the
coefficient table's best.

The same command records lag-aware equation-error regressions over all 17
maneuvers under `residual_diagnostics`. On the row the sweep's best variant
sits nearest, at a 50 mm forward reference point and 0.4 of the inferred
vertical wind, the fit explains the pitch moment well and the roll moment
less well (`channels.pitch.r_squared` `0.8295`, `channels.roll.r_squared`
`0.7069`), and the body-force residuals carry constant offsets of a few
newtons: `+2.67` forward, `-2.63` to the right and `-2.68` downward
(`channels.{X,Y,Z}.mean`). These residuals and the selected variants suggest
inertia, trim and wind hypotheses to test on another campaign. The best rows were selected on these
validation maneuvers.

The fitted reference arms this artifact compares against are the same two X8
fits the corpus row above records, at the same scores
(`glassbox_reference_models[<name>].score_vs_kinematic_persistence`).

## PX4 SITL corpora

SITL recording needs a running simulator container. The
[PX4 guide](guides/px4-ulog.md) describes recording and extracting trajectories.
These runs are outside the recorded-results manifest.
