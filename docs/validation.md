# Validation

This page is the library's evidence. Every number on it is the literal content
of a recorded artifact under [`results/`](results/), named here by its path and
its key so a reader can open the JSON and check it. Each artifact is machine
output produced by one manifest entry, and one command regenerates it:

```bash
uv run glassbox record-results --only <artifact-name>
```

`glassbox record-results --list` names every artifact, its tier and whether it
is recorded or still pending. The two local artifacts regenerate in this
repository with nothing downloaded and are checked in continuous integration.
The six corpus artifacts need a pinned public corpus on disk and are a
maintainer job; [`CONTRIBUTING.md`](../CONTRIBUTING.md) has the table and the
procedure.

Negative results and withdrawn approaches are not artifacts. They are recorded
as prose in the [literature review](literature-review.md) together with the
last commit that carried their code.

## Corpus evidence

Five public corpora, each with its own published protocol. A corpus that does
not verify against its pinned digests is a different corpus, so verification
is not optional and every artifact records the digest of every file its run
read.

| Corpus | Airframe | Pinned version | Protocol | Holdout | Score against the baseline | Artifact |
| --- | --- | --- | --- | --- | --- | --- |
| Nano-Quadrotor | 27 g quadrotor, measured rotor speeds | commit `2d921b57` | `nanodrone`, rolling 1 to 50 steps against hold-state | the three protected Melon flights, by label | no single score; per metric below | [`validation-nanodrone-results.json`](results/validation-nanodrone-results.json) |
| ARP | 3.35 kg PX4 quadrotor, normalized motor commands | commit `2d267dd0` | `windowed`, against kinematic persistence | log 66 reserved, logs 63 to 65 trained | `1.219`, worse than persistence | [`validation-arp-results.json`](results/validation-arp-results.json) |
| IDF-DS | conventional fixed wing, one motor and three surfaces | Zenodo record `16992976` | `windowed`, leave one source group out over 13 sessions | every session held out in turn | no single score; beats persistence at 1 s and 2 s, loses at 0.1 s | [`validation-idf-results.json`](results/validation-idf-results.json) |
| Skywalker X8 | flying wing, throttle and generalized elevons | Dataverse `1.0` | `x8`, boundary-safe rolling windows | the four upstream validation maneuvers, by label | `0.436` residual, `0.507` structured | [`validation-x8-results.json`](results/validation-x8-results.json) |
| EPFL TOPOPlane2 | conventional fixed wing, 5 Hz fused state | Zenodo `v1` | `windowed` over two fit reports | the last two chronological segments of one flight | `0.864` residual, `1.123` structured | [`validation-epfl-results.json`](results/validation-epfl-results.json) |

The abbreviated pins above are the leading characters of each artifact's own
`corpus.citation.pinned_version`, which carries the full value. Every row is
now recorded. No number here was carried over from the experiment pages this
page replaced, because the code that produced those numbers no longer exists;
where a corpus has no scalar score its protocol says so and the section below
gives the comparison the artifact actually records.

## Diagnostics

Three closed-loop and simulator diagnostics. None of them is a pass/fail gate
on the library, and none is a flight-safety claim.

| Diagnostic | What it measures | Tier | Headline | Artifact |
| --- | --- | --- | --- | --- |
| NMPC acceptance | the solver on 16 synthetic scenarios, 8 nominal and 8 under parameter mismatch | local | tracking ratio `0.651` nominal and `0.552` under mismatch | [`nmpc-acceptance-results.json`](results/nmpc-acceptance-results.json) |
| Adaptive recovery | the whole belief-to-control path across a configuration change | local | resolved rank `1` to `15`, independent prediction ratio `0.868` | [`adaptive-recovery-results.json`](results/adaptive-recovery-results.json) |
| Cascade X8 | an unfitted published physics model against the same X8 campaign | corpus | `2.760` as published, `0.679` at its best documented variant | [`cascade-x8-validation-results.json`](results/cascade-x8-validation-results.json) |

## Nano-Quadrotor

The [IDSIA Nano-Quadrotor benchmark](https://github.com/idsia-robotics/nanodrone-sysid-benchmark)
is 15 recordings of a 27 g quadrotor at 100 Hz, with motion-capture position
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

ARP Laboratory's four large-quadrotor ULogs are the first real PX4 multirotor
reference beyond synthetic and Nano-Quadrotor data. `glassbox corpus prepare
arp` retains the longest sustained powered interval in each recording and
writes four 50 Hz canonical trajectories, each with a stable path-independent
`source_group` so a leave-one-recording-out split cannot mix segments of one
flight. These recordings omit the usual arming and land-detection streams and
their local-position origin is not the takeoff point, so the adapter selects
on telemetry completeness and the powered interval instead of the operational
armed and height gates.

The recorded chain trains on logs 63 to 65 and reserves log 66, which is the
protected evaluation this corpus was always developed under, then scores the
reserved log against kinematic persistence under the `windowed` protocol. The
structured model scores `1.219` (`results.score_vs_baseline`, above one being
worse than the baseline), so on this held-out log at the standard 0.1, 0.5 and
2 second horizons the fitted model loses to constant-velocity,
constant-body-rate persistence. This is the first recording of this chain;
there is no earlier number for the same protocol to compare it against.

The boundary is the map from normalized motor command to translational
acceleration, which is the clearest multirotor limitation this corpus
exposes. The four recordings are replicates from one vehicle, so they are a
real-ULog integration and identification reference rather than four
independent airframes, and a promotion decision on a more expressive force law
needs an untouched second normalized-command airframe because log 66 is spent.

## IDF-DS

The [IDF-DS fixed-wing telemetry release](https://zenodo.org/records/16992976)
is 13 independent recording sessions of one conventional airframe, one motor
with split ailerons, elevator and rudder. Preparation verifies the published
MD5 of a 2.12 GB archive, extracts the 13 raw ULogs, verifies each member's
size and CRC32, and converts every telemetry-complete interval to 50 Hz
canonical trajectories, keeping only contiguous segments of at least ten
seconds. Every dropout-separated segment of a session keeps that session's
group identity.

The recorded chain is the leave-one-source-group-out run of the structured
residual over all 13 sessions (`results.fold_count`), 119 trajectories of
estimated state in total. There is no single scalar score for this corpus. The
equal-session macro table is `results.aggregate.horizon_rollouts` and the
comparison against kinematic persistence is
`results.aggregate.model_over_baseline`, per horizon and per metric. Read
those ratios in the direction the artifact states them, model error over
baseline error, so a value below one favours the model. That is the opposite
direction from the Nano-Quadrotor artifact's `model_vs_baseline`, which is
baseline over model, and the two must not be quoted in the same sentence
without saying which is which.

The result is a horizon crossover, and it is the most useful thing this corpus
records. At 0.1 seconds the fitted model is worse than constant-velocity,
constant-body-rate persistence on all four metrics, by `2.951` on position,
`3.216` on velocity, `2.056` on attitude and `1.922` on angular velocity. At
0.5 seconds it is still behind on position, velocity and attitude and ahead on
angular velocity (`1.376`, `1.011`, `1.184`, `0.954`). At 1 second it is ahead
on all four (`0.754`, `0.589`, `0.740`, `0.684`) and at 2 seconds it is roughly
twice as good as the baseline (`0.493`, `0.469`, `0.462`, `0.538`). Persistence
is simply very strong over one or two samples and degrades with horizon, so a
learned dynamics model earns its place at the horizons a controller plans
over, not at the horizons an estimator already covers.

The spread across sessions is tight for a thirteen-fold leave-one-out. At two
seconds the p90 held-out-session errors are `1.135` m, `1.287` m/s, `13.12`
degrees and `0.176` rad/s against medians of `0.927` m, `1.069` m/s, `10.47`
degrees and `0.150` rad/s (`results.distribution.horizon_rollouts["2s"]`),
with the worst session at `1.216` m.

The boundary is that this is a cross-session generalization result on one
conventional airframe, not evidence of zero-shot parameter transfer, and this
artifact compares the model only against the persistence baseline: it runs no
comparison against an earlier model class, so nothing here ranks one
parameterization against another. Multi-minute sessions still expose
long-rollout instability, and this is the one registry entry large enough for
the window budget to bind, so its numbers move when that budget changes.

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
(`results.comparisons.structured_residual_vs_structured.score`). Of the rows
recorded so far it is the only one where a fitted model beats persistence on
an independent holdout.

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
into one characterization. The structured residual scores `0.8641` against
kinematic persistence and the structured model `1.1228`
(`results.models[<name>].score_vs_baseline`), and `results.selected_model` is
the residual with promotion refused.

The boundary is that all segments come from one flight, so no partition can
establish independent-flight generalization. The evaluator records that fact:
`protocol.independent_holdout` is false and `protocol.can_promote_model` is
false regardless of the scores in the same file. Body rates are reconstructed
from the attitude derivative at 5 Hz rather than measured, which is a
documented adapter limitation.

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
equal-scenario geometric mean of normalized tracking RMS was `0.6509` of the
non-optimizing trim baseline nominally and `0.5522` under parameter mismatch
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
adapted belief produced `1.683` recovery-tail tracking RMS and `1.120`
recovery-tail attitude and rate RMS; against the hidden oracle point model
those ratios were `1.827` and `1.416` (`comparisons`). Every trace was finite,
all commands stayed within bounds, and no solve fell back
(`observations.all_recovery_traces_finite`, `all_commands_within_bounds`,
`all_recovery_traces_without_fallback`). The negative tracking comparison is
part of the evidence, not an acceptance threshold.

Every trace also left the validity envelope. Maximum actual utilization was
`1.085`, `1.173`, `1.063` and `1.136` for the `stale_belief`,
`adapted_belief`, `adapted_mean_point` and `oracle_mean_point` arms
(`recovery[*].maximum_actual_validity_utilization`), so
`observations.all_actual_recovery_within_validity_support` is false.

The absolute rank cutoff retains weakly informed directions as stronger ones
accumulate information. The posterior covariance trace is consequently
`3954.364`, compared with `0.0625` on the seed's single resolved direction
(`evidence.fleet`). These are not comparable total uncertainty bounds: the
seed has unresolved parameters. The three incomplete arms explicitly allow
partial-information planning and report an unbounded uncertainty margin as
JSON `null`; the complete adapted arm reports `1.491`
(`recovery[*].parameter_uncertainty_complete`, `unresolved_parameters_allowed`,
`maximum_normalized_model_uncertainty_standard_deviation`). Resolving a
parameter direction does not mean estimating it precisely. This diagnostic
exposes that limitation instead of deleting weak directions from the reported
uncertainty.

## Cascade X8

`glassbox benchmark cascade-x8` evaluates an unfitted published physics model
against the same four untouched X8 validation maneuvers, under exactly the
protocol, horizons, metrics and persistence baseline of `glassbox evaluate
--protocol x8`. [Cascade](https://github.com/Rylandl/cascade) assembles its
Skywalker X8 from the published NTNU model: wind-tunnel statics, XFLR5 rate
derivatives, bifilar-pendulum inertia, the pyfly stall blend and an exact map
of the NTNU propulsion law. Nothing in it was fitted to this campaign. The
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
(`channels.{X,Y,Z}.mean`). Three things follow, and all three are testable
outside this campaign. The published model is untrimmed at the flight
condition, which is why every scoring row above the baseline moves the
pitching-moment reference point forward. Its inertia is understated, because
the sweep's interior optimum is the bifilar-pendulum tensor doubled
(`best_model`). And the campaign's inferred vertical wind is too large: every
best-scoring row applies a fraction of it, half for the coefficient table and
a quarter for the panels (`best_model`, `component_panels.best_model`).

This is characterization evidence for the simulator and the campaign, not a
candidate under any development contract, and the sweep rows are not a fit.
The fitted reference arms this artifact compares against are the same two X8
fits the corpus row above records, at the same scores
(`glassbox_reference_models[<name>].score_vs_kinematic_persistence`).

## PX4 SITL corpora

Recording a PX4 SITL corpus needs a running simulator container, so these
corpora are not entries in the manifest and no artifact under `results/`
backs them. The numbers below were recorded by hand before the 2026-09-01
estimator revisions, from runs whose ULogs are not in the repository, and they
cannot be regenerated with the documented commands without Docker. They are
kept as prose, and they are not comparable with the corpus tables above.

The multirotor corpus is 24 ULogs and 428.0 seconds of usable ground truth
from the `px4_sih_quadx` configuration, balanced six per maneuver family,
eight per excitation condition and twelve per initial heading. On a
leave-one-maneuver-family-out benchmark the structured model reached
ground-truth position and attitude errors of 0.00012 m / 0.13 degrees at 0.1
seconds, 0.0216 m / 3.36 degrees at 1 second, 0.167 m / 6.57 degrees at 2
seconds and 1.23 m / 10.18 degrees at 5 seconds, with complete flights at
7.39 m / 16.56 degrees. The matched estimated-state benchmark reached
0.0117 m / 0.34 degrees, 0.109 m / 4.19 degrees, 0.284 m / 7.19 degrees and
1.34 m / 10.87 degrees, with complete flights at 5.85 m / 15.84 degrees. The
worst family was `lateral_steps` in both cases. The decision on that run was
no model promotion: balanced data preserved useful short-horizon translation
but did not close rotational or complete-flight error.

The fixed-wing corpus is 12 ULogs and 177.58 seconds of ground truth at 50 Hz
across 5.00 to 6.78 m/s. On the same kind of benchmark the structured model
reached 0.00058 m / 0.071 degrees at 0.1 seconds, 0.050 m / 2.07 degrees at 1
second, 0.185 m / 4.19 degrees at 2 seconds and 1.01 m / 10.94 degrees at 5
seconds; the matched estimated-state results were 0.0119 m / 0.23 degrees,
0.126 m / 3.13 degrees, 0.294 m / 5.08 degrees and 0.855 m / 9.71 degrees.
Complete-profile open-loop rollout still diverged, at 6.54 m / 30.4 degrees on
ground truth and 14.24 m / 69.6 degrees on the estimated state. Both corpora
are simulator evidence for the ingestion and training contracts, not
real-flight evidence.

The [PX4 ULog guide](guides/px4-ulog.md) has the recorder and the extraction
commands.
