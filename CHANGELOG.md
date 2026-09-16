# Changelog

All notable changes to Glassbox are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## Unreleased

- Measure the live improvement row for the first time.
  `docs/harness/live-v2.json` is the fifth frozen gate, with its own digest
  constant and a `live` harness command: `control-v3`'s plant, task,
  calibration, seeds and both fitted arms, with the structured belief flying
  from the first interval while the generic learner receives the trial's own
  aligned transitions through the existing `TransitionBuffer` and
  `RefinementWorker` seam, refits with `update(recordings)` on whole
  40-interval blocks inside a declared budget, and is handed the controller
  through the acknowledged handoff when its final-step forecast error on the
  most recent block it did not fit on is at or below the structured belief's on
  the same rows. The trajectory is computed in simulated time and reproduces:
  the loop is unpaced, the solver is given no deadline, the worker is driven
  synchronously, and a candidate is released a declared block period after the
  block it was scored on, so two runs produce byte-identical tracking arrays,
  block forecasts, scores, swap intervals and metrics, and every wall
  measurement is recorded in artifacts nothing reads back. `RefinementWorker`
  now takes either a belief or a caller's own refiner keeping the new `Refiner`
  protocol, and a `synchronous` option; the structured path and both examples
  are unchanged, and the recipe, the learner modules and all eight earlier gate
  files are byte identical. Measured, not enforced: the swap happened in both
  trials at interval 140, and tracking after it was 19.4 m and 5.7 m against
  0.87 m and 0.90 m before it, because a held-out forecast comparison in a
  regime the structured arm holds almost still reads no command authority.
- Every generic forecast carries a measured error envelope, and the controller
  consumes it. The recipe already reserves a quarter of the supplied recordings
  as its development role; the windows cut from them now calibrate a
  split-conformal half-width per horizon step and per channel, at a nominal 90%,
  in the channel's own physical units, stored in the artifact and the report and
  read back through `LearnedDynamics.envelope(horizon_steps)`. There is no caller
  option: `fit`, `predict` and `update` are unchanged, and `update` recalibrates
  on the same pinned development cache it refits against. The recipe is
  `generic-memory-v3-prototype` and the artifact format `v3`; a `v2` artifact,
  which carried no envelope, is rejected on load rather than migrated.
  `glassbox.experimental.learned_plan` maps that envelope into the controller's
  tangent coordinates — velocity and body rates straight through, position by
  integrating the velocity half-widths the way the mean integrates the
  velocities, attitude through the derivative of the same polar projection the
  mean uses — declares `uncertainty_available`, and lets the seam's two
  robustness terms charge it exactly as they charge a belief's forecast-error
  covariance. `uncertainty_complete` stays false: no parameter direction is
  resolved.
- Freeze the evidence gate as `docs/harness/evidence-v1.json`, with its own
  digest constant. It declares the envelope, the channel groups coverage is
  reported over, and the 85% to 95% band held-out coverage has to land in.
  Coverage is measured inside the synthetic, platform and control runs, on the
  rows they already score, per horizon step and per channel group; every run
  saves an envelope half-width array beside every prediction and `verify`
  rebuilds it from the saved model and recomputes every number. The band is not
  enforced for its first measurement and gates from the first candidate after it,
  under the semantics the platform and control tiers already use.
- Freeze one gate semantics on the platform and control tiers as
  `docs/harness/platform-v3.json` and `docs/harness/control-v3.json`, and give
  the control tier the tracking task of `docs/cascade-accuracy.md`. A run is
  accepted when no metric regresses past its reference value times 1.05 plus
  0.005, the rule holds on every case the reference already meets it on, and
  nothing structural fails; `rule_met` is reported separately from `accepted`,
  so an enforced rule the incumbent fails no longer blocks every change.
  `platform-v3` carries `platform-v2`'s protocol byte for byte and its reference
  forward unchanged. `control-v3` replaces the cruise reference that rewarded
  claiming nothing with lateral `sin(0.35 t)` m and altitude
  `100 + 0.75 sin(0.3 t)` m over 16 s trials from declared perturbed initial
  states, records the page's 0.5 m in 95% pass criterion beside the RMSE rule,
  and requires every command channel to move at least 10% of its declared range
  in each calibration recording, measured by the run and failed closed before
  either fit. `docs/harness/control-reference.json` holds the incumbent
  measurement. `platform-v2.json` and `control-v2.json` are deleted.
- Enforce the control tier's decision rule. `docs/harness/control-v2.json`
  carries `control-v1.json`'s protocol constant for constant — the plant hash
  and pinned revision, the calibration recordings and seeds, the reference, the
  trial duration and repetitions, both arms and the controller policy — with
  `"enforced": true`, so one trial whose generic position or attitude RMSE is
  above the structured arm's, or one terminated trial, rejects the run.
  `control-v1.json` is deleted. Frozen before any candidate fit; the run it
  gates rejects the current recipe, and the iteration that froze it fitted no
  candidate, because the named mechanism's remedy is refuted by measurement.
- Add a control tier to the experimental harness and present the generic
  learner to the NMPC seam as a `PlanModel`. `docs/harness/control-v1.json` is
  frozen before any trial and `harness control` tracks one Cascade X8 trial set
  with each arm, which `harness verify` replays from the saved per-interval
  arrays without rerunning the plant or the solver; `PlanValues` gains an
  optional `observed_history` slot so a model with its own memory can carry it
  through the seam, and how a belief is presented is unchanged. First
  measurement: the rule is not met, with the generic arm at 60.35 m and 121.9
  degrees against the structured arm's 1.21 m and 1.02 degrees on every trial.
- Freeze `docs/harness/platform-v2.json` as the experimental accuracy gate.
  The decision rule is enforced, so one corpus above its structured comparator
  or its task allowance rejects the run, and `docs/harness/platform-reference.json`
  adds a per-corpus regression gate on the generic model's final-step velocity
  and body-rate RMSE, anchored to the committed file the way the synthetic
  reference already is. `platform-v1.json` is deleted; nothing is kept for old
  runs. Each manifest now pins the evaluation plan it is cut from rather than
  the whole recipe, so a changed recipe can be measured against a frozen gate.
- Add a platform tier to the experimental harness that measures the generic
  recipe against the structured model on the five pinned corpora. One corpus
  adapter, a frozen `docs/harness/platform-v1.json`, and
  `harness platform --corpora ROOT`, whose runs `harness verify` replays from
  saved artifacts. The recipe and the learner's arithmetic are unchanged.
- Reduce the experimental generic learner to one recipe, one module and one
  harness. `generic-memory-v2-prototype` is the only recipe and the only saved
  format; `fit(recordings)`, `predict` and `update(recordings)` are unchanged
  and still take no options. `filter_mlp` is the only sequence model. The
  dead experimental modules, the research scripts, their tests, the 85 MB of
  investigation archives and the research pages they reported are deleted, and
  `glassbox.experimental.harness` replaces the two acceptance runners with one
  frozen manifest, `run` and `verify`.
- Adopt `generic-memory-v2-prototype` as the maintained recipe of the
  experimental generic learner: the retained 100 ms explicit history plus a
  causal eight-coordinate memory over a 500 ms in-recording context that starts
  at rest, never bridges a segment boundary, and composes exactly when carried
  within a recording. The frozen M2 acceptance replaced the incumbent with a
  0.49 primary error ratio and a 0.92 ordinary-family ratio, resolving the
  delayed-input witness that explicit history cannot identify. Recipes are
  versioned: saved models carry their own recipe, `update` refits it, and
  `generic-history-v1-prototype` artifacts load, predict, and update unchanged.
  `fit`, `predict`, and `update` signatures are unchanged; forecasts from the
  v2 recipe need eleven observations of history instead of three.
- Freeze the generic learner's first engineering acceptance contract and add
  an internal bounded L-BFGS fitter with explicit budget, termination, and
  checkpoint evidence. The 25-case comparison rejects this default replacement;
  preserve the incumbent recipe and provide saved-run verification without refits.
- Recognize bounded `surface_angle_command` channels as directly actionable in
  their declared units, preserving the distinction from measured surface angles.
- Add a Cascade X8 calibration, held-out prediction and streaming tracking
  example through the shared observation/command plant interface. Update the
  optional Cascade lock to version 0.2.0 at `d6613886`.
- Add bounded aligned-transition buffering and a background refinement worker
  with explicit gap accounting, capped command history and revision retention,
  and acknowledged controller handoff. Compare frozen and adopting models during
  synthetic tracking on two platform variants in each supported family.
- Allow belief rollouts to skip propagated parameter covariance explicitly,
  returning `None` for that component while retaining the forecast mean,
  empirical error measurements, and parameter information semantics.
- Accept in-memory trajectories and mixed trajectory/path sources in `fit`,
  through the same coordinator and source resolution used by evaluation.
- Record canonical content identities for fit data and label reserved
  forecast-error calibration explicitly. Disambiguate colliding in-memory
  source names in reports.
- Add a runnable fit, evaluate, predict, and update walkthrough for both
  supported dynamics families, with separate data roles and update evaluation.
- Add an experimental recorded-telemetry refinement workflow with pinned active
  revisions, candidate scoring before absorption, session interval accounting,
  and explicit adoption of evaluated revisions. Exercise both families with
  saved model revisions, block forecasts, and an adoption ledger.

## 0.3.0rc1 - 2026-09-08

This candidate streamlines the identification workflow and corrects actuator
history, numerical derivatives, parameter evidence, and controller behavior.

### Migration from 0.2

- The main `fit(sources, FitSpec)`, `DynamicsBelief.save/load`, and `absorb`
  entry points remain. Fitted models now derive their input contract and sample
  period directly from training windows.
- `TrajectoryWindows.input_spec` replaces its separate channel metadata fields.
  Build windows with the existing extraction functions. Trajectories own their
  preceding controls; prediction and diagnostics no longer take a separate
  control-history argument.
- Canonical NPZ format 5 preserves preceding commands; formats 3 and 4 remain
  readable. Belief format 6 preserves direct-map bounds; formats 3–5 remain
  readable, and external actuator maps require explicit rebinding. Legacy
  parameter evidence that cannot be converted loads at rank zero with a warning;
  refit to recover parameter information.
- Forecasts expose `forecast_error_covariance` and `parameter_covariance`
  separately. The combined `tangent_covariance` and `tangent_standard_deviation`
  shortcuts are removed.
- Evaluation report format 2 removes `can_promote_model`.
  `independent_holdout` records the caller's split declaration.
- Install the source-only Cascade simulator with `uv sync --dev --group cascade`
  instead of `--extra cascade`. Published telemetry extras remain `px4` and `ros`.
- `one_step_innovations` lives in `core.metrics`. Pass its residuals to
  `one_step_innovation_diagnostics` and `innovation_noise`. Use
  `RolloutLossConfiguration.validity_envelope` in place of separate support arrays.
- Direct flight-supervisor callers must supply model-support utilization;
  custom supervisors must accept that keyword. Missing or exceeded support
  selects the existing latched intervention.

### Corrections

- Preserve causal actuator history through ingestion, nested holdouts, forecasts,
  parameter evidence, and updates. Correct the RK4 actuator-response quadrature,
  including the instantaneous-response limit.
- Keep rotation derivatives finite at zero, preserve feasible derivatives at
  command bounds, and retain the best evaluated full-batch optimizer iterate.
- Backtrack nonlinear belief updates, measure accepted-model innovations, and
  preserve the information rank cutoff as evidence accumulates. Keep unresolved
  parameter directions explicit in predictions and planning.
- Advance controller warm starts by one model sample, fix cache identity for
  actuator maps and bounds, and check command bounds and initial-state support.
- Invalidate interrupted holdout runs and include effective fit settings in
  resume checks. Recorded-result generation always refits its folds.
- Fix `evaluate --fit-reports` after the evaluation schema change, preserve PX4
  reception times, and repair the flown SITL command fixture.
- Suppress innovation-correlation flags below numerical resolution and use a
  fixed starting state for the recovery benchmark across CPU architectures.

### Verification

- CI builds a source distribution and wheel, installs the wheel outside the
  checkout, and exercises fitting, evaluation, persistence, and updates on
  Python 3.11, 3.12, and 3.13.
- Re-record all six corpus chains at their documented budgets, including all
  13 IDF folds. Refresh EPFL and IDF figures; other headline scores are unchanged.
- Recorded results fingerprint package sources, including adapters. Numerical
  investigations retain their executed sources and results under
  `docs/investigations/`.

## 0.2.0 - 2026-09-04

This breaking release consolidates telemetry into `Trajectory`, fitted models
and evidence into `DynamicsBelief`, and planning behind `PlanModel`.

### The public surface

`glassbox.__all__` is 41 names, down from about a hundred. A name is exported
because the README or a concept page uses it, or because it is the type of one
of their arguments or return values.

- **Telemetry.** One `Channel` type replaces three, with `Trajectory` and
  `TrajectorySpec` over canonical NPZ v4.
- **The fit.** `fit(sources, FitSpec) -> FitOutcome` is the one entry point.
  `FitSpec` replaces `FitRequest`'s twenty-two fields with a spec, a
  `LossPolicy` and a `WeightingPolicy`; `Holdout` has three rules
  (`by_label`, `by_group`, `temporal`) instead of six modes. Every fit now
  accumulates its own parameter evidence, so the belief it returns knows which
  directions the data resolved and not only how noisy its one-step
  predictions are. `glassbox fit` no longer ties that to `--model` and has no
  flag to turn it off; it costs about a fifth more on a three-flight synthetic
  fit and leaves every loss and metric unchanged.
- **The belief.** `DynamicsBelief(model, information, forecast_error)` plus
  provenance. `ExecutableModel` replaces three runtime types and is owned by
  the belief. `ParameterInformation` replaces four parameter-belief types; a
  point belief is rank zero rather than a separate class.
  `ForecastErrorEnvelope` is the held-out second moment by horizon.
  `belief.absorb(telemetry) -> (DynamicsBelief, UpdateResult)` is the update.
- **Control.** `PlanModel` is the whole interface a solver has to a model, and
  `BoundedShootingSolver` is the one solver behind it, with a compile cache
  keyed on shape rather than on belief values. `plan_model` adapts a fitted
  belief and `NMPCController` is the thin factory over both.
  `RecursiveBootstrapIdentifier` now produces a `DynamicsBelief` over the new
  `BootstrapMultirotorParams` family, so the same solver plans over a fitted
  belief and one built in flight from nothing.
  `MultirotorFlightSupervisor` takes its allocation by injection instead of
  assuming a mixer.
- **Integration.** `VehicleLink` and `run_control_loop` are one control
  interval for every integration; PX4 is a read-only link and the Cascade
  plant is a writable one, so shadow mode and closed-loop control differ by
  one property of the link.
- **Evaluation.** One `evaluate` with three named scoring policies
  (`windowed`, `x8`, `nanodrone`), one `evaluate_holdout` by label, and one
  `ReferenceCorpus` registry over the five pinned corpora.
- The `glassbox.experimental` subpackage is gone. The identifier and the
  supervisor are ordinary components of `glassbox.control`.

### The commands

Nine commands, each one line in `glassbox --help` and its whole contract in
its own `--help`: `extract`, `corpus`, `synthetic`, `fit`, `evaluate`,
`benchmark`, `record-results`, `sitl-profile` and `px4-shadow`. They replace a
tree of twenty nodes and forty-two leaves. `glassbox corpus list | fetch |
prepare` replaces the per-corpus subcommands; `glassbox extract` replaces the
`ulog` group and `scripts/extract_ulog_dataset.sh`; `glassbox evaluate`
replaces five scoring entry points; `glassbox benchmark` replaces four
benchmark leaves; `record-results --tier local|corpus` replaces
`--include-slow`. `--skip-checksum` is gone from every corpus command: a
pinned corpus that does not verify is a different corpus.

### Removed

Every removal names the last commit that can still run it; the
[literature review](docs/literature-review.md) records the finding each one
rests on. The range is `d10bb24..8f55517`.

- The frozen observation research program and its seven artifacts (`4c119a8`),
  and the observation-first initializer that outlived it (`2e16ebc`).
- The fleet parameter prior with its conditioning, initialization and `prior`
  command; plan assessment and `PlanAssessment`; the error-radius quantiles
  (all `478c063`).
- The transactional belief update, 2,006 lines, with its improvement margin,
  trust bound, line search, fingerprints, replay detection and thirty-five
  field report, and the runtime tangent bias that went with it (`8d3400f`).
- The batch bootstrap identifier, the cascade controller and thrust cascade,
  the identifier's certification transaction, and its research switches
  (`aab0b42`), with the last two switches becoming the only behaviour at
  `2f5adc2`.
- The NMPC support filter with its six modes and the bounded-authority
  post-pass (`9570fa1`); robustness moved into the objective instead.
- The research promotion machinery: policy selection, the fixed-wing gate,
  acceptance and selection, with their two pages; the predictive-ensemble
  workflow with its page and four notes; the angular-authority and
  Nano-drone-rotation sweeps; the adaptation benchmark; the dead CLI `Group`
  (all `bd48419`).
- The lagged multirotor rotational-response branch and its sentinel machinery
  (`9d59e4a`); the multirotor latent state is four wide instead of seven.
- The certified prediction horizon and its two runtime-spec fields
  (`f5656a9`); the forecast-error envelope is the only thing that caps a
  horizon.
- The singular fit path and its second report shape, `TrajectoryWindows`
  re-validation, `with_angular_dynamics_authority` (all `2e16ebc`); the
  adapter protocol (`3db7ce5`); the shadow runner's forty-key report and
  streaming evaluation (`1acdd50`); five corpus evaluation modules and
  `with_constant_angular_rate` (`c250233`); `rebind_belief` and the solver
  backend protocol (`aab0b42`); `RuntimeDynamicsBelief` (`2123c94`); the
  vectorized log-mean reduction (`68eb163`).
- Eighteen write-only fit-report keys, read by no module, test, page or
  recorded artifact: the `fit_statistics` block, `horizon_duration_s`,
  `stride_steps_by_horizon`, `training_source_group_weights` and
  `candidate_training_windows_per_flight_by_horizon` from the configuration;
  `total_duration_s`, `source_type`, `source_grouping`, `coordinate_frames`,
  `vehicle_configuration`, `profile_counts` and the three `observation_*`
  fields from the pooled dataset contract; `sample_weighted_aggregate` from
  each model's validation; and `net_displacement_m`, `position_range_xyz_m`
  and `maximum_angular_speed_rad_s` from each flight's characteristics. The
  characterization report's `fit_report.size_bytes` goes with them.
- `fit_dynamics`'s `loss_normalization_params` and
  `loss_normalization_window_sets`, which every caller passed a copy of the
  fit's own initial parameters and window sets, and the duplicate platform
  validation `resolve_dataset` already ran.
- Two recorded files that were not machine output:
  `docs/results/multirotor-profile-results.json` and the four
  predictive-ensemble notes.

### Recorded numbers that moved

Every number in `docs/results/adaptive-recovery-results.json` moved once, at
the start of the number-moving batch, and is now at format 5, method version
7. Three causes, in order of size:

- the benchmark's belief is seeded by inverting the five sibling
  configurations' sample covariance to precision, so it starts knowing one
  direction rather than declaring one uncertain and eighteen certain;
- the update is `absorb` over all forty one-step transitions instead of a
  transaction over eight windows split into a proposal half and a validation
  half;
- the held-out forecast bias is no longer applied at runtime.

Resolved rank goes from `1` to `9` of `15` estimable coordinates, the
information gain is `1.698` nats where the transaction recorded `null`, the
whitened one-step innovation falls `0.2514` to `0.0264`, and independent
0.6-second prediction RMS falls `0.033394` to `0.010120`, a ratio of `0.303`
against the transaction's `0.458`. The recovery comparison changes sign:
adapted against seeded is `0.993` tail tracking and `0.830` tail attitude and
rate, against `1.059` and `1.003` before.

Six non-timing values in `docs/results/nmpc-acceptance-results.json` moved by
between 1e-9 and 1e-8 relative in the same re-record, because the solver's
mean rollout stopped applying a zero tangent correction. The suite still
passes every check and the two ratios the documentation quotes, `0.651`
nominal and `0.552` under mismatch, are unchanged at the precision they are
quoted.

Nothing after that re-record moved a recorded local number. Three changes move
numbers no local artifact records, and each lands in the corpus tier: a
single-horizon fit moves by the loss normalization, the IDF corpus's
long-horizon folds train on a thinner window set, and the X8 scoring policy
reduces in the sequential order.

### Evidence and documentation

- One manifest describes eight artifacts in two tiers. The local tier runs in
  this repository with nothing downloaded and `record-results --check --tier
  local` runs in continuous integration; the corpus tier needs the pinned
  corpora on disk and is a maintainer job. The five corpus validation
  artifacts are new entries: the headline claims on the Nano-Quadrotor, ARP,
  IDF-DS, X8 and EPFL corpora had no artifact before this release. All
  eight are now recorded.
- Each manifest entry's `doc_page` names the section of `docs/validation.md` a
  re-record has to update, anchor included, and a test checks that both the
  file and the heading exist. The Cascade X8 assembly is pinned under its
  entry's own tolerance rather than compared for exact equality, because it
  folds in the X8 reference-model scores and moved by one ulp when those
  refit.
- The documentation is eight pages, down from twenty-two. `docs/validation.md`
  is new and is the library's evidence, every number named by the artifact and
  key it comes from. The eleven experiment pages, the recorded-results guide
  and the flight-supervisor page are folded into it, into `CONTRIBUTING.md`
  and into `docs/concepts/nmpc.md`. `docs/scope.md` restates the evidence
  standard: negative results are prose in the literature review with the
  commit that carried their code, and their code and artifacts are not kept.
- A report records a path as the command named it. Resolving inputs made two
  artifacts carry this machine's absolute paths, and made the EPFL entry
  record the directory its `artifacts/epfl_topoplane` symlink points at
  instead of the one the manifest documents; both now reproduce on any
  checkout.
- A fit report is identified by a digest of its content with the wall-clock
  keys removed, not by the bytes of the file, so a provenance digest cannot
  move with the clock while the numbers stand still. `wall_time_s` is
  declared volatile once and covers every fit block a validation artifact
  carries.
- One recursive comparison, in `glassbox.workflows.recorded`, backs both
  `record-results --check` and every pinned test, so a manifest entry's
  volatile paths and its test's ignore list cannot disagree about what counts
  as a difference.
- `docs/literature-review.md` gains a dated section recording every mechanism
  this release retired, grouped, each with its last commit.

### Fixed

- `glassbox evaluate --protocol nanodrone` no longer fails when it prints its
  summary; the per-step protocol records a null score, which the command had
  formatted as a float.
- `glassbox fit --model --report` no longer fails on a NumPy scalar.
- Fits stop on non-finite loss and return the best finite iterate with a flag.
- Physical parameter constructors validate their inputs instead of clipping.
- Holdouts follow `benchmark_split` labels when present.

## 0.1.0

Initial development snapshot.
