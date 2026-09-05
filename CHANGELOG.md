# Changelog

All notable changes to Glassbox are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## Unreleased

- Separate passive PX4 telemetry waits from the model's solve deadline. Repair
  the flown SITL fixture's stale CLI entry point and synchronize sampling with
  actual maneuver excitation.
- Record recovery uncertainty ablations and an offline optimizer reference.
  Additional independent telemetry removes the small-disturbance tracking gap;
  full information rank alone does not qualify a local uncertainty approximation.
- Allow up to 16 line-search trials, so newly retained parameter uncertainty
  can produce a descent step without changing the objective or its weights.
- Fix controller cache identity for custom actuator maps and changed command
  bounds; reject out-of-bounds plan output before marking a command usable.
- Bound and backtrack nonlinear belief updates; validate transformed physical
  coefficients and measure noise from the accepted model's actual residual.
- Preserve the absolute rank cutoff as information accumulates. Carry unknown
  parameter support explicitly into predictions and controller diagnostics.
  Planning with unresolved parameters now requires an explicit policy override.
  Propagate every resolved direction without applying another covariance cutoff.
- Belief format 6 preserves direct-map bounds and requires explicit rebinding
  for external actuator maps. Formats 3 through 5 remain readable.
- Share actuator history extraction between fitting and absorption, including
  prefixes on segments, and preserve PX4 reception timestamps through alignment.
- Match the first two exponential actuator-response moments in RK4, fixing the
  instantaneous-response limit and the no-lag ablation. This changes rollout,
  fit, and benchmark numbers; affected recorded artifacts are regenerated.
- Disable cached holdout reuse during result recording, and report failed
  recovery diagnostics as JSON `null` with explicit solve-status counts.

## 0.2.0 - 2026-09-04

Glassbox becomes a production library. Forty commits took the package from
about 41,700 to about 28,200 source lines and its suite from about 14,900 to
about 13,100, folded three runtime types into one, and replaced the
transactional belief update with a recursive one.
The result is one type per concept and one path per job: telemetry becomes one
canonical flight object, a fit produces one `DynamicsBelief`, `absorb` keeps
it current from live telemetry, and one bounded solver turns it into a command
every control interval, for a vehicle with a fitted model and for a vehicle
with none.

This is a breaking release. Every consumer resyncs once;
[glassbox-throw](https://github.com/Rylandl/glassbox-throw) stays pinned at
`d10bb24`, the last revision before the migration. Its resync targets
`plan_model` over the bootstrap belief and the current identifier API, which
is what its own 3,258-line controller was reimplementing, so most of that
controller goes when it lands.

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
