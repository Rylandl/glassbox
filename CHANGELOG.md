# Changelog

All notable changes to Glassbox are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## Unreleased

### Added
- Dual-control NMPC pass five (`dual_control_nmpc_pass5`): one goal over a
  one-second horizon of slew-bounded moves, with the spread propagated from the
  full-regressor planned posterior along the planned trajectory and coupled
  through `|f| sigma_tilt`, a declared maximum body rate charged as a chance
  penalty, and every multi-start seed derived from the posterior and the state
  instead of a declared amplitude ladder. `RecursiveBootstrapBelief` now exposes
  the two accumulated regression Grams the seeds and the spread read from. The
  pass is a recorded negative result: it does not recover the throw diagnostic
  on any release, and `docs/concepts/dual-control-nmpc.md` records why.
- Dual-control NMPC pass six (`dual_control_nmpc_pass6`): the fifth pass's
  horizon, slew moves, seeds, and coupling with the spread charged on the box
  average of the full regressor set, the goal charged only as far as the
  incumbent posterior can see, one descent seed per goal term, and collective-
  first probing at zero information. Measured on a sub-minute single-release
  gate, then the seven cases, then arm-only ensembles: 10 to 14 of 112
  recoveries against the fifth pass's zero, with the remaining losses split
  between early floor contacts and a lateral drift just outside the hover
  envelope. `run_throw_study_trial` accepts a `dual_config` override for
  single-release iteration, and `scripts/throw_gate.py` is that gate.
  Round two derives the knowledge term's neighbourhood from the posterior's
  own hover solution combined with the box prior, decides the goal horizon
  over the box, and advances the warm start in real time (`"step"`) instead
  of one block per interval, and offers each goal seed laid over the
  excitation cycle as a further candidate; pass five keeps the block shift.
  Committed round two recovers 54 of 112 on the arm-only ensemble, an interval
  that excludes every earlier learned arm and the working cascade and contains
  the certified cascade's point estimate. Explicit `block_lengths` are
  available and were measured as a negative result. Round three uses the
  posterior-mean maps in the rollout at the identifier's own authority, which
  stops a rank-one map from driving the goal the wrong way, and recovers 57
  of 112. `DualControlResult` exposes every multi-start candidate's objective,
  and the gate can fly any of the ensemble's own perturbed releases
  (`--draw`) and print the lowest candidates (`--candidates`). Six rejected formulations are recorded in
  `docs/concepts/dual-control-nmpc.md`.
- `RecursiveBootstrapConfig.transition_aggregation_steps` assimilates one
  sample per window of transitions, the window's means weighted by its length,
  so differenced measurement noise telescopes away while the per-transition
  counts and floors stand; one is bit-identical to before. The throw study's
  `_fly_trial` and the release ensemble accept `identifier_options`, and the
  gate accepts `--identifier key=value`. The learned throw-study arm runs a
  window of two and recovers 68 of 112 on the arm-only ensemble, above the
  certified cascade's point estimate for the first time. Round five decides
  the goal horizon from the command maps' uncertainty alone
  (`horizon_neighbourhood="box_commands"`), 69 of 112 with fewer floor
  contacts, and records four switches measured and left off: maps at face
  value at full rank, a probe overlay until supported, and the identifier's
  prequential residual in two forms.
- `RecursiveBootstrapConfig.integrated_collective` fits the collective map on
  the cumulative target with one anchor column, the exact least-squares form
  for white measurement noise on the velocity, exported to the rest of the
  identifier as an equivalent per-transition Gram; off by default and
  bit-identical when off. On the learned throw-study arm it lifts the
  state-noise case from one to nine of sixteen and the pooled recovery to 90
  of 112 on the second release distribution, above the certified cascade's
  84 on the same releases.
- The release ensemble's second distribution never throws weaker than the
  case declares (velocity scale on `[1.0, 1.2]`) and puts its width into the
  angular impulse (angular velocity scale on `[0.5, 1.5]`); earlier recorded
  tables were measured on the first distribution and are not comparable.
- Throw-study trials stop at the first floor contact and are failures from
  then on: the contact sample is kept, the contact time is recorded, and the
  terminal and settled metrics are absent rather than read off the ground.
  Recovery counts are unchanged; post-contact metrics in earlier recorded
  reports are superseded.
- The Crazyflow plant raises `CrazyflowDivergenceError`, a `ValueError`, when
  the simulator hands back a non-finite state, and the throw ensemble records
  such a release as a diverged, unrecovered trial instead of ending the run.
- `glassbox record-results` regenerates the recorded artifacts under
  `docs/results/` from one manifest, in-process, with `--list`, `--dry-run`,
  `--only`, and `--include-slow`.

### Changed
- Rollout error statistics exclude the measured initial sample and every metrics
  dict carries a `metric_policy` identifier; the minibatch objective averages
  sampled windows uniformly (`deterministic_weighted_minibatch_v3`); complete
  flight rollouts apply logged wind per step.
- Belief updates score candidates without the held-out bias against the
  bias-corrected incumbent with a noise-scaled margin, bound every whitened
  prior coordinate to one standard deviation, and condition only on numerically
  resolved directions. Conditioning, commits, and prior initialization stale
  predictive-error evidence; `recalibrate_predictive_error` rebuilds it.
- NMPC reports stalls as `STALLED`, tests convergence on the projected gradient,
  shifts warm starts by command block, and has no dead command blocks.
- Bootstrap identifiers threshold nuisance directions at 0.002 of the leading
  direction and expose nuisance ranks; the online controller returns a bounded
  unusable decision on non-finite input; the supervisor uses a geodesic tilt
  error.
- PX4 ULog ingestion resolves a separate actuator hold-age tolerance and records
  per-topic source rates and segment coverage.
- `pymavlink`, `pyulog`, and `rosbags` are optional extras (`px4`, `ros`).
- The `model_family` module is now `families`.
- Documentation is restructured around a short README with concept,
  experiment, and results directories.
- The 24 `glassbox-*` console scripts are replaced by a single `glassbox`
  command with a subcommand tree: `glassbox fit`, `glassbox ulog extract`,
  `glassbox crazyflow throw`, and so on. `glassbox --help` lists every command
  and runs with no optional extra installed, because subcommand modules are
  imported only when one is dispatched.
- `workflows.fitting` is split into a library and a front end. A frozen
  `FitRequest` carries every fitting knob, `plan_holdout` resolves the
  training/validation split on its own, report assembly moved into named
  builders, and the argparse entry point is now `cli.fit`. Fit reports are
  byte-for-byte unchanged.
- The package is organized into `core`, `belief`, `control`, `io`, `workflows`,
  `integrations`, and `cli` subpackages; the root exports the stable surface
  and `glassbox.experimental` holds bootstrap identification, online
  bootstrap, the flight supervisor, and predictive ensembles.
- Duplicated helpers are consolidated: one NumPy quaternion-to-rotation and
  Euler helper in `core.geometry`, one pinned-download helper in
  `io.pinned_download`, one persistence score in `core.evaluation`, one set of
  selection thresholds in `workflows.selection`, and one set of finite-vector,
  world-up, and thrust-cascade helpers in `control._common`; all verified
  bit-identical against recorded outputs.
- `NMPCController.solve`, the recursive bootstrap update, the progressive
  bootstrap command, and the Crazyflow prototype are split into named phases
  and modules; the prototype module shrinks from about 2,450 lines to about
  540 with `crazyflow_telemetry`, `crazyflow_fleet`, `crazyflow_online`, and
  `crazyflow_supervisor_campaign` beside it.
- Test collection drops from about 21 s to under 2 s; the three
  benchmark-scale tests carry a `slow` marker.
- The NMPC solver policy is public as `SolverPolicy`, exported from
  `glassbox.control.nmpc` and the package root; `NMPCController` takes it as
  the `policy` keyword instead of a private one. The NumPy and JAX rotation
  helpers move from `control._common` to `core.geometry`, verbatim, so no
  recorded number moves. `RecursiveBootstrapBelief`'s field list is pinned as
  a downstream contract.

### Removed
- The batch bootstrap identifier: `control/bootstrap_identification.py` with
  `BootstrapMultirotorIdentifier`, `BootstrapIdentificationConfig`,
  `BootstrapIdentificationResult`, `BootstrapExcitationConfig`,
  `BootstrapExcitationPlan`, `plan_bootstrap_excitation`,
  `BootstrapArrestCommand`, `BootstrapVelocityArrestCommand` and
  `BootstrapModelNotReadyError`; its nine `glassbox.experimental` re-exports;
  its nine tests; and the batch half of
  `docs/concepts/bootstrap-identification.md`, including the Crazyflow arrest
  result that only the batch fit and its excitation plan produced. The
  recursive identifier's Gram accumulation is the same fit, so a batch fit is
  folding N transitions and reading the belief. Last commit carrying the code:
  `aab0b42`.
- The cascade controller: `ProgressiveBootstrapController`,
  `ProgressiveBootstrapControllerConfig` and `ProgressiveBootstrapCommand` in
  `control/online_bootstrap.py` with their excitation scan and its fixed
  pseudo-random patterns, `ThrustCascade` and `thrust_cascade` in
  `control/_common.py`, their three `glassbox.experimental` re-exports, their
  three tests, and their documentation. It was the hand-gained baseline arm of
  a comparison that lives in the demo repository and was retired there by the
  learned controller. Last commit carrying the code: `aab0b42`.
- The Crazyflow throw demo moved to
  [glassbox-throw](https://github.com/Rylandl/glassbox-throw) at this commit:
  the Crazyflow plant adapter, the throw trial and study, the bootstrap and
  prototype trials, the annotated animation renderer, and the experimental
  dual-control NMPC.
- The frozen 2026-08-29 observation research program: static observation
  correction, first-order temporal filtering, state-channel timing alignment,
  and body-rate observation rollout scoring, along with its seven recorded
  artifacts, including the observation-first spike, the post-freeze
  innovation diagnostic, and the rejected residual-innovation observer. The
  literature review keeps the decision and the numbers as prose. Last commit
  carrying the code and artifacts: `4c119a8`.
- The predictive-ensemble uncertainty workflow: `workflows/predictive_ensemble.py`,
  the `glassbox ensemble-benchmark` leaf, `docs/concepts/predictive-ensembles.md`,
  its four recorded notes under `docs/results/` with their manifest entries, the
  seven `glassbox.experimental` re-exports, and `tests/test_predictive_ensemble.py`.
  Five versions ended in a clean negative on the IDF-DS corpus; the literature
  review keeps the finding as prose. Last commit carrying the code and
  artifacts: `bd48419`.
- `workflows/adaptation_benchmark.py`, its `glassbox adaptation-benchmark` leaf,
  and `tests/test_adaptation_benchmark.py`. The benchmark asserted that every
  update applied while its own report recorded the acceptance gate as failed, it
  backed no recorded artifact, and the adaptive-recovery benchmark already
  covers the belief-to-control path. Last commit carrying it: `bd48419`.
- The research promotion machinery: `workflows/policy_selection.py`,
  `workflows/fixedwing_gate.py`, `workflows/acceptance.py`,
  `workflows/selection.py`, the `glassbox select-policy` and
  `glassbox fixedwing-gate` leaves, `docs/experiments/fitting-policy.md`,
  `docs/experiments/fixedwing-gate.md`, and their three tests. The verdicts were
  already recorded and nothing in CI ran the contracts, so `profile-benchmark`
  no longer reports an accuracy-contract status. The two reusable divergence
  helpers moved into `core.evaluation` as the public `summarize_divergence` and
  `p90_horizons`; the literature review keeps the verdicts as prose. Last commit
  carrying the code: `bd48419`.
- `workflows/angular_authority.py`, `workflows/nanodrone_rotation.py`, and their
  two tests. The angular-authority selection sweep was maintainer-only and its
  numbers had no recorded artifact, so the NanoDrone and ARP paragraphs that
  rested on it are withdrawn on those pages.
  `core.dynamics.with_angular_dynamics_authority` is untouched. Last commit
  carrying the code: `bd48419`.
- The dead `Group` class in `cli/_tree.py`, the `Node` union, and the
  `isinstance` branches over them in `cli/__init__.py` and the console-script
  test: the command tree has no groups, and every leaf sits at the top level.
  Last commit carrying it: `bd48419`.
- `scripts/reextract_profile_dataset.py`, a one-off migration into canonical
  trajectory format v3, and its invocation on the PX4 SITL multirotor page.
  Corpora are re-extracted from raw telemetry instead. Last commit carrying
  it: `bd48419`.
- `RuntimeDynamicsBelief.assess_plan` and the public `PlanAssessment` type, with their four test sites and the plan-scoring
  block of `docs/concepts/dynamics-beliefs.md`. Nothing in the package called
  them, and every ingredient an exploration policy needs is already on the
  rollout and on `parameter_evidence`. Last commit carrying them: `478c063`.
- `EmpiricalHorizonPredictiveError.group_radius_quantiles` and its
  `radius_quantiles` interpolator, the matching `PredictiveTrajectory` field,
  and the `TANGENT_GROUP_SLICES` and weighted-quantile helpers behind them. No
  recorded artifact carried the key and nothing read the radii. The
  predictive-error payload is now format version 3; the loader still accepts
  version 2 and ignores the key. Last commit carrying them: `478c063`.
- The fleet parameter prior: `belief/parameter_prior.py` with
  `StructuredParameterPrior` and `initialize_belief`,
  `DynamicsBelief.condition_parameter_prior`,
  `DynamicsBelief.with_parameter_members`, the `glassbox prior` leaf, and
  `tests/test_parameter_prior.py` with the four conditioning tests in
  `tests/test_belief.py`. At the only scale it ever ran, five members over
  twenty-two parameters, 99.6 percent of the prior's normalized covariance
  trace was unit-ball assumption on the unresolved nullspace, and nothing in
  the package conditioned on it in production. An information seed replaces it
  when a real fleet exists; `LocalGaussianParameterBelief.from_members` already
  summarizes members that do span a direction. The adaptive-recovery benchmark
  now takes its configuration-delta covariance from that, so
  `docs/results/adaptive-recovery-results.json` is re-recorded at method
  version 5. Last commit carrying the code: `478c063`.

### Fixed
- `glassbox fit --model --report` no longer fails on a NumPy scalar.
- Fits stop on non-finite loss and return the best finite iterate with a flag.
- Physical parameter constructors validate their inputs instead of clipping.
- Holdouts follow `benchmark_split` labels when present.

## 0.1.0

Initial development snapshot.
