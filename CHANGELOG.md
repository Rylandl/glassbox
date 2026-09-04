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
- One channel type. `ControlChannel`, `ExogenousChannel` and `ObservationChannel`
  are one frozen `Channel(name, role, semantic, unit, kind, frame, minimum,
  maximum)` with `kind` in `control`, `exogenous`, `observation`, and one
  `to_dict`/`from_dict`. `TrajectorySpec` holds one ordered `channels` tuple and
  exposes `controls`, `exogenous` and `observations` as filtered views, so a
  call site that reads `spec.controls[i].name` is unchanged. `Channel` joins the
  public `glassbox` surface and the three old names leave it. Canonical
  trajectory NPZ files are written at format version 4, whose spec payload is
  one `channels` list; `load_trajectory_npz` reads version 3 as well, because
  the corpora extracted under the untracked `artifacts/` tree are version 3
  until Phase 3 re-extracts them. The per-channel `source` of an observation
  channel is gone with the three types, so
  `specific_force_observation_channels` and
  `angular_acceleration_observation_channels` take no argument; the PX4 ULog
  adapter already recorded the source topic in provenance.
  `VehicleConfigurationSpec` drops `propulsion`, which nothing read; `family`,
  `configuration_id`, `controlled_axes`, `fixed_states` and `auxiliary_controls`
  stay, the last two because the NMPC benchmark and three corpus tests read
  them.
- Three holdout rules, one value. `plan_holdout` and its six modes are one
  frozen `Holdout` with `Holdout.by_label(key, values)`,
  `Holdout.by_group(count, key="source_group")` and
  `Holdout.temporal(fraction)`, and `Holdout.plan(trajectories, paths)` returns
  the same `HoldoutPlan`. The automatic `benchmark_split` rule is gone: a label
  holdout is now requested, which is what `--holdout-label KEY=VALUE` is for,
  and the profile holdout is `Holdout.by_label("profile", ...)`. `by_group`
  falls back to whole flights in argument order whenever its label separates
  nothing, which covers both an absent label and the single-group
  characterization corpora, so `chronological_segments_within_source_group_characterization`
  is now reported as `leave_complete_flights_out` and
  `benchmark_split_holdout`/`leave_profiles_out` as `leave_labeled_out`. The
  fit report's `split` section records the rule under `holdout` and drops
  `held_out_profiles`, `benchmark_split_holdout`, `benchmark_split_training`
  and `benchmark_split_validation`; `configuration` drops
  `train_fraction_for_single_flight`, whose value the rule now carries.
  `BenchmarkSplitHoldoutConflict` is gone, since no two rules can be requested
  at once. `glassbox fit` keeps `--holdout-count`, `--holdout-profile` and
  `--train-fraction`, adds `--holdout-label`, and rejects two rules at once
  instead of silently preferring one. The X8 fit commands on
  `docs/experiments/x8.md` and in the recorded-artifact manifest now pass
  `--holdout-label benchmark_split=validation` and reserve exactly the four
  maneuvers they reserved before.
- One weighting mode. `trajectory_windows` takes `weights`, a mapping from
  group key to that group's share of the total loss weight, and `group_of`, a
  `(index, trajectory) -> str` key function defaulting to the trajectory's
  `source_group` label and otherwise its index. It replaces
  `balance_trajectories`, `trajectory_weights`, `trajectory_groups` and
  `trajectory_group_weights`, which were mutually exclusive; every one of them
  is expressible in the new form and the resulting `window_weights`,
  `trajectory_indices`, `start_indices` and selection policy are bit-identical,
  which a pinned three-flight test asserts. `FitRequest` keeps
  `balance_training_flights` and `training_source_group_weights` and expresses
  them in the new form. `FitRequest.normalization_source_group_weights` and the
  shared outer-training statistics it selected are deleted with the predictive
  ensemble that was their only caller, so the fit report's `fit_statistics`
  block keeps `policy` and `data_derived_values` and drops
  `shared_across_resampled_members`, `normalization_source_group_weights` and
  its `selected_windows_by_horizon`. Last commit carrying them: `2750399`.
- The control layer is a solver, a protocol, and one adapter.
  `glassbox.control.nmpc` is gone as a subpackage and its contents are three
  flat modules. `glassbox.control.plan` carries the public `PlanModel` protocol,
  the new `Prediction` and `PlanMeasurements` types, `SolverPolicy`,
  `SolveStatus`, `TrackingTolerances`, `SafetyEnvelope`, `ReferenceTrajectory`,
  `NMPCWarmStart`, `NMPCDiagnostics`, and the command-block layout helpers,
  which are now public as `block_steps_for`, `blocks_cover_horizon` and
  `maintained_block_count`. `glassbox.control.solver` carries
  `BoundedShootingSolver`, which imports nothing from the belief layer.
  `glassbox.control.fitted` carries `plan_model`, `FittedPlanModel` and
  `NMPCController`. `NMPCResult` is renamed `SolveResult`; its fields and its
  `command_usable` property are unchanged. `PlanModel`, `Prediction`,
  `SolveResult`, `BoundedShootingSolver` and `plan_model` join the public
  `glassbox` surface. Every recorded acceptance and recovery number is
  unchanged, which is what says the split reproduced the objective. The
  recursive identifier's belief does not get an adapter here; `PlanModel` is
  the protocol the demo's dual-control controller will target when it resyncs.
- Compiled solver kernels are cached at module scope under the plan model's
  static signature: the input and runtime specs, the tolerances, the envelope,
  the policy, the parameter tree structure, and the belief's own forecast-error
  and parameter-covariance content. The fitted parameters travel through every
  kernel as an argument rather than as part of the signature, so a re-fitted or
  re-adapted belief reuses the compiled code of the belief it came from. Two
  controllers built from one configuration used to compile twice; the second
  now compiles nothing.
- The belief owns one executable model. `DynamicsBelief.model` replaces the flat
  `params`, `input_spec` and `runtime_spec` triple, which stay as read-only
  properties delegating to the model, and the belief's serialized JSON keys are
  unchanged: a belief written before this change reloads to identical
  parameters, evidence and error moments. `ExecutableModel.actuation` is now
  optional and defaults to the identity map on the declared control channels,
  which is what every actionable model was already given; a model whose inputs
  are observations of actuation gets no map instead of refusing to exist, still
  integrates, still reports validity, still serializes, and raises
  `NonActionableModelError` from `command_size`, `command_minimum`,
  `command_maximum`, `initial_latent_state` and `transition`. `NMPCController`
  raises the same error when handed one. Constructing an `ExecutableModel` with
  the identity map no longer traces and differentiates it to check it, because
  it is the identity; a caller's own map is still checked.
- The NMPC objective charges what the belief knows about its own error, in two
  terms and with no configuration of its own. The tracking cost is now an
  expectation: every predicted stage charges `l(mean) + trace(W Sigma)` for the
  tracking weight `W` the objective already builds from `TrackingTolerances` and
  the predicted tangent covariance `Sigma`. The model-validity term charges the
  robust utilization, the mean utilization plus each envelope feature's marginal
  standard deviation, in place of the mean utilization alone. Both terms are
  exactly zero when the belief carries no covariance, so a point model is scored
  by the objective it was always scored by, bit for bit, and the recorded NMPC
  acceptance numbers are unchanged. The parameter contribution is carried
  through one forward-mode rollout per resolved parameter direction rather than
  a full parameter Jacobian. Every recovery number in
  `docs/results/adaptive-recovery-results.json` moved.
- One executable model type, owned by one belief. `core/runtime.py` becomes
  `core/model.py` and `RuntimeDynamicsModel` becomes `ExecutableModel`, keeping
  its fields and methods; `RuntimeModelSpec`, `ModelValidityEnvelope`,
  `ActuationMap`, `DirectActuationMap`, `NonActionableModelError`,
  `runtime_spec_from_trajectory` and `runtime_spec_from_fit_report` move with it
  under their own names. `ExecutableModel.rebind_parameters` keeps its
  structure, shape, dtype and finiteness checks and then uses
  `dataclasses.replace`. `RuntimeDynamicsBelief` is now a frozen view holding a
  `DynamicsBelief` and the `ExecutableModel` compiled from it: its `nominal`
  field is `model`, and `predictive_error`, `parameter_belief`,
  `predictive_error_parameter_update_count` and `predictive_error_current`
  delegate to the belief instead of being re-implemented. `compile_for_nmpc` is
  unchanged for callers.
- The belief is the only artifact the library writes.
  `core.model_io.save_dynamics_model` is deleted; the profile and source-group
  benchmarks write a point belief through `belief_io.save_dynamics_belief`.
  `model_payload` and `parameter_dict` stay where they were. As a temporary
  tolerance, `load_dynamics_belief` reads a bare nominal-model payload and wraps
  it as a belief with no predictive-error and no parameter evidence, because
  model-only artifacts written before this change still exist under the
  untracked `artifacts/` tree; the tolerance is removed once those are
  re-recorded.
- Four modules move out of `core` to the layer that owns them, with no logic
  change: `glassbox.core.px4_frames` is now `glassbox.io.px4_frames`,
  `glassbox.core.streaming_evaluation` is now
  `glassbox.integrations.streaming_evaluation`, `glassbox.core.linearization`
  is now `glassbox.belief.linearization`, and `glassbox.core.covariance` is now
  `glassbox.belief.covariance`. The linearization move removes a core-to-belief
  import inversion. None of these modules were on the public `glassbox` surface.
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
- `glassbox fit --fixed-response-time-constant` now runs the ordinary
  multi-flight path with the response time pinned, so it accepts more than one
  trajectory, `--training-horizons`, `--model-class structured_residual` and
  `--model`, and writes the one fit-report shape with held-out predictive error
  and parameter evidence. It now requires `--skip-no-lag-ablation`, since the
  no-lag ablation pins the same constant, and the four rejections it used to
  raise for the flags the singular path could not accept are gone.
  `FitRequest` and `fit_trajectory_artifacts` carry the constant as
  `fixed_motor_time_constant_s`, and the fit report's `configuration` records
  it as `fixed_response_time_constant_s`, null when the response time is
  learned. No flag was removed.

### Removed
- `RuntimeDynamicsBelief` and `DynamicsBelief.compile_for_nmpc`. The belief now
  holds the executable model directly, so the compiled view had nothing left to
  hold: `rollout`, `corrected_state`, `error_moments`, `maximum_error_horizon_s`,
  `uncertainty_available`, `predictive_error_available` and
  `parameter_uncertainty_available` are methods and properties of
  `DynamicsBelief` itself, with the same signatures and the same numbers.
  `RuntimeDynamicsBelief` leaves the public `glassbox` surface. Last commit
  carrying it: `2123c94`.
- The NMPC support filter and the bounded-authority post-pass, and with them
  every field only they wrote. `SupportFilterMode` and its six members are gone
  from `glassbox` and `glassbox.control.nmpc`; `_select_support_command` and the
  candidate enumeration, the batched candidate kernel, the actuator-reaction
  horizon, and `_bounded_authority_plan` are gone from the solver. `SolveStatus`,
  the bounded hold on every failure, `hold_reference`, warm starts, the
  certified-horizon cap, latent state, and exogenous forecasts are unchanged.
  `NMPCDiagnostics` drops `final_gradient_inf_norm`,
  `command_authority_fraction`, `uncertainty_aware_command_selection`,
  `model_uncertainty_available`, `prediction_error_model_available`,
  `prediction_error_model_current`, `prediction_error_horizon_supported`,
  `parameter_uncertainty_available`, and the twelve support fields; the twelve
  that remain describe the plan that was returned. The PX4 shadow report drops
  the support rows and records the predicted spread instead (schema version 6),
  and the adaptive-recovery artifact drops the authority and support rows,
  reports `inside_support_step_count` and `outside_support_step_count` from the
  states the solves actually saw, and is at method version 6. Last commit
  carrying them: `9570fa1`.
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
- The recursive identifier's certification transaction:
  `RecursiveBeliefValidationReport` with its `glassbox.experimental` re-export,
  the pending-proposal machinery, and the `RecursiveBootstrapConfig` fields
  `minimum_certification_interval_count`, `validation_interval_count`,
  `minimum_validation_improvement`, `maximum_model_movement_fraction` and
  `proposal_cooldown_interval_count`. `RecursiveBootstrapIdentifier` now
  exposes one belief, `belief`, and the properties named for the transaction
  are gone: `certified_belief`, `predictive_belief`, `control_belief`,
  `pending_proposal`, `validation_history`, `accepted_update_count`,
  `rejected_update_count`, and the five `shadow_*` readings. It was the same
  gate twice; authority per direction is what governs, and the learned
  controller in the demo repository flew the working belief with authority
  scaling. Last commit carrying it: `aab0b42`.
- `RecursiveBootstrapConfig.control_model` and the working-versus-certified
  distinction it selected, with `flies_working_belief`, `control_model_ready`
  and `working_support_reached`. `working_belief_supported` is the one support
  question the identifier answers. Last commit carrying them: `aab0b42`.
- The three identifier switches measured worse on the release ensemble and
  shipped off: `staged_regressors` with `staging_sample_multiple` and the four
  belief fields `collective_nuisance_staged`, `angular_nuisance_staged`,
  `collective_staging_interval_count` and `angular_staging_interval_count`;
  `enforce_collective_sign` with the two belief fields
  `collective_sign_projection_count` and
  `collective_sign_projection_magnitude`; and `prequential_residual` with the
  prequential error accumulator and the concept page's section on it. Each
  becomes the default-off behaviour with no flag, so
  `RecursiveBootstrapBelief` drops from thirty-nine fields to thirty-three and
  `to_dict` loses the matching keys. Last commit carrying them: `aab0b42`.
- `RecursiveBootstrapConfig.forgetting_factor`, which was pinned to 1.0 and
  rejected at any other value. The accumulation is the constant behaviour and
  never decays. Last commit carrying it: `aab0b42`.
- `NMPCController.rebind_belief` with the backend `rebind` it delegated to,
  its seven structural-equality checks and the `_parameter_tree_signature`
  helper, the `_SolverBackend` Protocol the controller held its one backend
  behind, their two tests, and the rebinding paragraph in
  `docs/concepts/nmpc.md`. Nothing in the package or its tests rebound a
  belief, and only one backend was ever written, so the indirection carried no
  second implementation. No numerical path changes. Last commit carrying them:
  `aab0b42`.
- The observation-first initializer: `workflows/observation_identification.py`
  with `fit_multirotor_observations`, `ObservationFitResult` and
  `AlignmentDiagnostic`, the `_observation_fit` helper and the
  `observation_initializer` parameter of the fitting workflow, the
  `observation_identification` block written into both fit reports, and
  `tests/test_observation_identification.py`. The stage initialized nothing:
  its parameters were discarded at both call sites, and the forty report keys
  it emitted were read by nothing in the package, its tests, or the docs. The
  literature review keeps the negative promotion result as prose, and the PX4
  ULog guide now says only that typed sensor channels are recorded and unused
  by the fitter. Last commit carrying the code: `2e16ebc`.
- `core.dynamics.with_angular_dynamics_authority` and its three tests. The
  angular-authority sweep that selected with it was removed at `bd48419` and
  the two doc paragraphs resting on it were withdrawn then, which left the
  transform with no caller. It set no parameter field of its own, so no model
  payload or report changes. `with_constant_angular_rate` stays: the published
  nanodrone protocol's baseline arm uses it. Last commit carrying it:
  `2e16ebc`.
- The re-validation in `TrajectoryWindows.__post_init__`: the shape, dtype,
  finiteness, uniqueness and coverage checks over the window arrays and their
  channel labels. `trajectory_windows` is the one constructor and had already
  validated the same inputs, so the class was checking its own output. The
  optional defaults and the dtype normalization stay, unchanged. No test
  asserted any of the deleted raises. Last commit carrying them: `2e16ebc`.
- The singular fit path: `workflows.fitting.fit_trajectory_artifact`, the
  `SINGLE_FLIGHT_INTERPRETATION` string, and the second fit-report shape it
  produced, with the `trajectory`, `source`, `validation_rollout` and
  `validation_innovation` blocks nothing else read. Its one caller was
  `glassbox fit --fixed-response-time-constant`, which now runs the
  multi-flight path like every other invocation, and the multi-flight path
  already splits one trajectory temporally. `runtime_spec_from_fit_report` and
  `glassbox fit` lose their branches over the two report shapes. Last commit
  carrying it: `2e16ebc`.

### Fixed
- `glassbox fit --model --report` no longer fails on a NumPy scalar.
- Fits stop on non-finite loss and return the best finite iterate with a flag.
- Physical parameter constructors validate their inputs instead of clipping.
- Holdouts follow `benchmark_split` labels when present.

## 0.1.0

Initial development snapshot.
