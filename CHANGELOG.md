# Changelog

All notable changes to Glassbox are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## Unreleased

### Added
- `glassbox.BootstrapMultirotorParams` is the bootstrap parameterization as a
  model family: the collective map
  (`collective_acceleration_per_command`, `collective_velocity_coefficient`,
  `collective_intercept_m_s2`), the angular maps
  (`angular_acceleration_per_command`, `angular_rate_coefficient`,
  `angular_rate_product_coefficient`, `angular_intercept_rad_s2`), and
  `hover_command()` derived from the collective map rather than fitted. It
  joins the `ModelParams` union in `glassbox.core.dynamics`, and its forty-one
  structured parameter names, in that field order, are the coordinate system
  every information state over the family is stated in. Its `state_derivative`
  is the identifier's own prediction on the thirteen-wide canonical state:
  body-`z` specific force from the collective map, body angular acceleration
  from the angular maps, and exactly zero on the other two body force axes,
  which the identifier never modeled.
- `glassbox.core.families.BOOTSTRAP_MULTIROTOR_FAMILY` registers it under the
  platform `multirotor_bootstrap`, with the four motor commands and the
  applied-command latent layout. The family models no actuator lag:
  `glassbox.core.dynamics.models_actuator_lag(params)` says so, the latent
  applied command is the command, and
  `latent_response_time_constants` returns an empty array rather than a zero.
  `DynamicsModelFamily.has_one_control_layout` replaces the multirotor
  platform test that decided whether control names and roles are checked
  against the declared order; it is true for any family with no optional
  roles, which is both multirotor families.
- `glassbox.BootstrapEvidence` is what the recursive identifier's own
  thresholds say about one accumulated fit: the two Grams in its own feature
  order, the residual scales, the ranks and support projectors, the per-axis
  authority, the exploration completion, and the hover command when the
  collective map implies one inside the command box. `supported` and
  `has_any_control_authority` are properties on it, and `to_dict()` records it.
  `RecursiveBootstrapIdentifier.evidence` returns it, and the same payload is
  in `belief.provenance["bootstrap_evidence"]`.
- `RecursiveBootstrapConfig.sample_period_s` declares the control period the
  identified model is executed at, defaulting to `0.01`. Every transition is
  still assimilated at its own measured interval; this is the period the
  produced belief's runtime contract declares, so a plan model over it has a
  stable timing contract from the first sample.
- `glassbox.control.identifier.TRANSITION_AGGREGATION_STEPS` and
  `BOOTSTRAP_VALIDITY_ENVELOPE` are the two constants the identifier now
  declares instead of configuring: the window width, two, and the wide
  operating envelope the bootstrap parameterization claims, which is wide
  enough never to bind because the family is affine in body velocity and body
  rate with no saturation and the identifier measures no supported region.
- `glassbox.core.model_io` reads and writes the family:
  `BOOTSTRAP_MODEL_TYPE` is `recursive_bootstrap_multirotor_command_effects_v1`
  at model format 4, `BOOTSTRAP_PARAMETER_NAMES` is its payload contract, and
  the parameters are recorded in double precision because they are direct
  estimates in physical units. A bootstrap belief therefore saves and loads
  through `belief_io` like any other.
- `glassbox.ParameterInformation` is the belief's account of what it knows:
  `names`, an accumulated `precision` over the structured coefficient block,
  the per-coordinate `scale` the rank test is stated in, the `estimable` mask
  the fitter declares, the one-step `innovation_noise` `R` with the declared
  `noise_floor` beside it, an `effective_count`, and a
  `rank_relative_tolerance`. `resolved_rank()` and `resolved_subspace()` say
  what the evidence resolved; `covariance()` is the pseudo-inverse of the
  precision on that subspace, exactly zero along every direction it does not
  resolve; `authority(direction)` is in `[0, 1]`, one along the best-resolved
  direction and zero off the resolved subspace; `information_gain_nats(delta)`
  prices an increment along the directions already resolved; `unknown(params)`
  is the rank-zero point estimate; and
  `seeded_from_members(nominal, members)` inverts the members' sample
  covariance on its supported subspace to precision, which is the forty-line
  replacement for the deleted fleet prior. It lives in
  `glassbox.belief.information`, together with the folded-in
  `SupportedCovariance`, `supported_covariance`, `structured_parameter_scale`
  and `estimable_structured_parameters`.
- `glassbox.ForecastErrorEnvelope` is the held-out forecast-error second moment
  by horizon in the twelve rigid-body local coordinates, with
  `covariance_at(horizon_s)` and `maximum_horizon_s`. It lives in
  `glassbox.belief.forecast_error` with `EmpiricalErrorSample`,
  `mean_error_by_horizon`, `HorizonEndpointErrorEvidence` and
  `endpoint_error_evidence_by_horizon`.
- `DynamicsBelief.absorb(telemetry) -> (DynamicsBelief, UpdateResult)` in
  `glassbox.belief.update` is the recursive information update. It takes
  one-step windows at the belief's own sample period, drops any sample that is
  non-finite or outside the model's validity envelope, adds
  `sum_w J_w' R^-1 J_w` to the accumulated precision, and steps by that
  precision's pseudo-inverse applied to the whitened innovation, which is
  exactly zero along every unresolved direction and is bounded by the current
  covariance rather than by a declared trust region. Information accumulates
  and is never discounted. The realized error can raise the noise floor only by
  the part the step did not explain, so the error an empty belief makes is
  explained away by its own first step instead of being recorded as irreducible
  noise. `glassbox.UpdateResult` carries `absorbed`, `reason`, `window_count`,
  `innovation_rms_before`, `innovation_rms_after`, `information_gain_nats`,
  `step_norm_prior_sigma` and `maximum_validity_utilization`; the belief's
  `provenance` accumulates `update_count` and
  `parameter_distance_since_measurement`.
- `DynamicsBelief.support` is the model's own validity envelope, and
  `DynamicsBelief.sample_period_s`, `update_count` and
  `parameter_distance_since_measurement` are properties.
- `core/diagnostics.py::one_step_innovations(params, trajectory)` returns every
  interval's one-step innovation in the twelve local coordinates. It is the
  array the diagnostics report already summarized, and it is now also the
  belief's noise model.
- `core/geometry.py::state_plus_tangent(state, tangent)` is the retraction
  inverse to `rigid_body_local_error`. It is the body of the deleted
  `apply_tangent_correction`, moved to the module that owns the log map so the
  tangent contract is complete in one place.
- `belief/parameter_evidence.py::innovation_noise(params, flights)` measures the
  noise model, and `parameter_information(model, trajectories, groups, ...)`
  accumulates the precision. The fit report's `models[*].validation` gains
  `innovation_noise` and `held_out_mean_tangent_error`.
- `tests/test_absorb.py` pins the update: a direction the evidence cannot
  resolve receives no step, a well-resolved direction moves less than a poorly
  resolved one for the same innovation, information accumulates and never
  decays, the realized noise rises only by what the step could not explain,
  out-of-envelope and non-finite telemetry are refused with a reason, absorb is
  a pure function of the belief it is given, and a fit-then-absorb round trip on
  a synthetic vehicle whose configuration changed reduces the one-step
  innovation and raises the resolved rank from zero.
- `glassbox.control` exports the in-flight identifier and the command
  supervisor: `RecursiveBootstrapIdentifier`, `RecursiveBootstrapConfig`,
  `RecursiveBootstrapBelief`, `RecursiveBootstrapSampleReport`,
  `MultirotorFlightSupervisor`, `MultirotorSupervisorConfig`,
  `SupervisedCommand`, `SupervisorMode` and `SupervisorReason`. They are
  library components with the same standing as the solver: what they compute
  is a belief and a bounded command, not an experiment. The six the README or
  a concept page names are also exported from the root.
- The `glassbox` command tree is nine commands: `extract`, `corpus`,
  `synthetic`, `fit`, `evaluate`, `benchmark`, `record-results`,
  `sitl-profile` and `px4-shadow`. Each has one summary line in
  `glassbox --help` and its full contract in its own `--help`, and `cli/` is
  one module per command plus the static tree.
- `glassbox extract LOG... OUT [--family multirotor|fixedwing] [--inspect]`
  replaces `ulog inspect`, `ulog extract` and `ulog extract-fixedwing`. One
  log writes one NPZ; several logs write `OUT/<log stem>_<state source>.npz`
  each, which is what `scripts/extract_ulog_dataset.sh` looped to do.
  `--inspect` prints what `ulog inspect` printed.
- `glassbox evaluate` replaces `nanodrone evaluate`, `x8 evaluate`,
  `epfl evaluate`, `profile-benchmark` and `source-benchmark`. It scores one
  model (`MODEL NPZ...`), several named models (`--model NAME=PATH`, repeated),
  a leave-one-label-out sweep (`--hold-out KEY`), or models one fit already
  measured (`--fit-reports`). `--corpus NAME` checks the flights against that
  corpus's published evaluation split and records its citation.
- `glassbox.workflows.evaluate.evaluate_models(models, trajectories, ...)`
  scores several named models on one held-out set under one policy and reports
  every ordered pair's comparison. The X8 multi-model report assembled in
  `cli/x8.py` is this function, so the cascade benchmark and the manifest keep
  reading `models[name]["aggregate"]` and `score_vs_kinematic_persistence`.
- `glassbox benchmark nmpc | recovery | cascade-x8 [--diagnose]` replaces
  `nmpc-benchmark`, `adaptive-recovery`, `x8 evaluate-cascade` and
  `x8 diagnose-cascade`.
- `glassbox synthetic OUT --family multirotor|fixedwing` replaces
  `fixedwing-synthetic` and writes both families' synthetic corpora.
- `glassbox record-results --tier local|corpus` replaces `--include-slow`.
  The local tier is what runs in this repository with nothing downloaded; the
  corpus tier is the maintainer job that needs a pinned corpus on disk. A
  `--dry-run` with neither `--only` nor `--tier` prints the plan for every
  artifact in the manifest.
- `glassbox sitl-profile PROFILE --family multirotor|fixedwing` replaces
  `sitl-profile` and `fixedwing-sitl-profile`, and
  `scripts/record_sitl_profiles.sh --family` replaces the two recorder
  scripts. Its `baseline` profile is the old `scripts/record_sitl.sh`: PX4's
  own takeoff and landing, extracted at 250 Hz from `actuator_outputs_sim`
  and fitted.
- `glassbox px4-shadow` is `px4-nmpc-shadow` renamed.
- `glassbox.io.corpus` is one registry of the pinned reference corpora.
  `ReferenceCorpus(name, citation, files, adapter, extra, protocol, ...)`
  carries a `Citation(doi_or_url, license, pinned_version)`, a table of
  `PinnedFile(url, relative_path, size_bytes, digest, algorithm)`, the concrete
  adapter class that parses them, the optional extra that adapter needs, and
  the published scoring protocol its evaluation uses. `fetch(dest)` and
  `prepare(dest)` are implemented once over `io/pinned_download.py` for all
  five corpora, and `REFERENCE_CORPORA` maps `nanodrone`, `arp`, `idf`, `x8`
  and `epfl` to their entries. An entry names its parser module rather than
  importing it, so `glassbox corpus list` renders with no optional extra
  installed.
- `glassbox corpus list | fetch NAME DIR | prepare NAME DIR` replaces the
  fetch, prepare, inspect, extract and extract-dataset subcommands of
  `glassbox nanodrone`, `glassbox x8`, `glassbox epfl`, and
  `glassbox ulog prepare-arp` and `prepare-idf`. `prepare` writes verified
  sources under `DIR/raw` and canonical trajectories under `DIR/canonical`,
  preserving the upstream split as subdirectories where the corpus publishes
  one, and `--raw` reuses an already-verified source tree so a second canonical
  copy of one corpus does not download it twice.
- `ReferenceCorpus.load_evaluation_trajectories(paths)` loads a set of
  trajectories and checks it against the corpus's published evaluation split,
  which the entry also describes in `validation_split`. The X8 and Nano-drone
  evaluations get their split contract from the registry rather than importing
  it from an adapter module.
- `glassbox.workflows.evaluate.evaluate(belief_or_params, trajectories,
  protocol=..., horizons_s=..., maximum_horizon_steps=...,
  independent_holdout=..., report_path=...)` scores one model on held-out
  flight under one of three named `ScoringPolicy` values in `PROTOCOLS`:
  `windowed`, `x8` and `nanodrone`. Every report it writes carries `protocol`,
  `baseline`, `stride`, `floors` and `independent_holdout`, plus the whole
  policy under `scoring`, so two numbers produced under different conventions
  can be told apart. `can_promote_model` restates `independent_holdout`: a
  same-flight characterization cannot promote a model.
- `evaluate_fit_reports` scores models whose held-out horizon tables one fit
  already measured, against the policy's baseline over the same flights. This
  is what the EPFL same-flight characterization is: two fit reports, one
  baseline, and `can_promote_model: false`. `baseline_horizon_rollouts` and
  `score_against_baseline` expose the two halves for callers that hold their
  own metrics, and `save_report` writes any of them.
- `glassbox.workflows.holdout.evaluate_holdout(trajectories, hold_out=...,
  spec=..., output_dir=..., resume=True)` is one leave-one-label-out runner.
  `hold_out` names the label the folds are keyed by, as the key itself or as
  the `Holdout` rule carrying it. `profile-benchmark` and `source-benchmark`
  both call it, and both gain the other's flags: the profile leaf gains
  resume, and the source-group leaf gains every training knob. Each keeps its
  own `--steps` and `--learning-rate` defaults, and both gain `--hold-out` and
  `--no-resume`.
- `io/x8_reference.load_validation_trajectories` and
  `io/nanodrone_reference.validate_benchmark_test_trajectories` hold the two
  corpus split contracts, which belong with the corpus rather than with the
  scorer.
- `glassbox.predict(params, trajectory)` and
  `glassbox.predict_windows(params, trajectory, horizon_steps=..., stride=...)`
  are the two prediction entry points, returning a `RolloutPrediction` that
  carries the predicted and measured states, the scored duration and the sample
  interval. `glassbox.rollout_metrics(prediction)` scores one of them, and
  `RolloutPrediction.endpoint_tangent_errors()` returns each window's final-step
  error in the twelve rigid-body local coordinates. All three names, and
  `RolloutPrediction`, are public.
- `FitSpec.diagnostics` and `glassbox fit --diagnostics` run the one-step
  innovation diagnostics on every held-out flight. They default to off, and the
  fit report records the choice as `configuration.diagnostics`.
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
- `glassbox.fit(sources, spec=FitSpec()) -> FitOutcome` is public Python. It
  returns the belief the fit supports, the report that records how it was
  produced, and one belief per requested ablation, so a caller no longer
  assembles a belief out of a report. `fit`, `FitSpec`, `FitOutcome`,
  `Holdout`, `LossPolicy` and `WeightingPolicy` join the public `glassbox`
  surface, and `import glassbox` still loads only core, belief and control.
- `glassbox.integrations.loop` is the one control interval. `VehicleLink` is a
  vehicle a loop can read an `Observation` from and, when it is `writable`,
  hand a bounded command to; a read-only link raises from `write` rather than
  accepting a command it will never transmit. `run_control_loop(link,
  controller, supervisor=None, *, steps, reference, on_sample=None)` reads the
  link, solves from the previous interval's warm start with the controller's
  sample period as both the read timeout and the solver deadline, supervises
  the candidate when a supervisor is given, writes it when the link accepts
  writes, and returns a `LoopSummary`. Every interval reaches the caller as a
  `LoopSample` through `on_sample` as it happens, so a long run costs whatever
  the caller keeps rather than a growing document. A failed solve never ends a
  run: the solver's bounded hold is what the loop records and passes on.
- `Observation` carries the state, the command the vehicle was applying, the
  host receive clock, and the alignment diagnostics a link measures while
  pairing them: `source_time_s`, `message_skew_s`, `receive_age_s`,
  `source_clock_lag_s`, `applied_command_skew_s` and `armed`.
- `PX4MavlinkLink` is the read-only PX4 `VehicleLink`. It pairs canonical state
  with either a declared fixed command or the vehicle's own actuator
  telemetry, refuses a pair further apart than its alignment limit, and raises
  from `write`. `px4_shadow_link` builds one from a fitted artifact.
- `BoundedShootingSolver.sample_period_s` and `NMPCController.sample_period_s`
  report the control interval a solver plans on, which is what a loop needs
  from a controller.

- `glassbox.PlanValues(parameters, covariance_factor,
  forecast_error_covariance)` is everything a compiled solve kernel reads as an
  argument rather than as a traced constant. It is what a `PlanModel` carries
  as `values` and what every kernel entry point takes in place of the bare
  parameters, so an implementer states its numbers in one object.

### Changed
- The solver's compile cache keys on shape, never on a belief's values.
  `PlanModel.parameters` becomes `PlanModel.values`, a `PlanValues` carrying
  the parameters, the factor of the resolved parameter covariance, and the
  forecast-error covariance at each predicted stage; `initial_latent` and
  `rollout` take one in place of the parameters. `BeliefPlanModel.parameters`
  and `.covariance_factor` are `values.parameters` and
  `values.covariance_factor`, and `_compile_signature` hashes the factor's
  shape, the stage-covariance table's shape and the information state's
  resolved rank instead of the factor's entries and the whole serialized
  forecast-error envelope. Before this, a belief that absorbed telemetry every
  interval got a new signature every interval and rebuilt the solve kernel:
  measured on a synthetic quadrotor whose belief had already saturated at rank
  13, building a plan model, absorbing one flight, and building a second plan
  model compiled two kernel sets, and the second solver's first solve cost as
  much as the first. The two now share one signature and one compiled kernel
  set, and the second solver's first solve is more than two orders of
  magnitude cheaper than the first, which is a warm solve rather than a
  rebuild. A belief of a different resolved rank still gets its own signature
  and compiles. The solve itself is bit-identical either way: on that vehicle
  the command, both objectives, the iteration count, the two horizon maxima,
  and the exclusive-or of the bit patterns of every predicted state and
  command agree exactly before and after. No recorded number moves.
- `RecursiveBootstrapIdentifier.update` and `.belief` return a
  `glassbox.DynamicsBelief` over `BootstrapMultirotorParams` instead of a
  `RecursiveBootstrapBelief`. The estimator is unchanged, number for number:
  the Gram accumulation, the Schur-complement support fit, the residual floors,
  the hover-command solve and every authority scalar are bit-for-bit what they
  were. What changed is the container. The eight parameter fields are
  `belief.model.params`; `effective_interval_count` is
  `belief.information.effective_count`; everything else is
  `RecursiveBootstrapIdentifier.evidence`. `belief.information.precision` is
  the two Grams congruence-transformed into the parameters' own coordinates and
  divided by each regression's residual variance, with the angular Gram
  entering once per body axis at that axis's own residual, so the collective
  block is eight coordinates and each angular axis eleven, forty-one in all,
  every one of them estimable. `belief.information.scale` declares the
  normalized coordinate the rank test is stated in: one unit is the coefficient
  perturbation that moves that regression's prediction by one residual standard
  deviation at the reference excitation. `belief.forecast_error` is `None`,
  because the identifier holds nothing out.
- `glassbox.plan_model` now accepts both families. A belief the identifier
  built in flight goes through the same adapter and the same
  `BoundedShootingSolver` as a belief fitted from a corpus, charging the
  tangent covariance from `belief.information.covariance()` through the same
  factored path; the solver never learns which family it is planning over. A
  rank-zero bootstrap belief resolves nothing, contributes exactly zero spread,
  and is priced by the point objective.
- `glassbox.control.fitted.FittedPlanModel` is `BeliefPlanModel`. The class
  serves both families, so a name that says "fitted" would be wrong; the module
  keeps its path.
- `TrackingTolerances.for_platform` returns the multirotor defaults for
  `multirotor_bootstrap`, and `default_solver_policy` gives both multirotor
  families the same 0.6-second, 40-step horizon cap. The horizon is a property
  of the vehicle rather than of how its model was obtained.
- `physics_parameters`, `with_thrust_command_offset` and
  `with_diagonal_angular_control` refuse any structured block that is not the
  fitted multirotor one, rather than only the fixed-wing one, and their
  messages say so. `zero_response_time_gradient`,
  `zero_thrust_command_offset_gradient` and
  `zero_angular_cross_coupling_gradient` return the parameters unchanged for a
  family that has no such leaf.
- `docs/concepts/bootstrap-identification.md` is written around the identifier
  producing a `DynamicsBelief`, and states the window and the integrated
  collective as behaviour rather than as switches.
  `docs/concepts/dynamics-beliefs.md` records that one belief type carries
  either family. The `ParameterInformation` docstring states what one unit of
  precision means for the bootstrap collective block, where the exported Gram
  is rescaled to the declared force floor.
- `docs/results/adaptive-recovery-results.json` was re-recorded at format 5 and
  method version 7, and every number in it moved. Three causes, in order of
  size. The benchmark's belief is now seeded by inverting the five sibling
  configurations' sample covariance to precision, so it starts knowing one
  direction rather than declaring one direction uncertain and the other
  eighteen certain. The update is `absorb` over all forty one-step transitions
  instead of the transaction over eight windows split into a proposal half and
  a validation half. And the held-out forecast bias is no longer applied at
  runtime, so the envelope is the uncentered second moment and a moved
  parameter vector no longer stales it. Resolved rank goes from `1` to `9` of
  `15` estimable coordinates, the information gain is `1.698` nats where the
  transaction recorded `null`, the whitened one-step innovation on the absorbed
  telemetry falls `0.2514` to `0.0264`, and independent 0.6-second prediction
  RMS falls `0.033394` to `0.010120`, a ratio of `0.303x` against the
  transaction's `0.458x`. The recovery comparison changes sign: adapted against
  seeded is `0.993x` tail tracking and `0.830x` tail attitude/rate, against
  `1.059x` and `1.003x` before, and adapted against oracle is `1.078x` and
  `1.049x`, against `1.123x` and `1.223x`. Maximum actual validity utilization
  is `1.085`, `1.056`, `1.091`, `1.136`. The `oracle_mean_point` arm, which
  carries no evidence at all, moved only by between 1e-8 and 1e-7 relative,
  because dropping the bias correction also drops a quaternion multiply by the
  identity and a renormalization from every predicted stage. The
  `evidence.adaptation` block is `UpdateResult`'s eight fields in place of
  `BeliefUpdateReport`'s thirty-five, `evidence.fleet` records the seed and
  posterior rank, both covariance traces and the measured noise model, the
  `semantics` keys describe an absorb rather than a transaction, and the two
  per-trace fields `predictive_error_current` and
  `parameter_uncertainty_available` are `forecast_error_available` and
  `parameter_information_rank`.
- `docs/results/nmpc-acceptance-results.json` was re-recorded. It builds point
  models, which carried no bias, so nothing about it moved structurally. Six
  non-timing values moved by between 1e-9 and 1e-8 relative, for the same
  reason the oracle recovery arm did: the solver's mean rollout no longer
  applies a zero tangent correction, and that correction was a quaternion
  multiply by the identity followed by a renormalization. The suite still
  passes every check and the two numbers `docs/concepts/nmpc.md` quotes,
  `0.651x` nominal and `0.552x` mismatch, are unchanged at the precision they
  are quoted. `docs/results/cascade-x8-validation-results.json` is untouched.
- The belief is three objects: `DynamicsBelief(model, information,
  forecast_error, provenance)`. `ParameterInformation` replaces
  `LocalParameterInformation`, `LocalGaussianParameterBelief` and
  `PointParameterBelief`, which are gone along with `UnavailableParameterEvidence`,
  `ParameterBelief`, `ParameterEvidence`, `parameter_belief_from_dict`,
  `parameter_evidence_from_dict` and `ResolvedLocalGeometry`; a point belief is
  rank zero rather than a separate type. `ForecastErrorEnvelope` replaces
  `EmpiricalHorizonPredictiveError` and `UnavailablePredictiveError`, which are
  gone along with `PredictiveErrorModel` and `predictive_error_from_dict`; an
  absent envelope is `None` rather than a type that answers zero. The
  `DynamicsBelief` fields `predictive_error`, `parameter_belief`,
  `parameter_evidence` and `predictive_error_parameter_update_count` and the
  properties `predictive_error_available`, `predictive_error_current`,
  `parameter_uncertainty_available` and `error_moments` are gone;
  `error_covariance(horizon_s)` is what a caller reads now. Last commit
  carrying the replaced types: `8d3400f`.
- `ErrorCovarianceScope` is deleted and with it every branch keyed on it. The
  fit measures one-step innovation covariance and that is the only kind of
  noise the belief has. `LocalParameterInformation`'s `group_score_vectors`,
  `group_score_second_moment`, `score_vector`, `unresolved_direction_basis`,
  `center`, `horizons_s`, `window_count_by_horizon`,
  `residual_precision_rank_by_horizon`, `group_labels`,
  `independent_group_count`, `trajectory_count` and the three provenance
  strings go with it; the sandwich inputs backed no production reader.
- The runtime forecast bias is gone. `EmpiricalHorizonPredictiveError.tangent_bias`,
  `apply_tangent_correction`, `DynamicsBelief.corrected_state` and the `bias`
  argument of every entry point in `belief/linearization.py` are deleted, and
  the forecast-error envelope is the **uncentered** second moment of the
  held-out endpoint error rather than a covariance about a bias that nothing
  applies. The held-out mean error is recorded as
  `models[*].validation.aggregate.held_out_mean_tangent_error` and read by
  nothing. This is what removes the staleness lifecycle: with no correction to
  invalidate, a moved parameter vector does not invalidate the envelope, so the
  NMPC horizon cap never disappears and the controller never becomes more
  confident because it adapted.
- `PredictiveTrajectory` is `states`, `latent_states`, `commands`,
  `forecast_error_covariance`, `parameter_covariance`, `validity_utilization`,
  `forecast_error_available`, `forecast_error_horizon_supported` and
  `parameter_information_rank`, with `tangent_covariance` the sum of the two
  covariances. `nominal_states` and `mean_states` were the same trajectory once
  the bias went, so there is one; `tangent_bias`, `quantile_levels`,
  `parameter_tangent_jacobian`, `predictive_error_current`,
  `empirical_error_covariance_scope`, `parameter_covariance_combined_with_empirical_error`
  and `uncertainty_horizon_supported` are gone. The parameter contribution is
  computed through a factor of the covariance, one forward rollout per resolved
  direction, instead of a full reverse-mode Jacobian.
- `control/fitted.py` reads `belief.information.covariance()` and
  `belief.forecast_error`. `parameter_covariance_factor` returns `None` only
  when the belief resolves no direction at all, so a belief that has parameter
  information now charges its predicted spread in the objective on every path
  rather than on none. The solver's mean rollout is the model's own rollout,
  with no bias correction applied to it.
- `belief_io` writes format 5: `nominal_model`, `information`,
  `forecast_error` and `provenance`. A belief written under format 3 or 4 loads
  with a warning. Its predictive-error bias is folded back into the envelope,
  which is exact because the uncentered second moment is the centered
  covariance plus the outer product of the bias. Its parameter *covariance*, if
  it carried one, converts exactly, by the same inversion on the supported
  subspace that seeds a belief from members. Its rank-aware parameter
  *evidence* does not convert: it was whitened by a horizon-averaged held-out
  forecast covariance rather than by a one-step innovation covariance, and
  neither the noise it assumed nor the number of independent transitions behind
  it can be recovered from the artifact, so such a belief loads at rank zero
  with a second warning, keeping its names, scale and estimable mask. The
  tolerance covers the fitted corpus models under the gitignored `artifacts/`
  tree and goes away when Phase 3 re-records them.
- The fit's parameter evidence is accumulated over the training flights'
  one-step transitions rather than over the training windows' endpoints, under
  a budget of 512 windows spread evenly across the independent source groups.
  The declared noise model is one-step innovation covariance, so one-step
  windows are the windows it weights correctly: whitening a multi-step endpoint
  by it would overstate the information by roughly the horizon and would count
  the same transition once per training horizon. The fit and `absorb` therefore
  run the same estimator and their information is in one currency. The fit
  report's `configuration.parameter_evidence` records
  `method: training_one_step_information_v1`, `maximum_windows` and
  `noise_model` in place of `method: grouped_local_rollout_information_v1`,
  `maximum_windows_per_horizon` and `residual_scale_source`, and
  `models[*].parameter_evidence` is the serialized `ParameterInformation`.
  `FitSpec.parameter_evidence` still governs the precision; the noise model is
  measured on every fit, because a belief that cannot say how wrong its
  one-step predictions are cannot weight the next observation either.
- `models[*].validation.predictive_error` is `models[*].validation.forecast_error`
  and is `null` rather than an unavailable object when the held-out flights
  were shorter than every evaluation horizon.
- `structured_parameter_names`, `structured_parameter_vector` and
  `with_structured_parameter_vector` moved from `glassbox.belief.belief` to
  `glassbox.core.dynamics`, beside `structured_parameters`, and
  `TANGENT_STATE_SIZE`, `TANGENT_STATE_ORDER` and `TANGENT_GROUP_ORDER` moved
  to `glassbox.core.geometry`, beside `rigid_body_local_error`. Neither was
  public and the values are unchanged; the move is what lets the information
  and envelope modules exist without importing the belief that holds them.
- `fitted_structured_parameter_mask` is
  `glassbox.belief.information.estimable_structured_parameters` and returns the
  same mask.
- `glassbox.__all__` is thirty-eight names. `ParameterInformation`,
  `ForecastErrorEnvelope` and `UpdateResult` join it;
  `LocalParameterInformation`, `LocalGaussianParameterBelief`,
  `PointParameterBelief` and `EmpiricalHorizonPredictiveError` leave it with
  the types they named.
- The multirotor latent state is four wide, the applied motor commands, down
  from seven. `ExecutableModel.latent_size` is one entry per declared control
  channel for both families, `DynamicsParams` has nineteen structured
  coordinates instead of twenty-two, and `state_derivative` no longer takes a
  `rotational_response_state` argument. Control-generated torque is the
  memoryless map `diag(angular_accel) @ (I + cross_coupling) @ MOTOR_MIXER @
  applied_control`; the bounded cross-axis coupling is part of that map and is
  kept.
- `DynamicsParams.from_physical` no longer accepts
  `angular_response_time_constant`, and `physical()` no longer reports it.
  `fit_dynamics` and `fit_dynamics_multi_horizon` no longer accept
  `instantaneous_rotational_response`, and `LossPolicy` no longer carries it.
  The fit report's `rotational_response` key is `angular_control_coupling`,
  whose values are `diagonal_mixer_reference`, `learned_cross_coupled_mixer`
  or `not_applicable_fixedwing`, and the leave-one-label-out summary drops
  `instantaneous_rotational_response` and keeps `diagonal_angular_control`.
  `fitted_structured_parameter_mask` drops its
  `instantaneous_rotational_response` argument.
- The model payload format is version 4 and the multirotor model type is
  `effective_quadrotor_command_offset_v4`, because the old type string named
  the deleted branch. The dynamics-belief format is version 4. Its
  `latent_state_order` is one applied-control entry per channel with no
  control-generated angular-acceleration entries.
- `core/model_io.py` decodes model payloads strictly: a parameter set that is
  missing an expected name or declares an unrecognized one is refused instead
  of producing a silently different model. The one tolerated exception is a
  format-3 multirotor payload carrying `angular_response_time_constant`, which
  loads with that entry dropped and a warning, and a format-3 belief, whose
  parameter evidence and parameter belief have their three rotational-response
  coordinates projected out with a warning. Those coordinates were frozen on
  every path that ever wrote an artifact, so the projection is exact and a
  payload that does carry information there is refused rather than quietly
  weakened. The tolerance covers every belief and model written before this
  change, including the fitted corpus models under the gitignored `artifacts/`
  tree, and goes away when Phase 3 re-records them.
- `docs/results/adaptive-recovery-results.json` and
  `nmpc-acceptance-results.json` were re-recorded. One number moved
  structurally: the recovery artifact's `evidence.fleet.parameter_count` is 19
  instead of 22, because the multirotor structured parameter vector loses the
  three `log_angular_response_time_constant` coordinates. Every other
  non-timing number that moved is a multirotor closed-loop value that moved by
  between 2e-8 and 6e-6 relative. The rollouts themselves are bit-identical;
  what changed is the reverse-mode accumulation order, because the deleted
  latent carried a second differentiable path from the command into the
  control-generated torque whose forward value equalled the memoryless map
  exactly. The solver's line search consumes that gradient, so the closed-loop
  traces drift at the same scale. No fixed-wing scenario moved at all, and
  `docs/results/cascade-x8-validation-results.json` is untouched. The NMPC
  acceptance suite still passes every check, and the two numbers the docs quote
  from it, `0.651x` nominal and `0.552x` mismatch, are unchanged at the
  precision they are quoted.
- `tests/test_metrics.py` and `tests/test_evaluate.py` re-pin the values
  measured from `initial_parameter_guess()`, which was the one object in the
  package that selected the lagged branch. It was only ever a starting point:
  every fit replaced its rotational-response leaves with the memoryless
  sentinel before the first loss evaluation, so no fitted coefficient, loss
  value or held-out metric in any fit report moved.
- `glassbox.__all__` is thirty-nine names, down from eighty-eight. A name is
  public because the README or a `docs/concepts` page uses it, or because it
  is the type of one of their arguments or return values; the list is grouped
  in the order a reader meets it, and `tests/test_public_api.py` carries the
  same grouping with its reason per group. Every other name is unchanged and
  is imported from the module that owns it, for example
  `from glassbox.core.data import load_trajectory_npz`. README's layout
  section lists the surface.
- `control/online_bootstrap.py` is `control/identifier.py` and
  `control/flight_supervisor.py` is `control/supervisor.py`.
- `io/sitl_profile.py` is one recorder for both families: two target types,
  two profile tables, two condition tables and one streaming loop.
  `io/fixedwing_sitl_profile.py` is gone, and `PROFILES` and `CONDITIONS` are
  keyed by family because both families declare a `combined` profile.
- The leave-one-label-out runner has one set of defaults, `--steps 400` and
  `--learning-rate 0.02`, instead of one pair per leaf. The two documented
  invocations that relied on the old profile defaults now pass their values
  explicitly, so the command in the docs is the command that produced the
  numbers beside it.
- `glassbox extract --actuator-topic` and `--actuator-field` name the motor
  topic for both families; the fixed-wing `--motor-topic` and `--motor-field`
  spellings are gone, and `--servo-topic` and `--servo-field` are unchanged.
- The argparse front ends of the recorded-results manifest, the SITL recorder
  and the PX4 shadow moved into `glassbox.cli`; `workflows.record_results`,
  `io.sitl_profile` and `integrations.px4_nmpc_shadow` keep the work and no
  longer parse arguments.
- The reference-corpus adapters keep their parsing and lose their plumbing.
  Each one now declares its immutable file table as `PINNED_FILES` and the
  registry owns the download-and-convert loop: `fetch_nanodrone_benchmark`,
  `extract_nanodrone_benchmark`, `fetch_arp_reference`,
  `extract_arp_reference`, `fetch_idf_archive`, `extract_idf_reference`,
  `fetch_x8_reference`, `extract_x8_reference`,
  `fetch_epfl_topoplane_reference` and `extract_epfl_topoplane_reference` are
  gone. `extract_idf_ulogs` is `unpack_pinned_ulogs`, the archive-member
  verification the registry calls before the adapter runs, and
  `io.x8_reference.load_validation_trajectories` is
  `validate_validation_trajectories`, which checks loaded trajectories rather
  than loading them.
- `idf_corpus_report` is part of `glassbox corpus prepare idf`, which writes it
  to `DIR/corpus_report.json`; `save_idf_corpus_report` is gone, because the
  registry writes every corpus audit the same way. The IDF page's retained
  duration and segment counts still come from that file.
- `integrations/cascade.py` keeps the plant and nothing else. The X8 validation,
  the residual regressions and the variant grid move to
  `glassbox.workflows.benchmarks.cascade_x8`, which is where a fixed-wing corpus
  experiment belongs: `X8Variant`, `x8_variant_models`,
  `shift_center_of_gravity`, `shift_center_of_gravity_of_spec`,
  `actuator_states_over_controls`, `cascade_window_predictions`,
  `evaluate_x8_cascade`, `save_x8_cascade_report`, `ResidualRegression`,
  `residual_regressions`, `diagnose_x8_cascade` and the four X8 constants.
  `x8 evaluate-cascade` and `x8 diagnose-cascade` keep every flag and are now
  callers of the new module, which produces byte-identical output to the module
  it was split out of.
- `CascadePlant` is a writable `VehicleLink`: `read` reports where the plant is,
  carrying its `applied_control` as the applied command and its own clock as the
  source time, and `write` advances it one control interval. `command_size` and
  `command_bounds` come from the aircraft specification, propellers as
  normalized throttle and each control channel bounded by the tightest surface
  deflection limit it drives. The one control loop that shadows PX4 telemetry
  flies a simulated aircraft unchanged.
- `require_cascade` is public, so the benchmark that needs the optional
  simulator does not reach for a private name in another module.
- `workflows/nmpc_benchmark.py` and `workflows/adaptive_recovery_benchmark.py`
  are `workflows/benchmarks/nmpc.py` and `workflows/benchmarks/recovery.py`.
  `nmpc-benchmark` and `adaptive-recovery` keep their names and every flag. The
  recovery artifact's `source_files` provenance follows the move, which is the
  only thing about it that changes.
  `adaptive_recovery_source_fingerprint` now anchors that list at the
  `glassbox` package root rather than at the module's own depth inside it, so
  the digest does not depend on where in the package the benchmark lives.
- `MultirotorFlightSupervisor(config, *, allocate=None)` no longer assumes the
  canonical motor mixer. `allocate` maps the desired `(roll, pitch, yaw)`
  differential to the four motor increments the arrest adds to the configured
  collective hold, and `has_allocation` reports whether one was given. Without
  it the supervisor still runs every freshness rule, every limit and the latch,
  but its arrest is the collective hold alone, rate-limited toward the
  previously applied command, reported as `SupervisorMode.COLLECTIVE_HOLD` and
  carrying the new `SupervisorReason.NO_ALLOCATION`. With an allocation every
  arrest command is bit-identical to before. `MOTOR_MIXER` keeps its one home
  in `core/dynamics.py`, where the multirotor family defines it; the supervisor
  no longer imports it.
- `glassbox px4-nmpc-shadow` is one control loop instead of one report
  assembly. It writes one JSON object per interval, carrying the state that was
  read, the applied and solved commands, the solver's status and solve time,
  and the plan's diagnostics, and prints a closing summary of status counts,
  usable and fallback counts, deadline misses, solve-time median, p90 and
  maximum, and the worst message skew, receive age, source-clock lag and
  state-to-command skew. `--output` now names a JSON-lines file rather than one
  document. `run_px4_nmpc_shadow(link, controller, *, steps, write_line=None)`
  takes the link and the controller it drives and returns a `LoopSummary`; it
  no longer builds a controller, a report, or the cold and warm warm-up solves
  that only existed to be recorded in one.
- One latched PX4 receiver. `PX4MavlinkStateSource` and `PX4HILActuatorSource`
  were the same daemon-thread receiver twice, differing in how a message is
  decoded and how a reader selects from what has been latched, so both are now
  thin decoders over one `_LatchedReceiver` sharing the heartbeat check, the
  drain loop, the source-system filter, the sequence gate, and the close path.
  The state assembler, frame conversions and boot-clock unwrap are untouched.
- One evaluation, three named policies. `workflows/nanodrone_evaluation.py`,
  `workflows/x8_evaluation.py` and `workflows/epfl_evaluation.py` are folded
  into `workflows/evaluate.py`; `workflows/profile_benchmark.py` and
  `workflows/source_group_benchmark.py` into `workflows/holdout.py`. The
  `nanodrone evaluate`, `x8 evaluate`, `epfl evaluate`, `profile-benchmark`
  and `source-benchmark` commands keep every flag they had and are now thin
  callers. Each policy reproduces its module's numbers exactly;
  `tests/test_evaluate.py` pins all three. Last commit carrying the five
  modules: `c250233`.
- The holdout summary is keyed by the label, not by the label's name. `folds`,
  `fold_count`, `per_fold`, `baseline_horizon_rollouts` and
  `model_over_baseline` replace `profiles`/`source_groups`,
  `profile_count`/`source_group_count`, `per_profile`/`per_source_group`,
  `kinematic_persistence_horizon_rollouts` and
  `model_over_kinematic_persistence`. `aggregate.weighting` stays
  `equal_profile` or `equal_source_group`, named after the label held out.
  The summary carries `holdout_label`, and both runners now report the
  persistence baseline and the fold distribution, which only the source-group
  runner did.
- The holdout runner's fold `configuration` records the loss policy directly
  (`learn_thrust_command_offset`, `instantaneous_rotational_response`,
  `diagonal_angular_control`, `ablations`) instead of the two derived prose
  strings `multirotor_thrust_command_offset` and `rotational_response` that
  were computed from them.
- Every fold is planned with `Holdout.by_label`, where the source-group
  benchmark reordered paths and used `Holdout.by_group(1)`. The split is the
  same and so are the numbers; each fold report's `split.mode` is now
  `leave_labeled_out`. The resume request records the fold label key, so a
  directory written by the previous runner refits rather than resuming.
- `integrations/cascade.py`'s window predictor is
  `cascade_window_predictions`, so it no longer shares a name with
  `glassbox.predict_windows`. Its scores come from the `x8` policy through
  `score_against_baseline`, which is the same reduction over the same floors.
- `core/evaluation.py` splits into `core/metrics.py` (predictions, the rollout
  RMSE convention, the kinematic-persistence baseline, both persistence-score
  reductions, both floor tables, divergence and aggregation) and
  `core/diagnostics.py` (one-step innovation, kinematic compatibility, the
  Pearson helpers, `attitude_innovation`). `parameter_dict` moves to
  `core/model_io.py`, which is what serializes it. Every metric value is
  bit-identical; `tests/test_metrics.py` pins the numbers on two synthetic
  flights and compares them exactly. Last commit carrying `core/evaluation.py`:
  `b342624`.
- Five prediction entry points become two. `rollout_predictions`,
  `windowed_rollout_predictions`, `windowed_rollout_evaluation`,
  `windowed_rollout_metrics` and the old `rollout_metrics(params, trajectory)`
  are replaced by `predict`, `predict_windows` and
  `rollout_metrics(prediction)`. `windowed_rollout_metrics` leaves the public
  `glassbox` surface; `predict`, `predict_windows` and `RolloutPrediction`
  join it.
- The window stride keyword is `stride` everywhere. `predict_windows` and
  `kinematic_persistence_windowed_metrics` take `stride`, matching
  `trajectory_windows`, where the deleted entry points took `stride_steps`.
- The innovation diagnostics no longer run on every fit. `validation.aggregate`
  and each `validation.per_flight` entry carry `one_step_innovation` only when
  the fit asked for diagnostics.
- One fit spec. `workflows/fitting.py` is now `glassbox/fitting.py`, because
  `fit` is public and `glassbox.workflows` is a deferred subpackage.
  `FitRequest`'s twenty-two fields become `FitSpec`'s eleven plus a
  `LossPolicy` holding the loss geometry (`endpoint_weight`,
  `stability_regularization`, `learn_thrust_command_offset`,
  `instantaneous_rotational_response`, `diagonal_angular_control`) and a
  `WeightingPolicy` holding `balanced` and `group_weights`.
  `training_horizons_s` is `horizons_s`, `horizon` is `horizon_steps`,
  `fixed_motor_time_constant_s` is `fixed_response_time_constant_s`, and
  `build_parameter_evidence` is `parameter_evidence`.
  `fit_trajectory_artifacts` and `fit_from_request` are replaced by `fit`.
- The no-lag ablation is opt-in. `FitSpec.ablations` defaults to no ablation
  and `ablations=("no_lag",)` fits it, so a plain fit is half the work it was.
  `glassbox fit` drops `--skip-no-lag-ablation` and gains `--ablation no-lag`;
  `--baseline-model` requires it, and `--fixed-response-time-constant` is
  simply incompatible with it rather than requiring another flag. Every
  documented corpus command that turned the ablation off just drops the flag.
  The fit report's `configuration.no_lag_ablation` becomes
  `configuration.ablations`.
- The fit resolves the runtime contract. `ExecutableModel.runtime_spec` is
  built by the fitter from the pooled sample rate and the objective's own
  training envelope and travels on the returned belief, so
  `core.model.runtime_spec_from_fit_report` is deleted along with its four
  callers and leaves the public `glassbox` surface;
  `runtime_spec_from_trajectory` is unchanged. A belief written before this
  change reloads to identical parameters, error moments, parameter evidence,
  input spec and runtime spec. Last commit carrying it: `733548d`.
- The fit report drops the keys nothing read: the `interpretation` prose,
  which is now in `docs/concepts/dynamics-beliefs.md`; the per-flight and
  training `excitation` blocks; `dataset.condition_counts`,
  `dataset.unlabeled_flight_count` and `dataset.unlabeled_condition_count`;
  `configuration.training_weight_share_per_flight_by_horizon` and
  `configuration.training_weight_share_per_source_group_by_horizon`;
  `training_window_selection.selection_policy_by_horizon`,
  `candidate_windows_by_horizon` and `selection_fraction_by_horizon`; and
  `optimization_data_policy.batch_size_by_horizon`,
  `window_coverage_by_horizon`, `maximum_windows_per_horizon_per_step` and
  `maximum_transitions_per_horizon_per_step`. `dataset`, `split`, `fit` with
  its losses and loss configuration, `models[*].validation`, `configuration`
  and provenance are unchanged. The weighting guarantees the deleted share
  keys carried are now asserted directly on the extracted window weights.
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
  `benchmark_split_holdout`/`leave_profiles_out` as `leave_labeled_out`, which
  the EPFL corpus evaluation's split check follows. The
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
- `RecursiveBootstrapConfig.transition_aggregation_steps` and
  `RecursiveBootstrapConfig.integrated_collective`, the identifier's last two
  research switches, and every branch on them. The window is
  `TRANSITION_AGGREGATION_STEPS`, two, and the collective map is always fit on
  the integrated target: both were measured better on the release ensemble in
  the dual-control design and are now the only behaviour. The last commit that
  could run either alternative is `2f5adc2`.
- `RecursiveBootstrapBelief`, its thirty-three fields, its `to_dict`, its
  `predict_collective_specific_force` and `predict_angular_acceleration`
  methods, and the field-list contract test
  `RECURSIVE_BOOTSTRAP_BELIEF_FIELDS`. The identifier produces a
  `DynamicsBelief` instead; the parameters are on the model, the prediction
  methods are the model's `transition`, and the estimator-specific fields are
  `BootstrapEvidence`. `update_wall_time_s` went with it, since the belief a
  transition produces is now a pure function of the evidence and the elapsed
  time is already on `RecursiveBootstrapSampleReport`. The new contract test
  pins the family's forty-one parameter names in order and the evidence
  summary's twenty-four fields. Last commit with the old belief: `2f5adc2`.
- The transactional belief update, at `glassbox/belief/adaptation.py`, 2,006
  lines. `BeliefUpdateProposal`, `BeliefUpdateReport` and its thirty-five
  fields, `propose_dynamics_belief_update`,
  `validate_and_commit_dynamics_belief_update`, `update_dynamics_belief`,
  `recalibrate_predictive_error`, `DynamicsBelief.update`, `propose_update`,
  `commit_update` and `recalibrate_predictive_error`, the two-sigma improvement
  margin `IMPROVEMENT_MARGIN_STANDARD_ERRORS`, the maximum-norm trust bound
  `MAXIMUM_LOCAL_PARAMETER_STEP_SIGMA`, the line-search fractions, the revision
  and control-history fingerprints and the transition replay detection are all
  gone, together with `tests/test_adaptation.py` and its nineteen tests.
  `HorizonEndpointErrorEvidence` and `endpoint_error_evidence_by_horizon` moved
  to `glassbox.belief.forecast_error`, which is the only part of the module
  anything else read.

  The reason is not cost. On every path a shipped artifact could reach, the
  account of what is unknown never changed: the fit wrote total-forecast-scoped
  evidence and every contraction mechanism was gated on a conditional
  innovation scope that nothing produced, so the recorded artifact said
  covariance not updated, posterior trace equal to prior trace, information
  gain null. A commit then zeroed the error moments, which removed the NMPC
  horizon cap and showed the controller zero model uncertainty, so adapting
  made it more confident than its evidence supported. The margin and the trust
  bound existed to patch the null-acceptance rate of an improvement-threshold
  gate, which is the mechanism the design does not want. `absorb` is the
  recursion the in-flight identifier already ran, generalized by one Jacobian.
  Last commit carrying the transaction: `8d3400f`.
- The 64-seed null-acceptance calibration, which measured how often the
  transaction's threshold committed when there was nothing to learn. With no
  gate there is no acceptance rate to calibrate; `tests/test_absorb.py` pins a
  step-size property in its place, at the same 64 seeds. Under the null the
  step has covariance `P dL P = P - P L P`, at most the posterior covariance
  and therefore at most the prior covariance, so the mean step over `S` seeds
  along any resolved direction is within `k / sqrt(S)` prior sigma; the test
  uses `k = 4`, a bound of `0.5` prior sigma. The constant is derived from that
  inequality rather than from a run.
- `glassbox/belief/covariance.py`, folded into
  `glassbox/belief/information.py` with its contents unchanged.
- The lagged multirotor rotational-response branch and its sentinel machinery:
  `INSTANTANEOUS_ROTATIONAL_RESPONSE_S`, `_angular_response_at`,
  `_split_latent_state`, `_initial_latent_state`,
  `MULTIROTOR_ROTATIONAL_STATE_SIZE`,
  `with_instantaneous_rotational_response`,
  `has_instantaneous_rotational_response`,
  `zero_rotational_response_gradient` and
  `_warn_if_rotational_response_frozen`. No caller in the package, its tests or
  the parked demo ever selected the branch, and it cost three latent dimensions
  on every multirotor rollout and three frozen coordinates in every multirotor
  parameter vector. Its promotion failures on the Nano-drone and ARP corpora
  are recorded in `docs/literature-review.md`; the two experiment pages no
  longer carry them. Last commit carrying the branch: `9d59e4a`.
- The `glassbox.experimental` subpackage. It re-exported the recursive
  bootstrap identifier and the flight supervisor under a promise that their
  contracts could change without notice; both are now ordinary components of
  `glassbox.control`, and the deferred-subpackage test no longer names an
  experimental tier. Last commit carrying it: `2faf563`.
- `core/metrics.py::p90_horizons` and `summarize_divergence`, which had no
  caller left after the gates that used them were deleted. Last commit
  carrying them: `2faf563`.
- The synthetic parameter-recovery demonstration behind `glassbox synthetic`,
  which fitted a model on generated flights and printed the loss reduction. It
  was a demonstration with no artifact and no test; `glassbox synthetic` now
  writes the corpus and `glassbox fit` fits it. Last commit carrying it:
  `e17c88c`.
- `scripts/extract_ulog_dataset.sh`, `scripts/record_sitl.sh` and
  `scripts/record_fixedwing_sitl_profiles.sh`. The first is
  `glassbox extract LOG... OUTDIR`, and the other two are
  `scripts/record_sitl_profiles.sh --family` and its `baseline` profile. Last
  commit carrying them: `e17c88c`.
- The EPFL characterization report's `evaluation`, `split`, `interpretation`
  and `limitations` keys, which were prose the command pasted into JSON. The
  EPFL page carries that interpretation. Last commit carrying them: `e17c88c`.
- The Nano-drone benchmark report's `benchmark` and `test_artifacts` keys, in
  favour of the `corpus` block `--corpus NAME` records for any corpus. Last
  commit carrying them: `e17c88c`.
- `--skip-checksum` on every corpus command. A pinned corpus that does not
  verify is a different corpus, and a number measured on it is not comparable
  to the published one, so verification is not optional on the command line.
  Last commit carrying it: `3db7ce5`.
- `glassbox.core.adapter` and the `TrajectoryAdapter` protocol, with its entry
  in the public surface. It had no production consumer; the registry's
  `adapter` field is typed by the concrete adapter class's `inspect`, `load`
  and `load_all` shape, which the registry's docstring states. Last commit
  carrying it: `3db7ce5`.
- The shadow runner's report: `schema_version` 6, the forty-key sample rows,
  the cold and warm warm-up rows, the clock-ratio audit, and every summary key
  derived from them. No recorded artifact was produced from it and no test
  outside its own read it. `run_control_loop`'s per-interval record and
  `LoopSummary` replace it. Last commit carrying it: `1acdd50`.
- `glassbox.integrations.streaming_evaluation` and
  `StreamingOneStepEvaluator`, with `tests/test_streaming_evaluation.py`. The
  shadow report was its only consumer, and a live transport fixture is not
  evidence about prediction quality; the held-out horizon tables in the fit
  report are. Last commit carrying it: `1acdd50`.
- `AppliedCommandSource`, `run_px4_nmpc_shadow`'s `source`, `model`,
  `previous_command`, `applied_command_source`, `sample_count` and
  `telemetry_timeout_s` parameters, and the `PX4TelemetryError` branch that
  reported a receiver publishing an empty sample, which no path could reach.
  Last commit carrying them: `1acdd50`.
- `with_constant_angular_rate` and the `constant_angular_rate_diagnostic` arm
  of the nanodrone report. The published protocol's baseline is hold-state, so
  the constant-rate arm was a model variant with no artifact and no reader; it
  is the last of the model variants the migration retires. Last commit
  carrying it: `c250233`.
- `evaluate_nanodrone_benchmark`, `evaluate_nanodrone_model_artifact`,
  `save_nanodrone_benchmark_report`, `evaluate_x8_reference_models`,
  `save_x8_reference_report`, `geometric_ratio`,
  `evaluate_epfl_characterization`, `save_epfl_characterization`,
  `benchmark_profiles` and `benchmark_source_groups`, with the five modules
  that held them. `evaluate`, `evaluate_fit_reports` and `evaluate_holdout`
  replace them. Last commit carrying them: `c250233`.
- `rollout_divergence_metrics` no longer reports `final_errors` or
  `first_nonfinite_time_s`; nothing in the package, its tests, its recorded
  artifacts or its documentation read either. `stable_fraction` stays because
  `summarize_divergence` reduces it. Last commit carrying them: `b342624`.
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
