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

### Fixed
- `glassbox fit --model --report` no longer fails on a NumPy scalar.
- Fits stop on non-finite loss and return the best finite iterate with a flag.
- Physical parameter constructors validate their inputs instead of clipping.
- Holdouts follow `benchmark_split` labels when present.

## 0.1.0

Initial development snapshot.
