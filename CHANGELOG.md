# Changelog

All notable changes to Glassbox are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## Unreleased

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

### Fixed
- `glassbox-fit --model --report` no longer fails on a NumPy scalar.
- Fits stop on non-finite loss and return the best finite iterate with a flag.
- Physical parameter constructors validate their inputs instead of clipping.
- Holdouts follow `benchmark_split` labels when present.

## 0.1.0

Initial development snapshot.
