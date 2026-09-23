# Changelog

## Unreleased

Replace the preceding shared-physics network and rate-memory implementation with
one episode-fitted causal actuator equation. Gravity, rigid-body kinematics and
frame transforms remain analytic. A single no-pretraining fitting procedure
learns hidden command response, body force, torque, inertia and actuator
momentum for any input count, without vehicle-family or layout metadata.

Keep the public fit/predict/update lifecycle with immutable, fingerprinted
revisions. `OnlineFit` now collects contiguous observations and publishes
background-fitted revisions intermittently. Prediction needs complete issued-
command history from the segment start. Offline fitting uses all supplied
recordings; no unqualified calibration envelope is emitted. The revision and
session archive formats change, so older archives require their source checkout
or a fresh fit.

The corrected [matched-prefix and publication qualification](docs/causal-publication.md)
records full-state errors, command response, fit
and publication timing, and known regressions. Superseded implementations,
experiments, tests and narrative have been removed from the working tree;
Git retains earlier history. No live Throw/Dart controller success or unseen-
configuration accuracy is claimed by this release.
