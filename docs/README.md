# Glassbox documentation

Start with the repository [README](../README.md), then the
[runnable fit/predict/update guide](guides/platform-onboarding.md).

- [Learner contract](learner.md): generic recordings, prediction history,
  immutable revisions, error envelopes and the fixed recipe.
- [Scope](scope.md) and [onboarding responsibilities](platform-onboarding.md):
  what the caller and learner each provide.
- [Charter](charter.md) and [status](status.md): the adopted direction,
  definition of done, current measurements and next named gap.
- [PX4 ULogs](guides/px4-ulog.md): canonical telemetry extraction, explicit
  generic-recording conversion and reproducible SITL recording.

## Control and retained structured consumers

The public learner is generic. The following documents describe experimental
control seams and the structured implementations still used by benchmarks and
unmigrated integrations.

- [Nonlinear model-predictive control](concepts/nmpc.md): plan models, bounded
  optimization, objectives, vehicle links and supervision.
- [Dynamics beliefs](concepts/dynamics-beliefs.md): retained structured model
  artifacts, parameter information and their `absorb` workflow.
- [Bootstrap identification](concepts/bootstrap-identification.md): the
  experimental local-authority identification contract.

## Research evidence

Historical experiments preserve their original settings and outcomes.
Their acceptance flags do not replace the current [charter](charter.md) or a
declared application's success criterion.

- [Validation](validation.md): recorded prediction and control comparisons.
- [Streaming refinement](streaming-refinement.md) and
  [Cascade refinement](cascade-refinement.md): retained live-learning experiments.
- [Repeated-fit uncertainty](repeated-uncertainty-calibration.md): parameter
  evidence and forecast errors under noise and reduced excitation.
- [Accuracy requirements](accuracy-requirements.md): the distinction between
  application requirements and descriptive forecast-error measurements.
- [Forecast accuracy versus ordinary tracking](cascade-accuracy.md): control
  failures hidden by low held-out forecast scores.
- [Bounded recovery](recovery-investigation.md),
  [terminal suffixes](terminal-suffix-investigation.md),
  [fast suffixes](fast-suffix-investigation.md) and
  [shift uncertainty](shift-uncertainty-audit.md): control investigations.

## Background and artifacts

[The literature review](literature-review.md) records research decisions;
the [original proposal](history/idea-2026-08.md) is historical background.

[`results/`](results/) holds machine-readable comparisons linked from
[validation](validation.md). [Contributing](../CONTRIBUTING.md#recorded-results)
describes their manifests and reproduction.

The related [glassbox-throw](https://github.com/Rylandl/glassbox-throw) project
contains a Crazyflow throw demonstration and its control diagnostics.
