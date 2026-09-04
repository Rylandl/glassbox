# Adaptive configuration-change recovery

`glassbox adaptive-recovery` is a fixed synthetic diagnostic for the complete
belief-to-control path. It asks whether the spread of a few sibling arm
configurations and 0.8 seconds of telemetry from a previously unseen
adjustable-arm configuration can improve a prewarmed NMPC recovery without
hiding uncertainty behind a point estimate.

The diagnostic is deliberately not a pass/fail gate. It records the result of
one reproducible scenario, including negative evidence, so estimator or
controller changes cannot silently redefine success.

## Scenario

The quadrotor arm-length ratio changes from `1.0` to `1.2214`. The synthetic
plant maps that change into inverse roll/pitch angular authority, while
Glassbox continues to estimate effective dynamics coefficients rather than
requiring a geometric airframe decomposition.

The evidence path is:

1. Summarize five sibling arm configurations as the parameter covariance around
   the prechange model. One direction carries the whole spread; directions no
   configuration moved carry no variance and take no step, and nothing
   completes them with an assumption.
2. Start from the known prechange vehicle model.
3. Propose an update from the first half of 0.8 seconds of target telemetry and
   validate it on the disjoint second half. Pre-split commands initialize the
   validation actuator state but are fingerprinted separately and do not count
   as validation evidence.
4. Evaluate the accepted model on independent 0.6-second prediction windows.
5. Compare four 1.2-second, prewarmed closed-loop recoveries from the same
   bounded initial disturbance: stale belief, adapted belief, adapted point
   mean, and hidden oracle point mean.

The controller charges the belief's own predicted spread inside its objective:
the tracking cost carries the trace of the predicted tangent covariance against
the tracking weights, and the validity term carries the mean utilization plus a
componentwise one-standard-deviation margin. Nothing edits the optimized command
afterwards. It includes no motor mixer, attitude/rate law, independent fallback
controller, PX4 integration, or flight-authority handoff. Compilation time is
excluded, while per-solve runtime is reported.

## Recorded result

Recorded on 2026-09-03 with the current acceptance criterion (candidate scored
without the held-out bias against the bias-corrected incumbent, with a
noise-scaled margin, and no whitened prior coordinate moving more than one
standard deviation), rollout error statistics that exclude the shared initial
sample, and the block-granular NMPC warm start. This run is the first without
the structured fleet prior. The configuration spread is now the plain sample
covariance of the five members around the prechange model, which is four fifths
of the prior's between-member covariance along the same single direction, so
the bounded step is slightly shorter and every number below moved a little.

The accepted update reduced independent normalized 0.6-second prediction RMS
from `0.033394` to `0.015286` (`0.458x`). The recovery advantage that earlier
runs attributed to adaptation has disappeared now that the warm start actually
advances the plan, and charging the spread in the objective removes what was
left of it: relative to the stale belief, the adapted belief produced `1.059x`
recovery-tail tracking RMS and `1.003x` recovery-tail attitude/rate RMS, and
relative to the oracle point model those ratios were `1.123x` and `1.223x`.
Useful parameter evidence reaches the predictive mean; on this scenario it
buys support, not tracking.

With actuator history correctly carried across the split, the disjoint
validation RMS is `1.1994 → 0.5447`. The earlier `1.7534 → 1.6154` values came
from incorrectly treating the first post-split command as a steady actuator
state; they are no longer part of the recorded evidence.

Those numbers show that the architecture can move useful configuration evidence
through an immutable belief update into the predictive mean. They are not an
acceptance threshold or a general recovery claim, and the recovery-tail ratios
do not support a claim about command selection.

The support result is intentionally negative. Maximum actual validity
utilization was `1.101`, `1.045`, `1.081`, and `1.137` for stale belief, adapted
belief, adapted point mean, and oracle point mean, and the maximum utilization
the full prediction reached was `1.282`, `1.146`, `1.152`, and `1.226`. All
traces remained finite and bounded with no solver fallback, and every trace
still left support. What the charged spread does buy is visible in the ordering:
the adapted belief, the only arm carrying parameter uncertainty, is the arm that
stays closest to supported ground on both measures, and it is the arm that pays
for it in tracking.

This is the behavior the diagnostic should expose. The benchmark establishes
useful adaptation evidence and a clear controller limitation, not an
invariant-set, envelope-expansion, flight-safety, or throw-to-recover result.

On the recorded run, the uncertainty-bearing adapted belief cost roughly twice
the point-model solve per step against a `20 ms` model period, because charging
the spread costs one extra forward rollout per resolved parameter direction.
Absolute solve times depend on the host and its load, so they are kept only in
the results artifact, where the benchmark marks them nondeterministic and
excludes them from its comparison. This is not a hard real-time claim.

## Reproduce

```bash
uv run glassbox adaptive-recovery \
  --output docs/results/adaptive-recovery-results.json
```

The checked-in [result artifact](../results/adaptive-recovery-results.json) records the
scenario contract, environment, evidence, all four recovery traces, direct
comparisons, observations, and limitations. Its `acceptance_gate`,
`flight_safety_claim`, and `throw_to_recover_claim` fields are all false.
The artifact also stores source and scenario SHA-256 fingerprints. Its test
regenerates the complete report and compares every deterministic field after
excluding only platform metadata and measured wall-clock timings, so changed
results must be recorded deliberately without becoming a performance gate.
