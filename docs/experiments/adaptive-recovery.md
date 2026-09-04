# Adaptive configuration-change recovery

`glassbox benchmark recovery` is a fixed synthetic diagnostic for the complete
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

1. Seed the belief from five sibling arm configurations. Their sample
   covariance around the prechange model inverts, on its rank-one supported
   subspace, to precision: the belief starts at the prechange mean and claims
   to know exactly one direction, the one the siblings moved. Every other
   coefficient is at zero precision, which is to say unknown, and nothing
   completes it with an assumption.
2. Measure the one-step innovation of the prechange model on the fleet's own
   telemetry. That is the noise model every later observation is weighted by.
3. Absorb 0.8 seconds of telemetry from the changed vehicle. Every usable
   one-step transition adds its information; the step is the pseudo-inverse of
   the accumulated precision applied to the whitened innovation, so it is
   exactly zero along directions the telemetry does not resolve. There is no
   proposal, no validation split and no acceptance threshold.
4. Evaluate the updated model on independent 0.6-second prediction windows.
5. Compare four 1.2-second, prewarmed closed-loop recoveries from the same
   bounded initial disturbance: seeded belief, adapted belief, adapted point
   mean, and hidden oracle point mean.

The controller charges the belief's own predicted spread inside its objective:
the tracking cost carries the trace of the predicted tangent covariance against
the tracking weights, and the validity term carries the mean utilization plus a
componentwise one-standard-deviation margin. Nothing edits the optimized command
afterwards. It includes no motor mixer, attitude/rate law, independent fallback
controller, PX4 integration, or flight-authority handoff. Compilation time is
excluded, while per-solve runtime is reported.

## Recorded result

Recorded on 2026-09-04, the first run of the recursive `absorb` update. The
previous run measured the transactional propose, validate, commit update
against a rank-one parameter *covariance*, and every number below moved. Three
changes account for it. The member spread is now inverted to precision, so it
seeds one known direction instead of declaring one uncertain direction and
nineteen certain ones. The update accumulates information from all forty
one-step transitions instead of splitting eight windows into a proposal half
and a validation half. And the held-out forecast bias is no longer applied at
runtime, so the envelope is the uncentered second moment of the held-out error
and a moved parameter vector no longer makes it stale.

The absorbed update raised the belief's resolved rank from `1` to `9` of the
`15` estimable coordinates and reported an information gain of `1.698` nats,
where the transaction reported `null` because its covariance never moved. The
whitened one-step innovation on the absorbed telemetry fell from `0.2514` to
`0.0264`, and the step was `0.321` of the prior precision's own standard
deviation. Independent normalized 0.6-second prediction RMS fell from
`0.033394` to `0.010120`, a ratio of `0.303x` where the transaction reached
`0.458x`.

The recovery comparison changed sign. Relative to the seeded belief, the
adapted belief produced `0.993x` recovery-tail tracking RMS and `0.830x`
recovery-tail attitude/rate RMS, where the transactional update produced
`1.059x` and `1.003x`. Relative to the oracle point model those ratios were
`1.078x` and `1.049x`, down from `1.123x` and `1.223x`. The adapted arm is now
better than the arm it started from on both tail measures and close to the
oracle on both.

The support result is still negative, and less so. Maximum actual validity
utilization was `1.085`, `1.056`, `1.091`, and `1.136` for seeded belief,
adapted belief, adapted point mean, and oracle point mean, and the maximum
utilization the full prediction reached was `1.205`, `1.155`, `1.211`, and
`1.225`. All traces remained finite and bounded with no solver fallback, and
every trace still left support. The ordering the charged spread buys is
unchanged: the adapted belief, which carries the most parameter information, is
the arm that stays closest to supported ground on both measures.

The belief's own reported spread now moves in the direction the evidence does.
The seeded belief carries a rank-one covariance and reports a maximum
normalized model uncertainty of `0.661`; the adapted belief, having resolved
eight more directions from telemetry, reports `0.342`. The covariance trace
rises from `0.0625` to `0.3164` over the same update, which is not a
contradiction: an unresolved direction contributes exactly zero variance under
the pseudo-inverse convention, so resolving eight new directions adds eight new
variances while shrinking the one that was already there. The rank, not the
trace, is what says how much is known.

This is the behavior the diagnostic should expose. The benchmark establishes
useful adaptation evidence and a remaining controller limitation, not an
invariant-set, envelope-expansion, flight-safety, or throw-to-recover result.

On the recorded run, the adapted belief cost several times the point-model
solve per step against a `20 ms` model period, because charging the spread
costs one extra forward rollout per resolved parameter direction and this
belief resolves nine. Absolute solve times depend on the host and its load, so
they are kept only in the results artifact, where the manifest entry lists them
as volatile so neither the pinned test nor `record-results --check` compares
them. This is not a hard real-time claim.

## Reproduce

```bash
uv run glassbox record-results --only adaptive-recovery-results
```

which is one step:

```bash
uv run glassbox benchmark recovery \
  --output docs/results/adaptive-recovery-results.json
```

The checked-in [result artifact](../results/adaptive-recovery-results.json) records the
scenario contract, environment, evidence, all four recovery traces, direct
comparisons, observations, and limitations. Its `acceptance_gate`,
`flight_safety_claim`, and `throw_to_recover_claim` fields are all false, and
its `parameter_covariance_updated_by_the_update` field is true, which is now
true by construction rather than by a gate's verdict.
The artifact also stores source and scenario SHA-256 fingerprints. Its test
regenerates the complete report and compares every deterministic field after
excluding only platform metadata and measured wall-clock timings, so changed
results must be recorded deliberately without becoming a performance gate.
