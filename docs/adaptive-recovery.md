# Adaptive configuration-change recovery

`glassbox-adaptive-recovery` is a fixed synthetic diagnostic for the complete
belief-to-control path. It asks whether fleet evidence and 0.8 seconds of
telemetry from a previously unseen adjustable-arm configuration can improve a
prewarmed NMPC recovery without hiding uncertainty behind a point estimate.

The diagnostic is deliberately not a pass/fail gate. It records the result of
one reproducible scenario, including negative evidence, so estimator or
controller changes cannot silently redefine success.

## Scenario

The quadrotor arm-length ratio changes from `1.0` to `1.2214`. The synthetic
plant maps that change into inverse roll/pitch angular authority, while
Glassbox continues to estimate effective dynamics coefficients rather than
requiring a geometric airframe decomposition.

The evidence path is:

1. Build a five-configuration fleet prior. Only one natural-coordinate
   direction is empirically spanned; assumed completion of the other directions
   remains separately reported.
2. Start from the known prechange vehicle model.
3. Propose an update from the first half of 0.8 seconds of target telemetry and
   validate it on the disjoint second half. Pre-split commands initialize the
   validation actuator state but are fingerprinted separately and do not count
   as validation evidence.
4. Evaluate the accepted model on independent 0.6-second prediction windows.
5. Compare four 1.2-second, prewarmed closed-loop recoveries from the same
   bounded initial disturbance: stale belief, adapted belief, adapted point
   mean, and hidden oracle point mean.

No watchdog, rate-arrest controller, PX4 integration, or independent safety
supervisor is present. Compilation time is excluded, while per-solve runtime is
reported.

## Recorded result

The accepted update reduced independent normalized 0.6-second prediction RMS
from `0.033394` to `0.013706` (`0.410x`). Relative to the stale belief, the
adapted belief produced `0.818x` recovery-tail tracking RMS and `0.561x`
recovery-tail attitude/rate RMS. Its tail tracking was `0.997x` the oracle point
model in this trace.

With actuator history correctly carried across the split, the disjoint
validation RMS is `1.1994 → 0.4982`. The earlier `1.7534 → 1.6154` values came
from incorrectly treating the first post-split command as a steady actuator
state; they are no longer part of the recorded evidence.

Those numbers show that the architecture can move useful configuration evidence
through an immutable belief update and into command selection. They are not an
acceptance threshold or a general recovery claim.

The important negative result is that every recovery condition reached roughly
`1.6-1.7x` learned validity utilization. The disturbance begins inside support,
but the controller's state-support cost is still soft. Consequently this result
does not establish safe recovery, envelope expansion, or throw-to-recover
capability. It points to the next general controller requirement: a hard
support/authority boundary plus an independent attitude/rate-arrest supervisor.

## Reproduce

```bash
uv run glassbox-adaptive-recovery \
  --output docs/adaptive-recovery-results.json
```

The checked-in [result artifact](adaptive-recovery-results.json) records the
scenario contract, environment, evidence, all four recovery traces, direct
comparisons, observations, and limitations. Its `acceptance_gate`,
`flight_safety_claim`, and `throw_to_recover_claim` fields are all false.
The artifact also stores source and scenario SHA-256 fingerprints. Its test
regenerates the complete report and compares every deterministic field after
excluding only platform metadata and measured wall-clock timings, so changed
results must be recorded deliberately without becoming a performance gate.
