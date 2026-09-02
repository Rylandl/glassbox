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

The controller includes the maintained vehicle-agnostic belief-support
projection. Every alternative is a blend between the optimized NMPC command
and the previous bounded command. It includes no motor mixer, attitude/rate law,
independent fallback controller, PX4 integration, or flight-authority handoff.
Compilation time is excluded, while per-solve runtime is reported. Both nominal
and alternative projection paths are compiled during prewarming.

## Recorded result

The accepted update reduced independent normalized 0.6-second prediction RMS
from `0.033394` to `0.013706` (`0.410x`). Relative to the stale belief, the
adapted belief produced `0.811x` recovery-tail tracking RMS and `0.539x`
recovery-tail attitude/rate RMS. Relative to the oracle point model, those
ratios were `0.990x` and `0.799x`. Thus useful parameter evidence reaches both
the predictive mean and uncertainty-aware command selection without a hidden
platform-specific controller.

With actuator history correctly carried across the split, the disjoint
validation RMS is `1.1994 → 0.4982`. The earlier `1.7534 → 1.6154` values came
from incorrectly treating the first post-split command as a steady actuator
state; they are no longer part of the recorded evidence.

Those numbers show that the architecture can move useful configuration evidence
through an immutable belief update and into command selection. They are not an
acceptance threshold or a general recovery claim.

The support result is intentionally negative. Maximum actual validity
utilization was `1.624`, `1.666`, `1.669`, and `1.631` for stale belief, adapted
belief, adapted point mean, and oracle point mean. Maximum returned one-step
robust utilization was `1.633`, `1.674`, `1.670`, and `1.631`; reaction-horizon
utilization reached `1.710`, `1.837`, `1.712`, and `1.652`. All traces remained
finite and bounded with no solver fallback, but every trace left support and
some steps had no enumerated projection satisfying the progress condition.

This is the behavior the diagnostic should expose. Removing the independent
quadrotor recovery law eliminates the earlier appearance that the generic NMPC
path kept the experiment inside support. The benchmark now establishes useful
adaptation evidence and a clear controller limitation—not an invariant-set,
envelope-expansion, flight-safety, or throw-to-recover result.

On the recorded run, the uncertainty-bearing adapted belief cost roughly twice
the point-model solve per step against a `20 ms` model period. Alternative-path
compilation was prewarmed. Absolute solve times depend on the host and its
load, so they are kept only in the results artifact, where the benchmark marks
them nondeterministic and excludes them from its comparison. This is not a
hard real-time claim.

## Reproduce

```bash
uv run glassbox-adaptive-recovery \
  --output docs/adaptive-recovery-results.json
```

The checked-in [result artifact](../results/adaptive-recovery-results.json) records the
scenario contract, environment, evidence, all four recovery traces, direct
comparisons, observations, and limitations. Its `acceptance_gate`,
`flight_safety_claim`, and `throw_to_recover_claim` fields are all false.
The artifact also stores source and scenario SHA-256 fingerprints. Its test
regenerates the complete report and compares every deterministic field after
excluding only platform metadata and measured wall-clock timings, so changed
results must be recorded deliberately without becoming a performance gate.
