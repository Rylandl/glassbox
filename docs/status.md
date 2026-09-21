# Status: gap against the charter

Updated 2026-09-21. Read [the charter](charter.md) first.

The requested shared-physics vehicle learner is built. One formulation learns
configuration-dependent accelerations and hidden response from recordings while
sharing gravity, body/world transforms and SO(3) integration. It requires no
vehicle family, physical parameters, mixer, actuator layout or tuning choices.
Separate fitted weights per configuration are expected. The
[implementation and measured results](shared-vehicle-physics.md) describe its
research API, scope and remaining limitations.

The public recipe remains `generic-memory-v4-prototype`. The shared-physics
candidate produces large aggregate gains but fails its frozen wind and response
tail regression limits. Its development status must not be confused with a
passed acceptance verdict or arbitrary-system readiness.

| Criterion | Current | Target |
| --- | --- | --- |
| One recipe | One public recipe without tuning or vehicle dispatch. The shared-physics successor is an isolated research formulation with the same fixed fitting procedure across configurations. | One generic learner and consumer contract. |
| Accuracy | Shared physics reduces fresh matched aggregate forecast/response errors 43.95%/22.12% versus retained v4. Crazyflow improves 67.96%/45.76%; Cascade improves forecast 1.93% but response worsens 11.83%. Wind and one response-tail guard fail. The last evaluated long-horizon refinement lowers fresh aggregate forecast/response errors only 0.60%/3.18% versus shared-v1, with Crazyflow regressions and Dart response blowups; it is not accepted. | Low held-out physical forecast and command-response residuals across conditions and horizons; application adequacy declared separately. |
| Model usability | Shared candidate supplies `fit`, differentiable batched `predict`, immutable `update`, save/load, canonical signal semantics and error-envelope provenance. Two flight archives load through the integrated API. Public v4's previous numerical qualification boundaries remain unchanged. | Self-contained artifacts usable independently of Glassbox controller internals, with clear scope and measured evidence. |
| Capability | Public v4 passes its 27 frozen synthetic cases. The new rigid-body candidate has analytic integration, causality, arbitrary-command-count, lifecycle and gradient regressions; it does not inherit the arbitrary-channel suite verdict. | Capability appropriate to the declared system class, measured separately from physical readiness. |
| Reference control | Shared physics executes 0.81 s, then fails on a nonfinite planning objective/gradient without contact. V4 fails before execution. The structured causal-history comparator passes with an 8.17 mm miss. One nominal trial per arm. The last evaluated refinement executes 0.78 s, has zero converged solves and no contact. | General learner meets Dart's declared contact task through Dart's unchanged controller. |
| Live improvement | Public v4 has one demonstrated bounded offline update. Candidate update preserves revisions and development roles in lifecycle tests; physical update improvement and safe controller swapping remain unqualified. | Bounded immutable revisions with held-out improvement/regression checks; downstream adoption evaluated separately. |
| Evidence | Candidate envelopes use development data also used for checkpoint selection and stop at 250 ms. New held-out coverage and physical derivative fidelity remain unqualified. | Measured coverage within a predeclared band with explicit provenance. |
| Lean | Not met: legacy structured comparators and research harnesses remain. | Learner, artifacts/interfaces, telemetry adapters, harness and optional consumers. |

## Baseline evidence

The completed [shared-vehicle experiment](shared-vehicle-physics.md) provides the
saved starting models and current research API. Its wind and response-tail
failures remain obligations. Its 250 ms training and error-envelope provenance
do not qualify accurate longer-horizon forecasts, physical derivatives, contact
success or live model replacement. Public v4 remains the adopted public recipe.

## Latest iteration

The [cross-vehicle supported-motion comparison](cross-vehicle-supported-motion.md)
is **complete and passes its frozen known-data preservation hypothesis**. The
unchanged map uses the original saved weights on all 8,064 Crazyflow/Cascade
queries from both inspected cohorts. Its 360-cell local-response ratio is
0.998619 versus original and 0.989091 versus tanh. Crazyflow is effectively
unchanged from original and 4.15% better than tanh; Cascade is 0.28% better than
original but 2.07% worse than tanh. Both cohorts show that tradeoff. All three
arms remain finite/nonextreme on input-eligible prefixes; missing truth and
263 unusable histories remain visible. This supports transfer of the Dart
preservation fix, with no new fit or simulator/controller call. It leaves real
accuracy gaps: Cascade wind forecast velocity RMSE is 0.426–0.434 m/s at
250 ms across the two known cohorts (parent p95: 0.751–0.778 m/s), and prior
Dart long-response errors remain 3.942 m/s and 9.601 rad/s. The [result record](harness/cross-vehicle-supported-motion-v1-result.json)
anchors 76 passing tests, 12,768 exact historical arrays on each pass, 38,304
exactly replayed arrays, independent physical/hypothesis reductions and four
rejected numeric alterations. No calls remain pending, no source correction
was needed, and no public model or earlier frozen verdict changes.

## Next named gap

**Dart planner compatibility of the fixed supported shared-physics revision.**
Keep the map, saved revision and controller fixed. Freeze the existing task,
planner work budget, objective/gradient finiteness checks and contact criteria
before a no-fit consumer trial. The concrete question is whether this revision
gets past the previous nonfinite planning objective/gradient failure and can
execute the task. Numerical compatibility, task success and model accuracy
remain separate outcomes. A controller success would not settle long-response
residuals, shifted flight forecasts, physical derivative fidelity or calibration.
No next protocol has been frozen or run yet.

The [completed bounded fits](bounded-motion-features.md) still establish a
separate long-horizon training-generalization problem; do not repeat them.
The [supported-map Dart result](supported-motion-saturation.md) removes observed
runaway and recovers local response while worsening long forecasts versus tanh.
Both flight cohorts preserve original local accuracy but contain truth only
through 250 ms. The next consumer check follows those measurements; it does not
assume that bounding six input features bounds the entire rollout Jacobian or
qualifies the model for arbitrary systems.

## Preserved boundaries

- The [public v4 adoption](public-mean-adoption.md) is a policy decision on known
  evidence; the original numerical derivative gates retain their failed verdicts.
- The [independent-recording update](independent-training-recordings.md) qualifies
  one bounded offline intervention, not universal updates or safe live swapping.
- Shared-physics forecasts beyond the fitted 250 ms horizon are recursive means,
  with no extrapolated calibrated envelope. Long-horizon fit and task adequacy
  remain separate from short-horizon improvement.
- The historical 1.47 mm structured Dart result used simulator-applied actuator
  state. The current comparator reconstructs latent response from issued history.
  Dart constrains the contact body's top axis and leaves yaw free; this is not a
  full six-degree-of-freedom pose qualification.
- Signal units, frames, timing and boundaries must be interpretable from telemetry.
  The rigid-body vehicle formulation has not demonstrated every articulated,
  flexible or arbitrary system. JSBSim breadth remains deferred.

The [continuation record](/private/tmp/glassbox-public-mean-qualification/artifacts/2026-09-19/public-mean-qualification-v1-continuation.json)
holds active stage handles and immutable source/evidence anchors. Source-bound
workers and all completed fits remain unchanged.
