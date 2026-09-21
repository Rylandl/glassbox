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

The [support-preserving saturation experiment](supported-motion-saturation.md)
is **complete and passes its frozen known-data hypothesis**. The new feature
map is exactly the identity inside each training-derived motion envelope and
smoothly approaches the same bound outside it. All weights, normalizations,
filters and mechanics are preserved; no fit or new simulation occurs.
All 1,075,248 observed training motion coordinates are bitwise unchanged in
float32 and float64. On all 189 known Dart queries, the map retains zero
nonfinite or extreme-growth trajectories and largely recovers original local
response accuracy: the separate pilot/task short-response ratios versus the
previous tanh bound are 0.91647/0.98765, and versus shared-v1 are
1.00172/1.00000018. This is a tradeoff: every 1.2-second forecast prefix group
worsens versus tanh, including rate errors by 27.91%/14.63%/36.78% on
pilot/task/test. Long pilot response endpoint errors remain 3.942 m/s and
9.601 rad/s. The [result record](harness/supported-motion-saturation-v1-result.json)
anchors all evidence: 124 tests pass, all 567 historical reference predictions
reproduce exactly on both passes, all 756 predictions replay exactly, the
independent reductions and four distinct alteration checks pass. The final
hypothesis is independently verified after qualification. No calls or fits
remain pending, no public model is promoted, and earlier frozen verdicts
remain unchanged.

## Next named gap

**Cross-vehicle preservation of local response under bounded recurrence.**
Keep the exact supported map fixed and compare saved-weight original, tanh and
supported revisions on the complete Crazyflow and Cascade condition rosters.
Use each configuration's existing training cache to derive its support; change
no weights, filters, mechanics, timing or consumer contract. Freeze matched
horizons, precision, physical forecast/response errors, tail metrics, full
population/failure accounting and decision rules before execution. This tests
transfer of the Dart tradeoff before more Dart-specific tuning or expensive
fitting. Existing inspected flight evidence stays known evidence, and any
later fresh evaluation must be declared separately. No next protocol has
been frozen or run yet.

The [completed bounded fits](bounded-motion-features.md) still establish a
separate long-horizon training-generalization problem: Dart's training decrease
worsens development and much of its measured response. Do not repeat those
fits. The supported map's remaining long errors, Crazyflow response/tail
regressions and Cascade crosswind performance need direct measurement; Dart
control, physical derivatives and calibration retain their own obligations.

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
