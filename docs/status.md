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
| Accuracy | Shared physics reduces fresh matched aggregate forecast/response errors 43.95%/22.12% versus retained v4. Crazyflow improves 67.96%/45.76%; Cascade improves forecast 1.93% but response worsens 11.83%. Wind and one response-tail guard fail. The latest long-horizon refinement lowers fresh aggregate forecast/response errors only 0.60%/3.18% versus shared-v1, with Crazyflow regressions and Dart response blowups; it is not accepted. | Low held-out physical forecast and command-response residuals across conditions and horizons; application adequacy declared separately. |
| Model usability | Shared candidate supplies `fit`, differentiable batched `predict`, immutable `update`, save/load, canonical signal semantics and error-envelope provenance. Two flight archives load through the integrated API. Public v4's previous numerical qualification boundaries remain unchanged. | Self-contained artifacts usable independently of Glassbox controller internals, with clear scope and measured evidence. |
| Capability | Public v4 passes its 27 frozen synthetic cases. The new rigid-body candidate has analytic integration, causality, arbitrary-command-count, lifecycle and gradient regressions; it does not inherit the arbitrary-channel suite verdict. | Capability appropriate to the declared system class, measured separately from physical readiness. |
| Reference control | Shared physics executes 0.81 s, then fails on a nonfinite planning objective/gradient without contact. V4 fails before execution. The structured causal-history comparator passes with an 8.17 mm miss. One nominal trial per arm. The latest refinement executes 0.78 s, has zero converged solves and no contact. | General learner meets Dart's declared contact task through Dart's unchanged controller. |
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

The [optimizer step-selection experiment](optimizer-step-selection.md) is
**complete and not accepted**. Persistent bounded Armijo backtracking resolves
rejected updates: all three configurations accept 1,000 updates, and selected
weighted development loss falls 3.68% for Dart, 35.19% for Crazyflow and 73.84%
for Cascade. Fresh physical forecast/response errors improve 11.10%/19.91% for
Cascade but worsen 11.15%/17.04% for Crazyflow; the joint aggregate hides those
opposing changes. Crazyflow's velocity-forecast parent-p95 rises 1.94×. Dart
still produces nonfinite predictions on eight of 96 fresh response branches
at 1.2 s, although simulator truth and commands are finite and commands remain
within training coordinate ranges. The unchanged controller fails after 0.78 s
without contact. All 39 required evidence receipts and five actual alteration
classes are verified; the failed control outcome remains failed. The
[result record](harness/optimizer-step-selection-v1-result.json) anchors the
complete evidence and separate training/evaluation source identities. An
import-only evaluation correction reused the three saved fits and nine completed
training checks without refitting. No new model is promoted or integrated.

## Next named gap

**Dart recursive command-response stability.** The optimizer blocker is resolved;
further long fitting is not the next step. The next iteration should freeze a
bounded **no-fit recurrence-attribution diagnostic** using the nine failed
candidate/shared branch identities, matched factual controls and both saved
models. Trace affine, quadratic and nonlinear acceleration contributions and
state growth; compare 1/2/4 integration substeps while preserving command and
history updates at the original observation intervals. This is a convergence
diagnostic, not a menu from which to select a better-scoring model. No successor
protocol is frozen yet.

Saved predictions show huge velocity/rate growth before NaNs while rotation
matrices remain nearly orthogonal. Unrestricted learned state feedback and
explicit integration are plausible contributors; the existing outputs do not
isolate the cause. All eight candidate failures occur within recorded command
coordinate ranges, which does not establish joint state/action coverage. One
additional branch stays finite only by reaching millions in motion units, so
finiteness alone is insufficient. Use the diagnostic to choose one model
intervention before another fit. Retain Crazyflow short-horizon response/tail
regressions and Cascade crosswind accuracy as explicit obligations.

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
