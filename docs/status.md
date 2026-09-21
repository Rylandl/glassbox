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

The [conditioning attribution diagnostic](conditioning-attribution.md) is
**complete and its evidence is qualified**. Four saved revisions are compared
on all 189 previously inspected Dart queries, with no fitting, gradients,
simulator collection or controller trials. Conditioning alone removes the
original model's nine nonfinite and four extreme finite response trajectories;
both bounded revisions have none. It also reduces 1.2-second pilot forecast
prefix body-rate RMSE from 3.320 to 1.536 rad/s. This is a measured repair of
observed runaway on known evidence, not a stability or fresh-accuracy guarantee.
At 250 ms, conditioned pilot response prefix body-rate error worsens 27.06%;
further fitting adds another 18.84% rate and 33.92% velocity error relative to
the conditioned start. Full 1.2-second response endpoint errors remain
3.992 m/s and 8.920 rad/s. The [result record](harness/conditioning-attribution-v1-result.json)
preserves every query, metric, failure and comparison: all 378 historical
reference predictions reproduce exactly, all 756 predictions replay exactly,
the independent reduction and four actual alteration checks pass, and 80
preflight tests pass. There are no pending fits or prediction calls. The
previous bounded-fit prerequisite remains failed; no model is promoted and
all earlier fresh physical/control verdicts remain unchanged.

## Next named gap

**Preserve local command-response fidelity while bounding recursive growth.**
The leading hypothesis is a smooth bounded motion representation that is
exactly the identity inside the recorded training envelope. The current tanh
map perturbs supported features, increases the starting training objective
about 177-fold, and worsens short response accuracy despite fixing runaway.
Freeze one replacement function and derive its support solely from the existing
training cache. Test exact feature/derivative identity within that support,
then compare saved-weight predictions on the same complete 189 known queries
before any further fitting. Keep filters, learned weights, existing norms,
mechanics and timing unchanged. This is a representation hypothesis: recursive
states may leave support, and preserving local features need not restore full
rollout accuracy. No next protocol has been frozen or run yet.

The [completed bounded fits](bounded-motion-features.md) also establish a
separate generalization problem: Dart's large training decrease worsens the
development objective and much of the measured command response. Do not repeat
those fits. After the representation tradeoff is resolved, fresh accuracy work
must retain Crazyflow response/tail regressions and Cascade crosswind
performance, and qualify Dart control separately.

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
