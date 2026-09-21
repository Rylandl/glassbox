# Status: gap against the charter

Updated 2026-09-20. Read [the charter](charter.md) first.

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
| Accuracy | Shared physics reduces fresh matched aggregate forecast/response errors 43.95%/22.12% versus retained v4. Crazyflow improves 67.96%/45.76%; Cascade improves forecast 1.93% but response worsens 11.83%. Wind and one response-tail guard fail. | Low held-out physical forecast and command-response residuals across conditions and horizons; application adequacy declared separately. |
| Model usability | Shared candidate supplies `fit`, differentiable batched `predict`, immutable `update`, save/load, canonical signal semantics and error-envelope provenance. Two flight archives load through the integrated API. Public v4's previous numerical qualification boundaries remain unchanged. | Self-contained artifacts usable independently of Glassbox controller internals, with clear scope and measured evidence. |
| Capability | Public v4 passes its 27 frozen synthetic cases. The new rigid-body candidate has analytic integration, causality, arbitrary-command-count, lifecycle and gradient regressions; it does not inherit the arbitrary-channel suite verdict. | Capability appropriate to the declared system class, measured separately from physical readiness. |
| Reference control | Shared physics executes 0.81 s, then fails on a nonfinite planning objective/gradient without contact. V4 fails before execution. The structured causal-history comparator passes with an 8.17 mm miss. One nominal trial per arm. | General learner meets Dart's declared contact task through Dart's unchanged controller. |
| Live improvement | Public v4 has one demonstrated bounded offline update. Candidate update preserves revisions and development roles in lifecycle tests; physical update improvement and safe controller swapping remain unqualified. | Bounded immutable revisions with held-out improvement/regression checks; downstream adoption evaluated separately. |
| Evidence | Candidate envelopes use development data also used for checkpoint selection and stop at 250 ms. New held-out coverage and physical derivative fidelity remain unqualified. | Measured coverage within a predeclared band with explicit provenance. |
| Lean | Not met: legacy structured comparators and research harnesses remain. | Learner, artifacts/interfaces, telemetry adapters, harness and optional consumers. |

## Current evidence

The frozen [shared-vehicle protocol](harness/shared-vehicle-physics-v1.json) changes
one named mechanism: **vehicle mechanics and recursive planning horizon**. Four
fixed fits completed: two physical candidates and matched v4/shared candidates
on Dart's original recordings. Physical v4 controls were loaded without refitting.
The candidate uses the same prepared caches, actual objective scales/weights and
1,000 gradient attempts as its comparator. This tests complete learning recipes,
not an isolated causal attribution to gravity or rotation integration alone.

On untouched +14M flight recordings, 135/180 endpoint comparisons and 15/20 scope
aggregates improve. The joint forecast/response ratio is 0.66071. The failed
checks are Cascade wind forecast/response scope ratios 3.73823/1.64347 and primary
rotation-response parent-p95 ratio 1.52989, all against a 1.5 limit. Cascade wind
250 ms velocity error rises 0.09547 → 0.44723 m/s; this is a material deficit.
All four Crazyflow angular retention checks pass. Crazyflow completes 66/84
parents, with 18 altitude failures; Cascade completes 84/84. Both arms use
identical available truth. Both exact physical replays, the independent reduction
of 257,040 raw rows and all four alteration challenges pass without refitting.

Dart uses eight three-second training recordings and two development recordings.
The 1,536 windows overlap and are not independent experiments. V4 rejects every
permitted optimizer proposal and remains at its initialization; the shared model
selects step 400 with weighted development loss 0.00925356 versus 0.10027121.
Untouched test-10/test-11 forecasts reduce 250 ms velocity/rate errors from
0.64790/1.02180 to 0.08772/0.83528 in m/s and rad/s. At 1.2 s, errors still reach
4.36472 m/s and 10.29175 rad/s. All 72 forecasts and command JVPs per model are
finite; neither this nor lower residuals establishes contact-task adequacy.

The completed Dart trials preserve the unchanged controller and original seed.
The shared model has 27 completed solves, only one converged; the structured
comparator has 40, with 38 converged. Neither receives simulator actuator state;
both use ideal physical motion observations. The composite control replay is
exact, with zero optimizer calls. Harness import and diagnostic-serialization
failures are preserved; the saved first solve was resumed without repetition.
All four fits, physical evidence audits, Dart forecasts and required replays are
complete. The [result record](harness/shared-vehicle-physics-v1-result.json) anchors
the evidence. This iteration is finished; the model remains a research candidate.

## Next named gap

**Dart task-horizon command-response fidelity.** The shared model is fitted over
250 ms but the planner requires 1.2 s. In the saved initial plan, 33/120 command
rows leave at least one observed training channel range; channel differences
reach 0.313 versus training maxima of 0.098–0.126. Its predicted angular rates
leave the training ranges after 0.58 s while orientation remains valid. The
learned acceleration field must stay accurate and numerically well conditioned
over the task's horizon and commands. These observations do not isolate the
precise nonfinite trigger: failed optimizer proposals were not saved.

The next iteration should freeze one targeted remedy and direct long-horizon
command-response evidence. Retain Cascade crosswind and calm response-tail
failures as regression obligations: the calm training/development collection
does not cover the crosswind body velocities, and errors appear already at
50 ms. Neither a remedy nor a sufficient training collection has been established.
No next experiment has been frozen or run.

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
