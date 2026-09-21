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
| Reference control | The fixed supported shared-physics revision passes the nominal Dart task: 6.19 mm miss, 1.33° axis error, contact at 1.1944 s; all 4,144 gradients, 40 selected means and 120 native steps are finite. Exact replay and independent audit qualify numerical compatibility and task success. 26/40 solves converge. One known task/seed; neighborhood robustness and real-time execution remain unqualified. Original shared-v1 and the last fitted refinement retain their failures; structured context passes at 8.17 mm. | General learner meets Dart's declared contact task through Dart's unchanged controller. |
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

The [supported Dart compatibility trial](supported-dart-compatibility.md) is
**complete: numerical compatibility and nominal task success both pass**. The
fixed saved revision executes all 120 intervals through the unchanged controller,
contacts at 1.194419 s, and meets every original limit: 6.19 mm miss, 1.33°
top-axis error, 1.490 m/s normal speed and 0.1861 m/s tangential speed. Every
one of 4,144 actual objective/gradient callbacks stays finite; all 40 full selected
means and 120 native steps are finite. Twenty-six solves converge and fourteen
reach the unchanged iteration limit. Exact replay covers every actual call plus
the 50 imported prelude steps, independent saved-array reconstruction agrees,
and all four numeric alterations are rejected. The [result record](harness/supported-dart-compatibility-v1-result.json)
anchors the clean committed implementation, 48 passing tests and full evidence.
No new fit, controller change, scientific restart or source correction occurred.
The control stage takes 36.69 seconds for 1.2 simulated seconds, so real-time
operation remains unqualified. Public promotion remains false.

The [prior cross-vehicle comparison](cross-vehicle-supported-motion.md) preserves
original local response on both known Crazyflow/Cascade cohorts. This task success
does not remove its Cascade wind forecast errors of 0.426–0.434 m/s at 250 ms,
or the prior Dart long-response errors of 3.942 m/s and 9.601 rad/s. Successful
feedback control and model accuracy remain separate evidence.

## Next named gap

**Neighborhood reliability and prediction accuracy around the successful Dart
task.** Keep the saved revision, map and controller fixed. Freeze a modest
nearby-start/target roster, the work budget, all task outcomes and direct
prediction-error reductions before another trial. The concrete question is
whether this known nominal success extends beyond its one inspected task and
command seed, with all attempted conditions visible. The narrow tangential-speed
margin and fourteen budget-limited solves motivate the test; they are not
retroactive failures of this iteration. Report model residuals alongside control
outcomes. No next protocol has been frozen or run yet.

Do not restart the [completed bounded fits](bounded-motion-features.md).
Physical derivative accuracy, long-horizon generalization, shifted flight
forecasts, uncertainty calibration and public recipe adoption remain separate
obligations. One successful consumer task does not finish those rows.

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
