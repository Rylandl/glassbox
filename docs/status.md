# Current state

Updated 2026-09-21. Read [the charter](charter.md) first.

The supported shared-physics learner is the **adopted single implementation**.
It learns each configuration separately from observations and issued commands,
without vehicle-family, layout, mass or inertia inputs. The model is the product;
controllers and simulators remain external.

| Area | Evidence and remaining gap |
| --- | --- |
| Recipe | Shared gravity, rigid-body kinematics and coordinate transforms; learned command response, effective accelerations and memory. Arbitrary ordered command dimensions; no platform dispatch. |
| Offline accuracy | Original shared physics reduced matched aggregate forecast/response errors 43.95%/22.12% versus the retired model. The supported map preserves local response on the inspected Crazyflow/Cascade cohorts. Cascade wind velocity RMSE remains 0.426–0.434 m/s at 250 ms; known Dart 1.2 s response errors remain 3.942 m/s and 9.601 rad/s. |
| Dart | The unchanged supported model reaches **0.720 mm** nominal contact miss through an external radius-aware objective and 10 ms feedback. Fine float64 replays reach 0.756 / 0.755 mm and converge within 0.152 micrometers / 0.228 microseconds. One known task/seed; neighborhood reliability remains open. See [precision evidence](dart-precision.md). |
| Runtime | Dart takes 59.24 s for 1.2 simulated seconds. V8 quad update p95 is 78.38–92.55 ms against 10 ms observations; fixed-wing p95 is 9.86–9.89 ms against 50 ms. Real-time quad fitting is not qualified. |
| Online fitting | Adopted **online v8** reduces equal-family velocity/rate error **41.76% against v6** and 87.73% against frozen startup across six known tapes. All 3,137 targets are scored before assimilation; all twelve primary cells improve. Worst-decile error falls 42.86%, orientation RMSE 53.53%. Fixed-wing velocity/rate errors remain 1.6–4.5× the no-fit kinematic baseline; rare errors still regress. See [online evidence](online-fitting.md). |
| Updates and calibration | Offline updates preserve immutable revisions and development roles. Online sessions retain bounded causal caches. Development-selected calibration does not establish independent coverage; streaming sessions have no calibrated envelope. |
| Scope | Rigid-body motion across recorded quad/fixed-wing configurations. Broad airframe, articulated-system and arbitrary-system readiness remain unproved. No current-learner online closed-loop recovery trial has run. |
| Preservation | Saved baseline contains the unchanged adopted models, recordings and replay evidence: 12,768 flight arrays, 40 Dart trajectories and 4,144 gradients. All **15 package files remain byte-identical to adopted v8** during the cost profile, so its previous exact baseline replay remains applicable. All **464 tests pass**. |

## Latest completed iteration

**V8 cost diagnosis — measurements complete; component qualification fails.**
The [saved-session profile](online-cost-profile.md) measures all 35 frozen
checkpoints: 29 genuine next observations and six final-cache probes. All 725
native replays exactly reproduce original v8 model, report and session state.
An independent audit validates 257 profile payloads, 400 parent payloads and
12,675 raw timing samples without model calls. The learner is unchanged.

Quad initial medians are **75.44 ms for the native proposal versus 77.38 ms for
the public update**; the nested public snapshot costs only 0.054 ms. Passing
setup/trust diagnostics and a 4.68 ms cached curvature application consistently
point to repeated derivative work. The sixteen-step proposal already reuses one
nonlinear linearization. Separate component times are nonadditive; the fraction
specifically spent propagating through history is not yet measured.

Four initial-quad standalone solver prefixes fail the frozen numerical tolerance
and are excluded from conclusions. Every full instrumented/native proposal and
trust checkpoint matches exactly; all other comparisons pass. The failed first
attempt remains sealed. A committed failure-collection change completed the same
roster without changing kernels or criteria, while retaining the failed overall
qualification. Across both attempts: 726 native replays, zero initializations.
This is a useful cost diagnosis, not a speedup or new real-time qualification.

## Next scientific gap

**Reduce repeated derivative application cost.** Freeze one prospective candidate:
cache transport through the learned-memory recurrence and batch history parameter
JVP/VJP calculations. Preserve the model, conditioning, prior, sixteen solver
steps, trust bound and backtracking. Measure construction cost and total proposal
latency, then verify numerical and forecast preservation against adopted v8.
History transport is a plausible target, not a measured share or promised route
to 10 ms. Reordered arithmetic needs qualification; do not lower solver effort,
add consumer choices or trade away measured gains.

Fixed-wing absolute forecast errors, known-tape coverage, physical derivatives,
calibration and live consumer recovery remain separate open gaps. V8 remains
the one maintained fitter; this iteration changes profiling tools only.
