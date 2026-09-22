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
| Preservation | Saved baseline contains the unchanged adopted models, recordings and replay evidence: 12,768 flight arrays, 40 Dart trajectories and 4,144 gradients. All **15 maintained package files remain byte-identical to adopted v8**; its exact baseline replay remains applicable. The maintained implementation has 464 passing tests. The separate history-compression candidate passes 474 tests but is not adopted. |

## Latest completed iteration

**Learned temporal compression — smaller, mixed accuracy, not adopted.**
The [frozen architecture screen](history-compression.md) replaces separate lag
inputs with two learned temporal summaries, preserving the generic mechanics,
nonlinear head, recurrent memory and online solver. Quad parameter count falls
**10,130 → 3,894 (61.56%)**; fixed-wing changes 3,399 → 3,403. Source `3fc490a`
is retained on `codex/learned-history-compression`, outside the maintained learner.

All six original tapes and 3,137 causal predictions/updates complete. Against
adopted v8, equal-family combined velocity/rate error falls **5.13%**, and
worst-decile error falls **8.56%**. Velocity alone improves 16.66%, while body-rate
error worsens 8.00% and orientation worsens **6.71%**. Size, primary, family, tail
and rotation/rate gates pass; the aggregate orientation gate fails its frozen
2% allowance. This is not a single-case veto. Every case and regression remains
in the saved result, verified without fitting; 474 candidate tests pass.

Warm quad updates remain **85.52–86.05 ms** median (89.03–90.56 ms p95), with no
qualified speedup. No offline fits, Dart trials or later paired timing were run
after the screen failed. Neither information loss nor changed optimizer
coordinates is established as the cause of the angular regression.

The preceding arithmetic-only batching candidate `c631a1c` also remains
unadopted: 198 gradient and 17 objective arrays fail saved numerical-preservation
bounds. Its timing comparison had mismatched background load. Both experiments
and their sealed authorities are retained in the [result index](history-compression.json).
The maintained package remains unchanged.

## Next scientific gap

**Compact temporal memory with accurate angular response and cheaper derivatives.**
Investigate stable accumulators with learned time constants as the next single
architecture candidate. Before freezing it, inspect compiled derivative work
and the angular failures of rank-two compression. Parameter count alone has not
explained update cost. The prior [cost diagnosis](online-cost-profile.md) places
most warmed update latency inside the sixteen-step compiled proposal, with one
nonlinear linearization reused throughout; the history-specific share remains
unmeasured.

Preserve shared rigid-body mechanics, arbitrary ordered command dimensions and
one recipe. No consumer choices or platform branches. Caching recurrent state
requires accounting for changes to learned timescales and feature coordinates.
A successful online screen must still pass matched offline forecast/response
and Dart qualification before replacing the adopted model. Fixed-wing absolute
errors, calibration and live recovery remain separate open gaps.
