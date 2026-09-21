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
| Runtime | Dart takes 59.24 s for 1.2 simulated seconds. V6 quad update p95 is 28.16–29.53 ms against 10 ms observations; fixed-wing p95 is 4.38–4.60 ms against 50 ms. Real-time quad fitting is not qualified. |
| Online fitting | Working **online v6** reduces equal-family velocity/rate error **21.94% against v4** and 78.93% against frozen startup across six known tapes. All 3,137 targets are scored before assimilation. Worst-decile errors improve 22.96%; angular outliers and individual regressions remain. See [online evidence](online-fitting.md). |
| Updates and calibration | Offline updates preserve immutable revisions and development roles. Online sessions retain bounded causal caches. Development-selected calibration does not establish independent coverage; streaming sessions have no calibrated envelope. |
| Scope | Rigid-body motion across recorded quad/fixed-wing configurations. Broad airframe, articulated-system and arbitrary-system readiness remain unproved. No current-learner online closed-loop recovery trial has run. |
| Preservation | Saved baseline contains the unchanged adopted models, recordings and replay evidence: 12,768 flight arrays, 40 Dart trajectories and 4,144 gradients. All 15 package source files remain byte-identical to adopted v6. All **288 tests pass**, and the complete saved baseline replays exactly. |

## Latest completed iteration

**Causal proposal audit — completed; v6 retained.** The [read-only audit](online-proposal-audit.md)
reconstructs all 46 selected fixed-wing updates exactly, including 28 complete
adjacent session identities and ten predecessor-to-prediction links. Both the
independent NumPy audit and source-bound derivative recomputation pass. No new
learner was fitted and no diagnostic proposal was applied.

Four PCG steps achieve a median 30.25%/11.82% of the longer probe's measured
local quadratic decrease under v6/v7. Yet a longer solve with the same forecast
trust rule beats the original exact retained objective in only 11/46 snapshots
and would be accepted in 14/46; 31/46 losses exceed the starting value. All longer
trusted directions descend data and prior to first order, but exact data loss
falls in only 15/46. The prior falls in all 46. This distinguishes unfinished
linear solves from unreliable nonlinear steps; simply increasing PCG is not
supported. Only four references converge at 64 iterations; 42 cap at 128.

The audit passes 58,568 arithmetic checks and all 46 derivative recomputations
under frozen tolerances. All 288 tests pass; all 15 package files remain
byte-identical to adopted v6. These are known fixed-wing training objectives,
not held-out gains or evidence about quad optimizer behavior. The previous
[v7 candidate](online-fit-v7.md) remains rejected: 13.30% aggregate improvement
against v6 accompanied quad/angular regressions and failed its frozen gates.

## Next scientific gap

**Nonlinear step control at bounded solve cost.** Freeze a small backtracking
ladder along the saved longer directions and check exact data/prior/combined
loss against the original four-step update. Determine whether shortening those
directions makes their local improvement usable before changing the online
solver. Keep the current prior, residuals and data fixed. V6 remains the sole
learner; no new candidate has been fitted.

Any resulting candidate requires a separately frozen six-tape evaluation against
v6, retaining velocity/rate, orientation, tails, kinematic and timing comparisons.
Known-tape improvement does not qualify blind generalization, calibrated
uncertainty, real-time quad fitting or online closed-loop recovery.
