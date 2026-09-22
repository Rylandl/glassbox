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
| Preservation | Saved baseline contains the unchanged adopted models, recordings and replay evidence: 12,768 flight arrays, 40 Dart trajectories and 4,144 gradients. All 15 package source files remain byte-identical to adopted v6. All **334 tests pass**, and the complete saved baseline replays exactly. |

## Latest completed iteration

**Backtracking the saved directions — completed; v6 retained.** The [frozen
read-only test](online-backtracking.md) uses the same 46 fixed-wing snapshots
and step scales 1, 1/2, 1/4, 1/8 and 1/16. Choosing the first step that passes the
existing exact-loss/gain rule makes all 46 longer directions acceptable,
compared with 14 unshortened. The original four-step updates already accepted
46/46; this rescues longer directions, not incumbent acceptance.

Selected retained loss beats the original four-step proposal in **34/46** cases
(17/23 under each recipe), up from 11/46 before backtracking. Twelve selections
still lose to four. Seven of ten anchor-producing updates improve, up from two.
Selected data/prior losses decrease from their starting values in 45/46 and
46/46 respectively. These are assimilated fitting objectives, not future
prediction gains. Forty-two parent directions remain capped rather than
converged; no counterfactual trajectory or new learner was fitted.

The experiment uses 184 new forward residual evaluations and zero new solver,
derivative, conditioning or observe calls. Independent verification passes
6,762 point plus 161 aggregate checks; all 230 residual recomputations pass
frozen tolerances. All 334 tests pass. All 15 learner files and the parent's
28-file source inventory remain unchanged. V7 remains rejected and v6 adopted.

## Next scientific gap

**Forecast value and runtime of bounded online backtracking.** Freeze one
v6-based candidate with 16 PCG steps and the same first-acceptable ladder. Keep
the prior, residual model, conditioning, forecast trust radius and ordinary
rejection/rollback/damping rules fixed; add no consumer tuning option. Sixteen
is an engineering budget choice, not a proven optimum from the longer saved
reference directions.

Run the full six-tape prequential comparison against v6, retaining velocity/rate,
orientation, tails, kinematic and timing results plus actual solver/residual-call
counts. This moves the hypothesis to prediction evidence; no candidate has run
yet. Known-tape improvement does not qualify blind generalization, calibrated
uncertainty, real-time quad fitting or online closed-loop recovery.
