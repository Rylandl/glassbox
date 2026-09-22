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
| Preservation | Saved baseline contains the unchanged adopted models, recordings and replay evidence: 12,768 flight arrays, 40 Dart trajectories and 4,144 gradients. All **15 maintained package files remain byte-identical to adopted v8**; its exact baseline replay remains applicable. The maintained implementation has 464 passing tests. The separate accumulator candidate passes 471 tests but is not adopted. |

## Latest completed iteration

**Learned accumulator memory — nearly twice as fast on quads; fixed-wing angular
regression remains unexplained.** The [paired screen](accumulator-memory.md)
retains all explicit lag inputs and replaces nonlinear hidden recurrence with
eight learned exponential accumulators. Quad parameters fall 10,130 → 8,714;
fixed-wing parameters fall 3,399 → 3,103. Scientific source `808772b` is retained
on `codex/accumulator-memory`, outside the maintained learner.

Both arms complete all six tapes and 3,137 causal predictions/updates. The baseline
reproduces all original v8 predictions, post-update fingerprints and reports
exactly. Exclusive alternating 25-update blocks reduce quad median latency
**47.69%** and p95 **49.46%**; all frozen speed checks pass. Independent saved-data
verification confirms 127 granted intervals per arm with no overlap.

Quad combined velocity/rate error improves 0.29%. Fixed-wing velocity improves
11.56%, but angular-rate error worsens **5.58%** and rotation/rate defect worsens
9.74%. Equal-family combined error improves 1.84%; aggregate rate (+2.49%) and
rotation/rate defect (+4.25%) fail their frozen 2% allowances. The failed screen
is preserved. No offline fits or Dart trials were run. All 471 candidate tests
pass; the maintained package remains byte-identical to adopted v8.

The accumulator remains the leading candidate because the measured speed/accuracy
tradeoff warrants further investigation. V8 remains the validated implementation
pending explanation of the angular regression and broader qualification. The
prior [rank-two compression](history-compression.md) and arithmetic batching
experiments remain unadopted; their evidence is retained in that result index.

## Next scientific gap

**Explain the accumulator's fixed-wing angular regression.** First localize excess
error by time and axis in the saved paired predictions, and compare initial
shared parameters, normalization, and optimizer decisions. Then freeze controlled
interventions that separate optimization/initialization effects from altered
memory representation. Preserve every attempted result and distinguish diagnostic
replays on known tapes from new generalization evidence.

Preserve shared rigid-body mechanics, arbitrary command dimensions and one recipe;
no consumer options or platform branches. The accumulator still needs matched
offline forecast/response and Dart qualification before replacing the adopted
model. Fixed-wing absolute errors, calibration and live recovery remain open.
