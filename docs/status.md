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
| Preservation | Saved baseline contains the unchanged adopted models, recordings and replay evidence: 12,768 flight arrays, 40 Dart trajectories and 4,144 gradients. Only `online.py` changes from v6; all 14 other package files are byte-identical. All **414 tests pass**, and the complete saved baseline replays exactly. |

## Latest completed iteration

**Bounded online solve with backtracking — completed; v8 adopted.** The
[frozen six-tape run](online-fitting.md) tests one v6-based candidate with sixteen
PCG iterations and first-acceptable step scales 1, 1/2, 1/4, 1/8 and 1/16. The
prior, conditioning, initialization, dynamics, trust bound and rollback/damping
rules stay fixed. No platform branch, consumer option or parameter sweep is added.

The primary ratio is **0.58240** against v6: quad 0.46812, fixed-wing 0.72459.
The separate tail/orientation/rotation-consistency gates also pass. Every one of
3,137 updates accepts; 3,098 use the full step, 35 half and 4 quarter. All quad
updates use full steps, so their gains come from the longer solve without active
shortening. Backtracking is used on 39/450 fixed-wing updates and adds 43 objective
evaluations overall. Solver work increases fourfold to 50,192 scheduled iterations;
quad p95 update time rises to 78–93 ms. This passes the declared accuracy decision,
not real-time quad qualification.

Both saved-data verifiers pass, including 400 authenticated payloads, 9,411 journal
events and 23 prospective snapshots; no refitting is used. All 414 tests pass and
the offline/Dart baseline replays exactly. Four descriptive quantile/maximum
regressions remain, including FW80 maximum velocity error 1.297→1.578 m/s. Quad125
and quad-change share their first 4 s and differ only slightly after the declared
change; these known cases do not establish independent generalization. No
current-learner online closed-loop recovery trial has run.

## Next scientific gap

**Cost of the adopted v8 online solve.** Freeze a component profile on saved
sessions before changing implementation. Measure conditioning, linearization,
repeated curvature products and exact trial evaluation separately; identify
reusable numerical work while preserving the sixteen-step proposal and prediction
accuracy. The quad 10 ms cadence is still far out of reach on this measured runtime.
Do not silently lower the budget, add tuning options or trade away the measured
gains. A changed numerical recipe needs a new prospective comparison.

Fixed-wing absolute forecast errors, known-tape coverage, physical derivatives,
calibration and live consumer recovery remain separate open gaps. V6 remains
reproducible from its pinned scientific source; v8 is the one maintained fitter.
