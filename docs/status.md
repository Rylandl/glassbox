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

## Migration in progress

The user requested completion of accumulator validation and migration, followed
by a review of the remaining network structure. Qualification runs in
`codex/accumulator-migration` at `/private/tmp/glassbox-accumulator-migration`.
No production implementation has been replaced yet.

The corrected-initialization online qualification **passes every frozen accuracy
and speed check** on all six streams / 3,137 updates. Equal-family primary error
improves **4.11%**, body-rate error **4.01%**, velocity **4.22%**, and worst-decile
error **5.33%**. Quad median/p95 update time improve **47.63% / 49.76%**;
fixed-wing median/p95 improve **16.60% / 16.01%**. Baseline predictions and updates
reproduce adopted v8 exactly. Source `76f06f9`, protocol `dbc254d`, sealed pack
`artifacts/accumulator-migration-v1/online`, authority
`65be54facc67666bdc2a60d55e414b5b47574eed85523dd09c9efc04fac73024`.
All 471 package/harness tests pass.

Offline qualification is frozen in `45834a3`: six full-budget fits, fresh v8 and
corrected accumulator for Dart/Crazyflow/Cascade, with exactly matching saved
training/development caches. Compare both fresh fits and the deployed incumbent
against every saved flight forecast/response query, held-out Dart recordings and
numerical command derivatives. Fits are running under
`artifacts/accumulator-migration-v1/offline-fits`; no offline outcome or adoption
is claimed. Unchanged nominal Dart and native precision checks remain pending.

## Latest completed iteration

**Fixed-wing angular diagnosis — initialization and solver budget dominate the
observed architecture difference.** The [controlled experiment](accumulator-angular-diagnosis.md)
loads identical saved starting sessions for seven arms and completes all 3,150
causal updates across the two known fixed-wing tapes. Both unchanged arms reproduce
every saved prediction, model fingerprint and report exactly. Scientific harness
`353c44e`, coordinator correction `f74449f`, and saved-data explanation `1deac89`
live on `codex/accumulator-angular-diagnosis`; four diagnostic tests pass.

The accumulator's current-feature projection was initialized **1.880× stronger**
than the same rows in v8 because its normalization denominator changed with the
smaller matrix. Restoring only that initial scale changes equal-case angular
error from **5.58% worse to 7.27% better than v8**, with 8.42% better velocity.
All other parameters, architecture and the 16-iteration PCG budget remain the same.
The stronger initialization also saturates more prefix memory-drive values.
This supports a conditioning explanation, not proof of an irreducible capacity gap.

Increasing PCG from 16 to 64 iterations reduces angular error **55.99% for v8**
and **58.27% for the original accumulator**, relative to each arm's 16-iteration
result. At 64 iterations the architectures differ by only **0.12%** on aggregate
angular error. Both cached fitting error and next-observation prediction improve.
Removing recurrent feedback or lag drives produces mixed, path-dependent effects;
these tapes do not establish that nonlinear feedback is required.

All arms and the one coordinator bookkeeping failure are retained. Independent
verification uses saved arrays without importing JAX or calling a model. This is
diagnosis on previously inspected tapes, not new generalization evidence. No
quad runs of the modified initialization, offline fits or Dart trials were run.
No 64-iteration speed claim is made. The original [paired accumulator screen](accumulator-memory.md)
measured 47.69% lower quad median update time and 49.46% lower p95 at 16 iterations;
its failed frozen accuracy flags remain unchanged. V8 remains maintained.

## Next scientific gap

**Qualify the accumulator with controlled initial memory-drive scale.** The next
single candidate should restore or control that scale generically, retaining
all lag inputs and the measured speed advantage. Freeze a full six-stream paired
comparison before changing the initializer; follow with matched offline
forecast/response and Dart evidence if the result warrants qualification. No
consumer knobs or platform branches. The known-tape diagnosis does not select a
universally optimal scale.

Solver conditioning is the subsequent efficiency target: recover the much better
64-iteration solution with less work. Merely increasing the budget is an accuracy
reference, not evidence of faster fitting. Fixed-wing absolute errors, calibration
and live recovery remain separate open gaps.
