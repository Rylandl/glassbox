# Current state

Updated 2026-09-21. Read [the charter](charter.md) first.

The supported shared-physics learner is the **adopted single implementation**.
This follows the user's explicit decision after the successful Dart trial.
Public v4, structured model catalogs, belief objects and old experiment/control
frameworks have been removed. Adoption
does not rewrite prior frozen failures or establish arbitrary-system readiness.

| Area | Evidence and remaining gap |
| --- | --- |
| Recipe | One shared rigid-body formulation with learned forces/effective accelerations, command filtering and memory. Separate weights per configuration; no vehicle-type or layout inputs. |
| Accuracy | Original shared physics reduced matched aggregate forecast/response errors 43.95%/22.12% versus v4. The supported map preserves original local response on both inspected Crazyflow/Cascade cohorts. Cascade wind forecast velocity RMSE remains 0.426–0.434 m/s at 250 ms; known Dart 1.2 s response errors remain 3.942 m/s and 9.601 rad/s. Original wind/tail acceptance failures remain historical facts. |
| Dart | The unchanged supported model now reaches **0.720 mm** nominal miss through a radius-aware lateral objective and 10 ms feedback: 0.93° axis error, 0.126 m/s tangent speed, contact at 1.192093 s. All 9,874 gradients, 120 selected means and 120 native steps are finite; 103/120 solves converged. Finer float64 replays reach 0.756 / 0.755 mm and converge within 0.152 micrometers / 0.228 microseconds. The original float32 fine-grid convergence failure remains recorded. One known task with an existing seed; neighborhood reliability remains open. |
| Runtime | The precision trial took 59.24 s for 1.2 simulated seconds, including durable callback recording. It uses 120 solves versus the baseline’s 40, with the same 100-iteration cap per solve. Real-time execution is not qualified. |
| Updates | Offline updates preserve immutable revisions and development roles. The new causal streaming fitter improves equal-family error 54.34% versus frozen startup on six known tapes, with no family-specific model. Fixed-wing absolute accuracy and quad update latency remain insufficient for live adoption. |
| Calibration | Fresh fits estimate envelopes on development data also used for checkpoint selection. Independent coverage remains open. Adopted migrated revisions carry no transferred old-map envelope. |
| System scope | Canonical rigid-body velocity, angular rate and orientation with arbitrary ordered command dimensions. Articulated/flexible systems and broad JSBSim coverage remain unproved. |

## Completed iteration

**Single implementation and repository cleanup.** At cleanup, the package had
14 Python modules / 2,216 lines with one dynamics implementation. All 12,768 saved flight
arrays, 40 selected Dart means and 4,144 actual objective/gradient callbacks
reproduce exactly. The installed wheel passes 76 tests on Python 3.12 and 3.13.
Five obsolete artifact roots and 60 retired experiment worktrees were deleted
after preserving the models, relevant recordings and current proof in the
192 MB baseline pack. See [the cleanup result](cleanup.md),
[baseline identity](baseline.json) and [replay instructions](../CONTRIBUTING.md#replay-the-adopted-baseline).
No new long fit, controller optimization or simulator trial was run.

## Completed precision iteration

**Dart contact precision below 1 mm.** The closed-loop nominal trial reached
0.720 mm without changing the learner or model weights. The winning consumer
change tightens lateral position cost with target radius while retaining the
original normal-depth scale, plus feedback at every 10 ms observation. All
original alignment, velocity and deadline gates pass. The original 6.19 mm
baseline was reproduced exactly before interventions. All six native trials,
including unsuccessful changes, are recorded in [the result index](dart-precision.json).

The first fine-grid audit reached 0.607 / 0.696 mm, but failed the declared
20 micrometer / 20 microsecond convergence gate. The separately frozen
[arithmetic-precision audit](harness/dart-resolution-precision-v1.json) passes
with the exact issued commands and rounded plant coefficients: 0.7556 / 0.7555 mm
miss, 0.152 micrometer point convergence and 0.228 microsecond time convergence.
The original float32 failure remains recorded. No model fit or package change.
The winning objective and four regression tests are applied in Dart; 10 ms
feedback is specified by this benchmark, not changed in Dart launcher defaults.
See [the result](dart-precision.md). The full suite passes 105 tests.

## Completed online-fitting iteration

The same shared dynamics engine now supports bounded causal streaming fits on
four-command quads and three-command fixed wings. The first Adam candidate failed;
its result is preserved. Full-cache damped Gauss-Newton improves the equal-family
velocity/rate aggregate 54.34% versus identical frozen startup models and 83.38%
versus that first candidate. Eleven of twelve primary case/metric comparisons
improve over frozen; `fixedwing-80` rate error regresses 3.41% and remains included.
All 3,137 observations are scored before assimilation. Inputs and frozen predictions
match bitwise across candidates; both packs verify without fitting. All 138 tests
pass. The offline learner, original baseline and Dart model are unchanged.

This passes the frozen relative-improvement gate, not broad readiness. Fixed-wing
errors remain 6.9–21.6 times kinematic velocity error and 7.9–12.9 times rate error.
Quad update p95 is 26.9–27.9 ms against 10 ms sampling; fixed-wing p95 is 3.70–3.75 ms
against 50 ms. One quad tape is floor-truncated, and the in-flight geometry change
occurs in weakly excited hover. No current-learner closed-loop trial has run.
See [the complete result](online-fitting.md) and [evidence identities](online-fitting.json).

## Active scientific gap

**Physical-vector loss conditioning.** The v3 coordinate correction failed its
frozen gate: equal-family error was 1.41% worse than adopted v2 (quad 17.26% better,
fixed-wing 24.28% worse). Fixedwing-81 rate error increased 6.51 times. All cases
remain included in [the v3 result](online-fit-v3.json).

The frozen [v4 protocol](harness/online-fit-v4.json) now tests a fixed scale for
each physical vector and radial Huber loss, keeping v3's other changes. It removes
startup-axis imbalance and weights velocity, rate and orientation equally.
Acceptance remains against adopted v2, with v3 attribution reported separately.
No v4 fit has started. Motion support, latency and control integration remain open.
