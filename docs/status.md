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
| Updates | Fixed-budget refitting produces immutable revisions and retains development roles. Accuracy improvement and live adoption require new evaluation. |
| Calibration | Fresh fits estimate envelopes on development data also used for checkpoint selection. Independent coverage remains open. Adopted migrated revisions carry no transferred old-map envelope. |
| System scope | Canonical rigid-body velocity, angular rate and orientation with arbitrary ordered command dimensions. Articulated/flexible systems and broad JSBSim coverage remain unproved. |

## Completed iteration

**Single implementation and repository cleanup.** The package is now 14 Python
modules / 2,216 lines with one dynamics implementation. All 12,768 saved flight
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

## Active scientific gap

**Causal online fitting for Throw, then fixed-wing support.** The user redirected
work from Dart neighborhood trials to the older glassbox-throw project. Its
original controller uses an older, quad-specific identifier; the maintained
learner currently starts a full batch fit from scratch on each update.

The frozen [online-fit-v1 protocol](harness/online-fit-v1.json) tests bounded,
persistent optimization of the same shared-physics model. Four newly collected
quad streams and two known Cascade fixed-wing streams compare pre-assimilation
predictions against an identical learner frozen after the startup prefix. The
candidate sees observed motion and issued commands, with no platform data.
This is identification evidence; candidate-controlled recovery and fresh
fixed-wing generalization remain separate work. No scientific run has started.
