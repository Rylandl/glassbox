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
| Dart | The supported revision completes the nominal task: 6.19 mm miss, 1.33° axis error, contact at 1.194419 s; 4,144 gradients, 40 selected means and 120 native steps all finite. Exact replay and independent contact audit passed. 26/40 solves converged. One known task with an existing command seed; neighborhood reliability remains open. |
| Runtime | The instrumented control stage took 36.69 s for 1.2 simulated seconds. Real-time execution is not qualified. |
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

## Current iteration

**Dart contact precision below 1 mm.** The user set this target after cleanup.
First attribute the saved 6.19 mm miss to the learned forecast, the controller's
deadline objective, or optimization. The [saved-evidence protocol](harness/dart-precision-v1.json)
scores all 40 plans and their actually executed 10/20/30 ms prefixes. The
[native reproduction protocol](harness/dart-precision-native-v1.json) then checks
the unchanged nominal trial through the lean public implementation. No precision
intervention has been selected or measured yet. All original attitude, speed and
deadline limits remain required alongside the stricter miss criterion.

## Following scientific gap

Freeze neighborhood reliability and prediction accuracy around
the successful Dart task, keeping the model/controller fixed and reporting every
attempted condition. Measure direct model residuals alongside control outcomes.
The narrow tangential-speed margin and fourteen budget-limited solves motivate
that evaluation. No neighborhood protocol has been run.
