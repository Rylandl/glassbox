# Current state

Updated 2026-09-21. Read [the charter](charter.md) first.

The supported shared-physics learner is the **adopted single implementation**.
This follows the user's explicit decision after the successful Dart trial.
Public v4, structured model catalogs, belief objects and old experiment/control
frameworks are being removed in the current maintenance iteration. Adoption
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

## Current iteration

**Single implementation and repository cleanup.** The user has adopted the
supported learner and authorized removal of dead experiments, corresponding
artifacts, unused interfaces and tooling. The [frozen maintenance contract](harness/single-learner-adoption-v1.json)
requires identical saved model arrays, all preserved flight predictions and
Dart means/gradients; a small active test suite; an installed-package check;
and protected baseline evidence before deletion. No new long fit, controller
optimization or simulator trial is part of this iteration.

## Next scientific gap

After cleanup, freeze neighborhood reliability and prediction accuracy around
the successful Dart task, keeping the model/controller fixed and reporting every
attempted condition. Measure direct model residuals alongside control outcomes.
The narrow tangential-speed margin and fourteen budget-limited solves motivate
that evaluation. No neighborhood protocol has been run.
