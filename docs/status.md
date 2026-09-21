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
| Updates | Offline updates preserve immutable revisions and development roles. The causal streaming fitter improves equal-family error 40.89% versus the previous working online fitter and 73.01% versus frozen startup on six known tapes. Every primary case/metric improves. Fixed-wing forecast spikes, one quad orientation regression and update latency remain open. |
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

## Completed online-fitting iterations

The same shared dynamics engine supports bounded causal streaming fits on
four-command quads and three-command fixed wings. After the first Adam attempt
failed, v2's full-cache Gauss-Newton improved aggregate error 54.34% over frozen
startup. The separately frozen v3 coordinate correction failed against v2:
1.41% worse overall, including a 6.51-times fixedwing-81 rate regression. Its
[failed result](online-fit-v3.json) remains recorded.

V4 adds shared physical-vector loss scales and radial Huber penalties to that
compensated coordinate conditioning. It reduces equal-family velocity/rate
error **40.89% versus working v2** (quad 24.46%, fixed wing 53.74%) and 73.01%
versus frozen startup. All twelve primary case/metric comparisons improve.
All 3,137 observations are scored before assimilation; inputs and fixed
comparators match prior versions byte-for-byte. The saved results verify with
zero fits or model calls. All 161 tests pass. Offline fitting and the saved Dart
model remain unchanged. This becomes the single maintained streaming fitter.

The secondary quad-arm-115 orientation error regresses 3.05 times to 0.004024 rad.
Fixed-wing velocity errors remain 1.77 / 16.98 times kinematic and rate errors
3.00 / 7.66 times. The harder fixed-wing tape contains a 31.14 m/s forecast-error
spike and persistent later error; no sample is excluded. Quad update p95 is
28.0–28.6 ms against 10 ms sampling; fixed-wing p95 is 3.92–3.97 ms against 50 ms.
No current-learner closed-loop trial has run. See [the report](online-fitting.md)
and [evidence identities](online-fitting.json).

## Completed consistency experiment

The separately frozen [v5 result](online-fit-v5.json) is **not adopted**. Adding a
learning-only coarse/refined integration penalty improves aggregate velocity/rate
error 4.66% against v4, missing the declared 20% target. The separate robustness
flag passes: upper-decile error improves 4.23%, orientation 1.59% and truth-relative
rotation/rate discrepancy 13.71% in the equal-family aggregate. This does not
resolve all failures: fixed-wing orientation worsens 13.13%, and quad tail error
worsens 0.41%.

Quad-arm-115 orientation improves 47.38%, but fixedwing-81 rate RMSE worsens
22.98% to 3.573 rad/s. Its maximum velocity error falls 31.14 to 16.55 m/s while
maximum rate error grows 17.19 to 33.76 rad/s. Quad update p95 grows to
43.91–45.83 ms. The modest aggregate gain and greater cost do not justify
replacing v4; no individual-cell veto was used. All 3,137 rows and unchanged
inputs, initial models and fixed comparators verify independently with zero fits
or model calls. The candidate passed 175 tests and is reproducible from source
`5db9059`; only v4 remains maintained. All 169 maintained tests pass after restoring v4
and retaining the archive verifier. The core and offline predictions are exact.

## Active scientific gap

**Causal fixed-wing forecast failures.** The consistency experiment suppresses the
inspected quiet-hover numerical artifact, but fixed-wing nonlinear stage failure
remains. Saved final-model snapshots cannot reconstruct the parameters that
produced an earlier causal spike; a milder local Jacobian also does not guarantee
accurate integration along the whole step. Freeze a diagnostic replay that saves
the actual pre-prediction models at the declared v4 velocity/rate spikes and
nearby ordinary origins. Inspect each integration stage, motion-support
compression and available command excitation before selecting another correction.
Do not tune a stronger penalty from these aggregate scores. Latency and live
control remain separate qualifications. No successor fit has run.

The [causal trace protocol](harness/online-causal-trace-v1.json) is frozen before
implementation:450 exact paired replay transitions and20 declared snapshots.
No replay has started. The working v4 learner remains unchanged.
