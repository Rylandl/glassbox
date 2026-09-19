# Affine-centered quadratic initialization

This study tests whether the full quadratic learner can retain its broad
forecast and command-response gains while preserving useful affine angular
dynamics. The preceding bilinear ablation removed autonomous products and lost
substantial velocity and rotation accuracy. Its saved development checkpoints
placed much of the angular deficit in the full quadratic initialization.

The [protocol](harness/affine-anchored-quadratic-v1.json) was frozen in `9f75362`
before implementation, fitting or trials. Implementation `c6459da` adds an
isolated research candidate and evidence capture; the public learner is unchanged.

## Mechanism and comparison

The current full quadratic initializer already computes an affine precursor.
The candidate retains that precursor as the center of the existing joint ridge
penalty. For the same design `D`, normalized target `Y`, penalty matrix `P` and
stacked affine coefficients `W0`, it solves

```
(D.T @ D + P) @ W = D.T @ Y + P @ W0
```

`W0` has the existing affine coefficients, zero product coefficients and the
existing bias. The bias remains unpenalized. All joint coefficients can move;
the affine block is not fixed. The architecture, ridge strength, objective,
optimizer and selection rule remain unchanged. The prior is used only during
initialization; there is no persistent gradient regularizer, extra affine fit,
diagnostic solve or hyperparameter sweep.

The three fitted arms are public-excited (`baseline`), unchanged full
quadratic-excited (`quadratic`) and affine-anchored full quadratic (`candidate`).
Hold-current is a simple forecast/zero-response reference. Both public and both
quadratic refits reproduce their trusted saved revisions exactly.

All fitted arms share the imported 72 training and 24 development recordings
per simulator, 384/256 cached windows, base normalizations, numerical loss scale,
minibatch draws and 1,000-update budget. The two quadratic arms also share both
product normalizations and parameter counts: 15,170 in Crazyflow and 6,420 in
Cascade, versus public's 12,470 and 3,945. This matches data and updates, not
end-to-end compute or FLOPs. Candidate preparation requires the successful
current public fit; failure never causes a historical-cache fallback fit.

Checkpoint selection remains the first strict minimum of the original finite
development loss at steps 0, 100, ..., 1,000. Only the selected revision receives
held-out evaluation. The fresh 168 test-parent IDs are disjoint from six earlier
cohorts. Common calibration and imported excitation data retain their prior
identities and failures.

## Evidence and interpretation

Read-only, source-bound observation records the existing 66 development
checkpoints and each joint initialization. Candidate witnesses fingerprint the
actual design, target, penalty, system matrix, uncentered RHS, affine prior and
centered RHS immediately before the existing solve, and retain the installed
coefficients. Historical quadratic design, target and penalty are observed;
its unnamed system/RHS products are reconstructed and explicitly labeled.

Every checkpoint has physical forecast diagnostics, objective contributions and
coefficient displacement from the common affine reference. The actual applied
prior is recorded separately: zero for the historical quadratic solve, affine
for the candidate. Replay reconstructs matrix products and executes saved
predictions; it does not re-solve ridge systems or repeat optimizer updates.
Shared-initialization checks compare actual affine precursors across all
available arms. Failed prefixes and unavailable preparation remain explicit.

The frozen decisions require public progress, retention of quadratic gains and
material Crazyflow angular repair. Broad criteria permit predeclared bounded
local losses. The contextual quadratic/public result cannot veto an otherwise
successful candidate. These are experiment criteria, not application tolerances.
No result here alone establishes calibrated uncertainty, physically correct
derivatives, live updates, controller qualification or arbitrary-system transfer.

## Result

The anchored candidate is not promoted. Its weighted forecast/response ratios
against public are **0.83833 / 0.79286**, but the Crazyflow primary angular
forecast parent-error p95 ratio is **1.89018**, above the frozen 1.5 limit.
Against unchanged quadratic, weighted ratios are **1.06452 / 1.04048**; the
forecast ratio exceeds the 1.05 retained-gain limit. Targeted angular repair also
fails: primary 250 ms forecast RMSE is **0.34666 versus 0.34503 rad/s**, with
parent p95 **0.59913 versus 0.54923 rad/s**. Response mean and p95 remain within
their frozen limits. Every eligible prediction is finite.

Primary 250 ms component RMSE on the identical available-truth cohort:

| Simulator / quantity | Public | Quadratic | Anchored |
| --- | ---: | ---: | ---: |
| Crazyflow forecast velocity, m/s | 0.15231 | 0.06323 | 0.07250 |
| Crazyflow forecast body rate, rad/s | 0.19671 | 0.34503 | 0.34666 |
| Crazyflow forecast rotation entries | 0.10553 | 0.03174 | 0.03735 |
| Crazyflow response velocity, m/s | 0.11815 | 0.03615 | 0.03654 |
| Crazyflow response body rate, rad/s | 0.24610 | 0.32094 | 0.32110 |
| Crazyflow response rotation entries | 0.10436 | 0.02748 | 0.02823 |
| Cascade forecast velocity, m/s | 0.16665 | 0.13129 | 0.13473 |
| Cascade forecast body rate, rad/s | 0.10630 | 0.09646 | 0.10028 |
| Cascade forecast rotation entries | 0.01866 | 0.01165 | 0.01145 |
| Cascade response velocity, m/s | 0.04816 | 0.03743 | 0.03969 |
| Cascade response body rate, rad/s | 0.03531 | 0.03593 | 0.03709 |
| Cascade response rotation entries | 0.00691 | 0.00369 | 0.00378 |

Quadratic remains the strongest aggregate research model on this cohort:
weighted public-relative forecast/response ratios are **0.78751 / 0.76201**.
Its angular forecast and response tails still fail the existing limits. These
are matched results on this cohort, not replacements for earlier frozen scores.

Crazyflow completes **62/84** test parents, with 22 altitude failures. Primary
250 ms truth covers **448/480** factual and **736/768** response queries.
Cascade completes **84/84**, with **480/480** and **576/576** eligible queries.
Crazyflow has fewer complete parents but more eligible primary queries than the
previous cohort; parent completion alone is not a comparable accuracy denominator.
The imported training collection is unchanged, including its failed prefixes.

## Initialization worked locally; training lost much of the benefit

On development windows, anchoring reduces initial Crazyflow 250 ms angular
RMSE from quadratic's **0.31508 to 0.18853 rad/s**, and parent p95 from
**0.54078 to 0.27881 rad/s**. It also starts below public's angular RMSE of
0.21359. These are development comparisons; step zero was not evaluated on
held-out test recordings.

At the selected step 900, anchored angular RMSE rises to **0.31029 rad/s** and
p95 to **0.48520 rad/s**. Velocity improves **0.09902→0.06489 m/s** and rotation
entries **0.07234→0.03394**. The original development loss falls
**0.00931460→0.00467118**, while its angular contribution grows
**0.00052289→0.00168156**, offset by rotation's **0.00846811→0.00272587**.
All later recorded angular RMSEs exceed the anchored initialization's error.
This supports investigating how training preserves useful initial dynamics;
it does not establish whether objective weighting, generalization or parameter
regularization is the right remedy.

Cascade's anchored fit improves all three physical groups between initialization
and selected step 1,000. Its development endpoint errors are
**0.10911 m/s / 0.08879 rad/s / 0.00983 rotation entries**, versus quadratic's
**0.11405 / 0.08868 / 0.01003**. The next mechanism must remain generic and
preserve these gains.

A separate retrospective development reduction changes only the reporting
normalizer to training state standard deviation. Its minimum on the existing
Crazyflow anchored trajectory is still step 900. This does not simulate training
with that objective or select another model; a simple retrospective rescaling
does not recover the initialization behavior.

A second retrospective diagnosis executes all 66 saved checkpoints on their
exact training and development windows. Anchored Crazyflow training angular
RMSE also worsens: **0.12881→0.28250 rad/s**. Its angular normalized-loss
contribution rises **0.00045248→0.00140551**, while total training loss falls
**0.00785872→0.00372229**. Training velocity and rotation errors improve
**0.08479→0.05940 m/s** and **0.05529→0.03055**. The regression is therefore
not explained solely by overfitting: the optimizer accepts worse angular training
and development errors while reducing the combined objective. This favors an
objective-balancing experiment over merely changing the development selector.
These additional executions are training/development diagnostics, with no
intermediate held-out forecasts, fitting or reselection. Fresh development
predictions differ from captured values by at most **5.24e-14**; objective
differences are at most **5.00e-16**.

## Verification and reproduction

The [result record](harness/affine-anchored-quadratic-v1-result.json) anchors the
**14,666-payload** bundle at
`artifacts/2026-09-18/affine-anchored-quadratic-v1`, SHA256
`8ff03e5fba20761000374463fb9d1f1423c2a94861744201aa2898529a1823dc`.
All **13 production stages** complete on their first attempts without timeouts.
Both simulators' fresh replay verifies **58,812** common/imported physical-data
arrays, **4,032** selected-model query results and **342,720** metric rows.
All 66 checkpoints and actual affine-precursor comparisons replay.

All **554 focused tests** and Ruff pass. **31 alteration challenges** pass on
their first attempt: seven raw-integrity cases and 24 coherently resealed cases,
including nine initializer/coefficient challenges and the current-baseline
preparation dependency. Alterations use disposable Crazyflow copies; clean
full replay covers both simulators. No original bundle is modified.

Independent NumPy analysis verifies its declared 250 ms score/tail scope,
15,360 cached arrays, 392 imported files, 396 checkpoint diagnostic arrays,
66 coefficient decompositions and actual initialization fingerprints. Its first
two attempts stopped on auxiliary reconstruction bugs (bytecode serialization
and array layout); the corrected third attempt passes without relaxed tolerances.
The production code, sources, fits, protocol and sealed results did not change.
Failed attempts and source snapshots remain among the pinned auxiliaries.
The separate objective-rescaling diagnostic also retains its corrected
environment and reporting-label attempts; none affects experimental decisions.

Use the frozen runtime and source pins. The verified local runtime is Dart's
Python 3.12 environment with CPU JAX x64 and the pinned Cascade checkout:

```sh
SCIPY_ARRAY_API=1 JAX_ENABLE_X64=1 \
PYTHONPATH=src:/private/tmp/glassbox-cascade-e8f6ba6/src \
/Users/ryland/autonomy/dart/.venv/bin/python -m \
  glassbox.experimental.affine_anchored_experiment replay crazyflow \
  --output artifacts/2026-09-18/affine-anchored-quadratic-v1 \
  --expected-bundle-sha 8ff03e5fba20761000374463fb9d1f1423c2a94861744201aa2898529a1823dc
```

Repeat with `cascade` for its fresh replay. The saved stage supervisor and logs
record original commands, durations and bounds. Timings include concurrent,
unpaced CPU work and instrumentation; they are not real-time qualifications.

## Next hypothesis

Test fixed per-output weights derived from each channel's initial training
residual, using the same anchored model and original horizonwise normalization.
Floor the initial squared residual at `hold_scale_floor ** 2`, invert it, and
normalize weights to mean one. Keep the weights fixed for both optimization and
development selection. This tests an objective intervention; it does not isolate
the contributions of changed gradients and changed checkpoint selection.

[GradNorm](https://proceedings.mlr.press/v80/chen18a/chen18a.pdf) motivates examining
relative initial losses and normalized task weights, but its algorithm adaptively
balances gradients. The proposed fixed inverse-loss rule is a simpler heuristic,
not GradNorm or a reproduction of its static baseline. Initial prediction error
is also not the learned uncertainty used by
[Kendall et al.](https://openaccess.thecvf.com/content_cvpr_2018/html/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.html).
The multi-objective perspective of
[Sener and Koltun](https://papers.nips.cc/paper/2018/file/432aca3a1e345e339f35a30c8f65edce-Paper.pdf)
supports retaining explicit gain/loss reporting rather than assuming one weighted
sum can improve every output.

The rule uses no simulator or physical-group labels. It can overweight channels
that were initially accurate, and it still counts every represented coordinate
separately. The denominator floor is an engineering safeguard, not an application
tolerance. A fresh, predeclared experiment must test whether the changed tradeoff
improves physical forecasts and command responses without unacceptable regressions.
