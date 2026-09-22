# Backtracking the saved online directions

**Smaller steps make the longer directions useful enough to test in the online fitter.** The frozen first-acceptable rule finds an acceptable step in all 46 snapshots and beats the original four-step fitting objective in **34/46**, with 17 wins under each recipe. Twelve selected steps still lose to four. V6 remains the sole learner; no proposal was applied and no future forecast was evaluated.

The original four-step updates already passed acceptance in all 46 snapshots. This diagnostic rescues the longer directions, whose unshortened versions passed only 14/46; it does not improve an incumbent acceptance or reliability rate.

## Frozen test

The [protocol](harness/online-backtracking-v1.json) was committed in `2ef3d60` and clarified in `7377f1e`, before scientific source `41714bd` and the single completed run. It reuses all 46 authenticated directions from the [proposal audit](online-proposal-audit.md), with their conditioned parameters, retained caches, prior, gradients and forecast trust scaling fixed. No solve, derivative, reconditioning or new observation is involved. Forty-two parent references are capped at 128 iterations; they are not converged solutions.

The ladder is exactly **1, 1/2, 1/4, 1/8, 1/16** times the saved trusted direction. All five residuals are retained. Selection takes the first finite step with lower starting loss and gain at least 0.1, using the original undamped prediction. It does not choose the best ladder entry or silently fall back to four when an accepted step is worse. The comparison uses the authenticated original `production_trial` scalar; copied alpha1 residuals are byte-exact parent evidence.

## Results

| Diagnostic | V6, 23 snapshots | V7, 23 snapshots | Total |
| --- | ---: | ---: | ---: |
| Full longer step acceptable | 9 | 5 | 14 |
| First acceptable ladder step found | 23 | 23 | 46 |
| Selected loss lower than original four | 17 | 17 | 34 |
| Selected loss higher than original four | 6 | 6 | 12 |
| Selected data loss lower than starting value | 23 | 22 | 45 |
| Selected prior loss lower than starting value | 23 | 23 | 46 |

Across the entire ladder, acceptance counts are 14, 32, 44, 46 and 46 respectively. First-pass selections are 14 full steps, 18 half steps, 12 quarter steps and two eighth steps; none reaches the last rung and none uses the no-pass fallback. All 230 saved residual arrays are finite.

The rule rescues all 32 previously rejected longer directions. Relative to the previous unshortened comparison, wins against four rise from 11/46 to 34/46. Every selected prior loss is also below the instrumented four-step prior, but selected data loss beats the instrumented four in only 30/46. Four of the v7 combined-loss wins therefore trade a data-loss increase against a larger prior reduction. This is useful retained-objective evidence, not forecast improvement.

Post-measurement descriptive statistics show a median selected loss reduction against four of 1.49% for v6 and 0.080% for v7. The worst selected regressions are 27.17% and 4.55%. Absolute objectives across the recipes are not comparable because their priors differ. Complete per-point losses and gains remain in the sealed pack.

All twelve losses to the original update remain: v6 FW80 updates 27/32/35/36 and FW81 151/156; v7 FW80 32/36/37 and FW81 63/64/65. Passing the ordinary acceptance rule does not guarantee a better step than the incumbent proposal.

## Updates preceding the earlier failure states

The parent identities preserve the causal distinction: update k occurs after prediction k and produces the model used at prediction k+1. Seven of the ten linked predecessors now beat four, compared with two using the unshortened reference. Each cell below gives original-four → selected retained loss and the chosen scale.

| Update → prediction | V6 loss (scale) | V7 loss (scale) |
| --- | --- | --- |
| FW80 30 → 31 | 0.101197 → 0.078233 (1) | 0.220373 → 0.102924 (0.5) |
| FW80 35 → 36 | 0.067457 → 0.078935 (0.5) | 1.760712 → 1.751802 (0.125) |
| FW81 63 → 64 | 2.830007 → 2.774408 (1) | 14.476301 → 14.579640 (1) |
| FW81 149 → 150 | 3.588959 → 3.507229 (0.5) | 46.486726 → 46.449521 (0.25) |
| FW81 156 → 157 | 3.862741 → 3.868754 (0.5) | 49.229866 → 49.202777 (0.25) |

## Verification and scope

The independent NumPy verifier passes **6,762 point checks and 161 aggregate checks**, authenticating the full parent pack and preserving all ten causal links. A separate residual-only path recomputes all five steps at all 46 points, including the reused full step, within the frozen 1e-10 absolute plus 1e-8 relative tolerances. It calls no optimizer, derivative, conditioning, initialization or observe operation. Nonfinite handling, first-pass rather than best-loss selection, original-production comparisons, provenance tampering and exact loss thresholds have analytic regression tests.

The run performs **184 new residual evaluations**, reuses 46 parent residuals, and applies zero updates. Mandatory recomputation performs 230 residual evaluations. A hypothetical early-stopping evaluator would use 94 residual calls including full steps, averaging 2.04 per snapshot; it would still need to compute its directions. These counts do not qualify an online latency budget. The new pack is about 1.1 MB and retains the authenticated parent as a required dependency instead of copying its solver traces.

**All 334 tests and Ruff pass.** The 15 package files and all 28 files bound by the parent audit remain unchanged. No offline/Dart replay or controller trial was rerun. The [compact result index](online-backtracking.json) binds the protocol, source, both artifact authorities and complete supplementary checks.

## Next iteration

Freeze one **v6-based online candidate with 16 PCG steps and this first-acceptable backtracking ladder**. Keep the model, data objective, prior, conditioning and forecast trust rule unchanged. When no scale passes, keep ordinary rejection, rollback and damping behavior; introduce no consumer option or model catalog. Evaluate the full six tapes against v6 with forecast, angular, tail, kinematic and timing comparisons and explicit work counts.

Sixteen is a bounded engineering choice, not an optimum established by this diagnostic. This audit tested longer saved directions, not 16-step ones or an adapted online trajectory. The next experiment must establish whether the fitting-objective gain survives that budget and improves predictions. V6 stays adopted until that evidence exists.
