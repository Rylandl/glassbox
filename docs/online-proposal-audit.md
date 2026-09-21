# Online proposal audit: an unfinished solve and unreliable long steps

**Keep v6. Improving the local solve alone usually worsens the actual retained objective.** All 46 historical updates reconstruct exactly. The longer direction improves the damped local quadratic in all 46 snapshots, but after the existing trust rule it beats the original four-step update in only **11/46** and would be accepted in **14/46**. This supports testing nonlinear step control before another penalty change or a larger online solver budget.

## Frozen question and evidence

The [protocol](harness/online-proposal-audit-v1.json), committed in `78a62fb` and clarified in `ef583b6`, precedes implementation. Scientific source `1e60774` plus pre-measurement correction `d9cecb9` precedes the single completed audit. No learner source changes, fitted models, applied proposals or counterfactual trajectories were produced.

The cohort is all 23 previously selected fixed-wing origins under each of adopted v6 and rejected v7. Each snapshot reconstitutes the actual post-reveal bootstrap/recent cache, original conditioning, prior, damping and proposal. The separately called original implementation reproduces every saved after-model fingerprint, counter, acceptance decision and damping value; 28 adjacent pairs also match the complete after-session fingerprint. Historical scalar objectives were not logged, so their reconstructed values are not independently matched historical scalars.

For each of these 46 systems, the audit continues the same preconditioned conjugate-gradient recurrence from four to at most 128 iterations, checking the true preconditioned residual at 4/8/16/32/64/128. The residual target is 1e-6 relative to its initial norm. No restart, residual replacement, different damping or relinearization is used. Four raw/trusted and reference raw/trusted directions are evaluated under the exact retained Huber-data plus quadratic-prior objective. These are assimilated training objectives, not held-out forecast scores.

## What the measurements distinguish

| Within-update diagnostic | Adopted v6, 23 points | Rejected v7, 23 points |
| --- | ---: | ---: |
| Median relative true residual after four steps | 0.5465 | 0.5604 |
| Median relative true residual at reference stop | 0.01008 | 0.04266 |
| Median four/reference achieved damped quadratic decrease | 30.25% | 11.82% |
| Reference converged / capped | 2 / 21 | 2 / 21 |
| Original four-step trust clipped | 3 | 1 |
| Longer reference trust clipped | 13 | 15 |
| Trusted reference has lower exact loss than four | 7 | 4 |
| Trusted reference would be accepted | 9 | 5 |
| Trusted reference raises loss above the starting model | 13 | 18 |

Four steps leave substantial measured local improvement available. The longer directions improve the quadratic in all 46 cases, while the true residual improves in 45. However, **42 references reach the 128-step cap**; they are bounded probes, not converged solutions. Only the early FW80 origins 20/21 converge, at 64 steps, under each recipe. The remaining quadratic gap upper bounds can be very loose; neither the achieved decrease nor those bounds establishes nonlinear optimality.

The forecast trust rule is not simply preventing good original updates: 42/46 original directions are unscaled. Scaling improves the exact loss of 27/28 clipped reference directions, yet nine unscaled reference directions would also be rejected. The local forecast-change norm does not sufficiently constrain the nonlinear effect of a longer parameter step.

All 46 longer trusted directions descend both the data and prior terms to first order. They also reduce the exact prior in all 46, including more prior reduction than the four-step proposal in every case. But the exact data loss decreases from the starting model in only 15/46. The resulting nonlinear data increase overwhelms the prior benefit frequently: 31/46 combined losses exceed the starting value. Only 11/46 improve either data or combined loss relative to the four-step proposal. Thus a larger PCG budget alone is unsupported, even though the current local solve is visibly incomplete.

Raw and preconditioned gradient norms tell very different stories and are coordinate dependent. Their preconditioned data/prior cosines range only from -0.0451 to 0.0324, so strong opposing gradients do not explain these snapshots. The saved directional products and exact loss changes carry the useful evidence; raw gradient magnitude is not a physical measure of prior influence.

## Updates that actually produced the earlier failure states

Update k follows prediction k. The audit explicitly binds each predecessor's after-model fingerprint to the next captured prediction; it does not attribute an earlier error to a later update. The following are normalized retained-objective values within each update, not scores comparable between v6 and v7.

| Recipe | Update → prediction | Starting loss | Four trusted loss | Reference trusted loss | Reference accepted |
| --- | --- | ---: | ---: | ---: | --- |
| v6 | FW80 30 → 31 | 0.134303 | 0.101197 | 0.078233 | yes |
| v6 | FW80 35 → 36 | 0.088376 | 0.067457 | 0.368244 | no |
| v6 | FW81 63 → 64 | 3.098018 | 2.830007 | 2.774408 | yes |
| v6 | FW81 149 → 150 | 3.613499 | 3.588959 | 3.973815 | no |
| v6 | FW81 156 → 157 | 3.901467 | 3.862741 | 4.240731 | no |
| v7 | FW80 30 → 31 | 0.240256 | 0.220373 | 0.317073 | no |
| v7 | FW80 35 → 36 | 1.801262 | 1.760712 | 4.110449 | no |
| v7 | FW81 63 → 64 | 14.643392 | 14.476301 | 14.579640 | yes |
| v7 | FW81 149 → 150 | 46.517093 | 46.486726 | 46.916735 | no |
| v7 | FW81 156 → 157 | 49.234006 | 49.229866 | 49.682336 | no |

All ten anchor-producing references cap at 128 steps. Only v6 updates 30 and 63 beat the original update; none of the five v7 references does. For example, v7 FW80 update 35 predicts first-order data descent but raises exact data loss by 2.54970 while reducing prior loss by 0.24052, despite no trust clipping. These probes do not show what a counterfactual trajectory or its next prediction would do.

## Verification and computation

The sealed pack contains 419 payloads plus its manifest. The independent NumPy path authenticates the causal identities and passes **58,568 arithmetic checks**, including recurrence/guards, losses, trust scaling and normalized Krylov-span derivative identities. The Krylov identities have a maximum normwise discrepancy of 6.84e-15 against the frozen 1e-8 tolerance. No undefined linearity check was skipped on actual data. This verification path makes zero model or optimizer calls.

The separate source-bound derivative verification reconstructs all 46 caches/conditioned charts and recomputes every saved residual, gradient, prior and derivative action. It passes the original tolerances, with no PCG, observe, initialization or applied update. Analytic fixtures independently test explicit nonlinear Jacobians, dense SPD solves, trust and damping algebra, source tampering, rollback and failed-reference reporting. **All 288 tests and Ruff pass.** All 15 package files remain byte-identical to adopted v6; no offline/Dart replay or controller trial was rerun in this diagnostic.

This is read-only work, not zero computation. It reconstructs 46 original proposals (184 original CG iterations), then performs 46 diagnostic solves totaling 5,632 CG iterations, 272 additional checkpoint derivative pairs, 92 raw-probe JVPs and 184 exact trial residual calls, plus 46 linearizations/gradient VJPs. Mandatory recomputation adds 5,632 trace and 318 checkpoint derivative pairs, 184 probe JVPs and 184 trial residual calls, plus 46 linearizations/gradient VJPs and 46 conditioning calls. The run also conditions each of its 46 original reconstructions. No proposal is applied.

[The compact result index](online-proposal-audit.json) records source, authority and hashes of complete verification/results. The full per-origin arrays, gradients, losses, solver traces and work counts remain in that sealed pack.

## Next gap

**Nonlinear step control at bounded solve cost.** Test a small, frozen backtracking ladder along the already recorded longer directions, checking exact combined loss and data/prior changes against the original four-step update. This can establish whether those directions become useful when shortened before committing to an online solver change. Keep the prior, residual formulation and data fixed. Any resulting learner candidate must then pass a separately frozen six-tape evaluation against adopted v6, with all angular, tail, kinematic and timing comparisons retained.

These selected fixed-wing snapshots establish neither quad optimizer behavior nor future prediction improvement. Model identifiability, blind generalization, real-time quad fitting and online closed-loop recovery remain open.
