# Conditioning attribution: runaway removed, response accuracy still limited

Bounding the motion features removes the observed numerical and extreme finite
runaways on all 189 previously inspected Dart queries. The benefit is already
present before any additional weight fitting. Further fitting preserves this
behavior but worsens much of the short-horizon command response. This separates
a useful representation change from an unsuccessful attempt to improve it by
long-horizon fitting; it does not establish controller readiness or fresh
generalization.

The [frozen protocol](harness/conditioning-attribution-v1.json) compares four
saved revisions: original shared-v1, its conditioned initialization, the bounded
terminal checkpoint after 1,000 attempts, and the previous unbounded refinement.
All original parameters and normalization arrays are bitwise identical between
shared-v1 and the conditioned initialization; only six motion bounds are added.
Both refinements preserve their starting normalization. No fit, gradient,
simulator rollout or controller trial occurs in this diagnostic.

The [result record](harness/conditioning-attribution-v1-result.json) anchors the
source, complete query roster, raw predictions, reductions and qualification.
All 378 historical reference predictions reproduce exactly. The 756 new
predictions replay exactly, including NaNs; an independent NumPy implementation
agrees on all 15,768 metric rows, 576 summaries, 720 paired comparisons and 756
trajectory diagnostics. Four actual evidence alterations are rejected, and all
80 preflight tests pass. There are no model-call exceptions. This qualifies
the diagnostic evidence, not a public model replacement.

## Population and scoring

All queries have been inspected previously. The old archive label `fresh` is
retained as provenance but reported here as **known-pilot**. There is no newly
held-out population in this iteration.

| Scope | Parents | Forecast queries | Response branches | Horizons |
| --- | ---: | ---: | ---: | --- |
| Known pilot | 6 | 12 | 96 | 50, 150, 250, 600, 1,200 ms |
| Known task | 1 | 1 | 8 | 50, 150, 250, 600, 1,200 ms |
| Known test | 2 | 72 | 0 | 20 queries each at 50/150/250 ms; 12 at 1,200 ms |

Each model receives the same 51 motion samples spanning 500 ms, 50 historical
issued commands and proposed future commands. Native simulator states and future truth
are used only for scoring. Inference uses the exact historical float32 JAX
boundary and the immutable original implementation for each model.

Physical errors are component RMSEs for velocity (m/s), body rate (rad/s) and
rotation-matrix entries. Prefix error uses every step through the declared
horizon and is eligible only when all those truth steps are valid; endpoint
error uses its final step. Responses compare the
signed branch-minus-factual prediction with the signed branch-minus-factual
native truth. Branches are weighted equally within an origin, origins within a
parent, and parents within a scope. Scopes are never pooled into a global score.
Every planned query remains represented. Any eligible prediction failure makes
the corresponding full-cohort RMSE undefined.

## What conditioning repairs

All observed runaways occur among the 96 known-pilot response branches at the
1.2-second horizon. The other scopes and pilot factual forecasts remain finite
and below the declared extreme-growth threshold in every arm.

| Revision | Finite and below growth threshold | Finite but extreme | Nonfinite |
| --- | ---: | ---: | ---: |
| Original shared-v1 | 83 / 96 | 4 / 96 | 9 / 96 |
| Conditioned initialization | 96 / 96 | 0 / 96 | 0 / 96 |
| Bounded terminal fit | 96 / 96 | 0 / 96 | 0 / 96 |
| Prior unbounded refinement | 84 / 96 | 4 / 96 | 8 / 96 |

Extreme growth means a finite velocity norm of at least 1,000 m/s or body-rate
norm of at least 1,000 rad/s. It is a catastrophe diagnostic, not a stability or
accuracy tolerance. Across all known queries, the conditioned initialization's
largest predicted norms are 23.665 m/s and 17.472 rad/s; the bounded terminal
model's are 23.618 m/s and 16.503 rad/s. Thus the change does more than replace
NaNs with extremely large finite numbers. These finite trajectories can still
be inaccurate.

Conditioning also improves the complete 1.2-second forecast prefixes. On the
12 known-pilot forecasts, body-rate RMSE falls from 3.320 to 1.536 rad/s
(−53.73%), while velocity falls from 1.488 to 1.431 m/s (−3.81%). On the 12
long known-test forecasts, body rate falls from 4.735 to 2.562 rad/s (−45.88%)
and velocity from 2.259 to 2.128 m/s (−5.79%). These are matched complete
cohorts, not comparisons restricted to surviving predictions.
Endpoint gains are not universal: on the single known-task forecast,
conditioning increases the 1.2-second endpoint body-rate error from 2.458 to
3.733 rad/s (+51.86%), even though its prefix error improves from 1.787 to
1.514 rad/s. Both statistics remain in the complete results.

## What it does not repair

Short-horizon response accuracy regresses. On all 96 known-pilot branches,
the 250 ms prefix measurements are:

| Revision | Velocity RMSE (m/s) | Body-rate RMSE (rad/s) | Rotation-entry RMSE |
| --- | ---: | ---: | ---: |
| Original shared-v1 | 0.01919 | 0.13633 | 0.008830 |
| Conditioned initialization | 0.01972 | 0.17321 | 0.010618 |
| Bounded terminal fit | 0.02640 | 0.20584 | 0.013106 |
| Prior unbounded refinement | 0.02139 | 0.13573 | 0.008761 |

Conditioning alone raises body-rate response RMSE by 27.06% and rotation-entry
RMSE by 20.26% at this horizon. The subsequent bounded fit adds another 18.84%
body-rate and 33.92% velocity error relative to the conditioned start. Every
known-pilot response prefix group worsens with that fit at 50, 150, 250 and
600 ms. The eight known-task responses show the same direction at those
horizons. This supports the development selector's decision to keep step 0.
The terminal fit is not uniformly worse: for example, its 1.2-second pilot
forecast body-rate prefix improves from 1.536 to 1.472 rad/s.

![Known-pilot growth and physical error comparison](../artifacts/2026-09-21/conditioning-attribution-v1-evidence/figures/conditioning-attribution-summary.png)

The conditioned model's full 1.2-second known-pilot response prefix errors are
1.633 m/s and 4.133 rad/s; endpoint errors are 3.992 m/s and 8.920 rad/s.
The terminal fit has prefix errors of 1.690 m/s and 4.290 rad/s and endpoint
errors of 3.822 m/s and 9.072 rad/s. Finite prediction is now measurable across
this cohort, but these residuals leave a substantial accuracy gap.

Original shared-v1 and its unbounded refinement have undefined full-cohort
1.2-second response RMSE because 9 and 8 branches fail, respectively. The saved
paired comparisons explicitly retain their common-finite coverage: 87/96 for
conditioned-versus-original and 88/96 for conditioned-versus-unbounded-refined.
Their very large conditional velocity/rate gains include the unbounded arms'
extreme finite survivors. They are not full-population accuracy percentages.

## Decision and remaining gap

Keep the bounded representation as a research direction: it addresses the
observed recurrence failure before fitting. Do not promote the terminal fit
or treat its 99.44% training-objective reduction as evidence of response
improvement. The earlier bounded-fit prerequisite remains failed, and its
physical/control evaluation remains unrun. Public v4 and all earlier fresh
physical and control verdicts are unchanged.

The next named gap is **preserve local command-response fidelity while bounding
recursive growth**. The leading single-change hypothesis is a smooth bounded
motion map that is exactly the identity within the recorded training envelope,
replacing the current tanh map that perturbs even supported inputs. Test its
representation effect first using saved weights, with no new fit. Derive the
envelope solely from the existing training cache, then measure both response
regressions and renewed growth on the same complete known query roster. Its
exact function, bounds and measurements must be frozen before implementation.
The 177-fold increase in training objective caused by the current transformation
motivates preserving supported features, but does not prove this will improve
response accuracy or prevent rollout excursions. Training generalization
remains a separate demonstrated limitation.
Fresh Crazyflow/Cascade accuracy, Dart controller adequacy, physical
derivatives and calibrated error envelopes remain separate obligations.
