# Current state

Updated 2026-09-22. Read [the charter](charter.md) first.

The **accumulator is the single maintained dynamics implementation**. It learns
one fit per configuration from observed motion and issued commands, without
vehicle-family, layout, mass or inertia inputs. The public API remains
`fit`, `predict`, immutable `update`, save/load and bounded `OnlineFit`.

| Area | Evidence and remaining gap |
| --- | --- |
| Architecture | Shared gravity, rigid-body mechanics and frames; linear, quadratic and nonlinear acceleration heads; all 100 ms lag inputs; eight stable accumulators with learned time constants. No catalog or consumer tuning option. |
| Online accuracy | Six known streams / 3,137 causal updates: equal-family velocity/rate error improves **4.11% versus v8**, rate **4.01%**, velocity **4.22%**, worst-decile **5.33%**. All frozen online accuracy/speed checks pass. |
| Runtime | Accumulator migration reduced quad median/p95 update time **47.63% / 49.76%** versus v8. The subsequent history-projection refactor reduces snapshot medians a further **19.17% for quads / 4.85% for fixed wings**: about **32 ms / 8 ms**. These are separate comparisons, not a combined trajectory benchmark. Real-time fitting remains unqualified. |
| Offline accuracy | Across the full known flight query set, forecast/response errors improve **8.29% / 9.78% versus fresh equal-budget v8**, and **6.19% / 2.23% versus deployed v8 revisions**. Cascade velocity response remains worse than the deployed revision. All frozen aggregate/derivative checks pass. |
| Dart | Same controller, task and fitting budget: accumulator **3.669 mm**, fresh v8 **3.597 mm**. Both pass attitude/speed checks with finite derivatives and miss the strict 1 mm target. Historical **0.720 mm** used a separately refined v8 revision; it is not current accumulator performance. |
| Model size | 10,130 → **8,714** parameters at four commands / 10 ms; 3,399 → **3,103** at three commands / 50 ms. Dimensions follow recordings, not vehicle type. |
| Derivatives and persistence | Migration passed all 60 finite-difference directions and exact offline replay, with 471 full-suite, five added harness and 25 installed-wheel checks. The projection refactor passes all 18 saved-model prediction/JVP/VJP comparisons and 32 focused tests. Accumulator archives remain compatible, but regrouped arithmetic changes rounding and fitted weights; strict post-update tolerances fail on some snapshots. Old v8 archives require historical source or a new fit. |
| Calibration and scope | Offline envelopes use development data that also select the fit. Independent coverage, long-horizon accuracy, broad configuration generalization and current-model closed-loop online recovery remain open. |

## Adoption decision

The user directed evaluation to support an overall engineering decision rather
than make every regression a veto. We adopted the accumulator for its much lower
online cost, slightly better online accuracy, favorable aggregate offline result
and simpler stable memory. Frozen outcomes remain unchanged. The Dart comparison
shows a 0.072 mm architecture difference under matched fresh fitting, while both
fresh fits lose the older refinement's submillimeter result. This is evidence
for a fitting-procedure gap, not proof of lost accumulator capacity.

The [migration report](accumulator-migration.md) records all comparisons and
artifact authorities. The older v8, original failed accumulator screen and
initialization diagnosis remain historical evidence, not maintained alternatives.
The correction preserves the original current-feature projection scale; changing
matrix size must not silently strengthen the initialization again.

The [history-projection refactor](history-projection-reuse.md) is also adopted.
It preserves model capacity and reduces equal-family whole-update snapshot
median cost by 12.30%. All update acceptance decisions match. The largest
post-update differences are 0.0000126 m/s and 0.0000963 rad/s; strict numerical
flags and one slower latency tail remain reported. These are acceptable tradeoffs
for the measured gain, not reasons to launch another fitting sweep.

## Next iteration

**Parallelize the observed command-filter history calculation.** Its current
sequential scan has fixed exponential coefficients during a prediction. Test a
stable parallel prefix or convolution while preserving the first-command initial
condition and derivatives through learned time constants. Freeze a focused
comparison of saved predictions, derivatives and complete update cost before
implementation. The [network review](network-review.md) explains the motivation.

Solver conditioning is the following opportunity. Reproducing the refined offline
fit, reducing fixed-wing residuals, calibration and live recovery remain separate
improvement work.
