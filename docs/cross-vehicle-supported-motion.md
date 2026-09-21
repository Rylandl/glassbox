# Cross-vehicle supported motion

The fixed support-preserving map passes its frozen **known-data preservation
hypothesis** on Crazyflow and Cascade. It preserves the original model's local
command-response accuracy while retaining the bounded representation that removed
observed Dart runaway. The flight comparison itself has no baseline runaway to
repair. This is a useful transfer result; the substantial remaining prediction
errors still need work.

| Frozen response index | Supported / original | Supported / tanh |
| --- | ---: | ---: |
| Crazyflow | 1.00000031 | 0.95848365 |
| Cascade | 0.99724019 | 1.02067591 |
| Equal-weight aggregate | 0.99861930 | 0.98909108 |

Lower is better. The aggregate response error decreases 0.14% versus original
weights and 1.09% versus the previous tanh map. Crazyflow is effectively unchanged
from original and improves 4.15% versus tanh; Cascade improves 0.28% versus
original but worsens 2.07% versus tanh. Both saved seed cohorts show the same
tradeoff. The prospective criterion allowed up to 5% aggregate loss versus
original, required improvement versus tanh, and prohibited new nonfinite or
extreme-growth trajectory identities. It passes without an every-vehicle-win
requirement. This criterion is a practical diagnostic margin, not statistical
equivalence, controller adequacy or public adoption.

The exact previously evaluated function, fitted arrays, normalizations, filters,
mechanics and time contracts are preserved. The map is identity within each
training-derived support and smoothly saturates outside it. The 1,898,496 cached
training motion coordinates are bitwise unchanged in both float32 and float64.
These overlapping training windows are not independent samples. No fitting,
gradients, simulator execution or controller trials occurred.

## Scope and scoring

The [frozen protocol](harness/cross-vehicle-supported-motion-v1.json) includes
both complete previously inspected +14M/+15M cohorts: 8,064 queries across
84 parents and 42 condition cells per simulator/cohort. Original, tanh-initial
and supported-initial revisions use the same weights. The primary response
index gives equal weight to 2 vehicles × 2 cohorts × 5 condition scopes ×
3 horizons × 3 physical groups × 2 statistics: 360 comparison cells.
Horizon values are 50, 150 and 250 ms; Crazyflow's 10 ms results are also
retained descriptively. The groups are velocity, body angular rate and rotation
matrix entries, with endpoint and prefix component errors. Branches are balanced
within origins, then parents, cells and scopes. Frozen group floors stabilize
ratios. The aggregate is a fixed diagnostic index, not an estimated population
average or a confidence interval.

All planned queries remain present. Crazyflow has 116/147 unusable histories
in the two cohorts; Cascade has none. At 250 ms, Crazyflow retains factual truth
for 762/840 and 753/840 queries and response truth for 1,224/1,344 and
1,240/1,344. Cascade retains all 840 factual and 1,008 response queries per
cohort. Missing truth is not success. Each eligible metric includes all its
truth-eligible predictions; none fail here. Input-eligible prefixes are evaluated
even beyond available truth, and padded missing-input NaNs are tracked separately.
The seven historical Crazyflow training-parent exclusions are preserved.

All three arms have zero nonfinite or ≥1,000 m/s or rad/s trajectories on their
input-eligible prefixes. That threshold is a catastrophic-growth diagnostic,
not a flight safety or application tolerance. Parent tails, all physical cells,
conditional paired comparisons and failure identities remain in the saved metrics.
The largest supported/original regression among the reported physical summary
comparisons is 0.75%, on Cascade +14M wind-response velocity prefix error at
150 ms (0.005350 versus 0.005311 m/s).

## Remaining physical errors

Representative **known +15M, 250 ms endpoint component RMSEs** for the supported
revision show why local preservation does not finish the model:

| Vehicle / condition | Forecast velocity (m/s) | Forecast rate (rad/s) | Response velocity (m/s) | Response rate (rad/s) |
| --- | ---: | ---: | ---: | ---: |
| Crazyflow primary | 0.00870 | 0.06479 | 0.01074 | 0.09133 |
| Crazyflow wind | 0.10090 | 0.04426 | 0.00754 | 0.07330 |
| Cascade primary | 0.04422 | 0.03561 | 0.01717 | 0.02031 |
| Cascade wind | 0.42624 | 0.20362 | 0.03645 | 0.03064 |

Cascade's wind forecast velocity parent p95 is 0.75106 m/s and its maximum is
0.80047 m/s. Its wind velocity forecast RMSE improves from original's
0.44097 m/s, while tanh achieves 0.40309 m/s. The map does not resolve that
condition-dependent error. The +14M cohort shows the same direction.

These flight response traces end at 250 ms. The prior known Dart 1.2-second
pilot response endpoint errors remain 3.942 m/s and 9.601 rad/s, and the
supported map's long-forecast losses versus tanh retain their original meaning.
No long-horizon flight accuracy, physical derivative fidelity, uncertainty
calibration or controller success is established here.

## Reproducibility and decision

Protocol/roster commit: `fbaf07ac8d53ee8aafeb6e7a483bad6f2cc86175`.
Complete implementation commit: `a0c5b57230b5506522e81fb3a3608674c5db4885`,
on `codex/cross-vehicle-supported-motion`. The immutable source bundle contains
that branch and all three pinned core implementations. The model function itself
is unchanged from the preceding supported-motion iteration.

All 76 bounded tests pass and Ruff is clean. There are 37,107 actual model calls
and 38,304 saved arrays per pass, including 1,197 arrays for unusable input paths.
All 12,768 historical original-model arrays reproduce exactly on **both** passes.
All 38,304 arrays, errors, metrics and provisional hypothesis replay exactly.
An independent NumPy implementation verifies the physical reduction of 514,080
score rows, hierarchy, tails, coverage, paired results, trajectory identities and
the 360-cell readout. Four distinct numeric alterations—input, model, prediction
and summary—are rejected after positive controls. Final qualification and the
qualified hypothesis are independently verified. No stage, fit or call remains
pending, and no correction or scientific restart was needed.

The [result record](harness/cross-vehicle-supported-motion-v1-result.json) anchors
source, candidates, preparation, runtime, stage receipts, qualification and the
complete evidence inventory. The public recipe remains
`generic-memory-v4-prototype`; this experiment qualifies a known-data preservation
result and changes no earlier frozen acceptance verdict.

The next named gap is **Dart planner compatibility of the fixed supported
shared-physics revision**. Freeze the exact saved revision, unchanged controller
and task, planner work budget, objective/gradient finiteness checks and existing
contact criteria before a no-fit consumer trial. Keep numerical compatibility,
task success and model accuracy as separate outcomes. This addresses the previous
concrete planning failure without repeating failed fits or tuning this map by
vehicle. The wind and long-response accuracy gaps remain explicit obligations.
