# Adopting the expanded-cache public mean

This is an explicit policy decision on known evidence. The original public
qualification and the later finite-difference assessment remain failed under
their frozen criteria. They are not rewritten or described as passing.

The corrected `generic-memory-v4-prototype` mean is adopted through the existing
public API. Its unchanged numerical replay reproduced all 15,990 saved arrays
exactly, and all 89 integration regression tests pass. The version replaces the previous recipe; there is no
consumer selector. Historical artifacts continue to require their original
pinned source. The [learner guide](learner.md) describes the current contract.

## Evidence for the decision

The fresh matched physical cohort measured 52.82% lower aggregate forecast error
and 55.26% lower aggregate command-response error against saved public v3. Every
one of the twenty simulator/scope/kind aggregates improved. These gains include
different model terms, fitting mechanics and a larger training cache, so this
comparison does not isolate an architectural mechanism.

Thirteen of the 180 reported 50/150/250 ms endpoint component/scope/horizon RMSE
cells worsen in absolute terms.
Eleven are 50 ms command-response cells and two are Crazyflow wind-shift velocity
forecasts at 50 and 150 ms. For example, Crazyflow maneuver-shift 50 ms response
velocity error rises from 0.001414 to 0.001786 m/s, and wind-shift 50 ms forecast
velocity error rises from 0.006282 to 0.006668 m/s. Two worsening response cells
remain below the frozen normalization floor and consequently have floored cell
ratios of one. The absolute regressions remain visible; aggregate gains do not
mean every component or horizon improved.

Primary 250 ms RMSE, public v3 to the candidate:

| Simulator | Forecast velocity, m/s | Forecast rate, rad/s | Response velocity, m/s | Response rate, rad/s |
| --- | ---: | ---: | ---: | ---: |
| Crazyflow | 0.16468 → 0.05060 | 0.19015 → 0.17146 | 0.11889 → 0.02364 | 0.24254 → 0.15553 |
| Cascade | 0.17089 → 0.07448 | 0.10220 → 0.04757 | 0.05195 → 0.02082 | 0.03609 → 0.02013 |

All 27 synthetic absolute-capability cases pass. The runtime precision correction
preserves the physical predictions and development calibration arrays exactly in
both supported precision modes. The corrected-source replay reproduces all
428,400 physical metric rows and Dart's 168 consumer arrays, including its JVPs.
It also passes the saved synthetic and lifecycle checks. The original physical
replay's finalization schema failure is preserved; a separate saved-only
finalizer validates the original provenance fields and scientific decision.

The original qualification, precision correction and corrected-source regression
have distinct result records. The [adoption record](harness/public-mean-adoption-v1-result.json) anchors the
completed evidence-integrity and replay checks.

## Why the failed derivative gates do not veto adoption

The original eleven float32 finite-difference failures are reproducible.
The fixed-grid diagnosis attributes them primarily to coarse-step truncation.
The later assessment also fails: two of 111 required directions exceed its
adjacent-Richardson agreement limit. All 111 final Richardson estimates agree
with automatic differentiation within the original absolute/relative tolerance,
and all 111 original small-step float64 finite differences pass. All 444 required
float32 endpoint/grid checks pass. The 2,542 other required checks pass: 2,526
on the mandatory numerical cases and sixteen causality/dtype checks retained
on the diagnostic stress cases.

For the two failures, the finest derivative errors are 3.285e-8 and 5.106e-8,
using 12.3% and 8.9% of their respective allowances. Their adjacent estimates
differ by 4.983e-7 and 7.554e-7, exceeding those allowances by factors 1.86
and 1.31.

Our retrospective interpretation is that the adjacent check is conservative
here. A smooth centered difference has expansion `D(h) = A + a h² + b h⁴ + …`.
The frozen Richardson formula cancels the quadratic term, leaving a leading
fourth-order error. When halving the step, successive Richardson errors therefore
shrink by about sixteen, and their difference is about fifteen times the finest
error. The observed adjacent/final-error ratios are 15.17 and 14.79. This agrees
with the measured convergence behavior; it is not a new passing threshold.
[SciPy's differentiation documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.differentiate.derivative.html)
likewise distinguishes the adjacent-estimate error estimate from true error and
describes the expected dependence on step size and formula order. That reference
does not qualify Glassbox or establish these particular errors.

The direct derivative comparisons, observed convergence, endpoint checks and
physical prediction gains support adopting the mean despite the failed
conservative diagnostic. No step-size search, tolerance relaxation, refitting or
new simulation is used to change either failed verdict.

## Limits retained after adoption

Crazyflow completed 62 of 84 test trajectories; 22 failed the altitude condition.
Its primary 250 ms forecast/response truth covers 421/480 and 695/768 queries.
Cascade completed all 84 trajectories. Every comparator uses the same truth
masks. These results establish performance on available truth, not robustness
of the missing trajectories.

Against the separate retained balanced research baseline, the expanded-cache
mean previously worsened Crazyflow primary factual angular RMSE by 2.62%, within
the frozen 5% limit, while improving both angular tails. That localized loss
remains improvement work.

The eight added numerical inputs come from the same historical flight cohort
and parents. They are computational confirmation, not independent physical
evidence. Computational derivatives passing accuracy checks does not establish
their agreement with physical intervention responses. Calibrated uncertainty,
update improvement, controller qualification, safe revision swaps and arbitrary
system readiness remain separate open gaps.
