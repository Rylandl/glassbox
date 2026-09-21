# Online fitting: physical curvature prior

**Online v6 is the maintained streaming fitter.** Across all 3,137 causal forecasts
on six known tapes, velocity/rate error is **21.94% lower than working v4**:
13.69% for quads and 29.41% for fixed wings. The frozen primary gate passes
(aggregate ratio 0.78057, target ≤0.8; both families improve). Ten of twelve
primary case/metric cells improve; no case is excluded or given a veto.
Error is 78.93% lower than keeping the identical startup fit frozen.

## What changed

The [saved-state diagnosis](online-response-support.md) found excessive learned
command curvature and context-dependent amplification during integration. The
retained origin features alone do not distinguish affine and quadratic heads.
V6 adds a fixed quadratic-head curvature prior, scaled with measured physical
motion, issued commands and known unit-gravity geometry. The fixed strength 0.01
was declared before fitting and was not swept. It is an engineering choice,
not calibrated uncertainty or an optimum established by theory.

The objective retains v4's equal-role, equal-time, radial physical-vector Huber
forecast loss. The prior adds its exact gradient and diagonal curvature to four
preconditioned conjugate-gradient iterations. The forecast-change trust bound
stays unchanged; acceptance checks actual combined loss against predicted
improvement. Existing compensated normalization preserves the prior's physical
value. Complete optimizer/frame invariance is not claimed.

All parameters remain trainable. The prior covers the explicit quadratic head,
not every nonlinearity in delayed, supported-motion or recurrent dynamics.
It changes online fitting only: initialization, shared dynamics, deployed
integration, immutable model format and offline/Dart models stay unchanged.
There is no vehicle branch, extra user option or system-specific parameter.
The mutable optimizer-session format is now `glassbox-online-fit-v6`.

## Physical errors and remaining regressions

The primary reference is authenticated **working v4**, not the weaker frozen
startup fit. Each cell is **v6 / v4 / no-fit kinematic**. The kinematic predictor
holds world velocity and body rate and integrates orientation. Quads are scored
at 10 ms; fixed wings at 50 ms, before target assimilation.

| Case | Velocity RMSE, m/s | Body-rate RMSE, rad/s | Forecasts |
| --- | ---: | ---: | ---: |
| quad-arm-115 | 0.02375 / 0.02208 / 0.03967 | 0.23295 / 0.27256 / 0.11166 | 875 |
| quad-arm-125 | 0.01025 / 0.01220 / 0.02942 | 0.08107 / 0.10730 / 0.12058 | 875 |
| quad-arm-135 | 1.17741 / 1.19929 / 1.17389 | 0.91260 / 1.07632 / 1.12088 | 62 |
| quad-change | 0.01025 / 0.01220 / 0.02942 | 0.08107 / 0.10730 / 0.12058 | 875 |
| fixedwing-80 | 0.30899 / 0.29907 / 0.16863 | 0.78329 / 1.13019 / 0.37618 | 225 |
| fixedwing-81 | 1.24428 / 2.89098 / 0.17023 | 2.34088 / 2.90528 / 0.37920 | 225 |

The geometric aggregate weights each family equally and each case equally within
its family. The four quad scenarios use one Crazyflow `cf21B_500` with arm/inertia
changes; two known Cascade recordings use one Skywalker-X8. The 62-row quad tape
remains truncated by the original behavior controller's floor contact. This is
known-tape improvement, not independent generalization or candidate-controlled
recovery. The old Throw identifier saw applied rotor telemetry and is not a
matched-input comparator.

Velocity RMSE worsens 7.57% on quad115 and 3.32% on FW80. FW81 orientation RMSE
rises 7.16%, from 0.10164 to 0.10892 rad. Fixed-wing orientation and truth-relative
rotation/rate-defect family scores worsen 2.13% and 6.49% respectively. These
regressions remain part of the passing aggregate result. Worst-decile error also
worsens for quad115 velocity (+10.78%), quad135 orientation (+11.23%), FW80
velocity (+1.89%) and FW81 orientation (+27.52%). FW80's rotation/rate defect
rises 13.94%. Maximum-error regressions include quad115 velocity/rate, quad125
and quad-change orientation slightly, quad135 orientation, FW80
velocity/orientation and FW81 rate/orientation. All quantiles and maxima remain
in the sealed per-case summaries and independent audit.

The separate robustness gate passes: worst-decile velocity/rate error falls
22.96%, orientation RMSE 16.75%, and truth-relative rotation/rate defect 16.89%
under equal-family aggregation. That gate permits family tradeoffs; it does
not mean every angular metric improved.

| Fixed-wing extreme | v4 | v6 |
| --- | ---: | ---: |
| FW81 maximum velocity error, m/s | 31.139 | 7.704 |
| FW81 maximum rate error, rad/s | 17.194 | 18.227 |
| FW81 maximum orientation error, rad | 0.481 | 0.601 |
| FW80 maximum velocity error, m/s | 1.088 | 1.297 |
| FW80 maximum rate error, rad/s | 8.272 | 5.136 |

FW81 velocity errors are less concentrated: the largest five samples account
for 57.59% of squared error, down from 77.82%. Large angular failures remain.
Absolute fixed-wing errors also remain worse than the kinematic predictor:
velocity 1.83 / 7.31 times and rate 2.08 / 6.17 times on FW80/FW81. An improved
comparison to the preceding learner does not establish adequate short-horizon
accuracy or useful control derivatives.

## Timing, objective diagnostics and verification

Quad update p95 is 28.16–29.53 ms against 10 ms observations; fixed-wing update
p95 is 4.38–4.60 ms against 50 ms. Warmed prediction p95 is 0.89–1.07 ms. The first
compiled quad/fixed-wing updates take 2.50 / 2.15 s. These timings exclude durable
journal overhead. The quad real-time qualification still fails.

Initial and final cache forecasts are saved after measured case timing. Data,
prior and combined loss are recomputed independently from those predictions,
measured caches and physical coefficients, without fitting or model calls.
Combined improvement may trade data fit for lower curvature. Endpoint caches
and domain sizes differ, so their totals are not a monotone optimization trace.
These diagnostics add no persistent optimizer arrays or public report fields.
A separate descriptive final-cache comparison checks v4/v6 against identical
measured caches, raw normalizers and loss scales: the physical curvature-prior
value falls 97.4–99.0% on the quads and by more than 99.99999% on both fixed-wing
tapes. That verifies strong explicit-curvature suppression; it neither proves
which part caused the accuracy gain nor makes the remaining dynamics correct.

All **204 tests pass**. Independent arithmetic tests cover physical-domain
weighting, Hessian factors, combined objective, truncated four-step solve,
compensation, causality, rejection and save/resume. The built wheel completes
predict/observe/save/load outside the checkout. All 12,768 saved flight arrays,
40 Dart trajectories and 4,144 gradients reproduce exactly. Every package source
file except `online.py` is byte-identical to working-v4 source.

An independent NumPy audit checks 242 payloads and all 3,137 causal rows.
The saved-data verifier authenticates all six cases and nested v4/v2 references,
recomputes every metric and gate with zero fits/model calls, and checks exact
input, initialization, fixed-comparator and causal-journal pairing. The
[frozen v6 protocol](harness/online-fit-v6.json) and
[result index](online-fitting.json) bind source, runtime and artifact authority.
See [replay instructions](../CONTRIBUTING.md#reproduce-streaming-fitting).

## Next gap

**Residual fixed-wing angular response and forecast outliers.** Capture v6's
actual pre-assimilation state at its remaining angular extremes, distinguish
incorrect learned acceleration from integration error, and test whether
unconstrained delayed-linear/recurrent response now limits accuracy. Preserve
velocity/rate, orientation, tails, kinematic comparisons and timing. No stronger
prior, extra head or finer solver is justified by the aggregate alone.

Historical verdicts remain unchanged: [v4](online-fit-v4.json) was adopted;
[v3](online-fit-v3.json) and [v5](online-fit-v5.json) were not. Their source and
sealed evidence remain reproducible. Only v6 is maintained as the online fitter.
