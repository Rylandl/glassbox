# Nonlinear temporal compression

Adopted 2026-09-22. One generic model remains; no architecture selector or
vehicle-family branch was added. This is a compactness and accuracy tradeoff,
not a CPU speed improvement.

## What changed

Only the nonlinear acceleration head compresses the previous 100 ms of feature
differences. More than four samples use four fixed orthonormal discrete
polynomial components (constant through cubic); four or fewer retain their
original coordinates. The full linear lag head, quadratic current-feature head,
32 nonlinear units, eight stable accumulators and shared mechanics remain.
Command count and sample interval determine dimensions.

At four commands and ten lags the nonlinear input width falls from 195 to 93:
6,240 → 2,976 input weights and **8,714 → 5,450 total parameters**. The
three-command, two-lag model remains 3,103 parameters. It is an identity control,
not evidence for a compressed fixed wing sampled at 10 ms. The fixed basis
removes nonlinear access to discarded temporal components; arbitrary linear
lag response remains representable.

Compact features have their own normalization and compensated online scaling.
Initialization projects the original physical lag coefficients into the fixed
basis while preserving current/memory coefficients, random draw scale, the
linear/quadratic ridge solution and zero nonlinear output weights. Initial
physical predictions and latent activation statistics were checked. The
runtime keeps compact coefficients rather than expanding them back to a dense
head. No simulator information enters the model.

## Measurements and decision

The online protocol was committed before implementation; the matched offline
protocol was committed before all six fits. The comparator is the current
full-history accumulator **including the adopted projection reuse**, not v8 or
the original accumulator initialization. One fixed seed and fitting recipe;
no outcome-driven tuning, extra fits or controller trials.

| Measure | Compact / current full model | Interpretation |
| --- | ---: | --- |
| Parameters, 4 commands / 10 ms | 0.6254 | 37.46% fewer |
| Online velocity/rate error, equal-family | 0.99991 | Essentially equal |
| Quad error after first 100 updates | 1.0280 | 2.80% worse; short 62-row tape unavailable here |
| Quad median whole update | 1.0610 | 6.10% slower, 31.9 → 33.9 ms |
| Quad p95 whole update | 1.0639 | 6.39% slower |
| Offline forecast error, equal-family | 1.0208 | 2.08% worse |
| Offline command-response error, equal-family | 0.9782 | 2.18% better |
| Crazyflow forecast / response error | 1.0420 / 0.9569 | 4.20% worse / 4.31% better |
| Cascade forecasts and responses | 1.0000 | Exactly equal |
| Dart 250 ms velocity/rate error | 0.8501 | 14.99% better |

Primary error is the geometric mean of physical velocity-vector and body-rate-vector
RMSE ratios, not a percentage of successful cases. Online cases have equal
weight within family, then families have equal weight. Offline groups have equal
scope/cohort/horizon weights within family and forecast/response kind, then equal
family weight. Absolute per-group scores and all tails remain in the saved audit.
There are no confidence intervals or multiple-seed robustness claims.

All frozen online accuracy/size flags pass; the **5% quad speed-improvement target
fails**. Every frozen offline check passes, including numerical derivatives.
We accept the forecast/runtime cost for the substantial size reduction and
response/Dart improvements. This does not show a generally faster learner: both
online arms use one proposal and 16 CG iterations per observation, with identical
acceptance/objective-call counts. Online learning after 100 updates is worse.

Offline both arms used 1,536 training and 256 development windows, 250 ms fitting
horizons, 1,000 full-cache safeguarded-Adam attempts and checkpoint selection every
100 attempts. Every fit selected step 1,000. Final weighted development losses
are 12.42% lower on Dart, 13.09% higher on Crazyflow, and identical on Cascade.
Full objective calls: Dart 1,650 → 1,737; Crazyflow 1,038 → 1,006; Cascade 1,188
unchanged. Training progress is retained; no uniform work-to-accuracy gain exists.
Concurrent offline fit times are not speed evidence. Only the exclusive,
alternating 25-update online blocks support the latency comparison (127 per arm,
no overlap).

The full known flight roster includes 8,064 factual/counterfactual queries and
140 scored groups per arm. Command-response error compares predicted changes
under actual command perturbations with truth changes, separately from numerical
Jacobian checks. All 40 seeded finite-difference directions pass. Eligibility,
command prefixes, missing values, query identities and score reconstruction are
independently checked from sealed arrays.

## Dart forecasts, not a new controller result

Twelve origins in the two known held-out Dart recordings were forecast with
identical issued commands. These were not new blind recordings.

| Horizon | Full velocity RMSE (m/s) | Compact velocity | Full rate RMSE (rad/s) | Compact rate |
| --- | ---: | ---: | ---: | ---: |
| 10 ms | 0.002689 | 0.002474 | 0.008295 | 0.007717 |
| 250 ms | 0.11244 | 0.09599 | 1.36660 | 1.15676 |
| 600 ms | 2.56140 | 2.22705 | 6.60263 | 5.65753 |
| 1,200 ms | 8.78462 | 8.02639 | 14.64288 | 13.23357 |

Orientation RMSE also improves at every horizon. Long-horizon errors are still
large and exceed the fitting/calibration horizon. These gains do not establish
submillimeter contact, successful live online recovery or arbitrary-system
accuracy. Earlier 3.669 mm accumulator and 0.720 mm refined-v8 controller trials
belong to different saved revisions.

## A correction exposed by the fresh baseline

A supplemental saved-data comparison of the new full-model baseline with the
pre-projection accumulator gives **2.69% higher primary online error** and
**4.70% higher rate error**. Fixedwing-81 contributes most: primary +12.25%, rate
+15.12% (1.4603 → 1.6810 rad/s). This is separate from temporal compression,
whose fixed-wing predictions match the new baseline exactly.

The projection refactor preserved equations and passed saved-model comparisons,
but its snapshot tests did not establish equivalence over repeated fitting.
The measured divergence is consistent with small rounding changes propagating
through the optimizer; this comparison does not isolate every numerical cause.
Keep its earlier speed benefit and frozen failures visible rather than rewriting
history or automatically reverting it on one cell. Current compact / original
v8 ratios are primary 0.98460, velocity 0.96531, rate 1.00427, tail 0.96412.
The old 4.11% gain is historical, not current-model accuracy.

## Persistence and verification

The public recipe is `shared-vehicle-temporal-v1`; model/session formats change.
Use historical source for older full-history archives; do not relabel them.
The experiment's compact public archive metadata initially retained the old
recipe label while its core format and optimization report were already temporal.
A committed one-time adoption script corrects only that metadata in separately
sealed revisions. Every numeric array, retained window and envelope is exactly
unchanged. Both original and finalized revisions exactly replay all 17 held-out
prediction/availability arrays with zero fitting or optimizer calls.

The [index](nonlinear-temporal.json) records independent authorities, source
commits, finalized model paths, full outcomes and test logs. Original experiment
packs are unchanged. The protocol has inherited wording mentioning “fresh v8”
in its data paragraph; its explicit two-arm roster and package hashes control
this comparison, which actually uses fresh full-history and compact models.

The candidate full suite initially reported 483 passes and five stale-fixture
failures (missing compact normalization and an old archive-format expectation).
All 19 tests in those modules pass after fixture correction: **488 distinct
suite cases pass**. The public recipe consistency assertion and six learner
lifecycle tests pass after the metadata correction. **25 installed-wheel checks**
pass outside the source tree. Analytic coverage includes basis span, compact vs
independently expanded projections, JVP/VJP, recursive compensation, mechanics,
causality and session persistence.

Two reporting defects were corrected without repeating scientific runs: an
empty after-100 bin on the short tape, and the original v8 row field being named
`origin`. Original failure logs are retained. The first installed-wheel attempt
was stopped after a sandboxed installer failure; the successful offline install
and 25 checks are recorded separately.

Saved-score verification does not fit or predict:

```bash
PYTHONPATH=src:scripts SCIPY_ARRAY_API=1 python scripts/screen_temporal.py verify \
  --output /Users/ryland/autonomy/glassbox/artifacts/nonlinear-temporal-v1/online \
  --manifest-sha256 979225f0739e606a2aebbc3f5e521b0cd5378335bbab990bd9465344f4c75dba
PYTHONPATH=src:scripts SCIPY_ARRAY_API=1 python scripts/qualify_temporal.py verify \
  --index docs/nonlinear-temporal.json
```

For original experimental bitwise replay, use source `0827275` and its recorded
CPU/arm64 runtime with the artifact `replay-index.json`. Finalized revisions load
with current source; their numerical arrays are identical. Other runtimes are
not promised bitwise equivalence. Historical source is not a supported alternative
architecture inside the package.
