# Frozen features under the existing recursive objective

Completed 2026-09-22. **The catastrophic fixed-wing result is not an unavoidable
consequence of frozen features.** Keep the production learner unchanged; retain
this as a diagnostic, not an adopted alternative. The remaining speed/accuracy
tradeoff does not capture the large benefit of the fast estimator.

## Controlled comparison

Protocol `81a2d57`, scientific source/verifier `05437fc`. One run, six known
recordings, **382 updates per arm**, at most 64 per recording. Both arms initialize
from the same fresh episode prefix, with exact matching initial arrays and no
pretraining, prior flight state, fleet features or vehicle metadata.

The only intervention is restricting trainable parameters to `linear`,
`quadratic`, `bias` and `w2`. The feature, command-filter and accumulator functions
remain fixed. Measured reconditioning still rescales coordinates, compensating
feature weights exactly; this changes their numeric representation, not their
physical function. The recursive 50 ms group-Huber objective, quadratic curvature
prior, 16-PCG budget, trust radius and exact-loss backtracking are shared.

The four-command model trains 2,286 of 5,450 parameters; the three-command model
trains 1,236 of 3,103. Forecast physics and all input/history requirements remain
identical. Parameter selection is internal to the experiment; no consumer option
or vehicle-family dispatch is introduced.

## Results

Primary error is the geometric mean of velocity and body-rate RMSE ratios,
equal case weights within family then equal family weights. Lower is better.
The [index](cold-readout-rollout.json) links every physical score, early/late
result, proposal record, initialization cost and timing sample.

| Candidate / full learner | Quad | Fixed wing |
| --- | ---: | ---: |
| One-step primary error | **1.0004** | **1.2655** |
| 250 ms primary error | **0.9988** | **1.1201** |
| Warm complete-update time ratio | **0.8208** | **0.8616** |
| Candidate median update | **27.93–28.06 ms** | **6.65–6.66 ms** |
| Baseline median update | **34.07–34.22 ms** | **7.72–7.73 ms** |

Across families: one-step error **12.52% higher**, 250 ms error **5.77% higher**,
updates **15.90% faster**. Three accuracy diagnostic flags pass; the proposed
20% speed improvement flag fails. This is an engineering tradeoff, not rejection
because one threshold failed. The consistent fixed-wing loss and modest savings
make permanently freezing these functions unattractive at present.

| Case | One-step ratio | 250 ms ratio |
| --- | ---: | ---: |
| quad-arm-115 | 0.9945 | 1.0002 |
| quad-arm-125 | 1.0030 | 0.9973 |
| quad-arm-135 | 1.0012 | 1.0006 |
| quad-change | 1.0030 | 0.9973 |
| fixedwing-80 | 1.1989 | 1.0685 |
| fixedwing-81 | 1.3358 | 1.1741 |

For fixedwing-80, 250 ms velocity/rate RMSE is **3.607 m/s / 5.325 rad/s**, versus
**3.478 / 4.838** for the full learner. For fixedwing-81 it is **13.670 / 19.276**,
versus **7.066 / 27.051**. The latter trades better angular forecasts for worse
velocity forecasts; the aggregate alone hides that distinction. Absolute longer
forecasts remain weak in both arms.

## What this resolves, and what it does not

The previous raw midpoint-increment RLS screen had fixed-wing 250 ms error
**278.16×** the same baseline. Here the frozen representation produces **1.12×**.
Every baseline prediction and final model array reproduces that previous screen
exactly, so the changed conclusion is not caused by baseline drift. The large
failure is avoidable without pretraining or a fixed-wing-specific architecture.

Restoring the objective, conditioning, regularization and update safeguards as a
bundle removes the blow-up. This does **not** isolate which of them is decisive.
The raw RLS and current candidate are distinct optimizers. In this run both arms
accepted all 382 proposals, using 768 versus 767 exact objective evaluations and
6,112 PCG iterations each. Consequently, the evidence does not specifically show
that rejecting bad proposals was the cure. Regularized search directions, trust
limits and fitting actual integrated motion remain plausible explanations.

The remaining 26.5% fixed-wing one-step loss shows value in allowing these
feature/filter/memory functions to adapt under the present finite solver budget.
It does not prove a fixed feature basis cannot represent the dynamics, or tell us
which frozen component matters. Quad parity is also local to this short interval.

**Next named gap: stabilize the fast measured-increment readout through generic
curvature regularization.** Keep its cold-start feature basis and measurement
model fixed, add the existing physical quadratic-curvature penalty with explicit
objective/scale accounting, and compare to raw RLS and the full learner on the
same short causal roster. Freeze the new protocol before implementation. This
would isolate an omitted structural constraint before adding a catalog, learned
prior or more feature capacity. A recursive acceptance check can be a later,
separate experiment if regularization alone is insufficient. No pretraining.

## Scope and verification

The screen uses the existing recorded behavior policy; no new simulator or
controller trial was run. Quad scoring begins 1.25 s after release, after 75
prefix transitions including 25 actuated initialization samples. Fixed-wing
scoring begins at 0.75 s, after 15 transitions and five initialization windows.
Those observation costs are counted, not hidden as free initialization.

Sample interval and family are confounded: quads are sampled at 10 ms, fixed
wings at 50 ms. The 64-update budgets span about 0.64 s and 3.2 s respectively.
The 50 ms training horizon contains five quad samples versus one fixed-wing
sample (with two internal integration substeps). The quad-change prefix still
duplicates quad-arm-125 and does not reach its later configuration change.
These results do not establish intrinsic family-specific representational limits.

One-step predictions precede target revelation. Conditional 50/250 ms predictions
are taken every 16 updates with the actual future command tape, solely for
retrospective evaluation. There are only four such query origins per full case
and three on the truncated tape. No blind generalization, counterfactual response,
noise robustness, uncertainty coverage, live deadline or catch-rate claim follows.

Timing covers the complete observation update and synchronized model snapshot,
with alternating arm order and no competing compute job. Warm medians exclude
the first update only. First-use updates on the first quad take **2.682 s / 1.844 s**
(baseline/candidate), and on the first fixed wing **2.217 s / 1.606 s**. Shared JIT
caches mean these are not independent cold-process startup comparisons. Saved
online compute totals include initialization, one-step prediction and updates;
conditional query/scoring overhead is separate diagnostic work.

**37 tests passed**, covering the existing online machinery and three new
analytic/causal checks. A strengthened masked-JVP check also passed separately.
A test-only warning concerns float64 scalar checks outside the x64 context;
scientific updates run in x64 and inference remains ambient float32. No failed
scientific run or tuning sweep occurred. The saved-data audit authenticates tapes
and results, recomputes all metrics, checks work counts and proposal decisions,
verifies physical feature-function preservation, and verifies exact baseline
replay without fitting.

The experimental optimizer hook, runner and tests are removed from the maintained
tree after this diagnosis. Source `05437fc` preserves them; the production source
is identical to `6ac8b93`. In that historical checkout, audit with:

```bash
PYTHONPATH=src:scripts SCIPY_ARRAY_API=1 python scripts/screen_cold_readout_rollout.py verify \
  --output /Users/ryland/autonomy/glassbox/artifacts/cold-readout-rollout-v1/screen \
  --manifest-sha256 6c70804ed7dae2286560c3774eb8309db428b55cd86e712ab40ad0e0eee1ef22
```
