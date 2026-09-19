# Expanded training cache: substantially lower physical residuals

The [frozen experiment](harness/expanded-training-cache-v1.json) increases the
training cache from 384 to 1,536 windows drawn from the same admitted recordings.
On the fresh common cohort, forecast and command-response errors fall
**38.04% and 32.73% against the retained balanced model**. All three frozen
physical comparisons and Crazyflow angular-retention checks pass.

This experiment retained the expanded-cache candidate as the strongest verified
research mean; integrity, replay and alteration checks pass. Public adoption
subsequently followed a separate [qualification and policy decision](public-mean-adoption.md).
That decision does not change this experiment's results or original scope.

## What changed

The candidate uses the same 72 training parents per simulator, preserves the
exact original 384-window prefix, and appends 1,152 distinct eligible origins.
The 24 development parents and all 256 development windows remain exact.
All 1,536 windows are reconstructed from authenticated recordings; both
training partitions remain training data. There are no new independent
recordings, platform-specific learner rules, seeds selected by results, or
intermediate held-out checkpoint predictions.

Architecture, initialization and objective formulas, the full-cache Adam
optimizer, strict backtracking acceptance and the development selector stay
fixed. Training-derived normalization, ridge systems and coefficients, hold
scales and channel weights naturally change. Raw weighted losses across these
arms are therefore not directly comparable. The five data-independent initial
parameter arrays are externally anchored to the saved 384-window control;
complete initial-parameter identity is deliberately not claimed.

Only the two candidates are newly fitted. Public-excited, balanced and
fullcache384 models are imported byte-for-byte under their historical
provenance. Hold predictions provide simple physical context. Every predictor
uses the same current test queries and truth masks.

## Matched physical accuracy

Ratios below one favor the candidate. These are the prospectively weighted
geometric means across the declared simulators, scopes, horizons and groups.
They are not percentages of application adequacy.

| Reference | Forecast ratio | Response ratio |
| --- | ---: | ---: |
| Public-excited | 0.47308 | 0.45947 |
| Retained balanced | 0.61964 | 0.67268 |
| Full-cache 384-window control | 0.56914 | 0.59144 |

The frozen paired-parent bootstrap gives these descriptive 95% percentile
intervals. They summarize variation within this cohort; they do not establish
coverage across untested systems or replace the declared promotion criteria.

| Reference | Forecast ratio interval | Response ratio interval |
| --- | ---: | ---: |
| Public-excited | [0.46061, 0.48448] | [0.44835, 0.46939] |
| Retained balanced | [0.60504, 0.63522] | [0.65863, 0.68507] |
| Full-cache 384-window control | [0.55391, 0.58916] | [0.57766, 0.60746] |

Primary 250 ms component RMSE, balanced → expanded cache:

| Simulator / target | Velocity, m/s | Body rate, rad/s | Rotation entries, unitless |
| --- | ---: | ---: | ---: |
| Crazyflow forecast | 0.11621 → 0.04926 | 0.15653 → 0.16063 | 0.07356 → 0.03036 |
| Crazyflow response | 0.04010 → 0.02335 | 0.16675 → 0.15673 | 0.04023 → 0.02123 |
| Cascade forecast | 0.13096 → 0.06895 | 0.10421 → 0.04942 | 0.01203 → 0.00598 |
| Cascade response | 0.03669 → 0.02039 | 0.04090 → 0.01865 | 0.00384 → 0.00177 |

Crazyflow factual rate remains a concrete exception: mean error rises **2.62%**.
Its parent-p95 falls **0.27004→0.24830 rad/s**, a **0.919487** ratio. Response
rate mean and p95 ratios are **0.939913** and **0.945371**, respectively;
response p95 falls **0.25606→0.24207 rad/s**. All four angular checks pass the
prospective 1.05 retention limit. Passing permits this measured loss; it does
not imply improvement in every channel or condition.

![Matched physical errors](../artifacts/2026-09-19/expanded-training-cache-v1-figures-attempt2/heldout-250ms.png)

The frozen policy requires response ratios ≤0.90 against public and ≤0.95
against balanced and fullcache384, with forecast ratios ≤1.05 for each pair.
All pairs also retain the 1.15 per-simulator primary limits, 1.50 scope limits,
1.50 primary 250 ms parent-tail limits, finite predictions and matched cohorts.
The separate optimization diagnostic passes but cannot veto physical progress.
No threshold was changed after observing these results.

Crazyflow completes **66/84** test parents; 18 fail the altitude condition.
Primary 250 ms truth is **438/480** forecast and **720/768** response queries.
Cascade completes **84/84**, with all **480/480** forecast and **576/576**
response queries. All five predictors are finite on eligible truth. Crazyflow
numbers remain conditional on available truth. Its improved completion count
relative to earlier cohorts is a fresh-cohort difference, not a learner gain.

## Training diagnosis and compute

Both candidates select checkpoint 1,000. Crazyflow training/development losses
fall **84.91%/46.65%** from its own initialization; Cascade falls
**90.76%/81.12%**. These within-fit ratios use each candidate's own fixed
numerical objective.

A remaining Crazyflow generalization gap is visible: training rate RMSE falls
**0.14523→0.07347 rad/s**, while development rate rises
**0.13647→0.16375 rad/s**. Development velocity and rotation improve enough
to offset that angular loss in the fixed selector. Prefix and added training
windows both improve; their diagnostics reuse saved full-cache forecasts and
never select a checkpoint. This evidence does not uniquely identify inadequate
independent recording diversity, objective tradeoffs or model bias as the cause.

Each fit completes 1,000 gradients and **1,536,000 gradient-window visits**, four
times the imported fullcache384 control. Crazyflow accepts 984 proposals and
rejects 16; Cascade accepts 997 and rejects 3. Acceptance-objective calls are
**4,911/2,464**, each over all 1,536 windows. Fitter/capture times are approximately
**342.19/42.45 seconds**, with preparation, calibration and checkpoint diagnostics
recorded separately. Fitter-only peak memory was not measured; supervisors record child-process
high-water RSS separately. Equal proposal counts do not
mean equal compute; fourfold window work does not imply fourfold elapsed time.

![Training and development progress](../artifacts/2026-09-19/expanded-training-cache-v1-figures-attempt2/training-diagnosis.png)

## Verification and preserved failure

Protocol `23d4b07` was committed before implementation `f14cfad`. The first
attempt completed generation, fitting and evaluation, then failed finalization:
the resolved comparison helper received outcomes prose where it required a
shared-guard dictionary. That unfinalized attempt and its logs are preserved.
Correction `8b61830` restores the frozen guard dictionary and adds a regression
test exercising actual resolved-policy comparison dispatch. The protocol,
numerical learner, seeds, selection and thresholds remain unchanged. The
corrected harness reruns the frozen production stages rather than treating the
failed first attempt as a finalized result.

The corrected sealed root has **14,782 payloads**, SHA-256
`88b59d60be28ffed1e93fa6e6ecaa026b1d49950a23f7f2b1e0f9b65b3b9d99d`.

Verification is complete: **1,136 focused tests across 41 files**, both
canonical simulator replays, independent physical/preparation/initialization/
optimizer audits, and all **37 alteration challenges** pass. Replay reproduces
**4,032 prediction queries**, **428,400 metric rows**, 88 checkpoint snapshots
and **7,375 candidate acceptance-objective calls** exactly. The independent
physical audit reduces all 1,050 endpoint groups. The optimizer audit checks
2,000 gradient attempts, 3,072,000 gradient-window visits, 44 candidate full
training/development diagnostic sets and 66 saved partition reductions.

The preserved first attempt and corrected run agree exactly on 103,302 numerical
arrays and 2,979 scientific JSON files, including selected parameters, norms,
checkpoints, truth, queries, predictions and metrics. Source/runtime provenance
and truthful timing differ and are explicitly excluded from whole-artifact
equality. The failed attempt has no finalized scientific decision.

The physical audit's first invocation supplied the resolved protocol where the
frozen delta was required. Its script and scientific inputs stayed unchanged;
the corrected invocation passes. Both audit attempts and both figure renders
are preserved. The final figures were visually reviewed.

The alteration audit passed 23 cases before a stale expected error message
stopped it: the current verifier correctly rejected the altered training cache.
An auxiliary assertion correction then ran the remaining 14 cases. The composite
report anchors all 37 successful checks across the two attempts and preserves
the failed assertion; no production verifier or scientific artifact changed.

The [result record](harness/expanded-training-cache-v1-result.json) pins the
bundle, scripts, logs, reports and preserved-attempt manifest. No fitting recipe,
seed, selection rule or acceptance threshold was changed after observation.

Replay qualification must retain its scope: fresh predictions and saved-proposal
acceptance checks without refitting, re-solving initialization or independently
recomputing gradients/Adam. Observed gradient and moment fingerprints are
execution and continuity evidence, not an independent derivative calculation.

## Next priority

This result makes a usable best mean a concrete project priority. The recommended
next iteration is **public integration qualification**: freeze the existing
synthetic absolute caps and public fit/predict/update, save/load, timing,
JAX/differentiation and replay checks for one candidate recipe. Successful
research comparisons do not by themselves authorize a public accuracy,
uncertainty, derivative-fidelity, update or controller claim.

The integration must resolve concrete interface differences: the research
float64 assumption, public loading and rollout paths tied to the old model,
recording-ledger and update semantics, and dependence on a fitted public
precursor. Freeze precision, recipe and promotion rules before that work; keep
one consumer API without adding platform or model-selection options. Adoption
of a qualified mean remains separate from error-coverage, live-update and
controller qualification.

The next accuracy hypothesis remains independent recording diversity: add one
predeclared excited cohort across the existing primary conditions while retaining
a fixed 1,536-window fit budget and unchanged development cache. This would test
the remaining parent-to-parent angular gap rather than add more overlapping
windows from the same 72 parents. It is a proposal for a separate frozen study,
not an adopted mechanism or a reason to postpone making the current mean usable.
Public integration qualification is the next named iteration; independent
recording diversity remains the subsequent accuracy hypothesis.

## Reproduction

Use the bundle's pinned Dart Python environment, CPU x64 JAX runtime and Cascade
checkout. From the repository, run each simulator's replay against the external
bundle anchor:

```sh
SCIPY_ARRAY_API=1 JAX_ENABLE_X64=1 \
PYTHONPATH=src:/private/tmp/glassbox-cascade-e8f6ba6/src \
/Users/ryland/autonomy/dart/.venv/bin/python \
  -m glassbox.experimental.expanded_training_cache_experiment replay crazyflow \
  --output artifacts/2026-09-19/expanded-training-cache-v1 \
  --expected-bundle-sha 88b59d60be28ffed1e93fa6e6ecaa026b1d49950a23f7f2b1e0f9b65b3b9d99d
```

Replace `crazyflow` with `cascade` for the second simulator. Historical evidence
requires its recorded sources and runtime. No public, synthetic, uncertainty,
physical-derivative, live-update or controller qualification is implied.
