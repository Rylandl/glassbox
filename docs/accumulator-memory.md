# Learned accumulator memory

This iteration tests whether removing sequential nonlinear history propagation
reduces complete online-update latency while preserving the adopted generic
model's accuracy. The full explicit lag inputs remain. The previous rank-two
history compression reduced size but did not establish faster updates and lost
angular accuracy; it is not the baseline for this experiment.

The candidate replaces only the eight-dimensional nonlinear hidden recurrence:

```
drive = tanh(normalized_current_features @ projection + bias)
a = exp(-dt / learned_positive_time_constant)
hidden_next = a * hidden + (1 - a) * drive
```

Eight initial time constants span 10–500 ms in geometric order. They are learned
from observations, with the same initialization for every configuration. They
are not supplied physical actuator constants. Nonlinear feedback among hidden
coordinates is removed; nonlinear feature projections and the nonlinear
acceleration head remain. Explicit lag inputs retain sharp delayed information.

For observed history, the final hidden state is a weighted sum of the individual
drives. The implementation batches their projection and reduces them with
exponential weights, eliminating the sequential nonlinear memory scan. Forecast
memory advances once per observation interval and stays fixed through physical
substeps. Its state is reconstructed from observations whenever weights change;
there is no stale persistent latent cache. Analytic tests compare this closed
form with literal sequential dynamics and their forward/reverse derivatives.

Gravity, rigid-body mechanics, frame conventions, command filtering, all lag
features, the nonlinear head, causal caches and the full sixteen-step optimizer
are retained. Measured feature normalization compensates the smaller memory
projection algebraically. The model uses one procedure with arbitrary ordered
command dimensions and no consumer controls or platform branches.

Parameter counts are 8,714 versus 10,130 for the quad, and 3,103 versus 3,399 for
fixed-wing. The purpose is cheaper derivative execution, not the smallest model.

## Prospective decision

The [frozen protocol](harness/accumulator-memory-v1.json) compares against adopted
v8 on all six known streams and 3,137 targets, always predicting before target
assimilation. It preserves the previous primary, family, orientation, tail and
rotation/rate checks, and adds separate velocity/rate guards to prevent gains in
one from concealing losses in the other. No individual-case veto is used.

The baseline and candidate run in separate persistent processes, alternating
25-update blocks. Only one worker is granted permission to compute at a time;
initialization, compilation and untimed checkpoint work obey the same rule.
Source/runtime bindings and monotonic block intervals are saved. The baseline
must reproduce original v8 predictions, model fingerprints and reports exactly.

A meaningful speed result requires at least 25% lower quad median AND p95
complete-update latency, with fixed-wing p95 no more than 10% worse. The timer
includes public observe, immutable model construction and synchronization. The
first update in each case is reported separately; no warmed outliers are dropped.
These gates do not imply meeting the 10 ms quad observation interval.

If either accuracy or speed fails, retain adopted v8 and redirect effort toward
profiling and optimizer/runtime support. A complete pass remains an online
qualification; matched offline forecast/response and Dart checks are required
before replacing the maintained model. No architecture sweep is authorized by
this frozen iteration.

## Completed paired result

The candidate cuts quad whole-update time substantially, but does not pass all
accuracy guards. **It is not adopted.** The maintained implementation stays v8.

All 3,137 baseline predictions, post-update model fingerprints and reports exactly
reproduce the original v8 evaluation. The candidate also completes all 3,137
causal updates. Both workers execute 127 granted intervals (startup plus 126
update blocks); independent saved-data verification confirms the complete roster,
alternating order and zero overlaps. The verifier imports no JAX and makes no
model calls. All 471 tests pass.

| Case | Velocity error ratio | Rate error ratio | Median update ms: v8 → candidate | p95 update ms: v8 → candidate |
| --- | ---: | ---: | ---: | ---: |
| quad115 | 0.9790 | 0.9947 | 78.41 → 40.32 | 110.57 → 52.11 |
| quad125 | 1.0092 | 1.0074 | 75.34 → 39.64 | 81.10 → 41.61 |
| quad135 | 1.0000 | 0.9708 | 75.25 → 39.61 | 78.00 → 40.89 |
| quad-change | 1.0092 | 1.0074 | 75.19 → 39.53 | 78.39 → 40.36 |
| fixedwing80 | 0.9954 | 1.0598 | 10.06 → 8.36 | 16.38 → 18.87 |
| fixedwing81 | 0.7858 | 1.0518 | 9.79 → 8.12 | 10.15 → 8.48 |

Geometric means of per-case candidate/v8 ratios show **47.69% lower quad median**
and **49.46% lower quad p95**. Fixed-wing median falls 16.96%; its geometric p95
ratio improves only 1.86%, with an individual p95 regression on fixedwing80.
All three prospective speed gates pass. These are measured full-update speedups
on this CPU/runtime, not universal timing claims. Quad updates remain about four
times the 10 ms observation interval.

Quad combined velocity/rate error improves 0.29%, with all its reported family
metrics slightly improved. Fixed-wing combined error improves 3.37%, driven by
11.56% lower velocity error; its rate error worsens **5.58%** and rotation/rate
defect worsens **9.74%**. Across equal families: primary error improves 1.84%,
worst-decile error improves 2.74%, orientation worsens 0.42%, body-rate error
worsens **2.49%**, and rotation/rate defect worsens **4.25%**. The latter two exceed
their frozen 2% aggregate allowances. The primary, family, velocity, orientation
and tail checks pass. No threshold was changed after measurement.

The speed result supports targeting recurrent derivative execution. It does not
establish that the smaller recurrence class preserves the more expressive model
across configurations. The experiment also changes initialization and optimization
coordinates, so the fixed-wing loss is not proved to be an irreducible capacity
loss. No offline fit or Dart trial was launched after the online gate failed.

The engineering recommendation is to keep the accumulator as the leading
candidate while v8 remains the validated implementation. Nearly halving quad
update time with comparable quad accuracy and improved aggregate error warrants
further investigation despite the failed frozen screen. The next named gap is
explaining the fixed-wing angular regression: separate initialization and online
optimization effects from changed memory capacity using saved residuals and
controlled interventions. Offline forecast/response and Dart qualification remain
required before adoption. This recommendation does not change the frozen outcome.

The [result index](accumulator-memory.json) binds candidate source `808772b`,
baseline source `2f5c76b`, protocol `f61d333`, both workers and every case.
The complete paired pack is `artifacts/accumulator-memory-v1/evaluation`, authority
`1b3207154a5863b8aa5684bd587c3ebd1b819ced5df7caca8424712d5de1ac6b`.
Reproduce its verification from candidate source `808772b`:

```sh
PYTHONPATH=scripts SCIPY_ARRAY_API=1 python scripts/screen_accumulator.py verify \
  --output /Users/ryland/autonomy/glassbox/artifacts/accumulator-memory-v1/evaluation \
  --manifest-sha256 1b3207154a5863b8aa5684bd587c3ebd1b819ced5df7caca8424712d5de1ac6b
```
