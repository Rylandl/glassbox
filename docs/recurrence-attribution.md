# Recurrence attribution: smaller steps do not cure the runaway

The frozen no-fit diagnostic is complete. Both saved shared-physics models
exhibit runaway state feedback on the selected Dart response branches. Smaller
mechanical integration steps make those failures earlier and introduce failures
on matched factual controls. This identifies a concrete architecture problem to
investigate before another long fit. It does not establish continuous-time
blowup, physical derivative accuracy or a successful model intervention.

The [protocol](harness/recurrence-attribution-v1.json) was committed before
implementation. Its [roster](harness/recurrence-attribution-v1-roster.json)
contains the union of nine previously failing candidate/shared response queries,
six deduplicated factual controls and both saved models. These are selected
diagnostic cases, not an estimate of failure rates over all flight conditions.
The [result record](harness/recurrence-attribution-v1-result.json) anchors the
source, inputs, raw traces, independent reductions, replay and alteration tests.

## What was compared

Each of the 15 queries was evaluated with both models and 1/2/4 mechanical
substeps: 90 authoritative predictions and 90 local traces, followed by one
exact replay. All arithmetic used the historical float32 application runtime.
The original one-step predictor was unchanged. Commands, memory and history
retained their original 10 ms timing; the actuator filter followed its analytic
exponential within each interval. The other subdivisions changed mechanical
integration resolution only.

Trace instrumentation can alter floating-point rounding. Each local trace was
therefore forced through the authoritative observation-boundary states and
qualified against its next states using a tolerance frozen before execution.
All 30 historical predictions reproduced exactly, all 90 local traces qualified,
and every raw trace array matched its replay. There were no fits, gradients,
simulator rollouts or controller calls.

## Results

Nonfinite predictions within the 1.2 s horizon:

| Saved model and query group | 1 substep | 2 substeps | 4 substeps |
| --- | ---: | ---: | ---: |
| Refined candidate, response | 8/9 | 9/9 | 9/9 |
| Shared v1, response | 9/9 | 9/9 | 9/9 |
| Refined candidate, factual | 0/6 | 0/6 | 2/6 |
| Shared v1, factual | 0/6 | 1/6 | 2/6 |

All 54 response trajectories cross both 1,000 m/s and 1,000 rad/s, including
the candidate's single finite original response. The thresholds describe
extreme growth; they are not application tolerances. Candidate response failures
move from 0.92–1.16 s with one substep to 0.78–0.92 s with four. Shared-v1
failures move from 0.91–1.19 s to 0.77–0.91 s. The newly failing factual controls
are fresh-00 and fresh-04, both at the 1.5 s origin.

All predictions remain finite through 0.60 s. Across all 30 query/model pairs
and all three physical groups, the RMS discrepancy between two and four
substeps is smaller than between one and two at 0.05, 0.15, 0.25 and 0.60 s.
There is numerical convergence evidence over these finite prefixes, alongside
earlier catastrophic growth later. This does not qualify any subdivision as a
better model, and three resolutions cannot establish the continuous-time limit.

![Saved branch growth with 1, 2 and 4 integration substeps](../artifacts/2026-09-21/recurrence-attribution-v1-evidence/recurrence-growth.png)

In every one of the 58 nonfinite trajectories, the first recorded nonfinite
quantity is a quadratic feature. Rotation-exponential overflow is not the first
recorded event. However, the quadratic term is not the whole explanation:

- In the interval first crossing 10 rad/s, the observation-start head's linear
  term supplies a median 93.5% of the candidate's signed angular-speed growth,
  measured by its contribution to `omega · angular_acceleration`. At the
  100 rad/s crossing it still supplies 78.8%.
  The shared model gives similar values, 93.8% and 79.6%. The linear
  contribution is positive on all nine branches for both models.
- The quadratic term becomes increasingly important as motion grows. For the
  candidate at 100 rad/s, its midpoint contribution accounts for a median
  66% of the sum of translational component norms and 20% of angular component
  norms. At 1,000 rad/s these become 96% and 49%; immediately before overflow
  the quadratic term dominates nearly completely.

These are observations along the saved trajectories, not counterfactual
ablations. They support targeting the complete learned state/history feedback,
and do not prove that deleting one term would remove the failure.

## Next intervention and limits

The next named gap is **growth of the complete learned motion-state feedback**.
Test one smooth bounded representation of normalized motion-state features,
used consistently by the current, historical and quadratic readout paths.
Derive its scale from training recordings alone. Keep direct command channels,
learned response timing and shared mechanics. This remains one generic recipe
without vehicle parameters, family selection or consumer options. Freeze the
exact transformation and its evaluation before fitting.

Bounded features alone establish neither accuracy nor stable closed-loop
behavior. The next evaluation must retain physical forecast and command-response
residuals, extreme finite outputs, Crazyflow response/tail regressions and Cascade
crosswind accuracy. These now-inspected branches are diagnostic regression cases;
they cannot become fresh held-out evidence for the next model. Dart controller
adequacy requires its own trial.

All 29 preflight tests passed. Exact replay, independent NumPy reduction and
four actual query/model/trace/summary alterations passed their integrity checks.
The diagnostic source remains isolated at `bc5adaba56b54094ba7cd18220c7706c1b0a26f2`;
no diagnostic option or new model has entered the public API. The public recipe
remains `generic-memory-v4-prototype`.
