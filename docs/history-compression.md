# Learned temporal compression

The current quad recipe has 10,130 parameters. Its 17 current features expand to
195 dense-head inputs through ten separately weighted lag rows and eight hidden
values. Most of the parameter count is the repeated connection from these lag
features into the linear, nonlinear and memory heads. The physics equations do
not require that representation.

The first candidate learns two temporal basis vectors, shared across feature
channels and all three heads. For each feature it computes two weighted sums of
past-minus-current differences. The heads therefore receive 59 inputs on the
quad, reducing its parameter count to 3,894 (61.56%). The three-command, 50 ms
fixed-wing case changes from 3,399 to 3,403 parameters. There is no vehicle-class
branch. Input count and sample interval determine dimensions through one recipe.

Initialization uses orthonormal average and trend vectors; subsequent fitting
learns both vectors. They can emphasize particular lags or form signed temporal
contrasts. The 100 ms lag span and 500 ms observed context stay fixed. This is
not a learned context horizon or persistent accumulator. The tiny basis still
grows with the number of samples, while the large dense matrices do not.

## Why start here

This changes the history representation while retaining known mechanics, the
32-unit nonlinear head, eight memory coordinates, command filtering and the
entire online solver. A full-rank basis is an exact linear change of coordinates;
two columns explicitly restrict the independent temporal patterns. Whether two
are sufficient is the empirical question. Learned free basis coefficients also
introduce scaling freedom with the head weights, so optimizer conditioning must
be measured rather than presumed unchanged.

Two other structures remain useful hypotheses:

- Stable accumulators with learned decay times: `s_next = a*s + (1-a)*feature`,
  with `a = exp(-dt/tau)` and positive learned tau. These have a fixed state size
  and timescales expressed in seconds. Several states can represent fast and slow
  response, but a small exponential bank may approximate sharp delays poorly.
- Learned lag locations or a polynomial memory basis: these can retain delayed
  events or trends using few coordinates. Interpolated lag positions need care
  around knot boundaries; polynomial projection adds a more structured memory
  mechanism and a different approximation assumption.

Persistently caching an accumulator is not automatically valid during online
fitting: changing its learned time constants or upstream feature normalization
changes the state that the same observed history would have produced. A future
implementation must reconstruct that state or verify an exact transformation.

[HiPPO](https://arxiv.org/abs/2008.07669) provides a principled formulation of
compressing histories into fixed-size polynomial projections.
[LRU](https://arxiv.org/abs/2303.06349) studies efficient linear recurrence design.
They motivate this direction; this rank-two experiment implements neither method
and borrows no accuracy or speed result from their benchmarks.

## Evaluation contract

[The frozen screen](harness/history-basis-v1.json) uses the same six known online
tapes and all 3,137 targets as adopted v8. Each configuration initializes only
from its original visible prefix. Every prediction precedes assimilation of its
target; accuracy is compared with the adopted online fitter, not startup-only
predictions. Aggregate, family and tail gates are declared prospectively.

All mechanics, causality, derivative and persistence tests must pass. Initial
and final sessions, row-level forecasts, source truth, timings and update reports
are retained for verification without refitting. Historical timing is descriptive;
a fresh paired comparison follows only if accuracy passes. A successful online
screen still requires matched offline forecast/response fits and Dart evaluation
before adoption. The main implementation stays unchanged during this experiment.

## Completed screen: retained, not adopted

Source `3fc490a`, protocol `89303fd`. All six known tapes completed: 3,137
predictions made before their targets were assimilated, with six prefix-only
initializations. Saved-data verification passes, including source truth, causal
journals, cache/session identities, counter/acceptance transitions and recomputed
scores. There were 472 existing/architecture test passes and two screen-test
passes. No offline fit or closed-loop controller trial was run.

| Equal-family metric | Candidate / adopted v8 | Change |
| --- | ---: | ---: |
| Velocity/rate combined | 0.94870 | 5.13% lower error |
| Velocity alone | 0.83338 | 16.66% lower error |
| Body rate alone | 1.07999 | 8.00% higher error |
| Worst-decile velocity/rate | 0.91440 | 8.56% lower error |
| Orientation | 1.06712 | 6.71% higher error |
| Rotation/rate defect | 0.97770 | 2.23% lower error |

Combined primary error falls 4.12% for quad and 6.13% for fixed-wing. The size,
primary, family, tail and rotation/rate gates pass. The aggregate orientation
gate fails its prospectively frozen maximum ratio of 1.02. This is not a veto
on a single case. All individual results remain in the linked JSON evidence.
In particular, quad135 body-rate RMSE rises from 0.35416 to 0.61630 rad/s; quad115
improves both velocity and rate, whereas quad125/change trade lower velocity
error for higher rate error.

Warm full-update medians are 85.52–86.05 ms for quad and 11.32 ms for fixed-wing;
quad p95 is 89.03–90.56 ms. The parameter reduction has not established a runtime
benefit or reached the 10 ms quad observation cadence. These are synchronized
end-to-end measurements, but not a fresh paired comparison with the incumbent.
That later timing stage was not launched after the orientation gate failed.

The result supports investigating compact temporal representations. It does not
show that this particular rank restriction is sufficient. Free basis learning
also changes optimization coordinates and initialization; the experiment does
not isolate information loss from fitting behavior. A recurrence redesign needs
both angular-response evidence and a compiled-execution cost explanation, rather
than assuming fewer weights means faster updates.

The next architectural hypothesis is stable accumulators with learned time
constants and a fixed state dimension. Before freezing that candidate, inspect
the compiled derivative work and the angular errors here. Preserve the generic
mechanics and command-count contract; test against the adopted fitter with the
same physical metrics. Broad offline/response and Dart qualification remains
required before any replacement.

Evidence is at `artifacts/history-basis-v1/evaluation`, externally sealed by
`24f942c2532eaedc4791b436874698fa8db0897bf34adb0bcb520c2b6e9f942b`.
The [machine-readable result](history-compression.json) records every case and
source/authority. Reproduce verification in a checkout of `3fc490a`:

```sh
PYTHONPATH=scripts SCIPY_ARRAY_API=1 python scripts/screen_history_basis.py verify \
  --output /Users/ryland/autonomy/glassbox/artifacts/history-basis-v1/evaluation \
  --manifest-sha256 24f942c2532eaedc4791b436874698fa8db0897bf34adb0bcb520c2b6e9f942b
```

The preceding arithmetic-only history-projection candidate (`c631a1c`) also
remains unadopted. Its online primary error improved 3.48%, but saved offline
preservation failed on 198 gradient arrays and 17 objective values out of 21,096
comparisons. Forecast arrays and selected trajectories passed their bounds.
Its first paired timing attempt encountered different background load; the
planned interleaved rerun was stopped after the preservation failure. Both
attempts and their authorities are retained in the JSON result. Neither change
has entered the maintained implementation.
