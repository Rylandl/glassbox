# Independent recordings improve bounded public updates

Adding 72 independent excited training recordings per simulator reduced held-out
forecast error by **13.88%** and command-response error by **5.81%** against the
exact saved public-v4 controls. The unchanged public `update` call used the same
1,536 training windows, original 256 development windows and 1,000 fitting steps.
The [frozen result](harness/independent-training-recordings-v1-result.json) passes
its residual criteria, exact replays, independent reduction and all eight
artifact-alteration challenges. It qualifies this bounded offline update
intervention; the public recipe remains `generic-memory-v4-prototype`.

## Physical accuracy

Primary 250 ms component RMSE, saved control → updated revision:

| Simulator | Forecast velocity, m/s | Forecast rate, rad/s | Response velocity, m/s | Response rate, rad/s |
| --- | ---: | ---: | ---: | ---: |
| Crazyflow, available truth | 0.04757 → 0.03630 | 0.15216 → 0.12329 | 0.02613 → 0.02075 | 0.16336 → 0.14159 |
| Cascade | 0.06802 → 0.06712 | 0.04791 → 0.04625 | 0.01876 → 0.02014 | 0.01757 → 0.01788 |

Eighteen of twenty simulator/scope/kind aggregates improve. Cascade's primary
and wind-shift response aggregates worsen by 3.72% and 9.61%, within the frozen
limits. Thirty-eight of 180 endpoint component cells worsen; only 37 have a
floored ratio above one. The largest relative loss is Cascade maneuver-shift
50 ms response velocity, 0.001105 → 0.001684 m/s. A larger absolute velocity
loss occurs in its maneuver-shift 150 ms forecast, 0.03902 → 0.04907 m/s.

Crazyflow's targeted primary angular forecast/response means improve by
18.98%/13.33%; their parent-p95 errors improve by 27.03%/11.81%. All twelve
primary tail guards pass. These are matched physical errors, not an application
accuracy threshold or evidence of safe controller swapping.

## What changed and what was held fixed

Each candidate adds 72 new seeded recordings to the original 72 training
parents, preserving their condition allocation. Its actual deterministic cache
contains 768 old and 768 new windows, represents all 144 parents, and exactly
equals full extraction from that pool. The original development set and saved
control revision remain unchanged. Training normalizers, initialization and
channel weights are recomputed from the new cache, so this is a data-support
intervention rather than an architecture-only comparison.

Every new recording has a usable prefix. Crazyflow completes 24 training
flights and has 48 later altitude failures; valid prefixes contain 148–300
transitions against a 75-transition minimum. Cascade completes all 72. Fresh
confirmation completes 59/84 Crazyflow and 84/84 Cascade flights. Crazyflow's
primary 250 ms forecast/response truth is 422/480 and 712/768 queries. Both
revisions use identical available truth; changed completion counts from earlier
cohorts are not learner gains.

Exactly two public updates ran, selecting step 1,000 with 1,536,000 gradient-window
visits each. Their supervised stage times were 347.36 s and 46.92 s. Backtracking
work differed: 5,035 and 2,520 full-training objective calls. No control refit,
scientific retry, replay fit or replay initializer ran.

## Verification and remaining gap

All 195 focused regression tests pass. Native replay reproduces 56,148 arrays
from all 312 newly collected training/confirmation parents. Saved-model replay
reproduces 12,772 prediction/envelope arrays and exact metrics. An independent
NumPy reduction agrees on 257,040 raw rows, 1,260 summaries, 21,168 parent
summaries, 10,584 cell summaries and 840 tail summaries. Every one of eight
locally resealed altered artifacts is rejected by its named validator and the
unchanged external seal. The
[durable evidence inventory](../artifacts/2026-09-19/independent-training-recordings-v1-evidence/copy-inventory.json)
retains 18,279 independently copied files; all hashes and distinct inodes were
checked.

More independent data helped, but did not eliminate the within-fit generalization
gap. Crazyflow training 250 ms angular RMSE falls 0.15617 → 0.08900, while
development rises 0.09784 → 0.13045 from initialization. Every development angular
horizon worsens, even though the selected updated model beats the previous
model's development angular error by 20.3%. Velocity and rotation improvements
allow total development loss to fall.

The next proposed experiment is one fixed, generic penalty on forecast drift
from the initialized mean, with data, caches and work held fixed and fresh
confirmation. It may improve generalization or suppress useful learned dynamics;
neither outcome is established yet. Its strength, harness and criteria must be
frozen before fitting. Calibration, physical derivative fidelity, controller
readiness and safe live swaps remain separately unqualified.
