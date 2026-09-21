# Dart contact precision

On 2026-09-21, the adopted Glassbox model reached **0.720 mm first-contact miss**
in the nominal Crazyflow/Dart task, down from 6.190 mm. The learner, its fitted
weights and its public interfaces are unchanged. This is a consumer-control
improvement with the existing learned model, not new dynamics-accuracy evidence.

The controller now observes every 10 ms, retaining 30 ms planning blocks and the
original 100-iteration cap per solve. Its lateral position cost scales with the
requested target radius; its normal penetration cost retains the original 10 mm
scale. The previous isotropic tightening spent precision on an arbitrary 2 mm
penetration depth at the deadline. The winning change preserves the angular-rate,
axis, contact-velocity, floor, early-crossing and command-smoothness costs.

## Every native attempt

All use the original model, native plant, seed and prelude. Rows after the first
are judged against the strict 1 mm target; the baseline retains its original
20 mm target for exact reproduction. “Other gates” includes alignment, normal and
tangent speeds, and contact before 1.2 s.

| Intervention | Miss (mm) | Other gates | Feedback interval | Gradients |
| --- | ---: | --- | ---: | ---: |
| Exact baseline reproduction | 6.190 | Pass | 30 ms | 4,144 |
| Radius-scaled isotropic position cost | 5.855 | Pass | 30 ms | 3,120 |
| Same cost, faster feedback | 1.897 | Pass | 10 ms | 8,831 |
| Finer 10 ms planning blocks | 4.889 | Pass | 10 ms | 9,204 |
| Remove terminal angular-rate cost | 18.448 | Fail: axis and tangent speed | 10 ms | 7,541 |
| **Lateral precision, original depth weight** | **0.720** | **Pass** | **10 ms** | **9,874** |

Every actual callback, selected mean and native state in these six runs was
finite. The winning trial has 0.930° axis error, 1.456 m/s normal speed,
0.126 m/s tangent speed and contact at 1.192093 s. It took 59.24 s including
callback recording; 103/120 solves converged. Three times as many solves as the
baseline means this is neither an equal-total-compute comparison nor real-time
qualification.

The winning path’s matched 10 ms contact-point forecast RMSE is 0.0349 mm
(p95 0.0720 mm, maximum 0.2103 mm); world-velocity RMSE is 0.00685 m/s and
body-rate RMSE is 0.02735 rad/s. These are on-policy residuals on a changed path,
not a controlled improvement in model accuracy. The saved baseline attribution
compared only prefixes actually executed before replanning; later planned states
were not scored against a different command tape.

## Numerical qualification

The exact winning issued commands were replayed at 1 ms and 0.5 ms, with five
native RK4 substeps per interval. Both float32 replays met every task limit:
0.6070 and 0.6963 mm miss. However, the contact points differed by 97.05 micrometers
and contact times by 136.43 microseconds. **The original float32 convergence
qualification failed** its frozen 20 micrometer / 20 microsecond limits.

A separately frozen float64 audit is in progress. It preserves the exact original
float32 initial state, actual issued commands and rounded plant coefficients,
including the inverse inertia already computed by the original simulator.
It changes native arithmetic precision, not controller decisions or task gates.

## Scope and reproduction

This is one known nominal trajectory with the original warm-start command seed,
exact-state simulated observations and an existing fitted revision. It establishes
neither neighborhood reliability nor hardware precision. No refit, platform branch,
new model option or learned correction was added. The next scientific gap is
performance across prospectively selected neighboring conditions.

[The result index](dart-precision.json) records every trial’s source commit,
protocol hash, manifest hash, numerical outcome and orchestration failures.
The preserved local evidence root is `artifacts/dart-precision-v1`; it and the
original `artifacts/baseline` pack remain outside Git. The winning external Dart
source is retained in `dart-lateral`, with its standalone
[consumer patch](harness/dart-lateral-precision-v1.patch). Historical protocols
and attribution tooling are recoverable from their scientific commits.

Use the [development guide](../CONTRIBUTING.md) for the pinned runtime and commands.
