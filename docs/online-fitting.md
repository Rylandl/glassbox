# Causal online fitting: quad and fixed-wing result

The shared dynamics model now has a bounded streaming fitter, `OnlineFit`. It
initializes from a short observed prefix, predicts before assimilating each new
target, retains a bounded replay cache and saves enough state for exact restart.
The same procedure handles four-command quads and three-command fixed wings.
No vehicle family, mixer, mass, inertia, layout or applied-actuator telemetry
enters the learner. The existing offline learner and Dart model are unchanged.

**The second candidate passes the frozen improvement gate:** equal-family
aggregate velocity/rate error is **54.34% lower than the identical frozen
startup fit** (68.30% for quads, 34.24% for fixed wings). It is 83.38% lower than
the failed first online candidate. All 3,137 scored predictions and observations
complete with finite values. This is a known-tape identification result, not a
controller recovery or broad generalization result. The comparator is this same
learner held at its startup fit; this is not a comparison with Throw’s old
identifier, which received applied rotor telemetry unavailable to this fitter.

The cohort contains arm/inertia variants of one Crazyflow `cf21B_500` quad and
two known recordings of one Cascade Skywalker-X8 fixed wing. It does not cover
a broad set of airframes. The old Throw checkout and runtime remain unchanged.

## What changed

The first candidate used persistent Adam with acceptance on four sampled windows.
It failed: aggregate error was 2.748 times the frozen fit. Saved-model checks
found no training/prediction mismatch or material float32 explanation. Later
commands reached 102 startup standard deviations, and locally accepted changes
could worsen the full retained recent buffer. That failed result is preserved
in [the v1 evidence](online-fit-v1.json).

The maintained candidate replaces that optimizer with one damped Gauss-Newton
proposal per observation, four matrix-free conjugate-gradient iterations, and
acceptance against the full bounded replay cache. A prediction-change limit and
actual-versus-predicted improvement ratio govern the step. Initialization,
features, normalization, objective and dynamics remain unchanged. The Adam
streaming implementation was deleted; there is no selectable optimizer catalog.

## Physical errors

One-step vector RMSE, measured before each target is assimilated. Each cell is
**online / frozen startup / no-fit kinematic**. Velocity is m/s; rate is rad/s.
The kinematic predictor holds velocity and rate and integrates orientation.
The aggregate is the geometric mean of online/frozen velocity and rate error
ratios, with equal case weights within each family and equal family weights.
These are 10 ms forecasts for quads and 50 ms forecasts for fixed wings.

| Case | Velocity RMSE | Body-rate RMSE | Scored transitions |
| --- | ---: | ---: | ---: |
| quad-arm-115 | 0.02996 / 0.13472 / 0.03967 | 0.27417 / 2.30922 / 0.11166 | 875 |
| quad-arm-125 | 0.02003 / 0.03761 / 0.02942 | 0.13206 / 0.68032 / 0.12058 | 875 |
| quad-arm-135 | 1.27341 / 1.33648 / 1.17389 | 1.71333 / 4.52537 / 1.12088 | 62 |
| quad-change | 0.02006 / 0.03761 / 0.02942 | 0.13208 / 0.68033 / 0.12058 | 875 |
| fixedwing-80 | 1.16253 / 1.30268 / 0.16863 | 2.95346 / 2.85618 / 0.37618 | 225 |
| fixedwing-81 | 3.68517 / 7.20476 / 0.17023 | 4.89786 / 12.36555 / 0.37920 | 225 |

`fixedwing-80` rate error regresses 3.41% versus its frozen fit; it remains in the
aggregate. The other 11 primary case/metric comparisons improve. More seriously,
fixed-wing errors remain 6.9–21.6 times the kinematic velocity error and 7.9–12.9
times its rate error. The relative improvement is useful; absolute accuracy is
not yet adequate for promoting the learner into the Throw controller.

`quad-arm-135` is a 62-transition truncated tape: the old collection controller hit the
floor at 1.87 s. The terminal sample includes the simulator zeroing translational
velocity. It remains in the primary result; a descriptive preterminal check
still shows improved error versus frozen. The other quad tapes span 10 s. The
hidden arm/inertia change occurs at 4 s, but commands are already nearly equal
and steady. Its result is weak evidence about identifying new control authority.

## Runtime and qualification

Warmed quad updates have median 26.5 ms and p95 26.9–27.9 ms against a 10 ms sample
interval. Fixed-wing updates have median 3.43–3.44 ms and p95 3.70–3.75 ms against
50 ms. First compiled updates take 2.09 s and 1.81 s respectively. Startup ridge
initialization takes roughly 0.9–3.0 ms; first compiled prediction is separate.
Prediction p95 is below 1 ms in these runs. The declared real-time target fails
for quads; no asynchronous fitting or controller deadline claim is established.

All 138 tests pass; the wheel builds and imports outside the checkout. Saved
v1/v2 input tapes, source rows, timestamps, truth, frozen predictions and
kinematic predictions match bitwise. Both result packs verify their checksums,
causal journals, optimizer accounting and metrics with zero fits/model calls.
See [machine-readable results and authorities](online-fitting.json), the
[frozen v2 protocol](harness/online-fit-v2.json) and
[replay commands](../CONTRIBUTING.md#reproduce-streaming-fitting).

## Next named gap

**Cold-start conditioning and motion support.** Sparse startup observations
produce tiny scales in weakly excited command, rotation and rate directions.
Nearly every later observation leaves the initial support. The optimizer fix
substantially improves stability but does not restore lost state information or
supply missing excitation. The next iteration should address that conditioning
causally, preserve model meaning during any coordinate change, and use physical
errors against the kinematic baseline as well as the frozen-start comparison.
Keep update latency separate. Current-learner Throw recovery and fresh fixed-wing
flight trials follow identification improvement; neither has run here.
