# Online fitting: physical-vector errors

The bounded `OnlineFit` procedure now reduces aggregate velocity/rate error
**40.89% against the previous working online fitter**: 24.46% for quads and
53.74% for fixed wings. All twelve primary case/metric comparisons improve.
Error is 73.01% lower than keeping the identical startup fit frozen.

This is the same shared dynamics engine, with changed optimization coordinates
and loss weighting. No vehicle family, mass, inertia, mixer, layout or applied
actuator telemetry enters the learner. The offline learner and saved Dart model
are unchanged. No current-learner Throw recovery or fixed-wing control trial has
run; the original Throw checkout and runtime remain untouched.

## Why the fitting procedure changed

A short prefix can make a nearly constant direction look much more important
than the other axes. In a saved final-cache diagnostic for fixed-wing case 80,
pitch rate error was 0.707 rad/s and yaw error 0.0082 rad/s, yet yaw contributed
about 15 times more loss. Startup-axis normalization rewarded the wrong correction.

Two separately frozen experiments addressed this:

| Candidate | Aggregate error relative to working v2 | Decision |
| --- | ---: | --- |
| v3: compensated feature/output normalization | 1.0141 | Failed; retained as evidence |
| v4: add physical-vector scales and radial Huber loss | 0.5911 | Passes the prospective improvement gate |

V3 grew scales from the causal bounded cache and compensated weights so that
rescaling itself preserved recurrent predictions and command derivatives.
It improved quads, but fixed-wing case 81 rate error increased 6.51 times.
The failed result remains [recorded](online-fit-v3.json).

V4 retains that conditioning and gives each physical vector one fixed bootstrap
scale. Velocity, angular rate and rotation each receive one equally weighted
radial Huber term. Rotation uses chordal matrix distance, approximately angular
error near zero. The loss and its scales respect coordinate-frame rotations;
this does not establish rotational equivariance of the whole supported model.
The bounded replay cache, four-CG proposal, initialization and physical equations
remain unchanged. Motion support is still fixed at startup.

## Comparison and physical errors

The primary comparator is the authenticated saved **v2 online fitter**. The old
Throw identifier supplied the behavior-policy recordings and observed applied
rotor telemetry; this is not a matched-input comparison against that identifier.

The cohort has four arm/inertia variants or scenarios of one Crazyflow
`cf21B_500` quad and two known recordings of one Cascade Skywalker-X8 fixed wing.
Forecasts are scored before their targets are assimilated: **10 ms for quads and
50 ms for fixed wings**. These are known-tape iterations, not blind airframe tests.
The aggregate is a geometric mean of candidate/v2 velocity and rate RMSE ratios,
with equal cases within each family and equal family weights. Every case remains
included, including the quad tape truncated by the old controller's floor contact.

Each cell below is **current / previous online / no-fit kinematic**. Velocity is
m/s; angular rate is rad/s. The kinematic predictor holds velocity and rate and
integrates orientation.

| Case | Velocity RMSE | Body-rate RMSE | Scored transitions |
| --- | ---: | ---: | ---: |
| quad-arm-115 | 0.02208 / 0.02996 / 0.03967 | 0.27256 / 0.27417 / 0.11166 | 875 |
| quad-arm-125 | 0.01220 / 0.02003 / 0.02942 | 0.10730 / 0.13206 / 0.12058 | 875 |
| quad-arm-135 | 1.19929 / 1.27341 / 1.17389 | 1.07632 / 1.71333 / 1.12088 | 62 |
| quad-change | 0.01220 / 0.02006 / 0.02942 | 0.10730 / 0.13208 / 0.12058 | 875 |
| fixedwing-80 | 0.29907 / 1.16253 / 0.16863 | 1.13019 / 2.95346 / 0.37618 | 225 |
| fixedwing-81 | 2.89098 / 3.68517 / 0.17023 | 2.90528 / 4.89786 / 0.37920 | 225 |

Fixed-wing absolute accuracy remains insufficient: the current velocity errors
are 1.77 and 16.98 times the kinematic errors; rate errors are 3.00 and 7.66 times.
There is also a secondary regression: `quad-arm-115` orientation RMSE grows from
0.001319 to 0.004024 rad (3.05 times). The other five orientation scores improve.
These limitations remain visible despite the primary gate passing. The harder
fixed-wing tape includes a 31.14 m/s velocity-error spike at 7.90 s, accounting
for 51.6% of its velocity squared error. It also remains inaccurate afterward,
so removing that sample would not resolve the gap; no sample is removed here.

## Runtime and verification

Quad update median is about 27.4 ms and p95 is 28.0–28.6 ms against a 10 ms
sample interval. Fixed-wing update p95 is 3.92–3.97 ms against 50 ms sampling.
First compiled updates take approximately 2.48 s and 2.07 s respectively.
Warmed prediction p95 stays below 1 ms. The quad real-time target fails.

All **161 tests pass**, including independent scale and IRLS calculations,
frame-invariant loss checks, zero-residual derivatives, prediction-preserving
coordinate transforms, causality, rollback, bounded retention and save/resume.
The wheel imports outside the checkout. Existing saved model predictions and
Dart gradients reproduce exactly; the core dynamics and offline learner did not
change. All 3,137 scored rows, physical metrics and causal journals verify from
archives with zero fits or model calls. Inputs and fixed comparators match v2
and v3 byte-for-byte. The isolated v4/v3 loss change reduces aggregate error
41.71% (8.71% for quads, 62.77% for fixed wings).
Evidence identities and physical metrics are in
[the result index](online-fitting.json); the evaluation contract is
[frozen v4](harness/online-fit-v4.json). See
[replay instructions](../CONTRIBUTING.md#reproduce-streaming-fitting).

## Remaining work

**Forecast spikes and rotation/rate consistency** is the next named gap. In the
quad orientation regression, the predicted rotation implies intermediate angular
motion that disagrees strongly with the endpoint-average rate. The discrepancy
is broad over 2–4 s, not a few initial outliers. The fixed-wing spike and persistent
tail errors accompany heavy startup-support compression, but that does not prove
compression caused them. Inspect internal integration states and derivatives
before selecting a correction. Latency remains separate.

Any next change needs a new frozen comparison against this version. These
recordings do not qualify unseen control authority, a reliability percentage,
or current-model closed-loop recovery.
