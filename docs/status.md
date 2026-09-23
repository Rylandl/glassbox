# Current state

Updated 2026-09-23. Read [the charter](charter.md) first. The shared-physics
learner with compact nonlinear history, stable accumulators, and an
episode-fitted angular-rate memory is the **single maintained implementation**.
It fits each configuration from that episode's motion, issued commands and
timing; no vehicle family, mixer, mass or inertia is supplied. Public `fit`,
`predict`, immutable `update`, save/load and bounded `OnlineFit` remain intact.
The Throw requirement excludes any pretraining.

| Area | Current evidence and limit |
| --- | --- |
| Maintained model | Generic gravity, rigid-body kinematics and frames; learned three-axis force readout and episode-fitted command/rate response. Four-command/10 ms model has 4,340 parameters; three-command/50 ms has 2,512. [Current result](public-rate-memory.md). |
| Online identification | The public learner completed 4,187 causal updates and 263 frozen forecast origins. Fixed-wing 250 ms body-rate RMSE is 0.345/0.597 rad/s; quad arm-115/125/135 is 0.841/0.551/6.237. Warm CPU updates were about 0.92 ms fixed-wing and 1.7 ms quad; cold compilation and first updates are far slower. [Current result](public-rate-memory.md). |
| Command response | Arm-125 six-probe mean relative error is 0.331, but the first underexcited probe remains 1.071. The separately excited branch reaches 0.417 at a different state; this does not establish better recovery. Fixed-wing counterfactual truth remains absent. [Current result](public-rate-memory.md). |
| New configurations | On two newly collected arm geometries, 1.40 reaches 0.455 rad/s 250 ms rate RMSE across 35 origins, while 0.85 reaches 19.827 rad/s across two high-spin origins, worse than the 9.073 rad/s hold-rate reference. [Held-out result](heldout-quad.md). |
| High-spin diagnosis | A sealed counterfactual probe finds 0.247 relative 50 ms command-response error at the worst 0.85 origin, but 27.360 rad/s factual 250 ms rate error. Inertial coupling contributes; attempted short-window and momentum fits have not qualified a replacement. [Diagnostic result](high-spin-angular.md). |
| Offline and Dart | `fit/predict/update` use the new angular structure and lifecycle tests pass. No matched offline accuracy fit or live Dart/Throw controller trial has been run for this revision; preceding compact-model results are [historical](nonlinear-temporal.md). |
| Scope | One fixed-wing airframe across two recordings, known quad conditions and two new arm configurations; a paired quad flight was gentle. High-spin recovery, long-horizon fidelity, fresh-start real-time operation, calibrated envelopes, unseen vehicle classes and live Throw recovery remain open. |

## Latest result

The [high-spin response diagnosis](high-spin-angular.md) shows that the worst
0.85 origin's initial command map is substantially closer than its 250 ms
recursion: relative response error is 0.247 at 50 ms and 0.899 at 250 ms,
while the factual 250 ms body-rate error is 27.360 rad/s. Explicit episode-fit
inertia helps but does not yet beat the hold-rate reference consistently, and
other apparently good fits fail different origins. No model change was adopted.
The corrected simulator truth and public baseline verify from sealed arrays
without fitting. The [held-out arm qualification](heldout-quad.md) remains the
aggregate reference; known-recording performance is [documented separately](public-rate-memory.md).

## Next iteration

The next gap remains **high-spin angular generalization**, now narrowed to an
identifiable angular-momentum recurrence with hidden actuator response. Fit
constant physical terms and changing disturbances from the current episode
without assuming an unpowered segment or exposing vehicle metadata. Use the
frozen 50–250 ms factual and counterfactual measurements for fast development,
then qualify any candidate on fresh high-spin configurations and the known
fixed-wing/quad suite. The 0.85/1.40 recordings are development evidence, not
blind holdouts. Cold-start latency, live Throw controller integration,
fixed-wing counterfactual response and unseen classes remain separate.
