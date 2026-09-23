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
| High-spin diagnosis | The worst 0.85 origin has 0.247 relative 50 ms response error but 27.360 rad/s 250 ms rate error. Independent-input training cuts those to 0.151 and 11.642 at the same target. Native inertial and rotor-momentum terms are large; an oracle with hidden motor state reaches about 1 rad/s. A causal shared-actuator, coupled-inertia screen reached 10.21/15.23 rad/s on the original/excited 0.85 prefixes and did not qualify. [Diagnostic result](high-spin-angular.md). |
| Offline and Dart | `fit/predict/update` use the new angular structure and lifecycle tests pass. No matched offline accuracy fit or live Dart/Throw controller trial has been run for this revision; preceding compact-model results are [historical](nonlinear-temporal.md). |
| Scope | One fixed-wing airframe across two recordings, known quad conditions and two new arm configurations; a paired quad flight was gentle. High-spin recovery, long-horizon fidelity, fresh-start real-time operation, calibrated envelopes, unseen vehicle classes and live Throw recovery remain open. |

## Latest result

The [high-spin response diagnosis](high-spin-angular.md) shows that the worst
0.85 origin's initial command map is substantially closer than its 250 ms
recursion: relative response error is 0.247 at 50 ms and 0.899 at 250 ms,
while the factual 250 ms body-rate error is 27.360 rad/s. Its fitted roll head
has no rate feedback and a large constant canceled by command terms on the
recent fit rows. It fails on later **measured** states, so the immediate cause
is a non-generalizing frozen acceleration equation, not numerical rollout
instability. Explicit episode-fit inertia helps but does not yet beat the
hold-rate reference consistently; no model change was adopted.
The corrected simulator truth and public baseline verify from sealed arrays
without fitting. The [held-out arm qualification](heldout-quad.md) remains the
aggregate reference; known-recording performance is [documented separately](public-rate-memory.md).
The [controlled excitation diagnosis](high-spin-angular.md) shows that
independent 10 ms command perturbations in a separate simulated prefix reduce
the original 0.85 row-150 250 ms error from 27.360 to 11.642 rad/s **when
both fitted models forecast from the same target state**. The matched 50 ms
command-response error falls from 0.247 to 0.151. This supports the
input-information hypothesis, though the training state trajectories also
changed; the endpoint remains slightly worse than the 11.516 rad/s hold-rate
reference. Keeping 50 angular
transitions with that excited fit reaches 8.52 rad/s, while a fixed 50-row
public smoke run regresses fixedwing-80 and two quad arms. Direct trajectory
loss also improved a factual endpoint at the expense of early command-response
accuracy. The public learner is unchanged; all three new saved-data packs
verify without fitting.
The exploratory native-physics ablation at that same origin found about
−7.40 rad/s of roll change from rigid-body inertia and −2.42 from rotor
momentum over 250 ms. A torque map given the simulator's true inertia and
motor trajectory rolled out within 0.97–1.26 rad/s when rotor momentum was
included, versus 4.58–8.20 without it. These are oracle diagnostics, not
causal learner results; the model remains unchanged.
The subsequent [causal shared-actuator screen](high-spin-angular.md#causal-shared-actuator-screen-2026-09-23)
fit nonlinear command memory, coupled inertia, actuator momentum and force
from the same episode. Its 0.85 control/excited matched-state errors were
10.21/15.23 rad/s; the excited fit regressed against the public 11.64, and
simpler coupled solves regressed known quad smoke origins. The unpowered
prefix identified the normalized inertia accurately from observations, but
the hidden applied-actuator trajectory was not recovered well enough from
commands and motion. No public code or benchmark contract changed.

## Next iteration

The named gap remains **high-spin angular generalization**, now narrowed to
**causal applied-actuator identification**. Coupled rigid-body physics has
substantial oracle capacity, and the episode's unpowered motion identifies
inertia. Joint one-step and trajectory fits still let the nonlinear actuator
state trade off against torque coefficients, producing incompatible future
responses from similar prefix losses. The next single-model experiment should
make the actuator state observable from the combined force and angular
measurements, then fit the coupled torque readout. It must learn all response
parameters from the same episode, preserve a zero-momentum solution for
systems without rotating actuation, and screen fixed-wing and known quad
recordings before full qualification. Use the existing frozen suite; do not
add platform branches, consumer tuning or more acceptance guards. Cold-start
latency, live Throw control, fixed-wing counterfactual response and unseen
classes remain separate.
