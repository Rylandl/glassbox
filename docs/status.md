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
| High-spin diagnosis | The worst 0.85 origin has 0.247 relative 50 ms response error but 27.360 rad/s 250 ms rate error. A matched-state independent-input replay cuts those to 0.151 and 11.642; a fixed longer angular window helps there but regresses known cases. No replacement qualified. [Diagnostic result](high-spin-angular.md). |
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

## Next iteration

The next gap remains **high-spin angular generalization**: retain persistent,
regularized information about command effects across the episode while
allowing state-dependent dynamics to adapt locally. A single angular-momentum
head with positive-definite inertia and a causal applied-command state is the
structural candidate, but the naive joint fit already failed known cases.
Screen a compact information-retaining update on the frozen 50–250 ms factual
and counterfactual suite, including the new excitation packs, and judge
control response alongside trajectory accuracy. Qualify a successful candidate
on fresh high-spin configurations. The 0.85/1.40 recordings are development
evidence, not blind holdouts. Cold-start latency, live Throw controller
integration, fixed-wing counterfactual response and unseen classes remain
separate.
