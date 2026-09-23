# Current state

Updated 2026-09-22. Read [the charter](charter.md) first. The shared-physics
learner with compact nonlinear history and stable accumulators is the **single
maintained implementation**. It fits each configuration from that episode's
motion, issued commands and timing; no vehicle family, mixer, mass or inertia
is supplied. Public `fit`, `predict`, immutable `update`, save/load and bounded
`OnlineFit` remain intact. The Throw requirement excludes any pretraining.

| Area | Current evidence and limit |
| --- | --- |
| Maintained model | Generic gravity, rigid-body kinematics and frames; learned command response, accelerations and delayed/hidden dynamics. Four-command/10 ms model has 5,450 parameters; three-command/50 ms has 3,103. [Architecture evidence](nonlinear-temporal.md). |
| Offline and Dart | Matched offline fits changed equal-family forecast error +2.08% and command-response error −2.18% versus the prior full-history model. The compact-model Dart forecast improved at 10–1,200 ms, but no new controller trial established its catch performance. [Results](nonlinear-temporal.md). |
| Maintained online learner | Six recorded streams/3,137 causal updates were effectively equal in aggregate accuracy to the former full-history learner, but the compact model did not speed whole CPU updates. Quad median was 33.9 ms. No demonstrated real-time fitting or held-out vehicle calibration. [Results](nonlinear-temporal.md). |
| Experimental online rate head | Current and leaky commands plus positive rate damping, fitted from 15–25 causal transitions, cut arm-125 250 ms rate RMSE to 0.624 rad/s and hard arm-135 to 6.770. Force and rate now share one JAX rollout but remain an experimental wrapper, **not in production**. The leaky basis still scales with sample interval. [Full screen](readout-distributed-lag.md). |
| Command response | The frozen arm-125 simulator probe measures actual issued-command impact. Mean relative error is 0.379 versus 0.629 for the prior hybrid, but the first probe worsens 0.808 → 1.071. Fixed-wing counterfactual truth remains absent. [Full screen](readout-distributed-lag.md). |
| Scope | One fixed-wing airframe across two recordings and known quad configurations/conditions; a paired quad flight was gentle. Long-horizon fidelity, real-time hardware, calibrated envelopes, held-out configurations and live Throw recovery remain open. |

## Latest result

The [distributed-lag screen](readout-distributed-lag.md) fits prompt and
leaky command effects separately in the compact causal rate head and advances
it with force in one JAX rollout. Against the earlier fixed-lag hybrid, 250 ms
body-rate RMSE moved **1.237 → 0.624 rad/s on arm 125** and **9.001 → 6.770
on hard arm 135**. Fixed-wing rate moved 0.403 → 0.486 and 0.578 → 0.599;
fixed-wing-81 velocity moved 0.768 → 1.198 m/s. Arm-125 mean six-point
response error improved **0.629 → 0.379**, but its first probe worsened 0.808
→ 1.071. The full 263-origin artifact passed saved-data verification, and a
code cleanup preserved every saved numerical array exactly. This remains an
experimental wrapper and its lag basis is sampling-interval dependent. No
controller or held-out configuration trial has been run.

## Next iteration

The next gap is **physical-time command response without losing early Throw
accuracy**. Replace the sample-interval-scaled leaky basis with one generic,
causally fitted physical-time response in the experimental JAX learner. Use
the same frozen forecast and arm-125 counterfactual suite; require both early
response and hard-case recursion to be visible in the decision. If that
structure survives, fold the rate head into one public `fit/predict/update`
implementation and then test held-out conditions and Throw from a fresh
start.
