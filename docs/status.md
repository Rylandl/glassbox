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
| Experimental online rate head | The fast independent prompt/delayed head has arm-125 250 ms rate RMSE 0.624 rad/s and hard arm-135 6.770. A causal physical-time coupled head improves fixed-wing-81 and hard arm-135 but loses arm-125/115 accuracy and takes 3.5–7.3 ms per warm update. Both remain experimental wrappers, **not in production**. [Comparison](readout-coupled-lag.md). |
| Command response | The arm-125 simulator probe measures issued-command impact. Coupling the response improves the first probe 1.071 → 0.533 relative error but worsens six-point mean 0.379 → 0.478. Fixed-wing counterfactual truth remains absent. [Comparison](readout-coupled-lag.md). |
| Scope | One fixed-wing airframe across two recordings and known quad configurations/conditions; a paired quad flight was gentle. Long-horizon fidelity, real-time hardware, calibrated envelopes, held-out configurations and live Throw recovery remain open. |

## Latest result

The [coupled physical-time screen](readout-coupled-lag.md) tested a single
causally selected lag in seconds and one gain direction per command on all
263 frozen origins. Versus the faster independent prompt/delayed head,
fixed-wing-81 250 ms rate/velocity RMSE improves 0.599/1.198 → 0.482/0.794,
and hard arm-135 rate improves 6.770 → 6.094 rad/s. Arm-125 rate worsens
0.624 → 0.928 and arm-115 1.165 → 1.404; warm updates cost several times
more. The first arm-125 command-response error improves **1.071 → 0.533**, but
the six-probe mean worsens 0.379 → 0.478. Saved-data verification passed. The
physical-time model clarifies the early-sensitivity/recursion tradeoff, but it
is not adopted into the public learner. No controller or held-out vehicle
trial has been run.

## Next iteration

The next gap is **short-episode command-response identifiability**. Diagnose
which control directions and physical lag scales are actually excited during
the first 25 transitions, then test one compact response structure that keeps
the coupled head's early sensitivity while preserving the independent head's
quad rollout. Use the same one-command frozen suite and a targeted, physically
excited episode if the existing prefix cannot separate the effects. A model
that survives that test should be folded into one public `fit/predict/update`
implementation, then evaluated on held-out conditions and Throw from a fresh
start.
