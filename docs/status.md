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
| Experimental direct readout | A fresh per-episode feature map with a regularized linear readout is much faster than the maintained online learner. One correction on the completed prefix trajectory materially improves fixed-wing recurrence with unchanged warm-update speed, but arm 125 loses versus direct and the hard quad remains poor. It is **not in production** and does not yet supply the public offline/update workflow. [Trajectory screen](readout-trajectory.md), [gravity screens](readout-structure.md). |
| Command response | An exact replay of the pinned arm-125 plant exposes a large counterfactual 10 ms command-response error in both the fast readout and maintained full learner, despite low on-trajectory one-step errors. [Early quad diagnosis](readout-early-quad.md). |
| Scope | One fixed-wing airframe across two recordings and known quad configurations/conditions; a paired quad flight was gentle. Long-horizon fidelity, real-time hardware, calibrated envelopes, held-out configurations and live Throw recovery remain open. |

## Latest result

The [early quad investigation](readout-early-quad.md) exactly replayed the
pinned arm-125 plant and measured its 10 ms response to each issued motor
command. The cold fast readout's control-response Jacobian has **1.11× the
plant Jacobian's norm in error** at the first origin and remains 0.84–0.94×
after 1.25–2.25 s of actuation. The maintained full learner also starts at
1.08× and reaches 0.57–0.71×. Small on-trajectory one-step errors conceal
this gap. Only 0.25 s of the frozen prefix contains actuated fitting
transitions, and the next commands explore a weakly observed combination.
Simple rate-feedback, lag, trajectory-refit and extra-prefix-data screens did
not fix arm 125. An orthogonal-command simulator probe improved first-origin
response error to 0.705× but changed the flight and left its 250 ms forecast
poor. A 21-parameter angular-momentum screen learned from the same 25 actuated
observations and improved original-branch first-origin response error to 0.873×
and rate endpoint error 8.75 → 6.60 rad/s; it remains inaccurate and has not
been tested on fixed wing. No model or frozen harness was changed; the previous
[full trajectory result](readout-trajectory.md) still defines the candidate's
aggregate accuracy and runtime.

## Next iteration

The next gap is **identified command-to-wrench response**. Add a small,
precomputed, physically matched command-response probe to the existing fast
one-command benchmark, then test one generic additive command-to-wrench head
with compact body-state dependence and angular-momentum recursion. Compare
counterfactual response error and first/late 250 ms forecasts on quads and
fixed wing. Keep the same-episode cold start, no user-provided geometry or
vehicle class, and no pretrained dynamics. The cold trajectory
candidate still needs offline/update, held-out configuration and Throw
qualification before adoption.
