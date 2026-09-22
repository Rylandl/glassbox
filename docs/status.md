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
| Experimental direct readout | A fresh per-episode frozen feature map with a regularized linear readout produces much faster complete updates and better local accuracy. Gravity-free features improve fixed-wing and hard-quad forecasts but hurt ordinary-quad settled forecasts. Smoothly bounded gravity recovers late quad behavior but creates an early arm-115 outlier; physical-scale gravity behaves like gravity-free. The readout is **not in production** and does not yet supply the public offline/update workflow. [Structural screens](readout-structure.md). |
| Scope | One fixed-wing airframe across two recordings and known quad configurations/conditions; a paired quad flight was gentle. Long-horizon fidelity, real-time hardware, calibrated envelopes, held-out configurations and live Throw recovery remain open. |

## Latest result

The frozen [gravity-structure screens](readout-structure.md) each refit from a
fresh episode prefix and used **4,187 causal updates and 263 conditional
origins**. Gravity-free features improved fixed-wing 250 ms body-rate error
**2.593 → 1.520** and **6.027 → 1.906 rad/s**, and the hard quad **62.867 →
36.821**, but ordinary quad arm 125 worsened **2.171 → 3.143**. Smoothly
bounded gravity recovered late ordinary-quad error (arm 125 median 0.210 →
0.048 rad/s versus gravity-free) yet raised the early arm-115 worst error to
28.275 rad/s. Physical-scale gravity nearly reproduced gravity-free. Both new
full packs verify without refitting; their experimental code was reverted.
The maintained learner, public interfaces and frozen harness remain unchanged.

## Next iteration

The next gap is **angular recurrence**: one-step rate accuracy is close, but
small errors grow over 250 ms in ordinary-quad hover and early fixed-wing
rollouts. Test whether a short trajectory-error objective for the same generic
readout fixes that divergence at an acceptable whole-update cost, using the
existing [shared benchmark](benchmarking.md) and separately reporting first,
late, velocity and rate error. Avoid a vehicle branch, pretrained core,
acceptance guard or another gravity-scale sweep. If a better model survives,
qualify offline fit/update and held-out configurations before adoption. A fresh
Throw controller trial remains separate.
