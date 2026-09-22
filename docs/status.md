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
| Experimental online rate head | A causal 15–25-transition additive command map with nonnegative diagonal rate damping, combined with the previous force readout, improves every known recording's 250 ms body-rate RMSE. It is an experimental wrapper, **not in production**, and lacks unified offline/update and a learned actuator lag. [Full screen](readout-positive-damping.md). |
| Command response | The frozen arm-125 simulator probe measures actual issued-command impact. The previous cold readout's mean relative error is 0.934; the positive-damping hybrid reaches 0.629. Fixed-wing counterfactual truth remains absent. [Full screen](readout-positive-damping.md). |
| Scope | One fixed-wing airframe across two recordings and known quad configurations/conditions; a paired quad flight was gentle. Long-horizon fidelity, real-time hardware, calibrated envelopes, held-out configurations and live Throw recovery remain open. |

## Latest result

The [positive-damping screen](readout-positive-damping.md) promoted the
exact-replay arm-125 counterfactual probe into the one-command benchmark and
compared the previous cold readout with a compact causal rate head over all
eight known recordings and 263 forecast origins. The rate head's 250 ms
body-rate RMSE moved **3.170 → 1.237 rad/s on arm 125**, **35.224 → 9.001 on
hard arm 135**, **1.104 → 0.403 on fixedwing 80**, and **0.650 → 0.578 on
fixedwing 81**. Arm-125 six-point mean response error moved **0.934 → 0.629**
relative to the plant Jacobian. The gentle paired quad's velocity error rose
0.009 → 0.044 m/s, and fixed-wing velocity also regressed. Saved-data
verification passed. The screen is still a force/rate wrapper with a fixed
0.05 s lag, not the one maintained public learner, and lag sensitivity is
large. No controller or held-out configuration trial has been run.

## Next iteration

The next gap is **episode-identified actuator lag and a unified rate head**.
Fit command lag from the same causal observations, integrate the small
positive-damping angular model and the learned force model into one JAX
rollout and one public `fit/predict/update` procedure, and rerun the frozen
suite without retuning to its outcomes. Preserve the quick closed-form warm
update. Then test held-out conditions and Throw from a fresh start; the current
hybrid result alone is not adoption evidence for the public model.
