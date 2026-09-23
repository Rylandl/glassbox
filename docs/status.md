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
| Experimental online rate head | The fast independent prompt/delayed head remains the accuracy reference: arm-125 250 ms rate RMSE 0.624 rad/s and hard arm-135 6.770. A 0.75 s rate-fit window improves arm-115 but worsens arm-125 response, so it is not adopted. All rate heads remain experimental wrappers, **not in production**. [Early identification](readout-early-identification.md). |
| Command response | The original arm-125 first-probe error is 1.071 with the fast head; a separately excited episode reaches 0.431. Extra unactuated history alone changes it only to 1.064. The excited branch has a different state and does not establish better recovery. Fixed-wing counterfactual truth remains absent. [Early identification](readout-early-identification.md). |
| Scope | One fixed-wing airframe across two recordings and known quad configurations/conditions; a paired quad flight was gentle. Long-horizon fidelity, real-time hardware, calibrated envelopes, held-out configurations and live Throw recovery remain open. |

## Latest result

The [early-identification diagnostic](readout-early-identification.md) used
the original arm-125 prefix and a separately excited Crazyflow branch. The
fast independent head's first command-response error fell **1.071 → 0.431**
relative to each branch's exact plant Jacobian as the weakest/strongest
command singular-value ratio rose 0.046 → 0.170. The branches have different
states; this is evidence for better local identification, not a matched
recovery gain. Using 50 more unactuated transitions changes original response
only 1.071 → 1.064. In the full 263-origin suite a 0.75 s rate-fit window
improves arm-115 250 ms rate RMSE 1.165 → 0.794 rad/s, but arm-125 worsens
0.624 → 0.709 and six-probe mean response worsens 0.379 → 0.693. The window
is not adopted. Saved-data verification passed for every paired and full pack.
No controller or held-out vehicle trial has been run.

## Next iteration

The next gap is **angular recursion after early command identification**.
Keep the fast independent command effects, since excitation makes their local
response substantially more accurate. Test one generic state-dependent
angular term on both the frozen full suite and the separately excited early
episode; require a better 250 ms trajectory without losing the measured
command Jacobian or warm update speed. If that succeeds, fold the result into
one public `fit/predict/update` learner. The Glassbox model can expose
identification uncertainty; an active probe policy belongs with Throw's
controller, and its recovery effect still needs a matched trial.
