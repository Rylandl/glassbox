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
| Scope | One fixed-wing airframe across two recordings and known quad configurations/conditions; a paired quad flight was gentle. Long-horizon fidelity, real-time hardware, calibrated envelopes, held-out configurations and live Throw recovery remain open. |

## Latest result

The frozen [trajectory screen](readout-trajectory.md) fit the gravity-free fast
readout against an observed 250 ms prefix trajectory, then replayed **4,187
causal updates and 263 conditional origins**. Fixed-wing 250 ms body-rate
error fell from 1.520 to **1.104** and from 1.906 to **0.650 rad/s** relative
to gravity-free, with warm whole updates unchanged at about 0.54 ms. Quad arm
115 improved 2.608 → **2.570**, but arm 125 stayed worse than direct
(**3.170** versus 2.171), and hard quad remained at **35.224 rad/s**. First
shape-specific compilation brought cold initialization plus first forecast to
about **1.7–2.0 s**. Three full packs verify without refitting. A per-update
trajectory correction helped late hover but raised warm quad updates to about
3.1 ms without resolving arm 125; it was removed from the current candidate.
The maintained learner, public interfaces and frozen harness remain unchanged.

## Next iteration

The next gap is **early quad angular response**. On arm 125, the true pitch
rate reverses to about +5 rad/s during the first 250 ms, while all readout
variants remain negative; the next commands are within the prefix's individual
channel ranges. Determine whether their combined command direction and rate
state were identified from the causal prefix, then test one generic network
change that addresses the missing coupling. Use the existing
[shared benchmark](benchmarking.md) and keep first and late physical errors
separate. Do not add a vehicle branch, pretrained core, acceptance guard or
strength sweep. The cold trajectory candidate still needs offline/update,
held-out configuration and Throw qualification before adoption.
