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
| Experimental direct readout | A fresh per-episode frozen feature map with a regularized linear readout produces much faster complete updates and better local accuracy. Removing learned body-gravity features improved fixed-wing and hard-quad 250 ms forecasts but worsened ordinary-quad late forecasts. It is **not in production** and does not yet supply the public offline/update workflow. [Structural screen](readout-structure.md). |
| Scope | One fixed-wing airframe across two recordings and known quad configurations/conditions; a paired quad flight was gentle. Long-horizon fidelity, real-time hardware, calibrated envelopes, held-out configurations and live Throw recovery remain open. |

## Latest result

The frozen [gravity-feature screen](readout-structure.md) refit each episode
prefix with body-gravity direction removed from the learned acceleration map;
exact gravity remained in the integrator. Across **4,187 causal updates and
263 conditional origins**, fixed-wing 250 ms body-rate error improved **2.593
→ 1.520** and **6.027 → 1.906 rad/s** against the prior direct readout. The
hard quad improved **62.867 → 36.821 rad/s** but remains unacceptable. Ordinary
quad arm 125 worsened **2.171 → 3.143 rad/s**, and its median per-origin rate
error worsened more sharply. First-step accuracy was similar. All saved scores,
source controls and copied artifact packs verify without fitting. A linear-only
gravity variant lost the fixed-wing gain in a four-case smoke and was rejected.
Production source and public interfaces remain unchanged. This is a
shape-preserving ablation, not yet a smaller network or a held-out result.

## Next iteration

The next gap is to retain the gravity-free fixed-wing and hard-quad gains while
recovering ordinary-quad late forecasts through a better generic
command/hidden-response representation. Use the existing [shared readout
benchmark](benchmarking.md), which runs all eight cases in seconds, and score
first and late physical errors separately. Do not add a vehicle branch,
pretrained core, acceptance guard or strength sweep. If a genuinely better
representation survives, remove inactive gravity coordinates from the actual
arrays, then qualify offline fit/update and held-out configurations before
adoption. A fresh Throw controller trial remains separate.
