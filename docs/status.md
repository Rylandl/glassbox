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
| Experimental online rate head | A dissipative angular-rate memory improves 250 ms rate RMSE on arm-115 1.165 → 0.841, arm-125 0.624 → 0.551, and hard arm-135 6.770 → 6.238 rad/s. Fixed-wing rate RMSE worsens 0.486 → 0.509 and 0.599 → 0.634. Warm updates remain about 0.8–1.5 ms. This is the experimental reference, **not in production**. [Rate-memory result](readout-rate-memory.md). |
| Command response | Arm-125 six-probe mean relative error improves 0.379 → 0.331 with angular memory, but the first underexcited probe remains 1.071. The separately excited branch reaches 0.417 at a different state; this does not establish better recovery. Fixed-wing counterfactual truth remains absent. [Rate-memory result](readout-rate-memory.md). |
| Scope | One fixed-wing airframe across two recordings and known quad configurations/conditions; a paired quad flight was gentle. Long-horizon fidelity, real-time hardware, calibrated envelopes, held-out configurations and live Throw recovery remain open. |

## Latest result

The [rate-memory screen](readout-rate-memory.md) added one nonnegative,
state-dependent angular damping term to the fast independent rate head. The
frozen 263-origin suite improves 250 ms body-rate RMSE on all three meaningful
quad arm cases, while the two fixed-wing recordings lose modestly. Arm-125
six-probe response mean improves 0.379 → 0.331; its first underexcited error
stays 1.071. The separately excited branch improves 0.431 → 0.417 and its
own 250 ms endpoint 4.385 → 4.119 rad/s. Cold compilation and first-update
latency remain large. The cleaned implementation reproduced every saved result
exactly, and both final artifact packs verified without refitting. No public
model, controller or held-out vehicle trial has been run.

## Next iteration

The next gap is **one public learner with the tested angular structure**.
Integrate the episode-only prompt/delayed command effects and dissipative
angular memory into `fit/predict/update` and bounded `OnlineFit`, rather than
retain a separate experimental prediction path. Preserve immutable revisions,
saved-model reproducibility, analytic command derivatives and the frozen
physical-error benchmark. Measure any fixed-wing loss openly. Then run the
fresh-start Throw workflow and a held-out configuration; those are still
needed before claiming general control or cross-platform accuracy.
