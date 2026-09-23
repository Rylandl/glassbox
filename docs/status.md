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
| Offline and Dart | `fit/predict/update` use the new angular structure and lifecycle tests pass. No matched offline accuracy fit or live Dart/Throw controller trial has been run for this revision; preceding compact-model results are [historical](nonlinear-temporal.md). |
| Scope | One fixed-wing airframe across two recordings and known quad configurations/conditions; a paired quad flight was gentle. Long-horizon fidelity, fresh-start real-time operation, calibrated envelopes, held-out configurations and live Throw recovery remain open. |

## Latest result

The [public rate-memory integration](public-rate-memory.md) put the experimental
angular structure into one public model and replaced sample-count command lag
with a fixed physical-time lag. The fixed-wing 250 ms rate errors improved from
the experimental wrapper's 0.509/0.634 to 0.345/0.597 rad/s; quad rate results
are essentially unchanged. The cleaned public implementation reproduced every
saved benchmark forecast and response exactly. Both final packs verified from
saved data without refitting. The hard arm-135 and first underexcited response
remain substantial errors.

## Next iteration

The next gap is **fresh-start consumer and configuration qualification**. Run a
Throw workflow from an empty episode with startup data and cold compilation
time counted; determine whether the measured first-update delay is acceptable
or needs architectural removal. Test a genuinely held-out configuration with
the frozen physical-error and response protocol. Keep live control success,
model residuals and derivative fidelity as separate claims. Do not infer
generality from the known eight recordings or the gentle paired flight.
