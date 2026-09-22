# First-order sensitivity: useful tradeoff, incomplete stability surrogate

One frozen-strength experiment completed **3,137 updates per arm** on all six
recordings. Adding a first-derivative penalty to the fast readout lowers aggregate
250 ms velocity/rate error **8.65%**, raises one-step error **3.30%**, and costs
**21.31% more update time** versus the previous curvature-only readout. The
candidate remains **18.45× faster** than the full learner on this CPU.

The targeted fixed-wing angular failure remains. More importantly, the penalty
greatly reduces instantaneous angular feedback while barely changing the full
recurrence gain. This separates a useful local sensitivity surrogate from the
coupled dynamics we actually need to control. Production remains unchanged;
this result is retained as an architectural tradeoff, not adopted as a completed
solution. The [index](readout-sensitivity.json) preserves all scores and artifacts.

## Frozen change and data budget

The [protocol](harness/readout-sensitivity-v1.json) was committed as `c9dc92e`
before implementation; tested fitting code was committed as `942a5fb` before the
single full-roster run. No strength sweep or second candidate fit followed.

Three arms start from separate, exactly equal fresh `OnlineFit(prefix)`
constructions: the unchanged full learner, unchanged curvature readout, and the
new sensitivity readout. A prediction-only copy of the initial model is also
scored. Every episode resets learned state. The readouts freeze only values
learned from that episode's prefix: no pretraining, learned fleet prior, reused
normalization or vehicle metadata. Initial acquisition remains 0.75 s of source
data, with scoring beginning at tape time 1.25 s for the quad recordings and
0.75 s for fixed wings. Prefix acquisition, fitting and compilation are not free.

The new term uses D = d(phi)/d(xi), where xi contains independent **supported,
normalized body velocity and body rate** at the measured midpoint and each lag.
It includes the minus-current derivative in every lag-difference feature, the
quadratic head, and the frozen nonlinear head with temporal compression.
Gravity direction, commands, command filter and accumulator values are held fixed
in this derivative. Coordinates are after the existing saturation; this avoids
weakening the penalty merely because a recorded coordinate is saturated.

Starting with S=0, accumulate S += DDᵀ from each causally revealed midpoint, then
solve `(G + diag(curvature) + 0.01 S) M = Q`. The original data/ridge statistics,
curvature penalty, targets, features and physical integrator remain unchanged.
Analytic derivatives and the shared Cholesky solve cover all six acceleration
outputs. This is a generic input-perturbation penalty at normalized scale 0.1,
not a claim about sensor noise or Bayesian uncertainty. It penalizes stabilizing
and destabilizing gains alike and remains constant-strength relative to mean data
error. It does not assert consistency or nonlinear stability.

## Full-roster results

Ratios are candidate/control, equal cases within each family and equal families
geometrically. Primary error combines velocity-vector and body-rate-vector RMSE.

| Compared with previous fast readout | Quad | Fixed wing | Equal-family aggregate |
| --- | ---: | ---: | ---: |
| One-step primary | 1.0334 | 1.0325 | **1.0330** |
| 250 ms velocity | 0.8078 | 0.9328 | **0.8681** |
| 250 ms body rate | 1.0000 | 0.9242 | **0.9613** |
| 250 ms primary | 0.8988 | 0.9285 | **0.9135** |
| Warm median update time | 1.2896 | 1.1412 | **1.2131** |

The quad forecast gain is principally velocity accuracy; its aggregate angular
error is essentially unchanged and its one-step rate error rises 22.2%.
Individual conditions differ:

| Recording | 250 ms velocity, old → new (m/s) | 250 ms body rate, old → new (rad/s) |
| --- | ---: | ---: |
| fixedwing-80 | 1.328 → **1.209** | 15.065 → **12.249** |
| fixedwing-81 | 7.669 → **7.330** | 39.984 → **42.002** |
| quad-arm-115 | 0.725 → **0.538** | 5.526 → **4.630** |
| quad-arm-125 | 0.433 → **0.345** | 2.304 → **2.627** |
| quad-arm-135 | 5.327 → **4.817** | 69.421 → **63.759** |
| quad-change | 0.433 → **0.345** | 2.304 → **2.627** |

Orientation is retained separately: quad-115 worsens from 0.398 to 0.440 rad and
the truncated quad-135 from 2.216 to 2.634 rad at 250 ms, while the other four
conditions improve. The truncated tape supplies only 62 updates. Quad-change and
quad-125 share their early motion; these are not six independent generalization
trials. First/late update and pre/post-change scores remain in the index.

![Fixed-wing forecast error versus horizon](/Users/ryland/autonomy/glassbox/artifacts/readout-sensitivity-v1/horizon-errors.png)

Warm complete updates, including derivative construction, Gram accumulation,
cache management, solve and model snapshot:

| Model | Quad median range | Fixed-wing median range |
| --- | ---: | ---: |
| Full learner | 33.34–33.45 ms | 7.503–7.519 ms |
| Previous fast readout | 1.013–1.018 ms | 0.493–0.494 ms |
| Sensitivity readout | **1.292–1.331 ms** | **0.561–0.565 ms** |

First-use candidate updates take 0.223 s / 0.285 s in the first fixed-wing/quad
cases, including compilation; these are excluded only from warmed timing.
Construction, first predictions, every latency and total update time are saved.
The paired run used one CPU process with rotating arm order. This is not a live
scheduler or portable-hardware result.

## Why the local improvement does not solve the recurrence

A [declared no-fit follow-up](harness/readout-sensitivity-feedback-v1.json)
uses all 14 saved origins and five measured states per origin, with readout
coefficients fixed at each origin. The first audit measured instantaneous angular
feedback. After observing its large reduction, a separately committed addition
measured the full augmented recurrence using the earlier tested tangent Jacobian.
Both stages are explicitly posthoc and add no candidate fitting.

| Median local diagnostic, old → new | Fixedwing-80 | Fixedwing-81 |
| --- | ---: | ---: |
| Maximum real eigenvalue of d(angular acceleration)/d(body rate), /s | **13.04 → 4.03** | **13.59 → 3.24** |
| Full 50-dimensional recurrence spectral radius | **3.411 → 3.376** | **3.342 → 3.390** |
| Physical 9-dimensional partial radius, latent variables fixed | **3.573 → 3.526** | **3.685 → 3.703** |
| Velocity/rate 6-dimensional partial radius, attitude also fixed | **2.115 → 1.971** | **2.338 → 2.045** |

Instantaneous angular gain falls about 69% / 76%, yet the full local amplification
barely moves. The physical-state block also stays high with history held fixed.
Therefore the prior linear-lag ablation does **not** justify blaming history alone
or deleting it. Coupling through the physical state, including attitude, remains
important. Local radii and partial-block interventions are diagnostics, not
global stability certificates or proof of a unique causal coefficient block.

The tested penalty perturbs six body-motion feature coordinates, **not attitude**.
A physical attitude perturbation changes body-relative velocity and gravity
direction together; these directions are not equivalent to independent changes
of the six penalized coordinates. The complete physical evolution also rotates
body force into world acceleration. Shrinking the angular-rate self-derivative
is therefore an incomplete surrogate for recursive stability.

The fixed-wing error remains broad enough to matter: candidate 250 ms angular
error improves at 8/14 origins for recording 80 and 5/14 for recording 81 (initial
queries are equal). Median per-origin angular error actually rises from 7.61 to
9.17 rad/s and from 16.85 to 17.95 rad/s. Aggregate improvement in recording 80
comes partly from reducing the largest errors; it is not uniform recovery.

## Decision and next iteration

Versus the full learner, aggregate one-step/250 ms primary ratios are
**0.17085 / 1.07470**, and update time is **0.05421**. Frozen one-step, aggregate
250 ms, family velocity and speed flags pass; the **fixed-wing primary and rate
flags fail** at 1.913 and 3.621 respectively. Every prediction is finite, which
does not make the large angular trajectories useful.

Retain this as evidence of a modest accuracy/runtime tradeoff. Do not promote it
to production as a resolved fast learner, and do not conclude that sensitivity
regularization cannot work from one strength. The immediate priority is the
specific gap exposed by the derivative measurements, not a blind strength sweep
or an automatic veto for each individual regression.

**Next: first-order regularization under physically consistent state
perturbations, including an SO(3) attitude tangent.** Freeze the perturbation
scales, coupled feature changes, causal probe domain, complete-update cost and
comparison before fitting. Compare against both saved fast readouts and the
full learner. Preserve one generic procedure, learned readout freedom, and the
same-episode cold start. With a frozen feature map, body-acceleration derivatives
under state perturbations remain linear in the readout; any treatment of the
world-frame force rotation must state its output coupling explicitly. A local
physical derivative penalty still would not prove full-recursion stability.

The matched quad at 50 ms remains a separate missing fixture. No new simulator,
controller, noise-calibration or held-out-configuration claim is made here.

## Verification and retained evidence

Five focused tests pass: analytic derivatives against JAX and finite differences
at 3/4 commands and 2/10 lags; independent augmented least-squares solves; and
causal frozen-feature updates for both sample intervals. The full learner and
curvature control reproduce **all previous one-step/conditional predictions,
initial/final arrays and curvature solves exactly**. Candidate design, targets
and curvature penalties are also exactly unchanged.

Independent NumPy calculus reconstructs every derivative penalty and all retained
statistics. All **3,137 candidate solves** pass componentwise normal-equation
backward error at most **3.22e-15**, and final regularized systems are positive
definite. Metrics are recomputed from saved predictions without fitting.
Eight actual-state physical rate derivative comparisons agree with JAX autodiff
to maximum absolute difference **4.55e-13**. The recurrence audit reuses the
previously tested full tangent derivative implementation.

There was one fitting run and no failed or repeated fitting attempts: **3,137
candidate + 3,137 curvature control + 3,137 full-learner updates**. Both posthoc
audits load saved coefficients only. Their manifests, runtime/source bindings,
all reports and test logs are in the index. Experimental source remains in Git
through `17e4921` and is removed from the maintained tree after this report.
