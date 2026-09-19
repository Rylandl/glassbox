# Safeguarded Adam: optimization improves, physical accuracy does not

The [frozen experiment](harness/safeguarded-adam-v1.json) rejects this specific
optimizer change. Relative to the retained balanced model, weighted held-out
forecast errors increase **6.29%** and command-response errors **15.37%**.
The balanced model remains the strongest aggregate research result. The public
recipe remains `generic-memory-v3-prototype`.

The mechanism makes real training progress from Crazyflow's useful initializer,
but that does not establish a better dynamics model. No threshold or checkpoint
selection was changed after seeing these results. The
[result record](harness/safeguarded-adam-v1-result.json) anchors the complete
evidence and verification scope.

## What changed

Every existing clipped 64-window Adam proposal receives a full-training weighted
loss check. A fixed eight-step ladder tries scales 1 through 1/128, accepting
the first finite strict decrease. Otherwise the parameters stay unchanged.
Moments and the attempt counter advance on every finite proposal, including
rejection. There are 1,000 proposals per simulator and at most 8,001 additional
full-training objective calls, with a 7,200-second internal fitting bound and
separate 14,400-second stage watchdog.

The architecture, exact initializer, fixed initial-channel weights, normalization,
384/256 training/development windows, Adam formulas, minibatch draws and original
development selector stay fixed. Each new initial state and weight witness is
byte-exactly equal to its balanced reference. Two candidates are fitted; six
historical reference fits are imported unchanged. A common fresh cohort supplies
168 test parents. Extra full-cache loss evaluations are an explicit compute
intervention: proposal counts match, FLOPs and accepted updates do not.

## Matched physical results

All values below are from the same fresh cohort. Ratios below one favor the
candidate. Intervals are descriptive paired-parent bootstrap percentile ranges,
conditional on available truth; they are not application-adequacy thresholds.

| Reference | Forecast ratio [95% interval] | Response ratio [95% interval] |
| --- | ---: | ---: |
| Balanced | 1.06289 [1.05133, 1.07490] | 1.15371 [1.13844, 1.16623] |
| Public-excited | 0.79796 [0.78129, 0.81338] | 0.78958 [0.77033, 0.80460] |
| Quadratic | 1.01890 [1.00281, 1.03652] | 1.04030 [1.02077, 1.05885] |

Primary 250 ms RMSE, balanced → safeguarded:

| Simulator / target | Velocity, m/s | Body rate, rad/s | Rotation entries, unitless |
| --- | ---: | ---: | ---: |
| Crazyflow forecast | 0.10959 → 0.10989 | 0.15447 → 0.17116 | 0.07417 → 0.07238 |
| Crazyflow response | 0.03886 → 0.03866 | 0.16598 → 0.16801 | 0.03756 → 0.03527 |
| Cascade forecast | 0.13334 → 0.15885 | 0.10421 → 0.13498 | 0.01197 → 0.01419 |
| Cascade response | 0.03665 → 0.04462 | 0.04182 → 0.06003 | 0.00379 → 0.00447 |

![Matched forecast and response errors](../artifacts/2026-09-18/safeguarded-adam-v1-figures/heldout-250ms.png)

All three required comparison gates fail. Balanced-relative aggregate forecast,
response and simulator-primary guards fail, although its broad 1.5 tail guards
pass. Public-relative aggregate gains pass, but Cascade rate parent-p95 ratios
are **1.56569** forecast and **1.54965** response. Quadratic-relative aggregates
pass, while scope, simulator-primary and four tail guards fail. The tighter
Crazyflow angular retention requirement fails for forecast mean/p95 ratios
**1.10808/1.19136**; response ratios **1.01221/0.98394** pass. These are localized
diagnostics of this optimizer, not reasons to reject the generic approach.

Crazyflow completes **68/84** test parents; 16 fail the altitude condition.
Primary 250 ms truth is **442/480** forecast and **720/768** response queries.
Cascade completes **84/84**, with all **480/480** forecast and **576/576** response
queries. Every eligible prediction is finite for all five models, and every
comparison uses matched truth masks. Completion differs from the preceding
cohort and is not a learner gain. Original incomplete training collection and
comparator provenance remain visible.

## What the optimizer evidence establishes

| Simulator | Selected attempt | Accepted / rejected | Full-training objective calls | Selected training / initial | Selected development / initial |
| --- | ---: | ---: | ---: | ---: | ---: |
| Crazyflow | 1,000 | 717 / 283 | 7,529 | 0.544846 | 0.954162 |
| Cascade | 900 | 931 / 69 | 4,427 | 0.506511 | 0.949403 |

Neither simulator accepts a full-scale proposal. Crazyflow accepts 348 proposals
at 1/128 and 269 at 1/64. Its training objective falls **45.5%**, and all three
training physical groups improve. Development loss falls only **4.58%**, narrowly
missing the separately frozen 5% diagnostic margin. Development 250 ms rate RMSE
barely changes, **0.188526→0.187567 rad/s**, while the held-out rate error grows.
The negative physical result is much broader than this narrow diagnostic miss.

Cascade's selected training/development losses are **0.010024/0.027485**, compared
with the balanced model's **0.005902/0.019490**. Its development rate error grows
**0.143452→0.156553 rad/s** from initialization, despite a lower weighted objective.
The candidate's training-and-capture times are 114.65 seconds for Crazyflow and
11.84 seconds for Cascade; diagnostics take another 5.02/2.42 seconds. These
measurements exclude stage preflight, and are not an equal-compute comparison.

![Training and development trajectories](../artifacts/2026-09-18/safeguarded-adam-v1-figures/training-diagnosis.png)

Small accepted steps and rejections motivate a proposal-direction test. They do
not prove minibatch noise: curvature or accumulated Adam moments could also
explain them. Crazyflow already shows a substantial training/development gap,
so a better optimizer may still fail to improve generalization.

The next named experiment should use all 384 training windows for each gradient
while preserving the safeguard, objective, initializer and selection. Compare
against both this optimizer control and the retained balanced model on another
fresh cohort. Freeze the gradient arithmetic, accounting, budget and physical
criteria first. If better training progress again fails to transfer, shift the
next investigation toward data support or model generalization instead of
extending optimization blindly.

## Evidence and reproduction

The protocol was committed at `6ba6336` before implementation; implementation
`c750b23` passed **896 focused checks** and Ruff before any real collection or
fitting. The sealed root contains **14,760** payloads and has SHA-256
`0c881cac071540b89c72d6343d7da6394900a0735bc3d2549fd8aea9226f9319`.

Independent NumPy checks reduce all **1,050 endpoint groups**, all **2,000**
observed proposal attempts and **44** new-candidate training/development diagnostic sets.
Actual initial arrays, imported references, caches, norms and weights match.
This audit checks observed gradient/moment fingerprints and continuity; it does
not independently recompute gradients or Adam. Canonical replay separately
reexecutes damped proposals' full-training losses, checkpoint forecasts, simulator
truth and selected-model predictions without fitting or re-solving initialization.
The result record gives the final replay and alteration-challenge outcomes,
including preserved auxiliary attempts and their limits.

Use the pinned Dart Python environment, CPU x64 JAX and Cascade source checkout
recorded by the bundle. From the repository root, replay either simulator:

```sh
SCIPY_ARRAY_API=1 JAX_ENABLE_X64=1 \
PYTHONPATH=src:/private/tmp/glassbox-cascade-e8f6ba6/src \
/Users/ryland/autonomy/dart/.venv/bin/python \
  -m glassbox.experimental.safeguarded_adam_experiment replay crazyflow \
  --output artifacts/2026-09-18/safeguarded-adam-v1 \
  --expected-bundle-sha 0c881cac071540b89c72d6343d7da6394900a0735bc3d2549fd8aea9226f9319
```

Replace `crazyflow` with `cascade` for the second replay. Historical reference
bundles remain required at their authenticated locations. The current result
does not qualify public adoption, uncertainty, physical derivatives, live
updates, controller use or arbitrary-system readiness.
