# Full-cache gradients: stronger fitting exposes a generalization gap

The [frozen experiment](harness/full-cache-gradient-v1.json) rejects replacing
the retained balanced model with this candidate. On the same fresh recordings,
aggregate forecast errors rise **9.90%** and command-response errors **13.34%**.
Training loss falls more than 90% in both simulators. Better optimization has
not produced a better general dynamics model.

The balanced model remains the strongest aggregate research result; the public
recipe remains `generic-memory-v3-prototype`. The
[result record](harness/full-cache-gradient-v1-result.json) anchors the evidence.

## One mechanism, explicit additional compute

Each Adam gradient uses all 384 saved training windows exactly once, in saved
order, instead of sampling 64 windows with replacement. The architecture,
actual initializer, fixed channel weights, normalization, 256 development
windows, clipping, Adam formulas, acceptance ladder and checkpoint selector
are unchanged. Two new candidates are fitted; six saved public, balanced and
safeguarded references are imported unchanged. All models predict the same
fresh cohort of 168 planned test parents.

Each fit completes 1,000 gradients and 384,000 gradient-window visits, versus
64,000 visits for the safeguarded control. Acceptance-objective calls are
5,392 for Crazyflow and 2,477 for Cascade. Equal proposal counts do not imply
equal compute. The candidate uses no optimizer sampling RNG. Initialization
randomness remains fixed. No fit, seed, checkpoint rule or criterion was retried.

## Physical accuracy

Ratios below one favor full-cache gradients. Intervals are descriptive 95%
paired-parent bootstrap ranges, conditional on available truth.

| Reference | Forecast ratio [interval] | Response ratio [interval] |
| --- | ---: | ---: |
| Balanced | 1.09896 [1.08091, 1.11597] | 1.13337 [1.11200, 1.15431] |
| Safeguarded control | 1.02532 [1.01057, 1.04029] | 0.99877 [0.98166, 1.01701] |
| Public-excited | 0.83109 [0.81127, 0.84910] | 0.78677 [0.76885, 0.80278] |

Primary 250 ms component RMSE, balanced → full-cache:

| Simulator / target | Velocity, m/s | Body rate, rad/s | Rotation entries, unitless |
| --- | ---: | ---: | ---: |
| Crazyflow forecast | 0.11725 → 0.09474 | 0.14626 → 0.23795 | 0.07633 → 0.06138 |
| Crazyflow response | 0.03938 → 0.03610 | 0.16239 → 0.18974 | 0.03929 → 0.03243 |
| Cascade forecast | 0.13485 → 0.15047 | 0.10790 → 0.12828 | 0.01185 → 0.01395 |
| Cascade response | 0.03882 → 0.04378 | 0.04255 → 0.05285 | 0.00404 → 0.00449 |

![Matched physical errors](../artifacts/2026-09-19/full-cache-gradient-v1-figures/heldout-250ms.png)

The public-relative comparison passes. The balanced and safeguarded-control
comparisons fail; the latter requires a meaningful 5% response improvement,
whereas the observed change is only 0.12%. Crazyflow angular retention fails
for forecast mean/p95 ratios **1.62686/1.72181** and response mean **1.16845**.
The response p95 ratio **1.04637** passes the 1.05 retention limit. The separate
optimization diagnostic fails but does not veto the physical decision.

Crazyflow completes **62/84** test parents, with 22 altitude-condition failures.
Primary 250 ms truth is **423/480** forecast and **704/768** response queries.
Cascade completes **84/84**, with all **480/480** forecast and **576/576** response
queries. All five predictors are finite on every eligible query. Comparisons
use the same truth masks; differences from earlier cohorts' completion are
not model improvements. This cohort has no quadratic reference or fresh
quadratic-retention claim.

## Why this changes the next investigation

Both models select checkpoint 1,000. Crazyflow accepts 990 proposals and
rejects 10; Cascade accepts 995 and rejects 5. Training loss falls
**90.54%/92.60%** and development loss falls **4.08%/29.05%**, respectively,
from the same initialization. Cascade's selected development loss, **0.020541**,
still exceeds the balanced model's **0.019490**.

Crazyflow's development rate RMSE increases **0.188526→0.256304 rad/s** while
velocity and rotation improve. The weighted development objective masks this
tradeoff: the rate contribution increases but other channels offset it.
Training rate error also improves, so an inability to optimize the training
objective is not a sufficient explanation for the remaining error.

The held-out angular harm is broad: all 24 primary conditions and 41/42
conditions overall worsen at 250 ms factual rate against both balanced and
safeguarded models. Crazyflow's factual rate parent-p95 rises
**0.231795→0.399108 rad/s**. Cascade's extreme 18 m/s calm, heading-zero condition
has response rate error **0.054739→0.198550 rad/s** against balanced, with
complete truth for that condition. These inspected results are diagnostic
evidence for a future experiment, not untouched confirmation for it.

![Training and development losses](../artifacts/2026-09-19/full-cache-gradient-v1-figures/training-diagnosis.png)

The next named gap is training-data support and generalization. The recommended
bounded intervention is 384→1,536 training windows from the existing admitted
recordings, preserving the original cache prefix. Freeze the experiment first. Keep the development cache and role boundaries unchanged,
declare any extra data and compute, and use a fresh common confirmation cohort.
This evidence does not yet distinguish inadequate support from excessive model
flexibility or a poor objective/selection tradeoff; it does not justify further
optimizer tuning as the next mechanism merely because training loss can decrease further.

The separate cache audit finds 12,042 unused legal training origins in Crazyflow
and 2,928 in Cascade. The current cache covers 50.79%/44.14% of unique eligible
forecast target timesteps. All 72 training parents appear, with five or six
windows each. A larger cache changes data-derived normalization, initialization,
hold scales and initial-channel weights naturally; an unchanged recipe formula
would not imply identical numerical values. Training covers all 24 primary
conditions but none of the 18 shifted conditions, and 48/72 Crazyflow training
parents are truncated. Additional overlapping windows do not create independent
recordings or repair those omissions.

## Evidence and reproduction

Protocol `13cc4aa` was committed before implementation. Implementation `55e8a17`
passed **1,013 focused tests**, Ruff and formatting before collection or fitting.
The sealed root contains **14,758** payloads and has SHA-256
`42e45170a3fc39c33bfca4bad979dacccf9c17c9ef583c2929211943484afa7d`.
Independent NumPy audits verify all **1,050** endpoint groups, actual initial
arrays, weights, cached data, reference identities, **2,000** ordered gradient
attempts and **768,000** known gradient-window visits. They also reduce 154
saved training/development diagnostic sets, including 44 new-candidate sets.

Both canonical replays pass: 4,032 prediction queries, 428,400 metric rows,
88 checkpoint snapshots and 7,869 candidate acceptance-objective calls reproduce
exactly. All 36 alteration challenges pass on their first attempt: two raw
integrity checks and 34 coherently resealed semantic changes. These include
false gradient coverage/dtype/work accounting and acceptance-loss changes that
pass scalar consistency but fail fresh objective replay. The original sealed
bundle remains unchanged. Details are recorded in the result record.
Replay executes simulator truth, selected-model predictions, checkpoint
forecasts and saved proposal acceptance objectives without refitting,
re-solving initialization, or recomputing gradients/Adam. Gradient and moment
fingerprints are observed execution/continuity evidence, not an independent
recomputation of those arrays. Bootstrap intervals are saved descriptive values;
the independent reducer verifies point reductions and parent-draw identity.

Use the pinned Dart Python environment, CPU x64 JAX and Cascade checkout
recorded by the bundle. From the repository, for each simulator:

```sh
SCIPY_ARRAY_API=1 JAX_ENABLE_X64=1 \
PYTHONPATH=src:/private/tmp/glassbox-cascade-e8f6ba6/src \
/Users/ryland/autonomy/dart/.venv/bin/python \
  -m glassbox.experimental.full_cache_gradient_experiment replay crazyflow \
  --output artifacts/2026-09-19/full-cache-gradient-v1 \
  --expected-bundle-sha 42e45170a3fc39c33bfca4bad979dacccf9c17c9ef583c2929211943484afa7d
```

Replace `crazyflow` with `cascade` for the second replay. No public, synthetic,
uncertainty, derivative-fidelity, update or controller qualification is implied.
