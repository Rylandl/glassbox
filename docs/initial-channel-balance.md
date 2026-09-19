# Fixed initial-training channel balance

The balanced candidate has the lowest aggregate forecast and command-response
errors among the four learned models on this fresh Crazyflow/Cascade cohort.
Relative to the public-excited reference, weighted forecast error falls **24.7%**
and response error **32.2%**. The angular repair target passes. Three Crazyflow
forecast tail comparisons against the stronger research references fail their
predeclared limits, so the combined mechanism criterion fails. The public recipe
remains unchanged; these research gains and losses guide the next iteration.

[Protocol](harness/initial-channel-balance-v1.json) was frozen at `a14b3a5` before
implementation `a2d64bc`. [Result record](harness/initial-channel-balance-v1-result.json)
anchors the complete evidence and verification. Historical sources and results
are unchanged.

## The intervention

Keep the full quadratic predictor, canonical affine-centered initialization,
original hold-current scales, exact 384 training/256 development windows,
minibatches, random draws and 1,000 Adam updates. Compute each channel's mean
squared normalized residual from one recursive forecast over the initial training
cache. Use its floored inverse, normalized to mean one, as a fixed multiplier
in both training loss and development checkpoint selection. The floor is the
existing `hold_scale_floor ** 2`; it is not an application accuracy tolerance.
There are no platform labels, physical groups, adaptive weights or consumer options.

Only the two candidates are newly fitted. Six exact historical public, quadratic
and anchored reference fits, including their full checkpoints and provenance,
are imported unchanged. All four selected models forecast the same fresh test
queries. The prior seven inspected cohorts are diagnostic evidence only.
Data and update counts match across arms; candidate and anchored control also
match initial arrays and model dimensions. The public architecture is smaller.
Total computation does not match.
The added initial training forecast and full-cache diagnostics are recorded.

## Matched held-out results

Weighted geometric-mean error ratios, lower is better:

| Candidate / reference | Forecast | Command response |
| --- | ---: | ---: |
| Public-excited | 0.75267 | 0.67820 |
| Unchanged anchored | 0.92277 | 0.87839 |
| Unchanged quadratic | 0.96260 | 0.89291 |

All three weighted comparisons, simulator-primary guards and scope guards pass.
Public-relative tail guards also pass. The only required retention failures are
Crazyflow primary 250 ms forecast parent-RMSE p95:

| Signal / reference | Reference p95 | Candidate p95 | Ratio / limit |
| --- | ---: | ---: | ---: |
| Rotation entries / anchored | 0.081070 | 0.144850 | 1.787 / 1.5 |
| Velocity, m/s / quadratic | 0.131721 | 0.234800 | 1.783 / 1.5 |
| Rotation entries / quadratic | 0.068212 | 0.144850 | 2.124 / 1.5 |

Primary 250 ms component RMSE, public / quadratic / anchored / balanced:

| Simulator / quantity | Public | Quadratic | Anchored | Balanced |
| --- | ---: | ---: | ---: | ---: |
| Crazyflow forecast velocity, m/s | 0.178705 | 0.075684 | 0.090045 | 0.124348 |
| Crazyflow forecast rate, rad/s | 0.213321 | 0.414684 | 0.397615 | 0.170441 |
| Crazyflow forecast rotation entries | 0.126274 | 0.038126 | 0.043433 | 0.084666 |
| Crazyflow response velocity, m/s | 0.112342 | 0.033190 | 0.033613 | 0.037695 |
| Crazyflow response rate, rad/s | 0.226081 | 0.303712 | 0.300216 | 0.152198 |
| Crazyflow response rotation entries | 0.102035 | 0.026018 | 0.026937 | 0.038245 |
| Cascade forecast velocity, m/s | 0.160517 | 0.116402 | 0.118997 | 0.128029 |
| Cascade forecast rate, rad/s | 0.104266 | 0.095119 | 0.092385 | 0.107033 |
| Cascade forecast rotation entries | 0.017036 | 0.010822 | 0.010549 | 0.011743 |
| Cascade response velocity, m/s | 0.049180 | 0.037415 | 0.037084 | 0.037351 |
| Cascade response rate, rad/s | 0.037481 | 0.035537 | 0.034500 | 0.038597 |
| Cascade response rotation entries | 0.006995 | 0.003713 | 0.003640 | 0.003786 |

The targeted Crazyflow angular forecast mean/p95 ratios versus anchored are
**0.42866 / 0.45198**; response mean/p95 ratios are **0.50696 / 0.50851**.
The candidate improves all six Crazyflow primary mean errors relative to public.
Cascade primary angular forecast/response means worsen about 2.7%/3.0% versus
public, within the frozen broad limits. Physical rotation errors are raw matrix
entries, not angles; no projection improves the scores.

![Matched held-out means and parent tails](../artifacts/2026-09-18/initial-channel-balance-v1-figures/heldout-250ms.png)

## What training actually achieved

Crazyflow selects **step zero**, preserving the exact anchored initializer.
Every recorded later full-training weighted loss exceeds its initial value:
**0.00328876** initially, **0.01138641** at 100, **0.00442898** at 900 and
**0.00498218** at 1,000. Development loss likewise rises from **0.00400372**
to **0.00632106** at 1,000. Angular development RMSE rises
**0.188526→0.401337 rad/s**. This is failure to improve the new training objective
at recorded checkpoints, so development overfitting alone does not explain it.
It does not establish which optimizer mechanism causes the failure.

The old unweighted development loss would favor step 900 on this same weighted
trajectory. That observation is descriptive: it is neither another training run
nor a tested alternative selection. No intermediate checkpoint receives held-out
forecasts or response queries. Crazyflow's held-out gains come from preserving
initialization, not from a newly improved optimized checkpoint.

Cascade selects **step 1,000** and reduces weighted training loss
**0.01978948→0.00590152** and development loss **0.02894970→0.01949018**.
No channel activates the denominator floor in either simulator. Initial weighted
contributions give each coordinate one vote: the physical groups still have
3:3:9 coordinate counts. This is neither equal physical-task weighting nor
uncertainty estimation.

![Recorded optimization and angular errors](../artifacts/2026-09-18/initial-channel-balance-v1-figures/training-diagnosis.png)

## Scope and reproduction

Crazyflow completes **64/84** test parents, with 20 altitude failures. Primary
250 ms truth is available for **413/480** forecast and **664/768** response
queries. Cascade completes **84/84**, with all **480/480** and **576/576** queries.
Every eligible prediction is finite. All models share the same truth masks;
these are conditional Crazyflow errors, not complete-cohort accuracy. More
complete parents than a prior cohort do not establish learner improvement.
The original incomplete excited training collection remains unchanged.

The frozen bundle is `artifacts/2026-09-18/initial-channel-balance-v1`, with root
SHA256 `2f2b87157fae730e08547f7f4730b71f2c33e94d9aa67bb1fbd33789303435e4`.
Use the pinned Dart Python 3.12 environment, CPU JAX x64 and Cascade checkout
specified by the protocol. The module
`glassbox.experimental.initial_channel_balance_experiment` exposes `generate`,
`fit --arm candidate`, `evaluate`, `finalize` and `replay`; replay requires the
external bundle SHA. The saved stage supervisor and logs record exact commands,
implementation commit and 7,200-second bounds.

Both full replays reproduce 4,032 prediction queries and 428,400 metric rows.
All 784 focused tests and 32 alteration challenges pass. Independent reductions
verify the reported physical errors and weighting arithmetic. Verification details,
the preserved auxiliary audit correction and artifact hashes live in the result
record. Saved predictions and checkpoints replay without fitting or ridge
re-solving. Optimizer updates are not independently rerun. Independent arithmetic
checks complement production replay; hashes alone do not establish semantic truth.
No public contract, synthetic-cap, derivative-fidelity, uncertainty, update or
controller qualification follows from this experiment.
