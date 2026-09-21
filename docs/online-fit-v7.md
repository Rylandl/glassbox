# Online v7: broader prior domains

**V6 remains the maintained learner.** V7 improves equal-family velocity/rate error 13.30% against adopted v6, but misses the frozen primary and angular-robustness gates. Fixed-wing aggregate error improves 26.65%; quads worsen 2.47%. Worst-decile errors improve 15.02%, while equal-family orientation error worsens 4.79% and the truth-relative rotation/rate defect worsens 1.74%.

This tests the same generic model and does not improve generality over v6. The unresolved angular behavior, rather than one isolated losing case, is the reason to continue diagnosis before replacing it.

## Frozen change and comparison

V7 replaces typical RMS lengths in the explicit quadratic-head prior with the immutable full supported-motion envelope and maximum absolute normalized issued commands over actual completed-cache windows, including early history. It copies the command envelope to filtered coordinates because the existing filter forms convex combinations of issued commands. Gravity geometry, scalar strength 0.01, initialization, data loss, four-step preconditioned solve, forecast trust rule, budgets and inference stay unchanged. The larger domains substantially strengthen effective regularization; they are not physical admissibility or all-history bounds, and cache eviction can shrink the command envelope.

Protocol `9be72e9` preceded implementation; tested source `3ca9498` preceded the single six-tape run. The primary reference is adopted v6, with exact initial core, inputs, origins, truth, frozen-startup predictions and kinematic predictions. All 3,137 transitions remain, including the truncated quad135 tape. There was no parameter sweep, new data or controller trial.

| Case | Velocity v6 (m/s) | Velocity v7 | Rate v6 (rad/s) | Rate v7 | Orientation v6 (rad) | Orientation v7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| quad-arm-115 | 0.0237478 | 0.0244137 | 0.232947 | 0.236598 | 0.00178017 | 0.00160433 |
| quad-arm-125 | 0.0102457 | 0.0108966 | 0.0810659 | 0.0795135 | 0.000375131 | 0.000375947 |
| quad-arm-135 | 1.17741 | 1.17608 | 0.912604 | 0.977341 | 0.00442763 | 0.00461092 |
| quad-change | 0.0102458 | 0.0108973 | 0.0810741 | 0.0795196 | 0.000375166 | 0.000375976 |
| fixedwing-80 | 0.308985 | 0.210141 | 0.783293 | 0.83351 | 0.0256415 | 0.027117 |
| fixedwing-81 | 1.24428 | 0.603487 | 2.34088 | 1.9309 | 0.108922 | 0.127931 |

Six of 12 primary case/metric cells improve. The primary gate required aggregate<=0.8 and each family<1; the observed ratios are 0.866978 overall, 1.024693 quad and 0.733537 fixed wing. The separate robustness gate required upper-decile<1, orientation<=1 and rotation/rate-defect<1; the observed ratios are 0.849810, 1.047928 and 1.017422. All cases are finite and complete. These frozen verdicts are unchanged.

FW81 velocity RMSE falls 1.24428→0.60349m/s and its maximum 7.70413→2.97598m/s. Its rate maximum also falls 18.22699→13.47365rad/s, yet its orientation RMSE rises 17.45% and maximum 0.60064→0.72945rad. FW80 velocity improves 31.99%, but rate/orientation worsen 6.41%/5.75%. Both fixed-wing velocity and rate errors still exceed kinematic hold. Full per-case tails, adaptation, timing and baseline comparisons remain in the authenticated pack.

## What changed in the causal models

The preceding [v6 trace](online-angular-response.md) reproduced all 450 fixed-wing predictions and updates exactly, recovering 23 pre-assimilation snapshots. V7 saved the same 23 origins prospectively, before each forecast or target reveal, outside timed model calls. Every captured float32 forecast reproduces exactly. The unchanged factor 1/2/4/16/64 diagnostic then reconstructs all heads and their angular increments/derivatives.

At FW81 row 64 the maximum quadratic rate-Jacobian block norm drops 47,280→658, but the signed quadratic pitch increment grows in magnitude, -26.38→-34.28rad/s. Current/delayed-linear increments change +4.18/+2.67→+9.24/+10.50rad/s. Smaller local sensitivity does not imply smaller forcing or less cancellation. Within the quadratic head, motion×motion pitch increment changes -2.813→-0.028rad/s while issued×issued grows -11.597→-25.498rad/s. Similar redistribution occurs at rows 150/157; this is not merely transfer into an unpenalized neural head.

A posthoc comparison evaluates both saved models on the **same v7 physical prior domain and identical retained caches**, eliminating the difference between their own prior definitions:

| Origin | V6 physical prior on common domain | V7 physical prior on common domain |
| --- | ---: | ---: |
| FW80 row 31 | 0.3972 | 0.1755 |
| FW80 row 36 | 1.6608 | 1.6455 |
| FW81 row 64 | 12.2095 | 14.1189 |
| FW81 row 150 | 26.0288 | 46.0272 |
| FW81 row 157 | 28.7360 | 48.5094 |

Thus the stronger prior does **not** reliably suppress physical quadratic curvature at the causal failure states. This does not by itself prove optimizer failure: the models followed different nonlinear online trajectories, and an accepted proposal after four PCG iterations does not establish convergence.

At FW81 row 150 the native rate error improves 2.121→0.494rad/s, while its refined 64 error worsens 1.263→3.708rad/s. V7's 16-to-64 disagreement is 0.00857rad/s. Some native accuracy therefore benefits from numerical error canceling learned-field error. Refinement is diagnostic model output, not plant truth or a proposed solver change. Signed head contributions describe the integrated trajectory; removing a head changes that path and is not tested here.

The next named gap is **causal optimizer effectiveness under the existing physical prior**. Freeze a read-only audit of retained data/prior gradients, four-PCG residual against a bounded more accurate solve of the same local linear system, forecast trust shrink and exact objective gain. Distinguish an inadequately solved update from an objective that permits inaccurate dynamics before choosing another penalty or model change. No further candidate has been fitted.

## Verification and limits

The candidate passes 235 tests and Ruff; all 14 other package source files remain byte-identical to v6. The saved baseline replays 12,768 flight arrays, 40 Dart trajectories and 4,144 gradients exactly. V7 quad update p95 is 27.63–28.17ms against 10ms observations; fixed-wing p95 is 3.96–4.12ms against 50ms. Quad real-time qualification remains false.

The independent NumPy evaluation audit authenticates 400 payloads and nested v6/v4/v2 references; it checks 3,137 paired outcomes, 9,411 journal events, all 23 full causal snapshots, endpoint objectives and all metrics with zero model/optimizer calls. A separate independent integration audit verifies 123 payloads, the 23 diagnostic snapshots, 12,006 stage states and 178,273 numerical assertions with no model calls. An additional independent comparison checks that all 23 paired caches/raw normalizers/scales are byte-identical and reproduces the common-domain prior values. The maintained v6 checkout can verify v7's sealed evaluation without loading a v7 learner; the rejected implementation is retained only in Git. After restoring v6, all 234 tests pass; all 15 package source files are byte-identical to the adopted v6 source.

[The result index](online-fit-v7.json) binds protocol, scientific source, both sealed artifact authorities and supplementary checks. These known tapes, selected causal snapshots and data-derived bounds do not qualify blind generalization, calibrated uncertainty or online closed-loop recovery.
