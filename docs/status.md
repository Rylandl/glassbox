# Status: gap against the charter

Updated 2026-09-18. **The generic approach is adopted as the development
baseline.** Generality and its four-corpus advantage justify the documented
ARP deficit; this policy decision does not rewrite historical gates. The
adopted learner remains `generic-memory-v3-prototype`: `glassbox.fit` returns
`LearnedDynamics` with `predict` and `update`, without consumer tuning options.
The paired-response learner and anticipatory controller remain isolated
research implementations. **The latest oracle controller qualifies: all sixteen trajectories and
5,088 optimizer decisions/forecasts replay exactly, and four rehashed
alterations are rejected.**

Read [the charter](charter.md) first. This page records the current gaps and
next iteration; git and frozen result records retain experiment history.

| Criterion | Current | Target |
| --- | --- | --- |
| One recipe | **Met.** The public API and fit/evaluate commands use the single adopted generic recipe, with no model-selection options. | One generic learner and consumer contract. |
| Accuracy | **Adopted with a known tradeoff.** Four of five corpora beat structured comparators on both metrics; ARP loses both. In a separate simulator study, paired-response supervision improves aggregate response error 65.1% and factual error 31.4% against the same-data, same-budget forecast-only learner. Public integration and arbitrary-system transfer remain unproved. | Broad competitive forecasts from one platform-independent recipe; task sufficiency measured separately. |
| Capability | **Met.** All 27 frozen synthetic cases pass; saved models replay and reject alteration. These are regression guards, not platform readiness. | Every synthetic absolute cap passes. |
| Control | **Not met for the generic learner.** The latest anticipatory oracle controller reports 2,248/2,248 qualifying samples and all eight trials passing the 95% requirement, versus 912/2,248 (40.57%) and zero passing trials for its fresh baseline. Independent replay and integrity checks pass. The learned predictor has not yet been tested with this controller. | The generic learner controls Cascade within the declared tracking requirement. |
| Live improvement | **Not met.** Last live-v3 swaps at intervals 140/220 increase position error from 0.80/0.98 m to 36.1/11.8 m. Not remeasured. | Bounded refits and swaps that do not worsen tracking. |
| Evidence | **Not met.** The 85–95% coverage band still fails on ARP, the reserved control recording and shifted synthetic regimes. Borrowed constant spread is not calibrated uncertainty for a new predictor. | Measured coverage in the declared band, consumed usefully by control. |
| Lean | **Not met.** Structured dynamics, fitting, belief code and research scripts remain. | Learner, harness, telemetry adapters and controller only. |

## Current forecast evidence

Whole recordings are held out and both arms forecast identical rows and
commands. The table gives generic / best structured endpoint component RMSE
at the recipe's approximately 250 ms horizon. The best structured comparator
can differ by metric. One generic recipe is fitted separately to each system;
these are not shared weights transferred unseen to five systems.

| Corpus | Velocity, m/s | Body rate, rad/s |
| --- | --- | --- |
| nanodrone | 0.136 / 0.179 | 0.543 / 0.597 |
| x8 | 0.222 / 0.287 | 0.132 / 0.187 |
| idf | 0.158 / 0.554 | 0.122 / 0.174 |
| epfl | 0.146 / 0.526 | 0.070 / 0.217 |
| arp | **0.176 / 0.174** | **0.715 / 0.285** |

ARP also loses to hold-current (0.149 m/s, 0.362 rad/s). Its body-rate error
is about 2.5 times the structured result. On the historical, different
whole-prefix percentile statistic, X8 body rate remains **0.888 rad/s versus
the 0.764 ceiling**. Those ceilings were descriptive forecast statistics, not
application requirements; IDF and EPFL have no declared ceiling. These results
support adoption with visible losses, not arbitrary-system readiness or a
permanent four-of-five promotion rule.

[Intervention response v1](harness/intervention-response-v1.json), isolated at
`7ea4674`, establishes a stronger simulator response-learning mechanism.
Eighty independent parents supply 3,920 matched-history branches, split by
parent: 40 train, 12 development, 12 ordinary test and 16 shifted test.
The predictor receives observed signals and applied commands; only the
evaluator/generator accesses hidden simulator state. Physical state cloning
is not an assumed consumer capability.

With architecture, training data, initialization, batch draws and 6,000-step
budget fixed, paired supervision improves all twelve response scores and all
six factual scores. Response geometric-mean ratio is **0.34899**, worst ratio
0.57638, with parent-bootstrap 95% interval **[0.33116, 0.36895]**. Factual
ratio is **0.68608**, worst ratio 0.75047, interval **[0.66703, 0.70556]**.
The selected forecast-only/paired checkpoints are 5,700/5,900; normalized
development response loss is 0.37793/0.05732 and factual loss 0.05660/0.02938.
These intervals concern parents within this aircraft population, not systems.

Earlier ablations separate additional computation from the new supervision:
capped-to-full-data 1,000-step fits improve response/factual error 2.3%/7.0%,
and extending that trajectory to 6,000 steps improves them 38.8%/28.1%.
The first comparison bundles data use, weighting, minibatching and development
budget; it does not isolate those effects. The final 65.1%/31.4% improvement
comes from paired loss and its checkpoint criterion at the same 6,000-step
budget.

Some response signs remain wrong. First-step aileron velocity reversals fall
64→9 and rotation reversals 170→27 among 390 nonweak probes. Throttle body-rate
reversals at 150 ms rise 55→163/392 and rotation reversals at 250 ms rise
25→176/392, despite lower absolute errors. All **40,040 forecasts are finite
and replay exactly**; all parents/branches, selected objectives, metrics and
2,000 bootstrap draws replay, and five rehashed alterations are rejected.
Intermediate optimizer updates are not independently re-solved. The
[result record](harness/intervention-response-v1-result.json) anchors this
qualification; it promotes no public recipe, envelope or controller.

## Current controller evidence

[Controller anticipation v1](harness/controller-anticipation-v1.json) compares
two oracle arms on eight fresh matched seeds, 110–117. Both forecast only
five steps/250 ms and use the same float64 L-BFGS-B solver, 64-iteration budget,
bounds and remaining costs. The candidate replaces each position residual
with position error plus **1.5 seconds times velocity error** in the stage
and terminal terms. The coefficient anticipates motion in the cost; it does
not extend the prediction horizon. Each arm drives its own trajectory and
warm starts.

| Verified controller comparison | Baseline oracle | Anticipatory oracle |
| --- | --- | --- |
| Simultaneous lateral/altitude tolerance samples | 912/2,248 (40.57%) | **2,248/2,248 (100%)** |
| Trials meeting the 95% requirement | 0/8 | **8/8** |
| Backend failures / fallbacks / bound violations | 0 / 0 / 0 | 0 / 0 / 0 |
| Returned residuals at or below 0.002 | 2,544/2,544 | 2,498/2,544 |
| Median instrumented decision time | 235 ms | 274 ms |

The verified candidate meets both frozen material-progress and application
conditions. Pooled fraction improves 59.43 percentage points; paired geometric
lateral/altitude RMSE ratios are 0.02666/0.19151. The solver reports 49 raw
iteration-limit messages and 46 returned residuals above 0.002, so task success
does not mean every solve converged. These limits are distinct from backend
failure. All candidate decisions exceed the 50 ms sample interval; timings
include diagnostic audits, the simulation is unpaced, and no deadline affects
commands. This is not real-time readiness.

The application requirement remains simultaneous absolute lateral and altitude
error at most ±0.5 m for at least 95% of 281 samples at t≥2 s in **each** trial,
with complete finite trajectories and no declared reliability failure. It is
an internal provisional target with historical feasibility evidence (843/843
samples), not an external standard. The new oracle result supports feasibility
on the current calm, truth-sensed task. It does not establish learned-control,
hardware, disturbance or arbitrary-system performance. The covariance is a
borrowed constant objective offset, not oracle uncertainty.

The last saved generic controller remains poor (position component RMSE
60.80/49.31 m in the earlier two-trial diagnostic). It has not been rerun with
the better controller or paired-response model. Earlier solver and control
results remain historical evidence; none is retrospectively changed to pass.

## Evidence and compatibility

The current trial report is at
`artifacts/2026-09-18/controller-anticipation-v1/report.json`, SHA-256
`c050b77168e3df8e0608665e80f7fb14888ab431e011d75d08b16b073ebeaeb6`.
All **16 trajectories and 5,088 optimizer decisions/forecasts replay exactly**,
with zero physical-state difference. Four coherently rehashed changes to commands,
anticipation configuration, full gradients and task scores are rejected. A separate
NumPy implementation reproduces every task score and saved-gradient residual.
The [result record](harness/controller-anticipation-v1-result.json) pins the
93-file evidence bundle and separate audits. Implementation is isolated at
`9cf2874`, with audit `bfb8d24`, on `codex/controller-anticipation`.
Focused validation passes 108 remote checks and 107 local checks with one Cascade
skip; Ruff passes. Instrumented clocks are checked for consistency, not authenticated.

The preceding intervention qualification has 190 focused local tests passing
(six skips), and 186 remote tests passing (nine skips) with one inherited
migration source-check failure. That failure is a Python 3.12/3.13 AST
serialization difference on byte-identical sources; complete AST comparison
and all 93 raw-byte source pins pass. The original remote suite is not claimed
fully green. Compatibility correction `b4e8126` passed 53 local and 45 remote
checks (eight remote artifact skips) but remains isolated because integration
changed pinned historical sources. Main restoration `c69b20c` passed all six
affected tests and ten source validators; the
[containment record](harness/ast-compatibility-containment-result.json) anchors
that result. Historical replay uses its pinned interpreter and checkout.
The last broad public API/regression check passed 1,287 tests; current research
does not change that recipe or consumer behavior.

## Next named gap

**Transfer of qualified command-response learning into closed-loop tracking.**
Freeze a no-fit comparison of saved
`full_mse_6000` and `paired_6000` under the same anticipatory controller, using
fresh matched initial-state seeds and each arm's own observed trajectory.
Use ten real initial hold intervals to fill the research models' validated
history, with no padding or hidden oracle state; the first solve is at 0.5 s.
Both arms retain the same bounds, cost, solver budget and five-step horizon.
Give both the exact same borrowed covariance coefficients to isolate the
mean-model change, explicitly without claiming calibrated candidate coverage.
Keep material tracking improvement and the unchanged per-trial 95% requirement
separate. No public learner/controller promotion follows automatically.

Public integration still needs a recording-based learning mechanism that does
not assume simulator state cloning, measured uncertainty, and a successful
live-update gate. Existing failures constrain that work: affine command-column
constraints did not control nonlinear/memory response and broke nonlinear
synthetic cases; weak excitation and unconditional response moments did not
identify state-dependent response; widening envelopes by distance repaired some
cases while over-covering others. An explicit latent-state architecture remains
a researched option if a named representation failure warrants it. Current
evidence does not establish that the present architecture is exhausted.
