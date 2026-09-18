# Status: gap against the charter

Updated 2026-09-18. **The generic approach is adopted as the development
baseline.** Generality and its four-corpus advantage justify the documented
ARP deficit; this policy decision does not rewrite historical gates. The
adopted learner remains `generic-memory-v3-prototype`: `glassbox.fit` returns
`LearnedDynamics` with `predict` and `update`, without consumer tuning options.
The paired-response learner and anticipatory controller remain isolated
research implementations. **Both saved research learners now meet the declared
tracking requirement: each scores 2,248/2,248 qualifying samples and passes all
eight fresh trials. All sixteen trajectories and 4,960 optimizer decisions/forecasts
replay exactly, and five rehashed alterations are rejected.**

**Current evaluation priority, 2026-09-18:** the user has requested Crazyflow
and Cascade environments spanning a broader variety of flight conditions.
Dart reports severe direct forecast deficits for the generic learner on its
quad recordings and fixed-wing turn. Reproduce and investigate those failures,
then establish a fresh controlled comparison before changing the learner.
Further JSBSim setup/breadth work is deferred. Frozen results and qualification
flags retain their original meaning.

Read [the charter](charter.md) first. This page records the current gaps and
next iteration; git and frozen result records retain experiment history.

| Criterion | Current | Target |
| --- | --- | --- |
| One recipe | **Met.** The public API and fit/evaluate commands use the single adopted generic recipe, with no model-selection options. | One generic learner and consumer contract. |
| Accuracy | **Adopted with a known tradeoff.** Four of five corpora beat structured comparators on both metrics; ARP loses both. In a separate simulator study, paired-response supervision improves aggregate response error 65.1% and factual error 31.4% against the same-data, same-budget forecast-only learner. JSBSim breadth and arbitrary-system transfer remain unmeasured. | Broad competitive forecasts and command responses from one platform-independent recipe; direct model errors and downstream task sufficiency measured separately. |
| Model usability | **Partly met.** The public model has a saved signal/time contract, batched JAX-compatible forecasts, immutable revisions and error envelopes. Independent controller integration and broader runtime/export portability have not been demonstrated. | A documented model artifact and public interface usable independently of the Glassbox controller, with explicit scope and evidence. |
| Capability | **Met.** All 27 frozen synthetic cases pass; saved models replay and reject alteration. These are regression guards, not platform readiness. | Every synthetic absolute cap passes. |
| Reference control | **Demonstrated by both isolated research learners; adopted public model not yet tested with the new controller.** Forecast-only and paired-response means each score 2,248/2,248 and pass all eight trials. Exact replay and integrity checks pass. | Separately qualified downstream demonstrations; one universal controller is optional for the model product. |
| Live improvement | **Not met.** No new model-update forecast/response qualification has been run under the clarified priorities. Last live-v3 swaps at intervals 140/220 increase position error from 0.80/0.98 m to 36.1/11.8 m; those failures remain unresolved. | Bounded immutable revisions with held-out model improvement/regression checks; consumer adoption and any live-control claim qualified separately. |
| Evidence | **Not met.** The 85–95% coverage band still fails on ARP, the reserved control recording and shifted synthetic regimes. Borrowed constant spread is not calibrated uncertainty for a new predictor. | Measured prediction coverage in the declared band, exposed with calibration provenance independently of a controller. |
| Lean | **Not met.** Structured dynamics, fitting, belief code and research scripts remain. | Learner, model artifacts/interfaces, harness, telemetry adapters and optional downstream consumers. |

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

## Current JSBSim benchmark evidence

The frozen [startup and response-timescale diagnosis](harness/jsbsim-excitation-v1.json),
implemented at `be994c9`, compares all 66 pinned JSBSim 1.3.1 candidates under
shipped initialization and one explicit engine-startup call. Each arm retains
37 completed configurations, 25 missing initializations, three load failures
and the `minisgs` numerical failure. The [original onboarding result](harness/jsbsim-onboarding-v1-result.json)
and all 96 inherited source pins remain unchanged.

On the **same 189 commands in the original 37 completed configurations**, detected
cumulative responses are:

| Horizon | As shipped | Engine startup | Matched gains / losses |
| --- | ---: | ---: | ---: |
| 0.25 s | 45/189 | 115/189 | 70 / 0 |
| 1 s | 47/189 | 123/189 | 76 / 0 |
| 2 s | 48/189 | 125/189 | 77 / 0 |
| 5 s | 48/189 | 125/189 | 77 / 0 |

Throttles improve from 1/78 to 35/78 at 0.25 seconds and from 3/78 to 45/78
at five seconds. These are small-threshold excitation measurements, **not model
accuracy or useful control-authority scores**. Fixed endpoint activity differs:
47/189 versus 122/189 at five seconds. Startup increases the number of completed
configurations with any detected response from 16 to 29 at 0.25 seconds, and
17 to 29 at five seconds. The remaining eight still have no detected response.
The all-candidate 0.25-second denominator additionally includes three active
`minisgs` channels before its later failure: 48/192 versus 118/192. Those finite
prefixes are not admitted as a useful operating regime.

Saved actuator/engine traces distinguish actual setup and interface problems.
Global5000 already has running engines but takes about 0.9 seconds to respond;
F450's indexed throttles change their commands without changing their effective
motor positions, thrust or motion. Engine-running flags do not universally
indicate power. Some finite trajectories pass below ground; L410 reaches about
507 m/s in the startup arm with suspect thrust. A successful startup call and
finite completion therefore do not establish a valid learning condition.

**All 132 outcomes and 34,716 arrays reproduce byte for byte in fresh execution.**
Independent NumPy reductions reproduce horizon activity, physical endpoint
magnitudes and paired counts. Eight rehashed alteration tests are rejected,
including a coherent trajectory rewrite that passes internal reduction and fails
fresh physics replay. All 146 focused tests and Ruff pass. There were no timeouts
or native crashes. The [result record](harness/jsbsim-excitation-v1-result.json)
anchors the 403-file evidence bundle and independent audits; the
[experiment guide](jsbsim-excitation.md) documents its scope and reproduction.
No learner was fitted, model accuracy measured, or controller run.

## Current controller evidence

[Learned controller transfer v1](harness/learned-controller-transfer-v1.json)
compares the saved `full_mse_6000` and `paired_6000` means without fitting, on eight
fresh matched seeds, 118–125. Both use the same anticipatory controller, five-step
250 ms horizon, captured float32-to-float64 planning, solver budget and constant
covariance. Ten actual initial hold transitions supply their full validated
history; there is no padding. Each arm controls its own trajectory and warm starts.
All 1,176 development preflight witnesses pass history/reset parity, geometry and
every gradient-coordinate check before the trials begin.

| Verified learned comparison | Forecast-only | Paired response |
| --- | --- | --- |
| Simultaneous lateral/altitude tolerance samples | **2,248/2,248 (100%)** | **2,248/2,248 (100%)** |
| Trials meeting the 95% requirement | **8/8** | **8/8** |
| Backend failures / fallbacks / bound violations | 0 / 0 / 0 | 0 / 0 / 0 |
| Returned residuals at or below 0.002 | 811/2,480 | 2,346/2,480 |
| New optimizer objective evaluations | 163,330 | 91,835 |
| Median instrumented decision time | 13.62 ms | 10.59 ms |

Paired geometric lateral/altitude RMSE ratios are **0.82497/0.53919**: improvements
of 17.5%/46.1%. Lateral RMSE improves on six seeds and worsens on two (3.15% and
23.37%); altitude improves on all eight. Both models pass the absolute application
condition. The separate frozen material-progress flag remains **false** because
tolerance-fraction gain is zero at the baseline ceiling, and lateral ratio exceeds
its declared 0.8 bound. This does not undo application qualification; no rule is
rewritten after seeing the result. Paired loss is beneficial on several measured
quantities, but is not demonstrated necessary for this tracking task.

The application requires simultaneous absolute lateral and altitude errors at most
±0.5 m for at least 95% of 281 samples at t≥2 s in **each** trial, complete finite
trajectories and no declared reliability failure. It is an internal provisional
target, not an external standard. Eight seed pairs are the experimental units;
2,248 correlated samples are not independent demonstrations across systems.
This is one calm, truth-sensed aircraft task with initial-state perturbations.

There are no nonfinite evaluations or returned rotation-geometry violations.
Iteration limits remain: 1,669 forecast-only and 134 paired returned residuals
exceed 0.002. Task success does not imply every solve converged. Although median
latency is below the 50 ms sample interval, 15/2,480 and 13/2,480 decisions exceed
it; timing includes diagnostics, the simulation is unpaced and no deadline changes
commands. No real-time qualification follows. The shared covariance is borrowed
from the adopted archive, not calibrated uncertainty for either research mean.

The preceding [oracle comparison](harness/controller-anticipation-v1-result.json)
showed that replacing each position residual with position error plus 1.5 seconds
times velocity error raised fresh oracle performance from 40.57% to 100% without
extending the forecast horizon. Those two-hold trials are diagnostic context,
not a matched comparator for these ten-hold learned trials. The old poor generic
controller result (60.80/49.31 m position RMSE) used a different controller and
startup, and cannot establish that the saved adopted predictor is inadequate here.

## Evidence and compatibility

The current trial report is at
`artifacts/2026-09-18/learned-controller-transfer-v1/report.json`.
The [result record](harness/learned-controller-transfer-v1-result.json) pins the
1,296-file evidence bundle, separate full replay/tamper audit and independent NumPy
analysis. All sixteen physical trajectories and **4,960 optimizer decisions and
forecasts replay exactly**, including histories, seeds, warm starts, raw forecasts,
gradients and work. Five coherently rehashed model/history/configuration/gradient/
summary alterations are rejected. Independent NumPy analysis reproduces every task
score, paired RMSE, saved-gradient residual and descriptive timing/work reduction.
Implementation `4aeaaaf`, test correction `709e047` and audit `cf5a69a` remain isolated
on `codex/learned-controller-transfer`. All **147 remote focused checks pass** and
Ruff passes. The initial remote run had one test-only eager/JIT rounding expectation
failure (3.7e-9), corrected before any saved-model measurement; the original log
remains available. No experiment source, numerical rule or threshold changed.
Instrumented clocks are checked for consistency, not authenticated. This work
promotes no public recipe, envelope or controller.

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

**Generic prediction performance across controlled Crazyflow and Cascade flight conditions.**
The user has redirected the next iteration to these two simulators. Start from
Dart's concrete saved-data comparison, identify its exact predictors, data roles,
signal/timing contracts and simulator versions, and get isolated, pinned runtime
environments working. The generic learner remains the adopted baseline; reported
losses are improvement work, not a reason to return to a model catalog.

Freeze one bounded evaluation before fresh flight trials or fitting. It must
cover materially different headings, speeds and maneuvers, with wind/condition
shifts where the simulator interface supports them. Declare simulation reset,
trim, stabilization and hidden-state assumptions. Keep failed or unsupported
conditions visible. Separate training coverage from held-out seeds, recordings
and conditions; a previously inspected Dart failure is a regression diagnostic,
not untouched confirmation evidence.

Compare forecasts on identical observed histories, commands and physical targets,
with clear treatment of each model's actual history contract. Report errors in
physical units across supported horizons, per condition and per simulator;
include structured and hold-current references, command-response measurements,
error-envelope coverage and computation/data budgets. Freeze aggregate weights
and improvement/regression decisions before fitting. No controller success claim
substitutes for model accuracy, and the truth simulator must not supply hidden
state or equations to the generic learner.

Use the current public learner through its recording and prediction contract.
No aircraft branches, model-selection menu or consumer tuning options are added.
The initial iteration establishes a trustworthy broader baseline and diagnoses
whether data coverage, representation, optimization or response learning is the
next mechanism to change. No fresh two-simulator benchmark has run yet.
