# Status: gap against the charter

Updated 2026-09-18. **The generic approach is adopted as the development
baseline.** Generality and its four-corpus advantage justify the documented
ARP deficit; this policy decision does not rewrite historical gates. The
adopted learner remains `generic-memory-v3-prototype`: `glassbox.fit` returns
`LearnedDynamics` with `predict` and `update`, without consumer tuning options.
The independent-command-excitation study now establishes a strong collection benefit
for the quadratic research learner: weighted forecast/response error falls
27.8%/71.8% against public, and 10.5%/67.0% against that same architecture trained
on original data. Every frozen residual criterion passes. The public recipe is
unchanged pending the next matched-data comparison; the added quadratic terms
have not yet been shown necessary once training inputs are independently excited.

**Current evaluation priority, 2026-09-18:** reduce held-out Crazyflow/Cascade
forecast and command-response residuals in physical units. The latest accepted
collection mechanism applies one generic bounded command perturbation rule to
both simulators. Crazyflow's angular forecast and tail losses, wind-shift loss and
incomplete trajectories remain visible. Next fit the existing public learner on
the exact same excited recordings before deciding whether quadratic complexity
is justified. Structured comparisons are context, not the finish line. Dart's
prior failures and earlier qualification flags retain their original meaning;
JSBSim breadth work remains deferred.

Read [the charter](charter.md) first. This page records the current gaps and
next iteration; git and frozen result records retain experiment history.

| Criterion | Current | Target |
| --- | --- | --- |
| One recipe | **Met.** The public API and fit/evaluate commands use the single adopted generic recipe, with no model-selection options. | One generic learner and consumer contract. |
| Accuracy | **Adopted; strong collection gain verified in research.** The quadratic model trained on independently excited recordings has 71.8% lower weighted response error than public on fresh matched tests; the isolated within-quadratic collection gain is 67.0%. Crazyflow primary 250 ms velocity response improves 0.17590→0.03527 m/s; Cascade 0.13221→0.03919 m/s. Crazyflow rate forecasts remain 28.3% worse than public and its errors condition on available truth. Public-excited performance and arbitrary-system transfer remain unmeasured. | Low held-out forecast and command-response residuals in physical units across conditions and horizons, with progress against the adopted generic baseline; application adequacy measured separately. |
| Model usability | **Partly met.** The public model has a saved signal/time contract, batched JAX-compatible forecasts, immutable revisions and error envelopes. Independent controller integration and broader runtime/export portability have not been demonstrated. | A documented model artifact and public interface usable independently of the Glassbox controller, with explicit scope and evidence. |
| Capability | **Met.** All 27 frozen synthetic cases pass; saved models replay and reject alteration. These are regression guards, not platform readiness. | Every synthetic absolute cap passes. |
| Reference control | **Demonstrated by both isolated research learners; adopted public model not yet tested with the new controller.** Forecast-only and paired-response means each score 2,248/2,248 and pass all eight trials. Exact replay and integrity checks pass. | Separately qualified downstream demonstrations; one universal controller is optional for the model product. |
| Live improvement | **Not met.** No new model-update forecast/response qualification has been run under the clarified priorities. Last live-v3 swaps at intervals 140/220 increase position error from 0.80/0.98 m to 36.1/11.8 m; those failures remain unresolved. | Bounded immutable revisions with held-out model improvement/regression checks; consumer adoption and any live-control claim qualified separately. |
| Evidence | **Not met.** In the original two-simulator public baseline, primary 250 ms velocity/rate coverage is 85.8–89.7%, but extreme-maneuver coverage falls to 47.4–68.3%. Crazyflow additionally has missing truth after altitude failures. Earlier ARP, reserved-control and shifted-synthetic failures remain. Borrowed constant spread is not calibrated uncertainty for a new predictor. | Measured prediction coverage in the declared band, exposed with calibration provenance independently of a controller. |
| Lean | **Not met.** Structured dynamics, fitting, belief code and research scripts remain. | Learner, model artifacts/interfaces, harness, telemetry adapters and optional downstream consumers. |

## Current independent-command-excitation experiment

[Independent command excitation v1](harness/independent-command-excitation-v1-result.json)
retains the quadratic architecture, 384/256 training/development windows and
1,000-update budget. It changes only training collection: preserve 0.75 s, then
add independent ±5% command-range offsets every 0.10 s and clip to existing bounds.
All development recordings remain byte-identical. Both original-data comparator
refits exactly reproduce their trusted saved revisions.

Weighted factual/response ratios against public are **0.72197 / 0.28220**, with
paired 95% intervals **[0.70267, 0.74617] / [0.27536, 0.28744]**. Against original-data
quadratic, ratios are **0.89498 / 0.33043**. These are weighted geometric RMSE ratios,
not pooled physical errors. Primary 250 ms public→candidate velocity forecast/
response errors are **0.16416→0.06346 / 0.17590→0.03527 m/s** in Crazyflow and
**0.16124→0.12252 / 0.13221→0.03919 m/s** in Cascade. Crazyflow angular response
improves **1.07374→0.30831 rad/s** against original-data quadratic, slightly below
public's **0.31712**. Cascade angular response improves **0.06316→0.03675 rad/s**
against public.

The collection mechanism **passes every frozen residual criterion**. Remaining
losses matter: Crazyflow angular forecasts worsen **0.25568→0.32796 rad/s** against
public, including 23/24 primary cells. Its angular forecast/response parent p95
errors rise **0.42752→0.47976 / 0.42756→0.54544 rad/s**, within the predeclared limits.
Its 10–150 ms angular responses are 46–59% worse than public despite 250 ms
near-parity. Its wind-shift factual aggregate worsens 23.3%. Cascade's angular forecast remains
better than public but is 5.2% worse than original-data quadratic. Broad gains
justify retaining the mechanism; they do not imply improvement in every channel.

Training-only Crazyflow completion falls **49/72→24/72**, and valid duration
**199.18→177.54 s**; all parents remain admitted, with 61/384 window origins changed.
Cascade completes all 72 and retains all window origins. Command covariance
conditioning improves **389→37 / 144→46**. Crazyflow's improvement persists on
identical valid time prefixes, but its visited states change substantially too.
The result supports practical collection, not isolated causal identification.
Training-derived normalization and loss scales change; development data do not.

The common fresh test cohort completes **64/84 Crazyflow** parents and **84/84
Cascade** parents. Crazyflow primary truth is **437/480 factual / 712/768 response**
at 250 ms; Cascade retains **480/480 / 576/576**. All eligible predictions are finite.
All 58,596 data/query arrays, saved predictions, 342,720 metric rows and both
decisions/bootstrap draws replay exactly; independent reductions, 14 alteration
tests and 204 focused tests pass. The [experiment guide](independent-command-excitation.md) records full physical
comparisons, collection and fitting effects, verification and reproduction.
No public recipe, update or controller is promoted by this collection experiment.

## Crazyflow and Cascade baseline evidence

The [frozen two-simulator baseline](harness/two-simulator-flight-v1-result.json)
compares the unchanged public generic learner, freshly fitted structured models
and hold-current. Each simulator has 24 primary cells spanning four headings,
three speeds and two maneuver strengths, plus 18 cells covering held-out initial
heading/speed settings, stronger maneuvers and wind. Achieved state ranges can
overlap; initial speed is not maintained throughout a trajectory. Each has 72
training, 24 development and 84 test parents; paired interventions are reserved
for evaluation. Forecasts
and response queries share physical targets and issued commands, with each
model's actual history contract retained. Structured models retain observed
positions; the generic model uses the 15 velocity, rotation and body-rate
channels. Neither receives hidden simulator state.

Primary 250 ms endpoint component RMSE, **generic / structured**:

| Simulator | Factual velocity, m/s | Factual body rate, rad/s | Command-response velocity, m/s | Command-response body rate, rad/s |
| --- | --- | --- | --- | --- |
| Crazyflow, available truth | 0.1606 / 0.0313 | 0.2734 / 0.2343 | 0.1815 / 0.0166 | 0.3322 / 0.1959 |
| Cascade | 0.1753 / 0.1873 | 0.1186 / 0.1249 | 0.1398 / 0.0451 | 0.0606 / 0.0316 |

Cascade's factual velocity/rate gains are 6.4%/5.0%; its rotation-entry error
still loses (0.0191/0.0161). Crazyflow loses all three factual groups at 250 ms,
but wins primary body-rate forecasts at 10/50/150 ms and wind-shift velocity at
250 ms. Both generic models beat hold-current on the primary 250 ms factual
scores, but their velocity response errors exceed even the zero-response
reference (0.1191 m/s Crazyflow,
0.0898 m/s Cascade). Response RMSE includes weak probes; direction diagnostics
separately apply the frozen physical response thresholds. No all-case win rule
is imposed, and this iteration promotes no learner or controller.

Crazyflow's collection pilot completes 136/180 parents; 44 cross the 0.5 m
altitude boundary. These are collection outcomes, not learned-controller trials.
This includes eight primary test parents and all four extreme-maneuver test
parents. Valid prefixes yield 447/480 primary factual and 720/768 response
queries at 250 ms; missing slots remain in the report. These are conditional
errors, not complete-cohort Crazyflow scores. Cascade completes all 180 parents
and all 480 factual/576 response primary queries. Every eligible prediction is
finite. Requested conditions, achieved valid-prefix motion and invalid tails
are reported separately. Full requested breadth is not established by finite
prefixes alone.

Each generic fit uses 384 training windows, 256 development windows, 1,000
updates and batch size 64. Structured fits use 600 full-batch steps over 6,077
Crazyflow or 6,624 Cascade training windows across three horizons, with
different priors, objectives and history handling. This compares the current
fitting workflows, not architectures at equal data consumption or compute.
Crazyflow selects step zero; every measured later checkpoint has worse
development loss. Cascade selects step 1,000. The saved Crazyflow nonlinear
output and hidden-memory readout weights are exactly zero: its forecast is
affine, with 100 ms effective explicit history despite a 500 ms public context
requirement. Identical additive command changes have history-independent
predicted responses. This is a demonstrated limitation and a plausible
contributor to world-velocity response error, not a causal ablation.

All **55,860 physical/data/query arrays**, saved prediction arrays and **257,040
metric rows** reproduce exactly. Independent NumPy reductions verify all 180
score groups at 250 ms endpoints; other horizons and whole-prefix metrics pass
the production replay. Nine alteration tests pass, including coherent physics
and prediction rewrites that only fresh execution rejects. All 52 pinned-runtime
focused tests and Ruff pass. A Cascade requested-angle semantic correction was
committed before the repeated evaluation; the failed attempt is preserved,
all physical arrays/roles/query inputs are unchanged and completed Crazyflow
scores repeat exactly. The [experiment guide](two-simulator-flight.md) records
the sources, budgets, shifts, coverage, limits and reproduction procedure.

## Earlier five-corpus and intervention evidence

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

**Architecture necessity after improved command excitation.** The successful
collection intervention was measured within the quadratic learner. Its public
comparator still used original recordings, so the study cannot establish that
quadratic terms are needed once commands supply better training information.

Compare three arms on the exact sealed original and excited calibration data:
public-original, public-excited and quadratic-excited. Keep recording roles,
development data, window budgets, optimizer update budgets and candidate
selection rules fixed; use a fresh unperturbed held-out cohort. The public learner
must consume the excited recordings through its ordinary `fit` contract, without
excitation labels or simulator state. Refit trusted comparator revisions exactly.

Freeze the metrics, public-baseline improvement requirements and acceptable
matched-data differences before fitting. Public-original→public-excited measures
the collection benefit in the adopted recipe; public-excited→quadratic-excited
measures the added architecture's benefit at matched data/update budgets, without
claiming equal FLOPs. Prefer the existing simpler public recipe if physical errors
and tails are comparable under those frozen criteria; otherwise retain the
quadratic mechanism for a separately declared public capability/contract
qualification. Do not redesign the collection after inspecting these test results.

Retain Crazyflow's widespread angular-forecast loss, wind-shift loss, missing
truth and increased training failures as explicit improvement targets. Forecast,
response, derivative, envelope, update and controller claims remain separate.
