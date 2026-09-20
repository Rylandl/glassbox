# Status: gap against the charter

Updated 2026-09-20. Read [the charter](charter.md) first. The public recipe is
`generic-memory-v4-prototype`: the retained expanded-cache mean is available
through ordinary `fit`, `predict`, `update` and save/load. The current loader
supports this recipe only; historical archives require their pinned source.

**Current priority:** run Dart with one general vehicle learner, using shared
physical structure where it improves learning across vehicle types and
configurations. The user has clarified that generality permits common mechanics;
it excludes a vehicle-family catalog or user-supplied vehicle parameters/layout.
A separate fit is expected; the recipe and automatic fitting procedure stay
shared across configurations. First establish current v4 performance
and the planning-interface gap on Dart's ordinary recordings and fixed task.
The independent-recording update passed its frozen criteria: forecast/response
aggregate errors fell 13.88%/5.81% at fixed cache and fitting work, with documented
localized regressions. The initialized-mean prior has now been tested and rejected:
forecast error fell 1.53%, response error rose 4.45%, and their joint ratio was
1.01415 against the frozen limit of 0.97. The retained v4 revisions stay in place.
The earlier proposed paired-response experiment was not frozen or run. It is
superseded as the immediate next step by the
[Dart general-vehicle plan](dart-general-vehicle-plan.md).
Public mean adoption is an explicit
[policy decision on known evidence](public-mean-adoption.md). The original
qualification and later finite-difference assessment retain their failed frozen
verdicts. The independent-recording result qualifies one bounded offline update intervention;
calibrated uncertainty, physical derivative fidelity and controller readiness
remain unqualified. JSBSim breadth is deferred.

| Criterion | Current | Target |
| --- | --- | --- |
| One recipe | **Met.** One public recipe, with no consumer tuning options or platform dispatch. | One generic learner and consumer contract. |
| Accuracy | **Improved.** Fixed-budget public updates reduce fresh matched forecast/response errors 13.88%/5.81% versus saved v4. Eighteen of twenty scope aggregates improve; 38/180 endpoint cells worsen within the frozen aggregate limits. Earlier matched public-v4 errors were 52.82%/55.26% below saved v3 on a different cohort. | Low held-out physical forecast and command-response errors across conditions and horizons; application adequacy measured separately. |
| Model usability | **Demonstrated within the Python/JAX contract.** Public fitting, persistence, lifecycle semantics, precision switching, saved-mean replay and independent Dart integration pass. Two frozen derivative gates remain failed; adoption rationale and numerical limits are explicit. Broader runtime/export portability remains unqualified. | Self-contained artifacts usable independently of Glassbox controller internals, with clear scope and measured evidence. |
| Capability | **Met.** The adopted public recipe passes all 27 frozen synthetic absolute-capability cases. | Every synthetic cap passes; synthetic results do not establish platform readiness. |
| Reference control | **Dart integration remains open.** Dart's structured mean achieves its nominal contact task. Current v4 has short-horizon forecast/JVP consumer evidence, but has not run the original Dart terminal MPC. Earlier isolated research means passed eight Cascade tracking trials. | The general learner meets Dart's declared pose/contact task through Dart's controller; a universal controller is optional. |
| Live improvement | **Demonstrated for one bounded offline intervention.** Two ordinary public updates add 72 recordings each at fixed cache/work, preserve original revisions, and pass fresh held-out residual/regression checks and replay. Safe live swaps remain unqualified; historical live-v3 swaps worsened position error from 0.80/0.98 to 36.1/11.8 m. | Bounded immutable revisions with held-out improvement/regression checks; adoption and live-control claims qualified separately. |
| Evidence | **Not met.** Development-calibrated spreads have no new coverage qualification. Earlier public primary 250 ms velocity/rate coverage was 85.8–89.7%, falling to 47.4–68.3% for extreme maneuvers; ARP, reserved-control and shifted-synthetic failures remain. | Measured coverage in a predeclared band with calibration provenance. |
| Lean | **Not met.** Structured dynamics, belief code and research scripts remain. | Learner, artifacts/interfaces, harness, telemetry adapters and optional consumers. |

## Latest matched evidence

The [initialized-mean prior](initialized-mean-prior.md) tested one fixed prediction
penalty on the unchanged 144-parent, 1,536-window training cache and original
development cache. It improved Crazyflow forecast/response aggregates by
6.34%/2.29%, but worsened Cascade by 3.52%/11.66%. Twelve of twenty scope
aggregates improved and 93/180 endpoint cells regressed. Only the frozen joint
progress criterion failed; all regression and evidence-integrity checks passed.
Both exact replays, an independent 257,040-row reduction and all eight alteration
checks confirmed the result without refitting. Fresh collection completed 63/84
Crazyflow and 84/84 Cascade parents; all arms used identical available truth.
This rejected research loss is absent from the public learner. The
[result](harness/initialized-mean-prior-v1-result.json) preserves the physical
errors, actual work, failed verdict and immutable source/evidence anchors.

The [independent-recording update](independent-training-recordings.md) qualifies
one fixed addition of 72 excited training recordings per simulator through public
`update`, retaining the same 1,536 training windows, exact 256 development windows
and 1,000 steps. Fresh weighted forecast/response ratios are 0.86116/0.94186
against saved v4. Crazyflow primary 250 ms angular forecast/response errors fall
0.15216 → 0.12329 and 0.16336 → 0.14159 rad/s; their parent-p95 ratios are
0.72970/0.88186. Cascade primary and wind response aggregates worsen
3.72%/9.61%, within frozen limits. Raw losses remain visible in 38/180 cells.
All 195 tests, both canonical replays, the independent 257,040-row reduction and
eight alteration checks pass. Fresh confirmation completes 59/84 Crazyflow and
84/84 Cascade parents; every predictor uses identical available truth. The
[result](harness/independent-training-recordings-v1-result.json) records absolute
errors, costs and evidence anchors. The public recipe is unchanged.

The [original public qualification](harness/public-mean-qualification-v1-result.json)
records 32 completed fits: two flight, 27 synthetic and three lifecycle. All
6,384 float64 flight prediction arrays match the retained research mean exactly.
No completed fit should be repeated for replay or source bookkeeping.

Primary 250 ms component RMSE, saved public v3 → adopted public v4:

| Simulator | Forecast velocity, m/s | Forecast rate, rad/s | Response velocity, m/s | Response rate, rad/s |
| --- | ---: | ---: | ---: | ---: |
| Crazyflow, available truth | 0.16468 → 0.05060 | 0.19015 → 0.17146 | 0.11889 → 0.02364 | 0.24254 → 0.15553 |
| Cascade | 0.17089 → 0.07448 | 0.10220 → 0.04757 | 0.05195 → 0.02082 | 0.03609 → 0.02013 |

All twenty simulator/scope/kind aggregates improve against saved public v3.
Thirteen of 180 reported 50/150/250 ms endpoint RMSE cells worsen: eleven 50 ms response
cells and two Crazyflow wind-shift velocity forecasts. For example, Crazyflow
maneuver-shift 50 ms response velocity rises 0.001414 → 0.001786 m/s. Two tiny
response regressions fall below the frozen normalization floor, so absolute
changes rather than floored ratios identify the complete regression count.
The comparison includes changed fitting mechanics and a larger training cache;
it does not isolate architecture. All predictors use the same available truth.
Crazyflow completes 62/84 test parents, with 22 altitude failures; primary 250 ms
forecast/response truth is 421/480 and 695/768. Cascade completes 84/84 parents,
with full truth. Incomplete collection remains a benchmark limitation.

The separate [expanded-cache experiment](harness/expanded-training-cache-v1-result.json)
used 1,536 training windows from the same 72 training recordings per simulator,
versus a 384-window control. Forecast/response ratios were 0.61964/0.67268 against
the balanced research mean and 0.56914/0.59144 against the full-cache 384 control.
It used four times the gradient-window work. Crazyflow primary factual angular
RMSE increased from 0.15653 to 0.16063 rad/s (2.62%), within the declared limit,
while forecast/response angular parent-p95 errors improved. That experiment's
66/84 completed Crazyflow parents belong to a different cohort, not an accuracy
gain attributable to the learner.

## Public integration and numerical limits

The [precision correction](harness/public-mean-precision-correction-v1-result.json)
locally materializes model constants in the caller's ambient inference precision.
It fixes the observed same-process JIT32-to64 failure without changing saved
model arrays or fixed-mode outputs. Fitting and calibration own float64 precision;
there is no consumer precision option.

The [corrected-source regression](harness/public-mean-corrected-replay-result.json)
reproduces both full physical prediction arms, all 428,400 metric rows, development
calibration, all 27 saved synthetic models, three lifecycle revisions and Dart's
168 arrays including JVPs. The physical replay's original finalization schema
failure is preserved; a separate finalizer authenticates the original provenance
and completes the scientific comparison without repeating flight forecasts.

The [finite-difference assessment](harness/public-mean-fd-assessment-v1-result.json)
retains the original 33 numerical cases and eight input-only confirmation cases
from the same historical cohort. All 111 required final Richardson/AD comparisons,
111 original small-step float64 derivative checks, 444 float32 endpoint/grid
checks and 2,542 other required checks pass. Two adjacent-Richardson agreement
checks fail; the original eleven coarse-step float32 failures also remain.
The final derivative errors in the two convergence failures consume only 12.3%
and 8.9% of their allowances. Their adjacent differences behave as expected for
fourth-order convergence and are conservative estimates of the finest error.
This supports the explicit adoption judgment; neither frozen verdict is changed.
The added inputs and corrected-code replays are numerical/regression evidence,
not a new independent physical-accuracy cohort or physical derivative qualification.

Evidence mirrors retain original execution roots, source bindings and failed
attempts. The [continuation record](/private/tmp/glassbox-public-mean-qualification/artifacts/2026-09-19/public-mean-qualification-v1-continuation.json)
holds process handles and anchors. Source-bound workers remain immutable.

## Next named gap

**Dart task compatibility and useful vehicle generalization.** The current
learner predicts Euclidean changes to velocity, angular rate and nine rotation
entries. It has learned memory but no built-in gravity, rotation integration or
body-frame acceleration factorization. The structured comparator supplies that
mechanical structure alongside vehicle-specific force/actuator assumptions.
Their contributions to the performance difference have not been isolated.

The documented Dart generic comparison used v3, not current v4. Current v4's
public forecast requires 500 ms of observed history and supports 250 ms ahead;
the original contact planner optimizes a roughly 1.2 s trajectory from a
physical state and simulator-provided applied actuator state. A direct import replacement cannot
establish an apples-to-apples task result. Fix data roles, command/frame mapping,
history initialization, differentiable pose reconstruction, long-rollout policy
and controller settings before measuring the deficit. Do not silently shorten
the task horizon or treat chained forecasts as qualified beyond 250 ms.

The [next plan](dart-general-vehicle-plan.md) makes this diagnostic the first
iteration, then tests one shared mechanical factorization with learned effective
accelerations and hidden response if the evidence supports it. Keep simulator
equations, mixers and family dispatch out of that learner. Physical signal
semantics and frames are legitimate input contracts. Measure both forecast
and command-response errors and the fixed Dart task; one does not replace the
other. No next protocol, fit or controller trial has run. The inspected +12M
and +13M cohorts remain diagnostic, and completed workers at `1f46773` and
`0d5f3b1` remain immutable.

## Preserved qualification boundaries

- [Original two-simulator result](harness/two-simulator-flight-v1-result.json):
  public versus structured fitting workflows use different data consumption and
  budgets. Those comparisons do not isolate architecture.
- [Affine-centered initialization result](harness/affine-anchored-quadratic-v1-result.json):
  the preceding unweighted optimizer sacrifices angular progress on both training
  and development; its candidate remains unpromoted under that frozen protocol.
- [Intervention response](harness/intervention-response-v1-result.json) and
  [controller transfer](harness/learned-controller-transfer-v1-result.json):
  earlier isolated research demonstrations, with separate model/controller scopes;
  neither qualifies the current public or expanded-cache model.
- [JSBSim startup diagnosis](harness/jsbsim-excitation-v1-result.json): excitation
  and setup evidence only, with missing/invalid configurations; no fitted-model
  accuracy or fleet-wide flying claim.
- [Compatibility containment](harness/ast-compatibility-containment-result.json):
  historical replay retains pinned sources/interpreters. Historical compatibility
  failures and corrections keep their recorded outcomes.

The earlier five-corpus adoption includes the known ARP body-rate deficit
(0.715 versus structured 0.285 rad/s) and X8 tail limitation. Generality justified
adoption; neither those wins nor the new simulator gains establish arbitrary-system
readiness, calibrated uncertainty, physical derivative fidelity or safe live swaps.
