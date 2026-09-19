# Status: gap against the charter

Updated 2026-09-19. Read [the charter](charter.md) first. The public recipe is
`generic-memory-v4-prototype`: the retained expanded-cache mean is available
through ordinary `fit`, `predict`, `update` and save/load. The current loader
supports this recipe only; historical archives require their pinned source.

**Current priority:** reduce the remaining angular generalization error by
adding independent excited training recordings, with the training-window cache,
development cache and fitting budget fixed. Public mean adoption is an explicit
[policy decision on known evidence](public-mean-adoption.md). The original
qualification and later finite-difference assessment retain their failed frozen
verdicts. Adoption does not establish calibrated uncertainty, physical derivative
fidelity, update improvement or controller readiness. JSBSim breadth is deferred.

| Criterion | Current | Target |
| --- | --- | --- |
| One recipe | **Met.** One public recipe, with no consumer tuning options or platform dispatch. | One generic learner and consumer contract. |
| Accuracy | **Improved.** Fresh matched public-v4 forecast/response aggregate errors are 52.82%/55.26% lower than saved public v3. The earlier comparison with the retained balanced research model includes a 2.62% Crazyflow primary factual angular loss, within its frozen limit. | Low held-out physical forecast and command-response errors across conditions and horizons; application adequacy measured separately. |
| Model usability | **Demonstrated within the Python/JAX contract.** Public fitting, persistence, lifecycle semantics, precision switching, saved-mean replay and independent Dart integration pass. Two frozen derivative gates remain failed; adoption rationale and numerical limits are explicit. Broader runtime/export portability remains unqualified. | Self-contained artifacts usable independently of Glassbox controller internals, with clear scope and measured evidence. |
| Capability | **Met.** The adopted public recipe passes all 27 frozen synthetic absolute-capability cases. | Every synthetic cap passes; synthetic results do not establish platform readiness. |
| Reference control | **Demonstrated for earlier isolated research means.** Forecast-only and paired-response means each achieved 2,248/2,248 tolerance samples across eight trials. The adopted public mean has not been tested with that controller. | Separately qualified downstream demonstrations; a universal controller is optional. |
| Live improvement | **Not met.** Public update semantics pass, but update accuracy is unqualified. Historical live-v3 swaps at intervals 140/220 worsened position error from 0.80/0.98 to 36.1/11.8 m. | Bounded immutable revisions with held-out improvement/regression checks; adoption and live-control claims qualified separately. |
| Evidence | **Not met.** Development-calibrated spreads have no new coverage qualification. Earlier public primary 250 ms velocity/rate coverage was 85.8–89.7%, falling to 47.4–68.3% for extreme maneuvers; ARP, reserved-control and shifted-synthetic failures remain. | Measured coverage in a predeclared band with calibration provenance. |
| Lean | **Not met.** Structured dynamics, belief code and research scripts remain. | Learner, artifacts/interfaces, harness, telemetry adapters and optional consumers. |

## Latest matched evidence

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

**Angular generalization across independent recordings.** The next protocol and
literal roster are frozen at `681daed` in
`/private/tmp/glassbox-independent-training-recordings`; implementation is in
progress and no new collection or fitting has run. In the retained
expanded-cache fit, Crazyflow training angular RMSE improved
0.14523 → 0.07347 rad/s while development angular RMSE worsened
0.13647 → 0.16375 rad/s. The cache already covers 90.38%/93.11% of eligible
forecast transitions but adds no independent parents. Limited independent
recording support, objective tradeoffs and model bias remain competing causes.

Test one additional fixed cohort of 72 independent excited training recordings
per simulator across the existing primary conditions. Hold the 1,536 training
windows, exact original 256 development windows and 1,000-step fitting budget
fixed. Import the exact saved adopted control instead of refitting it. Declare
the added recordings as a data intervention, including changes to data-derived
normalizers, initialization and objective weights. Preserve every collection
failure and use a fresh, common held-out cohort for confirmation.

Use the existing public `model.update(new_recordings)` path, which preserves the
original development set and bounded cache. Verify that its deterministic cache
merge equals full extraction from the original and added training recordings;
do not add a consumer role-selection option. The experiment can measure offline
update accuracy without implying safe live controller swaps. Freeze the protocol
and implementation before collection/fitting. Do not expand shifted training
conditions in the same experiment. Use its result to distinguish whether more
independent data helps before changing the architecture or objective again.

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
