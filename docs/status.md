# Status: gap against the charter

Updated 2026-09-19. Read [the charter](charter.md) first. The generic approach is
adopted; the public recipe remains `generic-memory-v3-prototype`. Research
accuracy, public adoption and application qualification remain separate.

**Current priority:** complete numerical qualification of the expanded-cache
public candidate. Fresh public fits reproduce the retained research mean; their
held-out forecast/response aggregate errors are 52.82%/55.26% lower than saved
public v3. Physical comparisons, all 27 synthetic capability cases, independent
Dart integration and the twelve-case artifact alteration audit pass. Public
promotion remains false: the original numerical gate has 11 reproducible Cascade
float32 finite-difference failures. A separate runtime correction passes both
precision-switching orders on all 24 actual cases, with 432 output arrays exactly
matching prior fixed-mode predictions, and 36 targeted regression tests. Its
complete 33-case arithmetic replay is still running. JSBSim breadth is deferred.

The qualification is unmerged. Immutable workers are
`/private/tmp/glassbox-public-mean-qualification` (original 32 fitting operations,
all complete), `/private/tmp/glassbox-public-mean-adjudication` (physical/consumer
results), and `/private/tmp/glassbox-public-mean-integrity` (diagnosis/audit).
The runtime correction is committed at `41e35ab` in
`/private/tmp/glassbox-public-mean-precision`; the test-only ownership correction
is `355fb75` in `/private/tmp/glassbox-public-mean-precision-review`, with identical
learner source. Do not modify source-bound workers or repeat completed fits.
The [continuation record](/private/tmp/glassbox-public-mean-qualification/artifacts/2026-09-19/public-mean-qualification-v1-continuation.json)
holds the current process handles and artifact anchors.

| Criterion | Current | Target |
| --- | --- | --- |
| One recipe | **Met.** Public `fit` returns `LearnedDynamics` with `predict` and `update`; no consumer tuning options or platform dispatch. | One generic learner and consumer contract. |
| Accuracy | **Expanded-cache research mean retained.** Forecast/response ratios against balanced are 0.61964/0.67268 and against public-excited are 0.47308/0.45947 on one fresh matched cohort. Crazyflow primary 250 ms factual rate worsens 2.62%, within the frozen limit; both angular tails improve. | Low held-out physical forecast and command-response errors across conditions and horizons, with progress against the adopted generic baseline; application adequacy measured separately. |
| Model usability | **Partly met.** The unmerged candidate passes independent Dart integration and saved-mean parity. Its precision-switching correction passes actual flight cases; full arithmetic replay and revised numerical assessment remain. Broader export/runtime portability is unqualified. | A self-contained model artifact usable independently of the Glassbox controller, with explicit scope and measured evidence. |
| Capability | **Met for the adopted recipe and unmerged public candidate.** Both pass all 27 frozen synthetic absolute-capability cases. Candidate public adoption remains a separate unfinished qualification. | Every synthetic absolute cap passes; synthetic results do not establish platform readiness. |
| Reference control | **Demonstrated for earlier isolated research means.** Forecast-only and paired-response means each achieved 2,248/2,248 tolerance samples across eight trials. The adopted public and expanded-cache research means have not been tested with that controller. | Separately qualified downstream demonstrations; a universal controller is optional. |
| Live improvement | **Not met.** No update qualification under the clarified model-first priorities. Historical live-v3 swaps at intervals 140/220 worsened position error from 0.80/0.98 to 36.1/11.8 m. | Bounded immutable revisions with held-out improvement/regression checks; adoption and any live-control claim qualified separately. |
| Evidence | **Not met.** Original public primary 250 ms velocity/rate coverage is 85.8–89.7%, falling to 47.4–68.3% for extreme maneuvers. Crazyflow truth is incomplete. Development-calibrated research spreads have no new coverage qualification. Earlier ARP, reserved-control and shifted-synthetic failures remain. | Measured coverage in a predeclared band with calibration provenance, independent of the controller. |
| Lean | **Not met.** Structured dynamics, belief code and research scripts remain. | Learner, model artifacts/interfaces, harness, telemetry adapters and optional consumers. |

## Latest matched evidence

[Expanded training cache v1](harness/expanded-training-cache-v1-result.json)
increases the fixed training cache from 384 to 1,536 windows using the same
72 admitted training recordings per simulator, with the original 384-window
prefix and all 256 development windows unchanged. Architecture and fitting
formulas stay fixed; training-derived norms, initialization and objective weights
change. Two candidates and six exact imported reference models predict the same
fresh cohort. Gradient-window work is four times the 384-window control.

| Reference | Candidate forecast ratio | Candidate response ratio |
| --- | ---: | ---: |
| Retained balanced model | 0.61964 | 0.67268 |
| Full-cache 384-window control | 0.56914 | 0.59144 |
| Public-excited | 0.47308 | 0.45947 |

All three comparisons, angular retention, finite predictions and matched-truth
checks pass. Retain the expanded-cache research mean; public promotion remains
false. The separate optimization diagnostic passes and cannot veto physical
progress. See the [guide](expanded-training-cache.md) for intervals, figures,
computation costs, limitations and reproduction.

Primary 250 ms component RMSE, balanced → expanded cache:

| Simulator | Forecast velocity, m/s | Forecast rate, rad/s | Response velocity, m/s | Response rate, rad/s |
| --- | ---: | ---: | ---: | ---: |
| Crazyflow, available truth | 0.11621 → 0.04926 | 0.15653 → 0.16063 | 0.04010 → 0.02335 | 0.16675 → 0.15673 |
| Cascade | 0.13096 → 0.06895 | 0.10421 → 0.04942 | 0.03669 → 0.02039 | 0.04090 → 0.01865 |

Crazyflow factual angular mean worsens 2.62%, while its parent-p95 improves
0.27004→0.24830 rad/s. Response angular mean and p95 improve 6.01% and 5.46%.
Generality and broad gains do not require every metric to win; this localized
loss satisfies the prospectively declared limit.

Crazyflow completes 66/84 test parents; 18 fail the altitude condition. Primary
250 ms truth is 438/480 forecast and 720/768 response queries. Cascade completes
84/84 with all truth. All five predictors are finite on every eligible query.
Comparisons share the same truth masks. Cohort-completion changes are not
learner gains; no new safeguarded or quadratic comparison was run.

Both fits select checkpoint 1,000. Crazyflow training/development weighted
losses fall 84.91%/46.65% from its own initialization; Cascade falls
90.76%/81.12%. These are within-fit reductions, not comparable raw objectives
across models. The remaining angular gap is concrete: Crazyflow training rate
RMSE improves 0.14523→0.07347 rad/s while development rate worsens
0.13647→0.16375 rad/s. Independent-recording support, objective tradeoffs and
model bias remain competing explanations.

Verification is complete: 1,136 focused tests, both full simulator replays,
independent physical/preparation/optimizer reductions and all 37 alteration
challenges pass. Replay reproduces 4,032 prediction queries, 428,400 metric rows,
88 checkpoint snapshots and 7,375 candidate acceptance-objective calls.
The first attempt's finalization schema failure is preserved; the correction
changes no numerical recipe or thresholds, and exact scientific equality with
the corrected run is independently verified.

## Next named gap

**Qualify and expose the retained mean through the single public API.** The
research result should become usable by consumers through ordinary
`fit`, `predict`, `update`, save/load and JAX interfaces. Freeze one recipe,
precision policy and qualification protocol before implementation. Reuse the
fixed synthetic absolute caps and public semantic checks, require saved-mean
prediction/derivative parity, and confirm fresh public fitting on a common
held-out physical cohort with predeclared broad improvement and regression
limits. Demonstrate an independent model consumer without controller internals.

The candidate now reproduces research preparation, initialization and selected
parameters through ordinary public fitting. Both physical replays reproduce
428,400 metric rows; all 6,384 float64 forecast arrays match research exactly.
Crazyflow completes 62/84 fresh test parents, with 22 altitude failures; primary
forecast/response truth is 421/480 and 695/768. Cascade completes all 84 parents.
These are a different cohort from the retained research result above, not a
change in collection success attributable to the model.

Fixed-step diagnosis finds coarse-step truncation dominates all eleven failed
Cascade numerical checks; float32 arithmetic grows when the step becomes very
small. The original failed gate is preserved. Separately, explicit model-array
typing fixes the observed same-process JIT32-to64 failure on the frozen actual
queries without changing their outputs. Finish the running arithmetic replay,
then prospectively qualify a numerically justified finite-difference assessment;
do not select whichever already-inspected step happens to pass. Preserve one
maintained learner rather than exposing precision options or research wrappers.
Mean adoption remains separate from calibrated uncertainty, update improvement,
physical derivative fidelity and controller qualification; preserve those gaps
explicitly rather than treating API semantics as evidence of their performance.

The next accuracy hypothesis is a separate fixed cohort of independent excited
training recordings across the existing primary conditions, holding the
1,536-window cache, fitting budget and development cache fixed. Additional
independent recordings test a different question from denser overlapping
windows. The current cache covers 90.38%/93.11% of eligible forecast transitions,
but adds no independent parents or coverage of the 18 shifted conditions absent
from training. Do not postpone making the current best mean usable while
pursuing that later accuracy experiment.

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
