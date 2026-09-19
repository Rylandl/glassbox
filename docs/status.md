# Status: gap against the charter

Updated 2026-09-18. Read [the charter](charter.md) first. The generic approach is
adopted; the public recipe remains `generic-memory-v3-prototype`. Current research
has demonstrated better learned means, but public adoption and application
qualification remain separate steps. Git and frozen result records retain history.

**Current priority:** reduce physical forecast and command-response residuals
across Crazyflow and Cascade. The balanced model remains the strongest aggregate
research result. Safeguarded Adam improves Crazyflow training loss but regresses
held-out forecast/response errors by 6.29%/15.37% relative to that model. Next test
whether gradients over all cached training windows produce better update directions;
keep the objective fixed and require physical gains. JSBSim breadth remains deferred.

| Criterion | Current | Target |
| --- | --- | --- |
| One recipe | **Met.** Public `fit` returns `LearnedDynamics` with `predict` and `update`; no consumer tuning options or platform dispatch. | One generic learner and consumer contract. |
| Accuracy | **Broad gains verified; latest optimizer rejected.** Balanced/public forecast/response ratios were 0.75267/0.67820 on its frozen cohort. On the new cohort, safeguarded/balanced ratios are 1.06289/1.15371. Crazyflow training improves 45.5%, development only 4.58%, and forecast rate error worsens. Cascade loses against the balanced model. | Low held-out physical forecast and command-response errors across conditions and horizons, with progress against the adopted generic baseline; application adequacy measured separately. |
| Model usability | **Partly met.** Public saved revisions expose signal/time contracts, batched JAX-compatible forecasts, derivatives and error envelopes. Independent consumer integration and broader export/runtime portability remain unqualified. | A self-contained model artifact usable independently of the Glassbox controller, with explicit scope and measured evidence. |
| Capability | **Met for the adopted recipe.** All 27 frozen synthetic cases pass and artifacts replay/reject alteration. The new research mean has not received this public qualification. | Every synthetic absolute cap passes; synthetic results do not establish platform readiness. |
| Reference control | **Demonstrated for earlier isolated research means.** Forecast-only and paired-response means each achieved 2,248/2,248 tolerance samples across eight trials. The adopted public and latest research means have not been tested with that controller. | Separately qualified downstream demonstrations; a universal controller is optional. |
| Live improvement | **Not met.** No update qualification under the clarified model-first priorities. Historical live-v3 swaps at intervals 140/220 worsened position error from 0.80/0.98 to 36.1/11.8 m. | Bounded immutable revisions with held-out improvement/regression checks; adoption and any live-control claim qualified separately. |
| Evidence | **Not met.** Original public primary 250 ms velocity/rate coverage is 85.8–89.7%, falling to 47.4–68.3% for extreme maneuvers. Crazyflow truth is incomplete. Development-calibrated research spreads have no new coverage qualification. Earlier ARP, reserved-control and shifted-synthetic failures remain. | Measured coverage in a predeclared band with calibration provenance, independent of the controller. |
| Lean | **Not met.** Structured dynamics, belief code and research scripts remain. | Learner, model artifacts/interfaces, harness, telemetry adapters and optional consumers. |

## Latest matched evidence

[Safeguarded Adam v1](harness/safeguarded-adam-v1-result.json) changes only
acceptance of the existing Adam proposals: try a fixed halving ladder using the
full-training weighted objective, accepting the first strict decrease. Preserve
initialization, weights, caches, architecture, 1,000 proposals and development
selection. Two new fits and six exact imported reference fits predict the same
fresh queries. Additional full-cache evaluations are an explicit compute change.

| Reference | Candidate forecast ratio | Candidate response ratio |
| --- | ---: | ---: |
| Retained balanced model | 1.06289 | 1.15371 |
| Public-excited | 0.79796 | 0.78958 |
| Quadratic | 1.01890 | 1.04030 |

All three required comparison gates fail. The balanced comparison loses on
aggregate and simulator-primary guards. Public-relative aggregate gains pass but
Cascade rate tails fail. Quadratic-relative aggregates pass but primary, scope
and tail guards fail. Crazyflow angular forecast mean/p95 retention fails at
**1.10808/1.19136**; response retention passes. The separate optimization diagnostic
also fails. This specific optimizer is rejected; public promotion remains false.

Primary 250 ms physical RMSE, balanced → safeguarded:

| Simulator | Forecast velocity, m/s | Forecast rate, rad/s | Response velocity, m/s | Response rate, rad/s |
| --- | ---: | ---: | ---: | ---: |
| Crazyflow, available truth | 0.10959 → 0.10989 | 0.15447 → 0.17116 | 0.03886 → 0.03866 | 0.16598 → 0.16801 |
| Cascade | 0.13334 → 0.15885 | 0.10421 → 0.13498 | 0.03665 → 0.04462 | 0.04182 → 0.06003 |

Crazyflow completes **68/84** test parents; 16 fail the altitude condition.
Primary 250 ms truth is **442/480** factual and **720/768** response queries.
Cascade completes **84/84** with all truth. Every eligible prediction is finite
for all five models. Accuracy is conditional on available truth with matched
masks. Different cohort completion is not learner progress. The original
incomplete excited training pool and reference provenance remain unchanged.

Independent reductions verify all **1,050** endpoint groups, actual initial
identity, weights, **2,000** observed proposal attempts and **44** new-candidate checkpoint
diagnostic sets. **896** focused checks and Ruff pass. The
[result record](harness/safeguarded-adam-v1-result.json) contains final replay,
alteration challenges, hashes and preserved auxiliary attempts; the
[guide](safeguarded-adam.md) provides physical tables, figures and reproduction.

## What the current diagnosis establishes

Crazyflow now selects attempt **1,000**, not initialization. It accepts **717**
proposals and rejects **283**, using **7,529** full-training objective calls.
Selected training loss falls **0.00328876→0.00179187**; development falls only
**0.00400372→0.00382020**, a 4.58% reduction below the frozen 5% diagnostic margin.
Every training physical group improves, but development 250 ms rate error barely
moves **0.188526→0.187567 rad/s** and held-out rate error grows. The physical
regressions are broader than this narrow diagnostic miss.

Cascade selects attempt **900**, accepting **931** proposals and rejecting **69**
across the full 1,000 attempts, with **4,427** objective calls. Selected training/
development losses are **0.010024/0.027485**, both worse than the retained balanced
model's **0.005902/0.019490**. Development rate error increases from initialization
**0.143452→0.156553 rad/s** despite a lower weighted objective. Neither simulator
accepts a full-scale Adam proposal. Tiny scales do not prove minibatch noise;
curvature and accumulated moments could also explain them.

The prior [initial-channel balance result](harness/initial-channel-balance-v1-result.json)
remains the strongest aggregate research model: forecast/response errors were
24.7%/32.2% below public on its own matched cohort. Its three failed research tail
retention limits and lack of public qualification remain unchanged. Latest
safeguard evidence is integrated for diagnosis, not as a replacement model.

## Next named gap

**Test proposal-direction quality without changing the objective.** Use all
384 cached training windows for each gradient instead of a 64-window sample.
Preserve the retained initialization, fixed channel weights, Adam moments and
clipping, 1,000 attempts, bounded acceptance ladder and development selection.
Freeze exact arithmetic, compute and time budgets, evidence, comparisons and
physical progress criteria before implementation or fitting. Do not add a
learning-rate, seed or batch-size sweep.

Import both the safeguarded model as the mechanism control and the balanced model
as the performance reference, retaining public context. Use one fresh common
cohort and the original training/development caches. Full-window gradients cost
more; same proposal count does not mean equal compute. Require useful held-out
forecast/response progress, not just monotonic training loss. If stronger training
progress still does not transfer, move the next named gap toward data support or
model generalization instead of extending optimization blindly.

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
  neither qualifies the current public or balanced model.
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
