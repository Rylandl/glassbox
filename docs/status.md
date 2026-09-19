# Status: gap against the charter

Updated 2026-09-19. Read [the charter](charter.md) first. The generic approach is
adopted; the public recipe remains `generic-memory-v3-prototype`. Research
accuracy, public adoption and application qualification remain separate.

**Current priority:** reduce physical forecast and command-response residuals
across Crazyflow and Cascade. Retain the balanced research model. Full-cache
gradients solve the current training objective much better but fail to transfer;
next investigate unused training-recording support, not further optimizer tuning.
JSBSim breadth remains deferred.

| Criterion | Current | Target |
| --- | --- | --- |
| One recipe | **Met.** Public `fit` returns `LearnedDynamics` with `predict` and `update`; no consumer tuning options or platform dispatch. | One generic learner and consumer contract. |
| Accuracy | **Broad gains verified; full-cache optimizer rejected.** Balanced remains strongest. On the fresh cohort full-cache/balanced forecast/response ratios are 1.09896/1.13337 despite more than 90% lower training loss from initialization. Crazyflow angular harm spans all 24 primary conditions. | Low held-out physical forecast and command-response errors across conditions and horizons, with progress against the adopted generic baseline; application adequacy measured separately. |
| Model usability | **Partly met.** Public saved revisions expose signal/time contracts, batched JAX-compatible forecasts, derivatives and error envelopes. Independent consumer integration and broader export/runtime portability remain unqualified. | A self-contained model artifact usable independently of the Glassbox controller, with explicit scope and measured evidence. |
| Capability | **Met for the adopted recipe.** All 27 frozen synthetic cases pass and artifacts replay/reject alteration. The new research mean has not received this public qualification. | Every synthetic absolute cap passes; synthetic results do not establish platform readiness. |
| Reference control | **Demonstrated for earlier isolated research means.** Forecast-only and paired-response means each achieved 2,248/2,248 tolerance samples across eight trials. The adopted public and latest research means have not been tested with that controller. | Separately qualified downstream demonstrations; a universal controller is optional. |
| Live improvement | **Not met.** No update qualification under the clarified model-first priorities. Historical live-v3 swaps at intervals 140/220 worsened position error from 0.80/0.98 to 36.1/11.8 m. | Bounded immutable revisions with held-out improvement/regression checks; adoption and any live-control claim qualified separately. |
| Evidence | **Not met.** Original public primary 250 ms velocity/rate coverage is 85.8–89.7%, falling to 47.4–68.3% for extreme maneuvers. Crazyflow truth is incomplete. Development-calibrated research spreads have no new coverage qualification. Earlier ARP, reserved-control and shifted-synthetic failures remain. | Measured coverage in a predeclared band with calibration provenance, independent of the controller. |
| Lean | **Not met.** Structured dynamics, belief code and research scripts remain. | Learner, model artifacts/interfaces, harness, telemetry adapters and optional consumers. |

## Latest matched evidence

[Full-cache gradients v1](harness/full-cache-gradient-v1-result.json) changes
only gradient coverage: all 384 cached training windows once per proposal,
instead of 64 sampled windows. The objective, actual initialization, weights,
architecture, safeguards and development selector stay fixed. Two fresh fits
and six exact imported reference fits predict one fresh common cohort.
Gradient-window work is six times the safeguarded control; equal compute is
not claimed.

| Reference | Candidate forecast ratio | Candidate response ratio |
| --- | ---: | ---: |
| Retained balanced model | 1.09896 | 1.13337 |
| Safeguarded mechanism control | 1.02532 | 0.99877 |
| Public-excited | 0.83109 | 0.78677 |

Public-relative physical criteria pass. Balanced, safeguarded-control and
Crazyflow angular retention criteria fail. The separate optimization diagnostic
is descriptive and does not veto physical progress. Retain balanced; public
promotion remains false. See the [guide](full-cache-gradient.md) for tables,
figures, limitations and reproduction.

Primary 250 ms component RMSE, balanced → full-cache:

| Simulator | Forecast velocity, m/s | Forecast rate, rad/s | Response velocity, m/s | Response rate, rad/s |
| --- | ---: | ---: | ---: | ---: |
| Crazyflow, available truth | 0.11725 → 0.09474 | 0.14626 → 0.23795 | 0.03938 → 0.03610 | 0.16239 → 0.18974 |
| Cascade | 0.13485 → 0.15047 | 0.10790 → 0.12828 | 0.03882 → 0.04378 | 0.04255 → 0.05285 |

Crazyflow completes 62/84 test parents; 22 fail the altitude condition. Primary
250 ms truth is 423/480 forecast and 704/768 response queries. Cascade completes
84/84 with all truth. Every eligible prediction is finite for all five models.
Comparisons have matched masks; changes in cohort completion are not model gains.
No quadratic reference was evaluated on this cohort.

Training loss falls 90.54% in Crazyflow and 92.60% in Cascade. Crazyflow's
training rate RMSE improves 0.128813→0.050295 rad/s, while development rate
worsens 0.188526→0.256304 rad/s. Other development channels offset this regression
in the weighted selector. Held-out factual rate worsens in all 24 primary
conditions and 41/42 conditions overall. Cascade remains worse than balanced
in all primary 250 ms physical groups. This exposes a generalization gap; further
training-loss optimization alone is not the next justified mechanism.

The strongest retained research result remains
[initial-channel balance](harness/initial-channel-balance-v1-result.json): on
its own cohort forecast/response errors were 24.7%/32.2% below public. Its three
failed research tail-retention limits and missing public qualification remain.
Verification is complete: 1,013 focused tests, both full simulator replays,
independent physical/optimizer reductions and all 36 alteration challenges pass.

## Next named gap

**Test whether using more admitted training data reduces the generalization gap.**
The 384-window cache uses only 384/12,426 legal origins in Crazyflow and 384/3,312
in Cascade. Its forecast targets cover 7,189/14,154 and 1,589/3,600 unique eligible
native transitions, respectively. Every training parent appears, but each
contributes only five or six windows. A larger fixed cache is therefore a
concrete support intervention, not a claim that more optimizer steps will help.

Freeze one cache expansion from 384 to 1,536 windows on the existing admitted
training recordings, preserving the original 384-window prefix.
Keep development windows and role boundaries unchanged, retain the model and
objective formulas, and declare changed data-derived norms, initialization,
weights and fourfold gradient-window work. Compare physical errors, not raw
weighted losses across arms whose numerical objective weights differ. Compare
against the saved 384-window full-cache control, retained balanced model and
public context on another fresh common
cohort. Require meaningful physical gains with predeclared regression limits.
No batch/seed/cache-size sweep and no public option.

The [expanded-cache protocol](harness/expanded-training-cache-v1.json) is now
frozen before implementation and fitting. Preparation feasibility and three
independent scope reviews pass. The implementation passes 1,135 focused tests across 41 files; all 183 source
pins and 14,758 prior payloads verify. Fresh numerical evaluation remains pending;
no accuracy gain or public promotion is claimed.

Larger caches add no independent recordings. They cannot fill the 18 shifted
conditions absent from training, repair incomplete Crazyflow recording support,
or make overlapping windows independent. Use these inspected results only for
diagnosis; the next confirmation cohort must remain untouched until frozen.

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
