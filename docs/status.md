# Status: gap against the charter

Updated 2026-09-18. Read [the charter](charter.md) first. The generic approach is
adopted; the public recipe remains `generic-memory-v3-prototype`. Current research
has demonstrated better learned means, but public adoption and application
qualification remain separate steps. Git and frozen result records retain history.

**Current priority:** reduce physical forecast and command-response residuals
across Crazyflow and Cascade. The latest fixed initial-channel objective preserves
useful angular initialization and has the lowest aggregate errors among the four
matched learned models. Its Crazyflow optimizer still fails to improve the
initialized training objective at every recorded later checkpoint. Repair that
optimization gap while retaining the angular gains and reducing velocity/rotation
tails. JSBSim breadth work remains deferred.

| Criterion | Current | Target |
| --- | --- | --- |
| One recipe | **Met.** Public `fit` returns `LearnedDynamics` with `predict` and `update`; no consumer tuning options or platform dispatch. | One generic learner and consumer contract. |
| Accuracy | **Broad gains verified; optimization and tails unresolved.** Balanced/public weighted forecast/response ratios are 0.75267/0.67820 on the fresh cohort. Crazyflow primary 250 ms angular forecast RMSE is 0.17044 versus unchanged anchored 0.39761 rad/s. Three research-reference velocity/rotation p95 comparisons fail retention. Crazyflow selects initialization because every later recorded weighted training loss is worse. | Low held-out physical forecast and command-response errors across conditions and horizons, with progress against the adopted generic baseline; application adequacy measured separately. |
| Model usability | **Partly met.** Public saved revisions expose signal/time contracts, batched JAX-compatible forecasts, derivatives and error envelopes. Independent consumer integration and broader export/runtime portability remain unqualified. | A self-contained model artifact usable independently of the Glassbox controller, with explicit scope and measured evidence. |
| Capability | **Met for the adopted recipe.** All 27 frozen synthetic cases pass and artifacts replay/reject alteration. The new research mean has not received this public qualification. | Every synthetic absolute cap passes; synthetic results do not establish platform readiness. |
| Reference control | **Demonstrated for earlier isolated research means.** Forecast-only and paired-response means each achieved 2,248/2,248 tolerance samples across eight trials. The adopted public and latest research means have not been tested with that controller. | Separately qualified downstream demonstrations; a universal controller is optional. |
| Live improvement | **Not met.** No update qualification under the clarified model-first priorities. Historical live-v3 swaps at intervals 140/220 worsened position error from 0.80/0.98 to 36.1/11.8 m. | Bounded immutable revisions with held-out improvement/regression checks; adoption and any live-control claim qualified separately. |
| Evidence | **Not met.** Original public primary 250 ms velocity/rate coverage is 85.8–89.7%, falling to 47.4–68.3% for extreme maneuvers. Crazyflow truth is incomplete. Development-calibrated research spreads have no new coverage qualification. Earlier ARP, reserved-control and shifted-synthetic failures remain. | Measured coverage in a predeclared band with calibration provenance, independent of the controller. |
| Lean | **Not met.** Structured dynamics, belief code and research scripts remain. | Learner, model artifacts/interfaces, harness, telemetry adapters and optional consumers. |

## Latest matched evidence

[Initial-channel balance v1](harness/initial-channel-balance-v1-result.json)
changes only the per-output objective weights used for optimization and checkpoint
selection. They are fixed inverse initial-training per-channel normalized mean squared
residuals, floored
at `hold_scale_floor ** 2` and normalized to mean one. The architecture, canonical
anchored initializer, original hold scales, exact 384/256 training/development
windows, minibatches and 1,000 proposal updates are unchanged. Physical labels
never enter the learner. No floor activates in this experiment.

Only two candidates are newly fitted; six exact historical reference fits are
imported with their original provenance. The four selected models and hold baseline
predict the same fresh queries. Weighted candidate/reference forecast and response
ratios are:

| Reference | Forecast | Response |
| --- | ---: | ---: |
| Public-excited | 0.75267 | 0.67820 |
| Unchanged anchored | 0.92277 | 0.87839 |
| Unchanged quadratic | 0.96260 | 0.89291 |

Public progress and targeted angular repair pass. All aggregate, simulator-primary
and scope limits pass. Combined mechanism qualification fails only these
Crazyflow primary 250 ms factual parent-RMSE p95 retention limits (cap 1.5):
rotation/anchored **1.787**, velocity/quadratic **1.783**, rotation/quadratic **2.124**.
Those losses coexist with strong public-relative gains; they remain improvement
work rather than evidence against the generic approach. Public recipe promotion
is false, and no threshold is changed retrospectively.

Representative primary 250 ms physical RMSE, public → balanced:

| Simulator | Forecast velocity, m/s | Forecast body rate, rad/s | Response velocity, m/s | Response body rate, rad/s |
| --- | ---: | ---: | ---: | ---: |
| Crazyflow, available truth | 0.17871 → 0.12435 | 0.21332 → 0.17044 | 0.11234 → 0.03769 | 0.22608 → 0.15220 |
| Cascade | 0.16052 → 0.12803 | 0.10427 → 0.10703 | 0.04918 → 0.03735 | 0.03748 → 0.03860 |

Crazyflow completes **64/84** test parents; 20 fail the altitude condition.
Primary 250 ms truth is **413/480** factual and **664/768** response queries.
Cascade completes **84/84** with all truth. Every eligible prediction is finite.
All comparisons use matched truth masks. Accuracy is conditional on available
Crazyflow truth, and cohort-completion differences are not learner gains.
The incomplete original excited training pool is preserved.

Both fresh full replays pass: **4,032** prediction queries and **428,400** metric
rows reproduce exactly. The source-bound observer captures **22** new snapshots;
**66** imported snapshots keep their historical meaning. Exact initial arrays,
weight construction and source/cache/selection identities replay without fitting
or re-solving initialization. **784** focused checks and Ruff pass. The result
record contains the independent reductions, alteration challenges, auxiliary
attempts, hashes and scope. See the [guide](initial-channel-balance.md) for complete
physical tables, tails, figures and reproduction commands.

## What the current diagnosis establishes

Crazyflow selects **step zero**, exactly the retained anchored initialization.
Full-cache weighted training loss is **0.00328876** initially, **0.01138641** at
100 proposals, **0.00442898** at 900 and **0.00498218** at 1,000. Every recorded
postinitial value is worse. Weighted development loss likewise favors zero;
angular development RMSE worsens **0.188526→0.401337 rad/s** by 1,000. Thus its
held-out gain comes from retaining initialization, not useful optimized progress.
Development overfitting alone does not explain a worse training objective.

This does not show that every unrecorded update worsens loss, that step size is
the sole cause, or that improved training loss guarantees repaired held-out tails.
The original-loss minimum at step 900 on the weighted trajectory is descriptive
only; no alternate checkpoint was tested. Cascade does improve: weighted training
loss **0.01978948→0.00590152**, development **0.02894970→0.01949018**, selected at
1,000. Initial equal per-coordinate losses still give physical groups 3:3:9 votes.

## Next named gap

**Make useful recursive-loss progress from the retained initializer.** The
[safeguarded Adam protocol](harness/safeguarded-adam-v1.json) is now frozen for
the next iteration, before implementation or fitting. It isolates bounded
full-training-loss backtracking around the existing Adam proposals. Preserve the model, fixed channel weights, initialization, recordings,
cache sizes, minibatch draws, clipped gradients, moment formulas and development
selection. Test whether a bounded acceptance check on all training windows can
produce useful progress while retaining angular accuracy and repairing tails.

The prospective design is 1,000 proposal attempts, each with the existing
64-window gradient. Try a fixed halving ladder from the existing proposal toward
the current parameters; accept the first finite strict full-training-loss decrease,
otherwise retain the current parameters. Advance moments/counter once per attempt,
including rejection, and disclose this policy. Freeze the exact arithmetic,
ladder, evaluation budget, safeguards, evidence and physical progress criteria
before implementation or fitting. No seed or learning-rate sweep is authorized
by this hypothesis.

Use the saved balanced model as the direct control, retain the public and relevant
quadratic/anchored references, and evaluate selected revisions on one fresh common
cohort. Require useful progress beyond preserving initialization. Extra full-cache
objective evaluations are an explicit compute intervention; equal accepted updates
or equal FLOPs must not be claimed. Record accepted/rejected attempts and scales.
Backtracking can still stall if a stochastic Adam direction fails to decrease the
full objective at every tested scale. That would motivate a separate change to
how directions are computed, rather than an unplanned restart or parameter sweep.

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
