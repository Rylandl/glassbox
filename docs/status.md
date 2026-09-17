# Status: gap against the charter

Updated 2026-09-17. **The generic approach is adopted as the development
baseline.** The user prioritizes generality over superiority on every corpus;
the four-corpus advantage justifies accepting the known ARP deficit. This is
an explicit policy decision on existing evidence, not a new experimental
pass. The adopted learner remains `generic-memory-v3-prototype`. Its recipe
and numerical logic from `98f77d3` now serve the public `glassbox.fit` API;
the returned `LearnedDynamics` provides `predict` and `update`. Generic
recording archives also serve the public fit/evaluate commands. Structured
code remains in its owning modules for benchmarks and unmigrated control
consumers.
The latest rejected learner remains isolated on `codex/full-response-identification`
at `5467943`, with same-data comparator `codex/independent-calibration` at `7c82db9`.
Read [the charter](charter.md) first. Git holds the experiment history;
this page records the current evidence, limitations and next named gap.

| Criterion | Current | Target |
| --- | --- | --- |
| One recipe | **Met.** The public API and fit/evaluate commands use the single `generic-memory-v3-prototype` recipe. `fit`, `predict` and `update` have no tuning or model-selection options. Replaced experimental module paths and structured root exports are removed. | One generic learner and consumer contract. |
| Accuracy | **Adopted with a known tradeoff.** Four of five corpora beat the structured comparator on both metrics, including under the corrected prefix-percentile readout. ARP loses both; this does not block adoption. | Broad competitive performance from one recipe; improve weak cases and establish task sufficiency separately. |
| Capability | **Met.** All 27 frozen synthetic cases pass; saved models replay and reject alteration. This is a regression guard, not platform readiness. | Every synthetic absolute cap passes. |
| Control | **Not met.** Generic position RMSE 60.80/49.31 m versus structured 1.179/1.179 m. This compares complete pipelines: generic plans 250 ms, structured 800 ms. Both structured trials also fail the application tracking criterion. | Meet the declared application tracking requirement; use structured and oracle arms diagnostically. |
| Live improvement | **Not met.** Last live-v3 evidence swaps at intervals 140/220; position error rises from 0.80/0.98 m before the swap to 36.1/11.8 m afterward. Not rerun today. | Bounded refits and swaps that do not worsen tracking. |
| Evidence | **Not met.** The 85–95% coverage band still fails on ARP, the reserved control recording and shifted synthetic regimes. Constant spread cannot rank command plans, but can affect finite-iteration stopping through the absolute objective. | Measured coverage in the declared band, useful to control. |
| Lean | **Not met.** Generic research code was reduced; structured dynamics, fitting, belief code and their supporting scripts remain. | Learner, harness, telemetry adapters and controller only. |

## Current platform evidence

Whole recordings are held out; both arms forecast identical rows and commands.
Final-step RMSE is at the recipe's approximately 250 ms horizon on each sample
grid. Values below are generic / best structured comparator per metric;
the best structured arm can differ between metrics, so this benchmark envelope
is not itself one deployable model. One generic recipe is fitted separately to
each system; these are not shared weights applied unseen to five systems.

| Corpus | Velocity, m/s | Body rate, rad/s |
| --- | --- | --- |
| nanodrone | 0.136 / 0.179 | 0.543 / 0.597 |
| x8 | 0.222 / 0.287 | 0.132 / 0.187 |
| idf | 0.158 / 0.554 | 0.122 / 0.174 |
| epfl | 0.146 / 0.526 | 0.070 / 0.217 |
| arp | **0.176 / 0.174** | **0.715 / 0.285** |

These are pooled endpoint component RMSEs. The old platform ceilings came from
descriptive model-error percentiles, not application requirements, and used a
different statistic. On the original statistic (worst recording's nearest-rank
95th percentile of maximum whole-prefix vector error), X8 body rate is
**0.888 rad/s versus the historical 0.764 ceiling**. IDF and EPFL have no declared
ceiling. No task sufficiency follows from these comparisons. ARP also loses to
hold-current (endpoint 0.149 m/s, 0.362 rad/s).

The generic recipe wins both metrics on four corpora and the structured
comparators win on ARP; neither dominates everywhere. Generality and breadth
of measured advantage justify adopting the generic recipe with that known
loss. ARP body-rate endpoint RMSE is about 2.5 times the structured value;
the loss remains visible rather than disappearing into a win count. These
five corpora do not establish performance on arbitrary systems. Future
comparisons start from the adopted generic baseline. The next iteration
must freeze its own promotion criteria; this decision does not create a
permanent four-of-five rule or invent aggregate weights from known scores.

The existing protocols retain their original historical acceptance meaning
and scores, including their per-case no-regression checks; they no longer
decide whether to adopt the generic approach. The frozen protocols are
[synthetic v1](harness/v1.json),
[platform v4](harness/platform-v4.json), [control v5](harness/control-v5.json),
[live v3](harness/live-v3.json) and [evidence v2](harness/evidence-v2.json).

## Accepted repairs and validation

**Slow sample grids:** freeze `52e707a`, implementation `aacd43b`. Consumed
context now retains one memory step beyond explicit delay, including at
500 ms sampling. Fit, prediction, update, saved replay and altered-artifact
rejection pass. Previously valid sample grids follow the same numerical path;
the recipe remains v3. The synthetic run accepts 27/27 cases with every model,
score and coverage value unchanged, and 54 replays differ by at most `8.9e-16`.
Its existing 253 coverage-band breaches remain; none is a new regression.

**Canonical recording identity:** freeze `5e06325`, implementation `56a9f98`,
accepted evidence `98f77d3`. Platform v4 pins content and labels for all 162
recordings, checks all corpora before any fit, and passes immutable loaded
trajectories to both fitters. Changed content, labels, filenames and source
files after loading cannot silently change the experiment. The five-corpus run
accepts with exact reference parity; 17 artifact replays pass with maximum
difference `5.0e-14`. Modified inventory and model copies are rejected.

The fresh control-v5 baseline ran at `56a9f98`, whose executable code is the
accepted branch's code. The generic model fingerprint and every tracking
metric exactly reproduce the old baseline; four trial replays pass. Both bug fixes
repair correctness without improving the unresolved model-adequacy metrics.
Validation after the repairs: 1,183 tests passed, three skipped and 24
slow/Cascade/PX4-SITL tests deselected; lint and formatting passed.

## Latest iteration: public API adoption

The migration contract was frozen at `1f54940`, implementation `12d6df7`.
The learner now lives in `glassbox.learner`, with recording types in
`glassbox.recordings` and numerical
helpers kept private. Its recipe, artifact format and numerical logic are
unchanged. Python and CLI consumers use the generic path; the CLI reads
fingerprinted recording archives, and an explicit telemetry adapter preserves
recording boundaries and channel identities. Evaluation scores untouched
recordings without fitting. Structured benchmark consumers import their owning
modules instead of the replaced public API.

The fixed same-host fit/save/load/update experiment produces exactly the same
models, fingerprints, reports, caches, envelopes and forecasts before and after
migration. All 33 pinned generic artifacts load with unchanged identity, and
all 60 public forecast datasets match their saved values exactly. Independent
historical replay retains the original synthetic and platform decisions:
27 synthetic cases / 54 replays and five corpora / 17 replays. These checks
establish API migration parity, not improved accuracy or task readiness.

The telemetry adapter normalizes timestamp roundoff when inferring a sample
grid and carries command units, frames and semantics in its channel identities.
Explicit generic sample intervals remain exact. Renamed or resegmented data
are not proof of independence: the evaluator reports its ID/content checks
without claiming independent recordings. The walkthrough exercises the same
public API and CLI, including comparisons on common untouched rows after an
update.

Validation: **1,287 tests pass**, three skip and 27 slow/Cascade/PX4-SITL tests
are deselected (756.50 seconds). The runnable onboarding example and public CLI
fit/evaluate smoke checks pass.
Saved replay rejects both a forged summary and an altered forecast even when
their outer hashes are recomputed. Ruff lint and formatting checks pass.
This iteration does not rerun the slow benchmark fits, Cascade control trials
or PX4 SITL tests.

## Control qualification evidence

The no-fit protocol was frozen at `f844d2f`. It pins 32 existing input files
and the historical source contracts, preserves every old gate, and separates
historical no-regression, comparative progress and application success. Its
retrospective report recomputes the original per-recording prefix statistic
for every saved platform predictor. The four-versus-one result is explicitly
symmetric: neither predictor dominates everywhere, and the win count alone
ignores the sizes of the gains and losses. This iteration chooses no aggregate
promotion weights from previously inspected scores.

The prospective test substitutes public Cascade equations for the generic
observed-channel mean, keeping its 250 ms horizon, five command blocks, four
solver iterations, two warm-up holds, state reconstruction, task and borrowed
covariance offset. The oracle's hidden state is initialized at the declared
actuator equilibrium and advanced only by issued commands. A second arm uses
the saved structured model at the same horizon and warm-up, retaining its own
adapter and support penalty; that remains a contextual comparison. All four
trials complete. Oracle position component RMSE is **0.565/0.457 m**, versus
**60.80/49.31 m** for the saved generic arm. The oracle meets the actual
±0.5 m lateral-and-altitude criterion in only **120/281 and 73/281 samples**
(**42.7/26.0%**, required 95%). Its lateral RMSE is **0.937/0.725 m** and
altitude RMSE **0.215/0.185 m**. The structured model at 250 ms scores
**1.816/1.795 m**, with **9/281 and 0/281** qualifying samples.

The learner's mean remains a major measured weakness, but accurate observed
predictions through the retained state map are insufficient for this
controller to meet the task. The historical 800 ms structured comparator
also misses the task (0/281 qualifying samples in both trials). These are two
previously declared initial conditions in one deterministic simulator, not
an estimate of performance on arbitrary systems. No learner was fitted or
selected, and no old acceptance decision changed.

The four trials ran at `0dd094b`; verifier `2500607` checks the same saved
artifacts without fitting or rerunning trials. All 47 qualification tests pass
in the pinned Linux environment. Four plant trajectories replay exactly;
636 oracle forecasts and 636 optimizer solves verify, with exact reproduced
objectives. Seven altered prospective artifact copies and three altered
retrospective copies are rejected, including forged outer hashes. The
verifier corrections preserve every frozen numerical tolerance. All 15
pinned inherited source files and the original numerical gates are unchanged.

The last learner candidate remains rejected at `5467943`: the one-step
assignment-moment objective reduced assignment-correlated errors but worsened
multistep prediction and control. Its source remains isolated. This evaluation
iteration does not reverse that result or call the generic approach a failure.

## Constraints established by prior measurements

- ARP's error compounds across the forecast and transfers poorly between
  recordings. Optimizer changes and pooled shrinkage did not repair it without
  regressions; more training steps alone are not an established remedy.
- Closed-loop forecast accuracy does not establish usable command response.
  Holding affine command columns does not hold the full model's derivative:
  nonlinear and memory paths can reverse it. The affine-column constraint
  broke nonlinear synthetic cases while still permitting incorrect full-model
  command derivatives.
- Smooth dither in short live blocks supplied weak identifying variation.
  Independent signs increased useful command variation but failed the control
  gate (105.05/101.91 m position RMSE at `7c82db9`). Realized pitch injection
  retained only 70.7–72.9% of requested RMS because 45.6–48.1% of intervals
  clipped. Preserve raw assignment separately from applied commands.
- The full-response moment assumes sequentially zero-mean assignment and an
  adequate conditional state/mean model. Three unconditional assignment
  directions and two training recordings do not establish state-dependent
  causal response. Reducing these moments alone did not repair multistep
  prediction. Calibration and state-dependent identification are not exhausted.
- Widening envelopes solely by distance repaired some unsupported cases while
  over-covering others. Neither the coverage band nor any reference is relaxed.

## Evidence and replay

Local runs live under `artifacts/2026-09-17/`: `slow-sampling`, `platform-pins`,
`control-baseline-fixed`, `independent-calibration-control`,
`full-response-identification-synthetic` and
`full-response-identification-control`, `evaluation-reference-audit` and
`evaluation-qualification-fixed`. Verify/tamper reports sit beside their run
directories. The earlier `evaluation-qualification` attempt stopped on a
numerical construction mismatch and supplies no accepted outcome. Platform
and control runs also live on ryserv under
`/home/ryland/autonomy/glassbox-evidence/2026-09-17/`.
Use the original Linux environment for authoritative control replay; existing
sine regeneration has last-bit libm differences on macOS. Read-only model
diagnostics and their scripts are in `calibration-response-diagnostic`,
`command-moment-diagnostic` and `assignment-objective-diagnostic` under the
same local artifact root.

Public migration evidence is in `generic-public-api-baseline`,
`generic-public-api-candidate`, `generic-public-api-replay` and
`generic-public-onboarding` under the same dated root. Comparison, source-audit
and verification reports sit beside them. The baseline capture ran on
`3ab800f`; the public implementation is `12d6df7`. Neither replay requires
refitting the learner.

From disposable checkouts of the matching source commits, without refitting:

```sh
# Accepted runs: checkout 98f77d3 (56a9f98 also matches the control baseline).
PYTHONPATH=src python -m glassbox.experimental.harness verify /absolute/path/to/slow-sampling
PYTHONPATH=src python -m glassbox.experimental.harness verify /absolute/path/to/platform-pins
PYTHONPATH=src python -m glassbox.experimental.harness verify /absolute/path/to/control-baseline-fixed
# Rejected experiment: checkout 7c82db9; its control-v6 contract differs.
PYTHONPATH=src python -m glassbox.experimental.harness verify /absolute/path/to/independent-calibration-control
# Latest rejected experiment: checkout 5467943; its archives use recipe v4.
PYTHONPATH=src python -m glassbox.experimental.harness verify /absolute/path/to/full-response-identification-synthetic
PYTHONPATH=src python -m glassbox.experimental.harness verify /absolute/path/to/full-response-identification-control
# Qualification trials: 0dd094b; use verifier 2500607 (or this accepted merge).
# No fit; prospective replay also recomputes optimizer outputs.
PYTHONPATH=src python -m glassbox.experimental.qualification verify /absolute/path/to/evaluation-reference-audit
PYTHONPATH=src python -m glassbox.experimental.qualification verify /absolute/path/to/evaluation-qualification-fixed
# Public API migration: checkout 12d6df7 (or this accepted merge); no refit.
PYTHONPATH=src python -m glassbox.experimental.api_migration compare --baseline /absolute/path/to/generic-public-api-baseline --candidate /absolute/path/to/generic-public-api-candidate
PYTHONPATH=src python -m glassbox.experimental.api_migration verify /absolute/path/to/generic-public-api-replay --artifacts /absolute/path/to/artifacts/2026-09-17
```

## Next named gap

**Control qualification: establish task success with accurate dynamics through
the adopted generic controller.** Audit the objective and task relationship
with the oracle before another learner loss change. Horizon, finite
optimization, objective tradeoffs, state mapping and stopping behavior remain
possible contributors. The large generic-to-oracle gap separately motivates
improving the learner. This work develops the adopted approach; it does not
reimpose superiority on every corpus as an adoption requirement.
