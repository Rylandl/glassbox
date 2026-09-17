# Status: gap against the charter

Updated 2026-09-17. The accepted learner remains the code from `98f77d3` on
`experiment/generic-transition-support`; this iteration adds evaluation only.
The latest rejected learner remains isolated on `codex/full-response-identification`
at `5467943`, with same-data comparator `codex/independent-calibration` at `7c82db9`.
Read [the charter](charter.md) first. Git holds the experiment history;
this page records the current evidence, limitations and next named gap.

| Criterion | Current | Target |
| --- | --- | --- |
| One recipe | **Met.** One fixed `generic-memory-v3-prototype` recipe; option-free `fit`, `predict`, `update`, with saved forecast envelopes. The public structured stack remains necessary. | One generic learner and consumer contract. |
| Accuracy | **Broad progress; replacement across all pinned cases not met.** Four of five corpora beat the structured comparator on both metrics, including under the corrected prefix-percentile readout. ARP loses both. | Beat the structured comparator on every pinned corpus; establish task sufficiency separately. |
| Capability | **Met.** All 27 frozen synthetic cases pass; saved models replay and reject alteration. This is a regression guard, not platform readiness. | Every synthetic absolute cap passes. |
| Control | **Not met.** Generic position RMSE 60.80/49.31 m versus structured 1.179/1.179 m. This compares complete pipelines: generic plans 250 ms, structured 800 ms. Both structured trials also fail the separate application tracking criterion. | Meet or beat the structured arm on every trial; report actual task success separately. |
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
comparators win on ARP; neither dominates everywhere. Reversing the incumbent
does not change this tradeoff. Historical acceptance is a no-regression check,
while the charter's all-corpus target is replacement across all pinned cases.
No aggregate tradeoff weights have been chosen from these known scores. The active model
protocols remain [synthetic v1](harness/v1.json),
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

## Latest iteration: evaluation qualification

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
```

## Next named gap

**Control evaluation: establish task success with accurate dynamics through
the generic controller before treating task failure as a learner-only
readiness verdict.** Audit the objective and task relationship with the oracle
before another learner loss change. Horizon, finite optimization, objective
tradeoffs, state mapping and stopping behavior remain possible contributors;
this iteration does not select a remedy. The large generic-to-oracle gap
remains a separate reason to improve the learner. Comparative progress across
corpora remains visible independently of the all-pinned-cases target.
