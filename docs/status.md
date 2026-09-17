# Status: gap against the charter

Updated 2026-09-17. Accepted code is `98f77d3` on
`experiment/generic-transition-support`. The latest rejected model experiment
remains isolated on `codex/full-response-identification` at `5467943`;
its same-data comparator is `codex/independent-calibration` at `7c82db9`.
Read [the charter](charter.md) first. Git holds the experiment history;
this page records the current evidence, limitations and next named gap.

| Criterion | Current | Target |
| --- | --- | --- |
| One recipe | **Met.** One fixed `generic-memory-v3-prototype` recipe; option-free `fit`, `predict`, `update`, with saved forecast envelopes. The public structured stack remains necessary. | One generic learner and consumer contract. |
| Accuracy | **Not met.** Fresh platform-v4 measurement reproduces all ten reference metrics exactly; four of five corpora beat the structured comparator. ARP fails velocity and body rate. | Beat the structured comparator on every pinned corpus and meet each task allowance. |
| Capability | **Met.** All 27 frozen synthetic cases pass; saved models replay and reject alteration. This is a regression guard, not platform readiness. | Every synthetic absolute cap passes. |
| Control | **Not met.** Fresh control-v5 baseline: generic position RMSE 60.80/49.31 m versus structured 1.179/1.179 m; attitude 97.18/86.51° versus 1.32/1.28°. | Meet or beat the matched structured arm on every trial. |
| Live improvement | **Not met.** Last live-v3 evidence swaps at intervals 140/220; position error rises from 0.80/0.98 m before the swap to 36.1/11.8 m afterward. Not rerun today. | Bounded refits and swaps that do not worsen tracking. |
| Evidence | **Not met.** The 85–95% coverage band still fails on ARP, the reserved control recording and shifted synthetic regimes. The controller consumes the envelope, but its plan-independent cost cannot select a better command. | Measured coverage in the declared band, useful to control. |
| Lean | **Not met.** Generic research code was reduced; structured dynamics, fitting, belief code and their supporting scripts remain. | Learner, harness, telemetry adapters and controller only. |

## Current platform evidence

Whole recordings are held out; both arms forecast identical rows and commands.
Final-step RMSE is at the recipe's approximately 250 ms horizon on each sample
grid. Values below are generic / best structured comparator.

| Corpus | Velocity, m/s | Body rate, rad/s |
| --- | --- | --- |
| nanodrone | 0.136 / 0.179 | 0.543 / 0.597 |
| x8 | 0.222 / 0.287 | 0.132 / 0.187 |
| idf | 0.158 / 0.554 | 0.122 / 0.174 |
| epfl | 0.146 / 0.526 | 0.070 / 0.217 |
| arp | **0.176 / 0.174** | **0.715 / 0.285** |

All declared task allowances hold. ARP still loses to hold-current as well
(0.149 m/s, 0.362 rad/s). Acceptance means no frozen regression, not that the
charter target is met. The active protocols are [synthetic v1](harness/v1.json),
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

## Latest iteration: rejected

The full-response identification objective was frozen at `10627a1`, with a
physical-unit clarification at `700d8ab`, before fitting implementation
`5467943`. It adds one full-model, one-step residual assignment-moment term to
the existing multistep loss. It uses the raw pre-clipping randomized assignment
on the existing forecast origins; architecture, optimizer, seeds, data, splits
and numerical gates are unchanged. Training and development projections are
separate; the reserved recording never selects the checkpoint. The synthetic
gate passes all 27 cases with exact numerical parity to the accepted learner
where assignment metadata is absent. Control rejects: generic position RMSE
is **89.89/102.62 m** and attitude **107.58/114.73°**, versus the accepted
baseline's **60.80/49.31 m** and **97.18/86.51°**. The structured arm is
unchanged from the same-data comparator at **0.819/0.815 m** and **1.20/1.18°**.
Four control-reference checks, four coverage-excess regression checks and one
new gating coverage-band check fail; there are no structural breaches. No
platform or live candidate run followed the rejection. The model is not merged.

Matched saved-model diagnostics isolate the objective's effect from the prior
calibration change. Against `7c82db9` on identical recordings, windows and
normalizers, assignment-moment energy falls **30.0/2.3/5.4%** on
training/development/reserved data. One-step normalized MSE improves by less
than 0.4%, but multistep MSE worsens **0.84/0.64/0.25%**; even the new combined
objective is slightly worse on every split. Both models select step 100. At
the identical steady-input query, 250 ms roll/pitch drift improves slightly
from -0.0802/+0.0843 to -0.0788/+0.0813 rad/s, while vertical-velocity drift
worsens from -0.0351 to -0.0398 m/s. These are model diagnostics, not measured
plant derivatives or proof of what caused the tracking failure.

All 54 synthetic artifact replays and four control trial replays pass. Altered
model, requested assignment, diagnostic and frozen-plan copies are rejected,
including forged file hashes. Focused objective, data/archive, harness and
evidence tests pass; lint and formatting pass. The 1,183-test count above
belongs to the accepted repairs, not a new full-suite run. The latest candidate
retains its separate v4 archive format and control-v7/live-v5 contracts only
in its worktree.

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
`full-response-identification-control`. Verify/tamper reports sit beside their
run directories. Platform and control runs also live on ryserv under
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
```

## Next named gap

**Control: multistep forecast fidelity beyond unconditional one-step assignment
moments.** The objective reduced assignment-correlated residuals without
improving error growth over the controller's horizon. The next iteration must
address coupled forecast error at steady inputs and under command changes,
while preserving nonlinear, state-dependent behavior. No further mechanism,
threshold change or parameter sweep has been selected. This result does not
establish a need for a platform catalog. Accuracy, live improvement, evidence
and lean remain open.
