# Status: gap against the charter

Updated 2026-09-17. Accepted code is `98f77d3` on
`experiment/generic-transition-support`. The rejected calibration experiment
remains isolated on `codex/independent-calibration` at `7c82db9`.
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

The independent-calibration protocol was frozen at `da93181` and implemented
at `7c82db9`. It replaced only calibration sine perturbations with independent
signs at each old waveform's requested RMS, preserved the learner and live
trial waveform, recorded requested versus realized clipped injection, and
added safeguards against degrading the structured comparator. Generic position
RMSE worsened to **105.05/101.91 m** and attitude to **117.56/115.51°**; the
structured arm improved to **0.819/0.815 m** and **1.20/1.18°**. The run failed
four control-reference checks, four coverage-excess regression checks and one
new gating coverage-band check. All four trials replayed; altered model,
audit and comparator-summary copies were rejected. The implementation is not
accepted. No live or synthetic candidate run followed the control rejection;
no new platform improvement is claimed. Candidate unit validation passed
328 tests, with nine optional Cascade tests deselected.

Artifact diagnostics confirm that the intervention changed the learned response.
Command innovation after conditioning on pre-command history rose from
0.034/0.033/0.389 to 0.195/0.168/0.405 for throttle/roll/pitch; the trim roll
derivative changed from -0.985 to +0.168. On identical saved trim inputs,
250 ms vertical-velocity drift improved from -0.172 to -0.035 m/s, while
roll-rate drift changed from +0.039 to -0.080 rad/s and pitch-rate drift from
-0.064 to +0.084 rad/s. The larger angular drift is a prediction defect,
not proof of the tracking failure's cause. Conditional data associations
use only two training recordings and are not oracle sensitivities.

Requested RMS matched within `1.4e-15`; realized pitch RMS was only 70.7–72.9%
of requested RMS because 45.6–48.1% of intervals clipped, mostly from the base
command. Randomized requested signs do not make the clipped injection
unconditionally exogenous. The old sine's standard deviations as fractions of
declared command ranges were 2.35/2.42/1.93%; the previously reported
5.7/2.5/2.3% used observed spans.

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
  gate; calibration and state-dependent identification are not exhausted.
- Widening envelopes solely by distance repaired some unsupported cases while
  over-covering others. Neither the coverage band nor any reference is relaxed.

## Evidence and replay

Local runs live under `artifacts/2026-09-17/`: `slow-sampling`, `platform-pins`,
`control-baseline-fixed`, and `independent-calibration-control`. Verify/tamper
reports sit beside their run directories. The latter three also live on
ryserv under `/home/ryland/autonomy/glassbox-evidence/2026-09-17/`.
Use the original Linux environment for authoritative control replay; existing
sine regeneration has last-bit libm differences on macOS. Read-only model
diagnostics and their scripts are in `calibration-response-diagnostic` and
`command-moment-diagnostic` under the same local artifact root.

From disposable checkouts of the matching source commits, without refitting:

```sh
# Accepted runs: checkout 98f77d3 (56a9f98 also matches the control baseline).
PYTHONPATH=src python -m glassbox.experimental.harness verify /absolute/path/to/slow-sampling
PYTHONPATH=src python -m glassbox.experimental.harness verify /absolute/path/to/platform-pins
PYTHONPATH=src python -m glassbox.experimental.harness verify /absolute/path/to/control-baseline-fixed
# Rejected experiment: checkout 7c82db9; its control-v6 contract differs.
PYTHONPATH=src python -m glassbox.experimental.harness verify /absolute/path/to/independent-calibration-control
```

## Next named gap

**Control: coupled forecast accuracy at steady inputs and under command
changes.** Local command-response sign recovery was insufficient. Measure
angular drift and error growth over the controller's forecast horizon, together
with residual command-response error, before freezing one generic mechanism.
Preserve nonlinear, state-dependent behavior; no further mechanism, threshold
change or parameter sweep has been selected. Accuracy, live improvement,
evidence and lean remain open.
