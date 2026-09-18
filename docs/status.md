# Status: gap against the charter

Updated 2026-09-18. **The generic approach is adopted as the development
baseline.** The user prioritizes generality over superiority on every corpus;
the four-corpus advantage justifies accepting the known ARP deficit. This is
an explicit policy decision on existing evidence, not a new experimental
pass. The adopted learner remains `generic-memory-v3-prototype`. Its recipe
and numerical logic from `98f77d3` now serve the public `glassbox.fit` API;
the returned `LearnedDynamics` provides `predict` and `update`. Generic
recording archives also serve the public fit/evaluate commands. Structured
code remains in its owning modules for benchmarks and unmigrated control
consumers.
The latest unqualified candidate remains isolated on `codex/conditional-response`
at `8070d4f`; none of its implementation enters the adopted learner.
Read [the charter](charter.md) first. Git holds the experiment history;
this page records the current evidence, limitations and next named gap.

| Criterion | Current | Target |
| --- | --- | --- |
| One recipe | **Met.** The public API and fit/evaluate commands use the single `generic-memory-v3-prototype` recipe. `fit`, `predict` and `update` have no tuning or model-selection options. Replaced experimental module paths and structured root exports are removed. | One generic learner and consumer contract. |
| Accuracy | **Adopted with a known tradeoff.** Four of five corpora beat the structured comparator on both metrics; ARP loses both. The latest conditional loss improves aggregate command-response error 3.65%, but initial aileron velocity/rotation responses remain reversed in all 248 probes; the candidate is not promoted. | Broad competitive performance from one recipe; improve weak cases and establish task sufficiency separately. |
| Capability | **Met.** All 27 frozen synthetic cases pass; saved models replay and reject alteration. This is a regression guard, not platform readiness. | Every synthetic absolute cap passes. |
| Control | **Not met.** In twelve new oracle tracking trials, pooled samples within the ±0.5 m lateral/altitude tolerance are 32.12% for the four-step solver, 39.59% for float32 L-BFGS-B and 40.04% for float64 L-BFGS-B; 95% is required. All 1,272 float64 solves meet the gradient threshold without backend failure, but no trial meets the application criterion. Generic controller tracking was not remeasured; the learner is unchanged. | Meet the declared application tracking requirement; use structured and oracle arms diagnostically. |
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

## Current control evidence

The earlier same-horizon diagnostic substitutes public Cascade equations for
only the generic forecast mean while retaining its 250 ms controller seam.
Oracle position component RMSE is **0.565/0.457 m**, versus **60.80/49.31 m**
for the saved generic arm. The 250 ms structured arm scores **1.816/1.795 m**;
the historical 800 ms structured pipeline scores **1.179/1.179 m**. None
meets the tracking application criterion. These comparisons establish both a
large learned-mean weakness and an inadequate controller even with accurate
dynamics; they do not establish performance on arbitrary systems.

The **95% within ±0.5 m** target was chosen internally as a provisional
application requirement, not taken from an external standard. It has historical
feasibility evidence: the earlier [research controller](cascade-accuracy.md)
reached **281/281 qualifying samples on each of three seeds**, using the same
Cascade equations and aircraft specification. Its constant-command Gauss–Newton
solver, 1.5 s anticipatory cost term, float64 runtime and pretrial history holds
differ from the current controller. Today's provenance audit matches all three
saved trajectory hashes to `bf97de3` and independently recomputes **843/843**
qualifying samples. It does not rerun those historical optimizers. This supports
feasibility for the declared calm, truth-sensed simulator task; it proves neither
current-controller adequacy nor real-time or arbitrary-system performance.
The user explicitly retains this target as a driver for further improvement.

The public API, slow-sampling correction and content-pinned recording loader
retain the adopted v3 numerical recipe. The last broad API/regression check
passed 1,287 tests. The latest candidate remains isolated; the adopted learner and maintained
consumer behavior are unchanged. Validation below is focused on this experiment.
Git and the saved protocols hold the implementation and experiment history.

The latest frozen controller comparison, [tracking v1](harness/solver-tracking-v1.json),
uses seeds 106–109 and each solver's own trajectory/warm starts on the 250 ms
task-scaled oracle seam. All twelve trials complete without fallback or bound
violations; every scored altitude sample passes, so all tolerance misses are lateral.

| Measure across four trials per arm | Four-step projected gradient | Float32 L-BFGS-B | Float64 L-BFGS-B |
| --- | --- | --- | --- |
| Samples within task tolerance / 1,124 | 361 (32.12%) | 445 (39.59%) | 450 (40.04%) |
| Trials meeting the 95% requirement | 0/4 | 0/4 | 0/4 |
| Converged returned solves without failure / 1,272 | 0 | 1,022 | 1,272 |
| Backend failures | 0 | 26 | 0 |
| Median instrumented decision time | 73.7 ms | 197.7 ms | 230.7 ms |

Float64 adds five qualifying samples over float32, only 0.445 percentage points,
and removes its 26 backend failures. The old precision gate remains failed
(residual ratio 0.803 versus 0.5 required); the prospective tracking diagnostic
was explicitly authorized without rewriting that result. None of these solvers
is promoted. These are different trajectories from the old fixed-seed 70/128
precision population. Neither native optimizer-score comparisons across
different states nor instrumented timings establish controller quality or
real-time readiness. All 12 trajectories and 3,816 solves/forecasts replayed
exactly, four rehashed challenges were rejected, and 266 focused remote checks
passed. Git and saved protocols hold the detailed tracking experiment history.

## Latest iteration: conditional command-response learning

[Conditional response v1](harness/conditional-response-v1.json) was frozen at
**`60b1726`**, implemented at **`919aafc`**, and tested for replay integrity at
**`22aff6c`**. A pre-fit check incorrectly required exact equality between the
reset quaternion and its normalized canonical observation, differing by
2.22e-16. Correction **`8070d4f`** applies the already frozen physical tolerance;
no fitting occurred before the correction and no thresholds changed.
The [committed result record](harness/conditional-response-v1-result.json)
anchors the complete evidence inventory and supporting audits.

One candidate adds a first-step conditional assignment-moment loss to the
complete predictor, retaining the adopted architecture, multistep forecast
loss and optimizer budget. Nine fixed observed-history features interact with
three independently randomized requested commands; all 27 instrument directions
pass the input-only support screens in both fitting roles. Requested assignment
remains separate from clipped applied commands and realized injection.
The four calibration trajectories replay in float64 within 2.22e-16.

**The comparison isolates the loss, not the calibration data.** Both arms use
exactly 292 training windows from recordings 0–1 and 146 development windows
from recording 2; recording 3 remains reserved. The one fresh unchanged-v3 fit
is byte-identical to the pinned randomized-calibration baseline. The candidate
also selects step 100, using development scores alone. It lowers the training
conditional moment from 0.00733 to 0.00543, but the development moment rises from
0.03005 to 0.03047. Better training identification does not transfer reliably.

Both arms forecast the same 124 known diagnostic origins, seven command tapes
per origin, and all 146 reserved windows: **1,014 finite sequences per arm**.
These origins were excluded from fitting and checkpoint selection, but were
already inspected during prior diagnostics. They are correlated simulator
histories, not independent systems or an untouched controller benchmark.

| Paired response RMS vector error | Same-data v3 | Conditional candidate | Change |
| --- | --- | --- | --- |
| Velocity, 50 ms | 0.001006 m/s | 0.000896 m/s | −10.93% |
| Velocity, flattened 250 ms prefix | 0.079413 m/s | 0.077994 m/s | −1.79% |
| Body rate, 50 ms | 0.017584 rad/s | 0.017419 rad/s | −0.94% |
| Body rate, flattened 250 ms prefix | 0.054279 rad/s | 0.054690 rad/s | +0.76% |
| Rotation entries, 50 ms | 0.001102 | 0.001032 | −6.42% |
| Rotation entries, flattened 250 ms prefix | 0.018216 | 0.017838 | −2.08% |

Five of six response scores improve. Their equal-weight geometric mean ratio
is **0.96352**, a **3.65% improvement**, versus the prospectively frozen **0.80**
requirement. The worst ratio is only 1.00757, within the 1.25 regression limit:
**the failure is insufficient aggregate progress, not a veto for one loss**.
All six factual forecast scores regress slightly, by 0.25–1.40%; the aggregate
ratio is **1.00831**, within its 1.05 limit. The mechanism does not qualify, and
the frozen stop rule skips broader, control and live trials. No recipe,
controller, uncertainty envelope or tracking result is promoted.

The candidate still reverses the first aileron velocity and rotation responses
in **248/248 probes**. Later sign improvements are real: velocity reversals at
250 ms fall from 169 to 110, and rotation reversals at 150 ms fall from 84 to 2.
The unchanged same-data baseline already has **0/248 first-step body-rate
reversals**, versus 248/248 for the historical model trained on different
calibration recordings. The conditional loss cannot take credit for that data
change. Its first-step body-rate gain remains weak, around 0.15 of the physical
reference, compared with around 0.14 for the same-data baseline.

Validation: **225 unique focused local tests pass**, with three Cascade tests
skipped locally; **228 pass on ryserv**. Ruff passes. All 68 evidence files
match their inventory. Replay reconstructs calibration data, physical source
trajectories, selected-model objectives and every held-out forecast without
refitting. An independent NumPy audit reproduces all twelve qualification
scores exactly. Four coherently rehashed alterations—raw assignment, candidate
parameter, held-out forecast with updated metrics, and qualification flag—are
rejected. Checkpoint structure and selected scores are verified; historical
intermediate optimization steps are not independently re-solved.

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

Latest evidence: `artifacts/2026-09-18/conditional-response-v1` locally and
`/home/ryland/autonomy/glassbox-evidence/2026-09-18/conditional-response-v1` on
`ryserv`. The [result record](harness/conditional-response-v1-result.json) pins
its run manifest and the independent audit, replay logs and challenge script
beside it. Candidate implementation stays on `codex/conditional-response` at
`8070d4f`; the working tree is `/private/tmp/glassbox-conditional-response`
locally and `/tmp/glassbox-conditional-response-test` on `ryserv`.

Action-response and solver-tracking evidence remains under
`artifacts/2026-09-18`. `tracking-target-provenance-audit.json` documents the
earlier 843/843 target-feasibility result. Adopted API, platform, control, live
and uncertainty evidence remains under `artifacts/2026-09-17`. Historical
protocols and outcomes are not rewritten; git holds their implementation.

```sh
# Isolated candidate checkout at 8070d4f; no refitting.
PYTHONPATH=src python -m glassbox.experimental.conditional_response verify /absolute/path/to/conditional-response-v1
# Maintained no-fit diagnostics.
PYTHONPATH=src python -m glassbox.experimental.action_response verify /absolute/path/to/action-response-v1
PYTHONPATH=src python -m glassbox.experimental.solver_tracking verify /absolute/path/to/solver-tracking-v1
```

## Next named gap

**Generalization of command-response identification across independent observed
histories.** The conditional moment improves on training data but not on
independent development data; small response gains leave immediate velocity
and rotation signs wrong. Before another loss or architecture change, freeze
one experiment that strengthens independent command-response evidence and
measures whether that evidence transfers. Use the unchanged v3 learner on
identical data as the comparator so any data benefit remains separate from a
learning-mechanism benefit.

Prospectively chosen matched-history command interventions can help diagnose
this gap in the simulator; they are a harness capability, not an assumed
capability of arbitrary real systems. Keep parent histories and all their
branches together when separating training, development and evaluation.
Never use these 124 inspected diagnostic origins or their oracle responses as
training targets. Preserve the platform-independent signal/command contract,
actual applied-command accounting and separate factual/response metrics at
every horizon. No post-result moment-weight sweep or controller tuning belongs
to this iteration. The insufficient current oracle controller remains a
separate gap against the retained 95% tracking requirement.
