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
The latest rejected learner remains isolated on `codex/full-response-identification`
at `5467943`, with same-data comparator `codex/independent-calibration` at `7c82db9`.
Read [the charter](charter.md) first. Git holds the experiment history;
this page records the current evidence, limitations and next named gap.

| Criterion | Current | Target |
| --- | --- | --- |
| One recipe | **Met.** The public API and fit/evaluate commands use the single `generic-memory-v3-prototype` recipe. `fit`, `predict` and `update` have no tuning or model-selection options. Replaced experimental module paths and structured root exports are removed. | One generic learner and consumer contract. |
| Accuracy | **Adopted with a known tradeoff.** Four of five corpora beat the structured comparator on both metrics, including under the corrected prefix-percentile readout. ARP loses both; this does not block adoption. | Broad competitive performance from one recipe; improve weak cases and establish task sufficiency separately. |
| Capability | **Met.** All 27 frozen synthetic cases pass; saved models replay and reject alteration. This is a regression guard, not platform readiness. | Every synthetic absolute cap passes. |
| Control | **Not met.** In twelve new oracle tracking trials, pooled samples within the ±0.5 m lateral/altitude tolerance are 32.12% for the four-step solver, 39.59% for float32 L-BFGS-B and 40.04% for float64 L-BFGS-B; 95% is required. All 1,272 float64 solves meet the gradient threshold without backend failure, but no trial meets the application criterion. Generic model accuracy was not changed or remeasured. | Meet the declared application tracking requirement; use structured and oracle arms diagnostically. |
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

The public API, slow-sampling correction and content-pinned recording loader
retain the adopted v3 numerical recipe. The last broad API/regression check
passed 1,287 tests. This iteration changes neither the learner nor maintained
consumer behavior; the latest validation below is focused on the diagnostic.
Git and the saved protocols hold the implementation and experiment history.

## Latest iteration: task relevance of solver accuracy

Freeze **`69c9d4d`**, implementation **`29cf9df`**, harness preflight correction
**`ae8df61`**, protocol [tracking v1](harness/solver-tracking-v1.json). Four
prospectively chosen initial-state seeds, **106–109**, each run all three arms
for 320 intervals on the same task-scaled oracle seam. The horizon remains
250 ms, the objective and command limits stay fixed, and each arm uses its
own trajectory and preceding returned plan. Float64 selects its seed in
float32 without a shadow optimizer, then lifts the exact selected blocks and
runtime values. Plant and causal reconstruction remain float32. No model is
fitted or updated; no wall-clock deadline influences commands.

| Measure across four trials per arm | Four-step projected gradient | Float32 L-BFGS-B | Float64 L-BFGS-B |
| --- | --- | --- | --- |
| Samples within task tolerance / 1,124 | 361 (32.12%) | 445 (39.59%) | 450 (40.04%) |
| Trials meeting the 95% requirement | 0/4 | 0/4 | 0/4 |
| Converged returned solves without failure / 1,272 | 0 | 1,022 | 1,272 |
| Backend failures | 0 | 26 | 0 |
| Median instrumented decision time | 73.7 ms | 197.7 ms | 230.7 ms |
| 95th-percentile decision time | 76.4 ms | 479.4 ms | 322.1 ms |

All twelve trajectories complete without fallback, nonfinite evaluations or
command-bound violations. Every scored altitude sample is within tolerance;
all task misses are lateral. Float64 adds **five** qualifying samples over
float32, a pooled **0.445 percentage-point** gain. Its lateral RMSE is only
**0.6–2.4 mm** better per trial, and altitude RMSE is slightly worse in three
of four trials (at most 0.221 mm). Against the four-step solver, both L-BFGS-B
arms lower lateral RMSE on every seed; float64's reduction is 26–30%.
These are useful partial gains, but **first-order convergence with accurate
dynamics is still insufficient for the declared tracking task**. It does not
establish global optimality or identify a single remaining controller cause.

The earlier precision comparison's **70/128** and this run's **1,272/1,272**
are different populations: old saved states and fixed historical warm starts
versus new seeds and each solver's own evolving closed loop. They are not an
improvement measured on one unchanged test set. Native objectives and residuals
are descriptive within each arm, not paired quality comparisons across its
different states. Four initial perturbations of one deterministic simulator
do not establish arbitrary-system readiness.

Float64 uses **23,828** new objective/gradient evaluations versus **29,875**
for float32 (20.2% fewer), with all preprocessing and audit work recorded
separately. Its timed wrapper includes 2,540 float32 seed-selection calls,
1,272 float64 seed evaluations, 1,272 final audits and 2,544 extra float64
diagnostic audits. The four-step solver's internal evaluation count is
unavailable. Every L-BFGS-B decision exceeds the 50 ms sample interval on this
host; timings include diagnostic overhead and do not establish deployment
latency. The simulated-time loop never feeds timing back into commands.

The old precision gate remains **failed** (paired residual ratio 0.803 versus
0.5 required). This protocol explicitly authorized diagnostic tracking before
measurement without converting that failure into a pass. **No maintained
solver is promoted**, and generic-model adoption is unchanged. The remaining
controller objective/horizon/seam question is separately scoped: Glassbox owns
control qualification, while `~/autonomy/dart` owns mission references,
objectives and execution. No Dart code changes belong to this iteration.

Validation: **259** focused local tests pass, with seven Cascade tests skipped
locally; **266** pass in the pinned Linux environment across the preflight and
its environment-metadata repair. That repair was tested before any full trial
started and changes no numerical protocol. All 90 inherited Python sources and
five input files remain byte-identical. The 72-file artifact inventory, saved
hashes and locally recomputed report match.

Fresh verification reproduces all **12 trajectories**, **3,816 optimizer
solves** and **3,816 oracle forecasts**, with exactly zero state replay
difference. Four rehashed challenges—an issued command, a full float64 gradient,
a work counter and the qualification result—are all rejected. After the full
fresh replay, those challenges reuse copies of its solver outputs only after
exact plan/input and per-trial manifest/trajectory checks. Physical replay
remains enabled; integrity checks can reject earlier. The public verifier
always reruns every optimizer solve.

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

Current artifacts are in
`artifacts/2026-09-18/solver-tracking-v1` locally and
`/home/ryland/autonomy/glassbox-evidence/2026-09-18/solver-tracking-v1` on
`ryserv`. Protocol SHA256:
`ed0d9429e85cb9bfdeb9263eae868625fc1624033b63689369fa27fe9654c8f0`.
`tracking-readout.json`, test/run logs, `solver-tracking-verification.py`
and verification/tamper reports sit beside the run. Each trial retains its
trajectory, forecasts, task metrics, timing arrays and full solver records.
Precision records include exact seed captures, full gradients and dtype audits.
Timing summaries are checked against saved arrays; timing authenticity cannot
be established by deterministic replay.

The prior precision artifacts remain in
`artifacts/2026-09-18/solver-precision-v1`; source-fixed termination evidence is
in `solver-termination-v1` beside it. Adopted learner/API, platform, control,
live and uncertainty evidence remains under `artifacts/2026-09-17` and its
corresponding `ryserv` evidence directory. The latest rejected learner remains
isolated at `5467943`; reduced assignment moments did not repair multistep
prediction and control. Prior outcomes and protocols are not rewritten.

```sh
# Current tracking diagnostic; use ae8df61 or this accepted result commit.
PYTHONPATH=src python -m glassbox.experimental.solver_tracking verify /absolute/path/to/solver-tracking-v1
# Previous fixed-seed precision comparison; use b11fb23 or this commit.
PYTHONPATH=src python -m glassbox.experimental.solver_precision verify /absolute/path/to/solver-precision-v1
# Original API/benchmark evidence and older diagnostic replay commands are in git.
```

## Next named gap

**Action-conditioned multistep learner accuracy.** Freeze a diagnostic using
identical held-out observed histories, state origins and feasible future
command sequences for the adopted generic model and a public-equation
reference. Separate absolute forecast error from the predicted response to
command changes over the full horizon; measure response direction and
magnitude as well as endpoint error. The learner receives only its existing
observed-signal/command contract, never oracle hidden state. Pin the population
and perturbations before measurement, preserve all failures, and choose one
learner mechanism only after this diagnostic identifies a specific weakness.

Do not resume controller residual, horizon or objective tuning in that
iteration. Accurate-dynamics tracking remains inadequate and needs a separate
controller investigation; it does not erase the much larger generic-to-oracle
prediction/control gap. Repair historical replay's Python-dependent AST
fingerprint in a separate correctness iteration with unchanged numerical gates.
