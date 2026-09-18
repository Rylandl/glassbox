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
| Accuracy | **Adopted with a known tradeoff.** Four of five corpora beat the structured comparator on both metrics; ARP loses both. A new shared-history diagnostic finds wrong early command-response signs in the unchanged learner: all 248 aileron probes reverse the first-step body-rate response. | Broad competitive performance from one recipe; improve weak cases and establish task sufficiency separately. |
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
passed 1,287 tests. This iteration changes neither the learner nor maintained
consumer behavior; the latest validation below is focused on the diagnostic.
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

## Latest iteration: action-conditioned multistep learner accuracy

Protocol [action response v1](harness/action-response-v1.json), frozen at
**`6a6ace7`**, with source initial-state precision clarified before measurement
at **`bfb6fc8`**; implementation **`5293929`**. No learner is fitted or changed.
The saved adopted v3 model consumes exactly ten past command intervals and
eleven observations at **124 origins** from all four float64-oracle trajectories.
These trajectories were excluded from its fit, but were already used for
controller evaluation; this is new model-response evidence on a known simulator
population, not an untouched controller benchmark.

At every origin, the factual five-command tape and six channel-isolated
alternatives are shared between the learner and public-equation reference.
Each alternative moves one channel 5% of the remaining distance toward its
lower or upper command bound. The reference reconstructs its internal state
only from equilibrium reset and issued-command replay; the learner receives
no hidden simulator state or future observations. Both predictions run in float32.
All **868 forecast sequences** complete and all **744 paired command changes**
are nonzero, with no invalid or weak-response exclusions in any output group.
Every factual reference forecast exactly matches the saved physical future.
These errors describe this controller-generated population; earlier platform
and calibration scores cover different rows.

| Factual forecast component RMSE | 50 ms | 250 ms |
| --- | --- | --- |
| World velocity, m/s | 0.00816 | 0.32438 |
| Body rate, rad/s | 0.05345 | 0.25984 |
| Rotation-matrix entries, unitless | 0.00112 | 0.03074 |

Paired response subtracts each predictor's own factual forecast. Negative
projected gain means that its response has a component against the true
response, not that every output coordinate has the wrong sign. All counts
below are over **248 aileron probes** (two signs at every origin):

| Aileron response group | Negative at 50 ms | Negative at 250 ms | Negative over flattened five-step response | Median full-response cosine |
| --- | --- | --- | --- | --- |
| World velocity | 248/248 | 248/248 | 248/248 | −0.434 |
| Body rate | 248/248 | 0/248 | 0/248 | +0.881 |
| Rotation entries | 248/248 | 219/248 | 248/248 | −0.465 |

The velocity response opposes the reference at every measured horizon for all
248 probes. Body rate is almost exactly reversed at 50 ms (median cosine
−0.994), then aligned at later steps but weak: median full-response projected
gain is 0.341. Rotation response is reversed in every probe through 200 ms.
These patterns occur on each of the four trajectories; an endpoint-only or
pooled-horizon body-rate score would hide the immediate reversal.

Elevator responses are strongly aligned over the full horizon (median cosine
roughly 0.99), but attenuated: projected gains are about 0.58 for velocity,
0.73 for body rate and 0.62 for rotation entries. Throttle probes produce large
relative errors where the true rate/rotation effects are small; the report
retains their physical response magnitudes and absolute errors, so those ratios
are not treated as the dominant failure merely because they are large.

This identifies a concrete **early command-response sign and timing** weakness,
plus compounding factual forecast error. It does not uniquely identify whether
calibration/feedback confounding, representation, memory or fitting caused it.
No physics-specific correction, model promotion or task pass follows from this
diagnostic; the separate controller insufficiency remains visible.

Validation: **109 local tests pass**, with three Cascade tests skipped locally;
**112 pass on ryserv**, including those three. All 92 inherited Python sources
and 15 inputs are unchanged. The complete 28-file inventory and locally
recomputed report match. Fresh verification reproduces every physical source
trajectory and all 868 learner/reference forecast pairs exactly. Four coherently
rehashed alterations—history, command, forecast and response summary—are
rejected; the three numeric-array challenges each receive a full fresh replay.

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

The latest evidence is in `artifacts/2026-09-18/action-response-v1` locally and
`/home/ryland/autonomy/glassbox-evidence/2026-09-18/action-response-v1` on
`ryserv`. Protocol SHA256:
`4b40ee76aa5e0e7b1b5a2eafdbce90b95566ba45dcc769e5c3c04453df52dba6`.
The verification script/reports, test/run logs and `action-response-readout.json`
sit beside it. Each trial retains exact histories, future commands, responses,
forecasts and failure slots. `tracking-target-provenance-audit.json` and its
script independently document the earlier 843/843 target-feasibility result.

The control comparison remains in `artifacts/2026-09-18/solver-tracking-v1`,
with precision and termination evidence beside it. Adopted learner/API,
platform, control, live and uncertainty evidence remains under
`artifacts/2026-09-17` and the corresponding `ryserv` evidence directory.
The rejected learner remains isolated at `5467943`; earlier outcomes and
protocols are not rewritten. Git holds their replay commands and explanations.

```sh
# Latest diagnostic; use 5293929 or this accepted result commit.
PYTHONPATH=src python -m glassbox.experimental.action_response verify /absolute/path/to/action-response-v1
# Latest frozen controller comparison; same code remains available.
PYTHONPATH=src python -m glassbox.experimental.solver_tracking verify /absolute/path/to/solver-tracking-v1
```

## Next named gap

**State-conditioned early command-response identification.** Correct the
measured first-step response reversal while retaining full-prefix validation;
endpoint body-rate alignment is insufficient. The proposed next mechanism is
a conditional response loss on the complete predictor: independently assigned
excitation interacted with fixed, platform-neutral observed-history features.
This extends beyond the prior three unconditional moments or constraints on
affine columns alone. This is a hypothesis to test, not a diagnosed unique cause
or an implemented learner change.

Freeze one concrete mechanism, independent training/development evidence,
realized-excitation checks and gain/loss criteria before fitting. Do not reuse
these 124 observed histories or their oracle responses as training targets.
Use platform-independent signals, commands and recording facts; preserve the
one `fit`/`predict`/`update` contract. Require paired full-response and factual
forecast readouts across every horizon, including 50 ms, and retain broad
benchmark/synthetic regression checks. No controller residual, horizon or
objective tuning belongs to that learner iteration. The inadequate current
oracle controller and historical Python-dependent AST replay fingerprint
remain separately scoped work.
