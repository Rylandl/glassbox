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
The latest paired-response mechanism qualifies in a larger simulator study,
with implementation isolated on `codex/intervention-response` at `7ea4674`.
The adopted learner and consumer data contract remain unchanged.
Read [the charter](charter.md) first. Git holds the experiment history;
this page records the current evidence, limitations and next named gap.

| Criterion | Current | Target |
| --- | --- | --- |
| One recipe | **Met.** The public API and fit/evaluate commands use the single `generic-memory-v3-prototype` recipe. `fit`, `predict` and `update` have no tuning or model-selection options. Replaced experimental module paths and structured root exports are removed. | One generic learner and consumer contract. |
| Accuracy | **Adopted with a known tradeoff.** Four of five corpora beat the structured comparator on both metrics; ARP loses both. Paired-response training improves aggregate held-out command-response error 65.1% and factual forecast error 31.4% versus the same-data, same-budget forecast-only learner. This qualifies a simulator research mechanism; public integration and arbitrary-system transfer remain unproved. | Broad competitive performance from one recipe; improve weak cases and establish task sufficiency separately. |
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

## Latest iteration: direct command-intervention learning

[Intervention response v1](harness/intervention-response-v1.json) was frozen at
**`2b069e4`**, implemented at **`b9b3f71`**, and independently audited at
**`7ea4674`**. The [result record](harness/intervention-response-v1-result.json)
anchors all 174 evidence files and supporting replay/audit records.

The experiment generated **80 independent parent trajectories** and **3,920
matched-history command branches**, with all branches of a parent kept together.
Forty parents train, twelve select checkpoints, twelve form the ordinary test
population, and sixteen form a prescribed larger-setpoint shift. Hidden plant
state is used only to generate the physical branches; the unchanged architecture
receives observed signals and applied commands. This is one aircraft and two
operating populations, not evidence from arbitrary systems. Matching physical
branches is a simulator capability, not an assumed consumer capability.

Exactly three optimization trajectories produce four fresh models; a historical
saved model supplies context. The primary comparison holds architecture, all
data, initialization, minibatch draws and 6,000-step budget fixed. The candidate
adds direct alternative-minus-baseline response supervision to the forecast
loss, and selects its checkpoint using the corresponding development objective.

| Prospective comparison | Aggregate response-error change | Aggregate factual-error change |
| --- | --- | --- |
| Capped v3 to full-data 1,000-step fit | −2.3% | −7.0% |
| Full-data 1,000 to 6,000 steps | −38.8% | −28.1% |
| Full-data forecast loss to paired-response loss, both 6,000 steps | **−65.1%** | **−31.4%** |

The first contrast bundles evidence use, weighting, minibatching and development
budget; it cannot identify those effects individually. The second is a retained
milestone of one trajectory. The third isolates the paired objective and its
checkpoint criterion. Historical-to-fresh changes data population and is context
only. More fitting compute and stronger supervision both help in this study.

**The primary mechanism passes.** All twelve response scores and all six factual
scores improve. Response geometric-mean ratio is **0.34899**, with worst ratio
0.57638; factual ratio is **0.68608**, with worst ratio 0.75047. The equal-weight
scores span both populations and velocity, body rate and rotation entries.
Paired parent-bootstrap 95% intervals are **[0.33116, 0.36895]** and
**[0.66703, 0.70556]**, respectively. These intervals describe parent sampling
within this aircraft population, not uncertainty across systems.

The forecast-only model selects step 5,700; the paired model selects step 5,900.
Normalized development response loss falls from 0.37793 to 0.05732; development
factual loss falls from 0.05660 to 0.02938. The gain transfers beyond training
parents and persists under the prescribed shift. This does not prove that this
architecture can identify every hidden state or extrapolate to arbitrary dynamics.

Early aileron response signs improve substantially: pooled first-step velocity
reversals fall **64→9**, and rotation reversals **170→27**, among 390 nonweak
probes; two weak probes remain separately counted. All signs are not repaired.
Throttle body-rate reversals at 150 ms rise 55→163/392, and throttle rotation
reversals at 250 ms rise 25→176/392, even though absolute response errors fall.
The remaining errors on these small cross-responses exceed their physical response magnitudes.

All **40,040 forecasts are finite and replay exactly without refitting**.
Physical replay reconstructs every parent and branch; data roles, initialization,
selected objectives, metrics and all 2,000 bootstrap draws are verified. An
independent NumPy audit reproduces all eighteen point scores. Five coherently
rehashed changes to physical data, lineage, model parameters, predictions with
updated reports, and qualification decisions are rejected. Intermediate optimizer
updates are not independently re-solved.

Validation: **190 focused local tests pass**, six skip; **186 remote tests pass**,
nine skip and one inherited migration source-check test fails. The failure is a
Python 3.12/3.13 AST serialization difference on byte-identical sources, confirmed
with complete AST comparison; the experiment's 93 raw-byte source pins pass.
Ruff passes. The original remote suite is not described as fully green. The
separate compatibility correction `b4e8126` preserves every historical protocol/hash;
53 focused local checks pass, and 45 remote checks pass with eight artifact skips.
Historical experiment replay still uses its original isolated checkout.
No public recipe, uncertainty envelope, controller or live-update result is
promoted by this research qualification.

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

Latest evidence: `artifacts/2026-09-18/intervention-response-v1` locally and
`/home/ryland/autonomy/glassbox-evidence/2026-09-18/intervention-response-v1` on
`ryserv`. The [result record](harness/intervention-response-v1-result.json) pins
its run manifest and separate audit/replay records. The isolated implementation
is `codex/intervention-response` at `7ea4674`, in
`/private/tmp/glassbox-intervention-response` locally and
`/tmp/glassbox-intervention-response-test` on `ryserv`.

Historical platform, control, live and uncertainty results remain unchanged.
`tracking-target-provenance-audit.json` documents the earlier 843/843 tracking
result. Git and each experiment's frozen protocol retain prior outcomes.

```sh
# Isolated experiment checkout at 7ea4674; no refitting.
PYTHONPATH=src python -m glassbox.experimental.intervention_response verify /absolute/path/to/intervention-response-v1
```

## Next named gap

**Anticipatory position tracking in the oracle controller at the existing
250 ms forecast horizon.** Better response learning now has stronger evidence,
but even exact equations achieve only 40.04% on the latest tracking population.
Freeze a separate no-fit comparison replacing only the position residual with
position error plus 1.5 seconds times velocity error. This coefficient is drawn
from the historical successful controller before new measurements; it is not
a longer prediction horizon or a promise of 95% success. Retain the existing
solver, other costs, bounds and causal sequencing, use fresh matched trials,
and distinguish material progress from the unchanged 95% application criterion.

Then test whether the qualified learned response transfers into an adequate
controller under its own trajectories. Public integration still needs a generic
way to exploit informative recordings without assuming simulator state cloning.
An explicit history encoder and latent transition remain a researched alternative
if a named representation or transfer failure warrants them; today’s evidence
does not require an architecture replacement to explain the response gains.
