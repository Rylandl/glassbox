# Public and quadratic learners on identical excited recordings

Better training recordings substantially improve the unchanged public learner.
On fresh matched tests, weighted forecast/command-response error falls
**5.9% / 63.7%** against the same recipe trained on original recordings. The
quadratic learner improves those aggregates a further **22.6% / 23.8%** on
identical excited data, but Crazyflow angular-error tails exceed the predeclared
limits against this stronger public baseline. Both workflow improvements pass;
the frozen architecture preference is **unresolved tradeoff**.

The [protocol](harness/excited-public-architecture-v1.json) was committed at
`b24f4d5`, and implementation at `974171b`, before this experiment's collection
or fitting. The [result record](harness/excited-public-architecture-v1-result.json)
anchors saved evidence, audits and reproduction. No public recipe changes here.

## What the comparison establishes

The three fitted arms are public-original (`baseline`), public-excited
(`candidate`) and quadratic-excited (`quadratic`), plus hold-current. Each public
fit calls ordinary `glassbox.fit(collection)` once, with no options, excitation
labels or hidden state. The quadratic fit repeats the accepted excitation study's
unchanged learner. Original-data public and excited-data quadratic refits must
exactly reproduce their prior saved revisions, including every array and metadata
field. Both do.

The accepted excitation recordings are copied byte for byte, including the
historical seals and provenance. They preserve the same 72 training and 24
development parents per simulator, training failures, issued commands, clipping
and unchanged development recordings. Separate reuse records bind those historical
files to the new experiment; their old provenance is not relabeled as new data.
Original calibration is regenerated and compared exactly with its trusted source.

The two excited architectures use identical 384 training and 256 development
windows, all cached arrays, six base normalization arrays and numerical loss
weights. Both use seed zero, width 32, memory eight, batch size 64, 1,000 optimizer
updates and minimum finite development-loss checkpoint selection every 100 steps.
Parameter counts are 12,470 versus 15,170 in Crazyflow and 3,945 versus 6,420 in
Cascade. This matches data and update budgets, not FLOPs. It compares the combined
state-command and autonomous-state product paths; it does not isolate either one.

The fresh unperturbed test cohort has 42 condition cells and 84 parents per
simulator. All predictors receive matching observed histories and issued commands;
only evaluation accesses simulator state for cloned response queries. Previously
inspected test cohorts are not used as confirmation for this comparison.

## Physical accuracy and retained losses

Primary 250 ms endpoint component RMSE:

| Simulator | Quantity | Public-original | Public-excited | Quadratic-excited |
| --- | --- | ---: | ---: | ---: |
| Crazyflow | Forecast velocity, m/s | 0.15580 | 0.16025 | **0.07033** |
| Crazyflow | Forecast body rate, rad/s | 0.24609 | **0.19334** | 0.35358 |
| Crazyflow | Forecast rotation entries | 0.09971 | 0.10647 | **0.03237** |
| Crazyflow | Response velocity, m/s | 0.18829 | 0.12468 | **0.03967** |
| Crazyflow | Response body rate, rad/s | 0.34000 | **0.25343** | 0.34742 |
| Crazyflow | Response rotation entries | 0.12460 | 0.10768 | **0.03000** |
| Cascade | Forecast velocity, m/s | 0.16643 | 0.16834 | **0.12056** |
| Cascade | Forecast body rate, rad/s | 0.11768 | 0.10686 | **0.09487** |
| Cascade | Forecast rotation entries | 0.01836 | 0.01818 | **0.01080** |
| Cascade | Response velocity, m/s | 0.12761 | 0.04767 | **0.03747** |
| Cascade | Response body rate, rad/s | 0.06387 | 0.03756 | **0.03306** |
| Cascade | Response rotation entries | 0.01225 | 0.00704 | **0.00367** |

[Physical-error chart](../artifacts/2026-09-18/excited-public-architecture-v1-performance.png)
and [horizon chart](../artifacts/2026-09-18/excited-public-architecture-v1-horizons.png)
include the simple reference and horizon dependence.

The public collection gain is broad, not universal. Primary 250 ms velocity
forecasts worsen 2.9% in Crazyflow and 1.1% in Cascade; Crazyflow rotation forecasts
worsen 6.8%. Its 50/150 ms angular responses worsen 1.7%/7.3%, despite improving
25.5% at 250 ms. Speed-shift forecast aggregates worsen 9.3%/6.7% in
Crazyflow/Cascade, and Crazyflow wind-shift forecasts worsen 3.5%. Every frozen
public collection criterion nevertheless passes.

On the identical excited data, quadratic improves Crazyflow velocity and rotation
forecasts and responses in all 24 primary cells, but worsens angular forecasts in
all 24. Public-excited has better angular responses in 16/24. Moving from public-excited
to quadratic, the 250 ms angular forecast/response parent p95 errors rise from **0.30015 / 0.36559** with public to
**0.58434 / 0.60416 rad/s** with quadratic: **1.947 / 1.653 times**, exceeding the
frozen 1.5 limits. In the converse comparison, the public model has substantially
worse velocity and rotation tails. Neither direction satisfies the architecture
preference criteria; this does not negate either workflow's improvement over
public-original.

The public collection gain still leaves Crazyflow's 250 ms velocity-response
error **0.12468 m/s**, only **0.74%** below the zero-response reference's
**0.12561 m/s** on the same 728 available queries. Quadratic reaches **0.03967 m/s**.
Improvement against the original learner is therefore not sufficient response
fidelity by itself.

Crazyflow completes **59/84** test parents; 25 cross the altitude boundary.
Primary 250 ms truth includes **426/480** factual and **728/768** response queries.
Cascade completes **84/84**, with **480/480** factual and **576/576** response
queries. Every eligible prediction is finite. Scores condition on available truth,
and all arms share the same missing slots. Different completion counts from prior
cohorts do not establish a performance change.

## Frozen decisions

| Numerator / denominator | Forecast ratio (paired 95% interval) | Response ratio (paired 95% interval) | Decision |
| --- | --- | --- | --- |
| Public-excited / public-original | 0.94118 [0.92629, 0.96173] | 0.36314 [0.35526, 0.36934] | Collection progress |
| Quadratic-excited / public-original | 0.72846 [0.71364, 0.74775] | 0.27683 [0.27046, 0.28132] | Workflow progress |
| Public-excited / quadratic-excited | 1.29202 [1.26290, 1.31861] | 1.31178 [1.28575, 1.33809] | Simpler recipe not comparable |
| Quadratic-excited / public-excited | 0.77398 [0.75837, 0.79183] | 0.76232 [0.74734, 0.77776] | Material-value preference fails angular tails |

These are weighted geometric RMSE ratios, not pooled physical errors. Weighting
is equal across simulators, half primary and one eighth per each of four shifts,
equal across three groups and 50/150/250 ms horizons. Floors and weights were
frozen. The 1,000 paired parent-bootstrap draws are shared across all comparisons;
intervals describe these parents and conditions, not arbitrary-system uncertainty.

Workflow progress requires response ratio at most 0.90 and forecast ratio at most
1.05. Simpler comparability requires both at most 1.05; quadratic material value
requires 0.90 response and 1.05 forecast. All apply the same per-simulator primary
1.15, scope 1.5 and primary 250 ms parent-p95 1.5 limits. Architecture preference
also requires its corresponding workflow progress. Pairwise fit availability
cannot erase a valid gain in an independent comparison. No criterion was changed
after seeing these results.

## What development evidence explains

Crazyflow's public-excited fit still selects step zero, with exactly zero neural
output and hidden-memory readout matrices. The improved recordings improve its
affine response, but do not make that selected response state-dependent. Quadratic
selects step 900. Under exactly the same development objective, public/quadratic
losses are **0.01917885 / 0.00373505**. Rotation's contribution falls
**0.01769437→0.00149996**, velocity falls **0.00072155→0.00025948**, and rate worsens
**0.00076293→0.00197561**. The optimizer's selected result gains much more on
rotation than it loses on rates. This is a measured objective tradeoff, not proof
of whether initialization or optimization causes it. Cascade selects step 1,000
for both and all three development groups improve with quadratic.

Collection changes visited states, failure prefixes, normalization and loss
scales relative to original data. Prior excitation evidence retains Crazyflow
training completion **49/72→24/72**; all those failed parents remain imported.
The public collection gain therefore supports a practical recording intervention,
not isolated causal identification. Matched excited-architecture comparisons do
not have that data/weight confound.

## Verification and next gap

Both simulator replays reproduce **58,668 common and excited data/query arrays**,
saved predictions on **4,032 queries**, **342,720 metric rows**, all four decisions
and bootstrap draws. Imported excited physics is regenerated against its original
trusted protocol and then compared with the copied bytes; 144 training parents
and 48 unchanged development copies are checked. No optimizer is rerun in replay.

Independent NumPy analysis verifies 240 endpoint score groups, 240 comparison
entries, 480 parent-tail entries and 48 tail gates at 250 ms; it also checks the
complete metric roster and all-horizon truth/finiteness, both comparator revisions,
392 imported files and matched caches/norms/loss weights. Other-horizon scores and
bootstrap regeneration are verified by production replay. All **278 focused
tests** pass, and Ruff passes. The initial test-only frozen-dataclass patch failure
is retained in its attempt log; it was fixed before implementation commit and any
trials. No new native-fixture test run is claimed for this iteration.

Fourteen disposable-copy alteration challenges pass on their first attempt,
covering changed model weights, imported commands, source identity, histories,
commands, branch associations, physical trajectories, predictions, metrics and
decisions. Coherent rewrites recompute local hashes and are still rejected by
trusted provenance or fresh execution. All thirteen experiment stages complete
without timeout or failure; the original sealed bundle remains unchanged.

The next named gap is **retaining quadratic velocity/rotation gains while repairing
Crazyflow angular forecasts and tails**. Test the bilinear-only learner on the
same excited recordings, removing the autonomous state-product path with data,
loss and optimizer budgets fixed. This missing comparison can establish whether
those autonomous products earn their added complexity; the current evidence does
not isolate their effect. Save initial and existing-checkpoint development
diagnostics prospectively. Objective allocation is a subsequent hypothesis, not a
second change bundled into that ablation.

Public recipe, derivative, envelope, update and controller qualification remain
separate. This experiment integrates collection evidence and a precise architecture
tradeoff, with no platform branch, consumer option or per-channel model mixing.

## Reproduction

Use pinned Python 3.12.12, JAX/JAXlib 0.11.1, NumPy 2.5.3, SciPy 1.18.1 and the
frozen Crazyflow/Cascade sources on CPU/x64. Set `SCIPY_ARRAY_API=1`,
`JAX_ENABLE_X64=1`, and `PYTHONPATH` to this checkout's `src` and pinned Cascade
`src`. The trusted predecessor bundles named in the protocol must verify.

Run `python -m glassbox.experimental.excited_public_experiment` with a stage,
simulator and `--output <bundle>`. Generate both simulators, then fit `--arm baseline`,
`--arm quadratic`, and `--arm candidate` for each; evaluate both; finalize once.
Capture the printed root SHA256 externally. Replay both simulators with
`--expected-bundle-sha <sha>`. The accepted root SHA256 is
`db758fd158cee2a2b595ac570c72232ed2d245cbd12193e80bac7ec5791b899d`.
Audit scripts and logs remain outside the sealed bundle and are anchored by the
result record.
