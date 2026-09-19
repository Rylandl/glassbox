# Independent training command excitation

This experiment tests whether independent command variation in ordinary training
recordings improves the unchanged autonomous quadratic learner. The protocol was
committed at `0721953`, and the implementation at `6460ca6`, before collection or
fitting. The [frozen protocol](harness/independent-command-excitation-v1.json)
defines the population, seeds, data roles, budgets and residual criteria.

## Mechanism and comparison

Each simulator has the same 72 training and 24 development recording identities as
the prior experiment. Training commands remain unchanged for the first 0.75 s.
Thereafter an independent sign per command channel adds 5% of that channel's
declared range, held for 0.10 s, before clipping to the existing bounds. A separate
SHA256-derived PCG64 stream assigns all 23 blocks before simulation. The identical
normalized rule applies to both simulators. Every issued command, clipping event,
initial prefix and collection outcome is retained; all development files are
copied exactly. No failed parent is replaced.

The three fitted arms are the public generic learner on original recordings,
the preceding quadratic learner on original recordings, and that same quadratic
recipe on excited training recordings. Hold-current is a fourth reference. Both
original-data models are independently refitted and must match their trusted
saved revisions in every array, metadata field and fingerprint. The candidate
uses 384 training and 256 unchanged development windows, one 1,000-update optimizer
run, batch size 64 and the same architecture, initialization draws and checkpoint
rule. Training-derived normalization and loss scales are recomputed from the
changed data. Equal development recordings therefore do not mean equal numerical
checkpoint weights. This is a collection intervention, not an equal-data claim.

The fresh held-out cohort retains 42 condition cells and 84 test parents per
simulator. Ordinary forecasts and cloned command-response queries use common,
unperturbed evaluation recordings for all arms. Simulator hidden state is used
only by the evaluator. Responses are assessed separately from factual forecasts,
with weak probes, failed collections and missing truth retained in the reports.

## Residual results

The candidate passes every predeclared residual criterion. Weighted factual and
response ratios against public are **0.72197 / 0.28220**, with paired-parent 95%
intervals **[0.70267, 0.74617] / [0.27536, 0.28744]**. Against the same quadratic
architecture trained on original recordings, ratios are **0.89498 / 0.33043**,
with intervals **[0.87255, 0.92297] / [0.32238, 0.33786]**. These represent 27.8% /
71.8% gains against public and 10.5% / 67.0% gains from the collection intervention
within the quadratic recipe. They are conditional on available truth.

Primary 250 ms endpoint component RMSE:

| Simulator | Quantity | Public, original data | Quadratic, original data | Quadratic, excited data |
| --- | --- | ---: | ---: | ---: |
| Crazyflow | Factual velocity, m/s | 0.16416 | 0.08536 | **0.06346** |
| Crazyflow | Factual body rate, rad/s | **0.25568** | 0.37498 | 0.32796 |
| Crazyflow | Factual rotation entries | 0.10970 | 0.04380 | **0.02961** |
| Crazyflow | Response velocity, m/s | 0.17590 | 0.09229 | **0.03527** |
| Crazyflow | Response body rate, rad/s | 0.31712 | 1.07374 | **0.30831** |
| Crazyflow | Response rotation entries | 0.11835 | 0.08418 | **0.02621** |
| Cascade | Factual velocity, m/s | 0.16124 | 0.12639 | **0.12252** |
| Cascade | Factual body rate, rad/s | 0.10682 | **0.08990** | 0.09455 |
| Cascade | Factual rotation entries | 0.01773 | **0.01087** | 0.01097 |
| Cascade | Response velocity, m/s | 0.13221 | 0.11558 | **0.03919** |
| Cascade | Response body rate, rad/s | 0.06316 | 0.05698 | **0.03675** |
| Cascade | Response rotation entries | 0.01198 | 0.00875 | **0.00378** |

The largest targeted gain is Crazyflow's quadratic angular response error,
**1.07374→0.30831 rad/s**, a 71.3% reduction. However, its factual body-rate error
remains 28.3% worse than public, worsening in 23/24 primary cells. Its factual/response body-rate parent p95 errors
increase **0.42752→0.47976 / 0.42756→0.54544 rad/s** against public, within the
frozen 1.5 ratio limits. Crazyflow's wind-shift factual aggregate worsens 23.3%;
other public scope aggregates improve. Cascade's primary rate forecast worsens
5.2% against original-data quadratic while remaining better than public. These
losses are retained as improvement work, not erased by the broad gains.

Short horizons expose a further Crazyflow rate-response deficit: public→candidate
errors are **0.00384→0.00560 rad/s at 10 ms**, **0.02127→0.03390 at 50 ms**, and
**0.10885→0.16444 at 150 ms**, 46–59% regressions. The candidate improves against
original-data quadratic at every measured horizon, but its near-parity with
public at 250 ms is not uniform response fidelity. No derivative-accuracy claim
is inferred from those endpoint gains.

[Physical-error chart](../artifacts/2026-09-18/independent-command-excitation-v1-performance.png)
and [horizon chart](../artifacts/2026-09-18/independent-command-excitation-v1-horizons.png)
show the separate forecast and response comparisons.

The shared test cohort has **64/84 complete Crazyflow test parents**, with 20
altitude failures; primary 250 ms truth includes **437/480 factual and 712/768
response queries**. Cascade completes all 84 and retains **480/480 factual and
576/576 response queries**. All eligible predictions are finite. Differences from
previous cohorts' completion counts do not contribute to the within-study gains.

## Collection and fitting effects

Among the 72 training parents, Crazyflow completion drops **49→24**, altitude
failures rise **23→48**, and valid duration falls **199.18→177.54 s**. All parents
remain admitted; 323 of 384 selected training window origins are retained and 61
change. Cascade completes every training parent and retains all 384 window
origins. Both development caches are exactly unchanged.

Crazyflow's parent-centered command covariance condition number falls **389→37**;
on identical valid parent/time prefixes it falls **373→37**, so shortened survival
alone does not explain that change. Those common prefixes also change state
support: speed RMS **3.31→4.19 m/s** and body-rate-norm RMS **2.03→3.17 rad/s**.
Realized offsets retain 91.4–92.4% of assigned RMS, with 15.5–17.4% clipping.
Cascade's command conditioning improves **144→46**, while its state support changes
little. Its elevator already oscillates substantially and clips 25.6% of the new
offsets. Independent excitation is not synonymous with absence of prior movement.

Crazyflow selects candidate step 900, versus step 700 for original-data quadratic
and step zero for public; Cascade selects step 1,000 in all three arms. Changed
training-derived normalization prevents interpreting their raw checkpoint losses
as a common physical accuracy scale. Architecture and optimizer update budgets are
fixed within the quadratic collection comparison; end-to-end FLOPs are not claimed
equal across model families.

## Interpretation and qualification

Independent assigned signs do not guarantee independent realized commands after
clipping and feedback. The intervention changes visited states, valid durations,
selected training windows, normalization and potentially fitting behavior. Any
gain supports this collection recipe; it does not by itself isolate causal
identification. Fresh seed results must be compared between arms on this cohort,
not subtracted from scores on a preceding cohort.

The frozen public-baseline criteria require weighted response RMSE ratio at most
0.90 and factual ratio at most 1.05, with per-simulator primary, condition-scope
and parent-tail limits. The mechanism comparison additionally requires response
ratio at most 0.90, factual ratio at most 1.05, and Crazyflow primary 250 ms angular
response ratio at most 0.80 against the original-data quadratic learner.
Aggregation gives equal simulator weights, half the weight to primary conditions,
and one eighth to each of four shifts; groups and 50/150/250 ms horizons are equal.
These are weighted geometric RMSE ratios, not pooled errors in physical units.
Paired parent-bootstrap intervals are descriptive, not additional gates.

This experiment alone cannot promote the public learner, qualify live updates or
a controller, or establish calibrated envelopes under shifts. Synthetic capability
and public-contract checks remain necessary for a public recipe change.

## Verification and next experiment

The [result record](harness/independent-command-excitation-v1-result.json) anchors
14,453 bundle payloads, both replay reports and the auxiliary analyses. All
**58,596 common/excited data and query arrays**, saved predictions on 4,032 queries,
**342,720 metric rows**, decisions and bootstrap draws replay exactly. The 144
excited training parents are physically regenerated; the 48 development copies
remain exact. Both comparator refits reproduce their saved revisions exactly.

Independent NumPy reductions verify 240 endpoint score groups at 250 ms,
120 physical-comparison entries, 240 parent-tail entries and 12 public tail gates;
they also check all-horizon truth/finiteness counts, training caches, training-only
normalization, assigned schedules and issued-command arithmetic. Other-horizon
score and bootstrap reductions are covered by production replay, not a second
independent implementation. Fourteen disposable-copy alteration cases reject
changed models, data, source claims, schedules, predictions, metrics and decisions,
including coherent rewrites with recomputed local hashes. All **204 focused tests**
pass; Ruff passes. Stage logs preserve every executed experiment stage.

The supported change is the collection mechanism in this controlled benchmark.
Before promoting quadratic complexity, the next experiment will fit the adopted
public learner on these exact excited recordings. Public-original, public-excited
and quadratic-excited will share fresh test conditions. This separates collection
benefit within the public recipe from the architecture comparison on matching
training data. It retains the accepted improvement while testing whether the
simpler existing public learner is sufficient.

## Reproduction

Use the pinned Python 3.12.12, JAX/JAXlib 0.11.1, NumPy 2.5.3 and SciPy 1.18.1
CPU/x64 runtime, installed Crazyflow 0.3.2 and the protocol-pinned Cascade source.
Set `SCIPY_ARRAY_API=1`, `JAX_ENABLE_X64=1`, and put this checkout's `src` plus the
pinned Cascade `src` on `PYTHONPATH`. The trusted predecessor bundles named in the
protocol must exist and verify before new stages run.

Run `python -m glassbox.experimental.command_excitation_experiment` with a stage,
simulator (`crazyflow` or `cascade`) and `--output <bundle>`. Generate both simulators
before fitting either. Fit `--arm baseline`, then `--arm original`, then
`--arm candidate` for each simulator; evaluate both; finalize once. Capture the
printed root SHA256 externally. Replay both with `--expected-bundle-sha <sha>`.
Replay regenerates common and excited physical data and saved-model predictions
without fitting, and reproduces metric rows, decision criteria and bootstrap
draws. Stage logs, independent reductions and disposable-copy alteration tests
are stored outside the sealed bundle and anchored by the result record.
