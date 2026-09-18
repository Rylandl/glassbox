# Learned state–command interactions

The [frozen protocol](harness/state-input-interaction-v1.json) investigates one
limitation of the selected Crazyflow predictor: its affine forecast gives
identical additive command perturbations history-independent effects. The
candidate adds a learned bilinear product of every standardized current
observation with every standardized current command. A joint ridge fit
initializes this output path alongside the existing affine output. The existing
neural and memory paths are unchanged, and receive no new product features.

Both arms use identical training/development windows, normalization for existing
features, optimizer settings, minibatch draws and checkpoint selection. Added
parameters and arithmetic are measured costs, so this is matched data and
optimizer budget, not equal FLOPs. No platform identities, equations or paired
response supervision enter fitting. The candidate remains isolated until its
residual, capability and consumer-contract evidence supports promotion.

Calibration recordings, conditions and physics are unchanged. Only test seeds
and identities receive a fixed offset of 1,000,000. Regenerated calibration
arrays and roles must match the preceding trusted bundle before fitting. All
failed collection conditions and missing truth remain visible. The previous
test set is diagnostic evidence, not fresh confirmation.

The primary progress score compares candidate and current generic residuals,
with equal simulator weights, half the weight on primary conditions and half
equally across the four shift families. Within each scope, the three signal
groups and shared 50/150/250 ms horizons have equal weight. Factual and raw
finite command-response errors have separate scores; weak probes are included.
Report absolute physical errors alongside geometric-mean ratios, per-condition
results and parent-error tails. The protocol defines small physical floors to
make ratios near zero well defined.

Research acceptance requires at least 10% aggregate response improvement,
at most 5% aggregate factual regression, bounded per-simulator primary and
per-scope regressions, finite eligible predictions, matching truth cohorts,
and passing integrity checks. Primary 250 ms parent-error tails also have a
predeclared regression bound. These are iteration decisions, not application
accuracy tolerances. Whole-parent bootstrap intervals describe uncertainty
within the two fixed configurations; they do not establish arbitrary-system
readiness. Structured results remain context, not a promotion gate.

Every stage is bounded and sealed. Replays regenerate physics and saved-model
predictions, reductions and decisions. Historical protocols and public source
files remain unchanged during this research comparison.

## Verified result

The bilinear candidate reduces the frozen weighted response-error score by
**14.8%** and factual-error score by **3.8%**, relative to the unchanged generic
learner. These are geometric means of RMSE ratios across the declared systems,
scopes, horizons and groups, not percentage reductions in pooled physical MSE.
Paired parent-bootstrap 95% ratio intervals are **[0.83934, 0.86241]** for
responses and **[0.94970, 0.97334]** for forecasts; all 1,000 draws are available.

Primary 250 ms component RMSE, **current generic → bilinear candidate**:

| Simulator | Quantity | Forecast | Command response |
| --- | --- | --- | --- |
| Crazyflow, available truth | Velocity, m/s | 0.16345 → 0.16323 | 0.17752 → 0.13612 |
| Crazyflow, available truth | Body rate, rad/s | 0.24323 → 0.24648 | 0.31675 → 0.58325 |
| Crazyflow, available truth | Rotation entries | 0.10777 → 0.11325 | 0.11885 → 0.10274 |
| Cascade | Velocity, m/s | 0.16074 → 0.15200 | 0.13391 → 0.08665 |
| Cascade | Body rate, rad/s | 0.11258 → 0.11628 | 0.06233 → 0.05728 |
| Cascade | Rotation entries | 0.01840 → 0.01763 | 0.01245 → 0.01152 |

The candidate **does not pass the frozen research acceptance rule**. Crazyflow's
primary 250 ms body-rate response parent-RMSE p95 increases from **0.45006 to
0.69747 rad/s**, a ratio of **1.549745**, exceeding the declared 1.5 limit.
Every other residual check passes, including aggregate improvement, factual
regression, simulator/scope limits and finite eligible predictions. The
tail-limit miss does not erase the measured mean gains, but it is not waived
after observing results. The public recipe remains unchanged; the saved
candidate is evidence for the next mechanism, not a new adopted baseline.
The regression itself is broader than one tail outlier: Crazyflow's 250 ms
rate-response error increases in all 24 primary cells, by factors of 1.48–2.15.

Cascade response ratios improve in every scope: 0.70584 primary, 0.72066 heading,
0.61304 speed, 0.72333 maneuver and 0.68882 wind. Crazyflow's corresponding
response ratios are 1.04527, 1.04479, 1.04008, 1.04345 and 1.03669. These
scope scores average all three shared horizons and signal groups; individual
velocity gains coexist with rate losses. Full physical-unit results and tails
remain in the sealed decision and summaries.

Both refitted baseline artifacts reproduce their previous fingerprints exactly.
All 192 calibration parents, their 2,016 arrays and automatic 72/24 train/dev
roles per simulator are unchanged. The four fits use the frozen 384/256 windows,
1,000 updates and batch size 64. Crazyflow selects step zero in both arms;
Cascade selects step 1,000 in both. The candidate adds 900/675 coefficients.
The optimizer functions and loop match the baseline AST except the candidate's
dedicated numerical-failure exception. This is matched data and update count,
not an equal-FLOP comparison.

The new Crazyflow collection cohort completes 133/180 parents, with 47 altitude
failures: 27 calibration parents and 20 test parents. Primary 250 ms truth is
available for **426/480 factual and 688/768 response queries**. The same slots
are used by both arms; all missing slots remain represented. Cascade completes
180/180 parents and all primary 480 factual/576 response queries. These
collection failures are not learned-controller failures. Error-envelope coverage
is measured descriptively and is not qualified by this result.

The response-rate regression grows over the Crazyflow forecast: baseline and
candidate errors are 0.00386/0.00388 rad/s at 10 ms, 0.02139/0.02962 at 50 ms,
0.10787/0.21051 at 150 ms and 0.31675/0.58325 at 250 ms. Because the selected
model is the ridge initialization, later Adam updates do not cause this saved
model's regression. Separating autonomous nonlinear state evolution from
state-dependent command effects is the next hypothesis to test; this result
alone does not establish confounding or prove that adding state-state products
will fix it.

Read-only diagnosis attributes 78.5% of the added rate-response MSE to roll and
21.3% to pitch. Roll-response RMSE grows 0.403→0.853 rad/s, with predicted
response RMS shrinking 1.438→0.961 against truth RMS 1.780. Learned affine
command gains shift substantially in the joint fit. All 60 state–command
features correlate above 0.5 in magnitude with some state–state product in the
training data. This motivates a complementary autonomous quadratic basis, but
also warns that simply adding features can increase collinearity. The next
comparison must include both this bilinear mechanism and the adopted baseline.

The externally anchored bundle is
`artifacts/2026-09-18/state-input-interaction-v1`, with `run.json` SHA-256
`cd258714842a73f941d04da33ec2cdc11aa82212cc7b69dae0320cda02bd2184`.
All **55,788 physical/data/query arrays**, saved predictions, **257,040 metric
rows**, decisions and bootstrap results reproduce exactly. Independent NumPy
reductions verify all 180 250 ms score groups, 60 physical decision comparisons,
120 parent-tail records and all 12 primary tail checks. Other horizons and the
bootstrap are covered by production replay, not that independent reduction.
All 12 tamper challenges and 78 focused tests pass; Ruff is clean. No model
optimizer trajectory is independently refitted as part of replay.

The [result record](harness/state-input-interaction-v1-result.json) anchors the
14,063-payload bundle, separate replay and audit reports, logs and
[performance plot](../artifacts/2026-09-18/state-input-interaction-v1-performance.png).
