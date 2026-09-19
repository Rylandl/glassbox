# Bilinear ablation and checkpoint diagnosis

Removing the autonomous state-product path repairs much of the quadratic
learner's Crazyflow angular error, but sacrifices too much velocity and rotation
accuracy. Bilinear does not pass the frozen public-progress or quadratic-gain
retention criteria. The useful quadratic structure should be retained for the
next experiment; its fitting tradeoff needs attention. No public recipe changes.

The [protocol](harness/excited-bilinear-ablation-v1.json) was committed at
`d7df690` and the implementation at `60ec112`, before simulation generation or
fitting. The [result record](harness/excited-bilinear-ablation-v1-result.json)
anchors the saved bundle and auxiliary verification. This is one architectural
ablation, including the resulting joint ridge
initialization and optimizer trajectory. It is not inference-time parameter
masking, an objective change, or a mixture of outputs from different models.

## Comparison

Public-excited (`baseline`), quadratic-excited (`quadratic`) and bilinear-excited
(`candidate`) use the exact accepted excited recordings: 72 training and 24
development parents per simulator. Original calibration is regenerated exactly;
excited recordings, historical seals and failed prefixes are imported byte for
byte. Separate provenance binds these old recordings to the new experiment.
Each fit uses identical 384/256 training/development windows, six base
normalization arrays and numerical objective weights. Bilinear and quadratic
also share state-command product scales.

All three receive 1,000 Adam updates, batch size 64, seed zero, width 32,
memory eight, learning rate 0.002 and the unchanged minimum-development-loss
selection rule. This matches data and update budgets, not FLOPs. Parameter
counts in public/bilinear/quadratic are 12,470/13,370/15,170 for Crazyflow and
3,945/4,620/6,420 for Cascade. All four public and quadratic reference refits
reproduce their trusted saved revisions exactly, including caches and metadata.

The fresh cohort has 42 condition cells and 84 test parents per simulator,
using test seeds offset by 5,000,000 from the original benchmark. Only the
selected model from each fit is evaluated on these tests. Intermediate
checkpoints are diagnosed on development recordings only.

## Physical errors

Primary 250 ms endpoint component RMSE, conditional on available truth:

| Simulator | Quantity | Public | Bilinear | Quadratic |
| --- | --- | ---: | ---: | ---: |
| Crazyflow | Forecast velocity, m/s | 0.17064 | 0.15472 | 0.07036 |
| Crazyflow | Forecast body rate, rad/s | 0.21294 | 0.25413 | 0.38413 |
| Crazyflow | Forecast rotation entries | 0.11189 | 0.11294 | 0.03385 |
| Crazyflow | Response velocity, m/s | 0.11629 | 0.06105 | 0.03596 |
| Crazyflow | Response body rate, rad/s | 0.24195 | 0.25330 | 0.31593 |
| Crazyflow | Response rotation entries | 0.10294 | 0.06773 | 0.02751 |
| Cascade | Forecast velocity, m/s | 0.17076 | 0.16309 | 0.11777 |
| Cascade | Forecast body rate, rad/s | 0.10475 | 0.11263 | 0.09273 |
| Cascade | Forecast rotation entries | 0.01750 | 0.01800 | 0.01008 |
| Cascade | Response velocity, m/s | 0.04717 | 0.04690 | 0.03605 |
| Cascade | Response body rate, rad/s | 0.03410 | 0.03725 | 0.03548 |
| Cascade | Response rotation entries | 0.00690 | 0.00721 | 0.00360 |

The [held-out comparison figure](../artifacts/2026-09-18/excited-bilinear-ablation-v1-figures/primary-250ms-heldout.png)
also includes hold-current. Separate [development trajectories](../artifacts/2026-09-18/excited-bilinear-ablation-v1-figures/development-physical-trajectories.png)
and [objective contributions](../artifacts/2026-09-18/excited-bilinear-ablation-v1-figures/development-objective-contributions.png)
show the observed training tradeoffs without evaluating intermediate models on tests.

Bilinear reduces Crazyflow angular forecast/response RMSE by 33.8%/19.8%
against quadratic. Parent-error p95 improves **0.60502→0.41602** for forecasts
and **0.56264→0.41144 rad/s** for responses. All four prospective angular-repair
criteria pass. However, bilinear still loses angular accuracy against public,
and its velocity/rotation losses against quadratic are substantial. Crazyflow
forecast velocity more than doubles and rotation more than triples.

## Frozen decisions

| Numerator / denominator | Forecast ratio (paired 95% interval) | Response ratio (paired 95% interval) | Result |
| --- | --- | --- | --- |
| Bilinear / public | 1.03157 [1.01248, 1.04822] | 0.97414 [0.95655, 0.99043] | Public progress fails |
| Bilinear / quadratic | 1.31502 [1.29315, 1.33337] | 1.29269 [1.27233, 1.31365] | Gain retention fails |
| Quadratic / public | 0.78445 [0.76928, 0.80091] | 0.75357 [0.73818, 0.76656] | Contextual tail guard fails |

The bilinear/public forecast aggregate stays within its 1.05 allowance, but
response improvement is below the required 10%, and Cascade primary response
ratio is 1.17254, exceeding 1.15. Bilinear/quadratic fails both 1.05 aggregate
retention limits, several condition limits and velocity/rotation tail limits.
Angular repair alone therefore does not establish success for the proposed
mechanism. Quadratic/public remains a contextual comparison; it cannot veto an
otherwise successful candidate.

Ratios use the frozen physical floors and equal weights across simulators,
three groups and 50/150/250 ms horizons, with half the condition weight on primary
and one eighth on each of four shifts. The 1,000 paired parent-bootstrap draws
are shared across all comparisons. Intervals are descriptive and conditional
on these parents and available truth; acceptance uses the frozen point criteria.
No criterion was revised after seeing the results.

## What checkpoint evidence establishes

The unchanged fitters were observed at initialization and every 100 updates.
All 66 checkpoints are saved with their actual current parameters, normalization,
cache identities, fit settings and scalar losses. Capturing current parameters
rather than the best-so-far model matters when training regresses. Toy tests show
exact fitting parity for all three architectures, and real trusted-reference
refits provide a second check that observation did not change the selected fits.

Crazyflow public and bilinear select step zero. Every later recorded checkpoint
has worse 250 ms development RMSE in all three physical groups than its own
initialization. This is not solely a checkpoint-selection tradeoff between
groups. It does not prove that every unobserved update is worse.

Quadratic already has much of both its benefit and angular deficit before
gradient training. Development 250 ms velocity/rate/rotation RMSE is
**0.06443/0.31508/0.03787** at step zero, compared with public's
**0.15489/0.21359/0.11204**. At selected step 900, quadratic reaches
**0.06284/0.33368/0.02920**. Its angular deficit therefore begins in the expanded
initial fit; changing only the selection objective would not explain or remove
that initial deficit.

From quadratic step zero to 900, total development objective falls
**0.00483563→0.00373505**. Rotation contribution falls
**0.00314791→0.00149996**, while rate rises **0.00149033→0.00197561** and velocity
rises **0.00019739→0.00025948**. Those are normalized whole-horizon objective
contributions, which need not move with a single physical endpoint. In Cascade,
quadratic training improves all three 250 ms development groups, reaching
**0.11405 m/s / 0.08868 rad/s / 0.01003** at step 1,000. Bilinear selects step 800.

These observations separate initialization from subsequent training empirically.
They do not isolate whether the initial problem comes from feature correlations,
regularization, limited state support or one-step versus recursive fitting.
There is no intermediate-checkpoint held-out evaluation or hindsight reselection.

## Evidence boundaries

Crazyflow primary 250 ms truth covers **416/480 forecast** and **680/768 response**
queries; Cascade covers **480/480** and **576/576**. All eligible predictions are
finite. Crazyflow completes **65/84** test parents and Cascade **84/84**. Missing
slots and failed collection prefixes are retained for every arm.
Excited Crazyflow training still completes only 24/72 parents, versus 49/72 in
the original collection. These are conditional accuracy results, not evidence of
full flight-envelope coverage or controlled flight success.

Public promotion, physically accurate derivatives, uncertainty calibration,
updates and downstream control remain separately unqualified by this ablation.
Checkpoint replay recomputes predictions and diagnostics from saved parameters;
it does not re-solve the optimizer. Timing separates diagnostic work and measured
snapshot callbacks, but general Python tracing overhead remains within fit time.

## Verification and next experiment

All **58,452 common and imported data/query arrays**, selected forecasts on
**4,032 queries**, **342,720 metric rows**, the decisions and bootstrap draws
replay exactly. All 66 checkpoints also pass fresh development prediction and
diagnostic replay; 1,518 snapshot/diagnostic arrays are verified. Replay never
refits an initializer or optimizer.

The independent NumPy audit checks 14,557 sealed payloads, all-horizon
truth/finiteness, 240 endpoint score groups at 250 ms, 180 pairwise physical
comparisons, 360 pairwise tails, 36 broad tail gates and the four angular gates.
It checks exact comparator revisions, 15,360 cache arrays, 392 imported files,
all six checkpoint manifests and 396 derived development diagnostic arrays.
The separate analysis checks selected-test scope/cell/horizon reductions from
saved predictions and truth and supplies the checkpoint figures. Neither auxiliary executes
models or simulators; production replay supplies fresh execution. Bootstrap
draw regeneration is production replay, not an independent implementation.

All **21 disposable-copy alteration challenges** pass, including seven
checkpoint attacks. Coherently altered intermediate parameters or development
predictions survive rebuilt local hashes and fail fresh replay. Altered roster
and settings fail external consistency checks; changed scalar losses fail fresh
objective reconstruction. All **401
focused tests** pass, and Ruff passes. All thirteen real stages complete on their
first attempt with no timeout; no new native-fixture test run is claimed. Figure
layout was revised without changing numerical evidence; both render logs remain.

The next mechanism is an **affine-centered joint ridge initializer** for the full
quadratic model. Change the existing coefficient penalty's center from zero to
the affine initializer already constructed by the current code; retain the same
strength, zero-centered product coefficients and unpenalized bias. The affine
coefficients remain free to move in the joint solve. This adds no fit or tuning
option and changes neither data nor the subsequent objective or optimizer.
Fresh held-out confirmation and the same broad-progress, retained-gain and
angular-repair limits are required. Whether this prior improves the starting
tradeoff is an untested hypothesis.

## Reproduction

Use the pinned Python 3.12.12, JAX/JAXlib 0.11.1, NumPy 2.5.3, SciPy 1.18.1
CPU/x64 runtime and the frozen Crazyflow/Cascade sources. Set
`SCIPY_ARRAY_API=1`, `JAX_ENABLE_X64=1` and `PYTHONPATH` to this checkout's `src`
and pinned Cascade `src`. The trusted predecessor bundles must verify.

Run `python -m glassbox.experimental.excited_bilinear_experiment` with a stage,
simulator and `--output <bundle>`. Generate both simulators, fit `--arm baseline`
for both, then fit `--arm quadratic` and `--arm candidate` for each. Evaluate both
and finalize once. Replay both with `--expected-bundle-sha
d3b5ffc96e0b4e0a4f39887369973583b781af029162d740870f45fe88e19c88`.
Auxiliary scripts and logs remain outside the sealed bundle.
