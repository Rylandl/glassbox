# Predictive ensembles

Glassbox's first uncertainty workflow is an empirical grouped ensemble around
the deterministic fitter. It is designed to answer one question before the
library adopts a probabilistic runtime contract:

> When independent training groups change, does model disagreement predict
> held-out rollout error?

It does not estimate a Bayesian posterior. A current manifest therefore uses
`artifact_type: empirical_calibrated_predictive_ensemble`,
`method: balanced_group_calibrated_bootstrap_v4`, and `posterior: false`.

## Evaluation contract

For each outer fold, Glassbox:

1. Holds out an entire maneuver profile for outer evaluation when typed profiles
   are present; otherwise it holds out one complete source group.
2. Reserves complete source groups inside every outer-training profile for
   disagreement calibration. When condition labels and replicate groups exist,
   the split is balanced within each profile×condition stratum; otherwise it is
   balanced by profile. Fitting, calibration, and outer-evaluation groups are
   disjoint, while fitting and calibration retain the same maneuver families.
3. Removes every calibration and outer-evaluation trajectory before deriving
   fit statistics or constructing any ensemble member.
4. Resamples the fitting source groups with replacement. When profile labels
   are available, resampling occurs independently within each training profile
   so every member retains the original profile draw count.
5. Represents repeated draws as source-group loss weights. A group omitted by a
   draw contributes no member-fitting windows, but remains available to the
   separate shared-statistics stage. Files and correlated trajectory segments
   are never duplicated or split. Every optimizer window has positive weight.
6. Derives state-error scales, the stability envelope, multi-horizon initial-loss
   normalizers, and residual feature/correction normalization once from the full
   fitting partition. Every member therefore uses the same objective coordinate
   system; bootstrap multiplicity changes only empirical loss and window allocation.
7. Gives each fitting profile equal total empirical-loss mass, then distributes
   that mass according to bootstrap multiplicity among the profile's source
   groups. Unequal numbers of groups per profile cannot silently change the base
   estimator.
8. Fits every member with the existing deterministic model, initialization,
   bounds, multi-horizon loss, and automatic window budget.
9. Fits disagreement-radius scales only on the calibration partition, then
   evaluates raw and scaled radii once on the untouched outer trajectories.

The normal command chooses four to eight members from the number of available
training groups. The Python function accepts an explicit member count for tests
and research audits, but the CLI intentionally does not expose that knob.

Runs are resumable. The request records input hashes, method/artifact version,
source-tree digest, Git revision and tracked-worktree state, Python/JAX/NumPy
versions, JAX backend, and host platform. Each manifest records content hashes
for every model and fit report in addition to its exact training-group
multiplicities. A changed implementation or environment cannot silently reuse
an old summary.

## Prediction geometry

Euclidean state groups use the arithmetic ensemble center. Quaternion members
are normalized, aligned across the double cover, averaged, and normalized
again. Attitude errors and member deviations are shortest-path rotation vectors
in the center's tangent space.

For each requested horizon, calibration is evaluated at the rollout endpoint.
This matters: pooling every intermediate lead time would allow horizon growth
alone to create an apparently strong relationship between disagreement and
error. Predictions are initialized at non-overlapping points by the maintained
window policy, and the common measured initial state is excluded.

Each state group reports:

- ensemble-center vector RMSE;
- mean and p90 member-disagreement radius;
- empirical coverage and radius for 50%, 80%, and 90% member balls;
- the finite-member mass actually attained by each requested radius;
- Spearman rank correlation between disagreement radius and realized error;
- a multivariate energy score.

Four to eight members make this a disagreement ensemble, not a resolved interval
distribution. With eight finite members, the higher-order 90% radius is the
maximum member radius; with four, both 80% and 90% use the maximum. Reports expose
requested and attained mass, finite-member counts, unique bootstrap resamples,
unique fitted parameter members, and unique predictions. Raw coverage is not a
promotion metric at this resolution. A later interval claim requires either a
substantially larger offline ensemble or a separately learned calibration factor
evaluated on subsequently untouched groups.

The current scale stage uses source groups as its finite-sample unit. For each
state group, horizon, and requested coverage, it finds the multiplier required
to attain that coverage within each complete calibration group. It then selects
the corrected rank `ceil((group_count + 1) * coverage)` across groups. A level is
reported as unavailable when that rank exceeds the number of independent groups;
Glassbox never substitutes the much larger number of correlated rollout windows.
For example, 90% calibration requires at least nine independent calibration
groups. The scaled sets remain empirical diagnostics rather than a calibrated
probability distribution.

Endpoint finite-member fraction describes usable forecasts at the requested
horizon. Full-path finite-member fraction and the fraction of members finite on
every evaluated path separately expose trajectories that diverged before
returning an apparently finite endpoint.

The raw member-disagreement balls are uncalibrated. The independently scaled
balls are still diagnostics rather than confidence or credible intervals:
coverage is evidence about this bootstrap construction, not a guarantee for
future flights.

## What is and is not represented

The ensemble currently measures sensitivity to finite fitting groups and their
effective fitted parameters. The scale stage measures how that disagreement
related to error on a separate calibration partition. Neither represents:

- pose, velocity, or rate uncertainty in the initial state;
- wind, turbulence, actuator variability, or sensor noise;
- physics missing from every member's shared model structure;
- probability assigned to structured versus structured-residual candidates;
- out-of-distribution safety beyond the existing validity envelope.

This distinction is especially important for the current multirotor corpus:
shared model-form bias can produce a narrow ensemble whose members are all
wrong. That outcome is a useful failure of the uncertainty hypothesis, not a
reason to widen intervals after inspecting an outer fold.

## Promotion boundary

The benchmark deliberately reports `promotion.status: diagnostic_only` and no
pass threshold. The first outer-fold results are untouched evaluation evidence
only until they are inspected. Once they influence thresholds, calibration, or
implementation choices, they become development evidence. Promotion must then
use a subsequently untouched corpus, airframe, or configuration rather than
relabeling the same folds as protected. Promotion to a serialized runtime
ensemble requires that new evidence show:

- disagreement ranks held-out errors at fixed horizons;
- nominal coverage is reasonably calibrated across complete outer groups;
- useful coverage does not require operationally meaningless radii;
- all members remain finite through the evaluated path;
- the result adds information beyond the validity envelope and a simple
  held-out residual baseline.

Runtime batching, scenario reduction, robust NMPC, online coefficient updates,
and airframe-family priors remain later stages. In particular, bootstrap members
cannot be updated with new data without refitting, and evaluating eight models
inside the current real-time NMPC loop would multiply a workload already subject
to a 20 ms deadline.

## First development result

The first complete v2 run used the 24-flight, four-profile PX4 SIH ground-truth
corpus: six independent source groups each for combined, lateral, vertical, and
yaw maneuvers. Each leave-one-profile-out fold fitted eight unique members at
0.1, 0.5, and 2 seconds, producing 32 models for each of the structured and
structured-residual classes. Every resample, fitted parameter member, and
fixed-horizon prediction was distinct; every member remained finite through all
evaluated paths.

The result partially supports the uncertainty hypothesis:

| Model | Fixed-horizon error/spread | Error/disagreement Spearman | Maximum-member coverage | Energy score |
| --- | ---: | ---: | ---: | --- |
| Structured | 12.2--17.4x | 0.34--0.72 | 0--1.9% | Baseline |
| Structured residual | 1.6--2.9x | 0.33--0.64 | 24.8--76.8% | Better for all 16 state-group/horizon comparisons |

The structured ensemble is a useful negative result: its members often rank
harder predictions, but shared systematic error makes their absolute spread
nearly meaningless. The structured-residual ensemble is materially healthier.
It retains moderate fixed-horizon ranking skill, reduces the error/spread scale
mismatch by roughly an order of magnitude, and improves the multivariate energy
score everywhere. Its maximum-member ball still under-covers, so it is neither a
calibrated distribution nor ready for runtime control.

The validity envelope also explains part of the result. Sixteen of 24 held-out
flights exceeded utilization one, including every lateral and yaw flight. A
post-hoc trajectory-level rank analysis retained positive residual-ensemble
signal after controlling for each flight's maximum utilization, but this is only
exploratory: it uses six trajectories per profile and a complete-flight scalar,
not rollout-local validity. A new corpus must test that relationship directly.

Inspection converted this corpus into development evidence. The next promotion
attempt must use a new airframe, configuration, or untouched corpus and should
evaluate the v3 independent calibration stage without touching that final
evidence. Exact v2 metrics, implementation fingerprints,
artifact hashes, and the evidence decision are recorded in
[`predictive-ensemble-results.json`](predictive-ensemble-results.json).

## First independent calibration result

The v3 benchmark then introduced a strict three-way split. For each outer
profile, eight residual members fitted only two of the remaining profiles, a
third complete profile selected disagreement scales, and the fourth remained
untouched for evaluation. Six calibration source groups support corrected 50%
and 80% ranks but not 90%, which correctly remained unavailable.

Independent scaling worked at the aggregate level:

| Level | Raw coverage MAE | Scaled coverage MAE | Scaled aggregate range |
| ---: | ---: | ---: | ---: |
| 50% | 0.212 | 0.046 | 49.6--59.7% |
| 80% | 0.342 | 0.079 | 82.7--92.9% |
| 90% | n/a | unavailable | six groups cannot support the corrected rank |

All 32 members, resamples, and predictions remained distinct and finite. The
result therefore supports the scale-calibration hypothesis as development
evidence. It does not support the whole-profile calibration partition as the
default: excluding an entire maneuver family from fitting increased center
errors by 1.33--3.33x relative to the v2 ensemble trained on every outer-training
profile. Per-fold coverage also remained heterogeneous even though the aggregate
result was strong.

The next partition should reserve balanced source groups within every
outer-training profile. That keeps fitting and calibration independent while
preserving maneuver-family coverage on both sides. Exact v3 metrics and the
decision are recorded in
[`predictive-ensemble-calibration-results.json`](predictive-ensemble-calibration-results.json).

## Balanced calibration result

The v4 benchmark replaced the whole-profile calibration partition with a
balanced source-group split. In each outer fold, every one of the three
remaining profiles contributed one replicate of each low, medium, and high
condition to fitting and the other replicate to calibration. The resulting
9/9/6 fitting/calibration/evaluation source-group split retained the same
maneuver and condition support on both sides without sharing a source group.

This resolves the principal v3 failure. Relative to the v2 ensemble that used
all 18 outer-training groups for fitting, v4 center errors were 0.92--1.08x as
large in 15 of 16 state-group/horizon cells. The exception was 2-second
velocity at 1.23x. Every member, resample, and prediction remained distinct and
finite, while fixed-horizon error/disagreement rank correlations remained
positive at 0.33--0.61.

Independent scaling again corrected severe raw under-coverage:

| Level | Raw coverage MAE | Scaled coverage MAE | Scaled aggregate range |
| ---: | ---: | ---: | ---: |
| 50% | 0.216 | 0.055 | 51.1--64.3% |
| 80% | 0.352 | 0.079 | 84.3--91.8% |
| 90% | 0.333 | 0.064 | 93.4--98.7% |

The result supports balanced group calibration as the current default
estimator. It does not support a runtime interval contract. With nine
calibration groups, corrected 90% calibration selects the maximum group score;
the resulting bands are consistently conservative and become operationally
weak at long horizons. At two seconds, the mean 90% position and velocity radii
were 2.74 m and 5.79 m/s. The 50% bands were much sharper but heterogeneous
across individual folds, spanning 28.8--86.0% over fold/state/horizon cells.

This corpus has now influenced every ensemble iteration and remains development
evidence. The implementation should stay fixed until it is evaluated on a new
airframe or configuration with enough independent groups to preserve separate
fit, calibration, and outer-evaluation partitions. Exact v4 metrics and the
artifact fingerprint are recorded in
[`predictive-ensemble-balanced-calibration-results.json`](predictive-ensemble-balanced-calibration-results.json).
