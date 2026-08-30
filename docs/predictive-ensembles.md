# Predictive ensembles

Glassbox's first uncertainty workflow is an empirical grouped ensemble around
the deterministic fitter. It is designed to answer one question before the
library adopts a probabilistic runtime contract:

> When independent training groups change, does model disagreement predict
> held-out rollout error?

It does not estimate a Bayesian posterior. A saved manifest therefore uses
`artifact_type: empirical_predictive_ensemble`,
`method: group_bootstrap_shared_statistics_v2`, and `posterior: false`.

## Evaluation contract

For each outer fold, Glassbox:

1. Holds out an entire maneuver profile when at least two profiles are present;
   otherwise it holds out one complete source group.
2. Removes every trajectory belonging to that outer fold before constructing
   any ensemble member.
3. Resamples the remaining source groups with replacement. When profile labels
   are available, resampling occurs independently within each training profile
   so every member retains the original profile draw count.
4. Represents repeated draws as source-group loss weights. A group omitted by a
   draw contributes no member-fitting windows, but remains available to the
   separate shared-statistics stage; files and correlated trajectory segments are
   never duplicated or split. Every window reaching an optimizer therefore has
   positive weight.
5. Derives state-error scales, the stability envelope, multi-horizon initial-loss
   normalizers, and residual feature/correction normalization once from the full
   outer-training fold. Every member therefore uses the same objective coordinate
   system; bootstrap multiplicity changes only empirical loss and window allocation.
6. Gives each training profile equal total empirical-loss mass, then distributes
   that mass according to bootstrap multiplicity among the profile's source
   groups. Unequal numbers of groups per profile cannot silently change the base
   estimator.
7. Fits every member with the existing deterministic model, initialization,
   bounds, multi-horizon loss, and automatic window budget.
8. Evaluates ensemble predictions only on the untouched outer trajectories.

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

Endpoint finite-member fraction describes usable forecasts at the requested
horizon. Full-path finite-member fraction and the fraction of members finite on
every evaluated path separately expose trajectories that diverged before
returning an apparently finite endpoint.

These are uncalibrated member-disagreement balls, not confidence or credible
intervals. Coverage is evidence about the bootstrap construction; it is not a
guarantee for future flights.

## What is and is not represented

The ensemble currently measures sensitivity to finite training groups and their
effective fitted parameters. It does not represent:

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
