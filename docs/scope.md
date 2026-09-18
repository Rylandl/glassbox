# Scope

Glassbox learns a differentiable dynamics model from observed signals and
commands using one maintained recipe. The public workflow is
`fit(recordings)`, `model.predict(...)` and `model.update(recordings)`.
The generic learner is adopted; current evidence and remaining gaps are in
[status](status.md).

## Opinionated onboarding

The [charter](charter.md) governs development. Callers supply observed and
commanded signals, units, frames, a uniform sample interval, source recording
boundaries and configuration identity. Glassbox owns scaling, context,
representation, training windows, optimizer, checkpoint selection and
revision-specific evidence. These are algorithm decisions, not a menu of
onboarding options.

One recipe is fitted separately to each system. Current measurements favor it
over structured comparators on four of five flight corpora, with a known ARP
deficit. This supports the adopted development direction; it does not establish
transfer of one set of weights to arbitrary unseen systems.

## Inputs and models

The public data types are `SequenceSegment` and `SequenceCollection`.
A segment contains a finite observation array, commands aligned to its
transitions and a sample interval. A collection preserves recording identities,
contiguous segment boundaries, ordered channel descriptions and configuration
identity. The [learner contract](learner.md#recording-contract) gives the array
shapes and validation rules.

Observations may be any declared Euclidean signals. The learner does not
require a vehicle family, rigid-body equations or a physical parameter
catalog. Clock alignment, sensor validity and the meaning of commands are
supplied by the caller. Missing intervals form separate segments; they are
not filled with inferred observations.

`LearnedDynamics` carries the fitted recipe, prediction contract, measured
error envelope and retained update data. Predictions are conditional on the
provided future commands. Updates return a new revision and preserve the old
one, with provenance linking the two.

The generic recording NPZ format is owned by `glassbox.io.recordings`.
Canonical flight trajectories from telemetry adapters remain a separate
format. `from_trajectories` converts named canonical recordings to the
established 15 observed channels with an explicit configuration ID.

## Product boundary

Glassbox owns validated recording collections, dynamics learning,
differentiable forecasts, saved model revisions and forecast-error evaluation.
Applications own calibration data collection, state estimation, future command
requests, objectives, execution scheduling and model adoption.

Model accuracy and a usable representation are the primary product goals.
Controllers may come from Glassbox or another project. A saved model should
explain what each input/output means, how much history and which forecast
horizons it supports, what error evidence exists, and which recipe and revision
produced it. A usable representation does not imply recovered physical
coefficients or human-readable governing equations.

The current Python/JAX interface provides batched, differentiable forecasts and
saved revisions. Portable runtimes, solver-specific exports and dedicated
linearization interfaces are possible integration work, not current guarantees.
One general controller is a desirable reference application, not a requirement
that all consumers share objectives, actuation layouts or scheduling.

A stabilizer, pilot or test fixture used to gather data is a calibration
dependency. Learning from those recordings does not establish that an unknown
vehicle can safely collect its own data.

Forecast error, envelope coverage and control adequacy are separate claims.
The envelope's development data also select checkpoints; new-recording
coverage is measured rather than guaranteed. Euclidean forecasts do not enforce
physical constraints or prove support outside the observed conditions.

## Design

| Responsibility | Owner |
| --- | --- |
| Public fit, prediction, immutable updates and saved revisions | `glassbox.learner` |
| Recording/segment contracts and window provenance | `glassbox.recordings` |
| Generic recording archives and explicit canonical conversion | `glassbox.io.recordings` |
| Read-only held-out forecast evaluation | `glassbox.workflows.forecast` |
| Frozen experiments and artifact replay | `glassbox.experimental.harness` and qualification tools |
| Planning and transport consumers | `glassbox.control` and `glassbox.integrations` |

Structured dynamics, fitting, belief artifacts and their remaining consumers
are still present in their owning modules for benchmarks and unmigrated
integrations. They are outside the package-root consumer API. Their eventual
removal remains part of the charter's lean criterion.

## Onboarding as the evaluation unit

Measure calibration data, prior knowledge, engineering effort and performance
on untouched recordings and declared downstream tasks. Compare gains and
losses against the adopted generic baseline with criteria chosen before
fitting. A structured comparator is performance context; its failure or
success does not define application sufficiency.

Whole-recording holdout measures a fitted system on fresh recordings. Claims
about onboarding new systems additionally require held-out systems and a
frozen workflow. Keep repeated model selection separate from final evaluation.

The [onboarding contract](platform-onboarding.md) assigns responsibilities, and
the [runnable guide](guides/platform-onboarding.md) shows the public API.

## Downstream work and evidence

Control, bootstrap identification and live model swapping remain experimental
consumers. The qualified anticipatory oracle and both isolated research learners
now meet the declared simulator tracking task. The adopted public learner has
not yet been tested with that controller, and earlier live-swap failures remain
unresolved. These are separate from direct qualification of model accuracy.

The proposed JSBSim benchmark prioritizes held-out prediction and command-response
accuracy, coverage, computation and onboarding effort under one fitting recipe.
It must pin the inventory, supplied observations, operating conditions, training
budgets and evaluation criteria before measuring models. Simulator internals
belong to the evaluator; the learner receives only its declared recordings.
Keep every inventoried model and setup failure visible. Control demonstrations
add application evidence without defining every model's accuracy score.

[Status](status.md) records current evidence and replay instructions.
[Validation](validation.md) and the research reports retain historical
benchmarks. [Contributing](../CONTRIBUTING.md#recorded-results) describes their
artifact procedures.
