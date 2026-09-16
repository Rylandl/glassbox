# Platform onboarding interfaces

Design record, 12 September 2026. In-memory fitting, the cross-family
[walkthrough](guides/platform-onboarding.md), and an experimental
[streaming replay](guides/platform-onboarding.md#streaming-refinement), and a
[bounded learner during simulated tracking](streaming-refinement.md) are
implemented. The remaining
extensions below are proposals. Blocks marked `text` are illustrative call
sequences rather than runnable examples. The [scope](scope.md) defines
the product boundary; [validation](validation.md) owns measured results.

**Direction clarified, 14 September 2026:** one opinionated model and learning
recipe is the product target. The [opinionated onboarding contract](scope.md#opinionated-onboarding)
governs the proposals below. Experimental representation choices, windowing
controls, and selection policies stay with the research harness. Adding a
reusable research type is not, by itself, a reason to expose another consumer
type or configuration option.

## The promise and the unit of reuse

Given declared platform facts and limited calibration telemetry, produce an
executable dynamics model with enough evidence for a consumer to evaluate its
use in a specified control task. Measure the engineering and data required to
reach that point. The target is repeatable onboarding across airframes and
revisions using one maintained learning recipe. The current supported path uses
structured family-specific dynamics; the generic-model research must demonstrate
the shared no-tuning recipe before replacing it.

Reuse three existing objects:

| Object | Consumer question | Responsibility |
| --- | --- | --- |
| `Trajectory` | What was observed and commanded, on which vehicle? | Arrays, time base, signal contract, labels, and provenance |
| `ExecutableModel` | How does this vehicle respond to these inputs? | Parameters, command mapping, actuator dynamics, timing, and operating support |
| `DynamicsBelief` | What evidence accompanies this prediction? | Executable model, parameter information, measured forecast error, and update history |

Use `fit(sources)` and its `FitOutcome` as the primary workflow. Existing
`FitSpec` controls remain available for compatibility and advanced experiments;
ordinary onboarding should not need them.
There is no demonstrated need for a new `Platform`, `Fleet`, or `Autonomy`
object wrapping these owners. A platform catalog can store artifacts and their
configuration identities without becoming a required runtime dependency.

## Workflow and ownership

| Step | Platform owner supplies | Glassbox supplies | Consumer owns |
| --- | --- | --- | --- |
| Describe | Family, configuration identity, state source, channel meanings, bounds, timing, and available actuation mapping | Contract validation and supported family schema | Sensor calibration and hardware integration |
| Calibrate | Canonical telemetry and declared data roles | Fit, parameter evidence, and forecast-error measurements | Means of obtaining telemetry and the calibration budget |
| Evaluate | Reserved flights, requested horizons, and declared comparisons | Reproducible prediction metrics and provenance | Performance requirements and independent behavior evaluation |
| Use | Current state, actuator initialization, commands, and exogenous inputs | Differentiable prediction and model evidence | Objective, optimizer, estimator, and execution scheduling |
| Update | Fresh telemetry from the declared configuration | New belief and update diagnostics | Re-evaluation and adoption of the revised artifact |

Required prior knowledge must be visible. Record the source of channel mapping,
actuator bounds, state estimates, model initialization, and any inherited
parameters. Record an existing stabilizer or pilot as a calibration dependency.
Do not silently count simulator ground truth as available platform telemetry.

The same task API need not request the same motion from each family. A consumer
must respect available authority and declared limits. Sharing a model interface
does not make a fixed wing and a quadrotor dynamically interchangeable.

The call sequence can stay small. This sketch uses the current API. Inputs are
supplied by the application, so this block is not a runnable example:

```text
from glassbox import fit
from glassbox.workflows.evaluate import evaluate

outcome = fit(calibration_flights)
belief = outcome.belief
evaluation = evaluate(belief, test_flights, horizons_s=(0.1, 0.4))

prediction = belief.rollout(
    observed_state,
    candidate_commands,
    command_history=preceding_commands,
    exogenous=forecast_inputs,
)
# Consumer checks prediction evidence against its own requirements.

updated_belief, update = belief.absorb(fresh_telemetry)
# Evaluate the revised model before making new performance claims.
```

Here calibration flights contain distinct source groups for training and
forecast-error measurement; test flights are reserved for final evaluation.
The current default reserves the final source group, falling back to the final
flight when groups do not separate the inputs. The caller declares source
identity; Glassbox performs the split and records which data supplied evidence.
Commands and their history use the model's declared command coordinates and
sample period. A measured-actuation model needs an explicit command mapping
before this prediction call. The family and channel layout come from the
artifact; the surrounding workflow does not branch on the airframe name.

## 1. Fit arrays and files through the same coordinator

**Implemented.** `fit` accepts trajectories, paths, or both. It shares source
resolution with `workflows.evaluate.evaluate`, while `DynamicsBelief.absorb`
accepts a trajectory. Simulator applications can enter the public fit
coordinator directly with complete in-memory trajectories.

The existing input type is extended:

```text
fit(
    sources: Sequence[Trajectory | str | Path],
    spec: FitSpec = ...,
) -> FitOutcome
```

Inputs resolve once, then use the existing dataset validation, `Holdout`,
window selection, fitting, and reporting path. The file API and CLI remain
available. Callers do not need to construct training windows.

Source display names and data identity are different. Both input forms receive
a stable identity from canonical content and the signal contract, retaining
caller provenance and group labels. Reports identify fit-reserved data as
forecast-error calibration, and temporal splits preserve source-segment
provenance and preceding commands. Temporary files, list positions, or a
user-supplied name do not prove that datasets are independent. Content identity
alone also cannot prove that adjacent segments came from independent flights.

Keep the present distinction between fit-reserved data, which calibrates
forecast error, and final evaluation data. Today `independent_holdout` in
`evaluate` is a caller declaration. The proposed workflow should expose known
overlap and unknown source relationships rather than presenting that flag as
verified independence. Reuse the existing group and temporal split semantics.

**Acceptance.** Equivalent array and NPZ inputs use the same selected windows,
model computation, and evidence semantics, with serialization tolerances
declared. Mixed inputs preserve provenance. Both families reject incompatible
contracts and configuration pooling identically. Fit calibration data cannot
be mistaken for untouched final evaluation data in the report.

## 2. Make the prediction interface useful without a controller

**Current.** `ExecutableModel.transition` takes physical state, latent actuator
state, a command, and optional exogenous inputs. It returns the next physical
and latent states. `initial_latent_state` reconstructs actuator state from
command history at the model sample period. `DynamicsBelief.rollout` adds
parameter and forecast-error information. These are useful existing seams.

**Recommendation.** Make these operations the primary downstream walkthrough.
Show a caller loading a belief, checking its input contract, initializing
actuation from available data, predicting a bounded command sequence, and
reading the evidence. A consumer should be able to do this before choosing
Glassbox's NMPC or any other solver.

Three initialization sources need explicit treatment:

| Available source | Required semantics |
| --- | --- |
| Command history | State its sampling and initial-history assumption; reconstruction is an estimate |
| Measured actuation | Convert through the declared measurement/actuation contract; measurements and requested commands are distinct |
| An externally estimated latent state | Validate model coordinates and shape; passing an array does not propagate estimator uncertainty |

Prefer improving the existing initialization methods and documentation to
adding a competing model-state container. A later initializer result could
carry the latent array plus source and assumption diagnostics, if examples
show those diagnostics being reconstructed in multiple consumers. It should
never imply that an arbitrary short command history exactly determines the
actuator state.

Keep host-side validation and JAX execution responsibilities explicit.
Parameters and numerical inputs may change within an established static
contract. Shape, channel layout, and integration structure determine when
compilation must change. Existing concrete-interval transitions remain the
supported interface; new timing semantics require their own design and tests.
Executing a longer or differently sampled forecast does not extend its
measured error evidence.

**Acceptance.** The same prediction walkthrough works for multirotor and
fixed-wing beliefs, including a measured-actuation artifact requiring a command
map. It needs neither the stock controller nor private simulator state. Invalid
input semantics fail at a named boundary; estimates and missing evidence remain
visible. Changing fitted values under the same contract must not accidentally
reuse stale values in compiled prediction.

## 3. Answer evidence questions without a single readiness flag

**Current.** A belief exposes parameter completeness, measured error horizons,
operating support, and parameter movement since measurement. Rollouts keep
parameter covariance and empirical forecast-error covariance separate.
`ForecastErrorEnvelope` contains second moments, not calibrated probability
bounds. Updates retain prior support and forecast-error measurements.

**Recommendation.** Begin with a documented inspection recipe over those
properties. If repeated consumers justify a typed summary, it should contain
the following independently addressable facts:

| Fact | Example interpretation |
| --- | --- |
| Command mapping and contract compatibility | The artifact can accept this application's commands |
| Requested versus measured horizon | The computation is possible, but error measurements do not reach this horizon |
| Parameter information | Some structured directions remain unresolved |
| Forecast-error statistic and source | These empirical measurements came from these reserved flights |
| Operating support | This query exceeds an observed velocity/rate range |
| Model revision and evidence revision | Parameters changed after the forecast-error measurements |
| Data relationship | Final evaluation is separate, overlapping, or of unverified independence |

Use a typed reason and associated evidence for a missing or mismatched fact.
Do not collapse these into `ready_to_fly`, a confidence percentage, or a
universal accepted/rejected model. The consumer declares its requirements and
decides what to do with the facts. An application can have a separate admission
policy without changing the meaning of the model's evidence.

An experimental [accuracy readout](accuracy-requirements.md) now compares
observed forecast errors with declared physical allowances, including the
entire forecast prefix and per-recording empirical tails. This supplies
descriptive evidence rather than a readiness flag. Performance requirements
belong to the consumer and do not add learner tuning options.

An unresolved parameter direction does not prove that every behavior is
unusable. Equally, a successful behavior does not resolve that direction.
Task-dependent use of partial information requires its own evidence and an
explicit consumer policy. Preserve existing controller admission defaults.

**Acceptance.** A consumer can distinguish absent error data, an unsupported
horizon, incomplete parameter information, an operating-range excursion, and
measurements predating an update without interpreting JSON nulls or controller
statuses. No output describes zero-padded unresolved covariance as certainty.

## 4. Treat configuration changes and refitting as explicit events

**Current.** `TrajectorySpec.vehicle` carries `VehicleConfigurationSpec`,
including a family and optional configuration identity. Fitting requires
compatible dataset contracts. `FitSpec` does not expose an initial physical
parameter guess; `absorb` updates an existing belief using fresh telemetry.

**Proposal to evaluate after the input change.** Allow an explicit same-family
parameter seed for fitting, either a compatible parameter tree or a named
previous artifact whose parameters are extracted with recorded provenance.
Choose one public spelling after exercising both families. An initialization
guess and a statistical prior are different operations; supplying a seed
must not silently import its information matrix or validation measurements.

For a new revision, fit and evaluate evidence for that revision. Treat
same-configuration incremental updates separately from transferring parameters
to another configuration. A new model artifact must retain which revision it
describes, which artifact initialized it, and which measurements were refreshed.
Existing `absorb` behavior must not be relabeled as automatic revalidation.

**Acceptance.** An inherited seed can reduce fitting work without transferring
its evidence. An incompatible family or channel contract is rejected before
optimization. Old and new artifacts remain independently inspectable. An
update that changes parameters visibly leaves prior error measurements at
their original revision.

## 5. Keep behavior composition downstream

The current `PlanModel` protocol already separates a solver from a model, and
`plan_model` adapts a belief to the standard tracking objective. The platform
layer should remain useful to consumers that supply a different objective.
Custom task costs and constraints belong to those consumers. If they need to
copy actuator rollout or covariance propagation, improve that reusable model
boundary instead of adding each task to `DynamicsBelief`.

Reference generation, objective evaluation, optimizer warm starts, and elapsed
execution belong to the planning/control layer. A reference controller is a
useful compatibility test for the model interface. It does not define the
product's scope, and a new control feature needs independent justification.
Do not make this proposal depend on a new universal planner or solver.

## A small first delivery and a platform benchmark

For subsequent generic-model work, freeze one proposed default recipe and test
the complete onboarding workflow. Ablations should justify a design change or
remove an unnecessary mechanism. Do not turn an inconclusive comparison into
another required knob. Record per-platform overrides as exceptions to the
product goal, even when they improve benchmark error.

In-memory fitting and the current-API walkthrough for both families are the
first delivery, preserving existing file behavior and split semantics:
calibration data, fit, independent evaluation, command-based prediction, and
explicit update. Use this walkthrough to decide whether evidence inspection
or latent initialization actually needs another public type. Keep the seed
proposal separate from that first compatibility change.

Then freeze the onboarding recipe and evaluate held-out airframes or revisions
within both supported families. Each test platform supplies its own limited
calibration data. Reserve separate flights for final evaluation, and keep
development platforms separate from test platforms. Record every later
family, adapter, model, or controller modification as additional onboarding
work rather than silently retuning the benchmark.

| Outcome | What to record |
| --- | --- |
| Prior knowledge | Platform facts, inherited models, state source, and calibration dependencies |
| Calibration cost | Flight duration, dataset size, compute, and repeated collection |
| Engineering effort | Human time and interventions, distinguishing a new adapter from a new model family |
| Model quality | Held-out forecasts at declared horizons and conditions, with evidence semantics |
| Behavioral utility | Predeclared tracking and disturbance-rejection tasks, controller configuration, and actual observation assumptions |
| Execution cost | Target hardware, initialization/compilation, steady computation, and deadline behavior |
| Revision handling | What was reused, what was remeasured, and whether old evidence was carried forward |

Use an inherited/default model, the fitted model, and an expert-developed
reference where available. Report the performance-versus-onboarding-effort
tradeoff. A simulator with privileged state is useful for development but must
not be counted as a demonstrated low-effort hardware onboarding result.

## Decisions to revisit with evidence

The streaming replay adds a session coordinator in `workflows.refinement`.
It owns recording cursors and model selection; `DynamicsBelief` continues to
own the numerical update. This is a concrete need for coordination, without
introducing a platform catalog or changing the root API. It retains active,
scored candidate, and updated candidate as distinct revisions. The application
adopts the revision that was scored, even though the learner has moved on.

The first replay accepts direct command models. A measured-actuation model's
training controls and its mapped runtime commands are different coordinates;
one telemetry array cannot silently serve both. A future adapter must supply
both synchronized streams or an explicit conversion before supporting that
case. The tracking experiment now supplies bounded aligned-transition transport,
explicit gaps, limited command history, and consumer-owned controller handoff.
It also exposed unnecessary parameter-covariance propagation for a consumer
that only needs the forecast mean. `belief.rollout` can now omit that calculation
explicitly, returning `None` for that covariance while preserving other evidence.

Raw telemetry still needs an estimator/timestamp contract and durable accounting.
The worker shares a process and compute resources with tracking; its queue does
not establish scheduling isolation. Recorded deadline misses and mixed tracking
outcomes make execution isolation and stronger adoption evaluation the next
questions to investigate. Keep those separate from score-before-learn ordering.

- Whether an inspection recipe needs a typed result once two independent
  consumers use it.
- Whether initialization diagnostics warrant a new return type or an optional
  companion diagnostic.
- Whether parameter seeds belong in `FitSpec` or in a separate fit argument,
  given serialization and provenance needs.
- Whether reusable belief prediction suffices for task composition before
  extending `plan_model`.

These decisions should follow the smallest working cross-family examples.

The [Cascade follow-up](cascade-refinement.md) exposed a concrete command-contract
gap: requested surface angles in radians could be fitted but were not recognized
as actionable inputs. `surface_angle_command` now declares those requested
coordinates explicitly, while measured angles remain non-actionable without a
mapping. The downstream experiment also replaces its embedded synthetic dynamics
with a small observation/command plant interface. The learner itself needs no
simulator-specific branch. These are demonstrated interface changes; the
platform benchmark still needs broader airframes and observation conditions.

The first delivery does not require a fleet registry, a generic model plugin
system, new uncertainty claims, or changes to the existing control policies.
