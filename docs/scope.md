# Scope

Glassbox turns platform telemetry into differentiable effective dynamics,
paired with evidence about their prediction quality and operating support.
It currently supports quadrotor and fixed-wing model families. Its core workflow
is to ingest telemetry, fit a model, evaluate reserved motion against a
baseline, and update with new telemetry.

The platform-level goal is to reduce the calibration data and engineering
effort needed to make a new airframe or hardware revision useful to downstream
control and autonomy. The model and its evidence are the deliverable. Control
experiments measure whether that deliverable supports a declared capability.
Reduced onboarding effort across heterogeneous platforms is an objective to
measure; current results are recorded in [validation](validation.md).

## Opinionated onboarding

The [generic engineering record](generic-engineering.md) governs current work:
one incumbent, a fixed acceptance contract, bounded implementation milestones,
and explicit decisions that survive new sessions. Research investigations remain
evidence; their historical next-step proposals are not an open-ended work queue.

Design decision, 14 September 2026: the product should provide one maintained
model and training recipe through a small fit, predict, and update workflow.
Success includes using that recipe on a new platform without choosing features,
model families, optimizers, regularization strengths, or selection policies.
Any internal adaptation is part of the versioned algorithm and is evaluated
with it. Research variants are tools for settling design decisions, not a menu
of supported onboarding configurations.

The platform owner supplies facts: observed and commanded signals, their units
and timing, recording boundaries, and configuration identity. A downstream
consumer supplies its prediction request and performance requirements. Glassbox
owns compatible-data checks, segment handling, scaling, temporal representation,
fitting, validation against simple references, and revision-specific evidence.
Routine use should not require callers to assemble windows or choose a guard.

The generic learner should use one shared recipe across supported signal
contracts, without caller-supplied quad or fixed-wing equations. A recipe may
contain learned components, but its components and update rules are maintained
as one model design. The target is a coherent executable dynamics model with
one prediction/update contract. Independent horizon heads and selectors that
combine incompatible outputs remain research diagnostics.

Evaluate the entire default workflow, frozen before new-platform evaluation,
under declared calibration budgets. Platform-specific tuning counts as
additional onboarding work and a failure of the no-tuning target for that case.
A change to the default is a versioned model change with regression evidence;
it must not alter an already serialized model's behavior or evidence.

Simple use must preserve honest failure behavior. Missing signal meanings,
insufficient excitation, unsupported prediction conditions, or an update that
has not been re-evaluated should be reported as concrete evidence limitations.
The user should not need to tune the learner to discover them. A single
confidence number or automatic claim of control readiness would not satisfy
this contract.

**Current status:** `fit(sources)` already supplies defaults and returns a
`FitOutcome` containing a `DynamicsBelief`. The shipped model families remain
structured, and `FitSpec` retains advanced controls. The experimental generic
learners have not yet established a default that meets this contract. A
[single fixed generic candidate](default-recipe.md) now implements the small
workflow with automatic data handling and immutable batch updates. Its
cross-recording prediction failures keep it experimental; it is not promoted
to the supported workflow.

## Product boundary

Glassbox owns signal and actuation contracts, dynamics identification,
executable prediction, parameter evidence, forecast-error measurements, and
updates with provenance. A fitted artifact should let a consumer determine
what inputs it accepts, how to initialize and execute it, and what evidence
exists for the requested prediction.

State estimation, mission objectives, trajectory selection, hardware transport,
and deployment decisions belong to consumers and integrations. The experimental
control package provides a reference consumer of the same model interface.
Using another controller should not require reimplementing actuator dynamics
or interpreting serialized belief internals.

The current structured implementation supports family-specific equations,
actuator layouts, and feasible behaviors. Adding an airframe within a supported
family primarily requires its signal contract and calibration data. Extending
those structured families can require new equations and tests. Record that
engineering work separately; the generic learning research targets the shared
recipe described above.

Limited calibration assumes a declared way to obtain informative telemetry.
An existing stabilizer, pilot, state-estimation system, or test fixture is part
of that starting point. Bootstrap identification is a separate experimental
capability with its own assumptions. Neither fitting nor a successful rollout
establishes that an arbitrary uncalibrated vehicle can be flown.

## Inputs and models

Inputs are actuator commands or measured actuation, paired with rigid-body
state. `TrajectorySpec` records their roles, units, frames and timing. The
canonical state uses NWU position and velocity, a WXYZ body-to-world quaternion,
and FLU body rates.

The structured models describe thrust, aerodynamic and rotational acceleration,
with actuator response and optional learned residuals. Fitted coefficients
belong to an airframe and its signal contract. See [validation](validation.md)
for the datasets, prediction horizons and comparisons measured so far.

Configuration identity distinguishes airframes and revisions even when their
channel schemas match. Reusing an earlier model can be an explicit starting
point for identification; it does not transfer that model's validation evidence
to a changed configuration. A shared workflow does not imply pooling telemetry
from different configurations into one vehicle fit.

## Design

Three objects carry the identification workflow:

- `Trajectory`: the recorded signals and their meaning.
- `ExecutableModel`: equations bound to input channels, sample period, actuation
  mapping and a training-derived velocity/rate operating envelope.
- `DynamicsBelief`: the model, local structured-parameter information and
  empirical forecast error.

Construct these objects where the fit or measurement produces them. Reports
summarize results; serialization belongs at artifact boundaries. Prediction,
parameter information and forecast error have different meanings and remain
separate in the [belief API](concepts/dynamics-beliefs.md).

| Responsibility | Owner |
| --- | --- |
| Signal contract and flight boundaries | `core.data`; source translation in `io` |
| Fit orchestration and numerical optimization | `fitting` and `core.identification` |
| Vehicle/actuator rollout and prediction errors | `core.dynamics` and `core.metrics` |
| Parameter evidence, updates and forecast-error statistics | `belief` |
| Scoring protocols and reserved-data evaluation | `workflows.evaluate` and `workflows.holdout` |
| Planning, bootstrap and supervision | `control`; transport in `integrations` |

Reuse these owners before introducing another representation or interface.
Keep parameter updates and calibration on the same prediction equations used
by evaluation. Split modules when responsibilities need independent ownership,
not to distribute line count.

Keep measured prediction error, local parameter information, and observed
operating support distinguishable. In particular, unresolved parameter
directions are unknown, an operating-envelope check is not an accuracy bound,
and the current forecast-error second moments are not calibrated probability
limits. Evidence belongs to the model revision and conditions at which it was
measured. Updating parameters does not refresh those measurements.

## Onboarding as the evaluation unit

Evaluate the time and evidence needed to reach a predeclared downstream
capability. Record required prior knowledge, calibration duration, engineering
time, manual interventions, and held-out control performance. Compare an
inherited or default model, the fitted model, and an expert-tuned reference
where one is available. The expert reference provides performance context;
onboarding effort is a separate outcome.

Keep calibration, model selection, and final evaluation roles explicit. A
platform benchmark must hold out airframes or revisions as well as flights,
freeze the workflow before evaluation, and record platform-specific changes.
Ordinary flight holdout measures performance on new flights of the fitted
vehicle; it does not demonstrate onboarding a new vehicle.

Implemented and proposed ergonomic changes, with their acceptance criteria,
are described in [platform onboarding interfaces](platform-onboarding.md).
The [onboarding walkthrough](guides/platform-onboarding.md) exercises the
current API; further proposals are marked separately.

## Downstream work

NMPC, bootstrap identification and simulator integrations are experimental
consumers of the belief. The current PX4 link is read-only. Controller results
are simulation experiments, with no flight-safety or hard real-time guarantee.
Their interfaces and experiments are described under [NMPC](concepts/nmpc.md)
and [bootstrap identification](concepts/bootstrap-identification.md).

Prioritize the identification workflow over additional controller features or
integration surfaces. Keep research variants tied to their experiments rather
than extending the production API to accommodate each one.

## Evidence

[Validation](validation.md) owns the recorded comparisons and their artifact
links. [Contributing](../CONTRIBUTING.md#recorded-results) describes reproduction.
The [literature review](literature-review.md) and investigation pages retain
experimental decisions; the [original proposal](history/idea-2026-08.md) is
historical background.
