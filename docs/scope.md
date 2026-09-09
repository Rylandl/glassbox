# Scope

Glassbox fits differentiable effective dynamics for quadrotors and fixed-wing
aircraft. Its core workflow is to ingest telemetry, fit a model, evaluate
reserved motion against a baseline, and update with new telemetry. The model
and its evidence are the deliverable; control experiments evaluate their use.

## Inputs and models

Inputs are actuator commands or measured actuation, paired with rigid-body
state. `TrajectorySpec` records their roles, units, frames and timing. The
canonical state uses NWU position and velocity, a WXYZ body-to-world quaternion,
and FLU body rates.

The structured models describe thrust, aerodynamic and rotational acceleration,
with actuator response and optional learned residuals. Fitted coefficients
belong to an airframe and its signal contract. See [validation](validation.md)
for the datasets, prediction horizons and comparisons measured so far.

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
