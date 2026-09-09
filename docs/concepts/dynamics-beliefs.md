# Dynamics beliefs

A `DynamicsBelief` contains:

| Field | Meaning |
| --- | --- |
| `model` | Executable equations, input contract, timing, operating envelope and optional command map |
| `information` | Local information over structured parameters, weighted by one-step innovation noise |
| `forecast_error` | Empirical endpoint error second moments from reserved data, or `None` |
| `provenance` | Sources and update history |

`params`, `input_spec`, `runtime_spec` and `support` delegate to the model.

## Fit and use

```python
from pathlib import Path

import glassbox
from glassbox.core.data import load_trajectory_npz

outcome = glassbox.fit(sorted(Path("flights").glob("*.npz")), glassbox.FitSpec())
belief = outcome.belief
belief.save("artifacts/belief.json")
belief = glassbox.DynamicsBelief.load("artifacts/belief.json")

telemetry = load_trajectory_npz("new-flight.npz")
forecast = belief.rollout(telemetry.states[0], telemetry.controls[:30])
updated, result = belief.absorb(telemetry)
```

The fit selects training windows, optimizes rollouts from their observed initial
states and measures reserved trajectories. Normalization, residual bounds and the
operating envelope come from training data. Reserved data calibrates the one-step
noise and forecast error. Use further data for evaluation of the resulting belief.

A command map is required for `belief.rollout` and control. Direct command
channels get an identity map; measured actuation such as rotor speed needs a
supplied map. Without one, command-based calls raise `NonActionableModelError`.
The underlying dynamics can still predict from measured actuation.

## Parameter information

Fitting and absorption accumulate `sum J.T @ inv(R) @ J` over selected one-step
transitions. `J` differentiates predicted motion with respect to structured
parameters; `R` is the diagonal one-step error second moment, subject to a noise
floor. This is a local information approximation under that noise model.

`scale` normalizes parameter coordinates. Eigenvalues above `rank_threshold`
are called resolved; `estimable` selects which coordinates may be learned.

- `resolved_rank()` and `resolved_subspace()` describe those directions.
- `covariance()` inverts precision on the resolved subspace and pads the rest
  with zeros. Read it with `unresolved_subspace()` and `complete`; the zeros
  are not zero uncertainty.
- `authority(direction)` gives relative information along a direction.
- `information_gain_nats(delta)` measures gain within the previously resolved
  subspace; newly resolved directions appear as a rank change.
- `seeded_from_members(nominal, members)` constructs information from variation
  among related parameter vectors.

`complete` means every estimable structured direction is resolved. Control
requires an explicit override to plan with partial information.

## Forecast error and prediction

`ForecastErrorEnvelope` stores uncentered tangent error second moments at
measured horizons, weighted equally by source group, then trajectory, then
endpoint. It includes mean error and is not a calibrated probability bound.
`covariance_at` interpolates between horizons and holds the last value beyond
them; check `forecast_error_horizon_supported` on a prediction.

`belief.rollout` returns state and actuator traces, `parameter_covariance`,
`forecast_error_covariance`, and operating-envelope utilization. The two matrices
remain separate. `forecast_error_available` distinguishes absent measurements
from zero measured error. The [repeated-fit study](../repeated-uncertainty-calibration.md)
compares both with independent outcomes.

Concrete commands are checked against declared channel bounds. The velocity/rate
operating envelope is advisory: utilization above one reports an excursion.
JAX-traced callers are responsible for command bounds; NMPC projects commands
into them.

## Updates

`absorb` expects fresh, nonoverlapping telemetry at the model's sample period.
It selects finite transitions inside the operating envelope, reconstructs
actuation from all preceding commands and any `Trajectory.control_prefix`, and
adds their parameter information. A capped step with backtracking must reduce
the actual whitened error plus the prior quadratic; otherwise the mean stays
unchanged. Previously accumulated information is retained.

The returned `UpdateResult` includes the usable transition count, before/after
innovation error, information gain and step size. The original belief is
unchanged. Updates retain the original operating envelope and forecast-error
measurements; provenance tracks parameter movement since calibration.

See [adaptive recovery](../validation.md#adaptive-recovery) for the recorded
update experiment and [NMPC](nmpc.md) for using a belief in control.

## Serialization

Direct command maps and their bounds are saved with the belief. Loading an
external map requires `DynamicsBelief.load(path, actuation=your_map)` with a
matching channel contract. Legacy artifacts are loaded through the compatibility
rules in the codec; formats lacking usable parameter evidence load at rank zero.
