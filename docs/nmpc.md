# Glassbox nonlinear model-predictive control

Glassbox NMPC turns one eligible fitted dynamics artifact into a finite-horizon
rigid-body tracker. The interface is intentionally small: a runtime model, a
state estimate, a state reference, the previous command, optional applied
control or latent actuator state, physical tracking tolerances, and optional
state limits. Horizon length, command blocking, line search, regularization,
and iteration count are maintained policies rather than routine user knobs.

The controller is independent of reference generation, state estimation, PX4
transport, and hardware mixing. Terminal-pose docking is not part of this
module.

## Minimal use

```python
import jax.numpy as jnp

from glassbox import (
    NMPCController,
    ReferenceTrajectory,
    RuntimeDynamicsModel,
    SafetyEnvelope,
    TrackingTolerances,
)

model = RuntimeDynamicsModel.load("artifacts/model.json")
controller = NMPCController(
    model,
    TrackingTolerances.for_platform(model.input_spec.vehicle.family),
    SafetyEnvelope(
        minimum_position_m=(-100.0, -100.0, -20.0),
        maximum_position_m=(100.0, 100.0, 100.0),
    ),
)

# NWU position/velocity, FLU body attitude/rates, WXYZ quaternion.
state_estimate = jnp.asarray(state_estimator_output)
previous_command = jnp.asarray(last_command)
reference = ReferenceTrajectory(reference_states, exogenous_forecast)
previous_result = None

result = controller.solve(
    state_estimate,
    reference,
    previous_command,
    applied_command=measured_applied_command,
    warm_start=previous_result.warm_start if previous_result else None,
    deadline_s=control_deadline_s,
)

if result.command_usable:
    canonical_command = result.command
else:
    canonical_command = result.command  # explicit bounded fallback
```

`reference_states` must have `controller.prediction_steps + 1` rows and 13
columns. An exogenous forecast has one row per prediction interval in the exact
typed order recorded by the artifact. `controller.hold_reference(state)` is the
short path for regulation.

`applied_command` is an optional measured actuator value expressed in the
controller's actionable command coordinates. Advanced estimators can instead
provide the complete `latent_state`; passing both is rejected. If neither is
available, the controller initializes lag state from `previous_command`.

The first cold solve and the first warm-started solve compile separate JAX
paths. Run both and discard their commands before entering a timed control loop.
Compilation must never happen after arming.

## Eligible models and airframes

The runtime contract requires a sample period, a training-derived body-velocity
and angular-rate envelope, complete control roles and bounds, latent actuator
state semantics, and an optional prediction-horizon certificate. Loading the
artifact never invents missing runtime facts.

Direct control is allowed only for `normalized_command` and
`normalized_generalized_command` channels. Measured RPM, squared rotor speed,
surface angle, and normalized actuator output are observations, not commands.
Those artifacts fail closed unless the integrator supplies an explicit typed,
JAX-compatible `ActuationMap` from bounded commands to model inputs.

The solver itself has no conventional-airframe or motor-layout branches. It
uses ordered typed channels and roles. Current tests cover:

- four-motor multirotors;
- conventional throttle/roll/pitch/yaw fixed wings;
- three-channel flying wings with generalized elevon roll and pitch commands;
- optional flap authority; and
- structured and structured-residual dynamics.

Learned parameters remain airframe-specific. Hardware mixing stays downstream:
for example, a flying-wing adapter converts the returned generalized roll and
pitch commands into left/right elevon commands.

## Objective, constraints, and outcomes

State error has 12 local coordinates: position, velocity, shortest quaternion
log-map attitude error, and angular velocity. Quaternion signs are equivalent;
components are never subtracted as a tracking metric. Errors are divided by
physical tolerances before aggregation.

Command limits are hard: every direct-shooting iterate is projected into the
typed channel bounds. Command change, model-validity excess, and supported
state limits are dimensionless soft penalties. `SafetyEnvelope` currently
supports minimum/maximum world position, maximum world speed, and maximum body
angular speed. These state limits are preferences, not invariant-set or
collision guarantees.

Important result fields are:

- `status`: converged, finite iteration-limit plan, or an explicit failure;
- `command_usable`: whether the command comes from a finite optimized plan;
- `predicted_states`, `predicted_latent_states`, and `predicted_commands`;
- initial/final objective, iteration count, gradient norm, and solve time;
- maximum command-bound violation, model-validity utilization, and normalized
  safety-limit violation; and
- an opaque `warm_start` for the next receding-horizon solve.

An iteration-limit result is a finite bounded best plan and is marked as such;
it is not falsely labeled converged. Invalid estimates, non-finite objectives,
line-search failure, and exceeded deadlines return the finite bounded previous
command (or channel midpoint when the previous command itself is invalid) with
`used_fallback=True`. A deadline cannot preempt an already executing JAX device
call; the elapsed-time check rejects its output afterward. A supervising loop
still needs an independent watchdog.

## Measured capability

The maintained gate and its fixed pre-tuning thresholds are defined in
[`nmpc-design.md`](nmpc-design.md). The latest machine-readable run is
[`nmpc-acceptance-results.json`](nmpc-acceptance-results.json).

On the recorded Apple M3 CPU run with JAX's CPU backend and a 50 ms model step:

- all eight nominal and eight model-mismatch cases were finite;
- there were no fallbacks and no command-bound violations;
- every mismatch case stayed within the fitted model-validity envelope;
- equal-scenario geometric tracking error was `0.646x` the non-optimizing trim
  baseline nominally and `0.520x` under parameter mismatch; and
- the median of per-scenario post-JIT median solve times was about `60 ms`.

Each scenario records median, p90, and maximum post-JIT time. Cold compilation
took multiple seconds for each novel model/control shape. The observed solve
distribution does **not** establish a 20 Hz real-time capability: the median is
already longer than the 50 ms model step and tail latency is materially higher.
No real-time or flight-safety claim is made.

The baseline holds the model-derived hover or level-flight trim command. It is
deliberately non-optimizing and recorded with every result. The gate also
requires no hidden per-scenario regression, so the aggregate cannot conceal a
failed airframe or maneuver.

## PX4 SITL integration path

The next deployment stage is shadow-mode PX4 SITL, not real hardware:

1. Fit and promote an artifact whose control inputs are actionable commands,
   or bind a reviewed `ActuationMap`.
2. Convert PX4 NED/FRD state estimates into Glassbox NWU/FLU/WXYZ at the
   telemetry boundary. Reuse the audited frame conversions in the ULog adapter;
   do not duplicate signs inside the controller.
3. Build an absolute `ReferenceTrajectory` at every cycle. Reference generation
   owns path feasibility and terminal behavior.
4. Maintain actuator latent state from measured applied command, command
   history, or a dedicated estimator. Feed forecast wind only through the
   artifact's typed exogenous channels.
5. Compile both cold and warm solver paths before starting the SITL clock. Run
   shadow-only solves first and record missed deadlines, validity utilization,
   fallback rate, and prediction error without transmitting commands.
6. Add a platform adapter that consumes the returned canonical command roles
   and writes the selected PX4 offboard/SITL interface. The adapter owns mixing,
   arming state, mode checks, stale-setpoint rejection, and command timestamps.
7. Enable bounded SITL actuation only after p90 latency fits the selected loop
   deadline, the artifact remains inside its envelope, and an independent
   watchdog can return PX4 to its existing safe mode.

The current repository does not provide that command-output adapter or watchdog.
It must not be connected directly to real actuators.

## Reproducing the gate

```bash
uv run glassbox-nmpc-benchmark \
  --output artifacts/nmpc-acceptance-results.json
```

The report names the runtime environment, baseline, normalized error, fixed
thresholds, every scenario result, and timing distribution. The same function
is exercised by the test suite.
