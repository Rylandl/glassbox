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

## PX4 SITL integration

PX4 is an outer, opt-in contract test rather than a Glassbox runtime dependency.
The package talks only to PX4's standard MAVLink telemetry. It does not import
PX4 code, vendor a simulator, add Gazebo or ROS, or launch Docker from production
code. The maintained external fixture uses PX4's internal SIH physics, so the
only heavyweight component is a disposable prebuilt container owned by the
integration test.

The live boundary in `glassbox.integrations.px4` passively receives
`LOCAL_POSITION_NED` and `ATTITUDE_QUATERNION`. It verifies the PX4 heartbeat and
source system, pairs fresh messages with bounded boot-time skew, normalizes and
makes quaternion signs continuous, and returns the canonical 13-state
NWU/FLU/WXYZ representation. Its frame operations are the same functions used
by offline ULog ingestion. The source exposes no send method and never requests
stream rates, arms, changes mode, or transmits a setpoint.

The MAVLink reader runs continuously on a daemon thread and retains only the
latest coherent state. This is required even in shadow mode: a solver can block
long enough for a UDP receive buffer to preserve old datagrams while dropping
newer ones. Each state also reports estimated source-clock lag relative to the
best observed PX4-boot-time/host-time alignment. That diagnostic reveals
whether freshly decoded data's source time is keeping pace with the host; it
does not mislabel decode time as vehicle-state time.

Ordinary `pytest` runs deterministic tests with fake MAVLink messages and skips
the external fixture. To exercise a real PX4 binary and real MAVLink encoding,
run:

```bash
GLASSBOX_RUN_PX4_SITL=1 \
  uv run pytest -m px4_sitl tests/integration/test_px4_sitl.py -v
```

That test launches the multi-architecture PX4 SIH image pinned by immutable
digest, gives it an explicit Docker host gateway, receives canonical telemetry,
and always stops the container. It does not require a local PX4 checkout or a
Python Docker dependency. The maintained first-stage fixture is `sihsim_quadx`;
additional PX4 vehicle configurations should be separate fixture parameters,
not branches in the model or telemetry contract.

### Artifact-backed NMPC shadow mode

The external test intentionally does not invent a dynamics artifact for the PX4
vehicle. A synthetic or unrelated fit would prove plumbing while producing
misleading model-performance evidence. When a promoted actionable multirotor
artifact and its actual applied command are available, include the complete
telemetry-to-solver path with:

```bash
GLASSBOX_RUN_PX4_SITL=1 \
GLASSBOX_PX4_NMPC_MODEL=artifacts/px4/model.json \
GLASSBOX_PX4_NMPC_COMMAND=0.5,0.5,0.5,0.5 \
  uv run pytest -m px4_sitl tests/integration/test_px4_sitl.py -v
```

For an already-running PX4 instance, the equivalent operator-facing command is:

```bash
uv run glassbox-px4-nmpc-shadow artifacts/px4/model.json \
  --previous-command 0.5,0.5,0.5,0.5 \
  --output artifacts/px4/nmpc-shadow.json
```

The shadow runner executes cold and warm-up controller solves before sampling,
holds the current state as the regulation reference, and applies the artifact's
sample period as the deadline for every measured solve. It records solver
status, fallback rate, model-period deadline misses, message skew, estimated
source-clock lag and real-time ratio, current and predicted validity, and
command bounds. Returned commands are written only to the report. The previous
command remains the measured/applied command because the shadow command is not
being actuated.

A transport test can pass while the real-time gate fails. In particular, PX4
SIH and JAX share host resources in this fixture, so the report treats source
clock progress and solver latency as separate measurements. Any
`deadline_exceeded` sample returns the bounded previous command and is counted
as a fallback; it is not presented as a usable controller output.

This establishes transport and solver integration, not closed-loop PX4 control.
A future command-output adapter remains a separate boundary and must own mixing,
arming and mode checks, stale-setpoint rejection, command timestamps, an
independent watchdog, and safe-mode handoff. The repository still provides no
such adapter and must not be connected directly to real actuators.

## Reproducing the gate

```bash
uv run glassbox-nmpc-benchmark \
  --output artifacts/nmpc-acceptance-results.json
```

The report names the runtime environment, baseline, normalized error, fixed
thresholds, every scenario result, and timing distribution. The same function
is exercised by the test suite.
