# Multirotor flight supervisor

`MultirotorFlightSupervisor` is a thin, stateful authority boundary around a
four-motor controller. It does not consume a Glassbox belief, fit a model, run
an optimizer, or track position. Its job is to decide whether a candidate motor
command may reach the plant.

For fresh valid telemetry and a fresh, finite, bounded, command-usable candidate,
the supervisor returns the candidate unchanged. Invalid or stale telemetry
selects a configured collective hold because geometric arrest cannot be trusted
without a usable attitude and body rate. Command faults, an unusable controller,
or excessive tilt/body rates select a bounded geometric attitude/rate-arrest
command. Arrest remains latched for a minimum interval and until tighter release
limits are met.

The arrest command drives the geodesic attitude error, which is the rotation
vector that would level the vehicle. Its magnitude is the tilt angle itself, so
restoring authority holds across the whole `[0, pi]` range instead of fading out
near inversion the way a `sin(tilt)` cross product does. For small tilts the two
agree to third order in the angle, so nothing changes in normal flight. At exact
inversion the axis is undefined and a fixed positive roll direction is chosen.
Candidate commands within a rounding width of the command bounds, defined as
`1e-6` of the span, are clipped rather than treated as out of bounds, so a
representation-width overshoot cannot latch an arrest. The arrest latch always
starts from the newest time the supervisor has seen, never from a regressed
clock reading, so a clock that jumps backwards cannot let the next valid tick
skip the minimum arrest duration.

```python
import time

from glassbox import MultirotorFlightSupervisor, MultirotorSupervisorConfig

supervisor = MultirotorFlightSupervisor(
    MultirotorSupervisorConfig(
        collective_hold_command=(0.53, 0.53, 0.53, 0.53),
        maximum_state_age_s=0.04,
        maximum_command_age_s=0.02,
    )
)

now = time.perf_counter()
decision = supervisor.supervise(
    state=state_wxyz_nwu_flu,
    state_received_at_s=state_received_at,
    candidate_command=nmpc_result.command,
    command_generated_at_s=command_generated_at,
    now_s=now,
    controller_command_usable=nmpc_result.command_usable,
    previous_applied_command=applied_motor_command,
)
plant.write_motor_command(decision.command)
```

The caller must use one monotonic clock for all timestamps. State layout is the
canonical 13-element Glassbox rigid-body state with WXYZ quaternion storage and
FLU body rates. Motor order and normalized command semantics must match the
canonical multirotor mixer. The collective hold command, command bounds, limits,
gains, and maximum arrest slew are vehicle integration values—not identified
dynamics parameters.

Every decision reports a `SupervisorMode`, typed `SupervisorReason` values,
state and command ages, measured tilt/rate, whether the nominal command was
accepted, and an immutable four-motor command. `reset()` clears the time and
arrest latch for an explicit lifecycle restart.

## Maintained integration evidence

The optional Crazyflow prototype includes a fixed controller→supervisor→hidden
plant campaign with one nominal case and one case for every typed rejection
reason. Each case advances the true plant through one 50 Hz command interval and
requires the expected authority mode/reason, transparent nominal pass-through,
finite bounded motor output, and a finite post-step plant state. Campaign
telemetry is not reused for fitting or belief updates. See the
[Crazyflow prototype report](../experiments/crazyflow-prototype.md) for the recorded result.

This matrix establishes deterministic single-interval contracts. It does not
establish recovery under sustained, repeated, or interacting faults.

This component is not a flight-safety system. It does not implement estimator
health beyond timestamp and value validity, motor-failure allocation, arming,
radio/firmware transport, ground handling, battery compensation, landing, or a
certified safe mode. Those remain outside the library boundary.
