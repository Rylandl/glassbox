# Nonlinear model-predictive control

Glassbox NMPC turns one belief into a finite-horizon rigid-body tracker. The
interface is intentionally small: a belief, a state estimate, a state
reference, the previous command, optional applied control or latent actuator
state, physical tracking tolerances, and optional state limits. Horizon
length, command blocking, line search, regularization and iteration count are
maintained policies rather than routine user knobs.

The layer is three modules with one seam between them. `control.plan` declares
`PlanModel`, the whole interface a solver has to a model: command bounds,
uncertainty completeness, a horizon, the `PlanValues` its numbers travel in,
a rollout that returns predicted states with their tangent covariance, and a
stage cost.
`control.solver` is `BoundedShootingSolver`, which knows nothing about
beliefs; it moves normalized command blocks inside their box and returns an
auditable result. `control.fitted` is the boundary between them:
`plan_model(belief, tolerances, envelope)` presents a belief as a `PlanModel`,
settles the horizon against the belief's own error evidence, and raises
`NonActionableModelError` for a model with no command space. `NMPCController`
is the thin factory that wires the two together, and it is what most callers
use.

Because the seam is the protocol and not the belief, one solver serves a
belief fitted from a corpus and a belief built in flight from nothing. See
[bootstrap identification](bootstrap-identification.md) for the second case.

Compiling a solver costs orders of magnitude more than solving with it, so the
compiled kernels are cached at module scope under the plan model's static
signature: the input and runtime specs, the tolerances, the envelope, the
policy, the actuator map and its command bounds, the parameter tree's
structure and leaf shapes, and the shape of what the belief resolved. No belief
value is in that signature. The parameters, the
factor of the resolved parameter covariance and the stage forecast-error
covariance travel to every kernel together as `PlanValues`, so two controllers
built from the same configuration share compiled code, and so does a belief
that absorbs telemetry every control interval: only a change of resolved rank
compiles again. Direct maps with equal channels share kernels. Arbitrary
maps share only when they are the same immutable instance; construct a new
map when calibration changes. A final command-bound check also guards against
incorrect output from a custom plan model.

The parameter covariance factor uses the information's resolved directions
directly. It applies no second relative cutoff to their variances: a small
parameter variance can still have a large effect on a sensitive prediction.

The controller is independent of reference generation, state estimation, PX4
transport and hardware mixing. Terminal-pose docking is not part of this
module.

## Minimal use

```python
import jax.numpy as jnp

from glassbox import (
    DynamicsBelief,
    NMPCController,
    SafetyEnvelope,
    TrackingTolerances,
)
from glassbox.core.dynamics import hover_control

belief = DynamicsBelief.load("artifacts/belief.json")
controller = NMPCController(
    belief,
    TrackingTolerances.for_platform(belief.input_spec.vehicle.family),
    SafetyEnvelope(
        minimum_position_m=(-100.0, -100.0, -20.0),
        maximum_position_m=(100.0, 100.0, 100.0),
    ),
)

# NWU position and velocity, WXYZ quaternion, FLU body rates.
state = jnp.asarray([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
result = controller.solve(
    state, controller.hold_reference(state), hover_control(belief.params)
)
result = controller.solve(
    state,
    controller.hold_reference(state),
    result.command,
    warm_start=result.warm_start,
)
command = result.command  # bounded even when result.command_usable is False
```

If the belief does not resolve every estimable parameter direction, a solve
returns `UNRESOLVED_MODEL` and a bounded hold. A caller deliberately accepting
partial parameter uncertainty can opt in:

```python
from dataclasses import replace
from glassbox.control.fitted import default_solver_policy

controller = NMPCController(
    belief,
    policy=replace(default_solver_policy(belief), allow_unresolved_parameters=True),
)
```

This override permits a cost based on the mean and supported covariance; it
supplies no uncertainty bound for unresolved directions. Diagnostics report
`parameter_uncertainty_complete=False`, `unresolved_parameters_allowed=True`,
and an infinite uncertainty margin (JSON `null`). The simulation benchmarks
make this choice explicitly. PX4 shadow exposes the same choice through
`--allow-unresolved-parameters` and retains the loaded belief's information.

A `ReferenceTrajectory`'s `states` must have `controller.prediction_steps + 1`
rows and 13 columns; `controller.hold_reference(state)` is the short path for
regulation. An exogenous forecast has one row per prediction interval in the
exact typed order the artifact records.

`applied_command` is an optional measured actuator value expressed in the
controller's actionable command coordinates. Advanced estimators can instead
provide the complete `latent_state`; passing both is rejected. If neither is
available, the controller initializes lag state from `previous_command`.

The first eligible cold solve compiles the solve path and the first warm-started solve
compiles the receding-horizon path, which is why the snippet above solves
twice. Run both and discard their commands before entering a timed control
loop, and pass `deadline_s` only afterwards. Compilation must never happen
after arming. A request rejected as `UNRESOLVED_MODEL` does not compile either
path; prewarm again after choosing an explicit override.

## Eligible models and airframes

The runtime contract requires a sample period, a training-derived
body-velocity and angular-rate envelope, complete control roles and bounds,
and latent actuator state semantics. Loading the artifact never invents
missing runtime facts. Nothing in the contract certifies a prediction horizon:
what shortens the horizon is the belief's own held-out forecast-error
envelope, and a model with no envelope plans the maintained default for its
vehicle.

Direct control is allowed only for `normalized_command` and
`normalized_generalized_command` channels. Measured RPM, squared rotor speed,
surface angle and normalized actuator output are observations, not commands.
Those artifacts fail closed unless the integrator supplies an explicit typed,
JAX-compatible `ActuationMap` from bounded commands to model inputs.

The solver itself has no conventional-airframe or motor-layout branches. It
uses ordered typed channels and roles. Current tests cover four-motor
multirotors; conventional throttle, roll, pitch and yaw fixed wings;
three-channel flying wings with generalized elevon roll and pitch commands;
optional flap authority; and both structured and structured-residual dynamics.

Learned parameters remain airframe specific. Hardware mixing stays downstream:
a flying-wing adapter, for example, converts the returned generalized roll and
pitch commands into left and right elevon commands.

## Objective, constraints and outcomes

State error has 12 local coordinates: position, velocity, shortest quaternion
log-map attitude error, and angular velocity. Quaternion signs are equivalent;
components are never subtracted as a tracking metric. Errors are divided by
physical tolerances before aggregation.

Command limits are hard: every direct-shooting iterate is projected into the
typed channel bounds. Command change, full-horizon model-validity excess and
`SafetyEnvelope` state limits remain dimensionless soft penalties.
`SafetyEnvelope` supports minimum and maximum world position, maximum world
speed and maximum body angular speed. These mission limits are preferences,
not invariant-set or collision guarantees.

### The two robustness terms

Nothing edits the command after optimization. What the belief knows about its
own error is charged inside the objective, in two places, with no
configuration of its own.

The tracking cost approximates an expectation using local tangent errors. At every
predicted stage it charges `l(mean) + trace(W Sigma)`, where `W` is the
diagonal tracking weight the objective already builds from the declared
tolerances and `Sigma` is the predicted tangent covariance: the belief's
forecast-error covariance at that horizon plus the parameter covariance
carried through the plan. A plan that drives the vehicle into a region the
belief forecasts poorly therefore costs more than the same tracking error in a
region it forecasts well. The parameter term is a first-order propagation,
`J C J.T`, rather than the exact nonlinear predictive covariance. A complete
information matrix does not establish that this approximation is accurate:
weakly observed directions can have very large spread. Check nonlinear
perturbations and independent prediction evidence before promoting an adapted
belief for control; the [recovery investigation](../recovery-investigation.md)
shows a case where rank completeness alone is misleading.

The model-validity term charges the robust utilization instead of the mean
utilization. The tangent covariance is mapped onto the six envelope features,
body velocity and body rates, and each feature's marginal standard deviation
is added to its mean utilization before the excess over one is squared.

Both covariance terms vanish for a mean without evidence. Such a cost is
available only through the explicit unresolved-parameter override; it does
not turn missing evidence into zero risk. The
parameter contribution is written through a factor of the covariance, so a
belief that resolves two directions costs two extra forward rollouts rather
than a full Jacobian. Neither term is a calibration claim, an invariant-set
proof or a hard constraint: the full prediction can still leave support, and
the result records how far it did.

### The result

`SolveResult` carries the status, the bounded command, the predicted state,
latent and command traces, an opaque `warm_start` for the next
receding-horizon solve, `used_fallback`, an optional message, and
`NMPCDiagnostics` fields: iterations, solve time, initial and final objective,
the bound-projected gradient infinity norm, maximum command-bound violation,
maximum validity utilization, normalized safety violation, normalized model
uncertainty standard deviation, whether a warm start was used, and the
prediction horizon the plan covers. Two flags record whether parameter
uncertainty is complete and whether the caller permits unresolved parameters.
`command_usable` says whether the command came from a finite optimized plan
within command bounds and permitted parameter support. It does not establish
that the predicted motion stays inside the model's validity envelope.
`maximum_validity_utilization` measures the mean trajectory, including the
initial state. A value above one means some declared feature bound is exceeded;
the objective's covariance-expanded validity penalty is a separate quantity.

`converged` is reserved for the first-order criterion, and that criterion
tests the bound-projected gradient, `blocks - clip(blocks - gradient)`,
against the maintained tolerance rather than the raw gradient, because a raw
gradient component pointing outward at an active command bound never shrinks
however optimal the iterate is. The projected residual is reported as
`final_projected_gradient_inf_norm`, so the status can be audited from the
result.

Two outcomes report a finite bounded best plan without claiming convergence.
`iteration_limit` exhausted the maintained iteration budget. `stalled` stopped
earlier because the bounded line search ran out of progress: either relative
improvement fell below the maintained tolerance, or no acceptable step
remained after at least one accepted iteration. Both are finite optimized
plans with `used_fallback=False` and `command_usable=True`, and neither is
labeled converged.

`invalid_input`, `unresolved_model`, `command_bound_violation`,
`nonfinite_objective`, `line_search_failed` before any
iteration is accepted, and `deadline_exceeded` set `used_fallback=True` and
`command_usable=False`. The returned value is only an explicit bounded hold of
the previous command, or the channel midpoint if the previous command is
invalid; Glassbox does not silently replace NMPC with a second controller. A
deadline cannot preempt an already executing device call; the elapsed-time
check rejects its output afterwards.

## Measured capability

The recorded evidence is
[`nmpc-acceptance-results.json`](../results/nmpc-acceptance-results.json),
regenerated by `glassbox record-results --only nmpc-acceptance-results`, which
is one step:

```bash
uv run glassbox benchmark nmpc \
  --output docs/results/nmpc-acceptance-results.json
```

What that run measured is on
[validation](../validation.md#nmpc-acceptance), which is the one page that
quotes the numbers: the equal-scenario tracking ratios against the
non-optimizing trim baseline, nominally and under parameter mismatch, the nine
checks in `summary.checks`, and the per-scenario timing distribution. They are
not restated here, so a re-record moves one page rather than two. In outline,
the optimizing controller tracked better than the baseline in both conditions
and by a wider margin under mismatch, every scenario was finite with no
fallback and no command-bound violation, and no individual scenario hid inside
an aggregate. Absolute solve times depend on the host and its load, so they
live in the artifact and not in prose; passing the functional gates is not a
real-time claim and not a flight-safety claim.

The acceptance thresholds themselves are recorded in the artifact's
`thresholds` block and are part of the contract rather than of a run: every
solve finite with no hard command bound violated beyond `1e-6` in normalized
command coordinates; analytic gradients agreeing with central differences to
`2e-3` relative error on structured multirotor, structured fixed-wing and
structured-residual fixtures; a nominal aggregate at most `0.80` of the
declared baseline with no individual nominal scenario above `1.05`; mismatch
scenarios finite, inside the validity guard, below the same baseline in
aggregate and with no individual ratio above `1.10`; warm starting never worse
than the cold-start policy at the same state; and every failure path returning
an explicit status with a finite bounded fallback. Thresholds may only change
in a reviewed contract revision made before the candidate being judged is
tuned.

## One control interval

`glassbox.integrations.loop` declares the interval every integration shares.
`VehicleLink` is a vehicle a loop can read an `Observation` from and, if
`writable`, hand a bounded command to; a read-only link raises from `write`
rather than silently accepting a command it will never transmit. `writable` is
the whole difference between shadow mode and closed-loop control, and it is a
property of the link rather than a flag on the loop.

```text
obs = link.read(timeout_s=period)
result = solver.solve(obs.state, reference.at(obs.t), obs.applied_command,
                      warm_start=warm, deadline_s=period)
decision = supervisor.supervise(...)          # when a supervisor is given
if link.writable: link.write(decision.command)
```

`run_control_loop(link, controller, supervisor, *, steps, reference)` is that
sequence. It reads the link, solves from the previous interval's warm start
with the interval as the deadline, supervises the candidate when a supervisor
is given, writes it when the link accepts writes, and records one `LoopSample`
per interval carrying the observation, the solve result, the command, whether
it was written and the supervisor's decision. A failed solve never ends a run:
the solver's bounded hold is what the loop records and passes on. The closing
`LoopSummary` counts statuses, usable commands, fallbacks, written commands,
supervisor interventions and deadline misses, and reports the solve-time
median, p90 and maximum alongside the worst message skew, receive age,
source-clock lag and state-to-command skew the run saw.

The loop passes `result.diagnostics.maximum_validity_utilization` to the
supervisor as `controller_maximum_validity_utilization`. Custom implementations
of `CommandSupervisor` must accept this keyword. An optional `clock` callable
supplies supervision timestamps for discrete simulations; observation reception
timestamps must use that same clock. Its default is `time.monotonic`. Solver
CPU durations and `host_elapsed_s` still use the actual host clock.

PX4 telemetry is a read-only link and the Cascade plant is a writable one, so
shadow mode and simulated closed-loop control are the same code differing by
one property of the link.

## PX4 shadow mode

PX4 is an outer, opt-in contract test rather than a runtime dependency. The
package talks only to PX4's standard MAVLink telemetry. It does not import PX4
code, vendor a simulator, add Gazebo or ROS, or launch Docker from production
code.

The live boundary in `glassbox.integrations.px4` passively receives
`LOCAL_POSITION_NED` and `ATTITUDE_QUATERNION`. It verifies the PX4 heartbeat
and source system, pairs fresh messages with bounded boot-time skew,
normalizes and makes quaternion signs continuous, and returns the canonical
13-state NWU/FLU/WXYZ representation. Its frame operations are the same
functions offline ULog ingestion uses. The source exposes no send method and
never requests stream rates, arms, changes mode or transmits a setpoint. The
MAVLink reader runs continuously on a daemon thread and retains only what it
has latched: the latest coherent state, and a bounded recent history for the
actuator stream. That is required even in shadow mode, because a solver can
block long enough for a receive buffer to preserve old datagrams while
dropping newer ones. Each state also reports estimated source-clock lag
relative to the best observed boot-time and host-time alignment.

For a running PX4 instance the operator-facing command is one
`run_control_loop` over a read-only `PX4MavlinkLink`:

```bash
uv run glassbox px4-shadow artifacts/px4/model.json \
  --previous-command 0.5,0.5,0.5,0.5 \
  --output artifacts/px4/nmpc-shadow.jsonl
```

It holds the current state as the regulation reference, applies the artifact's
sample period as the solver deadline, and writes one JSON object per interval.
The passive telemetry wait defaults to one second and can be changed with
`--telemetry-timeout-s`; it neither extends the solve deadline nor resets
the received state's age. The general control loop retains its sample-period
read timeout unless `read_timeout_s` is explicitly supplied. Nothing is transmitted: the link is not
writable, so the loop never calls its writer.

Ordinary `pytest` runs deterministic tests with fake MAVLink messages and
skips the external fixture. To exercise a real PX4 binary and real MAVLink
encoding:

```bash
GLASSBOX_RUN_PX4_SITL=1 \
  uv run pytest -m px4_sitl tests/integration/test_px4_sitl.py -v
```

That test launches the multi-architecture PX4 SIH image pinned by immutable
digest, gives it an explicit Docker host gateway, receives canonical
telemetry, and always stops the container. Setting `GLASSBOX_PX4_NMPC_MODEL`
adds the complete telemetry-to-solver path against a promoted actionable
artifact, and `GLASSBOX_RUN_PX4_FLIGHT_SHADOW=1` additionally flies a bounded
profile matrix so the applied commands come from PX4 rather than from a
constant. The two opt-in modes are deliberately separate: when the flown flag
is set the fixed-command test is skipped, so solver warm-up or PX4 state from
one lifecycle cannot change the evidence produced by the other.

This establishes transport and solver integration, not closed-loop PX4
control. A transport test can pass while a real-time budget fails, so the
summary keeps telemetry skew and solver latency as separate measurements. Any
`deadline_exceeded` sample returns the bounded previous command and is counted
as a fallback. A future command-output adapter remains a separate boundary and
must own mixing, arming and mode checks, stale-setpoint rejection, command
timestamps, an independent watchdog and safe-mode handoff. The repository
provides no such PX4 adapter.

## The flight supervisor

`MultirotorFlightSupervisor` is the bounded command's last check: a thin,
stateful authority boundary around a four-motor controller. It does not
consume a belief, fit a model, run an optimizer or track position. Its job is
to decide whether a candidate motor command may reach the plant, which is why
it sits outside `NMPCController` and must be configured by the vehicle
integration.

For fresh valid telemetry and a fresh, finite, bounded, command-usable
candidate whose reported plan remains inside model support, the supervisor
returns the candidate unchanged. Invalid or stale
telemetry selects a configured collective hold, because geometric arrest
cannot be trusted without a usable attitude and body rate. Command faults, an
unusable controller, unknown or exceeded model support, or excessive tilt or body rates select a bounded
geometric attitude and rate-arrest command. Arrest remains latched for a
minimum interval and until tighter release limits are met.

The support input must be finite, nonnegative and at most `1 + 1e-6`.
Omitting it withholds nominal control with `MODEL_SUPPORT_UNKNOWN`; an
exceeded envelope produces `MODEL_SUPPORT_EXCEEDED`. The supervisor consumes
the controller's reported support rather than evaluating a model itself.
Its physical tilt and rate limits remain independent. A supported nominal
forecast does not prove that arrest commands or subsequent physical states
remain inside that envelope. The [supervised recovery investigation](../recovery-investigation.md#supervised-recovery-and-model-support)
records this distinction and the tested fault responses.

The supervisor does not know how this airframe turns a desired body-axis
differential into motor commands, and it must not assume one: an identifier
that has not resolved the canonical mixer says so in its own report, and a
supervisor that assumed it anyway would command the wrong motors during the
one interval it exists for. That knowledge is injected by whoever holds it.
`allocate` maps the desired roll, pitch and yaw differential to the four motor
increments the arrest adds to the configured collective hold, and
`has_allocation` reports whether one was given. Constructed without it, the
supervisor still runs every freshness rule, every limit and the latch, but its
arrest is the collective hold alone, rate-limited toward the previously
applied command and reported as `SupervisorMode.COLLECTIVE_HOLD` carrying
`SupervisorReason.NO_ALLOCATION`, so a refusal to guess is visible in the
audit rather than mistaken for an attitude arrest.

The arrest command drives the geodesic attitude error, the rotation vector
that would level the vehicle. Its magnitude is the tilt angle itself, so
restoring authority holds across the whole `[0, pi]` range instead of fading
out near inversion the way a `sin(tilt)` cross product does. For small tilts
the two agree to third order, so nothing changes in normal flight; at exact
inversion the axis is undefined and a fixed positive roll direction is chosen.
Candidate commands within a rounding width of the command bounds, `1e-6` of
the span, are clipped rather than treated as out of bounds, so a
representation-width overshoot cannot latch an arrest. The latch always starts
from the newest time the supervisor has seen, never from a regressed clock
reading, so a clock that jumps backwards cannot let the next valid tick skip
the minimum arrest duration.

```python
import time

import numpy as np

from glassbox import MultirotorFlightSupervisor, MultirotorSupervisorConfig
from glassbox.core.dynamics import MOTOR_MIXER

supervisor = MultirotorFlightSupervisor(
    MultirotorSupervisorConfig(
        collective_hold_command=(0.53, 0.53, 0.53, 0.53),
        maximum_state_age_s=0.04,
        maximum_command_age_s=0.02,
    ),
    allocate=lambda differential: 0.25 * np.asarray(MOTOR_MIXER).T @ differential,
)

# `state` and `result` are the ones the minimal-use snippet above produced.
now = time.perf_counter()
decision = supervisor.supervise(
    state=np.asarray(state),
    state_received_at_s=now,
    candidate_command=np.asarray(result.command),
    command_generated_at_s=now,
    now_s=now,
    controller_command_usable=result.command_usable,
    controller_maximum_validity_utilization=result.diagnostics.maximum_validity_utilization,
    previous_applied_command=np.asarray(result.command),
)
motor_command = decision.command
```

The caller must use one monotonic clock for all timestamps. Motor order and
normalized command semantics must match the allocation that was handed in. The
collective hold command, command bounds, limits, gains and maximum arrest slew
are vehicle integration values, not identified dynamics parameters. Every
decision reports a `SupervisorMode`, typed `SupervisorReason` values, state
and command ages, measured tilt and rate, `maximum_model_validity_utilization`
(`None` when unknown), whether the nominal command was
accepted, and an immutable four-motor command; `reset()` clears the time and
arrest latch for an explicit lifecycle restart.

This component is not a flight-safety system. It does not implement estimator
health beyond timestamp and value validity, motor-failure allocation, arming,
radio or firmware transport, ground handling, battery compensation, landing,
or a certified safe mode. Those remain outside the library boundary.
