# Glassbox nonlinear model-predictive control

Glassbox NMPC turns one eligible fitted dynamics belief into a finite-horizon
rigid-body tracker. The interface is intentionally small: a belief, a state
estimate, a state reference, the previous command, optional applied control or
latent actuator state, physical tracking tolerances, and optional state limits.
Horizon length, command blocking, line search, regularization, and iteration
count are maintained policies rather than routine user knobs. When a belief
supplies predictive-error evidence, the maintained horizon is capped at that
evidence boundary.

The layer is three modules with one seam between them. `control.plan` declares
`PlanModel`, the whole interface a solver has to a model: command bounds, a
horizon, the `PlanValues` its numbers travel in, a rollout that returns
predicted states with their tangent covariance, and a stage cost. `control.solver` is `BoundedShootingSolver`, which knows
nothing about beliefs; it moves normalized command blocks inside their box and
returns an auditable result. `control.fitted` is the boundary between them:
`plan_model(belief, tolerances, envelope)` presents a fitted belief as a
`PlanModel`, settles the horizon against the belief's own error evidence, and
raises `NonActionableModelError` for a model with no command space.
`NMPCController` is the thin factory that wires the two together, and it is
what most callers use.

Compiling a solver costs orders of magnitude more than solving with it, so the
compiled kernels are cached at module scope under the plan model's static
signature: the input and runtime specs, the tolerances, the envelope, the
policy, the parameter tree's structure and leaf shapes, and the shape of what
the belief resolved. No belief value is in that signature. The parameters, the
factor of the resolved parameter covariance and the stage forecast-error
covariance travel to every kernel together as `PlanValues`, so two controllers
built from the same configuration share compiled code, and so does a belief
that absorbs telemetry every control interval: only a change of resolved rank
compiles again.

The controller is independent of reference generation, state estimation, PX4
transport, and hardware mixing. Terminal-pose docking is not part of this
module.

## Minimal use

```python
import jax.numpy as jnp

from glassbox import (
    DynamicsBelief,
    NMPCController,
    ReferenceTrajectory,
    SafetyEnvelope,
    TrackingTolerances,
)

belief = DynamicsBelief.load("artifacts/vehicle-belief.json")
controller = NMPCController(
    belief,
    TrackingTolerances.for_platform(belief.input_spec.vehicle.family),
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

The first cold solve compiles the solve path; the first warm-started solve
compiles the receding-horizon path. Run both and discard their commands before
entering a timed control loop. Compilation must never happen after arming.

## Eligible models and airframes

The runtime contract requires a sample period, a training-derived body-velocity
and angular-rate envelope, complete control roles and bounds, and latent
actuator state semantics. Loading the artifact never invents missing runtime
facts. Nothing in the contract certifies a prediction horizon: what shortens
the horizon is the belief's own held-out forecast-error envelope, and a model
with no envelope plans the maintained default for its vehicle.

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
typed channel bounds.

Command change, full-horizon model-validity excess, and `SafetyEnvelope` state
limits remain dimensionless soft penalties. `SafetyEnvelope` currently supports
minimum/maximum world position, maximum world speed, and maximum body angular
speed. These mission limits are preferences, not invariant-set or collision
guarantees.

### The two robustness terms

Nothing edits the command after optimization. What the belief knows about its
own error is charged inside the objective, in two places, with no configuration
of its own.

The tracking cost is an expectation rather than a point evaluation. At every
predicted stage it charges `l(mean) + trace(W Sigma)`, where `W` is the
diagonal tracking weight the objective already builds from the declared
tolerances and `Sigma` is the predicted tangent covariance: the belief's
forecast-error covariance at that horizon plus, where the evidence scopes it
separately, the parameter covariance carried through the plan. A plan that
drives the vehicle into a region the belief forecasts poorly therefore costs
more than the same tracking error in a region it forecasts well.

The model-validity term charges the robust utilization instead of the mean
utilization. The tangent covariance is mapped onto the six envelope features,
body velocity and body rates, and each feature's marginal standard deviation is
added to its mean utilization before the excess over one is squared.

Both terms are exactly zero for a belief that carries no covariance, so a point
model is scored by the point objective it was always scored by. Neither is a
calibration claim, an invariant-set proof, or a hard constraint: the full
prediction can still leave support, and the result records how far it did.

Important result fields are:

- `status`: converged, stalled, finite iteration-limit plan, or an explicit
  failure;
- `command_usable`: whether the command comes from a finite optimized plan;
- `predicted_states`, `predicted_latent_states`, and `predicted_commands`;
- initial and final objective, iteration count, the bound-projected gradient
  infinity norm, and solve time;
- maximum command-bound violation, model-validity utilization, normalized
  safety-limit violation, and normalized model-uncertainty standard deviation;
- the prediction horizon the plan covers; and
- an opaque `warm_start` for the next receding-horizon solve.

`converged` is reserved for the first-order criterion. That criterion tests the
bound-projected gradient, `blocks - clip(blocks - gradient)`, against the
maintained tolerance rather than the raw gradient, because a raw gradient
component pointing outward at an active command bound never shrinks however
optimal the iterate is. The projected residual is reported as
`final_projected_gradient_inf_norm`, so the status can be audited from the
result.

Two outcomes report a finite bounded best plan without claiming convergence.
An iteration-limit result exhausted the maintained iteration budget. A
`stalled` result stopped earlier because the bounded line search ran out of
progress: either relative improvement fell below the maintained tolerance, or
no acceptable step remained after at least one accepted iteration. Both are
finite optimized plans with `used_fallback=False` and `command_usable=True`;
`stalled` carries exactly the same usability as an iteration-limit plan and is
not a fallback. Neither is labeled converged.

Invalid estimates, non-finite objectives, a line search that fails before any
iteration is accepted, and exceeded deadlines set `used_fallback=True` and
`command_usable=False`. The returned value is only an explicit bounded hold of
the previous command, or the channel midpoint if the previous command is
invalid; Glassbox does not silently replace NMPC with a second controller. A
deadline cannot preempt an already executing JAX device call; the elapsed-time
check rejects its output afterward. Process isolation and authority handoff
remain integration concerns, not alternate controllers embedded here. Glassbox
does provide a separate `MultirotorFlightSupervisor` building block for command
and telemetry freshness, finite/bounded command checks, and attitude/rate
arrest; it is intentionally outside `NMPCController` and must be configured by
the vehicle integration. See the
[multirotor supervisor contract](flight-supervisor.md).

## Measured capability

The maintained gate and its fixed pre-tuning thresholds are defined in
[the design and acceptance contract](#design-and-acceptance-contract). The latest machine-readable run is
[`nmpc-acceptance-results.json`](../results/nmpc-acceptance-results.json).

On the recorded Apple M3 CPU run with JAX's CPU backend and a 50 ms model step:

- all eight nominal and eight model-mismatch cases were finite;
- there were no fallbacks and no command-bound violations;
- every mismatch case stayed within the fitted model-validity envelope;
- equal-scenario geometric tracking error was `0.651x` the non-optimizing trim
  baseline nominally and `0.552x` under parameter mismatch; and
- every scenario's post-JIT median solve time was well below the 50 ms model
  step.

Each scenario records median, p90, and maximum post-JIT time. Cold compilation
took multiple seconds for each novel model/control shape. On the recorded run
even the maximum observed solve stayed under half the model step.
Absolute times depend on the host and its load, so they live only in the
results artifact; they establish margin for that benchmark and hardware
combination, not a portable hard real-time guarantee. No flight-safety claim
is made.

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

One control interval is the same everywhere. `glassbox.integrations.loop`
declares `VehicleLink`, which is a vehicle a loop can read an `Observation`
from and, if it is writable, hand a bounded command to; `run_control_loop`
reads the link, solves from the previous interval's warm start with the
interval as the deadline, supervises the candidate when a supervisor is given,
writes it when the link accepts writes, and records what happened. PX4
telemetry is a read-only link and the Cascade plant is a writable one, so
shadow mode and simulated closed-loop control are the same code differing by
one property of the link. A failed solve never ends a run: the solver's bounded
hold is what the loop records and passes on.

The live boundary in `glassbox.integrations.px4` passively receives
`LOCAL_POSITION_NED` and `ATTITUDE_QUATERNION`. It verifies the PX4 heartbeat and
source system, pairs fresh messages with bounded boot-time skew, normalizes and
makes quaternion signs continuous, and returns the canonical 13-state
NWU/FLU/WXYZ representation. Its frame operations are the same functions used
by offline ULog ingestion. The source exposes no send method and never requests
stream rates, arms, changes mode, or transmits a setpoint.

The MAVLink reader runs continuously on a daemon thread and retains only what
it has latched: the latest coherent state here, and a bounded recent history
for the actuator stream. One receiver serves both, differing only in how each
message is decoded and how a reader selects from the history. This is required even in shadow mode: a solver can block
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
uv run glassbox px4-shadow artifacts/px4/model.json \
  --previous-command 0.5,0.5,0.5,0.5 \
  --output artifacts/px4/nmpc-shadow.jsonl
```

The leaf is one `run_control_loop` over a read-only `PX4MavlinkLink`. It holds
the current state as the regulation reference, applies the artifact's sample
period as both the telemetry timeout and the solver deadline, and writes one
JSON object per interval carrying the state that was read, the command PX4 was
applying, the command the solver returned, its status, its solve time, and the
plan's diagnostics. The closing summary counts statuses, usable commands,
fallbacks and deadline misses, and reports solve-time median, p90 and maximum
alongside the worst message skew, receive age, source-clock lag and
state-to-command skew the run saw. Nothing is transmitted: the link is not
writable, so the loop never calls its writer. The previous command remains the
measured applied command because the shadow command is not being actuated.

The maintained fixture can also exercise a dynamically flown profile matrix
using the commands PX4 actually applies rather than a constant supplied by the
operator:

```bash
GLASSBOX_RUN_PX4_SITL=1 \
GLASSBOX_RUN_PX4_FLIGHT_SHADOW=1 \
GLASSBOX_PX4_NMPC_MODEL=artifacts/px4/model.json \
  uv run pytest -m px4_sitl tests/integration/test_px4_sitl.py -k flown -v
```

This higher opt-in level arms only fresh disposable `sihsim_quadx` containers.
Every profile gets an independent PX4 lifecycle, so a cleanup failure cannot
contaminate the following case. Each waits for PX4's normal readiness margin,
invokes the ordinary takeoff command, then runs one bounded OFFBOARD vertical,
lateral, yaw, or combined profile and verifies its intended state excitation,
landing, and disarm. The state source remains passive on PX4's onboard MAVLink
link. A second passive source consumes `HIL_ACTUATOR_CONTROLS` from the simulator
link and maps the pinned quad-X output geometry into the artifact's canonical
motor order. It retains a bounded recent history and selects the command nearest
each state's PX4 boot timestamp; solver load therefore cannot turn two
individually fresh streams into a mismatched pair. The profile driver is the
only process that transmits anything.

Every measured applied command is checked against the artifact's dimensions and
bounds before solving. State and command source timestamps must be within one
model sample period on PX4's boot clock, with a 100 ms absolute ceiling;
otherwise evaluation stops instead of silently pairing unrelated samples. Each
interval record carries its own state-to-command skew, receive age and armed
state, and the per-channel excitation is read off the recorded applied
commands. This makes the fixture evidence for asynchronous telemetry, varying
commands, moving states, solver deadlines, and cleanup, not a closed-loop
control test and not a general claim that HIL actuator order is shared by other
PX4 configurations.

The fixture gates what the loop measures: every interval produces a record,
every applied command lies inside the artifact's dimensions and bounds, state
and command stay inside the alignment limit, the commands actually vary, the
profile excites the states it claims to, and the container is always cleaned
up. Model promotion remains the job of held-out recorded-flight benchmarks with
airframe-relevant horizons; a live transport fixture is not evidence about
prediction quality, and no longer pretends to be by scoring one-step
predictions against a persistence baseline in the same document.

A transport test can pass while the real-time gate fails. In particular, PX4
SIH and JAX share host resources in this fixture, so the summary keeps
telemetry skew and solver latency as separate measurements rather than folding
them into one number. Any `deadline_exceeded` sample returns the bounded
previous command and is counted as a fallback; it is not presented as a usable
controller output.

The fixed-command shadow and flown-telemetry matrix are deliberately separate
opt-in modes. When the flown flag is set, the fixed-command test is skipped even
if its command variable is also present. This prevents solver warm-up or PX4
state from one lifecycle changing the evidence produced by the other.

This establishes transport and solver integration, not closed-loop PX4 control.
A future command-output adapter remains a separate boundary and must own mixing,
arming and mode checks, stale-setpoint rejection, command timestamps, an
independent watchdog, and safe-mode handoff. The repository still provides no
such PX4 adapter. The model-independent multirotor supervisor can form one part
of that boundary, but it does not implement PX4 transport, arming, estimator
health, or vehicle-specific safe-mode handoff and must not be connected directly
to real actuators.

## Reproducing the gate

```bash
uv run glassbox benchmark nmpc \
  --output artifacts/nmpc-acceptance-results.json
```

The report names the runtime environment, baseline, normalized error, fixed
thresholds, every scenario result, and timing distribution. The same function
is exercised by the test suite.

## Design and acceptance contract

### Scope

The NMPC layer consumes the same canonical rigid-body and control semantics as
the fitted dynamics model. It provides finite-horizon tracking and regulation;
terminal-pose docking, state estimation, reference generation, and hardware
mixing remain separate projects or adapters.

The dependency direction is deliberately one-way:

```text
dynamics belief -> executable model view -> NMPC -> canonical command
                                               |              |
                         state/reference/context       actuation adapter
```

Runtime and NMPC modules may depend on the canonical data, dynamics, and model
serialization layers. They must not import fitting CLIs, benchmark workflows,
policy selection, or research-only experiments.

### Runtime contract

A control-eligible artifact must carry:

- the typed prediction `TrajectorySpec`;
- its fixed integration/sample period;
- the training-derived body-velocity and angular-rate validity envelope;
- the complete latent applied-control state layout; and
- an actionable command mapping with finite command bounds.

`normalized_command` and `normalized_generalized_command` are directly
actionable. Measured rotor speed, generalized surface angle, and normalized
actuator output are observations of actuation, not commands. Such artifacts
require an explicit typed actuation map and otherwise fail closed.

### Initial solver policy

The first maintained solver is warm-started direct shooting. It optimizes a
bounded sequence of command blocks, expands them over fixed model integration
steps, and differentiates the complete rollout with JAX. The normal interface
exposes physical tracking tolerances and a safety envelope rather than raw
state/control weight matrices. Solver iteration limits, line-search policy,
regularization, and control-block policy are maintainer-owned defaults.
The maintained backend uses at most eight warm-started outer iterations per
control update; accuracy and latency changes to that policy are judged against
the complete acceptance suite, not exposed as operator tuning knobs.
Its bounded Armijo search carries an accepted step size into the next outer
iteration and cautiously expands it, avoiding repeated backtracking from the
same deliberately conservative maximum step.

The command-block layout is the largest divisor of the horizon that is at most
ten blocks, so every block is held for the same number of model steps and the
expansion covers the horizon exactly. A horizon of ten steps or fewer gets one
block per step. Only a prime horizon longer than that cap needs a shortened
final block. No layout ever carries a block that drives no prediction step,
which would otherwise cost gradient, line-search, and warm-start work on
coordinates the plan cannot use.

The returned `warm_start` is shifted at that same block granularity for the
next solve: its first block is the previous plan's second block, and its final
block repeats the previous plan's last block. Shifting the expanded command
sequence by a single model step instead would land back inside the same old
block whenever a block spans more than one step, which reproduces the previous
plan unshifted. The shifted seed is used only when its objective is no worse
than the controller's cold-start policy.

For a dynamics belief, the default prediction horizon cannot exceed maintained
predictive-error evidence. Predicted spread is charged inside the objective, as
described under [the two robustness terms](#the-two-robustness-terms), so it
shapes the plan the optimizer converges to rather than editing the plan
afterwards. The parameter contribution is carried through one forward-mode
rollout per resolved parameter direction, which is why a belief whose evidence
resolved a single direction costs one extra rollout rather than a full Jacobian.
Glassbox deliberately does not embed a second airframe-specific controller
behind this NMPC path.

Rigid-body error has 12 local coordinates: position, velocity, the shortest
quaternion log-map rotation vector, and angular velocity. Quaternion component
subtraction is never a tracking metric. Command bounds are hard constraints;
command rate and model-validity terms use dimensionless physical
normalization. Because those bounds are hard, the first-order convergence test
uses the bound-projected gradient; a converged status therefore means no
feasible descent direction remains, not merely that the raw gradient is small.
A solve returns the predicted state/latent/control traces, initial and final
costs, the outcome status separating convergence from an iteration limit and
from a line-search stall, timing, constraint diagnostics, and the explicit
bounded hold returned on failure. That hold is marked unusable and is not
presented as an independently functioning controller.

### Acceptance thresholds fixed before tuning

The following gates apply to the maintained synthetic scenario suite:

1. Every nominal solve and closed-loop sample is finite. No hard command bound
   may be violated beyond `1e-6` in normalized command coordinates.
2. Analytic objective gradients must agree with central finite differences to
   `2e-3` relative error on structured multirotor, structured fixed-wing, and
   structured-residual fixtures.
3. The equal-scenario geometric mean of normalized tracking RMS must be at
   most `0.80` of the declared non-optimizing baseline. No individual nominal
   scenario may exceed `1.05` times its baseline error.
4. Model-mismatch scenarios must remain finite and within the validity guard.
   Their aggregate normalized tracking RMS must be below the same baseline;
   no aggregate gain may hide an individual ratio above `1.10`.
5. Warm starting must not worsen the initial objective compared with the
   controller's cold-start policy for the same receding-horizon state.
6. Forced non-convergence, non-finite objectives, invalid estimates, and
   deadline failures must return an explicit failure status and a finite,
   bounded fallback command.
7. Post-JIT solve latency is reported as median, p90, and maximum on named
   hardware. Passing functional gates does not constitute a real-time claim.

Gate 6 covers the failure paths that produce no usable iterate at all. A
bounded line search that stops after at least one accepted outer iteration is
not one of them: the carried iterate is a finite improvement on the seed, so it
is returned as a `stalled` plan rather than discarded for a previous-command
hold. A line search that fails before anything is accepted still returns
`line_search_failed` with the bounded fallback.

The baseline and normalized error definition are recorded with each report.
Thresholds may only change in a reviewed contract revision made before the
candidate being judged is tuned.

### Validation progression

The maintained progression is synthetic truth, fitted-model mismatch, then PX4
SITL. Real hardware is excluded from this project goal. Synthetic coverage must
include multirotor hover, translation, and attitude; fixed-wing trim, altitude
or path tracking, turning, and a flap-enabled configuration. Both structured
and structured-residual models must remain differentiable through the runtime
and solver paths.
