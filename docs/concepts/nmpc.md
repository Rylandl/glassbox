# Glassbox nonlinear model-predictive control

Glassbox NMPC turns one eligible fitted dynamics belief into a finite-horizon
rigid-body tracker. The interface is intentionally small: a runtime belief, a
state estimate, a state reference, the previous command, optional applied
control or latent actuator state, physical tracking tolerances, and optional
state limits. Horizon length, command blocking, line search, regularization,
and iteration count are maintained policies rather than routine user knobs.
When a belief supplies predictive-error evidence, the maintained horizon is
capped at that evidence boundary.

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

The first cold solve compiles both nominal and support-filter paths; the first
warm-started solve compiles the receding-horizon path. Run both and discard
their commands before entering a timed control loop. Compilation must never
happen after arming.

For an online belief mean change, `controller.rebind_belief(updated_belief)`
returns an immutable controller handle that shares the original JIT functions
and supplies the new `ModelParams` PyTree dynamically. Rebinding is intentionally
strict: input/runtime specifications, actuation, parameter-tree shapes,
uncertainty numerics, prediction horizon, and derived support horizon must match
the precompiled template. A mismatch raises `ValueError` instead of compiling a
different program on the control path. The integrator must construct and fully
prewarm the expected post-update template before arming; rebinding does not turn
an arbitrary belief change into a compatible hot swap.

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
typed channel bounds. After nominal optimization, Glassbox evaluates total
forecast spread in the same normalized tracking coordinates. If its maximum
standard deviation exceeds one declared tracking tolerance, the optimized
change from the previous command is scaled by the reciprocal spread. The
diagnostic `command_authority_fraction` records the result. This is a maintained
bounded-authority policy, not an uncertainty calibration claim.

Command change, full-horizon model-validity excess, and `SafetyEnvelope` state
limits remain dimensionless soft penalties. `SafetyEnvelope` currently supports
minimum/maximum world position, maximum world speed, and maximum body angular
speed. These mission limits are preferences, not invariant-set or collision
guarantees.

The command returned for the next interval has a separate belief-support
projection. It derives an actuator-reaction horizon from twice the model's
slowest learned actuator time constant, bounded to 0.1--0.3 seconds and never
longer than the NMPC horizon. The optimized sequence is evaluated first. If it
is unsupported,
Glassbox evaluates maintained blends between the optimized NMPC command and the
previous bounded command. This path uses only the typed dynamics belief and the
NMPC solution; it contains no motor mixer, control-surface law, family-specific
recovery policy, or independently operating controller.

Inside the learned envelope, a candidate is accepted only when its predicted
body velocity and angular rate stay within support over that reaction horizon.
The margin adds one componentwise standard deviation from the current dynamics
belief; a point model contributes no invented spread. Outside support, the
terminal robust utilization and normalized angular-rate energy must both
decrease. If no maintained candidate satisfies the relevant condition, the
least-bad bounded command is returned with an explicit `boundary_best_effort`
or `recovery_best_effort` mode. The full NMPC prediction can still leave support:
this receding projection is not a robust invariant-set proof.

Important result fields are:

- `status`: converged, stalled, finite iteration-limit plan, or an explicit
  failure;
- `command_usable`: whether the command comes from a finite optimized plan;
- `predicted_states`, `predicted_latent_states`, and `predicted_commands`;
- initial/final objective, iteration count, raw and bound-projected gradient
  infinity norms, and solve time;
- maximum command-bound violation, model-validity utilization, and normalized
  safety-limit violation;
- total normalized model uncertainty, the uncertainty-aware command-authority
  fraction, plus separate predictive-error availability, error-evidence
  currency, horizon support, and parameter-uncertainty flags;
- support-filter mode and intervention, current/next-step robust utilization,
  reaction-horizon maximum and terminal utilization, rate energy, retained
  nominal-command fraction; and
- an opaque `warm_start` for the next receding-horizon solve.

`converged` is reserved for the first-order criterion. That criterion tests the
bound-projected gradient, `blocks - clip(blocks - gradient)`, against the
maintained tolerance rather than the raw gradient, because a raw gradient
component pointing outward at an active command bound never shrinks however
optimal the iterate is. The projected residual is reported as
`final_projected_gradient_inf_norm` next to the raw
`final_gradient_inf_norm`, so the status can be audited from the result.

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
even the maximum observed solve stayed at roughly half the 50 ms model step.
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
uv run glassbox px4-nmpc-shadow artifacts/px4/model.json \
  --previous-command 0.5,0.5,0.5,0.5 \
  --output artifacts/px4/nmpc-shadow.json
```

The shadow runner executes cold and warm-up controller solves before sampling,
holds the current state as the regulation reference, and applies the artifact's
sample period as the deadline for every measured solve. It records solver
status, fallback rate, model-period deadline misses, message skew, estimated
source-clock lag and real-time ratio, current and predicted validity, and
command bounds. It also records support-filter modes, interventions, and
reaction-horizon robustness. Returned commands are written
only to the report. The previous command remains the measured/applied command
because the shadow command is not being actuated.

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
otherwise evaluation stops instead of silently pairing unrelated samples. The
report includes their actual skew, receive age, armed state, and per-channel
peak-to-peak excitation. This makes the fixture evidence for asynchronous
telemetry, varying commands, moving states, solver deadlines, and cleanup—not a
closed-loop control test and not a general claim that HIL actuator order is
shared by other PX4 configurations.

The same report performs a short-horizon logged-input model audit. Each
prediction starts from a measured state, carries the model's latent actuator
state, holds the timestamp-aligned starting command, and integrates the
continuous dynamics across the state stream's actual source-time interval.
This matters because the maintained onboard estimator stream advances at a
quantized 24/32/40 ms cadence even though the fitted model and controller use a
20 ms period. Intervals from 0.5 to 2.5 model periods are scored at their actual
duration; larger coalesced gaps are reported and re-anchored because their
intermediate inputs are unknown.

The audit reports position, velocity, attitude, and body-rate RMSE against both
the next telemetry state and a constant-world-velocity/constant-body-rate
persistence baseline, plus their ratios. It is diagnostic evidence, not an
acceptance gate. The maneuver matrix has shown why one global ratio would be
misleading: a model can improve translational prediction while exposing a
rotational deficiency on the same profile. Model promotion remains the job of
held-out recorded-flight benchmarks with airframe-relevant horizons. This live
fixture gates at least 90% temporal eligibility, finiteness of every error and
ratio, maneuver-specific excitation, synchronization, command bounds, and
cleanup.

A transport test can pass while the real-time gate fails. In particular, PX4
SIH and JAX share host resources in this fixture, so the report treats source
clock progress and solver latency as separate measurements. Any
`deadline_exceeded` sample returns the bounded previous command and is counted
as a fallback; it is not presented as a usable controller output.

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
uv run glassbox nmpc-benchmark \
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
dynamics belief -> compact runtime belief -> NMPC -> canonical command
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
- an optional prediction horizon certified by named external evidence;
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
predictive-error evidence. After direct-shooting optimization, total forecast
standard deviation is normalized by the declared tracking tolerances. Spread
above one tolerance proportionally bounds the optimized change from the
previous command. This small discrete safety coupling avoids differentiating
through covariance Jacobians inside every solver iteration and remains visible
as `command_authority_fraction`. Glassbox deliberately does not embed a second
airframe-specific controller behind this NMPC path.

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
