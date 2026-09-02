# Crazyflow adjustable-arm prototype

`glassbox crazyflow prototype` is the first simulator-independent vertical
slice of the adjustable-arm recovery demonstration. It uses Crazyflow 0.3.2 as
a hidden first-principles plant with rotor dynamics at 500 Hz while Glassbox
receives canonical telemetry and returns normalized per-motor thrust commands
at 50 Hz.

This is a diagnostic prototype, not an acceptance gate, flight-safety claim, or
throw-to-recover claim. Crazyflow is an optional dependency and is not imported
by the core Glassbox package.

## Boundary

The adapter makes the following assumptions explicit and tested:

- Crazyflow XYZW quaternions are converted to canonical WXYZ storage.
- Crazyflow's z-up world and body-rate representation are treated as canonical
  NWU/FLU for this isolated experiment.
- Crazyflow motor order is permuted into Glassbox's front-left, front-right,
  rear-right, rear-left order. The two mixer matrices agree after permutation.
- Canonical normalized motor commands represent a fraction of a fixed maximum
  per-motor thrust. The adapter inverts Crazyflow's quadratic thrust curve into
  rotor RPM rather than pretending normalized command is linear in RPM.
- Requested commands and simulated applied rotor thrust are separate. Applied
  rotor state is retained as four typed, state-aligned observations.

The experiment driver changes arm length and scales the inertia tensor with the
square of arm length inside Crazyflow. The numeric target ratio, inertia,
physical arm length, mass, thrust curve, and torque curve are never passed to
Glassbox fitting or adaptation. The experiment report records the scenario
ratio, but the target trajectory presented to Glassbox has only an opaque
configuration ID.

Crazyflow's [first-principles model](https://learnsyslab.github.io/crazyflow/user-guide/dynamics/)
is deliberately more detailed than Glassbox's effective multirotor family: it
includes quadratic rotor force/torque curves, asymmetric rotor dynamics,
propeller gyroscopic terms, a full inertia tensor, and body drag.

## Evidence path

The full prototype performs these fixed steps:

1. Generate two prechange trajectories and fit one effective Glassbox model
   from state and motor-command telemetry.
2. Generate five arm configurations with two independent profiles each, fit
   every member independently, regress their effective parameters against the
   one known fleet arm coordinate, and retain only that rank-1 configuration
   direction. The rank-4 unprojected fit scatter remains reported but cannot
   masquerade as additional configuration evidence.
3. Build group-balanced 0.1 and 0.6 second forecast-error evidence from the
   fleet trajectories.
4. Propose an update from the first half of 0.8 seconds of unknown-target
   telemetry and validate it on the disjoint second half, with preceding motor
   commands used only as actuator initialization context.
5. Evaluate the accepted model on independent target telemetry.
6. Compare stale belief, adapted belief, adapted point mean, and an independent
   longer target-telemetry fit in rotor-level closed loop against the same
   modified Crazyflow plant.
7. In a separate real-time-paced recovery, begin on the stale controller, log
   the entire arrest, submit the first 0.8 second contiguous segment inside the
   learned validity envelope, and update the belief in a persistent spawned
   process prewarmed only with known-fleet telemetry. The worker is lower
   priority and thread-limited; on POSIX it is stopped with an acknowledged OS
   signal during every control calculation and resumed only in control slack.
   Only numeric parameter leaves and belief metadata cross IPC. The foreground
   retains stale control until the precompiled controller's first solve with
   the new mean is command-usable at a control boundary.
8. Pass every candidate command through a model-independent supervisor. Fresh,
   finite, bounded commands pass transparently. Invalid telemetry selects a
   configured collective hold; stale, invalid, unusable, or out-of-bounds
   commands and excessive tilt or body rates select a bounded geometric
   attitude/rate arrest with hysteresis. A fixed stale-command fault is injected
   after the adapted controller installs.
9. Run a separate fixed 16-case campaign through the adapted controller,
   supervisor, and hidden plant: one transparent nominal case plus every typed
   supervisor reason. Each case gets fresh plant/supervisor state and advances
   the true plant for one complete control interval. Corrupted campaign telemetry
   is never reused for model fitting or a belief update.

The initial Crazyflow-scale fit has roughly an order of magnitude more angular
command sensitivity than the older synthetic plant. The maintained eight-step
line search returns an explicit fallback at that scale. The prototype therefore
records a fixed 16-step backtracking policy for every condition. Its optimizer
ceiling is six iterations to reserve 50 Hz timing headroom for command
supervision. This is an experiment-specific numerical contract, not a silent
change to the library-wide NMPC policy.

## Current result

Recorded on 2026-09-01 after the belief-update criterion was corrected to score
candidates without the held-out bias against the bias-corrected incumbent with a
noise-scaled margin, and after the NMPC warm start was made to advance the plan
at block granularity. Both changes moved this result materially; the previous
record is described at the end of this section.

On the recorded Apple M3 CPU run, the telemetry-only baseline fit reduced its
fixed 0.1 second fitting loss from `0.02499` to `8.62e-6`. Independently fitted
fleet members recovered the expected decreasing effective angular authority as
arm length increased without receiving physical plant parameters.

The target update was rejected: no bounded local step improved the proposal
telemetry once candidates were scored without the held-out bias correction, so
the belief was returned unchanged (`no bounded local step improved proposal telemetry`). Independent
normalized 0.6 second prediction RMS therefore stayed at `0.04641`
(`1.000x`), and the adapted and stale recoveries are the same trajectory
(`1.000x` tail tracking and attitude/rate RMS). The stale belief's tail tracking
was `1.089x` the independent full target-telemetry fit. Every trace was
finite and command-bounded with zero fallback.

The validity result is now positive in this scenario: maximum actual validity
utilization was `0.946`, `0.946`, `0.947`, and `0.897` for stale belief, adapted belief,
adapted point, and the target-telemetry fit, so every recovery stayed inside
learned support with the corrected warm start. This run demonstrates neither
useful mean adaptation nor uncertainty-aware command differentiation; it
records that the honest update criterion declined to move the belief on 0.8 s
of this telemetry, which is the behavior the criterion is designed to have when
the evidence does not beat the corrected forecast.

The previous record, produced with the bias-corrected acceptance criterion and
the unshifted warm start, accepted the update (`0.503x` independent prediction
RMS, `0.936x` and `0.942x` recovery-tail ratios) and left support in every
recovery (utilization `1.234` to `1.293`). Those numbers are superseded.

The online path met every computation and absolute wall-clock deadline in the
recorded run, but its transactional update was likewise rejected, so no
candidate controller was installed and the stale controller flew the whole
window. With every recovery inside support, the evidence window began at
`0.00 s` with `41` samples and no excluded prefix. The
online gate, which requires an installed update, therefore fails; the fixed
supervisor fault injection that follows installation was not exercised in this
run, and the separate 16-case supervisor campaign below still passes.

The controller now treats fitted `ModelParams` as a dynamic JAX PyTree. Before
release it compiles and twice prewarms a template with the exact uncertainty
semantics expected after one accepted update. Rebinding an accepted mean then
shares those compiled functions and is rejected if any static model, actuation,
uncertainty, horizon, or support contract changes.

The earlier `3.846 s` update measurement was a second JAX compilation, not
steady-state fitting work. Timestamp-derived floating-point noise made the
observed sample period differ from the prewarmed runtime period even though it
passed the timing tolerance; because integration step size is static, that
created a different compilation key. Adaptation now validates the observed
period and then uses the exact runtime-contract period. Candidate rollouts and
validity utilization are also evaluated by one compiled batched kernel instead
of nested eager transforms.

The recorded worker still reports its cold prewarm separately from the
prewarmed path. After that prewarm, the target belief update, candidate
first-solve validation, and controller installation each completed within a
few control periods, and the accepted controller was installed at simulation
time `1.30 s` after two stale-controller ticks. Every complete step of the
audited 50 Hz loop finished inside the 20 ms computation budget: zero deadline
misses and zero fallback. Fitting is therefore no longer the dominant
post-evidence delay in this prewarmed path. The report records the wall and
CPU times.

The supervisor passed the adapted controller's nominal commands unchanged
except for the fixed injected fault. A command timestamp made `40 ms` stale was
rejected and replaced with finite, bounded rate-arrest output, followed by four
latched arrest steps before nominal authority returned. Its decision time is
negligible next to the command period (the report records the maximum). The
supervisor uses no fitted dynamics, optimizer,
position target, or trajectory; its configured collective calibration and
fixed attitude/rate gains remain an integration responsibility.

The separate campaign covered all 15 typed supervisor reasons plus a nominal
pass-through. Every expected authority mode and reason matched; every supervised
command and post-step true plant state was finite and bounded. Invalid state and
quaternion observations caused real NMPC `invalid_input` results, and a forced
controller deadline caused `deadline_exceeded`; the supervisor still selected
the configured collective hold or rate arrest. Supervisor decision time again
stayed negligible next to the command period. These are deterministic
one-interval fault
contracts, not evidence of recovery from sustained or interacting faults.

The report separately records desktop scheduling jitter. One command
completion crossed its absolute wall-clock schedule by a fraction of a
millisecond despite remaining below the 20 ms computation budget. That metric is not hidden
inside the passing computation gate: Python, macOS, and a simulator are not a
hard-real-time flight runtime. Commands and states remained finite and bounded,
and the machine-readable artifact remains authoritative for the recorded run.

## Run

Install the pinned optional simulator and execute the full prototype:

```bash
uv sync --extra crazyflow
uv run --extra crazyflow glassbox crazyflow prototype \
  artifacts/crazyflow_prototype
```

The output is `artifacts/crazyflow_prototype/report.json`. To exercise only the
fast frame, motor-order, thrust-map, and telemetry contracts:

```bash
uv run --extra crazyflow glassbox crazyflow prototype \
  artifacts/crazyflow_contract --plant-contract-only
```

## Next gate

The next gate is a time-series disturbance campaign: adaptation-worker failure
or hang, sensor noise, delay and dropout bursts, actuator/motor mismatch,
battery variation, ground contact, randomized initial throws, interacting
faults, and repeated seeded runs. Passing one isolated interval is insufficient;
the campaign must measure arrest, bounded dwell, authority restoration, and
post-fault recovery. The supervisor boundary must then be integrated below a
firmware-representative estimator and command transport before cautiously
expanding the release distribution.
