# No-prior multirotor bootstrap identification

`BootstrapMultirotorIdentifier` is the first deliberately incomplete model in
Glassbox. It estimates only what is needed to establish local collective and
three-axis angular authority:

- four motor-command effects on body-specific-force z;
- the full `3 × 4` motor-command effect on body angular acceleration;
- linear and quadratic body-rate nuisance terms;
- a level zero-velocity hover command; and
- the command subspace actually supported by the evidence.

It receives timestamps, canonical rigid-body states, measured applied motor
inputs, and command bounds. It does not receive a motor mixer, hover command,
mass, inertia, arm length, thrust coefficient, fleet model, or nominal
`DynamicsParams`. Requested commands are insufficient because actuator lag can
make them materially different from the inputs that affected the vehicle.

The identifier has a fixed evidence shape. `prewarm()` compiles that exact JAX
shape using generic synthetic arrays; `fit()` then reuses it. Motor effects are
fit only in singular-vector directions supported by the observed applied
inputs. Unsupported directions remain zero. Separate held-out gates report
whether collective evidence supports a hover estimate and whether angular
evidence supports three-axis rate arrest.

The nuisance block follows the same rule. Body velocity, body rate, and rate
product columns are inverted only along directions the window actually excited,
at `nuisance_rank_relative_tolerance` of the leading nuisance direction. The
constant intercept column supplies the unit scale for that comparison: its
root-mean-square is exactly one, so a relative threshold is also an absolute
floor in each feature's own units. Without it, a feature that barely moves takes
a coefficient set by measurement noise divided by a near-zero excursion, and
that coefficient then enters every later prediction. `collective_nuisance_rank`
and `angular_nuisance_rank` report how many directions survived. The threshold
bounds unexcited directions only. A weakly but genuinely excited direction, such
as a rate product over a short evidence window, is still inverted and its
coefficient is still only as good as its signal-to-noise ratio.

```python
from glassbox.experimental import (
    BootstrapIdentificationConfig,
    BootstrapMultirotorIdentifier,
    plan_bootstrap_excitation,
)

provisional_identifier = BootstrapMultirotorIdentifier(
    BootstrapIdentificationConfig(interval_count=24)
)
final_identifier = BootstrapMultirotorIdentifier(
    BootstrapIdentificationConfig(interval_count=28)
)
provisional_identifier.prewarm()
final_identifier.prewarm()
provisional = provisional_identifier.fit(
    timestamps_s,
    states_wxyz_nwu_flu,
    applied_motor_commands,
)

follow_up = plan_bootstrap_excitation(provisional)
# Apply follow_up.commands and append the measured states/applied inputs.
bootstrap = final_identifier.fit(
    combined_timestamps_s,
    combined_states_wxyz_nwu_flu,
    combined_applied_motor_commands,
)

if bootstrap.ready:
    decision = bootstrap.velocity_attitude_rate_arrest_command(
        world_velocity_m_s=states_wxyz_nwu_flu[-1, 3:6],
        quaternion_wxyz=states_wxyz_nwu_flu[-1, 6:10],
        angular_velocity_rad_s=states_wxyz_nwu_flu[-1, 10:13],
        previous_command=applied_motor_commands[-1],
        maximum_motor_step=0.08,
    )
```

The first controller is intentionally local and explanatory. It solves for the
smallest motor change whose identified direct effect produces the requested
angular deceleration. Rate and rigid-body coupling terms are useful for fitting
and held-out validation but are not extrapolated by this controller. Commands
are clipped to known bounds and can be slew limited. An unready fit raises
`BootstrapModelNotReadyError` instead of silently inserting a canonical mixer.

## The recursive counterpart

`RecursiveBootstrapIdentifier` runs the same estimator continuously, updating a
working belief after every measured actuation interval and applying the same
command and nuisance support rules to its running Gram matrices. Its
`forgetting_factor` is pinned at `1.0`. A decaying window would drop the working
belief to rank zero once excitation stops, and committed excitation is capped far
too low to rebuild that rank, so any smaller value is rejected rather than
silently flown.

`ProgressiveBootstrapController` is the command side of that estimator, and it is
not an optimizer. A fixed stabilizing cascade with hand-set gains maps velocity
error to a thrust direction, thrust direction to a tilt error, and tilt error to
motor deltas through the identified angular effect and its pseudo-inverse. The
belief enters only by scaling authority and supplying that allocation. A bounded
five-point scan then chooses how much information excitation rides on top of the
cascade, scoring each candidate by its distance from the cascade, minus the
excitation it buys, plus an altitude-risk penalty, plus a supported-covariance
term left inert by its configured weight. The scan takes the largest excitation
unless altitude risk or the bound and slew clipping prefer less, so the score
selects the probing, not the stabilizing action.

Neither entry point ends a control loop over one bad sample. `update` refuses a
non-finite or out-of-bounds transition, leaves the belief exactly as it was, and
records the refusal in `last_sample_report`. `command` returns a decision with
`command_usable` false, a reason, and the previous command clipped into bounds,
falling back to the bounded midpoint hold. Applied and candidate commands within
a rounding width of the bounds are clipped rather than refused.

## Crazyflow result

The maintained hidden-plant diagnostic starts all motors at the midpoint of the
known normalized command bounds and applies independent bounded excitation. A
24-interval provisional fit rejected itself because yaw validation was worse
than its nuisance-only baseline. `plan_bootstrap_excitation` selected that
weakest output and generated four symmetric commands from its learned input
direction. No hidden mixer or plant value was used to select them. On the
recorded modified-arm run:

- after both fixed fit shapes were prewarmed, the provisional and final fits
  each completed in well under a millisecond (the report records the wall
  times);
- 28 total intervals reduced evidence duration to `0.56 s`;
- all four command directions and all three angular output directions were
  supported;
- estimated per-motor hover was `0.53213`, versus the hidden evaluation-only
  value `0.53195` (`0.034%` relative error); and
- an independent velocity/attitude/rate rollout reduced linear-speed norm from
  `1.526 m/s` to `0.231 m/s` (`0.151×`), angular-rate norm from `1.598 rad/s`
  to `0.184 rad/s` (`0.115×`), and tilt from `0.249 rad` to `0.034 rad`.
  Terminal vertical speed was `0.0049 m/s`; altitude excursion was `0.487 m`;
  commands remained finite and bounded.

Run it with:

```bash
uv sync --extra crazyflow
uv run --extra crazyflow glassbox crazyflow bootstrap \
  artifacts/crazyflow_bootstrap/report.json
```

The same canonical run can be replayed through Crazyflow's offscreen MuJoCo
renderer as an annotated H.264 animation. The video preserves the distinction
between the rejected provisional fit, the four learned follow-up pulses, and
the independently reset stabilization trial. It does not turn the diagnostic
into a physical-throw or flight-safety claim.

```bash
uv sync --extra crazyflow-animation
uv run --extra crazyflow-animation glassbox crazyflow animation \
  artifacts/crazyflow_bootstrap/no-prior-bootstrap.mp4 \
  --poster artifacts/crazyflow_bootstrap/no-prior-bootstrap-poster.png \
  --gif artifacts/crazyflow_bootstrap/no-prior-bootstrap-preview.gif
```

The exporter requires `ffmpeg` on `PATH`. It renders the real Crazyflow drone
mesh and state trace; playback timing is deliberately stretched for legibility,
while every displayed telemetry value remains tied to the recorded simulator
samples. `CrazyflowBootstrapTrace` is the typed state-aligned replay boundary,
so presentation code does not need access to hidden plant parameters.

The checked-in [machine-readable result](../results/crazyflow-bootstrap-results.json)
records the fitted matrix, support projectors, validation metrics, timing, and
independent recovery outcome.

This closes a narrow but important loop: direct input effects can be identified
quickly without an airframe parameter prior and immediately used for bounded
velocity, level-attitude, and rate stabilization. It is not yet a
throw-to-recover result.
The experiment uses exact simulator state, measured applied rotor state, a
normalized command interval, and an airborne midpoint excitation. The recovery
starts from throw-like velocities but uses a simulator reset rather than a
physical toss. It does not yet include sensor noise, delay, position regulation,
ground handling, or a safe policy for choosing initial thrust when normalized
bounds have weak physical meaning.
