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

This closed-loop Crazyflow result, its reproduction commands, and the
annotated animation renderer now live in the
[glassbox-throw](https://github.com/Rylandl/glassbox-throw) repository, along
with the checked-in machine-readable result it cites.

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


## Aggregated transitions

`RecursiveBootstrapConfig.transition_aggregation_steps` makes the identifier
assimilate one sample per that many measured transitions: the window's mean
features and mean targets, weighted by the window length. One, the default,
is bit-for-bit the identifier as it was.

The reason is measurement noise on a differenced target. The collective
target is the specific force implied by the velocity change over one
interval, so white noise on the measured velocity is multiplied by the loop
rate: at a hundred hertz, two centimetres per second of velocity noise
becomes nearly three metres per second squared of target noise, against under
two from a probe of a tenth of the command range. The mean over a window
telescopes most of that away, because consecutive differences share their
inner samples with opposite signs. Weighting the aggregated sample by the
window length keeps the sample count, the support thresholds, and the
residual floor exactly per transition, so with a noise-free measurement the
information rate is unchanged for a command held across the window, and
under noise the residual variance the identifier estimates falls by about
the window length.

The cost is the variation inside the window: a window that straddles two
excitation blocks averages their difference away. On the throw study a
window of five slowed identification by half; two and three are measured on
the release ensemble in the dual-control NMPC design, now documented in the
[glassbox-throw](https://github.com/Rylandl/glassbox-throw) repository.


## Prequential residual

`RecursiveBootstrapConfig.prequential_residual` floors each residual scale at
the belief's own recent prediction error: the error it makes predicting each
new transition before absorbing it, averaged with exponential forgetting over
`minimum_certification_interval_count` transitions. Off, the default, the
identifier is bit-for-bit as it was.

The in-sample residual of a regression with as many samples as parameters is
nothing, so a rank-deficient angular map fitted on a tumbling vehicle reads
as certain: its coefficient covariance is the residual floor over a handful
of samples, and a controller that trusts that covariance commands hard
through a map that is wrong. The prequential error is what the map actually
gets wrong, and it falls as soon as the map is right. The two are combined by
taking the larger, so the scale can only rise on the switch, never fall.

An error is only a model's error once there is a model. Before the command
evidence supports a fit the prediction is the intercept and the error is the
target itself, which is ignorance rather than misspecification, so nothing is
recorded until the command evidence has rank one, and each angular axis is
recorded only while its own authority is positive. Recording from the first
transition instead kept the authority at zero for half a second on the throw
study and crashed most releases.


## Integrated collective fit

`RecursiveBootstrapConfig.integrated_collective` fits the collective map on
the cumulative target rather than the per-interval one. The per-interval
target is the body-z specific force implied by the velocity change over one
interval, so measurement noise on the velocity enters divided by the
interval. Summed over the transitions since the anchor, the projected
velocity changes telescope: the sum carries the anchor's noise once, common
to every row, the latest sample's noise once, and a small term from how far
the body axis rotated each step. Regressing the cumulative target on the
cumulative features with one constant column for the anchor is therefore
the exact least-squares form for white measurement noise on the velocity,
and its rows are independent given that column.

The integrated system's information is exported to the rest of the
identifier as an equivalent per-transition Gram: the anchor column is
marginalized, the system's residual scale is estimated from its own residual
with the declared force floor over one interval as its minimum, and the
marginal is rescaled so that dividing it by the declared floor, which is the
residual then reported, gives the integrated precision. Support, authority,
the belief, and any planner reading the belief see the honest information
without changing. The angular regression is untouched.

Two things this settles. In a noise-free simulation the form behaves like
the per-interval one. Under measurement noise it says what the per-interval
form cannot: the collective level is known well within a tenth of a second,
while the differential coefficients, whose probes integrate to a few
millimetres per second against two centimetres of noise, are not, and the
per-interval fit's confidence in them was optimism. A first version with
three world-axis rows per transition was measured and dropped: the model
explains only the body-z force, so the other two rows carried unmodeled
force and the residual came out forty times the floor.
