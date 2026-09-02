# Online Crazyflow throw-and-recover diagnostic

This fixed diagnostic has no collect-then-control phase. A modified-arm
Crazyflow plant is thrown with its motors stopped. For exactly the first
`1.00 s`, motor requests, applied motor state, model updates, and controller
authority are all zero. At `1.00 s`, bounded actuation and recursive
identification begin together on the same uninterrupted simulation.

Every subsequent `10 ms` interval updates a working local belief. The
controller uses only currently supported input/output directions, so command
authority grows continuously rather than appearing at a single model-ready
handoff. Authority depends on supported parameter covariance, command
information, and estimated effect signal-to-noise—not elapsed sample count.

Persistent belief updates are transactional. A supported working belief is
frozen as a proposal, scored prequentially on the next `16` intervals before
those observations enter that proposal, and committed only if it improves
future force and angular-acceleration predictions. Later candidates use the
last committed belief as their reference and must also pass a bounded-movement
check.

That model transaction does not create another controller, and there is no
collect-then-fly mode change. The command itself is simpler than an optimizer.
A fixed stabilizing cascade with hand-set gains maps velocity error to a thrust
direction, thrust direction to a tilt error, and tilt error to motor deltas
through the identified angular effect and its pseudo-inverse. The belief enters
that cascade only by scaling its authority and supplying the allocation, and the
cascade is computed before and independently of any candidate score.

An information excitation is then added on top of the cascade, at one of five
fixed fractions of its amplitude chosen by a bounded scan. The scan scores each
candidate by its distance from the cascade, minus the excitation it buys, plus
an altitude-risk penalty, plus a supported-covariance term whose configured
weight leaves it inert at every reachable magnitude. Because the reward is
linear in the fraction and the distance is quadratic at the same scale, the scan
takes the largest excitation unless the altitude-risk term or the command bound
and slew clipping prefer less. Command bounds and slew limits define the
feasible set. There is one controller tier and no fallback policy, but the
belief-space score selects only how much excitation rides on a fixed cascade,
not the stabilizing action itself.

Glassbox receives timestamps, exact canonical simulator state, measured applied
motor state, normalized command bounds, and four-channel output shape. It does
not receive hover command, mass, inertia, arm length, thrust coefficients, or a
canonical motor mixer.

## Fixed release and result

The canonical run uses:

- `1.2 m` release height;
- world velocity `[1.0, -0.6, 10.0] m/s` (`10.068 m/s` norm);
- body rates `[0.8, -0.6, 0.4] rad/s` (`1.077 rad/s` norm);
- `0.249 rad` initial tilt;
- stopped motors and no model/controller activity through `1.00 s`; and
- a hidden `1.25x` arm-length configuration.

The unpowered first second is visibly ballistic: tilt reaches `1.202 rad`, and
the airframe reaches `4.821 m` altitude before online actuation starts. After
activation:

- (the interval and time values in this list are outcomes of the recorded
  run; which interval the candidate freezes on, and everything downstream of
  it, flips on last-bit numerical differences and is not a stable property)
- the first supported stabilization contribution occurs at `1.03 s`;
- the accepted initial proposal is frozen after interval `172`, then admitted
  after `16` future predictions at `2.88 s`;
- each recursive update completes in a small fraction of the `10 ms` motor
  interval on the recorded host (the report records the median);
- the working belief is updated on all `900` post-enable intervals; and
- no simulator reset or controller handoff occurs.

Across the full run, the validator admits the initial belief and rejects all
seven later candidates: none beats the honestly conditioned nuisance-only
reference on the next `16` intervals, so no replacement commits. The bounded
scan selects a nonzero information action on `725` of `900` intervals,
reaching `0.12` of normalized motor range before validation and at most
`0.0005` afterward. That excitation is added to the fixed stabilizing
cascade, not issued by a second policy.

At the `10.00 s` end of the uninterrupted run:

- speed is `0.0110 m/s`, or `0.00110x` release speed;
- body-rate norm is `0.0119 rad/s`, or `0.0110x` release rate;
- vertical speed is below `0.0005 m/s` in magnitude;
- tilt is `0.00097 rad`;
- minimum altitude is the `1.200 m` release height; and
- the strict hover envelope remains satisfied for the final `5.47 s`.

The validated predictive belief estimates hover at `0.53211`, versus the hidden
evaluation-only value `0.53195` (`0.030%` relative error). All requested and
measured motor values remain finite and bounded.

Run the diagnostic with:

```bash
uv sync --extra crazyflow
uv run --extra crazyflow glassbox crazyflow throw \
  docs/results/crazyflow-throw-results.json
```

The [machine-readable result](../results/crazyflow-throw-results.json) records the full
configuration, working and validated beliefs, objective diagnostics, timing, state metrics,
observations, and limitations.

## Development campaign

A fixed five-scenario campaign varies hidden arm length and release state. It
is deliberately recorded as development evidence—not held-out validation—and
retains failures instead of reducing the output to successful examples:

```bash
uv run --extra crazyflow glassbox crazyflow throw \
  docs/results/crazyflow-throw-campaign-results.json --campaign
```

In the recorded run, two of five scenarios pass the complete canonical gate: a
`1.15x` shorter-arm high release and a milder low-energy release. Which cases
pass the replacement criterion is an outcome of that run, not a stable
property; the flight-quality bounds below are. Every scenario keeps commands
finite and bounded, and all five finish with speed at or below `0.0165 m/s`,
rate at or below `0.0128 rad/s`, and at least `4.83 s` in the strict hover
envelope.

The three failed gates matter. The canonical case and the `1.35x` longer-arm
cross-axis tumble commit no belief replacement after initial admission, and
the tumble also misses the final vertical speed threshold. A reversed tumble
misses the same vertical speed threshold and dips to `0.845 m`, below the
`1 m` altitude floor, before stabilizing. The
[campaign artifact](../results/crazyflow-throw-campaign-results.json) records all three
failures and a `2/5` pass rate. This small tuned scenario set does not measure
a generalization probability.

## Real-time animations

The animation exporter now maps simulation time directly to video time. It
adds no title holds, fit pauses, repeated frames, or time-compressed recovery.
The throw-only video is the exact first second of the same telemetry used by
the full recovery video, so their trajectories match sample-for-sample until
online activation.

Render both in one run with:

```bash
uv sync --extra crazyflow-animation
uv run --extra crazyflow-animation glassbox crazyflow throw-animation \
  artifacts/crazyflow_throw/online-recovery.mp4 \
  --poster artifacts/crazyflow_throw/online-recovery-poster.png \
  --gif artifacts/crazyflow_throw/online-recovery-preview.gif \
  --throw-only-output artifacts/crazyflow_throw/unpowered-throw.mp4 \
  --throw-only-poster artifacts/crazyflow_throw/unpowered-throw-poster.png \
  --throw-only-gif artifacts/crazyflow_throw/unpowered-throw-preview.gif
```

## Why keep transactional model evidence?

Continuous fitting and continuous control do not require treating every new
regression as more informative. The working belief assimilates every interval
and contributes current information geometry. The validated predictive mean is
the last frozen candidate that predicted a disjoint future window better than
its reference. Both are inputs to one command objective; neither is a fallback
controller or a safety net.

This prevents a transient closed-loop regression from being mistaken for
information gain while still allowing validated replacements. The short
one-step validation window is evidence of local predictive improvement; it is
not a calibrated safety probability or a proof of trajectory-wide validity.

## What this does not establish

Crazyflow injects the release state. It does not model a person's hand,
contact, separation detection, propeller clearance, estimator startup, or the
radio/firmware scheduling boundary. Exact simulator state and measured rotor
state are available without noise or latency. The first post-enable input is
centered at the midpoint of known normalized command bounds. The controller
arrests velocity, attitude, and rates; it does not hold position, so the large
horizontal throw displacement is expected.

This is a simulation diagnostic, not a physical throw-to-recover or
flight-safety claim. Hardware progression still needs release detection,
sensor delay/noise, netted testing, and physical constraints represented inside
the same command objective.

The replacement criterion is the least robust part of this gate. Once the
nuisance-only reference is conditioned with the same rank policy as the
candidate, the prequential bar is honest and no post-admission candidate clears
it in the recorded run. The same criterion also flips on last-bit numerical
differences amplified over the 900 closed-loop steps, so a pass or fail on it
should not be read as a stable property of the controller; every flight-quality
observation is unchanged. Whether the gate should keep that criterion is an
open decision recorded here rather than tuned away.
