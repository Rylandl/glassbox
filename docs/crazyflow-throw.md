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
check. Feedback uses the committed belief while information-directed probing
uses the live working belief, so validation does not introduce a temporal
collect-then-fly phase.

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

- the first supported feedback contribution occurs at `1.05 s`;
- the accepted initial proposal is frozen after interval `110`, then admitted
  after `16` future predictions at `2.26 s`;
- the recursive update median is about `0.34 ms` on the recorded machine;
- the working belief is updated on all `900` post-enable intervals; and
- no simulator reset or controller handoff occurs.

Across the full run, the validator accepts the initial belief and seven later
replacements while rejecting eight candidates that do not demonstrate enough
future improvement. A post-admission information probe is capped at `0.0005`
of normalized motor range, independently of the larger initial identification
signal.

At the `10.00 s` end of the uninterrupted run:

- speed is `0.0111 m/s`, or `0.00110x` release speed;
- body-rate norm is `0.0120 rad/s`, or `0.0111x` release rate;
- vertical speed is below `0.00008 m/s` in magnitude;
- tilt is `0.00099 rad`;
- minimum altitude is the `1.200 m` release height; and
- the strict hover envelope remains satisfied for the final `5.73 s`.

The certified belief estimates hover at `0.53194742`, versus the hidden
evaluation-only value `0.53194727` (`0.000028%` relative error). All requested and
measured motor values remain finite and bounded.

Run the diagnostic with:

```bash
uv sync --extra crazyflow
uv run --extra crazyflow glassbox-crazyflow-throw \
  docs/crazyflow-throw-results.json
```

The [machine-readable result](crazyflow-throw-results.json) records the full
configuration, working and certified beliefs, timing, state metrics,
observations, and limitations.

## Development campaign

A fixed five-scenario campaign varies hidden arm length and release state. It
is deliberately recorded as development evidence—not held-out validation—and
retains failures instead of reducing the output to successful examples:

```bash
uv run --extra crazyflow glassbox-crazyflow-throw \
  docs/crazyflow-throw-campaign-results.json --campaign
```

Three of five scenarios pass the complete canonical gate. The canonical case,
a `1.15x` shorter-arm case, and a `1.35x` longer-arm cross-axis tumble all pass.
Every scenario keeps commands finite and bounded, and all five finish with
speed at or below `0.0154 m/s`, rate at or below `0.0127 rad/s`, and at least
`5.07 s` in the strict hover envelope.

The two failed gates matter. A lower-energy release misses the final vertical
speed threshold by `0.00078 m/s`. A reversed tumble eventually stabilizes but
contacts the simulated ground first, reaching `-0.001 m`; it therefore cannot
count as a successful recovery. The
[campaign artifact](crazyflow-throw-campaign-results.json) records both failures
and a `3/5` pass rate. This small tuned scenario set does not measure a
generalization probability.

## Real-time animations

The animation exporter now maps simulation time directly to video time. It
adds no title holds, fit pauses, repeated frames, or time-compressed recovery.
The throw-only video is the exact first second of the same telemetry used by
the full recovery video, so their trajectories match sample-for-sample until
online activation.

Render both in one run with:

```bash
uv sync --extra crazyflow-animation
uv run --extra crazyflow-animation glassbox-crazyflow-throw-animation \
  artifacts/crazyflow_throw/online-recovery.mp4 \
  --poster artifacts/crazyflow_throw/online-recovery-poster.png \
  --gif artifacts/crazyflow_throw/online-recovery-preview.gif \
  --throw-only-output artifacts/crazyflow_throw/unpowered-throw.mp4 \
  --throw-only-poster artifacts/crazyflow_throw/unpowered-throw-poster.png \
  --throw-only-gif artifacts/crazyflow_throw/unpowered-throw-preview.gif
```

## Why separate working and committed beliefs?

Continuous fitting and continuous control do not require treating every new
regression as more informative. The working belief assimilates every interval
and decides which input directions still need information. The committed
belief is the last frozen candidate that predicted a disjoint future window
better than its reference. Only it supplies feedback after initial admission.

This is an epistemic separation inside one real-time loop, not two temporal
controllers. It prevents a transient closed-loop regression from immediately
rewriting the controller, while still allowing validated replacements. The
short one-step validation window is evidence of local predictive improvement;
it is not a calibrated safety probability or a proof of trajectory-wide
validity.

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
sensor delay/noise, a rate-arrest supervisor, netted testing, and an explicit
validity/abort envelope.
