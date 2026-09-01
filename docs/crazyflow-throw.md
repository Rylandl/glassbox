# Online Crazyflow throw-and-recover diagnostic

This fixed diagnostic has no collect-then-control phase. A modified-arm
Crazyflow plant is thrown with its motors stopped. For exactly the first
`1.00 s`, motor requests, applied motor state, model updates, and controller
authority are all zero. At `1.00 s`, bounded actuation and recursive
identification begin together on the same uninterrupted simulation.

Every subsequent `10 ms` interval updates a working local belief. The
controller uses only currently supported input/output directions, so command
authority grows continuously rather than appearing at a single model-ready
handoff. Once a fully supported belief passes the structural admission
contract, it is retained for control. That first admission checks input rank,
output rank, authority, sample count, and feasible hover; it is not disjoint
predictive validation. Later closed-loop samples still update the working
candidate, but loss of independent excitation cannot erase previously
established support.

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

- the first supported feedback contribution occurs at `1.09 s`;
- four input directions and three angular-effect directions are certified at
  `1.53 s`;
- the recursive update median is about `0.15 ms` on the recorded machine;
- the working belief is updated on all `900` post-enable intervals; and
- no simulator reset or controller handoff occurs.

At the `10.00 s` end of the uninterrupted run:

- speed is `0.0054 m/s`, or `0.00054x` release speed;
- body-rate norm is `0.0184 rad/s`, or `0.0170x` release rate;
- vertical speed is `-0.0050 m/s`;
- tilt is below `0.00003 rad`;
- minimum altitude is the `1.200 m` release height; and
- the strict hover envelope remains satisfied for the final `4.26 s`.

The certified belief estimates hover at `0.53071`, versus the hidden
evaluation-only value `0.53195` (`0.23%` relative error). All requested and
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

## Why retain a certified belief?

Continuous fitting and continuous control do not require treating every new
regression as more informative. Once the vehicle is nearly stable, feedback
commands become correlated and stop independently exciting all motor
directions. The terminal working candidate correctly reports less independent
support than the earlier maneuver. It therefore cannot replace the supported
control belief. Candidate replacement is deliberately not implemented until
there is independent predictive validation. This follows the broader
`propose -> validate -> commit` design, but the current demo implements only
the initial structural admission and the reject-on-lost-support behavior.

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
