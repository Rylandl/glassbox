# PX4 ULogs

The ULog and SITL tools need the `px4` extra: `uv sync --extra px4`, or
`pip install 'glassbox[px4]'`. `uv sync --dev` includes it.

This guide covers inspecting and extracting PX4 ULogs into the canonical
trajectory format, fitting a belief from what comes out, and recording a
reproducible PX4 SIH simulator flight end to end. The numbers measured on PX4
SITL corpora are on [validation](../validation.md#px4-sitl-corpora); recording
one needs a simulator container, so they are prose rather than artifacts.

## Extract a ULog

Inspect the topics and fields in a log:

```bash
uv run glassbox extract --inspect path/to/flight.ulg
```

Extract estimated state and actuator commands to the canonical trajectory
format:

```bash
uv run glassbox extract path/to/flight.ulg artifacts/flight.npz --rate 50
```

For SITL logs containing the PX4 ground-truth topics, add
`--state-source ground_truth`. Several logs at once write
`OUT/<log stem>_<state source>.npz` for each, so one invocation builds a whole
corpus directory:

```bash
uv run glassbox extract path/to/logs/*.ulg artifacts/dataset_50hz --rate 50
```

PX4 motor function order depends on the airframe configuration. Glassbox
derives the front-left, front-right, rear-right, rear-left channel order from
the logged `CA_ROTOR*_PX/PY` geometry. For unusual geometries, provide the
channel order explicitly with `--motor-indices`.

For fixed-wing logs, join the normalized motor and servo allocator topics
with:

```bash
uv run glassbox extract --family fixedwing path/to/plane.ulg \
  artifacts/plane.npz --rate 50
```

Automatic fixed-wing mapping requires one logged PX4 rotor and an independent
roll and pitch control-surface allocation; yaw authority is optional. Glassbox
reconstructs canonical aerodynamic-axis controls from `CA_SV_CS*_TRQ_R/P/Y`,
converts PX4 FRD pitch and yaw moment signs into FLU, and records the raw and
canonical mixing matrices under `provenance["px4"]`. Two-surface elevon
configurations therefore ingest as `throttle, roll, pitch` without a fictional
rudder channel. A nonzero `CA_SV_CS*_FLAP` allocation adds a typed `flap`
channel. Explicit conventional aileron, elevator and rudder slots remain
available for older logs with `--surface-indices`. Pass a stable
`--vehicle-id` when producing a multi-airframe corpus so the fitter cannot
silently pool different physical vehicles.

### Telemetry gap tolerances

Two independent tolerances gate how much of a flight survives extraction.
`--max-gap` (default 0.10 s) bounds how far state topics such as position,
attitude and angular velocity may be linearly or spherically interpolated
across a genuine telemetry dropout. A separate actuator hold-age tolerance
bounds how long the last `actuator_motors` or `actuator_servos` sample may be
held valid before the next one arrives.

PX4's default logging profile publishes actuator topics roughly every 100 ms
with ordinary scheduling jitter. Reusing `--max-gap` for the actuator hold age
treats that normal jitter as a dropout and fragments an otherwise continuous
flight into many short segments, keeping only the longest one and silently
discarding the rest. Unless you pass `--actuator-hold-max-age` explicitly,
Glassbox resolves the hold-age tolerance per log as
`max(max_gap_s, 1.5 * median actuator sample period)`, measured from the
actuator topic actually used. Pass `--actuator-hold-max-age` to pin an
explicit value instead, for example to reproduce one fixed tolerance across a
corpus recorded at different logging rates.

After extraction, `glassbox extract` reports how many contiguous valid
segments the log produced and what fraction of the armed and in-air span the
written segment covers:

```text
wrote artifacts/flight.npz: 9000 intervals, 180.020s at 50 Hz
segments: 1 valid, coverage: 98.4%
```

If a log splits into more than one valid segment, or the written segment
covers less than half of the flight's armed and in-air time, the command also
prints a warning to stderr naming the resolved actuator hold age and pointing
at `--max-gap` and `--actuator-hold-max-age`. That warning means most of the
recorded flight was likely dropped; widen the tolerances or inspect the log
for a genuine telemetry dropout before trusting the extracted trajectory.

### Reading `source_rates` in provenance

Every extracted trajectory records `provenance["px4"]["source_rates"]`, a
mapping from each ingested topic's logical role (`position`, `attitude`,
`angular_velocity`, the actuator topics, and any of `armed`, `land`, `wind`,
`specific_force`, `angular_acceleration` that were present) to its median
sample period in seconds, its maximum observed gap, its sample count, and the
resampling method Glassbox used for it (`linear`, `slerp` or `hold`).
`provenance["px4"]["resolved_actuator_hold_max_age_s"]` records the tolerance
that was actually applied, and `valid_segment_count` and
`selected_segment_coverage` record the same segment and coverage figures the
command prints. Together these let a consumer tell a native 50 Hz trajectory
from one upsampled from a slower topic, and confirm why a particular segment
boundary occurred.

## The pinned public corpora

The five reference corpora produce the same canonical NPZ format without any
of the flags above:

```bash
uv run glassbox corpus list
uv run glassbox corpus prepare arp artifacts/arp_reference
```

`prepare` writes verified sources under `DIR/raw` and canonical trajectories
under `DIR/canonical`, preserving the upstream split as subdirectories where
the corpus publishes one. `--raw` reuses an already-verified source tree so a
second canonical copy does not download the corpus twice, and `fetch` obtains
the sources without converting them. Verification is not optional: a pinned
corpus that does not verify is a different corpus. The registry names each
corpus's citation, license, pinned files, parser and published evaluation
split, and `glassbox corpus list` renders with no optional extra installed.

## Fit from what came out

```bash
uv run glassbox fit artifacts/flight.npz \
  --model artifacts/flight_model.json \
  --report artifacts/flight_fit.json
```

There are three holdout rules and one is always chosen. With one trajectory
the fit is split chronologically, training on the first 70 percent
(`--train-fraction`) and reserving the rest for a contiguous held-out rollout.
With multiple trajectories the final complete source group is reserved when
`source_group` labels separate the flights, and the final input trajectories
in argument order otherwise (`--holdout-count`). `--holdout-profile` and the
general `--holdout-label KEY=VALUE` instead reserve every flight whose label
matches, whatever the argument order:

```bash
uv run glassbox fit artifacts/dataset_50hz/*_ground_truth.npz \
  --holdout-count 2 \
  --training-horizons 0.1,0.5,2.0 \
  --evaluation-horizons 0.1,0.5,1.0,2.0 \
  --model artifacts/multi_flight_model.json \
  --report artifacts/multi_flight_report.json
```

Every trajectory extracted from a PX4 ULog carries the source recording as its
`source_group`. If telemetry gaps produce multiple retained intervals, those
segments keep the same group, so a flight cannot leak across a source-level
holdout.

`--training-horizons` expresses rollout lengths in seconds, so the objective is
independent of telemetry sample rate. Every horizon is trained on its own
initial-loss-normalized objective, and several comma-separated values combine
those objectives with equal weight. Longer horizons directly penalize
compounding rollout drift, while shorter horizons emphasize fast local
dynamics. Multi-flight training gives every complete source group equal total
loss weight and weights windows uniformly inside each group, so a long log
cannot dominate and splitting one log around dropouts cannot increase its
influence. `--duration-weighted-training` weights by extracted window count
instead. One deterministic window budget governs both what the fitter extracts
and what one gradient step processes; a corpus with fewer windows than the
budget uses every valid one.

`--holdout-profile` reserves an entire maneuver family rather than the last
flight, which measures extrapolation to a type of motion absent from training:

```bash
uv run glassbox fit artifacts/sitl/multirotor_v2/*_ground_truth.npz \
  --holdout-profile combined \
  --training-horizons 0.1,0.5,2.0 \
  --model artifacts/sitl/profile_holdout_combined_model.json \
  --report artifacts/sitl/profile_holdout_combined_report.json
```

When every input has a profile label, training first gives every included
maneuver family equal total loss weight and then divides each family's weight
equally among its replicate flights. To run every fold and write a macro
summary instead of one fit, use the holdout runner:

```bash
uv run glassbox evaluate --hold-out profile \
  artifacts/sitl/multirotor_v2/*_ground_truth.npz \
  --output-dir artifacts/sitl/profile_benchmark
```

The fit report contains aggregate and per-flight metrics for the complete
rollout and each requested horizon. `--ablation no-lag` also fits an otherwise
identical near-zero-lag model and reports the learned-lag improvement over it;
with `--model` that ablation is written beside the belief.
`--fixed-response-time-constant` pins the applied-control response instead of
learning it, which is the cleaner baseline when the recorded input is measured
actuator state rather than a command. `--diagnostics` additionally runs the
one-step innovation and kinematic compatibility checks on every held-out
flight and records them in the report; they are opt-in because they are
telemetry auditing rather than a promotion criterion. Estimator filtering,
closed-loop feedback and incompatible state channels can raise a correlation
flag without implying a missing aerodynamic term.

Typed sensor channels such as specific force are recorded in the trajectory
for telemetry auditing and are not used by the fitter. Model artifacts contain
effective predictive coefficients, the exact runtime prediction contract, the
training-only observation schema, fitting provenance, and the parameter
information the fit resolved. That information records unresolved directions
rather than turning a rank-deficient inverse into covariance, and the
artifacts explicitly do not claim that effective coefficients are uniquely
recovered physical parameters.

Use `--include-ground` only with a model that includes ground-contact
dynamics.

## Record a SITL flight

With Docker running, `scripts/record_sitl_profiles.sh` records one bounded
maneuver profile per run in a disposable PX4 SIH container and extracts both
estimated and ground-truth trajectories:

```bash
GLASSBOX_PROFILE_CONDITIONS=medium GLASSBOX_PROFILE_REPLICATES=1 \
  ./scripts/record_sitl_profiles.sh --family multirotor \
  artifacts/sitl/baseline baseline
```

The `baseline` profile flies PX4's own takeoff and landing rather than an
offboard maneuver table, and additionally fits a model from the ground-truth
extraction. It is the smallest end-to-end check that the whole pipeline still
runs. For SIH the script extracts the normalized `actuator_outputs_sim` signal
the simulator consumes, at 250 Hz.

Without a profile argument the script flies its family's default matrix. The
multirotor profiles are `vertical_steps`, `lateral_steps`, `yaw_steps` and
`combined`, streamed as local NED position and yaw setpoints in Offboard mode
around PX4's normal takeoff and landing. The fixed-wing profiles are
`throttle_steps`, `roll_steps`, `pitch_steps` and `combined`, streamed as
attitude and throttle setpoints; that family reconciles the SIH plant's
airspeed envelope with PX4's runway-takeoff parameters, climbs before
excitation, and rotates the logger after takeoff so the maneuver is not
diluted by full-power takeoff samples.

`GLASSBOX_PROFILE_REPLICATES`, `GLASSBOX_PROFILE_REPLICATE_START`,
`GLASSBOX_PROFILE_CONDITIONS` and `GLASSBOX_PROFILE_INITIAL_YAWS` override the
matrix, whose default is four profiles by three excitation conditions by two
replicates. Each extracted artifact stores its maneuver family, excitation
condition, replicate and initial-yaw variant in trajectory labels, which is
what `--holdout-profile` and `--hold-out profile` later select on. The script
refuses to overwrite an existing run directory.

The script uses the same immutable multi-architecture PX4 SIH image digest as
the integration gate and stores generated data under the ignored
`artifacts/sitl/` directory. Set `GLASSBOX_PX4_IMAGE` only to make an explicit
image comparison. [`logger_topics.txt`](../../config/logging/logger_topics.txt)
overrides the dynamics topics to their full publication rates; the default PX4
logging profile is intended for flight review and records some actuator
signals too slowly for identification.

To fly one profile against a PX4 instance you started yourself, use the
command the script calls:

```bash
uv run glassbox sitl-profile lateral_steps --family multirotor --condition medium
```

It transmits setpoints only inside the declared profile table and records
nothing itself: PX4 writes the log, and `glassbox extract` converts it. The
passive shadow path, which never transmits at all, is
[`glassbox px4-shadow`](../concepts/nmpc.md#px4-shadow-mode).
