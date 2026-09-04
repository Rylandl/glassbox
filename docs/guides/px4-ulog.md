# PX4 ULogs

## Overview

The ULog and SITL tools need the `px4` extra: `uv sync --extra px4`, or
`pip install 'glassbox[px4]'`. `uv sync --dev` includes it.


This guide covers inspecting and extracting PX4 ULogs into Glassbox's canonical trajectory format, and recording a reproducible PX4 SIH simulator flight end to end. For benchmarks built on top of extracted PX4 telemetry, see the [multirotor](../experiments/px4-sitl-multirotor.md) and [fixed-wing](../experiments/px4-sitl-fixedwing.md) PX4 SITL corpora, or the real-flight [ARP](../experiments/arp.md), [IDF-DS](../experiments/idf.md), and [X8](../experiments/x8.md) references.

## Extract a ULog

Inspect the topics and fields in a log:

```bash
uv run glassbox extract --inspect path/to/flight.ulg
```

Extract estimated state and actuator commands to the canonical trajectory format:

```bash
uv run glassbox extract path/to/flight.ulg artifacts/flight.npz \
  --rate 50
```

For SITL logs containing the PX4 ground-truth topics, add `--state-source ground_truth`.

PX4 motor function order depends on the airframe configuration. Glassbox derives the front-left, front-right, rear-right, rear-left channel order from the logged `CA_ROTOR*_PX/PY` geometry. For unusual geometries, provide the channel order explicitly with `--motor-indices`.

For fixed-wing logs, join the normalized motor and servo allocator topics with:

```bash
uv run glassbox extract --family fixedwing path/to/plane.ulg \
  artifacts/plane.npz --rate 50
```

Automatic fixed-wing mapping requires one logged PX4 rotor and an independent roll/pitch control-surface allocation; yaw authority is optional. Glassbox reconstructs canonical aerodynamic-axis controls from `CA_SV_CS*_TRQ_R/P/Y`, converts PX4 FRD pitch/yaw moment signs into FLU, and records the raw and canonical mixing matrices under `provenance["px4"]`. Two-surface elevon configurations therefore ingest as `throttle, roll, pitch` without a fictional rudder channel. A nonzero `CA_SV_CS*_FLAP` allocation adds a typed `flap` channel. Explicit conventional aileron/elevator/rudder slots remain available for older logs. Pass a stable `--vehicle-id` when producing a multi-airframe corpus so the fitter cannot silently pool different physical vehicles.

### Telemetry gap tolerances

Two independent tolerances gate how much of a flight survives extraction. `--max-gap` (`max_gap_s`, default 0.10s) bounds how far state topics such as position, attitude, and angular velocity may be linearly or spherically interpolated across a genuine telemetry dropout. A separate actuator hold-age tolerance bounds how long the last `actuator_motors`/`actuator_servos` sample may be held valid before the next one arrives.

PX4's default logging profile publishes actuator topics roughly every 100 ms with ordinary scheduling jitter. Reusing `--max-gap` for the actuator hold-age tolerance treats that normal jitter as a dropout and fragments an otherwise continuous flight into many short segments, keeping only the longest one and silently discarding the rest. Unless you pass `--actuator-hold-max-age` explicitly, Glassbox resolves the hold-age tolerance per log as `max(max_gap_s, 1.5 * median actuator sample period)`, measured from the actuator topic actually used (the motor topic, and for fixed-wing logs the larger of the motor and servo topics). Pass `--actuator-hold-max-age` to pin an explicit value instead, for example to reproduce one fixed tolerance across a corpus recorded at different logging rates.

After extraction, `glassbox extract` reports how many contiguous valid segments the log produced and what fraction of the armed/in-air span the written segment covers:

```
wrote artifacts/flight.npz: 9000 intervals, 180.020s at 50 Hz
segments: 1 valid, coverage: 98.4%
```

If a log splits into more than one valid segment, or the written segment covers less than half of the flight's armed/in-air time, the command also prints a warning to stderr naming the resolved actuator hold age and pointing at `--max-gap` and `--actuator-hold-max-age`. That warning means most of the recorded flight was likely dropped; widen the tolerances or inspect the log for a genuine telemetry dropout before trusting the extracted trajectory.

### Reading `source_rates` in provenance

Every extracted trajectory records `provenance["px4"]["source_rates"]`, a mapping from each ingested topic's logical role (`position`, `attitude`, `angular_velocity`, `actuator` or `motor_actuator`/`servo_actuator`, and any of `armed`, `land`, `wind`, `specific_force`, `angular_acceleration` that were present in the log) to its median sample period in seconds, its maximum observed gap in seconds, its sample count, and the resampling method Glassbox used for it (`linear`, `slerp`, or `hold`). `provenance["px4"]["resolved_actuator_hold_max_age_s"]` records the tolerance that was actually applied, and `valid_segment_count`/`selected_segment_coverage` record the same segment and coverage figures the CLI prints. Together these let a consumer tell a native 50 Hz trajectory from one upsampled from a slower topic, and confirm why a particular segment boundary occurred.

## Record a SITL flight

With Docker Desktop running, record a PX4 SIH quadrotor takeoff-hover-landing flight and extract both estimated and ground-truth trajectories:

```bash
GLASSBOX_PROFILE_CONDITIONS=medium GLASSBOX_PROFILE_REPLICATES=1 \
  ./scripts/record_sitl_profiles.sh --family multirotor \
  artifacts/sitl/baseline baseline
```

The `baseline` profile flies PX4's own takeoff and landing rather than an offboard maneuver table, and fits a model from the ground-truth extraction. The script uses the same immutable multi-architecture PX4 SIH image digest as the integration gate and stores generated data under the ignored `artifacts/sitl/` directory. Set `GLASSBOX_PX4_IMAGE` only to make an explicit image comparison. [`logger_topics.txt`](../../config/logging/logger_topics.txt) overrides the dynamics topics to their full publication rates; the default PX4 logging profile is intended for flight review and records some actuator signals too slowly for identification. For SIH, the script extracts the normalized `actuator_outputs_sim` signal consumed by the simulator at 250 Hz.

The same fitting step can be run on any extracted trajectory:

```bash
uv run glassbox fit artifacts/flight.npz \
  --model artifacts/flight_model.json \
  --report artifacts/flight_fit.json
```

There are three holdout rules and one is always chosen. With one trajectory the fit is split chronologically, training on the first 70% (`--train-fraction`) and reserving the rest for a contiguous held-out rollout. With multiple trajectories the final complete source group is reserved when `source_group` labels separate the flights, and the final input trajectories in argument order otherwise (`--holdout-count`). `--holdout-profile` and the general `--holdout-label KEY=VALUE` instead reserve every flight whose label matches, whatever the argument order:

```bash
uv run glassbox fit artifacts/flight_1.npz artifacts/flight_2.npz \
  artifacts/flight_3.npz \
  --training-horizons 2.0 \
  --evaluation-horizons 0.1,0.5,1.0,2.0 \
  --model artifacts/multi_flight_model.json \
  --report artifacts/multi_flight_report.json
```

Every trajectory extracted from a PX4 ULog carries the source recording as its `source_group`. If telemetry gaps produce multiple retained intervals, those segments keep the same group, preventing a flight from leaking across a source-level holdout.

`--training-horizons` expresses rollout lengths in seconds, so the objective is independent of telemetry sample rate. Every horizon is trained on its own initial-loss-normalized objective, and several comma-separated values combine those objectives with equal weight. Longer horizons directly penalize compounding rollout drift, while shorter horizons emphasize fast local dynamics.

The SITL recorder uses a 2-second training horizon. This substantially reduces uninterrupted long-rollout drift while the evaluation report still exposes 0.1, 0.5, 1, and 2-second behavior. The same multi-flight command works for simulator ground-truth and estimated-state artifacts; estimated-state coefficients should be interpreted as predictive effective values because estimator filtering and noise are part of the observed signal.

The report contains aggregate and per-flight metrics for the complete rollout and each requested horizon. `--ablation no-lag` also fits an otherwise identical near-zero-lag model and reports the learned-lag improvement over it; with `--model` that ablation is written beside the belief with `_no_motor_lag` appended. Use `--holdout-count` to reserve more than one final source group, or more than one trajectory when groups are unlabeled.

Typed sensor channels such as specific force are recorded in the trajectory for telemetry auditing and are not used by the fitter today. Model files contain effective predictive coefficients, the exact runtime prediction contract, the training-only observation schema, fitting provenance, and bounded-compute local parameter information whitened by held-out forecast errors. That information records unresolved directions rather than turning a rank-deficient inverse into covariance. Model artifacts explicitly do not claim that effective coefficients are uniquely recovered physical parameters.

## Notes

Held-out evaluation also reports measured-state-reset one-step innovations. The latent actuator state is carried causally, but the 13-element rigid-body state is reset at every interval so temporal autocorrelation and current/past-control correlation can be inspected without complete-rollout drift. A companion model-independent compatibility check compares position increments with reported world velocity and attitude increments with reported body rate. Correlation flags are diagnostic rather than promotion criteria: estimator filtering, closed-loop feedback, and incompatible state channels can produce them without implying a missing aerodynamic term. The policy is maintained internally and adds no CLI knobs.

That closes the bounded literature-guided architecture cycle. Until materially new measurements or an externally validated method changes the evidence, Glassbox keeps the current dynamics model as an audited gray-box baseline.

The evaluation hierarchy is deliberately asymmetric. A material improvement in one identifiable state channel may authorize a research A/B for that channel; blanket/default adoption must improve every eligible channel. Reused evaluation flights are classified as research validation, and any production promotion would additionally require a genuinely fresh lockbox.
