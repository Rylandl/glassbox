# PX4 SITL Multirotor Corpus Scaling and Maneuver-Family Benchmark

**What this establishes:** the typed trajectory contract and window-budget training policy scale from a handful of homogeneous SITL logs to a balanced 24-flight, four-maneuver-family corpus. On that expanded corpus, the structured model's leave-one-maneuver-family-out benchmark meets its ground-truth position development target only through 2 seconds and its attitude target only at 0.1 seconds, indicating the model is the limiting factor rather than the amount of repeated data.

## Purpose

This benchmark scales structured and structured-residual fitting across many PX4 SITL logs and maneuver families to verify that the semantic trajectory contract, dataset pooling rules, and window/weighting policy generalize beyond a single flight, then measures leave-one-maneuver-family-out generalization to motion types absent from training.

## Data

Canonical trajectory artifacts carry a typed, versioned semantic contract:

- `spec` defines the 13-state schema, observation source, ordered control roles/units/bounds, optional measured exogenous channels, optional state-aligned identification observations, and vehicle configuration identity.
- `labels` contains profile, condition, replicate, vehicle, payload, environment, and source-group annotations.
- `provenance` contains the source path, adapter identity, PX4 topics, raw actuator mapping, filters, and discarded intervals.

Source integrations implement the small `TrajectoryAdapter` boundary: `inspect(path)` audits the source schema and `load(path)` returns a canonical trajectory. Source-specific fields stop at `provenance`; fitting and dynamics consume the typed spec and numerical arrays only. For long recordings containing dropouts, `load_px4_trajectories()` preserves every valid contiguous interval rather than silently keeping only the longest. Reference-corpus adapters expose this behavior as `load_all()` while retaining the one-trajectory `load()` protocol for ordinary integrations.

Dataset pooling compares `spec`, not source details. Logs using different PX4 topics, fields, motor slots, or surface allocation matrices may pool when the adapter has converted them into the same verified canonical semantics. Logs with different observation sources, control meanings/order, fixed configuration states, or vehicle configuration IDs are rejected. Sample rate is checked separately because rollout windows share one integration step. The canonical NPZ format is version 3 and contains the state, control, exogenous, and training-only observation arrays plus typed `spec`, `labels`, and `provenance`; the loader rejects every other version, so derived trajectories should be re-extracted from raw telemetry rather than migrated through compatibility shims.

Observation channels are measured outputs used to identify dynamics, such as accelerometer specific force and filtered body angular acceleration. They are state-aligned but are not assumed to exist at rollout time. Model artifacts therefore exclude them from `input_spec` and record them separately as `identification_observations`. Exogenous channels are measured non-control inputs available when prediction starts, such as trusted wind. Each rollout samples them only at its initial timestamp and holds that value through the horizon, preventing future telemetry from leaking into an open-loop prediction. Trusted wind roles modify the air-relative velocity in the structured force law; estimated context may instead condition the generic residual, with estimated-wind features barred from directly changing its angular correction. PX4's sensor-aided `airspeed_wind` estimate is fully typed and retained by the general ULog adapter. IDF-DS excludes that derived estimate from model input by default because a matched experiment worsened every representative metric, while the Skywalker X8 adapter makes the opposite evidence-backed choice for its independently validated wind estimate; trust is a source-semantic fact, not a global setting.

The six usable SITL logs total about 73 seconds, but their paths, speed ranges, and actuator excitation show that they are repetitions of essentially the same takeoff-hover-landing profile, not a diverse flight corpus.

The original maneuver-family profile corpus contains eight flights and 150.1 seconds of usable ground truth. It spans about 0.9-2.1 m/s peak speed, 0.34-3.02 rad/s peak body rate, vertical-only motion, multi-metre lateral steps, large yaw steps, and combined translation/yaw.

The expanded `multirotor_v2` corpus contains 24 raw ULogs, 24 ground-truth trajectories, and 24 estimated-state trajectories. Its 428.0 seconds of usable ground truth are balanced at six flights per maneuver family, eight per excitation condition, and twelve per initial heading; all artifacts are finite and share the verified canonical motor schema. Peak speed reaches 3.16 m/s.

## Reproduce

For recorded profile corpora, re-extract them into the current trajectory format with:

```bash
uv run python scripts/reextract_profile_dataset.py \
  artifacts/sitl/multirotor_v2 \
  --platform multirotor \
  --vehicle-id px4_sih_quadx
```

Build a consistent 50 Hz dataset from a directory of raw ULogs:

```bash
./scripts/extract_ulog_dataset.sh path/to/logs artifacts/dataset_50hz 50
```

The script attempts both ground-truth and estimated-state extraction, reports logs without a valid airborne interval, and consistently uses the normalized `actuator_motors.control` command available in operational PX4 logs. Pass the desired state source from every flight to `glassbox-fit`; this example reserves two complete flights:

```bash
uv run glassbox-fit artifacts/dataset_50hz/*_ground_truth.npz \
  --holdout-count 2 \
  --training-horizons 0.1,0.5,2.0 \
  --model artifacts/scaled_model.json \
  --report artifacts/scaled_report.json
```

Multi-flight training gives every complete source group equal total loss weight by default and weights windows uniformly inside each group, so a long log cannot dominate and splitting one log around dropouts cannot increase its influence. Large corpora use an automatic deterministic window budget rather than exposing another fitting knob; small and medium corpora use every valid window, and only the fixed memory bound and the transition cost of long horizons thin large corpora.

Record the 24-flight expansion matrix (four bounded PX4 SITL profiles, three excitation conditions, and two replicates):

```bash
./scripts/record_sitl_profiles.sh artifacts/sitl/multirotor_v2
```

The recorder uses PX4's normal takeoff and landing behavior, streams local-NED position/yaw setpoints in Offboard mode, and extracts both estimated and ground-truth trajectories at 50 Hz using the operational `actuator_motors.control` signal. Each artifact stores its maneuver family, low/medium/high excitation condition, replicate, and initial-yaw variant in trajectory labels. The default output root is new and the recorder refuses to overwrite an existing run directory. Override the matrix with `GLASSBOX_PROFILE_REPLICATES`, `GLASSBOX_PROFILE_CONDITIONS`, and `GLASSBOX_PROFILE_INITIAL_YAWS`.

Reserve an entire maneuver family, not merely the last flight, with `--holdout-profile`:

```bash
uv run glassbox-fit \
  $(find artifacts/sitl/profile_dataset -name '*_ground_truth.npz' | sort) \
  --holdout-profile combined \
  --training-horizons 0.1,0.5,2.0 \
  --model artifacts/sitl/profile_holdout_combined_model.json \
  --report artifacts/sitl/profile_holdout_combined_report.json
```

When all inputs have profile labels, training first gives every included maneuver family equal total loss weight and then divides each family's weight equally among its replicate flights. Run every holdout fold and write a macro summary with:

```bash
uv run glassbox-profile-benchmark \
  $(find artifacts/sitl/profile_dataset -name '*_ground_truth.npz' | sort) \
  --output-dir artifacts/sitl/profile_benchmark
```

This leave-one-maneuver-family-out benchmark measures extrapolation to a type of motion absent from training rather than interpolation to another execution of a familiar flight. `summary.json` applies the versioned `multirotor_prediction_v1` development contract: a horizon passes only when both the equal-profile aggregate and worst held-out profile satisfy position and attitude limits, and full-flight position error is normalized by logged path length. These targets evaluate predictive usefulness and are not flight-safety or certification limits.

## Results

On a four-flight/two-flight split of the six-log SITL corpus, the ground-truth model trained at 0.1, 0.5, and 2 seconds reaches roughly 0.66 m position and 1.98 degrees attitude RMSE across the two complete holdouts; the estimated-state counterpart reaches roughly 0.43 m and 2.38 degrees. This is a more conservative baseline than the earlier homogeneous three-log result, and the scaled benchmark is a pipeline stress test, not evidence of general dynamics coverage.

The ground-truth position/attitude development targets are 0.001 m/0.25 degrees at 0.1 s, 0.01 m/1 degree at 0.5 s, 0.05 m/2 degrees at 1 s, 0.20 m/5 degrees at 2 s, and 0.75 m/10 degrees at 5 s. Estimated-state targets are 0.02 m/0.5 degrees, 0.08 m/2 degrees, 0.15 m/4 degrees, 0.30 m/7 degrees, and 1.0 m/12 degrees at the same horizons. Full-flight targets are at most 10% of path length and 10 degrees for ground truth, or 15% and 15 degrees for estimated state. The 0.1, 0.5, 1, and 2-second horizons plus full flight are required for an overall pass; missing required horizons produce an `incomplete` result.

On the original eight-flight profile corpus, the structured model's equal-profile ground-truth benchmark reaches 0.00014 m / 0.12 degrees at 0.1 seconds, 0.024 m / 3.52 degrees at 1 second, and 0.165 m / 6.51 degrees at 2 seconds. Complete 17-20-second open-loop rollouts reach 6.71 m / 15.27 degrees. Estimated-state training reaches 0.266 m / 6.73 degrees at 2 seconds and 7.59 m / 16.08 degrees over complete flights. Same-profile replicate holdouts have nearly identical error, showing that systematic rollout drift, not merely unseen profile labels, is the dominant limitation.

Re-extracting the retained ULogs through canonical format v3 and rerunning the expanded structured leave-one-family-out benchmark on `multirotor_v2` gives ground-truth position/attitude errors of 0.00012 m / 0.13 degrees at 0.1 seconds, 0.0031 m / 1.32 degrees at 0.5 seconds, 0.0216 m / 3.36 degrees at 1 second, 0.167 m / 6.57 degrees at 2 seconds, and 1.23 m / 10.18 degrees at 5 seconds. Complete flights reach 7.39 m / 16.56 degrees. The matched estimated-state benchmark reaches 0.0117 m / 0.34 degrees, 0.0543 m / 2.01 degrees, 0.109 m / 4.19 degrees, 0.284 m / 7.19 degrees, and 1.34 m / 10.87 degrees at the same horizons; complete flights reach 5.85 m / 15.84 degrees. The machine-readable rerun and no-promotion decision are recorded in [`multirotor-profile-results.json`](../results/multirotor-profile-results.json).

## Boundary

Both development-contract horizons fail on the expanded corpus: position meets the ground-truth target through 2 seconds, while attitude meets it only at 0.1 seconds. The modest local change relative to the smaller corpus, combined with poor full-flight behavior over a broader motion envelope, indicates that the structured model is the limiting factor, not the amount of repeated data.
