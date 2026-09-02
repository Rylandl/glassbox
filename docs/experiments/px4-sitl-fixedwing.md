# PX4 SITL Fixed-Wing Baseline and Platform-Neutral Objective

**What this establishes:** the shared fixed-wing model-family contract, force law, and long-rollout training objective produce a stable baseline on both a closed-world synthetic corpus and a real PX4 SIH airplane maneuver-family corpus. Short and medium-horizon accuracy is strong, but complete-profile open-loop rollout still diverges on both corpora, and neither participates in the real-flight cross-airframe development contract.

## Purpose

Every fixed-wing fit in Glassbox shares one model-family contract, force law, and long-rollout training objective. This page defines that contract for both the compact structured law and its optional structured residual, then benchmarks the structured baseline first on a synthetic smoke corpus and then on a real PX4 SITL fixed-wing maneuver-family corpus. For real-flight fixed-wing results, see [IDF-DS](idf.md) and [X8](x8.md); those feed the [cross-airframe development gate](fixedwing-gate.md) that this simulator-only corpus does not.

The shared model-family contract separates platform, control names, semantic roles, latent applied-control state, and residual support. Fixed-wing requires the roles `throttle`, `roll`, and `pitch`; `yaw` and `flap` are optional. Control columns may use airframe-specific names or order because the dynamics indexes the static role layout carried by `TrajectorySpec`. A conventional trajectory uses `throttle, aileron, elevator, rudder`, while a flying wing can use `throttle, roll, pitch`. Each fit compiles one fixed layout, so incompatible configurations are rejected during pooling rather than mixed inside a JAX batch.

The compact fixed-wing force law has body-X thrust, quadratic air-relative lift and drag, a linearized angle-of-attack lift term, lateral velocity damping, surface moments proportional to airspeed squared, pitch and lateral-directional stability, airspeed-scaled angular-rate damping, signed lateral surface cross-coupling, bounded learned surface-neutral offsets, and a first-order actuator response. Its coefficients are effective acceleration parameters: mass, inertia, density, and reference area are intentionally absorbed. The model uses typed trusted wind when it is available and otherwise assumes zero wind. It remains a low-angle attached-flow model and does not model stall, propeller slipstream, unresolved gusts, or configuration changes. When a moving flap role is present, additional learned coefficients model its incremental lift, deployment drag, and signed pitch moment. Lateral cross-coupling allows rudder-to-roll and aileron-to-yaw moments without assuming either a conventional tail or a particular mixer. This expanded structure serializes as `effective_fixedwing_role_aerodynamic_lag_v3`; model artifacts use a single current version and reject obsolete parameter layouts.

## Data

The closed-world synthetic smoke corpus is generated on demand (see Reproduce) and is used to verify family dispatch, differentiability, fitting, evaluation, holdouts, and JSON round-tripping. It is not a real-airframe accuracy result.

The `fixedwing_v2` profile-only PX4 SIH corpus contains 12 ULogs, 12 ground-truth trajectories, and 12 estimated-state trajectories. It provides 177.58 seconds of ground truth at 50 Hz across 5.00-6.78 m/s, with one verified motor/surface mapping in every artifact. Pooled throttle spans 0.755-1.0; aileron, elevator, and rudder contain nonzero variation without surface saturation. An earlier takeoff-inclusive corpus is retained separately because full-power takeoff occupied 65% of its throttle samples and would bias a pooled maneuver fit.

## Reproduce

Generate and benchmark the closed-world synthetic smoke corpus:

```bash
uv run glassbox-fixedwing-synthetic artifacts/fixedwing/synthetic_v1 \
  --flights 6 --duration 3
uv run glassbox-profile-benchmark artifacts/fixedwing/synthetic_v1/*.npz \
  --output-dir artifacts/fixedwing/profile_benchmark_synthetic_v1 \
  --training-horizons 0.1,0.5,1.0 \
  --evaluation-horizons 0.1,0.5,1.0,2.0 \
  --steps 600
```

Record the standard PX4 SIH airplane matrix:

```bash
./scripts/record_fixedwing_sitl_profiles.sh artifacts/sitl/fixedwing_v2
```

The recorder explicitly reconciles the SIH plant's 5-6 m/s envelope with PX4 runway-takeoff parameters, climbs before excitation, rotates the logger after takeoff, streams bounded throttle/roll/pitch/combined attitude profiles in Offboard mode, and extracts estimated and ground-truth trajectories. The default matrix is four profiles by three excitation conditions by two replicates. Set `GLASSBOX_PROFILE_REPLICATES=1` for the initial 12-flight pass; set `GLASSBOX_PROFILE_REPLICATE_START=2` with a final replicate count of two to append only the second replicate.

Fit the optional structured residual on any fixed-wing trajectory set:

```bash
uv run glassbox-fit trajectory_1.npz trajectory_2.npz trajectory_3.npz \
  --model-class structured_residual \
  --training-horizons 0.1,0.5,2.0,5.0 \
  --skip-no-lag-ablation
```

The wrapper retains the selected vehicle family's force law, exact rigid-body position/quaternion kinematics, and latent applied-control response. A 16-unit network predicts only six bounded corrections: body-linear and body-angular acceleration. Its inputs are the frame-invariant body velocity, angular rate, the typed canonical applied controls, and any typed start-of-rollout exogenous context. Feature normalization and correction bounds are derived from the training windows and serialized with the model; there are no multirotor motor indices, hover assumptions, fixed-wing surface names, or NanoDrone operating-range constants in the residual. Zero initialization exactly reproduces the structured base model on both current vehicle families.

Both the endpoint weight and stability regularization used by the shared long-rollout objective are explicit fit flags, recorded in every fit report:

```bash
uv run glassbox-fit trajectory_1.npz trajectory_2.npz \
  --training-horizons 0.1,0.5,1.0 \
  --endpoint-weight 3.0 \
  --stability-regularization 0.01
```

Every structured and structured-residual fit uses the same training policy. Position, velocity, attitude, and angular velocity are treated as four equally weighted semantic groups after scaling by robust within-window motion observed only in the training data. Position scales use displacement rather than world coordinates, so changing the trajectory origin does not change the objective. Quaternion attitude error is sign-invariant. Time weights increase linearly from one at the first predicted step to three at the final step by default. A separate soft regularizer detects predicted body-frame velocity or angular rate outside a generous training-derived envelope: the half-width is at least four robust scales and covers at least the 99.5th percentile of training motion. Position is deliberately unbounded, and the objective does not assume that an aircraft or multirotor is contractive; the regularizer targets numerical escape, not legitimate travel or open-loop vehicle modes.

## Results

Across three leave-one-multisine-profile-out folds, the fitted synthetic baseline reaches 0.0031 m position and 0.016 degrees attitude RMSE over the complete three-second synthetic flights.

On a leave-one-maneuver-family-out benchmark of `fixedwing_v2` trained at 0.1, 0.5, and 2 seconds, the structured model reaches ground-truth position/attitude RMSE of 0.00058 m / 0.071 degrees at 0.1 seconds, 0.0122 m / 0.87 degrees at 0.5 seconds, 0.050 m / 2.07 degrees at 1 second, 0.185 m / 4.19 degrees at 2 seconds, and 1.01 m / 10.94 degrees at 5 seconds. The matched estimated-state results are 0.0119 m / 0.23 degrees, 0.0578 m / 1.51 degrees, 0.126 m / 3.13 degrees, 0.294 m / 5.08 degrees, and 0.855 m / 9.71 degrees. Complete-profile open-loop rollout still diverges: 6.54 m / 30.4 degrees on ground truth and 14.24 m / 69.6 degrees on estimated state. Those numbers support an evidence-based operating envelope of roughly two seconds for sub-0.3 m / 6-degree prediction and five seconds for approximately one-metre / 11-degree prediction on this simulator.

On the held-out synthetic fixed-wing smoke flight, the shared long-rollout objective remains stable for both model classes. After 100 steps, the structured model reaches 0.0213 m / 0.086 degrees over the complete flight; the structured residual reaches 0.0088 m / 0.206 degrees. The opposing position/attitude trade supports retaining the structured family as the default rather than selecting a residual solely from one aggregate score.

## Boundary

The synthetic corpus verifies the pipeline, not real-airframe accuracy, and does not participate in the real-flight fixed-wing development contract. The `fixedwing_v2` results do not participate in the real-flight cross-airframe contract either: the threshold-setting data and scoring data are the same single-airframe, single-replicate-per-condition corpus, and complete-profile rollout still diverges. The structured model remains the recommended default until a residual improves leave-profile-out long-rollout accuracy without destabilizing another motion family.
