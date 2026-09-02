# Nano-Quadrotor System Identification Benchmark

**What this establishes:** the promoted structured model reaches boundary-safe accuracy on the IDSIA Nano-Quadrotor benchmark's protected Melon test flights that is near-comparable to the published Phys+Res reference (an equal-metric geometric cumulative ratio of `0.9957`), while remaining worse than a naive baseline on angular velocity. This is competitive benchmark performance, not a state-of-the-art claim or evidence of complete-flight open-loop stability.

> **Recorded before the 2026-09-01 estimator revisions.** The artifacts behind
> this page were not regenerated because the run is long, so three conventions
> differ from current code. Rollout error statistics now exclude the shared
> measured initial sample (`metric_policy` v2); the absolute RMSE values here
> are therefore lower than a fresh run would report, by at most a factor of
> sqrt(H / (H + 1)) at horizon H steps, while every model-versus-baseline ratio
> is unaffected. Minibatched fits, which large corpora trigger at the 0.5 s and
> 2 s horizons, weighted each training window by the square of its intended
> weight; the current objective is `deterministic_weighted_minibatch_v3`.
> Complete-flight rollouts held logged wind at its first sample; fixed-horizon
> window metrics are unaffected.

## Purpose

The [IDSIA Nano-Quadrotor benchmark](https://github.com/idsia-robotics/nanodrone-sysid-benchmark) tests whether Glassbox's effective dynamics models can identify a small multirotor from real motion-capture and onboard telemetry, then predict held-out flights at rolling horizons up to 0.5 seconds using the benchmark's own 1-to-50-step rolling metric.

## Data

`glassbox-nanodrone prepare` downloads all 15 official CSV recordings from the pinned upstream commit, verifies their Git LFS SHA-256 values, and writes 12 canonical training trajectories and three canonical Melon test trajectories under `artifacts/nanodrone/canonical/`. The raw upstream files remain separate under `artifacts/nanodrone/raw/`; Glassbox does not vendor the dataset. Checksum verification is strict by default so an upstream revision cannot silently change a benchmark run. For individual files, use `glassbox-nanodrone inspect` or `glassbox-nanodrone extract`.

The adapter preserves the benchmark's 100 Hz sampling, 13-state rigid-body trajectory, and three published body-specific-force channels. The state is a processed mixed-source observation, not uniform ground truth: position and attitude include motion capture, while velocity and angular rate come from onboard sensing/estimation. The accelerometer outputs are typed as state-aligned FLU observations used only during identification. The upstream world and body frames already have Glassbox's z-up/FLU signs; the adapter reorders the scalar-last quaternion to wxyz.

The benchmark inputs are measured rotor angular velocities rather than motor commands. The source order `m1,m2,m3,m4` (front-right, rear-right, rear-left, front-left) is verified from the published allocation matrix and reordered to Glassbox's front-left, front-right, rear-right, rear-left order. Each input is stored as `(omega / 2500 rad/s)^2`, a dimensionless thrust proxy that retains the published quadratic rotor relationship and keeps the optimizer well scaled. The 2500 rad/s reference is only a coordinate scale and is recorded in the typed spec; it is not an assumed motor limit. Consequently, a fitted control time constant describes any residual alignment/filtering response in measured RPM, not command-to-motor physical lag.

The v3 multirotor model can represent one bounded shared normalized-command offset in its collective thrust map. NanoDrone's typed `squared_rotor_speed_ratio` inputs are already physical thrust proxies, so this parameter is held exactly at zero and cannot be enabled for this dataset.

## Reproduce

```bash
uv run glassbox-nanodrone prepare artifacts/nanodrone
```

Fit against the benchmark's trajectory-level split:

```bash
uv run glassbox-fit \
  artifacts/nanodrone/canonical/train/*.npz \
  artifacts/nanodrone/canonical/test/*.npz \
  --holdout-profile melon \
  --training-horizons 0.1,0.5,1.0 \
  --evaluation-horizons 0.1,0.5,1.0 \
  --model-class structured_residual \
  --skip-no-lag-ablation \
  --model artifacts/nanodrone/model.json \
  --report artifacts/nanodrone/report.json
```

Score either saved model with the benchmark's rolling 1-to-50-step metrics:

```bash
uv run glassbox-nanodrone evaluate \
  artifacts/nanodrone/model.json \
  artifacts/nanodrone/canonical/test/*.npz \
  --report artifacts/nanodrone/benchmark_report.json
```

This initializes a 0.5-second open-loop prediction at every admissible sample in each Melon run and reports per-horizon mean Euclidean position, velocity, attitude-geodesic, and angular-velocity errors. It never lets a window cross a flight boundary. The upstream released metric script concatenates the three runs before shifting truth rows by horizon; those shifts cross two recording boundaries and reproduce the paper's slightly higher naive values exactly. Glassbox deliberately reports the boundary-safe version.

## Results

The checked workflow was run for 400 Adam steps at 0.1 and 0.5-second training horizons, with the 12 Square/Random/Chirp flights used for fitting and all three Melon runs held out. Because the input is measured RPM, the fixed 0.1 ms response model is the cleaner baseline; learning the response produced an effectively equivalent 3.96 ms value.

Boundary-safe errors for the structured no-lag model are:

| Horizon | Position [m] | Velocity [m/s] | Attitude [rad] | Angular velocity [rad/s] |
| ---: | ---: | ---: | ---: | ---: |
| 0.01 s | 0.00126 | 0.01591 | 0.00112 | 0.09102 |
| 0.10 s | 0.01135 | 0.13463 | 0.02801 | 0.48293 |
| 0.50 s | 0.10837 | 0.36582 | 0.36871 | 1.17149 |

The cumulative 1-to-50-step errors are 2.2875 m, 11.2634 m/s, 7.4349 rad, and 39.6054 rad/s, respectively. Relative to the boundary-safe hold-state naive baseline, this is 7.73x better on position, 3.44x on velocity, 1.10x on attitude, and 0.73x on angular velocity. The hold-state naive baseline holds the initial state fixed rather than propagating constant velocity, so it is weaker than the kinematic constant-velocity persistence baseline used by the other benchmarks in this project. The last value is the important failure: the compact structured angular-rate model is worse than holding the measured rate constant.

The platform-neutral structured residual is trained at 0.1, 0.5, and 1.0 seconds with the shared long-rollout objective (see [Cross-platform fitting-policy selection](fitting-policy.md)). The default multirotor reference uses the exact memoryless torque map because these inputs are measured rotor speed. Automatic batching is governed by unrolled transition count, so this medium corpus uses all 7,192 valid multi-horizon windows while larger or longer-horizon corpora remain bounded.

A maintainer-owned train-only sweep also evaluates a bounded angular-dynamics authority. It adds no user-facing fitting knob: authority one is the fitted model and zero approaches constant measured body rate. Leave-one-profile-out selection on Chirp, Random, and Square chose 0.75, improving the equal-profile, equal-horizon, equal-metric score by 4.98%. More conservative candidates were rejected because their worst individual metric regressed by more than 25%. Melon is excluded from selection and used once for the protected promotion check.

The promoted NanoDrone artifact has boundary-safe cumulative 1-to-50-step position, velocity, attitude, and angular-rate errors of 2.2176 m, 12.2297 m/s, 5.4456 rad, and 29.1772 rad/s. At 0.1 seconds the corresponding errors are 0.0118 m, 0.0927 m/s, 0.0263 rad, and 0.4642 rad/s; at 0.5 seconds they are 0.1176 m, 0.5569 m/s, 0.2308 rad, and 0.7093 rad/s. The protected check improved all eight cumulative and 0.5-second metrics relative to full angular authority.

Against Phys+Res in the paper's Table 6, its equal-metric geometric cumulative ratio is 0.9957, effectively parity: Glassbox is better on cumulative position and attitude, 17.6% worse on velocity, and 0.66% worse on angular rate. At 0.5 seconds position, velocity, and attitude are within 5.2% of the published values, while angular-rate error remains 18.6% higher. The comparison is near-comparable, not exact, because Glassbox never crosses recording boundaries while the released upstream metric code concatenates the three Melon runs. This is competitive benchmark performance, not a state-of-the-art claim or evidence of complete-flight open-loop stability.

## Boundary

A learned latent rotational response and cross-axis mixer were also evaluated. They improved the training-profile score but failed the one-shot Melon promotion check, so neither is enabled by default. The expressive implementation and audit artifacts remain available for future airframes whose telemetry supports them.

The angular-rate model remains worse than the hold-state naive baseline at the cumulative 1-to-50-step horizon (`0.73x`), the clearest open weakness in this benchmark. The upstream metric concatenates the three Melon runs across recording boundaries; Glassbox's boundary-safe protocol is a deliberate deviation, so published comparisons are near-comparable rather than exact.
