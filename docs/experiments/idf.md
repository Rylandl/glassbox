# IDF-DS Fixed-Wing Telemetry

**What this establishes:** on the platform-neutral structured residual, leave-one-source-group-out generalization across all 13 independent IDF-DS sessions reaches equal-session macro position/velocity/attitude/angular-velocity errors that improve on the preceding speed-scaled-damping model by 4.57% geometrically, with every fold and every aggregate horizon/metric improving. This is a strong model-selection result on one conventional airframe, not evidence of zero-shot parameter transfer.

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

The [IDF-DS fixed-wing telemetry release](https://zenodo.org/records/16992976) measures whether Glassbox's fixed-wing models generalize across independent real flight sessions of the same conventional airframe, using a leave-one-source-group-out benchmark.

## Data

`glassbox-ulog prepare-idf` is a substantial acquisition: the pinned Holybro archive is 2.12 GB. Glassbox verifies its published MD5, extracts only the 13 raw ULogs, verifies each member's size and CRC32, and converts every telemetry-complete interval to 50 Hz canonical trajectories. A 0.2-second gap tolerance accommodates the logged 10 Hz actuator streams; gaps are never bridged into a rollout, and only contiguous segments of at least 10 seconds are retained. The command also writes `artifacts/idf_reference/corpus_report.json` with retained-duration coverage, per-session segment counts, duration quantiles, motion ranges, control excitation, and the common typed trajectory contract.

The pinned extraction contains 119 contiguous trajectories from 13 independent sessions and retains 34,909.56 seconds (9.70 hours) of telemetry, per `corpus_report.json`. Session identity, rather than segment count or duration, determines both training weight and held-out folds.

The archive's 120 published "flights" are processed circuit laps, not 120 raw ULogs. The 13 available ULogs are long sessions containing the source telemetry for laps 1-116. Processed laps 117-120 refer to another source log that is not included in the archive, and this limitation is recorded in provenance. All available sessions use the same conventional fixed-wing configuration: one motor, split ailerons, elevator, and rudder. The adapter reconstructs `throttle, aileron, elevator, rudder` from the logged PX4 allocation instead of hard-coding channel indices.

## Reproduce

```bash
uv run glassbox-ulog prepare-idf artifacts/idf_reference
```

Measure generalization across recording sessions with the opinionated leave-one-source-group-out benchmark:

```bash
uv run glassbox-source-benchmark \
  artifacts/idf_reference/canonical/*.npz \
  --output-dir artifacts/idf_reference/source_benchmark_structured
```

Every dropout-separated segment from the held-out ULog moves into the same fold. Training uses the automatic deterministic window budget, and the summary reports equal-session macro metrics plus median, 90th-percentile, and worst-fold errors. The default policy evaluates every source group using the standard 0.1, 0.5, and 2-second training horizons; it intentionally exposes no dataset-specific sampling or optimizer knobs. Dataset inputs are SHA-256 recorded, and each completed fold is resumable only when that exact request still matches.

Use `--model-class structured_residual` with a separate output directory to compare the generic residual under exactly the same folds:

```bash
uv run glassbox-source-benchmark \
  artifacts/idf_reference/canonical/*.npz \
  --model-class structured_residual \
  --output-dir artifacts/idf_reference/source_benchmark_structured_residual
```

## Results

The best fixed-wing result uses the platform-neutral residual around the v3 structured model. Its equal-session macro errors are:

| Horizon | Position [m] | Velocity [m/s] | Attitude [deg] | Angular velocity [rad/s] |
| ---: | ---: | ---: | ---: | ---: |
| 0.1 s | 0.0255 | 0.511 | 0.786 | 0.159 |
| 0.5 s | 0.199 | 0.721 | 4.54 | 0.174 |
| 1.0 s | 0.408 | 0.785 | 7.16 | 0.159 |
| 2.0 s | 0.951 | 1.10 | 10.62 | 0.152 |

At two seconds the p90 held-out-session errors are 1.19 m, 1.33 m/s, 12.63 degrees, and 0.174 rad/s. Relative to the preceding speed-scaled-damping model, learned lateral surface cross-coupling improves the geometric aggregate by 4.57%; all 13 fold scores improve and every aggregate horizon/metric improves. The learned aileron-to-yaw coefficient is also repeatable across all folds.

## Boundary

These are strong model-selection results on one conventional airframe, not evidence of zero-shot parameter transfer. The result feeds the [cross-airframe fixed-wing development gate](fixedwing-gate.md) rather than standing alone as a performance claim.
