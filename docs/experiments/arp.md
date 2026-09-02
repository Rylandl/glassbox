# ARP Quadrotor System Identification

**What this establishes:** on ARP Laboratory's four-log real-multirotor PX4 dataset, a learned rotational-response candidate improves the fitted reference's equal-horizon, equal-metric score on a protected held-out log by 11.49%, but the result remains `improves_reference_only`: it is 34.08% worse than kinematic persistence overall and still crosses the configured divergence threshold. A separate command-offset candidate shows dataset evidence for a more expressive force law but is not promoted because the protected log is already spent.

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

ARP Laboratory's four large-quadrotor ULogs are the first real PX4 multirotor references beyond synthetic and NanoDrone data. Glassbox uses logs 63-65 for all rotational-structure and authority development decisions, then evaluates the selected candidate once on protected log 66.

## Data

`glassbox ulog prepare-arp` downloads the 58.4 MB snapshot released with ARP Laboratory's [data-driven system-identification work](https://github.com/arplaboratory/data-driven-system-identification) from a pinned upstream commit, verifies every SHA-256 checksum, and writes four 50 Hz canonical trajectories under `artifacts/arp_reference/canonical/`. The adapter derives the canonical motor order from each log's PX4 control-allocation geometry; it does not carry a dataset-specific motor-order override. It retains the longest sustained powered interval in each recording, yielding four trajectories totaling 215.98 seconds (28.08, 53.88, 57.64, and 76.38 seconds). Each recording receives a stable, path-independent `source_group`, so leave-one-recording-out evaluation cannot accidentally mix segments from the same flight.

These recordings omit the usual arming and land-detection streams, and their local-position origin is not the takeoff point. The reference adapter therefore uses telemetry completeness followed by the powered-interval selection instead of armed, landed, or local-height gates. Operational ULog ingestion keeps the normal armed, airborne, and 0.2 m height defaults. The four recordings are replicates from one vehicle, so they are useful real-ULog integration and system-identification references, not four independent airframes.

## Reproduce

```bash
uv run glassbox ulog prepare-arp artifacts/arp_reference
```

The rotational-structure and command-offset candidates below were compared with maintainer-owned evaluation code rather than a public CLI flag: both are airframe-neutral selection mechanisms with fixed candidate policies, not additional end-user fitting knobs, so this page has no further command to reproduce them beyond the prepared corpus.

## Results

### Rotational-structure candidate

Glassbox used logs 63-65 for all rotational-structure and authority decisions, then evaluated the selected candidate once on protected log 66. On the development folds, a learned latent rotational response with cross-axis control coupling improved the instantaneous diagonal reference by 4.93%; the shared train-only authority selector then chose 0.75 for a further 4.62% improvement.

On protected log 66, the combined candidate improved the fitted reference's equal-horizon, equal-metric geometric score by 11.49%, with its worst individual metric changing by only +0.07%. It reduced the aggregate rotational score by 17.91% and delayed the first configured complete-rollout divergence threshold from 1.06 to 2.00 seconds. However, it remained 34.08% worse than kinematic persistence overall and still crossed the divergence threshold. At 0.5 seconds, candidate position and velocity errors were 1.63x and 1.68x persistence; at one second they were 2.03x and 2.39x. The promotion result is therefore `improves_reference_only`, not a promoted default or a complete-flight claim.

### Command-offset candidate

The next development-only comparison added one shared bounded command offset to the v3 multirotor force law. The zero-offset map remains the normal fitting default, and the experimental parameter is available only through maintainer evaluation code; there is no new CLI knob. Control semantics enforce the boundary: normalized motor commands may fit the offset, while measured squared-rotor-speed thrust proxies must use the identity map.

On leave-one-recording-out folds over logs 63-65, the offset alone was rejected: it improved the aggregate score by only 0.40% and its worst individual metric regressed 79.0%. Jointly scoring the fixed authority grid against the original zero-offset reference instead selected offset plus 0.5 angular authority. That composite improved the reference by 6.27%, with a 19.32% worst individual regression, and beat kinematic persistence by 27.85% geometrically; its worst persistence cell was 4.19% higher. The three held-out fits learned consistent offsets of -0.111, -0.105, and -0.072 normalized command.

## Boundary

The rotational structure transfers as a useful hypothesis, while normalized motor-command-to-translational-acceleration modeling is the clearest multirotor limitation to address on development data.

The command-offset result is useful development evidence for a more expressive force law, but it is not promoted: log 66 was already consumed by the preceding one-shot evaluation, so an untouched second normalized-command airframe is required for a valid promotion decision.
