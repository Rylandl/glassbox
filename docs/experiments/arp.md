# ARP Quadrotor System Identification

**What this establishes:** ARP Laboratory's four-log real-multirotor PX4 dataset ingests through the public corpus registry into four canonical 50 Hz trajectories with a per-recording source group. A command-offset candidate measured on it shows dataset evidence for a more expressive force law but is not promoted because the protected log is already spent. The rotational-response candidate this corpus was also used to select failed promotion and its branch has been deleted; the finding is recorded in the [literature review](../literature-review.md).

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

ARP Laboratory's four large-quadrotor ULogs are the first real PX4 multirotor references beyond synthetic and NanoDrone data. Glassbox uses logs 63-65 for all development decisions, then evaluates a selected candidate once on protected log 66.

## Data

`glassbox corpus prepare arp` downloads the 58.4 MB snapshot released with ARP Laboratory's [data-driven system-identification work](https://github.com/arplaboratory/data-driven-system-identification) from a pinned upstream commit, verifies every SHA-256 checksum, and writes four 50 Hz canonical trajectories under `artifacts/arp_reference/canonical/`. The adapter derives the canonical motor order from each log's PX4 control-allocation geometry; it does not carry a dataset-specific motor-order override. It retains the longest sustained powered interval in each recording, yielding four trajectories totaling 215.98 seconds (28.08, 53.88, 57.64, and 76.38 seconds). Each recording receives a stable, path-independent `source_group`, so leave-one-recording-out evaluation cannot accidentally mix segments from the same flight.

These recordings omit the usual arming and land-detection streams, and their local-position origin is not the takeoff point. The reference adapter therefore uses telemetry completeness followed by the powered-interval selection instead of armed, landed, or local-height gates. Operational ULog ingestion keeps the normal armed, airborne, and 0.2 m height defaults. The four recordings are replicates from one vehicle, so they are useful real-ULog integration and system-identification references, not four independent airframes.

## Reproduce

```bash
uv run glassbox corpus prepare arp artifacts/arp_reference
```

The command-offset candidate below was compared with maintainer-owned evaluation code rather than a public CLI flag: it is an airframe-neutral selection mechanism with a fixed candidate policy, not an additional end-user fitting knob, so this page has no further command to reproduce it beyond the prepared corpus.

## Results

### Command-offset candidate

A development-only comparison added one shared bounded command offset to the structured multirotor force law. The zero-offset map remains the normal fitting default, and the experimental parameter is available only through maintainer evaluation code; there is no new CLI knob. Control semantics enforce the boundary: normalized motor commands may fit the offset, while measured squared-rotor-speed thrust proxies must use the identity map.

On leave-one-recording-out folds over logs 63-65, the offset alone was rejected: it improved the aggregate score by only 0.40% and its worst individual metric regressed 79.0%. The composite of the offset with a fixed angular authority was scored against the same reference, but the angular-authority selection sweep was retired at this commit and that composite's numbers are withdrawn because they had no recorded artifact. The three held-out fits learned consistent offsets of -0.111, -0.105, and -0.072 normalized command.

## Boundary

Normalized motor-command-to-translational-acceleration modeling is the clearest multirotor limitation this corpus exposes on development data.

The command-offset result is useful development evidence for a more expressive force law, but it is not promoted: log 66 was already consumed by the one-shot rotational-structure evaluation recorded in the literature review, so an untouched second normalized-command airframe is required for a valid promotion decision.
