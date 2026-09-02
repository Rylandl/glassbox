# EPFL TOPOPlane2 Conventional Fixed Wing

**What this establishes:** a reproducible same-flight characterization of the structured and structured-residual models on a second conventional fixed-wing configuration. The evaluator records that training and validation share a source flight and refuses to mark either model as promotable, so this is characterization evidence, not a promotion benchmark.

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

EPFL's pinned [TOPOPlane2 navigation flight](https://zenodo.org/records/10337559) adds a second conventional configuration and a third fixed-wing dataset family, used to characterize model behavior on a genuinely different aircraft and sensor suite rather than to promote a default.

## Data

The 5 Hz fused state is paired with GNSS-tagged autopilot outputs for aileron, elevator, throttle, and rudder, plus a measured pitot-airspeed channel. The adapter converts the upstream NED/FRD state to NWU/FLU, decodes the source's scalar-first quaternion storage, maps PWM to typed normalized actuator outputs, and converts WGS84 coordinates to a segment-local tangent frame.

The published flight intentionally exercises navigation outages. Glassbox does not mistake the resulting INS position drift for vehicle motion: it compares the fused altitude against the independent barometric-altitude signal, keeps only the dominant navigation-consistent mode, and removes two seconds around every boundary. The pinned extraction retains seven segments and about 791 seconds of dynamics-grade flight. Because the upstream fused message leaves angular velocity at zero, body rates are reconstructed from the attitude derivative at the native 5 Hz rate and this limitation is recorded explicitly.

All segments come from one flight, so no train/validation partition can establish independent-flight generalization. The adapter labels them accordingly and keeps one source-group identity across every segment.

## Reproduce

```bash
uv run glassbox epfl prepare artifacts/epfl_topoplane
```

Fit the two maintained model classes on the canonical segments, using one consistent artifact directory:

```bash
uv run glassbox fit \
  artifacts/epfl_topoplane/canonical/*.npz \
  --evaluation-horizons 0.2,0.5,1,2 \
  --training-horizons 0.2,1,2 \
  --holdout-count 2 \
  --model artifacts/epfl_topoplane/structured_model.json \
  --report artifacts/epfl_topoplane/structured_report.json

uv run glassbox fit \
  artifacts/epfl_topoplane/canonical/*.npz \
  --evaluation-horizons 0.2,0.5,1,2 \
  --training-horizons 0.2,1,2 \
  --holdout-count 2 \
  --model-class structured_residual \
  --model artifacts/epfl_topoplane/residual_model.json \
  --report artifacts/epfl_topoplane/residual_report.json
```

Combine their reports into a same-flight characterization:

```bash
uv run glassbox epfl evaluate \
  --structured-report artifacts/epfl_topoplane/structured_report.json \
  --residual-report artifacts/epfl_topoplane/residual_report.json \
  --output artifacts/epfl_topoplane/characterization_report.json
```

The evaluator scores the chronological final segments against kinematic persistence, records that training and validation share a source flight, and refuses to mark either model as promotable.

## Results

`characterization_report.json` records each model's score against kinematic persistence and its horizon-rollout metrics on the two held-out chronological segments. Because the two fits share one source flight, the report's `can_promote_model` field is always false regardless of the scores it records.

## Boundary

This is characterization evidence, not a promotion benchmark. All segments come from one flight, so no train/validation partition can establish independent-flight generalization; the evaluator enforces this by refusing to mark either model as promotable. Body rates are reconstructed from the attitude derivative at 5 Hz rather than measured directly, a documented adapter limitation.
