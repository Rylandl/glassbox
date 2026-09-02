# Cross-Platform Fitting-Policy Selection

**What this establishes:** the maintained `maintained_v1` five-candidate sweep provisionally selected the structured model at 0.1/0.5/1.0-second horizons with endpoint weight 3 and stability regularization 0.01, a 2.54% equal-platform improvement over the explicit reference. It failed one-shot promotion against the protected NanoDrone Melon split (an 8.85% worse equal-group geometric score), so the reference configuration remains the promotion baseline.

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

Use one leave-profile-out matrix to choose model class, training horizons, and long-rollout loss settings without tuning on the NanoDrone benchmark test set.

## Data

Telemetry uses a 13-element observed state vector:

```text
[position_xyz, velocity_xyz, quaternion_wxyz, angular_velocity_xyz]
```

Controls are named and ordered by the selected model family. Multirotor uses four normalized motor commands ordered front-left, front-right, rear-right, rear-left. Fixed-wing consumes typed roles: throttle/roll/pitch are required, while yaw and flap are optional. The coordinate convention is world Z-up and body X-forward, Y-left, Z-up. During rollout each model carries a latent applied-control vector through a learned first-order time constant, making the complete simulated state Markovian without requiring actuator-position or motor-speed telemetry.

The CLI runs a maintained five-candidate plan rather than exposing optimizer, horizon, regularization, model-class, and decision-threshold knobs. It compares the structured and structured-residual models at short and extended horizons under the shared stable-rollout objective, using 400 optimizer steps per fold. The plan includes an explicit unregularized structured reference. It is named and serialized, so changing it is a reviewed Glassbox policy change rather than an undocumented per-user tuning choice. Use `--smoke` to validate a new corpus and the complete workflow with two candidates and one optimizer step; smoke results are not an accuracy comparison.

Each candidate is scored against the reference over position, velocity, attitude, and angular-rate RMSE. Metrics and evaluation horizons are geometric-mean aggregated within each held-out fold; folds are equally weighted within each platform; platforms are equally weighted globally. A corpus with at least two profiles uses profile holdouts. A single-profile corpus such as IDF-DS automatically uses its independent `source_group` sessions instead. Thus adding more logs, segments, or profiles for one vehicle family cannot give it more influence in selection.

A candidate is ineligible if any rollout is non-finite, any individual metric regresses by more than 50%, or an apparent aggregate improvement hides more than a 5% platform-level regression. Small metric floors prevent ratios near numerical zero from dominating the decision. A non-reference candidate must also improve the overall score by at least 1%; otherwise the explicit reference wins the near-tie. Trajectories labeled with any `benchmark_split` other than `train` are rejected before fitting, so the public NanoDrone test flights cannot enter policy selection accidentally.

## Reproduce

```bash
uv run glassbox-select-policy \
  --dataset 'nanodrone=artifacts/nanodrone/canonical/train/*.npz' \
  --dataset 'fixedwing=artifacts/fixedwing/synthetic_v1/*.npz' \
  --output-dir artifacts/policy_selection
```

Every fold writes its model, fit report, benchmark summary, and an exact request record containing input SHA-256 hashes. Re-running the command resumes only matching completed folds. The final `selection.json` records all candidates, rejections, ranking, and the selected shared configuration.

## Results

The `maintained_v1` development sweep used all 12 NanoDrone training logs and six synthetic fixed-wing flights. It provisionally selected the structured model trained at 0.1, 0.5, and 1.0 seconds with endpoint weight 3 and stability regularization 0.01. Its equal-platform score improved 2.54% over the reference: NanoDrone improved 6.20%, while fixed-wing regressed 1.27%. Both residual candidates were rejected for individual metric regressions above 50%; the short residual also concentrated its gains in fixed-wing while regressing NanoDrone.

The selected policy was then fitted on all NanoDrone training logs and evaluated once against the protected Melon test split, alongside an exact fit of the reference. It failed promotion: cumulative position and velocity error were 30.4% and 39.4% worse, while attitude and angular-rate error were 11.1% and 13.1% better. The equal-group geometric score was 8.85% worse. The complete development decision and test reports are under `artifacts/policy_selection_v1/` and `artifacts/nanodrone/`.

## Boundary

This is explicitly a provisional model-development result: the selector never reads external test data, and promotion to a maintained default requires a separate untouched-data validation gate. The reference is therefore retained as the promotion baseline, and `maintained_v1` should not be re-tuned against the Melon result.
