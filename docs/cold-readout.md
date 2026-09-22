# Cold-start readout screen

Completed 2026-09-22. **Reject the raw recursive least-squares candidate; keep
the production temporal learner unchanged.** The experiment exposes a large
speed opportunity, but the proposed measurement-only estimator does not preserve
recursive forecast accuracy. It does not justify pretraining.

## Small experiment, fresh initialization

Protocol `fdfc5c9`, implementation and verifier `4970559`. Six known recordings,
first 64 updates each (62 on the truncated tape): **382 updates per arm**.
Both arms initialize afresh from exactly the same episode prefix; initial
parameters and normalizers match exactly. No previously fitted model, fleet
representation, learned class prior or cross-episode state is loaded.

The candidate freezes nonlinear features, command filters, accumulator parameters
and normalization. It adapts every acceleration output path: 381 features and
2,286 readout weights for four commands / 10 ms. It fits measured midpoint
velocity/gyro increments using float64 RLS, a prior centered on the current
prefix's ridge initializer, and precision `0.01 * prefix_transition_count * I`.
The nonlinear physical forecast equations remain unchanged. No forgetting,
curvature penalty, rollout acceptance guard or outcome-driven tuning is added.
This changes both the trainable parameter subset and fitting objective; it is
not a clean ablation of frozen features alone.

The baseline is the current full-model 16-PCG update. One-step predictions are
saved before assimilating each target. Every 16 updates, both frozen current
models also forecast 50/250 ms with the actual future command sequence. Those
are retrospective conditional forecasts, not an online-available command plan.
The prefix-frozen predictor is retained as a separate diagnostic.

## Results

Primary error is the geometric mean of velocity-vector and body-rate-vector RMSE
ratios, equal case weights within family then equal family weights. Lower is
better. All per-case physical scores, orientations, early/late errors and timing
samples are preserved in the [index](cold-readout.json).

| Measure | Quad | Fixed wing |
| --- | ---: | ---: |
| One-step primary error, candidate / baseline | **0.196** (80.4% lower) | **1.993** (99.3% higher) |
| 250 ms primary error, candidate / baseline | **1.951** | **278.161** |
| Warm complete update median | **0.36–0.41 ms** | **0.264–0.266 ms** |
| Baseline complete update median | **34.1–36.1 ms** | **7.71–7.74 ms** |
| Median update ratio | **0.0110** (~91× faster) | **0.0343** (~29× faster) |

Equal-family one-step error falls 37.5%, but 250 ms error is **23.3× higher**.
The speed and aggregate one-step diagnostic flags pass; the forecast and
per-family flags fail. These are substantial consistent forecast losses, not an
isolated veto. All six cases completed with finite values; finite predictions
can still be physically useless. For fixedwing-80, the candidate's 250 ms errors
are 450 m/s and 2,445 rad/s versus baseline 3.48 m/s and 4.84 rad/s. For
fixedwing-81 they are 2,531 m/s and 6,908 rad/s versus 7.07 m/s and 27.05 rad/s.
The baseline itself remains poor on some longer forecasts.

Updates were measured in one process, alternating arm order, with synchronization
and model snapshot costs included. Warm timings omit only each case's first
update. First-use update times for the first quad were 2.758 s baseline and
0.287 s candidate; for the first fixed wing, 2.137 s and 0.234 s. All samples
remain saved. Shared compilation caches mean these are not independent cold
process startup comparisons. No other backend or live deadline is qualified.

## What this says about the proposed architecture

A saved-array check evaluates the exact candidate quadratic objective
`||Phi M - Y||² + lambda ||M - M0||²` after the screen, without refitting.
It falls to between 0.00000029 and 0.0000416 of its initial value across cases,
while every saved inverse-precision matrix remains positive definite. The
estimator can fit observed increments extremely well while producing unusable
recursive trajectories. These retrospective objective values are not held-out
accuracy, and positive definiteness does not establish good conditioning.

This supports testing the objective/safeguards before declaring the random
feature representation inadequate. It does not isolate the cause: missing
curvature regularization, unconstrained readout changes, measured-midpoint
quadrature and frozen normalization/features all differ from the baseline.
The improved quad one-step forecasts also argue against dismissing the entire
fast-readout direction on this screen.

**Next bounded experiment:** keep the existing recursive forecast loss,
reconditioning, curvature prior and proposal acceptance; restrict optimization
to the same linear/quadratic/bias/nonlinear-output weights. Compare with the full
learner on the same short causal roster. This isolates which weights need to
learn before spending time designing a new estimator. Fresh episode
initialization remains mandatory. If promising, investigate accelerating that
trajectory-aware update; no pretrained core or catalog is introduced.

## Startup and limits

The original Throw code explicitly tumbles unpowered for one second, then
identifies while controlling (`glassbox_throw/throw.py`, `run_crazyflow_throw_trial`,
source recorded by the collection protocol). Our quad screen uses tape rows
50:125 for initialization and begins scoring at **1.25 s after release**. It has
75 context/initialization transitions, including 25 with actuation; no model
prediction is credited before that point. Fixed-wing scoring starts after
15 transitions / **0.75 s**. Prefix acquisition time and initialization/compute
costs are separate recorded quantities, not free pretraining.

This screens the fitting method, not earlier model availability, a successful
catch or live scheduling. The quad-change case duplicates quad-arm-125 before
its later configuration change, which this short screen never reaches. These
are known recordings with the old behavior controller; no new blind
configuration, counterfactual truth, noise robustness, uncertainty coverage or
controller trial was added. The RLS matrix is not presented as calibrated error
evidence. The current public model and archive formats are unchanged.

## Reproducibility

Five analytic tests pass: reconstruction of all acceleration paths at two input
shapes, recursive estimator agreement with an independent batch ridge solution,
and frozen-feature/duplicate-row checks at both sample intervals. Initial fixture
failures expected the wrong exception class and were corrected before the screen.
No scientific run was repeated or retuned. The saved-data verifier authenticates
tapes/payloads, checks row and truth identities, frozen arrays and recomputes all
physical scores. Source and runtime bindings are in the sealed pack.

The unsuccessful estimator and its experiment-only tests are removed from the
maintained tree; source `4970559` preserves them. In that checkout, audit without
fitting:

```bash
PYTHONPATH=src:scripts SCIPY_ARRAY_API=1 python scripts/screen_cold_readout.py verify \
  --output /Users/ryland/autonomy/glassbox/artifacts/cold-readout-v1/screen \
  --manifest-sha256 d340ce722387bf07f8786aa1f36962d059bc51c08f4e96583b190f9526cd2f85
```
