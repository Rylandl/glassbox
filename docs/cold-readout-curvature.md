# Fast readout with physical curvature regularization

Completed 2026-09-22. **A strong fast-fitting candidate, with a remaining
fixed-wing rollout gap.** Full recordings show 83.5% lower one-step primary
error and about 22× faster updates. Quad 250 ms forecasts improve; fixed-wing
250 ms forecasts degrade. Keep the current production learner while testing a
causal forecast acceptance check on this fast proposal.

## What changed

The earlier raw RLS estimator fit measured increments quickly but lost recursive
forecast accuracy. The matched recursive-objective experiment showed that frozen
features do not inevitably cause that failure. This iteration adds the current
learner's **physical quadratic-curvature penalty** to the fast increment fit.

Features, command filters, accumulators, normalization, measurement construction
and prefix ridge mean stay fixed. There is no pretrained core, fleet prior,
vehicle metadata, family branch, forgetting or recursive acceptance guard.
Every episode starts with a fresh `OnlineFit(prefix)` initialization.

The retained statistics are `G = lambda I + sum(phi phi.T)` and
`Q = lambda M0 + sum(phi y.T)`, where `lambda = 0.01 * prefix_initialization_count`.
At update n, solve `(G + diag(d_n)) M = Q` by equilibrated float64 Cholesky. One
matrix serves all six acceleration outputs. `d_n` is nonzero only on quadratic
features: `0.015*n*(Hessian_factor/quadratic_scale)^2`. Its domain comes from the
same measured bootstrap/recent cache as the current learner.

The factor n keeps curvature strength constant against **average** data error.
The protocol derives this coefficient from the existing `0.01/4` physical Hessian
penalty, including output and time scales. It was not selected by a sweep. The
unpenalized optimum remains the raw estimator's optimum. The numerical solver
changes to handle the varying penalty; analytic checks compare it with independent
augmented least squares. Only the readout learns after prefix initialization.

## Two frozen stages

1. **Short diagnostic:** 382 updates per arm on six recordings, comparing the full
   learner, raw RLS and the regularized candidate. Protocol `cba69a6`, source
   `be38729`. Both controls, including all final arrays, reproduce the previous
   raw-RLS screen exactly. Candidate measurement features/targets also match.
2. **Full recordings:** 3,137 updates per arm against the full learner, extending
   through the recorded configuration change. Protocol `57715cb`, source
   `6d2b9c4`. All five scientific candidate definitions are unchanged. Both arms'
   first 64 predictions (62 on the truncated tape), overlapping conditional
   forecasts, candidate solved weights, measurements and penalties reproduce the
   short screen exactly. No retuning or failed fitting run occurred.

The [index](cold-readout-curvature.json) links authenticated packs, complete
physical scores, initializations, work counts, every solve and timing sample.

## Results

Primary error combines velocity-vector and body-rate-vector RMSE geometrically,
equal cases within family then equal families. These are candidate/full-learner
ratios; lower is better.

| Stage and measure | Quad | Fixed wing | Equal-family aggregate |
| --- | ---: | ---: | ---: |
| Short: one-step error | 0.1838 | 0.2403 | **0.2102** |
| Short: 250 ms error | 0.7834 | 0.7155 | **0.7487** |
| Full: one-step error | 0.1769 | 0.1547 | **0.1654** |
| Full: 250 ms error | 0.6719 | 2.0600 | **1.1765** |
| Full: median update time | 0.0299 | 0.0702 | **0.0458** |

Full-run warm median updates are **1.006–1.035 ms for quads** and
**0.535–0.540 ms for fixed wings**, versus **33.9–34.2 ms / 7.65–7.67 ms**.
Candidate p95 is **1.13–1.23 ms / 0.618–0.637 ms**. These include cache/domain
construction, matrix statistics, factorization/solve, synchronization and model
snapshot. Arm order alternates; no other fitting or testing ran during timings.

| Full recording | One-step ratio | 250 ms ratio | One-step after 100 updates |
| --- | ---: | ---: | ---: |
| quad-arm-115 | 0.0906 | 0.7789 | 0.1964 |
| quad-arm-125 | 0.1732 | 0.6143 | 0.1232 |
| quad-arm-135 | 0.3601 | 0.6933 | unavailable: 62 updates total |
| quad-change | 0.1732 | 0.6143 | 0.1239 |
| fixedwing-80 | 0.2082 | 1.8829 | 0.1424 |
| fixedwing-81 | 0.1149 | 2.2536 | 0.0679 |

After the recorded change at four seconds, quad-change one-step primary error is
**40.2% lower**. Absolute velocity/rate RMSE is **0.000101 m/s / 0.000808 rad/s**,
versus **0.000172 / 0.001330**. This portion is already near hover with narrow
command variation; it is not evidence of recovery from a new violent disturbance.
The prefix is shared with quad-arm-125, so those cases are not independent before
the change.

## The unresolved forecast issue

The short-screen fixed-wing 250 ms ratio fell from raw RLS's **278.16×** to
**0.715×** the full learner. Curvature regularization addresses a major failure
without pretraining. That success does not persist across the complete recording:

| Full 250 ms RMSE | Full learner | Fast candidate |
| --- | ---: | ---: |
| fixedwing-80 velocity | 2.088 m/s | 1.328 m/s |
| fixedwing-80 body rate | 2.702 rad/s | **15.065 rad/s** |
| fixedwing-81 velocity | 4.157 m/s | **7.669 m/s** |
| fixedwing-81 body rate | 14.523 rad/s | **39.984 rad/s** |

A posthoc per-query view of the saved forecasts shows this is not one outlier:
after excluding the identical initialization forecasts, the candidate's combined
terminal error is worse on **20 of 26** fixed-wing query origins. The one-step
improvement persists after 100 updates even as recursive angular predictions
worsen. Low increment error still does not establish a good trajectory model.

All frozen aggregate flags pass, including aggregate 250 ms error below 1.25×.
The family flag checks one-step error only. Passing these flags does not erase
the consistent fixed-wing rollout loss. Equally, this loss does not justify
abandoning a method with dramatically better local accuracy and runtime.

**Next named gap: causal 250 ms forecast acceptance for fast readout proposals.**
Keep this estimator unchanged, then accept or damp its proposed weight change
using completed observed trajectories through 250 ms. Score predictions before
revealing targets and maintain the full causal data budget. That adds forward
rollouts without the expensive repeated trajectory derivatives. Its runtime,
rejection/staleness and retained accuracy need a separately frozen comparison;
it is a hypothesis, not an implemented solution. Do not merely reuse the current
50 ms guard: fixed-wing 50 ms forecasts contain only one observation interval.

## Evidence limits and reproducibility

Nine focused tests pass: feature reconstruction, raw-RLS batch equivalence,
regularized solves versus augmented least squares, equality to the existing
physical Hessian penalty, frozen arrays, causal row rejection and independent
measured-domain reconstruction at both sample intervals. All 3,137 saved candidate
solves pass the stronger componentwise normal-equation audit; maximum error is
**2.91e-15**. Final precision matrices are positive definite. Scores are recomputed
from saved predictions without fitting. This certifies the specified estimator,
not identified dynamics or calibrated uncertainty.

The stricter short-data audit initially completed its computations but hit
`FileExistsError` when writing the existing verification filename. The original
result remained intact; the new result is saved separately as
`verified-componentwise.json`. Its log is retained. No measurement run was repeated.

All initialization data and work remain counted. Quad scoring starts 1.25 s after
release, using 75 prefix transitions including 25 actuated initialization samples;
fixed-wing scoring starts at 0.75 s, with 15 prefix transitions and five
initialization windows. In the full run, first-use candidate construction costs
**0.394 s / 0.120 s** and its first update **0.236 s / 0.230 s** for the first quad
and fixed wing. Shared compilation caches mean these are not independent process
startup or live recovery measurements. Warm timing excludes only the first update
per case. No other hardware or live deadline is qualified.

Sample interval and family remain confounded (10/50 ms). The data use the existing
behavior controller and known configurations; no new closed-loop trial, blind
configuration, counterfactual response, observation-noise test or uncertainty
coverage is provided. Conditional forecasts use actual future command tapes only
for retrospective scoring, every 16 observations when truth is complete. The
truncated quad-arm-135 recording still has large absolute forecast errors in both
arms; it supplies no long-run evidence. Lower average error is not a catch-rate
or Dart precision result.

The production learner and public interfaces remain unchanged. The experimental
runner and tests are preserved in Git, not retained as a second implementation.
Audit the full pack in source `6d2b9c4` with:

```bash
PYTHONPATH=src:scripts SCIPY_ARRAY_API=1 python scripts/screen_cold_readout_curvature.py verify --full \
  --output /Users/ryland/autonomy/glassbox/artifacts/cold-readout-curvature-full-v1/evaluation \
  --manifest-sha256 adef6d7fe24fcab8c07bacc9f3868432fc1c3281a9fcf87c61926ad1f90032cb
```

The CLI preserves existing files rather than overwriting them. If `verified.json`
already exists, invoke `verify(output, authority)` directly in that checkout to
inspect the returned audit, or save it to a new filename.
