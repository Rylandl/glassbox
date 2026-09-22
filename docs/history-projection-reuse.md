# Reusing acceleration-head projections

**Adopted on 2026-09-22.** Reusing the unchanged history projection cuts whole
online-update median time by **19.17% for quads** and **4.85% for fixed wings** in
the focused snapshot comparison. The equal-family reduction is **12.30%**.
The model keeps every parameter, lag input, accumulator, integration stage and
fitting setting. Existing accumulator revisions and sessions remain loadable.

## Computation change

The linear and nonlinear acceleration heads share a combined input projection.
For each observation interval, compute that projection once at the initial
feature vector. Later midpoint stages add the current-feature difference times
the effective current weight, which subtracts the contributions of all lag
blocks. The history and incoming memory stay fixed during these stages.

Centering on the initial features retains the small `past - current` differences;
separately projecting large absolute history and current values would risk
cancellation. Gradients still pass through the anchor, history, parameters and
commands. This is an algebraic computation change, not a smaller model.

## Measurement and result

The [protocol](harness/history-projection-reuse-v1.json) was committed before
implementation. Both arms load identical saved online sessions at offsets zero
and 32 on six existing streams. Capturing those sessions exactly reproduced all
192 consumed baseline transitions and reports. The two `quad-change` contexts
match `quad-arm-125` before the configuration change; these are not twelve
independent conditions or evidence about later adaptation.

Each point has 25 retained timings per arm in five alternating exclusive blocks.
The timer includes public `observe`, immutable model snapshot and synchronization;
loading, saving, diagnostics and three warmup repetitions are outside it. First
calls are recorded separately. No other tests or fits ran during timing.

| Measure | Quad | Fixed wing |
| --- | ---: | ---: |
| Whole-update median reduction | **19.17%** | **4.85%** |
| Whole-update p95 reduction | **11.56%** | **5.04%** |
| Point median range, baseline | 39.69–40.01 ms | 8.18–8.34 ms |
| Point median range, candidate | 32.01–32.30 ms | 7.77–8.06 ms |

Family summaries are geometric means of point ratios, not pooled latency
quantiles. Every point's median improves. One quad point's p95 worsens from
40.73 to 59.83 ms; its candidate samples include a 133.14 ms outlier. All samples
remain included. We did not rerun or discard that tail. The frozen recommendation
of at least 5% equal-family median reduction, with neither family more than 5%
slower, passes. This is a CPU/arm64 snapshot benchmark, not real-time or streaming
trajectory qualification; 32 ms still exceeds the quad's 10 ms observation period.

## Numerical interpretation

For each of three saved fitted revisions, compare eight retained development
windows over the full 250 ms horizon, in float32 and float64. All 18 prediction,
parameter/command JVP and VJP comparisons pass the frozen tolerances. Maximum
absolute differences are 0.0000062 for float32 states and 0.0000134 for float32
derivatives; float64 differences are at most 1.6e-14. These are mathematical
equivalence checks on saved inputs, not new held-out accuracy results.

All twelve updates preserve acceptance counts, objective-evaluation counts,
selected step sizes and damping. Floating-point regrouping nevertheless produces
small differences after the truncated optimizer. Eight points fail the strict
parameter tolerance; four fail the post-update prediction tolerance. The combined
`numerical_checks_pass` flag therefore remains **false**. We did not loosen it.

The largest parameter difference is 0.0000101. The largest post-update prediction
differences are **0.0000126 m/s** in velocity, **0.0000963 rad/s** in angular rate,
and 0.000000358 in a rotation-matrix entry. All values are finite. These differences
are small relative to the residuals at those observations, with no changed step
decision. The practical speedup justifies adoption despite those strict flags and
the one tail regression. Long adaptive trajectories need not remain bitwise
identical; this experiment does not establish unchanged trajectories everywhere.

## Verification

The [evidence index](history-projection-reuse.json) records source commits,
artifact authorities, frozen outcomes and focused tests. The single completed
paired attempt contains 600 retained timings and 146 exclusive work intervals.
The saved-data verifier independently reproduced the complete result without
model calls or refitting:

```bash
env -u JAX_ENABLE_X64 PYTHONPATH=src:scripts SCIPY_ARRAY_API=1 uv run python \
  scripts/compare_projection.py verify \
  --output artifacts/history-projection-reuse-v1/comparison \
  --manifest-sha256 40d0ac7753cc7b16ea63b30484b0785fbbf47cc22f33bcbcf599db9a2cf0681b
```

Twenty-eight focused dynamics/accumulator tests and four online
gradient/conditioning/resume tests pass. The new analytic fixture checks small
lag differences, variable command/history dimensions and forward/reverse
derivatives in both precisions. Ruff passes. No offline fits, new Dart trials or
redundant full-suite run were needed for this computation change.

The next iteration is the observed command-filter history scan described in the
[network review](network-review.md). It is another linear recurrence whose full
history may be computed in parallel without removing information.

## Subsequent full-stream evidence

The fresh baseline in [nonlinear temporal qualification](nonlinear-temporal.md)
replayed all 3,137 causal updates under this refactor. Compared with the original
pre-projection accumulator, primary error is 2.69% higher and rate error 4.70%
higher, predominantly fixedwing-81 (primary +12.25%, rate +15.12%). This does not
change the frozen snapshot results above; it demonstrates their limited scope.
Tiny differences after individual updates can compound over repeated fitting.
The adopted speed/accuracy tradeoff remains explicit, and solver conditioning
is the next measured gap. The original 4.11% improvement versus v8 should not be
quoted as the refactored model's current full-stream accuracy.
