# Network structure after the accumulator

The next opportunity is to remove repeated computation before reducing model
capacity. Parameter count alone was a poor predictor of update cost: replacing
recurrent memory removed only 14.0% of quad parameters but reduced median online
update time by 47.6%. The same change improved equal-family online prediction
error by 4.1% on the six known streams.

This is a source and parameter inventory, not a fresh component timing profile.
The old v8 profile cannot identify the accumulator's remaining runtime shares.
The paired whole-update measurements remain the authority for its actual speed.

## What the model spends parameters on

Counts come from the authenticated initial accumulator sessions in
`artifacts/accumulator-migration-v1/online/candidate`. They depend on command count
and sample interval, not vehicle-family dispatch.

| Component | 4 commands, 10 ms samples | 3 commands, 50 ms samples |
| --- | ---: | ---: |
| Dense projection into 32 nonlinear units | 6,240 | 1,696 |
| Linear acceleration head | 1,170 | 318 |
| Quadratic current-feature head | 918 | 720 |
| Nonlinear output projection | 192 | 192 |
| Accumulator input projection | 136 | 120 |
| Biases and learned time constants | 58 | 57 |
| Total | **8,714** | **3,103** |

Each current feature vector contains body velocity, angular velocity, gravity
direction, issued commands and filtered commands: `C = 9 + 2U`. The acceleration
head also receives every sampled difference over the preceding 100 ms and eight
accumulator values. Its input size is `(D + 1)C + 8`, where `D` is the number of
100 ms delay samples. The model learns acceleration; shared gravity, rotations
and integration turn that into future motion.

## Recommended order

**Completed: reuse the unchanged history projection across integration stages.**
The linear and nonlinear input heads repeatedly multiply the concatenation of
current features, `past - current`, and memory. Algebraically this is

```
current @ (W_current - sum(W_lags))
    + sum(past_lag @ W_lag)
    + memory @ W_memory
```

with normalization folded into each weight block. The implementation uses the
equivalent centered expression `projection_at_start + (current - start) @
effective_current_weight` to preserve small lag differences. It retains every
lag, parameter and representable function. The [focused comparison](history-projection-reuse.md)
measured **19.17% lower quad** and **4.85% lower fixed-wing** whole-update medians.
All saved-model prediction/derivative comparisons pass. Small post-update
differences exceed strict tolerances, and one latency tail worsens; both remain
reported. The practical gain justified adoption.

**Next: apply the accumulator lesson to the observed command filter.**
Its history still uses a sequential scan, although it is a linear exponential
filter with fixed coefficients during each prediction. A stable parallel prefix
or convolution can produce all filtered history values while retaining gradients
through the learned time constants. Preserve its real first-command initial
condition and small/large-time-constant behavior. This is another way to remove
sequential work without discarding information or adding a user option.

**Then: improve solver conditioning before shrinking the heads.**
The earlier controlled experiment showed that 64 PCG iterations reduced
fixed-wing angular error by roughly 56–58% relative to 16 for both architectures.
The current preconditioner contains damping and the explicit prior diagonal; it
does not capture the forecast Jacobian's correlations. Adjacent lag features are
also likely correlated. A measured block or feature-Gram preconditioner is a
better-motivated next accuracy experiment than blindly widening or narrowing the
network. Its construction cost must count in the entire update.

A compact learned delay basis remains a capacity trade-off worth revisiting
later. The old rank-two experiment did not establish a worthwhile accuracy/speed
trade-off, and the accumulator has changed the bottleneck since then. Keep the
full lag representation until a smaller one offers a measured practical win.

## Keep the decision small

Continue with one function-preserving change. Use saved candidate revisions
and real saved update snapshots, check mathematical equivalence, and measure the
whole update. No offline refit or new Dart optimization is needed merely to time
an algebraic refactor that preserves predictions and derivatives within the
predeclared tolerance. Broaden evaluation when a measured discrepancy or actual
behavior change warrants it. Keep improvements only when their practical benefit
outweighs complexity; an isolated benchmark loss is evidence to interpret, not an
automatic veto.
