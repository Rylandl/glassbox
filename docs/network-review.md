# Network structure after the accumulator

The next priority is architectural efficiency: preserve useful dynamics with
fewer independent coefficients and less arithmetic, and measure how much fitting
work achieves a given accuracy. The user prefers this to prioritizing additional
parallelization. Parameter count alone was a poor predictor of update cost: replacing
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

**Next: compress only the nonlinear head's temporal inputs.**
The nonlinear input projection accounts for 6,240 of the quad's 8,714 parameters.
Test a compact temporal representation there while retaining all explicit lag
inputs in the linear acceleration path. Keep the quadratic current-feature head,
32 nonlinear units, eight accumulators and shared physical mechanics unchanged.
This preserves arbitrary linear delayed response; nonlinear interactions among
discarded temporal components are the explicit capacity tradeoff.

The proposed first candidate uses `R = min(4, D)` fixed orthonormal summaries of
the 100 ms lag differences, shared across feature channels. For more than four
lags, constant through cubic temporal components are a concrete starting point;
for four or fewer, keep the original coordinates. Four is an engineering
hypothesis to test, not a known optimal memory dimension. The basis and its
normalization must be fixed before fitting. This is a finite-window projection,
not a new learned recurrence or an implementation of HiPPO. [HiPPO](https://arxiv.org/abs/2008.07669)
provides related motivation for polynomial history compression; its theoretical
and empirical results do not establish this candidate's flight performance.

For four commands and ten lags, the nonlinear input width becomes
`17 + 4*17 + 8 = 93`, down from 195. Its input weights fall from 6,240 to 2,976;
total learned parameters would fall **8,714 → 5,450 (37.46%)**. Forming the temporal
summaries costs arithmetic too: a dense count gives 680 multiplications for the
projection plus 2,176 for its nonlinear lag weights, versus 5,440 for the current
nonlinear lag projection. These are local operation counts, not whole-update
speed predictions. The three-command / two-lag configuration retains its input
dimension and 3,103 parameters. Dimensions follow sample interval, not family.

The old rank-two experiment compressed the linear, nonlinear and recurrent-memory
heads together and learned an unconstrained basis. It improved aggregate error
but lost angular accuracy and did not establish faster updates. It neither proves
that all compression fails nor isolates nonlinear history capacity. This narrower
candidate retains the linear path and accumulator architecture, uses more temporal
components where available, and avoids a freely learned basis/head rescaling
ambiguity. Those distinctions motivate the experiment; they do not guarantee it.

Control initialization and normalization explicitly. Keep the original full linear
and quadratic ridge initializer and current/memory projection scales; reducing a
matrix dimension must not strengthen random initialization again. Define temporal
summaries before their own normalization, with separate compensation for linear
and nonlinear input scales. Per-lag scaling cannot in general be pushed through
a truncated temporal projection. Preserve the same initial physical prediction
where possible and report any changed latent activation statistics. No fitting
or model implementation has been run for this proposal.

Freeze matched data, fitting budgets and aggregate weights before implementation.
Measure causal online error, held-out forecast and command-response errors, and
full update cost. Include model size, arithmetic and derivative work as explanations
for portability; measure learning progress against observations and solver work
so a slower or harder fit cannot masquerade as efficiency. Use existing quad and
fixed-wing evidence and report its known-configuration limits. A hardware-portability
claim ultimately needs measurements on another backend. Keep frozen losses visible
and make an overall adoption decision; broad controller reruns should answer a
specific remaining question rather than follow every minor regression.

**Secondary execution opportunity: parallelize the observed command filter.**
Its history still uses a sequential scan, although it is a linear exponential
filter with fixed coefficients during each prediction. A stable parallel prefix
or convolution can produce all filtered history values while retaining gradients
through the learned time constants. Preserve its real first-command initial
condition and small/large-time-constant behavior. This is another way to remove
sequential work without discarding information or adding a user option.

**Separate accuracy opportunity: improve solver conditioning.**
The earlier controlled experiment showed that 64 PCG iterations reduced
fixed-wing angular error by roughly 56–58% relative to 16 for both architectures.
The current preconditioner contains damping and the explicit prior diagonal; it
does not capture the forecast Jacobian's correlations. Adjacent lag features are
also likely correlated. A measured block or feature-Gram preconditioner is a
better-motivated next accuracy experiment than blindly widening or narrowing the
network. Its construction cost must count in the entire update.

## Keep the decision small

Work on one hypothesis at a time. The completed projection reuse was algebraic;
saved-revision comparisons were sufficient for that scope. Nonlinear history
compression changes the function class and needs matched learning experiments.
Keep improvements only when their practical benefit outweighs complexity; an
isolated benchmark loss is evidence to interpret, not an automatic veto.
