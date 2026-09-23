# Dissipative angular-rate memory in the online readout

The fast prompt/delayed command head identified local control directions but
still accumulated substantial angular error over 250 ms. This iteration added
one generic state-dependent term to its angular acceleration readout:

`omega_dot = b + B0 u + B1 a - d omega - k (omega - z)`,
`z_dot = (omega - z) / 0.1 s`, with `d, k >= 0` per rate axis.

Here `u` is the issued command, `a` is the existing leaky command state, and
`z` is a passive rate memory. The coefficients use episode-only least squares
on the latest 25 completed transitions; the nonnegative terms use a four-case
active-set solve per axis. The rate memory has a fixed physical 0.1 s time
constant and follows the predicted rate inside the JAX rigid-body rollout.
There is no vehicle label, actuator geometry, fleet pretraining or consumer
option. The force path retains the existing cold learned readout. This is still
an **experimental wrapper**, not the public `fit/predict/update` model.

The frozen [online benchmark](online-readout-benchmark.json) replayed 4,187
causal updates and 263 forecast origins across eight known recordings. The
comparison is the preceding independent prompt/delayed rate head, on the same
recorded origins and issued future commands. Errors below are physical 250 ms
RMSE, with rate in rad/s and velocity in m/s.

| Recording | Rate: prior → memory | Velocity: prior → memory |
| --- | ---: | ---: |
| fixedwing-80 | **0.486 → 0.509** | **0.632 → 0.686** |
| fixedwing-81 | **0.599 → 0.634** | **1.198 → 1.226** |
| quad-arm-115 | 1.165 → **0.841** | 0.269 → **0.256** |
| quad-arm-125 | 0.624 → **0.551** | 0.130 → **0.127** |
| quad-arm-135 | 6.770 → **6.238** | 0.671 → **0.597** |
| quad-change | 0.624 → **0.551** | 0.130 → **0.127** |
| paired-quad-fine | 0.004 → **0.003** | **0.042 → 0.043** |
| paired-quad-coarse | **0.001 → 0.003** | 0.007 → 0.007 |

`quad-change` largely duplicates arm 125 in these columns. Hard arm 135 has
only three origins and remains inaccurate despite the improvement. The paired
quad flights are gentle and their milliradian-per-second rate errors give little
evidence about hard recovery. Fixed-wing angular and velocity errors worsen
modestly and consistently. The
new rate errors still remain far below the older direct readout (2.593 and
6.027 rad/s for fixedwing-80/81, 2.171 for arm 125, 62.867 for arm 135).

The six exactly replayed arm-125 next-step command-response probes improve in
mean relative Jacobian error **0.379 → 0.331**. The new errors are 1.071,
0.238, 0.140, 0.166, 0.139 and 0.230. The first, underexcited probe is
unchanged at **1.071**; passive angular memory cannot create missing command
excitation. On the separately excited early episode, relative response error
improves 0.431 → 0.417 and its own 250 ms rate endpoint improves 4.385 →
4.119 rad/s. The original and excited branches end at different states, so
this is not a matched recovery comparison. The JAX analytic command Jacobian
matched the saved finite-difference response to maximum absolute error
`2.83e-14` on the excited branch and `2.06e-14` on the original branch.

Warm CPU update medians were about 0.79 ms on the fixed-wing recordings and
1.5 ms on quad. Cold forecast compilation took up to about 1.9 s and first
updates up to 0.48 s; neither cold start nor hardware timing is qualified.
The **command** memory still uses `8 × sample interval`, so the complete
model is not yet sampling-rate invariant. These are known configurations,
not held-out vehicles, controller trials or a public-model result.

The candidate was committed at `df30ae9` before the full fit. Cleanup into
one JAX prediction path and removal of obsolete screens was committed at
`c370f75`. Every saved forecast, one-step prediction and response array in
the cleaned full and paired runs matched the initial candidate exactly
(maximum absolute difference 0.0). The final saved artifacts are
`artifacts/readout-rate-memory-v1/rate-memory-lean-full`, manifest
`6bbd5b75fbe60ec95d8f8e4d3190173393e7b99f1d746e3e74de157dfa1a03da`,
and `artifacts/readout-rate-memory-v1/early-excitation-rate-memory-lean`,
manifest
`6e52a52825a7e736ac27b2fde9f85640b819a5ab4da62451de66c0552ec30c8c`.
Both copied packs passed their saved-data verifiers without fitting.

The engineering decision is to retain this as the experimental reference for
public integration. The structural improvement and fast warm updates outweigh
the measured small fixed-wing losses, but the public learner remains unchanged
until its fit, prediction, revision and derivative contracts are validated with
this same formulation.
