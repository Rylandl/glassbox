# Public angular-rate memory result

The single public dynamics model now uses a learned three-axis force readout and
an episode-fitted angular readout. Its angular equation is

`omega_dot = b + B0 u + B1 a - d omega - k (omega - z)`,

where `u` is the issued command, `a` is a passive command state with a fixed
0.08 s time constant, `z` is a passive angular-rate state with a fixed 0.1 s
time constant, and `d, k >= 0` on each axis. The same physical rollout serves
offline `fit/predict/update` and `OnlineFit`; there is no vehicle-family branch,
pretrained prior, metadata requirement or consumer option. Online rate
coefficients use the latest 25 completed transitions and the exact command/rate
memory entering that window. The force readout uses causal closed-form updates
with fixed regularization and bounded replay. Forecasts reconstruct both passive
states from the caller's observed history, so saved model snapshots remain
self-contained.

The [frozen online benchmark](online-readout-benchmark.json) covers 4,187 causal
updates and 263 forecast origins across eight recorded cases. The measured
candidate is the *public* `OnlineFit` and its `model.predict` rollout, exposed
to the frozen runner through a thin adapter. Errors below are 250 ms physical
RMSE. The earlier experimental rate-memory wrapper is a useful architecture
reference, but its fitter and command-time interpretation differed. The
benchmark's `direct` and `full` columns are saved predictions from older
learners, not a fresh matched retraining of those methods.

| Recording | Origins | Rate, rad/s | Velocity, m/s | Earlier rate-memory rate, rad/s |
| --- | ---: | ---: | ---: | ---: |
| fixedwing-80 | 14 | 0.345 | 0.506 | 0.509 |
| fixedwing-81 | 14 | 0.597 | 0.751 | 0.634 |
| quad-arm-115 | 54 | 0.841 | 0.257 | 0.841 |
| quad-arm-125 | 54 | 0.551 | 0.124 | 0.551 |
| quad-arm-135 | 3 | 6.237 | 0.596 | 6.238 |
| quad-change | 54 | 0.551 | 0.124 | 0.551 |
| paired-quad-fine | 35 | 0.003 | 0.033 | 0.003 |
| paired-quad-coarse | 35 | 0.004 | 0.006 | 0.003 |

The fixed-wing rate improvements came from replacing the experimental `8 × dt`
command lag with a physical 0.08 s lag and carrying command/rate memory through
public history. Their velocity errors also fell from 0.686/1.226 to
0.506/0.751 m/s. The quad rate results were essentially unchanged. The hard
arm-135 case remains inaccurate and has only three forecast origins;
`quad-change` largely duplicates arm-125. The paired flights are gentle and do
not establish recovery accuracy. The first underexcited arm-125 command-response
probe remains at **1.071 relative error**. Its six-probe mean is 0.331. A
separately excited branch reaches 0.417 relative response error and 4.119 rad/s
at its own 250 ms endpoint, but the branches end at different states.

The new model has 2,512 parameters with three commands at 50 ms, versus 3,103
in the preceding model, and 4,340 with four commands at 10 ms, versus 5,450.
Warm measured CPU updates were about 0.92 ms on the fixed-wing recordings and
1.7 ms on quads. This is a large improvement against the former public quad
median of 33.9 ms, though those runs were not contemporaneous latency trials.
Cold JAX forecast compilation took about 1.9–2.2 s and first updates about
0.36–0.46 s in the frozen run. Thus fresh-start real-time fitting is **not yet
qualified**.

Focused lifecycle, derivative, persistence and invalid-input tests pass. The
full suite passes 385 tests, and Ruff passes. Fixed-wing and quad fitted public
snapshots matched their pre-integration experimental JAX forecasts to at most
floating-point roundoff. After source cleanup, every saved one-step forecast,
multi-step forecast and response array matched the committed public candidate
exactly (maximum absolute difference 0.0). Both final packs pass saved-data
verification without fitting:

```bash
PYTHONPATH=src:scripts python scripts/benchmark_online_readout.py verify \
  --output artifacts/readout-public-rate-v1/public-rate-lean-full \
  --manifest-sha256 dbf4a47f7892d1417ebe07b29d2439be5997197156c5e711746e4c07a208f5b5
PYTHONPATH=src:scripts python scripts/benchmark_early_excitation.py verify \
  --output artifacts/readout-public-rate-v1/early-excitation-public-final \
  --manifest-sha256 830c26f58e3f19431f02bab46e859021a05251808acf964db968af98161a4613
```

The candidate was committed before the full public run at `10199a8`; the lean
implementation and final benchmark were captured at `e01471d` (with an
import-only early-runner fix at `7e00583`). This is evidence for model fitting
and command-response accuracy on known recordings. There has been no new live
Throw controller trial, held-out configuration evaluation, independent coverage
qualification or matched offline `fit/update` accuracy comparison for this
revision.
