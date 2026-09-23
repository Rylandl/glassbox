# Prompt and delayed command response in the online rate head

The frozen [v2 online benchmark](online-readout-benchmark.json) evaluates factual
forecasts on eight known recordings and the six exactly replayed arm-125
command-response probes. A single generic angular readout now fits a current
command effect, a leaky command-state effect and nonnegative diagonal rate
damping from the latest 15–25 completed transitions. It uses no vehicle family,
actuator metadata, previous flight or pretrained weights. The leaky state has
one fixed time constant, `8 × recording interval`; the coefficients are fitted
from each episode. Body force still comes from the cold learned readout. One
JAX scan advances force, rate, attitude and both memories together. This is an
**experimental screen**, not the public `fit/predict/update` learner.

| Recording | Rate, prior → new (rad/s) | Velocity, prior → new (m/s) |
| --- | ---: | ---: |
| fixedwing-80 | **0.403 → 0.486** | 0.610 → 0.632 |
| fixedwing-81 | **0.578 → 0.599** | 0.768 → 1.198 |
| quad-arm-115 | **2.316 → 1.165** | 0.241 → 0.269 |
| quad-arm-125 | **1.237 → 0.624** | 0.270 → 0.130 |
| quad-arm-135 | **9.001 → 6.770** | 0.642 → 0.671 |
| quad-change | **1.238 → 0.624** | 0.270 → 0.130 |
| paired-quad-fine | **0.002 → 0.004** | 0.044 → 0.042 |
| paired-quad-coarse | **0.003 → 0.001** | 0.010 → 0.007 |

These are physical 250 ms RMSE over the same 263 forecast origins, compared
with the earlier fixed-lag positive-damping hybrid. `quad-change` largely
duplicates arm 125 in these columns, and hard arm 135 has only three origins.
The fixed-wing body-rate losses are small but real; fixed-wing-81 velocity
worsens materially. Against the older direct readout rather than the prior
hybrid, new rate RMSE is 0.486 versus 2.593 on fixedwing-80, 0.599 versus 6.027
on fixedwing-81, 0.624 versus 2.171 on arm 125, and 6.770 versus 62.867 on
arm 135. The resulting 250 ms body-rate recursion is much better than that
direct baseline on every known case.

Arm-125 mean counterfactual next-step rate-response error falls from **0.629
to 0.379** relative to the plant Jacobian. The six new errors are 1.071,
0.244, 0.301, 0.192, 0.173 and 0.293. The first probe **worsens** from 0.808
to 1.071, exactly where early Throw identification matters. No fixed-wing
counterfactual plant response is available. JAX differentiation of the first
arm-125 response matched the benchmark's finite difference to `2.06e-14`
maximum absolute error. Warm CPU updates measured about 0.6–0.8 ms on fixed
wing and 1.4–1.7 ms on quad; cold initialization plus compilation took roughly
2 s, and a first update could take 0.4–0.6 s. No hardware timing or controller
trial has qualified this candidate.

The basis time constant is **sampling-interval dependent**, not an identified
physical lag. A 0.10 s physical-time variant in the frozen smoke worsened the
first arm-125 rate endpoint from 3.313 to 5.828 rad/s. A generic two-scale
physical-time rate-only variant also diverged on hard arm 135. These screens
were discarded. Good results for the gentle paired quad at two sample rates
do not establish sampling-rate invariance. The generic prompt/delayed
decomposition is worth keeping as a structural hypothesis, but it needs a
single causal physical-time treatment before public adoption.

The final experimental implementation is `fe2681a`, following `ed57014`
(prompt/delayed readout) and `c522c73` (joint JAX rollout). Obsolete Python
prediction code was deleted. The final full saved artifact is
`artifacts/readout-distributed-lag-v1/distributed-lag-jax-lean-full`, manifest
`e33e90737f83276539dbfdd39ca7fc2ef020400a7e6b17447a50b3a535be167f`.
The frozen runner verified its saved forecasts, responses, controls, scores and
table without fitting. Refactoring from the earlier JAX pack changed every
saved forecast, one-step and response array by exactly 0.0.
