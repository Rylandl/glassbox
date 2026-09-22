# A compact rate head outperforms the general readout on known online flights

The [v2 fast benchmark](online-readout-benchmark.json) now measures a physically
matched 10 ms command-response Jacobian at six points of the arm-125 Crazyflow
recording. Its source is the exactly replayed, sealed plant probe from the
[early quad diagnosis](readout-early-quad.md). The candidate only receives the
same causal recording prefix and later observations as before. The runner
computes its response with ±0.01 command perturbations at the recorded state;
the plant Jacobian is used only to score saved predictions. Smoke includes two
of the six points; full includes all six. The v1 archive still verifies with
the updated runner. This is one command for factual forecasts, response,
runtime and saved-data verification.

The first structural screen split current command response from the nonlinear
state/history readout. It was linear in issued and filtered commands, with
velocity-dependent coefficients, while the frozen nonlinear features saw past
commands but not the current one. This lowered the arm-125 mean relative
response error from **0.934 to 0.777**. It also raised fixed-wing-81 250 ms
body-rate RMSE from 0.650 to 1.900 rad/s and the gentle paired quad from 0.007
to 0.021. Adding a state-conditioned linearization of the nonlinear control
features did not repair the early fixed-wing loss, so that subvariant was
discarded. The additive screen remains an experiment, not the maintained
learner.

A smaller angular model was more effective. At each origin, a causal least
squares fit over at most the latest 25 transitions estimates a 3-vector bias,
one unconstrained 3-vector effect per issued command, and three nonnegative
diagonal rate-damping coefficients. The same equation and 0.05 s generic
first-order command filter apply to every recording; command count comes from
the data. A midpoint step advances body rate. The screen uses the prior cold
readout for body force, and replaces its rate and attitude recursion with this
fit. This is a **hybrid experimental wrapper**, not yet a unified public model.
It consumes 15 prefix transitions in each fixed-wing case and 25 in the quad
cases. Every later fit uses completed observations only.

| Recording | Previous cold readout rate RMSE | Hybrid rate RMSE | Previous velocity RMSE | Hybrid velocity RMSE |
| --- | ---: | ---: | ---: | ---: |
| fixedwing-80 | 1.104 | **0.403** | **0.476** | 0.610 |
| fixedwing-81 | 0.650 | **0.578** | **0.587** | 0.768 |
| quad-arm-115 | 2.570 | **2.316** | 0.363 | **0.241** |
| quad-arm-125 | 3.170 | **1.237** | 0.380 | **0.270** |
| quad-arm-135 | 35.224 | **9.001** | 4.731 | **0.642** |
| quad-change | 3.170 | **1.238** | 0.380 | **0.270** |
| paired-quad-fine | 0.007 | **0.002** | **0.009** | 0.044 |
| paired-quad-coarse | 0.005 | **0.003** | **0.006** | 0.010 |

Errors are physical 250 ms forecast RMSE, body rate in rad/s and velocity in
m/s, across all 263 frozen origins. Quad-change largely repeats arm 125 in
these rounded columns; it is not an independent vehicle. The hard arm-135 case
has only three origins. The previous cold readout and hybrid were run through
the identical v2 benchmark. Arm-125 counterfactual response mean error is
**0.934 → 0.629** relative to the plant Jacobian, with all six points saved.
The first arm-135 rate endpoint error is **47.8 → 11.0 rad/s**. Fixed-wing
command-response truth is not yet available, so its gain is factual rollout
evidence only. Warm updates remain roughly 0.6 ms fixed-wing and 1.5 ms quad;
cold fit plus first forecast is about 2 s on first compilation, and the first
update can take 0.4–0.5 s. These timings are not a hardware qualification.

The rate structure mattered more than learned inertia in a five-case
first-origin screen. A positive-damping fit without a gyro term had first-rate
endpoint errors of 0.51, 2.05, 1.56, 6.24 and 10.74 rad/s on fixedwing-80,
fixedwing-81, arms 115, 125 and 135 respectively. Letting three inertia ratios
produce the Euler gyroscopic term gave 0.51, 1.79, 1.77, 6.60 and 11.59.
This does not rule out gyroscopic terms on other systems; it does not justify
adding them to this candidate. Both screens used only each case's available
prefix observations.

The fixed 0.05 s lag is a material limitation. An exploratory rate-only replay
at 0.02, 0.05 and 0.10 s gave arm-125 250 ms rate RMSE of 2.759, 0.714 and
0.561 rad/s, and response mean error of 2.415, 0.629 and 0.495. The 0.10 s
value was **not** adopted from this evaluation. Lag must be identified from
the causal episode, not selected against these test outcomes. The force/rate
wrapper also duplicates rollout logic and does not implement the public
offline `fit/update` workflow. It should be folded into one dynamics model
before adoption, retaining the fast causal solve and the benchmark's measured
gains. Velocity regressions on the fixed wing and gentle paired quad remain
visible.

Frozen full baseline artifact: `artifacts/readout-positive-damping-v1/response-baseline-full`, manifest
`e06fbb3fdccef8bd8d6145fd362c2929f683938f0e3d02c79a064fdde6ec33b1`.
Additive screen: `artifacts/readout-positive-damping-v1/additive-control-full`, manifest
`6b1b8c9ef6582ac040abf2f0afc9bd4ea02357c875a81b42bb10fbfda40afd5e`.
Positive-damping hybrid: `artifacts/readout-positive-damping-v1/positive-damping-hybrid-full`, manifest
`8fe3a03c6e452dd9b929b3d31119a41f5a1bdef4ba73bdd56a7ffcb8c426cfd2`.
The v2 runner re-authenticated all input sources and verified the saved
forecasts, response arrays, controls, scores and table without refitting.
Candidate commits are `ebfc024` (additive) and `11774f3` (hybrid); the
benchmark was frozen first at `79d2fdd`.
