# Short-trajectory fit of the fast readout

The fast experimental readout fits one-step acceleration increments accurately,
yet some 250 ms body-rate forecasts diverge. This iteration tested a single
preconditioned Gauss-Newton correction on an **observed, completed 250 ms
trajectory**. The feature map remains gravity-free, the vehicle is fit only from
its own causal prefix, and gravity and rigid-body kinematics remain exact. The
correction changes the same linear readout coefficients; it adds no vehicle
branch, prior flight fit or consumer setting. Short-rollout fitting is motivated
by [operator inference with rollouts](https://arxiv.org/abs/2212.01418), not
validated for Glassbox by that paper.

The [frozen runner](benchmarking.md) replayed the same eight recordings,
**4,187 causal updates and 263 conditional origins** for each full candidate.
Every copied pack in `artifacts/readout-trajectory-v1` passed the saved-data
verifier, which reauthenticated source controls and rescored forecasts without
fitting. `online-full` used a correction after every new observation (source
`f28ca6e`, manifest
`95585cb0cb08fc0205c2f89c139b98776206208f0f5a93f155696979a56afb77`);
`cold-online-full` also corrected the observed prefix (source `54b9c54`,
manifest `7fd2d14670efa8e65da701ed69ffb8c96eb600c6736c4567373fd43ed7034a89`).
The current isolated candidate keeps **only the prefix correction**, followed
by the original fast online readout (source `d4630a8` on
`codex/readout-recursion-structure`; `cold-only-full` manifest
`9dd47d51fd26a0c0e86b622bc29ad9d1f836624ed214623ddb140f0a803d52a0`).
The public learner and benchmark were not changed.

| Recording | 250 ms rate, cold-only / gravity-free / direct (rad/s) | 250 ms velocity, cold-only / direct (m/s) |
| --- | ---: | ---: |
| Fixed wing 80 | **1.104 / 1.520 / 2.593** | **0.476 / 0.923** |
| Fixed wing 81 | **0.650 / 1.906 / 6.027** | **0.587 / 2.242** |
| Quad arm 115 | **2.570 / 2.608 / 3.282** | **0.363 / 0.521** |
| Quad arm 125 | 3.170 / 3.143 / **2.171** | 0.380 / **0.330** |
| Truncated quad arm 135 | **35.224 / 36.821 / 62.867** | 4.731 / **4.710** |
| Paired quad 10 ms | 0.007 / 0.009 / **0.005** | 0.009 / **0.002** |
| Paired quad 50 ms | 0.005 / 0.006 / **0.004** | 0.006 / **0.003** |

The quad-change tape repeats arm 125's scored trajectory and is not independent
evidence. The first fixed-wing-81 origin fell from 6.717 rad/s with gravity-free
features to **0.456 rad/s** after the prefix correction; the direct readout was
22.063. Native one-step errors stayed close. The hard quad remains far from
useful. Arm 125 is a substantial regression versus direct, including its first
origin (8.750 versus 7.478 rad/s). Its observed pitch rate reverses and rises
to about +5 rad/s during that first 250 ms; all readout variants forecast a
negative pitch rate there. The next 250 ms of commands are within the prefix's
per-channel ranges, but weakly excited command directions and unseen coupled
motion still limit what this recording can establish.

Cold-only warm whole updates stayed near **0.54 ms** for fixed wing and **1.40
ms** for quads, comparable to gravity-free. The first shape-specific JAX
compilation raised measured initialization plus first forecast to roughly
**1.7–2.0 s**; first online update still compiled for about 0.4 s. Later
same-shape runs reused compilation. These elapsed costs count against a real
cold-start Throw budget. Correcting every online step reduced settled arm-125
median per-origin error from 0.210 to 0.007 rad/s, but did not resolve the
aggregate arm-125 loss; quad warm updates rose to about **3.1 ms**. It was
removed from the current candidate. The stronger fixed-wing result therefore
comes chiefly from fitting the completed prefix trajectory, not from an
expensive per-update loop.

Two smaller exploratory checks did not change the decision. Scoring only the
six learned motion coordinates in the trajectory residual changed the
fixed-wing tradeoff but left arm 125 near 3.16 rad/s; this was a smoke and
two-case replay, not a full run. Attempting multiple prefix windows produced
the same single window because the frozen prefix holds only enough history for
one complete 250 ms trajectory. A one-step per-command lag-time fit from that
prefix reduced its own residual but improved arm 125's first-origin rate error
only from 8.750 to 8.539 rad/s. That lag diagnostic was exploratory and was
not promoted into the candidate.

The current candidate is a meaningful step in recursive fitting but is **not
adopted**. It has not passed public offline/update qualification, held-out
configurations, counterfactual control-response tests or a Throw controller
trial. The next model gap is the early arm-125 angular reversal and the hard
quad's large rate error. Diagnose command-to-rate coupling and whether these
directions were identifiable from the causal prefix before changing the network
again. Keep the one-command benchmark run and score physical first and late
errors; avoid a catalog, acceptance guard or another scale sweep.
