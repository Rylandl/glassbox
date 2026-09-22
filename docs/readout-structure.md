# Fast readout: learned gravity as a recursive feedback path

One structural change produced a large known-recording gain: remove body-gravity
direction from the *learned* acceleration features while retaining exact world
gravity and rigid-body kinematics in the integrator. Each vehicle and schedule
was refit from its own causal prefix. No vehicle metadata, fleet-trained core,
class branch, controller guard or new consumer option entered the experiment.
This is a shape-preserving ablation; unused feature slots remain and parameter
count has not yet fallen. The scientific source is commit `e35cfe3` on
`codex/readout-recursion-structure`; production source remains unchanged.

The frozen [shared benchmark](benchmarking.md) ran all eight cases, 4,187 causal
updates and 263 previously fixed forecast origins. The full pack is
`artifacts/readout-structure-v1/gravity-free-full`, manifest SHA-256
`3691a48e700ab91ae61cadd6fb77973576776c456e5928ce96ed08b36f41fb1c`.
The saved-data verifier reauthenticated controls and recomputed scores from
saved predictions without refitting after the pack was copied into `artifacts`.

| Case | 250 ms rate: candidate / direct (rad/s) | 250 ms velocity: candidate / direct (m/s) | Native one-step rate: candidate / direct (rad/s) |
| --- | ---: | ---: | ---: |
| Fixed wing 80 | **1.520 / 2.593** | **0.476 / 0.923** | 0.146 / 0.138 |
| Fixed wing 81 | **1.906 / 6.027** | **0.627 / 2.242** | 0.186 / 0.183 |
| Quad arm 115 | **2.608 / 3.282** | **0.372 / 0.521** | 0.015 / 0.012 |
| Quad arm 125 | 3.143 / 2.171 | 0.370 / 0.330 | 0.017 / 0.012 |
| Truncated quad arm 135 | **36.821 / 62.867** | 4.758 / 4.710 | 0.056 / 0.049 |
| Paired quad 10 ms | 0.009 / 0.005 | 0.008 / 0.002 | 0.00008 / 0.00002 |
| Paired quad 50 ms | 0.006 / 0.004 | 0.005 / 0.003 | 0.00081 / 0.00042 |

The quad-change tape is nearly identical to arm 125 on these scored conditions.
First-origin fixed-wing rate error fell from 8.96 to 5.15 and 22.06 to 6.72
rad/s. The fixed-wing improvement persisted at later origins. Ordinary-quad
median per-origin rate error worsened from 0.026 to 0.243 rad/s on arm 115 and
from 0.040 to 0.213 on arm 125. The hard quad improved but remains far from a
useful 250 ms forecast. Warm whole updates stayed around 0.54 ms fixed-wing and
1.38 ms quad; first-use compilation remained hundreds of milliseconds and the
0.75/1.25 s observed prefix is part of cold-start cost.

Two exploratory alternatives narrowed the mechanism. A forecast-only mask of
the full quadratic head gave modest 250 ms gains but noticeably worse hard-quad
one-step error; the four-case smoke pack is
`artifacts/readout-structure-v1/no-quadratic-smoke`, manifest
`f5ad36c6de73effb2032e9347b11a5492b7890272678191e9b0e9e1d7b0313e6`.
It did not refit a reduced head and is only an ablation. Keeping gravity in the
linear readout while removing its quadratic and nonlinear interactions nearly
restored both original fixed-wing first-origin errors (8.89 and 21.33 rad/s).
That prototype is preserved at `1913776` and reverted at `1edf731`; its
four-case smoke pack is `artifacts/readout-structure-v1/linear-gravity-smoke`,
manifest `2a8730952f5380f8b88f00a5f4d8c3bfb41566630787269ed1aec6edcf18040a`.
It did not warrant a full run. Both copied smoke packs passed saved-data
verification. The no-quadratic source was exploratory and not committed.

The result supports separating known gravity from learned free-flight forces,
but is not proof that gravity is irrelevant to every rigid-body system. Gravity
can break symmetry in contact and interacting systems, as the
[subequivariant dynamics study](https://arxiv.org/abs/2210.06876) demonstrates.
The [SINDYc study](https://arxiv.org/abs/1605.06682) likewise motivates
testing a parsimonious input-aware basis rather than keeping every polynomial
interaction. These papers motivate the structure; neither validates this
particular fit. The ordinary-quad loss and absent counterfactual truth mean
there is no production adoption, offline/update qualification, held-out vehicle
claim or Throw controller claim here.

The next named gap is to retain the fixed-wing and hard-quad gains while
recovering ordinary-quad late forecasts, ideally by representing the missing
command/hidden response directly rather than restoring a gravity proxy. If that
works, remove the inactive gravity coordinates from the actual arrays and
measure parameter count and whole-update time before considering adoption.
