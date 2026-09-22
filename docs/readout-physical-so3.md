# Physical attitude sensitivity in the fast readout

One fixed generic physical-attitude penalty substantially improved the fast
readout's known fixed-wing 250 ms forecasts. It did not make the recursive
dynamics plant-accurate. The result is an architectural screen, not a production
replacement or a controller qualification.

The candidate starts from a fresh per-recording prefix. Its feature, filter,
memory and normalization parameters are then frozen, as in the prior direct
readouts. It retains the measured-increment target, prefix ridge, curvature
penalty, and previous body-motion sensitivity penalty. The only new term is
`0.0001 * S_attitude`, where `S_attitude = sum D_i D_i.T` and `D_i` is the
derivative of the complete frozen readout feature vector with respect to a
right SO(3) perturbation of the measured midpoint attitude in radians. This
changes body-relative velocity and body-gravity direction together. One shared
Cholesky solve updates all six acceleration outputs after each observation;
there is no vehicle-specific rule, prior fit, sweep or fleet pretraining.

The protocol and code were committed before the eight candidate fits. The six
earlier recordings and two paired quad observation schedules were evaluated at
their previously frozen origins. Authenticated full-learner, curvature and
motion-sensitivity predictions served as unchanged controls. The new candidate
made **4,187 causal updates**. Its first prediction occurs after the same
recorded prefix as each control: 0.75 s on fixed wing and 1.25 s on quads.

| Recording | 250 ms velocity RMSE, candidate / previous sensitivity (m/s) | 250 ms body-rate RMSE, candidate / previous sensitivity (rad/s) | Candidate / full body-rate RMSE |
| --- | ---: | ---: | ---: |
| Fixed wing 80 | 0.923 / 1.209 | **2.593 / 12.249** | 0.960 |
| Fixed wing 81 | 2.242 / 7.330 | **6.027 / 42.002** | 0.415 |
| Quad arm 115 | 0.521 / 0.538 | 3.282 / 4.630 | 0.541 |
| Quad arm 125 | 0.330 / 0.345 | 2.171 / 2.627 | 0.410 |
| Quad arm 135, truncated | 4.710 / 4.817 | **62.867 / 63.759** | 0.710 |
| Quad change | 0.330 / 0.345 | 2.171 / 2.627 | 0.410 |
| Paired quad 10 ms | 0.001670 / 0.001637 | 0.005106 / 0.005101 | 0.0371 |
| Paired quad 50 ms | 0.003407 / 0.003390 | 0.004447 / 0.004420 | 0.0325 |

The candidate improves both fixed-wing recordings and all four earlier quad
recordings at 250 ms. The paired gentle quad regressions are small in absolute
terms. At 50–100 ms, some cases worsen: fixed-wing 80 velocity, and the early
body-rate horizons in the truncated quad recording. That truncated recording
contains just 62 updates and three conditional origins; its absolute 250 ms
rate error remains unacceptable. Per-horizon and per-origin scores are preserved
in the artifact rather than hidden by the table.

Against the full learner, equal-family geometric 250 ms ratios are **0.210**
for velocity and **0.361** for body rate; all old one-step and family ratio
flags pass. On fixed wing alone the candidate/full 250 ms rate ratio is
**0.631**. These are known-recording results, not cross-vehicle generalization.
Warm complete candidate updates take **0.584–0.721 ms** on fixed wing,
**1.37–1.47 ms** on the earlier 10 ms quads, and **1.54/0.94 ms** on the paired
10/50 ms quads. The historical sensitivity controls took roughly 0.56 ms,
1.29–1.33 ms and 1.32/0.65 ms respectively. Those control measurements came
from separate earlier runs, so the differences are directional rather than a
strict paired timing claim. Construction, first prediction, first update and
every complete-update duration are saved.

The saved-data verifier authenticates all source packs, checks exact prefix
models, design features, increment targets and old curvature penalties, then
rebuilds every new normal equation in NumPy. The largest componentwise solve
backward error is **1.12e-14**. It recomputes every physical metric, timing
summary and origin count without fitting. Three independent derivative checks
cover three and four commands, 10 and 50 ms, body-gravity coupling, history
signs, and finite differences. A pre-fit harness invocation stopped on a
string-versus-Path authentication bug before any candidate fit or output was
created; the committed fix preceded the sole eight-case fit.

## The remaining physical gap

A second, committed no-fit audit differentiated each saved origin model's full
50 ms and 250 ms rollout against the previously verified Cascade tangent data.
All 14 origins and five measured steps per fixed-wing recording were included.

| Fixed-wing recording | Median local attitude → rate norm: candidate / plant | Median five-step attitude → rate gain: candidate / plant |
| --- | ---: | ---: |
| 80 | 0.847 / 2.912 | **161.8 / 3.90** |
| 81 | 0.986 / 2.905 | **265.6 / 3.89** |

The preceding motion-sensitivity readout's five-step gains were **6,912 /
6,275**. The new penalty reduces amplification by about 43×/24×, consistent
with its much better realized forecasts, but it also suppresses local angular
response *below* the plant and leaves five-step gain about 41×/68× too high.
Directional alignment of the five-step matrices with the plant is near zero.
The initially unfitted readout is especially poor: its first-origin gain is
**19,325 / 22,122** and its first 250 ms body-rate error is **8.96 / 22.06
rad/s**. The penalty has not acted at that first scored origin. The no-fit
Jacobian checks agree with centered finite differences to relative error below
**3.6e-6**; the pack verifies from saved matrices without model or simulator
calls.

Thus a head-local sensitivity penalty is helpful but cannot by itself align
the complete recurrence. The next experiment should fit or constrain short
physical trajectories directly, including the causal prefix, without vehicle
metadata or a pretrained core. It should test whether absolute fixed-wing and
hard-quad rollouts improve while retaining the direct head's speed. Fitting a
stronger local penalty would risk further suppressing legitimate physical
response. The current candidate remains experimental until this gap and the
public offline/update workflow are addressed.

The exact protocols, source commits and authenticated artifact hashes are in
[the index](readout-physical-so3.json). Both fixed-wing recordings come from
one airframe and the paired quad flight is gentle; neither is a held-out
configuration test or a Throw controller trial.
