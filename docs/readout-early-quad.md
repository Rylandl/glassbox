# Early quad response: the fitted trajectory does not identify control effects

The next arm-125 gap is a **counterfactual command-response error as well as a
recursive forecast error**. The cold trajectory readout's first 250 ms forecast
ends 8.750 rad/s from the observed body rate. Its pitch rate stays negative
while the plant reaches +5.18 rad/s. Giving the frozen cold model the *observed*
states at each step reduces its pitch-rate one-step RMSE over that interval to
0.409 rad/s; causal updates reduce the on-trajectory value to 0.042 rad/s.
Neither number measures the effect of changing a command.

The pinned arm-125 Crazyflow plant was replayed from the recorded release with
zero applied rotor thrust and the exact issued commands. Its native state at row
125 matched the sealed recording exactly. At six recorded rows, the simulator
was branched from its exact hidden motor state and each commanded thrust fraction
was varied by 0.01, clipped to [0, 1]. Central or endpoint finite differences
gave the 3×4 Jacobian of the **next observed body rate** with respect to the four
issued commands. No plant state or parameter entered either Glassbox fit. This
is a local 10 ms command-response probe, not a 250 ms controller trial.

| Row / time since first actuation | Plant Jacobian Frobenius norm | Cold readout relative Jacobian error | Maintained full learner relative error |
| --- | ---: | ---: | ---: |
| 125 / 0.25 s | 0.490 | 1.11 | 1.08 |
| 135 / 0.35 s | 0.482 | 0.95 | 0.97 |
| 145 / 0.45 s | 0.475 | 0.86 | 0.83 |
| 175 / 0.75 s | 0.495 | 0.92 | 0.75 |
| 225 / 1.25 s | 0.473 | 0.84 | 0.57 |
| 325 / 2.25 s | 0.478 | 0.94 | 0.71 |

Relative error is `||J_model−J_plant||_F / ||J_plant||_F`, at the same recorded
physical state and issued-command history. At the first origin, the actual pitch
row is `[-0.147, -0.150, +0.136, +0.134]`; the cold model gives
`[+0.043, -0.178, +0.046, +0.201]` rad/s per unit command. The first motor's
sign is wrong. The maintained model also misses the early response, so this is
not solely a frozen-readout defect. Both model comparisons used the declared
causal 0.75 s prefix and consumed each later observation only after prediction.

Only the last 0.25 s of that prefix contains **actuated fitting transitions**;
the first second of the source flight has no actuation. The 25 issued-command
rows used by initialization have centered singular values
`[2.640, 0.972, 0.188, 0.122]`. The next 250 ms reaches 5.2× the training
maximum along the weakest command direction. The fitted pitch-rate range is
−0.60 to +2.91 rad/s, versus −3.26 to +5.53 in that forecast. These are
descriptive support diagnostics: they do not apportion the error uniquely
between missing excitation, state extrapolation and unstable recurrence.

Focused checks did not rescue the first arm-125 forecast. Projecting positive
instantaneous angular-rate feedback lowered its endpoint error only to
7.49 rad/s and made fixed-wing 81 diverge to 152.57 rad/s. Smoothing the linear
lag coefficients left arm 125 at 8.76; a refitted rate-independent angular
head gave 8.71; four additional completed-prefix trajectory corrections gave
about 8.8. Shortening the raw history to 0.2 s and fitting 0.55 s of the *same*
0.75 s prefix worsened arm 125 to 22.82 while improving arms 115 and 135. An
extra 0.5 s of earlier recording, mostly unactuated and outside the frozen data
budget, gave 13.93. These are exploratory first-origin screens, not full-suite
comparisons or adopted model variants; the original candidate source was
restored unchanged afterward.

An exploratory simulator branch added a fixed ±0.15 orthogonal motor-command
probe to the first 25 actuated rows, then replayed the original commands. The
weakest/strongest prefix-command singular-value ratio rose from 0.046 to 0.170,
and first-origin relative response-Jacobian error fell from 1.11 to 0.705.
The altered commands also changed the origin state substantially; that new
trajectory's 250 ms rate forecast was poor (57.99 rad/s endpoint error).
The response improvement therefore supports testing excitation, but does not
show that probing alone fixes recurrence or establish a matched flight gain.

A 21-parameter exploratory angular-momentum model then used the same 25
actuated observations in each branch. It learned an arbitrary 3×4 additive
command-effectiveness matrix, positive diagonal rate damping and positive
diagonal inertia; the gyroscopic `−J⁻¹(ω×Jω)` term was fixed by that inertia.
One-step and completed-prefix trajectory residuals were fitted together. On
the original branch its first-origin response error was **0.873×** and its
250 ms rate endpoint error **6.60 rad/s**, versus the cold readout's 1.11× and
8.75. On the independently excited branch these were **0.472× / 4.36 rad/s**,
versus 0.705× / 57.99. This is useful evidence for a compact physically
structured angular head, but neither forecast is accurate enough and the
standalone model has no force or fixed-wing dynamics. A simpler unconstrained
rate-product regression fit the prefix increments even more closely but its
free rollout became nonfinite; local equation fit alone was not sufficient.

The saved `cold-only-full` pack, manifest
`9dd47d51fd26a0c0e86b622bc29ad9d1f836624ed214623ddb140f0a803d52a0`,
was reverified without fitting. Simulator truth came from the sealed
`online-fit-v1` arm-125 native tape using pinned Crazyflow 0.3.2 and the
source plant's 1.25 arm ratio; exact replay before perturbation checked the
state and hidden-actuator trajectory. The local
`artifacts/readout-early-quad-v1` pack holds the simulator Jacobians, dither
branch, protocol, physics-screen script and output; manifest
`006e073a613a6b0e1916193135d0eda99769e5b6312fc1d25d01c87e3452259e`.
The new probes are exploratory and have not been added to the frozen benchmark
or promoted to an acceptance flag.

**Next:** freeze a small saved command-response probe inside the one-command
benchmark, then test a single generic additive command-to-wrench head with a
compact body-state dependence and angular-momentum recursion. Its coefficients
must be learned from the current episode; no actuator geometry, mixer, vehicle
class or pretrained feature map may enter. Score the probe alongside existing
first/late 250 ms forecasts on quads and fixed wing. Do not claim a better
dynamics model from teacher-forced errors alone.
