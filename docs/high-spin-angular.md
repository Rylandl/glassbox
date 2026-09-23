# High-spin angular recurrence diagnosis

The new 0.85-arm Crazyflow recording exposed a large 250 ms angular forecast
error in the public rate-memory model. Its second origin (row 150) misses the
measured body rate by **27.360 rad/s** at 250 ms, compared with **11.516 rad/s**
for holding the measured rate. The model still follows native one-step motion
well when it refits after each observation. This is a frozen-origin model error,
not a live Glassbox controller result.

The [frozen response protocol](harness/high-spin-response-v1.json) replays the
recorded flights exactly in Crazyflow, branches at rows 125 and 150 from the
true recorded state and hidden applied rotor condition, changes one *issued*
command for one 10 ms interval, then replays the original future commands. The
public learner sees only motion, issued commands and time. The simulator's
hidden rotor state and arm ratio are used only to establish counterfactual
truth. Model response uses the same perturbation on `OnlineFit.predict` at each
causal origin. These are development recordings; this test is diagnostic and
does not qualify a newly proposed model on unseen configurations.

| Arm | Origin | 250 ms rate error / hold, rad/s | Relative command-response error at 10 / 50 / 100 / 250 ms |
| --- | ---: | ---: | ---: |
| 0.85 | 125 | 6.135 / 5.658 | 1.280 / 0.361 / 0.318 / 0.513 |
| 0.85 | 150 | 27.360 / 11.516 | 0.325 / 0.247 / 0.407 / 0.899 |
| 1.40 | 125 | 2.228 / 4.247 | 0.854 / 0.566 / 0.574 / 0.757 |
| 1.40 | 150 | 1.189 / 2.951 | 0.200 / 0.137 / 0.151 / 0.339 |

The 0.85 row-150 command map is much closer over the first 50 ms than its
factual 250 ms forecast. Its response error also grows over the forecast. A
wrong instantaneous command coefficient alone does not explain the failure.
Response error at row 125 is less orderly, and the long-arm row 125 response is
inaccurate despite a better-than-hold factual forecast. Neither factual
forecasts nor local response alone are a sufficient qualification.

A follow-up decomposition locates the roll-axis failure more precisely. At
row 150 the fitted equation has a **+141.3 rad/s² constant term, large opposing
issued/delayed-command coefficients, and zero roll damping and rate-memory
gain**. The last 25 fitting transitions give 0.009 rad/s RMS roll-increment
error per 10 ms step; feeding the *measured* next 25 states and recorded
commands into the frozen equation raises that to 1.171 rad/s per step. Its
future roll acceleration averages +121.5 rad/s² versus +16.2 measured. The
resulting +26.32 rad/s roll error is already present under teacher forcing;
this axis does not need a free-rollout state excursion or numerical instability
to fail. The nine-column constant/issued/delayed-command design on the 25 fit
rows has condition number about 785, so distinct coefficient combinations can
explain the recent tape but imply very different later responses.

An oracle decomposition of the pinned simulator on that recorded trajectory
finds about +54.8 rad/s² average roll acceleration from applied thrust,
countered by −9.8 from propeller gyroscopic torque and −29.6 from rigid-body
inertial coupling. The public roll head has no cross-axis rate term to
represent that changing cancellation. These oracle quantities diagnose the
failure; none is supplied to Glassbox. A deliberately maximal sustained roll
command from the same state can reach much higher rates in this simulator, so
the error is a failure **conditioned on the recorded commands**, not a hard
actuator-envelope violation.

Focused exploratory screens tested fixed command-lag changes, direct-command
shrinkage, fit-window length, trajectory fitting, generic gyroscopic quadratic
terms, and a three-parameter skew rate coupling. None improved all four early
origins. In particular, using all available history in the existing rate head
reduced 0.85 row-150 error from 27.360 to 16.568 rad/s, but increased 1.40
row-150 error from 1.189 to 3.745. Unconstrained rotational coupling often
lowered training error while producing implausible coefficients or worse
rollouts.

A stronger physics diagnostic identified a plausible normalized inertia tensor
from the unpowered 0.5–1.0 s portion of each recording using only observed
rates. Holding that tensor fixed and refitting the current short-window command
head reduced 0.85 row-150 error to about 12.4 rad/s, while 0.85 row 125 rose to
about 7.6. This establishes that missing inertial coupling contributes, but it
does not solve the high-spin case. The passive segment was particularly
informative in these two quad flights; a universal learner cannot depend on
that segment existing. Jointly fitting unrestricted inertia and control terms
from all data produced unstable or implausible inertia on other origins. No
candidate from this screen was adopted into the public model.

A further exploratory screen replaced the independent issued/delayed command
maps with one causal applied-command map and a trace-normalized learned inertia
tensor. At the 0.85 row-150 origin, fitting the available prefix reduced the
250 ms rate error from 27.360 to 7.68 rad/s, but some 1.40 fits produced
non-positive inertia. Jointly adding an actuator angular-momentum term reached
4.86 rad/s at that origin in one setting, while worsening known quad origins:
arm-125 smoke row 125 went from 3.31 to 8.67 rad/s and arm-135 row 141 from
0.81 to 3.88 rad/s. These were exploratory NumPy/SciPy fits, not public-model
revisions or a qualified common recipe. A sweep of the assumed command time
constant changed high-spin errors by tens of rad/s and sometimes made learned
inertia nearly singular. That constant cannot be selected using the future
forecast being scored.

The [frozen independent-input replay](harness/high-spin-excitation-v1.json)
then tested whether the current episode supplies enough information. Starting
from the same pinned simulator state at row 100, it added balanced, independent
perturbations of amplitude 0.15 to the four issued commands through row 149.
The unexcited replay exactly reproduced the source and sealed public baseline.
The [slower 60 ms pattern](harness/high-spin-excitation-v2.json) was frozen
separately. Both change the vehicle's state, so direct branch-to-branch
forecast errors do not isolate the effect of the training data.

The [matched-state comparison](harness/high-spin-cross-state-v1.json) fits a
fresh public learner on each saved prefix, then forecasts from the **same
original row-150 state, observed history and future commands**. The separate
prefixes are a simulator diagnostic, not one physically continuous Throw
episode or fleet pretraining. Body-rate errors are at 250 ms; response errors
are relative to the simulator's 50 ms command Jacobian.

| Prefix used for fit | 0.85 rate, rad/s | 0.85 response | 1.40 rate, rad/s | 1.40 response |
| --- | ---: | ---: | ---: | ---: |
| Original | 27.360 | 0.247 | 1.189 | 0.137 |
| Independent 10 ms perturbations | 11.642 | 0.151 | 0.845 | 0.139 |
| Independent 60 ms perturbations | 13.928 | 0.172 | 2.937 | 0.143 |

The shared target rules out an easier forecast starting state as the reason
for the improvement. The training prefixes also contain different state
trajectories, so this does not isolate the command perturbations from every
other change in the training data. The fast-pattern fit nearly reaches the
original hold-rate error of 11.516 rad/s but does not beat it at 250 ms. Using
50 rather than 25 completed angular transitions with that excited fit reduced
the 0.85 endpoint
to 8.52 rad/s and kept its 50 ms response error near 0.16. A fixed 50-row
public-model smoke run, however, worsened the first two frozen origins of
fixedwing-80 (0.122 to 0.340 rad/s 250 ms RMSE), arm-125 (2.635 to 3.396),
and arm-135 (5.300 to 6.515). Thus simply lengthening the public window is
not a general replacement. A causal trajectory-loss prototype likewise cut
the worst factual endpoint while raising its 50 ms response error from 0.16
to 0.49. Both screens remained exploratory; the public learner is unchanged.

All three simulator packs verify from saved predictions and truth without
refitting. Their manifest SHA-256 values are `d41d08de98ff02f4480a1c79150711d530935eed3a467d80ff25345f1e09247e`,
`6913661d2530baba01fc1c3e19b123a3d4707721083d3a1b8097106e4ff7c054`,
and `54ca6159bb60fa9e9053efb0688d5e4981868f338f0946ae2a3849b70dd281a5`.

## Structural check at the difficult 250 ms origin

An exploratory forensic replay used the same pinned 0.85-arm simulator and
matched original row-150 forecast target. The simulator reproduced every
recorded state to `1e-5`. At that target, the omitted rigid-body Euler term
accounts for **−7.40 rad/s** of roll-rate change over the next 250 ms, and
propeller angular momentum for another **−2.42 rad/s**. The corresponding
terms on the 1.40-arm target are only −0.14 and +0.06 rad/s. These are
contributions to the native equation, not independent corrections to a fitted
model's error; the present coefficients can absorb some of them on the
training trajectory.

The frozen public head has a 27.36 rad/s endpoint error on the original
prefix and 11.64 after the independent-input prefix. Summing its one-step
errors while feeding it the **measured** future rates gives 28.06 and 12.68
rad/s respectively. Thus most of the failure is in the equation evaluated on
the real trajectory, before free-rollout drift. The public 50-step observed
history reconstructs its own lag state within `0.0002` at the origin, ruling
out history truncation as the main error.

For a structural upper-bound test, the roll/pitch/yaw torque map was refitted
on each branch's preceding 50 transitions while supplying the simulator's
**true inertia, true applied motor trajectory, rotor spin directions and
propeller inertia**. Only the torque map was estimated from the prefix. The
rate was then rolled out freely from the common target; the true *future*
motor trajectory was supplied to isolate the missing dynamics. This is not a
deployable Glassbox forecast or evidence that those hidden quantities can be
identified from issued commands alone.

| Fit prefix | Oracle dynamics excluding rotor momentum | Including rotor momentum |
| --- | ---: | ---: |
| Original | 8.20 rad/s | 0.97 rad/s |
| Independently excited | 4.58 rad/s | 1.26 rad/s |

In another oracle sensitivity check, true inertia and rotor-momentum physics
were retained, but motor speed was reconstructed causally using the
simulator's static command-to-RPM conversion and a single first-order RPM
filter. With an 80 ms filter the original-prefix endpoint error was 12.88
rad/s; at 50 ms it was 4.75 rad/s. Both are well above the 0.97 rad/s result
with the true motor trajectory. These checks use otherwise hidden simulator
parameters and must not be read as candidate-model scores. They show why
merely adding an Euler term to the current fixed-lag, axis-local head is
unlikely to close the gap.

The maintained angular head fits each axis separately from 25 increments,
with two independent command maps, a fixed 80 ms command lag and diagonal
rate feedback. Its force head uses a separate 50 ms command lag, and neither
head uses the other's evidence to estimate applied actuation. The angular
head has no cross-axis angular-momentum term or velocity-dependent moment;
the latter is an additional structural limit for fixed-wing flight across
airspeed and angle of attack, although this high-spin replay does not test
that limit. The next candidate should share a causal actuator representation
between force and torque, express rate dynamics through a positive-definite
episode-fitted inertia and an actuator-momentum term that can fit to zero,
and retain control-response information across the episode. Its low-dimensional
local residual can adapt without refitting the whole command map from the
last 25 samples. This is one generic rigid-body recipe: no motor count, mixer,
vehicle class, simulator state or pretrained core is input to the learner.
The candidate must beat the frozen factual and command-response evidence with
only causal episode observations before replacing the public head.

The truth and baseline evaluation are small sealed packs:

- [Counterfactual truth](../artifacts/high-spin-response-v1/truth/manifest.json):
  `583b307fc075bb038e8c353ddcc914b241442fba5e308857da9ee41e9c8a4c0e`
- [Public-model evaluation](../artifacts/high-spin-response-v1/baseline/manifest.json):
  `6fd3999cc0094c48a5fe0421d93fb32e1fab679506bfacd7cd879c9f2b813fb2`

The harness was committed before collection at `72d31e0`. Verify the physical
response and forecast scores without fitting:

```bash
PYTHONPATH=src:scripts python scripts/qualify_high_spin_response.py verify \
  --truth artifacts/high-spin-response-v1/truth \
  --truth-manifest-sha256 583b307fc075bb038e8c353ddcc914b241442fba5e308857da9ee41e9c8a4c0e \
  --output artifacts/high-spin-response-v1/baseline \
  --manifest-sha256 6fd3999cc0094c48a5fe0421d93fb32e1fab679506bfacd7cd879c9f2b813fb2
```
