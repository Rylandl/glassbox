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

The next architecture should represent rigid-body angular momentum and
unobserved actuator response in one episode-fitted formulation, while keeping
constant physical parameters separate from changing disturbances. Positive-
definite inertia and one shared actuator state must be learned from the causal
prefix, with trajectory error and command-response error evaluated together.
The recent command window is poorly conditioned and the following commands
change substantially; absent independent input excitation, the learner must
retain uncertainty rather than infer a precise torque map. The method cannot
require a passive prelude, vehicle metadata, a pretrained prior, or a
per-platform rule. Screen causal 50–250 ms response and forecast on the frozen
known suite, then collect new high-spin configurations before adoption. A
single favorable 0.85 endpoint is insufficient.

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
