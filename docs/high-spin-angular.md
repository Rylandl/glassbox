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
wrong instantaneous command coefficient alone does not explain the failure;
state evolution and hidden actuator dynamics matter. Response error at row 125
is less orderly, and the long-arm row 125 response is inaccurate despite a
better-than-hold factual forecast. Neither factual forecasts nor local response
alone are a sufficient qualification.

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

The next architecture should represent rigid-body angular momentum and
unobserved actuator response in one episode-fitted formulation, while keeping
constant physical parameters separate from changing disturbances. It must
identify them without requiring a passive prelude, vehicle metadata, a
pretrained prior, or a per-platform rule. First screen its causal 50–250 ms
response and forecast on the frozen known suite, then collect new high-spin
configurations before an adoption decision. A single favorable 0.85 endpoint
is insufficient.

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
