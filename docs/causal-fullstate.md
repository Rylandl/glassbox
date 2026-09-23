# Causal actuator model: full-state qualification

**Historical comparison, superseded for adoption decisions.** This run fitted
the readout on each benchmark prefix but reconstructed actuator state and fitted
inertia using earlier rows on several quad recordings. The incumbent's frozen
online runner began at row 50 (row 10 for paired coarse), so the candidate had
additional causal data. The [current status](status.md) reports the corrected
matched-prefix and timed-publication comparisons. The numbers below remain the
original saved result; they should not be described as equal-data evidence.

The episode-fitted causal actuator model is now the **selected architecture for
public integration**. It is not yet the maintained learner. The decision rests
on full-state predictions at every frozen origin in the eight-case online
readout benchmark and six matched command-response probes. Each candidate fit
uses only observed states and issued commands through its origin, with no fleet
training or vehicle metadata. The published public `OnlineFit` predictions are
the comparison. The candidate refits each prefix as a batch, whereas the public
model updates incrementally; these results compare complete fitting procedures,
not an isolated network change.

The candidate shares a fitted nonlinear applied-command state between body
specific force and rigid-body torque. The torque equation includes a learned
inertia tensor and an actuator angular-momentum term, which can fit near zero.
Gravity, body/world transforms and rotation kinematics remain analytic. One
parameterization and fit procedure serves all cases. Input count comes from the
recording; no quad or fixed-wing branch is used.

Errors below are 250 ms RMSE over **all** previously frozen forecast origins;
each cell is candidate / current public model in physical units. The velocity
and body-rate errors are separate because they can move in different directions.

| Recording | Origins | Velocity, m/s | Body rate, rad/s |
| --- | ---: | ---: | ---: |
| fixedwing-80 | 14 | 0.358 / 0.506 | 0.299 / 0.345 |
| fixedwing-81 | 14 | 0.426 / 0.751 | 0.350 / 0.597 |
| quad-arm-115 | 54 | 0.033 / 0.257 | 0.065 / 0.841 |
| quad-arm-125 | 54 | 0.023 / 0.124 | 0.110 / 0.551 |
| quad-arm-135 | 3 | 0.404 / 0.596 | 0.510 / 6.237 |
| quad-change | 54 | 0.023 / 0.124 | 0.110 / 0.551 |
| paired-quad-fine | 35 | 0.0034 / 0.0330 | 0.0056 / 0.0027 |
| paired-quad-coarse | 35 | 0.0067 / 0.0058 | 0.0105 / 0.0037 |

The fixedwing-80 early origins still lose in body rate: rows 15 and 31 have
0.277 / 0.165 and 0.490 / 0.052 rad/s endpoint error. Later origins make its
all-origin rate RMSE lower. The paired flights are gentle and have small
absolute rate errors; the relative losses there remain recorded. `quad-change`
shares most of its early dynamics and data with arm-125, so it is not a fully
independent configuration result. The new 0.85-arm high-spin origin remains a
separate diagnostic: candidate body-rate endpoint error is 0.410 rad/s versus
27.360 for the public model, with 0.605 m/s candidate velocity error and
0.066 rad candidate orientation error at 250 ms. It is not included in the
eight-case table.

At the six frozen arm-125 counterfactual probes, relative 10 ms body-rate
command-Jacobian error averages **0.072** for the candidate versus **0.331**
for the public model. The underexcited first probe improves from 1.071 to
0.106; all six improve. The historical high-spin response screen
also found a large 0.85-arm improvement. Fixed-wing counterfactual truth is
still absent.

The full-state forecast evaluator was committed at `8f22cdd` before the
263-origin run; the response scorer was committed at `6bf3d0d` before its six
fits. Source lives on `codex/causal-fullstate`. Saved candidate forecasts and
scores are in `artifacts/causal-fullstate-v1/` (ignored by Git, as other local
benchmark artifacts are). Running the evaluator's `--full --verify` and
`--response --verify` modes recomputes both reports from saved predictions
without refitting. SHA-256 of `summary.json` is
`ce5f88bffa41b2359a2f166e307da222dc2294feb2d648a5abed85025e55ab15`;
SHA-256 of `response.json` is
`2bbe3476476235c6b1d491745181925ea0dd67fd56e2b796f84a89db8c083658`.
The sealed pack manifest hash, covering those scores and all saved forecasts,
is `5eab1307ac8484fa56eda8785073f05e803ce6eade2e1d4dcd9362bd40f4b8dd`.

The candidate does not yet meet the public product contract: it has no
fingerprinted immutable revision, JAX prediction derivatives, or background
publication path. The full benchmark fits a fresh causal prefix at each origin;
it does not measure which revisions a live controller would have available
between publications. Measurement-noise tolerance, unseen vehicle classes and
live Throw/Dart control remain unqualified. Fit time is a measurement for that
publication design, not a gate that outweighs these accuracy gains. The next
iteration should replace the maintained learner with one public implementation
of this architecture, preserving the fit/predict/update contract and scoring
actual revision availability separately.
