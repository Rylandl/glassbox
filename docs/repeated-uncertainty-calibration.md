# Repeated-fit uncertainty

Held-out forecast error is useful evidence of future error scale in this study.
Local parameter information does not consistently describe repeated-fit error,
and adding its propagated covariance to empirical error often worsens the estimate.
`PredictiveTrajectory` therefore exposes the two matrices separately; the combined
covariance and standard-deviation shortcuts have been removed.

The study uses the library's known synthetic quadrotor, 16 independent fits per
condition, and paired random seeds across conditions. Each fit uses two six-second
training flights, two separate calibration flights, and the default 400-step
optimizer at 50 Hz. Three shared test command designs receive independent
observation noise for each fit. Noise is Gaussian in rigid-body tangent coordinates;
base standard deviations are 2 mm position, 0.02 m/s velocity, 0.002 rad attitude
and 0.01 rad/s body rate. The plant has no process noise. The collective condition
uses equal motor commands throughout training, calibration and testing.

At one second, restricting test windows to nominal and true motion inside the
fitted operating envelope:

| Condition | Empirical error / test MSE | (Empirical + parameter) / test MSE |
| --- | ---: | ---: |
| Clean | 0.93–1.11 | 1.07–1.37 |
| Base noise | 0.88–1.01 | 1.53–2.55 |
| Triple noise | 0.83–1.09 | 1.54–2.27 |
| Collective, base noise | 0.83–0.97 | 0.83–1.88 |

Ranges span position, velocity, attitude and body-rate groups; each group averages
its three coordinates. Windows are averaged within each fit before averaging fits.
Every fit contributes inside-support windows: 89–90% of windows for the first three
conditions, 100% for collective. The saved report also includes all-window results,
0.1 and 0.5 second horizons, per-fit ratios and errors against latent truth.
These are second-moment comparisons, not probability coverage measurements.

Parameter comparisons use each fit's resolved, scaled eigendirections. Squared
parameter error is weighted by local precision and averaged per direction, then
fit. Centered variation replaces error against truth with deviation from the
across-fit mean, with the sample-variance correction.

| Condition | Resolved rank / 9 | Parameter error / local variance | Centered variation / local variance |
| --- | ---: | ---: | ---: |
| Clean | 3 | 0.055 | 0.0068 |
| Base noise | 5 | 0.38 | 0.49 |
| Triple noise | 5–8 | 11.3 | 0.34 |
| Collective, base noise | 1–4 | 707 | 3.90 |

No fit resolves all estimable directions. Rank uses a fit-local relative threshold;
noise-induced parameter bias is especially large with collective excitation.
The local inverse information is neither a consistent estimate of sampling
variation nor of total parameter error in these fits.

The library retains local information for diagnostics and updates, and empirical
forecast errors as measured evidence. The planner's covariance sum remains its
risk-penalty policy.

[Machine report](investigations/fit-uncertainty/report.json) and
[evidence archive](investigations/fit-uncertainty/evidence.zip) retain all 64 fits'
arrays and beliefs, optimization controls, the design, environment versions and
hash-verified executed sources. The earlier
[six-update pilot](investigations/repeated-uncertainty-calibration/report.json)
remains as historical data; this runner replaces its script.

```sh
uv run python scripts/evaluate_fit_uncertainty.py --output /tmp/fit-uncertainty
uv run pytest tests/test_fit_uncertainty.py -q
```

## Initial-state follow-up

Fixing each window's initial state to one noisy observation lets the fitted
damping suppress that noise. In a follow-up on four fresh datasets per condition,
substituting known initial states reduced parameter MSE by about 98% in the three
noisy conditions, while the future observations remained noisy.

The candidate instead jointly fitted twelve tangent offsets per window alongside
the model, charging the initial observation in the loss. This follows the
[initial-state estimation](https://www.mathworks.com/help/ident/ref/findstates.html)
approach. Both candidate and current fitter were tested at 400 and 4,000 optimizer
steps on new training and test seeds.

At 4,000 steps, candidate/current error ratios were:

| Condition | Parameter MSE | Forecast MSE from noisy starts | Forecast MSE from true starts |
| --- | ---: | ---: | ---: |
| Base noise | 0.20 | 0.83–0.93 | 0.062–0.085 |
| Triple noise | 0.82 | 0.95–1.82 | 0.031–0.070 |
| Collective, base noise | 0.37 | 1.56–12.55 | 0.031–0.041 |

Parameter MSE covers the nine estimable log coordinates. Forecast ratios are at
one second on common windows inside every compared prediction's and the true
trajectory's operating support. Ranges span the four state groups; collective
true-start angular errors are exactly zero for both models, so their ratios are
omitted. The report retains both iteration budgets, clean controls, absolute
errors and forecasts from noisy starts scored against latent truth.

The candidate is not the default: its better dynamics estimates do not consistently
improve the existing noisy-start prediction workflow. Initialization during fitting
and forecasting needs to be evaluated together.

[Comparison report](investigations/initial-state-fitting/report.json) and
[executed sources and arrays](investigations/initial-state-fitting/evidence.zip).

## Local state reconstruction

A follow-up prototype solves twelve starting-state coordinates locally using
the existing rollout. The same numerical solve supports fitting and causal
reconstruction from past telemetry; it passed quadrotor, fixed-wing and
three-control residual-model checks.

Adding half a second of past telemetry to the preceding joint-fit candidate
reduced one-second forecast MSE by 33–78% across state groups in the two excited
noise conditions, including cold starts. Collective-only motion still had
regressions, and reconstruction worsened forecasts with the current ARP model.
A separate 400-step fitting probe tested eliminating the local state variables
before each parameter update. These remain research variants; production
defaults are unchanged.

[Comparison report](investigations/state-reconstruction/report.json) and
[prototype, inputs and arrays](investigations/state-reconstruction/evidence.zip).

## Observation and transition error

Adjacent one-step residuals share an observation. A follow-up likelihood
prototype modeled that dependence, recovering injected observation-noise scales
within 5% with the known plant and matching a dense Gaussian conditioning check.
It largely retained the observed ARP states. However, jointly fitting dynamics
and noise produced 5.6–23.6 times the current model's one-second ARP forecast
MSE at 400 steps.
Reconstructing only training-window starts also failed to improve every group.

The multi-step rollout objective remains in place. Diagnostics now distinguish
adjacent from longer-lag correlation and preserve timing across missing
intervals. Full-batch fitting retains its best evaluated iterate after the
refit probe exposed final-step overshoot.

[Pilot report](investigations/noise-separation/report.json) and
[executed sources and arrays](investigations/noise-separation/evidence.zip).
