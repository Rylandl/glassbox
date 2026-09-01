# Cascade Skywalker X8: an unfitted physics model against the NTNU flight campaign

Date: 2026-09-01. Reproduce with:

```bash
uv sync --inexact --extra cascade
uv run glassbox-x8 extract-dataset artifacts/x8_reference/raw artifacts/x8_cascade/canonical
uv run glassbox-x8 evaluate-cascade artifacts/x8_cascade \
  --report artifacts/x8_cascade/cascade_report.json \
  --reference-report artifacts/x8_reference/benchmark_report.json
```

## What was evaluated

[Cascade](https://github.com/Rylandl/cascade) is a differentiable JAX fixed-wing simulator whose
canonical state boundary is Glassbox's own 13-vector. Its Skywalker X8 is assembled from the
published NTNU model: Gryte et al. (ICUAS 2018) wind-tunnel statics and XFLR5 rate derivatives,
Reinhardt et al. (ICUAS 2022) bifilar-pendulum inertia, the pyfly stall blend, and an exact map of
the NTNU exit-velocity propulsion law. Nothing in it was fitted to this campaign. Each rolling
window starts from the measured state, holds each logged control for one 40 Hz interval with ten
RK4 sub-steps, keeps actuators at their equilibrium for the last control before the window, and
holds the typed wind at the window start. The protocol, horizons, metrics, and persistence
baseline are exactly those of `glassbox-x8 evaluate`.

Rows other than the primary one vary documented uncertainties of the published model, applied
identically to every window and never tuned per window: the pitching-moment reference point
(Gryte et al. warn a 30 mm CG shift moves the trim alpha from 11 to 3.25 degrees; the NTNU
repository's "flight-tuned" pitch triple is exactly the wind-tunnel triple moved 30 mm forward),
the mass (3.364 kg in the NTNU parameter file, about 4 kg per EUCASS 2019), the yaw damping
(XFLR5 value or the repository's sixfold value), and the fraction of the campaign's inferred
vertical wind that is applied.

## Result on the four untouched validation maneuvers

Position / velocity / attitude / body-rate RMSE, equal weight per maneuver.

| Model | 0.1 s | 0.5 s | 2 s | Score vs persistence |
| --- | --- | --- | --- | ---: |
| Kinematic persistence | 0.008 m / 0.17 m/s / 1.2° / 0.22 rad/s | 0.12 m / 0.56 m/s / 12.8° / 0.56 rad/s | 0.86 m / 1.00 m/s / 42.9° / 0.58 rad/s | 1.000 |
| Glassbox structured (fitted) | 0.006 / 0.14 / 0.9° / 0.16 | 0.08 / 0.38 / 4.3° / 0.22 | 0.53 / 0.63 / 7.3° / 0.23 | 0.510 |
| Glassbox structured residual (fitted) | 0.006 / 0.12 / 0.8° / 0.13 | 0.06 / 0.26 / 3.4° / 0.18 | 0.43 / 0.59 / 7.4° / 0.18 | 0.414 |
| Cascade X8, published as-is (primary) | 0.022 / 0.50 / 1.7° / 0.35 | 0.52 / 2.83 / 17.3° / 0.70 | 7.79 / 9.28 / 72.0° / 0.68 | 2.766 |
| Cascade X8, CG +50 mm, 3.364 kg, ¼ vertical wind (best) | 0.007 / 0.14 / 1.5° / 0.28 | 0.07 / 0.39 / 6.9° / 0.31 | 1.01 / 1.53 / 16.5° / 0.31 | 0.735 |

The best unfitted physics row is 1.44 times the fitted structured model's score and 1.78 times
the residual's. At 0.1 s its position and velocity errors match the fitted models; its attitude
and body-rate errors do not.

Score against persistence over the two dominant axes, 3.364 kg, XFLR5 yaw damping:

| Forward CG shift | wind 0 | ¼ | ½ | ¾ | full |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 mm | 2.45 | 2.34 | 2.45 | 2.63 | 2.77 |
| 20 mm | 1.39 | 1.29 | 1.32 | 1.40 | 1.49 |
| 30 mm | 1.09 | 1.00 | 1.02 | 1.11 | 1.22 |
| 50 mm | 0.82 | 0.74 | 0.74 | 0.85 | 0.99 |

Yaw damping changes scores by under 0.01. A 4 kg mass helps at small CG shifts (0.83 at 30 mm and
half wind) and is equivalent at 50 mm.

## Findings

1. **The published model is untrimmed at the flight condition.** With the wind-tunnel pitch
   triple about its nominal CG, the pitching moment at the flown alpha and elevator is about
   +0.03 in `C_m`, roughly 10 rad/s² of nose-up acceleration, so the as-is model pitches away
   within a second. A 50 mm forward reference point removes the mean pitch bias at the wind
   fraction that balances lift; at the campaign's full vertical wind the required shift is about
   15 mm. This is the paper's own stated sensitivity, quantified.
2. **The campaign's vertical wind is inconsistent with the published lift curve.** At the
   measured load factor of 1.02 the model's mean vertical specific-force error is zero with 0.4 of
   the inferred 2.8 m/s vertical wind and +16 m/s² with all of it. The upstream method inferred
   the vertical component from pitot airspeed minus horizontal relative airspeed; a 1.2% pitot
   bias would produce the entire estimate. The typed wind should be treated as an estimate whose
   vertical component is uncertain by a factor of about two.
3. **A constant forward-force excess of about 2.7 N.** Either the clean wind-tunnel drag is low
   for the instrumented 2023 airframe, or the propulsion law's thrust is high; the residual grows
   with alpha, which favours drag.
4. **Angular-rate residuals point at the rate signals, not one coefficient.** Regressing the roll
   and pitch acceleration residuals on the flight variables (R² 0.88 and 0.87) makes every damping
   and control derivative look several times too strong at once, including a positive apparent
   pitch damping, which is not physical. The adapter takes body rates from the ArduPilot EKF
   states; the raw 200 Hz IMU gyro columns are in the CSV and would give unfiltered angular
   accelerations. The X8 aileron scaling (half-difference versus full-difference of the elevons)
   is the second suspect for the roll channel.

## Boundary

This is characterization evidence for the simulator and the campaign, not a candidate under the
fixed-wing development contract, and the sensitivity rows are not a fit. The next evidence-bearing
steps are on the data side (IMU rates, pitot calibration) and on Cascade's side (a component-panel
X8 fitted to the coefficient backend, which would test whether rate derivatives follow from
geometry). See [the recorded result](cascade-x8-validation-results.json).
