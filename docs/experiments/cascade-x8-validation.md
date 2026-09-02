# Cascade Skywalker X8: an unfitted physics model against the NTNU flight campaign

Date: 2026-09-01. Reproduce with:

```bash
uv sync --inexact --extra cascade
uv run glassbox x8 extract-dataset artifacts/x8_reference/raw artifacts/x8_cascade/canonical
uv run glassbox x8 evaluate-cascade artifacts/x8_cascade \
  --report artifacts/x8_cascade/cascade_report.json \
  --reference-report artifacts/x8_reference/benchmark_report.json \
  --cg-shifts 0,0.03,0.05 --inertia-scales 1,2,3.5 --vertical-wind-fractions 0.25,0.5,1
uv run glassbox x8 diagnose-cascade artifacts/x8_cascade --split all \
  --cg-shift 0.05 --vertical-wind-fraction 0.4 --inertia-scale 1
```

## What was evaluated

[Cascade](https://github.com/Rylandl/cascade) is a differentiable JAX fixed-wing simulator whose
canonical state boundary is Glassbox's own 13-vector. Its Skywalker X8 is assembled from the
published NTNU model: Gryte et al. (ICUAS 2018) wind-tunnel statics and XFLR5 rate derivatives,
Reinhardt et al. (ICUAS 2022) bifilar-pendulum inertia, the pyfly stall blend, and an exact map of
the NTNU exit-velocity propulsion law. Nothing in it was fitted to this campaign. Each rolling
window starts from the measured state, holds each logged control for one 40 Hz interval with ten
RK4 sub-steps, carries actuators that have integrated their lag over the logged control history,
and holds the typed wind at the window start. The protocol, horizons, metrics, and persistence
baseline are exactly those of `glassbox x8 evaluate`.

Rows other than the primary one vary documented uncertainties of the published model, applied
identically to every window and never tuned per window:

- the pitching-moment reference point, as a forward CG shift through the lever-arm transform
  (Gryte et al. warn a 30 mm shift moves the trim alpha from 11 to 3.25 degrees; the NTNU
  repository's "flight-tuned" pitch triple is exactly the wind-tunnel triple moved 30 mm forward);
- the mass (3.364 kg in the NTNU parameter file, about 4 kg per EUCASS 2019);
- a uniform scale on the inertia tensor (the pendulum tensor is for a bare airframe; the older
  parameter file lists a roll inertia 3.7 times larger; the 2023 instrumented airframe is
  undocumented);
- the fraction of the campaign's inferred vertical wind that is applied.

## Result on the four untouched validation maneuvers

Recorded on 2026-09-01 with lag-integrated actuator initialization, rollout error
statistics that exclude the shared initial sample (`metric_policy` v2), and fitted
reference models trained under `deterministic_weighted_minibatch_v3`.

Position / velocity / attitude / body-rate RMSE, equal weight per maneuver.

| Model | 0.1 s | 0.5 s | 2 s | Score vs persistence |
| --- | --- | --- | --- | ---: |
| Kinematic persistence | 0.008 m / 0.19 m/s / 1.3° / 0.24 rad/s | 0.12 m / 0.58 m/s / 13.1° / 0.58 rad/s | 0.87 m / 1.01 m/s / 43.2° / 0.58 rad/s | 1.000 |
| Glassbox structured (fitted) | 0.007 / 0.16 / 1.0° / 0.18 | 0.08 / 0.38 / 4.4° / 0.22 | 0.53 / 0.63 / 7.2° / 0.23 | 0.507 |
| Glassbox structured residual (fitted) | 0.006 / 0.14 / 0.9° / 0.15 | 0.06 / 0.28 / 3.7° / 0.19 | 0.49 / 0.68 / 8.7° / 0.19 | 0.438 |
| Cascade X8, published as-is (primary) | 0.03 / 0.56 / 1.8° / 0.38 | 0.53 / 2.90 / 17.7° / 0.71 | 7.84 / 9.34 / 72.4° / 0.68 | 2.760 |
| Cascade X8, CG +50 mm, 4 kg, inertia ×2, ½ vertical wind (best) | 0.006 / 0.14 / 1.1° / 0.21 | 0.07 / 0.40 / 6.7° / 0.29 | 1.05 / 1.47 / 14.9° / 0.31 | 0.679 |

The best unfitted physics row is 1.34 times the fitted structured model's score and 1.55 times
the residual's. At 0.1 s it matches the fitted models on all four metrics; the gap opens with
horizon and is carried by attitude and body rate, which points at the damping derivatives below.

Score against persistence, 3.364 kg, XFLR5 yaw damping, columns ¼ / ½ / full vertical wind:

| Forward CG shift | inertia ×1 | inertia ×2 | inertia ×3.5 |
| ---: | --- | --- | --- |
| 0 mm | 2.34 / 2.45 / 2.76 | 2.01 / 2.09 / 2.41 | 1.76 / 1.82 / 2.14 |
| 30 mm | 1.00 / 1.01 / 1.22 | 0.94 / 0.95 / 1.15 | 0.96 / 0.97 / 1.18 |
| 50 mm | 0.73 / 0.74 / 0.99 | 0.69 / 0.70 / 0.93 | 0.74 / 0.74 / 0.95 |

At 4 kg the 50 mm rows are 0.77 / 0.73 / 0.93, 0.72 / 0.68 / 0.87, and 0.75 / 0.71 / 0.88.
Yaw damping changes scores by under 0.01. The inertia axis has an interior optimum near ×2.

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
   bias would produce the entire estimate. There is no independent airspeed channel to check:
   the CSV's `IndicatedSpeed` is by construction the inertial velocity minus the 3D wind, and its
   `TrueSpeed` column tracks GPS ground speed to 0.1 m/s. The typed wind should be treated as an
   estimate whose vertical component is uncertain by a factor of about two.
3. **Constant force offsets of a few newtons.** Predicted minus measured: +2.7 N forward,
   −2.6 N to the right, −2.7 N downward (slightly too much lift at 3.364 kg and 0.4 wind). The
   forward excess grows with alpha, which favours the clean wind-tunnel drag being low for the
   instrumented 2023 airframe over the propulsion law being high.
4. **The rotational dynamics disagree with the published derivatives, and it is not a data
   artifact.** Raw IMU gyro rates match the EKF rate states to 0.08 rad/s rms, so finite-difference
   angular accelerations are trustworthy. Lag-aware equation-error regressions
   (`glassbox x8 diagnose-cascade`, all 17 maneuvers, R² 0.79 roll and 0.84 pitch) with the
   pendulum inertia give flight-effective aileron effectiveness of about 0.06 against the
   wind-tunnel 0.12, roll damping about −0.10 against XFLR5's −0.404, pitch damping about −0.46
   against −1.3, and elevator effectiveness about −0.16 against −0.21. Doubling the inertia
   reconciles the aileron term and is the best-scoring inertia in the grid, but the damping terms
   stay two to four times too strong. Read together: the instrumented airframe's inertia is about
   twice the bare-airframe pendulum value, and the XFLR5 damping derivatives are too large for
   this airframe. Both are testable outside this campaign, by weighing and swinging the 2023
   airframe and by a component-model prediction of the damping from geometry.

## The component-panel X8

Cascade also ships `skywalker_x8_panels`, a from-geometry component model (center body, swept
inner and outer panels per side with flapped elevons, tip winglets) whose static coefficients are
fitted to the published polynomial and whose rate derivatives are predictions from geometry
(Cascade `docs/skywalker-x8.md`). Run with `--aircraft skywalker_x8_panels`, on the same grid:

| Model, best variant (CG +50 mm, 4 kg, inertia ×2, ½ vertical wind) | 0.5 s | 2 s | Score |
| --- | --- | --- | ---: |
| Coefficient table | 0.073 m / 0.40 m/s / 6.7° / 0.29 rad/s | 1.052 m / 1.47 m/s / 14.9° / 0.31 rad/s | 0.679 |
| Component panels | 0.065 m / 0.37 m/s / 6.5° / 0.29 rad/s | 1.065 m / 1.60 m/s / 17.6° / 0.32 rad/s | 0.675 |

As published (no shift, pendulum inertia, full wind) the panels score 2.19 against the table's 2.76.
On its own grid the panels' best variant is CG +50 mm, 3.364 kg, inertia ×2, ¼ vertical
wind at 0.661, within 0.02 of the table's best. The one-step residual regressions separate the two channels: the panel model's roll
moment residual is 2.0 N m rms against 3.4 N m for the table, with its aileron and dihedral
terms essentially matching flight and only mild excess damping (its geometric `C_lp` of −0.29
sits between XFLR5's −0.40 and the flight-effective value), whereas its pitch damping from
geometry (`C_mq` −2.6) is further from flight than XFLR5's −1.3. Over two seconds the pitch
channel dominates attitude error, so the roll improvement does not show in the score. Rate
derivatives from geometry are therefore a mixed but informative prediction: right in roll,
wrong in pitch for a tailless wing whose damping this panel layout overstates.

## Boundary

This is characterization evidence for the simulator and the campaign, not a candidate under the
fixed-wing development contract, and the sensitivity rows are not a fit. The next evidence-bearing
steps are on the data side (an independent airspeed reference, the instrumented airframe's
inertia) and on Cascade's side (a panel layout or unsteady term that reproduces the flying
wing's weak pitch damping). See
[the recorded result](../results/cascade-x8-validation-results.json).
