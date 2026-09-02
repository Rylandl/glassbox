# Cross-Airframe Fixed-Wing Development Gate

**What this establishes:** the structured residual is selected for continued development over the structured baseline, with an equal-airframe candidate/reference score of 0.791 and every complete rollout finite. It does not yet pass the accuracy contract: IDF p90 position error at 0.5 seconds is 0.2523 m against a 0.2500 m limit, while X8 passes every contract requirement.

## Purpose

The versioned `fixedwing_prediction_development_v1` contract prevents progress on one airframe from hiding regressions on another. It gives the conventional IDF aircraft and the X8 flying wing equal weight and requires each airframe to meet both equal-flight aggregate and p90 flight errors.

## Data

| Horizon | Maximum position RMSE | Maximum attitude RMSE |
| ---: | ---: | ---: |
| 0.5 s | 0.25 m | 6 deg |
| 1.0 s | 0.60 m | 9 deg |
| 2.0 s | 1.20 m | 13 deg |

Every airframe must also beat constant-world-velocity, constant-body-rate kinematic persistence by at least 5% over the 0.5, 1, and 2-second metrics, and every complete rollout must remain finite. These are held-out prediction development targets, not flight-safety or certification limits. Candidate selection is separate from acceptance: a candidate must improve at least 1% overall, may not regress either airframe by more than 5%, may not regress an individual metric by more than 50%, and must keep all complete rollouts finite.

This gate consumes the [IDF-DS](idf.md) leave-one-source-group-out summary and the [X8](x8.md) benchmark report as its two airframe inputs.

## Reproduce

Evaluate the two model classes and make the selection:

```bash
uv run glassbox-fixedwing-gate evaluate \
  --idf-summary artifacts/idf_reference/source_benchmark_lateral_cross_coupling_structured_v3/summary.json \
  --x8-report artifacts/x8_reference/benchmark_report.json \
  --x8-model-name structured \
  --candidate-name structured_v3 \
  --output artifacts/fixedwing_cross_airframe/structured_v3_gate.json

uv run glassbox-fixedwing-gate evaluate \
  --idf-summary artifacts/idf_reference/source_benchmark_lateral_cross_coupling_residual_v3/summary.json \
  --x8-report artifacts/x8_reference/benchmark_report.json \
  --x8-model-name structured_residual \
  --candidate-name structured_residual_v3 \
  --output artifacts/fixedwing_cross_airframe/residual_v3_gate.json

uv run glassbox-fixedwing-gate compare \
  --reference artifacts/fixedwing_cross_airframe/structured_v3_gate.json \
  --candidate artifacts/fixedwing_cross_airframe/residual_v3_gate.json \
  --output artifacts/fixedwing_cross_airframe/selection_v1.json
```

The second `evaluate` call uses `--x8-model-name structured_residual`, not `residual`: the X8 benchmark report names its two models `structured` and `structured_residual` (see [x8.md](x8.md)), and `glassbox-fixedwing-gate` looks the name up directly in the report's `models` object.

Use the fail-fast single-airframe screen before paying for a full cross-airframe fit:

```bash
uv run glassbox-fixedwing-gate screen \
  --reference-report artifacts/x8_reference/benchmark_report.json \
  --candidate-report artifacts/x8_reference/candidate/benchmark_report.json \
  --model-name structured_residual \
  --airframe-name x8_flying_wing \
  --candidate-name candidate \
  --output artifacts/fixedwing_cross_airframe/candidate_screen.json
```

The screen uses the contract's 0.5, 1, and 2-second metrics, requires at least 1% aggregate improvement, and rejects any individual metric regression above 50%. Passing only authorizes the more expensive cross-airframe evaluation; a single airframe can never promote a model.

## Results

The residual is selected for continued development. Its candidate/reference score is 0.780 on IDF and 0.802 on X8, for an equal-airframe score of 0.791; median stable rollout duration also increases from 2.82 to 3.46 seconds on IDF and from 1.34 to 1.75 seconds on X8. All complete rollouts remain finite. It does not yet pass the accuracy contract: IDF p90 position error at 0.5 seconds is 0.2523 m against a 0.2500 m limit. X8 passes every contract requirement. The miss is reported rather than rounded away or used to relax the threshold.

The divergence diagnostics make candidate hypotheses concrete: IDF failures are dominated by velocity error, while three of four X8 validation maneuvers first cross the angular-rate threshold. They do not, by themselves, prove which extra coefficient belongs in the shared model.

This screen rejected four physically plausible follow-up experiments before running IDF's 13 folds:

| Candidate | X8 candidate/reference score | Decision |
| --- | ---: | --- |
| surface/rate force plus cross-rate moment derivatives | 1.176 | reject |
| cross-rate moments only | 1.133 | reject |
| cross-rate moments with a frozen-base staged residual | 1.056 | reject |
| v3 with a frozen-base staged residual | 1.011 | reject |

Lower is better. The accepted joint residual v3 is therefore retained.

## Boundary

The negative screening result is useful: residual Jacobians are not sufficient evidence for promoting a structured term because the base and residual co-adapt during fitting. The next broadly useful evidence should come from another independent airframe or better aerodynamic observations such as trusted airspeed and wind, not from tuning more coefficients against IDF or X8.
