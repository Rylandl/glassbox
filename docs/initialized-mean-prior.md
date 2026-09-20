# Initialized-mean prior: rejected

The fixed initialized-mean penalty failed its frozen progress criterion.
Forecast error fell **1.53%**, response error rose **4.45%**, and their joint
ratio was **1.01415**, above the required **0.97**. All regression limits and
verification checks passed. The retained public `generic-memory-v4-prototype`
recipe and both saved public revisions remain unchanged; this research loss
was not integrated.

The [protocol](harness/initialized-mean-prior-v1.json) was committed before
fitting. The candidate reused the exact 1,536 training windows from 144 parents,
256 development windows from 24 parents, initialization, normalization and fixed
channel weights of the preceding independent-recording update. It added
`0.25 × mean(w × ((prediction − initial_prediction) / scale)²)` to the existing
training loss and safeguarded acceptance objective. The initial prediction was
already available from weight construction. Both candidates used 1,000 full-cache
gradient attempts and selected step 1,000 using unchanged data-only development
loss. No training collection, baseline refit, coefficient sweep or retry occurred.

The symmetric rule required the geometric mean of the forecast and response
ratios to be at most 0.97, with each at most 1.05. Existing primary, shifted-scope,
parent-p95 and Crazyflow angular limits also applied. Only the joint progress
check failed; the rule did not require every cell to improve.

| Simulator | Forecast aggregate change | Response aggregate change | Worsening endpoint cells |
| --- | ---: | ---: | ---: |
| Crazyflow | −6.34% | −2.29% | 25/90 |
| Cascade | +3.52% | +11.66% | 68/90 |

Twelve of twenty simulator/scope/kind aggregates improve. All ten Crazyflow
aggregates improve, while every Cascade response scope worsens. **93/180**
endpoint cells have larger absolute RMSE; ten of twelve primary parent-p95
errors also rise, within the frozen limits. These counts use absolute changes,
so normalization floors cannot hide small losses.

Primary 250 ms RMSE, retained public revision → candidate:

| Simulator | Forecast velocity, m/s | Forecast rate, rad/s | Response velocity, m/s | Response rate, rad/s |
| --- | ---: | ---: | ---: | ---: |
| Crazyflow, available truth | 0.03644 → 0.03878 | 0.12377 → 0.10596 | 0.02015 → 0.02105 | 0.13865 → 0.13954 |
| Cascade | 0.06630 → 0.07130 | 0.04312 → 0.04411 | 0.01977 → 0.02272 | 0.01845 → 0.02035 |

The [compact result](harness/initialized-mean-prior-v1-result.json) includes
rotation errors, all twenty scope ratios, all twelve guarded tails and exact
references to every endpoint loss in the sealed decision. Descriptive paired
parent-bootstrap 95% intervals are 0.96801–1.00098 for forecast and
1.03219–1.05625 for response, conditional on available truth; they add no veto.

Fresh confirmation used the frozen +13M seeds after both fits were sealed.
Crazyflow completed 63/84 parents, with 21 altitude failures; Cascade completed
84/84. Crazyflow primary 250 ms forecast/response truth covers 434/480 and
712/768 planned comparisons. All arms, including hold, use identical available
truth and query masks. Different completion counts on earlier cohorts are not
model improvements.

Saved-cache diagnostics show a partial effect on the named gap. Crazyflow
250 ms development angular RMSE falls from 0.13045 to 0.11502 rad/s versus the
selected control, but remains above its 0.09784 initialization value; all 25
native angular horizons still worsen versus initialization. Cascade development
weighted data loss is 13.60% worse than the retained control. The penalty reduces some angular drift while
also constraining useful learning. This fixed mechanism does not establish a
general forecasting or response improvement.

| Actual work per candidate | Crazyflow | Cascade |
| --- | ---: | ---: |
| Fitter wall time, including capture | 318.91 s | 42.80 s |
| Calibration wall time | 0.44 s | 0.81 s |
| Full-training acceptance objective calls | 5,318 | 2,567 |
| Accepted / rejected attempts | 988 / 12 | 997 / 3 |

Each fit made 1,536,000 gradient-window visits, one initializer call, two ridge
solves and one initial-reference forecast. Equal attempt/window budgets do not
mean equal FLOPs or elapsed time. The prior shared the training forward pass.

Both exact no-fit replays passed: 53,988 native arrays, 12,772 prediction/envelope
arrays and 22 saved checkpoint snapshots were checked. Independent NumPy
reduction reproduced 257,040 raw metric rows; all eight frozen alteration cases
were rejected by their external anchors and intended semantic checks. The final
experiment suite passed 101 tests after the earlier 238-test combined run;
Ruff, formatting and diff checks passed. Protocol, source archive, logs, costs
and all external SHA anchors are in the result record and
[durable evidence inventory](../artifacts/2026-09-20/initialized-mean-prior-v1-evidence/copy-inventory.json).

This rejected prepared-cache research fit adds no public lifecycle,
uncertainty-coverage, derivative or controller qualification. The retained
baseline remains the accuracy reference. The initial paired-response proposal
was not frozen or run. The user's subsequent clarification makes the
[Dart general-vehicle plan](dart-general-vehicle-plan.md) the next priority.
