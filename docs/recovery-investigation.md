# Recovery after identification: investigation

The recovery regression comes primarily from charging a local linearized
uncertainty model for parameter directions that 0.8 seconds of telemetry barely
identifies. Increasing optimizer effort helps tracking, but the uncertainty
penalty still nearly doubles small-disturbance error with a well-converged
reference. Additional independent telemetry removes most of that gap without
changing the objective, its weights, or the covariance rank cutoff.

The [recorded investigation](investigations/recovery.json) contains 20 controlled
recoveries, nonlinear uncertainty probes, independent prediction errors, and
source fingerprints. Reproduce it with:

```bash
uv run python scripts/investigate_recovery.py \
  --output docs/investigations/recovery.json
```

This is an offline diagnostic. It retains the original recovery benchmark and
its negative result in [validation](validation.md#adaptive-recovery). The new
report is written by this script; it is separate from the manifest-owned
`docs/results/` artifacts.

## Separate uncertainty from optimization

The small disturbance starts at position `(0.025, -0.020, -0.015)` m, Euler
attitude `(0.024, -0.017, 0.012)` rad, zero velocity, and body rates
`(0.03, -0.03, 0.03)` rad/s. Every small-disturbance trace stays inside the
original validity envelope, with maximum utilization below 0.36.

Every ablation uses the same adapted parameters, disturbance, true plant,
0.6-second prediction horizon, 1.2-second recovery, tolerances, and objective
weights. Only the included uncertainty terms and optimizer differ. The point
ablations omit uncertainty for diagnosis; they are not saved as deployable
beliefs or treated as fully known models.

Lower recovery-tail normalized tracking RMS is better:

| Optimizer | Mean alone | Forecast error only | Parameter uncertainty only | Both uncertainty terms |
| --- | ---: | ---: | ---: | ---: |
| Maintained, 8 iterations | 0.05895 | 0.05895 | 0.11253 | 0.11253 |
| Projected descent, 128 iterations | 0.03746 | 0.03742 | 0.07519 | 0.07518 |
| Offline L-BFGS-B reference | 0.02889 | — | — | 0.05528 |

The longer projected run also tightens the relative-improvement tolerance
from `1e-5` to `1e-8`. The offline reference uses the maintained JAX objective
and derivatives, the same bounded command variables and warm-start selection,
and up to 500 L-BFGS-B iterations. It reports convergence only when the same
projected-gradient criterion passes. All 60 mean-only small-disturbance solves
converge; 58 of 60 uncertainty-enabled solves converge, and the other two have
projected residuals at most 0.00257 against the 0.002 criterion. Thus substantial
under-optimization is not a sufficient explanation for the remaining gap.

Forecast-error covariance is fixed by horizon in this case. Its tracking
contribution is independent of the commands, and its validity penalty remains
inactive in these small-disturbance runs. Parameter uncertainty accounts for
the material difference.

## Full rank does not establish a useful local covariance

After the short adaptation, all 15 estimable directions are resolved, but the
largest normalized parameter standard deviation is 52.714. Marginal standard
deviations of the three log angular-damping parameters are approximately 20.3,
27.3, and 52.6. A linearization around the mean cannot be assumed accurate over
such large log-parameter displacements.

At the small initial disturbance and a held hover command, three damping-heavy
covariance directions contribute 99.797% of the parameter tracking-spread
penalty. The probe evaluates both signs of every column of the covariance
factor, giving 30 points on the one-Mahalanobis-radius ellipsoid. Four nonlinear
rollouts become nonfinite, all in those three directions, while their JVP-based
covariance contributions remain finite. Even the finite opposite-sign probes
in the two largest-cost modes differ from the linear approximation by about
96% and 98% relative to the linearized tangent-error norm.

This demonstrates failure of the local approximation as a general uncertainty
bound for this belief. It does not establish that the physical vehicle would
diverge: the nonlinear model and its explicit numerical integration are also
being evaluated at extreme coefficients. These deterministic probes are not
posterior samples or a probabilistic coverage test.

## More identifying evidence resolves the observed regression

The follow-up uses six new two-second telemetry blocks, seeds 31 through 36.
Only transitions inside the existing validity envelope are absorbed: 582 of
600. No telemetry from the independent evaluation trajectory (seed 22) is used
in the updates. The parameter rank remains 15 throughout, illustrating why
rank alone cannot distinguish the short adaptation from the better-informed
belief.

After these updates:

- Maximum normalized parameter standard deviation falls from 52.714 to 0.345.
- All 30 nonlinear uncertainty probes are finite. The dominant mode agrees
  with its local approximation to about 0.02%; the next two have relative
  discrepancies up to about 17% and 5%. This is improved local behavior, not a
  calibration claim.
- Independent 0.6-second normalized prediction RMS falls from 0.028970 after
  the short adaptation to 0.002004. The stale value was 0.033394.

The two control arms now agree closely:

| Disturbance and optimizer | Updated mean alone | Updated belief with uncertainty |
| --- | ---: | ---: |
| Small, maintained 8 iterations | 0.05300 | 0.05302 |
| Small, offline L-BFGS-B | 0.02819 | 0.02824 |
| Original, maintained 8 iterations | 0.78830 | 0.78851 |
| Original, offline L-BFGS-B | 0.53660 | 0.53665 |

The maintained uncertainty-enabled controller's original-disturbance result
improves from 1.43857 after the short adaptation to 0.78851 after the additional
evidence. The original benchmark's oracle point arm was 0.78751. The original
disturbance still leaves validity support: additional evidence does not turn
that stress case into an accepted operating condition. Its offline reference
also has many stalled solves, so it is not a converged optimum or a safety
result.

## Architectural decision

Keep evidence assimilation separate from permission to rely on a belief for
control. `absorb` should retain usable evidence; `ParameterInformation.complete`
means the precision resolves the estimable subspace. Neither statement proves
nonlinear uncertainty calibration or control suitability.

For this workflow, qualify an adapted belief by checking independent prediction
error, nonlinear behavior over its retained uncertainty directions, and
representative closed-loop performance inside its declared validity envelope.
A nonfinite uncertainty probe is a concrete reason to withhold promotion.
Finite probes are only a necessary check; application-specific error and
validity requirements still need to be met. The short-adaptation belief fails
this qualification; the additional-evidence belief supports the tested small
recovery, while the original disturbance remains outside the qualified domain.

The supported remedy here is more informative telemetry. Where telemetry is
limited, a finite prior must come from declared physical or fleet knowledge.
Deleting weak covariance directions or inserting an arbitrary narrow prior
would remove the evidence of the problem. The existing objective and solver
remain unchanged. Improving solver efficiency is a separate opportunity,
supported by the offline reference, rather than the explanation for this
uncertainty regression.

## Supervised recovery and model support

The [supervised investigation](investigations/supervised-recovery.json) drives
the production `run_control_loop` and `MultirotorFlightSupervisor` against the
same synthetic target, after the additional identification evidence above.
It records twelve cases, support exits by feature, counterfactual prediction
errors, solver outcomes and supervisor transitions. Reproduce it with:

```bash
uv run python scripts/investigate_supervised_recovery.py \
  --output docs/investigations/supervised-recovery.json
```

Each recovery lasts 2.4 seconds, with tracking RMS measured over the final
0.4 seconds. These tails therefore differ from the earlier 1.2-second runs.
The supervisor uses a discrete clock shared with observation timestamps.
Nominal offline solves have no CPU deadline; durations remain recorded as host
time. The controller objective, gains, iteration budget and physical supervisor
limits are unchanged. No command reaches hardware.

### An unsupported nominal plan was allowed through

Previously, the supervisor saw `command_usable` but no model-support diagnostic.
A numerically usable plan could leave the declared envelope while staying
within the independent physical attitude and rate limits. In
`scenarios[name=original_without_supervisor]`, the first unsupported forecast
appears at 0.02 seconds and predicts a roll-rate exit 0.18 seconds ahead.
The actual state first crosses at 0.22 seconds and reaches utilization 1.136139.

The loop now forwards `maximum_validity_utilization`, including the initial
state, and the supervisor withholds nominal commands for unknown or exceeded
support. A missing, negative or nonfinite value records `MODEL_SUPPORT_UNKNOWN`;
a value above `1 + 1e-6` records `MODEL_SUPPORT_EXCEEDED`. Direct callers must
provide this input, and custom loop supervisors must accept the keyword.
This is the mean trajectory's support; the objective separately charges
covariance-expanded utilization. Neither is a physical safety certificate.

The counterfactual check propagates each usable candidate's commands through
the true synthetic model, starting from that candidate's initial state and
actuator state. It compares planned states with that counterfactual rollout,
not with the later trajectory after replanning or intervention. For the
unsupervised original case, normalized prediction RMS is 0.003712 inside
support and 0.004254 outside it (`counterfactual_planned_prediction_error`).
The modest error increase does not establish a model breakdown at the box
edge, or justify ignoring its declared boundary.

### Arrest does not preserve model support

With the original envelope, intervention keeps roll rate inside support but
lateral body velocity exits at 0.40 seconds, reaching utilization 1.089833.
The final tracking RMS increases from 0.505660 without supervision to 1.123089
with it. These are `original_without_supervisor` and
`original_with_supervisor` in the report. Withholding an unsupported plan
does not establish that the substitute command preserves support.

The script also tests whether independent roll excitation supplies the missing
coverage. Two six-second calibration flights, at phases 0 and pi/2, extend
the original envelope only to observed body-velocity and body-rate extrema.
They do not change the fitted mean or covariance. Recovery traces and held-out
flights never contribute to those bounds. The resulting marginal box does
not establish coverage of every combination of its features.

Independent 0.35-radian profiles at phases pi/4 and 3pi/4 still leave this box,
reaching utilization 1.153915 and 1.013899. Their prediction errors pass, but
their coverage fails. The report retains this result under
`unqualified_higher_amplitude_validation`. Fresh, narrower 0.25-radian profiles
at phases pi/6 and 5pi/6 remain inside it; their worst component endpoint RMS
is 0.041334 and 0.064972 times the tracking tolerance. The declared limit is
0.10 per component over independent 0.6-second windows (`coverage_validation`).
This narrower check was added after the larger profiles failed; it supports
only those tested profiles, without retroactively qualifying the larger ones.

Even after this calibration, the original recovery's first unsupported
forecast is now in pitch rate, and arrest still leaves support in lateral
velocity. Its maximum actual utilization is 1.059398 and tail RMS is 1.124988
(`calibrated_original_recovery`). All original-disturbance fault cases also
leave support. They end with attitude and rates inside the tracking tolerances
and return to nominal, but fail the requirement to remain within model support.

### Supported small recovery survives the injected faults

The small disturbance uses the same initial state as the earlier investigation.
Individual faults occupy intervals 10 through 12: stale observations deliver
the actual state and actuator state from three ticks earlier, with reception
timestamps 60 ms old; deadline faults pass a negligible solver budget and
produce `deadline_exceeded`; unresolved-evidence faults use a rank-zero belief
over the same model and produce `unresolved_model`. The combined case separates
these faults into intervals 10–12, 30–32 and 50–52. These are delivered stale
observations and solver failures; link read exceptions are not exercised.

For the calibrated envelope, the report's `calibrated_small_*` scenarios show:

| Case | Tail tracking RMS | Peak actual support utilization | Nominal / arrest / hold intervals |
| --- | ---: | ---: | ---: |
| No fault | 0.035306 | 0.306185 | 120 / 0 / 0 |
| Stale state | 0.033893 | 0.306068 | 113 / 4 / 3 |
| Deadline exceeded | 0.034049 | 0.306092 | 114 / 6 / 0 |
| Unresolved parameters | 0.034049 | 0.306092 | 114 / 6 / 0 |
| Combined | 0.036462 | 0.305277 | 102 / 15 / 3 |

Every case has finite states, bounded commands, supported actual states,
terminal attitude and rates within tracking tolerances, and a final nominal
decision. Every accepted nominal command has fresh telemetry, a usable solver
result and a supported forecast. The report records these checks explicitly;
the stale case selects collective hold, and solver refusal selects latched
rate arrest. The slightly lower RMS in some fault cases is specific to these
traces and does not establish that intervention improves tracking generally.

The next unresolved issue is the fallback's recovery region. Its geometric
arrest regulates tilt and rates but does not regulate lateral velocity or
position. Before treating the original disturbance as supported, that fallback
needs an independently evaluated operating region and a defined response when
it cannot preserve that region. Expanding the nominal model's envelope alone
did not resolve this failure.

## PX4 integration defects found during validation

Two integration issues were independently reproduced and fixed:

1. Shadow mode used the model's 20 ms sample period as its wait for fresh
   telemetry. A passive stream can arrive later even when the received sample
   is fresh. Shadow telemetry now has a separate bounded timeout, defaulting
   to one second; the solver retains its 20 ms deadline and reception timestamps
   remain intact. The general loop keeps its existing timeout by default.
2. The flown-profile fixture invoked `glassbox.io.sitl_profile`, which no longer
   has a command entry point and exited without flying a profile. It now invokes
   the current CLI, observes the driver's first excitation target, and compiles
   the controller before the finite maneuver starts. Excitation thresholds are
   unchanged, so takeoff motion alone cannot substitute for the requested
   maneuver.

The shadow tests use the pinned PX4 SIH image and the existing
`artifacts/sitl/px4_runtime_model.json`, loaded and re-saved through the current
artifact format. The model is a legacy point comparison; these tests validate
telemetry, actuator alignment, command bounds, and deadline behavior, not
closed-loop command authority or the learned covariance's calibration.

All four flown profiles and the canonical state-stream test passed; the
separate fixed-command shadow test also passed. The
[recorded flown-profile summaries](investigations/px4-shadow.json) cover 640
shadow solves. Commands remained bounded, actuator/state skew stayed at or
below 4 ms, and maximum recorded state receive age was 52 ms. One solve missed
the 20 ms deadline and returned the explicit bounded hold; the other 639
returned finite plans at the iteration limit. Those are usable plans under the
solver contract, not converged solutions or a hard real-time guarantee.
