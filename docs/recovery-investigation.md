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
