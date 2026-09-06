# Recovery after identification: investigation

The uncertainty, supervision and SLSQP reports below are historical snapshots
from commits `0c4ec6a`, `777652f` and `0926d49`, respectively. Their source
fingerprints identify the code that produced the numbers. They precede the
command-bound derivative and warm-start corrections in the
[SQP follow-up](#faster-constrained-nmpc-and-correct-bound-derivatives).
The earlier commands rerun those experiments against the current code into
temporary files; reproducing the snapshots requires their recorded revisions.

The recovery regression comes primarily from charging a local linearized
uncertainty model for parameter directions that 0.8 seconds of telemetry barely
identifies. Increasing optimizer effort helps tracking, but the uncertainty
penalty still nearly doubles small-disturbance error with a well-converged
reference. Additional independent telemetry removes most of that gap without
changing the objective, its weights, or the covariance rank cutoff.

The [recorded investigation](investigations/recovery.json) contains 20 controlled
recoveries, nonlinear uncertainty probes, independent prediction errors, and
source fingerprints. Rerun the experiment with:

```bash
uv run python scripts/investigate_recovery.py \
  --output /tmp/glassbox-recovery.json
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
errors, solver outcomes and supervisor transitions. Rerun the experiment with:

```bash
uv run python scripts/investigate_supervised_recovery.py \
  --output /tmp/glassbox-supervised-recovery.json
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

The geometric arrest regulates tilt and rates but does not regulate lateral
velocity or position. Its recovery region remains unestablished, and expanding
the nominal model's envelope alone did not resolve this failure. This motivates
testing support constraints inside NMPC before developing a separate recovery
controller.

## Explicit support constraints in NMPC

The formulation has a mismatch: the objective permits trading model-support
excess against tracking cost, while the supervisor refuses the resulting plan.
The [constrained investigation](investigations/constrained-recovery.json) tests
whether NMPC alone can find a supported recovery under the original envelope.
It introduces no secondary controller and makes no production solver change.

```bash
uv run python scripts/investigate_constrained_recovery.py \
  --output /tmp/glassbox-constrained-recovery.json
```

Each arm uses the additional-evidence belief above, its original model-support
box, retained covariance, objective, horizon, command blocks and tracking
tolerances. The original disturbance is identical across arms. The comparison
changes the optimizer and adds explicit constraints in the SLSQP arms; it does
not enlarge the envelope, reduce uncertainty or adjust penalty weights.

The mean constraint requires every predicted feature utilization to remain at
most one. The robust constraint adds the existing marginal standard-deviation
radius for that feature, divided by its envelope half-width. This is the same
covariance expansion the current objective already penalizes. It is a local
uncertainty calculation, with no joint probability or invariant-set claim.
The initial state is fixed during optimization and its support is recorded
separately, including any violation inherited from actual preceding motion.

Two optimizer contracts need different treatment for this reference:

- Feasibility takes precedence over objective improvement. A feasible result
  may cost more than an infeasible seed. Only feasible candidates compete on
  objective; an infeasible result cannot replace an available feasible seed.
- Stationarity includes constraint multipliers. The reference records the
  projected Lagrangian gradient, complementarity and dual feasibility using
  SLSQP's multipliers. The ordinary command-box gradient need not vanish at
  a constrained optimum. Consequently the reference does not set the
  production `CONVERGED` flag for these solves.

A returned plan must satisfy its explicit constraints to the declared
`1e-6` numerical tolerance. If neither seed nor optimizer result is feasible,
the scenario ends before applying a command. There is no hidden hold or arrest
path. Nominal CPU deadlines are disabled, all timings remain recorded, and
the smaller SLSQP iteration budget is a separate diagnostic rather than an
equal-compute comparison with projected descent.

The driver supplies the last emitted command as `previous_command` and the
current actuator state as `applied_command`, consistently in every arm. The
preceding supervised experiment supplies the observed actuator value in both
positions through the production loop. Its metrics are therefore background
context; the direct comparisons here are the arms in this report.

For the original disturbance, the report's `scenarios` record:

| Optimizer and support treatment | Tail tracking RMS | Peak actual support utilization | Peak robust forecast utilization |
| --- | ---: | ---: | ---: |
| Maintained projected descent, soft cost | 0.505589 | 1.136126 | 1.259270 |
| Offline L-BFGS-B, same soft cost | 0.234357 | 1.495204 | 1.667946 |
| Offline SLSQP, explicit mean constraints | 0.293031 | 1.000749 | 1.049734 |
| Offline SLSQP, explicit robust constraints | 0.294711 | 0.999870 | 1.0000005 |

All four arms complete the recovery with finite states, bounded commands and
terminal attitude and rates within the tracking tolerances. More optimization
of the soft objective improves tracking while increasing the support excursion.
Mean constraints nearly eliminate it, but model mismatch still produces a small
actual exit. With the existing uncertainty radius included in the constraints,
the tested trajectory stays inside support. Its closest approach is narrow,
so this is evidence for that trajectory, not a qualified recovery region.

The small-disturbance robust reference also completes: tail tracking RMS is
0.012787, maximum actual utilization is 0.339054, and maximum robust forecast
utilization is 0.358299 (`slsqp_robust_small_100`). Its commands remain bounded
and terminal attitude and rates meet the tolerances. These results establish
that an NMPC-only supported recovery exists in the tested formulation, without
broadening the model domain or inserting a conventional recovery controller.

This reference does not establish production readiness. The original robust
arm's median solve consumes about 15 model intervals. Capping SLSQP at eight
iterations returns no feasible plan on the original cold start; that scenario
ends with zero executed intervals and a null tracking tail
(`slsqp_robust_original_8`). This failure is specific to those seeds and that
algorithm, rather than evidence that no feasible eight-iteration method exists.

SLSQP reports success for all 120 original robust solves, but only 90 have a
projected Lagrangian residual below the maintained `2e-3` gradient tolerance;
the largest residual is 0.383985. Feasible recovery is the supported finding.
The optimizer's success flag does not establish that every solve meets the
stricter stationarity check, and the artifact retains the residuals for review.

The next implementation target is constrained NMPC with feasible warm starts
and explicit feasibility and optimality diagnostics. The current `PlanModel`
interface exposes an objective and aggregate measurements; a constrained solver
needs the individual inequalities and their derivatives, with candidate
selection that preserves feasibility. Its cold-start and subsequent solve costs
must fit the runtime budget. These experiments supply a reference for that work;
the production soft-cost solver and its failure responses remain unchanged.

## Faster constrained NMPC and correct bound derivatives

The [SQP investigation](investigations/sqp-recovery.json) exposes two library
defects while testing a faster constrained optimizer. Both corrections now
apply to the maintained solver; the new SQP algorithm remains an offline
experiment. Reproduce this report without concurrent benchmark processes:

```bash
uv run python scripts/investigate_sqp_recovery.py \
  --output docs/investigations/sqp-recovery.json
```

### Derivatives at active command bounds

The planner clipped normalized variables to their box, mapped them into motor
commands, and clipped those commands again. The optimizer already projected
its variables into that box. At an exact bound, each redundant JAX clip
contributed a derivative factor of one half. Their product made the planner
report one quarter of the true derivative for a feasible inward move.

The command map is now an affine convex combination of the two endpoints.
It reproduces each endpoint exactly and preserves the inward derivative.
Projection remains the solver's responsibility, and output command bounds
are still checked before a result is usable. Regression tests compare the
map and the full rollout's derivatives with feasible inward finite differences,
including asymmetric command bounds.

### A warm start must advance the elapsed interval

The loop executes one model sample, but the old warm start advanced an entire
command block. For three-sample blocks, it discarded two unexecuted samples
as well. The corrected seed shifts the expanded plan one sample, extends the
last command, and averages within each new block. This is the least-squares
projection onto the new block layout. A truncated final block uses only its
actual samples. Projection alone does not establish nonlinear feasibility;
the constrained optimizer must still check the resulting plan.

### Curvature-aware constrained steps

The experiment expresses the same objective as squared residuals and obtains
their Jacobian together with the support-constraint Jacobian in a fused JAX
linearization. Tests check the residual form against the maintained objective
and its gradient with uncertainty and active penalties. Gauss-Newton curvature
plus a `1e-4` numerical regularization defines each quadratic subproblem.
Whitening its variables gives that subproblem identity curvature; SLSQP then
solves only the quadratic problem with linear constraints. A nonlinear merit
line search checks the candidate against the original objective and support.

The subproblem targets a `1e-5` interior numerical margin, while final
feasibility retains the `1e-6` tolerance. This slightly tightens the numerical
target without changing the declared model envelope or uncertainty. Every
evaluated feasible iterate is eligible for retention; a cheaper infeasible
iterate cannot replace it. No feasible output means no command is applied.
Neither feasibility nor a subproblem success flag is labeled convergence of
the nonlinear problem.

All cases use the same additional-evidence belief, original envelope, objective
and 2.4-second driver. Each controller is prewarmed, then its optimizer state is
reset before measurement. The SQP cold solve has eight updates; the subsequent
budget is the controlled variable. The report's `scenarios` record:

| Method | Executed intervals | Tail tracking RMS | Peak actual utilization | Median subsequent solve / model interval |
| --- | ---: | ---: | ---: | ---: |
| Full nonlinear SLSQP, 100 iterations | 5 / 120 | — | 0.520645 | 16.974* |
| SQP, eight subsequent updates | 120 / 120 | 0.295062 | 0.999860 | 2.449 |
| SQP, two subsequent updates | 120 / 120 | 0.295111 | 0.999860 | 0.902 |
| Two-update SQP with the old whole-block shift | 1 / 120 | — | 0.350000 | 0.987* |
| Two-update SQP with the old double clipping | 0 / 120 | — | 0.350000 | — |
| Two-update SQP, small disturbance | 120 / 120 | 0.012553 | 0.339608 | 0.899 |

The starred timings cover incomplete runs, including their failed solves, and
are not full-recovery timing comparisons. The renewed full-SLSQP reference
returns no feasible candidate at the sixth interval under the corrected code;
its earlier historical result is not silently reused. Every completed SQP arm
has finite states, bounded commands, supported actual states and robust forecasts,
and terminal attitude and rates within tracking tolerances. The old derivative
rule fails even at the cold start. The old shift fails at the next interval,
where the two-update budget cannot restore a feasible plan.

The reduced budget is promising but does not meet a hard deadline. For the
original disturbance, 16 of 119 subsequent solves exceed the model interval;
the maximum is 1.150 intervals and the cold solve takes 3.182 intervals.
The small case has 15 subsequent misses. These are complete solve durations,
including seed scoring and output checks, not just the optimizer kernel.
Deadlines are disabled in the experiment, so this does not demonstrate the
same trajectory under enforced deadlines. The profile identifies repeated
linearization as the dominant optimizer cost.

The remaining work is to reduce the worst-case solve cost and integrate an
explicit constraint contract into the production solver, including cold-start
feasibility and deadline behavior. This result keeps recovery in NMPC and gives
that work a measured reference. It does not introduce a secondary controller
or establish a larger recovery region.

The two library corrections change ordinary soft-cost controller outputs too.
Both local benchmark artifacts were regenerated, and all NMPC acceptance gates
still pass. Their current numbers live in
[validation](validation.md#nmpc-acceptance); the short-adaptation recovery remains
a negative result.

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
