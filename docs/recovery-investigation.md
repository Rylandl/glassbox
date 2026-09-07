# Recovery after identification: investigation

The uncertainty, supervision, SLSQP, first SQP, seed-runtime, deadline-budget
and horizon-shift reports below are historical snapshots from commits
`0c4ec6a`, `777652f`, `0926d49`, `8a79306`,
`5607f3f`, `98d2fe5` and `4f5cddb`, respectively. Their source
fingerprints identify the code that produced the numbers. The first three precede the
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
  --output /tmp/glassbox-sqp-recovery.json
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

## Constraint interface, seed reuse, and deadlines

The [runtime follow-up](investigations/sqp-runtime.json) moves the formulation
into the model contract and measures complete solves with and without enforced
deadlines:

```bash
uv run python scripts/investigate_sqp_runtime.py \
  --output /tmp/glassbox-sqp-runtime.json
```

`ConstrainedLeastSquaresPlanModel.optimization_terms` returns `PlanTerms` with
objective residuals and signed inequality margins. The scalar cost and residual
form share physical components; tests compare values and derivatives for point
models in both vehicle families and for command-dependent uncertainty with
active penalties. The SQP implementation reads these arrays without assuming
six features or a particular vehicle family. This is an optional extension to
`PlanModel`; the default bounded solver's objective and policy are unchanged.

For a belief, the margins cover the supplied initial state's mean support,
then the existing covariance-expanded support at each future stage. An initial
state outside the declared envelope cannot become feasible by changing future
commands. Constant constraints at the boundary are not artificially tightened
by the quadratic subproblem's interior target. Optional safety limits retain
their existing soft-cost meaning, and unresolved uncertainty remains unknown.

The former seed path evaluated cold and warm objectives with full gradients,
then differentiated the selected seed again for SQP. The new path evaluates
residuals and margins first, prefers feasibility before cost, and computes one
selected-seed linearization that the first update reuses. When neither seed is
feasible, it prefers the smaller summed violation before cost. All derivatives
are checked for finiteness, and the returned plan still receives its own final
objective and gradient. The cached linearization is cleared on every new
request, including after a deadline rejection.

Iteration budgets now follow the input warm start, rather than whether the
optimizer has past report entries. A request without a correctly shaped warm
plan receives the eight-update cold budget even after earlier calls; a warm
request receives two updates. Reports cannot change controller behavior.
The production solver also checks the deadline after assembling the result,
including warm-start validation, and records that elapsed duration. Tests
exercise just-before, exactly-at, and after-deadline completion.

The seven fresh scenarios retain the same additional-evidence belief, support,
objective, uncertainty and plant as the earlier experiment. Compilation is
prewarmed without advancing the plant. Times below include seed work, the
optimizer, output checks and result assembly; they are host measurements.

| Case | Executed intervals | Cold solve (ms) | Subsequent median / maximum (ms) | Subsequent solves over 20 ms |
| --- | ---: | ---: | ---: | ---: |
| Former seed path, deadlines disabled | 120 / 120 | 65.390 | 18.266 / 23.635 | 19 / 119 |
| Reused seed, original, deadlines disabled | 120 / 120 | 62.285 | 14.996 / 20.252 | 1 / 119 |
| Reused seed, small, deadlines disabled | 120 / 120 | 28.190 | 15.092 / 19.611 | 0 / 119 |
| Original, 20 ms from the cold solve | 0 / 120 | 62.575 | — | — |
| Original, 100 ms startup then 20 ms | 120 / 120 | 62.681 | 15.001 / 19.944 | 0 / 119 |
| Small, 100 ms startup then 20 ms | 23 / 120 | 28.221 | 15.101 / 22.287* | 1 / 23* |
| Initial support utilization 1.1, deadlines disabled | 0 / 120 | 28.674 | — | — |

The starred row includes its failed 24th solve. That deadline failure ends the
scenario before any hold or other command is applied; it has no recovery-tail
score. The strict cold-start case is likewise rejected, and the outside-support
case returns no feasible plan. These failures are retained in the report.

The complete original run with seed reuse has tail normalized tracking RMS
0.295058 and peak actual support utilization 0.999860, compared with 0.295111
and 0.999860 for the former seed path. Its maximum robust forecast utilization
is 0.999999344. The small deadline-free run retains tail RMS 0.012553 and peak
actual utilization 0.339608. Every completed case remains finite and bounded,
inside actual and forecast support, and within terminal attitude/rate
tolerances. None is labeled first-order converged.

Seed reuse reduces median solve time by about 18%, but the occasional warm
overrun and the rejected small-disturbance deadline case prevent a timing
qualification. The 100 ms startup case is a separately budgeted initialization
experiment; it does not validate a delayed first command during flight.
The next backend work must address cold-start feasibility, interruptible
iteration budgets, and retaining a checked feasible plan before the output
budget expires. The constrained algorithm remains an offline reference while
those runtime behaviors are unresolved.

## Cooperative deadline budgets and fused output

The [deadline-budget report](investigations/sqp-budget.json) measures cooperative
stopping inside SQP, with the same model, objective, support and uncertainty:

```bash
uv run python scripts/investigate_sqp_runtime.py --experiment budget \
  --output /tmp/glassbox-sqp-budget.json
```

The solve now passes an immutable absolute deadline to its seed and optimizer
phases. SQP checks that the next seed evaluation, linearization, quadratic
subproblem or nonlinear trial leaves time for output. Checks occur between
operations; an executing JAX or SciPy call cannot be preempted. The request's
final elapsed-time check remains authoritative.

The host estimates are explicit inputs: 5.5 ms for a linearization, 1.5 ms for
a quadratic subproblem, 0.75 ms for a nonlinear evaluation, and 3 ms reserved
for output. The output reserve follows the probe's roughly 2.1–2.4 ms final
kernel and materialization time, with room for the remaining checks and result
assembly. These are admission estimates, not upper bounds on execution time.
The larger-reserve arms change only the output reserve to 6 ms.

The optimizer retains each finite, bounded candidate that passes the original
nonlinear inequalities. A seed is checked against this request's state,
actuator state, forecast and model values. Its feasibility does not carry over
from a previous solve. Running out of work budget returns a checked candidate
if it can be finalized in time, or a previously prepared feasible result.
Otherwise it returns an explicit deadline failure. A quadratic subproblem's linear constraints alone
cannot establish a candidate's nonlinear feasibility.

Final scoring, its exact gradient, prediction, measurements and inequality
margins now share a compiled kernel. The backend can hand that evaluation to
the common solve boundary without repeating the rollout. Final margins are
checked again, and the usual finite-value, command-bound and deadline checks
still apply.

Linearization also provides its prediction, measurements and scalar cost as
auxiliary outputs. For a feasible point, the residual Jacobian supplies the
objective gradient through `2 J.T r`, allowing a complete result to be prepared
without another model evaluation. Tests compare that gradient with the scalar
objective's derivative. The best prepared result is retained within this solve.
It is reused directly when selected, or when a better trial cannot be finalized
in time. This avoids discarding a known feasible plan just because the final
kernel no longer fits. The outer deadline check still rejects a late return.

Controlled-clock tests cover early return, infeasible seeds,
unexpectedly slow evaluations, protected output time, and an unchecked
quadratic step. Tests also compare fused and separate evaluations and inject
invalid prepared output to verify the common checks still reject it.

The eleven scenarios use fresh identification evidence and prewarmed kernels.
Except where stated otherwise, budgeted cases allow 100 ms for initialization
and 20 ms thereafter. Subsequent timings include a failed final solve when a
run ends early; incomplete runs have no tail tracking score.

| Case | Executed intervals | Subsequent median / maximum (ms) | Subsequent overruns | Usable early returns |
| --- | ---: | ---: | ---: | ---: |
| Separate output, original; no deadlines | 120 / 120 | 15.696 / 20.466 | 1 | 0 |
| Budgeted original without prepared results | 4 / 120 | 18.596 / 19.518 | 0 | 1 |
| Fused output, original; no deadlines | 120 / 120 | 15.489 / 20.141 | 2 | 0 |
| Fused output, small; no deadlines | 120 / 120 | 15.286 / 19.003 | 0 | 0 |
| Budgeted original, 3 ms output reserve | 4 / 120 | 17.239 / 17.568 | 0 | 3 |
| Budgeted small, 3 ms output reserve | 120 / 120 | 15.329 / 16.979 | 0 | 2 |
| Original, 20 ms from cold start | 0 / 120 | — | 0 | 0 |
| Original, 6 ms output reserve | 4 / 120 | 9.871 / 11.223 | 0 | 3 |
| Small, 6 ms output reserve | 6 / 120 | 15.399 / 43.012 | 1 | 3 |
| Original, 3 ms total seed budget | 0 / 120 | — | 0 | 0 |
| Initial support utilization 1.1 | 0 / 120 | — | 0 | 0 |

The completed small recovery with a 3 ms output reserve has tail normalized
tracking RMS 0.012553, peak actual support utilization 0.339608 and peak robust
forecast utilization 0.358257. All 119 subsequent solves finish below 17 ms in
this run. It returns early twice; its 22 prepared-result selections also
include cases where the prepared point is already the lowest-cost plan.

The original disturbance returns three prepared feasible plans early, then
stops at the fifth solve before finding another feasible plan in the remaining
budget. No subsequent solve overruns and no fallback is applied. The arm
without prepared results also stops at the fifth solve, after finding no
feasible iterate within its updates. Thus the new output mechanism works but
does not extend the demonstrated recovery region under a 20 ms budget.
The 6 ms reserve makes the original case return sooner, without restoring its
missing fifth plan. A feasible output at one interval does not guarantee that
its shifted, block-averaged warm start is feasible at the next interval.

Cold feasibility is still unresolved within 20 ms: that case now refuses the
solve at 14.678 ms, before applying any command. A 3 ms seed budget is refused
at 0.650 ms, before starting an expensive seed evaluation. An initial state at
support utilization 1.1 remains infeasible.

A larger output reserve is not a timing guarantee. The small 6 ms-reserve case
records a 43.012 ms solve and stops after six executed intervals. A prepared
result existed, but the outer deadline check rejects its late return. The
profile records the delay inside optimizer work outside its timed
linearization, quadratic-subproblem and line-search calls; it does not
establish the cause. This failure remains in the artifact.

The constrained algorithm remains experimental. The next formulation work is
to preserve a usable plan across horizon shifts so a shortened solve can start
from a feasible candidate. Deadline admission also needs to remain paired with
the final rejection check on this host; these measurements establish neither
worst-case execution time nor delayed-actuation performance in flight.

## Horizon shifts and terminal feasibility

The [horizon-shift report](investigations/horizon-shift.json) separates command
projection, the updated plant state, and the appended prediction interval:

```bash
uv run python scripts/investigate_horizon_shift.py \
  --output /tmp/glassbox-horizon-shift.json
```

It rebuilds the additional-evidence belief and retains the original objective,
support, uncertainty, 30-step horizon, and 40 command variables. The deterministic
audit limits warm solves to one SQP update, reproducing loss of feasibility
without making that result depend on a host deadline. It evaluates the exact
shifted commands with the last command repeated, then compares the regular
block projection. It also starts the exact sequence from the previous plan's
predicted next state and actuator state to separate plant mismatch.

At the failed fifth solve, at 0.08 seconds, robust support utilization is:

| Seed construction | Retained 29-step prefix | New final step |
| --- | ---: | ---: |
| Exact shift, previous prediction's next state and actuator state | 0.983069 | 1.278043 |
| Exact shift, actual next state and actuator state | 0.980753 | 1.277239 |
| Regular block averages, actual next state and actuator state | 0.979219 | 1.300649 |

The main violation is at the appended endpoint. Averaging makes that violation
slightly larger, but removing it does not restore feasibility. An 18-start
bounded search varies only the last four motor commands while keeping the
retained prefix fixed. The starts include every command-box vertex, its center,
and the held command. The best result found still has roll-rate support
utilization 1.159888. This is a local-search diagnostic, not a proof that no
terminal command is feasible. It indicates that changing commands earlier in
the horizon deserves attention.

Nor can the retained prefix inherit its old robust margins without evaluation.
For example, the first shifted prefix evaluated from the previous predicted
state reaches 1.000791, even though the previous plan passed. The new forecast
recomputes parameter and forecast-error covariance from its new initial state.
Preserving the mean command sequence alone does not preserve those margins.

The experiment also implements moving block boundaries. For three-step blocks,
the first block cycles through lengths three, two, and one while the last block
absorbs the extension. After the phase wraps, the expired first block disappears
and the repeated terminal block splits. This represents the exact shifted
sequence with the same ten command blocks. The warm start carries its phase;
report history does not choose it. Foreign or edited warm starts are checked
for membership in that layout. Each phase has a distinct compilation signature,
and all phases are prewarmed before timing. Dispatch and result assembly count
against the solve deadline.

The closed-loop results argue against adopting this layout as a recovery fix:

| Case | Executed intervals | Subsequent median / maximum (ms) | Subsequent solves over 20 ms |
| --- | ---: | ---: | ---: |
| Fixed blocks, one update | 4 / 120 | 9.431 / 9.636 | 0 |
| Moving blocks, one update | 5 / 120 | 10.179 / 15.846 | 0 |
| Fixed blocks, two updates | 120 / 120 | 15.476 / 20.457 | 2 |
| Moving blocks, two updates | 9 / 120 | 19.728 / 21.309 | 2 |
| Moving blocks, original, 100 ms startup then 20 ms | 11 / 120 | 16.998 / 19.345 | 0 |
| Moving blocks, small, 100 ms startup then 20 ms | 120 / 120 | 15.575 / 18.550 | 0 |
| Moving blocks, initial support utilization 1.1 | 0 / 120 | — | — |

Deadlines are disabled except in the two explicitly budgeted cases. Timing
includes the failed final solve where present; no failed command is applied.
The budgeted original run stops because SQP finds no feasible plan, despite
staying within its deadline. Its budget-driven early returns alter its command
history, so eleven intervals do not establish an advantage over the nine-step
deadline-free run. Every executed trace starting inside support stays inside
actual support, with finite bounded commands and checked robust forecasts.
The outside-support case rejects the initial request.

Fixed blocks with two updates retain tail normalized tracking RMS 0.295058 and
peak actual support utilization 0.999860. The completed small moving-block case
has tail RMS 0.011607, peak actual utilization 0.339314, and peak robust forecast
utilization 0.357808. Both complete runs satisfy terminal attitude/rate tolerances.
The occasional deadline-free overruns still preclude a timing qualification.

`BeliefPlanModel.rollout_commands` now evaluates an exact physical command
sequence through the same actuator dynamics and uncertainty propagation. The
simulation uses it to check returned forecasts, replacing reconstruction from
fixed block starts, which would check a different waveform under a moving
layout. Tests compare mean and actuator trajectories with direct model steps,
covariance with a full parameter Jacobian, and costs and derivatives in both
vehicle families. The maintained solver's formulation and block layout remain
unchanged; moving blocks stay in this experimental script.

The next formulation target is a terminal condition or optimized terminal
suffix that accounts for the ability to continue inside support. A supported
endpoint alone does not establish that ability, especially with actuator lag.
The current 0.6-second horizon already reaches the belief's maximum
forecast-error evidence: adding a 0.62-second robust guard would require an
explicit treatment of that missing evidence. This work belongs inside NMPC;
the experiment supplies no secondary controller and claims no recursive
feasibility guarantee.

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


## Parallel terminal, uncertainty, and interface follow-up

Three bounded investigations separated formulation, uncertainty assumptions,
and result semantics. They shared one freshly rebuilt belief and known failing
request; each used an isolated checkout. Production control defaults remain
unchanged.

- The [terminal suffix experiment](terminal-suffix-investigation.md) repairs
  the failed forecast by optimizing six commands inside the existing horizon.
  Peak support utilization falls from 1.277239 to 0.980753. A 36-interval
  continuation remains feasible and within actual support, including twelve
  intervals applying commands from the optimized suffix. Each restricted solve
  takes 22–50 SLSQP iterations, so this is a formulation result without a
  deadline qualification or completed-recovery claim.
- The [uncertainty audit](shift-uncertainty-audit.md) explains the first
  shifted-prefix violation through lost cancellation between parameter
  sensitivities. Carrying the old joint sensitivity reproduces the old
  covariance to 8.01e-11; resetting a known current state is a different
  prediction. No covariance shrinking, covariance carry-over, or runtime
  uncertainty change follows from this result.
- The interface slice below exposes checked nonlinear feasibility separately
  from optimizer status and deadline completion. Interval logs preserve these
  fields, including explicit `not_assessed` outcomes.

The [bounded GN-SQP follow-up](fast-suffix-investigation.md) tests suffix freedom
and feasible-candidate retention through the normal solve boundary. Its
[head-and-suffix extension](fast-suffix-investigation.md#immediate-feedback-with-head-and-suffix-freedom)
also permits immediate command changes and demonstrates a local response to
matched state perturbations. Existing work-admission checks allow a bounded
continuation to return earlier feasible checkpoints when needed. The note retains
the failed runtime case and narrow timing headroom. The subsequent
[cold-start/full-recovery study](fast-suffix-investigation.md#cold-startup-and-full-recovery)
completes the original, small and perturbed cases without deadlines, with all
final-state tolerances satisfied. Two timed cold-start recoveries complete;
the scheduled-kick arm stops on a seed-stage deadline overrun before its kick.
An identical request passed in another recorded arm. The subsequent
[seed-timing probe](fast-suffix-investigation.md#seed-timing-and-process-history)
records 256 successful identical-input replays, followed by a separate traced
scenario-order pass that stops all three cases. A new seed overrun is dominated
by JAX-output materialization without overlapping GC; the original spike's cause
remains unidentified. A timely return can also lack a feasible candidate. Runtime
variability and the margin available for repair remain unresolved. The costly
reference solve remains evidence of repairability, not a deployable second controller.
The [single-seed reuse comparison](fast-suffix-investigation.md#reusing-a-single-seeds-linearization-values)
removes one redundant rollout evaluation with bitwise parity on 26 saved requests.
Paired acceptance improves, but all six timed cold-start arms stop. Measured work
costs exceed the configured admission estimate, so the next runtime question is
whether request-local observations can prevent starting work that no longer fits.

The [repeated identification pilot](repeated-uncertainty-calibration.md)
separately tests uncertainty components using independent training and
calibration sources. It exposes substantial estimator bias and a mismatch
between declared parameter covariance and repeated-fit variation. It does
not establish calibrated double counting or justify covariance subtraction.
The note records the small sample count and the test cases outside support.

## Returned-plan feasibility evidence

The normal SQP `SolveResult` now carries `nonlinear_feasibility` for the
exact prepared checkpoint or fused final prediction selected for return.
The assessment reports constraint count, maximum signed-margin violation,
and numerical tolerance, independently of the `stalled` optimizer status
and the host `deadline_met` observation. Optimizer trial histories remain
experiment reports. No new timing run or production-backend change
accompanies this interface.

A rejected late, nonfinite, or unbounded prediction returns an unassessed
hold. The ordinary bounded solver, SLSQP reference, and separate-output SQP
ablation remain `not_assessed` because this slice does not bind their
constraint evaluation to a returned prediction. Checked empty constraints
mean zero declared inequalities; missing checks never mean feasible.
The fitted model's checks describe initial mean support and future marginal
uncertainty-expanded support, not the soft `SafetyEnvelope` preferences,
a constrained KKT convergence assertion, or a hardware deadline guarantee.
Passing these numerical margins does not establish completeness of parameter
uncertainty; that independent diagnostic retains its meaning.
