# Dynamics beliefs

Glassbox's primary product object is a predictive dynamics belief, not a point
parameter file and not a bootstrap ensemble. The belief preserves a compact,
differentiable nominal model while making prediction error, parameter evidence,
operating support, and update history explicit.

The motivating runtime is broader than ordinary batch identification. A vehicle
may enter with only an airframe-family prior, stabilize using conservative
authority estimates, learn from the resulting motion, and then choose
increasingly informative maneuvers while respecting the support accumulated so
far. The architecture must permit that lifecycle even though no autonomous
flight-envelope exploration demo is part of the current roadmap.

## Product contract

The opinionated public lifecycle is:

```python
belief = glassbox.DynamicsBelief.load("artifacts/vehicle-belief.json")
fleet_prior = glassbox.StructuredParameterPrior.load("artifacts/fleet-prior.json")

# Conditioning moves the parameters, so the attached error evidence becomes
# stale. Measuring it again around the new parameters is what makes the belief
# usable for plan assessment, horizon capping, and further updates.
belief = belief.condition_parameter_prior(fleet_prior)
belief = belief.recalibrate_predictive_error(calibration_telemetry)

forecast = belief.compile_for_nmpc().rollout(initial_state, commands)
updated_belief, update = belief.update(recent_telemetry)
controller = glassbox.NMPCController(updated_belief)
result = controller.solve(state, reference, previous_command)
```

`recent_telemetry` must be disjoint from `calibration_telemetry`; the
recalibration provenance records a content hash of the telemetry it consumed so
that can be shown after the fact. A belief that was fitted offline with fresh
held-out flights already carries current evidence and does not need the
recalibration step until its parameters move.

The serialized object owns:

```text
DynamicsBelief
├── nominal differentiable dynamics and latent actuator state
├── typed state, control, exogenous, timing, and validity contract
├── parameter belief and update history
├── rank-aware local parameter information from grouped training evidence
├── predictive error model in 12 rigid-body tangent coordinates
└── evidence and provenance
```

The 12 local coordinates are position, velocity, shortest-path attitude
rotation vector, and body angular velocity. Quaternion components are never
assigned Euclidean covariance or independent error bars.

Every user-facing fitted artifact is a belief. When parameter or error evidence is absent,
the corresponding component says `available: false`; absence is never encoded
as zero uncertainty. Deterministic parameters remain available as the nominal
member so existing differentiable dynamics do not become conditional on a
probabilistic framework.

## Prediction contract

A rollout returns:

- the nominal state trajectory;
- the evidence-corrected predictive mean;
- tangent-space bias and covariance at every horizon;
- empirical group-radius quantiles when available;
- latent applied-control state;
- validity-envelope utilization; and
- whether the requested horizon is supported by the error evidence.

The error-model interface accepts horizon, nominal state, command, and
exogenous context. The first implementation is deliberately horizon-only, but
the signature permits heteroscedastic state/control-conditioned errors without
changing fitting, serialization, or NMPC.

The initial `EmpiricalHorizonPredictiveError` is fitted only from held-out
rollout endpoints. It gives every independent source group equal mass, then
every trajectory within a group equal mass, then every endpoint within a
trajectory equal mass. It records bias, full tangent covariance, empirical
state-group radii, raw sample count, effective sample count, and independent
group count. Covariance and radii are centered on the reported predictive bias.
These are forecast-error statistics, not a posterior or calibrated probability
distribution.

Every empirical covariance carries one of two scopes:

- `total_forecast_error` is the complete held-out error around the nominal
  forecast. It may already contain parameter variation, process variability,
  observation error, and model-form error. Runtime returns it directly as total
  covariance; adding propagated parameter covariance could count the same
  variation twice.
- `conditional_innovation_error` is separately justified measurement/process
  error conditional on the parameter state. Only this scope may be added to
  propagated parameter covariance or inverted to claim information gain and
  covariance contraction.

Held-out rollout errors default to `total_forecast_error`. Zero empirical
eigenvalues are absent evidence, not noiseless measurements: whitening and
information calculations use only the numerically supported subspace.

Group-bootstrap disagreement is one possible future input to the belief. The
IDF-DS evidence shows that it should not define the abstraction or be the
default error model: independently calibrated residual error was more useful
than adaptive bootstrap spread on that corpus.

## Parameter belief and live updates

Parameter evidence is distinct from predictive residual error. A useful online
implementation needs a local belief over the small structured block of
effective coefficients—control authority, damping, trim or command offset, and
actuator response—without requiring the residual network to move on every
control cycle.

The maintained update is a bounded proposal followed by disjoint validation:

```text
early measured transitions
        ↓
prediction innovations and parameter Jacobians
        ↓
prior-scaled proposal and one-standard-deviation trust bound
        ↓
later, nonoverlapping validation transitions
        ↓
uncorrected candidate scored against the bias-corrected incumbent
        ↓
commit only past a noise-scaled margin, else return the original belief
        ↓
predictive-error evidence marked stale after commit
```

Updates are functional: they produce a new immutable belief and an audit report.
Proposal arrays are owned, read-only copies and each proposal fingerprints the
complete belief revision and target trajectory specification. Transition replay
is detected from physical transition content rather than absolute timestamps.
The report records whether a proposal existed, whether validation ran, why a
commit was accepted or rejected, coefficient movement, prior-standardized step,
validity utilization, and evidence counts. Unsupported horizons,
out-of-envelope telemetry or candidate paths, stale or changed belief evidence,
reused proposal transitions, target-configuration changes, non-finite rollouts,
and validation improvements inside the noise margin all fail closed.
Conditional covariance contraction
is recomputed from the disjoint validation evidence; proposal-carried geometry
cannot make the committed belief overconfident. One contiguous telemetry block
is one evidence unit regardless of its window count.

Actuator commands immediately preceding the validation boundary are carried as
separately fingerprinted initialization context. They are used to recompute the
candidate-dependent latent actuator state, but they are not validation samples
and are excluded from transition-overlap checks. The low-level two-stage commit
API requires callers to supply this context explicitly; missing or malformed
history returns the original belief unchanged.

Stale predictive-error artifacts remain attached for provenance and
recalibration, but runtime forecasts and NMPC no longer apply their bias,
covariance, or quantiles. Independently maintained parameter uncertainty remains
active around the updated nominal model. Because a commit stales the bias, a
candidate is scored without it; see the acceptance criterion below.

Ordinary point fits explicitly use a `PointParameterBelief`; they do not invent
covariance. When `glassbox-fit` writes a model, it also differentiates a bounded,
group-balanced sample of the training rollouts and stores
`LocalParameterInformation`. This is local loss geometry around the
fitted structured coefficients, not a posterior. Each complete source group
contributes one unit of information, horizons are averaged within a group, and
the tangent predictive-error covariance is inverted only on the subspace
supported by held-out errors. The artifact records the numerical rank,
information spectrum, coordinates excluded by the fitter, unresolved
directions, and one local score vector per independent group. Those group scores
preserve the ingredients for cluster-robust sandwich or influence diagnostics
without rerunning the fitter.

The distinction matters: inverting a rank-deficient Hessian would assign zero
variance to directions the flight never excited. Glassbox leaves the ordinary
fit as a point belief plus partial information instead. A complete full-rank
fleet or configuration prior can be combined with that geometry using
`fit_belief.condition_parameter_prior(fleet_prior)`. Supported directions move
toward the vehicle fit. Covariance contracts only when the geometry used
conditional innovation noise; total-forecast-scaled geometry performs a
regularized mean update and preserves prior covariance. Unsupported directions
retain the prior mean and covariance. Conditioning is valid only around the
center the geometry was linearized at, so it refuses to run once the parameters
have moved away from `parameter_evidence.center`, and the belief it returns
carries stale error evidence until that evidence is recalibrated.

`StructuredParameterPrior` makes the unavoidable completion of a small fleet
explicit. In natural structured-parameter coordinates it stores three separate
terms:

```text
between-vehicle covariance      empirical
+ mean within-vehicle covariance empirical, when every member supplies it
+ unit covariance only on the unresolved numerical nullspace
                                 structural prior assumption
```

The completion does not perturb directions spanned by fleet evidence and is
never relabeled as observed variance, a posterior, or a calibrated
distribution. Mixed fleets in which only some members supply covariance are
rejected because their within-vehicle uncertainty has no coherent weighting.
State schema, vehicle family, shared control semantics, and target control-role
coverage are checked at the artifact boundary. Shared and optional parameter
blocks are configuration-aware. Shared aerodynamics use every compatible
member; yaw-surface coordinates use only yaw-equipped members; flap coordinates
use only flap-equipped members. Cross-block covariance is deliberately zero
rather than inferred from unmatched samples. A prior built only from flapless
aircraft cannot initialize a flap-equipped target, while flapless members still
contribute valid evidence to its shared aerodynamic block.

The two supported entry paths are intentionally distinct:

```python
# Existing vehicle telemetry supplies local loss geometry. The conditioned
# parameters need error evidence measured around them before they can be used
# for plan assessment, horizon capping, or another update.
belief = vehicle_fit.condition_parameter_prior(fleet_prior)
belief = belief.recalibrate_predictive_error(calibration_telemetry)

# No vehicle-local fit yet; a typed shell supplies controls/runtime/error model.
# Moving to the prior mean stales the shell's error evidence too, unless the
# caller asserts it is family-level evidence that already holds at the mean.
belief = fleet_prior.initialize_belief(vehicle_shell)
belief = belief.recalibrate_predictive_error(calibration_telemetry)
```

`glassbox-prior` builds the artifact from fitted vehicle beliefs without tuning
flags. `glassbox-adaptation-benchmark` then exercises prior initialization,
family-level predictive error, an immutable live update, and disjoint held-out
prediction for both maintained vehicle families. It is deliberately a
directional synthetic diagnostic rather than a fractional-performance gate.

`LocalGaussianParameterBelief` still covers only the compact structured
coefficient block. A residual network remains fixed during fast conditioning
and online updates. `with_parameter_members()` is useful for recording empirical
fleet spread, but its covariance is not automatically a complete prior when the
members do not span every structured coordinate.

`belief.update(recent_telemetry)` is the opinionated one-call transaction. It
splits complete horizon-aligned windows into early proposal and later validation
partitions. Streaming callers may instead use
`belief.propose_update(telemetry)` and
`belief.commit_update(proposal, later_telemetry)`. The proposal is a
prior-scaled batch Gauss--Newton move capped at one RMS prior standard deviation
and line-searched for improvement. Commit line-searches again on disjoint
telemetry. Total forecast error supplies generalized loss coordinates but leaves
parameter covariance unchanged; conditional innovation error additionally
supports rank-aware contraction and information-gain reporting.

### The acceptance criterion

A commit stales the held-out bias, so the runtime stops applying it. The
acceptance test scores each side the way the vehicle would actually fly it:

- the **incumbent** is the current parameters *with* the bias correction the
  runtime applies today;
- a **candidate** is the moved parameters *without* any bias correction.

Both are measured on the same disjoint validation windows as whitened endpoint
error in the 12 rigid-body tangent coordinates, and the Gauss--Newton proposal
is linearized on the same uncorrected objective it will be judged by. A commit
therefore cannot trade a good corrected forecast for a worse uncorrected one.
The report records this convention as
`validation_scoring: candidate_uncorrected_vs_nominal_bias_corrected`, keeps
`normalized_validation_rms_before` for the bias-corrected incumbent, and keeps
`normalized_validation_rms_after` for the accepted uncorrected candidate.

Improvement is an effect size, not a sign test. Windows are the independent
evidence units, so for each validation window the paired reduction in whitened
squared error, incumbent minus candidate, is summed and compared against a
one-sided margin of `IMPROVEMENT_MARGIN_STANDARD_ERRORS = 2.0` standard errors
of that total. The per-window variance behind that standard error is the sample
variance across validation windows, floored by the chi-square scale of the
incumbent's own error: `2 s^2 / k` for `k` supported error dimensions per window
and mean incumbent whitened squared error `s`. A short evidence block whose
window-to-window spread happens to be small therefore cannot manufacture
significance, and a single window still carries a usable scale. The floor is
anchored to the error level actually observed rather than assuming `s = k`,
because held-out error covariance is empirical and not calibrated; the two
expressions agree exactly when the whitening is calibrated.

Two standard errors is a one-sided level of roughly two percent under a normal
approximation, and the same hurdle is cleared twice on disjoint telemetry, once
to propose and once to commit, so the transaction as a whole is much more
conservative than its per-stage level. Under the null, a belief already at the
true parameters observed through i.i.d. state noise, the pre-margin rule
committed on 30 to 54 percent of seeds while this rule committed on none of 64.
The achieved statistic and the margin it had to clear are both recorded in the
report, at the proposal stage and at the validation stage.

The constant is a documented library invariant, not a tuning knob: no
configuration surface exposes it.

### Stale error evidence and recalibration

Both a commit and `condition_parameter_prior` move the parameters, and both
therefore mark the held-out predictive-error model not current. This is
deliberately different from deleting it, carrying it forward as if it still
applied, or treating it as newly validated. The artifact stays attached for
provenance, runtime forecasts still expose it together with the stale flag, but
its bias, covariance, and quantiles stop being applied, the NMPC horizon cap
disappears, and further updates and plan-information claims are rejected until
the evidence is refreshed.

`belief.recalibrate_predictive_error(telemetry)` is the way back. It rolls out
nonoverlapping windows at the maintained horizons around the belief's
**current** parameters, refits the empirical tangent moments from those
endpoints, and returns a belief whose error evidence is current again. Its
provenance records `source: recalibrated_from_telemetry`, the horizons, window
counts, and a content hash of the telemetry, so a caller can later show that
recalibration evidence was not the block that validated an update. The same
window-and-endpoint routine, `endpoint_error_evidence_by_horizon`, backs both
this path and the held-out evaluation that `glassbox-fit` reports, so online and
offline error evidence are fitted identically.

`condition_parameter_prior` refuses to run when the belief's parameters no
longer equal `parameter_evidence.center`. The local information is a
linearization about that center, so conditioning after an online update would
silently discard the update while still incrementing the evidence counters.
Refit the local geometry around the current parameters instead.

## NMPC compilation

Offline evidence can be rich, but the control loop consumes a compact runtime
belief:

- one nominal differentiable model;
- a small structured-parameter covariance or deterministic sigma points;
- a differentiable predictive-error model;
- typed validity support; and
- no training data or optimizer state.

NMPC first optimizes the predictive mean, then bounds how far the command plan
may move from the previous command when forecast spread exceeds one of the same
physical tracking tolerances used by its objective. The normal horizon is
capped at the maintained predictive-error evidence. Its
diagnostics separately state whether empirical predictive error and parameter
uncertainty are available, whether error evidence is current, whether the
requested error horizon is supported, and the selected command-authority
fraction.

The returned next command then passes through a belief-aware support projection.
It tests the optimized sequence over a bounded actuator-reaction horizon and,
when needed, searches maintained blends between that NMPC command and the
previous bounded command. The mechanism is vehicle-agnostic and cannot generate
a command from a separate attitude, rate, mixer, or airframe-specific control
law. Optimizer failure stays explicit and returns only a bounded hold with
`command_usable=False`; it does not silently transfer authority to another
controller.

The complete NMPC horizon and mission state limits are still soft; CVaR,
worst-scenario objectives, or invariant-set methods can evolve through the same
runtime boundary. A large offline bootstrap ensemble is never required in the
real-time loop.

The model-validity envelope and predictive uncertainty have different meanings.
The former asks whether a query resembles observed operating conditions; the
latter asks how wrong predictions were within the evidence. Both remain visible
to the controller and neither substitutes for the other.

## Active exploration

Safe exploration needs expected information, not merely large uncertainty. For
a candidate control sequence, the runtime belief must be able to expose:

```python
runtime = belief.compile_for_nmpc()
assessment = runtime.assess_plan(state, candidate_commands)

assessment.prediction
assessment.maximum_validity_utilization
assessment.expected_parameter_information_gain_nats
assessment.expected_parameter_covariance
```

`assess_plan()` differentiates the candidate endpoint with respect to the local
structured coefficients. It reports Gaussian information gain only when the
predictive-error evidence is current and explicitly conditional. Rank-zero
directions report information unavailable rather than enormous precision.
Constraint risk remains a controller or exploration-policy concern because it
depends on a mission safety envelope, not only the system model.

This supports the conceptual progression:

1. **Arrest and stabilize.** Use a family prior, broad uncertainty, known command
   bounds, and belief-aware NMPC. An entirely unknown thrown vehicle cannot be
   guaranteed recoverable before it produces informative motion.
2. **Exploit passive excitation.** The throw and recovery provide transitions
   that update control authority, damping, and actuator response.
3. **Probe safely.** Choose bounded maneuvers that add information while their
   predicted trajectories remain acceptable under current uncertainty.
4. **Expand support.** Add operating regions only after observed transitions
   support them; novelty alone never expands the validity envelope.
5. **Increase maneuver complexity.** Trade tracking performance and information
   gain explicitly rather than hiding excitation in controller noise.

Exploration policy belongs above the model and NMPC layers. Glassbox supplies
the differentiable forecasts, information geometry, and evidence updates; it
does not hard-code a flight-test script into the dynamics artifact.

## Evidence versus architecture

Benchmark thresholds govern claims and maintained defaults. They do not decide
whether error modeling is first-class. A candidate can be serialized and
evaluated through the belief interface while remaining labeled uncalibrated,
unsupported, or worse than a baseline.

This keeps negative results useful:

- the IDF bootstrap result rejects bootstrap disagreement as the current
  fixed-wing runtime signal;
- it does not reject predictive-error modeling;
- the matched held-out total-forecast model becomes the honest initial
  implementation;
  and
- future error or parameter-belief candidates can be compared without another
  system-wide artifact migration.
