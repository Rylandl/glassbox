# Dynamics beliefs

Glassbox's primary product object is a predictive dynamics belief, not a point
parameter file and not a bootstrap ensemble. The belief preserves a compact,
differentiable nominal model while making prediction error, parameter evidence,
operating support, and update history explicit.

The nominal model has one type, and the belief owns one of them.
`ExecutableModel` binds fitted parameters to the input spec, the runtime spec,
and an actuation map, and it is what runs: one transition, one rollout,
validity utilization, and the hard command bounds. The belief holds that model
as its `model` field and answers `params`, `input_spec` and `runtime_spec` from
it, so there is exactly one place where the mean lives and exactly one place
where prediction error and parameter uncertainty are answered.

Executable is not the same as actionable. The actuation map is optional: it is
the identity on the declared control channels when those channels are commands
with finite bounds, and it is absent when they are observations of actuation,
such as measured rotor speeds. A model without one still integrates, still
reports validity, and still serializes; every method that needs a command space
raises `NonActionableModelError` and says so. The belief is also the one
artifact the library writes; a bare model payload is still read, as a belief
with no evidence attached.

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

forecast = belief.rollout(initial_state, commands)
updated_belief, update = belief.update(recent_telemetry)

# The commit moved the parameters, so the attached error evidence is stale.
# Measuring it again around the new parameters is what makes the belief usable
# for horizon capping and further updates.
updated_belief = updated_belief.recalibrate_predictive_error(calibration_telemetry)

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
- latent applied-control state;
- validity-envelope utilization; and
- whether the requested horizon is supported by the error evidence.

Declared command bounds and the validity envelope are enforced differently, on
purpose. Bounds are a hard execution contract: a rollout or runtime transition
given a concrete command outside its channel bounds raises and names the
channel, and only a violation within `1e-6` of the channel span, which is
normalization or serialization rounding, is clipped onto the bound. The
validity envelope is advisory: utilization is reported at every step and the
caller decides, because an out-of-envelope forecast is unsupported rather than
impossible. Bound checks need concrete values, so a JAX-traced command is left
to its caller; NMPC constructs every command inside the bounds.

The error-model interface accepts horizon, nominal state, command, and
exogenous context. The first implementation is deliberately horizon-only, but
the signature permits heteroscedastic state/control-conditioned errors without
changing fitting, serialization, or NMPC.

The initial `EmpiricalHorizonPredictiveError` is fitted only from held-out
rollout endpoints. It gives every independent source group equal mass, then
every trajectory within a group equal mass, then every endpoint within a
trajectory equal mass. It records bias, full tangent covariance, raw sample
count, effective sample count, and independent group count. The covariance is
centered on the reported predictive bias. These are forecast-error statistics,
not a posterior or calibrated probability distribution.

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

Every artifact the shipped CLIs write carries `total_forecast_error`:
`glassbox fit` fits held-out rollout error and records that scope, and the
local parameter information it stores inherits the scope of the predictive
error it was whitened with. No flag changes this. Everything in this document
that depends on `conditional_innovation_error`, commit-time covariance
contraction above all, is therefore reachable only by a caller who builds and
attaches conditional innovation evidence programmatically, having separately
justified that the covariance is measurement and process noise conditional on
the parameters. With a shipped artifact those paths report unavailable and the
parameter covariance is preserved rather than contracted.

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
prior-scaled proposal, no coordinate past one prior standard deviation
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
commit was accepted or rejected, coefficient movement, the bounded and
root-mean-square prior-standardized step, validity utilization, and evidence
counts. Unsupported horizons,
out-of-envelope telemetry or candidate paths, stale or changed belief evidence,
reused proposal transitions, target-configuration changes, non-finite rollouts,
and validation improvements inside the noise margin all fail closed.
Covariance contraction happens only for a belief whose error evidence is
conditional innovation, which no shipped artifact is; when it does, it is
recomputed from the disjoint validation evidence, so proposal-carried geometry
cannot make the committed belief overconfident. A total-forecast belief commits
the mean move and leaves its parameter covariance alone. One contiguous
telemetry block is one evidence unit regardless of its window count.

Actuator commands immediately preceding the validation boundary are carried as
separately fingerprinted initialization context. They are used to recompute the
candidate-dependent latent actuator state, but they are not validation samples
and are excluded from transition-overlap checks. The low-level two-stage commit
API requires callers to supply this context explicitly; missing or malformed
history returns the original belief unchanged.

Stale predictive-error artifacts remain attached for provenance and
recalibration, but runtime forecasts and NMPC no longer apply their bias or
covariance. Independently maintained parameter uncertainty remains
active around the updated nominal model. Because a commit stales the bias, a
candidate is scored without it; see the acceptance criterion below.

Ordinary point fits explicitly use a `PointParameterBelief`; they do not invent
covariance. When `glassbox fit` writes a model, it also differentiates a bounded,
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
fit as a point belief plus partial information instead, and the online update
takes no step along a direction that information does not resolve.

Where a covariance over the structured block is genuinely available, for
instance from several vehicles of one family or from several configurations of
one vehicle, `LocalGaussianParameterBelief.from_members` summarizes those
members around the nominal model. That covariance is exactly the spread the
members show: directions no member moved carry no variance, and nothing
completes them with an assumption. It covers only the compact structured
coefficient block, and a residual network stays fixed during an online update.

`belief.update(recent_telemetry)` is the opinionated one-call transaction. It
splits complete horizon-aligned windows into early proposal and later validation
partitions. Streaming callers may instead use
`belief.propose_update(telemetry)` and
`belief.commit_update(proposal, later_telemetry)`. The proposal is a
prior-scaled batch Gauss--Newton move, bounded and line-searched for
improvement. The bound is a maximum, not an average: the step is scaled down
uniformly, keeping its direction, until no whitened supported prior coordinate
moves by more than one prior standard deviation
(`MAXIMUM_LOCAL_PARAMETER_STEP_SIGMA = 1.0`). A root-mean-square bound would be
a weaker claim, since a step concentrated in one direction of the rank-22
structured block could move about 4.7 standard deviations along it. The report
records both the bounded maximum, `prior_standardized_step_max`, and the
root-mean-square spread of the same step. Commit line-searches again on disjoint
telemetry. Total forecast error supplies generalized loss coordinates but leaves
parameter covariance unchanged; conditional innovation error additionally
supports rank-aware contraction and information-gain reporting, and only a
caller who attaches that evidence programmatically has it.

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

A commit moves the parameters and therefore marks the held-out predictive-error
model not current. This is deliberately different from deleting it, carrying it
forward as if it still applied, or treating it as newly validated. The artifact
stays attached for
provenance, runtime forecasts still expose it together with the stale flag, but
its bias and covariance stop being applied, the NMPC horizon cap disappears, and
further updates are rejected until the evidence is refreshed.

`belief.recalibrate_predictive_error(telemetry)` is the way back. It rolls out
nonoverlapping windows at the maintained horizons around the belief's
**current** parameters, refits the empirical tangent moments from those
endpoints, and returns a belief whose error evidence is current again. Its
provenance records `source: recalibrated_from_telemetry`, the horizons, window
counts, and a content hash of the telemetry, so a caller can later show that
recalibration evidence was not the block that validated an update. The same
window-and-endpoint routine, `endpoint_error_evidence_by_horizon`, backs both
this path and the held-out evaluation that `glassbox fit` reports, so online and
offline error evidence are fitted identically.

`parameter_evidence` carries the same caveat in the other coordinate. It is a
linearization about `parameter_evidence.center`, so once an online update has
moved the parameters away from that center the stored geometry describes the
belief the fit produced, not the belief in hand. Refit the local geometry around
the current parameters before reading it as current curvature.

## NMPC compilation

Offline evidence can be rich, but the control loop consumes a compact runtime
belief:

- one nominal differentiable model;
- a small structured-parameter covariance or deterministic sigma points;
- a differentiable predictive-error model;
- typed validity support; and
- no training data or optimizer state.

NMPC optimizes the expected cost of its own forecast rather than the cost of
the predictive mean alone. Predicted tangent spread is charged in the tracking
cost against the same physical tracking tolerances the objective already uses,
and the model-validity term is widened by the marginal standard deviation of
the six envelope features, so a belief that knows less plans nearer to ground
it has evidence for. The normal horizon is capped at the maintained
predictive-error evidence, and the diagnostics record the largest normalized
spread the returned plan carries.

Nothing edits the command after optimization. The mechanism is vehicle-agnostic
and cannot generate a command from a separate attitude, rate, mixer, or
airframe-specific control law. Optimizer failure stays explicit and returns only
a bounded hold with `command_usable=False`; it does not silently transfer
authority to another controller.

The complete NMPC horizon and mission state limits are still soft; CVaR,
worst-scenario objectives, or invariant-set methods can evolve through the same
runtime boundary. A large offline bootstrap ensemble is never required in the
real-time loop.

The model-validity envelope and predictive uncertainty have different meanings.
The former asks whether a query resembles observed operating conditions; the
latter asks how wrong predictions were within the evidence. Both remain visible
to the controller and neither substitutes for the other.

## Active exploration

Safe exploration needs expected information, not merely large uncertainty. The
belief exposes the pieces that calculation is built from rather than a scoring
entry point of its own. `belief.rollout(...)` returns the parameter tangent
Jacobian, the propagated parameter covariance at every horizon, and validity utilization along the candidate path, and
`parameter_evidence` carries the local information matrix with its numerical
rank. An exploration policy forms expected information gain from those, on the
coordinates and horizon it cares about, and must decide for itself that a
rank-zero direction carries no information rather than enormous precision.
Whether that gain is meaningful still depends on the error scope: only
conditional innovation covariance can be inverted to claim contraction, and no
shipped artifact carries it.
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
