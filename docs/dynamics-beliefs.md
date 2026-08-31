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
fleet_prior = glassbox.DynamicsBelief.load("artifacts/fleet-prior.json")
belief = belief.condition_parameter_prior(fleet_prior)
forecast = belief.compile_for_nmpc().rollout(initial_state, commands)
updated_belief, update = belief.update(recent_telemetry)
controller = glassbox.NMPCController(updated_belief)
result = controller.solve(state, reference, previous_command)
```

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

The intended update is an information-form local Gaussian step:

```text
recent measured transitions
        ↓
prediction innovations and parameter Jacobians
        ↓
prior precision + observed information
        ↓
updated effective coefficients and covariance
        ↓
residual evidence marked stale until revalidated
```

Updates are functional: they produce a new immutable belief and an audit report.
The report records whether the update was applied and why, coefficient movement,
information gain, covariance contraction, validity utilization, and evidence
counts. Fast coefficient updates and slower residual-model refits use the same
outer contract but remain distinguishable in provenance.

Ordinary point fits explicitly use a `PointParameterBelief`; they do not invent
covariance. When `glassbox-fit` writes a model, it also differentiates a bounded,
group-balanced sample of the training rollouts and stores
`LocalParameterInformation`. This is a local likelihood geometry around the
fitted structured coefficients, not a posterior. Each complete source group
contributes one unit of information, horizons are averaged within a group, and
the tangent residual covariance is inverted only on the subspace supported by
held-out errors. The artifact records the numerical rank, information spectrum,
coordinates excluded by the fitter, unresolved directions, and one local score
vector per independent group. Those group scores preserve the ingredients for
cluster-robust sandwich or influence diagnostics without rerunning the fitter.

The distinction matters: inverting a rank-deficient Hessian would assign zero
variance to directions the flight never excited. Glassbox leaves the ordinary
fit as a point belief plus partial information instead. A complete full-rank
fleet or configuration prior can be combined with that information using
`fit_belief.condition_parameter_prior(fleet_prior)`. Supported directions move
toward the vehicle fit and contract; unsupported directions retain the prior
mean and covariance. A rank-deficient empirical member cloud is rejected as an
incomplete prior rather than silently regularized into certainty.

`LocalGaussianParameterBelief` still covers only the compact structured
coefficient block. A residual network remains fixed during fast conditioning
and online updates. `with_parameter_members()` is useful for recording empirical
fleet spread, but its covariance is not automatically a complete prior when the
members do not span every structured coordinate.

`belief.update(recent_telemetry)` is now implemented as a local Gaussian update.
It selects nonoverlapping windows at the shortest held-out error horizon, uses
the matched empirical tangent covariance for the innovations, differentiates
the complete latent-actuator rollout with respect to structured coefficients,
and returns a new immutable belief plus `BeliefUpdateReport`. The report exposes
innovation before and after the update, information gain, coefficient movement,
covariance contraction, validity utilization, and the evidence counts used.

The held-out residual model remains attached after a parameter update but is
marked not current. This is deliberately different from deleting it or treating
it as newly validated: another update and runtime forecast can use the inherited
noise scale, while NMPC and plan-assessment diagnostics continue to expose that
the residual evidence predates the current coefficients.

## NMPC compilation

Offline evidence can be rich, but the control loop consumes a compact runtime
belief:

- one nominal differentiable model;
- a small structured-parameter covariance or deterministic sigma points;
- a differentiable predictive-error model;
- typed validity support; and
- no training data or optimizer state.

NMPC optimizes the predictive mean and reports total local model uncertainty
relative to the same physical tracking tolerances used by its objective. Its
diagnostics separately state whether empirical predictive error and parameter
uncertainty are available, whether residual evidence is current, and whether
the requested error horizon is supported. State/control-dependent error and
parameter scenarios can later affect constraint tightening, CVaR, or
worst-scenario objectives through this runtime boundary. A large offline
bootstrap ensemble is never required in the real-time loop.

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
structured coefficients and computes the Gaussian information gain implied by
the current parameter covariance and empirical residual covariance. Constraint
risk remains a controller or exploration-policy concern because it depends on a
mission safety envelope, not only the system model.

This supports the conceptual progression:

1. **Arrest and stabilize.** Use a family prior, broad uncertainty, known command
   bounds, and a fallback controller. An entirely unknown thrown vehicle cannot
   be guaranteed recoverable before it produces informative motion.
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
- the matched held-out residual model becomes the honest initial implementation;
  and
- future error or parameter-belief candidates can be compared without another
  system-wide artifact migration.
