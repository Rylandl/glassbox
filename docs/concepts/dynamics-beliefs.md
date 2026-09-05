# Dynamics beliefs

Glassbox's product object is a dynamics belief, not a point parameter file and
not an ensemble. A belief is three things and nothing else:

```text
DynamicsBelief
├── model             ExecutableModel: the parameters, the typed prediction
│                     contract, the runtime timing and validity envelope, and
│                     the actuation map when its inputs are commands
├── information       ParameterInformation: which directions of the structured
│                     coefficient block the evidence has resolved, how
│                     precisely, and the one-step innovation noise that weights
│                     every new observation
└── forecast_error    ForecastErrorEnvelope: how wrong forecasts of a given
                      length have been on flights the fit did not see
```

plus `provenance`, which records where those came from and how far the
parameters have moved since. `belief.support` is the model's own validity
envelope, the operating region the evidence covers.

The model is the belief's mean and the one place it lives; `params`,
`input_spec` and `runtime_spec` are answered from it. Everything else says how
wrong that mean has been and which of its coefficients the evidence resolved.

Executable is not the same as actionable. The actuation map is optional: it is
the identity on the declared control channels when those channels are commands
with finite bounds, and it is absent when they are observations of actuation,
such as measured rotor speeds. A model without one still integrates, still
reports validity and still serializes; every method that needs a command space
raises `NonActionableModelError` and says so. The belief is also the one
artifact the library writes; a bare model payload is still read, as a belief
with rank-zero information and no envelope.

One belief type carries either model family. A fit over a corpus produces a
belief whose model is a structured multirotor or fixed-wing model; the
in-flight `RecursiveBootstrapIdentifier` produces a belief whose model is the
bootstrap parameterization, `BootstrapMultirotorParams`, whose free parameters
are the direct command effects it can learn from motor input and output alone.
Nothing downstream branches on which: `plan_model` takes either, and the solver
reads a mean, a tangent covariance and a command box.
[Bootstrap identification](bootstrap-identification.md) covers what the second
family estimates and what it deliberately does not.

## Product contract

```python
from pathlib import Path

import glassbox
from glassbox.core.data import load_trajectory_npz

flight_paths = sorted(Path("flights").glob("*.npz"))
outcome = glassbox.fit(flight_paths, glassbox.FitSpec())
belief = outcome.belief

# or, from an artifact a previous fit wrote:
belief = glassbox.DynamicsBelief.load("artifacts/belief.json")

telemetry = load_trajectory_npz(flight_paths[-1])
forecast = belief.rollout(telemetry.states[0], telemetry.controls[:30])
belief, update = belief.absorb(telemetry)

controller = glassbox.NMPCController(belief)
state = telemetry.states[0]
result = controller.solve(
    state, controller.hold_reference(state), telemetry.controls[0]
)
```

`absorb` adds information from usable transitions and safeguards the mean
update against nonlinear divergence. The forecast envelope remains measured
at the earlier parameters; `parameter_distance_since_measurement` records
that drift. The controller refuses unresolved parameter uncertainty by default.
Explicit partial-information planning is described in [NMPC](nmpc.md).

## The twelve local coordinates

Prediction error, parameter sensitivity and innovation noise are all stated in
the same twelve rigid-body local coordinates: position, velocity,
shortest-path attitude rotation vector, and body angular velocity.
`rigid_body_local_error` is the map into them and `state_plus_tangent` is the
retraction back out, so a tangent-space error, a tangent-space perturbation
and a tangent-space covariance describe the same thing. Quaternion components
are never assigned Euclidean covariance or independent error bars.

## What the fit produces

`glassbox.fit(sources, spec)` returns a `FitOutcome`: the belief the fit
supports, the report that records how it was produced, and one belief per
requested ablation. `FitSpec` carries the user's choices, with the loss
geometry in a `LossPolicy` and the training weighting in a `WeightingPolicy`,
and the `Holdout` names which evidence the fit is not allowed to see.

Fitted parameters are effective predictive coefficients, not uniquely
recovered physical constants. Complete-flight holdout results test cross-flight
generalization; the `no-lag` ablation's ratios above one say that modeling
latent applied-control response improved prediction on evidence the fit did not
see. Every training horizon is normalized by its own initial loss, and several
are then combined with equal weight. Each labeled source group contributes
equal total loss weight with uniform weight inside the group; without source
groups each training flight contributes equally, and when every flight declares
a maneuver profile each family contributes equally before its replicates split
it. Large candidate sets are deterministically thinned across every group's
timeline under one window budget, a window ceiling and an unrolled-transition
ceiling per horizon, which is the same budget the optimizer batches under, so
the set a fit is extracted on is the set one gradient step processes. Every
model class uses equal semantic state-group loss after scaling by
training-window motion, linearly emphasizes later rollout steps, and softly
penalizes velocity or rate escape beyond a generous training-derived body-frame
envelope. For a structured residual, frame-invariant feature normalization and
six-axis correction bounds are derived only from the training windows, kept
fixed during fitting, and serialized with the model.

The fit also resolves the model's runtime contract, the sample period and the
training-supported validity envelope, so the belief it returns is executable
without anything being recovered from the report afterwards.

### The noise model

Every fit measures the one-step innovation of the model it just produced on the
held-out flights, per coordinate, as a second moment about zero. That is the
belief's `innovation_noise`: a belief that cannot say how wrong its one-step
predictions are cannot weight the next observation either. A declared per-group
floor travels beside it as `noise_floor` and the measured value never falls
below it, which keeps the whitening finite when a model reproduces held-out
telemetry to integration accuracy.

The held-out **mean** error is recorded in the fit report's validation block,
under `held_out_mean_tangent_error`, and nothing applies it. Correcting a
forecast by a mean measured on other flights moves the model without moving the
account of what is known about it, which is why the envelope is the uncentered
second moment: it carries the whole of how wrong the forecast has been rather
than only its spread about a correction that is never made.

### The information

Every fit also accumulates `sum_w J_w' R^-1 J_w` over the training flights'
one-step transitions, with the fitter's estimable mask, a bounded window
budget spread evenly across the independent source groups, and the balanced
effective count. That is what makes the returned belief say which parameter
directions its evidence resolved, so it is not optional: `FitSpec` carries
`parameter_evidence` for a caller who wants the point estimate alone, and no
command turns it off.

This is deliberately the same estimator `absorb` runs. The noise model is
one-step innovation covariance, so the windows it weights correctly are
one-step windows: whitening a multi-step endpoint by it would overstate the
information by roughly the horizon and would count the same transition once per
training horizon. A fitted belief and a belief that has absorbed telemetry
therefore state their information in one currency and can be added.

## `ParameterInformation`

```text
ParameterInformation(names, precision, scale, estimable, innovation_noise,
                     noise_floor, effective_count, rank_relative_tolerance)
```

`precision` is the accumulated information in parameter coordinates. `scale`
defines the normalized coordinate `u` by `parameters = center + diag(scale) u`,
so the normalized precision `diag(scale) @ precision @ diag(scale)` has
eigenvalues comparable across coordinates of different physical units, and the
rank test is stated on those. The first nonzero precision sets an absolute
`rank_threshold` using `rank_relative_tolerance` times its largest normalized
eigenvalue. `with_precision` preserves that threshold as information grows,
so strengthening one direction cannot erase previously resolved directions.
The threshold is serialized with the information state. `estimable` is
the mask the fitter declares; a coordinate outside it is held fixed by
construction, and its rows and columns of the precision are exactly zero.

- `resolved_rank()` and `resolved_subspace()` report what the evidence has
  resolved.
- `covariance()` is the inverse on the resolved subspace, padded with zeros.
  It is a partial covariance, not a claim of zero variance elsewhere.
  `unresolved_subspace()` carries the missing directions, and `complete`
  reports whether all estimable directions are resolved. Prediction reports
  unbounded future standard deviations when that completeness is missing;
  the controller requires an explicit override to plan in that case.
- `authority(direction)` is in `[0, 1]`: the variance along the direction
  compared against the smallest the evidence achieves anywhere, scaled by the
  share of the direction inside the resolved subspace. It is one along the
  single best-resolved direction, `lambda_i / lambda_max` along any other
  resolved eigendirection, and exactly zero for a direction the evidence does
  not resolve, so a mixed direction is dominated by its worst-resolved
  supported component.
- `information_gain_nats(delta)` is `0.5 log det(I + Lambda^-1 V' dL V)` on the
  directions already resolved. Directions an increment newly resolves are a
  rank change rather than a finite gain, so a rank-zero belief reports zero
  gain and states its progress through its rank instead.
- `seeded_from_members(nominal, members)` is what a family of related vehicles
  or configurations can hand a new belief: the members' sample covariance
  around the nominal inverts, on its supported subspace, to precision, and
  every other direction stays at zero precision, which is to say unknown. It
  invents no precision on directions no member moved.

Rank zero describes a mean with no resolved parameter uncertainty. Its zero
pseudoinverse is useful for linear algebra but cannot represent certainty.
The unresolved basis and completeness flag must accompany it into prediction
and control.

## `ForecastErrorEnvelope`

The envelope is fitted only from held-out rollout endpoints. It gives every
independent source group equal mass, then every trajectory within a group equal
mass, then every endpoint within a trajectory equal mass, and records the
uncentered tangent second moment per horizon along with the raw, effective and
independent-group counts. Zero empirical eigenvalues are absent evidence, not
noiseless measurements. These are forecast-error statistics, not a posterior or
a calibrated probability distribution.

Two things read it: the horizon cap, which shortens a maintained horizon to the
evidence that supports it, and the solver's charged spread. Nothing else
shortens a horizon; the library certifies no prediction horizon of its own.

Group-bootstrap disagreement is one possible future input to the belief. The
IDF-DS evidence says it should not define the abstraction or be the default
error model, and the [literature review](../literature-review.md) records that
result and the commit that carried its code.

## Prediction contract

`belief.rollout(...)` returns a `PredictiveTrajectory`: the state trajectory,
the latent applied-control state, the commands, the forecast-error covariance
and the parameter covariance at every horizon, validity-envelope utilization,
whether the requested horizon is supported by the envelope, and the parameter
information rank, completeness, and unresolved basis behind it. The two
covariances are reported separately because
they answer different questions: the envelope was measured on held-out flights
of the model as it was fitted, and the parameter contribution is the plan's own
sensitivity to the coefficients the evidence resolved. The plan model sums them
into the `tangent_covariance` the solver charges.

Declared command bounds and the validity envelope are enforced differently, on
purpose. Bounds are a hard execution contract: a rollout or runtime transition
given a concrete command outside its channel bounds raises and names the
channel, and only a violation within `1e-6` of the channel span, which is
normalization or serialization rounding, is clipped onto the bound. The
validity envelope is advisory: utilization is reported at every step and the
caller decides, because an out-of-envelope forecast is unsupported rather than
impossible. Bound checks need concrete values, so a JAX-traced command is left
to its caller; NMPC constructs every command inside the bounds.

## `absorb`

```python
updated, result = belief.absorb(telemetry)
```

1. Finite one-step transitions inside the validity envelope are selected at
   the model's sample period. No usable transition returns the original belief
   and a refusal reason.
2. Innovations and structured-parameter Jacobians share one causal actuator
   history, including `Trajectory.control_prefix` on segments. Fitting and
   absorption use the same history extraction.
3. `dL = sum_w J_w' R^-1 J_w` and `L' = L + dL`. Information is added once;
   there is no forgetting or held-out acceptance split.
4. The resolved pseudoinverse gives a direction `d = P sum_w J_w' R^-1 nu_w`.
   Its norm in parameter-scale coordinates is capped at one. Up to twenty
   halvings find a finite physical model whose actual whitened squared error
   plus the prior quadratic `step' L step` does not exceed the incoming error.
   If no such step is found, the mean stays unchanged and the information is
   still recorded. Scaling preserves the unresolved null space.
5. The innovation noise increases, when necessary, to the actual nonlinear
   residual's mean square after the accepted step. A linearized cancellation
   cannot conceal a divergent or poorly fitting model.

`UpdateResult` reports the measured innovation before and after, information
gain on the previously resolved subspace, the step length in prior standard
deviations, the usable window count, and validity utilization. Updates are
functional. The original belief remains unchanged.

The covariance argument for an unscaled linear Gaussian update describes a
statistical null, not a deterministic trust region for a nonlinear model.
`tests/test_absorb.py` checks that null empirically; separate regressions check
large initial model errors, finite physical coefficients, and actual descent.
The [adaptive recovery diagnostic](../validation.md#adaptive-recovery) records
the resulting estimator driving a controller.

## Saving the command interface

Belief format 6 records direct actuator maps with their own command channels
and bounds. Arbitrary JAX/Python maps are recorded as external dependencies;
loading them requires `DynamicsBelief.load(path, actuation=your_map)` or
`ExecutableModel.load(path, actuation=your_map)`. The supplied map must match
the saved channel contract, and the caller owns its calibration. No loader
silently substitutes an identity map for an external map.

Formats 3 through 5 remain readable with their historical command-interface
assumptions; they did not record external mappings.

## Active exploration

Safe exploration needs expected information, not merely large uncertainty. The
belief exposes the pieces that calculation is built from rather than a scoring
entry point of its own: `information.covariance()` and
`information.resolved_subspace()` say what a plan would be improving on,
`information.authority(direction)` says how well one direction is currently
known, `information.information_gain_nats(delta)` prices a candidate increment,
and `belief.rollout(...)` reports the propagated parameter covariance and the
validity utilization along a candidate path. An exploration policy forms
expected information gain from those, on the coordinates and horizon it cares
about.

This supports the conceptual progression the library is built for: arrest and
stabilize on a family seed with broad uncertainty and known command bounds;
exploit the passive excitation the recovery itself provides; probe with
bounded maneuvers whose predicted trajectories stay acceptable under current
uncertainty; expand support only after observed transitions justify it, never
on novelty alone; and increase maneuver complexity by trading tracking against
information gain explicitly rather than hiding excitation in controller noise.

Exploration policy belongs above the model and control layers. Glassbox
supplies the differentiable forecasts, the information geometry and the
evidence update; it does not hard-code a flight-test script into the dynamics
artifact.
