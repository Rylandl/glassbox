# Dynamics beliefs

Glassbox's primary product object is a dynamics belief, not a point parameter
file and not a bootstrap ensemble. A belief is three things and nothing else:

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
reports validity, and still serializes; every method that needs a command space
raises `NonActionableModelError` and says so. The belief is also the one
artifact the library writes; a bare model payload is still read, as a belief
with rank-zero information and no envelope.

One belief type carries either model family. A fit over a corpus produces a
belief whose model is a structured multirotor or fixed-wing model; the
in-flight `RecursiveBootstrapIdentifier` produces a belief whose model is the
bootstrap parameterization, `BootstrapMultirotorParams`, whose free parameters
are the direct command effects it can learn from motor input and output alone.
Both are an `ExecutableModel`, a `ParameterInformation` over that model's
structured parameters, and a forecast-error envelope when held-out evidence
measured one. Nothing downstream branches on which: `plan_model` takes either,
and the solver reads a mean, a tangent covariance and a command box.
[Bootstrap identification](bootstrap-identification.md) covers what the second
family estimates and what it deliberately does not.

The motivating runtime is broader than ordinary batch identification. A vehicle
may enter with only a family seed, stabilize using conservative authority
estimates, learn from the resulting motion, and then choose increasingly
informative maneuvers while respecting the support accumulated so far. The
architecture permits that lifecycle because information is a first-class,
monotone quantity rather than a label.

## Product contract

```python
outcome = glassbox.fit(flight_paths, glassbox.FitSpec())
belief = outcome.belief

# or, from an artifact a previous fit wrote:
belief = glassbox.DynamicsBelief.load("artifacts/vehicle-belief.json")

forecast = belief.rollout(initial_state, commands)
belief, update = belief.absorb(recent_telemetry)

controller = glassbox.NMPCController(belief)
result = controller.solve(state, reference, previous_command)
```

There is no recalibration step and no lifecycle to keep in mind. `absorb`
returns a belief that is strictly better informed than the one it was given,
and the controller reads that belief directly.

## The twelve local coordinates

Prediction error, parameter sensitivity and innovation noise are all stated in
the same twelve rigid-body local coordinates: position, velocity, shortest-path
attitude rotation vector, and body angular velocity. `rigid_body_local_error`
is the map into them and `state_plus_tangent` is the retraction back out, so a
tangent-space error, a tangent-space perturbation and a tangent-space
covariance describe the same thing. Quaternion components are never assigned
Euclidean covariance or independent error bars.

## What the fit produces

`glassbox.fit(sources, spec)` returns a `FitOutcome`: the belief the fit
supports, the report that records how it was produced, and one belief per
requested ablation. `FitSpec` carries the user's choices, with the loss
geometry in a `LossPolicy` and the training weighting in a `WeightingPolicy`,
and the `Holdout` names which evidence the fit is not allowed to see.

Fitted parameters are effective predictive coefficients, not uniquely
recovered physical constants. Complete-flight holdout results test
cross-flight generalization; the `no_lag` ablation's ratios above one say that
modeling latent applied-control response improved prediction on evidence the
fit did not see. Every training horizon is normalized by its own initial
loss, and several are then combined with equal weight. Each labeled source group
contributes equal total loss weight with uniform weight inside the group;
without source groups each training flight contributes equally, and when every
flight declares a maneuver profile each family contributes equally before its
replicates split it. Large candidate sets are deterministically thinned across
every group's timeline under an automatic corpus- and horizon-aware compute
budget. Every model class uses equal semantic state-group loss after scaling by
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
belief's `innovation_noise`, and it is measured whether or not the caller asked
for parameter evidence: a belief that cannot say how wrong its one-step
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

`FitSpec.parameter_evidence` additionally accumulates
`sum_w J_w' R^-1 J_w` over the training flights' one-step transitions, with the
fitter's estimable mask, a bounded window budget spread evenly across the
independent source groups, and the balanced effective count. Without it the
belief is an honest point estimate that knows its own noise and is ready to
absorb.

This is deliberately the same estimator `absorb` runs. The noise model is
one-step innovation covariance, so the windows it weights correctly are
one-step windows: whitening a multi-step endpoint by it would overstate the
information by roughly the horizon and would count the same transition once per
training horizon. A fitted belief and a belief that has absorbed telemetry
therefore state their information in one currency and can be added.

## `ParameterInformation`

```python
ParameterInformation(names, precision, scale, estimable, innovation_noise,
                     noise_floor, effective_count, rank_relative_tolerance)
```

`precision` is the accumulated information in parameter coordinates. `scale`
defines the normalized coordinate `u` by `parameters = center + diag(scale) u`,
so the normalized precision `diag(scale) @ precision @ diag(scale)` has
eigenvalues comparable across coordinates of different physical units, and the
rank test is stated on those: a direction is *resolved* when its normalized
eigenvalue exceeds `rank_relative_tolerance` times the largest.
`estimable` is the mask the fitter declares; a coordinate outside it is held
fixed by construction, such as the diagonal of the angular control coupling
that the effective angular authority already carries, and its rows and columns
of the precision are exactly zero.

- `resolved_rank()` and `resolved_subspace()` report what the evidence has
  resolved.
- `covariance()` is the pseudo-inverse of the precision on that subspace. An
  unresolved direction has **exactly zero** variance rather than a large one:
  this object is what is known, and inventing a spread along a direction it
  says nothing about would be an assumption. The complementary statement, that
  such a direction may be arbitrarily wrong, is carried by the rank.
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

Rank zero is a point estimate. Inverting a rank-deficient information matrix
would assign zero variance to directions the flight never excited; the
pseudo-inverse on the resolved subspace assigns them zero *information*
instead, and the update takes no step along them.

## `ForecastErrorEnvelope`

The envelope is fitted only from held-out rollout endpoints. It gives every
independent source group equal mass, then every trajectory within a group equal
mass, then every endpoint within a trajectory equal mass, and records the
uncentered tangent second moment per horizon along with the raw, effective and
independent-group counts. Zero empirical eigenvalues are absent evidence, not
noiseless measurements. These are forecast-error statistics, not a posterior or
a calibrated probability distribution.

Two things read it: the NMPC horizon cap, which shortens a maintained horizon
to the evidence that supports it, and the solver's charged spread.

Group-bootstrap disagreement is one possible future input to the belief. The
IDF-DS evidence shows that it should not define the abstraction or be the
default error model: independently calibrated residual error was more useful
than adaptive bootstrap spread on that corpus.

## Prediction contract

`belief.rollout(...)` returns the state trajectory, the latent applied-control
state, the commands, the forecast-error covariance and the parameter covariance
at every horizon, validity-envelope utilization, and whether the requested
horizon is supported by the envelope. `tangent_covariance` is the sum of the
two covariances: the envelope was measured on held-out flights of the model as
it was fitted, the parameter contribution is the plan's own sensitivity to the
coefficients the evidence resolved, and they answer different questions.

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

1. One-step windows are taken at the belief's own sample period. A sample
   outside the model's validity envelope, or a non-finite one, is dropped
   rather than downweighted: the fit never claimed to describe that region, so
   an innovation measured there would be charged against parameters that were
   never fitted to explain it. If nothing remains the belief comes back
   unchanged with `absorbed=False` and a reason.
2. Each window contributes an innovation, measured minus predicted from the
   measured previous state under the applied command, with the latent actuator
   state carried causally, and a Jacobian of the predicted endpoint tangent
   with respect to the structured parameters, restricted to the estimable
   coordinates.
3. `dL = sum_w J_w' R^-1 J_w` and `L' = L + dL`. There is no forgetting factor
   and no discount. Information accumulates, always.
4. `P` is the pseudo-inverse of `L'` on its resolved subspace and the step is
   `P sum_w J_w' R^-1 nu_w`. It is exactly zero along every unresolved
   direction by construction, and a well-resolved direction moves less than a
   poorly resolved one for the same innovation, because the step is bounded by
   what the belief already knows rather than by a declared trust region.
5. `R' = max(R, mean_w (nu_w - J_w dtheta)^2)`, per coordinate. Realized error
   can raise the noise floor only by the part the step did not explain: the
   error an empty belief makes is explained away by its own first step and does
   not get recorded as irreducible noise. This is the prequential lesson stated
   so that it discounts ignorance.

`UpdateResult` carries `absorbed`, the `reason` when it is not, the window
count, the whitened one-step innovation before and after, the information gain
in nats, the step's length in the metric of the prior precision, and the worst
validity utilization the evidence reached. `provenance` accumulates the update
count and `parameter_distance_since_measurement`, the normalized distance the
parameters have moved since the envelope was measured. Nothing gates on that
distance; it is the number a caller reads to decide whether an envelope
measured around older parameters still describes the model in hand.

Updates are functional: `absorb` returns a new immutable belief and never
mutates the one it was given.

### The pinned step-size property

Under the null the belief is already at the true parameters and sees telemetry
whose one-step innovations are i.i.d. with exactly the declared covariance `R`.
The step then has covariance `P dL P = P - P L P`, which is at most the
posterior covariance `P` and therefore at most the prior covariance. So the
mean step over `S` independent seeds, measured along any resolved direction in
prior standard deviations, has standard deviation at most `1 / sqrt(S)`.
`tests/test_absorb.py` pins that at `S = 64` seeds and `k = 4`, a bound of
`0.5` prior sigma with a two-sided level near `6e-5` per direction. The
constant is derived from the inequality above, not from what a run produced.

This replaces the 64-seed null-acceptance calibration the transactional update
carried. That statistic answered a question about a gate: how often does the
threshold let a step through when there is nothing to learn. The recursive
update has no gate, so the honest question is about size rather than
acceptance: when there is nothing to learn, is the step as small as the
belief's own covariance says it should be.

### What was retired, and why

The transactional update was deleted at commit `8d3400f`. It proposed a bounded
local move on early telemetry, validated it on a disjoint later split, and
committed only past a two-sigma improvement margin, then marked the held-out
error evidence stale. It also carried a maximum-norm trust bound, a line
search, revision fingerprints and replay detection, and a 35-field report.

The reason it went is not that it was expensive. It is that on every path a
shipped artifact could reach, the account of what is unknown never changed. The
fit wrote total-forecast-scoped error evidence, and every mechanism that would
have contracted or grown the parameter information was gated on a conditional
innovation scope that nothing produced. The recorded adaptive-recovery artifact
said so plainly: covariance not updated, posterior trace equal to prior trace,
information gain null. Worse, a commit zeroed the error moments, so the
controller lost its horizon cap and saw zero model uncertainty: adaptation made
it more confident than its evidence supported. The two-sigma margin and the
trust bound existed to patch the null-acceptance rate of a gate that the owner
had already decided against, and an improvement threshold is exactly the
mechanism the design does not want.

The recursive information update was already running elsewhere in the library,
in the in-flight identifier, and it is the design that works: a step bounded by
the current covariance, zero along unresolved directions, no proposal, no
split, no margin, no staleness. The runtime forecast bias went with the
transaction, because it was the root cause of the staleness lifecycle; the
held-out mean error it applied stays in the fit report as a number.

## NMPC

The control loop consumes a compact runtime belief: one differentiable model,
the resolved parameter covariance, the forecast-error envelope, typed validity
support, and no training data or optimizer state.

NMPC optimizes the expected cost of its own forecast rather than the cost of
the predictive mean alone. Predicted tangent spread is charged in the tracking
cost against the same physical tracking tolerances the objective already uses,
and the model-validity term is widened by the marginal standard deviation of
the six envelope features, so a belief that knows less plans nearer to ground
it has evidence for. The parameter contribution is written through a factor of
the covariance, so a belief that resolves two directions costs two extra
forward rollouts rather than a full Jacobian, and a point belief costs nothing
and is priced by the point objective. The normal horizon is capped at the
forecast-error evidence that supports it.

Nothing edits the command after optimization. The mechanism is vehicle-agnostic
and cannot generate a command from a separate attitude, rate, mixer, or
airframe-specific control law. Optimizer failure stays explicit and returns only
a bounded hold with `command_usable=False`; it does not silently transfer
authority to another controller.

The model-validity envelope and predictive uncertainty have different meanings.
The former asks whether a query resembles observed operating conditions; the
latter asks how wrong predictions were within the evidence. Both remain visible
to the controller and neither substitutes for the other.

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

This supports the conceptual progression:

1. **Arrest and stabilize.** Use a family seed, broad uncertainty, known command
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
- the matched held-out forecast envelope becomes the honest initial
  implementation; and
- future error or information candidates can be compared without another
  system-wide artifact migration.
