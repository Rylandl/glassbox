# No-prior multirotor bootstrap identification

`RecursiveBootstrapIdentifier` is how Glassbox obtains a belief for a vehicle
with no prior. It produces the same `DynamicsBelief` a fit produces, over a
deliberately incomplete model family, `BootstrapMultirotorParams`, that
estimates only what is needed to establish local collective and three-axis
angular authority:

- four motor-command effects on body-specific-force z;
- the full `3 x 4` motor-command effect on body angular acceleration;
- linear and quadratic body-rate nuisance terms; and
- a linear body-velocity nuisance term on the collective.

A level zero-velocity hover command and the command subspace the evidence
supports follow from those, and are reported rather than fitted.

It receives canonical rigid-body states, measured applied motor inputs, an
interval and command bounds. It does not receive a motor mixer, hover command,
mass, inertia, arm length, thrust coefficient, fleet model or nominal
`DynamicsParams`. Requested commands are insufficient, because actuator lag
can make them materially different from the inputs that affected the vehicle.
For the same reason the family carries no actuator lag of its own: its latent
applied command *is* the command, because the identifier regressed on the
command the vehicle applied.

The identifier updates its belief after every measured actuation interval,
accumulating the two regressions' Gram matrices and solving them again each
time. Motor effects are fit only in singular-vector directions supported by
the accumulated applied inputs. Unsupported directions remain zero, and each
output direction is granted authority in proportion to what the evidence
spans, how strong the weakest supported information direction is, and how far
the fitted effect stands above its own residual noise. There is deliberately
no evidence-collection and model-running phase boundary, and no certification
step: authority per direction is what governs.

The nuisance block follows the same rule. Body velocity, body rate and rate
product columns are inverted only along directions the evidence actually
excited, at `nuisance_rank_relative_tolerance` of the leading nuisance
direction. The constant intercept column supplies the unit scale for that
comparison: its root-mean-square is exactly one, so a relative threshold is
also an absolute floor in each feature's own units. Without it, a feature that
barely moves takes a coefficient set by measurement noise divided by a
near-zero excursion, and that coefficient then enters every later prediction.
`collective_nuisance_rank` and `angular_nuisance_rank` report how many
directions survived. The threshold bounds unexcited directions only; a weakly
but genuinely excited direction is still inverted, and its coefficient is only
as good as its signal-to-noise ratio.

`RecursiveBootstrapIdentifier` never ends a control loop over one bad sample.
`update` refuses a non-finite or out-of-bounds transition, leaves the belief
exactly as it was, and records the refusal in `last_sample_report`. Applied
commands within a rounding width of the bounds are clipped rather than
refused.

## The belief it produces

```python
import numpy as np

from glassbox import RecursiveBootstrapConfig, RecursiveBootstrapIdentifier

identifier = RecursiveBootstrapIdentifier(RecursiveBootstrapConfig())

rng = np.random.default_rng(0)
state = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
for _ in range(40):
    previous_state = state
    state = previous_state + 0.001 * rng.standard_normal(13)
    state[6:10] /= np.linalg.norm(state[6:10])
    applied_command = 0.5 + 0.1 * rng.standard_normal(4)
    belief = identifier.update(previous_state, state, applied_command, 0.01)

evidence = identifier.evidence
```

`belief` is a `DynamicsBelief` and nothing about it is special-cased
downstream. Its `model` is an `ExecutableModel` over
`BootstrapMultirotorParams`, actionable through the identity actuation map on
the declared command box, executing at
`RecursiveBootstrapConfig.sample_period_s`. Its `forecast_error` is `None`,
because nothing held any evidence out. Its `information` is a
`ParameterInformation` over the family's 41 structured parameters: the two
accumulated Grams, congruence-transformed into the parameters' own coordinates
and divided by each regression's residual variance, so one unit of precision
is one transition's worth of information at the estimated residual scale. The
angular Gram is shared by the three body axes and enters once per axis at that
axis's own residual. Every coordinate is estimable, nuisance terms included,
and `resolved_rank()`, `covariance()` and `authority(direction)` mean exactly
what they mean for a fitted belief.

The normalized coordinate the rank test is stated in is declared by
`information.scale`: one unit is the coefficient perturbation that moves that
regression's prediction by one residual standard deviation at the reference
excitation, the full command box for a command coefficient and a unit feature
for a nuisance one. That is what makes a collective coefficient in metres per
second squared and an angular one in radians per second squared comparable.

Because the model is a `DynamicsBelief`, the plan-model seam takes it
unchanged:

```python
import glassbox
from glassbox import ReferenceTrajectory, SafetyEnvelope, TrackingTolerances

tolerances = TrackingTolerances.for_platform("multirotor")
envelope = SafetyEnvelope(
    minimum_position_m=(-100.0, -100.0, -20.0),
    maximum_position_m=(100.0, 100.0, 100.0),
)

plan = glassbox.plan_model(belief, tolerances, envelope)
solver = glassbox.BoundedShootingSolver(
    plan, glassbox.SolverPolicy(plan.horizon_steps, plan.block_count)
)
reference = ReferenceTrajectory(
    np.repeat(state[None, :], plan.horizon_steps + 1, axis=0)
)
result = solver.solve(state, reference, applied_command)
```

The same `plan_model` and the same `BoundedShootingSolver` serve a belief
fitted from a corpus and a belief built in flight from nothing. The solver
never learns which it is planning over: it reads a mean, a tangent covariance
and a command box. A fresh identifier's belief resolves nothing, so its
covariance is exactly zero and the point objective prices the plan.

`BootstrapEvidence` carries what is specific to this estimator rather than to
beliefs in general: the two accumulated Grams in the identifier's own feature
order, the residual scales, the ranks and support projectors its declared
tolerances define, the per-axis authority they imply, the exploration
completion, and the hover command when the collective map implies one inside
the command box. `evidence.supported` is the condition control cares about:
four motors spanned, three angular axes spanned, and a hover command in the
box. The same summary is recorded in
`belief.provenance["bootstrap_evidence"]`, so a saved belief carries its own
audit trail.

## Aggregated transitions

The identifier assimilates one sample per `TRANSITION_AGGREGATION_STEPS`
measured transitions: the window's mean features and mean targets, weighted by
the window length. The width is two and is not configurable.

The reason is measurement noise on a differenced target. The collective target
is the specific force implied by the velocity change over one interval, so
white noise on the measured velocity is multiplied by the loop rate. The
arithmetic makes the scale plain: at a hundred hertz, two centimetres per
second of velocity noise is a few metres per second squared of target noise,
which is the same order as the signal a probe of a tenth of the command range
produces. The mean over a window telescopes most of that away, because
consecutive differences share their inner samples with opposite signs.
Weighting the aggregated sample by the window length keeps the sample count,
the support thresholds and the residual floor exactly per transition, so with
a noise-free measurement the information rate is unchanged for a command held
across the window, and under noise the residual variance the identifier
estimates falls by about the window length.

The cost is the variation inside the window: a window that straddles two
excitation blocks averages their difference away. The width was chosen by
paired measurement on the release ensemble of the dual-control design, whose
artifacts live in the
[glassbox-throw](https://github.com/Rylandl/glassbox-throw) repository rather
than here; wider windows were measurably worse for that reason, and two was
the width that measured best.

## Integrated collective fit

The collective map is fit on the cumulative target rather than the
per-interval one. The per-interval target is the body-z specific force implied
by the velocity change over one interval, so measurement noise on the velocity
enters divided by the interval. Summed over the transitions since the anchor,
the projected velocity changes telescope: the sum carries the anchor's noise
once, common to every row, the latest sample's noise once, and a small term
from how far the body axis rotated each step. Regressing the cumulative target
on the cumulative features with one constant column for the anchor is
therefore the exact least-squares form for white measurement noise on the
velocity, and its rows are independent given that column.

The integrated system's information is exported to the rest of the identifier
as an equivalent per-transition Gram: the anchor column is marginalized, the
system's residual scale is estimated from its own residual with the declared
force floor over one interval as its minimum, and the marginal is rescaled so
that dividing it by the declared floor, which is the residual then reported,
gives the integrated precision. Support, authority, the belief's information
state and any planner reading it therefore see the honest information without
changing. One unit of precision on the collective block is one transition's
worth of information at that declared floor. The angular regression is
untouched, and its Gram is the plain accumulated outer products.

Two things this settles. In a noise-free simulation the form behaves like the
per-interval one. Under measurement noise it says what the per-interval form
cannot: the collective level is known quickly, while the differential
coefficients, whose probes integrate to a small fraction of the velocity
noise, are not, and the per-interval fit's confidence in them was optimism. A
first version with three world-axis rows per transition was measured and
dropped: the model explains only the body-z force, so the other two rows
carried unmodeled force and the residual came out far above the floor.
