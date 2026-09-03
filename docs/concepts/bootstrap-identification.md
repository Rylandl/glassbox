# No-prior multirotor bootstrap identification

`RecursiveBootstrapIdentifier` is the deliberately incomplete model Glassbox
fits for a vehicle with no prior. It estimates only what is needed to establish
local collective and three-axis angular authority:

- four motor-command effects on body-specific-force z;
- the full `3 × 4` motor-command effect on body angular acceleration;
- linear and quadratic body-rate nuisance terms;
- a level zero-velocity hover command; and
- the command subspace actually supported by the evidence.

It receives canonical rigid-body states, measured applied motor inputs, an
interval, and command bounds. It does not receive a motor mixer, hover command,
mass, inertia, arm length, thrust coefficient, fleet model, or nominal
`DynamicsParams`. Requested commands are insufficient because actuator lag can
make them materially different from the inputs that affected the vehicle.

The identifier updates a working belief after every measured actuation
interval, accumulating the two regressions' Gram matrices and solving them
again each time. Motor effects are fit only in singular-vector directions
supported by the accumulated applied inputs. Unsupported directions remain
zero, and each output direction is granted authority in proportion to what the
evidence spans, how strong the weakest supported information direction is, and
how far the fitted effect stands above its own residual noise. There is
deliberately no evidence-collection/model-running phase boundary.

The nuisance block follows the same rule. Body velocity, body rate, and rate
product columns are inverted only along directions the evidence actually
excited, at `nuisance_rank_relative_tolerance` of the leading nuisance
direction. The constant intercept column supplies the unit scale for that
comparison: its root-mean-square is exactly one, so a relative threshold is
also an absolute floor in each feature's own units. Without it, a feature that
barely moves takes a coefficient set by measurement noise divided by a
near-zero excursion, and that coefficient then enters every later prediction.
`collective_nuisance_rank` and `angular_nuisance_rank` report how many
directions survived. The threshold bounds unexcited directions only. A weakly
but genuinely excited direction, such as a rate product over a short evidence
window, is still inverted and its coefficient is still only as good as its
signal-to-noise ratio.

`RecursiveBootstrapIdentifier` never ends a control loop over one bad sample.
`update` refuses a non-finite or out-of-bounds transition, leaves the belief
exactly as it was, and records the refusal in `last_sample_report`. Applied
commands within a rounding width of the bounds are clipped rather than refused.


## Aggregated transitions

`RecursiveBootstrapConfig.transition_aggregation_steps` makes the identifier
assimilate one sample per that many measured transitions: the window's mean
features and mean targets, weighted by the window length. One, the default,
is bit-for-bit the identifier as it was.

The reason is measurement noise on a differenced target. The collective
target is the specific force implied by the velocity change over one
interval, so white noise on the measured velocity is multiplied by the loop
rate: at a hundred hertz, two centimetres per second of velocity noise
becomes nearly three metres per second squared of target noise, against under
two from a probe of a tenth of the command range. The mean over a window
telescopes most of that away, because consecutive differences share their
inner samples with opposite signs. Weighting the aggregated sample by the
window length keeps the sample count, the support thresholds, and the
residual floor exactly per transition, so with a noise-free measurement the
information rate is unchanged for a command held across the window, and
under noise the residual variance the identifier estimates falls by about
the window length.

The cost is the variation inside the window: a window that straddles two
excitation blocks averages their difference away. On the throw study a
window of five slowed identification by half; two and three are measured on
the release ensemble in the dual-control NMPC design, now documented in the
[glassbox-throw](https://github.com/Rylandl/glassbox-throw) repository.


## Integrated collective fit

`RecursiveBootstrapConfig.integrated_collective` fits the collective map on
the cumulative target rather than the per-interval one. The per-interval
target is the body-z specific force implied by the velocity change over one
interval, so measurement noise on the velocity enters divided by the
interval. Summed over the transitions since the anchor, the projected
velocity changes telescope: the sum carries the anchor's noise once, common
to every row, the latest sample's noise once, and a small term from how far
the body axis rotated each step. Regressing the cumulative target on the
cumulative features with one constant column for the anchor is therefore
the exact least-squares form for white measurement noise on the velocity,
and its rows are independent given that column.

The integrated system's information is exported to the rest of the
identifier as an equivalent per-transition Gram: the anchor column is
marginalized, the system's residual scale is estimated from its own residual
with the declared force floor over one interval as its minimum, and the
marginal is rescaled so that dividing it by the declared floor, which is the
residual then reported, gives the integrated precision. Support, authority,
the belief, and any planner reading the belief see the honest information
without changing. The angular regression is untouched.

Two things this settles. In a noise-free simulation the form behaves like
the per-interval one. Under measurement noise it says what the per-interval
form cannot: the collective level is known well within a tenth of a second,
while the differential coefficients, whose probes integrate to a few
millimetres per second against two centimetres of noise, are not, and the
per-interval fit's confidence in them was optimism. A first version with
three world-axis rows per transition was measured and dropped: the model
explains only the body-z force, so the other two rows carried unmodeled
force and the residual came out forty times the floor.
