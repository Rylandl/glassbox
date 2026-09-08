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

Each accepted interval updates the collective and angular regressions. Rank
thresholds select command and nuisance directions; `BootstrapEvidence` reports
the resulting ranks, residual scales and authority. Non-finite or out-of-bounds
samples leave the belief unchanged and produce a `last_sample_report` refusal.

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

The returned belief uses `BootstrapMultirotorParams`, the declared command
bounds and `RecursiveBootstrapConfig.sample_period_s`. Its parameter
information comes from the regression Grams and residual scales;
`forecast_error` is `None` because the identifier does not reserve data.

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

A fresh identifier's belief resolves nothing, so its
resolved covariance is zero, while its parameter uncertainty is unknown.
The solver returns a bounded hold with `UNRESOLVED_MODEL` unless the caller
explicitly enables `SolverPolicy.allow_unresolved_parameters`. That override
prices the mean and supported covariance and records incomplete uncertainty.

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

The collective regression uses cumulative body-z specific force and cumulative
features, with an intercept for the anchor. The intercept is eliminated by a
Schur complement and the Gram is scaled by the measured residual variance.
This is an approximate information model: changing attitude and noisy velocity
features can correlate the regression errors. The angular regression uses its
per-interval Gram.
