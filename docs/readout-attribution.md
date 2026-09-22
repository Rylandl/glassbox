# Fast readout: physical attitude response is the main excess feedback path

This no-fit investigation makes the next architectural test more specific. The
latest fast readout's fixed-wing angular recurrence is driven primarily by its
learned attitude-to-angular-acceleration response, especially the body-gravity
direction feature. Delayed-state feedback is not the leading amplifier in the
tested linearization. Cascade simulator truth is far less sensitive to attitude
than either learned model on these two recordings. The fast readout is still not
adopted; production source and public interfaces did not change.

## Frozen data and checks

The committed [recurrence protocol](harness/readout-attribution-v1.json) covers
all 14 saved conditional origins and all five 50 ms steps in each fixed-wing
recording. It reconstructs the saved curvature-only and sensitivity readouts at
each origin, uses the exact issued future commands, and differentiates the same
50-coordinate recurrence on both recorded and free-running states. The
stop-gradient comparison has **identical nominal steps** and retains known force
rotation, gravity, attitude kinematics, filtering and memory. It removes only
derivatives through the learned body-acceleration output. Linearized path
interventions are diagnostics; they are not alternative nonlinear forecasts.

The [plant-truth protocol](harness/readout-attribution-truth-v1.json) replays both
original Cascade recordings from their recorded initial state and actuator
equilibrium, preserving the complete actuator and aerodynamic state at every
origin. Maximum replay difference from the saved 13-state telemetry is
**0.000376 / 0.000204** in the native recorded units. A simulator branch from a
replayed complete state agrees with the unperturbed next state within
**2.98e-8**. Its physical nine-coordinate derivative is verified against
centered finite differences to relative error **0.00100 / 0.00175**, below the
frozen 0.05 float32 limit. These are near-exact replayed simulator states, not
bitwise copies of the original internal state. Simulator internals never enter
the learner.

The model's full Jacobians match the earlier saved Jacobians **exactly** at all
140 measured state/arm points per recording. Full-recurrence finite differences
agree within **1.14e-9 / 7.19e-10** relative error. Model float64 forecasts differ
from the saved float32 forecasts by at most **0.000036 / 0.000109** in relative
whole-state norm. The initially committed verifier incorrectly attempted to
finite-difference a stop-gradient function; that measures the original
function's derivative. It stopped before saving a case. The corrected, committed
verifier uses finite differences on the full map and exact kinematic identities
on the stop-gradient map. Both complete packs verify from saved arrays without
fitting or simulator calls.

## Plant response versus learned response

The local column is the median Frobenius norm of the 50 ms derivative from a
right SO(3) attitude tangent (rad) to body rate (rad/s), across all 70 recorded
states. The 250 ms column is the median largest singular value of the ordered
five-step derivative from origin attitude to final body rate across 14 origins.
The plant derivative lets its hidden state evolve; the Glassbox derivative lets
its filter, lag and memory evolve. Identical coordinate units are used.

| Recording | Model | Local attitude → rate | 250 ms attitude → rate gain |
| --- | --- | ---: | ---: |
| fixedwing-80 | Cascade truth | **2.91** | **3.90** |
|  | Maintained full learner | 208.55 | 467.95 |
|  | Curvature fast readout | 65.66 | 6,782.26 |
|  | Sensitivity fast readout | 56.34 | 6,912.23 |
| fixedwing-81 | Cascade truth | **2.90** | **3.89** |
|  | Maintained full learner | 378.74 | 1,932.86 |
|  | Curvature fast readout | 74.93 | 6,976.23 |
|  | Sensitivity fast readout | 58.77 | 6,274.80 |

All three learners have locally wrong attitude sensitivity on these recordings;
the fast readout has the largest five-step amplification. This is independent
evidence beyond comparison with the full learner. A large tangent gain is not
itself a forecast error: the full learner's 250 ms rate RMSE is **2.70 / 14.52
rad/s**, versus **12.25 / 42.00 rad/s** for the latest fast readout. Different
one-step defects and direction cancellations also affect the realized rollout.
The full learner is not a reliable truth reference simply because one of its
recordings rolls out better.

Replacing just the *learned* attitude-to-rate rows of the latest readout's
linearized five-step recurrence with known-propagation rows lowers median
attitude-to-rate gain to **0.102 / 0.143** of its original free-running value,
at **14/14 origins in each recording**. Replacing its attitude-to-force rows
alone leaves gain essentially unchanged. Zeroing latent-to-physical return
instead raises median gain to **1.28 / 1.52** of original; it helps only
**0/14 and 3/14** origins. These path deletions do not recompute a nonlinear
forecast and cannot predict the accuracy of a refitted model. They do reject
the narrower hypothesis that delayed-state return alone explains the excess.

## Which attitude feature carries the derivative?

The separate committed [feature-path protocol](harness/readout-attitude-paths-v1.json)
applies the exact chain rule to the instantaneous angular-acceleration head at
all 70 recorded states per case and arm. It holds past lag, filter and memory
fixed, and splits an attitude perturbation between supported body-relative
velocity and body-gravity direction. The two 3×3 components sum to an independent
physical-attitude autodiff derivative within **3.64e-12**.

| Latest readout, median norm in rad/s² per rad | fixedwing-80 | fixedwing-81 |
| --- | ---: | ---: |
| Through body-relative velocity | 567 | 112 |
| Through body-gravity direction | **702** | **792** |
| Total, including direction/cancellation | 994 | 978 |

The previous sensitivity penalty perturbed body velocity and rate features but
held gravity direction fixed. Its small effect on the complete recurrence now
has a concrete explanation: the omitted gravity-direction path is the larger
instantaneous contribution in both cases, especially recording 81. There are
also very large body-velocity contributions at some early origins, so deleting
gravity as a feature without refitting would not be a justified fix. The head
numbers are continuous acceleration derivatives; compare them with each other,
not directly with the discrete 50 ms plant gains in the preceding table.

## Decision

The evidence warrants one frozen, generic **physical-attitude sensitivity**
experiment. Perturb the actual SO(3) state so body-relative velocity and gravity
direction change together, and penalize the resulting angular-acceleration
derivative in physical output units while retaining the direct readout solve.
Do not impose global contraction or a vehicle-family rule: the true plant has
nonzero response and other systems may have legitimate unstable modes. Keep the
curvature-only and earlier body-motion penalty readouts as exact controls, and
score one-step, 50–250 ms physical errors, initialization and whole-update cost.
The body-rate failure is a capability gap, not an automatic veto from one flag.

Before that fit, collect the already planned paired quad 10/50 ms observation
fixture with commands held through each 50 ms interval and equal elapsed prefix
time. This isolates the sampling-rate confound and gives the new penalty a
cleaner cross-family test. If the physical-attitude penalty does not improve
realized rollout accuracy, the earlier short-recursive-loss control suggests
testing a bounded trajectory-fitting correction rather than another proxy
strength sweep. No pretraining, fleet prior, vehicle metadata or controller
claim enters either path.

The authenticated pack hashes and source revisions are in
[readout-attribution.json](readout-attribution.json). Both recordings come from
one Skywalker-X8 configuration; this diagnoses a known-data failure and does
not establish performance on a held-out vehicle.
