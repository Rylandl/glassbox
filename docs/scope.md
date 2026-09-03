# Scope

Glassbox identifies differentiable vehicle dynamics from recorded state and
actuator telemetry, wraps the fitted model in an explicit statement of its
predictive error and parameter uncertainty, and uses that belief for bounded
online adaptation and model-predictive control. This page states what the
library covers today, what evidence backs it, and where the boundary is. The
original August 2026 proposal is kept as [history](history/idea-2026-08.md);
several things it listed as out of scope have since been built.

## The question

Given recorded vehicle state and actuator telemetry, can Glassbox identify a
differentiable dynamics model that accurately and efficiently predicts unseen
vehicle motion, and can that model carry an honest account of its own
uncertainty into control?

## System boundary

The model represents the vehicle plant below the flight controller. Its inputs
are the lowest-level actuator commands reliably present in telemetry, never
position, velocity, or attitude setpoints, so the learned dynamics do not absorb
controller and estimator behavior.

The canonical rigid-body state is position, velocity, attitude, and body rate
in a north-west-up world frame with a forward-left-up body frame and a
scalar-first unit quaternion (`RIGID_BODY_STATE_SCHEMA`). Actuator response is
carried as a latent applied-control state when it is not measured. Known
kinematics are kept fixed; the identification problem is the map from actuator
behavior to forces, torques, and accelerations.

## What exists

- **Dynamics and identification.** Structured multirotor and fixed-wing
  rigid-body models with analytic actuator lag, an optional compact residual,
  RK4 rollouts in JAX, and multi-horizon rollout fitting with group-balanced
  losses. Evaluation uses complete-flight and maneuver-family holdouts against
  a kinematic persistence baseline.
- **Dynamics beliefs.** The fitted artifact is a `DynamicsBelief`: the nominal
  model, held-out predictive error in 12 local rigid-body coordinates,
  rank-aware local parameter information, a validity envelope, and update
  provenance. Fleet priors can be built from several beliefs. See
  [dynamics beliefs](concepts/dynamics-beliefs.md).
- **Online adaptation.** A transactional propose, validate, commit update from
  recent telemetry that returns the original belief when disjoint later
  telemetry does not improve. See the same page.
- **Bootstrap identification.** For a vehicle with no prior, a smaller contract
  that fits collective acceleration and a motor-to-angular-acceleration map from
  applied motor inputs, offline and recursively online. See
  [bootstrap identification](concepts/bootstrap-identification.md).
- **Control.** A bounded JAX NMPC controller driven by a belief, with hard
  command bounds, explicit failure statuses, and belief-aware support
  projection, plus a model-independent flight supervisor. See
  [NMPC](concepts/nmpc.md) and [the supervisor](concepts/flight-supervisor.md).
- **Telemetry and corpora.** PX4 ULog ingestion for multirotors and fixed
  wings, scripted PX4 SITL recording, and adapters for the Nano-Quadrotor, ARP,
  IDF-DS, Skywalker X8, and EPFL TOPOPlane2 reference datasets. See the
  [PX4 ULog guide](guides/px4-ulog.md) and the experiment pages.
- **Simulator integrations.** The Cascade plant used as an independently
  implemented vehicle for closed-loop diagnostics. The Crazyflow integration
  and its dual-control NMPC throw demo moved to
  [glassbox-throw](https://github.com/Rylandl/glassbox-throw).

## Evidence standard

Every quantitative claim in the documentation points to a recorded artifact in
`docs/results/` or to a reproducible command. Holdouts are complete flights,
recording sessions, or maneuver families, never mixed samples. Normalization
statistics come from training data only. Negative results and withdrawn
approaches are kept in the record rather than removed. Absolute timing figures
are kept out of prose because they depend on the host and its load.

## Current boundary

Both vehicle families have differentiable rollout, fitting, serialization, and
PX4 ULog ingestion. Fixed-wing ingestion joins motor and servo allocator
topics, reconstructs signed aerodynamic-axis controls from logged allocation
parameters, and rejects unverifiable mappings.

The present boundary is model validity. Fixed-wing short and medium rollouts
transfer across maneuver families, recording sessions, and both a conventional
tail configuration and a three-control flying wing. Nano-Quadrotor performance
is competitive with its published structured-residual reference, while the
second multirotor airframe shows that normalized-command translation is not yet
consistently better than kinematic persistence. Parameter artifacts remain
airframe specific: the evidence validates the shared interfaces and testable
model hypotheses, not zero-shot coefficient transfer.

Multi-minute IDF sessions, complete X8 maneuvers, and the protected ARP flight
still expose long-rollout instability. Flap-equipped and configuration-changing
aircraft remain unvalidated. More same-vehicle segments are therefore lower
value than another airframe or configuration, or a better generally applicable
command-to-force model. Use `--include-ground` only with a model that includes
ground-contact dynamics.

## Not claimed

- Flight safety or certification of any kind. Every controller result is a
  simulation diagnostic with bounded commands, not a safety case.
- Hard real-time guarantees. Solve and update times are recorded per run and
  marked nondeterministic.
- Hardware demonstrations. Physical propeller, estimator startup, and hand
  contact remain outside every recorded result.
- Zero-shot transfer of fitted coefficients between airframes.
