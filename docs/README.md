# Glassbox documentation

Start with [scope](scope.md) for what the library covers and where its
boundary is. Concept pages explain the contracts; experiment pages record the
evidence behind every quantitative claim; `results/` holds the machine-readable
artifact each experiment page cites.

## Concepts

- [Dynamics beliefs and live adaptation](concepts/dynamics-beliefs.md): the
  fitted artifact, its error and parameter evidence, fleet priors, and the
  transactional update.
- [Nonlinear model-predictive control](concepts/nmpc.md): the controller
  interface, eligible artifacts, safety boundaries, PX4 SITL shadow mode, and
  the design and acceptance contract.
- [Flight supervisor](concepts/flight-supervisor.md): the model-independent
  freshness, bounds, attitude, and rate-arrest layer.
- [Bootstrap identification](concepts/bootstrap-identification.md): the
  no-prior contract for local authority identification.
- [Predictive ensembles](concepts/predictive-ensembles.md): the diagnostic
  uncertainty workflow and its promotion boundary.

## Guides

- [PX4 ULogs](guides/px4-ulog.md): extracting canonical trajectories and
  recording reproducible SITL flights.
- [Recorded results](guides/recorded-results.md): the two-tier recorded-result
  tests, when to re-record, and the `glassbox record-results` command.

## Experiments

Multirotor:

- [Nano-Quadrotor benchmark](experiments/nanodrone.md)
- [ARP quadrotor corpus](experiments/arp.md)
- [PX4 SITL multirotor corpora](experiments/px4-sitl-multirotor.md)
- [Adaptive recovery diagnostic](experiments/adaptive-recovery.md)

Fixed wing:

- [IDF-DS telemetry corpus](experiments/idf.md)
- [Skywalker X8 benchmark](experiments/x8.md)
- [Cascade X8 validation](experiments/cascade-x8-validation.md)
- [EPFL TOPOPlane2 reference](experiments/epfl.md)
- [PX4 SITL fixed-wing corpora](experiments/px4-sitl-fixedwing.md)
- [Cross-airframe fixed-wing gate](experiments/fixedwing-gate.md)

Cross-platform:

- [Fitting-policy selection](experiments/fitting-policy.md)

## Background

- [Literature review](literature-review.md)
- [Original proposal, August 2026](history/idea-2026-08.md)

## Related

- [glassbox-throw](https://github.com/Rylandl/glassbox-throw): the Crazyflow
  throw demo built on this package, including the dual-control NMPC design
  and the closed-loop bootstrap, prototype, and throw diagnostics.
