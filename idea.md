# Glassbox: Project Overview

*Working title: Glassbox*

## Summary

Glassbox investigates whether recorded vehicle telemetry can be used to identify a fast, differentiable model of the vehicle's dynamics. The model should predict how the vehicle state evolves in response to actuator commands, reproduce unseen flights over meaningful rollout horizons, and expose useful derivatives with respect to its state, inputs, and parameters.

The initial platform is a simulated multirotor aircraft using PX4 Software in the Loop (SITL) and PX4 ULog telemetry. A simulator provides a controlled environment in which the true state and underlying vehicle behavior are available for evaluation. The same identification pipeline can later be applied to estimated-state telemetry and physical PX4 flight logs.

The project is deliberately limited to telemetry-driven dynamics identification. Active experiment design, trajectory optimization, model-predictive control, docking, and other downstream control demonstrations are possible future projects, not requirements for the initial work.

## Primary Question

The central question is:

> Given recorded vehicle state and actuator telemetry, can Glassbox identify a differentiable dynamics model that accurately and efficiently predicts unseen vehicle motion?

The model may be represented in continuous time,

\[
\dot{x} = f_\theta(x, u),
\]

or as a discrete transition model,

\[
x_{t + \Delta t} = F_\theta(x_t, u_t, \Delta t),
\]

where \(x\) is the vehicle state, \(u\) is the actuator input, and \(\theta\) contains learned physical parameters or model weights. The resulting implementation should be differentiable with respect to \(x\), \(u\), and \(\theta\).

## Why This Matters

Accurate vehicle models are useful for simulation, analysis, control design, trajectory optimization, and adaptation. First-principles models provide valuable structure, but their parameters can be difficult to measure and they often omit platform-specific behavior such as:

- Motor and actuator response delays
- Nonlinear thrust and torque mappings
- Aerodynamic drag and other velocity-dependent effects
- Payload and inertia variation
- Battery-dependent behavior
- Controller-to-actuator scaling and saturation
- Timing, filtering, and estimation effects present in real telemetry

Flight controllers and simulators already record large quantities of synchronized state, sensor, and actuator data. If those logs are sufficient to recover a useful differentiable model, a vehicle could be characterized from ordinary recorded operation without requiring a complete hand-derived model or extensive specialized measurement equipment.

## System Boundary

The initial model should represent the vehicle plant below the high-level flight controller. Its inputs should be the lowest-level actuator commands that are reliably available in the telemetry, rather than position, velocity, or attitude setpoints.

Using high-level setpoints as inputs would cause the learned dynamics to include PX4 controller and estimator behavior. That may be useful for predicting a complete closed-loop system, but it would not isolate the vehicle dynamics and would make the result less reusable.

A useful initial state is:

\[
x = (p, v, q, \omega, m),
\]

where \(p\) is position, \(v\) is linear velocity, \(q\) is attitude, \(\omega\) is angular velocity, and \(m\) represents measured or latent actuator state. The exact state can change with the available telemetry. If motor speed is not measured, motor response may need to be represented as a latent dynamical state inferred from actuator commands.

Known kinematic relationships should be preserved where practical. For example, position and attitude evolution need not be relearned from scratch when their mathematical structure is known. The main identification problem is the relationship between actuator behavior, forces and torques, and translational and rotational acceleration.

## Initial Data Source

The first data source is PX4 SITL, with the simulator treated as the unknown vehicle and PX4 ULog files treated as the telemetry record. Simulation provides two deliberately separated views of each flight:

- **Privileged state:** simulator ground truth used for early fitting, evaluation, and debugging.
- **Operational telemetry:** estimated states, sensor measurements, and actuator signals that resemble what would be available on physical hardware.

The first experiments should use privileged state to determine whether the dynamics-identification problem and implementation work under clean conditions. Later experiments should fit the same models using operational telemetry while retaining ground truth only for evaluation. This separates model limitations from errors caused by estimation, filtering, latency, incomplete observability, or log synchronization.

Telemetry ingestion is part of the research problem rather than incidental plumbing. The pipeline must establish consistent coordinate frames, units, timestamps, sampling rates, quaternion conventions, actuator ordering, and definitions of each logged signal. Small alignment errors can easily be misidentified as physical dynamics.

## Models to Compare

The project should begin with a small set of models that expose meaningful tradeoffs.

### 1. Nominal physics baseline

Use a conventional rigid-body multirotor model with nominal parameters. This establishes how well an unfitted model predicts the recorded vehicle.

### 2. Identified physics model

Fit interpretable parameters such as mass, inertia, thrust and torque coefficients, actuator scaling, motor time constants, command delay, and simple aerodynamic terms. Known rigid-body structure remains fixed while the unknown parameters are learned from telemetry.

### 3. Physics model with learned residual

Add a compact differentiable residual to the identified physics model. This tests whether unmodeled behavior can be captured without asking a black-box model to relearn all known mechanics.

### 4. Black-box differentiable model

Fit a general differentiable transition or derivative model as a comparison. Its purpose is to measure the benefit and cost of physical structure in data efficiency, rollout stability, extrapolation, and computational performance.

Model complexity should increase only when simpler models leave consistent, measurable errors. A highly flexible model can improve training loss while concealing timestamp problems, state-estimation artifacts, or non-identifiable parameters.

## Minimal Experimental Plan

### 1. Build the telemetry pipeline

Parse ULogs into a canonical dataset containing timestamps, vehicle state, actuator commands, available actuator feedback, and relevant flight metadata. Align signals onto a common time base while retaining enough information to audit how each value was produced.

### 2. Record a simulated dataset

Collect multiple complete SITL flights containing varied but bounded vehicle motion and actuator usage. The initial project does not need to optimize these flights for information content; it only needs enough variation to test whether the logged dynamics can be recovered.

### 3. Fit the initial models

Fit the nominal-parameter, identified-physics, residual, and black-box models using the same training flights. Begin with simulator ground truth and logged actuator signals.

Whenever possible, train through multi-step rollouts rather than relying exclusively on numerical derivative targets. Differentiating noisy or filtered state measurements can create labels dominated by measurement and timing error.

### 4. Evaluate on held-out flights

Separate training, validation, and test data by complete flights or maneuvers. Randomly mixing nearby samples would overstate generalization because adjacent telemetry samples are highly correlated.

Evaluate both short-horizon transitions and long rollouts. Feed each model a held-out initial state and the recorded sequence of actuator commands, then compare its predicted trajectory with the recorded trajectory as the rollout horizon grows.

### 5. Introduce telemetry realism

Repeat fitting with operational state estimates instead of simulator ground truth. Independently vary or measure the effects of sampling rate, noise, filtering, missing samples, estimator resets, command delay, and timestamp alignment.

### 6. Test physical-log compatibility

Once the simulation results are understood, ingest existing physical PX4 logs without changing the canonical data and model interfaces. Physical testing is an extension of the identification benchmark; it does not require autonomous experiment generation or a downstream control demonstration.

## Evaluation

One-step prediction error is necessary but insufficient. The project should measure:

- Position, velocity, attitude, and angular-rate error over increasing rollout horizons
- Error on complete held-out flights and maneuvers
- Generalization across speeds, attitudes, actuator ranges, and vehicle configurations
- Rollout stability and violation of known physical constraints
- Sensitivity to telemetry noise, sampling rate, filtering, delay, and time alignment
- Data quantity required to reach a specified predictive accuracy
- Parameter recovery when the simulator's true parameters are known
- Consistency of learned parameters across different subsets of flights
- Training time, rollout throughput, and memory use
- Correctness and numerical stability of derivatives with respect to state and input

Logged-input rollout and closed-loop simulation answer different questions. Logged-input rollout isolates how well a model responds to a fixed actuator sequence. Closed-loop simulation also measures how model errors interact with a controller. The initial benchmark should prioritize logged-input rollout, while treating closed-loop evaluation as a useful secondary check.

## Research Questions

The initial project should answer:

1. Which vehicle dynamics can be identified reliably from PX4 telemetry alone?
2. How much does physical structure improve data efficiency and rollout stability?
3. When does a learned residual outperform parameter identification without hiding data problems?
4. How does a black-box model compare with structured alternatives on unseen flights?
5. How quickly does prediction error grow with rollout horizon?
6. Which telemetry signals and sampling rates are necessary for useful identification?
7. Can actuator delay and other latent dynamics be separated from timestamp or filtering errors?
8. How much performance is lost when fitting estimated-state telemetry instead of simulator ground truth?
9. Are the learned derivatives stable and meaningful outside the exact training samples?
10. How much faster is the identified model than the simulator that generated its data?

## Key Risks and Failure Modes

- Closed-loop telemetry may not contain enough independent variation to identify all model parameters.
- Multiple parameter combinations may explain the same observed motion.
- Timestamp errors may be mistaken for actuator or vehicle dynamics.
- Unmeasured motor state may make command-to-force dynamics partially unobservable.
- A model may fit one-step transitions while drifting badly during rollouts.
- A flexible residual may learn state-estimation errors rather than vehicle physics.
- A model may interpolate familiar flights well but fail under unfamiliar states or inputs.
- Quaternion conventions, coordinate-frame errors, or actuator ordering mistakes may produce plausible but incorrect results.
- Simulator success may depend on privileged signals unavailable on physical hardware.
- Differentiability alone does not guarantee that the resulting gradients are accurate, stable, or useful.

These failure modes are central evaluation targets. A useful result should reveal where telemetry-based identification works, where it fails, and how confidently those cases can be distinguished.

## Intended Outcomes

A successful initial project would produce:

- A reproducible pipeline from PX4 ULogs to aligned dynamics datasets
- A collection of simulated telemetry with complete flight-level splits and metadata
- Several differentiable vehicle models fitted from the same telemetry
- Quantitative comparisons of structured, residual, and black-box approaches
- Multi-horizon benchmarks on entirely held-out flights
- Measurements of the impact of realistic estimation and logging artifacts
- A fast model interface suitable for later simulation, optimization, or control projects
- A documented path from simulated telemetry to physical PX4 logs

The result remains valuable if some dynamics cannot be recovered reliably. A careful negative result can establish which signals or operating regimes are insufficient, which parameters are confounded, and which model metrics best predict long-horizon accuracy.

## Out of Scope for the Initial Project

The following may build on Glassbox but are not part of its initial success criteria:

- Active or autonomous experiment design
- Information-maximizing flight trajectories
- Trajectory optimization and model-predictive control
- Terminal-pose docking or contact modeling
- End-to-end autonomous hardware demonstrations
- Generalization to every vehicle class or telemetry format

Keeping these applications separate allows the project to establish whether the learned dynamics model itself is credible before its behavior is combined with a planner, controller, or specialized task.

## Implementation Freedom

This overview does not prescribe a final architecture, API, optimizer, or model parameterization. PX4 SITL, ULog, and JAX are strong starting choices, but they should be treated as practical tools rather than permanent constraints.

Implementation should favor the smallest reproducible experiment that can distinguish model classes and expose failure modes. Interfaces should remain simple enough to replace a parser, simulator, or model without prematurely building a general vehicle-learning framework.

## Related Work and Starting References

- [Crazyflow: An Accurate, GPU-Accelerated, Differentiable Drone Simulator in JAX](https://arxiv.org/abs/2606.01478)
- [Drone Models: Physics-Based and Data-Driven Quadrotor Dynamics](https://github.com/learnsyslab/drone-models)
- [Data-Driven System Identification of Quadrotors Subject to Motor Delays](https://arxiv.org/abs/2404.07837)
- [PX4 Simulation Documentation](https://docs.px4.io/main/en/simulation/)
- [PX4 ULog File Format](https://docs.px4.io/main/en/dev_log/ulog_file_format)
- [PX4 pyulog](https://github.com/PX4/pyulog)
