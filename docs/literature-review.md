# Literature review: the next Glassbox model architecture

Date: 2026-08-29

## Decision

Do not freeze Glassbox yet, but stop expanding the airframe catalog and stop
tuning airframe-specific constants. The literature supports one bounded research
cycle around a missing system-identification layer:

1. preserve and model observations rather than treating estimator outputs as the
   physical state;
2. estimate signal delay and latent state from causal input/output history;
3. learn decoupled force and moment discrepancies inside the rigid-body model;
4. validate residual structure and finite-horizon predictions, with uncertainty.

If this architecture does not improve the existing Nano, ARP, and fixed-wing
references without per-airframe tuning, the current implementation should be
accepted as the useful limit of the project for now.

## Why the current loop has saturated

Glassbox already contains most of the commonly recommended deterministic model
ingredients: a structured rigid-body core, differentiable integration, latent
actuator response, bounded residual acceleration, normalized multi-step losses,
and complete-flight holdouts. Adding another airframe now tests adapters and data
coverage more than it tests a new modeling hypothesis.

The remaining failures have a common shape:

- On the Nano benchmark, the best experimental candidate is essentially tied
  with the published Physics + Residual baseline in aggregate, but it does not
  dominate the rotational metrics.
- On the ARP PX4 logs, the fitted model can improve over another learned model
  while still losing to kinematic persistence on the protected flight.
- Fixed-wing short- and medium-horizon behavior is useful, while long open-loop
  simulations remain sensitive to initial condition, wind, and actuator errors.

Those results are consistent with a model that has a reasonable vector field but
is being trained against the wrong abstraction of the measurements. A PX4 EKF
state, optical-flow velocity, motion-capture pose, gyro sample, normalized motor
command, measured RPM, and actual aerodynamic force are not interchangeable
samples of one fully observed Markov state.

## Strongest relevant evidence

| Work | Validation | Main result | Glassbox implication |
| --- | --- | --- | --- |
| [Nano-drone system-identification benchmark](https://arxiv.org/html/2512.14450) | Real Crazyflie flights; held-out trajectory; rolling 0.5 s evaluation | Careful clock and motor/acceleration alignment is part of the benchmark. Translation is modeled well, but the quadratic motor model misses slow rotational dynamics; the authors recommend temporal learning or richer actuator physics. | Keep its evaluation protocol. Preserve its IMU acceleration instead of discarding it. Treat rotation as an observation/actuation problem, not another scalar-authority search. |
| [Data-Driven System Identification of Quadrotors Subject to Motor Delays](https://arxiv.org/html/2404.07837) | Real 27 g and 3.35 kg quadrotors; about one minute of proprioceptive data; outdoor deployment | Uses accelerometer and gyro measurements directly, infers unobserved motor speed with a first-order model, and estimates motor delay with a MAP objective before solving structured parameter problems. | Add direct sensor likelihoods and latent actuator inference. This is more defensible than asking position/attitude rollout loss to identify thrust, inertia, delay, and estimator behavior simultaneously. |
| [NeuroBEM](https://arxiv.org/abs/2106.08015) | Real aggressive flight up to 18 m/s; held-out trajectories | Models residual forces and moments, and uses 50 ms of velocity, rate, and motor-speed history because airflow is a hidden state. The hybrid model reduces prediction error by about 50% and generalizes better than pure learned or simpler physical models. | A residual depending only on instantaneous state and applied control is under-specified in aggressive flight. History should initialize or drive a small latent discrepancy state. |
| [Physics-Inspired Temporal Learning](https://arxiv.org/html/2206.03305) | Real quadrotor flights and MPC | A causal temporal convolution over state/control history is materially less noise-sensitive than an instantaneous model; useful accuracy appears with short histories and is used successfully in receding-horizon control. | A single opinionated causal encoder is justified; a menu of RNN/TCN knobs is not. |
| [Learning Long-Horizon Predictions for Quadrotor Dynamics](https://arxiv.org/html/2407.12964) | Two real-flight datasets, unseen trajectories, up to 60-step predictions | History, multi-step loss, and separate velocity/attitude predictors outperform a larger monolithic predictor. The paper reports 21–31x velocity and 23–56x attitude error reductions for its TCN relative to an MLP without history. | Glassbox already has multi-step loss. The missing pieces are causal history and stronger decoupling, not more MLP capacity. |
| [Deep Subspace Encoders for Nonlinear System Identification](https://arxiv.org/html/2210.14816) | Nonlinear system-identification benchmarks; accepted in *Automatica* | Learns the initial latent state of each truncated rollout from past inputs and outputs. The method is designed for noisy observations, stable optimization, and overlapping windows, and can include an innovation noise model. | Replace “the observed 13-state vector is the exact initial state” with a learned reconstructability map that estimates latent actuator/disturbance state and, where justified, a small denoising correction. |
| [Deep learning of vehicle dynamics](https://research.tue.nl/en/publications/deep-learning-of-vehicle-dynamics/) | Crazyflie simulation and real ground-vehicle data | Applies the subspace-encoder state-space approach to vehicle dynamics and explicitly motivates it for latent state and measurement noise. | SUBNET-style initialization is not merely a generic benchmark trick; it has already been applied to nano-quadrotor dynamics, although the quadrotor result is simulation-only. |
| [Advances in Aircraft System Identification at NASA Langley](https://ntrs.nasa.gov/api/citations/20230001945/downloads/MorelliGrauerAdvancesAircraftSIDJOA_2023.pdf) | Decades of real aircraft flight-test programs | Separates data compatibility, equation-error initialization, output-error fitting, uncertainty, and residual diagnostics. Output error is preferred for matching measured responses, while force/moment equation error gives efficient structure discovery. | For fixed wings, sensor calibration, wind/airdata, actuator models, force/moment coefficients, and parameter uncertainty are first-class concerns. A raw state-rollout optimizer is not the whole identification pipeline. |
| [Quadrotor gray-box identification from high-speed flight](https://research.tudelft.nl/en/publications/quadrotor-gray-box-model-identification-from-high-speed-flight-da/) | Real wind-tunnel free flight | Physics-guided stepwise force/moment models reduce moment residuals by 80% and force residuals by 20%. | Force/moment residuals are a better discovery surface than adding arbitrary terms to the integrated state transition. |
| [Prediction intervals for data-driven quadrotor models](https://arxiv.org/html/2408.06036) | Simulation plus real high-speed quadrotor flights | Prediction intervals widen under extrapolation and expose when a learned aerodynamic model is outside its support. | After the observation-aware model works, calibrated interval coverage should become a promotion metric; raw mean error alone cannot tell users when to trust a model. |
| [Neural-Fly](https://arxiv.org/abs/2205.06908) | Real wind-tunnel and outdoor flight; transfer across wind and drones | Learns a shared residual basis offline and adapts only low-dimensional coefficients online. | This is a credible later path for fleet adaptation, but only after Glassbox has multiple cleanly identified systems and a trustworthy observation model. |

## What is actually missing

### 1. A typed observation model

`Trajectory` currently preserves a canonical 13-element state, controls, and
exogenous values. The source data often contain additional measurements that are
more directly related to dynamics, but they are discarded. The Nano CSV files,
for example, contain body specific acceleration; their canonical NPZ files do
not. The PX4 adapter similarly uses estimator position, velocity, attitude, and
angular rate but does not retain raw IMU observations in the trajectory.

The internal representation should distinguish:

- physical/model state;
- measured outputs and their sensor semantics;
- commanded controls;
- measured actuator state, when available;
- exogenous context;
- observation timing, filtering, and provenance.

This does not require generic arbitrary-dimensional state support. The rigid-body
state can remain fixed while typed observation channels are optional.

### 2. Automatic timing and compatibility diagnostics

Before fitting, Glassbox should estimate or verify relative delays between
actuation and inertial response, detect gaps and irregular sampling, and report
the evidence. It should not silently turn delay, estimator filtering, and clock
offset into aerodynamic parameters.

Clock/transport misalignment and physical actuator lag must remain distinct:
alignment corrects timestamp semantics, while the dynamics model estimates the
remaining causal motor or servo response.

The user-facing policy can remain opinionated:

- preserve source timestamps;
- perform bounded automatic alignment only when semantically compatible signals
  exist;
- reject ambiguous mappings;
- record every inferred shift and filtering operation in provenance;
- expose a diagnostic failure, not a delay-tuning knob.

### 3. A causal latent-state encoder

Use a small causal temporal convolution to map recent typed observations and
controls to:

- initial applied-actuator state when it is unmeasured;
- a low-dimensional force/moment discrepancy state;
- optionally, a bounded correction to noisy observed velocity and angular rate.

The rigid-body state remains interpretable and follows the existing integrator.
The encoder should not replace the dynamics with a black-box sequence model.
Separate force and moment heads preserve modularity and match the strongest
quadrotor ablations.

### 4. Identification diagnostics beyond rollout RMSE

Finite-horizon rollout error remains necessary, but it should be joined by:

- one-step sensor negative log likelihood or normalized residual error;
- residual whiteness and residual/input cross-correlation;
- parameter covariance or bootstrap stability;
- prediction-interval coverage and width;
- performance stratified by flight regime and distance from training support.

A 60-second free rollout of an open-loop-unstable vehicle should remain a stress
test, not the primary definition of model validity. The public Nano benchmark
uses rolling 0.5-second predictions, and the long-horizon literature evaluates
roughly comparable finite windows. The important current ARP failure is losing
to persistence at useful finite horizons, not eventual full-flight divergence.

## Bounded research program

### Phase A: observation-first spike

Build a research-only path, without changing the public fitting interface, that:

1. reads Nano body specific acceleration and ARP/PX4 IMU data;
2. estimates bounded actuator-to-IMU timing alignment;
3. fits thrust, motor lag, and rotational response using direct sensor residuals;
4. compares the resulting parameters and rollouts with the current end-to-end
   rollout fit.

This is the cheapest test of the strongest literature-supported hypothesis. Do
not migrate the canonical artifact until this experiment shows value.

### Phase B: structured history encoder

If Phase A improves parameter stability or held-out error, add one small causal
encoder that initializes a latent discrepancy state from past observations and
controls. Use independent linear/body-force and angular/body-moment heads. Keep
history duration and architecture as maintained internal policy selected by
cross-flight validation, not public CLI options.

### Phase C: promote or freeze

Run the same maintained configuration on the existing corpus only. No new
airframes are needed for the decision.

Promotion requires all of the following:

- Nano: a material improvement over the published Physics + Residual reference,
  not another aggregate tie, with no major state-group regression;
- ARP: beat kinematic persistence on the protected finite-horizon evaluation;
- fixed wing: improve held-out short/medium predictions or diagnostic residuals
  without reducing long-horizon stability;
- synthetic recovery: retain parameter recovery and numerical stability;
- interface: no airframe-specific public tuning knobs.

A reasonable materiality threshold is at least 10% improvement in the aggregate
held-out metric, accompanied by consistent per-flight direction and bootstrap
intervals that exclude a negligible change. If the method fails these gates,
freeze model development and present Glassbox as a well-audited baseline and
telemetry normalization/evaluation framework rather than a state-of-the-art
universal dynamics learner.

## Ideas to defer

- More airframes before resolving observation semantics.
- Larger instantaneous residual networks or more hidden-unit searches.
- Symbolic discovery/SINDy as the primary model. It may become useful for
  interpreting already-clean force/moment residuals, but derivative noise and
  latent actuation make it a poor first move.
- Generic PINNs. Glassbox already encodes the relevant rigid-body physics; the
  missing issue is stochastic observation and hidden temporal state.
- Diffusion dynamics, broad architecture search, or end-to-end world models.
- Zero-shot/meta-learning across airframes. Shared-basis adaptation is promising,
  but the project does not yet have enough consistently observed platforms to
  identify what should be shared.

## Bottom line

The project is not missing a clever integrator or one more aerodynamic
coefficient. It is missing the distinction between **dynamics**, **latent state**,
and **how telemetry observes those dynamics**. That distinction is established
system-identification practice, repeatedly validated on real aerial vehicles,
and broad enough to benefit multirotors, conventional fixed wings, flying wings,
and flap-equipped aircraft without adding user-facing model knobs.
