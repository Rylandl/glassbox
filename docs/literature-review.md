# Literature review: the next Glassbox model architecture

Date: 2026-08-29

## Decision

Freeze the Glassbox model architecture at its current audited baseline. The
bounded research cycle proposed below is now complete: typed direct observations,
an observation-first initializer, innovation diagnostics, static compatibility
correction, causal first-order filtering, and explicit timestamp alignment were
all tested. The original transfer gate incorrectly coupled unrelated state
channels: it would reject a transferable body-rate hypothesis because velocity
did not improve by the same amount. The corrected channel-level gate advanced
the strongest body-rate candidate to a fixed rollout A/B. That candidate was
safe but did not clear the predeclared rollout materiality threshold on any
corpus.

Retain the canonical observation contract and research diagnostics because they
improve telemetry auditing. Do not add a combined delay/filter search, history
encoder, or more force/moment capacity on the present evidence. Further model
development should require materially new measurements or a new externally
validated method, not another internal architecture or coefficient search.

The evidence ladder is now explicit:

1. Development data estimates a bounded candidate.
2. Reused complete flights are research-validation data. A material,
   identifiable channel improvement can authorize an A/B for that channel only;
   unrelated channels remain on their reference behavior and are not silently
   modified.
3. Rollout A/B requires at least 10% geometric improvement across the maintained
   horizons and no aggregate or complete-flight/horizon regression above 5%.
4. Production promotion additionally requires a genuinely fresh lockbox. None
   of the repeatedly consulted flights in this report qualifies as one.

## Why the current loop has saturated

Glassbox already contains most of the commonly recommended deterministic model
ingredients: a structured rigid-body core, differentiable integration, latent
actuator response, bounded residual acceleration, normalized multi-step losses,
and complete-flight evaluation. Adding another airframe now tests adapters and
data coverage more than it tests a new modeling hypothesis.

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

## What the review identified as missing

### 1. A typed observation model

At the start of the review, `Trajectory` preserved a canonical 13-element state,
controls, and exogenous values while discarding more direct dynamics
measurements. That gap led to canonical format v3: Nano body specific
acceleration and PX4 IMU observations are now retained as typed observation
channels without changing the rigid-body state schema.

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

#### Phase A result (2026-08-29)

The observation contract and direct-fit spike were implemented and evaluated
before promotion. Nano specific force fits cleanly on its chronological sensor
validation split: 0.193 m/s² RMSE versus 4.961 m/s² for a constant baseline.
ARP also contains useful force information (1.495 versus 4.919 m/s²), but its free motor
time-constant search ran to the 0.25 s bound. Glassbox now reports that as
non-identifiability and falls back to the independently measured 60--80 ms
command/accelerometer correlation delay rather than presenting the boundary as
a recovered physical parameter.

The good local residual fits did **not** translate into better rollouts. After
fixing multi-horizon normalization so the A/B objective was identical, the
Nano initializer increased the equal-metric geometric mean of cumulative
benchmark error by 8.7% relative to the maintained instantaneous reference and
13.1% relative to the previous best authority candidate. On then-protected ARP
log 66 it increased the four-metric geometric error by 5.3%, 14.4%, and 11.5% at
0.1, 0.5, and 1.0 seconds relative to the maintained instantaneous reference.

The promotion decision is therefore negative. Typed observations remain in
canonical format v3 because preserving source measurements and their semantics
is independently useful and does not change rollout inputs. The direct fit is
retained as an automatic diagnostic and research API, but its parameters are
not used to initialize production fitting. This result also blocks Phase B:
there is not yet evidence that a learned history encoder has earned the added
complexity.
The machine-readable comparison is retained in
[`observation-spike-results.json`](observation-spike-results.json).

#### Post-freeze innovation diagnostic (2026-08-29)

Glassbox now performs the next lower-risk step from the classical system-
identification workflow: it checks whether research-validation one-step
innovations are
white and independent of current or past controls before attributing rollout
error to model capacity. NASA's
[aircraft parameter-uncertainty work](https://ntrs.nasa.gov/citations/20160007740)
shows why this matters: colored residuals are routine in flight identification
and invalidate uncertainty calculations that assume white errors. Each interval
resets the rigid-body state to the measurement while carrying the latent
actuator state causally. The report uses one maintained 0.5 s correlation
horizon and a conservative simultaneous screen; it adds no fitting controls.

All evaluated research-validation flights--three Nano Melon flights, ARP log
66, four X8 validation flights, and eight segments from protected IDF session
13--contained
temporally colored and input-correlated innovations. That alone is not evidence
for a history model because every real corpus is closed loop: controller
feedback can correlate commands with estimator and process errors.

A model-independent data-compatibility check made the result more specific.
Nano, ARP, and IDF pose increments disagree systematically with their reported
velocities, with mean trapezoidal compatibility residuals of 0.198, 0.141, and
0.156 m/s. X8 position is constructed consistently with velocity to numerical
precision, but all four corpora have colored attitude/body-rate incompatibility;
their mean rotation-rate residuals are 0.167, 0.136, 0.256, and 0.046 rad/s for
Nano, ARP, X8, and IDF respectively.

This supports preserving an explicit observation boundary, but it does not
reverse the Phase A promotion failure or authorize Phase B. The current models
are being scored against state channels that do not describe one exactly sampled
rigid-body trajectory. A future observation-model experiment must first show
that it reduces these compatibility defects without using protected rollout
targets, then pass the unchanged cross-platform rollout gates. The complete
machine-readable result is in
[`innovation-diagnostic-results.json`](innovation-diagnostic-results.json).

#### Static observation-correction result (2026-08-29)

The first correction experiment followed NASA's interpretable
[real-time data-compatibility model](https://ntrs.nasa.gov/api/citations/20150000551/downloads/20150000551.pdf):
each reported world-velocity and body-rate axis received one bounded scale factor
and constant bias. Coefficients were estimated on development trajectories only,
then frozen for complete research-validation flights. Synthetic tests recover
known scale and bias errors to within 0.001, confirming that the implementation can detect
the error class it claims to model.

The real-data transfer gate failed:

| Corpus | Position/velocity ratio | Attitude/rate ratio |
| --- | ---: | ---: |
| Nano | 1.051 | 1.001 |
| ARP | 0.952 | 1.008 |
| X8 | 0.998 | 1.000 |
| IDF | 0.990 | 0.992 |

Values are corrected/original compatibility RMSE, so lower is better. No corpus
improved both material groups by the required 10%. Nano also crossed the 5%
regression guardrail. X8's position/velocity group was already consistent to the
maintained numerical floor and was excluded from its material gate.

No rollout refit was run. The experiment shows that the observed defects are not
well explained by transferable static calibration errors; estimator filtering,
time variation, and channel-specific temporal semantics remain more plausible.
The bounded implementation remains isolated as a research utility with explicit
provenance, but it is not imported by the normal Glassbox interface or applied
by fitting. The evidence is recorded in
[`state-observation-correction-results.json`](state-observation-correction-results.json).

#### Temporal observation-filter result (2026-08-29)

The next bounded experiment tested a causal first-order observation response,
motivated by the explicit treatment of filtering, time delay, and output error in
NASA flight-identification workflows. Each pose-implied velocity and angular-rate
axis was passed through one independently selected time constant, followed by the
same bounded scale and bias available to an instantaneous reference. Zero memory
was a candidate, the largest permitted time constant was 0.5 s, and the policy
introduced no public configuration. A 0.5 s/10%-of-flight warm-up cap prevented
filter initialization from determining the score.

Synthetic recovery selected 0.081 s on every axis for an injected 0.080 s
response and reduced held-out error to about 1.9% of the instantaneous model. A
deliberately out-of-range 2 s response selected the 0.5 s boundary and was
rejected, establishing both positive and negative test sensitivity.

Frozen real-data transfer produced this result:

| Corpus | Position/velocity ratio | Attitude/rate ratio | Body-rate channel gate |
| --- | ---: | ---: | --- |
| Nano | 0.943 | 1.000 | fail |
| ARP | 0.983 | 0.632 | pass |
| X8 | 1.000 (already consistent) | 0.599 | pass |
| IDF | 0.933 | 0.840 | pass |

Values are candidate/instantaneous-reference RMSE. ARP, X8, and IDF provide
strong evidence that their reported body rates have useful temporal semantics;
Nano selected zero angular-rate memory. The position channel did not meet the
10% material-improvement requirement on Nano, ARP, or IDF, and one Nano
research-validation flight regressed by 14.3%, beyond the 5% per-flight
guardrail.

The body-rate channel independently passed on ARP, X8, and IDF. That is sufficient
cross-platform evidence for a body-rate-only rollout A/B; requiring the unrelated
velocity channel to improve would test universality rather than transfer. Nano's
zero-memory selection is a valid platform-specific no-op, not evidence for the
candidate. Complete coefficients and split sizes are recorded in
[`temporal-observation-filter-results.json`](temporal-observation-filter-results.json).

The fixed rollout A/B changed only the reported body-rate output of existing
models. Dynamics parameters, physical trajectories, and position, velocity, and
attitude metrics were identical between candidate and reference:

| Corpus | 0.1 s ratio | 0.5 s ratio | 1.0 s ratio | Across-horizon geometric ratio | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| ARP | 0.886 | 0.993 | 1.002 | 0.959 | fail |
| X8 | 0.869 | 0.953 | 0.959 | 0.926 | fail |
| IDF | 0.912 | 0.990 | 0.994 | 0.965 | fail |

The filter consistently helps at 100 ms and does not cause a guarded regression,
but its advantage is almost gone by 0.5–1.0 s. None achieves the predeclared 0.90
geometric ratio across horizons. The post-result threshold was not relaxed, so
the observation layer remains a research utility and is not applied by fitting.
The full A/B is recorded in
[`body-rate-observation-rollout-results.json`](body-rate-observation-rollout-results.json).

#### State-channel alignment result and terminal decision (2026-08-29)

The final distinct compatibility hypothesis separated a pure relative timestamp
offset from first-order filtering. It fitted one shared signed delay for all
three velocity axes and one for all three angular-rate axes, over a maintained
±100 ms range. The candidate retained the same bounded affine capacity as its
zero-shift reference, used equal source-group fitting weight, trimmed a fixed
100 ms boundary for every candidate, and rejected protected splits.

Synthetic tests recovered injected +60 ms and -40 ms shifts within 10 ms and
rejected a +200 ms shift at the +100 ms boundary. Frozen real-data transfer was:

| Corpus | Position/velocity ratio | Attitude/rate ratio | Body-rate channel gate |
| --- | ---: | ---: | --- |
| Nano | 1.003 | 0.906 | fail |
| ARP | 0.977 | 0.906 | fail |
| X8 | 1.000 (already consistent) | 0.620 | pass |
| IDF | 0.954 | 0.827 | pass |

Timing alignment again explains a meaningful part of body-rate incompatibility,
especially on the fixed wings. It clears the conditional channel gate on X8 and
IDF, but not ARP; the temporal candidate was therefore the stronger transfer
hypothesis and the alignment candidate was not advanced. Nano also exceeded the
5% per-flight position regression guard.

This closes the bounded literature-guided research cycle. Combining delay and
filter candidates would increase flexibility after the stronger candidate failed
the rollout gate; a history encoder was explicitly conditional on rollout-level
success. The current dynamics architecture is therefore frozen. Glassbox remains
valuable as an opinionated telemetry normalization, differentiable gray-box
baseline, and evaluation framework, but the evidence does not support
presenting it as a state-of-the-art universal dynamics model. The full result
is recorded in
[`state-observation-alignment-results.json`](state-observation-alignment-results.json).

### Phase B: causal residual-innovation observer — tested and rejected

A distinct dynamics-history upper-bound showed that past one-step prediction
innovations sometimes contained useful short-horizon state even though the
observation-only branch failed. A bounded candidate was therefore implemented
temporarily: measured state/control history initialized six body-acceleration
discrepancy states, with one force and one moment decay time constant. The
instantaneous model remained an exact nested no-op, candidate values were fixed
internal policy, and no airframe-specific feature or public knob was added.

The decisive rerun used the strongest maintained model on every corpus, not the
older model on which the upper-bound was first observed:

| Corpus | Selected history | Development ratio | Held-out result | Capability gate |
| --- | --- | ---: | --- | --- |
| Nano | no-op | 1.000 | official published-reference ratio remains 0.996 | no material gain |
| ARP | force 0.20 s | 0.972 | 0.946 vs instantaneous; 1.269 vs persistence | fail |
| X8 | no-op | 1.000 | 1.000 vs instantaneous | no material gain |
| IDF | no-op | 1.000 | combined candidate was 0.920 but exceeded the guardrail | fail |

The IDF combined candidate had an attractive 0.907 development aggregate, but
its maximum per-flight metric ratio was 1.114 against the allowed 1.05. Force-
only and moment-only fallbacks were less safe. Retaining the guardrail matters
more than rescuing the aggregate. ARP was the only accepted per-airframe gain,
and it did not close the learned-model gap to kinematic persistence. The runtime
and fitter implementation was removed; the complete decision record is in
[`residual-innovation-observer-results.json`](residual-innovation-observer-results.json).

### Phase C: promote or freeze — frozen

The maintained criteria below were applied to the existing corpus without adding
new airframes:

Promotion requires all of the following:

- Nano: a material improvement over the published Physics + Residual reference,
  not another aggregate tie, with no major state-group regression;
- ARP: beat kinematic persistence on the protected finite-horizon evaluation;
- fixed wing: improve held-out short/medium predictions or diagnostic residuals
  without reducing long-horizon stability;
- synthetic recovery: retain parameter recovery and numerical stability;
- interface: no airframe-specific public tuning knobs.

The materiality threshold was at least 10% improvement in the aggregate
research-validation metric with consistent per-flight direction. The
observation-aware and dynamics-history candidates failed those gates, so model
development is frozen and Glassbox is presented as a well-audited baseline and
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

The project was not missing a clever integrator or one more aerodynamic
coefficient. Glassbox now preserves the distinction between **dynamics**,
**latent state**, and **how telemetry observes those dynamics**, but neither the
bounded observation-aware candidate nor the causal innovation observer cleared
the rollout promotion gate on current best models.
That is a useful technical result: the maintained system is an honest, general
gray-box baseline and telemetry framework, and further capacity is not justified
until new evidence changes the problem.
