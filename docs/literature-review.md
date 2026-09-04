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

> Note (2026-09-03): the code and recorded artifacts for the experiments
> below were removed from the repository once the program concluded. The
> last commit that carries them is `4c119a8`. This section keeps the
> decisive results as prose.

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
before promotion. Nano specific force fit cleanly on its sensor validation
split (0.193 m/s² RMSE versus a 4.961 m/s² constant baseline), and ARP also
carried useful force information (1.495 versus 4.919 m/s²), though its free
motor time-constant search was non-identifiable and fell back to the
independently measured 60-80 ms command/accelerometer delay. The good local
residual fits did not translate into better rollouts: the Nano initializer
increased the geometric-mean cumulative benchmark error by 8.7% relative to
the maintained instantaneous reference (13.1% relative to the previous best
candidate), and on ARP log 66 it increased four-metric geometric error by
5.3%, 14.4%, and 11.5% at 0.1, 0.5, and 1.0 seconds. The promotion decision
was therefore negative: typed observations stayed in canonical format v3 for
their independent telemetry value, the direct fit was retained only as a
research diagnostic and not used to initialize production fitting, and the
result blocked Phase B pending evidence that a learned history encoder would
earn its added complexity.

The diagnostic stage itself was removed from the library on 2026-09-03. It
initialized nothing on either of its call sites and its report keys were read
by nothing. The last commit that carries it is `2e16ebc`; the result above is
the record.

#### Post-freeze innovation diagnostic (2026-08-29)

Glassbox's maintained one-step innovation whiteness diagnostic found that
every evaluated research-validation flight (three Nano Melon flights, ARP
log 66, four X8 validation flights, and eight segments from protected IDF
session 13) contained temporally colored, input-correlated innovations. A
model-independent compatibility check made the result more specific: Nano,
ARP, and IDF pose increments disagreed with reported velocities by mean
residuals of 0.198, 0.141, and 0.156 m/s; X8 position was consistent with
velocity to numerical precision, but all four corpora showed colored
attitude/body-rate incompatibility, with mean rotation-rate residuals of
0.167, 0.136, 0.256, and 0.046 rad/s for Nano, ARP, X8, and IDF respectively.
This supported preserving an explicit observation boundary, but it did not
reverse the Phase A promotion failure or authorize Phase B.

#### Static observation-correction result (2026-08-29)

A static per-axis scale/bias observation-correction model, following NASA's
interpretable real-time data-compatibility approach, recovered known
synthetic scale and bias errors to within 0.001, confirming the
implementation could detect the error class it targeted. Its real-data
transfer gate then failed on every corpus: corrected/original compatibility
RMSE ratios were 1.051/1.001 (Nano), 0.952/1.008 (ARP), 0.998/1.000 (X8), and
0.990/0.992 (IDF) for the position/velocity and attitude/rate groups
respectively, with no corpus improving both groups by the required 10% and
Nano crossing the 5% regression guardrail. The observed defects were
therefore not well explained by transferable static calibration error, and
the bounded implementation was kept only as an isolated research utility,
not applied by fitting.

#### Temporal observation-filter result (2026-08-29)

A causal first-order observation filter, synthetically validated by
recovering an injected 0.080 s response as 0.081 s (and rejecting an
out-of-range 2 s response at the 0.5 s boundary), was transferred to frozen
real data. Attitude/rate compatibility improved on ARP (0.632), X8 (0.599),
and IDF (0.840) but not Nano (1.000); the position/velocity channel met the
10% bar on no corpus, and one Nano flight regressed by 14.3%, beyond the 5%
guardrail. The body-rate channel's independent pass on ARP, X8, and IDF was
sufficient cross-platform evidence for a body-rate-only rollout A/B, which
changed only reported body-rate output with dynamics, trajectories, and
other metrics held identical between candidate and reference. That A/B
reached an across-horizon geometric ratio of 0.959 (ARP), 0.926 (X8), and
0.965 (IDF), a consistent gain at 100 ms that was mostly gone by 0.5-1.0 s, and
none reached the predeclared 0.90 threshold, so the observation layer
remained a research utility and was not applied by fitting.

#### State-channel alignment result and terminal decision (2026-08-29)

A pure timestamp-alignment candidate, synthetically validated by recovering
injected +60 ms and -40 ms shifts within 10 ms (and rejecting a +200 ms shift
at the +100 ms boundary), again explained a meaningful part of body-rate
incompatibility on frozen real data: attitude/rate ratios were 0.906 (Nano,
gate fail), 0.906 (ARP, gate fail), 0.620 (X8, pass), and 0.827 (IDF, pass),
with Nano also exceeding the 5% position regression guard. Because it cleared
the channel gate on X8 and IDF but not ARP, the temporal filter remained the
stronger transfer hypothesis and the alignment candidate was not advanced.
This closed the bounded literature-guided research cycle and left the
dynamics architecture frozen.

### Phase B: causal residual-innovation observer — tested and rejected

A bounded causal residual-innovation observer (six body-acceleration
discrepancy states driven by measured state/control history, with one force
and one moment decay time constant, nested as an exact no-op on the
instantaneous model) was rerun on the strongest maintained model per corpus.
Nano and X8 selected the no-op (no material gain over a 0.996
published-reference ratio and a 1.000 instantaneous ratio respectively); ARP
selected a 0.20 s force time constant that reached 0.946 against the
instantaneous model on held-out data but only 1.269 against persistence,
failing the capability gate; IDF's combined candidate reached an attractive
0.907 development aggregate and 0.920 held-out ratio, but its maximum
per-flight metric ratio of 1.114 exceeded the 1.05 guardrail. ARP was the
only accepted per-airframe gain and it did not close the learned-model gap to
kinematic persistence, so the runtime and fitter implementation was removed.

### Phase C: promote or freeze — frozen

The maintained promotion criteria (a material Nano improvement over the
published Physics + Residual reference with no major state-group regression,
beating kinematic persistence on the protected ARP evaluation, improved
fixed-wing held-out prediction without reduced long-horizon stability,
retained synthetic recovery, and no airframe-specific public tuning knobs)
were applied to the existing corpus without adding new airframes, against a
materiality threshold of at least 10% aggregate research-validation
improvement with consistent per-flight direction. The observation-aware and
dynamics-history candidates failed those gates, so model development is
frozen and Glassbox is presented as a well-audited baseline and telemetry
normalization/evaluation framework rather than a state-of-the-art universal
dynamics learner.


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

The lagged multirotor rotational response is closed as a negative result.
The hypothesis was that control-generated torque follows the applied motor
state through a first-order lag of its own, so that slow rotor and aerodynamic
torque dynamics could be identified without delaying collective thrust. It was
tested on both real multirotor corpora and failed promotion on each. On the
Nano-drone benchmark a learned latent rotational response with a cross-axis
mixer improved the training-profile score but failed the one-shot Melon
promotion check, so neither was ever enabled by default. On the ARP PX4 logs it
improved the instantaneous diagonal reference by 4.93 percent on the
development folds (logs 63-65) and, on protected log 66, improved the fitted
reference's equal-horizon, equal-metric geometric score by 11.49 percent with
its worst individual metric changing by +0.07 percent, reduced the aggregate
rotational score by 17.91 percent, and delayed the first configured
complete-rollout divergence threshold from 1.06 to 2.00 seconds, while
remaining 34.08 percent worse than kinematic persistence overall and still
crossing that threshold; the verdict was `improves_reference_only`. No shipped
caller ever selected the branch, and it cost three latent dimensions on every
multirotor rollout and three frozen coordinates in every multirotor parameter
vector. The branch, its memoryless sentinel, and the fitting switches that
selected it were removed at this commit; the last commit carrying them is
`9d59e4a`. The multirotor torque map is now memoryless, and the bounded
cross-axis mixer it was measured with survives as part of that map.

Grouped predictive ensembles are closed as a negative result. The hypothesis
was that disagreement across a grouped bootstrap ensemble carries predictive
information beyond a constant residual radius, and it was tested through five
versions, from the first grouped bootstrap to a balanced, calibrated,
nested-group form. On the IDF-DS fixed-wing corpus, across 13 source-group
folds and 78 fitted members, coverage held but the two claims that mattered
failed: disagreement ranked held-out error at a median Spearman of 0.20
against the 0.30 the gate required, and the calibrated set score came out
1.89 percent worse than the constant-radius baseline it was meant to beat,
against a required 5 percent improvement. The workflow, its
`ensemble-benchmark` command, its concept page, and its four recorded notes
were removed at this commit; the last commit carrying them is `bd48419`.

## Bottom line

The project was not missing a clever integrator or one more aerodynamic
coefficient. Glassbox now preserves the distinction between **dynamics**,
**latent state**, and **how telemetry observes those dynamics**, but neither the
bounded observation-aware candidate nor the causal innovation observer cleared
the rollout promotion gate on current best models.
That is a useful technical result: the maintained system is an honest, general
gray-box baseline and telemetry framework, and further capacity is not justified
until new evidence changes the problem.

The versioned accuracy contracts and the cross-platform fitting-policy sweep
were retired at this commit with their verdicts standing: the sweep failed its
protected promotion check and the reference fitting defaults were retained,
and the fixed-wing residual was selected for continued development while
missing the cross-airframe development contract by 0.0023 m of p90 IDF
position error at the half-second horizon.

## What Phases 0 to 3 retired, 2026-09-04

The migration from the research repository to a production library took the
package from about 41,700 to about 28,200 source lines. Every removal below is
a research mechanism whose verdict was already recorded, and each names the
last commit that can still run it. The code and its artifacts are not kept; this list plus the
prose above is the record.

**The observation program.** Typed observation channels stayed, but every
mechanism built on top of them went: static observation correction, first-order
temporal filtering, state-channel timing alignment, body-rate observation
rollout scoring, and the seven recorded artifacts behind them, at `4c119a8`.
The observation-first initializer that survived as a fit stage followed at
`2e16ebc`; it initialized nothing, because its parameters were discarded at
both call sites and the forty report keys it emitted were read by nothing. The
Phase A, post-freeze and Phase B results above are the whole finding.

**The fleet prior and plan assessment.** `StructuredParameterPrior`,
`initialize_belief`, `DynamicsBelief.condition_parameter_prior`,
`with_parameter_members` and the `glassbox prior` command went at `478c063`.
At the only scale the prior ever ran, five members over twenty-two parameters,
99.6 percent of its normalized covariance trace was unit-ball assumption on
the unresolved nullspace, and nothing in the package conditioned on it in
production. `ParameterInformation.seeded_from_members` is the forty-line
replacement, and it invents no precision on a direction no member moved. The
same commit removed `RuntimeDynamicsBelief.assess_plan` and `PlanAssessment`,
which had no caller, and the error-radius quantiles, which no artifact carried.

**The transaction.** The transactional belief update, 2,006 lines, went at
`8d3400f`: `propose_dynamics_belief_update`,
`validate_and_commit_dynamics_belief_update`, the thirty-five-field report,
the two-sigma improvement margin, the maximum-norm trust bound, the line
search, the revision fingerprints and the replay detection. The reason was not
cost. On every path a shipped artifact could reach, the account of what is
unknown never changed: the fit wrote total-forecast-scoped evidence and every
contraction mechanism was gated on a conditional innovation scope nothing
produced, so the recorded artifact of the day said covariance not updated,
posterior trace equal to prior trace, information gain null. A commit then
zeroed the error moments, which removed the controller's horizon cap and
showed it zero model uncertainty, so adapting made it more confident than its
evidence supported. The margin and the trust bound existed to patch the
null-acceptance rate of an improvement-threshold gate, which is the mechanism
the design does not want. The runtime tangent bias went with it, because it
was the root cause of the staleness lifecycle. `absorb` is the recursion the
in-flight identifier already ran, generalized by one Jacobian, and the 64-seed
null-acceptance calibration is replaced by a pinned step-size property at the
same 64 seeds.

**The batch identifier and the cascade controller.** `BootstrapMultirotorIdentifier`
with its excitation planner and arrest commands, and
`ProgressiveBootstrapController` with `ThrustCascade` and its fixed
pseudo-random excitation scan, both went at `aab0b42`. The recursive
identifier's Gram accumulation is the same fit, so a batch fit is folding N
transitions and reading the belief; and the cascade controller was the
hand-gained baseline arm of a comparison that lives in the demo repository and
was retired there by the learned controller. The same commit removed the
identifier's own certification transaction, which was the same gate twice, and
the three switches measured worse on the release ensemble: staged regressors,
the collective sign rule and the prequential residual as implemented. The last
two switches, the aggregation window and the integrated collective, became the
only behaviour at `2f5adc2` after the measurement that selected them.

**The support filter.** `SupportFilterMode` with its six modes, the candidate
enumeration and batched candidate kernel, the actuator-reaction horizon, and
the bounded-authority post-pass went at `9570fa1`, together with the
diagnostic fields only they wrote. Neither backed a recorded claim. What the
belief knows about its own error is now charged inside the objective instead,
so it shapes the plan the optimizer converges to rather than editing the plan
afterwards.

**The gates and the selection machinery.** `policy_selection`,
`fixedwing_gate`, `acceptance` and `selection`, with the
`glassbox select-policy` and `glassbox fixedwing-gate` commands and the
fitting-policy and fixed-wing-gate pages, went at `bd48419`. They were
research promotion rules whose verdicts were already recorded and which
nothing in continuous integration ran: the cross-platform fitting-policy sweep
failed its protected promotion check and the reference fitting defaults were
retained, and the fixed-wing residual was selected for continued development
while missing the cross-airframe development contract by 0.0023 m of p90 IDF
position error at the half-second horizon. The two reusable divergence helpers
were promoted into evaluation first and then removed at `2faf563` when the
gates that called them were gone. The adaptation benchmark went at the same
commit as the gates: it asserted that every update applied while its own
report recorded the acceptance gate as failed.

**The predictive ensemble.** The workflow, its `ensemble-benchmark` command,
its concept page and its four recorded notes went at `bd48419`. The finding is
recorded above: across 13 IDF-DS source-group folds and 78 fitted members,
coverage held but disagreement ranked held-out error at a median Spearman of
0.20 against the 0.30 the gate required, and the calibrated set score came out
1.89 percent worse than the constant-radius baseline it was meant to beat.

**The authority sweep.** `workflows/angular_authority.py` and
`workflows/nanodrone_rotation.py` went at `bd48419`, and the two doc
paragraphs that rested on the sweep were withdrawn then, because its selection
numbers had no recorded artifact. The transform it selected with,
`core.dynamics.with_angular_dynamics_authority`, had no caller left and went at
`2e16ebc`; `with_constant_angular_rate` went at `c250233` once the published
Nano-drone protocol's own hold-state baseline made a constant-rate model
variant redundant. The command-offset candidate measured on ARP logs 63 to 65
goes with them as a development-only result: it improved the aggregate score
by 0.40 percent while its worst individual metric regressed 79.0 percent, and
the three held-out fits learned consistent offsets of -0.111, -0.105 and
-0.072 normalized command. It was never promoted, because log 66 was already
spent on the rotational-structure evaluation below.

**The lagged rotational response.** The branch, its sentinel machinery and the
fitting switches that selected it went at `9d59e4a`. The negative result is
recorded above under the deferred ideas; the multirotor torque map is now
memoryless, the latent state is four wide instead of seven, and the bounded
cross-axis mixer the branch was measured with survives as part of that map.

Two recorded files went with the same pass because they were not machine
output. `docs/results/multirotor-profile-results.json` was a hand-written lab
note in JSON clothing, with `date`, `hypothesis` and `decision` keys, produced
by no command and pinned by no test; it went at `c35224e` and its numbers are
now prose on [validation](validation.md#px4-sitl-corpora). The four
predictive-ensemble notes went with their workflow. Every artifact under
`docs/results/` is now machine output produced by one manifest entry.
