# Glassbox

Glassbox learns differentiable vehicle dynamics from recorded state and actuator telemetry.

The current implementation fits platform-specific multirotor or fixed-wing
structured dynamics through differentiable multi-step rollouts. It supports
synthetic parameter recovery, multirotor and fixed-wing PX4 ULog ingestion,
the real-flight Nano-Quadrotor and Skywalker X8 system-identification
benchmarks, an EPFL TOPOPlane2 conventional-airframe reference, multi-flight
training, complete-flight and maneuver-family
holdouts, multi-horizon evaluation, and applied-control response ablations.

## Run it

```bash
uv sync --dev
uv run glassbox-synthetic
uv run pytest
```

## Experimental predictive ensembles

Glassbox can now test whether corpus sensitivity contains useful predictive
uncertainty without describing the result as a Bayesian posterior. The offline
ensemble benchmark keeps a complete profile or source group outside every fit,
then builds each member by resampling only the remaining independent source
groups:

```bash
uv run glassbox-ensemble-benchmark artifacts/sitl/canonical/*.npz \
  --output-dir artifacts/sitl/predictive_ensemble
```

The outer axis is selected automatically: maneuver profile when the corpus has
multiple typed profiles, otherwise source group. Bootstrap draws are stratified
by profile when possible, and the normal interface automatically chooses four
to eight members rather than exposing resampling and calibration knobs.

Reports contain endpoint ensemble-center error, empirical 50/80/90% disagreement
coverage, interval radius, multivariate energy score, disagreement/error rank
correlation, and both endpoint and complete-path finiteness. Attitude is handled
as a shortest-path rotation vector rather than componentwise quaternion bounds.
The report also exposes the finite-member mass attained by each requested level,
unique resamples, unique fitted members, and unique predictions: four to eight
members are treated as a disagreement ensemble, not a resolved interval
distribution. Every member shares normalization and envelope statistics derived
once from the complete outer-training fold, while profile-balanced bootstrap
multiplicity affects only empirical loss.
Artifacts explicitly record `posterior: false`: this first stage measures
epistemic sensitivity to the available groups and does not claim to contain
process noise, observation noise, or model forms that were never fitted.

See [the predictive-ensemble guide](docs/predictive-ensembles.md) for the split
contract, metric interpretation, and promotion boundary. The workflow remains
diagnostic-only. Once the first outer-fold results influence development choices,
promotion requires a subsequently untouched corpus, airframe, or configuration.

## Nonlinear model-predictive control

Glassbox includes an opinionated JAX NMPC controller for actionable fitted
multirotor and fixed-wing artifacts. It tracks the canonical rigid-body state,
propagates learned actuator lag and residual dynamics, accepts physical tracking
tolerances and state limits, and returns a bounded command with explicit solver,
validity, and fallback diagnostics.

The maintained synthetic gate covers hover, translation, attitude, fixed-wing
trim, altitude, path, coordinated turn, optional flaps, flying-wing generalized
controls, and model mismatch. Run it with:

```bash
uv run glassbox-nmpc-benchmark --output artifacts/nmpc_report.json
```

See [the NMPC guide](docs/nmpc.md) for the programmatic API, measured results,
safety boundaries, supported command semantics, and the PX4 SITL integration
path. This is research control software, not a flight-safety system.

PX4 integration stays outside the normal dependency and test path. A pinned,
prebuilt PX4 SIH container can verify passive MAVLink-to-canonical-state
telemetry without a PX4 checkout, Gazebo, ROS, arming, mode changes, or command
transmission:

```bash
GLASSBOX_RUN_PX4_SITL=1 \
  uv run pytest -m px4_sitl tests/integration/test_px4_sitl.py -v
```

An eligible fitted artifact can then be exercised against continuously received
live state estimates with `glassbox-px4-nmpc-shadow`. The report separates PX4
source-clock progress from solver latency, applies the model period as the solve
deadline, and records proposed commands without ever transmitting them.

A separately gated SIH-only test performs normal takeoffs and a bounded
vertical, lateral, yaw, and combined flight matrix while the shadow controller
passively consumes both moving state and the commands PX4 actually applies. It
checks state/command timestamp alignment, maneuver-specific excitation,
short-horizon model error against a kinematic baseline, landing, and disarm;
see the NMPC guide for the explicit opt-in command and safety boundary.

## Nano-Quadrotor System Identification Benchmark

Prepare the complete
[IDSIA benchmark](https://github.com/idsia-robotics/nanodrone-sysid-benchmark)
with one command:

```bash
uv run glassbox-nanodrone prepare artifacts/nanodrone
```

This downloads all 15 official CSV recordings from the pinned upstream commit,
verifies their Git LFS SHA-256 values, and writes 12 canonical training
trajectories and three canonical Melon test trajectories under
`artifacts/nanodrone/canonical/`. The raw upstream files remain separate under
`artifacts/nanodrone/raw/`; Glassbox does not vendor the dataset.

Fit against the benchmark's trajectory-level split with:

```bash
uv run glassbox-fit \
  artifacts/nanodrone/canonical/train/*.npz \
  artifacts/nanodrone/canonical/test/*.npz \
  --holdout-profile melon \
  --training-horizons 0.1,0.5,1.0 \
  --evaluation-horizons 0.1,0.5,1.0 \
  --model-class structured_residual \
  --skip-no-lag-ablation \
  --model artifacts/nanodrone/model.json \
  --report artifacts/nanodrone/report.json
```

Score either saved model with the benchmark's rolling 1-to-50-step metrics:

```bash
uv run glassbox-nanodrone evaluate \
  artifacts/nanodrone/model.json \
  artifacts/nanodrone/canonical/test/*.npz \
  --report artifacts/nanodrone/benchmark_report.json
```

This initializes a 0.5-second open-loop prediction at every admissible sample
in each Melon run and reports per-horizon mean Euclidean position, velocity,
attitude-geodesic, and angular-velocity errors. It never lets a window cross a
flight boundary. The upstream released metric script concatenates the three
runs before shifting truth rows by horizon; those shifts cross two recording
boundaries and reproduce the paper's slightly higher naïve values exactly.
Glassbox deliberately reports the boundary-safe version.

The adapter preserves the benchmark's 100 Hz sampling, 13-state rigid-body
trajectory, and three published body-specific-force channels. The state is a
processed mixed-source observation, not uniform ground truth: position and
attitude include motion capture, while velocity and angular rate come from
onboard sensing/estimation. The accelerometer outputs are typed as
state-aligned FLU observations used only during identification. The upstream
world and body frames already have Glassbox's z-up/FLU signs; the adapter
reorders the scalar-last quaternion to wxyz.

The benchmark inputs are measured rotor angular velocities rather than motor
commands. The source order `m1,m2,m3,m4` (front-right, rear-right, rear-left,
front-left) is verified from the published allocation matrix and reordered to
Glassbox's front-left, front-right, rear-right, rear-left order. Each input is
stored as `(omega / 2500 rad/s)^2`, a dimensionless thrust proxy that retains
the published quadratic rotor relationship and keeps the optimizer well
scaled. The 2500 rad/s reference is only a coordinate scale and is recorded in
the typed spec; it is not an assumed motor limit. Consequently, a fitted
control time constant describes any residual alignment/filtering response in
measured RPM, not command-to-motor physical lag.

The v3 multirotor model can represent one bounded shared normalized-command
offset in its collective thrust map. NanoDrone's typed
`squared_rotor_speed_ratio` inputs are already physical thrust proxies, so this
parameter is held exactly at zero and cannot be enabled for that dataset.

For individual files, use `glassbox-nanodrone inspect` or
`glassbox-nanodrone extract`. Checksum verification is strict by default so an
upstream revision cannot silently change a benchmark run.

### Current structured baseline

The checked workflow was run for 400 Adam steps at 0.1 and 0.5-second training
horizons, with the 12 Square/Random/Chirp flights used for fitting and all three
Melon runs held out. Because the input is measured RPM, the fixed 0.1 ms
response model is the cleaner baseline; learning the response produced an
effectively equivalent 3.96 ms value.

Boundary-safe errors for the structured no-lag model are:

| Horizon | Position [m] | Velocity [m/s] | Attitude [rad] | Angular velocity [rad/s] |
| ---: | ---: | ---: | ---: | ---: |
| 0.01 s | 0.00126 | 0.01591 | 0.00112 | 0.09102 |
| 0.10 s | 0.01135 | 0.13463 | 0.02801 | 0.48293 |
| 0.50 s | 0.10837 | 0.36582 | 0.36871 | 1.17149 |

The cumulative 1-to-50-step errors are 2.2875 m, 11.2634 m/s, 7.4349 rad,
and 39.6054 rad/s, respectively. Relative to the boundary-safe persistence
baseline, this is 7.73x better on position, 3.44x on velocity, 1.10x on
attitude, and 0.73x on angular velocity. The last value is the important
failure: the compact structured angular-rate model is currently worse than
holding the measured rate constant.

The current platform-neutral structured residual is trained at 0.1, 0.5, and
1.0 seconds with the shared long-rollout objective described below. The default
multirotor reference uses the exact memoryless torque map because these inputs
are measured rotor speed. Automatic batching is governed by unrolled
transition count, so this medium corpus uses all 7,192 valid multi-horizon
windows while larger or longer-horizon corpora remain bounded.

A maintainer-owned train-only sweep also evaluates a bounded angular-dynamics
authority. It adds no user-facing fitting knob: authority one is the fitted
model and zero approaches constant measured body rate. Leave-one-profile-out
selection on Chirp, Random, and Square chose 0.75, improving the equal-profile,
equal-horizon, equal-metric score by 4.98%. More conservative candidates were
rejected because their worst individual metric regressed by more than 25%.
Melon is excluded from selection and used once for the protected promotion
check.

The promoted NanoDrone artifact has boundary-safe cumulative 1-to-50-step
position, velocity, attitude, and angular-rate errors of 2.2176 m, 12.2297 m/s,
5.4456 rad, and 29.1772 rad/s. At 0.1 seconds the corresponding errors are
0.0118 m, 0.0927 m/s, 0.0263 rad, and 0.4642 rad/s; at 0.5 seconds they are
0.1176 m, 0.5569 m/s, 0.2308 rad, and 0.7093 rad/s. The protected check improved
all eight cumulative and 0.5-second metrics relative to full angular authority.

Against Phys+Res in the paper's Table 6, its equal-metric geometric cumulative
ratio is 0.9957, effectively parity: Glassbox is better on cumulative position
and attitude, 17.6% worse on velocity, and 0.66% worse on angular rate. At 0.5
seconds position, velocity, and attitude are within 5.2% of the published values,
while angular-rate error remains 18.6% higher. The comparison is near-comparable,
not exact, because Glassbox never crosses recording boundaries while the released
upstream metric code concatenates the three Melon runs. This is competitive
benchmark performance, not a state-of-the-art claim or evidence of complete-flight
open-loop stability.

A learned latent rotational response and cross-axis mixer were also evaluated.
They improved the training-profile score but failed the one-shot Melon promotion
check, so neither is enabled by default. The expressive implementation and audit
artifacts remain available for future airframes whose telemetry supports them.

## Published PX4 ULog reference corpora

### ARP quadrotor system identification

Prepare the four raw large-quadrotor ULogs released with ARP Laboratory's
[data-driven system-identification work](https://github.com/arplaboratory/data-driven-system-identification)
with one command:

```bash
uv run glassbox-ulog prepare-arp artifacts/arp_reference
```

Glassbox downloads the 58.4 MB snapshot from a pinned upstream commit, verifies
every SHA-256 checksum, and writes four 50 Hz canonical trajectories under
`artifacts/arp_reference/canonical/`. The adapter derives the canonical motor
order from each log's PX4 control-allocation geometry; it does not carry a
dataset-specific motor-order override. It retains the longest sustained powered
interval in each recording, yielding four trajectories totaling 215.98 seconds
(28.08, 53.88, 57.64, and 76.38 seconds). Each recording receives a stable,
path-independent `source_group`, so leave-one-recording-out evaluation cannot
accidentally mix segments from the same flight.

These recordings omit the usual arming and land-detection streams, and their
local-position origin is not the takeoff point. The reference adapter therefore
uses telemetry completeness followed by the powered-interval selection instead
of armed, landed, or local-height gates. Operational ULog ingestion keeps the
normal armed, airborne, and 0.2 m height defaults. The four recordings are
replicates from one vehicle, so they are useful real-ULog integration and
system-identification references, not four independent airframes.

Glassbox used logs 63--65 for all rotational-structure and authority decisions,
then evaluated the selected candidate once on protected log 66. On the
development folds, a learned latent rotational response with cross-axis control
coupling improved the instantaneous diagonal reference by 4.93%; the shared
train-only authority selector then chose 0.75 for a further 4.62% improvement.
This is an airframe-neutral multirotor selection mechanism with a fixed
candidate policy, not an additional end-user fitting knob.

On protected log 66, the combined candidate improved the fitted reference's
equal-horizon, equal-metric geometric score by 11.49%, with its worst individual
metric changing by only +0.07%. It reduced the aggregate rotational score by
17.91% and delayed the first configured complete-rollout divergence threshold
from 1.06 to 2.00 seconds. However, it remained 34.08% worse than kinematic
persistence overall and still crossed the divergence threshold. At 0.5 seconds,
candidate position and velocity errors were 1.63x and 1.68x persistence; at one
second they were 2.03x and 2.39x. The promotion result is therefore
`improves_reference_only`, not a promoted default or a complete-flight claim.
The rotational structure transfers as a useful hypothesis, while normalized
motor-command-to-translational-acceleration modeling is now the clearest
multirotor limitation to address on development data.

The next development-only comparison added one shared bounded command offset to
the v3 multirotor force law. The zero-offset map remains the normal fitting
default, and the experimental parameter is available only through maintainer
evaluation code; there is no new CLI knob. Control semantics enforce the
boundary: normalized motor commands may fit the offset, while measured
squared-rotor-speed thrust proxies must use the identity map.

On leave-one-recording-out folds over logs 63--65, the offset alone was rejected:
it improved the aggregate score by only 0.40% and its worst individual metric
regressed 79.0%. Jointly scoring the fixed authority grid against the original
zero-offset reference instead selected offset plus 0.5 angular authority. That
composite improved the reference by 6.27%, with a 19.32% worst individual
regression, and beat kinematic persistence by 27.85% geometrically; its worst
persistence cell was 4.19% higher. The three held-out fits learned consistent
offsets of -0.111, -0.105, and -0.072 normalized command. This is useful
development evidence for the more expressive force law, but it is not promoted:
log 66 was already consumed by the preceding one-shot evaluation, so an
untouched second normalized-command airframe is required for a valid promotion
decision.

### IDF-DS fixed-wing telemetry

Prepare the raw PX4 portion of the
[IDF-DS fixed-wing telemetry release](https://zenodo.org/records/16992976)
with:

```bash
uv run glassbox-ulog prepare-idf artifacts/idf_reference
```

This is a substantial acquisition: the pinned Holybro archive is 2.12 GB.
Glassbox verifies its published MD5, extracts only the 13 raw ULogs, verifies
each member's size and CRC32, and converts every telemetry-complete interval to
50 Hz canonical trajectories. A 0.2-second gap tolerance accommodates the
logged 10 Hz actuator streams; gaps are never bridged into a rollout, and only
contiguous segments of at least 10 seconds are retained. The command also writes
`artifacts/idf_reference/corpus_report.json` with retained-duration coverage,
per-session segment counts, duration quantiles, motion ranges, control
excitation, and the common typed trajectory contract.

The current pinned extraction contains 119 contiguous trajectories from 13
independent sessions and retains 34,908.46 seconds (9.70 hours) of telemetry.
Session identity, rather than segment count or duration, determines both
training weight and held-out folds.

The archive's 120 published “flights” are processed circuit laps, not 120 raw
ULogs. The 13 available ULogs are long sessions containing the source telemetry
for laps 1–116. Processed laps 117–120 refer to another source log that is not
included in the archive, and this limitation is recorded in provenance. All
available sessions use the same conventional fixed-wing configuration: one
motor, split ailerons, elevator, and rudder. The adapter reconstructs
`throttle, aileron, elevator, rudder` from the logged PX4 allocation instead of
hard-coding channel indices.

Measure generalization across recording sessions with the opinionated
leave-one-source-group-out benchmark:

```bash
uv run glassbox-source-benchmark \
  artifacts/idf_reference/canonical/*.npz \
  --output-dir artifacts/idf_reference/source_benchmark_structured
```

Every dropout-separated segment from the held-out ULog moves into the same
fold. Training uses the automatic deterministic window budget, and the summary
reports equal-session macro metrics plus median, 90th-percentile, and worst-fold
errors. The default policy evaluates every source group using the standard
0.1, 0.5, and 2-second training horizons; it intentionally exposes no
dataset-specific sampling or optimizer knobs. Use `--model-class
structured_residual` with a separate output directory to compare the generic
residual under exactly the same folds. Dataset inputs are SHA-256 recorded, and
each completed fold is resumable only when that exact request still matches.

The current best fixed-wing result uses the platform-neutral residual around
the v3 structured model. Its equal-session macro errors are:

| Horizon | Position [m] | Velocity [m/s] | Attitude [deg] | Angular velocity [rad/s] |
| ---: | ---: | ---: | ---: | ---: |
| 0.1 s | 0.0255 | 0.511 | 0.786 | 0.159 |
| 0.5 s | 0.199 | 0.721 | 4.54 | 0.174 |
| 1.0 s | 0.408 | 0.785 | 7.16 | 0.159 |
| 2.0 s | 0.951 | 1.10 | 10.62 | 0.152 |

At two seconds the p90 held-out-session errors are 1.19 m, 1.33 m/s,
12.63 degrees, and 0.174 rad/s. Relative to the preceding speed-scaled-damping
model, learned lateral surface cross-coupling improves the geometric aggregate
by 4.57%; all 13 fold scores improve and every aggregate horizon/metric
improves. The learned aileron-to-yaw coefficient is also repeatable across all
folds. These are strong model-selection results on one conventional airframe,
not evidence of zero-shot parameter transfer. The result now feeds the
cross-airframe development contract below rather than standing alone as a
performance claim.

### Skywalker X8 flying-wing system identification

Prepare NTNU's pinned
[Skywalker X8 system-identification campaign](https://doi.org/10.18710/U4TLYV)
with:

```bash
uv run glassbox-x8 prepare artifacts/x8_reference
```

This is the first materially different real fixed-wing configuration in the
Glassbox evaluation matrix. It contains 17 dedicated, approximately ten-second
maneuvers at 40 Hz: 13 upstream training maneuvers and four untouched validation
maneuvers. The adapter emits the same rigid-body state contract but a genuine
flying-wing control layout: normalized throttle plus roll and pitch generalized
elevon angles in radians. No yaw control, conventional tail, PX4 topic, or
hard-coded surface index is required by the model.

The source combines a STIM300 IMU, ArduPilot EKF state, and separately generated
surface excitation. It also provides a strong north-west wind estimate whose
vertical component was validated by the authors against an earlier five-hole
probe flight. Glassbox types the three wind components as trusted NWU context,
samples them at rollout initialization, and holds them through each prediction.
The lower-rate upsampled GPS position is visibly staircase-like, so the adapter
reconstructs local position by trapezoidally integrating the internally
consistent 40 Hz EKF velocity and records the GPS endpoint discrepancy.

Fit both maintained model classes on the exact upstream split:

```bash
uv run glassbox-fit \
  artifacts/x8_reference/canonical/training/*.npz \
  artifacts/x8_reference/canonical/validation/*.npz \
  --holdout-count 4 \
  --training-horizons 0.1,0.5,2.0 \
  --skip-no-lag-ablation \
  --model artifacts/x8_reference/structured_model.json \
  --report artifacts/x8_reference/structured_report.json

uv run glassbox-fit \
  artifacts/x8_reference/canonical/training/*.npz \
  artifacts/x8_reference/canonical/validation/*.npz \
  --holdout-count 4 \
  --training-horizons 0.1,0.5,2.0 \
  --model-class structured_residual \
  --skip-no-lag-ablation \
  --model artifacts/x8_reference/residual_model.json \
  --report artifacts/x8_reference/residual_report.json
```

Then run the boundary-safe rolling comparison against constant-velocity,
constant-body-rate persistence:

```bash
uv run glassbox-x8 evaluate artifacts/x8_reference \
  --structured-model artifacts/x8_reference/structured_model.json \
  --residual-model artifacts/x8_reference/residual_model.json \
  --report artifacts/x8_reference/benchmark_report.json
```

The residual's equal-validation-maneuver rolling errors are:

| Horizon | Position [m] | Velocity [m/s] | Attitude [deg] | Angular velocity [rad/s] |
| ---: | ---: | ---: | ---: | ---: |
| 0.1 s | 0.00574 | 0.122 | 0.754 | 0.128 |
| 0.5 s | 0.0555 | 0.260 | 3.42 | 0.175 |
| 1.0 s | 0.146 | 0.354 | 4.94 | 0.181 |
| 2.0 s | 0.429 | 0.585 | 7.41 | 0.184 |

Across every horizon and state metric, its geometric score is 0.414 relative
to kinematic persistence and 0.811 relative to the structured model, so it is
58.6% and 18.9% better respectively. The structured model retains the better
complete-maneuver rollout: 6.41 m / 23.8 degrees over the four 9--12 second
validation maneuvers versus 10.44 m / 34.2 degrees for the residual. A matched
no-wind ablation is decisive: two-second structured accuracy degrades from
0.548 m / 7.02 degrees to 0.896 m / 24.3 degrees, and every complete maneuver
becomes non-finite. This supports the typed-wind mechanism without claiming
that every estimator-derived wind source is trustworthy.

### EPFL TOPOPlane2 conventional fixed wing

Prepare EPFL's pinned
[TOPOPlane2 navigation flight](https://zenodo.org/records/10337559) with:

```bash
uv run glassbox-epfl prepare artifacts/epfl_topoplane
```

This adds a second conventional configuration and a third fixed-wing dataset
family. The 5 Hz fused state is paired with GNSS-tagged autopilot outputs for
aileron, elevator, throttle, and rudder, plus a measured pitot-airspeed channel.
The adapter converts the upstream NED/FRD state to NWU/FLU, decodes the
source's scalar-first quaternion storage, maps PWM to typed normalized actuator
outputs, and converts WGS84 coordinates to a segment-local tangent frame.

The published flight intentionally exercises navigation outages. Glassbox does
not mistake the resulting INS position drift for vehicle motion: it compares
the fused altitude against the independent barometric-altitude signal, keeps
only the dominant navigation-consistent mode, and removes two seconds around
every boundary. The pinned extraction retains seven segments and about 791
seconds of dynamics-grade flight. Because the upstream fused message leaves
angular velocity at zero, body rates are reconstructed from the attitude
derivative at the native 5 Hz rate and this limitation is recorded explicitly.

This is characterization evidence, not a promotion benchmark. All segments
come from one flight, so no train/validation partition can establish
independent-flight generalization. The adapter labels them accordingly and
keeps one source-group identity across every segment.

For a reproducible same-flight characterization, fit the two maintained model
classes on the canonical segments and combine their reports with:

```bash
uv run glassbox-epfl evaluate \
  --structured-report artifacts/epfl_topoplane/structured_report.json \
  --residual-report artifacts/epfl_topoplane/residual_report.json \
  --output artifacts/epfl_topoplane/characterization_report.json
```

The evaluator scores the chronological final segments against kinematic
persistence, records that training and validation share a source flight, and
refuses to mark either model as promotable.

### Cross-airframe fixed-wing development gate

The versioned `fixedwing_prediction_development_v1` contract prevents progress
on one airframe from hiding regressions on another. It gives the conventional
IDF aircraft and the X8 flying wing equal weight, and requires each airframe to
meet both equal-flight aggregate and p90 flight errors:

| Horizon | Maximum position RMSE | Maximum attitude RMSE |
| ---: | ---: | ---: |
| 0.5 s | 0.25 m | 6 deg |
| 1.0 s | 0.60 m | 9 deg |
| 2.0 s | 1.20 m | 13 deg |

Every airframe must also beat constant-world-velocity, constant-body-rate
kinematic persistence by at least 5% over the 0.5, 1, and 2-second metrics, and
every complete rollout must remain finite. These are held-out prediction
development targets, not flight-safety or certification limits. Candidate
selection is separate from acceptance: a candidate must improve at least 1%
overall, may not regress either airframe by more than 5%, may not regress an
individual metric by more than 50%, and must keep all complete rollouts finite.

Evaluate the two model classes and make the selection with:

```bash
uv run glassbox-fixedwing-gate evaluate \
  --idf-summary artifacts/idf_reference/source_benchmark_lateral_cross_coupling_structured_v3/summary.json \
  --x8-report artifacts/x8_reference/benchmark_report.json \
  --x8-model-name structured \
  --candidate-name structured_v3 \
  --output artifacts/fixedwing_cross_airframe/structured_v3_gate.json

uv run glassbox-fixedwing-gate evaluate \
  --idf-summary artifacts/idf_reference/source_benchmark_lateral_cross_coupling_residual_v3/summary.json \
  --x8-report artifacts/x8_reference/benchmark_report.json \
  --x8-model-name residual \
  --candidate-name structured_residual_v3 \
  --output artifacts/fixedwing_cross_airframe/residual_v3_gate.json

uv run glassbox-fixedwing-gate compare \
  --reference artifacts/fixedwing_cross_airframe/structured_v3_gate.json \
  --candidate artifacts/fixedwing_cross_airframe/residual_v3_gate.json \
  --output artifacts/fixedwing_cross_airframe/selection_v1.json
```

The residual is selected for continued development. Its candidate/reference
score is 0.780 on IDF and 0.802 on X8, for an equal-airframe score of 0.791;
median stable rollout duration also increases from 2.82 to 3.46 seconds on IDF
and from 1.34 to 1.75 seconds on X8. All complete rollouts remain finite. It
does not yet pass the accuracy contract: IDF p90 position error at 0.5 seconds
is 0.2523 m against a 0.2500 m limit. X8 passes every contract requirement.
The miss is reported rather than rounded away or used to relax the threshold.

The divergence diagnostics make candidate hypotheses concrete: IDF failures
are dominated by velocity error, while three of four X8 validation maneuvers
first cross the angular-rate threshold. They do not, by themselves, prove which
extra coefficient belongs in the shared model. Use the fail-fast single-airframe
screen before paying for a full cross-airframe fit:

```bash
uv run glassbox-fixedwing-gate screen \
  --reference-report artifacts/x8_reference/benchmark_report.json \
  --candidate-report artifacts/x8_reference/candidate/benchmark_report.json \
  --model-name structured_residual \
  --airframe-name x8_flying_wing \
  --candidate-name candidate \
  --output artifacts/fixedwing_cross_airframe/candidate_screen.json
```

The screen uses the contract's 0.5, 1, and 2-second metrics, requires at least
1% aggregate improvement, and rejects any individual metric regression above
50%. Passing only authorizes the more expensive cross-airframe evaluation; a
single airframe can never promote a model.

This screen rejected four physically plausible follow-up experiments before
running IDF's 13 folds:

| Candidate | X8 candidate/reference score | Decision |
| --- | ---: | --- |
| surface/rate force plus cross-rate moment derivatives | 1.176 | reject |
| cross-rate moments only | 1.133 | reject |
| cross-rate moments with a frozen-base staged residual | 1.056 | reject |
| v3 with a frozen-base staged residual | 1.011 | reject |

Lower is better. The accepted joint residual v3 is therefore retained. The
negative result is useful: residual Jacobians are not sufficient evidence for
promoting a structured term because the base and residual co-adapt during
fitting. The next broadly useful evidence should come from another independent
airframe or better aerodynamic observations such as trusted airspeed and wind,
not from tuning more coefficients against IDF or X8.

## PX4 ULogs

Inspect the topics and fields in a log:

```bash
uv run glassbox-ulog inspect path/to/flight.ulg
```

Extract estimated state and actuator commands to the canonical trajectory format:

```bash
uv run glassbox-ulog extract path/to/flight.ulg artifacts/flight.npz \
  --rate 50
```

For SITL logs containing the PX4 ground-truth topics, add `--state-source ground_truth`.

PX4 motor function order depends on the airframe configuration. Glassbox derives the front-left, front-right, rear-right, rear-left channel order from the logged `CA_ROTOR*_PX/PY` geometry. For unusual geometries, provide the channel order explicitly with `--motor-indices`.

For fixed-wing logs, join the normalized motor and servo allocator topics with:

```bash
uv run glassbox-ulog extract-fixedwing path/to/plane.ulg \
  artifacts/plane.npz --rate 50
```

Automatic fixed-wing mapping requires one logged PX4 rotor and an independent
roll/pitch control-surface allocation; yaw authority is optional. Glassbox
reconstructs canonical aerodynamic-axis controls from
`CA_SV_CS*_TRQ_R/P/Y`, converts PX4 FRD pitch/yaw moment signs into FLU, and
records the raw and canonical mixing matrices under `provenance["px4"]`.
Two-surface elevon configurations therefore ingest as `throttle, roll, pitch`
without a fictional rudder channel. A nonzero `CA_SV_CS*_FLAP` allocation adds
a typed `flap` channel. Explicit conventional aileron/elevator/rudder slots
remain available for older logs. Pass a stable `--vehicle-id` when producing a
multi-airframe corpus so the fitter cannot silently pool different physical
vehicles.

### Record a reproducible SITL flight

With Docker Desktop running, record a PX4 SIH quadrotor takeoff–hover–landing flight and extract both estimated and ground-truth trajectories:

```bash
./scripts/record_sitl.sh
```

The script uses the same immutable multi-architecture PX4 SIH image digest as
the integration gate and stores generated data under the ignored
`artifacts/sitl/` directory. Set `GLASSBOX_PX4_IMAGE` only to make an explicit
image comparison. [logger_topics.txt](config/logging/logger_topics.txt)
overrides the dynamics topics to their full publication rates; the default PX4
logging profile is intended for flight review and records some actuator signals
too slowly for identification. For SIH, the script extracts the normalized
`actuator_outputs_sim` signal consumed by the simulator at 250 Hz.

The same fitting step can be run on any extracted trajectory:

```bash
uv run glassbox-fit artifacts/flight.npz \
  --model artifacts/flight_model.json \
  --report artifacts/flight_fit.json
```

With one trajectory, the fitter uses the first 70% for multi-step training and
reserves the final 30% for a contiguous held-out rollout. With multiple
trajectories, it reserves the final complete source group when `source_group`
labels are present; otherwise it falls back to the final input trajectory:

```bash
uv run glassbox-fit artifacts/flight_1.npz artifacts/flight_2.npz \
  artifacts/flight_3.npz \
  --training-horizons 2.0 \
  --evaluation-horizons 0.1,0.5,1.0,2.0 \
  --model artifacts/multi_flight_model.json \
  --report artifacts/multi_flight_report.json
```

Every trajectory extracted from a PX4 ULog carries the source recording as its
`source_group`. If telemetry gaps produce multiple retained intervals, those
segments keep the same group, preventing a flight from leaking across a
source-level holdout.

`--training-horizons` expresses rollout lengths in seconds, so the objective is independent of telemetry sample rate. One value trains at that horizon; multiple comma-separated values combine their initial-loss-normalized objectives. Longer horizons directly penalize compounding rollout drift, while shorter horizons emphasize fast local dynamics.

The SITL recorder currently uses a 2-second training horizon. This substantially reduces uninterrupted long-rollout drift while the evaluation report still exposes 0.1, 0.5, 1, and 2-second behavior. The same multi-flight command works for simulator ground-truth and estimated-state artifacts; estimated-state coefficients should be interpreted as predictive effective values because estimator filtering and noise are part of the observed signal.

By default, the command fits both the learned-lag model and an otherwise identical near-zero-lag ablation. The report contains aggregate and per-flight metrics for the complete rollout and each requested horizon. When `--model` is provided, the ablation is written beside it with `_no_motor_lag` appended. Use `--holdout-count` to reserve more than one final source group—or more than one trajectory when groups are unlabeled—or `--skip-no-lag-ablation` when the comparison is not needed. When every multirotor training trajectory carries typed specific force, the command automatically runs a chronological sensor-residual diagnostic for thrust, timing, and rotational response. It records identifiability and boundary diagnostics without adding user-facing controls. This stage remains diagnostic-only because its initializer failed the maintained Nano and ARP rollout gates; rollout optimization still starts from the stable reference parameters. Model files contain effective predictive coefficients, the exact runtime prediction contract, the training-only observation schema, and fitting provenance. They explicitly do not claim that those coefficients are uniquely recovered physical parameters.

Held-out evaluation also reports measured-state-reset one-step innovations. The
latent actuator state is carried causally, but the 13-element rigid-body state is
reset at every interval so temporal autocorrelation and current/past-control
correlation can be inspected without complete-rollout drift. A companion
model-independent compatibility check compares position increments with reported
world velocity and attitude increments with reported body rate. Correlation
flags are diagnostic rather than promotion criteria: estimator filtering,
closed-loop feedback, and incompatible state channels can produce them without
implying a missing aerodynamic term. The policy is maintained internally and
adds no CLI knobs.

A research-only compatibility utility also tests whether reported velocity and
body rate behave like bounded first-order observations of pose-consistent
motion. It fits development flights only and compares against an equally
flexible zero-memory reference on complete research-validation flights. Its
channel-level gate found transferable body-rate memory on ARP, X8, and IDF and
authorized a body-rate-only rollout A/B without modifying velocity or the
physical dynamics. The filter improved the 0.1 s rate metric by 8.8–13.1%, but
the advantage became nearly neutral by 0.5–1.0 s; no corpus met the maintained
10% across-horizon materiality threshold. It is therefore not applied by the
fitter and adds no user-facing configuration. The evidence is documented in
[`docs/temporal-observation-filter-results.json`](docs/temporal-observation-filter-results.json)
and
[`docs/body-rate-observation-rollout-results.json`](docs/body-rate-observation-rollout-results.json).

A final signed timestamp-alignment diagnostic likewise found substantial
body-rate improvements on X8 and IDF. It was not advanced because the temporal
candidate transferred to more platforms and subsequently failed the rollout
gate. Its implementation is also isolated from normal fitting; the evidence is
in
[`docs/state-observation-alignment-results.json`](docs/state-observation-alignment-results.json).

A separate dynamics-history experiment then tested a bounded causal innovation
observer against the strongest maintained model on each corpus. The observer
selected the exact no-op on Nano and X8. It improved ARP relative to its own
instantaneous model, but still lost to kinematic persistence by 26.9%. An IDF
candidate improved the aggregate while regressing one flight metric by 11.4%, so
the guardrail rejected it. The observer is therefore not retained in the runtime
or fitter. The negative result is documented in
[`docs/residual-innovation-observer-results.json`](docs/residual-innovation-observer-results.json).

That closes the bounded literature-guided architecture cycle. Until materially
new measurements or an externally validated method changes the evidence,
Glassbox keeps the current dynamics model as an audited gray-box baseline.

The evaluation hierarchy is deliberately asymmetric. A material improvement in
one identifiable state channel may authorize a research A/B for that channel;
blanket/default adoption must improve every eligible channel. Reused evaluation
flights are classified as research validation, and any production promotion
would additionally require a genuinely fresh lockbox.

### Scaling across logs

Canonical trajectory artifacts now carry a typed, versioned semantic contract:

- `spec` defines the 13-state schema, observation source, ordered control
  roles/units/bounds, optional measured exogenous channels, optional
  state-aligned identification observations, and vehicle configuration
  identity.
- `labels` contains profile, condition, replicate, vehicle, payload,
  environment, and source-group annotations.
- `provenance` contains the source path, adapter identity, PX4 topics, raw
  actuator mapping, filters, and discarded intervals.

Source integrations implement the small `TrajectoryAdapter` boundary:
`inspect(path)` audits the source schema and `load(path)` returns a canonical
trajectory. Source-specific fields stop at `provenance`; fitting and dynamics
consume the typed spec and numerical arrays only.

For long recordings containing dropouts, `load_px4_trajectories()` preserves
every valid contiguous interval rather than silently keeping only the longest.
Reference-corpus adapters expose this behavior as `load_all()` while retaining
the one-trajectory `load()` protocol for ordinary integrations.

Dataset pooling compares `spec`, not source details. Logs using different PX4
topics, fields, motor slots, or surface allocation matrices may pool when the
adapter has converted them into the same verified canonical semantics. Logs
with different observation sources, control meanings/order, fixed
configuration states, or vehicle configuration IDs are rejected. Sample rate
is checked separately because rollout windows share one integration step.

The canonical NPZ format is version 3 and contains the state, control,
exogenous, and training-only observation arrays plus typed `spec`, `labels`,
and `provenance`. The loader
intentionally rejects every other version; derived trajectories should be
re-extracted from their raw telemetry rather than migrated through compatibility
shims. For recorded profile corpora, use
`scripts/reextract_profile_dataset.py`.

```bash
uv run python scripts/reextract_profile_dataset.py \
  artifacts/sitl/multirotor_v2 \
  --platform multirotor \
  --vehicle-id px4_sih_quadx
```

Observation channels are measured outputs used to identify dynamics, such as
accelerometer specific force and filtered body angular acceleration. They are
state-aligned but are not assumed to exist at rollout time. Model artifacts
therefore exclude them from `input_spec` and record them separately as
`identification_observations`.

Exogenous channels are measured non-control inputs available when prediction
starts, such as trusted wind. Each rollout samples them only at its initial
timestamp and holds that value through the horizon, preventing future telemetry
from leaking into an open-loop prediction. Trusted wind roles modify the
air-relative velocity in the structured force law. Estimated context may
instead condition the generic residual, with estimated-wind features barred
from directly changing its angular correction.

PX4's sensor-aided `airspeed_wind` estimate is fully typed and retained by the
general ULog adapter, including its variance audit. IDF-DS excludes that derived
estimate from model input by default: a matched experiment treating it as exact
wind worsened every representative metric, while residual conditioning traded
better position for worse attitude and angular-rate prediction. The exclusion
and source metadata remain in provenance, so this negative result is explicit
rather than a hidden dataset special case. The Skywalker X8 adapter makes the
opposite evidence-backed choice for its independently documented and validated
wind estimate; trust is a source-semantic fact, not a global setting.
Build a consistent 50 Hz dataset from a directory of raw ULogs with:

```bash
./scripts/extract_ulog_dataset.sh path/to/logs artifacts/dataset_50hz 50
```

The script attempts both ground-truth and estimated-state extraction, reports
logs without a valid airborne interval, and consistently uses the normalized
`actuator_motors.control` command available in operational PX4 logs. Then pass
the desired state source from every flight to `glassbox-fit`. For example, this
reserves two complete flights:

```bash
uv run glassbox-fit artifacts/dataset_50hz/*_ground_truth.npz \
  --holdout-count 2 \
  --training-horizons 0.1,0.5,2.0 \
  --model artifacts/scaled_model.json \
  --report artifacts/scaled_report.json
```

Multi-flight training gives every complete source group equal total loss weight
by default and weights windows uniformly inside each group. A long log cannot
dominate, and splitting one log around dropouts cannot increase its influence.
Without source-group labels, each trajectory remains one group. The report
records each trajectory's duration, motion and excitation characteristics,
candidate and selected window counts, effective training weight, source-group
weight share, validation metrics, an equal-trajectory macro aggregate, and a
sample-weighted aggregate.

Large corpora use an automatic deterministic window budget rather than exposing
another fitting knob. Small and medium corpora use every valid window; only the
fixed memory bound and the transition cost of long horizons thin large corpora.
Within that budget,
midpoint-stratified selection spans each group's complete concatenated timeline;
small datasets still use every candidate. Optimization then uses deterministic
weighted batches capped by both window count and unrolled transition count.
Short rollouts therefore retain far more examples per step than expensive long
rollouts. A low-discrepancy
phase schedule covers the selected corpus without random seeds while preserving
the configured source/profile weights. Initial and final losses are always
rescored on every selected window, and the report records batch sizes and total
window coverage. Use `--duration-weighted-training` only when weighting by
recorded duration is an intentional dataset choice.

The current six usable SITL logs total about 73 seconds, but their paths, speed
ranges, and actuator excitation show that they are repetitions of essentially
the same takeoff-hover-landing profile—not a diverse flight corpus. The scaled
benchmark is therefore a pipeline stress test, not evidence of general dynamics
coverage. On a four-flight/two-flight split, the ground-truth model trained at
0.1, 0.5, and 2 seconds reaches roughly 0.66 m position and 1.98 degrees
attitude RMSE across the two complete holdouts; the estimated-state counterpart
reaches roughly 0.43 m and 2.38 degrees. This is a more conservative baseline
than the earlier homogeneous three-log result.

### Maneuver-family dataset

Record the 24-flight expansion matrix—four bounded PX4 SITL profiles, three
excitation conditions, and two replicates—with:

```bash
./scripts/record_sitl_profiles.sh artifacts/sitl/multirotor_v2
```

The recorder uses PX4's normal takeoff and landing behavior, streams local-NED
position/yaw setpoints in Offboard mode, and extracts both estimated and
ground-truth trajectories at 50 Hz using the operational
`actuator_motors.control` signal. The setpoints only generate varied closed-loop
telemetry; they are not inputs to the learned dynamics model. Each artifact
stores its maneuver family, low/medium/high excitation condition, replicate,
and initial-yaw variant in trajectory labels. Low conditions use smaller, slower steps;
high conditions use larger, faster steps. Replicates rotate the profiles through
different initial headings. The default output root is new and the recorder
refuses to overwrite an existing run directory. Override the matrix with
`GLASSBOX_PROFILE_REPLICATES`, `GLASSBOX_PROFILE_CONDITIONS`, and
`GLASSBOX_PROFILE_INITIAL_YAWS`.

Canonical trajectory files now serialize ordered `control_names` and allow any
positive number of control channels. Pooling rejects mismatched channel counts
or order. The current quadrotor model family still explicitly requires four
canonical motor channels; this keeps the shared telemetry layer ready for
fixed-wing throttle and surface controls without silently feeding them to the
wrong dynamics model.

Reserve an entire family—not merely the last flight—with
`--holdout-profile`:

```bash
uv run glassbox-fit \
  $(find artifacts/sitl/profile_dataset -name '*_ground_truth.npz' | sort) \
  --holdout-profile combined \
  --training-horizons 0.1,0.5,2.0 \
  --model artifacts/sitl/profile_holdout_combined_model.json \
  --report artifacts/sitl/profile_holdout_combined_report.json
```

When all inputs have profile labels, training first gives every included
maneuver family equal total loss weight and then divides each family's weight
equally among its replicate flights. Run every holdout fold and write a macro
summary with:

```bash
uv run glassbox-profile-benchmark \
  $(find artifacts/sitl/profile_dataset -name '*_ground_truth.npz' | sort) \
  --output-dir artifacts/sitl/profile_benchmark
```

This leave-one-maneuver-family-out benchmark measures extrapolation to a type
of motion absent from training rather than interpolation to another execution
of a familiar flight. Its top-level aggregate gives every held-out profile equal
weight. `summary.json` also applies the versioned
`multirotor_prediction_v1` development contract. A horizon passes only when
both the equal-profile aggregate and worst held-out profile satisfy position and
attitude limits; full-flight position error is normalized by logged path length.
These targets evaluate predictive usefulness and are not flight-safety or
certification limits.

The ground-truth position/attitude targets are 0.001 m/0.25 degrees at 0.1 s,
0.01 m/1 degree at 0.5 s, 0.05 m/2 degrees at 1 s, 0.20 m/5 degrees at 2 s,
and 0.75 m/10 degrees at 5 s. Estimated-state targets are 0.02 m/0.5 degrees,
0.08 m/2 degrees, 0.15 m/4 degrees, 0.30 m/7 degrees, and 1.0 m/12 degrees at
the same horizons. Full-flight targets are at most 10% of path length and 10
degrees for ground truth, or 15% and 15 degrees for estimated state. The 0.1,
0.5, 1, and 2-second horizons plus full flight are required for an overall pass;
missing required horizons produce an `incomplete` result.

The original profile corpus contains eight flights and 150.1 seconds of usable
ground truth. It spans about 0.9–2.1 m/s peak speed, 0.34–3.02 rad/s peak body
rate, vertical-only motion, multi-metre lateral steps, large yaw steps, and
combined translation/yaw. The structured model's equal-profile ground-truth
benchmark reaches 0.00014 m / 0.12 degrees at 0.1 seconds, 0.024 m / 3.52
degrees at 1 second, and 0.165 m / 6.51 degrees at 2 seconds. Complete
17–20-second open-loop rollouts reach 6.71 m / 15.27 degrees. Estimated-state
training reaches 0.266 m / 6.73 degrees at 2 seconds and 7.59 m / 16.08 degrees
over complete flights. Same-profile replicate holdouts have nearly identical
error, showing that systematic rollout drift—not merely unseen profile labels—
is the dominant limitation.

The expanded `multirotor_v2` corpus contains 24 raw ULogs, 24 ground-truth
trajectories, and 24 estimated-state trajectories. Its 428.0 seconds of usable
ground truth are balanced at six flights per maneuver family, eight per
excitation condition, and twelve per initial heading; all artifacts are finite
and share the verified canonical motor schema. Peak speed reaches 3.16 m/s.

Re-extracting the retained ULogs through canonical format v3 and rerunning the
expanded structured leave-one-family-out benchmark gives ground-truth
position/attitude errors of 0.00012 m / 0.13 degrees at 0.1 seconds, 0.0031 m /
1.32 degrees at 0.5 seconds, 0.0216 m / 3.36 degrees at 1 second, 0.167 m / 6.57
degrees at 2 seconds, and 1.23 m / 10.18 degrees at 5 seconds. Complete flights
reach 7.39 m / 16.56 degrees. The matched estimated-state benchmark reaches
0.0117 m / 0.34 degrees, 0.0543 m / 2.01 degrees, 0.109 m / 4.19 degrees, 0.284
m / 7.19 degrees, and 1.34 m / 10.87 degrees at the same horizons; complete
flights reach 5.85 m / 15.84 degrees. Both contracts fail: position meets the
ground-truth target through 2 seconds, while attitude meets it only at 0.1
seconds. The modest local change and poor full-flight behavior over a broader
envelope indicate that the structured model is now the limiting factor, not
the amount of repeated data. The machine-readable rerun and no-promotion
decision are recorded in
[`multirotor-profile-results.json`](docs/multirotor-profile-results.json).

### Fixed-wing structured baseline

The shared model-family contract now separates platform, control names,
semantic roles, latent applied-control state, and residual support. Fixed-wing
requires the roles `throttle`, `roll`, and `pitch`; `yaw` and `flap` are
optional. Control columns may use airframe-specific names or order because the
dynamics indexes the static role layout carried by `TrajectorySpec`. A
conventional trajectory still uses `throttle, aileron, elevator, rudder`, while
a flying wing can use `throttle, roll, pitch`. Each fit compiles one fixed
layout, so incompatible configurations are rejected during pooling rather than
mixed inside a JAX batch.

The compact fixed-wing force law has body-X thrust, quadratic air-relative
lift and drag, a linearized angle-of-attack lift term, lateral velocity damping,
surface moments proportional to airspeed squared, pitch and lateral-directional
stability, airspeed-scaled angular-rate damping, signed lateral surface
cross-coupling, bounded learned surface-neutral offsets, and a first-order
actuator response. Its coefficients are effective acceleration
parameters: mass, inertia, density, and reference area are intentionally
absorbed. The model uses typed trusted wind when it is available and otherwise
assumes zero wind. It remains a low-angle attached-flow model and does not
model stall, propeller slipstream, unresolved gusts, or configuration changes.
When a moving flap role is present, additional learned coefficients model its
incremental lift, deployment drag, and signed pitch moment. Lateral
cross-coupling allows rudder-to-roll and aileron-to-yaw moments without assuming
either a conventional tail or a particular mixer. This expanded structure
serializes as
`effective_fixedwing_role_aerodynamic_lag_v3`. Model artifacts also use a
single current version and intentionally reject obsolete parameter layouts.

Generate and benchmark the closed-world synthetic smoke corpus with:

```bash
uv run glassbox-fixedwing-synthetic artifacts/fixedwing/synthetic_v1 \
  --flights 6 --duration 3
uv run glassbox-profile-benchmark artifacts/fixedwing/synthetic_v1/*.npz \
  --output-dir artifacts/fixedwing/profile_benchmark_synthetic_v1 \
  --training-horizons 0.1,0.5,1.0 \
  --evaluation-horizons 0.1,0.5,1.0,2.0 \
  --steps 600
```

Across three leave-one-multisine-profile-out folds, the fitted baseline reaches
0.0031 m position and 0.016 degrees attitude RMSE over the complete three-second
synthetic flights. This verifies family dispatch, differentiability, fitting,
evaluation, holdouts, and JSON round-tripping. It is not a real-airframe
accuracy result and does not participate in the real-flight fixed-wing
development contract. The same pipeline now also crosses the real PX4
telemetry boundary.

Record the standard PX4 SIH airplane matrix with:

```bash
./scripts/record_fixedwing_sitl_profiles.sh artifacts/sitl/fixedwing_v2
```

The recorder explicitly reconciles the SIH plant's 5--6 m/s envelope with PX4
runway-takeoff parameters, climbs before excitation, rotates the logger after
takeoff, streams bounded throttle/roll/pitch/combined attitude profiles in
Offboard mode, and extracts estimated and ground-truth trajectories. The
default matrix is four profiles by three excitation conditions by two
replicates. Set `GLASSBOX_PROFILE_REPLICATES=1` for the initial 12-flight pass;
set `GLASSBOX_PROFILE_REPLICATE_START=2` with a final replicate count of two to
append only the second replicate.

The current profile-only `fixedwing_v2` corpus contains 12 ULogs, 12
ground-truth trajectories, and 12 estimated-state trajectories. It provides
177.58 seconds of ground truth at 50 Hz across 5.00--6.78 m/s, with one verified
motor/surface mapping in every artifact. Pooled throttle spans 0.755--1.0;
aileron, elevator, and rudder contain nonzero variation without surface
saturation. An earlier takeoff-inclusive corpus is retained separately because
full-power takeoff occupied 65% of its throttle samples and would bias a pooled
maneuver fit.

On a leave-one-maneuver-family-out benchmark trained at 0.1, 0.5, and 2 seconds,
the current structured model reaches ground-truth position/attitude RMSE of
0.00058 m / 0.071 degrees at 0.1 seconds, 0.0122 m / 0.87 degrees at 0.5
seconds, 0.050 m / 2.07 degrees at 1 second, 0.185 m / 4.19 degrees at 2
seconds, and 1.01 m / 10.94 degrees at 5 seconds. The matched estimated-state
results are 0.0119 m / 0.23 degrees, 0.0578 m / 1.51 degrees, 0.126 m / 3.13
degrees, 0.294 m / 5.08 degrees, and 0.855 m / 9.71 degrees. Complete-profile
open-loop rollout still diverges: 6.54 m / 30.4 degrees on ground truth and
14.24 m / 69.6 degrees on estimated state.

Those numbers support an evidence-based operating envelope of roughly two
seconds for sub-0.3 m / 6-degree prediction and five seconds for approximately
one-metre / 11-degree prediction on this simulator. They do not participate in
the real-flight cross-airframe contract: the threshold-setting data and scoring
data are the same single-airframe, single-replicate-per-condition corpus.

### Platform-neutral structured residual

Fit the optional structured residual with:

```bash
uv run glassbox-fit trajectory_1.npz trajectory_2.npz trajectory_3.npz \
  --model-class structured_residual \
  --training-horizons 0.1,0.5,2.0,5.0 \
  --skip-no-lag-ablation
```

The wrapper retains the selected vehicle family's force law, exact rigid-body
position/quaternion kinematics, and latent applied-control response. A 16-unit
network predicts only six bounded corrections: body-linear and body-angular
acceleration. Its inputs are the frame-invariant body velocity, angular rate,
the typed canonical applied controls, and any typed start-of-rollout exogenous
context. Feature normalization and correction bounds are derived from the
training windows and serialized with the model; there are no multirotor motor
indices, hover assumptions, fixed-wing surface names, or NanoDrone
operating-range constants in the residual.

The same artifact and fitting path wraps multirotor, conventional fixed-wing,
flying-wing, flap-equipped, and future rigid-body base models. The control
feature count follows `TrajectorySpec`, so a three-control flying wing and a
five-control flap-equipped airplane compile distinct, audited model layouts.
Zero initialization exactly reproduces the structured base model on both
current vehicle families. The structured model remains the recommended default
until a residual improves leave-profile-out long-rollout accuracy without
destabilizing another motion family.

### Platform-neutral long-rollout objective

Every structured and structured-residual fit uses the same training policy.
Position, velocity, attitude, and angular velocity are treated as four equally
weighted semantic groups after scaling by robust within-window motion observed
only in the training data. Position scales use displacement rather than world
coordinates, so changing the trajectory origin does not change the objective.
Quaternion attitude error is sign-invariant.

Time weights increase linearly from one at the first predicted step to three at
the final step by default. A separate soft regularizer detects predicted
body-frame velocity or angular rate outside a generous training-derived
envelope: the half-width is at least four robust scales and covers at least the
99.5th percentile of training motion. Position is deliberately unbounded, and
the objective does not assume that an aircraft or multirotor is contractive.
The regularizer therefore targets numerical escape, not legitimate travel or
open-loop vehicle modes.

Both choices are explicit and recorded in every fit report:

```bash
uv run glassbox-fit trajectory_1.npz trajectory_2.npz \
  --training-horizons 0.1,0.5,1.0 \
  --endpoint-weight 3.0 \
  --stability-regularization 0.01
```

On the held-out synthetic fixed-wing smoke flight, the shared objective remains
stable for both model classes. After 100 steps, the structured model reaches
0.0213 m / 0.086 degrees over the complete flight; the structured residual
reaches 0.0088 m / 0.206 degrees. The opposing position/attitude trade again
supports retaining the structured family as the default rather than selecting
a residual solely from one aggregate score.

### Cross-platform fitting-policy selection

Use one leave-profile-out matrix to choose model class, training horizons, and
long-rollout loss settings without tuning on the NanoDrone benchmark test set:

```bash
uv run glassbox-select-policy \
  --dataset 'nanodrone=artifacts/nanodrone/canonical/train/*.npz' \
  --dataset 'fixedwing=artifacts/fixedwing/synthetic_v1/*.npz' \
  --output-dir artifacts/policy_selection
```

The CLI runs a maintained five-candidate plan rather than exposing optimizer,
horizon, regularization, model-class, and decision-threshold knobs. It compares
the structured and structured-residual models at short and extended horizons
under the shared stable-rollout objective, using 400 optimizer steps per fold.
The plan includes an explicit unregularized structured reference. It is named
and serialized, so changing it is a reviewed Glassbox policy change rather than
an undocumented per-user tuning choice. Use `--smoke` to validate a new corpus
and the complete workflow with two candidates and one optimizer step; smoke
results are not an accuracy comparison.

Each candidate is scored against the reference over position, velocity,
attitude, and angular-rate RMSE. Metrics and evaluation horizons are
geometric-mean aggregated within each held-out fold; folds are equally weighted
within each platform; platforms are equally weighted globally. A corpus with
at least two profiles uses profile holdouts. A single-profile corpus such as
IDF-DS automatically uses its independent `source_group` sessions instead.
Thus adding more logs, segments, or profiles for one vehicle family cannot give
it more influence in selection.

A candidate is ineligible if any rollout is non-finite, any individual metric
regresses by more than 50%, or an apparent aggregate improvement hides more
than a 5% platform-level regression. Small metric floors prevent ratios near
numerical zero from dominating the decision. A non-reference candidate must
also improve the overall score by at least 1%; otherwise the explicit reference
wins the near-tie. Trajectories labeled with any
`benchmark_split` other than `train` are rejected before fitting, so the public
NanoDrone test flights cannot enter policy selection accidentally.

Every fold writes its model, fit report, benchmark summary, and an exact request
record containing input SHA-256 hashes. Re-running the command resumes only
matching completed folds. The final `selection.json` records all candidates,
rejections, ranking, and the selected shared configuration. This is explicitly
a provisional model-development result: the selector never reads external test
data, and promotion to a maintained default requires a separate untouched-data
validation gate.

The `maintained_v1` development sweep used all 12 NanoDrone training logs and
six synthetic fixed-wing flights. It provisionally selected the structured
model trained at 0.1, 0.5, and 1.0 seconds with endpoint weight 3 and stability
regularization 0.01. Its equal-platform score improved 2.54% over the reference:
NanoDrone improved 6.20%, while fixed-wing regressed 1.27%. Both residual
candidates were rejected for individual metric regressions above 50%; the short
residual also concentrated its gains in fixed-wing while regressing NanoDrone.

The selected policy was then fitted on all NanoDrone training logs and evaluated
once against the protected Melon test split, alongside an exact fit of the
reference. It failed promotion: cumulative position and velocity error were
30.4% and 39.4% worse, while attitude and angular-rate error were 11.1% and
13.1% better. The equal-group geometric score was 8.85% worse. The reference is
therefore retained as the promotion baseline, and `maintained_v1` should not be
re-tuned against the Melon result. The complete development decision and test
reports are under `artifacts/policy_selection_v1/` and `artifacts/nanodrone/`.

Telemetry uses a 13-element observed state vector:

```text
[position_xyz, velocity_xyz, quaternion_wxyz, angular_velocity_xyz]
```

Controls are named and ordered by the selected model family. Multirotor uses
four normalized motor commands ordered front-left, front-right, rear-right,
rear-left. Fixed-wing consumes typed roles: throttle/roll/pitch are required,
while yaw and flap are optional. The coordinate convention is world Z-up and
body X-forward, Y-left, Z-up.

During rollout each model carries a latent applied-control vector through a
learned first-order time constant, making the complete simulated state Markovian
without requiring actuator-position or motor-speed telemetry.

## Current boundary

Both vehicle families now have differentiable rollout, fitting, serialization,
and PX4 ULog ingestion. Fixed-wing ingestion joins motor and servo allocator
topics, reconstructs signed aerodynamic-axis controls from logged allocation
parameters, and rejects unverifiable mappings. The present boundary is model
validity: fixed-wing short and medium rollouts transfer across maneuver
families, recording sessions, and both a conventional tail configuration and a
three-control flying wing. NanoDrone performance is competitive with its
published structured-residual reference, while the second multirotor airframe
shows that normalized-command translation is not yet consistently better than
kinematic persistence. The parameter artifacts remain airframe specific; this
evidence validates the shared interfaces and testable model hypotheses, not
zero-shot coefficient transfer.

Multi-minute IDF sessions, complete X8 maneuvers, and the protected ARP flight
still expose long-rollout instability. Flap-equipped or configuration-changing
aircraft also remain unvalidated. More same-vehicle segments are therefore
lower value than another airframe/configuration or a better generally applicable
command-to-force model. Use `--include-ground` only with a model that includes
ground-contact dynamics.

See [idea.md](idea.md) for the complete project scope.
