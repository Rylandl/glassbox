# Controlled Crazyflow and Cascade model evaluation

The [frozen protocol](harness/two-simulator-flight-v1.json) compares the unchanged
public generic learner, a freshly fitted structured reference and hold-current.
The two simulators each have 24 calibration/primary conditions and 18 declared
shift conditions, spanning heading, speed, maneuver intensity and wind. Whole
parent recordings separate training, development and testing. This is a direct
prediction and finite command-response baseline, not controller qualification.

Crazyflow uses the pinned Dart free-flight equations and 100 Hz recordings.
Cascade uses clean committed source `e8f6ba6`, native integration and 20 Hz
recordings. Native hidden state is saved for truth replay and branching; the
learner receives only its declared observed channels and issued commands.
Current Cascade physics differs from Dart's earlier comparison. A new result
does not erase the earlier saved-data failures.

Use the interpreter and package versions declared in the protocol. Set
`SCIPY_ARRAY_API=1`, `JAX_ENABLE_X64=1`, and `PYTHONPATH` to this checkout's `src`
and the clean Cascade archive's `src`. The module
`glassbox.experimental.two_simulator_flight` provides `generate`, `fit`,
`evaluate`, `finalize` and `replay` stages. Every invocation takes a simulator
name and `--output`. Fit additionally takes `--arm generic` or
`--arm structured`. Generate and seal both simulator datasets before fitting.
Finalize binds every data, model and evaluation artifact. Replay requires
`--expected-bundle-sha` from the committed result record, regenerates physical
parents and branches, and predicts from saved models without refitting.

Every planned test slot remains represented. Reports distinguish requested
conditions, achieved valid-prefix motion, invalid-tail diagnostics, missing
truth, unavailable models and nonfinite predictions. Errors are averaged over
origins, parents and cells before taking the square root; shift families remain
separate. The two fixed fitting workflows have different priors, history,
objectives and window/optimizer budgets. Their comparison does not isolate
architecture at equal compute.

## Verified baseline result

The [result record](harness/two-simulator-flight-v1-result.json) accepts these
evaluation environments and the descriptive baseline. The public learner is
unchanged. No new model, envelope or controller qualification is promoted.
Protocol `786c626` preceded trials and fitting; implementation `dabf7cf` produced
the final sealed bundle.

Each simulator has 180 three-second parent recordings: 96 calibration parents
(72 training, 24 development), 48 primary test parents and 36 shifted test
parents. Primary conditions cross initial headings 0°, 90°, −90° and 180°,
three speeds and gentle/strong collection maneuvers. Crazyflow primary speeds
are 0, 2 and 4 m/s; Cascade's are 16, 18 and 20 m/s. Shift families cover four
additional headings, speeds of 1/6 m/s or 14/22 m/s respectively, extreme
maneuvers and 2 m/s constant/gust wind. Speed is requested at initialization,
not guaranteed throughout a trajectory; unseen condition labels do not imply
disjoint achieved state support. Wind is unobserved by both predictors.

There are 1,680 planned factual forecast origins and 2,352 planned signed
response queries. The simulator generates 2,296 signed branches and 329
factual branch clones; unavailable origins remain in the full query roster.
Responses compare a perturbed command with the factual command from an exactly
cloned physical state and identical observed history. The generic learner
receives ordinary recordings, not paired response supervision or hidden state.

### Ordinary forecasts

Primary endpoint component RMSE, **generic / structured**:

| Simulator | Horizon | Velocity, m/s | Body rate, rad/s |
| --- | --- | --- | --- |
| Crazyflow, available truth | 10 ms | 0.00086 / 0.00051 | 0.00178 / 0.00995 |
| Crazyflow, available truth | 50 ms | 0.00688 / 0.00254 | 0.01682 / 0.04981 |
| Crazyflow, available truth | 150 ms | 0.04901 / 0.00963 | 0.11106 / 0.14691 |
| Crazyflow, available truth | 250 ms | 0.16060 / 0.03129 | 0.27335 / 0.23430 |
| Cascade | 50 ms | 0.01504 / 0.02073 | 0.05576 / 0.06450 |
| Cascade | 150 ms | 0.07452 / 0.07929 | 0.10832 / 0.11621 |
| Cascade | 250 ms | 0.17533 / 0.18726 | 0.11859 / 0.12490 |

At 250 ms, Cascade's velocity and rate errors improve 6.4% and 5.0% over the
structured reference. Rotation-entry error is worse: 0.01913 / 0.01611.
Crazyflow loses all three groups at 250 ms, despite better body-rate forecasts
at shorter horizons. Both generic models outperform hold-current on the
primary 250 ms factual metrics. The full report retains endpoint and whole-prefix
errors for every supported horizon, condition and signal group.

Shift-family 250 ms factual RMSE, **generic / structured**:

| Simulator | Shift | Velocity, m/s | Body rate, rad/s |
| --- | --- | --- | --- |
| Crazyflow, available truth | Heading | 0.1650 / 0.0269 | 0.2399 / 0.1963 |
| Crazyflow, available truth | Extreme maneuver | 0.3659 / 0.0568 | 0.5566 / 0.3755 |
| Crazyflow | Speed | 0.0721 / 0.0105 | 0.1137 / 0.0747 |
| Crazyflow | Wind | 0.0917 / 0.1003 | 0.1183 / 0.1062 |
| Cascade | Heading | 0.1636 / 0.2111 | 0.1129 / 0.1324 |
| Cascade | Extreme maneuver | 0.2391 / 0.2398 | 0.1736 / 0.1526 |
| Cascade | Speed | 0.2008 / 0.2271 | 0.1379 / 0.1164 |
| Cascade | Wind | 0.1512 / 0.2164 | 0.1208 / 0.2083 |

These descriptive comparisons concern two fixed simulator configurations.
They neither estimate performance across aircraft populations nor establish
real-flight or arbitrary-system readiness.

### Command response

Primary 250 ms endpoint response RMSE, **generic / structured / zero-response**:

| Simulator | Velocity, m/s | Body rate, rad/s |
| --- | --- | --- |
| Crazyflow, available truth | 0.1815 / 0.0166 / 0.1191 | 0.3322 / 0.1959 / 1.4936 |
| Cascade | 0.1398 / 0.0451 / 0.0898 | 0.0606 / 0.0316 / 0.0943 |

Both generic models lose velocity and rate response accuracy to the structured
reference. Velocity response error even exceeds predicting no response. Rate
responses beat that simple reference, substantially so in Crazyflow. Raw RMSE
includes weak probes; the saved direction diagnostics separately apply the
frozen physical response thresholds. These are finite-intervention measurements,
not qualifications of infinitesimal derivatives. Accurate ordinary forecasts
alone do not establish accurate command effects.

### Collection validity and error coverage

Crazyflow's collection pilot completes 136/180 parents; 44 cross the declared
0.5 m altitude boundary. Of these, 27 are calibration parents, eight are
primary test parents, five are heading-shift parents and four are all of the
extreme-maneuver parents. These are collection-pilot outcomes, not failures of
a learned controller. All 96 calibration recordings still have sufficient
valid prefixes for admission under the frozen rule. Every primary cell has
training coverage; the fixed development split covers 17 Crazyflow and 18
Cascade primary cells.

At 250 ms Crazyflow has valid truth for 447/480 primary factual queries and
720/768 primary response queries. Every planned slot remains visible; the
complete-cohort RMSE is unavailable. Reported Crazyflow errors condition on
available truth and do not certify the full requested envelope. Cascade
completes all 180 parents and all 480 primary factual/576 response queries.
Every eligible prediction is finite in both simulators.

Generic nominal 90% envelope coverage for primary 250 ms velocity/rate
components is 89.7%/89.4% in Crazyflow and 85.8%/88.5% in Cascade. Extreme
maneuvers reduce coverage to 55.4%/47.4% and 65.0%/68.3%, respectively.
Crazyflow coverage is also conditional on valid truth. These results do not
qualify the envelopes across the declared shifts.

### Fitting budget and diagnosis

Each generic fit uses the unchanged public recipe: 384 training windows, 256
development windows, 1,000 updates and batch size 64. Crazyflow selects step
zero: development loss is 0.04650 initially and 0.07539 at step 1,000, with
every sampled later checkpoint worse than initialization. Cascade selects step
1,000, improving development loss from 0.22884 to 0.17114. No fit-budget or
seed sweep was performed.

Structured fits use 600 full-batch steps, over 6,077 Crazyflow or 6,624 Cascade
training windows across 50/150/250 ms horizons. They receive the same raw
recording pool and parent roles, but use family priors, different objectives,
different history handling and more windows. Structured prediction replays the
full observed command prefix, with a 0.5 s equilibrium prefix; the generic
contract consumes 0.5 s observed history. Structured trajectories retain
observed positions; the generic model uses only the 15 velocity, rotation and
body-rate channels. This comparison evaluates the two
current fitting workflows and does not isolate architecture at equal compute.
Neither fit receives truth coefficients or hidden simulator state.

Inspection of the saved Crazyflow model establishes a specific limitation:
its nonlinear output weights and hidden-memory readout weights are exactly
zero. The remaining predictor is affine in observations and commands, using
only the explicit 100 ms delay history. Consequently, identical additive command
perturbations have history-independent predicted effects. The frozen probes'
actual deltas vary with the baseline command, so this does not imply identical
response traces for all saved probes. It is a plausible contributor to poor
world-velocity command response across attitudes, not proof of the sole cause.

The next named iteration should test one learned state–input interaction
mechanism under matched generic budgets, frozen factual-regression limits and
fresh held-out parent seeds. The present test set is now diagnostic evidence.
No platform equations or consumer tuning choices are needed to investigate
this limitation.

### Corrections, evidence and replay

The first attempt completed fitting but Cascade evaluation stopped before test
predictions: its requested surface-angle channels used a semantic unsupported
by the existing executable actuation map. Correction `dabf7cf` uses the existing
`surface_angle_command` semantic; units, bounds, channel order and physics are
unchanged. A serialized-spec-to-actuation regression test was added before
repeating both simulators' fits and evaluations. The failed attempt is preserved.
All 55,860 physical/data/query arrays, data roles and numeric query inputs repeat
byte for byte; completed Crazyflow scores also repeat exactly. Cascade's scores
first complete after this correction. The
[correction record](harness/two-simulator-flight-v1-correction.json) retains the
precise scope; this is not a silently repaired result.

The final bundle is `artifacts/2026-09-18/two-simulator-flight-v1`, anchored by
SHA-256 `5807e9f85454e5edefba6677e973a0e3851f4b60e8d8227c7d3bdc880ae656ad`
for `run.json`. Its 14,074 sealed payloads contain data, models and evaluations.
Fresh replay reproduces all 55,860 physical/data/query arrays, saved prediction
arrays and 257,040 metric rows exactly, without refitting. A separate NumPy
implementation verifies all 180 score groups at 250 ms endpoints, roles, rosters, branch
associations and coverage. Nine tamper cases pass, including coherently
rewritten physics and prediction/metric artifacts that fresh execution rejects.
Optimizer trajectories are not independently replayed.

All 52 focused tests pass in the pinned runtime; the base environment without
optional simulator packages passes 30 with 22 skips. Ruff passes. The result
record additionally anchors replay reports, independent scripts/results, audit
reports, the preserved failure, test logs and the clean Cascade source archive.
The [performance plot](../artifacts/2026-09-18/two-simulator-flight-v1-performance.png)
shows the primary 250 ms comparisons with their scope caveats.
