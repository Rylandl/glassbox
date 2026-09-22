# Online fitting: bounded solve and backtracking

**Online v8 is the sole maintained streaming fitter.** It reduces equal-family
one-step velocity/rate error **41.76% against adopted v6** across all 3,137 causal
forecasts: 53.19% on quads and 27.54% on fixed wings. All twelve primary case/metric
comparisons improve. The prospectively frozen primary gate passes (ratio 0.58240,
target ≤0.8; both families improve), as does the separate robustness gate.
Error is 87.73% lower than keeping the identical startup fit frozen.

This is improved future prediction from the same model formulation and observed
data. It does not establish a new vehicle class, blind generalization or online
controller recovery. Quad fitting is substantially slower and remains below its
required observation rate.

## One maintained procedure

The [frozen v8 protocol](harness/online-fit-v8.json) was committed as `ddc182b`
before implementation. Scientific source `899e0d9` was committed before the one
full six-tape run. There was no sweep or change to gates after measurement.

V8 runs sixteen scheduled preconditioned conjugate-gradient iterations, up from
four. It clips the resulting direction once using the existing forecast-change
bound, then tries scales 1, 1/2, 1/4, 1/8 and 1/16. A cached bounded loop stops at
the first finite step that decreases the exact retained loss with sufficient
actual-to-predicted gain. It does not select the best loss, rebuild a direction,
change damping between trials or fall back to the old four-step proposal.

The dynamics, initialization, v6 measured-RMS quadratic-head prior, scalar prior
strength, radial physical-group loss, replay roles, conditioning and inference
remain unchanged. Rejection retains the complete original model and normalizers;
damping changes once after the final decision. There is no vehicle branch or
consumer tuning option. Sixteen is an engineering budget, not a proven optimum.

Bounded `last_proposal` evidence records attempted steps and actual work. The
mutable session format is `glassbox-online-fit-v8`; older optimizer sessions are
rejected rather than silently migrated. Immutable model archives are unchanged.

## Physical forecast errors

The primary comparator is **adopted v6**, with exact matching inputs, targets,
initial core arrays, frozen predictions and kinematic predictions. Each cell
below is **v8 / v6 / no-fit kinematic**. The latter holds world velocity and body
rate while integrating orientation. Predictions precede target revelation and
assimilation: quads at 10 ms, fixed wings at 50 ms.

| Case | Velocity RMSE, m/s | Body-rate RMSE, rad/s | Forecasts |
| --- | ---: | ---: | ---: |
| quad-arm-115 | 0.01099 / 0.02375 / 0.03967 | 0.08674 / 0.23295 / 0.11166 | 875 |
| quad-arm-125 | 0.00414 / 0.01025 / 0.02942 | 0.03741 / 0.08107 / 0.12058 | 875 |
| quad-arm-135 | 1.16529 / 1.17741 / 1.17389 | 0.35416 / 0.91260 / 1.12088 | 62 |
| quad-change | 0.00414 / 0.01025 / 0.02942 | 0.03741 / 0.08107 / 0.12058 | 875 |
| fixedwing-80 | 0.27540 / 0.30899 / 0.16863 | 0.60397 / 0.78329 / 0.37618 | 225 |
| fixedwing-81 | 0.69000 / 1.24428 / 0.17023 | 1.69313 / 2.34088 / 0.37920 | 225 |

No sample or case is removed, including the 62-row tape ended by the behavior
controller's floor contact. The aggregate weights each family equally and each
case equally within its family. These are four Crazyflow scenarios of one
`cf21B_500` and two known Cascade Skywalker-X8 recordings. Quad125 and quad-change
share identical states/commands through 4.00 s, first differing at 4.01 s; their
post-change differences remain small (maximum velocity 0.0090 m/s,
rate 0.0453 rad/s, orientation 0.0205 rad). This is weakly excited change evidence,
not six independent airframes. The original Throw identifier used applied rotor
telemetry and is not the matched-input reference.

Fixed-wing absolute accuracy remains inadequate against the simple kinematic
check: velocity error is 1.63/4.05 times and rate error 1.61/4.47 times the
kinematic error on FW80/FW81.
Quad135 velocity improves only 1.03% against v6 and is just 0.73% better than
kinematics. Quad115 orientation remains 9.48% above the kinematic predictor.

## Tails and angular consistency

Equal-family worst-decile velocity/rate error decreases **42.86%**, orientation
RMSE **53.53%**, and truth-relative rotation/rate-defect RMSE **44.57%**. Both
families improve on every robustness aggregate; all eighteen case/group upper-
decile values and all six orientation RMSE values improve against v6. The defect
is a descriptive sampled-data consistency surrogate, not a zero-defect physical
law or a separate derivative-fidelity qualification.

Four descriptive quantile/maximum regressions remain: quad-change rate median
rises 2.13%; FW80 velocity p99 rises 8.37%, and its maximum rises from 1.297 to
1.578 m/s (+21.71%); FW81 rate p99 rises 1.85%. All quantiles, maxima and
concentration measures remain in the sealed pack and independent audit. Passing
aggregate gates does not imply every individual error improves.

## Actual work and latency

All 3,137 candidate updates accept, versus 3,135 for v6. Of these, 3,098 accept the
full step, 35 use half and 4 use a quarter. All 2,687 quad updates accept the full
sixteen-step direction. Backtracking is used on 39/450 fixed-wing updates.
Consequently the quad gains cannot be credited to shortening. The fixed-wing
result tests the combined procedure; there is no isolated sixteen-step-only
counterfactual, and the saved 64/128-step diagnosis is not that counterfactual.

The run uses 50,192 scheduled CG/curvature iterations, four times v6's 12,548;
gradient/conditioning calls remain 3,137 each. There are 3,180 trial evaluations,
or 1.014 per observation, and 6,317 total current/trial objectives versus 6,274
for v6: only 43 extra objectives (+0.69%). These counters exclude other internal
inference/derivative operations. The larger solve is the main increase in
scheduled work; backtracking itself rarely runs a second trial.

| Case | v6 update p95, ms | v8 median, ms | v8 p95, ms | v8 max, ms |
| --- | ---: | ---: | ---: | ---: |
| quad-arm-115 | 28.75 | 76.95 | 78.71 | 169.10 |
| quad-arm-125 | 28.45 | 77.08 | 78.38 | 143.35 |
| quad-arm-135 | 28.16 | 77.40 | 92.55 | 154.82 |
| quad-change | 29.53 | 77.02 | 79.87 | 131.76 |
| fixedwing-80 | 4.60 | 9.59 | 9.86 | 10.50 |
| fixedwing-81 | 4.38 | 9.60 | 9.89 | 11.04 |

The quad 10 ms target fails; fixed-wing 50 ms intervals accommodate the measured
9.86–9.89 ms p95 update times. Warmed prediction p95 is 0.81–1.18 ms. First compiled
quad/fixed-wing updates take 2.61/2.19 s. Timings synchronize calls and exclude
durable journaling, checkpoints and endpoint diagnostics. No other tests or
fits ran alongside this evaluation. Historical v6 latency is a recorded reference,
not a contemporaneously randomized timing comparison. These are local CPU
measurements, not a hardware-independent real-time guarantee.

Initial/final cache forecasts and their separate data/prior losses are retained
and independently recomputed. Caches and measured domains change over the stream;
endpoint totals are not a monotone training trace, and training loss is not the
primary accuracy evidence. On matched final caches, v8 data and prior losses can
be compared directly to v6 in the descriptive interpretation record.

## Verification and decision

All **414 tests pass** and Ruff passes. Analytic tests check the truncated 16-PCG
arithmetic with a nonzero prior, trust/predicted reduction, first acceptance,
nonfinite full-step rescue, zero/invalid rejection, actual dynamic trial calls,
original gain/damping thresholds, full rollback and archive save/resume/tampering.

The saved-data verifier passes with zero fits/model calls. A separate NumPy/SciPy
audit authenticates 400 manifest-listed payloads, including nested v6/v4/v2
evidence, and verifies all 3,137
paired rows, 9,411 journal events and 23 prospective snapshots. It checks
first-acceptable decisions from saved scalars, work, once-only damping, rollback,
endpoint arithmetic, metrics and gates. It does not regenerate every recorded
trial loss or forecast through an independent dynamics implementation.

All 14 package files other than `online.py` are byte-identical to v6. The complete
saved offline/Dart baseline replays exactly: 12,768 flight arrays, 40 trajectories
and 4,144 gradients. The previously qualified 0.720 mm Dart result is unchanged;
no new controlled-flight or precision trial was run here.

**Adopt v8** under the frozen accuracy/robustness decision. Real-time fitting,
fixed-wing absolute errors, calibration and live consumer recovery remain open.
See the [result index](online-fitting.json) for source, runtime, artifact hashes
and full verification records, and [replay instructions](../CONTRIBUTING.md#reproduce-streaming-fitting).

## Next named gap

**Cost of the adopted v8 online solve.** Freeze a saved-session component profile
before attempting an optimization. Identify the measured cost of conditioning,
linearization, repeated curvature actions and trial evaluation; pursue numerical
work reuse while preserving the sixteen-step proposal and the full forecast
comparison. Do not lower the solver budget, add a consumer option or trade away
the measured accuracy without a new prospective contract.

V6 remains reproducible at scientific source `2e465a4`; it is not a selectable
maintained implementation. The rejected v7 domain experiment and previous
read-only solver/backtracking diagnoses retain their historical verdicts.
