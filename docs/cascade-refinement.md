# Learning an X8 model from Cascade

Glassbox can fit an effective fixed-wing model from Cascade's Skywalker X8
telemetry and use that model for ordinary simulated tracking. The plant and
learner have independently implemented dynamics. Online refinement also runs
through the existing bounded worker and controller handoff, with modest changes
in this first comparison. This is one noise-free cruise case with a known
calibration pilot, not a heterogeneous-platform onboarding benchmark.

## Reproduce

```bash
uv run --group cascade python examples/cascade_refinement.py
uv run python scripts/audit_live_refinement.py artifacts/cascade-refinement
```

The optional dependency is locked to Cascade 0.2.0, commit `d6613886`. The
[example](../examples/cascade_refinement.py) defaults to three eight-second
calibration recordings, one further eight-second evaluation recording, 200 fit
steps, and two pairs of twelve-second tracking trials. `--calibration-only`
stops after prediction evaluation. `--reuse-calibration` checks the saved
recording and belief identities before reusing them; it does not refit.
`--output`, `--fit-steps`, `--recording-duration-s`, `--duration-s` and `--repeats`
make the experiment budget explicit. Reusing an output directory overwrites
reports; these are experiment artifacts, not resumable worker checkpoints.

## What each layer knows

Cascade advances the published X8 plant at 400 Hz with commands held for each
20 Hz observation interval. Its calibration pilot uses the published X8
stabilizer, configured for that command cadence, with simulator-derived trim
feedforward and small command/setpoint perturbations. That pilot and trim are
prior platform-specific work. The initial state and constant-command actuator
equilibrium also come from the simulator reset.

Glassbox receives a canonical NWU/FLU rigid-body state and three commands:
normalized throttle and generalized roll/pitch surface-angle requests in radians.
It has no yaw actuator. Surface requests are bounded to ±0.35 rad each, so their
additive elevon mix stays within the ±0.7 rad physical limits. All observations
are simulator truth; wind is fixed to zero. Fitting and refinement do not receive
the X8 coefficients, actual actuator states, propeller speed, or flow-separation
state. They use Glassbox's existing fixed-wing family and default initialization.

The consumer-side `TrackingPlant` interface supplies the initial observed state,
initial command, a command-to-next-observation callback and a reference function.
Synthetic and Cascade plants use the same tracking loop, `TransitionBuffer`,
`RefinementWorker`, prediction gate and acknowledged handoff. Only the plant
callback holds simulator internals. The learner has no Cascade-specific branch.

## Calibration and prediction

Recordings 0 and 1 fit the model and parameter information. Recording 2 measures
forecast error. Recording 3 is reserved for evaluating the initial and final
active models and never enters fitting, online absorption or the adoption gate.
It uses another excitation seed with the same pilot and flight condition, so it
tests a separate recording rather than a new maneuver distribution.

The [initial evaluation](investigations/cascade-refinement/held-out-evaluation.json)
reports the following back-to-back window metrics, excluding each shared initial
state. Position RMSE averages squared coordinate errors.

| Horizon | Position RMSE (m) | Attitude RMSE (degrees) | Persistence attitude RMSE (degrees) |
| --- | --- | --- | --- |
| 0.1 s | 0.00099 | 0.590 | 1.426 |
| 0.4 s | 0.02689 | 1.800 | 5.865 |
| 0.8 s | 0.10714 | 2.292 | 10.347 |

Values are from `model.horizon_rollouts` and `baseline_metrics.horizon_rollouts`.
The aggregate `score_vs_baseline` is **0.4203**, the geometric mean of the
floored model/persistence ratios for four metrics at three horizons. Lower is
better; this is not a single absolute-error percentage. Velocity error at 0.8 s
is slightly worse than persistence. The belief resolves 11 parameter directions
and remains incomplete.

## Tracking while learning

Both arms follow a straight cruise reference with a small altitude variation.
Both run the learner with an injected initial delay and six missing state
intervals in its telemetry stream; control still receives state observations.
Only the adopting arm prepares and applies replacement controllers. Ordering
alternates between pairs. The mean-based tracking configuration and the
[existing prediction gate](streaming-refinement.md#recorded-experiment) are
shared with the synthetic example, including the explicit allowance for
unresolved parameters. There is no forecast-error recalibration or forgetting.

The [comparison](investigations/cascade-refinement/comparison.json) records:

| Pair | Position RMSE, frozen / adopting (m) | Attitude RMSE, frozen / adopting (degrees) | Whole-tick deadline misses, frozen / adopting | Applied revisions |
| --- | --- | --- | --- | --- |
| 0 | 0.49990 / 0.47539 | 0.63957 / 0.63007 | 62 / 62 | 6 |
| 1 | 0.49738 / 0.47850 | 0.64763 / 0.63132 | 61 / 59 | 6 |

These are `comparisons[].trials.{frozen,adopting}.tracking_rmse`,
`deadline_misses` and `applied_adoptions`. All four trials completed 240 control
intervals with finite states, bounded commands and no worker error. Each learner
processed 17 or 18 of 53 submitted blocks; the other 35 or 36 were dropped from
its bounded pending queue.

The final active models' reserved-recording prediction ratios were 0.41967 and
0.41969, versus 0.42030 initially. This small aggregate improvement coexists with
slightly worse 0.4-second attitude errors. Their reports are collected under
`calibration.final_active_evaluations` in the [audit](investigations/cascade-refinement/audit.json).
Two paired runs do not separate model effects from scheduling variability.
The initial learned model's utility is stronger evidence here than the small
incremental gain from online refinement.

## Interface finding and retained evidence

The first attempt reused the real-flight X8 adapter's `generalized_surface_angle`
semantic. That describes measured actuation, so prediction evaluation worked
but the refinement worker correctly refused to treat it as a command model.
The available direct-command semantics only covered normalized requests.

`surface_angle_command` now explicitly declares requested angles with their
units and bounds. It receives an identity command map, survives serialization,
and works with the existing learner. Measured angles remain non-actionable
without a mapping. The fresh calibration arrays and fitted physical parameters
were identical across this contract correction; no numerical model change was
needed to enable the handoff. Tests cover this distinction and preserve the
synthetic plant's original numerical stepping through the shared plant interface.

The [archive](investigations/cascade-refinement/trials.zip) retains the initial
refused setup, completed trials, calibration data, revision artifacts, journals,
and 124 executed source/specification files with hashes. The
[audit source](investigations/cascade-refinement/audit.source.py) preserves the
saved-array checks for tracking metrics, command bounds, scored-revision
handoffs, interval accounting, calibration roles and model identities. It does
not recompute the held-out forecasts. Later reuse-validation changes are in the
maintained example; the archive preserves the executed snapshot.

The remaining execution limitation is unchanged: the paced fixed-step plant
slows its simulated clock when control runs late. Cascade adds independent
physics but does not by itself test commands arriving late to a continuously
advancing plant. Sensor noise, estimator errors, wind, platform variation and
different flight conditions also remain outside this first experiment.
