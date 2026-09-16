# Forecast accuracy versus ordinary tracking

The unchanged generic Glassbox learner does **not** meet the first ordinary
Cascade tracking requirement. With the same controller, the simulator-equation
predictor passes all three trials; the learned predictor diverges in all three.
Its reserved-recording 250 ms errors are only **0.149 m/s velocity RMSE** and
**0.121 rad/s body-rate RMSE**. Those scores do not establish that an optimizer
can use the model successfully.

This experiment also measures a conditional accuracy requirement. A prescribed
persistent vertical-velocity forecast error of **0.3 m/s at 250 ms** passes all
three trials, whereas **0.6 m/s** fails all three. The same 0.3 m/s RMS error
with alternating sign produces much smaller tracking error. Error direction,
time dependence, and response to commands matter alongside average magnitude.

The evidence bundle contains the
frozen protocol, all 33 outcomes, diagnostics, source snapshots, and replay
checks. This extends the earlier [accuracy-budget discussion](accuracy-requirements.md)
with actual offline simulated control trials. It does not change the
[opinionated learner](learner.md) or add consumer tuning options.


The left panel plots velocity error only; other output errors and command
derivatives differ between predictor families. Bars span the three trial
fractions, not confidence intervals. The right panel shows one declared seed;
the tables below include all three. Shading marks the excluded settling period.

## Declared task and comparison

The plant is Cascade's published Skywalker X8 model at nominal 18 m/s cruise
and 100 m altitude. The reference has lateral position `sin(0.35t)` metres and
altitude `100 + 0.75 sin(0.3t)` metres. Each trial runs for 16 simulated seconds.
A pass requires both absolute lateral and altitude errors to be at most
**0.5 m in at least 95% of the 20 Hz samples at t >= 2 s**, with no terminated
trial. There are 281 scored samples per complete trial. This is a provisional
application requirement, not a learned or universal flight tolerance.

Seeds 101, 102, and 103 perturb initial lateral/vertical position by up to
0.15 m, lateral/vertical velocity by 0.05 m/s, attitude tangent coordinates by
0.01 rad, and body-rate components by 0.02 rad/s. They are the only introduced
disturbances. Wind is zero; sensing is simulator truth; there is no added
measurement noise, command transport delay, or ongoing random disturbance.
Two actual trim-command intervals establish history before the trial begins.

The controller was developed with the oracle on separate seed 0. That first
development trial passed, after which the controller, predictor arms, and
evaluation seeds were frozen before fitting and comparison. No controller or
learner was tuned in response to evaluation failures.

Every arm uses one bounded predictive controller, the same reference, initial
conditions, command bounds, and timing. It optimizes one constant three-channel
command over five 50 ms steps using three damped Gauss–Newton iterations and
fixed backtracking. It penalizes lateral/vertical tracking, forward velocity,
body rates, rotation-matrix deviation from trim, and command magnitude/change.
The previous command initializes each solve. The full fixed coefficients are
in the evaluation plan.

Positions are reconstructed from predicted world velocities by the same
trapezoidal integration in all arms. A 1.5 s position/velocity lookahead term
in the cost provides anticipatory feedback; **the dynamics forecast remains
250 ms**, not 1.5 s. This is an offline research controller. Computation is
allowed to take longer than a control interval; no real-time performance has
been demonstrated.

The oracle uses Cascade's public functional equations, the same equilibrium
reset, and causal replay of issued commands. It does not read private running
actuator or separation states from the plant. Its exact internal model and
complete causal memory are advantages over the generic learner's short
observation history. Equality of external measurements does not remove that
model-information advantage.

## Data and unchanged learner

Four fresh 12 s calibration recordings use seeds 0–3. The automatic whole-
recording split assigns 36 s to training and 12 s to development. It retains
384 training windows and 234 development windows. The existing calibration
pilot and trim feedforward use Cascade-specific knowledge; this experiment
does not demonstrate calibration from an unknown airframe without a pilot.
The tracking controller does not use that calibration stabilizer.

Two further 12 s recordings, seeds 80 and 81, are reserved for common forecast
evaluation: 47 origins each, spaced 250 ms apart. They are excluded from fit,
checkpoint selection, and controller development. They are fresh recordings
of the same simulator/configuration, not new-platform or hardware validation.

The learner receives 15 Euclidean channels: world velocity, body angular rate,
and the nine entries of the body-to-world rotation matrix. Inputs are requested
throttle and generalized aileron/elevator angles, not measured actuator states.
The adapter preserves recording identities and declares ordered channels.
Positions are not learner inputs. Predicted matrices are not projected onto
SO(3); physical validity of this unconstrained representation remains a concern.

The fixed `generic-history-v1-prototype` recipe uses 100 ms history, 250 ms
recursive predictions, and the existing affine-plus-MLP model, training budget,
normalization, and automatic checkpoint selection. This fit selects step 1000.
Its recipe and observation provenance are in the
fit report; the saved model's
fingerprint is in model-identity.json. No aircraft
equations, new model structures, or recipe settings were added to the learner.

## Measured accuracy and outcomes

Errors below are physical vector RMSE at the final 250 ms horizon on the
same 94 reserved forecast windows, pooled equally across the two recordings.
They are not normalized training loss, worst-case bounds, or on-policy errors.

| Predictor | Velocity RMSE, m/s | Body-rate RMSE, rad/s | Trials passing |
| --- | ---: | ---: | ---: |
| Oracle | approximately 0 | approximately 0 | 3/3 |
| 75% oracle + 25% learned forecast | 0.0372 | 0.0303 | 0/3 |
| 50% oracle + 50% learned forecast | 0.0744 | 0.0605 | 0/3 |
| 25% oracle + 75% learned forecast | 0.1116 | 0.0908 | 0/3 |
| Learned | 0.1488 | 0.1211 | 0/3 |

These mixtures combine complete output forecasts. They are diagnostic
predictors, not physically consistent models, and they change output
derivatives as well as prediction values. The failure boundary below a 25%
learned contribution has not been localized.

Oracle post-settling lateral RMSE is 0.0051–0.0121 m and altitude RMSE is
0.0015–0.0084 m. Pure learned trials terminate after 4.05, 5.05, and 5.20 s;
the nine intermediate-mixture trials also terminate. Termination occurs for
nonfinite values or the declared divergence limits: 20 m altitude deviation,
45 m/s speed, or 5 rad/s body-rate norm. Actual saved states in these trials
remain finite but cross a divergence limit.

Every unexecuted interval counts as outside tolerance. Failed trials are not
discarded. Their full-duration tracking RMSE is unbounded under this accounting
and is serialized as `null`, not zero. Saved one-step errors on completed
intervals describe each arm's own evolving trajectory; they cannot be treated
as another matched comparison once the trajectories have diverged.

### A controlled error family brackets one conditional requirement

For these arms, the oracle forecast's vertical-velocity channel receives an
additive error ramp from one fifth of the listed magnitude at 50 ms to the
full magnitude at 250 ms. Rate and rotation predictions remain exact. The
perturbation does not change the forecast's command Jacobian. A persistent
error is always positive; an alternating error flips sign at every 50 ms
control update. These are prescribed prediction errors, not plant disturbances.

| 250 ms error, m/s | Persistent: altitude RMSE range, m | Persistent: passes | Alternating: altitude RMSE range, m | Alternating: passes |
| --- | ---: | ---: | ---: | ---: |
| 0.1 | 0.139–0.142 | 3/3 | 0.0021–0.0088 | 3/3 |
| 0.3 | 0.415–0.419 | 3/3 | 0.0060–0.0119 | 3/3 |
| 0.6 | 0.831–0.834 | 0/3 | 0.150–5.033 | 1/3 |

For this task, controller, and persistent error shape, 0.3 m/s passes and
0.6 m/s fails. The corresponding first-step errors are 0.06 and 0.12 m/s.
Those tested points bracket an observed change in outcome; they do not prove
a monotonic threshold, identify an exact critical value, or establish a bound
for another error direction or model.

The 0.3 m/s persistent and alternating arms have equal forecast-error RMS and
identical command Jacobians. Their different tracking results isolate an
effect of temporal structure in this prescribed family. Alternation is not
universally harmless: at 0.6 m/s two trials fail, including one with substantial
lateral error. No single velocity-RMSE cutoff separates all successes and
failures in this experiment.

## What the failure diagnosis establishes

A separate post-result diagnostic plan
was recorded before inspecting command coverage and local command derivatives.
These are explanatory checks, not a new model selection or control comparison.

At each shared initial history and the same trim command, we differentiate the
forecast with respect to a constant future command. Input coordinates are
normalized by the application command half-ranges `[0.5, 0.35, 0.35]`. The
relative error is `norm(J_learned - J_oracle) / norm(J_oracle)`, using a
Frobenius norm separately for velocity and rate outputs.

| Forecast horizon | Velocity Jacobian relative error across seeds | Body-rate Jacobian relative error across seeds |
| --- | ---: | ---: |
| 50 ms | 3.51–6.50 | 1.90–2.25 |
| 250 ms | 0.638–0.929 | 0.634–0.669 |

These are large errors in the local response to changing a command, not
percentages of observed velocity or attitude. A model can fit trajectories
from a calibration pilot while learning a poor response to alternative actions
that an optimizer considers. This evidence is consistent with that failure
mode, but three derivative probes do not identify its unique cause.

Completed pure learned trials place 93–100% of commands on an application
command-box boundary. Between 33% and 97% leave at least one per-channel range
observed among the retained training windows' future inputs. Oracle fractions
outside those training ranges are only 2.2–2.8%. The full aileron range already
appears in training; limited throttle coverage is one concrete gap.

However, the learned first command lies inside all three marginal training
ranges for every seed, while each oracle first command leaves the throttle
range and still tracks successfully. Marginal ranges are neither joint
state/history/action support nor calibrated uncertainty. These results do not
justify attributing failure solely to unsupported commands or simply declaring
that more of the same calibration data will fix it. Insufficient excitation,
short memory, model/representation restrictions, optimization behavior, and
later distribution shift remain competing or interacting explanations.

## Implication for Glassbox

The next useful qualification target is **response to alternative commands
at relevant histories**, alongside forecast accuracy and ordinary tracking.
Keep one fit/predict/update interface. A diagnostic should expose the evidence
supporting those responses and the consumer's declared task tolerance, rather
than asking users to choose another collection of regularizers and thresholds.

A follow-up experiment should separate the command-response question from
trajectory drift: reserve simulated local command variations at shared
histories, compare predicted and actual responses, and measure whether those
checks anticipate the existing tracking failures. That would test whether a
generic validation signal adds value before changing model capacity or
collecting additional calibration. The current work supplies the frozen
baseline; it does not yet implement such a qualification gate.

## Validation and limits

All 33 planned trials are included. The audit reconstructs 618 cached training
and development windows, both reserved recordings and all 22 forecast arms,
and every saved trial command trajectory. Learned predictions are independently
replayed with NumPy. It checks scoring, truncation, command bounds, cost descent,
and timing accounting. Across 2,946 numerical checks, the maximum absolute
difference is **8.85e-13**. Public Cascade core replay checks implementation
consistency, not independent physical truth; optimizer iterations are not
independently re-solved.

The focused source suite passes **34 tests** in float64; the same 34 pass
against the isolated built wheel in float32. All 81 packaged Python files
match source. Ruff and whitespace checks pass. The complete repository test
suite was not rerun for these research-only additions.

Three initial conditions on one deterministic simulator do not establish a
success probability or transfer to other platforms, flight regimes, sensors,
wind, unknown actuator initialization, or real-time execution. The unchanged
generic candidate remains experimental. This is an observed failure of one
fixed model/controller pairing and calibration procedure, not a demonstration
that generic dynamics learning cannot support control.
