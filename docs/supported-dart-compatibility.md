# Supported shared physics completes the nominal Dart task

The fixed supported shared-physics revision **passes both frozen numerical
compatibility and task success** through Dart's unchanged controller. It reaches
the target with a **6.19 mm miss**, completing the previously failing nominal
consumer task. The original shared model stopped at 81 intervals with a
nonfinite planning objective/gradient; this revision executes all 120 intervals
and contacts the target before the 1.2-second deadline. No fitting, support
preparation, parameter change, controller change or extra nominal trial occurred.

| Contact measurement | Supported revision | Frozen task limit |
| --- | ---: | ---: |
| Miss distance | 6.190 mm | ≤20 mm |
| Contact top-axis error (yaw free) | 1.330° | ≤5° |
| Incoming normal speed | 1.490 m/s | 0.2–2.5 m/s |
| Tangential speed | 0.1861 m/s | ≤0.2 m/s |
| First contact time | 1.194419 s | ≤1.2 s |

All 4,144 actual objective/gradient evaluations, 40 selected full-horizon means
and 120 native simulator steps return finite arrays. Every durable call closes;
there are no exceptions. All 40 optimizer attempts return plans; 26 converge
and 14 reach the unchanged 100-iteration limit. Nonconvergence was always a
separate readout, and neither it nor intermediate nonfinite line searches
introduced an additional stopping rule.

The historical causal-history structured comparator achieved an 8.17 mm miss,
38/40 converged solves and 0.0526 m/s tangential contact speed. Its result is
previously inspected context, not a rerun or a new superiority test. The supported
revision has a smaller miss here but a narrower tangential margin and fewer
converged solves. One outcome per model cannot establish a success rate.

## What this demonstrates

One generic learned formulation, with weights fitted for this configuration,
now supplies dynamics that a separate controller can use to complete the
nominal task. The model receives canonical observed motion and issued-command
history. Simulator actuator state remains solely on the truth side. The learner
uses shared rigid-body structure; it receives no vehicle family, physical
parameters, mixer, actuator layout or controller-specific dynamics equations.

The trial uses the previously inspected target, actual 50-step hover history,
and original saved 120-command seed. Dart replans every 30 ms, recursively
predicts 1.2 seconds on every solve, and retains its original objective, bounds
and L-BFGS-B options. The seed is an existing optimized command tape. This is
an offline instrumented trial: its control stage takes 36.69 seconds on the
bound CPU runtime for 1.2 simulated seconds. It does not qualify real-time use.

Feedback and a useful starting plan matter. This result does not establish
accurate open-loop forecasts throughout 1.2 seconds, physical derivative
fidelity, calibration, neighborhood reliability or arbitrary-vehicle readiness.
The frozen long-response errors and Cascade wind losses remain improvement
work. The 0.0139 m/s tangential-speed margin, 14 budget-limited solves and large
but finite gradients make nearby-task robustness a concrete next question.

## Descriptive prediction errors on the executed path

A separate saved-array analysis compares each selected plan's next three states
with the actual simulator states reached under those same issued commands. At
30 ms, across 40 replans, component RMSE is 0.269 mm in position, 0.0174 m/s in
velocity and 0.0868 rad/s in body angular rate. Maximum position-vector error is
2.18 mm; maximum velocity-vector error is 0.140 m/s. These are **posthoc,
closed-loop visited-path diagnostics**, not a frozen held-out accuracy gate.
They help explain successful frequent replanning without resolving known longer
forecast and command-response errors. The [saved calculation and results](harness/supported-dart-compatibility-v1-result.json)
are separately hashed and make no new model or simulator calls. Every optimizer
solve decreases its initial objective, but the maximum recorded gradient norm
is 88 million. Finiteness alone does not establish good conditioning.

## Evidence and replay

The [protocol](harness/supported-dart-compatibility-v1.json) was committed as
`cd7ce3b` before implementation. The full harness and 48 passing synthetic,
consumer-boundary and evidence-integrity tests were committed as
`ba2be053cff8625875687b0a47c9b1da5c016402` before any scientific call; Ruff passed.
The immutable implementation remains in the `codex/supported-dart-compatibility`
worktree. The source bundle includes its pinned supported core and old control
reference. The model archive, source bytes, controller, simulator and runtime
are bound in the [result record](harness/supported-dart-compatibility-v1-result.json).

Every actual objective/gradient callback was recorded without changing its
arguments or returns. The selected-mean wrapper was installed after planner
construction, preserving the objective's original closure and recording its
full 121-state output before validation. Native returned states were captured
before finite checks. Inputs and records were durably published before calls;
no scientific prefix was restarted.

Exactly one replay reproduced all 4,304 trial calls and the 50 imported native
prelude steps, including exact dtype/shape/value equality, causal histories,
command seeds, contact and provisional readout. It invoked no optimizer.
An independent NumPy audit reconstructed observations, command expansion,
seed shifting, selected/executed prefixes, native continuity, numerical counts
and quaternion/contact geometry. Contact/check/interval booleans agree exactly;
scalar differences lie within the prospectively frozen arithmetic budgets.
These budgets never widen task thresholds.

All four numeric alterations were rejected after positive controls: model
weights; an actual gradient proposal with repaired local hashes; a native input
state with repaired local hashes; and the numerical success readout. External
stage authority and independent causal reconstruction both reject the altered
proposal and native state. The final qualified readout is independently checked.
All stages close under the original committed source, with no harness correction.

Both scoped milestones pass. Public promotion and overall research acceptance
remain **false**, as frozen: the public recipe remains
`generic-memory-v4-prototype`. The next iteration should freeze a nearby-start
and target evaluation for the same revision/controller, report every trial,
and measure prediction errors along the executed paths alongside task outcomes.
No neighborhood protocol has been frozen or run yet.
