# Current state

Updated 2026-09-21. Read [the charter](charter.md) first.

The supported shared-physics learner is the **adopted single implementation**.
It learns each configuration separately from observations and issued commands,
without vehicle-family, layout, mass or inertia inputs. The model is the product;
controllers and simulators remain external.

| Area | Evidence and remaining gap |
| --- | --- |
| Recipe | Shared gravity, rigid-body kinematics and coordinate transforms; learned command response, effective accelerations and memory. Arbitrary ordered command dimensions; no platform dispatch. |
| Offline accuracy | Original shared physics reduced matched aggregate forecast/response errors 43.95%/22.12% versus the retired model. The supported map preserves local response on the inspected Crazyflow/Cascade cohorts. Cascade wind velocity RMSE remains 0.426–0.434 m/s at 250 ms; known Dart 1.2 s response errors remain 3.942 m/s and 9.601 rad/s. |
| Dart | The unchanged supported model reaches **0.720 mm** nominal contact miss through an external radius-aware objective and 10 ms feedback. Fine float64 replays reach 0.756 / 0.755 mm and converge within 0.152 micrometers / 0.228 microseconds. One known task/seed; neighborhood reliability remains open. See [precision evidence](dart-precision.md). |
| Runtime | Dart takes 59.24 s for 1.2 simulated seconds. Working online quad update p95 is 28.0–28.6 ms against 10 ms observations; fixed-wing p95 is 3.92–3.97 ms against 50 ms. Real-time quad fitting is not qualified. |
| Online fitting | Working **online v4** reduces equal-family velocity/rate error 40.89% against v2 and 73.01% against frozen startup across six known tapes. All 3,137 targets are scored before assimilation. Large fixed-wing errors and quad-115 orientation regression remain. See [online evidence](online-fitting.md). |
| Updates and calibration | Offline updates preserve immutable revisions and development roles. Online sessions retain bounded causal caches. Development-selected calibration does not establish independent coverage; streaming sessions have no calibrated envelope. |
| Scope | Rigid-body motion across recorded quad/fixed-wing configurations. Broad airframe, articulated-system and arbitrary-system readiness remain unproved. No current-learner online closed-loop recovery trial has run. |
| Preservation | Saved baseline contains the unchanged adopted models, recordings and replay evidence: 12,768 flight arrays, 40 Dart trajectories and 4,144 gradients. Current package source remains identical to the working v4 evaluation source. All **187 tests pass**. |

## Latest completed investigation

**Causal fixed-wing forecast failures.** The frozen diagnostic replay exactly
reproduces all **450 fixed-wing predictions and model transitions**. Twenty
pre-assimilation snapshots bind full optimizer/cache state to the original tape.
All saved diagnostics reproduce without optimizer updates; an independent audit
checks 10,440 integration-stage states. See [the diagnosis](online-causal-trace.md)
and [artifact identities](online-causal-trace.json).

At all four selected worst-error events, the immediately preceding accepted
update improves the forecast when compared on identical current inputs. Smaller
integration steps reduce FW81's velocity spike from 31.14 to 7.65 m/s and rate
spike from 17.19 to 8.74 rad/s, but increase FW80's rate error from 8.27 to
11.20 rad/s. Incorrect learned response remains after refinement.

Large quadratic command and delayed-linear contributions appear before
integration. Predicted attitude/rate changes can amplify them further through
quadratic feedback. In one severe step, tiny startup variation scales a known
unit gravity-direction coordinate to about 37, whose square generates enormous
learned acceleration. Support saturation and novelty occur at ordinary controls
too; neither alone diagnoses a failure.

The learner is unchanged. Earlier online v3 and v5 failures remain recorded;
v5's solver-consistency penalty improved aggregate error only 4.66%, with worse
fixed-wing angular errors and higher cost, and was not adopted. These historical
verdicts and reproduction commits are in [the online report](online-fitting.md).

## Next scientific gap

**Constrain unsupported nonlinear response during short online fits.** Use the
captured causal states to separate current-command effects from history/filter
context and assess which dominant quadratic directions the retained data
actually constrain. Then freeze one generic correction against working v4.
Known gravity-direction geometry supplies a system-independent scale; command,
rate and attitude curvature should not acquire large unsupported responses from
a narrow startup sample. No vehicle label, catalog or consumer tuning option.

Keep all velocity/rate, orientation, tail and timing comparisons. Finer inference
or a stronger consistency penalty alone does not address the remaining learned
field error. Latency and live control are separate qualifications. No successor
candidate has been fitted.
