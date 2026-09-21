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
| Runtime | Dart takes 59.24 s for 1.2 simulated seconds. V6 quad update p95 is 28.16–29.53 ms against 10 ms observations; fixed-wing p95 is 4.38–4.60 ms against 50 ms. Real-time quad fitting is not qualified. |
| Online fitting | Working **online v6** reduces equal-family velocity/rate error **21.94% against v4** and 78.93% against frozen startup across six known tapes. All 3,137 targets are scored before assimilation. Worst-decile errors improve 22.96%; angular outliers and individual regressions remain. See [online evidence](online-fitting.md). |
| Updates and calibration | Offline updates preserve immutable revisions and development roles. Online sessions retain bounded causal caches. Development-selected calibration does not establish independent coverage; streaming sessions have no calibrated envelope. |
| Scope | Rigid-body motion across recorded quad/fixed-wing configurations. Broad airframe, articulated-system and arbitrary-system readiness remain unproved. No current-learner online closed-loop recovery trial has run. |
| Preservation | Saved baseline contains the unchanged adopted models, recordings and replay evidence: 12,768 flight arrays, 40 Dart trajectories and 4,144 gradients. All 15 package source files remain byte-identical to adopted v6. All **234 tests pass**, and the complete saved baseline replays exactly. |

## Latest completed iteration

**Broader physical prior domains, online v7 — tested, not promoted.** The
[frozen six-tape comparison](online-fit-v7.md) improves equal-family velocity/rate
error 13.30% against adopted v6: fixed wings improve 26.65%, quads worsen 2.47%.
Worst-decile errors improve 15.02%, but orientation and truth-relative
rotation/rate-defect errors worsen 4.79%/1.74%. Both frozen gates fail; all 3,137
transitions and every regression remain reported. V6 stays the sole learner.

The candidate uses full supported-motion bounds and completed-cache command
maxima in the same quadratic prior. It changes no inference, initialization,
model head, vehicle-specific rule, user option or fitting budget. FW81 velocity
RMSE improves 1.244→0.603m/s, but orientation RMSE worsens 17.45%; absolute
fixed-wing velocity/rate errors still exceed kinematic hold.

The [v6 causal trace](online-angular-response.md) reproduces all 450 fixed-wing
forecasts/updates exactly. V7 prospectively captures the same 23 origins. Both
sets pass independent stage, head-increment and derivative audits. Common-domain
comparison shows that stronger regularization does not reliably reduce physical
quadratic curvature at actual failure states: FW81 row 150 rises 26.03→46.03.
Response shifts from motion terms toward command terms and opposing linear
contributions. Its native rate error improves 2.121→0.494rad/s while refined 64
error worsens 1.263→3.708rad/s, exposing numerical cancellation. These observations
do not establish optimizer failure or convergence.

Candidate source and sealed evidence remain reproducible in Git/artifacts.
Independent verification authenticates all 3,137 pairs, 9,411 journal events and
23 captures with zero model/optimizer calls. Restored v6 passes 234 tests; the
saved offline/Dart baseline remains exact. Quad real-time fitting and
current-learner online closed-loop recovery remain unqualified.

## Next scientific gap

**Causal optimizer effectiveness under the physical curvature prior.** Freeze
a read-only audit of retained data/prior gradients, the four-PCG residual versus
a bounded more accurate solve of the same local linear system, forecast trust
shrink and exact objective gain. Distinguish an inadequately solved update from
an objective that permits inaccurate, cancelling dynamics before selecting
another penalty or model change. No additional candidate has been fitted.

Retain all six tapes, velocity/rate, orientation, tails, kinematic and timing
comparisons; adopted v6 remains the next candidate's reference. Known-tape
improvement does not qualify blind generalization or calibrated uncertainty.
