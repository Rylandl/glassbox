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
| Preservation | Saved baseline contains the unchanged adopted models, recordings and replay evidence: 12,768 flight arrays, 40 Dart trajectories and 4,144 gradients. Shared dynamics and offline source remain byte-identical to v4; only the online fitter changed. All **204 tests pass**, and the complete saved baseline replays exactly. |

## Latest completed iteration

**Physical quadratic-head curvature prior, online v6.** Adopted as the sole
maintained streaming fitter after the prospectively frozen primary and separate
robustness gates passed. Both families improve: quads 13.69%, fixed wings 29.41%.
No case or failure was excluded. [The result](online-fitting.md) records the
frozen protocol, source and sealed artifact authority.

The prior uses measured motion/issued-command scales and known unit-gravity
geometry, with four preconditioned curvature iterations and no extra rollout.
It introduces no vehicle-specific rule or user option. Initialization, inference,
immutable models and the offline learner remain unchanged. Independent saved-data
verification checks all 3,137 causal forecasts, exact v4 pairing and endpoint
objectives without fitting or model calls. Historical v3/v5 failures remain.

FW81 velocity RMSE falls 2.891→1.244 m/s and its maximum error 31.139→7.704 m/s.
But its maximum rate error rises 17.194→18.227 rad/s, orientation RMSE worsens 7.16%,
and fixed-wing orientation/rotation-rate family scores worsen 2.13%/6.49%.
Quad115 velocity and FW80 velocity also regress 7.57%/3.32%.
Absolute fixed-wing velocity/rate errors remain worse than kinematic hold.

The accompanying [no-fit diagnosis](online-response-support.md) separates large
command response from context-dependent numerical amplification. Its 37 retained
origin features per snapshot do not uniquely separate affine and quadratic
heads; this does not establish nonidentifiability of the full rollout objective.
Twenty points pass independent physical-coordinate and nullspace audits.

## Next scientific gap

**Residual fixed-wing angular response and forecast outliers.** Recover v6's
actual pre-assimilation states at the remaining angular extremes, distinguish
incorrect learned acceleration from integration error, and inspect delayed-linear
and recurrent response after explicit curvature suppression. Freeze that
measurement before replay, then choose one correction supported by the findings.

Retain velocity/rate, orientation, tail, kinematic and timing comparisons on all
six tapes; use adopted v6 as the next candidate's primary reference. Known-tape
improvement does not qualify blind generalization, calibrated uncertainty or
current-learner closed-loop recovery. Quad latency remains a separate gap.
