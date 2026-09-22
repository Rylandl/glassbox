# Glassbox: one learned dynamics model

Glassbox learns differentiable system dynamics from recordings. The maintained
implementation shares gravity, rigid-body kinematics and coordinate transforms;
it learns configuration-dependent command response, effective accelerations and
hidden dynamics. One fixed procedure fits each configuration separately. Input
count comes from the data and is not assumed to equal actuator count.

The user supplies observations, issued commands, timestamps/segment boundaries
and interpretable signal units/frames. No vehicle family, mass, inertia, mixer,
actuator layout, control-surface assignment, aerodynamic coefficients or tuning
choices are required. Configuration and command names establish data identity;
they do not select equations. The present formulation covers rigid-body motion,
not arbitrary articulated or deformable systems.

The model is the product. Its public workflow is `fit(recordings)`,
`model.predict(...)` and `model.update(recordings)`, with fingerprinted save/load
and immutable revisions. A bounded `OnlineFit` session assimilates contiguous
observations into that same dynamics formulation; its mutable optimizer state is
separate from saved immutable revisions. Other projects own controllers, planners
and simulators.
Glassbox retains a small generic motion adapter and a saved-evidence verifier;
it does not maintain a competing controller framework or model catalog.

## Adoption

On 2026-09-21 the user adopted the supported shared-physics learner as the sole
implementation and requested removal of superseded models, experiments,
interfaces, tooling and their artifacts. This replaces public v4. It is an
explicit policy decision on known evidence, including the successful Dart task,
not a retrospective change to a frozen experiment's acceptance flags.

Generality has priority over winning every benchmark cell. Known wind errors,
response-tail losses, long-horizon errors and calibration limits remain measured
improvement work. One nominal control success does not establish a reliability
percentage, real-time operation or arbitrary-system readiness.

On 2026-09-22 the user clarified that evaluation must inform an overall engineering
decision, not make every regression a veto. Preserve and report frozen outcomes,
but weigh generality, accuracy, runtime and maintainability together. Broken
contracts, invalid numerical behavior and substantial consistent capability losses
need resolution; an isolated benchmark loss can become follow-up work. Distinguish
architecture changes from fitting provenance before attributing a regression.

## Definition of done

| Criterion | Required evidence |
| --- | --- |
| One recipe | One maintained learner and fit/predict/update contract, with no consumer tuning menu or vehicle-family dispatch. |
| Accuracy | Low held-out forecast and command-response residuals across configurations and conditions, in physical units and across supported horizons. Aggregate weights and regression limits are declared before measurement. |
| Usability | Self-contained revisions; clear signals, timing and history contract; reproducible predictions and usable derivatives; independent consumers use public interfaces. |
| Capability | Analytic mechanics, delayed/hidden response, causality, variable input dimension and numerical derivatives have meaningful regression tests. Broader system classes require their own evidence. |
| Consumer performance | Dart meets separately declared task limits. Controller outcomes supplement direct prediction/response evidence. |
| Updates | Immutable revisions preserve recording roles and fit budget. Improvement and regressions are measured on fresh recordings; live controller adoption requires separate evidence. |
| Error evidence | Error envelopes identify calibration data and horizon. Held-out coverage is measured; absent evidence stays absent. |
| Lean | Only the learner, necessary recording interfaces, small evaluation tools, active tests and evidence needed for current work remain. |

## Iteration rules

1. Read [status.md](status.md), choose one named gap, and work in an isolated
   worktree. User direction can change priorities or adoption policy.
2. Freeze the measurement and acceptance contract before implementation or
   fitting. Commit completed scientific code before executing an experiment.
3. Prefer shared physical structure. Never introduce a catalog, system-specific
   branch, hidden simulator information or consumer tuning option.
4. Keep fitting, derivative fidelity, calibration and task success as distinct
   claims. Report all attempted conditions, missing truth and failures.
5. Verify changed behavior from saved data without refitting and reject altered
   artifacts. Refactoring must preserve the claimed numerical behavior under
   the declared runtime. Use small analytic fixtures for ordinary tests.
6. Delete obsolete code rather than leave deprecated alternatives. Preserve
   the current reproducible evidence before removing redundant artifacts;
   committed history holds superseded source and reports.
7. Update status with the actual result, remaining gap and scope. Routine work
   within this charter needs no repeated permission question.
