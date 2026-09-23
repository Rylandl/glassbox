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

The Throw demo must learn from a fresh start, without pretraining. No fleet-trained
feature extractor, previously fitted dynamics, learned class prior or calibration
from previous flights may initialize the learner. General physical equations,
fixed generic bases and data-independent initialization/regularization are
allowed. Every learned quantity, including feature weights, normalization,
response coefficients and memory parameters, must come from observations causally
available in the current episode. Initialization and history accumulation count
in the data and elapsed-time budget; a fitted prefix is not free pretraining.
Reset learned state between evaluation episodes and never use future observations.
Adapting small variations around a pretrained dynamics core does not meet this
requirement. Existing offline fit/predict/update remains a workflow of the same
learner; its results do not qualify cold-start Throw performance.

The model is the product. Its public workflow is `fit(recordings)`,
`model.predict(...)` and `model.update(recordings)`, with fingerprinted save/load
and immutable revisions. A session may collect contiguous observations for the
same dynamics formulation without completing a fit at every observation. Its
mutable fitting state is separate from saved immutable revisions. A background
fit may publish new immutable revisions intermittently. Prediction uses a
published revision;
publication time and the causal data available at publication are part of
fresh-start evaluation. Other projects own controllers, planners and simulators.
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

On 2026-09-23 the user clarified that per-observation fitting time on the
current CPU is not an architecture gate. Prefer the model with stronger causal
full-state and command-response predictions, even if fitting belongs in a
background job that periodically publishes revisions. Report fit time and
publication cadence, and assess Throw by what model was actually available at
each point in the episode; do not grant a freshly fitted prefix zero elapsed
time. Predictor cost and consumer-specific deadlines remain measurements.

Future online architecture evaluations freeze and report per-family 250 ms
forecast flags, with velocity and body-rate components shown separately, alongside
aggregate and one-step scores. Preserve physical errors for each recording and
historical flags. A combined metric must not hide a large angular recurrence loss;
these flags still inform an overall engineering decision rather than make every
isolated regression a veto.

## Definition of done

| Criterion | Required evidence |
| --- | --- |
| One recipe | One maintained learner and fit/predict/update contract, with no consumer tuning menu or vehicle-family dispatch. |
| Accuracy | Low held-out forecast and command-response residuals across configurations and conditions, in physical units and across supported horizons. Aggregate weights and regression limits are declared before measurement. |
| Usability | Self-contained revisions; clear signals, timing and history contract; reproducible predictions and usable derivatives; independent consumers use public interfaces. |
| Capability | Analytic mechanics, delayed/hidden response, causality, variable input dimension and numerical derivatives have meaningful regression tests. Broader system classes require their own evidence. |
| Consumer performance | Throw requires identification from the current episode without pretraining, with initialization data/time counted. Dart meets separately declared task limits. Controller outcomes supplement direct prediction/response evidence. |
| Updates | Immutable revisions preserve recording roles and fit budget. Improvement and regressions are measured on fresh recordings; live controller adoption requires separate evidence. |
| Error evidence | Error envelopes identify calibration data and horizon. Held-out coverage is measured; absent evidence stays absent. |
| Lean | Only the learner, necessary recording interfaces, small evaluation tools, active tests and evidence needed for current work remain. |

## Iteration rules

1. Read [status.md](status.md), choose one named gap, and work in an isolated
   worktree. User direction can change priorities or adoption policy.
2. Reuse a frozen measurement contract and runner for quick candidate screens;
   do not rebuild evaluation plumbing for each model idea. Label exploratory
   smoke runs as such and record exact source hashes. Commit completed
   scientific code before a full result is used for an engineering decision.
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
