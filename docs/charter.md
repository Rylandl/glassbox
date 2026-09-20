# Charter: one generic dynamics learner

Glassbox aims to learn accurate, differentiable dynamics across vehicle types and
configurations from recordings, and produce improved revisions as new recordings
arrive. The immediate product milestone is running Dart's precise pose/contact
task with that general learner. One learning procedure fits each configuration;
one set of fitted weights need not describe every vehicle. System coverage and
revision improvements must be demonstrated by evaluation. The caller supplies
observed signals and commands, their meanings, units and coordinate frames,
timing, and recording boundaries. One recipe, one module, three calls: `fit`,
`predict`, `update`. No vehicle-family selector, model catalog, consumer tuning
options or branches selecting a vehicle's equations. Shared physical structure
is allowed; learning known mechanics from scratch is not a product requirement.

The product is an accurate learned dynamics model that other applications can
inspect, save, differentiate and use through a clear signal and timing contract.
Applications choose their own controller, planner, estimator or analysis tool.
A Glassbox controller is an optional reference consumer and a useful demonstration
of model quality. Using it is not a condition of using the learner.

The goal is to make hand-written system dynamics unnecessary where the learned
model meets the consumer's accuracy and evidence requirements. Generality takes
priority over winning every benchmark case.
The generic learner is the adopted development baseline; documented losses
on individual systems are improvement work, not an automatic veto on that
choice. [`status.md`](status.md) holds the current gap against the
definition of done below and is the only page a new session needs to read
after this one. Git holds the history; it is not carried forward as narrative.

## Definition of done

Adopting the approach and completing the project are separate decisions.
Completion requires the following, measured on held-out evidence under
protocols frozen and committed before each implementation change.

| Criterion | Done means |
| --- | --- |
| One recipe | A single versioned recipe in a single module. `fit(recordings)`, `model.predict(...)`, `model.update(recordings)`. The consumer contract has no options. |
| Accuracy | Low held-out prediction and command-response residuals across systems from one platform-independent recipe. Report errors across supported horizons in physical units, per-system gains and losses, and conditions tested. Improvement is measured against the adopted generic baseline; structured and simple models provide context, not an accuracy ceiling or a completion criterion. Application adequacy requires separately declared tolerances. Superiority on every corpus is not required. Derivatives existing computationally does not establish that they match physical responses. |
| Model usability | A self-contained saved revision with a documented signal, units, frames, timing, history and horizon contract; reproducible batched predictions and usable derivatives; measured error evidence and explicit limitations. Consumers use public interfaces without importing Glassbox controller internals. A third-party controller or analysis integration demonstrates that boundary. |
| Capability | The synthetic suite, including delayed inputs and hidden state, passes its absolute caps. It is a fast regression guard, not a place to win. |
| Reference control | The general learner powers Dart's pose/contact task through Dart's controller under a separately declared evaluation. Each controller is qualified for its own task; one universal controller is not a requirement for the model product. Control success does not replace direct model-accuracy evidence. |
| Live improvement | The learner produces immutable revisions within a bounded update budget, with improvement and regression criteria frozen before evaluation on untouched recordings and response queries. Applications decide when to adopt revisions. Any claim of safe live controller swapping additionally requires its own downstream no-regression trial. |
| Evidence | Predictions expose measured error envelopes and their calibration provenance; coverage on held-out recordings and declared shifts lands in a predeclared band. Any downstream use of that evidence is evaluated separately. |
| Lean | The structured models, the belief format, and obsolete research scripts are deleted. What remains is the learner, model artifacts and interfaces, the harness, telemetry adapters, and optional downstream consumers. |

## Product priority, 2026-09-18

The user's priority is accurate, usable learned system dynamics. A general
controller remains desirable as an optional consumer. JSBSim is a proposed
benchmark for model breadth, with flying demonstrations as additional evidence.
This is a prospective change in product priorities, not a newly passed experiment.
All frozen protocols, historical control/update failures and qualification flags
retain their original meaning.

## Evaluation priority, 2026-09-18

The user has redirected the next evaluation work to Crazyflow and Cascade.
Develop controlled flight-condition environments for these two simulators,
measure the generic learner against structured and simple references, and retain
the concrete prediction failures reported by the Dart consumer. This provides a
more focused setting for improving the learner than expanding JSBSim setup work.
JSBSim results remain valid diagnostic evidence; further breadth work is deferred.
This changes evaluation priority, not the one generic learner or its consumer
contract, and does not qualify any new model or controller.

The user's subsequent clarification makes reducing held-out residuals the
primary objective of this benchmark. Track forecast and command-response
errors in physical units across horizons and conditions, including large-error
cases. Beating a structured reference does not finish the work, and reducing
residuals remains useful even while that reference is better. Use matched
comparisons with the adopted generic learner to establish progress; freeze
aggregate weights and regression limits before each experiment. Keep missing
truth and failed conditions visible so a smaller or easier evaluated subset
cannot masquerade as lower error. No absolute adequacy threshold is inferred
from comparator performance or introduced retrospectively.

## Vehicle generality and Dart milestone, 2026-09-20

The user clarifies that the broad goal is running Dart with the general model
across vehicle types and configurations. Shared physical knowledge is welcome;
a catalog of vehicle-specific models is not. This supersedes interpreting
generality as an obligation to learn arbitrary signal dynamics without mechanical
structure. The present v4 recipe remains the adopted baseline until a successor
is evaluated; this policy change does not qualify a new model or controller.

Dart's structured-model success establishes a useful task/data reference.
The next work measures current v4 in that setting and accounts explicitly for
its history, state representation and shorter supported horizon. Shared mechanics
with learned configuration-dependent dynamics is the leading architecture
hypothesis. Ordinary-recording evidence and the Dart task take priority over
collecting privileged simulator response pairs. Crazyflow and Cascade retain
their role as cross-vehicle forecast and response benchmarks. Historical protocols,
results and qualification boundaries keep their original meaning.

## Rules

- Each iteration's measurements and promotion criteria are frozen and
  committed before fitting or new trials. Historical protocols and results
  stay immutable. An explicit change in product priorities is recorded as a
  policy decision on known evidence, never as a newly passed experiment.
- One change per iteration, addressing one named gap in the status table.
  Menus of context lengths, widths, optimizers, or seeds fail review. A new
  mechanism needs a named failure it addresses.
- Nothing is deprecated; it is deleted. Nothing is preserved for its own sake.
  Old results are not evidence for new code.
- Use shared structure: causality, memory, smoothness, observation noise,
  coordinate transformations, gravity and rigid-body kinematics. Learn the
  configuration-dependent command response, forces or effective accelerations,
  and hidden dynamics from recordings. Do not select hand-written multirotor,
  fixed-wing or other vehicle-family dynamics, assume a fixed mixer/actuator
  layout, or import simulator parameters into the learner. More general
  mechanical configurations require demonstrated coverage, not an assertion
  that a rigid-body approximation already represents all vehicles.
- Synthetic results never count as platform readiness. Fit quality, error
  calibration, and control adequacy are separate claims with separate
  evidence.
- Primary model qualification does not require every application to use the
  reference controller. Evaluate forecast accuracy, command-response fidelity,
  uncertainty and interface usability directly. Task objectives, scheduling and
  revision adoption belong to the consumer; task tolerances belong to its
  separately declared evaluation.
- Generality and broad empirical advantage can justify adoption despite
  localized losses. The earlier v3 adoption explicitly accepted its measured
  ARP deficit. Future improvements are judged against the generic baseline;
  the structured models are benchmarks, not an incumbent entitled to win.
  Report gain and loss magnitudes, breadth across systems, and task adequacy
  separately. A win count alone is not a permanent promotion rule. Choose
  any aggregate weights and unacceptable-regression limits before fitting,
  rather than requiring every metric to improve on every case.
- Every accepted iteration replays from saved artifacts without refitting, and
  an altered artifact is rejected by that replay.

## Process

Each iteration runs in its own worktree: measure the largest gap in
[`status.md`](status.md), change one thing, run the harness, keep the change
according to the iteration's predeclared promotion criteria, and write one
paragraph saying what changed and what the gap is now. Historical gate
outputs remain visible; they do not silently reimpose an all-case adoption
veto. Evidence-integrity checks remain mandatory. Review verifies by rerunning
the harness, injecting defects, and checking the consumer contract, not by
reading. On merge, the status table is updated. The loop ends when every row
meets its target.
