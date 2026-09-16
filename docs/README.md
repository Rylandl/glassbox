# Glassbox documentation

Start with the repository [README](../README.md) for the identification workflow.

- [Scope](scope.md): the core workflow and design decisions.
- [Charter](charter.md) and [status](status.md): the one generic learner we
  are building, its definition of done, the rules, and the current gap.
- [The generic learner](learner.md): the consumer workflow, the recipe
  constants, the memory contract, and how to run and verify the harness.
- [Platform onboarding interfaces](platform-onboarding.md): the proposed
  consumer workflow, current API friction, and acceptance criteria for changes.
- [Validation](validation.md): recorded prediction and control comparisons.

## Concepts

- [Dynamics beliefs](concepts/dynamics-beliefs.md): the product object, its
  information state and forecast-error envelope, and the recursive `absorb`
  update.
- [Nonlinear model-predictive control](concepts/nmpc.md): the plan-model seam,
  the bounded solver, the objective and its two robustness terms, the control
  loop and link contract, and the flight supervisor that is the bounded
  command's last check.
- [Bootstrap identification](concepts/bootstrap-identification.md): the
  no-prior contract for local authority identification, in
  `glassbox.control.identifier`.

## Guides

- [Platform onboarding](guides/platform-onboarding.md): a runnable in-memory
  fit, evaluate, predict, and update workflow for both supported families,
  plus recorded-telemetry refinement and a learner beside simulated tracking.
- [PX4 ULogs](guides/px4-ulog.md): extracting canonical trajectories with
  `glassbox extract`, the fit flags, and recording a reproducible PX4 SITL
  flight. The pinned public corpora are obtained with `glassbox corpus`, whose
  registry names each one's citation, license and evaluation split.

## Research investigations

These pages preserve experiment settings and evidence. They are not required
configuration steps for using Glassbox. The product target is the
[opinionated onboarding workflow](scope.md#opinionated-onboarding). Their
archives are not carried forward; the history holds them.

- [Streaming refinement](streaming-refinement.md): bounded learning during
  simulated tracking, controller handoff, and frozen/adopting comparisons.
- [Cascade refinement](cascade-refinement.md): learning an X8 model from an
  independent simulator, held-out prediction and ordinary tracking.
- [Repeated-fit uncertainty](repeated-uncertainty-calibration.md): parameter
  information and forecast error under observation noise and reduced excitation.
- [How accurate is accurate enough?](accuracy-requirements.md): task error
  budgets, measured forecast allowances, and the gap between prediction scores
  and demonstrated task performance.
- [Forecast accuracy versus ordinary tracking](cascade-accuracy.md): 33 matched
  Cascade trials, conditional error tolerances, and command-response failures
  hidden by reserved-recording forecast scores.
- [Bounded recovery control](recovery-investigation.md),
  [terminal suffixes](terminal-suffix-investigation.md),
  [fast suffixes](fast-suffix-investigation.md) and the
  [shift-uncertainty audit](shift-uncertainty-audit.md): the NMPC solver,
  supervisor and uncertainty studies behind the control seam.

## Background

- [Literature review](literature-review.md): the decision record and the
  negative results, each with the last commit that carried its code.
- [Original proposal, August 2026](history/idea-2026-08.md): history, kept for
  the record and excluded from the source distribution.

## Recorded artifacts

[`results/`](results/) holds the machine-readable comparisons linked from
[validation](validation.md). One manifest describes them in two tiers and
one command produces or checks each of them; see
[`CONTRIBUTING.md`](../CONTRIBUTING.md) for the table and the procedure.

## Related

- [glassbox-throw](https://github.com/Rylandl/glassbox-throw): the Crazyflow
  throw demo built on this package, including its dual-control NMPC design and
  the closed-loop bootstrap, prototype and throw diagnostics.
