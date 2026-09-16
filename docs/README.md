# Glassbox documentation

Start with the repository [README](../README.md) for the identification workflow.

- [Scope](scope.md): the core workflow and design decisions.
- [Charter](charter.md) and [status](status.md): the one generic learner we
  are building, its definition of done, the rules, and the current gap.
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
[opinionated onboarding workflow](scope.md#opinionated-onboarding).

- [Streaming refinement](streaming-refinement.md): bounded learning during
  simulated tracking, controller handoff, and frozen/adopting comparisons.
- [Cascade refinement](cascade-refinement.md): learning an X8 model from an
  independent simulator, held-out prediction and ordinary tracking.
- [Repeated-fit uncertainty](repeated-uncertainty-calibration.md): parameter
  information and forecast error under observation noise and reduced excitation.
- [Generic transition learning](generic-transition-support.md): an experimental
  general function learner, missing combinations, and directional support.
- [Learning nonlinear shapes](shape-learning.md): dead zones, saturation,
  multiple variation scales, and choosing additional observations.
- [Calibrated error and history](error-calibration.md): separate error evidence,
  local interval coverage, calibration budgets, and missing causal inputs.
- [Temporal structure on real flights](real-transition-learning.md): state-change
  learning, causal history, whole-flight splits, and empirical forecast coverage.
- [Diagnosing output sharing and history](transition-diagnosis.md): matched
  model comparisons, optimization effort, and nested observation budgets.
- [Experiments with generic model structures](model-structure-experiments.md):
  learned affine/nonlinear corrections, recursive memory, trajectory objectives,
  and the limits of compositional extrapolation.
- [Sequence learning across platforms](sequence-transfer.md): a shared recipe
  on Nano, X8 and causal ARP telemetry, development guards, regularization,
  and the remaining gap against simple reference predictions.
- [Direct forecasts and error diagnosis](forecast-diagnosis.md): matched
  direct/recursive fits, whole-flight evaluation, and the distinct meanings of
  support distance, prediction error and improvement over a reference.
- [Forecast representations and segment coverage](forecast-representation.md):
  generic feature ablations, a frozen Crazyflie sensor-log evaluation, and the
  separate effects of training coverage and model selection.
- [Recording-level selection and observation budgets](recording-selection.md):
  reusable segment provenance, model-choice stability across recordings, and
  separate fixed-window and fixed-observation comparisons.
- [One opinionated generic learner](default-recipe.md): a fixed fit, predict,
  and update workflow, automatic data handling, and the cross-recording failures
  that keep this candidate experimental.
- [How accurate is accurate enough?](accuracy-requirements.md): task error
  budgets, measured forecast allowances, and the gap between prediction scores
  and demonstrated task performance.
- [Forecast accuracy versus ordinary tracking](cascade-accuracy.md): 33 matched
  Cascade trials, conditional error tolerances, and command-response failures
  hidden by reserved-recording forecast scores.
- [Falsifying the generic fitting assumptions](model-qualification.md): 27
  synthetic fits testing command-response identifiability, insufficient history,
  and equivalent observation encodings with the unchanged recipe.
- [Observational sequence diagnostics](sequence-diagnostics.md): one read-only
  call for input predictability and older-history error evidence, tested against
  synthetic positive controls and demonstrated predictor blind spots.
- [History confounding and horizon weighting](history-confounding.md): a fully
  observed counterexample to interpreting history gain as missing memory, and
  a matched loss-normalization experiment with measured accuracy tradeoffs.
- [Horizon normalization across generic systems](horizon-generalization.md):
  a wider rejection of pooled normalization and fresh-seed testing of a narrower
  weight cap, including its confirmation regression.
- [Checkpoint attribution and development coverage](checkpoint-attribution.md):
  separate optimization, selection-criterion, and recording-sample effects,
  with fixed-budget coverage tests and retained confirmation failures.

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
