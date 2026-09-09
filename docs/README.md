# Glassbox documentation

Start with the repository [README](../README.md) for the identification workflow.

- [Scope](scope.md): the core workflow and design decisions.
- [Validation](validation.md): recorded prediction and control comparisons.
- [Repeated-fit uncertainty](repeated-uncertainty-calibration.md): parameter
  information and forecast error under observation noise and reduced excitation.

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

- [PX4 ULogs](guides/px4-ulog.md): extracting canonical trajectories with
  `glassbox extract`, the fit flags, and recording a reproducible PX4 SITL
  flight. The pinned public corpora are obtained with `glassbox corpus`, whose
  registry names each one's citation, license and evaluation split.

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
