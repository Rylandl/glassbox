# Glassbox documentation

Eight pages. Start with [scope](scope.md) for what the library covers and
where its boundary is; go to [validation](validation.md) for the evidence.

- [Scope and current boundary](scope.md): the question, the system boundary,
  what exists today, the evidence standard, and what is not claimed.
- [Validation](validation.md): the five corpus rows and the three diagnostic
  rows, each number named by the artifact and key it comes from, with one
  section per corpus and per diagnostic.

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

[`results/`](results/) holds the machine-readable artifact behind every number
on [validation](validation.md). One manifest describes them in two tiers and
one command produces or checks each of them; see
[`CONTRIBUTING.md`](../CONTRIBUTING.md) for the table and the procedure.

## Related

- [glassbox-throw](https://github.com/Rylandl/glassbox-throw): the Crazyflow
  throw demo built on this package, including its dual-control NMPC design and
  the closed-loop bootstrap, prototype and throw diagnostics.
