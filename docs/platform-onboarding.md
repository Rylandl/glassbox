# Platform onboarding contract

Glassbox's primary interface is the adopted generic learner:
`fit(SequenceCollection)` returns a `LearnedDynamics`, `predict` consumes
observed history and future commands, and `update` returns a new revision.
The [runnable walkthrough](guides/platform-onboarding.md) demonstrates it.
The [scope](scope.md#opinionated-onboarding) and [charter](charter.md) define the
development boundary.

## Responsibilities

| Step | Caller supplies | Glassbox provides |
| --- | --- | --- |
| Describe | Configuration identity, ordered observed/command channels, units, frames and sample interval | Contract validation |
| Record | Distinct source identities, aligned observations/commands and gap boundaries | Immutable contiguous segments and recording collections |
| Fit | At least two distinct recordings with usable windows | One fixed recipe, automatic development split, model and report |
| Predict | Sufficient recent observations and commands, plus a bounded future command sequence | Differentiable means and a measured error envelope |
| Evaluate | Untouched compatible recordings and application requirements | Per-channel forecast errors, hold-current comparison and coverage |
| Update | Fresh recordings of the same configuration and signal contract | New revision, refreshed development envelope and predecessor identity |

Collecting informative data is part of onboarding. The learner does not
choose a stabilizer, identify sensor semantics or guarantee sufficient command
excitation. Record these dependencies when comparing the effort needed to
bring a new system into use.

## Identity and revision policy

Keep all segments from one source recording under the same recording ID.
A different filename is not a new experiment. The learner rejects exact
duplicate content and previously seen update identities; the application
must still prevent overlap hidden by renaming or resegmentation.

An update does not replace the active model. Save it separately and compare
old and new models on the same untouched data. Adopting a model for control
requires a separately declared task and its measurements.

## What success means

The current recipe has a broad empirical advantage across the measured flight
corpora and a visible ARP deficit. Neither its interface nor that comparison
proves arbitrary-system performance. Current control and coverage failures
remain in [status](status.md).

Onboarding improvements should reduce calibration or engineering effort while
maintaining declared forecast and task requirements. Freeze the evaluation
plan before fitting. Report prior information, collection dependencies,
individual system gains/losses and task outcomes separately.
