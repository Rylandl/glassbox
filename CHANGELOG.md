# Changelog

## Unreleased

Compress temporal inputs only to the nonlinear acceleration head. Preserve the
full linear lag path and stable accumulators; reduce the four-command / 10 ms
model from 8,714 to 5,450 parameters. Matched tests improve command-response and
Dart forecast error with a modest forecast/runtime cost. This is not a speed
improvement. Model/session formats change; no alternate model option is retained.
See [the comparison](docs/nonlinear-temporal.md) for all results and the correction
to earlier full-stream accuracy claims.

Reuse the centered history projection across acceleration integration stages.
Model capacity, saved accumulator archives and fitting settings remain unchanged.
Whole-update snapshot medians improve 19.17% for quads and 4.85% for fixed wings;
strict post-update numerical flags and one tail regression remain documented in
[the comparison](docs/history-projection-reuse.md).

Replace nonlinear recurrent memory with eight learned stable accumulators. The
same fit/predict/update workflow, shared mechanics and full lag inputs remain.
Recorded history is reduced in parallel; quad online update time falls roughly
by half on the known paired benchmark, with slightly better aggregate accuracy.
Model and online-session archive formats change; refit old revisions or use
their historical checkout. No legacy dynamics implementation or model selector
is retained. See [migration evidence](docs/accumulator-migration.md).

Adopt the supported shared-physics learner as Glassbox's single public dynamics
implementation. The API is `fit(recordings)`, `model.predict(...)`, immutable
`model.update(recordings)` and fingerprinted save/load. Recordings describe
canonical rigid-body motion and any number of issued command channels.

Remove the previous generic learner, structured model catalog, belief/controller
interfaces, simulator and telemetry integrations, and obsolete experiments.
The CLI now contains only `fit` and `evaluate`. Saved formats and supported state
semantics have changed; old artifacts are not a compatibility interface.

Add bounded causal `OnlineFit` sessions with physical-vector forecast loss,
compensated normalization and a generic quadratic-curvature prior. The v6 mutable
session format stores deterministic optimizer/cache state; immutable model
archives and prediction equations are unchanged by this online iteration.

This is adoption on the existing evidence, not a new claim of arbitrary-system
accuracy, calibrated uncertainty, physical derivative fidelity or universal
controller success. Current results and limitations live in
[status](docs/status.md). Earlier release history remains in Git.
