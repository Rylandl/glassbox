# Changelog

## Unreleased

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
