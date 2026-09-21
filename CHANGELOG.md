# Changelog

## Unreleased

Adopt the supported shared-physics learner as Glassbox's single public dynamics
implementation. The API is `fit(recordings)`, `model.predict(...)`, immutable
`model.update(recordings)` and fingerprinted save/load. Recordings describe
canonical rigid-body motion and any number of issued command channels.

Remove the previous generic learner, structured model catalog, belief/controller
interfaces, simulator and telemetry integrations, and obsolete experiments.
The CLI now contains only `fit` and `evaluate`. Saved formats and supported state
semantics have changed; old artifacts are not a compatibility interface.

This is adoption on the existing evidence, not a new claim of arbitrary-system
accuracy, calibrated uncertainty, physical derivative fidelity or universal
controller success. Current results and limitations live in
[status](docs/status.md). Earlier release history remains in Git.
