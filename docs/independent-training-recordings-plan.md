# Independent training recordings: implementation map

Prospective frozen design, 2026-09-19. No collection, fitting or confirmation
prediction has run. Root reviewed the scientific choices and helper boundaries.
Commit the [protocol](harness/independent-training-recordings-v1.json), its
[literal roster](harness/independent-training-recordings-v1-roster.json), and this
map before implementation. Commit and bind tested implementation before scientific
execution. The base is `d8fddcc5e2f9c8f56b369a7cf9ebb16d10e69bb4`.

The intervention adds 72 planned independently seeded excited training recordings
per simulator, preserving the exact original training-condition allocation. The
control's 72 training parents and 24 development parents are externally anchored.
The candidate retains a 1,536-window training budget, the exact 256 development
windows and 1,000 full-cache optimizer attempts. One real public `update` call per
simulator produces the candidate; there is no control refit or new learner option.

## Why public update preserves the intended experiment

Calling `fit` on all 168 recordings would reserve 42 development recordings.
The existing `LearnedDynamics.update` already keeps the original development
cache, extracts up to 1,536 windows from the new recordings, merges that cache
with the control's 1,536 training windows, and runs the same numerical fitter.

For disjoint old/new parent IDs, `_indices` orders origins inside each parent by
the same SHA priority, then visits `(within-parent depth, recording name)` in
order. Any element in the first M entries of the combined pools has rank at most
M within its own pool. Therefore the union of each pool's first M entries contains
the combined first M; sorting/truncating that union with the same rule recovers
the same selection. This remains true when parents exhaust their legal origins.
Use M=1,536. The original cache must first be shown to be the exact full old-pool
prefix. Disjoint IDs, origin order and identical selector are essential.

The harness must check this equivalence on the actual recordings before fitting:
full 144-parent extraction versus the public merge of old/new 1,536-window caches,
including keys, source origins, all arrays and declared sidecars. Then observe the
actual `_train` arguments reached by `control.update(new_recordings)` and require
the same bytes. A failed equality stops the experiment; it does not authorize a
private bypass. No ID renaming or modified role selector is needed.

All 72 added planned parents remain in the collection report, including setup or
flight failures. A parent with a sufficiently long valid prefix is usable; a later
altitude failure remains visible. If any added parent lacks one complete native
history-plus-horizon window, record `preparation_unavailable` with exact support
counts. Do not silently run a smaller training pool, redraw, replace the parent,
or let public automatic splitting alter development. Candidate cache selection
must represent all 144 required training parents.

## Ownership and two modules

Protocol/roster/map: `excitation_decision`; implementation and tests:
`excitation_fit_design`. Root owns final review, commits, binding and execution.
No existing learner, model, simulator, historical harness or consumer file changes.

| New file | Boundary and intended signatures |
| --- | --- |
| `src/glassbox/experimental/independent_training_data.py` | `read_protocol(path, expected_sha256) -> protocol`; `collect_training(simulator, output, *, protocol, binding, replay=False) -> sealed outcome`; `collect_confirmation(...)`; `verify_data(directory, expected_sha256, *, kind, simulator, protocol) -> seal`; `load_added_recordings(directory, expected_sha256, contract) -> SequenceCollection`. |
| `src/glassbox/experimental/independent_training_experiment.py` | `prepare_update(simulator, control, old_recordings, new_recordings, protocol) -> PreparedUpdate`; `update_candidate(simulator, output, *, protocol, binding) -> fit outcome`; `evaluate(simulator, output, *, protocol, binding, replay=False) -> rows/summary`; `decide(rows, protocol) -> decision`; `finalize(output, ...) -> seal`; `replay(simulator, output, expected_sha256, ...) -> report`. Keep the small saved-work checker here. |
| `tests/test_independent_training_data.py` | Roster/seed exclusions, planned-versus-admitted failure visibility, issued commands, role isolation, source boundary and reconstruction. No new scientific fixtures before source freeze. |
| `tests/test_independent_training_experiment.py` | Deterministic merged/full extraction equality, all144 parents, immutable exact development and control, actual update call/arguments, failed-arm visibility, paired reducer/gates, work/selected-archive links. |

`PreparedUpdate` is a harness-only frozen dataclass with `expected_train`,
`development`, `contract`, `seen`, `new_recordings`, and `provenance`. It carries
expectations and does not execute a fitter. Save those expectations before the
single public update; intercept `_train` only to compare/persist actual arguments
and call the original function unchanged. The candidate report must name the
control fingerprint as `previous_revision`; the original model, cache, ledger,
report and saved archive must remain unchanged.

## Reuse boundaries

- **Simulation:** the existing pinned historical fixture factory and physical
  operators from `expanded_training_cache_experiment.flight()` in the isolated
  oracle checkout; `command_excitation_data.excite`, `_schedule`, `_issued` for
  the training intervention. Construct an explicit small protocol view with the
  original cells/generation and the new literal role lists. Replace the old
  fixed 72-role descriptive view only where the helper requires counts; keep
  numerical operators unchanged. Generate only the new 72 training trajectories,
  not an additional unexcited companion for each. No full old collection replay.
- **Source isolation:** the new data driver can execute by absolute file path in
  an old-source child, with its own new driver hash and the old oracle map both
  authenticated. Import old simulator helpers only inside that child. The current
  public fit/predict child imports current core only. Reuse existing inventory and
  import-map checks; do not globally patch historical `ROOT`, protocol constants
  or `public_mean_physics.fresh_test_roster`, which hardcodes +11M.
- **Public preparation:** `learner._contract`, `_recording_content`, `_extract`,
  `_merge_cache`, and existing recording save/load. Reconstruct the old control
  cache from its authenticated ordinary `recordings.npz`; use the new observed
  valid prefixes and actual issued command tape for the update collection.
- **Actual update and capture:** `LearnedDynamics.load`, `model.update`,
  `learner._train`, `core._observe_attempt` and
  `public_mean_lifecycle._capture_event`. Reuse flight-fit's one initializer-return
  wrapper, fixed weight capture and external timing pattern. Reuse the unchanged
  `public_mean_flight_fit._work` with a typed current-fit witness reference:
  actual initializer-return arrays, selected public-model arrays, saved objective
  arrays, actual cache and `.01*N*H` ridge/delay. First verify current-cache
  arithmetic norms/hold/e0/weights and the external five-array initializer subset.
  Label this within-fit stage consistency, not a historical reference equality.
  The lifecycle `_observed_work` function itself hardcodes one-step ridge and
  must not be called unchanged. No copied optimizer/observer stack, extra
  initializer or gradient replay.
- **Prediction:** current `LearnedDynamics.predict` in fixed default32; the same
  historical `predict_query` prefix logic when later truth is missing. Forecast
  candidate/control/hold only. No second precision arm or archived v3 contender.
- **Scoring:** `two_simulator_metrics.score_forecast`,
  `score_response_directions`, `aggregate`; copy the small per-query scoring loop
  with three explicit arm names. `public_mean_physical_evaluation.score` itself
  hardcodes the old five-arm role layout, so do not pass a fake roster to it.
- **Decision:** `state_input_decision.reduce` already reduces the baseline versus
  candidate pair from complete raw rows with the inherited weighting, floors,
  primary/scope/tail guards and paired bootstrap. Give it a fresh protocol copy
  containing the exact numeric guard dictionary. For the four targeted angular
  checks, reuse `_angular_repair` only through a private two-arm name view of the
  same rows and bootstrap draws; do not mutate module globals. Validate a complete
  baseline/candidate/hold query roster first. Keep all per-condition reports.

## Execution order and evidence

1. Authenticate protocol, literal roster, sources/runtime, exact old archives,
   old 72/24 roles, control cache and the completed corrected-source replay anchors.
2. Collect and seal both new training pools with all planned/failed outcomes.
   Construct and save full144-versus-public-merge preparation expectations.
3. Execute exactly one public update per simulator. Save preparation, all actual
   work/checkpoint witnesses, candidate archive, parent revision, and immutable
   control checks. Seal both outcomes before any confirmation generation/scoring.
4. Generate the frozen +12M confirmation cohort: 84 parents per simulator, all
   original scopes and probes. Added training uses the distinct +20M seeds.
5. Predict only the development-selected candidate and imported control; retain
   unavailable candidate and truth slots if a stage failed. Score all native and
   50/150/250 ms errors, then reduce the frozen progress/retention criteria.
6. Replay new training/test physics, exact expected caches, selected predictions
   and reductions without fitting. Independently reduce physical means/tails;
   execute the eight fixed focused alteration cases on disposable copies.

Initial and selected training/development physical diagnostics can explain the
named angular gap; they cannot select a checkpoint or alter a confirmation gate.
Report actual training-window/target overlap and old/new exposure. Identical
gradient-window counts do not imply equal wall time or acceptance-evaluation work.

The four angular retention limits apply only to Crazyflow primary 250 ms
forecast/response aggregate RMSE and parent p95. They do not require every cell to
win. Inherited broad weighting/guards remain unchanged. A successful study can
support bounded **offline update improvement on this fixed two-simulator
intervention**. It does not establish safe live controller swapping, calibrated
uncertainty, universally improving updates, or new-system coverage.
