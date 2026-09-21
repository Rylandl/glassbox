# Single learner adoption and cleanup

Completed 2026-09-21. The supported shared-physics learner is the sole maintained
implementation, adopted at the user's direction. The cleanup preserves its
measured predictions and derivatives; it does not change earlier scientific
acceptance flags or claim broader accuracy from a refactor.

## Maintained implementation

| Tracked Python code | Before | After |
| --- | ---: | ---: |
| Package modules | 161 | 14 |
| Package lines | 92,464 | 2,216 |
| Test modules | 154 | 8 |
| Test lines | 59,911 | 1,323 |
| Scripts | 23 | 1 |
| Examples | 5 | 1 |

The dynamics formulation and fitting arithmetic occupy 671 lines in
`src/glassbox/_dynamics.py`. The rest supplies the fixed fitting recipe, immutable
fit/predict/update lifecycle, recording validation and I/O, generic geometry,
forecast scoring and the two-command CLI. The retained 655-line script verifies
the saved baseline; it is not a second dynamics implementation.

Removed public v4, structured model catalogs, experimental learners, belief and
controller frameworks, simulator/telemetry integrations, configuration files,
optional dependency groups, obsolete tests, examples, scripts and research docs.
The lock contains 14 packages. Python 3.12 and 3.13 are supported.

## Verification

The [protocol](harness/single-learner-adoption-v1.json) was committed as
`f0bb67a` before changes. Complete implementation `5c74b41` passed full replay;
the final NumPy recording-metadata correction was committed as `f55f1f0` and
passed the complete replay again. Final results:

- All three adopted model cores preserve their 19 numeric arrays exactly.
  Altered archives are rejected. Retained training/development caches support
  updates; no old-map envelope is transferred to the adopted revisions.
- All 12,768 saved flight arrays match exactly across 8,064 queries. This includes
  the original 399 ineligible arrays; 12,369 predictions were recomputed.
- All 40 selected Dart predictions and 4,144 actual objective/gradient callbacks
  match exactly using the unchanged external Dart objective. Canonical contact
  replay and independent contact/history audits pass.
- The installed wheel passes all 76 tests on both Python 3.12 and 3.13, outside
  the checkout. The wheel contains exactly the 14 maintained source modules;
  building it from the source distribution succeeds.
- Ruff, formatting and offline dependency-lock checks pass.

Numerical replay used CPU/arm64, CPython 3.12.12 and JAX default32. No long fit,
optimizer solve or simulator trial was run. See the [replay instructions](../CONTRIBUTING.md#replay-the-adopted-baseline)
and [machine-readable result](harness/single-learner-adoption-v1-result.json).

## Preserved evidence and deleted artifacts

`artifacts/baseline` retains 383 hash-verified payloads totaling 191,883,113 bytes:
the three adopted public archives, small original numeric references, all saved
evaluation inputs/outputs, 348 relevant raw recording files, and the full last
Dart milestone proof compressed into one archive. Raw simulator recordings
retain their original schemas and contracts; they require interpretation into
canonical recordings before loading with the public recording API.

The baseline manifest is pinned in [baseline.json](baseline.json). The 70,241-file
Dart proof archive includes its source bundle. The public verifier reads neither
retired worktrees nor dated artifact directories. Historical absolute paths in
the frozen protocol and provenance identify their original locations only.

Deleted all five obsolete dated artifact roots, 2026-09-17 through 2026-09-21,
and 60 clean retired experiment worktrees. Their inventoried logical size was
47,627,718,893 bytes, including duplicate checkouts and results; this is not a
claim about physical space recovered on a filesystem with shared blocks. Git
retains committed history and branch references. External Dart and simulator
repositories and environments were left intact.

`artifacts/maintenance` contains the exact deletion plan and compressed file
ledger, completed deletion records, baseline replay bindings/results, and tested
wheel/source distributions and test reports. The temporary cleanup worktree is
removed after integration, leaving the original working checkout.

## Remaining scope

This intentionally narrows the public API and archive format. Consumers of the
deleted catalog/controller/experimental interfaces need migration in their own
projects; no compatibility implementation remains. The unchanged Dart objective
was replayed here, not every historical Dart CLI or experiment.

Known wind, response-tail and long-horizon errors, independent calibration,
runtime and neighborhood reliability remain open. The next scientific gap is
listed in [status.md](status.md); cleanup adds no new performance claim.
