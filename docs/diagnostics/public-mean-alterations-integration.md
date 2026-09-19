# Twelve-case alteration audit draft

No tests, alteration cases or numerical fixtures have been executed. Only Ruff
and Python compilation have run. Copy the two Python files into the successor's
`src/glassbox/experimental/` and `tests/`, review, commit and bind before execution.
The complete final qualification bundle must exist and have an external SHA.

API:

```python
run(
    bundle,
    output,
    expected_bundle_sha256=...,
    layout_path=...,
    expected_layout_sha256=...,
    binding_path=...,
    expected_binding_sha256=...,
    reference_root=...,
    reference_sha256=...,
)
```

CLI is `python -m glassbox.experimental.public_mean_alterations run BUNDLE OUTPUT`
with the seven keyword arguments above expressed as hyphenated `--` flags.
`OUTPUT` must not exist and must be outside `BUNDLE`. The launcher must preserve
command/stdout/stderr/exit evidence, including a preflight failure before output
creation. Do not retry or replace an existing attempt directory.

The layout is a separately SHA-anchored JSON document. It contains locations,
old stage associations, mirrors and hash links; it does not supply validator
implementations, arbitrary Python, thresholds, a reduced case list or new gate
values. Absolute paths are prohibited inside the bundle layout.

Required fields:

- `format`: `glassbox-public-mean-alterations-v1`.
- `protocol_sha256`: frozen `d3a8d0eeeeab2aac923cad7f1f5fe1c8a49fc0ffa765e84b3d7c6bc80cec8f99`.
- `root_manifest`: relative complete root manifest, whose `files` map covers
  every other file recursively. No unlisted auxiliary exceptions inside this
  final complete bundle.
- `paths`: exact logical names `public_model`, `port_model`, `lifecycle_m0`,
  `lifecycle_m1`, `decision`, `resolved_protocol`, `crazyflow_data`,
  `cascade_data`, `crazyflow_evaluation`, `cascade_evaluation`, `numeric_inputs`,
  `numeric_oracle`, `consumer_manifest`, `consumer_result`, `consumer_binding`.
  First two refer to Crazyflow fresh public fit and imported saved-mean fixture.
  The four data/evaluation values are directories; numeric inputs/oracle are
  directories containing `manifest.json`/`result.json`; others are files.
- `anchors`: `{path, sha256}` entries for every original stage manifest, numeric
  input/output seals, source binding, consumer input/result/adjudication seal and
  independent gate evidence used here. These preserve the externally captured
  identities when final bundle assembly copies files. The original consumer
  binding must appear explicitly.
- `stage_bindings`: entries `{path, sha256, unchanged_sources, associations}`.
  `unchanged_sources` enumerates reused helpers/numerical/public IO dependencies
  whose old and current source bytes must agree. Each association is
  `{path, pointer}` locating the old binding SHA in that old stage. Include
  saved-port, flight fits, lifecycle, actual numeric, consumer and corrected
  physical evaluation stage associations. Preserve distinct old bindings;
  this does not call a strict old-stage verifier with the new binding or claim
  an old stage executed at the new commit. A stage with an unchanged entire old
  source map may list all old sources. The saved-port stage needs a narrower
  truthful scope because an unrelated later synthetic source changed.
- `parity_case_ids`: all twelve Crazyflow `source=public_archive` case IDs in the
  authenticated actual-numeric packet's original order. The implementation
  independently requires that exact complete roster. It compares all seventeen
  scientific arrays to the retained research revision and freshly forecasts
  against authenticated `oracle64/original__eager` arrays after the frozen NumPy64
  normalization `(forecast - state_mean) / state_scale` using authenticated
  research norms. The overall numeric
  finite-difference gate is not required to pass and is not reinterpreted.
- `decision_pointer`: JSON keys locating the exact `promotion(...)` result within
  the final decision file; use `[]` if the file is precisely that result.
- `gates`: exact frozen promotion gate names, each `{path, sha256, pointer}` to an
  independently anchored Boolean in its actual stage adjudication. The decision
  file cannot be its own gate evidence. The physical gate is independently
  recomputed from raw query/prediction arrays and paired-bootstrap draws. Failed
  numerical qualification remains false. All other gate evidence remains under
  its original stage anchor, not guessed from an adoption Boolean.
- `mirrors`: exactly PM01 through PM12 in order, each a list of declared JSON
  mirror writes `{path, pointer, source}`. `source` is either
  `{kind: "detail", pointer: [...]}` into mutation details or
  `{kind: "json", path: ..., pointer: [...]}` into another declared JSON payload.
  PM01 must have `[]`; its raw archive fingerprint is deliberately unchanged.
  Model mutations expose details `model_metadata`, `model_sha256`,
  `model_fingerprint`; use them to update model report/identity/role/envelope
  mirrors without touching independent witnesses. PM04 additionally exposes
  swapped `training_identity` and `development_identity`; metadata already
  includes swapped ordered window IDs and all three per-parent report maps.
  PM07 directly updates data query/array counts, role prediction counters,
  queries, all-arm rows and summary; details include `remaining_data_queries`,
  `remaining_data_arrays` for any other count mirrors. PM09-11 expose the
  modified target via a JSON source pointer. PM12 metadata includes the changed
  `report.previous_revision`. Mirrors must not overwrite immutable scientific
  reference arrays, original numeric oracle predictions or gate evidence.
- `reseal_order`: explicit bottom-up list of hash links and inventory updates.
  Hash links are `{kind: "hash_links", path, links:[{pointer,payload}]}`. An
  inventory node is `{kind: "inventory", path, root, pointer, excluding}`;
  its `files` map covers all files below `root` except listed relative names.
  Some seal fields may need a hash-link pass after their own inventory pass;
  order must finish children and references before an enclosing inventory.
  The last node must be exactly:
  `{kind:"inventory",path:ROOT,root:".",pointer:["files"],excluding:[ROOT]}`.
  Internal validation reruns every declared link/inventory and four public
  archive fingerprints. A stale order is an explicit audit prerequisite failure.

Execution is twelve clean direct validators followed by twelve fresh full-bundle
copies in frozen order. `/bin/cp -cR` requests copy-on-write and every copied file
must have a different inode/device identity from its source. A recorded clone
failure permits an ordinary-copy fallback; hardlinks are never used. Logical
bundle bytes and every copy command/timing are recorded. Successful disposable
copies are removed sequentially; any failed case copy is retained. Root payloads
are reverified between challenges and at completion, including audit failure.

Every semantic case runs in a new pinned subprocess. Historical reconstruction
and source attestation run with the historical checkout; mean parity/calibration
run with public64; prediction replay and the consumer run with public32. The
consumer reruns against its original authenticated binding/root so absolute
import path attestations remain truthful. A fresh public-only import blocker
covers its load/predict/JIT/JVP execution. No mixed-precision compiled function,
initializer, optimizer, gradient update, solve or simulator is invoked.

The exact mutation and semantic check remain fixed by PM01-PM12. Clean checks,
repaired internal seals and external root rejection are distinct recorded
outcomes. A helper's typed semantic error may be translated only at its named
adapter boundary (query reconstruction/calibration); unexpected exceptions or
other check IDs fail the audit. Existing optimizer/initializer and physical
simulation replays are deliberately outside this bounded alteration scope.

## Noncircular final adjudication

The audited bundle is the complete **pre-audit evidence snapshot**, not a claim
that its own future alteration audit has already passed. Its independently
anchored `replay_and_evidence_integrity_pass` Boolean is false/not-yet-established,
with an explicit pending-audit scope; never manufacture true to assemble a
layout. The numerical gate is independently false, so the saved public-adoption
verdict is already false under the frozen rule. PM09 flips that actual saved
verdict and independently recomputes every then-available named gate.

After all twelve challenges complete, an external final composite anchors the
unchanged snapshot SHA, completed audit report SHA and final gate evidence. It
freshly recomputes promotion with the updated integrity evidence and records the
still-false adoption verdict. This final composition is an ordinary reduction,
not a thirteenth challenge. Do not rewrite the audited snapshot, move a pending
Boolean to true inside it, or recursively repeat alterations on a bundle that
contains its own prior audit. The final report must distinguish the audited
snapshot's stage scope from the subsequently completed composite adjudication.
