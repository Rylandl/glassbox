# Public mean qualification: bounded alteration roster (unfrozen draft)

Preparation only, 2026-09-19. No implementation, fitting, mutation or audit has
been run for this proposal. Freeze this roster, its fixtures and the named
validation contracts before implementing the public integration.

Use 12 actual-bundle challenges. Each starts from a fresh disposable copy of the
same externally anchored qualification bundle. Do not modify production source,
historical bundles or their protocols. The reference worker uses the pinned
historical checkout and runtime; the public worker uses the new implementation.

| ID | Single challenge and target | Independent semantic validator |
| --- | --- | --- |
| PM01 | Change one parameter byte in a public model archive without updating its internal fingerprint. | Public `LearnedDynamics.load` must reject the corrupted archive. Separately require the qualification bundle's external-anchor rejection. |
| PM02 | Change one quadratic mean coefficient in the saved public representation of the retained research-mean parity fixture; repair archive and local seals. | `retained_mean_parity`: compare the complete scientific parameter/norm roster against the externally anchored historical selected revision, then compare public forecasts on the frozen parity queries with the historical worker's predictions. This fixture is explicitly imported for parity, not represented as a new fit. |
| PM03 | Change an added training-cache command at row 384, leaving the original 384-prefix unchanged; repair archive and local seals. | `training_cache_reconstruction`: extract the fixed capped cache directly from authenticated recordings and compare ordered keys, absolute origins, all batch arrays, dtypes and excitation sidecars. No initializer or fit. |
| PM04 | Swap one training/development recording in the declared initial role assignment and all report mirrors; repair local seals. | `initial_role_identity`: independently recompute the content ledger and hash-based recording split from authenticated input recordings; check exact role membership, disjointness and cache membership. |
| PM05 | Scale one public envelope channel and its report mirror; repair archive and local seals. | `calibration_reduction`: predict the pinned development cache with the saved selected mean, then independently recompute the frozen absolute-error order statistic, rank, nominal level, shape and calibration provenance in NumPy. This checks calibration mechanics, not held-out coverage qualification. |
| PM06 | Change one saved held-out query's command/history value while retaining its origin ID; repair local seals. | `query_reconstruction`: recut the declared query from authenticated saved native/branch trajectories and the frozen query plan; compare full query arrays, timing, ordered channels and origin identity. No simulator rerun. |
| PM07 | Mark one originally eligible query unavailable and remove its metric contributions, updating count/summary mirrors and local seals. | `truth_roster_and_masks`: regenerate the complete planned roster and eligibility from source trajectories/branch truth and frozen validity rules, independent of predictions. Compare every query's status and the common-arm mask; a smaller evaluated population must fail. |
| PM08 | Change a selected-model prediction and consistently rewrite its derived metric rows/summary; repair local seals. | `saved_prediction_replay`: freshly predict the unmodified authenticated query with the saved public model and compare the array, then independently reduce physical errors. Matching forged summaries cannot satisfy model execution. |
| PM09 | Flip the declared public-mean adoption Boolean in the decision; repair local seals. | `decision_reduction`: recompute the frozen rule from validated raw rows with the complete arm/cohort roster and exact paired-bootstrap schedule. Bootstrap RNG is permitted; fitting RNG is not. |
| PM10 | Replace the historical worker's recorded `src/glassbox/learner.py` digest with the new public worker's digest, maintaining internally consistent provenance mirrors and local seals. | `worker_source_identity`: compare each role's source/runtime inventory against separately frozen pins and freshly attest actual imported module paths/file bytes in the correct subprocess. Historical source claims never authorize current learner code. |
| PM11 | Replace the independent consumer's recorded script SHA256 with 64 zeroes in its report and all provenance mirrors; repair local seals, leaving the actual consumer script unchanged. | `consumer_execution_identity`: rerun the pinned consumer's load/predict/JIT/JVP/output path in a fresh subprocess using public exports, with research/controller/simulator imports blocked; compare actual script/package identity, results and import-boundary report. No fit/update is run in this alteration case. |
| PM12 | Change the updated revision's predecessor fingerprint and report mirror, leaving the predecessor and authenticated update inputs unchanged; repair local seals. | `revision_chain_identity`: compare against the actual predecessor artifact, recompute the merged capped cache/content-ledger union from predecessor plus fresh recordings, and require byte-exact reserved development data. The predecessor's original fingerprint must remain unchanged. |

## Execution and rejection contract

Use Crazyflow's public fit for PM01, PM03-PM08 and its imported retained-mean
fixture for PM02. Use lifecycle M1/M0 for PM12. This is a bounded integrity
roster; normal replay still covers both simulators. Apply fixed mutations:
PM01 toggles the least-significant byte of the first mean-coefficient element;
PM02 adds .125 to autonomous quadratic coefficient [0,0]; PM03 adds .01 to
training future_inputs[384,0,0]; PM04 swaps the lexicographically first training
and development identities; PM05 adds .001 to envelope[0,0], including if zero.
PM06 adds .01 to future_inputs[0,0] of the first lexicographic input-complete
factual query. PM07 removes the first lexicographic primary factual query with
eligible 250 ms truth, with every arm's mask/contributions updated consistently.
PM08 adds .125 to candidate prediction[0,0] of that query and recomputes affected
physical rows and summaries. PM09 toggles the final public-mean adoption flag;
PM10/PM11 use the exact substitutions in the table; PM12 replaces M1's predecessor
identity with 64 zeroes after asserting it differs from the saved original.
Lexicographic query order is (parent,id); no target is selected by model error.
Missing required target/field/support is an explicit audit failure, not permission
to silently select another challenge or skip it. Every mutation must actually
change its target bytes, and only declared mirrored fields/seals may be repaired.

- PM01 is one raw integrity case; PM02-PM12 are 11 coherently resealed semantic
  cases. For every case require rejection by the unchanged external bundle SHA.
  For coherent cases also verify repaired internal hashes before invoking the
  named semantic validator directly. Do not count external-hash rejection alone
  as semantic evidence.
- Every named validator must pass on the clean bundle first. Its expected
  failure is a new harness-owned structured `check_id`/typed validation result,
  not a literal historical exception string. Unexpected exceptions, missing
  evidence and failures in a different prerequisite are audit failures.
- Retain original ordered case IDs, mutation detail, repair checks, both
  rejection outcomes, script/runtime/source identities, timings, attempt
  logs/status and cleanup results. Reverify original payloads at completion.
  Preserve every failed attempt; a resumed subset requires an explicitly
  anchored composite with exact case coverage.
- Public `load(path)` has no external expected-hash argument. A coherently
  changed but structurally valid archive may be another valid model: origin and
  experiment identity are established by the external anchor and replay, not by
  claiming that the archive's self-fingerprint authenticates its author.

## Reuse boundaries

- Reuse the existing public lifecycle malformed-archive tests for wrong
  format/recipe, missing/extra arrays, shape/dtype/nonfinite values, contract
  mismatch, duplicate recording IDs/content, immutable update, pinned
  development, causal prefix, batching/JIT and save/load parity. Extend them to
  the new parameter roster and smaller-than-cap caches. Do not duplicate every
  malformed-schema example as a costly actual-bundle mutation.
- Reuse the existing external root/payload sealing primitives and independent
  physical/paired reductions. Use one qualification runner and the isolated
  reference subprocess, rather than another model wrapper or a copy of the
  37-case research audit.
- The retained research bundle's completed initializer/optimizer/physics audit
  remains historical evidence under its own anchor. This roster does not rerun
  proposals, gradients, Adam, solves, simulator rollouts or every ancestor's
  audit. New public fits and lifecycle qualification are separate predeclared
  stages; replay and alteration checks never refit.
- PM11 establishes a real consumer boundary for the saved mean. Public update
  semantics, uncertainty-calibration mechanics and computational derivative
  parity remain distinct from update improvement, held-out uncertainty
  coverage, physical derivative fidelity or controller qualification.
