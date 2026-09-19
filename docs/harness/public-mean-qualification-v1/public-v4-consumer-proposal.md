# Prospective Dart public-model consumer demonstration

2026-09-19. Design only: no Dart/code edits, fits, forecasts or trials performed.
This is one external forecast/sensitivity consumer, not a controller migration.

## Ownership and command

Implement `src/crazydart/glassbox_forecast.py` and focused consumer tests in Dart.
Run `python -m crazydart.glassbox_forecast --manifest INPUT.json
--expected-manifest-sha256 TRUSTED_SHA256 --output OUTPUT`
from the Dart checkout in a fresh process. These are paths, not learner options.
Leave `glassbox_impact.py` unchanged: it fits models, imports obsolete experimental
APIs and compares a structured controller model, so it cannot establish the new
public boundary. No import from that file, no copying a Glassbox internal runner.

Allow NumPy/JAX/stdlib and only these Glassbox imports:
`from glassbox import LearnedDynamics` and
`from glassbox.io.recordings import load_recordings`.
Use public `.load`, `.contract`, `.recipe`, `.history_steps`, `.horizon_steps`,
`.fingerprint`, `.predict` and `.envelope`. No `_model`, caches, normalization
arrays, `_contract`, private extraction, research wrappers, simulator or controller
objects. In particular do not call `from_trajectories`, whose current implementation
imports structured/core geometry. The generic archive producer owns the observed
signal conversion; Dart verifies and consumes its explicit channel contract.

The source/runtime manifest must prove this module executes from Dart and imports
the new public Glassbox implementation, rather than Dart's existing nested editable
`glassbox` dependency. Check actual `__file__` paths, imported source hashes, package
versions and interpreter identity; reject a mismatch. Pin Dart's package initializer
as well: it sets `SCIPY_ARRAY_API` but does not select Glassbox numerical precision.
Begin ambient x64=False, with no `JAX_ENABLE_X64` prerequisite and no consumer flag
mutation. Imported experimental/core/control/compatibility modules are a failure.

## Exact input-only roster

Reuse, without reselection, the lifecycle note's fresh +11,000,000 cohort roster:
per simulator, primary/factual, primary/response, shifted/factual,
shifted/response; shifted is exactly heading_shift, maneuver_shift, speed_shift,
wind_shift. At the model's maximum horizon, filter only history_eligible, declared
shapes and finite complete input tapes. For each stratum sort `(parent,id)` with
case-sensitive lexicographic order and choose the first/last distinct identities.
Require two identities per stratum; unknown kinds/scopes or shortages fail.
Never filter targets, masks, errors, float32 results or derivative magnitudes.

There are eight query identities per simulator, four of them response pairs:
16 queries total, 24 individual mean-input cases, eight paired responses. Keep
both response branches as separately identified inputs even if histories coincide.
Selection and input hashes are sealed before inference, and checked independently
against the full fresh query manifest during replay. Reusing this declared subset
adds no independent accuracy cohort; full physical qualification remains separate.

The input manifest supplies exactly two fresh public model archives, one generic
recording archive per simulator, and input-only query NPZ/JSON payloads. For each
query record the model fingerprint, exact contract, parent recording ID, segment
ID, source origin row, kind/scope, dt, horizon, original query source hash and
branch identity. Preserve the command tapes' float64 source bytes; primary
inference casts them only through ordinary ambient JAX. It contains no targets,
validity-of-future-truth masks, expected predictions or expected derivatives.
The model archive necessarily contains its ordinary self-contained update cache;
the consumer must not inspect or use that private cache.

## Exact exchange schema: consumer-input-v1

`INPUT.json` is UTF-8 JSON with exactly the keys below. Reject duplicate JSON
keys, extra/missing keys, NaN/Infinity, wrong types and boolean values where an
integer is required. SHA256 strings are 64 lowercase hexadecimal characters;
commits are 40 lowercase hexadecimal characters. No optional extension fields.
Arrays listed here are JSON lists; all IDs are nonempty strings.

- Top level: `format`, `qualification`, `models`, `recordings`, `queries`.
  `format` is exactly `glassbox-public-consumer-input-v1`.
- `qualification`: `protocol_sha256`, `implementation_commit`, `exporter_module`,
  `exporter_source_sha256`, `query_selection_spec_sha256`,
  `test_seed_shift_from_original`. The shift is exactly 11000000; the selection
  spec hashes the frozen lifecycle note. `exporter_module` is the exact dotted
  module name of the new qualification exporter. Its implementation is committed
  before fresh generation; no historical producer modification is implied.
- Each of exactly two `models` entries: `id`, `path`, `sha256`, `fingerprint`,
  `contract`, `history_steps`, `horizon_steps`. IDs are exactly `crazyflow` and
  `cascade`, in that order, and are opaque routing labels in the consumer.
  `fingerprint` is the public model fingerprint; both step counts are positive
  integers. `contract` has exactly `configuration_id`, `state_channels`,
  `input_channels`, `dt_s`: nonempty configuration, nonempty ordered lists of
  distinct channel strings and finite positive interval. It must equal the
  loaded public contract without renaming/reordering or timing tolerance.
- Each of exactly two `recordings` entries: `id`, `model_id`, `path`, `sha256`,
  `parents`. IDs are exactly `crazyflow-recordings` and `cascade-recordings`,
  mapping one-to-one to the corresponding model IDs in that order.
  Every `parents` item has exactly `recording_id`, `source_parent_sha256`,
  `source_record_entry_sha256`, `segments`. The first source hash identifies
  the original parent NPZ; the second hashes the original generation-record
  entry encoded as sorted-key compact JSON UTF-8, without trailing newline.
  Every segment descriptor has exactly `segment_id`, `start_row`, `state_rows`,
  `input_rows`; start is a nonnegative integer, state_rows>=2 and
  input_rows=state_rows-1. Parent IDs are sorted/unique and equal exactly the
  distinct parents referenced by that model's queries. Segment descriptors are
  sorted by `(start_row,segment_id)` and must match every loaded public segment,
  including all retained runs and their original row offsets. No extra parents.
- Each of exactly sixteen `queries` entries: `model_id`, `recordings_id`,
  `parent`, `id`, `segment_id`, `source_origin`, `kind`, `scope`, `dt_s`,
  `horizon_steps`, `branches`, `path`, `sha256`, `source_query_sha256`.
  Source origin is a nonnegative integer in the original parent's row system,
  not a local segment index. `source_query_sha256` hashes the original sealed
  query NPZ before input-only export. Kind is `factual` or `response`; scope is
  one of the five lifecycle scopes. Branches are exactly `["factual"]` for
  factual or `["intervened","factual"]` for response. Query dt and horizon
  must equal the referenced model values. `(model_id,parent,id)` is unique;
  rows are sorted by that tuple. Each model has the four declared strata with
  exactly two query identities each; the full source-roster verifier, outside
  Dart, authenticates the first/last rule and input-only selection predicates.

Every `path` is a unique normalized POSIX relative path under the manifest's
packet directory: no absolute paths, `.`/`..`/empty components, backslashes or
symlinks in any component. Require a regular `.npz` file whose resolved path
stays in the packet. The packet contains only INPUT.json plus the declared
20 payload files (two model archives, two recording archives, sixteen query
archives). OUTPUT must be a separate new directory outside this packet. Never
follow source provenance hashes as filesystem paths or locate undeclared files.

Factual query NPZ keys are exactly `past_states`, `past_inputs`, `future_inputs`.
Response query NPZ keys are exactly those three plus `factual_inputs`. No object
arrays, pickles, duplicate ZIP/array names, nested metadata, targets, masks,
expected means/derivatives, or undeclared arrays. Require C-contiguous finite
little-endian float64 source arrays, preserving original values and byte layout:
`past_states` has shape `(C+1,d)`, `past_inputs` `(C,m)`, and each future tape
`(H,m)`, where C/H and d/m come only from the referenced public model/contract.
The input-only exporter copies these arrays from the sealed query; it neither
recomputes commands nor converts precision. It derives no weights or sensitivities.
Command delta is computed from the two response tapes; the one-unit JVP tangent
is constructed by Dart as specified below, not read from private model internals.

The two model NPZs retain their standard public self-contained format; the two
recording NPZs retain the standard generic-recording format. Their legitimate
model caches and observed recordings are not query target/expected-output files.
Dart accesses them only through the allowed public loaders/properties. All models,
recordings and sixteen query payloads must validate before the first forecast.
The CLI first checks raw INPUT.json bytes against the independently supplied
`--expected-manifest-sha256`, then every payload hash and all schema/cross-reference
constraints. A checksum recomputed inside this packet is not its own trust anchor.

The exporter and an independent source audit must prove the generic observed
recordings preserve original parent IDs, contiguous segment boundaries, source
origins and outgoing applied commands, and that every reconstructed history is
byte-identical to its sealed query. Source hashes establish provenance but cannot
replace that conversion check. Actual new source/record/model/query hashes are
populated and externally anchored only after the new implementation is committed
and the frozen generation/fitting stages produce them; none are invented here.

## Recording-to-query adapter owned by Dart

The producer exports the original fresh parents' valid contiguous observed runs
through generic recording IO, preserving original recording/segment identities,
source-row offsets, channels/units/frames, dt and outgoing applied-command
alignment. Deduplicate parents; do not invent one new recording per query or
create overlapping fake segments. Preserve all runs of selected parents, rather
than cropping to favorable forecasts. Bind exported bytes and the transformation
source to their original sealed parent recordings. This is data preparation, not
an extra simulator run or fit.

Dart loads those archives, checks configuration and ordered state/input channel
identities against `model.contract`, and checks the explicit uniform dt. It finds
exactly one segment containing the full required history ending at source row o.
With C=`history_steps` and local row j=o-segment.start_row, it independently slices
`states[j-C:j+1]` and `inputs[j-C:j]`; no zero padding, gap crossing, prefix beyond
the segment, sorting of channels, unit conversion or resampling is permitted.
Compare these slices byte-for-byte to the sealed query history and reject any
mismatch. The planned future commands are explicit query inputs for outgoing
intervals o through o+H-1. The response branch uses its declared intervened tape;
its factual branch uses `factual_inputs`. If the recording contains those factual
commands, verify them against the tape as an additional provenance check; never
require future observed states to perform a forecast.

Require H=`horizon_steps` and return H observations at offsets dt,...,H*dt, without
repeating the origin observation. The consumer reports source-row origin and
relative forecast time explicitly; it cannot invent an absolute timestamp absent
from generic recording IO. Contract equality is the consumer's responsibility:
equal-shape bare arrays cannot reveal reordered semantic channels.

## Bounded computation and useful output

Load each saved public model once; perform zero fits, updates, initializers,
optimizer steps or simulator calls. For each of the 24 mean inputs:

1. Compute one eager forecast and one ordinary outer-JIT forecast. Save both,
   their dtypes/shapes and public envelope at the same horizon.
2. Compute one outer-JIT JVP of the public forecast with respect to future_inputs,
   using a tangent zero everywhere except future row0/input channel0=1 in that
   declared native command unit. Save the full H-by-state sensitivity array and
   the explicit tangent/channel unit. A tangent is infinitesimal, not a bounded
   finite physical intervention or a reason to clip a query.
3. Emit a per-channel final forecast, envelope width and first-command sensitivity
   with channel names/units; do not combine unlike physical units into one score.

For eight response pairs, also save the eager and compiled forecast differences
(intervened minus factual), tied to both branch hashes and the exact command delta.
The analysis is useful to Dart as a model-inspection tool: it exposes predicted
command effects and computational sensitivities without requiring any controller.
No extra finite-difference sweep or parameter access is needed here; the separate
fixed numerical fixture stage already qualifies general JVP/gradient/FD behavior.

The new qualification runner adjudicates after the consumer exits. Match consumer
forecasts and pair subtraction to the corresponding same-path public outputs;
compare JVPs with the isolated float64 oracle at the same quantized caller inputs
and this explicit tangent. Use the frozen numerical note's normalized scoring and
tolerances externally, where trusted reference norms are available; those norms
are not consumer inputs. Same-path output identity is expected on the pinned
runtime; eager/JIT and default32/oracle comparisons use declared tolerances.
Require finite results, exact shapes/contracts/counts and unchanged model/input
file hashes and public fingerprints. Envelopes are reported provenance, not a
coverage acceptance gate or confidence claim.

## Authentication, negative checks and claim boundary

Freeze the actual consumer source/test hashes and interpreter/dependency/import
map before its first run. The consumer verifies the manifest against the
independently supplied `--expected-manifest-sha256` from the qualification runner,
then verifies all input hashes before loading. Save command, stdout/stderr, exit status,
source/runtime identities, selected-input manifest, result JSON/NPZ hashes and
model fingerprints. Independently replay the same saved inputs through that same
consumer executable without fitting; retain every failed attempt.

Bounded negative checks, on disposable input copies and before any prediction:
wrong model fingerprint, swapped semantic channel order, changed dt, origin/history
crossing a segment boundary, and one altered future-command payload. Raw corruption
must fail hashes; coherently resealed semantic mutations must fail the contract or
source-history checks against an external trusted manifest. Do not claim a
self-consistent archive can authenticate itself against malicious substitution.

Passing establishes a Dart-owned, separately launched, public-only saved-model
forecast/sensitivity integration on this declared input roster. It does not
establish independent third-party authorship, production adoption, physical
Jacobian accuracy, new model accuracy, uncertainty qualification, controller
performance, update quality or arbitrary-system support. Consumer output must be
produced without access to expected output files; a Glassbox internal assertion
that calls the public predictor is not this demonstration.

## Source anchors and remaining freeze work

Source inventory: `public-v4-consumer-source-anchors.json`, SHA256 d0d89c13d760ae2507a33903a3e1b0ce3957de3cdc33b6a31b4f0759e14de2c5.
Dart currently has no Git checkout identity; this proposal pins every current
`src/crazydart/*.py` recursively plus pyproject by file hash. The observed impact
module SHA is `c1493c2cb19cdcfde523a59d0c385ec9b0a746c5dc3953520c9f6025816870e1`;
its pyproject SHA is `7bdaf2746a6d975b7650efb3cb331c793c4a19a7a7e45e2a940aa5601134dfef`.
The inventory also pins the current Glassbox public/IO sources and both fixture
notes; those are design references, not an assertion that v4 already exists.

Before freeze: include this exact roster, tangent, exchange schema, exporter
module identity, consumer ownership/allowed imports and five negative checks in
the new protocol. Commit the new qualification exporter and consumer implementation
before fresh generation. Before consumer execution, bind their source/runtime
hashes, actual model/recording/query hashes and fresh selected identities, and
supply the manifest hash from outside the packet. No additional fitting budget
is allocated to this demonstration.
