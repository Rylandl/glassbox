# Prospective public v4 lifecycle fixture roster

2026-09-19. Read-only design from `tests/test_default_model.py`,
`tests/test_public_api.py`, `tests/test_sequence_collection.py`, `learner.py`,
`recordings.py` and `_learner_arrays.py`. No fixture, fit, initialization, prediction
or trial was executed. This supplements the two public-v4 numerical design notes.

## Exactly three successful public fitting operations

Use one base fit, one immutable update, and one all-constant fit. Run each at the
unchanged full recipe budget once; save all three outputs and reuse them throughout.
Do not repeat fits for each persistence, shape, duplicate, dtype or error test.

All recordings below have dt=.25, six observation rows, five input rows,
segment_id=`whole`, start_row=0, d=3, m=2. Thus the existing timing rule gives
context2, delay1, horizon1 and exactly three legal origins {2,3,4} per recording.
Configuration is `public-v4-small-v1`; channel identities are
`x0 [unitless,fixture]`, `constant_x [unitless,fixture]`,
`x2 [unitless,fixture]`, `u0 [unitless,fixture]`,
`constant_u [unitless,fixture]` in their respective state/input tuples.

For r=0,1,2 define U_r[t]=[.25*sin(.7*(t+1)+r), -.5], t=0,...,4.
Set X_r[0]=[.05*r,2,.03*r] and recursively:
`X[t+1,0]=.8*X[t,0]+.1*U[t,0]`, `X[t+1,1]=2`,
`X[t+1,2]=.6*X[t,2]+.05*X[t,0]+.02*U[t,0]^2`.
Compute these arrays in NumPy float64 without RNG.

1. **M0:** public fit of `small-a` (r0) and `small-b` (r1), in a fresh process
   starting with ambient x64=False. Automatic holdout selects one recording for
   each role; derive the expected IDs by the pinned hash rule, not manually
   assigning them. Assert Ntrain=Ndevelopment=3, three distinct origins, no padding,
   actual full-gradient batch3, ridge=.03, 3,000 returned gradient-window visits
   on completion, and current v4 archive/recipe identities. Constant state/input
   channels must have scale1; constant state delta scale is 1e-4; zero product
   columns have scale1. Require all weights/scales/model/envelope values finite.
2. **M1:** load M0, call public `update` once with `small-c` (r2), starting
   ambientFalse. This absorbs three fresh windows: Ntrain=6, Ndevelopment=3,
   ridge=.06 and 6,000 gradient-window visits. Development keys, origins and all
   four arrays remain exact; training content consists of old/fresh keys with
   deterministic merge order. The source ledger gains exactly `small-c`.
3. **MC:** public fit of two constant recordings `constant-a` (r0) and
   `constant-b` (r1), starting ambientTrue. For every row use
   X=[1+r/8,2+r/4,3+3*r/8], U=[1/4+r/8,-1/2]. The two contents differ, so they
   genuinely satisfy the duplicate-content check while each role is constant.
   All channel scales are1, delta scales1e-4, e0=0, raw weights10,000, normalized
   weights1. This uniquely tests the exact zero-loss branch: selection0, finite
   zero losses, 1,000 rejected proposals, zero accepted proposals and 8,001
   acceptance-objective calls. Zero residual/envelope is allowed. Any ratio with
   zero denominator must be explicitly unavailable rather than NaN/Infinity.

M0 combines minimum N, coarse timing, asymmetric dimensions and constant channels.
MC is justified separately because mixed channels do not exercise an entirely
zero objective and strict-decrease rejection. It is not an accuracy test or a
reason to tune the optimizer. If it fails, preserve the failure; do not add a
different constant fixture or reduce the public step count.

## Reuse M0/M1 for lifecycle checks

- Capture M0 fingerprint, full public report/contract/recipe/envelope, cache
  identities and arrays, source-file hash and one saved query prediction before
  update. After M1 succeeds or a later update fails, all predecessor facts and
  same-path predictions remain exact. M1 has a different fingerprint and its
  `previous_revision` equals M0. Do not demand better accuracy or changed mean
  coefficients merely to prove that a new revision was created.
- Independently recompute calibration from each model's own float64 development
  predictions and the pinned three-window cache. Rank=min(ceil(4*.9),3)=3. Require
  exact order-statistic envelope, correct physical shape and nonnegative finite
  widths. Do not require M1's envelope to differ numerically from M0; it may
  legitimately coincide. Calibration assertions run in the fit64 context;
  default32 empirical coverage is a separate measurement.
- Save/load M0 and M1, requiring exact metadata, parameter/norm/cache/envelope
  arrays and model fingerprints; compare predictions in the same execution path.
  Load both in fresh ambientFalse/True processes without fitting. Capture dtype
  and the inference/AD checks from the numerical roster; do not refit per dtype.
- Mutate returned report, contract, recipe and envelope copies; the revision
  fingerprint remains unchanged. Mutating original arrays after constructing a
  SequenceSegment cannot change its arrays; segment/cache data are read-only.
  This does not promise protection against deliberately editing private `_model`
  attributes, which are outside the consumer surface.
- Check root exports and signatures and the existing fresh-process deferred-import
  guard. Import/load/predict may not pull in experimental/controller/structured
  modules. No new public option or historical-format migration path is introduced.

## Rejections without numerical fitting

Instrument the private fitting entry to raise an unmistakable sentinel if called
for any of these invalid cases; expected validation must happen first. This is
test-only instrumentation, not an alternate fitter.

| Fixed mutation | Required outcome |
| --- | --- |
| Pass a list instead of SequenceCollection | Type error before initialization |
| Fit only small-a | Insufficient independent recording roles |
| Truncate both small records to five states/four inputs | Exactly two legal origins per role; reject without padding |
| Rename an exact small-a copy to `same-content` and fit together | Duplicate recording content rejected |
| Update M0 using ID `small-a` but r2 arrays | Reused identity rejected even though content differs |
| Update M0 using `fresh-name` but exact small-a arrays | Reused content rejected despite new identity |
| Update M1 with `again` carrying exact small-c arrays | Fresh content was added to the persisted ledger |
| Update with changed configuration, reordered state names, reordered input names, changed unit text, or dt=.2 | Contract mismatch rejected (one mutation per case) |
| Missing configuration/channels on an otherwise valid collection | Required public facts rejected |

Use constructor-only tests for X NaN, U Inf, row-count mismatch, invalid dt
(0,-1,NaN,Inf), invalid start_row (-1,True), duplicate segment identity, overlapping
segments, mixed intervals, duplicated channel names and width/name mismatch.
Retain the existing mask-gap test with its fixed arange16 data and invalid rows6:8;
it proves extraction never bridges a gap without another fit.

From M0's saved query, test these shape mutations in eager and outer-JIT tracing:
rank1; mismatched array ranks; incorrect state width; incorrect past/future input
width; mismatched batch counts; insufficient context; unaligned state/input row
counts; zero future steps; two future steps when maximumH=1. Require a clear
ValueError/TypeError, not a low-level compiler failure or a padded result.
Semantic column reordering of bare equal-shape prediction arrays cannot be
detected by this signature: require the consumer to check the declared contract.
Do not claim that the model infers signal names from values.

Nonfinite recordings are rejected at construction. Nonfinite *runtime prediction*
values require an explicit documented domain contract: do not silently require
Python value checks inside arbitrary JIT tracers or add host callbacks/checkify
as incidental loader work. These invalid queries cannot be admitted to physical
scoring as finite successes; failure must leave the revision unchanged.

## Persistence defects: no fits

Modify disposable copies of M0 only. First mutate one parameter, one cache target,
and one metadata contract value without updating the fingerprint; all three must
fail integrity verification. Then coherently recompute the archive fingerprint
for each semantic defect separately: predecessor v3 format/recipe; wrong current
recipe constant; missing envelope; negative envelope; wrong envelope shape;
missing one quadratic parameter; wrong parameter shape; nonfinite parameter;
nonpositive normalization scale; unsupported model format. Require refusal at
load. This distinguishes checksum verification from schema/meaning validation.
The current loader has only partial structural checks; these are prospective v4
requirements, not claims that every case already fails today. Never mutate the
saved original or use coherent resealing as a claim of cryptographic authenticity.

## Precision-scope restoration with no extra complete fits

The three real operations already cover successful fit at ambientFalse/True and
successful update at False. Run all load/predict/shape-negative paths from their
archives in both modes. Require caller global configuration and environment to
remain unchanged, including failed calls.

For owned-fit error restoration, use bounded test-only failure injection at two
places: (a) initializer entry, asserting x64=True before raising a sentinel,
(b) calibration entry, returning M0's already-saved core result from a stub fitting
entry, then raising a sentinel while asserting x64=True. Test both fit and update
under initial False and True. These do no real optimization and count only as
scope/control-flow tests, not numerical qualification. Update failures must not
modify M0/M1 or expose a partial revision.

For thread isolation, pause the initializer sentinel with an Event inside its
owned scope (bounded timeout5s). An independent thread in the default-false
process must still observe False; after release/failure the fitting thread must
restore False. Do not mutate JAX globals to make this test pass. The standard
caller-owned True context used by a test also restores its enclosing state.

## Bounded actual-fit JIT/AD/FD roster

Use the fresh frozen evaluation queries, not current prediction/error results.
The existing wire schema has kind `factual`/`response` and scopes `primary`,
`heading_shift`, `maneuver_shift`, `speed_shift`, `wind_shift`; response arrays
contain `future_inputs` and `factual_inputs` in the same NPZ.

Per simulator, map the four declared non-primary scopes to `shifted`, forming
four strata: primary/factual, primary/response, shifted/factual, shifted/response.
At the model's maximum fitted horizon, retain queries with history_eligible=True,
exact contract shapes and finite stored `past_states`, `past_inputs`, and
`future_inputs`; response also requires finite full `factual_inputs`. Sort by the
tuple `(parent,id)` using ordinary case-sensitive lexicographic string order and
select the first and last distinct identity. Reject an unknown scope/kind rather
than silently classifying it. If a stratum has fewer than two eligible identities,
report insufficient evidence; do not replace it with shorter-horizon/easier data.

Do not inspect/filter `target`, `factual_target`, `valid`, fitted predictions,
errors, gradient magnitudes, or outcomes after float32 conversion. Input-complete
does not imply future truth-complete. Save the selected identities and input-file
hashes before inference; reproduce selection during replay.

This gives eight queries per simulator, including four response pairs, hence
12 individual mean-input cases per simulator and 24 across both. Run the numerical
note's mandatory eager/JIT/JVP/reverse-grad/FD checks for every mean-input case,
retaining both response branches even if a factual branch can be computationally
cached. Pair subtraction is checked from those outputs with its own fingerprints;
no new learner fitting is required. Physical forecast/response scoring and replay
continue to use the full fresh query roster, not this derivative subset.
