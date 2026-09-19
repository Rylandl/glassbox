# Prospective public v4 synthetic scoring policy

2026-09-19. Read-only design and reductions of existing saved arrays. No synthetic
generation, prediction, fitting, initializer, optimizer, ridge solve or new trial.

## Required denominator: fixed original 384-window reference units

The original v1 manifest explicitly scores with each fitted model's training state
standard deviation; `harness._case` reads that learned scale. Public v4 changes
training support from384 to876 legal synthetic windows, so its learned scales can
change even if its physical errors do not. Required new capability scoring must
not use those candidate-dependent scales.

Prospectively, require all unchanged numerical caps using the per-case fixed
`norm_state_scale` of the authenticated adopted v3 384-window reference archives.
This is an explicit scoring-policy change for the new public qualification,
not a claim that v1 always used independent fixed units. It preserves the original
reference's normalization units and cap values, without letting a new model change
its own acceptance denominator. The new model still fits with its natural876-window
norms; no training or prediction quantity is replaced by the scoring reference.

Exactly27 cases,51 case/regime rows,255 finite horizon cap checks (five per regime)
and three physical delayed-input probes are required on ordinary default32 public
predictions. Score saved predictions against the unchanged float64 truth using
NumPy float64 reductions. No case, horizon, failed fit or nonfinite output is
removed; missing/extra/duplicate roster elements fail. Each required comparison is
inclusive <=. The x64 path and original candidate-scaled diagnostics are separately
reported under the prospective precision policy; they are not fallback candidates.
No cap, normalization, seed, dtype, cache or checkpoint tuning after observation.

## Independent reconstruction and source authentication

All27 required scale vectors and their old caches are already available locally:
`artifacts/2026-09-17/slow-sampling/<family>-<seed>/model.npz`.
Their exact bytes are externally pinned by the committed
`docs/harness/generic-public-api-v1.json`, SHA256
`a6c966dbc5940121b187aaf7552660f1e0293a9bcaec0f12ddfcc098ea49c77f`.
The entire slow-sampling subset comprises88 pinned files:27 model archives,
51 regime tapes, three probes and seven root metadata/decision/reference files.
All88 hashes were checked; all27 archived scales independently reconstruct exactly
from their saved training caches, with N384,C10,H5. The check used NumPy2.5.3;
no Glassbox import, model loading/prediction or initializer was needed.

For each case, reproduce precisely:

```
complete_x = concatenate((old_train_past_states, old_train_future_states), axis=1)
current = complete_x[:, 10:-1, :]       # shape [384,5,d]
raw = current.std(axis=(0,1), ddof=0)
s_ref = where(raw > 1e-8, raw, 1.0)
```

These are the current observations at each forecast transition, not all history
rows, next-state targets alone, development/evaluation data, or deduplicated native
transitions. Overlapping selected windows retain their original multiplicity.
Require finite positive scales of the exact declared channel shape and byte-exact
agreement with the authenticated archived scale. Missing anchors or disagreement
is an integrity failure, never permission to substitute candidate scales.

The attached anchor ledger contains all88 expected hashes and all27 scale vectors,
array-byte hashes, model fingerprints and reconstruction results:
`public-v4-synthetic-scoring-anchors.json`, SHA256 fa03c4fa7376f2e6e1d6cb4536c4d4b2cbd11cc316eb9e854f04b0f70a9f2ccc.
The artifact root is relocatable; committed relative-path hashes define trust.
The scalar-only `docs/harness/reference.json` has no scale vectors and cannot alone
authenticate the denominator. Its SHA256 is
`ee2aaeeaed0d4b102d519926cadf7209c9c6878d0916a64cc6d586311c7ad213`.
The frozen v1 manifest SHA256 is
`1ba15b3f466e91edf548f1d459a52eb0896310e7fbbb6c1c98544ec10513d781`.

Before each new fit, regenerate only the unchanged frozen calibration recordings
under the committed qualification protocol and independently reconstruct the old
whole-recording split/hash-ordered384-window selection. Compare its keys, source
origins, dt and four cache arrays against the authenticated old training cache;
compare the fixed256 development cache too. The new876-window training extraction
must preserve that exact384 prefix. This is preparation/replay arithmetic, not an
old public fit or an initializer. Any generation/runtime mismatch fails visibly;
no regeneration retries or relaxed denominator matching are allowed. The required
score uses the already authenticated fixed vector, not a refitted/reference model.

## Exact cap formula and preserved diagnostic flags

For squared physical errors E[q,h,c], reduce `M[h,c]=mean_q(E)`. Required scores are
`sqrt(mean_c(M[h,c]/s_ref[c]**2))` for each of the five horizons. The overall
fixed-reference score is `sqrt(mean_(h,c)(M[h,c]/s_ref[c]**2))`; preserve per-channel
physical RMSE and per-recording reductions as well. Use the same case's s_ref for
matched and shifted regimes and for both inference precisions.

Save candidate learned `state_scale`, reference `s_ref`, their per-channel ratio,
and both scoring tables. Separately compute the literal original candidate-scaled
metrics using its learned scale, then the original v1 cap/probe/reference decisions
with their original operators, including `old_overall*1.05+.005`. Label these
`legacy_candidate_scaled` diagnostics. Their historical reference/coverage flags
must not silently enter the new required-capability conjunction. Existing saved
v1 results/accepted flags and all historical protocols remain byte-unchanged.

A fixed-reference overall comparison against the old reference scores may be
reported with an explicit fixed-unit label, but it is not an additional promotion
veto. Do not relabel it as the literal candidate-scaled historical rule. Keep the
new `fixed_reference_absolute_capability_pass` distinct from either historical
`accepted` or the old `with_evidence` conjunction. Existing `decide` allows an
arbitrary nonempty horizon array, so new validation must explicitly enforce five
entries per each of51 rows, finite/nonnegative overall scores and the full roster.

## Delayed-input probes: preserve their physical requirement

The probe is different: `harness.paired_rmse` computes
`sqrt(mean((prediction-target)**2, axis=(branch,channel)))` without state scaling.
The frozen first-step cap is0.05 in the observed unitless state coordinate. Keep
that exact required inequality for each seed101/202/303: finite physical probe
RMSE at step0 <=0.05. All saved branch predictions must be finite; the complete
five-step probe is reported, but later steps receive no invented new cap.

For a consistent fixed-reference-unit presentation, additionally report
`probe_scaled[h]=probe_physical[h]/s_ref[0]` and
`probe_scaled_limit=0.05/s_ref[0]` for that same witness case. Scale BOTH the value
and cap; dividing only the value and still requiring0.05 would silently tighten
this challenge by roughly fivefold. The physical inequality is authoritative to
avoid an avoidable rounding disagreement between two algebraic presentations.
The three pinned scales and corresponding scaled limits are in the anchor ledger.
The old blind-floor construction and original probe flags remain unchanged.

## Prospective implementation obligations

Pin this scoring policy, its reference ledger and unchanged generation/selection
sources before implementation/fitting. Replay reconstructs fixed scales directly
from authenticated cached arrays and all scores from saved predictions; it never
fits or solves. Include one meaningful denominator-substitution challenge: change
candidate learned scales or substitute them into an otherwise valid score record;
the independent fixed-reference reduction must reject a changed required metric.
Do not rewrite the historical harness or add a consumer normalization option.
