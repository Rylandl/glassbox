# Generic learner: engineering decisions and current work

This is the current execution record for generic Glassbox work. Read it before
resuming after a new session or context compaction. Individual investigation
pages preserve evidence; their historical “next experiment” paragraphs do not
set the active agenda. The latest user instructions take precedence.

## Direction

Build one maintained identification pipeline that learns a useful differentiable
model from limited, informative recordings, with a small fit/predict/update
interface and evidence about where its predictions are supported. The caller
supplies signal identities, units, timing, and recording/configuration facts.
Architecture, optimization, regularization, and selection are internal algorithm
decisions. Generic does not require an absence of assumptions: causality,
temporal state, controlled complexity, and observation noise are legitimate
general modeling structure. Vehicle-specific force laws are not required inputs
to this generic path.

The engineering baseline is the recursive affine-plus-nonlinear sequence
model and its fixed consumer workflow. Since M2 the maintained recipe is
`generic-memory-v2-prototype`, which adds a causal memory over a 500 ms
in-recording context to the retained `generic-history-v1-prototype`. It remains
experimental, but it is the incumbent we improve. A model need not dominate
every alternative on every recorded metric to become the incumbent. Stable
structured `glassbox.fit` remains a separate supported path until an explicit,
versioned integration.

**Process correction:** counting better/worse cells is descriptive, not a
promotion decision. Requiring universal improvement freezes development.
Repeatedly finding a tradeoff without deciding its consequence is not progress.

## M1: dependable fitting of the chosen model

Status: **bounded comparison complete; retain the incumbent**. The acceptance
runner and internal bounded fitter are implemented. The tested replacement
failed qualification; dependable fitting of the incumbent is still unproven.
This is a completed implementation decision, not a claim of improved accuracy.

Keep the observation representation, history contract, model capacity, and
loss definition fixed during M1. Start from `generic-history-v1-prototype` and
its learned affine initialization plus recursive nonlinear correction. The
first-step-floor model remains an archived challenger, not a second selectable
consumer recipe. Do not open a representation or architecture search to address
an optimization or model-selection defect.

M1 delivers an improved, bounded fitting implementation and one reusable
offline acceptance runner. The fit should have an explicit optimization budget,
measured progress/convergence information, and a reproducible selection outcome.
“Ran 1,000 Adam updates” is not evidence that the chosen function class was
adequately fitted. A budget limit or unresolved selection ambiguity must remain
visible in the fit evidence. This does not require a new consumer option.

Implementation order:

1. Freeze the acceptance manifest once, using existing datasets, reference
   artifacts, and error-budget work. Identify required cases, diagnostic stress
   cases, one primary quality measure, absolute error caps, calibration budget,
   and compute budget. Populate numerical values before fitting a challenger.
   Performance caps are engineering targets with a rationale; they must not be
   advertised as universal task-readiness thresholds. Missing values make the
   acceptance definition unfinished, not the current model silently acceptable.
2. Route the engineering work through the maintained sequence fitter. Reuse
   existing window contracts, scaling, serialization, and evidence owners.
   Implement and diagnose the fitting change there, with meaningful regression
   tests. Do not create another long-lived copy of its Adam loop in an experiment.
3. Run one matched incumbent/challenger comparison through the acceptance
   runner. Preserve per-case failures and absolute errors alongside the primary
   score. Fix defects within this implementation and rerun affected checks.
4. Close M1 with a recorded retain-or-replace decision. If accepted, version the
   recipe and preserve old artifact loading and prediction behavior. If not,
   retain the incumbent and name the unmet requirement and demonstrated cause.
   Do not substitute “some better, some worse” for the decision.

The [M1 manifest](generic-fit-acceptance.json) was frozen before candidate fits.
Its digest is `c2c2f157da0c1b2d973b8d6a38d0ab08e2caeb3f8713138491cf4973738d6ae4`.
The nested `dataset` object retains the original generator plan verbatim as
provenance, including historical candidate prose and seeds. The top-level
`data_seeds`, `candidate`, `compute`, and acceptance fields govern M1.

### M1 decision, 2026-09-15 UTC

**Retain `generic-history-v1-prototype`.** A full-batch L-BFGS replacement with
160 actual objective/gradient calls scored **1.093809 candidate/incumbent** on
the frozen primary score; replacement required at most **0.95**. The shifted
dead-zone case at seed 6101 also exceeded its per-horizon error cap:
**0.550807 versus 0.55**. Relaxing that small breach would not change the failed
aggregate decision. No thresholds, training settings, or candidate fits were
changed after seeing these scores.

The same recurrence, 32-unit nonlinear correction, initialization, normalization,
100 ms history, 250 ms forecast, original loss, and cached training/development
observations were held fixed. Each required family has three seeds and two
evaluation regimes. Two stress families and one previous regression add hard
guards, not extra primary votes. All 25 candidate fits used 61,440 training
window-gradient evaluations and 11 development passes, versus the incumbent's
64,000 training window-gradient evaluations. This accounting is not equal
wall-clock time: candidate fit times were 0.281–0.527 seconds on the recorded host,
including the fitter's compilation, excluding the later evaluation audit.

| Required family | Primary error ratio | Selected training MSE ratio | Selected development MSE ratio |
| --- | ---: | ---: | ---: |
| Stable affine | 1.000 | 0.023 | 0.035 |
| Coupled nonlinear | 1.118 | 2.028 | 2.170 |
| Dead zone and saturation | 1.065 | 0.750 | 0.908 |
| Hidden hysteresis | 1.077 | 0.708 | 0.772 |
| Near periodic | 1.055 | 1.222 | 1.028 |
| Off periodic | 1.266 | 1.486 | 1.134 |

Ratios are geometric means, lower is better. Primary ratios use the predeclared
0.005 RMSE floor; the affine predictions were already below it. Training and
development columns use the unchanged fitting MSE without that floor and are
diagnostics, not extra selection criteria. The extra seed 5101 is excluded here.

What the saved evidence establishes:

- All 25 solves reached the objective-call budget; none reached the declared
  gradient tolerance. This procedure did not establish convergence. It does not
  prove that increasing the budget would improve reserved-data predictions.
- On coupled nonlinear and off-periodic systems, the final accepted iterate
  was selected on all three seeds and still had higher training loss than the
  incumbent. The incumbent parameters exhibit a better fit within the same
  function class. Capacity and checkpoint selection cannot explain that
  particular optimization gap.
- On hysteresis, lower training and development loss accompanied worse reserved
  predictions. Optimization progress is not sufficient evidence of
  generalization. The experiment does not identify a unique cause for this gap.
- Development selected the final accepted point in 14/25 fits; the two
  development recordings preferred different saved points in 10/25. Those are
  selection diagnostics, not calibrated uncertainty or permission to select
  using evaluation targets.

### Implemented and verified

The internal [`_sequence_fit.py`](../src/glassbox/experimental/_sequence_fit.py)
enforces the objective budget inside line search, uses scoped float64, retains
finite accepted checkpoints on numerical failure, and separates optimizer
termination from the selected model's gradient and development evidence. It is
an internal retained implementation for reproduction, **not a consumer solver
option or the default**. SciPy is declared directly; its resolved versions were
already present through JAX.

[`check_generic_fit.py`](../scripts/check_generic_fit.py) runs the fixed
comparison or verifies a saved run without fitting. It preserves source hashes,
input hashes, recording roles, model fingerprints, optimizer traces, per-channel
and per-horizon errors, and per-recording scores. Invalid scores, incomplete
cases/regimes, and invalid resource counts fail closed. The manifest digest is
enforced before any fitting, so changing its path cannot silently relax a gate. NumPy replay independently
checks both models' predictions and recomputes the decision and fit losses.

Validation: **47 distinct focused tests passed** for the fitter, decision rules, causal
replay, existing consumer fit/predict/update, serialization, and recording
contracts. Independent replay checked **25 cases / 100 model-regime predictions**.
Deliberately altered predictions, input caches, and manifests were rejected.
The wheel built and passed a fit/predict/save/load/update smoke test using its
packaged modules. All 82 library files present at the start retain their source
hashes; the internal fitter is a new module. No existing recipe, revision, or
consumer signature changed.

The lockfile's direct dependencies were structurally validated and the wheel
metadata built successfully. `uv lock --offline` itself could not run: the local
uv executable panicked in macOS system-configuration initialization. The existing
locked SciPy branches were added to Glassbox's direct dependency edges without
resolving or upgrading packages. This limitation is distinct from the passing
build and packaged-artifact smoke checks.

Machine-readable evidence: [summary](investigations/generic-fit/m1-summary.json)
and [portable saved run](investigations/generic-fit/m1-evidence.zip). The archive
includes exact input caches, outputs, executed sources, later verifier sources,
tests, and verification results. From the Glassbox repository, extract it to a
new local directory and verify:

```sh
python -m zipfile -e docs/investigations/generic-fit/m1-evidence.zip /tmp/glassbox-m1-evidence
python scripts/check_generic_fit.py --verify /tmp/glassbox-m1-evidence/acceptance-01
```

Verification uses no optimization and needs no historical parent-workspace
artifact paths. Those paths remain provenance keys for the original input hashes.
The manifest and executed source archive are immutable; verifier hardening and
cache preservation happened afterward and are recorded separately. These are
synthetic engineering regressions on previously inspected system families, not
untouched confirmation, platform validation, or task-accuracy guarantees.

## M2: one causal memory contract for demonstrated history limitations

Status: **closed; replace**. The `generic-memory-v2-prototype` recipe passed
the frozen M2 contract and is the maintained default. Old
`generic-history-v1-prototype` artifacts load, predict, and update with their
own recipe unchanged. This addresses G05 for the declared information budget;
it is not a general memory-adequacy claim and does not touch G11.

### Contract

`SequenceModel(kind="filter_mlp")` in
[`sequence_model.py`](../src/glassbox/experimental/sequence_model.py) keeps the
incumbent's explicit 100 ms difference history and adds an eight-coordinate
memory. The memory starts at rest at the first consumed observation, advances
once per observed transition inside one recording segment, continues from
predicted observations during the forecast, and composes exactly when carried
within a recording through `memory_state` and `rollout(..., memory=...)`.
Forecasts from rest consume exactly `history_steps + 1` observations; nothing
earlier is implied, and windows never bridge a segment boundary. The readout of
the rest state is zero, so checkpoint zero is the incumbent's affine
initialization rule on the same forecast rows.
[`test_causal_memory.py`](../tests/test_causal_memory.py) checks the rest start,
carried-memory composition at three split points, segment boundaries, causal
prefix consistency, an independent NumPy replay, initialization parity with the
delay model, unchanged contracts for other kinds, and that the memory recovers
the G05 delayed response while explicit history is pinned at its floor.

The consumer `fit`, `predict`, and `update` signatures are unchanged. A saved
model carries its recipe; `update` refits that recipe, so an older revision's
lineage never changes silently. A golden v1 artifact produced before the change
is a fixture: its fingerprint, predictions, and v1 update path are tested.
Archived research scripts reproduce through the retained v1 entry point.

### Frozen budget and decision, 2026-09-16 UTC

The [M2 manifest](generic-memory-acceptance.json), digest
`02ac887c28eae15efa9afc9168ed5d02998596e3e86680644b59382f7ef2b1b7`, was frozen
before any benchmark candidate fit. Information budget: ten 50 ms transitions of
context, the incumbent's two-step explicit history, eight memory coordinates,
one fixed choice with a stated rationale and no menu. The candidate uses the
incumbent's Adam recipe through the maintained sequence fitter, the incumbent's
frozen M1 window identities whose origin lies at least ten rows into their
segment (363 of 384 training and 244 of 256 development windows per case, no
replacements), and the M1 evaluation origins with a complete context (116 of
124 rows per regime), scored identically for both models with the incumbent's
M1 state scale. Calibration and evaluation recordings were regenerated from the
archived generator and reproduced the M1 caches to 5.1e-16.

Gates: M1's absolute caps on every family; both delayed families must at least
halve the incumbent's aggregate error; the G05 paired probe must reach at most
0.05 first-step paired RMSE on every seed; ordinary families may lose at most 5%
in aggregate; the eight-family primary ratio must be at most 0.95. The
`hidden_input_delay` family is the G05 witness itself, with the incumbent refit
through the frozen v1 consumer recipe on the recording host.

**Replace.** Primary ratio **0.488579** (limit 0.95); ordinary guard
**0.923712** (limit 1.05); no hard or capability gate breaches.

| Family | Candidate/incumbent ratio | Note |
| --- | ---: | --- |
| Stable affine | 1.000 | Both below the 0.005 floor |
| Coupled nonlinear | 0.967 | |
| Dead zone and saturation | 0.873 | The M1 breach case is inside its cap |
| Hidden hysteresis | 1.072 | Matched error rose on two seeds; worst horizon 0.0905 versus cap 0.12 |
| Near periodic | 0.922 | |
| Off periodic | 0.745 | |
| Delayed nonlinear (capability) | 0.102 | Matched 0.39–0.41 to 0.015–0.018; shifted 0.79–0.92 to 0.19–0.23 |
| Hidden input delay (capability) | 0.051 | Matched 0.60–0.62 to 0.027–0.035 |

Paired-probe first-step RMSE on the G05 witness: **0.00249, 0.00252, 0.00374**
versus the incumbent's 0.20003, 0.20017, 0.20000 and the construction floor of
0.2. The incumbent predicts identical branches; the candidate separates them.

All 28 candidate fits used 64,000 training window-gradient evaluations and 11
development passes, equal to the incumbent's window accounting; each candidate
window also carries eight memory updates, so this is not equal arithmetic. Fit
times were 0.41–0.60 s on the recorded host. Development selected the final
checkpoint in 16 of 28 fits and checkpoint zero or 100 in four, all in the
noisy-observation and extra near-periodic cases where the memory could not
help. The hysteresis tradeoff is accepted and recorded, not hidden: a latent
accumulating coordinate is not a delayed input, and the memory did not improve
it. The witness incumbent's macOS qualification fingerprints were not reproduced
bit-for-bit on this Linux host, but its selected checkpoints, paired-probe
errors, and physical forecast errors matched the recorded values.

### Implemented and verified

[`check_generic_memory.py`](../scripts/check_generic_memory.py) runs the frozen
comparison from the preserved M1 evidence or verifies a saved run without
fitting. The verifier replays both models with independent NumPy recurrences,
recomputes every score, the selected checkpoint's full-batch training and
development losses, cache hashes, the probe, and the decision. The run executed
once with the pre-hardening verifier and once with the final sources; all 28
candidate and incumbent fingerprints and the decision were identical, so the
preserved run is a byte-for-byte reproduction. Altered predictions, training
and development caches, probe predictions, a model, the manifest, a result row,
and the decision were each rejected.

Validation: the M1 saved run still verifies after the change, so old artifacts
and fingerprints are untouched. The focused generic suites and the full
non-slow repository suite pass; counts are in the
[summary](investigations/generic-fit/m2-summary.json). Machine-readable
evidence: that summary and the
[portable saved run](investigations/generic-fit/m2-evidence.zip). From the
Glassbox repository:

```sh
python -m zipfile -e docs/investigations/generic-fit/m2-evidence.zip /tmp/glassbox-m2-evidence
python scripts/check_generic_memory.py --verify /tmp/glassbox-m2-evidence/acceptance-02
```

These are synthetic engineering regressions and one constructed delay witness,
not platform validation, calibrated uncertainty, or evidence that a 500 ms
context suffices for any particular vehicle.

## Acceptance policy

Separate three kinds of checks:

| Check | Decision |
| --- | --- |
| Correctness and contract invariants | Hard gates: causal indexing, finite outputs, data-role separation, artifact compatibility, and declared units/timing |
| Required benchmark capabilities | Predeclared absolute error and resource limits; a breach cannot be hidden by an aggregate gain |
| Diagnostic stresses and secondary scores | Explain limits and tradeoffs; a small regression inside the accepted envelope is not an automatic veto |

Examples with deliberately indistinguishable histories or unexcited inputs
test information limitations. Do not demand impossible point accuracy there
or count it as a numerical fitter defect. Equally, do not silently remove a
real requirement by relabeling a failing case “unsupported.” Change the
requirement version explicitly if the intended capability changes.

Use a fixed primary score for choosing between candidates that pass the hard
gates, and an explicit tie rule favoring the simpler incumbent when gains are
not meaningful. Avoid equal votes for many correlated windows or repeated seeds.
The manifest must specify grouping and weights; a single physical system with
many recordings must not dominate merely through sample count.

The benchmark maintainer owns thresholds; ordinary users do not configure
them. Do not retune thresholds after inspecting the candidate. New requirements
create a new manifest version. Previously inspected datasets are regression
tests, even if a new seed or filename is used; describe fresh seeds as fresh
realizations, not new system families or untouched platform evidence.

An algorithm change may improve the primary objective while worsening some
secondary values inside the declared limits. Accept and document that tradeoff.
All candidates still need the same observation and compute accounting. Fit
quality, error calibration, and end-use adequacy are distinct claims.

## Decision ledger

“Closed as a standalone fix” preserves the implementation and result. It does
not mean the component can never be useful inside a different, justified design.

| ID | Decision and status | Evidence / reopening condition |
| --- | --- | --- |
| G01 | **Committed:** one causal recursive model with learned affine structure and nonlinear correction is the engineering incumbent. | [Model structures](model-structure-experiments.md), [fixed workflow](default-recipe.md). An architecture change must address a named failure the current representation cannot express, after checking fitting quality. |
| G02 | **Committed:** coherent recurrence and one consumer recipe. Independent horizon heads and representation menus remain diagnostics. | [Scope](scope.md), [forecast representations](forecast-representation.md). Reopen only with a demonstrated need and a coherent executable prediction contract. |
| G03 | **Closed as a general replacement:** full, additive, and pairwise kernel corrections were compared. Full RBF helped several direct forecasts; additive/pairwise structure did not establish reliable extrapolation. | [Model structures](model-structure-experiments.md). New kernel names or seeds are not new mechanisms; a materially different information or modeling assumption must be stated. |
| G04 | **Resolved in M2:** the earlier eight-coordinate latent model encoded only the same 100 ms window and did not beat explicit history. The accepted memory is a causal filter over ten consecutive in-segment transitions with a matched Adam budget, so it carries information the window does not. | [Model structures](model-structure-experiments.md), [M2 decision](#m2-one-causal-memory-contract-for-demonstrated-history-limitations). A new memory design needs a new manifest version and a named information requirement; do not sweep context lengths or memory widths. |
| G05 | **Addressed for the declared budget:** the witness that a fixed 100 ms history cannot identify is resolved by the 500 ms consumed context. A delay beyond that context remains unidentifiable from the supplied observations; the limitation is now the declared information budget. | [Qualification witness](model-qualification.md), [M2 decision](#m2-one-causal-memory-contract-for-demonstrated-history-limitations). Changing the budget is a new manifest version with a demonstrated requirement, not a tuning knob. |
| G06 | **Established limitation:** low logged-policy forecast loss and marginal input ranges do not prove identified input response. | [Qualification](model-qualification.md), [diagnostics](sequence-diagnostics.md). Informative variation is a data requirement; unexplained input residuals alone do not prove it. |
| G07 | **Closed inference:** older-history diagnostic gain is not proof of missing state. Nonlinear short-context features can explain the gain in a fully observed Markov system. | [History confounding](history-confounding.md). Retain alternative explanations; no automatic history-length increase from this diagnostic. |
| G08 | **Closed as a default:** pooled horizon normalization. Its cycle improvement comes with broader regressions. | [Horizon generalization](horizon-generalization.md). Reopen only when a changed acceptance contract or demonstrated mechanism makes the tradeoff relevant, not to rerun fresh-seed counting. |
| G09 | **Retained challenger, frozen for M1:** first-step loss-scale floor. It improves several near-periodic cases and preserves many objectives exactly; it also has known regressions. | [Generalization](horizon-generalization.md), [checkpoint attribution](checkpoint-attribution.md). Compare against the engineering acceptance contract when needed; do not rediscover these effects with another sweep. |
| G10 | **Closed as standalone fixes:** swapping the two tested checkpoint criteria, always taking the final checkpoint, or mandating 16 development recordings. None is a universal remedy. | [Checkpoint attribution](checkpoint-attribution.md). Broader data sometimes helps. Reopen a specific policy only with a named failure and a changed mechanism, not a guarantee inferred from recording count. |
| G11 | **M1 decision closed; incumbent retained:** the fixed-budget L-BFGS replacement fails the frozen acceptance contract. Optimization and generalization remain distinct unresolved limitations. | [M1 decision](#m1-decision-2026-09-15-utc), [checkpoint attribution](checkpoint-attribution.md). Reopen this solver design only for an identified implementation defect, a materially different optimization mechanism supported by existing traces, or an explicitly changed resource requirement. New seeds, a larger budget alone, or a different checkpoint count are not an unrecorded retry. |
| G12 | **Committed:** whole-recording provenance, separate data roles, observation-budget accounting, immutable revisions, and evidence tied to its fitted revision. | [Fixed workflow](default-recipe.md), [recording selection](recording-selection.md). Keep these when changing the fitter; do not add another wrapper hierarchy. The v2 recipe keeps all of them. |
| G13 | **Committed:** `generic-memory-v2-prototype` is the maintained default recipe; recipes are versioned, a saved model carries its own, and v1 artifacts load, predict, and update unchanged. | [M2 decision](#m2-one-causal-memory-contract-for-demonstrated-history-limitations), [golden v1 fixture test](../tests/test_default_model.py). A default change is a new manifest version with regression evidence; it must never alter a serialized model's behavior. |

## Bound the research and preserve the decisions

Before any new numerical study, put its related ledger ID, specific mechanism,
expected implementation decision, acceptance rule, and finite run budget in
this record or the acceptance manifest. A repeat for numerical parity is labeled
reproduction. A repeat with only new seeds is replication. Neither is a new
architectural discovery.

Reopen a decision when new requirements, a demonstrated implementation defect,
new information in the inputs, or materially different evidence changes its
premise. Record that reason under the same ID. Routine authorized work does not
need another permission question; a later explicit user direction can change
these priorities.

At the end of an implementation step, update this record with what changed,
which requirement it addressed, the acceptance outcome, and the next concrete
action. Link existing evidence instead of producing a new narrative for an
unchanged finding. Summaries and session handoffs point here. Do not restart
from historical investigation recommendations.

Later milestones, in order: M2 handled the demonstrated memory limitation
within one model contract; M3 adds independently validated error evidence and
update acceptance. These are queued responsibilities, not concurrent
architecture searches. Numerical fitting alone cannot satisfy M3.

## Current handoff

- Incumbent: `generic-memory-v2-prototype`, the maintained default of
  `glassbox.experimental.default_model.fit`; experimental, not declared adequate
  for all intended use cases. `generic-history-v1-prototype` is retained for
  old artifacts and archived research plans through the private v1 entry point.
- Last completed implementation: M2, the causal memory contract, its frozen
  manifest, the acceptance/replay runner, and the versioned consumer recipe.
  Exactly 28 benchmark candidate fits plus one byte-identical reproduction of
  the whole run with the hardened verifier; no defect reruns. The frozen
  decision is **replace**.
- Do not repeat L-BFGS, pooled scaling, first-step-floor, checkpoint-selection,
  development-recording, context-length, or memory-width sweeps after a
  restart. Read G08–G13 first. G11 (optimization) remains open and was not
  reopened by M2; the candidate used the incumbent's Adam recipe.
- Known accepted tradeoff: hidden hysteresis worsened 7% in aggregate inside
  its cap. It is a latent accumulating state, not a delayed input; do not
  present the memory as a fix for it.
- Next delivery is M3: independently validated error evidence and update
  acceptance for the v2 recipe. Start from the existing error-calibration and
  update evidence, freeze what an acceptable update must demonstrate on
  reserved recordings before changing `update`, and keep the one consumer
  recipe, revision ownership, and old artifact behavior. Platform telemetry
  with block-held-out recordings is the evidence M3 needs; another synthetic
  family sweep is not.
