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

The engineering baseline is the existing recursive affine-plus-nonlinear
sequence model and its fixed consumer workflow. It remains experimental, but
it is the incumbent we improve. A model need not dominate every alternative on
every recorded metric to become the incumbent. Stable structured `glassbox.fit`
remains a separate supported path until an explicit, versioned integration.

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
| G04 | **Deferred to M2:** explicit history and latent memory were already compared. The tested eight-coordinate latent model did not uniformly beat observed history. | [Model structures](model-structure-experiments.md). Reopen for a verified information/memory requirement, with an encoder/transition contract and matched budgets; do not claim latent memory is untested. |
| G05 | **Established limitation:** a fixed 100 ms history cannot identify every delayed system. More fitting of identical supplied histories cannot distinguish different true futures. | [Qualification witness](model-qualification.md). Changing available information is legitimate M2 work; wider networks on the same ambiguous inputs are not a fix. |
| G06 | **Established limitation:** low logged-policy forecast loss and marginal input ranges do not prove identified input response. | [Qualification](model-qualification.md), [diagnostics](sequence-diagnostics.md). Informative variation is a data requirement; unexplained input residuals alone do not prove it. |
| G07 | **Closed inference:** older-history diagnostic gain is not proof of missing state. Nonlinear short-context features can explain the gain in a fully observed Markov system. | [History confounding](history-confounding.md). Retain alternative explanations; no automatic history-length increase from this diagnostic. |
| G08 | **Closed as a default:** pooled horizon normalization. Its cycle improvement comes with broader regressions. | [Horizon generalization](horizon-generalization.md). Reopen only when a changed acceptance contract or demonstrated mechanism makes the tradeoff relevant, not to rerun fresh-seed counting. |
| G09 | **Retained challenger, frozen for M1:** first-step loss-scale floor. It improves several near-periodic cases and preserves many objectives exactly; it also has known regressions. | [Generalization](horizon-generalization.md), [checkpoint attribution](checkpoint-attribution.md). Compare against the engineering acceptance contract when needed; do not rediscover these effects with another sweep. |
| G10 | **Closed as standalone fixes:** swapping the two tested checkpoint criteria, always taking the final checkpoint, or mandating 16 development recordings. None is a universal remedy. | [Checkpoint attribution](checkpoint-attribution.md). Broader data sometimes helps. Reopen a specific policy only with a named failure and a changed mechanism, not a guarantee inferred from recording count. |
| G11 | **M1 decision closed; incumbent retained:** the fixed-budget L-BFGS replacement fails the frozen acceptance contract. Optimization and generalization remain distinct unresolved limitations. | [M1 decision](#m1-decision-2026-09-15-utc), [checkpoint attribution](checkpoint-attribution.md). Reopen this solver design only for an identified implementation defect, a materially different optimization mechanism supported by existing traces, or an explicitly changed resource requirement. New seeds, a larger budget alone, or a different checkpoint count are not an unrecorded retry. |
| G12 | **Committed:** whole-recording provenance, separate data roles, observation-budget accounting, immutable revisions, and evidence tied to its fitted revision. | [Fixed workflow](default-recipe.md), [recording selection](recording-selection.md). Keep these when changing the fitter; do not add another wrapper hierarchy. |

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

Later milestones, in order: M2 handles demonstrated memory/observation
limitations within one model contract; M3 adds independently validated error
evidence and update acceptance. These are queued responsibilities, not concurrent
architecture searches. Numerical fitting alone cannot satisfy M2 or M3.

## Current handoff

- Incumbent: `generic-history-v1-prototype`; experimental, not declared adequate
  for all intended use cases. Historical artifacts and raw studies are retained.
- Last completed implementation: M1, its bounded internal fitter and reusable
  acceptance/replay runner. Exactly 25 benchmark candidate fits; no defect reruns.
  The frozen decision is **retain**, not an unresolved better/worse tally.
- Do not repeat L-BFGS, pooled scaling, first-step-floor, checkpoint-selection,
  or development-recording sweeps after a restart. Read G08–G11 first.
- Next delivery is the M2 causal-memory contract, using the existing G05
  indistinguishable-history witness. First specify and test how one model
  consumes consecutive observations, preserves state within a recording, and
  resets at recording boundaries. Retain the one consumer recipe, revision
  ownership, and old artifact behavior. A longer context must add actual
  information; it is not inferred from a residual-history score alone.
- Before fitting an M2 candidate, freeze its information budget, required
  delayed-system capability, and regression guards in a new manifest version.
  Reuse M1's ordinary-system guards and runner structure. Do not relabel M1's
  optimizer failure as solved by moving to memory, or start a menu of latent
  dimensions and history lengths. M2 addresses G05, not a promise to solve G11.
