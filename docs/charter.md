# Charter: one generic dynamics learner

Glassbox learns a differentiable dynamics model of any uniformly sampled
system from recordings of its observed signals and commands, and improves that
model live as new recordings arrive. The caller supplies signals, units,
timing, and recording boundaries. Nothing else. One recipe, one module, three
calls: `fit`, `predict`, `update`. No platform families, no catalog, no
options, no branches on what the system is.

The goal is to make the structured quadrotor and fixed-wing models
unnecessary. Generality takes priority over winning every benchmark case.
The generic learner is the adopted development baseline; documented losses
on individual systems are improvement work, not an automatic veto on that
choice. [`status.md`](status.md) holds the current gap against the
definition of done below and is the only page a new session needs to read
after this one. Git holds the history; it is not carried forward as narrative.

## Definition of done

Adopting the approach and completing the project are separate decisions.
Completion requires the following, measured on held-out evidence under
protocols frozen and committed before each implementation change.

| Criterion | Done means |
| --- | --- |
| One recipe | A single versioned recipe in a single module. `fit(recordings)`, `model.predict(...)`, `model.update(recordings)`. The consumer contract has no options. |
| Accuracy | Broad competitive forecast performance on held-out recordings across systems from one platform-independent recipe. Report per-system gains and losses against structured comparators on identical rows; superiority on every corpus is not required. Task sufficiency uses a separately declared application requirement and its actual metric; historical model-error percentiles are not task requirements. |
| Capability | The synthetic suite, including delayed inputs and hidden state, passes its absolute caps. It is a fast regression guard, not a place to win. |
| Control | Cascade tracking through the plant and NMPC seam meets the declared application tracking requirement on held-out trials. Structured and oracle arms provide diagnostic comparisons; outperforming an inadequate comparator does not establish task success. |
| Live improvement | During a Cascade run the learner refits on streamed recordings within a bounded budget and swaps the active model when a predeclared held-out threshold is met. Tracking after the swap is no worse than before. |
| Evidence | Every forecast carries a measured error envelope whose held-out coverage lands in a declared band, and the controller's robustness terms consume it. |
| Lean | The structured models, the belief format, and the research scripts are deleted. What remains is the learner, the harness, the telemetry adapters, and the controller. |

## Rules

- Each iteration's measurements and promotion criteria are frozen and
  committed before fitting or new trials. Historical protocols and results
  stay immutable. An explicit change in product priorities is recorded as a
  policy decision on known evidence, never as a newly passed experiment.
- One change per iteration, addressing one named gap in the status table.
  Menus of context lengths, widths, optimizers, or seeds fail review. A new
  mechanism needs a named failure it addresses.
- Nothing is deprecated; it is deleted. Nothing is preserved for its own sake.
  Old results are not evidence for new code.
- Assumptions about structure are fine: causality, memory, smoothness,
  observation noise. Assumptions about the platform are not.
- Synthetic results never count as platform readiness. Fit quality, error
  calibration, and control adequacy are separate claims with separate
  evidence.
- Generality and broad empirical advantage can justify adoption despite
  localized losses. The adopted v3 baseline explicitly accepts its measured
  ARP deficit. Future improvements are judged against the generic baseline;
  the structured models are benchmarks, not an incumbent entitled to win.
  Report gain and loss magnitudes, breadth across systems, and task adequacy
  separately. A win count alone is not a permanent promotion rule. Choose
  any aggregate weights and unacceptable-regression limits before fitting,
  rather than requiring every metric to improve on every case.
- Every accepted iteration replays from saved artifacts without refitting, and
  an altered artifact is rejected by that replay.

## Process

Each iteration runs in its own worktree: measure the largest gap in
[`status.md`](status.md), change one thing, run the harness, keep the change
according to the iteration's predeclared promotion criteria, and write one
paragraph saying what changed and what the gap is now. Historical gate
outputs remain visible; they do not silently reimpose an all-case adoption
veto. Evidence-integrity checks remain mandatory. Review verifies by rerunning
the harness, injecting defects, and checking the consumer contract, not by
reading. On merge, the status table is updated. The loop ends when every row
meets its target.
