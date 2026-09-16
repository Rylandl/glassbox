# Charter: one generic dynamics learner

Glassbox learns a differentiable dynamics model of any uniformly sampled
system from recordings of its observed signals and commands, and improves that
model live as new recordings arrive. The caller supplies signals, units,
timing, and recording boundaries. Nothing else. One recipe, one module, three
calls: `fit`, `predict`, `update`. No platform families, no catalog, no
options, no branches on what the system is.

The goal is to make the structured quadrotor and fixed-wing models
unnecessary. [`status.md`](status.md) holds the current gap against the
definition of done below and is the only page a new session needs to read
after this one. Git holds the history; it is not carried forward as narrative.

## Definition of done

All of the following hold on the same held-out evidence, measured by one
harness whose manifest is frozen and committed before each change.

| Criterion | Done means |
| --- | --- |
| One recipe | A single versioned recipe in a single module. `fit(recordings)`, `model.predict(...)`, `model.update(recordings)`. The consumer contract has no options. |
| Accuracy | On every pinned platform corpus with whole recordings held out, 250 ms forecast error matches or beats the structured model on the same rows and sits inside the task allowance from the accuracy-requirements work. |
| Capability | The synthetic suite, including delayed inputs and hidden state, passes its absolute caps. It is a fast regression guard, not a place to win. |
| Control | Cascade tracking through the existing plant and NMPC seam meets or beats the structured model on the matched trial set. |
| Live improvement | During a Cascade run the learner refits on streamed recordings within a bounded budget and swaps the active model when a predeclared held-out threshold is met. Tracking after the swap is no worse than before. |
| Evidence | Every forecast carries a measured error envelope whose held-out coverage lands in a declared band, and the controller's robustness terms consume it. |
| Lean | The structured models, the belief format, and the research scripts are deleted. What remains is the learner, the harness, the telemetry adapters, and the controller. |

## Rules

- The harness and its thresholds are frozen and committed before a candidate
  is fitted. A threshold, seed, or case changed after scores were seen fails
  review.
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
- Every accepted iteration replays from saved artifacts without refitting, and
  an altered artifact is rejected by that replay.

## Process

Each iteration runs in its own worktree: measure the largest gap in
[`status.md`](status.md), change one thing, run the harness, keep the change
only if every frozen gate passes, and write one paragraph saying what changed
and what the gap is now. Review verifies by rerunning the harness, injecting
defects, and checking the consumer contract, not by reading. On merge, the
status table is updated. The loop ends when every row meets its target.
