# Optimizer step selection

The rejected-update blocker is resolved on all three tested configurations:
Dart, Crazyflow and Cascade each accepted 1,000 updates. This did not resolve
physical response accuracy or Dart contact control. The frozen residual verdict
fails, so this refinement is not accepted and there is no public promotion.
The public recipe remains `generic-memory-v4-prototype`.

The [frozen protocol](harness/optimizer-step-selection-v1.json) changed step
selection around the existing Adam displacement: persistent Armijo backtracking,
explicit call budgets, a 20-attempt feasibility gate and stagnation handling.
The model equations, recording parents, 1.2 s recursive objective, fixed channel
weights and normalization remained unchanged. Each fit loaded its exact prior
preparation cache and original shared-v1 selected model; no initialization,
ridge solve, short-horizon fit or earlier refinement was repeated.

| Configuration | Initial weighted development loss | Selected loss | Reduction | Selected attempt |
| --- | ---: | ---: | ---: | ---: |
| Dart | 0.44872469 | 0.43222596 | 3.68% | 1,000 |
| Crazyflow | 0.00811760 | 0.00526138 | 35.19% | 900 |
| Cascade | 0.01863442 | 0.00487511 | 73.84% | 1,000 |

All three passed the feasibility gate and ended at the 1,000-gradient budget.
Their 6,023 trial-objective calls and 3,000 gradient calls are captured, with
initiated and returned counts matching. These weighted development losses are
checkpoint-selection measures, not physical errors or task success rates.
Long-horizon supervision also adds compute; the experiment does not isolate a
causal benefit from horizon length alone.

The saved first Dart gradient and full proposal match the prior rejected fit.
The first accepted scale is 1/2048, sixteen times smaller than the old search
minimum. The first 20 updates reduce training loss 15.16%, directly supporting
deeper step selection as the cause of initial progress. Across the complete
Dart refinement, training loss falls 50.74% while development loss falls 3.68%.
Most remaining development loss is after 0.6 s; velocity error barely improves.

Accuracy is compared against each configuration's saved **shared-v1** model on
matched histories, commands and native truth. Public v4 is reported as context;
its failures do not veto this comparison. Required shared-v1 comparisons must
remain defined. Fresh physical recordings use previously uncollected +15M seeds;
the previously inspected +14M cohort is a separate regression set. Neither enters
fitting or checkpoint selection. Both arms use identical eligible truth, with
failed and missing native intervals retained in the evidence.

| Physical cohort | Aggregate forecast ratio | Aggregate response ratio |
| --- | ---: | ---: |
| Fresh +15M | 0.99404058 | 0.96817283 |
| Known +14M | 0.99019834 | 0.97720757 |

| Simulator | Fresh forecast / response change | Known forecast / response change |
| --- | ---: | ---: |
| Cascade | -11.10% / -19.91% | -10.75% / -19.40% |
| Crazyflow | +11.15% / +17.04% | +9.86% / +18.48% |

Negative changes mean lower error. These use the same frozen scope weights.

Ratios below one mean lower residuals versus shared-v1, using the predeclared
hierarchical geometric weighting. The small aggregate gains conceal opposing
changes. Cascade response improves in all five scopes; its wind forecast still
worsens, by 7.72% fresh and 6.70% known. Crazyflow response worsens in every scope
on both cohorts. Its fresh primary 250 ms velocity-forecast parent-p95 rises
from **0.0133096 to 0.0258166 m/s** (1.94×); the known-cohort ratio is 1.67×.
Both physical retention verdicts fail. Historical v4 wind/tail obligations also
remain failed; a gain against shared-v1 does not erase them.

Fresh Dart evidence contains six evaluation-only parents, 12 factual forecasts
and 96 perturbed command branches. The candidate becomes nonfinite on **8/96**
branches; shared-v1 fails on nine and public v4 on one. All scored failures occur
at 1.2 s; earlier scored horizons through 0.6 s remain finite. The associated
native truth, histories and commands are finite, and each paired factual
prediction stays finite. Failed candidate/shared commands remain within the
training command coordinate ranges, which does not establish joint state/action
coverage. Failed rows remain present, leaving required response ratios undefined
and the Dart residual verdict failed.

A lower NaN count is not sufficient progress: on the branch where shared-v1
fails but the candidate remains finite, candidate terminal velocity and angular
rate reach millions in m/s and rad/s. The finite factual forecast ratio of
0.93950227 therefore does not establish useful command-response prediction.

The single nominal candidate control trial executes **0.78 s** and ends with
`nonfinite initial objective or gradient`, without crossing the contact plane.
None of 26 completed solves converges; 27 optimizer attempts and 191 gradient
calls are preserved. The controller, bounds, target and seed are unchanged.
Contact adequacy remains separate from direct residuals and training progress.

Training source `717bcf55c950c50165658276d8b4e46b5bb7eb37` and its artifacts remain
immutable. An initial collection import failure occurred before the pilot ran.
The [evaluation-only correction](harness/optimizer-step-selection-evaluation-correction-v1.json)
uses actual legacy symbols from the same pinned source to satisfy the missing
export. Corrected source `67764ce80a2d1cfd79c97321de440b3b5f9fd1a3` preserves the
learner and scoring equations and consumes exactly the three original fits and
nine original training-integrity stages. The failed collection remains visible;
no fit or completed training audit was repeated for the correction.

Evidence integrity is complete. The qualification verifies all 39 required
stage receipts, preserves the failed nominal controller outcome, and confirms
all five actual alteration classes were rejected. Checkpoint, native truth,
forecast and control replays passed without refitting or repeating optimizer
solves; independent reductions and decision replay agree with the saved verdict.
The [result record](harness/optimizer-step-selection-v1-result.json) anchors the
full evidence inventory and distinguishes training from corrected evaluation
source and binding identities. The corrected harness passed 137 affected tests;
the runner and completed-report checks passed their synthetic regression suites.

Only frozen protocols and result documentation enter the main branch. Both
immutable research implementations and the fitted artifacts remain available
through their source bundles and evidence anchors; the failed refinement is
not integrated into the public recipe.

The next named gap is **recursive command-response stability**. Saved outputs
show physical-state magnitudes growing enormously before NaNs, while finite
rotation prefixes retain near-orthogonal matrices. The unrestricted affine and
quadratic acceleration terms, their feedback through predicted state, and the
explicit integration scheme remain possible contributors; the current evidence
does not isolate causation. A prospective no-fit diagnostic should instrument
the nine failed candidate/shared branch identities and matched factual controls,
trace head contributions and state growth, and compare 1/2/4 integration substeps
with the same saved models and observation-interval history updates. Freeze that
bounded diagnostic before execution, then choose one intervention from its
results. No new fit, clipping rule, vehicle branch or consumer option is implied.
