# Task-horizon supervision: incomplete experiment

The 2026-09-20 experiment was intentionally aborted before fresh confirmation.
Cascade's completed refinement improves known-cohort forecast and response errors;
Dart's completed refinement makes no change. Crazyflow was stopped after 650
unchanged, rejected updates. There is no complete research acceptance verdict,
and no refinement implementation or fitted model is promoted. The public recipe
remains `generic-memory-v4-prototype`.

The [frozen protocol](harness/task-horizon-supervision-v1.json) and
[result record](harness/task-horizon-supervision-v1-result.json) preserve the
partial evidence. The immutable implementation remains in git at
`e0cfebe88f747e90758affb4ad245c599c2448e2` and in the evidence source bundle;
only protocol and result documentation enter the main branch. The comparator
policy was prospectively clarified at `ea6ba08`, before scientific work.

## Intervention and comparison

The experiment planned one 1,000-attempt, 1.2-second recursive supervised
refinement to each saved shared-v1 configuration fit. It reuses the same source recordings, parent roles, fitted normalizations and
channel weights. The mechanics, model head,
optimizer algorithm and consumer controller are unchanged. This tests an added
long-horizon training stage with more compute; it does not isolate horizon length
from the extra optimization work.

Shared-v1 is the progress comparator. Public-v4 supplies context. Before fitting,
the protocol clarified that public-v4 numerical failures remain visible without
vetoing candidate acceptance. Candidate input-eligible predictions must be finite;
candidate/reference models must be available; every required shared-v1 comparison
must be defined. Failed reference queries cannot be dropped to improve ratios.

Fresh physical confirmation and known +14M regression evidence remain separate.
Dart response residuals, nominal contact success, evidence integrity and public
promotion also remain separate. Public promotion is not part of this iteration.

## Actual training work

| Configuration | Attempts / accepted | Selected refinement step | Initial → selected long-development loss |
| --- | ---: | ---: | ---: |
| Dart | 1,000 / 0 | 0 | 0.448724695 → 0.448724695 |
| Cascade | 1,000 / 984 | 1,000 | 0.018634415 → 0.004715377 |
| Crazyflow | 650 / 0; interrupted next attempt | No final selection | 0.008117596 at all seven saved checkpoints |

These loss values use configuration-specific training scales and weights; they
should not be compared across configurations as physical accuracy measurements.
The complete-window caches contain 1,536 training/256 development windows for
Crazyflow and Cascade, and 1,048/256 for Dart. Seven Crazyflow training parents
cannot supply complete 1.2 s windows and are excluded; source recordings and
roles remain fixed, but the represented parent set changes. Windows overlap
and are not independent recordings. Three refinements started, two completed and one was intentionally interrupted.
No initialization, ridge fit or previous training stage was repeated. Dart and
Cascade checkpoint/cache prediction and loss replays passed exactly with zero
refits or optimizer calls.

## Recorded optimizer limitations

Dart never changed its fitted weights. In the first 98 completed attempts, all
784 trial losses were finite. The best permitted scale, 1/128, gave a loss of
0.003732734 versus the unchanged current objective of 0.000186768: almost 20
times worse. The gradient remained identical and the proposal changed only at
floating-point roundoff. Its gradient/proposal dot product was negative, but none
of the permitted finite steps reduced the objective.

Crazyflow preserves 650 completed rejections, 651 returned gradients and 5,200
captured trial losses. All gradients/proposals were finite and the gradient was
identical throughout. Every full-scale trial was NaN; the other 4,550 trial losses
were finite. The best trial was still 17.93% worse than the current loss. All
completed parameter sets and all seven checkpoints match the initial model.
The first trial objective of attempt 651 was initiated, then interrupted during
host conversion; no result was captured and internal completion is unknown.
The final model/report are absent. SIGINT return code -2 and the worker's generic
`failed` outcome are preserved alongside the intentional-abort record.

Repeating essentially identical rejected proposals was wasteful. The fixed
eight-scale search did not find a usable step, and the fixed attempt count lacked
a stagnation exit. These observations identify an optimization limitation. They
do not establish exhausted model capacity, nor prove that smaller steps will
improve held-out physical response accuracy. Smaller steps have not been tried.
Dart's unchanged weights do not test a successfully optimized long-horizon model.
No model calls were added for these diagnostics; they use captured evidence.

## Cascade known regression evidence

On the saved +14M cohort, using the frozen single-simulator endpoint weighting,
forecast error is 10.92% lower and command-response error 20.42% lower than
shared-v1. Seventy of 90 endpoint comparisons improve. All 84 parent truth slots
remain available and all three model arms are finite.

Wind forecast error worsens 8.16% in aggregate. At 250 ms, wind velocity RMSE rises
0.44723 → 0.48639 m/s and rate RMSE rises 0.22194 → 0.24683 rad/s. Wind response
error improves 15.59%. Primary 250 ms response parent-p95, across 48 parents, improves from
0.03122 → 0.01698 m/s, 0.03374 → 0.02094 rad/s, and
0.00375177 → 0.00208434 for rotation entries. The rotation-entry statistic is
dimensionless and is not an angle in degrees or radians.

No Cascade retention guard against shared-v1 fails on this known cohort. This is
not a two-simulator or fresh-confirmation verdict. Prediction replay and independent
score reduction of 99,792 raw rows both passed. This does not establish that
the inherited v4 wind/tail guards pass.

## Dart nominal contact and failed proposals

The unchanged Dart fitted weights reproduce the previous failure: 81 intervals
(0.81 seconds), no plane crossing, minimum plane distance 0.345851 m, 27 completed
solves with one converged, and a terminal nonfinite initial objective/gradient on
the 28th solve. The single new trial captured all 184 objective/gradient calls.
Its native/selected-mean replay passed exactly with zero optimizer calls; the
altered-proposal challenge was rejected.

Of the 184 calls, 157 have finite objectives and gradients. All 27 failed calls
have a NaN objective and all 160 gradient coordinates NaN. Every proposed command
and causal input is finite, and all commands satisfy the controller bounds.

The first failed proposal follows two identical finite seed evaluations. It moves
commands by up to 0.89319 and places every expanded command coordinate at a bound;
all 120 command rows leave the recorded training marginals. The original seed
exceeds those marginals in 33 rows. Each of the 26 failed completed solves exhibits the same
first-moved-proposal saturation pattern. The terminal solve at step 81 fails on
its next seed, unchanged apart from float32 rounding.

The captured evidence identifies the failing proposals. It does not contain
internal rollout states for those proposals, so the exact internal source of NaNs
remains unidentified. Command-range comparisons are marginal-support diagnostics,
not calibrated coverage or proof of causation.

## Closure and next named step

Fresh +15M physical recordings and fresh Dart response branches were never
collected. Crazyflow refinement/known evaluation, the remaining Dart diagnostics,
fresh evaluations, full decision replay and the other three alteration challenges
were not completed. The archived evidence runner refuses to run after the abort
marker. This iteration is closed as incomplete and must not resume as a completed
frozen experiment.

The Crazyflow failed-stage replay checks source/inventory preservation only;
the separate prefix audit checks captured counts and unchanged saved arrays.
Neither is a completed-fit prediction/loss replay. Completed local integrity
checks do not qualify the entire protocol. Derivative fidelity, calibrated
long-horizon coverage, live updates and contact adequacy remain unqualified.

The next bounded step within **Dart task-horizon command-response fidelity** is
to diagnose and resolve rejected optimizer steps while keeping the model and
training objective fixed. First freeze and validate a bounded step search with
explicit stagnation/failure handling; then test physical accuracy on fresh
confirmation evidence. Adaptive step sizes are a hypothesis, not a validated
remedy. No successor experiment is implemented or frozen here.
