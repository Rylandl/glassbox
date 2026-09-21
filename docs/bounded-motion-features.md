# Bounded motion features: completed fits, evaluation prerequisite failed

The bounded-feature implementation and all three fits are complete. Their
saved optimizer arithmetic and 33 checkpoints replay exactly, and six deliberate
training-evidence alterations are rejected. The iteration stops before physical
evaluation because Dart selects its conditioned initialization rather than an
updated-weight checkpoint. This preserves the frozen failed prerequisite; it
does not establish that bounded features fail to reduce runaway trajectories.

The [protocol](harness/bounded-motion-features-v1.json) was frozen before
implementation and fitting. The [result record](harness/bounded-motion-features-v1-result.json)
anchors the implementation, models, complete evidence inventory, replay and
alteration checks. No public model is promoted.

## Intervention

The preceding [recurrence diagnostic](recurrence-attribution.md) found early
amplification from affine state/history feedback, followed by quadratic overflow.
This experiment changes one representation: normalized body velocity and angular
rate each become `L * tanh(z / L)`, where
`L = 4 * max(1, max_training_states(abs(z)))` independently for each of six
coordinates. The bound is derived from past and future states in the exact
cached training windows. Development and evaluation data do not set it.

Current features, stored history, history differences and quadratic readouts
all consume this representation. Issued commands, response filters, gravity,
body/world transforms, integration and timing retain their existing semantics.
The six bounds are saved with the model. There is no vehicle selector, physical
parameter input or consumer tuning option.

Observed training coordinates change by at most 2.04%, and their normalized
local derivative remains at least approximately 0.940. These are feature-level
statements, not prediction-error guarantees. With bounded commands, proper
rotations and finite weights, the learned acceleration head has a finite bound;
its magnitude need not be useful, dissipative or stabilizing. Saturation can
weaken state gradients far outside training support. Numerical finiteness,
physical accuracy and controller adequacy still require direct evaluation.

## Fit results

All three fits start from the original shared-v1 weights and use the exact
cached data, objective normalization and channel weights of the preceding
[unbounded refinement](optimizer-step-selection.md). The Armijo optimizer is
unchanged. Each fit has a budget of 1,000 attempts or 1,800 optimizer seconds,
with the same early progress check and development-selected checkpoint.
No initializer, ridge solve, restart or hyperparameter sweep occurred.

These are **weighted normalized development objectives used for selection**,
not fresh physical errors. Lower is better; percentages compare with the saved
unbounded refinement's selected checkpoint for the same configuration.

| Configuration | Attempts | Selected step | Previous unbounded objective | Bounded objective | Change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dart | 1,000 | 0 | 0.432226 | 0.392097 | −9.28% |
| Crazyflow | 980 | 700 | 0.00526138 | 0.00623228 | +18.45% |
| Cascade | 1,000 | 1,000 | 0.00487511 | 0.00467388 | −4.13% |

Dart and Cascade reached the update limit in 944.7 and 487.4 optimizer seconds.
Crazyflow reached the time limit at 1,800.04 seconds. Thus the budget rule is
matched, while completed work differs by 20 attempts on Crazyflow. All three
20-attempt feasibility checks pass. The same cached windows are retained,
including the seven previously missing Crazyflow training parents; none is
silently added or removed for this comparison.

![Training and development objectives](../artifacts/2026-09-21/bounded-motion-features-v1-evidence/fit-objectives.png)

On Dart, the feature transformation alone lowers development objective from
0.448725 to 0.392097, a 12.62% reduction. The subsequent fit lowers its new
training objective from 0.0331434 to 0.000184731, a 99.44% reduction, but raises
development objective to 0.405321, 3.37% above its conditioned start. Every
updated checkpoint is worse on development data, so selection correctly
retains step 0. The conditioned starting training objective is about 177 times
the unbounded starting objective; a small feature change can substantially
alter a recurrent trajectory. The large training decrease alone is therefore
not evidence of improved generalization.

Crazyflow selects changed weights but retains a development deficit versus
the prior unbounded fit. Cascade improves development objective modestly.
No aggregate win count or development ratio is treated as a physical-accuracy
or adoption verdict.

## Evaluation boundary and next diagnostic

The frozen protocol requires Dart's selected weights to differ from its
conditioned initialization before new simulation. This prerequisite fails,
and the original runner stops with `Dart did not select updated weights` after
all three fits complete. No fresh physical collection, known/fresh response
evaluation or controller trial runs. The full evaluation qualification is
incomplete; acceptance and public promotion remain false. The completed
fits are not crashed or unfinished workers and must not be restarted.

The prerequisite came from an optimizer-only experiment, where selecting step
0 meant retaining the original predictor. That implication does not apply to
this intervention: feature conditioning already changes the predictor at step
0. The gate is therefore a poor prerequisite for measuring conditioning, but
its frozen verdict is preserved. Neither a fresh accuracy improvement nor a
fresh accuracy failure has been measured here.

The next named gap is **separating feature-conditioning benefit from refinement
damage**. Freeze a small no-fit diagnostic using all 189 previously inspected
Dart queries and their archived native truth. Compare the original shared-v1,
conditioned initialization, bounded terminal checkpoint and previous unbounded
refinement. Measure nonfinite and extreme finite trajectories, endpoint and
prefix physical errors, and signed command-response errors. All queries remain
known diagnostic evidence. This can determine whether conditioning already
addresses runaway, and whether further fitting harms that behavior, before
another expensive fit. No follow-up protocol has been frozen or run yet.

The fit-only closure verifies 2,980 captured optimizer attempts, all 33
checkpoint witnesses and selection decisions, and rejects altered step scales
and directional derivatives for each configuration. These checks replay saved
fit evidence without fitting again or evaluating new command branches. All
172 analytic/mock preflight tests pass. The immutable implementation remains
isolated at `d855535943b15d6a4970de3315363bb9f939c47f`; the complete source bundle
and 21,058-file evidence inventory are anchored in the result record. Public v4
and all prior physical/control qualification boundaries remain unchanged.
