# Proposed iteration: learn finite command responses

The named gap is command-response identification from training data. The adopted
generic model is trained on absolute trajectories, although users need accurate
changes in those trajectories when commands change. The initialized-mean prior
partially repaired Crazyflow angular generalization but worsened Cascade's
response aggregate by 11.66%; the joint residual criterion failed. Another prior
strength is not the next experiment.

Test one mechanism: add a supervised finite-response loss to the unchanged
generic predictor. Given two continuations from a shared history, directly fit
`(F(history, alternative) - F(history, factual))` to the observed difference.
The learner receives observed signals and issued commands, never simulator
hidden states or physical equations. The research collector uses cloned hidden
state only to establish valid paired truth.

The [saved-array diagnosis](../artifacts/2026-09-20/paired-response-plan-diagnosis/initialized-mean-prior-next-mechanism-diagnosis.json)
found zero exact shared-history pairs among the 1,536 cached training windows
in either simulator. This does not mean unpaired recordings cannot identify
responses; it means the proposed contrast targets are absent from this cache.
For saved primary Cascade 250 ms prediction pairs, response-error MSE divided
by four contributes only 1.77%, 3.33% and 1.75% of mean pointwise pair MSE for
velocity, rate and rotation. The remainder is common error: for pair errors
`e0,e1`, `mean(e0²,e1²) = ((e0+e1)/2)² + (e1-e0)²/4`.
These descriptive, physical-unit calculations use inspected held-out arrays;
they are not the weighted training objective or new confirmation results. They
motivate measuring the two components separately, not a claim that capacity is
adequate or a guarantee that the new loss will improve unseen responses.

Use three comparison arms: the exact retained v4 revision without refitting;
a factual-only fit on a new training branch pool; and a factual-plus-response
fit on the identical pool. The two fresh fits must share initial parameters,
normalizers, absolute-trajectory weights, examples, development selection and
optimizer work. Both fit factual and alternative absolute targets. The only
intended fitting difference is the contrast term. Report any additional objective
arithmetic, backtracking calls, forecasts and elapsed time. Gains against saved
v4 establish progress; the matched new-data control tests whether the response
loss contributes beyond the data itself.

The earlier [intervention-response experiment](harness/intervention-response-v1-result.json)
reported response/forecast ratios of 0.34899/0.68608 for paired versus absolute
losses on matched data and 6,000-step budgets. These are historical results under
a different recipe and corpus, not predicted gains. The conditional-moment
experiment's smaller response improvement failed its own criterion. Neither
experiment proves that current residuals require a larger model.

Before implementation or new collection, freeze one explicit pair roster and
strength, training-only normalization, parent-level train/development separation,
exact fitting work and outcome handling. Branches from one parent must stay in
one role. Use fresh confirmation parents after both fitted outcomes are sealed;
the inspected +12M and +13M cohorts are no longer confirmation data. Require
broad forecast/response progress against retained v4 under predeclared regression
limits, and a predeclared response improvement against the matched factual-only
control. Preserve every failed collection and missing truth. Training contrast
loss is not the success measure.

This is a research plan, not a frozen protocol or a launched experiment. Pair
counts, sampling, loss scaling and the added-control criterion still need to be
specified before freezing. Exact hidden-state resets are available in these
simulators but generally unavailable in ordinary recordings. Identical observed
histories alone do not prove identical hidden state. A positive result would
establish learnability with stronger evidence; it would still require a generic
recording-based implementation and separate public qualification. Do not add a
pair-label option, platform branch or simulator dependency to the consumer API.
